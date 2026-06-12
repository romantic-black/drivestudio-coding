from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.streetforward.struct_decoders.common import (
    StructDecoderInput,
    StructDecoderOutput,
    StreetForwardStructDecoderBase,
    VoxelLayout,
    normalize_params_for_embed,
    offsets_to_batch_ids,
    scatter_mean,
)
from models.streetforward.struct_decoders.token_builders import StructTokenBuilder

try:
    import spconv.pytorch as spconv

    _SPCONV_AVAILABLE = True
except ImportError:
    spconv = None
    _SPCONV_AVAILABLE = False


def _make_norm(norm: str, channels: int) -> nn.Module:
    norm_name = str(norm).lower()
    if norm_name == "layernorm":
        return nn.LayerNorm(channels)
    raise ValueError(f"Unsupported xCPE norm: {norm!r}")


def _make_act(act: str) -> nn.Module:
    act_name = str(act).lower()
    if act_name == "gelu":
        return nn.GELU()
    if act_name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported xCPE activation: {act!r}")


class _XCPEResidualLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        use_spconv: bool,
        norm: str,
        act: str,
        residual_scale_init: float,
        indice_key: str,
    ) -> None:
        super().__init__()
        self.use_spconv = bool(use_spconv)
        if self.use_spconv:
            self.conv = spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=int(kernel_size),
                bias=True,
                indice_key=indice_key,
            )
        else:
            self.conv = nn.Linear(channels, channels, bias=True)
        self.linear = nn.Linear(channels, channels)
        self.norm = _make_norm(norm, channels)
        self.act = _make_act(act)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    @staticmethod
    def _hash_bzyx(coords_bzyx: torch.Tensor, spatial_shape_zyx: torch.Tensor) -> torch.Tensor:
        z = int(spatial_shape_zyx[0].item())
        y = int(spatial_shape_zyx[1].item())
        x = int(spatial_shape_zyx[2].item())
        b = coords_bzyx[:, 0].long()
        zz = coords_bzyx[:, 1].long()
        yy = coords_bzyx[:, 2].long()
        xx = coords_bzyx[:, 3].long()
        return (((b * z + zz) * y + yy) * x + xx).long()

    @classmethod
    def _remap_features_to_input_order(
        cls,
        *,
        features: torch.Tensor,
        out_indices_bzyx: torch.Tensor,
        input_indices_bzyx: torch.Tensor,
        spatial_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(out_indices_bzyx.shape) != tuple(input_indices_bzyx.shape):
            raise RuntimeError("spconv SubMConv3d changed active voxel count.")
        out_indices = out_indices_bzyx.to(device=input_indices_bzyx.device, dtype=input_indices_bzyx.dtype)
        if torch.equal(out_indices, input_indices_bzyx):
            return features
        input_hash = cls._hash_bzyx(input_indices_bzyx, spatial_shape_zyx)
        output_hash = cls._hash_bzyx(out_indices, spatial_shape_zyx)
        output_hash_sorted, order = torch.sort(output_hash)
        pos = torch.searchsorted(output_hash_sorted, input_hash)
        in_range = pos < int(output_hash_sorted.shape[0])
        if not bool(in_range.all().item()):
            raise RuntimeError("spconv SubMConv3d output is missing input voxels.")
        matched = output_hash_sorted[pos] == input_hash
        if not bool(matched.all().item()):
            raise RuntimeError("spconv SubMConv3d output voxel set differs from input voxel set.")
        return features.index_select(0, order.index_select(0, pos).to(device=features.device))

    def _neighbor_mean_3x3x3(
        self,
        voxel_feat: torch.Tensor,
        unique_key_bxyz: torch.Tensor,
        spatial_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        # unique_key format: [batch, x, y, z]
        coords_bzyx = torch.stack(
            [
                unique_key_bxyz[:, 0],
                unique_key_bxyz[:, 3],
                unique_key_bxyz[:, 2],
                unique_key_bxyz[:, 1],
            ],
            dim=1,
        ).long()
        hash_all = self._hash_bzyx(coords_bzyx, spatial_shape_zyx)
        hash_sorted, order = torch.sort(hash_all)
        feat_sorted = voxel_feat[order]

        z = int(spatial_shape_zyx[0].item())
        y = int(spatial_shape_zyx[1].item())
        x = int(spatial_shape_zyx[2].item())

        out = voxel_feat.new_zeros(voxel_feat.shape)
        cnt = voxel_feat.new_zeros((voxel_feat.shape[0], 1))

        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nz = coords_bzyx[:, 1] + dz
                    ny = coords_bzyx[:, 2] + dy
                    nx = coords_bzyx[:, 3] + dx
                    valid = (nz >= 0) & (nz < z) & (ny >= 0) & (ny < y) & (nx >= 0) & (nx < x)
                    if not bool(valid.any().item()):
                        continue
                    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                    nb = coords_bzyx[valid_idx, 0]
                    nzv = nz[valid_idx]
                    nyv = ny[valid_idx]
                    nxv = nx[valid_idx]
                    neighbor_hash = (((nb * z + nzv) * y + nyv) * x + nxv).long()
                    pos = torch.searchsorted(hash_sorted, neighbor_hash)
                    in_range = pos < hash_sorted.shape[0]
                    if not bool(in_range.any().item()):
                        continue
                    pos = pos[in_range]
                    dst_idx = valid_idx[in_range]
                    matched = hash_sorted[pos] == neighbor_hash[in_range]
                    if not bool(matched.any().item()):
                        continue
                    src_feat = feat_sorted[pos[matched]]
                    dst = dst_idx[matched]
                    out.index_add_(0, dst, src_feat)
                    cnt.index_add_(0, dst, src_feat.new_ones((src_feat.shape[0], 1)))
        return out / cnt.clamp(min=1.0)

    def forward(
        self,
        *,
        voxel_feat: torch.Tensor,
        unique_key_bxyz: torch.Tensor,
        indices_bzyx: torch.Tensor,
        spatial_shape_zyx: torch.Tensor,
        batch_size: int,
        debug_check_output_order: bool = False,
    ) -> torch.Tensor:
        if self.use_spconv:
            sp_tensor = spconv.SparseConvTensor(
                features=voxel_feat,
                indices=indices_bzyx.int(),
                spatial_shape=[int(v) for v in spatial_shape_zyx.tolist()],
                batch_size=int(batch_size),
            )
            conv_sparse = self.conv(sp_tensor)
            conv_out = self._remap_features_to_input_order(
                features=conv_sparse.features,
                out_indices_bzyx=conv_sparse.indices,
                input_indices_bzyx=indices_bzyx,
                spatial_shape_zyx=spatial_shape_zyx,
            )
        else:
            conv_in = self._neighbor_mean_3x3x3(voxel_feat, unique_key_bxyz, spatial_shape_zyx)
            conv_out = self.conv(conv_in)

        out = self.linear(conv_out)
        out = self.norm(out)
        out = self.act(out)
        return out


class StreetForwardXCPEDecoder(StreetForwardStructDecoderBase):
    def __init__(
        self,
        *,
        feat_2d_channels: int,
        out_channels: int,
        param_dim: int = 17,
        branch_embed_dim: int = 8,
        support_embed_dim: int = 8,
        param_embed_dim: int = 32,
        channels: int = 64,
        voxel_size: float = 0.20,
        num_layers: int = 2,
        kernel_size: int = 3,
        residual_scale_init: float = 1e-3,
        sparse_backend: str = "spconv",
        norm: str = "layernorm",
        act: str = "gelu",
        use_2d_feat: bool = True,
        use_support: bool = True,
        use_branch_embed: bool = True,
        use_param_embed: bool = True,
        zero_invalid_2d_feat: bool = True,
        clamp_grid_coord: bool = False,
    ) -> None:
        super().__init__()
        backend = str(sparse_backend).lower()
        if backend not in {"spconv", "fallback_neighbor_mean"}:
            raise ValueError(
                "Stage5_0 struct_decoder.sparse_backend must be 'spconv' or "
                "'fallback_neighbor_mean'."
            )
        self.voxel_size = float(voxel_size)
        if self.voxel_size <= 0.0:
            raise ValueError("struct_decoder.voxel_size must be > 0.")
        if bool(clamp_grid_coord):
            raise ValueError("Stage5_0 does not support clamp_grid_coord=true.")
        self.param_dim = int(param_dim)
        self.zero_invalid_2d_feat = bool(zero_invalid_2d_feat)
        self.backend = backend
        self.fallback_max_points = 20000

        if backend == "spconv":
            if not _SPCONV_AVAILABLE:
                raise ImportError(
                    "Stage5_0 requires spconv when sparse_backend='spconv'. "
                    "Install spconv or explicitly set sparse_backend='fallback_neighbor_mean' "
                    "for tiny unit tests."
                )
            self.use_spconv = True
        else:
            self.use_spconv = False

        self.token_builder = StructTokenBuilder(
            feat_2d_channels=int(feat_2d_channels),
            param_dim=int(param_dim),
            channels=int(channels),
            param_embed_dim=int(param_embed_dim),
            branch_embed_dim=int(branch_embed_dim),
            support_embed_dim=int(support_embed_dim),
            use_2d_feat=bool(use_2d_feat),
            use_support=bool(use_support),
            use_branch_embed=bool(use_branch_embed),
            use_param_embed=bool(use_param_embed),
        )

        self.layers = nn.ModuleList(
            [
                _XCPEResidualLayer(
                    int(channels),
                    kernel_size=int(kernel_size),
                    use_spconv=self.use_spconv,
                    norm=norm,
                    act=act,
                    residual_scale_init=float(residual_scale_init),
                    indice_key=f"sf_stage5_xcpe_{i}",
                )
                for i in range(int(num_layers))
            ]
        )

        self.struct_out_proj = (
            nn.Identity()
            if int(channels) == int(out_channels)
            else nn.Linear(int(channels), int(out_channels))
        )

    def _build_voxel_layout(
        self,
        coords: torch.Tensor,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor],
    ) -> VoxelLayout:
        if not torch.isfinite(coords).all():
            raise RuntimeError("Stage5_0 struct coords contain NaN/Inf.")

        bbx_min = aabb_min.to(device=coords.device, dtype=coords.dtype)
        bbx_max = aabb_max.to(device=coords.device, dtype=coords.dtype)

        outside_mask = ((coords < bbx_min) | (coords > bbx_max)).any(dim=-1)
        if bool(outside_mask.any().item()):
            raise RuntimeError("Stage5_0 P_struct contains points outside segment_aabb.")

        num_points = int(coords.shape[0])
        if batch_offsets is None:
            batch_offsets = torch.tensor([num_points], device=coords.device, dtype=torch.long)
        batch_ids = offsets_to_batch_ids(batch_offsets, num_points=num_points, device=coords.device)

        spatial_shape_xyz = torch.floor((bbx_max - bbx_min) / self.voxel_size).long() + 1
        if bool((spatial_shape_xyz <= 0).any().item()):
            raise RuntimeError("Invalid Stage5_0 spatial shape from segment_aabb / voxel_size.")

        grid_coord_xyz = torch.floor((coords - bbx_min) / self.voxel_size).long()

        grid_key = torch.cat([batch_ids[:, None], grid_coord_xyz], dim=1)
        unique_key, inverse = torch.unique(grid_key, dim=0, sorted=True, return_inverse=True)

        b = unique_key[:, 0]
        x = unique_key[:, 1]
        y = unique_key[:, 2]
        z = unique_key[:, 3]
        indices_bzyx = torch.stack([b, z, y, x], dim=1).int()
        spatial_shape_zyx = torch.stack(
            [
                spatial_shape_xyz[2],
                spatial_shape_xyz[1],
                spatial_shape_xyz[0],
            ],
            dim=0,
        ).long()

        return VoxelLayout(
            grid_coord_xyz=grid_coord_xyz,
            batch_ids=batch_ids,
            unique_key=unique_key,
            inverse=inverse,
            indices_bzyx=indices_bzyx,
            spatial_shape_zyx=spatial_shape_zyx,
            spatial_shape_xyz=spatial_shape_xyz,
        )

    def forward(
        self,
        x: StructDecoderInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> StructDecoderOutput:
        num_points = int(x.coords.shape[0])
        if num_points <= 0:
            return StructDecoderOutput(feat=x.coords.new_zeros((0, self.struct_out_proj.out_features if isinstance(self.struct_out_proj, nn.Linear) else x.feat_2d.shape[-1])))
        if int(x.split_bg + x.split_rigid_in) != num_points:
            raise ValueError("Stage5_0 struct split mismatch with total points.")
        if (not self.use_spconv) and num_points > self.fallback_max_points:
            raise RuntimeError(
                "fallback_neighbor_mean backend is for tiny unit tests only; "
                "use sparse_backend='spconv' for training-scale runs."
            )

        layout = self._build_voxel_layout(
            x.coords,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=batch_offsets,
        )

        param_vec = normalize_params_for_embed(
            x.params_for_embed,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        if int(param_vec.shape[-1]) != int(self.param_dim):
            raise RuntimeError(
                f"Stage5_0 param embed dim mismatch: got {param_vec.shape[-1]}, expected {self.param_dim}."
            )

        support_thr_bg = float(x.meta.get("support_threshold_bg", 0.0))
        support_thr_rigid = float(x.meta.get("support_threshold_rigid", support_thr_bg))
        support_thr = torch.where(
            x.branch_id.long() == 0,
            x.acc_w.new_full((num_points,), support_thr_bg),
            x.acc_w.new_full((num_points,), support_thr_rigid),
        )
        valid = x.acc_w > support_thr

        feat_2d = x.feat_2d
        if self.zero_invalid_2d_feat:
            feat_2d = feat_2d * valid.to(dtype=feat_2d.dtype).unsqueeze(-1)
        debug_check_spconv_order = bool(x.meta.get("debug_check_spconv_order", False))

        point_feat = self.token_builder(
            feat_2d=feat_2d,
            acc_w=x.acc_w,
            branch_id=x.branch_id,
            param_vec=param_vec,
            valid_mask=valid.to(dtype=feat_2d.dtype),
        )

        batch_size = int(
            (batch_offsets.numel() if batch_offsets is not None else 1)
        )
        for layer in self.layers:
            voxel_feat = scatter_mean(point_feat, layout.inverse, dim_size=int(layout.unique_key.shape[0]))
            voxel_delta = layer(
                voxel_feat=voxel_feat,
                unique_key_bxyz=layout.unique_key,
                indices_bzyx=layout.indices_bzyx,
                spatial_shape_zyx=layout.spatial_shape_zyx,
                batch_size=batch_size,
                debug_check_output_order=debug_check_spconv_order,
            )
            point_feat = point_feat + layer.residual_scale.to(dtype=point_feat.dtype) * voxel_delta[layout.inverse]

        feat_out = self.struct_out_proj(point_feat)
        residual_scales = (
            torch.stack([layer.residual_scale for layer in self.layers]).detach()
            if len(self.layers) > 0
            else feat_out.new_zeros((1,))
        )
        aux: Dict[str, Any] = {
            "num_struct_points": int(num_points),
            "num_struct_voxels": int(layout.unique_key.shape[0]),
            "xcpe_residual_scale": residual_scales,
            "backend": self.backend,
        }
        if bool(x.meta.get("debug_return_voxel_layout", False)):
            aux["voxel_layout"] = {
                "unique_key": layout.unique_key,
                "indices_bzyx": layout.indices_bzyx,
                "spatial_shape_zyx": layout.spatial_shape_zyx,
                "spatial_shape_xyz": layout.spatial_shape_xyz,
            }

        return StructDecoderOutput(feat=feat_out, aux=aux)
