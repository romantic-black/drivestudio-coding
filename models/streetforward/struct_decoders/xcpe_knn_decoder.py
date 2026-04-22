from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.streetforward.struct_decoders.common import (
    StructDecoderInput,
    StructDecoderOutput,
    StreetForwardStructDecoderBase,
    normalize_params_for_embed,
    scatter_mean,
)
from models.streetforward.struct_decoders.knn_attention import EdgeGatedKNNAttention
from models.streetforward.struct_decoders.token_builders import StructTokenBuilder
from models.streetforward.struct_decoders.xcpe_decoder import _XCPEResidualLayer, _SPCONV_AVAILABLE, offsets_to_batch_ids


class StreetForwardXCPEKNNDecoder(StreetForwardStructDecoderBase):
    def __init__(
        self,
        *,
        feat_2d_channels: int,
        out_channels: int,
        param_dim: int,
        branch_embed_dim: int,
        support_embed_dim: int,
        param_embed_dim: int,
        channels: int,
        voxel_size: float,
        xcpe_num_layers: int,
        xcpe_kernel_size: int,
        xcpe_residual_scale_init: float,
        sparse_backend: str,
        norm: str,
        act: str,
        knn_num_layers: int,
        knn_attn_dim: int,
        knn_pos_dim: int,
        knn_pos_scale: float,
        knn_chunk_size: int,
        knn_residual_scale_init: float,
        knn_use_same_branch_flag: bool,
        knn_use_support: bool,
        knn_use_pos_value: bool,
        debug_validate: bool,
        use_2d_feat: bool,
        use_support: bool,
        use_branch_embed: bool,
        use_param_embed: bool,
        zero_invalid_2d_feat: bool,
        clamp_grid_coord: bool,
    ) -> None:
        super().__init__()
        backend = str(sparse_backend).lower()
        if backend not in {"spconv", "fallback_neighbor_mean"}:
            raise ValueError("Stage5_1 struct_decoder.sparse_backend must be 'spconv' or 'fallback_neighbor_mean'.")
        if int(knn_num_layers) <= 0:
            raise ValueError(f"knn_num_layers must be > 0, got {knn_num_layers}")
        self.voxel_size = float(voxel_size)
        if self.voxel_size <= 0.0:
            raise ValueError("struct_decoder.voxel_size must be > 0.")
        if bool(clamp_grid_coord):
            raise ValueError("Stage5_1 does not support clamp_grid_coord=true.")

        self.param_dim = int(param_dim)
        self.zero_invalid_2d_feat = bool(zero_invalid_2d_feat)
        self.backend = backend
        self.fallback_max_points = 20000
        self.out_channels = int(out_channels)
        self.debug_validate = bool(debug_validate)

        if backend == "spconv":
            if not _SPCONV_AVAILABLE:
                raise ImportError(
                    "Stage5_1 requires spconv when sparse_backend='spconv'. "
                    "Install spconv or set sparse_backend='fallback_neighbor_mean' for tiny unit tests."
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

        self.xcpe_layers = nn.ModuleList(
            [
                _XCPEResidualLayer(
                    int(channels),
                    kernel_size=int(xcpe_kernel_size),
                    use_spconv=self.use_spconv,
                    norm=norm,
                    act=act,
                    residual_scale_init=float(xcpe_residual_scale_init),
                    indice_key=f"sf_stage5_1_xcpe_{i}",
                )
                for i in range(int(xcpe_num_layers))
            ]
        )

        self.knn_layers = nn.ModuleList(
            [
                EdgeGatedKNNAttention(
                    channels=int(channels),
                    attn_dim=int(knn_attn_dim),
                    pos_dim=int(knn_pos_dim),
                    residual_scale_init=float(knn_residual_scale_init),
                    chunk_size=int(knn_chunk_size),
                    pos_scale=float(knn_pos_scale),
                    use_same_branch_flag=bool(knn_use_same_branch_flag),
                    use_support=bool(knn_use_support),
                    use_pos_value=bool(knn_use_pos_value),
                    debug_validate=self.debug_validate,
                )
                for _ in range(int(knn_num_layers))
            ]
        )

        self.struct_out_proj = nn.Identity() if int(channels) == self.out_channels else nn.Linear(int(channels), self.out_channels)

    def _build_voxel_layout(
        self,
        coords: torch.Tensor,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor],
        run_heavy_checks: bool,
    ):
        if run_heavy_checks:
            if not torch.isfinite(coords).all():
                raise RuntimeError("Stage5_1 struct coords contain NaN/Inf.")

        bbx_min = aabb_min.to(device=coords.device, dtype=coords.dtype)
        bbx_max = aabb_max.to(device=coords.device, dtype=coords.dtype)
        if run_heavy_checks:
            outside_mask = ((coords < bbx_min) | (coords > bbx_max)).any(dim=-1)
            if bool(outside_mask.any().item()):
                raise RuntimeError("Stage5_1 P_struct contains points outside segment_aabb.")

        num_points = int(coords.shape[0])
        if batch_offsets is None:
            batch_offsets = torch.tensor([num_points], device=coords.device, dtype=torch.long)
        batch_ids = offsets_to_batch_ids(batch_offsets, num_points=num_points, device=coords.device)

        spatial_shape_xyz = torch.floor((bbx_max - bbx_min) / self.voxel_size).long() + 1
        if run_heavy_checks:
            if bool((spatial_shape_xyz <= 0).any().item()):
                raise RuntimeError("Invalid Stage5_1 spatial shape from segment_aabb / voxel_size.")

        grid_coord_xyz = torch.floor((coords - bbx_min) / self.voxel_size).long()
        grid_key = torch.cat([batch_ids[:, None], grid_coord_xyz], dim=1)
        unique_key, inverse = torch.unique(grid_key, dim=0, sorted=True, return_inverse=True)

        b = unique_key[:, 0]
        x = unique_key[:, 1]
        y = unique_key[:, 2]
        z = unique_key[:, 3]
        indices_bzyx = torch.stack([b, z, y, x], dim=1).int()
        spatial_shape_zyx = torch.stack([spatial_shape_xyz[2], spatial_shape_xyz[1], spatial_shape_xyz[0]], dim=0).long()
        return unique_key, inverse, indices_bzyx, spatial_shape_zyx, spatial_shape_xyz

    @staticmethod
    def _validate_neighbors(x: StructDecoderInput, num_points: int, *, run_heavy_checks: bool) -> None:
        if x.neighbor_idx is None or x.neighbor_mask is None:
            raise RuntimeError("Stage5_1 requires neighbor_idx and neighbor_mask.")
        if x.neighbor_idx.dim() != 2:
            raise RuntimeError(f"neighbor_idx must be [N,K], got {tuple(x.neighbor_idx.shape)}")
        if x.neighbor_mask.shape != x.neighbor_idx.shape:
            raise RuntimeError("neighbor_mask shape mismatch with neighbor_idx.")
        if int(x.neighbor_idx.shape[0]) != num_points:
            raise RuntimeError(f"neighbor_idx N mismatch: got {x.neighbor_idx.shape[0]}, expected {num_points}")
        if x.neighbor_mask.dtype != torch.bool:
            raise RuntimeError("neighbor_mask must be bool.")
        if run_heavy_checks:
            if not bool(x.neighbor_mask[:, 0].all().item()):
                raise RuntimeError("neighbor_mask slot-0 must be true for all rows.")
            if bool((x.neighbor_mask.sum(dim=1) <= 0).any().item()):
                raise RuntimeError("each row must have at least one valid neighbor.")
            if bool((x.neighbor_idx < 0).any().item()) or bool((x.neighbor_idx >= num_points).any().item()):
                raise RuntimeError(f"neighbor_idx out of range [0, {num_points}).")

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
            return StructDecoderOutput(feat=x.coords.new_zeros((0, self.out_channels)))
        if int(x.split_bg + x.split_rigid_in) != num_points:
            raise ValueError("Stage5_1 struct split mismatch with total points.")
        if (not self.use_spconv) and num_points > self.fallback_max_points:
            raise RuntimeError(
                "fallback_neighbor_mean backend is for tiny unit tests only; use sparse_backend='spconv' for training-scale runs."
            )
        run_heavy_checks = (not self.training) or self.debug_validate
        self._validate_neighbors(x, num_points, run_heavy_checks=run_heavy_checks)

        unique_key, inverse, indices_bzyx, spatial_shape_zyx, spatial_shape_xyz = self._build_voxel_layout(
            x.coords,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=batch_offsets,
            run_heavy_checks=run_heavy_checks,
        )

        param_vec = normalize_params_for_embed(
            x.params_for_embed,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        if int(param_vec.shape[-1]) != int(self.param_dim):
            raise RuntimeError(f"Stage5_1 param embed dim mismatch: got {param_vec.shape[-1]}, expected {self.param_dim}.")

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

        batch_size = int(batch_offsets.numel() if batch_offsets is not None else 1)
        for layer in self.xcpe_layers:
            voxel_feat = scatter_mean(point_feat, inverse, dim_size=int(unique_key.shape[0]))
            voxel_delta = layer(
                voxel_feat=voxel_feat,
                unique_key_bxyz=unique_key,
                indices_bzyx=indices_bzyx,
                spatial_shape_zyx=spatial_shape_zyx,
                batch_size=batch_size,
                debug_check_output_order=debug_check_spconv_order,
            )
            point_feat = point_feat + layer.residual_scale.to(dtype=point_feat.dtype) * voxel_delta[inverse]

        neighbor_idx = x.neighbor_idx.long().contiguous()
        neighbor_mask = x.neighbor_mask.contiguous()
        for layer in self.knn_layers:
            point_feat = layer(
                point_feat,
                coords=x.coords,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                branch_id=x.branch_id,
                acc_w=x.acc_w,
            )

        feat_out = self.struct_out_proj(point_feat)
        xcpe_residual_scales = (
            torch.stack([layer.residual_scale for layer in self.xcpe_layers]).detach()
            if len(self.xcpe_layers) > 0
            else feat_out.new_zeros((1,))
        )
        knn_residual_scales = torch.stack([layer.residual_scale for layer in self.knn_layers]).detach()

        aux: Dict[str, Any] = {
            "num_struct_points": int(num_points),
            "num_struct_voxels": int(unique_key.shape[0]),
            "xcpe_residual_scale": xcpe_residual_scales,
            "knn_residual_scale": knn_residual_scales,
            "backend": self.backend,
            "knn_k": float(neighbor_idx.shape[1]),
            "knn_valid_neighbor_mean": float(neighbor_mask[:, 1:].sum(dim=1).float().mean().item()),
        }
        if bool(x.meta.get("debug_return_voxel_layout", False)):
            aux["voxel_layout"] = {
                "unique_key": unique_key,
                "indices_bzyx": indices_bzyx,
                "spatial_shape_zyx": spatial_shape_zyx,
                "spatial_shape_xyz": spatial_shape_xyz,
            }
        return StructDecoderOutput(feat=feat_out, aux=aux)

