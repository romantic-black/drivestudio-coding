from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.streetforward.struct_decoders.common import (
    StructDecoderInput,
    VoxelLayout,
    normalize_params_for_embed,
    scatter_mean,
)
from models.streetforward.struct_decoders.voxel_layout_utils import build_voxel_layout
from models.streetforward.struct_decoders.xcpe_decoder import _SPCONV_AVAILABLE, _XCPEResidualLayer
from models.streetforward.stage6_0.event_encoder import EventPack


logger = logging.getLogger(__name__)


def _all_finite_chunked(x: torch.Tensor, *, max_elements: int = 1_000_000) -> bool:
    if int(x.numel()) == 0:
        return True
    flat = x.reshape(-1)
    for chunk in flat.split(int(max_elements)):
        if not bool(torch.isfinite(chunk).all().item()):
            return False
    return True


def _mem_debug_enabled() -> bool:
    return str(os.environ.get("STAGE6_MEM_DEBUG", "")).lower() in {"1", "true", "yes", "on"}


def _mem_debug(label: str, **extra: Any) -> None:
    if not _mem_debug_enabled() or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = float(torch.cuda.memory_allocated() / (1024.0 ** 3))
    reserved = float(torch.cuda.memory_reserved() / (1024.0 ** 3))
    peak = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
    extras = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.info("STAGE6_MEM %s alloc_gb=%.3f reserved_gb=%.3f peak_gb=%.3f %s", label, alloc, reserved, peak, extras)


@dataclass
class Stage6StructInput:
    feat_2d: torch.Tensor
    acc_w: torch.Tensor
    obs_code: torch.Tensor
    coords: torch.Tensor
    branch_id: torch.Tensor
    params_for_embed: Dict[str, torch.Tensor]
    split_0: int
    split_1: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage6StructEventOutput:
    event: torch.Tensor
    valid_mask: torch.Tensor
    support: torch.Tensor
    obs_code: torch.Tensor
    aux: Dict[str, Any] = field(default_factory=dict)


def _empty_param_dict(ref: torch.Tensor, *, sh_rest_bases: int) -> Dict[str, torch.Tensor]:
    return {
        "means": ref.new_zeros((0, 3)),
        "quats": ref.new_zeros((0, 4)),
        "scales_log": ref.new_zeros((0, 3)),
        "opacity_logit": ref.new_zeros((0, 1)),
        "sh_dc": ref.new_zeros((0, 3)),
        "sh_rest": ref.new_zeros((0, int(sh_rest_bases), 3)),
    }


def empty_stage6_struct_input(
    *,
    ref: torch.Tensor,
    feat_2d_dim: int,
    sh_rest_bases: int,
    path: str,
) -> Stage6StructInput:
    return Stage6StructInput(
        feat_2d=ref.new_zeros((0, int(feat_2d_dim))),
        acc_w=ref.new_zeros((0,)),
        obs_code=ref.new_zeros((0, 2)),
        coords=ref.new_zeros((0, 3)),
        branch_id=torch.zeros((0,), dtype=torch.long, device=ref.device),
        params_for_embed=_empty_param_dict(ref, sh_rest_bases=int(sh_rest_bases)),
        split_0=0,
        split_1=0,
        meta={"path": str(path)},
    )


def stage6_to_struct_decoder_input(x: Stage6StructInput) -> StructDecoderInput:
    return StructDecoderInput(
        feat_2d=x.feat_2d,
        acc_w=x.acc_w,
        coords=x.coords,
        branch_id=x.branch_id,
        params_for_embed=x.params_for_embed,
        split_bg=int(x.split_0),
        split_rigid_in=int(x.split_1),
        meta=dict(x.meta),
    )


class Stage6ParamObsCodec(nn.Module):
    def __init__(
        self,
        *,
        obs_code_dim: int = 2,
        support_dim: int = 2,
        branch_embed_dim: int = 4,
        output_dim: int = 24,
        detach_params: bool = True,
        detach_obs_code: bool = True,
        detach_acc_w: bool = True,
        norm: str = "layernorm",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.raw_param_dim = 17
        self.obs_code_dim = int(obs_code_dim)
        self.support_dim = int(support_dim)
        self.branch_embed_dim = int(branch_embed_dim)
        self.output_dim = int(output_dim)
        self.detach_params = bool(detach_params)
        self.detach_obs_code = bool(detach_obs_code)
        self.detach_acc_w = bool(detach_acc_w)
        self.branch_embed = nn.Embedding(2, int(branch_embed_dim))

        in_dim = self.raw_param_dim + int(obs_code_dim) + int(support_dim) + int(branch_embed_dim)
        layers: list[nn.Module] = [nn.Linear(in_dim, int(output_dim))]
        if str(norm).lower() == "layernorm":
            layers.append(nn.LayerNorm(int(output_dim)))
        elif str(norm).lower() not in {"none", "identity"}:
            raise ValueError(f"unsupported Stage6ParamObsCodec norm={norm!r}")
        if str(activation).lower() == "gelu":
            layers.append(nn.GELU())
        elif str(activation).lower() == "relu":
            layers.append(nn.ReLU())
        elif str(activation).lower() not in {"none", "identity"}:
            raise ValueError(f"unsupported Stage6ParamObsCodec activation={activation!r}")
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _detach_param_dict(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in params.items()}

    def forward(
        self,
        *,
        params_for_embed: Dict[str, torch.Tensor],
        obs_code: torch.Tensor,
        acc_w: torch.Tensor,
        branch_id: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = int(obs_code.shape[0])
        if n == 0:
            return obs_code.new_zeros((0, self.output_dim))
        if obs_code.dim() != 2 or int(obs_code.shape[1]) != int(self.obs_code_dim):
            raise ValueError(
                f"Stage6ParamObsCodec obs_code must be [N,{self.obs_code_dim}], got {tuple(obs_code.shape)}"
            )
        acc_flat = acc_w.reshape(-1)
        if int(acc_flat.shape[0]) != n:
            raise ValueError(f"Stage6ParamObsCodec acc_w row mismatch: {int(acc_flat.shape[0])} vs {n}")
        branch_flat = branch_id.reshape(-1).long()
        if int(branch_flat.shape[0]) != n:
            raise ValueError(f"Stage6ParamObsCodec branch_id row mismatch: {int(branch_flat.shape[0])} vs {n}")
        if bool(((branch_flat < 0) | (branch_flat > 1)).any().item()):
            raise ValueError("Stage6ParamObsCodec branch_id must be in {0,1}")

        params = self._detach_param_dict(params_for_embed) if self.detach_params else params_for_embed
        param_vec = normalize_params_for_embed(params, aabb_min=aabb_min, aabb_max=aabb_max)
        if int(param_vec.shape[0]) != n:
            raise ValueError(f"Stage6ParamObsCodec param row mismatch: {int(param_vec.shape[0])} vs {n}")
        obs = obs_code.detach() if self.detach_obs_code else obs_code
        acc = acc_flat.detach() if self.detach_acc_w else acc_flat
        if valid_mask is None:
            valid = acc > 0.0
        else:
            valid = valid_mask.reshape(-1).to(device=acc.device, dtype=torch.bool)
            if int(valid.shape[0]) != n:
                raise ValueError(f"Stage6ParamObsCodec valid_mask row mismatch: {int(valid.shape[0])} vs {n}")
        support = torch.stack([torch.log1p(acc.clamp_min(0.0)), valid.to(dtype=acc.dtype)], dim=-1)
        if int(self.support_dim) == 1:
            support = support[:, :1]
        elif int(self.support_dim) != 2:
            raise ValueError(f"Stage6ParamObsCodec P0 supports support_dim 1 or 2, got {self.support_dim}")
        branch = self.branch_embed(branch_flat).to(dtype=obs.dtype)
        x = torch.cat([param_vec.to(dtype=obs.dtype), obs, support.to(dtype=obs.dtype), branch], dim=-1)
        if not torch.isfinite(x).all():
            raise RuntimeError("Stage6ParamObsCodec input contains NaN/Inf")
        out = self.net(x)
        if not torch.isfinite(out).all():
            raise RuntimeError("Stage6ParamObsCodec output contains NaN/Inf")
        return out


class Stage6StructTokenBuilder(nn.Module):
    def __init__(
        self,
        *,
        feat_2d_dim: int,
        param_obs_dim: int,
        token_dim: int = 48,
        support_embed_dim: int = 4,
        branch_embed_dim: int = 4,
        use_2d_feat: bool = True,
        use_support: bool = True,
        use_branch_embed: bool = True,
        use_param_obs_embed: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.use_2d_feat = bool(use_2d_feat)
        self.use_support = bool(use_support)
        self.use_branch_embed = bool(use_branch_embed)
        self.use_param_obs_embed = bool(use_param_obs_embed)
        self.feat_proj = nn.Linear(int(feat_2d_dim), self.token_dim) if self.use_2d_feat else None
        self.param_obs_proj = nn.Linear(int(param_obs_dim), self.token_dim) if self.use_param_obs_embed else None
        self.support_proj = (
            nn.Sequential(
                nn.Linear(2, int(support_embed_dim)),
                nn.GELU(),
                nn.Linear(int(support_embed_dim), self.token_dim),
            )
            if self.use_support
            else None
        )
        if self.use_branch_embed:
            self.branch_embed = nn.Embedding(2, int(branch_embed_dim))
            self.branch_proj = nn.Linear(int(branch_embed_dim), self.token_dim)
        else:
            self.branch_embed = None
            self.branch_proj = None
        self.norm = nn.LayerNorm(self.token_dim)

    def forward(
        self,
        *,
        feat_2d: torch.Tensor,
        param_obs: torch.Tensor,
        acc_w: torch.Tensor,
        branch_id: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if feat_2d.dim() != 2:
            raise ValueError(f"feat_2d must be [N,C], got {tuple(feat_2d.shape)}")
        n = int(feat_2d.shape[0])
        out = feat_2d.new_zeros((n, self.token_dim))
        if self.feat_proj is not None:
            out = out + self.feat_proj(feat_2d)
        if self.param_obs_proj is not None:
            if param_obs.dim() != 2 or int(param_obs.shape[0]) != n:
                raise ValueError("param_obs must be [N,C]")
            out = out + self.param_obs_proj(param_obs.to(dtype=feat_2d.dtype))
        acc = acc_w.reshape(-1)
        valid = valid_mask.reshape(-1).to(dtype=feat_2d.dtype, device=feat_2d.device)
        if int(acc.shape[0]) != n or int(valid.shape[0]) != n:
            raise ValueError("support row mismatch in Stage6StructTokenBuilder")
        if self.support_proj is not None:
            support = torch.stack([torch.log1p(acc.clamp_min(0.0)), valid], dim=-1)
            out = out + self.support_proj(support.to(dtype=feat_2d.dtype))
        if self.branch_embed is not None and self.branch_proj is not None:
            branch = branch_id.reshape(-1).long()
            if int(branch.shape[0]) != n:
                raise ValueError("branch_id row mismatch in Stage6StructTokenBuilder")
            if bool(((branch < 0) | (branch > 1)).any().item()):
                raise ValueError("Stage6StructTokenBuilder branch_id must be in {0,1}")
            out = out + self.branch_proj(self.branch_embed(branch).to(dtype=feat_2d.dtype))
        out = self.norm(out)
        if not torch.isfinite(out).all():
            raise RuntimeError("Stage6StructTokenBuilder output contains NaN/Inf")
        return out


class Stage6NearXcpeEventDecoder(nn.Module):
    def __init__(
        self,
        *,
        feat_2d_dim: int,
        event_dim: int,
        token_dim: int = 48,
        param_obs_codec: Stage6ParamObsCodec,
        support_embed_dim: int = 4,
        branch_embed_dim: int = 4,
        num_blocks: int = 2,
        kernel_size: int = 3,
        voxel_size: float = 0.25,
        residual_scale_init: float = 5.0e-3,
        sparse_backend: str = "spconv",
        zero_invalid_2d_feat: bool = True,
    ) -> None:
        super().__init__()
        if int(event_dim) != int(token_dim):
            raise ValueError(
                f"Stage6NearXcpeEventDecoder requires event_dim==token_dim, "
                f"got event_dim={int(event_dim)} token_dim={int(token_dim)}"
            )
        self.event_dim = int(event_dim)
        self.token_dim = int(token_dim)
        backend = str(sparse_backend).lower()
        if backend not in {"spconv", "fallback_neighbor_mean"}:
            raise ValueError("Stage6 near sparse_backend must be 'spconv' or 'fallback_neighbor_mean'.")
        if backend == "spconv" and not _SPCONV_AVAILABLE:
            raise ImportError("Stage6 Phase A requires spconv for near xCPE when sparse_backend='spconv'.")
        self.backend = backend
        self.use_spconv = backend == "spconv"
        self.voxel_size = float(voxel_size)
        self.zero_invalid_2d_feat = bool(zero_invalid_2d_feat)
        self.param_obs_codec = param_obs_codec
        self.token_builder = Stage6StructTokenBuilder(
            feat_2d_dim=int(feat_2d_dim),
            param_obs_dim=int(param_obs_codec.output_dim),
            token_dim=int(token_dim),
            support_embed_dim=int(support_embed_dim),
            branch_embed_dim=int(branch_embed_dim),
        )
        self.layers = nn.ModuleList(
            [
                _XCPEResidualLayer(
                    int(token_dim),
                    kernel_size=int(kernel_size),
                    use_spconv=self.use_spconv,
                    norm="layernorm",
                    act="gelu",
                    residual_scale_init=float(residual_scale_init),
                    indice_key=f"sf_stage6_xcpe_{i}",
                )
                for i in range(int(num_blocks))
            ]
        )
        self.event_norm = nn.LayerNorm(int(token_dim))
        self.fallback_max_points = 20000

    def _build_voxel_layout(
        self,
        coords: torch.Tensor,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor],
    ) -> VoxelLayout:
        try:
            return build_voxel_layout(
                coords,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                voxel_size=float(self.voxel_size),
                batch_offsets=batch_offsets,
                strict_inside=True,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "outside segment_aabb" in msg:
                raise RuntimeError("Stage6 near xCPE received points outside segment_aabb") from exc
            if "spatial shape" in msg:
                raise RuntimeError("Invalid Stage6 near xCPE spatial shape") from exc
            raise

    def forward(
        self,
        x: Stage6StructInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> Stage6StructEventOutput:
        n = int(x.coords.shape[0])
        _mem_debug("near/begin", n=n)
        if int(x.split_0 + x.split_1) != n:
            raise ValueError("Stage6 near split mismatch with total points")
        if n == 0:
            event = x.coords.new_zeros((0, self.event_dim))
            return Stage6StructEventOutput(event=event, valid_mask=x.acc_w.new_zeros((0,), dtype=torch.bool), support=x.acc_w, obs_code=x.obs_code)
        if (not self.use_spconv) and n > self.fallback_max_points:
            raise RuntimeError("Stage6 fallback_neighbor_mean near xCPE is for tiny tests only; use spconv for training")
        support_thr_bg = float(x.meta.get("support_threshold_bg", x.meta.get("support_threshold_0", 0.0)))
        support_thr_rigid = float(x.meta.get("support_threshold_rigid", x.meta.get("support_threshold_1", support_thr_bg)))
        support_thr = torch.where(
            x.branch_id.long() == 0,
            x.acc_w.new_full((n,), support_thr_bg),
            x.acc_w.new_full((n,), support_thr_rigid),
        )
        valid = x.acc_w.reshape(-1) > support_thr
        feat_2d = x.feat_2d
        if self.zero_invalid_2d_feat:
            feat_2d = feat_2d * valid.to(dtype=feat_2d.dtype).unsqueeze(-1)
        _mem_debug("near/after_valid_feat", n=n, valid=float(valid.float().mean().item()) if n > 0 else 0.0)
        param_obs = self.param_obs_codec(
            params_for_embed=x.params_for_embed,
            obs_code=x.obs_code,
            acc_w=x.acc_w,
            valid_mask=valid,
            branch_id=x.branch_id,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        _mem_debug("near/after_param_obs", n=n)
        point_feat = self.token_builder(
            feat_2d=feat_2d,
            param_obs=param_obs,
            acc_w=x.acc_w,
            branch_id=x.branch_id,
            valid_mask=valid,
        )
        _mem_debug("near/after_token", n=n)
        layout = self._build_voxel_layout(x.coords, aabb_min=aabb_min, aabb_max=aabb_max, batch_offsets=batch_offsets)
        _mem_debug("near/after_layout", n=n, voxels=int(layout.unique_key.shape[0]))
        batch_size = int(batch_offsets.numel() if batch_offsets is not None else 1)
        for layer_idx, layer in enumerate(self.layers):
            voxel_feat = scatter_mean(point_feat, layout.inverse, dim_size=int(layout.unique_key.shape[0]))
            _mem_debug("near/after_scatter", layer=layer_idx, voxels=int(layout.unique_key.shape[0]))
            voxel_delta = layer(
                voxel_feat=voxel_feat,
                unique_key_bxyz=layout.unique_key,
                indices_bzyx=layout.indices_bzyx,
                spatial_shape_zyx=layout.spatial_shape_zyx,
                batch_size=batch_size,
                debug_check_output_order=bool(x.meta.get("debug_check_spconv_order", False)),
            )
            _mem_debug("near/after_layer", layer=layer_idx)
            point_delta = voxel_delta[layout.inverse]
            point_feat = point_feat + layer.residual_scale.to(dtype=point_feat.dtype) * point_delta
            del voxel_feat, voxel_delta, point_delta
            _mem_debug("near/after_point_update", layer=layer_idx)
        event = self.event_norm(point_feat)
        _mem_debug("near/after_event_norm", n=n)
        if not _all_finite_chunked(event):
            raise RuntimeError("Stage6 near xCPE event contains NaN/Inf")
        return Stage6StructEventOutput(
            event=event,
            valid_mask=valid,
            support=x.acc_w,
            obs_code=x.obs_code,
            aux={
                "stage6/struct/xcpe_num_voxels": float(layout.unique_key.shape[0]),
                "stage6/struct/xcpe_active_voxel_ratio": float(layout.unique_key.shape[0] / max(n, 1)),
                "stage6/struct/xcpe_backend": self.backend,
            },
        )


class Stage6FarMLPEventDecoder(nn.Module):
    def __init__(
        self,
        *,
        feat_2d_dim: int,
        event_dim: int,
        token_dim: int = 48,
        param_obs_codec: Stage6ParamObsCodec,
        support_embed_dim: int = 4,
        branch_embed_dim: int = 4,
        hidden_dim: int = 48,
        num_layers: int = 2,
        zero_invalid_2d_feat: bool = True,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("Stage6 far MLP num_layers must be >= 1")
        if int(event_dim) != int(token_dim):
            raise ValueError(
                f"Stage6FarMLPEventDecoder requires event_dim==token_dim, "
                f"got event_dim={int(event_dim)} token_dim={int(token_dim)}"
            )
        self.event_dim = int(event_dim)
        self.param_obs_codec = param_obs_codec
        self.zero_invalid_2d_feat = bool(zero_invalid_2d_feat)
        self.token_builder = Stage6StructTokenBuilder(
            feat_2d_dim=int(feat_2d_dim),
            param_obs_dim=int(param_obs_codec.output_dim),
            token_dim=int(token_dim),
            support_embed_dim=int(support_embed_dim),
            branch_embed_dim=int(branch_embed_dim),
        )
        layers: list[nn.Module] = []
        in_dim = int(token_dim)
        for _ in range(int(num_layers) - 1):
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(event_dim)))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: Stage6StructInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> Stage6StructEventOutput:
        _ = batch_offsets
        n = int(x.coords.shape[0])
        _mem_debug("far/begin", n=n)
        if int(x.split_0 + x.split_1) != n:
            raise ValueError("Stage6 far split mismatch with total points")
        if n == 0:
            event_dim = int(self.net[-1].out_features) if isinstance(self.net[-1], nn.Linear) else 0
            return Stage6StructEventOutput(
                event=x.coords.new_zeros((0, event_dim)),
                valid_mask=x.acc_w.new_zeros((0,), dtype=torch.bool),
                support=x.acc_w,
                obs_code=x.obs_code,
                aux={"stage6/struct/far_decoder": 1.0},
            )
        support_thr_distant = float(x.meta.get("support_threshold_distant", x.meta.get("support_threshold_0", 0.0)))
        support_thr_rigid = float(x.meta.get("support_threshold_rigid_out", x.meta.get("support_threshold_1", support_thr_distant)))
        support_thr = torch.where(
            x.branch_id.long() == 0,
            x.acc_w.new_full((n,), support_thr_distant),
            x.acc_w.new_full((n,), support_thr_rigid),
        )
        valid = x.acc_w.reshape(-1) > support_thr
        feat_2d = x.feat_2d
        if self.zero_invalid_2d_feat:
            feat_2d = feat_2d * valid.to(dtype=feat_2d.dtype).unsqueeze(-1)
        _mem_debug("far/after_valid_feat", n=n, valid=float(valid.float().mean().item()) if n > 0 else 0.0)
        param_obs = self.param_obs_codec(
            params_for_embed=x.params_for_embed,
            obs_code=x.obs_code,
            acc_w=x.acc_w,
            valid_mask=valid,
            branch_id=x.branch_id,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        _mem_debug("far/after_param_obs", n=n)
        token = self.token_builder(
            feat_2d=feat_2d,
            param_obs=param_obs,
            acc_w=x.acc_w,
            branch_id=x.branch_id,
            valid_mask=valid,
        )
        _mem_debug("far/after_token", n=n)
        event = self.net(token)
        _mem_debug("far/after_event", n=n)
        if not _all_finite_chunked(event):
            raise RuntimeError("Stage6 far MLP event contains NaN/Inf")
        return Stage6StructEventOutput(
            event=event,
            valid_mask=valid,
            support=x.acc_w,
            obs_code=x.obs_code,
            aux={"stage6/struct/far_decoder": 1.0},
        )


class Stage6RoutedStructEventDecoder(nn.Module):
    def __init__(
        self,
        *,
        feat_2d_dim: int,
        event_dim: int = 48,
        token_dim: int = 48,
        param_obs_dim: int = 24,
        support_embed_dim: int = 4,
        branch_embed_dim: int = 4,
        near_num_blocks: int = 2,
        near_kernel_size: int = 3,
        near_voxel_size: float = 0.25,
        near_residual_scale_init: float = 5.0e-3,
        near_sparse_backend: str = "spconv",
        far_hidden_dim: int = 48,
        far_num_layers: int = 2,
        param_obs_codec_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if int(event_dim) != int(token_dim):
            raise ValueError(
                f"Stage6RoutedStructEventDecoder requires event_dim==token_dim, "
                f"got event_dim={int(event_dim)} token_dim={int(token_dim)}"
            )
        codec_cfg = dict(param_obs_codec_cfg or {})
        codec_cfg.setdefault("output_dim", int(param_obs_dim))
        self.param_obs_codec = Stage6ParamObsCodec(**codec_cfg)
        self.event_dim = int(event_dim)
        self.token_dim = int(token_dim)
        self.feat_2d_dim = int(feat_2d_dim)
        self.near = Stage6NearXcpeEventDecoder(
            feat_2d_dim=int(feat_2d_dim),
            event_dim=int(event_dim),
            token_dim=int(token_dim),
            param_obs_codec=self.param_obs_codec,
            support_embed_dim=int(support_embed_dim),
            branch_embed_dim=int(branch_embed_dim),
            num_blocks=int(near_num_blocks),
            kernel_size=int(near_kernel_size),
            voxel_size=float(near_voxel_size),
            residual_scale_init=float(near_residual_scale_init),
            sparse_backend=str(near_sparse_backend),
        )
        self.far = Stage6FarMLPEventDecoder(
            feat_2d_dim=int(feat_2d_dim),
            event_dim=int(event_dim),
            token_dim=int(token_dim),
            param_obs_codec=self.param_obs_codec,
            support_embed_dim=int(support_embed_dim),
            branch_embed_dim=int(branch_embed_dim),
            hidden_dim=int(far_hidden_dim),
            num_layers=int(far_num_layers),
        )

    @staticmethod
    def _valid_ratio(mask: torch.Tensor) -> float:
        return float(mask.detach().float().mean().item()) if mask.numel() > 0 else 0.0

    def forward(
        self,
        *,
        near_in: Stage6StructInput,
        far_in: Stage6StructInput,
        route: Any,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        near_batch_offsets: Optional[torch.Tensor] = None,
        far_batch_offsets: Optional[torch.Tensor] = None,
    ) -> EventPack:
        near_out = self.near(
            near_in,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=near_batch_offsets,
        )
        far_out = self.far(
            far_in,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=far_batch_offsets,
        )
        n_bg = int(near_in.split_0)
        n_rigid_in = int(near_in.split_1)
        n_distant = int(far_in.split_0)
        n_rigid_out = int(far_in.split_1)
        if int(near_out.event.shape[0]) != n_bg + n_rigid_in:
            raise RuntimeError("Stage6 near output row mismatch")
        if int(far_out.event.shape[0]) != n_distant + n_rigid_out:
            raise RuntimeError("Stage6 far output row mismatch")
        event_bg = near_out.event[:n_bg]
        event_rigid_in = near_out.event[n_bg : n_bg + n_rigid_in] if n_rigid_in > 0 else None
        event_distant = far_out.event[:n_distant] if n_distant > 0 else None
        event_rigid_out = far_out.event[n_distant : n_distant + n_rigid_out] if n_rigid_out > 0 else None

        num_rigid_s = int(route.S.numel()) if route is not None and hasattr(route, "S") else 0
        event_rigid_s = None
        support_rigid_s = None
        valid_rigid_s = None
        obs_rigid_s = None
        if num_rigid_s > 0:
            inside_mask = route.inside_mask_S.to(device=event_bg.device, dtype=torch.bool)
            if int(inside_mask.numel()) != int(num_rigid_s):
                raise RuntimeError("inside_mask_S length mismatch with route.S.")
            if int(inside_mask.sum().item()) != int(n_rigid_in):
                raise RuntimeError("inside_mask_S true count mismatch with n_rigid_in.")
            if int((~inside_mask).sum().item()) != int(n_rigid_out):
                raise RuntimeError("inside_mask_S false count mismatch with n_rigid_out.")
            event_rigid_s = event_bg.new_zeros((num_rigid_s, self.event_dim))
            support_rigid_s = near_in.acc_w.new_zeros((num_rigid_s,))
            valid_rigid_s = torch.zeros((num_rigid_s,), dtype=torch.bool, device=event_bg.device)
            obs_rigid_s = near_in.obs_code.new_zeros((num_rigid_s, 2))
            if n_rigid_in > 0 and event_rigid_in is not None:
                event_rigid_s[inside_mask] = event_rigid_in
                support_rigid_s[inside_mask] = near_out.support[n_bg : n_bg + n_rigid_in]
                valid_rigid_s[inside_mask] = near_out.valid_mask[n_bg : n_bg + n_rigid_in]
                obs_rigid_s[inside_mask] = near_out.obs_code[n_bg : n_bg + n_rigid_in]
            if n_rigid_out > 0 and event_rigid_out is not None:
                event_rigid_s[~inside_mask] = event_rigid_out
                support_rigid_s[~inside_mask] = far_out.support[n_distant : n_distant + n_rigid_out]
                valid_rigid_s[~inside_mask] = far_out.valid_mask[n_distant : n_distant + n_rigid_out]
                obs_rigid_s[~inside_mask] = far_out.obs_code[n_distant : n_distant + n_rigid_out]

        aux = {
            **near_out.aux,
            **far_out.aux,
            "stage6/struct/near_num_bg": float(n_bg),
            "stage6/struct/near_num_rigid_in": float(n_rigid_in),
            "stage6/struct/far_num_distant": float(n_distant),
            "stage6/struct/far_num_rigid_out": float(n_rigid_out),
            "stage6/struct/near_valid_ratio_bg": self._valid_ratio(near_out.valid_mask[:n_bg]),
            "stage6/struct/near_valid_ratio_rigid_in": self._valid_ratio(near_out.valid_mask[n_bg : n_bg + n_rigid_in]),
            "stage6/struct/far_valid_ratio_distant": self._valid_ratio(far_out.valid_mask[:n_distant]),
            "stage6/struct/far_valid_ratio_rigid_out": self._valid_ratio(far_out.valid_mask[n_distant : n_distant + n_rigid_out]),
            "stage6/struct/event_bg_norm": float(event_bg.detach().norm(dim=-1).mean().item()) if event_bg.numel() else 0.0,
            "stage6/struct/event_distant_norm": (
                float(event_distant.detach().norm(dim=-1).mean().item()) if event_distant is not None and event_distant.numel() else 0.0
            ),
            "stage6/struct/event_rigid_S_norm": (
                float(event_rigid_s.detach().norm(dim=-1).mean().item()) if event_rigid_s is not None and event_rigid_s.numel() else 0.0
            ),
        }
        return EventPack(
            event_bg=event_bg,
            event_distant=event_distant,
            event_rigid=event_rigid_s,
            support_bg=near_out.support[:n_bg],
            support_distant=far_out.support[:n_distant] if n_distant > 0 else None,
            support_rigid=support_rigid_s,
            valid_bg=near_out.valid_mask[:n_bg],
            valid_distant=far_out.valid_mask[:n_distant] if n_distant > 0 else None,
            valid_rigid=valid_rigid_s,
            obs_code_bg=near_out.obs_code[:n_bg],
            obs_code_distant=far_out.obs_code[:n_distant] if n_distant > 0 else None,
            obs_code_rigid=obs_rigid_s,
            route=route,
            aux=aux,
        )


__all__ = [
    "Stage6FarMLPEventDecoder",
    "Stage6NearXcpeEventDecoder",
    "Stage6ParamObsCodec",
    "Stage6RoutedStructEventDecoder",
    "Stage6StructEventOutput",
    "Stage6StructInput",
    "empty_stage6_struct_input",
    "stage6_to_struct_decoder_input",
]
