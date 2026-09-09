from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.streetforward.math_utils import _normalize_quat, _quat_to_rotmat
from models.streetforward.stage6_0.event_encoder import EventPack
from models.streetforward.struct_decoders.common import normalize_params_for_embed

from .observation_feedback import scale_feedback
from .parent_ptv3 import ParentPTv3Encoder
from .parent_serialization import ParentSerializedLayout


@dataclass
class ParentStructInput:
    parent_context: torch.Tensor
    support: torch.Tensor
    valid: Optional[torch.Tensor]
    coords: torch.Tensor
    branch_id: torch.Tensor
    params_for_embed: Dict[str, torch.Tensor]
    split_0: int
    split_1: int
    meta: Dict[str, Any] = field(default_factory=dict)
    geometry_branch_id: Optional[torch.Tensor] = None
    geometry_alpha: float | torch.Tensor | None = None


@dataclass
class ParentStructOutput:
    event: torch.Tensor
    valid_mask: torch.Tensor
    support: torch.Tensor
    aux: Dict[str, Any] = field(default_factory=dict)
    layout_cache: Dict[str, ParentSerializedLayout] = field(default_factory=dict)


def _empty_param_dict(ref: torch.Tensor, *, sh_rest_bases: int) -> Dict[str, torch.Tensor]:
    return {
        "means": ref.new_zeros((0, 3)),
        "quats": ref.new_zeros((0, 4)),
        "scales_log": ref.new_zeros((0, 3)),
        "opacity_logit": ref.new_zeros((0, 1)),
        "sh_dc": ref.new_zeros((0, 3)),
        "sh_rest": ref.new_zeros((0, int(sh_rest_bases), 3)),
    }


def empty_parent_struct_input(
    *,
    ref: torch.Tensor,
    context_dim: int,
    sh_rest_bases: int,
    path: str,
) -> ParentStructInput:
    return ParentStructInput(
        parent_context=ref.new_zeros((0, int(context_dim))),
        support=ref.new_zeros((0,)),
        valid=ref.new_zeros((0,), dtype=torch.bool),
        coords=ref.new_zeros((0, 3)),
        branch_id=torch.zeros((0,), dtype=torch.long, device=ref.device),
        params_for_embed=_empty_param_dict(ref, sh_rest_bases=int(sh_rest_bases)),
        split_0=0,
        split_1=0,
        meta={"path": str(path)},
        geometry_branch_id=torch.zeros((0,), dtype=torch.long, device=ref.device),
        geometry_alpha=0.0,
    )


class Stage6ParentParamSupportCodec(nn.Module):
    def __init__(
        self,
        *,
        support_dim: int = 2,
        branch_embed_dim: int = 4,
        output_dim: int = 24,
        detach_params: bool = True,
        detach_support: bool = True,
        norm: str = "layernorm",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.raw_param_dim = 17
        self.support_dim = int(support_dim)
        self.branch_embed_dim = int(branch_embed_dim)
        self.output_dim = int(output_dim)
        self.detach_params = bool(detach_params)
        self.detach_support = bool(detach_support)
        self.branch_embed = nn.Embedding(2, int(branch_embed_dim))
        in_dim = self.raw_param_dim + int(support_dim) + int(branch_embed_dim)
        layers: list[nn.Module] = [nn.Linear(in_dim, int(output_dim))]
        if str(norm).lower() == "layernorm":
            layers.append(nn.LayerNorm(int(output_dim)))
        elif str(norm).lower() not in {"none", "identity"}:
            raise ValueError(f"unsupported Stage6ParentParamSupportCodec norm={norm!r}")
        if str(activation).lower() == "gelu":
            layers.append(nn.GELU())
        elif str(activation).lower() == "relu":
            layers.append(nn.ReLU())
        elif str(activation).lower() not in {"none", "identity"}:
            raise ValueError(f"unsupported Stage6ParentParamSupportCodec activation={activation!r}")
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _detach_param_dict(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in params.items()}

    def forward(
        self,
        *,
        params_for_embed: Dict[str, torch.Tensor],
        support: torch.Tensor,
        valid_mask: torch.Tensor,
        branch_id: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
    ) -> torch.Tensor:
        n = int(branch_id.numel())
        if n == 0:
            ref = support
            return ref.new_zeros((0, self.output_dim))
        branch = branch_id.reshape(-1).long()
        if int(branch.shape[0]) != n:
            raise ValueError("Stage6ParentParamSupportCodec branch row mismatch")
        if bool(((branch < 0) | (branch > 1)).any().item()):
            raise ValueError("Stage6ParentParamSupportCodec branch_id must be in {0,1}")
        params = self._detach_param_dict(params_for_embed) if self.detach_params else params_for_embed
        param_vec = normalize_params_for_embed(params, aabb_min=aabb_min, aabb_max=aabb_max)
        if int(param_vec.shape[0]) != n:
            raise ValueError(f"Stage6ParentParamSupportCodec param rows {int(param_vec.shape[0])} != {n}")
        supp = support.reshape(-1).to(device=param_vec.device, dtype=param_vec.dtype)
        valid = valid_mask.reshape(-1).to(device=param_vec.device, dtype=torch.bool)
        if int(supp.shape[0]) != n or int(valid.shape[0]) != n:
            raise ValueError("Stage6ParentParamSupportCodec support/valid row mismatch")
        supp = supp.detach() if self.detach_support else supp
        support_vec = torch.stack([torch.log1p(supp.clamp_min(0.0)), valid.to(dtype=param_vec.dtype)], dim=-1)
        if int(self.support_dim) == 1:
            support_vec = support_vec[:, :1]
        elif int(self.support_dim) != 2:
            raise ValueError(f"Stage6ParentParamSupportCodec P0 supports support_dim 1 or 2, got {self.support_dim}")
        branch_vec = self.branch_embed(branch).to(dtype=param_vec.dtype)
        x = torch.cat([param_vec, support_vec, branch_vec], dim=-1)
        out = self.net(x)
        if not torch.isfinite(out).all():
            raise RuntimeError("Stage6ParentParamSupportCodec output contains NaN/Inf")
        return out


class Stage34ParentGeometrySupportCodec(nn.Module):
    """Stage 3.4 geometry-only ParentGS parameter/support codec.

    The raw parameter vector is intentionally independent from the legacy 17D
    codec: normalized means (3), rot6d (6), normalized log-scales (3), and
    opacity (1).  SH tensors are neither required nor read, which keeps the
    Stage 3.4 ParentGS gradient contract geometry-only.
    """

    def __init__(
        self,
        *,
        support_dim: int = 2,
        branch_embed_dim: int = 4,
        output_dim: int = 24,
        detach_support: bool = True,
        norm: str = "layernorm",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.raw_param_dim = 13
        self.support_dim = int(support_dim)
        self.branch_embed_dim = int(branch_embed_dim)
        self.output_dim = int(output_dim)
        self.detach_params = False
        self.detach_support = bool(detach_support)
        self.branch_embed = nn.Embedding(2, int(branch_embed_dim))
        in_dim = self.raw_param_dim + int(support_dim) + int(branch_embed_dim)
        layers: list[nn.Module] = [nn.Linear(in_dim, int(output_dim))]
        if str(norm).lower() == "layernorm":
            layers.append(nn.LayerNorm(int(output_dim)))
        elif str(norm).lower() not in {"none", "identity"}:
            raise ValueError(f"unsupported Stage34ParentGeometrySupportCodec norm={norm!r}")
        if str(activation).lower() == "gelu":
            layers.append(nn.GELU())
        elif str(activation).lower() == "relu":
            layers.append(nn.ReLU())
        elif str(activation).lower() not in {"none", "identity"}:
            raise ValueError(f"unsupported Stage34ParentGeometrySupportCodec activation={activation!r}")
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _geometry_vector(
        params: Dict[str, torch.Tensor],
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
    ) -> torch.Tensor:
        # Deliberately index only the four geometry fields.  In particular,
        # callers may omit SH tensors entirely and still use this codec.
        means = params["means"]
        quats = params["quats"]
        scales_log = params["scales_log"]
        opacity_logit = params["opacity_logit"]
        bbx_min = aabb_min.to(device=means.device, dtype=means.dtype)
        bbx_max = aabb_max.to(device=means.device, dtype=means.dtype)
        means_norm = (means - bbx_min) / (bbx_max - bbx_min).clamp_min(1.0e-6) * 2.0 - 1.0
        rot = _quat_to_rotmat(_normalize_quat(quats))
        rot6d = rot[..., :3, :2].reshape(quats.shape[:-1] + (6,))
        scales_clamped = scales_log.clamp(-10.0, 10.0)
        scales_norm = F.layer_norm(scales_clamped, scales_clamped.shape[1:])
        opacity_norm = torch.tanh(opacity_logit)
        return torch.cat([means_norm, rot6d, scales_norm, opacity_norm], dim=-1)

    def forward(
        self,
        *,
        params_for_embed: Dict[str, torch.Tensor],
        support: torch.Tensor,
        valid_mask: torch.Tensor,
        branch_id: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
    ) -> torch.Tensor:
        n = int(branch_id.numel())
        if n == 0:
            return support.new_zeros((0, self.output_dim))
        branch = branch_id.reshape(-1).long()
        if bool(((branch < 0) | (branch > 1)).any().item()):
            raise ValueError("Stage34ParentGeometrySupportCodec branch_id must be in {0,1}")
        param_vec = self._geometry_vector(
            params_for_embed,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        if int(param_vec.shape[0]) != n or int(param_vec.shape[-1]) != self.raw_param_dim:
            raise ValueError(
                "Stage34ParentGeometrySupportCodec geometry shape mismatch: "
                f"got {tuple(param_vec.shape)}, expected ({n}, {self.raw_param_dim})"
            )
        supp = support.reshape(-1).to(device=param_vec.device, dtype=param_vec.dtype)
        valid = valid_mask.reshape(-1).to(device=param_vec.device, dtype=torch.bool)
        if int(supp.shape[0]) != n or int(valid.shape[0]) != n:
            raise ValueError("Stage34ParentGeometrySupportCodec support/valid row mismatch")
        supp = supp.detach() if self.detach_support else supp
        support_vec = torch.stack([torch.log1p(supp.clamp_min(0.0)), valid.to(dtype=param_vec.dtype)], dim=-1)
        if self.support_dim == 1:
            support_vec = support_vec[:, :1]
        elif self.support_dim != 2:
            raise ValueError(
                "Stage34ParentGeometrySupportCodec supports support_dim 1 or 2, "
                f"got {self.support_dim}"
            )
        branch_vec = self.branch_embed(branch).to(dtype=param_vec.dtype)
        out = self.net(torch.cat([param_vec, support_vec, branch_vec], dim=-1))
        if not torch.isfinite(out).all():
            raise RuntimeError("Stage34ParentGeometrySupportCodec output contains NaN/Inf")
        return out


class Stage34ParentGeometryResidualAdapter(nn.Module):
    """Forward-compatible Stage 3.4 geometry residual for Parent tokens.

    The adapter deliberately reads only means, log-scales, and opacity.  Its
    final projection is zero initialized so adding the residual preserves the
    exact Stage 3.3 Parent token distribution at weights-only initialization.
    ``geometry_branch_id`` uses the stable three-way schema bg=0,
    distant=1, rigid=2 and selects the same fixed scale bounds as the exact
    Parent projector.
    """

    GEOMETRY_BRANCH_BG = 0
    GEOMETRY_BRANCH_DISTANT = 1
    GEOMETRY_BRANCH_RIGID = 2

    def __init__(
        self,
        *,
        output_dim: int = 24,
        hidden_dim: int = 24,
        min_scale: float = 1.0e-3,
        max_scale_bg: float = 0.60,
        max_scale_distant: float = 3.0,
        max_scale_rigid: float = 0.45,
    ) -> None:
        super().__init__()
        self.raw_geometry_dim = 8
        self.raw_param_dim = self.raw_geometry_dim
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_scale = float(min_scale)
        self.max_scale_bg = float(max_scale_bg)
        self.max_scale_distant = float(max_scale_distant)
        self.max_scale_rigid = float(max_scale_rigid)
        if self.output_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("Stage34ParentGeometryResidualAdapter dimensions must be positive")
        if self.min_scale <= 0.0:
            raise ValueError("Stage34ParentGeometryResidualAdapter min_scale must be > 0")
        for name, value in (
            ("bg", self.max_scale_bg),
            ("distant", self.max_scale_distant),
            ("rigid", self.max_scale_rigid),
        ):
            if value < self.min_scale:
                raise ValueError(
                    "Stage34ParentGeometryResidualAdapter "
                    f"max_scale_{name} must be >= min_scale"
                )

        self.input_proj = nn.Linear(self.raw_geometry_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def is_zero_initialized(self) -> bool:
        """Return whether the residual output projection is still exactly zero."""

        return bool(
            int(torch.count_nonzero(self.output_proj.weight.detach()).item()) == 0
            and int(torch.count_nonzero(self.output_proj.bias.detach()).item()) == 0
        )

    def _branch_log_scale_bounds(
        self,
        geometry_branch_id: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch = geometry_branch_id.to(device=device, dtype=torch.long).reshape(-1)
        if bool(((branch < 0) | (branch > 2)).any().item()):
            raise ValueError(
                "Stage34ParentGeometryResidualAdapter geometry_branch_id must be in {0,1,2}"
            )
        min_log = math.log(self.min_scale)
        max_logs = torch.tensor(
            [
                math.log(self.max_scale_bg),
                math.log(self.max_scale_distant),
                math.log(self.max_scale_rigid),
            ],
            dtype=dtype,
            device=device,
        )
        lo = torch.full((int(branch.numel()), 1), min_log, dtype=dtype, device=device)
        hi = max_logs.index_select(0, branch).reshape(-1, 1)
        return lo, hi

    def geometry_vector(
        self,
        params_for_embed: Dict[str, torch.Tensor],
        *,
        geometry_branch_id: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
    ) -> torch.Tensor:
        # Do not index quats or SH: those fields must have no path through the
        # Stage 3.4 residual even when present in the legacy parameter dict.
        means = params_for_embed["means"]
        scales_log = params_for_embed["scales_log"]
        opacity_logit = params_for_embed["opacity_logit"]
        if means.ndim != 2 or tuple(means.shape[1:]) != (3,):
            raise ValueError("Stage34ParentGeometryResidualAdapter means must have shape [N,3]")
        n = int(means.shape[0])
        if tuple(scales_log.shape) != (n, 3):
            raise ValueError("Stage34ParentGeometryResidualAdapter scales_log must have shape [N,3]")
        if tuple(opacity_logit.shape) != (n, 1):
            raise ValueError("Stage34ParentGeometryResidualAdapter opacity_logit must have shape [N,1]")
        branch = geometry_branch_id.reshape(-1)
        if int(branch.numel()) != n:
            raise ValueError(
                "Stage34ParentGeometryResidualAdapter geometry_branch_id row mismatch"
            )

        bbx_min = aabb_min.to(device=means.device, dtype=means.dtype)
        bbx_max = aabb_max.to(device=means.device, dtype=means.dtype)
        means_norm = (means - bbx_min) / (bbx_max - bbx_min).clamp_min(1.0e-6) * 2.0 - 1.0

        lo, hi = self._branch_log_scale_bounds(
            branch,
            dtype=scales_log.dtype,
            device=scales_log.device,
        )
        clamped_log_scales = torch.maximum(torch.minimum(scales_log, hi), lo)
        log_size = clamped_log_scales.mean(dim=-1, keepdim=True)
        center = (lo + hi) * 0.5
        half_range = ((hi - lo) * 0.5).clamp_min(1.0e-6)
        log_size_norm = (log_size - center) / half_range
        log_shape_norm = (clamped_log_scales - log_size) / half_range
        opacity_norm = torch.tanh(opacity_logit)
        out = torch.cat(
            [means_norm, log_size_norm, log_shape_norm, opacity_norm],
            dim=-1,
        )
        if tuple(out.shape) != (n, self.raw_geometry_dim):
            raise RuntimeError(
                "Stage34ParentGeometryResidualAdapter internal geometry shape mismatch: "
                f"got {tuple(out.shape)}"
            )
        if not torch.isfinite(out).all():
            raise RuntimeError("Stage34ParentGeometryResidualAdapter geometry contains NaN/Inf")
        return out

    def forward(
        self,
        *,
        params_for_embed: Dict[str, torch.Tensor],
        geometry_branch_id: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        alpha: float | torch.Tensor,
    ) -> torch.Tensor:
        geometry = self.geometry_vector(
            params_for_embed,
            geometry_branch_id=geometry_branch_id,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        if int(geometry.shape[0]) == 0:
            return geometry.new_zeros((0, self.output_dim))
        geometry_used = scale_feedback(geometry, alpha)
        out = self.output_proj(self.activation(self.input_proj(geometry_used)))
        if not torch.isfinite(out).all():
            raise RuntimeError("Stage34ParentGeometryResidualAdapter output contains NaN/Inf")
        return out


class ParentTokenBuilder(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        param_support_dim: int,
        token_dim: int = 64,
        support_embed_dim: int = 4,
        branch_embed_dim: int = 4,
        frame_gap_embed_dim: int = 4,
        visit_kind_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.feat_proj = nn.Linear(int(context_dim), int(token_dim))
        self.param_support_proj = nn.Linear(int(param_support_dim), int(token_dim))
        self.support_proj = nn.Sequential(
            nn.Linear(2, int(support_embed_dim)),
            nn.GELU(),
            nn.Linear(int(support_embed_dim), int(token_dim)),
        )
        self.branch_embed = nn.Embedding(2, int(branch_embed_dim))
        self.branch_proj = nn.Linear(int(branch_embed_dim), int(token_dim))
        self.frame_gap_embed = nn.Embedding(3, int(frame_gap_embed_dim))
        self.frame_gap_proj = nn.Linear(int(frame_gap_embed_dim), int(token_dim))
        self.visit_kind_embed = nn.Embedding(3, int(visit_kind_embed_dim))
        self.visit_kind_proj = nn.Linear(int(visit_kind_embed_dim), int(token_dim))
        self.norm = nn.LayerNorm(int(token_dim))

    def forward(
        self,
        *,
        parent_context: torch.Tensor,
        param_support: torch.Tensor,
        support: torch.Tensor,
        valid_mask: torch.Tensor,
        branch_id: torch.Tensor,
        frame_gap: int | torch.Tensor = 0,
        visit_kind_id: int | torch.Tensor = 1,
    ) -> torch.Tensor:
        n = int(parent_context.shape[0])
        out = self.feat_proj(parent_context)
        out = out + self.param_support_proj(param_support.to(dtype=parent_context.dtype))
        support_vec = torch.stack(
            [
                torch.log1p(support.reshape(-1).clamp_min(0.0)),
                valid_mask.reshape(-1).to(device=parent_context.device, dtype=parent_context.dtype),
            ],
            dim=-1,
        )
        out = out + self.support_proj(support_vec.to(dtype=parent_context.dtype))
        branch = branch_id.reshape(-1).long()
        if int(branch.shape[0]) != n:
            raise ValueError("ParentTokenBuilder branch_id row mismatch")
        out = out + self.branch_proj(self.branch_embed(branch).to(dtype=parent_context.dtype))
        if torch.is_tensor(frame_gap):
            gap = frame_gap.to(device=parent_context.device, dtype=torch.long).reshape(-1)
            if int(gap.numel()) == 1:
                gap = gap.expand(n)
        else:
            gap = torch.full((n,), int(frame_gap), device=parent_context.device, dtype=torch.long)
        if torch.is_tensor(visit_kind_id):
            visit = visit_kind_id.to(device=parent_context.device, dtype=torch.long).reshape(-1)
            if int(visit.numel()) == 1:
                visit = visit.expand(n)
        else:
            visit = torch.full((n,), int(visit_kind_id), device=parent_context.device, dtype=torch.long)
        if int(gap.numel()) != n or int(visit.numel()) != n:
            raise ValueError("ParentTokenBuilder frame_gap/visit_kind row mismatch")
        gap = gap.clamp(0, 2)
        visit = visit.clamp(0, 2)
        out = out + self.frame_gap_proj(self.frame_gap_embed(gap).to(dtype=parent_context.dtype))
        out = out + self.visit_kind_proj(self.visit_kind_embed(visit).to(dtype=parent_context.dtype))
        out = self.norm(out)
        if not torch.isfinite(out).all():
            raise RuntimeError("ParentTokenBuilder output contains NaN/Inf")
        return out


class ParentSpatialBackbone(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int = 48,
        event_dim: int = 64,
        token_dim: int = 64,
        param_support_dim: int = 24,
        param_codec_mode: str = "legacy_17d",
        param_codec_detach_params: bool = True,
        param_codec_detach_support: bool = True,
        geometry_min_scale: float = 1.0e-3,
        geometry_max_scale_bg: float = 0.60,
        geometry_max_scale_distant: float = 3.0,
        geometry_max_scale_rigid: float = 0.45,
        ptv3_detach_coords: bool = False,
        support_embed_dim: int = 4,
        branch_embed_dim: int = 4,
        frame_gap_embed_dim: int = 4,
        visit_kind_embed_dim: int = 4,
        near_depth: int = 4,
        near_heads: int = 4,
        near_patch_size: int = 64,
        near_orders: tuple[str, ...] = ("z", "z_trans"),
        far_hidden_dim: int = 64,
        far_num_layers: int = 2,
        support_threshold_bg: float = 0.0,
        support_threshold_distant: float = 0.0,
        support_threshold_rigid: float = 0.0,
        support_threshold_rigid_out: float = 0.0,
        xcpe_backend: str = "fallback_neighbor_mean",
        xcpe_voxel_size: float = 0.5,
        use_xcpe: bool = True,
        zero_invalid_context: bool = True,
    ) -> None:
        super().__init__()
        if int(event_dim) != int(token_dim):
            raise ValueError("ParentSpatialBackbone P0 requires event_dim == token_dim")
        self.context_dim = int(context_dim)
        self.event_dim = int(event_dim)
        self.param_codec_mode = str(param_codec_mode).lower()
        self.param_codec_schema = "legacy_17d_v1"
        self.ptv3_detach_coords = bool(ptv3_detach_coords)
        self.detach_support_inputs = bool(param_codec_detach_support)
        self.zero_invalid_context = bool(zero_invalid_context)
        self.support_threshold_bg = float(support_threshold_bg)
        self.support_threshold_distant = float(support_threshold_distant)
        self.support_threshold_rigid = float(support_threshold_rigid)
        self.support_threshold_rigid_out = float(support_threshold_rigid_out)
        self.geometry_residual_adapter: Optional[Stage34ParentGeometryResidualAdapter] = None
        if self.param_codec_mode in {"legacy", "legacy_17d", "legacy_geometry_sh", "stage6_17d", "17d"}:
            self.param_support_codec = Stage6ParentParamSupportCodec(
                output_dim=int(param_support_dim),
                detach_params=bool(param_codec_detach_params),
                detach_support=bool(param_codec_detach_support),
            )
        elif self.param_codec_mode in {
            "geometry_only",
            "geometry_only_13d",
            "geometry_only_stage3_4",
            "stage3_4_geometry_13d",
            "13d",
        }:
            if bool(param_codec_detach_params):
                raise ValueError(
                    "Stage 3.4 geometry-only ParentGS codec requires param_codec_detach_params=false"
                )
            self.param_support_codec = Stage34ParentGeometrySupportCodec(
                output_dim=int(param_support_dim),
                detach_support=bool(param_codec_detach_support),
            )
            self.param_codec_schema = "geometry_only_13d_v1"
        elif self.param_codec_mode in {
            "legacy17d_plus_geometry8d_residual",
            "legacy_17d_plus_geometry_8d_residual",
        }:
            if bool(param_codec_detach_params):
                raise ValueError(
                    "Stage 3.4 residual ParentGS codec requires param_codec_detach_params=false"
                )
            if int(param_support_dim) != 24:
                raise ValueError(
                    "Stage 3.4 residual ParentGS codec requires param_support_dim=24"
                )
            # Preserve these legacy module names and shapes so a Stage 3.3
            # weights-only initialization loads the exact old token path.
            self.param_support_codec = Stage6ParentParamSupportCodec(
                output_dim=int(param_support_dim),
                detach_params=True,
                detach_support=bool(param_codec_detach_support),
            )
            self.geometry_residual_adapter = Stage34ParentGeometryResidualAdapter(
                output_dim=int(param_support_dim),
                hidden_dim=24,
                min_scale=float(geometry_min_scale),
                max_scale_bg=float(geometry_max_scale_bg),
                max_scale_distant=float(geometry_max_scale_distant),
                max_scale_rigid=float(geometry_max_scale_rigid),
            )
            self.param_codec_schema = "legacy17d_plus_geometry8d_residual_v1"
        else:
            raise ValueError(f"unsupported ParentSpatialBackbone param_codec_mode={param_codec_mode!r}")
        self.token_builder = ParentTokenBuilder(
            context_dim=int(context_dim),
            param_support_dim=int(param_support_dim),
            token_dim=int(token_dim),
            support_embed_dim=int(support_embed_dim),
            branch_embed_dim=int(branch_embed_dim),
            frame_gap_embed_dim=int(frame_gap_embed_dim),
            visit_kind_embed_dim=int(visit_kind_embed_dim),
        )
        self.near_ptv3 = ParentPTv3Encoder(
            dim=int(token_dim),
            depth=int(near_depth),
            num_heads=int(near_heads),
            patch_size=int(near_patch_size),
            orders=tuple(near_orders),
            use_xcpe=bool(use_xcpe),
            xcpe_backend=str(xcpe_backend),
            xcpe_voxel_size=float(xcpe_voxel_size),
        )
        layers: list[nn.Module] = []
        dim = int(token_dim)
        for _ in range(max(int(far_num_layers) - 1, 0)):
            layers.extend([nn.Linear(dim, int(far_hidden_dim)), nn.LayerNorm(int(far_hidden_dim)), nn.GELU()])
            dim = int(far_hidden_dim)
        layers.append(nn.Linear(dim, int(event_dim)))
        self.far_mlp = nn.Sequential(*layers)
        self.far_norm = nn.LayerNorm(int(event_dim))

    def _valid(self, x: ParentStructInput, *, bg_threshold: float, rigid_threshold: float) -> torch.Tensor:
        support = x.support.reshape(-1)
        valid = support > torch.where(
            x.branch_id.reshape(-1).long() == 0,
            support.new_full((int(support.numel()),), float(bg_threshold)),
            support.new_full((int(support.numel()),), float(rigid_threshold)),
        )
        if x.valid is not None:
            valid = valid & x.valid.reshape(-1).to(device=support.device, dtype=torch.bool)
        return valid

    def _tokens(
        self,
        x: ParentStructInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        valid: torch.Tensor,
        frame_gap: int | torch.Tensor = 0,
        visit_kind_id: int | torch.Tensor = 1,
    ) -> torch.Tensor:
        context = x.parent_context
        if self.zero_invalid_context:
            context = torch.where(valid[:, None], context, torch.zeros_like(context))
        support = x.support.detach() if self.detach_support_inputs else x.support
        param_support = self.param_support_codec(
            params_for_embed=x.params_for_embed,
            support=support,
            valid_mask=valid,
            branch_id=x.branch_id,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        if self.geometry_residual_adapter is not None:
            if x.geometry_branch_id is None:
                raise ValueError(
                    "Stage 3.4 residual ParentGS codec requires ParentStructInput.geometry_branch_id"
                )
            if x.geometry_alpha is None:
                raise ValueError(
                    "Stage 3.4 residual ParentGS codec requires ParentStructInput.geometry_alpha"
                )
            geometry_branch_id = x.geometry_branch_id.detach()
            residual = self.geometry_residual_adapter(
                params_for_embed=x.params_for_embed,
                geometry_branch_id=geometry_branch_id,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                alpha=x.geometry_alpha,
            )
            param_support = param_support + residual.to(dtype=param_support.dtype)
        return self.token_builder(
            parent_context=context,
            param_support=param_support,
            support=support,
            valid_mask=valid,
            branch_id=x.branch_id,
            frame_gap=frame_gap,
            visit_kind_id=visit_kind_id,
        )

    @staticmethod
    def _visit_kind_id(visit_kind: str | int | torch.Tensor) -> int | torch.Tensor:
        if torch.is_tensor(visit_kind):
            return visit_kind
        if isinstance(visit_kind, int):
            return int(visit_kind)
        mapping = {"bootstrap": 0, "causal_first": 1, "repair": 2}
        return int(mapping.get(str(visit_kind), 1))

    @staticmethod
    def _validate_input_rows(x: ParentStructInput, *, path: str) -> int:
        if not torch.is_tensor(x.coords) or x.coords.ndim != 2 or int(x.coords.shape[-1]) != 3:
            raise ValueError(f"ParentSpatial {path} coords must have shape [N,3]")
        n = int(x.coords.shape[0])
        row_tensors = {
            "parent_context": x.parent_context,
            "support": x.support,
            "branch_id": x.branch_id,
        }
        if x.geometry_branch_id is not None:
            row_tensors["geometry_branch_id"] = x.geometry_branch_id
        if x.valid is not None:
            row_tensors["valid"] = x.valid
        for name, value in row_tensors.items():
            if not torch.is_tensor(value) or value.ndim == 0 or int(value.shape[0]) != n:
                shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise ValueError(
                    f"ParentSpatial {path} {name} row mismatch: got {shape}, expected N={n}"
                )
        for name, value in x.params_for_embed.items():
            if not torch.is_tensor(value) or value.ndim == 0 or int(value.shape[0]) != n:
                shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise ValueError(
                    f"ParentSpatial {path} params_for_embed.{name} row mismatch: "
                    f"got {shape}, expected N={n}"
                )
        if int(x.split_0 + x.split_1) != n:
            raise ValueError(
                f"ParentSpatial {path} split mismatch: "
                f"split_0+split_1={int(x.split_0 + x.split_1)} N={n}"
            )
        return n

    def encode_near(
        self,
        x: ParentStructInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
        layout_cache: Optional[Dict[str, ParentSerializedLayout]] = None,
        frame_gap: int | torch.Tensor = 0,
        visit_kind: str | int | torch.Tensor = 1,
    ) -> ParentStructOutput:
        n = self._validate_input_rows(x, path="near")
        if n == 0:
            return ParentStructOutput(
                event=x.coords.new_zeros((0, self.event_dim)),
                valid_mask=x.support.new_zeros((0,), dtype=torch.bool),
                support=x.support,
            )
        valid = self._valid(x, bg_threshold=self.support_threshold_bg, rigid_threshold=self.support_threshold_rigid)
        token = self._tokens(
            x,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            valid=valid,
            frame_gap=frame_gap,
            visit_kind_id=self._visit_kind_id(visit_kind),
        )
        coords = x.coords.detach() if self.ptv3_detach_coords else x.coords
        if self.ptv3_detach_coords and (coords.requires_grad or coords.grad_fn is not None):
            raise RuntimeError("Stage 3.4 ParentPTv3 coords must be detached at the call boundary")
        event, layouts, aux = self.near_ptv3(
            token,
            coords=coords,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=batch_offsets,
            layout_cache=layout_cache,
        )
        if self.ptv3_detach_coords:
            aux = dict(aux or {})
            aux["feedback/ptv3_coords/boundary_assertion_passed"] = 1.0
        support = x.support.detach() if self.detach_support_inputs else x.support
        return ParentStructOutput(event=event, valid_mask=valid, support=support, aux=aux, layout_cache=layouts)

    def encode_far(
        self,
        x: ParentStructInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
        frame_gap: int | torch.Tensor = 0,
        visit_kind: str | int | torch.Tensor = 1,
    ) -> ParentStructOutput:
        _ = batch_offsets
        n = self._validate_input_rows(x, path="far")
        if n == 0:
            return ParentStructOutput(
                event=x.coords.new_zeros((0, self.event_dim)),
                valid_mask=x.support.new_zeros((0,), dtype=torch.bool),
                support=x.support,
            )
        valid = self._valid(
            x,
            bg_threshold=self.support_threshold_distant,
            rigid_threshold=self.support_threshold_rigid_out,
        )
        token = self._tokens(
            x,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            valid=valid,
            frame_gap=frame_gap,
            visit_kind_id=self._visit_kind_id(visit_kind),
        )
        event = self.far_norm(self.far_mlp(token))
        if not torch.isfinite(event).all():
            raise RuntimeError("ParentSpatial far event contains NaN/Inf")
        support = x.support.detach() if self.detach_support_inputs else x.support
        return ParentStructOutput(
            event=event,
            valid_mask=valid,
            support=support,
            aux={"iforward/parent_spatial/far_mlp": 1.0},
        )

    def forward(
        self,
        *,
        near_in: ParentStructInput,
        far_in: ParentStructInput,
        route: Any,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        near_batch_offsets: Optional[torch.Tensor] = None,
        far_batch_offsets: Optional[torch.Tensor] = None,
        near_layout_cache: Optional[Dict[str, ParentSerializedLayout]] = None,
        frame_gap: int | torch.Tensor = 0,
        visit_kind: str | int | torch.Tensor = 1,
    ) -> tuple[EventPack, Dict[str, ParentSerializedLayout]]:
        near = self.encode_near(
            near_in,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=near_batch_offsets,
            layout_cache=near_layout_cache,
            frame_gap=frame_gap,
            visit_kind=visit_kind,
        )
        far = self.encode_far(
            far_in,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=far_batch_offsets,
            frame_gap=frame_gap,
            visit_kind=visit_kind,
        )
        num_bg = int(near_in.split_0)
        num_rigid_in = int(near_in.split_1)
        num_distant = int(far_in.split_0)
        num_rigid_out = int(far_in.split_1)
        event_bg = near.event[:num_bg]
        support_bg = near.support[:num_bg]
        valid_bg = near.valid_mask[:num_bg]
        event_distant = far.event[:num_distant] if num_distant > 0 else None
        support_distant = far.support[:num_distant] if num_distant > 0 else None
        valid_distant = far.valid_mask[:num_distant] if num_distant > 0 else None
        total_rigid = int(getattr(route, "S", torch.zeros((0,), device=event_bg.device, dtype=torch.long)).numel())
        event_rigid = event_bg.new_zeros((total_rigid, self.event_dim)) if total_rigid > 0 else None
        support_rigid = support_bg.new_zeros((total_rigid,)) if total_rigid > 0 else None
        valid_rigid = valid_bg.new_zeros((total_rigid,), dtype=torch.bool) if total_rigid > 0 else None
        if total_rigid > 0:
            rows_in = getattr(route, "S_in").to(device=event_bg.device, dtype=torch.long)
            rows_out = getattr(route, "S_out").to(device=event_bg.device, dtype=torch.long)
            if num_rigid_in > 0:
                event_rigid.index_copy_(0, rows_in, near.event[num_bg : num_bg + num_rigid_in])
                support_rigid.index_copy_(0, rows_in, near.support[num_bg : num_bg + num_rigid_in])
                valid_rigid.index_copy_(0, rows_in, near.valid_mask[num_bg : num_bg + num_rigid_in])
            if num_rigid_out > 0:
                event_rigid.index_copy_(0, rows_out, far.event[num_distant : num_distant + num_rigid_out])
                support_rigid.index_copy_(0, rows_out, far.support[num_distant : num_distant + num_rigid_out])
                valid_rigid.index_copy_(0, rows_out, far.valid_mask[num_distant : num_distant + num_rigid_out])
        aux = {
            **dict(near.aux or {}),
            **dict(far.aux or {}),
            "iforward/parent_spatial/near_rows": float(int(near.event.shape[0])),
            "iforward/parent_spatial/far_rows": float(int(far.event.shape[0])),
            "iforward/parent_spatial/rigid_rows": float(total_rigid),
        }
        return (
            EventPack(
                event_bg=event_bg,
                event_distant=event_distant,
                event_rigid=event_rigid,
                support_bg=support_bg,
                support_distant=support_distant,
                support_rigid=support_rigid,
                valid_bg=valid_bg,
                valid_distant=valid_distant,
                valid_rigid=valid_rigid,
                obs_code_bg=None,
                obs_code_distant=None,
                obs_code_rigid=None,
                route=route,
                aux=aux,
            ),
            near.layout_cache,
        )


__all__ = [
    "ParentStructInput",
    "ParentStructOutput",
    "ParentSpatialBackbone",
    "ParentTokenBuilder",
    "Stage34ParentGeometryResidualAdapter",
    "Stage34ParentGeometrySupportCodec",
    "Stage6ParentParamSupportCodec",
    "empty_parent_struct_input",
]
