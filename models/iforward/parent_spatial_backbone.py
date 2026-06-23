from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.streetforward.stage6_0.event_encoder import EventPack
from models.streetforward.struct_decoders.common import normalize_params_for_embed

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
        self.zero_invalid_context = bool(zero_invalid_context)
        self.support_threshold_bg = float(support_threshold_bg)
        self.support_threshold_distant = float(support_threshold_distant)
        self.support_threshold_rigid = float(support_threshold_rigid)
        self.support_threshold_rigid_out = float(support_threshold_rigid_out)
        self.param_support_codec = Stage6ParentParamSupportCodec(output_dim=int(param_support_dim))
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
        param_support = self.param_support_codec(
            params_for_embed=x.params_for_embed,
            support=x.support,
            valid_mask=valid,
            branch_id=x.branch_id,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        return self.token_builder(
            parent_context=context,
            param_support=param_support,
            support=x.support,
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
        n = int(x.coords.shape[0])
        if int(x.split_0 + x.split_1) != n:
            raise ValueError("ParentSpatial near split mismatch")
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
        event, layouts, aux = self.near_ptv3(
            token,
            coords=x.coords,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=batch_offsets,
            layout_cache=layout_cache,
        )
        return ParentStructOutput(event=event, valid_mask=valid, support=x.support, aux=aux, layout_cache=layouts)

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
        n = int(x.coords.shape[0])
        if int(x.split_0 + x.split_1) != n:
            raise ValueError("ParentSpatial far split mismatch")
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
        return ParentStructOutput(
            event=event,
            valid_mask=valid,
            support=x.support,
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
    "Stage6ParentParamSupportCodec",
    "empty_parent_struct_input",
]
