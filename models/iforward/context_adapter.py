from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.streetforward.stage6_0 import ContextPack, EventPack

from .memory import IForwardMemoryStepContext
from .point_mamba_memory import IForwardPointMemoryPack


class IForwardContextAdapter(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 48,
        point_ctx_dim: int = 16,
        local_ctx_dim: int = 48,
        output_dim: int = 48,
        obs_code_dim: int = 2,
        branch_embed_dim: int = 4,
        repeat_embed_dim: int = 4,
        output_scale_init: float = 1.0,
        output_scale_learnable: bool = False,
    ) -> None:
        super().__init__()
        _ = event_dim
        self.point_ctx_dim = int(point_ctx_dim)
        self.local_ctx_dim = int(local_ctx_dim)
        self.output_dim = int(output_dim)
        self.obs_code_dim = int(obs_code_dim)
        self.branch_embed_dim = int(branch_embed_dim)
        self.repeat_embed_dim = int(repeat_embed_dim)
        self.branch_embed = nn.Embedding(3, int(branch_embed_dim))
        self.repeat_embed = nn.Sequential(nn.Linear(3, int(repeat_embed_dim)), nn.GELU())
        raw_dim = int(point_ctx_dim) + int(local_ctx_dim) + int(obs_code_dim) + 1 + 1 + int(branch_embed_dim) + int(repeat_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(raw_dim, int(output_dim)),
            nn.GELU(),
            nn.LayerNorm(int(output_dim)),
            nn.Linear(int(output_dim), int(output_dim)),
        )
        scale = torch.tensor(float(output_scale_init), dtype=torch.float32)
        if bool(output_scale_learnable):
            self.output_scale = nn.Parameter(scale)
        else:
            self.register_buffer("output_scale", scale)

    @staticmethod
    def _coerce_feature(
        value: Optional[torch.Tensor],
        *,
        n: int,
        dim: int,
        ref: torch.Tensor,
        default: float = 0.0,
    ) -> torch.Tensor:
        if value is None:
            return ref.new_full((int(n), int(dim)), float(default))
        x = value.to(device=ref.device, dtype=ref.dtype)
        if x.dim() == 1:
            x = x[:, None]
        if int(x.shape[0]) != int(n):
            return ref.new_full((int(n), int(dim)), float(default))
        if int(x.shape[1]) == int(dim):
            return x
        if int(x.shape[1]) > int(dim):
            return x[:, : int(dim)]
        return torch.cat([x, ref.new_full((int(n), int(dim) - int(x.shape[1])), float(default))], dim=-1)

    def _raw(
        self,
        *,
        point_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        obs_code: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        branch_id: int,
        step: IForwardMemoryStepContext,
    ) -> torch.Tensor:
        n = int(point_ctx.shape[0])
        if int(local_ctx.shape[0]) != n:
            raise ValueError("IForward ContextAdapter point/local row mismatch")
        ref = local_ctx
        obs = self._coerce_feature(obs_code, n=n, dim=self.obs_code_dim, ref=ref, default=0.0)
        support_x = self._coerce_feature(support, n=n, dim=1, ref=ref, default=0.0).clamp_min(0.0)
        valid_x = self._coerce_feature(valid, n=n, dim=1, ref=ref, default=1.0).clamp(0.0, 1.0)
        branch = self.branch_embed(torch.full((n,), int(branch_id), device=ref.device, dtype=torch.long)).to(dtype=ref.dtype)
        pos = ref.new_tensor(
            [float(step.repeat_pos_code), float(step.frame_pos_code), float(step.rollout_pos_code)]
        ).reshape(1, 3).expand(n, -1)
        repeat = self.repeat_embed(pos).to(dtype=ref.dtype)
        return torch.cat(
            [
                self._coerce_feature(point_ctx, n=n, dim=self.point_ctx_dim, ref=ref, default=0.0),
                self._coerce_feature(local_ctx, n=n, dim=self.local_ctx_dim, ref=ref, default=0.0),
                obs,
                support_x,
                valid_x,
                branch,
                repeat,
            ],
            dim=-1,
        )

    def _adapt_branch(
        self,
        *,
        point_ctx: Optional[torch.Tensor],
        local_ctx: Optional[torch.Tensor],
        obs_code: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        branch_id: int,
        step: IForwardMemoryStepContext,
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        if local_ctx is None and point_ctx is None:
            return None, {}
        ref = local_ctx if local_ctx is not None else point_ctx
        if ref is None:
            return None, {}
        n = int(ref.shape[0])
        point = point_ctx if point_ctx is not None else ref.new_zeros((n, self.point_ctx_dim))
        local = local_ctx if local_ctx is not None else ref.new_zeros((n, self.local_ctx_dim))
        raw = self._raw(
            point_ctx=point,
            local_ctx=local,
            obs_code=obs_code,
            support=support,
            valid=valid,
            branch_id=int(branch_id),
            step=step,
        )
        out = self.net(raw) * self.output_scale.to(device=raw.device, dtype=raw.dtype)
        if not torch.isfinite(out).all():
            raise RuntimeError("IForward ContextAdapter output contains NaN/Inf")
        prefix = {0: "bg", 1: "distant", 2: "rigid"}.get(int(branch_id), "branch")
        return out, {
            f"ctx_adapter/{prefix}_output_norm": float(out.detach().norm(dim=-1).mean().item()) if out.numel() else 0.0,
            f"ctx_adapter/{prefix}_point_ctx_contribution_norm": float(point.detach().norm(dim=-1).mean().item()) if point.numel() else 0.0,
            f"ctx_adapter/{prefix}_local_ctx_contribution_norm": float(local.detach().norm(dim=-1).mean().item()) if local.numel() else 0.0,
        }

    @staticmethod
    def _shuffle_rows(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or int(x.shape[0]) <= 1:
            return x
        return x[torch.randperm(int(x.shape[0]), device=x.device)]

    def forward(
        self,
        *,
        event: EventPack,
        point_ctx: IForwardPointMemoryPack,
        local_ctx: ContextPack,
        step_context: IForwardMemoryStepContext,
        ablation: str = "full",
    ) -> ContextPack:
        if str(ablation) == "no_memory":
            ctx_bg = event.event_bg.new_zeros((int(event.event_bg.shape[0]), self.output_dim))
            ctx_distant = None if event.event_distant is None else event.event_distant.new_zeros((int(event.event_distant.shape[0]), self.output_dim))
            ctx_rigid = None if event.event_rigid is None else event.event_rigid.new_zeros((int(event.event_rigid.shape[0]), self.output_dim))
            return ContextPack(ctx_bg=ctx_bg, ctx_distant=ctx_distant, ctx_rigid=ctx_rigid, aux={"ctx_adapter/disabled": 1.0})

        aux: Dict[str, float] = {}
        ctx_bg, bg_aux = self._adapt_branch(
            point_ctx=point_ctx.ctx_bg,
            local_ctx=local_ctx.ctx_bg,
            obs_code=getattr(event, "obs_code_bg", None),
            support=getattr(event, "support_bg", None),
            valid=getattr(event, "valid_bg", None),
            branch_id=0,
            step=step_context,
        )
        if ctx_bg is None:
            raise RuntimeError("IForward ContextAdapter requires bg context.")
        aux.update(bg_aux)
        ctx_distant, dist_aux = self._adapt_branch(
            point_ctx=point_ctx.ctx_distant,
            local_ctx=local_ctx.ctx_distant,
            obs_code=getattr(event, "obs_code_distant", None),
            support=getattr(event, "support_distant", None),
            valid=getattr(event, "valid_distant", None),
            branch_id=1,
            step=step_context,
        )
        aux.update(dist_aux)
        ctx_rigid, rigid_aux = self._adapt_branch(
            point_ctx=point_ctx.ctx_rigid,
            local_ctx=local_ctx.ctx_rigid,
            obs_code=getattr(event, "obs_code_rigid", None),
            support=getattr(event, "support_rigid", None),
            valid=getattr(event, "valid_rigid", None),
            branch_id=2,
            step=step_context,
        )
        aux.update(rigid_aux)
        if str(ablation) == "shuffle_context":
            ctx_bg = self._shuffle_rows(ctx_bg)
            ctx_distant = self._shuffle_rows(ctx_distant)
            ctx_rigid = self._shuffle_rows(ctx_rigid)
        event_norm = event.event_bg.detach().norm(dim=-1).mean() if event.event_bg.numel() else ctx_bg.new_tensor(0.0)
        output_norm = ctx_bg.detach().norm(dim=-1).mean() if ctx_bg.numel() else ctx_bg.new_tensor(0.0)
        aux["ctx_adapter/output_norm"] = float(output_norm.item())
        aux["ctx_adapter/output_event_ratio"] = float((output_norm / event_norm.clamp_min(1.0e-6)).item())
        return ContextPack(ctx_bg=ctx_bg, ctx_distant=ctx_distant, ctx_rigid=ctx_rigid, aux=aux)

