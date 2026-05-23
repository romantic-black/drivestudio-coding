from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.streetforward.math_utils import _num_sh_bases
from models.streetforward.stage6_0.event_encoder import EventPack


@dataclass
class ContextPack:
    ctx_bg: torch.Tensor
    ctx_distant: Optional[torch.Tensor] = None
    ctx_rigid: Optional[torch.Tensor] = None
    aux: Dict[str, Any] | None = None


@dataclass
class BranchDelta:
    means: torch.Tensor
    scales_log: torch.Tensor
    quat_axis_angle: torch.Tensor
    opacity_logit: torch.Tensor
    sh: torch.Tensor
    hidden: torch.Tensor
    confidence: torch.Tensor
    noop: torch.Tensor


@dataclass
class DeltaPack:
    bg: BranchDelta
    distant: Optional[BranchDelta] = None
    rigid: Optional[BranchDelta] = None
    aux: Dict[str, Any] | None = None


class CurrentContextAdapter(nn.Module):
    def __init__(self, *, event_dim: int = 96, ctx_dim: int = 96, hidden_dim: int = 128) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.ctx_dim = int(ctx_dim)
        self.net = nn.Sequential(
            nn.Linear(int(event_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(ctx_dim)),
        )
        if int(event_dim) == int(ctx_dim):
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def _adapt(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.numel() == 0:
            return x.new_zeros((int(x.shape[0]), self.ctx_dim))
        y = self.net(x)
        if int(x.shape[-1]) == int(self.ctx_dim):
            y = x + 0.01 * y
        if not torch.isfinite(y).all():
            raise RuntimeError("CurrentContextAdapter output contains NaN/Inf")
        return y

    def forward(self, event: EventPack) -> ContextPack:
        ctx_bg = self._adapt(event.event_bg)
        if ctx_bg is None:
            raise RuntimeError("EventPack.event_bg is required")
        return ContextPack(
            ctx_bg=ctx_bg,
            ctx_distant=self._adapt(event.event_distant),
            ctx_rigid=self._adapt(event.event_rigid),
            aux={
                "ctx_bg_norm": float(ctx_bg.detach().norm(dim=-1).mean().item()) if ctx_bg.numel() else 0.0,
            },
        )


class Stage6PosteriorUpdater(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 48,
        ctx_dim: int = 48,
        hidden_dim: int = 96,
        stage_hidden_dim: int = 48,
        sh_degree: int = 1,
        means_max_step_m: float = 0.25,
        scales_log_max_step: float = 0.08,
        quat_axis_angle_max_step_rad: float = 0.08,
        opacity_logit_max_step: float = 0.25,
        sh_max_step: float = 0.10,
        hidden_max_step: float = 1.0,
        accept_vsm_ctx: bool = True,
        vsm_ctx_dim: int = 48,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.ctx_dim = int(ctx_dim)
        self.stage_hidden_dim = int(stage_hidden_dim)
        self.sh_dim = int(_num_sh_bases(int(sh_degree)) * 3)
        self.means_max_step_m = float(means_max_step_m)
        self.scales_log_max_step = float(scales_log_max_step)
        self.quat_axis_angle_max_step_rad = float(quat_axis_angle_max_step_rad)
        self.opacity_logit_max_step = float(opacity_logit_max_step)
        self.sh_max_step = float(sh_max_step)
        self.hidden_max_step = float(hidden_max_step)
        in_dim = int(event_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.head_means = nn.Linear(int(hidden_dim), 3)
        self.head_scales = nn.Linear(int(hidden_dim), 3)
        self.head_quat = nn.Linear(int(hidden_dim), 3)
        self.head_opacity = nn.Linear(int(hidden_dim), 1)
        self.head_sh = nn.Linear(int(hidden_dim), self.sh_dim)
        self.head_hidden = nn.Linear(int(hidden_dim), int(stage_hidden_dim))
        self.head_confidence = nn.Linear(int(hidden_dim), 1)
        self.head_noop = nn.Linear(int(hidden_dim), 1)
        self.accept_vsm_ctx = bool(accept_vsm_ctx)
        self.vsm_ctx_adapter: Optional[nn.Linear]
        if self.accept_vsm_ctx:
            self.vsm_ctx_adapter = nn.Linear(int(vsm_ctx_dim), int(event_dim))
            nn.init.zeros_(self.vsm_ctx_adapter.weight)
            nn.init.zeros_(self.vsm_ctx_adapter.bias)
        else:
            self.vsm_ctx_adapter = None

    def _branch_forward(
        self,
        *,
        event: Optional[torch.Tensor],
        ctx_current: Optional[torch.Tensor],
        ctx_vsm: Optional[torch.Tensor],
    ) -> Optional[BranchDelta]:
        if event is None:
            return None
        if event.dim() != 2 or int(event.shape[-1]) != int(self.event_dim):
            raise ValueError(f"event must be [N,{self.event_dim}], got {tuple(event.shape)}")
        if ctx_current is not None:
            if int(ctx_current.shape[-1]) != int(self.event_dim):
                raise ValueError(
                    f"ctx_current dim mismatch: got {int(ctx_current.shape[-1])}, expected {int(self.event_dim)}"
                )
            if int(ctx_current.shape[0]) != int(event.shape[0]):
                raise ValueError(
                    f"event/ctx_current row mismatch: {int(event.shape[0])} vs {int(ctx_current.shape[0])}"
                )
        if event.numel() == 0:
            n = int(event.shape[0])
            z3 = event.new_zeros((n, 3))
            z1 = event.new_zeros((n, 1))
            return BranchDelta(
                means=z3,
                scales_log=z3,
                quat_axis_angle=z3,
                opacity_logit=z1,
                sh=event.new_zeros((n, self.sh_dim)),
                hidden=event.new_zeros((n, self.stage_hidden_dim)),
                confidence=z1,
                noop=z1,
            )
        ctx_in = event if ctx_current is None else ctx_current
        if ctx_vsm is not None:
            if self.vsm_ctx_adapter is None:
                raise ValueError("ctx_vsm was provided but vsm ctx adapter is disabled")
            ctx_in = ctx_in + self.vsm_ctx_adapter(ctx_vsm)
        h = self.trunk(ctx_in)
        noop = torch.sigmoid(self.head_noop(h))
        gate = 1.0 - noop
        delta = BranchDelta(
            means=gate * self.means_max_step_m * torch.tanh(self.head_means(h)),
            scales_log=gate * self.scales_log_max_step * torch.tanh(self.head_scales(h)),
            quat_axis_angle=gate * self.quat_axis_angle_max_step_rad * torch.tanh(self.head_quat(h)),
            opacity_logit=gate * self.opacity_logit_max_step * torch.tanh(self.head_opacity(h)),
            sh=gate * self.sh_max_step * torch.tanh(self.head_sh(h)),
            hidden=gate * self.hidden_max_step * torch.tanh(self.head_hidden(h)),
            confidence=torch.sigmoid(self.head_confidence(h)),
            noop=noop,
        )
        for name, value in delta.__dict__.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Stage6PosteriorUpdater delta {name} contains NaN/Inf")
        return delta

    def forward(
        self,
        *,
        event: EventPack,
        ctx_current: Optional[ContextPack] = None,
        ctx_vsm: Optional[ContextPack] = None,
    ) -> tuple[DeltaPack, Dict[str, Any]]:
        bg = self._branch_forward(
            event=event.event_bg,
            ctx_current=None if ctx_current is None else ctx_current.ctx_bg,
            ctx_vsm=None if ctx_vsm is None else ctx_vsm.ctx_bg,
        )
        if bg is None:
            raise RuntimeError("EventPack.event_bg is required")
        distant = self._branch_forward(
            event=event.event_distant,
            ctx_current=None if ctx_current is None else ctx_current.ctx_distant,
            ctx_vsm=None if ctx_vsm is None else ctx_vsm.ctx_distant,
        )
        rigid = self._branch_forward(
            event=event.event_rigid,
            ctx_current=None if ctx_current is None else ctx_current.ctx_rigid,
            ctx_vsm=None if ctx_vsm is None else ctx_vsm.ctx_rigid,
        )
        aux = {
            "confidence_mean": float(bg.confidence.detach().mean().item()) if bg.confidence.numel() else 0.0,
            "noop_mean": float(bg.noop.detach().mean().item()) if bg.noop.numel() else 0.0,
        }
        return DeltaPack(bg=bg, distant=distant, rigid=rigid, aux=aux), aux

    def base_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("vsm_ctx_adapter.")
        }
