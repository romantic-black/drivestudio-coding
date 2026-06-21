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
class AppearanceDetailPack:
    detail_bg: torch.Tensor
    detail_distant: Optional[torch.Tensor] = None
    detail_rigid: Optional[torch.Tensor] = None
    valid_bg: Optional[torch.Tensor] = None
    valid_distant: Optional[torch.Tensor] = None
    valid_rigid: Optional[torch.Tensor] = None
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
        branch_clamps: Optional[Dict[str, Dict[str, float]]] = None,
        output_hidden: bool = True,
        output_confidence: bool = True,
        output_noop: bool = True,
        appearance_detail_enable: bool = False,
        appearance_detail_dim: int = 8,
        appearance_detail_gate_init: Optional[Dict[str, float]] = None,
        appearance_detail_gate_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.ctx_dim = int(ctx_dim)
        self.stage_hidden_dim = max(int(stage_hidden_dim), 0)
        self.output_hidden = bool(output_hidden) and int(self.stage_hidden_dim) > 0
        self.output_confidence = bool(output_confidence)
        self.output_noop = bool(output_noop)
        self.appearance_detail_enable = bool(appearance_detail_enable)
        self.appearance_detail_dim = int(appearance_detail_dim)
        self.appearance_detail_gate_max = float(appearance_detail_gate_max)
        self.sh_dim = int(_num_sh_bases(int(sh_degree)) * 3)
        self.means_max_step_m = float(means_max_step_m)
        self.scales_log_max_step = float(scales_log_max_step)
        self.quat_axis_angle_max_step_rad = float(quat_axis_angle_max_step_rad)
        self.opacity_logit_max_step = float(opacity_logit_max_step)
        self.sh_max_step = float(sh_max_step)
        self.hidden_max_step = float(hidden_max_step)
        base_clamps = {
            "means_max_step_m": self.means_max_step_m,
            "scales_log_max_step": self.scales_log_max_step,
            "quat_axis_angle_max_step_rad": self.quat_axis_angle_max_step_rad,
            "opacity_logit_max_step": self.opacity_logit_max_step,
            "sh_max_step": self.sh_max_step,
            "hidden_max_step": self.hidden_max_step,
        }
        self.branch_clamps: Dict[str, Dict[str, float]] = {}
        raw_branch_clamps = dict(branch_clamps or {})
        for branch in ("bg", "distant", "rigid"):
            cfg = dict(raw_branch_clamps.get(branch, {}) or {})
            self.branch_clamps[branch] = {
                key: float(cfg.get(key, default))
                for key, default in base_clamps.items()
            }
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
        self.head_hidden = nn.Linear(int(hidden_dim), self.stage_hidden_dim) if self.output_hidden else None
        self.head_confidence = nn.Linear(int(hidden_dim), 1) if self.output_confidence else None
        self.head_noop = nn.Linear(int(hidden_dim), 1) if self.output_noop else None
        self.detail_adapter: Optional[nn.Module]
        self.detail_gate_raw: Optional[nn.Parameter]
        if self.appearance_detail_enable:
            self.detail_adapter = nn.Sequential(
                nn.Linear(int(self.appearance_detail_dim), int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            init_cfg = dict(appearance_detail_gate_init or {})
            init_values = [
                float(init_cfg.get("bg", 0.10)),
                float(init_cfg.get("distant", 0.10)),
                float(init_cfg.get("rigid", 0.05)),
            ]
            raw = []
            max_gate = max(float(self.appearance_detail_gate_max), 1.0e-6)
            for value in init_values:
                p = min(max(float(value) / max_gate, 1.0e-4), 1.0 - 1.0e-4)
                raw.append(torch.logit(torch.tensor(p, dtype=torch.float32)))
            self.detail_gate_raw = nn.Parameter(torch.stack(raw))
        else:
            self.detail_adapter = None
            self.detail_gate_raw = None
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
        branch_name: str,
        event: Optional[torch.Tensor],
        ctx_current: Optional[torch.Tensor],
        ctx_vsm: Optional[torch.Tensor],
        appearance_detail: Optional[torch.Tensor] = None,
        appearance_valid: Optional[torch.Tensor] = None,
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
                hidden=event.new_zeros((n, self.stage_hidden_dim if self.output_hidden else 0)),
                confidence=z1 if self.output_confidence else event.new_zeros((n, 0)),
                noop=z1,
            )
        ctx_in = event if ctx_current is None else ctx_current
        if ctx_vsm is not None:
            if self.vsm_ctx_adapter is None:
                raise ValueError("ctx_vsm was provided but vsm ctx adapter is disabled")
            ctx_in = ctx_in + self.vsm_ctx_adapter(ctx_vsm)
        h = self.trunk(ctx_in)
        h_app = h
        detail_gate = event.new_tensor(0.0)
        if self.appearance_detail_enable and appearance_detail is not None:
            if self.detail_adapter is None or self.detail_gate_raw is None:
                raise RuntimeError("appearance detail is enabled but adapter/gate are missing")
            if appearance_detail.dim() != 2 or int(appearance_detail.shape[0]) != int(event.shape[0]):
                raise ValueError(
                    f"appearance_detail row mismatch for {branch_name}: "
                    f"detail={tuple(appearance_detail.shape)} event={tuple(event.shape)}"
                )
            if int(appearance_detail.shape[-1]) != int(self.appearance_detail_dim):
                raise ValueError(
                    f"appearance_detail dim mismatch for {branch_name}: "
                    f"got {int(appearance_detail.shape[-1])}, expected {int(self.appearance_detail_dim)}"
                )
            detail = appearance_detail.to(device=event.device, dtype=event.dtype)
            if appearance_valid is not None:
                valid = appearance_valid.to(device=event.device, dtype=torch.bool).reshape(-1, 1)
                if int(valid.shape[0]) != int(event.shape[0]):
                    raise ValueError(f"appearance_valid row mismatch for {branch_name}")
                detail = torch.where(valid, detail, torch.zeros_like(detail))
            branch_idx = {"bg": 0, "distant": 1, "rigid": 2}.get(str(branch_name), 0)
            detail_gate = torch.sigmoid(self.detail_gate_raw[branch_idx]) * float(self.appearance_detail_gate_max)
            h_app = h + detail_gate.to(device=h.device, dtype=h.dtype) * self.detail_adapter(detail)
        noop = torch.sigmoid(self.head_noop(h)) if self.head_noop is not None else event.new_zeros((int(event.shape[0]), 1))
        gate = 1.0 - noop
        clamps = self.branch_clamps.get(str(branch_name), self.branch_clamps["bg"])
        hidden = (
            gate * float(clamps["hidden_max_step"]) * torch.tanh(self.head_hidden(h))
            if self.head_hidden is not None
            else event.new_zeros((int(event.shape[0]), 0))
        )
        confidence = (
            torch.sigmoid(self.head_confidence(h))
            if self.head_confidence is not None
            else event.new_zeros((int(event.shape[0]), 0))
        )
        delta = BranchDelta(
            means=gate * float(clamps["means_max_step_m"]) * torch.tanh(self.head_means(h)),
            scales_log=gate * float(clamps["scales_log_max_step"]) * torch.tanh(self.head_scales(h)),
            quat_axis_angle=gate * float(clamps["quat_axis_angle_max_step_rad"]) * torch.tanh(self.head_quat(h)),
            opacity_logit=gate * float(clamps["opacity_logit_max_step"]) * torch.tanh(self.head_opacity(h_app)),
            sh=gate * float(clamps["sh_max_step"]) * torch.tanh(self.head_sh(h_app)),
            hidden=hidden,
            confidence=confidence,
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
        appearance_detail: Optional[AppearanceDetailPack] = None,
    ) -> tuple[DeltaPack, Dict[str, Any]]:
        bg = self._branch_forward(
            branch_name="bg",
            event=event.event_bg,
            ctx_current=None if ctx_current is None else ctx_current.ctx_bg,
            ctx_vsm=None if ctx_vsm is None else ctx_vsm.ctx_bg,
            appearance_detail=None if appearance_detail is None else appearance_detail.detail_bg,
            appearance_valid=None if appearance_detail is None else appearance_detail.valid_bg,
        )
        if bg is None:
            raise RuntimeError("EventPack.event_bg is required")
        distant = self._branch_forward(
            branch_name="distant",
            event=event.event_distant,
            ctx_current=None if ctx_current is None else ctx_current.ctx_distant,
            ctx_vsm=None if ctx_vsm is None else ctx_vsm.ctx_distant,
            appearance_detail=None if appearance_detail is None else appearance_detail.detail_distant,
            appearance_valid=None if appearance_detail is None else appearance_detail.valid_distant,
        )
        rigid = self._branch_forward(
            branch_name="rigid",
            event=event.event_rigid,
            ctx_current=None if ctx_current is None else ctx_current.ctx_rigid,
            ctx_vsm=None if ctx_vsm is None else ctx_vsm.ctx_rigid,
            appearance_detail=None if appearance_detail is None else appearance_detail.detail_rigid,
            appearance_valid=None if appearance_detail is None else appearance_detail.valid_rigid,
        )
        aux = {
            "confidence_mean": float(bg.confidence.detach().mean().item()) if bg.confidence.numel() else 0.0,
            "noop_mean": float(bg.noop.detach().mean().item()) if bg.noop.numel() else 0.0,
        }
        if self.appearance_detail_enable and self.detail_gate_raw is not None:
            gate = torch.sigmoid(self.detail_gate_raw.detach()) * float(self.appearance_detail_gate_max)
            aux.update(
                {
                    "posterior/detail_gate_bg": float(gate[0].item()),
                    "posterior/detail_gate_distant": float(gate[1].item()),
                    "posterior/detail_gate_rigid": float(gate[2].item()),
                    "iforward/posterior/detail_gate_bg": float(gate[0].item()),
                    "iforward/posterior/detail_gate_distant": float(gate[1].item()),
                    "iforward/posterior/detail_gate_rigid": float(gate[2].item()),
                }
            )
        return DeltaPack(bg=bg, distant=distant, rigid=rigid, aux=aux), aux

    def base_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("vsm_ctx_adapter.")
        }
