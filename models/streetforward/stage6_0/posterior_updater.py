from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    active_attrs: Optional[Dict[str, bool]] = None

    def is_active(self, attr: str) -> bool:
        if self.active_attrs is None:
            return True
        return bool(self.active_attrs.get(str(attr), True))


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
    _SCOPE_KEYS = {
        "means": "update_means",
        "scales_log": "update_scales",
        "quat_axis_angle": "update_quat",
        "opacity_logit": "update_opacity",
        "sh": "update_sh",
        "hidden": "update_hidden",
    }

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
        appearance_detail_attribute_gates: Optional[Dict[str, Dict[str, float]]] = None,
        appearance_detail_attribute_gate_max: Optional[Dict[str, float]] = None,
        invalid_update_policy: str = "none",
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
        self.detail_attr_names = ("means", "scales", "quat", "opacity", "sh")
        self.detail_attribute_gates_enabled = bool(appearance_detail_attribute_gates)
        self.invalid_update_policy = str(invalid_update_policy or "none").lower()
        if self.invalid_update_policy not in {"none", "hard_zero"}:
            raise ValueError(
                "Stage6PosteriorUpdater invalid_update_policy must be one of "
                f"{{'none', 'hard_zero'}}, got {invalid_update_policy!r}"
            )
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
        self.detail_gate_raw_attr: Optional[nn.Parameter]
        if self.appearance_detail_enable:
            self.detail_adapter = nn.Sequential(
                nn.Linear(int(self.appearance_detail_dim), int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            if self.detail_attribute_gates_enabled:
                gate_cfg = dict(appearance_detail_attribute_gates or {})
                max_cfg = dict(appearance_detail_attribute_gate_max or {})
                max_values = [
                    float(max_cfg.get(name, self.appearance_detail_gate_max))
                    for name in self.detail_attr_names
                ]
                self.register_buffer("detail_gate_attr_max", torch.tensor(max_values, dtype=torch.float32), persistent=False)
                rows = []
                for branch_name, defaults in (
                    ("bg", {"means": 0.05, "scales": 0.05, "quat": 0.0, "opacity": 0.10, "sh": 0.10}),
                    ("distant", {"means": 0.05, "scales": 0.05, "quat": 0.0, "opacity": 0.10, "sh": 0.10}),
                    ("rigid", {"means": 0.03, "scales": 0.03, "quat": 0.0, "opacity": 0.05, "sh": 0.05}),
                ):
                    branch_cfg = dict(gate_cfg.get(branch_name, {}) or {})
                    raw_vals = []
                    for attr_idx, attr_name in enumerate(self.detail_attr_names):
                        max_gate = max(float(max_values[attr_idx]), 1.0e-6)
                        value = float(branch_cfg.get(attr_name, defaults[attr_name]))
                        p = min(max(value / max_gate, 1.0e-4), 1.0 - 1.0e-4)
                        raw_vals.append(torch.logit(torch.tensor(p, dtype=torch.float32)))
                    rows.append(torch.stack(raw_vals))
                self.detail_gate_raw_attr = nn.Parameter(torch.stack(rows, dim=0))
                self.detail_gate_raw = None
            else:
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
                self.detail_gate_raw_attr = None
        else:
            self.detail_adapter = None
            self.detail_gate_raw = None
            self.detail_gate_raw_attr = None
        self.accept_vsm_ctx = bool(accept_vsm_ctx)
        self.vsm_ctx_adapter: Optional[nn.Linear]
        if self.accept_vsm_ctx:
            self.vsm_ctx_adapter = nn.Linear(int(vsm_ctx_dim), int(event_dim))
            nn.init.zeros_(self.vsm_ctx_adapter.weight)
            nn.init.zeros_(self.vsm_ctx_adapter.bias)
        else:
            self.vsm_ctx_adapter = None

    @staticmethod
    def _linear_with_detail_gate(
        head: nn.Linear,
        h: torch.Tensor,
        detail_delta: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        # For a linear head, head(h + g * d) == head(h) + g * linear(d, W).
        # This keeps per-attribute gates exact without materializing five [N,H] h forks.
        gate_t = gate.to(device=h.device, dtype=h.dtype)
        return head(h) + gate_t * F.linear(detail_delta, head.weight, None)

    @classmethod
    def _active_attrs_from_scope(cls, scope: Optional[Dict[str, Any]]) -> Dict[str, bool]:
        raw = dict(scope or {})
        return {
            attr: bool(raw.get(scope_key, True))
            for attr, scope_key in cls._SCOPE_KEYS.items()
        }

    def _branch_forward(
        self,
        *,
        branch_name: str,
        event: Optional[torch.Tensor],
        ctx_current: Optional[torch.Tensor],
        ctx_vsm: Optional[torch.Tensor],
        appearance_detail: Optional[torch.Tensor] = None,
        appearance_valid: Optional[torch.Tensor] = None,
        branch_scope: Optional[Dict[str, Any]] = None,
    ) -> Optional[BranchDelta]:
        if event is None:
            return None
        if branch_scope is not None and not bool(dict(branch_scope).get("enable", True)):
            return None
        active_attrs = self._active_attrs_from_scope(branch_scope)
        if self.head_hidden is None:
            active_attrs["hidden"] = False
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
                active_attrs=active_attrs,
            )
        ctx_in = event if ctx_current is None else ctx_current
        if ctx_vsm is not None:
            if self.vsm_ctx_adapter is None:
                raise ValueError("ctx_vsm was provided but vsm ctx adapter is disabled")
            ctx_in = ctx_in + self.vsm_ctx_adapter(ctx_vsm)
        h = self.trunk(ctx_in)
        h_means = h_scales = h_quat = h_opacity = h_sh = h
        h_app = h
        detail_gate = event.new_tensor(0.0)
        attr_detail_delta: Optional[torch.Tensor] = None
        attr_detail_gates: Optional[torch.Tensor] = None
        if self.appearance_detail_enable and appearance_detail is not None:
            if self.detail_adapter is None:
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
            detail_delta = self.detail_adapter(detail)
            if self.detail_attribute_gates_enabled:
                if self.detail_gate_raw_attr is None:
                    raise RuntimeError("appearance attribute detail gates are enabled but missing")
                max_attr = getattr(self, "detail_gate_attr_max").to(device=h.device, dtype=h.dtype)
                gates = torch.sigmoid(self.detail_gate_raw_attr[branch_idx]).to(device=h.device, dtype=h.dtype) * max_attr
                attr_detail_delta = detail_delta
                attr_detail_gates = gates
                detail_gate = gates.detach().mean()
            else:
                if self.detail_gate_raw is None:
                    raise RuntimeError("appearance detail is enabled but single gate is missing")
                detail_gate = torch.sigmoid(self.detail_gate_raw[branch_idx]) * float(self.appearance_detail_gate_max)
                h_app = h + detail_gate.to(device=h.device, dtype=h.dtype) * detail_delta
                h_opacity = h_app
                h_sh = h_app
        noop = torch.sigmoid(self.head_noop(h)) if self.head_noop is not None else event.new_zeros((int(event.shape[0]), 1))
        gate = 1.0 - noop
        clamps = self.branch_clamps.get(str(branch_name), self.branch_clamps["bg"])
        n_rows = int(event.shape[0])
        hidden = event.new_zeros((n_rows, self.stage_hidden_dim if self.output_hidden else 0))
        if self.head_hidden is not None and bool(active_attrs["hidden"]):
            hidden = gate * float(clamps["hidden_max_step"]) * torch.tanh(self.head_hidden(h))
        confidence = (
            torch.sigmoid(self.head_confidence(h))
            if self.head_confidence is not None
            else event.new_zeros((int(event.shape[0]), 0))
        )
        def _raw_attr(attr: str, head: nn.Linear, h_attr: torch.Tensor, gate_idx: int, cols: int) -> torch.Tensor:
            if not bool(active_attrs[attr]):
                return event.new_zeros((n_rows, int(cols)))
            if attr_detail_delta is not None and attr_detail_gates is not None:
                return self._linear_with_detail_gate(head, h, attr_detail_delta, attr_detail_gates[gate_idx])
            return head(h_attr)

        raw_means = _raw_attr("means", self.head_means, h_means, 0, 3)
        raw_scales = _raw_attr("scales_log", self.head_scales, h_scales, 1, 3)
        raw_quat = _raw_attr("quat_axis_angle", self.head_quat, h_quat, 2, 3)
        raw_opacity = _raw_attr("opacity_logit", self.head_opacity, h_opacity, 3, 1)
        raw_sh = _raw_attr("sh", self.head_sh, h_sh, 4, self.sh_dim)
        def _scaled_delta(attr: str, raw: torch.Tensor, max_step: float) -> torch.Tensor:
            if not bool(active_attrs[attr]):
                return torch.zeros_like(raw)
            return gate * float(max_step) * torch.tanh(raw)

        delta = BranchDelta(
            means=_scaled_delta("means", raw_means, float(clamps["means_max_step_m"])),
            scales_log=_scaled_delta("scales_log", raw_scales, float(clamps["scales_log_max_step"])),
            quat_axis_angle=_scaled_delta("quat_axis_angle", raw_quat, float(clamps["quat_axis_angle_max_step_rad"])),
            opacity_logit=_scaled_delta("opacity_logit", raw_opacity, float(clamps["opacity_logit_max_step"])),
            sh=_scaled_delta("sh", raw_sh, float(clamps["sh_max_step"])),
            hidden=hidden,
            confidence=confidence,
            noop=noop,
            active_attrs=active_attrs,
        )
        for name, value in delta.__dict__.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError(f"Stage6PosteriorUpdater delta {name} contains NaN/Inf")
        return delta

    @staticmethod
    def _valid_mask(
        valid: Optional[torch.Tensor],
        *,
        n: int,
        ref: torch.Tensor,
        branch_name: str,
    ) -> Optional[torch.Tensor]:
        if valid is None:
            return None
        mask = valid.to(device=ref.device, dtype=torch.bool).reshape(-1)
        if int(mask.numel()) != int(n):
            raise ValueError(
                f"EventPack.valid_{branch_name} row mismatch: got {int(mask.numel())}, expected {int(n)}"
            )
        return mask

    def _apply_invalid_update_policy(
        self,
        delta: Optional[BranchDelta],
        valid: Optional[torch.Tensor],
        *,
        branch_name: str,
    ) -> Optional[BranchDelta]:
        if delta is None or self.invalid_update_policy != "hard_zero":
            return delta
        mask = self._valid_mask(
            valid,
            n=int(delta.means.shape[0]),
            ref=delta.means,
            branch_name=branch_name,
        )
        if mask is None:
            return delta
        valid_f = mask.unsqueeze(-1).to(dtype=delta.means.dtype)
        return BranchDelta(
            means=delta.means * valid_f,
            scales_log=delta.scales_log * valid_f,
            quat_axis_angle=delta.quat_axis_angle * valid_f,
            opacity_logit=delta.opacity_logit * valid_f,
            sh=delta.sh * valid_f,
            hidden=delta.hidden * valid_f if delta.hidden.numel() else delta.hidden,
            confidence=delta.confidence * valid_f if delta.confidence.numel() else delta.confidence,
            noop=torch.where(mask.unsqueeze(-1), delta.noop, torch.ones_like(delta.noop)),
            active_attrs=delta.active_attrs,
        )

    def forward(
        self,
        *,
        event: EventPack,
        ctx_current: Optional[ContextPack] = None,
        ctx_vsm: Optional[ContextPack] = None,
        appearance_detail: Optional[AppearanceDetailPack] = None,
        branch_scope: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tuple[DeltaPack, Dict[str, Any]]:
        bg = self._branch_forward(
            branch_name="bg",
            event=event.event_bg,
            ctx_current=None if ctx_current is None else ctx_current.ctx_bg,
            ctx_vsm=None if ctx_vsm is None else ctx_vsm.ctx_bg,
            appearance_detail=None if appearance_detail is None else appearance_detail.detail_bg,
            appearance_valid=None if appearance_detail is None else appearance_detail.valid_bg,
            branch_scope=None if branch_scope is None else branch_scope.get("bg"),
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
            branch_scope=None if branch_scope is None else branch_scope.get("distant"),
        )
        rigid = self._branch_forward(
            branch_name="rigid",
            event=event.event_rigid,
            ctx_current=None if ctx_current is None else ctx_current.ctx_rigid,
            ctx_vsm=None if ctx_vsm is None else ctx_vsm.ctx_rigid,
            appearance_detail=None if appearance_detail is None else appearance_detail.detail_rigid,
            appearance_valid=None if appearance_detail is None else appearance_detail.valid_rigid,
            branch_scope=None if branch_scope is None else branch_scope.get("rigid"),
        )
        bg = self._apply_invalid_update_policy(bg, event.valid_bg, branch_name="bg")
        if bg is None:
            raise RuntimeError("EventPack.event_bg is required")
        distant = self._apply_invalid_update_policy(distant, event.valid_distant, branch_name="distant")
        rigid = self._apply_invalid_update_policy(rigid, event.valid_rigid, branch_name="rigid")
        aux = {
            "confidence_mean": float(bg.confidence.detach().mean().item()) if bg.confidence.numel() else 0.0,
            "noop_mean": float(bg.noop.detach().mean().item()) if bg.noop.numel() else 0.0,
            "posterior/invalid_update_policy_hard_zero": 1.0
            if self.invalid_update_policy == "hard_zero"
            else 0.0,
        }
        for branch_name, valid in (
            ("bg", event.valid_bg),
            ("distant", event.valid_distant),
            ("rigid", event.valid_rigid),
        ):
            if valid is not None:
                aux[f"posterior/valid_{branch_name}_ratio"] = float(
                    valid.to(dtype=torch.float32).detach().mean().item()
                ) if valid.numel() else 0.0
        if self.appearance_detail_enable and self.detail_attribute_gates_enabled and self.detail_gate_raw_attr is not None:
            max_attr = getattr(self, "detail_gate_attr_max").to(device=self.detail_gate_raw_attr.device)
            gate = torch.sigmoid(self.detail_gate_raw_attr.detach()) * max_attr.reshape(1, -1)
            for branch_idx, branch_name in enumerate(("bg", "distant", "rigid")):
                for attr_idx, attr_name in enumerate(self.detail_attr_names):
                    aux[f"posterior/detail_gate_{branch_name}_{attr_name}"] = float(gate[branch_idx, attr_idx].item())
                    aux[f"iforward/posterior/detail_gate_{branch_name}_{attr_name}"] = float(gate[branch_idx, attr_idx].item())
        elif self.appearance_detail_enable and self.detail_gate_raw is not None:
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
