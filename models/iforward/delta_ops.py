from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


def _mul_attr(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return value * gate.to(device=value.device, dtype=value.dtype)


def gate_branch_delta(delta: BranchDelta, gate: Any) -> BranchDelta:
    """Apply an attribute gate to one Stage6 branch delta."""

    return BranchDelta(
        means=_mul_attr(delta.means, gate.means),
        scales_log=_mul_attr(delta.scales_log, gate.scales),
        quat_axis_angle=_mul_attr(delta.quat_axis_angle, gate.quat),
        opacity_logit=_mul_attr(delta.opacity_logit, gate.opacity),
        sh=_mul_attr(delta.sh, gate.sh),
        hidden=_mul_attr(delta.hidden, gate.hidden),
        confidence=delta.confidence,
        noop=delta.noop,
    )


def gate_delta_pack(delta: DeltaPack, gate: Any) -> DeltaPack:
    return DeltaPack(
        bg=gate_branch_delta(delta.bg, gate.bg),
        distant=(
            gate_branch_delta(delta.distant, gate.distant)
            if delta.distant is not None and getattr(gate, "distant", None) is not None
            else delta.distant
        ),
        rigid=(
            gate_branch_delta(delta.rigid, gate.rigid)
            if delta.rigid is not None and getattr(gate, "rigid", None) is not None
            else delta.rigid
        ),
        aux=delta.aux,
    )


def branch_means_update_norm(delta: Optional[BranchDelta], *, rows: Optional[int] = None, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    if delta is not None:
        return delta.means.detach().norm(dim=-1, keepdim=True)
    if ref is not None:
        return ref.new_zeros((int(rows or 0), 1))
    return torch.zeros((int(rows or 0), 1), dtype=torch.float32)


def delta_means_update_norms(delta: DeltaPack) -> Dict[str, torch.Tensor]:
    out = {"bg": branch_means_update_norm(delta.bg)}
    if delta.distant is not None:
        out["distant"] = branch_means_update_norm(delta.distant)
    if delta.rigid is not None:
        out["rigid"] = branch_means_update_norm(delta.rigid)
    return out


def branch_delta_norm_stats(delta: Optional[BranchDelta], *, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, value in (
        ("means", getattr(delta, "means", None) if delta is not None else None),
        ("scales", getattr(delta, "scales_log", None) if delta is not None else None),
        ("quat", getattr(delta, "quat_axis_angle", None) if delta is not None else None),
        ("opacity", getattr(delta, "opacity_logit", None) if delta is not None else None),
        ("sh", getattr(delta, "sh", None) if delta is not None else None),
    ):
        if torch.is_tensor(value) and int(value.numel()) > 0:
            out[f"{prefix}/{name}_norm_mean"] = float(value.detach().reshape(int(value.shape[0]), -1).norm(dim=-1).mean().item())
        else:
            out[f"{prefix}/{name}_norm_mean"] = 0.0
    return out


__all__ = [
    "branch_delta_norm_stats",
    "branch_means_update_norm",
    "delta_means_update_norms",
    "gate_branch_delta",
    "gate_delta_pack",
]
