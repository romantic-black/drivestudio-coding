from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from models.streetforward.math_utils import _normalize_quat, _quat_to_rotmat


def mass_mode_to_id(mode: str) -> int:
    mode_l = str(mode).lower()
    if mode_l in {"dynamic_tau_area", "dynamic", "tau_area"}:
        return 0
    if mode_l in {"static_assignment_mass", "static", "assignment"}:
        return 1
    raise ValueError(f"unsupported BigGS diagonal projector mass_mode={mode!r}")


def mass_mode_from_id(mode_id: int) -> str:
    return "static_assignment_mass" if int(mode_id) == 1 else "dynamic_tau_area"


def child_to_parent_from_grouped(
    *,
    child_order: torch.Tensor,
    parent_count: torch.Tensor,
    num_children: int,
) -> torch.Tensor:
    device = child_order.device
    parent_count_l = parent_count.to(device=device, dtype=torch.long).reshape(-1)
    parent_ids_ordered = torch.repeat_interleave(
        torch.arange(int(parent_count_l.numel()), dtype=torch.long, device=device),
        parent_count_l,
    )
    if int(parent_ids_ordered.numel()) != int(num_children):
        raise ValueError(
            "BigGS grouped assignment length mismatch: "
            f"sum(parent_count)={int(parent_ids_ordered.numel())} num_children={int(num_children)}"
        )
    out = torch.empty((int(num_children),), dtype=torch.long, device=device)
    out.scatter_(0, child_order.to(device=device, dtype=torch.long).reshape(-1), parent_ids_ordered)
    return out


def _top2_area(scales: torch.Tensor) -> torch.Tensor:
    if int(scales.shape[-1]) < 2:
        return scales.reshape(scales.shape[:-1] + (-1,)).prod(dim=-1)
    return torch.topk(scales, k=2, dim=-1).values.prod(dim=-1)


def project_biggs_parent_diag_reference_tensors(
    *,
    means: torch.Tensor,
    scales_log: torch.Tensor,
    quats: torch.Tensor,
    opacity_logit: torch.Tensor,
    sh_dc: torch.Tensor,
    sh_rest: torch.Tensor,
    child_mass: torch.Tensor,
    child_to_parent: Optional[torch.Tensor] = None,
    child_order: Optional[torch.Tensor] = None,
    parent_count: torch.Tensor,
    min_scale: float,
    max_scale: float,
    opacity_cap: float,
    opacity_min: float,
    tau_parent_scale: float,
    eps: float,
    min_mass: float,
    mass_mode: str | int,
) -> Tuple[torch.Tensor, ...]:
    n = int(means.shape[0])
    m = int(parent_count.numel())
    b = int(sh_rest.shape[1])
    if n == 0 or m == 0:
        parent_means = means.new_zeros((m, 3))
        parent_scales_log = means.new_zeros((m, 3))
        parent_quats = means.new_zeros((m, 4))
        if m:
            parent_quats[:, 0] = 1.0
        parent_opacity_logit = means.new_zeros((m, 1))
        parent_sh_dc = means.new_zeros((m, 3))
        parent_sh_rest = means.new_zeros((m, b, 3))
        mass_sum = means.new_zeros((m,))
        mass_mean = means.new_zeros((m,))
        return (
            parent_means,
            parent_scales_log,
            parent_quats,
            parent_opacity_logit,
            parent_sh_dc,
            parent_sh_rest,
            mass_sum,
            mass_mean,
        )

    if child_to_parent is None:
        if child_order is None:
            raise ValueError("child_to_parent or child_order is required for diagonal parent projection")
        child_to_parent = child_to_parent_from_grouped(
            child_order=child_order,
            parent_count=parent_count,
            num_children=n,
        )
    pid = child_to_parent.to(device=means.device, dtype=torch.long).reshape(-1)
    if int(pid.numel()) != n:
        raise ValueError("BigGS diagonal parent projection child_to_parent length mismatch")

    dtype = means.dtype
    counts = parent_count.to(device=means.device, dtype=dtype).reshape(-1)
    min_scale_f = float(min_scale)
    max_scale_f = float(max_scale)
    eps_f = float(eps)
    min_mass_f = float(min_mass)
    mass_mode_id = int(mass_mode) if isinstance(mass_mode, int) else mass_mode_to_id(str(mass_mode))

    scales = torch.exp(scales_log)
    tau_child = F.softplus(opacity_logit.reshape(-1))
    child_area = _top2_area(scales)
    if mass_mode_id == 0:
        mass = (tau_child * child_area).clamp_min(min_mass_f)
    elif mass_mode_id == 1:
        mass = child_mass.to(device=means.device, dtype=dtype).reshape(-1).clamp_min(min_mass_f)
    else:
        raise ValueError(f"unsupported BigGS diagonal projector mass_mode_id={mass_mode_id}")

    mass_sum = means.new_zeros((m,))
    mass_sum.index_add_(0, pid, mass)
    mass_safe = mass_sum.clamp_min(min_mass_f)
    mass_mean = mass_sum / counts.clamp_min(1.0)

    weighted_means = means.new_zeros((m, 3))
    weighted_means.index_add_(0, pid, means * mass[:, None])
    parent_means = weighted_means / mass_safe[:, None]

    rot = _quat_to_rotmat(_normalize_quat(quats))
    child_diag_cov = (rot.square() * scales.square()[:, None, :]).sum(dim=-1)
    second = child_diag_cov + means.square()
    weighted_second = means.new_zeros((m, 3))
    weighted_second.index_add_(0, pid, second * mass[:, None])
    var = weighted_second / mass_safe[:, None] - parent_means.square()
    var = torch.clamp(
        var + eps_f,
        min=min_scale_f * min_scale_f,
        max=max_scale_f * max_scale_f,
    )
    parent_scales = torch.sqrt(var).clamp(min=min_scale_f, max=max_scale_f)
    parent_scales_log = torch.log(parent_scales.clamp_min(min_scale_f))

    tau_area = means.new_zeros((m,))
    tau_area.index_add_(0, pid, tau_child * child_area)
    parent_area = _top2_area(parent_scales).clamp_min(eps_f)
    tau_parent = float(tau_parent_scale) * tau_area / (parent_area + eps_f)
    opacity_parent = float(opacity_cap) * (1.0 - torch.exp(-tau_parent))
    opacity_parent = opacity_parent.clamp(float(opacity_min), float(opacity_cap) - eps_f)
    parent_opacity_logit = torch.logit(opacity_parent).reshape(m, 1)

    parent_sh_dc = means.new_zeros((m, 3))
    parent_sh_dc.index_add_(0, pid, sh_dc * mass[:, None])
    parent_sh_dc = parent_sh_dc / mass_safe[:, None]
    parent_sh_rest = means.new_zeros((m, b, 3))
    parent_sh_rest.index_add_(0, pid, sh_rest * mass[:, None, None])
    parent_sh_rest = parent_sh_rest / mass_safe[:, None, None]

    valid = parent_count.to(device=means.device, dtype=torch.long).reshape(-1) > 0
    parent_quats = means.new_zeros((m, 4))
    parent_quats[:, 0] = 1.0
    parent_means = torch.where(valid[:, None], parent_means, torch.zeros_like(parent_means))
    parent_sh_dc = torch.where(valid[:, None], parent_sh_dc, torch.zeros_like(parent_sh_dc))
    parent_sh_rest = torch.where(valid[:, None, None], parent_sh_rest, torch.zeros_like(parent_sh_rest))
    return (
        parent_means,
        parent_scales_log,
        parent_quats,
        parent_opacity_logit,
        parent_sh_dc,
        parent_sh_rest,
        mass_sum,
        mass_mean,
    )


__all__ = [
    "child_to_parent_from_grouped",
    "mass_mode_from_id",
    "mass_mode_to_id",
    "project_biggs_parent_diag_reference_tensors",
]
