from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0 import DeltaPack, EventPack, LocalGSState
from models.streetforward.stage6_0.local_gs_state import LocalBranchState
from models.streetforward.stage6_0.posterior_updater import BranchDelta

from .utils import cfg_get


@dataclass
class HistoryGradBranch:
    means: Optional[torch.Tensor] = None
    scales_log: Optional[torch.Tensor] = None
    opacity_logit: Optional[torch.Tensor] = None
    sh: Optional[torch.Tensor] = None


@dataclass
class HistoryGradPack:
    bg: HistoryGradBranch
    distant: Optional[HistoryGradBranch] = None
    rigid: Optional[HistoryGradBranch] = None


_ATTR_TO_DELTA_FIELD = {
    "means": "means",
    "scales": "scales_log",
    "opacity": "opacity_logit",
    "sh": "sh",
}


def _bool_cfg(node: Any, key: str, default: bool) -> bool:
    return bool(cfg_get(node, key, default))


def _float_cfg(node: Any, key: str, default: float) -> float:
    return float(cfg_get(node, key, default))


def _zero_loss(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_tensor(0.0)


def _branch_row_count(branch: Optional[LocalBranchState]) -> int:
    return int(branch.means.shape[0]) if branch is not None else 0


def _clone_leaf(value: torch.Tensor, *, requires_grad: bool) -> torch.Tensor:
    out = value.detach().clone()
    if bool(requires_grad) and torch.is_floating_point(out):
        out.requires_grad_(True)
    return out


def _clone_probe_branch(
    branch: Optional[LocalBranchState],
    *,
    branch_name: str,
    attrs: Mapping[str, bool],
    tensor_items: List[Tuple[str, str, torch.Tensor]],
) -> Optional[LocalBranchState]:
    if branch is None:
        return None

    def leaf(name: str, value: torch.Tensor, enabled: bool) -> torch.Tensor:
        out = _clone_leaf(value, requires_grad=bool(enabled))
        if bool(enabled):
            tensor_items.append((str(branch_name), str(name), out))
        return out

    sh_enabled = bool(attrs.get("sh", False))
    return LocalBranchState(
        means=leaf("means", branch.means, bool(attrs.get("means", False))),
        scales_log=leaf("scales_log", branch.scales_log, bool(attrs.get("scales", False))),
        quats=_clone_leaf(branch.quats, requires_grad=False),
        opacity_logit=leaf("opacity_logit", branch.opacity_logit, bool(attrs.get("opacity", False))),
        sh_dc=leaf("sh_dc", branch.sh_dc, sh_enabled),
        sh_rest=leaf("sh_rest", branch.sh_rest, sh_enabled),
        hidden=_clone_leaf(branch.hidden, requires_grad=False),
        appearance_logvar=_clone_leaf(branch.appearance_logvar, requires_grad=False).float(),
    )


def make_probe_local_state(
    local_state: LocalGSState,
    *,
    attrs: Mapping[str, bool],
) -> Tuple[LocalGSState, List[Tuple[str, str, torch.Tensor]]]:
    tensor_items: List[Tuple[str, str, torch.Tensor]] = []
    probe_state = LocalGSState(
        bg=_clone_probe_branch(local_state.bg, branch_name="bg", attrs=attrs, tensor_items=tensor_items),
        distant=_clone_probe_branch(
            local_state.distant,
            branch_name="distant",
            attrs=attrs,
            tensor_items=tensor_items,
        ),
        rigid=_clone_probe_branch(
            local_state.rigid,
            branch_name="rigid",
            attrs=attrs,
            tensor_items=tensor_items,
        ),
        rigid_template=local_state.rigid_template.detach_clone() if local_state.rigid_template is not None else None,
    )
    return probe_state, tensor_items


def _zeros_like_optional(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if value is None else torch.zeros_like(value)


def _grad_map(
    tensor_items: List[Tuple[str, str, torch.Tensor]],
    grads: Tuple[Optional[torch.Tensor], ...],
) -> Dict[Tuple[str, str], torch.Tensor]:
    out: Dict[Tuple[str, str], torch.Tensor] = {}
    for (branch_name, attr_name, tensor), grad in zip(tensor_items, grads):
        if grad is None:
            grad = torch.zeros_like(tensor)
        out[(str(branch_name), str(attr_name))] = grad.detach()
    return out


def _grad_branch_from_map(
    *,
    branch_name: str,
    branch: Optional[LocalBranchState],
    grad_by_name: Mapping[Tuple[str, str], torch.Tensor],
    attrs: Mapping[str, bool],
) -> Optional[HistoryGradBranch]:
    if branch is None:
        return None
    means = grad_by_name.get((branch_name, "means")) if bool(attrs.get("means", False)) else None
    scales_log = grad_by_name.get((branch_name, "scales_log")) if bool(attrs.get("scales", False)) else None
    opacity_logit = grad_by_name.get((branch_name, "opacity_logit")) if bool(attrs.get("opacity", False)) else None
    sh = None
    if bool(attrs.get("sh", False)):
        sh_dc = grad_by_name.get((branch_name, "sh_dc"))
        sh_rest = grad_by_name.get((branch_name, "sh_rest"))
        sh_dc = torch.zeros_like(branch.sh_dc) if sh_dc is None else sh_dc
        sh_rest = torch.zeros_like(branch.sh_rest) if sh_rest is None else sh_rest
        sh = torch.cat([sh_dc[:, None, :], sh_rest], dim=1).reshape(int(branch.sh_dc.shape[0]), -1)
    return HistoryGradBranch(
        means=means,
        scales_log=scales_log,
        opacity_logit=opacity_logit,
        sh=sh,
    )


def grad_state_to_grad_pack(
    *,
    probe_state: LocalGSState,
    tensor_items: List[Tuple[str, str, torch.Tensor]],
    grads: Tuple[Optional[torch.Tensor], ...],
    attrs: Mapping[str, bool],
) -> HistoryGradPack:
    grad_by_name = _grad_map(tensor_items, grads)
    return HistoryGradPack(
        bg=_grad_branch_from_map(
            branch_name="bg",
            branch=probe_state.bg,
            grad_by_name=grad_by_name,
            attrs=attrs,
        )
        or HistoryGradBranch(),
        distant=_grad_branch_from_map(
            branch_name="distant",
            branch=probe_state.distant,
            grad_by_name=grad_by_name,
            attrs=attrs,
        ),
        rigid=_grad_branch_from_map(
            branch_name="rigid",
            branch=probe_state.rigid,
            grad_by_name=grad_by_name,
            attrs=attrs,
        ),
    )


def _select_rows(value: Optional[torch.Tensor], rows: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if rows is None:
        return value
    return value[rows.to(device=value.device, dtype=torch.long).reshape(-1)]


def _event_grad_branch(
    full: Optional[HistoryGradBranch],
    delta: Optional[BranchDelta],
    *,
    branch_name: str,
    rows: Optional[torch.Tensor] = None,
) -> Optional[HistoryGradBranch]:
    if delta is None:
        return None
    if full is None:
        return HistoryGradBranch(
            means=_zeros_like_optional(delta.means),
            scales_log=_zeros_like_optional(delta.scales_log),
            opacity_logit=_zeros_like_optional(delta.opacity_logit),
            sh=_zeros_like_optional(delta.sh),
        )
    out = HistoryGradBranch(
        means=_select_rows(full.means, rows),
        scales_log=_select_rows(full.scales_log, rows),
        opacity_logit=_select_rows(full.opacity_logit, rows),
        sh=_select_rows(full.sh, rows),
    )
    for attr, grad_value in (
        ("means", out.means),
        ("scales_log", out.scales_log),
        ("opacity_logit", out.opacity_logit),
        ("sh", out.sh),
    ):
        delta_value = getattr(delta, attr)
        if grad_value is not None and int(grad_value.shape[0]) != int(delta_value.shape[0]):
            raise ValueError(
                f"HSP {branch_name}.{attr} grad/event row mismatch: "
                f"grad={tuple(grad_value.shape)} delta={tuple(delta_value.shape)}"
            )
    return out


def grad_pack_to_event_rows(
    grad_full: HistoryGradPack,
    *,
    event: EventPack,
    delta_event: DeltaPack,
) -> HistoryGradPack:
    route = getattr(event, "route", None)
    rows = getattr(route, "S", None) if route is not None else None
    return HistoryGradPack(
        bg=_event_grad_branch(grad_full.bg, delta_event.bg, branch_name="bg") or HistoryGradBranch(),
        distant=_event_grad_branch(grad_full.distant, delta_event.distant, branch_name="distant"),
        rigid=_event_grad_branch(grad_full.rigid, delta_event.rigid, branch_name="rigid", rows=rows),
    )


def _is_finite_tensor(value: Optional[torch.Tensor]) -> bool:
    return value is None or bool(torch.isfinite(value).all().item())


def _grad_pack_is_finite(pack: HistoryGradPack) -> bool:
    for branch in (pack.bg, pack.distant, pack.rigid):
        if branch is None:
            continue
        for value in (branch.means, branch.scales_log, branch.opacity_logit, branch.sh):
            if not _is_finite_tensor(value):
                return False
    return True


def _stable_probe_hash(*values: int) -> int:
    h = 2166136261
    for raw in values:
        value = int(raw) & 0xFFFFFFFF
        h ^= value
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def select_history_probe_indices(
    *,
    resolved: Any,
    step: Any,
    frames_per_block: int = 1,
    cams_per_frame: int = 1,
) -> Tuple[int, ...]:
    history_indices = [int(x) for x in tuple(getattr(resolved, "history_rollout_target_indices", ()) or ())]
    if not history_indices:
        return ()
    target_refs = tuple(getattr(resolved, "target_refs", ()) or ())
    frame_to_indices: Dict[int, List[int]] = {}
    for idx in history_indices:
        if idx < 0 or idx >= len(target_refs):
            continue
        ref = target_refs[int(idx)]
        frame_to_indices.setdefault(int(ref[0]), []).append(int(idx))
    if not frame_to_indices:
        return ()
    ordered_frames: List[int] = []
    for idx in history_indices:
        if idx < 0 or idx >= len(target_refs):
            continue
        frame = int(target_refs[int(idx)][0])
        if frame not in frame_to_indices:
            continue
        if frame in ordered_frames:
            ordered_frames.remove(frame)
        ordered_frames.append(frame)
    if not ordered_frames:
        return ()
    selected_frames = ordered_frames[-max(int(frames_per_block), 1) :]
    out: List[int] = []
    window_hash = int(getattr(resolved, "window_hash", 0) or 0)
    block_id = int(getattr(step, "block_id", getattr(step, "episode_block_idx", 0)) or 0)
    step_idx = int(getattr(step, "step_idx", 0) or 0)
    for frame in selected_frames:
        indices = sorted(frame_to_indices[int(frame)], key=lambda i: int(target_refs[int(i)][1]))
        if int(cams_per_frame) <= 0 or int(cams_per_frame) >= len(indices):
            out.extend(indices)
            continue
        start = _stable_probe_hash(window_hash, block_id, step_idx, int(frame)) % len(indices)
        for j in range(int(cams_per_frame)):
            out.append(indices[(start + j) % len(indices)])
    return tuple(int(x) for x in out)


def _schedule_value(global_step: int, cfg: Any) -> float:
    start_step = int(cfg_get(cfg, "start_step", 0))
    warmup_steps = int(cfg_get(cfg, "warmup_steps", 0))
    start_value = float(cfg_get(cfg, "start_value", 0.0))
    end_value = float(cfg_get(cfg, "end_value", 0.0))
    if warmup_steps <= 0:
        return float(end_value)
    progress = (int(global_step) - int(start_step)) / float(max(int(warmup_steps), 1))
    progress = min(max(float(progress), 0.0), 1.0)
    return float(start_value + (end_value - start_value) * progress)


def project_attr(
    delta_attr: torch.Tensor,
    grad_attr: torch.Tensor,
    *,
    strength: float,
    tau_norm: float = 0.0,
    tau_cos: float = 0.0,
    eps: float = 1.0e-8,
    loss_type: str = "cosine_conflict",
    loss_enabled: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    n = int(delta_attr.shape[0])
    if n == 0:
        return delta_attr, delta_attr.new_tensor(0.0), {
            "active": 0.0,
            "damage_pos_ratio": 0.0,
            "damage_norm_mean": 0.0,
            "cos_damage_pos_ratio": 0.0,
            "cos_damage_mean": 0.0,
            "cos_damage_loss": 0.0,
            "projection_norm_ratio": 0.0,
            "delta_norm_before": 0.0,
            "delta_norm_after": 0.0,
        }
    d = delta_attr.reshape(n, -1)
    g = grad_attr.detach().to(device=delta_attr.device, dtype=delta_attr.dtype).reshape(n, -1)
    dot = (d * g).sum(dim=-1, keepdim=True)
    norm2 = (g * g).sum(dim=-1, keepdim=True) + float(eps)
    normalized_dot = dot / torch.sqrt(norm2)
    damage = torch.relu(normalized_dot - float(tau_norm))
    before_norm = d.norm(dim=-1)
    delta_norm = before_norm.reshape(n, 1)
    cos = dot / (torch.sqrt(norm2) * delta_norm.clamp_min(float(eps)))
    cos_damage = torch.relu(cos - float(tau_cos))
    raw_threshold = float(tau_norm) * torch.sqrt(norm2)
    violation = torch.relu(dot - raw_threshold)
    coeff = float(strength) * violation / norm2
    d_safe = d - coeff * g
    delta_safe = d_safe.reshape_as(delta_attr)
    proj_norm = (d - d_safe).norm(dim=-1)
    after_norm = d_safe.norm(dim=-1)
    ratio = proj_norm / before_norm.clamp_min(float(eps))
    active = bool(float(before_norm.detach().norm().item()) > float(eps))
    if not active:
        delta_safe = delta_attr
        d_safe = d
        proj_norm = torch.zeros_like(before_norm)
        after_norm = before_norm
        ratio = torch.zeros_like(before_norm)
    if not bool(loss_enabled) or not active:
        loss = delta_attr.new_tensor(0.0)
    elif str(loss_type) == "cosine_conflict":
        loss = cos_damage.pow(2).mean()
    elif str(loss_type) in {"normalized_dot", "normalized", "legacy_norm"}:
        loss = damage.pow(2).mean()
    else:
        raise ValueError(f"unsupported HSP damage_loss.type={loss_type!r}")
    stats = {
        "active": 1.0 if active else 0.0,
        "damage_pos_ratio": float((damage.detach() > 0).to(dtype=torch.float32).mean().item()) if damage.numel() else 0.0,
        "damage_norm_mean": float(damage.detach().mean().item()) if damage.numel() else 0.0,
        "cos_damage_pos_ratio": float((cos_damage.detach() > 0).to(dtype=torch.float32).mean().item()) if cos_damage.numel() else 0.0,
        "cos_damage_mean": float(cos_damage.detach().mean().item()) if cos_damage.numel() else 0.0,
        "cos_damage_loss": float(cos_damage.detach().pow(2).mean().item()) if cos_damage.numel() else 0.0,
        "projection_norm_ratio": float(ratio.detach().mean().item()) if ratio.numel() else 0.0,
        "delta_norm_before": float(before_norm.detach().mean().item()) if before_norm.numel() else 0.0,
        "delta_norm_after": float(after_norm.detach().mean().item()) if after_norm.numel() else 0.0,
    }
    return delta_safe, loss, stats


def _project_branch(
    delta: Optional[BranchDelta],
    grad: Optional[HistoryGradBranch],
    *,
    attrs: Mapping[str, bool],
    strength_by_attr: Mapping[str, float],
    tau_norm_by_attr: Mapping[str, float],
    tau_cos_by_attr: Mapping[str, float],
    attr_weights: Mapping[str, float],
    eps: float,
    prefix: str,
    mode: str,
    loss_type: str,
    loss_enabled: bool,
) -> Tuple[Optional[BranchDelta], List[torch.Tensor], Dict[str, float]]:
    if delta is None:
        return None, [], {}
    grad = grad or HistoryGradBranch()
    out = delta
    losses: List[torch.Tensor] = []
    aux: Dict[str, float] = {}
    replacements: Dict[str, torch.Tensor] = {}
    for attr, field_name in _ATTR_TO_DELTA_FIELD.items():
        if not bool(attrs.get(attr, False)):
            continue
        delta_attr = getattr(delta, field_name)
        grad_attr = getattr(grad, field_name)
        if grad_attr is None:
            grad_attr = torch.zeros_like(delta_attr)
        strength = float(strength_by_attr.get(attr, 0.0)) if str(mode) == "project_delta" else 0.0
        safe_attr, loss_attr, stats = project_attr(
            delta_attr,
            grad_attr,
            strength=float(strength),
            tau_norm=float(tau_norm_by_attr.get(attr, 0.0)),
            tau_cos=float(tau_cos_by_attr.get(attr, 0.0)),
            eps=float(eps),
            loss_type=str(loss_type),
            loss_enabled=bool(loss_enabled),
        )
        replacements[field_name] = safe_attr
        if bool(loss_enabled) and float(stats.get("active", 0.0)) > 0.0:
            losses.append(float(attr_weights.get(attr, 1.0)) * loss_attr)
        for key, value in stats.items():
            aux[f"hsp/{prefix}_{key}/{attr}"] = float(value)
    if replacements:
        out = replace(delta, **replacements)
    return out, losses, aux


def project_delta_pack(
    delta_event: DeltaPack,
    grad_event: HistoryGradPack,
    *,
    attrs: Mapping[str, bool],
    strength_by_attr: Mapping[str, float],
    tau_norm_by_attr: Mapping[str, float],
    tau_cos_by_attr: Optional[Mapping[str, float]] = None,
    attr_weights: Mapping[str, float],
    eps: float = 1.0e-8,
    mode: str = "project_delta",
    loss_type: str = "cosine_conflict",
    loss_enabled: bool = True,
) -> Tuple[DeltaPack, torch.Tensor, Dict[str, float]]:
    ref = delta_event.bg.means
    all_losses: List[torch.Tensor] = []
    aux: Dict[str, float] = {}
    tau_cos_by_attr = tau_cos_by_attr or {}
    bg, losses, stats = _project_branch(
        delta_event.bg,
        grad_event.bg,
        attrs=attrs,
        strength_by_attr=strength_by_attr,
        tau_norm_by_attr=tau_norm_by_attr,
        tau_cos_by_attr=tau_cos_by_attr,
        attr_weights=attr_weights,
        eps=float(eps),
        prefix="bg",
        mode=str(mode),
        loss_type=str(loss_type),
        loss_enabled=bool(loss_enabled),
    )
    all_losses.extend(losses)
    aux.update(stats)
    distant, losses, stats = _project_branch(
        delta_event.distant,
        grad_event.distant,
        attrs=attrs,
        strength_by_attr=strength_by_attr,
        tau_norm_by_attr=tau_norm_by_attr,
        tau_cos_by_attr=tau_cos_by_attr,
        attr_weights=attr_weights,
        eps=float(eps),
        prefix="distant",
        mode=str(mode),
        loss_type=str(loss_type),
        loss_enabled=bool(loss_enabled),
    )
    all_losses.extend(losses)
    aux.update(stats)
    rigid, losses, stats = _project_branch(
        delta_event.rigid,
        grad_event.rigid,
        attrs=attrs,
        strength_by_attr=strength_by_attr,
        tau_norm_by_attr=tau_norm_by_attr,
        tau_cos_by_attr=tau_cos_by_attr,
        attr_weights=attr_weights,
        eps=float(eps),
        prefix="rigid",
        mode=str(mode),
        loss_type=str(loss_type),
        loss_enabled=bool(loss_enabled),
    )
    all_losses.extend(losses)
    aux.update(stats)
    loss = torch.stack(all_losses).mean() if all_losses else _zero_loss(ref)
    for metric in (
        "damage_pos_ratio",
        "damage_norm_mean",
        "cos_damage_pos_ratio",
        "cos_damage_mean",
        "cos_damage_loss",
        "projection_norm_ratio",
        "delta_norm_before",
        "delta_norm_after",
        "active",
    ):
        for attr in ("means", "scales", "opacity", "sh"):
            values = [
                float(aux[key])
                for branch in ("bg", "distant", "rigid")
                for key in (f"hsp/{branch}_{metric}/{attr}",)
                if key in aux
            ]
            if values:
                aux[f"hsp/{metric}/{attr}"] = float(sum(values) / len(values))
            active_values = [
                float(aux[key])
                for branch in ("bg", "distant", "rigid")
                if float(aux.get(f"hsp/{branch}_active/{attr}", 0.0)) > 0.0
                for key in (f"hsp/{branch}_{metric}/{attr}",)
                if key in aux
            ]
            if active_values:
                aux[f"hsp/nonzero_{metric}/{attr}"] = float(sum(active_values) / len(active_values))
    active_losses = [
        float(aux[key])
        for branch in ("bg", "distant", "rigid")
        for attr in ("means", "scales", "opacity", "sh")
        if float(aux.get(f"hsp/{branch}_active/{attr}", 0.0)) > 0.0
        for key in (f"hsp/{branch}_cos_damage_loss/{attr}",)
        if key in aux
    ]
    active_means = [
        float(aux[key])
        for branch in ("bg", "distant", "rigid")
        for attr in ("means", "scales", "opacity", "sh")
        if float(aux.get(f"hsp/{branch}_active/{attr}", 0.0)) > 0.0
        for key in (f"hsp/{branch}_cos_damage_mean/{attr}",)
        if key in aux
    ]
    active_ratios = [
        float(aux[key])
        for branch in ("bg", "distant", "rigid")
        for attr in ("means", "scales", "opacity", "sh")
        if float(aux.get(f"hsp/{branch}_active/{attr}", 0.0)) > 0.0
        for key in (f"hsp/{branch}_cos_damage_pos_ratio/{attr}",)
        if key in aux
    ]
    aux["hsp/cos_damage_loss"] = float(sum(active_losses) / len(active_losses)) if active_losses else 0.0
    aux["hsp/cos_damage_mean"] = float(sum(active_means) / len(active_means)) if active_means else 0.0
    aux["hsp/cos_damage_pos_ratio"] = float(sum(active_ratios) / len(active_ratios)) if active_ratios else 0.0
    aux["hsp/active_attr_count"] = float(len(active_losses))
    return DeltaPack(bg=bg or delta_event.bg, distant=distant, rigid=rigid, aux=delta_event.aux), loss, aux


class IForwardHistorySafeProjection(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg or {}
        self.mode = str(cfg_get(self.cfg, "mode", "damage_loss_only"))
        if self.mode not in {"damage_loss_only", "project_delta"}:
            raise ValueError("history_safe_projection.mode must be damage_loss_only or project_delta")
        probe_cfg = cfg_get(self.cfg, "probe", {}) or {}
        self.probe_frequency = str(cfg_get(probe_cfg, "frequency", "block_enter"))
        self.reuse_within_block = bool(cfg_get(probe_cfg, "reuse_within_block", True))
        self.frames_per_block = int(cfg_get(probe_cfg, "frames_per_block", 1))
        self.cams_per_frame = int(cfg_get(probe_cfg, "cams_per_frame", 1))
        self.mask_policy = str(cfg_get(probe_cfg, "mask_policy", "non_sky_non_egocar"))
        attrs_cfg = cfg_get(self.cfg, "attrs", {}) or {}
        self.attrs = {
            "means": _bool_cfg(attrs_cfg, "means", True),
            "scales": _bool_cfg(attrs_cfg, "scales", True),
            "opacity": _bool_cfg(attrs_cfg, "opacity", True),
            "sh": _bool_cfg(attrs_cfg, "sh", True),
            "quat": False,
            "hidden": False,
        }
        proj_cfg = cfg_get(self.cfg, "projection", {}) or {}
        self.strength_cfg = cfg_get(proj_cfg, "strength", {}) or {}
        scale_cfg = cfg_get(proj_cfg, "attr_strength_scale", {}) or {}
        self.attr_strength_scale = {
            "means": _float_cfg(scale_cfg, "means", 1.0),
            "scales": _float_cfg(scale_cfg, "scales", 1.0),
            "opacity": _float_cfg(scale_cfg, "opacity", 1.0),
            "sh": _float_cfg(scale_cfg, "sh", 1.0),
        }
        tau_cfg = cfg_get(proj_cfg, "tau_norm", {}) or {}
        self.tau_norm = {
            "means": _float_cfg(tau_cfg, "means", 0.0),
            "scales": _float_cfg(tau_cfg, "scales", 0.0),
            "opacity": _float_cfg(tau_cfg, "opacity", 0.0),
            "sh": _float_cfg(tau_cfg, "sh", 0.0),
        }
        self.eps = float(cfg_get(proj_cfg, "eps", 1.0e-8))
        dmg_cfg = cfg_get(self.cfg, "damage_loss", {}) or {}
        self.damage_loss_enabled = bool(cfg_get(dmg_cfg, "enable", True))
        self.damage_loss_type = str(cfg_get(dmg_cfg, "type", "cosine_conflict"))
        tau_cos_cfg = cfg_get(dmg_cfg, "tau_cos", {}) or {}
        if isinstance(tau_cos_cfg, (int, float)):
            self.tau_cos = {attr: float(tau_cos_cfg) for attr in ("means", "scales", "opacity", "sh")}
        else:
            self.tau_cos = {
                "means": _float_cfg(tau_cos_cfg, "means", 0.0),
                "scales": _float_cfg(tau_cos_cfg, "scales", 0.0),
                "opacity": _float_cfg(tau_cos_cfg, "opacity", 0.0),
                "sh": _float_cfg(tau_cos_cfg, "sh", 0.0),
            }
        weights_cfg = cfg_get(dmg_cfg, "attr_weights", {}) or {}
        self.attr_weights = {
            "means": _float_cfg(weights_cfg, "means", 1.0),
            "scales": _float_cfg(weights_cfg, "scales", 0.5),
            "opacity": _float_cfg(weights_cfg, "opacity", 0.7),
            "sh": _float_cfg(weights_cfg, "sh", 0.7),
        }

    def _compute_grad_full(
        self,
        *,
        local_state: LocalGSState,
        resolved: Any,
        batch: Dict[str, Any],
        step: Any,
        bridge: Any,
    ) -> Tuple[Optional[HistoryGradPack], torch.Tensor, Dict[str, float]]:
        ref = local_state.bg.means
        probe_indices = select_history_probe_indices(
            resolved=resolved,
            step=step,
            frames_per_block=int(self.frames_per_block),
            cams_per_frame=int(self.cams_per_frame),
        )
        if not probe_indices:
            return None, _zero_loss(ref), {
                "hsp/skipped_no_history": 1.0,
                "hsp/probe_num_refs": 0.0,
            }
        probe_state, tensor_items = make_probe_local_state(local_state, attrs=self.attrs)
        if not tensor_items:
            return None, _zero_loss(ref), {
                "hsp/skipped_no_attrs": 1.0,
                "hsp/probe_num_refs": float(len(probe_indices)),
            }
        probe_tensors = [item[2] for item in tensor_items]
        with torch.enable_grad():
            if hasattr(bridge, "history_probe_loss"):
                probe_loss, probe_stats = bridge.history_probe_loss(
                    local_state=probe_state,
                    batch=batch,
                    target_indices=list(probe_indices),
                    mask_policy=str(self.mask_policy),
                )
            else:
                probe_loss, probe_stats = bridge.render_loss(
                    local_state=probe_state,
                    batch=batch,
                    target_indices=list(probe_indices),
                    mask_policy=str(self.mask_policy),
                )
            if bool(getattr(probe_loss, "requires_grad", False)):
                grads = torch.autograd.grad(
                    probe_loss,
                    probe_tensors,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )
            else:
                grads = tuple(None for _ in probe_tensors)
        grad_full = grad_state_to_grad_pack(
            probe_state=probe_state,
            tensor_items=tensor_items,
            grads=grads,
            attrs=self.attrs,
        )
        aux = {
            "hsp/skipped_no_history": 0.0,
            "hsp/probe_num_refs": float(len(probe_indices)),
            "hsp/probe_loss": float(probe_loss.detach().item()) if probe_loss.numel() else 0.0,
        }
        if isinstance(probe_stats, dict) and probe_stats.get("psnr") is not None:
            aux["hsp/probe_psnr"] = float(probe_stats["psnr"])
        return grad_full, probe_loss.detach(), aux

    def forward(
        self,
        *,
        local_state: LocalGSState,
        event: EventPack,
        delta_event: DeltaPack,
        resolved: Any,
        batch: Dict[str, Any],
        step: Any,
        step_context: Any,
        history_ema: Any,
        bridge: Any,
        probe_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DeltaPack, Dict[str, float], torch.Tensor]:
        del history_ema
        ref = local_state.bg.means
        aux: Dict[str, float] = {
            "hsp/enabled": 1.0,
            "hsp/skipped_nonfinite": 0.0,
        }
        cache = probe_cache if probe_cache is not None else {}
        is_block_enter = bool(getattr(step, "is_block_enter", False)) or int(getattr(step, "repeat_idx", 0)) == 0
        use_cache = bool(self.reuse_within_block) and "grad_full" in cache and not bool(is_block_enter)
        if use_cache:
            grad_full = cache.get("grad_full")
            aux.update(dict(cache.get("aux", {}) or {}))
            aux["hsp/cache_hit"] = 1.0
        else:
            grad_full, _probe_loss, probe_aux = self._compute_grad_full(
                local_state=local_state,
                resolved=resolved,
                batch=batch,
                step=step,
                bridge=bridge,
            )
            aux.update(probe_aux)
            aux["hsp/cache_hit"] = 0.0
            if bool(self.reuse_within_block):
                cache["grad_full"] = grad_full
                cache["aux"] = dict(probe_aux)
        if grad_full is None:
            return delta_event, aux, _zero_loss(ref)
        if not _grad_pack_is_finite(grad_full):
            aux["hsp/skipped_nonfinite"] = 1.0
            return delta_event, aux, _zero_loss(ref)
        grad_event = grad_pack_to_event_rows(grad_full, event=event, delta_event=delta_event)
        if not _grad_pack_is_finite(grad_event):
            aux["hsp/skipped_nonfinite"] = 1.0
            return delta_event, aux, _zero_loss(ref)
        base_strength = _schedule_value(int(getattr(step_context, "global_step", 0)), self.strength_cfg)
        strength_by_attr = {
            attr: float(base_strength) * float(self.attr_strength_scale.get(attr, 1.0))
            for attr in ("means", "scales", "opacity", "sh")
        }
        if self.mode == "damage_loss_only":
            strength_by_attr = {attr: 0.0 for attr in strength_by_attr}
        delta_safe, damage_loss, project_aux = project_delta_pack(
            delta_event,
            grad_event,
            attrs=self.attrs,
            strength_by_attr=strength_by_attr,
            tau_norm_by_attr=self.tau_norm,
            tau_cos_by_attr=self.tau_cos,
            attr_weights=self.attr_weights,
            eps=float(self.eps),
            mode=str(self.mode),
            loss_type=str(self.damage_loss_type),
            loss_enabled=bool(self.damage_loss_enabled),
        )
        aux.update(project_aux)
        aux["hsp/projection_strength"] = float(base_strength) if self.mode == "project_delta" else 0.0
        aux["hsp/damage_loss"] = float(damage_loss.detach().item()) if damage_loss.numel() else 0.0
        if not torch.isfinite(damage_loss).all():
            aux["hsp/skipped_nonfinite"] = 1.0
            return delta_event, aux, _zero_loss(ref)
        for branch in (delta_safe.bg, delta_safe.distant, delta_safe.rigid):
            if branch is None:
                continue
            for value in (branch.means, branch.scales_log, branch.opacity_logit, branch.sh):
                if not bool(torch.isfinite(value).all().item()):
                    aux["hsp/skipped_nonfinite"] = 1.0
                    return delta_event, aux, _zero_loss(ref)
        return delta_safe, aux, damage_loss


__all__ = [
    "HistoryGradBranch",
    "HistoryGradPack",
    "IForwardHistorySafeProjection",
    "grad_pack_to_event_rows",
    "grad_state_to_grad_pack",
    "make_probe_local_state",
    "project_attr",
    "project_delta_pack",
    "select_history_probe_indices",
]
