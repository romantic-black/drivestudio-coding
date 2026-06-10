from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from models.streetforward.math_utils import _axis_angle_to_quat, _normalize_quat, _quat_multiply
from models.streetforward.stage6_0 import DeltaPack, EventPack, LocalGSState
from models.streetforward.stage6_0.local_gs_state import LocalBranchState
from models.streetforward.stage6_0.posterior_updater import BranchDelta

from .history_gate import GATE_ATTRS, IForwardAttributeGate, IForwardGatePack
from .history_gradient_bank import GradientBankAttr, HGV2_ATTRS, HistoryGradientBank, HistoryGradientBranchBank
from .utils import cfg_get


HGV2_FEATURES_PER_ATTR = 5
HGV2_GRAD_FEATURE_DIM = len(GATE_ATTRS) * HGV2_FEATURES_PER_ATTR


@dataclass
class HistoryGateV2AttrDamage:
    cos: torch.Tensor
    pos: torch.Tensor
    neg: torch.Tensor
    log_norm: torch.Tensor
    valid: torch.Tensor


@dataclass
class HistoryGateV2BranchFeatures:
    features: torch.Tensor
    damage: Dict[str, HistoryGateV2AttrDamage]


@dataclass
class HistoryGateV2FeaturePack:
    bg: HistoryGateV2BranchFeatures
    distant: Optional[HistoryGateV2BranchFeatures] = None
    rigid: Optional[HistoryGateV2BranchFeatures] = None
    aux: Optional[Dict[str, float]] = None


def _attr_enabled(attrs_cfg: Any) -> Dict[str, bool]:
    return {name: bool(cfg_get(attrs_cfg, name, True)) for name in HGV2_ATTRS}


def _select_attr(attr: GradientBankAttr, rows: Optional[torch.Tensor]) -> GradientBankAttr:
    if rows is None:
        return attr
    idx = rows.to(device=attr.direction.device, dtype=torch.long).reshape(-1)
    return GradientBankAttr(
        direction=attr.direction[idx],
        log_norm=attr.log_norm[idx],
        valid=attr.valid[idx],
    )


def _select_branch(
    branch: Optional[HistoryGradientBranchBank],
    rows: Optional[torch.Tensor],
) -> Optional[HistoryGradientBranchBank]:
    if branch is None:
        return None
    return HistoryGradientBranchBank(
        means=_select_attr(branch.means, rows),
        scales=_select_attr(branch.scales, rows),
        quat=_select_attr(branch.quat, rows),
        opacity=_select_attr(branch.opacity, rows),
        sh=_select_attr(branch.sh, rows),
    )


def gather_history_grad_bank_for_event(
    bank: HistoryGradientBank,
    *,
    event: EventPack,
) -> Tuple[HistoryGradientBranchBank, Optional[HistoryGradientBranchBank], Optional[HistoryGradientBranchBank]]:
    route = getattr(event, "route", None)
    rigid_rows = getattr(route, "S", None) if route is not None else None
    return (
        _select_branch(bank.bg, None),
        _select_branch(bank.distant, None),
        _select_branch(bank.rigid, rigid_rows),
    )


def _zero_damage(n: int, ref: torch.Tensor) -> HistoryGateV2AttrDamage:
    z = ref.new_zeros((int(n), 1))
    return HistoryGateV2AttrDamage(
        cos=z,
        pos=z,
        neg=z,
        log_norm=z,
        valid=torch.zeros((int(n), 1), device=ref.device, dtype=torch.bool),
    )


def _flatten_rows(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(int(value.shape[0]), -1)


def _damage_attr(
    bank_attr: Optional[GradientBankAttr],
    delta_attr: torch.Tensor,
    *,
    enabled: bool,
    eps: float,
    detach_delta: bool,
) -> HistoryGateV2AttrDamage:
    n = int(delta_attr.shape[0])
    if n == 0:
        return _zero_damage(0, delta_attr)
    if (not bool(enabled)) or bank_attr is None:
        return _zero_damage(n, delta_attr)
    delta = delta_attr.detach() if bool(detach_delta) else delta_attr
    direction = bank_attr.direction.to(device=delta.device, dtype=delta.dtype)
    if tuple(direction.shape) != tuple(delta.shape):
        raise ValueError(f"HGV2 grad/delta shape mismatch: grad={tuple(direction.shape)} delta={tuple(delta.shape)}")
    d = _flatten_rows(delta)
    g = _flatten_rows(direction)
    delta_norm = d.norm(dim=-1, keepdim=True)
    cos = (d * g).sum(dim=-1, keepdim=True) / delta_norm.clamp_min(float(eps))
    valid = bank_attr.valid.to(device=delta.device).reshape(n, 1) & (delta_norm > float(eps))
    cos = torch.where(valid, cos, torch.zeros_like(cos))
    pos = torch.relu(cos)
    neg = torch.relu(-cos)
    log_norm = bank_attr.log_norm.to(device=delta.device, dtype=delta.dtype)
    if log_norm.dim() == 1:
        log_norm = log_norm[:, None]
    log_norm = torch.where(valid, log_norm.reshape(n, -1)[:, :1], torch.zeros_like(cos))
    return HistoryGateV2AttrDamage(cos=cos, pos=pos, neg=neg, log_norm=log_norm, valid=valid)


def _branch_quat_delta(
    branch: LocalBranchState,
    delta: BranchDelta,
    *,
    rows: Optional[torch.Tensor],
) -> torch.Tensor:
    q_old = branch.quats if rows is None else branch.quats[rows.to(device=branch.quats.device, dtype=torch.long).reshape(-1)]
    q_delta = _axis_angle_to_quat(delta.quat_axis_angle)
    q_new = _normalize_quat(_quat_multiply(q_old, q_delta))
    return q_new - q_old


def induced_quat_delta(
    *,
    branch: LocalBranchState,
    delta: BranchDelta,
    rows: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _branch_quat_delta(branch, delta, rows=rows)


def _branch_sh_state_ref(branch: LocalBranchState) -> torch.Tensor:
    n = int(branch.sh_dc.shape[0])
    return torch.cat([branch.sh_dc[:, None, :], branch.sh_rest], dim=1).reshape(n, -1)


def _branch_features(
    *,
    branch_bank: Optional[HistoryGradientBranchBank],
    local_branch: Optional[LocalBranchState],
    delta_branch: Optional[BranchDelta],
    rows: Optional[torch.Tensor],
    attrs: Mapping[str, bool],
    eps: float,
    detach_damage: bool,
) -> Optional[HistoryGateV2BranchFeatures]:
    if local_branch is None or delta_branch is None:
        return None
    ref = delta_branch.means
    n = int(ref.shape[0])
    if branch_bank is None:
        damage = {attr: _zero_damage(n, ref) for attr in GATE_ATTRS}
    else:
        damage = {
            "means": _damage_attr(
                branch_bank.means,
                delta_branch.means,
                enabled=bool(attrs.get("means", True)),
                eps=float(eps),
                detach_delta=bool(detach_damage),
            ),
            "scales": _damage_attr(
                branch_bank.scales,
                delta_branch.scales_log,
                enabled=bool(attrs.get("scales", True)),
                eps=float(eps),
                detach_delta=bool(detach_damage),
            ),
            "quat": _damage_attr(
                branch_bank.quat,
                _branch_quat_delta(local_branch, delta_branch, rows=rows),
                enabled=bool(attrs.get("quat", True)),
                eps=float(eps),
                detach_delta=bool(detach_damage),
            ),
            "opacity": _damage_attr(
                branch_bank.opacity,
                delta_branch.opacity_logit,
                enabled=bool(attrs.get("opacity", True)),
                eps=float(eps),
                detach_delta=bool(detach_damage),
            ),
            "sh": _damage_attr(
                branch_bank.sh,
                delta_branch.sh,
                enabled=bool(attrs.get("sh", True)),
                eps=float(eps),
                detach_delta=bool(detach_damage),
            ),
        }
    features = torch.cat(
        [
            part
            for attr in GATE_ATTRS
            for part in (
                damage[attr].cos,
                damage[attr].pos,
                damage[attr].neg,
                damage[attr].log_norm,
                damage[attr].valid.to(dtype=ref.dtype),
            )
        ],
        dim=-1,
    )
    if int(features.shape[-1]) != int(HGV2_GRAD_FEATURE_DIM):
        raise RuntimeError(f"HGV2 feature dim mismatch: expected {HGV2_GRAD_FEATURE_DIM}, got {features.shape[-1]}")
    if bool(detach_damage):
        features = features.detach()
        damage = {
            attr: HistoryGateV2AttrDamage(
                cos=item.cos.detach(),
                pos=item.pos.detach(),
                neg=item.neg.detach(),
                log_norm=item.log_norm.detach(),
                valid=item.valid.detach(),
            )
            for attr, item in damage.items()
        }
    return HistoryGateV2BranchFeatures(features=features, damage=damage)


def compute_history_gate_v2_features(
    *,
    bank: Optional[HistoryGradientBank],
    event: EventPack,
    delta_event: DeltaPack,
    local_state: LocalGSState,
    cfg: Any,
) -> Optional[HistoryGateV2FeaturePack]:
    if bank is None or not bool(getattr(bank, "valid", False)):
        return None
    attrs = _attr_enabled(cfg_get(cfg, "attrs", {}) or {})
    features_cfg = cfg_get(cfg, "features", {}) or {}
    aux_cfg = cfg_get(cfg, "auxiliary_loss", {}) or {}
    eps = float(cfg_get(features_cfg, "eps", 1.0e-8))
    detach_damage = bool(cfg_get(aux_cfg, "detach_damage", True))
    bank_bg, bank_distant, bank_rigid = gather_history_grad_bank_for_event(bank, event=event)
    route = getattr(event, "route", None)
    rigid_rows = getattr(route, "S", None) if route is not None else None
    bg = _branch_features(
        branch_bank=bank_bg,
        local_branch=local_state.bg,
        delta_branch=delta_event.bg,
        rows=None,
        attrs=attrs,
        eps=eps,
        detach_damage=detach_damage,
    )
    if bg is None:
        raise RuntimeError("HGV2 requires bg branch features.")
    distant = _branch_features(
        branch_bank=bank_distant,
        local_branch=local_state.distant,
        delta_branch=delta_event.distant,
        rows=None,
        attrs=attrs,
        eps=eps,
        detach_damage=detach_damage,
    )
    rigid = _branch_features(
        branch_bank=bank_rigid,
        local_branch=local_state.rigid,
        delta_branch=delta_event.rigid,
        rows=rigid_rows,
        attrs=attrs,
        eps=eps,
        detach_damage=detach_damage,
    )
    return HistoryGateV2FeaturePack(bg=bg, distant=distant, rigid=rigid, aux=history_gate_v2_feature_stats(bg, distant, rigid))


def _valid_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(device=value.device, dtype=value.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def _feature_attr_stats(
    branches: Tuple[Optional[HistoryGateV2BranchFeatures], ...],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for attr in GATE_ATTRS:
        pos_values = []
        valid_counts = []
        for branch in branches:
            if branch is None:
                continue
            dmg = branch.damage[attr]
            valid = dmg.valid.to(dtype=torch.float32)
            pos_values.append((dmg.pos.detach() > 0).to(dtype=torch.float32) * valid)
            valid_counts.append(valid)
        if not valid_counts:
            out[f"hgv2/damage_pos_ratio/{attr}"] = 0.0
            continue
        numerator = sum(v.sum() for v in pos_values)
        denom = sum(v.sum() for v in valid_counts).clamp_min(1.0)
        out[f"hgv2/damage_pos_ratio/{attr}"] = float((numerator / denom).detach().item())
    return out


def history_gate_v2_feature_stats(
    bg: HistoryGateV2BranchFeatures,
    distant: Optional[HistoryGateV2BranchFeatures],
    rigid: Optional[HistoryGateV2BranchFeatures],
) -> Dict[str, float]:
    return _feature_attr_stats((bg, distant, rigid))


def _gate_attr(gate: IForwardAttributeGate, attr: str) -> torch.Tensor:
    return getattr(gate, attr)


def _branch_aux_loss(
    *,
    branch_features: Optional[HistoryGateV2BranchFeatures],
    branch_gate: Optional[IForwardAttributeGate],
    tau_cos: float,
    tau_safe: float,
    attr_weights: Mapping[str, float],
    close_weight: float,
    safe_open_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if branch_features is None or branch_gate is None:
        return torch.tensor(0.0), {}
    ref = branch_features.features
    losses = []
    aux: Dict[str, float] = {}
    for attr in GATE_ATTRS:
        dmg = branch_features.damage[attr]
        gate = _gate_attr(branch_gate, attr)
        harm = torch.relu(dmg.cos - float(tau_cos)).detach()
        safe = torch.relu(-dmg.cos - float(tau_safe)).detach()
        valid = dmg.valid.to(device=gate.device)
        close = _valid_mean(harm.to(device=gate.device, dtype=gate.dtype) * gate, valid)
        open_loss = _valid_mean(safe.to(device=gate.device, dtype=gate.dtype) * (1.0 - gate), valid)
        weight = float(attr_weights.get(attr, 1.0))
        losses.append(weight * (float(close_weight) * close + float(safe_open_weight) * open_loss))
        harmful_mask = valid & (harm.to(device=valid.device) > 0)
        safe_mask = valid & (safe.to(device=valid.device) > 0)
        aux[f"gate_harmful_mean/{attr}"] = (
            float(_valid_mean(gate.detach(), harmful_mask).item()) if bool(harmful_mask.any().item()) else 0.0
        )
        aux[f"gate_safe_mean/{attr}"] = (
            float(_valid_mean(gate.detach(), safe_mask).item()) if bool(safe_mask.any().item()) else 0.0
        )
    loss = torch.stack(losses).sum() if losses else ref.new_tensor(0.0)
    return loss, aux


def history_gate_v2_auxiliary_loss(
    *,
    gate: IForwardGatePack,
    features: HistoryGateV2FeaturePack,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    aux_cfg = cfg_get(cfg, "auxiliary_loss", {}) or {}
    if not bool(cfg_get(aux_cfg, "enable", True)):
        return features.bg.features.new_tensor(0.0), {}
    close_weight = float(cfg_get(aux_cfg, "close_weight", 0.02))
    safe_open_weight = float(cfg_get(aux_cfg, "safe_open_weight", 0.002))
    tau_cos = float(cfg_get(aux_cfg, "tau_cos", 0.05))
    tau_safe = float(cfg_get(aux_cfg, "tau_safe", 0.10))
    weights_cfg = cfg_get(aux_cfg, "attr_weights", {}) or {}
    attr_weights = {
        "means": float(cfg_get(weights_cfg, "means", 0.25)),
        "scales": float(cfg_get(weights_cfg, "scales", 0.25)),
        "quat": float(cfg_get(weights_cfg, "quat", 0.10)),
        "opacity": float(cfg_get(weights_cfg, "opacity", 1.0)),
        "sh": float(cfg_get(weights_cfg, "sh", 1.0)),
    }
    branch_losses = []
    merged: Dict[str, list] = {}
    for branch_features, branch_gate in (
        (features.bg, gate.bg),
        (features.distant, gate.distant),
        (features.rigid, gate.rigid),
    ):
        if branch_features is None or branch_gate is None:
            continue
        loss, branch_aux = _branch_aux_loss(
            branch_features=branch_features,
            branch_gate=branch_gate,
            tau_cos=tau_cos,
            tau_safe=tau_safe,
            attr_weights=attr_weights,
            close_weight=close_weight,
            safe_open_weight=safe_open_weight,
        )
        if torch.is_tensor(loss):
            branch_losses.append(loss)
        for key, value in branch_aux.items():
            merged.setdefault(str(key), []).append(float(value))
    total = torch.stack(branch_losses).mean() if branch_losses else features.bg.features.new_tensor(0.0)
    aux = {f"hgv2/{key}": float(sum(values) / len(values)) for key, values in merged.items() if values}
    aux["hgv2/loss_gate_aux"] = float(total.detach().item()) if total.numel() else 0.0
    return total, aux


__all__ = [
    "HGV2_GRAD_FEATURE_DIM",
    "HistoryGateV2AttrDamage",
    "HistoryGateV2BranchFeatures",
    "HistoryGateV2FeaturePack",
    "compute_history_gate_v2_features",
    "gather_history_grad_bank_for_event",
    "history_gate_v2_auxiliary_loss",
    "history_gate_v2_feature_stats",
    "induced_quat_delta",
]
