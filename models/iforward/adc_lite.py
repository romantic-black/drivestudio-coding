from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from models.streetforward.stage6_0 import LocalGSState
from models.streetforward.stage6_0.local_gs_state import LocalBranchState
from models.streetforward.stage6_0.posterior_updater import BranchDelta

from .gru_memory import IForwardGRUBranchState, IForwardGRUMemoryState
from .history_ema import IForwardHistoryBranchEMA, IForwardHistoryEMAState
from .history_gradient_bank import GradientBankAttr, HistoryGradientBank, HistoryGradientBranchBank
from .utils import cfg_get


ADC_STAT_PREFIX = "adc_lite"


def _optional_detach_clone(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if value is None else value.detach().clone()


def _optional_to(
    value: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    return None if value is None else value.to(device=device, dtype=dtype or value.dtype)


@dataclass
class IForwardADCBank:
    valid: bool
    source_rollout_id: int
    source_episode_id: int
    source_num_current_refs: int
    source_num_history_refs: int
    score: torch.Tensor
    abs_grad_current: torch.Tensor
    abs_grad_history: torch.Tensor
    scale_score: torch.Tensor
    conflict_score: torch.Tensor
    candidate_mask: torch.Tensor
    score_topk_mean: torch.Tensor
    score_p90: torch.Tensor
    score_p99: torch.Tensor
    score_type: str = "fixed_score_v1"
    score_max: Optional[torch.Tensor] = None
    score_sum: Optional[torch.Tensor] = None
    score_count: Optional[torch.Tensor] = None
    parent_gate_mean: Optional[torch.Tensor] = None
    parent_delta_demand: Optional[torch.Tensor] = None
    parent_support_mean: Optional[torch.Tensor] = None

    def detach(self) -> "IForwardADCBank":
        return IForwardADCBank(
            valid=bool(self.valid),
            source_rollout_id=int(self.source_rollout_id),
            source_episode_id=int(self.source_episode_id),
            source_num_current_refs=int(self.source_num_current_refs),
            source_num_history_refs=int(self.source_num_history_refs),
            score=self.score.detach().clone(),
            abs_grad_current=self.abs_grad_current.detach().clone(),
            abs_grad_history=self.abs_grad_history.detach().clone(),
            scale_score=self.scale_score.detach().clone(),
            conflict_score=self.conflict_score.detach().clone(),
            candidate_mask=self.candidate_mask.detach().clone(),
            score_topk_mean=self.score_topk_mean.detach().clone(),
            score_p90=self.score_p90.detach().clone(),
            score_p99=self.score_p99.detach().clone(),
            score_type=str(self.score_type),
            score_max=_optional_detach_clone(self.score_max),
            score_sum=_optional_detach_clone(self.score_sum),
            score_count=_optional_detach_clone(self.score_count),
            parent_gate_mean=_optional_detach_clone(self.parent_gate_mean),
            parent_delta_demand=_optional_detach_clone(self.parent_delta_demand),
            parent_support_mean=_optional_detach_clone(self.parent_support_mean),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "IForwardADCBank":
        out_dtype = dtype or self.score.dtype
        return IForwardADCBank(
            valid=bool(self.valid),
            source_rollout_id=int(self.source_rollout_id),
            source_episode_id=int(self.source_episode_id),
            source_num_current_refs=int(self.source_num_current_refs),
            source_num_history_refs=int(self.source_num_history_refs),
            score=self.score.to(device=device, dtype=out_dtype),
            abs_grad_current=self.abs_grad_current.to(device=device, dtype=out_dtype),
            abs_grad_history=self.abs_grad_history.to(device=device, dtype=out_dtype),
            scale_score=self.scale_score.to(device=device, dtype=out_dtype),
            conflict_score=self.conflict_score.to(device=device, dtype=out_dtype),
            candidate_mask=self.candidate_mask.to(device=device),
            score_topk_mean=self.score_topk_mean.to(device=device, dtype=out_dtype),
            score_p90=self.score_p90.to(device=device, dtype=out_dtype),
            score_p99=self.score_p99.to(device=device, dtype=out_dtype),
            score_type=str(self.score_type),
            score_max=_optional_to(self.score_max, device=device, dtype=out_dtype),
            score_sum=_optional_to(self.score_sum, device=device, dtype=out_dtype),
            score_count=_optional_to(self.score_count, device=device, dtype=out_dtype),
            parent_gate_mean=_optional_to(self.parent_gate_mean, device=device, dtype=out_dtype),
            parent_delta_demand=_optional_to(self.parent_delta_demand, device=device, dtype=out_dtype),
            parent_support_mean=_optional_to(self.parent_support_mean, device=device, dtype=out_dtype),
        )


@dataclass
class IForwardADCStateMeta:
    original_bg_count: int
    num_bg_clones_created_episode: int = 0
    parent_index: Optional[torch.Tensor] = None
    birth_rollout_id: Optional[torch.Tensor] = None
    cooldown_until_rollout: Optional[torch.Tensor] = None

    def detach(self) -> "IForwardADCStateMeta":
        return IForwardADCStateMeta(
            original_bg_count=int(self.original_bg_count),
            num_bg_clones_created_episode=int(self.num_bg_clones_created_episode),
            parent_index=None if self.parent_index is None else self.parent_index.detach().clone(),
            birth_rollout_id=None if self.birth_rollout_id is None else self.birth_rollout_id.detach().clone(),
            cooldown_until_rollout=(
                None if self.cooldown_until_rollout is None else self.cooldown_until_rollout.detach().clone()
            ),
        )

    def to(self, *, device: torch.device) -> "IForwardADCStateMeta":
        return IForwardADCStateMeta(
            original_bg_count=int(self.original_bg_count),
            num_bg_clones_created_episode=int(self.num_bg_clones_created_episode),
            parent_index=None if self.parent_index is None else self.parent_index.to(device=device),
            birth_rollout_id=None if self.birth_rollout_id is None else self.birth_rollout_id.to(device=device),
            cooldown_until_rollout=(
                None if self.cooldown_until_rollout is None else self.cooldown_until_rollout.to(device=device)
            ),
        )


def _storage_dtype(cfg: Any, ref_dtype: torch.dtype) -> torch.dtype:
    raw = str(cfg_get(cfg_get(cfg, "bank", {}) or {}, "dtype", cfg_get(cfg, "dtype", "fp16"))).lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32", "single"}:
        return torch.float32
    if raw in {"same", "input"}:
        return ref_dtype
    raise ValueError(f"unsupported adc_lite bank dtype={raw!r}")


def _float_cfg(node: Any, key: str, default: float) -> float:
    return float(cfg_get(node, key, default))


def _bool_cfg(node: Any, key: str, default: bool) -> bool:
    return bool(cfg_get(node, key, default))


def _zero_vec(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_zeros((int(ref.shape[0]),))


def _aabb_bounds(
    *,
    ref: torch.Tensor,
    aabb_min: Optional[torch.Tensor],
    aabb_max: Optional[torch.Tensor],
    eps: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if aabb_min is None or aabb_max is None:
        return None, None
    lo = aabb_min.to(device=ref.device, dtype=ref.dtype).reshape(1, 3)
    hi = aabb_max.to(device=ref.device, dtype=ref.dtype).reshape(1, 3)
    if float(eps) > 0.0:
        room = (hi - lo) > (2.0 * float(eps))
        lo = torch.where(room, lo + float(eps), lo)
        hi = torch.where(room, hi - float(eps), hi)
    return lo, hi


def _inside_aabb_mask(means: torch.Tensor, *, aabb_min: Optional[torch.Tensor], aabb_max: Optional[torch.Tensor]) -> torch.Tensor:
    lo, hi = _aabb_bounds(ref=means, aabb_min=aabb_min, aabb_max=aabb_max)
    if lo is None or hi is None:
        return torch.ones((int(means.shape[0]),), device=means.device, dtype=torch.bool)
    return ((means >= lo) & (means <= hi)).all(dim=-1)


def _finite_percentile(values: torch.Tensor, percentile: float, eps: float) -> torch.Tensor:
    flat = values.detach().reshape(-1).to(dtype=torch.float32)
    finite = flat[torch.isfinite(flat)]
    if int(finite.numel()) == 0:
        return values.detach().new_tensor(float(eps))
    q = min(max(float(percentile) / 100.0, 0.0), 1.0)
    denom = torch.quantile(finite, q).to(device=values.device, dtype=values.dtype)
    return denom.clamp_min(float(eps))


def _normalize01(values: torch.Tensor, *, percentile: float, eps: float) -> torch.Tensor:
    if int(values.numel()) == 0:
        return values
    denom = _finite_percentile(values, float(percentile), float(eps))
    out = values / denom
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out.clamp(min=0.0, max=1.0)


def _masked_percentile(values: torch.Tensor, mask: torch.Tensor, percentile: float) -> torch.Tensor:
    selected = values.detach().reshape(-1)[mask.detach().reshape(-1).to(dtype=torch.bool)]
    selected = selected[torch.isfinite(selected)]
    if int(selected.numel()) == 0:
        return values.detach().new_tensor(0.0)
    return torch.quantile(selected.to(dtype=torch.float32), min(max(float(percentile) / 100.0, 0.0), 1.0)).to(
        device=values.device,
        dtype=values.dtype,
    )


def _score_type_is_gate_suppressed(score_type: str) -> bool:
    return str(score_type).lower() in {
        "gate_suppressed_update",
        "gate_suppressed_update_v1",
        "relative_gate_suppressed_update",
        "relative_gate_suppressed_update_v1",
    }


def _percentile_from_mode(mode: Any, default: float = 50.0) -> float:
    raw = str(mode).strip().lower()
    if raw in {"median", "p50"}:
        return 50.0
    if raw.startswith("p"):
        raw = raw[1:]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    selected = values.detach().reshape(-1)[mask.detach().reshape(-1).to(dtype=torch.bool)]
    selected = selected[torch.isfinite(selected)]
    if int(selected.numel()) == 0 or int(k) <= 0:
        return values.detach().new_tensor(0.0)
    topk = torch.topk(selected, k=min(int(k), int(selected.numel())), largest=True).values
    return topk.mean().to(device=values.device, dtype=values.dtype)


def _collect_bg_params(
    branch: LocalBranchState,
    *,
    params: List[torch.Tensor],
    names: List[str],
) -> None:
    for name, tensor in (
        ("means", branch.means),
        ("scales", branch.scales_log),
        ("opacity", branch.opacity_logit),
        ("sh_dc", branch.sh_dc),
        ("sh_rest", branch.sh_rest),
    ):
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and bool(tensor.requires_grad):
            params.append(tensor)
            names.append(str(name))


def _grad_map_from_loss(
    loss: Optional[torch.Tensor],
    *,
    params: List[torch.Tensor],
    names: List[str],
) -> Dict[str, Optional[torch.Tensor]]:
    if not params or loss is None or not torch.is_tensor(loss) or not bool(getattr(loss, "requires_grad", False)):
        return {name: None for name in names}
    if int(loss.numel()) != 1:
        loss = loss.reshape(-1).mean()
    if not bool(torch.isfinite(loss.detach()).all().item()):
        return {name: None for name in names}
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return {name: grad for name, grad in zip(names, grads)}


def _grad_or_zero(
    grad_map: Mapping[str, Optional[torch.Tensor]],
    name: str,
    ref: torch.Tensor,
) -> torch.Tensor:
    grad = grad_map.get(name)
    if grad is None:
        return torch.zeros_like(ref)
    return grad.detach().to(device=ref.device, dtype=ref.dtype)


def _combine_sh_grad(branch: LocalBranchState, grad_map: Mapping[str, Optional[torch.Tensor]]) -> torch.Tensor:
    sh_dc = _grad_or_zero(grad_map, "sh_dc", branch.sh_dc)
    sh_rest = _grad_or_zero(grad_map, "sh_rest", branch.sh_rest)
    n = int(branch.sh_dc.shape[0])
    return torch.cat([sh_dc[:, None, :], sh_rest], dim=1).reshape(n, -1)


def _abs_grad_score(
    branch: LocalBranchState,
    grad_map: Mapping[str, Optional[torch.Tensor]],
    *,
    attr_weights: Mapping[str, float],
) -> torch.Tensor:
    n = int(branch.means.shape[0])
    if n == 0:
        return _zero_vec(branch.means)
    means = _grad_or_zero(grad_map, "means", branch.means).reshape(n, -1).norm(dim=-1)
    scales = _grad_or_zero(grad_map, "scales", branch.scales_log).reshape(n, -1).norm(dim=-1)
    opacity = _grad_or_zero(grad_map, "opacity", branch.opacity_logit).reshape(n, -1).norm(dim=-1)
    sh = _combine_sh_grad(branch, grad_map).reshape(n, -1).abs().mean(dim=-1)
    return (
        float(attr_weights.get("means", 1.0)) * means
        + float(attr_weights.get("scales", 0.5)) * scales
        + float(attr_weights.get("opacity", 0.75)) * opacity
        + float(attr_weights.get("sh", 0.75)) * sh
    )


def _conflict_vectors(
    branch: LocalBranchState,
    grad_map: Mapping[str, Optional[torch.Tensor]],
    *,
    attr_weights: Mapping[str, float],
) -> torch.Tensor:
    n = int(branch.means.shape[0])
    pieces = [
        _grad_or_zero(grad_map, "means", branch.means).reshape(n, -1) * float(attr_weights.get("means", 1.0)),
        _grad_or_zero(grad_map, "scales", branch.scales_log).reshape(n, -1) * float(attr_weights.get("scales", 0.5)),
        _grad_or_zero(grad_map, "opacity", branch.opacity_logit).reshape(n, -1) * float(attr_weights.get("opacity", 0.75)),
        _combine_sh_grad(branch, grad_map).reshape(n, -1) * float(attr_weights.get("sh", 0.75)),
    ]
    return torch.cat(pieces, dim=-1) if pieces else branch.means.new_zeros((n, 0))


def _build_candidate_mask(
    *,
    final_local_state: LocalGSState,
    adc_meta: Optional[IForwardADCStateMeta],
    cfg: Any,
    score: torch.Tensor,
    rollout_id: int,
    aabb_min: Optional[torch.Tensor] = None,
    aabb_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    branch = final_local_state.bg
    n = int(branch.means.shape[0])
    candidate_cfg = cfg_get(cfg, "candidate", {}) or {}
    mask = torch.ones((n,), device=branch.means.device, dtype=torch.bool)
    if _bool_cfg(candidate_cfg, "exclude_clones_as_parent", True):
        original_count = int(adc_meta.original_bg_count) if adc_meta is not None else n
        mask = mask & (torch.arange(n, device=branch.means.device) < int(original_count))
    if adc_meta is not None and adc_meta.cooldown_until_rollout is not None:
        cooldown = adc_meta.cooldown_until_rollout.to(device=branch.means.device)
        if int(cooldown.numel()) == n:
            mask = mask & (cooldown.reshape(-1) <= int(rollout_id))
    alpha = torch.sigmoid(branch.opacity_logit.detach().reshape(n, -1).mean(dim=-1))
    scale_raw = torch.exp(branch.scales_log.detach()).reshape(n, -1).amax(dim=-1)
    mask = mask & torch.isfinite(score.detach()) & (alpha > _float_cfg(candidate_cfg, "alpha_min", 0.005))
    mask = mask & (scale_raw > _float_cfg(candidate_cfg, "scale_min", 1.0e-4))
    min_score = _float_cfg(candidate_cfg, "min_score", 0.0)
    mask = mask & (score.detach() >= float(min_score))
    if aabb_min is not None and aabb_max is not None:
        mask = mask & _inside_aabb_mask(branch.means.detach(), aabb_min=aabb_min, aabb_max=aabb_max)
        if _bool_cfg(candidate_cfg, "exclude_boundary_parents", True):
            clone_cfg = cfg_get(cfg, "clone", {}) or {}
            jitter_scale = abs(_float_cfg(clone_cfg, "mean_jitter_std_scale", 0.05))
            margin_eps = _float_cfg(candidate_cfg, "boundary_margin_eps", 1.0e-4)
            lo = aabb_min.to(device=branch.means.device, dtype=branch.means.dtype).reshape(1, 3)
            hi = aabb_max.to(device=branch.means.device, dtype=branch.means.dtype).reshape(1, 3)
            jitter_margin = torch.exp(branch.scales_log.detach()).mean(dim=-1, keepdim=True) * float(jitter_scale)
            margin = jitter_margin + float(margin_eps)
            safe_parent = ((branch.means.detach() > (lo + margin)) & (branch.means.detach() < (hi - margin))).all(dim=-1)
            mask = mask & safe_parent
    return mask


def _attr_norm(x: torch.Tensor) -> torch.Tensor:
    if int(x.numel()) == 0:
        return x.new_zeros((int(x.shape[0]),), dtype=torch.float32)
    flat = x.detach().reshape(int(x.shape[0]), -1).to(dtype=torch.float32)
    return torch.sqrt(flat.square().mean(dim=-1).clamp_min(0.0))


def _gate_column(gate: Any, raw_name: str, effective_name: str, ref: torch.Tensor) -> torch.Tensor:
    value = getattr(gate, raw_name, None)
    if value is None:
        value = getattr(gate, effective_name)
    out = value.detach().to(device=ref.device, dtype=torch.float32)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if out.dim() != 2 or int(out.shape[0]) != int(ref.shape[0]):
        raise ValueError(
            f"gate-suppressed ADC gate row mismatch for {effective_name}: "
            f"got {tuple(out.shape)}, expected rows={int(ref.shape[0])}"
        )
    if int(out.shape[1]) != 1:
        out = out.reshape(int(ref.shape[0]), -1).mean(dim=-1, keepdim=True)
    return out


def _optional_bool_column(value: Optional[torch.Tensor], *, n: int, ref: torch.Tensor, default: bool) -> torch.Tensor:
    if value is None:
        return torch.full((int(n),), bool(default), device=ref.device, dtype=torch.bool)
    out = value.detach().to(device=ref.device)
    if out.dim() == 2 and int(out.shape[1]) == 1:
        out = out[:, 0]
    else:
        out = out.reshape(int(n), -1).any(dim=-1)
    if int(out.shape[0]) != int(n):
        raise ValueError(f"gate-suppressed ADC bool row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    return out.to(dtype=torch.bool)


def _optional_float_column(value: Optional[torch.Tensor], *, n: int, ref: torch.Tensor, default: float) -> torch.Tensor:
    if value is None:
        return ref.detach().new_full((int(n),), float(default), dtype=torch.float32)
    out = value.detach().to(device=ref.device, dtype=torch.float32)
    if out.dim() == 2 and int(out.shape[1]) == 1:
        out = out[:, 0]
    else:
        out = out.reshape(int(n), -1).mean(dim=-1)
    if int(out.shape[0]) != int(n):
        raise ValueError(f"gate-suppressed ADC float row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    return out


def _masked_percentile_clamped(values: torch.Tensor, mask: torch.Tensor, percentile: float, eps: float) -> torch.Tensor:
    selected = values.detach().reshape(-1)[mask.detach().reshape(-1).to(dtype=torch.bool)]
    selected = selected[torch.isfinite(selected)]
    if int(selected.numel()) == 0:
        return values.detach().new_tensor(float(eps), dtype=torch.float32)
    q = min(max(float(percentile) / 100.0, 0.0), 1.0)
    return torch.quantile(selected.to(dtype=torch.float32), q).to(device=values.device).clamp_min(float(eps))


def _rank_percentile_for_indices(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    selected_idx: torch.Tensor,
) -> torch.Tensor:
    candidate_scores = scores.detach().reshape(-1)[candidate_mask.detach().reshape(-1).to(dtype=torch.bool)]
    candidate_scores = candidate_scores[torch.isfinite(candidate_scores)]
    selected = scores.detach().reshape(-1)[selected_idx.detach().reshape(-1).to(dtype=torch.long)]
    selected = selected[torch.isfinite(selected)]
    if int(candidate_scores.numel()) == 0 or int(selected.numel()) == 0:
        return scores.detach().new_tensor(0.0)
    return torch.stack([(candidate_scores <= value).to(dtype=torch.float32).mean() for value in selected]).mean()


def compute_gate_suppressed_score(
    *,
    delta_bg: BranchDelta,
    gate_bg: Any,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-bg-row suppression score and diagnostics.

    The score uses the learned pre-mask gate. `mask_update` is only a hard
    validity filter so unsupported rows are not counted as history suppression.
    """

    ref = delta_bg.means
    n = int(ref.shape[0])
    if n == 0:
        z = ref.detach().new_zeros((0,), dtype=torch.float32)
        return z, z, z, z, torch.zeros((0,), device=ref.device, dtype=torch.bool)

    score_cfg = cfg_get(cfg, "score", {}) or {}
    score_type = str(cfg_get(score_cfg, "type", "gate_suppressed_update")).lower()
    norm_cfg = cfg_get(score_cfg, "attr_normalize", {}) or {}
    percentile = _float_cfg(norm_cfg, "percentile", 95.0)
    eps = _float_cfg(norm_cfg, "eps", 1.0e-8)
    score_clip = _float_cfg(score_cfg, "score_clip", 10.0)

    mask_update = _optional_bool_column(getattr(gate_bg, "mask_update", None), n=n, ref=ref, default=True)
    support = _optional_float_column(getattr(gate_bg, "support_now", None), n=n, ref=ref, default=1.0)

    attr_specs = (
        ("means", delta_bg.means, _gate_column(gate_bg, "raw_means", "means", ref)),
        ("scales", delta_bg.scales_log, _gate_column(gate_bg, "raw_scales", "scales", ref)),
        ("quat", delta_bg.quat_axis_angle, _gate_column(gate_bg, "raw_quat", "quat", ref)),
        ("opacity", delta_bg.opacity_logit, _gate_column(gate_bg, "raw_opacity", "opacity", ref)),
        ("sh", delta_bg.sh, _gate_column(gate_bg, "raw_sh", "sh", ref)),
    )
    supp_normed: List[torch.Tensor] = []
    demand_normed: List[torch.Tensor] = []
    demand_norms: List[torch.Tensor] = []
    gate_values: List[torch.Tensor] = []
    for _, delta_attr, gate_attr in attr_specs:
        delta_f = delta_attr.detach().to(dtype=torch.float32)
        gate_f = gate_attr.to(device=delta_attr.device, dtype=torch.float32)
        supp = _attr_norm((1.0 - gate_f).to(device=delta_attr.device, dtype=delta_f.dtype) * delta_f)
        demand = _attr_norm(delta_f)
        denom = _masked_percentile_clamped(demand, mask_update, percentile, eps)
        normed = (supp / denom).clamp(min=0.0, max=float(score_clip))
        normed = torch.where(torch.isfinite(normed), normed, torch.zeros_like(normed))
        demand_norm = (demand / denom).clamp(min=0.0, max=float(score_clip))
        demand_norm = torch.where(torch.isfinite(demand_norm), demand_norm, torch.zeros_like(demand_norm))
        supp_normed.append(normed)
        demand_normed.append(demand_norm)
        demand_norms.append(torch.where(torch.isfinite(demand), demand, torch.zeros_like(demand)))
        gate_values.append(gate_f.reshape(n, -1).mean(dim=-1).to(device=ref.device))

    score_stack = torch.stack(supp_normed, dim=0)
    demand_norm_stack = torch.stack(demand_normed, dim=0)
    demand_stack = torch.stack(demand_norms, dim=0)
    gate_stack = torch.stack(gate_values, dim=0)
    if score_type in {"relative_gate_suppressed_update", "relative_gate_suppressed_update_v1"}:
        gate_mean_tmp = gate_stack.mean(dim=0)
        demand_norm = torch.sqrt(demand_norm_stack.square().mean(dim=0).clamp_min(0.0))
        rel_cfg = cfg_get(score_cfg, "gate_ref", {}) or {}
        gate_ref_pct = _percentile_from_mode(
            cfg_get(rel_cfg, "mode", cfg_get(rel_cfg, "percentile", "median")),
            default=50.0,
        )
        gate_ref = _masked_percentile_clamped(gate_mean_tmp, mask_update, gate_ref_pct, eps)
        relative_gate_suppression = (gate_ref - gate_mean_tmp).clamp_min(0.0) / gate_ref.clamp_min(float(eps))
        score = (demand_norm * relative_gate_suppression).clamp(min=0.0, max=float(score_clip))
        score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
    else:
        score = torch.sqrt(score_stack.square().mean(dim=0).clamp_min(0.0))
    delta_demand = torch.sqrt(demand_stack.square().mean(dim=0).clamp_min(0.0))
    gate_mean = gate_stack.mean(dim=0)
    valid = mask_update & torch.isfinite(score) & torch.isfinite(delta_demand) & torch.isfinite(gate_mean)
    score = torch.where(valid, score, torch.zeros_like(score))
    delta_demand = torch.where(valid, delta_demand, torch.zeros_like(delta_demand))
    gate_mean = torch.where(valid, gate_mean, torch.zeros_like(gate_mean))
    support = torch.where(valid, support, torch.zeros_like(support))
    return score, gate_mean, delta_demand, support, valid


class GateSuppressedADCAccumulator:
    def __init__(self, *, num_bg: int, device: torch.device) -> None:
        n = int(num_bg)
        self.score_sum = torch.zeros((n,), device=device, dtype=torch.float32)
        self.score_max = torch.zeros((n,), device=device, dtype=torch.float32)
        self.score_count = torch.zeros((n,), device=device, dtype=torch.float32)
        self.gate_sum = torch.zeros((n,), device=device, dtype=torch.float32)
        self.delta_demand_sum = torch.zeros((n,), device=device, dtype=torch.float32)
        self.support_sum = torch.zeros((n,), device=device, dtype=torch.float32)

    @classmethod
    def from_local_state(cls, local_state: LocalGSState) -> "GateSuppressedADCAccumulator":
        return cls(num_bg=int(local_state.bg.means.shape[0]), device=local_state.bg.means.device)

    def accumulate(
        self,
        *,
        score: torch.Tensor,
        gate_mean: torch.Tensor,
        delta_demand: torch.Tensor,
        support: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, float]:
        n = int(self.score_sum.numel())
        for name, value in (
            ("score", score),
            ("gate_mean", gate_mean),
            ("delta_demand", delta_demand),
            ("support", support),
            ("mask", mask),
        ):
            if int(value.reshape(-1).numel()) != n:
                raise ValueError(f"gate-suppressed ADC {name} rows mismatch: got {int(value.numel())}, expected {n}")
        valid = mask.detach().reshape(-1).to(device=self.score_sum.device, dtype=torch.bool)
        score_f = score.detach().reshape(-1).to(device=self.score_sum.device, dtype=torch.float32)
        gate_f = gate_mean.detach().reshape(-1).to(device=self.score_sum.device, dtype=torch.float32)
        demand_f = delta_demand.detach().reshape(-1).to(device=self.score_sum.device, dtype=torch.float32)
        support_f = support.detach().reshape(-1).to(device=self.score_sum.device, dtype=torch.float32)
        valid = valid & torch.isfinite(score_f) & torch.isfinite(gate_f) & torch.isfinite(demand_f) & torch.isfinite(support_f)
        valid_f = valid.to(dtype=torch.float32)
        score_f = torch.where(valid, score_f, torch.zeros_like(score_f))
        self.score_sum += score_f
        self.score_max = torch.maximum(self.score_max, score_f)
        self.score_count += valid_f
        self.gate_sum += torch.where(valid, gate_f, torch.zeros_like(gate_f))
        self.delta_demand_sum += torch.where(valid, demand_f, torch.zeros_like(demand_f))
        self.support_sum += torch.where(valid, support_f, torch.zeros_like(support_f))
        return {
            "adc_suppressed/step_valid_rows": float(valid_f.sum().item()),
            "adc_suppressed/step_score_mean": float(score_f[valid].mean().item()) if bool(valid.any().item()) else 0.0,
        }

    def accumulate_from_bg_delta_gate(self, *, delta_bg: BranchDelta, gate_bg: Any, cfg: Any) -> Dict[str, float]:
        score, gate_mean, delta_demand, support, mask = compute_gate_suppressed_score(
            delta_bg=delta_bg,
            gate_bg=gate_bg,
            cfg=cfg,
        )
        return self.accumulate(
            score=score,
            gate_mean=gate_mean,
            delta_demand=delta_demand,
            support=support,
            mask=mask,
        )

    def averaged(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denom = self.score_count.clamp_min(1.0)
        return self.gate_sum / denom, self.delta_demand_sum / denom, self.support_sum / denom

    def stats(self, *, topk: int = 0) -> Dict[str, float]:
        valid = self.score_count > 0
        score = self.score_max
        gate_mean, delta_demand, support = self.averaged()
        k = int(topk) if int(topk) > 0 else max(1, min(1000, int(valid.detach().to(dtype=torch.long).sum().item())))
        return {
            "adc_suppressed/accum_valid_rows": float(valid.detach().to(dtype=torch.float32).sum().item()),
            "adc_suppressed/score_mean": float(score.detach()[valid].mean().item()) if bool(valid.any().item()) else 0.0,
            "adc_suppressed/score_topk_mean": float(_masked_topk_mean(score, valid, k).detach().item()),
            "adc_suppressed/score_p90": float(_masked_percentile(score, valid, 90.0).detach().item()),
            "adc_suppressed/score_p99": float(_masked_percentile(score, valid, 99.0).detach().item()),
            "adc_suppressed/gate_mean": float(gate_mean.detach()[valid].mean().item()) if bool(valid.any().item()) else 0.0,
            "adc_suppressed/delta_demand_mean": (
                float(delta_demand.detach()[valid].mean().item()) if bool(valid.any().item()) else 0.0
            ),
            "adc_suppressed/support_mean": float(support.detach()[valid].mean().item()) if bool(valid.any().item()) else 0.0,
            "adc_suppressed/all_gate_mean": (
                float(gate_mean.detach()[valid].mean().item()) if bool(valid.any().item()) else 0.0
            ),
            "adc_suppressed/all_delta_demand_mean": (
                float(delta_demand.detach()[valid].mean().item()) if bool(valid.any().item()) else 0.0
            ),
            "adc_suppressed/all_support_mean": (
                float(support.detach()[valid].mean().item()) if bool(valid.any().item()) else 0.0
            ),
        }


def build_gate_suppressed_adc_bank(
    *,
    accumulator: Optional[GateSuppressedADCAccumulator],
    final_local_state: LocalGSState,
    history_ema: Optional[IForwardHistoryEMAState],
    cfg: Mapping[str, Any],
    rollout_id: int,
    episode_id: int,
    num_current_refs: int,
    num_history_refs: int,
    adc_meta: Optional[IForwardADCStateMeta] = None,
    aabb_min: Optional[torch.Tensor] = None,
    aabb_max: Optional[torch.Tensor] = None,
    diagnostics: Optional[Dict[str, float]] = None,
) -> Optional[IForwardADCBank]:
    diagnostics = diagnostics if diagnostics is not None else {}
    if accumulator is None or not bool(cfg_get(cfg, "enable", False)):
        return None
    branch = final_local_state.bg
    n = int(branch.means.shape[0])
    if n == 0 or int(accumulator.score_max.numel()) != n:
        diagnostics[f"{ADC_STAT_PREFIX}/bank_shape_mismatch"] = 1.0
        return None
    if int(num_current_refs) <= 0:
        return None
    candidate_cfg = cfg_get(cfg, "candidate", {}) or {}
    require_history = _bool_cfg(candidate_cfg, "require_history", True)
    if require_history and int(num_history_refs) <= 0:
        return None

    score = accumulator.score_max.detach().to(device=branch.means.device, dtype=torch.float32)
    score_sum = accumulator.score_sum.detach().to(device=branch.means.device, dtype=torch.float32)
    score_count = accumulator.score_count.detach().to(device=branch.means.device, dtype=torch.float32)
    gate_mean, delta_demand, support_mean = accumulator.averaged()
    gate_mean = gate_mean.detach().to(device=branch.means.device, dtype=torch.float32)
    delta_demand = delta_demand.detach().to(device=branch.means.device, dtype=torch.float32)
    support_mean = support_mean.detach().to(device=branch.means.device, dtype=torch.float32)

    min_count = int(cfg_get(candidate_cfg, "min_count", 1))
    mask = torch.isfinite(score) & (score_count >= int(min_count))
    if require_history:
        initialized = None if history_ema is None else getattr(history_ema.bg, "initialized", None)
        if initialized is None or int(initialized.numel()) != n:
            return None
        init_mask = initialized.detach().to(device=branch.means.device).reshape(n, -1).mean(dim=-1) > 0.0
        mask = mask & init_mask
    if _bool_cfg(candidate_cfg, "require_support", False):
        min_support = _float_cfg(candidate_cfg, "min_support", 0.0)
        mask = mask & torch.isfinite(support_mean) & (support_mean > float(min_support))
        diagnostics["adc_suppressed/support_threshold"] = float(min_support)
        diagnostics["adc_suppressed/candidate_count_after_support_filter"] = float(
            mask.detach().to(dtype=torch.float32).sum().item()
        )

    base_mask = _build_candidate_mask(
        final_local_state=final_local_state,
        adc_meta=adc_meta,
        cfg=cfg,
        score=score,
        rollout_id=int(rollout_id),
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )
    mask = mask & base_mask
    filter_ref_mask = mask.clone()
    diagnostics["adc_suppressed/base_candidate_count"] = float(filter_ref_mask.detach().to(dtype=torch.float32).sum().item())
    if bool(filter_ref_mask.any().item()):
        gate_p20 = _masked_percentile(gate_mean, filter_ref_mask, 20.0)
        gate_p50 = _masked_percentile(gate_mean, filter_ref_mask, 50.0)
        gate_p80 = _masked_percentile(gate_mean, filter_ref_mask, 80.0)
        for prefix in ("adc", "adc_suppressed"):
            diagnostics[f"{prefix}/gate_distribution_p20"] = float(gate_p20.detach().item())
            diagnostics[f"{prefix}/gate_distribution_p50"] = float(gate_p50.detach().item())
            diagnostics[f"{prefix}/gate_distribution_p80"] = float(gate_p80.detach().item())
    if _bool_cfg(candidate_cfg, "require_low_gate", False) and bool(filter_ref_mask.any().item()):
        gate_pct = _float_cfg(candidate_cfg, "gate_percentile_max", 40.0)
        gate_threshold = _masked_percentile(gate_mean, filter_ref_mask, gate_pct)
        mask = mask & (gate_mean <= gate_threshold)
        diagnostics["adc_suppressed/gate_threshold"] = float(gate_threshold.detach().item())
        diagnostics["adc_suppressed/candidate_count_after_gate_filter"] = float(
            mask.detach().to(dtype=torch.float32).sum().item()
        )
    if _bool_cfg(candidate_cfg, "require_high_delta_demand", False) and bool(filter_ref_mask.any().item()):
        demand_pct = _float_cfg(candidate_cfg, "delta_demand_percentile_min", 60.0)
        demand_threshold = _masked_percentile(delta_demand, filter_ref_mask, demand_pct)
        mask = mask & (delta_demand >= demand_threshold)
        diagnostics["adc_suppressed/delta_demand_threshold"] = float(demand_threshold.detach().item())
        diagnostics["adc_suppressed/candidate_count_after_delta_filter"] = float(
            mask.detach().to(dtype=torch.float32).sum().item()
        )
    percentile_raw = cfg_get(candidate_cfg, "min_score_percentile", None)
    if percentile_raw is not None and bool(mask.any().item()):
        pct_threshold = _masked_percentile(score, mask, float(percentile_raw))
        mask = mask & (score >= pct_threshold)
    score = torch.where(mask, score, torch.zeros_like(score))
    if not bool(mask.any().item()):
        diagnostics["adc_suppressed/candidate_count_pre_guard"] = 0.0
        return None

    budget_cfg = cfg_get(cfg, "budget", {}) or {}
    topk = int(cfg_get(budget_cfg, "max_new_points_per_rollout", 2000))
    if _bool_cfg(candidate_cfg, "require_gate_contrast", False):
        gate_ref_cfg = cfg_get(cfg_get(cfg, "score", {}) or {}, "gate_ref", {}) or {}
        gate_ref_pct = _percentile_from_mode(
            cfg_get(candidate_cfg, "gate_ref_mode", cfg_get(gate_ref_cfg, "mode", "median")),
            default=50.0,
        )
        gate_ref = _masked_percentile(gate_mean, filter_ref_mask, gate_ref_pct)
        probe_k = min(max(1, int(topk)), int(mask.detach().to(dtype=torch.long).sum().item()))
        if probe_k > 0:
            probe_score = torch.where(mask, score, torch.full_like(score, -torch.inf))
            probe_parent = torch.topk(probe_score, k=probe_k, largest=True).indices
            probe_gate_mean = gate_mean.detach()[probe_parent].mean()
        else:
            probe_gate_mean = gate_mean.detach().new_tensor(0.0)
        gate_contrast = gate_ref - probe_gate_mean
        min_gate_contrast = _float_cfg(candidate_cfg, "min_gate_contrast", 0.0)
        diagnostics["adc/parent_gate_contrast_pre_guard"] = float(gate_contrast.detach().item())
        diagnostics["adc_suppressed/parent_gate_contrast_pre_guard"] = float(gate_contrast.detach().item())
        diagnostics["adc_suppressed/gate_ref"] = float(gate_ref.detach().item())
        if float(gate_contrast.detach().item()) < float(min_gate_contrast):
            diagnostics[f"{ADC_STAT_PREFIX}/bank_low_gate_contrast"] = 1.0
            return None
        diagnostics[f"{ADC_STAT_PREFIX}/bank_low_gate_contrast"] = 0.0
    score_topk_mean = _masked_topk_mean(score, mask, topk)
    score_p90 = _masked_percentile(score, mask, 90.0)
    score_p99 = _masked_percentile(score, mask, 99.0)
    min_score_topk_mean_raw = cfg_get(cfg_get(cfg, "bank", {}) or {}, "min_score_topk_mean", None)
    diagnostics["adc_suppressed/candidate_count_pre_guard"] = float(mask.detach().to(dtype=torch.float32).sum().item())
    diagnostics["adc_suppressed/score_topk_mean_pre_guard"] = float(score_topk_mean.detach().item())
    diagnostics["adc_suppressed/score_p99_pre_guard"] = float(score_p99.detach().item())
    if min_score_topk_mean_raw is not None:
        min_score_topk_mean = float(min_score_topk_mean_raw)
        diagnostics[f"{ADC_STAT_PREFIX}/bank/min_score_topk_mean"] = float(min_score_topk_mean)
        if float(score_topk_mean.detach().item()) < float(min_score_topk_mean):
            diagnostics[f"{ADC_STAT_PREFIX}/bank_low_score_topk_mean"] = 1.0
            return None
        diagnostics[f"{ADC_STAT_PREFIX}/bank_low_score_topk_mean"] = 0.0
    dtype = _storage_dtype(cfg, branch.means.dtype)
    zeros = torch.zeros_like(score)
    bank = IForwardADCBank(
        valid=True,
        source_rollout_id=int(rollout_id),
        source_episode_id=int(episode_id),
        source_num_current_refs=int(num_current_refs),
        source_num_history_refs=int(num_history_refs),
        score=score.detach().to(dtype=dtype),
        abs_grad_current=zeros.detach().to(dtype=dtype),
        abs_grad_history=zeros.detach().to(dtype=dtype),
        scale_score=zeros.detach().to(dtype=dtype),
        conflict_score=zeros.detach().to(dtype=dtype),
        candidate_mask=mask.detach().to(dtype=torch.bool),
        score_topk_mean=score_topk_mean.detach().reshape(()).to(dtype=dtype),
        score_p90=score_p90.detach().reshape(()).to(dtype=dtype),
        score_p99=score_p99.detach().reshape(()).to(dtype=dtype),
        score_type=str(cfg_get(cfg_get(cfg, "score", {}) or {}, "type", "gate_suppressed_update")),
        score_max=accumulator.score_max.detach().to(dtype=dtype),
        score_sum=score_sum.detach().to(dtype=dtype),
        score_count=score_count.detach().to(dtype=dtype),
        parent_gate_mean=gate_mean.detach().to(dtype=dtype),
        parent_delta_demand=delta_demand.detach().to(dtype=dtype),
        parent_support_mean=support_mean.detach().to(dtype=dtype),
    )
    bank.valid = bool(bank.candidate_mask.any().item())
    return bank if bool(bank.valid) else None


def build_adc_lite_bank_from_losses(
    *,
    loss_current: torch.Tensor,
    loss_history: Optional[torch.Tensor],
    final_local_state: LocalGSState,
    cfg: Mapping[str, Any],
    rollout_id: int,
    episode_id: int,
    num_current_refs: int,
    num_history_refs: int,
    adc_meta: Optional[IForwardADCStateMeta] = None,
    aabb_min: Optional[torch.Tensor] = None,
    aabb_max: Optional[torch.Tensor] = None,
) -> Optional[IForwardADCBank]:
    if not bool(cfg_get(cfg, "enable", False)):
        return None
    if final_local_state.bg is None or int(final_local_state.bg.means.shape[0]) == 0:
        return None
    if int(num_current_refs) <= 0:
        return None
    require_history = bool(cfg_get(cfg, "require_history_for_clone", True))
    candidate_cfg = cfg_get(cfg, "candidate", {}) or {}
    require_history = bool(cfg_get(candidate_cfg, "require_history", require_history))
    if require_history and int(num_history_refs) <= 0:
        return None

    branch = final_local_state.bg
    params: List[torch.Tensor] = []
    names: List[str] = []
    _collect_bg_params(branch, params=params, names=names)
    if not params:
        return None

    current_grad = _grad_map_from_loss(loss_current, params=params, names=names)
    history_grad = (
        _grad_map_from_loss(loss_history, params=params, names=names)
        if int(num_history_refs) > 0
        else {name: None for name in names}
    )

    score_cfg = cfg_get(cfg, "score", {}) or {}
    attr_weights_raw = cfg_get(score_cfg, "grad_attr_weights", {}) or {}
    attr_weights = {
        "means": float(cfg_get(attr_weights_raw, "means", 1.0)),
        "scales": float(cfg_get(attr_weights_raw, "scales", 0.5)),
        "opacity": float(cfg_get(attr_weights_raw, "opacity", 0.75)),
        "sh": float(cfg_get(attr_weights_raw, "sh", 0.75)),
    }
    norm_cfg = cfg_get(score_cfg, "normalize", {}) or {}
    norm_pct = _float_cfg(norm_cfg, "percentile", 99.0)
    eps = _float_cfg(norm_cfg, "eps", 1.0e-8)

    abs_current_raw = _abs_grad_score(branch, current_grad, attr_weights=attr_weights)
    abs_history_raw = _abs_grad_score(branch, history_grad, attr_weights=attr_weights)
    abs_current = _normalize01(abs_current_raw, percentile=norm_pct, eps=eps)
    abs_history = _normalize01(abs_history_raw, percentile=norm_pct, eps=eps) if int(num_history_refs) > 0 else _zero_vec(abs_current)

    scale_cfg = cfg_get(score_cfg, "scale_proxy", {}) or {}
    scale_raw = torch.exp(branch.scales_log.detach()).reshape(int(branch.means.shape[0]), -1).amax(dim=-1)
    scale_score = _normalize01(
        scale_raw,
        percentile=_float_cfg(scale_cfg, "percentile", 95.0),
        eps=eps,
    )

    conflict_cfg = cfg_get(score_cfg, "conflict", {}) or {}
    conflict_eps = _float_cfg(conflict_cfg, "eps", eps)
    if int(num_history_refs) > 0:
        current_vec = _conflict_vectors(branch, current_grad, attr_weights=attr_weights)
        history_vec = _conflict_vectors(branch, history_grad, attr_weights=attr_weights)
        dot = (current_vec * history_vec).sum(dim=-1)
        denom = current_vec.norm(dim=-1) * history_vec.norm(dim=-1)
        cos = dot / denom.clamp_min(float(conflict_eps))
        cos = torch.where(torch.isfinite(cos), cos, torch.zeros_like(cos))
        neg_cos = torch.relu(-cos)
        conflict_mode = str(cfg_get(conflict_cfg, "mode", "relu_neg_cos_max_grad"))
        if conflict_mode in ("legacy_sqrt_abs_grad_cosine", "signed_gradient_cosine"):
            conflict_score = torch.sqrt((abs_current * abs_history).clamp_min(0.0)) * neg_cos
        elif conflict_mode in ("relu_neg_cos", "cosine_only"):
            conflict_score = neg_cos
        elif conflict_mode in ("relu_neg_cos_max_grad", "signed_gradient_cosine_max_grad"):
            grad_gate = torch.maximum(abs_current.detach(), abs_history.detach())
            conflict_score = neg_cos * grad_gate
        else:
            raise ValueError(f"unsupported adc_lite score.conflict.mode={conflict_mode!r}")
    else:
        conflict_score = _zero_vec(abs_current)

    weights_cfg = cfg_get(score_cfg, "weights", {}) or {}
    score = (
        _float_cfg(weights_cfg, "abs_grad_current", 1.0) * abs_current
        + _float_cfg(weights_cfg, "abs_grad_history", 0.5) * abs_history
        + _float_cfg(weights_cfg, "scale_or_screen_radius", 0.5) * scale_score
        + _float_cfg(weights_cfg, "current_history_conflict", 1.0) * conflict_score
    )
    score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
    candidate_mask = _build_candidate_mask(
        final_local_state=final_local_state,
        adc_meta=adc_meta,
        cfg=cfg,
        score=score,
        rollout_id=int(rollout_id),
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )
    score = score * candidate_mask.to(dtype=score.dtype)
    if not bool(candidate_mask.any().item()):
        return None

    budget_cfg = cfg_get(cfg, "budget", {}) or {}
    topk = int(cfg_get(budget_cfg, "max_new_points_per_rollout", 2000))
    dtype = _storage_dtype(cfg, branch.means.dtype)
    bank = IForwardADCBank(
        valid=True,
        source_rollout_id=int(rollout_id),
        source_episode_id=int(episode_id),
        source_num_current_refs=int(num_current_refs),
        source_num_history_refs=int(num_history_refs),
        score=score.detach().to(dtype=dtype),
        abs_grad_current=abs_current.detach().to(dtype=dtype),
        abs_grad_history=abs_history.detach().to(dtype=dtype),
        scale_score=scale_score.detach().to(dtype=dtype),
        conflict_score=conflict_score.detach().to(dtype=dtype),
        candidate_mask=candidate_mask.detach().to(dtype=torch.bool),
        score_topk_mean=_masked_topk_mean(score, candidate_mask, topk).detach().reshape(()).to(dtype=dtype),
        score_p90=_masked_percentile(score, candidate_mask, 90.0).detach().reshape(()).to(dtype=dtype),
        score_p99=_masked_percentile(score, candidate_mask, 99.0).detach().reshape(()).to(dtype=dtype),
    )
    bank.valid = bool(bank.candidate_mask.any().item())
    return bank if bool(bank.valid) else None


def ensure_adc_meta_for_state(
    *,
    local_state: LocalGSState,
    adc_meta: Optional[IForwardADCStateMeta],
    device: torch.device,
) -> IForwardADCStateMeta:
    n = int(local_state.bg.means.shape[0])
    if adc_meta is None:
        parent = torch.full((n,), -1, device=device, dtype=torch.long)
        birth = torch.full((n,), -1, device=device, dtype=torch.long)
        cooldown = torch.full((n,), -1, device=device, dtype=torch.long)
        return IForwardADCStateMeta(
            original_bg_count=n,
            num_bg_clones_created_episode=0,
            parent_index=parent,
            birth_rollout_id=birth,
            cooldown_until_rollout=cooldown,
        )
    meta = adc_meta.to(device=device)
    if meta.parent_index is None or int(meta.parent_index.numel()) != n:
        original = min(int(meta.original_bg_count), n)
        parent = torch.full((n,), -1, device=device, dtype=torch.long)
        birth = torch.full((n,), -1, device=device, dtype=torch.long)
        cooldown = torch.full((n,), -1, device=device, dtype=torch.long)
        if meta.parent_index is not None:
            m = min(int(meta.parent_index.numel()), n)
            parent[:m] = meta.parent_index.to(device=device, dtype=torch.long)[:m]
        if meta.birth_rollout_id is not None:
            m = min(int(meta.birth_rollout_id.numel()), n)
            birth[:m] = meta.birth_rollout_id.to(device=device, dtype=torch.long)[:m]
        if meta.cooldown_until_rollout is not None:
            m = min(int(meta.cooldown_until_rollout.numel()), n)
            cooldown[:m] = meta.cooldown_until_rollout.to(device=device, dtype=torch.long)[:m]
        meta = IForwardADCStateMeta(
            original_bg_count=original,
            num_bg_clones_created_episode=int(meta.num_bg_clones_created_episode),
            parent_index=parent,
            birth_rollout_id=birth,
            cooldown_until_rollout=cooldown,
        )
    return meta


def _logit(alpha: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    alpha = alpha.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(alpha) - torch.log1p(-alpha)


def _deterministic_jitter(parent_idx: torch.Tensor, rollout_id: int, scales_log: torch.Tensor, scale: float) -> torch.Tensor:
    if float(scale) == 0.0 or int(parent_idx.numel()) == 0:
        return scales_log.new_zeros((int(parent_idx.numel()), 3))
    idx = parent_idx.to(device=scales_log.device, dtype=scales_log.dtype).reshape(-1, 1) + 1.0
    freq = scales_log.new_tensor([[12.9898, 78.2330, 37.7190]])
    raw = torch.sin(idx * freq + float(rollout_id) * 0.12345) * 43758.5453
    frac = raw - torch.floor(raw)
    unit = frac * 2.0 - 1.0
    std = torch.exp(scales_log).mean(dim=-1, keepdim=True) * float(scale)
    return unit * std


def _clone_local_bg_branch(
    branch: LocalBranchState,
    *,
    parent_idx: torch.Tensor,
    rollout_id: int,
    cfg: Any,
    aabb_min: Optional[torch.Tensor] = None,
    aabb_max: Optional[torch.Tensor] = None,
    voxel_size: Optional[float] = None,
) -> Tuple[LocalBranchState, Dict[str, float]]:
    clone_cfg = cfg_get(cfg, "clone", {}) or {}
    parent_idx = parent_idx.to(device=branch.means.device, dtype=torch.long).reshape(-1)
    k = int(parent_idx.numel())
    parent_scales = branch.scales_log[parent_idx]
    jitter = _deterministic_jitter(
        parent_idx,
        int(rollout_id),
        parent_scales.detach(),
        _float_cfg(clone_cfg, "mean_jitter_std_scale", 0.05),
    ).to(dtype=branch.means.dtype)
    parent_means = branch.means[parent_idx]
    child_means_raw = parent_means + jitter
    child_means = child_means_raw
    child_oob_before = child_means_raw.new_zeros((k,), dtype=torch.bool)
    child_clamped = child_means_raw.new_zeros((k,), dtype=torch.bool)
    if aabb_min is not None and aabb_max is not None and k > 0:
        clone_eps = _float_cfg(clone_cfg, "aabb_eps", 1.0e-5)
        lo, hi = _aabb_bounds(ref=child_means_raw, aabb_min=aabb_min, aabb_max=aabb_max, eps=float(clone_eps))
        if lo is not None and hi is not None:
            child_oob_before = ((child_means_raw < lo) | (child_means_raw > hi)).any(dim=-1)
            child_means = child_means_raw.clamp(min=lo, max=hi)
            child_clamped = ((child_means.detach() - child_means_raw.detach()).abs() > 0.0).any(dim=-1)
    child_scales = parent_scales
    child_quats = branch.quats[parent_idx]
    child_sh_dc = branch.sh_dc[parent_idx]
    child_sh_rest = branch.sh_rest[parent_idx]
    hidden_mode = str(cfg_get(clone_cfg, "local_hidden_init", "parent"))
    child_hidden = torch.zeros((k, int(branch.hidden.shape[1])), device=branch.hidden.device, dtype=branch.hidden.dtype)
    if hidden_mode == "parent":
        child_hidden = branch.hidden[parent_idx]
    child_appearance_logvar = branch.appearance_logvar[parent_idx].to(dtype=torch.float32)

    opacity = branch.opacity_logit
    parent_alpha_old = torch.sigmoid(opacity[parent_idx])
    split_mode = str(cfg_get(clone_cfg, "opacity_split", "alpha_preserving"))
    if split_mode == "alpha_preserving":
        alpha_each = 1.0 - torch.sqrt((1.0 - parent_alpha_old).clamp_min(1.0e-8))
    elif split_mode in {"copy", "parent"}:
        alpha_each = parent_alpha_old
    else:
        alpha_each = (0.5 * parent_alpha_old).clamp_min(1.0e-8)
    split_logit = _logit(alpha_each).to(dtype=opacity.dtype)
    parent_opacity = opacity.clone()
    parent_opacity[parent_idx] = split_logit
    child_opacity = split_logit

    combined = 1.0 - (1.0 - alpha_each) * (1.0 - alpha_each)
    alpha_error = (combined - parent_alpha_old).abs()
    same_voxel_ratio = child_means_raw.new_tensor(0.0)
    diff_voxel_ratio = child_means_raw.new_tensor(0.0)
    if aabb_min is not None and aabb_max is not None and voxel_size is not None and float(voxel_size) > 0.0 and k > 0:
        lo, hi = _aabb_bounds(ref=child_means, aabb_min=aabb_min, aabb_max=aabb_max)
        if lo is not None and hi is not None:
            parent_grid = torch.floor((parent_means.detach().clamp(min=lo, max=hi) - lo) / float(voxel_size)).long()
            child_grid = torch.floor((child_means.detach().clamp(min=lo, max=hi) - lo) / float(voxel_size)).long()
            same_voxel = (parent_grid == child_grid).all(dim=-1)
            same_voxel_ratio = same_voxel.to(dtype=torch.float32).mean()
            diff_voxel_ratio = (~same_voxel).to(dtype=torch.float32).mean()
    aux = {
        f"{ADC_STAT_PREFIX}/parent_alpha_mean": float(parent_alpha_old.detach().mean().item()) if k else 0.0,
        f"{ADC_STAT_PREFIX}/child_alpha_mean": float(alpha_each.detach().mean().item()) if k else 0.0,
        f"{ADC_STAT_PREFIX}/alpha_combined_error_mean": float(alpha_error.detach().mean().item()) if k else 0.0,
        f"{ADC_STAT_PREFIX}/child_oob_before_clamp_ratio": (
            float(child_oob_before.detach().to(dtype=torch.float32).mean().item()) if k else 0.0
        ),
        f"{ADC_STAT_PREFIX}/child_clamped_ratio": (
            float(child_clamped.detach().to(dtype=torch.float32).mean().item()) if k else 0.0
        ),
        f"{ADC_STAT_PREFIX}/child_same_voxel_parent_ratio": float(same_voxel_ratio.detach().item()) if k else 0.0,
        f"{ADC_STAT_PREFIX}/child_new_voxel_ratio": float(diff_voxel_ratio.detach().item()) if k else 0.0,
    }
    return (
        LocalBranchState(
            means=torch.cat([branch.means, child_means], dim=0),
            scales_log=torch.cat([branch.scales_log, child_scales], dim=0),
            quats=torch.cat([branch.quats, child_quats], dim=0),
            opacity_logit=torch.cat([parent_opacity, child_opacity], dim=0),
            sh_dc=torch.cat([branch.sh_dc, child_sh_dc], dim=0),
            sh_rest=torch.cat([branch.sh_rest, child_sh_rest], dim=0),
            hidden=torch.cat([branch.hidden, child_hidden], dim=0),
            appearance_logvar=torch.cat(
                [branch.appearance_logvar.float(), child_appearance_logvar],
                dim=0,
            ),
        ),
        aux,
    )


def _extend_gru_branch(branch: IForwardGRUBranchState, *, num_new: int, ref: torch.Tensor) -> IForwardGRUBranchState:
    if int(num_new) <= 0:
        return branch
    h_new = ref.new_zeros((int(num_new), int(branch.h.shape[1])), dtype=branch.h.dtype)
    seen_new = torch.zeros((int(num_new),), device=ref.device, dtype=torch.bool)
    last_new = torch.full((int(num_new),), -1, device=ref.device, dtype=torch.long)
    return IForwardGRUBranchState(
        h=torch.cat([branch.h.to(device=ref.device), h_new], dim=0),
        seen=torch.cat([branch.seen.to(device=ref.device), seen_new], dim=0),
        last_visit_idx=torch.cat([branch.last_visit_idx.to(device=ref.device), last_new], dim=0),
        last_source_frame_idx=torch.cat([branch.last_source_frame_idx.to(device=ref.device), last_new.clone()], dim=0),
    )


def _extend_gru_memory(memory: Any, *, num_new: int, ref: torch.Tensor) -> Any:
    if not isinstance(memory, IForwardGRUMemoryState) or int(num_new) <= 0:
        return memory
    return IForwardGRUMemoryState(
        bg=_extend_gru_branch(memory.bg, num_new=int(num_new), ref=ref),
        distant=memory.distant,
        rigid=memory.rigid,
    )


def _extend_history_ema_branch(
    branch: IForwardHistoryBranchEMA,
    *,
    num_new: int,
    ref: torch.Tensor,
) -> IForwardHistoryBranchEMA:
    if int(num_new) <= 0:
        return branch
    z = ref.detach().new_zeros((int(num_new), 1), dtype=torch.float32)

    def cat(value: torch.Tensor) -> torch.Tensor:
        return torch.cat([value.to(device=ref.device), z.to(device=ref.device, dtype=value.dtype)], dim=0)

    return IForwardHistoryBranchEMA(
        support_fast=cat(branch.support_fast),
        error_fast=cat(branch.error_fast),
        update_norm_fast=cat(branch.update_norm_fast),
        support_slow=cat(branch.support_slow),
        error_slow=cat(branch.error_slow),
        update_norm_slow=cat(branch.update_norm_slow),
        initialized=cat(branch.initialized),
        block_support_sum=cat(branch.block_support_sum),
        block_present_count=cat(branch.block_present_count),
        block_visible_count=cat(branch.block_visible_count),
    )


def _extend_history_ema(history_ema: Any, *, num_new: int, ref: torch.Tensor) -> Any:
    if not isinstance(history_ema, IForwardHistoryEMAState) or int(num_new) <= 0:
        return history_ema
    return IForwardHistoryEMAState(
        bg=_extend_history_ema_branch(history_ema.bg, num_new=int(num_new), ref=ref),
        distant=history_ema.distant,
        rigid=history_ema.rigid,
    )


def _extend_gradient_attr(attr: GradientBankAttr, *, num_new: int, ref: torch.Tensor) -> GradientBankAttr:
    if int(num_new) <= 0:
        return attr
    direction_new = torch.zeros(
        (int(num_new),) + tuple(attr.direction.shape[1:]),
        device=ref.device,
        dtype=attr.direction.dtype,
    )
    log_new = torch.zeros((int(num_new), 1), device=ref.device, dtype=attr.log_norm.dtype)
    valid_new = torch.zeros((int(num_new),), device=ref.device, dtype=torch.bool)
    return GradientBankAttr(
        direction=torch.cat([attr.direction.to(device=ref.device), direction_new], dim=0),
        log_norm=torch.cat([attr.log_norm.to(device=ref.device), log_new], dim=0),
        valid=torch.cat([attr.valid.to(device=ref.device), valid_new], dim=0),
    )


def _extend_gradient_branch(
    branch: HistoryGradientBranchBank,
    *,
    num_new: int,
    ref: torch.Tensor,
) -> HistoryGradientBranchBank:
    return HistoryGradientBranchBank(
        means=_extend_gradient_attr(branch.means, num_new=int(num_new), ref=ref),
        scales=_extend_gradient_attr(branch.scales, num_new=int(num_new), ref=ref),
        quat=_extend_gradient_attr(branch.quat, num_new=int(num_new), ref=ref),
        opacity=_extend_gradient_attr(branch.opacity, num_new=int(num_new), ref=ref),
        sh=_extend_gradient_attr(branch.sh, num_new=int(num_new), ref=ref),
    )


def _extend_history_gradient_bank(bank: Any, *, num_new: int, ref: torch.Tensor) -> Any:
    if not isinstance(bank, HistoryGradientBank) or int(num_new) <= 0:
        return bank
    return HistoryGradientBank(
        bg=_extend_gradient_branch(bank.bg, num_new=int(num_new), ref=ref),
        distant=bank.distant,
        rigid=bank.rigid,
        valid=bool(bank.valid),
        source_rollout_id=int(bank.source_rollout_id),
        source_history_loss=float(bank.source_history_loss),
        source_history_num_refs=int(bank.source_history_num_refs),
    )


def _append_adc_meta(
    meta: IForwardADCStateMeta,
    *,
    parent_idx: torch.Tensor,
    rollout_id: int,
    cooldown_rollouts: int,
) -> IForwardADCStateMeta:
    k = int(parent_idx.numel())
    device = parent_idx.device
    parent = meta.parent_index.to(device=device, dtype=torch.long) if meta.parent_index is not None else None
    birth = meta.birth_rollout_id.to(device=device, dtype=torch.long) if meta.birth_rollout_id is not None else None
    cooldown = (
        meta.cooldown_until_rollout.to(device=device, dtype=torch.long)
        if meta.cooldown_until_rollout is not None
        else None
    )
    if parent is None or birth is None or cooldown is None:
        n = int(parent_idx.max().item()) + 1 if k else int(meta.original_bg_count)
        parent = torch.full((n,), -1, device=device, dtype=torch.long)
        birth = torch.full((n,), -1, device=device, dtype=torch.long)
        cooldown = torch.full((n,), -1, device=device, dtype=torch.long)
    cooldown = cooldown.clone()
    if k:
        cooldown[parent_idx] = torch.maximum(
            cooldown[parent_idx],
            torch.full((k,), int(rollout_id) + int(cooldown_rollouts), device=device, dtype=torch.long),
        )
    child_birth = torch.full((k,), int(rollout_id), device=device, dtype=torch.long)
    child_cooldown = torch.full((k,), -1, device=device, dtype=torch.long)
    return IForwardADCStateMeta(
        original_bg_count=int(meta.original_bg_count),
        num_bg_clones_created_episode=int(meta.num_bg_clones_created_episode) + k,
        parent_index=torch.cat([parent, parent_idx.to(device=device, dtype=torch.long)], dim=0),
        birth_rollout_id=torch.cat([birth, child_birth], dim=0),
        cooldown_until_rollout=torch.cat([cooldown, child_cooldown], dim=0),
    )


def _empty_apply_stats(*, enabled: bool, bank_valid: bool, bg_count: int) -> Dict[str, float]:
    return {
        f"{ADC_STAT_PREFIX}/enabled": 1.0 if bool(enabled) else 0.0,
        f"{ADC_STAT_PREFIX}/bank_valid": 1.0 if bool(bank_valid) else 0.0,
        f"{ADC_STAT_PREFIX}/bank_dropped_without_apply": 0.0,
        f"{ADC_STAT_PREFIX}/bank_shape_mismatch": 0.0,
        f"{ADC_STAT_PREFIX}/applied": 0.0,
        f"{ADC_STAT_PREFIX}/num_cloned_this_rollout": 0.0,
        f"{ADC_STAT_PREFIX}/bg_count_before": float(bg_count),
        f"{ADC_STAT_PREFIX}/bg_count_after": float(bg_count),
        f"{ADC_STAT_PREFIX}/planning/enabled": 0.0,
        f"{ADC_STAT_PREFIX}/planning/applied": 0.0,
        f"{ADC_STAT_PREFIX}/candidate_count_after_planning": 0.0,
        f"{ADC_STAT_PREFIX}/clone_fraction_of_candidates": 0.0,
        "adc_suppressed/parent_gate_mean": 0.0,
        "adc_suppressed/parent_delta_demand_mean": 0.0,
        "adc_suppressed/parent_support_mean": 0.0,
        "adc_suppressed/selected_parent_suppression_rank_percentile": 0.0,
        "adc/raw_score/selected_rank_percentile": 0.0,
        "adc/planning_score/selected_rank_percentile": 0.0,
        "adc/final_score/selected_rank_percentile": 0.0,
        "adc/raw_score/parent_mean": 0.0,
        "adc/planning_score/parent_mean": 0.0,
        "adc/final_score/parent_mean": 0.0,
        "adc/parent_gate_mean": 0.0,
        "adc/all_gate_mean": 0.0,
        "adc/parent_delta_demand_mean": 0.0,
        "adc/all_delta_demand_mean": 0.0,
        "adc/parent_gate_contrast": 0.0,
        "adc/gate_distribution_p20": 0.0,
        "adc/gate_distribution_p50": 0.0,
        "adc/gate_distribution_p80": 0.0,
    }


def apply_bg_clone_episode_local(
    *,
    state: Any,
    cfg: Mapping[str, Any],
    rollout_id: int,
    device: torch.device,
    planning_support_bg: Optional[torch.Tensor] = None,
    planning_valid_bg: Optional[torch.Tensor] = None,
    aabb_min: Optional[torch.Tensor] = None,
    aabb_max: Optional[torch.Tensor] = None,
    voxel_size: Optional[float] = None,
) -> Tuple[Any, Dict[str, float]]:
    enabled = bool(cfg_get(cfg, "enable", False))
    local_state = state.local_gs.to(device=device)
    state.local_gs = local_state
    meta = ensure_adc_meta_for_state(local_state=local_state, adc_meta=getattr(state, "adc_meta", None), device=device)
    state.adc_meta = meta
    bg_count_before = int(local_state.bg.means.shape[0])
    bank = getattr(state, "adc_bank", None)
    bank_valid = bank is not None and bool(getattr(bank, "valid", False))
    stats = _empty_apply_stats(enabled=enabled, bank_valid=bank_valid, bg_count=bg_count_before)
    stats[f"{ADC_STAT_PREFIX}/num_cloned_episode"] = float(meta.num_bg_clones_created_episode)
    if not enabled or not bank_valid:
        return state, stats

    bank = bank.to(device=device, dtype=torch.float32)
    state.adc_bank = None
    if int(bank.score.numel()) != bg_count_before or int(bank.candidate_mask.numel()) != bg_count_before:
        stats[f"{ADC_STAT_PREFIX}/bank_shape_mismatch"] = 1.0
        return state, stats

    budget_cfg = cfg_get(cfg, "budget", {}) or {}
    max_rollout = int(cfg_get(budget_cfg, "max_new_points_per_rollout", 2000))
    max_episode = int(cfg_get(budget_cfg, "max_new_points_per_episode", 8000))
    max_total = int(cfg_get(budget_cfg, "max_total_bg_points_episode", bg_count_before + max_rollout))
    remaining_episode = max(0, int(max_episode) - int(meta.num_bg_clones_created_episode))
    remaining_total = max(0, int(max_total) - int(bg_count_before))
    k = min(int(max_rollout), int(remaining_episode), int(remaining_total))
    if k <= 0:
        stats[f"{ADC_STAT_PREFIX}/budget_exhausted"] = 1.0
        return state, stats

    candidate = bank.candidate_mask.to(device=device, dtype=torch.bool) & torch.isfinite(bank.score)
    ranked_score = bank.score
    planning_cfg = cfg_get(cfg, "planning", {}) or {}
    planning_enabled = bool(cfg_get(planning_cfg, "enable", False))
    stats[f"{ADC_STAT_PREFIX}/planning/enabled"] = 1.0 if planning_enabled else 0.0
    if planning_enabled:
        if planning_support_bg is None:
            stats[f"{ADC_STAT_PREFIX}/planning/missing_support"] = 1.0
        else:
            support = planning_support_bg.to(device=device, dtype=torch.float32).reshape(-1)
            if int(support.numel()) != bg_count_before:
                stats[f"{ADC_STAT_PREFIX}/planning/shape_mismatch"] = 1.0
            else:
                valid = torch.isfinite(support)
                if planning_valid_bg is not None:
                    planning_valid = planning_valid_bg.to(device=device, dtype=torch.bool).reshape(-1)
                    if int(planning_valid.numel()) == bg_count_before:
                        valid = valid & planning_valid
                    else:
                        stats[f"{ADC_STAT_PREFIX}/planning/valid_shape_mismatch"] = 1.0
                min_support = _float_cfg(planning_cfg, "min_support", 0.0)
                if _bool_cfg(planning_cfg, "require_visible", True):
                    valid = valid & (support > float(min_support))
                candidate = candidate & valid
                stats[f"{ADC_STAT_PREFIX}/planning/applied"] = 1.0
                stats[f"{ADC_STAT_PREFIX}/planning/visible_ratio"] = (
                    float(valid.detach().to(dtype=torch.float32).mean().item()) if int(valid.numel()) else 0.0
                )
                stats[f"{ADC_STAT_PREFIX}/planning/support_mean"] = (
                    float(support.detach()[valid].mean().item()) if bool(valid.any().item()) else 0.0
                )
                support_weight = _float_cfg(planning_cfg, "support_score_weight", 0.25)
                stats[f"{ADC_STAT_PREFIX}/planning/support_score_weight"] = float(support_weight)
                if float(support_weight) != 0.0:
                    norm_pct = _float_cfg(planning_cfg, "support_normalize_percentile", 95.0)
                    eps = _float_cfg(planning_cfg, "eps", 1.0e-8)
                    support_score = _normalize01(support.clamp_min(0.0), percentile=norm_pct, eps=eps)
                    ranked_score = ranked_score * (1.0 + float(support_weight) * support_score.to(dtype=ranked_score.dtype))
    valid_count = int(candidate.sum().item())
    if valid_count <= 0:
        stats[f"{ADC_STAT_PREFIX}/no_candidates"] = 1.0
        return state, stats
    k = min(k, valid_count)
    masked_score = torch.where(candidate, ranked_score, torch.full_like(ranked_score, -torch.inf))
    top = torch.topk(masked_score, k=k, largest=True)
    parent_idx = top.indices.to(device=device, dtype=torch.long)

    parent_alpha = torch.sigmoid(local_state.bg.opacity_logit.detach()[parent_idx]).reshape(k, -1).mean(dim=-1)
    parent_scale = torch.exp(local_state.bg.scales_log.detach()[parent_idx]).reshape(k, -1).amax(dim=-1)
    raw_rank_percentile = _rank_percentile_for_indices(bank.score, candidate, parent_idx)
    planning_rank_percentile = _rank_percentile_for_indices(ranked_score, candidate, parent_idx)
    final_rank_percentile = _rank_percentile_for_indices(masked_score, candidate, parent_idx)
    all_valid = (
        bank.score_count.detach().reshape(-1).to(device=device, dtype=torch.float32) > 0
        if bank.score_count is not None and int(bank.score_count.numel()) == int(bank.score.numel())
        else candidate
    )
    clone_branch, clone_aux = _clone_local_bg_branch(
        local_state.bg,
        parent_idx=parent_idx,
        rollout_id=int(rollout_id),
        cfg=cfg,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        voxel_size=voxel_size,
    )
    state.local_gs = LocalGSState(
        bg=clone_branch,
        distant=local_state.distant,
        rigid=local_state.rigid,
        rigid_template=local_state.rigid_template,
    )
    ref = state.local_gs.bg.means
    state.memory = _extend_gru_memory(state.memory, num_new=k, ref=ref)
    state.history_ema = _extend_history_ema(getattr(state, "history_ema", None), num_new=k, ref=ref)
    state.history_gradient_bank = _extend_history_gradient_bank(
        getattr(state, "history_gradient_bank", None),
        num_new=k,
        ref=ref,
    )
    cooldown_rollouts = int(cfg_get(budget_cfg, "cooldown_rollouts", 0))
    state.adc_meta = _append_adc_meta(
        meta,
        parent_idx=parent_idx,
        rollout_id=int(rollout_id),
        cooldown_rollouts=int(cooldown_rollouts),
    )

    bg_count_after = int(state.local_gs.bg.means.shape[0])
    stats.update(clone_aux)
    stats.update(
        {
            f"{ADC_STAT_PREFIX}/applied": 1.0,
            f"{ADC_STAT_PREFIX}/num_cloned_this_rollout": float(k),
            f"{ADC_STAT_PREFIX}/num_cloned_episode": float(state.adc_meta.num_bg_clones_created_episode),
            f"{ADC_STAT_PREFIX}/bg_count_after": float(bg_count_after),
            f"{ADC_STAT_PREFIX}/parent_score_mean": float(top.values.detach().mean().item()) if k else 0.0,
            f"{ADC_STAT_PREFIX}/parent_alpha_mean": float(parent_alpha.detach().mean().item()) if k else 0.0,
            f"{ADC_STAT_PREFIX}/parent_scale_mean": float(parent_scale.detach().mean().item()) if k else 0.0,
            f"{ADC_STAT_PREFIX}/candidate_count_after_planning": float(valid_count),
            f"{ADC_STAT_PREFIX}/clone_fraction_of_candidates": float(k) / float(max(valid_count, 1)),
            f"{ADC_STAT_PREFIX}/parent_conflict_mean": (
                float(bank.conflict_score.detach()[parent_idx].mean().item()) if k else 0.0
            ),
            f"{ADC_STAT_PREFIX}/score_type_gate_suppressed": (
                1.0 if _score_type_is_gate_suppressed(str(getattr(bank, "score_type", ""))) else 0.0
            ),
            "adc_suppressed/parent_score_mean": float(top.values.detach().mean().item()) if k else 0.0,
            "adc_suppressed/selected_parent_suppression_rank_percentile": (
                float(raw_rank_percentile.detach().item()) if k else 0.0
            ),
            "adc/raw_score/selected_rank_percentile": float(raw_rank_percentile.detach().item()) if k else 0.0,
            "adc/planning_score/selected_rank_percentile": (
                float(planning_rank_percentile.detach().item()) if k else 0.0
            ),
            "adc/final_score/selected_rank_percentile": float(final_rank_percentile.detach().item()) if k else 0.0,
            "adc/raw_score/parent_mean": float(bank.score.detach()[parent_idx].mean().item()) if k else 0.0,
            "adc/planning_score/parent_mean": float(ranked_score.detach()[parent_idx].mean().item()) if k else 0.0,
            "adc/final_score/parent_mean": float(top.values.detach().mean().item()) if k else 0.0,
            f"{ADC_STAT_PREFIX}/bank_source_rollout_id": float(bank.source_rollout_id),
            f"{ADC_STAT_PREFIX}/bank_source_episode_id": float(bank.source_episode_id),
        }
    )
    if bank.parent_gate_mean is not None:
        parent_gate_mean = float(bank.parent_gate_mean.detach()[parent_idx].mean().item()) if k else 0.0
        all_gate_mean = (
            float(bank.parent_gate_mean.detach().float()[all_valid].mean().item()) if bool(all_valid.any().item()) else 0.0
        )
        stats["adc_suppressed/parent_gate_mean"] = parent_gate_mean
        stats["adc_suppressed/all_gate_mean"] = all_gate_mean
        stats["adc/parent_gate_mean"] = parent_gate_mean
        stats["adc/all_gate_mean"] = all_gate_mean
        stats["adc/parent_gate_contrast"] = all_gate_mean - parent_gate_mean
        stats["adc/gate_distribution_p20"] = float(_masked_percentile(bank.parent_gate_mean.float(), all_valid, 20.0).item())
        stats["adc/gate_distribution_p50"] = float(_masked_percentile(bank.parent_gate_mean.float(), all_valid, 50.0).item())
        stats["adc/gate_distribution_p80"] = float(_masked_percentile(bank.parent_gate_mean.float(), all_valid, 80.0).item())
        stats["adc_suppressed/gate_distribution_p20"] = stats["adc/gate_distribution_p20"]
        stats["adc_suppressed/gate_distribution_p50"] = stats["adc/gate_distribution_p50"]
        stats["adc_suppressed/gate_distribution_p80"] = stats["adc/gate_distribution_p80"]
        stats[f"{ADC_STAT_PREFIX}/parent_gate_mean"] = stats["adc_suppressed/parent_gate_mean"]
    if bank.parent_delta_demand is not None:
        parent_delta_mean = float(bank.parent_delta_demand.detach()[parent_idx].mean().item()) if k else 0.0
        all_delta_mean = (
            float(bank.parent_delta_demand.detach().float()[all_valid].mean().item())
            if bool(all_valid.any().item())
            else 0.0
        )
        stats["adc_suppressed/parent_delta_demand_mean"] = parent_delta_mean
        stats["adc_suppressed/all_delta_demand_mean"] = all_delta_mean
        stats["adc/parent_delta_demand_mean"] = parent_delta_mean
        stats["adc/all_delta_demand_mean"] = all_delta_mean
        stats[f"{ADC_STAT_PREFIX}/parent_delta_demand_mean"] = stats["adc_suppressed/parent_delta_demand_mean"]
    if bank.parent_support_mean is not None:
        stats["adc_suppressed/parent_support_mean"] = (
            float(bank.parent_support_mean.detach()[parent_idx].mean().item()) if k else 0.0
        )
        stats["adc_suppressed/all_support_mean"] = (
            float(bank.parent_support_mean.detach().float()[all_valid].mean().item()) if bool(all_valid.any().item()) else 0.0
        )
        stats[f"{ADC_STAT_PREFIX}/parent_support_mean"] = stats["adc_suppressed/parent_support_mean"]
    return state, stats


def adc_bank_stats(bank: Optional[IForwardADCBank], *, prefix: str = ADC_STAT_PREFIX) -> Dict[str, float]:
    if bank is None or not bool(getattr(bank, "valid", False)):
        return {
            f"{prefix}/next_bank_valid": 0.0,
            f"{prefix}/score_type_gate_suppressed": 0.0,
            f"{prefix}/score/topk_mean": 0.0,
            f"{prefix}/score/p90": 0.0,
            f"{prefix}/score/p99": 0.0,
            f"{prefix}/score/abs_grad_current_topk_mean": 0.0,
            f"{prefix}/score/abs_grad_history_topk_mean": 0.0,
            f"{prefix}/score/scale_topk_mean": 0.0,
            f"{prefix}/score/conflict_topk_mean": 0.0,
        }
    mask = bank.candidate_mask.detach().to(dtype=torch.bool)
    all_valid = (
        bank.score_count.detach().reshape(-1).to(dtype=torch.float32) > 0
        if bank.score_count is not None and int(bank.score_count.numel()) == int(bank.score.numel())
        else mask
    )
    k = int(mask.sum().item())
    out = {
        f"{prefix}/next_bank_valid": 1.0,
        f"{prefix}/score_type_gate_suppressed": (
            1.0 if _score_type_is_gate_suppressed(str(getattr(bank, "score_type", ""))) else 0.0
        ),
        f"{prefix}/score/topk_mean": float(bank.score_topk_mean.detach().item()),
        f"{prefix}/score/p90": float(bank.score_p90.detach().item()),
        f"{prefix}/score/p99": float(bank.score_p99.detach().item()),
        f"{prefix}/score/abs_grad_current_topk_mean": float(_masked_topk_mean(bank.abs_grad_current.float(), mask, k).item()),
        f"{prefix}/score/abs_grad_history_topk_mean": float(_masked_topk_mean(bank.abs_grad_history.float(), mask, k).item()),
        f"{prefix}/score/scale_topk_mean": float(_masked_topk_mean(bank.scale_score.float(), mask, k).item()),
        f"{prefix}/score/conflict_topk_mean": float(_masked_topk_mean(bank.conflict_score.float(), mask, k).item()),
        "adc_suppressed/score_mean": float(bank.score.detach().float()[mask].mean().item()) if k else 0.0,
        "adc_suppressed/score_topk_mean": float(bank.score_topk_mean.detach().item()),
        "adc_suppressed/score_p90": float(bank.score_p90.detach().item()),
        "adc_suppressed/score_p99": float(bank.score_p99.detach().item()),
        "adc_suppressed/candidate_count": float(k),
    }
    if bank.parent_gate_mean is not None:
        out["adc_suppressed/all_gate_mean"] = (
            float(bank.parent_gate_mean.detach().float()[all_valid].mean().item()) if bool(all_valid.any().item()) else 0.0
        )
        out["adc/gate_distribution_p20"] = float(_masked_percentile(bank.parent_gate_mean.float(), all_valid, 20.0).item())
        out["adc/gate_distribution_p50"] = float(_masked_percentile(bank.parent_gate_mean.float(), all_valid, 50.0).item())
        out["adc/gate_distribution_p80"] = float(_masked_percentile(bank.parent_gate_mean.float(), all_valid, 80.0).item())
        out["adc_suppressed/gate_distribution_p20"] = out["adc/gate_distribution_p20"]
        out["adc_suppressed/gate_distribution_p50"] = out["adc/gate_distribution_p50"]
        out["adc_suppressed/gate_distribution_p80"] = out["adc/gate_distribution_p80"]
    if bank.parent_delta_demand is not None:
        out["adc_suppressed/all_delta_demand_mean"] = (
            float(bank.parent_delta_demand.detach().float()[all_valid].mean().item())
            if bool(all_valid.any().item())
            else 0.0
        )
    if bank.parent_support_mean is not None:
        out["adc_suppressed/all_support_mean"] = (
            float(bank.parent_support_mean.detach().float()[all_valid].mean().item()) if bool(all_valid.any().item()) else 0.0
        )
    out.setdefault("adc_suppressed/all_gate_mean", 0.0)
    out.setdefault("adc_suppressed/all_delta_demand_mean", 0.0)
    out.setdefault("adc_suppressed/all_support_mean", 0.0)
    out.setdefault("adc/gate_distribution_p20", 0.0)
    out.setdefault("adc/gate_distribution_p50", 0.0)
    out.setdefault("adc/gate_distribution_p80", 0.0)
    return out


__all__ = [
    "ADC_STAT_PREFIX",
    "GateSuppressedADCAccumulator",
    "IForwardADCBank",
    "IForwardADCStateMeta",
    "adc_bank_stats",
    "apply_bg_clone_episode_local",
    "build_adc_lite_bank_from_losses",
    "build_gate_suppressed_adc_bank",
    "compute_gate_suppressed_score",
    "ensure_adc_meta_for_state",
]
