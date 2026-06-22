from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch


@dataclass
class FWHRLiftOutput:
    parent_context: torch.Tensor
    parent_support: torch.Tensor
    parent_obs_code: Optional[torch.Tensor]
    child_detail: torch.Tensor
    child_detail_support: torch.Tensor
    child_detail_valid: torch.Tensor
    aux: Dict[str, Any]


def aggregate_fwhr_child_lift(
    *,
    child_feature_sum: torch.Tensor,
    child_weight_sum_feature: torch.Tensor,
    child_support: torch.Tensor,
    child_to_parent: torch.Tensor,
    num_parents: int,
    context_dim: int = 48,
    detail_dim: int = 8,
    eps: float = 1.0e-6,
    detail_valid_threshold: Union[torch.Tensor, float] = 0.0,
    parent_obs_mode: str = "zero",
) -> FWHRLiftOutput:
    """Aggregate one fine-GS lift into parent context and centered child detail.

    The input `child_feature_sum` is expected to be `[N, context_dim + detail_dim]`,
    produced by a single fine-scene fused raster/backproject pass before per-child
    normalization. Context/detail aggregation uses the same feature denominator
    as the fused backprojector, while support remains a visibility/statistics
    signal.
    """
    if child_feature_sum.dim() != 2:
        raise ValueError(f"child_feature_sum must be [N,C], got {tuple(child_feature_sum.shape)}")
    n = int(child_feature_sum.shape[0])
    if int(child_feature_sum.shape[1]) != int(context_dim) + int(detail_dim):
        raise ValueError(
            "FW-HR child feature sum dim mismatch: "
            f"got {int(child_feature_sum.shape[1])}, expected {int(context_dim) + int(detail_dim)}"
        )
    if int(child_to_parent.numel()) != n:
        raise ValueError(f"child_to_parent row mismatch: {int(child_to_parent.numel())} vs {n}")
    if int(child_weight_sum_feature.numel()) != n:
        raise ValueError(f"child_weight_sum_feature row mismatch: {int(child_weight_sum_feature.numel())} vs {n}")
    if int(child_support.numel()) != n:
        raise ValueError(f"child_support row mismatch: {int(child_support.numel())} vs {n}")
    obs_mode = str(parent_obs_mode).strip().lower()
    if obs_mode not in {"zero", "none"}:
        raise ValueError(f"unsupported FW-HR parent_obs_mode={parent_obs_mode!r}; expected 'zero' or 'none'")

    device = child_feature_sum.device
    dtype = child_feature_sum.dtype
    pid = child_to_parent.to(device=device, dtype=torch.long).reshape(-1)
    support = child_support.to(device=device, dtype=dtype).reshape(-1).clamp_min(0.0)
    w_feat = child_weight_sum_feature.to(device=device, dtype=dtype).reshape(-1).clamp_min(0.0)
    if torch.is_tensor(detail_valid_threshold):
        valid_threshold = detail_valid_threshold.to(device=device, dtype=dtype).reshape(-1)
        if int(valid_threshold.numel()) != n:
            raise ValueError(
                f"detail_valid_threshold row mismatch: {int(valid_threshold.numel())} vs {n}"
            )
    else:
        valid_threshold = torch.full((n,), float(detail_valid_threshold), device=device, dtype=dtype)
    context_sum = child_feature_sum[:, : int(context_dim)]
    detail_sum = child_feature_sum[:, int(context_dim) :]
    m = int(num_parents)

    parent_context_sum = child_feature_sum.new_zeros((m, int(context_dim)))
    parent_context_weight = child_feature_sum.new_zeros((m,))
    parent_detail_sum = child_feature_sum.new_zeros((m, int(detail_dim)))
    parent_detail_weight = child_feature_sum.new_zeros((m,))
    parent_support = child_feature_sum.new_zeros((m,))
    valid = (w_feat > float(eps)) & (support >= valid_threshold)
    detail_weight = torch.where(valid, w_feat, torch.zeros_like(w_feat))
    if n > 0 and m > 0:
        parent_context_sum.index_add_(0, pid, context_sum)
        parent_context_weight.index_add_(0, pid, w_feat)
        parent_detail_sum.index_add_(0, pid, detail_sum * valid.to(dtype).unsqueeze(-1))
        parent_detail_weight.index_add_(0, pid, detail_weight)
        parent_support.index_add_(0, pid, support)

    parent_context = parent_context_sum / parent_context_weight.unsqueeze(-1).clamp_min(float(eps))
    parent_obs_code = child_feature_sum.new_zeros((m, 2)) if obs_mode == "zero" else None
    parent_detail_mean = parent_detail_sum / parent_detail_weight.unsqueeze(-1).clamp_min(float(eps))
    child_detail_raw = detail_sum / w_feat.unsqueeze(-1).clamp_min(float(eps))
    child_detail = child_detail_raw - parent_detail_mean.index_select(0, pid) if n > 0 and m > 0 else child_detail_raw
    child_detail = torch.where(valid.unsqueeze(-1), child_detail, torch.zeros_like(child_detail))

    detail_mean_check = child_feature_sum.new_zeros((m, int(detail_dim)))
    if n > 0 and m > 0:
        detail_mean_check.index_add_(0, pid, child_detail * detail_weight.unsqueeze(-1))
    mean_error = (
        (detail_mean_check / parent_detail_weight.unsqueeze(-1).clamp_min(float(eps))).detach().abs().max()
        if int(detail_mean_check.numel()) > 0
        else child_feature_sum.new_tensor(0.0)
    )
    aux = {
        "weighted_mean_error": mean_error,
        "valid_child_ratio": valid.detach().float().mean() if int(valid.numel()) > 0 else child_feature_sum.new_tensor(0.0),
        "parent_obs_zero_mode": child_feature_sum.new_tensor(1.0 if obs_mode == "zero" else 0.0),
        "parent_obs_none_mode": child_feature_sum.new_tensor(1.0 if obs_mode == "none" else 0.0),
    }
    return FWHRLiftOutput(
        parent_context=parent_context,
        parent_support=parent_support,
        parent_obs_code=parent_obs_code,
        child_detail=child_detail,
        child_detail_support=support,
        child_detail_valid=valid,
        aux=aux,
    )


__all__ = ["FWHRLiftOutput", "aggregate_fwhr_child_lift"]
