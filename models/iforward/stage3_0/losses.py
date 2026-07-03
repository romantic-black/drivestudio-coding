from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def stage3_gather_regularization(
    terms: Dict[str, torch.Tensor],
    *,
    offset_l2_weight: float = 0.0,
    out_of_bounds_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ref = None
    for value in terms.values():
        if torch.is_tensor(value):
            ref = value
            break
    if ref is None:
        return torch.tensor(0.0), {}
    offset = terms.get("offset_l2", ref.new_tensor(0.0)).float()
    oob = terms.get("out_of_bounds", ref.new_tensor(0.0)).float()
    loss = float(offset_l2_weight) * offset + float(out_of_bounds_weight) * oob
    return loss, {
        "iforward/stage3/loss_offset_l2_raw": float(offset.detach().item()),
        "iforward/stage3/loss_out_of_bounds_raw": float(oob.detach().item()),
        "iforward/stage3/loss_offset_l2_weight": float(offset_l2_weight),
        "iforward/stage3/loss_out_of_bounds_weight": float(out_of_bounds_weight),
    }


def merge_stage3_reg_terms(*items: Any) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if not torch.is_tensor(value):
                continue
            name = str(key)
            out[name] = value if name not in out else out[name] + value
            counts[name] = counts.get(name, 0) + 1
    for key, count in counts.items():
        if int(count) > 1:
            out[key] = out[key] / float(count)
    return out


__all__ = ["merge_stage3_reg_terms", "stage3_gather_regularization"]
