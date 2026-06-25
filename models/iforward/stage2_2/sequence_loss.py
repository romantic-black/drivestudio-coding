from __future__ import annotations

from typing import Dict, Mapping

import torch


def role_normalized_loss(loss_by_role: Mapping[str, torch.Tensor], weight_by_role: Mapping[str, float] | None = None) -> torch.Tensor:
    if not loss_by_role:
        return torch.zeros((), dtype=torch.float32)
    weight_by_role = dict(weight_by_role or {})
    terms = []
    weights = []
    for role, loss in loss_by_role.items():
        w = float(weight_by_role.get(str(role), 1.0))
        if w == 0.0:
            continue
        terms.append(loss.reshape(()) * w)
        weights.append(abs(w))
    if not terms:
        ref = next(iter(loss_by_role.values()))
        return ref.reshape(()) * 0.0
    return torch.stack(terms).sum() / max(float(sum(weights)), 1.0e-8)


__all__ = ["role_normalized_loss"]
