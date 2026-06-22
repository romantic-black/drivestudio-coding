from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class HistoryDamageProbe:
    target_indices: List[int]
    before_per_ref: torch.Tensor

    @classmethod
    def empty(cls, *, ref: torch.Tensor) -> "HistoryDamageProbe":
        return cls(target_indices=[], before_per_ref=ref.new_zeros((0,)))

    @property
    def valid(self) -> bool:
        return bool(self.target_indices) and int(self.before_per_ref.numel()) > 0


def history_damage_hinge(
    *,
    after_per_ref: torch.Tensor,
    before_per_ref: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    if int(after_per_ref.numel()) == 0 or int(before_per_ref.numel()) == 0:
        ref = after_per_ref if int(after_per_ref.numel()) > 0 else before_per_ref
        return ref.new_tensor(0.0)
    n = min(int(after_per_ref.numel()), int(before_per_ref.numel()))
    damage = after_per_ref.reshape(-1)[:n] - before_per_ref.detach().reshape(-1)[:n] - float(margin)
    return torch.relu(damage).mean()


__all__ = ["HistoryDamageProbe", "history_damage_hinge"]
