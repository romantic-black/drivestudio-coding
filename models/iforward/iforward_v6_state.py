from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .state import DenseMambaState, KeyedMambaState


@dataclass
class IForwardV6BranchPointState:
    point: Optional[KeyedMambaState] = None
    dense_point: Optional[DenseMambaState] = None

    def detach(self) -> "IForwardV6BranchPointState":
        return IForwardV6BranchPointState(
            point=None if self.point is None else self.point.detach(),
            dense_point=None if self.dense_point is None else self.dense_point.detach(),
        )


@dataclass
class IForwardV6MemoryState:
    bg: IForwardV6BranchPointState
    distant: IForwardV6BranchPointState
    rigid: IForwardV6BranchPointState

    @classmethod
    def empty(cls) -> "IForwardV6MemoryState":
        return cls(
            bg=IForwardV6BranchPointState(),
            distant=IForwardV6BranchPointState(),
            rigid=IForwardV6BranchPointState(),
        )

    def detach(self) -> "IForwardV6MemoryState":
        return IForwardV6MemoryState(
            bg=self.bg.detach(),
            distant=self.distant.detach(),
            rigid=self.rigid.detach(),
        )

    def count_tokens(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for branch_name in ("bg", "distant", "rigid"):
            branch = getattr(self, branch_name)
            state = branch.dense_point if branch.dense_point is not None else branch.point
            seen = state.seen.detach().to(dtype=torch.bool) if state is not None else None
            capacity = int(seen.numel()) if seen is not None else 0
            seen_count = int(seen.sum().item()) if seen is not None and capacity > 0 else 0
            out[f"{branch_name}_point"] = float(seen_count)
            out[f"{branch_name}_point_seen"] = float(seen_count)
            out[f"{branch_name}_point_capacity"] = float(capacity)
            out[f"{branch_name}_point_seen_ratio"] = float(seen_count) / float(max(capacity, 1))
        return out

