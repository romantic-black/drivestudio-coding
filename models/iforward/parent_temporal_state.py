from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .state import DenseMambaState, KeyedMambaState


@dataclass
class ParentTemporalBranchState:
    dense: Optional[DenseMambaState] = None
    keyed: Optional[KeyedMambaState] = None

    def detach(self) -> "ParentTemporalBranchState":
        return ParentTemporalBranchState(
            dense=None if self.dense is None else self.dense.detach(),
            keyed=None if self.keyed is None else self.keyed.detach(),
        )

    def count(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.dense is not None:
            seen = self.dense.seen
            out["dense_seen"] = float(seen.detach().float().sum().item())
            out["dense_capacity"] = float(int(seen.numel()))
        else:
            out["dense_seen"] = 0.0
            out["dense_capacity"] = 0.0
        if self.keyed is not None:
            seen = self.keyed.seen
            out["keyed_seen"] = float(seen.detach().float().sum().item())
            out["keyed_capacity"] = float(int(seen.numel()))
        else:
            out["keyed_seen"] = 0.0
            out["keyed_capacity"] = 0.0
        return out


@dataclass
class ParentTemporalState:
    bg: ParentTemporalBranchState
    distant: ParentTemporalBranchState
    rigid: ParentTemporalBranchState
    last_committed_block_id: int = -1

    @classmethod
    def empty(cls) -> "ParentTemporalState":
        return cls(
            bg=ParentTemporalBranchState(),
            distant=ParentTemporalBranchState(),
            rigid=ParentTemporalBranchState(),
            last_committed_block_id=-1,
        )

    def detach(self) -> "ParentTemporalState":
        return ParentTemporalState(
            bg=self.bg.detach(),
            distant=self.distant.detach(),
            rigid=self.rigid.detach(),
            last_committed_block_id=int(self.last_committed_block_id),
        )

    def count_tokens(self) -> Dict[str, float]:
        out: Dict[str, float] = {"parent_temporal_last_committed_block_id": float(self.last_committed_block_id)}
        for branch in ("bg", "distant", "rigid"):
            counts = getattr(self, branch).count()
            for key, value in counts.items():
                out[f"parent_temporal_{branch}_{key}"] = float(value)
        return out


__all__ = ["ParentTemporalBranchState", "ParentTemporalState"]
