from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ParentTemporalDenseStateV2:
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    seen: torch.Tensor
    last_timestamp_sec: torch.Tensor

    def detach(self) -> "ParentTemporalDenseStateV2":
        return ParentTemporalDenseStateV2(
            conv_state=self.conv_state.detach().clone(),
            ssm_state=self.ssm_state.detach().clone(),
            seen=self.seen.detach().clone(),
            last_timestamp_sec=self.last_timestamp_sec.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "ParentTemporalDenseStateV2":
        return ParentTemporalDenseStateV2(
            conv_state=self.conv_state.to(device=device, dtype=dtype or self.conv_state.dtype),
            ssm_state=self.ssm_state.to(device=device, dtype=dtype or self.ssm_state.dtype),
            seen=self.seen.to(device=device),
            last_timestamp_sec=self.last_timestamp_sec.to(device=device, dtype=dtype or self.last_timestamp_sec.dtype),
        )


@dataclass
class ParentTemporalKeyedStateV2:
    keys: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    seen: torch.Tensor
    last_timestamp_sec: torch.Tensor

    def detach(self) -> "ParentTemporalKeyedStateV2":
        return ParentTemporalKeyedStateV2(
            keys=self.keys.detach().clone(),
            conv_state=self.conv_state.detach().clone(),
            ssm_state=self.ssm_state.detach().clone(),
            seen=self.seen.detach().clone(),
            last_timestamp_sec=self.last_timestamp_sec.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "ParentTemporalKeyedStateV2":
        return ParentTemporalKeyedStateV2(
            keys=self.keys.to(device=device),
            conv_state=self.conv_state.to(device=device, dtype=dtype or self.conv_state.dtype),
            ssm_state=self.ssm_state.to(device=device, dtype=dtype or self.ssm_state.dtype),
            seen=self.seen.to(device=device),
            last_timestamp_sec=self.last_timestamp_sec.to(device=device, dtype=dtype or self.last_timestamp_sec.dtype),
        )


@dataclass
class ParentTemporalBranchStateV2:
    dense: Optional[ParentTemporalDenseStateV2] = None
    keyed: Optional[ParentTemporalKeyedStateV2] = None

    def detach(self) -> "ParentTemporalBranchStateV2":
        return ParentTemporalBranchStateV2(
            dense=None if self.dense is None else self.dense.detach(),
            keyed=None if self.keyed is None else self.keyed.detach(),
        )

    def count(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.dense is not None:
            seen = self.dense.seen.detach().to(dtype=torch.bool)
            out["dense_seen"] = float(seen.float().sum().item())
            out["dense_capacity"] = float(int(seen.numel()))
        else:
            out["dense_seen"] = 0.0
            out["dense_capacity"] = 0.0
        if self.keyed is not None:
            seen = self.keyed.seen.detach().to(dtype=torch.bool)
            out["keyed_seen"] = float(seen.float().sum().item())
            out["keyed_capacity"] = float(int(seen.numel()))
        else:
            out["keyed_seen"] = 0.0
            out["keyed_capacity"] = 0.0
        return out


@dataclass
class ParentTemporalStateV2:
    bg: ParentTemporalBranchStateV2
    distant: ParentTemporalBranchStateV2
    rigid: ParentTemporalBranchStateV2
    last_committed_block_id: int = -1
    last_timestamp_sec: float = -1.0

    @classmethod
    def empty(cls) -> "ParentTemporalStateV2":
        return cls(
            bg=ParentTemporalBranchStateV2(),
            distant=ParentTemporalBranchStateV2(),
            rigid=ParentTemporalBranchStateV2(),
            last_committed_block_id=-1,
            last_timestamp_sec=-1.0,
        )

    def detach(self) -> "ParentTemporalStateV2":
        return ParentTemporalStateV2(
            bg=self.bg.detach(),
            distant=self.distant.detach(),
            rigid=self.rigid.detach(),
            last_committed_block_id=int(self.last_committed_block_id),
            last_timestamp_sec=float(self.last_timestamp_sec),
        )

    def count_tokens(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "parent_temporal_v2_last_committed_block_id": float(self.last_committed_block_id),
            "parent_temporal_v2_last_timestamp_sec": float(self.last_timestamp_sec),
        }
        for branch in ("bg", "distant", "rigid"):
            counts = getattr(self, branch).count()
            for key, value in counts.items():
                out[f"parent_temporal_v2_{branch}_{key}"] = float(value)
        return out


__all__ = [
    "ParentTemporalBranchStateV2",
    "ParentTemporalDenseStateV2",
    "ParentTemporalKeyedStateV2",
    "ParentTemporalStateV2",
]
