from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class DenseOptimizerState:
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    seen: torch.Tensor
    update_count: torch.Tensor
    last_visit_step: torch.Tensor
    last_frame_id: torch.Tensor
    last_visit_kind: torch.Tensor

    def detach(self) -> "DenseOptimizerState":
        return DenseOptimizerState(
            conv_state=self.conv_state.detach().clone(),
            ssm_state=self.ssm_state.detach().clone(),
            seen=self.seen.detach().clone(),
            update_count=self.update_count.detach().clone(),
            last_visit_step=self.last_visit_step.detach().clone(),
            last_frame_id=self.last_frame_id.detach().clone(),
            last_visit_kind=self.last_visit_kind.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "DenseOptimizerState":
        return DenseOptimizerState(
            conv_state=self.conv_state.to(device=device, dtype=dtype or self.conv_state.dtype),
            ssm_state=self.ssm_state.to(device=device, dtype=dtype or self.ssm_state.dtype),
            seen=self.seen.to(device=device),
            update_count=self.update_count.to(device=device),
            last_visit_step=self.last_visit_step.to(device=device),
            last_frame_id=self.last_frame_id.to(device=device),
            last_visit_kind=self.last_visit_kind.to(device=device),
        )


@dataclass
class KeyedOptimizerState:
    keys: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    seen: torch.Tensor
    update_count: torch.Tensor
    last_visit_step: torch.Tensor
    last_frame_id: torch.Tensor
    last_visit_kind: torch.Tensor

    def detach(self) -> "KeyedOptimizerState":
        return KeyedOptimizerState(
            keys=self.keys.detach().clone(),
            conv_state=self.conv_state.detach().clone(),
            ssm_state=self.ssm_state.detach().clone(),
            seen=self.seen.detach().clone(),
            update_count=self.update_count.detach().clone(),
            last_visit_step=self.last_visit_step.detach().clone(),
            last_frame_id=self.last_frame_id.detach().clone(),
            last_visit_kind=self.last_visit_kind.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "KeyedOptimizerState":
        return KeyedOptimizerState(
            keys=self.keys.to(device=device),
            conv_state=self.conv_state.to(device=device, dtype=dtype or self.conv_state.dtype),
            ssm_state=self.ssm_state.to(device=device, dtype=dtype or self.ssm_state.dtype),
            seen=self.seen.to(device=device),
            update_count=self.update_count.to(device=device),
            last_visit_step=self.last_visit_step.to(device=device),
            last_frame_id=self.last_frame_id.to(device=device),
            last_visit_kind=self.last_visit_kind.to(device=device),
        )


@dataclass
class OptimizerBranchState:
    dense: Optional[DenseOptimizerState] = None
    keyed: Optional[KeyedOptimizerState] = None

    def detach(self) -> "OptimizerBranchState":
        return OptimizerBranchState(
            dense=None if self.dense is None else self.dense.detach(),
            keyed=None if self.keyed is None else self.keyed.detach(),
        )

    def count(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.dense is None:
            out["dense_seen"] = 0.0
            out["dense_capacity"] = 0.0
            out["dense_updates"] = 0.0
        else:
            seen = self.dense.seen.detach().to(dtype=torch.bool)
            updates = self.dense.update_count.detach().to(dtype=torch.float32)
            out["dense_seen"] = float(seen.float().sum().item())
            out["dense_capacity"] = float(int(seen.numel()))
            out["dense_updates"] = float(updates.sum().item()) if int(updates.numel()) else 0.0
        if self.keyed is None:
            out["keyed_seen"] = 0.0
            out["keyed_capacity"] = 0.0
            out["keyed_updates"] = 0.0
        else:
            seen = self.keyed.seen.detach().to(dtype=torch.bool)
            updates = self.keyed.update_count.detach().to(dtype=torch.float32)
            out["keyed_seen"] = float(seen.float().sum().item())
            out["keyed_capacity"] = float(int(seen.numel()))
            out["keyed_updates"] = float(updates.sum().item()) if int(updates.numel()) else 0.0
        return out


@dataclass
class ParentOptimizerMambaState:
    bg: OptimizerBranchState
    distant: OptimizerBranchState
    rigid: OptimizerBranchState
    global_update_step: int = 0

    @classmethod
    def empty(cls) -> "ParentOptimizerMambaState":
        return cls(
            bg=OptimizerBranchState(),
            distant=OptimizerBranchState(),
            rigid=OptimizerBranchState(),
            global_update_step=0,
        )

    def detach(self) -> "ParentOptimizerMambaState":
        return ParentOptimizerMambaState(
            bg=self.bg.detach(),
            distant=self.distant.detach(),
            rigid=self.rigid.detach(),
            global_update_step=int(self.global_update_step),
        )

    def count_tokens(self) -> Dict[str, float]:
        out: Dict[str, float] = {"parent_optimizer_mamba_global_update_step": float(self.global_update_step)}
        for branch in ("bg", "distant", "rigid"):
            for key, value in getattr(self, branch).count().items():
                out[f"parent_optimizer_mamba_{branch}_{key}"] = float(value)
        return out


__all__ = [
    "DenseOptimizerState",
    "KeyedOptimizerState",
    "OptimizerBranchState",
    "ParentOptimizerMambaState",
]
