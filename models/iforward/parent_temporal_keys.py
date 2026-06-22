from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from models.streetforward.stage6_0.event_encoder import EventPack


_BRANCH_KEY_OFFSET = {
    "bg": 0,
    "distant": 1_000_000_000_000,
    "rigid": 2_000_000_000_000,
}


@dataclass(frozen=True)
class ParentTemporalKeys:
    bg: torch.Tensor
    distant: Optional[torch.Tensor] = None
    rigid: Optional[torch.Tensor] = None


def _dense_keys(num_rows: int, *, device: torch.device, branch: str) -> torch.Tensor:
    return torch.arange(int(num_rows), device=device, dtype=torch.long) + int(_BRANCH_KEY_OFFSET[str(branch)])


def _rigid_keys_from_measurement(measurement: Dict[str, Any], *, ref: torch.Tensor) -> torch.Tensor:
    active = measurement.get("assign_rigid_active")
    if active is None or int(getattr(active, "active_parent_global", ref.new_zeros((0,), dtype=torch.long)).numel()) == 0:
        return torch.zeros((0,), device=ref.device, dtype=torch.long)
    global_parent = active.active_parent_global.to(device=ref.device, dtype=torch.long).reshape(-1)
    assign_rigid = measurement.get("assign_rigid")
    parent_object_id = getattr(assign_rigid, "parent_object_id", None) if assign_rigid is not None else None
    if parent_object_id is not None and int(parent_object_id.numel()) > int(global_parent.max().item()):
        obj = parent_object_id.to(device=ref.device, dtype=torch.long).index_select(0, global_parent)
        return int(_BRANCH_KEY_OFFSET["rigid"]) + obj * 10_000_000 + global_parent
    return int(_BRANCH_KEY_OFFSET["rigid"]) + global_parent


def build_parent_temporal_keys(
    *,
    parent_event: EventPack,
    measurement: Dict[str, Any],
) -> ParentTemporalKeys:
    ref = parent_event.event_bg
    bg = _dense_keys(int(parent_event.event_bg.shape[0]), device=ref.device, branch="bg")
    distant = None
    if parent_event.event_distant is not None:
        distant = _dense_keys(int(parent_event.event_distant.shape[0]), device=ref.device, branch="distant")
    rigid = None
    if parent_event.event_rigid is not None:
        rigid = _rigid_keys_from_measurement(measurement, ref=ref)
        if int(rigid.numel()) != int(parent_event.event_rigid.shape[0]):
            rigid = _dense_keys(int(parent_event.event_rigid.shape[0]), device=ref.device, branch="rigid")
    return ParentTemporalKeys(bg=bg, distant=distant, rigid=rigid)


__all__ = ["ParentTemporalKeys", "build_parent_temporal_keys"]
