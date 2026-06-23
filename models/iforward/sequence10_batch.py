from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .resolver import IForwardResolvedBatch, IForwardResolvedStep

SEQUENCE10_VISIT_BOOTSTRAP = "bootstrap"
SEQUENCE10_VISIT_CAUSAL_FIRST = "causal_first"
SEQUENCE10_VISIT_REPAIR = "repair"
SEQUENCE10_PHASE_BOOTSTRAP = "bootstrap"
SEQUENCE10_PHASE_CAUSAL = "causal"
SEQUENCE10_PHASE_REPAIR = "repair"
SEQUENCE10_VALID_VISITS = (
    SEQUENCE10_VISIT_BOOTSTRAP,
    SEQUENCE10_VISIT_CAUSAL_FIRST,
    SEQUENCE10_VISIT_REPAIR,
)
SEQUENCE10_VALID_PHASES = (
    SEQUENCE10_PHASE_BOOTSTRAP,
    SEQUENCE10_PHASE_CAUSAL,
    SEQUENCE10_PHASE_REPAIR,
)


@dataclass(frozen=True)
class Sequence10ResolvedFlags:
    scheduler_phase: str
    sequence_positions: Tuple[int, ...]
    temporal_commit_count: int
    repair: bool


def sequence10_flags(batch: IForwardResolvedBatch) -> Sequence10ResolvedFlags:
    positions = tuple(int(step.sequence_pos) for step in batch.steps if int(step.repeat_idx) == 0)
    phases = {str(step.scheduler_phase) for step in batch.steps}
    phase = next(iter(phases)) if len(phases) == 1 else ""
    return Sequence10ResolvedFlags(
        scheduler_phase=str(phase),
        sequence_positions=positions,
        temporal_commit_count=sum(1 for step in batch.steps if bool(step.temporal_commit)),
        repair=bool(phase == SEQUENCE10_PHASE_REPAIR),
    )


__all__ = [
    "SEQUENCE10_PHASE_BOOTSTRAP",
    "SEQUENCE10_PHASE_CAUSAL",
    "SEQUENCE10_PHASE_REPAIR",
    "SEQUENCE10_VALID_PHASES",
    "SEQUENCE10_VALID_VISITS",
    "SEQUENCE10_VISIT_BOOTSTRAP",
    "SEQUENCE10_VISIT_CAUSAL_FIRST",
    "SEQUENCE10_VISIT_REPAIR",
    "Sequence10ResolvedFlags",
    "sequence10_flags",
]
