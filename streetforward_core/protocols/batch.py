from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .refs import ImageRef
from .rollout import PhaseALocalUnrollPlan


RawBatch = Dict[str, Any]


@dataclass(frozen=True)
class ResolvedPhaseABatch:
    raw: RawBatch
    plan: PhaseALocalUnrollPlan
    source_index_by_ref: Dict[ImageRef, int]
    target_index_by_ref: Dict[ImageRef, int]
    evidence_source_indices_by_step: Tuple[Tuple[int, ...], ...]
    block_target_indices_by_step: Tuple[Tuple[int, ...], ...]
    nearby_target_indices_by_step: Tuple[Tuple[int, ...], ...]

    @property
    def inner_K(self) -> int:
        return int(self.plan.inner_K)

    @property
    def evidence_refs_by_step(self) -> List[List[Tuple[int, int]]]:
        return [[ref.as_tuple() for ref in step.evidence_refs] for step in self.plan.steps]

    @property
    def block_loss_refs_by_step(self) -> List[List[Tuple[int, int]]]:
        return [[ref.as_tuple() for ref in step.block_loss_refs] for step in self.plan.steps]

    @property
    def nearby_loss_refs_by_step(self) -> List[List[Tuple[int, int]]]:
        return [[ref.as_tuple() for ref in step.nearby_loss_refs] for step in self.plan.steps]

