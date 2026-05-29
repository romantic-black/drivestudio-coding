from __future__ import annotations

from typing import Any, Dict

from streetforward_core.protocols.rollout import PhaseALocalUnrollPlan


class V9ImageRefBatchAssembler:
    """Legacy V9 dataclass assembler.

    Phase A RolloutPlan materialization must use PhaseAImageRefBatchAssembler.
    """

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def materialize(self, plan: Any, *, include_test: bool = False) -> Dict[str, Any]:
        if isinstance(plan, PhaseALocalUnrollPlan):
            raise ValueError("PhaseALocalUnrollPlan must be materialized by PhaseAImageRefBatchAssembler")
        if not hasattr(self.dataset, "_assemble_segment_batch_from_v9_request"):
            raise ValueError("V9ImageRefBatchAssembler requires dataset._assemble_segment_batch_from_v9_request")
        batch = self.dataset._assemble_segment_batch_from_v9_request(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            v9_plan=plan,
            include_test=bool(include_test),
        )
        return batch
