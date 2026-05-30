from __future__ import annotations

from typing import Any, Dict

from streetforward_core.protocols.phase_b_long import (
    PhaseBLongRolloutPlan,
    phase_b_long_plan_to_request_meta,
)
from streetforward_core.protocols.validators import validate_phase_b_long_plan


class PhaseBLongBatchAssembler:
    """Materialize a Phase B Long rollout plan through dataset image refs."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def materialize(self, plan: PhaseBLongRolloutPlan, *, include_test: bool = False) -> Dict[str, Any]:
        validate_phase_b_long_plan(plan)
        if not hasattr(self.dataset, "_assemble_segment_batch_from_image_refs"):
            raise ValueError("PhaseBLongBatchAssembler requires dataset._assemble_segment_batch_from_image_refs")
        source_refs = [ref.as_tuple() for ref in plan.source_image_refs]
        target_refs = [ref.as_tuple() for ref in plan.target_image_refs]
        batch = self.dataset._assemble_segment_batch_from_image_refs(
            int(plan.scene_id),
            int(plan.segment_id),
            source_refs,
            target_refs,
            aux_image_refs=None,
            query_label_image_refs=[],
            include_test=bool(include_test),
            test_image_refs=None,
            enforce_target0_equals_source=False,
            target_ref_purpose="train",
        )
        request_meta = dict(batch.get("request_meta") or {})
        request_meta.update(phase_b_long_plan_to_request_meta(plan))
        batch["request_meta"] = request_meta
        batch["_scheduler_long_phase_b"] = dict(request_meta)
        batch["rollout_plan"] = plan
        return batch

