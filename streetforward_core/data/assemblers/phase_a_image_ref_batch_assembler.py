from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.roles import Role
from streetforward_core.protocols.rollout import PhaseALocalUnrollPlan
from streetforward_core.protocols.validators import validate_phase_a_plan


def _dedupe_refs_keep_order(refs: Iterable[ImageRef]) -> List[Tuple[int, int]]:
    seen = set()
    out: List[Tuple[int, int]] = []
    for ref in refs:
        item = ImageRef.from_raw(ref).as_tuple()
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _dedupe_refs_roles_keep_order(
    refs: Sequence[ImageRef],
    roles: Sequence[str],
) -> Tuple[List[Tuple[int, int]], List[str]]:
    if len(refs) != len(roles):
        raise ValueError(f"Phase A refs/roles length mismatch: {len(refs)} vs {len(roles)}")
    role_by_ref: Dict[Tuple[int, int], str] = {}
    out_refs: List[Tuple[int, int]] = []
    out_roles: List[str] = []
    for ref, role in zip(refs, roles):
        item = ImageRef.from_raw(ref).as_tuple()
        role_s = str(role)
        previous = role_by_ref.get(item)
        if previous is not None:
            if previous != role_s:
                raise ValueError(f"Phase A target ref {item} has conflicting roles: {previous} vs {role_s}")
            continue
        role_by_ref[item] = role_s
        out_refs.append(item)
        out_roles.append(role_s)
    return out_refs, out_roles


def _refs_by_step(plan: PhaseALocalUnrollPlan, attr: str) -> List[List[Tuple[int, int]]]:
    return [[ImageRef.from_raw(ref).as_tuple() for ref in getattr(step, attr)] for step in plan.steps]


class PhaseAImageRefBatchAssembler:
    """Materialize a Phase A RolloutPlan directly through dataset image refs."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def materialize(self, plan: PhaseALocalUnrollPlan, *, include_test: bool = False) -> Dict[str, Any]:
        validate_phase_a_plan(plan)
        if not hasattr(self.dataset, "_assemble_segment_batch_from_image_refs"):
            raise ValueError("PhaseAImageRefBatchAssembler requires dataset._assemble_segment_batch_from_image_refs")

        source_refs = _dedupe_refs_keep_order(ref for step in plan.steps for ref in step.evidence_refs)

        raw_target_refs: List[ImageRef] = []
        raw_target_roles: List[str] = []
        for step in plan.steps:
            raw_target_refs.extend(step.block_loss_refs)
            raw_target_roles.extend([Role.BLOCK_LOSS.value for _ in step.block_loss_refs])
            raw_target_refs.extend(step.nearby_loss_refs)
            raw_target_roles.extend([Role.NEARBY_LOSS.value for _ in step.nearby_loss_refs])
        target_refs, target_roles = _dedupe_refs_roles_keep_order(raw_target_refs, raw_target_roles)

        if len(source_refs) == 0:
            raise ValueError("Phase A plan requires non-empty evidence refs.")
        if len(target_refs) == 0:
            raise ValueError("Phase A plan requires non-empty render loss refs.")

        batch = self.dataset._assemble_segment_batch_from_image_refs(
            int(plan.scene_id),
            int(plan.segment_id),
            source_refs,
            target_refs,
            aux_image_refs=None,
            query_label_image_refs=None,
            include_test=bool(include_test),
            test_image_refs=None,
            enforce_target0_equals_source=False,
            target_ref_purpose="train",
        )

        request_meta = dict(batch.get("request_meta") or {})
        request_meta.update(
            {
                "scheduler_version": "phase_a_core_v1",
                "legacy_scheduler_version": str((plan.meta or {}).get("legacy_scheduler_version", "")),
                "scheduler_phase": str(plan.phase),
                "assembly_mode": "image_ref_v9",
                "scene_id": int(plan.scene_id),
                "segment_id": int(plan.segment_id),
                "episode_id": int(plan.episode_id),
                "episode_idx_global": int(plan.episode_id),
                "num_cams": int(plan.num_cams),
                "inner_K": int(plan.inner_K),
                "source_image_refs": [tuple(x) for x in source_refs],
                "source_image_ref": tuple(source_refs[0]),
                "target_image_refs": [tuple(x) for x in target_refs],
                "target_image_roles": [str(x) for x in target_roles],
                "evidence_refs_by_step": _refs_by_step(plan, "evidence_refs"),
                "block_loss_refs_by_step": _refs_by_step(plan, "block_loss_refs"),
                "nearby_loss_refs_by_step": _refs_by_step(plan, "nearby_loss_refs"),
                "prefix_loss_refs_by_step": [[] for _ in plan.steps],
                "query_label_refs": [],
                "aux_loss_refs": [],
                "role_policy": {
                    Role.EVIDENCE.value: "update_only",
                    Role.BLOCK_LOSS.value: "loss_only",
                    Role.NEARBY_LOSS.value: "loss_only",
                    Role.PREFIX_LOSS.value: "forbidden_phase_a",
                    Role.QUERY_LABEL.value: "forbidden_phase_a",
                    Role.AUX_LOSS.value: "forbidden_phase_a",
                },
            }
        )
        batch["request_meta"] = request_meta
        batch["rollout_plan"] = plan
        return batch

