from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from streetforward_core.data.assemblers.phase_a_image_ref_batch_assembler import PhaseAImageRefBatchAssembler
from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.rollout import PHASE_A_NAME, PHASE_A_PROTOCOL_VERSION, PhaseALocalUnrollPlan, RolloutStep
from streetforward_core.protocols.validators import validate_phase_a_plan


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first_present(meta: Dict[str, Any], sched: Any, key: str, default: Any = None) -> Any:
    if key in meta and meta[key] is not None:
        return meta[key]
    value = _field(sched, key, None)
    return default if value is None else value


def _ref_groups(raw: Any, name: str) -> Tuple[Tuple[ImageRef, ...], ...]:
    if raw is None:
        raise ValueError(f"{name} is required")
    return tuple(tuple(ImageRef.from_raw(ref) for ref in list(group)) for group in list(raw))


def _flat(groups: Sequence[Sequence[ImageRef]]) -> List[ImageRef]:
    return [ref for group in groups for ref in group]


def _infer_num_cams(*groups: Sequence[Sequence[ImageRef]]) -> int:
    max_cam = -1
    for group in groups:
        for ref in _flat(group):
            max_cam = max(max_cam, int(ref.cam_idx))
    return int(max_cam + 1) if max_cam >= 0 else 0


def _require_phase_a_no_legacy_roles(
    *,
    prefix_by_step: Sequence[Sequence[ImageRef]],
    query_refs: Sequence[ImageRef],
    aux_refs: Sequence[ImageRef],
) -> None:
    if any(len(group) > 0 for group in prefix_by_step):
        raise ValueError("Phase A must not receive prefix_loss_refs.")
    if len(query_refs) > 0:
        raise ValueError("Phase A must not receive query_label_refs.")
    if len(aux_refs) > 0:
        raise ValueError("Phase A must not receive aux_loss_refs.")


def convert_v9_phase_a_plan(v9: Any) -> PhaseALocalUnrollPlan:
    if str(_field(v9, "scheduler_version", "")) != "v9":
        raise ValueError("expected scheduler_version=v9")
    if str(_field(v9, "phase", "")) != PHASE_A_NAME:
        raise ValueError("convert_v9_phase_a_plan requires phase_A_block_local_unroll")

    steps_raw = list(_field(v9, "steps", []) or [])
    evidence_by_step = _ref_groups(_field(v9, "evidence_refs_by_step", None), "evidence_refs_by_step")
    block_by_step = _ref_groups(_field(v9, "block_loss_refs_by_step", None), "block_loss_refs_by_step")
    nearby_by_step = _ref_groups(_field(v9, "nearby_loss_refs_by_step", None), "nearby_loss_refs_by_step")
    prefix_by_step = _ref_groups(_field(v9, "prefix_loss_refs_by_step", None), "prefix_loss_refs_by_step")
    query_refs = tuple(ImageRef.from_raw(ref) for ref in list(_field(v9, "query_label_refs", []) or []))
    aux_refs = tuple(ImageRef.from_raw(ref) for ref in list(_field(v9, "aux_loss_refs", []) or []))
    _require_phase_a_no_legacy_roles(prefix_by_step=prefix_by_step, query_refs=query_refs, aux_refs=aux_refs)

    inner_K = int(_field(v9, "inner_K", len(evidence_by_step)) or 0)
    steps = []
    for idx in range(inner_K):
        raw_step = steps_raw[idx] if idx < len(steps_raw) else None
        steps.append(
            RolloutStep(
                step_idx=int(_field(raw_step, "step_idx", idx)),
                evidence_refs=tuple(evidence_by_step[idx]),
                block_loss_refs=tuple(block_by_step[idx]),
                nearby_loss_refs=tuple(nearby_by_step[idx]),
            )
        )

    first_step = steps_raw[0] if steps_raw else None
    meta = {
        "legacy_scheduler_version": "v9",
        "episode_start_keyframe_pos": int(_field(v9, "episode_start_keyframe_pos", -1)),
        "keyframe_window": [int(x) for x in list(_field(v9, "keyframe_window", []) or [])],
        "frame_chain": [int(x) for x in list(_field(v9, "frame_chain", []) or [])],
    }
    plan = PhaseALocalUnrollPlan(
        protocol_version=PHASE_A_PROTOCOL_VERSION,
        phase=PHASE_A_NAME,
        scene_id=int(_field(v9, "scene_id", -1)),
        segment_id=int(_field(v9, "segment_id", -1)),
        episode_id=int(_field(v9, "episode_id", -1)),
        num_cams=int(_field(v9, "num_cams", 0) or _infer_num_cams(evidence_by_step, block_by_step, nearby_by_step)),
        inner_K=inner_K,
        steps=tuple(steps),
        source_keyframe_idx=(
            None if _field(first_step, "source_keyframe_idx", None) is None else int(_field(first_step, "source_keyframe_idx"))
        ),
        block_idx=None if _field(first_step, "block_idx", None) is None else int(_field(first_step, "block_idx")),
        meta=meta,
    )
    validate_phase_a_plan(plan)
    return plan


def legacy_v9_batch_to_phase_a_plan(batch: Dict[str, Any]) -> PhaseALocalUnrollPlan:
    meta = dict(batch.get("request_meta") or {})
    sched = batch.get("_scheduler_v9") or {}

    scheduler_version = str(_first_present(meta, sched, "scheduler_version", ""))
    if scheduler_version != "v9":
        raise ValueError("Stage6 Phase A requires scheduler_v9 batch.")

    phase = str(_first_present(meta, sched, "scheduler_phase", _field(sched, "phase", "")))
    if phase != PHASE_A_NAME:
        raise ValueError("Stage6 Phase A requires phase_A_block_local_unroll.")

    inner_K = int(_first_present(meta, sched, "inner_K", 0) or 0)
    evidence_by_step = _ref_groups(_first_present(meta, sched, "evidence_refs_by_step"), "evidence_refs_by_step")
    block_by_step = _ref_groups(_first_present(meta, sched, "block_loss_refs_by_step"), "block_loss_refs_by_step")
    nearby_by_step = _ref_groups(_first_present(meta, sched, "nearby_loss_refs_by_step"), "nearby_loss_refs_by_step")
    prefix_raw = _first_present(meta, sched, "prefix_loss_refs_by_step", [[] for _ in range(inner_K)])
    prefix_by_step = _ref_groups(prefix_raw, "prefix_loss_refs_by_step")
    query_refs = tuple(ImageRef.from_raw(ref) for ref in list(_first_present(meta, sched, "query_label_refs", []) or []))
    aux_refs = tuple(ImageRef.from_raw(ref) for ref in list(_first_present(meta, sched, "aux_loss_refs", []) or []))
    _require_phase_a_no_legacy_roles(prefix_by_step=prefix_by_step, query_refs=query_refs, aux_refs=aux_refs)

    steps_raw = list(_field(sched, "steps", []) or [])
    steps = []
    for idx in range(inner_K):
        raw_step = steps_raw[idx] if idx < len(steps_raw) else None
        steps.append(
            RolloutStep(
                step_idx=int(_field(raw_step, "step_idx", idx)),
                evidence_refs=tuple(evidence_by_step[idx]),
                block_loss_refs=tuple(block_by_step[idx]),
                nearby_loss_refs=tuple(nearby_by_step[idx]),
            )
        )

    first_step = steps_raw[0] if steps_raw else None
    source_keyframe_idx: Optional[int] = None
    if _field(first_step, "source_keyframe_idx", None) is not None:
        source_keyframe_idx = int(_field(first_step, "source_keyframe_idx"))
    block_idx: Optional[int] = None
    if _field(first_step, "block_idx", None) is not None:
        block_idx = int(_field(first_step, "block_idx"))

    num_cams = int(_first_present(meta, sched, "num_cams", 0) or 0)
    if num_cams <= 0:
        num_cams = _infer_num_cams(evidence_by_step, block_by_step, nearby_by_step)
    episode_id = int(_first_present(meta, sched, "episode_id", _first_present(meta, sched, "episode_idx_global", -1)) or -1)
    plan = PhaseALocalUnrollPlan(
        protocol_version=PHASE_A_PROTOCOL_VERSION,
        phase=PHASE_A_NAME,
        scene_id=int(_first_present(meta, sched, "scene_id", batch.get("scene_id", -1)) or -1),
        segment_id=int(_first_present(meta, sched, "segment_id", batch.get("segment_id", -1)) or -1),
        episode_id=episode_id,
        num_cams=num_cams,
        inner_K=inner_K,
        steps=tuple(steps),
        source_keyframe_idx=source_keyframe_idx,
        block_idx=block_idx,
        meta={
            "legacy_scheduler_version": "v9",
            "episode_start_keyframe_pos": int(_first_present(meta, sched, "episode_start_keyframe_pos", -1) or -1),
        },
    )
    validate_phase_a_plan(plan)
    return plan


class LegacyV9PhaseASchedulerAdapter:
    """Expose a RolloutPlan-first Phase A batch path over the legacy V9 scheduler."""

    def __init__(self, v9_scheduler: Any, *, assembler: Optional[PhaseAImageRefBatchAssembler] = None):
        if str(getattr(v9_scheduler, "phase", "")) != PHASE_A_NAME:
            raise ValueError("LegacyV9PhaseASchedulerAdapter only supports phase_A_block_local_unroll")
        if not hasattr(v9_scheduler, "next_batch_with_v9_plan_materializer"):
            raise ValueError("TrainSchedulerV9 must provide next_batch_with_v9_plan_materializer")
        self.v9_scheduler = v9_scheduler
        self.assembler = assembler if assembler is not None else PhaseAImageRefBatchAssembler(v9_scheduler.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.v9_scheduler, name)

    def _materialize_v9_phase_a_plan(self, v9_plan: Any) -> Dict[str, Any]:
        plan = convert_v9_phase_a_plan(v9_plan)
        return self.assembler.materialize(plan, include_test=bool(getattr(self.v9_scheduler, "include_test", False)))

    def next_plan(self) -> PhaseALocalUnrollPlan:
        self.v9_scheduler._ensure_episode_state()
        st = self.v9_scheduler.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV9 internal state is not initialized")
        v9_plan = self.v9_scheduler._plan_from_state(st)
        self.v9_scheduler._validate_v9_plan(v9_plan)
        return convert_v9_phase_a_plan(v9_plan)

    def next_batch(self) -> Dict[str, Any]:
        return self.v9_scheduler.next_batch_with_v9_plan_materializer(
            self._materialize_v9_phase_a_plan,
            merge_plan_request_meta=False,
        )
