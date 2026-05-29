from types import SimpleNamespace

import pytest

from streetforward_core.data.assemblers.phase_a_image_ref_batch_assembler import PhaseAImageRefBatchAssembler
from streetforward_core.data.assemblers.v9_image_ref_batch_assembler import V9ImageRefBatchAssembler
from streetforward_core.data.schedulers.legacy_v9_phase_a_adapter import LegacyV9PhaseASchedulerAdapter
from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.roles import Role
from streetforward_core.protocols.rollout import PHASE_A_NAME, PHASE_A_PROTOCOL_VERSION, PhaseALocalUnrollPlan, RolloutStep


def _phase_a_plan():
    return PhaseALocalUnrollPlan(
        protocol_version=PHASE_A_PROTOCOL_VERSION,
        phase=PHASE_A_NAME,
        scene_id=1,
        segment_id=2,
        episode_id=3,
        num_cams=2,
        inner_K=2,
        steps=(
            RolloutStep(
                step_idx=0,
                evidence_refs=(ImageRef(10, 0), ImageRef(10, 1)),
                block_loss_refs=(ImageRef(10, 0), ImageRef(10, 1)),
            ),
            RolloutStep(
                step_idx=1,
                evidence_refs=(ImageRef(10, 0), ImageRef(10, 1)),
                block_loss_refs=(ImageRef(10, 0), ImageRef(10, 1)),
                nearby_loss_refs=(ImageRef(11, 0),),
            ),
        ),
        meta={"legacy_scheduler_version": "v9"},
    )


class _Dataset:
    def __init__(self):
        self.image_ref_calls = []
        self.v9_calls = []

    def _assemble_segment_batch_from_image_refs(self, *args, **kwargs):
        self.image_ref_calls.append((args, kwargs))
        source_refs = list(args[2])
        target_refs = list(args[3])
        return {
            "scene_id": int(args[0]),
            "segment_id": int(args[1]),
            "source": {"frame_indices": [r[0] for r in source_refs], "cam_indices": [r[1] for r in source_refs]},
            "target": {"frame_indices": [r[0] for r in target_refs], "cam_indices": [r[1] for r in target_refs]},
            "request_meta": {"dataset_meta": True},
        }

    def _assemble_segment_batch_from_v9_request(self, **kwargs):
        self.v9_calls.append(kwargs)
        raise AssertionError("Phase A assembler must not call V9 request assembly")


def _v9_plan():
    step0 = SimpleNamespace(
        step_idx=0,
        source_keyframe_idx=5,
        source_frame_idx=10,
        block_idx=0,
    )
    step1 = SimpleNamespace(
        step_idx=1,
        source_keyframe_idx=5,
        source_frame_idx=10,
        block_idx=0,
    )
    return SimpleNamespace(
        scheduler_version="v9",
        phase=PHASE_A_NAME,
        scene_id=1,
        segment_id=2,
        episode_id=3,
        episode_start_keyframe_pos=0,
        keyframe_window=[5],
        frame_chain=[10],
        num_cams=2,
        inner_K=2,
        steps=[step0, step1],
        evidence_refs_by_step=[[(10, 0), (10, 1)], [(10, 0), (10, 1)]],
        block_loss_refs_by_step=[[(10, 0), (10, 1)], [(10, 0), (10, 1)]],
        nearby_loss_refs_by_step=[[], [(11, 0)]],
        prefix_loss_refs_by_step=[[], []],
        query_label_refs=[],
        aux_loss_refs=[],
        request_meta={},
    )


def test_phase_a_image_ref_assembler_materializes_from_rollout_plan():
    dataset = _Dataset()
    plan = _phase_a_plan()
    batch = PhaseAImageRefBatchAssembler(dataset).materialize(plan)
    assert len(dataset.image_ref_calls) == 1
    assert dataset.v9_calls == []
    args, kwargs = dataset.image_ref_calls[0]
    assert args[2] == [(10, 0), (10, 1)]
    assert args[3] == [(10, 0), (10, 1), (11, 0)]
    assert kwargs["enforce_target0_equals_source"] is False
    assert batch["rollout_plan"] is plan
    assert batch["request_meta"]["scheduler_version"] == "phase_a_core_v1"
    assert batch["request_meta"]["target_image_roles"] == [
        Role.BLOCK_LOSS.value,
        Role.BLOCK_LOSS.value,
        Role.NEARBY_LOSS.value,
    ]


def test_v9_image_ref_assembler_rejects_phase_a_rollout_plan():
    with pytest.raises(ValueError, match="PhaseALocalUnrollPlan"):
        V9ImageRefBatchAssembler(_Dataset()).materialize(_phase_a_plan())


def test_legacy_v9_phase_a_scheduler_adapter_uses_phase_a_assembler():
    dataset = _Dataset()

    class Scheduler:
        phase = PHASE_A_NAME
        include_test = False
        steps_per_block = 1

        def __init__(self):
            self.dataset = dataset
            self.merge_flags = []

        def next_batch_with_v9_plan_materializer(self, materialize_plan, *, merge_plan_request_meta=True):
            self.merge_flags.append(bool(merge_plan_request_meta))
            return materialize_plan(_v9_plan())

    scheduler = Scheduler()
    batch = LegacyV9PhaseASchedulerAdapter(scheduler).next_batch()
    assert scheduler.merge_flags == [False]
    assert batch["rollout_plan"].phase == PHASE_A_NAME
    assert batch["request_meta"]["scheduler_version"] == "phase_a_core_v1"
    assert len(dataset.image_ref_calls) == 1
    assert dataset.v9_calls == []

