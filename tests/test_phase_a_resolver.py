import pytest

from streetforward_core.data.resolvers.phase_a_resolver import PhaseABatchResolver
from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.roles import Role
from streetforward_core.protocols.rollout import PHASE_A_NAME, PHASE_A_PROTOCOL_VERSION, PhaseALocalUnrollPlan, RolloutStep


def _plan():
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
    )


def _batch(plan=None):
    plan = plan if plan is not None else _plan()
    return {
        "rollout_plan": plan,
        "request_meta": {
            "scheduler_version": "phase_a_core_v1",
            "scheduler_phase": PHASE_A_NAME,
            "assembly_mode": "image_ref_v9",
            "scene_id": 1,
            "segment_id": 2,
            "episode_id": 3,
            "num_cams": 2,
            "inner_K": 2,
            "source_image_refs": [(10, 0), (10, 1)],
            "target_image_refs": [(10, 0), (10, 1), (11, 0)],
            "target_image_roles": [Role.BLOCK_LOSS.value, Role.BLOCK_LOSS.value, Role.NEARBY_LOSS.value],
            "evidence_refs_by_step": [[(10, 0), (10, 1)], [(10, 0), (10, 1)]],
            "block_loss_refs_by_step": [[(10, 0), (10, 1)], [(10, 0), (10, 1)]],
            "nearby_loss_refs_by_step": [[], [(11, 0)]],
            "prefix_loss_refs_by_step": [[], []],
            "query_label_refs": [],
            "aux_loss_refs": [],
        },
    }


def test_phase_a_resolver_resolves_rollout_plan_batch():
    resolved = PhaseABatchResolver().resolve(_batch())
    assert resolved.inner_K == 2
    assert resolved.evidence_source_indices_by_step == ((0, 1), (0, 1))
    assert resolved.block_target_indices_by_step == ((0, 1), (0, 1))
    assert resolved.nearby_target_indices_by_step == ((), (2,))


def test_phase_a_resolver_rejects_rollout_plan_request_meta_mismatch():
    bad = _batch()
    bad["request_meta"] = dict(bad["request_meta"])
    bad["request_meta"]["nearby_loss_refs_by_step"] = [[], [(12, 0)]]
    with pytest.raises(ValueError, match="nearby_loss mismatch at k=1"):
        PhaseABatchResolver().resolve(bad)


def test_phase_a_resolver_rejects_inner_k_drift():
    bad = _batch()
    bad["request_meta"] = dict(bad["request_meta"])
    bad["request_meta"]["inner_K"] = 3
    with pytest.raises(ValueError, match="inner_K disagrees"):
        PhaseABatchResolver().resolve(bad)


def test_phase_a_resolver_rejects_target_role_mismatch():
    bad = _batch()
    bad["request_meta"] = dict(bad["request_meta"])
    bad["request_meta"]["target_image_roles"] = [
        Role.NEARBY_LOSS.value,
        Role.BLOCK_LOSS.value,
        Role.NEARBY_LOSS.value,
    ]
    with pytest.raises(ValueError, match="mapped to target role"):
        PhaseABatchResolver().resolve(bad)

