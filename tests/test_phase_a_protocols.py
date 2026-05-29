import pytest

from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.roles import Role
from streetforward_core.protocols.rollout import PHASE_A_NAME, PHASE_A_PROTOCOL_VERSION, PhaseALocalUnrollPlan, RolloutStep
from streetforward_core.protocols.validators import validate_phase_a_plan


def _plan(**overrides):
    values = {
        "protocol_version": PHASE_A_PROTOCOL_VERSION,
        "phase": PHASE_A_NAME,
        "scene_id": 1,
        "segment_id": 2,
        "episode_id": 3,
        "num_cams": 2,
        "inner_K": 2,
        "steps": (
            RolloutStep(
                step_idx=0,
                evidence_refs=(ImageRef(10, 0),),
                block_loss_refs=(ImageRef(10, 0),),
            ),
            RolloutStep(
                step_idx=1,
                evidence_refs=(ImageRef(10, 0),),
                block_loss_refs=(ImageRef(10, 0),),
                nearby_loss_refs=(ImageRef(11, 0),),
            ),
        ),
    }
    values.update(overrides)
    return PhaseALocalUnrollPlan(**values)


def test_role_is_python38_compatible_str_enum():
    assert issubclass(Role, str)
    assert Role.EVIDENCE.value == "evidence"


def test_validate_phase_a_plan_rejects_shape_errors():
    validate_phase_a_plan(_plan())
    with pytest.raises(ValueError, match="len\\(steps\\)"):
        validate_phase_a_plan(_plan(inner_K=3))
    with pytest.raises(ValueError, match="requires evidence_refs"):
        validate_phase_a_plan(
            _plan(
                steps=(
                    RolloutStep(step_idx=0, evidence_refs=(), block_loss_refs=(ImageRef(10, 0),)),
                    RolloutStep(step_idx=1, evidence_refs=(ImageRef(10, 0),), block_loss_refs=(ImageRef(10, 0),)),
                )
            )
        )


def test_validate_phase_a_plan_rejects_nearby_evidence_leakage():
    with pytest.raises(ValueError, match="nearby_loss_refs leaked"):
        validate_phase_a_plan(
            _plan(
                steps=(
                    RolloutStep(step_idx=0, evidence_refs=(ImageRef(10, 0),), block_loss_refs=(ImageRef(10, 0),)),
                    RolloutStep(
                        step_idx=1,
                        evidence_refs=(ImageRef(10, 0),),
                        block_loss_refs=(ImageRef(10, 0),),
                        nearby_loss_refs=(ImageRef(10, 0),),
                    ),
                )
            )
        )

