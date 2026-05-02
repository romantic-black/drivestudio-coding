from __future__ import annotations

import pytest

from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0


def _base_cfg():
    return {
        "scheduler_v10": {"enable": True},
        "stage6_0": {
            "enable": True,
            "phase": "default",
            "student": {"valid_mask": {"apply_before_unet": True, "append_as_channel": True}},
            "bridge": {
                "live": {"enable": True, "rerun_teacher_2d_current_step": True, "student_loss_to_teacher_backbone": False},
                "cache": {"detach_write": True},
            },
        },
        "losses": {"stage6_0": {"probe": {"near": {"loss_weight": 0.0}}}},
    }


def test_stage6_fast_fail_forbidden_preserve_key() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    cfg = _base_cfg()
    cfg["stage6_0"]["teacher_preserve"] = {"enable": True}
    with pytest.raises(ValueError, match="forbids key"):
        stage._validate_stage6_0_config(cfg)


def test_stage6_fast_fail_live_without_rerun() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    cfg = _base_cfg()
    cfg["stage6_0"]["bridge"]["live"]["rerun_teacher_2d_current_step"] = False
    with pytest.raises(ValueError, match="rerun_teacher_2d_current_step"):
        stage._validate_stage6_0_config(cfg)


def test_stage6_fast_fail_near_loss_nonzero_without_explicit_phase() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    cfg = _base_cfg()
    cfg["losses"]["stage6_0"]["probe"]["near"]["loss_weight"] = 0.2
    with pytest.raises(ValueError, match="probe.near.loss_weight=0"):
        stage._validate_stage6_0_config(cfg)

