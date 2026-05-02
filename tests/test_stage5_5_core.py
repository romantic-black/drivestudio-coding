from __future__ import annotations

import torch

from models.streetforward.minimal_trainer_stage4_6 import RigidRoute
from models.streetforward.minimal_trainer_stage5_5 import MinimalStreetForwardStage5_5
from models.streetforward.teacher_student_prior import create_teacher_prior_cache


def _dummy_route(device: torch.device) -> RigidRoute:
    s = torch.tensor([0, 2], dtype=torch.long, device=device)
    return RigidRoute(
        S=s,
        S_in=s,
        S_out=torch.zeros((0,), dtype=torch.long, device=device),
        inside_mask_S=torch.ones((2,), dtype=torch.bool, device=device),
        route_inside_global=torch.tensor([True, False, True], dtype=torch.bool, device=device),
        means_world_S=torch.zeros((2, 3), dtype=torch.float32, device=device),
        quats_world_S=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
    )


def test_stage5_5_prior_available_from_bg() -> None:
    device = torch.device("cpu")
    cache = create_teacher_prior_cache(
        num_bg=4,
        num_distant=0,
        num_rigid=3,
        feat_dim=32,
        device=device,
        dtype=torch.float32,
    )
    route = _dummy_route(device)
    assert not MinimalStreetForwardStage5_5._teacher_prior_available(cache=cache, route=route)
    cache.bg.valid[1] = True
    assert MinimalStreetForwardStage5_5._teacher_prior_available(cache=cache, route=route)


def test_stage5_5_prior_available_from_rigid_subset() -> None:
    device = torch.device("cpu")
    cache = create_teacher_prior_cache(
        num_bg=1,
        num_distant=1,
        num_rigid=3,
        feat_dim=32,
        device=device,
        dtype=torch.float32,
    )
    route = _dummy_route(device)
    cache.rigid.valid[1] = True
    assert not MinimalStreetForwardStage5_5._teacher_prior_available(cache=cache, route=route)
    cache.rigid.valid[2] = True
    assert MinimalStreetForwardStage5_5._teacher_prior_available(cache=cache, route=route)


def test_stage5_5_build_prior_confidence_log1p_norm() -> None:
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    stage.stage5_5_prior_conf_norm = 1.0
    support = torch.tensor([0.0, 1.0, 9.0], dtype=torch.float32)
    valid = torch.tensor([True, True, False])
    conf = stage._build_prior_confidence(support=support, valid=valid)
    assert conf.shape == (3, 1)
    assert float(conf[0].item()) == 0.0
    assert float(conf[1].item()) > 0.0
    assert float(conf[2].item()) == 0.0


def test_stage5_5_target_role_weight_mapping() -> None:
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    stage._target_view_weight_cfg = {
        "source_weight": 1.0,
        "visited_weight": 0.7,
        "near_random_weight": 0.3,
        "near_random_schedule_enable": False,
    }
    stage._scheduler_v9_target_weights = {
        "teacher_source": 1.0,
        "student_source": 1.0,
        "teacher_preserve": 0.3,
        "visited": 0.2,
        "near_random": 0.2,
    }
    assert stage._target_role_weight("teacher_source", 0) == 1.0
    assert stage._target_role_weight("student_source", 0) == 1.0
    assert stage._target_role_weight("teacher_preserve", 0) == 0.3
    assert stage._target_role_weight("visited", 0) == 0.2
    assert stage._target_role_weight("near_random", 0) == 0.2
