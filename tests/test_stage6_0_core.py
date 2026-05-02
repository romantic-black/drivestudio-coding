from __future__ import annotations

import torch

from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.stage6_bridge import TeacherPriorAdapter
from models.streetforward.stage6_losses import aggregate_stage6_total_loss
from models.feature_extractors.student_prior_fusion_unet import StudentPriorFusionUNet


def test_stage6_teacher_prior_adapter_shape_stable() -> None:
    adapter = TeacherPriorAdapter(dim=32)
    x = torch.randn(2, 32, 8, 8)
    y = adapter(x)
    assert tuple(y.shape) == tuple(x.shape)


def test_stage6_student_unet_accepts_valid_mask_extra_channel() -> None:
    net = StudentPriorFusionUNet(
        prior_dim=32,
        out_dim=32,
        base_dim=32,
        use_confidence=True,
        extra_input_channels=1,
    )
    render = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    prior = torch.full((1, 32, 4, 4), 2.0, dtype=torch.float32)
    conf = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    valid = torch.zeros((1, 1, 4, 4), dtype=torch.float32) + 1.0
    out = net(
        render_rgb=render,
        prior_map=prior,
        prior_conf=conf,
        extra_inputs=valid,
    )
    assert out.shape == (1, 4, 4, 32)


def test_stage6_aggregate_loss_excludes_probe() -> None:
    z = torch.zeros((), dtype=torch.float32)
    out = aggregate_stage6_total_loss(
        self_teacher=z + 1.0,
        self_student=z + 2.0,
        teacher_anchor=z + 3.0,
        history=z + 4.0,
        w_self_teacher=0.2,
        w_self_student=1.0,
        w_teacher_anchor=0.1,
        w_history=0.1,
    )
    assert torch.is_tensor(out.total_train)
    assert abs(float(out.total_train.detach().item()) - 2.9) < 1.0e-6
    assert abs(float(out.self_loss["teacher"].detach().item()) - 1.0) < 1.0e-6
    assert abs(float(out.self_loss["student"].detach().item()) - 2.0) < 1.0e-6


def test_stage6_target_role_weight_mapping() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    stage._scheduler_v10_target_weights = {
        "teacher_source": 1.0,
        "student_source": 1.0,
        "teacher_anchor": 0.1,
        "history_visited": 0.2,
        "probe_near": 0.0,
    }
    assert stage._target_role_weight("teacher_source", 0) == 1.0
    assert stage._target_role_weight("teacher_anchor", 0) == 0.1
    assert stage._target_role_weight("history_visited", 0) == 0.2
    assert stage._target_role_weight("probe_near", 0) == 0.0

