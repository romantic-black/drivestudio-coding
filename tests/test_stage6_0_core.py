from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch

from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.stage6_bridge import TeacherPriorAdapter
from models.streetforward.stage6_losses import aggregate_stage6_total_loss
from models.streetforward.stage6_student import apply_student_valid_mask
from models.streetforward.stage6_teacher import build_teacher_observe_input
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


def test_stage6_student_valid_mask_preserves_nhwc_layout() -> None:
    render = torch.ones((2, 4, 5, 3), dtype=torch.float32)
    prior = torch.ones((2, 4, 5, 32), dtype=torch.float32)
    conf = torch.ones((2, 4, 5, 1), dtype=torch.float32)
    valid = torch.zeros((2, 4, 5, 1), dtype=torch.float32)
    valid[:, :2, :, :] = 1.0

    out = apply_student_valid_mask(
        render_rgb=render,
        prior_map=prior,
        prior_conf=conf,
        valid_mask=valid,
        append_as_channel=True,
    )

    assert out.render_rgb.shape == render.shape
    assert out.prior_map.shape == prior.shape
    assert out.prior_conf is not None
    assert out.prior_conf.shape == conf.shape
    assert out.extra_inputs is not None
    assert out.extra_inputs.shape == valid.shape
    assert float(out.render_rgb[:, 2:, :, :].sum().item()) == 0.0


def test_stage6_student_valid_mask_supports_mixed_layouts() -> None:
    render = torch.ones((2, 3, 4, 5), dtype=torch.float32)
    prior = torch.ones((2, 4, 5, 32), dtype=torch.float32)
    conf = torch.ones((2, 1, 4, 5), dtype=torch.float32)
    valid = torch.zeros((2, 4, 5, 1), dtype=torch.float32)
    valid[:, :2, :, :] = 1.0

    out = apply_student_valid_mask(
        render_rgb=render,
        prior_map=prior,
        prior_conf=conf,
        valid_mask=valid,
        append_as_channel=True,
        prior_dim=32,
    )

    assert out.render_rgb.shape == render.shape
    assert out.prior_map.shape == prior.shape
    assert out.prior_conf is not None
    assert out.prior_conf.shape == conf.shape
    assert out.extra_inputs is not None
    assert out.extra_inputs.shape == (2, 1, 4, 5)
    assert float(out.render_rgb[:, :, 2:, :].sum().item()) == 0.0
    assert float(out.prior_map[:, 2:, :, :].sum().item()) == 0.0
    assert float(out.prior_conf[:, :, 2:, :].sum().item()) == 0.0


def test_stage6_teacher_observe_input_supports_nchw_and_nhwc() -> None:
    gt_nchw = torch.zeros((1, 3, 4, 5))
    render_nchw = torch.ones((1, 3, 4, 5))
    assert build_teacher_observe_input(gt_rgb=gt_nchw, render_rgb=render_nchw, use_gt=True).shape == (1, 6, 4, 5)

    gt_nhwc = torch.zeros((1, 4, 5, 3))
    render_nhwc = torch.ones((1, 4, 5, 3))
    assert build_teacher_observe_input(gt_rgb=gt_nhwc, render_rgb=render_nhwc, use_gt=True).shape == (1, 4, 5, 6)


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
    stage._stage6_domain_loss_weights = {
        "teacher_source": 1.0,
        "student_source": 1.0,
        "teacher_anchor": 0.05,
        "history_visited": 0.0,
        "probe_near": 0.0,
    }
    assert stage._target_role_weight("teacher_source", 0) == 1.0
    assert stage._target_role_weight("teacher_anchor", 0) == 0.05
    assert stage._target_role_weight("history_visited", 0) == 0.0
    assert stage._target_role_weight("probe_near", 0) == 0.0


def test_stage6_build_target_view_weights_uses_stage6_loss_domains() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    stage.device = torch.device("cpu")
    stage._stage6_domain_loss_weights = {
        "teacher_source": 0.2,
        "student_source": 1.0,
        "teacher_anchor": 0.05,
        "history_visited": 0.0,
        "probe_near": 0.0,
    }
    batch = {
        "request_meta": {
            "scheduler_version": "v10",
            "train_target_image_refs": [(1, 0), (2, 0), (3, 0)],
            "train_target_image_roles": ["student_source", "teacher_anchor", "history_visited"],
            "train_target_image_loss_base_weights": [1.0, 1.0, 1.0],
        }
    }

    weights, roles = stage._build_target_view_weights(batch, step=0, num_targets=3)

    assert roles == ["student_source", "teacher_anchor", "history_visited"]
    assert torch.allclose(weights, torch.tensor([1.0, 0.05, 0.0]))


def test_stage6_prior_splat_detaches_geometry_and_opacity_by_default() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    stage.device = torch.device("cpu")
    stage.stage6_prior_dim = 4
    stage.stage6_prior_conf_norm = 1.0
    stage.stage6_prior_eps = 1.0e-6
    stage.stage6_live_detach_geometry = True
    stage.stage6_live_detach_opacity = True
    seen = {}

    def _renderer(**kwargs):
        for key in ("means", "quats", "scales", "opacities", "colors"):
            seen[key] = kwargs[key]
        colors = kwargs["colors"]
        v = int(kwargs["viewmats"].shape[0])
        h = int(kwargs["height"])
        w = int(kwargs["width"])
        rendered = colors.mean(dim=0).view(1, 1, 1, -1).expand(v, h, w, -1)
        return rendered, None, None

    stage.renderer = _renderer
    num = 3
    feat_bg = torch.randn(num, 4, requires_grad=True)
    support_bg = torch.ones(num)
    valid_bg = torch.ones(num, dtype=torch.bool)
    gaussians = {
        "means": torch.randn(num, 3, requires_grad=True),
        "quats": torch.randn(num, 4, requires_grad=True),
        "scales": torch.randn(num, 3, requires_grad=True),
        "opacities": torch.randn(num, requires_grad=True),
    }
    view = SimpleNamespace(camtoworlds=torch.eye(4), K=torch.eye(3))

    prior_map, conf_map = stage._render_prior_from_components(
        feat_bg=feat_bg,
        support_bg=support_bg,
        valid_bg=valid_bg,
        feat_distant=None,
        support_distant=None,
        valid_distant=None,
        feat_rigid_s=None,
        support_rigid_s=None,
        valid_rigid_s=None,
        gaussians_scene=gaussians,
        source_views=[view],
        height=2,
        width=2,
    )

    assert prior_map.requires_grad
    assert conf_map.requires_grad
    assert seen["colors"].requires_grad
    assert not seen["means"].requires_grad
    assert not seen["quats"].requires_grad
    assert not seen["scales"].requires_grad
    assert not seen["opacities"].requires_grad


def test_stage6_live_bridge_required_fast_fails_without_live_inputs() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    stage.stage6_live_enable = True
    stage.stage6_live_require_on_student = True
    with pytest.raises(RuntimeError, match="live bridge required"):
        stage._enforce_live_bridge_requirement(role="student", live_used=False)
    stage._enforce_live_bridge_requirement(role="student", live_used=True)
    stage._enforce_live_bridge_requirement(role="teacher", live_used=False)


def test_stage6_forward_keeps_parent_loss_and_logs_actual_role() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    stage._stage6_role_fallback = True
    stage._stage6_last_role = "teacher"
    loss = torch.tensor(5.0)
    batch = {
        "request_meta": {
            "scheduler_version": "v10",
            "stage6_role": "student",
            "probe_target_image_refs": [(1, 0)],
            "target_image_roles": ["student_source", "probe_near"],
            "target_image_loss_base_weights": [1.0, 0.0],
        }
    }
    with patch.object(MinimalStreetForwardStage5_4, "forward", return_value={"loss": loss}):
        out = stage.forward(batch)

    assert out["loss"] is loss
    assert out["stage6_0/role_requested_student"] == 1.0
    assert out["stage6_0/role_actual_teacher"] == 1.0
    assert out["stage6_0/role_fallback"] == 1.0
    assert out["scheduler/v10_is_compat_v9"] == 1.0
    assert out["probe/near/num_targets"] == 1.0
    assert out["loss/probe_near_weight_sum"] == 0.0


def test_stage6_train_step_renames_new_and_legacy_role_metrics() -> None:
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    parent_out = {
        "loss": torch.tensor(1.0),
        "monitor/l1/teacher_anchor": 0.2,
        "monitor/psnr/teacher_preserve": 30.0,
        "monitor/l1/probe_near": 0.3,
        "monitor/psnr/near_random": 28.0,
    }
    with patch.object(MinimalStreetForwardStage5_4, "train_step", return_value=dict(parent_out)):
        out = stage.train_step({})

    assert out["monitor/l1/teacher_anchor"] == 0.2
    assert out["loss/teacher_anchor"] == 0.2
    assert out["monitor/psnr/teacher_anchor"] == 30.0
    assert out["probe/near/l1"] == 0.3
    assert out["probe/near/psnr"] == 28.0
    assert out["loss/total_train"] == 1.0
