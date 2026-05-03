from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

sys.modules.setdefault("open3d", types.SimpleNamespace())

from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_2 import MinimalStreetForwardStage4_2
from models.streetforward.minimal_trainer_stage5_5 import FeatureSplatUncertaintyHeadV3, MinimalStreetForwardStage5_5
from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format


def test_stage5_5_collect_aux_targets_fast_fails_when_refs_not_materialized():
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    stage.stage5_5_aux_enabled = True
    stage.stage5_5_target_max_targets = 1

    with pytest.raises(RuntimeError, match=r"batch\['aux_targets'\] is missing or empty"):
        stage._collect_aux_targets({"request_meta": {"aux_image_refs": [(10, 0)]}})


def test_stage5_5_convert_batch_materializes_aux_targets():
    def role(num: int, frame0: int) -> dict:
        return {
            "image": torch.zeros((num, 2, 2, 3), dtype=torch.float32),
            "extrinsics": torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num, 1, 1),
            "intrinsics": torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num, 1, 1),
            "depth": torch.ones((num, 2, 2), dtype=torch.float32),
            "frame_indices": torch.arange(frame0, frame0 + num, dtype=torch.long),
            "cam_indices": torch.zeros((num,), dtype=torch.long),
            "sky_mask": torch.zeros((num, 2, 2), dtype=torch.float32),
            "egocar_mask": torch.zeros((num, 2, 2), dtype=torch.float32),
            "dynamic_mask": torch.ones((num, 2, 2), dtype=torch.float32),
            "viewdirs": torch.zeros((num, 2, 2, 3), dtype=torch.float32),
        }

    out = convert_batch_to_minimal_format(
        {
            "pointcloud": {"background": np.zeros((0, 6), dtype=np.float32)},
            "target": role(1, 10),
            "aux_target": role(2, 20),
            "request_meta": {"aux_image_refs": [(20, 0), (21, 0)]},
        },
        device=torch.device("cpu"),
        num_targets=1,
        include_source_for_2d=False,
    )

    assert len(out["targets"]) == 1
    assert len(out["aux_targets"]) == 2
    assert out["request_meta"]["aux_image_refs"] == [(20, 0), (21, 0)]
    assert "dynamic_mask" in out["aux_targets"][0]
    assert torch.allclose(out["aux_targets"][0]["dynamic_mask"], torch.ones((2, 2)))


def test_stage5_5_optimizer_fast_fail_detects_missing_aux_head_params():
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    torch.nn.Module.__init__(stage)
    stage.stage5_5_uncertainty_head = torch.nn.Linear(2, 1)
    unrelated = torch.nn.Parameter(torch.zeros(()))
    stage.optimizer = torch.optim.Adam([unrelated], lr=1.0e-3)

    with pytest.raises(RuntimeError, match="aux head parameters are not in optimizer"):
        stage._debug_check_stage5_5_optimizer_contains_aux_head()


def test_stage5_5_v3_head_predicts_error_and_rgb_residual_with_bounds():
    head = FeatureSplatUncertaintyHeadV3(
        in_ch=4,
        hidden_dim=8,
        error_max=0.5,
        residual_max=0.25,
        predict_rgb_residual=True,
    )

    e_pred, r_pred = head(torch.zeros((2, 4, 3, 5), dtype=torch.float32))

    assert tuple(e_pred.shape) == (2, 1, 3, 5)
    assert tuple(r_pred.shape) == (2, 3, 3, 5)
    assert float(e_pred.min().item()) >= 0.0
    assert float(e_pred.max().item()) <= 0.5
    assert float(r_pred.abs().max().item()) <= 0.25


def test_stage5_5_aux_loss_mask_accepts_channel_first_sky_mask():
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    stage.device = torch.device("cpu")
    stage.stage5_5_loss_mask_egocar = True
    stage.stage5_5_loss_mask_sky = True
    stage.require_sky_mask_for_loss = True

    target = {
        "sky_mask": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
        "egocar_mask": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
    }
    mask = stage._build_aux_loss_mask(target, 2, 2)

    assert tuple(mask.shape) == (2, 2)
    assert mask.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_stage5_5_feature_splat_renders_all_channels_once_and_keeps_feature_grad():
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    stage.device = torch.device("cpu")
    stage.stage5_5_splat_eps = 1.0e-6
    calls = {"count": 0}

    def renderer(**kwargs):
        calls["count"] += 1
        assert kwargs["viewmats"].device == kwargs["means"].device
        assert kwargs["viewmats"].dtype == kwargs["means"].dtype
        assert kwargs["Ks"].device == kwargs["means"].device
        assert kwargs["Ks"].dtype == kwargs["means"].dtype
        assert int(kwargs["means"].shape[0]) == 3
        colors = kwargs["colors"]
        assert tuple(colors.shape) == (3, 6)
        assert torch.allclose(colors[1], torch.zeros_like(colors[1]))
        assert colors[:, -1].tolist() == [1.0, 0.0, 1.0]
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        render = colors.sum(dim=0).view(1, 1, 1, -1).expand(1, height, width, colors.shape[-1])
        alpha = torch.full((1, height, width, 1), 2.0, dtype=colors.dtype, device=colors.device)
        return render, alpha, {}

    stage.renderer = renderer
    view = SimpleNamespace(camtoworlds=torch.eye(4, dtype=torch.float32), Ks=torch.eye(3, dtype=torch.float32).unsqueeze(0))
    render_params = {
        "means_r": torch.zeros((3, 3), dtype=torch.float64),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(3, 1),
        "scales_r": torch.ones((3, 3), dtype=torch.float64) * 0.1,
        "opacities_r": torch.ones((3,), dtype=torch.float64),
    }
    node_features = torch.arange(15, dtype=torch.float64).reshape(3, 5).requires_grad_(True)
    node_mask = torch.tensor([True, False, True])

    feat, support = stage._splat_node_features_to_view(
        render_params=render_params,
        node_features=node_features,
        node_mask=node_mask,
        view=view,
        height=2,
        width=3,
        detach_geometry=True,
        detach_weights=True,
    )

    assert calls["count"] == 1
    assert tuple(feat.shape) == (2, 3, 5)
    assert tuple(support.shape) == (2, 3)
    assert torch.allclose(support, torch.full((2, 3), 2.0, dtype=torch.float64))
    assert torch.allclose(feat[0, 0], (node_features.detach()[0] + node_features.detach()[2]) / 2.0)
    feat.sum().backward()
    assert node_features.grad is not None
    assert float(node_features.grad[0].abs().sum()) > 0.0
    assert float(node_features.grad[1].abs().sum()) == 0.0
    assert float(node_features.grad[2].abs().sum()) > 0.0


def test_stage5_5_feature_splat_fast_fails_detached_weights_without_detached_geometry():
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    stage.device = torch.device("cpu")
    stage.stage5_5_splat_eps = 1.0e-6
    view = SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3).unsqueeze(0))
    render_params = {
        "means_r": torch.zeros((1, 3)),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "scales_r": torch.ones((1, 3)) * 0.1,
        "opacities_r": torch.ones((1,)),
    }

    with pytest.raises(ValueError, match="detach_alpha_weights=true currently requires detach_geometry=true"):
        stage._splat_node_features_to_view(
            render_params=render_params,
            node_features=torch.ones((1, 4)),
            node_mask=torch.ones((1,), dtype=torch.bool),
            view=view,
            height=2,
            width=2,
            detach_geometry=False,
            detach_weights=True,
        )


def _render_params(n: int, *, feat_dim: int = 3) -> dict:
    return {
        "means_r": torch.zeros((n, 3), dtype=torch.float32),
        "scales_r": torch.ones((n, 3), dtype=torch.float32) * 0.1,
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1),
        "opacities_r": torch.ones((n,), dtype=torch.float32),
        "colors_r": torch.zeros((n, 1, feat_dim), dtype=torch.float32),
    }


def _bridge_stage() -> MinimalStreetForwardStage5_5:
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    torch.nn.Module.__init__(stage)
    stage.device = torch.device("cpu")
    stage.require_sky_mask_for_loss = True
    stage.stage5_5_bridge_enabled = True
    stage.stage5_5_bridge_start_after_steps = 0
    stage.stage5_5_bridge_warmup_steps = 0
    stage.stage5_5_bridge_weight = 1.0
    stage.stage5_5_bridge_max_weight = 1.0
    stage.stage5_5_bridge_error_mode = "exp"
    stage.stage5_5_bridge_error_tau = 0.15
    stage.stage5_5_bridge_error_min_conf = 0.0
    stage.stage5_5_bridge_error_max_conf = 1.0
    stage.stage5_5_bridge_support_mode = "soft"
    stage.stage5_5_bridge_support_tau = 1.0e-4
    stage.stage5_5_bridge_support_gamma = 0.5
    stage.stage5_5_bridge_support_hard_min = 1.0e-5
    stage.stage5_5_bridge_mask_use_valid_loss_mask = True
    stage.stage5_5_bridge_mask_sky = True
    stage.stage5_5_bridge_mask_egocar = True
    stage.stage5_5_bridge_mask_dynamic = False
    stage.stage5_5_bridge_require_render_alpha = True
    stage.stage5_5_bridge_render_alpha_min = 0.02
    stage.stage5_5_bridge_min_effective_pixel_ratio = 0.0
    stage.stage5_5_bridge_loss_type = "l1"
    stage.stage5_5_bridge_normalize_by_weight_sum = True
    stage.stage5_5_bridge_detach_confidence = True
    stage.stage5_5_bridge_detach_mask = True
    stage.stage5_5_bridge_rgb_reduce = "mean"
    stage.stage5_5_bridge_debug_save_maps_interval = 0
    stage.stage5_5_residual_max = 0.5
    stage.stage5_5_loss_rgb_residual_weight = 0.0
    stage.stage5_5_loss_rgb_residual_supported_weight = 0.0
    return stage


def test_stage5_5_bridge_weight_schedule_delays_warmups_and_clamps():
    stage = _bridge_stage()
    stage.stage5_5_bridge_enabled = False
    assert stage._stage5_5_bridge_weight(100) == 0.0

    stage.stage5_5_bridge_enabled = True
    stage.stage5_5_bridge_start_after_steps = 10
    stage.stage5_5_bridge_warmup_steps = 10
    stage.stage5_5_bridge_weight = 0.008
    stage.stage5_5_bridge_max_weight = 0.005

    assert stage._stage5_5_bridge_weight(9) == 0.0
    assert stage._stage5_5_bridge_weight(15) == pytest.approx(0.004)
    assert stage._stage5_5_bridge_weight(25) == pytest.approx(0.005)


def test_stage5_5_bridge_confidence_monotonicity():
    stage = _bridge_stage()

    err_conf = stage._stage5_5_bridge_error_confidence(torch.tensor([[0.03, 0.10, 0.20]]))
    assert float(err_conf[0, 0]) > float(err_conf[0, 1]) > float(err_conf[0, 2])

    support_conf = stage._stage5_5_bridge_support_confidence(torch.tensor([[1.0e-6, 1.0e-4, 1.0e-2]]))
    assert float(support_conf[0, 0]) == 0.0
    assert float(support_conf[0, 1]) < float(support_conf[0, 2])


def test_stage5_5_bridge_mask_combines_valid_sky_ego_dynamic_and_alpha():
    stage = _bridge_stage()
    stage.stage5_5_bridge_mask_dynamic = True
    target = {
        "sky_mask": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
        "egocar_mask": torch.tensor([[[0.0], [0.0]], [[1.0], [0.0]]]),
        "dynamic_mask": torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]),
    }
    alpha = torch.tensor([[[0.5], [0.5]], [[0.5], [0.0]]])

    mask = stage._build_stage5_5_bridge_mask(target, 2, 2, alpha)

    assert tuple(mask.shape) == (2, 2)
    assert mask.tolist() == [[1.0, 0.0], [0.0, 0.0]]


class _BiasHead(torch.nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return self.bias.reshape(1, 1, 1, 1).expand(int(x.shape[0]), 1, int(x.shape[2]), int(x.shape[3]))


class _DualBiasHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.err_bias = torch.nn.Parameter(torch.zeros(()))
        self.rgb_bias = torch.nn.Parameter(torch.zeros((3,), dtype=torch.float32))

    def forward(self, x):
        b, _c, h, w = x.shape
        e = self.err_bias.reshape(1, 1, 1, 1).expand(int(b), 1, int(h), int(w))
        r = self.rgb_bias.reshape(1, 3, 1, 1).expand(int(b), 3, int(h), int(w))
        return e, r


def _setup_bridge_compute_stage(
    pred_rgb_live: torch.Tensor,
    *,
    head: Optional[torch.nn.Module] = None,
    support_value: float = 1.0,
    min_effective_ratio: float = 0.0,
) -> MinimalStreetForwardStage5_5:
    stage = _bridge_stage()
    stage.stage5_5_aux_enabled = True
    stage.stage5_5_target_max_targets = 1
    stage.stage5_5_target_every_n_steps = 1
    stage.stage5_5_splat_eps = 1.0e-6
    stage.stage5_5_detach_geometry = True
    stage.stage5_5_detach_alpha_weights = True
    stage.stage5_5_detach_render_context = True
    stage.stage5_5_error_max = 0.5
    stage.stage5_5_min_valid_pixel_ratio = 0.0
    stage.stage5_5_render_context_dropout = 0.0
    stage.stage5_5_zero_invalid_input = False
    stage.stage5_5_concat_log_support = False
    stage.stage5_5_concat_valid_mask = False
    stage.stage5_5_use_render_rgb = False
    stage.stage5_5_use_render_alpha = False
    stage.stage5_5_support_min_for_extra_loss = 0.5
    stage.stage5_5_loss_all_weight = 0.0
    stage.stage5_5_loss_supported_weight = 0.0
    stage.stage5_5_loss_rgb_residual_weight = 0.0
    stage.stage5_5_loss_rgb_residual_supported_weight = 0.0
    stage.stage5_5_loss_mask_egocar = True
    stage.stage5_5_loss_mask_sky = True
    stage.stage5_5_start_weight_scale = 1.0
    stage.stage5_5_end_weight_scale = 1.0
    stage.stage5_5_warmup_steps = 0
    stage.stage5_5_no_render_probe = False
    stage.stage5_5_no_render_probe_interval = 0
    stage.stage5_5_bridge_min_effective_pixel_ratio = float(min_effective_ratio)
    stage.stage5_5_uncertainty_head = head if head is not None else _BiasHead(0.0)
    stage._current_loss_step = lambda batch: 0
    stage._build_aux_node_pack = lambda out, target_frame_idx: (
        _render_params(1),
        torch.ones((1, 2), dtype=torch.float32),
        torch.ones((1,), dtype=torch.bool),
    )
    stage._splat_node_features_to_view = lambda **kwargs: (
        torch.zeros((int(kwargs["height"]), int(kwargs["width"]), 2), dtype=torch.float32),
        torch.full((int(kwargs["height"]), int(kwargs["width"])), float(support_value), dtype=torch.float32),
    )
    stage._render_single_view = lambda merged_render, view, h, w: (
        pred_rgb_live,
        torch.ones((h, w, 1), dtype=pred_rgb_live.dtype),
    )
    return stage


def test_stage5_5_rgb_residual_loss_uses_detached_render_target():
    pred_rgb_live = torch.zeros((2, 2, 3), dtype=torch.float32, requires_grad=True)
    head = _DualBiasHead()
    stage = _setup_bridge_compute_stage(pred_rgb_live, head=head, support_value=1.0)
    stage.stage5_5_bridge_enabled = False
    stage.stage5_5_loss_rgb_residual_weight = 1.0
    target = {
        "gt_image": torch.ones((2, 2, 3), dtype=torch.float32),
        "view": SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3).unsqueeze(0)),
        "frame_idx": 3,
        "sky_mask": torch.zeros((2, 2), dtype=torch.float32),
        "egocar_mask": torch.zeros((2, 2), dtype=torch.float32),
    }

    pack = stage._compute_feature_splat_uncertainty_loss(
        batch={"aux_targets": [target]},
        out={"loss": torch.zeros(())},
    )
    pack["loss"].backward()

    assert float(pack["loss_rgb_residual"].detach().item()) > 0.0
    assert head.rgb_bias.grad is not None
    assert float(head.rgb_bias.grad.abs().sum().item()) > 0.0
    if pred_rgb_live.grad is not None:
        assert torch.allclose(pred_rgb_live.grad, torch.zeros_like(pred_rgb_live.grad))


def test_stage5_5_bridge_loss_uses_live_render_with_detached_context_and_not_head_grad():
    pred_rgb_live = torch.zeros((2, 2, 3), dtype=torch.float32, requires_grad=True)
    head = _BiasHead(0.0)
    stage = _setup_bridge_compute_stage(pred_rgb_live, head=head, support_value=1.0)
    target = {
        "gt_image": torch.ones((2, 2, 3), dtype=torch.float32),
        "view": SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3).unsqueeze(0)),
        "frame_idx": 3,
        "sky_mask": torch.zeros((2, 2), dtype=torch.float32),
        "egocar_mask": torch.zeros((2, 2), dtype=torch.float32),
    }

    pack = stage._compute_feature_splat_uncertainty_loss(
        batch={"aux_targets": [target]},
        out={"loss": torch.zeros(())},
    )
    pack["loss"].backward()

    assert float(pack["loss_bridge"].detach().item()) > 0.0
    assert pred_rgb_live.grad is not None
    assert float(pred_rgb_live.grad.abs().sum().item()) > 0.0
    if head.bias.grad is not None:
        assert torch.allclose(head.bias.grad, torch.zeros_like(head.bias.grad))


def test_stage5_5_bridge_skips_when_active_ratio_too_low():
    pred_rgb_live = torch.zeros((2, 2, 3), dtype=torch.float32, requires_grad=True)
    stage = _setup_bridge_compute_stage(pred_rgb_live, support_value=0.0, min_effective_ratio=0.005)
    target = {
        "gt_image": torch.ones((2, 2, 3), dtype=torch.float32),
        "view": SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3).unsqueeze(0)),
        "frame_idx": 3,
        "sky_mask": torch.zeros((2, 2), dtype=torch.float32),
        "egocar_mask": torch.zeros((2, 2), dtype=torch.float32),
    }

    pack = stage._compute_feature_splat_uncertainty_loss(
        batch={"aux_targets": [target]},
        out={"loss": torch.zeros(())},
    )

    assert torch.allclose(pack["loss_bridge"], torch.zeros_like(pack["loss_bridge"]))
    assert pack["bridge_skipped_low_active_ratio"] == 1.0
    assert torch.allclose(pack["bridge_active_ratio"], torch.zeros_like(pack["bridge_active_ratio"]))


def test_stage5_5_rigid_aux_uses_updated_local_params_and_reorders_features():
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    stage.device = torch.device("cpu")
    stage.stage5_5_src_support_min_bg = 0.0
    stage.stage5_5_src_support_min_rigid = 0.0
    stage.stage5_5_src_support_min_distant = 0.0
    captured = {}

    def build_rigid_world(node_state_rigid, frame_idx, idx_train, idx_frozen, render_params_rigid_local, U):
        captured["frame_idx"] = int(frame_idx)
        captured["idx_train"] = idx_train.detach().clone()
        captured["idx_frozen"] = idx_frozen.detach().clone()
        captured["rigid_local"] = render_params_rigid_local
        captured["U"] = U.detach().clone()
        return _render_params(4)

    stage._build_rigid_world_for_frame = build_rigid_world
    stage._rigid_point_valid_mask = lambda node_state_rigid, frame_idx: torch.tensor(
        [True, True, True, False, True],
        dtype=torch.bool,
    )
    route = SimpleNamespace(S=torch.tensor([2, 4, 1], dtype=torch.long))
    node_state_rigid = SimpleNamespace(means=torch.zeros((5, 3), dtype=torch.float32))
    rigid_feat_route_order = torch.tensor(
        [
            [20.0, 21.0],
            [40.0, 41.0],
            [10.0, 11.0],
        ]
    )
    out = {
        "_feat_2d_bg": torch.tensor([[1.0, 2.0]]),
        "_acc_w_bg": torch.tensor([1.0]),
        "_render_params_bg": _render_params(1),
        "_node_state_rigid": node_state_rigid,
        "_route": route,
        "_feat_2d_rigid_S": rigid_feat_route_order,
        "_acc_w_rigid_S": torch.tensor([0.2, 0.4, 0.1]),
        "_rigid_writeback_idx": torch.tensor([4], dtype=torch.long),
        "_render_params_rigid_local": {"sentinel": torch.tensor([1])},
        "_render_params_distant": None,
    }

    merged_render, merged_features, merged_mask = stage._build_aux_node_pack(out=out, target_frame_idx=7)

    assert captured["frame_idx"] == 7
    assert captured["idx_train"].tolist() == [4]
    assert captured["idx_frozen"].tolist() == [0, 1, 2]
    assert captured["U"].tolist() == [4]
    assert captured["rigid_local"] is out["_render_params_rigid_local"]
    assert tuple(merged_render["means_r"].shape) == (5, 3)
    assert merged_features.tolist() == [
        [1.0, 2.0],
        [40.0, 41.0],
        [0.0, 0.0],
        [10.0, 11.0],
        [20.0, 21.0],
    ]
    assert merged_mask.tolist() == [True, True, False, True, True]


def test_stage5_5_aux_target_requires_frame_idx():
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    torch.nn.Module.__init__(stage)
    stage.device = torch.device("cpu")
    stage.stage5_5_aux_enabled = True
    stage.stage5_5_target_max_targets = 1
    stage.stage5_5_target_every_n_steps = 1
    stage._current_loss_step = lambda batch: 0

    with pytest.raises(RuntimeError, match="aux target must provide frame_idx"):
        stage._compute_feature_splat_uncertainty_loss(
            batch={
                "aux_targets": [
                    {
                        "gt_image": torch.zeros((2, 2, 3), dtype=torch.float32),
                        "view": SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3).unsqueeze(0)),
                    }
                ]
            },
            out={"loss": torch.zeros(())},
        )


def test_stage5_5_supported_loss_skips_empty_support_targets_and_squeezes_alpha():
    stage = MinimalStreetForwardStage5_5.__new__(MinimalStreetForwardStage5_5)
    torch.nn.Module.__init__(stage)
    stage.device = torch.device("cpu")
    stage.stage5_5_aux_enabled = True
    stage.stage5_5_target_max_targets = 2
    stage.stage5_5_target_every_n_steps = 1
    stage.stage5_5_splat_eps = 1.0e-6
    stage.stage5_5_detach_geometry = True
    stage.stage5_5_detach_alpha_weights = True
    stage.stage5_5_detach_render_context = True
    stage.stage5_5_error_max = 0.5
    stage.stage5_5_min_valid_pixel_ratio = 0.0
    stage.stage5_5_render_context_dropout = 0.0
    stage.stage5_5_zero_invalid_input = False
    stage.stage5_5_concat_log_support = False
    stage.stage5_5_concat_valid_mask = False
    stage.stage5_5_use_render_rgb = False
    stage.stage5_5_use_render_alpha = False
    stage.stage5_5_support_min_for_extra_loss = 0.5
    stage.stage5_5_loss_all_weight = 1.0
    stage.stage5_5_loss_supported_weight = 1.0
    stage.stage5_5_start_weight_scale = 1.0
    stage.stage5_5_end_weight_scale = 1.0
    stage.stage5_5_warmup_steps = 0
    stage.stage5_5_no_render_probe = False
    stage.stage5_5_no_render_probe_interval = 0

    class ZeroHead(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros((int(x.shape[0]), 1, int(x.shape[2]), int(x.shape[3])))

    stage.stage5_5_uncertainty_head = ZeroHead()
    stage._current_loss_step = lambda batch: 0
    stage._build_aux_node_pack = lambda out, target_frame_idx: (
        _render_params(1),
        torch.ones((1, 2), dtype=torch.float32),
        torch.ones((1,), dtype=torch.bool),
    )
    splat_calls = {"count": 0}

    def splat(**kwargs):
        splat_calls["count"] += 1
        support_value = 1.0 if splat_calls["count"] == 1 else 0.0
        return (
            torch.zeros((int(kwargs["height"]), int(kwargs["width"]), 2), dtype=torch.float32),
            torch.full((int(kwargs["height"]), int(kwargs["width"])), support_value, dtype=torch.float32),
        )

    stage._splat_node_features_to_view = splat
    stage._render_single_view = lambda merged_render, view, h, w: (
        torch.zeros((h, w, 3), dtype=torch.float32),
        torch.ones((h, w, 1), dtype=torch.float32),
    )
    stage._build_aux_loss_mask = lambda target, h, w: torch.ones((h, w), dtype=torch.float32)

    target = {
        "gt_image": torch.full((2, 2, 3), 0.25, dtype=torch.float32),
        "view": SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3).unsqueeze(0)),
        "frame_idx": 3,
    }
    pack = stage._compute_feature_splat_uncertainty_loss(
        batch={"aux_targets": [dict(target), dict(target)]},
        out={"loss": torch.zeros(())},
    )

    assert pack["processed_targets"] == 2.0
    assert pack["skipped_empty_supported"] == 1.0
    assert torch.allclose(pack["loss_support"], pack["loss_all"])


def test_stage5_5_retain_graph_flag_allows_aux_and_proxy_backward():
    class ProbeTrainer(MinimalStreetForwardStage4_2):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.device = torch.device("cpu")
            self.x = torch.nn.Parameter(torch.ones((1, 3), dtype=torch.float32))
            self.optimizer = torch.optim.SGD([self.x], lr=0.0)
            self.update_node_state_interval = 0
            self.reset_node_state_interval = 0

        def _compute_branch_grad_norms(self):
            return {}

        def forward(self, batch):
            means = self.x * self.x
            render_params = {
                "means_r": means,
                "scales_r": torch.ones((1, 3), dtype=torch.float32),
                "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                "opacities_r": torch.ones((1,), dtype=torch.float32),
                "colors_r": torch.zeros((1, 1, 3), dtype=torch.float32),
            }
            proxies = _create_proxy_params(render_params)
            proxy_loss = proxies["means_p"].sum()
            aux_loss = 3.0 * render_params["means_r"].sum()
            img = torch.zeros((1, 1, 3), dtype=torch.float32)
            return {
                "loss": proxy_loss + aux_loss,
                "loss_l1": proxy_loss * 0.0,
                "loss_ssim": proxy_loss * 0.0,
                "loss_mask": proxy_loss * 0.0,
                "loss_opacity_entropy": proxy_loss * 0.0,
                "render_params": render_params,
                "proxies": proxies,
                "_retain_graph_for_proxy_backward": True,
                "_node_state_bg": SimpleNamespace(means=torch.zeros((1, 3), dtype=torch.float32)),
                "_node_state_distant": None,
                "_node_state_rigid": None,
                "pred_rgbs": [img],
                "gt_images": [img],
                "pred_rgb": img,
                "gt_image": img,
            }

    trainer = ProbeTrainer()
    trainer.train_step({"targets": [{}], "source_views": []}, step=1)

    assert torch.allclose(trainer.x.grad, torch.full_like(trainer.x, 8.0))
