from __future__ import annotations

from collections import defaultdict

import pytest
import torch
from torch import nn

from models.feature_extractors.alpha_t_extractor import AlphaTWeightExtractor
from models.iforward.observation_feedback import (
    FeedbackMode,
    FrontendParameterModeScope,
    ObservationFeedbackPolicy,
)
from models.iforward.dino_feature_cache import DINOFeatureCache
from models.streetforward.minimal_trainer_stage4_5 import MinimalStreetForwardStage4_5
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0


class _View:
    def __init__(self, *, requires_grad: bool = False) -> None:
        self.camtoworlds = torch.eye(4, requires_grad=requires_grad).unsqueeze(0)
        self.Ks = torch.eye(3, requires_grad=requires_grad).unsqueeze(0)


class _DifferentiableRenderer:
    def __call__(self, **kwargs):
        means = kwargs["means"]
        scales = kwargs["scales"]
        quats = kwargs["quats"]
        opacities = kwargs["opacities"]
        colors = kwargs["colors"]
        views = int(kwargs["viewmats"].shape[0])
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        scalar = (
            means.mean()
            + scales.mean()
            + quats.mean()
            + opacities.mean()
            + colors.mean()
        )
        rgb = scalar.expand(views, height, width, 3)
        alpha = scalar.expand(views, height, width, 1)
        return rgb, alpha, {}


def _render_params() -> dict[str, torch.Tensor]:
    return {
        "means": torch.full((3, 3), 0.05, requires_grad=True),
        "scales": torch.full((3, 3), 0.05, requires_grad=True),
        "quats": torch.full((3, 4), 0.05, requires_grad=True),
        "opacities": torch.full((3,), 0.05, requires_grad=True),
        "colors": torch.full((3, 4, 3), 0.05, requires_grad=True),
    }


def test_feedback_renderer_matches_legacy_forward_and_only_grads_gaussians() -> None:
    extractor = AlphaTWeightExtractor.__new__(AlphaTWeightExtractor)
    extractor.renderer = _DifferentiableRenderer()
    extractor.sh_degree = 1
    extractor.tile_size = 16
    params = _render_params()
    views = [_View(requires_grad=True), _View(requires_grad=True)]

    legacy = extractor.render_rgb_only(params, views, 4, 5)
    feedback = extractor.render_rgb_feedback(params, views, 4, 5, absgrad=False)
    assert isinstance(feedback, list)
    torch.testing.assert_close(torch.stack(feedback), torch.stack(legacy), rtol=0.0, atol=0.0)

    torch.stack(feedback).sum().backward()
    for value in params.values():
        assert value.grad is not None
        assert float(value.grad.abs().sum()) > 0.0
    for view in views:
        assert view.camtoworlds.grad is None
        assert view.Ks.grad is None


class _SplitFrontend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.residual_unet = nn.Conv2d(6, 4, kernel_size=1)
        self.dino_calls = 0

    @staticmethod
    def get_feature_resolution(height: int, width: int) -> tuple[int, int]:
        return int(height), int(width)

    @staticmethod
    def dino_adapter_has_trainable_params() -> bool:
        return False

    @staticmethod
    def _to_nchw_6(images: torch.Tensor) -> torch.Tensor:
        if int(images.shape[-1]) == 6:
            return images.permute(0, 3, 1, 2).contiguous()
        return images

    def extract_dino_feature(
        self,
        rgb: torch.Tensor,
        *,
        target_hw: tuple[int, int],
        detach: bool = True,
    ) -> torch.Tensor:
        self.dino_calls += 1
        value = rgb[:, :2].permute(0, 2, 3, 1).contiguous()
        return value.detach() if detach else value

    def extract_residual_feature(self, images: torch.Tensor) -> torch.Tensor:
        if int(images.shape[-1]) == 6:
            images = images.permute(0, 3, 1, 2).contiguous()
        return self.residual_unet(images).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def fuse_features(dino_feat: torch.Tensor, residual_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([residual_feat, dino_feat.to(residual_feat)], dim=-1)


class _FeedbackExtractor:
    @staticmethod
    def render_rgb_feedback(
        gaussians,
        cameras,
        height,
        width,
        *,
        return_acc=False,
        absgrad=False,
    ):
        del return_acc, absgrad
        scalar = sum(value.mean() for value in gaussians.values())
        return [scalar.expand(height, width, 3) for _ in cameras]

    @staticmethod
    def render_rgb_only(
        gaussians,
        cameras,
        height,
        width,
        *,
        return_acc=False,
        return_debug_stats=False,
    ):
        del return_debug_stats
        scalar = sum(value.mean() for value in gaussians.values())
        rgbs = [scalar.detach().expand(height, width, 3) for _ in cameras]
        if not return_acc:
            return rgbs
        accs = [scalar.detach().expand(height, width, 1) for _ in cameras]
        return rgbs, accs


def _feedback_stage() -> MinimalStreetForwardStage4_5:
    stage = MinimalStreetForwardStage4_5.__new__(MinimalStreetForwardStage4_5)
    nn.Module.__init__(stage)
    stage.device = torch.device("cpu")
    stage.image_feature_extractor = _SplitFrontend()
    stage.alpha_t_extractor = _FeedbackExtractor()
    stage.stage6_cnn_view_chunk_size = 1
    stage.stage3_0_enabled = False
    stage.stage3_dino_native_enabled = False
    stage.dino_feature_cache = None
    stage.dino_feature_cache_level = "adapter_output"
    return stage


def _feedback_features(
    stage: MinimalStreetForwardStage4_5,
    params: dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
) -> torch.Tensor:
    out = stage._render_source_scene_only_for_cnn(
        gaussians_scene=params,
        source_views=[_View(), _View()],
        source_images=[torch.zeros(4, 5, 3), torch.ones(4, 5, 3)],
        source_sky_masks=[torch.zeros(4, 5), torch.zeros(4, 5)],
        source_egocar_masks=[torch.zeros(4, 5), torch.zeros(4, 5)],
        height=4,
        width=5,
        feedback_enabled=True,
        feedback_alpha=float(alpha),
        checkpoint_dynamic=True,
    )
    assert out["features_2d"].requires_grad
    return out["features_2d"]


def _run_feedback_frontend(
    stage: MinimalStreetForwardStage4_5,
    params: dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
) -> torch.Tensor:
    return _feedback_features(stage, params, alpha=float(alpha)).square().mean()


def test_full_dynamic_observation_checkpoint_recomputes_without_repeating_dino() -> None:
    stage = _feedback_stage()
    params = _render_params()
    loss = _run_feedback_frontend(stage, params)
    assert stage.image_feature_extractor.dino_calls == 2
    loss.backward()
    assert stage.image_feature_extractor.dino_calls == 2
    assert params["means"].grad is not None
    assert float(params["means"].grad.abs().sum()) > 0.0
    assert stage.image_feature_extractor.residual_unet.weight.grad is not None


def test_feedback_and_legacy_parity_share_static_dino_cache_entries() -> None:
    stage = _feedback_stage()
    stage.dino_feature_cache = DINOFeatureCache(
        dtype="float32",
        cpu_pinned=False,
        cpu_max_items=0,
        gpu_max_items=4,
        async_copy=False,
        fail_if_trainable=True,
    )
    params = _render_params()
    common = dict(
        gaussians_scene=params,
        source_views=[_View(), _View()],
        source_images=[torch.zeros(4, 5, 3), torch.ones(4, 5, 3)],
        source_sky_masks=[torch.zeros(4, 5), torch.zeros(4, 5)],
        source_egocar_masks=[torch.zeros(4, 5), torch.zeros(4, 5)],
        height=4,
        width=5,
        dino_cache_key=("parity", 1),
    )
    feedback = stage._render_source_scene_only_for_cnn(
        **common,
        feedback_enabled=True,
        feedback_alpha=1.0,
        checkpoint_dynamic=False,
    )
    assert stage.image_feature_extractor.dino_calls == 2
    legacy = stage._render_source_scene_only_for_cnn(
        **common,
        feedback_enabled=False,
        feedback_alpha=0.0,
        checkpoint_dynamic=False,
    )
    assert stage.image_feature_extractor.dino_calls == 2
    torch.testing.assert_close(feedback["features_2d"], legacy["features_2d"])


def test_frozen_frontend_checkpoint_keeps_input_gradient_and_restores_parameters() -> None:
    stage = _feedback_stage()
    params = _render_params()
    names = [
        "image_feature_extractor.residual_unet.weight",
        "image_feature_extractor.residual_unet.bias",
    ]
    with FrontendParameterModeScope(stage, names, FeedbackMode.FROZEN_INPUT_GRAD_CHECKPOINTED):
        loss = _run_feedback_frontend(stage, params)
        loss.backward()
        assert stage.image_feature_extractor.residual_unet.weight.grad is None
        assert params["means"].grad is not None
        assert float(params["means"].grad.abs().sum()) > 0.0
    assert stage.image_feature_extractor.residual_unet.weight.requires_grad


def test_two_block_later_observation_reaches_earlier_update_but_rollout_boundary_detaches() -> None:
    """Exercise the K=2 feedback transaction, including nested checkpoint replay."""

    torch.manual_seed(9)
    stage = _feedback_stage()
    base = {key: value.detach() for key, value in _render_params().items()}
    earlier_delta = torch.full_like(base["means"], 0.01, requires_grad=True)

    # Block 0 update -> observation -> block 1 update.  No state detach is
    # allowed between these two blocks in the same rollout.
    block0_scene = {**base, "means": base["means"] + earlier_delta}
    block0_features = _feedback_features(stage, block0_scene, alpha=1.0)
    later_delta = 0.01 * block0_features.mean().expand_as(block0_scene["means"])
    block1_scene = {**base, "means": block0_scene["means"] + later_delta}
    later_loss = _feedback_features(stage, block1_scene, alpha=1.0).square().mean()
    earlier_grad = torch.autograd.grad(later_loss, earlier_delta, retain_graph=False)[0]
    assert torch.isfinite(earlier_grad).all()
    assert float(earlier_grad.abs().sum()) > 0.0

    # The optimizer transaction boundary deliberately writes a graph-free
    # carried state.  A later rollout may see the value but not its activations.
    boundary_scene = {key: value.detach() for key, value in block1_scene.items()}
    boundary_loss = _feedback_features(stage, boundary_scene, alpha=1.0).square().mean()
    boundary_grad = torch.autograd.grad(
        boundary_loss,
        earlier_delta,
        allow_unused=True,
        retain_graph=False,
    )[0]
    assert boundary_grad is None


def test_feedback_alpha_zero_blocks_observation_jacobian() -> None:
    stage = _feedback_stage()
    params = _render_params()
    loss = _run_feedback_frontend(stage, params, alpha=0.0)
    grad = torch.autograd.grad(loss, params["means"], allow_unused=True)[0]
    assert grad is not None
    assert torch.count_nonzero(grad) == 0


def test_runtime_forward_parity_stats_compare_all_observation_outputs_and_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = {
        "source_rgb": torch.ones(2, 3, 4, 3),
        "features_2d": torch.ones(2, 2, 2, 5),
        "fwhr_detail_2d": torch.ones(2, 2, 2, 3),
        "stage3_dino_native_2d": torch.ones(2, 1, 1, 4),
        "source_pair_valid_mask": torch.ones(2, 3, 4),
    }
    reference = {key: value.clone() for key, value in actual.items()}
    stats = MinimalStreetForwardStage6_0._observation_feedback_forward_parity_stats(
        actual, reference
    )
    assert stats["iforward/feedback/parity/pass"] == 1.0
    assert stats["iforward/feedback/parity/tensor_count"] == 5.0

    reference["features_2d"][0, 0, 0, 0] = 2.0
    with pytest.raises(RuntimeError, match="forward parity failed"):
        MinimalStreetForwardStage6_0._observation_feedback_forward_parity_stats(
            actual, reference
        )

    # Repeated AMP frontend execution may have a noticeable absolute error at
    # large feature amplitudes.  The guard is scale-aware but still bounds both
    # aggregate RMS and the worst local deviation.
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda: True)
    reference_amp = dict(reference)
    reference_amp["features_2d"] = torch.full((1, 32, 32, 16), 20.0)
    actual_amp = dict(reference_amp)
    actual_amp["features_2d"] = reference_amp["features_2d"].clone()
    actual_amp["features_2d"].reshape(-1)[0] += 0.26
    amp_stats = MinimalStreetForwardStage6_0._observation_feedback_forward_parity_stats(
        actual_amp, reference_amp
    )
    assert amp_stats["iforward/feedback/parity/amp_relative_tolerance_used"] == 1.0
    assert amp_stats["iforward/feedback/parity/features_2d_max_rel_to_peak"] < 0.02

    actual_bad = dict(reference_amp)
    actual_bad["features_2d"] = reference_amp["features_2d"].clone()
    actual_bad["features_2d"].reshape(-1)[0] = 40.0
    with pytest.raises(RuntimeError, match="max_rel_to_peak"):
        MinimalStreetForwardStage6_0._observation_feedback_forward_parity_stats(
            actual_bad, reference_amp
        )


def test_first_feedback_mode_forces_source_gradient_probe_before_interval() -> None:
    policy = ObservationFeedbackPolicy.from_config(
        {
            "enable": True,
            "modes": {
                "repeat_refine": "trainable_checkpointed",
                "shuffled_coverage": "trainable_checkpointed",
                "high_block_repair": "frozen_input_grad_checkpointed",
            },
            "source_render": {
                "enable": True,
                "renderer_mode": "differentiable_rgb",
                "checkpoint_scope": "full_dynamic_observation",
                "absgrad": False,
                "alpha_schedule": [[0, 0.0], [1, 1.0]],
            },
            "debug": {"grad_probe_interval": 500},
        }
    )
    stage = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(stage)
    stage.observation_feedback_policy = policy
    stage._parent_vjp_drift_collector = type(
        "EmptyCollector",
        (),
        {"clear": lambda self: None, "records": lambda self, branch: []},
    )()
    stage._parent_vjp_sampled_branches = set()
    stage._parent_vjp_force_refresh_branches = set()
    stage._observation_feedback_grad_records = defaultdict(list)
    stage._observation_feedback_probe_modes_seen = set()
    stage._observation_feedback_probe_modes_current = set()
    stage._observation_feedback_force_probe_current = False
    stage.stage3_0_global_step = 7

    stage.reset_observation_feedback_runtime_stats()
    mode, alpha, _, _ = stage._observation_feedback_for_visit(
        {
            "global_step": 7,
            "feedback_schedule_step": 7,
            "distribution_type": "high_block_repair",
            "train_2d_mode": "frozen_input_grad_checkpointed",
        }
    )
    assert mode is FeedbackMode.FROZEN_INPUT_GRAD_CHECKPOINTED
    assert alpha == 1.0
    source = torch.ones(3, requires_grad=True)
    stage._register_observation_feedback_grad_probe(
        source, name="source_render_input/means"
    )
    (source * 2.0).sum().backward()
    metrics = stage.consume_observation_feedback_runtime_stats()
    assert metrics["feedback/grad_probe/first_mode/frozen_input_grad_checkpointed"] == 1.0
    assert metrics["feedback/source_render_input_grad_norm"] > 0.0
