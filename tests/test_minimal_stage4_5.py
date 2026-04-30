from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from models.streetforward.minimal_trainer_stage4_2 import MinimalStreetForwardStage4_2
from models.streetforward.minimal_trainer_stage4_5 import MinimalStreetForwardStage4_5


def test_stage4_5_train_step_adds_sky_compat_fields(monkeypatch):
    trainer = MinimalStreetForwardStage4_5.__new__(MinimalStreetForwardStage4_5)

    def _fake_super_train_step(self, *args, **kwargs):
        return {
            "loss": 0.0,
            "num_gaussians_bg": 11,
            "num_gaussians_distant": 2,
            "num_gaussians_rigid": 3,
        }

    monkeypatch.setattr(MinimalStreetForwardStage4_2, "train_step", _fake_super_train_step)
    out = MinimalStreetForwardStage4_5.train_step(trainer, batch={})
    assert out["num_gaussians_sky"] == 0
    assert out["num_sky_src_feat_valid"] == 0
    assert out["num_sky_update"] == 0
    assert out["sky_update_ratio"] == 0.0
    assert out["hidden_norm_sky_mean"] == 0.0
    assert out["grad_norm_sky"] == 0.0
    assert out["branch_presence"] == {
        "bg": True,
        "distant": True,
        "rigid": True,
        "sky": False,
    }


def test_stage4_5_render_source_scene_only_for_cnn_shape():
    trainer = MinimalStreetForwardStage4_5.__new__(MinimalStreetForwardStage4_5)
    trainer.device = torch.device("cpu")

    class _DummyAlphaExtractor:
        def render_rgb_only(self, gaussians: Dict[str, torch.Tensor], source_views: List[Any], height: int, width: int, return_acc: bool, return_debug_stats: bool):
            assert return_acc is True
            rgbs = [torch.zeros(height, width, 3), torch.zeros(height, width, 3)]
            accs = [torch.zeros(height, width), torch.zeros(height, width)]
            return rgbs, accs

    trainer.alpha_t_extractor = _DummyAlphaExtractor()
    trainer.image_feature_extractor = lambda x: torch.zeros(x.shape[0], x.shape[1], x.shape[2], 8)

    source_images = [torch.zeros(3, 4, 5), torch.zeros(3, 4, 5)]
    scene = {
        "means": torch.zeros(1, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "scales": torch.ones(1, 3),
        "opacities": torch.ones(1),
        "colors": torch.zeros(1, 4, 3),
    }
    out = trainer._render_source_scene_only_for_cnn(
        gaussians_scene=scene,
        source_views=[object(), object()],
        source_images=source_images,
        source_sky_masks=None,
        source_egocar_masks=None,
        height=4,
        width=5,
    )
    assert "features_2d" in out
    assert tuple(out["features_2d"].shape) == (2, 4, 5, 8)


def test_stage4_5_validation_api_methods_exist():
    required_methods = [
        "inference_step_from_train_batch",
        "export_3dgs_state",
        "render_views_from_scene_state",
        "reset_node_state",
    ]
    for name in required_methods:
        assert hasattr(MinimalStreetForwardStage4_5, name)


def test_stage4_5_build_source_pair_valid_mask_semantics():
    trainer = MinimalStreetForwardStage4_5.__new__(MinimalStreetForwardStage4_5)
    trainer.device = torch.device("cpu")

    source_images = [torch.zeros(3, 2, 3), torch.zeros(3, 2, 3)]
    source_sky_masks = [
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.zeros(2, 3),
    ]
    source_egocar_masks = [
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    ]

    valid = trainer._build_source_pair_valid_mask(
        source_images=source_images,
        source_sky_masks=source_sky_masks,
        source_egocar_masks=source_egocar_masks,
    )
    assert tuple(valid.shape) == (2, 2, 3)
    assert torch.allclose(
        valid[0],
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        valid[1],
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32),
    )


def test_stage4_5_backproject_scene_features_uses_fused_multicam_with_mask():
    trainer = MinimalStreetForwardStage4_5.__new__(MinimalStreetForwardStage4_5)
    trainer.feature_backprojector = object()
    trainer._perf_acc = {}
    called = {}

    class _DummyExtractor:
        def render_and_backproject_streaming_fused_multi_camera(
            self,
            gaussians,
            cameras,
            features_2d,
            height,
            width,
            num_gaussians,
            backprojector,
            source_pair_valid_mask,
            return_accumulated_weights,
            return_debug_stats,
        ):
            called["mask"] = source_pair_valid_mask
            called["num_gaussians"] = num_gaussians
            called["num_cameras"] = len(cameras)
            feat = torch.ones(num_gaussians, 4)
            acc = torch.full((num_gaussians,), 2.0)
            stats = {"pairs_total": 10, "pairs_after_mask": 6}
            return feat, acc, stats

    trainer.alpha_t_extractor_v3 = _DummyExtractor()
    gaussians_scene = {
        "means": torch.zeros(3, 3),
        "quats": torch.zeros(3, 4),
        "scales": torch.zeros(3, 3),
        "opacities": torch.zeros(3),
        "colors": torch.zeros(3, 1, 3),
    }
    source_pair_valid_mask = torch.ones(2, 5, 6)
    feat, acc = trainer._backproject_scene_features_multi_camera(
        gaussians_scene=gaussians_scene,
        source_views=[object(), object()],
        features_2d=torch.zeros(2, 5, 6, 8),
        source_pair_valid_mask=source_pair_valid_mask,
        height=5,
        width=6,
    )
    assert tuple(feat.shape) == (3, 4)
    assert tuple(acc.shape) == (3,)
    assert called["num_gaussians"] == 3
    assert called["num_cameras"] == 2
    assert torch.equal(called["mask"], source_pair_valid_mask)


def test_stage4_5_build_source_pair_valid_mask_fast_fail_on_length_mismatch():
    trainer = MinimalStreetForwardStage4_5.__new__(MinimalStreetForwardStage4_5)
    trainer.device = torch.device("cpu")

    source_images = [torch.zeros(3, 2, 3), torch.zeros(3, 2, 3)]
    with pytest.raises(ValueError, match="source_egocar_masks length"):
        trainer._build_source_pair_valid_mask(
            source_images=source_images,
            source_sky_masks=None,
            source_egocar_masks=[torch.zeros(2, 3)],
        )
    with pytest.raises(ValueError, match="source_sky_masks length"):
        trainer._build_source_pair_valid_mask(
            source_images=source_images,
            source_sky_masks=[torch.zeros(2, 3)],
            source_egocar_masks=None,
        )


def test_stage4_5_requires_mask_require_sky_mask_true(monkeypatch):
    def _fake_super_init(self, config, device, **kwargs):
        self.renderer = object()
        self.sh_degree = 1

    class _Cfg:
        def __init__(self):
            self.model = {
                "use_fused_cuda_backproject_v4": False,
                "fused_cuda_backproject_v4_force_fallback": False,
                "use_fused_cuda_backproject_v3": True,
            }
            self._losses = {
                "photometric": {"exclude_sky_region": True},
                "mask": {"require_sky_mask": False},
            }

        def get(self, key: str, default=None):
            if key == "losses":
                return self._losses
            return default

    monkeypatch.setattr(MinimalStreetForwardStage4_2, "__init__", _fake_super_init)

    with pytest.raises(ValueError, match="losses.mask.require_sky_mask=true"):
        MinimalStreetForwardStage4_5(config=_Cfg(), device=torch.device("cpu"))


def test_stage4_5_requires_fused_v3_enabled(monkeypatch):
    def _fake_super_init(self, config, device, **kwargs):
        self.renderer = object()
        self.sh_degree = 1

    class _Cfg:
        def __init__(self):
            self.model = {
                "use_fused_cuda_backproject_v4": False,
                "fused_cuda_backproject_v4_force_fallback": False,
                "use_fused_cuda_backproject_v3": False,
            }
            self._losses = {
                "photometric": {"exclude_sky_region": True},
                "mask": {"require_sky_mask": True},
            }

        def get(self, key: str, default=None):
            if key == "losses":
                return self._losses
            return default

    monkeypatch.setattr(MinimalStreetForwardStage4_2, "__init__", _fake_super_init)

    with pytest.raises(ValueError, match="use_fused_cuda_backproject_v3=true"):
        MinimalStreetForwardStage4_5(config=_Cfg(), device=torch.device("cpu"))


def test_stage4_5_rejects_direct_v4_path_for_non_stage5_4(monkeypatch):
    def _fake_super_init(self, config, device, **kwargs):
        del device, kwargs
        self.renderer = object()
        self.sh_degree = 1
        self.config = config

    class _Cfg:
        def __init__(self):
            self.model = {
                "stage": "4_5",
                "use_fused_cuda_backproject_v4": True,
                "fused_cuda_backproject_v4_force_fallback": False,
                "use_fused_cuda_backproject_v3": True,
            }
            self._losses = {
                "photometric": {"exclude_sky_region": True},
                "mask": {"require_sky_mask": True},
            }

        def get(self, key: str, default=None):
            if key == "losses":
                return self._losses
            return default

    monkeypatch.setattr(MinimalStreetForwardStage4_2, "__init__", _fake_super_init)
    with pytest.raises(ValueError, match="does not implement fused_cuda_backproject_v4 yet"):
        MinimalStreetForwardStage4_5(config=_Cfg(), device=torch.device("cpu"))


def test_stage4_5_allows_stage5_4_to_enable_direct_v4_path(monkeypatch):
    def _fake_super_init(self, config, device, **kwargs):
        del device, kwargs
        self.renderer = object()
        self.sh_degree = 1
        self.config = config

    class _Cfg:
        def __init__(self):
            self.model = {
                "stage": "5_4",
                "use_fused_cuda_backproject_v4": True,
                "fused_cuda_backproject_v4_force_fallback": False,
                "use_fused_cuda_backproject_v3": True,
            }
            self._losses = {
                "photometric": {"exclude_sky_region": True},
                "mask": {"require_sky_mask": True},
            }

        def get(self, key: str, default=None):
            if key == "losses":
                return self._losses
            return default

    monkeypatch.setattr(MinimalStreetForwardStage4_2, "__init__", _fake_super_init)
    trainer = MinimalStreetForwardStage4_5(config=_Cfg(), device=torch.device("cpu"))
    assert trainer.use_fused_cuda_backproject_v4 is True
