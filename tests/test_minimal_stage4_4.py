from __future__ import annotations

from types import SimpleNamespace

import torch

from models.streetforward.minimal_trainer_stage4_4 import MinimalStreetForwardStage4_4


def test_stage4_4_uses_multicam_v3_path_and_reports_stats():
    trainer = MinimalStreetForwardStage4_4.__new__(MinimalStreetForwardStage4_4)
    trainer.device = torch.device("cpu")
    trainer.use_fused_cuda_backproject_v4 = False
    trainer.fused_cuda_backproject_v4_force_fallback = False
    trainer.use_fused_cuda_backproject_v3 = True
    trainer.use_fused_cuda_backproject_v2 = False
    trainer._perf_acc = {}

    class _DummyRgbExtractor:
        def render_rgb_only(self, *args, **kwargs):
            return [torch.zeros(2, 2, 3), torch.zeros(2, 2, 3)], {"num_views": 2}

    called = {"multi": False, "per_view": False}

    class _DummyV3:
        def render_and_backproject_streaming_fused_multi_camera(self, **kwargs):
            called["multi"] = True
            n = int(kwargs["num_gaussians"])
            c = int(kwargs["features_2d"].shape[-1])
            feat = torch.zeros(n, c)
            stats = {
                "build_multi_meta_ms": 1.0,
                "nnz_total": 4,
                "isects_total": 8,
                "pairs_total": 10,
                "pairs_after_threshold": 7,
            }
            if kwargs.get("return_accumulated_weights"):
                return feat, torch.zeros(n), stats
            return feat, stats

        def render_and_backproject_streaming_fused_per_view_fallback(self, **kwargs):
            called["per_view"] = True
            return torch.zeros(1, 1), {"pairs_total": 1, "pairs_after_threshold": 1}

    trainer.alpha_t_extractor = _DummyRgbExtractor()
    trainer.alpha_t_extractor_v3 = _DummyV3()
    trainer.alpha_t_extractor_v2 = SimpleNamespace(render_and_backproject_streaming_fused=lambda **kwargs: None)
    trainer.feature_backprojector = SimpleNamespace(eps=1e-8, weight_threshold=0.0)
    trainer.image_feature_extractor = lambda x: torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2)

    feat, acc = trainer._compute_2d_features_for_gaussians(
        gaussians={
            "means": torch.zeros(1, 3),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "scales": torch.ones(1, 3),
            "opacities": torch.ones(1),
            "colors": torch.zeros(1, 4, 3),
        },
        source_views=[object(), object()],
        source_images=[torch.zeros(3, 2, 2), torch.zeros(3, 2, 2)],
        height=2,
        width=2,
        return_accumulated_weights=False,
    )

    assert called["multi"] is True
    assert called["per_view"] is False
    assert feat.shape == (1, 2)
    assert acc is None
    assert "2d_bp_build_multi_meta_ms" in trainer._perf_acc
    assert "2d_bp_nnz_total" in trainer._perf_acc
    assert "2d_bp_isects_total" in trainer._perf_acc
