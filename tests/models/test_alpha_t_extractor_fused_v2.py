from __future__ import annotations

import torch
import pytest

import models.feature_extractors.alpha_t_extractor_v2 as extractor_v2_mod
from models.feature_extractors.alpha_t_extractor_v2 import AlphaTWeightExtractorV2


def test_fused_streaming_fast_fail_on_cpu():
    extractor = AlphaTWeightExtractorV2.__new__(AlphaTWeightExtractorV2)
    with pytest.raises(RuntimeError, match="fast-fail"):
        extractor.render_and_backproject_streaming_fused(
            gaussians={},
            cameras=[],
            features_2d=torch.zeros(1, 2, 2, 3),
            height=2,
            width=2,
            num_gaussians=0,
            backprojector=object(),
            return_accumulated_weights=False,
            return_debug_stats=False,
        )


def test_extract_single_weight_fused_allows_feat2d_requires_grad(monkeypatch):
    extractor = AlphaTWeightExtractorV2.__new__(AlphaTWeightExtractorV2)
    monkeypatch.setattr(extractor_v2_mod, "rasterize_and_backproject_in_range", lambda **kwargs: None)
    monkeypatch.setattr(extractor_v2_mod, "backproject_feature_grad_in_range", lambda **kwargs: torch.zeros(2, 2, 4))

    def _fake_apply(*args, **kwargs):
        num_gaussians = int(args[10])
        feat = torch.zeros(num_gaussians, 4, dtype=torch.float32)
        w = torch.ones(num_gaussians, dtype=torch.float32)
        z = torch.zeros(1, dtype=torch.long)
        return feat, w, w, z, z

    monkeypatch.setattr(extractor_v2_mod._RasterizeAndBackprojectFeatOnlyFn, "apply", _fake_apply)
    meta = {
        "means2d": torch.zeros(3, 2, dtype=torch.float32),
        "conics": torch.zeros(3, 3, dtype=torch.float32),
        "opacities": torch.ones(3, dtype=torch.float32),
        "tile_size": 16,
        "isect_offsets": torch.zeros(1, 1, dtype=torch.int32),
        "flatten_ids": torch.zeros(0, dtype=torch.int32),
        "gaussian_ids": torch.tensor([0, 1, 2], dtype=torch.long),
    }
    feat_sum, _, _, pairs_total, pairs_kept = extractor.extract_single_weight_fused(
        meta=meta,
        feat_2d=torch.zeros(2, 2, 4, dtype=torch.float32, requires_grad=True),
        height=2,
        width=2,
        num_gaussians=3,
        weight_threshold=0.0,
    )
    assert feat_sum.shape == (3, 4)
    assert pairs_total == 0
    assert pairs_kept == 0


def test_extract_single_weight_fused_returns_pair_counters(monkeypatch):
    extractor = AlphaTWeightExtractorV2.__new__(AlphaTWeightExtractorV2)

    def _fake_fused(**kwargs):
        n = int(kwargs["num_gaussians"])
        feat = torch.zeros(n, 4, dtype=torch.float32)
        w_feat = torch.ones(n, dtype=torch.float32)
        w_sup = torch.full((n,), 2.0, dtype=torch.float32)
        return feat, w_feat, w_sup, torch.tensor([10], dtype=torch.long), torch.tensor([7], dtype=torch.long)

    monkeypatch.setattr(extractor_v2_mod, "rasterize_and_backproject_in_range", _fake_fused)
    meta = {
        "means2d": torch.zeros(3, 2, dtype=torch.float32),
        "conics": torch.zeros(3, 3, dtype=torch.float32),
        "opacities": torch.ones(3, dtype=torch.float32),
        "tile_size": 16,
        "isect_offsets": torch.zeros(1, 1, dtype=torch.int32),
        "flatten_ids": torch.zeros(0, dtype=torch.int32),
        "gaussian_ids": torch.tensor([0, 1, 2], dtype=torch.long),
    }
    feat_sum, w_feat, w_sup, pairs_total, pairs_kept = extractor.extract_single_weight_fused(
        meta=meta,
        feat_2d=torch.zeros(2, 2, 4, dtype=torch.float32),
        height=2,
        width=2,
        num_gaussians=3,
        weight_threshold=0.0,
    )
    assert feat_sum.shape == (3, 4)
    assert torch.allclose(w_feat, torch.ones(3))
    assert torch.allclose(w_sup, torch.full((3,), 2.0))
    assert pairs_total == 10
    assert pairs_kept == 7


def test_extract_single_weight_fused_rejects_bad_meta_dtype():
    extractor = AlphaTWeightExtractorV2.__new__(AlphaTWeightExtractorV2)
    meta = {
        "means2d": torch.zeros(3, 2, dtype=torch.float32),
        "conics": torch.zeros(3, 3, dtype=torch.float32),
        "opacities": torch.ones(3, dtype=torch.float32),
        "tile_size": 16,
        "isect_offsets": torch.zeros(1, 1, dtype=torch.int32),
        "flatten_ids": torch.zeros(0, dtype=torch.int32),
        "gaussian_ids": torch.tensor([0, 1, 2], dtype=torch.int32),
    }
    try:
        extractor.extract_single_weight_fused(
            meta=meta,
            feat_2d=torch.zeros(2, 2, 4, dtype=torch.float32),
            height=2,
            width=2,
            num_gaussians=3,
            weight_threshold=0.0,
        )
    except TypeError as e:
        assert "int64" in str(e)
    else:
        raise AssertionError("Expected TypeError for non-int64 gaussian_ids")


def test_feat_only_autograd_function_backpropagates_to_feat2d(monkeypatch):
    called = {"backward": False}

    def _fake_forward(**kwargs):
        feat2d = kwargs["feat2d"]
        num_gaussians = int(kwargs["num_gaussians"])
        channels = int(feat2d.shape[-1])
        feat_sum = feat2d.sum().expand(num_gaussians, channels).contiguous()
        w = torch.ones(num_gaussians, dtype=torch.float32, device=feat2d.device)
        z = torch.zeros(1, dtype=torch.long, device=feat2d.device)
        return feat_sum, w, w, z, z

    def _fake_backward(**kwargs):
        called["backward"] = True
        feat_h = int(kwargs["feat_h"])
        feat_w = int(kwargs["feat_w"])
        channels = int(kwargs["channels"])
        return torch.ones(feat_h, feat_w, channels, dtype=torch.float32, device=kwargs["grad_feat_sum"].device)

    monkeypatch.setattr(extractor_v2_mod, "rasterize_and_backproject_in_range", _fake_forward)
    monkeypatch.setattr(extractor_v2_mod, "backproject_feature_grad_in_range", _fake_backward)

    feat2d = torch.randn(2, 2, 3, requires_grad=True)
    out = extractor_v2_mod._RasterizeAndBackprojectFeatOnlyFn.apply(
        torch.zeros(1, 2),
        torch.zeros(1, 3),
        torch.ones(1),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(0, dtype=torch.int32),
        torch.zeros(1, dtype=torch.long),
        feat2d,
        2,
        2,
        16,
        1,
        0.0,
        True,
    )
    feat_sum = out[0]
    loss = feat_sum.sum()
    loss.backward()
    assert called["backward"] is True
    assert feat2d.grad is not None
