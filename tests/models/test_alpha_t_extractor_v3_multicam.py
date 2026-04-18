from __future__ import annotations

import pytest
import torch

import models.feature_extractors.alpha_t_extractor_v3 as extractor_v3_mod
from models.feature_extractors.alpha_t_extractor_v3 import AlphaTWeightExtractorV3


def test_multicam_feat_only_autograd_function_backpropagates_to_feat2d(monkeypatch):
    called = {"backward": False}

    def _fake_forward(**kwargs):
        feat2d = kwargs["feat2d"]  # [V, Hf, Wf, C]
        num_gaussians = int(kwargs["num_gaussians"])
        channels = int(feat2d.shape[-1])
        feat_sum = feat2d.sum().expand(num_gaussians, channels).contiguous()
        w = torch.ones(num_gaussians, dtype=torch.float32, device=feat2d.device)
        z = torch.zeros(1, dtype=torch.long, device=feat2d.device)
        return feat_sum, w, w, z, z

    def _fake_backward(**kwargs):
        called["backward"] = True
        V = int(kwargs["isect_offsets"].shape[0])
        feat_h = int(kwargs["feat_h"])
        feat_w = int(kwargs["feat_w"])
        channels = int(kwargs["channels"])
        return torch.ones(V, feat_h, feat_w, channels, dtype=torch.float32, device=kwargs["grad_feat_sum"].device)

    monkeypatch.setattr(extractor_v3_mod, "rasterize_and_backproject_multi_camera_in_range", _fake_forward)
    monkeypatch.setattr(extractor_v3_mod, "backproject_feature_grad_multi_camera_sharded_in_range", _fake_backward)

    feat2d = torch.randn(2, 2, 2, 3, requires_grad=True)
    out = extractor_v3_mod._RasterizeAndBackprojectFeatOnlyMultiCamFn.apply(
        torch.zeros(1, 2),
        torch.zeros(1, 3),
        torch.ones(1),
        torch.zeros(2, 1, 1, dtype=torch.int32),
        torch.zeros(0, dtype=torch.int32),
        torch.zeros(1, dtype=torch.long),
        feat2d,
        2,
        2,
        16,
        1,
        None,
        0.0,
        True,
    )
    feat_sum = out[0]
    loss = feat_sum.sum()
    loss.backward()
    assert called["backward"] is True
    assert feat2d.grad is not None


def test_v3_multicam_streaming_matches_sum_then_divide_semantics(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for v3 multi-camera streaming path")
    extractor = AlphaTWeightExtractorV3.__new__(AlphaTWeightExtractorV3)
    extractor.tile_size = 16
    extractor.sh_degree = 0

    means2d = torch.zeros(2, 2, dtype=torch.float32)
    conics = torch.zeros(2, 3, dtype=torch.float32)
    opac = torch.ones(2, dtype=torch.float32)
    gids = torch.tensor([0, 1], dtype=torch.long)
    flatten = torch.tensor([0, 1], dtype=torch.int32)
    offsets = torch.zeros(2, 1, 1, dtype=torch.int32)

    class _DummyCam:
        def __init__(self):
            self.camtoworlds = torch.eye(4).unsqueeze(0)
            self.K = torch.eye(3).unsqueeze(0)

    cams = [_DummyCam(), _DummyCam()]

    def _fake_resolve_intrinsics(cam):
        return cam.K

    called = {"render": 0}

    def _fake_renderer(**kwargs):
        called["render"] += 1
        viewmats = kwargs["viewmats"]
        ks = kwargs["Ks"]
        assert viewmats.shape[0] == len(cams)
        assert ks.shape[0] == len(cams)
        assert kwargs["packed"] is True
        del kwargs
        meta = {
            "means2d": means2d,
            "conics": conics,
            "opacities": opac,
            "gaussian_ids": gids,
            "flatten_ids": flatten,
            "isect_offsets": offsets,
            "tile_size": 16,
        }
        return None, None, meta

    def _fake_multi_fused(**kwargs):
        n = int(kwargs["num_gaussians"])
        feat = torch.zeros(n, 3, dtype=torch.float32)
        feat[0] = torch.tensor([6.0, 6.0, 6.0])
        feat[1] = torch.tensor([2.0, 2.0, 2.0])
        w_feat = torch.tensor([3.0, 2.0], dtype=torch.float32)
        w_sup = torch.tensor([9.0, 8.0], dtype=torch.float32)
        total = torch.tensor([12], dtype=torch.long)
        kept = torch.tensor([7], dtype=torch.long)
        return feat, w_feat, w_sup, total, kept

    extractor._resolve_intrinsics = _fake_resolve_intrinsics
    extractor.renderer = _fake_renderer
    monkeypatch.setattr(extractor_v3_mod, "rasterize_and_backproject_multi_camera_in_range", _fake_multi_fused)
    monkeypatch.setattr(extractor_v3_mod, "backproject_feature_grad_multi_camera_sharded_in_range", _fake_multi_fused)

    class _BP:
        eps = 1e-8
        weight_threshold = 0.0

    features_2d = torch.zeros(2, 2, 2, 3, dtype=torch.float32, device="cuda")
    feat_out, acc_w, stats = extractor.render_and_backproject_streaming_fused_multi_camera(
        gaussians={
            "means": torch.zeros(2, 3, device="cuda"),
            "quats": torch.zeros(2, 4, device="cuda"),
            "scales": torch.zeros(2, 3, device="cuda"),
            "opacities": torch.zeros(2, device="cuda"),
            "colors": torch.zeros(2, 3, device="cuda"),
        },
        cameras=cams,
        features_2d=features_2d,
        height=2,
        width=2,
        num_gaussians=2,
        backprojector=_BP(),
        return_accumulated_weights=True,
        return_debug_stats=True,
    )
    # sum-then-divide semantics
    assert torch.allclose(feat_out[0], torch.tensor([2.0, 2.0, 2.0]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(feat_out[1], torch.tensor([1.0, 1.0, 1.0]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(acc_w, torch.tensor([9.0, 8.0]), atol=1e-6, rtol=1e-6)
    assert stats["pairs_total"] == 12
    assert stats["pairs_after_threshold"] == 7
    assert stats["nnz_total"] == 2
    assert stats["isects_total"] == 2
    assert called["render"] == 1


def test_v3_no_gated_api_contract():
    assert not hasattr(AlphaTWeightExtractorV3, "render_and_backproject_streaming_fused_multi_camera_gated")


def test_v3_multicam_forwards_pair_valid_mask_to_fused_op(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for v3 multi-camera streaming path")
    extractor = AlphaTWeightExtractorV3.__new__(AlphaTWeightExtractorV3)
    extractor.tile_size = 16
    extractor.sh_degree = 0

    means2d = torch.zeros(2, 2, dtype=torch.float32)
    conics = torch.zeros(2, 3, dtype=torch.float32)
    opac = torch.ones(2, dtype=torch.float32)
    gids = torch.tensor([0, 1], dtype=torch.long)
    flatten = torch.tensor([0, 1], dtype=torch.int32)
    offsets = torch.zeros(2, 1, 1, dtype=torch.int32)

    class _DummyCam:
        def __init__(self):
            self.camtoworlds = torch.eye(4).unsqueeze(0)
            self.K = torch.eye(3).unsqueeze(0)

    cams = [_DummyCam(), _DummyCam()]

    def _fake_resolve_intrinsics(cam):
        return cam.K

    def _fake_renderer(**kwargs):
        del kwargs
        meta = {
            "means2d": means2d,
            "conics": conics,
            "opacities": opac,
            "gaussian_ids": gids,
            "flatten_ids": flatten,
            "isect_offsets": offsets,
            "tile_size": 16,
        }
        return None, None, meta

    called = {"pair_valid_masks": []}

    def _fake_multi_fused(**kwargs):
        called["pair_valid_masks"].append(kwargs["pair_valid_mask"])
        n = int(kwargs["num_gaussians"])
        feat = torch.zeros(n, 3, dtype=torch.float32)
        w_feat = torch.ones(n, dtype=torch.float32)
        w_sup = torch.ones(n, dtype=torch.float32)
        total = torch.tensor([2], dtype=torch.long)
        kept = torch.tensor([2], dtype=torch.long)
        return feat, w_feat, w_sup, total, kept

    extractor._resolve_intrinsics = _fake_resolve_intrinsics
    extractor.renderer = _fake_renderer
    monkeypatch.setattr(extractor_v3_mod, "rasterize_and_backproject_multi_camera_in_range", _fake_multi_fused)
    monkeypatch.setattr(extractor_v3_mod, "backproject_feature_grad_multi_camera_sharded_in_range", _fake_multi_fused)

    class _BP:
        eps = 1e-8
        weight_threshold = 0.0

    features_2d = torch.zeros(2, 2, 2, 3, dtype=torch.float32, device="cuda")
    source_pair_valid_mask = torch.ones(2, 2, 2, dtype=torch.bool, device="cuda")
    source_pair_valid_mask[0, 0, 0] = False
    _feat_out, _acc_w, _stats = extractor.render_and_backproject_streaming_fused_multi_camera(
        gaussians={
            "means": torch.zeros(2, 3, device="cuda"),
            "quats": torch.zeros(2, 4, device="cuda"),
            "scales": torch.zeros(2, 3, device="cuda"),
            "opacities": torch.zeros(2, device="cuda"),
            "colors": torch.zeros(2, 3, device="cuda"),
        },
        cameras=cams,
        features_2d=features_2d,
        height=2,
        width=2,
        num_gaussians=2,
        backprojector=_BP(),
        source_pair_valid_mask=source_pair_valid_mask,
        return_accumulated_weights=True,
        return_debug_stats=True,
    )
    assert any(m is not None for m in called["pair_valid_masks"])
    assert any(torch.equal(m, source_pair_valid_mask) for m in called["pair_valid_masks"] if m is not None)
