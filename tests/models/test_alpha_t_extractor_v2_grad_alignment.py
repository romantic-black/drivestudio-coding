from __future__ import annotations

import torch

import models.feature_extractors.alpha_t_extractor_v2 as extractor_v2_mod
from models.feature_extractors.feature_2d_backprojector import FeatureBackprojector


def test_feat_only_fn_marks_non_diff_outputs(monkeypatch):
    def _fake_forward(**kwargs):
        feat2d = kwargs["feat2d"]
        feat_sum = torch.ones(2, feat2d.shape[-1], dtype=feat2d.dtype, device=feat2d.device)
        w = torch.ones(2, dtype=feat2d.dtype, device=feat2d.device)
        pairs = torch.zeros(1, dtype=torch.long, device=feat2d.device)
        return feat_sum, w, w, pairs, pairs

    def _fake_backward(**kwargs):
        return torch.ones(
            int(kwargs["feat_h"]),
            int(kwargs["feat_w"]),
            int(kwargs["channels"]),
            dtype=torch.float32,
            device=kwargs["grad_feat_sum"].device,
        )

    monkeypatch.setattr(extractor_v2_mod, "rasterize_and_backproject_in_range", _fake_forward)
    monkeypatch.setattr(extractor_v2_mod, "backproject_feature_grad_in_range", _fake_backward)

    feat2d = torch.randn(2, 2, 4, requires_grad=True)
    feat_sum, w_feat, w_sup, pairs_total, pairs_kept = extractor_v2_mod._RasterizeAndBackprojectFeatOnlyFn.apply(
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
        2,
        0.0,
        True,
    )
    assert feat_sum.requires_grad is True
    assert w_feat.requires_grad is False
    assert w_sup.requires_grad is False
    assert pairs_total.requires_grad is False
    assert pairs_kept.requires_grad is False

    feat_sum.sum().backward()
    assert feat2d.grad is not None


def test_feat_only_fn_forward_backward_align_with_v1_toy(monkeypatch):
    # Toy setup: sampling at exact pixel centers on a 2x2 map (align_corners=True),
    # so bilinear reduces to direct pixel fetch and gradients are easy to verify.
    gaussian_ids = torch.tensor([0, 1], dtype=torch.long)
    pixel_ids = torch.tensor([0, 3], dtype=torch.long)  # (0,0) and (1,1)
    weights = torch.tensor([0.2, 0.8], dtype=torch.float32)

    def _fake_forward(**kwargs):
        feat2d = kwargs["feat2d"]  # [H, W, C]
        threshold = float(kwargs["weight_threshold"])
        num_gaussians = int(kwargs["num_gaussians"])
        channels = int(feat2d.shape[-1])

        feat_sum = torch.zeros(num_gaussians, channels, dtype=feat2d.dtype, device=feat2d.device)
        w_feat = torch.zeros(num_gaussians, dtype=feat2d.dtype, device=feat2d.device)
        w_sup = torch.zeros(num_gaussians, dtype=feat2d.dtype, device=feat2d.device)
        kept = 0
        for gid, pid, w in zip(gaussian_ids.tolist(), pixel_ids.tolist(), weights.tolist()):
            i = pid // 2
            j = pid % 2
            w_sup[gid] += w
            if w >= threshold:
                feat_sum[gid] += w * feat2d[i, j]
                w_feat[gid] += w
                kept += 1
        pairs = torch.tensor([len(gaussian_ids)], dtype=torch.long, device=feat2d.device)
        kept_pairs = torch.tensor([kept], dtype=torch.long, device=feat2d.device)
        return feat_sum, w_feat, w_sup, pairs, kept_pairs

    def _fake_backward(**kwargs):
        grad_feat_sum = kwargs["grad_feat_sum"]  # [N, C]
        threshold = float(kwargs["weight_threshold"])
        grad_feat2d = torch.zeros(2, 2, grad_feat_sum.shape[1], dtype=grad_feat_sum.dtype, device=grad_feat_sum.device)
        for gid, pid, w in zip(gaussian_ids.tolist(), pixel_ids.tolist(), weights.tolist()):
            if w >= threshold:
                i = pid // 2
                j = pid % 2
                grad_feat2d[i, j] += w * grad_feat_sum[gid]
        return grad_feat2d

    monkeypatch.setattr(extractor_v2_mod, "rasterize_and_backproject_in_range", _fake_forward)
    monkeypatch.setattr(extractor_v2_mod, "backproject_feature_grad_in_range", _fake_backward)

    feat2d_v2 = torch.randn(2, 2, 3, requires_grad=True)
    feat2d_v1 = feat2d_v2.detach().clone().requires_grad_(True)
    grad_feat_sum = torch.randn(2, 3)

    # v2 custom fn
    feat_sum_v2, w_feat_v2, w_sup_v2, _, _ = extractor_v2_mod._RasterizeAndBackprojectFeatOnlyFn.apply(
        torch.zeros(2, 2),
        torch.zeros(2, 3),
        torch.ones(2),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(0, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.long),
        feat2d_v2,
        2,
        2,
        16,
        2,
        0.5,
        True,
    )

    # v1 reference path
    bp = FeatureBackprojector(weight_threshold=0.5)
    weight_info = {"gaussian_ids": gaussian_ids, "pixel_ids": pixel_ids, "weights": weights}
    feat_sum_v1, w_feat_v1, w_sup_v1 = bp.backproject_single_view(
        feat2d_v1,
        weight_info,
        height=2,
        width=2,
        num_gaussians=2,
        return_support_weight=True,
    )

    assert torch.allclose(feat_sum_v2, feat_sum_v1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(w_feat_v2, w_feat_v1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(w_sup_v2, w_sup_v1, atol=1e-6, rtol=1e-6)

    feat_sum_v2.backward(grad_feat_sum)
    feat_sum_v1.backward(grad_feat_sum)
    assert torch.allclose(feat2d_v2.grad, feat2d_v1.grad, atol=1e-6, rtol=1e-6)


def test_feat_only_fn_threshold_semantics_and_backward_kept_pairs(monkeypatch):
    gaussian_ids = torch.tensor([0, 1], dtype=torch.long)
    pixel_ids = torch.tensor([0, 3], dtype=torch.long)
    weights = torch.tensor([0.2, 0.8], dtype=torch.float32)

    def _fake_forward(**kwargs):
        feat2d = kwargs["feat2d"]
        threshold = float(kwargs["weight_threshold"])
        feat_sum = torch.zeros(2, feat2d.shape[-1], dtype=feat2d.dtype, device=feat2d.device)
        w_feat = torch.zeros(2, dtype=feat2d.dtype, device=feat2d.device)
        w_sup = torch.zeros(2, dtype=feat2d.dtype, device=feat2d.device)
        kept = 0
        for gid, pid, w in zip(gaussian_ids.tolist(), pixel_ids.tolist(), weights.tolist()):
            i = pid // 2
            j = pid % 2
            w_sup[gid] += w
            if w >= threshold:
                feat_sum[gid] += w * feat2d[i, j]
                w_feat[gid] += w
                kept += 1
        return (
            feat_sum,
            w_feat,
            w_sup,
            torch.tensor([2], dtype=torch.long, device=feat2d.device),
            torch.tensor([kept], dtype=torch.long, device=feat2d.device),
        )

    def _fake_backward(**kwargs):
        grad_feat_sum = kwargs["grad_feat_sum"]
        threshold = float(kwargs["weight_threshold"])
        grad_feat2d = torch.zeros(2, 2, grad_feat_sum.shape[-1], dtype=grad_feat_sum.dtype, device=grad_feat_sum.device)
        for gid, pid, w in zip(gaussian_ids.tolist(), pixel_ids.tolist(), weights.tolist()):
            if w >= threshold:
                i = pid // 2
                j = pid % 2
                grad_feat2d[i, j] += w * grad_feat_sum[gid]
        return grad_feat2d

    monkeypatch.setattr(extractor_v2_mod, "rasterize_and_backproject_in_range", _fake_forward)
    monkeypatch.setattr(extractor_v2_mod, "backproject_feature_grad_in_range", _fake_backward)

    feat2d_a = torch.randn(2, 2, 2, requires_grad=True)
    feat2d_b = feat2d_a.detach().clone().requires_grad_(True)
    grad_feat_sum = torch.ones(2, 2)

    out0 = extractor_v2_mod._RasterizeAndBackprojectFeatOnlyFn.apply(
        torch.zeros(2, 2), torch.zeros(2, 3), torch.ones(2),
        torch.zeros(1, 1, dtype=torch.int32), torch.zeros(0, dtype=torch.int32), torch.tensor([0, 1], dtype=torch.long),
        feat2d_a, 2, 2, 16, 2, 0.0, True,
    )
    out1 = extractor_v2_mod._RasterizeAndBackprojectFeatOnlyFn.apply(
        torch.zeros(2, 2), torch.zeros(2, 3), torch.ones(2),
        torch.zeros(1, 1, dtype=torch.int32), torch.zeros(0, dtype=torch.int32), torch.tensor([0, 1], dtype=torch.long),
        feat2d_b, 2, 2, 16, 2, 0.5, True,
    )

    feat_sum0, w_feat0, w_sup0, _, _ = out0
    feat_sum1, w_feat1, w_sup1, _, _ = out1
    assert torch.allclose(w_sup0, w_sup1, atol=1e-6, rtol=1e-6)  # support independent of threshold
    assert not torch.allclose(w_feat0, w_feat1, atol=1e-6, rtol=1e-6)  # feature weight affected

    feat_sum0.backward(grad_feat_sum)
    feat_sum1.backward(grad_feat_sum)
    # threshold=0.5 removes gaussian 0 / pixel (0,0) weight 0.2 contribution
    assert feat2d_a.grad[0, 0].abs().sum() > 0
    assert torch.allclose(feat2d_b.grad[0, 0], torch.zeros_like(feat2d_b.grad[0, 0]), atol=1e-6, rtol=1e-6)

