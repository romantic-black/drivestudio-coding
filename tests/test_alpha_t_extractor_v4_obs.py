from __future__ import annotations

import torch

import models.feature_extractors.alpha_t_extractor_v4 as extractor_v4_mod


def test_multicam_feat_obs_autograd_function_backpropagates_to_feat2d(monkeypatch):
    called = {"backward": False}

    def _fake_forward(**kwargs):
        feat2d = kwargs["feat2d"]
        num_gaussians = int(kwargs["num_gaussians"])
        channels = int(feat2d.shape[-1])
        feat_sum = feat2d.sum().expand(num_gaussians, channels).contiguous()
        w = torch.ones(num_gaussians, dtype=torch.float32, device=feat2d.device)
        obs = torch.zeros(num_gaussians, 2, dtype=torch.float32, device=feat2d.device)
        z = torch.zeros(1, dtype=torch.long, device=feat2d.device)
        return feat_sum, w, w, obs, z, z

    def _fake_backward(**kwargs):
        called["backward"] = True
        v = int(kwargs["isect_offsets"].shape[0])
        feat_h = int(kwargs["feat_h"])
        feat_w = int(kwargs["feat_w"])
        channels = int(kwargs["channels"])
        return torch.ones(v, feat_h, feat_w, channels, dtype=torch.float32, device=kwargs["grad_feat_sum"].device)

    monkeypatch.setattr(extractor_v4_mod, "rasterize_and_backproject_multi_camera_obs_in_range", _fake_forward)
    monkeypatch.setattr(extractor_v4_mod, "backproject_feature_grad_multi_camera_sharded_in_range", _fake_backward)

    feat2d = torch.randn(2, 2, 2, 3, requires_grad=True)
    out = extractor_v4_mod._RasterizeAndBackprojectFeatObsMultiCamFn.apply(
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
        1.0e-6,
    )
    feat_sum = out[0]
    loss = feat_sum.sum()
    loss.backward()
    assert called["backward"] is True
    assert feat2d.grad is not None


def test_obs_formula_single_camera_overlap_zero():
    rho_v = torch.tensor([[1.3], [0.0]], dtype=torch.float32)  # [N, V=1]
    rho = rho_v.sum(dim=1)
    overlap = (rho - rho_v.max(dim=1).values) / (rho + 1.0e-6)
    overlap = torch.where(rho > 0.0, overlap, torch.zeros_like(overlap))
    assert torch.allclose(overlap, torch.zeros_like(overlap), atol=1.0e-6, rtol=1.0e-6)


def test_obs_formula_balanced_two_camera_overlap_half():
    rho_v = torch.tensor([[2.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
    rho = rho_v.sum(dim=1)
    overlap = (rho - rho_v.max(dim=1).values) / (rho + 1.0e-6)
    overlap = torch.where(rho > 0.0, overlap, torch.zeros_like(overlap))
    assert torch.allclose(overlap[0], torch.tensor(0.5), atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(overlap[1], torch.tensor(0.0), atol=1.0e-6, rtol=1.0e-6)
