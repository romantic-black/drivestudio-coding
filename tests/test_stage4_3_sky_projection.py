"""Projection tests for MinimalStreetForwardStage4_3 sky selective mask (pinhole vs trainer)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.feature_extractors.alpha_t_extractor import AlphaTWeightExtractor, _get_viewmat
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3


def _manual_uv(
    means_world: torch.Tensor,
    c2w: torch.Tensor,
    k_mat: torch.Tensor,
    h_img: int,
    w_img: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c2w = c2w.unsqueeze(0) if c2w.dim() == 2 else c2w
    if tuple(c2w.shape[-2:]) == (3, 4):
        pad = torch.tensor([0.0, 0.0, 0.0, 1.0], device=c2w.device, dtype=c2w.dtype).view(1, 1, 4).expand(
            c2w.shape[0], 1, 4
        )
        c2w = torch.cat([c2w, pad], dim=-2)
    vm = _get_viewmat(c2w.squeeze(0))
    if vm.dim() == 2:
        vm = vm.unsqueeze(0)
    vm = vm[0]
    n = means_world.shape[0]
    ones = torch.ones(n, 1, device=means_world.device, dtype=means_world.dtype)
    pts_h = torch.cat([means_world, ones], dim=-1)
    cam = (vm @ pts_h.T).T
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
    k = k_mat if k_mat.dim() == 2 else k_mat[0]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    u = (fx * x / z + cx).long()
    v = (fy * y / z + cy).long()
    return u, v, z


@pytest.mark.parametrize(
    "mean_xyz,expected_u,expected_v",
    [
        ((0.0, 0.0, 5.0), 160, 120),
        ((1.0, -0.5, 10.0), 170, 115),
    ],
)
def test_sky_projection_matches_manual_pinhole(
    mean_xyz: tuple[float, float, float],
    expected_u: int,
    expected_v: int,
) -> None:
    h_img, w_img = 240, 320
    fx, fy, cx, cy = 100.0, 100.0, 160.0, 120.0
    k_mat = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)

    means = torch.tensor([mean_xyz], dtype=torch.float32)
    u_e, v_e, z_e = _manual_uv(means, c2w, k_mat, h_img, w_img)
    assert z_e[0].item() > 0
    assert int(u_e[0].item()) == expected_u
    assert int(v_e[0].item()) == expected_v

    sky_mask = torch.zeros(h_img, w_img, dtype=torch.float32)
    sky_mask[expected_v, expected_u] = 1.0
    gt_image = torch.zeros(h_img, w_img, 3, dtype=torch.float32)
    view = SimpleNamespace(camtoworlds=c2w, Ks=k_mat.unsqueeze(0))
    target = {"view": view, "gt_image": gt_image, "sky_mask": sky_mask}

    probe = SimpleNamespace(
        device=torch.device("cpu"),
        _ensure_c2w_4x4=MinimalStreetForwardStage4_3._ensure_c2w_4x4,
    )
    hit = MinimalStreetForwardStage4_3._sky_points_visible_in_target_sky_pixels(probe, means, target)
    assert hit.shape == (1,)
    assert bool(hit[0].item()) is True


def test_sky_projection_false_behind_camera() -> None:
    h_img, w_img = 64, 64
    k_mat = torch.tensor([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    means = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32)
    sky_mask = torch.ones(h_img, w_img, dtype=torch.float32)
    gt_image = torch.zeros(h_img, w_img, 3, dtype=torch.float32)
    view = SimpleNamespace(camtoworlds=c2w, Ks=k_mat.unsqueeze(0))
    target = {"view": view, "gt_image": gt_image, "sky_mask": sky_mask}
    probe = SimpleNamespace(
        device=torch.device("cpu"),
        _ensure_c2w_4x4=MinimalStreetForwardStage4_3._ensure_c2w_4x4,
    )
    hit = MinimalStreetForwardStage4_3._sky_points_visible_in_target_sky_pixels(probe, means, target)
    assert not bool(hit[0].item())


def test_resolve_intrinsics_matches_view_ks() -> None:
    k_mat = torch.tensor([[100.0, 0.0, 160.0], [0.0, 100.0, 120.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    view = SimpleNamespace(Ks=k_mat.unsqueeze(0), camtoworlds=torch.eye(4))
    k2 = AlphaTWeightExtractor._resolve_intrinsics(view)[0]
    assert torch.allclose(k2, k_mat)
