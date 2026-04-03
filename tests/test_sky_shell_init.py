"""Unit tests for models/streetforward/sky_shell_init.py (MultiSceneDataset axes only)."""

from __future__ import annotations

import pytest
import torch

from models.streetforward.sky_shell_init import (
    SKY_UP_MULTISCENE,
    fibonacci_hemisphere_means,
    fibonacci_shell_means,
    sky_base_from_aabb,
)


def test_sky_base_ground_is_y_max() -> None:
    """MultiSceneDataset: y increases downward; ground / dome anchor uses y_max face center."""
    bbx_min = torch.tensor([-1.0, -5.0, -3.0], dtype=torch.float32)
    bbx_max = torch.tensor([3.0, 10.0, 7.0], dtype=torch.float32)
    base = sky_base_from_aabb(bbx_min, bbx_max)
    assert base.shape == (3,)
    assert base[0].item() == pytest.approx(1.0)
    assert base[1].item() == pytest.approx(10.0)
    assert base[2].item() == pytest.approx(2.0)


@pytest.mark.parametrize("resolution", [4, 8, 16])
def test_fibonacci_hemisphere_count_and_sky_halfspace(resolution: int) -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    n_target = resolution**2
    origin = torch.tensor([0.0, 5.0, 0.0], dtype=dtype)
    means = fibonacci_hemisphere_means(
        resolution,
        radius=2.0,
        sky_origin=origin,
        device=device,
        dtype=dtype,
        up=SKY_UP_MULTISCENE,
    )
    assert means.shape == (n_target, 3)
    dirs = (means - origin.unsqueeze(0)) / 2.0
    up = torch.tensor(SKY_UP_MULTISCENE, dtype=dtype)
    dots = (dirs * up.unsqueeze(0)).sum(dim=-1)
    assert (dots >= -1e-4).all()


@pytest.mark.parametrize("resolution", [4, 8])
def test_fibonacci_full_sphere_count(resolution: int) -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    n_target = resolution**2
    origin = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    means = fibonacci_shell_means(
        resolution,
        radius=1.0,
        sky_origin=origin,
        hemisphere=False,
        device=device,
        dtype=dtype,
        up=SKY_UP_MULTISCENE,
    )
    assert means.shape == (n_target, 3)
    dirs = means - origin.unsqueeze(0)
    norms = dirs.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-4
