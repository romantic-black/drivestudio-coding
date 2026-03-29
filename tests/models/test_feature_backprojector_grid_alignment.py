"""grid_sample align_corners consistency for FeatureBackprojector pixel_id mapping."""

import pytest
import torch
import torch.nn.functional as F

from models.feature_extractors.feature_2d_backprojector import (
    pixel_ids_to_grid_sample_coords_align_corners,
)


def test_align_corners_formula_matches_pytorch_indexing():
    """Integer pixel index k should map to grid 2*k/(S-1)-1 for align_corners=True."""
    height, width = 5, 7
    for i in range(height):
        for j in range(width):
            pid = torch.tensor([i * width + j], dtype=torch.long)
            coords = pixel_ids_to_grid_sample_coords_align_corners(
                pid, height, width, torch.float32, pid.device
            )
            gx, gy = coords[0, 0].item(), coords[0, 1].item()
            exp_gx = 2.0 * j / float(width - 1) - 1.0
            exp_gy = 2.0 * i / float(height - 1) - 1.0
            assert abs(gx - exp_gx) < 1e-5 and abs(gy - exp_gy) < 1e-5


def test_grid_sample_hits_pixel_centers_under_align_corners():
    """Sampling at computed grid should read the same storage as direct indexing."""
    H, W = 4, 6
    # [H, W, C] channels-last for backprojector path
    feat = torch.arange(H * W, dtype=torch.float32).view(H, W, 1)

    for i in range(H):
        for j in range(W):
            pid = torch.tensor([i * W + j], dtype=torch.long)
            coords = pixel_ids_to_grid_sample_coords_align_corners(
                pid, H, W, torch.float32, pid.device
            ).view(1, 1, 1, 2)
            feat_chw = feat.permute(2, 0, 1).unsqueeze(0)
            sampled = F.grid_sample(
                feat_chw, coords, mode="bilinear", align_corners=True, padding_mode="zeros"
            )
            assert torch.allclose(sampled.view(()), feat[i, j, 0])


def test_singleton_hw_no_div_zero():
    pid = torch.tensor([0], dtype=torch.long)
    c = pixel_ids_to_grid_sample_coords_align_corners(
        pid, 1, 1, torch.float32, pid.device
    )
    assert c.shape == (1, 2)
    assert torch.allclose(c, torch.zeros(1, 2))


def test_old_j_over_w_differs_from_align_corners_at_boundary():
    """Regression note: j/W*2-1 does not reach +1 at j=W-1 when W>1."""
    W = 8
    j = W - 1
    wrong = 2.0 * j / float(W) - 1.0
    right = 2.0 * j / float(W - 1) - 1.0
    assert wrong < 1.0 - 1e-6
    assert abs(right - 1.0) < 1e-6
