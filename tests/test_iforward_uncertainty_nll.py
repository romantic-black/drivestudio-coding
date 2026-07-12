from __future__ import annotations

import math

import torch

from models.iforward.uncertainty_losses import (
    grouped_uncertainty_calibration_metrics,
    masked_gaussian_rgb_nll,
    masked_gaussian_rgb_nll_components,
    masked_uncertainty_variance_stats,
    masked_uncertainty_photometric_loss,
    uncertainty_calibration_metrics,
    uncertainty_v2_loss_weights,
)


def test_normalized_precision_uses_relative_bounded_mean_weights_only() -> None:
    pred = torch.full((1, 2, 3), 0.2, requires_grad=True)
    gt = torch.zeros_like(pred)
    # True precisions are [1, 2], so normalized mean weights are [2/3, 4/3].
    logvar = (-torch.log(torch.tensor([[1.0, 2.0]]))).requires_grad_()
    components = masked_gaussian_rgb_nll_components(
        pred,
        gt,
        logvar,
        mask=None,
        normalize_precision_per_reference=True,
        precision_weight_min=0.25,
        precision_weight_max=4.0,
    )
    assert torch.allclose(
        components.mean_precision,
        torch.tensor([[2.0 / 3.0, 4.0 / 3.0]]),
    )
    assert torch.allclose(components.precision_normalizer, torch.tensor(1.5))
    assert torch.allclose(components.precision, torch.tensor([[1.0, 2.0]]))

    components.mean_path.backward()
    per_pixel_grad = pred.grad.detach().abs().mean(dim=-1)
    assert torch.allclose(per_pixel_grad[0, 1] / per_pixel_grad[0, 0], torch.tensor(2.0))
    assert logvar.grad is None


def test_normalized_precision_clamps_and_low_alpha_falls_back_to_one() -> None:
    pred = torch.zeros(1, 2, 3)
    gt = torch.zeros_like(pred)
    logvar = -torch.log(torch.tensor([[1000.0, 1.0]]))
    components = masked_gaussian_rgb_nll_components(
        pred,
        gt,
        logvar,
        mask=None,
        normalize_precision_per_reference=True,
        precision_weight_min=0.25,
        precision_weight_max=4.0,
        alpha=torch.tensor([[0.0, 1.0]]),
        alpha_uncertainty_min=0.25,
        alpha_uncertainty_full=0.75,
    )
    assert torch.allclose(components.mean_precision, torch.tensor([[1.0, 0.25]]))
    assert components.precision_clipped_low.tolist() == [[False, True]]
    stats_loss, stats = masked_gaussian_rgb_nll(
        pred,
        gt,
        logvar,
        mask=None,
        normalize_precision_per_reference=True,
        precision_weight_min=0.25,
        precision_weight_max=4.0,
        alpha=torch.tensor([[0.0, 1.0]]),
        collect_precision_quantiles=True,
    )
    assert torch.isfinite(stats_loss)
    assert stats["precision_alpha_fallback_ratio"] == 0.5
    assert stats["precision_clipped_low_ratio"] == 0.5
    assert stats["precision_weight_p50"] >= 0.25


def test_decoupled_nll_routes_mean_and_uncertainty_gradients() -> None:
    pred = torch.full((2, 2, 3), 0.25, requires_grad=True)
    gt = torch.zeros_like(pred)
    logvar = torch.full((2, 2), math.log(0.05), requires_grad=True)
    loss, stats = masked_gaussian_rgb_nll(
        pred,
        gt,
        logvar,
        mask=torch.ones(2, 2),
        calibration_weight=0.1,
    )
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0.0
    assert logvar.grad is not None and logvar.grad.abs().sum().item() > 0.0
    assert stats["loss_uncertainty_mean_path"] > 0.0


def test_nll_calibration_optimum_matches_residual_energy() -> None:
    pred = torch.full((1, 1, 3), 0.2)
    gt = torch.zeros_like(pred)
    optimum = math.log(0.2**2)
    for offset, expected_sign in ((-0.5, -1), (0.5, 1)):
        logvar = torch.tensor([[optimum + offset]], requires_grad=True)
        loss, _ = masked_gaussian_rgb_nll(pred, gt, logvar, mask=None, calibration_weight=1.0)
        loss.backward()
        assert int(torch.sign(logvar.grad).item()) == expected_sign


def test_invalid_mask_is_finite_and_raw_metrics_remain_unweighted() -> None:
    pred = torch.ones(2, 2, 3, requires_grad=True)
    gt = torch.zeros_like(pred)
    logvar = torch.zeros(2, 2, requires_grad=True)
    loss0, stats0 = masked_gaussian_rgb_nll(pred, gt, logvar, mask=torch.zeros(2, 2))
    assert torch.isfinite(loss0)
    assert stats0["skipped_no_valid_pixels"] == 1.0

    _, stats = masked_uncertainty_photometric_loss(
        pred,
        gt,
        torch.full((2, 2), math.log(0.25)),
        mask=None,
        nll_weight=0.5,
        calibration_weight=0.1,
        raw_l1_anchor_weight=0.25,
        raw_ssim_anchor_weight=0.0,
    )
    assert stats["l1"] == 1.0
    assert stats["psnr"] == 0.0


def test_calibration_metrics_rank_good_uncertainty_above_bad() -> None:
    gt = torch.zeros(2, 2, 3)
    pred = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], [[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]]
    )
    error = pred.square().mean(dim=-1)
    good = uncertainty_calibration_metrics(pred, gt, error + 1.0e-4, mask=None)
    bad = uncertainty_calibration_metrics(pred, gt, torch.flip(error, dims=(0, 1)) + 1.0e-4, mask=None)
    assert good["error_uncertainty_spearman"] > bad["error_uncertainty_spearman"]
    assert good["ause"] <= bad["ause"]


def test_v2_warmup_weights_have_exact_boundaries() -> None:
    expected = {
        0: (0.0, 1.0),
        1999: (0.0, 1.0),
        2000: (0.0, 1.0),
        6000: (0.25, 0.625),
        10000: (0.50, 0.25),
    }
    for step, (mean_weight, l1_weight) in expected.items():
        weights = uncertainty_v2_loss_weights(step)
        assert weights["mean_nll_weight"] == mean_weight
        assert weights["raw_l1_anchor_weight"] == l1_weight
        assert weights["calibration_path_weight"] == 0.05


def test_v2_explicit_paths_use_alpha_only_for_calibration() -> None:
    pred = torch.full((1, 2, 3), 0.2, requires_grad=True)
    gt = torch.zeros_like(pred)
    logvar = torch.full((1, 2), math.log(0.04), requires_grad=True)
    loss, stats = masked_uncertainty_photometric_loss(
        pred,
        gt,
        logvar,
        mask=torch.ones(1, 2),
        calibration_mask=torch.tensor([[False, True]]),
        nll_weight=0.5,
        calibration_weight=0.1,
        mean_nll_weight=0.0,
        calibration_path_weight=0.05,
        raw_l1_anchor_weight=0.0,
        raw_ssim_anchor_weight=0.0,
    )
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() == 0.0
    assert logvar.grad is not None
    assert logvar.grad[0, 0].item() == 0.0
    assert torch.isfinite(logvar.grad[0, 1])
    assert stats["calibration_valid_ratio"] == 0.5
    assert stats["loss_weight_mean_nll"] == 0.0
    assert stats["loss_weight_calibration"] == 0.05


def test_masked_variance_and_grouped_calibration_stats_are_finite() -> None:
    shape = (3, 3)
    within = torch.full(shape, 0.01)
    background = torch.full(shape, 0.002)
    disagreement = torch.arange(9, dtype=torch.float32).reshape(shape) * 0.001
    total = (within + background + disagreement).clamp(max=0.25)
    valid = torch.zeros(shape)
    stats = masked_uncertainty_variance_stats(
        within_variance=within,
        background_variance=background,
        disagreement_variance=disagreement,
        total_variance=total,
        loss_variance=within,
        mask=valid,
        variance_floor=1.0e-4,
        variance_max=0.25,
    )
    assert stats["pixel_valid_count"] == 0.0
    assert all(math.isfinite(value) for value in stats.values())

    gt = torch.zeros(3, 3, 3)
    pred = torch.linspace(0.0, 0.3, 27).reshape(3, 3, 3)
    grouped = grouped_uncertainty_calibration_metrics(
        pred,
        gt,
        {"within": within, "disagreement": disagreement, "total": total},
        mask=torch.ones(shape),
        alpha=torch.tensor(
            [[0.0, 0.02, 0.2], [0.3, 0.6, 0.8], [1.0, 1.0, 1.0]]
        ),
        dynamic_mask=torch.eye(3),
    )
    assert "calibration/within/error_uncertainty_spearman" in grouped
    assert "calibration/total/dynamic/pixel_count" in grouped
    assert all(math.isfinite(value) for value in grouped.values())
