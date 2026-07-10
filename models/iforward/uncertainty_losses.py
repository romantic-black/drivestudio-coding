from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from models.streetforward.stage6_0.phase_a_losses import masked_rgb_loss


def _mask2d(mask: Optional[torch.Tensor], *, ref: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(ref.shape[:2], device=ref.device, dtype=torch.bool)
    out = mask.to(device=ref.device)
    if out.dim() == 3 and int(out.shape[-1]) == 1:
        out = out.squeeze(-1)
    if out.dim() != 2:
        raise ValueError(f"mask must be [H,W] or [H,W,1], got {tuple(out.shape)}")
    return out > 0.5


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not bool(valid.any().item()):
        return value.sum() * 0.0
    return value[valid].mean()


@dataclass
class DecoupledGaussianNLLComponents:
    mean_path: torch.Tensor
    calibration_path: torch.Tensor
    precision: torch.Tensor
    mean_valid: torch.Tensor
    calibration_valid: torch.Tensor


def masked_gaussian_rgb_nll_components(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    pixel_logvar: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    calibration_mask: Optional[torch.Tensor] = None,
    precision_floor: float = 0.0,
) -> DecoupledGaussianNLLComponents:
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred_rgb.shape)} vs {tuple(gt_rgb.shape)}")
    if tuple(pixel_logvar.shape) != tuple(pred_rgb.shape[:2]):
        raise ValueError(
            f"pixel_logvar must be [H,W], got {tuple(pixel_logvar.shape)} for RGB {tuple(pred_rgb.shape)}"
        )
    mean_valid = _mask2d(mask, ref=pred_rgb)
    calibration_valid = (
        mean_valid
        if calibration_mask is None
        else mean_valid & _mask2d(calibration_mask, ref=pred_rgb)
    )
    resid2 = (pred_rgb - gt_rgb).square().mean(dim=-1)
    precision = torch.exp(-pixel_logvar)
    mean_precision = precision.detach()
    if float(precision_floor) > 0.0 and bool(mean_valid.any().item()):
        relative_floor = torch.median(mean_precision[mean_valid]) * float(precision_floor)
        mean_precision = mean_precision.clamp_min(relative_floor)
    mean_path = _masked_mean(0.5 * mean_precision * resid2, mean_valid)
    calibration_path = _masked_mean(
        0.5 * precision * resid2.detach() + 0.5 * pixel_logvar,
        calibration_valid,
    )
    return DecoupledGaussianNLLComponents(
        mean_path=mean_path,
        calibration_path=calibration_path,
        precision=precision,
        mean_valid=mean_valid,
        calibration_valid=calibration_valid,
    )


def masked_gaussian_rgb_nll(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    pixel_logvar: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    calibration_weight: float = 0.10,
    precision_floor: float = 0.0,
    calibration_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    components = masked_gaussian_rgb_nll_components(
        pred_rgb,
        gt_rgb,
        pixel_logvar,
        mask=mask,
        calibration_mask=calibration_mask,
        precision_floor=precision_floor,
    )
    mean_path = components.mean_path
    calibration_path = components.calibration_path
    loss = mean_path + float(calibration_weight) * calibration_path
    mean_valid = components.mean_valid
    calibration_valid = components.calibration_valid
    precision_mean = (
        float(components.precision.detach()[mean_valid].mean().item())
        if bool(mean_valid.any().item())
        else 0.0
    )
    return loss, {
        "loss_uncertainty_nll": float(loss.detach().item()),
        "loss_uncertainty_mean_path": float(mean_path.detach().item()),
        "loss_uncertainty_calibration": float(calibration_path.detach().item()),
        "precision_mean": precision_mean,
        "calibration_valid_ratio": float(calibration_valid.float().mean().item()),
        "skipped_no_valid_pixels": 0.0 if bool(mean_valid.any().item()) else 1.0,
    }


def masked_uncertainty_photometric_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    pixel_logvar: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    nll_weight: float,
    calibration_weight: float,
    raw_l1_anchor_weight: float,
    raw_ssim_anchor_weight: float,
    precision_floor: float = 0.0,
    mean_nll_weight: Optional[float] = None,
    calibration_path_weight: Optional[float] = None,
    calibration_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    components = masked_gaussian_rgb_nll_components(
        pred_rgb,
        gt_rgb,
        pixel_logvar,
        mask=mask,
        calibration_mask=calibration_mask,
        precision_floor=float(precision_floor),
    )
    legacy_nll = components.mean_path + float(calibration_weight) * components.calibration_path
    effective_mean_weight = float(nll_weight) if mean_nll_weight is None else float(mean_nll_weight)
    effective_calibration_weight = (
        float(nll_weight) * float(calibration_weight)
        if calibration_path_weight is None
        else float(calibration_path_weight)
    )
    raw_anchor, raw_stats = masked_rgb_loss(
        pred_rgb,
        gt_rgb,
        mask=mask,
        l1_weight=float(raw_l1_anchor_weight),
        ssim_weight=float(raw_ssim_anchor_weight),
    )
    loss = (
        effective_mean_weight * components.mean_path
        + effective_calibration_weight * components.calibration_path
        + raw_anchor
    )
    mean_valid = components.mean_valid
    calibration_valid = components.calibration_valid
    stats = {
        **raw_stats,
        "loss_uncertainty_nll": float(legacy_nll.detach().item()),
        "loss_uncertainty_mean_path": float(components.mean_path.detach().item()),
        "loss_uncertainty_calibration": float(components.calibration_path.detach().item()),
        "loss_uncertainty_mean_weighted": float(
            (effective_mean_weight * components.mean_path).detach().item()
        ),
        "loss_uncertainty_calibration_weighted": float(
            (effective_calibration_weight * components.calibration_path).detach().item()
        ),
        "precision_mean": (
            float(components.precision.detach()[mean_valid].mean().item())
            if bool(mean_valid.any().item())
            else 0.0
        ),
        "calibration_valid_ratio": float(calibration_valid.float().mean().item()),
        "skipped_no_valid_pixels": 0.0 if bool(mean_valid.any().item()) else 1.0,
    }
    stats.update(
        {
            "loss_raw_l1_anchor": float(raw_l1_anchor_weight) * float(raw_stats.get("l1", 0.0)),
            "loss_raw_ssim_anchor": float(raw_ssim_anchor_weight) * float(raw_stats.get("ssim", 0.0)),
            "loss_uncertainty_photometric": float(loss.detach().item()),
            "loss_weight_mean_nll": effective_mean_weight,
            "loss_weight_calibration": effective_calibration_weight,
            "loss_weight_raw_l1": float(raw_l1_anchor_weight),
            "loss_weight_raw_ssim": float(raw_ssim_anchor_weight),
        }
    )
    return loss, stats


def uncertainty_v2_loss_weights(
    global_step: int,
    *,
    warmup_start: int = 2000,
    warmup_end: int = 10000,
    mean_nll_final: float = 0.50,
    calibration: float = 0.05,
    raw_l1_initial: float = 1.0,
    raw_l1_final: float = 0.25,
) -> Dict[str, float]:
    step = int(global_step)
    if step < int(warmup_start):
        progress = 0.0
    elif step >= int(warmup_end):
        progress = 1.0
    else:
        progress = float(step - int(warmup_start)) / float(max(int(warmup_end) - int(warmup_start), 1))
    return {
        "mean_nll_weight": float(mean_nll_final) * progress,
        "calibration_path_weight": float(calibration),
        "raw_l1_anchor_weight": float(raw_l1_initial)
        + progress * (float(raw_l1_final) - float(raw_l1_initial)),
        "warmup_progress": progress,
    }


def masked_uncertainty_variance_stats(
    *,
    within_variance: torch.Tensor,
    background_variance: torch.Tensor,
    disagreement_variance: torch.Tensor,
    total_variance: torch.Tensor,
    loss_variance: torch.Tensor,
    mask: Optional[torch.Tensor],
    variance_floor: float,
    variance_max: float,
) -> Dict[str, float]:
    valid = _mask2d(mask, ref=within_variance)
    count = int(valid.sum().item())
    if count == 0:
        return {
            "pixel_valid_count": 0.0,
            "pixel_within_mean": 0.0,
            "pixel_background_mean": 0.0,
            "pixel_disagreement_mean": 0.0,
            "pixel_total_variance_mean": 0.0,
            "pixel_loss_variance_mean": 0.0,
            "pixel_disagreement_fraction": 0.0,
            "pixel_loss_floor_ratio": 0.0,
            "pixel_loss_max_ratio": 0.0,
            "pixel_total_max_ratio": 0.0,
        }
    within = within_variance.detach().float()[valid]
    background = background_variance.detach().float()[valid]
    disagreement = disagreement_variance.detach().float()[valid]
    total = total_variance.detach().float()[valid]
    loss = loss_variance.detach().float()[valid]
    return {
        "pixel_valid_count": float(count),
        "pixel_within_mean": float(within.mean().item()),
        "pixel_background_mean": float(background.mean().item()),
        "pixel_disagreement_mean": float(disagreement.mean().item()),
        "pixel_total_variance_mean": float(total.mean().item()),
        "pixel_loss_variance_mean": float(loss.mean().item()),
        "pixel_disagreement_fraction": float(
            (disagreement.sum() / total.sum().clamp_min(1.0e-12)).item()
        ),
        "pixel_loss_floor_ratio": float((loss <= float(variance_floor) + 1.0e-8).float().mean().item()),
        "pixel_loss_max_ratio": float((loss >= float(variance_max) - 1.0e-8).float().mean().item()),
        "pixel_total_max_ratio": float((total >= float(variance_max) - 1.0e-8).float().mean().item()),
    }


def uncertainty_calibration_metrics(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    variance: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    alpha: Optional[torch.Tensor] = None,
    alpha_valid_min: float = 0.01,
    coverages: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
) -> Dict[str, float]:
    valid = _mask2d(mask, ref=pred_rgb)
    if alpha is not None:
        alpha2 = alpha.squeeze(-1) if alpha.dim() == 3 and int(alpha.shape[-1]) == 1 else alpha
        valid = valid & (alpha2.to(device=pred_rgb.device) >= float(alpha_valid_min))
    if int(valid.sum().item()) < 2:
        return {"error_uncertainty_pearson": 0.0, "error_uncertainty_spearman": 0.0, "ause": 0.0}
    error = (pred_rgb.detach() - gt_rgb.detach()).square().mean(dim=-1)[valid].float()
    uncertainty = variance.detach()[valid].float()

    def pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = torch.sqrt(x0.square().sum() * y0.square().sum()).clamp_min(1.0e-12)
        return (x0 * y0).sum() / denom

    rank_error = torch.argsort(torch.argsort(error)).float()
    rank_uncertainty = torch.argsort(torch.argsort(uncertainty)).float()
    out = {
        "error_uncertainty_pearson": float(pearson(error, uncertainty).item()),
        "error_uncertainty_spearman": float(pearson(rank_error, rank_uncertainty).item()),
    }
    pred_order = torch.argsort(uncertainty, descending=False)
    oracle_order = torch.argsort(error, descending=False)
    pred_risks = []
    oracle_risks = []
    n = int(error.numel())
    for coverage in coverages:
        k = max(1, min(n, int(round(float(coverage) * n))))
        pred_risk = error[pred_order[:k]].mean()
        oracle_risk = error[oracle_order[:k]].mean()
        pred_risks.append(pred_risk)
        oracle_risks.append(oracle_risk)
        key = int(round(float(coverage) * 100.0))
        out[f"risk_coverage_{key}"] = float(pred_risk.item())
        out[f"oracle_risk_coverage_{key}"] = float(oracle_risk.item())
    coverage_t = error.new_tensor([float(x) for x in coverages]).flip(0)
    diff_t = torch.stack(pred_risks).flip(0) - torch.stack(oracle_risks).flip(0)
    full_risk = error.mean().clamp_min(1.0e-12)
    ause = torch.trapz(diff_t / full_risk, coverage_t) / (coverage_t[-1] - coverage_t[0]).clamp_min(1.0e-12)
    out["ause"] = float(ause.clamp_min(0.0).item())
    return out


def _gt_sobel_edge_mask(gt_rgb: torch.Tensor, *, threshold: float) -> torch.Tensor:
    image = gt_rgb.detach().float().permute(2, 0, 1).unsqueeze(0)
    kernel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(1)
    kernel_y = kernel_x.transpose(-1, -2)
    kernel_x = kernel_x.expand(int(image.shape[1]), 1, 3, 3)
    kernel_y = kernel_y.expand(int(image.shape[1]), 1, 3, 3)
    gx = F.conv2d(image, kernel_x, padding=1, groups=int(image.shape[1]))
    gy = F.conv2d(image, kernel_y, padding=1, groups=int(image.shape[1]))
    magnitude = torch.sqrt(gx.square() + gy.square() + 1.0e-12).mean(dim=1).squeeze(0)
    return magnitude >= float(threshold)


def grouped_uncertainty_calibration_metrics(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    variance_maps: Dict[str, torch.Tensor],
    *,
    mask: Optional[torch.Tensor],
    alpha: torch.Tensor,
    dynamic_mask: Optional[torch.Tensor] = None,
    alpha_valid_min: float = 0.01,
    alpha_bins: Sequence[float] = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 1.01),
    edge_threshold: float = 0.05,
) -> Dict[str, float]:
    base_valid = _mask2d(mask, ref=pred_rgb)
    alpha2 = alpha.squeeze(-1) if alpha.dim() == 3 and int(alpha.shape[-1]) == 1 else alpha
    alpha2 = alpha2.detach().to(device=pred_rgb.device, dtype=torch.float32)
    calibration_valid = base_valid & (alpha2 >= float(alpha_valid_min))
    edge = _gt_sobel_edge_mask(gt_rgb, threshold=float(edge_threshold))
    groups: Dict[str, torch.Tensor] = {
        "edge": calibration_valid & edge,
        "non_edge": calibration_valid & ~edge,
    }
    boundaries = [float(x) for x in alpha_bins]
    for idx, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        label = f"alpha_bin_{idx}_{lo:.2f}_{hi:.2f}".replace(".", "p")
        upper = alpha2 <= hi if idx == len(boundaries) - 2 else alpha2 < hi
        groups[label] = calibration_valid & (alpha2 >= lo) & upper
    if dynamic_mask is not None:
        dynamic = _mask2d(dynamic_mask, ref=pred_rgb)
        groups["dynamic"] = calibration_valid & dynamic
        groups["static"] = calibration_valid & ~dynamic

    out: Dict[str, float] = {}
    for source_name, variance in variance_maps.items():
        base_metrics = uncertainty_calibration_metrics(
            pred_rgb,
            gt_rgb,
            variance,
            mask=calibration_valid,
            alpha=None,
        )
        for key, value in base_metrics.items():
            out[f"calibration/{source_name}/{key}"] = float(value)
        for group_name, group_mask in groups.items():
            out[f"calibration/{source_name}/{group_name}/pixel_count"] = float(group_mask.sum().item())
            metrics = uncertainty_calibration_metrics(
                pred_rgb,
                gt_rgb,
                variance,
                mask=group_mask,
                alpha=None,
            )
            for key, value in metrics.items():
                out[f"calibration/{source_name}/{group_name}/{key}"] = float(value)
    return out


__all__ = [
    "DecoupledGaussianNLLComponents",
    "grouped_uncertainty_calibration_metrics",
    "masked_gaussian_rgb_nll",
    "masked_gaussian_rgb_nll_components",
    "masked_uncertainty_variance_stats",
    "masked_uncertainty_photometric_loss",
    "uncertainty_calibration_metrics",
    "uncertainty_v2_loss_weights",
]
