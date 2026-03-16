from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_SSIM_UNAVAILABLE = False
_LPIPS_UNAVAILABLE = False


def compute_l1_loss_masked(
    pred_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    valid_mask: torch.Tensor,
    sky_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L1 loss only on pixels where valid_mask (and optionally sky_mask) is non-zero.
    Same logic as ProxyRenderingMixin.compute_loss with required valid_mask.
    """
    diff = torch.abs(pred_rgb - gt_image)
    mask_2d = valid_mask.to(diff.device).float()
    if mask_2d.dim() == 3:
        mask_2d = mask_2d.squeeze(-1)
    if sky_mask is not None:
        sky_2d = sky_mask.to(diff.device).float()
        if sky_2d.dim() == 3:
            sky_2d = sky_2d.squeeze(-1)
        mask_2d = mask_2d * sky_2d
    valid_pixels = mask_2d.sum()
    if valid_pixels > 0:
        diff = diff * mask_2d.unsqueeze(-1)
        return diff.sum() / (valid_pixels * diff.shape[-1])
    return diff.sum() * 0.0


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    计算 PSNR（峰值信噪比）。
    """
    mse = torch.mean((pred - gt) ** 2)
    mse_val = float(mse.item())
    if mse_val <= 0:
        return float("inf")
    psnr = -10 * torch.log10(torch.tensor(mse_val, device=pred.device))
    return float(psnr.item())


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    计算 SSIM（结构相似性指数），若缺少依赖返回 NaN。
    """
    global _SSIM_UNAVAILABLE
    try:
        from pytorch_msssim import ssim
    except ImportError:
        if not _SSIM_UNAVAILABLE:
            logging.warning("pytorch_msssim not installed; returning NaN for SSIM")
            _SSIM_UNAVAILABLE = True
        return float("nan")

    pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
    gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
    return float(ssim(pred_4d, gt_4d, data_range=1.0).item())


def compute_lpips(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lpips_model=None,
    device: Optional[torch.device] = None,
) -> Tuple[float, Optional[object]]:
    """
    计算 LPIPS，返回数值与可复用的 lpips_model。
    """
    global _LPIPS_UNAVAILABLE
    try:
        from lpips import LPIPS
    except ImportError:
        if not _LPIPS_UNAVAILABLE:
            logging.warning("lpips not installed; returning NaN for LPIPS")
            _LPIPS_UNAVAILABLE = True
        return float("nan"), lpips_model

    if lpips_model is None:
        lpips_model = LPIPS(net="alex").to(device or pred.device)

    pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
    gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
    value = lpips_model(pred_4d, gt_4d)
    return float(value.item()), lpips_model


def evaluate_test_views(
    render_fn: Callable[[object, int, int], Tuple[torch.Tensor, torch.Tensor]],
    test_views: List,
    test_images: List[torch.Tensor],
    device: torch.device,
    lpips_model=None,
) -> Tuple[Optional[Dict[str, float]], Optional[object]]:
    """
    使用给定 render_fn 评估一组测试视角，返回指标和可能更新的 lpips_model。
    """
    if test_views is None or len(test_views) == 0:
        return None, lpips_model

    psnr_list: List[float] = []
    ssim_list: List[float] = []
    lpips_list: List[float] = []

    for view, gt_img in zip(test_views, test_images):
        height, width = gt_img.shape[0], gt_img.shape[1]
        rgb_pred, _ = render_fn(view, height, width)
        rgb_gt = gt_img.to(device)

        psnr_list.append(compute_psnr(rgb_pred, rgb_gt))
        ssim_list.append(compute_ssim(rgb_pred, rgb_gt))
        lpips_val, lpips_model = compute_lpips(rgb_pred, rgb_gt, lpips_model=lpips_model, device=device)
        lpips_list.append(lpips_val)

    if len(psnr_list) == 0:
        return None, lpips_model

    metrics = {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
        "num_test_views": len(psnr_list),
    }
    return metrics, lpips_model
