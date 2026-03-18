from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_SSIM_UNAVAILABLE = False
_LPIPS_UNAVAILABLE = False


def compute_ssim_loss_masked(
    pred_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    sky_mask: Optional[torch.Tensor] = None,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    SSIM loss (1 - SSIM)，支持可选的 valid_mask 和 sky_mask。

    注意：SSIM 是局部窗口指标，严格的“只在 mask 区域计算”实现较复杂。
    这里采用一种稳定、可微且简单的做法：对 mask 外像素，将 pred 替换为 gt，
    使得无效区域对 SSIM 的贡献趋近于“完美匹配”（不惩罚、不引入梯度噪声）。
    """
    try:
        from pytorch_msssim import ssim
    except ImportError as e:
        # 训练用 loss，缺依赖应直接失败（fast-fail）。
        raise ImportError(
            "pytorch_msssim is required for SSIM loss. Please install it in the training environment."
        ) from e

    if pred_rgb.dim() != 3 or gt_image.dim() != 3 or pred_rgb.shape[-1] != 3 or gt_image.shape[-1] != 3:
        raise ValueError(
            f"pred_rgb/gt_image must have shape [H, W, 3], got pred={tuple(pred_rgb.shape)} gt={tuple(gt_image.shape)}"
        )
    if pred_rgb.shape[0] != gt_image.shape[0] or pred_rgb.shape[1] != gt_image.shape[1]:
        raise ValueError(
            f"pred_rgb/gt_image spatial shapes must match, got pred={tuple(pred_rgb.shape)} gt={tuple(gt_image.shape)}"
        )

    mask_2d: Optional[torch.Tensor] = None
    if valid_mask is not None:
        mask_2d = valid_mask.to(pred_rgb.device).float()
        if mask_2d.dim() == 3:
            mask_2d = mask_2d.squeeze(-1)
    if sky_mask is not None:
        sky_2d = sky_mask.to(pred_rgb.device).float()
        if sky_2d.dim() == 3:
            sky_2d = sky_2d.squeeze(-1)
        mask_2d = (mask_2d * sky_2d) if mask_2d is not None else sky_2d

    pred_in = pred_rgb
    gt_in = gt_image
    if mask_2d is not None:
        if mask_2d.dim() != 2:
            raise ValueError(f"mask must have shape [H, W] (or [H, W, 1]), got {tuple(mask_2d.shape)}")
        if mask_2d.shape[0] != pred_rgb.shape[0] or mask_2d.shape[1] != pred_rgb.shape[1]:
            raise ValueError(
                f"mask spatial shape must match image, got mask={tuple(mask_2d.shape)} image={tuple(pred_rgb.shape)}"
            )
        m = mask_2d.clamp(0.0, 1.0).unsqueeze(-1)
        pred_in = pred_rgb * m + gt_image * (1.0 - m)

    pred_4d = pred_in.permute(2, 0, 1).unsqueeze(0)
    gt_4d = gt_in.permute(2, 0, 1).unsqueeze(0)
    ssim_val = ssim(pred_4d, gt_4d, data_range=float(data_range), size_average=True)
    return (1.0 - ssim_val)


def compute_l1_loss_masked(
    pred_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    sky_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L1 损失，支持可选的 valid_mask 和 sky_mask。

    - 若 valid_mask 和 sky_mask 均为 None，则在全图上计算 mean L1。
    - 若只提供 valid_mask，则只在 valid_mask>0 的像素上计算。
    - 若只提供 sky_mask，则只在 sky_mask>0 的像素上计算。
    - 若二者都提供，则在二者相乘后的区域上计算。
    """
    diff = torch.abs(pred_rgb - gt_image)

    mask_2d = None
    if valid_mask is not None:
        mask_2d = valid_mask.to(diff.device).float()
        if mask_2d.dim() == 3:
            mask_2d = mask_2d.squeeze(-1)
    if sky_mask is not None:
        sky_2d = sky_mask.to(diff.device).float()
        if sky_2d.dim() == 3:
            sky_2d = sky_2d.squeeze(-1)
        if mask_2d is not None:
            mask_2d = mask_2d * sky_2d
        else:
            mask_2d = sky_2d

    if mask_2d is None:
        # 无任何掩码，退化为全图 L1
        return diff.mean()

    valid_pixels = mask_2d.sum()
    if valid_pixels > 0:
        diff = diff * mask_2d.unsqueeze(-1)
        return diff.sum() / (valid_pixels * diff.shape[-1])
    # 没有有效像素（例如无点投影到该视角），返回 0，避免破坏整体梯度
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
