from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .sky_render_utils import build_sky_regions, get_cfg


def charbonnier(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(x * x + float(eps) * float(eps))


def masked_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if w.dim() == x.dim() - 1:
        denom = w.sum() * float(x.shape[-1])
    elif w.dim() == x.dim() and int(w.shape[-1]) == 1 and int(x.shape[-1]) != 1:
        denom = w.sum() * float(x.shape[-1])
    else:
        denom = w.sum()
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    denom = denom.clamp_min(1e-8)
    return (x * w).sum() / denom


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    diff2 = (pred - gt).pow(2)
    if mask is not None:
        mse = masked_mean(diff2, mask)
    else:
        mse = diff2.mean()
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def skybranch_loss(
    *,
    comp_rgb: torch.Tensor,
    sky_rgb: torch.Tensor,
    sky_alpha: torch.Tensor,
    gt_rgb: torch.Tensor,
    sky_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    eps = float(get_cfg(cfg, "eps", 1.0e-3))
    comp_weight = float(get_cfg(cfg, "comp_weight", 1.0))
    sky_direct_weight = float(get_cfg(cfg, "sky_direct_weight", 0.2))
    sky_alpha_weight = float(get_cfg(cfg, "sky_alpha_weight", 0.05))
    sem_cfg = get_cfg(cfg, "semantic_weight", {}) or {}
    w_sky_core = float(get_cfg(sem_cfg, "sky_core", 1.0))
    w_sky_boundary = float(get_cfg(sem_cfg, "sky_boundary", 0.2))
    erode_kernel = int(get_cfg(cfg, "sky_core_erode_kernel", 5))

    if sky_mask.dim() == 4 and int(sky_mask.shape[-1]) == 1:
        sky_mask_vhw = sky_mask[..., 0]
    else:
        sky_mask_vhw = sky_mask
    if valid_mask.dim() == 4 and int(valid_mask.shape[-1]) == 1:
        valid_vhw = valid_mask[..., 0]
    else:
        valid_vhw = valid_mask
    if sky_alpha.dim() == 4 and int(sky_alpha.shape[-1]) == 1:
        sky_alpha_vhw = sky_alpha[..., 0]
    else:
        sky_alpha_vhw = sky_alpha

    sky_core, sky_boundary, non_sky = build_sky_regions(sky_mask_vhw, erode_kernel=erode_kernel)
    sky_region = sky_mask_vhw.float().clamp(0.0, 1.0)
    w_sem = w_sky_core * sky_core + w_sky_boundary * sky_boundary
    w = valid_vhw.float().clamp(0.0, 1.0) * sky_region * w_sem

    loss_comp = masked_mean(charbonnier(comp_rgb - gt_rgb, eps), w) if float(w.sum().item()) > 0.0 else comp_rgb.sum() * 0.0
    core_w = valid_vhw.float().clamp(0.0, 1.0) * sky_core
    loss_sky_direct = (
        masked_mean(charbonnier(sky_rgb - gt_rgb, eps), core_w)
        if float(core_w.sum().item()) > 0.0
        else sky_rgb.sum() * 0.0
    )
    loss_alpha = (
        masked_mean((sky_alpha_vhw - 1.0).abs(), core_w)
        if float(core_w.sum().item()) > 0.0
        else sky_alpha_vhw.sum() * 0.0
    )
    loss = comp_weight * loss_comp + sky_direct_weight * loss_sky_direct + sky_alpha_weight * loss_alpha
    core_has_pixels = float(core_w.sum().item()) > 0.0
    logs = {
        "loss_comp": loss_comp.detach(),
        "loss_sky_direct": loss_sky_direct.detach(),
        "loss_alpha": loss_alpha.detach(),
        "composite_psnr": compute_psnr(comp_rgb.detach(), gt_rgb.detach()),
        "sky_psnr": compute_psnr(sky_rgb.detach(), gt_rgb.detach(), core_w.detach()) if core_has_pixels else sky_rgb.detach().sum() * 0.0,
        "non_sky_psnr": compute_psnr(comp_rgb.detach(), gt_rgb.detach(), non_sky.detach()),
        "sky_alpha_mean": sky_alpha_vhw.detach().mean(),
        "sky_alpha_core_mean": masked_mean(sky_alpha_vhw.detach(), core_w.detach()) if core_has_pixels else sky_alpha_vhw.detach().mean() * 0.0,
        "sky_loss_valid_pixels": w.detach().sum(),
        "sky_core_valid_pixels": core_w.detach().sum(),
    }
    return loss, logs
