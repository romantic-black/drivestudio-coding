from __future__ import annotations

from typing import Any

import torch


def output_image_panels(out: Any, *, max_pairs: int = 2) -> list[torch.Tensor]:
    panels: list[torch.Tensor] = []
    pred_rgbs = list(getattr(out, "pred_rgbs", []) or [])
    gt_images = list(getattr(out, "gt_images", []) or [])
    for pred_raw, gt_raw in list(zip(pred_rgbs, gt_images))[: int(max_pairs)]:
        pred = _to_chw(pred_raw)
        gt = _to_chw(gt_raw)
        err = (pred - gt).abs().clamp(0.0, 1.0)
        panels.append(torch.cat([gt, pred, err], dim=-1))
    return panels


def _to_chw(value: Any) -> torch.Tensor:
    image = torch.as_tensor(value).detach().float().cpu()
    while int(image.ndim) > 3:
        image = image[0]
    if int(image.ndim) == 2:
        image = image.unsqueeze(0)
    if int(image.shape[0]) in {1, 3}:
        out = image
    elif int(image.shape[-1]) in {1, 3}:
        out = image.permute(2, 0, 1)
    else:
        raise ValueError(f"expected image with 1 or 3 channels, got shape={tuple(image.shape)}")
    if int(out.shape[0]) == 1:
        out = out.repeat(3, 1, 1)
    return out.clamp(0.0, 1.0)


__all__ = ["output_image_panels"]
