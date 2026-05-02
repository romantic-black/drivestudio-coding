from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class Stage6StudentInput:
    render_rgb: torch.Tensor
    prior_map: torch.Tensor
    prior_conf: torch.Tensor
    valid_mask: torch.Tensor
    history_context: Optional[torch.Tensor] = None


def _resize_mask_nearest(valid_mask: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if int(valid_mask.shape[-2]) == int(height) and int(valid_mask.shape[-1]) == int(width):
        return valid_mask
    return F.interpolate(valid_mask.float(), size=(int(height), int(width)), mode="nearest")


def apply_student_valid_mask(
    *,
    render_rgb: torch.Tensor,
    prior_map: torch.Tensor,
    prior_conf: torch.Tensor,
    valid_mask: torch.Tensor,
    history_context: Optional[torch.Tensor] = None,
    append_as_channel: bool = True,
) -> torch.Tensor:
    """
    Build student UNet input with pre-UNet hard mask.
    """

    vm = _resize_mask_nearest(valid_mask, height=int(render_rgb.shape[-2]), width=int(render_rgb.shape[-1]))
    render_m = render_rgb * vm
    prior_m = prior_map * vm
    conf_m = prior_conf * vm
    parts = [render_m, prior_m, conf_m]
    if history_context is not None:
        hm = _resize_mask_nearest(vm, height=int(history_context.shape[-2]), width=int(history_context.shape[-1]))
        parts.append(history_context * hm)
    if append_as_channel:
        parts.append(vm)
    return torch.cat(parts, dim=1)

