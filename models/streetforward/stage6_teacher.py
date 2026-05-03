from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Stage6TeacherObserveOutput:
    teacher_input: torch.Tensor
    valid_mask: Optional[torch.Tensor]


def build_teacher_observe_input(*, gt_rgb: torch.Tensor, render_rgb: torch.Tensor, use_gt: bool) -> torch.Tensor:
    if not bool(use_gt):
        return render_rgb
    if gt_rgb.dim() != 4 or render_rgb.dim() != 4:
        raise ValueError("teacher input expects 4D tensors.")
    if int(gt_rgb.shape[1]) == 3 and int(render_rgb.shape[1]) == 3:
        return torch.cat([gt_rgb, render_rgb], dim=1)
    if int(gt_rgb.shape[-1]) == 3 and int(render_rgb.shape[-1]) == 3:
        return torch.cat([gt_rgb, render_rgb], dim=-1)
    raise ValueError(
        f"cannot infer teacher input layout: gt={tuple(gt_rgb.shape)}, render={tuple(render_rgb.shape)}"
    )


def build_student_valid_mask(
    *,
    source_pair_valid_mask: torch.Tensor,
    camera_valid_mask: Optional[torch.Tensor] = None,
    not_sky_mask: Optional[torch.Tensor] = None,
    not_egocar_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = source_pair_valid_mask.bool()
    if camera_valid_mask is not None:
        mask = mask & camera_valid_mask.bool()
    if not_sky_mask is not None:
        mask = mask & not_sky_mask.bool()
    if not_egocar_mask is not None:
        mask = mask & not_egocar_mask.bool()
    return mask.float()
