from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Stage6TeacherObserveOutput:
    teacher_input: torch.Tensor
    valid_mask: Optional[torch.Tensor]


def build_teacher_observe_input(*, gt_rgb: torch.Tensor, render_rgb: torch.Tensor, use_gt: bool) -> torch.Tensor:
    if bool(use_gt):
        return torch.cat([gt_rgb, render_rgb], dim=1)
    return render_rgb


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

