from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.streetforward.teacher_student_prior import TeacherPriorCache


class TeacherPriorAdapter(nn.Module):
    """Residual 1x1 adapter for live teacher prior."""

    def __init__(self, dim: int = 32, hidden_channels: Optional[int] = None):
        super().__init__()
        h = int(hidden_channels or dim)
        self.net = nn.Sequential(
            nn.Conv2d(int(dim), h, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h, int(dim), kernel_size=1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale.to(device=x.device, dtype=x.dtype) * self.net(x)


@dataclass
class Stage6BridgeStats:
    live_enabled: float = 0.0
    live_prior_conf_mean: float = 0.0
    live_prior_conf_nonzero_ratio: float = 0.0
    cache_prior_conf_mean: float = 0.0
    cache_prior_conf_nonzero_ratio: float = 0.0
    cache_fallback_ratio: float = 0.0


def update_teacher_prior_cache_detached(
    *,
    cache: TeacherPriorCache,
    feat_bg: torch.Tensor,
    acc_bg: torch.Tensor,
    feat_distant: Optional[torch.Tensor],
    acc_distant: Optional[torch.Tensor],
    feat_rigid: Optional[torch.Tensor],
    acc_rigid: Optional[torch.Tensor],
    rigid_idx: torch.Tensor,
    global_step: int,
    support_min_bg: float,
    support_min_distant: float,
    support_min_rigid: float,
) -> None:
    """Persistent cache writer. Always detach to prevent cross-step graph retention."""

    if int(cache.bg.feat.shape[-1]) != int(feat_bg.shape[-1]):
        raise ValueError("teacher prior bg feat dim mismatch.")
    bg_valid = acc_bg > float(support_min_bg)
    cache.bg.feat[bg_valid] = feat_bg[bg_valid].detach().float()
    cache.bg.support[bg_valid] = acc_bg[bg_valid].detach().float()
    cache.bg.valid[bg_valid] = True
    cache.bg.last_update_step[bg_valid] = int(global_step)

    if feat_distant is not None and acc_distant is not None and int(feat_distant.shape[0]) > 0:
        if int(cache.distant.feat.shape[-1]) != int(feat_distant.shape[-1]):
            raise ValueError("teacher prior distant feat dim mismatch.")
        distant_valid = acc_distant > float(support_min_distant)
        cache.distant.feat[distant_valid] = feat_distant[distant_valid].detach().float()
        cache.distant.support[distant_valid] = acc_distant[distant_valid].detach().float()
        cache.distant.valid[distant_valid] = True
        cache.distant.last_update_step[distant_valid] = int(global_step)

    if feat_rigid is not None:
        if rigid_idx is None:
            raise ValueError("rigid_idx is required when updating rigid teacher prior.")
        if acc_rigid is None:
            raise ValueError("acc_rigid is required when updating rigid teacher prior.")
        if int(rigid_idx.numel()) != int(feat_rigid.shape[0]):
            raise ValueError("rigid_idx length must match feat_rigid rows.")
        if int(cache.rigid.feat.shape[-1]) != int(feat_rigid.shape[-1]):
            raise ValueError("teacher prior rigid feat dim mismatch.")
    if feat_rigid is not None and acc_rigid is not None and int(feat_rigid.shape[0]) > 0:
        rigid_valid_s = acc_rigid > float(support_min_rigid)
        rigid_local_idx = rigid_idx[rigid_valid_s]
        cache.rigid.feat[rigid_local_idx] = feat_rigid[rigid_valid_s].detach().float()
        cache.rigid.support[rigid_local_idx] = acc_rigid[rigid_valid_s].detach().float()
        cache.rigid.valid[rigid_local_idx] = True
        cache.rigid.last_update_step[rigid_local_idx] = int(global_step)
