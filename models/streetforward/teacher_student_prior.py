from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BranchTeacherPrior:
    feat: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor
    last_update_step: torch.Tensor


@dataclass
class TeacherPriorCache:
    bg: BranchTeacherPrior
    distant: BranchTeacherPrior
    rigid: BranchTeacherPrior


def _make_branch_prior(*, num_points: int, feat_dim: int, device: torch.device, dtype: torch.dtype) -> BranchTeacherPrior:
    return BranchTeacherPrior(
        feat=torch.zeros((int(num_points), int(feat_dim)), device=device, dtype=dtype),
        support=torch.zeros((int(num_points),), device=device, dtype=torch.float32),
        valid=torch.zeros((int(num_points),), device=device, dtype=torch.bool),
        last_update_step=torch.full((int(num_points),), -1, device=device, dtype=torch.long),
    )


def create_teacher_prior_cache(
    *,
    num_bg: int,
    num_distant: int,
    num_rigid: int,
    feat_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TeacherPriorCache:
    return TeacherPriorCache(
        bg=_make_branch_prior(num_points=int(num_bg), feat_dim=int(feat_dim), device=device, dtype=dtype),
        distant=_make_branch_prior(num_points=int(num_distant), feat_dim=int(feat_dim), device=device, dtype=dtype),
        rigid=_make_branch_prior(num_points=int(num_rigid), feat_dim=int(feat_dim), device=device, dtype=dtype),
    )


__all__ = [
    "BranchTeacherPrior",
    "TeacherPriorCache",
    "create_teacher_prior_cache",
]
