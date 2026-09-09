from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0.event_encoder import EventPack


@dataclass
class ParentDeltaSummaryBranch:
    summary7: torch.Tensor


@dataclass
class ParentDeltaSummaryPack:
    bg: Optional[ParentDeltaSummaryBranch] = None
    distant: Optional[ParentDeltaSummaryBranch] = None
    rigid: Optional[ParentDeltaSummaryBranch] = None


@dataclass
class ParentAssignmentPack:
    """Graph-free ParentGS assignments used by optimizer write-token aggregation.

    The assignments are topology state, not Parent geometry state.  In
    particular, ``child_mass`` is the static mass stored by assignment
    construction; live dynamic projector mass must not enter the GDKV delta
    summary.
    """

    bg: Optional[object] = None
    distant: Optional[object] = None
    rigid_active: Optional[object] = None


def _branch_delta_summary(delta_branch: Optional[object], rows: int, *, ref: torch.Tensor) -> torch.Tensor:
    if delta_branch is None or int(rows) == 0:
        return ref.new_zeros((int(rows), 7))
    summary7 = getattr(delta_branch, "summary7", None)
    if torch.is_tensor(summary7) and int(summary7.shape[0]) == int(rows) and int(summary7.reshape(int(rows), -1).shape[1]) == 7:
        return summary7.to(device=ref.device, dtype=ref.dtype).reshape(int(rows), 7)
    attrs = [
        getattr(delta_branch, "means", None),
        getattr(delta_branch, "opacity_logit", None),
        getattr(delta_branch, "sh", None),
        getattr(delta_branch, "scales_log", None),
    ]
    norms = []
    for attr in attrs:
        if torch.is_tensor(attr) and int(attr.shape[0]) == int(rows):
            norms.append(attr.reshape(int(rows), -1).norm(dim=-1, keepdim=True))
        else:
            norms.append(ref.new_zeros((int(rows), 1)))
    noop = getattr(delta_branch, "noop", None)
    confidence = getattr(delta_branch, "confidence", None)
    if torch.is_tensor(noop) and int(noop.shape[0]) == int(rows):
        noop_v = noop.reshape(int(rows), -1).mean(dim=-1, keepdim=True)
    else:
        noop_v = ref.new_zeros((int(rows), 1))
    if torch.is_tensor(confidence) and int(confidence.shape[0]) == int(rows):
        conf_v = confidence.reshape(int(rows), -1).mean(dim=-1, keepdim=True)
    else:
        conf_v = ref.new_zeros((int(rows), 1))
    quat = getattr(delta_branch, "quat_axis_angle", None)
    if torch.is_tensor(quat) and int(quat.shape[0]) == int(rows):
        quat_v = quat.reshape(int(rows), -1).norm(dim=-1, keepdim=True)
    else:
        quat_v = ref.new_zeros((int(rows), 1))
    return torch.cat([norms[0], norms[1], norms[2], norms[3], noop_v, conf_v, quat_v], dim=-1)


def _first_tensor_rows(delta_branch: Optional[object]) -> Optional[int]:
    if delta_branch is None:
        return None
    for name in ("summary7", "means", "opacity_logit", "sh", "scales_log", "noop", "confidence", "quat_axis_angle"):
        value = getattr(delta_branch, str(name), None)
        if torch.is_tensor(value):
            return int(value.shape[0])
    return None


def _select_branch_attr(
    delta_branch: object,
    name: str,
    *,
    rows: int,
    ref: torch.Tensor,
    select_idx: Optional[torch.Tensor] = None,
    fail_fast: bool,
    branch: str,
) -> torch.Tensor:
    value = getattr(delta_branch, str(name), None)
    if not torch.is_tensor(value):
        return ref.new_zeros((int(rows), 0))
    value = value.to(device=ref.device, dtype=ref.dtype)
    if int(value.shape[0]) == int(rows):
        return value
    if select_idx is not None:
        select_idx = select_idx.to(device=ref.device, dtype=torch.long)
        if int(select_idx.numel()) == int(rows) and (int(value.shape[0]) == 0 or int(select_idx.max().item()) < int(value.shape[0])):
            return value.index_select(0, select_idx)
    message = (
        f"Stage2_3 parent delta summary {branch}.{name} row mismatch: "
        f"got {tuple(value.shape)}, expected rows={int(rows)}"
    )
    if fail_fast:
        raise ValueError(message)
    return ref.new_zeros((int(rows), 0))


def _norm_column(x: torch.Tensor, *, rows: int, ref: torch.Tensor) -> torch.Tensor:
    if int(x.numel()) == 0:
        return ref.new_zeros((int(rows), 1))
    flat = x.reshape(int(rows), -1)
    if int(flat.shape[1]) == 0:
        return ref.new_zeros((int(rows), 1))
    return flat.norm(dim=-1, keepdim=True)


def _mean_column(x: torch.Tensor, *, rows: int, ref: torch.Tensor) -> torch.Tensor:
    if int(x.numel()) == 0:
        return ref.new_zeros((int(rows), 1))
    flat = x.reshape(int(rows), -1)
    if int(flat.shape[1]) == 0:
        return ref.new_zeros((int(rows), 1))
    return flat.mean(dim=-1, keepdim=True)


def _child_summary7(
    delta_branch: object,
    *,
    rows: int,
    ref: torch.Tensor,
    select_idx: Optional[torch.Tensor],
    fail_fast: bool,
    branch: str,
) -> torch.Tensor:
    direct = getattr(delta_branch, "summary7", None)
    if torch.is_tensor(direct):
        direct = direct.to(device=ref.device, dtype=ref.dtype)
        if int(direct.shape[0]) == int(rows) and int(direct.reshape(int(rows), -1).shape[1]) == 7:
            return direct.reshape(int(rows), 7)
        if select_idx is not None:
            select_idx = select_idx.to(device=ref.device, dtype=torch.long)
            if int(select_idx.numel()) == int(rows) and (int(direct.shape[0]) == 0 or int(select_idx.max().item()) < int(direct.shape[0])):
                selected = direct.index_select(0, select_idx)
                if int(selected.reshape(int(rows), -1).shape[1]) == 7:
                    return selected.reshape(int(rows), 7)
        if fail_fast:
            raise ValueError(f"Stage2_3 parent delta summary {branch}.summary7 row/width mismatch")
    means = _select_branch_attr(delta_branch, "means", rows=rows, ref=ref, select_idx=select_idx, fail_fast=fail_fast, branch=branch)
    opacity = _select_branch_attr(delta_branch, "opacity_logit", rows=rows, ref=ref, select_idx=select_idx, fail_fast=fail_fast, branch=branch)
    sh = _select_branch_attr(delta_branch, "sh", rows=rows, ref=ref, select_idx=select_idx, fail_fast=fail_fast, branch=branch)
    scales = _select_branch_attr(delta_branch, "scales_log", rows=rows, ref=ref, select_idx=select_idx, fail_fast=fail_fast, branch=branch)
    noop = _select_branch_attr(delta_branch, "noop", rows=rows, ref=ref, select_idx=select_idx, fail_fast=fail_fast, branch=branch)
    confidence = _select_branch_attr(delta_branch, "confidence", rows=rows, ref=ref, select_idx=select_idx, fail_fast=fail_fast, branch=branch)
    quat = _select_branch_attr(delta_branch, "quat_axis_angle", rows=rows, ref=ref, select_idx=select_idx, fail_fast=fail_fast, branch=branch)
    return torch.cat(
        [
            _norm_column(means, rows=rows, ref=ref),
            _norm_column(opacity, rows=rows, ref=ref),
            _norm_column(sh, rows=rows, ref=ref),
            _norm_column(scales, rows=rows, ref=ref),
            _mean_column(noop, rows=rows, ref=ref),
            _mean_column(confidence, rows=rows, ref=ref),
            _norm_column(quat, rows=rows, ref=ref),
        ],
        dim=-1,
    )


def _weighted_parent_mean(
    child_summary: torch.Tensor,
    *,
    child_to_parent: torch.Tensor,
    child_mass: Optional[torch.Tensor],
    parent_rows: int,
    ref: torch.Tensor,
    fail_fast: bool,
    branch: str,
) -> torch.Tensor:
    child_to_parent = child_to_parent.to(device=ref.device, dtype=torch.long).reshape(-1)
    if int(child_summary.shape[0]) != int(child_to_parent.numel()):
        message = (
            f"Stage2_3 parent delta summary {branch} child rows mismatch: "
            f"summary={int(child_summary.shape[0])} mapping={int(child_to_parent.numel())}"
        )
        if fail_fast:
            raise ValueError(message)
        return ref.new_zeros((int(parent_rows), 7))
    valid = (child_to_parent >= 0) & (child_to_parent < int(parent_rows))
    if not bool(valid.all()):
        if fail_fast:
            raise ValueError(f"Stage2_3 parent delta summary {branch} mapping contains invalid parent ids")
        child_to_parent = child_to_parent[valid]
        child_summary = child_summary[valid]
    if int(child_to_parent.numel()) == 0:
        return ref.new_zeros((int(parent_rows), 7))
    if torch.is_tensor(child_mass):
        weights = child_mass.to(device=ref.device, dtype=ref.dtype).reshape(-1, 1)
        if int(weights.shape[0]) != int(child_summary.shape[0]):
            if fail_fast:
                raise ValueError(f"Stage2_3 parent delta summary {branch} child_mass row mismatch")
            weights = child_summary.new_ones((int(child_summary.shape[0]), 1))
        else:
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    else:
        weights = child_summary.new_ones((int(child_summary.shape[0]), 1))
    numer = ref.new_zeros((int(parent_rows), 7))
    denom = ref.new_zeros((int(parent_rows), 1))
    numer.index_add_(0, child_to_parent, child_summary * weights)
    denom.index_add_(0, child_to_parent, weights)
    return torch.where(denom > 1.0e-12, numer / denom.clamp_min(1.0e-12), torch.zeros_like(numer))


def _already_parent_summary(
    delta_branch: Optional[object],
    *,
    parent_rows: int,
    ref: torch.Tensor,
) -> Optional[torch.Tensor]:
    if delta_branch is None:
        return None
    rows = _first_tensor_rows(delta_branch)
    if rows is None or int(rows) != int(parent_rows):
        return None
    return _branch_delta_summary(delta_branch, int(parent_rows), ref=ref)


def _summary_stats(branch: str, summary: Optional[torch.Tensor], *, missing: bool) -> Dict[str, float]:
    prefix = f"iforward/parent_optimizer_mamba/delta_summary_{branch}"
    if summary is None or int(summary.numel()) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_nonzero_ratio": 0.0,
            f"{prefix}_missing": float(1.0 if missing else 0.0),
        }
    detached = summary.detach()
    nonzero = detached.abs().sum(dim=-1) > 1.0e-12
    return {
        f"{prefix}_mean": float(detached.float().mean().item()),
        f"{prefix}_nonzero_ratio": float(nonzero.float().mean().item()),
        f"{prefix}_missing": float(1.0 if missing else 0.0),
    }


def _build_branch_parent_delta_summary(
    *,
    branch: str,
    delta_branch: Optional[object],
    spatial: Optional[torch.Tensor],
    assignment: Optional[object],
    fail_fast: bool,
    rigid_active: bool = False,
) -> Tuple[Optional[ParentDeltaSummaryBranch], Dict[str, float]]:
    if spatial is None:
        return None, _summary_stats(branch, None, missing=False)
    parent_rows = int(spatial.shape[0])
    if parent_rows == 0:
        summary = spatial.new_zeros((0, 7))
        return ParentDeltaSummaryBranch(summary7=summary), _summary_stats(branch, summary, missing=False)
    if delta_branch is None:
        if fail_fast:
            raise ValueError(f"Stage2_3 parent delta summary {branch} missing delta branch")
        summary = spatial.new_zeros((parent_rows, 7))
        return ParentDeltaSummaryBranch(summary7=summary), _summary_stats(branch, summary, missing=True)
    if assignment is None:
        parent_ready = _already_parent_summary(delta_branch, parent_rows=parent_rows, ref=spatial)
        if parent_ready is not None:
            return ParentDeltaSummaryBranch(summary7=parent_ready), _summary_stats(branch, parent_ready, missing=False)
        if fail_fast:
            raise ValueError(f"Stage2_3 parent delta summary {branch} requires BigGS parent assignment")
        summary = spatial.new_zeros((parent_rows, 7))
        return ParentDeltaSummaryBranch(summary7=summary), _summary_stats(branch, summary, missing=True)
    if bool(rigid_active):
        child_to_parent = getattr(assignment, "child_to_active_parent_S", None)
        child_mass = getattr(assignment, "child_mass_S", None)
        fine_idx = getattr(assignment, "fine_S", None)
        expected_parent_rows = int(getattr(assignment, "active_parent_count", torch.empty((parent_rows,), device=spatial.device)).numel())
    else:
        child_to_parent = getattr(assignment, "child_to_parent", None)
        child_mass = getattr(assignment, "child_mass", None)
        fine_idx = None
        expected_parent_rows = int(getattr(assignment, "num_parents", parent_rows))
    if not torch.is_tensor(child_to_parent):
        if fail_fast:
            raise ValueError(f"Stage2_3 parent delta summary {branch} assignment missing child_to_parent")
        summary = spatial.new_zeros((parent_rows, 7))
        return ParentDeltaSummaryBranch(summary7=summary), _summary_stats(branch, summary, missing=True)
    if int(expected_parent_rows) != int(parent_rows):
        if fail_fast:
            raise ValueError(
                f"Stage2_3 parent delta summary {branch} parent rows mismatch: "
                f"event={parent_rows} assignment={expected_parent_rows}"
            )
        summary = spatial.new_zeros((parent_rows, 7))
        return ParentDeltaSummaryBranch(summary7=summary), _summary_stats(branch, summary, missing=True)
    child_rows = int(child_to_parent.numel())
    child_summary = _child_summary7(
        delta_branch,
        rows=child_rows,
        ref=spatial,
        select_idx=fine_idx if torch.is_tensor(fine_idx) else None,
        fail_fast=fail_fast,
        branch=branch,
    )
    summary = _weighted_parent_mean(
        child_summary,
        child_to_parent=child_to_parent,
        child_mass=child_mass if torch.is_tensor(child_mass) else None,
        parent_rows=parent_rows,
        ref=spatial,
        fail_fast=fail_fast,
        branch=branch,
    )
    return ParentDeltaSummaryBranch(summary7=summary), _summary_stats(branch, summary, missing=False)


def build_parent_delta_summary(
    delta: Optional[object],
    spatial_event: EventPack,
    *,
    assignments: Optional[ParentAssignmentPack] = None,
    runtime: Optional[object] = None,
    fail_fast: bool = True,
) -> Tuple[ParentDeltaSummaryPack, Dict[str, float]]:
    """Aggregate child deltas using assignments without requiring Parent runtime.

    ``assignments=`` is the Stage 3.4 path. ``runtime=`` is retained solely for
    legacy Stage 3.0/3.1/3.3 callers. Supplying both would make the topology
    source ambiguous and is therefore rejected.
    """

    if assignments is not None and runtime is not None:
        raise ValueError("build_parent_delta_summary accepts either assignments or runtime, not both")

    if assignments is not None:
        bg_assignment = getattr(assignments, "bg", None)
        distant_assignment = getattr(assignments, "distant", None)
        rigid_assignment = getattr(assignments, "rigid_active", None)
    else:
        bg_assignment = None if runtime is None else getattr(runtime, "bg_assignment", None)
        distant_assignment = None if runtime is None else getattr(runtime, "distant_assignment", None)
        rigid_assignment = None if runtime is None else getattr(runtime, "rigid_active_assignment", None)

    aux: Dict[str, float] = {}
    bg, bg_aux = _build_branch_parent_delta_summary(
        branch="bg",
        delta_branch=None if delta is None else getattr(delta, "bg", None),
        spatial=spatial_event.event_bg,
        assignment=bg_assignment,
        fail_fast=bool(fail_fast),
        rigid_active=False,
    )
    aux.update(bg_aux)
    distant, distant_aux = _build_branch_parent_delta_summary(
        branch="distant",
        delta_branch=None if delta is None else getattr(delta, "distant", None),
        spatial=spatial_event.event_distant,
        assignment=distant_assignment,
        fail_fast=bool(fail_fast),
        rigid_active=False,
    )
    aux.update(distant_aux)
    rigid, rigid_aux = _build_branch_parent_delta_summary(
        branch="rigid",
        delta_branch=None if delta is None else getattr(delta, "rigid", None),
        spatial=spatial_event.event_rigid,
        assignment=rigid_assignment,
        fail_fast=bool(fail_fast),
        rigid_active=True,
    )
    aux.update(rigid_aux)
    return ParentDeltaSummaryPack(bg=bg, distant=distant, rigid=rigid), aux


class OptimizerWriteTokenBuilder(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        visit_dim: int,
        token_dim: int,
        delta_summary_dim: int = 7,
        hidden_dim: int = 96,
        include_spatial_event: bool = True,
        include_parent_event: bool = True,
        include_delta_summary: bool = True,
        include_visit_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.visit_dim = int(visit_dim)
        self.delta_summary_dim = int(delta_summary_dim)
        self.token_dim = int(token_dim)
        self.include_spatial_event = bool(include_spatial_event)
        self.include_parent_event = bool(include_parent_event)
        self.include_delta_summary = bool(include_delta_summary)
        self.include_visit_embedding = bool(include_visit_embedding)
        in_dim = int(event_dim) * 2 + int(visit_dim) + int(delta_summary_dim) + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(token_dim)),
        )

    @staticmethod
    def _support_valid(support: Optional[torch.Tensor], valid: Optional[torch.Tensor], rows: int, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if support is None:
            support_v = ref.new_zeros((int(rows), 1))
        else:
            support_v = support.to(device=ref.device, dtype=ref.dtype).reshape(int(rows), -1).mean(dim=-1, keepdim=True)
        if valid is None:
            valid_v = ref.new_ones((int(rows), 1))
        else:
            valid_v = valid.to(device=ref.device, dtype=torch.bool).reshape(int(rows), -1).any(dim=-1, keepdim=True).to(dtype=ref.dtype)
        return support_v, valid_v

    def branch(
        self,
        *,
        spatial: Optional[torch.Tensor],
        fused: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        visit: Optional[torch.Tensor],
        delta_branch: Optional[object],
    ) -> Optional[torch.Tensor]:
        if spatial is None:
            return None
        rows = int(spatial.shape[0])
        spatial_v = spatial if self.include_spatial_event else torch.zeros_like(spatial)
        fused_v = fused if fused is not None else spatial
        if not self.include_parent_event:
            fused_v = torch.zeros_like(spatial)
        if visit is None:
            visit_v = spatial.new_zeros((rows, self.visit_dim))
        elif int(visit.shape[0]) == 1:
            visit_v = visit.to(device=spatial.device, dtype=spatial.dtype).expand(rows, -1)
        else:
            visit_v = visit.to(device=spatial.device, dtype=spatial.dtype)
        if not self.include_visit_embedding:
            visit_v = torch.zeros_like(visit_v)
        support_v, valid_v = self._support_valid(support, valid, rows, spatial)
        delta_v = _branch_delta_summary(delta_branch, rows, ref=spatial)
        if not self.include_delta_summary:
            delta_v = torch.zeros_like(delta_v)
        return self.net(torch.cat([spatial_v, fused_v, visit_v, delta_v, support_v, valid_v], dim=-1))

    def forward(
        self,
        *,
        spatial_event: EventPack,
        fused_event: EventPack,
        visit_bg: Optional[torch.Tensor],
        visit_distant: Optional[torch.Tensor],
        visit_rigid: Optional[torch.Tensor],
        delta: Optional[object] = None,
    ) -> EventPack:
        return EventPack(
            event_bg=self.branch(
                spatial=spatial_event.event_bg,
                fused=fused_event.event_bg,
                support=spatial_event.support_bg,
                valid=spatial_event.valid_bg,
                visit=visit_bg,
                delta_branch=None if delta is None else getattr(delta, "bg", None),
            ),
            event_distant=self.branch(
                spatial=spatial_event.event_distant,
                fused=fused_event.event_distant,
                support=spatial_event.support_distant,
                valid=spatial_event.valid_distant,
                visit=visit_distant,
                delta_branch=None if delta is None else getattr(delta, "distant", None),
            ),
            event_rigid=self.branch(
                spatial=spatial_event.event_rigid,
                fused=fused_event.event_rigid,
                support=spatial_event.support_rigid,
                valid=spatial_event.valid_rigid,
                visit=visit_rigid,
                delta_branch=None if delta is None else getattr(delta, "rigid", None),
            ),
            support_bg=spatial_event.support_bg,
            support_distant=spatial_event.support_distant,
            support_rigid=spatial_event.support_rigid,
            valid_bg=spatial_event.valid_bg,
            valid_distant=spatial_event.valid_distant,
            valid_rigid=spatial_event.valid_rigid,
            view_code_bg=spatial_event.view_code_bg,
            obs_code_bg=None,
            obs_code_distant=None,
            obs_code_rigid=None,
            acc_w_bg=spatial_event.acc_w_bg,
            route=spatial_event.route,
            branch_slices=dict(spatial_event.branch_slices or {}),
            aux=dict(spatial_event.aux or {}),
        )


__all__ = [
    "OptimizerWriteTokenBuilder",
    "ParentAssignmentPack",
    "ParentDeltaSummaryBranch",
    "ParentDeltaSummaryPack",
    "build_parent_delta_summary",
]
