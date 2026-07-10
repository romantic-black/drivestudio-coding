from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


PARENT_DIAGNOSTICS_VERSION = "parent_assignment_topk_v1"


@dataclass
class ParentDiagnosticsResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def build_parent_diagnostics(
    *,
    previous_state: Any,
    next_state: Any,
    topk: int = 16,
) -> ParentDiagnosticsResult:
    """Build a compact parent-level structural attribution summary.

    This is intentionally assignment-based, not per-pixel attribution: child
    Gaussian deltas are aggregated through the existing BigGS parent mappings.
    """

    if next_state is None:
        return ParentDiagnosticsResult(rows=[], summary=_empty_summary())

    biggs_state = getattr(next_state, "biggs_state", None) or getattr(previous_state, "biggs_state", None)
    next_local = getattr(next_state, "local_gs", None)
    prev_local = getattr(previous_state, "local_gs", None) if previous_state is not None else None
    if biggs_state is None or next_local is None:
        return ParentDiagnosticsResult(rows=[], summary=_empty_summary())

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "version": PARENT_DIAGNOSTICS_VERSION,
        "topk": int(topk),
        "branches": {},
    }
    memory_state = getattr(next_state, "parent_temporal", None)
    for branch_name in ("bg", "distant", "rigid"):
        assignment = getattr(biggs_state, branch_name, None)
        branch_next = getattr(next_local, branch_name, None)
        branch_prev = getattr(prev_local, branch_name, None) if prev_local is not None else None
        if assignment is None or branch_next is None:
            continue
        branch_rows, branch_summary = _branch_rows(
            branch=str(branch_name),
            assignment=assignment,
            branch_prev=branch_prev,
            branch_next=branch_next,
            memory_branch=getattr(memory_state, branch_name, None) if memory_state is not None else None,
            topk=int(topk),
        )
        rows.extend(branch_rows)
        summary["branches"][str(branch_name)] = branch_summary

    summary["num_rows"] = int(len(rows))
    if rows:
        summary["max_impact_score"] = _safe_float(max(float(row.get("impact_score", 0.0)) for row in rows))
        summary["max_delta_norm_rms"] = _safe_float(max(float(row.get("delta_norm_rms", 0.0)) for row in rows))
    else:
        summary["max_impact_score"] = 0.0
        summary["max_delta_norm_rms"] = 0.0
    return ParentDiagnosticsResult(rows=rows, summary=summary)


def _branch_rows(
    *,
    branch: str,
    assignment: Any,
    branch_prev: Any,
    branch_next: Any,
    memory_branch: Any,
    topk: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    child_to_parent, child_mass, parent_ids, source_rows = _assignment_tensors(assignment)
    num_parents = int(getattr(assignment, "num_parents", 0) or int(parent_ids.numel()))
    if int(child_to_parent.numel()) == 0 or num_parents <= 0:
        return [], _branch_summary(num_parents=0, num_children=0, rows=[])

    delta_norm_all = _child_delta_norm(branch_prev, branch_next)
    source_rows = source_rows.to(device=delta_norm_all.device, dtype=torch.long)
    source_valid = (source_rows >= 0) & (source_rows < int(delta_norm_all.numel()))
    delta_norm = torch.zeros((int(child_to_parent.numel()),), device=delta_norm_all.device, dtype=torch.float32)
    if bool(source_valid.any().item()):
        delta_norm[source_valid] = delta_norm_all.index_select(0, source_rows[source_valid])
    parent_ids = parent_ids.to(dtype=torch.long, device=child_to_parent.device)
    valid = (child_to_parent >= 0) & (child_to_parent < max(num_parents, 1))
    if not bool(valid.any().item()):
        return [], _branch_summary(num_parents=num_parents, num_children=int(child_to_parent.numel()), rows=[])
    parent_idx = child_to_parent[valid].to(dtype=torch.long)
    mass = child_mass.to(device=parent_idx.device, dtype=torch.float32)[valid].clamp_min(0.0)
    delta = delta_norm.to(device=parent_idx.device, dtype=torch.float32)[valid]
    ones = torch.ones_like(delta, dtype=torch.float32)

    count = torch.zeros((num_parents,), device=parent_idx.device, dtype=torch.float32).scatter_add_(0, parent_idx, ones)
    mass_sum = torch.zeros_like(count).scatter_add_(0, parent_idx, mass)
    delta_sum = torch.zeros_like(count).scatter_add_(0, parent_idx, delta)
    delta_sq_sum = torch.zeros_like(count).scatter_add_(0, parent_idx, delta * delta)
    denom = count.clamp_min(1.0)
    support_mean = mass_sum / denom
    delta_mean = delta_sum / denom
    delta_rms = torch.sqrt(delta_sq_sum / denom + 1.0e-12)
    impact = support_mean * delta_rms
    active = count > 0
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    if int(active_indices.numel()) == 0:
        return [], _branch_summary(num_parents=num_parents, num_children=int(child_to_parent.numel()), rows=[])

    candidate_scores = impact[active_indices]
    k = min(max(int(topk), 1), int(candidate_scores.numel()))
    _, order = torch.topk(candidate_scores, k=k, largest=True)
    selected = active_indices[order].detach().cpu().tolist()
    gdkv_rms = _memory_row_rms(memory_branch, num_parents=num_parents)
    child_to_parent_cpu = child_to_parent.detach().cpu()
    child_mass_cpu = child_mass.detach().float().cpu()
    delta_cpu = delta_norm.detach().float().cpu()
    parent_ids_cpu = parent_ids.detach().cpu()

    rows: list[dict[str, Any]] = []
    for rank, parent_row in enumerate(selected):
        parent_row_i = int(parent_row)
        mask = child_to_parent_cpu == parent_row_i
        support_vals = child_mass_cpu[mask]
        delta_vals = delta_cpu[mask]
        parent_id = int(parent_ids_cpu[parent_row_i].item()) if parent_row_i < int(parent_ids_cpu.numel()) else parent_row_i
        rows.append(
            {
                "version": PARENT_DIAGNOSTICS_VERSION,
                "branch": str(branch),
                "rank": int(rank),
                "parent_row": parent_row_i,
                "parent_id": int(parent_id),
                "child_count": int(count[parent_row_i].detach().cpu().item()),
                "mass_sum": _safe_float(mass_sum[parent_row_i].detach().cpu().item()),
                "support_mean": _safe_float(support_mean[parent_row_i].detach().cpu().item()),
                "support_max": _safe_float(support_vals.max().item() if int(support_vals.numel()) else 0.0),
                "delta_norm_mean": _safe_float(delta_mean[parent_row_i].detach().cpu().item()),
                "delta_norm_rms": _safe_float(delta_rms[parent_row_i].detach().cpu().item()),
                "delta_norm_max": _safe_float(delta_vals.max().item() if int(delta_vals.numel()) else 0.0),
                "parent_event_norm": 0.0,
                "gdkv_ctx_norm": _safe_float(gdkv_rms[parent_row_i].detach().cpu().item()) if gdkv_rms is not None and parent_row_i < int(gdkv_rms.numel()) else 0.0,
                "impact_score": _safe_float(impact[parent_row_i].detach().cpu().item()),
            }
        )
    return rows, _branch_summary(num_parents=num_parents, num_children=int(child_to_parent.numel()), rows=rows)


def _assignment_tensors(assignment: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(assignment, "child_to_active_parent_S"):
        child_to_parent = getattr(assignment, "child_to_active_parent_S").detach().to(dtype=torch.long)
        child_mass = getattr(assignment, "child_mass_S").detach().to(dtype=torch.float32)
        parent_ids = getattr(assignment, "active_parent_global").detach().to(dtype=torch.long)
        source_rows = getattr(assignment, "fine_S").detach().to(dtype=torch.long)
        return child_to_parent, child_mass, parent_ids, source_rows
    child_to_parent = getattr(assignment, "child_to_parent").detach().to(dtype=torch.long)
    child_mass = getattr(assignment, "child_mass").detach().to(dtype=torch.float32)
    num_parents = int(getattr(assignment, "num_parents", 0) or 0)
    parent_object_id = getattr(assignment, "parent_object_id", None)
    if torch.is_tensor(parent_object_id) and int(parent_object_id.numel()) >= num_parents:
        parent_ids = parent_object_id.detach().to(dtype=torch.long)
    else:
        parent_ids = torch.arange(num_parents, device=child_to_parent.device, dtype=torch.long)
    source_rows = torch.arange(int(child_to_parent.numel()), device=child_to_parent.device, dtype=torch.long)
    return child_to_parent, child_mass, parent_ids, source_rows


def _child_delta_norm(branch_prev: Any, branch_next: Any) -> torch.Tensor:
    device = getattr(getattr(branch_next, "means", None), "device", torch.device("cpu"))
    pieces: list[torch.Tensor] = []
    rows = int(getattr(getattr(branch_next, "means", None), "shape", (0,))[0] or 0)
    if rows <= 0:
        return torch.zeros((0,), device=device, dtype=torch.float32)
    if branch_prev is None:
        return torch.zeros((int(rows),), device=device, dtype=torch.float32)
    for name in ("means", "scales_log", "opacity_logit", "sh_dc", "sh_rest", "hidden"):
        prev = getattr(branch_prev, name, None)
        nxt = getattr(branch_next, name, None)
        if not torch.is_tensor(prev) or not torch.is_tensor(nxt):
            continue
        if int(prev.shape[0]) < int(rows) or int(nxt.shape[0]) < int(rows):
            continue
        delta = (nxt[: int(rows)].detach().float() - prev[: int(rows)].detach().float()).reshape(int(rows), -1)
        if int(delta.numel()) == 0:
            continue
        pieces.append(torch.mean(delta * delta, dim=1))
    if not pieces:
        return torch.zeros((int(rows),), device=device, dtype=torch.float32)
    return torch.sqrt(torch.stack(pieces, dim=0).sum(dim=0) + 1.0e-12)


def _memory_row_rms(memory_branch: Any, *, num_parents: int) -> torch.Tensor | None:
    dense = getattr(memory_branch, "dense", None) if memory_branch is not None else None
    kv_state = getattr(dense, "kv_state", None)
    if not torch.is_tensor(kv_state) or int(kv_state.numel()) == 0:
        return None
    rows = min(int(num_parents), int(kv_state.shape[0]))
    if rows <= 0:
        return None
    flat = kv_state[:rows].detach().float().reshape(rows, -1)
    return torch.sqrt(torch.mean(flat * flat, dim=1) + 1.0e-12)


def _branch_summary(*, num_parents: int, num_children: int, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_l = list(rows)
    return {
        "num_parents": int(num_parents),
        "num_children": int(num_children),
        "topk_rows": int(len(rows_l)),
        "max_impact_score": _safe_float(max((float(row.get("impact_score", 0.0)) for row in rows_l), default=0.0)),
        "max_delta_norm_rms": _safe_float(max((float(row.get("delta_norm_rms", 0.0)) for row in rows_l), default=0.0)),
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "version": PARENT_DIAGNOSTICS_VERSION,
        "topk": 0,
        "branches": {},
        "num_rows": 0,
        "max_impact_score": 0.0,
        "max_delta_norm_rms": 0.0,
    }


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


__all__ = ["PARENT_DIAGNOSTICS_VERSION", "ParentDiagnosticsResult", "build_parent_diagnostics"]
