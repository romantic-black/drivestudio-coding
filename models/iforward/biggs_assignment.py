from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .biggs_state import BigGSBranchAssignment, BigGSRigidActiveAssignment
from .utils import cfg_get


def _empty_assignment(branch: str, ref: torch.Tensor) -> BigGSBranchAssignment:
    return BigGSBranchAssignment(
        branch=str(branch),
        child_to_parent=torch.zeros((0,), dtype=torch.long, device=ref.device),
        child_order=torch.zeros((0,), dtype=torch.long, device=ref.device),
        parent_start=torch.zeros((0,), dtype=torch.long, device=ref.device),
        parent_count=torch.zeros((0,), dtype=torch.long, device=ref.device),
        child_mass=ref.new_zeros((0,)),
        num_children=0,
        num_parents=0,
    )


def _child_mass_from_params(
    *,
    scales_log: torch.Tensor,
    opacity_logit: torch.Tensor,
    mode: str = "tau_area",
    min_mass: float = 1.0e-8,
) -> torch.Tensor:
    n = int(scales_log.shape[0])
    if n == 0:
        return scales_log.new_zeros((0,))
    opacity = torch.sigmoid(opacity_logit.reshape(n, -1)[:, 0]).clamp(0.0, 1.0 - 1.0e-6)
    scales = torch.exp(scales_log)
    area = torch.topk(scales, k=min(2, int(scales.shape[1])), dim=-1).values.prod(dim=-1)
    mode_l = str(mode).lower()
    if mode_l == "uniform":
        mass = torch.ones_like(opacity)
    elif mode_l == "opacity_area":
        mass = opacity * area
    elif mode_l == "tau_area":
        tau = -torch.log1p(-opacity)
        mass = tau * area
    else:
        raise ValueError(f"unsupported BigGS mass_init={mode!r}")
    return mass.clamp_min(float(min_mass)).detach()


def _cfg_branch(assignment_cfg: Any, branch: str) -> Dict[str, Any]:
    return dict(cfg_get(assignment_cfg, branch, {}) or {})


def _merged_branch_cfg(assignment_cfg: Any, branch: str) -> Dict[str, Any]:
    base = {
        "mass_init": cfg_get(assignment_cfg, "mass_init", "tau_area"),
        "min_child_mass": cfg_get(assignment_cfg, "min_child_mass", 1.0e-8),
        "sort_children": cfg_get(assignment_cfg, "sort_children", "morton"),
        "builder": cfg_get(assignment_cfg, "builder", "python_bucket"),
        "radius_voxel_safety_factor": cfg_get(assignment_cfg, "radius_voxel_safety_factor", 2.0),
    }
    base.update(_cfg_branch(assignment_cfg, branch))
    return base


def _sorted_group_children(means_cpu: torch.Tensor, children: Sequence[int], sort_mode: str) -> List[int]:
    if not children:
        return []
    mode = str(sort_mode).lower()
    if mode == "none":
        return [int(x) for x in children]
    pts = means_cpu[torch.tensor(children, dtype=torch.long)]
    if mode == "xyz":
        keys = [
            (float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]), int(children[i]))
            for i in range(len(children))
        ]
    else:
        lo = pts.amin(dim=0)
        span = (pts.amax(dim=0) - lo).clamp_min(1.0e-6)
        q = torch.clamp(((pts - lo) / span * 1023.0).long(), min=0, max=1023)

        def morton(row: torch.Tensor) -> int:
            x, y, z = [int(v) for v in row.tolist()]
            out = 0
            for bit in range(10):
                out |= ((x >> bit) & 1) << (3 * bit)
                out |= ((y >> bit) & 1) << (3 * bit + 1)
                out |= ((z >> bit) & 1) << (3 * bit + 2)
            return int(out)

        keys = [(morton(q[i]), int(children[i])) for i in range(len(children))]
    order = sorted(range(len(children)), key=lambda i: keys[i])
    return [int(children[i]) for i in order]


def _split_group(
    *,
    means_cpu: torch.Tensor,
    children: List[int],
    max_children: int,
    max_radius: Optional[float],
    sort_mode: str,
) -> List[List[int]]:
    pending = [_sorted_group_children(means_cpu, children, sort_mode)]
    out: List[List[int]] = []
    while pending:
        group = pending.pop()
        if len(group) <= 1:
            out.append(group)
            continue
        pts = means_cpu[torch.tensor(group, dtype=torch.long)]
        radius_ok = True
        if max_radius is not None and float(max_radius) > 0.0:
            center = pts.mean(dim=0)
            radius_ok = bool(torch.linalg.norm(pts - center, dim=-1).max().item() <= float(max_radius))
        if len(group) <= int(max_children) and radius_ok:
            out.append(group)
            continue
        extent = pts.amax(dim=0) - pts.amin(dim=0)
        axis = int(torch.argmax(extent).item())
        order = torch.argsort(pts[:, axis], stable=True)
        sorted_group = [int(group[int(i)]) for i in order.tolist()]
        chunk = max(1, int(max_children))
        if len(sorted_group) > chunk:
            for i in range(0, len(sorted_group), chunk):
                pending.append(sorted_group[i : i + chunk])
        else:
            mid = max(1, len(sorted_group) // 2)
            pending.append(sorted_group[:mid])
            pending.append(sorted_group[mid:])
    return [_sorted_group_children(means_cpu, group, sort_mode) for group in out if group]


def _build_biggs_branch_assignment_python(
    *,
    branch: str,
    means: torch.Tensor,
    scales_log: torch.Tensor,
    opacity_logit: torch.Tensor,
    cfg: Any,
    object_id: Optional[torch.Tensor] = None,
) -> BigGSBranchAssignment:
    n = int(means.shape[0])
    if n == 0:
        return _empty_assignment(str(branch), means)
    branch_cfg = dict(cfg or {})
    voxel_size = float(cfg_get(branch_cfg, "voxel_size", 0.5))
    max_children = int(cfg_get(branch_cfg, "max_children_per_parent", cfg_get(branch_cfg, "target_children_per_parent", 8)))
    max_children = max(1, int(max_children))
    max_radius = cfg_get(branch_cfg, "max_parent_radius", None)
    max_radius_f = None if max_radius is None else float(max_radius)
    sort_mode = str(cfg_get(branch_cfg, "sort_children", cfg_get(branch_cfg, "sort", "morton")))
    mass_mode = str(cfg_get(branch_cfg, "mass_init", "tau_area"))
    min_mass = float(cfg_get(branch_cfg, "min_child_mass", 1.0e-8))
    mass = _child_mass_from_params(
        scales_log=scales_log,
        opacity_logit=opacity_logit,
        mode=mass_mode,
        min_mass=float(min_mass),
    )
    means_cpu = means.detach().float().cpu()
    vox = torch.floor(means_cpu / max(float(voxel_size), 1.0e-6)).long()
    object_cpu = (
        object_id.detach().reshape(-1).long().cpu()
        if object_id is not None
        else torch.zeros((n,), dtype=torch.long)
    )
    buckets: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    for i in range(n):
        key = (
            int(object_cpu[i].item()),
            int(vox[i, 0].item()),
            int(vox[i, 1].item()),
            int(vox[i, 2].item()),
        )
        buckets[key].append(int(i))

    groups: List[List[int]] = []
    parent_object: List[int] = []
    for key in sorted(buckets.keys()):
        split = _split_group(
            means_cpu=means_cpu,
            children=buckets[key],
            max_children=max_children,
            max_radius=max_radius_f,
            sort_mode=sort_mode,
        )
        for group in split:
            groups.append(group)
            parent_object.append(int(key[0]))

    num_parents = int(len(groups))
    child_to_parent = torch.empty((n,), dtype=torch.long)
    child_order_list: List[int] = []
    parent_start = torch.empty((num_parents,), dtype=torch.long)
    parent_count = torch.empty((num_parents,), dtype=torch.long)
    for parent_idx, group in enumerate(groups):
        parent_start[parent_idx] = int(len(child_order_list))
        parent_count[parent_idx] = int(len(group))
        for child in group:
            child_to_parent[int(child)] = int(parent_idx)
            child_order_list.append(int(child))
    child_order = torch.tensor(child_order_list, dtype=torch.long)
    parent_object_id = torch.tensor(parent_object, dtype=torch.long)
    return BigGSBranchAssignment(
        branch=str(branch),
        child_to_parent=child_to_parent.to(device=means.device),
        child_order=child_order.to(device=means.device),
        parent_start=parent_start.to(device=means.device),
        parent_count=parent_count.to(device=means.device),
        child_mass=mass.to(device=means.device),
        num_children=n,
        num_parents=num_parents,
        object_id=object_id.detach().clone().to(device=means.device) if object_id is not None else None,
        parent_object_id=parent_object_id.to(device=means.device),
    )


def _build_biggs_branch_assignment_vectorized(
    *,
    branch: str,
    means: torch.Tensor,
    scales_log: torch.Tensor,
    opacity_logit: torch.Tensor,
    cfg: Any,
    object_id: Optional[torch.Tensor] = None,
) -> BigGSBranchAssignment:
    n = int(means.shape[0])
    if n == 0:
        return _empty_assignment(str(branch), means)
    branch_cfg = dict(cfg or {})
    sort_mode = str(cfg_get(branch_cfg, "sort_children", cfg_get(branch_cfg, "sort", "none"))).lower()
    if sort_mode != "none":
        return _build_biggs_branch_assignment_python(
            branch=branch,
            means=means,
            scales_log=scales_log,
            opacity_logit=opacity_logit,
            cfg=cfg,
            object_id=object_id,
        )

    voxel_size = float(cfg_get(branch_cfg, "voxel_size", 0.5))
    voxel_size = max(float(voxel_size), 1.0e-6)
    max_children = int(cfg_get(branch_cfg, "max_children_per_parent", cfg_get(branch_cfg, "target_children_per_parent", 8)))
    max_children = max(1, int(max_children))
    max_radius = cfg_get(branch_cfg, "max_parent_radius", None)
    if max_radius is not None and float(max_radius) > 0.0:
        safety = float(cfg_get(branch_cfg, "radius_voxel_safety_factor", 2.0))
        safety = max(float(safety), 1.0e-6)
        if float(voxel_size) > float(max_radius) / float(safety) + 1.0e-12:
            raise ValueError(
                "BigGS vectorized_sort_segment uses voxel-size radius control instead of per-group "
                f"radius checks; branch={branch!r} voxel_size={voxel_size} must be <= "
                f"max_parent_radius/radius_voxel_safety_factor={float(max_radius) / safety}"
            )

    mass_mode = str(cfg_get(branch_cfg, "mass_init", "tau_area"))
    min_mass = float(cfg_get(branch_cfg, "min_child_mass", 1.0e-8))
    mass = _child_mass_from_params(
        scales_log=scales_log,
        opacity_logit=opacity_logit,
        mode=mass_mode,
        min_mass=float(min_mass),
    ).to(device=means.device)

    means_key = means.detach().float()
    vox = torch.floor(means_key / float(voxel_size)).long()
    q = vox - vox.amin(dim=0, keepdim=True)
    spans = q.amax(dim=0) + 1
    object_values = (
        object_id.detach().reshape(-1).long().to(device=means.device)
        if object_id is not None
        else torch.zeros((n,), dtype=torch.long, device=means.device)
    )
    obj_norm = object_values - object_values.amin()

    max_possible = (
        ((obj_norm.amax() * spans[0] + (spans[0] - 1)) * spans[1] + (spans[1] - 1)) * spans[2]
        + (spans[2] - 1)
    )
    if int(max_possible.detach().cpu().item()) > torch.iinfo(torch.long).max:
        return _build_biggs_branch_assignment_python(
            branch=branch,
            means=means,
            scales_log=scales_log,
            opacity_logit=opacity_logit,
            cfg=cfg,
            object_id=object_id,
        )
    key = ((obj_norm * spans[0] + q[:, 0]) * spans[1] + q[:, 1]) * spans[2] + q[:, 2]
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    key_sorted = key.index_select(0, order)
    new_bucket = torch.ones((n,), dtype=torch.bool, device=means.device)
    if n > 1:
        new_bucket[1:] = key_sorted[1:] != key_sorted[:-1]
    bucket_start = torch.nonzero(new_bucket, as_tuple=False).flatten()
    num_buckets = int(bucket_start.numel())
    bucket_end = torch.cat(
        [
            bucket_start[1:],
            torch.tensor([n], dtype=torch.long, device=means.device),
        ],
        dim=0,
    )
    bucket_count = bucket_end - bucket_start
    parents_per_bucket = torch.div(
        bucket_count + int(max_children) - 1,
        int(max_children),
        rounding_mode="floor",
    )
    num_parents = int(parents_per_bucket.sum().detach().cpu().item())
    bucket_parent_offset = torch.cumsum(parents_per_bucket, dim=0) - parents_per_bucket
    bucket_id = torch.cumsum(new_bucket.to(dtype=torch.long), dim=0) - 1
    row = torch.arange(n, dtype=torch.long, device=means.device)
    offset_in_bucket = row - bucket_start.index_select(0, bucket_id)
    parent_sorted = bucket_parent_offset.index_select(0, bucket_id) + torch.div(
        offset_in_bucket,
        int(max_children),
        rounding_mode="floor",
    )
    child_to_parent = torch.empty((n,), dtype=torch.long, device=means.device)
    child_to_parent.scatter_(0, order, parent_sorted)
    parent_count = torch.bincount(parent_sorted, minlength=int(num_parents)).to(dtype=torch.long)
    parent_start = torch.cumsum(parent_count, dim=0) - parent_count

    parent_bucket_id = torch.repeat_interleave(
        torch.arange(num_buckets, dtype=torch.long, device=means.device),
        parents_per_bucket,
    )
    first_child_per_bucket = order.index_select(0, bucket_start)
    first_object_per_bucket = object_values.index_select(0, first_child_per_bucket)
    parent_object_id = first_object_per_bucket.index_select(0, parent_bucket_id)

    return BigGSBranchAssignment(
        branch=str(branch),
        child_to_parent=child_to_parent,
        child_order=order,
        parent_start=parent_start,
        parent_count=parent_count,
        child_mass=mass,
        num_children=n,
        num_parents=num_parents,
        object_id=object_values.detach().clone() if object_id is not None else None,
        parent_object_id=parent_object_id,
    )


def build_biggs_branch_assignment(
    *,
    branch: str,
    means: torch.Tensor,
    scales_log: torch.Tensor,
    opacity_logit: torch.Tensor,
    cfg: Any,
    object_id: Optional[torch.Tensor] = None,
) -> BigGSBranchAssignment:
    branch_cfg = dict(cfg or {})
    builder = str(cfg_get(branch_cfg, "builder", "python_bucket")).lower()
    if builder in ("vectorized_sort_segment", "vectorized"):
        return _build_biggs_branch_assignment_vectorized(
            branch=branch,
            means=means,
            scales_log=scales_log,
            opacity_logit=opacity_logit,
            cfg=cfg,
            object_id=object_id,
        )
    if builder != "python_bucket":
        raise ValueError(f"unsupported BigGS assignment builder={builder!r}")
    return _build_biggs_branch_assignment_python(
        branch=branch,
        means=means,
        scales_log=scales_log,
        opacity_logit=opacity_logit,
        cfg=cfg,
        object_id=object_id,
    )


def build_biggs_assignments(
    *,
    bg: Any,
    distant: Optional[Any],
    rigid: Optional[Any],
    assignment_cfg: Any,
) -> Tuple[BigGSBranchAssignment, Optional[BigGSBranchAssignment], Optional[BigGSBranchAssignment]]:
    bg_assign = build_biggs_branch_assignment(
        branch="bg",
        means=bg.means,
        scales_log=bg.scales_log,
        opacity_logit=bg.opacity_logit,
        cfg=_merged_branch_cfg(assignment_cfg, "bg"),
    )
    distant_assign = None
    if distant is not None:
        distant_assign = build_biggs_branch_assignment(
            branch="distant",
            means=distant.means,
            scales_log=distant.scales_log,
            opacity_logit=distant.opacity_logit,
            cfg=_merged_branch_cfg(assignment_cfg, "distant"),
        )
    rigid_assign = None
    if rigid is not None:
        object_id = rigid.point_ids[:, 0].long() if hasattr(rigid, "point_ids") else None
        rigid_assign = build_biggs_branch_assignment(
            branch="rigid",
            means=rigid.means,
            scales_log=rigid.scales_log,
            opacity_logit=rigid.opacity_logit,
            cfg=_merged_branch_cfg(assignment_cfg, "rigid"),
            object_id=object_id,
        )
    return bg_assign, distant_assign, rigid_assign


def build_rigid_active_assignment(
    *,
    rigid_assignment: Optional[BigGSBranchAssignment],
    fine_S: torch.Tensor,
    inside_mask_S: torch.Tensor,
) -> Optional[BigGSRigidActiveAssignment]:
    if rigid_assignment is None:
        return None
    n_s = int(fine_S.numel())
    device = fine_S.device
    if n_s == 0:
        empty_l = torch.zeros((0,), dtype=torch.long, device=device)
        empty_b = torch.zeros((0,), dtype=torch.bool, device=device)
        empty_f = rigid_assignment.child_mass.new_zeros((0,)).to(device=device)
        return BigGSRigidActiveAssignment(
            fine_S=fine_S.detach().clone(),
            child_to_active_parent_S=empty_l,
            active_parent_global=empty_l,
            active_parent_count=empty_l,
            active_parent_start=empty_l,
            active_child_order_S=empty_l,
            child_mass_S=empty_f,
            parent_inside_mask=empty_b,
            child_inside_mask_S=empty_b,
        )
    parent_global = rigid_assignment.child_to_parent.to(device=device)[fine_S.long()]
    inside = inside_mask_S.to(device=device, dtype=torch.bool).reshape(-1)
    if int(inside.numel()) != n_s:
        raise ValueError("inside_mask_S length mismatch for BigGS rigid active assignment")
    active_key_to_idx: Dict[Tuple[int, int], int] = {}
    active_parent_global: List[int] = []
    active_inside: List[bool] = []
    child_to_active = torch.empty((n_s,), dtype=torch.long, device=device)
    for row in range(n_s):
        key = (int(parent_global[row].item()), int(bool(inside[row].item())))
        idx = active_key_to_idx.get(key)
        if idx is None:
            idx = len(active_parent_global)
            active_key_to_idx[key] = idx
            active_parent_global.append(int(key[0]))
            active_inside.append(bool(key[1]))
        child_to_active[row] = int(idx)
    m = int(len(active_parent_global))
    starts = torch.zeros((m,), dtype=torch.long, device=device)
    counts = torch.zeros((m,), dtype=torch.long, device=device)
    ordered_rows: List[int] = []
    for parent_idx in range(m):
        starts[parent_idx] = int(len(ordered_rows))
        rows = torch.nonzero(child_to_active == parent_idx, as_tuple=False).reshape(-1)
        counts[parent_idx] = int(rows.numel())
        ordered_rows.extend(int(x) for x in rows.tolist())
    return BigGSRigidActiveAssignment(
        fine_S=fine_S.detach().clone(),
        child_to_active_parent_S=child_to_active,
        active_parent_global=torch.tensor(active_parent_global, dtype=torch.long, device=device),
        active_parent_count=counts,
        active_parent_start=starts,
        active_child_order_S=torch.tensor(ordered_rows, dtype=torch.long, device=device),
        child_mass_S=rigid_assignment.child_mass.to(device=device)[fine_S.long()],
        parent_inside_mask=torch.tensor(active_inside, dtype=torch.bool, device=device),
        child_inside_mask_S=inside,
    )


__all__ = [
    "build_biggs_assignments",
    "build_biggs_branch_assignment",
    "build_rigid_active_assignment",
]
