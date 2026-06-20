from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
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
        "build_whdd_basis": cfg_get(assignment_cfg, "build_whdd_basis", False),
        "whdd_basis": cfg_get(assignment_cfg, "whdd_basis", {}) or {},
    }
    base.update(_cfg_branch(assignment_cfg, branch))
    return base


def _whdd_basis_cfg(cfg: Any) -> Dict[str, Any]:
    return dict(cfg_get(cfg, "whdd_basis", {}) or {})


def _whdd_basis_enabled(cfg: Any) -> bool:
    return bool(cfg_get(cfg, "build_whdd_basis", False))


def _whdd_basis_dtype(cfg: Any) -> torch.dtype:
    dtype_l = str(cfg_get(_whdd_basis_cfg(cfg), "dtype", "float16")).lower()
    if dtype_l in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_l in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_l in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported BigGS WHDD basis dtype={dtype_l!r}")


def _build_weighted_parent_local_xyz_basis(
    *,
    coords: torch.Tensor,
    parent_id: torch.Tensor,
    child_mass: torch.Tensor,
    parent_count: torch.Tensor,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(coords.shape[0])
    m = int(parent_count.numel())
    if n == 0 or m == 0:
        return (
            coords.new_zeros((n, 3), dtype=_whdd_basis_dtype(cfg)),
            torch.zeros((m, 3), dtype=torch.bool, device=coords.device),
            coords.new_zeros((m,), dtype=torch.float32),
        )
    pid = parent_id.to(device=coords.device, dtype=torch.long).reshape(-1)
    if int(pid.numel()) != n:
        raise ValueError("BigGS WHDD basis parent_id length mismatch")
    xyz = coords.detach().to(device=coords.device, dtype=torch.float32)
    mass = child_mass.detach().to(device=coords.device, dtype=torch.float32).reshape(-1).clamp_min(1.0e-8)
    weight_sum = xyz.new_zeros((m,))
    weight_sum.index_add_(0, pid, mass)
    denom = weight_sum.clamp_min(1.0e-8)
    center_sum = xyz.new_zeros((m, 3))
    center_sum.index_add_(0, pid, xyz * mass[:, None])
    center = center_sum / denom[:, None]
    rel = xyz - center.index_select(0, pid)
    min_std = float(cfg_get(_whdd_basis_cfg(cfg), "min_std", 1.0e-4))
    min_eigenvalue = float(cfg_get(_whdd_basis_cfg(cfg), "min_eigenvalue", float(min_std) ** 2))
    cov_sum = xyz.new_zeros((m, 3, 3))
    cov_sum.index_add_(0, pid, rel[:, :, None] * rel[:, None, :] * mass[:, None, None])
    cov = cov_sum / denom[:, None, None]
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(-1)
    eigvecs = eigvecs.flip(-1)
    valid = eigvals >= float(min_eigenvalue)
    inv_sqrt = torch.where(valid, torch.rsqrt(eigvals.clamp_min(float(min_eigenvalue))), torch.zeros_like(eigvals))
    whitening = eigvecs * inv_sqrt[:, None, :]
    basis = torch.bmm(rel[:, None, :], whitening.index_select(0, pid)).squeeze(1)
    basis = torch.where(valid.index_select(0, pid), basis, torch.zeros_like(basis))
    basis_dtype = _whdd_basis_dtype(cfg)
    basis_q = basis.to(dtype=basis_dtype)
    # Recenter after dtype quantization so fp16/bf16 storage does not reintroduce
    # a static weighted mean in each parent group.
    basis_q_f = basis_q.to(dtype=torch.float32)
    q_mean_sum = xyz.new_zeros((m, 3))
    q_mean_sum.index_add_(0, pid, basis_q_f * mass[:, None])
    q_mean = q_mean_sum / denom[:, None]
    basis_q_f = basis_q_f - q_mean.index_select(0, pid)
    basis_q_f = torch.where(valid.index_select(0, pid), basis_q_f, torch.zeros_like(basis_q_f))
    parent_has_children = parent_count.to(device=coords.device, dtype=torch.long).reshape(-1) > 0
    valid = valid & parent_has_children[:, None]
    return basis_q_f.to(dtype=basis_dtype), valid, weight_sum


def _recenter_stored_child_basis(
    *,
    child_basis: torch.Tensor,
    parent_id: torch.Tensor,
    child_mass: torch.Tensor,
    parent_count: torch.Tensor,
    min_variance: float = 1.0e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(child_basis.shape[0])
    rank = int(child_basis.shape[1]) if child_basis.ndim >= 2 else 0
    m = int(parent_count.numel())
    if n == 0 or m == 0 or rank == 0:
        return (
            child_basis.new_zeros((n, rank)),
            torch.zeros((m, rank), dtype=torch.bool, device=child_basis.device),
            child_basis.new_zeros((m,), dtype=torch.float32),
        )
    pid = parent_id.to(device=child_basis.device, dtype=torch.long).reshape(-1)
    if int(pid.numel()) != n:
        raise ValueError("BigGS WHDD active basis parent_id length mismatch")
    dtype = child_basis.dtype
    phi = child_basis.detach().to(device=child_basis.device, dtype=torch.float32)
    mass = child_mass.detach().to(device=child_basis.device, dtype=torch.float32).reshape(-1).clamp_min(1.0e-8)
    weight_sum = phi.new_zeros((m,))
    weight_sum.index_add_(0, pid, mass)
    denom = weight_sum.clamp_min(1.0e-8)
    mean_sum = phi.new_zeros((m, rank))
    mean_sum.index_add_(0, pid, phi * mass[:, None])
    mean = mean_sum / denom[:, None]
    centered = phi - mean.index_select(0, pid)
    var_sum = phi.new_zeros((m, rank))
    var_sum.index_add_(0, pid, centered.square() * mass[:, None])
    valid = (var_sum / denom[:, None]) >= float(min_variance)
    valid = valid & (parent_count.to(device=child_basis.device, dtype=torch.long).reshape(-1) > 0)[:, None]
    centered = torch.where(valid.index_select(0, pid), centered, torch.zeros_like(centered))
    centered_q = centered.to(dtype=dtype)

    # Recenter once more in the storage dtype to keep fp16/bf16 active subsets
    # mean-preserving without changing the canonical basis scale.
    centered_f = centered_q.to(dtype=torch.float32)
    q_mean_sum = phi.new_zeros((m, rank))
    q_mean_sum.index_add_(0, pid, centered_f * mass[:, None])
    q_mean = q_mean_sum / denom[:, None]
    centered_f = centered_f - q_mean.index_select(0, pid)
    centered_f = torch.where(valid.index_select(0, pid), centered_f, torch.zeros_like(centered_f))
    return centered_f.to(dtype=dtype), valid, weight_sum


def _maybe_attach_whdd_basis(
    assignment: BigGSBranchAssignment,
    *,
    coords: torch.Tensor,
    cfg: Any,
) -> BigGSBranchAssignment:
    if not _whdd_basis_enabled(cfg):
        return assignment
    child_basis, basis_valid, basis_weight_sum = _build_weighted_parent_local_xyz_basis(
        coords=coords,
        parent_id=assignment.child_to_parent,
        child_mass=assignment.child_mass,
        parent_count=assignment.parent_count,
        cfg=cfg,
    )
    return replace(
        assignment,
        child_basis=child_basis,
        basis_valid=basis_valid,
        basis_weight_sum=basis_weight_sum,
        basis_version=1,
    )


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
    assignment = BigGSBranchAssignment(
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
    return _maybe_attach_whdd_basis(assignment, coords=means, cfg=branch_cfg)


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

    assignment = BigGSBranchAssignment(
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
    return _maybe_attach_whdd_basis(assignment, coords=means, cfg=branch_cfg)


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
            child_basis_S=None,
            basis_valid=None,
            basis_weight_sum=None,
        )
    parent_global = rigid_assignment.child_to_parent.to(device=device)[fine_S.long()]
    inside = inside_mask_S.to(device=device, dtype=torch.bool).reshape(-1)
    if int(inside.numel()) != n_s:
        raise ValueError("inside_mask_S length mismatch for BigGS rigid active assignment")
    key = parent_global.long() * 2 + inside.long()
    unique_key, child_to_active, counts = torch.unique(
        key,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    child_to_active = child_to_active.long()
    counts = counts.long()
    m = int(unique_key.numel())
    starts = torch.cumsum(counts, dim=0) - counts
    try:
        ordered_rows = torch.argsort(child_to_active, stable=True)
    except TypeError:
        ordered_rows = torch.argsort(child_to_active)
    active_parent_global = torch.div(unique_key, 2, rounding_mode="floor").long()
    active_inside = (unique_key.remainder(2) > 0)
    child_basis_S = None
    basis_valid = None
    basis_weight_sum = None
    if rigid_assignment.child_basis is not None:
        selected_basis = rigid_assignment.child_basis.to(device=device).index_select(0, fine_S.long())
        child_basis_S, basis_valid, basis_weight_sum = _recenter_stored_child_basis(
            child_basis=selected_basis,
            parent_id=child_to_active,
            child_mass=rigid_assignment.child_mass.to(device=device).index_select(0, fine_S.long()),
            parent_count=counts,
            min_variance=1.0e-8,
        )

    return BigGSRigidActiveAssignment(
        fine_S=fine_S.detach().clone(),
        child_to_active_parent_S=child_to_active,
        active_parent_global=active_parent_global,
        active_parent_count=counts,
        active_parent_start=starts,
        active_child_order_S=ordered_rows.long(),
        child_mass_S=rigid_assignment.child_mass.to(device=device)[fine_S.long()],
        parent_inside_mask=active_inside.to(dtype=torch.bool),
        child_inside_mask_S=inside,
        child_basis_S=child_basis_S,
        basis_valid=basis_valid,
        basis_weight_sum=basis_weight_sum,
    )


__all__ = [
    "build_biggs_assignments",
    "build_biggs_branch_assignment",
    "build_rigid_active_assignment",
]
