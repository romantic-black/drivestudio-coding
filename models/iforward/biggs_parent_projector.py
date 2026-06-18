from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from models.streetforward.math_utils import _normalize_quat, _quat_to_rotmat

from .biggs_parent_projector_diag import project_biggs_parent_diag_reference_tensors
from .biggs_state import BigGSBranchAssignment
from .utils import cfg_get


@dataclass
class BigGSParentProjection:
    params: Dict[str, torch.Tensor]
    child_mass_sum: torch.Tensor
    child_mass_mean: torch.Tensor
    aux_stats: Dict[str, float] = field(default_factory=dict)

    @property
    def num_parents(self) -> int:
        return int(self.params["means"].shape[0])


def _rotmat_to_quat(rot: torch.Tensor) -> torch.Tensor:
    if rot.numel() == 0:
        return rot.new_zeros((0, 4))
    r = rot
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    qw = torch.sqrt(torch.clamp(trace + 1.0, min=1.0e-8)) * 0.5
    qx = torch.sign(r[..., 2, 1] - r[..., 1, 2]) * torch.sqrt(torch.clamp(1.0 + r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2], min=0.0)) * 0.5
    qy = torch.sign(r[..., 0, 2] - r[..., 2, 0]) * torch.sqrt(torch.clamp(1.0 - r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2], min=0.0)) * 0.5
    qz = torch.sign(r[..., 1, 0] - r[..., 0, 1]) * torch.sqrt(torch.clamp(1.0 - r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2], min=0.0)) * 0.5
    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    fallback = rot.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(quat)
    quat = torch.where(torch.isfinite(quat).all(dim=-1, keepdim=True), quat, fallback)
    return _canonicalize_quat(quat)


def _canonicalize_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat(quat)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def _empty_projection(ref: torch.Tensor, *, sh_rest_bases: int) -> BigGSParentProjection:
    params = {
        "means": ref.new_zeros((0, 3)),
        "scales_log": ref.new_zeros((0, 3)),
        "quats": ref.new_zeros((0, 4)),
        "opacity_logit": ref.new_zeros((0, 1)),
        "sh_dc": ref.new_zeros((0, 3)),
        "sh_rest": ref.new_zeros((0, int(sh_rest_bases), 3)),
    }
    return BigGSParentProjection(
        params=params,
        child_mass_sum=ref.new_zeros((0,)),
        child_mass_mean=ref.new_zeros((0,)),
    )


def _params_from_branch(branch: Any) -> Dict[str, torch.Tensor]:
    return {
        "means": branch.means,
        "scales_log": branch.scales_log,
        "quats": branch.quats,
        "opacity_logit": branch.opacity_logit,
        "sh_dc": branch.sh_dc,
        "sh_rest": branch.sh_rest,
    }


def _finite_check_enabled(cfg: Any) -> bool:
    return bool(cfg_get(cfg, "finite_check", True))


def _backend_id(backend: str) -> float:
    backend_l = str(backend).lower()
    if backend_l in {"cuda_exact_diag", "cuda_exact_diagonal"}:
        return 2.0
    if backend_l in {"torch_exact_diag", "torch_diag_reference", "torch_diagonal", "diagonal"}:
        return 1.0
    return 0.0


def _project_params_to_parents_full_eigh(
    *,
    params: Dict[str, torch.Tensor],
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
    child_mass: torch.Tensor,
    cfg: Any,
    max_scale: Optional[float] = None,
) -> BigGSParentProjection:
    means = params["means"]
    n = int(means.shape[0])
    m = int(parent_count.numel())
    sh_rest_bases = int(params["sh_rest"].shape[1])
    if n == 0 or m == 0:
        return _empty_projection(means, sh_rest_bases=sh_rest_bases)
    dtype = means.dtype
    device = means.device
    pid = child_to_parent.to(device=device, dtype=torch.long).reshape(-1)
    if int(pid.numel()) != n:
        raise ValueError("BigGS parent projection child_to_parent length mismatch")
    counts = parent_count.to(device=device, dtype=torch.long).reshape(-1)
    min_scale = float(cfg_get(cfg, "min_scale", 1.0e-3))
    max_scale_f = float(max_scale if max_scale is not None else cfg_get(cfg, "max_scale", 10.0))
    opacity_cap = float(cfg_get(cfg, "opacity_cap", 0.98))
    eps = float(cfg_get(cfg, "covariance_eps", 1.0e-6))

    mass = child_mass.to(device=device, dtype=dtype).reshape(-1).clamp_min(1.0e-8)
    mass_sum = means.new_zeros((m,))
    mass_sum.index_add_(0, pid, mass)
    mass_safe = mass_sum.clamp_min(1.0e-8)
    mass_mean = mass_sum / counts.to(device=device, dtype=dtype).clamp_min(1.0)

    out_means = means.new_zeros((m, 3))
    out_means.index_add_(0, pid, means * mass[:, None])
    out_means = out_means / mass_safe[:, None]

    scales = torch.exp(params["scales_log"])
    rot = _quat_to_rotmat(_normalize_quat(params["quats"]))
    cov_child = rot @ torch.diag_embed(scales.square()) @ rot.transpose(-1, -2)
    centered = means - out_means.index_select(0, pid)
    cov_terms = cov_child + centered.unsqueeze(-1) @ centered.unsqueeze(-2)
    cov = means.new_zeros((m, 3, 3))
    cov.index_add_(0, pid, cov_terms * mass[:, None, None])
    cov = cov / mass_safe[:, None, None]
    cov = cov + torch.eye(3, device=device, dtype=dtype).reshape(1, 3, 3) * float(eps)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=float(min_scale) ** 2, max=float(max_scale_f) ** 2)
    det = torch.linalg.det(eigvecs)
    eigvecs_flipped = eigvecs.clone()
    eigvecs_flipped[:, :, 0] = -eigvecs_flipped[:, :, 0]
    eigvecs = torch.where((det < 0.0).reshape(-1, 1, 1), eigvecs_flipped, eigvecs)
    out_scales = torch.sqrt(eigvals).clamp(min=float(min_scale), max=float(max_scale_f))

    area_parent = torch.topk(out_scales, k=2, dim=-1).values.prod(dim=-1).clamp_min(1.0e-8)
    opacity_child = torch.sigmoid(params["opacity_logit"].reshape(-1)).clamp(0.0, 1.0 - 1.0e-6)
    tau_child = -torch.log1p(-opacity_child)
    child_area = torch.topk(scales, k=2, dim=-1).values.prod(dim=-1)
    tau_area = means.new_zeros((m,))
    tau_area.index_add_(0, pid, tau_child * child_area)
    opacity_parent = (1.0 - torch.exp(-(tau_area / area_parent))).clamp(1.0e-6, float(opacity_cap))

    out_sh_dc = means.new_zeros((m, 3))
    out_sh_dc.index_add_(0, pid, params["sh_dc"] * mass[:, None])
    out_sh_dc = out_sh_dc / mass_safe[:, None]
    out_sh_rest = means.new_zeros((m, sh_rest_bases, 3))
    out_sh_rest.index_add_(0, pid, params["sh_rest"] * mass[:, None, None])
    out_sh_rest = out_sh_rest / mass_safe[:, None, None]

    valid = counts > 0
    parent_params = {
        "means": torch.where(valid[:, None], out_means, means.new_zeros((m, 3))),
        "scales_log": torch.log(out_scales.clamp_min(float(min_scale))),
        "quats": _canonicalize_quat(_rotmat_to_quat(eigvecs)),
        "opacity_logit": torch.logit(opacity_parent).reshape(m, 1),
        "sh_dc": torch.where(valid[:, None], out_sh_dc, means.new_zeros((m, 3))),
        "sh_rest": torch.where(valid[:, None, None], out_sh_rest, means.new_zeros((m, sh_rest_bases, 3))),
    }
    if _finite_check_enabled(cfg):
        for key, value in parent_params.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"BigGS parent projection produced non-finite {key}")
    return BigGSParentProjection(
        params=parent_params,
        child_mass_sum=mass_sum,
        child_mass_mean=mass_mean,
        aux_stats={"projector_backend_id": 0.0},
    )


def _project_params_to_parents_diag(
    *,
    params: Dict[str, torch.Tensor],
    child_to_parent: torch.Tensor,
    child_order: Optional[torch.Tensor],
    parent_start: Optional[torch.Tensor],
    parent_count: torch.Tensor,
    child_mass: torch.Tensor,
    cfg: Any,
    max_scale: Optional[float] = None,
    backend: str = "torch_exact_diag",
) -> BigGSParentProjection:
    means = params["means"]
    n = int(means.shape[0])
    m = int(parent_count.numel())
    sh_rest_bases = int(params["sh_rest"].shape[1])
    if n == 0 or m == 0:
        out = _empty_projection(means, sh_rest_bases=sh_rest_bases)
        out.aux_stats["projector_backend_id"] = _backend_id(backend)
        return out
    min_scale = float(cfg_get(cfg, "min_scale", 1.0e-3))
    max_scale_f = float(max_scale if max_scale is not None else cfg_get(cfg, "max_scale", 10.0))
    opacity_cap = float(cfg_get(cfg, "opacity_cap", 0.98))
    opacity_min = float(cfg_get(cfg, "opacity_min", 1.0e-6))
    eps = float(cfg_get(cfg, "eps", cfg_get(cfg, "covariance_eps", 1.0e-6)))
    min_mass = float(cfg_get(cfg, "min_child_mass", 1.0e-8))
    mass_mode = str(cfg_get(cfg, "mass_mode", "dynamic_tau_area"))
    tau_parent_scale = float(cfg_get(cfg, "tau_parent_scale", 1.0))
    backend_l = str(backend).lower()
    used_backend = backend_l

    outputs = None
    if backend_l in {"cuda_exact_diag", "cuda_exact_diagonal"}:
        if means.is_cuda:
            if child_order is None or parent_start is None:
                raise ValueError("cuda_exact_diag requires grouped child_order and parent_start")
            try:
                from .cuda_parent_projector import project_biggs_parent_diag_cuda_tensors

                outputs = project_biggs_parent_diag_cuda_tensors(
                    means=means,
                    scales_log=params["scales_log"],
                    quats=params["quats"],
                    opacity_logit=params["opacity_logit"],
                    sh_dc=params["sh_dc"],
                    sh_rest=params["sh_rest"],
                    child_mass=child_mass,
                    child_order=child_order,
                    parent_start=parent_start,
                    parent_count=parent_count,
                    min_scale=min_scale,
                    max_scale=max_scale_f,
                    opacity_cap=opacity_cap,
                    opacity_min=opacity_min,
                    tau_parent_scale=tau_parent_scale,
                    eps=eps,
                    min_mass=min_mass,
                    mass_mode=mass_mode,
                )
            except BaseException:
                if not bool(cfg_get(cfg, "allow_torch_fallback", cfg_get(cfg, "allow_cpu_fallback", True))):
                    raise
                used_backend = "torch_exact_diag_fallback"
                outputs = None
        elif not bool(cfg_get(cfg, "allow_cpu_fallback", True)):
            raise RuntimeError("cuda_exact_diag requires CUDA tensors when allow_cpu_fallback=false")

    if outputs is None:
        if backend_l in {"cuda_exact_diag", "cuda_exact_diagonal"} and means.is_cuda:
            used_backend = "torch_exact_diag_fallback"
        outputs = project_biggs_parent_diag_reference_tensors(
            means=means,
            scales_log=params["scales_log"],
            quats=params["quats"],
            opacity_logit=params["opacity_logit"],
            sh_dc=params["sh_dc"],
            sh_rest=params["sh_rest"],
            child_mass=child_mass,
            child_to_parent=child_to_parent,
            parent_count=parent_count,
            min_scale=min_scale,
            max_scale=max_scale_f,
            opacity_cap=opacity_cap,
            opacity_min=opacity_min,
            tau_parent_scale=tau_parent_scale,
            eps=eps,
            min_mass=min_mass,
            mass_mode=mass_mode,
        )

    parent_means, parent_scales_log, parent_quats, parent_opacity_logit, parent_sh_dc, parent_sh_rest, mass_sum, mass_mean = outputs
    parent_params = {
        "means": parent_means,
        "scales_log": parent_scales_log,
        "quats": parent_quats,
        "opacity_logit": parent_opacity_logit,
        "sh_dc": parent_sh_dc,
        "sh_rest": parent_sh_rest,
    }
    if _finite_check_enabled(cfg):
        for key, value in parent_params.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"BigGS diagonal parent projection produced non-finite {key}")
    backend_id = _backend_id(backend_l)
    if used_backend == "torch_exact_diag_fallback":
        backend_id = 1.5
    return BigGSParentProjection(
        params=parent_params,
        child_mass_sum=mass_sum,
        child_mass_mean=mass_mean,
        aux_stats={
            "projector_backend_id": float(backend_id),
            "projector_covariance_mode_id": 1.0,
        },
    )


def _project_params_to_parents(
    *,
    params: Dict[str, torch.Tensor],
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
    child_mass: torch.Tensor,
    cfg: Any,
    max_scale: Optional[float] = None,
    child_order: Optional[torch.Tensor] = None,
    parent_start: Optional[torch.Tensor] = None,
) -> BigGSParentProjection:
    backend = str(cfg_get(cfg, "backend", cfg_get(cfg, "mode", "torch_full_eigh"))).lower()
    if backend in {"cuda_exact_diag", "cuda_exact_diagonal", "torch_exact_diag", "torch_diag_reference", "torch_diagonal", "diagonal"}:
        covariance_mode = str(cfg_get(cfg, "covariance_mode", "diagonal")).lower()
        if covariance_mode not in {"diagonal", "diag", "exact_diagonal"}:
            raise ValueError(f"{backend} requires covariance_mode=diagonal, got {covariance_mode!r}")
        return _project_params_to_parents_diag(
            params=params,
            child_to_parent=child_to_parent,
            child_order=child_order,
            parent_start=parent_start,
            parent_count=parent_count,
            child_mass=child_mass,
            cfg=cfg,
            max_scale=max_scale,
            backend=backend,
        )
    if backend != "torch_full_eigh":
        raise ValueError(f"unsupported BigGS parent_projector.backend={backend!r}")
    return _project_params_to_parents_full_eigh(
        params=params,
        child_to_parent=child_to_parent,
        parent_count=parent_count,
        child_mass=child_mass,
        cfg=cfg,
        max_scale=max_scale,
    )


def project_biggs_parents(
    *,
    branch: Any,
    assignment: Optional[BigGSBranchAssignment],
    cfg: Any,
    max_scale: Optional[float] = None,
) -> BigGSParentProjection:
    params = _params_from_branch(branch)
    means = params["means"]
    n = int(means.shape[0])
    sh_rest_bases = int(params["sh_rest"].shape[1])
    if assignment is None or n == 0 or int(assignment.num_parents) == 0:
        return _empty_projection(means, sh_rest_bases=sh_rest_bases)
    assign = assignment.to(device=means.device)
    return _project_params_to_parents(
        params=params,
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=max_scale,
        child_order=assign.child_order,
        parent_start=assign.parent_start,
    )


def project_biggs_active_rigid_parents(
    *,
    means_world_S: torch.Tensor,
    quats_world_S: torch.Tensor,
    scales_log_S: torch.Tensor,
    opacity_logit_S: torch.Tensor,
    sh_dc_S: torch.Tensor,
    sh_rest_S: torch.Tensor,
    child_to_active_parent_S: torch.Tensor,
    child_mass_S: torch.Tensor,
    active_parent_count: torch.Tensor,
    cfg: Any,
    max_scale: Optional[float] = None,
    active_child_order_S: Optional[torch.Tensor] = None,
    active_parent_start: Optional[torch.Tensor] = None,
) -> BigGSParentProjection:
    params = {
        "means": means_world_S,
        "scales_log": scales_log_S,
        "quats": quats_world_S,
        "opacity_logit": opacity_logit_S,
        "sh_dc": sh_dc_S,
        "sh_rest": sh_rest_S,
    }
    return _project_params_to_parents(
        params=params,
        child_to_parent=child_to_active_parent_S,
        parent_count=active_parent_count,
        child_mass=child_mass_S,
        cfg=cfg,
        max_scale=max_scale,
        child_order=active_child_order_S,
        parent_start=active_parent_start,
    )


def select_parent_projection_rows(proj: BigGSParentProjection, rows: torch.Tensor) -> BigGSParentProjection:
    rows = rows.long().to(device=proj.params["means"].device)
    return BigGSParentProjection(
        params={key: value.index_select(0, rows) for key, value in proj.params.items()},
        child_mass_sum=proj.child_mass_sum.index_select(0, rows),
        child_mass_mean=proj.child_mass_mean.index_select(0, rows),
        aux_stats=dict(proj.aux_stats),
    )


__all__ = [
    "BigGSParentProjection",
    "project_biggs_active_rigid_parents",
    "project_biggs_parents",
    "select_parent_projection_rows",
]
