from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

import torch

from .biggs_parent_projector import BigGSParentProjection
from .biggs_parent_projector_diag import compute_child_projection_stats
from .biggs_state import BigGSChildContributionCache, BigGSParentBranchRuntime, BigGSParentStats
from .utils import cfg_get


def _top2_area(scales: torch.Tensor) -> torch.Tensor:
    if int(scales.shape[-1]) < 2:
        return scales.reshape(scales.shape[:-1] + (-1,)).prod(dim=-1)
    return torch.topk(scales, k=2, dim=-1).values.prod(dim=-1)


def _identity_quats(ref: torch.Tensor, n: int) -> torch.Tensor:
    quats = ref.new_zeros((int(n), 4))
    if int(n) > 0:
        quats[:, 0] = 1.0
    return quats


def _child_cache_dtype(cfg: Any, ref: torch.Tensor) -> torch.dtype:
    dtype_l = str(cfg_get(cfg, "child_cache_dtype", str(ref.dtype).replace("torch.", ""))).lower()
    if dtype_l in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_l in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_l in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported BigGS parent_state.child_cache_dtype={dtype_l!r}")


def _backend_id(backend: str) -> float:
    backend_l = str(backend).lower()
    if backend_l in {"cuda_exact_diag_forward_only", "cuda_exact_diagonal_forward_only"}:
        return 3.0
    if backend_l in {"cuda_exact_diag", "cuda_exact_diagonal"}:
        return 2.0
    if backend_l in {"torch_exact_diag", "torch_diag_reference", "torch_diagonal", "diagonal"}:
        return 1.0
    if backend_l in {"torch_exact_diag_fallback", "torch_diag_reference_fallback"}:
        return 1.5
    if backend_l in {"incremental", "incremental_sufficient_stats"}:
        return 4.0
    return 0.0


def _compute_child_terms(
    *,
    params: Dict[str, torch.Tensor],
    child_mass: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_mass = float(cfg_get(cfg, "min_child_mass", 1.0e-8))
    mass_mode = str(cfg_get(cfg, "mass_mode", "dynamic_tau_area"))
    mass, tau_area, diag_cov = compute_child_projection_stats(
        scales_log=params["scales_log"],
        quats=params["quats"],
        opacity_logit=params["opacity_logit"],
        child_mass=child_mass,
        min_mass=min_mass,
        mass_mode=mass_mode,
    )
    return mass.detach(), tau_area.detach(), diag_cov.detach()


def _stats_dtype(ref: torch.Tensor) -> torch.dtype:
    return torch.float32 if torch.is_floating_point(ref) else torch.float32


def _segment_sum_ordered(
    values: torch.Tensor,
    *,
    child_order: Optional[torch.Tensor],
    parent_start: Optional[torch.Tensor],
    parent_count: torch.Tensor,
) -> Optional[torch.Tensor]:
    if child_order is None or parent_start is None:
        return None
    n = int(values.shape[0])
    m = int(parent_count.numel())
    order = child_order.to(device=values.device, dtype=torch.long).reshape(-1)
    starts = parent_start.to(device=values.device, dtype=torch.long).reshape(-1)
    counts = parent_count.to(device=values.device, dtype=torch.long).reshape(-1)
    if int(order.numel()) != n or int(starts.numel()) != m or int(counts.numel()) != m:
        return None
    if n == 0:
        return values.new_zeros((m,) + tuple(values.shape[1:]))
    max_count = int(counts.max().detach().cpu().item()) if m > 0 else 0
    if max_count <= 0:
        return values.new_zeros((m,) + tuple(values.shape[1:]))
    offsets = torch.arange(max_count, device=values.device, dtype=torch.long)
    pos = starts[:, None] + offsets[None, :]
    mask = offsets[None, :] < counts[:, None]
    safe_pos = pos.clamp(min=0, max=max(n - 1, 0))
    idx = order.index_select(0, safe_pos.reshape(-1)).reshape(m, max_count)
    gathered = values.index_select(0, idx.reshape(-1)).reshape((m, max_count) + tuple(values.shape[1:]))
    gathered = torch.where(mask.reshape((m, max_count) + (1,) * (values.dim() - 1)), gathered, torch.zeros_like(gathered))
    return gathered.sum(dim=1)


def _sum_by_parent(
    values: torch.Tensor,
    *,
    parent_id: torch.Tensor,
    parent_count: torch.Tensor,
    child_order: Optional[torch.Tensor],
    parent_start: Optional[torch.Tensor],
) -> torch.Tensor:
    ordered = _segment_sum_ordered(
        values,
        child_order=child_order,
        parent_start=parent_start,
        parent_count=parent_count,
    )
    if ordered is not None:
        return ordered
    m = int(parent_count.numel())
    out = values.new_zeros((m,) + tuple(values.shape[1:]))
    out.index_add_(0, parent_id.to(device=values.device, dtype=torch.long).reshape(-1), values)
    return out


def _stats_from_terms(
    *,
    params: Dict[str, torch.Tensor],
    parent_id: torch.Tensor,
    parent_count: torch.Tensor,
    mass: torch.Tensor,
    tau_area: torch.Tensor,
    diag_cov: torch.Tensor,
    child_order: Optional[torch.Tensor] = None,
    parent_start: Optional[torch.Tensor] = None,
) -> BigGSParentStats:
    means = params["means"]
    stat_dtype = _stats_dtype(means)
    pid = parent_id.to(device=means.device, dtype=torch.long).reshape(-1)
    mass = mass.to(device=means.device, dtype=stat_dtype).reshape(-1)
    tau_area = tau_area.to(device=means.device, dtype=stat_dtype).reshape(-1)
    diag_cov = diag_cov.to(device=means.device, dtype=stat_dtype)
    means_stat = means.to(dtype=stat_dtype)
    sh_dc_stat = params["sh_dc"].to(dtype=stat_dtype)
    sh_rest_stat = params["sh_rest"].to(dtype=stat_dtype)
    order = None if child_order is None else child_order.to(device=means.device, dtype=torch.long)
    starts = None if parent_start is None else parent_start.to(device=means.device, dtype=torch.long)
    weight_sum = _sum_by_parent(
        mass,
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    weighted_mean_sum = _sum_by_parent(
        means_stat * mass[:, None],
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    # Use the exact parent weighted mean as a fixed second-moment anchor. This
    # avoids the catastrophic cancellation in E[x^2] - E[x]^2 for small parents
    # embedded in large world coordinates.
    min_mass = 1.0e-8
    second_anchor = weighted_mean_sum / weight_sum.clamp_min(float(min_mass))[:, None]
    valid = parent_count.to(device=means.device, dtype=torch.long).reshape(-1) > 0
    second_anchor = torch.where(valid[:, None], second_anchor, torch.zeros_like(second_anchor))
    child_anchor = second_anchor.index_select(0, pid)
    weighted_second_sum = _sum_by_parent(
        (diag_cov + (means_stat - child_anchor).square()) * mass[:, None],
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    tau_area_sum = _sum_by_parent(
        tau_area,
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    weighted_sh_dc_sum = _sum_by_parent(
        sh_dc_stat * mass[:, None],
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    weighted_sh_rest_sum = _sum_by_parent(
        sh_rest_stat * mass[:, None, None],
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    return BigGSParentStats(
        weight_sum=weight_sum.detach(),
        weighted_mean_sum=weighted_mean_sum.detach(),
        weighted_second_sum=weighted_second_sum.detach(),
        tau_area_sum=tau_area_sum.detach(),
        weighted_sh_dc_sum=weighted_sh_dc_sum.detach(),
        weighted_sh_rest_sum=weighted_sh_rest_sum.detach(),
        parent_count=parent_count.to(device=means.device, dtype=torch.long).detach(),
        second_anchor=second_anchor.detach(),
    )


def _finalize_stats(
    *,
    stats: BigGSParentStats,
    ref: torch.Tensor,
    cfg: Any,
    max_scale: Optional[float] = None,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    m = int(stats.parent_count.numel())
    min_scale = float(cfg_get(cfg, "min_scale", 1.0e-3))
    max_scale_f = float(max_scale if max_scale is not None else cfg_get(cfg, "max_scale", 10.0))
    opacity_cap = float(cfg_get(cfg, "opacity_cap", 0.98))
    opacity_min = float(cfg_get(cfg, "opacity_min", 1.0e-6))
    tau_parent_scale = float(cfg_get(cfg, "tau_parent_scale", 1.0))
    eps = float(cfg_get(cfg, "eps", cfg_get(cfg, "covariance_eps", 1.0e-6)))
    min_mass = float(cfg_get(cfg, "min_child_mass", 1.0e-8))
    compute_dtype = stats.weight_sum.dtype if stats.weight_sum.is_floating_point() else ref.dtype
    if compute_dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32
    mass_safe = stats.weight_sum.to(device=ref.device, dtype=compute_dtype).clamp_min(float(min_mass))
    parent_means = stats.weighted_mean_sum.to(device=ref.device, dtype=compute_dtype) / mass_safe[:, None]
    second_sum = stats.weighted_second_sum.to(device=ref.device, dtype=compute_dtype)
    if stats.second_anchor is None:
        var = second_sum / mass_safe[:, None] - parent_means.square()
    else:
        second_anchor = stats.second_anchor.to(device=ref.device, dtype=compute_dtype)
        rel_mean = parent_means - second_anchor
        var = second_sum / mass_safe[:, None] - rel_mean.square()
    var = torch.clamp(var + float(eps), min=float(min_scale) ** 2, max=float(max_scale_f) ** 2)
    parent_scales = torch.sqrt(var).clamp(min=float(min_scale), max=float(max_scale_f))
    parent_area = _top2_area(parent_scales).clamp_min(float(eps))
    tau_parent = float(tau_parent_scale) * stats.tau_area_sum.to(device=ref.device, dtype=compute_dtype) / (parent_area + float(eps))
    opacity_parent = float(opacity_cap) * (1.0 - torch.exp(-tau_parent))
    opacity_parent = opacity_parent.clamp(float(opacity_min), float(opacity_cap) - float(eps))
    parent_sh_dc = stats.weighted_sh_dc_sum.to(device=ref.device, dtype=compute_dtype) / mass_safe[:, None]
    parent_sh_rest = stats.weighted_sh_rest_sum.to(device=ref.device, dtype=compute_dtype) / mass_safe[:, None, None]
    valid = stats.parent_count.to(device=ref.device, dtype=torch.long).reshape(-1) > 0
    parent_means = torch.where(valid[:, None], parent_means, torch.zeros_like(parent_means))
    parent_sh_dc = torch.where(valid[:, None], parent_sh_dc, torch.zeros_like(parent_sh_dc))
    parent_sh_rest = torch.where(valid[:, None, None], parent_sh_rest, torch.zeros_like(parent_sh_rest))
    params = {
        "means": parent_means.to(dtype=ref.dtype).detach(),
        "scales_log": torch.log(parent_scales.clamp_min(float(min_scale))).to(dtype=ref.dtype).detach(),
        "quats": _identity_quats(ref, m).detach(),
        "opacity_logit": torch.logit(opacity_parent).reshape(m, 1).to(dtype=ref.dtype).detach(),
        "sh_dc": parent_sh_dc.to(dtype=ref.dtype).detach(),
        "sh_rest": parent_sh_rest.to(dtype=ref.dtype).detach(),
    }
    child_mass_mean = stats.weight_sum.to(device=ref.device, dtype=compute_dtype) / stats.parent_count.to(
        device=ref.device,
        dtype=compute_dtype,
    ).clamp_min(1.0)
    return params, child_mass_mean.to(dtype=ref.dtype).detach()


def projection_from_runtime(runtime: BigGSParentBranchRuntime) -> BigGSParentProjection:
    return BigGSParentProjection(
        params={key: value.detach().clone() for key, value in runtime.params.items()},
        child_mass_sum=runtime.stats.weight_sum.detach(),
        child_mass_mean=runtime.child_mass_mean.detach(),
        aux_stats={
            "projector_backend_id": float(runtime.init_backend_id),
            "projector_covariance_mode_id": float(runtime.covariance_mode_id),
            "parent_runtime_incremental": 1.0,
            "parent_runtime_init_backend_id": float(runtime.init_backend_id),
            "parent_runtime_update_backend_id": float(runtime.update_backend_id),
        },
    )


@torch.no_grad()
def _init_parent_branch_runtime_cuda(
    *,
    params: Dict[str, torch.Tensor],
    child_mass: torch.Tensor,
    cfg: Any,
    child_order: torch.Tensor,
    parent_start: torch.Tensor,
    parent_count: torch.Tensor,
    max_scale: Optional[float],
    assignment_signature: str,
) -> BigGSParentBranchRuntime:
    ref = params["means"]
    min_scale = float(cfg_get(cfg, "min_scale", 1.0e-3))
    max_scale_f = float(max_scale if max_scale is not None else cfg_get(cfg, "max_scale", 10.0))
    opacity_cap = float(cfg_get(cfg, "opacity_cap", 0.98))
    opacity_min = float(cfg_get(cfg, "opacity_min", 1.0e-6))
    eps = float(cfg_get(cfg, "eps", cfg_get(cfg, "covariance_eps", 1.0e-6)))
    min_mass = float(cfg_get(cfg, "min_child_mass", 1.0e-8))
    mass_mode = str(cfg_get(cfg, "mass_mode", "dynamic_tau_area"))
    tau_parent_scale = float(cfg_get(cfg, "tau_parent_scale", 1.0))
    from .cuda_parent_projector import project_biggs_parent_diag_cuda_forward_only_with_stats_tensors

    outputs = project_biggs_parent_diag_cuda_forward_only_with_stats_tensors(
        means=ref,
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
    (
        parent_means,
        parent_scales_log,
        parent_quats,
        parent_opacity_logit,
        parent_sh_dc,
        parent_sh_rest,
        mass_sum,
        mass_mean,
        weighted_mean_sum,
        weighted_second_sum,
        tau_area_sum,
        weighted_sh_dc_sum,
        weighted_sh_rest_sum,
        cache_mass,
        cache_tau_area,
        cache_diag_cov,
    ) = outputs
    cache_dtype = _child_cache_dtype(cfg, ref)
    stat_dtype = _stats_dtype(ref)
    stats = BigGSParentStats(
        weight_sum=mass_sum.detach().to(dtype=stat_dtype),
        weighted_mean_sum=weighted_mean_sum.detach().to(dtype=stat_dtype),
        weighted_second_sum=weighted_second_sum.detach().to(dtype=stat_dtype),
        tau_area_sum=tau_area_sum.detach().to(dtype=stat_dtype),
        weighted_sh_dc_sum=weighted_sh_dc_sum.detach().to(dtype=stat_dtype),
        weighted_sh_rest_sum=weighted_sh_rest_sum.detach().to(dtype=stat_dtype),
        parent_count=parent_count.to(device=ref.device, dtype=torch.long).detach(),
        second_anchor=parent_means.detach().to(dtype=stat_dtype),
    )
    parent_params, child_mass_mean = _finalize_stats(stats=stats, ref=ref, cfg=cfg, max_scale=max_scale)
    return BigGSParentBranchRuntime(
        stats=stats,
        params=parent_params,
        child_cache=BigGSChildContributionCache(
            mass=cache_mass.detach().to(dtype=cache_dtype),
            tau_area=cache_tau_area.detach().to(dtype=cache_dtype),
            diag_cov=cache_diag_cov.detach().to(dtype=cache_dtype),
        ),
        child_mass_mean=child_mass_mean,
        assignment_signature=str(assignment_signature),
        init_backend_id=3.0,
        update_backend_id=0.0,
        covariance_mode_id=1.0,
    )


@torch.no_grad()
def init_parent_branch_runtime(
    *,
    params: Dict[str, torch.Tensor],
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
    child_mass: torch.Tensor,
    cfg: Any,
    child_order: Optional[torch.Tensor] = None,
    parent_start: Optional[torch.Tensor] = None,
    max_scale: Optional[float] = None,
    assignment_signature: str = "",
) -> BigGSParentBranchRuntime:
    ref = params["means"]
    cache_dtype = _child_cache_dtype(cfg, ref)
    stat_dtype = _stats_dtype(ref)
    backend_l = str(cfg_get(cfg, "backend", "torch_exact_diag")).lower()
    cuda_backend = backend_l in {
        "cuda_exact_diag",
        "cuda_exact_diagonal",
        "cuda_exact_diag_forward_only",
        "cuda_exact_diagonal_forward_only",
    }
    if int(ref.shape[0]) == 0 or int(parent_count.numel()) == 0:
        b = int(params["sh_rest"].shape[1])
        stats = BigGSParentStats(
            weight_sum=torch.zeros((int(parent_count.numel()),), device=ref.device, dtype=stat_dtype),
            weighted_mean_sum=torch.zeros((int(parent_count.numel()), 3), device=ref.device, dtype=stat_dtype),
            weighted_second_sum=torch.zeros((int(parent_count.numel()), 3), device=ref.device, dtype=stat_dtype),
            tau_area_sum=torch.zeros((int(parent_count.numel()),), device=ref.device, dtype=stat_dtype),
            weighted_sh_dc_sum=torch.zeros((int(parent_count.numel()), 3), device=ref.device, dtype=stat_dtype),
            weighted_sh_rest_sum=torch.zeros((int(parent_count.numel()), b, 3), device=ref.device, dtype=stat_dtype),
            parent_count=parent_count.to(device=ref.device, dtype=torch.long).detach(),
            second_anchor=torch.zeros((int(parent_count.numel()), 3), device=ref.device, dtype=stat_dtype),
        )
        cache = BigGSChildContributionCache(
            mass=ref.new_zeros((int(ref.shape[0]),), dtype=cache_dtype),
            tau_area=ref.new_zeros((int(ref.shape[0]),), dtype=cache_dtype),
            diag_cov=ref.new_zeros((int(ref.shape[0]), 3), dtype=cache_dtype),
        )
        init_backend_id = _backend_id(backend_l if cuda_backend and ref.is_cuda else "torch_exact_diag")
    else:
        if cuda_backend:
            if not ref.is_cuda:
                if not bool(cfg_get(cfg, "allow_cpu_fallback", True)):
                    raise RuntimeError(f"{backend_l} parent runtime init requires CUDA tensors when allow_cpu_fallback=false")
            elif child_order is None or parent_start is None:
                raise ValueError(f"{backend_l} parent runtime init requires grouped child_order and parent_start")
            else:
                try:
                    return _init_parent_branch_runtime_cuda(
                        params=params,
                        child_mass=child_mass,
                        cfg=cfg,
                        child_order=child_order,
                        parent_start=parent_start,
                        parent_count=parent_count,
                        max_scale=max_scale,
                        assignment_signature=assignment_signature,
                    )
                except BaseException:
                    if not bool(cfg_get(cfg, "allow_torch_fallback", cfg_get(cfg, "allow_cpu_fallback", True))):
                        raise
        mass, tau_area, diag_cov = _compute_child_terms(params=params, child_mass=child_mass, cfg=cfg)
        stats = _stats_from_terms(
            params=params,
            parent_id=child_to_parent,
            parent_count=parent_count,
            mass=mass,
            tau_area=tau_area,
            diag_cov=diag_cov,
            child_order=child_order,
            parent_start=parent_start,
        )
        cache = BigGSChildContributionCache(
            mass=mass.detach().to(dtype=cache_dtype),
            tau_area=tau_area.detach().to(dtype=cache_dtype),
            diag_cov=diag_cov.detach().to(dtype=cache_dtype),
        )
        init_backend_id = _backend_id("torch_exact_diag_fallback" if cuda_backend and ref.is_cuda else "torch_exact_diag")
    parent_params, child_mass_mean = _finalize_stats(stats=stats, ref=ref, cfg=cfg, max_scale=max_scale)
    return BigGSParentBranchRuntime(
        stats=stats,
        params=parent_params,
        child_cache=cache,
        child_mass_mean=child_mass_mean,
        assignment_signature=str(assignment_signature),
        init_backend_id=float(init_backend_id),
        update_backend_id=0.0,
        covariance_mode_id=1.0,
    )


@torch.no_grad()
def refresh_parent_branch_runtime_exact(
    *,
    runtime: BigGSParentBranchRuntime,
    params: Dict[str, torch.Tensor],
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
    child_mass: torch.Tensor,
    cfg: Any,
    child_order: Optional[torch.Tensor] = None,
    parent_start: Optional[torch.Tensor] = None,
    max_scale: Optional[float] = None,
) -> BigGSParentBranchRuntime:
    """Rebuild one graph-free runtime branch from exact diagonal statistics."""

    covariance_mode = str(cfg_get(cfg, "covariance_mode", "diagonal")).lower()
    if covariance_mode not in {"diag", "diagonal", "exact_diag", "exact_diagonal"}:
        raise ValueError(
            "exact parent runtime refresh requires diagonal covariance, "
            f"got covariance_mode={covariance_mode!r}"
        )
    refreshed = init_parent_branch_runtime(
        params=params,
        child_to_parent=child_to_parent,
        parent_count=parent_count,
        child_mass=child_mass,
        cfg=cfg,
        child_order=child_order,
        parent_start=parent_start,
        max_scale=max_scale,
        assignment_signature=str(runtime.assignment_signature),
    )
    runtime_tensors = (
        *refreshed.params.values(),
        refreshed.stats.weight_sum,
        refreshed.stats.weighted_mean_sum,
        refreshed.stats.weighted_second_sum,
        refreshed.stats.tau_area_sum,
        refreshed.stats.weighted_sh_dc_sum,
        refreshed.stats.weighted_sh_rest_sum,
        refreshed.child_cache.mass,
        refreshed.child_cache.tau_area,
        refreshed.child_cache.diag_cov,
        refreshed.child_mass_mean,
    )
    if any(value.requires_grad or value.grad_fn is not None for value in runtime_tensors):
        raise RuntimeError("exact parent runtime refresh must return graph-free persistent state")
    return refreshed


@torch.no_grad()
def update_parent_branch_runtime(
    *,
    runtime: BigGSParentBranchRuntime,
    old_params: Dict[str, torch.Tensor],
    new_params: Dict[str, torch.Tensor],
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
    child_mass: torch.Tensor,
    cfg: Any,
    child_order: Optional[torch.Tensor] = None,
    parent_start: Optional[torch.Tensor] = None,
    max_scale: Optional[float] = None,
) -> BigGSParentBranchRuntime:
    ref = new_params["means"]
    cache_dtype = _child_cache_dtype(cfg, ref)
    stat_dtype = runtime.stats.weight_sum.dtype if runtime.stats.weight_sum.is_floating_point() else _stats_dtype(ref)
    if stat_dtype in {torch.float16, torch.bfloat16}:
        stat_dtype = torch.float32
    if int(ref.shape[0]) == 0 or int(parent_count.numel()) == 0:
        return runtime
    pid = child_to_parent.to(device=ref.device, dtype=torch.long).reshape(-1)
    old_mass = runtime.child_cache.mass.to(device=ref.device, dtype=stat_dtype).reshape(-1)
    old_tau_area = runtime.child_cache.tau_area.to(device=ref.device, dtype=stat_dtype).reshape(-1)
    old_diag_cov = runtime.child_cache.diag_cov.to(device=ref.device, dtype=stat_dtype)
    new_mass, new_tau_area, new_diag_cov = _compute_child_terms(params=new_params, child_mass=child_mass, cfg=cfg)
    new_mass_cache = new_mass.to(device=ref.device, dtype=ref.dtype)
    new_tau_area_cache = new_tau_area.to(device=ref.device, dtype=ref.dtype)
    new_diag_cov_cache = new_diag_cov.to(device=ref.device, dtype=ref.dtype)
    new_mass = new_mass.to(device=ref.device, dtype=stat_dtype)
    new_tau_area = new_tau_area.to(device=ref.device, dtype=stat_dtype)
    new_diag_cov = new_diag_cov.to(device=ref.device, dtype=stat_dtype)
    old_means = old_params["means"].to(device=ref.device, dtype=stat_dtype)
    new_means = new_params["means"].to(device=ref.device, dtype=stat_dtype)
    old_sh_dc = old_params["sh_dc"].to(device=ref.device, dtype=stat_dtype)
    new_sh_dc = new_params["sh_dc"].to(device=ref.device, dtype=stat_dtype)
    old_sh_rest = old_params["sh_rest"].to(device=ref.device, dtype=stat_dtype)
    new_sh_rest = new_params["sh_rest"].to(device=ref.device, dtype=stat_dtype)

    m = int(parent_count.numel())
    order = None if child_order is None else child_order.to(device=ref.device, dtype=torch.long)
    starts = None if parent_start is None else parent_start.to(device=ref.device, dtype=torch.long)
    second_anchor = runtime.stats.second_anchor
    if second_anchor is None:
        old_second_term = old_diag_cov + old_means.square()
        new_second_term = new_diag_cov + new_means.square()
    else:
        anchor = second_anchor.to(device=ref.device, dtype=stat_dtype).index_select(0, pid)
        old_second_term = old_diag_cov + (old_means - anchor).square()
        new_second_term = new_diag_cov + (new_means - anchor).square()
    d_weight = _sum_by_parent(
        new_mass - old_mass,
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    d_mean = _sum_by_parent(
        new_means * new_mass[:, None] - old_means * old_mass[:, None],
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    d_second = _sum_by_parent(
        new_second_term * new_mass[:, None] - old_second_term * old_mass[:, None],
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    d_tau_area = _sum_by_parent(
        new_tau_area - old_tau_area,
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    d_sh_dc = _sum_by_parent(
        new_sh_dc * new_mass[:, None] - old_sh_dc * old_mass[:, None],
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    d_sh_rest = _sum_by_parent(
        new_sh_rest * new_mass[:, None, None] - old_sh_rest * old_mass[:, None, None],
        parent_id=pid,
        parent_count=parent_count,
        child_order=order,
        parent_start=starts,
    )
    stats = replace(
        runtime.stats,
        weight_sum=(runtime.stats.weight_sum.to(device=ref.device, dtype=stat_dtype) + d_weight).detach(),
        weighted_mean_sum=(runtime.stats.weighted_mean_sum.to(device=ref.device, dtype=stat_dtype) + d_mean).detach(),
        weighted_second_sum=(runtime.stats.weighted_second_sum.to(device=ref.device, dtype=stat_dtype) + d_second).detach(),
        tau_area_sum=(runtime.stats.tau_area_sum.to(device=ref.device, dtype=stat_dtype) + d_tau_area).detach(),
        weighted_sh_dc_sum=(runtime.stats.weighted_sh_dc_sum.to(device=ref.device, dtype=stat_dtype) + d_sh_dc).detach(),
        weighted_sh_rest_sum=(runtime.stats.weighted_sh_rest_sum.to(device=ref.device, dtype=stat_dtype) + d_sh_rest).detach(),
        parent_count=parent_count.to(device=ref.device, dtype=torch.long).detach(),
        second_anchor=None if second_anchor is None else second_anchor.to(device=ref.device, dtype=stat_dtype).detach(),
    )
    parent_params, child_mass_mean = _finalize_stats(stats=stats, ref=ref, cfg=cfg, max_scale=max_scale)
    return BigGSParentBranchRuntime(
        stats=stats,
        params=parent_params,
        child_cache=BigGSChildContributionCache(
            mass=new_mass_cache.detach().to(dtype=cache_dtype),
            tau_area=new_tau_area_cache.detach().to(dtype=cache_dtype),
            diag_cov=new_diag_cov_cache.detach().to(dtype=cache_dtype),
        ),
        child_mass_mean=child_mass_mean,
        assignment_signature=str(runtime.assignment_signature),
        init_backend_id=float(runtime.init_backend_id),
        update_backend_id=_backend_id("incremental_sufficient_stats"),
        covariance_mode_id=float(runtime.covariance_mode_id),
    )


__all__ = [
    "init_parent_branch_runtime",
    "projection_from_runtime",
    "refresh_parent_branch_runtime_exact",
    "update_parent_branch_runtime",
]
