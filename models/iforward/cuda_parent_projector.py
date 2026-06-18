from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.utils.cpp_extension import _get_build_directory, load

from .biggs_parent_projector_diag import (
    mass_mode_to_id,
    project_biggs_parent_diag_reference_tensors,
)

_EXT = None
_EXT_ERROR: Optional[BaseException] = None


def _load_biggs_parent_projector_ext():
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError("BigGS parent projector CUDA extension failed to load") from _EXT_ERROR
    try:
        root = Path(__file__).resolve().parent
        sources = [
            str(root / "csrc" / "biggs_parent_projector_ext.cpp"),
            str(root / "csrc" / "biggs_parent_projector_diag.cu"),
        ]
        verbose = os.getenv("VERBOSE", "0") == "1"
        opt = "-O0" if os.getenv("FAST_COMPILE", "0") == "1" else "-O3"
        _EXT = load(
            name="iforward_biggs_parent_projector",
            sources=sources,
            extra_cflags=[opt],
            extra_cuda_cflags=[opt, "--use_fast_math", "--expt-relaxed-constexpr"],
            build_directory=_get_build_directory("iforward_biggs_parent_projector", verbose=False),
            verbose=verbose,
        )
        return _EXT
    except BaseException as exc:  # pragma: no cover - exercised only on local CUDA toolchain issues.
        _EXT_ERROR = exc
        raise


class _BigGSParentProjectDiagFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        scales_log: torch.Tensor,
        quats: torch.Tensor,
        opacity_logit: torch.Tensor,
        sh_dc: torch.Tensor,
        sh_rest: torch.Tensor,
        child_mass: torch.Tensor,
        child_order: torch.Tensor,
        parent_start: torch.Tensor,
        parent_count: torch.Tensor,
        min_scale: float,
        max_scale: float,
        opacity_cap: float,
        opacity_min: float,
        tau_parent_scale: float,
        eps: float,
        min_mass: float,
        mass_mode_id: int,
    ) -> Tuple[torch.Tensor, ...]:
        ext = _load_biggs_parent_projector_ext()
        outputs = ext.biggs_parent_project_diag_forward(
            means.contiguous(),
            scales_log.contiguous(),
            quats.contiguous(),
            opacity_logit.contiguous(),
            sh_dc.contiguous(),
            sh_rest.contiguous(),
            child_mass.to(device=means.device, dtype=means.dtype).contiguous(),
            child_order.to(device=means.device, dtype=torch.long).contiguous(),
            parent_start.to(device=means.device, dtype=torch.long).contiguous(),
            parent_count.to(device=means.device, dtype=torch.long).contiguous(),
            float(min_scale),
            float(max_scale),
            float(opacity_cap),
            float(opacity_min),
            float(tau_parent_scale),
            float(eps),
            float(min_mass),
            int(mass_mode_id),
        )
        ctx.save_for_backward(
            means,
            scales_log,
            quats,
            opacity_logit,
            sh_dc,
            sh_rest,
            child_mass.to(device=means.device, dtype=means.dtype),
            child_order.to(device=means.device, dtype=torch.long),
            parent_count.to(device=means.device, dtype=torch.long),
        )
        ctx.scalar_args = (
            float(min_scale),
            float(max_scale),
            float(opacity_cap),
            float(opacity_min),
            float(tau_parent_scale),
            float(eps),
            float(min_mass),
            int(mass_mode_id),
        )
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            means,
            scales_log,
            quats,
            opacity_logit,
            sh_dc,
            sh_rest,
            child_mass,
            child_order,
            parent_count,
        ) = ctx.saved_tensors
        (
            min_scale,
            max_scale,
            opacity_cap,
            opacity_min,
            tau_parent_scale,
            eps,
            min_mass,
            mass_mode_id,
        ) = ctx.scalar_args
        with torch.enable_grad():
            means_r = means.detach().requires_grad_(True)
            scales_log_r = scales_log.detach().requires_grad_(True)
            quats_r = quats.detach().requires_grad_(True)
            opacity_logit_r = opacity_logit.detach().requires_grad_(True)
            sh_dc_r = sh_dc.detach().requires_grad_(True)
            sh_rest_r = sh_rest.detach().requires_grad_(True)
            ref_outputs = project_biggs_parent_diag_reference_tensors(
                means=means_r,
                scales_log=scales_log_r,
                quats=quats_r,
                opacity_logit=opacity_logit_r,
                sh_dc=sh_dc_r,
                sh_rest=sh_rest_r,
                child_mass=child_mass.detach(),
                child_order=child_order,
                parent_count=parent_count,
                min_scale=min_scale,
                max_scale=max_scale,
                opacity_cap=opacity_cap,
                opacity_min=opacity_min,
                tau_parent_scale=tau_parent_scale,
                eps=eps,
                min_mass=min_mass,
                mass_mode=mass_mode_id,
            )
            used_outputs = []
            used_grads = []
            for out, grad in zip(ref_outputs, grad_outputs):
                if grad is not None and out.requires_grad:
                    used_outputs.append(out)
                    used_grads.append(grad)
            inputs = (means_r, scales_log_r, quats_r, opacity_logit_r, sh_dc_r, sh_rest_r)
            if used_outputs:
                grads = torch.autograd.grad(
                    outputs=tuple(used_outputs),
                    inputs=inputs,
                    grad_outputs=tuple(used_grads),
                    allow_unused=True,
                )
            else:
                grads = (None,) * len(inputs)
        normalized = []
        for inp, grad in zip((means, scales_log, quats, opacity_logit, sh_dc, sh_rest), grads):
            normalized.append(torch.zeros_like(inp) if grad is None else grad)
        return (
            normalized[0],
            normalized[1],
            normalized[2],
            normalized[3],
            normalized[4],
            normalized[5],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def project_biggs_parent_diag_cuda_tensors(
    *,
    means: torch.Tensor,
    scales_log: torch.Tensor,
    quats: torch.Tensor,
    opacity_logit: torch.Tensor,
    sh_dc: torch.Tensor,
    sh_rest: torch.Tensor,
    child_mass: torch.Tensor,
    child_order: torch.Tensor,
    parent_start: torch.Tensor,
    parent_count: torch.Tensor,
    min_scale: float,
    max_scale: float,
    opacity_cap: float,
    opacity_min: float,
    tau_parent_scale: float,
    eps: float,
    min_mass: float,
    mass_mode: str,
) -> Tuple[torch.Tensor, ...]:
    return _BigGSParentProjectDiagFn.apply(
        means,
        scales_log,
        quats,
        opacity_logit,
        sh_dc,
        sh_rest,
        child_mass,
        child_order,
        parent_start,
        parent_count,
        float(min_scale),
        float(max_scale),
        float(opacity_cap),
        float(opacity_min),
        float(tau_parent_scale),
        float(eps),
        float(min_mass),
        int(mass_mode_to_id(mass_mode)),
    )


def cuda_extension_available() -> bool:
    try:
        _load_biggs_parent_projector_ext()
        return True
    except BaseException:
        return False


__all__ = [
    "cuda_extension_available",
    "project_biggs_parent_diag_cuda_tensors",
]
