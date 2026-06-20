from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import _get_build_directory, load

_EXT = None
_EXT_ERROR: Optional[BaseException] = None


def _load_grld_decode_ext():
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError("GRLD decode CUDA extension failed to load") from _EXT_ERROR
    try:
        root = Path(__file__).resolve().parent
        sources = [
            str(root / "csrc" / "grld_decode_ext.cpp"),
            str(root / "csrc" / "grld_decode.cu"),
        ]
        verbose = os.getenv("VERBOSE", "0") == "1"
        opt = "-O0" if os.getenv("FAST_COMPILE", "0") == "1" else "-O3"
        _EXT = load(
            name="iforward_grld_decode",
            sources=sources,
            extra_cflags=[opt],
            extra_cuda_cflags=[opt, "--use_fast_math", "--expt-relaxed-constexpr"],
            build_directory=_get_build_directory("iforward_grld_decode", verbose=False),
            verbose=verbose,
        )
        return _EXT
    except BaseException as exc:  # pragma: no cover - exercised only on local CUDA toolchain issues.
        _EXT_ERROR = exc
        raise


def grld_decode_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _load_grld_decode_ext()
    except BaseException:
        return False
    return True


class _GRLDDecodeFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base: torch.Tensor,
        detail: torch.Tensor,
        gate: torch.Tensor,
        coeff: torch.Tensor,
        child_to_parent: torch.Tensor,
        child_order: torch.Tensor,
        parent_start: torch.Tensor,
        parent_count: torch.Tensor,
        branch_scale: torch.Tensor,
    ) -> torch.Tensor:
        ext = _load_grld_decode_ext()
        scale = branch_scale.reshape(()).to(device=base.device, dtype=base.dtype)
        out = ext.grld_decode_forward(
            base.contiguous(),
            detail.contiguous(),
            gate.contiguous(),
            coeff.contiguous(),
            child_to_parent.to(device=base.device, dtype=torch.long).contiguous(),
            scale,
        )
        ctx.save_for_backward(
            base,
            detail,
            gate,
            coeff,
            child_to_parent.to(device=base.device, dtype=torch.long),
            child_order.to(device=base.device, dtype=torch.long),
            parent_start.to(device=base.device, dtype=torch.long),
            parent_count.to(device=base.device, dtype=torch.long),
            scale,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        base, detail, gate, coeff, child_to_parent, child_order, parent_start, parent_count, scale = ctx.saved_tensors
        ext = _load_grld_decode_ext()
        grad_base, grad_detail, grad_gate, grad_coeff, grad_scale = ext.grld_decode_backward(
            grad_out.contiguous(),
            base.contiguous(),
            detail.contiguous(),
            gate.contiguous(),
            coeff.contiguous(),
            child_to_parent.contiguous(),
            child_order.contiguous(),
            parent_start.contiguous(),
            parent_count.contiguous(),
            scale.contiguous(),
        )
        return grad_base, grad_detail, grad_gate, grad_coeff, None, None, None, None, grad_scale


def grld_decode(
    base: torch.Tensor,
    detail: torch.Tensor,
    gate: torch.Tensor,
    coeff: torch.Tensor,
    child_to_parent: torch.Tensor,
    child_order: torch.Tensor,
    parent_start: torch.Tensor,
    parent_count: torch.Tensor,
    branch_scale: torch.Tensor,
) -> torch.Tensor:
    if not base.is_cuda:
        raise RuntimeError("GRLD fused decode requires CUDA tensors")
    return _GRLDDecodeFn.apply(
        base,
        detail,
        gate,
        coeff,
        child_to_parent,
        child_order,
        parent_start,
        parent_count,
        branch_scale,
    )


__all__ = ["grld_decode", "grld_decode_available"]
