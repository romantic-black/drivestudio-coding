from __future__ import annotations

from contextlib import nullcontext
from typing import Literal, Optional, Tuple

import torch

from .sparse_grid_sample import prepare_value_nchw, sparse_grid_sample_prepared

SparseGatherBackend = Literal["auto", "cuda", "pytorch"]


def _cuda_fp32_context():
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def cuda_sparse_gather_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from gsplat.cuda._backend import _C  # type: ignore
    except Exception:
        return False
    return hasattr(_C, "sparse_gather_2d_fwd") and hasattr(_C, "sparse_gather_2d_bwd")


def cuda_sparse_gather_feature_only_backward_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from gsplat.cuda._backend import _C  # type: ignore
    except Exception:
        return False
    return hasattr(_C, "sparse_gather_2d_bwd_feature_only")


def can_use_cuda_sparse_gather(feature_map: torch.Tensor, *, backend: str = "auto") -> bool:
    backend = str(backend).lower()
    if backend == "pytorch":
        return False
    if backend not in {"auto", "cuda"}:
        raise ValueError(f"unsupported sparse gather backend={backend!r}")
    ok = (
        feature_map.is_cuda
        and feature_map.dtype == torch.float32
        and feature_map.dim() == 4
        and cuda_sparse_gather_available()
    )
    if backend == "cuda" and not ok:
        raise RuntimeError(
            "Stage3 CUDA sparse gather requires CUDA float32 feature_map and a gsplat.csrc "
            "build exposing sparse_gather_2d_fwd/bwd."
        )
    return bool(ok)


class _SparseGather2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        feature_map: torch.Tensor,
        uv: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from gsplat.cuda import _wrapper as gsplat_wrapper

        feature_map_c = feature_map.contiguous()
        uv_c = uv.contiguous()
        weights_c = weights.contiguous()
        valid_c = valid.contiguous()
        out, inbound = gsplat_wrapper.sparse_gather_2d_fwd(
            feature_map_c,
            uv_c,
            weights_c,
            valid_c,
            int(image_height),
            int(image_width),
        )
        save_feature_map = bool(uv.requires_grad or weights.requires_grad) or not cuda_sparse_gather_feature_only_backward_available()
        ctx.save_feature_map = bool(save_feature_map)
        ctx.feature_height = int(feature_map_c.shape[1])
        ctx.feature_width = int(feature_map_c.shape[2])
        if bool(save_feature_map):
            ctx.save_for_backward(feature_map_c, uv_c, weights_c, valid_c)
        else:
            ctx.save_for_backward(uv_c, weights_c, valid_c)
        ctx.image_height = int(image_height)
        ctx.image_width = int(image_width)
        return out, inbound

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_inbound: Optional[torch.Tensor] = None):
        del grad_inbound
        from gsplat.cuda import _wrapper as gsplat_wrapper

        need_grad_feature = bool(ctx.needs_input_grad[0])
        need_grad_uv = bool(ctx.needs_input_grad[1])
        need_grad_weights = bool(ctx.needs_input_grad[2])
        if bool(ctx.save_feature_map):
            feature_map, uv, weights, valid = ctx.saved_tensors
            grad_feature, grad_uv, grad_weights = gsplat_wrapper.sparse_gather_2d_bwd(
                grad_out.contiguous(),
                feature_map,
                uv,
                weights,
                valid,
                int(ctx.image_height),
                int(ctx.image_width),
                need_grad_uv=need_grad_uv,
                need_grad_weights=need_grad_weights,
            )
        else:
            uv, weights, valid = ctx.saved_tensors
            if bool(need_grad_uv or need_grad_weights):
                raise RuntimeError("Stage3 CUDA sparse gather feature-only backward cannot return uv/weight gradients.")
            grad_feature = (
                gsplat_wrapper.sparse_gather_2d_bwd_feature_only(
                    grad_out.contiguous(),
                    uv,
                    weights,
                    valid,
                    int(ctx.feature_height),
                    int(ctx.feature_width),
                    int(ctx.image_height),
                    int(ctx.image_width),
                )
                if bool(need_grad_feature)
                else None
            )
            grad_uv = None
            grad_weights = None
        if not need_grad_feature:
            grad_feature = None
        if not need_grad_uv:
            grad_uv = None
        if not need_grad_weights:
            grad_weights = None
        return grad_feature, grad_uv, grad_weights, None, None, None


def sparse_gather_2d_pytorch_reference(
    feature_map: torch.Tensor,
    uv: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    prepared_value_nchw: Optional[torch.Tensor] = None,
    chunk_size: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prepared = prepared_value_nchw if prepared_value_nchw is not None else prepare_value_nchw(feature_map)
    sampled, inbound = sparse_grid_sample_prepared(
        prepared,
        uv,
        image_height=int(image_height),
        image_width=int(image_width),
        chunk_size=int(chunk_size),
    )
    effective = torch.where(valid & inbound, weights, torch.zeros_like(weights))
    out = (sampled * effective.unsqueeze(-1)).sum(dim=(1, 2))
    return out, inbound


def sparse_gather_2d(
    feature_map: torch.Tensor,
    uv: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    backend: SparseGatherBackend = "auto",
    prepared_value_nchw: Optional[torch.Tensor] = None,
    chunk_size: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    backend = str(backend).lower()
    if backend not in {"auto", "cuda", "pytorch"}:
        raise ValueError(f"unsupported sparse gather backend={backend!r}")
    if uv.dim() != 4 or int(uv.shape[-1]) != 2:
        raise ValueError(f"uv must be [R,V,K,2], got {tuple(uv.shape)}")
    if weights.shape != uv.shape[:3] or valid.shape != uv.shape[:3]:
        raise ValueError("weights/valid must match uv.shape[:3]")
    if valid.dtype != torch.bool:
        valid = valid.to(dtype=torch.bool)
    cuda_feature = feature_map
    cuda_uv = uv
    cuda_weights = weights
    if backend != "pytorch" and feature_map.is_cuda:
        cuda_feature = feature_map.to(dtype=torch.float32)
        cuda_uv = uv.to(device=feature_map.device, dtype=torch.float32)
        cuda_weights = weights.to(device=feature_map.device, dtype=torch.float32)
    if backend != "pytorch" and can_use_cuda_sparse_gather(cuda_feature, backend=backend):
        if cuda_uv.dtype != torch.float32 or cuda_weights.dtype != torch.float32:
            if backend == "cuda":
                raise RuntimeError("Stage3 CUDA sparse gather v1 requires float32 uv and weights.")
        else:
            with _cuda_fp32_context():
                out, inbound = _SparseGather2DFunction.apply(
                    cuda_feature,
                    cuda_uv,
                    cuda_weights,
                    valid,
                    int(image_height),
                    int(image_width),
                )
            return out, inbound, "cuda"
    out, inbound = sparse_gather_2d_pytorch_reference(
        feature_map,
        uv,
        weights,
        valid,
        image_height=int(image_height),
        image_width=int(image_width),
        prepared_value_nchw=prepared_value_nchw,
        chunk_size=int(chunk_size),
    )
    return out, inbound, "pytorch"


__all__ = [
    "SparseGatherBackend",
    "can_use_cuda_sparse_gather",
    "cuda_sparse_gather_available",
    "cuda_sparse_gather_feature_only_backward_available",
    "sparse_gather_2d",
    "sparse_gather_2d_pytorch_reference",
]
