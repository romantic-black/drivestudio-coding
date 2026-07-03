from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.iforward.stage3_0.cuda_sparse_gather import (
    cuda_sparse_gather_available,
    cuda_sparse_gather_feature_only_backward_available,
    sparse_gather_2d,
    sparse_gather_2d_pytorch_reference,
)
from models.iforward.stage3_0.sparse_gather_lift import GatherConfig, SparseGatherLift


def _cuda_sparse_available() -> bool:
    return torch.cuda.is_available() and cuda_sparse_gather_available()


def _cuda_sparse_feature_only_available() -> bool:
    return _cuda_sparse_available() and cuda_sparse_gather_feature_only_backward_available()


def test_auto_backend_cpu_falls_back_to_pytorch_reference() -> None:
    feature = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3, 1)
    uv = torch.tensor(
        [
            [[[1.0, 1.0], [0.0, 0.0]], [[2.0, 1.0], [10.0, 10.0]]],
            [[[0.0, 2.0], [1.5, 1.5]], [[1.0, 0.0], [-2.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([[[0.6, 0.4], [0.5, 0.5]], [[0.2, 0.8], [1.0, 0.0]]], dtype=torch.float32)
    valid = torch.tensor([[[True, True], [True, True]], [[True, False], [True, True]]])
    out, inbound, backend = sparse_gather_2d(
        feature,
        uv,
        weights,
        valid,
        image_height=3,
        image_width=3,
        backend="auto",
    )
    expected, expected_inbound = sparse_gather_2d_pytorch_reference(
        feature,
        uv,
        weights,
        valid,
        image_height=3,
        image_width=3,
    )
    assert backend == "pytorch"
    assert torch.allclose(out, expected, atol=1.0e-6)
    assert torch.equal(inbound, expected_inbound)


def test_explicit_cuda_backend_failfast_on_cpu() -> None:
    feature = torch.zeros((1, 2, 2, 1), dtype=torch.float32)
    uv = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    weights = torch.ones((1, 1, 1), dtype=torch.float32)
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    with pytest.raises(RuntimeError, match="CUDA sparse gather"):
        sparse_gather_2d(
            feature,
            uv,
            weights,
            valid,
            image_height=2,
            image_width=2,
            backend="cuda",
        )


@pytest.mark.skipif(not _cuda_sparse_available(), reason="Stage3 CUDA sparse gather op unavailable")
def test_cuda_sparse_gather_forward_matches_pytorch_reference() -> None:
    device = torch.device("cuda")
    torch.manual_seed(5)
    feature = torch.randn((2, 5, 6, 4), device=device, dtype=torch.float32)
    uv = torch.rand((3, 2, 3, 2), device=device, dtype=torch.float32)
    uv[..., 0] = uv[..., 0] * 4.5 + 0.25
    uv[..., 1] = uv[..., 1] * 3.5 + 0.25
    uv[0, 0, 0] = torch.tensor([-2.0, 1.0], device=device)
    weights = torch.randn((3, 2, 3), device=device, dtype=torch.float32)
    valid = torch.ones((3, 2, 3), device=device, dtype=torch.bool)
    valid[1, 1, 2] = False

    out, inbound, backend = sparse_gather_2d(
        feature,
        uv,
        weights,
        valid,
        image_height=5,
        image_width=6,
        backend="cuda",
    )
    expected, expected_inbound = sparse_gather_2d_pytorch_reference(
        feature,
        uv,
        weights,
        valid,
        image_height=5,
        image_width=6,
    )
    assert backend == "cuda"
    assert torch.equal(inbound, expected_inbound)
    assert torch.allclose(out, expected, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.skipif(not _cuda_sparse_available(), reason="Stage3 CUDA sparse gather op unavailable")
def test_cuda_sparse_gather_force_fp32_kernel_under_autocast() -> None:
    device = torch.device("cuda")
    feature = torch.arange(18, device=device, dtype=torch.float16).reshape(2, 3, 3, 1)
    uv = torch.tensor(
        [[[[1.0, 1.0]], [[2.0, 1.0]]], [[[0.0, 2.0]], [[1.0, 0.0]]]],
        device=device,
        dtype=torch.float16,
    )
    weights = torch.ones((2, 2, 1), device=device, dtype=torch.float16)
    valid = torch.ones((2, 2, 1), device=device, dtype=torch.bool)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        out, _inbound, backend = sparse_gather_2d(
            feature,
            uv,
            weights,
            valid,
            image_height=3,
            image_width=3,
            backend="cuda",
        )
    assert backend == "cuda"
    assert out.dtype is torch.float32


@pytest.mark.skipif(not _cuda_sparse_available(), reason="Stage3 CUDA sparse gather op unavailable")
def test_cuda_sparse_gather_backward_matches_pytorch_reference() -> None:
    device = torch.device("cuda")
    torch.manual_seed(7)
    feature = torch.randn((2, 5, 6, 3), device=device, dtype=torch.float32, requires_grad=True)
    uv = torch.rand((4, 2, 3, 2), device=device, dtype=torch.float32)
    uv[..., 0] = uv[..., 0] * 4.0 + 0.5
    uv[..., 1] = uv[..., 1] * 3.0 + 0.5
    uv.requires_grad_()
    weights = torch.randn((4, 2, 3), device=device, dtype=torch.float32, requires_grad=True)
    valid = torch.rand((4, 2, 3), device=device) > 0.25
    grad = torch.randn((4, 3), device=device, dtype=torch.float32)

    out, _inbound, _backend = sparse_gather_2d(
        feature,
        uv,
        weights,
        valid,
        image_height=5,
        image_width=6,
        backend="cuda",
    )
    (out * grad).sum().backward()
    grads_cuda = (feature.grad.detach().clone(), uv.grad.detach().clone(), weights.grad.detach().clone())

    feature_ref = feature.detach().clone().requires_grad_()
    uv_ref = uv.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()
    out_ref, _ = sparse_gather_2d_pytorch_reference(
        feature_ref,
        uv_ref,
        weights_ref,
        valid,
        image_height=5,
        image_width=6,
    )
    (out_ref * grad).sum().backward()
    assert torch.allclose(grads_cuda[0], feature_ref.grad, atol=3.0e-4, rtol=3.0e-4)
    assert torch.allclose(grads_cuda[1], uv_ref.grad, atol=3.0e-4, rtol=3.0e-4)
    assert torch.allclose(grads_cuda[2], weights_ref.grad, atol=3.0e-4, rtol=3.0e-4)


@pytest.mark.skipif(not _cuda_sparse_feature_only_available(), reason="Stage3 CUDA sparse gather feature-only bwd op unavailable")
def test_cuda_sparse_gather_feature_only_backward_skips_uv_and_weight_grads(monkeypatch: pytest.MonkeyPatch) -> None:
    from gsplat.cuda import _wrapper as gsplat_wrapper

    device = torch.device("cuda")
    torch.manual_seed(11)
    calls = {"feature_only": 0, "full": 0}
    original_feature_only = gsplat_wrapper.sparse_gather_2d_bwd_feature_only
    original_full = gsplat_wrapper.sparse_gather_2d_bwd

    def _feature_only(*args, **kwargs):
        calls["feature_only"] += 1
        return original_feature_only(*args, **kwargs)

    def _full(*args, **kwargs):
        calls["full"] += 1
        return original_full(*args, **kwargs)

    monkeypatch.setattr(gsplat_wrapper, "sparse_gather_2d_bwd_feature_only", _feature_only)
    monkeypatch.setattr(gsplat_wrapper, "sparse_gather_2d_bwd", _full)
    feature = torch.randn((2, 4, 5, 3), device=device, dtype=torch.float32, requires_grad=True)
    uv = torch.rand((4, 2, 3, 2), device=device, dtype=torch.float32)
    uv[..., 0] = uv[..., 0] * 4.0
    uv[..., 1] = uv[..., 1] * 3.0
    weights = torch.randn((4, 2, 3), device=device, dtype=torch.float32)
    valid = torch.ones((4, 2, 3), device=device, dtype=torch.bool)
    grad = torch.randn((4, 3), device=device, dtype=torch.float32)

    out, _inbound, backend = sparse_gather_2d(
        feature,
        uv,
        weights,
        valid,
        image_height=4,
        image_width=5,
        backend="cuda",
    )
    assert backend == "cuda"
    (out * grad).sum().backward()
    feature_grad = feature.grad.detach().clone()
    assert calls["feature_only"] == 1
    assert calls["full"] == 0
    assert uv.grad is None
    assert weights.grad is None

    feature_ref = feature.detach().clone().requires_grad_()
    uv_ref = uv.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()
    out_ref, _inbound_ref = sparse_gather_2d_pytorch_reference(
        feature_ref,
        uv_ref,
        weights_ref,
        valid,
        image_height=4,
        image_width=5,
    )
    (out_ref * grad).sum().backward()
    assert torch.allclose(feature_grad, feature_ref.grad, atol=3.0e-4, rtol=3.0e-4)
    assert uv_ref.grad is not None
    assert weights_ref.grad is not None


class _RaiseIfCalled(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("module should not be called")


@pytest.mark.skipif(not _cuda_sparse_available(), reason="Stage3 CUDA sparse gather op unavailable")
def test_fixed_center_cuda_backend_skips_query_head_and_grid_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    import models.iforward.stage3_0.cuda_sparse_gather as cuda_sparse_gather
    import models.iforward.stage3_0.sparse_gather_lift as sparse_gather_lift

    def _raise_grid_sample(*args, **kwargs):
        raise AssertionError("PyTorch grid_sample fallback should not be called")

    def _raise_prepare(*args, **kwargs):
        raise AssertionError("Explicit CUDA sparse gather should not prepare NCHW value maps")

    monkeypatch.setattr(cuda_sparse_gather, "sparse_grid_sample_prepared", _raise_grid_sample)
    monkeypatch.setattr(sparse_gather_lift, "prepare_value_nchw", _raise_prepare)
    device = torch.device("cuda")
    value_map = torch.arange(18, device=device, dtype=torch.float32).reshape(2, 3, 3, 1)
    anchor_uv = torch.tensor([[[1.0, 1.0], [0.0, 2.0]], [[0.0, 0.0], [2.0, 2.0]]], device=device)
    support = torch.ones((2, 2), device=device, dtype=torch.float32)
    valid = torch.ones((2, 2), device=device, dtype=torch.bool)
    depth = torch.ones((2, 2), device=device, dtype=torch.float32)
    radius = torch.ones((2, 2), device=device, dtype=torch.float32)
    gather = SparseGatherLift(
        value_dim=1,
        config=GatherConfig(
            query_dim=4,
            num_taps=5,
            chunk_size=8,
            use_geometry_pe=False,
            fixed_center_steps=10,
            fixed_center_fast_path=True,
            backend="cuda",
        ),
    ).to(device)
    gather.head = _RaiseIfCalled()
    gather.view_logit_head = _RaiseIfCalled()
    gather.tap_logit_head = _RaiseIfCalled()
    gather.offset_head = _RaiseIfCalled()
    gather.gate_head = _RaiseIfCalled()
    out, conf, aux, _reg = gather(
        value_map=value_map,
        anchor_uv=anchor_uv,
        support=support,
        valid=valid,
        depth=depth,
        radius=radius,
        image_height=3,
        image_width=3,
        query=None,
        global_step=0,
    )
    assert out.shape == (2, 1)
    assert torch.all(conf > 0.0)
    assert aux["iforward/stage3/gather_fixed_fast_path_enabled"] == 1.0
    assert aux["iforward/stage3/gather_cuda_backend_enabled"] == 1.0
