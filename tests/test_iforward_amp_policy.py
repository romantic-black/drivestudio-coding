from __future__ import annotations

import torch
import pytest

from models.iforward.amp_policy import (
    amp_dtype_id,
    build_amp_policy,
    normalize_storage_dtype_name,
    resolve_amp_dtype,
    storage_dtype_from_name,
)
from models.iforward.stage3_0.losses import stage3_gather_regularization


def test_amp_policy_resolve_dtype_auto_prefers_bf16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    cfg = {"training": {"amp": {"enable": True, "dtype": "auto", "grad_scaler": "auto"}}}
    policy = build_amp_policy(cfg)
    assert resolve_amp_dtype(cfg) is torch.bfloat16
    assert policy.enabled is True
    assert policy.dtype is torch.bfloat16
    assert policy.use_grad_scaler is False
    assert amp_dtype_id(policy.dtype) == 2


def test_amp_policy_resolve_dtype_auto_falls_back_to_fp16_scaler(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    policy = build_amp_policy({"training": {"amp": {"enable": True, "dtype": "auto", "grad_scaler": "auto"}}})
    assert policy.enabled is True
    assert policy.dtype is torch.float16
    assert policy.use_grad_scaler is True
    assert amp_dtype_id(policy.dtype) == 1


def test_amp_policy_disabled_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    policy = build_amp_policy({"training": {"amp": {"enable": True, "dtype": "bf16", "grad_scaler": "auto"}}})
    assert policy.requested is True
    assert policy.enabled is False
    assert policy.dtype is None
    assert policy.use_grad_scaler is False


def test_amp_policy_storage_dtype_parsing_and_metrics(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    cfg = {
        "training": {
            "amp": {
                "enable": True,
                "dtype": "auto",
                "storage": {
                    "features_2d_cache_dtype": "bf16",
                    "parent_context_cache_dtype": "amp",
                },
                "stage3": {
                    "child_detail_output_dtype": "bf16",
                },
                "memory": {
                    "gdkv_state_dtype": "bf16",
                },
            }
        }
    }
    policy = build_amp_policy(cfg)
    metrics = policy.metrics()
    assert policy.dtype is torch.bfloat16
    assert policy.features_2d_cache_dtype == "bf16"
    assert policy.parent_context_cache_dtype == "amp"
    assert policy.child_detail_output_dtype == "bf16"
    assert policy.gdkv_state_dtype == "bf16"
    assert metrics["amp/dtype/features_2d_cache"] == 2.0
    assert metrics["amp/dtype/parent_context_cache"] == 2.0
    assert metrics["amp/dtype/child_detail"] == 2.0
    assert metrics["amp/gdkv_state_dtype_id"] == 2.0


def test_storage_dtype_from_name_supports_amp_and_rejects_unknown() -> None:
    assert normalize_storage_dtype_name("bfloat16") == "bf16"
    assert storage_dtype_from_name("amp", amp_dtype=torch.bfloat16) is torch.bfloat16
    assert storage_dtype_from_name("amp", amp_dtype=None, default=torch.float16) is torch.float16
    with pytest.raises(ValueError, match="unsupported AMP storage dtype"):
        storage_dtype_from_name("int8")


def test_stage3_gather_regularization_force_fp32_loss() -> None:
    terms = {
        "offset_l2": torch.tensor(2.0, dtype=torch.float16),
        "out_of_bounds": torch.tensor(3.0, dtype=torch.float16),
    }
    loss, stats = stage3_gather_regularization(terms, offset_l2_weight=0.5, out_of_bounds_weight=2.0)
    assert loss.dtype is torch.float32
    assert torch.allclose(loss, torch.tensor(7.0, dtype=torch.float32))
    assert stats["iforward/stage3/loss_offset_l2_raw"] == 2.0
