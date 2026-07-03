from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .utils import cfg_get


def _amp_cfg(config: Any) -> Any:
    training = cfg_get(config, "training", None)
    if training is not None:
        return cfg_get(training, "amp", {}) or {}
    return config or {}


def _bool_cfg(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def resolve_amp_dtype(config_or_amp_cfg: Any) -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    amp_cfg = _amp_cfg(config_or_amp_cfg)
    name = str(cfg_get(amp_cfg, "dtype", "auto")).strip().lower()
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half", "16"}:
        return torch.float16
    if name in {"fp32", "float32", "32", "none", "off", "false"}:
        return None
    raise ValueError(f"unsupported training.amp.dtype={name!r}")


def amp_dtype_name(dtype: Optional[torch.dtype]) -> str:
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    return "fp32"


def amp_dtype_id(dtype: Optional[torch.dtype]) -> int:
    if dtype is torch.float16:
        return 1
    if dtype is torch.bfloat16:
        return 2
    return 0


def normalize_storage_dtype_name(value: Any, *, default: str = "fp32") -> str:
    name = str(default if value is None else value).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return "bf16"
    if name in {"fp16", "float16", "half", "16"}:
        return "fp16"
    if name in {"fp32", "float32", "32", "none", "off", "false"}:
        return "fp32"
    if name == "amp":
        return "amp"
    raise ValueError(f"unsupported AMP storage dtype={name!r}")


def storage_dtype_from_name(name: Any, *, amp_dtype: Optional[torch.dtype] = None, default: torch.dtype = torch.float32) -> torch.dtype:
    label = normalize_storage_dtype_name(name)
    if label == "bf16":
        return torch.bfloat16
    if label == "fp16":
        return torch.float16
    if label == "amp":
        return amp_dtype or default
    return torch.float32


def storage_dtype_id(name: Any, *, amp_dtype: Optional[torch.dtype] = None) -> int:
    return amp_dtype_id(storage_dtype_from_name(name, amp_dtype=amp_dtype))


def _resolve_grad_scaler(amp_cfg: Any, *, enabled: bool, dtype: Optional[torch.dtype]) -> bool:
    value = cfg_get(amp_cfg, "grad_scaler", "auto")
    if isinstance(value, str) and value.strip().lower() == "auto":
        return bool(enabled and dtype is torch.float16)
    return bool(enabled and _bool_cfg(value, default=False))


@dataclass(frozen=True)
class AmpPolicy:
    requested: bool
    enabled: bool
    dtype: Optional[torch.dtype]
    use_grad_scaler: bool
    cache_enabled: bool = True
    device_type: str = "cuda"
    render_force_fp32: bool = True
    loss_force_fp32: bool = True
    geometry_fp32: bool = True
    scalar_anchor_force_fp32: bool = True
    sparse_gather_force_fp32: bool = True
    parent_lift_amp: bool = True
    child_gather_amp: bool = False
    child_detail_output_dtype: str = "fp32"
    gdkv_state_dtype: str = "fp32"
    features_2d_cache_dtype: str = "fp32"
    parent_context_cache_dtype: str = "fp32"

    def autocast(self) -> Any:
        if not self.enabled or self.dtype is None or self.device_type != "cuda" or not torch.cuda.is_available():
            return nullcontext()
        return torch.amp.autocast(
            device_type="cuda",
            dtype=self.dtype,
            enabled=True,
            cache_enabled=bool(self.cache_enabled),
        )

    def fp32(self) -> Any:
        if self.device_type != "cuda" or not torch.cuda.is_available():
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", enabled=False)

    def metrics(self) -> Dict[str, float]:
        return {
            "amp/enabled": 1.0 if bool(self.enabled) else 0.0,
            "amp/requested": 1.0 if bool(self.requested) else 0.0,
            "amp/dtype_id": float(amp_dtype_id(self.dtype)),
            "amp/grad_scaler_enabled": 1.0 if bool(self.use_grad_scaler) else 0.0,
            "amp/autocast_forward_enabled": 1.0 if bool(self.enabled and self.dtype is not None) else 0.0,
            "amp/render_fp32": 1.0 if bool(self.render_force_fp32) else 0.0,
            "amp/geometry_fp32": 1.0 if bool(self.geometry_fp32) else 0.0,
            "amp/parent_lift_amp": 1.0 if bool(self.parent_lift_amp) else 0.0,
            "amp/child_gather_amp": 1.0 if bool(self.child_gather_amp) else 0.0,
            "amp/gdkv_state_dtype_id": float(storage_dtype_id(self.gdkv_state_dtype, amp_dtype=self.dtype)),
            "amp/dtype/features_2d_cache": float(storage_dtype_id(self.features_2d_cache_dtype, amp_dtype=self.dtype)),
            "amp/dtype/parent_context_cache": float(storage_dtype_id(self.parent_context_cache_dtype, amp_dtype=self.dtype)),
            "amp/dtype/child_detail": float(storage_dtype_id(self.child_detail_output_dtype, amp_dtype=self.dtype)),
        }


def build_amp_policy(config: Any, *, inference_only: bool = False) -> AmpPolicy:
    amp_cfg = _amp_cfg(config)
    if isinstance(amp_cfg, bool):
        requested = bool(amp_cfg)
        amp_cfg = {"enable": requested}
    else:
        requested = _bool_cfg(cfg_get(amp_cfg, "enable", False), default=False)
    device_type = str(cfg_get(amp_cfg, "autocast_device", "cuda")).strip().lower()
    dtype = resolve_amp_dtype(amp_cfg) if requested and device_type == "cuda" else None
    enabled = bool(requested and dtype is not None and device_type == "cuda" and torch.cuda.is_available())
    use_grad_scaler = False if bool(inference_only) else _resolve_grad_scaler(amp_cfg, enabled=enabled, dtype=dtype)

    fp32_islands = cfg_get(amp_cfg, "fp32_islands", {}) or {}
    stage3_cfg = cfg_get(amp_cfg, "stage3", {}) or {}
    memory_cfg = cfg_get(amp_cfg, "memory", {}) or {}
    storage_cfg = cfg_get(amp_cfg, "storage", {}) or {}
    render_cfg = cfg_get(amp_cfg, "render", {}) or {}
    return AmpPolicy(
        requested=requested,
        enabled=enabled,
        dtype=dtype,
        use_grad_scaler=use_grad_scaler,
        cache_enabled=_bool_cfg(cfg_get(amp_cfg, "cache_enabled", True), default=True),
        device_type=device_type,
        render_force_fp32=_bool_cfg(cfg_get(render_cfg, "force_fp32", cfg_get(fp32_islands, "render", True)), default=True),
        loss_force_fp32=_bool_cfg(cfg_get(render_cfg, "loss_force_fp32", cfg_get(fp32_islands, "loss", True)), default=True),
        geometry_fp32=_bool_cfg(cfg_get(fp32_islands, "geometry", True), default=True),
        scalar_anchor_force_fp32=_bool_cfg(cfg_get(stage3_cfg, "scalar_anchor_force_fp32", cfg_get(fp32_islands, "scalar_anchor", True)), default=True),
        sparse_gather_force_fp32=_bool_cfg(
            cfg_get(stage3_cfg, "cuda_sparse_gather_force_fp32_kernel", cfg_get(fp32_islands, "sparse_gather_cuda", True)),
            default=True,
        ),
        parent_lift_amp=_bool_cfg(cfg_get(stage3_cfg, "parent_lift_amp", True), default=True),
        child_gather_amp=_bool_cfg(cfg_get(stage3_cfg, "child_gather_amp", False), default=False),
        child_detail_output_dtype=normalize_storage_dtype_name(cfg_get(stage3_cfg, "child_detail_output_dtype", "fp32")),
        gdkv_state_dtype=normalize_storage_dtype_name(cfg_get(memory_cfg, "gdkv_state_dtype", "fp32")),
        features_2d_cache_dtype=normalize_storage_dtype_name(cfg_get(storage_cfg, "features_2d_cache_dtype", "fp32")),
        parent_context_cache_dtype=normalize_storage_dtype_name(cfg_get(storage_cfg, "parent_context_cache_dtype", "fp32")),
    )


def make_grad_scaler(config: Any, policy: AmpPolicy) -> Any:
    amp_cfg = _amp_cfg(config)
    kwargs = {
        "enabled": bool(policy.enabled and policy.use_grad_scaler),
        "init_scale": float(cfg_get(amp_cfg, "init_scale", 65536.0)),
        "growth_factor": float(cfg_get(amp_cfg, "growth_factor", 2.0)),
        "backoff_factor": float(cfg_get(amp_cfg, "backoff_factor", 0.5)),
        "growth_interval": int(cfg_get(amp_cfg, "growth_interval", 2000)),
    }
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    if scaler_cls is not None:
        try:
            return scaler_cls("cuda", **kwargs)
        except TypeError:
            try:
                return scaler_cls(device_type="cuda", **kwargs)
            except TypeError:
                return scaler_cls(**kwargs)
    return torch.cuda.amp.GradScaler(**kwargs)


__all__ = [
    "AmpPolicy",
    "amp_dtype_id",
    "amp_dtype_name",
    "build_amp_policy",
    "make_grad_scaler",
    "normalize_storage_dtype_name",
    "resolve_amp_dtype",
    "storage_dtype_from_name",
    "storage_dtype_id",
]
