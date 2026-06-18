from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Optional, Tuple, Union

import torch

CacheValue = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


def _cache_dtype(name: str) -> torch.dtype:
    name_l = str(name).lower()
    if name_l in {"fp16", "float16", "half"}:
        return torch.float16
    if name_l in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name_l in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported DINO cache dtype={name!r}")


@dataclass
class DINOFeatureCacheStats:
    hit_l1: float = 0.0
    hit_l2: float = 0.0
    miss: float = 0.0
    h2d_ms: float = 0.0
    cpu_mb: float = 0.0
    gpu_mb: float = 0.0
    dtype_id: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "iforward/dino/cache_hit_l1": float(self.hit_l1),
            "iforward/dino/cache_hit_l2": float(self.hit_l2),
            "iforward/dino/cache_miss": float(self.miss),
            "iforward/dino/h2d_ms": float(self.h2d_ms),
            "iforward/dino/cache_cpu_mb": float(self.cpu_mb),
            "iforward/dino/cache_gpu_mb": float(self.gpu_mb),
            "iforward/dino/feature_dtype_id": float(self.dtype_id),
        }


class DINOFeatureCache:
    def __init__(
        self,
        *,
        dtype: str = "float16",
        cpu_pinned: bool = True,
        cpu_max_items: int = 64,
        gpu_max_items: int = 2,
        async_copy: bool = True,
        fail_if_trainable: bool = True,
    ) -> None:
        self.dtype = _cache_dtype(dtype)
        self.cpu_pinned = bool(cpu_pinned)
        self.cpu_max_items = max(int(cpu_max_items), 0)
        self.gpu_max_items = max(int(gpu_max_items), 0)
        self.async_copy = bool(async_copy)
        self.fail_if_trainable = bool(fail_if_trainable)
        self._cpu: "OrderedDict[Hashable, CacheValue]" = OrderedDict()
        self._gpu: "OrderedDict[tuple[Hashable, str], CacheValue]" = OrderedDict()

    def _dtype_id(self) -> float:
        if self.dtype == torch.float16:
            return 16.0
        if self.dtype == torch.bfloat16:
            return 161.0
        if self.dtype == torch.float32:
            return 32.0
        return -1.0

    @staticmethod
    def _tensor_mb(x: torch.Tensor) -> float:
        return float(x.numel() * x.element_size()) / (1024.0 * 1024.0)

    @classmethod
    def _value_mb(cls, value: CacheValue) -> float:
        if torch.is_tensor(value):
            return cls._tensor_mb(value)
        return float(sum(cls._tensor_mb(x) for x in value))

    @staticmethod
    def _detach_value(value: CacheValue) -> CacheValue:
        if torch.is_tensor(value):
            return value.detach()
        return tuple(x.detach() for x in value)

    def _to_cache_dtype_cpu(self, value: CacheValue) -> CacheValue:
        if torch.is_tensor(value):
            out = value.to(device="cpu", dtype=self.dtype).detach()
            return out.pin_memory() if self.cpu_pinned and torch.cuda.is_available() else out
        out_items = []
        for item in value:
            cpu = item.to(device="cpu", dtype=self.dtype).detach()
            if self.cpu_pinned and torch.cuda.is_available():
                cpu = cpu.pin_memory()
            out_items.append(cpu)
        return tuple(out_items)

    @staticmethod
    def _to_device(value: CacheValue, *, device: torch.device, non_blocking: bool) -> CacheValue:
        if torch.is_tensor(value):
            return value.to(device=device, non_blocking=bool(non_blocking)).detach()
        return tuple(x.to(device=device, non_blocking=bool(non_blocking)).detach() for x in value)

    def _totals(self) -> tuple[float, float]:
        cpu_mb = sum(self._value_mb(x) for x in self._cpu.values())
        gpu_mb = sum(self._value_mb(x) for x in self._gpu.values())
        return float(cpu_mb), float(gpu_mb)

    def _evict(self) -> None:
        while self.cpu_max_items >= 0 and len(self._cpu) > self.cpu_max_items:
            self._cpu.popitem(last=False)
        while self.gpu_max_items >= 0 and len(self._gpu) > self.gpu_max_items:
            self._gpu.popitem(last=False)

    def clear(self) -> None:
        self._cpu.clear()
        self._gpu.clear()

    def get_or_compute(
        self,
        *,
        key: Hashable,
        device: torch.device,
        compute: Callable[[], CacheValue],
        trainable: bool = False,
    ) -> tuple[CacheValue, DINOFeatureCacheStats]:
        if bool(trainable) and bool(self.fail_if_trainable):
            raise RuntimeError("DINO adapter-output cache cannot be used while DINO adapter parameters require grad")
        device_obj = torch.device(device)
        stats = DINOFeatureCacheStats(dtype_id=self._dtype_id())
        gpu_key = (key, str(device_obj))
        cached = self._gpu.get(gpu_key)
        if cached is not None:
            self._gpu.move_to_end(gpu_key)
            stats.hit_l1 = 1.0
            cpu_mb, gpu_mb = self._totals()
            stats.cpu_mb = cpu_mb
            stats.gpu_mb = gpu_mb
            return self._detach_value(cached), stats

        cached_cpu = self._cpu.get(key)
        if cached_cpu is not None:
            self._cpu.move_to_end(key)
            start = torch.cuda.Event(enable_timing=True) if device_obj.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device_obj.type == "cuda" else None
            if start is not None and end is not None:
                start.record()
            out = self._to_device(cached_cpu, device=device_obj, non_blocking=bool(self.async_copy))
            if end is not None:
                end.record()
                if not self.async_copy:
                    torch.cuda.synchronize(device_obj)
                    stats.h2d_ms = float(start.elapsed_time(end))
            if self.gpu_max_items > 0:
                self._gpu[gpu_key] = out
                self._gpu.move_to_end(gpu_key)
                self._evict()
            stats.hit_l2 = 1.0
            cpu_mb, gpu_mb = self._totals()
            stats.cpu_mb = cpu_mb
            stats.gpu_mb = gpu_mb
            return out, stats

        with torch.no_grad():
            out = self._detach_value(compute())
        if self.gpu_max_items > 0:
            self._gpu[gpu_key] = out
            self._gpu.move_to_end(gpu_key)
        if self.cpu_max_items > 0:
            self._cpu[key] = self._to_cache_dtype_cpu(out)
            self._cpu.move_to_end(key)
        self._evict()
        stats.miss = 1.0
        cpu_mb, gpu_mb = self._totals()
        stats.cpu_mb = cpu_mb
        stats.gpu_mb = gpu_mb
        return out, stats


__all__ = ["DINOFeatureCache", "DINOFeatureCacheStats"]
