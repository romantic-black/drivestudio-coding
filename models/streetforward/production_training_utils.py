from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ProductionOptimizerConfig:
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float


@dataclass(frozen=True)
class ProductionLRSchedulerConfig:
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float


@dataclass(frozen=True)
class ProductionAmpConfig:
    enable: bool
    dtype: str


@dataclass(frozen=True)
class ProductionGradClipConfig:
    enable: bool
    max_norm: float
    norm_type: float


@dataclass(frozen=True)
class ProductionBadStepConfig:
    fail_on_nonfinite_loss: bool
    fail_on_nonfinite_grad: bool
    fail_on_amp_overflow: bool
    fail_on_grad_norm_gt: Optional[float]


def parse_betas(raw: Any) -> Tuple[float, float]:
    if raw is None:
        raise ValueError("optimizer.betas is required for production training.")
    vals = tuple(float(x) for x in list(raw))
    if len(vals) != 2:
        raise ValueError(f"optimizer.betas must have length=2, got {len(vals)}")
    return vals[0], vals[1]


def iter_trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    for p in model.parameters():
        if p.requires_grad:
            yield p


def build_adamw_optimizer(
    model: torch.nn.Module,
    *,
    cfg: ProductionOptimizerConfig,
) -> torch.optim.Optimizer:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = str(name).lower()
        flat_tokens = lname.replace(".", "_").split("_")
        is_no_decay = bool(
            lname.endswith(".bias")
            or any(
                tok in {"bias", "layernorm", "groupnorm", "batchnorm", "instancenorm", "norm", "ln", "bn"}
                or tok.startswith("norm")
                or tok.startswith("ln")
                or tok.startswith("bn")
                or tok.startswith("embedding")
                for tok in flat_tokens
            )
        )
        if is_no_decay:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    if len(decay_params) + len(no_decay_params) == 0:
        raise ValueError("No trainable parameters found for production optimizer.")
    param_groups: List[Dict[str, Any]] = []
    if len(decay_params) > 0:
        param_groups.append(
            {
                "params": decay_params,
                "lr": float(cfg.lr),
                "weight_decay": float(cfg.weight_decay),
            }
        )
    if len(no_decay_params) > 0:
        param_groups.append(
            {
                "params": no_decay_params,
                "lr": float(cfg.lr),
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(param_groups, betas=(float(cfg.betas[0]), float(cfg.betas[1])), eps=float(cfg.eps))


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    cfg: ProductionLRSchedulerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(cfg.warmup_steps)
    total_steps = int(cfg.total_steps)
    min_lr_ratio = float(cfg.min_lr_ratio)
    if warmup_steps < 0:
        raise ValueError("lr_scheduler.warmup_steps must be >= 0.")
    if total_steps <= 0:
        raise ValueError("training.max_iterations must be > 0 for warmup_cosine scheduler.")
    if min_lr_ratio <= 0.0 or min_lr_ratio > 1.0:
        raise ValueError("lr_scheduler.min_lr_ratio must be in (0, 1].")

    def _lr_lambda(step_idx: int) -> float:
        step = int(max(step_idx, 0))
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return float(min_lr_ratio)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=_lr_lambda)


def build_grad_scaler(*, amp_cfg: ProductionAmpConfig, device: torch.device) -> torch.cuda.amp.GradScaler:
    enabled = bool(amp_cfg.enable and str(device.type) == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)  # type: ignore[return-value]
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)  # type: ignore[return-value]
    return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_amp_dtype(dtype_str: str) -> torch.dtype:
    v = str(dtype_str).strip().lower()
    if v == "fp16":
        return torch.float16
    if v == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported training.amp.dtype={dtype_str!r}, expected one of {{'fp16','bf16'}}.")


def read_checkpoint_cfg(cfg: Any) -> Dict[str, Any]:
    ckpt = cfg.get("checkpoint") if hasattr(cfg, "get") else None
    if ckpt is None:
        return {}
    out = {
        "type": ckpt.get("type"),
        "resume": ckpt.get("resume"),
        "save_at": ckpt.get("save_at"),
        "save_every_n_episodes": ckpt.get("save_every_n_episodes"),
        "keep_last_k": ckpt.get("keep_last_k"),
    }
    return out
