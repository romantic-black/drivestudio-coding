from __future__ import annotations

import torch
import torch.nn as nn

from models.streetforward.production_training_utils import (
    ProductionAmpConfig,
    ProductionOptimizerConfig,
    build_adamw_optimizer,
    build_grad_scaler,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.norm = nn.LayerNorm(3)
        self.embedding = nn.Embedding(8, 3)


def _group_by_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> dict:
    for group in optimizer.param_groups:
        if float(group.get("weight_decay", 0.0)) == float(weight_decay):
            return group
    raise AssertionError(f"missing param group with weight_decay={weight_decay}")


def test_build_adamw_optimizer_builds_decay_and_no_decay_groups() -> None:
    model = _TinyModel()
    cfg = ProductionOptimizerConfig(
        lr=1.0e-4,
        weight_decay=1.0e-2,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    optimizer = build_adamw_optimizer(model, cfg=cfg)

    decay_group = _group_by_weight_decay(optimizer, cfg.weight_decay)
    no_decay_group = _group_by_weight_decay(optimizer, 0.0)
    decay_ids = {id(p) for p in decay_group["params"]}
    no_decay_ids = {id(p) for p in no_decay_group["params"]}

    assert id(model.linear.weight) in decay_ids
    assert id(model.linear.bias) in no_decay_ids
    assert id(model.norm.weight) in no_decay_ids
    assert id(model.norm.bias) in no_decay_ids
    assert id(model.embedding.weight) in no_decay_ids


def test_build_grad_scaler_disables_on_cpu() -> None:
    scaler = build_grad_scaler(
        amp_cfg=ProductionAmpConfig(enable=True, dtype="fp16"),
        device=torch.device("cpu"),
    )
    assert hasattr(scaler, "is_enabled")
    assert bool(scaler.is_enabled()) is False
