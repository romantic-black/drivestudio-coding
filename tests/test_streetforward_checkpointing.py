from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from tools.streetforward_checkpointing import (
    load_stage5_3_production_lightweight_checkpoint,
    production_model_state_dict,
    save_stage5_3_production_lightweight_checkpoint,
)


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(4, 2)
        self.register_buffer("node_states_tmp", torch.ones(1))
        self.register_buffer("h_cache_tmp", torch.ones(1))
        self.register_buffer("stage5_2_history_tmp", torch.ones(1))
        self.register_buffer("stage5_2_block_support_tmp", torch.ones(1))
        self.register_buffer("_last_full_inputs", torch.ones(1))
        self.runtime_cleared = False

    def clear_runtime_state_for_lightweight_resume(self) -> None:
        self.runtime_cleared = True


class _DummyScheduler:
    def __init__(self, *, global_step: int = 0, epoch_idx: int = 0) -> None:
        self.global_step = int(global_step)
        self.epoch_idx = int(epoch_idx)
        self.loaded_state = None

    def is_at_episode_boundary(self) -> bool:
        return True

    def production_state_dict(self) -> dict:
        return {
            "global_step": int(self.global_step),
            "epoch_idx": int(self.epoch_idx),
            "current_episode_state": None,
        }

    def load_production_state_dict(self, state: dict) -> None:
        self.loaded_state = dict(state)
        self.global_step = int(state.get("global_step", 0))
        self.epoch_idx = int(state.get("epoch_idx", 0))


class _BadScheduler(_DummyScheduler):
    def load_production_state_dict(self, state: dict) -> None:
        super().load_production_state_dict(state)
        self.global_step += 1


def _build_training_bits(model: nn.Module) -> tuple:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    for _ in range(5):
        optimizer.step()
        lr_scheduler.step()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    return optimizer, lr_scheduler, grad_scaler


def _save_ckpt(path: Path, model: _DummyModel, scheduler: _DummyScheduler) -> None:
    optimizer, lr_scheduler, grad_scaler = _build_training_bits(model)
    save_stage5_3_production_lightweight_checkpoint(
        str(path),
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        train_scheduler=scheduler,
        global_step=int(scheduler.global_step),
        epoch_idx=int(scheduler.epoch_idx),
        cfg=OmegaConf.create({"training": {"max_iterations": 10}}),
    )


def test_production_model_state_dict_filters_runtime_prefixes() -> None:
    model = _DummyModel()
    state = production_model_state_dict(model)
    assert "core.weight" in state
    assert "core.bias" in state
    assert "node_states_tmp" not in state
    assert "h_cache_tmp" not in state
    assert "stage5_2_history_tmp" not in state
    assert "stage5_2_block_support_tmp" not in state
    assert "_last_full_inputs" not in state


def test_lightweight_checkpoint_load_roundtrip(tmp_path: Path) -> None:
    model = _DummyModel()
    scheduler = _DummyScheduler(global_step=5, epoch_idx=2)
    optimizer, lr_scheduler, grad_scaler = _build_training_bits(model)
    path = tmp_path / "sf_lightweight.pt"
    payload = save_stage5_3_production_lightweight_checkpoint(
        str(path),
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        train_scheduler=scheduler,
        global_step=5,
        epoch_idx=2,
        cfg=OmegaConf.create({"training": {"max_iterations": 10}}),
    )
    assert "node_states_tmp" not in payload["model_state_dict"]

    restored = load_stage5_3_production_lightweight_checkpoint(
        str(path),
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        train_scheduler=scheduler,
        strict=True,
    )
    assert int(restored["global_step"]) == 5
    assert bool(model.runtime_cleared) is True


def test_lightweight_checkpoint_rejects_scheduler_step_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "sf_lightweight_bad_scheduler.pt"
    model = _DummyModel()
    _save_ckpt(path, model, _DummyScheduler(global_step=5, epoch_idx=2))
    optimizer, lr_scheduler, grad_scaler = _build_training_bits(model)

    with pytest.raises(ValueError, match="Scheduler global_step mismatch"):
        load_stage5_3_production_lightweight_checkpoint(
            str(path),
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            train_scheduler=_BadScheduler(global_step=0, epoch_idx=0),
            strict=True,
        )


def test_lightweight_checkpoint_rejects_lr_last_epoch_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "sf_lightweight_bad_lr.pt"
    model = _DummyModel()
    _save_ckpt(path, model, _DummyScheduler(global_step=5, epoch_idx=2))
    raw = torch.load(str(path), map_location="cpu")
    raw["lr_scheduler_state_dict"]["last_epoch"] = 999
    torch.save(raw, str(path))

    optimizer, lr_scheduler, grad_scaler = _build_training_bits(model)
    with pytest.raises(ValueError, match="LR scheduler last_epoch mismatch"):
        load_stage5_3_production_lightweight_checkpoint(
            str(path),
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            train_scheduler=_DummyScheduler(global_step=0, epoch_idx=0),
            strict=True,
        )
