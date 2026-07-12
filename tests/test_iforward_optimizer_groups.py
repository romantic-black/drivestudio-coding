from __future__ import annotations

import torch
import torch.nn as nn

from models.iforward.trainer import IForwardTrainer


class _BranchMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point = nn.Linear(2, 2)
        self.fuse = nn.Linear(2, 2)


class _Memory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bg = _BranchMemory()
        self.distant = _BranchMemory()


class _Updater(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(2, 2)
        self.vsm_ctx_adapter = nn.Linear(2, 2)


class _Runtime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_feature_extractor = nn.Module()
        self.image_feature_extractor.residual_unet = nn.Linear(2, 2)
        self.image_feature_extractor.fusion_neck = nn.Linear(2, 2)
        self.stage6_posterior_updater = _Updater()
        self.stage6_struct_event_decoder = nn.Linear(2, 2)
        self.stage6_measurement_trainable_param_names = {
            "image_feature_extractor.residual_unet.weight",
            "image_feature_extractor.residual_unet.bias",
            "image_feature_extractor.fusion_neck.weight",
            "image_feature_extractor.fusion_neck.bias",
        }


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory = _Memory()
        self.phase_a_runtime = _Runtime()


class _V6Model(nn.Module):
    is_v6_point_mamba_xcpe = True

    def __init__(self) -> None:
        super().__init__()
        self.point_mamba = nn.Linear(2, 2)
        self.local_conflict = nn.Linear(2, 2)
        self.context_adapter = nn.Linear(2, 2)
        self.phase_a_runtime = _Runtime()


class _V3Model(nn.Module):
    is_v3_gru_history_gate = True

    def __init__(self) -> None:
        super().__init__()
        self.point_gru = nn.Linear(2, 2)
        self.history_gate = nn.Linear(2, 2)
        self.phase_a_runtime = _Runtime()


def _cfg() -> dict:
    return {
        "model": {
            "iforward": {
                "trainability": {
                    "train_memory": True,
                    "train_memory_fuse": True,
                    "train_vsm_ctx_adapter": True,
                    "train_measurement_frontend": True,
                    "unfreeze_updater_base_after_step": 0,
                    "train_stage6_struct_decoder": True,
                    "unfreeze_struct_decoder_after_step": 0,
                }
            }
        },
        "optimizer": {
            "type": "adamw",
            "lr": {
                "default": 1.0e-4,
                "memory": 1.0e-4,
                "memory_fuse": 1.0e-4,
                "vsm_ctx_adapter": 2.0e-4,
                "stage6_posterior_updater_base": 1.0e-5,
                "stage6_struct_decoder": 1.0e-5,
                "measurement_frontend": 1.0e-5,
                "stage6_measurement_frontend_residual_unet": 1.0e-5,
                "stage6_measurement_frontend_fusion_neck": 1.0e-5,
            },
            "weight_decay": 0.0,
        },
    }


def _v6_cfg() -> dict:
    cfg = _cfg()
    cfg["model"]["iforward"]["version"] = "v6_point_mamba_xcpe"
    cfg["model"]["iforward"]["trainability"].update(
        {
            "train_point_mamba": True,
            "train_local_conflict_xcpe": True,
            "train_context_adapter": True,
        }
    )
    cfg["optimizer"]["lr"].update(
        {
            "point_mamba": 1.1e-4,
            "local_conflict_xcpe": 1.2e-4,
            "context_adapter": 1.3e-4,
        }
    )
    return cfg


def _v3_cfg() -> dict:
    cfg = _cfg()
    cfg["model"]["iforward"]["version"] = "v3_gru_history_gate"
    cfg["model"]["iforward"]["trainability"].update(
        {
            "train_point_gru": True,
            "train_history_gate": True,
        }
    )
    cfg["optimizer"]["lr"].update(
        {
            "point_gru": 1.4e-4,
            "history_gate": 1.5e-4,
        }
    )
    return cfg


def _group(trainer: IForwardTrainer, name: str) -> dict:
    for group in trainer.optimizer.param_groups:
        if group.get("name") == name:
            return group
    raise AssertionError(f"missing optimizer group {name}")


def test_iforward_optimizer_groups_without_warmup_and_measurement_frontend() -> None:
    trainer = IForwardTrainer(config=_cfg(), device=torch.device("cpu"), model=_Model())

    names = {str(group.get("name")) for group in trainer.optimizer.param_groups}
    assert names == {
        "memory",
        "memory_fuse",
        "vsm_ctx_adapter",
        "stage6_posterior_updater_base",
        "stage6_struct_decoder",
        "stage6_measurement_frontend_residual_unet",
        "stage6_measurement_frontend_fusion_neck",
    }

    trainer._apply_trainability_schedule(0)
    assert _group(trainer, "memory")["lr"] == 1.0e-4
    assert _group(trainer, "vsm_ctx_adapter")["lr"] == 2.0e-4
    assert _group(trainer, "stage6_posterior_updater_base")["lr"] == 1.0e-5
    assert any(p.requires_grad for p in _group(trainer, "stage6_posterior_updater_base")["params"])
    assert _group(trainer, "stage6_struct_decoder")["lr"] == 1.0e-5
    assert any(p.requires_grad for p in _group(trainer, "stage6_struct_decoder")["params"])
    assert _group(trainer, "stage6_measurement_frontend_residual_unet")["lr"] == 1.0e-5
    assert any(p.requires_grad for p in _group(trainer, "stage6_measurement_frontend_residual_unet")["params"])
    assert _group(trainer, "stage6_measurement_frontend_fusion_neck")["lr"] == 1.0e-5
    assert any(p.requires_grad for p in _group(trainer, "stage6_measurement_frontend_fusion_neck")["params"])


def test_iforward_v6_optimizer_groups_use_point_xcpe_context_groups() -> None:
    trainer = IForwardTrainer(config=_v6_cfg(), device=torch.device("cpu"), model=_V6Model())

    names = {str(group.get("name")) for group in trainer.optimizer.param_groups}
    assert names == {
        "point_mamba",
        "local_conflict_xcpe",
        "context_adapter",
        "vsm_ctx_adapter",
        "stage6_posterior_updater_base",
        "stage6_struct_decoder",
        "stage6_measurement_frontend_residual_unet",
        "stage6_measurement_frontend_fusion_neck",
    }

    trainer._apply_trainability_schedule(0)
    assert _group(trainer, "point_mamba")["lr"] == 1.1e-4
    assert _group(trainer, "local_conflict_xcpe")["lr"] == 1.2e-4
    assert _group(trainer, "context_adapter")["lr"] == 1.3e-4
    assert any(p.requires_grad for p in _group(trainer, "point_mamba")["params"])
    assert any(p.requires_grad for p in _group(trainer, "local_conflict_xcpe")["params"])
    assert any(p.requires_grad for p in _group(trainer, "context_adapter")["params"])


def test_iforward_v3_optimizer_groups_use_gru_history_gate_groups() -> None:
    trainer = IForwardTrainer(config=_v3_cfg(), device=torch.device("cpu"), model=_V3Model())

    names = {str(group.get("name")) for group in trainer.optimizer.param_groups}
    assert names == {
        "point_gru",
        "history_gate",
        "vsm_ctx_adapter",
        "stage6_posterior_updater_base",
        "stage6_struct_decoder",
        "stage6_measurement_frontend_residual_unet",
        "stage6_measurement_frontend_fusion_neck",
    }

    trainer._apply_trainability_schedule(0)
    assert _group(trainer, "point_gru")["lr"] == 1.4e-4
    assert _group(trainer, "history_gate")["lr"] == 1.5e-4
    assert any(p.requires_grad for p in _group(trainer, "point_gru")["params"])
    assert any(p.requires_grad for p in _group(trainer, "history_gate")["params"])
