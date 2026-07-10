from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.iforward.trainer import IForwardTrainer
from models.iforward.state import IForwardShortWindowHistory, IForwardState
from models.iforward.versions import (
    is_stage3_3_iforward_version,
    is_stage3_lowrank_gdkv_iforward_version,
    is_stage3_optimizer_memory_iforward_version,
)
from models.streetforward.checkpoint_mixin import CheckpointMixin
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0.local_gs_state import LocalGSState
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


def _node(n: int = 3) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _rigid_node(n: int = 3) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.arange(n, dtype=torch.float32).reshape(n, 1).repeat(1, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
        point_ids=torch.zeros(n, 1, dtype=torch.long),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        instances_trans=torch.zeros(1, 1, 3),
        instances_fv=torch.ones(1, 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )


def _delta(ref: torch.Tensor, appearance: torch.Tensor) -> DeltaPack:
    n = int(ref.shape[0])
    return DeltaPack(
        bg=BranchDelta(
            means=ref.new_zeros(n, 3),
            scales_log=ref.new_zeros(n, 3),
            quat_axis_angle=ref.new_zeros(n, 3),
            opacity_logit=ref.new_zeros(n, 1),
            sh=ref.new_zeros(n, 12),
            hidden=ref.new_zeros(n, 0),
            confidence=ref.new_zeros(n, 0),
            noop=ref.new_zeros(n, 1),
            appearance_logvar_delta=appearance,
        )
    )


def test_missing_node_uncertainty_uses_branch_prior_and_stays_fp32() -> None:
    state = LocalGSState.from_node_states(bg=_node(), distant=None, rigid=None, hidden_dim=0)
    assert state.bg.appearance_logvar.dtype == torch.float32
    assert torch.allclose(
        state.bg.appearance_logvar,
        torch.full((3, 1), 2.0 * math.log(0.08)),
    )
    cast = state.to(device=torch.device("cpu"), dtype=torch.bfloat16)
    assert cast.bg.means.dtype == torch.bfloat16
    assert cast.bg.appearance_logvar.dtype == torch.float32
    node, _, _ = cast.to_node_states_detached()
    assert node.appearance_logvar is not None
    assert node.appearance_logvar.dtype == torch.float32
    assert not node.appearance_logvar.requires_grad


def test_uncertainty_update_applies_prior_pull_and_clamp() -> None:
    cfg = {
        "init_sigma": {"bg": 0.10},
        "sigma_min": 0.05,
        "sigma_max": 0.20,
        "prior_pull": 0.25,
    }
    state = LocalGSState.from_node_states(
        bg=_node(1),
        distant=None,
        rigid=None,
        hidden_dim=0,
        uncertainty_state_cfg=cfg,
    )
    old = state.bg.appearance_logvar.clone()
    updated = state.apply_delta(_delta(state.bg.means, torch.full((1, 1), 0.20)), uncertainty_state_cfg=cfg)
    expected = old + 0.20 + 0.25 * (old - old)
    assert torch.allclose(updated.bg.appearance_logvar, expected)
    clamped = state.apply_delta(_delta(state.bg.means, torch.full((1, 1), 100.0)), uncertainty_state_cfg=cfg)
    assert clamped.bg.appearance_logvar.item() == torch.tensor(2.0 * math.log(0.20)).item()


def test_node_detach_clone_preserves_optional_uncertainty() -> None:
    node = _node(2)
    node.appearance_logvar = torch.full((2, 1), -4.0, requires_grad=True)
    clone = node.detach_clone()
    assert torch.equal(clone.appearance_logvar, node.appearance_logvar)
    assert clone.appearance_logvar.data_ptr() != node.appearance_logvar.data_ptr()
    assert not clone.appearance_logvar.requires_grad


def test_old_checkpoint_fills_prior_and_rigid_rows_stay_aligned() -> None:
    class Loader(CheckpointMixin):
        device = torch.device("cpu")
        iforward_uncertainty_state_cfg = {
            "init_sigma": {"bg": 0.07, "distant": 0.11, "rigid": 0.09},
            "sigma_min": 0.01,
            "sigma_max": 0.50,
        }

    node = _node(2)
    legacy = Loader()._node_state_to_dict(node)
    assert "appearance_logvar" not in legacy
    restored = Loader()._node_state_from_dict(legacy)
    assert torch.allclose(restored.appearance_logvar, torch.full((2, 1), 2.0 * math.log(0.07)))

    rigid = _rigid_node(3)
    rigid.appearance_logvar = torch.tensor([[-5.0], [-4.0], [-3.0]])
    state = LocalGSState.from_node_states(bg=_node(1), distant=None, rigid=rigid, hidden_dim=0)
    _, _, routed = state.to_node_states_grad()
    assert routed is not None
    subset = torch.tensor([2, 0])
    assert torch.equal(routed.point_ids[subset], rigid.point_ids[subset])
    assert torch.equal(routed.appearance_logvar[subset], torch.tensor([[-3.0], [-5.0]]))


def test_stage3_3_config_reuses_stage3_2_scheduler_and_lowrank_memory_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs/iforward/iforward_stage3_3_uncertainty_v1.yaml")
    version = str(cfg.model.iforward.version)
    assert is_stage3_3_iforward_version(version)
    assert is_stage3_lowrank_gdkv_iforward_version(version)
    assert is_stage3_optimizer_memory_iforward_version(version)
    assert str(cfg.scheduler_stage3_2.version) == "stage3_2_distributional_episode_v1"
    assert cfg.model.iforward.uncertainty.updater.gate_by_main_noop is False
    assert float(cfg.model.iforward.uncertainty.loss.history_precision_floor) == 0.0
    assert cfg.model.stage6_0.posterior_updater.branch_scope.rigid.update_appearance_logvar is True
    cfg_v2 = OmegaConf.load(root / "configs/iforward/iforward_stage3_3_uncertainty_v2.yaml")
    version_v2 = str(cfg_v2.model.iforward.version)
    assert is_stage3_3_iforward_version(version_v2)
    assert is_stage3_lowrank_gdkv_iforward_version(version_v2)
    assert is_stage3_optimizer_memory_iforward_version(version_v2)
    assert cfg_v2.model.iforward.uncertainty.updater.mode == "state_conditioned_target_v2"
    assert cfg_v2.model.iforward.uncertainty.rasterizer.variance_mode == "aleatoric_only"
    assert float(cfg_v2.model.iforward.uncertainty.rasterizer.background_sigma_for_loss) == 0.0


def test_stage3_3_extra_state_schema_supports_strict_resume() -> None:
    def empty_trainer(version: str = "") -> IForwardTrainer:
        trainer = IForwardTrainer.__new__(IForwardTrainer)
        nn.Module.__init__(trainer)
        trainer.config = {"model": {"iforward": {"version": version, "uncertainty": {"state": {}}}}}
        trainer._state_cache = {}
        return trainer

    source = empty_trainer()
    state_dict = source.state_dict()
    assert state_dict["_extra_state"]["local_gs_state_schema_version"] == 2
    assert state_dict["_extra_state"]["uncertainty_state_version"] == "appearance_logvar_v1"
    target = empty_trainer()
    target.load_state_dict(state_dict, strict=True)
    assert target.get_extra_state()["uncertainty_raster_version"] == "detached_moments_v1"

    source_v2 = empty_trainer("stage3_3_uncertainty_v2")
    state_dict_v2 = source_v2.state_dict()
    target_v2 = empty_trainer("stage3_3_uncertainty_v2")
    target_v2.load_state_dict(state_dict_v2, strict=True)
    assert target_v2.get_extra_state()["uncertainty_updater_version"] == "state_conditioned_target_v2"
    assert target_v2.get_extra_state()["uncertainty_raster_version"] == "detached_moments_aleatoric_loss_v2"


def test_old_carried_state_is_migrated_to_branch_prior() -> None:
    trainer = IForwardTrainer.__new__(IForwardTrainer)
    nn.Module.__init__(trainer)
    trainer.config = {
        "model": {
            "iforward": {
                "uncertainty": {
                    "state": {
                        "init_sigma": {"bg": 0.07, "distant": 0.11, "rigid": 0.09},
                        "sigma_min": 0.01,
                        "sigma_max": 0.50,
                    }
                }
            }
        }
    }
    trainer._state_cache = {}
    local = LocalGSState.from_node_states(bg=_node(2), distant=None, rigid=None, hidden_dim=0)
    delattr(local.bg, "appearance_logvar")
    legacy_state = IForwardState(
        local_gs=local,
        memory=None,
        history=IForwardShortWindowHistory.empty(),
        scene_id=1,
        segment_id=2,
        episode_id=3,
    )
    trainer.set_extra_state({"state_cache": {(1, 2, 3): legacy_state}})
    migrated = trainer._state_cache[(1, 2, 3)].local_gs.bg.appearance_logvar
    assert migrated.dtype == torch.float32
    assert torch.allclose(migrated, torch.full((2, 1), 2.0 * math.log(0.07)))
