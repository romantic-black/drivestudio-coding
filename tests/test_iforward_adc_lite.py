from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from models.iforward import IForwardBatchResolver, IForwardModel, IForwardShortWindowHistory, IForwardState
from models.iforward.adc_lite import (
    IForwardADCBank,
    _deterministic_jitter,
    apply_bg_clone_episode_local,
    build_adc_lite_bank_from_losses,
)
from models.iforward.gru_memory import IForwardGRUMemoryState
from models.iforward.history_ema import IForwardHistoryEMAState
from models.iforward.history_gradient_bank import build_history_gradient_bank_from_loss
from models.streetforward.node_states import NodeStateBackground
from models.streetforward.stage6_0 import LocalGSState

from test_iforward_v3_rollout import FakeV3Bridge, _batch_b1r2_with_history_probe_target, _v3_cfg


def _node_state(n: int = 3) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _local_state(n: int = 3) -> LocalGSState:
    return LocalGSState.from_node_states(bg=_node_state(n), distant=None, rigid=None, hidden_dim=2)


def _adc_cfg(*, max_new: int = 2, max_episode: int = 4, max_total: int = 8) -> Dict[str, Any]:
    return {
        "enable": True,
        "require_history_for_clone": True,
        "bank": {"dtype": "fp32"},
        "score": {
            "weights": {
                "abs_grad_current": 1.0,
                "abs_grad_history": 0.5,
                "scale_or_screen_radius": 0.5,
                "current_history_conflict": 1.0,
            },
            "grad_attr_weights": {"means": 1.0, "scales": 0.0, "opacity": 0.0, "sh": 0.0},
            "normalize": {"percentile": 100.0, "eps": 1.0e-8},
            "scale_proxy": {"percentile": 100.0},
            "conflict": {"eps": 1.0e-8},
        },
        "candidate": {
            "exclude_clones_as_parent": True,
            "alpha_min": 0.005,
            "scale_min": 1.0e-4,
            "min_score": 0.0,
            "require_history": True,
        },
        "budget": {
            "max_new_points_per_rollout": int(max_new),
            "max_new_points_per_episode": int(max_episode),
            "max_total_bg_points_episode": int(max_total),
            "cooldown_rollouts": 2,
        },
        "clone": {
            "opacity_split": "alpha_preserving",
            "mean_jitter_std_scale": 0.0,
            "local_hidden_init": "parent",
        },
    }


def test_adc_lite_bank_scores_current_history_conflict() -> None:
    local = _local_state(3)
    weights = torch.tensor([0.1, 1.0, 2.0])
    loss_current = (local.bg.means[:, 0] * weights).sum()
    loss_history = -(local.bg.means[:, 0] * weights).sum()

    bank = build_adc_lite_bank_from_losses(
        loss_current=loss_current,
        loss_history=loss_history,
        final_local_state=local,
        cfg=_adc_cfg(),
        rollout_id=4,
        episode_id=7,
        num_current_refs=1,
        num_history_refs=1,
    )

    assert bank is not None
    assert bank.valid is True
    assert bank.score.shape == (3,)
    assert bank.candidate_mask.tolist() == [True, True, True]
    assert bank.conflict_score[2] > bank.conflict_score[0]
    assert bank.score[2] > bank.score[0]
    assert bank.score_topk_mean.item() > 0.0


def test_adc_lite_conflict_score_uses_max_grad_gate_not_sqrt_product() -> None:
    local = _local_state(2)
    loss_current = local.bg.means[:, 0].sum()
    loss_history = local.bg.means[:, 0].mul(torch.tensor([-0.01, 1.0])).sum()
    cfg = _adc_cfg(max_new=2)
    cfg["score"]["conflict"]["mode"] = "relu_neg_cos_max_grad"

    bank = build_adc_lite_bank_from_losses(
        loss_current=loss_current,
        loss_history=loss_history,
        final_local_state=local,
        cfg=cfg,
        rollout_id=4,
        episode_id=7,
        num_current_refs=1,
        num_history_refs=1,
    )

    assert bank is not None
    assert bank.conflict_score[0].item() == pytest.approx(1.0)
    assert bank.conflict_score[1].item() == pytest.approx(0.0)


def test_adc_lite_apply_clone_extends_episode_local_state() -> None:
    local = _local_state(3)
    memory = IForwardGRUMemoryState.from_local_state(local, hidden_dim=5)
    history_ema = IForwardHistoryEMAState.from_local_state(local)
    hgv2_bank = build_history_gradient_bank_from_loss(
        loss_history=(local.bg.means[:, 0] * torch.tensor([1.0, 2.0, 3.0])).sum(),
        final_local_state=local,
        rollout_id=0,
        history_num_refs=1,
        cfg={"bank": {"dtype": "fp32", "min_grad_norm": 1.0e-8}},
    )
    bank = IForwardADCBank(
        valid=True,
        source_rollout_id=0,
        source_episode_id=3,
        source_num_current_refs=1,
        source_num_history_refs=1,
        score=torch.tensor([0.1, 3.0, 2.0]),
        abs_grad_current=torch.tensor([0.1, 1.0, 0.9]),
        abs_grad_history=torch.tensor([0.1, 1.0, 0.9]),
        scale_score=torch.ones(3),
        conflict_score=torch.tensor([0.0, 0.8, 0.7]),
        candidate_mask=torch.tensor([True, True, True]),
        score_topk_mean=torch.tensor(2.5),
        score_p90=torch.tensor(2.8),
        score_p99=torch.tensor(2.98),
    )
    state = IForwardState(
        local_gs=local,
        memory=memory,
        history=IForwardShortWindowHistory.empty(),
        scene_id=1,
        segment_id=2,
        episode_id=3,
        history_ema=history_ema,
        history_gradient_bank=hgv2_bank,
        adc_bank=bank,
    )

    state, stats = apply_bg_clone_episode_local(
        state=state,
        cfg=_adc_cfg(max_new=2, max_episode=2, max_total=5),
        rollout_id=1,
        device=torch.device("cpu"),
    )

    assert stats["adc_lite/applied"] == pytest.approx(1.0)
    assert stats["adc_lite/num_cloned_this_rollout"] == pytest.approx(2.0)
    assert state.adc_bank is None
    assert state.local_gs.bg.means.shape[0] == 5
    assert state.memory.bg.h.shape[0] == 5
    assert not bool(state.memory.bg.seen[-2:].any().item())
    assert state.history_ema is not None
    assert state.history_ema.bg.initialized.shape[0] == 5
    assert state.history_gradient_bank is not None
    assert state.history_gradient_bank.bg.means.direction.shape[0] == 5
    assert not bool(state.history_gradient_bank.bg.means.valid[-2:].any().item())
    assert state.adc_meta is not None
    assert state.adc_meta.original_bg_count == 3
    assert state.adc_meta.num_bg_clones_created_episode == 2
    assert state.adc_meta.parent_index.tolist() == [-1, -1, -1, 1, 2]

    alpha_each = torch.sigmoid(state.local_gs.bg.opacity_logit[1])
    combined = 1.0 - (1.0 - alpha_each) * (1.0 - alpha_each)
    assert combined.item() == pytest.approx(0.5, abs=1.0e-5)
    assert torch.allclose(state.local_gs.bg.means[3], state.local_gs.bg.means[1])


def test_adc_clone_near_aabb_boundary_stays_inside() -> None:
    n = 8
    local = _local_state(n)
    aabb_min = torch.tensor([-1.0, -1.0, -1.0])
    aabb_max = torch.tensor([1.0, 1.0, 1.0])
    parent_idx = torch.arange(n)
    jitter = _deterministic_jitter(parent_idx, 3, local.bg.scales_log.detach(), 1.0)
    boundary_means = torch.where(jitter >= 0.0, aabb_max - 1.0e-4, aabb_min + 1.0e-4)
    local.bg.means = boundary_means.detach().clone().requires_grad_(True)
    memory = IForwardGRUMemoryState.from_local_state(local, hidden_dim=5)
    history_ema = IForwardHistoryEMAState.from_local_state(local)
    bank = IForwardADCBank(
        valid=True,
        source_rollout_id=2,
        source_episode_id=3,
        source_num_current_refs=1,
        source_num_history_refs=1,
        score=torch.arange(float(n), 0.0, -1.0),
        abs_grad_current=torch.ones(n),
        abs_grad_history=torch.ones(n),
        scale_score=torch.ones(n),
        conflict_score=torch.zeros(n),
        candidate_mask=torch.ones(n, dtype=torch.bool),
        score_topk_mean=torch.tensor(1.0),
        score_p90=torch.tensor(1.0),
        score_p99=torch.tensor(1.0),
    )
    state = IForwardState(
        local_gs=local,
        memory=memory,
        history=IForwardShortWindowHistory.empty(),
        scene_id=1,
        segment_id=2,
        episode_id=3,
        history_ema=history_ema,
        adc_bank=bank,
    )
    cfg = _adc_cfg(max_new=n, max_episode=n, max_total=2 * n)
    cfg["clone"]["mean_jitter_std_scale"] = 1.0
    cfg["clone"]["aabb_eps"] = 1.0e-5

    state, stats = apply_bg_clone_episode_local(
        state=state,
        cfg=cfg,
        rollout_id=3,
        device=torch.device("cpu"),
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        voxel_size=0.25,
    )

    child = state.local_gs.bg.means[n:]
    assert bool((child >= (aabb_min + 1.0e-5 - 1.0e-7)).all().item())
    assert bool((child <= (aabb_max - 1.0e-5 + 1.0e-7)).all().item())
    assert stats["adc_lite/child_oob_before_clamp_ratio"] > 0.0
    assert stats["adc_lite/child_clamped_ratio"] > 0.0
    assert "adc_lite/child_same_voxel_parent_ratio" in stats


def test_adc_planning_support_filters_candidate_parents() -> None:
    local = _local_state(3)
    memory = IForwardGRUMemoryState.from_local_state(local, hidden_dim=5)
    bank = IForwardADCBank(
        valid=True,
        source_rollout_id=0,
        source_episode_id=3,
        source_num_current_refs=1,
        source_num_history_refs=1,
        score=torch.tensor([10.0, 1.0, 0.5]),
        abs_grad_current=torch.ones(3),
        abs_grad_history=torch.ones(3),
        scale_score=torch.ones(3),
        conflict_score=torch.zeros(3),
        candidate_mask=torch.tensor([True, True, True]),
        score_topk_mean=torch.tensor(10.0),
        score_p90=torch.tensor(10.0),
        score_p99=torch.tensor(10.0),
    )
    state = IForwardState(
        local_gs=local,
        memory=memory,
        history=IForwardShortWindowHistory.empty(),
        scene_id=1,
        segment_id=2,
        episode_id=3,
        adc_bank=bank,
    )
    cfg = _adc_cfg(max_new=1, max_episode=1, max_total=4)
    cfg["planning"] = {
        "enable": True,
        "require_visible": True,
        "min_support": 0.0,
        "support_score_weight": 0.0,
    }

    state, stats = apply_bg_clone_episode_local(
        state=state,
        cfg=cfg,
        rollout_id=1,
        device=torch.device("cpu"),
        planning_support_bg=torch.tensor([0.0, 2.0, 0.0]),
        planning_valid_bg=torch.tensor([False, True, False]),
    )

    assert stats["adc_lite/planning/applied"] == pytest.approx(1.0)
    assert stats["adc_lite/num_cloned_this_rollout"] == pytest.approx(1.0)
    assert state.adc_meta is not None
    assert state.adc_meta.parent_index.tolist() == [-1, -1, -1, 1]


def _adc_model_cfg() -> Dict[str, Any]:
    cfg = _v3_cfg()
    cfg["model"]["iforward"]["adc_lite"] = _adc_cfg(max_new=1, max_episode=2, max_total=4)
    cfg["model"]["iforward"]["loss"] = {
        "current": {"weight": 1.0},
        "nearby": {"weight": 0.0},
        "in_rollout_history": {"weight": 1.0},
        "short_window_history": {"weight": 0.0},
        "delta_regularization": {"weight": 0.0},
    }
    return cfg


def _non_reset_next_rollout(batch: Dict[str, Any], rollout_id: int) -> Dict[str, Any]:
    out = dict(batch)
    ifwd = dict(batch["_iforward"])
    ifwd["reset_scene_state_before_rollout"] = False
    ifwd["rollout_id_global"] = int(rollout_id)
    ifwd["rollout_idx_in_episode"] = int(rollout_id)
    request_meta = dict(batch["request_meta"])
    request_meta["iforward"] = ifwd
    out["_iforward"] = ifwd
    out["request_meta"] = request_meta
    return out


def test_iforward_adc_lite_two_rollout_bank_create_consume() -> None:
    bridge = FakeV3Bridge()
    model = IForwardModel(
        config=_adc_model_cfg(),
        device=torch.device("cpu"),
        bridge=bridge,
        resolver=IForwardBatchResolver(),
    )

    out0 = model.forward_rollout(_batch_b1r2_with_history_probe_target())
    assert out0.next_state.adc_bank is not None
    assert out0.stats["adc_lite/bank_valid"] == pytest.approx(0.0)
    assert out0.stats["adc_lite/next_bank_valid"] == pytest.approx(1.0)

    out1 = model.forward_rollout(
        _non_reset_next_rollout(_batch_b1r2_with_history_probe_target(), 1),
        carried_state=out0.next_state.detach_for_next_rollout(),
    )

    assert out1.stats["adc_lite/bank_valid"] == pytest.approx(1.0)
    assert out1.stats["adc_lite/applied"] == pytest.approx(1.0)
    assert out1.stats["adc_lite/num_cloned_this_rollout"] == pytest.approx(1.0)
    assert out1.stats["adc_lite/bg_count_before"] == pytest.approx(2.0)
    assert out1.stats["adc_lite/bg_count_after"] == pytest.approx(3.0)
    assert out1.next_state.local_gs.bg.means.shape[0] == 3
    assert out1.next_state.adc_meta is not None
    assert out1.next_state.adc_meta.num_bg_clones_created_episode == 1
    assert out1.next_state.adc_bank is not None
    assert out1.next_state.adc_bank.source_rollout_id == 1


def test_iforward_adc_lite_rollout_start_planning_pass_runs() -> None:
    bridge = FakeV3Bridge()
    cfg = _adc_model_cfg()
    cfg["model"]["iforward"]["adc_lite"]["planning"] = {
        "enable": True,
        "scope": "first_step",
        "max_steps_per_rollout": 1,
        "require_visible": True,
        "min_support": 0.0,
        "support_score_weight": 0.0,
    }
    model = IForwardModel(
        config=cfg,
        device=torch.device("cpu"),
        bridge=bridge,
        resolver=IForwardBatchResolver(),
    )

    out0 = model.forward_rollout(_batch_b1r2_with_history_probe_target())
    out1 = model.forward_rollout(
        _non_reset_next_rollout(_batch_b1r2_with_history_probe_target(), 1),
        carried_state=out0.next_state.detach_for_next_rollout(),
    )

    assert out1.stats["adc_lite/planning/pass_enabled"] == pytest.approx(1.0)
    assert out1.stats["adc_lite/planning/pass_ran"] == pytest.approx(1.0)
    assert out1.stats["adc_lite/planning/applied"] == pytest.approx(1.0)
