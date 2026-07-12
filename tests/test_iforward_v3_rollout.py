from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.nn as nn

from models.iforward import IForwardBatchResolver, IForwardModel
from models.iforward.history_ema import IForwardResidualPack
from models.iforward.random_window_resolver import IForwardRandomWindowBatchResolver
from models.iforward.validation import DEFAULT_IFORWARD_V3_ABLATIONS, validate_iforward_memory_ablation
from models.streetforward.node_states import NodeStateBackground
from models.streetforward.stage6_0 import EventPack, LocalGSState
from models.streetforward.stage6_0.posterior_updater import BranchDelta, ContextPack, DeltaPack

from test_iforward_rollout import _random_window_batch


def _node_state(n: int = 2) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _branch_delta(local_state: LocalGSState, means: torch.Tensor) -> DeltaPack:
    n = int(local_state.bg.means.shape[0])
    ref = local_state.bg.means
    zeros3 = ref.new_zeros(n, 3)
    return DeltaPack(
        bg=BranchDelta(
            means=means,
            scales_log=zeros3,
            quat_axis_angle=zeros3,
            opacity_logit=ref.new_zeros(n, 1),
            sh=ref.new_zeros(n, 12),
            hidden=ref.new_zeros(n, int(local_state.bg.hidden.shape[1])),
            confidence=ref.new_ones(n, 1),
            noop=ref.new_zeros(n, 1),
        )
    )


class FakeV3Bridge(nn.Module):
    event_dim = 4
    hidden_dim = 2
    current_mask_policy = "none"
    nearby_mask_policy = "none"

    def __init__(self) -> None:
        super().__init__()
        self.delta_scale = nn.Parameter(torch.tensor(0.05))
        self.observe_calls: List[Tuple[int, Tuple[int, ...]]] = []
        self.residual_calls: List[Tuple[int, Tuple[int, ...]]] = []
        self.bg = _node_state()

    def make_local_state(self, *, batch: Dict[str, Any]):
        _ = batch
        local = LocalGSState.from_node_states(bg=self.bg, distant=None, rigid=None, hidden_dim=2)
        return local, self.bg, None, None

    def observe(self, *, local_state, batch, source_indices, source_frame_idx):
        _ = local_state, batch
        self.observe_calls.append((int(source_frame_idx), tuple(int(x) for x in source_indices)))
        return {"source_frame_idx": int(source_frame_idx)}

    def build_event(self, *, local_state, measurement):
        frame = float(measurement["source_frame_idx"]) / 100.0
        n = int(local_state.bg.means.shape[0])
        frame_col = local_state.bg.means.new_full((n, 1), frame)
        event_bg = torch.cat([local_state.bg.means + frame, frame_col], dim=-1)
        return EventPack(
            event_bg=event_bg,
            support_bg=local_state.bg.means.new_ones((n, 1)),
            valid_bg=torch.ones((n, 1), device=local_state.bg.means.device, dtype=torch.bool),
            obs_code_bg=local_state.bg.means.new_zeros((n, 2)),
        )

    def predict_delta(self, *, local_state, event, ctx_memory: ContextPack):
        ctx = ctx_memory.ctx_bg if ctx_memory is not None else torch.zeros_like(event.event_bg)
        means = self.delta_scale * torch.tanh(event.event_bg[:, :3] + 0.01 * ctx[:, :3])
        return _branch_delta(local_state, means), {"fake_predict": 1.0}

    def apply_branch_scope_event_rows(self, delta):
        return delta

    def expand_rigid_delta(self, *, delta, event, local_state):
        _ = event, local_state
        return delta

    def apply_delta_only(self, *, local_state, delta):
        return local_state.apply_delta(delta)

    def compute_block_residual_history(self, *, local_state, batch, source_indices, source_frame_idx):
        _ = batch
        self.residual_calls.append((int(source_frame_idx), tuple(int(x) for x in source_indices)))
        n = int(local_state.bg.means.shape[0])
        return IForwardResidualPack(
            error_bg=local_state.bg.means.new_full((n, 1), 0.25),
            support_bg=local_state.bg.means.new_ones((n, 1)),
        )

    def render_loss(self, *, local_state, batch, target_indices, mask_policy, pred_rgbs_out=None, gt_images_out=None):
        _ = batch, mask_policy
        if not target_indices:
            return local_state.bg.means.new_tensor(0.0), {"num_refs": 0.0, "valid_ratio": 0.0}
        loss = local_state.bg.means.pow(2).mean()
        if pred_rgbs_out is not None:
            pred_rgbs_out.append(local_state.bg.means.new_zeros(1, 1, 3))
        if gt_images_out is not None:
            gt_images_out.append(local_state.bg.means.new_zeros(1, 1, 3))
        return loss, {
            "num_refs": float(len(target_indices)),
            "num_metric_refs": float(len(target_indices)),
            "metric_valid": 1.0,
            "valid_ratio": 1.0,
            "psnr": 10.0,
            "l1": float(loss.detach().item()),
            "ssim": 0.0,
        }

    def render_loss_for_targets(self, *, local_state, ref_batch, targets, mask_policy, pred_rgbs_out=None, gt_images_out=None):
        _ = ref_batch, mask_policy
        if not targets:
            return local_state.bg.means.new_tensor(0.0), {"num_refs": 0.0, "valid_ratio": 0.0}
        if pred_rgbs_out is not None:
            pred_rgbs_out.append(local_state.bg.means.new_zeros(1, 1, 3))
        if gt_images_out is not None:
            gt_images_out.append(local_state.bg.means.new_zeros(1, 1, 3))
        return local_state.bg.means.pow(2).mean(), {"num_refs": float(len(targets)), "valid_ratio": 1.0}

    def delta_regularization(self, delta, *, local_state):
        _ = local_state
        return delta.bg.means.pow(2).mean() * 0.0, {"delta_l2": 0.0}


class FakeHSPBridge(FakeV3Bridge):
    def __init__(self) -> None:
        super().__init__()
        self.hsp_probe_calls = 0

    def predict_delta(self, *, local_state, event, ctx_memory: ContextPack):
        delta, aux = super().predict_delta(local_state=local_state, event=event, ctx_memory=ctx_memory)
        aux["fake_raw_means_norm"] = float(delta.bg.means.detach().norm(dim=-1).mean().item())
        return delta, aux

    def history_probe_loss(self, *, local_state, batch, target_indices, mask_policy):
        _ = batch, target_indices, mask_policy
        self.hsp_probe_calls += 1
        return local_state.bg.means[:, 0].sum(), {"num_refs": float(len(target_indices)), "psnr": 9.0}


def _v3_cfg() -> Dict[str, Any]:
    return {
        "model": {
            "iforward": {
                "version": "v3_gru_history_gate",
                "point_gru": {"hidden_dim": 5, "ctx_dim": 4},
                "history_gate": {"hidden_dim": 8, "history_embed_dim": 4},
                "short_window_history": {"max_entries": 24, "max_memory_entries": 0},
            }
        },
        "losses": {"phase_a": {"nearby_render": {"weight": 0.0}}},
    }


def _v3_hsp_cfg(*, damage_loss_enable: bool = True) -> Dict[str, Any]:
    cfg = _v3_cfg()
    cfg["model"]["iforward"]["history_safe_projection"] = {
        "enable": True,
        "mode": "project_delta",
        "probe": {"frequency": "block_enter", "reuse_within_block": True, "frames_per_block": 1, "cams_per_frame": 1},
        "attrs": {"means": True, "scales": False, "opacity": False, "sh": False, "quat": False},
        "projection": {
            "strength": {"start_step": 0, "warmup_steps": 0, "start_value": 1.0, "end_value": 1.0},
            "attr_strength_scale": {"means": 1.0},
            "tau_norm": {"means": 0.0},
        },
        "damage_loss": {
            "enable": bool(damage_loss_enable),
            "type": "cosine_conflict",
            "weight": 0.05,
            "attr_weights": {"means": 1.0},
        },
    }
    cfg["model"]["iforward"]["loss"] = {
        "current": {"weight": 0.0},
        "nearby": {"weight": 0.0},
        "in_rollout_history": {"weight": 0.0},
        "short_window_history": {"weight": 0.0},
        "delta_regularization": {"weight": 0.0},
    }
    return cfg


def _batch_b1r2() -> Dict[str, Any]:
    source_refs = [(10, 0), (10, 1), (10, 2)]
    target_refs = [(10, 0), (10, 1), (10, 2)]
    target_roles = ["final_current_recon"] * 3
    steps = [
        {
            "step_idx": 0,
            "block_id": 0,
            "episode_block_idx": 0,
            "source_frame_idx": 10,
            "repeat_idx": 0,
            "repeats_per_block": 2,
            "rollout_block_rank": 0,
            "is_block_enter": True,
            "is_block_exit": False,
            "evidence_refs": source_refs,
            "source_indices": [0, 1, 2],
            "commit_observation_memory": True,
            "update_optimizer_memory": True,
        },
        {
            "step_idx": 1,
            "block_id": 0,
            "episode_block_idx": 0,
            "source_frame_idx": 10,
            "repeat_idx": 1,
            "repeats_per_block": 2,
            "rollout_block_rank": 0,
            "is_block_enter": False,
            "is_block_exit": True,
            "evidence_refs": source_refs,
            "source_indices": [0, 1, 2],
            "commit_observation_memory": False,
            "update_optimizer_memory": True,
        },
    ]
    ifwd = {
        "scheduler_version": "iforward_v1",
        "model_family": "IForward",
        "scene_id": 1,
        "segment_id": 2,
        "episode_id": 3,
        "rollout_id_global": 0,
        "rollout_idx_in_episode": 0,
        "inner_K": 2,
        "steps": steps,
        "input_frame_indices": [10],
        "evidence_refs_flat": source_refs,
        "target_refs_flat": target_refs,
        "target_roles_flat": target_roles,
        "reset_scene_state_before_rollout": True,
        "carry_scene_state_after_rollout": True,
        "episode_end_after_rollout": False,
        "detach_graph_after_rollout": True,
    }
    return {
        "scene_id": 1,
        "segment_id": 2,
        "request_meta": {
            "scheduler_version": "iforward_v1",
            "model_family": "IForward",
            "assembly_mode": "image_ref_iforward_v1",
            "source_image_refs": source_refs,
            "target_image_refs": target_refs,
            "target_image_roles": target_roles,
            "iforward": ifwd,
        },
        "_iforward": ifwd,
        "targets": [{"gt_image": torch.zeros(1, 1, 3), "frame_idx": f, "cam_idx": c} for f, c in target_refs],
    }


def _batch_b1r2_with_history_probe_target() -> Dict[str, Any]:
    batch = _batch_b1r2()
    target_refs = [(5, 0), (10, 0), (10, 1), (10, 2)]
    target_roles = ["final_current_recon"] * len(target_refs)
    ifwd = dict(batch["_iforward"])
    ifwd.update(
        {
            "target_refs_flat": target_refs,
            "target_roles_flat": target_roles,
            "input_frame_indices": [5, 10],
        }
    )
    request_meta = dict(batch["request_meta"])
    request_meta.update(
        {
            "target_image_refs": target_refs,
            "target_image_roles": target_roles,
            "iforward": ifwd,
        }
    )
    out = dict(batch)
    out["_iforward"] = ifwd
    out["request_meta"] = request_meta
    out["targets"] = [{"gt_image": torch.zeros(1, 1, 3), "frame_idx": f, "cam_idx": c} for f, c in target_refs]
    return out


def _batch_b1r2_history_flags_disabled() -> Dict[str, Any]:
    batch = _batch_b1r2()
    ifwd = dict(batch["_iforward"])
    steps = [dict(step) for step in ifwd["steps"]]
    for step in steps:
        step["record_update_norm"] = False
        step["commit_support_on_exit"] = False
        step["commit_residual_on_exit"] = False
    ifwd["steps"] = steps
    request_meta = dict(batch["request_meta"])
    request_meta["iforward"] = ifwd
    out = dict(batch)
    out["_iforward"] = ifwd
    out["request_meta"] = request_meta
    return out


def _batch_same_source_two_blocks() -> Dict[str, Any]:
    batch = _batch_b1r2()
    ifwd = dict(batch["_iforward"])
    steps = []
    for step_idx, (block_id, repeat_idx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        steps.append(
            {
                **batch["_iforward"]["steps"][repeat_idx],
                "step_idx": step_idx,
                "block_id": block_id,
                "episode_block_idx": block_id,
                "rollout_block_rank": block_id,
                "repeat_idx": repeat_idx,
                "repeats_per_block": 2,
                "is_block_enter": repeat_idx == 0,
                "is_block_exit": repeat_idx == 1,
                "commit_observation_memory": repeat_idx == 0,
            }
        )
    ifwd.update(
        {
            "steps": steps,
            "inner_K": 4,
            "input_frame_indices": [10],
            "delivery_frame_indices": [10, 10],
        }
    )
    request_meta = dict(batch["request_meta"])
    request_meta["iforward"] = ifwd
    out = dict(batch)
    out["_iforward"] = ifwd
    out["request_meta"] = request_meta
    return out


def test_iforward_v3_rollout_b1r2_history_stats_and_detach() -> None:
    bridge = FakeV3Bridge()
    model = IForwardModel(config=_v3_cfg(), device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    out = model.forward_rollout(_batch_b1r2())

    assert torch.isfinite(out.loss)
    assert bridge.residual_calls == [(10, (0, 1, 2))]
    assert out.resolved.steps[0].block_id == 0
    assert out.resolved.steps[0].is_block_enter is True
    assert out.resolved.steps[1].is_block_exit is True
    assert out.next_state.memory.count_tokens()["bg_point_seen"] == 2.0
    assert out.next_state.history_ema is not None
    assert out.next_state.history_ema.bg.support_fast.mean().item() > 0.0
    assert out.next_state.history_ema.bg.error_fast.mean().item() > 0.0
    assert out.per_step[0]["v3/history/bg_support_snapshot_rows"] == 2.0
    assert "v3/history/bg_support_snapshot_rows" not in out.per_step[1]
    assert "v3/history/bg/support_fast_mean" in out.stats

    carried = out.next_state.detach_for_next_rollout()
    assert carried.history_ema is not None
    assert carried.history_ema.bg.support_fast.requires_grad is False
    assert carried.memory.bg.h.requires_grad is False


def test_iforward_v3_rollout_uses_explicit_block_exit_for_same_source_revisit() -> None:
    bridge = FakeV3Bridge()
    model = IForwardModel(config=_v3_cfg(), device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    out = model.forward_rollout(_batch_same_source_two_blocks())

    assert torch.isfinite(out.loss)
    assert bridge.residual_calls == [(10, (0, 1, 2)), (10, (0, 1, 2))]
    assert [bool(step.is_block_exit) for step in out.resolved.steps] == [False, True, False, True]
    support_snapshot_rows = sum(float(item.get("v3/history/bg_support_snapshot_rows", 0.0)) for item in out.per_step)
    assert support_snapshot_rows == 4.0


def test_iforward_v3_rollout_obeys_scheduler_history_commit_flags() -> None:
    bridge = FakeV3Bridge()
    model = IForwardModel(config=_v3_cfg(), device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    out = model.forward_rollout(_batch_b1r2_history_flags_disabled())

    assert torch.isfinite(out.loss)
    assert bridge.residual_calls == []
    assert out.next_state.history_ema is not None
    assert out.next_state.history_ema.bg.support_fast.mean().item() == 0.0
    assert out.next_state.history_ema.bg.error_fast.mean().item() == 0.0
    assert out.next_state.history_ema.bg.update_norm_fast.mean().item() == 0.0
    assert out.next_state.history_ema.bg.block_present_count.sum().item() == 2.0
    assert out.stats["v3/history/bg/pending_present_rows"] == 2.0


def test_iforward_v3_history_loss_weight_warmup() -> None:
    cfg = _v3_cfg()
    cfg["model"]["iforward"]["loss"] = {
        "current": {"weight": 1.0},
        "nearby": {"weight": 0.0},
        "in_rollout_history": {
            "weight": 1.0,
            "warmup": {"enable": True, "start_step": 0, "steps": 100, "start_factor": 0.25},
        },
        "short_window_history": {"weight": 0.0},
        "delta_regularization": {"weight": 0.0},
    }
    bridge = FakeV3Bridge()
    model = IForwardModel(config=cfg, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    batch = _batch_b1r2()

    batch["global_step"] = 0
    out0 = model.forward_rollout(batch)
    assert out0.stats["loss_weight/in_rollout_history"] == pytest.approx(0.25)
    assert out0.stats["loss_weight/in_rollout_history_warmup_factor"] == pytest.approx(0.25)

    batch["global_step"] = 50
    out_mid = model.forward_rollout(batch)
    assert out_mid.stats["loss_weight/in_rollout_history"] == pytest.approx(0.625)

    batch["global_step"] = 100
    out_end = model.forward_rollout(batch)
    assert out_end.stats["loss_weight/in_rollout_history"] == pytest.approx(1.0)


def test_iforward_v3_hsp_projects_delta_and_reuses_block_probe_cache() -> None:
    bridge = FakeHSPBridge()
    model = IForwardModel(config=_v3_hsp_cfg(), device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())

    out = model.forward_rollout(_batch_b1r2_with_history_probe_target(), ablation="no_history_gate")

    assert torch.isfinite(out.loss)
    assert bridge.hsp_probe_calls == 1
    assert out.per_step[0]["hsp/enabled"] == pytest.approx(1.0)
    assert out.per_step[0]["hsp/cache_hit"] == pytest.approx(0.0)
    assert out.per_step[1]["hsp/cache_hit"] == pytest.approx(1.0)
    assert out.stats["loss_weight/hsp_damage"] == pytest.approx(0.05)
    assert out.stats["hsp/damage_loss"] > 0.0
    assert out.stats["hsp/cos_damage_loss"] > 0.0
    assert out.stats["hsp/probe_num_refs"] == pytest.approx(1.0)
    assert out.next_state.local_gs.bg.means[:, 0].detach().abs().max().item() < 1.0e-5
    assert out.per_step[0]["v3/gru/bg_delta_means_norm_mean"] < out.per_step[0]["fake_raw_means_norm"]
    assert torch.allclose(out.loss, out.losses["hsp_damage_loss"] * 0.05)


def test_iforward_v3_hsp_damage_loss_enable_false_keeps_probe_but_removes_total_loss() -> None:
    bridge = FakeHSPBridge()
    model = IForwardModel(
        config=_v3_hsp_cfg(damage_loss_enable=False),
        device=torch.device("cpu"),
        bridge=bridge,
        resolver=IForwardBatchResolver(),
    )

    out = model.forward_rollout(_batch_b1r2_with_history_probe_target(), ablation="no_history_gate")

    assert torch.isfinite(out.loss)
    assert bridge.hsp_probe_calls == 1
    assert out.stats["loss_weight/hsp_damage"] == pytest.approx(0.0)
    assert out.stats["hsp/cos_damage_loss"] > 0.0
    assert out.stats["hsp/damage_loss"] == pytest.approx(0.0)
    assert out.losses["hsp_damage_loss"].item() == pytest.approx(0.0)
    assert out.loss.item() == pytest.approx(0.0)


def test_iforward_v3_guard_requires_v3_scheduler_version_when_scheduler_enabled() -> None:
    bad_cfg = _v3_cfg()
    bad_cfg["scheduler_iforward"] = {
        "enable": True,
        "rollout": {
            "block_selection_policy": "random_start_contiguous",
            "delivery_order_policy": "chronological",
        },
        "memory": {"carry_policy": "across_rollouts_until_episode_end"},
    }
    with pytest.raises(ValueError, match="iforward_v3_random_window"):
        IForwardModel(config=bad_cfg, device=torch.device("cpu"), bridge=FakeV3Bridge(), resolver=IForwardBatchResolver())

    good_cfg = _v3_cfg()
    good_cfg["scheduler_iforward"] = {
        "enable": True,
        "version": "iforward_v3_random_window",
        "rollout": {
            "window_policy": "random_with_replacement",
            "delivery_order_policy": "chronological_inside_window",
        },
        "memory": {"carry_policy": "across_rollouts_until_episode_end"},
    }
    IForwardModel(config=good_cfg, device=torch.device("cpu"), bridge=FakeV3Bridge(), resolver=IForwardBatchResolver())


def test_iforward_v3_rollout_random_window_resolver_commits_on_block_exit() -> None:
    bridge = FakeV3Bridge()
    model = IForwardModel(
        config=_v3_cfg(),
        device=torch.device("cpu"),
        bridge=bridge,
        resolver=IForwardRandomWindowBatchResolver(expected_cams_per_step=3),
    )
    out = model.forward_rollout(_random_window_batch(rollout_idx=0, frames=(10, 20, 30, 40)))

    assert torch.isfinite(out.loss)
    assert out.stats["scheduler_version"] == "random_window_v1"
    assert len(bridge.residual_calls) == 4
    support_snapshot_rows = sum(float(item.get("v3/history/bg_support_snapshot_rows", 0.0)) for item in out.per_step)
    assert support_snapshot_rows == 8.0
    assert out.next_state.history_ema.bg.initialized.mean().item() == 1.0


def test_iforward_v3_validation_defaults_use_v3_ablations() -> None:
    bridge = FakeV3Bridge()
    model = IForwardModel(config=_v3_cfg(), device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    rows = validate_iforward_memory_ablation(model=model, rollout_batches=[_batch_b1r2()])
    assert [row["mode"] for row in rows] == list(DEFAULT_IFORWARD_V3_ABLATIONS)
