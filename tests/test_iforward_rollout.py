from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.iforward import IForwardBatchResolver, IForwardModel, IForwardTrainer
from models.iforward.validation import validate_iforward_memory_ablation
from models.iforward.bridge import IForwardStage6Bridge
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import ContextPack, EventPack, LocalGSState
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


def _node_state(n: int = 2) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _rigid_node_state(frame_ids: List[int], n: int = 2) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
        point_ids=torch.zeros(n, 1, dtype=torch.long),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * len(frame_ids)),
        instances_trans=torch.zeros(len(frame_ids), 1, 3),
        instances_fv=torch.ones(len(frame_ids), 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[int(x) for x in frame_ids],
        cur_frame=0,
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


class FakeIForwardBridge(nn.Module):
    event_dim = 4
    hidden_dim = 2
    current_mask_policy = "none"
    nearby_mask_policy = "none"

    def __init__(self):
        super().__init__()
        self.delta_scale = nn.Parameter(torch.tensor(0.05))
        self.observe_calls: List[Tuple[int, Tuple[int, ...]]] = []
        self.render_calls: List[Tuple[int, ...]] = []
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
        return EventPack(event_bg=event_bg)

    def apply_update(self, *, local_state, event, ctx_memory: Optional[ContextPack]):
        ctx = ctx_memory.ctx_bg if ctx_memory is not None else torch.zeros_like(event.event_bg)
        means = self.delta_scale * torch.tanh(event.event_bg[:, :3] + 0.01 * ctx[:, :3])
        delta = _branch_delta(local_state, means)
        return local_state.apply_delta(delta), delta, {"fake_update": 1.0}

    def render_loss(self, *, local_state, batch, target_indices, mask_policy, pred_rgbs_out=None, gt_images_out=None):
        _ = batch, mask_policy
        self.render_calls.append(tuple(int(x) for x in target_indices))
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

    def render_loss_for_targets(self, *, local_state, ref_batch, targets, mask_policy):
        _ = ref_batch, mask_policy
        self.render_calls.append(tuple(range(len(targets))))
        if not targets:
            return local_state.bg.means.new_tensor(0.0), {"num_refs": 0.0, "valid_ratio": 0.0}
        return local_state.bg.means.pow(2).mean(), {"num_refs": float(len(targets)), "valid_ratio": 1.0}

    def delta_regularization(self, delta, *, local_state):
        _ = local_state
        return delta.bg.means.pow(2).mean() * 0.0, {"delta_l2": 0.0}


def _batch(*, rollout_idx=0, episode_end=False, repeat_only=False):
    source_refs = [(10, 0), (10, 1), (10, 2), (11, 0), (11, 1), (11, 2)]
    target_refs = [(10, 0), (10, 1), (10, 2), (11, 0), (11, 1), (11, 2), (12, 0), (12, 1), (12, 2)]
    target_roles = ["final_current_recon"] * 6 + ["final_nearby_rollout"] * 3
    steps = [
        {
            "step_idx": 0,
            "source_frame_idx": 10,
            "repeat_idx": 0,
            "rollout_block_rank": 0,
            "evidence_refs": [(10, 0), (10, 1), (10, 2)],
            "source_indices": [0, 1, 2],
            "commit_observation_memory": True,
            "update_optimizer_memory": True,
        },
        {
            "step_idx": 1,
            "source_frame_idx": 11,
            "repeat_idx": 0,
            "rollout_block_rank": 1,
            "evidence_refs": [(11, 0), (11, 1), (11, 2)],
            "source_indices": [3, 4, 5],
            "commit_observation_memory": True,
            "update_optimizer_memory": True,
        },
    ]
    input_frames = [10, 11]
    if repeat_only:
        source_refs = [(10, 0), (10, 1), (10, 2)]
        target_refs = [(10, 0), (10, 1), (10, 2)]
        target_roles = ["final_current_recon"] * 3
        steps = [
            {
                "step_idx": 0,
                "source_frame_idx": 10,
                "repeat_idx": 1,
                "rollout_block_rank": 0,
                "evidence_refs": [(10, 0), (10, 1), (10, 2)],
                "source_indices": [0, 1, 2],
                "commit_observation_memory": False,
                "update_optimizer_memory": True,
            }
        ]
        input_frames = [10]
    ifwd = {
        "scheduler_version": "iforward_v1",
        "model_family": "IForward",
        "scene_id": 1,
        "segment_id": 2,
        "episode_id": 3,
        "rollout_id_global": int(rollout_idx),
        "rollout_idx_in_episode": int(rollout_idx),
        "inner_K": len(steps),
        "steps": steps,
        "input_frame_indices": input_frames,
        "evidence_refs_flat": source_refs,
        "target_refs_flat": target_refs,
        "target_roles_flat": target_roles,
        "reset_scene_state_before_rollout": int(rollout_idx) == 0,
        "carry_scene_state_after_rollout": not bool(episode_end),
        "episode_end_after_rollout": bool(episode_end),
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


def test_iforward_forward_rollout_carries_memory_and_renders_only_final():
    bridge = FakeIForwardBridge()
    model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    out = model.forward_rollout(_batch())
    assert torch.isfinite(out.loss)
    assert bridge.observe_calls == [(10, (0, 1, 2)), (11, (3, 4, 5))]
    assert len(bridge.render_calls) == 3
    assert bridge.render_calls[0] == (3, 4, 5)
    assert bridge.render_calls[2] == (0, 1, 2)
    assert out.next_state.memory.count_tokens()["bg_point"] == 2
    assert len(out.next_state.history.entries) == 6
    assert out.image_refs == [(11, 0), (12, 0)]
    assert out.image_roles == ["current_latest", "nearby"]


def test_iforward_trainer_detaches_carry_and_discards_on_episode_end():
    bridge = FakeIForwardBridge()
    model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = IForwardTrainer(config={}, device=torch.device("cpu"), model=model, optimizer=optimizer)
    logs0 = trainer.train_step(_batch(rollout_idx=0), step=0)
    assert logs0["iforward/state_cache_size"] == 1
    cached = next(iter(trainer._state_cache.values()))
    assert cached.local_gs.bg.means.requires_grad is False
    bg_point_state = cached.memory.bg.dense_point if cached.memory.bg.dense_point is not None else cached.memory.bg.point
    assert bg_point_state is not None
    assert bg_point_state.conv_state.requires_grad is False

    logs1 = trainer.train_step(_batch(rollout_idx=1, episode_end=True), step=1)
    assert logs1["iforward/state_cache_size"] == 0
    assert bridge.delta_scale.grad is None


def test_iforward_trainer_restores_state_cache_from_state_dict():
    bridge = FakeIForwardBridge()
    model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = IForwardTrainer(config={}, device=torch.device("cpu"), model=model, optimizer=optimizer)
    trainer.train_step(_batch(rollout_idx=0), step=0)

    restored_bridge = FakeIForwardBridge()
    restored_model = IForwardModel(
        config=None,
        device=torch.device("cpu"),
        bridge=restored_bridge,
        resolver=IForwardBatchResolver(),
    )
    restored = IForwardTrainer(
        config={},
        device=torch.device("cpu"),
        model=restored_model,
        optimizer=torch.optim.SGD(restored_model.parameters(), lr=0.1),
    )
    restored.load_state_dict(trainer.state_dict())
    assert len(restored._state_cache) == 1
    logs = restored.train_step(_batch(rollout_idx=1, episode_end=True), step=1)
    assert logs["iforward/state_cache_size"] == 0


def test_iforward_non_reset_rollout_missing_state_fails_fast():
    bridge = FakeIForwardBridge()
    model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    try:
        model.forward_rollout(_batch(rollout_idx=1))
    except RuntimeError as exc:
        assert "missing carried_state" in str(exc)
    else:
        raise AssertionError("expected missing carried_state to fail")


def test_iforward_long_memory_writes_on_non_commit_optimizer_repeat():
    bridge = FakeIForwardBridge()
    model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    out = model.forward_rollout(_batch(repeat_only=True))
    counts = out.next_state.memory.count_tokens()
    assert counts["bg_point"] == 2
    assert counts["bg_cell"] == 1
    assert counts["bg_global_token"] == 1


def test_iforward_trainer_applies_grad_clip():
    bridge = FakeIForwardBridge()
    model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = IForwardTrainer(
        config={"training": {"grad_clip": {"enable": True, "max_norm": 1.0e-4}}},
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
    )
    logs = trainer.train_step(_batch(), step=0)
    assert logs["iforward/grad_clip_applied"] is True
    assert logs["iforward/grad_norm_after_clip"] <= 1.1e-4
    assert logs["iforward/grad_norm_unclipped"] >= logs["iforward/grad_norm_after_clip"]


def test_drop_short_window_drops_read_but_keeps_short_history_loss():
    bridge = FakeIForwardBridge()
    model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    out0 = model.forward_rollout(_batch(rollout_idx=0))
    bridge.render_calls.clear()
    out1 = model.forward_rollout(
        _batch(rollout_idx=1),
        carried_state=out0.next_state.detach_for_next_rollout(),
        ablation="drop_short_window",
    )
    assert torch.isfinite(out1.loss)
    assert "short_window_history" in out1.losses
    assert bridge.render_calls[-1] == tuple(range(len(out0.next_state.history.entries)))


def test_iforward_ablation_modes_are_accepted():
    for mode in (
        "full",
        "zero_all",
        "zero_point",
        "zero_cell",
        "zero_global",
        "drop_short_window",
        "freeze_write",
        "shuffle_memory",
        "bypass_memory",
    ):
        bridge = FakeIForwardBridge()
        model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
        out = model.forward_rollout(_batch(), ablation=mode)
        assert torch.isfinite(out.loss)
        assert out.stats["ablation"] == mode


def test_iforward_memory_ablation_validation_reports_retention_table():
    bridge = FakeIForwardBridge()
    model = IForwardModel(config=None, device=torch.device("cpu"), bridge=bridge, resolver=IForwardBatchResolver())
    rows = validate_iforward_memory_ablation(
        model=model,
        rollout_batches=[_batch(rollout_idx=0), _batch(rollout_idx=1, episode_end=True)],
        ablations=["full", "drop_short_window"],
    )
    assert [row["mode"] for row in rows] == ["full", "drop_short_window"]
    for row in rows:
        assert "current_psnr" in row
        assert "history_rollout_psnr" in row
        assert "history_short_window_psnr" in row
        assert "retention_gap_rollout" in row
        assert "retention_gap_short_window" in row


def test_iforward_bridge_syncs_carried_rigid_template_frame_slots():
    bg = _node_state()
    old_rigid = _rigid_node_state([10])
    new_rigid = _rigid_node_state([10, 11, 12])
    local = LocalGSState.from_node_states(bg=bg, distant=None, rigid=old_rigid, hidden_dim=2)

    class Runtime:
        stage6_hidden_dim = 2
        stage6_event_dim = 4

        def _get_or_init_node_states_bg_rigid_distant(self, batch):
            _ = batch
            return bg, new_rigid, None

    bridge = IForwardStage6Bridge(Runtime())
    node_bg, node_distant, node_rigid = bridge.sync_local_state_template_from_batch(
        local_state=local,
        batch={},
    )

    assert node_bg is bg
    assert node_distant is None
    assert node_rigid is new_rigid
    assert local.rigid_template is not None
    assert local.rigid_template.frame_ids == [10, 11, 12]
    assert local.rigid.means.requires_grad is True
