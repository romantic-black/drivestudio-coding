from __future__ import annotations

from types import SimpleNamespace

import torch

from models.iforward import IForwardMemoryState, IForwardShortMemoryEntry, IForwardShortWindowHistory, StreamingMambaCell
from models.iforward.memory import IForwardMemoryStepContext, IForwardSceneMemory
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import EventPack, LocalGSState


def test_streaming_mamba_cell_shapes_grad_and_no_write():
    torch.manual_seed(0)
    cell = StreamingMambaCell(input_dim=5, model_dim=7, state_dim=3, conv_kernel=2, output_dim=5)
    x = torch.randn(4, 5, requires_grad=True)
    state0 = cell.init_state(4, device=x.device, dtype=x.dtype)
    out0, state1 = cell(x, state0, write_mask=torch.zeros(4, dtype=torch.bool))
    assert out0.shape == (4, 5)
    assert torch.allclose(state1.conv_state, state0.conv_state)
    assert torch.allclose(state1.ssm_state, state0.ssm_state)
    assert not bool(state1.seen.any())

    out1, state2 = cell(x, state0)
    loss = out1.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert state2.conv_state.shape == (4, 7, 2)
    assert state2.ssm_state.shape == (4, 7, 3)
    assert bool(state2.seen.all())


def test_iforward_memory_state_and_history_detach():
    state = IForwardMemoryState.empty()
    detached = state.detach()
    assert detached.count_tokens()["bg_point"] == 0

    target = {"gt_image": torch.ones(1, 1, 3, requires_grad=True), "frame_idx": 1, "cam_idx": 0}
    hist = IForwardShortWindowHistory.empty(max_entries=1).commit_targets({"targets": [target]}, (0,))
    assert len(hist.entries) == 1
    hist2 = hist.detach()
    assert hist2.entries[0]["gt_image"].requires_grad is False
    assert torch.allclose(hist2.entries[0]["gt_image"], torch.ones(1, 1, 3))


def test_short_window_history_read_context_and_drop():
    entry = IForwardShortMemoryEntry(
        frame_idx=10,
        step_idx=0,
        branch="bg",
        point_keys=torch.tensor([10, 11]),
        cell_keys=torch.tensor([20, 21]),
        global_keys=torch.tensor([1, 1]),
        event=torch.zeros(2, 2),
        ctx=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    hist = IForwardShortWindowHistory.empty(max_memory_entries=4).commit_memory_entries([entry])
    ref = torch.zeros(2, 2)
    ctx, hit = hist.read_context(
        branch="bg",
        point_keys=torch.tensor([99, 11]),
        cell_keys=torch.tensor([20, 99]),
        global_keys=torch.tensor([99, 99]),
        ref=ref,
    )
    assert hit == 1.0
    assert torch.allclose(ctx, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    dropped, dropped_hit = hist.read_context(
        branch="bg",
        point_keys=torch.tensor([10, 11]),
        cell_keys=torch.tensor([20, 21]),
        global_keys=torch.tensor([1, 1]),
        ref=ref,
        drop=True,
    )
    assert dropped_hit == 0.0
    assert torch.allclose(dropped, torch.zeros_like(ref))


def test_short_window_history_detach_moves_memory_entries_to_cpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry = IForwardShortMemoryEntry(
        frame_idx=10,
        step_idx=0,
        branch="bg",
        point_keys=torch.tensor([10, 11], device=device),
        cell_keys=torch.tensor([20, 21], device=device),
        global_keys=torch.tensor([1, 1], device=device),
        event=torch.zeros(2, 2, device=device),
        ctx=torch.ones(2, 2, device=device),
    )
    hist = IForwardShortWindowHistory.empty(max_memory_entries=4).commit_memory_entries([entry])
    detached = hist.detach()
    assert detached.memory_entries[0].ctx.device.type == "cpu"
    assert detached.memory_entries[0].point_keys.device.type == "cpu"


def _state_tensors(n: int):
    return {
        "means": torch.zeros(n, 3),
        "scales_log": torch.zeros(n, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "opacity_logit": torch.zeros(n, 1),
        "sh_dc": torch.zeros(n, 3),
        "sh_rest": torch.zeros(n, 3, 3),
    }


def test_rigid_cell_keys_are_object_local_not_world_route_coords():
    bg = NodeStateBackground(**_state_tensors(1))
    rigid = NodeStateRigid(
        **_state_tensors(2),
        point_ids=torch.tensor([[42, 0], [42, 1]], dtype=torch.long),
        instances_quats=torch.zeros(1, 4),
        instances_trans=torch.zeros(1, 3),
        instances_fv=torch.zeros(1, 1),
        instance_ids=[42],
        frame_ids=[0],
        cur_frame=0,
    )
    local_state = LocalGSState.from_node_states(bg=bg, distant=None, rigid=rigid, hidden_dim=2)
    memory = IForwardSceneMemory(event_dim=4, model_dim=4, state_dim=2, conv_kernel=2, rigid_cell_size=0.5)
    step = IForwardMemoryStepContext(
        step_idx=0,
        source_frame_idx=10,
        commit_observation_memory=True,
        update_optimizer_memory=True,
        repeat_pos_code=0.0,
        frame_pos_code=0.0,
        rollout_pos_code=0.0,
        is_frame_exit=True,
    )

    def run_with_world_shift(shift: float):
        event = EventPack(
            event_bg=torch.zeros(1, 4),
            event_rigid=torch.ones(2, 4),
            route=SimpleNamespace(
                S=torch.tensor([0, 1], dtype=torch.long),
                means_world_S=torch.tensor([[shift, 0.0, 0.0], [shift + 10.0, 0.0, 0.0]]),
            ),
        )
        _, _, _, entries = memory(
            event=event,
            local_state=local_state,
            state=IForwardMemoryState.empty(),
            short_history=IForwardShortWindowHistory.empty(),
            step_context=step,
            commit_observation_memory=True,
            update_optimizer_memory=True,
        )
        return next(item.cell_keys for item in entries if item.branch == "rigid")

    assert torch.equal(run_with_world_shift(0.0), run_with_world_shift(1000.0))


def test_bool_valid_false_prevents_memory_write():
    bg = NodeStateBackground(**_state_tensors(2))
    local_state = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=2)
    memory = IForwardSceneMemory(
        event_dim=4,
        model_dim=4,
        state_dim=2,
        conv_kernel=2,
        dense_point_memory=True,
        enable_aux_stats=True,
        log_per_k_aux_interval=1,
    )
    step = IForwardMemoryStepContext(
        step_idx=0,
        source_frame_idx=10,
        commit_observation_memory=False,
        update_optimizer_memory=True,
        repeat_pos_code=1.0,
        frame_pos_code=0.0,
        rollout_pos_code=0.0,
        global_step=0,
        is_frame_exit=True,
    )
    event = EventPack(
        event_bg=torch.ones(2, 4),
        support_bg=torch.ones(2, 1),
        valid_bg=torch.tensor([[False], [True]], dtype=torch.bool),
    )
    state, _, aux, entries = memory(
        event=event,
        local_state=local_state,
        state=IForwardMemoryState.empty(),
        short_history=IForwardShortWindowHistory.empty(),
        step_context=step,
        commit_observation_memory=False,
        update_optimizer_memory=True,
    )
    counts = state.count_tokens()
    assert counts["bg_point_seen"] == 1
    assert counts["bg_point_capacity"] == 2
    assert aux["memory/bg/hard_write_ratio"] == 0.5
    assert aux["memory/bg/valid_true_ratio"] == 0.5
    bg_entry = next(item for item in entries if item.branch == "bg")
    assert torch.equal(bg_entry.point_keys.cpu(), torch.tensor([1]))


def test_short_memory_entries_only_on_frame_exit_but_long_memory_writes():
    bg = NodeStateBackground(**_state_tensors(2))
    local_state = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=2)
    memory = IForwardSceneMemory(event_dim=4, model_dim=4, state_dim=2, conv_kernel=2, dense_point_memory=True)
    state0 = IForwardMemoryState.empty()
    non_exit = IForwardMemoryStepContext(
        step_idx=0,
        source_frame_idx=10,
        commit_observation_memory=True,
        update_optimizer_memory=True,
        repeat_pos_code=0.0,
        frame_pos_code=0.0,
        rollout_pos_code=0.0,
        is_frame_exit=False,
    )
    event = EventPack(event_bg=torch.ones(2, 4), valid_bg=torch.ones(2, 1, dtype=torch.bool))
    state1, _, _, entries1 = memory(
        event=event,
        local_state=local_state,
        state=state0,
        short_history=IForwardShortWindowHistory.empty(),
        step_context=non_exit,
        commit_observation_memory=True,
        update_optimizer_memory=True,
    )
    assert state1.count_tokens()["bg_point_seen"] == 2
    assert entries1 == []

    exit_step = IForwardMemoryStepContext(
        step_idx=1,
        source_frame_idx=10,
        commit_observation_memory=False,
        update_optimizer_memory=True,
        repeat_pos_code=1.0,
        frame_pos_code=0.0,
        rollout_pos_code=0.0,
        is_frame_exit=True,
    )
    state2, _, _, entries2 = memory(
        event=event,
        local_state=local_state,
        state=state1,
        short_history=IForwardShortWindowHistory.empty(),
        step_context=exit_step,
        commit_observation_memory=False,
        update_optimizer_memory=True,
    )
    assert state2.count_tokens()["bg_point_seen"] == 2
    assert any(item.branch == "bg" for item in entries2)
