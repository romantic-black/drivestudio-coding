from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.iforward import IForwardMemoryState, IForwardShortMemoryEntry, IForwardShortWindowHistory, StreamingMambaCell
from models.iforward.context_adapter import IForwardContextAdapter
from models.iforward.iforward_v6_state import IForwardV6MemoryState
from models.iforward.local_conflict_xcpe import IForwardLocalConflictXcpe
from models.iforward.memory import IForwardMemoryStepContext, IForwardSceneMemory
from models.iforward.point_mamba_memory import IForwardPointMambaMemory
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
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


def test_short_window_history_row_aligned_read_uses_full_ctx_and_valid_ratio():
    entry = IForwardShortMemoryEntry(
        frame_idx=10,
        step_idx=0,
        branch="bg",
        point_keys=torch.tensor([10, 11, 12]),
        cell_keys=torch.tensor([20, 21, 22]),
        global_keys=torch.tensor([1, 1, 1]),
        event=torch.zeros(0, 2),
        ctx=torch.tensor([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]),
        valid=torch.tensor([[True], [False], [True]]),
        row_aligned=True,
    )
    hist = IForwardShortWindowHistory.empty(max_memory_entries=4).commit_memory_entries([entry])
    ref = torch.zeros(3, 2)
    ctx, hit = hist.read_context(
        branch="bg",
        point_keys=torch.tensor([99, 98, 97]),
        cell_keys=torch.tensor([99, 98, 97]),
        global_keys=torch.tensor([99, 99, 99]),
        ref=ref,
    )
    assert abs(hit - (2.0 / 3.0)) < 1.0e-6
    assert torch.allclose(ctx, entry.ctx)


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
    assert bg_entry.row_aligned is True
    assert torch.equal(bg_entry.point_keys.cpu(), torch.tensor([0, 1]))
    assert torch.equal(bg_entry.valid.cpu().reshape(-1), torch.tensor([False, True]))
    assert torch.allclose(bg_entry.ctx[0], torch.zeros_like(bg_entry.ctx[0]))


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
    bg_entry = next(item for item in entries2 if item.branch == "bg")
    assert bg_entry.row_aligned is True
    assert int(bg_entry.ctx.shape[0]) == 2


def _v6_step() -> IForwardMemoryStepContext:
    return IForwardMemoryStepContext(
        step_idx=0,
        source_frame_idx=10,
        commit_observation_memory=True,
        update_optimizer_memory=True,
        repeat_pos_code=0.5,
        frame_pos_code=0.25,
        rollout_pos_code=0.125,
        global_step=0,
        is_frame_exit=True,
    )


def _v6_local_state() -> LocalGSState:
    bg = NodeStateBackground(**_state_tensors(2))
    distant = NodeStateDistant(**_state_tensors(1))
    rigid = NodeStateRigid(
        **_state_tensors(2),
        point_ids=torch.tensor([[0], [0]], dtype=torch.long),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        instances_trans=torch.zeros(1, 1, 3),
        instances_fv=torch.ones(1, 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[10],
        cur_frame=10,
    )
    local_state = LocalGSState.from_node_states(bg=bg, distant=distant, rigid=rigid, hidden_dim=2)
    local_state.bg.means.data = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    local_state.rigid.means.data = torch.tensor([[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]])
    return local_state


def _v6_event() -> EventPack:
    return EventPack(
        event_bg=torch.randn(2, 4),
        event_distant=torch.randn(1, 4),
        event_rigid=torch.randn(2, 4),
        support_bg=torch.ones(2, 1),
        support_distant=torch.ones(1, 1),
        support_rigid=torch.ones(2, 1),
        valid_bg=torch.tensor([[False], [True]], dtype=torch.bool),
        valid_distant=torch.ones(1, 1, dtype=torch.bool),
        valid_rigid=torch.ones(2, 1, dtype=torch.bool),
        obs_code_bg=torch.zeros(2, 2),
        obs_code_distant=torch.zeros(1, 2),
        obs_code_rigid=torch.zeros(2, 2),
        route=SimpleNamespace(
            S=torch.tensor([0, 1], dtype=torch.long),
            S_in=torch.tensor([0], dtype=torch.long),
            S_out=torch.tensor([1], dtype=torch.long),
            inside_mask_S=torch.tensor([True, False]),
            means_world_S=torch.tensor([[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        ),
    )


def test_iforward_v6_point_mamba_dense_and_keyed_state_write_mask():
    torch.manual_seed(1)
    memory = IForwardPointMambaMemory(
        event_dim=4,
        point_ctx_dim=3,
        model_dim=5,
        state_dim=2,
        conv_kernel=2,
    )
    state, pack, aux = memory(
        event=_v6_event(),
        local_state=_v6_local_state(),
        state=IForwardV6MemoryState.empty(),
        step_context=_v6_step(),
    )
    counts = state.count_tokens()
    assert pack.ctx_bg.shape == (2, 3)
    assert pack.ctx_distant.shape == (1, 3)
    assert pack.ctx_rigid.shape == (2, 3)
    assert counts["bg_point_seen"] == 1
    assert counts["bg_point_capacity"] == 2
    assert counts["distant_point_seen"] == 1
    assert counts["rigid_point_seen"] == 2
    assert aux["point_mamba/bg_update_ratio"] == 0.5


def test_iforward_v6_local_conflict_fallback_routes_bg_rigid_and_distant():
    torch.manual_seed(2)
    event = _v6_event()
    local_state = _v6_local_state()
    point_pack = type(
        "PointPack",
        (),
        {
            "ctx_bg": torch.randn(2, 3),
            "ctx_distant": torch.randn(1, 3),
            "ctx_rigid": torch.randn(2, 3),
        },
    )()
    module = IForwardLocalConflictXcpe(
        event_dim=4,
        point_ctx_dim=3,
        hidden_dim=4,
        output_dim=4,
        num_blocks=1,
        sparse_backend="fallback_neighbor_mean",
        voxel_size=0.5,
    )
    pack = module(
        event=event,
        point_ctx=point_pack,
        local_state=local_state,
        step_context=_v6_step(),
        aabb_min=torch.tensor([-1.0, -1.0, -1.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
    )
    assert pack.ctx_bg.shape == (2, 4)
    assert pack.ctx_distant.shape == (1, 4)
    assert pack.ctx_rigid.shape == (2, 4)
    assert torch.isfinite(pack.ctx_bg).all()
    assert torch.isfinite(pack.ctx_rigid[0]).all()
    assert torch.allclose(pack.ctx_rigid[1], torch.zeros_like(pack.ctx_rigid[1]))
    assert pack.aux["local_xcpe/num_points"] == 3.0


def test_iforward_v6_local_conflict_near_rigid_outside_aabb_fails_strict_layout():
    torch.manual_seed(2)
    event = _v6_event()
    event.route.inside_mask_S = torch.tensor([True, True])
    local_state = _v6_local_state()
    point_pack = type(
        "PointPack",
        (),
        {
            "ctx_bg": torch.randn(2, 3),
            "ctx_distant": torch.randn(1, 3),
            "ctx_rigid": torch.randn(2, 3),
        },
    )()
    module = IForwardLocalConflictXcpe(
        event_dim=4,
        point_ctx_dim=3,
        hidden_dim=4,
        output_dim=4,
        num_blocks=1,
        sparse_backend="fallback_neighbor_mean",
        voxel_size=0.5,
    )
    with pytest.raises(RuntimeError, match="outside segment_aabb"):
        module(
            event=event,
            point_ctx=point_pack,
            local_state=local_state,
            step_context=_v6_step(),
            aabb_min=torch.tensor([-1.0, -1.0, -1.0]),
            aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        )


def test_iforward_v6_context_adapter_outputs_finite_nonzero_initialized_context():
    torch.manual_seed(3)
    event = _v6_event()
    point_pack = type(
        "PointPack",
        (),
        {
            "ctx_bg": torch.randn(2, 3),
            "ctx_distant": torch.randn(1, 3),
            "ctx_rigid": torch.randn(2, 3),
        },
    )()
    local_pack = type(
        "LocalPack",
        (),
        {
            "ctx_bg": torch.randn(2, 4),
            "ctx_distant": torch.randn(1, 4),
            "ctx_rigid": torch.randn(2, 4),
        },
    )()
    adapter = IForwardContextAdapter(event_dim=4, point_ctx_dim=3, local_ctx_dim=4, output_dim=4)
    out = adapter(event=event, point_ctx=point_pack, local_ctx=local_pack, step_context=_v6_step())
    assert out.ctx_bg.shape == (2, 4)
    assert out.ctx_distant.shape == (1, 4)
    assert out.ctx_rigid.shape == (2, 4)
    assert torch.isfinite(out.ctx_bg).all()
    assert adapter.net[-1].weight.detach().norm().item() > 0.0
