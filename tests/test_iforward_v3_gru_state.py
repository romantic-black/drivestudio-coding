from __future__ import annotations

from typing import List

import torch

from models.iforward.gru_memory import IForwardTimeAwarePointGRU
from models.iforward.history_gate import IForwardAttributeGate, IForwardGatePack
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import EventPack, LocalGSState
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


def _node_state(n: int) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _rigid_state(frame_ids: List[int], n: int) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
        point_ids=torch.arange(n, dtype=torch.long).view(n, 1),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * len(frame_ids)),
        instances_trans=torch.zeros(len(frame_ids), 1, 3),
        instances_fv=torch.ones(len(frame_ids), 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[int(x) for x in frame_ids],
        cur_frame=0,
    )


def _delta(n: int, hidden_dim: int, ref: torch.Tensor, value: float = 0.1) -> BranchDelta:
    return BranchDelta(
        means=ref.new_full((n, 3), float(value)),
        scales_log=ref.new_zeros((n, 3)),
        quat_axis_angle=ref.new_zeros((n, 3)),
        opacity_logit=ref.new_zeros((n, 1)),
        sh=ref.new_zeros((n, 12)),
        hidden=ref.new_zeros((n, hidden_dim)),
        confidence=ref.new_ones((n, 1)),
        noop=ref.new_zeros((n, 1)),
    )


def _gate(n: int, ref: torch.Tensor) -> IForwardAttributeGate:
    one = ref.new_ones((n, 1))
    return IForwardAttributeGate(means=one, scales=one, quat=one, opacity=one, sh=one, hidden=one)


class _Step:
    source_frame_idx = 10
    episode_visit_idx = 0
    repeat_pos_code = 0.0
    frame_pos_code = 0.0
    rollout_pos_code = 0.0
    update_optimizer_memory = True


def test_iforward_v3_gru_dt_write_masks_rigid_route_and_detach() -> None:
    local = LocalGSState.from_node_states(bg=_node_state(2), distant=None, rigid=_rigid_state([10], 3), hidden_dim=2)
    gru = IForwardTimeAwarePointGRU(event_dim=4, hidden_dim=5, ctx_dim=3, hard_valid_required=True)
    state = gru.init_state(local)
    route = type("Route", (), {"S": torch.tensor([2, 0], dtype=torch.long)})()
    event = EventPack(
        event_bg=torch.randn(2, 4),
        event_rigid=torch.randn(2, 4),
        support_bg=torch.ones(2, 1),
        support_rigid=torch.ones(2, 1),
        valid_bg=torch.tensor([[True], [False]]),
        valid_rigid=torch.ones(2, 1, dtype=torch.bool),
        obs_code_bg=torch.zeros(2, 2),
        obs_code_rigid=torch.zeros(2, 2),
        route=route,
    )

    ctx, prepared, _ = gru.read(event=event, local_state=local, state=state, step_context=_Step(), ablation="full")
    assert ctx.ctx_bg.shape == (2, 3)
    assert prepared.bg.dt.tolist() == [[0.0], [0.0]]

    delta = DeltaPack(
        bg=_delta(2, 2, local.bg.means),
        rigid=_delta(2, 2, local.bg.means),
    )
    gates = IForwardGatePack(bg=_gate(2, local.bg.means), rigid=_gate(2, local.bg.means))
    state, aux = gru.write_after_update(
        prepared=prepared,
        state=state,
        delta_raw=delta,
        gate=gates,
        step_context=_Step(),
        ablation="full",
    )

    assert aux["v3/gru/bg_write_ratio"] == 0.5
    assert state.bg.seen.tolist() == [True, False]
    assert state.bg.last_visit_idx.tolist() == [0, -1]
    assert state.bg.last_source_frame_idx.tolist() == [10, -1]
    assert state.rigid.seen.tolist() == [True, False, True]
    assert state.rigid.last_visit_idx.tolist() == [0, -1, 0]
    assert state.rigid.last_source_frame_idx.tolist() == [10, -1, 10]

    class _StepLater(_Step):
        source_frame_idx = 5
        episode_visit_idx = 3

    _, prepared_later, _ = gru.read(
        event=event,
        local_state=local,
        state=state,
        step_context=_StepLater(),
        ablation="full",
    )
    assert prepared_later.bg.dt.tolist() == [[3.0], [0.0]]
    assert torch.isfinite(prepared_later.bg.h_prior).all()

    event_no_write = EventPack(
        event_bg=event.event_bg,
        event_rigid=event.event_rigid,
        support_bg=torch.ones(2, 1),
        support_rigid=torch.ones(2, 1),
        valid_bg=torch.zeros(2, 1, dtype=torch.bool),
        valid_rigid=torch.zeros(2, 1, dtype=torch.bool),
        obs_code_bg=event.obs_code_bg,
        obs_code_rigid=event.obs_code_rigid,
        route=route,
    )
    _, prepared_no_write, _ = gru.read(
        event=event_no_write,
        local_state=local,
        state=state,
        step_context=_StepLater(),
        ablation="full",
    )
    state, aux = gru.write_after_update(
        prepared=prepared_no_write,
        state=state,
        delta_raw=delta,
        gate=gates,
        step_context=_StepLater(),
        ablation="full",
    )
    assert aux["v3/gru/bg_write_ratio"] == 0.0
    assert state.bg.seen.tolist() == [True, False]
    assert state.bg.last_visit_idx.tolist() == [3, -1]
    assert state.bg.last_source_frame_idx.tolist() == [10, -1]
    assert state.rigid.last_visit_idx.tolist() == [3, -1, 3]
    assert state.rigid.last_source_frame_idx.tolist() == [10, -1, 10]

    class _StepNext(_Step):
        source_frame_idx = 6
        episode_visit_idx = 4

    _, prepared_next, _ = gru.read(
        event=event,
        local_state=local,
        state=state,
        step_context=_StepNext(),
        ablation="full",
    )
    assert prepared_next.bg.dt.tolist() == [[1.0], [0.0]]

    detached = state.detach()
    assert detached.bg.h.requires_grad is False
    assert detached.rigid.h.requires_grad is False
