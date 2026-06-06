from __future__ import annotations

import torch
import pytest

from models.iforward.history_ema import IForwardHistoryEMAState, IForwardResidualPack
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import EventPack, LocalGSState
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


def _rigid_state(n: int = 3) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
        point_ids=torch.arange(n, dtype=torch.long).view(n, 1),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        instances_trans=torch.zeros(1, 1, 3),
        instances_fv=torch.ones(1, 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[10],
        cur_frame=0,
    )


def _delta(ref: torch.Tensor, value: float) -> DeltaPack:
    n = int(ref.shape[0])
    return DeltaPack(
        bg=BranchDelta(
            means=ref.new_full((n, 3), float(value)),
            scales_log=ref.new_zeros((n, 3)),
            quat_axis_angle=ref.new_zeros((n, 3)),
            opacity_logit=ref.new_zeros((n, 1)),
            sh=ref.new_zeros((n, 12)),
            hidden=ref.new_zeros((n, 2)),
            confidence=ref.new_ones((n, 1)),
            noop=ref.new_zeros((n, 1)),
        )
    )


def test_iforward_v3_history_ema_pending_support_update_norm_and_residual_commit() -> None:
    local = LocalGSState.from_node_states(bg=_node_state(2), distant=None, rigid=None, hidden_dim=2)
    hist = IForwardHistoryEMAState.from_local_state(local)
    event = EventPack(
        event_bg=torch.zeros(2, 4),
        support_bg=torch.tensor([[1.0], [3.0]]),
        valid_bg=torch.tensor([[True], [False]]),
    )

    aux = hist.record_block_support_snapshot(event=event, local_state=local)
    assert aux["v3/history/bg_support_snapshot_rows"] == 2.0
    assert aux["v3/history/bg_support_snapshot_present_rows"] == 2.0
    assert aux["v3/history/bg_support_snapshot_visible_rows"] == 1.0
    assert hist.bg.block_present_count.tolist() == [[1.0], [1.0]]
    assert hist.bg.block_visible_count.tolist() == [[1.0], [0.0]]
    assert torch.count_nonzero(hist.bg.initialized).item() == 0

    update_betas = {"fast_beta": 0.5, "slow_beta": 0.8}
    hist.record_update_norm(delta=_delta(local.bg.means, 0.2), update_betas=update_betas)
    first_fast = hist.bg.update_norm_fast.clone()
    hist.record_update_norm(delta=_delta(local.bg.means, 0.4), update_betas=update_betas)
    assert torch.all(hist.bg.update_norm_fast > first_fast)

    assert torch.count_nonzero(hist.bg.error_fast).item() == 0
    pack = IForwardResidualPack(
        error_bg=torch.tensor([[0.5], [0.7]]),
        support_bg=torch.tensor([[1.0], [0.0]]),
    )
    hist.commit_residual(
        pack,
        residual_betas={"fast_beta": 0.5, "slow_beta": 0.8},
        support_min={"bg": 0.0},
    )
    assert hist.bg.error_fast[0].item() > 0.0
    assert hist.bg.error_fast[1].item() == 0.0

    hist.bg.support_fast[1] = 2.0
    hist.bg.support_slow[1] = 2.0
    aux = hist.commit_block_support(
        support_betas={
            "fast_beta_visible": 0.5,
            "fast_beta_invisible": 0.6,
            "slow_beta_visible": 0.8,
            "slow_beta_invisible": 0.9,
        },
        support_min={"bg": 0.0},
    )
    assert aux["v3/history/bg_support_commit_rows"] == 2.0
    assert aux["v3/history/bg_support_present_ratio"] == 1.0
    assert aux["v3/history/bg_support_visible_ratio"] == 0.5
    assert aux["v3/history/bg_support_invisible_ratio"] == 0.5
    assert hist.bg.initialized.tolist() == [[1.0], [0.0]]
    assert hist.bg.support_fast[1].item() == pytest.approx(1.2)
    assert hist.bg.support_slow[1].item() == pytest.approx(1.8)
    assert torch.count_nonzero(hist.bg.block_present_count).item() == 0
    assert torch.count_nonzero(hist.bg.block_visible_count).item() == 0

    detached = hist.detach()
    assert detached.bg.support_fast.requires_grad is False


def test_iforward_v3_history_ema_rigid_support_snapshot_scatters_route_s_to_full_rows() -> None:
    local = LocalGSState.from_node_states(bg=_node_state(1), distant=None, rigid=_rigid_state(3), hidden_dim=2)
    hist = IForwardHistoryEMAState.from_local_state(local)
    route = type("Route", (), {"S": torch.tensor([2, 0], dtype=torch.long)})()
    event = EventPack(
        event_bg=torch.zeros(1, 4),
        event_rigid=torch.zeros(2, 4),
        support_bg=torch.ones(1, 1),
        support_rigid=torch.tensor([[1.0], [0.0]]),
        valid_bg=torch.ones(1, 1, dtype=torch.bool),
        valid_rigid=torch.tensor([[True], [False]]),
        route=route,
    )

    aux = hist.record_block_support_snapshot(event=event, local_state=local)

    assert aux["v3/history/rigid_support_snapshot_rows"] == 2.0
    assert aux["v3/history/rigid_support_snapshot_present_rows"] == 2.0
    assert aux["v3/history/rigid_support_snapshot_visible_rows"] == 1.0
    assert hist.rigid is not None
    assert hist.rigid.block_present_count.tolist() == [[1.0], [0.0], [1.0]]
    assert hist.rigid.block_visible_count.tolist() == [[0.0], [0.0], [1.0]]
    assert hist.rigid.block_support_sum[0].item() == pytest.approx(0.0)
    assert hist.rigid.block_support_sum[2].item() == pytest.approx(float(torch.log1p(torch.tensor(1.0)).item()))
