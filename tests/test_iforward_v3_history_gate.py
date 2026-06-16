from __future__ import annotations

import torch

from models.iforward.history_ema import IForwardHistoryEMAState
from models.iforward.history_gate import IForwardHistoryGate
from models.streetforward.node_states import NodeStateBackground
from models.streetforward.stage6_0 import ContextPack, EventPack, LocalGSState


def _node_state(n: int = 3) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def test_iforward_v3_history_gate_cold_open_hard_mask_min_gate_and_empty_branches() -> None:
    local = LocalGSState.from_node_states(bg=_node_state(3), distant=None, rigid=None, hidden_dim=2)
    hist = IForwardHistoryEMAState.from_local_state(local)
    gate = IForwardHistoryGate(
        event_dim=4,
        ctx_dim=3,
        hidden_dim=8,
        history_embed_dim=4,
        min_gate={"means": 0.2, "scales": 0.2, "quat": 0.2, "opacity": 0.2, "sh": 0.2},
        init_bias={"means": -10.0, "scales": -10.0, "quat": -10.0, "opacity": -10.0, "sh": -10.0},
        support_min={"bg": 0.5},
        cold_open_uninitialized=True,
        bind_with_mask_update=True,
    )
    event = EventPack(
        event_bg=torch.zeros(3, 4),
        support_bg=torch.tensor([[1.0], [0.1], [1.0]]),
        valid_bg=torch.tensor([[True], [True], [False]]),
        obs_code_bg=torch.zeros(3, 2),
    )
    ctx = ContextPack(ctx_bg=torch.zeros(3, 3))

    cold = gate(event=event, ctx_memory=ctx, history_ema=hist, local_state=local)
    assert torch.allclose(cold.bg.means[:, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(cold.bg.raw_means[:, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert cold.bg.mask_update[:, 0].tolist() == [True, False, False]
    assert torch.allclose(cold.bg.support_now[:, 0], torch.tensor([1.0, 0.1, 1.0]))
    assert cold.distant is None
    assert cold.rigid is None

    hist.bg.initialized[0] = 1.0
    warm = gate(event=event, ctx_memory=ctx, history_ema=hist, local_state=local)
    assert warm.bg.means[0].item() >= 0.2
    assert warm.bg.means[1].item() == 0.0
    assert warm.bg.means[2].item() == 0.0
    assert warm.bg.raw_means[1].item() == 1.0
    assert warm.bg.raw_means[2].item() == 1.0

    bypass = gate(event=event, ctx_memory=ctx, history_ema=hist, local_state=local, ablation="no_history_gate")
    assert torch.allclose(bypass.bg.means[:, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(bypass.bg.raw_means[:, 0], torch.ones(3))
