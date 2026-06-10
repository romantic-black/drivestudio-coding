from __future__ import annotations

import torch

from models.iforward.history_gradient_bank import build_history_gradient_bank_from_loss
from models.streetforward.node_states import NodeStateBackground
from models.streetforward.stage6_0 import LocalGSState


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


def _cfg(dtype: str = "fp32") -> dict:
    return {
        "bank": {"dtype": dtype, "min_grad_norm": 1.0e-8},
        "attrs": {"means": True, "scales": True, "quat": True, "opacity": True, "sh": True},
    }


def test_history_gradient_bank_invalid_without_history_refs() -> None:
    local = _local_state(2)
    loss = local.bg.means[:, 0].sum()

    bank = build_history_gradient_bank_from_loss(
        loss_history=loss,
        final_local_state=local,
        rollout_id=0,
        history_num_refs=0,
        cfg=_cfg(),
    )

    assert bank is None


def test_history_gradient_bank_shapes_validity_and_detach() -> None:
    local = _local_state(3)
    loss = (
        local.bg.means[:, 0].sum()
        + 2.0 * local.bg.scales_log[:, 1].sum()
        + 3.0 * local.bg.quats[:, 0].sum()
        + 4.0 * local.bg.opacity_logit.sum()
        + 5.0 * local.bg.sh_dc.sum()
        + 6.0 * local.bg.sh_rest.sum()
    )

    bank = build_history_gradient_bank_from_loss(
        loss_history=loss,
        final_local_state=local,
        rollout_id=7,
        history_num_refs=2,
        cfg=_cfg("fp32"),
    )

    assert bank is not None
    assert bank.valid is True
    assert bank.source_rollout_id == 7
    assert bank.source_history_num_refs == 2
    assert bank.bg.means.direction.shape == (3, 3)
    assert bank.bg.scales.direction.shape == (3, 3)
    assert bank.bg.quat.direction.shape == (3, 4)
    assert bank.bg.opacity.direction.shape == (3, 1)
    assert bank.bg.sh.direction.shape == (3, 12)
    assert bank.bg.means.valid.tolist() == [True, True, True]
    assert bank.bg.means.direction.requires_grad is False
    assert bank.bg.sh.log_norm.requires_grad is False
    assert torch.isfinite(bank.bg.sh.direction).all()

    detached = bank.detach()
    assert detached.bg.means.direction.requires_grad is False
    assert detached.bg.means.direction.data_ptr() != bank.bg.means.direction.data_ptr()


def test_history_gradient_bank_returns_none_for_non_grad_history_loss() -> None:
    local = _local_state(2)
    loss = torch.tensor(1.0)

    bank = build_history_gradient_bank_from_loss(
        loss_history=loss,
        final_local_state=local,
        rollout_id=0,
        history_num_refs=1,
        cfg=_cfg(),
    )

    assert bank is None
