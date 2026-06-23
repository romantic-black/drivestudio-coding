from __future__ import annotations

import torch

from models.iforward.parent_spatial_backbone import ParentTokenBuilder
from models.iforward.sequence10_history_bank import Sequence10HistoryBank, sequence10_damage_hinge_from_bank


def test_parent_token_builder_frame_gap_and_visit_kind_affect_output():
    torch.manual_seed(0)
    builder = ParentTokenBuilder(context_dim=4, param_support_dim=3, token_dim=8)
    parent_context = torch.randn(5, 4)
    param_support = torch.randn(5, 3)
    support = torch.ones(5)
    valid = torch.ones(5, dtype=torch.bool)
    branch_id = torch.zeros(5, dtype=torch.long)
    base = builder(
        parent_context=parent_context,
        param_support=param_support,
        support=support,
        valid_mask=valid,
        branch_id=branch_id,
        frame_gap=0,
        visit_kind_id=1,
    )
    changed_gap = builder(
        parent_context=parent_context,
        param_support=param_support,
        support=support,
        valid_mask=valid,
        branch_id=branch_id,
        frame_gap=2,
        visit_kind_id=1,
    )
    changed_visit = builder(
        parent_context=parent_context,
        param_support=param_support,
        support=support,
        valid_mask=valid,
        branch_id=branch_id,
        frame_gap=0,
        visit_kind_id=2,
    )
    assert torch.isfinite(base).all()
    assert not torch.allclose(base, changed_gap)
    assert not torch.allclose(base, changed_visit)


def test_sequence10_history_bank_detaches_best_loss_and_damage_hinge():
    bank = Sequence10HistoryBank.empty()
    loss = torch.tensor(0.5, requires_grad=True)
    psnr = torch.tensor(12.0, requires_grad=True)
    bank.update({3: loss}, {3: psnr})
    assert bank.valid[3].item() is True
    assert bank.best_loss[3].requires_grad is False
    before, valid = bank.before_for_positions([3], device=torch.device("cpu"), dtype=torch.float32)
    after_bad = torch.tensor([0.7], requires_grad=True)
    damage = sequence10_damage_hinge_from_bank(
        after_loss=after_bad,
        before_loss=before,
        valid=valid,
        margin=0.05,
    )
    assert torch.allclose(damage, torch.tensor(0.15))
    damage.backward()
    assert after_bad.grad is not None
    assert bank.best_loss.grad is None


def test_sequence10_damage_hinge_zero_for_improvement_or_no_valid_refs():
    bank = Sequence10HistoryBank.empty()
    bank.update({1: torch.tensor(0.5)})
    before, valid = bank.before_for_positions([1], device=torch.device("cpu"), dtype=torch.float32)
    improved = sequence10_damage_hinge_from_bank(
        after_loss=torch.tensor([0.45]),
        before_loss=before,
        valid=valid,
        margin=0.01,
    )
    empty = sequence10_damage_hinge_from_bank(
        after_loss=torch.tensor([0.7]),
        before_loss=torch.tensor([0.0]),
        valid=torch.tensor([False]),
        margin=0.01,
    )
    assert improved.item() == 0.0
    assert empty.item() == 0.0
