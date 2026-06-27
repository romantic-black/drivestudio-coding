from __future__ import annotations

import torch

from models.iforward.stage2_3 import EpisodeHistoryBankV3, history_damage_hinge_v3, role_normalized_loss_v3


def test_stage2_3_history_bank_best_last_and_damage():
    bank = EpisodeHistoryBankV3.empty()
    bank = bank.update(sequence_pos=0, loss=torch.tensor(0.4), psnr=torch.tensor(4.0), rollout_id=1)
    bank = bank.update(sequence_pos=0, loss=torch.tensor(0.2), psnr=torch.tensor(7.0), rollout_id=2)
    assert torch.allclose(bank.entries[0].best_loss, torch.tensor(0.2))
    assert torch.allclose(bank.entries[0].last_loss, torch.tensor(0.2))
    loss, stats = history_damage_hinge_v3(
        repair_losses=torch.tensor([0.35]),
        bank=bank,
        positions=[0],
        valid=torch.tensor([True]),
        margin=0.05,
    )
    assert torch.allclose(loss, torch.tensor(0.10), atol=1.0e-6)
    assert stats["stage2_3/best_damage_num_pos"] == 1.0


def test_stage2_3_role_normalized_loss_empty_uses_ref():
    ref = torch.ones(2)
    out = role_normalized_loss_v3([], ref=ref)
    assert out.item() == 0.0
    assert out.device == ref.device
