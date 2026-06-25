from __future__ import annotations

import torch

from models.iforward.stage2_2.episode_history_bank import EpisodeHistoryBankV2, history_damage_hinge_v2
from models.iforward.stage2_2.sequence_loss import role_normalized_loss


def test_stage2_2_history_bank_best_last_and_all_seen():
    bank = EpisodeHistoryBankV2.empty()
    bank = bank.update(sequence_pos=0, loss=torch.tensor(0.5), psnr=torch.tensor(20.0), rollout_id=1)
    bank = bank.update(sequence_pos=0, loss=torch.tensor(0.4), psnr=torch.tensor(22.0), rollout_id=2)
    assert torch.allclose(bank.entries[0].last_loss, torch.tensor(0.4))
    assert torch.allclose(bank.entries[0].best_loss, torch.tensor(0.4))
    assert bank.all_seen([0])
    assert not bank.all_seen([0, 1])


def test_stage2_2_history_repair_damage_and_role_normalization():
    bank = EpisodeHistoryBankV2.empty().update(
        sequence_pos=0,
        loss=torch.tensor(0.4),
        psnr=torch.tensor(22.0),
        rollout_id=2,
    )
    loss, stats = history_damage_hinge_v2(
        repair_losses=torch.tensor([0.6]),
        bank=bank,
        positions=[0],
        margin=0.1,
    )
    assert torch.allclose(loss, torch.tensor(0.1))
    assert stats["stage2_2/best_damage_num_pos"] == 1.0
    assert abs(stats["stage2_2/best_damage_p90"] - 0.1) < 1.0e-5
    assert abs(stats["stage2_2/best_damage_max"] - 0.1) < 1.0e-5
    zero, _ = history_damage_hinge_v2(
        repair_losses=torch.tensor([0.5]),
        bank=bank,
        positions=[0],
        margin=0.1,
    )
    assert torch.allclose(zero, torch.tensor(0.0))
    normalized = role_normalized_loss({"current": torch.tensor(2.0), "history": torch.tensor(4.0)}, {"current": 1.0, "history": 3.0})
    assert torch.allclose(normalized, torch.tensor(3.5))
