import torch

from models.iforward.history_damage_loss import HistoryDamageProbe, history_damage_hinge


def test_stage2_1_history_damage_zero_without_refs():
    after = torch.empty(0, dtype=torch.float32)
    before = torch.empty(0, dtype=torch.float32)

    loss = history_damage_hinge(after_per_ref=after, before_per_ref=before, margin=0.1)

    assert loss.item() == 0.0
    assert loss.dtype == torch.float32
    assert not HistoryDamageProbe.empty(ref=torch.zeros(3)).valid


def test_stage2_1_history_damage_zero_when_history_improves():
    before = torch.tensor([0.5, 0.7, 0.9])
    after = torch.tensor([0.4, 0.65, 0.8])

    loss = history_damage_hinge(after_per_ref=after, before_per_ref=before, margin=0.0)

    assert loss.item() == 0.0


def test_stage2_1_history_damage_positive_when_history_degrades():
    before = torch.tensor([0.5, 0.7, 0.9])
    after = torch.tensor([0.6, 0.75, 1.2])

    loss = history_damage_hinge(after_per_ref=after, before_per_ref=before, margin=0.05)

    expected = torch.tensor([0.05, 0.0, 0.25]).mean()
    assert torch.allclose(loss, expected)


def test_stage2_1_history_damage_per_ref_mean_normalization():
    before = torch.tensor([1.0, 1.0])
    after = torch.tensor([2.0, 4.0])

    loss = history_damage_hinge(after_per_ref=after, before_per_ref=before, margin=0.0)

    assert torch.allclose(loss, torch.tensor(2.0))


def test_stage2_1_history_damage_probe_stops_before_grad():
    before = torch.tensor([1.0, 1.0], requires_grad=True)
    after = torch.tensor([1.5, 0.5], requires_grad=True)

    probe = HistoryDamageProbe(target_indices=[3, 4], before_per_ref=before)
    loss = history_damage_hinge(after_per_ref=after, before_per_ref=probe.before_per_ref, margin=0.0)
    loss.backward()

    assert probe.valid
    assert before.grad is None
    assert torch.allclose(after.grad, torch.tensor([0.5, 0.0]))
