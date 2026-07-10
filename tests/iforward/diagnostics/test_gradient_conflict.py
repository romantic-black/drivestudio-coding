from __future__ import annotations

import torch

from models.iforward.diagnostics.gradient_conflict import gradient_conflict_cosines


def test_gradient_conflict_cosines_reports_group_and_all_pairs():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    x = torch.tensor([[1.0, 0.0]])
    y = model(x).sum()
    losses = {
        "current": y,
        "in_rollout_history": -y,
    }

    out = gradient_conflict_cosines(
        model=model,
        losses=losses,
        groups={"linear": ("0.",)},
    )

    assert out["all/current_vs_in_rollout_history"] < -0.99
    assert out["linear/current_vs_in_rollout_history"] < -0.99
