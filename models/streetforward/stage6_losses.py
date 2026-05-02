from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

@dataclass
class Stage6LossOutput:
    total_train: torch.Tensor
    self_loss: Dict[str, torch.Tensor] = field(default_factory=dict)
    teacher_anchor_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    history_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    probe_metrics: Dict[str, torch.Tensor] = field(default_factory=dict)


def aggregate_stage6_total_loss(
    *,
    self_teacher: torch.Tensor,
    self_student: torch.Tensor,
    teacher_anchor: torch.Tensor,
    history: torch.Tensor,
    w_self_teacher: float,
    w_self_student: float,
    w_teacher_anchor: float,
    w_history: float,
) -> Stage6LossOutput:
    total = (
        self_teacher * float(w_self_teacher)
        + self_student * float(w_self_student)
        + teacher_anchor * float(w_teacher_anchor)
        + history * float(w_history)
    )
    return Stage6LossOutput(
        total_train=total,
        self_loss={"teacher": self_teacher, "student": self_student},
        teacher_anchor_loss=teacher_anchor,
        history_loss=history,
        probe_metrics={},
    )

