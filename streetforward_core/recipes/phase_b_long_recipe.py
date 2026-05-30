from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from streetforward_core.legacy.stage6_facade import Stage6LegacyFacade


@dataclass
class PhaseBLongForwardOutput:
    loss: torch.Tensor
    legacy: Dict[str, Any]

    @property
    def per_step(self) -> List[Dict[str, float]]:
        return list(self.legacy.get("per_step") or [])

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self.legacy.get("stats") or {})

    def to_legacy_dict(self) -> Dict[str, Any]:
        return dict(self.legacy)


class PhaseBLongRecipe(nn.Module):
    """Legacy-backed Phase B Long recipe boundary.

    The existing MinimalStreetForwardStage6_0 implementation remains the
    parity reference while scheduler/batch/runner contracts are stabilized.
    """

    def __init__(self, *, facade: Stage6LegacyFacade):
        super().__init__()
        self.facade = facade

    def forward(self, batch: Dict[str, Any]) -> PhaseBLongForwardOutput:
        legacy = self.facade.runtime._forward_6_0_phase_b_long(batch)
        return PhaseBLongForwardOutput(loss=legacy["loss"], legacy=legacy)

