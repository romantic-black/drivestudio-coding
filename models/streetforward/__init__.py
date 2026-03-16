"""
StreetForward package exports.

Important: keep imports lazy to avoid pulling optional heavy dependencies (e.g. torchsparse)
when users only want lightweight modules (e.g. minimal trainer docs/tools).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "StreetForwardTrainer",
    "MinimalStreetForward",
    "MinimalStreetForwardStage1",
    "MinimalStreetForwardStage1_1",
    "MinimalStreetForwardStage2_0",
    "MinimalStreetForwardStage2_1",
    "MinimalStreetForwardStage2_2",
    "MinimalStreetForwardStage3_2d",
    "NodeState",
    "NodeStateBackground",
    "NodeStateRigid",
    "NodeStateDistant",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "StreetForwardTrainer":
        from models.streetforward.trainer import StreetForwardTrainer

        return StreetForwardTrainer
    if name == "MinimalStreetForward":
        from models.streetforward.minimal_trainer import MinimalStreetForward

        return MinimalStreetForward
    if name == "MinimalStreetForwardStage1":
        from models.streetforward.minimal_trainer_stage1 import MinimalStreetForwardStage1

        return MinimalStreetForwardStage1
    if name == "MinimalStreetForwardStage1_1":
        from models.streetforward.minimal_trainer_stage1_1 import MinimalStreetForwardStage1_1

        return MinimalStreetForwardStage1_1
    if name == "MinimalStreetForwardStage2_0":
        from models.streetforward.minimal_trainer_stage2_0 import MinimalStreetForwardStage2_0

        return MinimalStreetForwardStage2_0
    if name == "MinimalStreetForwardStage2_1":
        from models.streetforward.minimal_trainer_stage2_1 import MinimalStreetForwardStage2_1

        return MinimalStreetForwardStage2_1
    if name == "MinimalStreetForwardStage2_2":
        from models.streetforward.minimal_trainer_stage2_2 import MinimalStreetForwardStage2_2

        return MinimalStreetForwardStage2_2
    if name == "MinimalStreetForwardStage3_2d":
        from models.streetforward.minimal_trainer_stage3_2d import MinimalStreetForwardStage3_2d

        return MinimalStreetForwardStage3_2d
    if name in {"NodeState", "NodeStateBackground", "NodeStateRigid", "NodeStateDistant"}:
        from models.streetforward import node_states as _ns

        return getattr(_ns, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
