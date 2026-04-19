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
    "MinimalStreetForwardStage3_1",
    "MinimalStreetForwardStage3_2",
    "MinimalStreetForwardStage3_3",
    "MinimalStreetForwardStage4_0",
    "MinimalStreetForwardStage4_1",
    "MinimalStreetForwardStage4_2",
    "MinimalStreetForwardStage4_3",
    "MinimalStreetForwardStage4_4",
    "MinimalStreetForwardStage4_5",
    "MinimalStreetForwardStage4_6",
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
    if name == "MinimalStreetForwardStage3_1":
        from models.streetforward.minimal_trainer_stage3_1 import MinimalStreetForwardStage3_1

        return MinimalStreetForwardStage3_1
    if name == "MinimalStreetForwardStage3_2":
        from models.streetforward.minimal_trainer_stage3_2 import MinimalStreetForwardStage3_2

        return MinimalStreetForwardStage3_2
    if name == "MinimalStreetForwardStage3_3":
        from models.streetforward.minimal_trainer_stage3_3 import MinimalStreetForwardStage3_3

        return MinimalStreetForwardStage3_3
    if name == "MinimalStreetForwardStage4_0":
        from models.streetforward.minimal_trainer_stage4_0 import MinimalStreetForwardStage4_0

        return MinimalStreetForwardStage4_0
    if name == "MinimalStreetForwardStage4_1":
        from models.streetforward.minimal_trainer_stage4_1 import MinimalStreetForwardStage4_1

        return MinimalStreetForwardStage4_1
    if name == "MinimalStreetForwardStage4_2":
        from models.streetforward.minimal_trainer_stage4_2 import MinimalStreetForwardStage4_2

        return MinimalStreetForwardStage4_2
    if name == "MinimalStreetForwardStage4_3":
        from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3

        return MinimalStreetForwardStage4_3
    if name == "MinimalStreetForwardStage4_4":
        from models.streetforward.minimal_trainer_stage4_4 import MinimalStreetForwardStage4_4

        return MinimalStreetForwardStage4_4
    if name == "MinimalStreetForwardStage4_5":
        from models.streetforward.minimal_trainer_stage4_5 import MinimalStreetForwardStage4_5

        return MinimalStreetForwardStage4_5
    if name == "MinimalStreetForwardStage4_6":
        from models.streetforward.minimal_trainer_stage4_6 import MinimalStreetForwardStage4_6

        return MinimalStreetForwardStage4_6
    if name in {"NodeState", "NodeStateBackground", "NodeStateRigid", "NodeStateDistant"}:
        from models.streetforward import node_states as _ns

        return getattr(_ns, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
