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
    if name in {"NodeState", "NodeStateBackground", "NodeStateRigid", "NodeStateDistant"}:
        from models.streetforward import node_states as _ns

        return getattr(_ns, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
