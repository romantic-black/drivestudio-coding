from __future__ import annotations

from .mamba import StreamingMambaCell, StreamingMambaCellState
from .resolver import (
    IFORWARD_CURRENT_ROLE,
    IFORWARD_NEARBY_ROLE,
    IFORWARD_SCHEDULER_VERSION,
    IForwardResolvedBatch,
    IForwardResolvedStep,
    IForwardBatchResolver,
)
from .state import (
    BranchMemoryState,
    DenseMambaState,
    IForwardMemoryState,
    IForwardShortMemoryEntry,
    IForwardShortWindowHistory,
    IForwardState,
    KeyedMambaState,
)


def __getattr__(name: str):
    if name == "IForwardSceneMemory":
        from .memory import IForwardSceneMemory

        return IForwardSceneMemory
    if name == "IForwardStage6Bridge":
        from .bridge import IForwardStage6Bridge

        return IForwardStage6Bridge
    if name in {"IForwardModel", "IForwardRolloutOutput"}:
        from .model import IForwardModel, IForwardRolloutOutput

        return {"IForwardModel": IForwardModel, "IForwardRolloutOutput": IForwardRolloutOutput}[name]
    if name == "IForwardTrainer":
        from .trainer import IForwardTrainer

        return IForwardTrainer
    raise AttributeError(name)

__all__ = [
    "BranchMemoryState",
    "DenseMambaState",
    "IFORWARD_CURRENT_ROLE",
    "IFORWARD_NEARBY_ROLE",
    "IFORWARD_SCHEDULER_VERSION",
    "IForwardBatchResolver",
    "IForwardMemoryState",
    "IForwardModel",
    "IForwardResolvedBatch",
    "IForwardResolvedStep",
    "IForwardRolloutOutput",
    "IForwardSceneMemory",
    "IForwardShortMemoryEntry",
    "IForwardShortWindowHistory",
    "IForwardStage6Bridge",
    "IForwardState",
    "IForwardTrainer",
    "KeyedMambaState",
    "StreamingMambaCell",
    "StreamingMambaCellState",
]
