from __future__ import annotations

from .mamba import StreamingMambaCell, StreamingMambaCellState
from .iforward_v6_state import IForwardV6BranchPointState, IForwardV6MemoryState
from .resolver import (
    IFORWARD_CURRENT_ROLE,
    IFORWARD_NEARBY_ROLE,
    IFORWARD_SCHEDULER_VERSION,
    IForwardResolvedBatch,
    IForwardResolvedStep,
    IForwardBatchResolver,
)
from .random_window_batch import (
    RANDOM_WINDOW_ASSEMBLY_MODE,
    RANDOM_WINDOW_MODEL_FAMILY,
    RANDOM_WINDOW_SCHEDULER_VERSION,
    IForwardRandomWindowPlan,
    IForwardRandomWindowStep,
)
from .random_window_resolver import IForwardRandomWindowBatchResolver
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
    if name == "IForwardPointMambaMemory":
        from .point_mamba_memory import IForwardPointMambaMemory

        return IForwardPointMambaMemory
    if name == "IForwardLocalConflictXcpe":
        from .local_conflict_xcpe import IForwardLocalConflictXcpe

        return IForwardLocalConflictXcpe
    if name == "IForwardContextAdapter":
        from .context_adapter import IForwardContextAdapter

        return IForwardContextAdapter
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
    "RANDOM_WINDOW_ASSEMBLY_MODE",
    "RANDOM_WINDOW_MODEL_FAMILY",
    "RANDOM_WINDOW_SCHEDULER_VERSION",
    "IForwardBatchResolver",
    "IForwardRandomWindowBatchResolver",
    "IForwardRandomWindowPlan",
    "IForwardRandomWindowStep",
    "IForwardMemoryState",
    "IForwardV6BranchPointState",
    "IForwardV6MemoryState",
    "IForwardPointMambaMemory",
    "IForwardLocalConflictXcpe",
    "IForwardContextAdapter",
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
