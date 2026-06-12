from __future__ import annotations

from .mamba import StreamingMambaCell, StreamingMambaCellState
from .iforward_v6_state import IForwardV6BranchPointState, IForwardV6MemoryState
from .resolver import (
    IFORWARD_CURRENT_ROLE,
    IFORWARD_HISTORY_ROLE,
    IFORWARD_NEARBY_ROLE,
    IFORWARD_SCHEDULER_VERSION,
    IFORWARD_V3_SCHEDULER_VERSION,
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
    if name in {
        "IForwardGRUBranchPrepared",
        "IForwardGRUBranchState",
        "IForwardGRUMemoryState",
        "IForwardGRUPrepared",
        "IForwardTimeAwarePointGRU",
    }:
        from .gru_memory import (
            IForwardGRUBranchPrepared,
            IForwardGRUBranchState,
            IForwardGRUMemoryState,
            IForwardGRUPrepared,
            IForwardTimeAwarePointGRU,
        )

        return {
            "IForwardGRUBranchPrepared": IForwardGRUBranchPrepared,
            "IForwardGRUBranchState": IForwardGRUBranchState,
            "IForwardGRUMemoryState": IForwardGRUMemoryState,
            "IForwardGRUPrepared": IForwardGRUPrepared,
            "IForwardTimeAwarePointGRU": IForwardTimeAwarePointGRU,
        }[name]
    if name in {"IForwardHistoryBranchEMA", "IForwardHistoryEMAState", "IForwardResidualPack"}:
        from .history_ema import IForwardHistoryBranchEMA, IForwardHistoryEMAState, IForwardResidualPack

        return {
            "IForwardHistoryBranchEMA": IForwardHistoryBranchEMA,
            "IForwardHistoryEMAState": IForwardHistoryEMAState,
            "IForwardResidualPack": IForwardResidualPack,
        }[name]
    if name in {"IForwardAttributeGate", "IForwardGatePack", "IForwardHistoryGate"}:
        from .history_gate import IForwardAttributeGate, IForwardGatePack, IForwardHistoryGate

        return {
            "IForwardAttributeGate": IForwardAttributeGate,
            "IForwardGatePack": IForwardGatePack,
            "IForwardHistoryGate": IForwardHistoryGate,
        }[name]
    if name == "IForwardHistorySafeProjection":
        from .history_safe_projection import IForwardHistorySafeProjection

        return IForwardHistorySafeProjection
    if name in {"IForwardADCBank", "IForwardADCStateMeta"}:
        from .adc_lite import IForwardADCBank, IForwardADCStateMeta

        return {
            "IForwardADCBank": IForwardADCBank,
            "IForwardADCStateMeta": IForwardADCStateMeta,
        }[name]
    if name in {"GradientBankAttr", "HistoryGradientBank", "HistoryGradientBranchBank"}:
        from .history_gradient_bank import GradientBankAttr, HistoryGradientBank, HistoryGradientBranchBank

        return {
            "GradientBankAttr": GradientBankAttr,
            "HistoryGradientBank": HistoryGradientBank,
            "HistoryGradientBranchBank": HistoryGradientBranchBank,
        }[name]
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
    "GradientBankAttr",
    "IFORWARD_CURRENT_ROLE",
    "IFORWARD_HISTORY_ROLE",
    "IFORWARD_NEARBY_ROLE",
    "IFORWARD_SCHEDULER_VERSION",
    "IFORWARD_V3_SCHEDULER_VERSION",
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
    "IForwardAttributeGate",
    "IForwardADCBank",
    "IForwardADCStateMeta",
    "IForwardGatePack",
    "IForwardGRUBranchPrepared",
    "IForwardGRUBranchState",
    "IForwardGRUMemoryState",
    "IForwardGRUPrepared",
    "IForwardHistoryBranchEMA",
    "IForwardHistoryEMAState",
    "IForwardHistoryGate",
    "IForwardHistorySafeProjection",
    "IForwardModel",
    "IForwardResidualPack",
    "IForwardResolvedBatch",
    "IForwardResolvedStep",
    "IForwardRolloutOutput",
    "IForwardSceneMemory",
    "IForwardShortMemoryEntry",
    "IForwardShortWindowHistory",
    "IForwardStage6Bridge",
    "IForwardState",
    "IForwardTimeAwarePointGRU",
    "IForwardTrainer",
    "HistoryGradientBank",
    "HistoryGradientBranchBank",
    "KeyedMambaState",
    "StreamingMambaCell",
    "StreamingMambaCellState",
]
