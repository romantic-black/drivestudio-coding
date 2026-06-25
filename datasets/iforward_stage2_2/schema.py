from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from datasets.train_scheduler_iforward import (
    IFORWARD_MODEL_FAMILY,
    IForwardFinalSupervisionPlan,
    IForwardRolloutPlan,
    IForwardStepPlan,
    ImageRef,
)

from .index_format import IFORWARD_STAGE2_2_SCHEDULER_VERSION


STAGE22_CURRENT_ROLE = "final_current_recon"
STAGE22_HISTORY_ROLE = "final_history_replay"


@dataclass(frozen=True)
class ObservationSpec:
    scene_id: int
    segment_id: int
    raw_frame_idx: int
    keyframe_idx: int
    timestamp_us: int
    delta_t_sec: float
    frame_gap: int
    ego_delta_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ego_delta_yaw: float = 0.0
    visit_kind: str = "causal"
    sequence_pos: int = -1
    memory_read: bool = True
    memory_commit: bool = False


@dataclass(frozen=True)
class Stage22StepPlan(IForwardStepPlan):
    sequence_pos: int = -1
    visit_kind: str = ""
    frame_gap: int = 0
    temporal_read: bool = True
    temporal_commit: bool = False
    physical_time_advance: bool = False
    scheduler_phase: str = ""
    timestamp_us: int = 0
    timestamp_sec: float = 0.0
    delta_t_sec: float = 0.0
    ego_delta_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ego_delta_yaw: float = 0.0
    visit_memory_mask: bool = True
    repair_no_commit: bool = False


@dataclass(frozen=True)
class RolloutPlan(IForwardRolloutPlan):
    sequence_id: int = -1
    sequence_length: int = 10
    sequence_protocol: str = ""
    sequence_stride: int = 1
    sequence_start_local_frame: int = -1
    sequence_block_ids: List[int] = field(default_factory=list)
    sequence_keyframe_indices: List[int] = field(default_factory=list)
    sequence_source_frame_indices: List[int] = field(default_factory=list)
    sequence_timestamps_us: List[int] = field(default_factory=list)
    sequence_positions: List[int] = field(default_factory=list)
    history_positions: List[int] = field(default_factory=list)
    repair_positions: List[int] = field(default_factory=list)
    scheduler_phase: str = ""
    rollout_phase: str = ""
    repair_enabled: bool = False
    repair_permutation_hash: int = -1
    temporal_read_count: int = 0
    temporal_commit_count: int = 0
    observation_commit_count: int = 0
    optimizer_memory_update_count: int = 0


@dataclass(frozen=True)
class EpisodePlan:
    scene_id: int
    segment_id: int
    episode_id: int
    protocol: str
    sequence_id: int
    source_frame_indices: Tuple[int, ...]
    timestamps_us: Tuple[int, ...]
    rollouts: Tuple[RolloutPlan, ...]
    repair_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def make_final_supervision(
    *,
    refs: List[ImageRef],
    roles: List[str],
    current_frames: List[int],
    current_refs: List[ImageRef],
    history_frames: List[int],
    history_refs: List[ImageRef],
) -> IForwardFinalSupervisionPlan:
    return IForwardFinalSupervisionPlan(
        refs=[tuple(x) for x in refs],
        roles=[str(x) for x in roles],
        current_input_frames=[int(x) for x in current_frames],
        nearby_frames=[],
        skipped_nearby=True,
        nearby_skip_reason="stage2_2_disabled",
        current_ref_count=int(len(current_refs)),
        nearby_ref_count=0,
        current_frames=[int(x) for x in current_frames],
        current_refs=[tuple(x) for x in current_refs],
        history_frames=[int(x) for x in history_frames],
        history_refs=[tuple(x) for x in history_refs],
        history_ref_count_before_dedupe=int(len(history_refs)),
        history_skipped=bool(len(history_refs) == 0),
        history_skip_reason="" if history_refs else "no_seen_history",
        nearby_refs=[],
        nearby_block_id=-1,
        history_ref_count=int(len(history_refs)),
    )


__all__ = [
    "IFORWARD_MODEL_FAMILY",
    "IFORWARD_STAGE2_2_SCHEDULER_VERSION",
    "ObservationSpec",
    "EpisodePlan",
    "RolloutPlan",
    "STAGE22_CURRENT_ROLE",
    "STAGE22_HISTORY_ROLE",
    "Stage22StepPlan",
    "make_final_supervision",
]
