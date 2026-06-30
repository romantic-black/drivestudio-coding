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

from .index_format import IFORWARD_STAGE2_3_SCHEDULER_VERSION, IFORWARD_STAGE3_0_SCHEDULER_VERSION


STAGE23_CURRENT_ROLE = "final_current_recon"
STAGE23_HISTORY_ROLE = "final_history_replay"


@dataclass(frozen=True)
class Stage23StepPlan(IForwardStepPlan):
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
    visit_order_gap: int = 0
    physical_frame_gap_abs: int = 0
    previous_visit_sequence_pos: int = -1
    ego_delta_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ego_delta_yaw: float = 0.0
    visit_memory_mask: bool = True
    repair_no_commit: bool = False
    repeat_budget: int = 1
    visit_count_for_frame: int = 0
    is_first_visit_of_frame: bool = False
    is_last_update_of_episode: bool = False
    global_update_idx_in_episode: int = -1
    optimizer_memory_read: bool = True
    optimizer_memory_write: bool = True
    time_since_same_frame_visit: float = 0.0
    validation_render_only: bool = False


@dataclass(frozen=True)
class RolloutPlanV3(IForwardRolloutPlan):
    phase_max_inner_k: int = 0
    sequence_target_frames: int = 0
    sequence_min_frames: int = 0
    sequence_allow_short: bool = False
    sequence_id: int = -1
    sequence_length: int = 0
    sequence_protocol: str = "optimizer_sequence_v1"
    sequence_stride: int = 0
    sequence_start_local_frame: int = -1
    sequence_block_ids: List[int] = field(default_factory=list)
    sequence_keyframe_indices: List[int] = field(default_factory=list)
    sequence_source_frame_indices: List[int] = field(default_factory=list)
    sequence_timestamps_us: List[int] = field(default_factory=list)
    sequence_positions: List[int] = field(default_factory=list)
    episode_positions: List[int] = field(default_factory=list)
    rollout_positions: List[int] = field(default_factory=list)
    history_positions: List[int] = field(default_factory=list)
    repair_positions: List[int] = field(default_factory=list)
    repeat_budgets: List[int] = field(default_factory=list)
    frame_gaps: List[int] = field(default_factory=list)
    visit_kinds: List[str] = field(default_factory=list)
    scheduler_phase: str = ""
    rollout_phase: str = ""
    repair_enabled: bool = False
    repair_round_idx: int = -1
    repair_pattern_name: str = ""
    repair_permutation_hash: int = -1
    temporal_read_count: int = 0
    temporal_commit_count: int = 0
    optimizer_memory_read_count: int = 0
    optimizer_memory_write_count: int = 0
    observation_commit_count: int = 0


@dataclass(frozen=True)
class EpisodePlanV3:
    scene_id: int
    segment_id: int
    episode_id: int
    sequence_id: int
    frame_set: Tuple[int, ...]
    keyframe_set: Tuple[int, ...]
    sampled_order: Tuple[int, ...]
    rollouts: Tuple[RolloutPlanV3, ...]
    repair_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def make_final_supervision_v3(
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
        nearby_skip_reason="stage2_3_disabled",
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
    "EpisodePlanV3",
    "IFORWARD_MODEL_FAMILY",
    "IFORWARD_STAGE2_3_SCHEDULER_VERSION",
    "IFORWARD_STAGE3_0_SCHEDULER_VERSION",
    "RolloutPlanV3",
    "STAGE23_CURRENT_ROLE",
    "STAGE23_HISTORY_ROLE",
    "Stage23StepPlan",
    "make_final_supervision_v3",
]
