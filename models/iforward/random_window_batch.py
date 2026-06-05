from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

ImageRef = Tuple[int, int]

RANDOM_WINDOW_SCHEDULER_VERSION = "random_window_v1"
RANDOM_WINDOW_ASSEMBLY_MODE = "image_ref_iforward_random_window_v1"
RANDOM_WINDOW_MODEL_FAMILY = "IForward"


@dataclass(frozen=True)
class IForwardRandomWindowStep:
    step_idx: int
    block_id: int
    block_pos_in_window: int
    repeat_idx: int
    global_k: int
    source_frame_idx: int
    source_keyframe_idx: int
    evidence_refs: List[ImageRef]
    commit_observation_memory: bool
    update_optimizer_memory: bool
    is_frame_exit: bool
    source_indices: List[int] = field(default_factory=list)
    rollout_pos_code: float = 0.0
    frame_pos_code: float = 0.0
    repeat_pos_code: float = 0.0


@dataclass(frozen=True)
class IForwardRandomWindowPlan:
    scheduler_version: str
    model_family: str
    scene_id: int
    segment_id: int
    episode_id: int
    rollout_id_global: int
    rollout_idx_in_episode: int
    rollouts_per_episode: int
    window_start: int
    window_end: int
    window_block_ids: List[int]
    window_keyframe_indices: List[int]
    window_frame_indices: List[int]
    window_hash: int
    window_revisit_count: int
    unique_windows_seen: int
    is_repeated_window: bool
    blocks_per_rollout: int
    repeats_per_block: int
    inner_K: int
    reset_scene_state_before_rollout: bool
    carry_scene_state_after_rollout: bool
    episode_end_after_rollout: bool
    detach_graph_after_rollout: bool
    steps: List[IForwardRandomWindowStep]
    evidence_refs_flat: List[ImageRef]
    target_refs_flat: List[ImageRef]
    target_roles_flat: List[str]
    current_latest_refs: List[ImageRef]
    in_rollout_history_refs: List[ImageRef]
    short_window_history_refs: List[ImageRef]
    nearby_refs: List[ImageRef]
    input_frame_indices: List[int]
    input_keyframe_indices: List[int]
    nearby_frame_indices: List[int]
    request_meta: Dict[str, Any] = field(default_factory=dict)
    leakage_check: Dict[str, Any] = field(default_factory=dict)
