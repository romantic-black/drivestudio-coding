from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

ImageRef = Tuple[int, int]  # (frame_idx, cam_id)
_ALLOWED_TEACHER_PURPOSES = {"train_update", "live_bridge"}
_ALLOWED_BRIDGE_MODES = {"live", "cache", "none"}


class Stage6StepType(str, Enum):
    TEACHER_BOOTSTRAP = "teacher_bootstrap"
    STUDENT_SELF = "student_self"
    STUDENT_ANCHOR = "student_anchor"
    STUDENT_HISTORY = "student_history"
    TEACHER_REFRESH = "teacher_refresh"


@dataclass
class Stage6TeacherObsRequest:
    enable: bool
    frame_idx: Optional[int] = None
    image_refs: List[ImageRef] = field(default_factory=list)
    purpose: str = "train_update"  # train_update | live_bridge
    # Safe defaults: explicit builders must opt-in writes.
    update_state: bool = False
    update_teacher_prior_cache: bool = False
    update_observed_history: bool = False
    update_runtime_history: bool = False
    use_gt_input: bool = True


@dataclass
class Stage6StudentPropRequest:
    enable: bool
    frame_idx: Optional[int] = None
    image_refs: List[ImageRef] = field(default_factory=list)
    bridge_mode: str = "none"  # live | cache | none
    require_live_bridge: bool = False
    update_state: bool = False
    update_runtime_history: bool = False
    update_observed_history: bool = False
    use_gt_input: bool = False


@dataclass
class Stage6SupervisionRequest:
    self_teacher_refs: List[ImageRef] = field(default_factory=list)
    self_student_refs: List[ImageRef] = field(default_factory=list)
    teacher_anchor_refs: List[ImageRef] = field(default_factory=list)
    history_visited_refs: List[ImageRef] = field(default_factory=list)
    probe_near_refs: List[ImageRef] = field(default_factory=list)
    enable_self_teacher: bool = False
    enable_self_student: bool = False
    enable_teacher_anchor: bool = False
    enable_history_visited: bool = False
    enable_probe_near: bool = False


@dataclass
class Stage6StepRequest:
    scheduler_version: str
    step_type: Stage6StepType
    scene_id: int
    segment_id: int
    block_idx: int
    step_idx_in_block: int
    global_scheduler_step: int
    teacher_obs: Stage6TeacherObsRequest
    student_prop: Stage6StudentPropRequest
    supervision: Stage6SupervisionRequest
    teacher_anchor_frame_idx: Optional[int] = None
    student_frame_idx: Optional[int] = None
    committed_history_frame_indices: List[int] = field(default_factory=list)
    probe_near_frame_indices: List[int] = field(default_factory=list)
    history_record: Dict[str, object] = field(default_factory=dict)
    preload_hints: Dict[str, object] = field(default_factory=dict)
    compat: Dict[str, object] = field(default_factory=dict)


@dataclass
class EpisodePlanV10:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int
    keyframe_window: List[int]
    frame_chain: List[int]
    num_cams: int
    block_keyframe_indices: List[int]
    teacher_frame_by_block: Dict[int, int]
    student_candidates_by_block: Dict[int, List[int]]
    probe_near_candidates_by_block: Dict[int, List[int]]


@dataclass
class EpisodeStateV10:
    committed_history_frame_indices: List[int] = field(default_factory=list)
    teacher_seen_blocks: List[int] = field(default_factory=list)
    student_seen_blocks: List[int] = field(default_factory=list)
    last_committed_block_idx: Optional[int] = None


@dataclass
class BlockStateV10:
    block_idx: int
    keyframe_idx: int
    teacher_frame_idx: int
    student_candidates: List[int]
    step_idx: int = 0
    teacher_seen: bool = False
    student_seen: bool = False
    runtime_updated_frames: List[int] = field(default_factory=list)
    last_teacher_frame_idx: Optional[int] = None
    last_student_frame_idx: Optional[int] = None
    fallback_no_student_count: int = 0
    fallback_no_history_count: int = 0
    probe_near_empty_count: int = 0


def validate_teacher_obs_invariants(obs: Stage6TeacherObsRequest) -> None:
    purpose = str(obs.purpose)
    if purpose not in _ALLOWED_TEACHER_PURPOSES:
        raise ValueError(f"unknown teacher_obs purpose={purpose!r}")
    if not bool(obs.enable):
        if obs.update_state or obs.update_teacher_prior_cache or obs.update_observed_history or obs.update_runtime_history:
            raise ValueError("disabled teacher_obs must not write state/cache/history.")
        return
    if obs.frame_idx is None:
        raise ValueError("enabled teacher_obs requires frame_idx.")
    if len(obs.image_refs) == 0:
        raise ValueError("enabled teacher_obs requires non-empty image_refs.")
    if purpose == "live_bridge":
        if obs.update_state or obs.update_teacher_prior_cache or obs.update_observed_history or obs.update_runtime_history:
            raise ValueError(
                "teacher_obs purpose=live_bridge forbids any state/cache/history writes; "
                f"got update_state={obs.update_state}, cache={obs.update_teacher_prior_cache}, "
                f"observed={obs.update_observed_history}, runtime={obs.update_runtime_history}"
            )
        if not bool(obs.use_gt_input):
            raise ValueError("live_bridge teacher_obs requires use_gt_input=true.")


def validate_student_prop_invariants(prop: Stage6StudentPropRequest) -> None:
    if str(prop.bridge_mode) not in _ALLOWED_BRIDGE_MODES:
        raise ValueError(f"unknown bridge_mode={prop.bridge_mode!r}")
    if not bool(prop.enable):
        if prop.update_state or prop.update_runtime_history or prop.update_observed_history:
            raise ValueError("disabled student_prop must not write state/history.")
        return
    if prop.frame_idx is None:
        raise ValueError("enabled student_prop requires frame_idx.")
    if len(prop.image_refs) == 0:
        raise ValueError("enabled student_prop requires non-empty image_refs.")
    if bool(prop.use_gt_input):
        raise ValueError("student_prop must not use GT input.")
    if bool(prop.update_observed_history):
        raise ValueError("student_prop must not update observed history.")
    if str(prop.bridge_mode) == "live" and not bool(prop.require_live_bridge):
        raise ValueError("student_prop bridge_mode=live requires require_live_bridge=true.")


def _check_domain(enable: bool, refs: List[ImageRef], name: str, *, allow_empty_when_enabled: bool = False) -> None:
    if bool(enable) and len(refs) == 0 and not bool(allow_empty_when_enabled):
        raise ValueError(f"{name} enabled but refs are empty.")
    if len(refs) > 0 and not bool(enable):
        raise ValueError(f"{name} refs provided but enable=false.")


def validate_supervision_invariants(sup: Stage6SupervisionRequest) -> None:
    _check_domain(sup.enable_self_teacher, sup.self_teacher_refs, "self_teacher")
    _check_domain(sup.enable_self_student, sup.self_student_refs, "self_student")
    _check_domain(sup.enable_teacher_anchor, sup.teacher_anchor_refs, "teacher_anchor")
    _check_domain(sup.enable_history_visited, sup.history_visited_refs, "history_visited")
    _check_domain(sup.enable_probe_near, sup.probe_near_refs, "probe_near", allow_empty_when_enabled=True)


def validate_stage6_step_request(req: Stage6StepRequest) -> None:
    validate_teacher_obs_invariants(req.teacher_obs)
    validate_student_prop_invariants(req.student_prop)
    validate_supervision_invariants(req.supervision)
    step_type = Stage6StepType(req.step_type)
    if step_type == Stage6StepType.TEACHER_BOOTSTRAP:
        if not req.teacher_obs.enable or str(req.teacher_obs.purpose) != "train_update":
            raise ValueError("teacher_bootstrap requires teacher_obs train_update enabled.")
        if not req.teacher_obs.update_state or not req.teacher_obs.update_teacher_prior_cache:
            raise ValueError("teacher_bootstrap requires teacher state/cache updates enabled.")
        if not req.teacher_obs.update_observed_history or not req.teacher_obs.update_runtime_history:
            raise ValueError("teacher_bootstrap requires teacher history updates enabled.")
        if req.student_prop.enable:
            raise ValueError("teacher_bootstrap must disable student_prop.")
        if not req.supervision.enable_self_teacher:
            raise ValueError("teacher_bootstrap requires self_teacher supervision.")
        if req.supervision.enable_self_student or req.supervision.enable_teacher_anchor or req.supervision.enable_history_visited:
            raise ValueError("teacher_bootstrap forbids self_student/teacher_anchor/history_visited supervision.")
    elif step_type == Stage6StepType.STUDENT_SELF:
        if not req.student_prop.enable:
            raise ValueError("student_self requires student_prop enabled.")
        if not req.student_prop.update_state or not req.student_prop.update_runtime_history or req.student_prop.update_observed_history:
            raise ValueError("student_self requires update_state+runtime and forbids observed writes.")
        if not req.supervision.enable_self_student:
            raise ValueError("student_self requires self_student supervision.")
        if req.supervision.enable_teacher_anchor or req.supervision.enable_history_visited:
            raise ValueError("student_self forbids teacher_anchor/history_visited supervision.")
    elif step_type == Stage6StepType.STUDENT_ANCHOR:
        if not req.student_prop.enable:
            raise ValueError("student_anchor requires student_prop enabled.")
        if not req.supervision.enable_self_student or not req.supervision.enable_teacher_anchor:
            raise ValueError("student_anchor requires self_student + teacher_anchor supervision.")
    elif step_type == Stage6StepType.STUDENT_HISTORY:
        if not req.student_prop.enable:
            raise ValueError("student_history requires student_prop enabled.")
        if not req.supervision.enable_self_student or not req.supervision.enable_history_visited:
            raise ValueError("student_history requires self_student + history_visited supervision.")
    elif step_type == Stage6StepType.TEACHER_REFRESH:
        if not req.teacher_obs.enable or str(req.teacher_obs.purpose) != "train_update":
            raise ValueError("teacher_refresh requires teacher_obs train_update enabled.")
        if req.student_prop.enable:
            raise ValueError("teacher_refresh must disable student_prop.")
