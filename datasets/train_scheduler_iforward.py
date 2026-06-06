from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

ImageRef = Tuple[int, int]

IFORWARD_SCHEDULER_VERSION = "iforward_v1"
IFORWARD_V3_SCHEDULER_VERSION = "iforward_v3_random_window"
IFORWARD_MODEL_FAMILY = "IForward"


class TrainSchedulerIForwardDatasetLike(Protocol):
    _initialized: bool

    def initialize(self) -> None: ...
    def list_training_scene_ids(self) -> List[int]: ...
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> Any: ...


@dataclass(frozen=True)
class IForwardRolloutShape:
    name: str
    blocks_per_rollout: int
    repeats_per_block: int
    prob: float = 1.0

    @property
    def inner_K(self) -> int:
        return int(self.blocks_per_rollout) * int(self.repeats_per_block)


@dataclass(frozen=True)
class IForwardStepPlan:
    step_idx: int
    block_id: int
    episode_block_idx: int
    rollout_block_rank: int
    repeat_idx: int
    repeats_per_block: int
    is_block_enter: bool
    is_block_exit: bool
    source_keyframe_idx: int
    source_frame_idx: int
    evidence_refs: List[ImageRef]
    evidence_frame_indices: List[int]
    evidence_cam_indices: List[int]
    commit_observation_memory: bool
    update_optimizer_memory: bool
    detach_before_step: bool
    detach_after_step: bool
    allow_step_render_loss: bool
    step_loss_refs: List[ImageRef]
    rollout_pos_code: float
    frame_pos_code: float
    repeat_pos_code: float
    is_frame_exit: bool = False
    episode_visit_idx: int = -1
    rollout_visit_idx: int = -1
    optimizer_step_idx_in_episode: int = -1
    record_update_norm: bool = True
    commit_support_on_exit: bool = False
    commit_residual_on_exit: bool = False
    window_start: int = -1
    window_end: int = -1
    window_hash: int = -1
    window_revisit_count: int = 0
    is_repeated_window: bool = False
    block_visit_count_before: int = 0
    block_visit_count_after: int = 0


@dataclass(frozen=True)
class IForwardFinalSupervisionPlan:
    refs: List[ImageRef]
    roles: List[str]
    current_input_frames: List[int]
    nearby_frames: List[int]
    skipped_nearby: bool
    nearby_skip_reason: str
    current_ref_count: int
    nearby_ref_count: int
    current_frames: List[int] = field(default_factory=list)
    current_refs: List[ImageRef] = field(default_factory=list)
    history_frames: List[int] = field(default_factory=list)
    history_refs: List[ImageRef] = field(default_factory=list)
    history_ref_count_before_dedupe: int = 0
    history_skipped: bool = True
    history_skip_reason: str = ""
    nearby_refs: List[ImageRef] = field(default_factory=list)
    nearby_block_id: int = -1
    history_ref_count: int = 0


@dataclass(frozen=True)
class IForwardRolloutPlan:
    scheduler_version: str
    scene_id: int
    segment_id: int
    episode_id: int
    rollout_id_global: int
    rollout_idx_in_episode: int
    episode_start_keyframe_pos: int
    keyframe_window: List[int]
    frame_chain: List[int]
    num_cams: int
    shape_name: str
    blocks_per_rollout: int
    repeats_per_block: int
    requested_blocks_per_rollout: int
    actual_blocks_per_rollout: int
    requested_inner_K: int
    actual_inner_K: int
    short_rollout: bool
    short_rollout_reason: str
    episode_block_indices: List[int]
    input_keyframe_indices: List[int]
    input_frame_indices: List[int]
    delivery_frame_indices: List[int]
    delivery_order_policy: str
    inner_K: int
    steps: List[IForwardStepPlan]
    final_supervision: IForwardFinalSupervisionPlan
    reset_scene_state_before_rollout: bool
    carry_scene_state_after_rollout: bool
    episode_end_after_rollout: bool
    detach_graph_after_rollout: bool
    evidence_refs_flat: List[ImageRef]
    target_refs_flat: List[ImageRef]
    target_roles_flat: List[str]
    request_meta: Dict[str, Any] = field(default_factory=dict)
    leakage_check: Dict[str, Any] = field(default_factory=dict)
    tail_skipped_after_rollout: bool = False
    tail_skipped_remaining_blocks: int = 0
    model_family: str = IFORWARD_MODEL_FAMILY
    rollouts_per_episode: int = 1
    episode_num_blocks: int = 0
    window_policy: str = ""
    window_start: int = -1
    window_end: int = -1
    window_block_ids: List[int] = field(default_factory=list)
    window_keyframe_indices: List[int] = field(default_factory=list)
    window_frame_indices: List[int] = field(default_factory=list)
    window_hash: int = -1
    window_revisit_count: int = 0
    unique_windows_seen: int = 0
    is_repeated_window: bool = False


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        out = node.get(key, default)
        return default if out is None else out
    if hasattr(node, key):
        out = getattr(node, key)
        return default if out is None else out
    return default


def _dedupe_refs_keep_order(refs: Sequence[ImageRef]) -> List[ImageRef]:
    seen: set[ImageRef] = set()
    out: List[ImageRef] = []
    for ref in refs:
        r = (int(ref[0]), int(ref[1]))
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _stable_window_hash(scene_id: int, segment_id: int, block_ids: Sequence[int]) -> int:
    text = f"{int(scene_id)}:{int(segment_id)}:" + ",".join(str(int(x)) for x in block_ids)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _dedupe_refs_roles_keep_order(
    refs: Sequence[ImageRef],
    roles: Sequence[str],
) -> Tuple[List[ImageRef], List[str]]:
    if len(refs) != len(roles):
        raise ValueError(f"IForward refs/roles length mismatch: {len(refs)} vs {len(roles)}")
    seen: set[ImageRef] = set()
    role_by_ref: Dict[ImageRef, str] = {}
    out_refs: List[ImageRef] = []
    out_roles: List[str] = []
    for ref, role in zip(refs, roles):
        r = (int(ref[0]), int(ref[1]))
        role_s = str(role)
        prev = role_by_ref.get(r)
        if prev is not None and prev != role_s:
            raise ValueError(f"IForward target ref {r} has conflicting roles: {prev} vs {role_s}")
        role_by_ref[r] = role_s
        if r in seen:
            continue
        seen.add(r)
        out_refs.append(r)
        out_roles.append(role_s)
    return out_refs, out_roles


class TrainSchedulerIForward:
    """Standalone IForward short-sequence rollout scheduler.

    IForward v1 intentionally supports the documented full short-sequence
    contract: episode-serial traversal, contiguous chronological rollouts,
    all-camera evidence, and rollout-final supervision.
    """

    def __init__(
        self,
        *,
        dataset: TrainSchedulerIForwardDatasetLike,
        episode_cfg: Optional[Any] = None,
        rollout_cfg: Optional[Any] = None,
        traversal_cfg: Optional[Any] = None,
        evidence_cfg: Optional[Any] = None,
        supervision_cfg: Optional[Any] = None,
        memory_cfg: Optional[Any] = None,
        loss_timing_cfg: Optional[Any] = None,
        leakage_check_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        include_test: bool = False,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        seed: Optional[int] = None,
        version: str = IFORWARD_SCHEDULER_VERSION,
        fail_fast: bool = True,
    ) -> None:
        self.dataset = dataset
        self.episode_cfg = dict(episode_cfg or {})
        self.rollout_cfg = dict(rollout_cfg or {})
        self.traversal_cfg = dict(traversal_cfg or {})
        self.evidence_cfg = dict(evidence_cfg or {})
        self.supervision_cfg = dict(supervision_cfg or {})
        self.memory_cfg = dict(memory_cfg or {})
        self.loss_timing_cfg = dict(loss_timing_cfg or {})
        self.leakage_check_cfg = dict(leakage_check_cfg or {})
        self.preload_cfg = dict(preload_cfg or {})
        self.include_test = bool(include_test)
        self.version = str(version)
        if fixed_scene_id is None:
            fixed_scene_id = _cfg_get(self.traversal_cfg, "fixed_scene_id", None)
        if fixed_segment_id is None:
            fixed_segment_id = _cfg_get(self.traversal_cfg, "fixed_segment_id", None)
        self.fixed_scene_id = int(fixed_scene_id) if fixed_scene_id is not None else None
        self.fixed_segment_id = int(fixed_segment_id) if fixed_segment_id is not None else None
        raw_seed = seed if seed is not None else _cfg_get(self.traversal_cfg, "seed", None)
        self.rng = random.Random(int(raw_seed)) if raw_seed is not None else random.Random()
        self.fail_fast = bool(fail_fast)

        initialized = getattr(self.dataset, "_initialized", True)
        if initialized is False:
            self.dataset.initialize()

        self._validate_static_cfg()

        self.global_step = 0
        self.epoch_idx = 0
        self._episode_id_next = 0
        self._rollout_id_global = 0
        self._episode_plan: List[Dict[str, Any]] = []
        self._episode_plan_cursor = 0
        self._current_episode: Optional[Dict[str, Any]] = None
        self._pending_events: List[Dict[str, Any]] = []
        self._last_info: Dict[str, Any] = {
            "scheduler_version": str(self.version),
            "model_family": IFORWARD_MODEL_FAMILY,
            "global_step": 0,
        }
        self._rebuild_epoch_plan()

    def _is_v3(self) -> bool:
        return str(self.version) == IFORWARD_V3_SCHEDULER_VERSION

    def _scheduler_version(self) -> str:
        return IFORWARD_V3_SCHEDULER_VERSION if self._is_v3() else IFORWARD_SCHEDULER_VERSION

    def _validate_static_cfg(self) -> None:
        version = str(
            self.version
            or _cfg_get(self.rollout_cfg, "version", _cfg_get(self.traversal_cfg, "version", IFORWARD_SCHEDULER_VERSION))
        )
        if version in (IFORWARD_V3_SCHEDULER_VERSION,):
            self._validate_static_cfg_v3()
            return
        if version not in (IFORWARD_SCHEDULER_VERSION, "iforward_v1"):
            raise ValueError(
                f"scheduler_iforward.version must be {IFORWARD_SCHEDULER_VERSION!r} "
                f"or {IFORWARD_V3_SCHEDULER_VERSION!r}, got {version!r}"
            )

        traversal_mode = str(_cfg_get(self.traversal_cfg, "traversal_mode", _cfg_get(self.traversal_cfg, "mode", "episode_serial")))
        if traversal_mode != "episode_serial":
            raise ValueError("scheduler_iforward IForward v1 requires traversal.traversal_mode=episode_serial")
        for name in ("scene_order", "segment_order"):
            val = str(_cfg_get(self.traversal_cfg, name, "shuffle_per_epoch" if name == "scene_order" else "ascending"))
            if val not in ("ascending", "shuffle_per_epoch"):
                raise ValueError(f"scheduler_iforward.traversal.{name} must be ascending or shuffle_per_epoch")

        if str(_cfg_get(self.episode_cfg, "source_mode", "keyframes")) != "keyframes":
            raise ValueError("scheduler_iforward IForward v1 requires episode.source_mode=keyframes")
        block_source_frame_policy = str(
            _cfg_get(self.episode_cfg, "block_source_frame_policy", "random_within_keyframe_once_per_episode")
        )
        if block_source_frame_policy not in (
            "random_within_keyframe_once_per_episode",
            "random_within_keyframe_per_rollout",
        ):
            raise ValueError(
                "scheduler_iforward IForward v1 requires "
                "episode.block_source_frame_policy=random_within_keyframe_once_per_episode "
                "or random_within_keyframe_per_rollout"
            )
        if int(_cfg_get(self.episode_cfg, "blocks_per_episode", 8)) < 1:
            raise ValueError("scheduler_iforward.episode.blocks_per_episode must be >= 1")
        if int(_cfg_get(self.episode_cfg, "episode_stride", _cfg_get(self.episode_cfg, "blocks_per_episode", 8))) < 1:
            raise ValueError("scheduler_iforward.episode.episode_stride must be >= 1")
        if int(_cfg_get(self.episode_cfg, "min_blocks_per_episode", 2)) < 1:
            raise ValueError("scheduler_iforward.episode.min_blocks_per_episode must be >= 1")
        rollouts_per_episode = _cfg_get(self.episode_cfg, "rollouts_per_episode", None)
        if rollouts_per_episode is not None and int(rollouts_per_episode) < 1:
            raise ValueError("scheduler_iforward.episode.rollouts_per_episode must be >= 1")

        block_selection_policy = str(_cfg_get(self.rollout_cfg, "block_selection_policy", "next_contiguous"))
        if block_selection_policy not in ("next_contiguous", "random_start_contiguous"):
            raise ValueError(
                "scheduler_iforward IForward v1 requires rollout.block_selection_policy="
                "next_contiguous or random_start_contiguous"
            )
        if str(_cfg_get(self.rollout_cfg, "delivery_order_policy", "chronological")) != "chronological":
            raise ValueError("scheduler_iforward IForward v1 requires rollout.delivery_order_policy=chronological")
        if int(_cfg_get(self.rollout_cfg, "min_blocks_per_rollout", 1)) < 1:
            raise ValueError("scheduler_iforward.rollout.min_blocks_per_rollout must be >= 1")
        if not bool(_cfg_get(self.rollout_cfg, "detach_graph_after_rollout", True)):
            raise ValueError("scheduler_iforward IForward v1 requires rollout.detach_graph_after_rollout=true")
        self._validate_shapes(self._raw_rollout_shapes(), label="rollout.shapes")
        for stage in list(_cfg_get(self.rollout_cfg, "shapes_schedule", []) or []):
            self._validate_shapes(list(_cfg_get(stage, "shapes", []) or []), label="rollout.shapes_schedule.shapes")

        if str(_cfg_get(self.evidence_cfg, "camera_policy", "all_cams")) != "all_cams":
            raise ValueError("scheduler_iforward IForward v1 requires evidence.camera_policy=all_cams")
        if bool(_cfg_get(self.evidence_cfg, "allow_camera_dropout", False)):
            raise ValueError("scheduler_iforward IForward v1 requires evidence.allow_camera_dropout=false")

        if str(_cfg_get(self.loss_timing_cfg, "policy", "rollout_final_only")) != "rollout_final_only":
            raise ValueError("scheduler_iforward IForward v1 requires loss_timing.policy=rollout_final_only")
        if bool(_cfg_get(self.loss_timing_cfg, "intermediate_step_loss", False)):
            raise ValueError("scheduler_iforward IForward v1 requires loss_timing.intermediate_step_loss=false")

        memory_commit = str(_cfg_get(self.memory_cfg, "observation_commit_policy", "first_repeat_only"))
        memory_update = str(_cfg_get(self.memory_cfg, "optimizer_memory_update_policy", "every_repeat"))
        reset_policy = str(_cfg_get(self.memory_cfg, "reset_policy", "episode_begin"))
        carry_policy = str(_cfg_get(self.memory_cfg, "carry_policy", "across_rollouts_until_episode_end"))
        if memory_commit != "first_repeat_only":
            raise ValueError("scheduler_iforward IForward v1 requires memory.observation_commit_policy=first_repeat_only")
        if memory_update != "every_repeat":
            raise ValueError("scheduler_iforward IForward v1 requires memory.optimizer_memory_update_policy=every_repeat")
        if reset_policy != "episode_begin":
            raise ValueError("scheduler_iforward IForward v1 requires memory.reset_policy=episode_begin")
        if carry_policy != "across_rollouts_until_episode_end":
            raise ValueError("scheduler_iforward IForward v1 requires memory.carry_policy=across_rollouts_until_episode_end")

        current = dict(_cfg_get(self.supervision_cfg, "current", {}) or {})
        nearby = dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {})
        history = dict(_cfg_get(self.supervision_cfg, "history_replay", {}) or {})
        if not bool(_cfg_get(current, "enable", True)):
            raise ValueError("scheduler_iforward IForward v1 requires supervision.current.enable=true")
        if str(_cfg_get(current, "frame_policy", "all_input_frames")) != "all_input_frames":
            raise ValueError("scheduler_iforward IForward v1 requires supervision.current.frame_policy=all_input_frames")
        if str(_cfg_get(current, "camera_policy", "all_cams")) != "all_cams":
            raise ValueError("scheduler_iforward IForward v1 requires supervision.current.camera_policy=all_cams")
        if bool(_cfg_get(history, "enable", False)):
            raise ValueError("scheduler_iforward IForward v1 requires supervision.history_replay.enable=false")
        if bool(_cfg_get(nearby, "enable", True)):
            if str(_cfg_get(nearby, "scope", "rollout_keyframe_span")) != "rollout_keyframe_span":
                raise ValueError("scheduler_iforward IForward v1 requires supervision.nearby.scope=rollout_keyframe_span")
            if str(_cfg_get(nearby, "policy", "random_non_input_frames")) != "random_non_input_frames":
                raise ValueError("scheduler_iforward IForward v1 requires supervision.nearby.policy=random_non_input_frames")
            if str(_cfg_get(nearby, "insufficient_policy", "use_available_or_skip_if_none")) != "use_available_or_skip_if_none":
                raise ValueError(
                    "scheduler_iforward IForward v1 requires "
                    "supervision.nearby.insufficient_policy=use_available_or_skip_if_none"
                )
            if str(_cfg_get(nearby, "camera_policy", "all_cams")) != "all_cams":
                raise ValueError("scheduler_iforward IForward v1 requires supervision.nearby.camera_policy=all_cams")
            if bool(_cfg_get(nearby, "add_to_evidence", False)):
                raise ValueError("scheduler_iforward IForward v1 requires supervision.nearby.add_to_evidence=false")

        leakage_enable = bool(_cfg_get(self.leakage_check_cfg, "enable", True))
        if not leakage_enable:
            raise ValueError("scheduler_iforward IForward v1 requires leakage_check.enable=true")

    def _validate_static_cfg_v3(self) -> None:
        traversal_mode = str(_cfg_get(self.traversal_cfg, "traversal_mode", _cfg_get(self.traversal_cfg, "mode", "episode_serial")))
        if traversal_mode != "episode_serial":
            raise ValueError("scheduler_iforward v3 requires traversal.traversal_mode=episode_serial")
        for name in ("scene_order", "segment_order"):
            val = str(_cfg_get(self.traversal_cfg, name, "shuffle_per_epoch"))
            if val not in ("ascending", "shuffle_per_epoch"):
                raise ValueError(f"scheduler_iforward.traversal.{name} must be ascending or shuffle_per_epoch")

        if str(_cfg_get(self.episode_cfg, "source_mode", "keyframes")) != "keyframes":
            raise ValueError("scheduler_iforward v3 requires episode.source_mode=keyframes")
        frame_policy = str(_cfg_get(self.episode_cfg, "block_source_frame_policy", "random_within_keyframe_per_visit"))
        if frame_policy != "random_within_keyframe_per_visit":
            raise ValueError(
                "scheduler_iforward v3 requires "
                "episode.block_source_frame_policy=random_within_keyframe_per_visit"
            )
        if int(_cfg_get(self.episode_cfg, "blocks_per_episode", 8)) < 1:
            raise ValueError("scheduler_iforward.episode.blocks_per_episode must be >= 1")
        if int(_cfg_get(self.episode_cfg, "episode_stride", _cfg_get(self.episode_cfg, "blocks_per_episode", 8))) < 1:
            raise ValueError("scheduler_iforward.episode.episode_stride must be >= 1")
        if int(_cfg_get(self.episode_cfg, "min_blocks_per_episode", 1)) < 1:
            raise ValueError("scheduler_iforward.episode.min_blocks_per_episode must be >= 1")
        if int(_cfg_get(self.episode_cfg, "rollouts_per_episode", 8)) < 1:
            raise ValueError("scheduler_iforward.episode.rollouts_per_episode must be >= 1")

        window_policy = str(_cfg_get(self.rollout_cfg, "window_policy", "random_with_replacement"))
        if window_policy not in ("random_with_replacement", "fixed_random_with_replacement"):
            raise ValueError("scheduler_iforward v3 requires rollout.window_policy=random_with_replacement")
        delivery = str(_cfg_get(self.rollout_cfg, "delivery_order_policy", "chronological_inside_window"))
        if delivery not in ("chronological_inside_window", "chronological"):
            raise ValueError("scheduler_iforward v3 requires rollout.delivery_order_policy=chronological_inside_window")
        if int(_cfg_get(self.rollout_cfg, "min_blocks_per_rollout", 1)) < 1:
            raise ValueError("scheduler_iforward.rollout.min_blocks_per_rollout must be >= 1")
        if not bool(_cfg_get(self.rollout_cfg, "detach_graph_after_rollout", True)):
            raise ValueError("scheduler_iforward v3 requires rollout.detach_graph_after_rollout=true")
        self._validate_shapes(self._raw_rollout_shapes(), label="rollout.shapes")
        for stage in list(_cfg_get(self.rollout_cfg, "shapes_schedule", []) or []):
            self._validate_shapes(list(_cfg_get(stage, "shapes", []) or []), label="rollout.shapes_schedule.shapes")

        if str(_cfg_get(self.evidence_cfg, "camera_policy", "all_cams")) != "all_cams":
            raise ValueError("scheduler_iforward v3 requires evidence.camera_policy=all_cams")
        if bool(_cfg_get(self.evidence_cfg, "allow_camera_dropout", False)):
            raise ValueError("scheduler_iforward v3 requires evidence.allow_camera_dropout=false")
        if str(_cfg_get(self.loss_timing_cfg, "policy", "rollout_final_only")) != "rollout_final_only":
            raise ValueError("scheduler_iforward v3 requires loss_timing.policy=rollout_final_only")
        if bool(_cfg_get(self.loss_timing_cfg, "intermediate_step_loss", False)):
            raise ValueError("scheduler_iforward v3 requires loss_timing.intermediate_step_loss=false")

        memory_commit = str(_cfg_get(self.memory_cfg, "observation_commit_policy", "first_repeat_only"))
        memory_update = str(_cfg_get(self.memory_cfg, "optimizer_memory_update_policy", "every_repeat"))
        reset_policy = str(_cfg_get(self.memory_cfg, "reset_policy", "episode_begin"))
        carry_policy = str(_cfg_get(self.memory_cfg, "carry_policy", "across_rollouts_until_episode_end"))
        if memory_commit != "first_repeat_only" or memory_update != "every_repeat":
            raise ValueError("scheduler_iforward v3 requires first-repeat observation commit and every-repeat optimizer update")
        if reset_policy != "episode_begin" or carry_policy != "across_rollouts_until_episode_end":
            raise ValueError("scheduler_iforward v3 requires episode_begin reset and episode carry")

        current = dict(_cfg_get(self.supervision_cfg, "current", {}) or {})
        if not bool(_cfg_get(current, "enable", True)):
            raise ValueError("scheduler_iforward v3 requires supervision.current.enable=true")
        current_policy = str(_cfg_get(current, "frame_policy", "all_rollout_input_frames"))
        if current_policy not in ("all_rollout_input_frames", "all_input_frames"):
            raise ValueError("scheduler_iforward v3 requires current.frame_policy=all_rollout_input_frames")
        nearby = dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {})
        if bool(_cfg_get(nearby, "enable", True)):
            scope = str(_cfg_get(nearby, "scope", "current_rollout_random_block"))
            policy = str(_cfg_get(nearby, "policy", "random_unsupervised_frame_in_random_rollout_block"))
            if scope != "current_rollout_random_block" or policy != "random_unsupervised_frame_in_random_rollout_block":
                raise ValueError("scheduler_iforward v3 requires current-rollout random-block nearby supervision")
        if not bool(_cfg_get(self.leakage_check_cfg, "enable", True)):
            raise ValueError("scheduler_iforward v3 requires leakage_check.enable=true")

    def _raw_rollout_shapes(self) -> List[Dict[str, Any]]:
        raw = [dict(x) for x in list(_cfg_get(self.rollout_cfg, "shapes", []) or [])]
        if raw:
            return raw
        if self._is_v3():
            return [
                {"name": "r2b1", "blocks_per_rollout": 1, "repeats_per_block": 2, "prob": 0.20},
                {"name": "r4b1", "blocks_per_rollout": 1, "repeats_per_block": 4, "prob": 0.30},
                {"name": "r6b1", "blocks_per_rollout": 1, "repeats_per_block": 6, "prob": 0.25},
                {"name": "r8b1", "blocks_per_rollout": 1, "repeats_per_block": 8, "prob": 0.25},
            ]
        return [
            {"name": "b2_r8", "blocks_per_rollout": 2, "repeats_per_block": 8, "prob": 0.50},
            {"name": "b3_r6", "blocks_per_rollout": 3, "repeats_per_block": 6, "prob": 0.30},
            {"name": "b4_r4", "blocks_per_rollout": 4, "repeats_per_block": 4, "prob": 0.20},
        ]

    def _validate_shapes(self, raw_shapes: Sequence[Any], *, label: str) -> None:
        if not raw_shapes:
            raise ValueError(f"scheduler_iforward.{label} must not be empty")
        total = 0.0
        max_inner_k = int(_cfg_get(self.rollout_cfg, "max_inner_K", 0) or 0)
        for idx, raw in enumerate(list(raw_shapes)):
            item = dict(raw)
            blocks = int(_cfg_get(item, "blocks_per_rollout", 0) or 0)
            repeats = int(_cfg_get(item, "repeats_per_block", 0) or 0)
            prob = float(_cfg_get(item, "prob", 1.0))
            if blocks < 1 or repeats < 1:
                raise ValueError(f"scheduler_iforward.{label}[{idx}] blocks/repeats must be >= 1")
            if not math.isfinite(prob) or prob < 0.0:
                raise ValueError(f"scheduler_iforward.{label}[{idx}].prob must be finite and >= 0")
            inner_k = int(blocks * repeats)
            if max_inner_k > 0 and inner_k > max_inner_k:
                raise ValueError(
                    f"scheduler_iforward.{label}[{idx}] inner_K={inner_k} exceeds max_inner_K={max_inner_k}"
                )
            total += float(prob)
        if total <= 0.0:
            raise ValueError(f"scheduler_iforward.{label} probabilities must sum to > 0")

    def _parse_shapes(self, raw_shapes: Sequence[Any]) -> List[IForwardRolloutShape]:
        out: List[IForwardRolloutShape] = []
        for raw in list(raw_shapes):
            item = dict(raw)
            blocks = int(_cfg_get(item, "blocks_per_rollout", 0) or 0)
            repeats = int(_cfg_get(item, "repeats_per_block", 0) or 0)
            out.append(
                IForwardRolloutShape(
                    name=str(_cfg_get(item, "name", f"b{blocks}_r{repeats}")),
                    blocks_per_rollout=blocks,
                    repeats_per_block=repeats,
                    prob=float(_cfg_get(item, "prob", 1.0)),
                )
            )
        return out

    def _active_shapes(self) -> List[IForwardRolloutShape]:
        raw = self._raw_rollout_shapes()
        active_start = None
        for stage in sorted(list(_cfg_get(self.rollout_cfg, "shapes_schedule", []) or []), key=lambda x: int(_cfg_get(x, "start_step", 0))):
            start = int(_cfg_get(stage, "start_step", 0) or 0)
            if int(self.global_step) < start:
                continue
            if active_start is None or start >= active_start:
                active_start = start
                raw = [dict(x) for x in list(_cfg_get(stage, "shapes", []) or [])]
        self._validate_shapes(raw, label="active rollout.shapes")
        return self._parse_shapes(raw)

    def _sample_shape_from(self, shapes: Sequence[IForwardRolloutShape]) -> IForwardRolloutShape:
        return self._sample_shape_from_rng(shapes, self.rng)

    def _sample_shape_for_episode(
        self,
        episode: Optional[Dict[str, Any]] = None,
        *,
        rng: Optional[random.Random] = None,
    ) -> IForwardRolloutShape:
        shapes = self._active_shapes()
        fixed_names = [str(x) for x in list(_cfg_get(self.rollout_cfg, "fixed_shape_names", []) or [])]
        if fixed_names and episode is not None:
            rollout_idx = int(episode.get("rollout_idx_in_episode", 0))
            if rollout_idx < len(fixed_names):
                requested = str(fixed_names[rollout_idx])
                by_name = {str(shape.name): shape for shape in shapes}
                if requested not in by_name:
                    raise ValueError(f"scheduler_iforward fixed_shape_names requested unknown shape {requested!r}")
                return by_name[requested]
        return self._sample_shape_from_rng(shapes, rng or self.rng)

    @staticmethod
    def _sample_shape_from_rng(shapes: Sequence[IForwardRolloutShape], rng: random.Random) -> IForwardRolloutShape:
        vals = list(shapes)
        if not vals:
            raise ValueError("scheduler_iforward cannot sample from an empty rollout shape list")
        return rng.choices(vals, weights=[float(x.prob) for x in vals], k=1)[0]

    def _sample_shape(self) -> IForwardRolloutShape:
        return self._sample_shape_from(self._active_shapes())

    def _block_selection_policy(self) -> str:
        return str(_cfg_get(self.rollout_cfg, "block_selection_policy", "next_contiguous"))

    def _block_source_frame_policy(self) -> str:
        return str(_cfg_get(self.episode_cfg, "block_source_frame_policy", "random_within_keyframe_once_per_episode"))

    def _sample_shape_for_remaining(self, remaining_blocks: int) -> IForwardRolloutShape:
        shapes = self._active_shapes()
        remaining = int(remaining_blocks)
        if remaining < 1:
            raise ValueError("remaining_blocks must be >= 1")
        allow_short = bool(_cfg_get(self.rollout_cfg, "allow_short_final_rollout", True))
        if not allow_short:
            valid_full = [shape for shape in shapes if int(shape.blocks_per_rollout) <= remaining]
            if valid_full:
                return self._sample_shape_from(valid_full)
            return self._sample_shape_from(shapes)

        if not bool(_cfg_get(self.rollout_cfg, "avoid_single_block_tail", False)):
            return self._sample_shape_from(shapes)

        min_blocks = int(_cfg_get(self.rollout_cfg, "min_blocks_per_rollout", 1))
        valid: List[IForwardRolloutShape] = []
        for shape in shapes:
            blocks = int(shape.blocks_per_rollout)
            if blocks > remaining:
                continue
            tail = int(remaining - blocks)
            if tail == 0 or tail >= int(min_blocks):
                valid.append(shape)
        if valid:
            return self._sample_shape_from(valid)
        sampled = self._sample_shape_from(shapes)
        sampled_blocks = int(sampled.blocks_per_rollout)
        if sampled_blocks < remaining:
            return dataclasses.replace(
                sampled,
                name=f"{sampled.name}_tailmerge_b{remaining}",
                blocks_per_rollout=int(remaining),
            )
        return sampled

    def _sample_shape_for_episode_length(self, episode_blocks: int) -> IForwardRolloutShape:
        shapes = self._active_shapes()
        valid = [shape for shape in shapes if int(shape.blocks_per_rollout) <= int(episode_blocks)]
        if not valid:
            return self._sample_shape_from(shapes)
        return self._sample_shape_from(valid)

    def _min_active_shape_blocks(self) -> int:
        shapes = self._active_shapes()
        if not shapes:
            raise ValueError("scheduler_iforward rollout.shapes is empty")
        return min(int(shape.blocks_per_rollout) for shape in shapes)

    def _rollouts_per_episode_budget(self) -> Optional[int]:
        raw = _cfg_get(self.episode_cfg, "rollouts_per_episode", None)
        return None if raw is None else int(raw)

    def _rollout_budget_reached(self, episode: Dict[str, Any]) -> bool:
        budget = self._rollouts_per_episode_budget()
        if budget is None:
            return False
        return int(episode.get("rollout_idx_in_episode", 0)) >= int(budget)

    def _random_start_uses_rollout_budget(self) -> bool:
        return self._block_selection_policy() == "random_start_contiguous" and self._rollouts_per_episode_budget() is not None

    def _remaining_blocks_in_episode(self, episode: Dict[str, Any]) -> int:
        frame_chain = [int(x) for x in list(episode.get("frame_chain", []) or [])]
        return max(0, int(len(frame_chain)) - int(episode.get("block_cursor", 0)))

    def _should_skip_episode_tail(self, episode: Dict[str, Any]) -> bool:
        if self._is_v3():
            return False
        if self._random_start_uses_rollout_budget():
            frame_chain = [int(x) for x in list(episode.get("frame_chain", []) or [])]
            return len(frame_chain) < int(self._min_active_shape_blocks())
        remaining = int(self._remaining_blocks_in_episode(episode))
        if remaining <= 0:
            return True
        if bool(_cfg_get(self.rollout_cfg, "allow_short_final_rollout", True)):
            return False
        min_rollout = int(_cfg_get(self.rollout_cfg, "min_blocks_per_rollout", 1))
        min_shape = int(self._min_active_shape_blocks())
        return remaining < max(int(min_rollout), int(min_shape))

    def _skip_current_episode_tail(self, episode: Dict[str, Any], *, reason: str) -> None:
        remaining = int(self._remaining_blocks_in_episode(episode))
        self._emit(
            {
                "type": "episode_tail_skipped",
                "scheduler_version": IFORWARD_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(episode["scene_id"]),
                "segment_id": int(episode["segment_id"]),
                "episode_id": int(episode["episode_id"]),
                "rollout_idx_in_episode": int(episode["rollout_idx_in_episode"]),
                "remaining_blocks": int(remaining),
                "reason": str(reason),
            }
        )
        self._emit(
            {
                "type": "episode_end",
                "scheduler_version": IFORWARD_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(episode["scene_id"]),
                "segment_id": int(episode["segment_id"]),
                "episode_id": int(episode["episode_id"]),
                "rollout_id_global": int(self._rollout_id_global),
                "tail_skipped": True,
                "remaining_blocks": int(remaining),
            }
        )
        self._current_episode = None

    def _finish_current_episode(self, episode: Dict[str, Any], *, reason: str) -> None:
        self._emit(
            {
                "type": "episode_end",
                "scheduler_version": IFORWARD_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(episode["scene_id"]),
                "segment_id": int(episode["segment_id"]),
                "episode_id": int(episode["episode_id"]),
                "rollout_id_global": int(self._rollout_id_global),
                "rollout_idx_in_episode": int(episode["rollout_idx_in_episode"]),
                "reason": str(reason),
            }
        )
        self._current_episode = None

    def _ensure_episode_with_rollout_available(self) -> Dict[str, Any]:
        attempts = 0
        while True:
            episode = self._ensure_episode()
            if self._rollout_budget_reached(episode):
                self._finish_current_episode(episode, reason="rollouts_per_episode_reached")
                attempts += 1
                if attempts > max(8, len(self._episode_plan) + 2):
                    raise ValueError(
                        "scheduler_iforward could not find an episode with remaining rollout budget. "
                        "Check episode.rollouts_per_episode and episode window settings."
                    )
                continue
            if not self._should_skip_episode_tail(episode):
                return episode
            self._skip_current_episode_tail(episode, reason="remaining_blocks_lt_required_rollout")
            attempts += 1
            if attempts > max(8, len(self._episode_plan) + 2):
                raise ValueError(
                    "scheduler_iforward could not find an episode with enough blocks for a rollout. "
                    "Check rollout.min_blocks_per_rollout, rollout.shapes, and episode window settings."
                )

    def _emit(self, event: Dict[str, Any]) -> None:
        self._pending_events.append(dict(event))

    def pop_events(self) -> List[Dict[str, Any]]:
        out = list(self._pending_events)
        self._pending_events.clear()
        return out

    def get_current_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    def _rebuild_epoch_plan(self) -> None:
        self.epoch_idx += 1
        if self.fixed_scene_id is not None:
            scene_ids = [int(self.fixed_scene_id)]
        else:
            scene_ids = [int(x) for x in self.dataset.list_training_scene_ids()]
        if str(_cfg_get(self.traversal_cfg, "scene_order", "shuffle_per_epoch")) == "shuffle_per_epoch":
            self.rng.shuffle(scene_ids)

        specs: List[Dict[str, Any]] = []
        blocks_per_episode = int(_cfg_get(self.episode_cfg, "blocks_per_episode", 8))
        episode_stride = int(_cfg_get(self.episode_cfg, "episode_stride", blocks_per_episode))
        allow_short = bool(_cfg_get(self.episode_cfg, "allow_short_last_episode", True))
        min_blocks = int(_cfg_get(self.episode_cfg, "min_blocks_per_episode", 2))
        for scene_id in scene_ids:
            if self.fixed_segment_id is not None:
                segment_ids = [int(self.fixed_segment_id)]
            else:
                segment_ids = [int(x) for x in self.dataset.list_segment_ids(int(scene_id))]
            if str(_cfg_get(self.traversal_cfg, "segment_order", "ascending")) == "shuffle_per_epoch":
                self.rng.shuffle(segment_ids)
            for segment_id in segment_ids:
                sidx = self.dataset.get_segment_index(int(scene_id), int(segment_id))
                keyframes = [int(x) for x in list(getattr(sidx, "keyframe_indices", []) or [])]
                for start in range(0, len(keyframes), int(episode_stride)):
                    window = keyframes[start : start + blocks_per_episode]
                    if len(window) == blocks_per_episode:
                        pass
                    elif not allow_short or len(window) < min_blocks:
                        continue
                    specs.append(
                        {
                            "scene_id": int(scene_id),
                            "segment_id": int(segment_id),
                            "episode_start_keyframe_pos": int(start),
                            "keyframe_window": [int(x) for x in window],
                        }
                    )
        if not specs:
            raise ValueError("scheduler_iforward found no valid episode windows")
        self._episode_plan = specs
        self._episode_plan_cursor = 0

    @staticmethod
    def _keyframe_train_frames(sidx: Any, keyframe_idx: int) -> List[int]:
        train_set = set(int(x) for x in getattr(sidx, "train_frame_set", set()))
        raw = list(dict(getattr(sidx, "keyframe_to_frames", {}) or {}).get(int(keyframe_idx), []) or [])
        frames = [int(x) for x in raw if int(x) in train_set]
        if frames:
            return sorted(set(frames))
        if int(keyframe_idx) in train_set:
            return [int(keyframe_idx)]
        return []

    def _sample_train_frame_for_keyframe(self, sidx: Any, keyframe_idx: int, rng: Optional[random.Random] = None) -> int:
        candidates = self._keyframe_train_frames(sidx, int(keyframe_idx))
        if not candidates:
            raise ValueError(
                "scheduler_iforward keyframe has no train frames: "
                f"scene={getattr(sidx, 'scene_id', '?')} segment={getattr(sidx, 'segment_id', '?')} "
                f"keyframe={int(keyframe_idx)}"
            )
        return int((rng or self.rng).choice(candidates))

    def _start_next_episode(self) -> None:
        if self._episode_plan_cursor >= len(self._episode_plan):
            self._rebuild_epoch_plan()
        spec = dict(self._episode_plan[int(self._episode_plan_cursor)])
        self._episode_plan_cursor += 1

        sidx = self.dataset.get_segment_index(int(spec["scene_id"]), int(spec["segment_id"]))
        frame_policy = self._block_source_frame_policy()
        frame_chain: List[int] = []
        for keyframe_idx in list(spec["keyframe_window"]):
            candidates = self._keyframe_train_frames(sidx, int(keyframe_idx))
            if not candidates:
                raise ValueError(
                    "scheduler_iforward keyframe has no train frames: "
                    f"scene={spec['scene_id']} segment={spec['segment_id']} keyframe={int(keyframe_idx)}"
                )
            if self._is_v3() or frame_policy == "random_within_keyframe_per_rollout":
                frame_chain.append(int(candidates[0]))
            else:
                frame_chain.append(int(self.rng.choice(candidates)))

        episode_id = int(self._episode_id_next)
        self._episode_id_next += 1
        episode_rng = random.Random(self.rng.randrange(0, 2**63))
        self._current_episode = {
            "scene_id": int(spec["scene_id"]),
            "segment_id": int(spec["segment_id"]),
            "episode_id": int(episode_id),
            "episode_start_keyframe_pos": int(spec["episode_start_keyframe_pos"]),
            "keyframe_window": [int(x) for x in spec["keyframe_window"]],
            "frame_chain": [int(x) for x in frame_chain],
            "num_cams": int(getattr(sidx, "num_cams", 1)),
            "block_cursor": 0,
            "rollout_idx_in_episode": 0,
            "used_rollout_starts": [],
            "window_counts": {},
            "visited_frames": [],
            "visited_frame_set": set(),
            "visited_refs": [],
            "visited_ref_set": set(),
            "block_visit_counts": {},
            "episode_visit_idx": 0,
            "optimizer_step_idx_in_episode": 0,
            "episode_rng_state": episode_rng.getstate(),
        }
        self._emit(
            {
                "type": "episode_begin",
                "scheduler_version": self._scheduler_version(),
                "global_step": int(self.global_step),
                "scene_id": int(spec["scene_id"]),
                "segment_id": int(spec["segment_id"]),
                "episode_id": int(episode_id),
                "episode_start_keyframe_pos": int(spec["episode_start_keyframe_pos"]),
                "episode_num_blocks": int(len(frame_chain)),
                "rollouts_per_episode": self._rollouts_per_episode_budget(),
            }
        )

    def _ensure_episode(self) -> Dict[str, Any]:
        if self._current_episode is None:
            self._start_next_episode()
        if self._current_episode is None:
            raise ValueError("TrainSchedulerIForward internal episode state is not initialized")
        return self._current_episode

    @staticmethod
    def _refs_for_frames(num_cams: int, frames: Sequence[int]) -> List[ImageRef]:
        return [(int(frame_idx), int(cam_idx)) for frame_idx in frames for cam_idx in range(int(num_cams))]

    def _episode_rng(self, episode: Dict[str, Any]) -> random.Random:
        rng = random.Random()
        state = episode.get("episode_rng_state", None)
        if state is None:
            return self.rng
        rng.setstate(state)
        return rng

    @staticmethod
    def _store_episode_rng(episode: Dict[str, Any], rng: random.Random) -> None:
        episode["episode_rng_state"] = rng.getstate()

    def _v3_history_refs(
        self,
        *,
        episode: Dict[str, Any],
        num_cams: int,
        current_refs: Sequence[ImageRef],
        rng: random.Random,
    ) -> Tuple[List[ImageRef], int, bool, str]:
        history_cfg = dict(_cfg_get(self.supervision_cfg, "history_replay", {}) or {})
        if not bool(_cfg_get(history_cfg, "enable", True)):
            return [], 0, True, "history_disabled"
        if str(_cfg_get(history_cfg, "camera_policy", "all_cams")) != "all_cams":
            raise ValueError("scheduler_iforward v3 requires history_replay.camera_policy=all_cams")
        current_frames = {int(ref[0]) for ref in current_refs}
        visited_frames = [int(x) for x in list(episode.get("visited_frames", []) or [])]
        if not visited_frames:
            seen_frames: set[int] = set()
            for ref in list(episode.get("visited_refs", []) or []):
                frame = int(ref[0])
                if frame in seen_frames:
                    continue
                seen_frames.add(frame)
                visited_frames.append(frame)
        candidates = [int(frame) for frame in visited_frames if int(frame) not in current_frames]
        before = int(len(candidates) * max(int(num_cams), 1))
        if not candidates:
            return [], before, True, "no_prior_visited_frames"
        raw_max_frames = _cfg_get(history_cfg, "max_frames_per_rollout", None)
        if raw_max_frames is None:
            max_refs = int(_cfg_get(history_cfg, "max_refs_per_rollout", int(num_cams) * 8))
            max_frames = int(max_refs) // max(int(num_cams), 1)
        else:
            max_frames = int(raw_max_frames)
        if max_frames <= 0:
            return [], before, True, "max_frames_per_rollout_zero"
        if len(candidates) > int(max_frames):
            keep = sorted(rng.sample(range(len(candidates)), int(max_frames)))
            candidates = [candidates[int(i)] for i in keep]
        return self._refs_for_frames(int(num_cams), candidates), before, False, ""

    def _v3_sample_nearby(
        self,
        *,
        sidx: Any,
        num_cams: int,
        keyframe_window: Sequence[int],
        window_block_ids: Sequence[int],
        supervised_refs: Sequence[ImageRef],
        rng: random.Random,
    ) -> Tuple[List[int], List[ImageRef], int, bool, str]:
        nearby_cfg = dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {})
        if not bool(_cfg_get(nearby_cfg, "enable", True)):
            return [], [], -1, True, "nearby_disabled"
        frames_per_rollout = int(_cfg_get(nearby_cfg, "frames_per_rollout", 1))
        if frames_per_rollout <= 0:
            return [], [], -1, True, "frames_per_rollout_zero"
        supervised_frames = {int(ref[0]) for ref in supervised_refs}
        blocks = [int(x) for x in window_block_ids]
        rng.shuffle(blocks)
        train_set = set(int(x) for x in getattr(sidx, "train_frame_set", set()))
        for block_id in blocks:
            keyframe_idx = int(keyframe_window[int(block_id)])
            candidates = [
                int(frame)
                for frame in self._keyframe_train_frames(sidx, keyframe_idx)
                if int(frame) in train_set and int(frame) not in supervised_frames
            ]
            if not candidates:
                continue
            count = min(int(frames_per_rollout), len(candidates))
            frames = sorted(rng.sample(candidates, count))
            refs = self._refs_for_frames(num_cams, frames)
            max_refs = int(_cfg_get(nearby_cfg, "max_refs_per_rollout", len(refs)))
            if max_refs > 0 and len(refs) > max_refs:
                refs = refs[:max_refs]
                keep_frames = sorted({int(ref[0]) for ref in refs})
                frames = [int(frame) for frame in frames if int(frame) in set(keep_frames)]
            if not refs:
                return [], [], int(block_id), True, "max_refs_per_rollout_lt_num_cams"
            return frames, refs, int(block_id), False, ""
        return [], [], -1, True, "no_unsupervised_frame_in_current_rollout_block"

    def _sample_nearby_frames(
        self,
        *,
        sidx: Any,
        input_keyframes: Sequence[int],
        input_frames: Sequence[int],
        num_cams: int,
    ) -> Tuple[List[int], bool, str]:
        nearby_cfg = dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {})
        if not bool(_cfg_get(nearby_cfg, "enable", True)):
            return [], True, "nearby_disabled"
        frames_per_rollout = int(_cfg_get(nearby_cfg, "frames_per_rollout", 1))
        if frames_per_rollout <= 0:
            return [], True, "frames_per_rollout_zero"
        train_set = set(int(x) for x in getattr(sidx, "train_frame_set", set()))
        input_set = set(int(x) for x in input_frames)
        candidates: set[int] = set()
        kf_to_frames = dict(getattr(sidx, "keyframe_to_frames", {}) or {})
        for keyframe_idx in input_keyframes:
            for frame_idx in list(kf_to_frames.get(int(keyframe_idx), []) or []):
                frame_i = int(frame_idx)
                if frame_i in train_set and frame_i not in input_set:
                    candidates.add(frame_i)
        sorted_candidates = sorted(candidates)
        if not sorted_candidates:
            return [], True, "no_non_input_frame_in_rollout"
        max_refs = int(_cfg_get(nearby_cfg, "max_refs_per_rollout", 24))
        if max_refs > 0:
            max_frames_by_refs = int(max_refs) // max(int(num_cams), 1)
            if max_frames_by_refs <= 0:
                return [], True, "max_refs_per_rollout_lt_num_cams"
            frames_per_rollout = min(frames_per_rollout, max_frames_by_refs)
        count = min(int(frames_per_rollout), len(sorted_candidates))
        return sorted(self.rng.sample(sorted_candidates, count)), False, ""

    def _select_episode_blocks(
        self,
        *,
        episode: Dict[str, Any],
        shape: IForwardRolloutShape,
    ) -> Tuple[List[int], int, int]:
        keyframe_window = [int(x) for x in list(episode["keyframe_window"])]
        requested_blocks = int(shape.blocks_per_rollout)
        if requested_blocks < 1:
            raise ValueError("IForward requested_blocks_per_rollout must be >= 1")
        if len(keyframe_window) < requested_blocks:
            raise ValueError(
                "scheduler_iforward episode is shorter than requested rollout: "
                f"episode_blocks={len(keyframe_window)} requested={requested_blocks}"
            )

        policy = self._block_selection_policy()
        if policy == "next_contiguous":
            block_cursor = int(episode["block_cursor"])
            end = min(block_cursor + requested_blocks, len(keyframe_window))
            return list(range(block_cursor, end)), int(block_cursor), int(end)

        if policy != "random_start_contiguous":
            raise ValueError(f"Unsupported IForward block_selection_policy={policy!r}")

        max_start = int(len(keyframe_window) - requested_blocks)
        valid_starts = list(range(max_start + 1))
        used_starts = set(int(x) for x in list(episode.get("used_rollout_starts", []) or []))
        available = [int(x) for x in valid_starts if int(x) not in used_starts]
        if not available:
            available = valid_starts
        last_start_raw = episode.get("last_rollout_start_block_idx", None)
        if last_start_raw is not None and len(available) > 1:
            sequential_next = int(last_start_raw) + int(requested_blocks)
            non_sequential = [int(x) for x in available if int(x) != int(sequential_next)]
            if non_sequential:
                available = non_sequential
        start = int(self.rng.choice(available))
        end = int(start + requested_blocks)
        return list(range(start, end)), int(start), int(end)

    def _build_v3_rollout_plan(self, episode: Dict[str, Any]) -> IForwardRolloutPlan:
        rng = self._episode_rng(episode)
        sidx = self.dataset.get_segment_index(int(episode["scene_id"]), int(episode["segment_id"]))
        keyframe_window = [int(x) for x in list(episode["keyframe_window"])]
        frame_chain = [int(x) for x in list(episode["frame_chain"])]
        shape = self._sample_shape_for_episode(episode, rng=rng)
        requested_blocks = int(shape.blocks_per_rollout)
        repeats = int(shape.repeats_per_block)
        if requested_blocks < 1 or repeats < 1:
            raise ValueError("IForward v3 requested blocks/repeats must be >= 1")
        if len(keyframe_window) < requested_blocks:
            raise ValueError(
                "scheduler_iforward v3 episode is shorter than requested rollout: "
                f"episode_blocks={len(keyframe_window)} requested={requested_blocks}"
            )
        fixed_starts = [int(x) for x in list(_cfg_get(self.rollout_cfg, "fixed_window_starts", []) or [])]
        rollout_idx = int(episode.get("rollout_idx_in_episode", 0))
        max_start = int(len(keyframe_window) - requested_blocks)
        if rollout_idx < len(fixed_starts):
            start = int(fixed_starts[rollout_idx])
            if start < 0 or start > max_start:
                raise ValueError(f"scheduler_iforward v3 fixed_window_start={start} out of range [0,{max_start}]")
        else:
            start = int(rng.randint(0, max_start))
        end = int(start + requested_blocks)
        block_ids = list(range(start, end))
        window_keyframes = [int(keyframe_window[int(idx)]) for idx in block_ids]
        input_frames = [self._sample_train_frame_for_keyframe(sidx, int(kf), rng) for kf in window_keyframes]
        self._store_episode_rng(episode, rng)

        num_cams = int(episode["num_cams"])
        current_role = str(_cfg_get(dict(_cfg_get(self.supervision_cfg, "current", {}) or {}), "role_name", "final_current_recon"))
        history_cfg = dict(_cfg_get(self.supervision_cfg, "history_replay", {}) or {})
        history_role = str(_cfg_get(history_cfg, "role_name", "final_history_replay"))
        nearby_cfg = dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {})
        nearby_role = str(_cfg_get(nearby_cfg, "role_name", "final_nearby_rollout"))
        current_refs = self._refs_for_frames(num_cams, input_frames)
        history_refs, history_before, history_skipped, history_reason = self._v3_history_refs(
            episode=episode,
            num_cams=int(num_cams),
            current_refs=current_refs,
            rng=rng,
        )
        nearby_frames, nearby_refs, nearby_block_id, nearby_skipped, nearby_reason = self._v3_sample_nearby(
            sidx=sidx,
            num_cams=num_cams,
            keyframe_window=keyframe_window,
            window_block_ids=block_ids,
            supervised_refs=list(current_refs) + list(history_refs),
            rng=rng,
        )
        self._store_episode_rng(episode, rng)

        raw_target_refs = [tuple(x) for x in list(current_refs) + list(history_refs) + list(nearby_refs)]
        raw_target_roles = (
            [current_role for _ in current_refs]
            + [history_role for _ in history_refs]
            + [nearby_role for _ in nearby_refs]
        )
        target_refs, target_roles = _dedupe_refs_roles_keep_order(raw_target_refs, raw_target_roles)
        evidence_refs_flat = _dedupe_refs_keep_order(current_refs)
        window_hash = _stable_window_hash(int(episode["scene_id"]), int(episode["segment_id"]), block_ids)
        window_counts = dict(episode.get("window_counts", {}) or {})
        revisit_count = int(window_counts.get(int(window_hash), 0))
        block_counts = dict(episode.get("block_visit_counts", {}) or {})
        inner_k = int(requested_blocks * repeats)
        steps: List[IForwardStepPlan] = []
        for rollout_rank, block_idx in enumerate(block_ids):
            frame_idx = int(input_frames[int(rollout_rank)])
            keyframe_idx = int(window_keyframes[int(rollout_rank)])
            evidence_refs = self._refs_for_frames(num_cams, [frame_idx])
            block_visit_before = int(block_counts.get(int(block_idx), 0))
            episode_visit_idx = int(episode.get("episode_visit_idx", 0)) + int(rollout_rank)
            for repeat_idx in range(repeats):
                step_idx = len(steps)
                is_exit = bool(int(repeat_idx) == int(repeats) - 1)
                steps.append(
                    IForwardStepPlan(
                        step_idx=int(step_idx),
                        block_id=int(block_idx),
                        episode_block_idx=int(block_idx),
                        rollout_block_rank=int(rollout_rank),
                        repeat_idx=int(repeat_idx),
                        repeats_per_block=int(repeats),
                        is_block_enter=bool(int(repeat_idx) == 0),
                        is_block_exit=bool(is_exit),
                        source_keyframe_idx=int(keyframe_idx),
                        source_frame_idx=int(frame_idx),
                        evidence_refs=[tuple(x) for x in evidence_refs],
                        evidence_frame_indices=[int(frame_idx) for _ in evidence_refs],
                        evidence_cam_indices=[int(cam_idx) for _, cam_idx in evidence_refs],
                        commit_observation_memory=bool(int(repeat_idx) == 0),
                        update_optimizer_memory=True,
                        detach_before_step=False,
                        detach_after_step=False,
                        allow_step_render_loss=False,
                        step_loss_refs=[],
                        rollout_pos_code=float(step_idx) / float(max(inner_k - 1, 1)),
                        frame_pos_code=float(rollout_rank) / float(max(requested_blocks - 1, 1)),
                        repeat_pos_code=float(repeat_idx) / float(max(repeats - 1, 1)),
                        is_frame_exit=bool(is_exit),
                        episode_visit_idx=int(episode_visit_idx),
                        rollout_visit_idx=int(rollout_rank),
                        optimizer_step_idx_in_episode=int(episode.get("optimizer_step_idx_in_episode", 0)) + int(step_idx),
                        record_update_norm=True,
                        commit_support_on_exit=bool(is_exit),
                        commit_residual_on_exit=bool(is_exit),
                        window_start=int(start),
                        window_end=int(end),
                        window_hash=int(window_hash),
                        window_revisit_count=int(revisit_count),
                        is_repeated_window=bool(revisit_count > 0),
                        block_visit_count_before=int(block_visit_before),
                        block_visit_count_after=int(block_visit_before + 1),
                    )
                )

        rollout_budget = int(_cfg_get(self.episode_cfg, "rollouts_per_episode", 8))
        episode_end = bool(int(rollout_idx) + 1 >= int(rollout_budget))
        final_supervision = IForwardFinalSupervisionPlan(
            refs=[tuple(x) for x in target_refs],
            roles=[str(x) for x in target_roles],
            current_input_frames=[int(x) for x in input_frames],
            nearby_frames=[int(x) for x in nearby_frames],
            skipped_nearby=bool(nearby_skipped),
            nearby_skip_reason=str(nearby_reason),
            current_ref_count=int(len(current_refs)),
            nearby_ref_count=int(len(nearby_refs)),
            current_frames=[int(x) for x in input_frames],
            current_refs=[tuple(x) for x in current_refs],
            history_frames=list(dict.fromkeys(int(ref[0]) for ref in history_refs)),
            history_refs=[tuple(x) for x in history_refs],
            history_ref_count_before_dedupe=int(history_before),
            history_skipped=bool(history_skipped),
            history_skip_reason=str(history_reason),
            nearby_refs=[tuple(x) for x in nearby_refs],
            nearby_block_id=int(nearby_block_id),
            history_ref_count=int(len(history_refs)),
        )
        leakage_check = {
            "same_scene_segment_required": bool(_cfg_get(self.leakage_check_cfg, "same_scene_segment_required", True)),
            "forbid_test_refs_in_train": bool(_cfg_get(self.leakage_check_cfg, "forbid_test_refs_in_train", True)),
            "target_role_count_match": bool(len(target_refs) == len(target_roles)),
            "current_covers_all_input_frames": bool(
                {tuple(x) for x in current_refs} == {tuple(x) for x in self._refs_for_frames(num_cams, input_frames)}
            ),
            "history_prior_visited_only": True,
            "history_current_overlap": int(len(set(history_refs) & set(current_refs))),
            "nearby_evidence_overlap": int(len(set(nearby_refs) & set(evidence_refs_flat))),
            "nearby_current_overlap": int(len(set(nearby_refs) & set(current_refs))),
            "nearby_history_overlap": int(len(set(nearby_refs) & set(history_refs))),
            "nearby_block_in_window": bool(int(nearby_block_id) in set(block_ids)) if nearby_refs else True,
        }
        request_meta = {
            "scheduler_version": IFORWARD_V3_SCHEDULER_VERSION,
            "model_family": IFORWARD_MODEL_FAMILY,
            "loss_timing_policy": "rollout_final_only",
            "scene_id": int(episode["scene_id"]),
            "segment_id": int(episode["segment_id"]),
            "episode_id": int(episode["episode_id"]),
            "episode_idx_global": int(episode["episode_id"]),
            "rollout_id_global": int(self._rollout_id_global),
            "rollout_idx_in_episode": int(rollout_idx),
            "rollouts_per_episode": int(rollout_budget),
            "inner_K": int(inner_k),
            "shape_name": str(shape.name),
            "requested_shape_name": str(shape.name),
            "window_policy": str(_cfg_get(self.rollout_cfg, "window_policy", "random_with_replacement")),
            "delivery_order_policy": "chronological_inside_window",
            "window_start": int(start),
            "window_end": int(end),
            "window_block_ids": [int(x) for x in block_ids],
            "window_hash": int(window_hash),
            "window_revisit_count": int(revisit_count),
            "unique_windows_seen": int(len(set(list(window_counts.keys()) + [int(window_hash)]))),
            "is_repeated_window": bool(revisit_count > 0),
            "source_frame_sampling_policy": "random_within_keyframe_per_visit",
            "blocks_per_rollout": int(requested_blocks),
            "actual_blocks_per_rollout": int(requested_blocks),
            "repeats_per_block": int(repeats),
            "requested_inner_K": int(inner_k),
            "actual_inner_K": int(inner_k),
            "carry_scene_state_after_rollout": bool(not episode_end),
            "discard_scene_state_after_rollout": bool(episode_end),
            "source_image_refs": [tuple(x) for x in evidence_refs_flat],
            "source_image_ref": tuple(evidence_refs_flat[0]) if evidence_refs_flat else None,
            "target_image_refs": [tuple(x) for x in target_refs],
            "target_image_roles": [str(x) for x in target_roles],
            "current_refs": [tuple(x) for x in current_refs],
            "history_refs": [tuple(x) for x in history_refs],
            "nearby_refs": [tuple(x) for x in nearby_refs],
            "evidence_refs_by_step": [[tuple(x) for x in step.evidence_refs] for step in steps],
            "step_source_frame_indices": [int(step.source_frame_idx) for step in steps],
            "step_repeat_indices": [int(step.repeat_idx) for step in steps],
            "step_block_indices": [int(step.episode_block_idx) for step in steps],
            "step_episode_visit_indices": [int(step.episode_visit_idx) for step in steps],
            "step_optimizer_indices": [int(step.optimizer_step_idx_in_episode) for step in steps],
            "step_block_enter_flags": [bool(step.is_block_enter) for step in steps],
            "step_block_exit_flags": [bool(step.is_block_exit) for step in steps],
            "final_supervision": dataclasses.asdict(final_supervision),
            "leakage_check": dict(leakage_check),
            "assembly_mode": "image_ref_iforward_v1",
        }
        plan = IForwardRolloutPlan(
            scheduler_version=IFORWARD_V3_SCHEDULER_VERSION,
            scene_id=int(episode["scene_id"]),
            segment_id=int(episode["segment_id"]),
            episode_id=int(episode["episode_id"]),
            rollout_id_global=int(self._rollout_id_global),
            rollout_idx_in_episode=int(rollout_idx),
            episode_start_keyframe_pos=int(episode["episode_start_keyframe_pos"]),
            keyframe_window=[int(x) for x in keyframe_window],
            frame_chain=[int(x) for x in frame_chain],
            num_cams=int(num_cams),
            shape_name=str(shape.name),
            blocks_per_rollout=int(requested_blocks),
            repeats_per_block=int(repeats),
            requested_blocks_per_rollout=int(requested_blocks),
            actual_blocks_per_rollout=int(requested_blocks),
            requested_inner_K=int(inner_k),
            actual_inner_K=int(inner_k),
            short_rollout=False,
            short_rollout_reason="",
            episode_block_indices=[int(x) for x in block_ids],
            input_keyframe_indices=[int(x) for x in window_keyframes],
            input_frame_indices=[int(x) for x in input_frames],
            delivery_frame_indices=[int(x) for x in input_frames],
            delivery_order_policy="chronological_inside_window",
            inner_K=int(inner_k),
            steps=steps,
            final_supervision=final_supervision,
            reset_scene_state_before_rollout=bool(int(rollout_idx) == 0),
            carry_scene_state_after_rollout=bool(not episode_end),
            episode_end_after_rollout=bool(episode_end),
            detach_graph_after_rollout=True,
            evidence_refs_flat=[tuple(x) for x in evidence_refs_flat],
            target_refs_flat=[tuple(x) for x in target_refs],
            target_roles_flat=[str(x) for x in target_roles],
            request_meta=request_meta,
            leakage_check=leakage_check,
            model_family=IFORWARD_MODEL_FAMILY,
            rollouts_per_episode=int(rollout_budget),
            episode_num_blocks=int(len(keyframe_window)),
            window_policy=str(_cfg_get(self.rollout_cfg, "window_policy", "random_with_replacement")),
            window_start=int(start),
            window_end=int(end),
            window_block_ids=[int(x) for x in block_ids],
            window_keyframe_indices=[int(x) for x in window_keyframes],
            window_frame_indices=[int(x) for x in input_frames],
            window_hash=int(window_hash),
            window_revisit_count=int(revisit_count),
            unique_windows_seen=int(len(set(list(window_counts.keys()) + [int(window_hash)]))),
            is_repeated_window=bool(revisit_count > 0),
        )
        self._validate_plan(plan, sidx=sidx)
        return plan

    def _build_rollout_plan(self, episode: Dict[str, Any]) -> IForwardRolloutPlan:
        if self._is_v3():
            return self._build_v3_rollout_plan(episode)
        sidx = self.dataset.get_segment_index(int(episode["scene_id"]), int(episode["segment_id"]))
        block_cursor = int(episode["block_cursor"])
        frame_chain = [int(x) for x in list(episode["frame_chain"])]
        keyframe_window = [int(x) for x in list(episode["keyframe_window"])]
        random_start_rollout_budget = bool(self._random_start_uses_rollout_budget())
        if block_cursor >= len(frame_chain) and not random_start_rollout_budget:
            raise ValueError("IForward episode block cursor is already at end")

        remaining_blocks = int(len(frame_chain) - block_cursor)
        if self._block_selection_policy() == "random_start_contiguous":
            shape = self._sample_shape_for_episode_length(len(frame_chain))
        else:
            shape = self._sample_shape_for_remaining(remaining_blocks)
        requested_blocks = int(shape.blocks_per_rollout)
        repeats = int(shape.repeats_per_block)
        episode_blocks, rollout_start_block_idx, selected_end = self._select_episode_blocks(episode=episode, shape=shape)
        if len(episode_blocks) < requested_blocks:
            allow_short = bool(_cfg_get(self.rollout_cfg, "allow_short_final_rollout", True))
            min_blocks = int(_cfg_get(self.rollout_cfg, "min_blocks_per_rollout", 1))
            if not allow_short or len(episode_blocks) < min_blocks:
                raise ValueError(
                    "scheduler_iforward final rollout is shorter than requested and short rollout is disabled"
                )

        input_keyframes = [int(keyframe_window[int(idx)]) for idx in episode_blocks]
        if self._block_source_frame_policy() == "random_within_keyframe_per_rollout":
            input_frames = [self._sample_train_frame_for_keyframe(sidx, int(kf)) for kf in input_keyframes]
            rollout_frame_by_block = {int(block_idx): int(frame_idx) for block_idx, frame_idx in zip(episode_blocks, input_frames)}
            plan_frame_chain = [
                int(rollout_frame_by_block.get(int(idx), int(frame_chain[int(idx)])))
                for idx in range(len(frame_chain))
            ]
        else:
            input_frames = [int(frame_chain[int(idx)]) for idx in episode_blocks]
            rollout_frame_by_block = {int(block_idx): int(frame_chain[int(block_idx)]) for block_idx in episode_blocks}
            plan_frame_chain = [int(x) for x in frame_chain]
        delivery_blocks = list(episode_blocks)
        delivery_frames = [int(rollout_frame_by_block[int(idx)]) for idx in delivery_blocks]
        actual_blocks = int(len(delivery_blocks))
        requested_inner_k = int(requested_blocks * repeats)
        actual_inner_k = int(actual_blocks * repeats)
        short_rollout = bool(actual_blocks < requested_blocks)
        short_reason = ""
        if short_rollout:
            short_reason = "episode_tail_single_block" if actual_blocks == 1 else "episode_tail"
        shape_name = str(shape.name) if not short_rollout else f"{shape.name}_short_b{actual_blocks}_r{repeats}"

        num_cams = int(episode["num_cams"])
        steps: List[IForwardStepPlan] = []
        for rollout_rank, block_idx in enumerate(delivery_blocks):
            frame_idx = int(rollout_frame_by_block[int(block_idx)])
            keyframe_idx = int(keyframe_window[int(block_idx)])
            evidence_refs = self._refs_for_frames(num_cams, [frame_idx])
            for repeat_idx in range(repeats):
                step_idx = len(steps)
                steps.append(
                    IForwardStepPlan(
                        step_idx=int(step_idx),
                        block_id=int(block_idx),
                        episode_block_idx=int(block_idx),
                        rollout_block_rank=int(rollout_rank),
                        repeat_idx=int(repeat_idx),
                        repeats_per_block=int(repeats),
                        is_block_enter=bool(int(repeat_idx) == 0),
                        is_block_exit=bool(int(repeat_idx) == int(repeats) - 1),
                        source_keyframe_idx=int(keyframe_idx),
                        source_frame_idx=int(frame_idx),
                        evidence_refs=[tuple(x) for x in evidence_refs],
                        evidence_frame_indices=[int(frame_idx) for _ in evidence_refs],
                        evidence_cam_indices=[int(cam_idx) for _, cam_idx in evidence_refs],
                        commit_observation_memory=bool(int(repeat_idx) == 0),
                        update_optimizer_memory=True,
                        detach_before_step=False,
                        detach_after_step=False,
                        allow_step_render_loss=False,
                        step_loss_refs=[],
                        rollout_pos_code=float(step_idx) / float(max(actual_inner_k - 1, 1)),
                        frame_pos_code=float(rollout_rank) / float(max(actual_blocks - 1, 1)),
                        repeat_pos_code=float(repeat_idx) / float(max(repeats - 1, 1)),
                    )
                )

        current_role = str(_cfg_get(dict(_cfg_get(self.supervision_cfg, "current", {}) or {}), "role_name", "final_current_recon"))
        nearby_cfg = dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {})
        nearby_role = str(_cfg_get(nearby_cfg, "role_name", "final_nearby_rollout"))
        current_refs = self._refs_for_frames(num_cams, input_frames)
        nearby_frames, skipped_nearby, skip_reason = self._sample_nearby_frames(
            sidx=sidx,
            input_keyframes=input_keyframes,
            input_frames=input_frames,
            num_cams=num_cams,
        )
        nearby_refs = self._refs_for_frames(num_cams, nearby_frames)
        raw_target_refs = [tuple(x) for x in current_refs + nearby_refs]
        raw_target_roles = [current_role for _ in current_refs] + [nearby_role for _ in nearby_refs]
        target_refs, target_roles = _dedupe_refs_roles_keep_order(raw_target_refs, raw_target_roles)
        evidence_refs_flat = _dedupe_refs_keep_order([ref for step in steps for ref in step.evidence_refs])

        final_supervision = IForwardFinalSupervisionPlan(
            refs=[tuple(x) for x in target_refs],
            roles=[str(x) for x in target_roles],
            current_input_frames=[int(x) for x in input_frames],
            nearby_frames=[int(x) for x in nearby_frames],
            skipped_nearby=bool(skipped_nearby),
            nearby_skip_reason=str(skip_reason),
            current_ref_count=int(len(current_refs)),
            nearby_ref_count=int(len(nearby_refs)),
        )
        logical_end = int(block_cursor + actual_blocks)
        remaining_after_rollout = max(0, int(len(frame_chain) - logical_end))
        tail_skipped_after_rollout = False
        rollout_budget = self._rollouts_per_episode_budget()
        if random_start_rollout_budget:
            next_rollout_idx = int(episode["rollout_idx_in_episode"]) + 1
            episode_end = bool(rollout_budget is not None and int(next_rollout_idx) >= int(rollout_budget))
        else:
            episode_end = bool(logical_end >= len(frame_chain))
        if not random_start_rollout_budget and not episode_end and remaining_after_rollout > 0:
            allow_short_final = bool(_cfg_get(self.rollout_cfg, "allow_short_final_rollout", True))
            if not allow_short_final:
                min_rollout = int(_cfg_get(self.rollout_cfg, "min_blocks_per_rollout", 1))
                min_shape = int(self._min_active_shape_blocks())
                min_required = max(int(min_rollout), int(min_shape))
                if int(remaining_after_rollout) < int(min_required):
                    tail_skipped_after_rollout = True
                    episode_end = True
        reset_before = bool(int(episode["rollout_idx_in_episode"]) == 0)

        leakage_check = {
            "same_scene_segment_required": bool(_cfg_get(self.leakage_check_cfg, "same_scene_segment_required", True)),
            "forbid_test_refs_in_train": bool(_cfg_get(self.leakage_check_cfg, "forbid_test_refs_in_train", True)),
            "target_role_count_match": bool(len(target_refs) == len(target_roles)),
            "nearby_evidence_overlap": int(len(set(nearby_refs) & set(evidence_refs_flat))),
            "nearby_input_frame_overlap": int(len(set(nearby_frames) & set(input_frames))),
            "current_supervision_must_cover_all_inputs": bool(
                _cfg_get(self.leakage_check_cfg, "current_supervision_must_cover_all_inputs", True)
            ),
        }
        request_meta = {
            "scheduler_version": IFORWARD_SCHEDULER_VERSION,
            "model_family": IFORWARD_MODEL_FAMILY,
            "loss_timing_policy": "rollout_final_only",
            "scene_id": int(episode["scene_id"]),
            "segment_id": int(episode["segment_id"]),
            "episode_id": int(episode["episode_id"]),
            "episode_idx_global": int(episode["episode_id"]),
            "rollout_id_global": int(self._rollout_id_global),
            "rollout_idx_in_episode": int(episode["rollout_idx_in_episode"]),
            "rollouts_per_episode": None if rollout_budget is None else int(rollout_budget),
            "inner_K": int(actual_inner_k),
            "shape_name": str(shape_name),
            "requested_shape_name": str(shape.name),
            "block_selection_policy": str(self._block_selection_policy()),
            "source_frame_sampling_policy": str(self._block_source_frame_policy()),
            "rollout_start_block_idx": int(rollout_start_block_idx),
            "selected_block_end_exclusive": int(selected_end),
            "blocks_per_rollout": int(requested_blocks),
            "requested_blocks_per_rollout": int(requested_blocks),
            "actual_blocks_per_rollout": int(actual_blocks),
            "repeats_per_block": int(repeats),
            "requested_inner_K": int(requested_inner_k),
            "actual_inner_K": int(actual_inner_k),
            "short_rollout": bool(short_rollout),
            "short_rollout_reason": str(short_reason),
            "tail_skipped_after_rollout": bool(tail_skipped_after_rollout),
            "tail_skipped_remaining_blocks": int(remaining_after_rollout) if bool(tail_skipped_after_rollout) else 0,
            "carry_scene_state_after_rollout": bool(not episode_end),
            "discard_scene_state_after_rollout": bool(episode_end),
            "source_image_refs": [tuple(x) for x in evidence_refs_flat],
            "source_image_ref": tuple(evidence_refs_flat[0]) if evidence_refs_flat else None,
            "target_image_refs": [tuple(x) for x in target_refs],
            "target_image_roles": [str(x) for x in target_roles],
            "evidence_refs_by_step": [[tuple(x) for x in step.evidence_refs] for step in steps],
            "step_source_frame_indices": [int(step.source_frame_idx) for step in steps],
            "step_repeat_indices": [int(step.repeat_idx) for step in steps],
            "step_block_indices": [int(step.episode_block_idx) for step in steps],
            "step_block_enter_flags": [bool(step.is_block_enter) for step in steps],
            "step_block_exit_flags": [bool(step.is_block_exit) for step in steps],
            "final_supervision": dataclasses.asdict(final_supervision),
            "leakage_check": dict(leakage_check),
            "assembly_mode": "image_ref_iforward_v1",
        }

        plan = IForwardRolloutPlan(
            scheduler_version=IFORWARD_SCHEDULER_VERSION,
            scene_id=int(episode["scene_id"]),
            segment_id=int(episode["segment_id"]),
            episode_id=int(episode["episode_id"]),
            rollout_id_global=int(self._rollout_id_global),
            rollout_idx_in_episode=int(episode["rollout_idx_in_episode"]),
            episode_start_keyframe_pos=int(episode["episode_start_keyframe_pos"]),
            keyframe_window=[int(x) for x in keyframe_window],
            frame_chain=[int(x) for x in plan_frame_chain],
            num_cams=int(num_cams),
            shape_name=str(shape_name),
            blocks_per_rollout=int(requested_blocks),
            repeats_per_block=int(repeats),
            requested_blocks_per_rollout=int(requested_blocks),
            actual_blocks_per_rollout=int(actual_blocks),
            requested_inner_K=int(requested_inner_k),
            actual_inner_K=int(actual_inner_k),
            short_rollout=bool(short_rollout),
            short_rollout_reason=str(short_reason),
            episode_block_indices=[int(x) for x in episode_blocks],
            input_keyframe_indices=[int(x) for x in input_keyframes],
            input_frame_indices=[int(x) for x in input_frames],
            delivery_frame_indices=[int(x) for x in delivery_frames],
            delivery_order_policy="chronological",
            inner_K=int(actual_inner_k),
            steps=steps,
            final_supervision=final_supervision,
            reset_scene_state_before_rollout=bool(reset_before),
            carry_scene_state_after_rollout=bool(not episode_end),
            episode_end_after_rollout=bool(episode_end),
            tail_skipped_after_rollout=bool(tail_skipped_after_rollout),
            tail_skipped_remaining_blocks=int(remaining_after_rollout) if bool(tail_skipped_after_rollout) else 0,
            detach_graph_after_rollout=True,
            evidence_refs_flat=[tuple(x) for x in evidence_refs_flat],
            target_refs_flat=[tuple(x) for x in target_refs],
            target_roles_flat=[str(x) for x in target_roles],
            request_meta=request_meta,
            leakage_check=leakage_check,
        )
        self._validate_plan(plan, sidx=sidx)
        return plan

    def _validate_plan(self, plan: IForwardRolloutPlan, *, sidx: Any) -> None:
        if str(plan.scheduler_version) == IFORWARD_V3_SCHEDULER_VERSION:
            self._validate_v3_plan(plan, sidx=sidx)
            return
        if str(plan.scheduler_version) != IFORWARD_SCHEDULER_VERSION:
            raise ValueError(f"expected scheduler_version={IFORWARD_SCHEDULER_VERSION}")
        if int(plan.inner_K) < 1:
            raise ValueError("IForward inner_K must be >= 1")
        if len(plan.steps) != int(plan.inner_K):
            raise ValueError("IForward len(steps) must equal inner_K")
        if int(plan.inner_K) != int(plan.actual_blocks_per_rollout) * int(plan.repeats_per_block):
            raise ValueError("IForward inner_K must equal actual_blocks_per_rollout * repeats_per_block")
        if not plan.evidence_refs_flat:
            raise ValueError("IForward rollout requires non-empty evidence_refs_flat")
        if not plan.target_refs_flat:
            raise ValueError("IForward rollout requires non-empty target_refs_flat")
        if len(plan.target_refs_flat) != len(plan.target_roles_flat):
            raise ValueError("IForward target refs/roles length mismatch")

        expected_cams = set(range(int(plan.num_cams)))
        frame_repeat_commit_counts: Dict[int, int] = {}
        block_enter_counts: Dict[int, int] = {}
        block_exit_counts: Dict[int, int] = {}
        for step in plan.steps:
            if not step.evidence_refs:
                raise ValueError(f"IForward step {step.step_idx} has empty evidence_refs")
            frames = {int(ref[0]) for ref in step.evidence_refs}
            cams = {int(ref[1]) for ref in step.evidence_refs}
            if frames != {int(step.source_frame_idx)}:
                raise ValueError("IForward step evidence refs must match step.source_frame_idx")
            if cams != expected_cams:
                raise ValueError("IForward v1 step evidence must cover all cams")
            if bool(step.detach_before_step) or bool(step.detach_after_step):
                raise ValueError("IForward rollout steps must not detach inside rollout")
            if bool(step.allow_step_render_loss) or step.step_loss_refs:
                raise ValueError("IForward v1 forbids intermediate step render loss")
            if bool(step.commit_observation_memory) != (int(step.repeat_idx) == 0):
                raise ValueError("IForward commit_observation_memory must be true only on repeat_idx=0")
            if not bool(step.update_optimizer_memory):
                raise ValueError("IForward update_optimizer_memory must be true for every repeat")
            if int(step.block_id) != int(step.episode_block_idx):
                raise ValueError("IForward scheduler step.block_id must match episode_block_idx")
            if int(step.repeats_per_block) != int(plan.repeats_per_block):
                raise ValueError("IForward scheduler step.repeats_per_block must match plan.repeats_per_block")
            if bool(step.is_block_enter) != (int(step.repeat_idx) == 0):
                raise ValueError("IForward scheduler is_block_enter must be true only on repeat_idx=0")
            if bool(step.is_block_exit) != (int(step.repeat_idx) == int(plan.repeats_per_block) - 1):
                raise ValueError("IForward scheduler is_block_exit must be true only on the last repeat")
            if bool(step.is_block_enter):
                block_enter_counts[int(step.episode_block_idx)] = block_enter_counts.get(int(step.episode_block_idx), 0) + 1
            if bool(step.is_block_exit):
                block_exit_counts[int(step.episode_block_idx)] = block_exit_counts.get(int(step.episode_block_idx), 0) + 1
            if bool(step.commit_observation_memory):
                frame_repeat_commit_counts[int(step.source_frame_idx)] = frame_repeat_commit_counts.get(int(step.source_frame_idx), 0) + 1

        for block_idx in plan.episode_block_indices:
            if block_enter_counts.get(int(block_idx), 0) != 1:
                raise ValueError("Each IForward block must have exactly one is_block_enter step")
            if block_exit_counts.get(int(block_idx), 0) != 1:
                raise ValueError("Each IForward block must have exactly one is_block_exit step")

        for frame_idx in plan.input_frame_indices:
            if frame_repeat_commit_counts.get(int(frame_idx), 0) != 1:
                raise ValueError("Each IForward input frame must commit observation memory exactly once")

        current_role = str(_cfg_get(dict(_cfg_get(self.supervision_cfg, "current", {}) or {}), "role_name", "final_current_recon"))
        expected_current_refs = set(self._refs_for_frames(int(plan.num_cams), plan.input_frame_indices))
        actual_current_refs = {
            tuple(ref)
            for ref, role in zip(plan.target_refs_flat, plan.target_roles_flat)
            if str(role) == current_role
        }
        if actual_current_refs != expected_current_refs:
            missing = sorted(expected_current_refs - actual_current_refs)
            extra = sorted(actual_current_refs - expected_current_refs)
            raise ValueError(
                "IForward final_current_recon must exactly cover all cams of input frames. "
                f"missing={missing[:8]} extra={extra[:8]}"
            )

        nearby_cfg = dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {})
        nearby_role = str(_cfg_get(nearby_cfg, "role_name", "final_nearby_rollout"))
        nearby_refs = {
            tuple(ref)
            for ref, role in zip(plan.target_refs_flat, plan.target_roles_flat)
            if str(role) == nearby_role
        }
        input_frames = set(int(x) for x in plan.input_frame_indices)
        nearby_frames = set(int(x) for x in plan.final_supervision.nearby_frames)
        if nearby_frames & input_frames:
            raise ValueError("IForward nearby frames must not overlap input frames")
        if nearby_refs & set(plan.evidence_refs_flat):
            raise ValueError("IForward nearby refs must not overlap evidence refs")
        kf_to_frames = dict(getattr(sidx, "keyframe_to_frames", {}) or {})
        allowed_nearby = set()
        train_set = set(int(x) for x in getattr(sidx, "train_frame_set", set()))
        for kf in plan.input_keyframe_indices:
            allowed_nearby.update(int(x) for x in list(kf_to_frames.get(int(kf), []) or []) if int(x) in train_set)
        if not nearby_frames <= (allowed_nearby - input_frames):
            raise ValueError("IForward nearby frames must come from rollout keyframe span and exclude inputs")

        if not bool(plan.detach_graph_after_rollout):
            raise ValueError("IForward detach_graph_after_rollout must be true")
        if bool(plan.reset_scene_state_before_rollout) != (int(plan.rollout_idx_in_episode) == 0):
            raise ValueError("IForward reset_scene_state_before_rollout must be true only on first rollout")
        if bool(plan.carry_scene_state_after_rollout) == bool(plan.episode_end_after_rollout):
            raise ValueError(
                "IForward carry_scene_state_after_rollout must be the inverse of episode_end_after_rollout"
            )

        if bool(_cfg_get(self.leakage_check_cfg, "forbid_test_refs_in_train", True)) and hasattr(self.dataset, "validate_image_ref"):
            for ref in list(plan.evidence_refs_flat) + list(plan.target_refs_flat):
                self.dataset.validate_image_ref(int(plan.scene_id), int(plan.segment_id), tuple(ref), purpose="train")

    def _validate_v3_plan(self, plan: IForwardRolloutPlan, *, sidx: Any) -> None:
        if str(plan.scheduler_version) != IFORWARD_V3_SCHEDULER_VERSION:
            raise ValueError(f"expected scheduler_version={IFORWARD_V3_SCHEDULER_VERSION}")
        if int(plan.inner_K) != int(plan.blocks_per_rollout) * int(plan.repeats_per_block):
            raise ValueError("IForward v3 inner_K must equal blocks_per_rollout * repeats_per_block")
        if len(plan.steps) != int(plan.inner_K):
            raise ValueError("IForward v3 len(steps) must equal inner_K")
        if not plan.evidence_refs_flat or not plan.target_refs_flat:
            raise ValueError("IForward v3 requires non-empty evidence and target refs")
        if len(plan.target_refs_flat) != len(plan.target_roles_flat):
            raise ValueError("IForward v3 target refs/roles length mismatch")
        expected_cams = set(range(int(plan.num_cams)))
        enter_counts: Dict[int, int] = {}
        exit_counts: Dict[int, int] = {}
        for step in plan.steps:
            refs = [tuple(ref) for ref in list(step.evidence_refs)]
            if not refs:
                raise ValueError(f"IForward v3 step {step.step_idx} has empty evidence_refs")
            frames = {int(ref[0]) for ref in refs}
            cams = {int(ref[1]) for ref in refs}
            if frames != {int(step.source_frame_idx)}:
                raise ValueError("IForward v3 step evidence refs must match source_frame_idx")
            if cams != expected_cams:
                raise ValueError("IForward v3 step evidence must cover all cams")
            if bool(step.detach_before_step) or bool(step.detach_after_step):
                raise ValueError("IForward v3 forbids detach inside rollout")
            if bool(step.allow_step_render_loss) or step.step_loss_refs:
                raise ValueError("IForward v3 forbids intermediate step render loss")
            if bool(step.commit_observation_memory) != (int(step.repeat_idx) == 0):
                raise ValueError("IForward v3 commit_observation_memory must be first repeat only")
            if not bool(step.update_optimizer_memory):
                raise ValueError("IForward v3 update_optimizer_memory must be true for every repeat")
            if bool(step.is_block_enter) != (int(step.repeat_idx) == 0):
                raise ValueError("IForward v3 is_block_enter must be explicit on repeat_idx=0")
            expected_exit = bool(int(step.repeat_idx) == int(plan.repeats_per_block) - 1)
            if bool(step.is_block_exit) != expected_exit or bool(step.is_frame_exit) != expected_exit:
                raise ValueError("IForward v3 block/frame exit must be explicit on the last repeat")
            if int(step.episode_visit_idx) < 0 or int(step.optimizer_step_idx_in_episode) < 0:
                raise ValueError("IForward v3 requires visit and optimizer clocks")
            if int(step.window_hash) != int(plan.window_hash):
                raise ValueError("IForward v3 step.window_hash must match plan.window_hash")
            if bool(step.is_block_enter):
                enter_counts[int(step.block_id)] = enter_counts.get(int(step.block_id), 0) + 1
            if bool(step.is_block_exit):
                exit_counts[int(step.block_id)] = exit_counts.get(int(step.block_id), 0) + 1
        for block_id in plan.window_block_ids:
            if enter_counts.get(int(block_id), 0) != 1:
                raise ValueError("Each IForward v3 block must have exactly one enter event")
            if exit_counts.get(int(block_id), 0) != 1:
                raise ValueError("Each IForward v3 block must have exactly one exit event")

        current_role = str(_cfg_get(dict(_cfg_get(self.supervision_cfg, "current", {}) or {}), "role_name", "final_current_recon"))
        history_role = str(_cfg_get(dict(_cfg_get(self.supervision_cfg, "history_replay", {}) or {}), "role_name", "final_history_replay"))
        nearby_role = str(_cfg_get(dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {}), "role_name", "final_nearby_rollout"))
        by_role: Dict[str, set[ImageRef]] = {}
        for ref, role in zip(plan.target_refs_flat, plan.target_roles_flat):
            by_role.setdefault(str(role), set()).add(tuple(ref))
        current_refs = by_role.get(current_role, set())
        history_refs = by_role.get(history_role, set())
        nearby_refs = by_role.get(nearby_role, set())
        expected_current = set(self._refs_for_frames(int(plan.num_cams), plan.input_frame_indices))
        if current_refs != expected_current:
            raise ValueError("IForward v3 current refs must exactly cover all rollout input frames")
        if history_refs & current_refs:
            raise ValueError("IForward v3 history refs must exclude current refs")
        evidence_refs = set(tuple(ref) for ref in plan.evidence_refs_flat)
        if nearby_refs & (evidence_refs | current_refs | history_refs):
            raise ValueError("IForward v3 nearby refs must exclude evidence/current/history refs")
        kf_to_frames = dict(getattr(sidx, "keyframe_to_frames", {}) or {})
        train_set = set(int(x) for x in getattr(sidx, "train_frame_set", set()))
        allowed_nearby: set[int] = set()
        for block_id in plan.window_block_ids:
            keyframe_idx = int(plan.keyframe_window[int(block_id)])
            allowed_nearby.update(int(x) for x in list(kf_to_frames.get(keyframe_idx, []) or []) if int(x) in train_set)
        nearby_frames = {int(ref[0]) for ref in nearby_refs}
        if not nearby_frames <= allowed_nearby:
            raise ValueError("IForward v3 nearby frames must come from current rollout blocks")
        if not bool(plan.detach_graph_after_rollout):
            raise ValueError("IForward v3 detach_graph_after_rollout must be true")
        if bool(plan.reset_scene_state_before_rollout) != (int(plan.rollout_idx_in_episode) == 0):
            raise ValueError("IForward v3 reset flag must be true only on first rollout")
        if bool(plan.carry_scene_state_after_rollout) == bool(plan.episode_end_after_rollout):
            raise ValueError("IForward v3 carry flag must be inverse of episode_end")
        if bool(_cfg_get(self.leakage_check_cfg, "forbid_test_refs_in_train", True)) and hasattr(self.dataset, "validate_image_ref"):
            for ref in list(plan.evidence_refs_flat) + list(plan.target_refs_flat):
                self.dataset.validate_image_ref(int(plan.scene_id), int(plan.segment_id), tuple(ref), purpose="train")

    def _batch_from_plan(self, plan: IForwardRolloutPlan) -> Dict[str, Any]:
        if not hasattr(self.dataset, "_assemble_segment_batch_from_iforward_request"):
            raise ValueError("TrainSchedulerIForward requires dataset._assemble_segment_batch_from_iforward_request")
        return self.dataset._assemble_segment_batch_from_iforward_request(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            plan=plan,
            include_test=bool(self.include_test),
        )

    def _preload_refs_from_plan(self, plan: IForwardRolloutPlan) -> List[ImageRef]:
        return _dedupe_refs_keep_order(list(plan.evidence_refs_flat) + list(plan.target_refs_flat))

    def _emit_preload_hint_for_plan(
        self,
        plan: IForwardRolloutPlan,
        *,
        warm_flag_key: str = "warm_current_rollout_refs",
        emit_event: bool = True,
        event_type: str = "preload_hint",
    ) -> None:
        if not bool(_cfg_get(self.preload_cfg, "emit_hints", True)):
            return
        if not bool(_cfg_get(self.preload_cfg, warm_flag_key, True)):
            return
        if not hasattr(self.dataset, "build_preload_hint") or not hasattr(self.dataset, "submit_preload_hint"):
            return
        refs = self._preload_refs_from_plan(plan)
        if not refs:
            return
        scope = str(_cfg_get(self.preload_cfg, "hint_scope_for_exact_refs", "v9_role_refs"))
        hint = self.dataset.build_preload_hint(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            future_image_refs=refs,
            scope=scope,
        )
        self.dataset.submit_preload_hint(
            hint=hint,
            hint_scope=scope,
            epoch_idx=int(self.epoch_idx),
            global_step=int(self.global_step),
            block_idx_global=int(plan.rollout_id_global),
            include_test=bool(self.include_test),
        )
        if not bool(emit_event):
            return
        self._emit(
            {
                "type": str(event_type),
                "scheduler_version": self._scheduler_version(),
                "global_step": int(self.global_step),
                "scene_id": int(plan.scene_id),
                "segment_id": int(plan.segment_id),
                "episode_id": int(plan.episode_id),
                "rollout_id_global": int(plan.rollout_id_global),
                "hint_scope": scope,
                "num_future_image_refs": int(len(refs)),
            }
        )

    def _emit_preload_hint_for_next_rollout(self) -> None:
        if not bool(_cfg_get(self.preload_cfg, "warm_next_rollout_refs", False)):
            return
        state = self.state_dict()
        try:
            episode = self._ensure_episode_with_rollout_available()
            plan = self._build_rollout_plan(episode)
            self._emit_preload_hint_for_plan(
                plan,
                warm_flag_key="warm_next_rollout_refs",
                emit_event=False,
                event_type="preload_hint_next_rollout",
            )
        finally:
            self.load_state_dict(state)

    def _commit_v3_rollout_visits_to_episode(self, episode: Dict[str, Any], plan: IForwardRolloutPlan) -> None:
        counts = dict(episode.get("window_counts", {}) or {})
        counts[int(plan.window_hash)] = int(counts.get(int(plan.window_hash), 0)) + 1
        episode["window_counts"] = counts

        visited = [tuple(ref) for ref in list(episode.get("visited_refs", []) or [])]
        visited_set = {tuple(ref) for ref in list(episode.get("visited_ref_set", set()) or set())}
        visited_frames = [int(x) for x in list(episode.get("visited_frames", []) or [])]
        visited_frame_set = {int(x) for x in list(episode.get("visited_frame_set", set()) or set())}
        for ref in list(plan.evidence_refs_flat):
            item = (int(ref[0]), int(ref[1]))
            frame_idx = int(item[0])
            if frame_idx not in visited_frame_set:
                visited_frames.append(frame_idx)
                visited_frame_set.add(frame_idx)
            if item in visited_set:
                continue
            visited.append(item)
            visited_set.add(item)
        episode["visited_frames"] = visited_frames
        episode["visited_frame_set"] = visited_frame_set
        episode["visited_refs"] = visited
        episode["visited_ref_set"] = visited_set

        block_counts = dict(episode.get("block_visit_counts", {}) or {})
        for block_id in list(plan.window_block_ids):
            block_counts[int(block_id)] = int(block_counts.get(int(block_id), 0)) + 1
        episode["block_visit_counts"] = block_counts
        episode["episode_visit_idx"] = int(episode.get("episode_visit_idx", 0)) + int(plan.blocks_per_rollout)
        episode["optimizer_step_idx_in_episode"] = int(episode.get("optimizer_step_idx_in_episode", 0)) + int(plan.inner_K)

    def _next_batch_v3(self) -> Dict[str, Any]:
        episode = self._ensure_episode_with_rollout_available()
        plan = self._build_rollout_plan(episode)
        batch = self._batch_from_plan(plan)
        self._emit_preload_hint_for_plan(plan)
        self._commit_v3_rollout_visits_to_episode(episode, plan)
        first_step = plan.steps[0] if plan.steps else None
        first_source_ref = plan.evidence_refs_flat[0] if plan.evidence_refs_flat else (-1, -1)
        self._last_info = {
            "scheduler_version": IFORWARD_V3_SCHEDULER_VERSION,
            "model_family": IFORWARD_MODEL_FAMILY,
            "global_step": int(self.global_step),
            "scene_id": int(plan.scene_id),
            "segment_id": int(plan.segment_id),
            "episode_id": int(plan.episode_id),
            "episode_idx_global": int(plan.episode_id),
            "epoch_idx": int(self.epoch_idx),
            "block_idx_global": int(plan.rollout_id_global),
            "block_idx_in_episode": int(plan.window_block_ids[0]) if plan.window_block_ids else -1,
            "block_idx_in_segment": int(plan.window_start),
            "rollout_id_global": int(plan.rollout_id_global),
            "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
            "rollouts_per_episode": int(plan.rollouts_per_episode),
            "source_frame_idx": int(first_step.source_frame_idx) if first_step is not None else -1,
            "source_keyframe_idx": int(first_step.source_keyframe_idx) if first_step is not None else -1,
            "source_image_ref": (int(first_source_ref[0]), int(first_source_ref[1])),
            "target_image_refs": [(int(ref[0]), int(ref[1])) for ref in plan.target_refs_flat],
            "window_start": int(plan.window_start),
            "window_end": int(plan.window_end),
            "window_block_ids": [int(x) for x in plan.window_block_ids],
            "window_hash": int(plan.window_hash),
            "window_revisit_count": int(plan.window_revisit_count),
            "unique_windows_seen": int(plan.unique_windows_seen),
            "is_repeated_window": bool(plan.is_repeated_window),
            "U": int(plan.blocks_per_rollout),
            "K_u_nominal": int(plan.repeats_per_block),
            "K_u_effective": int(plan.repeats_per_block),
            "K_steps": int(plan.inner_K),
            "K_steps_effective": int(plan.inner_K),
            "R_steps": int(plan.repeats_per_block),
            "T_steps": int(plan.blocks_per_rollout),
            "inner_K": int(plan.inner_K),
            "shape_name": str(plan.shape_name),
            "block_order": "iforward_v3_random_window",
        }
        self._emit(
            {
                "type": "rollout_batch_emitted",
                "scheduler_version": IFORWARD_V3_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(plan.scene_id),
                "segment_id": int(plan.segment_id),
                "episode_id": int(plan.episode_id),
                "rollout_id_global": int(plan.rollout_id_global),
                "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
                "window_start": int(plan.window_start),
                "window_hash": int(plan.window_hash),
                "inner_K": int(plan.inner_K),
            }
        )
        episode["rollout_idx_in_episode"] = int(episode["rollout_idx_in_episode"]) + 1
        self.global_step += 1
        self._rollout_id_global += 1
        if bool(plan.episode_end_after_rollout):
            self._emit(
                {
                    "type": "episode_end",
                    "scheduler_version": IFORWARD_V3_SCHEDULER_VERSION,
                    "global_step": int(self.global_step),
                    "scene_id": int(plan.scene_id),
                    "segment_id": int(plan.segment_id),
                    "episode_id": int(plan.episode_id),
                    "rollout_id_global": int(plan.rollout_id_global),
                    "reason": "rollouts_per_episode_reached",
                }
            )
            self._current_episode = None
        self._emit_preload_hint_for_next_rollout()
        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
        return batch

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        state = self.state_dict()
        try:
            episode = self._ensure_episode_with_rollout_available()
            plan = self._build_rollout_plan(episode)
            batch = self._batch_from_plan(plan)
            batch["_iforward_peek"] = True
            return batch
        finally:
            self.load_state_dict(state)

    def next_batch(self) -> Dict[str, Any]:
        if self._is_v3():
            return self._next_batch_v3()
        episode = self._ensure_episode_with_rollout_available()
        plan = self._build_rollout_plan(episode)
        batch = self._batch_from_plan(plan)
        self._emit_preload_hint_for_plan(plan)
        first_step = plan.steps[0] if plan.steps else None
        first_source_ref = plan.evidence_refs_flat[0] if plan.evidence_refs_flat else (-1, -1)

        self._last_info = {
            "scheduler_version": IFORWARD_SCHEDULER_VERSION,
            "model_family": IFORWARD_MODEL_FAMILY,
            "global_step": int(self.global_step),
            "scene_id": int(plan.scene_id),
            "segment_id": int(plan.segment_id),
            "episode_id": int(plan.episode_id),
            "episode_idx_global": int(plan.episode_id),
            "epoch_idx": int(self.epoch_idx),
            "block_idx_global": int(plan.rollout_id_global),
            "block_idx_in_episode": int(plan.episode_block_indices[0]) if plan.episode_block_indices else -1,
            "block_idx_in_segment": int(plan.episode_block_indices[0]) if plan.episode_block_indices else -1,
            "rollout_id_global": int(plan.rollout_id_global),
            "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
            "source_frame_idx": int(first_step.source_frame_idx) if first_step is not None else -1,
            "source_keyframe_idx": int(first_step.source_keyframe_idx) if first_step is not None else -1,
            "source_image_ref": (int(first_source_ref[0]), int(first_source_ref[1])),
            "target_image_refs": [(int(ref[0]), int(ref[1])) for ref in plan.target_refs_flat],
            "U": int(plan.actual_blocks_per_rollout),
            "K_u_nominal": int(plan.repeats_per_block),
            "K_u_effective": int(plan.repeats_per_block),
            "K_steps": int(plan.inner_K),
            "K_steps_effective": int(plan.inner_K),
            "R_steps": int(plan.repeats_per_block),
            "T_steps": int(plan.actual_blocks_per_rollout),
            "inner_K": int(plan.inner_K),
            "shape_name": str(plan.shape_name),
            "block_selection_policy": str(plan.request_meta.get("block_selection_policy", self._block_selection_policy())),
            "source_frame_sampling_policy": str(plan.request_meta.get("source_frame_sampling_policy", self._block_source_frame_policy())),
            "rollout_start_block_idx": int(plan.request_meta.get("rollout_start_block_idx", plan.episode_block_indices[0] if plan.episode_block_indices else -1)),
            "actual_blocks_per_rollout": int(plan.actual_blocks_per_rollout),
            "repeats_per_block": int(plan.repeats_per_block),
            "block_order": "iforward_rollout",
        }
        self._emit(
            {
                "type": "rollout_batch_emitted",
                "scheduler_version": IFORWARD_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(plan.scene_id),
                "segment_id": int(plan.segment_id),
                "episode_id": int(plan.episode_id),
                "rollout_id_global": int(plan.rollout_id_global),
                "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
                "inner_K": int(plan.inner_K),
            }
        )

        if self._block_selection_policy() == "random_start_contiguous":
            used_starts = list(episode.get("used_rollout_starts", []) or [])
            used_starts.append(int(plan.request_meta.get("rollout_start_block_idx", plan.episode_block_indices[0])))
            episode["used_rollout_starts"] = [int(x) for x in used_starts]
            episode["last_rollout_start_block_idx"] = int(
                plan.request_meta.get("rollout_start_block_idx", plan.episode_block_indices[0])
            )
        episode["block_cursor"] = int(episode["block_cursor"]) + int(plan.actual_blocks_per_rollout)
        episode["rollout_idx_in_episode"] = int(episode["rollout_idx_in_episode"]) + 1
        self.global_step += 1
        self._rollout_id_global += 1

        if bool(plan.episode_end_after_rollout):
            if bool(plan.request_meta.get("tail_skipped_after_rollout", False)):
                self._emit(
                    {
                        "type": "episode_tail_skipped",
                        "scheduler_version": IFORWARD_SCHEDULER_VERSION,
                        "global_step": int(self.global_step),
                        "scene_id": int(plan.scene_id),
                        "segment_id": int(plan.segment_id),
                        "episode_id": int(plan.episode_id),
                        "rollout_idx_in_episode": int(plan.rollout_idx_in_episode) + 1,
                        "remaining_blocks": int(plan.request_meta.get("tail_skipped_remaining_blocks", 0)),
                        "reason": "remaining_blocks_lt_required_rollout_after_emit",
                    }
                )
            self._emit(
                {
                    "type": "episode_end",
                    "scheduler_version": IFORWARD_SCHEDULER_VERSION,
                    "global_step": int(self.global_step),
                    "scene_id": int(plan.scene_id),
                    "segment_id": int(plan.segment_id),
                    "episode_id": int(plan.episode_id),
                    "rollout_id_global": int(plan.rollout_id_global),
                }
            )
            self._current_episode = None

        self._emit_preload_hint_for_next_rollout()

        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_class": type(self).__name__,
            "scheduler_version": self._scheduler_version(),
            "global_step": int(self.global_step),
            "epoch_idx": int(self.epoch_idx),
            "episode_id_next": int(self._episode_id_next),
            "rollout_id_global": int(self._rollout_id_global),
            "episode_plan": copy.deepcopy(self._episode_plan),
            "episode_plan_cursor": int(self._episode_plan_cursor),
            "current_episode": copy.deepcopy(self._current_episode),
            "pending_events": copy.deepcopy(self._pending_events),
            "last_info": copy.deepcopy(self._last_info),
            "rng_state": copy.deepcopy(self.rng.getstate()),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if str(state.get("scheduler_version", "")) != self._scheduler_version():
            raise ValueError(f"expected scheduler_version={self._scheduler_version()}")
        self.global_step = int(state.get("global_step", 0))
        self.epoch_idx = int(state.get("epoch_idx", 0))
        self._episode_id_next = int(state.get("episode_id_next", 0))
        self._rollout_id_global = int(state.get("rollout_id_global", 0))
        self._episode_plan = copy.deepcopy(list(state.get("episode_plan", []) or []))
        self._episode_plan_cursor = int(state.get("episode_plan_cursor", 0))
        self._current_episode = copy.deepcopy(state.get("current_episode", None))
        self._pending_events = copy.deepcopy(list(state.get("pending_events", []) or []))
        self._last_info = copy.deepcopy(dict(state.get("last_info", {}) or {}))
        if self._current_episode is not None:
            visited = [tuple(ref) for ref in list(self._current_episode.get("visited_refs", []) or [])]
            self._current_episode["visited_refs"] = visited
            self._current_episode["visited_ref_set"] = {tuple(ref) for ref in visited}
            frames = [int(x) for x in list(self._current_episode.get("visited_frames", []) or [])]
            if not frames:
                seen_frames: set[int] = set()
                for ref in visited:
                    frame = int(ref[0])
                    if frame in seen_frames:
                        continue
                    seen_frames.add(frame)
                    frames.append(frame)
            self._current_episode["visited_frames"] = frames
            self._current_episode["visited_frame_set"] = {int(frame) for frame in frames}
        rng_state = state.get("rng_state", state.get("random_state", None))
        if rng_state is not None:
            self.rng.setstate(rng_state)


__all__ = [
    "IFORWARD_MODEL_FAMILY",
    "IFORWARD_SCHEDULER_VERSION",
    "IFORWARD_V3_SCHEDULER_VERSION",
    "IForwardFinalSupervisionPlan",
    "IForwardRolloutPlan",
    "IForwardRolloutShape",
    "IForwardStepPlan",
    "TrainSchedulerIForward",
]
