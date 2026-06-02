from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

ImageRef = Tuple[int, int]

IFORWARD_SCHEDULER_VERSION = "iforward_v1"
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
    episode_block_idx: int
    rollout_block_rank: int
    repeat_idx: int
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
            "scheduler_version": IFORWARD_SCHEDULER_VERSION,
            "model_family": IFORWARD_MODEL_FAMILY,
            "global_step": 0,
        }
        self._rebuild_epoch_plan()

    def _validate_static_cfg(self) -> None:
        version = str(
            self.version
            or _cfg_get(self.rollout_cfg, "version", _cfg_get(self.traversal_cfg, "version", IFORWARD_SCHEDULER_VERSION))
        )
        if version not in (IFORWARD_SCHEDULER_VERSION, "iforward_v1"):
            raise ValueError(f"IForward v1 requires version={IFORWARD_SCHEDULER_VERSION}, got {version!r}")

        traversal_mode = str(_cfg_get(self.traversal_cfg, "traversal_mode", _cfg_get(self.traversal_cfg, "mode", "episode_serial")))
        if traversal_mode != "episode_serial":
            raise ValueError("scheduler_iforward IForward v1 requires traversal.traversal_mode=episode_serial")
        for name in ("scene_order", "segment_order"):
            val = str(_cfg_get(self.traversal_cfg, name, "shuffle_per_epoch" if name == "scene_order" else "ascending"))
            if val not in ("ascending", "shuffle_per_epoch"):
                raise ValueError(f"scheduler_iforward.traversal.{name} must be ascending or shuffle_per_epoch")

        if str(_cfg_get(self.episode_cfg, "source_mode", "keyframes")) != "keyframes":
            raise ValueError("scheduler_iforward IForward v1 requires episode.source_mode=keyframes")
        if str(_cfg_get(self.episode_cfg, "block_source_frame_policy", "random_within_keyframe_once_per_episode")) != "random_within_keyframe_once_per_episode":
            raise ValueError(
                "scheduler_iforward IForward v1 requires "
                "episode.block_source_frame_policy=random_within_keyframe_once_per_episode"
            )
        if int(_cfg_get(self.episode_cfg, "blocks_per_episode", 8)) < 1:
            raise ValueError("scheduler_iforward.episode.blocks_per_episode must be >= 1")
        if int(_cfg_get(self.episode_cfg, "episode_stride", _cfg_get(self.episode_cfg, "blocks_per_episode", 8))) < 1:
            raise ValueError("scheduler_iforward.episode.episode_stride must be >= 1")
        if int(_cfg_get(self.episode_cfg, "min_blocks_per_episode", 2)) < 1:
            raise ValueError("scheduler_iforward.episode.min_blocks_per_episode must be >= 1")

        if str(_cfg_get(self.rollout_cfg, "block_selection_policy", "next_contiguous")) != "next_contiguous":
            raise ValueError("scheduler_iforward IForward v1 requires rollout.block_selection_policy=next_contiguous")
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

    def _raw_rollout_shapes(self) -> List[Dict[str, Any]]:
        raw = [dict(x) for x in list(_cfg_get(self.rollout_cfg, "shapes", []) or [])]
        if raw:
            return raw
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
        vals = list(shapes)
        if not vals:
            raise ValueError("scheduler_iforward cannot sample from an empty rollout shape list")
        return self.rng.choices(vals, weights=[float(x.prob) for x in vals], k=1)[0]

    def _sample_shape(self) -> IForwardRolloutShape:
        return self._sample_shape_from(self._active_shapes())

    def _sample_shape_for_remaining(self, remaining_blocks: int) -> IForwardRolloutShape:
        shapes = self._active_shapes()
        remaining = int(remaining_blocks)
        if remaining < 1:
            raise ValueError("remaining_blocks must be >= 1")
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

    def _start_next_episode(self) -> None:
        if self._episode_plan_cursor >= len(self._episode_plan):
            self._rebuild_epoch_plan()
        spec = dict(self._episode_plan[int(self._episode_plan_cursor)])
        self._episode_plan_cursor += 1

        sidx = self.dataset.get_segment_index(int(spec["scene_id"]), int(spec["segment_id"]))
        frame_chain: List[int] = []
        for keyframe_idx in list(spec["keyframe_window"]):
            candidates = self._keyframe_train_frames(sidx, int(keyframe_idx))
            if not candidates:
                raise ValueError(
                    "scheduler_iforward keyframe has no train frames: "
                    f"scene={spec['scene_id']} segment={spec['segment_id']} keyframe={int(keyframe_idx)}"
                )
            frame_chain.append(int(self.rng.choice(candidates)))

        episode_id = int(self._episode_id_next)
        self._episode_id_next += 1
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
        }
        self._emit(
            {
                "type": "episode_begin",
                "scheduler_version": IFORWARD_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(spec["scene_id"]),
                "segment_id": int(spec["segment_id"]),
                "episode_id": int(episode_id),
                "episode_start_keyframe_pos": int(spec["episode_start_keyframe_pos"]),
                "episode_num_blocks": int(len(frame_chain)),
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

    def _build_rollout_plan(self, episode: Dict[str, Any]) -> IForwardRolloutPlan:
        sidx = self.dataset.get_segment_index(int(episode["scene_id"]), int(episode["segment_id"]))
        block_cursor = int(episode["block_cursor"])
        frame_chain = [int(x) for x in list(episode["frame_chain"])]
        keyframe_window = [int(x) for x in list(episode["keyframe_window"])]
        if block_cursor >= len(frame_chain):
            raise ValueError("IForward episode block cursor is already at end")

        remaining_blocks = int(len(frame_chain) - block_cursor)
        shape = self._sample_shape_for_remaining(remaining_blocks)
        requested_blocks = int(shape.blocks_per_rollout)
        repeats = int(shape.repeats_per_block)
        end = min(block_cursor + requested_blocks, len(frame_chain))
        episode_blocks = list(range(block_cursor, end))
        if len(episode_blocks) < requested_blocks:
            allow_short = bool(_cfg_get(self.rollout_cfg, "allow_short_final_rollout", True))
            min_blocks = int(_cfg_get(self.rollout_cfg, "min_blocks_per_rollout", 1))
            if not allow_short or len(episode_blocks) < min_blocks:
                raise ValueError(
                    "scheduler_iforward final rollout is shorter than requested and short rollout is disabled"
                )

        input_keyframes = [int(keyframe_window[int(idx)]) for idx in episode_blocks]
        input_frames = [int(frame_chain[int(idx)]) for idx in episode_blocks]
        delivery_blocks = list(episode_blocks)
        delivery_frames = [int(frame_chain[int(idx)]) for idx in delivery_blocks]
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
            frame_idx = int(frame_chain[int(block_idx)])
            keyframe_idx = int(keyframe_window[int(block_idx)])
            evidence_refs = self._refs_for_frames(num_cams, [frame_idx])
            for repeat_idx in range(repeats):
                step_idx = len(steps)
                steps.append(
                    IForwardStepPlan(
                        step_idx=int(step_idx),
                        episode_block_idx=int(block_idx),
                        rollout_block_rank=int(rollout_rank),
                        repeat_idx=int(repeat_idx),
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
        episode_end = bool(end >= len(frame_chain))
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
            "inner_K": int(actual_inner_k),
            "shape_name": str(shape_name),
            "requested_shape_name": str(shape.name),
            "blocks_per_rollout": int(requested_blocks),
            "requested_blocks_per_rollout": int(requested_blocks),
            "actual_blocks_per_rollout": int(actual_blocks),
            "repeats_per_block": int(repeats),
            "requested_inner_K": int(requested_inner_k),
            "actual_inner_K": int(actual_inner_k),
            "short_rollout": bool(short_rollout),
            "short_rollout_reason": str(short_reason),
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
            frame_chain=[int(x) for x in frame_chain],
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
            if bool(step.commit_observation_memory):
                frame_repeat_commit_counts[int(step.source_frame_idx)] = frame_repeat_commit_counts.get(int(step.source_frame_idx), 0) + 1

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
                "scheduler_version": IFORWARD_SCHEDULER_VERSION,
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
            episode = self._ensure_episode()
            plan = self._build_rollout_plan(episode)
            self._emit_preload_hint_for_plan(
                plan,
                warm_flag_key="warm_next_rollout_refs",
                emit_event=False,
                event_type="preload_hint_next_rollout",
            )
        finally:
            self.load_state_dict(state)

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        state = self.state_dict()
        try:
            episode = self._ensure_episode()
            plan = self._build_rollout_plan(episode)
            batch = self._batch_from_plan(plan)
            batch["_iforward_peek"] = True
            return batch
        finally:
            self.load_state_dict(state)

    def next_batch(self) -> Dict[str, Any]:
        episode = self._ensure_episode()
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

        episode["block_cursor"] = int(episode["block_cursor"]) + int(plan.actual_blocks_per_rollout)
        episode["rollout_idx_in_episode"] = int(episode["rollout_idx_in_episode"]) + 1
        self.global_step += 1
        self._rollout_id_global += 1

        if bool(plan.episode_end_after_rollout):
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
            "scheduler_version": IFORWARD_SCHEDULER_VERSION,
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
        if str(state.get("scheduler_version", "")) != IFORWARD_SCHEDULER_VERSION:
            raise ValueError(f"expected scheduler_version={IFORWARD_SCHEDULER_VERSION}")
        self.global_step = int(state.get("global_step", 0))
        self.epoch_idx = int(state.get("epoch_idx", 0))
        self._episode_id_next = int(state.get("episode_id_next", 0))
        self._rollout_id_global = int(state.get("rollout_id_global", 0))
        self._episode_plan = copy.deepcopy(list(state.get("episode_plan", []) or []))
        self._episode_plan_cursor = int(state.get("episode_plan_cursor", 0))
        self._current_episode = copy.deepcopy(state.get("current_episode", None))
        self._pending_events = copy.deepcopy(list(state.get("pending_events", []) or []))
        self._last_info = copy.deepcopy(dict(state.get("last_info", {}) or {}))
        rng_state = state.get("rng_state", state.get("random_state", None))
        if rng_state is not None:
            self.rng.setstate(rng_state)


__all__ = [
    "IFORWARD_MODEL_FAMILY",
    "IFORWARD_SCHEDULER_VERSION",
    "IForwardFinalSupervisionPlan",
    "IForwardRolloutPlan",
    "IForwardRolloutShape",
    "IForwardStepPlan",
    "TrainSchedulerIForward",
]
