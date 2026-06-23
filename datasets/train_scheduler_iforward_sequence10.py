from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from datasets.train_scheduler_iforward import (
    IFORWARD_MODEL_FAMILY,
    IForwardFinalSupervisionPlan,
    IForwardRolloutPlan,
    IForwardStepPlan,
    ImageRef,
    _cfg_get,
    _dedupe_refs_keep_order,
)

IFORWARD_SEQUENCE10_SCHEDULER_VERSION = "iforward_sequence10_v1"
IFORWARD_SEQUENCE10_CURRENT_ROLE = "final_current_recon"
IFORWARD_SEQUENCE10_HISTORY_ROLE = "final_history_replay"


class Sequence10DatasetLike(Protocol):
    _initialized: bool

    def initialize(self) -> None: ...
    def list_training_scene_ids(self) -> List[int]: ...
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> Any: ...


@dataclass(frozen=True)
class Sequence10Spec:
    scene_id: int
    segment_id: int
    sequence_id: int
    stride: int
    start_block_pos: int
    block_ids: Tuple[int, ...]
    keyframe_indices: Tuple[int, ...]
    source_frame_indices: Tuple[int, ...]
    source_refs_by_pos: Tuple[Tuple[ImageRef, ...], ...]

    def __post_init__(self) -> None:
        if len(self.block_ids) != 10:
            raise ValueError("Sequence10Spec.block_ids must have length 10")
        if len(set(int(x) for x in self.block_ids)) != 10:
            raise ValueError("Sequence10Spec.block_ids must be unique")
        if len(self.keyframe_indices) != 10 or len(self.source_frame_indices) != 10:
            raise ValueError("Sequence10Spec keyframe/source frames must have length 10")
        if int(self.stride) not in (1, 2):
            raise ValueError("Sequence10Spec.stride must be 1 or 2")


@dataclass(frozen=True)
class Sequence10VisitSpec:
    sequence_pos: int
    block_id: int
    keyframe_idx: int
    source_frame_idx: int
    source_refs: Tuple[ImageRef, ...]
    visit_kind: str
    frame_gap: int
    repeats_per_block: int
    temporal_read: bool
    temporal_commit: bool
    observation_commit: bool
    update_optimizer_memory: bool
    physical_time_advance: bool


@dataclass(frozen=True)
class Sequence10StepPlan(IForwardStepPlan):
    sequence_pos: int = -1
    visit_kind: str = ""
    frame_gap: int = 0
    temporal_read: bool = True
    temporal_commit: bool = False
    physical_time_advance: bool = False
    scheduler_phase: str = ""


@dataclass(frozen=True)
class Sequence10RolloutPlan(IForwardRolloutPlan):
    sequence_id: int = -1
    sequence_length: int = 10
    sequence_stride: int = 1
    sequence_start_block_pos: int = -1
    sequence_block_ids: List[int] = field(default_factory=list)
    sequence_keyframe_indices: List[int] = field(default_factory=list)
    sequence_source_frame_indices: List[int] = field(default_factory=list)
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
class Sequence10EpisodePlan:
    spec: Sequence10Spec
    rollouts: Tuple[Sequence10RolloutPlan, ...]
    repair_enabled: bool


def _stable_hash_int(parts: Sequence[Any]) -> int:
    text = ":".join(str(x) for x in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _rng_token(rng: random.Random) -> int:
    return int(rng.getrandbits(63))


class TrainSchedulerIForwardSequence10:
    """Sequence10 scheduler for Stage 2_1.

    This scheduler intentionally does not expose the older shape/window policy
    surface. It emits one bootstrap rollout before step 5000, then fixed
    sequence10 causal episodes, with an optional B10R1 repair rollout.
    """

    def __init__(
        self,
        *,
        dataset: Sequence10DatasetLike,
        traversal_cfg: Optional[Any] = None,
        bootstrap_cfg: Optional[Any] = None,
        sequence_cfg: Optional[Any] = None,
        causal_cfg: Optional[Any] = None,
        repair_cfg: Optional[Any] = None,
        supervision_cfg: Optional[Any] = None,
        history_loss_cfg: Optional[Any] = None,
        damage_loss_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        include_test: bool = False,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        seed: Optional[int] = None,
        fail_fast: bool = True,
    ) -> None:
        self.dataset = dataset
        self.traversal_cfg = dict(traversal_cfg or {})
        self.bootstrap_cfg = dict(bootstrap_cfg or {})
        self.sequence_cfg = dict(sequence_cfg or {})
        self.causal_cfg = dict(causal_cfg or {})
        self.repair_cfg = dict(repair_cfg or {})
        self.supervision_cfg = dict(supervision_cfg or {})
        self.history_loss_cfg = dict(history_loss_cfg or {})
        self.damage_loss_cfg = dict(damage_loss_cfg or {})
        self.preload_cfg = dict(preload_cfg or {})
        self.include_test = bool(include_test)
        self.fail_fast = bool(fail_fast)
        if fixed_scene_id is None:
            fixed_scene_id = _cfg_get(self.traversal_cfg, "fixed_scene_id", None)
        if fixed_segment_id is None:
            fixed_segment_id = _cfg_get(self.traversal_cfg, "fixed_segment_id", None)
        self.fixed_scene_id = int(fixed_scene_id) if fixed_scene_id is not None else None
        self.fixed_segment_id = int(fixed_segment_id) if fixed_segment_id is not None else None
        raw_seed = seed if seed is not None else _cfg_get(self.traversal_cfg, "seed", None)
        self.rng = random.Random(int(raw_seed)) if raw_seed is not None else random.Random()

        initialized = getattr(self.dataset, "_initialized", True)
        if initialized is False:
            self.dataset.initialize()

        self._validate_static_cfg()
        self._eligibility_index = self._build_eligibility_index()
        if not self._eligibility_index:
            raise ValueError("iforward_sequence10_v1 found no eligible 10-frame sequences")
        self._eligible_by_scene_segment = self._group_eligibility(self._eligibility_index)
        self._eligible_scenes = sorted(int(x) for x in self._eligible_by_scene_segment.keys())
        self._scene_queue: List[int] = []
        self._segment_queues: Dict[int, List[int]] = {}
        self._last_scene_id: Optional[int] = None

        self.global_step = 0
        self.epoch_idx = 0
        self._episode_id_next = 0
        self._rollout_id_global = 0
        self._episode_plan: Optional[Sequence10EpisodePlan] = None
        self._episode_plan_cursor = 0
        self._pending_events: List[Dict[str, Any]] = []
        self._last_info: Dict[str, Any] = {
            "scheduler_version": IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
            "global_step": 0,
        }

    @property
    def steps_per_block(self) -> int:
        return int(_cfg_get(self.causal_cfg, "repeats_per_block", 4))

    def _validate_static_cfg(self) -> None:
        for legacy_key in ("shapes", "shapes_schedule", "fixed_shape_names", "block_selection_policy", "tail_policy"):
            if legacy_key in self.sequence_cfg or legacy_key in self.causal_cfg or legacy_key in self.repair_cfg:
                raise ValueError(f"iforward_sequence10_v1 forbids legacy shape field {legacy_key!r}")
        if int(_cfg_get(self.sequence_cfg, "length", 10)) != 10:
            raise ValueError("iforward_sequence10_v1 requires sequence.length=10")
        strides = [int(x) for x in list(_cfg_get(self.sequence_cfg, "strides", [1, 2]) or [])]
        if not strides or any(s not in (1, 2) for s in strides):
            raise ValueError("iforward_sequence10_v1 sequence.strides must contain only 1 and/or 2")
        if int(_cfg_get(self.causal_cfg, "blocks_per_rollout", 2)) != 2:
            raise ValueError("iforward_sequence10_v1 causal.blocks_per_rollout must be 2")
        if int(_cfg_get(self.causal_cfg, "repeats_per_block", 4)) != 4:
            raise ValueError("iforward_sequence10_v1 causal.repeats_per_block must be 4")
        if int(_cfg_get(self.repair_cfg, "blocks_per_rollout", 10)) != 10:
            raise ValueError("iforward_sequence10_v1 repair.blocks_per_rollout must be 10")
        if int(_cfg_get(self.repair_cfg, "repeats_per_block", 1)) != 1:
            raise ValueError("iforward_sequence10_v1 repair.repeats_per_block must be 1")
        max_inner_k = int(_cfg_get(self.sequence_cfg, "max_inner_K", 10))
        if max_inner_k != 10:
            raise ValueError("iforward_sequence10_v1 sequence.max_inner_K must be 10")
        bootstrap_end = int(_cfg_get(self.bootstrap_cfg, "end_step", 5000))
        causal_start = int(_cfg_get(self.causal_cfg, "start_step", bootstrap_end))
        if int(causal_start) != int(bootstrap_end):
            raise ValueError("iforward_sequence10_v1 requires causal.start_step == bootstrap.end_step")
        scene_order = str(_cfg_get(self.traversal_cfg, "scene_order", "shuffle_per_epoch"))
        segment_order = str(_cfg_get(self.traversal_cfg, "segment_order", "shuffle_per_epoch"))
        traversal_mode = str(_cfg_get(self.traversal_cfg, "traversal_mode", "scene_round_robin_episode"))
        if scene_order not in {"shuffle_per_epoch", "ordered"}:
            raise ValueError("iforward_sequence10_v1 traversal.scene_order must be shuffle_per_epoch or ordered")
        if segment_order not in {"shuffle_per_epoch", "ordered"}:
            raise ValueError("iforward_sequence10_v1 traversal.segment_order must be shuffle_per_epoch or ordered")
        if traversal_mode != "scene_round_robin_episode":
            raise ValueError("iforward_sequence10_v1 requires traversal.traversal_mode=scene_round_robin_episode")
        if not bool(_cfg_get(self.traversal_cfg, "forbid_consecutive_same_scene", True)):
            raise ValueError("iforward_sequence10_v1 requires traversal.forbid_consecutive_same_scene=true")
        if int(_cfg_get(self.causal_cfg, "rollouts_per_episode", 5)) != 5:
            raise ValueError("iforward_sequence10_v1 causal.rollouts_per_episode must be 5")
        fixed_bools = (
            (self.causal_cfg, "temporal_read", True),
            (self.causal_cfg, "temporal_commit", True),
            (self.causal_cfg, "physical_time_advance", True),
            (self.repair_cfg, "temporal_read", True),
            (self.repair_cfg, "temporal_commit", False),
            (self.repair_cfg, "observation_commit", False),
            (self.repair_cfg, "update_optimizer_memory", False),
            (self.repair_cfg, "physical_time_advance", False),
        )
        for cfg, key, expected in fixed_bools:
            if bool(_cfg_get(cfg, key, expected)) != bool(expected):
                raise ValueError(f"iforward_sequence10_v1 requires {key}={str(expected).lower()} for its fixed protocol")
        hist_cfg = dict(_cfg_get(self.supervision_cfg, "history_replay", {}) or {})
        if not bool(_cfg_get(hist_cfg, "enable", True)):
            raise ValueError("iforward_sequence10_v1 requires supervision.history_replay.enable=true")

    def _scene_segment_pairs(self) -> List[Tuple[int, int]]:
        scenes = [int(self.fixed_scene_id)] if self.fixed_scene_id is not None else [int(x) for x in self.dataset.list_training_scene_ids()]
        pairs: List[Tuple[int, int]] = []
        for scene_id in scenes:
            segments = (
                [int(self.fixed_segment_id)]
                if self.fixed_segment_id is not None
                else [int(x) for x in self.dataset.list_segment_ids(int(scene_id))]
            )
            for segment_id in segments:
                pairs.append((int(scene_id), int(segment_id)))
        if not pairs:
            raise ValueError("iforward_sequence10_v1 has no training scene/segment pairs")
        return pairs

    def _train_frames_for_keyframe(self, sidx: Any, keyframe_idx: int) -> List[int]:
        mapping = getattr(sidx, "keyframe_to_frames", {}) or {}
        frames = [int(x) for x in list(mapping.get(int(keyframe_idx), []))]
        train_set = set(int(x) for x in getattr(sidx, "train_frame_set", set()))
        if train_set:
            frames = [int(x) for x in frames if int(x) in train_set]
        if not frames:
            frame_to_keyframe = getattr(sidx, "frame_to_keyframe", {}) or {}
            frames = [int(f) for f, k in dict(frame_to_keyframe).items() if int(k) == int(keyframe_idx)]
            if train_set:
                frames = [int(x) for x in frames if int(x) in train_set]
        if not frames:
            raise ValueError(
                "iforward_sequence10_v1 keyframe has no train frames: "
                f"scene={getattr(sidx, 'scene_id', -1)} segment={getattr(sidx, 'segment_id', -1)} keyframe={int(keyframe_idx)}"
            )
        return sorted(frames)

    def _refs_for_frame(self, num_cams: int, frame_idx: int) -> Tuple[ImageRef, ...]:
        return tuple((int(frame_idx), int(cam_idx)) for cam_idx in range(int(num_cams)))

    def _build_eligibility_index(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        strides = [int(x) for x in list(_cfg_get(self.sequence_cfg, "strides", [1, 2]) or [])]
        block_source = str(_cfg_get(self.sequence_cfg, "block_source", "keyframes"))
        if block_source not in {"keyframes", "train_frames"}:
            raise ValueError("iforward_sequence10_v1 sequence.block_source must be keyframes or train_frames")
        seq_id = 0
        for scene_id, segment_id in self._scene_segment_pairs():
            sidx = self.dataset.get_segment_index(int(scene_id), int(segment_id))
            frame_to_keyframe = dict(getattr(sidx, "frame_to_keyframe", {}) or {})
            if block_source == "train_frames":
                train_set = set(int(x) for x in getattr(sidx, "train_frame_set", set()))
                frames = [int(x) for x in list(getattr(sidx, "frame_indices", []) or [])]
                if train_set:
                    frames = [int(x) for x in frames if int(x) in train_set]
                blocks = sorted({int(x) for x in frames})
            else:
                keyframes = [int(x) for x in list(getattr(sidx, "keyframe_indices", []) or [])]
                if not keyframes:
                    keyframes = sorted({int(k) for k in frame_to_keyframe.values()})
                blocks = sorted({int(x) for x in keyframes})
            for stride in strides:
                limit = int(len(blocks) - (10 - 1) * int(stride))
                if limit <= 0:
                    continue
                for start in range(limit):
                    positions = [int(start + i * int(stride)) for i in range(10)]
                    selected_blocks = tuple(int(blocks[pos]) for pos in positions)
                    if len(set(selected_blocks)) != 10:
                        continue
                    if block_source == "train_frames":
                        selected_frames = selected_blocks
                        selected_kfs = tuple(int(frame_to_keyframe.get(int(frame), int(frame))) for frame in selected_frames)
                    else:
                        selected_kfs = selected_blocks
                        selected_frames = None
                    out.append(
                        {
                            "scene_id": int(scene_id),
                            "segment_id": int(segment_id),
                            "sequence_id": int(seq_id),
                            "stride": int(stride),
                            "start_block_pos": int(start),
                            "block_ids": tuple(int(x) for x in selected_blocks),
                            "keyframe_indices": selected_kfs,
                            "source_frame_indices": None
                            if selected_frames is None
                            else tuple(int(x) for x in selected_frames),
                            "block_source": str(block_source),
                        }
                    )
                    seq_id += 1
        return out

    @staticmethod
    def _group_eligibility(items: Sequence[Dict[str, Any]]) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
        grouped: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        for item in items:
            scene_id = int(item["scene_id"])
            segment_id = int(item["segment_id"])
            grouped.setdefault(scene_id, {}).setdefault(segment_id, []).append(dict(item))
        return grouped

    def _refill_scene_queue(self) -> None:
        scenes = list(self._eligible_scenes)
        if str(_cfg_get(self.traversal_cfg, "scene_order", "shuffle_per_epoch")) == "shuffle_per_epoch":
            self.rng.shuffle(scenes)
        if (
            len(scenes) > 1
            and self._last_scene_id is not None
            and int(scenes[0]) == int(self._last_scene_id)
            and bool(_cfg_get(self.traversal_cfg, "forbid_consecutive_same_scene", True))
        ):
            scenes.append(scenes.pop(0))
        self._scene_queue = scenes

    def _next_scene_id(self) -> int:
        if not self._scene_queue:
            self._refill_scene_queue()
        if not self._scene_queue:
            raise RuntimeError("iforward_sequence10_v1 has no eligible scenes")
        scene_id = int(self._scene_queue.pop(0))
        if (
            bool(_cfg_get(self.traversal_cfg, "forbid_consecutive_same_scene", True))
            and self._last_scene_id is not None
            and int(scene_id) == int(self._last_scene_id)
            and self._scene_queue
        ):
            self._scene_queue.append(scene_id)
            scene_id = int(self._scene_queue.pop(0))
        self._last_scene_id = int(scene_id)
        return int(scene_id)

    def _refill_segment_queue(self, scene_id: int) -> None:
        segments = sorted(int(x) for x in self._eligible_by_scene_segment[int(scene_id)].keys())
        if str(_cfg_get(self.traversal_cfg, "segment_order", "shuffle_per_epoch")) == "shuffle_per_epoch":
            self.rng.shuffle(segments)
        self._segment_queues[int(scene_id)] = segments

    def _next_segment_id(self, scene_id: int) -> int:
        queue = self._segment_queues.get(int(scene_id))
        if not queue:
            self._refill_segment_queue(int(scene_id))
            queue = self._segment_queues.get(int(scene_id), [])
        if not queue:
            raise RuntimeError(f"iforward_sequence10_v1 has no eligible segments for scene {int(scene_id)}")
        return int(queue.pop(0))

    def _choose_source_frames(self, sidx: Any, keyframes: Sequence[int]) -> Tuple[int, ...]:
        policy = str(_cfg_get(self.sequence_cfg, "source_frame_policy", "random_within_keyframe_once_per_episode"))
        frames: List[int] = []
        for keyframe_idx in keyframes:
            candidates = self._train_frames_for_keyframe(sidx, int(keyframe_idx))
            if policy == "first":
                frame = int(candidates[0])
            elif policy == "last":
                frame = int(candidates[-1])
            elif policy == "random_within_keyframe_once_per_episode":
                frame = int(self.rng.choice(candidates))
            else:
                raise ValueError(f"unknown iforward_sequence10_v1 source_frame_policy={policy!r}")
            frames.append(int(frame))
        return tuple(frames)

    def _sample_sequence_spec(self) -> Sequence10Spec:
        scene_id = self._next_scene_id()
        segment_id = self._next_segment_id(scene_id)
        item = dict(self.rng.choice(self._eligible_by_scene_segment[int(scene_id)][int(segment_id)]))
        sidx = self.dataset.get_segment_index(int(item["scene_id"]), int(item["segment_id"]))
        if item.get("source_frame_indices") is not None:
            frames = tuple(int(x) for x in item["source_frame_indices"])
        else:
            frames = self._choose_source_frames(sidx, item["keyframe_indices"])
        num_cams = int(getattr(sidx, "num_cams", 3))
        refs_by_pos = tuple(self._refs_for_frame(num_cams, frame) for frame in frames)
        return Sequence10Spec(
            scene_id=int(item["scene_id"]),
            segment_id=int(item["segment_id"]),
            sequence_id=int(item["sequence_id"]),
            stride=int(item["stride"]),
            start_block_pos=int(item["start_block_pos"]),
            block_ids=tuple(int(x) for x in item["block_ids"]),
            keyframe_indices=tuple(int(x) for x in item["keyframe_indices"]),
            source_frame_indices=frames,
            source_refs_by_pos=refs_by_pos,
        )

    def _bootstrap_repeats(self) -> int:
        choices = list(_cfg_get(self.bootstrap_cfg, "repeat_choices", []) or [])
        if not choices:
            choices = [
                {"repeats": 4, "prob": 0.60},
                {"repeats": 6, "prob": 0.30},
                {"repeats": 8, "prob": 0.10},
            ]
        total = sum(float(x.get("prob", 1.0)) for x in choices)
        if total <= 0.0:
            raise ValueError("iforward_sequence10_v1 bootstrap repeat probabilities must sum to > 0")
        r = self.rng.random() * float(total)
        acc = 0.0
        for item in choices:
            acc += float(item.get("prob", 1.0))
            if r <= acc:
                return int(item.get("repeats", item.get("repeats_per_block", 4)))
        return int(choices[-1].get("repeats", choices[-1].get("repeats_per_block", 4)))

    def _make_steps(
        self,
        *,
        visits: Sequence[Sequence10VisitSpec],
        scheduler_phase: str,
        window_hash: int,
        episode_step_offset: int = 0,
    ) -> List[Sequence10StepPlan]:
        steps: List[Sequence10StepPlan] = []
        inner_k = sum(int(v.repeats_per_block) for v in visits)
        step_idx = 0
        for rollout_rank, visit in enumerate(visits):
            repeats = int(visit.repeats_per_block)
            for repeat_idx in range(repeats):
                is_exit = bool(int(repeat_idx) == repeats - 1)
                evidence_refs = [tuple(x) for x in visit.source_refs]
                steps.append(
                    Sequence10StepPlan(
                        step_idx=int(step_idx),
                        block_id=int(visit.sequence_pos),
                        episode_block_idx=int(visit.sequence_pos),
                        rollout_block_rank=int(rollout_rank),
                        repeat_idx=int(repeat_idx),
                        repeats_per_block=int(repeats),
                        is_block_enter=bool(int(repeat_idx) == 0),
                        is_block_exit=bool(is_exit),
                        source_keyframe_idx=int(visit.keyframe_idx),
                        source_frame_idx=int(visit.source_frame_idx),
                        evidence_refs=evidence_refs,
                        evidence_frame_indices=[int(visit.source_frame_idx) for _ in evidence_refs],
                        evidence_cam_indices=[int(cam_idx) for _, cam_idx in evidence_refs],
                        commit_observation_memory=bool(visit.observation_commit and int(repeat_idx) == 0),
                        update_optimizer_memory=bool(visit.update_optimizer_memory and is_exit),
                        detach_before_step=False,
                        detach_after_step=False,
                        allow_step_render_loss=False,
                        step_loss_refs=[],
                        rollout_pos_code=float(step_idx) / float(max(int(inner_k) - 1, 1)),
                        frame_pos_code=float(rollout_rank) / float(max(len(visits) - 1, 1)),
                        repeat_pos_code=float(repeat_idx) / float(max(int(repeats) - 1, 1)),
                        is_frame_exit=bool(is_exit),
                        episode_visit_idx=int(visit.sequence_pos),
                        rollout_visit_idx=int(rollout_rank),
                        optimizer_step_idx_in_episode=int(episode_step_offset) + int(step_idx),
                        record_update_norm=True,
                        commit_support_on_exit=bool(is_exit),
                        commit_residual_on_exit=bool(is_exit),
                        window_start=int(min(v.sequence_pos for v in visits)),
                        window_end=int(max(v.sequence_pos for v in visits) + 1),
                        window_hash=int(window_hash),
                        window_revisit_count=0,
                        is_repeated_window=False,
                        sequence_pos=int(visit.sequence_pos),
                        visit_kind=str(visit.visit_kind),
                        frame_gap=int(visit.frame_gap),
                        temporal_read=bool(visit.temporal_read),
                        temporal_commit=bool(visit.temporal_commit and is_exit),
                        physical_time_advance=bool(visit.physical_time_advance),
                        scheduler_phase=str(scheduler_phase),
                    )
                )
                step_idx += 1
        return steps

    def _final_supervision(
        self,
        *,
        current_refs: Sequence[ImageRef],
        history_refs: Sequence[ImageRef],
        current_frames: Sequence[int],
        history_frames: Sequence[int],
    ) -> Tuple[IForwardFinalSupervisionPlan, List[ImageRef], List[str]]:
        current_role = str(
            _cfg_get(
                dict(_cfg_get(self.supervision_cfg, "current", {}) or {}),
                "role_name",
                IFORWARD_SEQUENCE10_CURRENT_ROLE,
            )
        )
        history_role = str(
            _cfg_get(
                dict(_cfg_get(self.supervision_cfg, "history_replay", {}) or {}),
                "role_name",
                IFORWARD_SEQUENCE10_HISTORY_ROLE,
            )
        )
        refs = [tuple(x) for x in list(current_refs)] + [tuple(x) for x in list(history_refs)]
        roles = [current_role for _ in current_refs] + [history_role for _ in history_refs]
        final = IForwardFinalSupervisionPlan(
            refs=[tuple(x) for x in refs],
            roles=[str(x) for x in roles],
            current_input_frames=[int(x) for x in current_frames],
            nearby_frames=[],
            skipped_nearby=True,
            nearby_skip_reason="sequence10_no_nearby",
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
        return final, refs, roles

    def _rollout_from_visits(
        self,
        *,
        spec: Sequence10Spec,
        visits: Sequence[Sequence10VisitSpec],
        rollout_idx: int,
        rollouts_per_episode: int,
        phase: str,
        rollout_phase: str,
        history_positions: Sequence[int],
        repair_positions: Sequence[int] = (),
        repair_enabled: bool = False,
        repair_permutation_hash: int = -1,
        episode_step_offset: int = 0,
    ) -> Sequence10RolloutPlan:
        sidx = self.dataset.get_segment_index(int(spec.scene_id), int(spec.segment_id))
        num_cams = int(getattr(sidx, "num_cams", 3))
        positions = [int(v.sequence_pos) for v in visits]
        current_positions = positions if str(phase) != "repair" else list(range(10))
        current_frames = [int(spec.source_frame_indices[p]) for p in current_positions]
        history_frames = [int(spec.source_frame_indices[p]) for p in history_positions]
        current_refs = [ref for p in current_positions for ref in spec.source_refs_by_pos[int(p)]]
        history_refs = [ref for p in history_positions for ref in spec.source_refs_by_pos[int(p)]]
        final_supervision, target_refs, target_roles = self._final_supervision(
            current_refs=current_refs,
            history_refs=history_refs,
            current_frames=current_frames,
            history_frames=history_frames,
        )
        evidence_refs = _dedupe_refs_keep_order([ref for visit in visits for ref in visit.source_refs])
        window_hash = _stable_hash_int((spec.scene_id, spec.segment_id, spec.sequence_id, phase, *positions))
        steps = self._make_steps(
            visits=visits,
            scheduler_phase=phase,
            window_hash=window_hash,
            episode_step_offset=int(episode_step_offset),
        )
        repeats = int(visits[0].repeats_per_block) if visits else 0
        inner_k = int(len(steps))
        reset_before = bool(int(rollout_idx) == 0)
        episode_end = bool(int(rollout_idx) == int(rollouts_per_episode) - 1)
        request_meta = {
            "scheduler_version": IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
            "model_family": IFORWARD_MODEL_FAMILY,
            "loss_timing_policy": "rollout_final_only",
            "scene_id": int(spec.scene_id),
            "segment_id": int(spec.segment_id),
            "episode_id": int(self._episode_id_next),
            "episode_idx_global": int(self._episode_id_next),
            "rollout_id_global": int(self._rollout_id_global + int(rollout_idx)),
            "rollout_idx_in_episode": int(rollout_idx),
            "rollouts_per_episode": int(rollouts_per_episode),
            "inner_K": int(inner_k),
            "shape_name": str(f"sequence10_{rollout_phase}_b{len(visits)}_r{repeats}"),
            "sequence_id": int(spec.sequence_id),
            "sequence_length": 10,
            "sequence_stride": int(spec.stride),
            "sequence_positions": [int(x) for x in positions],
            "sequence_block_ids": [int(spec.block_ids[p]) for p in positions],
            "sequence_keyframe_indices": [int(spec.keyframe_indices[p]) for p in positions],
            "sequence_source_frame_indices": [int(spec.source_frame_indices[p]) for p in positions],
            "history_positions": [int(x) for x in history_positions],
            "repair_positions": [int(x) for x in repair_positions],
            "scheduler_phase": str(phase),
            "rollout_phase": str(rollout_phase),
            "repair_enabled": bool(repair_enabled),
            "repair_permutation_hash": int(repair_permutation_hash),
            "temporal_commit_count": int(sum(1 for step in steps if bool(step.temporal_commit))),
            "temporal_read_count": int(sum(1 for step in steps if bool(step.temporal_read))),
            "history_frame_count": int(len(history_frames)),
            "history_ref_count": int(len(history_refs)),
        }
        leakage_check = {
            "same_scene_segment_required": True,
            "forbid_test_refs_in_train": True,
            "target_role_count_match": bool(len(target_refs) == len(target_roles)),
            "nearby_evidence_overlap": 0,
            "nearby_input_frame_overlap": 0,
            "current_supervision_must_cover_all_inputs": True,
        }
        return Sequence10RolloutPlan(
            scheduler_version=IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
            scene_id=int(spec.scene_id),
            segment_id=int(spec.segment_id),
            episode_id=int(self._episode_id_next),
            rollout_id_global=int(self._rollout_id_global + int(rollout_idx)),
            rollout_idx_in_episode=int(rollout_idx),
            episode_start_keyframe_pos=int(spec.start_block_pos),
            keyframe_window=[int(x) for x in spec.keyframe_indices],
            frame_chain=[int(x) for x in spec.source_frame_indices],
            num_cams=int(num_cams),
            shape_name=str(request_meta["shape_name"]),
            blocks_per_rollout=int(len(visits)),
            repeats_per_block=int(repeats),
            requested_blocks_per_rollout=int(len(visits)),
            actual_blocks_per_rollout=int(len(visits)),
            requested_inner_K=int(inner_k),
            actual_inner_K=int(inner_k),
            short_rollout=False,
            short_rollout_reason="",
            episode_block_indices=[int(x) for x in positions],
            input_keyframe_indices=[int(spec.keyframe_indices[p]) for p in positions],
            input_frame_indices=[int(spec.source_frame_indices[p]) for p in positions],
            delivery_frame_indices=[int(spec.source_frame_indices[p]) for p in positions],
            delivery_order_policy="sequence10_order",
            inner_K=int(inner_k),
            steps=steps,
            final_supervision=final_supervision,
            reset_scene_state_before_rollout=bool(reset_before),
            carry_scene_state_after_rollout=bool(not episode_end),
            episode_end_after_rollout=bool(episode_end),
            detach_graph_after_rollout=True,
            evidence_refs_flat=[tuple(x) for x in evidence_refs],
            target_refs_flat=[tuple(x) for x in target_refs],
            target_roles_flat=[str(x) for x in target_roles],
            request_meta=request_meta,
            leakage_check=leakage_check,
            model_family=IFORWARD_MODEL_FAMILY,
            rollouts_per_episode=int(rollouts_per_episode),
            episode_num_blocks=10,
            window_policy="sequence10_fixed",
            window_start=int(min(positions) if positions else -1),
            window_end=int(max(positions) + 1 if positions else -1),
            window_block_ids=[int(x) for x in positions],
            window_keyframe_indices=[int(spec.keyframe_indices[p]) for p in positions],
            window_frame_indices=[int(spec.source_frame_indices[p]) for p in positions],
            window_hash=int(window_hash),
            window_revisit_count=0,
            unique_windows_seen=0,
            is_repeated_window=False,
            is_wraparound_rollout=False,
            sequence_id=int(spec.sequence_id),
            sequence_length=10,
            sequence_stride=int(spec.stride),
            sequence_start_block_pos=int(spec.start_block_pos),
            sequence_block_ids=[int(x) for x in spec.block_ids],
            sequence_keyframe_indices=[int(x) for x in spec.keyframe_indices],
            sequence_source_frame_indices=[int(x) for x in spec.source_frame_indices],
            sequence_positions=[int(x) for x in positions],
            history_positions=[int(x) for x in history_positions],
            repair_positions=[int(x) for x in repair_positions],
            scheduler_phase=str(phase),
            rollout_phase=str(rollout_phase),
            repair_enabled=bool(repair_enabled),
            repair_permutation_hash=int(repair_permutation_hash),
            temporal_read_count=int(request_meta["temporal_read_count"]),
            temporal_commit_count=int(request_meta["temporal_commit_count"]),
            observation_commit_count=int(sum(1 for step in steps if bool(step.commit_observation_memory))),
            optimizer_memory_update_count=int(sum(1 for step in steps if bool(step.update_optimizer_memory))),
        )

    def _make_visit(
        self,
        *,
        spec: Sequence10Spec,
        sequence_pos: int,
        visit_kind: str,
        frame_gap: int,
        repeats: int,
        temporal_read: bool,
        temporal_commit: bool,
        observation_commit: bool,
        update_optimizer_memory: bool,
        physical_time_advance: bool,
    ) -> Sequence10VisitSpec:
        pos = int(sequence_pos)
        return Sequence10VisitSpec(
            sequence_pos=pos,
            block_id=int(spec.block_ids[pos]),
            keyframe_idx=int(spec.keyframe_indices[pos]),
            source_frame_idx=int(spec.source_frame_indices[pos]),
            source_refs=tuple(tuple(x) for x in spec.source_refs_by_pos[pos]),
            visit_kind=str(visit_kind),
            frame_gap=int(frame_gap),
            repeats_per_block=int(repeats),
            temporal_read=bool(temporal_read),
            temporal_commit=bool(temporal_commit),
            observation_commit=bool(observation_commit),
            update_optimizer_memory=bool(update_optimizer_memory),
            physical_time_advance=bool(physical_time_advance),
        )

    def _build_bootstrap_rollout(self) -> Sequence10RolloutPlan:
        spec = self._sample_sequence_spec()
        sequence_pos = int(self.rng.randrange(10))
        repeats = self._bootstrap_repeats()
        visit = self._make_visit(
            spec=spec,
            sequence_pos=sequence_pos,
            visit_kind="bootstrap",
            frame_gap=0,
            repeats=int(repeats),
            temporal_read=False,
            temporal_commit=False,
            observation_commit=False,
            update_optimizer_memory=False,
            physical_time_advance=False,
        )
        return self._rollout_from_visits(
            spec=spec,
            visits=[visit],
            rollout_idx=0,
            rollouts_per_episode=1,
            phase="bootstrap",
            rollout_phase="bootstrap",
            history_positions=[],
            repair_enabled=False,
        )

    def _repair_permutation(self) -> Tuple[int, ...]:
        perm = list(range(10))
        for _ in range(32):
            self.rng.shuffle(perm)
            if any(int(a) != int(b) for a, b in zip(perm, range(10))):
                return tuple(int(x) for x in perm)
        return tuple(list(range(1, 10)) + [0])

    def _build_sequence_episode(self) -> Sequence10EpisodePlan:
        spec = self._sample_sequence_spec()
        repair_start = int(_cfg_get(self.repair_cfg, "start_step", 15000))
        repair_prob = float(_cfg_get(self.repair_cfg, "prob", 0.5))
        repair_enabled = bool(int(self.global_step) >= repair_start and self.rng.random() < repair_prob)
        repair_perm: Tuple[int, ...] = ()
        repair_hash = -1
        if repair_enabled:
            repair_perm = self._repair_permutation()
            repair_hash = _stable_hash_int((spec.scene_id, spec.segment_id, spec.sequence_id, *repair_perm, _rng_token(self.rng)))
        rollouts: List[Sequence10RolloutPlan] = []
        total_rollouts = 5 + (1 if repair_enabled else 0)
        repeats = int(_cfg_get(self.causal_cfg, "repeats_per_block", 4))
        episode_step_offset = 0
        for rollout_idx in range(5):
            positions = [int(rollout_idx * 2), int(rollout_idx * 2 + 1)]
            visits = [
                self._make_visit(
                    spec=spec,
                    sequence_pos=pos,
                    visit_kind="causal_first",
                    frame_gap=0 if pos == 0 else int(spec.stride),
                    repeats=int(repeats),
                    temporal_read=True,
                    temporal_commit=True,
                    observation_commit=True,
                    update_optimizer_memory=True,
                    physical_time_advance=True,
                )
                for pos in positions
            ]
            history_cfg = dict(_cfg_get(self.supervision_cfg, "history_replay", {}) or {})
            history_start = int(_cfg_get(history_cfg, "start_step", 5000))
            history_max_frames = int(_cfg_get(history_cfg, "max_frames_per_rollout", 10))
            if int(self.global_step) >= int(history_start) and int(history_max_frames) > 0:
                history_positions = list(range(int(rollout_idx * 2)))[-int(history_max_frames) :]
            else:
                history_positions = []
            rollouts.append(
                self._rollout_from_visits(
                    spec=spec,
                    visits=visits,
                    rollout_idx=rollout_idx,
                    rollouts_per_episode=total_rollouts,
                    phase="causal",
                    rollout_phase=f"causal_{rollout_idx}",
                    history_positions=history_positions,
                    repair_positions=repair_perm,
                    repair_enabled=repair_enabled,
                    repair_permutation_hash=repair_hash,
                    episode_step_offset=int(episode_step_offset),
                )
            )
            episode_step_offset += int(len(rollouts[-1].steps))
        if repair_enabled:
            repair_repeats = int(_cfg_get(self.repair_cfg, "repeats_per_block", 1))
            repair_visits = [
                self._make_visit(
                    spec=spec,
                    sequence_pos=pos,
                    visit_kind="repair",
                    frame_gap=0,
                    repeats=int(repair_repeats),
                    temporal_read=True,
                    temporal_commit=False,
                    observation_commit=False,
                    update_optimizer_memory=False,
                    physical_time_advance=False,
                )
                for pos in repair_perm
            ]
            rollouts.append(
                self._rollout_from_visits(
                    spec=spec,
                    visits=repair_visits,
                    rollout_idx=5,
                    rollouts_per_episode=total_rollouts,
                    phase="repair",
                    rollout_phase="repair",
                    history_positions=[],
                    repair_positions=repair_perm,
                    repair_enabled=True,
                    repair_permutation_hash=repair_hash,
                    episode_step_offset=int(episode_step_offset),
                )
            )
        return Sequence10EpisodePlan(spec=spec, rollouts=tuple(rollouts), repair_enabled=repair_enabled)

    def _batch_from_plan(self, plan: Sequence10RolloutPlan) -> Dict[str, Any]:
        return self.dataset._assemble_segment_batch_from_iforward_request(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            plan=plan,
            include_test=bool(self.include_test),
        )

    def _emit(self, event: Dict[str, Any]) -> None:
        self._pending_events.append(dict(event))

    def pop_events(self) -> List[Dict[str, Any]]:
        out = list(self._pending_events)
        self._pending_events.clear()
        return out

    def get_current_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    def _update_last_info(self, plan: Sequence10RolloutPlan) -> None:
        first_step = plan.steps[0] if plan.steps else None
        first_ref = plan.evidence_refs_flat[0] if plan.evidence_refs_flat else (-1, -1)
        self._last_info = {
            "scheduler_version": IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
            "model_family": IFORWARD_MODEL_FAMILY,
            "global_step": int(self.global_step),
            "scene_id": int(plan.scene_id),
            "segment_id": int(plan.segment_id),
            "episode_id": int(plan.episode_id),
            "episode_idx_global": int(plan.episode_id),
            "epoch_idx": int(self.epoch_idx),
            "rollout_id_global": int(plan.rollout_id_global),
            "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
            "block_idx_global": int(plan.rollout_id_global),
            "block_idx_in_episode": int(plan.sequence_positions[0]) if plan.sequence_positions else -1,
            "block_idx_in_segment": int(plan.sequence_keyframe_indices[plan.sequence_positions[0]]) if plan.sequence_positions else -1,
            "source_frame_idx": int(first_step.source_frame_idx) if first_step is not None else -1,
            "source_keyframe_idx": int(first_step.source_keyframe_idx) if first_step is not None else -1,
            "source_image_ref": (int(first_ref[0]), int(first_ref[1])),
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
            "scheduler_phase": str(plan.scheduler_phase),
            "rollout_phase": str(plan.rollout_phase),
            "sequence_id": int(plan.sequence_id),
            "sequence_stride": int(plan.sequence_stride),
            "sequence_positions": [int(x) for x in plan.sequence_positions],
            "sequence_block_ids": [int(x) for x in plan.sequence_block_ids],
            "sequence_keyframe_indices": [int(x) for x in plan.sequence_keyframe_indices],
            "sequence_source_frame_indices": [int(x) for x in plan.sequence_source_frame_indices],
            "history_positions": [int(x) for x in plan.history_positions],
            "repair_positions": [int(x) for x in plan.repair_positions],
            "repair_enabled": bool(plan.repair_enabled),
            "repair_permutation_hash": int(plan.repair_permutation_hash),
            "temporal_commit_count": int(plan.temporal_commit_count),
            "temporal_read_count": int(plan.temporal_read_count),
            "observation_commit_count": int(plan.observation_commit_count),
            "optimizer_memory_update_count": int(plan.optimizer_memory_update_count),
            "history_frame_count": int(len(plan.history_positions)),
            "history_ref_count": int(plan.final_supervision.history_ref_count),
            "repair_flag": bool(plan.scheduler_phase == "repair"),
            "window_start": int(plan.window_start),
            "window_end": int(plan.window_end),
            "window_block_ids": [int(x) for x in plan.window_block_ids],
            "window_keyframe_indices": [int(x) for x in plan.window_keyframe_indices],
            "window_frame_indices": [int(x) for x in plan.window_frame_indices],
            "block_order": "sequence10",
        }

    def next_batch(self) -> Dict[str, Any]:
        bootstrap_steps = int(_cfg_get(self.bootstrap_cfg, "end_step", 5000))
        if int(self.global_step) < bootstrap_steps:
            plan = self._build_bootstrap_rollout()
            episode_end = True
        else:
            if self._episode_plan is None or int(self._episode_plan_cursor) >= len(self._episode_plan.rollouts):
                self._episode_plan = self._build_sequence_episode()
                self._episode_plan_cursor = 0
            plan = self._episode_plan.rollouts[int(self._episode_plan_cursor)]
            episode_end = bool(int(self._episode_plan_cursor) == len(self._episode_plan.rollouts) - 1)
        batch = self._batch_from_plan(plan)
        self._update_last_info(plan)
        batch["_scheduler_v4_aligned_info"] = dict(self._last_info)
        self._emit(
            {
                "type": "iforward_sequence10_scheduler",
                "scheduler_version": IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(plan.scene_id),
                "segment_id": int(plan.segment_id),
                "episode_id": int(plan.episode_id),
                "rollout_id_global": int(plan.rollout_id_global),
                "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
                "scheduler_phase": str(plan.scheduler_phase),
                "rollout_phase": str(plan.rollout_phase),
                "sequence_positions": [int(x) for x in plan.sequence_positions],
                "sequence_stride": int(plan.sequence_stride),
                "sequence_keyframe_indices": [int(x) for x in plan.sequence_keyframe_indices],
                "sequence_source_frame_indices": [int(x) for x in plan.sequence_source_frame_indices],
                "history_positions": [int(x) for x in plan.history_positions],
                "repair_positions": [int(x) for x in plan.repair_positions],
                "repair_enabled": bool(plan.repair_enabled),
                "repair_permutation_hash": int(plan.repair_permutation_hash),
                "inner_K": int(plan.inner_K),
                "temporal_commit_count": int(plan.temporal_commit_count),
            }
        )
        self.global_step += 1
        self._rollout_id_global += 1
        if str(plan.scheduler_phase) == "bootstrap":
            self._episode_id_next += 1
        else:
            self._episode_plan_cursor += 1
            if episode_end:
                self._emit(
                    {
                        "type": "iforward_sequence10_episode",
                        "scheduler_version": IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
                        "global_step": int(self.global_step),
                        "scene_id": int(plan.scene_id),
                        "segment_id": int(plan.segment_id),
                        "episode_id": int(plan.episode_id),
                        "sequence_id": int(plan.sequence_id),
                        "repair_enabled": bool(plan.repair_enabled),
                    }
                )
                self._episode_plan = None
                self._episode_plan_cursor = 0
                self._episode_id_next += 1
        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_class": type(self).__name__,
            "scheduler_version": IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
            "global_step": int(self.global_step),
            "epoch_idx": int(self.epoch_idx),
            "episode_id_next": int(self._episode_id_next),
            "rollout_id_global": int(self._rollout_id_global),
            "episode_plan": copy.deepcopy(self._episode_plan),
            "episode_plan_cursor": int(self._episode_plan_cursor),
            "pending_events": copy.deepcopy(self._pending_events),
            "last_info": copy.deepcopy(self._last_info),
            "scene_queue": copy.deepcopy(self._scene_queue),
            "segment_queues": copy.deepcopy(self._segment_queues),
            "last_scene_id": self._last_scene_id,
            "rng_state": copy.deepcopy(self.rng.getstate()),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if str(state.get("scheduler_version", "")) != IFORWARD_SEQUENCE10_SCHEDULER_VERSION:
            raise ValueError(f"expected scheduler_version={IFORWARD_SEQUENCE10_SCHEDULER_VERSION}")
        self.global_step = int(state.get("global_step", 0))
        self.epoch_idx = int(state.get("epoch_idx", 0))
        self._episode_id_next = int(state.get("episode_id_next", 0))
        self._rollout_id_global = int(state.get("rollout_id_global", 0))
        self._episode_plan = copy.deepcopy(state.get("episode_plan", None))
        self._episode_plan_cursor = int(state.get("episode_plan_cursor", 0))
        self._pending_events = copy.deepcopy(list(state.get("pending_events", []) or []))
        self._last_info = copy.deepcopy(dict(state.get("last_info", {}) or {}))
        self._scene_queue = [int(x) for x in list(state.get("scene_queue", []) or [])]
        raw_segment_queues = dict(state.get("segment_queues", {}) or {})
        self._segment_queues = {
            int(scene_id): [int(x) for x in list(values or [])]
            for scene_id, values in raw_segment_queues.items()
        }
        last_scene = state.get("last_scene_id", None)
        self._last_scene_id = None if last_scene is None else int(last_scene)
        rng_state = state.get("rng_state", None)
        if rng_state is not None:
            self.rng.setstate(rng_state)


__all__ = [
    "IFORWARD_SEQUENCE10_SCHEDULER_VERSION",
    "Sequence10EpisodePlan",
    "Sequence10RolloutPlan",
    "Sequence10Spec",
    "Sequence10StepPlan",
    "Sequence10VisitSpec",
    "TrainSchedulerIForwardSequence10",
]
