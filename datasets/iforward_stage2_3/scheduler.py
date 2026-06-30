from __future__ import annotations

import copy
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datasets.train_scheduler_iforward import IFORWARD_MODEL_FAMILY, ImageRef, _dedupe_refs_keep_order

from .index_builder import build_stage2_3_index_from_dataset
from .index_format import (
    IFORWARD_STAGE2_3_SCHEDULER_VERSION,
    IFORWARD_STAGE3_0_SCHEDULER_VERSION,
    stable_uint64,
)
from .index_loader import Stage23Index, load_stage2_3_index
from .schema import (
    STAGE23_CURRENT_ROLE,
    STAGE23_HISTORY_ROLE,
    EpisodePlanV3,
    RolloutPlanV3,
    Stage23StepPlan,
    make_final_supervision_v3,
)


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


def _cfg_items(node: Any) -> List[Tuple[Any, Any]]:
    if node is None or isinstance(node, (str, bytes, list, tuple)):
        return []
    if isinstance(node, dict) or hasattr(node, "items"):
        try:
            return list(node.items())
        except Exception:
            return []
    return []


def _scheduler_cfg_and_version(cfg: Any) -> Tuple[Any, str]:
    stage3_cfg = _cfg_get(cfg, "scheduler_stage3_0", None)
    if stage3_cfg is not None and bool(_cfg_get(stage3_cfg, "enable", False)):
        return stage3_cfg, IFORWARD_STAGE3_0_SCHEDULER_VERSION
    stage23_cfg = _cfg_get(cfg, "scheduler_v3", {}) or {}
    return stage23_cfg, IFORWARD_STAGE2_3_SCHEDULER_VERSION


def _rng_token(rng: random.Random) -> int:
    return int(rng.getrandbits(63))


def _cams_from_mask(mask: int, fallback_num_cams: int = 3) -> List[int]:
    cams = [int(i) for i in range(32) if int(mask) & (1 << int(i))]
    return cams if cams else list(range(int(fallback_num_cams)))


def _refs_for_frame(frame_idx: int, cams: Sequence[int]) -> List[ImageRef]:
    return [(int(frame_idx), int(cam)) for cam in cams]


def _sample_weighted_map(rng: random.Random, raw: Any, *, default: int) -> int:
    items = []
    total = 0.0
    for key, value in _cfg_items(raw):
        try:
            k = int(key)
            p = float(value)
        except Exception:
            continue
        if k > 0 and p > 0.0:
            items.append((k, p))
            total += p
    if not items or total <= 0.0:
        return int(default)
    draw = rng.random() * total
    acc = 0.0
    for value, prob in items:
        acc += prob
        if draw <= acc:
            return int(value)
    return int(items[-1][0])


def _sample_weighted_pairs(rng: random.Random, raw: Any) -> Tuple[int, int]:
    items = [(pair, prob) for _, pair, prob in _iter_assimilation_repeat_pairs(raw)]
    total = sum(float(p) for _, p in items)
    draw = rng.random() * total
    acc = 0.0
    for pair, prob in items:
        acc += prob
        if draw <= acc:
            return int(pair[0]), int(pair[1])
    return items[-1][0]


def _parse_rollout_option_name(name: Any) -> Tuple[int, int]:
    text = str(name).upper()
    if not text.startswith("B") or "R" not in text:
        raise ValueError(f"invalid rollout option {name!r}; expected BxRy")
    frames_text, repeats_text = text.split("R", 1)
    frames = int(frames_text.replace("B", ""))
    repeats = int(repeats_text)
    if frames <= 0 or repeats <= 0:
        raise ValueError(f"invalid rollout option {name!r}; B and R must be positive")
    return frames, repeats


def _iter_assimilation_repeat_pairs(raw: Any) -> List[Tuple[str, Tuple[int, int], float]]:
    items: List[Tuple[str, Tuple[int, int], float]] = []
    mapped = _cfg_items(raw)
    if mapped:
        for key, value in mapped:
            try:
                text = str(key)
                if "," not in text:
                    continue
                a, b = text.split(",", 1)
                pair = (int(a.strip()), int(b.strip()))
                prob = float(value)
            except Exception:
                continue
            if min(pair) > 0 and prob > 0.0:
                items.append((text, pair, prob))
    else:
        for idx, item in enumerate(list(raw or [])):
            try:
                reps = list(_cfg_get(item, "repeats", []) or [])
                prob = float(_cfg_get(item, "prob", 0.0))
            except Exception:
                continue
            if len(reps) == 2 and prob > 0.0:
                pair = (int(reps[0]), int(reps[1]))
                if min(pair) > 0:
                    items.append((f"repeat_pairs[{idx}]", pair, prob))
    if not items:
        items = [("4,4", (4, 4), 1.0)]
    return items


def _parse_repair_option_name(name: Any) -> Tuple[int, int]:
    try:
        return _parse_rollout_option_name(name)
    except ValueError as exc:
        raise ValueError(f"invalid repair rollout option {name!r}; expected BxRy") from exc


def _iter_assimilation_candidates(assimilation_cfg: Any) -> List[Tuple[str, Tuple[int, ...], float, str]]:
    items: List[Tuple[str, Tuple[int, ...], float, str]] = []
    rollout_options = _cfg_get(assimilation_cfg, "rollout_options", None)
    for name, prob in _cfg_items(rollout_options):
        try:
            p = float(prob)
            frames, repeats = _parse_rollout_option_name(name)
        except Exception:
            continue
        if p > 0.0:
            items.append((str(name), tuple(int(repeats) for _ in range(int(frames))), p, "rollout_options"))

    repeat_pairs = _cfg_get(
        assimilation_cfg,
        "repeat_pairs",
        _cfg_get(assimilation_cfg, "repeat_pair_table", None),
    )
    if repeat_pairs is not None:
        for name, pair, prob in _iter_assimilation_repeat_pairs(repeat_pairs):
            if float(prob) > 0.0:
                items.append((str(name), (int(pair[0]), int(pair[1])), float(prob), "repeat_pairs"))

    if not items:
        items = [("4,4", (4, 4), 1.0, "repeat_pairs")]
    return items


def _repair_prob(schedule: Any, *, step: int, default: float = 0.0) -> float:
    out = float(default)
    for item in list(schedule or []):
        vals = list(item)
        if len(vals) != 2:
            continue
        if int(step) >= int(vals[0]):
            out = float(vals[1])
    return float(max(0.0, min(1.0, out)))


def _sequence_frame_rule_from_item(item: Any, *, fallback_start: int = 0) -> Dict[str, Any]:
    target = int(_cfg_get(item, "target_frames", _cfg_get(item, "frames", 0)) or 0)
    if target <= 0:
        raise ValueError("Stage2_3 scheduler_v3.sequence.frame_count_schedule target_frames must be > 0")
    min_frames = int(_cfg_get(item, "min_frames", target) or target)
    if min_frames <= 0:
        raise ValueError("Stage2_3 scheduler_v3.sequence.frame_count_schedule min_frames must be > 0")
    if min_frames > target:
        raise ValueError(
            "Stage2_3 scheduler_v3.sequence.frame_count_schedule min_frames must be <= target_frames: "
            f"min_frames={min_frames}, target_frames={target}"
        )
    return {
        "start_step": int(_cfg_get(item, "start_step", fallback_start) or 0),
        "target_frames": int(target),
        "min_frames": int(min_frames),
        "allow_short": bool(_cfg_get(item, "allow_short", False)),
        "scheduled": True,
    }


@dataclass
class _ProducedBatch:
    batch: Dict[str, Any]
    state_after: Dict[str, Any]
    info: Dict[str, Any]
    events: List[Dict[str, Any]]
    build_ms: float


class Stage23Scheduler:
    def __init__(
        self,
        *,
        dataset: Any,
        cfg: Optional[Any] = None,
        index: Optional[Stage23Index] = None,
        index_dir: Optional[str] = None,
        traversal_cfg: Optional[Any] = None,
        bootstrap_cfg: Optional[Any] = None,
        sequence_cfg: Optional[Any] = None,
        assimilation_cfg: Optional[Any] = None,
        repair_cfg: Optional[Any] = None,
        loss_cfg: Optional[Any] = None,
        producer_cfg: Optional[Any] = None,
        include_test: bool = False,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        seed: Optional[int] = None,
        fail_fast: bool = True,
    ) -> None:
        self.dataset = dataset
        self.cfg = cfg or {}
        sched_cfg, scheduler_version = _scheduler_cfg_and_version(self.cfg)
        sched_cfg = sched_cfg or {}
        self.scheduler_version = str(scheduler_version)
        self.traversal_cfg = dict(traversal_cfg or _cfg_get(sched_cfg, "traversal", {}) or {})
        self.bootstrap_cfg = dict(bootstrap_cfg or _cfg_get(sched_cfg, "bootstrap", {}) or {})
        self.sequence_cfg = dict(sequence_cfg or _cfg_get(sched_cfg, "sequence", {}) or {})
        self.assimilation_cfg = dict(assimilation_cfg or _cfg_get(sched_cfg, "assimilation", {}) or {})
        self.repair_cfg = dict(repair_cfg or _cfg_get(sched_cfg, "repair", {}) or {})
        self.loss_cfg = dict(loss_cfg or _cfg_get(sched_cfg, "loss", {}) or {})
        self.producer_cfg = dict(producer_cfg or _cfg_get(sched_cfg, "producer", {}) or {})
        self.include_test = bool(include_test)
        self.fail_fast = bool(fail_fast)
        self._validate_config()
        if getattr(self.dataset, "_initialized", True) is False:
            self.dataset.initialize()
        if fixed_scene_id is None:
            fixed_scene_id = _cfg_get(self.traversal_cfg, "fixed_scene_id", None)
        if fixed_segment_id is None:
            fixed_segment_id = _cfg_get(self.traversal_cfg, "fixed_segment_id", None)
        self.fixed_scene_id = None if fixed_scene_id is None else int(fixed_scene_id)
        self.fixed_segment_id = None if fixed_segment_id is None else int(fixed_segment_id)
        if index is not None:
            self.index = index
        elif index_dir:
            expected_fp = _cfg_get(sched_cfg, "index_fingerprint", None)
            if self.fail_fast and not str(expected_fp or "") and not bool(_cfg_get(sched_cfg, "allow_missing_index_fingerprint", False)):
                raise ValueError("Stage2_3 requires scheduler_v3.index_fingerprint when index_dir is configured")
            self.index = load_stage2_3_index(index_dir, expected_fingerprint=expected_fp)
        else:
            self.index = build_stage2_3_index_from_dataset(
                dataset=dataset,
                cfg=self.cfg,
                fixed_scene_id=self.fixed_scene_id,
                fixed_segment_id=self.fixed_segment_id,
            )
        raw_seed = seed if seed is not None else _cfg_get(self.traversal_cfg, "seed", None)
        self.rng = random.Random(int(raw_seed)) if raw_seed is not None else random.Random()
        self.global_step = 0
        self.epoch_idx = 0
        self._episode_id_next = 0
        self._rollout_id_global = 0
        self._episode_plan: Optional[EpisodePlanV3] = None
        self._episode_plan_cursor = 0
        self._pending_events: List[Dict[str, Any]] = []
        self._last_info: Dict[str, Any] = {
            "scheduler_version": self.scheduler_version,
            "global_step": 0,
            "index_fingerprint": self.index.fingerprint,
        }
        self._segment_rows = self._eligible_segment_rows()
        if not self._segment_rows:
            raise ValueError("Stage2_3 found no eligible segments")
        self._segment_cursor = 0
        self._bootstrap_pack: List[Tuple[int, int]] = []
        self._init_producer_runtime()
        self._emit_eligibility_summary()

    def _init_producer_runtime(self, *, force_disabled: bool = False) -> None:
        depth_configured = int(_cfg_get(self.producer_cfg, "queue_depth", 0) or 0)
        depth = int(depth_configured)
        cuda_depth_cap = _cfg_get(self.producer_cfg, "cuda_queue_depth_cap", None)
        dataset_device = getattr(self.dataset, "device", None)
        dataset_device_type = str(getattr(dataset_device, "type", dataset_device))
        if cuda_depth_cap is not None and dataset_device_type == "cuda":
            cap = int(cuda_depth_cap)
            if cap < 0:
                raise ValueError(
                    f"Stage2_3 scheduler_v3.producer.cuda_queue_depth_cap must be >= 0, got {cap}"
                )
            depth = min(int(depth), int(cap))
        enabled_raw = _cfg_get(self.producer_cfg, "enable", None)
        enabled = bool(depth > 0) if enabled_raw is None else bool(enabled_raw)
        if force_disabled or depth <= 0:
            enabled = False
        self._producer_enabled = bool(enabled)
        self._producer_queue_depth_configured = int(max(0, depth_configured))
        self._producer_cuda_queue_depth_cap = None if cuda_depth_cap is None else int(cuda_depth_cap)
        self._producer_queue_depth = int(max(0, depth))
        self._producer_queue: Optional[queue.Queue[_ProducedBatch]] = (
            queue.Queue(maxsize=max(1, int(depth))) if self._producer_enabled else None
        )
        self._producer_stop = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._producer_exception: Optional[BaseException] = None
        self._producer_lock = threading.Lock()
        self._producer_batches_produced = 0
        self._producer_worker_errors = 0

    def _apply_state_dict(self, state: Dict[str, Any]) -> None:
        self.global_step = int(state.get("global_step", 0))
        self.epoch_idx = int(state.get("epoch_idx", 0))
        self._episode_id_next = int(state.get("episode_id_next", 0))
        self._rollout_id_global = int(state.get("rollout_id_global", 0))
        self._episode_plan = state.get("episode_plan", None)
        self._episode_plan_cursor = int(state.get("episode_plan_cursor", 0))
        self._segment_rows = [int(x) for x in list(state.get("segment_rows", self._segment_rows))]
        self._segment_cursor = int(state.get("segment_cursor", 0))
        self._bootstrap_pack = [(int(a), int(b)) for a, b in list(state.get("bootstrap_pack", []))]
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])
        self._last_info = {
            "scheduler_version": self.scheduler_version,
            "global_step": int(self.global_step),
            "index_fingerprint": self.index.fingerprint,
        }

    def _make_producer_clone(self, state: Dict[str, Any]) -> "Stage23Scheduler":
        clone = object.__new__(Stage23Scheduler)
        for name in (
            "dataset",
            "cfg",
            "traversal_cfg",
            "bootstrap_cfg",
            "sequence_cfg",
            "assimilation_cfg",
            "repair_cfg",
            "loss_cfg",
            "producer_cfg",
            "include_test",
            "fail_fast",
            "fixed_scene_id",
            "fixed_segment_id",
            "index",
            "scheduler_version",
        ):
            setattr(clone, name, getattr(self, name))
        clone.rng = random.Random()
        clone.global_step = 0
        clone.epoch_idx = 0
        clone._episode_id_next = 0
        clone._rollout_id_global = 0
        clone._episode_plan = None
        clone._episode_plan_cursor = 0
        clone._pending_events = []
        clone._last_info = dict(self._last_info)
        clone._segment_rows = []
        clone._segment_cursor = 0
        clone._bootstrap_pack = []
        clone._init_producer_runtime(force_disabled=True)
        clone._apply_state_dict(copy.deepcopy(state))
        clone._pending_events.clear()
        return clone

    def _validate_config(self) -> None:
        bootstrap_end = int(_cfg_get(self.bootstrap_cfg, "end_step", 5000))
        if "start_step" in self.assimilation_cfg and self.assimilation_cfg.get("start_step") is not None:
            assimilation_start = int(self.assimilation_cfg.get("start_step"))
            if int(assimilation_start) != int(bootstrap_end):
                raise ValueError(
                    "Stage2_3 scheduler_v3.assimilation.start_step must be omitted or equal "
                    f"to bootstrap.end_step ({bootstrap_end}); got {assimilation_start}"
                )
        frame_schedule = list(_cfg_get(self.sequence_cfg, "frame_count_schedule", []) or [])
        prev_start: Optional[int] = None
        for idx, item in enumerate(frame_schedule):
            rule = _sequence_frame_rule_from_item(item)
            start = int(rule["start_step"])
            if start < 0:
                raise ValueError(
                    "Stage2_3 scheduler_v3.sequence.frame_count_schedule start_step must be >= 0: "
                    f"index={idx}, start_step={start}"
                )
            if prev_start is not None and start <= prev_start:
                raise ValueError(
                    "Stage2_3 scheduler_v3.sequence.frame_count_schedule start_step values must be strictly increasing: "
                    f"index={idx}, start_step={start}, previous={prev_start}"
                )
            prev_start = int(start)
        repair_schedule = list(_cfg_get(self.repair_cfg, "probability_schedule", []) or [])
        if repair_schedule and "prob" in self.repair_cfg and self.repair_cfg.get("prob") is not None:
            raise ValueError(
                "Stage2_3 scheduler_v3.repair.prob and repair.probability_schedule are mutually exclusive"
            )
        if repair_schedule:
            repair_start = int(_cfg_get(self.repair_cfg, "start_step", 15000))
            first = list(repair_schedule[0])
            if len(first) != 2:
                raise ValueError("Stage2_3 repair.probability_schedule entries must be [step, prob]")
            if int(first[0]) != int(repair_start):
                raise ValueError(
                    "Stage2_3 repair.probability_schedule first step must equal repair.start_step "
                    f"({repair_start}); got {int(first[0])}"
                )
            prev = int(first[0])
            for raw in repair_schedule[1:]:
                item = list(raw)
                if len(item) != 2:
                    raise ValueError("Stage2_3 repair.probability_schedule entries must be [step, prob]")
                step = int(item[0])
                if step <= prev:
                    raise ValueError("Stage2_3 repair.probability_schedule steps must be strictly increasing")
                prev = step
        assimilation_max_k = int(_cfg_get(self.assimilation_cfg, "max_inner_k", 12))
        for name, repeats, _prob, source_path in _iter_assimilation_candidates(self.assimilation_cfg):
            candidate_k = int(sum(int(x) for x in repeats))
            if candidate_k > assimilation_max_k:
                raise ValueError(
                    "Stage2_3 scheduler_v3.assimilation candidate exceeds max_inner_k: "
                    f"path=scheduler_v3.assimilation.{source_path}, candidate={name}, "
                    f"candidate_K={candidate_k}, max_inner_k={assimilation_max_k}"
                )

        repair_max_k = int(_cfg_get(self.repair_cfg, "max_inner_k", 12))
        rollout_options = _cfg_items(_cfg_get(self.repair_cfg, "rollout_options", None))
        if rollout_options:
            for name, prob in rollout_options:
                try:
                    p = float(prob)
                except Exception:
                    continue
                if p <= 0.0:
                    continue
                frames, repeats = _parse_repair_option_name(name)
                candidate_k = int(frames) * int(repeats)
                if candidate_k > repair_max_k:
                    raise ValueError(
                        "Stage2_3 scheduler_v3.repair.rollout_options candidate exceeds max_inner_k: "
                        f"path=scheduler_v3.repair.rollout_options, candidate={name}, "
                        f"candidate_K={candidate_k}, max_inner_k={repair_max_k}"
                    )
        elif int(6 * 1) > repair_max_k:
            raise ValueError(
                "Stage2_3 scheduler_v3.repair.rollout_options default candidate exceeds max_inner_k: "
                f"path=scheduler_v3.repair.rollout_options, candidate=B6R1, candidate_K=6, max_inner_k={repair_max_k}"
            )
        for idx, item in enumerate(list(_cfg_get(self.repair_cfg, "patterns", []) or [])):
            try:
                p = float(_cfg_get(item, "prob", 0.0))
            except Exception:
                continue
            if p <= 0.0:
                continue
            frames = int(_cfg_get(item, "frames", 6))
            repeats = int(_cfg_get(item, "repeats", 1))
            candidate_k = int(frames) * int(repeats)
            if candidate_k > repair_max_k:
                raise ValueError(
                    "Stage2_3 scheduler_v3.repair.patterns candidate exceeds max_inner_k: "
                    f"path=scheduler_v3.repair.patterns[{idx}], candidate={_cfg_get(item, 'name', idx)}, "
                    f"candidate_K={candidate_k}, max_inner_k={repair_max_k}"
                )

    def _eligible_segment_rows(self) -> List[int]:
        rows = []
        for idx, seg in enumerate(self.index.segments):
            if self.fixed_scene_id is not None and int(seg["scene_id"]) != int(self.fixed_scene_id):
                continue
            if self.fixed_segment_id is not None and int(seg["segment_id"]) != int(self.fixed_segment_id):
                continue
            if int(seg["frame_count"]) > 0:
                rows.append(int(idx))
        self.rng.shuffle(rows)
        return rows

    def _next_segment_row(self) -> int:
        if not self._segment_rows:
            raise ValueError("Stage2_3 has no segment rows")
        if self._segment_cursor >= len(self._segment_rows):
            self._segment_cursor = 0
            self.rng.shuffle(self._segment_rows)
            self.epoch_idx += 1
        row = int(self._segment_rows[self._segment_cursor])
        self._segment_cursor += 1
        return row

    def _segment_ids(self, segment_row: int) -> Tuple[int, int]:
        seg = self.index.segments[int(segment_row)]
        return int(seg["scene_id"]), int(seg["segment_id"])

    def _emit_eligibility_summary(self) -> None:
        indexed_scenes = sorted(int(x) for x in np.unique(self.index.segments["scene_id"]).tolist())
        self._pending_events.append(
            {
                "type": "iforward_stage2_3_eligibility",
                "scheduler_version": self.scheduler_version,
                "eligible_scene_count": int(len(indexed_scenes)),
                "eligible_segment_count": int(len(self._segment_rows)),
                "timestamp_source": str(self.index.timestamp_source),
                "num_cams": int(self.index.num_cams),
                "index_fingerprint": self.index.fingerprint,
            }
        )

    def _sample_bootstrap_repeats(self) -> int:
        raw = _cfg_get(self.bootstrap_cfg, "repeat_distribution", None)
        if raw is not None:
            return _sample_weighted_map(self.rng, raw, default=4)
        choices = _cfg_get(self.bootstrap_cfg, "repeat_choices", None)
        if choices:
            total = 0.0
            clean = []
            for item in list(choices or []):
                repeats = int(_cfg_get(item, "repeats", 0))
                prob = float(_cfg_get(item, "prob", 0.0))
                if repeats > 0 and prob > 0:
                    clean.append((repeats, prob))
                    total += prob
            if clean and total > 0:
                draw = self.rng.random() * total
                acc = 0.0
                for repeats, prob in clean:
                    acc += prob
                    if draw <= acc:
                        return int(repeats)
        return int(_cfg_get(self.bootstrap_cfg, "repeats", 4))

    def _ref_rows_for_positions(self, rows: np.ndarray, positions: Sequence[int]) -> Tuple[List[int], List[ImageRef]]:
        frames: List[int] = []
        refs: List[ImageRef] = []
        for pos in positions:
            row = rows[int(pos)]
            frame_idx = int(row["frame_idx"])
            cams = _cams_from_mask(int(row["available_camera_mask"]), int(self.index.num_cams))
            frames.append(frame_idx)
            refs.extend(_refs_for_frame(frame_idx, cams))
        return frames, _dedupe_refs_keep_order(refs)

    def _make_steps(
        self,
        *,
        rows: np.ndarray,
        positions: Sequence[int],
        repeat_budgets: Sequence[int],
        phase: str,
        visit_kind: str,
        episode_step_offset: int,
        visit_counts: Dict[int, int],
        last_visit_step_by_pos: Dict[int, int],
        last_visit_context: Dict[str, Any],
        optimizer_read: bool,
        optimizer_write: bool,
        skip_last_write: bool,
        is_last_rollout: bool,
    ) -> Tuple[List[Stage23StepPlan], List[ImageRef]]:
        steps: List[Stage23StepPlan] = []
        evidence_flat: List[ImageRef] = []
        for rank, pos_raw in enumerate(positions):
            pos = int(pos_raw)
            row = rows[pos]
            frame_idx = int(row["frame_idx"])
            keyframe_idx = int(row["keyframe_idx"])
            timestamp_us = int(row["timestamp_us"])
            cams = _cams_from_mask(int(row["available_camera_mask"]), int(self.index.num_cams))
            refs = _refs_for_frame(frame_idx, cams)
            evidence_flat.extend(refs)
            repeat_budget = int(repeat_budgets[int(rank)])
            count_before = int(visit_counts.get(pos, 0))
            for repeat in range(repeat_budget):
                step_idx = len(steps)
                is_enter = int(repeat) == 0
                is_exit = int(repeat) == int(repeat_budget) - 1
                global_idx = int(episode_step_offset + step_idx)
                is_last_update = bool(is_last_rollout and rank == len(positions) - 1 and repeat == repeat_budget - 1)
                write_now = bool(optimizer_write and not (bool(skip_last_write) and bool(is_last_update)))
                previous_sequence_pos = (
                    -1
                    if last_visit_context.get("sequence_pos", None) is None
                    else int(last_visit_context.get("sequence_pos"))
                )
                if previous_sequence_pos < 0:
                    frame_gap = 0
                    visit_order_gap = 0
                    physical_frame_gap_abs = 0
                    delta_t_sec = 0.0
                    ego_delta = np.zeros((3,), dtype=np.float32)
                    ego_delta_yaw = 0.0
                else:
                    previous_frame_idx = int(last_visit_context.get("frame_idx", frame_idx))
                    previous_timestamp_us = int(last_visit_context.get("timestamp_us", timestamp_us))
                    frame_gap = int(frame_idx - previous_frame_idx)
                    visit_order_gap = int(pos - previous_sequence_pos)
                    physical_frame_gap_abs = int(abs(frame_gap))
                    delta_t_sec = float(timestamp_us - previous_timestamp_us) / 1.0e6
                    previous_translation = np.asarray(
                        last_visit_context.get("ego_translation", np.asarray(row["ego_translation"], dtype=np.float32)),
                        dtype=np.float32,
                    )
                    ego_delta = np.asarray(row["ego_translation"], dtype=np.float32) - previous_translation
                    ego_delta_yaw = float(row["ego_yaw"] - float(last_visit_context.get("ego_yaw", row["ego_yaw"])))
                last_same = int(last_visit_step_by_pos.get(pos, -1))
                steps.append(
                    Stage23StepPlan(
                        step_idx=int(step_idx),
                        block_id=int(pos),
                        episode_block_idx=int(pos),
                        rollout_block_rank=int(rank),
                        repeat_idx=int(repeat),
                        repeats_per_block=int(repeat_budget),
                        is_block_enter=bool(is_enter),
                        is_block_exit=bool(is_exit),
                        source_keyframe_idx=int(keyframe_idx),
                        source_frame_idx=int(frame_idx),
                        evidence_refs=list(refs),
                        evidence_frame_indices=[int(frame_idx) for _ in refs],
                        evidence_cam_indices=[int(ref[1]) for ref in refs],
                        commit_observation_memory=bool(is_enter and str(phase) != "bootstrap"),
                        update_optimizer_memory=bool(write_now),
                        detach_before_step=False,
                        detach_after_step=False,
                        allow_step_render_loss=False,
                        step_loss_refs=[],
                        rollout_pos_code=float(rank) / float(max(len(positions) - 1, 1)),
                        frame_pos_code=float(pos) / float(max(int(rows.shape[0]) - 1, 1)),
                        repeat_pos_code=float(repeat) / float(max(int(repeat_budget) - 1, 1)),
                        is_frame_exit=bool(is_exit),
                        episode_visit_idx=int(pos),
                        rollout_visit_idx=int(rank),
                        optimizer_step_idx_in_episode=int(global_idx),
                        record_update_norm=bool(write_now),
                        commit_support_on_exit=bool(is_exit),
                        commit_residual_on_exit=bool(is_exit),
                        window_start=0,
                        window_end=int(rows.shape[0]) - 1,
                        window_hash=0,
                        sequence_pos=int(pos),
                        visit_kind=str(visit_kind),
                        frame_gap=int(frame_gap),
                        temporal_read=bool(optimizer_read),
                        temporal_commit=bool(write_now),
                        physical_time_advance=bool(str(phase) == "assimilation"),
                        scheduler_phase=str(phase),
                        timestamp_us=int(timestamp_us),
                        timestamp_sec=float(timestamp_us) / 1.0e6,
                        delta_t_sec=float(delta_t_sec),
                        visit_order_gap=int(visit_order_gap),
                        physical_frame_gap_abs=int(physical_frame_gap_abs),
                        previous_visit_sequence_pos=int(previous_sequence_pos),
                        ego_delta_translation=(float(ego_delta[0]), float(ego_delta[1]), float(ego_delta[2])),
                        ego_delta_yaw=float(ego_delta_yaw),
                        visit_memory_mask=bool(optimizer_read or write_now),
                        repair_no_commit=bool(str(phase) == "repair"),
                        repeat_budget=int(repeat_budget),
                        visit_count_for_frame=int(count_before),
                        is_first_visit_of_frame=bool(is_enter and count_before == 0),
                        is_last_update_of_episode=bool(is_last_update),
                        global_update_idx_in_episode=int(global_idx),
                        optimizer_memory_read=bool(optimizer_read),
                        optimizer_memory_write=bool(write_now),
                        time_since_same_frame_visit=float(0.0 if last_same < 0 else max(0, global_idx - last_same)),
                    )
                )
                last_visit_context.update(
                    {
                        "sequence_pos": int(pos),
                        "frame_idx": int(frame_idx),
                        "timestamp_us": int(timestamp_us),
                        "ego_translation": np.asarray(row["ego_translation"], dtype=np.float32),
                        "ego_yaw": float(row["ego_yaw"]),
                        "global_update_idx": int(global_idx),
                    }
                )
                last_visit_step_by_pos[pos] = int(global_idx)
            visit_counts[pos] = count_before + 1
        return steps, _dedupe_refs_keep_order(evidence_flat)

    def _rollout_from_positions(
        self,
        *,
        rows: np.ndarray,
        scene_id: int,
        segment_id: int,
        sequence_id: int,
        positions: Sequence[int],
        repeat_budgets: Sequence[int],
        rollout_idx: int,
        rollouts_per_episode: int,
        phase: str,
        visit_kind: str,
        history_positions: Sequence[int],
        repair_positions: Sequence[int],
        repair_enabled: bool,
        repair_hash: int,
        episode_step_offset: int,
        visit_counts: Dict[int, int],
        last_visit_step_by_pos: Dict[int, int],
        is_last_rollout: bool,
        last_visit_context: Optional[Dict[str, Any]] = None,
        repair_round_idx: int = -1,
        repair_pattern_name: str = "",
        phase_max_inner_k: Optional[int] = None,
        requested_inner_k: Optional[int] = None,
        requested_blocks_per_rollout: Optional[int] = None,
        sequence_target_frames: Optional[int] = None,
        sequence_min_frames: Optional[int] = None,
        sequence_allow_short: Optional[bool] = None,
    ) -> RolloutPlanV3:
        if last_visit_context is None:
            last_visit_context = {}
        optimizer_read = bool(str(phase) != "bootstrap")
        optimizer_write = bool(str(phase) != "bootstrap")
        skip_last = bool(str(phase) == "repair" and not bool(_cfg_get(self.repair_cfg, "last_update_write", True)))
        steps, evidence_refs = self._make_steps(
            rows=rows,
            positions=positions,
            repeat_budgets=repeat_budgets,
            phase=str(phase),
            visit_kind=str(visit_kind),
            episode_step_offset=int(episode_step_offset),
            visit_counts=visit_counts,
            last_visit_step_by_pos=last_visit_step_by_pos,
            last_visit_context=last_visit_context,
            optimizer_read=optimizer_read,
            optimizer_write=optimizer_write,
            skip_last_write=skip_last,
            is_last_rollout=bool(is_last_rollout),
        )
        current_frames, current_refs = self._ref_rows_for_positions(rows, positions)
        history_frames, history_refs_all = self._ref_rows_for_positions(rows, history_positions)
        current_set = set(current_refs)
        history_refs = [ref for ref in history_refs_all if ref not in current_set]
        target_refs = list(current_refs) + list(history_refs)
        target_roles = [STAGE23_CURRENT_ROLE for _ in current_refs] + [STAGE23_HISTORY_ROLE for _ in history_refs]
        final = make_final_supervision_v3(
            refs=target_refs,
            roles=target_roles,
            current_frames=current_frames,
            current_refs=current_refs,
            history_frames=history_frames,
            history_refs=history_refs,
        )
        frame_indices = [int(row["frame_idx"]) for row in rows]
        keyframes = [int(row["keyframe_idx"]) for row in rows]
        timestamps_us = [int(row["timestamp_us"]) for row in rows]
        window_hash = int(stable_uint64((scene_id, segment_id, sequence_id, *frame_indices)) & 0x7FFFFFFFFFFFFFFF)
        steps = [type(step)(**{**step.__dict__, "window_hash": int(window_hash)}) for step in steps]
        repeat_label = "x".join(str(int(x)) for x in repeat_budgets)
        actual_inner_k = int(len(steps))
        requested_inner_k_value = int(requested_inner_k) if requested_inner_k is not None else int(actual_inner_k)
        requested_blocks_value = (
            int(requested_blocks_per_rollout)
            if requested_blocks_per_rollout is not None
            else int(len(positions))
        )
        default_phase_cfg = self.repair_cfg if str(phase) == "repair" else self.assimilation_cfg
        default_phase_max = (
            int(_cfg_get(default_phase_cfg, "max_inner_k", actual_inner_k))
            if str(phase) in {"assimilation", "repair"}
            else int(actual_inner_k)
        )
        phase_max_inner_k_value = (
            int(phase_max_inner_k)
            if phase_max_inner_k is not None
            else int(default_phase_max)
        )
        request_meta = {
            "scheduler_version": self.scheduler_version,
            "model_family": IFORWARD_MODEL_FAMILY,
            "iforward_stage2_3": {
                "index_fingerprint": self.index.fingerprint,
                "phase": str(phase),
                "timestamp_source": str(self.index.timestamp_source),
                "frame_period_us": int(self.index.frame_period_us),
                "raw_frame_ids": frame_indices,
                "keyframe_ids": keyframes,
                "timestamps_us": timestamps_us,
                "episode_positions": [int(x) for x in range(int(rows.shape[0]))],
                "rollout_positions": [int(x) for x in positions],
                "sequence_positions": [int(x) for x in positions],
                "repeat_budgets": [int(x) for x in repeat_budgets],
                "frame_gaps": [int(s.frame_gap) for s in steps],
                "visit_kinds": [str(s.visit_kind) for s in steps],
                "repair_round_idx": int(repair_round_idx),
                "repair_pattern_name": str(repair_pattern_name),
                "phase_max_inner_k": int(phase_max_inner_k_value),
                "requested_inner_K": int(requested_inner_k_value),
                "actual_inner_K": int(actual_inner_k),
                "sequence_target_frames": int(sequence_target_frames or rows.shape[0]),
                "sequence_min_frames": int(sequence_min_frames or rows.shape[0]),
                "sequence_allow_short": bool(sequence_allow_short),
            },
        }
        return RolloutPlanV3(
            scheduler_version=self.scheduler_version,
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_id=int(self._episode_id_next),
            rollout_id_global=int(self._rollout_id_global + int(rollout_idx)),
            rollout_idx_in_episode=int(rollout_idx),
            episode_start_keyframe_pos=0,
            keyframe_window=keyframes,
            frame_chain=frame_indices,
            num_cams=int(self.index.num_cams),
            shape_name=f"{str(phase)}_b{len(positions)}r{repeat_label}",
            blocks_per_rollout=int(len(positions)),
            repeats_per_block=int(max(repeat_budgets) if repeat_budgets else 0),
            requested_blocks_per_rollout=int(requested_blocks_value),
            actual_blocks_per_rollout=int(len(positions)),
            requested_inner_K=int(requested_inner_k_value),
            actual_inner_K=int(actual_inner_k),
            sequence_target_frames=int(sequence_target_frames or rows.shape[0]),
            sequence_min_frames=int(sequence_min_frames or rows.shape[0]),
            sequence_allow_short=bool(sequence_allow_short),
            short_rollout=False,
            short_rollout_reason="",
            episode_block_indices=[int(x) for x in positions],
            input_keyframe_indices=[int(rows[int(pos)]["keyframe_idx"]) for pos in positions],
            input_frame_indices=current_frames,
            delivery_frame_indices=current_frames,
            delivery_order_policy="iforward_stage2_3_optimizer_sequence",
            inner_K=int(actual_inner_k),
            steps=list(steps),
            final_supervision=final,
            reset_scene_state_before_rollout=bool(int(rollout_idx) == 0),
            carry_scene_state_after_rollout=bool(int(rollout_idx) < int(rollouts_per_episode) - 1),
            episode_end_after_rollout=bool(int(rollout_idx) == int(rollouts_per_episode) - 1),
            detach_graph_after_rollout=True,
            evidence_refs_flat=list(evidence_refs),
            target_refs_flat=list(target_refs),
            target_roles_flat=list(target_roles),
            request_meta=request_meta,
            leakage_check={"same_scene_segment_required": True, "forbid_test_refs_in_train": True},
            model_family=IFORWARD_MODEL_FAMILY,
            rollouts_per_episode=int(rollouts_per_episode),
            episode_num_blocks=int(rows.shape[0]),
            window_policy="optimizer_sequence_v1",
            window_start=0,
            window_end=int(rows.shape[0]) - 1,
            window_block_ids=list(range(int(rows.shape[0]))),
            window_keyframe_indices=keyframes,
            window_frame_indices=frame_indices,
            window_hash=int(window_hash),
            sequence_id=int(sequence_id),
            sequence_length=int(rows.shape[0]),
            sequence_protocol="optimizer_sequence_v1",
            sequence_stride=0,
            sequence_start_local_frame=0,
            sequence_block_ids=list(range(int(rows.shape[0]))),
            sequence_keyframe_indices=keyframes,
            sequence_source_frame_indices=frame_indices,
            sequence_timestamps_us=timestamps_us,
            sequence_positions=[int(x) for x in positions],
            episode_positions=[int(x) for x in range(int(rows.shape[0]))],
            rollout_positions=[int(x) for x in positions],
            history_positions=[int(x) for x in history_positions],
            repair_positions=[int(x) for x in repair_positions],
            repeat_budgets=[int(x) for x in repeat_budgets],
            frame_gaps=[int(s.frame_gap) for s in steps],
            visit_kinds=[str(s.visit_kind) for s in steps],
            scheduler_phase=str(phase),
            rollout_phase=str(phase if phase != "assimilation" else f"assimilation_{int(rollout_idx)}"),
            repair_enabled=bool(repair_enabled),
            repair_round_idx=int(repair_round_idx),
            repair_pattern_name=str(repair_pattern_name),
            repair_permutation_hash=int(repair_hash),
            temporal_read_count=sum(1 for s in steps if bool(s.temporal_read)),
            temporal_commit_count=sum(1 for s in steps if bool(s.temporal_commit)),
            optimizer_memory_read_count=sum(1 for s in steps if bool(s.optimizer_memory_read)),
            optimizer_memory_write_count=sum(1 for s in steps if bool(s.optimizer_memory_write)),
            observation_commit_count=sum(1 for s in steps if bool(s.commit_observation_memory)),
            phase_max_inner_k=int(phase_max_inner_k_value),
        )

    def _build_bootstrap_rollout(self) -> RolloutPlanV3:
        if not self._bootstrap_pack:
            segment_row = self._next_segment_row()
            frames = self.index.frames_for_segment_row(segment_row)
            count = min(int(_cfg_get(self.bootstrap_cfg, "frames_per_asset_pack", 4)), int(frames.shape[0]))
            local = list(range(int(frames.shape[0])))
            self.rng.shuffle(local)
            self._bootstrap_pack = [(int(segment_row), int(x)) for x in local[:count]]
        segment_row, local_frame = self._bootstrap_pack.pop(0)
        scene_id, segment_id = self._segment_ids(segment_row)
        rows = self.index.frames_for_segment_row(segment_row)[[int(local_frame)]]
        repeats = int(self._sample_bootstrap_repeats())
        return self._rollout_from_positions(
            rows=rows,
            scene_id=scene_id,
            segment_id=segment_id,
            sequence_id=int(_rng_token(self.rng)),
            positions=[0],
            repeat_budgets=[repeats],
            rollout_idx=0,
            rollouts_per_episode=1,
            phase="bootstrap",
            visit_kind="bootstrap",
            history_positions=[],
            repair_positions=[],
            repair_enabled=False,
            repair_hash=-1,
            episode_step_offset=0,
            visit_counts={},
            last_visit_step_by_pos={},
            is_last_rollout=True,
            last_visit_context={},
        )

    def _active_sequence_frame_rule(self) -> Dict[str, Any]:
        schedule = list(_cfg_get(self.sequence_cfg, "frame_count_schedule", []) or [])
        if schedule:
            active = _sequence_frame_rule_from_item(schedule[0])
            for item in schedule:
                rule = _sequence_frame_rule_from_item(item)
                if int(self.global_step) >= int(rule["start_step"]):
                    active = rule
                else:
                    break
            return active
        min_frames = int(_cfg_get(self.sequence_cfg, "min_frames", 8))
        max_frames = int(_cfg_get(self.sequence_cfg, "max_frames", 10))
        if min_frames <= 0 or max_frames < min_frames:
            raise ValueError(
                "Stage2_3 scheduler_v3.sequence requires 0 < min_frames <= max_frames: "
                f"min_frames={min_frames}, max_frames={max_frames}"
            )
        return {
            "start_step": 0,
            "target_frames": int(max_frames),
            "min_frames": int(min_frames),
            "allow_short": False,
            "scheduled": False,
        }

    def _select_sequence_count(self, *, available: int, rule: Dict[str, Any]) -> Optional[int]:
        available = int(available)
        min_frames = int(rule["min_frames"])
        target = int(rule["target_frames"])
        if available < min_frames:
            return None
        if bool(rule.get("scheduled", False)):
            if available < target and not bool(rule.get("allow_short", False)):
                return None
            return int(min(target, available))
        return int(self.rng.randint(min_frames, min(target, available)))

    def _sample_sequence_rows(self) -> Tuple[int, np.ndarray, Tuple[int, ...], Dict[str, Any]]:
        frame_rule = self._active_sequence_frame_rule()
        min_keyframes = int(_cfg_get(self.sequence_cfg, "min_unique_keyframes", 3))
        min_span = int(_cfg_get(self.sequence_cfg, "min_frame_span", 8))
        max_span = int(_cfg_get(self.sequence_cfg, "max_frame_span", 30))
        for _seg_try in range(max(8, len(self._segment_rows) * 2)):
            segment_row = self._next_segment_row()
            frames = self.index.frames_for_segment_row(segment_row)
            n = self._select_sequence_count(available=int(frames.shape[0]), rule=frame_rule)
            if n is None:
                continue
            for _ in range(64):
                local = sorted(self.rng.sample(range(int(frames.shape[0])), int(n)))
                selected = frames[local]
                span = int(selected[-1]["frame_idx"]) - int(selected[0]["frame_idx"])
                keyframes = {int(row["keyframe_idx"]) for row in selected}
                if span >= min_span and span <= max_span and len(keyframes) >= min_keyframes:
                    return int(segment_row), selected, tuple(int(x) for x in local), dict(frame_rule)
            for start in range(0, int(frames.shape[0]) - int(n) + 1):
                selected = frames[start : start + int(n)]
                span = int(selected[-1]["frame_idx"]) - int(selected[0]["frame_idx"])
                keyframes = {int(row["keyframe_idx"]) for row in selected}
                if span >= min_span and span <= max_span and len(keyframes) >= min_keyframes:
                    return int(segment_row), selected, tuple(range(start, start + int(n))), dict(frame_rule)
        raise ValueError("Stage2_3 could not sample a valid optimizer sequence")

    def _assimilation_order(self, n: int) -> List[int]:
        order = list(range(int(n)))
        weights = dict(_cfg_items(_cfg_get(self.sequence_cfg, "assimilation_order", {}) or {}))
        local_prob = float(weights.get("local_shuffle", 0.2))
        if self.rng.random() < local_prob and len(order) >= 2:
            idx = 0
            while idx < len(order) - 1:
                if self.rng.random() < 0.35:
                    order[idx], order[idx + 1] = order[idx + 1], order[idx]
                    idx += 2
                else:
                    idx += 1
        return order

    def _repair_enabled(self) -> bool:
        if not bool(_cfg_get(self.repair_cfg, "enable", True)):
            return False
        start = int(_cfg_get(self.repair_cfg, "start_step", 15000))
        if int(self.global_step) < start:
            return False
        prob = _repair_prob(_cfg_get(self.repair_cfg, "probability_schedule", []), step=int(self.global_step), default=float(_cfg_get(self.repair_cfg, "prob", 0.0)))
        return bool(self.rng.random() < float(prob))

    def _repair_rounds(self) -> int:
        raw = _cfg_get(
            self.repair_cfg,
            "round_distribution",
            _cfg_get(
                self.repair_cfg,
                "rounds_per_episode_distribution",
                _cfg_get(self.repair_cfg, "rounds", _cfg_get(self.repair_cfg, "rounds_distribution", {1: 1.0})),
            ),
        )
        return _sample_weighted_map(self.rng, raw, default=1)

    def _repair_pattern(self) -> Tuple[int, int, str]:
        raw = _cfg_get(self.repair_cfg, "rollout_options", None)
        mapped = _cfg_items(raw)
        if mapped:
            valid_items = []
            total = 0.0
            for name, prob in mapped:
                try:
                    p = float(prob)
                except Exception:
                    continue
                if p > 0.0:
                    valid_items.append((name, p))
                    total += p
            if total <= 0.0:
                valid_items = []
        else:
            valid_items = []
        if valid_items:
            draw = self.rng.random() * total
            acc = 0.0
            for name, prob in valid_items:
                acc += float(prob)
                if draw <= acc:
                    frames, repeats = _parse_repair_option_name(name)
                    return frames, repeats, str(name)
        patterns = list(_cfg_get(self.repair_cfg, "patterns", []) or [])
        if patterns:
            total = sum(float(_cfg_get(x, "prob", 0.0)) for x in patterns)
            draw = self.rng.random() * max(total, 1.0e-8)
            acc = 0.0
            for item in patterns:
                acc += float(_cfg_get(item, "prob", 0.0))
                if draw <= acc:
                    return int(_cfg_get(item, "frames", 6)), int(_cfg_get(item, "repeats", 1)), str(_cfg_get(item, "name", "repair"))
        return 6, 1, "B6R1"

    def _assimilation_candidate(self, *, remaining: int) -> Tuple[Tuple[int, ...], str]:
        max_k = int(_cfg_get(self.assimilation_cfg, "max_inner_k", 12))
        candidates = []
        total = 0.0
        for name, repeats, prob, _source_path in _iter_assimilation_candidates(self.assimilation_cfg):
            if len(repeats) > int(remaining):
                continue
            if int(sum(int(x) for x in repeats)) > int(max_k):
                continue
            p = float(prob)
            if p <= 0.0:
                continue
            candidates.append((str(name), tuple(int(x) for x in repeats), p))
            total += p
        if candidates:
            draw = self.rng.random() * total
            acc = 0.0
            for name, repeats, prob in candidates:
                acc += float(prob)
                if draw <= acc:
                    return tuple(int(x) for x in repeats), str(name)
            name, repeats, _prob = candidates[-1]
            return tuple(int(x) for x in repeats), str(name)

        single_raw = _cfg_get(self.assimilation_cfg, "single_repeat_distribution", {4: 0.5, 6: 0.3, 8: 0.2})
        repeat = _sample_weighted_map(self.rng, single_raw, default=min(4, max(1, int(max_k))))
        repeat = min(int(repeat), int(max_k))
        return (int(repeat),), f"B1R{int(repeat)}"

    def _build_episode(self) -> EpisodePlanV3:
        segment_row, rows, _, sequence_frame_rule = self._sample_sequence_rows()
        scene_id, segment_id = self._segment_ids(segment_row)
        sequence_id = int(stable_uint64((scene_id, segment_id, tuple(int(x["frame_idx"]) for x in rows), _rng_token(self.rng))) & 0x7FFFFFFFFFFFFFFF)
        order = self._assimilation_order(int(rows.shape[0]))
        repair_enabled = self._repair_enabled()
        repair_rounds = self._repair_rounds() if repair_enabled else 0
        repair_plans: List[Tuple[List[int], int, str, int, int]] = []
        repair_positions_flat: List[int] = []
        covered: set[int] = set()
        for _round in range(int(repair_rounds)):
            frames, repeats, pattern_name = self._repair_pattern()
            candidates = [p for p in range(int(rows.shape[0])) if p not in covered]
            rest = [p for p in range(int(rows.shape[0])) if p in covered]
            self.rng.shuffle(candidates)
            self.rng.shuffle(rest)
            selected = (candidates + rest)[: min(int(frames), int(rows.shape[0]))]
            if bool(_cfg_get(self.repair_cfg, "random_order", True)):
                self.rng.shuffle(selected)
            covered.update(selected)
            repair_positions_flat.extend(int(x) for x in selected)
            repair_plans.append(([int(x) for x in selected], int(repeats), str(pattern_name), int(_round), int(frames)))
        repair_hash = int(stable_uint64((scene_id, segment_id, sequence_id, *repair_positions_flat, _rng_token(self.rng))) & 0x7FFFFFFFFFFFFFFF) if repair_enabled else -1
        rollouts: List[RolloutPlanV3] = []
        step_offset = 0
        visit_counts: Dict[int, int] = {}
        last_visit_step_by_pos: Dict[int, int] = {}
        last_visit_context: Dict[str, Any] = {}
        assimilation_plans: List[Tuple[List[int], List[int], str]] = []
        cursor = 0
        while int(cursor) < len(order):
            repeats_tuple, candidate_name = self._assimilation_candidate(remaining=len(order) - int(cursor))
            chunk = [int(x) for x in order[int(cursor) : int(cursor) + len(repeats_tuple)]]
            if not chunk:
                break
            assimilation_plans.append((chunk, [int(x) for x in repeats_tuple[: len(chunk)]], str(candidate_name)))
            cursor += len(chunk)
        total_rollouts = len(assimilation_plans) + len(repair_plans)
        for ridx, (chunk, repeats, _candidate_name) in enumerate(assimilation_plans):
            max_k = int(_cfg_get(self.assimilation_cfg, "max_inner_k", 12))
            requested_inner_k = int(sum(repeats))
            if requested_inner_k > max_k:
                raise ValueError(
                    "Stage2_3 scheduler_v3.assimilation sampled repeat pair exceeds max_inner_k; "
                    f"requested_inner_K={requested_inner_k}, max_inner_k={max_k}, repeats={repeats}"
                )
            history = [p for p in range(int(rows.shape[0])) if int(visit_counts.get(int(p), 0)) > 0 and int(p) not in set(chunk)]
            plan = self._rollout_from_positions(
                rows=rows,
                scene_id=scene_id,
                segment_id=segment_id,
                sequence_id=sequence_id,
                positions=chunk,
                repeat_budgets=repeats,
                rollout_idx=int(ridx),
                rollouts_per_episode=int(total_rollouts),
                phase="assimilation",
                visit_kind="assimilate",
                history_positions=history,
                repair_positions=repair_positions_flat,
                repair_enabled=repair_enabled,
                repair_hash=repair_hash,
                episode_step_offset=step_offset,
                visit_counts=visit_counts,
                last_visit_step_by_pos=last_visit_step_by_pos,
                is_last_rollout=bool(ridx == total_rollouts - 1),
                last_visit_context=last_visit_context,
                phase_max_inner_k=int(max_k),
                requested_inner_k=int(requested_inner_k),
                requested_blocks_per_rollout=int(len(chunk)),
                sequence_target_frames=int(sequence_frame_rule["target_frames"]),
                sequence_min_frames=int(sequence_frame_rule["min_frames"]),
                sequence_allow_short=bool(sequence_frame_rule["allow_short"]),
            )
            rollouts.append(plan)
            step_offset += int(len(plan.steps))
        for repair_idx, (positions, repeats_per_frame, repair_pattern_name, repair_round_idx, requested_frames) in enumerate(repair_plans):
            rollout_idx = len(rollouts)
            max_k = int(_cfg_get(self.repair_cfg, "max_inner_k", 12))
            repeats = [int(repeats_per_frame) for _ in positions]
            requested_inner_k = int(requested_frames) * int(repeats_per_frame)
            if requested_inner_k > max_k:
                raise ValueError(
                    "Stage2_3 scheduler_v3.repair sampled pattern exceeds max_inner_k; "
                    f"candidate={repair_pattern_name}, requested_inner_K={requested_inner_k}, max_inner_k={max_k}"
                )
            history = [p for p in range(int(rows.shape[0])) if int(p) not in set(positions)]
            plan = self._rollout_from_positions(
                rows=rows,
                scene_id=scene_id,
                segment_id=segment_id,
                sequence_id=sequence_id,
                positions=positions,
                repeat_budgets=repeats,
                rollout_idx=int(rollout_idx),
                rollouts_per_episode=int(total_rollouts),
                phase="repair",
                visit_kind="repair",
                history_positions=history,
                repair_positions=repair_positions_flat,
                repair_enabled=True,
                repair_hash=repair_hash,
                episode_step_offset=step_offset,
                visit_counts=visit_counts,
                last_visit_step_by_pos=last_visit_step_by_pos,
                is_last_rollout=bool(repair_idx == len(repair_plans) - 1),
                last_visit_context=last_visit_context,
                repair_round_idx=int(repair_round_idx),
                repair_pattern_name=str(repair_pattern_name),
                phase_max_inner_k=int(max_k),
                requested_inner_k=int(requested_inner_k),
                requested_blocks_per_rollout=int(requested_frames),
                sequence_target_frames=int(sequence_frame_rule["target_frames"]),
                sequence_min_frames=int(sequence_frame_rule["min_frames"]),
                sequence_allow_short=bool(sequence_frame_rule["allow_short"]),
            )
            rollouts.append(plan)
            step_offset += int(len(plan.steps))
        return EpisodePlanV3(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_id=int(self._episode_id_next),
            sequence_id=int(sequence_id),
            frame_set=tuple(int(row["frame_idx"]) for row in rows),
            keyframe_set=tuple(int(row["keyframe_idx"]) for row in rows),
            sampled_order=tuple(int(x) for x in order),
            rollouts=tuple(rollouts),
            repair_enabled=bool(repair_enabled),
            metadata={"segment_row": int(segment_row)},
        )

    def _batch_from_plan(self, plan: RolloutPlanV3) -> Dict[str, Any]:
        assembler = getattr(self.dataset, "_assemble_segment_batch_from_iforward_stage2_3_request", None)
        if not callable(assembler):
            assembler = getattr(self.dataset, "_assemble_segment_batch_from_iforward_request", None)
        if not callable(assembler):
            raise ValueError("Stage2_3 requires dataset._assemble_segment_batch_from_iforward_stage2_3_request")
        return assembler(scene_id=int(plan.scene_id), segment_id=int(plan.segment_id), plan=plan, include_test=bool(self.include_test))

    def _update_last_info(self, plan: RolloutPlanV3) -> None:
        self._last_info = {
            "scheduler_version": self.scheduler_version,
            "model_family": IFORWARD_MODEL_FAMILY,
            "global_step": int(self.global_step),
            "scene_id": int(plan.scene_id),
            "segment_id": int(plan.segment_id),
            "episode_id": int(plan.episode_id),
            "rollout_id_global": int(plan.rollout_id_global),
            "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
            "scheduler_phase": str(plan.scheduler_phase),
            "rollout_phase": str(plan.rollout_phase),
            "shape_name": str(plan.shape_name),
            "sequence_length": int(plan.sequence_length),
            "sequence_target_frames": int(plan.sequence_target_frames),
            "sequence_min_frames": int(plan.sequence_min_frames),
            "sequence_allow_short": bool(plan.sequence_allow_short),
            "sequence_id": int(plan.sequence_id),
            "sequence_positions": [int(x) for x in plan.sequence_positions],
            "episode_positions": [int(x) for x in plan.episode_positions],
            "rollout_positions": [int(x) for x in plan.rollout_positions],
            "sequence_source_frame_indices": [int(x) for x in plan.sequence_source_frame_indices],
            "sequence_keyframe_indices": [int(x) for x in plan.sequence_keyframe_indices],
            "sequence_timestamps_us": [int(x) for x in plan.sequence_timestamps_us],
            "history_positions": [int(x) for x in plan.history_positions],
            "repair_positions": [int(x) for x in plan.repair_positions],
            "repeat_budgets": [int(x) for x in plan.repeat_budgets],
            "frame_gaps": [int(x) for x in plan.frame_gaps],
            "visit_kinds": [str(x) for x in plan.visit_kinds],
            "repair_enabled": bool(plan.repair_enabled),
            "repair_round_idx": int(plan.repair_round_idx),
            "repair_pattern_name": str(plan.repair_pattern_name),
            "repair_permutation_hash": int(plan.repair_permutation_hash),
            "blocks_per_rollout": int(plan.blocks_per_rollout),
            "actual_blocks_per_rollout": int(plan.actual_blocks_per_rollout),
            "requested_blocks_per_rollout": int(plan.requested_blocks_per_rollout),
            "inner_K": int(plan.inner_K),
            "requested_inner_K": int(plan.requested_inner_K),
            "actual_inner_K": int(plan.actual_inner_K),
            "phase_max_inner_k": int(plan.phase_max_inner_k),
            "optimizer_memory_read_count": int(plan.optimizer_memory_read_count),
            "optimizer_memory_write_count": int(plan.optimizer_memory_write_count),
            "observation_commit_count": int(plan.observation_commit_count),
            "index_fingerprint": self.index.fingerprint,
        }

    def _episode_chain_refs_for_preload(self, plan: RolloutPlanV3) -> List[ImageRef]:
        if self._episode_plan is None:
            return []
        if int(plan.rollout_idx_in_episode) != 0:
            return []
        refs: List[ImageRef] = []
        for rollout in tuple(self._episode_plan.rollouts):
            refs.extend(list(rollout.evidence_refs_flat))
            refs.extend(list(rollout.target_refs_flat))
        return _dedupe_refs_keep_order(refs)

    def _submit_preload_hint(self, *, plan: RolloutPlanV3, refs: Sequence[ImageRef], scope: str) -> int:
        build_hint = getattr(self.dataset, "build_preload_hint_light", None) or getattr(self.dataset, "build_preload_hint", None)
        submit = getattr(self.dataset, "submit_preload_hint", None)
        if not callable(build_hint) or not callable(submit):
            return 0
        deduped = _dedupe_refs_keep_order(list(refs))
        if not deduped:
            return 0
        hint = build_hint(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            future_image_refs=deduped,
            scope=str(scope),
        )
        submit(
            hint=hint,
            hint_scope=str(scope),
            epoch_idx=int(self.epoch_idx),
            global_step=int(self.global_step),
            block_idx_global=int(plan.rollout_id_global),
            include_test=bool(self.include_test),
        )
        return int(len(deduped))

    def _emit_preload_hints(self, plan: RolloutPlanV3) -> Dict[str, Any]:
        stats = {"preload_hint_count": 0, "preload_episode_ref_count": 0}
        if not bool(_cfg_get(self.producer_cfg, "preload_next_episode", True)):
            return stats
        refs = _dedupe_refs_keep_order(list(plan.evidence_refs_flat) + list(plan.target_refs_flat))
        current_count = self._submit_preload_hint(
            plan=plan,
            refs=refs,
            scope="stage2_3_current_rollout_view_pack",
        )
        if current_count > 0:
            stats["preload_hint_count"] += 1
        episode_refs = self._episode_chain_refs_for_preload(plan)
        episode_count = self._submit_preload_hint(
            plan=plan,
            refs=episode_refs,
            scope="stage2_3_episode_chain_exact",
        )
        if episode_count > 0:
            stats["preload_hint_count"] += 1
            stats["preload_episode_ref_count"] = int(episode_count)
        return stats

    def pop_events(self) -> List[Dict[str, Any]]:
        out = list(self._pending_events)
        self._pending_events.clear()
        return out

    def get_current_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    def _producer_stats_snapshot(self, *, wait_ms: float = 0.0, build_ms: float = 0.0) -> Dict[str, Any]:
        qsize = int(self._producer_queue.qsize()) if self._producer_queue is not None else 0
        with self._producer_lock:
            produced = int(self._producer_batches_produced)
            errors = int(self._producer_worker_errors)
        return {
            "producer_enabled": bool(self._producer_enabled),
            "producer_queue_depth": int(self._producer_queue_depth),
            "producer_queue_depth_configured": int(getattr(self, "_producer_queue_depth_configured", self._producer_queue_depth)),
            "producer_cuda_queue_depth_cap": int(getattr(self, "_producer_cuda_queue_depth_cap", -1) or -1),
            "producer_queue_size": int(qsize),
            "producer_wait_ms": float(wait_ms),
            "producer_build_ms": float(build_ms),
            "producer_batches_produced": int(produced),
            "producer_worker_errors": int(errors),
        }

    def _run_dataset_log_hooks(self) -> None:
        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))

    def _apply_episode_pinned_scope(self, info: Dict[str, Any]) -> None:
        if not bool(_cfg_get(self.producer_cfg, "episode_pinned_cache", False)):
            return
        scene_id = int(info.get("scene_id", -1))
        segment_id = int(info.get("segment_id", -1))
        if scene_id < 0 or segment_id < 0:
            return
        set_active = getattr(self.dataset, "set_preload_active_scope", None)
        set_training = getattr(self.dataset, "set_preload_training_scope", None)
        if callable(set_active):
            set_active(scene_id, segment_id)
        if callable(set_training):
            set_training(scene_id, segment_id)

    def _next_batch_sync(self, *, run_dataset_log_hooks: bool = True) -> Dict[str, Any]:
        bootstrap_end = int(_cfg_get(self.bootstrap_cfg, "end_step", 5000))
        if int(self.global_step) < int(bootstrap_end):
            plan = self._build_bootstrap_rollout()
            episode_end = True
        else:
            if self._episode_plan is None or int(self._episode_plan_cursor) >= len(self._episode_plan.rollouts):
                self._episode_plan = self._build_episode()
                self._episode_plan_cursor = 0
            plan = self._episode_plan.rollouts[int(self._episode_plan_cursor)]
            episode_end = bool(int(self._episode_plan_cursor) == len(self._episode_plan.rollouts) - 1)
        batch = self._batch_from_plan(plan)
        preload_stats = self._emit_preload_hints(plan)
        self._update_last_info(plan)
        self._last_info.update(preload_stats)
        self._last_info.update(self._producer_stats_snapshot())
        batch["_scheduler_v4_aligned_info"] = dict(self._last_info)
        self._pending_events.append({**dict(self._last_info), "type": "iforward_stage2_3_scheduler"})
        self.global_step += 1
        if str(plan.scheduler_phase) == "bootstrap":
            self._rollout_id_global += 1
            self._episode_id_next += 1
        else:
            self._episode_plan_cursor += 1
            self._rollout_id_global += 1
            if episode_end:
                self._episode_plan = None
                self._episode_plan_cursor = 0
                self._episode_id_next += 1
        self._apply_episode_pinned_scope(self._last_info)
        if run_dataset_log_hooks:
            self._run_dataset_log_hooks()
        return batch

    def _clear_producer_queue(self) -> None:
        if self._producer_queue is None:
            return
        while True:
            try:
                self._producer_queue.get_nowait()
            except queue.Empty:
                return

    def _start_producer(self) -> None:
        if not self._producer_enabled:
            return
        with self._producer_lock:
            if self._producer_thread is not None and self._producer_thread.is_alive():
                return
            self._producer_exception = None
            self._producer_stop.clear()
            self._clear_producer_queue()
            start_state = copy.deepcopy(self.state_dict())
            self._producer_thread = threading.Thread(
                target=self._producer_main,
                args=(start_state,),
                name="Stage23Producer",
                daemon=True,
            )
            self._producer_thread.start()

    def _producer_main(self, start_state: Dict[str, Any]) -> None:
        clone = self._make_producer_clone(start_state)
        try:
            while not self._producer_stop.is_set():
                t0 = time.perf_counter()
                batch = clone._next_batch_sync(run_dataset_log_hooks=False)
                build_ms = float((time.perf_counter() - t0) * 1000.0)
                item = _ProducedBatch(
                    batch=batch,
                    state_after=copy.deepcopy(clone.state_dict()),
                    info=dict(clone._last_info),
                    events=clone.pop_events(),
                    build_ms=float(build_ms),
                )
                with self._producer_lock:
                    self._producer_batches_produced += 1
                while not self._producer_stop.is_set():
                    try:
                        assert self._producer_queue is not None
                        self._producer_queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - propagated on consumer thread with context.
            with self._producer_lock:
                self._producer_exception = exc
                self._producer_worker_errors += 1

    def _raise_producer_exception_if_any(self) -> None:
        with self._producer_lock:
            exc = self._producer_exception
        if exc is not None:
            self.shutdown()
            raise RuntimeError("Stage2_3 producer worker failed") from exc

    def _next_batch_from_producer(self) -> Dict[str, Any]:
        self._start_producer()
        wait_t0 = time.perf_counter()
        while True:
            self._raise_producer_exception_if_any()
            try:
                assert self._producer_queue is not None
                item = self._producer_queue.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        wait_ms = float((time.perf_counter() - wait_t0) * 1000.0)
        self._apply_state_dict(item.state_after)
        metrics = self._producer_stats_snapshot(wait_ms=wait_ms, build_ms=float(item.build_ms))
        self._last_info = dict(item.info)
        self._last_info.update(metrics)
        item.batch["_scheduler_v4_aligned_info"] = dict(self._last_info)
        for ev in item.events:
            ev_out = dict(ev)
            if str(ev_out.get("type", "")) == "iforward_stage2_3_scheduler":
                ev_out.update(metrics)
            self._pending_events.append(ev_out)
        self._apply_episode_pinned_scope(self._last_info)
        self._run_dataset_log_hooks()
        return item.batch

    def next_batch(self) -> Dict[str, Any]:
        if not bool(self._producer_enabled):
            return self._next_batch_sync()
        return self._next_batch_from_producer()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_version": self.scheduler_version,
            "global_step": int(self.global_step),
            "epoch_idx": int(self.epoch_idx),
            "episode_id_next": int(self._episode_id_next),
            "rollout_id_global": int(self._rollout_id_global),
            "episode_plan": self._episode_plan,
            "episode_plan_cursor": int(self._episode_plan_cursor),
            "segment_rows": list(self._segment_rows),
            "segment_cursor": int(self._segment_cursor),
            "bootstrap_pack": list(self._bootstrap_pack),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.shutdown()
        self._apply_state_dict(state)

    def shutdown(self) -> None:
        if hasattr(self, "_producer_stop"):
            self._producer_stop.set()
        self._clear_producer_queue()
        thread = getattr(self, "_producer_thread", None)
        if thread is not None:
            thread.join(timeout=5.0)
            if not thread.is_alive():
                self._producer_thread = None
        self._clear_producer_queue()
        clear_scope = getattr(self.dataset, "clear_preload_scheduler_scope", None)
        if callable(clear_scope):
            clear_scope()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


__all__ = ["IFORWARD_STAGE2_3_SCHEDULER_VERSION", "IFORWARD_STAGE3_0_SCHEDULER_VERSION", "Stage23Scheduler"]
