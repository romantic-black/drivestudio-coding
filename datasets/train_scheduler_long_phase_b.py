from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from models.streetforward.stage6_0.phase_b_long.types import (
    LONG_TARGET_ROLES,
    PHASE_B_LONG_NAME,
    ImageRef,
    LongEpisodeWindow,
    LongRolloutShape,
    LongVisit,
)
from streetforward_core.data.assemblers.phase_b_long_batch_assembler import PhaseBLongBatchAssembler
from streetforward_core.protocols.phase_b_long import (
    PHASE_B_LONG_PROTOCOL_VERSION,
    phase_b_long_plan_from_mapping,
)


class LongPhaseBDatasetLike(Protocol):
    _initialized: bool

    def initialize(self) -> None: ...
    def list_training_scene_ids(self) -> List[int]: ...
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> Any: ...


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        out = node.get(key, default)
        return default if out is None else out
    return getattr(node, key, default)


def _dedupe_refs(refs: Sequence[ImageRef]) -> List[ImageRef]:
    seen = set()
    out: List[ImageRef] = []
    for ref in refs:
        r = (int(ref[0]), int(ref[1]))
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


class TrainSchedulerLongPhaseB:
    """Standalone Long Phase B scheduler.

    Long terminology is deliberately independent of V8/V9 block semantics:
    an EpisodeWindow is a data pool, and a LongRollout is one training sample.
    """

    def __init__(
        self,
        *,
        dataset: LongPhaseBDatasetLike,
        episode_window_cfg: Optional[Any] = None,
        rollout_shapes: Optional[Sequence[Any]] = None,
        rollout_shapes_schedule: Optional[Sequence[Any]] = None,
        anchor_sampling_cfg: Optional[Any] = None,
        evidence_cfg: Optional[Any] = None,
        final_supervision_cfg: Optional[Any] = None,
        traversal_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        include_test: bool = False,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        rigid_meta_cfg: Optional[Any] = None,
        distant_meta_cfg: Optional[Any] = None,
        fail_fast: bool = True,
    ) -> None:
        self.dataset = dataset
        self.episode_window_cfg = dict(episode_window_cfg or {})
        self.rollout_shapes_cfg = [dict(x) for x in list(rollout_shapes or [])]
        self.rollout_shapes_schedule = [dict(x) for x in list(rollout_shapes_schedule or [])]
        self.anchor_sampling_cfg = dict(anchor_sampling_cfg or {})
        self.evidence_cfg = dict(evidence_cfg or {})
        self.final_supervision_cfg = dict(final_supervision_cfg or {})
        self.traversal_cfg = dict(traversal_cfg or {})
        self.preload_cfg = dict(preload_cfg or {})
        self.include_test = bool(include_test)
        self.fixed_scene_id = int(fixed_scene_id) if fixed_scene_id is not None else None
        self.fixed_segment_id = int(fixed_segment_id) if fixed_segment_id is not None else None
        self.rigid_meta_cfg = dict(rigid_meta_cfg or {})
        self.distant_meta_cfg = dict(distant_meta_cfg or {})
        self.fail_fast = bool(fail_fast)

        if not self.rollout_shapes_cfg and not self.rollout_shapes_schedule:
            self.rollout_shapes_schedule = [
                {
                    "start_step": 0,
                    "shapes": [
                        {"name": "r2_a2", "repeats_per_anchor": 2, "anchors_per_rollout": 2, "prob": 1.0}
                    ],
                }
            ]
        if int(_cfg_get(self.evidence_cfg, "cams_per_visit", 1)) != 1:
            raise ValueError("scheduler_long_phase_b V1 requires evidence.cams_per_visit=1.")

        initialized = getattr(self.dataset, "_initialized", True)
        if initialized is False:
            self.dataset.initialize()

        self.U = 1
        self.global_step = 0
        self.epoch_idx = 0
        self._episode_window_id = 0
        self._rollout_id_global = 0
        self._rollout_id_in_window = 0
        self._segment_cursor = 0
        self._pending_events: List[Dict[str, Any]] = []
        self._current_window: Optional[LongEpisodeWindow] = None
        self._last_info: Dict[str, Any] = {"scheduler_version": "long_v1", "U": 1, "global_step": 0}
        self._assembler = PhaseBLongBatchAssembler(dataset)
        self._segments = self._build_segment_plan()
        if not self._segments:
            raise ValueError("scheduler_long_phase_b found no training segments.")

    def _emit(self, event: Dict[str, Any]) -> None:
        self._pending_events.append(dict(event))

    def pop_events(self) -> List[Dict[str, Any]]:
        out = list(self._pending_events)
        self._pending_events.clear()
        return out

    def get_current_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    def _build_segment_plan(self) -> List[Tuple[int, int]]:
        if self.fixed_scene_id is not None:
            scene_ids = [int(self.fixed_scene_id)]
        else:
            scene_ids = [int(x) for x in self.dataset.list_training_scene_ids()]
        if str(_cfg_get(self.traversal_cfg, "scene_order", "ascending")) == "shuffle_per_epoch":
            random.shuffle(scene_ids)
        out: List[Tuple[int, int]] = []
        for sid in scene_ids:
            if self.fixed_segment_id is not None:
                seg_ids = [int(self.fixed_segment_id)]
            else:
                seg_ids = [int(x) for x in self.dataset.list_segment_ids(int(sid))]
            if str(_cfg_get(self.traversal_cfg, "segment_order", "ascending")) == "shuffle_per_epoch":
                random.shuffle(seg_ids)
            for seg in seg_ids:
                out.append((int(sid), int(seg)))
        return out

    @staticmethod
    def _segment_frames(sidx: Any) -> List[int]:
        raw = list(getattr(sidx, "frame_indices", []) or [])
        if raw:
            return sorted({int(x) for x in raw})
        frames: List[int] = []
        for group in dict(getattr(sidx, "keyframe_to_frames", {}) or {}).values():
            frames.extend(int(x) for x in list(group or []))
        return sorted(set(frames))

    def _new_episode_window(self) -> LongEpisodeWindow:
        scene_id, segment_id = self._segments[int(self._segment_cursor) % len(self._segments)]
        self._segment_cursor += 1
        sidx = self.dataset.get_segment_index(int(scene_id), int(segment_id))
        all_frames = self._segment_frames(sidx)
        min_required = int(_cfg_get(self.episode_window_cfg, "min_frames_required", 2))
        if len(all_frames) < min_required:
            raise ValueError(
                f"scheduler_long_phase_b requires at least {min_required} frames, got {len(all_frames)} "
                f"(scene={scene_id} segment={segment_id})"
            )
        frames_per_window = int(_cfg_get(self.episode_window_cfg, "frames_per_window", len(all_frames)))
        frames_per_window = max(1, min(int(frames_per_window), len(all_frames)))
        if len(all_frames) == frames_per_window:
            frame_pool = list(all_frames)
        else:
            start = random.randrange(0, len(all_frames) - frames_per_window + 1)
            frame_pool = list(all_frames[start : start + frames_per_window])
        cam_pool = list(range(int(getattr(sidx, "num_cams", 1))))
        if not cam_pool:
            raise ValueError("scheduler_long_phase_b requires num_cams >= 1.")
        window = LongEpisodeWindow(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            frame_pool=[int(x) for x in frame_pool],
            cam_pool=[int(x) for x in cam_pool],
            segment_start_frame=int(all_frames[0]),
            segment_end_frame=int(all_frames[-1]),
            rigid_meta=dict(self.rigid_meta_cfg),
            distant_meta=dict(self.distant_meta_cfg),
            episode_seed=random.randrange(0, 2**31 - 1),
            rollout_budget=int(_cfg_get(self.episode_window_cfg, "rollout_budget_per_episode", 1)),
        )
        self._episode_window_id += 1
        self._rollout_id_in_window = 0
        self._emit(
            {
                "type": "segment_begin",
                "scheduler_version": "long_v1",
                "global_step": int(self.global_step),
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "U": 1,
                "segment_budget_u": int(window.rollout_budget),
                "segment_step_budget": int(window.rollout_budget),
                "updates_per_block": 1,
            }
        )
        return window

    def _active_shapes(self) -> List[LongRolloutShape]:
        schedule = sorted(self.rollout_shapes_schedule, key=lambda x: int(_cfg_get(x, "start_step", 0)))
        raw_shapes = list(self.rollout_shapes_cfg)
        for item in schedule:
            if int(self.global_step) >= int(_cfg_get(item, "start_step", 0)):
                raw_shapes = [dict(x) for x in list(_cfg_get(item, "shapes", []) or [])]
        if not raw_shapes:
            raise ValueError("scheduler_long_phase_b active rollout shape list is empty.")
        shapes = []
        for raw in raw_shapes:
            shapes.append(
                LongRolloutShape(
                    name=str(_cfg_get(raw, "name", f"r{_cfg_get(raw, 'repeats_per_anchor', 1)}_a{_cfg_get(raw, 'anchors_per_rollout', 1)}")),
                    repeats_per_anchor=max(int(_cfg_get(raw, "repeats_per_anchor", 1)), 1),
                    anchors_per_rollout=max(int(_cfg_get(raw, "anchors_per_rollout", 1)), 1),
                )
            )
        return shapes

    def _sample_shape(self) -> LongRolloutShape:
        raw_shapes: List[Dict[str, Any]] = []
        schedule = sorted(self.rollout_shapes_schedule, key=lambda x: int(_cfg_get(x, "start_step", 0)))
        for item in schedule:
            if int(self.global_step) >= int(_cfg_get(item, "start_step", 0)):
                raw_shapes = [dict(x) for x in list(_cfg_get(item, "shapes", []) or [])]
        if not raw_shapes:
            raw_shapes = [dict(x) for x in self.rollout_shapes_cfg]
        if not raw_shapes:
            raise ValueError("scheduler_long_phase_b has no rollout_shapes.")
        probs = [max(float(_cfg_get(x, "prob", 1.0)), 0.0) for x in raw_shapes]
        if sum(probs) <= 0.0:
            probs = [1.0 for _ in raw_shapes]
        raw = random.choices(raw_shapes, weights=probs, k=1)[0]
        return LongRolloutShape(
            name=str(_cfg_get(raw, "name", f"r{_cfg_get(raw, 'repeats_per_anchor', 1)}_a{_cfg_get(raw, 'anchors_per_rollout', 1)}")),
            repeats_per_anchor=max(int(_cfg_get(raw, "repeats_per_anchor", 1)), 1),
            anchors_per_rollout=max(int(_cfg_get(raw, "anchors_per_rollout", 1)), 1),
        )

    def _active_order_probs(self) -> Tuple[float, float, float]:
        cfg = self.anchor_sampling_cfg
        chrono = float(_cfg_get(cfg, "allow_chronological_order_prob", 0.1))
        reverse = float(_cfg_get(cfg, "allow_reverse_order_prob", 0.1))
        random_p = float(_cfg_get(cfg, "allow_random_order_prob", 0.8))
        for item in sorted(list(_cfg_get(cfg, "order_prob_schedule", []) or []), key=lambda x: int(_cfg_get(x, "start_step", 0))):
            if int(self.global_step) >= int(_cfg_get(item, "start_step", 0)):
                chrono = float(_cfg_get(item, "chronological", chrono))
                reverse = float(_cfg_get(item, "reverse", reverse))
                random_p = float(_cfg_get(item, "random", random_p))
        total = max(float(chrono + reverse + random_p), 1.0e-8)
        return chrono / total, reverse / total, random_p / total

    def _sample_anchor_frames(self, window: LongEpisodeWindow, shape: LongRolloutShape) -> List[int]:
        count = int(shape.anchors_per_rollout)
        pool = [int(x) for x in window.frame_pool]
        if count > len(pool):
            raise ValueError(f"cannot sample {count} Long anchors from frame_pool len={len(pool)}")
        min_span = int(_cfg_get(self.anchor_sampling_cfg, "min_temporal_span", 1))
        max_span = int(_cfg_get(self.anchor_sampling_cfg, "max_temporal_span", max(pool) - min(pool) + 1))
        min_gap = int(_cfg_get(self.anchor_sampling_cfg, "min_pairwise_gap", 1))
        for _ in range(128):
            frames = sorted(random.sample(pool, count))
            span = int(frames[-1] - frames[0]) if len(frames) > 1 else 0
            gaps_ok = all(abs(int(a) - int(b)) >= int(min_gap) for a, b in zip(frames[:-1], frames[1:]))
            if span >= min_span and span <= max_span and gaps_ok:
                return frames
        if self.fail_fast:
            raise ValueError("could not sample Long anchors satisfying temporal span/gap constraints.")
        return sorted(random.sample(pool, count))

    def _order_anchors(self, frames_chrono: List[int]) -> Tuple[str, List[int]]:
        chrono_p, reverse_p, random_p = self._active_order_probs()
        mode = random.choices(["chronological", "reverse", "random"], weights=[chrono_p, reverse_p, random_p], k=1)[0]
        if mode == "chronological":
            return mode, list(frames_chrono)
        if mode == "reverse":
            return mode, list(reversed(frames_chrono))
        out = list(frames_chrono)
        random.shuffle(out)
        return mode, out

    def _repeat_cams(self, *, cam_pool: List[int], frame_idx: int, repeats: int) -> List[int]:
        _ = frame_idx
        cams = list(cam_pool)
        if not cams:
            raise ValueError("cam_pool must be non-empty.")

        reserve_raw = _cfg_get(self.evidence_cfg, "reserve_nvs_cams_per_anchor", None)
        if reserve_raw is None:
            reserve_raw = _cfg_get(self.final_supervision_cfg, "reserve_nvs_cams_per_anchor", 1)
        reserve = max(int(reserve_raw), 0)
        max_evidence = max(1, len(cams) - int(reserve))

        distinct = int(_cfg_get(self.evidence_cfg, "distinct_cams_per_anchor", 1))
        distinct = max(1, min(int(distinct), int(max_evidence), len(cams)))
        allow_same = bool(_cfg_get(self.evidence_cfg, "allow_same_cam_repeat", True))

        random.shuffle(cams)
        chosen = [int(c) for c in cams[:distinct]]
        if not allow_same and int(repeats) > len(chosen) and self.fail_fast:
            raise ValueError(
                "scheduler_long_phase_b evidence.allow_same_cam_repeat=false cannot satisfy "
                f"repeats_per_anchor={int(repeats)} with distinct_cams_per_anchor={int(distinct)}. "
                "Set allow_same_cam_repeat=true for iterative refinement repeats."
            )
        return [int(chosen[i % len(chosen)]) for i in range(int(repeats))]

    def _nvs_cams(self, *, frame_idx: int, count: int, evidence_refs: Sequence[ImageRef], cam_pool: List[int]) -> Tuple[List[int], int]:
        if int(count) <= 0:
            return [], 0
        evidence_cams = {int(c) for f, c in evidence_refs if int(f) == int(frame_idx)}
        non_evidence = [int(c) for c in cam_pool if int(c) not in evidence_cams]
        random.shuffle(non_evidence)
        if len(non_evidence) >= int(count):
            return non_evidence[: int(count)], 0
        allow_overlap = bool(_cfg_get(self.final_supervision_cfg, "allow_evidence_overlap_for_warmup", True))
        if not allow_overlap and self.fail_fast:
            raise ValueError("not enough non-evidence cameras for Long NVS target sampling.")
        fallback = [int(c) for c in cam_pool if int(c) in evidence_cams]
        random.shuffle(fallback)
        merged = non_evidence + fallback
        return [int(merged[i % len(merged)]) for i in range(int(count))], max(int(count) - len(non_evidence), 0)

    @staticmethod
    def _role_add(
        refs: List[ImageRef],
        roles: List[str],
        *,
        role: str,
        new_refs: Sequence[ImageRef],
    ) -> None:
        if role not in LONG_TARGET_ROLES:
            raise ValueError(f"unknown Long target role {role!r}")
        existing = set(refs)
        for ref in new_refs:
            r = (int(ref[0]), int(ref[1]))
            if r in existing:
                continue
            refs.append(r)
            roles.append(str(role))
            existing.add(r)

    def _sample_rollout(self, window: LongEpisodeWindow) -> Dict[str, Any]:
        shape = self._sample_shape()
        anchors_chrono = self._sample_anchor_frames(window, shape)
        order_mode, anchors_order = self._order_anchors(anchors_chrono)
        chrono_rank = {int(frame): int(i) for i, frame in enumerate(anchors_chrono)}
        frame_span = max(int(window.segment_end_frame) - int(window.segment_start_frame), 1)
        visits: List[LongVisit] = []
        evidence_refs_by_step: List[List[ImageRef]] = []
        for rollout_rank, frame_idx in enumerate(anchors_order):
            cams = self._repeat_cams(cam_pool=list(window.cam_pool), frame_idx=int(frame_idx), repeats=int(shape.repeats_per_anchor))
            for repeat_idx, cam_idx in enumerate(cams):
                step_idx = len(visits)
                visit = LongVisit(
                    step_idx=int(step_idx),
                    anchor_id=int(rollout_rank),
                    frame_idx=int(frame_idx),
                    cam_idx=int(cam_idx),
                    repeat_idx=int(repeat_idx),
                    rollout_order_rank=int(rollout_rank),
                    chronological_rank=int(chrono_rank[int(frame_idx)]),
                    visit_pos_code=float(step_idx) / float(max(shape.inner_K - 1, 1)),
                    frame_time_code=float(int(frame_idx) - int(window.segment_start_frame)) / float(frame_span),
                    chronological_rank_code=float(chrono_rank[int(frame_idx)]) / float(max(len(anchors_chrono) - 1, 1)),
                    repeat_idx_code=float(repeat_idx) / float(max(shape.repeats_per_anchor - 1, 1)),
                )
                visits.append(visit)
                evidence_refs_by_step.append([(int(frame_idx), int(cam_idx))])
        source_refs = _dedupe_refs([ref for step in evidence_refs_by_step for ref in step])
        evidence_refs = list(source_refs)
        terminal_frame = int(anchors_order[-1])
        history_frames = [int(x) for x in anchors_order[:-1]]
        history_count = int(_cfg_get(self.final_supervision_cfg, "history_anchor_count", min(3, len(history_frames))))
        history_frames = history_frames[-max(history_count, 0) :]

        def _count(name: str, fallback_key: str, default: int) -> int:
            node = _cfg_get(self.final_supervision_cfg, name, {}) or {}
            return max(int(_cfg_get(node, "cams_per_anchor", _cfg_get(node, "cams", _cfg_get(self.final_supervision_cfg, fallback_key, default)))), 0)

        history_recon_count = _count("final_history_recon", "history_recon_cams_per_anchor", 1)
        history_nvs_count = _count("final_history_nvs", "history_nvs_cams_per_anchor", 1)
        current_recon_count = _count("final_current_recon", "current_recon_cams", 1)
        current_nvs_count = _count("final_current_nvs", "current_nvs_cams", 1)
        target_refs: List[ImageRef] = []
        target_roles: List[str] = []
        nvs_fallback = 0
        for frame_idx in history_frames:
            ev_cams = [int(c) for f, c in evidence_refs if int(f) == int(frame_idx)]
            self._role_add(
                target_refs,
                target_roles,
                role="final_history_recon",
                new_refs=[(int(frame_idx), int(c)) for c in ev_cams[:history_recon_count]],
            )
            cams, fb = self._nvs_cams(frame_idx=int(frame_idx), count=history_nvs_count, evidence_refs=evidence_refs, cam_pool=list(window.cam_pool))
            nvs_fallback += int(fb)
            self._role_add(
                target_refs,
                target_roles,
                role="final_history_nvs",
                new_refs=[(int(frame_idx), int(c)) for c in cams],
            )
        terminal_ev_cams = [int(c) for f, c in evidence_refs if int(f) == int(terminal_frame)]
        self._role_add(
            target_refs,
            target_roles,
            role="final_current_recon",
            new_refs=[(int(terminal_frame), int(c)) for c in terminal_ev_cams[:current_recon_count]],
        )
        cams, fb = self._nvs_cams(frame_idx=int(terminal_frame), count=current_nvs_count, evidence_refs=evidence_refs, cam_pool=list(window.cam_pool))
        nvs_fallback += int(fb)
        self._role_add(
            target_refs,
            target_roles,
            role="final_current_nvs",
            new_refs=[(int(terminal_frame), int(c)) for c in cams],
        )
        if not target_refs:
            raise ValueError("scheduler_long_phase_b sampled zero final target refs.")
        nvs_fallback_ratio = float(nvs_fallback) / float(max(len(target_refs), 1))
        max_nvs_fallback_ratio = float(_cfg_get(self.final_supervision_cfg, "max_nvs_fallback_ratio", 0.25))
        if nvs_fallback_ratio > max_nvs_fallback_ratio and self.fail_fast:
            raise ValueError(
                "scheduler_long_phase_b NVS fallback ratio too high: "
                f"{nvs_fallback_ratio:.3f} > {max_nvs_fallback_ratio:.3f}. "
                "Evidence cameras are exhausting heldout NVS cameras."
            )
        required_final_roles = [
            str(x)
            for x in list(
                _cfg_get(
                    self.final_supervision_cfg,
                    "required_final_roles",
                    ["final_current_recon", "final_current_nvs"],
                )
                or []
            )
        ]

        request_meta = {
            "protocol_version": PHASE_B_LONG_PROTOCOL_VERSION,
            "scheduler_version": "long_v1",
            "scheduler_phase": PHASE_B_LONG_NAME,
            "assembly_mode": "image_ref_long_v1",
            "inner_K": int(shape.inner_K),
            "shape_name": str(shape.name),
            "repeats_per_anchor": int(shape.repeats_per_anchor),
            "anchors_per_rollout": int(shape.anchors_per_rollout),
            "episode_window_id": int(self._episode_window_id),
            "rollout_id": int(self._rollout_id_global),
            "rollout_id_in_episode": int(self._rollout_id_in_window),
            "anchor_order_mode": str(order_mode),
            "anchor_frames_chronological": [int(x) for x in anchors_chrono],
            "anchor_frames_rollout_order": [int(x) for x in anchors_order],
            "visits": [asdict(v) for v in visits],
            "evidence_refs_by_step": [[tuple(x) for x in step] for step in evidence_refs_by_step],
            "step_frame_indices": [int(v.frame_idx) for v in visits],
            "step_repeat_indices": [int(v.repeat_idx) for v in visits],
            "step_anchor_ids": [int(v.anchor_id) for v in visits],
            "step_rollout_order_ranks": [int(v.rollout_order_rank) for v in visits],
            "step_chronological_ranks": [int(v.chronological_rank) for v in visits],
            "visit_pos_codes": [float(v.visit_pos_code) for v in visits],
            "frame_time_codes": [float(v.frame_time_code) for v in visits],
            "chronological_rank_codes": [float(v.chronological_rank_code) for v in visits],
            "repeat_idx_codes": [float(v.repeat_idx_code) for v in visits],
            "source_image_refs": [tuple(x) for x in source_refs],
            "source_image_ref": tuple(source_refs[0]),
            "target_image_refs": [tuple(x) for x in target_refs],
            "target_image_roles": [str(x) for x in target_roles],
            "required_final_roles": list(required_final_roles),
            "final_history_recon_refs": [ref for ref, role in zip(target_refs, target_roles) if role == "final_history_recon"],
            "final_history_nvs_refs": [ref for ref, role in zip(target_refs, target_roles) if role == "final_history_nvs"],
            "final_current_recon_refs": [ref for ref, role in zip(target_refs, target_roles) if role == "final_current_recon"],
            "final_current_nvs_refs": [ref for ref, role in zip(target_refs, target_roles) if role == "final_current_nvs"],
            "target_role_set": list(LONG_TARGET_ROLES),
            "query_label_refs": [],
            "prefix_loss_refs_by_step": [[] for _ in evidence_refs_by_step],
            "nearby_loss_refs_by_step": [[] for _ in evidence_refs_by_step],
            "block_loss_refs_by_step": [[] for _ in evidence_refs_by_step],
            "num_cams": int(len(window.cam_pool)),
            "scene_id": int(window.scene_id),
            "segment_id": int(window.segment_id),
            "episode_id": int(self._episode_window_id),
            "episode_idx_global": int(self._episode_window_id),
            "rigid_meta": dict(window.rigid_meta),
            "distant_meta": dict(window.distant_meta),
            "tbptt": {"enable": False, "reset_vsm_per_rollout": True, "reset_offset_per_rollout": True},
            "nvs_fallback_count": int(nvs_fallback),
            "nvs_fallback_to_evidence_cam_ratio": float(nvs_fallback_ratio),
            "max_nvs_fallback_ratio": float(max_nvs_fallback_ratio),
        }
        return request_meta

    def _batch_from_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        plan = phase_b_long_plan_from_mapping(meta)
        return self._assembler.materialize(plan, include_test=bool(self.include_test))

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        if self._current_window is None:
            self._current_window = self._new_episode_window()
        meta = self._sample_rollout(self._current_window)
        batch = self._batch_from_meta(meta)
        batch["_scheduler_long_phase_b_peek"] = True
        return batch

    def next_batch(self) -> Dict[str, Any]:
        if self._current_window is None or self._rollout_id_in_window >= int(self._current_window.rollout_budget):
            self._current_window = self._new_episode_window()
        meta = self._sample_rollout(self._current_window)
        batch = self._batch_from_meta(meta)
        self._last_info = {
            "scheduler_version": "long_v1",
            "phase": PHASE_B_LONG_NAME,
            "global_step": int(self.global_step),
            "segment_local_step": int(self._rollout_id_in_window),
            "U": 1,
            "scene_id": int(meta["scene_id"]),
            "segment_id": int(meta["segment_id"]),
            "rollout_id": int(self._rollout_id_global),
            "episode_window_id": int(self._episode_window_id),
            "shape_name": str(meta["shape_name"]),
            "anchor_order_mode": str(meta["anchor_order_mode"]),
            "inner_K": int(meta["inner_K"]),
            "block_order": "long_rollout",
        }
        is_last_rollout = self._rollout_id_in_window + 1 >= int(self._current_window.rollout_budget)
        event_common = {
            "scheduler_version": "long_v1",
            "global_step": int(self.global_step),
            "scene_id": int(meta["scene_id"]),
            "segment_id": int(meta["segment_id"]),
            "episode_idx_global": int(self._episode_window_id),
            "rollout_id": int(self._rollout_id_global),
            "rollout_id_in_episode": int(self._rollout_id_in_window),
            "rollout_budget_per_episode": int(self._current_window.rollout_budget),
            "shape_name": str(meta["shape_name"]),
        }
        self._emit({"type": "rollout_end", **event_common})
        if is_last_rollout:
            self._emit({"type": "episode_end", **event_common})
        self.global_step += 1
        self._rollout_id_global += 1
        self._rollout_id_in_window += 1
        return batch


__all__ = ["TrainSchedulerLongPhaseB"]
