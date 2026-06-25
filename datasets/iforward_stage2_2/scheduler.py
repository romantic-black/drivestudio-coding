from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datasets.train_scheduler_iforward import IFORWARD_MODEL_FAMILY, ImageRef, _dedupe_refs_keep_order

from .episode_producer import EpisodeProducer
from .index_builder import build_stage2_2_index_from_dataset
from .index_format import (
    IFORWARD_STAGE2_2_SCHEDULER_VERSION,
    PROTOCOL_IDS,
    PROTOCOL_NAMES,
    stable_uint64,
    protocol_offsets,
)
from .index_loader import Stage22Index, load_stage2_2_index
from .protocol_sampler import ProtocolDeficitSampler
from .schema import (
    STAGE22_CURRENT_ROLE,
    STAGE22_HISTORY_ROLE,
    EpisodePlan,
    RolloutPlan,
    Stage22StepPlan,
    make_final_supervision,
)
from .traversal import Stage22Traversal


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


def _rng_token(rng: random.Random) -> int:
    return int(rng.getrandbits(63))


def _cams_from_mask(mask: int, fallback_num_cams: int = 3) -> List[int]:
    cams = [int(i) for i in range(32) if int(mask) & (1 << int(i))]
    if not cams:
        cams = list(range(int(fallback_num_cams)))
    return cams


def _refs_for_frame(frame_idx: int, cams: Sequence[int]) -> List[ImageRef]:
    return [(int(frame_idx), int(cam)) for cam in cams]


class Stage22Scheduler:
    def __init__(
        self,
        *,
        dataset: Any,
        cfg: Optional[Any] = None,
        index: Optional[Stage22Index] = None,
        index_dir: Optional[str] = None,
        traversal_cfg: Optional[Any] = None,
        bootstrap_cfg: Optional[Any] = None,
        protocol_cfg: Optional[Any] = None,
        causal_cfg: Optional[Any] = None,
        repair_cfg: Optional[Any] = None,
        supervision_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        include_test: bool = False,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        seed: Optional[int] = None,
        fail_fast: bool = True,
    ) -> None:
        self.dataset = dataset
        self.cfg = cfg or {}
        sched_cfg = _cfg_get(self.cfg, "scheduler_stage2_2", {}) or {}
        self.traversal_cfg = dict(traversal_cfg or _cfg_get(sched_cfg, "traversal", {}) or {})
        self.bootstrap_cfg = dict(bootstrap_cfg or _cfg_get(sched_cfg, "bootstrap", {}) or {})
        self.protocol_cfg = dict(protocol_cfg or _cfg_get(sched_cfg, "protocol", {}) or {})
        self.causal_cfg = dict(causal_cfg or _cfg_get(sched_cfg, "causal", {}) or {})
        self.repair_cfg = dict(repair_cfg or _cfg_get(sched_cfg, "repair", {}) or {})
        self.supervision_cfg = dict(supervision_cfg or _cfg_get(sched_cfg, "supervision", {}) or {})
        self.preload_cfg = dict(preload_cfg or _cfg_get(sched_cfg, "preload", {}) or {})
        self.include_test = bool(include_test)
        self.fail_fast = bool(fail_fast)
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
                raise ValueError("Stage2_2 requires scheduler_stage2_2.index_fingerprint when index_dir is configured")
            self.index = load_stage2_2_index(index_dir, expected_fingerprint=expected_fp)
        else:
            self.index = build_stage2_2_index_from_dataset(
                dataset=dataset,
                cfg=self.cfg,
                fixed_scene_id=self.fixed_scene_id,
                fixed_segment_id=self.fixed_segment_id,
            )
        raw_seed = seed if seed is not None else _cfg_get(self.traversal_cfg, "seed", None)
        self.rng = random.Random(int(raw_seed)) if raw_seed is not None else random.Random()
        weights = dict(_cfg_get(self.protocol_cfg, "weights", {}) or {"D1": 1.0, "D2": 1.0, "I123": 1.0})
        self.protocol_sampler = ProtocolDeficitSampler(weights)
        self.traversal = Stage22Traversal(
            self.index,
            scene_order=str(_cfg_get(self.traversal_cfg, "scene_order", "shuffle_per_epoch")),
            segment_order=str(_cfg_get(self.traversal_cfg, "segment_order", "shuffle_per_epoch")),
            forbid_consecutive_same_scene=bool(_cfg_get(self.traversal_cfg, "forbid_consecutive_same_scene", True)),
            seed=int(raw_seed) if raw_seed is not None else 0,
        )
        self.producer = EpisodeProducer(maxsize=int(_cfg_get(self.protocol_cfg, "producer_queue_size", 2)))
        self.global_step = 0
        self.epoch_idx = 0
        self._episode_id_next = 0
        self._rollout_id_global = 0
        self._episode_plan: Optional[EpisodePlan] = None
        self._episode_plan_cursor = 0
        self._pending_events: List[Dict[str, Any]] = []
        self._last_info: Dict[str, Any] = {
            "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
            "global_step": 0,
            "index_fingerprint": self.index.fingerprint,
        }
        self._validate_static_cfg()
        self._validate_index_against_cfg()
        self._emit_eligibility_summary()

    def _validate_static_cfg(self) -> None:
        if int(_cfg_get(self.causal_cfg, "rollouts_per_episode", 5)) != 5:
            raise ValueError("Stage2_2 requires causal.rollouts_per_episode=5")
        if int(_cfg_get(self.causal_cfg, "blocks_per_rollout", 2)) != 2:
            raise ValueError("Stage2_2 requires causal.blocks_per_rollout=2")
        if int(_cfg_get(self.causal_cfg, "repeats_per_block", 4)) != 4:
            raise ValueError("Stage2_2 requires causal.repeats_per_block=4")
        repair_blocks = int(_cfg_get(self.repair_cfg, "blocks_per_rollout", 10))
        if repair_blocks < 1 or repair_blocks > 10:
            raise ValueError("Stage2_2 requires repair.blocks_per_rollout in [1, 10]")
        if int(_cfg_get(self.repair_cfg, "repeats_per_block", 1)) != 1:
            raise ValueError("Stage2_2 requires repair.repeats_per_block=1")

    def _validate_index_against_cfg(self) -> None:
        if not self.fail_fast:
            return
        if "frame_idx_times_frame_period_us" in str(self.index.timestamp_source) and not bool(
            _cfg_get(_cfg_get(_cfg_get(self.cfg, "scheduler_stage2_2", {}) or {}, "time", {}) or {}, "allow_synthetic_timestamp", False)
        ):
            raise ValueError("Stage2_2 formal training index must use real timestamp_us; synthetic timestamps are test-only")
        if self.fixed_scene_id is not None:
            expected_scenes = {int(self.fixed_scene_id)}
        else:
            try:
                expected_scenes = {int(x) for x in self.dataset.list_training_scene_ids()}
            except Exception:
                expected_scenes = set()
        indexed_scenes = {int(x) for x in np.unique(self.index.segments["scene_id"]).tolist()}
        if expected_scenes and indexed_scenes != expected_scenes:
            missing = sorted(expected_scenes - indexed_scenes)
            extra = sorted(indexed_scenes - expected_scenes)
            raise ValueError(
                "Stage2_2 index scene set mismatch: "
                f"configured={len(expected_scenes)} indexed={len(indexed_scenes)} missing={missing[:16]} extra={extra[:16]}"
            )

    def _emit_eligibility_summary(self) -> None:
        by_protocol: Dict[str, int] = {}
        for name, pid in PROTOCOL_IDS.items():
            by_protocol[str(name)] = int(np.count_nonzero(self.index.windows["protocol_id"] == int(pid)))
        indexed_scenes = sorted(int(x) for x in np.unique(self.index.segments["scene_id"]).tolist())
        self._pending_events.append(
            {
                "type": "iforward_stage2_2_eligibility",
                "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
                "eligible_scene_count": int(len(indexed_scenes)),
                "eligible_segment_count": int(self.index.segments.shape[0]),
                "eligible_window_count": int(self.index.windows.shape[0]),
                "eligible_windows_by_protocol": dict(by_protocol),
                "timestamp_source": str(self.index.timestamp_source),
                "num_cams": int(self.index.num_cams),
                "index_fingerprint": self.index.fingerprint,
            }
        )

    def _frame_rows_for_window(self, window: np.void, protocol: str) -> np.ndarray:
        segment_row = int(window["segment_row"])
        start = int(window["start_local_frame"])
        offsets = np.asarray(protocol_offsets(str(protocol), int(window["pattern_id"])), dtype=np.int64)
        frames = self.index.frames_for_segment_row(int(segment_row))
        return frames[start + offsets]

    def _segment_ids_for_window(self, window: np.void) -> Tuple[int, int, int]:
        segment_row = int(window["segment_row"])
        seg = self.index.segments[int(segment_row)]
        return int(segment_row), int(seg["scene_id"]), int(seg["segment_id"])

    def _repair_blocks_per_rollout(self) -> int:
        return int(_cfg_get(self.repair_cfg, "blocks_per_rollout", 10))

    def _repair_permutation(self, count: Optional[int] = None) -> Tuple[int, ...]:
        base = list(range(10))
        count = int(self._repair_blocks_per_rollout() if count is None else count)
        if count < 1 or count > len(base):
            raise ValueError("Stage2_2 repair permutation count must be in [1, 10]")
        identity = tuple(int(x) for x in base[:count])
        for _ in range(16):
            perm = list(base)
            self.rng.shuffle(perm)
            selected = tuple(int(x) for x in perm[:count])
            if selected != identity:
                return selected
        if count < len(base):
            return tuple(range(1, count + 1))
        return tuple(list(range(1, 10)) + [0])

    def _make_steps(
        self,
        *,
        frame_rows: np.ndarray,
        positions: Sequence[int],
        repeats_per_block: int,
        visit_kind: str,
        phase: str,
        episode_step_offset: int,
        temporal_read: bool,
        temporal_commit: bool,
        observation_commit: bool,
        update_optimizer_memory: bool,
        physical_time_advance: bool,
        previous_physical_pos: Optional[int] = None,
    ) -> Tuple[List[Stage22StepPlan], List[ImageRef]]:
        steps: List[Stage22StepPlan] = []
        evidence_flat: List[ImageRef] = []
        zero_motion_phase = str(phase) in {"repair", "stress"}
        prev_pos: Optional[int] = None if bool(zero_motion_phase) else previous_physical_pos
        for rank, pos_raw in enumerate(positions):
            pos = int(pos_raw)
            row = frame_rows[int(pos)]
            frame_idx = int(row["frame_idx"])
            keyframe_idx = int(row["keyframe_idx"])
            timestamp_us = int(row["timestamp_us"])
            prev_row = frame_rows[int(prev_pos)] if prev_pos is not None else row
            if bool(zero_motion_phase):
                delta_t_sec = 0.0
                frame_gap = 0
                ego_delta = np.zeros((3,), dtype=np.float32)
                ego_delta_yaw = 0.0
            else:
                delta_t_sec = float(timestamp_us - int(prev_row["timestamp_us"])) / 1.0e6 if prev_pos is not None else 0.0
                frame_gap = int(frame_idx - int(prev_row["frame_idx"])) if prev_pos is not None else 0
                ego_delta = np.asarray(row["ego_translation"], dtype=np.float32) - np.asarray(prev_row["ego_translation"], dtype=np.float32)
                ego_delta_yaw = float(row["ego_yaw"] - float(prev_row["ego_yaw"]))
            cams = _cams_from_mask(int(row["available_camera_mask"]), int(self.index.num_cams))
            refs = _refs_for_frame(int(frame_idx), cams)
            evidence_flat.extend(refs)
            for repeat in range(int(repeats_per_block)):
                is_enter = int(repeat) == 0
                is_exit = int(repeat) == int(repeats_per_block) - 1
                step_idx = len(steps)
                commit_temporal = bool(temporal_commit and is_exit)
                commit_obs = bool(observation_commit and is_enter)
                update_opt = bool(update_optimizer_memory and is_exit)
                advance = bool(physical_time_advance and is_exit)
                steps.append(
                    Stage22StepPlan(
                        step_idx=int(step_idx),
                        block_id=int(pos),
                        episode_block_idx=int(pos),
                        rollout_block_rank=int(rank),
                        repeat_idx=int(repeat),
                        repeats_per_block=int(repeats_per_block),
                        is_block_enter=bool(is_enter),
                        is_block_exit=bool(is_exit),
                        source_keyframe_idx=int(keyframe_idx),
                        source_frame_idx=int(frame_idx),
                        evidence_refs=list(refs),
                        evidence_frame_indices=[int(frame_idx) for _ in refs],
                        evidence_cam_indices=[int(ref[1]) for ref in refs],
                        commit_observation_memory=bool(commit_obs),
                        update_optimizer_memory=bool(update_opt),
                        detach_before_step=False,
                        detach_after_step=False,
                        allow_step_render_loss=False,
                        step_loss_refs=[],
                        rollout_pos_code=float(rank) / float(max(len(positions) - 1, 1)),
                        frame_pos_code=float(pos) / 9.0,
                        repeat_pos_code=float(repeat) / float(max(int(repeats_per_block) - 1, 1)),
                        is_frame_exit=bool(is_exit),
                        episode_visit_idx=int(pos),
                        rollout_visit_idx=int(rank),
                        optimizer_step_idx_in_episode=int(episode_step_offset + step_idx),
                        record_update_norm=bool(update_opt),
                        commit_support_on_exit=bool(update_opt),
                        commit_residual_on_exit=bool(update_opt),
                        window_start=0,
                        window_end=9,
                        window_hash=0,
                        sequence_pos=int(pos),
                        visit_kind=str(visit_kind),
                        frame_gap=int(frame_gap),
                        temporal_read=bool(temporal_read),
                        temporal_commit=bool(commit_temporal),
                        physical_time_advance=bool(advance),
                        scheduler_phase=str(phase),
                        timestamp_us=int(timestamp_us),
                        timestamp_sec=float(timestamp_us) / 1.0e6,
                        delta_t_sec=float(delta_t_sec),
                        ego_delta_translation=(float(ego_delta[0]), float(ego_delta[1]), float(ego_delta[2])),
                        ego_delta_yaw=float(ego_delta_yaw),
                        visit_memory_mask=bool(temporal_read or commit_temporal),
                        repair_no_commit=bool(str(visit_kind) in {"repair", "stress"}),
                    )
                )
            if str(phase) != "repair":
                prev_pos = int(pos)
        return steps, _dedupe_refs_keep_order(evidence_flat)

    def _rollout_from_positions(
        self,
        *,
        frame_rows: np.ndarray,
        scene_id: int,
        segment_id: int,
        sequence_id: int,
        protocol: str,
        positions: Sequence[int],
        rollout_idx: int,
        rollouts_per_episode: int,
        phase: str,
        visit_kind: str,
        repeats_per_block: int,
        history_positions: Sequence[int],
        repair_positions: Sequence[int],
        repair_enabled: bool,
        repair_hash: int,
        episode_step_offset: int,
        previous_physical_pos: Optional[int] = None,
    ) -> RolloutPlan:
        is_repair = str(phase) == "repair"
        is_no_commit = str(phase) in {"bootstrap", "repair", "stress"}
        steps, evidence_refs = self._make_steps(
            frame_rows=frame_rows,
            positions=positions,
            repeats_per_block=int(repeats_per_block),
            visit_kind=str(visit_kind),
            phase=str(phase),
            episode_step_offset=int(episode_step_offset),
            temporal_read=bool(False if phase == "bootstrap" else True),
            temporal_commit=bool(not is_no_commit),
            observation_commit=bool(not is_no_commit),
            update_optimizer_memory=bool(not is_no_commit),
            physical_time_advance=bool(not is_no_commit),
            previous_physical_pos=previous_physical_pos,
        )
        current_frames = [int(frame_rows[int(pos)]["frame_idx"]) for pos in positions]
        current_refs = _dedupe_refs_keep_order(
            [
                ref
                for pos in positions
                for ref in _refs_for_frame(
                    int(frame_rows[int(pos)]["frame_idx"]),
                    _cams_from_mask(int(frame_rows[int(pos)]["available_camera_mask"]), int(self.index.num_cams)),
                )
            ]
        )
        current_ref_set = set(current_refs)
        history_frames = [int(frame_rows[int(pos)]["frame_idx"]) for pos in history_positions]
        history_refs = _dedupe_refs_keep_order(
            [
                ref
                for pos in history_positions
                for ref in _refs_for_frame(
                    int(frame_rows[int(pos)]["frame_idx"]),
                    _cams_from_mask(int(frame_rows[int(pos)]["available_camera_mask"]), int(self.index.num_cams)),
                )
                if ref not in current_ref_set
            ]
        )
        target_refs = list(current_refs) + list(history_refs)
        target_roles = [STAGE22_CURRENT_ROLE for _ in current_refs] + [STAGE22_HISTORY_ROLE for _ in history_refs]
        final = make_final_supervision(
            refs=target_refs,
            roles=target_roles,
            current_frames=current_frames,
            current_refs=current_refs,
            history_frames=history_frames,
            history_refs=history_refs,
        )
        frame_indices = [int(row["frame_idx"]) for row in frame_rows]
        timestamps_us = [int(row["timestamp_us"]) for row in frame_rows]
        window_hash = int(stable_uint64((scene_id, segment_id, protocol, sequence_id, *frame_indices)) & 0x7FFFFFFFFFFFFFFF)
        steps = [
            type(step)(**{**step.__dict__, "window_hash": int(window_hash)})
            for step in steps
        ]
        request_meta = {
            "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
            "model_family": IFORWARD_MODEL_FAMILY,
            "iforward_stage2_2": {
                "index_fingerprint": self.index.fingerprint,
                "protocol": str(protocol),
                "phase": str(phase),
                "timestamp_source": str(self.index.timestamp_source),
                "frame_period_us": int(self.index.frame_period_us),
                "raw_frame_ids": frame_indices,
                "timestamps_us": timestamps_us,
                "frame_gaps": [int(s.frame_gap) for s in steps if int(s.repeat_idx) == 0],
                "camera_masks": [int(row["available_camera_mask"]) for row in frame_rows],
            },
        }
        return RolloutPlan(
            scheduler_version=IFORWARD_STAGE2_2_SCHEDULER_VERSION,
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_id=int(self._episode_id_next),
            rollout_id_global=int(self._rollout_id_global + int(rollout_idx)),
            rollout_idx_in_episode=int(rollout_idx),
            episode_start_keyframe_pos=int(0),
            keyframe_window=[int(row["keyframe_idx"]) for row in frame_rows],
            frame_chain=frame_indices,
            num_cams=int(self.index.num_cams),
            shape_name=f"{str(phase)}_b{len(positions)}r{int(repeats_per_block)}",
            blocks_per_rollout=int(len(positions)),
            repeats_per_block=int(repeats_per_block),
            requested_blocks_per_rollout=int(len(positions)),
            actual_blocks_per_rollout=int(len(positions)),
            requested_inner_K=int(len(steps)),
            actual_inner_K=int(len(steps)),
            short_rollout=False,
            short_rollout_reason="",
            episode_block_indices=[int(x) for x in positions],
            input_keyframe_indices=[int(frame_rows[int(pos)]["keyframe_idx"]) for pos in positions],
            input_frame_indices=current_frames,
            delivery_frame_indices=current_frames,
            delivery_order_policy="stage2_2_stream10_rawframe",
            inner_K=int(len(steps)),
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
            leakage_check={
                "same_scene_segment_required": True,
                "forbid_test_refs_in_train": True,
                "repair_no_commit": bool(is_repair),
            },
            model_family=IFORWARD_MODEL_FAMILY,
            rollouts_per_episode=int(rollouts_per_episode),
            episode_num_blocks=10,
            window_policy=str(protocol),
            window_start=0,
            window_end=9,
            window_block_ids=list(range(10)),
            window_keyframe_indices=[int(row["keyframe_idx"]) for row in frame_rows],
            window_frame_indices=frame_indices,
            window_hash=int(window_hash),
            sequence_id=int(sequence_id),
            sequence_length=10,
            sequence_protocol=str(protocol),
            sequence_stride=1 if str(protocol) == "D1" else 2 if str(protocol) == "D2" else 0,
            sequence_start_local_frame=0,
            sequence_block_ids=list(range(10)),
            sequence_keyframe_indices=[int(row["keyframe_idx"]) for row in frame_rows],
            sequence_source_frame_indices=frame_indices,
            sequence_timestamps_us=timestamps_us,
            sequence_positions=[int(x) for x in positions],
            history_positions=[int(x) for x in history_positions],
            repair_positions=[int(x) for x in repair_positions],
            scheduler_phase=str(phase),
            rollout_phase=str(
                phase
                if phase in {"repair", "bootstrap", "stress"}
                else f"causal_{int(rollout_idx)}"
                if phase == "causal"
                else phase
            ),
            repair_enabled=bool(repair_enabled),
            repair_permutation_hash=int(repair_hash),
            temporal_read_count=sum(1 for s in steps if bool(s.temporal_read)),
            temporal_commit_count=sum(1 for s in steps if bool(s.temporal_commit)),
            observation_commit_count=sum(1 for s in steps if bool(s.commit_observation_memory)),
            optimizer_memory_update_count=sum(1 for s in steps if bool(s.update_optimizer_memory)),
        )

    def _sample_bootstrap_repeats(self) -> int:
        choices = list(_cfg_get(self.bootstrap_cfg, "repeat_choices", []) or [])
        if choices:
            clean = []
            total = 0.0
            for item in choices:
                repeats = int(_cfg_get(item, "repeats", 0))
                prob = float(_cfg_get(item, "prob", 0.0))
                if repeats > 0 and prob > 0.0:
                    clean.append((repeats, prob))
                    total += prob
            if clean and total > 0.0:
                draw = self.rng.random() * total
                acc = 0.0
                for repeats, prob in clean:
                    acc += prob
                    if draw <= acc:
                        return int(repeats)
                return int(clean[-1][0])
        return int(_cfg_get(self.bootstrap_cfg, "repeats", 8))

    def _build_bootstrap_rollout(self) -> RolloutPlan:
        raw = self.traversal.next_bootstrap_frame()
        seg = self.index.segments[int(raw["segment_row"])]
        frame_rows = self.index.frames_for_segment_row(int(raw["segment_row"]))[[int(raw["local_frame"])]]
        repeats = int(self._sample_bootstrap_repeats())
        return self._rollout_from_positions(
            frame_rows=frame_rows,
            scene_id=int(seg["scene_id"]),
            segment_id=int(seg["segment_id"]),
            sequence_id=int(_rng_token(self.rng)),
            protocol="bootstrap",
            positions=[0],
            rollout_idx=0,
            rollouts_per_episode=1,
            phase="bootstrap",
            visit_kind="bootstrap",
            repeats_per_block=int(repeats),
            history_positions=[],
            repair_positions=[],
            repair_enabled=False,
            repair_hash=-1,
            episode_step_offset=0,
        )

    def _build_episode(self) -> EpisodePlan:
        protocol = self.protocol_sampler.next(self.traversal.available_protocols())
        window = self.traversal.next_window(str(protocol))
        repair_prob = float(_cfg_get(self.repair_cfg, "prob", 0.5))
        repair_start = int(_cfg_get(self.repair_cfg, "start_step", 15000))
        repair_enabled = bool(int(self.global_step) >= int(repair_start) and self.rng.random() < float(repair_prob))
        return self.build_episode_for_window(window=window, protocol=str(protocol), repair_enabled=repair_enabled)

    def build_episode_for_window(self, *, window: np.void, protocol: str, repair_enabled: bool = False) -> EpisodePlan:
        segment_row, scene_id, segment_id = self._segment_ids_for_window(window)
        frame_rows = self._frame_rows_for_window(window, str(protocol))
        sequence_id = int(stable_uint64((scene_id, segment_id, protocol, int(window["start_local_frame"]), _rng_token(self.rng))) & 0x7FFFFFFFFFFFFFFF)
        repair_positions: Tuple[int, ...] = self._repair_permutation() if repair_enabled else ()
        repair_hash = (
            int(stable_uint64((scene_id, segment_id, sequence_id, *repair_positions, _rng_token(self.rng))) & 0x7FFFFFFFFFFFFFFF)
            if repair_enabled
            else -1
        )
        rollouts: List[RolloutPlan] = []
        total = 5 + (1 if repair_enabled else 0)
        step_offset = 0
        previous_physical_pos: Optional[int] = None
        for rollout_idx in range(5):
            positions = [int(rollout_idx * 2), int(rollout_idx * 2 + 1)]
            history_positions = list(range(int(rollout_idx * 2)))
            plan = self._rollout_from_positions(
                frame_rows=frame_rows,
                scene_id=scene_id,
                segment_id=segment_id,
                sequence_id=sequence_id,
                protocol=str(protocol),
                positions=positions,
                rollout_idx=int(rollout_idx),
                rollouts_per_episode=int(total),
                phase="causal",
                visit_kind="causal_first",
                repeats_per_block=4,
                history_positions=history_positions,
                repair_positions=repair_positions,
                repair_enabled=repair_enabled,
                repair_hash=repair_hash,
                episode_step_offset=step_offset,
                previous_physical_pos=previous_physical_pos,
            )
            rollouts.append(plan)
            step_offset += int(len(plan.steps))
            previous_physical_pos = int(positions[-1])
        if repair_enabled:
            plan = self._rollout_from_positions(
                frame_rows=frame_rows,
                scene_id=scene_id,
                segment_id=segment_id,
                sequence_id=sequence_id,
                protocol=str(protocol),
                positions=repair_positions,
                rollout_idx=5,
                rollouts_per_episode=int(total),
                phase="repair",
                visit_kind="repair",
                repeats_per_block=int(_cfg_get(self.repair_cfg, "repeats_per_block", 1)),
                history_positions=[],
                repair_positions=repair_positions,
                repair_enabled=True,
                repair_hash=repair_hash,
                episode_step_offset=step_offset,
                previous_physical_pos=None,
            )
            rollouts.append(plan)
        return EpisodePlan(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_id=int(self._episode_id_next),
            protocol=str(protocol),
            sequence_id=int(sequence_id),
            source_frame_indices=tuple(int(row["frame_idx"]) for row in frame_rows),
            timestamps_us=tuple(int(row["timestamp_us"]) for row in frame_rows),
            rollouts=tuple(rollouts),
            repair_enabled=bool(repair_enabled),
            metadata={"segment_row": int(segment_row), "window_start": int(window["start_local_frame"])},
        )

    def _batch_from_plan(self, plan: RolloutPlan) -> Dict[str, Any]:
        assembler = getattr(self.dataset, "_assemble_segment_batch_from_iforward_stage2_2_request", None)
        if not callable(assembler):
            assembler = getattr(self.dataset, "_assemble_segment_batch_from_iforward_request", None)
        if not callable(assembler):
            raise ValueError("Stage2_2 requires dataset._assemble_segment_batch_from_iforward_stage2_2_request")
        return assembler(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            plan=plan,
            include_test=bool(self.include_test),
        )

    def _update_last_info(self, plan: RolloutPlan) -> None:
        self._last_info = {
            "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
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
            "blocks_per_rollout": int(plan.blocks_per_rollout),
            "actual_blocks_per_rollout": int(plan.actual_blocks_per_rollout),
            "repeats_per_block": int(plan.repeats_per_block),
            "sequence_length": int(plan.sequence_length),
            "sequence_id": int(plan.sequence_id),
            "sequence_protocol": str(plan.sequence_protocol),
            "sequence_positions": [int(x) for x in plan.sequence_positions],
            "sequence_source_frame_indices": [int(x) for x in plan.sequence_source_frame_indices],
            "sequence_timestamps_us": [int(x) for x in plan.sequence_timestamps_us],
            "iforward/stage2_2/protocol": str(plan.sequence_protocol),
            "iforward/stage2_2/phase": str(plan.scheduler_phase),
            "iforward/stage2_2/raw_frame_ids": [int(x) for x in plan.sequence_source_frame_indices],
            "iforward/stage2_2/timestamps_us": [int(x) for x in plan.sequence_timestamps_us],
            "iforward/stage2_2/frame_gaps": [int(s.frame_gap) for s in plan.steps if int(s.repeat_idx) == 0],
            "iforward/stage2_2/keyframe_ids": [int(x) for x in plan.sequence_keyframe_indices],
            "iforward/stage2_2/index_fingerprint": self.index.fingerprint,
            "history_positions": [int(x) for x in plan.history_positions],
            "repair_positions": [int(x) for x in plan.repair_positions],
            "repair_enabled": bool(plan.repair_enabled),
            "repair_permutation_hash": int(plan.repair_permutation_hash),
            "inner_K": int(plan.inner_K),
            "temporal_read_count": int(plan.temporal_read_count),
            "temporal_commit_count": int(plan.temporal_commit_count),
            "history_ref_count": int(plan.final_supervision.history_ref_count),
            "index_fingerprint": self.index.fingerprint,
        }

    def _emit_preload_hints(self, plan: RolloutPlan) -> None:
        if not bool(_cfg_get(self.preload_cfg, "emit_hints", False)):
            return
        build_hint = getattr(self.dataset, "build_preload_hint_light", None) or getattr(self.dataset, "build_preload_hint", None)
        submit = getattr(self.dataset, "submit_preload_hint", None)
        if not callable(build_hint) or not callable(submit):
            return
        scopes: List[Tuple[str, List[ImageRef]]] = []
        if bool(_cfg_get(self.preload_cfg, "warm_current_rollout_refs", True)):
            scopes.append(("stage2_2_current_rollout", list(plan.evidence_refs_flat) + list(plan.target_refs_flat)))
        if self._episode_plan is not None and bool(_cfg_get(self.preload_cfg, "warm_next_rollout_refs", True)):
            next_idx = int(self._episode_plan_cursor) + 1
            if next_idx < len(self._episode_plan.rollouts):
                nxt = self._episode_plan.rollouts[next_idx]
                scopes.append(("next_block_exact", list(nxt.evidence_refs_flat) + list(nxt.target_refs_flat)))
        if self._episode_plan is not None and bool(_cfg_get(self.preload_cfg, "warm_episode_chain", True)):
            refs: List[ImageRef] = []
            for nxt in self._episode_plan.rollouts[int(self._episode_plan_cursor) + 1 :]:
                refs.extend(list(nxt.evidence_refs_flat))
                refs.extend(list(nxt.target_refs_flat))
            scopes.append(("episode_chain_exact", refs))
        for scope, refs in scopes:
            refs = _dedupe_refs_keep_order(refs)
            if not refs:
                continue
            hint = build_hint(scene_id=int(plan.scene_id), segment_id=int(plan.segment_id), future_image_refs=refs, scope=scope)
            submit(
                hint=hint,
                hint_scope=scope,
                epoch_idx=int(self.epoch_idx),
                global_step=int(self.global_step),
                block_idx_global=int(plan.rollout_id_global),
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

    def next_batch(self) -> Dict[str, Any]:
        bootstrap_end = int(_cfg_get(self.bootstrap_cfg, "end_step", 5000))
        if int(self.global_step) < int(bootstrap_end):
            plan = self._build_bootstrap_rollout()
            episode_end = True
        else:
            if self._episode_plan is None or int(self._episode_plan_cursor) >= len(self._episode_plan.rollouts):
                self._episode_plan = self.producer.get_or_build(self._build_episode)
                self._episode_plan_cursor = 0
            plan = self._episode_plan.rollouts[int(self._episode_plan_cursor)]
            episode_end = bool(int(self._episode_plan_cursor) == len(self._episode_plan.rollouts) - 1)
        batch = self._batch_from_plan(plan)
        self._emit_preload_hints(plan)
        self._update_last_info(plan)
        batch["_scheduler_v4_aligned_info"] = dict(self._last_info)
        self._emit({**dict(self._last_info), "type": "iforward_stage2_2_scheduler"})
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
        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_class": type(self).__name__,
            "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
            "index_fingerprint": self.index.fingerprint,
            "global_step": int(self.global_step),
            "epoch_idx": int(self.epoch_idx),
            "episode_id_next": int(self._episode_id_next),
            "rollout_id_global": int(self._rollout_id_global),
            "episode_plan": copy.deepcopy(self._episode_plan),
            "episode_plan_cursor": int(self._episode_plan_cursor),
            "pending_events": copy.deepcopy(self._pending_events),
            "last_info": copy.deepcopy(self._last_info),
            "rng_state": copy.deepcopy(self.rng.getstate()),
            "protocol_sampler": self.protocol_sampler.state_dict(),
            "traversal": self.traversal.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if str(state.get("scheduler_version", "")) != IFORWARD_STAGE2_2_SCHEDULER_VERSION:
            raise ValueError(f"expected scheduler_version={IFORWARD_STAGE2_2_SCHEDULER_VERSION}")
        if str(state.get("index_fingerprint", "")) and str(state.get("index_fingerprint", "")) != self.index.fingerprint:
            raise ValueError("Stage2_2 scheduler resume index fingerprint mismatch")
        self.global_step = int(state.get("global_step", 0))
        self.epoch_idx = int(state.get("epoch_idx", 0))
        self._episode_id_next = int(state.get("episode_id_next", 0))
        self._rollout_id_global = int(state.get("rollout_id_global", 0))
        self._episode_plan = copy.deepcopy(state.get("episode_plan", None))
        self._episode_plan_cursor = int(state.get("episode_plan_cursor", 0))
        self._pending_events = copy.deepcopy(list(state.get("pending_events", []) or []))
        self._last_info = copy.deepcopy(dict(state.get("last_info", {}) or {}))
        if state.get("rng_state", None) is not None:
            self.rng.setstate(state["rng_state"])
        self.protocol_sampler.load_state_dict(dict(state.get("protocol_sampler", {}) or {}))
        self.traversal.load_state_dict(dict(state.get("traversal", {}) or {}))


__all__ = ["IFORWARD_STAGE2_2_SCHEDULER_VERSION", "Stage22Scheduler"]
