from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


class SegmentIndexLike(Protocol):
    scene_id: int
    segment_id: int
    num_cams: int
    keyframe_indices: List[int]
    keyframe_to_frames: Dict[int, List[int]]
    test_frame_indices: List[int]


class TrainSchedulerDatasetV6(Protocol):
    _initialized: bool

    def initialize(self) -> None: ...
    def list_training_scene_ids(self) -> List[int]: ...
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexLike: ...
    def get_segment_batch_from_image_refs(
        self, request: Any, *, enforce_target0_equals_source: bool = True
    ) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class _BatchRequestV6Compat:
    scene_id: int
    segment_id: int
    source_image_ref: Tuple[int, int]
    target_image_refs: List[Tuple[int, int]]
    include_test: bool = False
    test_image_refs: Optional[List[Tuple[int, int]]] = None


class TrainSchedulerV6:
    """
    V6 phase-1 behavior:
    - no-overlap scoring
    - segment-local block mode (aligned with V5 state machine)
    - builds batches via the dataset v4 image-ref API

    Sampling is still **frame-level** (like V5): ``total_target_frames`` counts training **frames**
    (timesteps), not individual camera images. Each sampled frame is then expanded to **all
    cameras** via ``_frame_targets_to_image_refs``, so the number of target image refs per block is
    ``len(target_frame_indices) * num_cams`` (plus the analogous all-cam treatment implicit in the
    dataset batch for source/target packs).
    """

    def __init__(
        self,
        *,
        dataset: TrainSchedulerDatasetV6,
        state_write_interval_steps: int,
        updates_per_block: int,
        keyframes_per_episode: int,
        episodes_per_segment: int,
        total_target_frames: int,
        include_source_frame: bool,
        neighbor_ring: int,
        prefer_nearby_keyframes: bool,
        fallback_expand_to_segment: bool,
        with_replacement: bool,
        include_test: bool,
        fixed_scene_id: Optional[int],
        fixed_segment_id: Optional[int],
    ) -> None:
        if state_write_interval_steps < 1:
            raise ValueError("state_write_interval_steps must be >= 1")
        if updates_per_block < 1:
            raise ValueError("updates_per_block must be >= 1")
        if keyframes_per_episode < 1:
            raise ValueError("keyframes_per_episode must be >= 1")
        if episodes_per_segment < 1:
            raise ValueError("episodes_per_segment must be >= 1")
        if total_target_frames < 1:
            raise ValueError("total_target_frames must be >= 1")
        if neighbor_ring < 1:
            raise ValueError("neighbor_ring must be >= 1")
        self.dataset = dataset
        self.U = int(state_write_interval_steps)
        self.updates_per_block = int(updates_per_block)
        self.keyframes_per_episode = int(keyframes_per_episode)
        self.episodes_per_segment = int(episodes_per_segment)
        self.total_target_frames = int(total_target_frames)
        self.include_source_frame = bool(include_source_frame)
        self.neighbor_ring = int(neighbor_ring)
        self.prefer_nearby_keyframes = bool(prefer_nearby_keyframes)
        self.fallback_expand_to_segment = bool(fallback_expand_to_segment)
        self.with_replacement = bool(with_replacement)
        self.include_test = bool(include_test)
        self.fixed_scene_id = int(fixed_scene_id) if fixed_scene_id is not None else None
        self.fixed_segment_id = int(fixed_segment_id) if fixed_segment_id is not None else None

        self.epoch_idx = 0
        self.global_step = 0
        self.epoch_plan: List[Dict[str, Any]] = []
        self.plan_cursor = 0
        self.current_segment_state: Optional[Dict[str, Any]] = None
        self._pending_events: List[Dict[str, Any]] = []
        self._block_idx_global = 0
        self._reset_episode_idx = 0

        if not self.dataset._initialized:
            self.dataset.initialize()
        self.start_new_epoch()

    def pop_events(self) -> List[Dict[str, Any]]:
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def _emit(self, event: Dict[str, Any]) -> None:
        self._pending_events.append(event)

    def _init_epoch_segment_pair_iterator(self) -> None:
        if self.fixed_scene_id is not None:
            self._epoch_scene_queue = [int(self.fixed_scene_id)]
        else:
            q = list(self.dataset.list_training_scene_ids())
            if len(q) == 0:
                raise ValueError("No valid training scenes in dataset registry list")
            random.shuffle(q)
            self._epoch_scene_queue = q
        self._epoch_scene_q_idx = 0
        self._epoch_current_scene_id = None
        self._epoch_segment_ids = []
        self._epoch_segment_pos = 0

    def _next_scene_segment_pair(self) -> Optional[Tuple[int, int]]:
        while True:
            if self._epoch_segment_pos >= len(self._epoch_segment_ids):
                if self._epoch_scene_q_idx >= len(self._epoch_scene_queue):
                    return None
                sid = int(self._epoch_scene_queue[self._epoch_scene_q_idx])
                self._epoch_scene_q_idx += 1
                segment_ids = [int(x) for x in self.dataset.list_segment_ids(sid)]
                nseg = len(segment_ids)
                if self.fixed_segment_id is not None:
                    if self.fixed_segment_id < 0 or self.fixed_segment_id >= nseg:
                        raise ValueError(f"fixed_segment_id={self.fixed_segment_id} out of range in scene={sid}")
                    self._epoch_segment_ids = [int(segment_ids[int(self.fixed_segment_id)])]
                else:
                    self._epoch_segment_ids = list(segment_ids)
                    random.shuffle(self._epoch_segment_ids)
                self._epoch_segment_pos = 0
                self._epoch_current_scene_id = sid
            seg_id = int(self._epoch_segment_ids[self._epoch_segment_pos])
            self._epoch_segment_pos += 1
            assert self._epoch_current_scene_id is not None
            return (int(self._epoch_current_scene_id), seg_id)

    def _ensure_epoch_plan_index(self, idx: int) -> None:
        while len(self.epoch_plan) <= idx:
            p = self._next_scene_segment_pair()
            if p is None:
                break
            self.epoch_plan.append({"scene_id": int(p[0]), "segment_id": int(p[1])})

    def _hydrate_plan_item_budget(self, idx: int) -> None:
        self._ensure_epoch_plan_index(idx)
        if idx >= len(self.epoch_plan):
            return
        it = self.epoch_plan[idx]
        if "segment_budget_u" in it:
            return
        sidx = self.dataset.get_segment_index(int(it["scene_id"]), int(it["segment_id"]))
        num_keyframes = len(sidx.keyframe_indices)
        w_eff = int(min(self.keyframes_per_episode, num_keyframes))
        b_seg = int(self.episodes_per_segment * w_eff)
        segment_budget_u = int(b_seg * self.updates_per_block)
        it["num_keyframes"] = int(num_keyframes)
        it["num_cams"] = int(sidx.num_cams)
        it["w_eff"] = w_eff
        it["b_seg"] = b_seg
        it["segment_budget_u"] = segment_budget_u
        it["segment_step_budget"] = int(segment_budget_u * self.U)
        it["U"] = int(self.U)

    def build_epoch_plan(self) -> None:
        self._init_epoch_segment_pair_iterator()
        self.epoch_plan = []
        self.plan_cursor = 0

    def start_new_epoch(self) -> None:
        self.epoch_idx += 1
        self.build_epoch_plan()
        self.current_segment_state = None
        if hasattr(self.dataset, "clear_preload_scheduler_scope"):
            self.dataset.clear_preload_scheduler_scope()

    def _sample_contiguous_window(self, sidx: SegmentIndexLike) -> List[int]:
        seg_kfs = list(sidx.keyframe_indices)
        if len(seg_kfs) > self.keyframes_per_episode:
            start = random.randint(0, len(seg_kfs) - self.keyframes_per_episode)
            return list(seg_kfs[start : start + self.keyframes_per_episode])
        return list(seg_kfs)

    @staticmethod
    def _kf_positions(sidx: SegmentIndexLike) -> Dict[int, int]:
        return {int(k): i for i, k in enumerate(sidx.keyframe_indices)}

    def _neighbor_keyframe_pool(
        self,
        sidx: SegmentIndexLike,
        source_kf: int,
        episode_window: List[int],
    ) -> List[int]:
        pos = self._kf_positions(sidx)
        src_pos = int(pos[int(source_kf)])
        all_kf = [int(k) for k in sidx.keyframe_indices if int(k) != int(source_kf)]
        by_ring = [k for k in all_kf if abs(int(pos[k]) - src_pos) <= self.neighbor_ring]
        if self.prefer_nearby_keyframes:
            by_ring.sort(key=lambda k: abs(int(pos[k]) - src_pos))
        window_others = [int(k) for k in episode_window if int(k) != int(source_kf)]
        if self.prefer_nearby_keyframes:
            window_others.sort(key=lambda k: abs(int(pos[k]) - src_pos))
        out: List[int] = []
        seen = set()
        for k in window_others + by_ring:
            if k not in seen:
                seen.add(k)
                out.append(k)
        if self.fallback_expand_to_segment:
            for k in all_kf:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return out

    def _sample_target_frame_indices(
        self,
        sidx: SegmentIndexLike,
        source_frame_idx: int,
        source_keyframe_idx: int,
        episode_window: List[int],
    ) -> List[int]:
        refs: List[int] = []
        if self.include_source_frame:
            refs.append(int(source_frame_idx))
        extra_needed = self.total_target_frames - len(refs)
        if extra_needed <= 0:
            return refs
        pool_kf = self._neighbor_keyframe_pool(sidx, source_keyframe_idx, episode_window)
        if len(pool_kf) == 0:
            raise ValueError("TrainSchedulerV6: no candidate keyframes for extra targets")
        if len(pool_kf) >= extra_needed:
            chosen_kf = (
                [int(random.choice(pool_kf)) for _ in range(extra_needed)]
                if self.with_replacement
                else [int(x) for x in random.sample(pool_kf, extra_needed)]
            )
        else:
            if not self.with_replacement:
                raise ValueError("TrainSchedulerV6: not enough distinct keyframes for requested extras")
            chosen_kf = [int(random.choice(pool_kf)) for _ in range(extra_needed)]
        for kf in chosen_kf:
            refs.append(int(random.choice(sidx.keyframe_to_frames[int(kf)])))
        return refs

    @staticmethod
    def _pseudo_image_ref(frame_idx: int) -> Tuple[int, int]:
        return (int(frame_idx), -1)

    def _frame_targets_to_image_refs(self, sidx: SegmentIndexLike, frame_indices: List[int]) -> List[Tuple[int, int]]:
        refs: List[Tuple[int, int]] = []
        for frame_idx in frame_indices:
            for cam_idx in range(int(sidx.num_cams)):
                refs.append((int(frame_idx), int(cam_idx)))
        return refs

    def _start_episode(self) -> None:
        if self.current_segment_state is None:
            raise ValueError("TrainSchedulerV6 internal state is not initialized")
        st = self.current_segment_state
        if int(st["episodes_started"]) >= self.episodes_per_segment:
            raise ValueError("TrainSchedulerV6: _start_episode called when episode quota is exhausted")
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        window = self._sample_contiguous_window(sidx)
        pair_list = [int(kf) for kf in window]
        random.shuffle(pair_list)
        st["episodes_started"] = int(st["episodes_started"]) + 1
        st["episode_window_keyframes"] = list(window)
        st["pair_list"] = list(pair_list)
        st["pair_cursor"] = 0
        st["reset_episode_idx"] = int(self._reset_episode_idx)
        self._emit(
            {
                "type": "reset_event",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(st["scene_id"]),
                "segment_id": int(st["segment_id"]),
                "reset_episode_idx": int(st["reset_episode_idx"]),
                "reason": "episode_begin",
                "window_keyframes": list(window),
                "num_pairs": int(len(pair_list)),
                "scheduler_version": "v6",
            }
        )
        self._reset_episode_idx += 1

    def _emit_segment_begin(self, st: Dict[str, Any]) -> None:
        self._emit(
            {
                "type": "segment_begin",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(st["scene_id"]),
                "segment_id": int(st["segment_id"]),
                "U": int(self.U),
                "num_keyframes": int(st["num_keyframes"]),
                "num_cams": int(st["num_cams"]),
                "w_eff": int(st["w_eff"]),
                "b_seg": int(st["b_seg"]),
                "segment_budget_u": int(st["segment_budget_u"]),
                "segment_step_budget": int(st["segment_step_budget"]),
                "updates_per_block": int(self.updates_per_block),
                "scheduler_version": "v6",
            }
        )

    def _start_block(self) -> bool:
        if self.current_segment_state is None:
            raise ValueError("TrainSchedulerV6 internal state is not initialized")
        st = self.current_segment_state
        scene_id = int(st["scene_id"])
        segment_id = int(st["segment_id"])
        sidx = self.dataset.get_segment_index(scene_id, segment_id)

        if int(st["segment_local_u"]) >= int(st["segment_budget_u"]):
            return False
        while int(st["pair_cursor"]) >= len(st["pair_list"]):
            self._emit(
                {
                    "type": "episode_end",
                    "epoch_idx": int(self.epoch_idx),
                    "global_step": int(self.global_step),
                    "scene_id": scene_id,
                    "segment_id": segment_id,
                    "reset_episode_idx": int(st["reset_episode_idx"]),
                    "reason": "pair_list_exhausted",
                    "scheduler_version": "v6",
                }
            )
            if int(st["episodes_started"]) >= self.episodes_per_segment:
                return False
            self._start_episode()
            st = self.current_segment_state
            if st is None:
                raise ValueError("TrainSchedulerV6 internal state became None")
        source_kf = int(st["pair_list"][st["pair_cursor"]])
        st["pair_cursor"] = int(st["pair_cursor"]) + 1
        source_frame_idx = int(random.choice(sidx.keyframe_to_frames[int(source_kf)]))
        target_frame_indices = self._sample_target_frame_indices(
            sidx=sidx,
            source_frame_idx=source_frame_idx,
            source_keyframe_idx=source_kf,
            episode_window=list(st["episode_window_keyframes"]),
        )
        remaining_u = int(st["segment_budget_u"]) - int(st["segment_local_u"])
        effective_u = int(min(self.updates_per_block, max(remaining_u, 0)))
        if effective_u <= 0:
            return False
        st["u_in_block"] = 0
        st["effective_u_this_block"] = int(effective_u)
        st["source_keyframe_idx"] = int(source_kf)
        st["source_frame_idx"] = int(source_frame_idx)
        st["target_frame_indices"] = [int(x) for x in target_frame_indices]
        st["source_image_ref"] = (int(source_frame_idx), 0)
        st["target_image_refs"] = self._frame_targets_to_image_refs(sidx, target_frame_indices)
        st["block_idx_global"] = int(self._block_idx_global)
        self._emit(
            {
                "type": "block_begin",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": scene_id,
                "segment_id": segment_id,
                "reset_episode_idx": int(st["reset_episode_idx"]),
                "block_idx_in_segment": int(st["block_idx_in_segment"]),
                "block_idx_global": int(st["block_idx_global"]),
                "U": int(self.U),
                "K_u_nominal": int(self.updates_per_block),
                "K_u_effective": int(effective_u),
                "K_steps_effective": int(effective_u * self.U),
                "source_keyframe_idx": int(source_kf),
                "source_frame_idx": int(source_frame_idx),
                "source_image_ref": tuple(st["source_image_ref"]),
                "target_frame_indices": [int(x) for x in target_frame_indices],
                "target_image_refs": [tuple(x) for x in st["target_image_refs"]],
                "num_sampled_target_frames": int(len(target_frame_indices)),
                "num_cams": int(sidx.num_cams),
                "num_target_image_refs": int(len(st["target_image_refs"])),
                "total_target_frames_setting": int(self.total_target_frames),
                "scheduler_version": "v6",
            }
        )
        self._block_idx_global += 1
        st["block_idx_in_segment"] = int(st["block_idx_in_segment"]) + 1
        return True

    def _end_segment(self) -> None:
        if self.current_segment_state is None:
            raise ValueError("TrainSchedulerV6 internal state is not initialized")
        st = self.current_segment_state
        self._emit(
            {
                "type": "segment_end",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(st["scene_id"]),
                "segment_id": int(st["segment_id"]),
                "segment_local_u": int(st["segment_local_u"]),
                "segment_budget_u": int(st["segment_budget_u"]),
                "segment_local_step": int(st["segment_local_step"]),
                "segment_step_budget": int(st["segment_step_budget"]),
                "source_frame_idx": int(st.get("source_frame_idx", -1)),
                "source_image_ref": tuple(st.get("source_image_ref", (-1, -1))),
                "target_frame_indices": [int(x) for x in st.get("target_frame_indices", [])],
                "target_image_refs": [tuple(x) for x in st.get("target_image_refs", [])],
                "scheduler_version": "v6",
            }
        )
        self.current_segment_state = None
        self.plan_cursor += 1
        if hasattr(self.dataset, "clear_preload_scheduler_scope"):
            self.dataset.clear_preload_scheduler_scope()

    def _enter_segment(self) -> None:
        self._hydrate_plan_item_budget(self.plan_cursor)
        if self.plan_cursor >= len(self.epoch_plan):
            raise ValueError("TrainSchedulerV6: failed to hydrate epoch plan")
        item = self.epoch_plan[self.plan_cursor]
        sidx = self.dataset.get_segment_index(int(item["scene_id"]), int(item["segment_id"]))
        st = {
            "scene_id": int(item["scene_id"]),
            "segment_id": int(item["segment_id"]),
            "num_keyframes": int(item["num_keyframes"]),
            "num_cams": int(item["num_cams"]),
            "w_eff": int(item["w_eff"]),
            "b_seg": int(item["b_seg"]),
            "segment_budget_u": int(item["segment_budget_u"]),
            "segment_step_budget": int(item["segment_step_budget"]),
            "segment_local_u": 0,
            "segment_local_step": 0,
            "block_idx_in_segment": 0,
            "block_idx_global": int(self._block_idx_global),
            "u_in_block": 0,
            "effective_u_this_block": int(self.updates_per_block),
            "source_keyframe_idx": -1,
            "source_frame_idx": -1,
            "target_frame_indices": [],
            "source_image_ref": (-1, -1),
            "target_image_refs": [],
            "test_frame_indices": [int(x) for x in sidx.test_frame_indices] if self.include_test else None,
            "episode_window_keyframes": [],
            "pair_list": [],
            "pair_cursor": 0,
            "episodes_started": 0,
            "reset_episode_idx": -1,
        }
        if hasattr(self.dataset, "set_preload_active_scope"):
            self.dataset.set_preload_active_scope(int(item["scene_id"]), int(item["segment_id"]))
        if hasattr(self.dataset, "set_preload_training_scope"):
            self.dataset.set_preload_training_scope(int(item["scene_id"]), int(item["segment_id"]))
        self.current_segment_state = st
        self._emit_segment_begin(st)
        self._start_episode()
        if not self._start_block():
            raise ValueError("TrainSchedulerV6: could not start first block in segment")

    def _batch_from_state(self, st: Dict[str, Any]) -> Dict[str, Any]:
        source_image_ref = tuple(st["source_image_ref"])
        target_image_refs = [tuple(x) for x in st["target_image_refs"]]
        req = _BatchRequestV6Compat(
            scene_id=int(st["scene_id"]),
            segment_id=int(st["segment_id"]),
            source_image_ref=(int(source_image_ref[0]), int(source_image_ref[1])),
            target_image_refs=[(int(x[0]), int(x[1])) for x in target_image_refs],
            include_test=bool(self.include_test),
            test_image_refs=None,
        )
        return self.dataset.get_segment_batch_from_image_refs(
            req,
            enforce_target0_equals_source=bool(self.include_source_frame),
        )

    def _aligned_info(self, st: Dict[str, Any]) -> Dict[str, Any]:
        k_u_nominal = int(self.updates_per_block)
        k_u_effective = int(st.get("effective_u_this_block", k_u_nominal))
        k_steps_effective = int(k_u_effective * self.U)
        return {
            "epoch_idx": int(self.epoch_idx),
            "global_step": int(self.global_step),
            "scene_id": int(st["scene_id"]),
            "segment_id": int(st["segment_id"]),
            "segment_local_step": int(st["segment_local_step"]),
            "segment_step_budget": int(st["segment_step_budget"]),
            "segment_local_u": int(st["segment_local_u"]),
            "segment_budget_u": int(st["segment_budget_u"]),
            "block_idx_in_segment": int(st.get("block_idx_in_segment", -1)),
            "block_idx_global": int(st.get("block_idx_global", 0)),
            "source_frame_idx": int(st.get("source_frame_idx", -1)),
            "source_keyframe_idx": int(st.get("source_keyframe_idx", -1)),
            "source_cam_idx": int(st.get("source_image_ref", (-1, -1))[1]),
            "source_image_ref": tuple(st.get("source_image_ref", (-1, -1))),
            "target_frame_indices": [int(x) for x in st.get("target_frame_indices", [])],
            "target_image_refs": [tuple(x) for x in st.get("target_image_refs", [])],
            "U": int(self.U),
            "K_u_nominal": k_u_nominal,
            "K_u_effective": k_u_effective,
            "K_steps_effective": int(k_steps_effective),
            "K_steps": int(k_steps_effective),
            "R_steps": 0,
            "T_steps": int(k_steps_effective),
            "scheduler_version": "v6",
        }

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        self._ensure_epoch_plan_index(self.plan_cursor)
        if self.plan_cursor >= len(self.epoch_plan):
            self.start_new_epoch()
            self._ensure_epoch_plan_index(self.plan_cursor)
        if self.plan_cursor >= len(self.epoch_plan):
            raise ValueError("TrainSchedulerV6: epoch has no (scene, segment) pairs")
        if self.current_segment_state is None:
            self._enter_segment()
        st = self.current_segment_state
        if st is None:
            raise ValueError("TrainSchedulerV6 internal state is not initialized")
        batch = self._batch_from_state(st)
        batch["_scheduler_v4_aligned_info"] = self._aligned_info(st)
        batch["_scheduler_v5_aligned_info"] = dict(batch["_scheduler_v4_aligned_info"])
        batch["_scheduler_v6_aligned_info"] = dict(batch["_scheduler_v4_aligned_info"])
        batch["_scheduler_v6_peek"] = True
        return batch

    def next_batch(self) -> Dict[str, Any]:
        self._ensure_epoch_plan_index(self.plan_cursor)
        if self.plan_cursor >= len(self.epoch_plan):
            self.start_new_epoch()
            self._ensure_epoch_plan_index(self.plan_cursor)
        if self.plan_cursor >= len(self.epoch_plan):
            raise ValueError("TrainSchedulerV6: epoch has no (scene, segment) pairs")
        if self.current_segment_state is None:
            self._enter_segment()
        st = self.current_segment_state
        if st is None:
            raise ValueError("TrainSchedulerV6 internal state is not initialized")
        batch = self._batch_from_state(st)
        st["segment_local_step"] = int(st["segment_local_step"]) + 1
        self.global_step += 1
        batch["_scheduler_v4_aligned_info"] = self._aligned_info(st)
        batch["_scheduler_v5_aligned_info"] = dict(batch["_scheduler_v4_aligned_info"])
        batch["_scheduler_v6_aligned_info"] = dict(batch["_scheduler_v4_aligned_info"])
        if int(st["segment_local_step"]) % self.U == 0:
            st["segment_local_u"] = int(st["segment_local_u"]) + 1
            st["u_in_block"] = int(st["u_in_block"]) + 1
            if int(st["u_in_block"]) >= int(st["effective_u_this_block"]):
                eff_u_end = int(st["effective_u_this_block"])
                self._emit(
                    {
                        "type": "block_end",
                        "epoch_idx": int(self.epoch_idx),
                        "global_step": int(self.global_step),
                        "scene_id": int(st["scene_id"]),
                        "segment_id": int(st["segment_id"]),
                        "block_idx_in_segment": int(st["block_idx_in_segment"]),
                        "block_idx_global": int(st.get("block_idx_global", 0)),
                        "source_frame_idx": int(st.get("source_frame_idx", -1)),
                        "source_image_ref": tuple(st.get("source_image_ref", (-1, -1))),
                        "target_frame_indices": [int(x) for x in st.get("target_frame_indices", [])],
                        "target_image_refs": [tuple(x) for x in st.get("target_image_refs", [])],
                        "num_updates_in_block": eff_u_end,
                        "K_u_nominal": int(self.updates_per_block),
                        "K_u_effective": eff_u_end,
                        "K_steps_effective": int(eff_u_end * self.U),
                        "U": int(self.U),
                        "scheduler_version": "v6",
                    }
                )
                done_seg = int(st["segment_local_u"]) >= int(st["segment_budget_u"])
                if done_seg:
                    self._end_segment()
                else:
                    if not self._start_block():
                        self._end_segment()
        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
        return batch

    def get_current_info(self) -> Dict[str, Any]:
        st = self.current_segment_state
        if st is None:
            self._ensure_epoch_plan_index(self.plan_cursor)
        if st is None and self.plan_cursor < len(self.epoch_plan):
            self._hydrate_plan_item_budget(self.plan_cursor)
            item = self.epoch_plan[self.plan_cursor]
            return {
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(item["scene_id"]),
                "segment_id": int(item["segment_id"]),
                "segment_local_step": 0,
                "segment_step_budget": int(item["segment_step_budget"]),
                "segment_local_u": 0,
                "segment_budget_u": int(item["segment_budget_u"]),
                "block_idx_in_segment": 0,
                "block_idx_global": int(self._block_idx_global),
                "source_frame_idx": -1,
                "source_keyframe_idx": -1,
                "source_cam_idx": -1,
                "source_image_ref": (-1, -1),
                "target_frame_indices": [],
                "target_image_refs": [],
                "U": int(self.U),
                "K_u_nominal": int(self.updates_per_block),
                "K_u_effective": int(self.updates_per_block),
                "K_steps_effective": int(self.updates_per_block * self.U),
                "K_steps": int(self.updates_per_block * self.U),
                "R_steps": 0,
                "T_steps": int(self.updates_per_block * self.U),
                "scheduler_version": "v6",
            }
        if st is None:
            return {
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": -1,
                "segment_id": -1,
                "segment_local_step": 0,
                "segment_step_budget": 0,
                "segment_local_u": 0,
                "segment_budget_u": 0,
                "block_idx_in_segment": 0,
                "block_idx_global": int(self._block_idx_global),
                "source_frame_idx": -1,
                "source_keyframe_idx": -1,
                "source_cam_idx": -1,
                "source_image_ref": (-1, -1),
                "target_frame_indices": [],
                "target_image_refs": [],
                "U": int(self.U),
                "K_u_nominal": 0,
                "K_u_effective": 0,
                "K_steps_effective": 0,
                "K_steps": 0,
                "R_steps": 0,
                "T_steps": 0,
                "scheduler_version": "v6",
            }
        return self._aligned_info(st)

