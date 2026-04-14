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


class TrainSchedulerDatasetV7(Protocol):
    _initialized: bool

    def initialize(self) -> None: ...
    def list_training_scene_ids(self) -> List[int]: ...
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexLike: ...
    def get_segment_batch_from_image_refs(
        self, request: Any, *, enforce_target0_equals_source: bool = True
    ) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class _BatchRequestV7Compat:
    scene_id: int
    segment_id: int
    source_image_ref: Tuple[int, int]
    target_image_refs: List[Tuple[int, int]]
    source_image_refs: Optional[List[Tuple[int, int]]] = None
    include_test: bool = False
    test_image_refs: Optional[List[Tuple[int, int]]] = None


@dataclass(frozen=True)
class EpisodePlanV7:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int
    keyframe_window: List[int]
    frame_chain: List[int]
    block_windows: List[List[int]]
    num_cams: int


class TrainSchedulerV7:
    def __init__(
        self,
        *,
        dataset: TrainSchedulerDatasetV7,
        steps_per_block: int,
        blocks_per_episode: int,
        total_target_frames: int,
        include_source_frame: bool,
        frame_within_keyframe_policy: str,
        min_keyframes_required_policy: str,
        traversal_mode: str,
        switch_after_episode: bool,
        segment_order: str,
        scene_order: str,
        include_test: bool,
        fixed_scene_id: Optional[int],
        fixed_segment_id: Optional[int],
        emit_preload_hints: bool,
        warm_next_block_exact: bool,
        warm_next_episode_chain: bool,
    ) -> None:
        if steps_per_block < 1:
            raise ValueError("steps_per_block must be >= 1")
        if blocks_per_episode < 1:
            raise ValueError("blocks_per_episode must be >= 1")
        if total_target_frames < 1:
            raise ValueError("total_target_frames must be >= 1")
        if not include_source_frame:
            raise ValueError("TrainSchedulerV7 requires include_source_frame=true")
        if frame_within_keyframe_policy not in ("random_once_per_episode", "middle_frame"):
            raise ValueError(
                "frame_within_keyframe_policy must be one of {random_once_per_episode,middle_frame}"
            )
        if min_keyframes_required_policy != "skip_if_less_than_window":
            raise ValueError("min_keyframes_required_policy must be skip_if_less_than_window")
        if traversal_mode not in ("linear_scene_segment", "round_robin_episode_interleave"):
            raise ValueError("traversal.mode must be linear_scene_segment or round_robin_episode_interleave")
        if segment_order != "ascending":
            raise ValueError("traversal.segment_order must be ascending")
        if scene_order not in ("ascending", "shuffle_per_epoch"):
            raise ValueError("traversal.scene_order must be ascending or shuffle_per_epoch")
        if not switch_after_episode:
            raise ValueError("traversal.switch_after_episode must be true for TrainSchedulerV7")

        self.dataset = dataset
        self.steps_per_block = int(steps_per_block)
        self.blocks_per_episode = int(blocks_per_episode)
        self.total_target_frames = int(total_target_frames)
        self.include_source_frame = bool(include_source_frame)
        self.frame_within_keyframe_policy = str(frame_within_keyframe_policy)
        self.min_keyframes_required_policy = str(min_keyframes_required_policy)
        self.traversal_mode = str(traversal_mode)
        self.switch_after_episode = bool(switch_after_episode)
        self.segment_order = str(segment_order)
        self.scene_order = str(scene_order)
        self.include_test = bool(include_test)
        self.fixed_scene_id = int(fixed_scene_id) if fixed_scene_id is not None else None
        self.fixed_segment_id = int(fixed_segment_id) if fixed_segment_id is not None else None
        self.emit_preload_hints = bool(emit_preload_hints)
        self.warm_next_block_exact = bool(warm_next_block_exact)
        self.warm_next_episode_chain = bool(warm_next_episode_chain)

        self.episode_window_keyframes = int(self.blocks_per_episode + self.total_target_frames - 1)
        self.U = 1  # Compatibility shim for legacy scheduler-node-sync consumers.

        self.epoch_idx = 0
        self.global_step = 0
        self._pending_events: List[Dict[str, Any]] = []
        self._block_idx_global = 0
        self._episode_idx_global = 0

        self.epoch_plan: List[Dict[str, Any]] = []
        self.episode_cursor_plan: List[EpisodePlanV7] = []
        self.plan_cursor = 0
        self.current_episode_state: Optional[Dict[str, Any]] = None
        self._segment_runtime: Dict[Tuple[int, int], Dict[str, Any]] = {}

        if not self.dataset._initialized:
            self.dataset.initialize()
        self.start_new_epoch()

    def pop_events(self) -> List[Dict[str, Any]]:
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def _emit(self, event: Dict[str, Any]) -> None:
        self._pending_events.append(event)

    def _set_current_scheduler_scope(self, scene_id: int, segment_id: int) -> None:
        if hasattr(self.dataset, "set_preload_active_scope"):
            self.dataset.set_preload_active_scope(int(scene_id), int(segment_id))
        if hasattr(self.dataset, "set_preload_training_scope"):
            self.dataset.set_preload_training_scope(int(scene_id), int(segment_id))

    @staticmethod
    def _frame_targets_to_image_refs(num_cams: int, frame_indices: List[int]) -> List[Tuple[int, int]]:
        refs: List[Tuple[int, int]] = []
        for frame_idx in frame_indices:
            for cam_idx in range(int(num_cams)):
                refs.append((int(frame_idx), int(cam_idx)))
        return refs

    @staticmethod
    def _build_segment_episode_starts(num_keyframes: int, e_blocks: int, window_keyframes: int) -> List[int]:
        if num_keyframes < window_keyframes:
            return []
        starts = list(range(0, num_keyframes - window_keyframes + 1, e_blocks))
        tail = int(num_keyframes - window_keyframes)
        if starts[-1] != tail:
            starts.append(tail)
        return starts

    def _ordered_scene_ids(self) -> List[int]:
        if self.fixed_scene_id is not None:
            return [int(self.fixed_scene_id)]
        out = [int(x) for x in self.dataset.list_training_scene_ids()]
        if len(out) == 0:
            raise ValueError("No valid training scenes in dataset registry list")
        if self.scene_order == "shuffle_per_epoch":
            random.shuffle(out)
        else:
            out.sort()
        return out

    def _ordered_segment_ids(self, scene_id: int) -> List[int]:
        seg_ids = [int(x) for x in self.dataset.list_segment_ids(int(scene_id))]
        if len(seg_ids) == 0:
            return []
        seg_ids.sort()
        if self.fixed_segment_id is not None:
            if self.fixed_segment_id < 0 or self.fixed_segment_id >= len(seg_ids):
                raise ValueError(
                    f"fixed_segment_id={self.fixed_segment_id} out of range for scene_id={scene_id}"
                )
            return [int(seg_ids[int(self.fixed_segment_id)])]
        return seg_ids

    def _build_epoch_plan(self) -> None:
        self.epoch_plan = []
        self.episode_cursor_plan = []
        self.plan_cursor = 0
        self._segment_runtime = {}

        per_segment_episode_starts: Dict[Tuple[int, int], List[int]] = {}
        sidx_cache: Dict[Tuple[int, int], SegmentIndexLike] = {}
        scene_ids = self._ordered_scene_ids()
        for scene_id in scene_ids:
            segment_ids = self._ordered_segment_ids(scene_id)
            for segment_id in segment_ids:
                sidx = self.dataset.get_segment_index(int(scene_id), int(segment_id))
                sidx_cache[(int(scene_id), int(segment_id))] = sidx
                starts = self._build_segment_episode_starts(
                    num_keyframes=len(sidx.keyframe_indices),
                    e_blocks=self.blocks_per_episode,
                    window_keyframes=self.episode_window_keyframes,
                )
                if len(starts) == 0:
                    continue
                key = (int(scene_id), int(segment_id))
                per_segment_episode_starts[key] = [int(st) for st in starts]
                total_blocks = int(len(starts) * self.blocks_per_episode)
                total_steps = int(total_blocks * self.steps_per_block)
                self.epoch_plan.append(
                    {
                        "scene_id": int(scene_id),
                        "segment_id": int(segment_id),
                        "num_keyframes": int(len(sidx.keyframe_indices)),
                        "num_cams": int(sidx.num_cams),
                        "episode_starts": [int(x) for x in starts],
                        "num_episodes": int(len(starts)),
                        "total_blocks": int(total_blocks),
                        "segment_step_budget": int(total_steps),
                    }
                )
                self._segment_runtime[key] = {
                    "segment_local_step": 0,
                    "block_idx_in_segment": 0,
                    "episodes_total": int(len(starts)),
                    "episodes_started": 0,
                    "episodes_completed": 0,
                    "segment_begun": False,
                    "segment_ended": False,
                }

        ordered_episode_keys: List[Tuple[int, int, int]] = []
        if self.traversal_mode == "linear_scene_segment":
            for item in self.epoch_plan:
                key = (int(item["scene_id"]), int(item["segment_id"]))
                for st in per_segment_episode_starts[key]:
                    ordered_episode_keys.append((int(key[0]), int(key[1]), int(st)))
        else:
            rr_order: List[Tuple[int, int]] = [
                (int(item["scene_id"]), int(item["segment_id"])) for item in self.epoch_plan
            ]
            idx_map = {k: 0 for k in rr_order}
            while True:
                progressed = False
                for key in rr_order:
                    starts = per_segment_episode_starts[key]
                    pos = int(idx_map[key])
                    if pos >= len(starts):
                        continue
                    ordered_episode_keys.append((int(key[0]), int(key[1]), int(starts[pos])))
                    idx_map[key] = pos + 1
                    progressed = True
                if not progressed:
                    break

        for scene_id, segment_id, start_pos in ordered_episode_keys:
            key = (int(scene_id), int(segment_id))
            sidx = sidx_cache[key]
            self.episode_cursor_plan.append(
                self._build_episode_plan(
                    sidx,
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    episode_start_keyframe_pos=int(start_pos),
                )
            )

    def start_new_epoch(self) -> None:
        self.epoch_idx += 1
        self.current_episode_state = None
        self._build_epoch_plan()
        if hasattr(self.dataset, "clear_preload_scheduler_scope"):
            self.dataset.clear_preload_scheduler_scope()

    def _segment_plan_item(self, scene_id: int, segment_id: int) -> Dict[str, Any]:
        for item in self.epoch_plan:
            if int(item["scene_id"]) == int(scene_id) and int(item["segment_id"]) == int(segment_id):
                return item
        raise ValueError(f"segment plan item not found for scene={scene_id} segment={segment_id}")

    def _build_episode_plan(
        self,
        sidx: SegmentIndexLike,
        *,
        scene_id: int,
        segment_id: int,
        episode_start_keyframe_pos: int,
    ) -> EpisodePlanV7:
        kfs = list(sidx.keyframe_indices)
        st = int(episode_start_keyframe_pos)
        ed = st + self.episode_window_keyframes
        if ed > len(kfs):
            raise ValueError(
                f"episode window out of range: start={st}, W={self.episode_window_keyframes}, num_kf={len(kfs)}"
            )
        keyframe_window = [int(x) for x in kfs[st:ed]]
        frame_chain = [
            self._choose_frame_for_keyframe_once(list(sidx.keyframe_to_frames[int(kf)]))
            for kf in keyframe_window
        ]
        block_windows = [
            [int(x) for x in frame_chain[b : b + self.total_target_frames]]
            for b in range(self.blocks_per_episode)
        ]
        return EpisodePlanV7(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_start_keyframe_pos=int(st),
            keyframe_window=keyframe_window,
            frame_chain=frame_chain,
            block_windows=block_windows,
            num_cams=int(sidx.num_cams),
        )

    def _choose_frame_for_keyframe_once(self, frames: List[int]) -> int:
        if len(frames) == 0:
            raise ValueError("keyframe_to_frames must not be empty")
        if self.frame_within_keyframe_policy == "middle_frame":
            return int(frames[len(frames) // 2])
        return int(random.choice(frames))

    def _emit_preload_hint(
        self,
        *,
        scene_id: int,
        segment_id: int,
        future_image_refs: List[Tuple[int, int]],
        hint_scope: str,
        block_idx_global: int,
    ) -> None:
        if not self.emit_preload_hints:
            return
        if not hasattr(self.dataset, "build_preload_hint") or not hasattr(self.dataset, "submit_preload_hint"):
            return
        hint = self.dataset.build_preload_hint(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            future_image_refs=future_image_refs,
            scope=hint_scope,
        )
        self.dataset.submit_preload_hint(
            hint=hint,
            hint_scope=hint_scope,
            epoch_idx=int(self.epoch_idx),
            global_step=int(self.global_step),
            block_idx_global=int(block_idx_global),
            include_test=bool(self.include_test),
        )
        self._emit(
            {
                "type": "preload_hint",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "block_idx_global": int(block_idx_global),
                "hint_scope": str(hint_scope),
                "hint": hint,
                "scheduler_version": "v7",
            }
        )

    def _emit_segment_begin_if_needed(self, scene_id: int, segment_id: int) -> None:
        key = (int(scene_id), int(segment_id))
        rt = self._segment_runtime[key]
        if bool(rt["segment_begun"]):
            return
        rt["segment_begun"] = True
        item = self._segment_plan_item(scene_id, segment_id)
        self._emit(
            {
                "type": "segment_begin",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "num_keyframes": int(item["num_keyframes"]),
                "num_cams": int(item["num_cams"]),
                "num_episodes": int(item["num_episodes"]),
                "blocks_per_episode": int(self.blocks_per_episode),
                "segment_step_budget": int(item["segment_step_budget"]),
                "segment_budget_u": int(item["segment_step_budget"]),
                "steps_per_block": int(self.steps_per_block),
                "updates_per_block": int(self.steps_per_block),
                "U": int(self.U),
                "scheduler_version": "v7",
            }
        )
        if hasattr(self.dataset, "set_preload_active_scope"):
            self.dataset.set_preload_active_scope(int(scene_id), int(segment_id))
        if hasattr(self.dataset, "set_preload_training_scope"):
            self.dataset.set_preload_training_scope(int(scene_id), int(segment_id))
        # Segment-level preload: trigger static/meta warming even before first block.
        self._emit_preload_hint(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            future_image_refs=[],
            hint_scope="next_block_exact",
            block_idx_global=int(self._block_idx_global),
        )

    def _emit_segment_end_if_needed(self, scene_id: int, segment_id: int) -> None:
        key = (int(scene_id), int(segment_id))
        rt = self._segment_runtime[key]
        if bool(rt["segment_ended"]):
            return
        if int(rt["episodes_completed"]) < int(rt["episodes_total"]):
            return
        rt["segment_ended"] = True
        item = self._segment_plan_item(scene_id, segment_id)
        self._emit(
            {
                "type": "segment_end",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "segment_local_step": int(rt["segment_local_step"]),
                "segment_step_budget": int(item["segment_step_budget"]),
                "block_idx_in_segment": int(rt["block_idx_in_segment"]),
                "scheduler_version": "v7",
            }
        )
        if hasattr(self.dataset, "clear_preload_scheduler_scope"):
            self.dataset.clear_preload_scheduler_scope()

    def _start_episode_from_plan(self, plan: EpisodePlanV7) -> None:
        scene_id = int(plan.scene_id)
        segment_id = int(plan.segment_id)
        key = (scene_id, segment_id)
        self._set_current_scheduler_scope(scene_id, segment_id)
        self._emit_segment_begin_if_needed(scene_id, segment_id)

        keyframe_window = [int(x) for x in plan.keyframe_window]
        frame_chain = [int(x) for x in plan.frame_chain]
        block_windows = [[int(x) for x in window] for window in plan.block_windows]
        rt = self._segment_runtime[key]
        rt["episodes_started"] = int(rt["episodes_started"]) + 1

        self.current_episode_state = {
            "scene_id": scene_id,
            "segment_id": segment_id,
            "episode_idx_global": int(self._episode_idx_global),
            "episode_start_keyframe_pos": int(plan.episode_start_keyframe_pos),
            "keyframe_window": keyframe_window,
            "frame_chain": frame_chain,
            "block_windows": block_windows,
            "block_cursor": 0,
            "block_repeat_step": 0,
            "current_source_frame_idx": -1,
            "current_target_frame_indices": [],
            "source_keyframe_idx": -1,
            "source_image_ref": (-1, -1),
            "source_image_refs": [],
            "target_image_refs": [],
            "num_cams": int(plan.num_cams),
            "block_idx_global": int(self._block_idx_global),
            "block_idx_in_segment": int(rt["block_idx_in_segment"]),
        }
        self._emit(
            {
                "type": "reset_event",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": scene_id,
                "segment_id": segment_id,
                "reset_episode_idx": int(self._episode_idx_global),
                "reason": "episode_begin",
                "window_keyframes": list(keyframe_window),
                "num_pairs": int(self.blocks_per_episode),
                "scheduler_version": "v7",
            }
        )
        self._episode_idx_global += 1

        if self.warm_next_episode_chain:
            next_plan = self._peek_next_episode_plan()
            if next_plan is not None:
                refs = self._frame_targets_to_image_refs(
                    int(next_plan.num_cams),
                    [int(x) for x in next_plan.frame_chain],
                )
                self._emit_preload_hint(
                    scene_id=int(next_plan.scene_id),
                    segment_id=int(next_plan.segment_id),
                    future_image_refs=refs,
                    hint_scope="episode_chain_exact",
                    block_idx_global=int(self._block_idx_global),
                )

    def _peek_next_episode_plan(self) -> Optional[EpisodePlanV7]:
        if self.plan_cursor < len(self.episode_cursor_plan):
            return self.episode_cursor_plan[self.plan_cursor]
        return None

    def _start_block(self) -> None:
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV7 internal state is not initialized")
        bcur = int(st["block_cursor"])
        block_windows = st["block_windows"]
        if bcur < 0 or bcur >= len(block_windows):
            raise ValueError(f"invalid block_cursor={bcur} for episode")
        target_frames = [int(x) for x in block_windows[bcur]]
        source_frame = int(target_frames[0])
        num_cams = int(st["num_cams"])
        self._set_current_scheduler_scope(int(st["scene_id"]), int(st["segment_id"]))
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        source_keyframe_idx = int(sidx.frame_to_keyframe[int(source_frame)])
        source_image_ref = (int(source_frame), 0)
        source_image_refs = self._frame_targets_to_image_refs(num_cams, [int(source_frame)])
        target_image_refs = self._frame_targets_to_image_refs(num_cams, target_frames)
        st["current_source_frame_idx"] = int(source_frame)
        st["current_target_frame_indices"] = [int(x) for x in target_frames]
        st["source_keyframe_idx"] = int(source_keyframe_idx)
        st["source_image_ref"] = tuple(source_image_ref)
        st["source_image_refs"] = [tuple(x) for x in source_image_refs]
        st["target_image_refs"] = [tuple(x) for x in target_image_refs]
        st["block_repeat_step"] = 0
        st["block_idx_global"] = int(self._block_idx_global)

        self._emit(
            {
                "type": "block_begin",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(st["scene_id"]),
                "segment_id": int(st["segment_id"]),
                "episode_idx_global": int(st["episode_idx_global"]),
                "block_idx_in_episode": int(bcur),
                "block_idx_in_segment": int(st["block_idx_in_segment"]),
                "block_idx_global": int(st["block_idx_global"]),
                "source_frame_idx": int(source_frame),
                "source_keyframe_idx": int(source_keyframe_idx),
                "source_image_ref": tuple(source_image_ref),
                "target_frame_indices": [int(x) for x in target_frames],
                "target_image_refs": [tuple(x) for x in target_image_refs],
                "overlap_mode": "none",
                "steps_per_block": int(self.steps_per_block),
                "K_u_nominal": int(self.steps_per_block),
                "K_u_effective": int(self.steps_per_block),
                "K_steps_effective": int(self.steps_per_block),
                "U": int(self.U),
                "scheduler_version": "v7",
            }
        )
        self._block_idx_global += 1

        if self.warm_next_block_exact:
            nb = bcur + 1
            if nb < len(block_windows):
                next_frames = [int(x) for x in block_windows[nb]]
                next_refs = self._frame_targets_to_image_refs(num_cams, next_frames)
                self._emit_preload_hint(
                    scene_id=int(st["scene_id"]),
                    segment_id=int(st["segment_id"]),
                    future_image_refs=next_refs,
                    hint_scope="next_block_exact",
                    block_idx_global=int(st["block_idx_global"]),
                )

    def _batch_from_state(self, st: Dict[str, Any]) -> Dict[str, Any]:
        req = _BatchRequestV7Compat(
            scene_id=int(st["scene_id"]),
            segment_id=int(st["segment_id"]),
            source_image_ref=(int(st["source_image_ref"][0]), int(st["source_image_ref"][1])),
            target_image_refs=[(int(x[0]), int(x[1])) for x in st["target_image_refs"]],
            source_image_refs=[(int(x[0]), int(x[1])) for x in st["source_image_refs"]],
            include_test=bool(self.include_test),
            test_image_refs=None,
        )
        return self.dataset.get_segment_batch_from_image_refs(
            req,
            enforce_target0_equals_source=True,
        )

    def _aligned_info(self, st: Dict[str, Any]) -> Dict[str, Any]:
        key = (int(st["scene_id"]), int(st["segment_id"]))
        rt = self._segment_runtime[key]
        item = self._segment_plan_item(int(st["scene_id"]), int(st["segment_id"]))
        return {
            "epoch_idx": int(self.epoch_idx),
            "global_step": int(self.global_step),
            "scene_id": int(st["scene_id"]),
            "segment_id": int(st["segment_id"]),
            "segment_local_step": int(rt["segment_local_step"]),
            "segment_step_budget": int(item["segment_step_budget"]),
            "segment_local_u": int(rt["segment_local_step"]),
            "segment_budget_u": int(item["segment_step_budget"]),
            "block_idx_in_segment": int(st["block_idx_in_segment"]),
            "block_idx_global": int(st["block_idx_global"]),
            "source_frame_idx": int(st["current_source_frame_idx"]),
            "source_keyframe_idx": int(st.get("source_keyframe_idx", -1)),
            "source_cam_idx": int(st["source_image_ref"][1]),
            "source_image_ref": tuple(st["source_image_ref"]),
            "target_frame_indices": [int(x) for x in st["current_target_frame_indices"]],
            "target_image_refs": [tuple(x) for x in st["target_image_refs"]],
            "U": int(self.U),
            "K_u_nominal": int(self.steps_per_block),
            "K_u_effective": int(self.steps_per_block),
            "K_steps_effective": int(self.steps_per_block),
            "K_steps": int(self.steps_per_block),
            "R_steps": 0,
            "T_steps": int(self.steps_per_block),
            "episode_idx_global": int(st["episode_idx_global"]),
            "block_repeat_step": int(st["block_repeat_step"]),
            "scheduler_version": "v7",
        }

    def _ensure_episode_state(self) -> None:
        if self.current_episode_state is not None:
            return
        while self.plan_cursor >= len(self.episode_cursor_plan):
            self.start_new_epoch()
            if len(self.episode_cursor_plan) == 0:
                raise ValueError("TrainSchedulerV7: no valid episodes available in epoch plan")
        plan = self.episode_cursor_plan[self.plan_cursor]
        self.plan_cursor += 1
        self._start_episode_from_plan(plan)
        self._start_block()

    def _finalize_episode_if_needed(self) -> None:
        st = self.current_episode_state
        if st is None:
            return
        bcur = int(st["block_cursor"])
        if bcur < self.blocks_per_episode:
            return
        key = (int(st["scene_id"]), int(st["segment_id"]))
        rt = self._segment_runtime[key]
        rt["episodes_completed"] = int(rt["episodes_completed"]) + 1
        self._emit(
            {
                "type": "episode_end",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(st["scene_id"]),
                "segment_id": int(st["segment_id"]),
                "episode_idx_global": int(st["episode_idx_global"]),
                "reason": "episode_exhausted",
                "scheduler_version": "v7",
            }
        )
        self.current_episode_state = None
        self._emit_segment_end_if_needed(int(key[0]), int(key[1]))

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        self._ensure_episode_state()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV7 internal state is not initialized")
        batch = self._batch_from_state(st)
        aligned = self._aligned_info(st)
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_peek"] = True
        return batch

    def next_batch(self) -> Dict[str, Any]:
        self._ensure_episode_state()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV7 internal state is not initialized")
        batch = self._batch_from_state(st)
        key = (int(st["scene_id"]), int(st["segment_id"]))
        rt = self._segment_runtime[key]

        rt["segment_local_step"] = int(rt["segment_local_step"]) + 1
        st["block_repeat_step"] = int(st["block_repeat_step"]) + 1
        self.global_step += 1

        aligned = self._aligned_info(st)
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)

        if int(st["block_repeat_step"]) >= self.steps_per_block:
            self._emit(
                {
                    "type": "block_end",
                    "epoch_idx": int(self.epoch_idx),
                    "global_step": int(self.global_step),
                    "scene_id": int(st["scene_id"]),
                    "segment_id": int(st["segment_id"]),
                    "episode_idx_global": int(st["episode_idx_global"]),
                    "block_idx_in_episode": int(st["block_cursor"]),
                    "block_idx_in_segment": int(st["block_idx_in_segment"]),
                    "block_idx_global": int(st["block_idx_global"]),
                    "source_frame_idx": int(st["current_source_frame_idx"]),
                    "source_image_ref": tuple(st["source_image_ref"]),
                    "target_frame_indices": [int(x) for x in st["current_target_frame_indices"]],
                    "target_image_refs": [tuple(x) for x in st["target_image_refs"]],
                    "num_updates_in_block": int(self.steps_per_block),
                    "K_u_nominal": int(self.steps_per_block),
                    "K_u_effective": int(self.steps_per_block),
                    "K_steps_effective": int(self.steps_per_block),
                    "U": int(self.U),
                    "scheduler_version": "v7",
                }
            )
            rt["block_idx_in_segment"] = int(rt["block_idx_in_segment"]) + 1
            st["block_idx_in_segment"] = int(rt["block_idx_in_segment"])
            st["block_cursor"] = int(st["block_cursor"]) + 1
            st["block_repeat_step"] = 0
            if int(st["block_cursor"]) < self.blocks_per_episode:
                self._start_block()
            else:
                self._finalize_episode_if_needed()

        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
        return batch

    def get_current_info(self) -> Dict[str, Any]:
        st = self.current_episode_state
        if st is not None:
            return self._aligned_info(st)
        if len(self.episode_cursor_plan) == 0:
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
                "K_u_nominal": int(self.steps_per_block),
                "K_u_effective": int(self.steps_per_block),
                "K_steps_effective": int(self.steps_per_block),
                "K_steps": int(self.steps_per_block),
                "R_steps": 0,
                "T_steps": int(self.steps_per_block),
                "episode_idx_global": -1,
                "block_repeat_step": 0,
                "scheduler_version": "v7",
            }
        nxt = self.episode_cursor_plan[min(self.plan_cursor, len(self.episode_cursor_plan) - 1)]
        item = self._segment_plan_item(int(nxt.scene_id), int(nxt.segment_id))
        key = (int(nxt.scene_id), int(nxt.segment_id))
        rt = self._segment_runtime[key]
        return {
            "epoch_idx": int(self.epoch_idx),
            "global_step": int(self.global_step),
            "scene_id": int(nxt.scene_id),
            "segment_id": int(nxt.segment_id),
            "segment_local_step": int(rt["segment_local_step"]),
            "segment_step_budget": int(item["segment_step_budget"]),
            "segment_local_u": int(rt["segment_local_step"]),
            "segment_budget_u": int(item["segment_step_budget"]),
            "block_idx_in_segment": int(rt["block_idx_in_segment"]),
            "block_idx_global": int(self._block_idx_global),
            "source_frame_idx": -1,
            "source_keyframe_idx": -1,
            "source_cam_idx": -1,
            "source_image_ref": (-1, -1),
            "target_frame_indices": [],
            "target_image_refs": [],
            "U": int(self.U),
            "K_u_nominal": int(self.steps_per_block),
            "K_u_effective": int(self.steps_per_block),
            "K_steps_effective": int(self.steps_per_block),
            "K_steps": int(self.steps_per_block),
            "R_steps": 0,
            "T_steps": int(self.steps_per_block),
            "episode_idx_global": -1,
            "block_repeat_step": 0,
            "scheduler_version": "v7",
        }
