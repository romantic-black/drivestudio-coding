from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from datasets.train_scheduler_v7 import SegmentIndexLike, TrainSchedulerDatasetV7, TrainSchedulerV7


@dataclass(frozen=True)
class EpisodePlanV8:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int
    keyframe_window: List[int]
    frame_chain: List[int]
    num_cams: int


class TrainSchedulerV8(TrainSchedulerV7):
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
        block_order: str = "block_major",
        step_major_switch_interval_steps: int = 1,
        target_policy: str = "visited_episode_frames",
        history_target_policy: str = "nearest_visited",
        reset_policy: str = "episode_end",
        near_random_supervision_cfg: Optional[Any] = None,
        aux_feature_splat_targets_cfg: Optional[Any] = None,
        block_source_frame_policy: str = "fixed_once_per_episode",
        episode_source_mode: str = "keyframes",
    ) -> None:
        if total_target_frames < 1:
            raise ValueError("scheduler_v8.episode.total_target_frames must be >= 1")
        if total_target_frames > blocks_per_episode:
            raise ValueError(
                "scheduler_v8 does not use future frames; total_target_frames must be <= blocks_per_episode"
            )
        if str(target_policy) != "visited_episode_frames":
            raise ValueError("scheduler_v8 only supports target_policy=visited_episode_frames")
        if str(reset_policy) != "episode_end":
            raise ValueError("scheduler_v8 requires execution.reset_policy=episode_end")
        if not include_source_frame:
            raise ValueError("scheduler_v8 requires include_source_frame=true")

        # V7 __init__ calls self.start_new_epoch(); V8 must skip that initial call so
        # epoch/window state is initialized exactly once with V8 semantics (W = E).
        self._skip_next_start_new_epoch = True
        self.target_policy = str(target_policy)
        self.history_target_policy = str(history_target_policy)
        if self.history_target_policy not in ("nearest_visited", "random_visited"):
            raise ValueError(
                "scheduler_v8.episode.history_target_policy must be one of "
                "['nearest_visited', 'random_visited']"
            )
        self.reset_policy = str(reset_policy)
        self.block_source_frame_policy = str(block_source_frame_policy)
        self.episode_source_mode = str(episode_source_mode).strip().lower()
        if self.episode_source_mode not in ("keyframes", "segment_frames"):
            raise ValueError(
                "scheduler_v8.episode.source_mode must be one of ['keyframes', 'segment_frames']"
            )
        if self.block_source_frame_policy not in ("fixed_once_per_episode", "random_within_keyframe_per_visit"):
            raise ValueError(
                "scheduler_v8.episode.block_source_frame_policy must be one of "
                "['fixed_once_per_episode', 'random_within_keyframe_per_visit']"
            )
        if self.episode_source_mode == "segment_frames" and self.block_source_frame_policy != "fixed_once_per_episode":
            raise ValueError(
                "scheduler_v8.episode.source_mode=segment_frames requires "
                "block_source_frame_policy=fixed_once_per_episode"
            )
        self.near_random_cfg = near_random_supervision_cfg or {}
        self.near_random_enable = bool(self._cfg_get(self.near_random_cfg, "enable", False))
        self.near_random_frames_per_block = int(self._cfg_get(self.near_random_cfg, "frames_per_block", 1))
        self.near_random_same_keyframe_only = bool(self._cfg_get(self.near_random_cfg, "same_keyframe_only", True))
        self.near_random_insufficient_policy = str(self._cfg_get(self.near_random_cfg, "insufficient_policy", "skip"))
        self.near_random_exclude_source = bool(self._cfg_get(self.near_random_cfg, "exclude_source_frame", True))
        self.near_random_exclude_existing = bool(self._cfg_get(self.near_random_cfg, "exclude_existing_target_frames", True))
        self.near_random_sample_once_per_block = bool(self._cfg_get(self.near_random_cfg, "sample_once_per_block", True))
        self.near_random_camera_policy = str(self._cfg_get(self.near_random_cfg, "camera_policy", "all_cams"))
        self.near_random_role_name = str(self._cfg_get(self.near_random_cfg, "role_name", "near_random"))
        self.aux_feature_splat_cfg = aux_feature_splat_targets_cfg or {}
        self.aux_feature_splat_enable = bool(self._cfg_get(self.aux_feature_splat_cfg, "enable", False))
        self.aux_feature_splat_role_name = str(self._cfg_get(self.aux_feature_splat_cfg, "role_name", "aux_feature_splat"))
        self.aux_feature_splat_schedule = list(self._cfg_get(self.aux_feature_splat_cfg, "policy_schedule", []) or [])
        self.aux_feature_splat_max_refs_default = int(self._cfg_get(self.aux_feature_splat_cfg, "max_refs_per_step", 1))
        self.aux_feature_splat_camera_policy = str(
            self._cfg_get(self.aux_feature_splat_cfg, "aux_camera_policy", "random_from_source_cams")
        ).strip().lower()
        self.aux_feature_splat_fixed_cam_id = int(self._cfg_get(self.aux_feature_splat_cfg, "fixed_cam_id", 0))
        if self.near_random_enable:
            if self.near_random_frames_per_block < 1:
                raise ValueError("near_random_supervision.frames_per_block must be >= 1")
            if not self.near_random_same_keyframe_only:
                raise ValueError("v1 only supports near_random_supervision.same_keyframe_only=true")
            if self.near_random_insufficient_policy != "skip":
                raise ValueError("v1 only supports near_random_supervision.insufficient_policy=skip")
            if self.near_random_camera_policy != "all_cams":
                raise ValueError("v1 only supports near_random_supervision.camera_policy=all_cams")
        if self.aux_feature_splat_enable:
            if self.aux_feature_splat_camera_policy not in {"random_from_source_cams", "fixed_cam"}:
                raise ValueError(
                    "aux_feature_splat_targets.aux_camera_policy must be one of "
                    "['random_from_source_cams', 'fixed_cam']"
                )
        super().__init__(
            dataset=dataset,
            steps_per_block=steps_per_block,
            blocks_per_episode=blocks_per_episode,
            total_target_frames=total_target_frames,
            include_source_frame=include_source_frame,
            frame_within_keyframe_policy=frame_within_keyframe_policy,
            min_keyframes_required_policy=min_keyframes_required_policy,
            traversal_mode=traversal_mode,
            switch_after_episode=switch_after_episode,
            segment_order=segment_order,
            scene_order=scene_order,
            include_test=include_test,
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            emit_preload_hints=emit_preload_hints,
            warm_next_block_exact=warm_next_block_exact,
            warm_next_episode_chain=warm_next_episode_chain,
            block_order=block_order,
            step_major_switch_interval_steps=step_major_switch_interval_steps,
        )
        self.episode_window_keyframes = int(self.blocks_per_episode)
        self.start_new_epoch()

    @staticmethod
    def _cfg_get(node: Any, key: str, default: Any) -> Any:
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

    @staticmethod
    def _sample_no_replace(candidates: List[int], k: int) -> List[int]:
        if int(k) < 0:
            raise ValueError("k must be >= 0")
        if int(k) > len(candidates):
            raise ValueError(f"cannot sample {int(k)} without replacement from {len(candidates)} candidates")
        return [int(x) for x in random.sample(list(candidates), int(k))]

    @staticmethod
    def _resolve_block_source_frame_at_index(
        *,
        block_source_frames: List[int],
        frame_chain: List[int],
        block_idx: int,
    ) -> int:
        b = int(block_idx)
        if b < 0 or b >= len(frame_chain):
            raise ValueError(f"block_idx={b} out of range for frame_chain len={len(frame_chain)}")
        if b < len(block_source_frames):
            source = int(block_source_frames[b])
            if source >= 0:
                return int(source)
        return int(frame_chain[b])

    def _episode_source_candidates_for_keyframe_window(
        self,
        *,
        sidx: SegmentIndexLike,
        keyframe_window: List[int],
        frame_chain: List[int],
    ) -> List[int]:
        if str(getattr(self, "episode_source_mode", "keyframes")) == "segment_frames":
            return [int(x) for x in frame_chain]
        if self.block_source_frame_policy == "fixed_once_per_episode":
            return [int(x) for x in frame_chain]
        out: List[int] = []
        seen: Set[int] = set()
        for kf in keyframe_window:
            for f in list(sidx.keyframe_to_frames[int(kf)]):
                fi = int(f)
                if fi in seen:
                    continue
                out.append(int(fi))
                seen.add(int(fi))
        if len(out) == 0:
            return [int(x) for x in frame_chain]
        return out

    def _sample_source_frame_for_block(
        self,
        *,
        st: Dict[str, Any],
        sidx: SegmentIndexLike,
        block_idx: int,
    ) -> tuple[int, int]:
        bcur = int(block_idx)
        frame_chain = [int(x) for x in st["frame_chain"]]
        keyframe_window = [int(x) for x in st["keyframe_window"]]
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid block_idx={bcur} for episode")
        source_keyframe_idx = int(keyframe_window[bcur])
        if self.block_source_frame_policy == "fixed_once_per_episode":
            return int(source_keyframe_idx), int(frame_chain[bcur])
        frames = [int(x) for x in list(sidx.keyframe_to_frames[int(source_keyframe_idx)])]
        if len(frames) == 0:
            raise ValueError(f"keyframe_to_frames[{int(source_keyframe_idx)}] must not be empty")
        return int(source_keyframe_idx), int(random.choice(frames))

    def _near_random_cached_frames_valid(
        self,
        *,
        cached_frames: List[int],
        sidx: SegmentIndexLike,
        source_keyframe_idx: int,
        source_frame: int,
        existing_target_frames: List[int],
        num_frames: int,
    ) -> bool:
        cached = [int(x) for x in cached_frames]
        if len(cached) != int(num_frames):
            return False
        keyframe_frames = set(int(x) for x in list(sidx.keyframe_to_frames[int(source_keyframe_idx)]))
        existing = set(int(x) for x in existing_target_frames)
        if len(set(cached)) != len(cached):
            return False
        for f in cached:
            if int(f) not in keyframe_frames:
                return False
            if self.near_random_exclude_source and int(f) == int(source_frame):
                return False
            if self.near_random_exclude_existing and int(f) in existing:
                return False
        return True

    def start_new_epoch(self) -> None:
        if bool(getattr(self, "_skip_next_start_new_epoch", False)):
            self._skip_next_start_new_epoch = False
            return
        super().start_new_epoch()

    def _build_epoch_plan(self) -> None:
        if str(getattr(self, "episode_source_mode", "keyframes")) != "segment_frames":
            super()._build_epoch_plan()
            return

        self.epoch_plan = []
        self.episode_cursor_plan = []
        self.plan_cursor = 0
        self._segment_runtime = {}

        sidx_cache: Dict[Tuple[int, int], SegmentIndexLike] = {}
        ordered_episode_keys: List[Tuple[int, int, int]] = []
        for scene_id in self._ordered_scene_ids():
            for segment_id in self._ordered_segment_ids(int(scene_id)):
                sidx = self.dataset.get_segment_index(int(scene_id), int(segment_id))
                frames = [int(x) for x in list(getattr(sidx, "frame_indices", []))]
                if len(frames) == 0:
                    continue
                if len(frames) != int(self.blocks_per_episode):
                    raise ValueError(
                        "scheduler_v8.episode.source_mode=segment_frames requires "
                        "blocks_per_episode to equal len(segment.frame_indices). "
                        f"scene={int(scene_id)} segment={int(segment_id)} "
                        f"blocks_per_episode={int(self.blocks_per_episode)} frames={len(frames)}"
                    )
                key = (int(scene_id), int(segment_id))
                sidx_cache[key] = sidx
                total_blocks = int(self.blocks_per_episode)
                total_steps = int(total_blocks * self.steps_per_block)
                self.epoch_plan.append(
                    {
                        "scene_id": int(scene_id),
                        "segment_id": int(segment_id),
                        "num_keyframes": int(len(sidx.keyframe_indices)),
                        "num_frames": int(len(frames)),
                        "num_cams": int(sidx.num_cams),
                        "episode_starts": [0],
                        "num_episodes": 1,
                        "total_blocks": int(total_blocks),
                        "segment_step_budget": int(total_steps),
                    }
                )
                self._segment_runtime[key] = {
                    "segment_local_step": 0,
                    "block_idx_in_segment": 0,
                    "episodes_total": 1,
                    "episodes_started": 0,
                    "episodes_completed": 0,
                    "segment_begun": False,
                    "segment_ended": False,
                }
                ordered_episode_keys.append((int(scene_id), int(segment_id), 0))

        if self.traversal_mode == "round_robin_episode_interleave" and self.scene_order == "shuffle_per_epoch":
            random.shuffle(ordered_episode_keys)

        for scene_id, segment_id, start_pos in ordered_episode_keys:
            key = (int(scene_id), int(segment_id))
            self.episode_cursor_plan.append(
                self._build_episode_plan(
                    sidx_cache[key],
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    episode_start_keyframe_pos=int(start_pos),
                )
            )

    def _emit(self, event: Dict[str, Any]) -> None:
        out = dict(event)
        if str(out.get("scheduler_version", "")) == "v7":
            out["scheduler_version"] = "v8"
        super()._emit(out)

    def _build_episode_plan(
        self,
        sidx: SegmentIndexLike,
        *,
        scene_id: int,
        segment_id: int,
        episode_start_keyframe_pos: int,
    ) -> EpisodePlanV8:
        if str(getattr(self, "episode_source_mode", "keyframes")) == "segment_frames":
            if int(episode_start_keyframe_pos) != 0:
                raise ValueError("segment_frames episode mode supports only episode_start_keyframe_pos=0")
            frames = [int(x) for x in list(getattr(sidx, "frame_indices", []))]
            if len(frames) != int(self.blocks_per_episode):
                raise ValueError(
                    "segment_frames episode frame count mismatch: "
                    f"frames={len(frames)} blocks_per_episode={int(self.blocks_per_episode)}"
                )
            frame_to_keyframe = getattr(sidx, "frame_to_keyframe", {}) or {}
            keyframe_window = [int(frame_to_keyframe.get(int(f), -1)) for f in frames]
            return EpisodePlanV8(
                scene_id=int(scene_id),
                segment_id=int(segment_id),
                episode_start_keyframe_pos=0,
                keyframe_window=keyframe_window,
                frame_chain=[int(x) for x in frames],
                num_cams=int(sidx.num_cams),
            )

        kfs = list(sidx.keyframe_indices)
        st = int(episode_start_keyframe_pos)
        episode_blocks = self._episode_num_blocks_for_start(
            num_keyframes=len(kfs),
            episode_start_keyframe_pos=int(st),
        )
        if episode_blocks <= 0:
            raise ValueError(
                "episode has no usable blocks: "
                f"start={st}, num_kf={len(kfs)}, blocks_per_episode={int(self.blocks_per_episode)}"
            )
        ed = st + int(episode_blocks)
        if ed > len(kfs):
            raise ValueError(
                "episode window out of range: "
                f"start={st}, W={self.blocks_per_episode}, num_kf={len(kfs)}"
            )
        keyframe_window = [int(x) for x in kfs[st:ed]]
        frame_chain = [
            self._choose_frame_for_keyframe_once(list(sidx.keyframe_to_frames[int(kf)]))
            for kf in keyframe_window
        ]
        return EpisodePlanV8(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_start_keyframe_pos=int(st),
            keyframe_window=keyframe_window,
            frame_chain=[int(x) for x in frame_chain],
            num_cams=int(sidx.num_cams),
        )

    def _start_episode_from_plan(self, plan: EpisodePlanV8) -> None:
        scene_id = int(plan.scene_id)
        segment_id = int(plan.segment_id)
        key = (scene_id, segment_id)
        self._set_current_scheduler_scope(scene_id, segment_id)
        self._emit_segment_begin_if_needed(scene_id, segment_id)

        keyframe_window = [int(x) for x in plan.keyframe_window]
        frame_chain = [int(x) for x in plan.frame_chain]
        episode_num_blocks = int(len(frame_chain))
        if episode_num_blocks <= 0:
            raise ValueError("episode must contain at least one block")
        episode_block_visit_order = self._build_episode_block_visit_order(episode_num_blocks)
        episode_total_steps = int(len(episode_block_visit_order))
        sidx = self.dataset.get_segment_index(scene_id, segment_id)
        episode_source_candidates = self._episode_source_candidates_for_keyframe_window(
            sidx=sidx,
            keyframe_window=[int(x) for x in keyframe_window],
            frame_chain=[int(x) for x in frame_chain],
        )
        rt = self._segment_runtime[key]
        rt["episodes_started"] = int(rt["episodes_started"]) + 1
        episode_base_block_idx_global = int(self._block_idx_global)
        episode_base_block_idx_in_segment = int(rt["block_idx_in_segment"])
        if self._block_order_uses_episode_visit_order():
            self._block_idx_global = int(self._block_idx_global) + int(episode_num_blocks)

        self.current_episode_state = {
            "scene_id": scene_id,
            "segment_id": segment_id,
            "episode_idx_global": int(self._episode_idx_global),
            "episode_start_keyframe_pos": int(plan.episode_start_keyframe_pos),
            "keyframe_window": keyframe_window,
            "frame_chain": frame_chain,
            "episode_source_candidate_frames": [int(x) for x in episode_source_candidates],
            "block_current_source_frame_indices": [int(x) for x in frame_chain],
            "episode_num_blocks": int(episode_num_blocks),
            "episode_total_steps": int(episode_total_steps),
            "episode_block_visit_order": [int(x) for x in episode_block_visit_order],
            "block_cursor": int(episode_block_visit_order[0]) if self._block_order_uses_episode_visit_order() else 0,
            "block_repeat_step": 0,
            "episode_step_cursor": 0,
            "block_update_counts": [0 for _ in range(int(episode_num_blocks))],
            "block_started": [False for _ in range(int(episode_num_blocks))],
            "block_ended": [False for _ in range(int(episode_num_blocks))],
            "visited_block_indices": set(),
            "block_first_visit_order": {},
            "block_first_target_frame_indices": {},
            "block_last_target_frame_indices": {},
            "block_near_random_frame_indices": {},
            "block_target_frame_roles": {},
            "block_target_image_roles": {},
            "block_target_image_loss_base_weights": {},
            "near_random_attempted_blocks": 0,
            "near_random_sampled_blocks": 0,
            "near_random_skipped_blocks": 0,
            "near_random_candidate_frames_sum": 0.0,
            "current_source_frame_idx": -1,
            "current_target_frame_indices": [],
            "current_target_frame_roles": [],
            "source_keyframe_idx": -1,
            "source_image_ref": (-1, -1),
            "source_image_refs": [],
            "target_image_refs": [],
            "target_image_roles": [],
            "aux_image_refs": [],
            "aux_image_roles": [],
            "num_cams": int(plan.num_cams),
            "episode_base_block_idx_global": int(episode_base_block_idx_global),
            "episode_base_block_idx_in_segment": int(episode_base_block_idx_in_segment),
            "block_idx_global": int(episode_base_block_idx_global),
            "block_idx_in_segment": int(episode_base_block_idx_in_segment),
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
                "num_pairs": int(episode_num_blocks),
                "scheduler_version": "v8",
            }
        )
        self._episode_idx_global += 1

        refs = self._frame_targets_to_image_refs(
            int(plan.num_cams),
            [int(x) for x in episode_source_candidates],
        )
        if self._block_order_uses_episode_visit_order():
            self._emit_preload_hint(
                scene_id=int(plan.scene_id),
                segment_id=int(plan.segment_id),
                future_image_refs=refs,
                hint_scope="episode_chain_exact",
                block_idx_global=int(episode_base_block_idx_global),
            )
        if self.warm_next_episode_chain:
            next_plan = self._peek_next_episode_plan()
            if next_plan is not None:
                next_sidx = self.dataset.get_segment_index(int(next_plan.scene_id), int(next_plan.segment_id))
                next_episode_source_candidates = self._episode_source_candidates_for_keyframe_window(
                    sidx=next_sidx,
                    keyframe_window=[int(x) for x in next_plan.keyframe_window],
                    frame_chain=[int(x) for x in next_plan.frame_chain],
                )
                next_refs = self._frame_targets_to_image_refs(
                    int(next_plan.num_cams),
                    [int(x) for x in next_episode_source_candidates],
                )
                self._emit_preload_hint(
                    scene_id=int(next_plan.scene_id),
                    segment_id=int(next_plan.segment_id),
                    future_image_refs=next_refs,
                    hint_scope="episode_chain_exact",
                    block_idx_global=int(episode_base_block_idx_global),
                )

    def _build_target_frames_for_block_v8(
        self,
        *,
        frame_chain: List[int],
        block_source_frames: List[int],
        block_idx: int,
        source_frame: int,
        visited_block_indices: Set[int],
        max_target_frames: int,
    ) -> List[int]:
        chain = [int(x) for x in frame_chain]
        sources = [int(x) for x in block_source_frames]
        candidates = sorted(int(b) for b in visited_block_indices if int(b) != int(block_idx))
        prev_blocks = sorted([b for b in candidates if b < int(block_idx)], reverse=True)
        next_blocks = sorted([b for b in candidates if b > int(block_idx)])
        selected_blocks: List[int] = []
        max_history_frames = max(int(max_target_frames) - 1, 0)
        if self.history_target_policy == "random_visited":
            selected_blocks = self._sample_no_replace(candidates, min(max_history_frames, len(candidates)))
        else:
            for b in prev_blocks:
                if len(selected_blocks) >= max_history_frames:
                    break
                selected_blocks.append(int(b))
            for b in next_blocks:
                if len(selected_blocks) >= max_history_frames:
                    break
                selected_blocks.append(int(b))
        return [int(source_frame)] + [
            int(
                self._resolve_block_source_frame_at_index(
                    block_source_frames=sources,
                    frame_chain=chain,
                    block_idx=int(b),
                )
            )
            for b in selected_blocks
        ]

    def _resolve_aux_policy_for_step(self, step: int) -> tuple[str, int]:
        if not self.aux_feature_splat_enable:
            return "", 0
        phases = list(self.aux_feature_splat_schedule)
        if len(phases) == 0:
            policy = str(self._cfg_get(self.aux_feature_splat_cfg, "policy", "adjacent_frame_same_camera"))
            max_refs = int(self._cfg_get(self.aux_feature_splat_cfg, "max_refs_per_step", self.aux_feature_splat_max_refs_default))
            return policy, max(max_refs, 0)
        for phase in phases:
            until_step = int(self._cfg_get(phase, "until_step", -1))
            policy = str(self._cfg_get(phase, "policy", "adjacent_frame_same_camera"))
            max_refs = int(self._cfg_get(phase, "max_refs_per_step", self.aux_feature_splat_max_refs_default))
            if until_step < 0 or int(step) <= int(until_step):
                return policy, max(max_refs, 0)
        last = phases[-1]
        return (
            str(self._cfg_get(last, "policy", "adjacent_frame_same_camera")),
            max(int(self._cfg_get(last, "max_refs_per_step", self.aux_feature_splat_max_refs_default)), 0),
        )

    @staticmethod
    def _dedupe_image_refs_keep_order(refs: List[tuple[int, int]]) -> List[tuple[int, int]]:
        seen: Set[tuple[int, int]] = set()
        out: List[tuple[int, int]] = []
        for ref in refs:
            r = (int(ref[0]), int(ref[1]))
            if r in seen:
                continue
            seen.add(r)
            out.append(r)
        return out

    def _select_aux_source_cam(self, *, source_image_refs: Sequence[tuple[int, int]], num_cams: int) -> int:
        source_cams = sorted({int(ref[1]) for ref in source_image_refs})
        if len(source_cams) == 0:
            raise ValueError("aux_feature_splat_targets requires non-empty source_image_refs.")
        if self.aux_feature_splat_camera_policy == "fixed_cam":
            cam = int(self.aux_feature_splat_fixed_cam_id)
            if cam < 0 or cam >= int(num_cams):
                raise ValueError(
                    f"aux_feature_splat_targets.fixed_cam_id={cam} out of range for num_cams={int(num_cams)}."
                )
            if cam not in source_cams:
                raise ValueError(
                    f"aux_feature_splat_targets.fixed_cam_id={cam} is not present in source_image_refs={list(source_image_refs)}."
                )
            return cam
        if self.aux_feature_splat_camera_policy == "random_from_source_cams":
            return int(random.choice(source_cams))
        raise ValueError(f"unsupported aux_camera_policy={self.aux_feature_splat_camera_policy!r}")

    def _build_aux_image_refs_for_block(
        self,
        *,
        sidx: SegmentIndexLike,
        source_keyframe_idx: int,
        source_frame: int,
        source_cam: int,
        base_target_frames: List[int],
        num_cams: int,
        exclude_image_refs: Optional[Sequence[tuple[int, int]]] = None,
    ) -> tuple[List[tuple[int, int]], List[str]]:
        policy, max_refs = self._resolve_aux_policy_for_step(int(self.global_step))
        if max_refs <= 0 or policy == "":
            return [], []
        refs: List[tuple[int, int]] = []
        policy_norm = str(policy).strip().lower()
        source_cam = int(source_cam)
        if policy_norm == "adjacent_frame_same_camera":
            frames = [int(x) for x in list(sidx.keyframe_to_frames[int(source_keyframe_idx)])]
            frames_sorted = sorted(frames)
            if int(source_frame) in frames_sorted:
                idx = frames_sorted.index(int(source_frame))
                cand_frames: List[int] = []
                if idx - 1 >= 0:
                    cand_frames.append(int(frames_sorted[idx - 1]))
                if idx + 1 < len(frames_sorted):
                    cand_frames.append(int(frames_sorted[idx + 1]))
                for f in cand_frames[:max_refs]:
                    refs.append((int(f), int(source_cam)))
        elif policy_norm == "near_random":
            near_frames, _ = self._sample_near_random_frames_for_block(
                sidx=sidx,
                source_keyframe_idx=int(source_keyframe_idx),
                source_frame=int(source_frame),
                existing_target_frames=[int(x) for x in base_target_frames],
                num_frames=int(max_refs),
            )
            for f in near_frames:
                refs.append((int(f), int(source_cam)))
        else:
            raise ValueError(f"unsupported aux_feature_splat_targets policy={policy!r}")
        refs = self._dedupe_image_refs_keep_order(refs)
        if exclude_image_refs is not None:
            excluded = {(int(r[0]), int(r[1])) for r in exclude_image_refs}
            refs = [r for r in refs if (int(r[0]), int(r[1])) not in excluded]
        roles = [str(self.aux_feature_splat_role_name) for _ in refs]
        return refs[:max_refs], roles[:max_refs]

    def _select_block(self, block_idx: int) -> None:
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV8 internal state is not initialized")
        frame_chain = [int(x) for x in st["frame_chain"]]
        bcur = int(block_idx)
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid block_idx={bcur} for episode")

        self._set_current_scheduler_scope(int(st["scene_id"]), int(st["segment_id"]))
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        source_keyframe_idx, source_frame = self._sample_source_frame_for_block(
            st=st,
            sidx=sidx,
            block_idx=int(bcur),
        )
        block_source_frames = [int(x) for x in st.get("block_current_source_frame_indices", frame_chain)]
        if len(block_source_frames) != len(frame_chain):
            block_source_frames = [int(x) for x in frame_chain]
        base_target_frames = self._build_target_frames_for_block_v8(
            frame_chain=frame_chain,
            block_source_frames=block_source_frames,
            block_idx=bcur,
            source_frame=int(source_frame),
            visited_block_indices=set(st["visited_block_indices"]),
            max_target_frames=int(self.total_target_frames),
        )
        base_roles = ["source"] + ["visited" for _ in base_target_frames[1:]]
        num_cams = int(st["num_cams"])
        near_random_frames: List[int] = []
        if self.near_random_enable and not self.aux_feature_splat_enable:
            reuse_cached = False
            if self.near_random_sample_once_per_block and int(bcur) in st["block_near_random_frame_indices"]:
                cached = [int(x) for x in st["block_near_random_frame_indices"][int(bcur)]]
                if self._near_random_cached_frames_valid(
                    cached_frames=[int(x) for x in cached],
                    sidx=sidx,
                    source_keyframe_idx=int(source_keyframe_idx),
                    source_frame=int(source_frame),
                    existing_target_frames=[int(x) for x in base_target_frames],
                    num_frames=int(self.near_random_frames_per_block),
                ):
                    near_random_frames = [int(x) for x in cached]
                    reuse_cached = True
            if not reuse_cached:
                near_random_frames, num_candidates = self._sample_near_random_frames_for_block(
                    sidx=sidx,
                    source_keyframe_idx=int(source_keyframe_idx),
                    source_frame=int(source_frame),
                    existing_target_frames=[int(x) for x in base_target_frames],
                    num_frames=int(self.near_random_frames_per_block),
                )
                st["block_near_random_frame_indices"][int(bcur)] = [int(x) for x in near_random_frames]
                st["near_random_attempted_blocks"] = int(st.get("near_random_attempted_blocks", 0)) + 1
                st["near_random_candidate_frames_sum"] = float(st.get("near_random_candidate_frames_sum", 0.0)) + float(
                    num_candidates
                )
                if len(near_random_frames) > 0:
                    st["near_random_sampled_blocks"] = int(st.get("near_random_sampled_blocks", 0)) + 1
                else:
                    st["near_random_skipped_blocks"] = int(st.get("near_random_skipped_blocks", 0)) + 1
        if self.aux_feature_splat_enable:
            target_frames = [int(x) for x in base_target_frames]
            target_frame_roles = [str(x) for x in base_roles]
        else:
            target_frames = [int(x) for x in base_target_frames] + [int(x) for x in near_random_frames]
            target_frame_roles = [str(x) for x in base_roles] + [str(self.near_random_role_name) for _ in near_random_frames]
        source_image_ref = (int(source_frame), 0)
        source_image_refs = self._frame_targets_to_image_refs(num_cams, [int(source_frame)])
        target_image_refs = self._frame_targets_to_image_refs(num_cams, target_frames)
        target_image_roles = self._frame_roles_to_image_roles(
            num_cams=num_cams,
            target_frames=[int(x) for x in target_frames],
            target_frame_roles=[str(x) for x in target_frame_roles],
        )
        aux_source_cam = (
            self._select_aux_source_cam(source_image_refs=source_image_refs, num_cams=int(num_cams))
            if self.aux_feature_splat_enable
            else int(source_image_ref[1])
        )
        aux_image_refs, aux_image_roles = self._build_aux_image_refs_for_block(
            sidx=sidx,
            source_keyframe_idx=int(source_keyframe_idx),
            source_frame=int(source_frame),
            source_cam=int(aux_source_cam),
            base_target_frames=[int(x) for x in base_target_frames],
            num_cams=int(num_cams),
            exclude_image_refs=[tuple(x) for x in source_image_refs] + [tuple(x) for x in target_image_refs],
        )

        st["block_cursor"] = int(bcur)
        st["current_source_frame_idx"] = int(source_frame)
        st["current_target_frame_indices"] = [int(x) for x in target_frames]
        st["current_target_frame_roles"] = [str(x) for x in target_frame_roles]
        block_source_frames[int(bcur)] = int(source_frame)
        st["block_current_source_frame_indices"] = [int(x) for x in block_source_frames]
        st["source_keyframe_idx"] = int(source_keyframe_idx)
        st["source_image_ref"] = tuple(source_image_ref)
        st["source_image_refs"] = [tuple(x) for x in source_image_refs]
        st["aux_source_cam"] = int(aux_source_cam)
        st["target_image_refs"] = [tuple(x) for x in target_image_refs]
        st["target_image_roles"] = [str(x) for x in target_image_roles]
        st["aux_image_refs"] = [tuple(x) for x in aux_image_refs]
        st["aux_image_roles"] = [str(x) for x in aux_image_roles]
        st["block_last_target_frame_indices"][int(bcur)] = [int(x) for x in target_frames]
        st["block_target_frame_roles"][int(bcur)] = [str(x) for x in target_frame_roles]
        st["block_target_image_roles"][int(bcur)] = [str(x) for x in target_image_roles]
        if int(bcur) not in st["visited_block_indices"]:
            st["visited_block_indices"].add(int(bcur))
            st["block_first_visit_order"][int(bcur)] = int(st.get("episode_step_cursor", 0))
            st["block_first_target_frame_indices"][int(bcur)] = [int(x) for x in target_frames]

        if self._block_order_uses_episode_visit_order():
            st["block_repeat_step"] = int(st["block_update_counts"][bcur])
            st["block_idx_global"] = int(st["episode_base_block_idx_global"]) + int(bcur)
            st["block_idx_in_segment"] = int(st["episode_base_block_idx_in_segment"]) + int(bcur)
            if not bool(st["block_started"][bcur]):
                st["block_started"][bcur] = True
                self._emit_block_begin_for_current_state()
        else:
            st["block_repeat_step"] = 0
            st["block_idx_global"] = int(self._block_idx_global)
            self._emit_block_begin_for_current_state()
            self._block_idx_global += 1

    def _sample_near_random_frames_for_block(
        self,
        *,
        sidx: SegmentIndexLike,
        source_keyframe_idx: int,
        source_frame: int,
        existing_target_frames: List[int],
        num_frames: int,
    ) -> tuple[List[int], int]:
        if int(num_frames) <= 0:
            return [], 0
        if not self.near_random_same_keyframe_only:
            raise ValueError("v1 only supports same_keyframe_only=true")
        frames = [int(x) for x in list(sidx.keyframe_to_frames[int(source_keyframe_idx)])]
        existing = set(int(x) for x in existing_target_frames)
        candidates: List[int] = []
        for f in frames:
            if self.near_random_exclude_source and int(f) == int(source_frame):
                continue
            if self.near_random_exclude_existing and int(f) in existing:
                continue
            candidates.append(int(f))
        if len(candidates) == 0:
            return [], 0
        if len(candidates) < int(num_frames):
            if self.near_random_insufficient_policy != "skip":
                raise ValueError(f"unsupported near_random insufficient_policy={self.near_random_insufficient_policy!r}")
            return [], int(len(candidates))
        return [int(x) for x in self._sample_no_replace(candidates, int(num_frames))], int(len(candidates))

    @staticmethod
    def _frame_roles_to_image_roles(
        *,
        num_cams: int,
        target_frames: List[int],
        target_frame_roles: List[str],
    ) -> List[str]:
        if len(target_frames) != len(target_frame_roles):
            raise ValueError("target_frames and target_frame_roles length mismatch")
        out: List[str] = []
        for role in target_frame_roles:
            for _ in range(int(num_cams)):
                out.append(str(role))
        return out

    def _emit_block_begin_for_current_state(self) -> None:
        # V7 block-begin preload assumes `block_windows` exists, but V8 replaces it
        # with dynamic targets from `frame_chain`. Emit the shared block_begin event
        # without V7's next-block preload branch, then apply V8 preload below.
        original_warm_next_block_exact = bool(self.warm_next_block_exact)
        self.warm_next_block_exact = False
        try:
            super()._emit_block_begin_for_current_state()
        finally:
            self.warm_next_block_exact = original_warm_next_block_exact
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV8 internal state is not initialized")
        if original_warm_next_block_exact:
            refs = self._frame_targets_to_image_refs(
                int(st["num_cams"]),
                [int(x) for x in st.get("episode_source_candidate_frames", st["frame_chain"])],
            )
            self._emit_preload_hint(
                scene_id=int(st["scene_id"]),
                segment_id=int(st["segment_id"]),
                future_image_refs=refs,
                hint_scope="episode_chain_exact",
                block_idx_global=int(st["block_idx_global"]),
            )

    def _emit_block_end_for_block(self, st: Dict[str, Any], block_idx: int) -> None:
        frame_chain = [int(x) for x in st["frame_chain"]]
        block_source_frames = [int(x) for x in st.get("block_current_source_frame_indices", frame_chain)]
        bcur = int(block_idx)
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid block_idx={block_idx} for block_end")
        source_frame = self._resolve_block_source_frame_at_index(
            block_source_frames=[int(x) for x in block_source_frames],
            frame_chain=[int(x) for x in frame_chain],
            block_idx=int(bcur),
        )
        first_target_frames = st.get("block_first_target_frame_indices", {}).get(int(bcur))
        last_target_frames = st.get("block_last_target_frame_indices", {}).get(int(bcur))
        if last_target_frames is None:
            last_target_frames = self._build_target_frames_for_block_v8(
                frame_chain=frame_chain,
                block_source_frames=block_source_frames,
                block_idx=bcur,
                source_frame=int(source_frame),
                visited_block_indices=set(st.get("visited_block_indices", set())),
                max_target_frames=int(self.total_target_frames),
            )
        if first_target_frames is None:
            first_target_frames = [int(x) for x in last_target_frames]
        num_cams = int(st["num_cams"])
        source_image_ref = (int(source_frame), 0)
        target_image_refs = self._frame_targets_to_image_refs(num_cams, [int(x) for x in last_target_frames])
        if self._block_order_uses_episode_visit_order():
            num_updates_in_block = int(st["block_update_counts"][bcur])
        else:
            num_updates_in_block = int(st.get("block_repeat_step", self.steps_per_block))
        if self._block_order_uses_episode_visit_order():
            block_idx_global = int(st["episode_base_block_idx_global"]) + int(bcur)
            block_idx_in_segment = int(st["episode_base_block_idx_in_segment"]) + int(bcur)
        else:
            block_idx_global = int(st["block_idx_global"])
            block_idx_in_segment = int(st["block_idx_in_segment"])
        self._emit(
            {
                "type": "block_end",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(st["scene_id"]),
                "segment_id": int(st["segment_id"]),
                "episode_idx_global": int(st["episode_idx_global"]),
                "block_idx_in_episode": int(bcur),
                "block_idx_in_segment": int(block_idx_in_segment),
                "block_idx_global": int(block_idx_global),
                "source_frame_idx": int(source_frame),
                "source_image_ref": tuple(source_image_ref),
                "target_frame_indices": [int(x) for x in last_target_frames],
                "target_frame_indices_first_visit": [int(x) for x in first_target_frames],
                "target_frame_indices_last_visit": [int(x) for x in last_target_frames],
                "target_image_refs": [tuple(x) for x in target_image_refs],
                "num_updates_in_block": int(num_updates_in_block),
                "K_u_nominal": int(self.steps_per_block),
                "K_u_effective": int(self.steps_per_block),
                "K_steps_effective": int(self.steps_per_block),
                "U": int(self.U),
                "block_order": str(self.block_order),
                "scheduler_version": "v8",
            }
        )

    def _emit_block_exit_for_block(self, st: Dict[str, Any], block_idx: int) -> None:
        frame_chain = [int(x) for x in st["frame_chain"]]
        block_source_frames = [int(x) for x in st.get("block_current_source_frame_indices", frame_chain)]
        bcur = int(block_idx)
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid block_idx={block_idx} for block_exit")
        source_frame = self._resolve_block_source_frame_at_index(
            block_source_frames=[int(x) for x in block_source_frames],
            frame_chain=[int(x) for x in frame_chain],
            block_idx=int(bcur),
        )
        target_frames = st.get("block_last_target_frame_indices", {}).get(int(bcur))
        if target_frames is None:
            target_frames = self._build_target_frames_for_block_v8(
                frame_chain=frame_chain,
                block_source_frames=block_source_frames,
                block_idx=bcur,
                source_frame=int(source_frame),
                visited_block_indices=set(st.get("visited_block_indices", set())),
                max_target_frames=int(self.total_target_frames),
            )
        num_cams = int(st["num_cams"])
        source_image_ref = (int(source_frame), 0)
        target_image_refs = self._frame_targets_to_image_refs(num_cams, [int(x) for x in target_frames])
        if self._block_order_uses_episode_visit_order():
            block_idx_global = int(st["episode_base_block_idx_global"]) + int(bcur)
            block_idx_in_segment = int(st["episode_base_block_idx_in_segment"]) + int(bcur)
            num_updates_in_block = int(st["block_update_counts"][bcur])
        else:
            block_idx_global = int(st["block_idx_global"])
            block_idx_in_segment = int(st["block_idx_in_segment"])
            num_updates_in_block = int(st.get("block_repeat_step", self.steps_per_block))
        self._emit(
            {
                "type": "block_exit",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(st["scene_id"]),
                "segment_id": int(st["segment_id"]),
                "episode_idx_global": int(st["episode_idx_global"]),
                "block_idx_in_episode": int(bcur),
                "block_idx_in_segment": int(block_idx_in_segment),
                "block_idx_global": int(block_idx_global),
                "source_frame_idx": int(source_frame),
                "source_image_ref": tuple(source_image_ref),
                "target_frame_indices": [int(x) for x in target_frames],
                "target_image_refs": [tuple(x) for x in target_image_refs],
                "num_updates_in_block": int(num_updates_in_block),
                "K_u_nominal": int(self.steps_per_block),
                "K_u_effective": int(self.steps_per_block),
                "K_steps_effective": int(self.steps_per_block),
                "U": int(self.U),
                "block_order": str(self.block_order),
                "scheduler_version": "v8",
            }
        )

    def _aligned_info(self, st: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(super()._aligned_info(st))
        info["scheduler_version"] = "v8"
        info["target_policy"] = str(self.target_policy)
        info["history_target_policy"] = str(self.history_target_policy)
        info["block_source_frame_policy"] = str(self.block_source_frame_policy)
        info["episode_source_mode"] = str(getattr(self, "episode_source_mode", "keyframes"))
        info["visited_block_indices"] = sorted(int(x) for x in st.get("visited_block_indices", set()))
        info["block_current_source_frame_indices"] = [
            int(x) for x in st.get("block_current_source_frame_indices", st.get("frame_chain", []))
        ]
        info["block_first_visit_order"] = {
            int(k): int(v) for k, v in dict(st.get("block_first_visit_order", {})).items()
        }
        return info

    def _batch_from_state(self, st: Dict[str, Any]) -> Dict[str, Any]:
        source_refs = [(int(x[0]), int(x[1])) for x in st["source_image_refs"]]
        target_refs = [(int(x[0]), int(x[1])) for x in st["target_image_refs"]]
        aux_refs = [(int(x[0]), int(x[1])) for x in list(st.get("aux_image_refs", []))]
        if not hasattr(self.dataset, "_assemble_segment_batch_from_image_refs"):
            return super()._batch_from_state(st)
        return self.dataset._assemble_segment_batch_from_image_refs(
            int(st["scene_id"]),
            int(st["segment_id"]),
            source_refs,
            target_refs,
            aux_image_refs=aux_refs,
            include_test=bool(self.include_test),
            test_image_refs=None,
            enforce_target0_equals_source=True,
            target_ref_purpose="train",
        )

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        batch = super().materialize_current_batch_without_advance()
        aligned = dict(batch.get("_scheduler_v7_aligned_info") or batch.get("_scheduler_v4_aligned_info") or {})
        aligned["scheduler_version"] = "v8"
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_peek"] = True
        return batch

    def next_batch(self) -> Dict[str, Any]:
        self._ensure_episode_state()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV8 internal state is not initialized")
        batch = self._batch_from_state(st)
        request_meta = dict(batch.get("request_meta") or {})
        request_meta["target_frame_roles"] = list(st.get("current_target_frame_roles", []))
        request_meta["target_image_roles"] = list(st.get("target_image_roles", []))
        request_meta["aux_image_refs"] = list(st.get("aux_image_refs", []))
        request_meta["aux_image_roles"] = list(st.get("aux_image_roles", []))
        request_meta["aux_source_cam"] = int(st.get("aux_source_cam", 0))
        request_meta["aux_camera_policy"] = str(self.aux_feature_splat_camera_policy)
        request_meta["near_random_frame_indices"] = list(
            st.get("block_near_random_frame_indices", {}).get(int(st["block_cursor"]), [])
        )
        request_meta["near_random_supervision_enable"] = bool(self.near_random_enable)
        attempted = int(st.get("near_random_attempted_blocks", 0))
        skipped = int(st.get("near_random_skipped_blocks", 0))
        sampled = int(st.get("near_random_sampled_blocks", 0))
        candidate_sum = float(st.get("near_random_candidate_frames_sum", 0.0))
        request_meta["scheduler/near_random/enabled"] = float(1.0 if self.near_random_enable else 0.0)
        request_meta["scheduler/near_random/num_frames"] = float(len(request_meta["near_random_frame_indices"]))
        request_meta["scheduler/near_random/skip_ratio"] = float(skipped / max(attempted, 1))
        request_meta["scheduler/near_random/num_candidate_frames_mean"] = float(candidate_sum / max(attempted, 1))
        request_meta["scheduler/near_random/sampled_blocks"] = float(sampled)
        if request_meta.get("source_image_refs") is None:
            request_meta["source_image_refs"] = list(st.get("source_image_refs", []))
        if request_meta.get("target_image_refs") is None:
            request_meta["target_image_refs"] = list(st.get("target_image_refs", []))
        if request_meta.get("aux_image_refs") is None:
            request_meta["aux_image_refs"] = list(st.get("aux_image_refs", []))
        target_refs = request_meta.get("target_image_refs") or batch.get("target_image_refs") or []
        roles = request_meta.get("target_image_roles") or []
        if len(target_refs) != len(roles):
            raise ValueError(f"target_image_refs/target_image_roles mismatch: {len(target_refs)} vs {len(roles)}")
        aux_refs = request_meta.get("aux_image_refs") or batch.get("aux_image_refs") or []
        aux_roles = request_meta.get("aux_image_roles") or []
        if len(aux_roles) > 0 and len(aux_refs) != len(aux_roles):
            raise ValueError(f"aux_image_refs/aux_image_roles mismatch: {len(aux_refs)} vs {len(aux_roles)}")
        batch["request_meta"] = request_meta
        key = (int(st["scene_id"]), int(st["segment_id"]))
        rt = self._segment_runtime[key]

        rt["segment_local_step"] = int(rt["segment_local_step"]) + 1
        current_block_idx = int(st["block_cursor"])
        if self._block_order_uses_episode_visit_order():
            st["block_update_counts"][current_block_idx] = int(st["block_update_counts"][current_block_idx]) + 1
            st["block_repeat_step"] = int(st["block_update_counts"][current_block_idx])
            st["episode_step_cursor"] = int(st.get("episode_step_cursor", 0)) + 1
        else:
            st["block_repeat_step"] = int(st["block_repeat_step"]) + 1
        self.global_step += 1

        aligned = self._aligned_info(st)
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_aligned_info"] = dict(aligned)

        if self._block_order_uses_episode_visit_order():
            if (
                int(st["block_update_counts"][current_block_idx]) >= self.steps_per_block
                and not bool(st["block_ended"][current_block_idx])
            ):
                self._emit_block_end_for_block(st, current_block_idx)
                st["block_ended"][current_block_idx] = True
                rt["block_idx_in_segment"] = max(
                    int(rt["block_idx_in_segment"]),
                    int(st["episode_base_block_idx_in_segment"]) + int(current_block_idx) + 1,
                )
            episode_total_steps = int(self._episode_total_steps_from_state(st))
            if int(st.get("episode_step_cursor", 0)) >= int(episode_total_steps):
                # Final step has no "next block" switch, so emit a terminal block_exit
                # to keep block-exit-triggered record pass complete.
                self._emit_block_exit_for_block(st, current_block_idx)
                self._finalize_episode_if_needed()
            else:
                next_block_idx = int(self._episode_visit_order_from_state(st)[int(st["episode_step_cursor"])])
                if int(next_block_idx) != int(current_block_idx):
                    self._emit_block_exit_for_block(st, current_block_idx)
                self._select_block(next_block_idx)
        else:
            if int(st["block_repeat_step"]) >= self.steps_per_block:
                self._emit_block_exit_for_block(st, current_block_idx)
                self._emit_block_end_for_block(st, current_block_idx)
                rt["block_idx_in_segment"] = int(rt["block_idx_in_segment"]) + 1
                st["block_idx_in_segment"] = int(rt["block_idx_in_segment"])
                st["block_cursor"] = int(st["block_cursor"]) + 1
                st["block_repeat_step"] = 0
                if int(st["block_cursor"]) < int(self._episode_num_blocks_from_state(st)):
                    self._start_block()
                else:
                    self._finalize_episode_if_needed()

        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
        return batch

    def get_current_info(self) -> Dict[str, Any]:
        out = dict(super().get_current_info())
        out["scheduler_version"] = "v8"
        return out
