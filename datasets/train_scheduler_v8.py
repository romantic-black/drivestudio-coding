from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Set

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
        reset_policy: str = "episode_end",
        near_random_supervision_cfg: Optional[Any] = None,
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
        self.reset_policy = str(reset_policy)
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
        if self.near_random_enable:
            if self.near_random_frames_per_block < 1:
                raise ValueError("near_random_supervision.frames_per_block must be >= 1")
            if not self.near_random_same_keyframe_only:
                raise ValueError("v1 only supports near_random_supervision.same_keyframe_only=true")
            if self.near_random_insufficient_policy != "skip":
                raise ValueError("v1 only supports near_random_supervision.insufficient_policy=skip")
            if self.near_random_camera_policy != "all_cams":
                raise ValueError("v1 only supports near_random_supervision.camera_policy=all_cams")
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

    def start_new_epoch(self) -> None:
        if bool(getattr(self, "_skip_next_start_new_epoch", False)):
            self._skip_next_start_new_epoch = False
            return
        super().start_new_epoch()

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
        kfs = list(sidx.keyframe_indices)
        st = int(episode_start_keyframe_pos)
        ed = st + int(self.blocks_per_episode)
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
        rt = self._segment_runtime[key]
        rt["episodes_started"] = int(rt["episodes_started"]) + 1
        episode_base_block_idx_global = int(self._block_idx_global)
        episode_base_block_idx_in_segment = int(rt["block_idx_in_segment"])
        if self.block_order == "step_major":
            self._block_idx_global = int(self._block_idx_global) + int(self.blocks_per_episode)

        self.current_episode_state = {
            "scene_id": scene_id,
            "segment_id": segment_id,
            "episode_idx_global": int(self._episode_idx_global),
            "episode_start_keyframe_pos": int(plan.episode_start_keyframe_pos),
            "keyframe_window": keyframe_window,
            "frame_chain": frame_chain,
            "block_cursor": int(self._episode_block_visit_order[0]) if self.block_order == "step_major" else 0,
            "block_repeat_step": 0,
            "episode_step_cursor": 0,
            "block_update_counts": [0 for _ in range(int(self.blocks_per_episode))],
            "block_started": [False for _ in range(int(self.blocks_per_episode))],
            "block_ended": [False for _ in range(int(self.blocks_per_episode))],
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
                "num_pairs": int(self.blocks_per_episode),
                "scheduler_version": "v8",
            }
        )
        self._episode_idx_global += 1

        refs = self._frame_targets_to_image_refs(int(plan.num_cams), [int(x) for x in frame_chain])
        if self.block_order == "step_major":
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
                next_refs = self._frame_targets_to_image_refs(
                    int(next_plan.num_cams),
                    [int(x) for x in next_plan.frame_chain],
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
        block_idx: int,
        visited_block_indices: Set[int],
        max_target_frames: int,
    ) -> List[int]:
        source_frame = int(frame_chain[int(block_idx)])
        candidates = [int(b) for b in visited_block_indices if int(b) != int(block_idx)]
        prev_blocks = sorted([b for b in candidates if b < int(block_idx)], reverse=True)
        next_blocks = sorted([b for b in candidates if b > int(block_idx)])
        selected_blocks: List[int] = []
        for b in prev_blocks:
            if len(selected_blocks) >= int(max_target_frames) - 1:
                break
            selected_blocks.append(int(b))
        for b in next_blocks:
            if len(selected_blocks) >= int(max_target_frames) - 1:
                break
            selected_blocks.append(int(b))
        return [int(source_frame)] + [int(frame_chain[b]) for b in selected_blocks]

    def _select_block(self, block_idx: int) -> None:
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV8 internal state is not initialized")
        frame_chain = [int(x) for x in st["frame_chain"]]
        bcur = int(block_idx)
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid block_idx={bcur} for episode")

        base_target_frames = self._build_target_frames_for_block_v8(
            frame_chain=frame_chain,
            block_idx=bcur,
            visited_block_indices=set(st["visited_block_indices"]),
            max_target_frames=int(self.total_target_frames),
        )
        source_frame = int(frame_chain[bcur])
        base_roles = ["source"] + ["visited" for _ in base_target_frames[1:]]
        num_cams = int(st["num_cams"])
        self._set_current_scheduler_scope(int(st["scene_id"]), int(st["segment_id"]))
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        source_keyframe_idx = int(sidx.frame_to_keyframe[int(source_frame)])
        near_random_frames: List[int] = []
        if self.near_random_enable:
            if self.near_random_sample_once_per_block and int(bcur) in st["block_near_random_frame_indices"]:
                near_random_frames = [int(x) for x in st["block_near_random_frame_indices"][int(bcur)]]
            else:
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

        st["block_cursor"] = int(bcur)
        st["current_source_frame_idx"] = int(source_frame)
        st["current_target_frame_indices"] = [int(x) for x in target_frames]
        st["current_target_frame_roles"] = [str(x) for x in target_frame_roles]
        st["source_keyframe_idx"] = int(source_keyframe_idx)
        st["source_image_ref"] = tuple(source_image_ref)
        st["source_image_refs"] = [tuple(x) for x in source_image_refs]
        st["target_image_refs"] = [tuple(x) for x in target_image_refs]
        st["target_image_roles"] = [str(x) for x in target_image_roles]
        st["block_last_target_frame_indices"][int(bcur)] = [int(x) for x in target_frames]
        st["block_target_frame_roles"][int(bcur)] = [str(x) for x in target_frame_roles]
        st["block_target_image_roles"][int(bcur)] = [str(x) for x in target_image_roles]
        if int(bcur) not in st["visited_block_indices"]:
            st["visited_block_indices"].add(int(bcur))
            st["block_first_visit_order"][int(bcur)] = int(st.get("episode_step_cursor", 0))
            st["block_first_target_frame_indices"][int(bcur)] = [int(x) for x in target_frames]

        if self.block_order == "step_major":
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
                [int(x) for x in st["frame_chain"]],
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
        bcur = int(block_idx)
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid block_idx={block_idx} for block_end")
        first_target_frames = st.get("block_first_target_frame_indices", {}).get(int(bcur))
        last_target_frames = st.get("block_last_target_frame_indices", {}).get(int(bcur))
        if last_target_frames is None:
            last_target_frames = self._build_target_frames_for_block_v8(
                frame_chain=frame_chain,
                block_idx=bcur,
                visited_block_indices=set(st.get("visited_block_indices", set())),
                max_target_frames=int(self.total_target_frames),
            )
        if first_target_frames is None:
            first_target_frames = [int(x) for x in last_target_frames]
        source_frame = int(frame_chain[bcur])
        num_cams = int(st["num_cams"])
        source_image_ref = (int(source_frame), 0)
        target_image_refs = self._frame_targets_to_image_refs(num_cams, [int(x) for x in last_target_frames])
        if self.block_order == "step_major":
            num_updates_in_block = int(st["block_update_counts"][bcur])
        else:
            num_updates_in_block = int(st.get("block_repeat_step", self.steps_per_block))
        if self.block_order == "step_major":
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
        bcur = int(block_idx)
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid block_idx={block_idx} for block_exit")
        target_frames = st.get("block_last_target_frame_indices", {}).get(int(bcur))
        if target_frames is None:
            target_frames = self._build_target_frames_for_block_v8(
                frame_chain=frame_chain,
                block_idx=bcur,
                visited_block_indices=set(st.get("visited_block_indices", set())),
                max_target_frames=int(self.total_target_frames),
            )
        source_frame = int(frame_chain[bcur])
        num_cams = int(st["num_cams"])
        source_image_ref = (int(source_frame), 0)
        target_image_refs = self._frame_targets_to_image_refs(num_cams, [int(x) for x in target_frames])
        if self.block_order == "step_major":
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
        info["visited_block_indices"] = sorted(int(x) for x in st.get("visited_block_indices", set()))
        info["block_first_visit_order"] = {
            int(k): int(v) for k, v in dict(st.get("block_first_visit_order", {})).items()
        }
        return info

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
        target_refs = request_meta.get("target_image_refs") or batch.get("target_image_refs") or []
        roles = request_meta.get("target_image_roles") or []
        if len(target_refs) != len(roles):
            raise ValueError(f"target_image_refs/target_image_roles mismatch: {len(target_refs)} vs {len(roles)}")
        batch["request_meta"] = request_meta
        key = (int(st["scene_id"]), int(st["segment_id"]))
        rt = self._segment_runtime[key]

        rt["segment_local_step"] = int(rt["segment_local_step"]) + 1
        current_block_idx = int(st["block_cursor"])
        if self.block_order == "step_major":
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

        if self.block_order == "step_major":
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
            if int(st.get("episode_step_cursor", 0)) >= int(self.total_episode_steps):
                # Final step has no "next block" switch, so emit a terminal block_exit
                # to keep block-exit-triggered record pass complete.
                self._emit_block_exit_for_block(st, current_block_idx)
                self._finalize_episode_if_needed()
            else:
                next_block_idx = int(self._episode_block_visit_order[int(st["episode_step_cursor"])])
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
        out = dict(super().get_current_info())
        out["scheduler_version"] = "v8"
        return out
