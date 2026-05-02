from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets.train_scheduler_v7 import SegmentIndexLike, TrainSchedulerDatasetV7
from datasets.train_scheduler_v8 import EpisodePlanV8, TrainSchedulerV8


class TrainSchedulerV9(TrainSchedulerV8):
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
        block_source_frame_policy: str = "fixed_once_per_episode",
        role_sampling_cfg: Optional[Any] = None,
        targets_cfg: Optional[Any] = None,
        history_record_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
    ) -> None:
        self.role_sampling_cfg = role_sampling_cfg or {}
        self.targets_cfg = targets_cfg or {}
        self.history_record_cfg = history_record_cfg or {}
        self.preload_cfg = preload_cfg or {}
        self._student_cycle_ptr_by_block: Dict[int, int] = {}

        role_cfg = self.role_sampling_cfg
        self.first_step_role = str(self._cfg_get(role_cfg, "first_step_role", "teacher"))
        self.teacher_prob = float(self._cfg_get(role_cfg, "teacher_prob", 0.4))
        self.student_prob = float(self._cfg_get(role_cfg, "student_prob", 0.6))
        self.teacher_frame_policy = str(self._cfg_get(role_cfg, "teacher_frame_policy", "random_within_keyframe"))
        self.student_frame_policy = str(
            self._cfg_get(role_cfg, "student_frame_policy", "random_within_same_keyframe_except_teacher")
        )
        self.skip_student_if_single_source = bool(self._cfg_get(role_cfg, "skip_student_if_single_source", True))
        self.skip_student_if_no_prior = bool(self._cfg_get(role_cfg, "skip_student_if_no_prior", True))
        self.fallback_to_teacher = bool(self._cfg_get(role_cfg, "fallback_to_teacher", True))
        self.force_teacher_on_block_entry = bool(self._cfg_get(role_cfg, "force_teacher_on_block_entry", True))

        weights_cfg = self._cfg_get(self.targets_cfg, "weights", {}) or {}
        self.teacher_source_weight = float(self._cfg_get(weights_cfg, "teacher_source", 1.0))
        self.student_source_weight = float(self._cfg_get(weights_cfg, "student_source", 1.0))
        self.teacher_preserve_weight = float(self._cfg_get(weights_cfg, "teacher_preserve", 0.3))
        self.visited_weight = float(self._cfg_get(weights_cfg, "visited", 0.2))
        self.near_random_weight = float(self._cfg_get(weights_cfg, "near_random", 0.2))

        observed_cfg = self._cfg_get(self.history_record_cfg, "observed", {}) or {}
        self.observed_trigger = str(self._cfg_get(observed_cfg, "trigger", "teacher_exit"))
        self.observed_record_on_block_exit = bool(self._cfg_get(observed_cfg, "record_on_block_exit", False))
        self.runtime_cfg = self._cfg_get(self.history_record_cfg, "runtime", {}) or {}

        if str(target_policy) != "visited_episode_frames":
            raise ValueError("SchedulerV9 requires target_policy=visited_episode_frames")
        if str(reset_policy) != "episode_end":
            raise ValueError("SchedulerV9 requires reset_policy=episode_end")
        if not bool(include_source_frame):
            raise ValueError("SchedulerV9 requires include_source_frame=true")
        if self.first_step_role != "teacher":
            raise ValueError("SchedulerV9 requires first_step_role=teacher")
        if self.student_prob > 0.0 and self.teacher_prob <= 0.0:
            raise ValueError("SchedulerV9 requires teacher_prob > 0 when student_prob > 0")
        if self.observed_trigger != "teacher_exit":
            raise ValueError("SchedulerV9 requires observed history trigger=teacher_exit")
        if self.observed_record_on_block_exit:
            raise ValueError("SchedulerV9 must not record observed support/residual on block_exit")

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
            target_policy=target_policy,
            reset_policy=reset_policy,
            near_random_supervision_cfg=near_random_supervision_cfg,
            block_source_frame_policy=block_source_frame_policy,
        )

    def _emit(self, event: Dict[str, Any]) -> None:
        out = dict(event)
        if str(out.get("scheduler_version", "")) in ("v7", "v8"):
            out["scheduler_version"] = "v9"
        super()._emit(out)

    def _start_episode_from_plan(self, plan: EpisodePlanV8) -> None:
        super()._start_episode_from_plan(plan)
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV9 internal state is not initialized")
        self._student_cycle_ptr_by_block = {}
        st.update(
            {
                "scheduler_version": "v9",
                "block_teacher_frame_indices": [-1 for _ in range(self.blocks_per_episode)],
                "block_teacher_keyframe_indices": [-1 for _ in range(self.blocks_per_episode)],
                "block_student_candidate_frame_indices": [[] for _ in range(self.blocks_per_episode)],
                "block_last_teacher_frame_indices": [-1 for _ in range(self.blocks_per_episode)],
                "block_teacher_seen_counts": [0 for _ in range(self.blocks_per_episode)],
                "block_student_seen_counts": [0 for _ in range(self.blocks_per_episode)],
                "current_stage5_5_role": "teacher",
                "current_teacher_frame_idx": -1,
                "current_student_frame_idx": -1,
                "current_teacher_image_refs": [],
                "current_student_image_refs": [],
                "current_target_image_weights": [],
                "current_observed_record_mode": "none",
                "current_runtime_record_mode": "step_exit",
                "current_teacher_exit_record": False,
                "v9_role_teacher_count": 0,
                "v9_role_student_count": 0,
                "v9_student_skip_single_source_count": 0,
                "v9_student_skip_no_prior_count": 0,
                "v9_fallback_to_teacher_count": 0,
                "current_block_entry_step": 0,
                "current_block_entry_id": 0,
                "block_entry_counts": [0 for _ in range(self.blocks_per_episode)],
                "block_entry_teacher_counts": [0 for _ in range(self.blocks_per_episode)],
            }
        )

    def _sample_teacher_frame(self, *, frames: List[int], fallback_frame: int) -> int:
        candidates = [int(x) for x in frames]
        if self.teacher_frame_policy == "fixed_chain_frame":
            return int(fallback_frame)
        if self.teacher_frame_policy == "random_within_keyframe":
            return int(random.choice(candidates))
        raise ValueError(f"unsupported teacher_frame_policy={self.teacher_frame_policy!r}")

    def _sample_student_frame(self, *, student_candidates: List[int], teacher_frame: int, block_idx: int) -> int:
        candidates = [int(x) for x in student_candidates]
        if len(candidates) == 0:
            return int(teacher_frame)
        if self.student_frame_policy == "random_within_same_keyframe_except_teacher":
            return int(random.choice(candidates))
        if self.student_frame_policy == "cycle_within_same_keyframe_except_teacher":
            ptr = int(self._student_cycle_ptr_by_block.get(int(block_idx), 0))
            out = int(candidates[ptr % len(candidates)])
            self._student_cycle_ptr_by_block[int(block_idx)] = ptr + 1
            return int(out)
        raise ValueError(f"unsupported student_frame_policy={self.student_frame_policy!r}")

    def _weighted_role_sample(self) -> str:
        teacher_p = float(self.teacher_prob)
        student_p = float(self.student_prob)
        total = teacher_p + student_p
        if total <= 0.0:
            raise ValueError("teacher_prob + student_prob must be > 0")
        p_teacher = teacher_p / total
        return "teacher" if random.random() < p_teacher else "student"

    def _ensure_v9_block_role_state(self, *, st: Dict[str, Any], sidx: SegmentIndexLike, block_idx: int) -> None:
        b = int(block_idx)
        if int(st["block_teacher_frame_indices"][b]) >= 0:
            return

        keyframe_window = [int(x) for x in st["keyframe_window"]]
        frame_chain = [int(x) for x in st["frame_chain"]]
        source_keyframe_idx = int(keyframe_window[b])
        frames = [int(x) for x in list(sidx.keyframe_to_frames[int(source_keyframe_idx)])]
        if len(frames) == 0:
            raise ValueError(f"keyframe_to_frames[{int(source_keyframe_idx)}] is empty")
        teacher_frame = self._sample_teacher_frame(frames=frames, fallback_frame=int(frame_chain[b]))
        student_candidates = [int(f) for f in frames if int(f) != int(teacher_frame)]

        st["block_teacher_keyframe_indices"][b] = int(source_keyframe_idx)
        st["block_teacher_frame_indices"][b] = int(teacher_frame)
        st["block_last_teacher_frame_indices"][b] = int(teacher_frame)
        st["block_student_candidate_frame_indices"][b] = [int(x) for x in student_candidates]

    def _sample_v9_role_and_source(self, *, st: Dict[str, Any], block_idx: int) -> Tuple[str, int]:
        b = int(block_idx)
        teacher_frame = int(st["block_teacher_frame_indices"][b])
        student_candidates = [int(x) for x in st["block_student_candidate_frame_indices"][b]]
        num_updates = int(st["block_update_counts"][b])
        teacher_seen = int(st["block_teacher_seen_counts"][b]) > 0
        entry_step = int(st.get("current_block_entry_step", 0))

        if self.force_teacher_on_block_entry and entry_step == 0:
            return "teacher", int(teacher_frame)

        if num_updates == 0:
            return "teacher", int(teacher_frame)

        if len(student_candidates) == 0 and self.skip_student_if_single_source:
            st["v9_student_skip_single_source_count"] = int(st.get("v9_student_skip_single_source_count", 0)) + 1
            return "teacher", int(teacher_frame)

        if self.skip_student_if_no_prior and not teacher_seen:
            st["v9_student_skip_no_prior_count"] = int(st.get("v9_student_skip_no_prior_count", 0)) + 1
            return "teacher", int(teacher_frame)

        role = self._weighted_role_sample()
        if role == "teacher":
            return "teacher", int(teacher_frame)

        student_frame = int(
            self._sample_student_frame(
                student_candidates=[int(x) for x in student_candidates],
                teacher_frame=int(teacher_frame),
                block_idx=int(b),
            )
        )
        if student_frame == int(teacher_frame):
            if not self.fallback_to_teacher:
                raise ValueError("student sampling returned teacher frame and fallback_to_teacher=false")
            st["v9_fallback_to_teacher_count"] = int(st.get("v9_fallback_to_teacher_count", 0)) + 1
            return "teacher", int(teacher_frame)
        return "student", int(student_frame)

    def _resolve_visited_target_frames(self, st: Dict[str, Any], block_idx: int) -> List[int]:
        frame_chain = [int(x) for x in st["frame_chain"]]
        last_teachers = [int(x) for x in st.get("block_last_teacher_frame_indices", [])]
        visited_block_indices: Set[int] = set(int(x) for x in st.get("visited_block_indices", set()))
        candidates = [int(b) for b in visited_block_indices if int(b) != int(block_idx)]
        prev_blocks = sorted([b for b in candidates if b < int(block_idx)], reverse=True)
        next_blocks = sorted([b for b in candidates if b > int(block_idx)])
        out: List[int] = []
        for b in prev_blocks + next_blocks:
            if b < 0 or b >= len(frame_chain):
                raise ValueError(f"invalid visited block index={b} for frame_chain length={len(frame_chain)}")
            if b < len(last_teachers) and int(last_teachers[b]) >= 0:
                out.append(int(last_teachers[b]))
            else:
                out.append(int(frame_chain[b]))
        return out

    @staticmethod
    def _append_unique(
        target_frames: List[int],
        target_roles: List[str],
        target_weights: List[float],
        frame_idx: int,
        role: str,
        weight: float,
    ) -> None:
        fid = int(frame_idx)
        if fid in target_frames:
            return
        target_frames.append(int(fid))
        target_roles.append(str(role))
        target_weights.append(float(weight))

    @staticmethod
    def _frame_weights_to_image_weights(num_cams: int, target_frame_weights: List[float]) -> List[float]:
        out: List[float] = []
        for w in target_frame_weights:
            for _ in range(int(num_cams)):
                out.append(float(w))
        return out

    def _build_v9_target_frames_for_role(
        self,
        *,
        st: Dict[str, Any],
        block_idx: int,
        role: str,
        source_frame: int,
        teacher_frame: int,
    ) -> Tuple[List[int], List[str], List[float]]:
        visited_frames = self._resolve_visited_target_frames(st, int(block_idx))
        target_frames: List[int] = []
        target_roles: List[str] = []
        target_weights: List[float] = []

        if str(role) == "teacher":
            self._append_unique(
                target_frames,
                target_roles,
                target_weights,
                int(source_frame),
                "teacher_source",
                float(self.teacher_source_weight),
            )
            for f in visited_frames:
                self._append_unique(
                    target_frames,
                    target_roles,
                    target_weights,
                    int(f),
                    "visited",
                    float(self.visited_weight),
                )
        else:
            self._append_unique(
                target_frames,
                target_roles,
                target_weights,
                int(source_frame),
                "student_source",
                float(self.student_source_weight),
            )
            self._append_unique(
                target_frames,
                target_roles,
                target_weights,
                int(teacher_frame),
                "teacher_preserve",
                float(self.teacher_preserve_weight),
            )
            for f in visited_frames:
                self._append_unique(
                    target_frames,
                    target_roles,
                    target_weights,
                    int(f),
                    "visited",
                    float(self.visited_weight),
                )

        max_frames = int(self.total_target_frames)
        return (
            [int(x) for x in target_frames[:max_frames]],
            [str(x) for x in target_roles[:max_frames]],
            [float(x) for x in target_weights[:max_frames]],
        )

    def _select_block(self, block_idx: int) -> None:
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV9 internal state is not initialized")
        frame_chain = [int(x) for x in st["frame_chain"]]
        bcur = int(block_idx)
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid block_idx={bcur} for episode")
        prev_block = int(st.get("block_cursor", -1))

        self._set_current_scheduler_scope(int(st["scene_id"]), int(st["segment_id"]))
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        self._ensure_v9_block_role_state(st=st, sidx=sidx, block_idx=bcur)
        is_first_select = int(st.get("episode_step_cursor", 0)) == 0 and int(st["block_update_counts"][bcur]) == 0
        is_new_entry = bool(prev_block != bcur or is_first_select)
        if is_new_entry:
            st["current_block_entry_step"] = 0
            st["current_block_entry_id"] = int(st.get("current_block_entry_id", 0)) + 1
            entry_counts = [int(x) for x in st.get("block_entry_counts", [0 for _ in range(self.blocks_per_episode)])]
            if len(entry_counts) != int(self.blocks_per_episode):
                entry_counts = [0 for _ in range(self.blocks_per_episode)]
            entry_counts[bcur] = int(entry_counts[bcur]) + 1
            st["block_entry_counts"] = [int(x) for x in entry_counts]
        role, source_frame = self._sample_v9_role_and_source(st=st, block_idx=bcur)

        teacher_frame = int(st["block_teacher_frame_indices"][bcur])
        teacher_preserve_frame = int(st["block_last_teacher_frame_indices"][bcur])
        student_candidates = [int(x) for x in st["block_student_candidate_frame_indices"][bcur]]
        source_keyframe_idx = int(st["block_teacher_keyframe_indices"][bcur])

        target_frames, target_frame_roles, target_frame_weights = self._build_v9_target_frames_for_role(
            st=st,
            block_idx=bcur,
            role=str(role),
            source_frame=int(source_frame),
            teacher_frame=int(teacher_preserve_frame),
        )
        num_cams = int(st["num_cams"])
        near_random_frames: List[int] = []
        if self.near_random_enable:
            near_random_frames, num_candidates = self._sample_near_random_frames_for_block(
                sidx=sidx,
                source_keyframe_idx=int(source_keyframe_idx),
                source_frame=int(source_frame),
                existing_target_frames=[int(x) for x in target_frames],
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

        for f in near_random_frames:
            self._append_unique(
                target_frames,
                target_frame_roles,
                target_frame_weights,
                int(f),
                str(self.near_random_role_name),
                float(self.near_random_weight),
            )

        source_image_ref = (int(source_frame), 0)
        source_image_refs = self._frame_targets_to_image_refs(num_cams, [int(source_frame)])
        teacher_image_refs = self._frame_targets_to_image_refs(num_cams, [int(teacher_frame)])
        target_image_refs = self._frame_targets_to_image_refs(num_cams, [int(x) for x in target_frames])
        target_image_roles = self._frame_roles_to_image_roles(
            num_cams=num_cams,
            target_frames=[int(x) for x in target_frames],
            target_frame_roles=[str(x) for x in target_frame_roles],
        )
        target_image_weights = self._frame_weights_to_image_weights(
            num_cams=num_cams,
            target_frame_weights=[float(x) for x in target_frame_weights],
        )

        st["block_cursor"] = int(bcur)
        st["current_stage5_5_role"] = str(role)
        st["current_source_frame_idx"] = int(source_frame)
        st["current_teacher_frame_idx"] = int(teacher_frame)
        st["current_student_frame_idx"] = int(source_frame) if str(role) == "student" else -1
        st["current_teacher_image_refs"] = [tuple(x) for x in teacher_image_refs]
        st["current_student_image_refs"] = [tuple(x) for x in source_image_refs] if str(role) == "student" else []
        st["current_teacher_exit_record"] = bool(str(role) == "teacher")
        st["current_observed_record_mode"] = "teacher_exit" if str(role) == "teacher" else "none"
        st["current_runtime_record_mode"] = "step_exit"
        st["current_target_frame_indices"] = [int(x) for x in target_frames]
        st["current_target_frame_roles"] = [str(x) for x in target_frame_roles]
        st["current_target_frame_weights"] = [float(x) for x in target_frame_weights]
        st["current_target_image_weights"] = [float(x) for x in target_image_weights]
        block_source_frames = [int(x) for x in st.get("block_current_source_frame_indices", frame_chain)]
        if len(block_source_frames) != len(frame_chain):
            block_source_frames = [int(x) for x in frame_chain]
        block_source_frames[int(bcur)] = int(source_frame)
        st["block_current_source_frame_indices"] = [int(x) for x in block_source_frames]
        st["source_keyframe_idx"] = int(source_keyframe_idx)
        st["source_image_ref"] = tuple(source_image_ref)
        st["source_image_refs"] = [tuple(x) for x in source_image_refs]
        st["target_image_refs"] = [tuple(x) for x in target_image_refs]
        st["target_image_roles"] = [str(x) for x in target_image_roles]
        st["block_last_target_frame_indices"][int(bcur)] = [int(x) for x in target_frames]
        st["block_target_frame_roles"][int(bcur)] = [str(x) for x in target_frame_roles]
        st["block_target_image_roles"][int(bcur)] = [str(x) for x in target_image_roles]
        st["block_target_image_loss_base_weights"][int(bcur)] = [float(x) for x in target_image_weights]
        st["current_v9_student_candidates"] = [int(x) for x in student_candidates]
        if int(bcur) not in st.get("block_first_visit_order", {}):
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

    def _build_v9_request_meta(
        self,
        *,
        st: Dict[str, Any],
        block_idx: int,
        role: str,
        source_frame: int,
        teacher_frame: int,
        last_teacher_frame: int,
        student_candidates: List[int],
        source_image_refs: List[Tuple[int, int]],
        teacher_image_refs: List[Tuple[int, int]],
        target_frames: List[int],
        target_image_refs: List[Tuple[int, int]],
        target_roles: List[str],
        target_weights: List[float],
    ) -> Dict[str, Any]:
        is_teacher = str(role) == "teacher"
        is_student = str(role) == "student"
        entry_step = int(st.get("current_block_entry_step", 0))
        is_entry_first_step = entry_step == 0
        current_block = int(block_idx)
        entry_counts = [int(x) for x in st.get("block_entry_counts", [])]
        entry_teacher_counts = [int(x) for x in st.get("block_entry_teacher_counts", [])]
        entry_count_cur = int(entry_counts[current_block]) if current_block < len(entry_counts) else 0
        entry_teacher_count_cur = int(entry_teacher_counts[current_block]) if current_block < len(entry_teacher_counts) else 0
        total_roles = int(st.get("v9_role_teacher_count", 0)) + int(st.get("v9_role_student_count", 0))
        teacher_ratio = float(int(st.get("v9_role_teacher_count", 0)) / max(total_roles, 1))
        student_ratio = float(int(st.get("v9_role_student_count", 0)) / max(total_roles, 1))
        near_random_count = sum(1 for x in target_roles if str(x) == str(self.near_random_role_name))
        target_frame_weights = [float(x) for x in target_weights]
        if len(target_frame_weights) != len(target_frames):
            raise ValueError(
                f"target_frames/target_frame_weights mismatch: {len(target_frames)} vs {len(target_frame_weights)}"
            )
        target_image_roles = self._frame_roles_to_image_roles(
            num_cams=int(st["num_cams"]),
            target_frames=[int(x) for x in target_frames],
            target_frame_roles=[str(x) for x in target_roles],
        )
        target_image_weights = self._frame_weights_to_image_weights(
            num_cams=int(st["num_cams"]),
            target_frame_weights=[float(x) for x in target_frame_weights],
        )
        if len(target_image_weights) != len(target_image_refs):
            raise ValueError(
                f"target_image_refs/target_image_weights mismatch: {len(target_image_refs)} vs {len(target_image_weights)}"
            )
        if len(target_image_roles) != len(target_image_refs):
            raise ValueError(
                f"target_image_refs/target_image_roles mismatch: {len(target_image_refs)} vs {len(target_image_roles)}"
            )

        return {
            "scheduler_version": "v9",
            "stage5_5_role": str(role),
            "stage5_5_is_teacher": bool(is_teacher),
            "stage5_5_is_student": bool(is_student),
            "stage5_5_teacher_frame_idx": int(teacher_frame),
            "stage5_5_last_teacher_frame_idx": int(last_teacher_frame),
            "stage5_5_student_frame_idx": int(source_frame) if is_student else -1,
            "stage5_5_source_frame_idx": int(source_frame),
            "stage5_5_teacher_image_refs": [tuple(x) for x in teacher_image_refs],
            "stage5_5_student_candidate_frames": [int(x) for x in student_candidates],
            "stage5_5_has_student": bool(len(student_candidates) > 0),
            "stage5_5_block_entry_step": int(entry_step),
            "stage5_5_force_teacher_on_block_entry": bool(self.force_teacher_on_block_entry),
            "stage5_5_is_first_step_in_block": bool(is_entry_first_step),
            "stage5_5_is_first_update_in_block": int(st["block_update_counts"][int(block_idx)]) == 0,
            "target_frame_indices": [int(x) for x in target_frames],
            "target_image_refs": [tuple(x) for x in target_image_refs],
            "target_frame_roles": [str(x) for x in target_roles],
            "target_image_roles": [str(x) for x in target_image_roles],
            "target_frame_loss_base_weights": [float(x) for x in target_frame_weights],
            "target_image_loss_base_weights": [float(x) for x in target_image_weights],
            # Keep backward compatibility for older consumers.
            "target_loss_base_weights": [float(x) for x in target_frame_weights],
            "history_record/observed_record_trigger": "teacher_exit" if is_teacher else "none",
            "history_record/record_observed_on_step_exit": bool(is_teacher),
            "history_record/observed_record_image_refs": [tuple(x) for x in teacher_image_refs] if is_teacher else [],
            "history_record/observed_record_frame_idx": int(teacher_frame) if is_teacher else -1,
            "history_record/runtime_record_trigger": "step_exit",
            "history_record/record_runtime_on_step_exit": True,
            "history_record/runtime_record_image_refs": [tuple(x) for x in source_image_refs],
            "history_record/runtime_record_frame_idx": int(source_frame),
            "history_record/record_observed_on_block_exit": False,
            "history_record/block_exit_observed_record_disabled": True,
            "scheduler_v9/role_teacher": float(1.0 if is_teacher else 0.0),
            "scheduler_v9/role_student": float(1.0 if is_student else 0.0),
            "scheduler_v9/teacher_ratio": float(teacher_ratio),
            "scheduler_v9/student_ratio": float(student_ratio),
            "scheduler_v9/student_skip_single_source": float(st.get("v9_student_skip_single_source_count", 0)),
            "scheduler_v9/student_skip_no_prior": float(st.get("v9_student_skip_no_prior_count", 0)),
            "scheduler_v9/fallback_to_teacher": float(st.get("v9_fallback_to_teacher_count", 0)),
            "scheduler_v9/teacher_frame_idx": float(teacher_frame),
            "scheduler_v9/student_frame_idx": float(source_frame if is_student else -1),
            "scheduler_v9/num_student_candidates": float(len(student_candidates)),
            "scheduler_v9/target_num_frames": float(len(target_frames)),
            "scheduler_v9/target_has_teacher_preserve": float(1.0 if "teacher_preserve" in target_roles else 0.0),
            "scheduler_v9/target_num_visited": float(sum(1 for x in target_roles if str(x) == "visited")),
            "scheduler_v9/target_num_near_random": float(near_random_count),
            "scheduler_v9/observed_record_teacher_exit": float(1.0 if is_teacher else 0.0),
            "scheduler_v9/observed_record_block_exit_disabled": 1.0,
            "scheduler_v9/runtime_record_step_exit": 1.0,
            "scheduler_v9/block_entry_step": float(entry_step),
            "scheduler_v9/block_entry_teacher": float(1.0 if is_teacher and is_entry_first_step else 0.0),
            "scheduler_v9/block_entry_count": float(entry_count_cur),
            "scheduler_v9/block_entry_teacher_count": float(entry_teacher_count_cur),
            "scheduler_v9/force_teacher_on_block_entry": float(1.0 if self.force_teacher_on_block_entry else 0.0),
        }

    def _advance_after_emitting_current_batch_v9(self, st: Dict[str, Any], current_block_idx: int) -> None:
        b = int(current_block_idx)
        role = str(st.get("current_stage5_5_role", "teacher"))
        prev_entry_step = int(st.get("current_block_entry_step", 0))
        st["block_update_counts"][b] = int(st["block_update_counts"][b]) + 1
        st["episode_step_cursor"] = int(st.get("episode_step_cursor", 0)) + 1
        st["current_block_entry_step"] = int(prev_entry_step) + 1
        st["block_repeat_step"] = int(st["block_update_counts"][b]) if self.block_order == "step_major" else int(
            st.get("block_repeat_step", 0)
        ) + 1

        if role == "teacher":
            st["block_teacher_seen_counts"][b] = int(st["block_teacher_seen_counts"][b]) + 1
            st["block_last_teacher_frame_indices"][b] = int(st.get("current_teacher_frame_idx", -1))
            st["v9_role_teacher_count"] = int(st.get("v9_role_teacher_count", 0)) + 1
            if int(prev_entry_step) == 0:
                entry_teacher_counts = [
                    int(x) for x in st.get("block_entry_teacher_counts", [0 for _ in range(self.blocks_per_episode)])
                ]
                if len(entry_teacher_counts) != int(self.blocks_per_episode):
                    entry_teacher_counts = [0 for _ in range(self.blocks_per_episode)]
                entry_teacher_counts[b] = int(entry_teacher_counts[b]) + 1
                st["block_entry_teacher_counts"] = [int(x) for x in entry_teacher_counts]
        else:
            st["block_student_seen_counts"][b] = int(st["block_student_seen_counts"][b]) + 1
            st["v9_role_student_count"] = int(st.get("v9_role_student_count", 0)) + 1

        if int(st["block_update_counts"][b]) == 1:
            st["visited_block_indices"].add(int(b))

    def _aligned_info(self, st: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(super()._aligned_info(st))
        info["scheduler_version"] = "v9"
        info["stage5_5_role"] = str(st.get("current_stage5_5_role", "teacher"))
        info["stage5_5_block_entry_step"] = int(st.get("current_block_entry_step", 0))
        info["stage5_5_force_teacher_on_block_entry"] = bool(self.force_teacher_on_block_entry)
        info["teacher_frame_idx"] = int(st.get("current_teacher_frame_idx", -1))
        info["student_frame_idx"] = int(st.get("current_student_frame_idx", -1))
        target_frame_weights = [float(x) for x in st.get("current_target_frame_weights", [])]
        target_image_weights = self._frame_weights_to_image_weights(
            num_cams=int(st.get("num_cams", 0)),
            target_frame_weights=[float(x) for x in target_frame_weights],
        )
        info["target_frame_loss_base_weights"] = [float(x) for x in target_frame_weights]
        info["target_image_loss_base_weights"] = [float(x) for x in target_image_weights]
        # Keep backward compatibility for older consumers.
        info["target_loss_base_weights"] = [float(x) for x in target_frame_weights]
        return info

    def _inject_v9_request_meta_for_current_state(self, *, batch: Dict[str, Any], st: Dict[str, Any]) -> None:
        request_meta = dict(batch.get("request_meta") or {})
        role = str(st.get("current_stage5_5_role", "teacher"))
        block_idx = int(st["block_cursor"])
        source_frame = int(st["current_source_frame_idx"])
        teacher_frame = int(st.get("current_teacher_frame_idx", -1))
        last_teacher_frame = int(st["block_last_teacher_frame_indices"][block_idx])
        source_image_refs = [tuple(x) for x in st.get("source_image_refs", [])]
        teacher_image_refs = [tuple(x) for x in st.get("current_teacher_image_refs", [])]
        target_frames = [int(x) for x in st.get("current_target_frame_indices", [])]
        target_roles = [str(x) for x in st.get("current_target_frame_roles", [])]
        target_weights = [float(x) for x in st.get("current_target_frame_weights", [])]
        target_image_refs = [tuple(x) for x in st.get("target_image_refs", [])]
        student_candidates = [int(x) for x in st.get("current_v9_student_candidates", [])]

        request_meta.update(
            self._build_v9_request_meta(
                st=st,
                block_idx=int(block_idx),
                role=str(role),
                source_frame=int(source_frame),
                teacher_frame=int(teacher_frame),
                last_teacher_frame=int(last_teacher_frame),
                student_candidates=[int(x) for x in student_candidates],
                source_image_refs=[tuple(x) for x in source_image_refs],
                teacher_image_refs=[tuple(x) for x in teacher_image_refs],
                target_frames=[int(x) for x in target_frames],
                target_image_refs=[tuple(x) for x in target_image_refs],
                target_roles=[str(x) for x in target_roles],
                target_weights=[float(x) for x in target_weights],
            )
        )

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
            request_meta["source_image_refs"] = [tuple(x) for x in source_image_refs]
        if request_meta.get("target_image_refs") is None:
            request_meta["target_image_refs"] = [tuple(x) for x in target_image_refs]
        target_refs = request_meta.get("target_image_refs") or batch.get("target_image_refs") or []
        image_roles = request_meta.get("target_image_roles") or []
        if len(target_refs) != len(image_roles):
            raise ValueError(f"target_image_refs/target_image_roles mismatch: {len(target_refs)} vs {len(image_roles)}")
        frame_targets = request_meta.get("target_frame_indices") or []
        frame_weights = request_meta.get("target_frame_loss_base_weights") or []
        image_weights = request_meta.get("target_image_loss_base_weights") or []
        if len(frame_targets) != len(frame_weights):
            raise ValueError(
                f"target_frame_indices/target_frame_loss_base_weights mismatch: "
                f"{len(frame_targets)} vs {len(frame_weights)}"
            )
        if len(target_refs) != len(image_weights):
            raise ValueError(
                f"target_image_refs/target_image_loss_base_weights mismatch: "
                f"{len(target_refs)} vs {len(image_weights)}"
            )
        batch["request_meta"] = request_meta

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        batch = super().materialize_current_batch_without_advance()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV9 internal state is not initialized")
        self._inject_v9_request_meta_for_current_state(batch=batch, st=st)
        aligned = dict(
            batch.get("_scheduler_v8_aligned_info")
            or batch.get("_scheduler_v7_aligned_info")
            or batch.get("_scheduler_v4_aligned_info")
            or {}
        )
        aligned["scheduler_version"] = "v9"
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_aligned_info"] = dict(aligned)
        batch["_scheduler_v9_aligned_info"] = dict(aligned)
        batch["_scheduler_v9_peek"] = True
        return batch

    def next_batch(self) -> Dict[str, Any]:
        self._ensure_episode_state()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV9 internal state is not initialized")
        batch = self._batch_from_state(st)
        self._inject_v9_request_meta_for_current_state(batch=batch, st=st)

        key = (int(st["scene_id"]), int(st["segment_id"]))
        rt = self._segment_runtime[key]
        rt["segment_local_step"] = int(rt["segment_local_step"]) + 1
        current_block_idx = int(st["block_cursor"])
        self._advance_after_emitting_current_batch_v9(st, current_block_idx)
        self.global_step += 1

        aligned = self._aligned_info(st)
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_aligned_info"] = dict(aligned)
        batch["_scheduler_v9_aligned_info"] = dict(aligned)

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
        out["scheduler_version"] = "v9"
        return out
