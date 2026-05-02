from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from datasets.stage6_step_types import (
    BlockStateV10,
    EpisodePlanV10,
    EpisodeStateV10,
    ImageRef,
    Stage6StepRequest,
    Stage6StepType,
    Stage6StudentPropRequest,
    Stage6SupervisionRequest,
    Stage6TeacherObsRequest,
    validate_stage6_step_request,
    validate_teacher_obs_invariants,
)
from datasets.train_scheduler_v7 import _BatchRequestV7Compat, SegmentIndexLike, TrainSchedulerDatasetV7, TrainSchedulerV7


class TrainSchedulerV10(TrainSchedulerV7):
    def __init__(
        self,
        *,
        dataset: TrainSchedulerDatasetV7,
        steps_per_block: int,
        blocks_per_episode: int,
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
        frame_selection_cfg: Optional[Any] = None,
        step_program_cfg: Optional[Any] = None,
        supervision_cfg: Optional[Any] = None,
        history_record_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        bridge_cfg: Optional[Any] = None,
        probe_cfg: Optional[Any] = None,
        training_phase: str = "default",
        # Legacy V9-only args (accepted only for hard-fail validation).
        total_target_frames: Optional[int] = None,
        target_policy: Optional[str] = None,
        reset_policy: Optional[str] = None,
        near_random_supervision_cfg: Optional[Any] = None,
        block_source_frame_policy: Optional[str] = None,
        role_sampling_cfg: Optional[Any] = None,
        targets_cfg: Optional[Any] = None,
        camera_sampling_cfg: Optional[Any] = None,
    ) -> None:
        self.frame_selection_cfg = dict(frame_selection_cfg or {})
        self.step_program_cfg = dict(step_program_cfg or {})
        self.supervision_cfg = dict(supervision_cfg or {})
        self.history_record_cfg = dict(history_record_cfg or {})
        self.preload_cfg = dict(preload_cfg or {})
        self.bridge_cfg = dict(bridge_cfg or {})
        self.probe_cfg = dict(probe_cfg or {})
        self.training_phase = str(training_phase)
        self._fallback_no_student_count = 0
        self._fallback_no_history_count = 0
        self._probe_near_empty_count = 0
        self.current_episode_plan_v10: Optional[EpisodePlanV10] = None
        self.episode_state_v10 = EpisodeStateV10()
        self.block_states_v10: List[BlockStateV10] = []
        self._last_step_request: Optional[Stage6StepRequest] = None
        self._last_resolved_step_type: Optional[Stage6StepType] = None

        self._validate_cfg(
            total_target_frames=total_target_frames,
            target_policy=target_policy,
            reset_policy=reset_policy,
            near_random_supervision_cfg=near_random_supervision_cfg,
            block_source_frame_policy=block_source_frame_policy,
            role_sampling_cfg=role_sampling_cfg,
            targets_cfg=targets_cfg,
            camera_sampling_cfg=camera_sampling_cfg,
        )
        self.step_sequence = [Stage6StepType(str(x)) for x in list(self.step_program_cfg.get("sequence") or [])]
        if len(self.step_sequence) != int(steps_per_block):
            raise ValueError(
                "scheduler_v10.step_program.sequence length must equal block.steps_per_block, "
                f"got {len(self.step_sequence)} vs {int(steps_per_block)}"
            )
        self._skip_next_start_new_epoch = True
        super().__init__(
            dataset=dataset,
            steps_per_block=int(steps_per_block),
            blocks_per_episode=int(blocks_per_episode),
            total_target_frames=1,
            include_source_frame=bool(include_source_frame),
            frame_within_keyframe_policy=str(frame_within_keyframe_policy),
            min_keyframes_required_policy=str(min_keyframes_required_policy),
            traversal_mode=str(traversal_mode),
            switch_after_episode=bool(switch_after_episode),
            segment_order=str(segment_order),
            scene_order=str(scene_order),
            include_test=bool(include_test),
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            emit_preload_hints=bool(emit_preload_hints),
            warm_next_block_exact=bool(warm_next_block_exact),
            warm_next_episode_chain=bool(warm_next_episode_chain),
            block_order=str(block_order),
            step_major_switch_interval_steps=int(step_major_switch_interval_steps),
        )
        # V10 episode window is exactly blocks_per_episode keyframes.
        self.episode_window_keyframes = int(self.blocks_per_episode)
        self.start_new_epoch()

    @staticmethod
    def _cfg_get(node: Any, key: str, default: Any) -> Any:
        if node is None:
            return default
        if hasattr(node, "get"):
            out = node.get(key)
            return default if out is None else out
        if isinstance(node, dict):
            out = node.get(key)
            return default if out is None else out
        if hasattr(node, key):
            out = getattr(node, key)
            return default if out is None else out
        return default

    def _validate_cfg(
        self,
        *,
        total_target_frames: Optional[int],
        target_policy: Optional[str],
        reset_policy: Optional[str],
        near_random_supervision_cfg: Optional[Any],
        block_source_frame_policy: Optional[str],
        role_sampling_cfg: Optional[Any],
        targets_cfg: Optional[Any],
        camera_sampling_cfg: Optional[Any],
    ) -> None:
        # Hard-fail for legacy keys/inputs.
        if role_sampling_cfg not in (None, {}):
            raise ValueError("scheduler_v10 forbids role_sampling; use step_program + frame_selection.")
        if near_random_supervision_cfg not in (None, {}):
            raise ValueError("scheduler_v10 forbids near_random_supervision; use supervision.probe_near.")
        if targets_cfg not in (None, {}):
            raise ValueError("scheduler_v10 forbids targets.weights; use losses.stage6_0.")
        if camera_sampling_cfg not in (None, {}):
            raise ValueError("scheduler_v10 camera_sampling is not supported in V10 structured scheduler yet.")
        if block_source_frame_policy not in (None, "fixed_once_per_episode"):
            raise ValueError("scheduler_v10 forbids block_source_frame_policy override.")
        if target_policy not in (None, "visited_episode_frames"):
            raise ValueError("scheduler_v10 requires target_policy=visited_episode_frames.")
        if reset_policy not in (None, "episode_end"):
            raise ValueError("scheduler_v10 requires reset_policy=episode_end.")
        if total_target_frames not in (None, 1):
            raise ValueError("scheduler_v10 no longer uses total_target_frames; expected omitted or 1.")
        if str(self._cfg_get(self.step_program_cfg, "mode", "fixed_cycle")) != "fixed_cycle":
            raise ValueError("scheduler_v10.step_program.mode must be fixed_cycle.")

    def start_new_epoch(self) -> None:
        if bool(getattr(self, "_skip_next_start_new_epoch", False)):
            self._skip_next_start_new_epoch = False
            return
        super().start_new_epoch()

    def _emit(self, event: Dict[str, Any]) -> None:
        out = dict(event)
        out["scheduler_version"] = "v10"
        super()._emit(out)

    def _build_episode_plan(
        self,
        sidx: SegmentIndexLike,
        *,
        scene_id: int,
        segment_id: int,
        episode_start_keyframe_pos: int,
    ) -> EpisodePlanV10:
        kfs = [int(x) for x in sidx.keyframe_indices]
        st = int(episode_start_keyframe_pos)
        ed = int(st + self.blocks_per_episode)
        if ed > len(kfs):
            raise ValueError(
                "episode window out of range: "
                f"start={st}, W={self.blocks_per_episode}, num_kf={len(kfs)}"
            )
        keyframe_window = [int(x) for x in kfs[st:ed]]
        frame_chain = [
            self._choose_frame_for_keyframe_once([int(x) for x in list(sidx.keyframe_to_frames[int(kf)])])
            for kf in keyframe_window
        ]
        block_keyframe_indices = [int(x) for x in keyframe_window]
        teacher_frame_by_block: Dict[int, int] = {}
        student_candidates_by_block: Dict[int, List[int]] = {}
        probe_near_candidates_by_block: Dict[int, List[int]] = {}
        teacher_policy = str(self._cfg_get(self.frame_selection_cfg, "teacher_frame_policy", "random_within_keyframe"))
        for b, kf in enumerate(block_keyframe_indices):
            frames = [int(x) for x in list(sidx.keyframe_to_frames[int(kf)])]
            if len(frames) == 0:
                raise ValueError(f"empty keyframe_to_frames for keyframe={kf}")
            if teacher_policy == "random_within_keyframe":
                teacher_frame = int(random.choice(frames))
            elif teacher_policy == "middle_frame":
                teacher_frame = int(frames[len(frames) // 2])
            else:
                raise ValueError(f"unsupported scheduler_v10.frame_selection.teacher_frame_policy={teacher_policy!r}")
            teacher_frame_by_block[int(b)] = int(teacher_frame)
            students = [int(x) for x in frames if int(x) != int(teacher_frame)]
            student_candidates_by_block[int(b)] = [int(x) for x in students]
            probe_near_candidates_by_block[int(b)] = [int(x) for x in students]
        return EpisodePlanV10(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_start_keyframe_pos=int(st),
            keyframe_window=keyframe_window,
            frame_chain=[int(x) for x in frame_chain],
            num_cams=int(sidx.num_cams),
            block_keyframe_indices=block_keyframe_indices,
            teacher_frame_by_block=teacher_frame_by_block,
            student_candidates_by_block=student_candidates_by_block,
            probe_near_candidates_by_block=probe_near_candidates_by_block,
        )

    def _start_episode_from_plan(self, plan: EpisodePlanV10) -> None:
        super()._start_episode_from_plan(
            # keep v7 event/segment/runtime behavior
            type("EpisodePlanV7Compat", (), {
                "scene_id": int(plan.scene_id),
                "segment_id": int(plan.segment_id),
                "episode_start_keyframe_pos": int(plan.episode_start_keyframe_pos),
                "keyframe_window": [int(x) for x in plan.keyframe_window],
                "frame_chain": [int(x) for x in plan.frame_chain],
                "block_windows": [[int(plan.frame_chain[b])] for b in range(int(self.blocks_per_episode))],
                "num_cams": int(plan.num_cams),
            })()
        )
        self.current_episode_plan_v10 = plan
        self.episode_state_v10 = EpisodeStateV10()
        self.block_states_v10 = [
            BlockStateV10(
                block_idx=int(b),
                keyframe_idx=int(plan.block_keyframe_indices[b]),
                teacher_frame_idx=int(plan.teacher_frame_by_block[int(b)]),
                student_candidates=[int(x) for x in plan.student_candidates_by_block.get(int(b), [])],
            )
            for b in range(int(self.blocks_per_episode))
        ]

    def _active_cam_ids(self, num_cams: int) -> List[int]:
        return [int(x) for x in range(int(num_cams))]

    @staticmethod
    def _refs_for_frame(frame_idx: int, cam_ids: List[int]) -> List[ImageRef]:
        return [(int(frame_idx), int(c)) for c in cam_ids]

    def _pick_student_frame(self, block: BlockStateV10) -> Optional[int]:
        if len(block.student_candidates) == 0:
            return None
        cycle = str(self._cfg_get(self.frame_selection_cfg, "student_cycle_policy", "cycle"))
        if cycle == "cycle":
            idx = int(block.step_idx % len(block.student_candidates))
            return int(block.student_candidates[idx])
        return int(random.choice(block.student_candidates))

    def _pick_probe_frames(self, block_idx: int) -> List[int]:
        if not bool(self._cfg_get(self.supervision_cfg, "probe_near", {}).get("enable", False)):
            return []
        cfg = self._cfg_get(self.supervision_cfg, "probe_near", {}) or {}
        n = int(self._cfg_get(cfg, "frames_per_block", 1))
        candidates = list(self.current_episode_plan_v10.probe_near_candidates_by_block.get(int(block_idx), []))  # type: ignore[union-attr]
        if len(candidates) == 0:
            self._probe_near_empty_count += 1
            self.block_states_v10[int(block_idx)].probe_near_empty_count += 1
            return []
        if len(candidates) <= n:
            return [int(x) for x in candidates]
        return [int(x) for x in random.sample([int(x) for x in candidates], n)]

    def _history_targets(self, block: BlockStateV10, student_frame_idx: int, teacher_frame_idx: int) -> List[int]:
        cfg = self._cfg_get(self.supervision_cfg, "history_visited", {}) or {}
        max_targets = int(self._cfg_get(cfg, "max_targets", 2))
        pool = [int(x) for x in self.episode_state_v10.committed_history_frame_indices]
        pool = [int(x) for x in pool if int(x) not in {int(student_frame_idx), int(teacher_frame_idx)}]
        if len(pool) == 0:
            phase = str(self.training_phase).strip().lower()
            if phase == "warmup":
                self._fallback_no_history_count += 1
                block.fallback_no_history_count += 1
                return []
            raise ValueError("scheduler_v10 student_history requested but committed_history_frame_indices is empty.")
        if len(pool) <= max_targets:
            return [int(x) for x in pool]
        sampling = str(self._cfg_get(cfg, "sampling_policy", "most_recent"))
        if sampling == "most_recent":
            return [int(x) for x in pool[-max_targets:]]
        return [int(x) for x in random.sample(pool, max_targets)]

    def _resolve_step_type(self, block: BlockStateV10) -> Stage6StepType:
        step_type = Stage6StepType(self.step_sequence[int(block.step_idx)])
        if step_type in {Stage6StepType.STUDENT_SELF, Stage6StepType.STUDENT_ANCHOR}:
            if len(block.student_candidates) == 0 and bool(
                self._cfg_get(self.frame_selection_cfg, "skip_student_if_single_source", True)
            ):
                self._fallback_no_student_count += 1
                block.fallback_no_student_count += 1
                fb = str(self._cfg_get(self.frame_selection_cfg, "fallback_step_type_if_no_student", "teacher_bootstrap"))
                return Stage6StepType(fb)
        if step_type == Stage6StepType.STUDENT_HISTORY and len(self.episode_state_v10.committed_history_frame_indices) == 0:
            phase = str(self.training_phase).strip().lower()
            if phase == "warmup":
                self._fallback_no_history_count += 1
                block.fallback_no_history_count += 1
                fb = str(
                    self._cfg_get(
                        self.frame_selection_cfg,
                        "fallback_step_type_if_no_committed_history",
                        "student_self",
                    )
                )
                return Stage6StepType(fb)
            raise ValueError("scheduler_v10 student_history requested but committed_history_frame_indices is empty.")
        return step_type

    def _build_step_request(self, st: Dict[str, Any]) -> Stage6StepRequest:
        if self.current_episode_plan_v10 is None:
            raise ValueError("TrainSchedulerV10 internal state is not initialized")
        block_idx = int(st["block_cursor"])
        block = self.block_states_v10[int(block_idx)]
        cam_ids = self._active_cam_ids(int(self.current_episode_plan_v10.num_cams))
        step_type = self._resolve_step_type(block)
        self._last_resolved_step_type = step_type
        teacher_frame = int(block.teacher_frame_idx)
        teacher_refs = self._refs_for_frame(teacher_frame, cam_ids)
        student_frame = self._pick_student_frame(block)
        student_refs = self._refs_for_frame(int(student_frame), cam_ids) if student_frame is not None else []
        probe_frames = self._pick_probe_frames(block_idx=int(block_idx))
        probe_refs: List[ImageRef] = []
        for f in probe_frames:
            probe_refs.extend(self._refs_for_frame(int(f), cam_ids))

        teacher_obs = Stage6TeacherObsRequest(enable=False, frame_idx=int(teacher_frame), image_refs=list(teacher_refs))
        student_prop = Stage6StudentPropRequest(enable=False, frame_idx=student_frame, image_refs=list(student_refs))
        supervision = Stage6SupervisionRequest()
        compat_roles: List[str] = []
        compat_refs: List[ImageRef] = []

        def _add_domain(refs: List[ImageRef], role: str) -> None:
            for r in refs:
                compat_refs.append((int(r[0]), int(r[1])))
                compat_roles.append(str(role))

        live_bridge = bool(self._cfg_get(self.bridge_cfg, "student_steps_use_live_bridge", True))
        if step_type in {Stage6StepType.TEACHER_BOOTSTRAP, Stage6StepType.TEACHER_REFRESH}:
            teacher_obs.enable = True
            teacher_obs.purpose = "train_update"
            teacher_obs.update_state = True
            teacher_obs.update_teacher_prior_cache = True
            teacher_obs.update_observed_history = True
            teacher_obs.update_runtime_history = True
            student_prop.enable = False
            supervision.enable_self_teacher = True
            supervision.self_teacher_refs = list(teacher_refs)
            _add_domain(list(teacher_refs), "teacher_source")
        else:
            if student_frame is None:
                raise ValueError(f"scheduler_v10 step_type={step_type.value} has no student frame candidate.")
            student_prop.enable = True
            student_prop.frame_idx = int(student_frame)
            student_prop.image_refs = list(student_refs)
            student_prop.bridge_mode = "live" if live_bridge else "cache"
            student_prop.require_live_bridge = bool(live_bridge)
            student_prop.update_state = True
            student_prop.update_runtime_history = True
            student_prop.update_observed_history = False
            student_prop.use_gt_input = False
            supervision.enable_self_student = True
            supervision.self_student_refs = list(student_refs)
            _add_domain(list(student_refs), "student_source")
            if live_bridge:
                teacher_obs.enable = True
                teacher_obs.purpose = "live_bridge"
            if step_type == Stage6StepType.STUDENT_ANCHOR:
                supervision.enable_teacher_anchor = True
                supervision.teacher_anchor_refs = list(teacher_refs)
                _add_domain(list(teacher_refs), "teacher_anchor")
            if step_type == Stage6StepType.STUDENT_HISTORY:
                history_frames = self._history_targets(block, int(student_frame), int(teacher_frame))
                for hf in history_frames:
                    supervision.history_visited_refs.extend(self._refs_for_frame(int(hf), cam_ids))
                supervision.enable_history_visited = len(supervision.history_visited_refs) > 0
                _add_domain(list(supervision.history_visited_refs), "history_visited")
        if len(probe_refs) > 0:
            supervision.enable_probe_near = True
            supervision.probe_near_refs = list(probe_refs)
            _add_domain(list(probe_refs), "probe_near")

        validate_teacher_obs_invariants(teacher_obs)
        source_refs = list(student_refs if student_prop.enable else teacher_refs)
        if len(source_refs) == 0:
            source_refs = list(teacher_refs)
        target_refs = list(source_refs) + [tuple(x) for x in compat_refs if tuple(x) not in set(source_refs)]
        target_roles = ["student_source" if student_prop.enable else "teacher_source"] + [
            role
            for role, ref in zip(compat_roles, compat_refs)
            if tuple(ref) not in set(source_refs)
        ]
        request = Stage6StepRequest(
            scheduler_version="v10",
            step_type=step_type,
            scene_id=int(st["scene_id"]),
            segment_id=int(st["segment_id"]),
            block_idx=int(block_idx),
            step_idx_in_block=int(block.step_idx),
            global_scheduler_step=int(self.global_step),
            teacher_obs=teacher_obs,
            student_prop=student_prop,
            supervision=supervision,
            teacher_anchor_frame_idx=int(teacher_frame),
            student_frame_idx=int(student_frame) if student_frame is not None else None,
            committed_history_frame_indices=[int(x) for x in self.episode_state_v10.committed_history_frame_indices],
            probe_near_frame_indices=[int(x) for x in probe_frames],
            history_record={
                "observed_writer": "teacher_only",
                "runtime_writer": "teacher_and_student",
                "commit_policy": "step_exit",
            },
            preload_hints={"all_image_refs": [tuple(x) for x in target_refs]},
            compat={
                "source_image_refs": [tuple(x) for x in source_refs],
                "target_image_refs": [tuple(x) for x in target_refs],
                "target_image_roles": [str(x) for x in target_roles],
            },
        )
        validate_stage6_step_request(request)
        self._last_step_request = request
        return request

    @staticmethod
    def _domain_meta(enable: bool, refs: List[ImageRef], domain: str, *, log_only: bool = False, trainable: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {"enable": bool(enable), "image_refs": [tuple(x) for x in refs], "domain": str(domain)}
        if log_only:
            out["log_only"] = True
        if not trainable:
            out["trainable"] = False
        return out

    def _request_to_meta(self, req: Stage6StepRequest) -> Dict[str, Any]:
        teacher_obs = {
            "enable": bool(req.teacher_obs.enable),
            "purpose": str(req.teacher_obs.purpose),
            "frame_idx": int(req.teacher_obs.frame_idx) if req.teacher_obs.frame_idx is not None else -1,
            "image_refs": [tuple(x) for x in req.teacher_obs.image_refs],
            "use_gt_input": bool(req.teacher_obs.use_gt_input),
            "update_state": bool(req.teacher_obs.update_state),
            "update_teacher_prior_cache": bool(req.teacher_obs.update_teacher_prior_cache),
            "update_observed_history": bool(req.teacher_obs.update_observed_history),
            "update_runtime_history": bool(req.teacher_obs.update_runtime_history),
        }
        student_prop = {
            "enable": bool(req.student_prop.enable),
            "frame_idx": int(req.student_prop.frame_idx) if req.student_prop.frame_idx is not None else -1,
            "image_refs": [tuple(x) for x in req.student_prop.image_refs],
            "use_gt_input": bool(req.student_prop.use_gt_input),
            "bridge_mode": str(req.student_prop.bridge_mode),
            "require_live_bridge": bool(req.student_prop.require_live_bridge),
            "update_state": bool(req.student_prop.update_state),
            "update_runtime_history": bool(req.student_prop.update_runtime_history),
            "update_observed_history": bool(req.student_prop.update_observed_history),
        }
        supervision = {
            "self_teacher": self._domain_meta(
                req.supervision.enable_self_teacher, req.supervision.self_teacher_refs, "self_teacher"
            ),
            "self_student": self._domain_meta(
                req.supervision.enable_self_student, req.supervision.self_student_refs, "self_student"
            ),
            "teacher_anchor": self._domain_meta(
                req.supervision.enable_teacher_anchor, req.supervision.teacher_anchor_refs, "teacher_anchor"
            ),
            "history_visited": self._domain_meta(
                req.supervision.enable_history_visited, req.supervision.history_visited_refs, "history_visited"
            ),
            "probe_near": self._domain_meta(
                req.supervision.enable_probe_near,
                req.supervision.probe_near_refs,
                "probe_near",
                log_only=True,
                trainable=False,
            ),
        }
        target_refs = [tuple(x) for x in list(req.compat.get("target_image_refs") or [])]
        target_roles = [str(x) for x in list(req.compat.get("target_image_roles") or [])]
        source_refs = [tuple(x) for x in list(req.compat.get("source_image_refs") or [])]
        train_roles = {"teacher_source", "student_source", "teacher_anchor", "history_visited"}
        probe_roles = {"probe_near"}
        train_target_refs = [tuple(r) for r, role in zip(target_refs, target_roles) if str(role) in train_roles]
        train_target_roles = [str(role) for role in target_roles if str(role) in train_roles]
        probe_target_refs = [tuple(r) for r, role in zip(target_refs, target_roles) if str(role) in probe_roles]
        probe_target_roles = [str(role) for role in target_roles if str(role) in probe_roles]
        frame_roles = [str(x) for x in target_roles]
        frame_indices = [int(x[0]) for x in target_refs]
        meta = {
            "scheduler_version": "v10",
            "stage": "6_0",
            "step_type": str(req.step_type.value),
            "scene_id": int(req.scene_id),
            "segment_id": int(req.segment_id),
            "block_idx": int(req.block_idx),
            "step_idx_in_block": int(req.step_idx_in_block),
            "global_scheduler_step": int(req.global_scheduler_step),
            "teacher_obs": teacher_obs,
            "student_prop": student_prop,
            "supervision": supervision,
            "history_record": dict(req.history_record),
            "preload_hints": dict(req.preload_hints),
            "compat": dict(req.compat),
            "source_image_refs": list(source_refs),
            "target_image_refs": list(target_refs),
            "target_image_roles": list(target_roles),
            "target_image_loss_base_weights": [0.0 if role == "probe_near" else 1.0 for role in target_roles],
            "target_frame_indices": [int(x) for x in frame_indices],
            "target_frame_roles": list(frame_roles),
            "target_frame_loss_base_weights": [0.0 if role == "probe_near" else 1.0 for role in frame_roles],
            "train_target_image_refs": list(train_target_refs),
            "train_target_image_roles": list(train_target_roles),
            "probe_target_image_refs": list(probe_target_refs),
            "probe_target_image_roles": list(probe_target_roles),
            "stage6_role": "student" if bool(req.student_prop.enable) else "teacher",
            "scheduler_v10/fallback_no_student_count": float(self._fallback_no_student_count),
            "scheduler_v10/fallback_no_history_count": float(self._fallback_no_history_count),
            "scheduler_v10/probe_near_empty_count": float(self._probe_near_empty_count),
            "scheduler_v10/live_teacher_bridge_enable": float(
                1.0 if bool(req.teacher_obs.enable and req.teacher_obs.purpose == "live_bridge") else 0.0
            ),
            "scheduler_v10/train_target_num_frames": float(len(train_target_refs)),
            "scheduler_v10/probe_target_num_frames": float(len(probe_target_refs)),
        }
        # Transitional compat for old trainer readers.
        meta["scheduler_request_v10"] = {
            "scheduler_version": "v10",
            "stage": "6_0",
            "teacher_obs": {
                "enable": bool(req.teacher_obs.enable),
                "frame_idx": int(req.teacher_obs.frame_idx) if req.teacher_obs.frame_idx is not None else -1,
                "image_refs": [tuple(x) for x in req.teacher_obs.image_refs],
                "record_observed": bool(req.teacher_obs.update_observed_history),
                "update_cache": bool(req.teacher_obs.update_teacher_prior_cache),
                "purpose": str(req.teacher_obs.purpose),
            },
            "student_prop": {
                "enable": bool(req.student_prop.enable),
                "frame_idx": int(req.student_prop.frame_idx) if req.student_prop.frame_idx is not None else -1,
                "requires_teacher_anchor": bool(req.supervision.enable_teacher_anchor),
                "requires_live_bridge": bool(req.student_prop.require_live_bridge),
            },
            "live_teacher_bridge": {
                "enable": bool(req.teacher_obs.enable and req.teacher_obs.purpose == "live_bridge"),
                "frame_idx": int(req.teacher_obs.frame_idx) if req.teacher_obs.frame_idx is not None else -1,
                "image_refs": [tuple(x) for x in req.teacher_obs.image_refs],
                "record_observed": False,
                "update_cache": False,
                "rerun_teacher_2d": bool(req.teacher_obs.enable and req.teacher_obs.purpose == "live_bridge"),
            },
            "teacher_anchor": {
                "enable": bool(req.supervision.enable_teacher_anchor),
                "frame_idx": int(req.teacher_anchor_frame_idx) if req.teacher_anchor_frame_idx is not None else -1,
                "image_refs": [tuple(x) for x in req.supervision.teacher_anchor_refs],
                "role": "teacher_anchor",
            },
            "history_targets": {
                "enable": bool(req.supervision.enable_history_visited),
                "frame_indices": [int(x[0]) for x in req.supervision.history_visited_refs],
                "role": "history_visited",
            },
            "probe_targets": {
                "enable": bool(req.supervision.enable_probe_near),
                "image_refs": [tuple(x) for x in req.supervision.probe_near_refs],
                "role": "probe_near",
                "log_only": True,
            },
            "train_targets": {
                "image_refs": list(train_target_refs),
                "image_roles": list(train_target_roles),
                "image_loss_base_weights": [1.0 for _ in train_target_refs],
            },
            "fallback_to_teacher": True,
        }
        meta["scheduler/v10_is_compat_v9"] = 0.0
        return meta

    def _materialize_batch_from_request(self, req: Stage6StepRequest) -> Dict[str, Any]:
        source_refs = [tuple(x) for x in list(req.compat.get("source_image_refs") or [])]
        target_refs = [tuple(x) for x in list(req.compat.get("target_image_refs") or [])]
        if len(source_refs) == 0 or len(target_refs) == 0:
            raise ValueError("scheduler_v10 materialization requires non-empty source/target refs.")
        request = _BatchRequestV7Compat(
            scene_id=int(req.scene_id),
            segment_id=int(req.segment_id),
            source_image_ref=(int(source_refs[0][0]), int(source_refs[0][1])),
            target_image_refs=[(int(x[0]), int(x[1])) for x in target_refs],
            source_image_refs=[(int(x[0]), int(x[1])) for x in source_refs],
            include_test=bool(self.include_test),
            test_image_refs=None,
        )
        return self.dataset.get_segment_batch_from_image_refs(
            request,
            enforce_target0_equals_source=True,
        )

    def _aligned_info_v10(self, st: Dict[str, Any], req: Stage6StepRequest) -> Dict[str, Any]:
        base = dict(self._aligned_info(st))
        base["scheduler_version"] = "v10"
        base["step_type"] = str(req.step_type.value)
        base["block_idx_in_episode"] = int(req.block_idx)
        base["source_frame_idx"] = int(req.student_frame_idx if req.student_frame_idx is not None else req.teacher_anchor_frame_idx or -1)
        base["source_image_ref"] = tuple((base["source_frame_idx"], 0))
        return base

    def _patch_aligned_to_v10(self, batch: Dict[str, Any], aligned: Dict[str, Any]) -> None:
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_aligned_info"] = dict(aligned)
        batch["_scheduler_v9_aligned_info"] = dict(aligned)
        batch["_scheduler_v10_aligned_info"] = dict(aligned)

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        self._ensure_episode_state()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV10 internal state is not initialized")
        req = self._build_step_request(st)
        batch = self._materialize_batch_from_request(req)
        meta = self._request_to_meta(req)
        batch["request_meta"] = dict(meta)
        aligned = self._aligned_info_v10(st, req)
        self._patch_aligned_to_v10(batch, aligned)
        batch["_scheduler_v7_peek"] = True
        return batch

    def _commit_block(self, block: BlockStateV10) -> None:
        if bool(block.teacher_seen):
            self.episode_state_v10.committed_history_frame_indices.append(int(block.teacher_frame_idx))
            self.episode_state_v10.teacher_seen_blocks.append(int(block.block_idx))
            self.episode_state_v10.last_committed_block_idx = int(block.block_idx)
        if bool(block.student_seen):
            self.episode_state_v10.student_seen_blocks.append(int(block.block_idx))

    def next_batch(self) -> Dict[str, Any]:
        self._ensure_episode_state()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV10 internal state is not initialized")
        req = self._build_step_request(st)
        batch = self._materialize_batch_from_request(req)
        meta = self._request_to_meta(req)
        batch["request_meta"] = dict(meta)
        key = (int(st["scene_id"]), int(st["segment_id"]))
        rt = self._segment_runtime[key]
        rt["segment_local_step"] = int(rt["segment_local_step"]) + 1
        self.global_step += 1

        block_idx = int(req.block_idx)
        block = self.block_states_v10[int(block_idx)]
        if bool(req.teacher_obs.enable and str(req.teacher_obs.purpose) == "train_update"):
            block.teacher_seen = True
            block.last_teacher_frame_idx = int(req.teacher_obs.frame_idx) if req.teacher_obs.frame_idx is not None else None
            if req.teacher_obs.frame_idx is not None:
                block.runtime_updated_frames.append(int(req.teacher_obs.frame_idx))
        if bool(req.student_prop.enable):
            block.student_seen = True
            block.last_student_frame_idx = int(req.student_prop.frame_idx) if req.student_prop.frame_idx is not None else None
            if req.student_prop.frame_idx is not None:
                block.runtime_updated_frames.append(int(req.student_prop.frame_idx))
        block.step_idx = int(block.step_idx + 1)

        aligned = self._aligned_info_v10(st, req)
        self._patch_aligned_to_v10(batch, aligned)

        if self.block_order == "step_major":
            st["block_update_counts"][block_idx] = int(st["block_update_counts"][block_idx]) + 1
            st["episode_step_cursor"] = int(st.get("episode_step_cursor", 0)) + 1
            if int(block.step_idx) >= int(self.steps_per_block) and not bool(st["block_ended"][block_idx]):
                self._commit_block(block)
                self._emit_block_end_for_block(st, block_idx)
                st["block_ended"][block_idx] = True
                rt["block_idx_in_segment"] = max(
                    int(rt["block_idx_in_segment"]),
                    int(st["episode_base_block_idx_in_segment"]) + int(block_idx) + 1,
                )
            if int(st.get("episode_step_cursor", 0)) >= int(self.total_episode_steps):
                self._finalize_episode_if_needed()
            else:
                next_block_idx = int(self._episode_block_visit_order[int(st["episode_step_cursor"])])
                self._select_block(next_block_idx)
        else:
            st["block_repeat_step"] = int(st["block_repeat_step"]) + 1
            if int(block.step_idx) >= int(self.steps_per_block):
                self._commit_block(block)
                self._emit_block_end_for_block(st, block_idx)
                rt["block_idx_in_segment"] = int(rt["block_idx_in_segment"]) + 1
                st["block_idx_in_segment"] = int(rt["block_idx_in_segment"])
                st["block_cursor"] = int(st["block_cursor"]) + 1
                st["block_repeat_step"] = 0
                if int(st["block_cursor"]) < int(self.blocks_per_episode):
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
        out["scheduler_version"] = "v10"
        out["scheduler_v10/fallback_no_student_count"] = float(self._fallback_no_student_count)
        out["scheduler_v10/fallback_no_history_count"] = float(self._fallback_no_history_count)
        out["scheduler_v10/probe_near_empty_count"] = float(self._probe_near_empty_count)
        return out
