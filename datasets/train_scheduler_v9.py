from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

from datasets.train_scheduler_v7 import SegmentIndexLike, TrainSchedulerV7
from datasets.train_scheduler_v8 import TrainSchedulerDatasetV7, TrainSchedulerV8

ImageRef = Tuple[int, int]

SchedulerPhaseV9 = Literal[
    "phase_A_block_local_unroll",
    "phase_B_viewset_rollout",
]

RefRoleV9 = Literal[
    "evidence",
    "block_loss",
    "nearby_loss",
    "prefix_loss",
    "query_label",
    "aux_loss",
]


@dataclass(frozen=True)
class RefGroupV9:
    role: RefRoleV9
    refs: List[ImageRef]
    frame_indices: List[int]
    image_roles: List[str]
    allow_update_evidence: bool
    allow_render_loss: bool
    allow_query_label: bool
    mask_policy: str


@dataclass(frozen=True)
class StepPlanV9:
    step_idx: int
    source_keyframe_idx: int
    source_frame_idx: int
    block_idx: int

    evidence_refs: List[ImageRef]
    block_loss_refs: List[ImageRef]
    nearby_loss_refs: List[ImageRef]
    prefix_loss_refs: List[ImageRef]
    query_label_refs: List[ImageRef]
    aux_loss_refs: List[ImageRef]

    evidence_frame_indices: List[int]
    loss_frame_indices: List[int]
    nearby_frame_indices: List[int]
    query_frame_indices: List[int]


@dataclass(frozen=True)
class ViewSetRolloutBatchV9:
    scheduler_version: str
    phase: SchedulerPhaseV9

    scene_id: int
    segment_id: int
    episode_id: int
    episode_start_keyframe_pos: int
    keyframe_window: List[int]
    frame_chain: List[int]
    num_cams: int

    inner_K: int
    steps: List[StepPlanV9]

    evidence_refs_by_step: List[List[ImageRef]]
    block_loss_refs_by_step: List[List[ImageRef]]
    nearby_loss_refs_by_step: List[List[ImageRef]]
    prefix_loss_refs_by_step: List[List[ImageRef]]
    query_label_refs: List[ImageRef]
    aux_loss_refs: List[ImageRef]

    request_meta: Dict[str, Any] = field(default_factory=dict)
    leakage_check: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchRequestV9:
    scene_id: int
    segment_id: int
    evidence_refs: List[ImageRef]
    loss_refs: List[ImageRef]
    query_label_refs: List[ImageRef]
    aux_loss_refs: List[ImageRef]
    v9_plan: ViewSetRolloutBatchV9


class TrainSchedulerV9(TrainSchedulerV8):
    def __init__(
        self,
        *,
        dataset: TrainSchedulerDatasetV7,
        phase: SchedulerPhaseV9,
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
        warm_v9_role_refs: bool = True,
        block_order: str = "block_major",
        step_major_switch_interval_steps: int = 1,
        target_policy: str = "visited_episode_frames",
        reset_policy: str = "episode_end",
        block_source_frame_policy: str = "random_within_keyframe_per_visit",
        episode_source_mode: str = "keyframes",
        phase_a_cfg: Optional[Any] = None,
        phase_b_cfg: Optional[Any] = None,
        leakage_check_cfg: Optional[Any] = None,
        fail_fast: bool = True,
    ) -> None:
        if phase not in ("phase_A_block_local_unroll", "phase_B_viewset_rollout"):
            raise ValueError(f"unsupported scheduler_v9.phase={phase!r}")
        self.phase: SchedulerPhaseV9 = phase
        self.phase_a_cfg = phase_a_cfg or {}
        self.phase_b_cfg = phase_b_cfg or {}
        self.v9_leakage_check_cfg = leakage_check_cfg or {}
        self.v9_fail_fast = bool(fail_fast)
        self.warm_v9_role_refs = bool(warm_v9_role_refs)
        self._v9_prefetched_plan: Optional[ViewSetRolloutBatchV9] = None
        self._v9_prefetched_plan_key: Optional[Tuple[int, int, int, int, int, int]] = None

        phase_a_block = self._cfg_get(self.phase_a_cfg, "block", {}) or {}
        phase_a_near = self._cfg_get(self.phase_a_cfg, "nearby_supervision", {}) or {}
        phase_a_masks = self._cfg_get(self.phase_a_cfg, "masks", {}) or {}
        self.phase_a_mode = str(self._cfg_get(self.phase_a_cfg, "mode", "block_local_unroll"))
        self.phase_a_repeat_block_iteration = bool(
            self._cfg_get(phase_a_block, "repeat_block_iteration", True)
        )
        self.phase_a_source_frame_policy = str(
            self._cfg_get(phase_a_block, "source_frame_policy", "fixed_for_scheduler_step")
        )
        self.phase_a_inner_K_choices = [
            int(x) for x in list(self._cfg_get(phase_a_block, "inner_K_choices", [2, 4, 6]) or [2, 4, 6])
        ]
        self.phase_a_inner_K_probs = self._optional_float_list(
            self._cfg_get(phase_a_block, "inner_K_probs", [0.45, 0.40, 0.15])
        )
        self.phase_a_block_loss_policy = str(self._cfg_get(phase_a_block, "block_loss_policy", "source_frame_all_cams"))
        self.phase_a_nearby_enable = bool(self._cfg_get(phase_a_near, "enable", True))
        self.phase_a_nearby_policy = str(
            self._cfg_get(phase_a_near, "policy", "adjacent_then_random_same_keyframe")
        )
        self.phase_a_nearby_frames_per_block = int(self._cfg_get(phase_a_near, "frames_per_block", 2))
        self.phase_a_nearby_adjacent_radius = int(self._cfg_get(phase_a_near, "adjacent_radius", 1))
        self.phase_a_nearby_random_fill = bool(self._cfg_get(phase_a_near, "random_fill", True))
        self.phase_a_nearby_same_keyframe_only = bool(self._cfg_get(phase_a_near, "same_keyframe_only", True))
        self.phase_a_nearby_insufficient_policy = str(self._cfg_get(phase_a_near, "insufficient_policy", "skip"))
        self.phase_a_nearby_exclude_source = bool(self._cfg_get(phase_a_near, "exclude_source_frame", True))
        self.phase_a_nearby_exclude_existing_block_loss = bool(
            self._cfg_get(phase_a_near, "exclude_existing_block_loss_frames", True)
        )
        self.phase_a_nearby_camera_policy = str(self._cfg_get(phase_a_near, "camera_policy", "all_cams"))
        self.phase_a_nearby_apply_every_step = bool(self._cfg_get(phase_a_near, "apply_every_step", False))
        self.phase_a_nearby_final_step_only = bool(self._cfg_get(phase_a_near, "apply_final_step_only", True))
        self.phase_a_nearby_add_to_evidence_refs = bool(
            self._cfg_get(phase_a_near, "add_to_evidence_refs", False)
        )
        self.phase_a_nearby_add_to_source_image_refs = bool(
            self._cfg_get(phase_a_near, "add_to_source_image_refs", False)
        )
        self.phase_a_nearby_add_to_history_record_views = bool(
            self._cfg_get(phase_a_near, "add_to_history_record_views", False)
        )
        self.phase_a_nearby_role_name = str(self._cfg_get(phase_a_near, "role_name", "phaseA_nearby"))
        self.phase_a_nearby_max_refs_per_step = int(self._cfg_get(phase_a_near, "max_refs_per_step", 12))
        self.phase_a_nearby_loss_weight = float(self._cfg_get(phase_a_near, "loss_weight", 0.25))
        self.phase_a_nearby_loss_warmup_steps = int(self._cfg_get(phase_a_near, "loss_warmup_steps", 2000))
        self.phase_a_block_loss_mask = str(self._cfg_get(phase_a_masks, "block_loss_mask", "non_sky_non_egocar"))
        self.phase_a_nearby_loss_mask = str(self._cfg_get(phase_a_masks, "nearby_loss_mask", "non_sky_non_egocar"))

        phase_b_rollout = self._cfg_get(self.phase_b_cfg, "rollout", {}) or {}
        phase_b_episode = self._cfg_get(self.phase_b_cfg, "episode", {}) or {}
        phase_b_prefix = self._cfg_get(self.phase_b_cfg, "prefix_render", {}) or {}
        phase_b_query = self._cfg_get(self.phase_b_cfg, "query_observation", {}) or {}
        phase_b_masks = self._cfg_get(self.phase_b_cfg, "masks", {}) or {}
        self.phase_b_reset_vsm_on_episode_end = bool(
            self._cfg_get(phase_b_episode, "reset_vsm_on_episode_end", True)
        )
        self.phase_b_K_choices = [
            int(x) for x in list(self._cfg_get(phase_b_rollout, "K_choices", [2, 4]) or [2, 4])
        ]
        self.phase_b_K_probs = self._optional_float_list(self._cfg_get(phase_b_rollout, "K_probs", [0.65, 0.35]))
        self.phase_b_curriculum = list(self._cfg_get(phase_b_rollout, "curriculum", []) or [])
        self.phase_b_repeat_patterns = [
            dict(x) for x in list(self._cfg_get(phase_b_rollout, "repeat_patterns", []) or [])
        ]
        self.phase_b_repeat_max_inner_K = int(self._cfg_get(phase_b_rollout, "max_inner_K", 8))
        self.phase_b_repeat_source_frame_policy = str(
            self._cfg_get(phase_b_rollout, "repeat_source_frame_policy", "fixed_within_block")
        )
        self.phase_b_repeat_memory_write_policy = str(
            self._cfg_get(phase_b_rollout, "repeat_memory_write_policy", "first_repeat_only")
        )
        self.phase_b_evidence_recompute_policy = str(
            self._cfg_get(phase_b_rollout, "evidence_recompute_policy", "every_repeat")
        )
        self.phase_b_allow_short_final_chunk = bool(
            self._cfg_get(phase_b_rollout, "allow_short_final_chunk", True)
        )
        self.phase_b_sample_event_frames = str(
            self._cfg_get(phase_b_rollout, "sample_event_frames", "random_blocks_in_episode")
        )
        raw_phase_b_rollout_mode = self._cfg_get(phase_b_rollout, "mode", None)
        if raw_phase_b_rollout_mode is None:
            self.phase_b_rollout_mode = (
                "episode_stream_tbptt"
                if self.phase_b_sample_event_frames == "sequential_blocks_in_episode"
                else "random_viewset_local"
            )
        else:
            self.phase_b_rollout_mode = str(raw_phase_b_rollout_mode)
        self.phase_b_event_order = str(self._cfg_get(phase_b_rollout, "event_order", "chronological"))
        self.phase_b_distinct_event_frames = bool(self._cfg_get(phase_b_rollout, "distinct_event_frames", True))
        self.phase_b_prefix_policy = str(self._cfg_get(phase_b_prefix, "policy", "current_plus_random_previous"))
        self.phase_b_prefix_intermediate_views = int(self._cfg_get(phase_b_prefix, "intermediate_views", 2))
        self.phase_b_prefix_final_views = int(self._cfg_get(phase_b_prefix, "final_views", 3))
        self.phase_b_prefix_max_refs_per_step = int(self._cfg_get(phase_b_prefix, "max_refs_per_step", 18))
        self.phase_b_query_enable = bool(self._cfg_get(phase_b_query, "enable", True))
        self.phase_b_query_frame_policy = str(
            self._cfg_get(phase_b_query, "query_frame_policy", "heldout_inside_event_span")
        )
        self.phase_b_query_frames_per_rollout = int(self._cfg_get(phase_b_query, "frames_per_rollout", 1))
        self.phase_b_query_cameras_per_frame = str(self._cfg_get(phase_b_query, "cameras_per_frame", "all_cams"))
        self.phase_b_query_exclude_event_frames = bool(self._cfg_get(phase_b_query, "exclude_event_frames", True))
        self.phase_b_query_allow_empty_on_last_chunk = bool(
            self._cfg_get(phase_b_query, "allow_empty_on_last_chunk", False)
        )
        self.phase_b_vsm_scope = str(self._cfg_get(phase_b_masks, "vsm_scope", "bg_rigid"))
        self.phase_b_evidence_mask = str(
            self._cfg_get(phase_b_masks, "evidence_mask", "non_sky_non_egocar")
        )
        self.phase_b_prefix_loss_mask = str(
            self._cfg_get(phase_b_masks, "prefix_loss_mask", "non_sky_non_egocar")
        )
        self.phase_b_query_label_mask = str(
            self._cfg_get(phase_b_masks, "query_label_mask", "non_sky_non_egocar")
        )

        self.v9_leakage_check_enable = bool(self._cfg_get(self.v9_leakage_check_cfg, "enable", True))
        self.v9_same_scene_segment_required = bool(
            self._cfg_get(self.v9_leakage_check_cfg, "same_scene_segment_required", True)
        )
        self.v9_query_not_in_evidence = bool(self._cfg_get(self.v9_leakage_check_cfg, "query_not_in_evidence", True))
        self.v9_nearby_not_in_evidence = bool(self._cfg_get(self.v9_leakage_check_cfg, "nearby_not_in_evidence", True))
        self.v9_aux_not_in_evidence = bool(self._cfg_get(self.v9_leakage_check_cfg, "aux_not_in_evidence", True))
        self.v9_role_count_match_required = bool(
            self._cfg_get(self.v9_leakage_check_cfg, "role_count_match_required", True)
        )
        self.v9_forbid_test_refs_in_train = bool(
            self._cfg_get(self.v9_leakage_check_cfg, "forbid_test_refs_in_train", True)
        )

        if self.phase_a_mode != "block_local_unroll":
            raise ValueError("scheduler_v9 P0 requires phase_A.mode=block_local_unroll")
        if not self.v9_leakage_check_enable:
            raise ValueError("scheduler_v9 P0 requires leakage_check.enable=true")
        if not self.v9_same_scene_segment_required:
            raise ValueError("scheduler_v9 P0 requires leakage_check.same_scene_segment_required=true")
        if not self.phase_a_repeat_block_iteration:
            raise ValueError("scheduler_v9 P0 requires phase_A.block.repeat_block_iteration=true")
        if self.phase_a_source_frame_policy != "fixed_for_scheduler_step":
            raise ValueError("scheduler_v9 P0 requires phase_A.block.source_frame_policy=fixed_for_scheduler_step")
        if self.phase_a_block_loss_policy != "source_frame_all_cams":
            raise ValueError(
                "scheduler_v9 P0 only supports phase_A.block.block_loss_policy=source_frame_all_cams"
            )
        if self.phase_a_nearby_apply_every_step:
            raise ValueError("scheduler_v9 Phase A P0 requires nearby_supervision.apply_every_step=false")
        if not self.phase_a_nearby_final_step_only:
            raise ValueError("scheduler_v9 Phase A P0 requires nearby_supervision.apply_final_step_only=true")
        if self.phase_a_nearby_add_to_evidence_refs:
            raise ValueError("scheduler_v9 Phase A P0 requires nearby_supervision.add_to_evidence_refs=false")
        if self.phase_a_nearby_add_to_source_image_refs:
            raise ValueError("scheduler_v9 Phase A P0 requires nearby_supervision.add_to_source_image_refs=false")
        if self.phase_a_nearby_add_to_history_record_views:
            raise ValueError(
                "scheduler_v9 Phase A P0 requires nearby_supervision.add_to_history_record_views=false"
            )
        if self.phase_a_nearby_enable:
            if not self.phase_a_nearby_same_keyframe_only:
                raise ValueError("scheduler_v9.phase_A.nearby_supervision.same_keyframe_only must be true")
            if self.phase_a_nearby_insufficient_policy != "skip":
                raise ValueError("scheduler_v9.phase_A.nearby_supervision.insufficient_policy must be skip")
            if self.phase_a_nearby_camera_policy != "all_cams":
                raise ValueError("scheduler_v9 P0 supports nearby camera_policy=all_cams only")
            if self.phase_a_nearby_frames_per_block < 0:
                raise ValueError("scheduler_v9.phase_A.nearby_supervision.frames_per_block must be >= 0")
        if self.phase_b_event_order not in ("chronological", "sampled"):
            raise ValueError("scheduler_v9.phase_B.rollout.event_order must be chronological or sampled")
        if self.phase_b_rollout_mode not in (
            "random_viewset_local",
            "episode_stream_tbptt",
            "episode_block_repeat_tbptt",
            "episode_grouped_repeat_tbptt",
        ):
            raise ValueError(
                "scheduler_v9.phase_B.rollout.mode must be random_viewset_local, "
                "episode_stream_tbptt, episode_block_repeat_tbptt, or episode_grouped_repeat_tbptt"
            )
        if self.phase == "phase_B_viewset_rollout" and self.phase_b_rollout_mode == "episode_stream_tbptt":
            if self.phase_b_sample_event_frames != "sequential_blocks_in_episode":
                raise ValueError(
                    "Phase B episode_stream_tbptt requires "
                    "phase_B.rollout.sample_event_frames=sequential_blocks_in_episode"
                )
            if self.phase_b_event_order != "chronological":
                raise ValueError("Phase B episode_stream_tbptt requires event_order=chronological")
            if not self.phase_b_distinct_event_frames:
                raise ValueError("Phase B episode_stream_tbptt requires distinct_event_frames=true")
            if int(steps_per_block) != 1:
                raise ValueError("Phase B episode_stream_tbptt requires block.steps_per_block=1")
        if self.phase == "phase_B_viewset_rollout" and self.phase_b_rollout_mode == "episode_block_repeat_tbptt":
            raise ValueError("episode_block_repeat_tbptt is deprecated; use episode_grouped_repeat_tbptt")
        if self.phase == "phase_B_viewset_rollout" and self.phase_b_rollout_mode == "episode_grouped_repeat_tbptt":
            if self.phase_b_sample_event_frames != "sequential_blocks_in_episode":
                raise ValueError(
                    "Phase B episode_grouped_repeat_tbptt requires "
                    "phase_B.rollout.sample_event_frames=sequential_blocks_in_episode"
                )
            if self.phase_b_event_order != "chronological":
                raise ValueError("Phase B episode_grouped_repeat_tbptt requires event_order=chronological")
            if not self.phase_b_distinct_event_frames:
                raise ValueError("Phase B episode_grouped_repeat_tbptt requires distinct_event_frames=true")
            if int(steps_per_block) != 1:
                raise ValueError(
                    "Phase B episode_grouped_repeat_tbptt requires scheduler_v9.block.steps_per_block=1; "
                    "use phase_B.rollout.repeat_patterns instead"
                )
            if self.phase_b_repeat_source_frame_policy != "fixed_within_block":
                raise ValueError("Phase B episode_grouped_repeat_tbptt requires repeat_source_frame_policy=fixed_within_block")
            if self.phase_b_repeat_memory_write_policy != "first_repeat_only":
                raise ValueError("Phase B episode_grouped_repeat_tbptt requires repeat_memory_write_policy=first_repeat_only")
            if self.phase_b_evidence_recompute_policy != "every_repeat":
                raise ValueError("Phase B episode_grouped_repeat_tbptt requires evidence_recompute_policy=every_repeat")
            if int(self.phase_b_repeat_max_inner_K) < 1:
                raise ValueError("Phase B episode_grouped_repeat_tbptt requires max_inner_K >= 1")
            self._validate_phase_b_repeat_patterns(
                self.phase_b_repeat_patterns,
                blocks_per_episode=int(blocks_per_episode),
                label="phase_B.rollout.repeat_patterns",
            )
            for stage in self.phase_b_curriculum:
                raw_patterns = self._cfg_get(stage, "repeat_patterns", None)
                if raw_patterns is not None:
                    self._validate_phase_b_repeat_patterns(
                        [dict(x) for x in list(raw_patterns or [])],
                        blocks_per_episode=int(blocks_per_episode),
                        label="phase_B.rollout.curriculum.repeat_patterns",
                    )
        if self.phase == "phase_B_viewset_rollout" and self.phase_b_rollout_mode == "random_viewset_local":
            if self.phase_b_sample_event_frames != "random_blocks_in_episode":
                raise ValueError(
                    "Phase B random_viewset_local requires "
                    "phase_B.rollout.sample_event_frames=random_blocks_in_episode"
                )
        if self.phase_b_prefix_policy != "current_plus_random_previous":
            raise ValueError("scheduler_v9 P0 supports prefix_render.policy=current_plus_random_previous only")
        if self.phase_b_query_cameras_per_frame != "all_cams":
            raise ValueError("scheduler_v9 P0 supports query_observation.cameras_per_frame=all_cams only")
        if self.phase_b_query_allow_empty_on_last_chunk:
            raise ValueError("scheduler_v9 Phase B P0 requires query_observation.allow_empty_on_last_chunk=false")
        if self.phase_b_vsm_scope != "bg_rigid":
            raise ValueError("scheduler_v9 Phase B-R requires phase_B.masks.vsm_scope=bg_rigid")
        for mask_name, mask_value in (
            ("evidence_mask", self.phase_b_evidence_mask),
            ("prefix_loss_mask", self.phase_b_prefix_loss_mask),
            ("query_label_mask", self.phase_b_query_label_mask),
        ):
            if "dynamic" in str(mask_value):
                raise ValueError(f"scheduler_v9 Phase B-R must not use dynamic mask policy for {mask_name}")
        if not self.phase_b_reset_vsm_on_episode_end:
            raise ValueError("scheduler_v9 Phase B requires phase_B.episode.reset_vsm_on_episode_end=true")
        if (
            self.phase == "phase_B_viewset_rollout"
            and self.phase_b_distinct_event_frames
            and self.phase_b_rollout_mode != "episode_grouped_repeat_tbptt"
        ):
            max_k = max([int(x) for x in self.phase_b_K_choices] + [
                int(k)
                for stage in self.phase_b_curriculum
                for k in list(self._cfg_get(stage, "K_choices", []) or [])
            ])
            if int(max_k) > int(blocks_per_episode):
                raise ValueError(
                    "Phase B distinct_event_frames=true forbids K_choices greater than blocks_per_episode: "
                    f"max_K={int(max_k)} blocks_per_episode={int(blocks_per_episode)}"
                )

        # V9 owns role split semantics. Disable V8 target-role extensions and pass a
        # minimal V8 target window only to reuse its episode traversal/state machine.
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
            target_policy=str(target_policy),
            history_target_policy="nearest_visited",
            reset_policy=str(reset_policy),
            near_random_supervision_cfg={"enable": False},
            aux_feature_splat_targets_cfg={"enable": False},
            block_source_frame_policy=str(block_source_frame_policy),
            episode_source_mode=str(episode_source_mode),
        )

    @staticmethod
    def _optional_float_list(raw: Any) -> Optional[List[float]]:
        if raw is None:
            return None
        out = [float(x) for x in list(raw)]
        return out

    @staticmethod
    def _flatten(ref_groups: Sequence[Sequence[ImageRef]]) -> List[ImageRef]:
        out: List[ImageRef] = []
        for group in ref_groups:
            for ref in group:
                out.append((int(ref[0]), int(ref[1])))
        return out

    @staticmethod
    def _dedupe_refs_roles_keep_order(refs: Sequence[ImageRef], roles: Sequence[str]) -> tuple[List[ImageRef], List[str]]:
        if len(refs) != len(roles):
            raise ValueError(f"refs/roles length mismatch: {len(refs)} vs {len(roles)}")
        seen: Set[ImageRef] = set()
        out_refs: List[ImageRef] = []
        out_roles: List[str] = []
        for ref, role in zip(refs, roles):
            r = (int(ref[0]), int(ref[1]))
            if r in seen:
                continue
            seen.add(r)
            out_refs.append(r)
            out_roles.append(str(role))
        return out_refs, out_roles

    def _emit(self, event: Dict[str, Any]) -> None:
        out = dict(event)
        if str(out.get("scheduler_version", "")) in {"v7", "v8", ""}:
            out["scheduler_version"] = "v9"
        TrainSchedulerV7._emit(self, out)

    def _sample_inner_K(self, choices: Sequence[int], probs: Optional[Sequence[float]]) -> int:
        vals = [int(x) for x in list(choices)]
        if len(vals) == 0:
            raise ValueError("inner K choices must not be empty")
        for val in vals:
            if int(val) < 1:
                raise ValueError("inner K choices must all be >= 1")
        if probs is None:
            return int(random.choice(vals))
        weights = [float(x) for x in list(probs)]
        if len(weights) != len(vals):
            raise ValueError(f"K choices/probs length mismatch: {len(vals)} vs {len(weights)}")
        return int(random.choices(vals, weights=weights, k=1)[0])

    def _validate_phase_b_repeat_patterns(
        self,
        patterns: Sequence[Dict[str, Any]],
        *,
        blocks_per_episode: int,
        label: str,
    ) -> None:
        vals = [dict(x) for x in list(patterns or [])]
        if not vals:
            raise ValueError(f"episode_grouped_repeat_tbptt requires non-empty {label}")
        total_prob = 0.0
        for idx, pattern in enumerate(vals):
            repeats = int(self._cfg_get(pattern, "repeats_per_block", 0) or 0)
            blocks = int(self._cfg_get(pattern, "blocks_per_chunk", 0) or 0)
            if repeats < 1 or blocks < 1:
                raise ValueError(f"{label}[{idx}] repeats_per_block and blocks_per_chunk must be >= 1")
            inner_k = int(repeats * blocks)
            if blocks > int(blocks_per_episode):
                raise ValueError(
                    "Phase B grouped repeat blocks_per_chunk cannot exceed blocks_per_episode: "
                    f"blocks_per_chunk={blocks} blocks_per_episode={int(blocks_per_episode)}"
                )
            if inner_k > int(self.phase_b_repeat_max_inner_K):
                raise ValueError(
                    "Phase B grouped repeat inner_K exceeds max_inner_K: "
                    f"inner_K={inner_k} max_inner_K={int(self.phase_b_repeat_max_inner_K)}"
                )
            prob = float(self._cfg_get(pattern, "prob", 1.0))
            if not math.isfinite(prob) or prob < 0.0:
                raise ValueError(f"{label}[{idx}].prob must be finite and >= 0")
            total_prob += float(prob)
        if total_prob <= 0.0:
            raise ValueError(f"{label} probabilities must sum to > 0")

    def _phase_b_active_k_choices(self) -> tuple[List[int], Optional[List[float]]]:
        choices = list(self.phase_b_K_choices)
        probs = None if self.phase_b_K_probs is None else list(self.phase_b_K_probs)
        active_start = None
        for stage in self.phase_b_curriculum:
            start = int(self._cfg_get(stage, "start_step", 0) or 0)
            if int(self.global_step) < int(start):
                continue
            if active_start is None or int(start) >= int(active_start):
                active_start = int(start)
                choices = [int(x) for x in list(self._cfg_get(stage, "K_choices", choices) or choices)]
                raw_probs = self._cfg_get(stage, "K_probs", None)
                probs = self._optional_float_list(raw_probs) if raw_probs is not None else None
        return choices, probs

    def _phase_b_active_repeat_patterns(self) -> List[Dict[str, Any]]:
        patterns = [dict(x) for x in self.phase_b_repeat_patterns]
        active_start = None
        for stage in self.phase_b_curriculum:
            start = int(self._cfg_get(stage, "start_step", 0) or 0)
            if int(self.global_step) < int(start):
                continue
            raw_patterns = self._cfg_get(stage, "repeat_patterns", None)
            if raw_patterns is None:
                continue
            if active_start is None or int(start) >= int(active_start):
                active_start = int(start)
                patterns = [dict(x) for x in list(raw_patterns or [])]
        return patterns

    def _sample_phase_b_repeat_pattern(self) -> Dict[str, Any]:
        patterns = self._phase_b_active_repeat_patterns()
        self._validate_phase_b_repeat_patterns(
            patterns,
            blocks_per_episode=int(self.blocks_per_episode),
            label="active phase_B.rollout.repeat_patterns",
        )
        weights = [float(self._cfg_get(pattern, "prob", 1.0)) for pattern in patterns]
        return dict(random.choices(patterns, weights=weights, k=1)[0])

    def _sample_phase_a_nearby_frames(
        self,
        *,
        sidx: SegmentIndexLike,
        source_keyframe_idx: int,
        source_frame: int,
        existing_loss_frames: List[int],
        num_frames: int,
        policy: str,
        adjacent_radius: int,
        random_fill: bool,
    ) -> tuple[List[int], int]:
        if int(num_frames) <= 0:
            return [], 0
        if str(policy) != "adjacent_then_random_same_keyframe":
            raise ValueError(f"unsupported Phase A nearby policy={policy!r}")
        frames = sorted(int(x) for x in list(sidx.keyframe_to_frames[int(source_keyframe_idx)]))
        if len(frames) == 0:
            return [], 0
        excluded: Set[int] = set()
        if self.phase_a_nearby_exclude_existing_block_loss:
            excluded.update(int(x) for x in existing_loss_frames)
        if self.phase_a_nearby_exclude_source:
            excluded.add(int(source_frame))

        out: List[int] = []
        if int(source_frame) in frames:
            source_pos = frames.index(int(source_frame))
            for radius in range(1, int(adjacent_radius) + 1):
                for cand_pos in (source_pos - radius, source_pos + radius):
                    if cand_pos < 0 or cand_pos >= len(frames):
                        continue
                    cand = int(frames[cand_pos])
                    if cand in excluded or cand in out:
                        continue
                    out.append(cand)
                    if len(out) >= int(num_frames):
                        return [int(x) for x in out], len(frames)

        if bool(random_fill) and len(out) < int(num_frames):
            out_set = set(int(x) for x in out)
            candidates = [int(f) for f in frames if int(f) not in excluded and int(f) not in out_set]
            need = int(num_frames) - len(out)
            if len(candidates) < need:
                if self.phase_a_nearby_insufficient_policy == "skip":
                    return [int(x) for x in out], int(len(out) + len(candidates))
                raise ValueError(
                    f"not enough nearby frames for keyframe={int(source_keyframe_idx)} "
                    f"need={need} candidates={len(candidates)}"
                )
            out.extend(self._sample_no_replace(candidates, need))
        return [int(x) for x in out[: int(num_frames)]], len(frames)

    def _frame_refs_for_frames_with_cap(self, num_cams: int, frames: List[int], max_refs: int) -> tuple[List[int], List[ImageRef]]:
        if int(max_refs) <= 0:
            return [], []
        max_frames = max(int(max_refs) // max(int(num_cams), 1), 0)
        selected_frames = [int(x) for x in frames[:max_frames]]
        return selected_frames, self._frame_targets_to_image_refs(int(num_cams), selected_frames)

    def _build_phase_a_block_unroll_plan(self, st: Dict[str, Any]) -> ViewSetRolloutBatchV9:
        bcur = int(st["block_cursor"])
        source_keyframe_idx = int(st["source_keyframe_idx"])
        source_frame = int(st["current_source_frame_idx"])
        num_cams = int(st["num_cams"])
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        inner_K = self._sample_inner_K(self.phase_a_inner_K_choices, self.phase_a_inner_K_probs)

        evidence_refs = self._frame_targets_to_image_refs(num_cams, [source_frame])
        block_loss_frames = [int(source_frame)]
        block_loss_refs = self._frame_targets_to_image_refs(num_cams, block_loss_frames)

        nearby_frames: List[int] = []
        nearby_refs_all: List[ImageRef] = []
        if self.phase_a_nearby_enable:
            sampled_frames, _num_candidates = self._sample_phase_a_nearby_frames(
                sidx=sidx,
                source_keyframe_idx=source_keyframe_idx,
                source_frame=source_frame,
                existing_loss_frames=block_loss_frames,
                num_frames=int(self.phase_a_nearby_frames_per_block),
                policy=str(self.phase_a_nearby_policy),
                adjacent_radius=int(self.phase_a_nearby_adjacent_radius),
                random_fill=bool(self.phase_a_nearby_random_fill),
            )
            nearby_frames, nearby_refs_all = self._frame_refs_for_frames_with_cap(
                num_cams=num_cams,
                frames=[int(x) for x in sampled_frames],
                max_refs=int(self.phase_a_nearby_max_refs_per_step),
            )

        steps: List[StepPlanV9] = []
        for k in range(int(inner_K)):
            use_near = bool(self.phase_a_nearby_enable) and (
                not bool(self.phase_a_nearby_final_step_only) or int(k) == int(inner_K) - 1
            )
            steps.append(
                StepPlanV9(
                    step_idx=int(k),
                    source_keyframe_idx=int(source_keyframe_idx),
                    source_frame_idx=int(source_frame),
                    block_idx=int(bcur),
                    evidence_refs=[tuple(x) for x in evidence_refs],
                    block_loss_refs=[tuple(x) for x in block_loss_refs],
                    nearby_loss_refs=[tuple(x) for x in nearby_refs_all] if use_near else [],
                    prefix_loss_refs=[],
                    query_label_refs=[],
                    aux_loss_refs=[],
                    evidence_frame_indices=[int(source_frame)],
                    loss_frame_indices=[int(x) for x in block_loss_frames],
                    nearby_frame_indices=[int(x) for x in nearby_frames] if use_near else [],
                    query_frame_indices=[],
                )
            )
        return self._make_batch_plan(st=st, inner_K=int(inner_K), steps=steps, query_label_refs=[], aux_loss_refs=[])

    def _sample_event_blocks(self, st: Dict[str, Any], K: int) -> List[int]:
        blocks = list(range(int(self._episode_num_blocks_from_state(st))))
        if len(blocks) == 0:
            raise ValueError("Phase B rollout requires non-empty episode blocks")
        if self.phase_b_sample_event_frames != "random_blocks_in_episode":
            raise ValueError(f"unsupported Phase B sample_event_frames={self.phase_b_sample_event_frames!r}")
        if bool(self.phase_b_distinct_event_frames):
            if int(K) > len(blocks):
                raise ValueError(
                    "Phase B distinct_event_frames=true forbids sampled K greater than available episode blocks: "
                    f"K={int(K)} available_blocks={len(blocks)}"
                )
            out = self._sample_no_replace(blocks, int(K))
        else:
            out = [int(random.choice(blocks)) for _ in range(int(K))]
        if str(self.phase_b_event_order) == "chronological":
            out = sorted(int(x) for x in out)
        return [int(x) for x in out]

    def _phase_b_tbptt_stream_state(self, st: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(st.get("phase_b_tbptt") or {})
        return {
            "cursor": int(state.get("cursor", 0) or 0),
            "chunk_idx": int(state.get("chunk_idx", 0) or 0),
            "written_frames": [int(x) for x in list(state.get("written_frames", []) or [])],
        }

    def _select_event_blocks_sequential(self, st: Dict[str, Any], K: int) -> tuple[List[int], Dict[str, Any]]:
        blocks = list(range(int(self._episode_num_blocks_from_state(st))))
        if len(blocks) == 0:
            raise ValueError("Phase B rollout requires non-empty episode blocks")
        stream_state = self._phase_b_tbptt_stream_state(st)
        cursor = int(stream_state["cursor"])
        if cursor < 0 or cursor > len(blocks):
            raise ValueError(f"Phase B TBPTT cursor out of range: cursor={cursor} blocks={len(blocks)}")
        if cursor == len(blocks):
            raise ValueError("Phase B TBPTT cursor is exhausted; episode should have ended")
        end = min(cursor + int(K), len(blocks))
        selected = [int(x) for x in blocks[cursor:end]]
        if not selected:
            raise ValueError("Phase B TBPTT selected an empty chunk")
        meta = {
            "enable": True,
            "strict": True,
            "stream_id": "default",
            "chunk_idx": int(stream_state["chunk_idx"]),
            "is_first_chunk": bool(cursor == 0),
            "is_last_chunk": bool(end >= len(blocks)),
            "start_block_idx": int(cursor),
            "end_block_idx": int(end),
            "event_block_indices": [int(x) for x in selected],
            "prior_written_frames": [int(x) for x in stream_state["written_frames"]],
        }
        return selected, meta

    def _select_event_blocks_grouped_repeat(
        self,
        st: Dict[str, Any],
        blocks_per_chunk: int,
    ) -> tuple[List[int], Dict[str, Any]]:
        blocks = list(range(int(self._episode_num_blocks_from_state(st))))
        if len(blocks) == 0:
            raise ValueError("Phase B grouped repeat rollout requires non-empty episode blocks")
        stream_state = self._phase_b_tbptt_stream_state(st)
        cursor = int(stream_state["cursor"])
        if cursor < 0 or cursor > len(blocks):
            raise ValueError(f"Phase B grouped TBPTT cursor out of range: cursor={cursor} blocks={len(blocks)}")
        if cursor == len(blocks):
            raise ValueError("Phase B grouped TBPTT cursor is exhausted; episode should have ended")
        end = min(cursor + int(blocks_per_chunk), len(blocks))
        selected = [int(x) for x in blocks[cursor:end]]
        if not selected:
            raise ValueError("Phase B grouped TBPTT selected an empty chunk")
        if len(selected) < int(blocks_per_chunk) and not self.phase_b_allow_short_final_chunk:
            raise ValueError("Phase B grouped repeat final chunk is shorter than blocks_per_chunk")
        meta = {
            "enable": True,
            "strict": True,
            "stream_id": "grouped_repeat",
            "chunk_idx": int(stream_state["chunk_idx"]),
            "is_first_chunk": bool(cursor == 0),
            "is_last_chunk": bool(end >= len(blocks)),
            "start_block_idx": int(cursor),
            "end_block_idx": int(end),
            "event_block_indices": [int(x) for x in selected],
            "prior_written_frames": [int(x) for x in stream_state["written_frames"]],
            "grouped_repeat": True,
        }
        return selected, meta

    def _select_event_block_repeat(self, st: Dict[str, Any], K: int) -> tuple[List[int], Dict[str, Any]]:
        if int(K) != 1:
            raise ValueError("Phase B episode_block_repeat_tbptt requires inner K=1")
        blocks = list(range(int(self._episode_num_blocks_from_state(st))))
        if len(blocks) == 0:
            raise ValueError("Phase B rollout requires non-empty episode blocks")
        block_idx = int(st.get("block_cursor", 0))
        if block_idx < 0 or block_idx >= len(blocks):
            raise ValueError(f"Phase B repeat block cursor out of range: block={block_idx} blocks={len(blocks)}")
        stream_state = self._phase_b_tbptt_stream_state(st)
        episode_step_cursor = int(st.get("episode_step_cursor", 0))
        episode_total_steps = int(self._episode_total_steps_from_state(st))
        meta = {
            "enable": True,
            "strict": False,
            "stream_id": "block_repeat",
            "chunk_idx": int(stream_state["chunk_idx"]),
            "is_first_chunk": bool(episode_step_cursor == 0),
            "is_last_chunk": bool(episode_step_cursor + 1 >= episode_total_steps),
            "start_block_idx": int(block_idx),
            "end_block_idx": int(block_idx + 1),
            "event_block_indices": [int(block_idx)],
            "prior_written_frames": [int(x) for x in stream_state["written_frames"]],
            "repeat_block_mode": True,
            "episode_step_cursor": int(episode_step_cursor),
        }
        return [int(block_idx)], meta

    def _commit_phase_b_tbptt_plan(self, st: Dict[str, Any], plan: ViewSetRolloutBatchV9) -> None:
        if self.phase != "phase_B_viewset_rollout" or self.phase_b_rollout_mode not in (
            "episode_stream_tbptt",
            "episode_block_repeat_tbptt",
            "episode_grouped_repeat_tbptt",
        ):
            return
        tbptt = dict((plan.request_meta or {}).get("tbptt") or {})
        if not tbptt:
            raise ValueError(f"Phase B {self.phase_b_rollout_mode} plan is missing request_meta.tbptt")
        prior = {int(x) for x in list(tbptt.get("prior_written_frames", []) or [])}
        current = {int(x) for x in list(tbptt.get("event_frame_indices", []) or [])}
        cursor = (
            int(tbptt["end_block_idx"])
            if self.phase_b_rollout_mode in ("episode_stream_tbptt", "episode_grouped_repeat_tbptt")
            else int(st.get("block_cursor", 0))
        )
        st["phase_b_tbptt"] = {
            "cursor": int(cursor),
            "chunk_idx": int(tbptt["chunk_idx"]) + 1,
            "written_frames": sorted(prior | current),
        }

    def _sample_prefix_frames(self, *, written_frames: List[int], current_frame: int, step_idx: int, inner_K: int) -> List[int]:
        if str(self.phase_b_prefix_policy) != "current_plus_random_previous":
            raise ValueError(f"unsupported Phase B prefix policy={self.phase_b_prefix_policy!r}")
        max_views = (
            int(self.phase_b_prefix_final_views)
            if int(step_idx) == int(inner_K) - 1
            else int(self.phase_b_prefix_intermediate_views)
        )
        max_frames_by_ref_cap = max(int(self.phase_b_prefix_max_refs_per_step) // max(int(getattr(self, "_last_num_cams", 1)), 1), 0)
        max_views = min(int(max_views), int(max_frames_by_ref_cap)) if max_frames_by_ref_cap > 0 else 0
        if max_views <= 0:
            return []
        out = [int(current_frame)]
        previous = [int(x) for x in written_frames if int(x) != int(current_frame)]
        need = max(int(max_views) - 1, 0)
        if need > 0 and previous:
            out.extend(self._sample_no_replace(previous, min(need, len(previous))))
        return [int(x) for x in out]

    def _sample_heldout_query_frames(
        self,
        *,
        sidx: SegmentIndexLike,
        st: Dict[str, Any],
        event_frames: List[int],
        exclude_frames: Set[int],
    ) -> List[int]:
        if not self.phase_b_query_enable or int(self.phase_b_query_frames_per_rollout) <= 0:
            return []
        if str(self.phase_b_query_frame_policy) != "heldout_inside_event_span":
            raise ValueError(f"unsupported query_frame_policy={self.phase_b_query_frame_policy!r}")
        if len(event_frames) == 0:
            return []
        train_set = set(int(x) for x in getattr(sidx, "train_frame_set", []))
        if not train_set:
            train_set = set(int(x) for x in getattr(sidx, "frame_indices", []))
        lo = min(int(x) for x in event_frames)
        hi = max(int(x) for x in event_frames)
        candidates = [
            int(f)
            for f in sorted(train_set)
            if int(lo) <= int(f) <= int(hi) and int(f) not in exclude_frames
        ]
        if len(candidates) == 0:
            episode_frames: List[int] = []
            for kf in [int(x) for x in st.get("keyframe_window", [])]:
                episode_frames.extend(int(x) for x in list(sidx.keyframe_to_frames[int(kf)]))
            candidates = [
                int(f)
                for f in sorted(set(episode_frames))
                if int(f) in train_set and int(f) not in exclude_frames
            ]
        need = int(self.phase_b_query_frames_per_rollout)
        if len(candidates) < need:
            if self.v9_fail_fast:
                raise ValueError(f"Phase B query needs {need} held-out frames, got {len(candidates)}")
            need = len(candidates)
        if need <= 0:
            return []
        return self._sample_no_replace(candidates, need)

    def _build_phase_b_grouped_repeat_plan(self, st: Dict[str, Any]) -> ViewSetRolloutBatchV9:
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        pattern = self._sample_phase_b_repeat_pattern()
        repeats_per_block = int(self._cfg_get(pattern, "repeats_per_block", 0) or 0)
        blocks_per_chunk = int(self._cfg_get(pattern, "blocks_per_chunk", 0) or 0)
        pattern_name = str(self._cfg_get(pattern, "name", f"r{repeats_per_block}_b{blocks_per_chunk}"))
        unique_blocks, tbptt_meta = self._select_event_blocks_grouped_repeat(
            st,
            blocks_per_chunk=int(blocks_per_chunk),
        )
        inner_K = int(repeats_per_block * len(unique_blocks))
        if inner_K < 1:
            raise ValueError("Phase B grouped repeat generated empty inner rollout")
        if inner_K > int(self.phase_b_repeat_max_inner_K):
            raise ValueError(
                "Phase B grouped repeat actual inner_K exceeds max_inner_K: "
                f"inner_K={int(inner_K)} max_inner_K={int(self.phase_b_repeat_max_inner_K)}"
            )

        num_cams = int(st["num_cams"])
        self._last_num_cams = int(num_cams)
        prior_written_frames = [int(x) for x in list(tbptt_meta.get("prior_written_frames", []) or [])]
        prior_last_frame = max(prior_written_frames) if prior_written_frames else None
        written_unique_frames: List[int] = [int(x) for x in prior_written_frames]
        chunk_unique_frames: List[int] = []
        steps: List[StepPlanV9] = []
        step_block_indices: List[int] = []
        step_repeat_indices: List[int] = []
        step_memory_write_flags: List[bool] = []
        step_source_frame_indices: List[int] = []

        for block_idx in unique_blocks:
            source_keyframe_idx, source_frame = self._sample_source_frame_for_block(
                st=st,
                sidx=sidx,
                block_idx=int(block_idx),
            )
            if prior_last_frame is not None and int(source_frame) <= int(prior_last_frame):
                raise ValueError(
                    "episode_grouped_repeat_tbptt source frames must be chronological across chunks: "
                    f"source_frame={int(source_frame)} prior_last_frame={int(prior_last_frame)}"
                )
            if chunk_unique_frames and int(source_frame) <= int(chunk_unique_frames[-1]):
                raise ValueError(
                    "episode_grouped_repeat_tbptt requires source frames strictly chronological across blocks"
                )
            chunk_unique_frames.append(int(source_frame))

            for repeat_idx in range(int(repeats_per_block)):
                step_idx = int(len(steps))
                memory_write = bool(int(repeat_idx) == 0)
                evidence_refs = self._frame_targets_to_image_refs(num_cams, [int(source_frame)])
                if memory_write and int(source_frame) not in set(int(x) for x in written_unique_frames):
                    written_unique_frames.append(int(source_frame))
                prefix_frames = self._sample_prefix_frames(
                    written_frames=[int(x) for x in written_unique_frames],
                    current_frame=int(source_frame),
                    step_idx=int(step_idx),
                    inner_K=int(inner_K),
                )
                prefix_refs = self._frame_targets_to_image_refs(num_cams, prefix_frames)
                steps.append(
                    StepPlanV9(
                        step_idx=int(step_idx),
                        source_keyframe_idx=int(source_keyframe_idx),
                        source_frame_idx=int(source_frame),
                        block_idx=int(block_idx),
                        evidence_refs=[tuple(x) for x in evidence_refs],
                        block_loss_refs=[],
                        nearby_loss_refs=[],
                        prefix_loss_refs=[tuple(x) for x in prefix_refs],
                        query_label_refs=[],
                        aux_loss_refs=[],
                        evidence_frame_indices=[int(source_frame)],
                        loss_frame_indices=[int(x) for x in prefix_frames],
                        nearby_frame_indices=[],
                        query_frame_indices=[],
                    )
                )
                step_block_indices.append(int(block_idx))
                step_repeat_indices.append(int(repeat_idx))
                step_memory_write_flags.append(bool(memory_write))
                step_source_frame_indices.append(int(source_frame))

        exclude_frames = set(int(x) for x in written_unique_frames) if self.phase_b_query_exclude_event_frames else set()
        query_frames = self._sample_heldout_query_frames(
            sidx=sidx,
            st=st,
            event_frames=[int(x) for x in chunk_unique_frames],
            exclude_frames=exclude_frames,
        )
        query_refs = self._frame_targets_to_image_refs(num_cams, query_frames)
        tbptt_meta = dict(tbptt_meta)
        tbptt_meta["event_frame_indices"] = [int(x) for x in chunk_unique_frames]
        tbptt_meta["step_event_frame_indices"] = [int(x) for x in step_source_frame_indices]
        tbptt_meta["prior_written_refs"] = self._frame_targets_to_image_refs(num_cams, prior_written_frames)
        tbptt_meta["query_exclude_frames"] = sorted(int(x) for x in exclude_frames)
        repeat_meta = {
            "mode": "episode_grouped_repeat_tbptt",
            "pattern_name": str(pattern_name),
            "repeats_per_block": int(repeats_per_block),
            "blocks_per_chunk": int(blocks_per_chunk),
            "actual_blocks_per_chunk": int(len(unique_blocks)),
            "inner_K": int(inner_K),
            "repeat_source_frame_policy": str(self.phase_b_repeat_source_frame_policy),
            "repeat_memory_write_policy": str(self.phase_b_repeat_memory_write_policy),
            "evidence_recompute_policy": str(self.phase_b_evidence_recompute_policy),
            "step_block_indices": [int(x) for x in step_block_indices],
            "step_repeat_indices": [int(x) for x in step_repeat_indices],
            "step_memory_write_flags": [bool(x) for x in step_memory_write_flags],
            "step_source_frame_indices": [int(x) for x in step_source_frame_indices],
            "unique_event_block_indices": [int(x) for x in unique_blocks],
            "unique_event_frame_indices": [int(x) for x in chunk_unique_frames],
        }
        return self._make_batch_plan(
            st=st,
            inner_K=int(inner_K),
            steps=steps,
            query_label_refs=[tuple(x) for x in query_refs],
            aux_loss_refs=[],
            query_frame_indices=[int(x) for x in query_frames],
            tbptt_meta=tbptt_meta,
            repeat_meta=repeat_meta,
        )

    def _build_phase_b_rollout_plan(self, st: Dict[str, Any]) -> ViewSetRolloutBatchV9:
        if self.phase_b_rollout_mode == "episode_grouped_repeat_tbptt":
            return self._build_phase_b_grouped_repeat_plan(st)
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        k_choices, k_probs = self._phase_b_active_k_choices()
        inner_K = self._sample_inner_K(k_choices, k_probs)
        tbptt_meta: Optional[Dict[str, Any]] = None
        if self.phase_b_rollout_mode == "episode_stream_tbptt":
            selected_blocks, tbptt_meta = self._select_event_blocks_sequential(st, inner_K)
        elif self.phase_b_rollout_mode == "episode_block_repeat_tbptt":
            selected_blocks, tbptt_meta = self._select_event_block_repeat(st, inner_K)
        else:
            selected_blocks = self._sample_event_blocks(st, inner_K)
        inner_K = int(len(selected_blocks))
        num_cams = int(st["num_cams"])
        self._last_num_cams = int(num_cams)
        written_frames: List[int] = (
            [int(x) for x in list((tbptt_meta or {}).get("prior_written_frames", []) or [])]
            if tbptt_meta is not None
            else []
        )
        chunk_event_frames: List[int] = []
        steps: List[StepPlanV9] = []
        for k, block_idx in enumerate(selected_blocks):
            source_keyframe_idx, source_frame = self._sample_source_frame_for_block(
                st=st,
                sidx=sidx,
                block_idx=int(block_idx),
            )
            evidence_refs = self._frame_targets_to_image_refs(num_cams, [int(source_frame)])
            written_frames.append(int(source_frame))
            chunk_event_frames.append(int(source_frame))
            prefix_frames = self._sample_prefix_frames(
                written_frames=[int(x) for x in written_frames],
                current_frame=int(source_frame),
                step_idx=int(k),
                inner_K=int(inner_K),
            )
            prefix_refs = self._frame_targets_to_image_refs(num_cams, prefix_frames)
            steps.append(
                StepPlanV9(
                    step_idx=int(k),
                    source_keyframe_idx=int(source_keyframe_idx),
                    source_frame_idx=int(source_frame),
                    block_idx=int(block_idx),
                    evidence_refs=[tuple(x) for x in evidence_refs],
                    block_loss_refs=[],
                    nearby_loss_refs=[],
                    prefix_loss_refs=[tuple(x) for x in prefix_refs],
                    query_label_refs=[],
                    aux_loss_refs=[],
                    evidence_frame_indices=[int(source_frame)],
                    loss_frame_indices=[int(x) for x in prefix_frames],
                    nearby_frame_indices=[],
                    query_frame_indices=[],
                )
            )

        exclude_frames = set(int(x) for x in written_frames) if self.phase_b_query_exclude_event_frames else set()
        query_frames = self._sample_heldout_query_frames(
            sidx=sidx,
            st=st,
            event_frames=[int(x) for x in chunk_event_frames],
            exclude_frames=exclude_frames,
        )
        query_refs = self._frame_targets_to_image_refs(num_cams, query_frames)
        if tbptt_meta is not None:
            prior_written_frames = [int(x) for x in list(tbptt_meta.get("prior_written_frames", []) or [])]
            tbptt_meta = dict(tbptt_meta)
            tbptt_meta["event_frame_indices"] = [int(x) for x in chunk_event_frames]
            tbptt_meta["prior_written_refs"] = self._frame_targets_to_image_refs(num_cams, prior_written_frames)
            tbptt_meta["query_exclude_frames"] = sorted(int(x) for x in exclude_frames)
        return self._make_batch_plan(
            st=st,
            inner_K=int(inner_K),
            steps=steps,
            query_label_refs=[tuple(x) for x in query_refs],
            aux_loss_refs=[],
            query_frame_indices=[int(x) for x in query_frames],
            tbptt_meta=tbptt_meta,
        )

    def _make_batch_plan(
        self,
        *,
        st: Dict[str, Any],
        inner_K: int,
        steps: List[StepPlanV9],
        query_label_refs: List[ImageRef],
        aux_loss_refs: List[ImageRef],
        query_frame_indices: Optional[List[int]] = None,
        tbptt_meta: Optional[Dict[str, Any]] = None,
        repeat_meta: Optional[Dict[str, Any]] = None,
    ) -> ViewSetRolloutBatchV9:
        query_frames = [int(x) for x in (query_frame_indices or [])]
        if query_frames:
            new_steps: List[StepPlanV9] = []
            for step in steps:
                new_steps.append(
                    dataclasses.replace(
                        step,
                        query_label_refs=[tuple(x) for x in query_label_refs],
                        query_frame_indices=[int(x) for x in query_frames],
                    )
                )
            steps = new_steps
        plan = ViewSetRolloutBatchV9(
            scheduler_version="v9",
            phase=self.phase,
            scene_id=int(st["scene_id"]),
            segment_id=int(st["segment_id"]),
            episode_id=int(st["episode_idx_global"]),
            episode_start_keyframe_pos=int(st["episode_start_keyframe_pos"]),
            keyframe_window=[int(x) for x in st["keyframe_window"]],
            frame_chain=[int(x) for x in st["frame_chain"]],
            num_cams=int(st["num_cams"]),
            inner_K=int(inner_K),
            steps=list(steps),
            evidence_refs_by_step=[[tuple(x) for x in step.evidence_refs] for step in steps],
            block_loss_refs_by_step=[[tuple(x) for x in step.block_loss_refs] for step in steps],
            nearby_loss_refs_by_step=[[tuple(x) for x in step.nearby_loss_refs] for step in steps],
            prefix_loss_refs_by_step=[[tuple(x) for x in step.prefix_loss_refs] for step in steps],
            query_label_refs=[tuple(x) for x in query_label_refs],
            aux_loss_refs=[tuple(x) for x in aux_loss_refs],
        )
        request_meta = self._build_request_meta_v9(plan)
        if tbptt_meta is not None:
            request_meta["tbptt"] = dict(tbptt_meta)
        if repeat_meta is not None:
            request_meta["phase_b_repeat"] = dict(repeat_meta)
        leakage_check = dict(request_meta.get("leakage_check") or {})
        return dataclasses.replace(plan, request_meta=request_meta, leakage_check=leakage_check)

    def _build_request_meta_v9(self, plan: ViewSetRolloutBatchV9) -> Dict[str, Any]:
        evidence_refs = self._dedupe_image_refs_keep_order(self._flatten(plan.evidence_refs_by_step))
        block_refs_raw = self._flatten(plan.block_loss_refs_by_step)
        nearby_refs_raw = self._flatten(plan.nearby_loss_refs_by_step)
        prefix_refs_raw = self._flatten(plan.prefix_loss_refs_by_step)
        render_refs_raw = block_refs_raw + nearby_refs_raw + prefix_refs_raw
        render_roles_raw = (
            ["block_loss" for _ in block_refs_raw]
            + ["nearby_loss" for _ in nearby_refs_raw]
            + ["prefix_loss" for _ in prefix_refs_raw]
        )
        render_refs, render_roles = self._dedupe_refs_roles_keep_order(render_refs_raw, render_roles_raw)
        prefix_refs = [tuple(ref) for ref, role in zip(render_refs, render_roles) if str(role) == "prefix_loss"]
        query_refs = self._dedupe_image_refs_keep_order([tuple(x) for x in plan.query_label_refs])
        aux_refs = self._dedupe_image_refs_keep_order([tuple(x) for x in plan.aux_loss_refs])
        non_evidence_refs = self._dedupe_image_refs_keep_order(render_refs + query_refs + aux_refs)
        nearby_frames = sorted({int(ref[0]) for ref in nearby_refs_raw})
        query_frames = sorted({int(ref[0]) for ref in query_refs})
        leakage_check = {
            "nearby_evidence_overlap": int(len(set(nearby_refs_raw) & set(evidence_refs))),
            "query_evidence_overlap": int(len(set(query_refs) & set(evidence_refs))),
            "aux_evidence_overlap": int(len(set(aux_refs) & set(evidence_refs))),
            "same_scene_segment_required": bool(self.v9_same_scene_segment_required),
            "num_evidence_refs": int(len(evidence_refs)),
            "num_render_loss_refs": int(len(render_refs)),
            "num_query_label_refs": int(len(query_refs)),
            "target_role_count_match": bool(len(render_refs) == len(render_roles)),
        }
        mask_policy = {
            "phase_a_block_loss_mask": str(self.phase_a_block_loss_mask),
            "phase_a_nearby_loss_mask": str(self.phase_a_nearby_loss_mask),
            "phase_b_vsm_scope": str(self.phase_b_vsm_scope),
            "phase_b_evidence_mask": str(self.phase_b_evidence_mask),
            "phase_b_prefix_loss_mask": str(self.phase_b_prefix_loss_mask),
            "phase_b_query_label_mask": str(self.phase_b_query_label_mask),
        }
        vsm_reset_policy = {
            "reset_vsm_on_episode_end": bool(self.phase_b_reset_vsm_on_episode_end),
            "episode_id": int(plan.episode_id),
        }
        return {
            "scheduler_version": "v9",
            "scheduler_phase": str(plan.phase),
            "scene_id": int(plan.scene_id),
            "segment_id": int(plan.segment_id),
            "episode_id": int(plan.episode_id),
            "episode_idx_global": int(plan.episode_id),
            "episode_start_keyframe_pos": int(plan.episode_start_keyframe_pos),
            "num_cams": int(plan.num_cams),
            "inner_K": int(plan.inner_K),
            "evidence_refs_by_step": [[tuple(x) for x in refs] for refs in plan.evidence_refs_by_step],
            "block_loss_refs_by_step": [[tuple(x) for x in refs] for refs in plan.block_loss_refs_by_step],
            "nearby_loss_refs_by_step": [[tuple(x) for x in refs] for refs in plan.nearby_loss_refs_by_step],
            "prefix_loss_refs_by_step": [[tuple(x) for x in refs] for refs in plan.prefix_loss_refs_by_step],
            "query_label_refs": [tuple(x) for x in query_refs],
            "aux_loss_refs": [tuple(x) for x in aux_refs],
            "flat_evidence_refs": [tuple(x) for x in evidence_refs],
            "flat_render_loss_refs": [tuple(x) for x in render_refs],
            "flat_non_evidence_refs": [tuple(x) for x in non_evidence_refs],
            # Compatibility only. Trainers should use flat_render_loss_refs/query_label_refs
            # or the by-step role refs instead of treating this as render supervision.
            "flat_loss_refs": [tuple(x) for x in non_evidence_refs],
            "source_image_refs": [tuple(x) for x in evidence_refs],
            "source_image_ref": tuple(evidence_refs[0]) if evidence_refs else None,
            "target_image_refs": [tuple(x) for x in render_refs],
            "target_image_roles": [str(x) for x in render_roles],
            "nearby_loss_frame_indices": [int(x) for x in nearby_frames],
            "nearby_frame_indices": [int(x) for x in nearby_frames],
            "query_label_frame_indices": [int(x) for x in query_frames],
            "role_policy": {
                "evidence": "update_only",
                "block_loss": "loss_only",
                "nearby_loss": "loss_only",
                "prefix_loss": "loss_only",
                "query_label": "label_only",
                "aux_loss": "loss_only",
            },
            "mask_policy": mask_policy,
            "vsm_reset_policy": vsm_reset_policy,
            "phase_b_rollout_mode": str(self.phase_b_rollout_mode),
            "role_groups": [
                {
                    "role": "evidence",
                    "refs": [tuple(x) for x in evidence_refs],
                    "image_roles": ["evidence" for _ in evidence_refs],
                    "allow_update_evidence": True,
                    "allow_render_loss": False,
                    "allow_query_label": False,
                    "mask_policy": str(self.phase_b_evidence_mask),
                },
                {
                    "role": "render_loss",
                    "refs": [tuple(x) for x in render_refs],
                    "image_roles": [str(x) for x in render_roles],
                    "allow_update_evidence": False,
                    "allow_render_loss": True,
                    "allow_query_label": False,
                    "mask_policy": str(self.phase_b_prefix_loss_mask),
                },
                {
                    "role": "prefix_loss",
                    "refs": [tuple(x) for x in prefix_refs],
                    "image_roles": ["prefix_loss" for _ in prefix_refs],
                    "allow_update_evidence": False,
                    "allow_render_loss": True,
                    "allow_query_label": False,
                    "mask_policy": str(self.phase_b_prefix_loss_mask),
                },
                {
                    "role": "query_label",
                    "refs": [tuple(x) for x in query_refs],
                    "image_roles": ["query_label" for _ in query_refs],
                    "allow_update_evidence": False,
                    "allow_render_loss": False,
                    "allow_query_label": True,
                    "mask_policy": str(self.phase_b_query_label_mask),
                },
            ],
            "leakage_check": leakage_check,
            "assembly_mode": "image_ref_v9",
        }

    def _validate_phase_b_grouped_repeat_plan(self, plan: ViewSetRolloutBatchV9) -> None:
        meta = dict(plan.request_meta or {})
        repeat_meta = dict(meta.get("phase_b_repeat") or {})
        tbptt = dict(meta.get("tbptt") or {})
        if not repeat_meta:
            raise ValueError("episode_grouped_repeat_tbptt requires request_meta.phase_b_repeat")
        if not tbptt:
            raise ValueError("episode_grouped_repeat_tbptt requires request_meta.tbptt")
        if str(repeat_meta.get("mode", "")) != "episode_grouped_repeat_tbptt":
            raise ValueError("phase_b_repeat.mode must be episode_grouped_repeat_tbptt")
        inner_k = int(plan.inner_K)
        repeats_per_block = int(repeat_meta.get("repeats_per_block", 0) or 0)
        step_blocks = [int(x) for x in list(repeat_meta.get("step_block_indices", []) or [])]
        step_repeats = [int(x) for x in list(repeat_meta.get("step_repeat_indices", []) or [])]
        step_sources = [int(x) for x in list(repeat_meta.get("step_source_frame_indices", []) or [])]
        write_flags = [bool(x) for x in list(repeat_meta.get("step_memory_write_flags", []) or [])]
        for name, vals in (
            ("step_block_indices", step_blocks),
            ("step_repeat_indices", step_repeats),
            ("step_source_frame_indices", step_sources),
            ("step_memory_write_flags", write_flags),
        ):
            if len(vals) != inner_k:
                raise ValueError(f"phase_b_repeat.{name} length must equal inner_K")
        if len(plan.steps) != inner_k:
            raise ValueError("phase_b_repeat validation requires len(steps)==inner_K")
        if repeats_per_block < 1:
            raise ValueError("phase_b_repeat.repeats_per_block must be >= 1")
        unique_blocks = [int(x) for x in list(repeat_meta.get("unique_event_block_indices", []) or [])]
        unique_frames = [int(x) for x in list(repeat_meta.get("unique_event_frame_indices", []) or [])]
        if len(unique_blocks) == 0 or len(unique_blocks) != len(unique_frames):
            raise ValueError("phase_b_repeat unique block/frame metadata is invalid")
        if unique_frames != sorted(unique_frames) or len(set(unique_frames)) != len(unique_frames):
            raise ValueError("phase_b_repeat unique_event_frame_indices must be strictly chronological")
        if [int(x) for x in list(tbptt.get("event_frame_indices", []) or [])] != unique_frames:
            raise ValueError("tbptt.event_frame_indices must equal phase_b_repeat unique_event_frame_indices")
        if [int(x) for x in list(tbptt.get("event_block_indices", []) or [])] != unique_blocks:
            raise ValueError("tbptt.event_block_indices must equal phase_b_repeat unique_event_block_indices")
        if [int(x) for x in list(tbptt.get("step_event_frame_indices", []) or [])] != step_sources:
            raise ValueError("tbptt.step_event_frame_indices must equal phase_b_repeat step_source_frame_indices")

        for block in unique_blocks:
            positions = [idx for idx, val in enumerate(step_blocks) if int(val) == int(block)]
            if len(positions) != int(repeats_per_block):
                raise ValueError("each grouped repeat block must have repeats_per_block steps")
            if positions != list(range(positions[0], positions[0] + len(positions))):
                raise ValueError("grouped repeat block steps must be contiguous")
            source_vals = {int(step_sources[idx]) for idx in positions}
            if len(source_vals) != 1:
                raise ValueError("grouped repeat source frame must be fixed within each block")
            repeat_vals = [int(step_repeats[idx]) for idx in positions]
            if repeat_vals != list(range(int(repeats_per_block))):
                raise ValueError("grouped repeat indices must be [0, ..., repeats_per_block-1] per block")
            flag_vals = [bool(write_flags[idx]) for idx in positions]
            expected_flags = [True] + [False for _ in range(int(repeats_per_block) - 1)]
            if flag_vals != expected_flags:
                raise ValueError("grouped repeat requires exactly one first-repeat memory write per block")

        query_frames = {int(ref[0]) for ref in plan.query_label_refs}
        forbidden_query_frames = set(unique_frames)
        forbidden_query_frames.update(int(x) for x in list(tbptt.get("prior_written_frames", []) or []))
        if query_frames & forbidden_query_frames:
            raise ValueError("query_label_refs overlap grouped repeat written/evidence frames")

    def _validate_v9_plan(self, plan: ViewSetRolloutBatchV9) -> None:
        if str(plan.scheduler_version) != "v9":
            raise ValueError("expected scheduler_version=v9")
        if int(plan.inner_K) < 1:
            raise ValueError("inner_K must be >= 1")
        if len(plan.steps) != int(plan.inner_K):
            raise ValueError("len(steps) must equal inner_K")
        for attr in (
            "evidence_refs_by_step",
            "block_loss_refs_by_step",
            "nearby_loss_refs_by_step",
            "prefix_loss_refs_by_step",
        ):
            if len(getattr(plan, attr)) != int(plan.inner_K):
                raise ValueError(f"len({attr}) must equal inner_K")

        evidence = set(self._flatten(plan.evidence_refs_by_step))
        nearby = set(self._flatten(plan.nearby_loss_refs_by_step))
        prefix = set(self._flatten(plan.prefix_loss_refs_by_step))
        block_loss = set(self._flatten(plan.block_loss_refs_by_step))
        query = set(tuple(x) for x in plan.query_label_refs)
        aux = set(tuple(x) for x in plan.aux_loss_refs)
        if not evidence:
            raise ValueError("V9 requires non-empty evidence refs")
        if self.v9_nearby_not_in_evidence and nearby & evidence:
            raise ValueError("nearby_loss_refs must not overlap evidence_refs")
        if self.v9_aux_not_in_evidence and aux & evidence:
            raise ValueError("aux_loss_refs must not overlap evidence_refs")
        if self.v9_query_not_in_evidence and query & evidence:
            raise ValueError("query_label_refs must not overlap evidence_refs")

        if plan.phase == "phase_A_block_local_unroll":
            if query:
                raise ValueError("Phase A must not emit query_label_refs")
            if prefix:
                raise ValueError("Phase A must not emit prefix_loss_refs")
        if plan.phase == "phase_B_viewset_rollout":
            if nearby:
                raise ValueError("Phase B must not emit nearby_loss_refs")
            if self.phase_b_query_enable and not query:
                raise ValueError("Phase B query_observation enabled but no query_label_refs emitted")
            if self.phase_b_rollout_mode == "episode_grouped_repeat_tbptt":
                self._validate_phase_b_grouped_repeat_plan(plan)
        for step in plan.steps:
            if not step.evidence_refs:
                raise ValueError(f"step {step.step_idx} has empty evidence_refs")
            if self.phase == "phase_A_block_local_unroll" and step.nearby_loss_refs:
                sidx = self.dataset.get_segment_index(int(plan.scene_id), int(plan.segment_id))
                source_kf = int(step.source_keyframe_idx)
                for ref in step.nearby_loss_refs:
                    frame_idx = int(ref[0])
                    actual_kf = int(getattr(sidx, "frame_to_keyframe", {}).get(frame_idx, -1))
                    if actual_kf != source_kf:
                        raise ValueError("Phase A nearby frame is not in source keyframe")

        if self.v9_role_count_match_required:
            meta = dict(plan.request_meta or self._build_request_meta_v9(plan))
            refs = list(meta.get("target_image_refs") or [])
            roles = list(meta.get("target_image_roles") or [])
            if len(refs) != len(roles):
                raise ValueError(f"target_image_refs/target_image_roles mismatch: {len(refs)} vs {len(roles)}")

        if self.v9_forbid_test_refs_in_train and hasattr(self.dataset, "validate_image_ref"):
            for ref in sorted(evidence | block_loss | nearby | prefix | query | aux):
                self.dataset.validate_image_ref(int(plan.scene_id), int(plan.segment_id), tuple(ref), purpose="train")

    def _batch_from_v9_plan(self, plan: ViewSetRolloutBatchV9) -> Dict[str, Any]:
        if not hasattr(self.dataset, "_assemble_segment_batch_from_v9_request"):
            raise ValueError("TrainSchedulerV9 requires dataset._assemble_segment_batch_from_v9_request")
        return self.dataset._assemble_segment_batch_from_v9_request(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            v9_plan=plan,
            include_test=bool(self.include_test),
        )

    def _preload_refs_from_v9_plan(self, plan: ViewSetRolloutBatchV9) -> List[ImageRef]:
        return self._dedupe_image_refs_keep_order(
            self._flatten(plan.evidence_refs_by_step)
            + self._flatten(plan.block_loss_refs_by_step)
            + self._flatten(plan.nearby_loss_refs_by_step)
            + self._flatten(plan.prefix_loss_refs_by_step)
            + [tuple(x) for x in plan.query_label_refs]
            + [tuple(x) for x in plan.aux_loss_refs]
        )

    def _emit_v9_role_preload_hint(self, plan: ViewSetRolloutBatchV9) -> None:
        if not self.emit_preload_hints or not self.warm_v9_role_refs:
            return
        refs = self._preload_refs_from_v9_plan(plan)
        if not refs:
            return
        self._emit_preload_hint(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            future_image_refs=refs,
            hint_scope="v9_role_refs",
            block_idx_global=int(self.current_episode_state.get("block_idx_global", 0))
            if self.current_episode_state is not None
            else 0,
        )

    @staticmethod
    def _state_plan_key(st: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
        return (
            int(st["scene_id"]),
            int(st["segment_id"]),
            int(st["episode_idx_global"]),
            int(st["block_cursor"]),
            int(st.get("block_repeat_step", 0)),
            int(st.get("episode_step_cursor", 0)),
        )

    def _build_plan_from_state(self, st: Dict[str, Any]) -> ViewSetRolloutBatchV9:
        if self.phase == "phase_A_block_local_unroll":
            return self._build_phase_a_block_unroll_plan(st)
        if self.phase == "phase_B_viewset_rollout":
            return self._build_phase_b_rollout_plan(st)
        raise ValueError(f"unsupported scheduler_v9.phase={self.phase!r}")

    def _plan_from_state(self, st: Dict[str, Any]) -> ViewSetRolloutBatchV9:
        return self._build_plan_from_state(st)

    def _take_plan_for_state(self, st: Dict[str, Any]) -> ViewSetRolloutBatchV9:
        key = self._state_plan_key(st)
        if self._v9_prefetched_plan is not None and self._v9_prefetched_plan_key == key:
            plan = self._v9_prefetched_plan
            self._v9_prefetched_plan = None
            self._v9_prefetched_plan_key = None
            return plan
        self._v9_prefetched_plan = None
        self._v9_prefetched_plan_key = None
        return self._build_plan_from_state(st)

    def _prefetch_v9_plan_for_current_state(self) -> None:
        st = self.current_episode_state
        if st is None or not self.emit_preload_hints or not self.warm_v9_role_refs:
            self._v9_prefetched_plan = None
            self._v9_prefetched_plan_key = None
            return
        plan = self._build_plan_from_state(st)
        self._validate_v9_plan(plan)
        self._emit_v9_role_preload_hint(plan)
        self._v9_prefetched_plan = plan
        self._v9_prefetched_plan_key = self._state_plan_key(st)

    def _aligned_info(self, st: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(super()._aligned_info(st))
        info["scheduler_version"] = "v9"
        info["scheduler_phase"] = str(self.phase)
        return info

    def _set_grouped_repeat_block_state(self, st: Dict[str, Any], block_idx: int, source_frame: int) -> None:
        frame_chain = [int(x) for x in st["frame_chain"]]
        keyframe_window = [int(x) for x in st["keyframe_window"]]
        bcur = int(block_idx)
        if bcur < 0 or bcur >= len(frame_chain):
            raise ValueError(f"invalid grouped repeat block_idx={bcur}")
        num_cams = int(st["num_cams"])
        source_frame = int(source_frame)
        sidx = self.dataset.get_segment_index(int(st["scene_id"]), int(st["segment_id"]))
        source_keyframe_idx = int(getattr(sidx, "frame_to_keyframe", {}).get(source_frame, keyframe_window[bcur]))

        block_source_frames = [int(x) for x in st.get("block_current_source_frame_indices", frame_chain)]
        if len(block_source_frames) != len(frame_chain):
            block_source_frames = [int(x) for x in frame_chain]
        block_source_frames[bcur] = int(source_frame)
        target_frames = self._build_target_frames_for_block_v8(
            frame_chain=[int(x) for x in frame_chain],
            block_source_frames=[int(x) for x in block_source_frames],
            block_idx=int(bcur),
            source_frame=int(source_frame),
            visited_block_indices=set(st.get("visited_block_indices", set())),
            max_target_frames=int(self.total_target_frames),
        )
        target_frame_roles = ["source"] + ["visited" for _ in target_frames[1:]]
        source_image_ref = (int(source_frame), 0)
        source_image_refs = self._frame_targets_to_image_refs(num_cams, [int(source_frame)])
        target_image_refs = self._frame_targets_to_image_refs(num_cams, [int(x) for x in target_frames])
        target_image_roles = self._frame_roles_to_image_roles(
            num_cams=int(num_cams),
            target_frames=[int(x) for x in target_frames],
            target_frame_roles=[str(x) for x in target_frame_roles],
        )

        st["block_cursor"] = int(bcur)
        st["current_source_frame_idx"] = int(source_frame)
        st["current_target_frame_indices"] = [int(x) for x in target_frames]
        st["current_target_frame_roles"] = [str(x) for x in target_frame_roles]
        st["block_current_source_frame_indices"] = [int(x) for x in block_source_frames]
        st["source_keyframe_idx"] = int(source_keyframe_idx)
        st["source_image_ref"] = tuple(source_image_ref)
        st["source_image_refs"] = [tuple(x) for x in source_image_refs]
        st["aux_source_cam"] = int(source_image_ref[1])
        st["target_image_refs"] = [tuple(x) for x in target_image_refs]
        st["target_image_roles"] = [str(x) for x in target_image_roles]
        st["aux_image_refs"] = []
        st["aux_image_roles"] = []
        st["block_last_target_frame_indices"][int(bcur)] = [int(x) for x in target_frames]
        st["block_target_frame_roles"][int(bcur)] = [str(x) for x in target_frame_roles]
        st["block_target_image_roles"][int(bcur)] = [str(x) for x in target_image_roles]
        if int(bcur) not in st["visited_block_indices"]:
            st["visited_block_indices"].add(int(bcur))
            st["block_first_visit_order"][int(bcur)] = int(st.get("episode_step_cursor", 0))
            st["block_first_target_frame_indices"][int(bcur)] = [int(x) for x in target_frames]
        st["block_repeat_step"] = int(st["block_update_counts"][int(bcur)])
        st["block_idx_global"] = int(st["episode_base_block_idx_global"]) + int(bcur)
        st["block_idx_in_segment"] = int(st["episode_base_block_idx_in_segment"]) + int(bcur)

    def _finish_grouped_repeat_tbptt_chunk(
        self,
        *,
        st: Dict[str, Any],
        rt: Dict[str, Any],
        tbptt_meta: Dict[str, Any],
        tbptt_is_last_chunk: bool,
    ) -> None:
        tbptt_blocks = [int(x) for x in list(tbptt_meta.get("event_block_indices", []) or [])]
        tbptt_frames = [int(x) for x in list(tbptt_meta.get("event_frame_indices", []) or [])]
        if len(tbptt_blocks) == 0 or len(tbptt_blocks) != len(tbptt_frames):
            raise ValueError("grouped repeat TBPTT block/frame metadata is invalid")
        for block_idx, source_frame in zip(tbptt_blocks, tbptt_frames):
            self._set_grouped_repeat_block_state(st, int(block_idx), int(source_frame))
            if not bool(st["block_started"][int(block_idx)]):
                st["block_started"][int(block_idx)] = True
                self._emit_block_begin_for_current_state()
            st["block_update_counts"][int(block_idx)] = max(
                int(st["block_update_counts"][int(block_idx)]),
                int(self.steps_per_block),
            )
            st["block_repeat_step"] = int(st["block_update_counts"][int(block_idx)])
            if not bool(st["block_ended"][int(block_idx)]):
                self._emit_block_end_for_block(st, int(block_idx))
                st["block_ended"][int(block_idx)] = True
            self._emit_block_exit_for_block(st, int(block_idx))

        rt["block_idx_in_segment"] = max(
            int(rt["block_idx_in_segment"]),
            int(st["episode_base_block_idx_in_segment"]) + max(int(x) for x in tbptt_blocks) + 1,
        )
        episode_total_steps = int(self._episode_total_steps_from_state(st))
        tbptt_end = int(tbptt_meta.get("end_block_idx", int(st.get("episode_step_cursor", 0))))
        st["episode_step_cursor"] = int(episode_total_steps) if bool(tbptt_is_last_chunk) else int(tbptt_end)
        if int(st.get("episode_step_cursor", 0)) >= int(episode_total_steps):
            self._finalize_episode_if_needed()
            return
        next_block_idx = int(self._episode_visit_order_from_state(st)[int(st["episode_step_cursor"])])
        self._select_block(next_block_idx)

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        self._ensure_episode_state()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV9 internal state is not initialized")
        plan = self._plan_from_state(st)
        self._validate_v9_plan(plan)
        batch = self._batch_from_v9_plan(plan)
        batch["_scheduler_v9"] = dataclasses.asdict(plan)
        batch["_scheduler_v9_peek"] = True
        request_meta = dict(batch.get("request_meta") or {})
        request_meta.update(plan.request_meta)
        batch["request_meta"] = request_meta
        aligned = self._aligned_info(st)
        aligned["inner_K"] = int(plan.inner_K)
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_aligned_info"] = dict(aligned)
        batch["_scheduler_v9_aligned_info"] = dict(aligned)
        return batch

    def next_batch(self) -> Dict[str, Any]:
        self._ensure_episode_state()
        st = self.current_episode_state
        if st is None:
            raise ValueError("TrainSchedulerV9 internal state is not initialized")
        plan = self._take_plan_for_state(st)
        self._validate_v9_plan(plan)
        batch = self._batch_from_v9_plan(plan)
        request_meta = dict(batch.get("request_meta") or {})
        request_meta.update(plan.request_meta)
        batch["request_meta"] = request_meta
        batch["_scheduler_v9"] = dataclasses.asdict(plan)
        phase_b_stream_tbptt = bool(
            self.phase == "phase_B_viewset_rollout"
            and self.phase_b_rollout_mode in (
                "episode_stream_tbptt",
                "episode_block_repeat_tbptt",
                "episode_grouped_repeat_tbptt",
            )
        )
        tbptt_meta = dict(request_meta.get("tbptt") or {})
        tbptt_is_last_chunk = bool(tbptt_meta.get("is_last_chunk", False)) if phase_b_stream_tbptt else False
        if phase_b_stream_tbptt:
            self._commit_phase_b_tbptt_plan(st, plan)

        key = (int(st["scene_id"]), int(st["segment_id"]))
        rt = self._segment_runtime[key]
        rt["segment_local_step"] = int(rt["segment_local_step"]) + 1
        current_block_idx = int(st["block_cursor"])
        if self._block_order_uses_episode_visit_order():
            st["block_update_counts"][current_block_idx] = int(st["block_update_counts"][current_block_idx]) + 1
            st["block_repeat_step"] = int(st["block_update_counts"][current_block_idx])
            st["episode_step_cursor"] = int(st.get("episode_step_cursor", 0)) + 1
            if phase_b_stream_tbptt and self.phase_b_rollout_mode in (
                "episode_stream_tbptt",
                "episode_grouped_repeat_tbptt",
            ):
                tbptt_blocks = [int(x) for x in list(tbptt_meta.get("event_block_indices", []) or [])]
                for tb in tbptt_blocks:
                    if 0 <= int(tb) < len(st["block_update_counts"]):
                        st["block_update_counts"][int(tb)] = max(
                            int(st["block_update_counts"][int(tb)]),
                            int(self.steps_per_block),
                        )
                tbptt_end = int(tbptt_meta.get("end_block_idx", int(st.get("episode_step_cursor", 0))))
                st["episode_step_cursor"] = max(int(st.get("episode_step_cursor", 0)), int(tbptt_end))
        else:
            st["block_repeat_step"] = int(st["block_repeat_step"]) + 1
        self.global_step += 1

        aligned = self._aligned_info(st)
        aligned["inner_K"] = int(plan.inner_K)
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_aligned_info"] = dict(aligned)
        batch["_scheduler_v9_aligned_info"] = dict(aligned)

        if (
            self._block_order_uses_episode_visit_order()
            and phase_b_stream_tbptt
            and self.phase_b_rollout_mode == "episode_grouped_repeat_tbptt"
        ):
            self._finish_grouped_repeat_tbptt_chunk(
                st=st,
                rt=rt,
                tbptt_meta=tbptt_meta,
                tbptt_is_last_chunk=bool(tbptt_is_last_chunk),
            )
            self._prefetch_v9_plan_for_current_state()
            if hasattr(self.dataset, "maybe_log_preload_stats"):
                self.dataset.maybe_log_preload_stats(int(self.global_step))
            if hasattr(self.dataset, "maybe_log_overlap_stats"):
                self.dataset.maybe_log_overlap_stats(int(self.global_step))
            return batch

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
            if tbptt_is_last_chunk:
                st["episode_step_cursor"] = int(episode_total_steps)
            if int(st.get("episode_step_cursor", 0)) >= int(episode_total_steps):
                self._emit_block_exit_for_block(st, current_block_idx)
                self._finalize_episode_if_needed()
            else:
                next_block_idx = int(self._episode_visit_order_from_state(st)[int(st["episode_step_cursor"])])
                if int(next_block_idx) != int(current_block_idx):
                    self._emit_block_exit_for_block(st, current_block_idx)
                self._select_block(next_block_idx)
        else:
            if tbptt_is_last_chunk:
                st["block_cursor"] = max(int(self._episode_num_blocks_from_state(st)) - 1, int(st["block_cursor"]))
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

        self._prefetch_v9_plan_for_current_state()
        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
        return batch

    def get_current_info(self) -> Dict[str, Any]:
        out = dict(super().get_current_info())
        out["scheduler_version"] = "v9"
        out["scheduler_phase"] = str(self.phase)
        return out
