from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4
from datasets.train_scheduler_v8 import TrainSchedulerV8
from datasets.train_scheduler_v9 import TrainSchedulerV9
from tools.train_minimal_streetforward_stage4_3_v7_common import (
    _extract_knn_requirements_from_cfg,
    parse_include_test,
    validate_train_scene_for_fixed,
)


def build_multi_scene_dataset_v4(cfg: Any, device: torch.device) -> MultiSceneDatasetV4:
    knn_requirements = _extract_knn_requirements_from_cfg(cfg)
    return MultiSceneDatasetV4(
        dataset_cfg=cfg.dataset,
        data_cfg=cfg.data,
        device=device,
        knn_requirements=knn_requirements,
    )


def build_multi_scene_dataset_v4_for_demo(cfg: Any, device: torch.device) -> MultiSceneDatasetV4:
    knn_requirements = _extract_knn_requirements_from_cfg(cfg)
    dataset = MultiSceneDatasetV4(
        dataset_cfg=cfg.dataset,
        data_cfg=cfg.data,
        device=device,
        preload_cfg={"enable": False},
        knn_requirements=knn_requirements,
    )
    if getattr(dataset, "_preload_manager", None) is not None:
        raise ValueError("Demo dataset must not create preload manager.")
    return dataset


def _null_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    return int(x)


def resolve_fixed_scene_segment_v8(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    tr = (cfg.get("scheduler_v8") or {}).get("traversal") or {}
    return _null_int(tr.get("fixed_scene_id")), _null_int(tr.get("fixed_segment_id"))


def resolve_fixed_scene_segment_v9(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    tr = (cfg.get("scheduler_v9") or {}).get("traversal") or {}
    return _null_int(tr.get("fixed_scene_id")), _null_int(tr.get("fixed_segment_id"))


def build_train_scheduler_v8_from_cfg(cfg: Any, dataset: MultiSceneDatasetV4) -> TrainSchedulerV8:
    sv8 = cfg.get("scheduler_v8")
    if sv8 is None:
        raise ValueError("config must define scheduler_v8")
    if sv8.get("enable") is not True:
        raise ValueError("scheduler_v8.enable must be true")
    block = sv8.get("block")
    ep = sv8.get("episode")
    trav = sv8.get("traversal")
    preload = sv8.get("preload")
    execution = sv8.get("execution") or {}
    if block is None or ep is None or trav is None or preload is None:
        raise ValueError("scheduler_v8 must define block/episode/traversal/preload")
    block_order = str(execution.get("block_order", "block_major"))
    if block_order not in ("block_major", "step_major"):
        raise ValueError("scheduler_v8.execution.block_order must be one of ['block_major', 'step_major']")
    step_major_switch_interval_steps = int(execution.get("step_major_switch_interval_steps", 1))
    if step_major_switch_interval_steps < 1:
        raise ValueError("scheduler_v8.execution.step_major_switch_interval_steps must be >= 1")
    reset_policy = str(execution.get("reset_policy", "episode_end"))
    target_policy = str(ep.get("target_policy", "visited_episode_frames"))
    block_source_frame_policy = str(ep.get("block_source_frame_policy", "fixed_once_per_episode"))
    near_random_cfg = sv8.get("near_random_supervision") or {}

    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_v8(cfg)
    validate_train_scene_for_fixed(cfg, fixed_scene_id)
    include_test = parse_include_test(cfg)

    return dataset.create_train_scheduler_v8(
        steps_per_block=int(block["steps_per_block"]),
        blocks_per_episode=int(ep["blocks_per_episode"]),
        total_target_frames=int(ep["total_target_frames"]),
        include_source_frame=bool(ep["include_source_frame"]),
        frame_within_keyframe_policy=str(ep["frame_within_keyframe_policy"]),
        min_keyframes_required_policy=str(ep["min_keyframes_required_policy"]),
        traversal_mode=str(trav["mode"]),
        switch_after_episode=bool(trav["switch_after_episode"]),
        segment_order=str(trav["segment_order"]),
        scene_order=str(trav["scene_order"]),
        include_test=include_test,
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
        emit_preload_hints=bool(preload["emit_hints"]),
        warm_next_block_exact=bool(preload["warm_next_block_exact"]),
        warm_next_episode_chain=bool(preload["warm_next_episode_chain"]),
        block_order=block_order,
        step_major_switch_interval_steps=step_major_switch_interval_steps,
        target_policy=target_policy,
        reset_policy=reset_policy,
        near_random_supervision_cfg=near_random_cfg,
        block_source_frame_policy=block_source_frame_policy,
    )


def build_train_scheduler_v9_from_cfg(cfg: Any, dataset: MultiSceneDatasetV4) -> TrainSchedulerV9:
    sv9 = cfg.get("scheduler_v9")
    if sv9 is None:
        raise ValueError("config must define scheduler_v9")
    if sv9.get("enable") is not True:
        raise ValueError("scheduler_v9.enable must be true")

    block = sv9.get("block")
    ep = sv9.get("episode")
    trav = sv9.get("traversal")
    preload = sv9.get("preload")
    execution = sv9.get("execution") or {}
    role_sampling = sv9.get("role_sampling") or {}
    targets = sv9.get("targets") or {}
    history_record = sv9.get("history_record") or {}
    near_random_cfg = sv9.get("near_random_supervision") or {}

    if block is None or ep is None or trav is None or preload is None:
        raise ValueError("scheduler_v9 must define block/episode/traversal/preload")
    if not role_sampling:
        raise ValueError("scheduler_v9 must define role_sampling")
    if not targets:
        raise ValueError("scheduler_v9 must define targets")
    if not history_record:
        raise ValueError("scheduler_v9 must define history_record")

    block_order = str(execution.get("block_order", "block_major"))
    if block_order not in ("block_major", "step_major"):
        raise ValueError("scheduler_v9.execution.block_order must be one of ['block_major', 'step_major']")
    step_major_switch_interval_steps = int(execution.get("step_major_switch_interval_steps", 1))
    if step_major_switch_interval_steps < 1:
        raise ValueError("scheduler_v9.execution.step_major_switch_interval_steps must be >= 1")

    reset_policy = str(execution.get("reset_policy", "episode_end"))
    target_policy = str(ep.get("target_policy", "visited_episode_frames"))
    block_source_frame_policy = str(ep.get("block_source_frame_policy", "fixed_once_per_episode"))
    include_source_frame = bool(ep.get("include_source_frame"))
    first_step_role = str(role_sampling.get("first_step_role", "teacher"))
    teacher_prob = float(role_sampling.get("teacher_prob", 0.0))
    student_prob = float(role_sampling.get("student_prob", 0.0))
    observed_cfg = history_record.get("observed") or {}
    observed_trigger = str(observed_cfg.get("trigger", ""))
    observed_block_exit = bool(observed_cfg.get("record_on_block_exit", False))

    if target_policy != "visited_episode_frames":
        raise ValueError("SchedulerV9 requires target_policy=visited_episode_frames")
    if reset_policy != "episode_end":
        raise ValueError("SchedulerV9 requires reset_policy=episode_end")
    if not include_source_frame:
        raise ValueError("SchedulerV9 requires include_source_frame=true")
    if first_step_role != "teacher":
        raise ValueError("SchedulerV9 requires first_step_role=teacher")
    if student_prob > 0.0 and teacher_prob <= 0.0:
        raise ValueError("SchedulerV9 requires teacher_prob > 0 when student_prob > 0")
    if observed_trigger != "teacher_exit":
        raise ValueError("SchedulerV9 requires observed history trigger=teacher_exit")
    if observed_block_exit:
        raise ValueError("SchedulerV9 must not record observed support/residual on block_exit")

    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_v9(cfg)
    validate_train_scene_for_fixed(cfg, fixed_scene_id)
    include_test = parse_include_test(cfg)

    return dataset.create_train_scheduler_v9(
        steps_per_block=int(block["steps_per_block"]),
        blocks_per_episode=int(ep["blocks_per_episode"]),
        total_target_frames=int(ep["total_target_frames"]),
        include_source_frame=bool(ep["include_source_frame"]),
        frame_within_keyframe_policy=str(ep["frame_within_keyframe_policy"]),
        min_keyframes_required_policy=str(ep["min_keyframes_required_policy"]),
        traversal_mode=str(trav["mode"]),
        switch_after_episode=bool(trav["switch_after_episode"]),
        segment_order=str(trav["segment_order"]),
        scene_order=str(trav["scene_order"]),
        include_test=include_test,
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
        emit_preload_hints=bool(preload["emit_hints"]),
        warm_next_block_exact=bool(preload["warm_next_block_exact"]),
        warm_next_episode_chain=bool(preload["warm_next_episode_chain"]),
        block_order=block_order,
        step_major_switch_interval_steps=step_major_switch_interval_steps,
        target_policy=target_policy,
        reset_policy=reset_policy,
        near_random_supervision_cfg=near_random_cfg,
        block_source_frame_policy=block_source_frame_policy,
        role_sampling_cfg=role_sampling,
        targets_cfg=targets,
        history_record_cfg=history_record,
        preload_cfg=preload,
    )
