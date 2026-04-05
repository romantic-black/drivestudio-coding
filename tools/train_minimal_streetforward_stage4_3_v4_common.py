"""
Shared helpers for Stage 4.3 minimal trainers using MultiSceneDatasetV3 + TrainSchedulerV4.

Supports:
- One-segment fixed training via `one_segment.scene_id` / `one_segment.segment_id` (overrides scheduler_v4.traversal).
- Multi-scene training via `scheduler_v4.traversal.fixed_scene_id: null` and `fixed_segment_id: null`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from omegaconf import OmegaConf

from datasets.multi_scene_dataset_v3 import MultiSceneDatasetV3, TrainSchedulerV4


def build_multi_scene_dataset_v3(cfg: Any, device: torch.device) -> MultiSceneDatasetV3:
    ds_cfg = cfg.dataset
    data_cfg = cfg.data
    pc = OmegaConf.to_container(ds_cfg.pointcloud, resolve=True)
    kfc = ds_cfg.get("keyframe_split_config")
    if kfc is not None:
        kfc = OmegaConf.to_container(kfc, resolve=True)
    preload_raw = data_cfg.get("preload")
    preload_dict = OmegaConf.to_container(preload_raw, resolve=True) if preload_raw is not None else None
    ov_stats = 0
    sv4 = cfg.get("scheduler_v4")
    if sv4 is not None and sv4.get("overlap") is not None:
        ov_stats = int(sv4.overlap.get("stats_log_interval_steps", 0) or 0)
    return MultiSceneDatasetV3(
        data_cfg=data_cfg,
        train_scene_ids=list(data_cfg.train_scene_ids),
        eval_scene_ids=list(data_cfg.get("eval_scene_ids", [])),
        num_source_keyframes=int(ds_cfg.num_source_keyframes),
        num_target_keyframes=int(ds_cfg.num_target_keyframes),
        segment_overlap_ratio=float(ds_cfg.segment_overlap_ratio),
        keyframe_split_config=kfc,
        min_keyframes_per_scene=int(ds_cfg.min_keyframes_per_scene),
        min_keyframes_per_segment=int(ds_cfg.min_keyframes_per_segment),
        device=device,
        preload_scene_count=int(ds_cfg.preload_scene_count),
        segment_aabb=ds_cfg.segment_aabb,
        pointcloud_config=pc,
        preload_cfg=preload_dict,
        overlap_stats_log_interval_steps=ov_stats,
    )


def _null_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    return int(x)


def resolve_fixed_scene_segment(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    If `one_segment` defines scene_id and segment_id, use them as fixed traversal.
    Otherwise use `scheduler_v4.traversal.fixed_scene_id` / `fixed_segment_id` (may be null).
    """
    os_cfg = cfg.get("one_segment")
    if os_cfg is not None and os_cfg.get("scene_id") is not None and os_cfg.get("segment_id") is not None:
        return int(os_cfg.scene_id), int(os_cfg.segment_id)
    tr = cfg.get("scheduler_v4", {}).get("traversal") or {}
    return _null_int(tr.get("fixed_scene_id")), _null_int(tr.get("fixed_segment_id"))


def parse_include_test_v4(cfg: Any) -> bool:
    os_cfg = cfg.get("one_segment")
    if os_cfg is not None:
        return bool(os_cfg.get("include_test", True))
    ms = cfg.get("multi_scene")
    if ms is not None:
        return bool(ms.get("include_test", True))
    return True


def build_train_scheduler_v4_from_cfg(cfg: Any, dataset: MultiSceneDatasetV3) -> TrainSchedulerV4:
    sv4 = cfg.get("scheduler_v4")
    if sv4 is None:
        raise ValueError("config must define scheduler_v4")
    if sv4.get("enable") is not True:
        raise ValueError("scheduler_v4.enable must be true")
    tb = sv4.get("time_base")
    if tb is None or not hasattr(tb, "get"):
        raise ValueError(
            "scheduler_v4.time_base must be a mapping with state_write_interval_steps "
            "(check YAML indentation: state_write_interval_steps must be nested under time_base)."
        )
    sb = sv4["source_block"]
    re = sv4["reset_episode"]
    ts = sv4["target_sampling"]
    ov = sv4["overlap"]
    pl = sv4["preload"]
    include_test = parse_include_test_v4(cfg)
    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment(cfg)
    om = str(ov["mode"])
    overlap_point_sample_size = ov.get("point_sample_size")
    overlap_candidate_frame_policy = ov.get("candidate_frame_policy")
    overlap_score_type = ov.get("score_type")
    overlap_min = ov.get("overlap_min")
    if om == "none":
        overlap_point_sample_size = None
        overlap_candidate_frame_policy = None
        overlap_score_type = None
        overlap_min = None
    return dataset.create_train_scheduler_v4(
        state_write_interval_steps=int(tb["state_write_interval_steps"]),
        updates_per_block=int(sb["updates_per_block"]),
        keyframes_per_episode=int(re["keyframes_per_episode"]),
        episodes_per_segment=int(re["episodes_per_segment"]),
        keyframe_window_policy=str(re["keyframe_window_policy"]),
        pair_order_policy=str(re["pair_order_policy"]),
        total_target_images=int(ts["total_target_images"]),
        include_source=bool(ts["include_source"]),
        extra_target_policy=str(ts["extra_target_policy"]),
        prefer_nearby_keyframes=bool(ts["prefer_nearby_keyframes"]),
        fallback_expand_to_segment=bool(ts["fallback_expand_to_segment"]),
        fallback_with_replacement=bool(ts["fallback_with_replacement"]),
        overlap_mode=om,
        emit_preload_hints=bool(pl["emit_hints"]),
        execute_preload_hints=bool(pl["execute_hints"]),
        include_test=include_test,
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
        overlap_point_sample_size=(
            int(overlap_point_sample_size) if overlap_point_sample_size is not None else None
        ),
        overlap_candidate_frame_policy=(
            str(overlap_candidate_frame_policy) if overlap_candidate_frame_policy is not None else None
        ),
        overlap_score_type=str(overlap_score_type) if overlap_score_type is not None else None,
        overlap_min=float(overlap_min) if overlap_min is not None else None,
    )


def validate_train_scene_for_fixed(cfg: Any, fixed_scene_id: Optional[int]) -> None:
    train_ids = list(cfg.data.train_scene_ids)
    if fixed_scene_id is not None and int(fixed_scene_id) not in train_ids:
        raise ValueError(f"Fixed scene_id={fixed_scene_id} must appear in data.train_scene_ids={train_ids}")
