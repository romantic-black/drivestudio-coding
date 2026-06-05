from __future__ import annotations

from typing import Any, Optional, Tuple


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


def _null_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    return int(x)


def resolve_fixed_scene_segment_iforward_random_window(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    one_segment = cfg.get("one_segment") if hasattr(cfg, "get") else None
    if one_segment is not None:
        scene_id = _cfg_get(one_segment, "scene_id", None)
        segment_id = _cfg_get(one_segment, "segment_id", None)
        if scene_id is not None or segment_id is not None:
            if scene_id is None or segment_id is None:
                raise ValueError("one_segment requires both scene_id and segment_id")
            return _null_int(scene_id), _null_int(segment_id)

    sched = cfg.get("scheduler_iforward_random_window") if hasattr(cfg, "get") else None
    traversal = (_cfg_get(sched, "traversal", {}) or {}) if sched is not None else {}
    scene_id = _cfg_get(traversal, "fixed_scene_id", None)
    segment_id = _cfg_get(traversal, "fixed_segment_id", None)
    if scene_id is not None or segment_id is not None:
        if scene_id is None or segment_id is None:
            raise ValueError(
                "scheduler_iforward_random_window.traversal requires both "
                "fixed_scene_id and fixed_segment_id when either is set"
            )
    return _null_int(scene_id), _null_int(segment_id)


def build_train_scheduler_iforward_random_window_from_cfg(cfg: Any, dataset: Any) -> Any:
    sched = cfg.get("scheduler_iforward_random_window") if hasattr(cfg, "get") else None
    if sched is None:
        raise ValueError("config must define scheduler_iforward_random_window")
    if _cfg_get(sched, "enable", False) is not True:
        raise ValueError("scheduler_iforward_random_window.enable must be true")
    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_iforward_random_window(cfg)
    from tools.train_minimal_streetforward_stage4_3_v7_common import (
        parse_include_test,
        validate_train_scene_for_fixed,
    )

    validate_train_scene_for_fixed(cfg, fixed_scene_id)
    include_test = parse_include_test(cfg)
    traversal_cfg = dict(_cfg_get(sched, "traversal", {}) or {})
    segment_cfg = dict(_cfg_get(sched, "segment", {}) or {})
    episode_cfg = dict(_cfg_get(sched, "episode", {}) or {})
    rollout_cfg = dict(_cfg_get(sched, "rollout", {}) or {})
    evidence_cfg = dict(_cfg_get(sched, "evidence", {}) or {})
    supervision_cfg = dict(_cfg_get(sched, "supervision", {}) or {})
    memory_cfg = dict(_cfg_get(sched, "memory", {}) or {})
    loss_timing_cfg = dict(_cfg_get(sched, "loss_timing", {}) or {})
    preload_cfg = dict(_cfg_get(sched, "preload", {}) or {})
    return dataset.create_train_scheduler_iforward_random_window(
        traversal_cfg=traversal_cfg,
        segment_cfg=segment_cfg,
        episode_cfg=episode_cfg,
        rollout_cfg=rollout_cfg,
        evidence_cfg=evidence_cfg,
        supervision_cfg=supervision_cfg,
        memory_cfg=memory_cfg,
        loss_timing_cfg=loss_timing_cfg,
        preload_cfg=preload_cfg,
        include_test=bool(include_test),
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
        seed=_cfg_get(traversal_cfg, "seed", None),
        fail_fast=bool(_cfg_get(sched, "fail_fast", True)),
    )


def build_multi_scene_dataset_v4(cfg: Any, device: Any) -> Any:
    from tools.train_minimal_streetforward_stage4_3_v8_common import (
        build_multi_scene_dataset_v4 as _build_multi_scene_dataset_v4,
    )

    return _build_multi_scene_dataset_v4(cfg, device)


__all__ = [
    "build_multi_scene_dataset_v4",
    "build_train_scheduler_iforward_random_window_from_cfg",
    "resolve_fixed_scene_segment_iforward_random_window",
]
