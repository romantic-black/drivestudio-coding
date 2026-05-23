from __future__ import annotations

from typing import Any, Optional, Tuple

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4
from datasets.train_scheduler_v9 import TrainSchedulerV9
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
)
from tools.train_minimal_streetforward_stage4_3_v7_common import (
    parse_include_test,
    validate_train_scene_for_fixed,
)


def _null_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    return int(x)


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


def resolve_fixed_scene_segment_v9(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    sv9 = cfg.get("scheduler_v9") if hasattr(cfg, "get") else None
    tr = (_cfg_get(sv9, "traversal", {}) or {}) if sv9 is not None else {}
    return _null_int(_cfg_get(tr, "fixed_scene_id", None)), _null_int(_cfg_get(tr, "fixed_segment_id", None))


def build_train_scheduler_v9_from_cfg(cfg: Any, dataset: MultiSceneDatasetV4) -> TrainSchedulerV9:
    sv9 = cfg.get("scheduler_v9") if hasattr(cfg, "get") else None
    if sv9 is None:
        raise ValueError("config must define scheduler_v9")
    if _cfg_get(sv9, "enable", False) is not True:
        raise ValueError("scheduler_v9.enable must be true")

    ep = _cfg_get(sv9, "episode", None)
    trav = _cfg_get(sv9, "traversal", None)
    preload = _cfg_get(sv9, "preload", None)
    execution = _cfg_get(sv9, "execution", {}) or {}
    if ep is None or trav is None or preload is None:
        raise ValueError("scheduler_v9 must define episode/traversal/preload")

    block = _cfg_get(sv9, "block", {}) or {}
    phase = str(_cfg_get(sv9, "phase", "phase_A_block_local_unroll"))
    block_order = str(_cfg_get(execution, "block_order", "block_major"))
    if block_order not in ("block_major", "step_major"):
        raise ValueError("scheduler_v9.execution.block_order must be one of ['block_major', 'step_major']")
    step_major_switch_interval_steps = int(_cfg_get(execution, "step_major_switch_interval_steps", 1))
    if step_major_switch_interval_steps < 1:
        raise ValueError("scheduler_v9.execution.step_major_switch_interval_steps must be >= 1")

    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_v9(cfg)
    validate_train_scene_for_fixed(cfg, fixed_scene_id)
    include_test = parse_include_test(cfg)

    episode_source_mode = str(_cfg_get(ep, "source_mode", _cfg_get(ep, "episode_source_mode", "keyframes")))
    steps_per_block = int(_cfg_get(block, "steps_per_block", _cfg_get(execution, "steps_per_block", 1)))
    if steps_per_block < 1:
        raise ValueError("scheduler_v9.block.steps_per_block must be >= 1")

    return dataset.create_train_scheduler_v9(
        phase=phase,
        steps_per_block=steps_per_block,
        blocks_per_episode=int(_cfg_get(ep, "blocks_per_episode")),
        include_source_frame=bool(_cfg_get(ep, "include_source_frame", True)),
        frame_within_keyframe_policy=str(_cfg_get(ep, "frame_within_keyframe_policy", "random_once_per_episode")),
        min_keyframes_required_policy=str(
            _cfg_get(ep, "min_keyframes_required_policy", "skip_if_less_than_window")
        ),
        traversal_mode=str(_cfg_get(trav, "mode", "round_robin_episode_interleave")),
        switch_after_episode=bool(_cfg_get(trav, "switch_after_episode", True)),
        segment_order=str(_cfg_get(trav, "segment_order", "ascending")),
        scene_order=str(_cfg_get(trav, "scene_order", "shuffle_per_epoch")),
        include_test=include_test,
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
        emit_preload_hints=bool(_cfg_get(preload, "emit_hints", True)),
        warm_next_block_exact=bool(_cfg_get(preload, "warm_next_block_exact", True)),
        warm_next_episode_chain=bool(_cfg_get(preload, "warm_next_episode_chain", True)),
        block_order=block_order,
        step_major_switch_interval_steps=step_major_switch_interval_steps,
        target_policy=str(_cfg_get(ep, "target_policy", "visited_episode_frames")),
        reset_policy=str(_cfg_get(execution, "reset_policy", "episode_end")),
        block_source_frame_policy=str(
            _cfg_get(ep, "block_source_frame_policy", "random_within_keyframe_per_visit")
        ),
        episode_source_mode=episode_source_mode,
        phase_a_cfg=_cfg_get(sv9, "phase_A", {}) or {},
        phase_b_cfg=_cfg_get(sv9, "phase_B", {}) or {},
        leakage_check_cfg=_cfg_get(sv9, "leakage_check", {}) or {},
        fail_fast=bool(_cfg_get(sv9, "fail_fast", True)),
    )


__all__ = [
    "build_multi_scene_dataset_v4",
    "build_train_scheduler_v9_from_cfg",
    "resolve_fixed_scene_segment_v9",
]
