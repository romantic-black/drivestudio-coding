from __future__ import annotations

from typing import Any

import torch

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4
from datasets.train_scheduler_v6 import TrainSchedulerV6
from tools.train_minimal_streetforward_stage4_3_v4_common import (
    parse_include_test,
    resolve_fixed_scene_segment,
    validate_train_scene_for_fixed,
)


def build_multi_scene_dataset_v4(cfg: Any, device: torch.device) -> MultiSceneDatasetV4:
    return MultiSceneDatasetV4(
        dataset_cfg=cfg.dataset,
        data_cfg=cfg.data,
        device=device,
    )


def build_train_scheduler_v6_from_cfg(cfg: Any, dataset: MultiSceneDatasetV4) -> TrainSchedulerV6:
    sv6 = cfg.get("scheduler_v6")
    if sv6 is None:
        raise ValueError("config must define scheduler_v6")
    if sv6.get("enable") is not True:
        raise ValueError("scheduler_v6.enable must be true")
    tb = sv6.get("time_base")
    sb = sv6.get("source_block")
    re = sv6.get("reset_episode")
    ts = sv6.get("target_sampling")
    if tb is None or sb is None or re is None or ts is None:
        raise ValueError("scheduler_v6 must define time_base/source_block/reset_episode/target_sampling")

    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment(cfg)
    validate_train_scene_for_fixed(cfg, fixed_scene_id)
    include_test = parse_include_test(cfg)

    return dataset.create_train_scheduler_v6(
        state_write_interval_steps=int(tb["state_write_interval_steps"]),
        updates_per_block=int(sb["updates_per_block"]),
        keyframes_per_episode=int(re["keyframes_per_episode"]),
        episodes_per_segment=int(re["episodes_per_segment"]),
        total_target_frames=int(ts["total_target_frames"]),
        include_source_frame=bool(ts["include_source_frame"]),
        neighbor_ring=int(ts["neighbor_ring"]),
        prefer_nearby_keyframes=bool(ts["prefer_nearby_keyframes"]),
        fallback_expand_to_segment=bool(ts["fallback_expand_to_segment"]),
        with_replacement=bool(ts["with_replacement"]),
        include_test=include_test,
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
    )
