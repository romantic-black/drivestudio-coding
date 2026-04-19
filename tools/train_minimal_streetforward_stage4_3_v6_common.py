from __future__ import annotations

from typing import Any, Dict, List

import torch

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4
from datasets.train_scheduler_v6 import TrainSchedulerV6
from tools.train_minimal_streetforward_stage4_3_v4_common import (
    parse_include_test,
    resolve_fixed_scene_segment,
    validate_train_scene_for_fixed,
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _extract_knn_requirements_from_cfg(cfg: Any) -> Dict[str, Any]:
    branches = _cfg_get(_cfg_get(cfg, "model"), "branches")
    if branches is None:
        return {
            "enabled": False,
            "background_ks": [],
            "dynamic_ks": [],
            "required_branches": [],
        }

    bg_ks: List[int] = []
    dynamic_ks: List[int] = []
    required_branches: List[str] = []
    for branch_name in ("bg", "distant", "rigid"):
        branch_cfg = _cfg_get(branches, branch_name)
        if branch_cfg is None:
            continue
        scale_init = _cfg_get(_cfg_get(branch_cfg, "init"), "scale_init")
        if scale_init is None:
            continue
        mode = str(_cfg_get(scale_init, "mode", "isotropic")).strip()
        if mode != "knn":
            continue
        k = int(_cfg_get(scale_init, "knn_k", 0))
        if k <= 0:
            raise ValueError(
                f"model.branches.{branch_name}.init.scale_init.knn_k must be > 0 when mode=knn, got {k}"
            )
        required_branches.append(branch_name)
        if branch_name in {"bg", "distant"}:
            bg_ks.append(int(k))
        elif branch_name == "rigid":
            dynamic_ks.append(int(k))

    bg_ks = sorted(set(int(x) for x in bg_ks))
    dynamic_ks = sorted(set(int(x) for x in dynamic_ks))
    return {
        "enabled": bool(bg_ks or dynamic_ks),
        "background_ks": bg_ks,
        "dynamic_ks": dynamic_ks,
        "required_branches": sorted(set(required_branches)),
    }


def build_multi_scene_dataset_v4(cfg: Any, device: torch.device) -> MultiSceneDatasetV4:
    knn_requirements = _extract_knn_requirements_from_cfg(cfg)
    return MultiSceneDatasetV4(
        dataset_cfg=cfg.dataset,
        data_cfg=cfg.data,
        device=device,
        knn_requirements=knn_requirements,
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
