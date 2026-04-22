from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4
from datasets.train_scheduler_v7 import TrainSchedulerV7
from tools.train_minimal_streetforward_stage4_3_v4_common import (
    parse_include_test,
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

    bg_ks: List[int] = []
    dynamic_ks: List[int] = []
    required_branches: List[str] = []
    if branches is not None:
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

    model_cfg = _cfg_get(cfg, "model")
    stage = str(_cfg_get(model_cfg, "stage", "") or "").strip()
    struct_cfg = _cfg_get(model_cfg, "struct_decoder")
    struct_type = str(_cfg_get(struct_cfg, "type", "") or "").strip()
    knn_attn_cfg = _cfg_get(struct_cfg, "knn_attention")
    if knn_attn_cfg is None:
        knn_attn_cfg = _cfg_get(model_cfg, "knn_attention")
    stage5_1_like = stage == "5_1" or struct_type == "xcpe_knn_attn"
    knn_attn_enable = bool(_cfg_get(knn_attn_cfg, "enable", False)) if knn_attn_cfg is not None else False
    fixed_neighbor_enabled = bool(stage5_1_like or knn_attn_enable)
    neighbor_k_store = 0
    if fixed_neighbor_enabled:
        if knn_attn_cfg is None:
            raise ValueError(
                "Stage5_1 fixed cached KNN requires model.struct_decoder.knn_attention (or model.knn_attention)."
            )
        k_attn = int(_cfg_get(knn_attn_cfg, "k", 0))
        if k_attn <= 1:
            raise ValueError(f"knn_attention.k must be > 1 when fixed cached KNN is enabled, got {k_attn}")
        neighbor_k_store = int(k_attn - 1)

    bg_ks = sorted(set(int(x) for x in bg_ks))
    dynamic_ks = sorted(set(int(x) for x in dynamic_ks))
    return {
        "enabled": bool(bg_ks or dynamic_ks or fixed_neighbor_enabled),
        "background_ks": bg_ks,
        "dynamic_ks": dynamic_ks,
        "required_branches": sorted(set(required_branches)),
        "fixed_neighbor_enabled": bool(fixed_neighbor_enabled),
        "neighbor_k_store": int(neighbor_k_store),
    }


def build_multi_scene_dataset_v4(cfg: Any, device: torch.device) -> MultiSceneDatasetV4:
    knn_requirements = _extract_knn_requirements_from_cfg(cfg)
    return MultiSceneDatasetV4(
        dataset_cfg=cfg.dataset,
        data_cfg=cfg.data,
        device=device,
        knn_requirements=knn_requirements,
    )


def _null_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    return int(x)


def resolve_fixed_scene_segment_v7(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    tr = (cfg.get("scheduler_v7") or {}).get("traversal") or {}
    return _null_int(tr.get("fixed_scene_id")), _null_int(tr.get("fixed_segment_id"))


def build_train_scheduler_v7_from_cfg(cfg: Any, dataset: MultiSceneDatasetV4) -> TrainSchedulerV7:
    sv7 = cfg.get("scheduler_v7")
    if sv7 is None:
        raise ValueError("config must define scheduler_v7")
    if sv7.get("enable") is not True:
        raise ValueError("scheduler_v7.enable must be true")
    block = sv7.get("block")
    ep = sv7.get("episode")
    trav = sv7.get("traversal")
    preload = sv7.get("preload")
    if block is None or ep is None or trav is None or preload is None:
        raise ValueError("scheduler_v7 must define block/episode/traversal/preload")

    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_v7(cfg)
    validate_train_scene_for_fixed(cfg, fixed_scene_id)
    include_test = parse_include_test(cfg)

    return dataset.create_train_scheduler_v7(
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
    )
