from __future__ import annotations

import copy
from typing import Any, Optional, Tuple

from datasets.train_scheduler_iforward import IFORWARD_STAGE2_1_SCHEDULER_VERSION, TrainSchedulerIForward
from datasets.train_scheduler_iforward_sequence10 import (
    IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
    TrainSchedulerIForwardSequence10,
)
from datasets.iforward_stage2_2.scheduler import IFORWARD_STAGE2_2_SCHEDULER_VERSION, Stage22Scheduler
from datasets.iforward_stage2_3.scheduler import (
    IFORWARD_STAGE2_3_SCHEDULER_VERSION,
    IFORWARD_STAGE3_0_SCHEDULER_VERSION,
    IFORWARD_STAGE3_2_SCHEDULER_VERSION,
    Stage23Scheduler,
)


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


def _active_stage23_scheduler_cfg(cfg: Any) -> Tuple[Any, str]:
    sched32 = cfg.get("scheduler_stage3_2") if hasattr(cfg, "get") else None
    if sched32 is not None and bool(_cfg_get(sched32, "enable", False)):
        return sched32, "scheduler_stage3_2"
    sched30 = cfg.get("scheduler_stage3_0") if hasattr(cfg, "get") else None
    if sched30 is not None and bool(_cfg_get(sched30, "enable", False)):
        return sched30, "scheduler_stage3_0"
    return (cfg.get("scheduler_v3") if hasattr(cfg, "get") else None), "scheduler_v3"


def _to_plain(node: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(node):
            return OmegaConf.to_container(node, resolve=False)
    except Exception:
        pass
    if isinstance(node, dict):
        return {k: _to_plain(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_to_plain(x) for x in list(node)]
    return copy.deepcopy(node)


def _deep_merge(base: Any, override: Any) -> dict[str, Any]:
    out = dict(_to_plain(base) or {})
    over = dict(_to_plain(override) or {})
    for key, value in over.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _merged_stage23_scheduler_cfg(cfg: Any, sched: Any, sched_key: str) -> Any:
    if str(sched_key) != "scheduler_stage3_2":
        return sched
    inherit_from = str(_cfg_get(sched, "inherit_from", "scheduler_stage3_0") or "")
    base: Any = {}
    if inherit_from == "scheduler_stage3_0":
        base = cfg.get("scheduler_stage3_0") if hasattr(cfg, "get") else {}
    elif inherit_from == "scheduler_v3":
        base = cfg.get("scheduler_v3") if hasattr(cfg, "get") else {}
    elif inherit_from:
        raise ValueError(f"unsupported scheduler_stage3_2.inherit_from={inherit_from!r}")
    merged = _deep_merge(base, sched)
    merged["enable"] = True
    return merged


def _is_stage23_optimizer_sequence_cfg(sched: Any) -> bool:
    return (
        sched is not None
        and bool(_cfg_get(sched, "enable", False))
        and str(_cfg_get(sched, "version", "")) in {
            "optimizer_sequence_v1",
            IFORWARD_STAGE3_0_SCHEDULER_VERSION,
            IFORWARD_STAGE3_2_SCHEDULER_VERSION,
            "distributional_episode_v1",
        }
    )


def resolve_fixed_scene_segment_iforward(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    sched22 = cfg.get("scheduler_stage2_2") if hasattr(cfg, "get") else None
    sched23, sched23_key = _active_stage23_scheduler_cfg(cfg)
    if _is_stage23_optimizer_sequence_cfg(sched23):
        sched23 = _merged_stage23_scheduler_cfg(cfg, sched23, sched23_key)
        traversal23 = _cfg_get(sched23, "traversal", {}) or {}
        return _null_int(_cfg_get(traversal23, "fixed_scene_id", None)), _null_int(
            _cfg_get(traversal23, "fixed_segment_id", None)
        )
    if sched22 is not None and bool(_cfg_get(sched22, "enable", False)):
        traversal22 = _cfg_get(sched22, "traversal", {}) or {}
        return _null_int(_cfg_get(traversal22, "fixed_scene_id", None)), _null_int(
            _cfg_get(traversal22, "fixed_segment_id", None)
        )
    sched = cfg.get("scheduler_iforward") if hasattr(cfg, "get") else None
    traversal = (_cfg_get(sched, "traversal", {}) or {}) if sched is not None else {}
    return _null_int(_cfg_get(traversal, "fixed_scene_id", None)), _null_int(
        _cfg_get(traversal, "fixed_segment_id", None)
    )


def build_train_scheduler_iforward_from_cfg(
    cfg: Any,
    dataset: Any,
) -> TrainSchedulerIForward | TrainSchedulerIForwardSequence10 | Stage22Scheduler | Stage23Scheduler:
    sched23, sched23_key = _active_stage23_scheduler_cfg(cfg)
    if _is_stage23_optimizer_sequence_cfg(sched23):
        legacy = cfg.get("scheduler_iforward") if hasattr(cfg, "get") else None
        sched22_legacy = cfg.get("scheduler_stage2_2") if hasattr(cfg, "get") else None
        sched32_legacy = cfg.get("scheduler_stage3_2") if hasattr(cfg, "get") else None
        sched30_legacy = cfg.get("scheduler_stage3_0") if hasattr(cfg, "get") else None
        sched23_legacy = cfg.get("scheduler_v3") if hasattr(cfg, "get") else None
        if legacy is not None and bool(_cfg_get(legacy, "enable", False)):
            raise ValueError(f"{sched23_key} optimizer-sequence forbids enabled legacy scheduler_iforward")
        if sched22_legacy is not None and bool(_cfg_get(sched22_legacy, "enable", False)):
            raise ValueError(f"{sched23_key} optimizer-sequence forbids enabled scheduler_stage2_2")
        if sched23_key == "scheduler_stage3_0" and sched23_legacy is not None and bool(_cfg_get(sched23_legacy, "enable", False)):
            raise ValueError("scheduler_stage3_0 forbids enabled legacy scheduler_v3")
        if sched23_key == "scheduler_stage3_2" and sched23_legacy is not None and bool(_cfg_get(sched23_legacy, "enable", False)):
            raise ValueError("scheduler_stage3_2 forbids enabled legacy scheduler_v3")
        inherit_from = str(_cfg_get(sched32_legacy, "inherit_from", "scheduler_stage3_0") or "")
        if sched23_key == "scheduler_stage3_2" and sched30_legacy is not None and bool(inherit_from != "scheduler_stage3_0"):
            raise ValueError("scheduler_stage3_2 may coexist with scheduler_stage3_0 only via inherit_from=scheduler_stage3_0")
        sched23_runtime = _merged_stage23_scheduler_cfg(cfg, sched23, sched23_key)
        index_dir = str(_cfg_get(sched23, "index_dir", "") or "")
        if not index_dir:
            index_dir = str(_cfg_get(sched23_runtime, "index_dir", "") or "")
        if not index_dir:
            raise ValueError(f"{sched23_key}.index_dir is required; build it with tools/build_iforward_stage2_3_index.py")
        fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_iforward(cfg)
        from tools.train_minimal_streetforward_stage4_3_v7_common import (
            parse_include_test,
            validate_train_scene_for_fixed,
        )

        validate_train_scene_for_fixed(cfg, fixed_scene_id)
        include_test = parse_include_test(cfg)
        return Stage23Scheduler(
            dataset=dataset,
            cfg=cfg,
            index_dir=index_dir,
            traversal_cfg=dict(_cfg_get(sched23_runtime, "traversal", {}) or {}),
            bootstrap_cfg=dict(_cfg_get(sched23_runtime, "bootstrap", {}) or {}),
            sequence_cfg=dict(_cfg_get(sched23_runtime, "sequence", {}) or {}),
            assimilation_cfg=dict(_cfg_get(sched23_runtime, "assimilation", {}) or {}),
            repair_cfg=dict(_cfg_get(sched23_runtime, "repair", {}) or {}),
            loss_cfg=dict(_cfg_get(sched23_runtime, "loss", {}) or {}),
            producer_cfg=dict(_cfg_get(sched23_runtime, "producer", {}) or {}),
            include_test=bool(include_test),
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            seed=_cfg_get(_cfg_get(sched23_runtime, "traversal", {}) or {}, "seed", None),
            fail_fast=bool(_cfg_get(sched23_runtime, "fail_fast", True)),
        )
    sched22 = cfg.get("scheduler_stage2_2") if hasattr(cfg, "get") else None
    if sched22 is not None and bool(_cfg_get(sched22, "enable", False)):
        legacy = cfg.get("scheduler_iforward") if hasattr(cfg, "get") else None
        if legacy is not None and bool(_cfg_get(legacy, "enable", False)):
            raise ValueError("scheduler_stage2_2 forbids enabled legacy scheduler_iforward")
        version22 = str(_cfg_get(sched22, "version", IFORWARD_STAGE2_2_SCHEDULER_VERSION))
        if version22 != IFORWARD_STAGE2_2_SCHEDULER_VERSION:
            raise ValueError(
                f"scheduler_stage2_2.version must be {IFORWARD_STAGE2_2_SCHEDULER_VERSION}, got {version22!r}"
            )
        index_dir = str(_cfg_get(sched22, "index_dir", "") or "")
        if not index_dir:
            raise ValueError("scheduler_stage2_2.index_dir is required; build it with tools/build_iforward_stage2_2_index.py")
        fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_iforward(cfg)
        from tools.train_minimal_streetforward_stage4_3_v7_common import (
            parse_include_test,
            validate_train_scene_for_fixed,
        )

        validate_train_scene_for_fixed(cfg, fixed_scene_id)
        include_test = parse_include_test(cfg)
        return Stage22Scheduler(
            dataset=dataset,
            cfg=cfg,
            index_dir=index_dir,
            traversal_cfg=dict(_cfg_get(sched22, "traversal", {}) or {}),
            bootstrap_cfg=dict(_cfg_get(sched22, "bootstrap", {}) or {}),
            protocol_cfg=dict(_cfg_get(sched22, "protocol", {}) or {}),
            causal_cfg=dict(_cfg_get(sched22, "causal", {}) or {}),
            repair_cfg=dict(_cfg_get(sched22, "repair", {}) or {}),
            supervision_cfg=dict(_cfg_get(sched22, "supervision", {}) or {}),
            preload_cfg=dict(_cfg_get(sched22, "preload", {}) or {}),
            include_test=bool(include_test),
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            seed=_cfg_get(_cfg_get(sched22, "traversal", {}) or {}, "seed", None),
            fail_fast=bool(_cfg_get(sched22, "fail_fast", True)),
        )
    sched = cfg.get("scheduler_iforward") if hasattr(cfg, "get") else None
    if sched is None:
        raise ValueError("config must define scheduler_iforward")
    if _cfg_get(sched, "enable", False) is not True:
        raise ValueError("scheduler_iforward.enable must be true")
    version = str(_cfg_get(sched, "version", "iforward_v1"))
    if version not in {
        "iforward_v1",
        "iforward_v3_random_window",
        "iforward_v4_coverage_ordered",
        IFORWARD_STAGE2_1_SCHEDULER_VERSION,
        IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
    }:
        raise ValueError(
            "scheduler_iforward.version must be iforward_v1, "
            "iforward_v3_random_window, iforward_v4_coverage_ordered, "
            f"{IFORWARD_STAGE2_1_SCHEDULER_VERSION}, or {IFORWARD_SEQUENCE10_SCHEDULER_VERSION}"
        )

    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_iforward(cfg)
    from tools.train_minimal_streetforward_stage4_3_v7_common import (
        parse_include_test,
        validate_train_scene_for_fixed,
    )

    validate_train_scene_for_fixed(cfg, fixed_scene_id)
    include_test = parse_include_test(cfg)

    episode_cfg = dict(_cfg_get(sched, "episode", {}) or {})
    rollout_cfg = dict(_cfg_get(sched, "rollout", {}) or {})
    traversal_cfg = dict(_cfg_get(sched, "traversal", {}) or {})
    evidence_cfg = dict(_cfg_get(sched, "evidence", {}) or {})
    supervision_cfg = dict(_cfg_get(sched, "supervision", {}) or {})
    memory_cfg = dict(_cfg_get(sched, "memory", {}) or {})
    loss_timing_cfg = dict(_cfg_get(sched, "loss_timing", {}) or {})
    leakage_check_cfg = dict(_cfg_get(sched, "leakage_check", {}) or {})
    preload_cfg = dict(_cfg_get(sched, "preload", {}) or {})

    if version == IFORWARD_SEQUENCE10_SCHEDULER_VERSION:
        if dict(_cfg_get(sched, "episode", {}) or {}) or dict(_cfg_get(sched, "rollout", {}) or {}):
            raise ValueError("iforward_sequence10_v1 forbids legacy scheduler_iforward.episode/rollout config blocks")
        return TrainSchedulerIForwardSequence10(
            dataset=dataset,
            traversal_cfg=traversal_cfg,
            bootstrap_cfg=dict(_cfg_get(sched, "bootstrap", {}) or {}),
            sequence_cfg=dict(_cfg_get(sched, "sequence", {}) or {}),
            causal_cfg=dict(_cfg_get(sched, "causal", {}) or {}),
            repair_cfg=dict(_cfg_get(sched, "repair", {}) or {}),
            supervision_cfg=supervision_cfg,
            history_loss_cfg=dict(_cfg_get(sched, "history_loss", {}) or {}),
            damage_loss_cfg=dict(_cfg_get(sched, "damage_loss", {}) or {}),
            preload_cfg=preload_cfg,
            include_test=bool(include_test),
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            seed=_cfg_get(traversal_cfg, "seed", None),
            fail_fast=bool(_cfg_get(sched, "fail_fast", True)),
        )

    return dataset.create_train_scheduler_iforward(
        episode_cfg=episode_cfg,
        rollout_cfg=rollout_cfg,
        traversal_cfg=traversal_cfg,
        evidence_cfg=evidence_cfg,
        supervision_cfg=supervision_cfg,
        memory_cfg=memory_cfg,
        loss_timing_cfg=loss_timing_cfg,
        leakage_check_cfg=leakage_check_cfg,
        preload_cfg=preload_cfg,
        include_test=bool(include_test),
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
        seed=_cfg_get(traversal_cfg, "seed", None),
        version=version,
        fail_fast=bool(_cfg_get(sched, "fail_fast", True)),
    )


def build_multi_scene_dataset_v4(cfg: Any, device: Any) -> Any:
    from tools.train_minimal_streetforward_stage4_3_v8_common import (
        build_multi_scene_dataset_v4 as _build_multi_scene_dataset_v4,
    )

    return _build_multi_scene_dataset_v4(cfg, device)


__all__ = [
    "build_multi_scene_dataset_v4",
    "build_train_scheduler_iforward_from_cfg",
    "resolve_fixed_scene_segment_iforward",
]
