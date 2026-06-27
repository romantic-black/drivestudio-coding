from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from datasets.iforward_stage2_2.index_builder import build_stage2_2_index_from_dataset

from .index_loader import Stage23Index


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


def _stage22_compatible_cfg(cfg: Any) -> dict:
    sched_v3 = _cfg_get(cfg, "scheduler_v3", {}) or {}
    time_v3 = _cfg_get(sched_v3, "time", {}) or {}
    sched22 = {
        "time": {
            "frame_period_us": int(_cfg_get(time_v3, "frame_period_us", 100000)),
            "allow_synthetic_timestamp": bool(_cfg_get(time_v3, "allow_synthetic_timestamp", False)),
        }
    }
    return {"data": _cfg_get(cfg, "data", {}) or {}, "scheduler_stage2_2": sched22}


def build_stage2_3_index_from_dataset(
    *,
    dataset: Any,
    cfg: Optional[Any] = None,
    output_dir: Optional[str | Path] = None,
    frame_period_us: Optional[int] = None,
    fixed_scene_id: Optional[int] = None,
    fixed_segment_id: Optional[int] = None,
) -> Stage23Index:
    base = build_stage2_2_index_from_dataset(
        dataset=dataset,
        cfg=_stage22_compatible_cfg(cfg or {}),
        output_dir=output_dir,
        frame_period_us=frame_period_us,
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
    )
    return Stage23Index.from_stage22(base)


__all__ = ["build_stage2_3_index_from_dataset"]
