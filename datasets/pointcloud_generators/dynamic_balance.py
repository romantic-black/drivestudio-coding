from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np


def normalize_dynamic_point_balance_cfg(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if raw is None:
        return {"enable": False}
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"dataset.pointcloud.dynamic_point_balance must be a mapping, got {type(raw)}"
        )
    enable = bool(raw.get("enable", False))
    if not enable:
        return {"enable": False}

    mode = str(raw.get("mode", "bbox_volume")).strip()
    if mode != "bbox_volume":
        raise ValueError(
            "dataset.pointcloud.dynamic_point_balance.mode must be 'bbox_volume'"
        )

    volume_exponent = float(raw.get("volume_exponent", raw.get("exponent", 1.0)))
    if volume_exponent < 0.0:
        raise ValueError(
            "dataset.pointcloud.dynamic_point_balance.volume_exponent must be >= 0"
        )
    min_scale = float(raw.get("min_scale", 0.25))
    max_scale = float(raw.get("max_scale", 4.0))
    if min_scale <= 0.0:
        raise ValueError("dataset.pointcloud.dynamic_point_balance.min_scale must be > 0")
    if max_scale < min_scale:
        raise ValueError(
            "dataset.pointcloud.dynamic_point_balance.max_scale must be >= min_scale"
        )
    min_points = int(raw.get("min_points_per_instance", 1))
    if min_points < 0:
        raise ValueError(
            "dataset.pointcloud.dynamic_point_balance.min_points_per_instance must be >= 0"
        )
    max_points_raw = raw.get("max_points_per_instance")
    max_points = None if max_points_raw is None else int(max_points_raw)
    if max_points is not None and max_points <= 0:
        raise ValueError(
            "dataset.pointcloud.dynamic_point_balance.max_points_per_instance must be > 0"
        )
    if max_points is not None and max_points < min_points:
        raise ValueError(
            "dataset.pointcloud.dynamic_point_balance.max_points_per_instance must be >= "
            "min_points_per_instance"
        )

    ref_raw = raw.get("reference_volume_m3")
    reference_volume = None if ref_raw is None else float(ref_raw)
    if reference_volume is not None and reference_volume <= 0.0:
        raise ValueError(
            "dataset.pointcloud.dynamic_point_balance.reference_volume_m3 must be > 0"
        )

    return {
        "enable": True,
        "mode": mode,
        "volume_exponent": volume_exponent,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "min_points_per_instance": min_points,
        "max_points_per_instance": max_points,
        "reference_volume_m3": reference_volume,
    }


def dynamic_point_balance_enabled(cfg: Optional[Mapping[str, Any]]) -> bool:
    return bool(cfg is not None and cfg.get("enable", False))


def collect_instance_volumes_from_frames(
    instances_by_frame: Sequence[Sequence[Mapping[str, Any]]],
    *,
    skip_instance_intids: Optional[Set[int]] = None,
) -> Dict[int, float]:
    skip = {int(x) for x in skip_instance_intids} if skip_instance_intids else set()
    values: Dict[int, List[float]] = {}
    for frame_instances in instances_by_frame:
        for instance in frame_instances:
            intid = int(instance["intid"])
            if intid in skip:
                continue
            size_lwh = np.asarray(instance["size_lwh"], dtype=np.float32).reshape(-1)
            if int(size_lwh.shape[0]) != 3:
                continue
            if not np.isfinite(size_lwh).all():
                continue
            volume = float(np.prod(np.maximum(size_lwh, 0.0)))
            if not np.isfinite(volume) or volume <= 0.0:
                continue
            values.setdefault(intid, []).append(volume)
    return {
        int(intid): float(np.median(np.asarray(volumes, dtype=np.float32)))
        for intid, volumes in values.items()
        if len(volumes) > 0
    }


def merge_instance_volume_maps(*maps: Mapping[int, float]) -> Dict[int, float]:
    values: Dict[int, List[float]] = {}
    for volume_map in maps:
        for intid_raw, volume_raw in dict(volume_map or {}).items():
            intid = int(intid_raw)
            volume = float(volume_raw)
            if not np.isfinite(volume) or volume <= 0.0:
                continue
            values.setdefault(intid, []).append(volume)
    return {
        int(intid): float(np.median(np.asarray(volumes, dtype=np.float32)))
        for intid, volumes in values.items()
        if len(volumes) > 0
    }


def volume_map_to_jsonable(volume_by_intid: Mapping[int, float]) -> Dict[str, float]:
    return {
        str(int(intid)): float(volume)
        for intid, volume in sorted(volume_by_intid.items(), key=lambda kv: int(kv[0]))
    }


def volume_map_from_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[int, float]:
    if not isinstance(metadata, Mapping):
        return {}
    raw = metadata.get("dynamic_instance_volumes_m3")
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[int, float] = {}
    for intid_raw, volume_raw in raw.items():
        try:
            intid = int(intid_raw)
            volume = float(volume_raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(volume) and volume > 0.0:
            out[intid] = volume
    return out


def compute_volume_balanced_point_caps(
    base_max_points: Optional[int],
    volume_by_intid: Mapping[int, float],
    cfg: Optional[Mapping[str, Any]],
) -> Dict[int, int]:
    cfg_norm = normalize_dynamic_point_balance_cfg(cfg)
    if not dynamic_point_balance_enabled(cfg_norm):
        return {}
    if base_max_points is None:
        return {}
    base = int(base_max_points)
    if base <= 0:
        return {}

    valid_items = [
        (int(intid), float(volume))
        for intid, volume in dict(volume_by_intid or {}).items()
        if np.isfinite(float(volume)) and float(volume) > 0.0
    ]
    if not valid_items:
        return {}

    reference_volume = cfg_norm.get("reference_volume_m3")
    if reference_volume is None:
        reference_volume = float(np.median([v for _, v in valid_items]))
    reference_volume = max(float(reference_volume), 1e-6)

    exponent = float(cfg_norm["volume_exponent"])
    min_scale = float(cfg_norm["min_scale"])
    max_scale = float(cfg_norm["max_scale"])
    min_points = int(cfg_norm["min_points_per_instance"])
    max_points = cfg_norm.get("max_points_per_instance")

    caps: Dict[int, int] = {}
    for intid, volume in valid_items:
        scale = (float(volume) / reference_volume) ** exponent
        scale = float(np.clip(scale, min_scale, max_scale))
        cap = int(round(float(base) * scale))
        cap = max(int(min_points), cap)
        if max_points is not None:
            cap = min(int(max_points), cap)
        caps[int(intid)] = int(max(0, cap))
    return caps


def stride_limit_points(points: np.ndarray, max_count: Optional[int]) -> np.ndarray:
    if max_count is None:
        return points
    n = int(points.shape[0])
    cap = int(max_count)
    if cap <= 0:
        return points[:0]
    if n <= cap:
        return points
    step = max(1, n // cap)
    idx = np.arange(0, n, step, dtype=np.int64)
    if int(idx.shape[0]) > cap:
        idx = idx[:cap]
    return points[idx]


def cap_dynamic_points_by_intid(
    dynamic_points: Mapping[int, np.ndarray],
    *,
    cap_by_intid: Optional[Mapping[int, int]] = None,
    default_cap: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    caps = {int(k): int(v) for k, v in dict(cap_by_intid or {}).items()}
    for intid_raw, points_raw in dynamic_points.items():
        intid = int(intid_raw)
        points = np.asarray(points_raw, dtype=np.float32)
        cap = caps.get(intid, default_cap)
        out[intid] = np.ascontiguousarray(stride_limit_points(points, cap), dtype=np.float32)
    return out
