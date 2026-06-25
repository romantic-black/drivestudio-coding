from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .index_format import (
    BOOTSTRAP_DTYPE,
    FRAME_DTYPE,
    IFORWARD_STAGE2_2_SCHEDULER_VERSION,
    PROTOCOL_IDS,
    PROTOCOL_OFFSETS,
    PROTOCOL_PATTERNS,
    SEGMENT_DTYPE,
    STAGE22_INDEX_VERSION,
    WINDOW_DTYPE,
    canonical_json,
    fingerprint_payload,
    stable_uint64,
)
from .index_loader import Stage22Index


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


def _pose_root_from_cfg(cfg: Any) -> str:
    data = _cfg_get(cfg, "data", {}) or {}
    return str(_cfg_get(data, "data_root", "") or "")


def _read_pose(data_root: str, scene_id: int, frame_idx: int) -> Tuple[Tuple[float, float, float], float]:
    if not data_root:
        return (0.0, 0.0, 0.0), 0.0
    candidates = [
        Path(data_root) / f"{int(scene_id):03d}" / "lidar_pose" / f"{int(frame_idx):03d}.txt",
        Path(data_root) / str(int(scene_id)) / "lidar_pose" / f"{int(frame_idx)}.txt",
        Path(data_root) / f"{int(scene_id):03d}" / "lidar_pose" / f"{int(frame_idx)}.txt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return (0.0, 0.0, 0.0), 0.0
    try:
        mat = np.loadtxt(str(path), dtype=np.float64).reshape(4, 4)
    except Exception:
        return (0.0, 0.0, 0.0), 0.0
    trans = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
    yaw = float(math.atan2(float(mat[1, 0]), float(mat[0, 0])))
    return trans, yaw


def _sorted_train_frames(sidx: Any) -> List[int]:
    frames = [int(x) for x in list(getattr(sidx, "frame_indices", []) or [])]
    train_set = set(int(x) for x in set(getattr(sidx, "train_frame_set", set(frames)) or set(frames)))
    frames = sorted(f for f in frames if f in train_set)
    if not frames:
        refs = list(getattr(sidx, "train_image_refs", []) or [])
        frames = sorted({int(ref[0]) for ref in refs})
    return frames


def _frame_to_keyframe(sidx: Any, frame_idx: int) -> int:
    mapping = dict(getattr(sidx, "frame_to_keyframe", {}) or {})
    if int(frame_idx) in mapping:
        return int(mapping[int(frame_idx)])
    keyframes = [int(x) for x in list(getattr(sidx, "keyframe_indices", []) or [])]
    if not keyframes:
        return int(frame_idx)
    return min(keyframes, key=lambda k: abs(int(k) - int(frame_idx)))


def _camera_mask(sidx: Any, frame_idx: int) -> int:
    refs = list(getattr(sidx, "train_image_refs", []) or [])
    cams = [int(ref[1]) for ref in refs if int(ref[0]) == int(frame_idx)]
    if not cams:
        num_cams = int(getattr(sidx, "num_cams", 3))
        cams = list(range(max(num_cams, 0)))
    mask = 0
    for cam in cams:
        if 0 <= int(cam) < 32:
            mask |= 1 << int(cam)
    return int(mask)


def _timestamp_mapping_from_segment(sidx: Any) -> Dict[int, int]:
    for name in (
        "frame_timestamps_us",
        "frame_to_timestamp_us",
        "timestamp_us_by_frame",
        "timestamps_us_by_frame",
    ):
        value = getattr(sidx, name, None)
        if isinstance(value, dict):
            return {int(k): int(v) for k, v in value.items()}
    frames = [int(x) for x in list(getattr(sidx, "frame_indices", []) or [])]
    for name in ("timestamps_us", "frame_timestamps", "timestamps"):
        value = getattr(sidx, name, None)
        if value is None:
            continue
        values = list(value)
        if len(values) == len(frames):
            return {int(f): int(v) for f, v in zip(frames, values)}
    return {}


def _timestamp_us(
    *,
    sidx: Any,
    frame_idx: int,
    period_us: int,
    allow_synthetic: bool,
) -> Tuple[int, str]:
    mapping = _timestamp_mapping_from_segment(sidx)
    if int(frame_idx) in mapping:
        return int(mapping[int(frame_idx)]), "real_segment_index_timestamp_us"
    if bool(allow_synthetic):
        return int(frame_idx) * int(period_us), "frame_idx_times_frame_period_us"
    raise ValueError(
        "Stage2_2 requires real per-frame timestamp_us. Rebuild/export segment assets with "
        "frame_timestamps_us or set scheduler_stage2_2.time.allow_synthetic_timestamp=true for tests only."
    )


def _segment_pairs(dataset: Any, *, fixed_scene_id: Optional[int] = None, fixed_segment_id: Optional[int] = None) -> List[Tuple[int, int]]:
    scenes = [int(fixed_scene_id)] if fixed_scene_id is not None else [int(x) for x in dataset.list_training_scene_ids()]
    pairs: List[Tuple[int, int]] = []
    for scene_id in scenes:
        segments = (
            [int(fixed_segment_id)]
            if fixed_segment_id is not None
            else [int(x) for x in dataset.list_segment_ids(int(scene_id))]
        )
        for segment_id in segments:
            pairs.append((int(scene_id), int(segment_id)))
    return pairs


def _build_windows_for_segment(segment_row: int, frame_count: int) -> List[Tuple[int, int, int, int]]:
    rows: List[Tuple[int, int, int, int]] = []
    for protocol, patterns in PROTOCOL_PATTERNS.items():
        pid = int(PROTOCOL_IDS[str(protocol)])
        for pattern_id, offsets in enumerate(patterns):
            max_offset = int(max(offsets))
            if int(frame_count) <= max_offset:
                continue
            for start in range(0, int(frame_count) - max_offset):
                rows.append((int(segment_row), int(start), int(pid), int(pattern_id)))
    return rows


def _metadata_payload(
    *,
    cfg: Any,
    frame_period_us: int,
    segment_summaries: Sequence[Dict[str, Any]],
    timestamp_source: str,
    num_cams: int,
) -> Dict[str, Any]:
    data_cfg = _cfg_get(cfg, "data", {}) or {}
    scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "train_scene_ids", []) or [])]
    return {
        "index_version": STAGE22_INDEX_VERSION,
        "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
        "time": {
            "timestamp_source": str(timestamp_source),
            "frame_period_us": int(frame_period_us),
        },
        "camera": {"num_cams": int(num_cams)},
        "pose": {
            "source": "data_root_lidar_pose_aligned_segment0",
            "data_root": _pose_root_from_cfg(cfg),
        },
        "configured_train_scene_ids": scene_ids,
        "indexed_scene_ids": sorted({int(x["scene_id"]) for x in segment_summaries}),
        "segments": list(segment_summaries),
        "protocol_patterns": {
            str(k): [[int(x) for x in pattern] for pattern in patterns]
            for k, patterns in PROTOCOL_PATTERNS.items()
        },
        "protocol_offsets": {str(k): [int(x) for x in v] for k, v in PROTOCOL_OFFSETS.items()},
    }


def build_stage2_2_index_from_dataset(
    *,
    dataset: Any,
    cfg: Optional[Any] = None,
    output_dir: Optional[str | Path] = None,
    frame_period_us: Optional[int] = None,
    fixed_scene_id: Optional[int] = None,
    fixed_segment_id: Optional[int] = None,
) -> Stage22Index:
    cfg = cfg or {}
    sched = _cfg_get(cfg, "scheduler_stage2_2", {}) or {}
    time_cfg = _cfg_get(sched, "time", {}) or {}
    period_us = int(frame_period_us if frame_period_us is not None else _cfg_get(time_cfg, "frame_period_us", 100000))
    if period_us <= 0:
        raise ValueError("scheduler_stage2_2.time.frame_period_us must be > 0")
    allow_synthetic = bool(_cfg_get(time_cfg, "allow_synthetic_timestamp", False))
    if getattr(dataset, "_initialized", True) is False:
        dataset.initialize()
    data_root = _pose_root_from_cfg(cfg)
    segment_records: List[Tuple[int, int, int, int, int, int]] = []
    frame_records: List[Any] = []
    window_records: List[Tuple[int, int, int, int]] = []
    bootstrap_records: List[Tuple[int, int]] = []
    segment_summaries: List[Dict[str, Any]] = []
    timestamp_sources: set[str] = set()
    max_num_cams = 0
    for scene_id, segment_id in _segment_pairs(dataset, fixed_scene_id=fixed_scene_id, fixed_segment_id=fixed_segment_id):
        sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
        train_frames = _sorted_train_frames(sidx)
        if not train_frames:
            continue
        segment_row = len(segment_records)
        frame_start = len(frame_records)
        poses = [_read_pose(data_root, int(scene_id), int(frame_idx)) for frame_idx in train_frames]
        origin_t = poses[0][0]
        origin_yaw = poses[0][1]
        max_num_cams = max(max_num_cams, int(getattr(sidx, "num_cams", 0) or 0))
        for local, frame_idx in enumerate(train_frames):
            trans, yaw = poses[int(local)]
            timestamp_us, timestamp_source = _timestamp_us(
                sidx=sidx,
                frame_idx=int(frame_idx),
                period_us=int(period_us),
                allow_synthetic=allow_synthetic,
            )
            timestamp_sources.add(str(timestamp_source))
            aligned = (
                float(trans[0] - origin_t[0]),
                float(trans[1] - origin_t[1]),
                float(trans[2] - origin_t[2]),
            )
            frame_records.append(
                (
                    int(scene_id),
                    int(segment_id),
                    int(frame_idx),
                    int(_frame_to_keyframe(sidx, int(frame_idx))),
                    int(timestamp_us),
                    aligned,
                    float(yaw - origin_yaw),
                    1,
                    int(_camera_mask(sidx, int(frame_idx))),
                )
            )
            bootstrap_records.append((int(segment_row), int(local)))
        frame_count = len(train_frames)
        keyframe_count = len(set(_frame_to_keyframe(sidx, int(f)) for f in train_frames))
        asset_id = stable_uint64((int(scene_id), int(segment_id), int(frame_count), int(train_frames[0]), int(train_frames[-1])))
        segment_records.append((int(scene_id), int(segment_id), int(frame_start), int(frame_count), int(keyframe_count), int(asset_id)))
        window_records.extend(_build_windows_for_segment(int(segment_row), int(frame_count)))
        segment_summaries.append(
            {
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "frame_count": int(frame_count),
                "first_frame": int(train_frames[0]),
                "last_frame": int(train_frames[-1]),
                "keyframe_count": int(keyframe_count),
                "asset_id_hash": int(asset_id),
            }
        )
    if not segment_records:
        raise ValueError("Stage2_2 index found no eligible train segments")
    if not window_records:
        raise ValueError("Stage2_2 index found no D1/D2/I123 stream10 windows")
    segments = np.array(segment_records, dtype=SEGMENT_DTYPE)
    frames = np.array(frame_records, dtype=FRAME_DTYPE)
    windows = np.array(window_records, dtype=WINDOW_DTYPE)
    bootstrap = np.array(bootstrap_records, dtype=BOOTSTRAP_DTYPE)
    timestamp_source = "+".join(sorted(timestamp_sources)) if timestamp_sources else "unknown"
    payload = _metadata_payload(
        cfg=cfg,
        frame_period_us=int(period_us),
        segment_summaries=segment_summaries,
        timestamp_source=str(timestamp_source),
        num_cams=int(max_num_cams if max_num_cams > 0 else 3),
    )
    fingerprint = fingerprint_payload(payload)
    metadata = {
        "index_version": STAGE22_INDEX_VERSION,
        "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
        "fingerprint": str(fingerprint),
        "fingerprint_payload": payload,
        "time": dict(payload["time"]),
        "camera": dict(payload["camera"]),
        "pose": dict(payload["pose"]),
        "num_segments": int(len(segments)),
        "num_frames": int(len(frames)),
        "num_windows": int(len(windows)),
        "num_bootstrap_frames": int(len(bootstrap)),
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "segments.npy", segments)
        np.save(root / "frames.npy", frames)
        np.save(root / "windows.npy", windows)
        np.save(root / "bootstrap_frames.npy", bootstrap)
        with (root / "metadata.json").open("w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True))
            f.write("\n")
    scene_table: Dict[int, List[int]] = {}
    for idx, seg in enumerate(segments):
        scene_table.setdefault(int(seg["scene_id"]), []).append(int(idx))
    return Stage22Index(
        metadata=dict(metadata),
        segments=segments,
        frames=frames,
        windows=windows,
        bootstrap_frames=bootstrap,
        scene_table={int(k): tuple(int(x) for x in v) for k, v in scene_table.items()},
    )


def build_stage2_2_index(*args: Any, **kwargs: Any) -> Stage22Index:
    return build_stage2_2_index_from_dataset(*args, **kwargs)


__all__ = ["build_stage2_2_index", "build_stage2_2_index_from_dataset"]
