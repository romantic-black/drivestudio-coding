from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .index_format import (
    BOOTSTRAP_DTYPE,
    FRAME_DTYPE,
    IFORWARD_STAGE2_2_SCHEDULER_VERSION,
    PROTOCOL_IDS,
    PROTOCOL_NAMES,
    SEGMENT_DTYPE,
    STAGE22_INDEX_VERSION,
    WINDOW_DTYPE,
    fingerprint_payload,
)


@dataclass(frozen=True)
class Stage22Index:
    metadata: Dict[str, Any]
    segments: np.ndarray
    frames: np.ndarray
    windows: np.ndarray
    bootstrap_frames: np.ndarray
    scene_table: Dict[int, Sequence[int]]

    @property
    def fingerprint(self) -> str:
        return str(self.metadata.get("fingerprint", ""))

    @property
    def frame_period_us(self) -> int:
        time_cfg = dict(self.metadata.get("time", {}) or {})
        return int(time_cfg.get("frame_period_us", 100000))

    @property
    def timestamp_source(self) -> str:
        time_cfg = dict(self.metadata.get("time", {}) or {})
        return str(time_cfg.get("timestamp_source", ""))

    @property
    def num_cams(self) -> int:
        camera_cfg = dict(self.metadata.get("camera", {}) or {})
        return int(camera_cfg.get("num_cams", 3))

    def segment_row(self, scene_id: int, segment_id: int) -> int:
        rows = np.nonzero(
            (self.segments["scene_id"] == int(scene_id)) & (self.segments["segment_id"] == int(segment_id))
        )[0]
        if int(rows.size) != 1:
            raise KeyError(f"Stage2_2 index segment not found: scene={int(scene_id)} segment={int(segment_id)}")
        return int(rows[0])

    def frames_for_segment_row(self, segment_row: int) -> np.ndarray:
        seg = self.segments[int(segment_row)]
        start = int(seg["frame_start"])
        count = int(seg["frame_count"])
        return self.frames[start : start + count]

    def windows_for_protocol(self, protocol: str) -> np.ndarray:
        pid = int(PROTOCOL_IDS[str(protocol)])
        return self.windows[self.windows["protocol_id"] == pid]

    def validate_expected_fingerprint(self, expected: Optional[str]) -> None:
        if expected is None or str(expected) == "":
            return
        got = str(self.fingerprint)
        if got != str(expected):
            raise ValueError(f"Stage2_2 index fingerprint mismatch: expected={expected} got={got}")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Stage2_2 metadata must be a JSON object: {path}")
    return dict(data)


def _scene_table_from_segments(segments: np.ndarray) -> Dict[int, Sequence[int]]:
    table: Dict[int, list[int]] = {}
    for idx, seg in enumerate(segments):
        table.setdefault(int(seg["scene_id"]), []).append(int(idx))
    return {int(k): tuple(int(x) for x in v) for k, v in table.items()}


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if str(metadata.get("index_version", "")) != STAGE22_INDEX_VERSION:
        raise ValueError(
            f"Stage2_2 index_version must be {STAGE22_INDEX_VERSION!r}, got {metadata.get('index_version')!r}"
        )
    if str(metadata.get("scheduler_version", "")) != IFORWARD_STAGE2_2_SCHEDULER_VERSION:
        raise ValueError(
            "Stage2_2 scheduler_version mismatch: "
            f"{metadata.get('scheduler_version')!r} != {IFORWARD_STAGE2_2_SCHEDULER_VERSION!r}"
        )
    fp_payload = dict(metadata.get("fingerprint_payload", {}) or {})
    if fp_payload:
        computed = fingerprint_payload(fp_payload)
        if str(metadata.get("fingerprint", "")) != str(computed):
            raise ValueError("Stage2_2 metadata fingerprint does not match fingerprint_payload")


def load_stage2_2_index(
    index_dir: str | Path,
    *,
    expected_fingerprint: Optional[str] = None,
    mmap_mode: str | None = "r",
) -> Stage22Index:
    root = Path(index_dir)
    if not root.exists():
        raise FileNotFoundError(f"Stage2_2 index directory does not exist: {root}")
    metadata = _load_json(root / "metadata.json")
    _validate_metadata(metadata)
    segments = np.load(root / "segments.npy", mmap_mode=mmap_mode)
    frames = np.load(root / "frames.npy", mmap_mode=mmap_mode)
    windows = np.load(root / "windows.npy", mmap_mode=mmap_mode)
    bootstrap = np.load(root / "bootstrap_frames.npy", mmap_mode=mmap_mode)
    if segments.dtype != SEGMENT_DTYPE:
        raise ValueError(f"Stage2_2 segments dtype mismatch: {segments.dtype}")
    if frames.dtype != FRAME_DTYPE:
        raise ValueError(f"Stage2_2 frames dtype mismatch: {frames.dtype}")
    if windows.dtype != WINDOW_DTYPE:
        raise ValueError(f"Stage2_2 windows dtype mismatch: {windows.dtype}")
    if bootstrap.dtype != BOOTSTRAP_DTYPE:
        raise ValueError(f"Stage2_2 bootstrap dtype mismatch: {bootstrap.dtype}")
    out = Stage22Index(
        metadata=dict(metadata),
        segments=segments,
        frames=frames,
        windows=windows,
        bootstrap_frames=bootstrap,
        scene_table=_scene_table_from_segments(segments),
    )
    out.validate_expected_fingerprint(expected_fingerprint)
    protocols = set(PROTOCOL_NAMES.get(int(pid), "") for pid in np.unique(windows["protocol_id"]))
    if not {"D1", "D2", "I123"} & protocols:
        raise ValueError("Stage2_2 index contains no supported protocol windows")
    return out


__all__ = ["Stage22Index", "load_stage2_2_index"]
