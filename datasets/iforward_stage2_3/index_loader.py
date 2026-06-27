from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from datasets.iforward_stage2_2.index_loader import Stage22Index, load_stage2_2_index

from .index_format import IFORWARD_STAGE2_3_SCHEDULER_VERSION, STAGE23_INDEX_VERSION


@dataclass(frozen=True)
class Stage23Index:
    metadata: Dict[str, Any]
    segments: np.ndarray
    frames: np.ndarray
    windows: np.ndarray
    bootstrap_frames: np.ndarray
    scene_table: Dict[int, Sequence[int]]

    @classmethod
    def from_stage22(cls, base: Stage22Index) -> "Stage23Index":
        meta = dict(base.metadata)
        meta["index_version_stage2_3"] = STAGE23_INDEX_VERSION
        meta["scheduler_version_stage2_3"] = IFORWARD_STAGE2_3_SCHEDULER_VERSION
        return cls(
            metadata=meta,
            segments=base.segments,
            frames=base.frames,
            windows=base.windows,
            bootstrap_frames=base.bootstrap_frames,
            scene_table=dict(base.scene_table),
        )

    @property
    def fingerprint(self) -> str:
        return str(self.metadata.get("fingerprint", ""))

    @property
    def frame_period_us(self) -> int:
        return int(dict(self.metadata.get("time", {}) or {}).get("frame_period_us", 100000))

    @property
    def timestamp_source(self) -> str:
        return str(dict(self.metadata.get("time", {}) or {}).get("timestamp_source", ""))

    @property
    def num_cams(self) -> int:
        return int(dict(self.metadata.get("camera", {}) or {}).get("num_cams", 3))

    def frames_for_segment_row(self, segment_row: int) -> np.ndarray:
        seg = self.segments[int(segment_row)]
        start = int(seg["frame_start"])
        count = int(seg["frame_count"])
        return self.frames[start : start + count]

    def validate_expected_fingerprint(self, expected: Optional[str]) -> None:
        if expected is None or str(expected) == "":
            return
        got = str(self.fingerprint)
        if got != str(expected):
            raise ValueError(f"Stage2_3 index fingerprint mismatch: expected={expected} got={got}")


def load_stage2_3_index(
    index_dir: str | Path,
    *,
    expected_fingerprint: Optional[str] = None,
    mmap_mode: str | None = "r",
) -> Stage23Index:
    base = load_stage2_2_index(index_dir, expected_fingerprint=expected_fingerprint, mmap_mode=mmap_mode)
    return Stage23Index.from_stage22(base)


__all__ = ["Stage23Index", "load_stage2_3_index"]
