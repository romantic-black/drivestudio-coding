from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import numpy as np


IFORWARD_STAGE2_2_SCHEDULER_VERSION = "iforward_stage2_2_stream10_rawframe"
STAGE22_INDEX_VERSION = "iforward_stage2_2_index_v2"

SEGMENT_DTYPE = np.dtype(
    [
        ("scene_id", "<i4"),
        ("segment_id", "<i4"),
        ("frame_start", "<i4"),
        ("frame_count", "<i4"),
        ("keyframe_count", "<i4"),
        ("asset_id_hash", "<u8"),
    ]
)

FRAME_DTYPE = np.dtype(
    [
        ("scene_id", "<i4"),
        ("segment_id", "<i4"),
        ("frame_idx", "<i4"),
        ("keyframe_idx", "<i4"),
        ("timestamp_us", "<i8"),
        ("ego_translation", "<f4", (3,)),
        ("ego_yaw", "<f4"),
        ("is_train", "u1"),
        ("available_camera_mask", "<u4"),
    ]
)

WINDOW_DTYPE = np.dtype(
    [
        ("segment_row", "<i4"),
        ("start_local_frame", "<i4"),
        ("protocol_id", "<i2"),
        ("pattern_id", "<i2"),
    ]
)

BOOTSTRAP_DTYPE = np.dtype(
    [
        ("segment_row", "<i4"),
        ("local_frame", "<i4"),
    ]
)

PROTOCOL_IDS: Dict[str, int] = {"D1": 1, "D2": 2, "I123": 3}
PROTOCOL_NAMES = {int(v): str(k) for k, v in PROTOCOL_IDS.items()}
PROTOCOL_PATTERNS = {
    "D1": (tuple(range(10)),),
    "D2": (tuple(i * 2 for i in range(10)),),
    "I123": (
        (0, 1, 3, 6, 7, 9, 12, 13, 15, 18),
        (0, 1, 2, 4, 7, 8, 10, 13, 15, 18),
        (0, 2, 3, 5, 6, 9, 11, 12, 15, 18),
        (0, 1, 4, 5, 7, 10, 11, 14, 16, 18),
    ),
}
PROTOCOL_OFFSETS = {str(k): tuple(v[0]) for k, v in PROTOCOL_PATTERNS.items()}


def protocol_offsets(protocol: str, pattern_id: int = 0) -> tuple[int, ...]:
    patterns = PROTOCOL_PATTERNS[str(protocol)]
    idx = int(pattern_id)
    if idx < 0 or idx >= len(patterns):
        raise ValueError(f"Stage2_2 invalid pattern_id={idx} for protocol={protocol!r}")
    return tuple(int(x) for x in patterns[idx])


def stable_uint64(value: Any) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "BOOTSTRAP_DTYPE",
    "FRAME_DTYPE",
    "IFORWARD_STAGE2_2_SCHEDULER_VERSION",
    "PROTOCOL_IDS",
    "PROTOCOL_NAMES",
    "PROTOCOL_OFFSETS",
    "PROTOCOL_PATTERNS",
    "SEGMENT_DTYPE",
    "STAGE22_INDEX_VERSION",
    "WINDOW_DTYPE",
    "canonical_json",
    "fingerprint_payload",
    "protocol_offsets",
    "stable_uint64",
]
