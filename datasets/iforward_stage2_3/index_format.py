from __future__ import annotations

from datasets.iforward_stage2_2.index_format import (
    BOOTSTRAP_DTYPE,
    FRAME_DTYPE,
    SEGMENT_DTYPE,
    WINDOW_DTYPE,
    canonical_json,
    fingerprint_payload,
    stable_uint64,
)

IFORWARD_STAGE2_3_SCHEDULER_VERSION = "iforward_2_3_scheduler_v3_optimizer_mamba"
STAGE23_INDEX_VERSION = "iforward_stage2_3_index_v1"

__all__ = [
    "BOOTSTRAP_DTYPE",
    "FRAME_DTYPE",
    "IFORWARD_STAGE2_3_SCHEDULER_VERSION",
    "SEGMENT_DTYPE",
    "STAGE23_INDEX_VERSION",
    "WINDOW_DTYPE",
    "canonical_json",
    "fingerprint_payload",
    "stable_uint64",
]
