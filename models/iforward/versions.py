from __future__ import annotations

from typing import Any, FrozenSet


STAGE3_0_SCALAR_ANCHOR_CHILD_SUPPORT_PARENT_LEGACY_VERSION = (
    "stage3_0_scalar_anchor_child_support_parent_legacy"
)
STAGE3_0_LEGACY_FULL_SPARSE_GATHER_LIFT_VERSION = "stage3_0_full_sparse_gather_lift"

STAGE3_0_VERSION_ALIASES: FrozenSet[str] = frozenset(
    {
        STAGE3_0_SCALAR_ANCHOR_CHILD_SUPPORT_PARENT_LEGACY_VERSION,
        STAGE3_0_LEGACY_FULL_SPARSE_GATHER_LIFT_VERSION,
    }
)


def is_stage3_0_iforward_version(version: Any) -> bool:
    return str(version) in STAGE3_0_VERSION_ALIASES

