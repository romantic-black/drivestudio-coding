from __future__ import annotations

from typing import Any, FrozenSet


STAGE3_0_SCALAR_ANCHOR_CHILD_SUPPORT_PARENT_LEGACY_VERSION = (
    "stage3_0_scalar_anchor_child_support_parent_legacy"
)
STAGE3_0_LEGACY_FULL_SPARSE_GATHER_LIFT_VERSION = "stage3_0_full_sparse_gather_lift"
STAGE3_1_LOWRANK_GATED_DELTA_KV_LIFT_VERSION = "stage3_1_lowrank_gated_delta_kv_lift"

STAGE3_0_VERSION_ALIASES: FrozenSet[str] = frozenset(
    {
        STAGE3_0_SCALAR_ANCHOR_CHILD_SUPPORT_PARENT_LEGACY_VERSION,
        STAGE3_0_LEGACY_FULL_SPARSE_GATHER_LIFT_VERSION,
    }
)

STAGE3_1_VERSION_ALIASES: FrozenSet[str] = frozenset(
    {
        STAGE3_1_LOWRANK_GATED_DELTA_KV_LIFT_VERSION,
        "iforward_stage3_1_lowrank_gated_delta_kv_lift",
    }
)


def is_stage3_0_iforward_version(version: Any) -> bool:
    return str(version) in STAGE3_0_VERSION_ALIASES


def is_stage3_1_iforward_version(version: Any) -> bool:
    return str(version) in STAGE3_1_VERSION_ALIASES


def is_stage3_optimizer_memory_iforward_version(version: Any) -> bool:
    return is_stage3_0_iforward_version(version) or is_stage3_1_iforward_version(version)
