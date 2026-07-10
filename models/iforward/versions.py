from __future__ import annotations

from typing import Any, FrozenSet


STAGE3_0_SCALAR_ANCHOR_CHILD_SUPPORT_PARENT_LEGACY_VERSION = (
    "stage3_0_scalar_anchor_child_support_parent_legacy"
)
STAGE3_0_LEGACY_FULL_SPARSE_GATHER_LIFT_VERSION = "stage3_0_full_sparse_gather_lift"
STAGE3_1_LOWRANK_GATED_DELTA_KV_LIFT_VERSION = "stage3_1_lowrank_gated_delta_kv_lift"
STAGE3_3_UNCERTAINTY_V1_VERSION = "stage3_3_uncertainty_v1"
STAGE3_3_UNCERTAINTY_V2_VERSION = "stage3_3_uncertainty_v2"

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

STAGE3_3_VERSION_ALIASES: FrozenSet[str] = frozenset(
    {
        STAGE3_3_UNCERTAINTY_V1_VERSION,
        "iforward_stage3_3_uncertainty_v1",
        STAGE3_3_UNCERTAINTY_V2_VERSION,
        "iforward_stage3_3_uncertainty_v2",
    }
)


def is_stage3_0_iforward_version(version: Any) -> bool:
    return str(version) in STAGE3_0_VERSION_ALIASES


def is_stage3_1_iforward_version(version: Any) -> bool:
    return str(version) in STAGE3_1_VERSION_ALIASES


def is_stage3_3_iforward_version(version: Any) -> bool:
    return str(version) in STAGE3_3_VERSION_ALIASES


def is_stage3_3_uncertainty_v2_version(version: Any) -> bool:
    return str(version) in {
        STAGE3_3_UNCERTAINTY_V2_VERSION,
        "iforward_stage3_3_uncertainty_v2",
    }


def uncertainty_schema_versions(version: Any) -> dict[str, str]:
    if is_stage3_3_uncertainty_v2_version(version):
        return {
            "uncertainty_state_version": "appearance_logvar_v1",
            "uncertainty_updater_version": "state_conditioned_target_v2",
            "uncertainty_raster_version": "detached_moments_aleatoric_loss_v2",
            "uncertainty_loss_version": "decoupled_warmup_v2",
        }
    return {
        "uncertainty_state_version": "appearance_logvar_v1",
        "uncertainty_updater_version": "delta_head_v1",
        "uncertainty_raster_version": "detached_moments_v1",
        "uncertainty_loss_version": "decoupled_nll_v1",
    }


def is_stage3_lowrank_gdkv_iforward_version(version: Any) -> bool:
    return is_stage3_1_iforward_version(version) or is_stage3_3_iforward_version(version)


def is_stage3_optimizer_memory_iforward_version(version: Any) -> bool:
    return is_stage3_0_iforward_version(version) or is_stage3_lowrank_gdkv_iforward_version(version)
