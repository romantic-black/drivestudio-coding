from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union


MemoryMode = Literal[
    "full",
    "memory_off",
    "memory_read_only",
    "memory_read_write",
    "memory_shuffle_state",
    "memory_freeze_write",
]

LEGACY_MEMORY_MODE_ALIASES = {
    "off": "memory_off",
    "read_only": "memory_read_only",
    "read_write": "memory_read_write",
    "shuffled": "memory_shuffle_state",
    "shuffle_memory": "memory_shuffle_state",
    "bypass_memory": "memory_off",
    "freeze_write": "memory_freeze_write",
    "mamba_off": "memory_off",
    "mamba_read_only": "memory_read_only",
    "mamba_read_write": "memory_read_write",
    "mamba_shuffle_state": "memory_shuffle_state",
    "mamba_freeze_write": "memory_freeze_write",
}

VALID_MEMORY_MODES = {
    "full",
    "memory_off",
    "memory_read_only",
    "memory_read_write",
    "memory_shuffle_state",
    "memory_freeze_write",
}

MEMORY_MODE_TO_FORWARD_ABLATION = {
    "full": "full",
    "memory_off": "mamba_off",
    "memory_read_only": "mamba_read_only",
    "memory_read_write": "mamba_read_write",
    "memory_shuffle_state": "mamba_shuffle_state",
    "memory_freeze_write": "mamba_freeze_write",
}


def normalize_memory_mode(value: Any) -> str:
    name = str(value or "full")
    normalized = str(LEGACY_MEMORY_MODE_ALIASES.get(name, name))
    if normalized not in VALID_MEMORY_MODES:
        raise ValueError(f"unsupported IForward runtime memory mode {value!r}")
    return normalized


def memory_mode_to_forward_ablation(value: Any) -> str:
    mode = normalize_memory_mode(value)
    return str(MEMORY_MODE_TO_FORWARD_ABLATION[mode])


@dataclass(frozen=True)
class EpisodeSpec:
    scene_id: int
    segment_id: int
    sequence_id: int
    frame_ids: tuple[int, ...]
    frame_positions: tuple[int, ...]
    cam_ids: tuple[int, ...]
    init_state: Literal["asset_fresh", "checkpoint", "snapshot"] = "asset_fresh"
    seed: int = 0
    protocol_name: str = ""
    episode_uid: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateEvent:
    event_id: str
    kind: Literal["observe_update", "repair_update"]
    rollout_plan: Any
    phase: Literal["assimilation", "repair", "bootstrap", "final_all", "repeat_stability"] = "assimilation"
    input_positions: tuple[int, ...] = ()
    repeat_budgets: tuple[int, ...] = ()
    blocks_per_rollout: int = 0
    repeats_per_block: int = 0
    memory_read: bool = True
    memory_write: bool = True
    observation_commit: bool = True
    parent_state_update: bool = True
    local_state_update: bool = True
    repair_training: bool = False
    memory_mode: str = "full"
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeEvent:
    event_id: str
    kind: Literal["render_probe"]
    target_positions: tuple[int, ...] = ()
    target_frame_ids: tuple[int, ...] = ()
    target_cams: tuple[int, ...] = ()
    roles: tuple[str, ...] = ("current", "history")
    update_state: bool = False
    compute_loss: bool = False
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    rollout_plan: Any = None


@dataclass(frozen=True)
class ControlEvent:
    event_id: str
    kind: Literal["reset_state", "snapshot_state", "restore_state", "set_memory_mode"]
    name: str = ""
    memory_mode: str = "full"
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


RuntimeEvent = Union[UpdateEvent, ProbeEvent, ControlEvent]


__all__ = [
    "ControlEvent",
    "EpisodeSpec",
    "LEGACY_MEMORY_MODE_ALIASES",
    "MEMORY_MODE_TO_FORWARD_ABLATION",
    "MemoryMode",
    "ProbeEvent",
    "RuntimeEvent",
    "UpdateEvent",
    "memory_mode_to_forward_ablation",
    "normalize_memory_mode",
]
