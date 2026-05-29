from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .refs import ImageRef


PHASE_A_NAME = "phase_A_block_local_unroll"
PHASE_A_PROTOCOL_VERSION = "sf.phase_a.v1"


@dataclass(frozen=True)
class RolloutStep:
    step_idx: int
    evidence_refs: Tuple[ImageRef, ...]
    block_loss_refs: Tuple[ImageRef, ...]
    nearby_loss_refs: Tuple[ImageRef, ...] = ()


@dataclass(frozen=True)
class RolloutPlan:
    protocol_version: str
    phase: str
    scene_id: int
    segment_id: int
    episode_id: int
    num_cams: int
    inner_K: int
    steps: Tuple[RolloutStep, ...]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseALocalUnrollPlan(RolloutPlan):
    source_keyframe_idx: Optional[int] = None
    block_idx: Optional[int] = None


def _refs_from_raw(raw: Any) -> Tuple[ImageRef, ...]:
    return tuple(ImageRef.from_raw(x) for x in list(raw or []))


def rollout_step_from_mapping(raw: Mapping[str, Any], *, fallback_idx: int = 0) -> RolloutStep:
    return RolloutStep(
        step_idx=int(raw.get("step_idx", fallback_idx)),
        evidence_refs=_refs_from_raw(raw.get("evidence_refs") or ()),
        block_loss_refs=_refs_from_raw(raw.get("block_loss_refs") or ()),
        nearby_loss_refs=_refs_from_raw(raw.get("nearby_loss_refs") or ()),
    )


def phase_a_plan_from_mapping(raw: Mapping[str, Any]) -> PhaseALocalUnrollPlan:
    steps = tuple(
        rollout_step_from_mapping(step, fallback_idx=idx)
        for idx, step in enumerate(list(raw.get("steps") or []))
    )
    return PhaseALocalUnrollPlan(
        protocol_version=str(raw.get("protocol_version", PHASE_A_PROTOCOL_VERSION)),
        phase=str(raw.get("phase", PHASE_A_NAME)),
        scene_id=int(raw.get("scene_id", -1)),
        segment_id=int(raw.get("segment_id", -1)),
        episode_id=int(raw.get("episode_id", -1)),
        num_cams=int(raw.get("num_cams", 0) or 0),
        inner_K=int(raw.get("inner_K", len(steps)) or 0),
        steps=steps,
        meta=dict(raw.get("meta") or {}),
        source_keyframe_idx=(
            None if raw.get("source_keyframe_idx", None) is None else int(raw.get("source_keyframe_idx"))
        ),
        block_idx=None if raw.get("block_idx", None) is None else int(raw.get("block_idx")),
    )

