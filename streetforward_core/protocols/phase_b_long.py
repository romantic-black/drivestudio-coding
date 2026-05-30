from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from .refs import ImageRef
from .roles import LongRole


PHASE_B_LONG_NAME = "6_0_phase_b"
PHASE_B_LONG_PROTOCOL_VERSION = "sf.phase_b_long.v1"
PHASE_B_LONG_SCHEDULER_VERSION = "long_v1"
LONG_TARGET_ROLES = (
    LongRole.FINAL_HISTORY_RECON,
    LongRole.FINAL_HISTORY_NVS,
    LongRole.FINAL_CURRENT_RECON,
    LongRole.FINAL_CURRENT_NVS,
)


@dataclass(frozen=True)
class LongVisit:
    step_idx: int
    anchor_id: int
    frame_idx: int
    cam_idx: int
    repeat_idx: int
    rollout_order_rank: int
    chronological_rank: int
    visit_pos_code: float
    frame_time_code: float
    chronological_rank_code: float
    repeat_idx_code: float


@dataclass(frozen=True)
class PhaseBLongRolloutPlan:
    protocol_version: str
    scheduler_version: str
    phase: str
    scene_id: int
    segment_id: int
    episode_window_id: int
    rollout_id_in_episode: int
    shape_name: str
    repeats_per_anchor: int
    anchors_per_rollout: int
    inner_K: int
    anchor_frames_chronological: Tuple[int, ...]
    anchor_frames_rollout_order: Tuple[int, ...]
    visits: Tuple[LongVisit, ...]
    evidence_refs_by_step: Tuple[Tuple[ImageRef, ...], ...]
    final_history_recon_refs: Tuple[ImageRef, ...]
    final_history_nvs_refs: Tuple[ImageRef, ...]
    final_current_recon_refs: Tuple[ImageRef, ...]
    final_current_nvs_refs: Tuple[ImageRef, ...]
    source_image_refs: Tuple[ImageRef, ...]
    target_image_refs: Tuple[ImageRef, ...]
    target_image_roles: Tuple[LongRole, ...]
    rigid_meta: Dict[str, Any] = field(default_factory=dict)
    distant_meta: Dict[str, Any] = field(default_factory=dict)
    tbptt: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedLongPhaseBBatch:
    raw: Dict[str, Any]
    plan: PhaseBLongRolloutPlan
    source_index_by_ref: Dict[ImageRef, int]
    target_index_by_ref: Dict[ImageRef, int]
    evidence_source_indices_by_step: Tuple[Tuple[int, ...], ...]
    final_history_recon_target_indices: Tuple[int, ...]
    final_history_nvs_target_indices: Tuple[int, ...]
    final_current_recon_target_indices: Tuple[int, ...]
    final_current_nvs_target_indices: Tuple[int, ...]

    @property
    def inner_K(self) -> int:
        return int(self.plan.inner_K)

    @property
    def visits(self) -> Tuple[LongVisit, ...]:
        return tuple(self.plan.visits)

    @property
    def visit_time_codes(self) -> Tuple[Tuple[float, float, float, float], ...]:
        return tuple(
            (
                float(v.visit_pos_code),
                float(v.frame_time_code),
                float(v.chronological_rank_code),
                float(v.repeat_idx_code),
            )
            for v in self.plan.visits
        )


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _refs(raw: Any) -> Tuple[ImageRef, ...]:
    return tuple(ImageRef.from_raw(ref) for ref in list(raw or []))


def _ref_groups(raw: Any) -> Tuple[Tuple[ImageRef, ...], ...]:
    return tuple(tuple(ImageRef.from_raw(ref) for ref in list(group)) for group in list(raw or []))


def _visit_from_raw(raw: Any, *, fallback_idx: int = 0) -> LongVisit:
    return LongVisit(
        step_idx=int(_field(raw, "step_idx", fallback_idx)),
        anchor_id=int(_field(raw, "anchor_id", _field(raw, "rollout_order_rank", 0))),
        frame_idx=int(_field(raw, "frame_idx")),
        cam_idx=int(_field(raw, "cam_idx", 0)),
        repeat_idx=int(_field(raw, "repeat_idx", 0)),
        rollout_order_rank=int(_field(raw, "rollout_order_rank", _field(raw, "anchor_id", 0))),
        chronological_rank=int(_field(raw, "chronological_rank", 0)),
        visit_pos_code=float(_field(raw, "visit_pos_code", 0.0)),
        frame_time_code=float(_field(raw, "frame_time_code", 0.0)),
        chronological_rank_code=float(_field(raw, "chronological_rank_code", 0.0)),
        repeat_idx_code=float(_field(raw, "repeat_idx_code", 0.0)),
    )


def _role(raw: Any) -> LongRole:
    if isinstance(raw, LongRole):
        return raw
    return LongRole(str(raw))


def _roles(raw: Any) -> Tuple[LongRole, ...]:
    return tuple(_role(role) for role in list(raw or []))


def phase_b_long_plan_from_mapping(raw: Mapping[str, Any]) -> PhaseBLongRolloutPlan:
    meta = dict(raw.get("meta") or {})
    for key in (
        "required_final_roles",
        "nvs_fallback_count",
        "nvs_fallback_to_evidence_cam_ratio",
        "max_nvs_fallback_ratio",
        "anchor_order_mode",
        "target_role_set",
        "num_cams",
        "episode_id",
        "episode_idx_global",
        "rollout_id",
    ):
        if key in raw and key not in meta:
            meta[key] = raw[key]
    return PhaseBLongRolloutPlan(
        protocol_version=str(raw.get("protocol_version", PHASE_B_LONG_PROTOCOL_VERSION)),
        scheduler_version=str(raw.get("scheduler_version", raw.get("version", PHASE_B_LONG_SCHEDULER_VERSION))),
        phase=str(raw.get("scheduler_phase", raw.get("phase", PHASE_B_LONG_NAME))),
        scene_id=int(raw.get("scene_id", -1)),
        segment_id=int(raw.get("segment_id", -1)),
        episode_window_id=int(raw.get("episode_window_id", raw.get("episode_id", -1))),
        rollout_id_in_episode=int(raw.get("rollout_id_in_episode", 0)),
        shape_name=str(raw.get("shape_name", "")),
        repeats_per_anchor=int(raw.get("repeats_per_anchor", 0) or 0),
        anchors_per_rollout=int(raw.get("anchors_per_rollout", 0) or 0),
        inner_K=int(raw.get("inner_K", 0) or 0),
        anchor_frames_chronological=tuple(int(x) for x in list(raw.get("anchor_frames_chronological") or [])),
        anchor_frames_rollout_order=tuple(int(x) for x in list(raw.get("anchor_frames_rollout_order") or [])),
        visits=tuple(_visit_from_raw(item, fallback_idx=idx) for idx, item in enumerate(list(raw.get("visits") or []))),
        evidence_refs_by_step=_ref_groups(raw.get("evidence_refs_by_step") or ()),
        final_history_recon_refs=_refs(raw.get("final_history_recon_refs") or ()),
        final_history_nvs_refs=_refs(raw.get("final_history_nvs_refs") or ()),
        final_current_recon_refs=_refs(raw.get("final_current_recon_refs") or ()),
        final_current_nvs_refs=_refs(raw.get("final_current_nvs_refs") or ()),
        source_image_refs=_refs(raw.get("source_image_refs") or ()),
        target_image_refs=_refs(raw.get("target_image_refs") or ()),
        target_image_roles=_roles(raw.get("target_image_roles") or ()),
        rigid_meta=dict(raw.get("rigid_meta") or {}),
        distant_meta=dict(raw.get("distant_meta") or {}),
        tbptt=dict(raw.get("tbptt") or {}),
        meta=meta,
    )


def phase_b_long_plan_to_request_meta(plan: PhaseBLongRolloutPlan) -> Dict[str, Any]:
    meta = dict(plan.meta or {})
    visits = [
        {
            "step_idx": int(v.step_idx),
            "anchor_id": int(v.anchor_id),
            "frame_idx": int(v.frame_idx),
            "cam_idx": int(v.cam_idx),
            "repeat_idx": int(v.repeat_idx),
            "rollout_order_rank": int(v.rollout_order_rank),
            "chronological_rank": int(v.chronological_rank),
            "visit_pos_code": float(v.visit_pos_code),
            "frame_time_code": float(v.frame_time_code),
            "chronological_rank_code": float(v.chronological_rank_code),
            "repeat_idx_code": float(v.repeat_idx_code),
        }
        for v in plan.visits
    ]
    meta.update(
        {
            "protocol_version": str(plan.protocol_version),
            "scheduler_version": str(plan.scheduler_version),
            "scheduler_phase": str(plan.phase),
            "assembly_mode": "image_ref_long_v1",
            "scene_id": int(plan.scene_id),
            "segment_id": int(plan.segment_id),
            "episode_window_id": int(plan.episode_window_id),
            "episode_id": int(meta.get("episode_id", plan.episode_window_id)),
            "episode_idx_global": int(meta.get("episode_idx_global", plan.episode_window_id)),
            "rollout_id_in_episode": int(plan.rollout_id_in_episode),
            "shape_name": str(plan.shape_name),
            "repeats_per_anchor": int(plan.repeats_per_anchor),
            "anchors_per_rollout": int(plan.anchors_per_rollout),
            "inner_K": int(plan.inner_K),
            "anchor_frames_chronological": [int(x) for x in plan.anchor_frames_chronological],
            "anchor_frames_rollout_order": [int(x) for x in plan.anchor_frames_rollout_order],
            "visits": visits,
            "evidence_refs_by_step": [[ref.as_tuple() for ref in step] for step in plan.evidence_refs_by_step],
            "step_frame_indices": [int(v.frame_idx) for v in plan.visits],
            "step_repeat_indices": [int(v.repeat_idx) for v in plan.visits],
            "step_anchor_ids": [int(v.anchor_id) for v in plan.visits],
            "step_rollout_order_ranks": [int(v.rollout_order_rank) for v in plan.visits],
            "step_chronological_ranks": [int(v.chronological_rank) for v in plan.visits],
            "visit_pos_codes": [float(v.visit_pos_code) for v in plan.visits],
            "frame_time_codes": [float(v.frame_time_code) for v in plan.visits],
            "chronological_rank_codes": [float(v.chronological_rank_code) for v in plan.visits],
            "repeat_idx_codes": [float(v.repeat_idx_code) for v in plan.visits],
            "source_image_refs": [ref.as_tuple() for ref in plan.source_image_refs],
            "source_image_ref": plan.source_image_refs[0].as_tuple() if plan.source_image_refs else None,
            "target_image_refs": [ref.as_tuple() for ref in plan.target_image_refs],
            "target_image_roles": [role.value for role in plan.target_image_roles],
            "final_history_recon_refs": [ref.as_tuple() for ref in plan.final_history_recon_refs],
            "final_history_nvs_refs": [ref.as_tuple() for ref in plan.final_history_nvs_refs],
            "final_current_recon_refs": [ref.as_tuple() for ref in plan.final_current_recon_refs],
            "final_current_nvs_refs": [ref.as_tuple() for ref in plan.final_current_nvs_refs],
            "target_role_set": [role.value for role in LONG_TARGET_ROLES],
            "query_label_refs": [],
            "prefix_loss_refs_by_step": [[] for _ in plan.evidence_refs_by_step],
            "nearby_loss_refs_by_step": [[] for _ in plan.evidence_refs_by_step],
            "block_loss_refs_by_step": [[] for _ in plan.evidence_refs_by_step],
            "rigid_meta": dict(plan.rigid_meta or {}),
            "distant_meta": dict(plan.distant_meta or {}),
            "tbptt": dict(plan.tbptt or {}),
        }
    )
    return meta
