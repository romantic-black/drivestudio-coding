from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.streetforward.stage6_0.phase_b_long.types import PHASE_B_LONG_NAME, ImageRef, LongVisit


@dataclass(frozen=True)
class LongPhaseBValidationSpec:
    scene_id: int
    segment_id: int
    interval_T: int
    order: str
    frame_pool: List[int]
    evidence_frames: List[int]
    visits: List[LongVisit]
    evidence_refs_by_step: List[List[ImageRef]]
    target_image_refs: List[ImageRef]
    target_image_roles: List[str]
    validation_buckets: Dict[str, List[ImageRef]]
    request_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LongPhaseBValidationPlan:
    specs: List[LongPhaseBValidationSpec]
    interval_T_values: List[int]
    orders: List[str]


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        out = node.get(key, default)
        return default if out is None else out
    return getattr(node, key, default)


def _segment_frames(sidx: Any) -> List[int]:
    raw = list(getattr(sidx, "frame_indices", []) or [])
    if raw:
        return sorted({int(x) for x in raw})
    out: List[int] = []
    for group in dict(getattr(sidx, "keyframe_to_frames", {}) or {}).values():
        out.extend(int(x) for x in list(group or []))
    return sorted(set(out))


def _dedupe(refs: Sequence[ImageRef]) -> List[ImageRef]:
    seen = set()
    out: List[ImageRef] = []
    for ref in refs:
        r = (int(ref[0]), int(ref[1]))
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _order_frames(frames: List[int], order: str) -> List[int]:
    if order == "chronological":
        return list(frames)
    if order == "reverse":
        return list(reversed(frames))
    if order in {"random", "random_seeded"}:
        import random

        rng = random.Random(1337 + len(frames) * 17)
        out = list(frames)
        rng.shuffle(out)
        return out
    raise ValueError(f"unsupported validation_long_phase_b order={order!r}")


def _enabled(node: Any, *, default: bool = True) -> bool:
    return bool(_cfg_get(node, "enable", default))


def _segment_ids_for_scene(dataset: Any, scene_id: int, segment_cfg: Any) -> List[int]:
    available = [int(x) for x in dataset.list_segment_ids(int(scene_id))]
    if not available:
        return []
    fixed = [int(x) for x in list(_cfg_get(segment_cfg, "fixed_segment_ids", []) or [])]
    if fixed:
        available_set = set(available)
        out = [int(x) for x in fixed if int(x) in available_set]
        if not out:
            raise ValueError(
                "validation_long_phase_b.segment.fixed_segment_ids did not match any available segment "
                f"for scene_id={int(scene_id)}."
            )
        return out

    policy = str(_cfg_get(segment_cfg, "policy", "first"))
    max_segments = max(int(_cfg_get(segment_cfg, "max_segments_per_scene", 1)), 1)
    if policy in {"first", "fixed_eval_segments"}:
        return available[:max_segments]
    if policy == "all":
        return available[:max_segments]
    raise ValueError(f"unsupported validation_long_phase_b.segment.policy={policy!r}")


def _evidence_cams_for_frame(
    *,
    cam_pool: List[int],
    order_rank: int,
    count: int,
    policy: str,
) -> List[int]:
    if not cam_pool:
        raise ValueError("validation_long_phase_b requires num_cams >= 1.")
    n = max(1, min(int(count), len(cam_pool)))
    if policy == "fixed_round_robin":
        start = int(order_rank) % len(cam_pool)
        return [int(cam_pool[(start + j) % len(cam_pool)]) for j in range(n)]
    if policy == "first_n":
        return [int(x) for x in cam_pool[:n]]
    raise ValueError(f"unsupported validation_long_phase_b.evidence.cam_policy={policy!r}")


def build_validation_plan_long_phase_b(
    *,
    dataset: Any,
    cfg: Any,
    eval_scene_ids: Sequence[int],
) -> LongPhaseBValidationPlan:
    raw = _cfg_get(cfg, "validation_long_phase_b", {}) or {}
    if not bool(_cfg_get(raw, "enable", False)):
        return LongPhaseBValidationPlan(specs=[], interval_T_values=[], orders=[])
    if str(_cfg_get(raw, "version", "long_v1")) != "long_v1":
        raise ValueError("validation_long_phase_b.version must be long_v1.")
    if str(_cfg_get(raw, "phase", PHASE_B_LONG_NAME)) != PHASE_B_LONG_NAME:
        raise ValueError("validation_long_phase_b.phase must be 6_0_phase_b.")
    evidence_cfg = _cfg_get(raw, "evidence", {}) or {}
    render_cfg = _cfg_get(raw, "render_eval", {}) or {}
    segment_cfg = _cfg_get(raw, "segment", {}) or {}
    interval_T_values = [int(x) for x in list(_cfg_get(evidence_cfg, "interval_T_values", [1, 2, 4, 8]) or [])]
    if not interval_T_values:
        raise ValueError("validation_long_phase_b.evidence.interval_T_values must not be empty.")
    repeats = max(int(_cfg_get(evidence_cfg, "repeats_per_evidence_frame", 4)), 1)
    evidence_cams_per_frame = max(int(_cfg_get(evidence_cfg, "evidence_cams_per_frame", 1)), 1)
    evidence_cam_policy = str(_cfg_get(evidence_cfg, "cam_policy", "fixed_round_robin"))
    primary_order = str(_cfg_get(_cfg_get(raw, "order", {}) or {}, "primary", "chronological"))
    extra_orders = [str(x) for x in list(_cfg_get(_cfg_get(raw, "order", {}) or {}, "extra_orders", []) or [])]
    orders = [primary_order] + [x for x in extra_orders if x != primary_order]
    max_frames = int(_cfg_get(segment_cfg, "max_frames_per_segment", 80))
    target_stride = max(int(_cfg_get(segment_cfg, "target_frame_stride", 1)), 1)
    reconstruction_cfg = _cfg_get(render_cfg, "reconstruction", {}) or {}
    same_frame_cfg = _cfg_get(render_cfg, "nvs_same_frame", {}) or {}
    temporal_cfg = _cfg_get(render_cfg, "temporal_nvs", {}) or {}
    segment_all_cfg = _cfg_get(render_cfg, "segment_all", {}) or {}
    heldout_same = max(int(_cfg_get(same_frame_cfg, "heldout_cams_per_evidence_frame", 2)), 0)
    temporal_cams = max(int(_cfg_get(temporal_cfg, "cams_per_non_evidence_frame", 2)), 0)
    max_render_refs = max(int(_cfg_get(segment_all_cfg, "max_render_refs", 512)), 1)
    reconstruction_enabled = _enabled(reconstruction_cfg, default=True) and bool(
        _cfg_get(reconstruction_cfg, "use_evidence_frame_evidence_cam", True)
    )
    same_frame_enabled = _enabled(same_frame_cfg, default=True)
    temporal_enabled = _enabled(temporal_cfg, default=True) and bool(
        _cfg_get(temporal_cfg, "eval_non_evidence_frames", True)
    )

    specs: List[LongPhaseBValidationSpec] = []
    for scene_id in [int(x) for x in eval_scene_ids]:
        segment_ids = _segment_ids_for_scene(dataset, int(scene_id), segment_cfg)
        for segment_id in segment_ids:
            sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
            frames_all = _segment_frames(sidx)[:max_frames]
            if not frames_all:
                continue
            frames = [int(x) for x in frames_all[::target_stride]]
            if not frames:
                frames = [int(frames_all[0])]
            cam_pool = list(range(int(getattr(sidx, "num_cams", 1))))
            frame_span = max(int(frames[-1]) - int(frames[0]), 1)
            for T in interval_T_values:
                evidence_frames = [int(f) for idx, f in enumerate(frames) if idx % int(T) == 0]
                if not evidence_frames:
                    evidence_frames = [int(frames[0])]
                for order in orders:
                    ordered = _order_frames(evidence_frames, str(order))
                    visits: List[LongVisit] = []
                    evidence_refs_by_step: List[List[ImageRef]] = []
                    evidence_refs_used: List[ImageRef] = []
                    for order_rank, frame_idx in enumerate(ordered):
                        frame_evidence_cams = _evidence_cams_for_frame(
                            cam_pool=cam_pool,
                            order_rank=int(order_rank),
                            count=int(evidence_cams_per_frame),
                            policy=str(evidence_cam_policy),
                        )
                        for repeat_idx in range(int(repeats)):
                            cam_idx = int(frame_evidence_cams[int(repeat_idx) % len(frame_evidence_cams)])
                            step_idx = len(visits)
                            chron_rank = evidence_frames.index(int(frame_idx))
                            visit = LongVisit(
                                step_idx=int(step_idx),
                                anchor_id=int(order_rank),
                                frame_idx=int(frame_idx),
                                cam_idx=int(cam_idx),
                                repeat_idx=int(repeat_idx),
                                rollout_order_rank=int(order_rank),
                                chronological_rank=int(chron_rank),
                                visit_pos_code=float(step_idx) / float(max(len(evidence_frames) * repeats - 1, 1)),
                                frame_time_code=float(int(frame_idx) - int(frames[0])) / float(frame_span),
                                chronological_rank_code=float(chron_rank) / float(max(len(evidence_frames) - 1, 1)),
                                repeat_idx_code=float(repeat_idx) / float(max(repeats - 1, 1)),
                            )
                            visits.append(visit)
                            ref = (int(frame_idx), int(cam_idx))
                            evidence_refs_by_step.append([ref])
                            evidence_refs_used.append(ref)
                    evidence_set = set(evidence_refs_used)
                    reconstruction = _dedupe(evidence_refs_used) if reconstruction_enabled else []
                    same_frame_nvs: List[ImageRef] = []
                    if same_frame_enabled:
                        for frame_idx in evidence_frames:
                            heldout = [int(c) for c in cam_pool if (int(frame_idx), int(c)) not in evidence_set]
                            same_frame_nvs.extend((int(frame_idx), int(c)) for c in heldout[:heldout_same])
                    temporal_nvs: List[ImageRef] = []
                    if temporal_enabled:
                        for frame_idx in frames:
                            if int(frame_idx) in set(evidence_frames):
                                continue
                            temporal_nvs.extend((int(frame_idx), int(c)) for c in cam_pool[:temporal_cams])
                    requested_buckets = {
                        "reconstruction": list(reconstruction),
                        "nvs_same_frame": list(same_frame_nvs),
                        "temporal_nvs": list(temporal_nvs),
                    }
                    segment_all_full = _dedupe(reconstruction + same_frame_nvs + temporal_nvs)
                    if not _enabled(segment_all_cfg, default=True):
                        segment_all_full = []
                    segment_all = segment_all_full[:max_render_refs]
                    target_set = set(segment_all)
                    reconstruction = [ref for ref in reconstruction if ref in target_set]
                    same_frame_nvs = [ref for ref in same_frame_nvs if ref in target_set]
                    temporal_nvs = [ref for ref in temporal_nvs if ref in target_set]
                    materialized_counts = {
                        "reconstruction": len(reconstruction),
                        "nvs_same_frame": len(same_frame_nvs),
                        "temporal_nvs": len(temporal_nvs),
                        "segment_all": len(segment_all),
                    }
                    requested_counts = {
                        "reconstruction": len(requested_buckets["reconstruction"]),
                        "nvs_same_frame": len(requested_buckets["nvs_same_frame"]),
                        "temporal_nvs": len(requested_buckets["temporal_nvs"]),
                        "segment_all": len(segment_all_full),
                    }
                    dropped_counts = {
                        key: int(requested_counts.get(key, 0) - materialized_counts.get(key, 0))
                        for key in requested_counts
                    }
                    target_refs = list(segment_all)
                    if not target_refs:
                        continue
                    role_by_ref: Dict[ImageRef, str] = {}
                    for ref in reconstruction:
                        role_by_ref[(int(ref[0]), int(ref[1]))] = "final_history_recon"
                    for ref in same_frame_nvs:
                        role_by_ref.setdefault((int(ref[0]), int(ref[1])), "final_history_nvs")
                    for ref in temporal_nvs:
                        role_by_ref.setdefault((int(ref[0]), int(ref[1])), "final_history_nvs")
                    target_roles = [role_by_ref.get(ref, "final_history_nvs") for ref in target_refs]
                    request_meta = {
                        "scheduler_version": "long_v1",
                        "scheduler_phase": PHASE_B_LONG_NAME,
                        "assembly_mode": "image_ref_long_v1",
                        "validation_version": "long_v1",
                        "validation_mode": "inference_only",
                        "validation_interval_T": int(T),
                        "validation_order": str(order),
                        "inner_K": int(len(visits)),
                        "shape_name": f"val_T{int(T)}_{order}",
                        "repeats_per_anchor": int(repeats),
                        "anchors_per_rollout": int(len(evidence_frames)),
                        "anchor_frames_chronological": [int(x) for x in evidence_frames],
                        "anchor_frames_rollout_order": [int(x) for x in ordered],
                        "visits": [visit.__dict__ for visit in visits],
                        "evidence_refs_by_step": [[tuple(x) for x in step] for step in evidence_refs_by_step],
                        "source_image_refs": _dedupe(evidence_refs_used),
                        "target_image_refs": [tuple(x) for x in target_refs],
                        "target_image_roles": list(target_roles),
                        "required_final_roles": [],
                        "final_history_recon_refs": [ref for ref, role in zip(target_refs, target_roles) if role == "final_history_recon"],
                        "final_history_nvs_refs": [ref for ref, role in zip(target_refs, target_roles) if role == "final_history_nvs"],
                        "final_current_recon_refs": [ref for ref, role in zip(target_refs, target_roles) if role == "final_current_recon"],
                        "final_current_nvs_refs": [ref for ref, role in zip(target_refs, target_roles) if role == "final_current_nvs"],
                        "query_label_refs": [],
                        "prefix_loss_refs_by_step": [[] for _ in visits],
                        "nearby_loss_refs_by_step": [[] for _ in visits],
                        "block_loss_refs_by_step": [[] for _ in visits],
                        "rigid_meta": dict(_cfg_get(raw, "rigid_meta", {}) or {}),
                        "distant_meta": dict(_cfg_get(raw, "distant_meta", {"mode": "appearance_scale_only"}) or {}),
                        "validation_buckets": {
                            "reconstruction": reconstruction,
                            "nvs_same_frame": same_frame_nvs,
                            "temporal_nvs": temporal_nvs,
                            "segment_all": segment_all,
                        },
                        "validation_bucket_requested_counts": dict(requested_counts),
                        "validation_bucket_materialized_counts": dict(materialized_counts),
                        "validation_bucket_dropped_counts": dict(dropped_counts),
                    }
                    specs.append(
                        LongPhaseBValidationSpec(
                            scene_id=int(scene_id),
                            segment_id=int(segment_id),
                            interval_T=int(T),
                            order=str(order),
                            frame_pool=[int(x) for x in frames],
                            evidence_frames=[int(x) for x in evidence_frames],
                            visits=visits,
                            evidence_refs_by_step=evidence_refs_by_step,
                            target_image_refs=target_refs,
                            target_image_roles=target_roles,
                            validation_buckets=dict(request_meta["validation_buckets"]),
                            request_meta=request_meta,
                        )
                    )
    return LongPhaseBValidationPlan(specs=specs, interval_T_values=interval_T_values, orders=orders)


def materialize_validation_long_phase_b_batch(dataset: Any, spec: LongPhaseBValidationSpec, *, include_test: bool = False) -> Dict[str, Any]:
    meta = dict(spec.request_meta)
    batch = dataset._assemble_segment_batch_from_image_refs(
        int(spec.scene_id),
        int(spec.segment_id),
        [tuple(x) for x in meta["source_image_refs"]],
        [tuple(x) for x in meta["target_image_refs"]],
        aux_image_refs=None,
        query_label_image_refs=[],
        include_test=bool(include_test),
        test_image_refs=None,
        enforce_target0_equals_source=False,
        target_ref_purpose="train",
    )
    req = dict(batch.get("request_meta") or {})
    req.update(meta)
    batch["request_meta"] = req
    batch["_scheduler_long_phase_b"] = dict(req)
    return batch


__all__ = [
    "LongPhaseBValidationPlan",
    "LongPhaseBValidationSpec",
    "build_validation_plan_long_phase_b",
    "materialize_validation_long_phase_b_batch",
]
