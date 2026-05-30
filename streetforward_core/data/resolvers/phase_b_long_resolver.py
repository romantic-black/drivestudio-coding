from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from streetforward_core.protocols.phase_b_long import (
    LONG_TARGET_ROLES,
    PhaseBLongRolloutPlan,
    ResolvedLongPhaseBBatch,
    phase_b_long_plan_from_mapping,
)
from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.roles import LongRole
from streetforward_core.protocols.validators import validate_phase_b_long_plan


def _as_plan(raw: Any) -> PhaseBLongRolloutPlan:
    if isinstance(raw, PhaseBLongRolloutPlan):
        return raw
    if isinstance(raw, Mapping):
        return phase_b_long_plan_from_mapping(raw)
    if raw is not None and all(hasattr(raw, name) for name in ("scheduler_version", "phase", "visits")):
        return phase_b_long_plan_from_mapping(raw.__dict__)
    raise ValueError(f"unsupported Phase B Long rollout_plan type: {type(raw)!r}")


def _refs(raw: Any) -> Tuple[ImageRef, ...]:
    return tuple(ImageRef.from_raw(ref) for ref in list(raw or []))


def _ref_groups(raw: Any) -> Optional[Tuple[Tuple[ImageRef, ...], ...]]:
    if raw is None:
        return None
    return tuple(tuple(ImageRef.from_raw(ref) for ref in list(group)) for group in list(raw or []))


def _tensor_ref_list(role_batch: Any) -> Optional[Tuple[ImageRef, ...]]:
    if not isinstance(role_batch, dict):
        return None
    frames = role_batch.get("frame_indices")
    cams = role_batch.get("cam_indices")
    if frames is None or cams is None:
        return None
    frame_vals = frames.detach().cpu().reshape(-1).tolist() if hasattr(frames, "detach") else list(frames)
    cam_vals = cams.detach().cpu().reshape(-1).tolist() if hasattr(cams, "detach") else list(cams)
    if len(frame_vals) != len(cam_vals):
        raise ValueError("role batch frame_indices/cam_indices length mismatch.")
    return tuple(ImageRef(int(f), int(c)) for f, c in zip(frame_vals, cam_vals))


def _dict_ref_list(items: Any) -> Optional[Tuple[ImageRef, ...]]:
    if not isinstance(items, list) or not items:
        return None
    out = []
    for item in items:
        if not isinstance(item, dict) or "frame_idx" not in item or "cam_idx" not in item:
            return None
        out.append(ImageRef(int(item["frame_idx"]), int(item["cam_idx"])))
    return tuple(out)


def _require_order_matches(actual: Optional[Tuple[ImageRef, ...]], expected: Tuple[ImageRef, ...], name: str) -> None:
    if actual is None:
        return
    if len(actual) != len(expected):
        raise ValueError(f"{name} length mismatch: {len(actual)} vs {len(expected)}")
    for idx, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            raise ValueError(f"{name} order/content mismatch at index {idx}: got {got.as_tuple()} expected {want.as_tuple()}")


def _meta_plan_from_batch(batch: Dict[str, Any]) -> Optional[PhaseBLongRolloutPlan]:
    meta = dict(batch.get("request_meta") or {})
    sched = dict(batch.get("_scheduler_long_phase_b") or {})
    if not meta and not sched:
        return None
    merged = dict(sched)
    merged.update(meta)
    if not merged:
        return None
    return phase_b_long_plan_from_mapping(merged)


def _assert_plan_matches_meta(plan: PhaseBLongRolloutPlan, batch: Dict[str, Any]) -> None:
    meta_plan = _meta_plan_from_batch(batch)
    if meta_plan is None:
        return
    if int(plan.inner_K) != int(meta_plan.inner_K):
        raise ValueError("rollout_plan.inner_K disagrees with request_meta.inner_K")
    for key, a, b in (
        ("evidence_refs_by_step", plan.evidence_refs_by_step, meta_plan.evidence_refs_by_step),
        ("source_image_refs", plan.source_image_refs, meta_plan.source_image_refs),
        ("target_image_refs", plan.target_image_refs, meta_plan.target_image_refs),
        ("target_image_roles", plan.target_image_roles, meta_plan.target_image_roles),
    ):
        if tuple(a) != tuple(b):
            raise ValueError(f"rollout_plan/request_meta {key} mismatch")


def _lookup_refs(refs: Iterable[ImageRef], index_by_ref: Dict[ImageRef, int], *, label: str) -> Tuple[int, ...]:
    out = []
    for ref in refs:
        if ref not in index_by_ref:
            raise ValueError(f"{label} ref {ref.as_tuple()} missing from image refs")
        out.append(int(index_by_ref[ref]))
    return tuple(out)


class PhaseBLongBatchResolver:
    def resolve(self, batch: Dict[str, Any]) -> ResolvedLongPhaseBBatch:
        plan_raw = batch.get("rollout_plan")
        plan = _as_plan(plan_raw) if plan_raw is not None else _meta_plan_from_batch(batch)
        if plan is None:
            raise ValueError("Phase B Long requires rollout_plan or _scheduler_long_phase_b metadata")
        validate_phase_b_long_plan(plan)
        if plan_raw is not None:
            _assert_plan_matches_meta(plan, batch)

        meta = dict(batch.get("request_meta") or {})
        for key in ("prefix_loss_refs_by_step", "nearby_loss_refs_by_step", "block_loss_refs_by_step"):
            groups = _ref_groups(meta.get(key))
            if groups is not None and any(len(group) > 0 for group in groups):
                raise ValueError(f"Phase B Long requires empty {key}")
        if len(_refs(meta.get("query_label_refs"))) > 0:
            raise ValueError("Phase B Long requires empty query_label_refs")

        source_refs = tuple(plan.source_image_refs)
        target_refs = tuple(plan.target_image_refs)
        target_roles = tuple(plan.target_image_roles)
        if len(target_refs) != len(target_roles):
            raise ValueError("target_image_refs and target_image_roles length mismatch")
        observed = set(target_roles)
        if not observed.issubset(set(LONG_TARGET_ROLES)):
            if any(role.value in {"final_history", "final_current"} for role in observed):
                raise ValueError("Phase B Long rejects coarse final roles")
            raise ValueError(f"Phase B Long target roles must be split Long roles, got {observed}")

        _require_order_matches(_tensor_ref_list(batch.get("source")), source_refs, "source")
        _require_order_matches(_dict_ref_list(batch.get("source_views")), source_refs, "source_views")
        _require_order_matches(_tensor_ref_list(batch.get("target")), target_refs, "target")
        _require_order_matches(_dict_ref_list(batch.get("targets")), target_refs, "targets")

        source_index_by_ref = {ref: idx for idx, ref in enumerate(source_refs)}
        target_index_by_ref = {ref: idx for idx, ref in enumerate(target_refs)}
        target_role_by_ref = {ref: role for ref, role in zip(target_refs, target_roles)}

        def lookup_target(refs: Tuple[ImageRef, ...], role: LongRole) -> Tuple[int, ...]:
            out = []
            for ref in refs:
                if target_role_by_ref.get(ref) != role:
                    actual = target_role_by_ref.get(ref)
                    actual_s = actual.value if actual is not None else ""
                    raise ValueError(f"{role.value} ref {ref.as_tuple()} mapped to target role {actual_s!r}")
                out.extend(_lookup_refs((ref,), target_index_by_ref, label=role.value))
            return tuple(out)

        return ResolvedLongPhaseBBatch(
            raw=batch,
            plan=plan,
            source_index_by_ref=source_index_by_ref,
            target_index_by_ref=target_index_by_ref,
            evidence_source_indices_by_step=tuple(
                _lookup_refs(step, source_index_by_ref, label="evidence") for step in plan.evidence_refs_by_step
            ),
            final_history_recon_target_indices=lookup_target(plan.final_history_recon_refs, LongRole.FINAL_HISTORY_RECON),
            final_history_nvs_target_indices=lookup_target(plan.final_history_nvs_refs, LongRole.FINAL_HISTORY_NVS),
            final_current_recon_target_indices=lookup_target(plan.final_current_recon_refs, LongRole.FINAL_CURRENT_RECON),
            final_current_nvs_target_indices=lookup_target(plan.final_current_nvs_refs, LongRole.FINAL_CURRENT_NVS),
        )

