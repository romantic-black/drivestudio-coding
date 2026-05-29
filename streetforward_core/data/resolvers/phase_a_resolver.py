from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from streetforward_core.data.schedulers.legacy_v9_phase_a_adapter import legacy_v9_batch_to_phase_a_plan
from streetforward_core.protocols.batch import RawBatch, ResolvedPhaseABatch
from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.roles import Role
from streetforward_core.protocols.rollout import PhaseALocalUnrollPlan, phase_a_plan_from_mapping
from streetforward_core.protocols.validators import validate_phase_a_plan


def _as_plan(raw: Any) -> PhaseALocalUnrollPlan:
    if isinstance(raw, PhaseALocalUnrollPlan):
        return raw
    if isinstance(raw, Mapping):
        return phase_a_plan_from_mapping(raw)
    raise ValueError(f"unsupported Phase A rollout_plan type: {type(raw)!r}")


def _ref_list(raw: Any, name: str) -> List[ImageRef]:
    if raw is None:
        raise ValueError(f"{name} is required")
    return [ImageRef.from_raw(ref) for ref in list(raw)]


def _tensor_ref_list(role_batch: Any) -> Optional[List[ImageRef]]:
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
    return [ImageRef(int(f), int(c)) for f, c in zip(frame_vals, cam_vals)]


def _require_order_matches(actual: Optional[List[ImageRef]], expected: List[ImageRef], name: str) -> None:
    if actual is None:
        return
    if len(actual) != len(expected):
        raise ValueError(f"{name} length mismatch: {len(actual)} vs {len(expected)}")
    for idx, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            raise ValueError(f"{name} order/content mismatch at index {idx}: got {got.as_tuple()} expected {want.as_tuple()}")


def _flat(groups: Iterable[Iterable[ImageRef]]) -> List[ImageRef]:
    return [ref for group in groups for ref in group]


def _meta_ref_groups(meta: Dict[str, Any], sched: Any, key: str) -> Optional[Tuple[Tuple[ImageRef, ...], ...]]:
    raw = meta.get(key)
    if raw is None:
        raw = sched.get(key) if isinstance(sched, dict) else getattr(sched, key, None)
    if raw is None:
        return None
    return tuple(tuple(ImageRef.from_raw(ref) for ref in list(group)) for group in list(raw))


def _meta_scalar(meta: Dict[str, Any], sched: Any, key: str) -> Any:
    if key in meta and meta[key] is not None:
        return meta[key]
    if isinstance(sched, dict):
        return sched.get(key)
    return getattr(sched, key, None)


def _assert_plan_matches_request_meta(plan: PhaseALocalUnrollPlan, batch: RawBatch) -> None:
    meta = dict(batch.get("request_meta") or {})
    sched = batch.get("_scheduler_v9") or {}
    if not meta and not sched:
        return

    inner_k = _meta_scalar(meta, sched, "inner_K")
    if inner_k is not None and int(inner_k) != int(plan.inner_K):
        raise ValueError("rollout_plan.inner_K disagrees with request_meta.inner_K")

    expected_len = int(plan.inner_K)
    for key, attr, label in (
        ("evidence_refs_by_step", "evidence_refs", "evidence"),
        ("block_loss_refs_by_step", "block_loss_refs", "block_loss"),
        ("nearby_loss_refs_by_step", "nearby_loss_refs", "nearby_loss"),
    ):
        groups = _meta_ref_groups(meta, sched, key)
        if groups is None:
            continue
        if len(groups) != expected_len:
            raise ValueError(f"rollout_plan/request_meta {label} step count mismatch: {expected_len} vs {len(groups)}")
        for idx, (step, meta_refs) in enumerate(zip(plan.steps, groups)):
            plan_refs = tuple(getattr(step, attr))
            if tuple(plan_refs) != tuple(meta_refs):
                raise ValueError(f"rollout_plan/request_meta {label} mismatch at k={idx}")


class PhaseABatchResolver:
    def resolve(self, batch: RawBatch) -> ResolvedPhaseABatch:
        plan_raw = batch.get("rollout_plan")
        plan = _as_plan(plan_raw) if plan_raw is not None else legacy_v9_batch_to_phase_a_plan(batch)
        validate_phase_a_plan(plan)
        if plan_raw is not None:
            _assert_plan_matches_request_meta(plan, batch)

        meta = dict(batch.get("request_meta") or {})
        assembly_mode = str(meta.get("assembly_mode", ""))
        if assembly_mode and assembly_mode != "image_ref_v9":
            raise ValueError("Phase A requires request_meta.assembly_mode=image_ref_v9.")

        source_refs = _ref_list(meta.get("source_image_refs"), "source_image_refs")
        target_refs = _ref_list(meta.get("target_image_refs"), "target_image_refs")
        target_roles = [str(x) for x in list(meta.get("target_image_roles") or [])]
        if len(source_refs) == 0:
            raise ValueError("Phase A requires non-empty source_image_refs.")
        if len(target_refs) == 0:
            raise ValueError("Phase A requires non-empty target_image_refs.")
        if len(target_refs) != len(target_roles):
            raise ValueError("target_image_refs and target_image_roles length mismatch.")
        allowed_roles = {Role.BLOCK_LOSS.value, Role.NEARBY_LOSS.value}
        observed_roles = set(target_roles)
        if not observed_roles.issubset(allowed_roles):
            raise ValueError(f"Phase A target roles must be {allowed_roles}, got {observed_roles}.")

        _require_order_matches(_tensor_ref_list(batch.get("source")), source_refs, "source_image_refs")
        _require_order_matches(_tensor_ref_list(batch.get("target")), target_refs, "target_image_refs")

        source_index_by_ref = {ref: idx for idx, ref in enumerate(source_refs)}
        target_index_by_ref = {ref: idx for idx, ref in enumerate(target_refs)}

        def lookup_source(refs: Tuple[ImageRef, ...]) -> Tuple[int, ...]:
            out = []
            for ref in refs:
                if ref not in source_index_by_ref:
                    raise ValueError(f"evidence ref {ref.as_tuple()} missing from source_image_refs")
                out.append(int(source_index_by_ref[ref]))
            return tuple(out)

        def lookup_target(refs: Tuple[ImageRef, ...], role_name: str) -> Tuple[int, ...]:
            out = []
            for ref in refs:
                if ref not in target_index_by_ref:
                    raise ValueError(f"{role_name} ref {ref.as_tuple()} missing from target_image_refs")
                idx = int(target_index_by_ref[ref])
                actual_role = str(target_roles[idx])
                if actual_role != role_name:
                    raise ValueError(
                        f"{role_name} ref {ref.as_tuple()} mapped to target role {actual_role!r}, expected {role_name!r}"
                    )
                out.append(idx)
            return tuple(out)

        evidence = set(_flat(step.evidence_refs for step in plan.steps))
        nearby = set(_flat(step.nearby_loss_refs for step in plan.steps))
        if evidence & nearby:
            raise ValueError("nearby_loss_refs leaked into evidence_refs.")

        return ResolvedPhaseABatch(
            raw=batch,
            plan=plan,
            source_index_by_ref=source_index_by_ref,
            target_index_by_ref=target_index_by_ref,
            evidence_source_indices_by_step=tuple(lookup_source(step.evidence_refs) for step in plan.steps),
            block_target_indices_by_step=tuple(
                lookup_target(step.block_loss_refs, Role.BLOCK_LOSS.value) for step in plan.steps
            ),
            nearby_target_indices_by_step=tuple(
                lookup_target(step.nearby_loss_refs, Role.NEARBY_LOSS.value) for step in plan.steps
            ),
        )
