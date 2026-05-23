from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

ImageRef = Tuple[int, int]

PHASE_A_NAME = "phase_A_block_local_unroll"


@dataclass(frozen=True)
class ResolvedV9PhaseABatch:
    inner_K: int
    evidence_refs_by_step: List[List[ImageRef]]
    block_loss_refs_by_step: List[List[ImageRef]]
    nearby_loss_refs_by_step: List[List[ImageRef]]
    source_index_by_ref: Dict[ImageRef, int]
    target_index_by_ref: Dict[ImageRef, int]
    evidence_source_indices_by_step: List[List[int]]
    block_target_indices_by_step: List[List[int]]
    nearby_target_indices_by_step: List[List[int]]


def _as_ref(raw: Any) -> ImageRef:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"image ref must be a length-2 tuple/list, got {raw!r}")
    return (int(raw[0]), int(raw[1]))


def _as_ref_steps(raw: Any, name: str) -> List[List[ImageRef]]:
    if raw is None:
        raise ValueError(f"{name} is required")
    out: List[List[ImageRef]] = []
    for group in list(raw):
        out.append([_as_ref(x) for x in list(group)])
    return out


def _first_present(meta: Dict[str, Any], sched: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in meta and meta[key] is not None:
        return meta[key]
    return sched.get(key, default)


def resolve_v9_phase_a_batch(batch: Dict[str, Any]) -> ResolvedV9PhaseABatch:
    meta = dict(batch.get("request_meta") or {})
    sched = dict(batch.get("_scheduler_v9") or {})

    scheduler_version = str(_first_present(meta, sched, "scheduler_version", ""))
    if scheduler_version != "v9":
        raise ValueError("Stage6_0 Phase A requires scheduler_v9 batch.")

    phase = str(_first_present(meta, sched, "scheduler_phase", sched.get("phase", "")))
    if phase != PHASE_A_NAME:
        raise ValueError("Stage6_0 Phase A requires scheduler_v9.phase=phase_A_block_local_unroll.")

    assembly_mode = str(meta.get("assembly_mode", ""))
    if assembly_mode != "image_ref_v9":
        raise ValueError("Stage6_0 Phase A requires request_meta.assembly_mode=image_ref_v9.")

    inner_K = int(_first_present(meta, sched, "inner_K", 0) or 0)
    if inner_K < 1:
        raise ValueError("Stage6_0 Phase A requires inner_K >= 1.")

    evidence_by_step = _as_ref_steps(_first_present(meta, sched, "evidence_refs_by_step"), "evidence_refs_by_step")
    block_by_step = _as_ref_steps(_first_present(meta, sched, "block_loss_refs_by_step"), "block_loss_refs_by_step")
    nearby_by_step = _as_ref_steps(_first_present(meta, sched, "nearby_loss_refs_by_step"), "nearby_loss_refs_by_step")
    prefix_by_step = _as_ref_steps(
        _first_present(meta, sched, "prefix_loss_refs_by_step", [[] for _ in range(inner_K)]),
        "prefix_loss_refs_by_step",
    )
    query_refs = [_as_ref(x) for x in list(_first_present(meta, sched, "query_label_refs", []) or [])]

    for name, groups in (
        ("evidence_refs_by_step", evidence_by_step),
        ("block_loss_refs_by_step", block_by_step),
        ("nearby_loss_refs_by_step", nearby_by_step),
        ("prefix_loss_refs_by_step", prefix_by_step),
    ):
        if len(groups) != inner_K:
            raise ValueError(f"{name} length must equal inner_K.")
    if any(len(x) == 0 for x in evidence_by_step):
        raise ValueError("Each Phase A unroll step requires non-empty evidence refs.")
    if any(len(x) > 0 for x in prefix_by_step):
        raise ValueError("Phase A must not receive prefix_loss_refs.")
    if len(query_refs) > 0:
        raise ValueError("Phase A must not receive query_label_refs.")

    source_refs = [_as_ref(x) for x in list(meta.get("source_image_refs") or [])]
    target_refs = [_as_ref(x) for x in list(meta.get("target_image_refs") or [])]
    target_roles = [str(x) for x in list(meta.get("target_image_roles") or [])]
    if len(source_refs) == 0:
        raise ValueError("Stage6_0 Phase A requires non-empty source_image_refs.")
    if len(target_refs) == 0:
        raise ValueError("Stage6_0 Phase A requires non-empty target_image_refs.")
    if len(target_refs) != len(target_roles):
        raise ValueError("target_image_refs and target_image_roles length mismatch.")
    allowed_roles = {"block_loss", "nearby_loss"}
    observed_roles = set(target_roles)
    if not observed_roles.issubset(allowed_roles):
        raise ValueError(f"Phase A target roles must be {allowed_roles}, got {observed_roles}.")

    source_index_by_ref = {ref: i for i, ref in enumerate(source_refs)}
    target_index_by_ref = {ref: i for i, ref in enumerate(target_refs)}

    def lookup_source(refs: List[ImageRef]) -> List[int]:
        out: List[int] = []
        for ref in refs:
            if ref not in source_index_by_ref:
                raise ValueError(f"evidence ref {ref} missing from source_image_refs")
            out.append(int(source_index_by_ref[ref]))
        return out

    def lookup_target(refs: List[ImageRef], role_name: str) -> List[int]:
        out: List[int] = []
        for ref in refs:
            if ref not in target_index_by_ref:
                raise ValueError(f"{role_name} ref {ref} missing from target_image_refs")
            idx = int(target_index_by_ref[ref])
            actual_role = str(target_roles[idx])
            if actual_role != role_name:
                raise ValueError(
                    f"{role_name} ref {ref} mapped to target role {actual_role!r}, expected {role_name!r}"
                )
            out.append(idx)
        return out

    evidence = {ref for group in evidence_by_step for ref in group}
    nearby = {ref for group in nearby_by_step for ref in group}
    if evidence & nearby:
        raise ValueError("nearby_loss_refs leaked into evidence_refs.")

    return ResolvedV9PhaseABatch(
        inner_K=int(inner_K),
        evidence_refs_by_step=evidence_by_step,
        block_loss_refs_by_step=block_by_step,
        nearby_loss_refs_by_step=nearby_by_step,
        source_index_by_ref=source_index_by_ref,
        target_index_by_ref=target_index_by_ref,
        evidence_source_indices_by_step=[lookup_source(x) for x in evidence_by_step],
        block_target_indices_by_step=[lookup_target(x, "block_loss") for x in block_by_step],
        nearby_target_indices_by_step=[lookup_target(x, "nearby_loss") for x in nearby_by_step],
    )

