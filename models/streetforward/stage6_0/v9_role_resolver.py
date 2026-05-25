from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

ImageRef = Tuple[int, int]

PHASE_A_NAME = "phase_A_block_local_unroll"
PHASE_B_NAME = "phase_B_viewset_rollout"


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


@dataclass(frozen=True)
class ResolvedV9PhaseBBatch:
    inner_K: int
    evidence_refs_by_step: List[List[ImageRef]]
    prefix_loss_refs_by_step: List[List[ImageRef]]
    query_label_refs: List[ImageRef]
    memory_write_flags_by_step: List[bool]
    step_block_indices: List[int]
    step_repeat_indices: List[int]
    step_source_frame_indices: List[int]
    source_index_by_ref: Dict[ImageRef, int]
    target_index_by_ref: Dict[ImageRef, int]
    query_index_by_ref: Dict[ImageRef, int]
    evidence_source_indices_by_step: List[List[int]]
    prefix_target_indices_by_step: List[List[int]]
    query_target_indices: List[int]
    request_meta: Dict[str, Any]
    phase_b_repeat: Dict[str, Any]
    tbptt_meta: Dict[str, Any]


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


def _flat(groups: Iterable[Iterable[ImageRef]]) -> List[ImageRef]:
    return [tuple(x) for group in groups for x in group]


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
    return [(int(f), int(c)) for f, c in zip(frame_vals, cam_vals)]


def _dict_ref_list(items: Any) -> Optional[List[ImageRef]]:
    if not isinstance(items, list) or not items:
        return None
    out: List[ImageRef] = []
    for item in items:
        if not isinstance(item, dict) or "frame_idx" not in item or "cam_idx" not in item:
            return None
        out.append((int(item["frame_idx"]), int(item["cam_idx"])))
    return out


def _require_order_matches(actual: Optional[List[ImageRef]], expected: List[ImageRef], name: str) -> None:
    if actual is None:
        return
    if len(actual) != len(expected):
        raise ValueError(f"{name} length mismatch: {len(actual)} vs {len(expected)}")
    for idx, (a, e) in enumerate(zip(actual, expected)):
        if tuple(a) != tuple(e):
            raise ValueError(f"{name} order/content mismatch at index {idx}: got {tuple(a)} expected {tuple(e)}")


def _require_policy(meta: Dict[str, Any], role: str, expected: str) -> None:
    policy = dict(meta.get("role_policy") or {})
    actual = str(policy.get(role, ""))
    if actual != expected:
        raise ValueError(f"Phase B requires role_policy[{role!r}]={expected!r}, got {actual!r}")


def _role_group(meta: Dict[str, Any], role: str) -> Optional[Dict[str, Any]]:
    for group in list(meta.get("role_groups") or []):
        if str(group.get("role", "")) == role:
            return dict(group)
    return None


def _require_role_group_flag(meta: Dict[str, Any], role: str, flag: str, expected: bool) -> None:
    group = _role_group(meta, role)
    if group is None:
        raise ValueError(f"Phase B requires role_groups entry for {role!r}")
    actual = bool(group.get(flag, False))
    if actual is not bool(expected):
        raise ValueError(f"Phase B requires role_groups[{role!r}].{flag}={bool(expected)}, got {actual}")


def _phase_b_repeat_metadata(
    meta: Dict[str, Any],
    inner_K: int,
) -> tuple[Dict[str, Any], List[bool], List[int], List[int], List[int]]:
    rollout_mode = str(meta.get("phase_b_rollout_mode", ""))
    if rollout_mode != "episode_grouped_repeat_tbptt":
        return (
            {},
            [True for _ in range(int(inner_K))],
            [-1 for _ in range(int(inner_K))],
            [0 for _ in range(int(inner_K))],
            [],
        )

    repeat_meta = dict(meta.get("phase_b_repeat") or {})
    if not repeat_meta:
        raise ValueError("episode_grouped_repeat_tbptt requires request_meta.phase_b_repeat.")
    if str(repeat_meta.get("mode", "")) != "episode_grouped_repeat_tbptt":
        raise ValueError("phase_b_repeat.mode must be episode_grouped_repeat_tbptt.")
    step_blocks = [int(x) for x in list(repeat_meta.get("step_block_indices", []) or [])]
    step_repeats = [int(x) for x in list(repeat_meta.get("step_repeat_indices", []) or [])]
    step_sources = [int(x) for x in list(repeat_meta.get("step_source_frame_indices", []) or [])]
    memory_flags = [bool(x) for x in list(repeat_meta.get("step_memory_write_flags", []) or [])]
    for name, vals in (
        ("step_block_indices", step_blocks),
        ("step_repeat_indices", step_repeats),
        ("step_source_frame_indices", step_sources),
        ("step_memory_write_flags", memory_flags),
    ):
        if len(vals) != int(inner_K):
            raise ValueError(f"phase_b_repeat.{name} length must equal inner_K.")

    repeats_per_block = int(repeat_meta.get("repeats_per_block", 0) or 0)
    if repeats_per_block < 1:
        raise ValueError("phase_b_repeat.repeats_per_block must be >= 1.")
    unique_blocks = [int(x) for x in list(repeat_meta.get("unique_event_block_indices", []) or [])]
    if not unique_blocks:
        raise ValueError("phase_b_repeat.unique_event_block_indices must be non-empty.")
    for block in unique_blocks:
        positions = [idx for idx, val in enumerate(step_blocks) if int(val) == int(block)]
        if len(positions) != int(repeats_per_block):
            raise ValueError("each grouped repeat block must have repeats_per_block steps.")
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError("grouped repeat block steps must be contiguous.")
        if len({int(step_sources[idx]) for idx in positions}) != 1:
            raise ValueError("grouped repeat source frame must be fixed within each block.")
        if [int(step_repeats[idx]) for idx in positions] != list(range(int(repeats_per_block))):
            raise ValueError("grouped repeat indices must be [0, ..., repeats_per_block-1] per block.")
        expected_flags = [True] + [False for _ in range(int(repeats_per_block) - 1)]
        if [bool(memory_flags[idx]) for idx in positions] != expected_flags:
            raise ValueError("grouped repeat requires exactly one first-repeat memory write per block.")
    return repeat_meta, memory_flags, step_blocks, step_repeats, step_sources


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


def resolve_v9_phase_b_batch(
    batch: Dict[str, Any],
    *,
    written_refs: Optional[Iterable[ImageRef]] = None,
) -> ResolvedV9PhaseBBatch:
    meta = dict(batch.get("request_meta") or {})
    sched = dict(batch.get("_scheduler_v9") or {})

    scheduler_version = str(_first_present(meta, sched, "scheduler_version", ""))
    if scheduler_version != "v9":
        raise ValueError("Stage6_0 Phase B requires scheduler_v9 batch.")

    phase = str(_first_present(meta, sched, "scheduler_phase", sched.get("phase", "")))
    if phase != PHASE_B_NAME:
        raise ValueError("Stage6_0 Phase B requires scheduler_v9.phase=phase_B_viewset_rollout.")

    assembly_mode = str(meta.get("assembly_mode", ""))
    if assembly_mode != "image_ref_v9":
        raise ValueError("Stage6_0 Phase B requires request_meta.assembly_mode=image_ref_v9.")

    inner_K = int(_first_present(meta, sched, "inner_K", 0) or 0)
    if inner_K < 1:
        raise ValueError("Stage6_0 Phase B requires inner_K >= 1.")

    evidence_by_step = _as_ref_steps(_first_present(meta, sched, "evidence_refs_by_step"), "evidence_refs_by_step")
    block_by_step = _as_ref_steps(_first_present(meta, sched, "block_loss_refs_by_step"), "block_loss_refs_by_step")
    nearby_by_step = _as_ref_steps(_first_present(meta, sched, "nearby_loss_refs_by_step"), "nearby_loss_refs_by_step")
    prefix_by_step = _as_ref_steps(_first_present(meta, sched, "prefix_loss_refs_by_step"), "prefix_loss_refs_by_step")
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
        raise ValueError("Each Phase B rollout step requires non-empty evidence refs.")
    if any(len(x) == 0 for x in prefix_by_step):
        raise ValueError("Each Phase B rollout step requires non-empty prefix_loss refs.")
    if any(len(x) > 0 for x in nearby_by_step):
        raise ValueError("Phase B must not receive nearby_loss_refs.")
    if any(len(x) > 0 for x in block_by_step):
        raise ValueError("Phase B must not receive block_loss_refs.")
    if len(query_refs) == 0:
        raise ValueError("Phase B requires non-empty query_label_refs.")

    _require_policy(meta, "evidence", "update_only")
    _require_policy(meta, "prefix_loss", "loss_only")
    _require_policy(meta, "query_label", "label_only")
    _require_role_group_flag(meta, "evidence", "allow_update_evidence", True)
    _require_role_group_flag(meta, "evidence", "allow_render_loss", False)
    _require_role_group_flag(meta, "prefix_loss", "allow_render_loss", True)
    _require_role_group_flag(meta, "prefix_loss", "allow_update_evidence", False)
    _require_role_group_flag(meta, "query_label", "allow_query_label", True)
    _require_role_group_flag(meta, "query_label", "allow_update_evidence", False)
    (
        phase_b_repeat,
        memory_write_flags,
        step_block_indices,
        step_repeat_indices,
        step_source_frame_indices,
    ) = _phase_b_repeat_metadata(meta, int(inner_K))
    tbptt_meta = dict(meta.get("tbptt") or {})

    source_refs = [_as_ref(x) for x in list(meta.get("source_image_refs") or [])]
    target_refs = [_as_ref(x) for x in list(meta.get("target_image_refs") or [])]
    target_roles = [str(x) for x in list(meta.get("target_image_roles") or [])]
    if len(source_refs) == 0:
        raise ValueError("Stage6_0 Phase B requires non-empty source_image_refs.")
    if len(target_refs) == 0:
        raise ValueError("Stage6_0 Phase B requires non-empty target_image_refs.")
    if len(target_refs) != len(target_roles):
        raise ValueError("target_image_refs and target_image_roles length mismatch.")
    observed_roles = set(target_roles)
    if observed_roles != {"prefix_loss"}:
        raise ValueError(f"Phase B target roles must be only {{'prefix_loss'}}, got {observed_roles}.")

    evidence_flat = _flat(evidence_by_step)
    if len(step_source_frame_indices) != int(inner_K):
        step_source_frame_indices = [int(group[0][0]) for group in evidence_by_step]
    prefix_flat = _flat(prefix_by_step)
    flat_evidence_meta = [_as_ref(x) for x in list(meta.get("flat_evidence_refs") or source_refs)]
    flat_render_meta = [_as_ref(x) for x in list(meta.get("flat_render_loss_refs") or target_refs)]
    if set(flat_evidence_meta) != set(source_refs) or set(flat_evidence_meta) != set(evidence_flat):
        raise ValueError("Phase B source_image_refs must equal flat evidence refs.")
    if set(flat_render_meta) != set(target_refs) or set(flat_render_meta) != set(prefix_flat):
        raise ValueError("Phase B target_image_refs must equal flat prefix render refs.")

    source_set = set(source_refs)
    target_set = set(target_refs)
    evidence_set = set(evidence_flat)
    query_set = set(query_refs)
    if query_set & evidence_set:
        raise ValueError("query_label_refs leaked into evidence_refs.")
    if query_set & source_set:
        raise ValueError("query_label_refs must not appear in source_image_refs.")
    if query_set & target_set:
        raise ValueError("query_label_refs must not appear in target_image_refs.")
    written_set = {(_as_ref(x)) for x in list(written_refs or [])}
    if query_set & written_set:
        raise ValueError("query_label_refs already written into persistent VSM in this episode.")

    query_frames = {int(ref[0]) for ref in query_refs}
    if len(query_frames) != 1:
        raise ValueError("Phase B P0 query observation supports exactly one query frame per rollout.")
    num_cams = int(meta.get("num_cams", 0) or 0)
    if num_cams > 0 and len(query_refs) != int(num_cams):
        raise ValueError(
            f"Phase B P0 query observation expects all cams for one frame: got {len(query_refs)} refs num_cams={num_cams}"
        )

    _require_order_matches(_tensor_ref_list(batch.get("source")), source_refs, "source")
    _require_order_matches(_dict_ref_list(batch.get("source_views")), source_refs, "source_views")
    _require_order_matches(_tensor_ref_list(batch.get("target")), target_refs, "target")
    _require_order_matches(_dict_ref_list(batch.get("targets")), target_refs, "targets")
    _require_order_matches(_tensor_ref_list(batch.get("query_label")), query_refs, "query_label")
    _require_order_matches(_dict_ref_list(batch.get("query_targets")), query_refs, "query_targets")

    source_index_by_ref = {ref: i for i, ref in enumerate(source_refs)}
    target_index_by_ref = {ref: i for i, ref in enumerate(target_refs)}
    query_index_by_ref = {ref: i for i, ref in enumerate(query_refs)}

    def lookup_source(refs: List[ImageRef]) -> List[int]:
        out: List[int] = []
        for ref in refs:
            if ref not in source_index_by_ref:
                raise ValueError(f"evidence ref {ref} missing from source_image_refs")
            out.append(int(source_index_by_ref[ref]))
        return out

    def lookup_prefix(refs: List[ImageRef]) -> List[int]:
        out: List[int] = []
        for ref in refs:
            if ref not in target_index_by_ref:
                raise ValueError(f"prefix_loss ref {ref} missing from target_image_refs")
            idx = int(target_index_by_ref[ref])
            actual_role = str(target_roles[idx])
            if actual_role != "prefix_loss":
                raise ValueError(
                    f"prefix_loss ref {ref} mapped to target role {actual_role!r}, expected 'prefix_loss'"
                )
            out.append(idx)
        return out

    num_query_targets = len(list(batch.get("query_targets") or []))
    if num_query_targets and int(num_query_targets) != len(query_refs):
        raise ValueError(f"query_targets/query_label_refs length mismatch: {num_query_targets} vs {len(query_refs)}")
    query_indices = [int(query_index_by_ref[ref]) for ref in query_refs]

    return ResolvedV9PhaseBBatch(
        inner_K=int(inner_K),
        evidence_refs_by_step=evidence_by_step,
        prefix_loss_refs_by_step=prefix_by_step,
        query_label_refs=query_refs,
        memory_write_flags_by_step=[bool(x) for x in memory_write_flags],
        step_block_indices=[int(x) for x in step_block_indices],
        step_repeat_indices=[int(x) for x in step_repeat_indices],
        step_source_frame_indices=[int(x) for x in step_source_frame_indices],
        source_index_by_ref=source_index_by_ref,
        target_index_by_ref=target_index_by_ref,
        query_index_by_ref=query_index_by_ref,
        evidence_source_indices_by_step=[lookup_source(x) for x in evidence_by_step],
        prefix_target_indices_by_step=[lookup_prefix(x) for x in prefix_by_step],
        query_target_indices=query_indices,
        request_meta=meta,
        phase_b_repeat=phase_b_repeat,
        tbptt_meta=tbptt_meta,
    )
