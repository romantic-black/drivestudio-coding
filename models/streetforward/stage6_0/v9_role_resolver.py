from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

ImageRef = Tuple[int, int]

PHASE_A_NAME = "phase_A_block_local_unroll"
PHASE_B_NAME = "phase_B_viewset_rollout"
PHASE_B_FINAL_TARGET_ROLES = {
    "final_history_recon",
    "final_current_recon",
    "final_history_nvs",
    "final_current_nvs",
}


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
    final_supervision_step_idx: int = -1
    final_target_indices_by_role: Optional[Dict[str, List[int]]] = None
    final_target_indices: Optional[List[int]] = None
    phase_b_rollout: Optional[Dict[str, Any]] = None
    final_supervision: Optional[Dict[str, Any]] = None
    visit_time_codes: Optional[List[Tuple[float, float, float, float]]] = None


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
    if rollout_mode not in ("episode_grouped_repeat_tbptt", "episode_rollout_grouped_repeat_tbptt"):
        return (
            {},
            [True for _ in range(int(inner_K))],
            [-1 for _ in range(int(inner_K))],
            [0 for _ in range(int(inner_K))],
            [],
        )

    repeat_meta = dict(meta.get("phase_b_repeat") or {})
    if not repeat_meta:
        raise ValueError(f"{rollout_mode} requires request_meta.phase_b_repeat.")
    if str(repeat_meta.get("mode", "")) != str(rollout_mode):
        raise ValueError(f"phase_b_repeat.mode must be {rollout_mode}.")
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
    unique_frames = [int(x) for x in list(repeat_meta.get("unique_event_frame_indices", []) or [])]
    if unique_frames and len(unique_frames) != len(unique_blocks):
        raise ValueError("phase_b_repeat unique block/frame metadata length mismatch.")
    actual_blocks = int(repeat_meta.get("actual_blocks_per_rollout", len(unique_blocks)) or 0)
    if int(actual_blocks) != len(unique_blocks):
        raise ValueError("phase_b_repeat.actual_blocks_per_rollout must equal unique event block count.")
    if int(inner_K) != int(actual_blocks) * int(repeats_per_block):
        raise ValueError("Phase B actual inner_K must equal actual_blocks_per_rollout * repeats_per_block.")
    if int(repeat_meta.get("actual_inner_K", inner_K) or 0) != int(inner_K):
        raise ValueError("phase_b_repeat.actual_inner_K must equal inner_K.")
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
    final_only = str(meta.get("phase_b_loss_timing", "")) == "rollout_final_only"

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
    if final_only:
        loss_steps = [idx for idx, refs in enumerate(prefix_by_step) if len(refs) > 0]
        if loss_steps != [int(inner_K) - 1]:
            raise ValueError("Phase B rollout_final_only requires render refs only on the final step.")
    elif any(len(x) == 0 for x in prefix_by_step):
        raise ValueError("Each Phase B rollout step requires non-empty prefix_loss refs.")
    if any(len(x) > 0 for x in nearby_by_step):
        raise ValueError("Phase B must not receive nearby_loss_refs.")
    if any(len(x) > 0 for x in block_by_step):
        raise ValueError("Phase B must not receive block_loss_refs.")
    if len(query_refs) == 0 and not final_only:
        raise ValueError("Phase B requires non-empty query_label_refs.")

    _require_policy(meta, "evidence", "update_only")
    if not final_only:
        _require_policy(meta, "prefix_loss", "loss_only")
        _require_policy(meta, "query_label", "label_only")
    else:
        for role in PHASE_B_FINAL_TARGET_ROLES:
            policy = dict(meta.get("role_policy") or {})
            if role in policy and str(policy.get(role, "")) != "loss_only":
                raise ValueError(f"Phase B final role {role!r} must be loss_only.")
    _require_role_group_flag(meta, "evidence", "allow_update_evidence", True)
    _require_role_group_flag(meta, "evidence", "allow_render_loss", False)
    if not final_only:
        _require_role_group_flag(meta, "prefix_loss", "allow_render_loss", True)
        _require_role_group_flag(meta, "prefix_loss", "allow_update_evidence", False)
        _require_role_group_flag(meta, "query_label", "allow_query_label", True)
        _require_role_group_flag(meta, "query_label", "allow_update_evidence", False)
    else:
        _require_role_group_flag(meta, "render_loss", "allow_render_loss", True)
        _require_role_group_flag(meta, "render_loss", "allow_update_evidence", False)
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
    if final_only:
        if not observed_roles.issubset(PHASE_B_FINAL_TARGET_ROLES):
            raise ValueError(f"Phase B final target roles must be final supervision roles, got {observed_roles}.")
        if "final_current_recon" not in observed_roles:
            raise ValueError("Phase B final rollout requires final_current_recon targets.")
    elif observed_roles != {"prefix_loss"}:
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
        expected = "final rollout render refs" if final_only else "flat prefix render refs"
        raise ValueError(f"Phase B target_image_refs must equal {expected}.")

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

    num_cams = int(meta.get("num_cams", 0) or 0)
    if final_only:
        if num_cams > 0:
            expected_cams = set(range(int(num_cams)))
            for step_refs in evidence_by_step:
                frames = {int(ref[0]) for ref in step_refs}
                cams = {int(ref[1]) for ref in step_refs}
                if len(frames) != 1 or cams != expected_cams:
                    raise ValueError("Phase B final rollout requires all-cams evidence for exactly one frame per step.")
        final_meta = dict(meta.get("phase_b_final_supervision") or {})
        final_refs = [_as_ref(x) for x in list(final_meta.get("refs", []) or [])]
        final_roles = [str(x) for x in list(final_meta.get("roles", []) or [])]
        if len(final_refs) != len(final_roles) or not final_refs:
            raise ValueError("Phase B final rollout requires phase_b_final_supervision refs/roles.")
        if set(final_refs) != set(target_refs):
            raise ValueError("Phase B final supervision refs must match target_image_refs.")
        nvs_refs = {ref for ref, role in zip(final_refs, final_roles) if str(role).endswith("_nvs")}
        if nvs_refs & evidence_set:
            raise ValueError("Phase B final NVS refs leaked into evidence_refs.")
        rollout_meta = dict(meta.get("phase_b_rollout") or {})
        event_frames = [
            int(x)
            for x in list(
                rollout_meta.get(
                    "trained_current_frame_indices",
                    rollout_meta.get("current_event_frame_indices", []),
                )
                or []
            )
        ]
        current_recon_frames = [
            int(x) for x in list(final_meta.get("current_recon_frames", []) or [])
        ]
        if current_recon_frames != event_frames:
            raise ValueError(
                "Phase B final_current_recon frames must equal current rollout event frames: "
                f"current_recon_frames={current_recon_frames}, event_frames={event_frames}"
            )
        if num_cams > 0:
            expected_current_refs = {
                (int(frame_idx), int(cam_idx))
                for frame_idx in event_frames
                for cam_idx in range(int(num_cams))
            }
            actual_current_refs = {
                ref
                for ref, role in zip(final_refs, final_roles)
                if str(role) == "final_current_recon"
            }
            if actual_current_refs != expected_current_refs:
                missing = sorted(expected_current_refs - actual_current_refs)
                extra = sorted(actual_current_refs - expected_current_refs)
                raise ValueError(
                    "Phase B final_current_recon refs must exactly cover all cams of trained current frames. "
                    f"missing={missing[:8]} extra={extra[:8]}"
                )
        actual_blocks = int(rollout_meta.get("actual_blocks_per_rollout", len(event_frames)) or 0)
        repeats = int(phase_b_repeat.get("repeats_per_block", 0) or 0)
        if int(actual_blocks) != len(event_frames):
            raise ValueError("Phase B actual_blocks_per_rollout must equal number of trained current frames.")
        if int(inner_K) != int(actual_blocks) * int(repeats):
            raise ValueError("Phase B actual inner_K must equal actual_blocks_per_rollout * repeats_per_block.")
    else:
        query_frames = {int(ref[0]) for ref in query_refs}
        if len(query_frames) != 1:
            raise ValueError("Phase B P0 query observation supports exactly one query frame per rollout.")
        if num_cams > 0 and len(query_refs) != int(num_cams):
            raise ValueError(
                f"Phase B P0 query observation expects all cams for one frame: got {len(query_refs)} refs num_cams={num_cams}"
            )

    _require_order_matches(_tensor_ref_list(batch.get("source")), source_refs, "source")
    _require_order_matches(_dict_ref_list(batch.get("source_views")), source_refs, "source_views")
    _require_order_matches(_tensor_ref_list(batch.get("target")), target_refs, "target")
    _require_order_matches(_dict_ref_list(batch.get("targets")), target_refs, "targets")
    if not final_only or query_refs:
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
                raise ValueError(f"render loss ref {ref} missing from target_image_refs")
            idx = int(target_index_by_ref[ref])
            actual_role = str(target_roles[idx])
            if not final_only and actual_role != "prefix_loss":
                raise ValueError(
                    f"prefix_loss ref {ref} mapped to target role {actual_role!r}, expected 'prefix_loss'"
                )
            if final_only and actual_role not in PHASE_B_FINAL_TARGET_ROLES:
                raise ValueError(
                    f"final render ref {ref} mapped to target role {actual_role!r}, expected final role"
                )
            out.append(idx)
        return out

    num_query_targets = len(list(batch.get("query_targets") or []))
    if num_query_targets and int(num_query_targets) != len(query_refs):
        raise ValueError(f"query_targets/query_label_refs length mismatch: {num_query_targets} vs {len(query_refs)}")
    query_indices = [int(query_index_by_ref[ref]) for ref in query_refs]
    final_supervision_step_idx = -1
    final_target_indices_by_role: Dict[str, List[int]] = {}
    final_target_indices: List[int] = []
    final_supervision: Dict[str, Any] = {}
    phase_b_rollout: Dict[str, Any] = {}
    visit_time_codes: List[Tuple[float, float, float, float]] = []
    if final_only:
        final_supervision = dict(meta.get("phase_b_final_supervision") or {})
        final_supervision_step_idx = int(final_supervision.get("step_idx", int(inner_K) - 1))
        if int(final_supervision_step_idx) != int(inner_K) - 1:
            raise ValueError("Phase B final supervision step_idx must be inner_K - 1.")
        for idx, role in enumerate(target_roles):
            final_target_indices_by_role.setdefault(str(role), []).append(int(idx))
        final_target_indices = list(range(len(target_refs)))
        phase_b_rollout = dict(meta.get("phase_b_rollout") or {})
        raw_visit_time_codes = list(meta.get("visit_time_codes") or [])
        if raw_visit_time_codes:
            if len(raw_visit_time_codes) != int(inner_K):
                raise ValueError("visit_time_codes length must equal inner_K.")
            visit_time_codes = [
                (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
                for item in raw_visit_time_codes
            ]
        else:
            visit_time_codes = [(0.0, 0.0, 0.0, 0.0) for _ in range(int(inner_K))]

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
        final_supervision_step_idx=int(final_supervision_step_idx),
        final_target_indices_by_role=final_target_indices_by_role,
        final_target_indices=final_target_indices,
        phase_b_rollout=phase_b_rollout,
        final_supervision=final_supervision,
        visit_time_codes=visit_time_codes,
    )
