from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .types import ImageRef, LONG_TARGET_ROLES, PHASE_B_LONG_NAME, LongVisit, ResolvedLongPhaseBBatch


def _as_ref(raw: Any) -> ImageRef:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"image ref must be a length-2 tuple/list, got {raw!r}")
    return int(raw[0]), int(raw[1])


def _as_ref_list(raw: Any) -> List[ImageRef]:
    return [_as_ref(x) for x in list(raw or [])]


def _as_ref_steps(raw: Any, name: str) -> List[List[ImageRef]]:
    if raw is None:
        raise ValueError(f"{name} is required")
    return [[_as_ref(x) for x in list(group)] for group in list(raw)]


def _first_present(meta: Dict[str, Any], sched: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in meta and meta[key] is not None:
        return meta[key]
    return sched.get(key, default)


def _dict_ref_list(items: Any) -> Optional[List[ImageRef]]:
    if not isinstance(items, list) or not items:
        return None
    out: List[ImageRef] = []
    for item in items:
        if not isinstance(item, dict) or "frame_idx" not in item or "cam_idx" not in item:
            return None
        out.append((int(item["frame_idx"]), int(item["cam_idx"])))
    return out


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


def _require_order_matches(actual: Optional[List[ImageRef]], expected: List[ImageRef], name: str) -> None:
    if actual is None:
        return
    if len(actual) != len(expected):
        raise ValueError(f"{name} length mismatch: {len(actual)} vs {len(expected)}")
    for idx, (a, e) in enumerate(zip(actual, expected)):
        if tuple(a) != tuple(e):
            raise ValueError(f"{name} order/content mismatch at index {idx}: got {tuple(a)} expected {tuple(e)}")


def _ensure_empty_groups(raw: Any, *, name: str, inner_K: int) -> None:
    if raw is None:
        return
    groups = _as_ref_steps(raw, name)
    if len(groups) not in {0, int(inner_K)}:
        raise ValueError(f"{name} length must equal inner_K when present.")
    if any(len(group) > 0 for group in groups):
        raise ValueError(f"6_0_phase_b V1 requires empty {name}.")


def _lookup_refs(
    refs: Iterable[ImageRef],
    index_by_ref: Dict[ImageRef, int],
    roles_by_ref: Optional[Dict[ImageRef, str]],
    *,
    expected_role: Optional[str],
    label: str,
) -> List[int]:
    out: List[int] = []
    for ref in refs:
        if ref not in index_by_ref:
            raise ValueError(f"{label} ref {ref} missing from image refs")
        if roles_by_ref is not None and expected_role is not None:
            actual = str(roles_by_ref.get(ref, ""))
            if actual != str(expected_role):
                raise ValueError(f"{label} ref {ref} mapped to target role {actual!r}, expected {expected_role!r}")
        out.append(int(index_by_ref[ref]))
    return out


def _float_list(raw: Any, *, length: int, name: str, default: float = 0.0) -> List[float]:
    if raw is None:
        return [float(default) for _ in range(int(length))]
    out = [float(x) for x in list(raw)]
    if len(out) != int(length):
        raise ValueError(f"{name} length must equal inner_K.")
    return out


def _int_list(raw: Any, *, length: int, name: str, default: int = 0) -> List[int]:
    if raw is None:
        return [int(default) for _ in range(int(length))]
    out = [int(x) for x in list(raw)]
    if len(out) != int(length):
        raise ValueError(f"{name} length must equal inner_K.")
    return out


def _parse_visits(meta: Dict[str, Any], sched: Dict[str, Any], *, inner_K: int, evidence_by_step: List[List[ImageRef]]) -> List[LongVisit]:
    raw = _first_present(meta, sched, "visits", None)
    if raw is not None:
        visits: List[LongVisit] = []
        for item in list(raw):
            if not isinstance(item, dict):
                raise ValueError("6_0_phase_b visits entries must be dictionaries.")
            visits.append(
                LongVisit(
                    step_idx=int(item.get("step_idx", len(visits))),
                    anchor_id=int(item.get("anchor_id", item.get("rollout_order_rank", 0))),
                    frame_idx=int(item["frame_idx"]),
                    cam_idx=int(item.get("cam_idx", evidence_by_step[len(visits)][0][1])),
                    repeat_idx=int(item.get("repeat_idx", 0)),
                    rollout_order_rank=int(item.get("rollout_order_rank", item.get("anchor_id", 0))),
                    chronological_rank=int(item.get("chronological_rank", 0)),
                    visit_pos_code=float(item.get("visit_pos_code", 0.0)),
                    frame_time_code=float(item.get("frame_time_code", 0.0)),
                    chronological_rank_code=float(item.get("chronological_rank_code", 0.0)),
                    repeat_idx_code=float(item.get("repeat_idx_code", 0.0)),
                )
            )
        if len(visits) != int(inner_K):
            raise ValueError("visits length must equal inner_K.")
        for idx, visit in enumerate(visits):
            if int(visit.step_idx) != int(idx):
                raise ValueError("visits.step_idx must be contiguous and match list order.")
            if int(visit.frame_idx) != int(evidence_by_step[idx][0][0]):
                raise ValueError("visits frame_idx must match evidence_refs_by_step first ref.")
            if int(visit.cam_idx) != int(evidence_by_step[idx][0][1]):
                raise ValueError("visits cam_idx must match evidence_refs_by_step first ref.")
        return visits

    step_frame_indices = _int_list(
        _first_present(meta, sched, "step_frame_indices", None),
        length=inner_K,
        name="step_frame_indices",
    )
    step_repeat_indices = _int_list(
        _first_present(meta, sched, "step_repeat_indices", None),
        length=inner_K,
        name="step_repeat_indices",
    )
    step_anchor_ids = _int_list(
        _first_present(meta, sched, "step_anchor_ids", None),
        length=inner_K,
        name="step_anchor_ids",
    )
    rollout_ranks = _int_list(
        _first_present(meta, sched, "step_rollout_order_ranks", None),
        length=inner_K,
        name="step_rollout_order_ranks",
    )
    chrono_ranks = _int_list(
        _first_present(meta, sched, "step_chronological_ranks", None),
        length=inner_K,
        name="step_chronological_ranks",
    )
    visit_pos_codes = _float_list(
        _first_present(meta, sched, "visit_pos_codes", None),
        length=inner_K,
        name="visit_pos_codes",
    )
    frame_time_codes = _float_list(
        _first_present(meta, sched, "frame_time_codes", None),
        length=inner_K,
        name="frame_time_codes",
    )
    chrono_codes = _float_list(
        _first_present(meta, sched, "chronological_rank_codes", None),
        length=inner_K,
        name="chronological_rank_codes",
    )
    repeat_codes = _float_list(
        _first_present(meta, sched, "repeat_idx_codes", None),
        length=inner_K,
        name="repeat_idx_codes",
    )
    visits = []
    for idx in range(int(inner_K)):
        visits.append(
            LongVisit(
                step_idx=int(idx),
                anchor_id=int(step_anchor_ids[idx]),
                frame_idx=int(step_frame_indices[idx]),
                cam_idx=int(evidence_by_step[idx][0][1]),
                repeat_idx=int(step_repeat_indices[idx]),
                rollout_order_rank=int(rollout_ranks[idx]),
                chronological_rank=int(chrono_ranks[idx]),
                visit_pos_code=float(visit_pos_codes[idx]),
                frame_time_code=float(frame_time_codes[idx]),
                chronological_rank_code=float(chrono_codes[idx]),
                repeat_idx_code=float(repeat_codes[idx]),
            )
        )
    return visits


def resolve_long_phase_b_batch(batch: Dict[str, Any]) -> ResolvedLongPhaseBBatch:
    meta = dict(batch.get("request_meta") or {})
    sched = dict(batch.get("_scheduler_long_phase_b") or {})

    scheduler_version = str(_first_present(meta, sched, "scheduler_version", sched.get("version", "")))
    if scheduler_version != "long_v1":
        raise ValueError("6_0_phase_b requires scheduler_version=long_v1.")

    phase = str(_first_present(meta, sched, "scheduler_phase", sched.get("phase", "")))
    if phase != PHASE_B_LONG_NAME:
        raise ValueError("6_0_phase_b requires scheduler_phase=6_0_phase_b.")

    assembly_mode = str(meta.get("assembly_mode", sched.get("assembly_mode", "")))
    if assembly_mode != "image_ref_long_v1":
        raise ValueError("6_0_phase_b requires request_meta.assembly_mode=image_ref_long_v1.")

    inner_K = int(_first_present(meta, sched, "inner_K", 0) or 0)
    if inner_K < 1:
        raise ValueError("6_0_phase_b requires inner_K >= 1.")

    evidence_by_step = _as_ref_steps(
        _first_present(meta, sched, "evidence_refs_by_step"),
        "evidence_refs_by_step",
    )
    if len(evidence_by_step) != int(inner_K):
        raise ValueError("evidence_refs_by_step length must equal inner_K.")
    if any(len(step) == 0 for step in evidence_by_step):
        raise ValueError("6_0_phase_b requires non-empty evidence refs at every step.")

    query_refs = _as_ref_list(_first_present(meta, sched, "query_label_refs", []))
    if len(query_refs) > 0:
        raise ValueError("6_0_phase_b V1 requires empty query_label_refs.")
    for key in ("prefix_loss_refs_by_step", "nearby_loss_refs_by_step", "block_loss_refs_by_step"):
        _ensure_empty_groups(_first_present(meta, sched, key, None), name=key, inner_K=inner_K)

    source_refs = _as_ref_list(meta.get("source_image_refs") or sched.get("source_image_refs") or [])
    target_refs = _as_ref_list(meta.get("target_image_refs") or sched.get("target_image_refs") or [])
    target_roles = [str(x) for x in list(meta.get("target_image_roles") or sched.get("target_image_roles") or [])]
    if not source_refs:
        raise ValueError("6_0_phase_b requires non-empty source_image_refs.")
    if not target_refs:
        raise ValueError("6_0_phase_b requires non-empty target_image_refs.")
    if len(target_refs) != len(target_roles):
        raise ValueError("target_image_refs and target_image_roles length mismatch.")
    allowed_roles = set(LONG_TARGET_ROLES)
    observed_roles = set(target_roles)
    if not observed_roles.issubset(allowed_roles):
        if observed_roles & {"final_history", "final_current"}:
            raise ValueError("6_0_phase_b Long V1 rejects coarse final_history/final_current roles; use split recon/NVS roles.")
        raise ValueError(f"6_0_phase_b target roles must be split Long roles {allowed_roles}, got {observed_roles}.")

    source_index_by_ref = {ref: i for i, ref in enumerate(source_refs)}
    target_index_by_ref = {ref: i for i, ref in enumerate(target_refs)}
    target_role_by_ref = {ref: role for ref, role in zip(target_refs, target_roles)}
    target_role_indices = {role: [idx for idx, item_role in enumerate(target_roles) if item_role == role] for role in LONG_TARGET_ROLES}
    required_final_roles = [
        str(x)
        for x in list(
            _first_present(
                meta,
                sched,
                "required_final_roles",
                ["final_current_recon", "final_current_nvs"],
            )
            or []
        )
    ]
    for role in required_final_roles:
        if role not in allowed_roles:
            raise ValueError(f"6_0_phase_b required_final_roles contains unknown Long role {role!r}.")
        if len(target_role_indices.get(role, [])) == 0:
            raise ValueError(f"required Long role {role} is empty.")

    evidence_source_indices = [
        _lookup_refs(step, source_index_by_ref, None, expected_role=None, label="evidence") for step in evidence_by_step
    ]
    final_history_recon_indices = target_role_indices["final_history_recon"]
    final_history_nvs_indices = target_role_indices["final_history_nvs"]
    final_current_recon_indices = target_role_indices["final_current_recon"]
    final_current_nvs_indices = target_role_indices["final_current_nvs"]
    final_history_recon_refs = [target_refs[i] for i in final_history_recon_indices]
    final_history_nvs_refs = [target_refs[i] for i in final_history_nvs_indices]
    final_current_recon_refs = [target_refs[i] for i in final_current_recon_indices]
    final_current_nvs_refs = [target_refs[i] for i in final_current_nvs_indices]

    expected_role_refs = {
        "final_history_recon": final_history_recon_refs,
        "final_history_nvs": final_history_nvs_refs,
        "final_current_recon": final_current_recon_refs,
        "final_current_nvs": final_current_nvs_refs,
    }
    for role_name, expected in expected_role_refs.items():
        explicit = _as_ref_list(meta.get(f"{role_name}_refs") or sched.get(f"{role_name}_refs") or [])
        if explicit and explicit != expected:
            raise ValueError(f"{role_name}_refs must match target_image_roles order.")

    final_history_refs = final_history_recon_refs + final_history_nvs_refs
    final_current_refs = final_current_recon_refs + final_current_nvs_refs
    final_history_indices = final_history_recon_indices + final_history_nvs_indices
    final_current_indices = final_current_recon_indices + final_current_nvs_indices

    if "step_block_indices" in meta or "step_block_indices" in sched:
        raise ValueError("6_0_phase_b Long V1 forbids step_block_indices; use LongVisit anchor metadata.")

    visits = _parse_visits(meta, sched, inner_K=int(inner_K), evidence_by_step=evidence_by_step)
    step_frame_indices = [int(v.frame_idx) for v in visits]
    step_repeat_indices = [int(v.repeat_idx) for v in visits]
    step_anchor_ids = [int(v.anchor_id) for v in visits]
    step_rollout_order_ranks = [int(v.rollout_order_rank) for v in visits]
    step_chronological_ranks = [int(v.chronological_rank) for v in visits]
    visit_time_codes = [
        (
            float(v.visit_pos_code),
            float(v.frame_time_code),
            float(v.chronological_rank_code),
            float(v.repeat_idx_code),
        )
        for v in visits
    ]

    _require_order_matches(_tensor_ref_list(batch.get("source")), source_refs, "source")
    _require_order_matches(_dict_ref_list(batch.get("source_views")), source_refs, "source_views")
    _require_order_matches(_tensor_ref_list(batch.get("target")), target_refs, "target")
    _require_order_matches(_dict_ref_list(batch.get("targets")), target_refs, "targets")

    return ResolvedLongPhaseBBatch(
        inner_K=int(inner_K),
        evidence_refs_by_step=evidence_by_step,
        evidence_source_indices_by_step=evidence_source_indices,
        visits=visits,
        shape_name=str(_first_present(meta, sched, "shape_name", "")),
        repeats_per_anchor=int(_first_present(meta, sched, "repeats_per_anchor", 0) or 0),
        anchors_per_rollout=int(_first_present(meta, sched, "anchors_per_rollout", 0) or 0),
        anchor_frames_chronological=[int(x) for x in list(_first_present(meta, sched, "anchor_frames_chronological", []) or [])],
        anchor_frames_rollout_order=[int(x) for x in list(_first_present(meta, sched, "anchor_frames_rollout_order", []) or [])],
        target_role_indices=target_role_indices,
        final_history_recon_refs=final_history_recon_refs,
        final_history_nvs_refs=final_history_nvs_refs,
        final_current_recon_refs=final_current_recon_refs,
        final_current_nvs_refs=final_current_nvs_refs,
        final_history_recon_target_indices=final_history_recon_indices,
        final_history_nvs_target_indices=final_history_nvs_indices,
        final_current_recon_target_indices=final_current_recon_indices,
        final_current_nvs_target_indices=final_current_nvs_indices,
        final_history_refs=final_history_refs,
        final_current_refs=final_current_refs,
        final_history_target_indices=final_history_indices,
        final_current_target_indices=final_current_indices,
        step_frame_indices=step_frame_indices,
        step_repeat_indices=step_repeat_indices,
        step_anchor_ids=step_anchor_ids,
        step_rollout_order_ranks=step_rollout_order_ranks,
        step_chronological_ranks=step_chronological_ranks,
        visit_time_codes=visit_time_codes,
        source_index_by_ref=source_index_by_ref,
        target_index_by_ref=target_index_by_ref,
        rigid_meta=dict(meta.get("rigid_meta") or sched.get("rigid_meta") or {}),
        distant_meta=dict(meta.get("distant_meta") or sched.get("distant_meta") or {}),
        tbptt_meta=dict(meta.get("tbptt") or sched.get("tbptt") or {}),
        request_meta=meta,
    )
