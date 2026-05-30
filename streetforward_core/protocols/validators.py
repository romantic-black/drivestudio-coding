from __future__ import annotations

from .phase_b_long import (
    LONG_TARGET_ROLES,
    PHASE_B_LONG_NAME,
    PHASE_B_LONG_PROTOCOL_VERSION,
    PHASE_B_LONG_SCHEDULER_VERSION,
    PhaseBLongRolloutPlan,
)
from .roles import LongRole
from .rollout import PHASE_A_NAME, PHASE_A_PROTOCOL_VERSION, RolloutPlan


def validate_phase_a_plan(plan: RolloutPlan) -> None:
    if plan.protocol_version != PHASE_A_PROTOCOL_VERSION:
        raise ValueError(f"Phase A requires protocol_version={PHASE_A_PROTOCOL_VERSION}")
    if plan.phase != PHASE_A_NAME:
        raise ValueError("Phase A plan has wrong phase")
    if int(plan.inner_K) < 1:
        raise ValueError("Phase A requires inner_K >= 1")
    if len(plan.steps) != int(plan.inner_K):
        raise ValueError("len(steps) must equal inner_K")

    evidence = set()
    nearby = set()
    for expected_idx, step in enumerate(plan.steps):
        if int(step.step_idx) != int(expected_idx):
            raise ValueError(f"step index mismatch at position {expected_idx}: got {step.step_idx}")
        if not step.evidence_refs:
            raise ValueError(f"step {step.step_idx} requires evidence_refs")
        evidence.update(step.evidence_refs)
        nearby.update(step.nearby_loss_refs)
    if evidence & nearby:
        raise ValueError("Phase A nearby_loss_refs leaked into evidence_refs")


def _stable_unique(refs):
    seen = set()
    out = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
    return tuple(out)


def _long_role(raw):
    if isinstance(raw, LongRole):
        return raw
    return LongRole(str(raw))


def validate_phase_b_long_plan(plan: PhaseBLongRolloutPlan) -> None:
    if str(plan.protocol_version) != PHASE_B_LONG_PROTOCOL_VERSION:
        raise ValueError(f"Phase B Long requires protocol_version={PHASE_B_LONG_PROTOCOL_VERSION}")
    if str(plan.scheduler_version) != PHASE_B_LONG_SCHEDULER_VERSION:
        raise ValueError("Phase B Long requires scheduler_version=long_v1")
    if str(plan.phase) != PHASE_B_LONG_NAME:
        raise ValueError("Phase B Long requires phase=6_0_phase_b")
    if int(plan.inner_K) < 1:
        raise ValueError("Phase B Long requires inner_K >= 1")
    if int(plan.repeats_per_anchor) * int(plan.anchors_per_rollout) != int(plan.inner_K):
        raise ValueError("Phase B Long inner_K must equal repeats_per_anchor * anchors_per_rollout")
    if len(plan.visits) != int(plan.inner_K):
        raise ValueError("Phase B Long visits length must equal inner_K")
    if len(plan.evidence_refs_by_step) != int(plan.inner_K):
        raise ValueError("Phase B Long evidence_refs_by_step length must equal inner_K")
    for idx, (visit, refs) in enumerate(zip(plan.visits, plan.evidence_refs_by_step)):
        if int(visit.step_idx) != int(idx):
            raise ValueError("Phase B Long visits.step_idx must be contiguous")
        if len(refs) == 0:
            raise ValueError("Phase B Long requires non-empty evidence refs at every step")
        first = refs[0]
        if int(first.frame_idx) != int(visit.frame_idx) or int(first.cam_idx) != int(visit.cam_idx):
            raise ValueError("Phase B Long visit frame/cam must match first evidence ref")
        for code_name in ("visit_pos_code", "frame_time_code", "chronological_rank_code", "repeat_idx_code"):
            value = float(getattr(visit, code_name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Phase B Long {code_name} must be in [0, 1]")

    flat_evidence = tuple(ref for group in plan.evidence_refs_by_step for ref in group)
    if tuple(plan.source_image_refs) != _stable_unique(flat_evidence):
        raise ValueError("Phase B Long source_image_refs must be stable_unique(flatten(evidence_refs_by_step))")
    if len(plan.target_image_refs) == 0:
        raise ValueError("Phase B Long requires non-empty target_image_refs")
    if len(plan.target_image_refs) != len(plan.target_image_roles):
        raise ValueError("Phase B Long target_image_refs/target_image_roles length mismatch")
    allowed = set(LONG_TARGET_ROLES)
    observed = set(plan.target_image_roles)
    if not observed.issubset(allowed):
        if any(str(role) in {"final_history", "final_current"} for role in observed):
            raise ValueError("Phase B Long rejects coarse final_history/final_current roles")
        raise ValueError(f"Phase B Long target roles must be split Long roles, got {observed}")

    if "required_final_roles" in (plan.meta or {}):
        required = tuple(_long_role(role) for role in list((plan.meta or {}).get("required_final_roles") or []))
    else:
        required = (LongRole.FINAL_CURRENT_RECON, LongRole.FINAL_CURRENT_NVS)
    for role in required:
        if role not in observed:
            raise ValueError(f"Phase B Long required role {role.value} is empty")

    for key in ("query_label_refs", "prefix_loss_refs_by_step", "nearby_loss_refs_by_step", "block_loss_refs_by_step"):
        raw = (plan.meta or {}).get(key)
        if raw is None:
            continue
        if key.endswith("_by_step"):
            if any(len(group) > 0 for group in list(raw or [])):
                raise ValueError(f"Phase B Long requires empty {key}")
        elif len(list(raw or [])) > 0:
            raise ValueError(f"Phase B Long requires empty {key}")

    if tuple(plan.anchor_frames_chronological) != tuple(sorted(int(x) for x in plan.anchor_frames_chronological)):
        raise ValueError("Phase B Long anchor_frames_chronological must be sorted")
    if set(plan.anchor_frames_rollout_order) != set(plan.anchor_frames_chronological):
        raise ValueError("Phase B Long anchor_frames_rollout_order must be a permutation of chronological anchors")

    evidence_set = set(flat_evidence)
    nvs_refs = set(plan.final_history_nvs_refs) | set(plan.final_current_nvs_refs)
    overlap = len(evidence_set & nvs_refs)
    max_ratio = float((plan.meta or {}).get("max_nvs_fallback_ratio", 0.25))
    ratio = float((plan.meta or {}).get("nvs_fallback_to_evidence_cam_ratio", 0.0))
    if nvs_refs:
        ratio = max(ratio, float(overlap) / float(max(len(nvs_refs), 1)))
    if ratio > max_ratio:
        raise ValueError(f"Phase B Long NVS/evidence overlap ratio too high: {ratio:.3f} > {max_ratio:.3f}")
