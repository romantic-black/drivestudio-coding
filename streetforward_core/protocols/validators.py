from __future__ import annotations

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

