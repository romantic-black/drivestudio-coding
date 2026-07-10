from __future__ import annotations

from types import SimpleNamespace

import torch

from models.iforward.biggs_state import BigGSBranchAssignment, BigGSRigidActiveAssignment
from models.iforward.diagnostics.parent_attribution import build_parent_diagnostics


def _branch_state(values: torch.Tensor) -> SimpleNamespace:
    n = int(values.shape[0])
    return SimpleNamespace(
        means=values.reshape(n, 1).repeat(1, 3),
        scales_log=torch.zeros((n, 3)),
        opacity_logit=torch.zeros((n, 1)),
        sh_dc=torch.zeros((n, 3)),
        sh_rest=torch.zeros((n, 1, 3)),
        hidden=torch.zeros((n, 2)),
    )


def _state(*, assignment: object, prev_values: torch.Tensor | None, next_values: torch.Tensor) -> tuple[object | None, object]:
    prev = None
    if prev_values is not None:
        prev = SimpleNamespace(
            biggs_state=SimpleNamespace(bg=assignment, distant=None, rigid=None),
            local_gs=SimpleNamespace(bg=_branch_state(prev_values), distant=None, rigid=None),
            parent_temporal=None,
        )
    nxt = SimpleNamespace(
        biggs_state=SimpleNamespace(bg=assignment, distant=None, rigid=None),
        local_gs=SimpleNamespace(bg=_branch_state(next_values), distant=None, rigid=None),
        parent_temporal=None,
    )
    return prev, nxt


def test_parent_diagnostics_aggregates_branch_assignment_topk():
    assignment = BigGSBranchAssignment(
        branch="bg",
        child_to_parent=torch.tensor([0, 0, 1, 1]),
        child_order=torch.arange(4),
        parent_start=torch.tensor([0, 2]),
        parent_count=torch.tensor([2, 2]),
        child_mass=torch.tensor([0.5, 0.5, 1.0, 1.0]),
        num_children=4,
        num_parents=2,
    )
    prev, nxt = _state(
        assignment=assignment,
        prev_values=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        next_values=torch.tensor([0.1, 0.1, 1.0, 1.0]),
    )

    result = build_parent_diagnostics(previous_state=prev, next_state=nxt, topk=2)

    assert result.summary["num_rows"] == 2
    assert result.rows[0]["branch"] == "bg"
    assert result.rows[0]["parent_id"] == 1
    assert result.rows[0]["child_count"] == 2
    assert result.rows[0]["impact_score"] > result.rows[1]["impact_score"]


def test_parent_diagnostics_accepts_rigid_active_assignment_shape():
    assignment = BigGSRigidActiveAssignment(
        fine_S=torch.tensor([0, 2, 4]),
        child_to_active_parent_S=torch.tensor([0, 1, 1]),
        active_parent_global=torch.tensor([10, 11]),
        active_parent_count=torch.tensor([1, 2]),
        active_parent_start=torch.tensor([0, 1]),
        active_child_order_S=torch.tensor([0, 1, 2]),
        child_mass_S=torch.tensor([1.0, 0.5, 0.5]),
        parent_inside_mask=torch.tensor([True, True]),
        child_inside_mask_S=torch.tensor([True, True, True]),
    )
    prev = SimpleNamespace(
        biggs_state=SimpleNamespace(bg=None, distant=None, rigid=assignment),
        local_gs=SimpleNamespace(bg=None, distant=None, rigid=_branch_state(torch.zeros(5))),
        parent_temporal=None,
    )
    nxt = SimpleNamespace(
        biggs_state=SimpleNamespace(bg=None, distant=None, rigid=assignment),
        local_gs=SimpleNamespace(bg=None, distant=None, rigid=_branch_state(torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0]))),
        parent_temporal=None,
    )

    result = build_parent_diagnostics(previous_state=prev, next_state=nxt, topk=2)

    assert result.rows
    assert {row["parent_id"] for row in result.rows} == {10, 11}
