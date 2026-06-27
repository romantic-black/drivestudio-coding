from __future__ import annotations

import torch
import pytest
from types import SimpleNamespace

from models.iforward.biggs_state import BigGSBranchAssignment
from models.streetforward.stage6_0.event_encoder import EventPack
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack
from models.iforward.stage2_2.parent_temporal_keys_v2 import ParentTemporalKeysV2
from models.iforward.stage2_3 import ParentOptimizerMamba, ParentOptimizerMambaState, VisitMeta, build_parent_delta_summary


def _memory():
    torch.manual_seed(1)
    return ParentOptimizerMamba(event_dim=4, ctx_dim=3, model_dim=5, state_dim=2, conv_kernel=2, visit_dim=4)


def _event():
    return EventPack(
        event_bg=torch.randn(3, 4),
        event_distant=torch.randn(2, 4),
        event_rigid=torch.randn(3, 4),
        support_bg=torch.ones(3, 1),
        support_distant=torch.ones(2, 1),
        support_rigid=torch.tensor([[1.0], [3.0], [1.0]]),
        valid_bg=torch.ones(3, dtype=torch.bool),
        valid_distant=torch.ones(2, dtype=torch.bool),
        valid_rigid=torch.ones(3, dtype=torch.bool),
    )


def _keys():
    return ParentTemporalKeysV2(
        bg=torch.arange(3),
        distant=torch.arange(2) + 100,
        rigid=torch.tensor([200, 200, 201], dtype=torch.long),
    )


def _visit(kind="assimilate", repeat=0):
    return VisitMeta(
        visit_kind=kind,
        frame_id=7,
        keyframe_id=7,
        sequence_pos=1,
        timestamp_us=700000,
        frame_gap_from_previous_visit=1,
        time_since_same_frame_visit=0.0,
        visit_count_for_frame=0,
        repeat_idx=repeat,
        repeat_budget=4,
        global_update_idx_in_episode=repeat,
        is_first_visit_of_frame=repeat == 0,
        is_last_update_of_episode=False,
    )


def test_stage2_3_optimizer_preview_no_mutation_and_unseen_zero():
    mem = _memory()
    event = _event()
    state = ParentOptimizerMambaState.empty()
    preview = mem.preview(event=event, state=state, keys=_keys(), visit_meta=_visit())
    assert state.bg.dense is None
    assert torch.allclose(preview.event.event_bg, event.event_bg)
    assert preview.aux["iforward/parent_optimizer_mamba/bg_preview_seen_ratio"] == 0.0


def test_stage2_3_optimizer_every_repeat_writes_and_repair_writes():
    mem = _memory()
    event = _event()
    state = ParentOptimizerMambaState.empty()
    for repeat in range(2):
        preview = mem.preview(event=event, state=state, keys=_keys(), visit_meta=_visit("repair", repeat))
        state, aux = mem.write(
            spatial_event=event,
            fused_event=preview.event,
            state=state,
            keys=_keys(),
            visit_meta=_visit("repair", repeat),
        )
        assert aux["iforward/parent_optimizer_mamba/write"] == 1.0
    assert state.global_update_step == 2
    assert state.bg.dense is not None
    assert int(state.bg.dense.update_count.sum().item()) == 6
    assert state.rigid.keyed is not None
    assert state.rigid.keyed.keys.detach().cpu().tolist() == [200, 201]


def test_stage2_3_optimizer_visit_repeat_changes_context_after_seen():
    mem = _memory()
    event = _event()
    state, _ = mem.write(spatial_event=event, state=ParentOptimizerMambaState.empty(), keys=_keys(), visit_meta=_visit("assimilate", 0))
    a = mem.preview(event=event, state=state, keys=_keys(), visit_meta=_visit("assimilate", 1))
    b = mem.preview(event=event, state=state, keys=_keys(), visit_meta=_visit("repair", 1))
    assert (a.event.event_bg - b.event.event_bg).abs().max() > 1.0e-8


def test_stage2_3_optimizer_detach_keeps_state_value():
    mem = _memory()
    event = _event()
    state, _ = mem.write(spatial_event=event, state=ParentOptimizerMambaState.empty(), keys=_keys(), visit_meta=_visit())
    detached = state.detach()
    assert detached is not state
    assert detached.bg.dense is not state.bg.dense
    assert detached.count_tokens()["parent_optimizer_mamba_global_update_step"] == 1.0


def _branch_delta(rows: int) -> BranchDelta:
    means = torch.arange(1, rows * 3 + 1, dtype=torch.float32).reshape(rows, 3)
    return BranchDelta(
        means=means,
        scales_log=torch.ones(rows, 3) * 0.2,
        quat_axis_angle=torch.ones(rows, 3) * 0.3,
        opacity_logit=torch.ones(rows, 1) * 0.4,
        sh=torch.ones(rows, 6) * 0.5,
        hidden=torch.zeros(rows, 1),
        confidence=torch.linspace(0.1, 0.9, rows).reshape(rows, 1),
        noop=torch.linspace(0.9, 0.1, rows).reshape(rows, 1),
    )


def test_stage2_3_parent_delta_summary_scatters_child_rows_to_parent_rows():
    event = EventPack(event_bg=torch.zeros(2, 4), support_bg=torch.ones(2, 1), valid_bg=torch.ones(2, dtype=torch.bool))
    assignment = BigGSBranchAssignment(
        branch="bg",
        child_to_parent=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        child_order=torch.arange(4),
        parent_start=torch.tensor([0, 2], dtype=torch.long),
        parent_count=torch.tensor([2, 2], dtype=torch.long),
        child_mass=torch.tensor([1.0, 3.0, 2.0, 2.0]),
        num_children=4,
        num_parents=2,
    )
    runtime = SimpleNamespace(bg_assignment=assignment, distant_assignment=None, rigid_active_assignment=None)
    summary, aux = build_parent_delta_summary(
        delta=DeltaPack(bg=_branch_delta(4)),
        runtime=runtime,
        spatial_event=event,
        fail_fast=True,
    )
    assert summary.bg is not None
    assert tuple(summary.bg.summary7.shape) == (2, 7)
    assert torch.count_nonzero(summary.bg.summary7).item() > 0
    assert aux["iforward/parent_optimizer_mamba/delta_summary_bg_nonzero_ratio"] == 1.0
    assert aux["iforward/parent_optimizer_mamba/delta_summary_bg_missing"] == 0.0


def test_stage2_3_parent_delta_summary_missing_mapping_fail_fast():
    event = EventPack(event_bg=torch.zeros(2, 4), support_bg=torch.ones(2, 1), valid_bg=torch.ones(2, dtype=torch.bool))
    runtime = SimpleNamespace(bg_assignment=None, distant_assignment=None, rigid_active_assignment=None)
    with pytest.raises(ValueError, match="requires BigGS parent assignment"):
        build_parent_delta_summary(
            delta=DeltaPack(bg=_branch_delta(4)),
            runtime=runtime,
            spatial_event=event,
            fail_fast=True,
        )
