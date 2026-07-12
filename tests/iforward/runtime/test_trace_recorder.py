from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from models.iforward.biggs_state import BigGSBranchAssignment
from models.iforward.runtime.event import ControlEvent, EpisodeSpec
from models.iforward.runtime.plan import EpisodePlan
from models.iforward.runtime.trace import TraceRecorder


def test_trace_recorder_writes_plan_trace_and_summary(tmp_path):
    plan = EpisodePlan(
        plan_id="",
        version="iforward_episode_plan_v1",
        episode=EpisodeSpec(
            scene_id=1,
            segment_id=2,
            sequence_id=3,
            frame_ids=(10, 11),
            frame_positions=(0, 1),
            cam_ids=(0, 1, 2),
            protocol_name="unit",
        ),
        events=(ControlEvent(event_id="reset", kind="reset_state"),),
    ).with_stable_plan_id()
    recorder = TraceRecorder(tmp_path, record_images=False)
    trace = recorder.begin_plan(plan)
    recorder.record_control(plan.events[0], None, event_idx=0, memory_mode="full")
    recorder.end_plan(trace)

    assert (tmp_path / "plan.json").is_file()
    assert (tmp_path / "trace.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()
    row = json.loads((tmp_path / "trace.jsonl").read_text().strip())
    assert row["event_id"] == "reset"
    assert row["plan_id"] == plan.plan_id


def test_trace_recorder_writes_parent_diagnostics_artifacts(tmp_path):
    plan = EpisodePlan(
        plan_id="",
        version="iforward_episode_plan_v1",
        episode=EpisodeSpec(
            scene_id=1,
            segment_id=2,
            sequence_id=3,
            frame_ids=(10, 11),
            frame_positions=(0, 1),
            cam_ids=(0,),
            protocol_name="unit",
        ),
        events=(ControlEvent(event_id="update", kind="reset_state"),),
    ).with_stable_plan_id()
    assignment = BigGSBranchAssignment(
        branch="bg",
        child_to_parent=torch.tensor([0, 1]),
        child_order=torch.tensor([0, 1]),
        parent_start=torch.tensor([0, 1]),
        parent_count=torch.tensor([1, 1]),
        child_mass=torch.tensor([0.5, 1.0]),
        num_children=2,
        num_parents=2,
    )

    def branch(values):
        return SimpleNamespace(
            means=values.reshape(2, 1).repeat(1, 3),
            scales_log=torch.zeros((2, 3)),
            opacity_logit=torch.zeros((2, 1)),
            sh_dc=torch.zeros((2, 3)),
            sh_rest=torch.zeros((2, 1, 3)),
            hidden=torch.zeros((2, 1)),
        )

    previous_state = SimpleNamespace(
        biggs_state=SimpleNamespace(bg=assignment, distant=None, rigid=None),
        local_gs=SimpleNamespace(bg=branch(torch.tensor([0.0, 0.0])), distant=None, rigid=None),
        parent_temporal=None,
    )
    next_state = SimpleNamespace(
        biggs_state=SimpleNamespace(bg=assignment, distant=None, rigid=None),
        local_gs=SimpleNamespace(bg=branch(torch.tensor([0.0, 1.0])), distant=None, rigid=None),
        parent_temporal=None,
    )
    out = SimpleNamespace(
        resolved=SimpleNamespace(meta={}),
        stats={},
        losses={},
        loss=torch.tensor(0.0),
        pred_rgbs=[],
        gt_images=[],
        next_state=next_state,
    )
    recorder = TraceRecorder(tmp_path, record_images=False)
    trace = recorder.begin_plan(plan)
    recorder.record_update(plan.events[0], out, next_state, event_idx=0, memory_mode="full", previous_state=previous_state)
    recorder.end_plan(trace)

    row = json.loads((tmp_path / "trace.jsonl").read_text().strip())
    assert row["artifacts"]["parent_topk_csv"].startswith("parent_diagnostics/")
    assert row["metadata"]["parent_diagnostics"]["num_rows"] == 2
    assert (tmp_path / "parent_diagnostics_summary.csv").is_file()
