from __future__ import annotations

import json

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
