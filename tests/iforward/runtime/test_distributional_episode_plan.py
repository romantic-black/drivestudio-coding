from __future__ import annotations

from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.plan import EpisodePlan
from models.iforward.runtime.runner import IForwardRunner, RunnerOptions
from models.iforward.runtime.trace import TraceRecorder
from tests.test_iforward_stage2_3_validation import _FakeModel
from tests.test_iforward_stage3_2_distributional_scheduler import _scheduler


def test_distributional_episode_plan_serializes_and_traces_metadata(tmp_path):
    scheduler = _scheduler(seed=31)
    episode = scheduler._build_episode()
    adapter = Stage3SchedulerAdapter(scheduler)
    plan = adapter.plan_from_episode_v3(episode, "distribution_assimilation_timeline/seq16/entry0")
    restored = EpisodePlan.from_json_dict(plan.to_json_dict())

    assert restored.version == "iforward_episode_plan_v1"
    assert restored.events[0].metadata["distribution_type"] in {
        "repeat_refine",
        "shuffled_coverage",
        "high_block_repair",
    }

    model = _FakeModel()
    trace = IForwardRunner(model, adapter).run(
        restored,
        TraceRecorder(tmp_path, record_images=False),
        RunnerOptions.for_mode("validate", device="cpu", trigger_step=11),
    )

    assert trace.events
    assert any(event.metadata.get("iforward_stage3_2") for event in trace.events)
