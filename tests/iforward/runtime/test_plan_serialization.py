from __future__ import annotations

import json

from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.plan import EpisodePlan
from tests.test_iforward_stage2_3_scheduler import _scheduler


def test_episode_plan_json_roundtrip_preserves_rollout_plan():
    scheduler = _scheduler(seed=11)
    episode = scheduler._build_episode()
    plan = Stage3SchedulerAdapter(scheduler).plan_from_episode_v3(
        episode,
        "assimilation_timeline/seq10/entry0",
    )
    data = plan.to_json_dict()
    encoded = json.loads(json.dumps(data))
    restored = EpisodePlan.from_json_dict(encoded)

    assert restored.plan_id == plan.plan_id
    assert restored.to_json_dict()["plan_id"] == plan.plan_id
    assert restored.episode.scene_id == plan.episode.scene_id
    assert len(restored.events) == len(plan.events)
    assert restored.events[0].rollout_plan.sequence_id == plan.events[0].rollout_plan.sequence_id
    assert restored.events[0].rollout_plan.steps[0].sequence_pos == plan.events[0].rollout_plan.steps[0].sequence_pos


def test_episode_plan_id_is_stable_for_same_payload():
    scheduler = _scheduler(seed=12)
    episode = scheduler._build_episode()
    adapter = Stage3SchedulerAdapter(scheduler)
    first = adapter.plan_from_episode_v3(episode, "assimilation_timeline/seq10/entry0")
    second = EpisodePlan.from_json_dict(first.to_json_dict()).with_stable_plan_id()
    assert first.plan_id == second.plan_id
