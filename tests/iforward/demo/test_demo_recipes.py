from __future__ import annotations

import pytest

from models.iforward.protocols.demo_recipes import build_demo_v0_plans
from models.iforward.runtime.event import ControlEvent, ProbeEvent, UpdateEvent
from tests.test_iforward_stage2_3_scheduler import _Dataset


def _cfg():
    return {
        "scheduler_v3": {
            "time": {"allow_synthetic_timestamp": True},
            "bootstrap": {"end_step": 0},
            "sequence": {
                "min_frames": 8,
                "max_frames": 24,
                "min_unique_keyframes": 3,
                "min_frame_span": 8,
                "max_frame_span": 30,
            },
            "assimilation": {"max_inner_k": 12, "repeat_pairs": {"4,4": 1.0}},
            "repair": {"enable": False},
        },
        "iforward_demo": {
            "memory_ablation": [
                "full",
                "memory_off",
                "memory_read_write",
                "memory_freeze_write",
                "memory_shuffle_state",
            ]
        },
    }


def test_repair_showcase_24_respects_fixed_scene_segment_and_events():
    result = build_demo_v0_plans(
        cfg=_cfg(),
        dataset=_Dataset(scene_ids=(3,), segment_ids=(5,), frames=range(40)),
        recipe="repair_showcase_24",
        scene_id=3,
        segment_id=5,
        seed=31,
    )

    assert result.recipe == "repair_showcase_24"
    assert len(result.plans) == 1
    plan = result.plans[0]
    assert plan.episode.scene_id == 3
    assert plan.episode.segment_id == 5
    assert len(plan.episode.frame_ids) == 24
    assert any(isinstance(event, ControlEvent) and event.kind == "snapshot_state" for event in plan.events)
    assert any(isinstance(event, ProbeEvent) and event.event_id.endswith("probe_before_repair") for event in plan.events)
    assert any(isinstance(event, ProbeEvent) and event.event_id.endswith("probe_after_repair") for event in plan.events)
    assert any(isinstance(event, UpdateEvent) and event.kind == "repair_update" for event in plan.events)


def test_repair_showcase_20_uses_twenty_frame_plan():
    result = build_demo_v0_plans(
        cfg=_cfg(),
        dataset=_Dataset(scene_ids=(3,), segment_ids=(5,), frames=range(40)),
        recipe="repair_showcase_20",
        scene_id=3,
        segment_id=5,
        seed=31,
    )

    assert result.recipe == "repair_showcase_20"
    assert len(result.plans) == 1
    assert len(result.plans[0].episode.frame_ids) == 20


def test_memory_ablation_showcase_emits_one_plan_per_mode():
    result = build_demo_v0_plans(
        cfg=_cfg(),
        dataset=_Dataset(scene_ids=(1,), segment_ids=(0,), frames=range(40)),
        recipe="memory_ablation_showcase",
        scene_id=1,
        segment_id=0,
        seed=32,
        memory_ablation=["full", "memory_off", "memory_read_write"],
    )

    assert len(result.plans) == 3
    modes = [str(plan.metadata.get("memory_mode", "")) for plan in result.plans]
    assert modes == ["full", "memory_off", "memory_read_write"]
    assert all(plan.source == "demo_recipe" for plan in result.plans)


def test_memory_ablation_showcase_freeze_after_prefill_uses_full_prefill():
    result = build_demo_v0_plans(
        cfg=_cfg(),
        dataset=_Dataset(scene_ids=(1,), segment_ids=(0,), frames=range(40)),
        recipe="memory_ablation_showcase",
        scene_id=1,
        segment_id=0,
        seed=32,
        memory_ablation=["memory_freeze_after_prefill"],
    )

    assert len(result.plans) == 1
    modes = [getattr(event, "memory_mode", "") for event in result.plans[0].events if hasattr(event, "memory_mode")]
    assert modes[:-1]
    assert set(modes[:-1]) == {"full"}
    assert modes[-1] == "memory_freeze_after_prefill"
    assert [step.visit_kind for step in result.plans[0].events[-1].rollout_plan.steps] == ["assimilation"]


def test_demo_recipe_rejects_unknown_recipe():
    with pytest.raises(ValueError, match="unsupported"):
        build_demo_v0_plans(
            cfg=_cfg(),
            dataset=_Dataset(scene_ids=(1,), segment_ids=(0,), frames=range(40)),
            recipe="interactive_viewer",
            scene_id=1,
            segment_id=0,
        )
