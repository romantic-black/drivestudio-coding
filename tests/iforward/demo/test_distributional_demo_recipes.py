from __future__ import annotations

from models.iforward.protocols.demo_recipes import build_demo_v0_plans
from models.iforward.runtime.event import ProbeEvent, UpdateEvent
from tests.test_iforward_stage2_3_scheduler import _Dataset
from tests.test_iforward_stage3_2_distributional_scheduler import _cfg


def test_distributional_demo_showcase_contains_distribution_metadata():
    cfg = _cfg(repair_min=1, repair_max=1)
    cfg.iforward_demo = {"memory_ablation": ["full", "memory_off"]}

    result = build_demo_v0_plans(
        cfg=cfg,
        dataset=_Dataset(scene_ids=(1,), segment_ids=(0,), frames=range(48)),
        recipe="distributional_episode_showcase",
        scene_id=1,
        segment_id=0,
        seed=41,
    )

    assert result.recipe == "distributional_episode_showcase"
    plan = result.plans[0]
    assert any(isinstance(event, ProbeEvent) and event.event_id.endswith("probe_before_repair") for event in plan.events)
    distribution_types = {
        event.metadata.get("iforward_stage3_2", {}).get("distribution_type")
        for event in plan.events
        if isinstance(event, UpdateEvent)
    }
    assert {"repeat_refine", "shuffled_coverage", "high_block_repair"}.issubset(distribution_types)


def test_memory_ablation_distribution_showcase_emits_modes():
    cfg = _cfg(repair_min=1, repair_max=1)
    cfg.iforward_demo = {"memory_ablation": ["full", "memory_off"]}

    result = build_demo_v0_plans(
        cfg=cfg,
        dataset=_Dataset(scene_ids=(1,), segment_ids=(0,), frames=range(48)),
        recipe="memory_ablation_distribution_showcase",
        scene_id=1,
        segment_id=0,
        seed=42,
    )

    assert [plan.metadata.get("memory_mode") for plan in result.plans] == ["full", "memory_off"]
    assert all(plan.metadata.get("recipe") == "memory_ablation_distribution_showcase" for plan in result.plans)
