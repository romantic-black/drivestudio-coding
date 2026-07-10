from __future__ import annotations

from models.iforward.protocols.validation_recipes import build_validation_v4_plans
from tests.test_iforward_stage2_3_scheduler import _Dataset
from tests.test_iforward_stage3_2_distributional_scheduler import _cfg


def test_distributional_validation_aliases_emit_stage3_2_plans():
    cfg = _cfg(repair_min=1, repair_max=1)
    cfg.iforward_validation_v4 = {
        "max_entries_debug": 1,
        "frame_sets": [{"name": "seq16", "target_frames": 16, "min_frames": 10, "allow_short": True}],
        "protocols": {
            "assimilation_timeline": False,
            "distribution_assimilation_timeline": True,
            "memory_ablation": False,
            "memory_ablation_by_distribution": True,
            "repair_before_after": False,
            "repair_tail_before_after": True,
            "order_robustness": False,
            "shuffle_order_robustness": True,
            "repeat_stability": False,
            "repeat_refine_stability": True,
        },
        "memory_ablation": ["full"],
    }

    plans = build_validation_v4_plans(cfg=cfg, dataset=_Dataset(frames=range(48)), max_entries=1, frame_sets=["seq16"])

    protocols = [plan.episode.protocol_name for plan in plans]
    assert any(name.startswith("distribution_assimilation_timeline/") for name in protocols)
    assert any(name.startswith("memory_ablation_by_distribution/") for name in protocols)
    assert any(name.startswith("repair_tail_before_after/") for name in protocols)
    assert any(name.startswith("shuffle_order_robustness/") for name in protocols)
    assert any(name.startswith("repeat_refine_stability/") for name in protocols)
    assert any(
        event.metadata.get("iforward_stage3_2", {}).get("distribution_type") == "high_block_repair"
        for plan in plans
        for event in plan.events
        if hasattr(event, "metadata")
    )


def test_distributional_validation_seq24_overrides_stage3_2_curriculum_frame_count():
    cfg = _cfg(repair_min=0, repair_max=0)
    cfg.iforward_validation_v4 = {
        "max_entries_debug": 1,
        "frame_sets": [{"name": "seq24", "target_frames": 24, "min_frames": 8, "allow_short": True}],
        "protocols": {
            "assimilation_timeline": True,
            "memory_ablation": False,
            "repair_before_after": False,
            "order_robustness": False,
            "repeat_stability": False,
        },
    }

    plans = build_validation_v4_plans(cfg=cfg, dataset=_Dataset(frames=range(48)), max_entries=1, frame_sets=["seq24"])

    assert plans
    plan = plans[0]
    assert plan.episode.metadata["frame_set"] == "seq24"
    assert len(plan.episode.frame_ids) == 24
    assert plan.events[0].rollout_plan.sequence_target_frames == 24
    assert plan.events[0].rollout_plan.sequence_min_frames == 8
    assert plan.events[0].metadata["sequence_length"] == 24


def test_memory_freeze_after_prefill_validation_plan_prefills_then_freezes():
    cfg = _cfg(repair_min=0, repair_max=0)
    cfg.iforward_validation_v4 = {
        "max_entries_debug": 1,
        "frame_sets": [{"name": "seq10", "target_frames": 10, "min_frames": 10, "allow_short": False}],
        "protocols": {
            "assimilation_timeline": False,
            "memory_ablation": True,
            "repair_before_after": False,
            "order_robustness": False,
            "repeat_stability": False,
        },
        "memory_ablation": ["memory_freeze_after_prefill"],
    }

    plans = build_validation_v4_plans(cfg=cfg, dataset=_Dataset(frames=range(48)), max_entries=1, frame_sets=["seq10"])

    assert len(plans) == 1
    modes = [getattr(event, "memory_mode", "") for event in plans[0].events if hasattr(event, "memory_mode")]
    assert modes[:-1]
    assert set(modes[:-1]) == {"full"}
    assert modes[-1] == "memory_freeze_after_prefill"
    assert [step.visit_kind for step in plans[0].events[-1].rollout_plan.steps] == ["assimilation"]
