from __future__ import annotations

import json

from models.iforward.demo.report_builder import build_demo_report
from models.iforward.protocols.demo_recipes import build_demo_v0_plans, make_demo_scheduler
from tests.iforward.demo.test_demo_recipes import _cfg
from tests.test_iforward_stage2_3_scheduler import _Dataset
from tests.test_iforward_stage2_3_validation import _FakeModel


def test_demo_report_builder_writes_static_report_for_repair_showcase(tmp_path):
    dataset = _Dataset(scene_ids=(1,), segment_ids=(0,), frames=range(40))
    cfg = _cfg()
    plans = build_demo_v0_plans(
        cfg=cfg,
        dataset=dataset,
        recipe="repair_showcase_24",
        scene_id=1,
        segment_id=0,
        seed=41,
    ).plans
    scheduler = make_demo_scheduler(cfg=cfg, dataset=dataset, scene_id=1, segment_id=0, seed=41)
    result = build_demo_report(
        recipe="repair_showcase_24",
        plans=plans,
        model=_FakeModel(),
        scheduler=scheduler,
        output_dir=tmp_path,
        device="cpu",
        trigger_step=7,
    )

    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "trace.jsonl").is_file()
    assert (tmp_path / "plan.json").is_file()
    assert list((tmp_path / "plans").glob("repair_showcase_24_*.json"))
    assert result.traces
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["recipe"] == "repair_showcase_24"
    assert "questions" in summary
    assert summary["questions"]["repeat_stability"] == "not run"


def test_demo_report_builder_summarizes_memory_ablation_modes(tmp_path):
    dataset = _Dataset(scene_ids=(1,), segment_ids=(0,), frames=range(40))
    cfg = _cfg()
    plans = build_demo_v0_plans(
        cfg=cfg,
        dataset=dataset,
        recipe="memory_ablation_showcase",
        scene_id=1,
        segment_id=0,
        seed=42,
        memory_ablation=["full", "memory_off"],
    ).plans
    scheduler = make_demo_scheduler(cfg=cfg, dataset=dataset, scene_id=1, segment_id=0, seed=42)
    build_demo_report(
        recipe="memory_ablation_showcase",
        plans=plans,
        model=_FakeModel(),
        scheduler=scheduler,
        output_dir=tmp_path,
        device="cpu",
        trigger_step=7,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    modes = {str(row.get("mode", "")) for row in summary["protocols"]}
    assert {"full", "memory_off"} <= modes
