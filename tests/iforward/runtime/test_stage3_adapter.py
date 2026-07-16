from __future__ import annotations

from models.iforward.protocols.validation_recipes import build_validation_v4_plans
from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.runner import IForwardRunner, RunnerOptions
from models.iforward.runtime.trace import TraceRecorder
from tests.test_iforward_stage2_3_scheduler import _Dataset, _scheduler
from tests.test_iforward_stage2_3_validation import _FakeModel


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
        "iforward_validation_v4": {
            "max_entries_debug": 1,
            "frame_sets": [
                {"name": "seq10", "target_frames": 10, "min_frames": 10, "allow_short": False},
                {"name": "seq24", "target_frames": 24, "min_frames": 8, "allow_short": True},
            ],
            "protocols": {"assimilation_timeline": True, "memory_ablation": False, "repair_before_after": False, "order_robustness": False, "repeat_stability": False},
        },
    }


def test_stage3_adapter_converts_episode_and_materializes_batch():
    scheduler = _scheduler(seed=13)
    episode = scheduler._build_episode()
    adapter = Stage3SchedulerAdapter(scheduler)
    plan = adapter.plan_from_episode_v3(episode, "assimilation_timeline/seq10/entry0")
    batch = adapter.batch_from_rollout_plan(plan.events[0].rollout_plan)

    assert plan.plan_id
    assert plan.events[0].metadata["scheduler_phase"] == "assimilation"
    assert batch["_iforward"]["sequence_id"] == plan.events[0].rollout_plan.sequence_id


def test_validation_v4_seq24_recipe_forces_long_frame_set():
    plans = build_validation_v4_plans(cfg=_cfg(), dataset=_Dataset(frames=range(40)), max_entries=1, frame_sets=["seq24"])
    assert plans
    assert len(plans[0].episode.frame_ids) == 24
    assert plans[0].events[0].rollout_plan.sequence_target_frames == 24


def test_runner_executes_validation_plan_with_fake_model(tmp_path):
    scheduler = _scheduler(seed=14)
    episode = scheduler._build_episode()
    adapter = Stage3SchedulerAdapter(scheduler)
    plan = adapter.plan_from_episode_v3(episode, "assimilation_timeline/seq10/entry0")
    model = _FakeModel()
    recorder = TraceRecorder(tmp_path, record_images=False)
    trace = IForwardRunner(model, adapter).run(
        plan,
        recorder,
        RunnerOptions.for_mode("validate", device="cpu", trigger_step=7),
    )

    assert trace.events
    assert model.calls
    assert model.calls[0]["ablation"] == "full"
    assert (tmp_path / "plan.json").is_file()
    assert (tmp_path / "trace.jsonl").is_file()


def test_runner_marks_all_non_training_batches_as_read_only_feedback():
    runner = IForwardRunner(
        model=None,
        convert_batch_to_minimal_format=lambda raw, _device, _step: raw,
    )
    raw = {
        "request_meta": {
            "iforward_stage3_2": {
                "distribution_type": "shuffled_coverage",
                "train_2d_mode": "trainable_checkpointed",
            }
        },
        "_iforward": {"request_meta": {}},
    }

    batch = runner._convert(
        raw,
        RunnerOptions.for_mode("validate", device="cpu", trigger_step=9999),
    )

    assert batch["request_meta"]["observation_feedback_eval_mode"] == "frozen_no_grad"
    assert batch["_iforward"]["request_meta"]["observation_feedback_eval_mode"] == "frozen_no_grad"
    assert batch["request_meta"]["iforward_stage3_2"]["distribution_type"] == "shuffled_coverage"
