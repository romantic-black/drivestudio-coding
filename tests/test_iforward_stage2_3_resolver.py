from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from datasets.iforward_stage2_3.index_builder import build_stage2_3_index_from_dataset
from datasets.iforward_stage2_3.resolver import Stage23BatchResolver
from datasets.iforward_stage2_3.scheduler import IFORWARD_STAGE3_0_SCHEDULER_VERSION, Stage23Scheduler
from tests.test_iforward_stage2_3_scheduler import _Dataset, _omegaconf_scheduler, _scheduler


def test_stage2_3_resolver_accepts_scheduler_batch():
    sched = _scheduler(seed=2)
    resolved = Stage23BatchResolver().resolve(sched.next_batch())
    assert resolved.scheduler_version == "iforward_2_3_scheduler_v3_optimizer_mamba"
    assert resolved.steps[0].repeat_budget >= 1
    assert resolved.steps[0].optimizer_memory_write is True


def test_stage2_3_resolver_accepts_stage3_0_scheduler_batch():
    ds = _Dataset()
    cfg = OmegaConf.create(
        {
            "scheduler_stage3_0": {
                "enable": True,
                "version": "stage3_0_optimizer_sequence_v1",
                "time": {"allow_synthetic_timestamp": True},
                "bootstrap": {"end_step": 0, "frames_per_asset_pack": 4},
                "sequence": {"min_frames": 8, "max_frames": 10, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
                "assimilation": {"max_inner_k": 8, "rollout_options": {"B1R8": 1.0}},
                "repair": {"enable": False},
            }
        }
    )
    index = build_stage2_3_index_from_dataset(dataset=ds, cfg=cfg)
    sched = Stage23Scheduler(dataset=ds, cfg=cfg, index=index, seed=32)
    resolved = Stage23BatchResolver().resolve(sched.next_batch())
    assert resolved.scheduler_version == IFORWARD_STAGE3_0_SCHEDULER_VERSION
    assert resolved.inner_K == 8


def test_stage2_3_resolver_rejects_bootstrap_mamba_write():
    sched = _scheduler(bootstrap_end=1, seed=2)
    batch = sched.next_batch()
    batch["_iforward"]["steps"][0]["optimizer_memory_write"] = True
    batch["_iforward"]["steps"][0]["update_optimizer_memory"] = True
    batch["request_meta"]["iforward"]["steps"][0]["optimizer_memory_write"] = True
    batch["request_meta"]["iforward"]["steps"][0]["update_optimizer_memory"] = True
    with pytest.raises(ValueError, match="bootstrap"):
        Stage23BatchResolver().resolve(batch)


def test_stage2_3_resolver_rejects_assimilation_missing_write():
    sched = _scheduler(seed=3)
    batch = sched.next_batch()
    batch["_iforward"]["steps"][0]["optimizer_memory_write"] = False
    batch["_iforward"]["steps"][0]["update_optimizer_memory"] = False
    batch["request_meta"]["iforward"]["steps"][0]["optimizer_memory_write"] = False
    batch["request_meta"]["iforward"]["steps"][0]["update_optimizer_memory"] = False
    with pytest.raises(ValueError, match="assimilation"):
        Stage23BatchResolver().resolve(batch)


def test_stage2_3_resolver_rejects_missing_rollout_positions_metadata():
    sched = _scheduler(seed=4)
    batch = sched.next_batch()
    batch["_iforward"].pop("rollout_positions", None)
    batch["request_meta"]["iforward"].pop("rollout_positions", None)
    with pytest.raises(ValueError, match="rollout_positions"):
        Stage23BatchResolver().resolve(batch)


def test_stage2_3_resolver_accepts_repair_phase_cap_16():
    sched = _omegaconf_scheduler(
        {
            "time": {"allow_synthetic_timestamp": True},
            "bootstrap": {"end_step": 0, "frames_per_asset_pack": 4},
            "sequence": {"min_frames": 8, "max_frames": 8, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
            "assimilation": {"max_inner_k": 8, "repeat_pairs": {"4,4": 1.0}},
            "repair": {
                "enable": True,
                "start_step": 0,
                "probability_schedule": [[0, 1.0]],
                "rounds": {"1": 1.0},
                "rollout_options": {"B4R4": 1.0},
                "last_update_write": False,
                "max_inner_k": 16,
            },
        },
        seed=46,
    )
    repair = None
    while True:
        batch = sched.next_batch()
        if batch["_iforward"]["scheduler_phase"] == "repair":
            repair = batch
        if batch["_iforward"]["episode_end_after_rollout"]:
            break
    assert repair is not None
    resolved = Stage23BatchResolver().resolve(repair)
    assert resolved.inner_K == 16

    repair["_iforward"]["phase_max_inner_k"] = 12
    repair["request_meta"]["iforward"]["phase_max_inner_k"] = 12
    repair["request_meta"]["iforward_stage2_3"]["phase_max_inner_k"] = 12
    with pytest.raises(ValueError, match="repair inner_K.*phase_max_inner_k"):
        Stage23BatchResolver().resolve(repair)
