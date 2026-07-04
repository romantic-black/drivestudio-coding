from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from datasets.iforward_stage2_3.index_builder import build_stage2_3_index_from_dataset
from datasets.iforward_stage2_3.distributional_episode import _clamp_b_r_for_max_k
from datasets.iforward_stage2_3.scheduler import IFORWARD_STAGE3_2_SCHEDULER_VERSION, Stage23Scheduler
from tests.test_iforward_stage2_3_scheduler import _Dataset


def _cfg(*, repair_min: int = 1, repair_max: int = 1):
    return OmegaConf.create(
        {
            "scheduler_stage3_0": {
                "enable": True,
                "version": "stage3_0_optimizer_sequence_v1",
                "time": {"allow_synthetic_timestamp": True},
                "bootstrap": {"end_step": 0, "frames_per_asset_pack": 4},
                "sequence": {"min_frames": 8, "max_frames": 24, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
                "assimilation": {"max_inner_k": 8, "rollout_options": {"B2R4": 1.0}},
                "repair": {"enable": True, "last_update_write": False, "max_inner_k": 20, "rollout_options": {"B6R1": 1.0}},
            },
            "scheduler_stage3_2": {
                "enable": True,
                "version": "stage3_2_distributional_episode_v1",
                "inherit_from": "scheduler_stage3_0",
                "curriculum": [
                    {
                        "name": "test",
                        "start_step": 0,
                        "end_step": 100,
                        "sequence_target_frames": 16,
                        "min_frames": 10,
                        "allow_short": True,
                        "weights": {"repeat_refine": 0.25, "shuffled_coverage": 0.50, "high_block_repair": 1.0},
                        "max_k": {
                            "train_2d": {"repeat_refine": 8, "shuffled_coverage": 10, "high_block_repair": 12},
                            "frozen_2d": {"repeat_refine": 12, "shuffled_coverage": 16, "high_block_repair": 20},
                        },
                    }
                ],
                "episode_recipe": {
                    "prelude": {"min_rollouts": 2, "max_rollouts": 3, "order_policy": "mixed_random"},
                    "repair_tail": {"min_rollouts": repair_min, "max_rollouts": repair_max},
                    "train_2d_policy": {
                        "repeat_refine": "trainable",
                        "shuffled_coverage": "trainable",
                        "high_block_repair": "frozen_no_grad",
                    },
                },
            },
        }
    )


def _scheduler(seed: int = 17, *, repair_min: int = 1, repair_max: int = 1) -> Stage23Scheduler:
    ds = _Dataset(frames=range(48))
    cfg = _cfg(repair_min=repair_min, repair_max=repair_max)
    index = build_stage2_3_index_from_dataset(dataset=ds, cfg=cfg)
    return Stage23Scheduler(dataset=ds, cfg=cfg, index=index, seed=seed)


def _stage32(rollout):
    return dict((dict(rollout.request_meta).get("iforward_stage3_2") or {}))


def test_stage3_2_parser_rejects_repeat_refine_b_above_two():
    ds = _Dataset(frames=range(48))
    cfg = _cfg()
    cfg.scheduler_stage3_2.distributions = {"repeat_refine": {"b_choices": {3: 1.0}}}
    index = build_stage2_3_index_from_dataset(dataset=ds, cfg=cfg)

    with pytest.raises(ValueError, match="repeat_refine"):
        Stage23Scheduler(dataset=ds, cfg=cfg, index=index, seed=11)


def test_stage3_2_high_block_repair_clamp_preserves_b_first():
    assert _clamp_b_r_for_max_k(8, 2, 15, 24, preserve_b=True) == (8, 1, "preserve_b_reduce_r")
    assert _clamp_b_r_for_max_k(12, 2, 15, 24, preserve_b=True) == (12, 1, "preserve_b_reduce_r")
    assert _clamp_b_r_for_max_k(20, 2, 15, 24, preserve_b=True) == (15, 1, "preserve_b_reduce_r")
    assert _clamp_b_r_for_max_k(8, 2, 15, 24) == (7, 2, "reduce_b")


def test_stage3_2_samples_repeat_shuffle_and_repair_with_constraints():
    sched = _scheduler(seed=19)
    episode = sched._build_episode()

    assert sched.scheduler_version == IFORWARD_STAGE3_2_SCHEDULER_VERSION
    metas = [_stage32(rollout) for rollout in episode.rollouts]
    dists = {meta["distribution_type"] for meta in metas}
    assert {"repeat_refine", "shuffled_coverage", "high_block_repair"}.issubset(dists)

    for rollout, meta in zip(episode.rollouts, metas):
        assert meta["K"] <= meta["maxK"]
        assert rollout.requested_inner_K == meta["K"]
        if meta["distribution_type"] == "repeat_refine":
            assert meta["B"] <= 2
            assert rollout.scheduler_phase == "assimilation"
            assert set(rollout.visit_kinds) == {"assimilation"}
        if meta["distribution_type"] == "shuffled_coverage":
            assert rollout.scheduler_phase == "assimilation"
            assert set(rollout.visit_kinds) == {"assimilation"}
        if meta["distribution_type"] == "high_block_repair":
            assert rollout.scheduler_phase == "repair"
            assert meta["train_2d_mode"] == "frozen_no_grad"
            assert meta["repair_visited_ratio"] > 0.0
            assert "raw_B" in meta
            assert "raw_R" in meta
            assert "R" in meta
            assert meta["clamp_strategy"] in {"none", "cap_b", "preserve_b_reduce_r"}
            assert rollout.steps[-1].optimizer_memory_write is False


def test_stage3_2_state_restore_replays_same_rollout_metadata():
    sched = _scheduler(seed=23, repair_min=0, repair_max=0)
    state = sched.state_dict()
    first = sched.next_batch()["_iforward"]
    restored = _scheduler(seed=999, repair_min=0, repair_max=0)
    restored.load_state_dict(state)
    second = restored.next_batch()["_iforward"]

    assert first["scheduler_version"] == IFORWARD_STAGE3_2_SCHEDULER_VERSION
    assert first["request_meta"]["iforward_stage3_2"] == second["request_meta"]["iforward_stage3_2"]
    assert first["rollout_positions"] == second["rollout_positions"]


def test_stage3_2_producer_clone_preserves_distributional_compiler():
    sched = _scheduler(seed=29, repair_min=0, repair_max=0)
    state = sched.state_dict()
    clone = sched._make_producer_clone(state)
    meta = clone.next_batch()["_iforward"]

    assert clone.scheduler_version == IFORWARD_STAGE3_2_SCHEDULER_VERSION
    assert meta["request_meta"]["iforward_stage3_2"]["distribution_type"] in {"repeat_refine", "shuffled_coverage"}
