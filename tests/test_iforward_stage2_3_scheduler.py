from __future__ import annotations

import dataclasses
import threading
import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from datasets.iforward_stage2_3.index_builder import build_stage2_3_index_from_dataset
from datasets.iforward_stage2_3.scheduler import IFORWARD_STAGE2_3_SCHEDULER_VERSION, Stage23Scheduler


class _Dataset:
    _initialized = True

    def __init__(self, *, scene_ids=(1,), segment_ids=(0,), frames=range(40), num_cams=3):
        self.scene_ids = [int(x) for x in scene_ids]
        self.segment_ids = [int(x) for x in segment_ids]
        self.frames = [int(x) for x in frames]
        self.num_cams = int(num_cams)
        self.preload_hints = []
        self.active_scopes = []
        self.training_scopes = []
        self.cleared_scopes = 0

    def list_training_scene_ids(self):
        return list(self.scene_ids)

    def list_segment_ids(self, scene_id):
        return list(self.segment_ids)

    def get_segment_index(self, scene_id, segment_id):
        return SimpleNamespace(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            num_cams=self.num_cams,
            frame_indices=list(self.frames),
            train_frame_set=set(self.frames),
            test_frame_indices=[],
            test_frame_set=set(),
            keyframe_indices=list(range(len(self.frames))),
            frame_to_keyframe={int(f): int(i) for i, f in enumerate(self.frames)},
            keyframe_to_frames={int(i): [int(f)] for i, f in enumerate(self.frames)},
            train_image_refs=tuple((int(f), int(c)) for f in self.frames for c in range(self.num_cams)),
            frame_timestamps_us={int(f): int(f) * 100000 for f in self.frames},
        )

    def _assemble_segment_batch_from_iforward_request(self, *, scene_id, segment_id, plan, include_test=False):
        ifwd = dataclasses.asdict(plan)
        request_meta = dict(plan.request_meta)
        request_meta.update(
            {
                "assembly_mode": "image_ref_iforward_v1",
                "source_image_refs": [tuple(x) for x in plan.evidence_refs_flat],
                "target_image_refs": [tuple(x) for x in plan.target_refs_flat],
                "target_image_roles": [str(x) for x in plan.target_roles_flat],
                "iforward": ifwd,
            }
        )
        return {"request_meta": request_meta, "_iforward": ifwd}

    def build_preload_hint_light(self, *, scene_id, segment_id, future_image_refs, scope):
        return {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "future_image_refs": [tuple(x) for x in future_image_refs],
            "scope": str(scope),
        }

    def submit_preload_hint(self, *, hint, hint_scope, epoch_idx, global_step, block_idx_global, include_test):
        self.preload_hints.append(
            {
                "hint": dict(hint),
                "hint_scope": str(hint_scope),
                "epoch_idx": int(epoch_idx),
                "global_step": int(global_step),
                "block_idx_global": int(block_idx_global),
                "include_test": bool(include_test),
            }
        )

    def set_preload_active_scope(self, scene_id, segment_id):
        self.active_scopes.append((int(scene_id), int(segment_id)))

    def set_preload_training_scope(self, scene_id, segment_id):
        self.training_scopes.append((int(scene_id), int(segment_id)))

    def clear_preload_scheduler_scope(self):
        self.cleared_scopes += 1


def _scheduler(**kwargs):
    ds = kwargs.pop("dataset", _Dataset())
    cfg = {
        "scheduler_v3": {
            "time": {"allow_synthetic_timestamp": True},
            "bootstrap": {"end_step": int(kwargs.get("bootstrap_end", 0)), "frames_per_asset_pack": 4},
            "sequence": {
                "min_frames": 8,
                "max_frames": 10,
                "min_unique_keyframes": 3,
                "min_frame_span": 8,
                "max_frame_span": 30,
            },
            "assimilation": {"max_inner_k": 12, "repeat_pairs": {"4,4": 1.0}},
            "repair": {
                "enable": bool(kwargs.get("repair_enable", False)),
                "start_step": int(kwargs.get("repair_start", 0)),
                "probability_schedule": [[0, float(kwargs.get("repair_prob", 0.0))]],
                "rounds": {1: 1.0},
                "rollout_options": {"B6R1": 1.0},
                "last_update_write": False,
                "max_inner_k": 12,
            },
        }
    }
    producer = kwargs.get("producer", None)
    if producer is not None:
        cfg["scheduler_v3"]["producer"] = dict(producer)
    index = build_stage2_3_index_from_dataset(dataset=ds, cfg=cfg)
    return Stage23Scheduler(dataset=ds, cfg=cfg, index=index, seed=int(kwargs.get("seed", 3)))


def _omegaconf_scheduler(scheduler_v3, *, seed=3):
    ds = _Dataset()
    cfg = OmegaConf.create({"scheduler_v3": scheduler_v3})
    index = build_stage2_3_index_from_dataset(dataset=ds, cfg=cfg)
    return Stage23Scheduler(dataset=ds, cfg=cfg, index=index, seed=int(seed))


def test_stage2_3_scheduler_assimilation_inner_k_and_every_repeat_write():
    sched = _scheduler(seed=4)
    meta = sched.next_batch()["_iforward"]
    assert meta["scheduler_version"] == IFORWARD_STAGE2_3_SCHEDULER_VERSION
    assert meta["scheduler_phase"] == "assimilation"
    assert meta["inner_K"] <= 12
    assert 1 <= len(meta["sequence_positions"]) <= 2
    assert all(s["optimizer_memory_read"] for s in meta["steps"])
    assert all(s["optimizer_memory_write"] for s in meta["steps"])
    assert all(s["update_optimizer_memory"] for s in meta["steps"])
    assert meta["rollout_positions"] == meta["sequence_positions"]
    assert meta["actual_blocks_per_rollout"] == len(meta["rollout_positions"])
    assert len(meta["repeat_budgets"]) == len(meta["rollout_positions"])
    assert len(meta["frame_gaps"]) == len(meta["steps"])


def test_stage2_3_scheduler_bootstrap_asset_pack_fresh_state_no_mamba():
    sched = _scheduler(bootstrap_end=4, seed=5)
    metas = [sched.next_batch()["_iforward"] for _ in range(4)]
    assert all(m["scheduler_phase"] == "bootstrap" for m in metas)
    assert len({int(m["segment_id"]) for m in metas}) == 1
    assert all(m["reset_scene_state_before_rollout"] for m in metas)
    assert all(not m["carry_scene_state_after_rollout"] for m in metas)
    assert all(not any(s["optimizer_memory_read"] or s["optimizer_memory_write"] for s in m["steps"]) for m in metas)


def test_stage2_3_scheduler_repair_random_and_last_write_optional():
    sched = _scheduler(repair_enable=True, repair_prob=1.0, seed=6)
    metas = []
    while True:
        meta = sched.next_batch()["_iforward"]
        metas.append(meta)
        if meta["episode_end_after_rollout"]:
            break
    repair = [m for m in metas if m["scheduler_phase"] == "repair"][-1]
    assert repair["inner_K"] <= 12
    assert len(set(repair["sequence_positions"])) == len(repair["sequence_positions"])
    assert repair["steps"][-1]["optimizer_memory_write"] is False
    assert all(s["optimizer_memory_read"] for s in repair["steps"])


def test_stage2_3_scheduler_omegaconf_bootstrap_repeat_distribution():
    sched = _omegaconf_scheduler(
        {
            "time": {"allow_synthetic_timestamp": True},
            "bootstrap": {"end_step": 1, "frames_per_asset_pack": 4, "repeat_distribution": {"8": 1.0}},
            "sequence": {"min_frames": 8, "max_frames": 10, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
            "assimilation": {"start_step": 1, "max_inner_k": 12, "repeat_pairs": {"4,4": 1.0}},
            "repair": {"enable": False},
        },
        seed=41,
    )
    meta = sched.next_batch()["_iforward"]
    assert meta["scheduler_phase"] == "bootstrap"
    assert meta["repeat_budgets"] == [8]
    assert meta["inner_K"] == 8


def test_stage2_3_scheduler_omegaconf_repeat_pairs_are_not_defaulted():
    sched = _omegaconf_scheduler(
        {
            "time": {"allow_synthetic_timestamp": True},
            "bootstrap": {"end_step": 0, "frames_per_asset_pack": 4},
            "sequence": {"min_frames": 8, "max_frames": 10, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
            "assimilation": {"max_inner_k": 12, "repeat_pairs": {"4,6": 1.0}},
            "repair": {"enable": False},
        },
        seed=42,
    )
    meta = sched.next_batch()["_iforward"]
    assert meta["scheduler_phase"] == "assimilation"
    assert meta["repeat_budgets"] == [4, 6]
    assert meta["inner_K"] == 10


def test_stage2_3_scheduler_omegaconf_repair_options_and_two_rounds():
    sched = _omegaconf_scheduler(
        {
            "time": {"allow_synthetic_timestamp": True},
            "bootstrap": {"end_step": 0, "frames_per_asset_pack": 4},
            "sequence": {"min_frames": 8, "max_frames": 10, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
            "assimilation": {"max_inner_k": 12, "repeat_pairs": {"4,4": 1.0}},
            "repair": {
                "enable": True,
                "start_step": 0,
                "probability_schedule": [[0, 1.0]],
                "rounds": {"2": 1.0},
                "rollout_options": {"B8R1": 1.0},
                "last_update_write": False,
                "max_inner_k": 12,
            },
        },
        seed=43,
    )
    repairs = []
    while True:
        meta = sched.next_batch()["_iforward"]
        if meta["scheduler_phase"] == "repair":
            repairs.append(meta)
        if meta["episode_end_after_rollout"]:
            break
    assert len(repairs) == 2
    assert [int(m["repair_round_idx"]) for m in repairs] == [0, 1]
    assert all(m["repair_pattern_name"] == "B8R1" for m in repairs)
    assert all(len(m["rollout_positions"]) == 8 for m in repairs)
    assert all(m["repeat_budgets"] == [1] * 8 for m in repairs)


def test_stage2_3_scheduler_omegaconf_repair_b6r2_pattern():
    sched = _omegaconf_scheduler(
        {
            "time": {"allow_synthetic_timestamp": True},
            "bootstrap": {"end_step": 0, "frames_per_asset_pack": 4},
            "sequence": {"min_frames": 8, "max_frames": 10, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
            "assimilation": {"max_inner_k": 12, "repeat_pairs": {"4,4": 1.0}},
            "repair": {
                "enable": True,
                "start_step": 0,
                "probability_schedule": [[0, 1.0]],
                "rounds": {"1": 1.0},
                "rollout_options": {"B6R2": 1.0},
                "last_update_write": False,
                "max_inner_k": 12,
            },
        },
        seed=44,
    )
    repair = None
    while True:
        meta = sched.next_batch()["_iforward"]
        if meta["scheduler_phase"] == "repair":
            repair = meta
        if meta["episode_end_after_rollout"]:
            break
    assert repair is not None
    assert repair["repair_pattern_name"] == "B6R2"
    assert len(repair["rollout_positions"]) == 6
    assert repair["repeat_budgets"] == [2] * 6
    assert repair["inner_K"] == 12


def test_stage2_3_scheduler_resume_determinism():
    sched = _scheduler(seed=9, repair_enable=True, repair_prob=1.0)
    first = sched.next_batch()["_iforward"]
    state = sched.state_dict()
    restored = _scheduler(seed=9, repair_enable=True, repair_prob=1.0)
    restored.load_state_dict(state)
    assert sched.next_batch()["_iforward"] == restored.next_batch()["_iforward"]
    assert first["scheduler_phase"] == "assimilation"


def test_stage2_3_scheduler_frame_gap_is_episode_global_across_rollouts():
    sched = _scheduler(seed=12)
    first = sched.next_batch()["_iforward"]
    second = sched.next_batch()["_iforward"]
    assert first["episode_id"] == second["episode_id"]
    last_first = first["steps"][-1]
    first_second = second["steps"][0]
    assert first_second["previous_visit_sequence_pos"] == last_first["sequence_pos"]
    assert first_second["frame_gap"] == int(first_second["source_frame_idx"]) - int(last_first["source_frame_idx"])
    assert first_second["physical_frame_gap_abs"] == abs(int(first_second["frame_gap"]))
    repeated = [s for s in second["steps"] if int(s["repeat_idx"]) > 0]
    assert repeated
    assert all(int(s["frame_gap"]) == 0 for s in repeated)
    assert all(float(s["delta_t_sec"]) == 0.0 for s in repeated)


def test_stage2_3_scheduler_assimilation_start_must_match_bootstrap_end():
    ds = _Dataset()
    cfg = {
        "scheduler_v3": {
            "bootstrap": {"end_step": 20},
            "sequence": {"min_frames": 8, "max_frames": 10, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
            "assimilation": {"start_step": 5000, "repeat_pairs": {"4,4": 1.0}},
            "repair": {"enable": False},
        }
    }
    index = build_stage2_3_index_from_dataset(dataset=ds, cfg=cfg)
    with pytest.raises(ValueError, match="assimilation.start_step"):
        Stage23Scheduler(dataset=ds, cfg=cfg, index=index)


def test_stage2_3_scheduler_repair_prob_and_schedule_are_mutually_exclusive():
    ds = _Dataset()
    cfg = {
        "scheduler_v3": {
            "bootstrap": {"end_step": 0},
            "sequence": {"min_frames": 8, "max_frames": 10, "min_unique_keyframes": 3, "min_frame_span": 8, "max_frame_span": 30},
            "assimilation": {"repeat_pairs": {"4,4": 1.0}},
            "repair": {"enable": True, "start_step": 200, "prob": 0.5, "probability_schedule": [[200, 0.5]]},
        }
    }
    index = build_stage2_3_index_from_dataset(dataset=ds, cfg=cfg)
    with pytest.raises(ValueError, match="mutually exclusive"):
        Stage23Scheduler(dataset=ds, cfg=cfg, index=index)


def test_stage2_3_producer_matches_sync_metadata():
    sync = _scheduler(seed=21, producer={"enable": False, "queue_depth": 0})
    prod = _scheduler(seed=21, producer={"enable": True, "queue_depth": 2, "preload_next_episode": False})
    try:
        for _ in range(6):
            assert prod.next_batch()["_iforward"] == sync.next_batch()["_iforward"]
            info = prod.get_current_info()
            assert info["producer_enabled"] is True
            assert int(info["producer_queue_depth"]) == 2
            assert int(info["producer_batches_produced"]) >= 1
    finally:
        sync.shutdown()
        prod.shutdown()


def test_stage2_3_producer_state_dict_resume_is_consumed_state_only():
    prod = _scheduler(seed=22, producer={"enable": True, "queue_depth": 4, "preload_next_episode": False})
    restored = _scheduler(seed=22, producer={"enable": True, "queue_depth": 4, "preload_next_episode": False})
    try:
        _ = prod.next_batch()
        state = prod.state_dict()
        restored.load_state_dict(state)
        assert prod.next_batch()["_iforward"] == restored.next_batch()["_iforward"]
    finally:
        prod.shutdown()
        restored.shutdown()


def test_stage2_3_producer_load_state_clears_prefetch_queue():
    prod = _scheduler(seed=23, producer={"enable": True, "queue_depth": 4, "preload_next_episode": False})
    try:
        _ = prod.next_batch()
        deadline = time.time() + 2.0
        while time.time() < deadline and prod._producer_queue is not None and prod._producer_queue.qsize() == 0:
            time.sleep(0.01)
        state = prod.state_dict()
        prod.load_state_dict(state)
        assert prod._producer_queue is not None
        assert prod._producer_queue.qsize() == 0
    finally:
        prod.shutdown()


def test_stage2_3_producer_worker_exception_is_raised_and_thread_stops():
    class _BrokenDataset(_Dataset):
        def _assemble_segment_batch_from_iforward_request(self, *, scene_id, segment_id, plan, include_test=False):
            raise RuntimeError("broken assembler")

    prod = _scheduler(
        dataset=_BrokenDataset(),
        seed=24,
        producer={"enable": True, "queue_depth": 2, "preload_next_episode": False},
    )
    with pytest.raises(RuntimeError, match="producer worker failed"):
        prod.next_batch()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any(t.name == "Stage23Producer" and t.is_alive() for t in threading.enumerate()):
            break
        time.sleep(0.01)
    assert not any(t.name == "Stage23Producer" and t.is_alive() for t in threading.enumerate())


def test_stage2_3_producer_shutdown_stops_thread():
    prod = _scheduler(seed=25, producer={"enable": True, "queue_depth": 2, "preload_next_episode": False})
    _ = prod.next_batch()
    prod.shutdown()
    assert not any(t.name == "Stage23Producer" and t.is_alive() for t in threading.enumerate())


def test_stage2_3_preload_hints_and_episode_pinned_scope():
    ds = _Dataset()
    sched = _scheduler(
        dataset=ds,
        seed=26,
        producer={
            "enable": False,
            "queue_depth": 0,
            "preload_next_episode": True,
            "episode_pinned_cache": True,
        },
    )
    try:
        meta = sched.next_batch()["_iforward"]
        scopes = [x["hint_scope"] for x in ds.preload_hints]
        assert "stage2_3_current_rollout_view_pack" in scopes
        assert "stage2_3_episode_chain_exact" in scopes
        assert ds.active_scopes[-1] == (int(meta["scene_id"]), int(meta["segment_id"]))
        assert ds.training_scopes[-1] == (int(meta["scene_id"]), int(meta["segment_id"]))
        info = sched.get_current_info()
        assert int(info["preload_hint_count"]) == 2
        assert int(info["preload_episode_ref_count"]) > 0
    finally:
        sched.shutdown()
    assert ds.cleared_scopes >= 1
