from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from datasets.iforward_stage2_2.index_builder import build_stage2_2_index_from_dataset
from datasets.iforward_stage2_2.scheduler import IFORWARD_STAGE2_2_SCHEDULER_VERSION, Stage22Scheduler


class _Dataset:
    _initialized = True

    def __init__(self, *, scene_ids=(1,), segment_ids=(0,), frames=range(40), num_cams=3):
        self.scene_ids = [int(x) for x in scene_ids]
        self.segment_ids = [int(x) for x in segment_ids]
        self.frames = [int(x) for x in frames]
        self.num_cams = int(num_cams)

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

    def _assemble_segment_batch_from_iforward_stage2_2_request(self, *, scene_id, segment_id, plan, include_test=False):
        return {"request_meta": dict(plan.request_meta), "_iforward": dataclasses.asdict(plan)}


def _scheduler(ds=None, **kwargs):
    ds = ds or _Dataset()
    index = build_stage2_2_index_from_dataset(dataset=ds, cfg={})
    return Stage22Scheduler(
        dataset=ds,
        index=index,
        bootstrap_cfg={
            "end_step": int(kwargs.get("bootstrap_end", 0)),
            "repeats": 8,
            "repeat_choices": kwargs.get("repeat_choices", []),
        },
        protocol_cfg={"weights": kwargs.get("weights", {"D1": 1.0})},
        repair_cfg={
            "start_step": int(kwargs.get("repair_start", 999999)),
            "prob": float(kwargs.get("repair_prob", 0.0)),
            "blocks_per_rollout": int(kwargs.get("repair_blocks", 10)),
            "repeats_per_block": int(kwargs.get("repair_repeats", 1)),
        },
        seed=int(kwargs.get("seed", 3)),
    )


def test_stage2_2_scheduler_causal_episode_is_10_raw_frames_same_segment():
    sched = _scheduler()
    seen = []
    for rollout_idx in range(5):
        meta = sched.next_batch()["_iforward"]
        assert meta["scheduler_version"] == IFORWARD_STAGE2_2_SCHEDULER_VERSION
        assert meta["scheduler_phase"] == "causal"
        assert meta["actual_blocks_per_rollout"] == 2
        assert meta["repeats_per_block"] == 4
        assert meta["inner_K"] == 8
        assert meta["sequence_positions"] == [rollout_idx * 2, rollout_idx * 2 + 1]
        assert len({(s["scene_id"], s["segment_id"]) for s in [meta]}) == 1
        exits = [s for s in meta["steps"] if s["is_block_exit"]]
        assert len(exits) == 2
        assert all(s["temporal_commit"] for s in exits)
        seen.extend(meta["sequence_positions"])
    assert seen == list(range(10))


def test_stage2_2_scheduler_gap_protocol_d2():
    sched = _scheduler(weights={"D2": 1.0})
    metas = [sched.next_batch()["_iforward"] for _ in range(2)]
    first_repeats = [s for s in metas[0]["steps"] if s["repeat_idx"] == 0]
    second_repeats = [s for s in metas[1]["steps"] if s["repeat_idx"] == 0]
    assert [s["frame_gap"] for s in first_repeats] == [0, 2]
    assert [s["frame_gap"] for s in second_repeats] == [2, 2]
    assert [s["delta_t_sec"] for s in first_repeats] == [0.0, 0.2]
    assert [round(float(s["delta_t_sec"]), 4) for s in second_repeats] == [0.2, 0.2]
    assert metas[0]["sequence_protocol"] == "D2"
    assert metas[0]["sequence_source_frame_indices"][:3] == [0, 2, 4]


def test_stage2_2_scheduler_repair_b8r1_no_commit():
    sched = _scheduler(repair_start=0, repair_prob=1.0, seed=5, repair_blocks=8)
    metas = [sched.next_batch()["_iforward"] for _ in range(6)]
    repair = metas[-1]
    assert repair["scheduler_phase"] == "repair"
    assert repair["shape_name"] == "repair_b8r1"
    assert repair["actual_blocks_per_rollout"] == 8
    assert repair["inner_K"] == 8
    assert len(set(repair["sequence_positions"])) == 8
    assert all(0 <= int(pos) < 10 for pos in repair["sequence_positions"])
    assert all(s["repair_no_commit"] for s in repair["steps"])
    assert all(not s["temporal_commit"] for s in repair["steps"])
    assert all(not s["update_optimizer_memory"] for s in repair["steps"])
    assert all(int(s["frame_gap"]) == 0 for s in repair["steps"])
    assert all(float(s["delta_t_sec"]) == 0.0 for s in repair["steps"])
    assert all(tuple(float(x) for x in s["ego_delta_translation"]) == (0.0, 0.0, 0.0) for s in repair["steps"])


def test_stage2_2_scheduler_resume_determinism():
    sched = _scheduler(seed=9, repair_start=0, repair_prob=1.0)
    first = sched.next_batch()["_iforward"]
    state = sched.state_dict()
    restored = _scheduler(seed=9, repair_start=0, repair_prob=1.0)
    restored.load_state_dict(state)
    assert sched.next_batch()["_iforward"] == restored.next_batch()["_iforward"]
    assert first["sequence_positions"] == [0, 1]


def test_stage2_2_scheduler_scene_round_robin_ratio():
    ds = _Dataset(scene_ids=(1, 2), segment_ids=(0,), frames=range(40))
    sched = _scheduler(ds=ds, weights={"D1": 1.0}, seed=1)
    scenes = []
    for _ in range(4):
        scenes.append(int(sched.next_batch()["_iforward"]["scene_id"]))
        for _ in range(4):
            sched.next_batch()
    assert set(scenes) == {1, 2}
    assert all(a != b for a, b in zip(scenes, scenes[1:]))


def test_stage2_2_scheduler_bootstrap_repeat_choices():
    sched = _scheduler(bootstrap_end=20, repeat_choices=[{"repeats": 4, "prob": 1.0}], seed=11)
    meta = sched.next_batch()["_iforward"]
    assert meta["scheduler_phase"] == "bootstrap"
    assert meta["repeats_per_block"] == 4
    assert all(not s["temporal_read"] for s in meta["steps"])
    assert all(not s["temporal_commit"] for s in meta["steps"])
