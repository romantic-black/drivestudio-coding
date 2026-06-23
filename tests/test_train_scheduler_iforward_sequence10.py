from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from datasets.train_scheduler_iforward_sequence10 import (
    IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
    TrainSchedulerIForwardSequence10,
)


class _FakeSequence10Dataset:
    _initialized = True

    def __init__(self, *, num_keyframes: int = 20, num_cams: int = 3):
        self.num_cams = int(num_cams)
        self.keyframe_to_frames = {int(k): [int(k * 10)] for k in range(int(num_keyframes))}
        self.frames = [int(v[0]) for v in self.keyframe_to_frames.values()]
        self.frame_to_keyframe = {int(frame): int(k) for k, frames in self.keyframe_to_frames.items() for frame in frames}

    def initialize(self):
        return None

    def list_training_scene_ids(self):
        return [1]

    def list_segment_ids(self, scene_id):
        assert int(scene_id) == 1
        return [0]

    def get_segment_index(self, scene_id, segment_id):
        assert int(scene_id) == 1
        assert int(segment_id) == 0
        return SimpleNamespace(
            scene_id=1,
            segment_id=0,
            num_cams=int(self.num_cams),
            frame_indices=list(self.frames),
            test_frame_indices=[],
            train_frame_set=set(self.frames),
            test_frame_set=set(),
            keyframe_indices=list(self.keyframe_to_frames.keys()),
            keyframe_to_frames=dict(self.keyframe_to_frames),
            frame_to_keyframe=dict(self.frame_to_keyframe),
            segment_first_frame_idx=min(self.frames) if self.frames else 0,
            train_image_refs=tuple((int(f), int(c)) for f in self.frames for c in range(int(self.num_cams))),
            test_image_refs=tuple(),
        )

    def validate_image_ref(self, scene_id, segment_id, ref, purpose="train"):
        frame_idx, cam_idx = int(ref[0]), int(ref[1])
        assert int(scene_id) == 1
        assert int(segment_id) == 0
        assert frame_idx in set(self.frames)
        assert 0 <= cam_idx < int(self.num_cams)

    def _assemble_segment_batch_from_iforward_request(self, *, scene_id, segment_id, plan, include_test=False):
        return {
            "request_meta": dict(plan.request_meta),
            "_iforward": dataclasses.asdict(plan),
            "source": {"refs": list(plan.evidence_refs_flat)},
            "target": {"refs": list(plan.target_refs_flat), "roles": list(plan.target_roles_flat)},
        }


class _FakeMultiSceneSequence10Dataset(_FakeSequence10Dataset):
    def __init__(self, *, scene_ids=(1, 2), segment_ids=(0, 1), num_keyframes: int = 20, num_cams: int = 3):
        super().__init__(num_keyframes=num_keyframes, num_cams=num_cams)
        self.scene_ids = [int(x) for x in scene_ids]
        self.segment_ids = [int(x) for x in segment_ids]

    def list_training_scene_ids(self):
        return list(self.scene_ids)

    def list_segment_ids(self, scene_id):
        assert int(scene_id) in set(self.scene_ids)
        return list(self.segment_ids)

    def get_segment_index(self, scene_id, segment_id):
        assert int(scene_id) in set(self.scene_ids)
        assert int(segment_id) in set(self.segment_ids)
        base = super().get_segment_index(1, 0)
        base.scene_id = int(scene_id)
        base.segment_id = int(segment_id)
        return base

    def validate_image_ref(self, scene_id, segment_id, ref, purpose="train"):
        assert int(scene_id) in set(self.scene_ids)
        assert int(segment_id) in set(self.segment_ids)
        frame_idx, cam_idx = int(ref[0]), int(ref[1])
        assert frame_idx in set(self.frames)
        assert 0 <= cam_idx < int(self.num_cams)


def _scheduler(ds: _FakeSequence10Dataset, **kwargs) -> TrainSchedulerIForwardSequence10:
    repair = {"start_step": int(kwargs.get("repair_start_step", 15000)), "prob": float(kwargs.get("repair_prob", 0.5))}
    return TrainSchedulerIForwardSequence10(
        dataset=ds,
        traversal_cfg={"fixed_scene_id": 1, "fixed_segment_id": 0, "seed": int(kwargs.get("seed", 3))},
        bootstrap_cfg={
            "end_step": int(kwargs.get("bootstrap_end_step", 5000)),
            "repeat_choices": [
                {"repeats": 4, "prob": 0.60},
                {"repeats": 6, "prob": 0.30},
                {"repeats": 8, "prob": 0.10},
            ],
        },
        sequence_cfg={"length": 10, "strides": list(kwargs.get("strides", [1, 2])), "max_inner_K": 10},
        causal_cfg={"blocks_per_rollout": 2, "repeats_per_block": 4},
        repair_cfg=repair,
        supervision_cfg={"history_replay": {"role_name": "final_history_replay"}},
    )


def test_sequence10_bootstrap_is_single_block_current_only():
    sched = _scheduler(_FakeSequence10Dataset(), seed=11)
    batch = sched.next_batch()
    meta = batch["_iforward"]
    assert meta["scheduler_version"] == IFORWARD_SEQUENCE10_SCHEDULER_VERSION
    assert meta["scheduler_phase"] == "bootstrap"
    assert meta["actual_blocks_per_rollout"] == 1
    assert meta["repeats_per_block"] in {4, 6, 8}
    assert len(meta["history_positions"]) == 0
    assert all(step["visit_kind"] == "bootstrap" for step in meta["steps"])
    assert all(step["temporal_read"] is False for step in meta["steps"])
    assert all(step["temporal_commit"] is False for step in meta["steps"])
    assert all(step["commit_observation_memory"] is False for step in meta["steps"])
    assert all(step["update_optimizer_memory"] is False for step in meta["steps"])


def test_sequence10_causal_rollouts_cover_all_positions_and_commit_on_exit_only():
    sched = _scheduler(_FakeSequence10Dataset(), seed=7, repair_prob=0.0)
    sched.global_step = 5000
    seen = []
    for rollout_idx in range(5):
        batch = sched.next_batch()
        meta = batch["_iforward"]
        assert meta["scheduler_phase"] == "causal"
        assert meta["actual_blocks_per_rollout"] == 2
        assert meta["repeats_per_block"] == 4
        assert meta["inner_K"] == 8
        assert meta["sequence_positions"] == [rollout_idx * 2, rollout_idx * 2 + 1]
        assert meta["history_positions"] == list(range(rollout_idx * 2))
        exits = [step for step in meta["steps"] if step["is_block_exit"]]
        assert len(exits) == 2
        assert all(step["temporal_commit"] is True for step in exits)
        assert all((step["temporal_commit"] is False) for step in meta["steps"] if not step["is_block_exit"])
        assert meta["steps"][0]["optimizer_step_idx_in_episode"] == rollout_idx * 8
        seen.extend(meta["sequence_positions"])
    assert seen == list(range(10))


def test_sequence10_repair_is_unique_non_identity_and_does_not_commit_memory():
    sched = _scheduler(_FakeSequence10Dataset(), seed=5, repair_prob=1.0)
    sched.global_step = 15000
    batches = [sched.next_batch() for _ in range(6)]
    repair = batches[-1]["_iforward"]
    assert repair["scheduler_phase"] == "repair"
    assert repair["actual_blocks_per_rollout"] == 10
    assert repair["repeats_per_block"] == 1
    assert repair["inner_K"] == 10
    positions = repair["sequence_positions"]
    assert sorted(positions) == list(range(10))
    assert positions != list(range(10))
    assert all(step["visit_kind"] == "repair" for step in repair["steps"])
    assert all(step["temporal_read"] is True for step in repair["steps"])
    assert all(step["temporal_commit"] is False for step in repair["steps"])
    assert all(step["commit_observation_memory"] is False for step in repair["steps"])
    assert all(step["update_optimizer_memory"] is False for step in repair["steps"])


def test_sequence10_stride2_fallback_and_short_segment_skip():
    sched = _scheduler(_FakeSequence10Dataset(num_keyframes=19), strides=[2], seed=13)
    assert len(sched._eligibility_index) == 1
    assert sched._eligibility_index[0]["stride"] == 2
    sched.global_step = 5000
    meta = sched.next_batch()["_iforward"]
    first_repeat_steps = [step for step in meta["steps"] if step["repeat_idx"] == 0]
    assert [step["frame_gap"] for step in first_repeat_steps] == [0, 2]
    with pytest.raises(ValueError, match="no eligible"):
        _scheduler(_FakeSequence10Dataset(num_keyframes=9), seed=13)


def test_sequence10_scene_round_robin_and_segment_queue():
    ds = _FakeMultiSceneSequence10Dataset(scene_ids=(1, 2), segment_ids=(0, 1), num_keyframes=20)
    sched = TrainSchedulerIForwardSequence10(
        dataset=ds,
        traversal_cfg={
            "seed": 23,
            "scene_order": "ordered",
            "segment_order": "ordered",
            "traversal_mode": "scene_round_robin_episode",
            "forbid_consecutive_same_scene": True,
        },
        bootstrap_cfg={"end_step": 0},
        sequence_cfg={"length": 10, "strides": [1], "max_inner_K": 10},
        causal_cfg={"blocks_per_rollout": 2, "repeats_per_block": 4, "rollouts_per_episode": 5},
        repair_cfg={"start_step": 999999, "prob": 0.0, "blocks_per_rollout": 10, "repeats_per_block": 1},
        supervision_cfg={"history_replay": {"enable": True}},
    )
    episode_first = []
    for _episode in range(4):
        batch = sched.next_batch()
        meta = batch["_iforward"]
        episode_first.append((int(meta["scene_id"]), int(meta["segment_id"])))
        for _ in range(4):
            sched.next_batch()
    scenes = [scene for scene, _segment in episode_first]
    assert scenes == [1, 2, 1, 2]
    assert episode_first[0][1] == 0
    assert episode_first[2][1] == 1


def test_sequence10_state_dict_resume_preserves_next_rollout():
    ds = _FakeSequence10Dataset()
    sched = _scheduler(ds, seed=17, repair_prob=1.0)
    sched.global_step = 15000
    first = sched.next_batch()["_iforward"]
    state = sched.state_dict()
    restored = _scheduler(ds, seed=17, repair_prob=1.0)
    restored.load_state_dict(state)
    assert first["sequence_positions"] == [0, 1]
    assert sched.next_batch()["_iforward"] == restored.next_batch()["_iforward"]


def test_sequence10_rejects_legacy_shape_fields():
    with pytest.raises(ValueError, match="legacy shape"):
        TrainSchedulerIForwardSequence10(
            dataset=_FakeSequence10Dataset(),
            traversal_cfg={"fixed_scene_id": 1, "fixed_segment_id": 0, "seed": 1},
            sequence_cfg={"length": 10, "strides": [1], "max_inner_K": 10, "shapes": []},
        )
