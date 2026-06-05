from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from datasets.iforward_random_window_scheduler import IForwardRandomWindowScheduler


class _FakeRandomWindowDataset:
    _initialized = True

    def __init__(self, *, segment_lengths=None, num_cams: int = 3, multi_frame_per_keyframe: bool = False):
        self.num_cams = int(num_cams)
        self.segment_lengths = dict(segment_lengths or {0: 6})
        self.multi_frame_per_keyframe = bool(multi_frame_per_keyframe)

    def initialize(self):
        return None

    def list_training_scene_ids(self):
        return [1]

    def list_segment_ids(self, scene_id):
        assert int(scene_id) == 1
        return sorted(int(x) for x in self.segment_lengths)

    def _frames_for_keyframe(self, keyframe_idx: int):
        base = int(keyframe_idx) * 10
        if self.multi_frame_per_keyframe:
            return [base, base + 1]
        return [base]

    def get_segment_index(self, scene_id, segment_id):
        assert int(scene_id) == 1
        n = int(self.segment_lengths[int(segment_id)])
        keyframes = list(range(n))
        keyframe_to_frames = {int(k): self._frames_for_keyframe(int(k)) for k in keyframes}
        frames = [int(f) for values in keyframe_to_frames.values() for f in values]
        frame_to_keyframe = {int(f): int(k) for k, values in keyframe_to_frames.items() for f in values}
        return SimpleNamespace(
            scene_id=1,
            segment_id=int(segment_id),
            num_cams=int(self.num_cams),
            frame_indices=list(frames),
            test_frame_indices=[],
            train_frame_set=set(frames),
            test_frame_set=set(),
            keyframe_indices=list(keyframes),
            keyframe_to_frames=keyframe_to_frames,
            frame_to_keyframe=frame_to_keyframe,
            segment_first_frame_idx=min(frames) if frames else 0,
            train_image_refs=tuple((int(f), int(c)) for f in frames for c in range(int(self.num_cams))),
            test_image_refs=tuple(),
        )

    def validate_image_ref(self, scene_id, segment_id, ref, purpose="train"):
        sidx = self.get_segment_index(int(scene_id), int(segment_id))
        frame_idx, cam_idx = int(ref[0]), int(ref[1])
        assert str(purpose) == "train"
        assert frame_idx in set(int(x) for x in sidx.frame_indices)
        assert 0 <= cam_idx < int(sidx.num_cams)

    def _assemble_segment_batch_from_iforward_random_window_request(self, *, scene_id, segment_id, plan, include_test=False):
        _ = include_test
        return {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "request_meta": dict(plan.request_meta),
            "_iforward": dataclasses.asdict(plan),
            "source": {"refs": list(plan.evidence_refs_flat)},
            "target": {"refs": list(plan.target_refs_flat), "roles": list(plan.target_roles_flat)},
        }


def _scheduler(ds: _FakeRandomWindowDataset, **kwargs) -> IForwardRandomWindowScheduler:
    fixed_starts = kwargs.get("fixed_window_starts")
    window_policy = "fixed_random_with_replacement" if fixed_starts is not None else "random_with_replacement"
    return IForwardRandomWindowScheduler(
        dataset=ds,
        traversal_cfg={
            "scene_order": str(kwargs.get("scene_order", "ascending")),
            "segment_order": str(kwargs.get("segment_order", "ascending")),
            "seed": int(kwargs.get("seed", 41)),
            "fixed_scene_id": kwargs.get("fixed_scene_id", 1),
            "fixed_segment_id": kwargs.get("fixed_segment_id", None),
        },
        segment_cfg={"source_mode": "keyframes", "min_blocks": 4},
        episode_cfg={"rollouts_per_episode": int(kwargs.get("rollouts_per_episode", 8))},
        rollout_cfg={
            "blocks_per_rollout": 4,
            "repeats_per_block": 2,
            "window_policy": window_policy,
            "delivery_order": "chronological",
            "detach_graph_after_rollout": True,
        },
        evidence_cfg={"camera_policy": "all_cams", "mask_policy": "non_sky_non_egocar"},
        supervision_cfg={
            "current_latest": {"enable": True, "camera_policy": "all_cams"},
            "in_rollout_history": {"enable": True, "camera_policy": "all_cams"},
            "short_window_history": {"enable": True, "max_entries": int(kwargs.get("short_history_max", 24))},
            "nearby": {
                "enable": bool(kwargs.get("nearby_enable", True)),
                "frames_per_rollout": 1,
                "camera_policy": "all_cams",
                "policy": "random_non_input_frame_in_segment",
                "max_refs_per_rollout": int(kwargs.get("nearby_max_refs", 3)),
            },
        },
        memory_cfg={
            "observation_commit_policy": "first_repeat_only",
            "optimizer_memory_update_policy": "every_repeat",
            "reset_policy": "episode_begin",
            "carry_policy": "episode",
        },
        loss_timing_cfg={"policy": "rollout_final_only"},
        preload_cfg={"emit_hints": False},
        include_test=False,
        fixed_scene_id=kwargs.get("fixed_scene_id", 1),
        fixed_segment_id=kwargs.get("fixed_segment_id", None),
        seed=int(kwargs.get("seed", 41)),
        fixed_window_starts=fixed_starts,
    )


def test_random_window_scheduler_skips_segments_with_too_few_blocks():
    ds = _FakeRandomWindowDataset(segment_lengths={0: 3, 1: 4})
    batch = _scheduler(ds).next_batch()
    plan = batch["_iforward"]
    assert plan["segment_id"] == 1
    assert plan["window_block_ids"] == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="no valid segments"):
        _scheduler(_FakeRandomWindowDataset(segment_lengths={0: 3}), fixed_segment_id=0)


def test_random_window_scheduler_repeats_windows_with_revisit_metadata():
    ds = _FakeRandomWindowDataset(segment_lengths={0: 6})
    scheduler = _scheduler(ds, fixed_segment_id=0, fixed_window_starts=[1, 1], rollouts_per_episode=2)
    first = scheduler.next_batch()["_iforward"]
    second = scheduler.next_batch()["_iforward"]

    assert first["window_start"] == 1
    assert first["window_block_ids"] == [1, 2, 3, 4]
    assert first["window_revisit_count"] == 0
    assert first["unique_windows_seen"] == 1
    assert first["is_repeated_window"] is False
    assert second["window_start"] == 1
    assert second["window_hash"] == first["window_hash"]
    assert second["window_revisit_count"] == 1
    assert second["unique_windows_seen"] == 1
    assert second["is_repeated_window"] is True


def test_random_window_scheduler_chronological_b4_r2_steps_and_episode_flags():
    ds = _FakeRandomWindowDataset(segment_lengths={0: 6})
    scheduler = _scheduler(ds, fixed_segment_id=0, fixed_window_starts=[2, 0], rollouts_per_episode=2)
    first = scheduler.next_batch()["_iforward"]
    second = scheduler.next_batch()["_iforward"]
    third = scheduler.next_batch()["_iforward"]

    assert first["window_block_ids"] == [2, 3, 4, 5]
    assert first["window_frame_indices"] == [20, 30, 40, 50]
    steps = first["steps"]
    assert [int(s["block_id"]) for s in steps] == [2, 2, 3, 3, 4, 4, 5, 5]
    assert [int(s["block_pos_in_window"]) for s in steps] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [int(s["repeat_idx"]) for s in steps] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert [bool(s["commit_observation_memory"]) for s in steps] == [True, False] * 4
    assert all(bool(s["update_optimizer_memory"]) for s in steps)
    assert [bool(s["is_frame_exit"]) for s in steps] == [False, True] * 4

    assert first["reset_scene_state_before_rollout"] is True
    assert first["carry_scene_state_after_rollout"] is True
    assert first["episode_end_after_rollout"] is False
    assert second["reset_scene_state_before_rollout"] is False
    assert second["carry_scene_state_after_rollout"] is False
    assert second["episode_end_after_rollout"] is True
    assert third["reset_scene_state_before_rollout"] is True
    assert third["episode_id"] != first["episode_id"]


def test_random_window_scheduler_short_history_refs_cap_at_24():
    ds = _FakeRandomWindowDataset(segment_lengths={0: 8}, num_cams=3)
    scheduler = _scheduler(
        ds,
        fixed_segment_id=0,
        fixed_window_starts=[0, 1, 2, 3],
        rollouts_per_episode=4,
        short_history_max=24,
    )
    p0 = scheduler.next_batch()["_iforward"]
    p1 = scheduler.next_batch()["_iforward"]
    p2 = scheduler.next_batch()["_iforward"]
    p3 = scheduler.next_batch()["_iforward"]

    assert len(p0["short_window_history_refs"]) == 0
    assert len(p1["short_window_history_refs"]) == 12
    assert len(p2["short_window_history_refs"]) == 24
    assert len(p3["short_window_history_refs"]) == 24
    assert set(tuple(x) for x in p1["short_window_history_refs"]) == set(tuple(x) for x in p0["evidence_refs_flat"])


def test_random_window_scheduler_state_dict_restores_next_batch():
    ds = _FakeRandomWindowDataset(segment_lengths={0: 7}, multi_frame_per_keyframe=True)
    scheduler = _scheduler(ds, fixed_segment_id=0, seed=123)
    scheduler.next_batch()
    state = scheduler.state_dict()
    expected = scheduler.next_batch()["_iforward"]

    restored = _scheduler(ds, fixed_segment_id=0, seed=999)
    restored.load_state_dict(state)
    actual = restored.next_batch()["_iforward"]

    for key in (
        "episode_id",
        "rollout_id_global",
        "rollout_idx_in_episode",
        "window_start",
        "window_block_ids",
        "window_frame_indices",
        "window_hash",
        "window_revisit_count",
        "short_window_history_refs",
    ):
        assert actual[key] == expected[key]
