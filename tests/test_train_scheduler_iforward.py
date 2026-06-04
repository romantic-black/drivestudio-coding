from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datasets.train_scheduler_iforward import TrainSchedulerIForward


class _FakeDataset:
    _initialized = True

    def __init__(self, *, multi_frame_per_keyframe: bool = True, num_keyframes: int = 6, num_cams: int = 3):
        self.num_cams = int(num_cams)
        self.frames = []
        self.keyframe_to_frames = {}
        self.frame_to_keyframe = {}
        for k in range(int(num_keyframes)):
            if bool(multi_frame_per_keyframe):
                frames = [int(k * 10), int(k * 10 + 1)]
            else:
                frames = [int(k * 10)]
            self.keyframe_to_frames[int(k)] = frames
            for frame in frames:
                self.frames.append(int(frame))
                self.frame_to_keyframe[int(frame)] = int(k)
        self.build_preload_hint = MagicMock(
            side_effect=lambda **kwargs: {
                "scene_id": kwargs["scene_id"],
                "segment_id": kwargs["segment_id"],
                "future_image_refs": kwargs["future_image_refs"],
                "scope": kwargs["scope"],
            }
        )
        self.submit_preload_hint = MagicMock()

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
            segment_first_frame_idx=min(self.frames),
            train_image_refs=tuple((int(f), int(c)) for f in self.frames for c in range(int(self.num_cams))),
            test_image_refs=tuple(),
        )

    def validate_image_ref(self, scene_id, segment_id, ref, purpose="train"):
        frame_idx, cam_idx = int(ref[0]), int(ref[1])
        assert int(scene_id) == 1
        assert int(segment_id) == 0
        assert str(purpose) == "train"
        assert frame_idx in set(self.frames)
        assert 0 <= cam_idx < int(self.num_cams)

    def _assemble_segment_batch_from_iforward_request(self, *, scene_id, segment_id, plan, include_test=False):
        return {
            "request_meta": dict(plan.request_meta),
            "_iforward": dataclasses.asdict(plan),
            "source": {"refs": list(plan.evidence_refs_flat)},
            "target": {"refs": list(plan.target_refs_flat), "roles": list(plan.target_roles_flat)},
        }


def _scheduler(ds: _FakeDataset, **kwargs) -> TrainSchedulerIForward:
    blocks = int(kwargs.get("blocks_per_rollout", 2))
    repeats = int(kwargs.get("repeats_per_block", 3))
    return TrainSchedulerIForward(
        dataset=ds,
        episode_cfg={
            "source_mode": "keyframes",
            "blocks_per_episode": int(kwargs.get("blocks_per_episode", 4)),
            "episode_stride": int(kwargs.get("episode_stride", kwargs.get("blocks_per_episode", 4))),
            "allow_short_last_episode": bool(kwargs.get("allow_short_last_episode", True)),
            "min_blocks_per_episode": int(kwargs.get("min_blocks_per_episode", 1)),
            "block_source_frame_policy": str(
                kwargs.get("block_source_frame_policy", "random_within_keyframe_once_per_episode")
            ),
        },
        rollout_cfg={
            "block_selection_policy": str(kwargs.get("block_selection_policy", "next_contiguous")),
            "delivery_order_policy": "chronological",
            "allow_short_final_rollout": bool(kwargs.get("allow_short_final_rollout", True)),
            "min_blocks_per_rollout": int(kwargs.get("min_blocks_per_rollout", 1)),
            "avoid_single_block_tail": bool(kwargs.get("avoid_single_block_tail", False)),
            "detach_graph_after_rollout": True,
            "shapes": [
                {
                    "name": f"b{blocks}_r{repeats}",
                    "blocks_per_rollout": blocks,
                    "repeats_per_block": repeats,
                    "prob": 1.0,
                }
            ],
        },
        traversal_cfg={
            "traversal_mode": "episode_serial",
            "scene_order": "ascending",
            "segment_order": "ascending",
            "seed": int(kwargs.get("seed", 1)),
        },
        evidence_cfg={"camera_policy": "all_cams", "allow_camera_dropout": False},
        supervision_cfg={
            "current": {
                "enable": True,
                "role_name": "final_current_recon",
                "frame_policy": "all_input_frames",
                "camera_policy": "all_cams",
            },
            "nearby": {
                "enable": bool(kwargs.get("nearby_enable", True)),
                "role_name": "final_nearby_rollout",
                "scope": "rollout_keyframe_span",
                "policy": "random_non_input_frames",
                "frames_per_rollout": int(kwargs.get("nearby_frames_per_rollout", 1)),
                "insufficient_policy": "use_available_or_skip_if_none",
                "camera_policy": "all_cams",
                "max_refs_per_rollout": 24,
                "add_to_evidence": False,
            },
            "history_replay": {"enable": False},
        },
        memory_cfg={
            "observation_commit_policy": "first_repeat_only",
            "optimizer_memory_update_policy": "every_repeat",
            "reset_policy": "episode_begin",
            "carry_policy": "across_rollouts_until_episode_end",
        },
        loss_timing_cfg={"policy": "rollout_final_only", "intermediate_step_loss": False},
        leakage_check_cfg={"enable": True, "forbid_test_refs_in_train": True},
        preload_cfg={
            "emit_hints": bool(kwargs.get("emit_preload_hints", False)),
            "warm_current_rollout_refs": True,
            "hint_scope_for_exact_refs": "v9_role_refs",
        },
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )


def test_iforward_rollout_shape_steps_repeat_flags_and_current_supervision():
    ds = _FakeDataset(multi_frame_per_keyframe=True, num_cams=3)
    batch = _scheduler(ds, blocks_per_rollout=2, repeats_per_block=3).next_batch()

    plan = batch["_iforward"]
    assert plan["shape_name"] == "b2_r3"
    assert plan["inner_K"] == 6
    assert plan["actual_blocks_per_rollout"] == 2
    assert plan["input_frame_indices"] == [0, 10]
    assert plan["delivery_frame_indices"] == [0, 10]

    steps = plan["steps"]
    assert [int(s["source_frame_idx"]) for s in steps] == [0, 0, 0, 10, 10, 10]
    assert [int(s["repeat_idx"]) for s in steps] == [0, 1, 2, 0, 1, 2]
    assert [bool(s["commit_observation_memory"]) for s in steps] == [True, False, False, True, False, False]
    assert all(bool(s["update_optimizer_memory"]) for s in steps)
    assert all(not bool(s["detach_before_step"]) and not bool(s["detach_after_step"]) for s in steps)
    assert steps[0]["rollout_pos_code"] == pytest.approx(0.0)
    assert steps[-1]["rollout_pos_code"] == pytest.approx(1.0)
    assert steps[0]["frame_pos_code"] == pytest.approx(0.0)
    assert steps[-1]["frame_pos_code"] == pytest.approx(1.0)
    assert steps[2]["repeat_pos_code"] == pytest.approx(1.0)

    expected_current = {(0, c) for c in range(3)} | {(10, c) for c in range(3)}
    actual_current = {
        tuple(ref)
        for ref, role in zip(plan["target_refs_flat"], plan["target_roles_flat"])
        if str(role) == "final_current_recon"
    }
    assert actual_current == expected_current


def test_iforward_reset_carry_and_episode_end_flags():
    ds = _FakeDataset(multi_frame_per_keyframe=False, num_cams=2)
    scheduler = _scheduler(ds, blocks_per_rollout=2, repeats_per_block=1, blocks_per_episode=4)
    first = scheduler.next_batch()["_iforward"]
    second = scheduler.next_batch()["_iforward"]
    third = scheduler.next_batch()["_iforward"]

    assert first["reset_scene_state_before_rollout"] is True
    assert first["carry_scene_state_after_rollout"] is True
    assert first["episode_end_after_rollout"] is False
    assert second["reset_scene_state_before_rollout"] is False
    assert second["carry_scene_state_after_rollout"] is False
    assert second["episode_end_after_rollout"] is True
    assert third["reset_scene_state_before_rollout"] is True
    assert int(third["episode_id"]) != int(first["episode_id"])


def test_iforward_short_final_rollout_is_explicit():
    ds = _FakeDataset(multi_frame_per_keyframe=False, num_cams=2)
    scheduler = _scheduler(ds, blocks_per_rollout=2, repeats_per_block=2, blocks_per_episode=3)
    first = scheduler.next_batch()["_iforward"]
    second = scheduler.next_batch()["_iforward"]

    assert first["short_rollout"] is False
    assert second["short_rollout"] is True
    assert second["short_rollout_reason"] == "episode_tail_single_block"
    assert second["requested_blocks_per_rollout"] == 2
    assert second["actual_blocks_per_rollout"] == 1
    assert second["requested_inner_K"] == 4
    assert second["actual_inner_K"] == 2
    assert second["inner_K"] == 2
    assert second["episode_end_after_rollout"] is True


def test_iforward_nearby_uses_rollout_local_non_input_frames_and_skips_when_empty():
    ds = _FakeDataset(multi_frame_per_keyframe=True, num_cams=2)
    batch = _scheduler(ds, blocks_per_rollout=2, repeats_per_block=1).next_batch()
    plan = batch["_iforward"]
    nearby = set(int(x) for x in plan["final_supervision"]["nearby_frames"])
    assert nearby
    assert nearby.isdisjoint(set(int(x) for x in plan["input_frame_indices"]))
    assert nearby <= {1, 11}
    evidence = set(tuple(x) for x in plan["evidence_refs_flat"])
    nearby_refs = {
        tuple(ref)
        for ref, role in zip(plan["target_refs_flat"], plan["target_roles_flat"])
        if str(role) == "final_nearby_rollout"
    }
    assert nearby_refs
    assert nearby_refs.isdisjoint(evidence)

    ds_single = _FakeDataset(multi_frame_per_keyframe=False, num_cams=2)
    batch_single = _scheduler(ds_single, blocks_per_rollout=2, repeats_per_block=1).next_batch()
    final = batch_single["_iforward"]["final_supervision"]
    assert final["nearby_frames"] == []
    assert final["skipped_nearby"] is True
    assert final["nearby_skip_reason"] == "no_non_input_frame_in_rollout"


def test_iforward_state_dict_restores_next_rollout_cursor_and_rng_state():
    ds = _FakeDataset(multi_frame_per_keyframe=True, num_cams=2)
    scheduler = _scheduler(ds, blocks_per_rollout=1, repeats_per_block=1, blocks_per_episode=4)
    scheduler.next_batch()
    state = scheduler.state_dict()
    expected = scheduler.next_batch()["_iforward"]

    restored = _scheduler(ds, blocks_per_rollout=1, repeats_per_block=1, blocks_per_episode=4, seed=999)
    restored.load_state_dict(state)
    actual = restored.next_batch()["_iforward"]

    for key in (
        "episode_id",
        "rollout_id_global",
        "rollout_idx_in_episode",
        "episode_block_indices",
        "input_frame_indices",
        "target_refs_flat",
        "target_roles_flat",
    ):
        assert actual[key] == expected[key]


def test_iforward_preload_hint_uses_deduped_evidence_and_target_refs():
    ds = _FakeDataset(multi_frame_per_keyframe=True, num_cams=2)
    batch = _scheduler(
        ds,
        blocks_per_rollout=2,
        repeats_per_block=1,
        emit_preload_hints=True,
    ).next_batch()

    plan = batch["_iforward"]
    expected_refs = []
    seen = set()
    for ref in list(plan["evidence_refs_flat"]) + list(plan["target_refs_flat"]):
        r = tuple(ref)
        if r in seen:
            continue
        seen.add(r)
        expected_refs.append(r)

    ds.build_preload_hint.assert_called_once()
    ds.submit_preload_hint.assert_called_once()
    kwargs = ds.build_preload_hint.call_args.kwargs
    assert kwargs["scope"] == "v9_role_refs"
    assert [tuple(x) for x in kwargs["future_image_refs"]] == expected_refs


def test_iforward_peek_restores_full_scheduler_state():
    ds = _FakeDataset(multi_frame_per_keyframe=True, num_cams=2)
    scheduler = _scheduler(ds, blocks_per_rollout=2, repeats_per_block=1, blocks_per_episode=4)

    peek = scheduler.materialize_current_batch_without_advance()["_iforward"]
    next_batch = scheduler.next_batch()["_iforward"]

    for key in (
        "episode_id",
        "rollout_id_global",
        "rollout_idx_in_episode",
        "shape_name",
        "input_frame_indices",
        "target_refs_flat",
        "target_roles_flat",
    ):
        assert next_batch[key] == peek[key]


def test_iforward_tail_aware_sampler_avoids_single_block_tail_when_possible():
    ds = _FakeDataset(multi_frame_per_keyframe=False, num_cams=2)
    scheduler = TrainSchedulerIForward(
        dataset=ds,
        episode_cfg={
            "source_mode": "keyframes",
            "blocks_per_episode": 5,
            "episode_stride": 5,
            "allow_short_last_episode": True,
            "min_blocks_per_episode": 1,
            "block_source_frame_policy": "random_within_keyframe_once_per_episode",
        },
        rollout_cfg={
            "block_selection_policy": "next_contiguous",
            "delivery_order_policy": "chronological",
            "allow_short_final_rollout": True,
            "min_blocks_per_rollout": 2,
            "avoid_single_block_tail": True,
            "detach_graph_after_rollout": True,
            "shapes": [
                {"name": "b4_r1", "blocks_per_rollout": 4, "repeats_per_block": 1, "prob": 1.0},
                {"name": "b3_r1", "blocks_per_rollout": 3, "repeats_per_block": 1, "prob": 1.0},
            ],
        },
        traversal_cfg={"traversal_mode": "episode_serial", "scene_order": "ascending", "segment_order": "ascending", "seed": 1},
        evidence_cfg={"camera_policy": "all_cams", "allow_camera_dropout": False},
        supervision_cfg={
            "current": {"enable": True, "role_name": "final_current_recon", "frame_policy": "all_input_frames", "camera_policy": "all_cams"},
            "nearby": {"enable": False},
            "history_replay": {"enable": False},
        },
        memory_cfg={
            "observation_commit_policy": "first_repeat_only",
            "optimizer_memory_update_policy": "every_repeat",
            "reset_policy": "episode_begin",
            "carry_policy": "across_rollouts_until_episode_end",
        },
        loss_timing_cfg={"policy": "rollout_final_only", "intermediate_step_loss": False},
        leakage_check_cfg={"enable": True, "forbid_test_refs_in_train": True},
        preload_cfg={"emit_hints": False},
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    first = scheduler.next_batch()["_iforward"]
    second = scheduler.next_batch()["_iforward"]
    assert first["actual_blocks_per_rollout"] == 3
    assert second["actual_blocks_per_rollout"] == 2


def test_iforward_short_final_rollout_disabled_skips_tail():
    ds = _FakeDataset(multi_frame_per_keyframe=False, num_keyframes=6, num_cams=2)
    scheduler = TrainSchedulerIForward(
        dataset=ds,
        episode_cfg={
            "source_mode": "keyframes",
            "blocks_per_episode": 6,
            "episode_stride": 6,
            "allow_short_last_episode": True,
            "min_blocks_per_episode": 1,
            "block_source_frame_policy": "random_within_keyframe_once_per_episode",
        },
        rollout_cfg={
            "block_selection_policy": "next_contiguous",
            "delivery_order_policy": "chronological",
            "allow_short_final_rollout": False,
            "min_blocks_per_rollout": 4,
            "avoid_single_block_tail": True,
            "detach_graph_after_rollout": True,
            "shapes": [
                {"name": "b4_r2", "blocks_per_rollout": 4, "repeats_per_block": 2, "prob": 1.0},
            ],
        },
        traversal_cfg={"traversal_mode": "episode_serial", "scene_order": "ascending", "segment_order": "ascending", "seed": 1},
        evidence_cfg={"camera_policy": "all_cams", "allow_camera_dropout": False},
        supervision_cfg={
            "current": {"enable": True, "role_name": "final_current_recon", "frame_policy": "all_input_frames", "camera_policy": "all_cams"},
            "nearby": {"enable": False},
            "history_replay": {"enable": False},
        },
        memory_cfg={
            "observation_commit_policy": "first_repeat_only",
            "optimizer_memory_update_policy": "every_repeat",
            "reset_policy": "episode_begin",
            "carry_policy": "across_rollouts_until_episode_end",
        },
        loss_timing_cfg={"policy": "rollout_final_only", "intermediate_step_loss": False},
        leakage_check_cfg={"enable": True, "forbid_test_refs_in_train": True},
        preload_cfg={"emit_hints": False},
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    first = scheduler.next_batch()["_iforward"]
    assert first["actual_blocks_per_rollout"] == 4
    assert first["inner_K"] == 8
    events = scheduler.pop_events()
    assert first["episode_end_after_rollout"] is True
    assert first["carry_scene_state_after_rollout"] is False
    assert first["tail_skipped_after_rollout"] is True
    assert any(event.get("type") == "episode_tail_skipped" and int(event.get("remaining_blocks", 0)) == 2 for event in events)
    assert not any(str(event.get("type", "")).startswith("rollout") and int(event.get("inner_K", 8)) != 8 for event in events)


def test_iforward_random_start_rollout_resamples_start_and_frames_per_rollout():
    ds = _FakeDataset(multi_frame_per_keyframe=True, num_keyframes=8, num_cams=2)
    scheduler = _scheduler(
        ds,
        blocks_per_rollout=4,
        repeats_per_block=2,
        blocks_per_episode=8,
        block_selection_policy="random_start_contiguous",
        block_source_frame_policy="random_within_keyframe_per_rollout",
        allow_short_final_rollout=False,
        min_blocks_per_rollout=4,
        seed=7,
    )

    first = scheduler.next_batch()["_iforward"]
    second = scheduler.next_batch()["_iforward"]

    for plan in (first, second):
        assert plan["request_meta"]["block_selection_policy"] == "random_start_contiguous"
        assert plan["request_meta"]["source_frame_sampling_policy"] == "random_within_keyframe_per_rollout"
        assert plan["actual_blocks_per_rollout"] == 4
        assert plan["inner_K"] == 8
        assert plan["short_rollout"] is False
        blocks = [int(x) for x in plan["episode_block_indices"]]
        assert blocks == list(range(blocks[0], blocks[0] + 4))
        assert 0 <= blocks[0] <= 4
        assert [int(x) for x in plan["input_keyframe_indices"]] == blocks

    first_start = int(first["request_meta"]["rollout_start_block_idx"])
    second_start = int(second["request_meta"]["rollout_start_block_idx"])
    assert second_start != first_start
    if first_start + 4 <= 4:
        assert second_start != first_start + 4


def test_iforward_events_use_batch_emitted_name():
    ds = _FakeDataset(multi_frame_per_keyframe=False, num_cams=2)
    scheduler = _scheduler(ds, blocks_per_rollout=2, repeats_per_block=1, blocks_per_episode=2)
    scheduler.next_batch()
    events = scheduler.pop_events()
    assert any(event.get("type") == "rollout_batch_emitted" for event in events)
    assert not any(event.get("type") == "rollout_end" for event in events)
