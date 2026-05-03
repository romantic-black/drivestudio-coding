from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4, SegmentIndexV4
from datasets.train_scheduler_v8 import TrainSchedulerV8


def _make_sidx(*, scene_id: int, segment_id: int, base_frame: int = 10) -> SegmentIndexV4:
    keyframes = [0, 1, 2, 3, 4, 5]
    keyframe_to_frames = {k: [base_frame + k] for k in keyframes}
    frame_to_keyframe = {base_frame + k: k for k in keyframes}
    frames = [base_frame + k for k in keyframes]
    return SegmentIndexV4(
        scene_id=scene_id,
        segment_id=segment_id,
        num_cams=2,
        frame_indices=frames,
        test_frame_indices=[],
        train_frame_set=frozenset(frames),
        test_frame_set=frozenset(),
        keyframe_indices=keyframes,
        keyframe_to_frames=keyframe_to_frames,
        frame_to_keyframe=frame_to_keyframe,
        segment_first_frame_idx=frames[0],
        train_image_refs=tuple((f, 0) for f in frames),
        test_image_refs=tuple(),
    )


def _make_sidx_multi_frame_per_keyframe(*, scene_id: int, segment_id: int, base_frame: int = 10) -> SegmentIndexV4:
    keyframes = [0, 1, 2, 3, 4, 5]
    keyframe_to_frames = {k: [base_frame + k, base_frame + 100 + k] for k in keyframes}
    frame_to_keyframe = {}
    frames = []
    for k in keyframes:
        for f in keyframe_to_frames[k]:
            frame_to_keyframe[int(f)] = int(k)
            frames.append(int(f))
    return SegmentIndexV4(
        scene_id=scene_id,
        segment_id=segment_id,
        num_cams=2,
        frame_indices=frames,
        test_frame_indices=[],
        train_frame_set=frozenset(frames),
        test_frame_set=frozenset(),
        keyframe_indices=keyframes,
        keyframe_to_frames=keyframe_to_frames,
        frame_to_keyframe=frame_to_keyframe,
        segment_first_frame_idx=frames[0],
        train_image_refs=tuple((f, 0) for f in frames),
        test_image_refs=tuple(),
    )


def _make_mock_dataset() -> MagicMock:
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    ds.get_segment_index = MagicMock(return_value=_make_sidx(scene_id=1, segment_id=0, base_frame=10))
    ds.get_segment_batch_from_image_refs = MagicMock(side_effect=lambda *args, **kwargs: {"ok": True})
    ds._assemble_segment_batch_from_image_refs = MagicMock(side_effect=lambda *args, **kwargs: {"ok": True})
    ds.build_preload_hint = MagicMock(
        side_effect=lambda **kwargs: {
            "scene_id": kwargs["scene_id"],
            "segment_id": kwargs["segment_id"],
            "future_image_refs": kwargs["future_image_refs"],
            "scope": kwargs["scope"],
        }
    )
    ds.submit_preload_hint = MagicMock()
    return ds


def _build_scheduler(ds: MagicMock, **kwargs) -> TrainSchedulerV8:
    return TrainSchedulerV8(
        dataset=ds,
        steps_per_block=int(kwargs.get("steps_per_block", 2)),
        blocks_per_episode=int(kwargs.get("blocks_per_episode", 3)),
        total_target_frames=int(kwargs.get("total_target_frames", 3)),
        include_source_frame=True,
        frame_within_keyframe_policy="middle_frame",
        min_keyframes_required_policy="skip_if_less_than_window",
        traversal_mode="linear_scene_segment",
        switch_after_episode=True,
        segment_order="ascending",
        scene_order="ascending",
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
        emit_preload_hints=bool(kwargs.get("emit_preload_hints", False)),
        warm_next_block_exact=bool(kwargs.get("warm_next_block_exact", False)),
        warm_next_episode_chain=bool(kwargs.get("warm_next_episode_chain", False)),
        block_order=str(kwargs.get("block_order", "block_major")),
        step_major_switch_interval_steps=int(kwargs.get("step_major_switch_interval_steps", 1)),
        target_policy="visited_episode_frames",
        reset_policy="episode_end",
        aux_feature_splat_targets_cfg=kwargs.get("aux_feature_splat_targets_cfg"),
    )


def test_v8_block_major_targets_follow_visited_history():
    ds = _make_mock_dataset()
    sch = _build_scheduler(ds, steps_per_block=1, block_order="block_major")
    targets = []
    for _ in range(3):
        batch = sch.next_batch()
        info = batch["_scheduler_v8_aligned_info"]
        targets.append([int(x) for x in info["target_frame_indices"]])
    assert targets == [[10], [11, 10], [12, 11, 10]]


def test_v8_step_major_targets_expand_after_first_round():
    ds = _make_mock_dataset()
    sch = _build_scheduler(
        ds,
        steps_per_block=8,
        block_order="step_major",
        step_major_switch_interval_steps=4,
    )
    targets = []
    for _ in range(24):
        batch = sch.next_batch()
        info = batch["_scheduler_v8_aligned_info"]
        targets.append((int(info["source_frame_idx"]), tuple(int(x) for x in info["target_frame_indices"])))

    # first 12 steps => first visit rounds (4 per block)
    assert targets[0] == (10, (10,))
    assert targets[4] == (11, (11, 10))
    assert targets[8] == (12, (12, 11, 10))
    # second rounds revisit b0/b1/b2 with full visited context
    assert targets[12] == (10, (10, 11, 12))
    assert targets[16] == (11, (11, 10, 12))
    assert targets[20] == (12, (12, 11, 10))


def test_v8_fast_fail_total_target_frames_larger_than_blocks():
    ds = _make_mock_dataset()
    with pytest.raises(ValueError, match="total_target_frames must be <= blocks_per_episode"):
        _build_scheduler(ds, blocks_per_episode=3, total_target_frames=4)


def test_v8_factory_returns_scheduler():
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    ds.get_segment_index = MagicMock(return_value=_make_sidx(scene_id=1, segment_id=0, base_frame=10))
    ds._assemble_segment_batch_from_image_refs = MagicMock(side_effect=lambda *args, **kwargs: {"ok": True})
    sch = MultiSceneDatasetV4.create_train_scheduler_v8(
        ds,
        steps_per_block=1,
        blocks_per_episode=3,
        total_target_frames=3,
        include_source_frame=True,
        frame_within_keyframe_policy="middle_frame",
        min_keyframes_required_policy="skip_if_less_than_window",
        traversal_mode="linear_scene_segment",
        switch_after_episode=True,
        segment_order="ascending",
        scene_order="ascending",
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
        emit_preload_hints=False,
        warm_next_block_exact=False,
        warm_next_episode_chain=False,
        block_order="step_major",
        step_major_switch_interval_steps=4,
        target_policy="visited_episode_frames",
        reset_policy="episode_end",
    )
    assert isinstance(sch, TrainSchedulerV8)


def test_v8_init_starts_from_epoch_one_once():
    ds = _make_mock_dataset()
    sch = _build_scheduler(ds)
    assert int(sch.epoch_idx) == 1
    assert int(ds.get_segment_index.call_count) == 1


def test_v8_step_major_block_end_reports_first_last_targets_and_actual_updates():
    ds = _make_mock_dataset()
    sch = _build_scheduler(
        ds,
        steps_per_block=8,
        block_order="step_major",
        step_major_switch_interval_steps=4,
    )
    for _ in range(24):
        sch.next_batch()
    events = sch.pop_events()
    block_end_events = [e for e in events if e.get("type") == "block_end"]
    assert [int(e["block_idx_in_episode"]) for e in block_end_events] == [0, 1, 2]
    assert [int(e["num_updates_in_block"]) for e in block_end_events] == [8, 8, 8]
    assert [int(x) for x in block_end_events[0]["target_frame_indices_first_visit"]] == [10]
    assert [int(x) for x in block_end_events[0]["target_frame_indices_last_visit"]] == [10, 11, 12]
    assert [int(x) for x in block_end_events[0]["target_frame_indices"]] == [10, 11, 12]
    assert [int(x) for x in block_end_events[1]["target_frame_indices_first_visit"]] == [11, 10]
    assert [int(x) for x in block_end_events[1]["target_frame_indices_last_visit"]] == [11, 10, 12]
    assert [int(x) for x in block_end_events[2]["target_frame_indices_first_visit"]] == [12, 11, 10]
    assert [int(x) for x in block_end_events[2]["target_frame_indices_last_visit"]] == [12, 11, 10]


def test_v8_block_begin_additional_chain_hint_uses_episode_chain_scope():
    ds = _make_mock_dataset()
    sch = _build_scheduler(
        ds,
        steps_per_block=1,
        block_order="block_major",
        emit_preload_hints=True,
        warm_next_block_exact=True,
        warm_next_episode_chain=False,
    )
    sch.next_batch()
    events = sch.pop_events()
    scopes = [str(e.get("hint_scope")) for e in events if e.get("type") == "preload_hint"]
    assert "episode_chain_exact" in scopes


def test_v8_random_source_policy_resamples_source_per_visit_and_updates_history_targets():
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    ds.get_segment_index = MagicMock(
        return_value=_make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, base_frame=10)
    )
    ds.get_segment_batch_from_image_refs = MagicMock(side_effect=lambda *args, **kwargs: {"ok": True})
    ds._assemble_segment_batch_from_image_refs = MagicMock(side_effect=lambda *args, **kwargs: {"ok": True})
    ds.build_preload_hint = MagicMock(
        side_effect=lambda **kwargs: {
            "scene_id": kwargs["scene_id"],
            "segment_id": kwargs["segment_id"],
            "future_image_refs": kwargs["future_image_refs"],
            "scope": kwargs["scope"],
        }
    )
    ds.submit_preload_hint = MagicMock()

    sch = TrainSchedulerV8(
        dataset=ds,
        steps_per_block=2,
        blocks_per_episode=3,
        total_target_frames=3,
        include_source_frame=True,
        frame_within_keyframe_policy="middle_frame",
        min_keyframes_required_policy="skip_if_less_than_window",
        traversal_mode="linear_scene_segment",
        switch_after_episode=True,
        segment_order="ascending",
        scene_order="ascending",
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
        emit_preload_hints=False,
        warm_next_block_exact=False,
        warm_next_episode_chain=False,
        block_order="step_major",
        step_major_switch_interval_steps=1,
        target_policy="visited_episode_frames",
        reset_policy="episode_end",
        block_source_frame_policy="random_within_keyframe_per_visit",
    )

    with patch(
        "datasets.train_scheduler_v8.random.choice",
        side_effect=[110, 111, 112, 10, 11, 12],
    ):
        batches = [sch.next_batch() for _ in range(4)]

    aligned0 = batches[0]["_scheduler_v8_aligned_info"]
    aligned1 = batches[1]["_scheduler_v8_aligned_info"]
    aligned2 = batches[2]["_scheduler_v8_aligned_info"]
    aligned3 = batches[3]["_scheduler_v8_aligned_info"]
    assert int(aligned0["source_frame_idx"]) == 110
    assert [int(x) for x in aligned0["target_frame_indices"]] == [110]
    assert int(aligned1["source_frame_idx"]) == 111
    assert [int(x) for x in aligned1["target_frame_indices"]] == [111, 110]
    assert int(aligned2["source_frame_idx"]) == 112
    assert [int(x) for x in aligned2["target_frame_indices"]] == [112, 111, 110]
    assert int(aligned3["source_frame_idx"]) == 10
    assert [int(x) for x in aligned3["target_frame_indices"]] == [10, 111, 112]


def test_v8_random_source_with_near_random_skip_when_single_frame_only():
    ds = _make_mock_dataset()
    sch = TrainSchedulerV8(
        dataset=ds,
        steps_per_block=1,
        blocks_per_episode=3,
        total_target_frames=3,
        include_source_frame=True,
        frame_within_keyframe_policy="middle_frame",
        min_keyframes_required_policy="skip_if_less_than_window",
        traversal_mode="linear_scene_segment",
        switch_after_episode=True,
        segment_order="ascending",
        scene_order="ascending",
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
        emit_preload_hints=False,
        warm_next_block_exact=False,
        warm_next_episode_chain=False,
        block_order="step_major",
        step_major_switch_interval_steps=1,
        target_policy="visited_episode_frames",
        reset_policy="episode_end",
        near_random_supervision_cfg={
            "enable": True,
            "frames_per_block": 1,
            "same_keyframe_only": True,
            "insufficient_policy": "skip",
            "sample_once_per_block": False,
            "exclude_source_frame": True,
            "exclude_existing_target_frames": True,
            "camera_policy": "all_cams",
            "role_name": "near_random",
        },
        block_source_frame_policy="random_within_keyframe_per_visit",
    )
    batch = sch.next_batch()
    req = batch.get("request_meta") or {}
    assert [int(x) for x in req.get("near_random_frame_indices", [])] == []
    assert float(req.get("scheduler/near_random/skip_ratio", 0.0)) == pytest.approx(1.0)


def test_v8_aux_adjacent_uses_explicit_source_cam():
    ds = _make_mock_dataset()
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, base_frame=10)
    sch = _build_scheduler(
        ds,
        aux_feature_splat_targets_cfg={
            "enable": True,
            "policy": "adjacent_frame_same_camera",
            "max_refs_per_step": 1,
            "role_name": "aux_feature_splat",
        },
    )
    refs, roles = sch._build_aux_image_refs_for_block(
        sidx=sidx,
        source_keyframe_idx=0,
        source_frame=110,
        source_cam=1,
        base_target_frames=[110],
        num_cams=3,
        exclude_image_refs=[],
    )
    assert refs == [(10, 1)]
    assert roles == ["aux_feature_splat"]


def test_v8_aux_same_timestamp_policy_is_removed():
    ds = _make_mock_dataset()
    sidx = _make_sidx(scene_id=1, segment_id=0, base_frame=10)
    sch = _build_scheduler(
        ds,
        aux_feature_splat_targets_cfg={
            "enable": True,
            "policy": "same_timestamp_neighbor_camera",
            "max_refs_per_step": 2,
            "role_name": "aux_feature_splat",
        },
    )
    with pytest.raises(ValueError, match="unsupported aux_feature_splat_targets policy"):
        sch._build_aux_image_refs_for_block(
            sidx=sidx,
            source_keyframe_idx=0,
            source_frame=10,
            source_cam=0,
            base_target_frames=[10],
            num_cams=3,
            exclude_image_refs=[],
        )


def test_v8_aux_fixed_camera_must_be_source_camera():
    ds = _make_mock_dataset()
    sch = _build_scheduler(
        ds,
        aux_feature_splat_targets_cfg={
            "enable": True,
            "policy": "adjacent_frame_same_camera",
            "aux_camera_policy": "fixed_cam",
            "fixed_cam_id": 2,
            "max_refs_per_step": 1,
            "role_name": "aux_feature_splat",
        },
    )
    with pytest.raises(ValueError, match="is not present in source_image_refs"):
        sch._select_aux_source_cam(source_image_refs=[(10, 0), (10, 1)], num_cams=3)
