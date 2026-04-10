from __future__ import annotations

from unittest.mock import MagicMock

from datasets.multi_scene_dataset_v3 import (
    MultiSceneDatasetV3,
    SegmentIndex,
    TrainSchedulerV5,
    _build_segment_index_dict,
)


def _make_sidx() -> SegmentIndex:
    scene_data = {
        "dataset": MagicMock(num_cams=3),
        "keyframe_segments": [[10, 11], [12, 13], [14, 15]],
        "segments": [
            {
                "frame_indices": [10, 11, 12, 13, 14, 15],
                "test_frame_indices": [90, 91],
                "keyframe_indices": [0, 1, 2],
            }
        ],
    }
    return _build_segment_index_dict(1, 0, scene_data)


def _make_mock_dataset(sidx: SegmentIndex) -> MagicMock:
    ds = MagicMock(spec=MultiSceneDatasetV3)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.get_segment_index = MagicMock(return_value=sidx)
    ds.get_scene = MagicMock(return_value={"segments": [{"keyframe_indices": [0, 1, 2]}]})
    ds.get_segment_batch_from_frames = MagicMock(return_value={"ok": True})
    ds.get_segment_batch_from_image_refs = MagicMock(side_effect=AssertionError("image-ref API must not be used"))
    return ds


def test_v5_next_batch_uses_frame_api_and_source_first_target():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV5(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        total_target_frames=3,
        include_source_frame=True,
        neighbor_ring=1,
        prefer_nearby_keyframes=True,
        fallback_expand_to_segment=True,
        with_replacement=True,
        include_test=True,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    batch = sch.next_batch()
    assert batch["ok"] is True
    ds.get_segment_batch_from_frames.assert_called_once()
    args, kwargs = ds.get_segment_batch_from_frames.call_args
    assert kwargs["target_frame_indices"][0] == kwargs["source_frame_idx"]
    assert kwargs["enforce_target0_equals_source"] is True
    ds.get_segment_batch_from_image_refs.assert_not_called()


def test_v5_include_source_frame_false_excludes_source_from_targets():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV5(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=1,
        keyframes_per_episode=2,
        episodes_per_segment=1,
        total_target_frames=2,
        include_source_frame=False,
        neighbor_ring=1,
        prefer_nearby_keyframes=True,
        fallback_expand_to_segment=True,
        with_replacement=True,
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    sch.next_batch()
    _, kwargs = ds.get_segment_batch_from_frames.call_args
    assert len(kwargs["target_frame_indices"]) == 2
    assert kwargs["source_frame_idx"] not in kwargs["target_frame_indices"]
    assert kwargs["enforce_target0_equals_source"] is False


def test_v5_episode_limit_is_hard_cap_even_if_budget_is_oversized():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV5(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=1,
        keyframes_per_episode=1,
        episodes_per_segment=1,
        total_target_frames=1,
        include_source_frame=True,
        neighbor_ring=1,
        prefer_nearby_keyframes=True,
        fallback_expand_to_segment=True,
        with_replacement=True,
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    sch._ensure_epoch_plan_index(0)
    sch._hydrate_plan_item_budget(0)
    sch.epoch_plan[0]["segment_budget_u"] = 5
    sch.epoch_plan[0]["segment_step_budget"] = 5

    sch.next_batch()
    events = sch.pop_events()
    reset_events = [e for e in events if e.get("type") == "reset_event"]
    segment_end_events = [e for e in events if e.get("type") == "segment_end"]
    assert len(reset_events) == 1
    assert len(segment_end_events) == 1


def test_v5_enter_end_segment_manage_preload_scope_like_v4():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV5(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=1,
        keyframes_per_episode=1,
        episodes_per_segment=1,
        total_target_frames=1,
        include_source_frame=True,
        neighbor_ring=1,
        prefer_nearby_keyframes=True,
        fallback_expand_to_segment=True,
        with_replacement=True,
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    sch.next_batch()
    ds.set_preload_active_scope.assert_called_with(1, 0)
    ds.set_preload_training_scope.assert_called_with(1, 0)
    ds.clear_preload_scheduler_scope.assert_called()


def test_v5_factory_returns_scheduler():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = MultiSceneDatasetV3.create_train_scheduler_v5(
        ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=1,
        total_target_frames=2,
        include_source_frame=True,
        neighbor_ring=1,
        prefer_nearby_keyframes=True,
        fallback_expand_to_segment=True,
        with_replacement=True,
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    assert isinstance(sch, TrainSchedulerV5)
