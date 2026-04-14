from __future__ import annotations

from unittest.mock import MagicMock

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4, SegmentIndexV4
from datasets.train_scheduler_v6 import TrainSchedulerV6


def _make_sidx() -> SegmentIndexV4:
    return SegmentIndexV4(
        scene_id=1,
        segment_id=0,
        num_cams=3,
        frame_indices=[10, 11, 12, 13, 14, 15],
        test_frame_indices=[90, 91],
        train_frame_set=frozenset([10, 11, 12, 13, 14, 15]),
        test_frame_set=frozenset([90, 91]),
        keyframe_indices=[0, 1, 2],
        keyframe_to_frames={0: [10, 11], 1: [12, 13], 2: [14, 15]},
        frame_to_keyframe={10: 0, 11: 0, 12: 1, 13: 1, 14: 2, 15: 2},
        segment_first_frame_idx=10,
        train_image_refs=((10, 0),),
        test_image_refs=((90, 0),),
    )


def _make_mock_dataset(sidx: SegmentIndexV4) -> MagicMock:
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    ds.get_segment_index = MagicMock(return_value=sidx)
    ds.get_segment_batch_from_image_refs = MagicMock(return_value={"ok": True})
    return ds


def test_v6_next_batch_uses_image_ref_api_and_source_first_target():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV6(
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
    ds.get_segment_batch_from_image_refs.assert_called_once()
    req = ds.get_segment_batch_from_image_refs.call_args.args[0]
    assert tuple(req.target_image_refs[0]) == tuple(req.source_image_ref)
    assert ds.get_segment_batch_from_image_refs.call_args.kwargs["enforce_target0_equals_source"] is True


def test_v6_include_source_frame_false_excludes_source_from_targets():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV6(
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
    req = ds.get_segment_batch_from_image_refs.call_args.args[0]
    src_frame = int(req.source_image_ref[0])
    tgt_frames = [int(r[0]) for r in req.target_image_refs]
    assert src_frame not in tgt_frames
    assert ds.get_segment_batch_from_image_refs.call_args.kwargs["enforce_target0_equals_source"] is False


def test_v6_episode_limit_is_hard_cap_even_if_budget_is_oversized():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV6(
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


def test_v6_enter_end_segment_manage_preload_scope():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV6(
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


def test_v6_factory_returns_scheduler():
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    sch = MultiSceneDatasetV4.create_train_scheduler_v6(
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
    assert isinstance(sch, TrainSchedulerV6)

