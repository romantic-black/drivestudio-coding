from __future__ import annotations

from unittest.mock import MagicMock

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4, SegmentIndexV4
from datasets.train_scheduler_v7 import TrainSchedulerV7


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


def _make_mock_dataset() -> MagicMock:
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    ds.get_segment_index = MagicMock(return_value=_make_sidx(scene_id=1, segment_id=0, base_frame=10))
    ds.get_segment_batch_from_image_refs = MagicMock(return_value={"ok": True})
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


def test_v7_build_segment_episode_starts_tail_aligned():
    starts = TrainSchedulerV7._build_segment_episode_starts(num_keyframes=6, e_blocks=3, window_keyframes=5)
    assert starts == [0, 1]


def test_v7_next_batch_rolling_chain_and_aligned_info():
    ds = _make_mock_dataset()
    sch = TrainSchedulerV7(
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
    )

    req_sources = []
    for _ in range(3):
        b = sch.next_batch()
        info = b["_scheduler_v4_aligned_info"]
        req = ds.get_segment_batch_from_image_refs.call_args.args[0]
        req_sources.append(int(req.source_image_ref[0]))
        assert req.source_image_refs is not None
        assert len(req.source_image_refs) == 2
        assert set(int(x[1]) for x in req.source_image_refs) == {0, 1}
        assert int(info["U"]) == 1
        assert str(info["scheduler_version"]) == "v7"
        assert ds.get_segment_batch_from_image_refs.call_args.kwargs["enforce_target0_equals_source"] is True
    assert req_sources == [10, 11, 12]


def test_v7_round_robin_episode_interleave_plan():
    ds = _make_mock_dataset()
    ds.list_segment_ids = MagicMock(return_value=[0, 1])

    def _get_sidx(scene_id: int, segment_id: int):
        return _make_sidx(scene_id=scene_id, segment_id=segment_id, base_frame=10 if segment_id == 0 else 100)

    ds.get_segment_index = MagicMock(side_effect=_get_sidx)
    sch = TrainSchedulerV7(
        dataset=ds,
        steps_per_block=1,
        blocks_per_episode=3,
        total_target_frames=3,
        include_source_frame=True,
        frame_within_keyframe_policy="middle_frame",
        min_keyframes_required_policy="skip_if_less_than_window",
        traversal_mode="round_robin_episode_interleave",
        switch_after_episode=True,
        segment_order="ascending",
        scene_order="ascending",
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=None,
        emit_preload_hints=False,
        warm_next_block_exact=False,
        warm_next_episode_chain=False,
    )
    order = [(c.scene_id, c.segment_id, c.episode_start_keyframe_pos) for c in sch.episode_cursor_plan]
    assert order[:4] == [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]


def test_v7_preload_hints_emitted_for_episode_and_next_block():
    ds = _make_mock_dataset()
    sch = TrainSchedulerV7(
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
        emit_preload_hints=True,
        warm_next_block_exact=True,
        warm_next_episode_chain=True,
    )
    sch.next_batch()
    events = sch.pop_events()
    scopes = [e.get("hint_scope") for e in events if e.get("type") == "preload_hint"]
    assert "episode_chain_exact" in scopes
    assert "next_block_exact" in scopes
    assert ds.submit_preload_hint.call_count >= 2
    episode_hints = [
        e for e in events if e.get("type") == "preload_hint" and e.get("hint_scope") == "episode_chain_exact"
    ]
    assert len(episode_hints) == 1
    expected_frames = [11, 12, 13, 14, 15]
    expected_refs = [(f, cam) for f in expected_frames for cam in (0, 1)]
    assert episode_hints[0]["hint"]["future_image_refs"] == expected_refs


def test_v7_round_robin_updates_scope_each_episode():
    ds = _make_mock_dataset()
    ds.list_segment_ids = MagicMock(return_value=[0, 1])

    def _get_sidx(scene_id: int, segment_id: int):
        return _make_sidx(scene_id=scene_id, segment_id=segment_id, base_frame=10 if segment_id == 0 else 100)

    ds.get_segment_index = MagicMock(side_effect=_get_sidx)
    sch = TrainSchedulerV7(
        dataset=ds,
        steps_per_block=1,
        blocks_per_episode=3,
        total_target_frames=3,
        include_source_frame=True,
        frame_within_keyframe_policy="middle_frame",
        min_keyframes_required_policy="skip_if_less_than_window",
        traversal_mode="round_robin_episode_interleave",
        switch_after_episode=True,
        segment_order="ascending",
        scene_order="ascending",
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=None,
        emit_preload_hints=False,
        warm_next_block_exact=False,
        warm_next_episode_chain=False,
    )
    for _ in range(4):
        sch.next_batch()
    scopes = [tuple(c.args) for c in ds.set_preload_active_scope.call_args_list]
    assert (1, 0) in scopes
    assert (1, 1) in scopes


def test_v7_factory_returns_scheduler():
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    ds.get_segment_index = MagicMock(return_value=_make_sidx(scene_id=1, segment_id=0, base_frame=10))
    sch = MultiSceneDatasetV4.create_train_scheduler_v7(
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
    )
    assert isinstance(sch, TrainSchedulerV7)


def test_v7_step_major_round_robin_block_visits_and_events():
    ds = _make_mock_dataset()
    sch = TrainSchedulerV7(
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
    )

    source_frames = []
    block_idx_global = []
    block_idx_in_episode = []
    block_repeat_step = []
    for _ in range(6):
        batch = sch.next_batch()
        info = batch["_scheduler_v7_aligned_info"]
        source_frames.append(int(info["source_frame_idx"]))
        block_idx_global.append(int(info["block_idx_global"]))
        block_idx_in_episode.append(int(info["block_idx_in_episode"]))
        block_repeat_step.append(int(info["block_repeat_step"]))
        assert str(info["block_order"]) == "step_major"

    assert source_frames == [10, 11, 12, 10, 11, 12]
    assert block_idx_global == [0, 1, 2, 0, 1, 2]
    assert block_idx_in_episode == [0, 1, 2, 0, 1, 2]
    assert block_repeat_step == [1, 1, 1, 2, 2, 2]

    events = sch.pop_events()
    begin_blocks = [int(e["block_idx_in_episode"]) for e in events if e.get("type") == "block_begin"]
    end_blocks = [int(e["block_idx_in_episode"]) for e in events if e.get("type") == "block_end"]
    episode_end = [e for e in events if e.get("type") == "episode_end"]
    assert begin_blocks == [0, 1, 2]
    assert end_blocks == [0, 1, 2]
    assert len(episode_end) == 1


def test_v7_step_major_switch_interval_steps_keeps_total_and_per_block_counts():
    ds = _make_mock_dataset()
    sch = TrainSchedulerV7(
        dataset=ds,
        steps_per_block=5,
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
        step_major_switch_interval_steps=2,
    )

    source_frames = []
    for _ in range(15):
        batch = sch.next_batch()
        info = batch["_scheduler_v7_aligned_info"]
        source_frames.append(int(info["source_frame_idx"]))
        assert int(info["step_major_switch_interval_steps"]) == 2

    assert source_frames == [10, 10, 11, 11, 12, 12, 10, 10, 11, 11, 12, 12, 10, 11, 12]

    final_info = sch.get_current_info()
    assert final_info["block_update_counts"] == []

    events = sch.pop_events()
    end_blocks = [int(e["block_idx_in_episode"]) for e in events if e.get("type") == "block_end"]
    assert end_blocks == [0, 1, 2]
