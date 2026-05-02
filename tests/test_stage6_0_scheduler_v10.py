from __future__ import annotations

from unittest.mock import MagicMock

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4, SegmentIndexV4
from datasets.train_scheduler_v10 import TrainSchedulerV10


def _make_sidx(*, scene_id: int, segment_id: int, base_frame: int = 10) -> SegmentIndexV4:
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
        num_cams=6,
        frame_indices=frames,
        test_frame_indices=[],
        train_frame_set=frozenset(frames),
        test_frame_set=frozenset(),
        keyframe_indices=keyframes,
        keyframe_to_frames=keyframe_to_frames,
        frame_to_keyframe=frame_to_keyframe,
        segment_first_frame_idx=frames[0],
        train_image_refs=tuple((f, c) for f in frames for c in range(6)),
        test_image_refs=tuple(),
    )


def _make_mock_dataset() -> MagicMock:
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    sidx = _make_sidx(scene_id=1, segment_id=0, base_frame=10)
    ds.get_segment_index = MagicMock(return_value=sidx)
    ds.get_segment_batch_from_image_refs = MagicMock(side_effect=lambda *args, **kwargs: {"ok": True, "request_meta": {}})
    ds.build_preload_hint = MagicMock(side_effect=lambda **kwargs: {"scene_id": kwargs["scene_id"], "segment_id": kwargs["segment_id"]})
    ds.submit_preload_hint = MagicMock()
    return ds


def test_v10_emits_structured_request_and_new_roles() -> None:
    ds = _make_mock_dataset()
    sch = TrainSchedulerV10(
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
        near_random_supervision_cfg={"enable": True, "frames_per_block": 1, "same_keyframe_only": True, "insufficient_policy": "skip", "role_name": "probe_near"},
        role_sampling_cfg={"first_step_role": "teacher", "teacher_prob": 0.4, "student_prob": 0.6, "teacher_resample_policy": "fixed_per_block"},
        targets_cfg={"weights": {"teacher_source": 1.0, "student_source": 1.0, "teacher_preserve": 0.1, "visited": 0.2, "near_random": 0.0}},
        history_record_cfg={"observed": {"trigger": "teacher_exit", "record_on_block_exit": False}, "runtime": {"trigger": "step_exit"}},
        camera_sampling_cfg={},
    )
    batch = sch.next_batch()
    meta = batch["request_meta"]
    assert str(meta["scheduler_version"]) == "v10"
    assert "scheduler_request_v10" in meta
    req = meta["scheduler_request_v10"]
    assert "live_teacher_bridge" in req
    assert "train_targets" in req
    assert "probe_targets" in req
    roles = [str(x) for x in meta.get("target_frame_roles", [])]
    assert "teacher_preserve" not in roles
    assert "teacher_anchor" in roles or "teacher_source" in roles

