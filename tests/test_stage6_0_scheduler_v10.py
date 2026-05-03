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
        frame_selection_cfg={
            "teacher_frame_policy": "random_within_keyframe",
            "student_cycle_policy": "cycle",
            "skip_student_if_single_source": True,
            "fallback_step_type_if_no_student": "teacher_bootstrap",
            "fallback_step_type_if_no_committed_history": "student_self",
        },
        step_program_cfg={"mode": "fixed_cycle", "sequence": ["teacher_bootstrap", "student_self"]},
        supervision_cfg={
            "probe_near": {"enable": True, "frames_per_block": 1},
            "history_visited": {"max_targets": 1, "sampling_policy": "most_recent"},
        },
        history_record_cfg={"observed": {"trigger": "teacher_update_exit"}, "runtime": {"trigger": "every_state_update_exit"}},
        bridge_cfg={"student_steps_use_live_bridge": True},
        probe_cfg={"enable": True, "frames_per_block": 1, "same_keyframe_only": True},
    )
    batch = sch.next_batch()
    meta = batch["request_meta"]
    assert str(meta["scheduler_version"]) == "v10"
    assert "teacher_obs" in meta
    assert "student_prop" in meta
    assert "supervision" in meta
    req = meta["scheduler_request_v10"]
    assert "live_teacher_bridge" in req  # compat block still present
    assert meta["scheduler/v10_is_compat_v9"] == 0.0
    roles = [str(x) for x in meta.get("target_frame_roles", [])]
    assert "teacher_preserve" not in roles
    assert "teacher_anchor" in roles or "teacher_source" in roles
