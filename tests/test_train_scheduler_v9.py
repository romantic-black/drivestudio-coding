from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4, SegmentIndexV4
from datasets.train_scheduler_v9 import TrainSchedulerV9
from tools.train_minimal_streetforward_stage4_3_v8_common import build_train_scheduler_v9_from_cfg


def _make_sidx_single_frame(*, scene_id: int, segment_id: int, base_frame: int = 10) -> SegmentIndexV4:
    keyframes = [0, 1, 2, 3, 4, 5]
    keyframe_to_frames = {k: [base_frame + k] for k in keyframes}
    frame_to_keyframe = {base_frame + k: k for k in keyframes}
    frames = [base_frame + k for k in keyframes]
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


def _make_sidx_multi_frame(*, scene_id: int, segment_id: int, base_frame: int = 10) -> SegmentIndexV4:
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


def _make_mock_dataset(*, multi_frame: bool) -> MagicMock:
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    sidx = (
        _make_sidx_multi_frame(scene_id=1, segment_id=0, base_frame=10)
        if multi_frame
        else _make_sidx_single_frame(scene_id=1, segment_id=0, base_frame=10)
    )
    ds.get_segment_index = MagicMock(return_value=sidx)
    ds.get_segment_batch_from_image_refs = MagicMock(side_effect=lambda *args, **kwargs: {"ok": True, "request_meta": {}})
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


def _build_scheduler(ds: MagicMock, **kwargs) -> TrainSchedulerV9:
    role_sampling_cfg = {
        "first_step_role": "teacher",
        "force_teacher_on_block_entry": True,
        "teacher_prob": 0.4,
        "student_prob": 0.6,
        "teacher_frame_policy": "random_within_keyframe",
        "student_frame_policy": "random_within_same_keyframe_except_teacher",
        "skip_student_if_single_source": True,
        "skip_student_if_no_prior": True,
        "fallback_to_teacher": True,
    }
    role_sampling_cfg.update(kwargs.get("role_sampling_cfg", {}))
    return TrainSchedulerV9(
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
        block_order=str(kwargs.get("block_order", "step_major")),
        step_major_switch_interval_steps=int(kwargs.get("step_major_switch_interval_steps", 1)),
        target_policy="visited_episode_frames",
        reset_policy="episode_end",
        near_random_supervision_cfg=kwargs.get("near_random_supervision_cfg"),
        role_sampling_cfg=role_sampling_cfg,
        targets_cfg={
            "weights": {
                "teacher_source": 1.0,
                "student_source": 1.0,
                "teacher_preserve": 0.3,
                "visited": 0.2,
                "near_random": 0.2,
            }
        },
        history_record_cfg={
            "observed": {
                "trigger": "teacher_exit",
                "record_on_block_exit": False,
            },
            "runtime": {
                "trigger": "step_exit",
            },
        },
        camera_sampling_cfg=kwargs.get("camera_sampling_cfg", {}),
    )


def test_v9_single_source_keyframe_all_teacher():
    ds = _make_mock_dataset(multi_frame=False)
    sch = _build_scheduler(ds)
    for _ in range(4):
        batch = sch.next_batch()
        meta = batch["request_meta"]
        assert str(meta["stage5_5_role"]) == "teacher"
        assert bool(meta["stage5_5_has_student"]) is False
        assert bool(meta["history_record/record_observed_on_step_exit"]) is True
        assert bool(meta["stage5_5_force_teacher_on_block_entry"]) is True


def test_v9_rejects_unsupported_teacher_resample_policy():
    ds = _make_mock_dataset(multi_frame=True)
    with pytest.raises(ValueError, match="teacher_resample_policy"):
        _build_scheduler(
            ds,
            role_sampling_cfg={
                "teacher_resample_policy": "resample_per_step",
            },
        )


def test_v9_accepts_active_episode_cams_near_random_policy():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(
        ds,
        near_random_supervision_cfg={
            "enable": True,
            "frames_per_block": 1,
            "same_keyframe_only": True,
            "insufficient_policy": "skip",
            "camera_policy": "active_episode_cams",
        },
        camera_sampling_cfg={
            "enable": True,
            "scope": "episode",
            "policy": "explicit_groups",
            "camera_groups": [[0, 1, 2]],
            "group_order": "cycle",
            "freeze_within_episode": True,
            "apply_to_source": True,
            "apply_to_teacher": True,
            "apply_to_target": True,
            "apply_to_history_record": True,
            "apply_to_preload": True,
        },
    )
    assert str(sch.near_random_camera_policy_v9) == "active_episode_cams"
    assert str(sch.near_random_camera_policy) == "all_cams"


def test_v9_first_step_in_block_must_be_teacher():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(ds)
    batch0 = sch.next_batch()
    meta = batch0["request_meta"]
    assert str(meta["stage5_5_role"]) == "teacher"
    assert bool(meta["history_record/record_observed_on_step_exit"]) is True


def test_v9_student_target_contains_teacher_preserve():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(
        ds,
        block_order="step_major",
        blocks_per_episode=2,
        total_target_frames=2,
        role_sampling_cfg={
            "teacher_prob": 0.2,
            "student_prob": 0.8,
            "skip_student_if_no_prior": False,
            "force_teacher_on_block_entry": False,
        },
    )
    batch = None
    for _ in range(8):
        with patch.object(TrainSchedulerV9, "_weighted_role_sample", return_value="student"):
            candidate = sch.next_batch()
        roles = [str(x) for x in candidate["request_meta"]["target_frame_roles"]]
        if "student_source" in roles:
            batch = candidate
            break
    assert batch is not None, "expected at least one student batch in the first 8 scheduler steps"
    roles = [str(x) for x in batch["request_meta"]["target_frame_roles"]]
    assert "teacher_preserve" in roles
    assert bool(batch["request_meta"]["history_record/record_observed_on_step_exit"]) is False


def test_v9_step_major_block_entry_first_step_forces_teacher():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(
        ds,
        block_order="step_major",
        blocks_per_episode=2,
        total_target_frames=2,
        steps_per_block=4,
        step_major_switch_interval_steps=2,
        role_sampling_cfg={
            "teacher_prob": 0.2,
            "student_prob": 0.8,
            "skip_student_if_no_prior": False,
            "force_teacher_on_block_entry": True,
        },
    )
    with patch.object(TrainSchedulerV9, "_weighted_role_sample", return_value="student"):
        b0 = sch.next_batch()["request_meta"]
        b1 = sch.next_batch()["request_meta"]
        b2 = sch.next_batch()["request_meta"]
        b3 = sch.next_batch()["request_meta"]

    assert str(b0["stage5_5_role"]) == "teacher"
    assert int(b0["stage5_5_block_entry_step"]) == 0
    assert float(b0["scheduler_v9/block_entry_teacher"]) == 1.0
    assert bool(b0["stage5_5_force_teacher_on_block_entry"]) is True

    assert str(b1["stage5_5_role"]) == "student"
    assert int(b1["stage5_5_block_entry_step"]) == 1
    assert float(b1["scheduler_v9/block_entry_teacher"]) == 0.0

    assert str(b2["stage5_5_role"]) == "teacher"
    assert int(b2["stage5_5_block_entry_step"]) == 0
    assert float(b2["scheduler_v9/block_entry_teacher"]) == 1.0

    assert str(b3["stage5_5_role"]) == "student"
    assert int(b3["stage5_5_block_entry_step"]) == 1
    assert float(b3["scheduler_v9/block_entry_teacher"]) == 0.0


def test_v9_block_exit_observed_record_disabled():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(ds)
    batch = sch.next_batch()
    assert bool(batch["request_meta"]["history_record/record_observed_on_block_exit"]) is False


def test_v9_teacher_exit_records_observed():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(ds)
    batch = sch.next_batch()
    meta = batch["request_meta"]
    assert str(meta["history_record/observed_record_trigger"]) == "teacher_exit"
    assert len(meta["history_record/observed_record_image_refs"]) > 0


def test_v9_fast_fail_requires_teacher_exit_trigger():
    ds = _make_mock_dataset(multi_frame=True)
    with pytest.raises(ValueError, match="teacher_exit"):
        TrainSchedulerV9(
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
            role_sampling_cfg={
                "first_step_role": "teacher",
                "teacher_prob": 0.4,
                "student_prob": 0.6,
            },
            targets_cfg={"weights": {}},
            history_record_cfg={
                "observed": {
                    "trigger": "block_exit",
                    "record_on_block_exit": False,
                }
            },
        )


def test_build_train_scheduler_v9_reads_fixed_scene_from_scheduler_v9_traversal():
    cfg = OmegaConf.create(
        {
            "data": {
                "train_scene_ids": [7],
            },
            "scheduler_v8": {
                "traversal": {
                    "fixed_scene_id": 123,
                    "fixed_segment_id": 456,
                }
            },
            "scheduler_v9": {
                "enable": True,
                "block": {
                    "steps_per_block": 2,
                },
                "episode": {
                    "blocks_per_episode": 3,
                    "total_target_frames": 3,
                    "include_source_frame": True,
                    "target_policy": "visited_episode_frames",
                    "block_source_frame_policy": "fixed_once_per_episode",
                    "frame_within_keyframe_policy": "middle_frame",
                    "min_keyframes_required_policy": "skip_if_less_than_window",
                },
                "traversal": {
                    "mode": "linear_scene_segment",
                    "switch_after_episode": True,
                    "fixed_scene_id": 7,
                    "fixed_segment_id": 3,
                    "segment_order": "ascending",
                    "scene_order": "ascending",
                },
                "preload": {
                    "emit_hints": False,
                    "warm_next_block_exact": False,
                    "warm_next_episode_chain": False,
                },
                "execution": {
                    "block_order": "step_major",
                    "step_major_switch_interval_steps": 1,
                    "reset_policy": "episode_end",
                },
                "role_sampling": {
                    "first_step_role": "teacher",
                    "teacher_prob": 0.4,
                    "student_prob": 0.6,
                },
                "targets": {
                    "weights": {
                        "teacher_source": 1.0,
                        "student_source": 1.0,
                        "teacher_preserve": 0.3,
                        "visited": 0.2,
                        "near_random": 0.2,
                    }
                },
                "history_record": {
                    "observed": {
                        "trigger": "teacher_exit",
                        "record_on_block_exit": False,
                    },
                    "runtime": {
                        "trigger": "step_exit",
                    },
                },
                "camera_sampling": {
                    "enable": True,
                    "scope": "episode",
                    "policy": "explicit_groups",
                    "camera_groups": [[0, 1, 2], [1, 2, 3]],
                    "group_order": "shuffle_cycle",
                    "freeze_within_episode": True,
                    "apply_to_source": True,
                    "apply_to_teacher": True,
                    "apply_to_target": True,
                    "apply_to_history_record": True,
                    "apply_to_preload": True,
                },
            },
        }
    )
    ds = MagicMock(spec=MultiSceneDatasetV4)
    sentinel = object()
    ds.create_train_scheduler_v9 = MagicMock(return_value=sentinel)
    out = build_train_scheduler_v9_from_cfg(cfg, ds)
    assert out is sentinel
    _, kwargs = ds.create_train_scheduler_v9.call_args
    assert int(kwargs["fixed_scene_id"]) == 7
    assert int(kwargs["fixed_segment_id"]) == 3
    assert bool(kwargs["camera_sampling_cfg"]["enable"]) is True


def test_v9_resolve_visited_prefers_last_teacher_frame():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(ds)
    sch._ensure_episode_state()
    st = sch.current_episode_state
    assert st is not None
    st["visited_block_indices"] = {0, 2}
    st["block_last_teacher_frame_indices"][0] = 110
    st["block_last_teacher_frame_indices"][2] = -1
    st["block_current_source_frame_indices"] = [999, 998, 997]
    out = sch._resolve_visited_target_frames(st, block_idx=1)
    assert out[0] == 110
    assert out[1] == int(st["frame_chain"][2])


def test_v9_materialize_current_batch_injects_v9_request_meta():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(ds)
    batch = sch.materialize_current_batch_without_advance()
    meta = batch["request_meta"]
    assert batch["_scheduler_v9_peek"] is True
    assert str(meta["stage5_5_role"]) == "teacher"
    assert "stage5_5_teacher_frame_idx" in meta
    assert "history_record/observed_record_trigger" in meta
    assert "history_record/runtime_record_trigger" in meta
    assert "target_frame_loss_base_weights" in meta
    assert "target_image_loss_base_weights" in meta
    assert "camera_sampling/active_cam_ids" in meta


def test_v9_camera_group_fixed_within_episode_and_switches_across_episodes():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(
        ds,
        steps_per_block=1,
        blocks_per_episode=2,
        total_target_frames=2,
        camera_sampling_cfg={
            "enable": True,
            "scope": "episode",
            "policy": "explicit_groups",
            "camera_groups": [[0, 1, 2], [1, 2, 3]],
            "group_order": "cycle",
            "freeze_within_episode": True,
            "apply_to_source": True,
            "apply_to_teacher": True,
            "apply_to_target": True,
            "apply_to_history_record": True,
            "apply_to_preload": True,
            "camera_names": {0: "front_left", 1: "front", 2: "front_right", 3: "rear_right"},
        },
    )
    b0 = sch.next_batch()["request_meta"]
    b1 = sch.next_batch()["request_meta"]
    b2 = sch.next_batch()["request_meta"]

    assert [int(x) for x in b0["camera_sampling/active_cam_ids"]] == [0, 1, 2]
    assert [int(x) for x in b1["camera_sampling/active_cam_ids"]] == [0, 1, 2]
    assert [int(x) for x in b2["camera_sampling/active_cam_ids"]] == [1, 2, 3]

    source_refs = [tuple(x) for x in b0["source_image_refs"]]
    teacher_refs = [tuple(x) for x in b0["stage5_5_teacher_image_refs"]]
    target_refs = [tuple(x) for x in b0["target_image_refs"]]
    target_frames = [int(x) for x in b0["target_frame_indices"]]
    target_roles = [str(x) for x in b0["target_image_roles"]]
    target_weights = [float(x) for x in b0["target_image_loss_base_weights"]]
    observed_refs = [tuple(x) for x in b0["history_record/observed_record_image_refs"]]
    runtime_refs = [tuple(x) for x in b0["history_record/runtime_record_image_refs"]]

    assert len(source_refs) == 3
    assert len(teacher_refs) == 3
    assert len(target_refs) == len(target_frames) * 3
    assert len(target_roles) == len(target_refs)
    assert len(target_weights) == len(target_refs)
    assert observed_refs == teacher_refs
    assert runtime_refs == source_refs
    assert float(b0["scheduler_v9/active_num_cams"]) == 3.0
    assert [str(x) for x in b0["camera_sampling/active_cam_names"]] == ["front_left", "front", "front_right"]


def test_v9_preload_hints_only_include_active_episode_cams():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(
        ds,
        steps_per_block=1,
        blocks_per_episode=2,
        total_target_frames=2,
        emit_preload_hints=True,
        warm_next_block_exact=True,
        warm_next_episode_chain=True,
        camera_sampling_cfg={
            "enable": True,
            "scope": "episode",
            "policy": "explicit_groups",
            "camera_groups": [[2, 3, 4]],
            "group_order": "cycle",
            "freeze_within_episode": True,
            "apply_to_source": True,
            "apply_to_teacher": True,
            "apply_to_target": True,
            "apply_to_history_record": True,
            "apply_to_preload": True,
        },
    )
    batch = sch.next_batch()
    meta = batch["request_meta"]
    assert ds.build_preload_hint.call_count > 0
    scopes = [str(call.kwargs["scope"]) for call in ds.build_preload_hint.call_args_list]
    assert "episode_chain_exact" in scopes
    assert "next_block_exact" in scopes
    next_exact_refs = set()
    for call in ds.build_preload_hint.call_args_list:
        refs = [tuple(x) for x in call.kwargs["future_image_refs"]]
        assert len(refs) > 0
        assert set(int(ref[1]) for ref in refs).issubset({2, 3, 4})
        if str(call.kwargs["scope"]) == "next_block_exact":
            next_exact_refs.update((int(ref[0]), int(ref[1])) for ref in refs)
    required_refs = set(
        [tuple(x) for x in list(meta["source_image_refs"])]
        + [tuple(x) for x in list(meta["stage5_5_teacher_image_refs"])]
        + [tuple(x) for x in list(meta["target_image_refs"])]
    )
    assert required_refs.issubset(next_exact_refs)


def test_v9_target_loss_weights_have_frame_and_image_levels():
    ds = _make_mock_dataset(multi_frame=True)
    sch = _build_scheduler(ds)
    batch = sch.next_batch()
    meta = batch["request_meta"]
    frame_weights = [float(x) for x in meta["target_frame_loss_base_weights"]]
    image_weights = [float(x) for x in meta["target_image_loss_base_weights"]]
    target_frames = [int(x) for x in meta["target_frame_indices"]]
    target_refs = [tuple(x) for x in meta["target_image_refs"]]
    assert len(frame_weights) == len(target_frames)
    assert len(image_weights) == len(target_refs)
    num_cams = int(len(target_refs) // max(len(target_frames), 1))
    expected: list[float] = []
    for w in frame_weights:
        for _ in range(num_cams):
            expected.append(float(w))
    assert image_weights == expected
    assert [float(x) for x in meta["target_loss_base_weights"]] == frame_weights
