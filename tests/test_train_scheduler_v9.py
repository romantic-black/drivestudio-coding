from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4, SegmentIndexV4
from datasets.train_scheduler_v9 import TrainSchedulerV9


def _make_sidx_multi_frame_per_keyframe(*, scene_id: int, segment_id: int, base_frame: int = 10) -> SegmentIndexV4:
    keyframes = [0, 1, 2, 3, 4, 5]
    keyframe_to_frames = {
        int(k): [int(base_frame + k * 10 + i) for i in range(3)]
        for k in keyframes
    }
    frame_to_keyframe = {}
    frames = []
    for k in keyframes:
        for f in keyframe_to_frames[int(k)]:
            frame_to_keyframe[int(f)] = int(k)
            frames.append(int(f))
    return SegmentIndexV4(
        scene_id=scene_id,
        segment_id=segment_id,
        num_cams=2,
        frame_indices=frames,
        test_frame_indices=[999],
        train_frame_set=frozenset(frames),
        test_frame_set=frozenset({999}),
        keyframe_indices=keyframes,
        keyframe_to_frames=keyframe_to_frames,
        frame_to_keyframe=frame_to_keyframe,
        segment_first_frame_idx=frames[0],
        train_image_refs=tuple((f, 0) for f in frames),
        test_image_refs=((999, 0),),
    )


def _make_mock_dataset(sidx: SegmentIndexV4) -> MagicMock:
    ds = MagicMock(spec=MultiSceneDatasetV4)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[int(sidx.scene_id)])
    ds.list_segment_ids = MagicMock(return_value=[int(sidx.segment_id)])
    ds.get_segment_index = MagicMock(return_value=sidx)
    ds._assemble_segment_batch_from_v9_request = MagicMock(side_effect=lambda **kwargs: {"request_meta": {}})

    def _validate(scene_id: int, segment_id: int, image_ref, purpose: str) -> None:
        assert int(scene_id) == int(sidx.scene_id)
        assert int(segment_id) == int(sidx.segment_id)
        frame_idx, cam_idx = int(image_ref[0]), int(image_ref[1])
        assert 0 <= cam_idx < int(sidx.num_cams)
        if purpose == "train":
            assert frame_idx in sidx.train_frame_set
        elif purpose == "test":
            assert frame_idx in sidx.test_frame_set
        else:
            raise AssertionError(f"unexpected purpose={purpose!r}")

    ds.validate_image_ref = MagicMock(side_effect=_validate)
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
    phase = str(kwargs.get("phase", "phase_A_block_local_unroll"))
    phase_a_cfg = kwargs.get(
        "phase_a_cfg",
        {
            "block": {"inner_K_choices": [3], "inner_K_probs": [1.0]},
            "nearby_supervision": {
                "enable": True,
                "frames_per_block": 2,
                "adjacent_radius": 1,
                "random_fill": True,
                "same_keyframe_only": True,
                "insufficient_policy": "skip",
                "exclude_source_frame": True,
                "exclude_existing_block_loss_frames": True,
                "camera_policy": "all_cams",
                "apply_final_step_only": True,
                "max_refs_per_step": 12,
            },
        },
    )
    phase_b_cfg = kwargs.get(
        "phase_b_cfg",
        {
            "rollout": {
                "K_choices": [3],
                "K_probs": [1.0],
                "sample_event_frames": "random_blocks_in_episode",
                "event_order": "chronological",
                "distinct_event_frames": True,
            },
            "prefix_render": {
                "policy": "current_plus_random_previous",
                "intermediate_views": 2,
                "final_views": 3,
                "max_refs_per_step": 12,
            },
            "query_observation": {
                "enable": True,
                "query_frame_policy": "heldout_inside_event_span",
                "frames_per_rollout": 1,
                "cameras_per_frame": "all_cams",
                "exclude_event_frames": True,
            },
        },
    )
    return TrainSchedulerV9(
        dataset=ds,
        phase=phase,  # type: ignore[arg-type]
        steps_per_block=1,
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
        block_order=str(kwargs.get("block_order", "block_major")),
        step_major_switch_interval_steps=1,
        target_policy="visited_episode_frames",
        reset_policy="episode_end",
        block_source_frame_policy=str(kwargs.get("block_source_frame_policy", "fixed_once_per_episode")),
        episode_source_mode="keyframes",
        phase_a_cfg=phase_a_cfg,
        phase_b_cfg=phase_b_cfg,
        leakage_check_cfg={
            "nearby_not_in_evidence": True,
            "query_not_in_evidence": True,
            "aux_not_in_evidence": True,
            "role_count_match_required": True,
            "forbid_test_refs_in_train": True,
        },
    )


def test_phase_a_nearby_same_keyframe():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(_make_mock_dataset(sidx))
    batch = sch.next_batch()
    plan = batch["_scheduler_v9"]
    final_step = plan["steps"][-1]
    source_kf = int(final_step["source_keyframe_idx"])
    nearby_frames = [int(x) for x in final_step["nearby_frame_indices"]]
    assert nearby_frames
    assert all(int(sidx.frame_to_keyframe[int(f)]) == source_kf for f in nearby_frames)


def test_phase_a_nearby_excludes_source():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(_make_mock_dataset(sidx))
    batch = sch.next_batch()
    final_step = batch["_scheduler_v9"]["steps"][-1]
    assert int(final_step["source_frame_idx"]) not in [int(x) for x in final_step["nearby_frame_indices"]]


def test_phase_a_nearby_not_in_evidence():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(_make_mock_dataset(sidx))
    batch = sch.next_batch()
    meta = batch["request_meta"]
    evidence = {tuple(x) for refs in meta["evidence_refs_by_step"] for x in refs}
    nearby = {tuple(x) for refs in meta["nearby_loss_refs_by_step"] for x in refs}
    assert nearby
    assert nearby.isdisjoint(evidence)


def test_phase_a_final_step_only():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(_make_mock_dataset(sidx))
    batch = sch.next_batch()
    nearby_by_step = batch["request_meta"]["nearby_loss_refs_by_step"]
    assert nearby_by_step[0] == []
    assert nearby_by_step[1] == []
    assert nearby_by_step[2] != []


def test_phase_b_query_not_in_evidence():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(_make_mock_dataset(sidx), phase="phase_B_viewset_rollout")
    with patch("datasets.train_scheduler_v9.random.sample", side_effect=lambda population, k: list(population)[: int(k)]):
        batch = sch.next_batch()
    meta = batch["request_meta"]
    evidence = {tuple(x) for refs in meta["evidence_refs_by_step"] for x in refs}
    query = {tuple(x) for x in meta["query_label_refs"]}
    assert query
    assert query.isdisjoint(evidence)


def test_phase_b_event_order():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(_make_mock_dataset(sidx), phase="phase_B_viewset_rollout")
    with patch("datasets.train_scheduler_v9.random.sample", side_effect=lambda population, k: list(reversed(list(population)))[: int(k)]):
        batch = sch.next_batch()
    frames = [int(step["source_frame_idx"]) for step in batch["_scheduler_v9"]["steps"]]
    assert frames == sorted(frames)


def test_role_count_match():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(_make_mock_dataset(sidx))
    batch = sch.next_batch()
    meta = batch["request_meta"]
    assert len(meta["target_image_refs"]) == len(meta["target_image_roles"])
    assert meta["leakage_check"]["target_role_count_match"] is True


def test_no_test_refs_in_train():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    ds = _make_mock_dataset(sidx)
    sch = _build_scheduler(ds, phase="phase_B_viewset_rollout")
    batch = sch.next_batch()
    meta = batch["request_meta"]
    all_refs = list(meta["flat_evidence_refs"]) + list(meta["flat_loss_refs"])
    assert all(int(ref[0]) in sidx.train_frame_set for ref in all_refs)
    assert all(int(ref[0]) not in sidx.test_frame_set for ref in all_refs)


def test_v9_factory_returns_scheduler():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    ds = _make_mock_dataset(sidx)
    sch = MultiSceneDatasetV4.create_train_scheduler_v9(
        ds,
        phase="phase_A_block_local_unroll",
        steps_per_block=1,
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
        block_order="block_major",
        target_policy="visited_episode_frames",
        reset_policy="episode_end",
        phase_a_cfg={
            "block": {"inner_K_choices": [2], "inner_K_probs": [1.0]},
            "nearby_supervision": {"enable": True},
        },
    )
    assert isinstance(sch, TrainSchedulerV9)
