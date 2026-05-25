from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4, SegmentIndexV4
from datasets.train_scheduler_v9 import TrainSchedulerV9


def _make_sidx_multi_frame_per_keyframe(
    *,
    scene_id: int,
    segment_id: int,
    base_frame: int = 10,
    num_keyframes: int = 6,
) -> SegmentIndexV4:
    keyframes = list(range(int(num_keyframes)))
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
            "episode": {
                "reset_vsm_on_episode_end": True,
            },
            "masks": {
                "vsm_scope": "bg_rigid",
            },
        },
    )
    return TrainSchedulerV9(
        dataset=ds,
        phase=phase,  # type: ignore[arg-type]
        steps_per_block=int(kwargs.get("steps_per_block", 1)),
        blocks_per_episode=int(kwargs.get("blocks_per_episode", 3)),
        include_source_frame=True,
        frame_within_keyframe_policy="middle_frame",
        min_keyframes_required_policy=str(kwargs.get("min_keyframes_required_policy", "skip_if_less_than_window")),
        traversal_mode="linear_scene_segment",
        switch_after_episode=True,
        segment_order="ascending",
        scene_order="ascending",
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
        emit_preload_hints=bool(kwargs.get("emit_preload_hints", False)),
        warm_next_block_exact=False,
        warm_next_episode_chain=False,
        warm_v9_role_refs=bool(kwargs.get("warm_v9_role_refs", True)),
        block_order=str(kwargs.get("block_order", "block_major")),
        step_major_switch_interval_steps=int(kwargs.get("step_major_switch_interval_steps", 1)),
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


def _strict_tbptt_phase_b_cfg(k: int = 2) -> dict:
    return {
        "rollout": {
            "mode": "episode_stream_tbptt",
            "K_choices": [int(k)],
            "K_probs": [1.0],
            "sample_event_frames": "sequential_blocks_in_episode",
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
        "episode": {"reset_vsm_on_episode_end": True},
        "masks": {"vsm_scope": "bg_rigid"},
    }


def _grouped_repeat_phase_b_cfg(
    *,
    repeats_per_block: int = 2,
    blocks_per_chunk: int = 4,
    allow_short_final_chunk: bool = True,
) -> dict:
    return {
        "rollout": {
            "mode": "episode_grouped_repeat_tbptt",
            "repeat_patterns": [
                {
                    "name": f"r{int(repeats_per_block)}_b{int(blocks_per_chunk)}",
                    "repeats_per_block": int(repeats_per_block),
                    "blocks_per_chunk": int(blocks_per_chunk),
                    "prob": 1.0,
                }
            ],
            "max_inner_K": 8,
            "sample_event_frames": "sequential_blocks_in_episode",
            "event_order": "chronological",
            "distinct_event_frames": True,
            "repeat_source_frame_policy": "fixed_within_block",
            "repeat_memory_write_policy": "first_repeat_only",
            "evidence_recompute_policy": "every_repeat",
            "allow_short_final_chunk": bool(allow_short_final_chunk),
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
            "allow_empty_on_last_chunk": False,
        },
        "episode": {"reset_vsm_on_episode_end": True},
        "masks": {"vsm_scope": "bg_rigid"},
    }


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


def test_v9_emits_mask_reset_and_flat_non_evidence_metadata():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(_make_mock_dataset(sidx), phase="phase_B_viewset_rollout")
    batch = sch.next_batch()
    meta = batch["request_meta"]
    assert meta["mask_policy"]["phase_b_vsm_scope"] == "bg_rigid"
    assert meta["vsm_reset_policy"]["reset_vsm_on_episode_end"] is True
    render_group = next(group for group in meta["role_groups"] if group["role"] == "render_loss")
    assert render_group["mask_policy"] == "non_sky_non_egocar"
    assert isinstance(meta["mask_policy"], dict)
    assert "flat_non_evidence_refs" in meta
    assert set(tuple(x) for x in meta["flat_loss_refs"]) == set(tuple(x) for x in meta["flat_non_evidence_refs"])


def test_v9_preload_hint_includes_actual_role_refs():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    ds = _make_mock_dataset(sidx)
    sch = _build_scheduler(ds, block_order="step_major", emit_preload_hints=True)
    sch.next_batch()
    assert sch._v9_prefetched_plan is not None
    expected = set(sch._preload_refs_from_v9_plan(sch._v9_prefetched_plan))
    role_hint_calls = [
        call
        for call in ds.submit_preload_hint.call_args_list
        if call.kwargs["hint_scope"] == "v9_role_refs"
    ]
    assert role_hint_calls
    hinted = {
        tuple(x)
        for call in role_hint_calls
        for x in call.kwargs["hint"]["future_image_refs"]
    }
    assert expected <= hinted


def test_v9_preload_hint_can_disable_role_refs():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    ds = _make_mock_dataset(sidx)
    sch = _build_scheduler(
        ds,
        block_order="step_major",
        emit_preload_hints=True,
        warm_v9_role_refs=False,
    )
    sch.next_batch()
    assert all(
        call.kwargs["hint_scope"] != "v9_role_refs"
        for call in ds.submit_preload_hint.call_args_list
    )


def test_v9_use_available_keyframes_when_segment_shorter_than_episode():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=2)
    sch = _build_scheduler(
        _make_mock_dataset(sidx),
        blocks_per_episode=8,
        min_keyframes_required_policy="use_available_if_less_than_window",
    )
    assert sch.epoch_plan[0]["total_blocks"] == 2
    batch0 = sch.next_batch()
    assert batch0["_scheduler_v9"]["keyframe_window"] == [0, 1]
    assert len(batch0["_scheduler_v9"]["frame_chain"]) == 2
    assert batch0["_scheduler_v9"]["steps"][0]["block_idx"] == 0
    batch1 = sch.next_batch()
    assert batch1["_scheduler_v9"]["steps"][0]["block_idx"] == 1


def test_v9_random_without_replacement_block_order_visits_each_block_once_per_round():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(
        _make_mock_dataset(sidx),
        blocks_per_episode=4,
        steps_per_block=2,
        block_order="random_without_replacement",
    )
    sch._ensure_episode_state()
    order = list(sch.current_episode_state["episode_block_visit_order"])  # type: ignore[index]
    assert len(order) == 8
    assert sorted(order[:4]) == [0, 1, 2, 3]
    assert sorted(order[4:]) == [0, 1, 2, 3]


def test_v9_random_with_replacement_block_order_allows_repeats():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    sch = _build_scheduler(
        _make_mock_dataset(sidx),
        blocks_per_episode=3,
        steps_per_block=3,
        block_order="random_with_replacement",
    )
    sch._ensure_episode_state()
    order = list(sch.current_episode_state["episode_block_visit_order"])  # type: ignore[index]
    assert len(order) == 9
    assert all(0 <= int(x) < 3 for x in order)
    assert sch._block_order_requires_all_blocks_completed() is False


@pytest.mark.parametrize(
    "phase_a_cfg",
    [
        {"mode": "other", "block": {"inner_K_choices": [2], "inner_K_probs": [1.0]}},
        {
            "block": {
                "inner_K_choices": [2],
                "inner_K_probs": [1.0],
                "block_loss_policy": "unsupported",
            }
        },
        {
            "block": {"inner_K_choices": [2], "inner_K_probs": [1.0], "repeat_block_iteration": False}
        },
        {
            "block": {
                "inner_K_choices": [2],
                "inner_K_probs": [1.0],
                "source_frame_policy": "moving",
            }
        },
        {
            "block": {"inner_K_choices": [2], "inner_K_probs": [1.0]},
            "nearby_supervision": {"apply_every_step": True},
        },
        {
            "block": {"inner_K_choices": [2], "inner_K_probs": [1.0]},
            "nearby_supervision": {"add_to_evidence_refs": True},
        },
    ],
)
def test_phase_a_rejects_unsupported_p0_config_values(phase_a_cfg):
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    with pytest.raises(ValueError):
        _build_scheduler(_make_mock_dataset(sidx), phase_a_cfg=phase_a_cfg)


def test_phase_b_rejects_non_bg_rigid_vsm_scope():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    bad_phase_b = {
        "rollout": {"K_choices": [2], "K_probs": [1.0]},
        "prefix_render": {"policy": "current_plus_random_previous"},
        "query_observation": {"cameras_per_frame": "all_cams"},
        "masks": {"vsm_scope": "all"},
    }
    with pytest.raises(ValueError):
        _build_scheduler(_make_mock_dataset(sidx), phase_b_cfg=bad_phase_b)


def test_phase_b_rejects_dynamic_mask_policy():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    bad_phase_b = {
        "rollout": {"K_choices": [2], "K_probs": [1.0]},
        "prefix_render": {"policy": "current_plus_random_previous"},
        "query_observation": {"cameras_per_frame": "all_cams"},
        "episode": {"reset_vsm_on_episode_end": True},
        "masks": {
            "vsm_scope": "bg_rigid",
            "prefix_loss_mask": "valid_non_sky_non_egocar_non_dynamic",
        },
    }
    with pytest.raises(ValueError, match="dynamic mask"):
        _build_scheduler(_make_mock_dataset(sidx), phase_b_cfg=bad_phase_b)


def test_phase_b_rejects_disabled_episode_vsm_reset():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    bad_phase_b = {
        "rollout": {"K_choices": [2], "K_probs": [1.0]},
        "prefix_render": {"policy": "current_plus_random_previous"},
        "query_observation": {"cameras_per_frame": "all_cams"},
        "episode": {"reset_vsm_on_episode_end": False},
        "masks": {"vsm_scope": "bg_rigid"},
    }
    with pytest.raises(ValueError):
        _build_scheduler(_make_mock_dataset(sidx), phase_b_cfg=bad_phase_b)


def test_phase_b_rejects_distinct_k_above_blocks_per_episode():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    bad_phase_b = {
        "rollout": {
            "K_choices": [4],
            "K_probs": [1.0],
            "sample_event_frames": "random_blocks_in_episode",
            "event_order": "chronological",
            "distinct_event_frames": True,
        },
        "prefix_render": {"policy": "current_plus_random_previous"},
        "query_observation": {"cameras_per_frame": "all_cams"},
        "episode": {"reset_vsm_on_episode_end": True},
        "masks": {"vsm_scope": "bg_rigid"},
    }
    with pytest.raises(ValueError, match="K_choices greater than blocks_per_episode"):
        _build_scheduler(_make_mock_dataset(sidx), phase="phase_B_viewset_rollout", blocks_per_episode=3, phase_b_cfg=bad_phase_b)


def test_phase_b_curriculum_controls_rollout_k():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    phase_b_cfg = {
        "rollout": {
            "K_choices": [2],
            "K_probs": [1.0],
            "curriculum": [{"start_step": 0, "K_choices": [3], "K_probs": [1.0]}],
            "sample_event_frames": "random_blocks_in_episode",
            "event_order": "chronological",
            "distinct_event_frames": True,
        },
        "prefix_render": {"policy": "current_plus_random_previous"},
        "query_observation": {"enable": True, "cameras_per_frame": "all_cams"},
        "episode": {"reset_vsm_on_episode_end": True},
        "masks": {"vsm_scope": "bg_rigid"},
    }
    sch = _build_scheduler(
        _make_mock_dataset(sidx),
        phase="phase_B_viewset_rollout",
        blocks_per_episode=3,
        phase_b_cfg=phase_b_cfg,
    )
    batch = sch.next_batch()
    assert int(batch["request_meta"]["inner_K"]) == 3


def test_phase_b_episode_stream_tbptt_emits_contiguous_chunks_and_excludes_prior_query():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=4)
    sch = _build_scheduler(
        _make_mock_dataset(sidx),
        phase="phase_B_viewset_rollout",
        blocks_per_episode=4,
        phase_b_cfg=_strict_tbptt_phase_b_cfg(k=2),
    )
    batch0 = sch.next_batch()
    meta0 = batch0["request_meta"]["tbptt"]
    assert meta0["chunk_idx"] == 0
    assert meta0["is_first_chunk"] is True
    assert meta0["is_last_chunk"] is False
    assert meta0["event_block_indices"] == [0, 1]
    assert meta0["event_frame_indices"] == sorted(meta0["event_frame_indices"])
    assert meta0["prior_written_frames"] == []

    batch1 = sch.next_batch()
    meta1 = batch1["request_meta"]["tbptt"]
    assert meta1["chunk_idx"] == 1
    assert meta1["is_first_chunk"] is False
    assert meta1["is_last_chunk"] is True
    assert meta1["event_block_indices"] == [2, 3]
    assert set(meta1["prior_written_frames"]) == set(meta0["event_frame_indices"])
    query_frames = {int(ref[0]) for ref in batch1["request_meta"]["query_label_refs"]}
    assert query_frames.isdisjoint(set(meta1["prior_written_frames"]))
    assert query_frames.isdisjoint(set(meta1["event_frame_indices"]))


def test_phase_b_episode_stream_tbptt_rejects_random_sampling():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=4)
    bad = _strict_tbptt_phase_b_cfg(k=2)
    bad["rollout"]["sample_event_frames"] = "random_blocks_in_episode"
    with pytest.raises(ValueError, match="sequential_blocks_in_episode"):
        _build_scheduler(
            _make_mock_dataset(sidx),
            phase="phase_B_viewset_rollout",
            blocks_per_episode=4,
            phase_b_cfg=bad,
        )


def test_phase_b_grouped_repeat_tbptt_emits_repeated_steps_and_unique_tbptt_frames():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=6)
    phase_b_cfg = _grouped_repeat_phase_b_cfg(repeats_per_block=2, blocks_per_chunk=4)
    sch = _build_scheduler(
        _make_mock_dataset(sidx),
        phase="phase_B_viewset_rollout",
        blocks_per_episode=6,
        steps_per_block=1,
        block_order="step_major",
        step_major_switch_interval_steps=1,
        phase_b_cfg=phase_b_cfg,
    )
    batch0 = sch.next_batch()
    meta0 = batch0["request_meta"]
    repeat0 = meta0["phase_b_repeat"]
    tbptt0 = meta0["tbptt"]
    assert int(meta0["inner_K"]) == 8
    assert repeat0["step_block_indices"] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert repeat0["step_repeat_indices"] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert repeat0["step_memory_write_flags"] == [True, False, True, False, True, False, True, False]
    assert tbptt0["event_block_indices"] == [0, 1, 2, 3]
    assert tbptt0["event_frame_indices"] == repeat0["unique_event_frame_indices"]
    assert tbptt0["step_event_frame_indices"] == repeat0["step_source_frame_indices"]
    assert len(set(tbptt0["event_frame_indices"])) == 4
    assert len(tbptt0["step_event_frame_indices"]) == 8
    assert meta0["evidence_refs_by_step"][0] == meta0["evidence_refs_by_step"][1]
    assert meta0["evidence_refs_by_step"][2] == meta0["evidence_refs_by_step"][3]
    st_after_chunk0 = sch.current_episode_state
    assert st_after_chunk0 is not None
    assert [int(x) for x in st_after_chunk0["block_update_counts"][:4]] == [1, 1, 1, 1]
    assert [bool(x) for x in st_after_chunk0["block_ended"][:4]] == [True, True, True, True]
    assert int(st_after_chunk0["episode_step_cursor"]) == 4
    assert int(st_after_chunk0["block_cursor"]) == 4

    batch1 = sch.next_batch()
    meta1 = batch1["request_meta"]
    assert meta1["tbptt"]["event_block_indices"] == [4, 5]
    assert set(meta1["tbptt"]["prior_written_frames"]) == set(tbptt0["event_frame_indices"])
    assert int(meta1["inner_K"]) == 4


def test_phase_b_grouped_repeat_tbptt_rejects_nonchronological_cross_chunk_source_frame():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=6)
    phase_b_cfg = _grouped_repeat_phase_b_cfg(repeats_per_block=2, blocks_per_chunk=2)
    sch = _build_scheduler(
        _make_mock_dataset(sidx),
        phase="phase_B_viewset_rollout",
        blocks_per_episode=6,
        steps_per_block=1,
        block_order="step_major",
        step_major_switch_interval_steps=1,
        phase_b_cfg=phase_b_cfg,
    )
    batch0 = sch.next_batch()
    prior_last = max(int(x) for x in batch0["request_meta"]["tbptt"]["event_frame_indices"])
    st = sch.current_episode_state
    assert st is not None
    st["frame_chain"][2] = int(prior_last)
    with pytest.raises(ValueError, match="chronological across chunks"):
        sch.next_batch()


def test_phase_b_grouped_repeat_tbptt_rejects_short_final_chunk_when_disabled():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=6)
    phase_b_cfg = _grouped_repeat_phase_b_cfg(
        repeats_per_block=2,
        blocks_per_chunk=4,
        allow_short_final_chunk=False,
    )
    sch = _build_scheduler(
        _make_mock_dataset(sidx),
        phase="phase_B_viewset_rollout",
        blocks_per_episode=6,
        steps_per_block=1,
        block_order="step_major",
        phase_b_cfg=phase_b_cfg,
    )
    sch.next_batch()
    with pytest.raises(ValueError, match="shorter than blocks_per_chunk"):
        sch.next_batch()


def test_phase_b_episode_block_repeat_tbptt_is_deprecated():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=4)
    bad = _strict_tbptt_phase_b_cfg(k=1)
    bad["rollout"]["mode"] = "episode_block_repeat_tbptt"
    with pytest.raises(ValueError, match="use episode_grouped_repeat_tbptt"):
        _build_scheduler(
            _make_mock_dataset(sidx),
            phase="phase_B_viewset_rollout",
            blocks_per_episode=4,
            phase_b_cfg=bad,
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda cfg: cfg["rollout"].__setitem__("sample_event_frames", "random_blocks_in_episode"), "sequential_blocks_in_episode"),
        (lambda cfg: cfg["rollout"].__setitem__("event_order", "sampled"), "event_order=chronological"),
        (lambda cfg: cfg["rollout"].__setitem__("distinct_event_frames", False), "distinct_event_frames=true"),
        (lambda cfg: cfg["rollout"]["repeat_patterns"][0].__setitem__("repeats_per_block", 0), "must be >= 1"),
        (lambda cfg: cfg["rollout"]["repeat_patterns"][0].__setitem__("blocks_per_chunk", 5), "blocks_per_chunk cannot exceed"),
        (lambda cfg: cfg["rollout"].__setitem__("max_inner_K", 3), "inner_K exceeds max_inner_K"),
    ],
)
def test_phase_b_grouped_repeat_tbptt_rejects_invalid_config(mutator, match):
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=4)
    bad = _grouped_repeat_phase_b_cfg(repeats_per_block=2, blocks_per_chunk=4)
    mutator(bad)
    with pytest.raises(ValueError, match=match):
        _build_scheduler(
            _make_mock_dataset(sidx),
            phase="phase_B_viewset_rollout",
            blocks_per_episode=4,
            steps_per_block=1,
            phase_b_cfg=bad,
        )


def test_phase_b_grouped_repeat_tbptt_rejects_outer_steps_per_block_repeat():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0, num_keyframes=4)
    with pytest.raises(ValueError, match="steps_per_block=1"):
        _build_scheduler(
            _make_mock_dataset(sidx),
            phase="phase_B_viewset_rollout",
            blocks_per_episode=4,
            steps_per_block=2,
            phase_b_cfg=_grouped_repeat_phase_b_cfg(repeats_per_block=2, blocks_per_chunk=4),
        )


def test_v9_requires_native_dataset_assembly():
    sidx = _make_sidx_multi_frame_per_keyframe(scene_id=1, segment_id=0)
    ds = _make_mock_dataset(sidx)
    delattr(ds, "_assemble_segment_batch_from_v9_request")
    sch = _build_scheduler(ds)
    with pytest.raises(ValueError, match="_assemble_segment_batch_from_v9_request"):
        sch.next_batch()


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
