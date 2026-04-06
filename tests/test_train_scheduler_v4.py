from __future__ import annotations

from types import MethodType
from typing import Any, Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest

import datasets.multi_scene_dataset_v3 as msv3
from datasets.multi_scene_dataset_v3 import (
    MultiSceneDatasetV3,
    SegmentIndex,
    TrainSchedulerV4,
    _build_segment_index_dict,
)


def _make_sidx() -> SegmentIndex:
    scene_data = {
        "dataset": MagicMock(num_cams=2),
        "keyframe_segments": [[10, 11], [12, 13]],
        "segments": [
            {
                "frame_indices": [10, 11, 12, 13],
                "test_frame_indices": [],
                "keyframe_indices": [0, 1],
            }
        ],
    }
    return _build_segment_index_dict(1, 0, scene_data)


def _make_sidx_three_kf() -> SegmentIndex:
    scene_data = {
        "dataset": MagicMock(num_cams=1),
        "keyframe_segments": [[10], [11], [12]],
        "segments": [
            {
                "frame_indices": [10, 11, 12],
                "test_frame_indices": [],
                "keyframe_indices": [0, 1, 2],
            }
        ],
    }
    return _build_segment_index_dict(1, 0, scene_data)


def _make_sidx_four_kf() -> SegmentIndex:
    scene_data = {
        "dataset": MagicMock(num_cams=2),
        "keyframe_segments": [[10], [11], [12], [13]],
        "segments": [
            {
                "frame_indices": [10, 11, 12, 13],
                "test_frame_indices": [],
                "keyframe_indices": [0, 1, 2, 3],
            }
        ],
    }
    return _build_segment_index_dict(1, 0, scene_data)


def _make_sidx_single_keyframe() -> SegmentIndex:
    scene_data = {
        "dataset": MagicMock(num_cams=2),
        "keyframe_segments": [[10, 11]],
        "segments": [
            {
                "frame_indices": [10, 11],
                "test_frame_indices": [],
                "keyframe_indices": [0],
            }
        ],
    }
    return _build_segment_index_dict(1, 0, scene_data)


def _make_mock_dataset(sidx: SegmentIndex) -> MagicMock:
    ds = MagicMock(spec=MultiSceneDatasetV3)
    ds._initialized = True
    ds.initialize = MagicMock()
    ds.get_segment_index = MagicMock(return_value=sidx)
    ds.get_segment_batch_from_image_refs = MagicMock(return_value={"ok": True})
    ds.get_segment_batch_from_frames = MagicMock(side_effect=AssertionError("legacy API must not be called"))
    ds.build_preload_hint = MagicMock(return_value={"hint_version": 1})
    ds.get_scene = MagicMock(
        return_value={
            "segments": [{"keyframe_indices": [0, 1]}],
        }
    )
    return ds


def _sch_kwargs_common():
    return dict(
        keyframe_window_policy="random_contiguous_window",
        pair_order_policy="shuffle_without_replacement",
        total_target_images=2,
        include_source=True,
        extra_target_policy="same_cam_different_keyframe",
        prefer_nearby_keyframes=True,
        overlap_mode="none",
        emit_preload_hints=False,
        execute_preload_hints=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )


def _sch_fallback_defaults():
    return dict(
        fallback_expand_to_segment=True,
        fallback_with_replacement=True,
        include_test=False,
    )


def test_v4_next_batch_uses_image_refs_only():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        **_sch_kwargs_common(),
        **_sch_fallback_defaults(),
    )
    sch.next_batch()
    ds.get_segment_batch_from_image_refs.assert_called_once()
    ds.get_segment_batch_from_frames.assert_not_called()


def test_v4_segment_budget_u_derived_from_episode_block():
    """segment_budget_u = episodes_per_segment * w_eff * num_cams * updates_per_block."""
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=4,
        **_sch_kwargs_common(),
        **_sch_fallback_defaults(),
    )
    sch.next_batch()
    evs = sch.pop_events()
    seg = [e for e in evs if e.get("type") == "segment_begin"][0]
    assert seg["w_eff"] == 2
    assert seg["b_seg"] == 4 * 2 * 2
    assert seg["segment_budget_u"] == (4 * 2 * 2) * 2


def test_v4_reset_event_and_get_current_info_raw_steps():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        **_sch_kwargs_common(),
        **_sch_fallback_defaults(),
    )
    sch.next_batch()
    info = sch.get_current_info()
    # U=1; segment_budget_u = 2 * 2 * 2 * 2 = 16
    assert info["segment_step_budget"] == 16
    assert info["segment_local_step"] == 1
    evs = sch.pop_events()
    types = [e["type"] for e in evs]
    assert "segment_begin" in types
    assert "reset_event" in types
    assert any(e.get("type") == "reset_event" and e.get("reason") == "episode_begin" for e in evs)
    assert "block_begin" in types


def test_v4_pop_events_clears():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        **_sch_kwargs_common(),
        **_sch_fallback_defaults(),
    )
    sch.next_batch()
    assert len(sch.pop_events()) > 0
    assert sch.pop_events() == []


def test_v4_rejects_single_keyframe_when_multi_target():
    sidx = _make_sidx_single_keyframe()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=1,
        episodes_per_segment=2,
        **_sch_kwargs_common(),
        **_sch_fallback_defaults(),
    )
    with pytest.raises(ValueError, match="at least 2 keyframes"):
        sch.next_batch()


def test_v4_execute_preload_hints_calls_dataset_submit():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    ds.submit_preload_hint = MagicMock()
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        **{**_sch_kwargs_common(), **_sch_fallback_defaults(), "execute_preload_hints": True},
    )
    sch.next_batch()
    assert ds.submit_preload_hint.called
    ds.set_preload_active_scope.assert_called()


def test_create_train_scheduler_v4_factory():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = MultiSceneDatasetV3.create_train_scheduler_v4(
        ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        **_sch_kwargs_common(),
        **_sch_fallback_defaults(),
    )
    assert isinstance(sch, TrainSchedulerV4)


def test_v4_window_infeasible_without_expand_or_replacement():
    """Segment may have enough keyframes globally, but a 1-kf window cannot supply extra targets."""
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=1,
        episodes_per_segment=2,
        **_sch_kwargs_common(),
        fallback_expand_to_segment=False,
        fallback_with_replacement=False,
        include_test=False,
    )
    with pytest.raises(ValueError, match="window_extra_cap"):
        sch.next_batch()


def test_v4_include_test_pins_batch_test_image_refs():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    ds.resolve_test_image_refs_deterministic = MagicMock(return_value=[(99, 0), (99, 1)])
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        **_sch_kwargs_common(),
        fallback_expand_to_segment=True,
        fallback_with_replacement=True,
        include_test=True,
    )
    sch.next_batch()
    req = ds.get_segment_batch_from_image_refs.call_args[0][0]
    assert req.test_image_refs == [(99, 0), (99, 1)]
    ds.resolve_test_image_refs_deterministic.assert_called_once_with(1, 0)


def test_v4_pointcloud_topk_picks_highest_scored_keyframe():
    """With overlap_min=0.5 only the highest mocked score qualifies for the random pool; one pick => that keyframe."""
    sidx = _make_sidx_three_kf()
    ds = _make_mock_dataset(sidx)
    ds.pointcloud_generator = MagicMock()

    def gcs(
        scene_id: int,
        seg_id: int,
        src: tuple,
        tgt: tuple,
        mode: str = "none",
        *,
        point_sample_size: Optional[int] = None,
        counts_out: Optional[dict] = None,
        **kwargs: Any,
    ):
        assert mode == "pointcloud_topk"
        assert point_sample_size == 1024
        if tgt[0] == 11:
            return 0.2
        if tgt[0] == 12:
            return 0.8
        return 0.0

    ds.get_or_compute_pair_score = MagicMock(side_effect=gcs)
    ds.is_pair_score_cached = MagicMock(return_value=False)

    kw = _sch_kwargs_common()
    kw["overlap_mode"] = "pointcloud_topk"
    kw["total_target_images"] = 2
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=1,
        keyframes_per_episode=3,
        episodes_per_segment=1,
        overlap_point_sample_size=1024,
        overlap_candidate_frame_policy="middle",
        overlap_score_type="nab_over_na",
        overlap_min=0.5,
        **kw,
        **_sch_fallback_defaults(),
    )
    with mock.patch.object(msv3.random, "randint", return_value=0), mock.patch.object(
        msv3.random, "shuffle", lambda seq: None
    ), mock.patch.object(msv3.random, "choice", side_effect=lambda seq: seq[0]):
        sch.next_batch()

    req = ds.get_segment_batch_from_image_refs.call_args[0][0]
    assert req.target_image_refs[0] == (10, 0)
    assert req.target_image_refs[1] == (12, 0)

    evs = sch.pop_events()
    assert any(e.get("type") == "overlap_select" for e in evs)
    os_ev = [e for e in evs if e.get("type") == "overlap_select"][0]
    assert os_ev["overlap_mode"] == "pointcloud_topk"
    assert os_ev["cache_hits"] + os_ev["cache_misses"] >= 2
    bb = [e for e in evs if e.get("type") == "block_begin"][0]
    assert bb["overlap_mode"] == "pointcloud_topk"
    assert bb["selected_target_scores"] == [0.8]


def test_v4_episode_preload_hint_includes_future_overlap_pairs_pointcloud_topk():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    ds.pointcloud_generator = MagicMock()
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    ds.build_preload_hint = MethodType(MultiSceneDatasetV3.build_preload_hint, v3)
    ds.get_or_compute_pair_score = MagicMock(return_value=0.5)
    ds.is_pair_score_cached = MagicMock(return_value=False)
    sk = dict(_sch_kwargs_common())
    sk["overlap_mode"] = "pointcloud_topk"
    sk["emit_preload_hints"] = True
    sk["execute_preload_hints"] = False
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        overlap_point_sample_size=1024,
        overlap_candidate_frame_policy="middle",
        overlap_score_type="nab_over_na",
        overlap_min=0.0,
        **sk,
        **_sch_fallback_defaults(),
    )
    sch.next_batch()
    evs = sch.pop_events()
    ph = [
        e
        for e in evs
        if e.get("type") == "preload_hint" and e.get("hint_scope") == "episode_source_superset"
    ]
    assert len(ph) == 1
    hint = ph[0]["hint"]
    assert hint["hint_version"] == 2
    assert "future_overlap_pairs" in hint
    assert hint["overlap_meta"]["mode"] == "pointcloud_topk"
    assert hint["overlap_meta"]["point_sample_size"] == 1024
    assert len(hint["future_overlap_pairs"]) >= 1


def test_v4_temporal_ring_requires_max_ring():
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    ds.pointcloud_generator = MagicMock()
    kw = dict(_sch_kwargs_common())
    del kw["overlap_mode"]
    with pytest.raises(ValueError, match="temporal_neighbor_max_ring"):
        TrainSchedulerV4(
            dataset=ds,
            state_write_interval_steps=1,
            updates_per_block=1,
            keyframes_per_episode=2,
            episodes_per_segment=1,
            temporal_neighbor_pool="ring",
            temporal_neighbor_max_ring=None,
            overlap_mode="pointcloud_topk",
            overlap_point_sample_size=1024,
            overlap_candidate_frame_policy="middle",
            overlap_score_type="nab_over_na",
            overlap_min=0.0,
            **kw,
            **_sch_fallback_defaults(),
        )


def test_v4_temporal_ring_pointcloud_topk_scores_ring_subset_first():
    """Source at last keyframe: ring R=1 intersects pool in one keyframe only => one pair score call."""
    sidx = _make_sidx_four_kf()
    ds = _make_mock_dataset(sidx)
    ds.pointcloud_generator = MagicMock()
    n_calls = {"n": 0}

    def gcs(
        scene_id: int,
        seg_id: int,
        src: tuple,
        tgt: tuple,
        mode: str = "none",
        **kwargs: Any,
    ):
        n_calls["n"] += 1
        return 0.9

    ds.get_or_compute_pair_score = MagicMock(side_effect=gcs)
    ds.is_pair_score_cached = MagicMock(return_value=False)

    def pair_shuffle_shim(seq):
        seq.sort(key=lambda t: (0 if t == (3, 0) else 1, t[0], t[1]))

    kw = dict(_sch_kwargs_common())
    del kw["overlap_mode"]
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=4,
        keyframes_per_episode=4,
        episodes_per_segment=1,
        temporal_neighbor_pool="ring",
        temporal_neighbor_max_ring=4,
        temporal_neighbor_cams=None,
        overlap_mode="pointcloud_topk",
        overlap_point_sample_size=1024,
        overlap_candidate_frame_policy="middle",
        overlap_score_type="nab_over_na",
        overlap_min=0.5,
        **kw,
        **_sch_fallback_defaults(),
    )
    with mock.patch.object(msv3.random, "shuffle", pair_shuffle_shim), mock.patch.object(
        msv3.random, "choice", side_effect=lambda seq: seq[0]
    ):
        sch.next_batch()

    assert n_calls["n"] == 1
    evs = sch.pop_events()
    bb = [e for e in evs if e.get("type") == "block_begin"][0]
    assert bb["temporal_neighbor_pool"] == "ring"
    assert bb["temporal_neighbor_ring_effective"] == 1
    assert bb["temporal_neighbor_fallback_full_pool"] is False


def test_v4_temporal_ring_cam_gate_skips_ring():
    """temporal_neighbor_cams=[0]: cam 1 uses full-pool overlap scoring (more pair evaluations)."""
    sidx = _make_sidx_four_kf()
    ds = _make_mock_dataset(sidx)
    ds.pointcloud_generator = MagicMock()
    n_calls = {"n": 0}

    def gcs(*args: Any, **kwargs: Any):
        n_calls["n"] += 1
        return 0.9

    ds.get_or_compute_pair_score = MagicMock(side_effect=gcs)
    ds.is_pair_score_cached = MagicMock(return_value=False)

    def pair_shuffle_shim(seq):
        seq.sort(key=lambda t: (0 if t == (3, 1) else 1, t[0], t[1]))

    kw = dict(_sch_kwargs_common())
    del kw["overlap_mode"]
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=1,
        updates_per_block=4,
        keyframes_per_episode=4,
        episodes_per_segment=1,
        temporal_neighbor_pool="ring",
        temporal_neighbor_max_ring=4,
        temporal_neighbor_cams=[0],
        overlap_mode="pointcloud_topk",
        overlap_point_sample_size=1024,
        overlap_candidate_frame_policy="middle",
        overlap_score_type="nab_over_na",
        overlap_min=0.5,
        **kw,
        **_sch_fallback_defaults(),
    )
    with mock.patch.object(msv3.random, "shuffle", pair_shuffle_shim), mock.patch.object(
        msv3.random, "choice", side_effect=lambda seq: seq[0]
    ):
        sch.next_batch()

    assert n_calls["n"] == 3
    bb = [e for e in sch.pop_events() if e.get("type") == "block_begin"][0]
    assert bb["temporal_neighbor_ring_effective"] is None
    assert bb["temporal_neighbor_fallback_full_pool"] is False


def test_v4_all_blocks_full_when_budget_is_multiple_of_updates_per_block():
    """Derived segment_budget_u is always divisible by updates_per_block — no tail K_u_effective < nominal."""
    sidx = _make_sidx()
    ds = _make_mock_dataset(sidx)
    sch = TrainSchedulerV4(
        dataset=ds,
        state_write_interval_steps=4,
        updates_per_block=2,
        keyframes_per_episode=2,
        episodes_per_segment=2,
        **_sch_kwargs_common(),
        **_sch_fallback_defaults(),
    )
    begins = []
    for _ in range(64):
        sch.next_batch()
        for e in sch.pop_events():
            if e.get("type") == "block_begin":
                begins.append(e)
                assert e.get("K_u_effective") == e.get("K_u_nominal")

