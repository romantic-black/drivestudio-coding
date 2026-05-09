from __future__ import annotations

from unittest.mock import patch

from datasets.multi_scene_dataset_v4 import SegmentIndexV4
from datasets.validation_scheduler_v8 import (
    build_segment_episode_starts,
    build_validation_episode_specs_v8,
)


class _DummyDataset:
    def __init__(self) -> None:
        self._sidx = SegmentIndexV4(
            scene_id=10,
            segment_id=0,
            num_cams=2,
            frame_indices=[100, 101, 102, 103, 104, 105],
            test_frame_indices=[],
            train_frame_set=frozenset([100, 101, 102, 103, 104, 105]),
            test_frame_set=frozenset(),
            keyframe_indices=[0, 1, 2, 3, 4, 5],
            keyframe_to_frames={0: [100], 1: [101], 2: [102], 3: [103], 4: [104], 5: [105]},
            frame_to_keyframe={100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 105: 5},
            segment_first_frame_idx=100,
            train_image_refs=((100, 0),),
            test_image_refs=tuple(),
        )

    def list_segment_ids(self, scene_id: int):
        assert int(scene_id) == 10
        return [0]

    def get_segment_index(self, scene_id: int, segment_id: int):
        assert int(scene_id) == 10
        assert int(segment_id) == 0
        return self._sidx


def test_build_segment_episode_starts_tail_aligned():
    starts = build_segment_episode_starts(num_keyframes=6, blocks_per_episode=3, window_keyframes=3)
    assert starts == [0, 3]


def test_build_validation_episode_specs_v8_middle_episode_and_visit_targets():
    ds = _DummyDataset()
    specs = build_validation_episode_specs_v8(
        dataset=ds,
        eval_scene_ids=[10],
        blocks_per_episode=3,
        total_target_frames=3,
        steps_per_block=8,
        block_order="step_major",
        step_major_switch_interval_steps=4,
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.episode_start_keyframe_pos == 3
    assert len(spec.frame_chain) == 3
    assert len(spec.block_visit_order) == 24
    assert len(spec.visit_target_windows) == 24
    assert len(spec.eval_image_refs) == 6  # E * num_cams
    assert spec.visit_target_windows[0] == [103]
    assert spec.visit_target_windows[4] == [104, 103]
    assert spec.visit_target_windows[8] == [105, 104, 103]
    assert spec.visit_target_windows[12] == [103, 104, 105]


def test_build_validation_episode_specs_v8_random_history_policy():
    ds = _DummyDataset()
    with patch(
        "datasets.validation_scheduler_v8.random.sample",
        side_effect=lambda population, k: sorted([int(x) for x in population])[: int(k)],
    ):
        specs = build_validation_episode_specs_v8(
            dataset=ds,
            eval_scene_ids=[10],
            blocks_per_episode=3,
            total_target_frames=3,
            steps_per_block=8,
            block_order="step_major",
            step_major_switch_interval_steps=4,
            history_target_policy="random_visited",
        )
    assert specs[0].visit_target_windows[8] == [105, 103, 104]
