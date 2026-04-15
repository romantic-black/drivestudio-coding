from __future__ import annotations

from datasets.multi_scene_dataset_v4 import SegmentIndexV4
from datasets.validation_scheduler_v7 import (
    build_segment_episode_starts,
    build_validation_episode_specs_v7,
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
    starts = build_segment_episode_starts(num_keyframes=6, blocks_per_episode=3, window_keyframes=5)
    assert starts == [0, 1]


def test_build_validation_episode_specs_middle_episode_and_view_count():
    ds = _DummyDataset()
    specs = build_validation_episode_specs_v7(
        dataset=ds,
        eval_scene_ids=[10],
        blocks_per_episode=3,
        total_target_frames=3,
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.episode_start_keyframe_pos == 1
    assert len(spec.frame_chain) == 5  # E + 2
    assert len(spec.block_windows) == 3
    assert len(spec.eval_image_refs) == 10  # (E+2) * num_cams

