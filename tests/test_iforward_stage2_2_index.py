from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from datasets.iforward_stage2_2.index_builder import build_stage2_2_index_from_dataset
from datasets.iforward_stage2_2.index_format import PROTOCOL_PATTERNS, protocol_offsets
from datasets.iforward_stage2_2.index_loader import load_stage2_2_index


class _Dataset:
    _initialized = True

    def __init__(self, *, frames=range(30), scene_ids=(1,), segment_ids=(0,), num_cams=3):
        self.frames = [int(x) for x in frames]
        self.scene_ids = [int(x) for x in scene_ids]
        self.segment_ids = [int(x) for x in segment_ids]
        self.num_cams = int(num_cams)

    def list_training_scene_ids(self):
        return list(self.scene_ids)

    def list_segment_ids(self, scene_id):
        return list(self.segment_ids)

    def get_segment_index(self, scene_id, segment_id):
        keyframes = list(range(len(self.frames)))
        return SimpleNamespace(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            num_cams=self.num_cams,
            frame_indices=list(self.frames),
            train_frame_set=set(self.frames),
            test_frame_indices=[],
            test_frame_set=set(),
            keyframe_indices=keyframes,
            frame_to_keyframe={int(f): int(i) for i, f in enumerate(self.frames)},
            keyframe_to_frames={int(i): [int(f)] for i, f in enumerate(self.frames)},
            train_image_refs=tuple((int(f), int(c)) for f in self.frames for c in range(self.num_cams)),
            frame_timestamps_us={int(f): int(f) * 100000 for f in self.frames},
        )


def test_stage2_2_index_fingerprint_includes_frame_period():
    ds = _Dataset()
    a = build_stage2_2_index_from_dataset(dataset=ds, cfg={"scheduler_stage2_2": {"time": {"frame_period_us": 100000}}})
    b = build_stage2_2_index_from_dataset(dataset=ds, cfg={"scheduler_stage2_2": {"time": {"frame_period_us": 200000}}})
    assert a.fingerprint != b.fingerprint
    assert a.frames[3]["timestamp_us"] == 300000


def test_stage2_2_index_requires_real_timestamp_unless_explicit_synthetic():
    ds = _Dataset()
    original = _Dataset.get_segment_index

    def without_timestamps(self, scene_id, segment_id):
        out = original(self, scene_id, segment_id)
        delattr(out, "frame_timestamps_us")
        return out

    _Dataset.get_segment_index = without_timestamps
    try:
        with pytest.raises(ValueError, match="requires real per-frame timestamp_us"):
            build_stage2_2_index_from_dataset(dataset=ds, cfg={})
        index = build_stage2_2_index_from_dataset(
            dataset=ds,
            cfg={"scheduler_stage2_2": {"time": {"allow_synthetic_timestamp": True}}},
        )
        assert index.timestamp_source == "frame_idx_times_frame_period_us"
    finally:
        _Dataset.get_segment_index = original


def test_stage2_2_index_protocol_windows_match_bruteforce():
    index = build_stage2_2_index_from_dataset(dataset=_Dataset(frames=range(30)), cfg={})
    for protocol, patterns in PROTOCOL_PATTERNS.items():
        expected = sum(30 - max(offsets) for offsets in patterns)
        got = int(index.windows_for_protocol(protocol).shape[0])
        assert got == expected


def test_stage2_2_index_i123_pattern_and_bootstrap_coverage():
    index = build_stage2_2_index_from_dataset(dataset=_Dataset(frames=range(40)), cfg={})
    win = index.windows_for_protocol("I123")[0]
    frames = index.frames_for_segment_row(int(win["segment_row"]))
    offsets = protocol_offsets("I123", int(win["pattern_id"]))
    observed = [int(frames[int(win["start_local_frame"]) + off]["frame_idx"]) for off in offsets]
    assert observed == list(offsets)
    assert sorted(int(x["local_frame"]) for x in index.bootstrap_frames) == list(range(40))


def test_stage2_2_index_mmap_roundtrip_and_fingerprint_failfast(tmp_path):
    original = build_stage2_2_index_from_dataset(dataset=_Dataset(frames=range(30)), cfg={}, output_dir=tmp_path)
    loaded = load_stage2_2_index(tmp_path, expected_fingerprint=original.fingerprint)
    assert loaded.fingerprint == original.fingerprint
    assert np.asarray(loaded.frames["frame_idx"]).tolist() == np.asarray(original.frames["frame_idx"]).tolist()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_stage2_2_index(tmp_path, expected_fingerprint="bad")
