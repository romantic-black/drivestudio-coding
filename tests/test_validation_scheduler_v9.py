from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from datasets.multi_scene_dataset_v4 import SegmentIndexV4
from datasets.validation_scheduler_v9 import (
    build_validation_plan_v9,
    choose_blocks,
    make_phase_a_eval_rollout_batch,
    materialize_validation_v9_batch,
)


class _DummyDataset:
    def __init__(self) -> None:
        self._segments = {
            0: self._make_sidx(0),
            1: self._make_sidx(1),
            2: self._make_sidx(2),
        }
        self._assemble_segment_batch_from_v9_request = MagicMock(
            side_effect=lambda **kwargs: {"request_meta": dict(kwargs["v9_plan"].request_meta)}
        )

    @staticmethod
    def _make_sidx(segment_id: int) -> SegmentIndexV4:
        keyframes = list(range(6))
        keyframe_to_frames = {
            int(k): [int(1000 + segment_id * 100 + k * 10 + i) for i in range(3)]
            for k in keyframes
        }
        frames = []
        frame_to_keyframe = {}
        for k, vals in keyframe_to_frames.items():
            for f in vals:
                frames.append(int(f))
                frame_to_keyframe[int(f)] = int(k)
        return SegmentIndexV4(
            scene_id=10,
            segment_id=int(segment_id),
            num_cams=2,
            frame_indices=frames,
            test_frame_indices=[],
            train_frame_set=frozenset(frames),
            test_frame_set=frozenset(),
            keyframe_indices=keyframes,
            keyframe_to_frames=keyframe_to_frames,
            frame_to_keyframe=frame_to_keyframe,
            segment_first_frame_idx=frames[0],
            train_image_refs=tuple((int(f), 0) for f in frames),
            test_image_refs=tuple(),
        )

    def list_segment_ids(self, scene_id: int):
        assert int(scene_id) == 10
        return [0, 1, 2]

    def get_segment_index(self, scene_id: int, segment_id: int):
        assert int(scene_id) == 10
        return self._segments[int(segment_id)]


def _cfg(**overrides):
    out = {
        "phase": "phase_A_block_local_unroll",
        "fail_fast": True,
        "selection": {
            "seed": 20260524,
            "segments_per_scene": 1,
            "segment_policy": "random_seeded",
            "episode_policy": "random_seeded",
            "blocks_per_segment": 2,
            "block_policy": "random_without_replacement",
            "source_frame_policy": "middle_in_keyframe",
        },
        "phase_A": {
            "k_values": [0, 2, 4],
            "max_K": 4,
            "nearby": {
                "enable": True,
                "frames_per_block": 1,
                "policy": "adjacent_then_random_same_keyframe",
                "same_keyframe_only": True,
                "camera_policy": "all_cams",
            },
        },
    }
    out.update(overrides)
    return out


def test_validation_v9_requires_eval_scene_ids():
    with pytest.raises(ValueError, match="eval_scene_ids"):
        build_validation_plan_v9(dataset=_DummyDataset(), eval_scene_ids=[], cfg=_cfg(), blocks_per_episode=3)


def test_validation_v9_selects_one_segment_per_scene():
    plan = build_validation_plan_v9(dataset=_DummyDataset(), eval_scene_ids=[10], cfg=_cfg(), blocks_per_episode=3)
    segments = {int(s.segment_id) for s in plan.block_specs}
    assert len(segments) == 1
    assert len(plan.block_specs) == 2


def test_validation_v9_random_blocks_without_replacement():
    blocks = choose_blocks(
        blocks_per_episode=8,
        n=4,
        seed=123,
        scene_id=10,
        segment_id=2,
        policy="random_without_replacement",
    )
    assert len(blocks) == 4
    assert len(set(blocks)) == 4
    assert blocks == choose_blocks(
        blocks_per_episode=8,
        n=4,
        seed=123,
        scene_id=10,
        segment_id=2,
        policy="random_without_replacement",
    )


def test_validation_v9_k_values_include_zero():
    cfg = _cfg()
    cfg["phase_A"] = dict(cfg["phase_A"])
    cfg["phase_A"]["k_values"] = [2, 4]
    with pytest.raises(ValueError, match="include 0"):
        build_validation_plan_v9(dataset=_DummyDataset(), eval_scene_ids=[10], cfg=cfg, blocks_per_episode=3)


def test_validation_v9_phase_a_no_prefix_no_query():
    plan = build_validation_plan_v9(dataset=_DummyDataset(), eval_scene_ids=[10], cfg=_cfg(), blocks_per_episode=3)
    rollout = make_phase_a_eval_rollout_batch(plan.block_specs[0], max_K=plan.max_K, k_values=plan.k_values)
    assert rollout.prefix_loss_refs_by_step == [[], [], [], []]
    assert rollout.query_label_refs == []
    assert rollout.aux_loss_refs == []


def test_validation_v9_nearby_not_in_evidence():
    plan = build_validation_plan_v9(dataset=_DummyDataset(), eval_scene_ids=[10], cfg=_cfg(), blocks_per_episode=3)
    spec = plan.block_specs[0]
    assert spec.nearby_loss_refs
    assert set(spec.nearby_loss_refs).isdisjoint(set(spec.evidence_refs))


def test_validation_v9_batch_assembly_uses_image_ref_v9():
    ds = _DummyDataset()
    plan = build_validation_plan_v9(dataset=ds, eval_scene_ids=[10], cfg=_cfg(), blocks_per_episode=3)
    rollout = make_phase_a_eval_rollout_batch(plan.block_specs[0], max_K=plan.max_K, k_values=plan.k_values)
    batch = materialize_validation_v9_batch(ds, rollout)
    assert batch["request_meta"]["assembly_mode"] == "image_ref_v9"
    assert batch["request_meta"]["validation_mode"] == "phase_a_k_sweep"
    assert batch["request_meta"]["target_image_roles"]
    assert ds._assemble_segment_batch_from_v9_request.call_count == 1
