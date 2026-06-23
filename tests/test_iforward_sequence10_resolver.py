from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import pytest

from datasets.train_scheduler_iforward_sequence10 import TrainSchedulerIForwardSequence10
from models.iforward.sequence10_resolver import IForwardSequence10Resolver


class _Dataset:
    _initialized = True

    def __init__(self):
        self.num_cams = 3
        self.keyframe_to_frames = {k: [k * 10] for k in range(20)}
        self.frames = [k * 10 for k in range(20)]
        self.frame_to_keyframe = {k * 10: k for k in range(20)}

    def list_training_scene_ids(self):
        return [1]

    def list_segment_ids(self, scene_id):
        return [0]

    def get_segment_index(self, scene_id, segment_id):
        return SimpleNamespace(
            scene_id=1,
            segment_id=0,
            num_cams=3,
            frame_indices=list(self.frames),
            train_frame_set=set(self.frames),
            keyframe_indices=list(self.keyframe_to_frames.keys()),
            keyframe_to_frames=dict(self.keyframe_to_frames),
            frame_to_keyframe=dict(self.frame_to_keyframe),
            train_image_refs=tuple((f, c) for f in self.frames for c in range(3)),
        )

    def _assemble_segment_batch_from_iforward_request(self, *, scene_id, segment_id, plan, include_test=False):
        return {
            "request_meta": dict(plan.request_meta),
            "_iforward": dataclasses.asdict(plan),
            "source": {"refs": list(plan.evidence_refs_flat)},
            "target": {"refs": list(plan.target_refs_flat), "roles": list(plan.target_roles_flat)},
        }


def _scheduler(**kwargs):
    return TrainSchedulerIForwardSequence10(
        dataset=_Dataset(),
        traversal_cfg={"fixed_scene_id": 1, "fixed_segment_id": 0, "seed": int(kwargs.get("seed", 1))},
        bootstrap_cfg={"end_step": 5000},
        sequence_cfg={"length": 10, "strides": [1, 2], "max_inner_K": 10},
        causal_cfg={"blocks_per_rollout": 2, "repeats_per_block": 4},
        repair_cfg={"start_step": 15000, "prob": float(kwargs.get("repair_prob", 0.0))},
    )


def _causal_batch():
    sched = _scheduler(repair_prob=0.0)
    sched.global_step = 5000
    return sched.next_batch()


def _repair_batch():
    sched = _scheduler(repair_prob=1.0, seed=8)
    sched.global_step = 15000
    out = None
    for _ in range(6):
        out = sched.next_batch()
    assert out is not None
    return out


def test_sequence10_resolver_accepts_legal_causal_protocol():
    resolved = IForwardSequence10Resolver().resolve(_causal_batch())
    assert resolved.scheduler_version == "iforward_sequence10_v1"
    assert resolved.inner_K == 8
    assert [step.sequence_pos for step in resolved.steps if step.repeat_idx == 0] == [0, 1]
    assert sum(1 for step in resolved.steps if step.temporal_commit) == 2


def test_sequence10_resolver_rejects_causal_missing_commit():
    batch = _causal_batch()
    bad = copy.deepcopy(batch)
    for step in bad["_iforward"]["steps"]:
        if step["is_block_exit"]:
            step["temporal_commit"] = False
            break
    with pytest.raises(ValueError, match="temporal_commit"):
        IForwardSequence10Resolver().resolve(bad)


def test_sequence10_resolver_rejects_repair_temporal_commit():
    batch = _repair_batch()
    bad = copy.deepcopy(batch)
    bad["_iforward"]["steps"][0]["temporal_commit"] = True
    with pytest.raises(ValueError, match="repair must not commit temporal"):
        IForwardSequence10Resolver().resolve(bad)


def test_sequence10_resolver_accepts_legal_repair_protocol():
    resolved = IForwardSequence10Resolver().resolve(_repair_batch())
    positions = [step.sequence_pos for step in resolved.steps if step.repeat_idx == 0]
    assert sorted(positions) == list(range(10))
    assert positions != list(range(10))
    assert all(not step.temporal_commit for step in resolved.steps)
    assert all(not step.commit_observation_memory for step in resolved.steps)
    assert all(not step.update_optimizer_memory for step in resolved.steps)
