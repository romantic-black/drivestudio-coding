from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
import torch

from models.iforward.random_window_batch import (
    RANDOM_WINDOW_ASSEMBLY_MODE,
    RANDOM_WINDOW_MODEL_FAMILY,
    RANDOM_WINDOW_SCHEDULER_VERSION,
)
from models.iforward.random_window_resolver import IForwardRandomWindowBatchResolver


ImageRef = Tuple[int, int]


def _refs_for_frame(frame_idx: int, num_cams: int = 3) -> List[ImageRef]:
    return [(int(frame_idx), int(cam_idx)) for cam_idx in range(int(num_cams))]


def _random_window_batch(*, nearby_refs: List[ImageRef] | None = None) -> Dict:
    input_frames = [10, 20, 30, 40]
    source_refs = [ref for frame in input_frames for ref in _refs_for_frame(frame)]
    current_latest_refs = _refs_for_frame(40)
    in_rollout_history_refs = [ref for frame in input_frames[:-1] for ref in _refs_for_frame(frame)]
    short_window_history_refs = _refs_for_frame(0)
    nearby_refs = list(nearby_refs) if nearby_refs is not None else _refs_for_frame(50)
    target_refs = current_latest_refs + in_rollout_history_refs + short_window_history_refs + nearby_refs
    target_roles = (
        ["current_latest"] * len(current_latest_refs)
        + ["in_rollout_history"] * len(in_rollout_history_refs)
        + ["short_window_history"] * len(short_window_history_refs)
        + ["nearby"] * len(nearby_refs)
    )
    steps = []
    for block_pos, frame_idx in enumerate(input_frames):
        refs = _refs_for_frame(frame_idx)
        for repeat_idx in range(2):
            step_idx = len(steps)
            steps.append(
                {
                    "step_idx": int(step_idx),
                    "block_id": int(block_pos + 3),
                    "block_pos_in_window": int(block_pos),
                    "repeat_idx": int(repeat_idx),
                    "global_k": int(step_idx),
                    "source_frame_idx": int(frame_idx),
                    "source_keyframe_idx": int(block_pos),
                    "evidence_refs": list(refs),
                    "commit_observation_memory": bool(repeat_idx == 0),
                    "update_optimizer_memory": True,
                    "is_frame_exit": bool(repeat_idx == 1),
                    "rollout_pos_code": float(step_idx) / 7.0,
                    "frame_pos_code": float(block_pos) / 3.0,
                    "repeat_pos_code": float(repeat_idx),
                }
            )
    ifwd = {
        "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
        "model_family": RANDOM_WINDOW_MODEL_FAMILY,
        "scene_id": 1,
        "segment_id": 2,
        "episode_id": 9,
        "rollout_id_global": 17,
        "rollout_idx_in_episode": 3,
        "rollouts_per_episode": 8,
        "window_start": 3,
        "window_end": 7,
        "window_block_ids": [3, 4, 5, 6],
        "window_hash": 123456789,
        "window_revisit_count": 1,
        "unique_windows_seen": 3,
        "is_repeated_window": True,
        "blocks_per_rollout": 4,
        "repeats_per_block": 2,
        "inner_K": 8,
        "input_frame_indices": list(input_frames),
        "evidence_refs_flat": list(source_refs),
        "target_refs_flat": list(target_refs),
        "target_roles_flat": list(target_roles),
        "current_latest_refs": list(current_latest_refs),
        "in_rollout_history_refs": list(in_rollout_history_refs),
        "short_window_history_refs": list(short_window_history_refs),
        "nearby_refs": list(nearby_refs),
        "steps": steps,
        "reset_scene_state_before_rollout": False,
        "carry_scene_state_after_rollout": True,
        "episode_end_after_rollout": False,
        "detach_graph_after_rollout": True,
    }
    return {
        "scene_id": 1,
        "segment_id": 2,
        "request_meta": {
            "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
            "model_family": RANDOM_WINDOW_MODEL_FAMILY,
            "assembly_mode": RANDOM_WINDOW_ASSEMBLY_MODE,
            "source_image_refs": list(source_refs),
            "target_image_refs": list(target_refs),
            "target_image_roles": list(target_roles),
            "iforward": ifwd,
        },
        "_iforward": ifwd,
        "source": {
            "frame_indices": torch.tensor([int(f) for f, _ in source_refs], dtype=torch.long),
            "cam_indices": torch.tensor([int(c) for _, c in source_refs], dtype=torch.long),
        },
        "target": {
            "frame_indices": torch.tensor([int(f) for f, _ in target_refs], dtype=torch.long),
            "cam_indices": torch.tensor([int(c) for _, c in target_refs], dtype=torch.long),
        },
        "targets": [{"frame_idx": int(f), "cam_idx": int(c), "gt_image": torch.zeros(1, 1, 3)} for f, c in target_refs],
    }


def test_random_window_resolver_uses_explicit_role_refs():
    resolved = IForwardRandomWindowBatchResolver(expected_cams_per_step=3).resolve(_random_window_batch())

    assert resolved.scheduler_version == RANDOM_WINDOW_SCHEDULER_VERSION
    assert resolved.rollouts_per_episode == 8
    assert resolved.window_start == 3
    assert resolved.window_end == 7
    assert resolved.window_block_ids == (3, 4, 5, 6)
    assert resolved.window_hash == 123456789
    assert resolved.window_revisit_count == 1
    assert resolved.unique_windows_seen == 3
    assert resolved.is_repeated_window is True

    assert resolved.current_latest_target_indices == (0, 1, 2)
    assert {resolved.target_refs[idx][0] for idx in resolved.current_latest_target_indices} == {40}
    assert resolved.history_rollout_target_indices == tuple(range(3, 12))
    assert {resolved.target_refs[idx][0] for idx in resolved.history_rollout_target_indices} == {10, 20, 30}
    assert resolved.short_window_history_target_indices == (12, 13, 14)
    assert resolved.nearby_target_indices == (15, 16, 17)
    assert resolved.history_commit_target_indices == tuple(range(3, 12)) + (0, 1, 2)
    assert resolved.current_target_indices == resolved.current_latest_target_indices

    assert len(resolved.steps) == 8
    assert [step.source_frame_idx for step in resolved.steps] == [10, 10, 20, 20, 30, 30, 40, 40]
    assert [step.repeat_idx for step in resolved.steps] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert [step.commit_observation_memory for step in resolved.steps] == [True, False] * 4


def test_random_window_resolver_rejects_nearby_refs_in_input_window():
    batch = _random_window_batch(nearby_refs=_refs_for_frame(10))
    with pytest.raises(ValueError, match="nearby refs leaked into evidence"):
        IForwardRandomWindowBatchResolver(expected_cams_per_step=3).resolve(batch)


def test_random_window_resolver_requires_new_schema_not_old_role_names():
    batch = _random_window_batch()
    batch["_iforward"].pop("current_latest_refs")
    with pytest.raises(ValueError, match="current_latest_refs"):
        IForwardRandomWindowBatchResolver(expected_cams_per_step=3).resolve(batch)
