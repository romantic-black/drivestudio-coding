from __future__ import annotations

import pytest
import torch

from models.iforward import IForwardBatchResolver


def _iforward_batch(*, bad_cams: bool = False):
    source_refs = [(10, 0), (10, 1), (10, 2), (11, 0), (11, 1), (11, 2)]
    if bad_cams:
        source_refs = [(10, 0), (10, 1), (11, 0), (11, 1)]
    target_refs = [(10, 0), (10, 1), (10, 2), (11, 0), (11, 1), (11, 2), (12, 0), (12, 1), (12, 2)]
    target_roles = ["final_current_recon"] * 6 + ["final_nearby_rollout"] * 3
    steps = [
        {
            "step_idx": 0,
            "source_frame_idx": 10,
            "repeat_idx": 0,
            "rollout_block_rank": 0,
            "evidence_refs": [(10, 0), (10, 1), (10, 2)] if not bad_cams else [(10, 0), (10, 1)],
            "source_indices": [0, 1, 2] if not bad_cams else [0, 1],
            "commit_observation_memory": True,
            "update_optimizer_memory": True,
        },
        {
            "step_idx": 1,
            "source_frame_idx": 11,
            "repeat_idx": 0,
            "rollout_block_rank": 1,
            "evidence_refs": [(11, 0), (11, 1), (11, 2)] if not bad_cams else [(11, 0), (11, 1)],
            "source_indices": [3, 4, 5] if not bad_cams else [2, 3],
            "commit_observation_memory": True,
            "update_optimizer_memory": True,
        },
    ]
    ifwd = {
        "scheduler_version": "iforward_v1",
        "model_family": "IForward",
        "scene_id": 1,
        "segment_id": 2,
        "episode_id": 3,
        "rollout_id_global": 4,
        "rollout_idx_in_episode": 0,
        "inner_K": 2,
        "steps": steps,
        "input_frame_indices": [10, 11],
        "evidence_refs_flat": source_refs,
        "target_refs_flat": target_refs,
        "target_roles_flat": target_roles,
        "reset_scene_state_before_rollout": True,
        "carry_scene_state_after_rollout": True,
        "episode_end_after_rollout": False,
        "detach_graph_after_rollout": True,
        "source_ref_to_index_keyed": {f"{f}:{c}": i for i, (f, c) in enumerate(source_refs)},
    }
    return {
        "scene_id": 1,
        "segment_id": 2,
        "request_meta": {
            "scheduler_version": "iforward_v1",
            "model_family": "IForward",
            "assembly_mode": "image_ref_iforward_v1",
            "source_image_refs": source_refs,
            "target_image_refs": target_refs,
            "target_image_roles": target_roles,
            "iforward": ifwd,
        },
        "_iforward": ifwd,
        "source": {
            "frame_indices": torch.tensor([f for f, _ in source_refs]),
            "cam_indices": torch.tensor([c for _, c in source_refs]),
        },
        "target": {
            "frame_indices": torch.tensor([f for f, _ in target_refs]),
            "cam_indices": torch.tensor([c for _, c in target_refs]),
        },
    }


def test_iforward_resolver_validates_contract_and_step_source_selection():
    resolved = IForwardBatchResolver().resolve(_iforward_batch())
    assert resolved.inner_K == 2
    assert resolved.steps[0].source_indices == (0, 1, 2)
    assert resolved.steps[1].source_indices == (3, 4, 5)
    assert resolved.current_target_indices == (0, 1, 2, 3, 4, 5)
    assert resolved.latest_input_frame_idx == 11
    assert resolved.current_latest_target_indices == (3, 4, 5)
    assert resolved.history_rollout_target_indices == (0, 1, 2)
    assert resolved.nearby_target_indices == (6, 7, 8)
    assert resolved.reset_scene_state_before_rollout is True
    assert resolved.carry_scene_state_after_rollout is True


def test_iforward_resolver_rejects_non_3cam_step():
    with pytest.raises(ValueError, match="cams"):
        IForwardBatchResolver().resolve(_iforward_batch(bad_cams=True))


def test_iforward_resolver_rejects_nearby_evidence_leakage():
    batch = _iforward_batch()
    batch["_iforward"]["target_refs_flat"][-1] = (10, 0)
    batch["_iforward"]["target_roles_flat"][-1] = "final_nearby_rollout"
    batch["request_meta"]["target_image_refs"][-1] = (10, 0)
    batch["target"]["frame_indices"][-1] = 10
    batch["target"]["cam_indices"][-1] = 0
    with pytest.raises(ValueError, match="leaked"):
        IForwardBatchResolver().resolve(batch)
