from __future__ import annotations

import numpy as np
import pytest
import torch

from models.streetforward.minimal_trainer_stage4_0 import MinimalStreetForwardStage4_0
from models.streetforward.node_states import NodeStateRigid


def _make_rigid_state(device: torch.device) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.tensor([[1.0, 0.0, 0.0]], device=device),
        scales_log=torch.zeros((1, 3), device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        opacity_logit=torch.zeros((1, 1), device=device),
        sh_dc=torch.zeros((1, 3), device=device),
        sh_rest=torch.zeros((1, 3, 3), device=device),
        point_ids=torch.tensor([[0]], dtype=torch.long, device=device),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device),
        instances_trans=torch.tensor([[[0.5, 0.0, 0.0]]], device=device),
        instances_fv=torch.tensor([[True]], dtype=torch.bool, device=device),
        instance_ids=[0],
        frame_ids=[7],
        cur_frame=7,
    )


def test_assert_src_target_consistent_accepts_aligned_pairs():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    trainer.device = torch.device("cpu")
    eye = torch.eye(4)
    source_view = type("View", (), {"camtoworlds": eye.clone(), "Ks": torch.eye(3).unsqueeze(0)})()
    target_view = type("View", (), {"camtoworlds": eye.clone(), "Ks": torch.eye(3).unsqueeze(0)})()
    source_image = torch.zeros(4, 5, 3)
    target_image = torch.zeros(4, 5, 3)
    batch = {
        "source_views": [source_view],
        "source_images": [source_image],
        "source_frame_idx": 7,
        "targets": [
            {
                "frame_idx": 7,
                "view": target_view,
                "gt_image": target_image,
                "sky_mask": torch.ones(4, 5),
                "viewdirs": torch.zeros(4, 5, 3),
            }
        ],
    }
    trainer._assert_src_target_consistent(batch, batch["targets"])


def test_rigid_local_to_world_transform_applies_instance_translation():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    state = _make_rigid_state(torch.device("cpu"))
    world = trainer._transform_rigid_to_world(state, state.means, frame_idx=7)
    expected = torch.tensor([[1.5, 0.0, 0.0]])
    assert torch.allclose(world, expected, atol=1e-6)


def test_rigid_frame_missing_fast_fail():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    state = _make_rigid_state(torch.device("cpu"))
    try:
        trainer._rigid_point_valid_mask(state, frame_idx=99)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when frame_idx is missing.")


def test_rigid_invalid_instance_filtered():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    state = _make_rigid_state(torch.device("cpu"))
    state.instances_fv[0, 0] = False
    valid = trainer._rigid_point_valid_mask(state, frame_idx=7)
    assert valid.shape[0] == 1
    assert bool(valid[0]) is False


def test_stage4_bg_distant_init_applies_runtime_point_caps():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    trainer.device = torch.device("cpu")
    trainer.bbx_min = torch.tensor([-1.0, -1.0, -1.0])
    trainer.bbx_max = torch.tensor([1.0, 1.0, 1.0])
    trainer.sh_degree = 1
    trainer.near_max_points = 3
    trainer.distant_max_points = 2
    trainer.node_states_bg = {}
    trainer.node_states_distant = {}
    trainer.bg_cfg = {
        "init": {
            "scale_init_mode": "isotropic",
            "isotropic_log_value": -2.3,
            "knn_k": 8,
            "knn_log_scale_bias": -1.5,
            "opacity_init": 0.1,
        }
    }
    trainer.distant_cfg = {
        "init": {
            "scale_init_mode": "isotropic",
            "isotropic_log_value": -1.7,
            "knn_k": 8,
            "knn_log_scale_bias": -1.2,
            "opacity_init": 0.05,
        }
    }

    near = np.zeros((5, 6), dtype=np.float32)
    distant = np.zeros((4, 6), dtype=np.float32)
    distant[:, 0] = 3.0
    batch = {
        "scene_id": 1,
        "segment_id": 0,
        "pointcloud": {"background": np.concatenate([near, distant], axis=0)},
    }

    node_state_bg, node_state_distant = trainer._get_or_init_node_states_bg_distant(batch)

    assert int(node_state_bg.means.shape[0]) == 3
    assert node_state_distant is not None
    assert int(node_state_distant.means.shape[0]) == 2


def test_rigid_row_space_metadata_accepts_matching_layout():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    batch = {
        "knn_struct_neighbors": {
            "rigid_instance_intids": [9, 11],
            "rigid_instance_offsets": [0, 2, 5],
            "rigid_knn_row_ids": [0, 1, 2, 3, 4],
        }
    }
    trainer._validate_rigid_row_space_metadata(
        batch=batch,
        instance_ids=[9, 11],
        point_counts_by_instance={9: 2, 11: 3},
    )


def test_rigid_row_space_metadata_rejects_instance_order_mismatch():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    batch = {
        "knn_struct_neighbors": {
            "rigid_instance_intids": [11, 9],
            "rigid_instance_offsets": [0, 3, 5],
            "rigid_knn_row_ids": [0, 1, 2, 3, 4],
        }
    }
    with pytest.raises(ValueError, match="rigid_instance_intids mismatch"):
        trainer._validate_rigid_row_space_metadata(
            batch=batch,
            instance_ids=[9, 11],
            point_counts_by_instance={9: 2, 11: 3},
        )
