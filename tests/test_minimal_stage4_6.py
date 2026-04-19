from __future__ import annotations

import pytest
import torch

from models.streetforward.minimal_trainer_stage4_6 import (
    MinimalStreetForwardStage4_5BaseNoRigidHead,
    MinimalStreetForwardStage4_6,
)
from models.streetforward.node_states import NodeStateRigid


def _base_config_dict() -> dict:
    return {
        "model": {
            "branches": {
                "bg": {
                    "limits": {
                        "offset_max": 0.1,
                        "scale_max": 0.1,
                        "omega_max": 0.1,
                        "opacity_max": 0.1,
                        "sh_dc_max": 0.1,
                        "sh_rest_max": 0.1,
                    }
                },
                "rigid": {
                    "src_backproject_support_min": 1e-2,
                    "init": {
                        "scale_init": {
                            "mode": "isotropic",
                            "isotropic_log_value": -2.3,
                            "knn_k": 3,
                            "knn_log_scale_bias": 0.0,
                        },
                        "opacity_init": 0.1,
                    },
                    "eta": {
                        "means": 1.0,
                        "scales": 1.0,
                        "opacity": 1.0,
                        "sh_dc": 1.0,
                        "sh_rest": 1.0,
                    },
                },
            },
            "rigid_routed": {
                "route_space": "source_frame_world",
                "route_aabb": "segment_aabb",
                "inside_decoder": "bg",
                "outside_decoder": "distant",
                "update_means": True,
                "update_quat": True,
            },
        }
    }


def _dummy_rigid_state() -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        scales_log=torch.zeros(3, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float32),
        opacity_logit=torch.zeros(3, 1),
        sh_dc=torch.zeros(3, 3),
        sh_rest=torch.zeros(3, 3, 3),
        point_ids=torch.zeros(3, 1, dtype=torch.long),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        instances_trans=torch.zeros(1, 1, 3),
        instances_fv=torch.ones(1, 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )


def _dummy_render_chunk(n: int, *, base: float = 0.0) -> dict:
    return {
        "means_r": torch.full((n, 3), base + 1.0),
        "scales_log_r": torch.full((n, 3), base + 2.0),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, dtype=torch.float32),
        "opacity_logit_r": torch.full((n, 1), base + 3.0),
        "sh_dc_r": torch.full((n, 3), base + 4.0),
        "sh_rest_r": torch.full((n, 3, 3), base + 5.0),
        "scales_r": torch.full((n, 3), base + 6.0),
        "opacities_r": torch.full((n,), base + 7.0),
        "colors_r": torch.full((n, 4, 3), base + 8.0),
    }


def test_stage4_6_validate_forbidden_rigid_fields_fast_fail():
    trainer = MinimalStreetForwardStage4_6.__new__(MinimalStreetForwardStage4_6)
    cfg = _base_config_dict()
    cfg["model"]["branches"]["rigid"]["mlp"] = {"hidden_dim": 64}
    with pytest.raises(ValueError, match="removes rigid.mlp"):
        trainer._validate_stage4_6_config(cfg)


def test_stage4_6_base_compat_config_injects_rigid_compat_fields():
    trainer = MinimalStreetForwardStage4_5BaseNoRigidHead.__new__(MinimalStreetForwardStage4_5BaseNoRigidHead)
    cfg = _base_config_dict()
    compat = trainer._make_stage4_6_compat_config(cfg)
    rigid = compat["model"]["branches"]["rigid"]
    assert "mlp" in rigid
    assert "limits" in rigid
    assert "freeze_means" in rigid


def test_stage4_6_route_rigid_source_points_splits_by_aabb():
    trainer = MinimalStreetForwardStage4_6.__new__(MinimalStreetForwardStage4_6)
    trainer.device = torch.device("cpu")
    trainer.bbx_min = torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float32)
    trainer.bbx_max = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    node = _dummy_rigid_state()
    S = torch.tensor([0, 1, 2], dtype=torch.long)
    route = trainer._route_rigid_source_points(node, source_frame_idx=0, S=S)
    assert route.S_in.tolist() == [0, 2]
    assert route.S_out.tolist() == [1]
    assert route.route_inside_global.tolist() == [True, False, True]


def test_stage4_6_subset_writeback_uses_u_index_alignment():
    trainer = MinimalStreetForwardStage4_6.__new__(MinimalStreetForwardStage4_6)
    node = _dummy_rigid_state()
    U = torch.tensor([2, 0], dtype=torch.long)
    render_params = {
        "means_r": torch.tensor([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]),
        "scales_log_r": torch.ones(2, 3),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "opacity_logit_r": torch.ones(2, 1),
        "sh_dc_r": torch.ones(2, 3),
        "sh_rest_r": torch.ones(2, 3, 3),
    }
    trainer._update_node_state_rigid_local_subset(node, render_params, U)
    assert torch.allclose(node.means[2], torch.tensor([5.0, 5.0, 5.0]))
    assert torch.allclose(node.means[0], torch.tensor([6.0, 6.0, 6.0]))


def test_stage4_6_pack_only_u_in_shape_ok():
    render_in = _dummy_render_chunk(2, base=10.0)
    packed = MinimalStreetForwardStage4_6._pack_rigid_local_subsets(render_in=render_in, render_out=None)
    assert packed["means_r"].shape == (2, 3)
    assert packed["sh_rest_r"].shape == (2, 3, 3)
    assert torch.allclose(packed["means_r"], render_in["means_r"])


def test_stage4_6_pack_only_u_out_shape_ok():
    render_out = _dummy_render_chunk(3, base=20.0)
    packed = MinimalStreetForwardStage4_6._pack_rigid_local_subsets(render_in=None, render_out=render_out)
    assert packed["means_r"].shape == (3, 3)
    assert packed["colors_r"].shape == (3, 4, 3)
    assert torch.allclose(packed["means_r"], render_out["means_r"])


def test_stage4_6_writeback_does_not_call_parent_rigid_writeback(monkeypatch: pytest.MonkeyPatch):
    trainer = MinimalStreetForwardStage4_6.__new__(MinimalStreetForwardStage4_6)
    node = _dummy_rigid_state()

    def _parent_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("parent writeback should not be called")

    monkeypatch.setattr(
        MinimalStreetForwardStage4_5BaseNoRigidHead,
        "_writeback_node_states_from_out",
        _parent_should_not_be_called,
    )

    U = torch.tensor([1], dtype=torch.long)
    render_params = {
        "means_r": torch.tensor([[7.0, 8.0, 9.0]]),
        "scales_log_r": torch.ones(1, 3),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "opacity_logit_r": torch.ones(1, 1),
        "sh_dc_r": torch.ones(1, 3),
        "sh_rest_r": torch.ones(1, 3, 3),
    }
    trainer._writeback_node_states_from_out(
        {
            "_node_state_rigid": node,
            "_render_params_rigid_local": render_params,
            "_rigid_writeback_idx": U,
        }
    )
    assert torch.allclose(node.means[1], torch.tensor([7.0, 8.0, 9.0]))


def test_stage4_6_source_mask_prefers_plural_keys():
    trainer = MinimalStreetForwardStage4_6.__new__(MinimalStreetForwardStage4_6)
    trainer._warned_source_mask_legacy_keys = False
    plural_sky = [torch.zeros(2, 2)]
    plural_ego = [torch.ones(2, 2)]
    singular_sky = [torch.full((2, 2), 2.0)]
    singular_ego = [torch.full((2, 2), 3.0)]
    sky, ego = trainer._get_source_masks_from_batch(
        {
            "source_sky_masks": plural_sky,
            "source_egocar_masks": plural_ego,
            "source_sky_mask": singular_sky,
            "source_egocar_mask": singular_ego,
        }
    )
    assert sky is plural_sky
    assert ego is plural_ego


def test_stage4_6_source_mask_falls_back_to_legacy_keys():
    trainer = MinimalStreetForwardStage4_6.__new__(MinimalStreetForwardStage4_6)
    trainer._warned_source_mask_legacy_keys = False
    legacy_sky = [torch.zeros(2, 2)]
    legacy_ego = [torch.ones(2, 2)]
    sky, ego = trainer._get_source_masks_from_batch(
        {
            "source_sky_mask": legacy_sky,
            "source_egocar_mask": legacy_ego,
        }
    )
    assert sky is legacy_sky
    assert ego is legacy_ego
    assert trainer._warned_source_mask_legacy_keys is True
