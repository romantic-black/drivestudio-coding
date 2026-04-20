from __future__ import annotations

from types import MethodType

import pytest
import torch

from models.streetforward.minimal_trainer_stage4_6 import RigidRoute
from models.streetforward.minimal_trainer_stage4_6 import MinimalStreetForwardStage4_6
from models.streetforward.minimal_trainer_stage5_0 import MinimalStreetForwardStage5_0
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.offsets_mixin import _quat_to_rot6d as _stage4_quat_to_rot6d
from models.streetforward.struct_decoders.common import _quat_to_rot6d as _stage5_quat_to_rot6d
from models.streetforward.struct_decoders.token_builders import StructTokenBuilder
from models.streetforward.struct_decoders import StructDecoderInput, StreetForwardXCPEDecoder

try:
    import spconv.pytorch  # noqa: F401

    _HAS_SPCONV = True
except ImportError:
    _HAS_SPCONV = False


def _base_stage5_cfg() -> dict:
    return {
        "model": {
            "stage": "5_0",
            "branches": {
                "rigid": {},
            },
            "rigid_routed": {
                "route_space": "source_frame_world",
                "route_aabb": "segment_aabb",
                "inside_decoder": "bg",
                "outside_decoder": "distant",
                "update_means": True,
                "update_quat": True,
            },
            "struct_decoder": {
                "enable": True,
                "type": "xcpe",
                "scope": "bg_rigid_in",
                "output_role": "gru_input",
                "point_preserving": True,
                "include_distant": False,
                "include_rigid_out": False,
                "sparse_backend": "spconv",
                "clamp_grid_coord": False,
                "future": {
                    "allow_pooling": False,
                    "allow_serialized_attention": False,
                },
                "token": {
                    "use_anchor_rgb": False,
                    "use_hidden_state": False,
                },
            },
        }
    }


def _dummy_rigid_state(num_points: int = 10) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.arange(num_points * 3, dtype=torch.float32).view(num_points, 3) * 0.1,
        scales_log=torch.zeros(num_points, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_points, dtype=torch.float32),
        opacity_logit=torch.zeros(num_points, 1),
        sh_dc=torch.zeros(num_points, 3),
        sh_rest=torch.zeros(num_points, 3, 3),
        point_ids=torch.zeros(num_points, 1, dtype=torch.long),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        instances_trans=torch.zeros(1, 1, 3),
        instances_fv=torch.ones(1, 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )


def _dummy_bg_state(num_points: int = 5) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.arange(num_points * 3, dtype=torch.float32).view(num_points, 3) * 0.2,
        scales_log=torch.zeros(num_points, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_points, dtype=torch.float32),
        opacity_logit=torch.zeros(num_points, 1),
        sh_dc=torch.zeros(num_points, 3),
        sh_rest=torch.zeros(num_points, 3, 3),
    )


def _params_from_state(state) -> dict:
    return {
        "means": state.means,
        "scales_log": state.scales_log,
        "quats": state.quats,
        "opacity_logit": state.opacity_logit,
        "sh_dc": state.sh_dc,
        "sh_rest": state.sh_rest,
    }


def test_stage5_0_config_fast_fail_type_must_be_xcpe():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    cfg = _base_stage5_cfg()
    cfg["model"]["struct_decoder"]["type"] = "other"
    with pytest.raises(ValueError, match="type must be 'xcpe'"):
        trainer._validate_stage5_0_config(cfg)


def test_stage5_0_config_fast_fail_include_distant_false():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    cfg = _base_stage5_cfg()
    cfg["model"]["struct_decoder"]["include_distant"] = True
    with pytest.raises(ValueError, match="include_distant must be false"):
        trainer._validate_stage5_0_config(cfg)


def test_stage5_0_config_fast_fail_include_rigid_out_false():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    cfg = _base_stage5_cfg()
    cfg["model"]["struct_decoder"]["include_rigid_out"] = True
    with pytest.raises(ValueError, match="include_rigid_out must be false"):
        trainer._validate_stage5_0_config(cfg)


def test_stage5_0_config_fast_fail_point_preserving_true():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    cfg = _base_stage5_cfg()
    cfg["model"]["struct_decoder"]["point_preserving"] = False
    with pytest.raises(ValueError, match="point_preserving must be true"):
        trainer._validate_stage5_0_config(cfg)


def test_stage5_0_config_fast_fail_allow_pooling_false():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    cfg = _base_stage5_cfg()
    cfg["model"]["struct_decoder"]["future"]["allow_pooling"] = True
    with pytest.raises(ValueError, match="allow_pooling must be false"):
        trainer._validate_stage5_0_config(cfg)


def test_stage5_0_config_fast_fail_use_anchor_rgb_true():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    cfg = _base_stage5_cfg()
    cfg["model"]["struct_decoder"]["token"]["use_anchor_rgb"] = True
    with pytest.raises(ValueError, match="use_anchor_rgb=true"):
        trainer._validate_stage5_0_config(cfg)


def test_stage5_0_config_fast_fail_clamp_grid_coord_true():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    cfg = _base_stage5_cfg()
    cfg["model"]["struct_decoder"]["clamp_grid_coord"] = True
    cfg["model"]["struct_decoder"]["sparse_backend"] = "spconv"
    with pytest.raises(ValueError, match="clamp_grid_coord=true"):
        trainer._validate_stage5_0_config(cfg)


def test_stage4_6_default_bg_rigid_in_hook_preserves_s_in_order():
    trainer = MinimalStreetForwardStage4_6.__new__(MinimalStreetForwardStage4_6)
    trainer.device = torch.device("cpu")
    trainer.bg_src_backproject_support_min = 0.05

    def _build_3d(self, node_state_bg, node_state_rigid, route):
        _ = node_state_rigid
        feat_bg = node_state_bg.means
        feat_rigid_in = route.means_world_S[route.inside_mask_S] if route.S_in.numel() > 0 else None
        return feat_bg, feat_rigid_in

    def _fuse(self, feat_3d, feat_2d, visibility):
        _ = visibility
        return feat_3d + feat_2d

    trainer._build_3d_features_bg_plus_rigid_in = MethodType(_build_3d, trainer)
    trainer._fuse_features = MethodType(_fuse, trainer)

    node_bg = _dummy_bg_state(3)
    route = RigidRoute(
        S=torch.tensor([2, 3, 8, 9], dtype=torch.long),
        S_in=torch.tensor([2, 8], dtype=torch.long),
        S_out=torch.tensor([3, 9], dtype=torch.long),
        inside_mask_S=torch.tensor([True, False, True, False]),
        route_inside_global=torch.tensor([False] * 10, dtype=torch.bool),
        means_world_S=torch.tensor(
            [
                [1.0, 0.0, 0.0],  # S[0] -> in
                [2.0, 0.0, 0.0],  # S[1] -> out
                [3.0, 0.0, 0.0],  # S[2] -> in
                [4.0, 0.0, 0.0],  # S[3] -> out
            ],
            dtype=torch.float32,
        ),
        quats_world_S=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=torch.float32),
    )
    feat_2d_bg = torch.ones(3, 3, dtype=torch.float32)
    feat_2d_rigid_S = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # S[0] -> in
            [20.0, 0.0, 0.0],  # S[1] -> out
            [30.0, 0.0, 0.0],  # S[2] -> in
            [40.0, 0.0, 0.0],  # S[3] -> out
        ],
        dtype=torch.float32,
    )
    acc_w_bg = torch.ones(3, dtype=torch.float32)

    out = trainer._compute_bg_rigid_in_gru_inputs(
        source_frame_idx=0,
        node_state_bg=node_bg,
        node_state_rigid=None,
        route=route,
        feat_2d_bg=feat_2d_bg,
        feat_2d_rigid_S=feat_2d_rigid_S,
        acc_w_bg=acc_w_bg,
        acc_w_rigid_S=None,
    )
    assert out.feat_bg_input.shape == (3, 3)
    assert out.feat_rigid_in_input_all is not None
    # rigid S_in rows correspond to inside_mask_S rows [0, 2]
    assert float(out.feat_rigid_in_input_all[0, 0].item()) == 11.0
    assert float(out.feat_rigid_in_input_all[1, 0].item()) == 33.0


def test_stage5_0_build_struct_input_keeps_rigid_s_in_order_alignment():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    trainer.device = torch.device("cpu")
    trainer.bg_src_backproject_support_min = 1e-2
    trainer.rigid_src_backproject_support_min = 2e-2

    def _build_params(self, state, coord_space="world"):
        _ = coord_space
        return _params_from_state(state)

    def _build_rigid_params(self, node_state_rigid, source_frame_idx, u):
        _ = source_frame_idx
        return {
            "means": node_state_rigid.means[u],
            "scales_log": node_state_rigid.scales_log[u],
            "quats": node_state_rigid.quats[u],
            "opacity_logit": node_state_rigid.opacity_logit[u],
            "sh_dc": node_state_rigid.sh_dc[u],
            "sh_rest": node_state_rigid.sh_rest[u],
        }

    trainer._build_params_for_embed = MethodType(_build_params, trainer)
    trainer._build_rigid_params_for_embed_source_world = MethodType(_build_rigid_params, trainer)

    node_bg = _dummy_bg_state(5)
    node_rigid = _dummy_rigid_state(10)
    route = RigidRoute(
        S=torch.tensor([2, 3, 8, 9], dtype=torch.long),
        S_in=torch.tensor([2, 8], dtype=torch.long),
        S_out=torch.tensor([3, 9], dtype=torch.long),
        inside_mask_S=torch.tensor([True, False, True, False]),
        route_inside_global=torch.tensor([False, False, True, False, False, False, False, False, True, False]),
        means_world_S=torch.tensor(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4],
            ],
            dtype=torch.float32,
        ),
        quats_world_S=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=torch.float32),
    )

    feat_2d_bg = torch.arange(5 * 32, dtype=torch.float32).view(5, 32)
    feat_2d_rigid_S = torch.tensor(
        [
            [10.0] * 32,
            [20.0] * 32,
            [30.0] * 32,
            [40.0] * 32,
        ],
        dtype=torch.float32,
    )
    acc_w_bg = torch.ones(5, dtype=torch.float32)
    acc_w_rigid_S = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

    struct_in = trainer._build_struct_decoder_input_bg_rigid_in(
        source_frame_idx=0,
        node_state_bg=node_bg,
        node_state_rigid=node_rigid,
        route=route,
        feat_2d_bg=feat_2d_bg,
        feat_2d_rigid_S=feat_2d_rigid_S,
        acc_w_bg=acc_w_bg,
        acc_w_rigid_S=acc_w_rigid_S,
    )

    assert int(struct_in.split_bg) == 5
    assert int(struct_in.split_rigid_in) == 2
    assert struct_in.feat_2d.shape[0] == 7
    # rows in rigid S selected by inside_mask_S -> [0, 2] => features 10 and 30
    assert float(struct_in.feat_2d[5, 0].item()) == 10.0
    assert float(struct_in.feat_2d[6, 0].item()) == 30.0


def test_stage5_0_init_fast_fail_output_dim_must_match_fused_in_dim():
    trainer = MinimalStreetForwardStage5_0.__new__(MinimalStreetForwardStage5_0)
    trainer.device = torch.device("cpu")
    trainer.fused_in_dim = 64
    cfg = _base_stage5_cfg()
    cfg["model"]["feat_2d_channels"] = 32
    cfg["model"]["struct_decoder"].update(
        {
            "feat_2d_channels": 32,
            "output_dim": 32,
            "branch_embed_dim": 8,
            "support_embed_dim": 8,
            "param_embed_dim": 32,
            "channels": 64,
            "voxel_size": 0.2,
            "sparse_backend": "spconv",
            "xcpe": {
                "num_layers": 2,
                "kernel_size": 3,
                "residual_scale_init": 1e-3,
            },
            "token": {
                "use_2d_feat": True,
                "use_support": True,
                "use_branch_embed": True,
                "use_param_embed": True,
                "use_anchor_rgb": False,
                "use_hidden_state": False,
                "zero_invalid_2d_feat": True,
            },
        }
    )
    with pytest.raises(ValueError, match="output_dim must match GRU input dim"):
        trainer._init_stage5_0_struct_decoder(cfg)


def _make_decoder() -> StreetForwardXCPEDecoder:
    backend = "spconv" if _HAS_SPCONV else "fallback_neighbor_mean"
    return StreetForwardXCPEDecoder(
        feat_2d_channels=8,
        out_channels=12,
        channels=12,
        num_layers=2,
        voxel_size=0.5,
        param_dim=17,
        sparse_backend=backend,
        use_2d_feat=True,
        use_support=True,
        use_branch_embed=True,
        use_param_embed=True,
        zero_invalid_2d_feat=True,
    )


def test_stage5_0_decoder_spconv_backend_requires_spconv():
    if _HAS_SPCONV:
        pytest.skip("spconv is installed; requirement check is only meaningful when unavailable.")
    with pytest.raises(ImportError, match="requires spconv"):
        StreetForwardXCPEDecoder(
            feat_2d_channels=8,
            out_channels=12,
            channels=12,
            num_layers=1,
            voxel_size=0.5,
            param_dim=17,
            sparse_backend="spconv",
        )


def test_stage5_0_decoder_fast_fail_clamp_grid_coord_true():
    backend = "spconv" if _HAS_SPCONV else "fallback_neighbor_mean"
    with pytest.raises(ValueError, match="clamp_grid_coord=true"):
        StreetForwardXCPEDecoder(
            feat_2d_channels=8,
            out_channels=12,
            channels=12,
            num_layers=1,
            voxel_size=0.5,
            param_dim=17,
            sparse_backend=backend,
            clamp_grid_coord=True,
        )


def _make_struct_input(num_points: int = 6, split_bg: int = 4) -> StructDecoderInput:
    split_rigid = num_points - split_bg
    feat_2d = torch.randn(num_points, 8)
    acc_w = torch.rand(num_points)
    coords = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.3, 0.1, 0.1],
            [0.6, 0.2, 0.2],
            [0.9, 0.2, 0.2],
            [0.1, 0.6, 0.1],
            [0.6, 0.6, 0.6],
        ],
        dtype=torch.float32,
    )[:num_points]
    branch_id = torch.cat(
        [
            torch.zeros(split_bg, dtype=torch.long),
            torch.ones(split_rigid, dtype=torch.long),
        ],
        dim=0,
    )
    params = {
        "means": coords.clone(),
        "scales_log": torch.zeros(num_points, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_points, dtype=torch.float32),
        "opacity_logit": torch.zeros(num_points, 1),
        "sh_dc": torch.zeros(num_points, 3),
        "sh_rest": torch.zeros(num_points, 3, 3),
    }
    return StructDecoderInput(
        feat_2d=feat_2d,
        acc_w=acc_w,
        coords=coords,
        branch_id=branch_id,
        params_for_embed=params,
        split_bg=split_bg,
        split_rigid_in=split_rigid,
        meta={
            "support_threshold_bg": 0.0,
            "support_threshold_rigid": 0.0,
        },
    )


def test_stage5_0_struct_decoder_point_preserving_shape():
    decoder = _make_decoder()
    struct_in = _make_struct_input(num_points=6, split_bg=4)
    out = decoder(
        struct_in,
        aabb_min=torch.tensor([0.0, 0.0, 0.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        batch_offsets=torch.tensor([6], dtype=torch.long),
    )
    assert out.feat.shape == (6, 12)


def test_stage5_0_voxel_axis_mapping_is_bzyx_and_spatial_shape_zyx():
    decoder = _make_decoder()
    coords = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.9, 0.4, 0.6],
            [0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    layout = decoder._build_voxel_layout(
        coords,
        aabb_min=torch.tensor([0.0, 0.0, 0.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        batch_offsets=torch.tensor([3], dtype=torch.long),
    )
    assert layout.indices_bzyx.shape[1] == 4
    for i in range(layout.unique_key.shape[0]):
        b, x, y, z = layout.unique_key[i].tolist()
        ib, iz, iy, ix = layout.indices_bzyx[i].tolist()
        assert ib == b
        assert iz == z
        assert iy == y
        assert ix == x
    assert layout.spatial_shape_zyx.tolist() == [3, 3, 3]


def test_stage5_0_aabb_max_boundary_point_is_valid():
    decoder = _make_decoder()
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    layout = decoder._build_voxel_layout(
        coords,
        aabb_min=torch.tensor([0.0, 0.0, 0.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        batch_offsets=torch.tensor([2], dtype=torch.long),
    )
    assert [2, 2, 2] in layout.grid_coord_xyz.tolist()


def test_stage5_0_multi_batch_voxel_isolation_for_same_grid_coord():
    decoder = _make_decoder()
    coords = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    layout = decoder._build_voxel_layout(
        coords,
        aabb_min=torch.tensor([0.0, 0.0, 0.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        batch_offsets=torch.tensor([1, 2], dtype=torch.long),
    )
    assert layout.unique_key.shape[0] == 2
    assert int(layout.unique_key[0, 0].item()) == 0
    assert int(layout.unique_key[1, 0].item()) == 1
    assert layout.unique_key[0, 1:].tolist() == layout.unique_key[1, 1:].tolist()


def test_stage5_0_token_builder_validates_1d_shapes_and_branch_range():
    builder = StructTokenBuilder(
        feat_2d_channels=8,
        param_dim=17,
        channels=16,
        param_embed_dim=8,
        branch_embed_dim=4,
        support_embed_dim=4,
        use_2d_feat=True,
        use_support=True,
        use_branch_embed=True,
        use_param_embed=True,
    )
    n = 5
    feat_2d = torch.randn(n, 8)
    param_vec = torch.randn(n, 17)
    branch_ok = torch.zeros(n, dtype=torch.long)

    out = builder(
        feat_2d=feat_2d,
        acc_w=torch.randn(n, 1),
        branch_id=branch_ok,
        param_vec=param_vec,
    )
    assert out.shape == (n, 16)

    with pytest.raises(ValueError, match="acc_w must have 5 elements"):
        builder(
            feat_2d=feat_2d,
            acc_w=torch.randn(n, 2),
            branch_id=branch_ok,
            param_vec=param_vec,
        )

    with pytest.raises(ValueError, match="branch_id must be 0=bg or 1=rigid_in"):
        builder(
            feat_2d=feat_2d,
            acc_w=torch.randn(n),
            branch_id=torch.full((n,), 2, dtype=torch.long),
            param_vec=param_vec,
        )


def test_stage5_0_rot6d_semantics_match_stage4():
    quats = torch.randn(16, 4)
    stage4_rot6d = _stage4_quat_to_rot6d(quats)
    stage5_rot6d = _stage5_quat_to_rot6d(quats)
    assert stage4_rot6d.shape == stage5_rot6d.shape
    assert torch.allclose(stage4_rot6d, stage5_rot6d, atol=1e-6, rtol=1e-6)
