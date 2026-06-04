from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import (
    LocalGSState,
    Stage6PosteriorUpdater,
    Stage6ParamObsCodec,
    Stage6RoutedStructEventDecoder,
    Stage6StructInput,
    resolve_v9_phase_a_batch,
)
from models.streetforward.stage6_0.phase_a_losses import delta_regularization
from models.streetforward.stage6_0.phase_a_losses import masked_rgb_loss
from models.streetforward.stage6_0.phase_a_losses import target_valid_mask
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack
from models.streetforward.validation_v9_runner import aggregate_validation_v9_phase_a_rows


def _phase_a_batch_meta():
    return {
        "scheduler_version": "v9",
        "scheduler_phase": "phase_A_block_local_unroll",
        "assembly_mode": "image_ref_v9",
        "inner_K": 2,
        "evidence_refs_by_step": [[(10, 0)], [(10, 0)]],
        "block_loss_refs_by_step": [[(10, 0)], [(10, 0)]],
        "nearby_loss_refs_by_step": [[], [(11, 0)]],
        "prefix_loss_refs_by_step": [[], []],
        "query_label_refs": [],
        "source_image_refs": [(10, 0)],
        "target_image_refs": [(10, 0), (11, 0)],
        "target_image_roles": ["block_loss", "nearby_loss"],
    }


def _phase_a_batch():
    meta = _phase_a_batch_meta()
    return {
        "request_meta": meta,
        "_scheduler_v9": {
            "scheduler_version": "v9",
            "phase": "phase_A_block_local_unroll",
            "inner_K": 2,
        },
    }


def _validation_v9_minimal_batch(*, zero_valid_mask: bool = False):
    meta = _phase_a_batch_meta()
    meta["nearby_loss_refs_by_step"] = [[(11, 0)], [(11, 0)]]
    valid = torch.zeros(2, 2) if bool(zero_valid_mask) else torch.ones(2, 2)
    target_common = {
        "gt_image": torch.ones(2, 2, 3),
        "sky_mask": torch.zeros(2, 2),
        "egocar_mask": torch.zeros(2, 2),
        "valid_mask": valid,
    }
    return {
        "scene_id": 1,
        "segment_id": 0,
        "request_meta": meta,
        "_scheduler_v9": {
            "scheduler_version": "v9",
            "phase": "phase_A_block_local_unroll",
            "inner_K": 2,
        },
        "source_views": [object()],
        "targets": [
            dict(target_common, frame_idx=10, cam_idx=0),
            dict(target_common, frame_idx=11, cam_idx=0),
        ],
    }


def _validation_v9_runner_model(flags=None):
    bg = NodeStateBackground(
        means=torch.zeros(2, 3),
        scales_log=torch.zeros(2, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        opacity_logit=torch.zeros(2, 1),
        sh_dc=torch.zeros(2, 3),
        sh_rest=torch.zeros(2, 3, 3),
    )
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.stage6_hidden_dim = 5
    model.loss_w_ssim = 0.0
    model.stage6_block_mask_policy = "none"
    model.stage6_nearby_mask_policy = "none"
    model.node_states_bg = {(1, 0): bg}
    model.node_states_distant = {}
    model.node_states_rigid = {}
    model.node_states_sky = {}
    model.h_cache_bg = {}
    model.h_cache_distant = {}
    model.h_cache_rigid = {}
    model.h_cache_sky = {}

    def _get_or_init(_batch):
        return bg, None, None

    def _observe_v4_measurement(*, local_state, batch, source_indices, source_frame_idx):
        _ = local_state, batch, source_indices, source_frame_idx
        if flags is not None:
            flags.append(bool(torch.is_grad_enabled()))
        return {"source_frame_idx": int(source_frame_idx)}

    def _encode_and_update(*, local_state, measurement):
        _ = measurement
        if flags is not None:
            flags.append(bool(torch.is_grad_enabled()))
        n = int(local_state.bg.means.shape[0])
        zeros3 = local_state.bg.means.new_zeros((n, 3))
        delta = BranchDelta(
            means=local_state.bg.means.new_full((n, 3), 0.1),
            scales_log=zeros3,
            quat_axis_angle=zeros3,
            opacity_logit=local_state.bg.means.new_zeros((n, 1)),
            sh=local_state.bg.means.new_zeros((n, 12)),
            hidden=local_state.bg.means.new_zeros((n, 5)),
            confidence=local_state.bg.means.new_full((n, 1), 0.75),
            noop=local_state.bg.means.new_full((n, 1), 0.25),
        )
        return local_state.apply_delta(DeltaPack(bg=delta)), DeltaPack(bg=delta), {}

    def _render_target(*, local_state, target):
        gt = target["gt_image"]
        val = local_state.bg.means[:, 0].mean().clamp(0.0, 1.0)
        return torch.ones_like(gt) * val, torch.ones(gt.shape[:2])

    model._get_or_init_node_states_bg_rigid_distant = _get_or_init
    model._observe_v4_measurement = _observe_v4_measurement
    model._encode_and_update = _encode_and_update
    model._render_target = _render_target
    return model, bg


def _route_empty():
    return type(
        "Route",
        (),
        {
            "S": torch.zeros((0,), dtype=torch.long),
            "S_in": torch.zeros((0,), dtype=torch.long),
            "S_out": torch.zeros((0,), dtype=torch.long),
            "inside_mask_S": torch.zeros((0,), dtype=torch.bool),
            "means_world_S": torch.zeros((0, 3)),
        },
    )()


def test_stage6_missing_rigid_frame_is_invisible_not_fatal():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    state = NodeStateRigid(
        means=torch.zeros(2, 3),
        scales_log=torch.zeros(2, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        opacity_logit=torch.zeros(2, 1),
        sh_dc=torch.zeros(2, 3),
        sh_rest=torch.zeros(2, 3, 3),
        point_ids=torch.zeros(2, 1, dtype=torch.long),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        instances_trans=torch.zeros(1, 1, 3),
        instances_fv=torch.ones(1, 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[7],
        cur_frame=7,
    )
    valid = model._stage6_rigid_point_valid_mask(state, frame_idx=93)
    assert valid.shape == (2,)
    assert valid.dtype == torch.bool
    assert not bool(valid.any())


def _param_dict(n: int, *, requires_grad: bool = False):
    return {
        "means": torch.zeros(n, 3, requires_grad=requires_grad),
        "scales_log": torch.zeros(n, 3, requires_grad=requires_grad),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1).requires_grad_(requires_grad),
        "opacity_logit": torch.zeros(n, 1, requires_grad=requires_grad),
        "sh_dc": torch.zeros(n, 3, requires_grad=requires_grad),
        "sh_rest": torch.zeros(n, 3, 3, requires_grad=requires_grad),
    }


def _stage6_decoder(*, feat_dim: int = 4, event_dim: int = 8, token_dim: int = 8) -> Stage6RoutedStructEventDecoder:
    if int(event_dim) != int(token_dim):
        raise ValueError("test Stage6 decoder requires event_dim==token_dim")
    return Stage6RoutedStructEventDecoder(
        feat_2d_dim=feat_dim,
        event_dim=event_dim,
        token_dim=token_dim,
        param_obs_dim=6,
        support_embed_dim=4,
        branch_embed_dim=4,
        near_num_blocks=1,
        near_voxel_size=0.5,
        near_sparse_backend="fallback_neighbor_mean",
        far_hidden_dim=token_dim,
        far_num_layers=2,
        param_obs_codec_cfg={"output_dim": 6, "branch_embed_dim": 4},
    )


def test_phase_a_resolver_maps_indices():
    roles = resolve_v9_phase_a_batch(_phase_a_batch())
    assert roles.inner_K == 2
    assert roles.evidence_source_indices_by_step == [[0], [0]]
    assert roles.block_target_indices_by_step == [[0], [0]]
    assert roles.nearby_target_indices_by_step == [[], [1]]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda b: b["request_meta"].update({"scheduler_version": "v8"}), "scheduler_v9"),
        (lambda b: b["request_meta"].update({"scheduler_phase": "phase_B_viewset_rollout"}), "Phase A"),
        (lambda b: b["request_meta"].update({"query_label_refs": [(12, 0)]}), "query_label"),
        (lambda b: b["request_meta"]["prefix_loss_refs_by_step"].__setitem__(0, [(12, 0)]), "prefix"),
        (lambda b: b["request_meta"].update({"target_image_roles": ["block_loss"]}), "length mismatch"),
        (lambda b: b["request_meta"]["evidence_refs_by_step"].__setitem__(0, []), "non-empty evidence"),
        (lambda b: b["request_meta"]["nearby_loss_refs_by_step"].__setitem__(1, [(10, 0)]), "leaked"),
    ],
)
def test_phase_a_resolver_rejects_bad_batches(mutate, match):
    batch = _phase_a_batch()
    mutate(batch)
    with pytest.raises(ValueError, match=match):
        resolve_v9_phase_a_batch(batch)


def test_stage6_struct_event_and_updater_receive_gradients():
    decoder = _stage6_decoder(feat_dim=4, event_dim=8)
    updater = Stage6PosteriorUpdater(event_dim=8, ctx_dim=8, hidden_dim=16, stage_hidden_dim=5, sh_degree=1)
    feat = torch.randn(6, 4, requires_grad=True)
    near_in = Stage6StructInput(
        feat_2d=feat,
        acc_w=torch.ones(6),
        obs_code=torch.randn(6, 2),
        coords=torch.rand(6, 3) * 1.6 - 0.8,
        branch_id=torch.zeros(6, dtype=torch.long),
        params_for_embed=_param_dict(6, requires_grad=True),
        split_0=6,
        split_1=0,
        meta={"support_threshold_bg": 0.0, "support_threshold_rigid": 0.0},
    )
    far_in = Stage6StructInput(
        feat_2d=torch.zeros(0, 4),
        acc_w=torch.zeros(0),
        obs_code=torch.zeros(0, 2),
        coords=torch.zeros(0, 3),
        branch_id=torch.zeros(0, dtype=torch.long),
        params_for_embed=_param_dict(0),
        split_0=0,
        split_1=0,
    )
    event = decoder(
        near_in=near_in,
        far_in=far_in,
        route=_route_empty(),
        aabb_min=torch.tensor([-1.0, -1.0, -1.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
    )
    delta, _ = updater(event=event, ctx_current=None, ctx_vsm=None)
    loss = delta.bg.means.pow(2).mean() + delta.bg.sh.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in decoder.parameters())
    assert any(p.grad is not None for p in updater.parameters())
    assert feat.grad is not None
    assert near_in.params_for_embed["means"].grad is None
    measurement_frontend = nn.Linear(2, 2)
    for p in measurement_frontend.parameters():
        p.requires_grad_(False)
    assert all(p.grad is None for p in measurement_frontend.parameters())


def test_stage6_struct_event_rejects_obs_code_shape():
    codec = Stage6ParamObsCodec(output_dim=6, branch_embed_dim=4)
    with pytest.raises(ValueError, match="obs_code"):
        codec(
            params_for_embed=_param_dict(2),
            obs_code=torch.zeros(2, 3),
            acc_w=torch.ones(2),
            branch_id=torch.zeros(2, dtype=torch.long),
            aabb_min=torch.tensor([-1.0, -1.0, -1.0]),
            aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        )


def test_stage6_param_obs_codec_uses_obs_code_and_detaches_params():
    params = _param_dict(4, requires_grad=True)
    obs = torch.randn(4, 2)
    encoder = Stage6ParamObsCodec(output_dim=6, branch_embed_dim=4)
    out = encoder(
        params_for_embed=params,
        obs_code=obs,
        acc_w=torch.ones(4),
        branch_id=torch.zeros(4, dtype=torch.long),
        aabb_min=torch.tensor([-1.0, -1.0, -1.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
    )
    assert out.shape == (4, 6)
    out.sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())
    assert params["means"].grad is None


def test_stage6_param_obs_codec_uses_thresholded_valid_mask():
    encoder = Stage6ParamObsCodec(
        output_dim=25,
        branch_embed_dim=4,
        norm="none",
        activation="none",
    )
    encoder.net = nn.Identity()
    out = encoder(
        params_for_embed=_param_dict(2),
        obs_code=torch.zeros(2, 2),
        acc_w=torch.tensor([0.1, 0.1]),
        branch_id=torch.zeros(2, dtype=torch.long),
        aabb_min=torch.tensor([-1.0, -1.0, -1.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        valid_mask=torch.tensor([False, True]),
    )
    valid_support_column = 17 + 2 + 1
    assert out[0, valid_support_column].item() == 0.0
    assert out[1, valid_support_column].item() == 1.0


def test_stage6_aabb_requires_explicit_bounds():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    with pytest.raises(RuntimeError, match="segment AABB"):
        MinimalStreetForwardStage6_0._stage6_aabb(model, torch.zeros(1, 3))


def test_stage6_rigid_reassembly_validates_route_counts():
    decoder = _stage6_decoder(feat_dim=4, event_dim=8)
    near_in = Stage6StructInput(
        feat_2d=torch.randn(3, 4),
        acc_w=torch.ones(3),
        obs_code=torch.zeros(3, 2),
        coords=torch.zeros(3, 3),
        branch_id=torch.tensor([0, 0, 1], dtype=torch.long),
        params_for_embed=_param_dict(3),
        split_0=2,
        split_1=1,
        meta={"support_threshold_bg": 0.0, "support_threshold_rigid": 0.0},
    )
    far_in = Stage6StructInput(
        feat_2d=torch.zeros(0, 4),
        acc_w=torch.zeros(0),
        obs_code=torch.zeros(0, 2),
        coords=torch.zeros(0, 3),
        branch_id=torch.zeros(0, dtype=torch.long),
        params_for_embed=_param_dict(0),
        split_0=0,
        split_1=0,
    )
    route = type(
        "Route",
        (),
        {
            "S": torch.tensor([0]),
            "inside_mask_S": torch.tensor([False]),
        },
    )()
    with pytest.raises(RuntimeError, match="true count"):
        decoder(
            near_in=near_in,
            far_in=far_in,
            route=route,
            aabb_min=torch.tensor([-1.0, -1.0, -1.0]),
            aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        )


def _stage6_encode_update_test_model(*, detach_v4_outputs: bool):
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.sh_degree = 1
    model.num_sh_bases = 4
    model.stage6_hidden_dim = 5
    model.stage6_detach_v4_outputs = bool(detach_v4_outputs)
    model.stage6_feat_2d_dim = 4
    model.stage6_event_dim = 8
    model.bbx_min = torch.tensor([-1.0, -1.0, -1.0])
    model.bbx_max = torch.tensor([1.0, 1.0, 1.0])
    model.bg_src_backproject_support_min = 0.0
    model.distant_src_backproject_support_min = 0.0
    model.rigid_src_backproject_support_min = 0.0
    model.stage6_struct_event_decoder = _stage6_decoder(feat_dim=4, event_dim=8)
    model.stage6_posterior_updater = Stage6PosteriorUpdater(
        event_dim=8,
        ctx_dim=8,
        hidden_dim=16,
        stage_hidden_dim=5,
        sh_degree=1,
    )
    model.stage6_branch_scope = {
        "bg": {
            "update_means": True,
            "update_scales": True,
            "update_quat": True,
            "update_opacity": True,
            "update_sh": True,
        },
        "distant": {
            "update_means": False,
            "update_scales": False,
            "update_quat": False,
            "update_opacity": True,
            "update_sh": True,
        },
        "rigid": {
            "update_means": True,
            "update_scales": True,
            "update_quat": True,
            "update_opacity": True,
            "update_sh": True,
        },
    }
    return model


def test_stage6_from_scratch_keeps_2d_measurement_gradient_path():
    bg = NodeStateBackground(
        means=torch.zeros(3, 3),
        scales_log=torch.zeros(3, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        opacity_logit=torch.zeros(3, 1),
        sh_dc=torch.zeros(3, 3),
        sh_rest=torch.zeros(3, 3, 3),
    )
    local = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=5)
    z = torch.randn(3, 4, requires_grad=True)
    measurement = {
        "route": _route_empty(),
        "feat_2d_bg": z,
        "acc_w_bg": torch.ones(3),
        "obs_bg": torch.zeros(3, 2),
        "source_frame_idx": 0,
    }
    model = _stage6_encode_update_test_model(detach_v4_outputs=False)
    updated, _delta, _aux = MinimalStreetForwardStage6_0._encode_and_update(
        model,
        local_state=local,
        measurement=measurement,
    )
    updated.bg.means.sum().backward()
    assert z.grad is not None


def test_stage6_updater_only_detaches_measurement_outputs():
    bg = NodeStateBackground(
        means=torch.zeros(3, 3),
        scales_log=torch.zeros(3, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        opacity_logit=torch.zeros(3, 1),
        sh_dc=torch.zeros(3, 3),
        sh_rest=torch.zeros(3, 3, 3),
    )
    local = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=5)
    z = torch.randn(3, 4, requires_grad=True)
    measurement = {
        "route": _route_empty(),
        "feat_2d_bg": z,
        "acc_w_bg": torch.ones(3),
        "obs_bg": torch.zeros(3, 2),
        "source_frame_idx": 0,
    }
    model = _stage6_encode_update_test_model(detach_v4_outputs=True)
    updated, _delta, _aux = MinimalStreetForwardStage6_0._encode_and_update(
        model,
        local_state=local,
        measurement=measurement,
    )
    updated.bg.means.sum().backward()
    assert z.grad is None


def test_stage6_encode_update_clamps_bg_means_to_segment_aabb():
    bg = NodeStateBackground(
        means=torch.tensor([[0.95, 0.0, 0.0], [-0.95, 0.0, 0.0]]),
        scales_log=torch.zeros(2, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        opacity_logit=torch.zeros(2, 1),
        sh_dc=torch.zeros(2, 3),
        sh_rest=torch.zeros(2, 3, 3),
    )
    local = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=5)
    measurement = {
        "route": _route_empty(),
        "feat_2d_bg": torch.randn(2, 4),
        "acc_w_bg": torch.ones(2),
        "obs_bg": torch.zeros(2, 2),
        "source_frame_idx": 0,
    }
    model = _stage6_encode_update_test_model(detach_v4_outputs=True)

    class FixedUpdater(nn.Module):
        def forward(self, *, event, ctx_current=None, ctx_vsm=None):
            _ = ctx_current, ctx_vsm
            n = int(event.event_bg.shape[0])
            zeros3 = event.event_bg.new_zeros((n, 3))
            delta = BranchDelta(
                means=event.event_bg.new_tensor([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
                scales_log=zeros3,
                quat_axis_angle=zeros3,
                opacity_logit=event.event_bg.new_zeros((n, 1)),
                sh=event.event_bg.new_zeros((n, 12)),
                hidden=event.event_bg.new_zeros((n, 5)),
                confidence=event.event_bg.new_zeros((n, 1)),
                noop=event.event_bg.new_zeros((n, 1)),
            )
            return DeltaPack(bg=delta), {}

    model.stage6_posterior_updater = FixedUpdater()
    updated, delta, _aux = MinimalStreetForwardStage6_0._encode_and_update(
        model,
        local_state=local,
        measurement=measurement,
    )
    assert torch.allclose(delta.bg.means[:, 0], torch.tensor([0.2, -0.2]))
    assert torch.allclose(
        updated.bg.means,
        torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )


def test_stage6_local_writeback_is_detached():
    bg = NodeStateBackground(
        means=torch.zeros(4, 3),
        scales_log=torch.zeros(4, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
        opacity_logit=torch.zeros(4, 1),
        sh_dc=torch.zeros(4, 3),
        sh_rest=torch.zeros(4, 3, 3),
    )
    local = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=5)
    zero = torch.zeros
    delta = DeltaPack(
        bg=BranchDelta(
            means=torch.ones(4, 3) * 0.01,
            scales_log=zero(4, 3),
            quat_axis_angle=zero(4, 3),
            opacity_logit=zero(4, 1),
            sh=zero(4, 12),
            hidden=zero(4, 5),
            confidence=zero(4, 1),
            noop=zero(4, 1),
        )
    )
    local = local.apply_delta(delta)
    local.writeback_detached(bg=bg, distant=None, rigid=None)
    for value in bg.__dict__.values():
        if torch.is_tensor(value):
            assert value.requires_grad is False
    assert torch.allclose(bg.means, torch.ones(4, 3) * 0.01)


def test_stage6_render_loss_batches_targets_per_frame():
    bg = NodeStateBackground(
        means=torch.zeros(1, 3),
        scales_log=torch.zeros(1, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity_logit=torch.zeros(1, 1),
        sh_dc=torch.zeros(1, 3),
        sh_rest=torch.zeros(1, 3, 3),
    )
    local = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=5)
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.sh_degree = 1
    model.loss_w_ssim = 0.0
    model.stage6_render_grouped_multiview_train = True
    calls = []

    def fake_multi_view(render_params, targets):
        assert "means_r" in render_params
        calls.append([int(t["frame_idx"]) for t in targets])
        out = []
        for target in targets:
            gt = target["gt_image"]
            pred = torch.full_like(gt, float(target["pred_value"])).requires_grad_(True)
            out.append((pred, torch.ones(gt.shape[:2])))
        return out

    def fail_single_view(*args, **kwargs):
        raise AssertionError("Stage6 Phase A should batch same-frame target cameras.")

    model._render_multi_view = fake_multi_view
    model._render_single_view = fail_single_view
    targets = [
        {"frame_idx": 10, "cam_idx": 0, "view": object(), "gt_image": torch.zeros(2, 2, 3), "pred_value": 0.1},
        {"frame_idx": 10, "cam_idx": 1, "view": object(), "gt_image": torch.zeros(2, 2, 3), "pred_value": 0.2},
        {"frame_idx": 11, "cam_idx": 0, "view": object(), "gt_image": torch.zeros(2, 2, 3), "pred_value": 0.3},
    ]

    pred_rgbs = []
    gt_images = []
    loss, stats = MinimalStreetForwardStage6_0._render_loss_for_indices(
        model,
        local_state=local,
        batch={"targets": targets},
        target_indices=[0, 1, 2],
        mask_policy="none",
        pred_rgbs_out=pred_rgbs,
        gt_images_out=gt_images,
    )

    assert calls == [[10, 10], [11]]
    assert stats["num_refs"] == 3.0
    assert torch.isclose(loss, torch.tensor(0.2), atol=1.0e-6)
    assert [float(x[0, 0, 0]) for x in pred_rgbs] == pytest.approx([0.1, 0.2, 0.3])
    assert len(gt_images) == 3
    assert all(x.device.type == "cpu" and not x.requires_grad for x in pred_rgbs)
    assert all(x.device.type == "cpu" and not x.requires_grad for x in gt_images)


def test_stage6_render_loss_can_disable_grouped_multiview():
    bg = NodeStateBackground(
        means=torch.zeros(1, 3),
        scales_log=torch.zeros(1, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity_logit=torch.zeros(1, 1),
        sh_dc=torch.zeros(1, 3),
        sh_rest=torch.zeros(1, 3, 3),
    )
    local = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=5)
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.sh_degree = 1
    model.loss_w_ssim = 0.0
    model.stage6_render_grouped_multiview_train = False
    calls = []

    def fail_multi_view(*args, **kwargs):
        raise AssertionError("grouped multiview render should be disabled")

    def fake_single_view(render_params, view, height, width):
        assert "means_r" in render_params
        calls.append((int(height), int(width)))
        value = 0.1 * float(len(calls))
        return torch.full((int(height), int(width), 3), value), torch.ones(int(height), int(width))

    model._render_multi_view = fail_multi_view
    model._render_single_view = fake_single_view
    targets = [
        {"frame_idx": 10, "cam_idx": 0, "view": object(), "gt_image": torch.zeros(2, 2, 3)},
        {"frame_idx": 10, "cam_idx": 1, "view": object(), "gt_image": torch.zeros(2, 2, 3)},
    ]

    loss, stats = MinimalStreetForwardStage6_0._render_loss_for_indices(
        model,
        local_state=local,
        batch={"targets": targets},
        target_indices=[0, 1],
        mask_policy="none",
    )

    assert calls == [(2, 2), (2, 2)]
    assert stats["num_refs"] == 2.0
    assert torch.isclose(loss, torch.tensor(0.15), atol=1.0e-6)


def test_validation_v9_runner_no_grad():
    flags = []
    model, _bg = _validation_v9_runner_model(flags)
    _ = MinimalStreetForwardStage6_0.validate_v9_phase_a(
        model,
        _validation_v9_minimal_batch(),
        k_values=[0, 2],
        max_K=2,
        mask_cfg={"block_loss_mask": "none", "nearby_loss_mask": "none", "min_valid_pixels": 1},
    )
    assert flags
    assert all(flag is False for flag in flags)


def test_validation_v9_runner_no_writeback():
    model, bg = _validation_v9_runner_model()
    before = bg.means.detach().clone()
    _ = MinimalStreetForwardStage6_0.validate_v9_phase_a(
        model,
        _validation_v9_minimal_batch(),
        k_values=[0, 2],
        max_K=2,
        mask_cfg={"block_loss_mask": "none", "nearby_loss_mask": "none", "min_valid_pixels": 1},
    )
    assert torch.allclose(bg.means, before)


def test_validation_v9_k_curve_outputs_all_k_values():
    model, _bg = _validation_v9_runner_model()
    row = MinimalStreetForwardStage6_0.validate_v9_phase_a(
        model,
        _validation_v9_minimal_batch(),
        k_values=[0, 1, 2],
        max_K=2,
        mask_cfg={"block_loss_mask": "none", "nearby_loss_mask": "none", "min_valid_pixels": 1},
    )
    for k in (0, 1, 2):
        assert f"val_v9/phaseA/block_psnr@{k}" in row
        assert f"val_v9/phaseA/nearby_psnr@{k}" in row
    assert "val_v9/phaseA/block_psnr_gain@2" in row
    assert "val_v9/phaseA/time_per_iter_ms" in row


def test_validation_v9_mask_skip_is_logged():
    model, _bg = _validation_v9_runner_model()
    model.stage6_block_mask_policy = "non_sky_non_egocar"
    model.stage6_nearby_mask_policy = "non_sky_non_egocar"
    row = MinimalStreetForwardStage6_0.validate_v9_phase_a(
        model,
        _validation_v9_minimal_batch(zero_valid_mask=True),
        k_values=[0, 2],
        max_K=2,
        mask_cfg={
            "block_loss_mask": "non_sky_non_egocar",
            "nearby_loss_mask": "non_sky_non_egocar",
            "min_valid_pixels": 32,
        },
    )
    assert row["val_v9/phaseA/block_skipped_no_valid_pixels@0"] == 1.0
    assert row["val_v9/phaseA/nearby_skipped_no_valid_pixels@0"] == 1.0
    assert row["val_v9/phaseA/block_metric_valid@0"] == 0.0
    assert row["val_v9/phaseA/nearby_metric_valid@0"] == 0.0
    assert "val_v9/phaseA/block_psnr@0" not in row
    assert "val_v9/phaseA/nearby_psnr@0" not in row


def test_validation_v9_aggregate_skips_missing_nearby_psnr():
    summary = aggregate_validation_v9_phase_a_rows(
        [
            {"scene_id": 1, "nearby_psnr@2": 20.0, "nearby_valid_ratio@2": 1.0},
            {"scene_id": 39, "nearby_valid_ratio@2": 0.0, "nearby_metric_valid@2": 0.0},
        ],
        k_values=[2],
    )
    assert summary["val_v9/phaseA/mean_nearby_psnr@2"] == 20.0


def _valid_stage6_config():
    return {
        "model": {
            "stage": "6_0",
            "phase": "phase_A_block_local_unroll",
            "backprojector_version": "v4",
            "history_memory": {"enable": False},
            "update_gate": {"enable": False},
            "view_transient": {"enable": False},
            "stage6_0": {
                "base_measurement": {
                    "type": "stage5_4_v4",
                    "require_fused_v4": True,
                    "require_obs_code": True,
                    "obs_code_dim": 2,
                    "source_evidence_grad_mode": "no_grad_v4",
                    "detach_v4_outputs": True,
                },
                "struct_event_decoder": {
                    "enable": True,
                    "event_dim": 8,
                    "feat_2d_dim": 4,
                    "token": {"token_dim": 8},
                    "param_obs_codec": {"output_dim": 6, "branch_embed_dim": 4},
                    "near": {
                        "type": "xcpe",
                        "sparse_backend": "fallback_neighbor_mean",
                        "num_blocks": 1,
                        "voxel_size": 0.5,
                    },
                    "far": {"type": "point_mlp", "hidden_dim": 8, "num_layers": 2},
                },
                "event_encoder": {
                    "enable": False,
                    "mode": "disabled_direct_concat_mlp",
                },
                "current_context_adapter": {"enable": False},
                "posterior_updater": {
                    "event_dim": 8,
                    "input_current_ctx": False,
                    "branch_scope": {
                        "distant": {
                            "update_means": False,
                            "update_scales": False,
                            "update_quat": False,
                            "update_opacity": True,
                            "update_sh": True,
                        }
                    }
                },
                "vsm": {"enable": False},
                "query_decoder": {"enable": False},
            },
        },
        "scheduler_v9": {"enable": True, "phase": "phase_A_block_local_unroll"},
        "losses": {
            "phase_a": {
                "disabled": {
                    "query_observation": True,
                    "prefix_render": True,
                }
            }
        },
    }


def _validate_with_parent_noop(cfg, monkeypatch):
    monkeypatch.setattr(MinimalStreetForwardStage5_4, "_validate_stage5_3_config", lambda self, config: None)
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    return MinimalStreetForwardStage6_0._validate_stage5_3_config(model, cfg)


@pytest.mark.parametrize(
    "path, value",
    [
        (("model", "stage6_0", "vsm", "enable"), True),
        (("model", "stage6_0", "query_decoder", "enable"), True),
        (("model", "stage6_0", "base_measurement", "type"), "other"),
        (("model", "stage6_0", "base_measurement", "require_fused_v4"), False),
        (("model", "stage6_0", "base_measurement", "source_evidence_grad_mode"), "full_debug"),
        (("model", "stage6_0", "base_measurement", "train_v4_lift"), True),
        (("model", "stage6_0", "base_measurement", "train_dinov2"), True),
        (("model", "stage6_0", "event_encoder", "enable"), True),
        (("model", "stage6_0", "event_encoder", "mode"), "direct_concat_mlp"),
        (("model", "stage6_0", "current_context_adapter", "enable"), True),
        (("model", "stage6_0", "posterior_updater", "input_current_ctx"), True),
        (("model", "history_memory", "enable"), True),
        (("model", "update_gate", "enable"), True),
        (("model", "view_transient", "enable"), True),
        (("model", "stage6_0", "posterior_updater", "branch_scope", "distant", "update_means"), True),
    ],
)
def test_stage6_config_validation_rejects_forbidden_phase_a_values(path, value, monkeypatch):
    cfg = _valid_stage6_config()
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        _validate_with_parent_noop(cfg, monkeypatch)


def test_stage6_config_validation_rejects_scheduler_v8_runtime(monkeypatch):
    cfg = _valid_stage6_config()
    cfg["scheduler_v8"] = {"enable": True}
    with pytest.raises(ValueError, match="scheduler_v8"):
        _validate_with_parent_noop(cfg, monkeypatch)


def test_stage6_config_validation_accepts_from_scratch_2d_frontend_and_v4_lift(monkeypatch):
    cfg = _valid_stage6_config()
    cfg["model"]["stage6_0"]["phase_a_mode"] = "from_scratch"
    cfg["model"]["stage6_0"]["base_measurement"].update(
        {
            "source_evidence_grad_mode": "train_2d_detach_alpha",
            "train_2d_frontend": True,
            "train_residual_unet": True,
            "train_fusion_neck": True,
            "train_v4_lift": True,
            "train_dinov2": False,
            "detach_v4_outputs": False,
        }
    )
    _validate_with_parent_noop(cfg, monkeypatch)


def test_stage6_config_validation_rejects_from_scratch_detached_outputs(monkeypatch):
    cfg = _valid_stage6_config()
    cfg["model"]["stage6_0"]["phase_a_mode"] = "from_scratch"
    cfg["model"]["stage6_0"]["base_measurement"].update(
        {
            "source_evidence_grad_mode": "train_2d_detach_alpha",
            "train_2d_frontend": True,
            "train_residual_unet": True,
            "train_fusion_neck": True,
            "detach_v4_outputs": True,
        }
    )
    with pytest.raises(ValueError, match="detach_v4_outputs=false"):
        _validate_with_parent_noop(cfg, monkeypatch)


def test_stage6_parent_bootstrap_uses_stage5_4_compat_config(monkeypatch):
    captured = {}

    def fake_parent_init(self, config, device, **kwargs):
        _ = kwargs
        nn.Module.__init__(self)
        captured["stage"] = config["model"]["stage"]
        captured["scheduler_v8_enable"] = bool(config["scheduler_v8"]["enable"])
        captured["history_enable"] = bool(config["model"]["history_memory"]["enable"])
        captured["update_gate_enable"] = bool(config["model"]["update_gate"]["enable"])
        captured["view_transient_enable"] = bool(config["model"]["view_transient"]["enable"])
        captured["optimizer_lr"] = float(config["optimizer"]["lr"])
        self.config = config
        self.device = device
        self.sh_degree = 1
        self.stage5_2_feat_2d_channels = 4

    monkeypatch.setattr(MinimalStreetForwardStage5_4, "__init__", fake_parent_init)
    cfg = _valid_stage6_config()
    cfg["scheduler_v9"]["episode"] = {"blocks_per_episode": 2, "include_source_frame": True}
    cfg["optimizer"] = {
        "type": "adamw",
        "lr": {
            "struct_event_decoder_near": 1.0e-4,
            "struct_event_decoder_far": 1.0e-4,
            "param_obs_codec": 1.0e-4,
            "posterior_updater": 1.0e-4,
        },
    }
    model = MinimalStreetForwardStage6_0(cfg, torch.device("cpu"))
    assert captured == {
        "stage": "5_4",
        "scheduler_v8_enable": True,
        "history_enable": True,
        "update_gate_enable": True,
        "view_transient_enable": True,
        "optimizer_lr": 1.0e-3,
    }
    assert model.config["model"]["stage"] == "6_0"
    assert model.config["model"]["history_memory"]["enable"] is False


def _branch_delta(fill: float = 1.0) -> BranchDelta:
    return BranchDelta(
        means=torch.full((2, 3), fill),
        scales_log=torch.full((2, 3), fill),
        quat_axis_angle=torch.full((2, 3), fill),
        opacity_logit=torch.full((2, 1), fill),
        sh=torch.full((2, 12), fill),
        hidden=torch.full((2, 5), fill),
        confidence=torch.full((2, 1), fill),
        noop=torch.full((2, 1), fill),
    )


def test_stage6_distant_branch_scope_freezes_geometry():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    model.stage6_branch_scope = {
        "bg": {
            "update_means": True,
            "update_scales": True,
            "update_quat": True,
            "update_opacity": True,
            "update_sh": True,
        },
        "distant": {
            "update_means": False,
            "update_scales": False,
            "update_quat": False,
            "update_opacity": True,
            "update_sh": True,
        },
        "rigid": {
            "update_means": True,
            "update_scales": True,
            "update_quat": True,
            "update_opacity": True,
            "update_sh": True,
        },
    }
    delta = DeltaPack(bg=_branch_delta(), distant=_branch_delta(), rigid=_branch_delta())
    masked = MinimalStreetForwardStage6_0._apply_branch_scope(model, delta)
    assert masked.distant is not None
    assert torch.count_nonzero(masked.distant.means) == 0
    assert torch.count_nonzero(masked.distant.scales_log) == 0
    assert torch.count_nonzero(masked.distant.quat_axis_angle) == 0
    assert torch.count_nonzero(masked.distant.opacity_logit) > 0
    assert torch.count_nonzero(masked.distant.sh) > 0


def test_phase_a_target_valid_mask_combines_valid_sky_egocar_and_dynamic():
    target = {
        "valid_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "sky_mask": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "egocar_mask": torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        "dynamic_mask": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    }
    mask = target_valid_mask(
        target,
        mask_policy="valid_non_sky_non_egocar_non_dynamic",
        device=torch.device("cpu"),
    )
    assert torch.equal(mask, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_phase_a_target_valid_mask_requires_sky_and_egocar_masks():
    with pytest.raises(ValueError, match="sky_mask"):
        target_valid_mask(
            {"egocar_mask": torch.zeros(2, 2)},
            mask_policy="non_sky_non_egocar",
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="egocar_mask"):
        target_valid_mask(
            {"sky_mask": torch.zeros(2, 2)},
            mask_policy="non_sky_non_egocar",
            device=torch.device("cpu"),
        )


def test_phase_a_zero_valid_mask_skips_rgb_loss():
    pred = torch.ones(2, 2, 3)
    gt = torch.zeros(2, 2, 3)
    loss, stats = masked_rgb_loss(
        pred,
        gt,
        mask=torch.zeros(2, 2),
        l1_weight=1.0,
        ssim_weight=0.0,
    )
    assert loss.item() == 0.0
    assert stats["skipped_no_valid_pixels"] == 1.0


def test_phase_a_render_loss_zero_valid_mask_omits_psnr_metric():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    model.loss_w_ssim = 0.0

    def _render_targets_grouped_by_frame(*, local_state, targets_with_indices):
        _ = local_state
        return {int(idx): (torch.ones(2, 2, 3), torch.ones(2, 2)) for idx, _target in targets_with_indices}

    model._render_targets_grouped_by_frame = _render_targets_grouped_by_frame
    local_state = SimpleNamespace(bg=SimpleNamespace(means=torch.zeros(1, 3)))
    batch = {
        "targets": [
            {
                "gt_image": torch.zeros(2, 2, 3),
                "valid_mask": torch.zeros(2, 2),
                "sky_mask": torch.zeros(2, 2),
                "egocar_mask": torch.zeros(2, 2),
            }
        ]
    }

    loss, stats = MinimalStreetForwardStage6_0._render_loss_for_indices(
        model,
        local_state=local_state,
        batch=batch,
        target_indices=[0],
        mask_policy="non_sky_non_egocar",
    )

    assert loss.item() == 0.0
    assert stats["metric_valid"] == 0.0
    assert stats["num_metric_refs"] == 0.0
    assert "psnr" not in stats


def test_stage6_delta_regularization_uses_extra_weights_and_scale_barrier():
    delta = DeltaPack(bg=_branch_delta(1.0))
    bg = NodeStateBackground(
        means=torch.zeros(2, 3),
        scales_log=torch.full((2, 3), 10.0),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        opacity_logit=torch.zeros(2, 1),
        sh_dc=torch.zeros(2, 3),
        sh_rest=torch.zeros(2, 3, 3),
    )
    local = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=5)
    loss, stats = delta_regularization(
        delta,
        weight=0.0,
        local_state=local,
        opacity_delta_l2_weight=0.5,
        sh_delta_l2_weight=0.25,
        scale_barrier_weight=0.1,
        scale_log_min=-1.0,
        scale_log_max=1.0,
    )
    assert loss.item() > 0.0
    assert stats["delta_opacity_l2"] > 0.0
    assert stats["delta_sh_l2"] > 0.0
    assert stats["scale_barrier"] > 0.0


def test_stage6_required_group_grad_fast_fails():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    param = nn.Parameter(torch.ones(1))
    with pytest.raises(RuntimeError, match="zero gradient"):
        MinimalStreetForwardStage6_0._assert_group_nonzero_grad(
            model,
            group_name="test_group",
            params=[param],
            required=True,
        )
    param.grad = torch.ones_like(param)
    assert MinimalStreetForwardStage6_0._assert_group_nonzero_grad(
        model,
        group_name="test_group",
        params=[param],
        required=True,
    ) == 1.0


def test_stage6_optimizer_splits_measurement_groups_and_no_weight_decay():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.stage6_struct_event_decoder = _stage6_decoder(feat_dim=4, event_dim=8)
    model.stage6_posterior_updater = Stage6PosteriorUpdater(
        event_dim=8,
        ctx_dim=8,
        hidden_dim=16,
        stage_hidden_dim=5,
        sh_degree=1,
    )
    model.image_feature_extractor = nn.Module()
    model.image_feature_extractor.residual_unet = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    model.image_feature_extractor.fusion_neck = nn.Linear(2, 2)
    for _name, param in model.named_parameters():
        param.requires_grad_(True)
    model.stage6_measurement_trainable_param_names = {
        name
        for name, _param in model.named_parameters()
        if name.startswith("image_feature_extractor.")
    }
    cfg = {
        "optimizer": {
            "type": "adamw",
            "lr": {
                "struct_event_decoder_near": 1.0e-4,
                "struct_event_decoder_far": 1.0e-4,
                "param_obs_codec": 1.0e-4,
                "posterior_updater": 1.0e-4,
                "measurement_frontend": 5.0e-5,
                "default": 1.0e-4,
            },
            "weight_decay": 0.1,
            "no_weight_decay": {"enable": True, "name_keywords": ["bias", "norm"], "ndim_leq": 1},
            "groups": {
                "residual_unet": {
                    "match": {"prefixes": ["image_feature_extractor.residual_unet"]},
                    "lr": 1.0e-3,
                    "weight_decay": 0.2,
                },
                "fusion_neck": {
                    "match": {"prefixes": ["image_feature_extractor.fusion_neck"]},
                    "lr": 1.5e-3,
                    "weight_decay": 0.3,
                },
            },
        }
    }
    MinimalStreetForwardStage6_0._rebuild_stage6_optimizer(model, cfg)
    groups = model.optimizer.param_groups
    residual = [g for g in groups if g.get("logical_name", "").startswith("stage6_measurement_frontend_residual_unet")]
    fusion = [g for g in groups if g.get("logical_name", "").startswith("stage6_measurement_frontend_fusion_neck")]
    assert residual and {float(g["lr"]) for g in residual} == {1.0e-3}
    assert fusion and {float(g["lr"]) for g in fusion} == {1.5e-3}
    assert any(g.get("logical_name", "").endswith("_no_weight_decay") and float(g["weight_decay"]) == 0.0 for g in groups)


def test_stage6_nonfinite_grad_fast_fails():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.config = {
        "training": {
            "grad_clip": {"enable": False, "max_norm": 1.0},
            "bad_step": {"fail_on_nonfinite_grad": True, "fail_on_grad_norm_gt": 100.0},
        }
    }
    model.stage6_struct_event_decoder = nn.Linear(1, 1)
    param = next(model.stage6_struct_event_decoder.parameters())
    param.requires_grad_(True)
    param.grad = torch.full_like(param, float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        MinimalStreetForwardStage6_0._stage6_compute_and_check_grad_norm(model)


def test_stage6_phase_b_export_contains_required_keys():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.image_feature_extractor = nn.Linear(1, 1)
    model.stage5_2_gate_mlp = nn.Linear(1, 1)
    model.stage6_event_dim = 8
    model.stage6_feat_2d_dim = 4
    model.stage6_struct_event_decoder = _stage6_decoder(feat_dim=4, event_dim=8)
    model.stage6_posterior_updater = Stage6PosteriorUpdater(
        event_dim=8,
        ctx_dim=8,
        hidden_dim=16,
        stage_hidden_dim=5,
        sh_degree=1,
    )
    payload = MinimalStreetForwardStage6_0.build_phase_b_export_checkpoint(model)
    assert payload["export_type"] == "stage6_0_phase_a_for_phase_b"
    for key in (
        "measurement_frontend",
        "struct_event_decoder",
        "param_obs_codec",
        "posterior_updater_base",
        "normalizer_stats",
        "event_schema",
    ):
        assert key in payload
    assert "image_feature_extractor.weight" in payload["measurement_frontend"]
    assert not any(k.startswith("stage5_2_gate_mlp") for k in payload["measurement_frontend"])
    assert payload["event_schema"]["near_path"] == "bg+rigid_in:xCPE"
    assert payload["legacy_event_encoder"] is None
    assert "vsm" not in payload
    assert "query_decoder" not in payload
