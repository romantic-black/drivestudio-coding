from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import (
    EventPack,
    LocalGSState,
    PHASE_B_NAME,
    Stage6PosteriorUpdater,
    Stage6QueryDecoder,
    Stage6ViewSetMemory,
    resolve_v9_phase_b_batch,
)


def _role_groups():
    return [
        {
            "role": "evidence",
            "allow_update_evidence": True,
            "allow_render_loss": False,
            "allow_query_label": False,
        },
        {
            "role": "prefix_loss",
            "allow_update_evidence": False,
            "allow_render_loss": True,
            "allow_query_label": False,
        },
        {
            "role": "query_label",
            "allow_update_evidence": False,
            "allow_render_loss": False,
            "allow_query_label": True,
        },
    ]


def _phase_b_meta():
    return {
        "scheduler_version": "v9",
        "scheduler_phase": PHASE_B_NAME,
        "assembly_mode": "image_ref_v9",
        "inner_K": 2,
        "scene_id": 1,
        "segment_id": 0,
        "episode_id": 7,
        "num_cams": 1,
        "evidence_refs_by_step": [[(10, 0)], [(11, 0)]],
        "block_loss_refs_by_step": [[], []],
        "nearby_loss_refs_by_step": [[], []],
        "prefix_loss_refs_by_step": [[(10, 0)], [(10, 0), (11, 0)]],
        "query_label_refs": [(12, 0)],
        "source_image_refs": [(10, 0), (11, 0)],
        "target_image_refs": [(10, 0), (11, 0)],
        "target_image_roles": ["prefix_loss", "prefix_loss"],
        "flat_evidence_refs": [(10, 0), (11, 0)],
        "flat_render_loss_refs": [(10, 0), (11, 0)],
        "role_policy": {
            "evidence": "update_only",
            "prefix_loss": "loss_only",
            "query_label": "label_only",
        },
        "role_groups": _role_groups(),
    }


def _add_tbptt_meta(batch, *, chunk_idx=0, is_first=True, is_last=False, frames=None, prior_frames=None):
    out = deepcopy(batch)
    frames = [int(x) for x in (frames if frames is not None else [10, 11])]
    prior_frames = [int(x) for x in (prior_frames or [])]
    out["request_meta"]["tbptt"] = {
        "enable": True,
        "strict": True,
        "stream_id": "default",
        "chunk_idx": int(chunk_idx),
        "is_first_chunk": bool(is_first),
        "is_last_chunk": bool(is_last),
        "start_block_idx": int(chunk_idx) * len(frames),
        "end_block_idx": int(chunk_idx) * len(frames) + len(frames),
        "event_block_indices": list(range(int(chunk_idx) * len(frames), int(chunk_idx) * len(frames) + len(frames))),
        "event_frame_indices": [int(x) for x in frames],
        "prior_written_frames": [int(x) for x in prior_frames],
        "prior_written_refs": [(int(x), 0) for x in prior_frames],
    }
    return out


def _phase_b_batch_for_frames(frame0: int, frame1: int, query_frame: int):
    batch = _phase_b_batch()
    meta = deepcopy(batch["request_meta"])
    meta["evidence_refs_by_step"] = [[(int(frame0), 0)], [(int(frame1), 0)]]
    meta["prefix_loss_refs_by_step"] = [[(int(frame0), 0)], [(int(frame0), 0), (int(frame1), 0)]]
    meta["query_label_refs"] = [(int(query_frame), 0)]
    meta["source_image_refs"] = [(int(frame0), 0), (int(frame1), 0)]
    meta["target_image_refs"] = [(int(frame0), 0), (int(frame1), 0)]
    meta["flat_evidence_refs"] = [(int(frame0), 0), (int(frame1), 0)]
    meta["flat_render_loss_refs"] = [(int(frame0), 0), (int(frame1), 0)]
    batch["request_meta"] = meta
    for idx, frame in enumerate([int(frame0), int(frame1)]):
        batch["targets"][idx]["frame_idx"] = int(frame)
    batch["query_targets"][0]["frame_idx"] = int(query_frame)
    return batch


def _phase_b_grouped_repeat_batch():
    batch = _phase_b_batch()
    meta = deepcopy(batch["request_meta"])
    meta.update(
        {
            "inner_K": 4,
            "phase_b_rollout_mode": "episode_grouped_repeat_tbptt",
            "evidence_refs_by_step": [[(10, 0)], [(10, 0)], [(11, 0)], [(11, 0)]],
            "block_loss_refs_by_step": [[], [], [], []],
            "nearby_loss_refs_by_step": [[], [], [], []],
            "prefix_loss_refs_by_step": [[(10, 0)], [(10, 0)], [(10, 0), (11, 0)], [(10, 0), (11, 0)]],
            "source_image_refs": [(10, 0), (11, 0)],
            "target_image_refs": [(10, 0), (11, 0)],
            "flat_evidence_refs": [(10, 0), (11, 0)],
            "flat_render_loss_refs": [(10, 0), (11, 0)],
            "phase_b_repeat": {
                "mode": "episode_grouped_repeat_tbptt",
                "pattern_name": "r2_b2",
                "repeats_per_block": 2,
                "blocks_per_chunk": 2,
                "actual_blocks_per_chunk": 2,
                "inner_K": 4,
                "step_block_indices": [0, 0, 1, 1],
                "step_repeat_indices": [0, 1, 0, 1],
                "step_memory_write_flags": [True, False, True, False],
                "step_source_frame_indices": [10, 10, 11, 11],
                "unique_event_block_indices": [0, 1],
                "unique_event_frame_indices": [10, 11],
            },
            "tbptt": {
                "enable": True,
                "strict": True,
                "stream_id": "grouped_repeat",
                "chunk_idx": 0,
                "is_first_chunk": True,
                "is_last_chunk": False,
                "start_block_idx": 0,
                "end_block_idx": 2,
                "event_block_indices": [0, 1],
                "event_frame_indices": [10, 11],
                "step_event_frame_indices": [10, 10, 11, 11],
                "prior_written_frames": [],
                "prior_written_refs": [],
                "query_exclude_frames": [10, 11],
            },
        }
    )
    batch["request_meta"] = meta
    batch["_scheduler_v9"]["inner_K"] = 4
    return batch


def _phase_b_batch():
    meta = _phase_b_meta()
    return {
        "scene_id": 1,
        "segment_id": 0,
        "request_meta": meta,
        "_scheduler_v9": {
            "scheduler_version": "v9",
            "phase": PHASE_B_NAME,
            "inner_K": 2,
        },
        "source_views": [object(), object()],
        "source_images": [torch.zeros(2, 2, 3), torch.zeros(2, 2, 3)],
        "source_sky_masks": [torch.zeros(2, 2), torch.zeros(2, 2)],
        "source_egocar_masks": [torch.zeros(2, 2), torch.zeros(2, 2)],
        "targets": [
            {
                "frame_idx": 10,
                "cam_idx": 0,
                "view": object(),
                "gt_image": torch.zeros(2, 2, 3),
                "sky_mask": torch.zeros(2, 2),
                "egocar_mask": torch.zeros(2, 2),
                "dynamic_mask": torch.zeros(2, 2),
                "valid_mask": torch.ones(2, 2),
            },
            {
                "frame_idx": 11,
                "cam_idx": 0,
                "view": object(),
                "gt_image": torch.zeros(2, 2, 3),
                "sky_mask": torch.zeros(2, 2),
                "egocar_mask": torch.zeros(2, 2),
                "dynamic_mask": torch.zeros(2, 2),
                "valid_mask": torch.ones(2, 2),
            },
        ],
        "query_targets": [
            {
                "frame_idx": 12,
                "cam_idx": 0,
                "view": object(),
                "gt_image": torch.zeros(2, 2, 3),
                "sky_mask": torch.zeros(2, 2),
                "egocar_mask": torch.zeros(2, 2),
                "dynamic_mask": torch.zeros(2, 2),
                "valid_mask": torch.ones(2, 2),
            }
        ],
    }


def test_phase_b_resolver_maps_indices_and_allows_prefix_evidence_overlap():
    resolved = resolve_v9_phase_b_batch(_phase_b_batch())
    assert resolved.inner_K == 2
    assert resolved.evidence_source_indices_by_step == [[0], [1]]
    assert resolved.prefix_target_indices_by_step == [[0], [0, 1]]
    assert resolved.query_target_indices == [0]
    assert resolved.memory_write_flags_by_step == [True, True]
    assert resolved.step_block_indices == [-1, -1]
    assert resolved.step_repeat_indices == [0, 0]
    assert resolved.step_source_frame_indices == [10, 11]


def test_phase_b_resolver_accepts_grouped_repeat_and_maps_repeated_evidence():
    resolved = resolve_v9_phase_b_batch(_phase_b_grouped_repeat_batch())
    assert resolved.inner_K == 4
    assert resolved.evidence_source_indices_by_step == [[0], [0], [1], [1]]
    assert resolved.prefix_target_indices_by_step == [[0], [0], [0, 1], [0, 1]]
    assert resolved.memory_write_flags_by_step == [True, False, True, False]
    assert resolved.step_block_indices == [0, 0, 1, 1]
    assert resolved.step_repeat_indices == [0, 1, 0, 1]
    assert resolved.step_source_frame_indices == [10, 10, 11, 11]
    assert resolved.phase_b_repeat["step_repeat_indices"] == [0, 1, 0, 1]
    assert resolved.tbptt_meta["step_event_frame_indices"] == [10, 10, 11, 11]


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda m: m["phase_b_repeat"].__setitem__("step_memory_write_flags", [True]), "length must equal inner_K"),
        (
            lambda m: m["phase_b_repeat"].__setitem__("step_memory_write_flags", [True, True, True, False]),
            "exactly one first-repeat",
        ),
        (
            lambda m: m["phase_b_repeat"].__setitem__("step_source_frame_indices", [10, 12, 11, 11]),
            "source frame must be fixed",
        ),
    ],
)
def test_phase_b_resolver_rejects_bad_grouped_repeat_meta(mutator, match):
    batch = _phase_b_grouped_repeat_batch()
    meta = deepcopy(batch["request_meta"])
    mutator(meta)
    batch["request_meta"] = meta
    with pytest.raises(ValueError, match=match):
        resolve_v9_phase_b_batch(batch)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda m: m.__setitem__("scheduler_phase", "phase_A_block_local_unroll"), "phase_B_viewset_rollout"),
        (lambda m: m["nearby_loss_refs_by_step"].__setitem__(0, [(13, 0)]), "nearby_loss_refs"),
        (lambda m: m["block_loss_refs_by_step"].__setitem__(0, [(13, 0)]), "block_loss_refs"),
        (lambda m: m.__setitem__("query_label_refs", [(10, 0)]), "query_label_refs leaked"),
        (lambda m: m.__setitem__("target_image_roles", ["prefix_loss", "block_loss"]), "target roles"),
    ],
)
def test_phase_b_resolver_fast_fails_bad_roles(mutator, match):
    batch = _phase_b_batch()
    meta = deepcopy(batch["request_meta"])
    mutator(meta)
    batch["request_meta"] = meta
    with pytest.raises(ValueError, match=match):
        resolve_v9_phase_b_batch(batch)


def test_phase_b_resolver_fast_fails_query_written_leakage():
    with pytest.raises(ValueError, match="already written"):
        resolve_v9_phase_b_batch(_phase_b_batch(), written_refs={(12, 0)})


def test_phase_b_resolver_fast_fails_order_mismatch():
    batch = _phase_b_batch()
    batch["targets"][1]["frame_idx"] = 99
    with pytest.raises(ValueError, match="targets order/content mismatch"):
        resolve_v9_phase_b_batch(batch)


def test_stage6_vsm_query_decoder_shapes_finite_and_gradients():
    torch.manual_seed(0)
    memory = Stage6ViewSetMemory(event_dim=8, view_code_dim=2, num_tokens=4, token_dim=8, proto_dim=3, global_dim=8, ctx_dim=8)
    decoder = Stage6QueryDecoder(input_dim=8, event_dim=8, obs_code_dim=2, hidden_dim=16)
    state = memory.init_state(num_bg=5, num_rigid=3, device=torch.device("cpu"), dtype=torch.float32)
    event = torch.randn(5, 8)
    view = torch.randn(5, 2)
    valid = torch.ones(5, 1)
    support = torch.rand(5, 1)
    state = memory.update(state=state, event_bg=event, view_code_bg=view, valid_bg=valid, support_bg=support)
    rigid_indices = torch.tensor([2, 0], dtype=torch.long)
    rigid_event = torch.randn(2, 8)
    rigid_view = torch.randn(2, 2)
    state_before = state.tokens_rigid.detach().clone()
    state = memory.update_rigid(
        state=state,
        indices=rigid_indices,
        event_rigid=rigid_event,
        view_code_rigid=rigid_view,
        valid_rigid=torch.ones(2, 1),
        support_rigid=torch.rand(2, 1),
    )
    ctx, aux = memory.query(state=state, view_code_bg=view)
    ctx_rigid, rigid_aux = memory.query_rigid(state=state, indices=rigid_indices, view_code_rigid=rigid_view)
    pred = decoder(
        state=state,
        query_view_code_bg=view,
        query_view_code_rigid=rigid_view,
        rigid_indices=rigid_indices,
        memory=memory,
    )
    assert ctx.shape == (5, 8)
    assert ctx_rigid.shape == (2, 8)
    assert pred.event_bg_hat.shape == (5, 8)
    assert pred.visible_logit_bg.shape == (5, 1)
    assert pred.support_log_bg_hat.shape == (5, 1)
    assert pred.obs_code_bg_hat.shape == (5, 2)
    assert pred.rigid is not None
    assert pred.rigid.event_hat.shape == (2, 8)
    assert torch.equal(state.tokens_rigid[1], state_before[1])
    assert not torch.equal(state.tokens_rigid[0], state_before[0])
    assert not torch.equal(state.tokens_rigid[2], state_before[2])
    assert aux["vsm_ctx_norm"] >= 0.0
    assert rigid_aux["vsm_rigid_vsm_ctx_norm"] >= 0.0
    loss = (
        pred.event_bg_hat.square().mean()
        + pred.visible_logit_bg.square().mean()
        + pred.rigid.event_hat.square().mean()
    )
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in memory.parameters() if p.grad is not None) > 0.0
    assert sum(float(p.grad.abs().sum()) for p in decoder.parameters() if p.grad is not None) > 0.0


def test_stage6_vsm_update_uses_token_specific_proposals_and_reports_diagnostics():
    torch.manual_seed(7)
    memory = Stage6ViewSetMemory(
        event_dim=8,
        view_code_dim=2,
        num_tokens=4,
        token_dim=8,
        proto_dim=3,
        global_dim=8,
        ctx_dim=8,
        hidden_dim=16,
    )
    state = memory.init_state(num_bg=3, num_rigid=0, device=torch.device("cpu"), dtype=torch.float32)
    state, aux = memory.update_bg(
        state=state,
        event_bg=torch.randn(3, 8),
        view_code_bg=torch.randn(3, 2),
        valid_bg=torch.ones(3, 1),
        support_bg=torch.ones(3, 1),
        return_aux=True,
    )
    assert "vsm_update_assign_entropy" in aux
    assert "vsm_bg_vsm_update_assign_entropy" in aux
    assert aux["vsm_update_token_delta_norm"] > 0.0
    assert aux["vsm_update_proto_delta_norm"] > 0.0
    assert aux["vsm_token_pair_cosine_max"] <= 1.0
    assert not torch.allclose(state.tokens_bg[:, 0], state.tokens_bg[:, 1])


def test_stage6_vsm_no_rigid_state_uses_zero_length_tensors():
    memory = Stage6ViewSetMemory(event_dim=8, view_code_dim=2, num_tokens=4, token_dim=8, proto_dim=3, global_dim=8, ctx_dim=8)
    state = memory.init_state(num_bg=2, num_rigid=0, device=torch.device("cpu"), dtype=torch.float32)
    assert state.tokens_rigid.shape == (0, 4, 8)
    ctx, aux = memory.query_rigid(state=state, indices=torch.zeros((0,), dtype=torch.long), view_code_rigid=None)
    assert ctx.shape == (0, 8)
    assert aux["vsm_rigid_vsm_ctx_norm"] == 0.0
    state.assert_finite()


def test_stage6_vsm_zero_unseen_bg_ctx_masks_biases():
    torch.manual_seed(4)
    memory = Stage6ViewSetMemory(
        event_dim=8,
        view_code_dim=2,
        num_tokens=4,
        token_dim=8,
        proto_dim=3,
        global_dim=8,
        ctx_dim=8,
        bg_zero_unseen_ctx=True,
    )
    with torch.no_grad():
        memory.bg_memory.ctx_norm.bias.fill_(0.5)
        memory.bg_memory.global_to_ctx.bias.fill_(1.0)
    state = memory.init_state(num_bg=3, num_rigid=0, device=torch.device("cpu"), dtype=torch.float32)
    ctx_empty, aux_empty = memory.query(state=state, view_code_bg=torch.zeros(3, 2))
    assert torch.equal(ctx_empty, torch.zeros_like(ctx_empty))
    assert aux_empty["vsm_seen_ratio"] == 0.0


def test_stage6_vsm_zero_unseen_rigid_ctx_masks_biases():
    torch.manual_seed(3)
    memory = Stage6ViewSetMemory(
        event_dim=8,
        view_code_dim=2,
        num_tokens=4,
        token_dim=8,
        proto_dim=3,
        global_dim=8,
        ctx_dim=8,
        rigid_zero_unseen_ctx=True,
    )
    with torch.no_grad():
        memory.rigid_memory.ctx_norm.bias.fill_(0.5)
        memory.rigid_memory.global_to_ctx.bias.fill_(1.0)
    state = memory.init_state(num_bg=1, num_rigid=3, device=torch.device("cpu"), dtype=torch.float32)
    all_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    ctx_empty, aux_empty = memory.query_rigid(state=state, indices=all_indices, view_code_rigid=torch.zeros(3, 2))
    assert torch.equal(ctx_empty, torch.zeros_like(ctx_empty))
    assert aux_empty["vsm_rigid_vsm_seen_ratio"] == 0.0

    state = memory.update_rigid(
        state=state,
        indices=torch.tensor([1], dtype=torch.long),
        event_rigid=torch.randn(1, 8),
        view_code_rigid=torch.randn(1, 2),
        valid_rigid=torch.ones(1, 1),
        support_rigid=torch.ones(1, 1),
    )
    ctx, aux = memory.query_rigid(state=state, indices=all_indices, view_code_rigid=torch.zeros(3, 2))
    assert torch.equal(ctx[0], torch.zeros_like(ctx[0]))
    assert not torch.equal(ctx[1], torch.zeros_like(ctx[1]))
    assert torch.equal(ctx[2], torch.zeros_like(ctx[2]))
    assert aux["vsm_rigid_vsm_seen_ratio"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"valid_rigid": torch.ones(3, 1)}, "valid_rigid"),
        ({"support_rigid": torch.ones(3, 1)}, "support_rigid"),
        ({"view_code_rigid": torch.ones(2, 3)}, "view_code_rigid"),
    ],
)
def test_stage6_vsm_rigid_optional_shape_mismatch_fails_directly(kwargs, match):
    memory = Stage6ViewSetMemory(event_dim=8, view_code_dim=2, num_tokens=4, token_dim=8, proto_dim=3, global_dim=8, ctx_dim=8)
    state = memory.init_state(num_bg=1, num_rigid=3, device=torch.device("cpu"), dtype=torch.float32)
    args = {
        "state": state,
        "indices": torch.tensor([0, 2], dtype=torch.long),
        "event_rigid": torch.randn(2, 8),
        "view_code_rigid": torch.randn(2, 2),
        "valid_rigid": torch.ones(2, 1),
        "support_rigid": torch.ones(2, 1),
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        memory.update_rigid(**args)


def _minimal_phase_b_config():
    return {
        "model": {
            "stage": "6_0",
            "phase": PHASE_B_NAME,
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
                    "train_2d_frontend": False,
                    "train_residual_unet": False,
                    "train_fusion_neck": False,
                    "train_dinov2": False,
                    "train_v4_lift": False,
                },
                "struct_event_decoder": {"enable": True, "freeze": True},
                "event_encoder": {"enable": False},
                "current_context_adapter": {"enable": False},
                "vsm": {"enable": True, "scope": "bg_rigid", "branches": ["bg", "rigid"]},
                "query_decoder": {"enable": True, "branches": ["bg", "rigid"]},
                "posterior_updater": {
                    "input_event": True,
                    "input_current_ctx": False,
                    "input_vsm_ctx": True,
                    "freeze_base": True,
                    "train_vsm_ctx_adapter": True,
                    "phase_b_hooks": {"accept_vsm_ctx": True},
                    "branch_scope": {
                        "bg": {"enable": True},
                        "distant": {"enable": False},
                        "rigid": {"enable": True, "update_quat": False},
                    },
                },
            },
        },
        "scheduler_v9": {
            "enable": True,
            "phase": PHASE_B_NAME,
            "episode": {"blocks_per_episode": 4},
            "phase_B": {
                "rollout": {"K_choices": [2, 4], "distinct_event_frames": True},
                "masks": {
                    "vsm_scope": "bg_rigid",
                    "evidence_mask": "non_sky_non_egocar",
                    "prefix_loss_mask": "non_sky_non_egocar",
                    "query_label_mask": "non_sky_non_egocar",
                },
            },
        },
        "validation_v9": {"enable": False},
        "losses": {"phase_b": {}},
    }


def test_phase_b_config_validation_and_trainability():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    cfg = _minimal_phase_b_config()
    MinimalStreetForwardStage6_0._validate_stage6_0_phase_b_config(model, cfg)
    model.stage6_vsm = Stage6ViewSetMemory(event_dim=8, token_dim=8, global_dim=8, ctx_dim=8)
    model.stage6_query_decoder = Stage6QueryDecoder(input_dim=8, event_dim=8)
    model.stage6_posterior_updater = Stage6PosteriorUpdater(event_dim=8, hidden_dim=16, stage_hidden_dim=5, sh_degree=1, vsm_ctx_dim=8)
    MinimalStreetForwardStage6_0._configure_stage6_trainability_after_module_init(model, cfg)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("stage6_vsm.") for name in trainable)
    assert any(name.startswith("stage6_query_decoder.") for name in trainable)
    assert any(name.startswith("stage6_posterior_updater.vsm_ctx_adapter.") for name in trainable)
    assert not any(name.startswith("stage6_posterior_updater.trunk.") for name in trainable)


def test_stage6_posterior_updater_rigid_branch_clamps_are_independent():
    torch.manual_seed(4)
    updater = Stage6PosteriorUpdater(
        event_dim=8,
        hidden_dim=16,
        stage_hidden_dim=5,
        sh_degree=1,
        means_max_step_m=0.25,
        scales_log_max_step=0.08,
        quat_axis_angle_max_step_rad=0.08,
        opacity_logit_max_step=0.25,
        sh_max_step=0.10,
        hidden_max_step=1.0,
        vsm_ctx_dim=8,
        branch_clamps={
            "rigid": {
                "means_max_step_m": 0.02,
                "scales_log_max_step": 0.02,
                "quat_axis_angle_max_step_rad": 0.0,
                "opacity_logit_max_step": 0.10,
                "sh_max_step": 0.05,
                "hidden_max_step": 0.25,
            }
        },
    )
    event = EventPack(
        event_bg=torch.randn(3, 8),
        event_rigid=torch.randn(2, 8),
        support_bg=torch.ones(3, 1),
        valid_bg=torch.ones(3, 1),
        support_rigid=torch.ones(2, 1),
        valid_rigid=torch.ones(2, 1),
    )
    delta, _ = updater(event=event, ctx_current=None, ctx_vsm=None)
    assert delta.rigid is not None
    assert float(delta.rigid.means.detach().abs().max()) <= 0.02 + 1.0e-6
    assert float(delta.rigid.scales_log.detach().abs().max()) <= 0.02 + 1.0e-6
    assert float(delta.rigid.quat_axis_angle.detach().abs().max()) == 0.0
    assert float(delta.rigid.opacity_logit.detach().abs().max()) <= 0.10 + 1.0e-6
    assert float(delta.rigid.sh.detach().abs().max()) <= 0.05 + 1.0e-6
    assert float(delta.rigid.hidden.detach().abs().max()) <= 0.25 + 1.0e-6


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda cfg: cfg["model"]["stage6_0"]["vsm"].__setitem__("scope", "bg_static"), "scope=bg_rigid"),
        (
            lambda cfg: cfg["scheduler_v9"]["phase_B"]["masks"].__setitem__(
                "prefix_loss_mask", "valid_non_sky_non_egocar_non_dynamic"
            ),
            "dynamic mask",
        ),
        (
            lambda cfg: cfg["model"]["stage6_0"]["posterior_updater"]["branch_scope"]["rigid"].__setitem__(
                "enable", False
            ),
            "rigid.enable=true",
        ),
        (
            lambda cfg: cfg["model"]["stage6_0"]["posterior_updater"]["branch_scope"]["distant"].__setitem__(
                "enable", True
            ),
            "distant.enable=false",
        ),
        (lambda cfg: cfg["model"]["stage6_0"]["query_decoder"].__setitem__("branches", ["bg"]), "branches=\\[bg, rigid\\]"),
    ],
)
def test_phase_b_r_config_validation_rejects_legacy_or_missing_rigid(mutator, match):
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    cfg = _minimal_phase_b_config()
    mutator(cfg)
    with pytest.raises(ValueError, match=match):
        MinimalStreetForwardStage6_0._validate_stage6_0_phase_b_config(model, cfg)


def _make_bg_state(n: int = 4) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _make_rigid_state(n: int = 3) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
        point_ids=torch.arange(n, dtype=torch.long).reshape(n, 1),
        instances_quats=torch.zeros(1, 1, 4),
        instances_trans=torch.zeros(1, 1, 3),
        instances_fv=torch.ones(1, 1, dtype=torch.bool),
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )


def _phase_b_forward_model(*, with_rigid: bool = False) -> MinimalStreetForwardStage6_0:
    torch.manual_seed(1)
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.config = {"training": {"grad_clip": {"enable": False}, "bad_step": {"fail_on_nonfinite_grad": True}}}
    model.sh_degree = 1
    model.stage6_phase = PHASE_B_NAME
    model.stage6_hidden_dim = 5
    model.stage6_event_dim = 8
    model.stage6_branch_scope = {
        "bg": {
            "enable": True,
            "update_means": True,
            "update_scales": True,
            "update_quat": True,
            "update_opacity": True,
            "update_sh": True,
            "update_hidden": True,
        },
        "distant": {"enable": False, "update_hidden": False},
        "rigid": {
            "enable": True,
            "update_means": True,
            "update_scales": True,
            "update_quat": False,
            "update_opacity": True,
            "update_sh": True,
            "update_hidden": True,
        },
    }
    model.bbx_min = torch.full((3,), -10.0)
    model.bbx_max = torch.full((3,), 10.0)
    model.stage6_scale_log_min = -10.0
    model.stage6_scale_log_max = 4.0
    model.stage6_phase_b_tbptt_enable = True
    model.stage6_phase_b_tbptt_max_items = 4
    model.stage6_phase_b_tbptt_strict = False
    model.stage6_phase_b_tbptt_forbid_cache_eviction = False
    model.stage6_phase_b_tbptt_cache = {}
    model.stage6_writeback_policy = "block_end_detached"
    model.stage6_phase_b_prefix_enable = True
    model.stage6_phase_b_prefix_weight = 1.0
    model.stage6_phase_b_prefix_l1_weight = 1.0
    model.stage6_phase_b_prefix_ssim_weight = 0.0
    model.stage6_phase_b_prefix_mask_policy = "non_sky_non_egocar"
    model.stage6_phase_b_prefix_step_weight = "late_heavy_linear"
    model.stage6_phase_b_query_enable = True
    model.stage6_phase_b_query_weight = 0.05
    model.stage6_phase_b_query_warmup_steps = 1
    model.stage6_phase_b_query_event_weight = 1.0
    model.stage6_phase_b_query_visible_weight = 0.2
    model.stage6_phase_b_query_support_weight = 0.2
    model.stage6_phase_b_query_obs_code_weight = 0.1
    model.stage6_phase_b_delta_norm_weight = 0.0
    model.stage6_vsm = Stage6ViewSetMemory(event_dim=8, view_code_dim=2, num_tokens=4, token_dim=8, proto_dim=3, global_dim=8, ctx_dim=8, hidden_dim=16)
    model.stage6_query_decoder = Stage6QueryDecoder(input_dim=8, event_dim=8, obs_code_dim=2, hidden_dim=16)
    model.stage6_posterior_updater = Stage6PosteriorUpdater(event_dim=8, hidden_dim=16, stage_hidden_dim=5, sh_degree=1, vsm_ctx_dim=8)
    bg = _make_bg_state(4)
    rigid = _make_rigid_state(3) if with_rigid else None
    model._get_or_init_node_states_bg_rigid_distant = lambda batch: (bg, rigid, None)
    model._observe_v4_measurement = lambda **kwargs: {"source_frame_idx": int(kwargs["source_frame_idx"])}

    def build_event(*, local_state, measurement):
        n = int(local_state.bg.means.shape[0])
        frame = float(measurement["source_frame_idx"])
        base = torch.linspace(0.0, 1.0, steps=n * 8).reshape(n, 8)
        return EventPack(
            event_bg=base + frame * 0.01,
            event_rigid=(base[:2] + frame * 0.02) if with_rigid else None,
            support_bg=torch.ones(n, 1),
            support_rigid=torch.ones(2, 1) if with_rigid else None,
            valid_bg=torch.ones(n, 1),
            valid_rigid=torch.ones(2, 1) if with_rigid else None,
            view_code_bg=torch.full((n, 2), frame * 0.01),
            obs_code_bg=torch.full((n, 2), frame * 0.01),
            obs_code_rigid=torch.full((2, 2), frame * 0.02) if with_rigid else None,
            route=SimpleNamespace(S=torch.tensor([0, 2], dtype=torch.long)) if with_rigid else None,
            aux={},
        )

    def render_loss(*, local_state, batch, target_indices, mask_policy, pred_rgbs_out=None, gt_images_out=None, **kwargs):
        _ = batch, target_indices, mask_policy, kwargs
        loss = (local_state.bg.means[:, 0].mean() - 0.25).square()
        return loss, {"psnr": 10.0, "l1": float(loss.detach().item()), "ssim": 0.0, "valid_ratio": 1.0}

    model._build_stage6_event_from_measurement = build_event
    model._render_loss_for_indices = render_loss
    return model


def test_phase_b_forward_gradients_and_tbptt_cache_store():
    model = _phase_b_forward_model()
    batch = _phase_b_batch()
    out = MinimalStreetForwardStage6_0._forward_phase_b(model, batch)
    assert out["roles"].inner_K == 2
    assert out["vsm_state"].tokens_bg.shape[:2] == (4, 4)
    out["loss"].backward()
    adapter_grad = sum(
        float(p.grad.abs().sum())
        for p in model.stage6_posterior_updater.vsm_ctx_adapter.parameters()
        if p.grad is not None
    )
    vsm_grad = sum(float(p.grad.abs().sum()) for p in model.stage6_vsm.parameters() if p.grad is not None)
    query_grad = sum(float(p.grad.abs().sum()) for p in model.stage6_query_decoder.parameters() if p.grad is not None)
    assert adapter_grad > 0.0
    assert vsm_grad > 0.0
    assert query_grad > 0.0
    MinimalStreetForwardStage6_0._phase_b_store_state(
        model,
        key=out["tbptt_key"],
        local_state=out["local_G"],
        vsm_state=out["vsm_state"],
        written_refs=set(out["written_refs"]),
    )
    assert out["tbptt_key"] in model.stage6_phase_b_tbptt_cache
    with pytest.raises(ValueError, match="already written"):
        resolve_v9_phase_b_batch(batch, written_refs={(12, 0)})


def test_phase_b_grouped_repeat_writes_memory_once_per_block_and_keeps_gradients():
    model = _phase_b_forward_model()
    model.stage6_phase_b_tbptt_strict = True
    update_calls = {"bg": 0}
    original_update_bg = model.stage6_vsm.update_bg

    def counted_update_bg(**kwargs):
        update_calls["bg"] += 1
        return original_update_bg(**kwargs)

    model.stage6_vsm.update_bg = counted_update_bg
    out = MinimalStreetForwardStage6_0._forward_phase_b(model, _phase_b_grouped_repeat_batch())
    assert out["roles"].inner_K == 4
    assert update_calls["bg"] == 2
    assert [int(x["memory_write"]) for x in out["per_step"]] == [1, 0, 1, 0]
    assert [int(x["repeat_idx"]) for x in out["per_step"]] == [0, 1, 0, 1]
    assert set(out["written_refs"]) == {(10, 0), (11, 0)}
    assert out["tbptt_meta"]["event_frame_indices"] == [10, 11]
    assert out["tbptt_meta"]["step_event_frame_indices"] == [10, 10, 11, 11]
    assert out["loss_total_norm_by_weight"] > 0.0
    assert out["loss_total_norm_by_K"] > 0.0
    final = out["per_step"][-1]
    assert final["vsm_update_assign_entropy"] > 0.0
    assert final["vsm_update_token_delta_norm"] > 0.0
    out["loss"].backward()
    adapter_grad = sum(
        float(p.grad.abs().sum())
        for p in model.stage6_posterior_updater.vsm_ctx_adapter.parameters()
        if p.grad is not None
    )
    vsm_grad = sum(float(p.grad.abs().sum()) for p in model.stage6_vsm.parameters() if p.grad is not None)
    query_grad = sum(float(p.grad.abs().sum()) for p in model.stage6_query_decoder.parameters() if p.grad is not None)
    assert adapter_grad > 0.0
    assert vsm_grad > 0.0
    assert query_grad > 0.0


def test_phase_b_rigid_forward_passes_ctx_and_updates_rigid_vsm():
    model = _phase_b_forward_model(with_rigid=True)
    out = MinimalStreetForwardStage6_0._forward_phase_b(model, _phase_b_batch())
    assert out["vsm_state"].tokens_rigid.shape[:2] == (3, 4)
    assert out["vsm_state"].valid_count_rigid[0].item() > 0.0
    assert out["vsm_state"].valid_count_rigid[1].item() == 0.0
    assert out["vsm_state"].valid_count_rigid[2].item() > 0.0
    final = out["per_step"][-1]
    assert final["rigid_seen_ratio"] > 0.0
    assert "vsm_rigid_vsm_ctx_norm" in final
    assert out["query_stats"]["query_rows_rigid"] > 0.0


def test_phase_b_rigid_route_duplicate_fails_fast():
    model = _phase_b_forward_model(with_rigid=True)

    def bad_build_event(*, local_state, measurement):
        n = int(local_state.bg.means.shape[0])
        frame = float(measurement["source_frame_idx"])
        base = torch.linspace(0.0, 1.0, steps=n * 8).reshape(n, 8)
        return EventPack(
            event_bg=base + frame * 0.01,
            event_rigid=base[:2] + frame * 0.02,
            support_bg=torch.ones(n, 1),
            support_rigid=torch.ones(2, 1),
            valid_bg=torch.ones(n, 1),
            valid_rigid=torch.ones(2, 1),
            view_code_bg=torch.full((n, 2), frame * 0.01),
            obs_code_bg=torch.full((n, 2), frame * 0.01),
            obs_code_rigid=torch.full((2, 2), frame * 0.02),
            route=SimpleNamespace(S=torch.tensor([0, 0], dtype=torch.long)),
            aux={},
        )

    model._build_stage6_event_from_measurement = bad_build_event
    with pytest.raises(ValueError, match="duplicate rigid row indices"):
        MinimalStreetForwardStage6_0._forward_phase_b(model, _phase_b_batch())


def test_phase_b_vsm_local_rigid_shape_mismatch_fails_before_cache_store():
    model = _phase_b_forward_model(with_rigid=True)
    local_state = LocalGSState.from_node_states(bg=_make_bg_state(4), distant=None, rigid=_make_rigid_state(3), hidden_dim=5)
    bad_vsm = model.stage6_vsm.init_state(
        num_bg=4,
        num_rigid=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="VSM/local row mismatch"):
        MinimalStreetForwardStage6_0._phase_b_store_state(
            model,
            key=(1, 0, 7, "default"),
            local_state=local_state,
            vsm_state=bad_vsm,
            written_refs=set(),
        )


def test_phase_b_strict_tbptt_requires_cache_for_non_first_chunk():
    model = _phase_b_forward_model()
    model.stage6_phase_b_tbptt_strict = True
    batch = _add_tbptt_meta(
        _phase_b_batch_for_frames(20, 21, 22),
        chunk_idx=1,
        is_first=False,
        frames=[20, 21],
        prior_frames=[10, 11],
    )
    with pytest.raises(ValueError, match="non-first TBPTT chunk requires cache hit"):
        MinimalStreetForwardStage6_0._forward_phase_b(model, batch)


def test_phase_b_strict_tbptt_continuity_and_chunk_idx_checks():
    model = _phase_b_forward_model()
    model.stage6_phase_b_tbptt_strict = True
    first = _add_tbptt_meta(_phase_b_batch_for_frames(10, 11, 12), frames=[10, 11])
    out0 = MinimalStreetForwardStage6_0._forward_phase_b(model, first)
    MinimalStreetForwardStage6_0._phase_b_store_state(
        model,
        key=out0["tbptt_key"],
        local_state=out0["local_G"],
        vsm_state=out0["vsm_state"],
        written_refs=set(out0["written_refs"]),
        tbptt_meta=out0["tbptt_meta"],
    )
    second = _add_tbptt_meta(
        _phase_b_batch_for_frames(20, 21, 22),
        chunk_idx=1,
        is_first=False,
        frames=[20, 21],
        prior_frames=[10, 11],
    )
    out1 = MinimalStreetForwardStage6_0._forward_phase_b(model, second)
    assert out1["tbptt_cache_hit"] is True

    bad_chunk = _add_tbptt_meta(
        _phase_b_batch_for_frames(30, 31, 32),
        chunk_idx=3,
        is_first=False,
        frames=[30, 31],
        prior_frames=[10, 11, 20, 21],
    )
    with pytest.raises(ValueError, match="chunk_idx discontinuity"):
        MinimalStreetForwardStage6_0._forward_phase_b(model, bad_chunk)


def test_phase_b_strict_tbptt_rejects_out_of_order_event_frames():
    model = _phase_b_forward_model()
    model.stage6_phase_b_tbptt_strict = True
    batch = _add_tbptt_meta(_phase_b_batch_for_frames(11, 10, 12), frames=[11, 10])
    with pytest.raises(ValueError, match="not strictly chronological"):
        MinimalStreetForwardStage6_0._forward_phase_b(model, batch)


def test_phase_b_strict_tbptt_forbids_cache_eviction():
    model = _phase_b_forward_model()
    model.stage6_phase_b_tbptt_strict = True
    model.stage6_phase_b_tbptt_forbid_cache_eviction = True
    model.stage6_phase_b_tbptt_max_items = 1
    out = MinimalStreetForwardStage6_0._forward_phase_b(
        model,
        _add_tbptt_meta(_phase_b_batch_for_frames(10, 11, 12), frames=[10, 11]),
    )
    MinimalStreetForwardStage6_0._phase_b_store_state(
        model,
        key=out["tbptt_key"],
        local_state=out["local_G"],
        vsm_state=out["vsm_state"],
        written_refs=set(out["written_refs"]),
        tbptt_meta=out["tbptt_meta"],
    )
    with pytest.raises(RuntimeError, match="TBPTT cache full"):
        MinimalStreetForwardStage6_0._phase_b_store_state(
            model,
            key=(1, 0, 8, "default"),
            local_state=out["local_G"],
            vsm_state=out["vsm_state"],
            written_refs=set(out["written_refs"]),
            tbptt_meta=out["tbptt_meta"],
        )


def test_phase_b_strict_tbptt_rejects_cached_written_frame_query_overlap():
    model = _phase_b_forward_model()
    model.stage6_phase_b_tbptt_strict = True
    tbptt_meta = {
        "enable": True,
        "strict": True,
        "chunk_idx": 1,
        "is_first_chunk": False,
        "event_frame_indices": [20, 21],
        "prior_written_frames": [],
        "prior_written_refs": [],
    }
    cached_item = {
        "next_chunk_idx": 1,
        "last_event_frame_idx": 11,
        "written_refs": {(12, 1)},
    }
    with pytest.raises(ValueError, match="cached TBPTT written frame"):
        MinimalStreetForwardStage6_0._phase_b_validate_strict_tbptt_start(
            model,
            key=(1, 0, 7, "default"),
            tbptt_meta=tbptt_meta,
            query_label_refs=[(12, 0)],
            cache_hit=True,
            cached_item=cached_item,
        )


def test_phase_b_tbptt_cache_only_avoids_node_state_writeback():
    model = _phase_b_forward_model()
    model.stage6_phase_b_tbptt_strict = True
    model.stage6_writeback_policy = "tbptt_cache_only"
    model.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    bg_before = model._get_or_init_node_states_bg_rigid_distant({})[0].means.clone()
    batch = _add_tbptt_meta(_phase_b_batch_for_frames(10, 11, 12), frames=[10, 11])
    logs = MinimalStreetForwardStage6_0._train_step_phase_b(model, batch=batch, scheduler_node_sync=None)
    bg_after = model._get_or_init_node_states_bg_rigid_distant({})[0].means
    assert torch.equal(bg_before, bg_after)
    assert logs["phase_b/tbptt_cache_size"] == 1
