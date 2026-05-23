from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.node_states import NodeStateBackground
from models.streetforward.stage6_0 import (
    CurrentContextAdapter,
    LocalGSState,
    Stage6EventEncoder,
    Stage6ParamEncoder,
    Stage6PosteriorUpdater,
    resolve_v9_phase_a_batch,
)
from models.streetforward.stage6_0.phase_a_losses import target_valid_mask
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


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


def test_stage6_event_and_updater_receive_gradients():
    encoder = Stage6EventEncoder(z_dim=4, output_dim=8, hidden_dim=16, param_embed_dim=3)
    adapter = CurrentContextAdapter(event_dim=8, ctx_dim=8, hidden_dim=16)
    updater = Stage6PosteriorUpdater(event_dim=8, ctx_dim=8, hidden_dim=16, stage_hidden_dim=5, sh_degree=1)
    assert updater.trunk[0].in_features == 8
    z = torch.randn(6, 4)
    acc = torch.rand(6)
    obs = torch.randn(6, 2)
    view = torch.zeros(6, 2)
    param = torch.randn(6, 3)
    event = encoder(
        z_bg=z,
        acc_w_bg=acc,
        obs_code_bg=obs,
        view_code_bg=view,
        param_embed_bg=param,
    )
    ctx = adapter(event)
    delta, _ = updater(event=event, ctx_current=ctx, ctx_vsm=None)
    loss = delta.bg.means.pow(2).mean() + delta.bg.sh.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in encoder.parameters())
    assert any(p.grad is not None for p in updater.parameters())
    measurement_frontend = nn.Linear(2, 2)
    for p in measurement_frontend.parameters():
        p.requires_grad_(False)
    assert all(p.grad is None for p in measurement_frontend.parameters())


def test_stage6_event_encoder_rejects_feature_dim_mismatch():
    encoder = Stage6EventEncoder(z_dim=4, output_dim=8, hidden_dim=16, param_embed_dim=3)
    z = torch.randn(2, 4)
    kwargs = {
        "z_bg": z,
        "acc_w_bg": torch.ones(2),
        "obs_code_bg": torch.zeros(2, 2),
        "view_code_bg": torch.zeros(2, 2),
        "param_embed_bg": torch.zeros(2, 3),
    }
    bad_z = dict(kwargs)
    bad_z["z_bg"] = torch.randn(2, 5)
    with pytest.raises(ValueError, match="z dim mismatch"):
        encoder(**bad_z)
    bad_param = dict(kwargs)
    bad_param["param_embed_bg"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="param_embed dim mismatch"):
        encoder(**bad_param)


def test_stage6_param_encoder_uses_compact_summaries():
    bg = NodeStateBackground(
        means=torch.zeros(4, 3),
        scales_log=torch.zeros(4, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
        opacity_logit=torch.zeros(4, 1),
        sh_dc=torch.zeros(4, 3),
        sh_rest=torch.zeros(4, 3, 3),
    )
    encoder = Stage6ParamEncoder(sh_rest_input_dim=9, quat_scales_summary_dim=4, sh_rest_summary_dim=8)
    out = encoder(branch=bg, aabb_min=torch.tensor([-1.0, -1.0, -1.0]), aabb_max=torch.tensor([1.0, 1.0, 1.0]))
    assert out.shape == (4, 19)
    out.sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())
    assert bg.means.grad is None


def _stage6_encode_update_test_model(*, detach_v4_outputs: bool):
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.sh_degree = 1
    model.stage6_hidden_dim = 5
    model.stage6_param_embed_dim = 10
    model.stage6_param_encoder = None
    model.stage6_detach_v4_outputs = bool(detach_v4_outputs)
    model.stage6_event_encoder = Stage6EventEncoder(z_dim=4, output_dim=8, hidden_dim=16, param_embed_dim=10)
    model.stage6_current_context_adapter = CurrentContextAdapter(event_dim=8, ctx_dim=8, hidden_dim=16)
    model.stage6_posterior_updater = Stage6PosteriorUpdater(
        event_dim=8,
        ctx_dim=8,
        hidden_dim=16,
        stage_hidden_dim=5,
        sh_degree=1,
    )
    model.stage6_view_code_policy = "zero_phase_a_debug"
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
        "route": type("Route", (), {"S": torch.zeros((0,), dtype=torch.long)})(),
        "feat_2d_bg": z,
        "acc_w_bg": torch.ones(3),
        "obs_bg": torch.zeros(3, 2),
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
        "route": type("Route", (), {"S": torch.zeros((0,), dtype=torch.long)})(),
        "feat_2d_bg": z,
        "acc_w_bg": torch.ones(3),
        "obs_bg": torch.zeros(3, 2),
    }
    model = _stage6_encode_update_test_model(detach_v4_outputs=True)
    updated, _delta, _aux = MinimalStreetForwardStage6_0._encode_and_update(
        model,
        local_state=local,
        measurement=measurement,
    )
    updated.bg.means.sum().backward()
    assert z.grad is None


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
                "event_encoder": {
                    "param_embed_dim": 10,
                    "view_code_policy": "zero_phase_a_debug",
                    "allow_zero_view_code_phase_a": False,
                },
                "posterior_updater": {
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


def test_stage6_config_validation_accepts_from_scratch_2d_frontend(monkeypatch):
    cfg = _valid_stage6_config()
    cfg["model"]["stage6_0"]["phase_a_mode"] = "from_scratch"
    cfg["model"]["stage6_0"]["base_measurement"].update(
        {
            "source_evidence_grad_mode": "train_2d_detach_alpha",
            "train_2d_frontend": True,
            "train_residual_unet": True,
            "train_fusion_neck": True,
            "train_v4_lift": False,
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
            "event_encoder": 1.0e-4,
            "current_context_adapter": 1.0e-4,
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


def test_stage6_nonfinite_grad_fast_fails():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.config = {
        "training": {
            "grad_clip": {"enable": False, "max_norm": 1.0},
            "bad_step": {"fail_on_nonfinite_grad": True, "fail_on_grad_norm_gt": 100.0},
        }
    }
    model.stage6_event_encoder = nn.Linear(1, 1)
    param = next(model.stage6_event_encoder.parameters())
    param.requires_grad_(True)
    param.grad = torch.full_like(param, float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        MinimalStreetForwardStage6_0._stage6_compute_and_check_grad_norm(model)


def test_stage6_phase_b_export_contains_required_keys():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(model)
    model.image_feature_extractor = nn.Linear(1, 1)
    model.stage5_2_gate_mlp = nn.Linear(1, 1)
    model.stage6_event_encoder = Stage6EventEncoder(z_dim=4, output_dim=8, hidden_dim=16, param_embed_dim=3)
    model.stage6_current_context_adapter = CurrentContextAdapter(event_dim=8, ctx_dim=8, hidden_dim=16)
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
        "event_encoder",
        "posterior_updater_base",
        "current_context_adapter",
        "normalizer_stats",
    ):
        assert key in payload
    assert "image_feature_extractor.weight" in payload["measurement_frontend"]
    assert not any(k.startswith("stage5_2_gate_mlp") for k in payload["measurement_frontend"])
    assert "vsm" not in payload
    assert "query_decoder" not in payload
