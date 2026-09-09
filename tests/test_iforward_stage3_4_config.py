from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from models.iforward.model import IForwardModel
from models.iforward.observation_feedback import ObservationFeedbackPolicy
from models.iforward.parent_spatial_backbone import (
    Stage34ParentGeometryResidualAdapter,
    Stage6ParentParamSupportCodec,
)
from models.iforward.stage2_3 import ParentOptimizerGatedDeltaKV
from models.iforward.versions import (
    IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
    is_stage3_0_iforward_version,
    is_stage3_1_iforward_version,
    is_stage3_4_iforward_version,
    is_stage3_optimizer_memory_iforward_version,
)
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0


_ROOT = Path(__file__).resolve().parents[1]
_STAGE33 = _ROOT / "configs" / "iforward" / "iforward_stage3_3_observation_feedback.yaml"
_STAGE34 = _ROOT / "configs" / "iforward" / "iforward_stage3_4_functional_parentgs_lift.yaml"


def test_stage3_4_has_independent_identity_and_optimizer_memory_routing() -> None:
    version = IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION
    assert version == "stage3_4_functional_parentgs_lift"
    assert is_stage3_4_iforward_version(version)
    assert is_stage3_optimizer_memory_iforward_version(version)
    assert not is_stage3_0_iforward_version(version)
    assert not is_stage3_1_iforward_version(version)
    assert is_stage3_optimizer_memory_iforward_version(
        "stage3_0_scalar_anchor_child_support_parent_legacy"
    )
    assert is_stage3_optimizer_memory_iforward_version("stage3_1_lowrank_gated_delta_kv_lift")


def test_stage3_4_complete_config_enforces_functional_parent_contract() -> None:
    stage33 = OmegaConf.load(_STAGE33)
    cfg = OmegaConf.load(_STAGE34)
    # Stage 3.4 is a complete runnable config derived from Stage 3.3, not a
    # partial override whose behavior depends on an implicit merge.
    assert set(cfg.keys()) == set(stage33.keys())

    ifwd = cfg.model.iforward
    assert cfg.output_name == "iforward_stage3_4_functional_parentgs_lift"
    assert ifwd.version == IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION
    assert ifwd.training_variant == "stage3_4_functional_parentgs_lift"
    assert list(cfg.initialization.skip_keys) == []

    policy = ObservationFeedbackPolicy.from_config(cfg)
    policy.validate_scheduler_modes(dict(cfg.scheduler_stage3_2.episode_recipe.train_2d_policy))
    assert policy.schedule.origin == "activation_step"
    assert policy.schedule.activation_step == 0
    assert policy.source_render.enable is True
    assert policy.source_render.checkpoint_scope == "full_dynamic_observation"
    assert policy.source_render.alpha_schedule.points[0] == (0, 0.0)
    assert policy.functional_parent.enable is True
    assert policy.functional_parent.branches == ("bg", "distant", "rigid_active")
    assert policy.functional_parent.start_after_model_updates == 1
    assert policy.functional_parent.alpha_schedule.points == policy.source_render.alpha_schedule.points
    assert policy.parent_projection.enable is False
    assert policy.parent_projection.forward_mode == "functional_per_visit"
    assert policy.parent_projection.backward_mode == "disabled"
    assert policy.relation.enable is False

    lift = ifwd.lifting
    assert lift.detach_geometry is True
    assert lift.parent.type == "functional_parent_direct_lift"
    assert lift.parent.feature_source == "features_2d"
    assert lift.parent.geometry_grad is False
    assert lift.parent.color_mode == "constant_zero"

    projector = ifwd.biggs.parent_projector
    assert projector.backend == "cuda_exact_diag"
    assert projector.covariance_mode == "diagonal"
    assert projector.mass_mode == "dynamic_tau_area"
    assert projector.recompute_every_visit is True
    assert projector.grad_to_local_state is True
    assert projector.allow_cpu_fallback is False
    assert projector.allow_torch_fallback is False
    assert projector.allow_forward_only is False
    assert projector.allow_surrogate_runtime_vjp is False
    assert ifwd.biggs.parent_state.mode == "functional_per_visit"
    assert ifwd.biggs.parent_state.persistent_geometry is False
    assert ifwd.biggs.parent_state.incremental_update is False

    contract = ifwd.biggs.gradient_contract
    assert contract.param_codec_geometry is True
    assert contract.lifting_geometry is False
    assert contract.ptv3_coords is False
    assert contract.assignment is False
    assert contract.relation_child_geometry is False
    assert contract.relation_parent_geometry is False
    assert ifwd.biggs.child_decoder.relation_source == "functional_detached_stats"
    assert ifwd.biggs.child_decoder.detach_relation_inputs is True
    assert ifwd.biggs.child_decoder.detach_child_code_inputs is True
    assert ifwd.biggs.child_decoder.detach_child_params is True
    assert ifwd.biggs.child_decoder.detach_parent_params is True
    assert ifwd.biggs.child_decoder.rigid_relation_space == "canonical"

    codec = ifwd.parent_spatial.param_codec
    assert codec.mode == "legacy17d_plus_geometry8d_residual"
    assert codec.schema == "legacy17d_plus_geometry8d_residual_v1"
    assert codec.output_dim == 24
    assert codec.grad_to_parent_params is True
    assert codec.detach_legacy_params is True
    assert codec.detach_support is True
    assert ifwd.parent_spatial.ptv3.detach_coords is True
    assert ifwd.parent_optimizer_memory.enable is True
    assert ifwd.parent_optimizer_memory.type == "lowrank_gated_delta_kv"
    assert ifwd.parent_optimizer_memory.detach_scope == "rollout_boundary"
    assert cfg.initialization.init_weights_only is True


def test_stage3_4_model_routes_biggs_parent_spatial_and_gdkv_modules() -> None:
    cfg = OmegaConf.load(_STAGE34)
    bridge = type("ConfigOnlyBridge", (), {"event_dim": 32})()
    model = IForwardModel(
        config=cfg,
        device=torch.device("cpu"),
        bridge=bridge,
    )
    assert model.is_stage3_4_functional_parentgs is True
    assert model.is_stage3_0_full_sparse_gather_lift is True
    assert model.is_stage2_0_biggs_parent_lifting is True
    assert model.is_stage2_3_optimizer_mamba is True
    assert isinstance(model.parent_spatial_backbone.param_support_codec, Stage6ParentParamSupportCodec)
    assert isinstance(
        model.parent_spatial_backbone.geometry_residual_adapter,
        Stage34ParentGeometryResidualAdapter,
    )
    assert model.parent_spatial_backbone.param_support_codec.detach_params is True
    assert isinstance(model.parent_temporal_mamba, ParentOptimizerGatedDeltaKV)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("model.iforward.lifting.parent.type", "legacy_direct_lift"),
        ("model.iforward.lifting.parent.geometry_grad", True),
        ("model.iforward.biggs.parent_projector.backend", "cuda_exact_diag_forward_only"),
        ("model.iforward.biggs.parent_projector.allow_torch_fallback", True),
        ("model.iforward.biggs.parent_projector.allow_forward_only", True),
        ("model.iforward.biggs.parent_projector.allow_surrogate_runtime_vjp", True),
        ("model.iforward.biggs.parent_projector.recompute_every_visit", False),
        ("model.iforward.biggs.parent_projector.recompute_every_visit", "false"),
        ("model.iforward.biggs.parent_projector.grad_to_local_state", False),
        ("model.iforward.biggs.parent_projector.grad_to_local_state", "false"),
        ("model.iforward.biggs.parent_projector.grad_mode", "stop_geometry"),
        ("model.iforward.biggs.parent_state.mode", "incremental_sufficient_stats"),
        ("model.iforward.biggs.parent_state.exact_refresh_policy", "block_enter"),
        ("model.iforward.biggs.gradient_contract.lifting_geometry", True),
        ("model.iforward.biggs.child_decoder.detach_parent_params", False),
        ("model.iforward.parent_spatial.param_codec.grad_to_parent_params", False),
        ("model.iforward.parent_spatial.param_codec.grad_to_parent_params", "false"),
        ("model.iforward.parent_spatial.param_codec.mode", "geometry_only_stage3_4"),
        ("model.iforward.parent_spatial.param_codec.schema", "geometry_only_13d_v1"),
        ("model.iforward.parent_spatial.param_codec.detach_legacy_params", False),
        ("model.iforward.parent_spatial.ptv3.detach_coords", False),
        ("model.iforward.parent_spatial.ptv3.detach_coords", "false"),
        ("model.iforward.observation_feedback.parent_projection.enable", True),
        ("model.iforward.observation_feedback.parent_projection.forward_mode", "incremental_runtime"),
        (
            "model.iforward.observation_feedback.parent_projection.backward_mode",
            "exact_diag_recompute_surrogate_vjp",
        ),
        ("model.iforward.observation_feedback.schedule.activation_step", 7),
        ("model.iforward.observation_feedback.functional_parent.enable", False),
        ("model.iforward.observation_feedback.functional_parent.branches", ["bg", "distant"]),
        ("model.iforward.observation_feedback.functional_parent.start_after_model_updates", 2),
        (
            "model.iforward.observation_feedback.functional_parent.alpha_schedule",
            [[0, 1.0]],
        ),
        ("model.iforward.parent_optimizer_memory.enable", False),
        ("model.iforward.parent_optimizer_memory.type", "mamba"),
        ("model.iforward.parent_optimizer_memory.reset_scope", "block"),
        ("model.iforward.parent_optimizer_memory.detach_scope", "visit"),
        ("model.iforward.training_variant", "stage3_3_observation_feedback"),
    ],
)
def test_stage3_4_runtime_contract_fails_fast_on_tampering(path: str, value: object) -> None:
    cfg = OmegaConf.load(_STAGE34)
    OmegaConf.update(cfg, path, value, merge=False)
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    with pytest.raises((TypeError, ValueError), match="Stage 3.4|functional ParentGS|parent_projection"):
        runtime._validate_stage3_4_functional_parentgs_runtime_config(
            cfg.model.iforward,
            cfg,
        )


def test_stage3_4_alias_cannot_bypass_canonical_contract() -> None:
    cfg = OmegaConf.load(_STAGE34)
    cfg.model.iforward.version = "iforward_stage3_4_functional_parentgs_lift"
    assert is_stage3_4_iforward_version(cfg.model.iforward.version)
    bridge = type("ConfigOnlyBridge", (), {"event_dim": 32})()
    with pytest.raises(ValueError, match="model.iforward.version"):
        IForwardModel(config=cfg, device=torch.device("cpu"), bridge=bridge)
