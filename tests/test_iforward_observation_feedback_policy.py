from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from models.iforward.observation_feedback import (
    FeedbackAlphaSchedule,
    FeedbackMode,
    FrontendParameterModeScope,
    ObservationFeedbackPolicy,
    scale_feedback,
)
from models.iforward.trainer import IForwardTrainer


def _feedback_cfg() -> dict:
    return {
        "enable": True,
        "scope": "within_rollout",
        "schedule": {"origin": "activation_step", "activation_step": 30000},
        "modes": {
            "repeat_refine": "trainable_checkpointed",
            "shuffled_coverage": "trainable_checkpointed",
            "high_block_repair": "frozen_input_grad_checkpointed",
        },
        "source_render": {
            "enable": True,
            "renderer_mode": "differentiable_rgb",
            "checkpoint_scope": "full_dynamic_observation",
            "absgrad": False,
            "alpha_schedule": [[0, 0.0], [1000, 0.1], [3000, 0.25]],
        },
        "parent_projection": {
            "enable": False,
            "branches": ["bg", "distant"],
            "forward_mode": "incremental_runtime",
            "backward_mode": "exact_diag_recompute_surrogate_vjp",
            "alpha_schedule": [[0, 0.0], [3000, 0.05]],
            "drift": {
                "check_interval": 500,
                "warn_threshold": 1.0e-3,
                "skip_vjp_threshold": 5.0e-3,
                "exact_refresh_threshold": 1.0e-2,
            },
        },
        "relation": {
            "enable": False,
            "branches": ["bg", "distant"],
            "differentiable_diag_cov": True,
            "checkpoint": True,
            "grad_to_child_geometry": True,
            "grad_to_parent_geometry": True,
            "grad_to_child_code": False,
            "grad_to_parent_event": True,
            "grad_to_support": False,
            "alpha_schedule": [[0, 0.0], [3000, 0.05]],
        },
        "scalar_anchor": {"geometry_grad": False},
        "discrete_routing_grad": False,
        "rollout_boundary_grad": False,
        "debug": {
            "grad_probe_interval": 500,
            "forward_parity_interval": 1000,
            "log_feedback_memory": True,
        },
    }


def test_feedback_alpha_schedule_interpolates_and_clamps_endpoints():
    schedule = FeedbackAlphaSchedule.from_config([[100, 0.1], [200, 0.3], [400, 0.9]])

    assert schedule(0) == pytest.approx(0.1)
    assert schedule(150) == pytest.approx(0.2)
    assert schedule(300) == pytest.approx(0.6)
    assert schedule(1000) == pytest.approx(0.9)


@pytest.mark.parametrize(
    "points,match",
    [
        ([], "at least one"),
        ([[0, 0.0], [0, 0.5]], "strictly increasing"),
        ([[0, -0.1]], "in \\[0, 1\\]"),
        ([[0, 1.1]], "in \\[0, 1\\]"),
    ],
)
def test_feedback_alpha_schedule_rejects_invalid_points(points, match):
    with pytest.raises(ValueError, match=match):
        FeedbackAlphaSchedule.from_config(points)


def test_scale_feedback_preserves_forward_and_scales_only_gradient():
    source = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64, requires_grad=True)
    output = scale_feedback(source, 0.3)

    assert torch.equal(output.detach(), source.detach())
    output.sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 0.3))


def test_observation_feedback_policy_strict_parse_and_scheduler_match():
    cfg = _feedback_cfg()
    policy = ObservationFeedbackPolicy.from_config(cfg)

    assert policy.enable is True
    assert policy.mode_for("repeat_refine") is FeedbackMode.TRAINABLE_CHECKPOINTED
    assert policy.mode_for("high_block_repair") is FeedbackMode.FROZEN_INPUT_GRAD_CHECKPOINTED
    assert policy.mode_for_visit(
        {"distribution_type": "repeat_refine", "train_2d_mode": "trainable_checkpointed"}
    ) is FeedbackMode.TRAINABLE_CHECKPOINTED
    assert policy.alpha_for("source_render", 500) == pytest.approx(0.05)
    assert policy.source_alpha(500) == pytest.approx(0.05)
    assert policy.parent_alpha(3000) == 0.0
    assert policy.relation_alpha(3000) == 0.0
    assert policy.schedule_step(29999) == 0
    assert policy.schedule_step(30000) == 0
    assert policy.schedule_step(30500) == 500
    assert policy.alpha_for("parent_projection", 3000) == 0.0
    policy.validate_scheduler_modes(cfg["modes"])

    with pytest.raises(ValueError, match="exactly match"):
        policy.validate_scheduler_modes({**cfg["modes"], "high_block_repair": "frozen_no_grad"})


def test_observation_feedback_policy_uses_explicit_read_only_evaluation_mode():
    policy = ObservationFeedbackPolicy.from_config(_feedback_cfg())
    mode = policy.mode_for_visit(
        {
            "distribution_type": "shuffled_coverage",
            "train_2d_mode": "trainable_checkpointed",
            "observation_feedback_eval_mode": "frozen_no_grad",
        }
    )
    assert mode is FeedbackMode.FROZEN_NO_GRAD

    with pytest.raises(ValueError, match="must be 'frozen_no_grad'"):
        policy.mode_for_visit({"observation_feedback_eval_mode": "trainable_checkpointed"})


def test_unrelated_partial_config_resolves_to_disabled_policy():
    policy = ObservationFeedbackPolicy.from_config(
        {"training": {"grad_clip": {"enable": True, "max_norm": 1.0}}}
    )
    assert policy.enable is False


def test_observation_feedback_policy_rejects_unknown_or_unsafe_configuration():
    cfg = _feedback_cfg()
    cfg["unexpected"] = True
    with pytest.raises(ValueError, match="unsupported keys"):
        ObservationFeedbackPolicy.from_config(cfg)

    cfg = _feedback_cfg()
    cfg["relation"]["enable"] = True
    with pytest.raises(ValueError, match="requires parent_projection.enable"):
        ObservationFeedbackPolicy.from_config(cfg)

    cfg = _feedback_cfg()
    cfg["modes"]["repeat_refine"] = "typo"
    with pytest.raises(ValueError, match="unsupported observation_feedback.modes.repeat_refine"):
        ObservationFeedbackPolicy.from_config(cfg)

    cfg = _feedback_cfg()
    cfg["schedule"]["origin"] = "resume_magic"
    with pytest.raises(ValueError, match="schedule.origin"):
        ObservationFeedbackPolicy.from_config(cfg)


def test_stage3_3_config_uses_safe_feedback_defaults_and_matching_scheduler_modes():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "iforward"
        / "iforward_stage3_3_observation_feedback.yaml"
    )
    cfg = OmegaConf.load(config_path)
    policy = ObservationFeedbackPolicy.from_config(cfg)

    policy.validate_scheduler_modes(dict(cfg.scheduler_stage3_2.episode_recipe.train_2d_policy))
    assert cfg.output_name == "iforward_stage3_3_observation_feedback"
    assert cfg.model.iforward.training_variant == "stage3_3_observation_feedback"
    assert policy.schedule.origin == "activation_step"
    assert policy.schedule.activation_step == 30000
    assert cfg.scheduler_stage3_2.max_inner_k_hard_cap == 15
    assert policy.source_render.enable is True
    assert policy.parent_projection.enable is False
    assert policy.relation.enable is False
    assert cfg.model.iforward.repair_training.enable is False
    assert cfg.model.iforward.repair_training.freeze_2d_frontend is False
    assert cfg.model.iforward.repair_training.no_grad_2d_forward is False
    assert cfg.model.stage6_0.base_measurement.detach_source_render_for_cnn is False
    assert cfg.model.stage6_0.local_rollout.local_G_no_detach_between_steps is True
    assert cfg.model.stage6_0.local_rollout.detach_persistent_state_at_block_start is False


def test_frontend_parameter_scope_freezes_and_restores_on_exception():
    module = nn.Linear(3, 2)
    module.bias.requires_grad_(False)
    original = [parameter.requires_grad for parameter in module.parameters()]

    with pytest.raises(RuntimeError, match="sentinel"):
        with FrontendParameterModeScope(
            module,
            ["weight", "bias"],
            FeedbackMode.FROZEN_INPUT_GRAD_CHECKPOINTED,
        ):
            assert all(not parameter.requires_grad for parameter in module.parameters())
            raise RuntimeError("sentinel")

    assert [parameter.requires_grad for parameter in module.parameters()] == original


def test_frontend_parameter_scope_keeps_trainable_mode_unchanged():
    module = nn.Linear(3, 2)
    original = [parameter.requires_grad for parameter in module.parameters()]
    with FrontendParameterModeScope(module, ["weight", "bias"], FeedbackMode.TRAINABLE_CHECKPOINTED):
        assert [parameter.requires_grad for parameter in module.parameters()] == original
    assert [parameter.requires_grad for parameter in module.parameters()] == original


def test_frozen_then_trainable_transactions_restore_frontend_updates():
    torch.manual_seed(3)
    module = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    names = ["weight", "bias"]
    initial = module.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    frozen_input = torch.ones(2, 3, requires_grad=True)
    with FrontendParameterModeScope(
        module,
        names,
        FeedbackMode.FROZEN_INPUT_GRAD_CHECKPOINTED,
    ):
        module(frozen_input).sum().backward()
        optimizer.step()
        assert module.weight.grad is None
        assert frozen_input.grad is not None
    torch.testing.assert_close(module.weight, initial)

    optimizer.zero_grad(set_to_none=True)
    with FrontendParameterModeScope(module, names, FeedbackMode.TRAINABLE_CHECKPOINTED):
        module(torch.ones(2, 3)).sum().backward()
        optimizer.step()
    assert not torch.equal(module.weight.detach(), initial)
    assert all(parameter.requires_grad for parameter in module.parameters())


def test_trainer_resolves_production_nested_stage3_2_request_metadata():
    nested = {
        "iforward_stage3_2": {
            "distribution_type": "high_block_repair",
            "train_2d_mode": "frozen_input_grad_checkpointed",
        }
    }
    batch = {
        "request_meta": {"assembly_mode": "image_ref_iforward_v1"},
        "_iforward": {"request_meta": nested},
    }
    assert IForwardTrainer._observation_feedback_request_meta(batch) == nested


def test_trainer_rejects_dynamic_frontend_freeze_in_distributed_execution(monkeypatch):
    trainer = IForwardTrainer.__new__(IForwardTrainer)
    trainer.observation_feedback_policy = ObservationFeedbackPolicy.from_config(_feedback_cfg())
    monkeypatch.setattr(
        IForwardTrainer,
        "_distributed_training_active",
        staticmethod(lambda: True),
    )
    with pytest.raises(RuntimeError, match="distributed/DDP"):
        trainer._validate_observation_feedback_distributed_mode()


def _schedule_only_trainer() -> IForwardTrainer:
    trainer = IForwardTrainer.__new__(IForwardTrainer)
    trainer.config = {
        "model": {
            "iforward": {
                "training_variant": "stage3_3_observation_feedback",
                "observation_feedback": _feedback_cfg(),
            }
        }
    }
    trainer.observation_feedback_policy = ObservationFeedbackPolicy.from_config(_feedback_cfg())
    trainer._feedback_activation_global_step = 30000
    trainer._feedback_schedule_step = 0
    trainer._feedback_schedule_state_restored = False
    return trainer


def test_feedback_schedule_checkpoint_resume_and_stage32_migration_semantics():
    trainer = _schedule_only_trainer()
    payload = trainer.build_light_checkpoint_extra(step=31000)
    assert payload["training_variant"] == "stage3_3_observation_feedback"
    assert payload["observation_feedback_schedule"] == {
        "format": "iforward_observation_feedback_schedule_v1",
        "origin": "activation_step",
        "feedback_activation_global_step": 30000,
        "feedback_schedule_step": 1000,
    }

    resumed = _schedule_only_trainer()
    resumed._feedback_activation_global_step = 999
    assert resumed.load_feedback_schedule_state_from_checkpoint(payload) is True
    assert resumed._feedback_activation_global_step == 30000
    assert resumed._feedback_schedule_step == 1000
    assert resumed._feedback_schedule_state_restored is True

    migrated = _schedule_only_trainer()
    assert migrated.load_feedback_schedule_state_from_checkpoint({"step": 29999}) is False
    assert migrated._feedback_activation_global_step == 30000
    assert migrated.observation_feedback_policy.schedule_step(
        30000, activation_step=migrated._feedback_activation_global_step
    ) == 0
