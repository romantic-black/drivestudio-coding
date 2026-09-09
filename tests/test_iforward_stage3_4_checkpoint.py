from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from models.iforward.model import IForwardModel
from models.iforward.parent_spatial_backbone import (
    Stage34ParentGeometryResidualAdapter,
    Stage6ParentParamSupportCodec,
)
from models.iforward.trainer import IForwardTrainer
from models.iforward.versions import (
    IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
    IFORWARD_STAGE3_4_PARENT_CODEC_SCHEMA,
)


STAGE3_4_VARIANT = "stage3_4_functional_parentgs_lift"


class _FakeParentSpatialBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param_support_codec = Stage6ParentParamSupportCodec(
            output_dim=24,
            detach_params=True,
            detach_support=True,
        )
        self.geometry_residual_adapter = Stage34ParentGeometryResidualAdapter(output_dim=24)
        self.token_builder = nn.Module()
        self.token_builder.param_support_proj = nn.Linear(24, 64)
        self.ptv3 = nn.Linear(3, 3)


def _make_stage3_4_model_stub() -> IForwardModel:
    model = IForwardModel.__new__(IForwardModel)
    nn.Module.__init__(model)
    model.config = {
        "initialization": {"skip_keys": []},
        "model": {
            "iforward": {
                "parent_spatial": {
                    "param_codec": {"schema": IFORWARD_STAGE3_4_PARENT_CODEC_SCHEMA}
                }
            }
        },
    }
    model.device = torch.device("cpu")
    model.is_stage3_4_functional_parentgs = True
    model.parent_spatial_backbone = _FakeParentSpatialBackbone()
    model.measurement_frontend = nn.Linear(2, 2)
    model.parent_optimizer_gdkv = nn.Linear(2, 2)
    model.updater = nn.Linear(2, 2)
    return model


def _fill_checkpoint_tensor(tensor: torch.Tensor, value: float) -> torch.Tensor:
    if tensor.is_floating_point():
        return torch.full_like(tensor, float(value))
    return tensor.detach().clone()


def test_stage34_init_checkpoint_requires_weights_only() -> None:
    model = _make_stage3_4_model_stub()

    with pytest.raises(ValueError, match="Stage 3.3 -> Stage 3.4 initialization is weights-only"):
        model.load_init_checkpoint_payload(
            {"model_state_dict": model.state_dict()},
            weights_only=False,
            path="stage3_3.pt",
        )


def test_stage34_weights_only_migration_reinitializes_codec_and_preserves_compatible_weights(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _make_stage3_4_model_stub()
    original_adapter = {
        name: value.detach().clone()
        for name, value in model.parent_spatial_backbone.geometry_residual_adapter.state_dict().items()
    }

    # Real IForward training checkpoints store IForwardModel below the
    # IForwardTrainer's ``model.`` prefix.  Exercise that normalization path.
    checkpoint_state = {
        f"model.{name}": _fill_checkpoint_tensor(value, 7.0)
        for name, value in model.state_dict().items()
        if not name.startswith("parent_spatial_backbone.geometry_residual_adapter.")
    }

    with caplog.at_level(logging.INFO, logger="models.iforward.model"):
        loaded = model.load_init_checkpoint_payload(
            {"model_state_dict": checkpoint_state},
            weights_only=True,
            path="stage3_3.pt",
        )

    assert loaded is True
    for name, value in model.parent_spatial_backbone.geometry_residual_adapter.state_dict().items():
        torch.testing.assert_close(value, original_adapter[name])
    assert model.parent_spatial_backbone.geometry_residual_adapter.is_zero_initialized()

    for module in (
        model.parent_spatial_backbone.param_support_codec,
        model.parent_spatial_backbone.token_builder.param_support_proj,
    ):
        for value in module.state_dict().values():
            if value.is_floating_point():
                torch.testing.assert_close(value, torch.full_like(value, 7.0))

    # PTV3, GDKV, updater and frontend are representative compatible modules
    # that must survive the Stage 3.3 -> 3.4 weights-only migration.
    for module in (
        model.parent_spatial_backbone.ptv3,
        model.measurement_frontend,
        model.parent_optimizer_gdkv,
        model.updater,
    ):
        for value in module.state_dict().values():
            if value.is_floating_point():
                torch.testing.assert_close(value, torch.full_like(value, 7.0))

    assert model.stage3_4_checkpoint_migration_stats == {
        "stage3_4/checkpoint/legacy_parent_codec_loaded": 1.0,
        "stage3_4/checkpoint/legacy_parent_codec_keys": 5.0,
        "stage3_4/checkpoint/parent_token_proj_loaded": 1.0,
        "stage3_4/checkpoint/parent_token_proj_keys": 2.0,
        "stage3_4/checkpoint/geometry_residual_zero_initialized": 1.0,
        "stage3_4/checkpoint/unexpected_runtime_vjp_keys": 0.0,
    }
    assert "stage3_4/checkpoint/legacy_parent_codec_loaded=1" in caplog.text
    assert "stage3_4/checkpoint/legacy_parent_codec_keys=5" in caplog.text
    assert "stage3_4/checkpoint/parent_token_proj_keys=2" in caplog.text
    assert "stage3_4/checkpoint/geometry_residual_zero_initialized=1" in caplog.text
    assert "unexpected_runtime_vjp_keys=0" in caplog.text


def test_stage34_weights_only_migration_rejects_pre_v57_stage34_checkpoint() -> None:
    model = _make_stage3_4_model_stub()
    with pytest.raises(ValueError, match="Pre-v57 Stage 3.4 checkpoints"):
        model.load_init_checkpoint_payload(
            {
                "iforward_version": IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
                "training_variant": STAGE3_4_VARIANT,
                "model_state_dict": model.state_dict(),
            },
            weights_only=True,
            path="stage3_4_pre_v57.pt",
        )


def test_stage34_weights_only_migration_requires_legacy_token_projection() -> None:
    model = _make_stage3_4_model_stub()
    checkpoint_state = {
        f"model.{name}": value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("parent_spatial_backbone.geometry_residual_adapter.")
        and not name.startswith("parent_spatial_backbone.token_builder.param_support_proj.")
    }
    with pytest.raises(ValueError, match="downstream token projection"):
        model.load_init_checkpoint_payload(
            {"model_state_dict": checkpoint_state},
            weights_only=True,
            path="broken_stage3_3.pt",
        )


def _make_trainer_resume_guard(
    *,
    version: str = IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
    variant: str = STAGE3_4_VARIANT,
) -> IForwardTrainer:
    trainer = IForwardTrainer.__new__(IForwardTrainer)
    nn.Module.__init__(trainer)
    trainer.config = {
        "model": {
            "iforward": {
                "version": version,
                "training_variant": variant,
                "parent_spatial": {
                    "param_codec": {"schema": IFORWARD_STAGE3_4_PARENT_CODEC_SCHEMA}
                },
            }
        }
    }
    return trainer


def test_stage34_native_resume_accepts_exact_version_and_variant() -> None:
    trainer = _make_trainer_resume_guard()

    trainer.validate_resume_checkpoint_payload(
        {
            "iforward_version": IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
            "training_variant": STAGE3_4_VARIANT,
            "parent_codec_schema": IFORWARD_STAGE3_4_PARENT_CODEC_SCHEMA,
        }
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "requires checkpoint iforward_version metadata"),
        (
            {
                "iforward_version": "stage3_0_scalar_anchor_child_support_parent_legacy",
                "training_variant": "stage3_3_observation_feedback",
            },
            "Cross-version strict resume is forbidden",
        ),
        (
            {
                "iforward_version": IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
                "training_variant": "stage3_4_wrong_variant",
                "parent_codec_schema": IFORWARD_STAGE3_4_PARENT_CODEC_SCHEMA,
            },
            "training_variant mismatch",
        ),
        (
            {
                "iforward_version": IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
                "training_variant": STAGE3_4_VARIANT,
            },
            "requires checkpoint parent_codec_schema",
        ),
        (
            {
                "iforward_version": IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
                "training_variant": STAGE3_4_VARIANT,
                "parent_codec_schema": "geometry_only_13d_v1",
            },
            "parent_codec_schema mismatch",
        ),
    ],
)
def test_stage34_native_resume_rejects_missing_or_cross_version_metadata(
    payload: dict[str, str],
    message: str,
) -> None:
    trainer = _make_trainer_resume_guard()

    with pytest.raises(ValueError, match=message):
        trainer.validate_resume_checkpoint_payload(payload)


def test_stage34_checkpoint_cannot_strict_resume_into_legacy_version() -> None:
    trainer = _make_trainer_resume_guard(
        version="stage3_0_scalar_anchor_child_support_parent_legacy",
        variant="stage3_3_observation_feedback",
    )

    with pytest.raises(ValueError, match="Cross-version strict resume is forbidden"):
        trainer.validate_resume_checkpoint_payload(
            {
                "iforward_version": IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
                "training_variant": STAGE3_4_VARIANT,
                "parent_codec_schema": IFORWARD_STAGE3_4_PARENT_CODEC_SCHEMA,
            }
        )


def test_legacy_resume_keeps_pre_stage34_metadata_compatibility() -> None:
    trainer = _make_trainer_resume_guard(
        version="stage3_0_scalar_anchor_child_support_parent_legacy",
        variant="stage3_3_observation_feedback",
    )

    trainer.validate_resume_checkpoint_payload({})
