from __future__ import annotations

import pytest
import torch

from tools.iforward_validate_v4 import _load_model_weights


class _GuardedModule(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)
        self.guard_payload = None

    def validate_resume_checkpoint_payload(self, payload):
        self.guard_payload = payload
        raise ValueError("native resume guard rejected checkpoint")


def test_validate_v4_calls_native_resume_guard_before_loading_weights(tmp_path):
    model = _GuardedModule()
    checkpoint = tmp_path / "checkpoint.pt"
    payload = {
        "model_state_dict": {
            "weight": torch.full_like(model.weight, 7.0),
            "bias": torch.full_like(model.bias, 7.0),
        },
        "iforward_version": "stage3_4_functional_parentgs_lift",
        "training_variant": "stage3_4_functional_parentgs_lift",
        "parent_codec_schema": "old_13d_schema",
    }
    torch.save(payload, checkpoint)
    weight_before = model.weight.detach().clone()

    with pytest.raises(ValueError, match="native resume guard rejected checkpoint"):
        _load_model_weights(model, str(checkpoint))

    assert model.guard_payload is not None
    torch.testing.assert_close(model.weight, weight_before)


def test_validate_v4_requires_checkpoint_when_native_guard_rejects_empty_payload():
    model = _GuardedModule()

    with pytest.raises(ValueError, match="native resume guard rejected checkpoint"):
        _load_model_weights(model, "")

    assert model.guard_payload == {}
