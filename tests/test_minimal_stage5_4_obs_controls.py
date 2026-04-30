from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.streetforward.minimal_trainer_stage5_3 import FullRoutedGRUInputs, MinimalStreetForwardStage5_3
from models.streetforward.minimal_trainer_stage5_3_production import MinimalStreetForwardStage5_3_Production
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.minimal_trainer_stage5_4_production import MinimalStreetForwardStage5_4_Production


class _ConstEmbed:
    def __init__(self, out_dim: int, value: float = 1.0):
        self.out_dim = int(out_dim)
        self.value = float(value)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (int(obs.shape[0]), self.out_dim),
            self.value,
            dtype=obs.dtype,
            device=obs.device,
        )


def test_stage5_4_init_reads_obs_input_switches(monkeypatch):
    def _fake_super_init(self, config):
        del config
        self.stage5_2_feat_2d_channels = 6
        self.fused_in_dim = 10
        self.device = torch.device("cpu")

    monkeypatch.setattr(MinimalStreetForwardStage5_3, "_init_stage5_3_modules", _fake_super_init)

    trainer = MinimalStreetForwardStage5_4.__new__(MinimalStreetForwardStage5_4)
    torch.nn.Module.__init__(trainer)
    cfg = {
        "current_observation": {
            "enable": True,
            "dim": 2,
            "eps": 1.0e-6,
            "input_to_struct_decoder": False,
            "input_to_far_mlp": True,
            "input_to_gru": False,
            "input_to_history_gate": True,
        }
    }
    trainer._init_stage5_3_modules(cfg)

    assert trainer.stage5_4_input_to_struct_decoder is False
    assert trainer.stage5_4_input_to_far_mlp is True
    assert trainer.stage5_4_input_to_gru is False
    assert trainer.stage5_4_input_to_history_gate is True


def test_stage5_4_split_obs_code_fast_fail_on_none():
    trainer = MinimalStreetForwardStage5_4.__new__(MinimalStreetForwardStage5_4)
    trainer._stage5_4_obs_code_all = None
    with pytest.raises(RuntimeError, match="expected obs_code"):
        trainer._split_obs_code(
            num_bg=2,
            num_distant=1,
            num_rigid_s=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_stage5_4_split_obs_code_fast_fail_on_shape():
    trainer = MinimalStreetForwardStage5_4.__new__(MinimalStreetForwardStage5_4)
    trainer._stage5_4_obs_code_all = torch.zeros(4, 3)
    with pytest.raises(RuntimeError, match="must have shape \\[N,2\\]"):
        trainer._split_obs_code(
            num_bg=2,
            num_distant=1,
            num_rigid_s=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_stage5_4_split_obs_code_fast_fail_on_length_mismatch():
    trainer = MinimalStreetForwardStage5_4.__new__(MinimalStreetForwardStage5_4)
    trainer._stage5_4_obs_code_all = torch.zeros(3, 2)
    with pytest.raises(RuntimeError, match="length mismatch"):
        trainer._split_obs_code(
            num_bg=2,
            num_distant=1,
            num_rigid_s=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_stage5_4_split_obs_code_allows_empty_when_total_zero():
    trainer = MinimalStreetForwardStage5_4.__new__(MinimalStreetForwardStage5_4)
    trainer._stage5_4_obs_code_all = None
    obs_bg, obs_distant, obs_rigid = trainer._split_obs_code(
        num_bg=0,
        num_distant=0,
        num_rigid_s=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert tuple(obs_bg.shape) == (0, 2)
    assert obs_distant is None
    assert obs_rigid is None


def test_stage5_4_struct_and_far_flags_control_obs_injection(monkeypatch):
    def _fake_super_near(self, **kwargs):
        del self
        return kwargs

    def _fake_super_far(self, **kwargs):
        del self
        return kwargs

    monkeypatch.setattr(MinimalStreetForwardStage5_3, "_build_struct_decoder_input_near", _fake_super_near)
    monkeypatch.setattr(MinimalStreetForwardStage5_3, "_build_struct_decoder_input_far", _fake_super_far)

    trainer = MinimalStreetForwardStage5_4.__new__(MinimalStreetForwardStage5_4)
    trainer.current_obs_struct_embed = object()
    trainer.current_obs_far_embed = object()
    trainer._stage5_4_active_obs = {
        "obs_bg": torch.zeros(2, 2),
        "obs_distant": torch.zeros(1, 2),
        "obs_rigid": torch.zeros(2, 2),
    }

    monkeypatch.setattr(
        MinimalStreetForwardStage5_4,
        "_apply_obs_feat_add",
        staticmethod(lambda feat_2d, obs, embed: feat_2d + 2.0 if (feat_2d is not None and obs is not None) else feat_2d),
    )

    feat_bg = torch.zeros(2, 4)
    feat_distant = torch.zeros(1, 4)
    feat_rigid = torch.zeros(2, 4)

    trainer.stage5_4_input_to_struct_decoder = False
    near_out_off = trainer._build_struct_decoder_input_near(
        route=SimpleNamespace(),
        feat_2d_bg=feat_bg.clone(),
        feat_2d_rigid_S=feat_rigid.clone(),
    )
    assert torch.allclose(near_out_off["feat_2d_bg"], feat_bg)
    assert torch.allclose(near_out_off["feat_2d_rigid_S"], feat_rigid)

    trainer.stage5_4_input_to_struct_decoder = True
    near_out_on = trainer._build_struct_decoder_input_near(
        route=SimpleNamespace(),
        feat_2d_bg=feat_bg.clone(),
        feat_2d_rigid_S=feat_rigid.clone(),
    )
    assert torch.allclose(near_out_on["feat_2d_bg"], feat_bg + 2.0)
    assert torch.allclose(near_out_on["feat_2d_rigid_S"], feat_rigid + 2.0)

    trainer.stage5_4_input_to_far_mlp = False
    far_out_off = trainer._build_struct_decoder_input_far(
        feat_2d_distant=feat_distant.clone(),
        feat_2d_rigid_S=feat_rigid.clone(),
    )
    assert torch.allclose(far_out_off["feat_2d_distant"], feat_distant)
    assert torch.allclose(far_out_off["feat_2d_rigid_S"], feat_rigid)

    trainer.stage5_4_input_to_far_mlp = True
    far_out_on = trainer._build_struct_decoder_input_far(
        feat_2d_distant=feat_distant.clone(),
        feat_2d_rigid_S=feat_rigid.clone(),
    )
    assert torch.allclose(far_out_on["feat_2d_distant"], feat_distant + 2.0)
    assert torch.allclose(far_out_on["feat_2d_rigid_S"], feat_rigid + 2.0)


def test_stage5_4_history_gate_flag_controls_obs_injection(monkeypatch):
    captured = {"feat": None}

    def _fake_super_gate(self, *, feat, branch_id, **kwargs):
        del self, branch_id, kwargs
        captured["feat"] = feat
        return feat

    monkeypatch.setattr(MinimalStreetForwardStage5_3, "_compute_gate", _fake_super_gate)

    trainer = MinimalStreetForwardStage5_4.__new__(MinimalStreetForwardStage5_4)
    trainer.current_obs_gate_embed = _ConstEmbed(out_dim=4, value=1.0)
    trainer._stage5_4_active_obs = {
        "obs_bg": torch.zeros(2, 2),
        "obs_rigid_in": torch.zeros(1, 2),
        "obs_distant": torch.zeros(1, 2),
        "obs_rigid_out": torch.zeros(1, 2),
    }
    feat = torch.zeros(2, 4)

    trainer.stage5_4_input_to_history_gate = False
    out_off = trainer._compute_gate(feat=feat.clone(), branch_id=0)
    assert torch.allclose(out_off, feat)
    assert torch.allclose(captured["feat"], feat)

    trainer.stage5_4_input_to_history_gate = True
    out_on = trainer._compute_gate(feat=feat.clone(), branch_id=0)
    assert torch.allclose(out_on, feat + 1.0)
    assert torch.allclose(captured["feat"], feat + 1.0)


def test_stage5_4_gru_flag_controls_obs_injection(monkeypatch):
    def _fake_super_gru(self, **kwargs):
        del self, kwargs
        return FullRoutedGRUInputs(
            feat_bg_input=torch.zeros(2, 4),
            feat_distant_input=torch.zeros(1, 4),
            feat_rigid_in_input_all=torch.zeros(1, 4),
            feat_rigid_out_input_all=torch.zeros(1, 4),
            gate_bg=None,
            gate_distant=None,
            gate_rigid_in=None,
            gate_rigid_out=None,
            aux={},
        )

    monkeypatch.setattr(MinimalStreetForwardStage5_3, "_compute_full_routed_gru_inputs", _fake_super_gru)

    trainer = MinimalStreetForwardStage5_4.__new__(MinimalStreetForwardStage5_4)
    trainer.current_obs_gru_embed = _ConstEmbed(out_dim=4, value=1.0)
    trainer.stage5_4_input_to_struct_decoder = False
    trainer.stage5_4_input_to_far_mlp = False
    trainer.stage5_4_input_to_history_gate = False
    trainer._stage5_4_obs_code_all = torch.zeros(5, 2)

    kwargs = {
        "feat_2d_bg": torch.zeros(2, 3),
        "feat_2d_distant": torch.zeros(1, 3),
        "feat_2d_rigid_S": torch.zeros(2, 3),
        "route": SimpleNamespace(inside_mask_S=torch.tensor([True, False], dtype=torch.bool)),
    }

    trainer.stage5_4_input_to_gru = False
    out_off = trainer._compute_full_routed_gru_inputs(**kwargs)
    assert torch.allclose(out_off.feat_bg_input, torch.zeros(2, 4))
    assert torch.allclose(out_off.feat_distant_input, torch.zeros(1, 4))
    assert torch.allclose(out_off.feat_rigid_in_input_all, torch.zeros(1, 4))
    assert torch.allclose(out_off.feat_rigid_out_input_all, torch.zeros(1, 4))

    trainer.stage5_4_input_to_gru = True
    out_on = trainer._compute_full_routed_gru_inputs(**kwargs)
    assert torch.allclose(out_on.feat_bg_input, torch.ones(2, 4))
    assert torch.allclose(out_on.feat_distant_input, torch.ones(1, 4))
    assert torch.allclose(out_on.feat_rigid_in_input_all, torch.ones(1, 4))
    assert torch.allclose(out_on.feat_rigid_out_input_all, torch.ones(1, 4))


def test_stage5_4_production_requires_use_fused_v4_true(monkeypatch):
    monkeypatch.setattr(
        MinimalStreetForwardStage5_3_Production,
        "_validate_production_config",
        lambda self, config: None,
    )
    trainer = MinimalStreetForwardStage5_4_Production.__new__(MinimalStreetForwardStage5_4_Production)
    bad_cfg = {
        "model": {
            "stage": "5_4",
            "backprojector_version": "v4",
            "use_fused_cuda_backproject_v4": False,
        },
        "current_observation": {"enable": True},
    }
    with pytest.raises(ValueError, match="use_fused_cuda_backproject_v4=true"):
        trainer._validate_production_config(bad_cfg)

    good_cfg = {
        "model": {
            "stage": "5_4",
            "backprojector_version": "v4",
            "use_fused_cuda_backproject_v4": True,
        },
        "current_observation": {"enable": True},
    }
    trainer._validate_production_config(good_cfg)
