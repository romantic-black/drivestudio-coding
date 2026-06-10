from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.iforward import IForwardBatchResolver, IForwardModel
from models.iforward.history_ema import IForwardHistoryEMAState
from models.iforward.history_gate import IForwardAttributeGate, IForwardGatePack, IForwardHistoryGate
from models.iforward.history_gate_v2_features import (
    HGV2_GRAD_FEATURE_DIM,
    HistoryGateV2AttrDamage,
    HistoryGateV2BranchFeatures,
    HistoryGateV2FeaturePack,
    compute_history_gate_v2_features,
    history_gate_v2_auxiliary_loss,
    induced_quat_delta,
)
from models.iforward.history_gradient_bank import build_history_gradient_bank_from_loss
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import ContextPack, EventPack, LocalGSState
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack

from test_iforward_v3_rollout import (
    FakeV3Bridge,
    _batch_b1r2,
    _batch_b1r2_with_history_probe_target,
    _v3_cfg,
)


def _node_state(n: int = 3) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _rigid_state(n: int = 5) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
        point_ids=torch.arange(n).view(n, 1),
        instances_quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        instances_trans=torch.zeros(1, 3),
        instances_fv=torch.zeros(1, 2, dtype=torch.long),
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )


def _local_state(n: int = 3, *, rigid: bool = False) -> LocalGSState:
    return LocalGSState.from_node_states(
        bg=_node_state(n),
        distant=None,
        rigid=_rigid_state(n) if rigid else None,
        hidden_dim=2,
    )


def _branch_delta(ref: torch.Tensor, n: int, *, means: torch.Tensor | None = None, quat: torch.Tensor | None = None) -> BranchDelta:
    return BranchDelta(
        means=ref.new_zeros(n, 3) if means is None else means,
        scales_log=ref.new_zeros(n, 3),
        quat_axis_angle=ref.new_zeros(n, 3) if quat is None else quat,
        opacity_logit=ref.new_zeros(n, 1),
        sh=ref.new_zeros(n, 12),
        hidden=ref.new_zeros(n, 2),
        confidence=ref.new_ones(n, 1),
        noop=ref.new_zeros(n, 1),
    )


def _hgv2_cfg() -> dict:
    return {
        "enable": True,
        "bank": {"dtype": "fp32", "min_grad_norm": 1.0e-8},
        "attrs": {"means": True, "scales": True, "quat": True, "opacity": True, "sh": True},
        "features": {"grad_embed_dim": 4, "grad_prior_scale_init": 0.0},
        "auxiliary_loss": {
            "enable": True,
            "weight": 1.0,
            "close_weight": 0.02,
            "safe_open_weight": 0.002,
            "tau_cos": 0.05,
            "tau_safe": 0.10,
            "detach_damage": True,
            "attr_weights": {"means": 1.0, "scales": 0.0, "quat": 0.0, "opacity": 0.0, "sh": 0.0},
        },
    }


def _zero_feature_pack(n: int, ref: torch.Tensor) -> HistoryGateV2FeaturePack:
    z = ref.new_zeros(n, 1)
    valid = torch.zeros(n, 1, dtype=torch.bool, device=ref.device)
    damage = {
        attr: HistoryGateV2AttrDamage(cos=z, pos=z, neg=z, log_norm=z, valid=valid)
        for attr in ("means", "scales", "quat", "opacity", "sh")
    }
    return HistoryGateV2FeaturePack(
        bg=HistoryGateV2BranchFeatures(features=ref.new_zeros(n, HGV2_GRAD_FEATURE_DIM), damage=damage)
    )


def test_history_gate_v2_zero_features_preserve_gate_output() -> None:
    local = _local_state(3)
    hist = IForwardHistoryEMAState.from_local_state(local)
    hist.bg.initialized[:] = 1.0
    gate = IForwardHistoryGate(
        event_dim=4,
        ctx_dim=3,
        hidden_dim=8,
        history_embed_dim=4,
        grad_feature_dim=HGV2_GRAD_FEATURE_DIM,
        grad_embed_dim=4,
        grad_prior_scale_init=0.0,
    )
    event = EventPack(
        event_bg=torch.zeros(3, 4),
        support_bg=torch.ones(3, 1),
        valid_bg=torch.ones(3, 1, dtype=torch.bool),
        obs_code_bg=torch.zeros(3, 2),
    )
    ctx = ContextPack(ctx_bg=torch.zeros(3, 3))

    no_features = gate(event=event, ctx_memory=ctx, history_ema=hist, local_state=local)
    zero_features = gate(
        event=event,
        ctx_memory=ctx,
        history_ema=hist,
        local_state=local,
        grad_features=_zero_feature_pack(3, event.event_bg),
    )

    assert torch.allclose(no_features.bg.means, zero_features.bg.means)
    assert torch.allclose(no_features.bg.sh, zero_features.bg.sh)


def _aux_feature_pack(cos_value: float) -> HistoryGateV2FeaturePack:
    cos = torch.full((2, 1), float(cos_value))
    pos = torch.relu(cos)
    neg = torch.relu(-cos)
    valid = torch.ones(2, 1, dtype=torch.bool)
    z = torch.zeros(2, 1)
    damage = {
        "means": HistoryGateV2AttrDamage(cos=cos, pos=pos, neg=neg, log_norm=z, valid=valid),
        "scales": HistoryGateV2AttrDamage(cos=z, pos=z, neg=z, log_norm=z, valid=valid),
        "quat": HistoryGateV2AttrDamage(cos=z, pos=z, neg=z, log_norm=z, valid=valid),
        "opacity": HistoryGateV2AttrDamage(cos=z, pos=z, neg=z, log_norm=z, valid=valid),
        "sh": HistoryGateV2AttrDamage(cos=z, pos=z, neg=z, log_norm=z, valid=valid),
    }
    return HistoryGateV2FeaturePack(
        bg=HistoryGateV2BranchFeatures(features=torch.zeros(2, HGV2_GRAD_FEATURE_DIM), damage=damage)
    )


def test_history_gate_v2_harmful_aux_loss_pushes_gate_closed() -> None:
    means_gate = torch.full((2, 1), 0.5, requires_grad=True)
    gate = IForwardGatePack(
        bg=IForwardAttributeGate(
            means=means_gate,
            scales=torch.ones(2, 1),
            quat=torch.ones(2, 1),
            opacity=torch.ones(2, 1),
            sh=torch.ones(2, 1),
            hidden=torch.ones(2, 1),
        )
    )

    loss, _ = history_gate_v2_auxiliary_loss(gate=gate, features=_aux_feature_pack(0.9), cfg=_hgv2_cfg())
    loss.backward()

    assert means_gate.grad is not None
    assert means_gate.grad.mean().item() > 0.0


def test_history_gate_v2_safe_aux_loss_pushes_gate_open() -> None:
    means_gate = torch.full((2, 1), 0.5, requires_grad=True)
    gate = IForwardGatePack(
        bg=IForwardAttributeGate(
            means=means_gate,
            scales=torch.ones(2, 1),
            quat=torch.ones(2, 1),
            opacity=torch.ones(2, 1),
            sh=torch.ones(2, 1),
            hidden=torch.ones(2, 1),
        )
    )

    loss, _ = history_gate_v2_auxiliary_loss(gate=gate, features=_aux_feature_pack(-0.9), cfg=_hgv2_cfg())
    loss.backward()

    assert means_gate.grad is not None
    assert means_gate.grad.mean().item() < 0.0


def test_history_gate_v2_rigid_route_gather_and_quat_delta_are_finite() -> None:
    local = _local_state(5, rigid=True)
    weights = torch.arange(5, dtype=local.rigid.means.dtype)
    loss = (local.rigid.means[:, 0] * weights).sum()
    bank = build_history_gradient_bank_from_loss(
        loss_history=loss,
        final_local_state=local,
        rollout_id=0,
        history_num_refs=1,
        cfg=_hgv2_cfg(),
    )
    assert bank is not None
    route_rows = torch.tensor([3, 1], dtype=torch.long)
    event = EventPack(
        event_bg=torch.zeros(5, 4),
        event_rigid=torch.zeros(2, 4),
        route=SimpleNamespace(S=route_rows),
    )
    rigid_delta = _branch_delta(
        local.rigid.means,
        2,
        means=torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        quat=torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]]),
    )
    delta = DeltaPack(bg=_branch_delta(local.bg.means, 5), rigid=rigid_delta)

    features = compute_history_gate_v2_features(
        bank=bank,
        event=event,
        delta_event=delta,
        local_state=local,
        cfg=_hgv2_cfg(),
    )
    quat_delta = induced_quat_delta(branch=local.rigid, delta=rigid_delta, rows=route_rows)

    assert features is not None
    assert features.rigid is not None
    assert features.rigid.features.shape == (2, HGV2_GRAD_FEATURE_DIM)
    assert torch.allclose(features.rigid.damage["means"].cos, torch.ones(2, 1))
    assert torch.isfinite(quat_delta).all()
    assert quat_delta.shape == (2, 4)


def _hgv2_model_cfg() -> dict:
    cfg = _v3_cfg()
    cfg["model"]["iforward"]["history_gate_v2"] = _hgv2_cfg()
    cfg["model"]["iforward"]["loss"] = {
        "current": {"weight": 0.0},
        "nearby": {"weight": 0.0},
        "in_rollout_history": {"weight": 1.0},
        "short_window_history": {"weight": 0.0},
        "delta_regularization": {"weight": 0.0},
    }
    return cfg


def _non_reset_next_rollout(batch: dict, rollout_id: int) -> dict:
    out = dict(batch)
    ifwd = dict(batch["_iforward"])
    ifwd["reset_scene_state_before_rollout"] = False
    ifwd["rollout_id_global"] = int(rollout_id)
    ifwd["rollout_idx_in_episode"] = int(rollout_id)
    request_meta = dict(batch["request_meta"])
    request_meta["iforward"] = ifwd
    out["_iforward"] = ifwd
    out["request_meta"] = request_meta
    return out


def test_iforward_hgv2_two_rollout_bank_create_consume_and_reset() -> None:
    bridge = FakeV3Bridge()
    model = IForwardModel(
        config=_hgv2_model_cfg(),
        device=torch.device("cpu"),
        bridge=bridge,
        resolver=IForwardBatchResolver(),
    )

    out0 = model.forward_rollout(_batch_b1r2_with_history_probe_target())
    assert out0.next_state.history_gradient_bank is not None
    assert out0.stats["hgv2/bank_valid"] == pytest.approx(0.0)
    assert out0.stats["hgv2/next_bank_valid"] == pytest.approx(1.0)

    carried = out0.next_state.detach_for_next_rollout()
    out1 = model.forward_rollout(
        _non_reset_next_rollout(_batch_b1r2_with_history_probe_target(), 1),
        carried_state=carried,
    )
    assert out1.stats["hgv2/bank_valid"] == pytest.approx(1.0)
    assert out1.stats["hgv2/bank_rollout_gap"] == pytest.approx(1.0)
    assert "hgv2/damage_pos_ratio/means" in out1.per_step[0]
    assert out1.losses["hgv2_gate"].requires_grad

    out_reset = model.forward_rollout(_batch_b1r2(), carried_state=out1.next_state.detach_for_next_rollout())
    assert out_reset.stats["hgv2/bank_valid"] == pytest.approx(0.0)
    assert out_reset.next_state.history_gradient_bank is None
