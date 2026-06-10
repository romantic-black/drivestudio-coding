from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch

from models.iforward.history_safe_projection import (
    IForwardHistorySafeProjection,
    HistoryGradBranch,
    HistoryGradPack,
    grad_pack_to_event_rows,
    grad_state_to_grad_pack,
    make_probe_local_state,
    project_attr,
    project_delta_pack,
    select_history_probe_indices,
)
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import EventPack, LocalGSState
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


def _node_state(n: int = 2) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _local_state(n: int = 2, *, rigid: bool = False) -> LocalGSState:
    bg = _node_state(n)
    rigid_state = None
    if rigid:
        rigid_state = NodeStateRigid(
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
    return LocalGSState.from_node_states(bg=bg, distant=None, rigid=rigid_state, hidden_dim=2)


def _branch_delta(ref: torch.Tensor, means: torch.Tensor, *, sh_dim: int = 12) -> BranchDelta:
    n = int(means.shape[0])
    return BranchDelta(
        means=means,
        scales_log=ref.new_zeros(n, 3),
        quat_axis_angle=ref.new_zeros(n, 3),
        opacity_logit=ref.new_zeros(n, 1),
        sh=ref.new_zeros(n, sh_dim),
        hidden=ref.new_zeros(n, 2),
        confidence=ref.new_ones(n, 1),
        noop=ref.new_zeros(n, 1),
    )


def _delta_pack(local_state: LocalGSState, *, means: torch.Tensor | None = None) -> DeltaPack:
    ref = local_state.bg.means
    bg_means = means if means is not None else ref.new_zeros(int(ref.shape[0]), 3)
    rigid = None
    if local_state.rigid is not None:
        rigid = _branch_delta(ref, ref.new_zeros(2, 3))
    return DeltaPack(bg=_branch_delta(ref, bg_means), rigid=rigid)


def _hsp_cfg(mode: str = "damage_loss_only") -> Dict[str, Any]:
    return {
        "enable": True,
        "mode": mode,
        "probe": {"frames_per_block": 1, "cams_per_frame": 1, "reuse_within_block": True},
        "attrs": {"means": True, "scales": False, "opacity": False, "sh": False, "quat": False},
        "projection": {
            "strength": {"start_step": 0, "warmup_steps": 0, "start_value": 1.0, "end_value": 1.0},
            "attr_strength_scale": {"means": 1.0},
            "tau_norm": {"means": 0.0},
            "eps": 1.0e-8,
        },
        "damage_loss": {"enable": True, "type": "cosine_conflict", "weight": 0.05, "attr_weights": {"means": 1.0}},
    }


class _ProbeBridge:
    def __init__(self) -> None:
        self.calls = 0

    def history_probe_loss(self, *, local_state, batch, target_indices, mask_policy):
        _ = batch, target_indices, mask_policy
        self.calls += 1
        return local_state.bg.means[:, 0].sum(), {"psnr": 12.5}


class _RigidWeightedProbeBridge:
    def __init__(self) -> None:
        self.calls = 0

    def history_probe_loss(self, *, local_state, batch, target_indices, mask_policy):
        _ = batch, target_indices, mask_policy
        self.calls += 1
        if local_state.rigid is None:
            return local_state.bg.means.sum() * 0.0, {"psnr": 10.0}
        weights = torch.arange(
            int(local_state.rigid.means.shape[0]),
            device=local_state.rigid.means.device,
            dtype=local_state.rigid.means.dtype,
        )
        return (local_state.rigid.means[:, 0] * weights).sum(), {"psnr": 10.0}


def test_select_history_probe_uses_visit_order_not_source_frame_filter() -> None:
    resolved = SimpleNamespace(
        history_rollout_target_indices=(0, 1, 2, 3),
        target_refs=((120, 0), (120, 1), (80, 0), (130, 0)),
        window_hash=7,
    )
    step = SimpleNamespace(source_frame_idx=90, block_id=0, step_idx=0)

    selected = select_history_probe_indices(resolved=resolved, step=step, frames_per_block=1, cams_per_frame=1)

    assert selected == (3,)


def test_project_attr_removes_harmful_component() -> None:
    d = torch.tensor([[1.0, 1.0]])
    g = torch.tensor([[1.0, 0.0]])
    safe, loss, stats = project_attr(d, g, strength=1.0, tau_norm=0.0, eps=1.0e-8)

    assert torch.allclose(safe, torch.tensor([[0.0, 1.0]]), atol=1.0e-6)
    assert loss.item() > 0.0
    assert stats["damage_pos_ratio"] == pytest.approx(1.0)
    assert stats["cos_damage_loss"] == pytest.approx(0.5)


def test_project_attr_cosine_conflict_loss_is_dimensionless() -> None:
    harmful = torch.tensor([[2.0, 0.0]])
    orthogonal = torch.tensor([[0.0, 2.0]])
    safe_dir = torch.tensor([[-2.0, 0.0]])
    grad = torch.tensor([[3.0, 0.0]])

    _safe, harmful_loss, harmful_stats = project_attr(harmful, grad, strength=0.0, tau_cos=0.0)
    _orth_safe, orth_loss, orth_stats = project_attr(orthogonal, grad, strength=0.0, tau_cos=0.0)
    _safe_dir, safe_loss, safe_stats = project_attr(safe_dir, grad, strength=0.0, tau_cos=0.0)

    assert harmful_loss.item() == pytest.approx(1.0)
    assert harmful_stats["cos_damage_pos_ratio"] == pytest.approx(1.0)
    assert orth_loss.item() == pytest.approx(0.0)
    assert orth_stats["cos_damage_pos_ratio"] == pytest.approx(0.0)
    assert safe_loss.item() == pytest.approx(0.0)
    assert safe_stats["cos_damage_pos_ratio"] == pytest.approx(0.0)


def test_project_attr_leaves_safe_direction_unchanged() -> None:
    d = torch.tensor([[-1.0, 1.0]])
    g = torch.tensor([[1.0, 0.0]])
    safe, loss, stats = project_attr(d, g, strength=1.0, tau_norm=0.0, eps=1.0e-8)

    assert torch.allclose(safe, d)
    assert loss.item() == pytest.approx(0.0)
    assert stats["projection_norm_ratio"] == pytest.approx(0.0)


def test_project_delta_pack_preserves_attr_shapes() -> None:
    local = _local_state(3)
    delta = _delta_pack(local, means=torch.ones(3, 3))
    delta.bg.scales_log = torch.ones(3, 3)
    delta.bg.opacity_logit = torch.ones(3, 1)
    delta.bg.sh = torch.ones(3, 12)
    grad = HistoryGradPack(
        bg=HistoryGradBranch(
            means=torch.ones(3, 3),
            scales_log=torch.ones(3, 3),
            opacity_logit=torch.ones(3, 1),
            sh=torch.ones(3, 12),
        )
    )

    safe, loss, aux = project_delta_pack(
        delta,
        grad,
        attrs={"means": True, "scales": True, "opacity": True, "sh": True},
        strength_by_attr={"means": 1.0, "scales": 1.0, "opacity": 1.0, "sh": 1.0},
        tau_norm_by_attr={"means": 0.0, "scales": 0.0, "opacity": 0.0, "sh": 0.0},
        attr_weights={"means": 1.0, "scales": 0.5, "opacity": 0.7, "sh": 0.7},
        mode="project_delta",
    )

    assert safe.bg.means.shape == delta.bg.means.shape
    assert safe.bg.scales_log.shape == delta.bg.scales_log.shape
    assert safe.bg.opacity_logit.shape == delta.bg.opacity_logit.shape
    assert safe.bg.sh.shape == delta.bg.sh.shape
    assert loss.item() > 0.0
    assert "hsp/projection_norm_ratio/sh" in aux


def test_zero_delta_attr_is_diagnostic_only_not_training_loss() -> None:
    local = _local_state(2)
    ref = local.bg.means
    delta = DeltaPack(
        bg=_branch_delta(ref, torch.ones(2, 3)),
        distant=_branch_delta(ref, torch.zeros(2, 3)),
    )
    grad = HistoryGradPack(
        bg=HistoryGradBranch(means=torch.ones(2, 3)),
        distant=HistoryGradBranch(means=torch.ones(2, 3)),
    )

    _safe, loss, aux = project_delta_pack(
        delta,
        grad,
        attrs={"means": True, "scales": False, "opacity": False, "sh": False},
        strength_by_attr={"means": 0.0},
        tau_norm_by_attr={"means": 0.0},
        tau_cos_by_attr={"means": 0.0},
        attr_weights={"means": 1.0},
        mode="damage_loss_only",
        loss_type="cosine_conflict",
    )

    assert loss.item() == pytest.approx(1.0)
    assert aux["hsp/bg_active/means"] == pytest.approx(1.0)
    assert aux["hsp/distant_active/means"] == pytest.approx(0.0)
    assert aux["hsp/active_attr_count"] == pytest.approx(1.0)
    assert aux["hsp/nonzero_cos_damage_loss/means"] == pytest.approx(1.0)


def test_sh_grad_packing_matches_delta_shape() -> None:
    local = _local_state(2)
    probe, items = make_probe_local_state(local, attrs={"means": False, "scales": False, "opacity": False, "sh": True})
    grads = tuple(torch.ones_like(item[2]) for item in items)
    pack = grad_state_to_grad_pack(probe_state=probe, tensor_items=items, grads=grads, attrs={"sh": True})

    assert pack.bg.sh is not None
    assert pack.bg.sh.shape == (2, 12)


def test_rigid_route_gathers_grad_rows() -> None:
    local = _local_state(5, rigid=True)
    delta = _delta_pack(local)
    assert delta.rigid is not None
    event = EventPack(event_bg=torch.zeros(5, 4), route=SimpleNamespace(S=torch.tensor([2, 4])))
    grad_full = HistoryGradPack(
        bg=HistoryGradBranch(means=torch.zeros(5, 3)),
        rigid=HistoryGradBranch(means=torch.arange(15, dtype=torch.float32).view(5, 3)),
    )

    event_grad = grad_pack_to_event_rows(grad_full, event=event, delta_event=delta)

    assert event_grad.rigid is not None
    assert torch.equal(event_grad.rigid.means, grad_full.rigid.means[torch.tensor([2, 4])])


def test_no_history_fallback_identity_and_zero_loss() -> None:
    local = _local_state(2)
    delta = _delta_pack(local, means=torch.ones(2, 3))
    hsp = IForwardHistorySafeProjection(_hsp_cfg("damage_loss_only"))

    safe, aux, loss = hsp(
        local_state=local,
        event=EventPack(event_bg=torch.zeros(2, 4)),
        delta_event=delta,
        resolved=SimpleNamespace(history_rollout_target_indices=(), target_refs=()),
        batch={"targets": []},
        step=SimpleNamespace(source_frame_idx=0, repeat_idx=0, is_block_enter=True),
        step_context=SimpleNamespace(global_step=0),
        history_ema=None,
        bridge=_ProbeBridge(),
        probe_cache={},
    )

    assert safe is delta
    assert loss.item() == pytest.approx(0.0)
    assert aux["hsp/skipped_no_history"] == pytest.approx(1.0)


def test_damage_loss_only_keeps_delta_but_returns_positive_loss() -> None:
    local = _local_state(2)
    delta = _delta_pack(local, means=torch.ones(2, 3))
    hsp = IForwardHistorySafeProjection(_hsp_cfg("damage_loss_only"))

    safe, aux, loss = hsp(
        local_state=local,
        event=EventPack(event_bg=torch.zeros(2, 4)),
        delta_event=delta,
        resolved=SimpleNamespace(history_rollout_target_indices=(0,), target_refs=((0, 0),), window_hash=1),
        batch={"targets": [{"frame_idx": 0, "cam_idx": 0}]},
        step=SimpleNamespace(source_frame_idx=0, repeat_idx=0, is_block_enter=True, block_id=0, step_idx=0),
        step_context=SimpleNamespace(global_step=0),
        history_ema=None,
        bridge=_ProbeBridge(),
        probe_cache={},
    )

    assert torch.equal(safe.bg.means, delta.bg.means)
    assert loss.item() > 0.0
    assert aux["hsp/cos_damage_loss"] > 0.0
    assert aux["hsp/projection_strength"] == pytest.approx(0.0)


def test_project_delta_reduces_first_order_damage_and_keeps_delta_grad() -> None:
    local = _local_state(2)
    means = torch.ones(2, 3, requires_grad=True)
    delta = _delta_pack(local, means=means)
    hsp = IForwardHistorySafeProjection(_hsp_cfg("project_delta"))

    safe, aux, loss = hsp(
        local_state=local,
        event=EventPack(event_bg=torch.zeros(2, 4)),
        delta_event=delta,
        resolved=SimpleNamespace(history_rollout_target_indices=(0,), target_refs=((0, 0),), window_hash=1),
        batch={"targets": [{"frame_idx": 0, "cam_idx": 0}]},
        step=SimpleNamespace(source_frame_idx=0, repeat_idx=0, is_block_enter=True, block_id=0, step_idx=0),
        step_context=SimpleNamespace(global_step=0),
        history_ema=None,
        bridge=_ProbeBridge(),
        probe_cache={},
    )

    assert torch.allclose(safe.bg.means[:, 0], torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(safe.bg.means[:, 1:], torch.ones(2, 2), atol=1.0e-6)
    assert aux["hsp/projection_norm_ratio/means"] > 0.0
    (safe.bg.means.sum() + loss).backward()
    assert means.grad is not None
    assert torch.isfinite(means.grad).all()
    assert local.bg.means.grad is None


def test_cached_full_grad_regathers_rigid_rows_per_repeat() -> None:
    local = _local_state(5, rigid=True)
    hsp = IForwardHistorySafeProjection(_hsp_cfg("project_delta"))
    bridge = _RigidWeightedProbeBridge()
    cache: Dict[str, Any] = {}
    resolved = SimpleNamespace(history_rollout_target_indices=(0,), target_refs=((0, 0),), window_hash=1)
    bg_delta = _branch_delta(local.bg.means, torch.zeros(5, 3))

    delta_first = DeltaPack(
        bg=bg_delta,
        rigid=_branch_delta(local.bg.means, torch.zeros(2, 3)),
    )
    _safe_first, aux_first, _loss_first = hsp(
        local_state=local,
        event=EventPack(event_bg=torch.zeros(5, 4), route=SimpleNamespace(S=torch.tensor([0, 1]))),
        delta_event=delta_first,
        resolved=resolved,
        batch={"targets": [{"frame_idx": 0, "cam_idx": 0}]},
        step=SimpleNamespace(source_frame_idx=0, repeat_idx=0, is_block_enter=True, block_id=0, step_idx=0),
        step_context=SimpleNamespace(global_step=0),
        history_ema=None,
        bridge=bridge,
        probe_cache=cache,
    )

    delta_second = DeltaPack(
        bg=bg_delta,
        rigid=_branch_delta(local.bg.means, torch.tensor([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])),
    )
    safe_second, aux_second, loss_second = hsp(
        local_state=local,
        event=EventPack(event_bg=torch.zeros(5, 4), route=SimpleNamespace(S=torch.tensor([3, 4]))),
        delta_event=delta_second,
        resolved=resolved,
        batch={"targets": [{"frame_idx": 0, "cam_idx": 0}]},
        step=SimpleNamespace(source_frame_idx=0, repeat_idx=1, is_block_enter=False, block_id=0, step_idx=1),
        step_context=SimpleNamespace(global_step=0),
        history_ema=None,
        bridge=bridge,
        probe_cache=cache,
    )

    assert bridge.calls == 1
    assert aux_first["hsp/cache_hit"] == pytest.approx(0.0)
    assert aux_second["hsp/cache_hit"] == pytest.approx(1.0)
    assert safe_second.rigid is not None
    assert torch.allclose(safe_second.rigid.means[:, 0], torch.zeros(2), atol=1.0e-6)
    assert loss_second.item() == pytest.approx(1.0)
