from __future__ import annotations

import torch

from models.iforward.delta_ops import gate_branch_delta
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.stage6_0.event_encoder import EventPack
from models.streetforward.stage6_0.posterior_updater import (
    AppearanceDetailPack,
    AppearanceLogvarStatePack,
    Stage6PosteriorUpdater,
)


def _updater() -> Stage6PosteriorUpdater:
    return Stage6PosteriorUpdater(
        event_dim=4,
        hidden_dim=8,
        stage_hidden_dim=0,
        sh_degree=1,
        output_hidden=False,
        output_confidence=False,
        output_noop=True,
        output_appearance_logvar_delta=True,
        appearance_logvar_detach_input=True,
        appearance_logvar_gate_by_valid=True,
        appearance_logvar_gate_by_main_noop=False,
        appearance_logvar_max_step={"bg": 0.08},
    )


def _target_updater() -> Stage6PosteriorUpdater:
    return Stage6PosteriorUpdater(
        event_dim=4,
        hidden_dim=8,
        stage_hidden_dim=0,
        sh_degree=1,
        output_hidden=False,
        output_confidence=False,
        output_noop=True,
        output_appearance_logvar_delta=True,
        appearance_logvar_update_mode="state_conditioned_target_v2",
        appearance_logvar_target_temperature=1.0,
        appearance_logvar_state_cfg={
            "init_sigma": {"bg": 0.08, "distant": 0.12, "rigid": 0.10},
            "sigma_min": 0.01,
            "sigma_max": 0.50,
        },
        appearance_logvar_max_step={"bg": 0.08},
    )


def test_uncertainty_head_is_zero_initialized() -> None:
    updater = _updater()
    delta, _ = updater(event=EventPack(event_bg=torch.randn(3, 4)))
    assert torch.equal(delta.bg.appearance_logvar_delta, torch.zeros(3, 1))


def test_uncertainty_delta_is_bounded_validity_gated_and_noop_independent() -> None:
    updater = _updater()
    updater.head_appearance_logvar_delta.bias.data.fill_(20.0)
    updater.head_noop.bias.data.fill_(20.0)
    detail = AppearanceDetailPack(
        detail_bg=torch.zeros(3, 1),
        valid_bg=torch.tensor([True, False, True]),
    )
    delta, _ = updater(
        event=EventPack(event_bg=torch.randn(3, 4), valid_bg=torch.ones(3, dtype=torch.bool)),
        appearance_detail=detail,
    )
    assert delta.bg.noop.min().item() > 0.99
    assert delta.bg.appearance_logvar_delta[0].item() > 0.079
    assert delta.bg.appearance_logvar_delta[1].item() == 0.0
    assert delta.bg.appearance_logvar_delta.abs().max().item() <= 0.080001


def test_uncertainty_head_detaches_trunk_and_history_gate_does_not_scale_delta() -> None:
    updater = _updater()
    updater.head_appearance_logvar_delta.bias.data.fill_(1.0)
    event_bg = torch.randn(2, 4, requires_grad=True)
    delta, _ = updater(event=EventPack(event_bg=event_bg))
    delta.bg.appearance_logvar_delta.sum().backward()
    assert event_bg.grad is None or torch.equal(event_bg.grad, torch.zeros_like(event_bg))
    assert updater.head_appearance_logvar_delta.weight.grad is not None

    class Gate:
        means = scales = quat = opacity = sh = hidden = torch.zeros(2, 1)

    gated = gate_branch_delta(delta.bg, Gate())
    assert torch.equal(gated.appearance_logvar_delta, delta.bg.appearance_logvar_delta)


def test_uncertainty_disabled_ablation_keeps_head_but_disables_branch_update() -> None:
    updater = _updater()
    assert updater.head_appearance_logvar_delta is not None
    runtime = object.__new__(MinimalStreetForwardStage6_0)
    runtime.iforward_uncertainty_update_enabled = False
    scope = runtime._parse_stage6_branch_scope(
        {
            "branch_scope": {
                "bg": {"update_appearance_logvar": True},
                "distant": {"update_appearance_logvar": True},
                "rigid": {"update_appearance_logvar": True},
            }
        }
    )
    assert all(not scope[branch]["update_appearance_logvar"] for branch in ("bg", "distant", "rigid"))


def test_legacy_updater_weights_leave_missing_uncertainty_head_zero_initialized() -> None:
    legacy = Stage6PosteriorUpdater(
        event_dim=4,
        hidden_dim=8,
        stage_hidden_dim=0,
        sh_degree=1,
        output_hidden=False,
        output_confidence=False,
        output_noop=True,
        output_appearance_logvar_delta=False,
    )
    upgraded = _updater()
    missing, unexpected = upgraded.load_state_dict(legacy.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        "head_appearance_logvar_delta.weight",
        "head_appearance_logvar_delta.bias",
    }
    assert torch.equal(upgraded.head_appearance_logvar_delta.weight, torch.zeros_like(upgraded.head_appearance_logvar_delta.weight))
    assert torch.equal(upgraded.head_appearance_logvar_delta.bias, torch.zeros_like(upgraded.head_appearance_logvar_delta.bias))


def test_v2_target_head_zero_init_is_prior_fixed_point_and_state_conditioned() -> None:
    updater = _target_updater()
    prior = 2.0 * torch.log(torch.tensor(0.08))
    event = EventPack(event_bg=torch.randn(3, 4))
    state = AppearanceLogvarStatePack(
        bg=torch.stack([prior, prior + 0.5, prior - 0.5]).reshape(3, 1)
    )
    delta, aux = updater(event=event, appearance_logvar_state=state)
    assert updater.head_appearance_logvar_delta is None
    assert updater.head_appearance_logvar_target is not None
    assert abs(delta.bg.appearance_logvar_delta[0].item()) < 1.0e-7
    assert delta.bg.appearance_logvar_delta[1].item() < 0.0
    assert delta.bg.appearance_logvar_delta[2].item() > 0.0
    assert delta.bg.appearance_logvar_delta.abs().max().item() <= 0.080001
    assert aux["uncertainty/bg/delta_positive_ratio"] > 0.0
    assert aux["uncertainty/bg/delta_negative_ratio"] > 0.0


def test_v2_target_head_is_noop_independent_validity_gated_and_detaches_inputs() -> None:
    updater = _target_updater()
    updater.head_appearance_logvar_target.bias.data.fill_(2.0)
    updater.head_noop.bias.data.fill_(20.0)
    prior = 2.0 * torch.log(torch.tensor(0.08))
    event_bg = torch.randn(2, 4, requires_grad=True)
    detail = AppearanceDetailPack(
        detail_bg=torch.zeros(2, 1),
        valid_bg=torch.tensor([True, False]),
    )
    delta, _ = updater(
        event=EventPack(event_bg=event_bg),
        appearance_detail=detail,
        appearance_logvar_state=AppearanceLogvarStatePack(bg=torch.full((2, 1), prior)),
    )
    assert delta.bg.noop.min().item() > 0.99
    assert delta.bg.appearance_logvar_delta[0].item() > 0.0
    assert delta.bg.appearance_logvar_delta[1].item() == 0.0
    delta.bg.appearance_logvar_delta.sum().backward()
    assert event_bg.grad is None or torch.equal(event_bg.grad, torch.zeros_like(event_bg))
    assert updater.head_appearance_logvar_target.weight.grad is not None


def test_v1_to_v2_updater_warmstart_has_only_expected_head_key_changes() -> None:
    v1 = _updater()
    v2 = _target_updater()
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert set(missing) == {
        "head_appearance_logvar_target.weight",
        "head_appearance_logvar_target.bias",
    }
    assert set(unexpected) == {
        "head_appearance_logvar_delta.weight",
        "head_appearance_logvar_delta.bias",
    }
    assert torch.equal(
        v2.head_appearance_logvar_target.weight,
        torch.zeros_like(v2.head_appearance_logvar_target.weight),
    )
