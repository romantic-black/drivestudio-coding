from __future__ import annotations

import math

import torch

from models.iforward.uncertainty_renderer import (
    evaluate_spherical_harmonics,
    render_detached_uncertainty_moments,
)


class WeightedRasterizer:
    def __init__(self, weights: torch.Tensor) -> None:
        self.weights = weights

    def __call__(self, **kwargs):
        features = kwargs["colors"]
        weights = self.weights.to(device=features.device, dtype=features.dtype)
        rendered = torch.einsum("chwn,cnd->chwd", weights, features)
        alpha = weights.sum(dim=-1, keepdim=True)
        return rendered, alpha, {}


def _params(logvar: torch.Tensor, coeffs: torch.Tensor) -> dict[str, torch.Tensor]:
    n = int(logvar.shape[0])
    return {
        "means_r": torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]], requires_grad=True)[:n],
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True).repeat(n, 1),
        "scales_r": torch.ones(n, 3, requires_grad=True),
        "opacities_r": torch.ones(n, requires_grad=True),
        "colors_r": coeffs,
        "appearance_logvar_r": logvar,
    }


def _rgb_for_params(params: dict[str, torch.Tensor], viewmats: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    campos = torch.linalg.inv(viewmats)[..., :3, 3]
    dirs = params["means_r"].detach().unsqueeze(0) - campos.unsqueeze(1)
    coeffs = params["colors_r"].detach().unsqueeze(0).expand(int(viewmats.shape[0]), -1, -1, -1)
    colors = torch.clamp_min(evaluate_spherical_harmonics(0, dirs, coeffs) + 0.5, 0.0)
    return torch.einsum("chwn,cnd->chwd", weights, colors)


def test_detached_moment_raster_matches_two_gaussian_reference_and_grad_routing() -> None:
    weights = torch.tensor([[[[0.4, 0.6]]]])
    logvar = torch.log(torch.tensor([[0.01], [0.04]], requires_grad=True))
    logvar.retain_grad()
    coeffs = torch.zeros(2, 1, 3, requires_grad=True)
    params = _params(logvar, coeffs)
    viewmats = torch.eye(4).unsqueeze(0)
    Ks = torch.eye(3).unsqueeze(0)
    rgb = _rgb_for_params(params, viewmats, weights)
    bundle = render_detached_uncertainty_moments(
        rasterizer=WeightedRasterizer(weights),
        render_params=params,
        viewmats=viewmats,
        Ks=Ks,
        width=1,
        height=1,
        rgb=rgb,
        alpha=torch.ones(1, 1, 1),
        sh_degree=0,
        background_sigma=0.0,
        variance_floor=1.0e-8,
        variance_max=1.0,
    )
    assert bundle.disagreement_variance.item() < 1.0e-7
    assert bundle.variance.item() == torch.tensor(0.4 * 0.01 + 0.6 * 0.04).item()
    bundle.variance.sum().backward()
    assert logvar.grad is not None and logvar.grad.abs().sum().item() > 0.0
    assert params["means_r"].grad is None
    assert params["opacities_r"].grad is None
    assert coeffs.grad is None


def test_color_disagreement_and_gaussian_variance_are_monotonic() -> None:
    weights = torch.tensor([[[[0.5, 0.5]]]])
    viewmats = torch.eye(4).unsqueeze(0)
    Ks = torch.eye(3).unsqueeze(0)

    def render(coeff_value: float, variance: float):
        coeffs = torch.zeros(2, 1, 3)
        coeffs[1] = coeff_value
        params = _params(torch.full((2, 1), math.log(variance)), coeffs)
        rgb = _rgb_for_params(params, viewmats, weights)
        return render_detached_uncertainty_moments(
            rasterizer=WeightedRasterizer(weights),
            render_params=params,
            viewmats=viewmats,
            Ks=Ks,
            width=1,
            height=1,
            rgb=rgb,
            alpha=torch.ones(1, 1, 1),
            sh_degree=0,
            background_sigma=0.0,
            variance_floor=1.0e-8,
            variance_max=1.0,
        )

    same = render(0.0, 0.01)
    different = render(1.0, 0.01)
    higher_variance = render(0.0, 0.04)
    assert different.disagreement_variance.item() > same.disagreement_variance.item()
    assert higher_variance.variance.item() > same.variance.item()


def test_grouped_and_single_view_uncertainty_are_consistent() -> None:
    weights = torch.tensor([[[[0.4, 0.6]]], [[[0.7, 0.3]]]])
    viewmats = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    viewmats[1, 0, 3] = 0.1
    Ks = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    params = _params(
        torch.log(torch.tensor([[0.01], [0.04]])),
        torch.tensor([[[0.0, 0.0, 0.0]], [[0.5, 0.2, 0.0]]]),
    )
    rgb = _rgb_for_params(params, viewmats, weights)
    grouped = render_detached_uncertainty_moments(
        rasterizer=WeightedRasterizer(weights),
        render_params=params,
        viewmats=viewmats,
        Ks=Ks,
        width=1,
        height=1,
        rgb=rgb,
        alpha=torch.ones(2, 1, 1),
        sh_degree=0,
        background_sigma=0.0,
        variance_floor=1.0e-8,
        variance_max=1.0,
    )
    for view_idx in range(2):
        single_weights = weights[view_idx : view_idx + 1]
        single = render_detached_uncertainty_moments(
            rasterizer=WeightedRasterizer(single_weights),
            render_params=params,
            viewmats=viewmats[view_idx : view_idx + 1],
            Ks=Ks[view_idx : view_idx + 1],
            width=1,
            height=1,
            rgb=rgb[view_idx],
            alpha=torch.ones(1, 1),
            sh_degree=0,
            background_sigma=0.0,
            variance_floor=1.0e-8,
            variance_max=1.0,
        ).select_view(0)
        assert torch.allclose(grouped.variance[view_idx], single.variance)
        assert torch.allclose(grouped.aleatoric_variance[view_idx], single.aleatoric_variance)
        assert torch.allclose(grouped.disagreement_variance[view_idx], single.disagreement_variance)


def test_v2_loss_variance_excludes_disagreement_and_background_on_alpha_edge() -> None:
    weights = torch.tensor([[[[0.5]]]])
    viewmats = torch.eye(4).unsqueeze(0)
    Ks = torch.eye(3).unsqueeze(0)
    logvar = torch.full((1, 1), math.log(0.01), requires_grad=True)
    coeffs = torch.full((1, 1, 3), 2.0, requires_grad=True)
    params = _params(logvar, coeffs)
    rgb = _rgb_for_params(params, viewmats, weights)
    alpha = torch.full((1, 1, 1), 0.5, requires_grad=True)
    bundle = render_detached_uncertainty_moments(
        rasterizer=WeightedRasterizer(weights),
        render_params=params,
        viewmats=viewmats,
        Ks=Ks,
        width=1,
        height=1,
        rgb=rgb,
        alpha=alpha,
        sh_degree=0,
        background_sigma=0.10,
        background_sigma_for_loss=0.0,
        variance_floor=1.0e-4,
        variance_max=0.25,
        variance_mode="aleatoric_only",
        detach_first_pass_alpha=True,
    )
    assert bundle.disagreement_variance.item() > 0.20
    assert bundle.total_variance.item() == 0.25
    assert abs(bundle.loss_variance.item() - 0.005) < 1.0e-6
    assert abs(bundle.background_variance.item() - 0.005) < 1.0e-6
    bundle.loss_variance.sum().backward()
    assert logvar.grad is not None and logvar.grad.abs().sum().item() > 0.0
    assert alpha.grad is None
    assert params["means_r"].grad is None
    assert params["opacities_r"].grad is None
    assert coeffs.grad is None


def test_v2_color_disagreement_does_not_change_loss_precision() -> None:
    weights = torch.tensor([[[[0.5, 0.5]]]])
    viewmats = torch.eye(4).unsqueeze(0)
    Ks = torch.eye(3).unsqueeze(0)

    def render(coeffs: torch.Tensor):
        params = _params(torch.full((2, 1), math.log(0.02)), coeffs)
        return render_detached_uncertainty_moments(
            rasterizer=WeightedRasterizer(weights),
            render_params=params,
            viewmats=viewmats,
            Ks=Ks,
            width=1,
            height=1,
            rgb=_rgb_for_params(params, viewmats, weights),
            alpha=torch.ones(1, 1, 1),
            sh_degree=0,
            background_sigma=0.10,
            background_sigma_for_loss=0.0,
            variance_floor=1.0e-4,
            variance_max=0.25,
            variance_mode="aleatoric_only",
        )

    same = render(torch.zeros(2, 1, 3))
    different_coeffs = torch.zeros(2, 1, 3)
    different_coeffs[1] = 2.0
    different = render(different_coeffs)
    assert different.disagreement_variance.item() > same.disagreement_variance.item()
    assert torch.allclose(different.loss_variance, same.loss_variance)
