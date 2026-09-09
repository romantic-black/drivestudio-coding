from __future__ import annotations

from typing import Dict

import pytest
import torch
import torch.nn.functional as F

from models.iforward.biggs_parent_projector import BigGSParentProjection
from models.iforward.biggs_parent_projector_diag import (
    compute_child_projection_stats,
    project_biggs_parent_diag_reference_tensors,
)
from models.iforward.biggs_parent_stats import init_parent_branch_runtime
from models.iforward.biggs_state import BigGSBranchAssignment, BigGSRigidActiveAssignment
from models.iforward.functional_parentgs import (
    FunctionalChildStats,
    FunctionalParentBranch,
    FunctionalParentPack,
    FunctionalParentProjectorConfig,
    build_functional_parent_pack,
    build_parent_lift_scene,
)
from models.iforward.parent_spatial_backbone import ParentSpatialBackbone, ParentStructInput


def _assignment(
    child_to_parent: torch.Tensor,
    *,
    branch: str = "bg",
) -> BigGSBranchAssignment:
    child_to_parent = child_to_parent.long()
    n = int(child_to_parent.numel())
    m = int(child_to_parent.max().item()) + 1 if n else 0
    counts = torch.bincount(child_to_parent, minlength=m)
    order = torch.argsort(child_to_parent, stable=True)
    starts = torch.cumsum(counts, dim=0) - counts
    return BigGSBranchAssignment(
        branch=branch,
        child_to_parent=child_to_parent,
        child_order=order,
        parent_start=starts,
        parent_count=counts,
        child_mass=torch.linspace(0.5, 1.5, n, device=child_to_parent.device),
        num_children=n,
        num_parents=m,
    )


def _params(device: torch.device, *, requires_grad: bool = False) -> Dict[str, torch.Tensor]:
    params = {
        "means": torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, -0.1, 0.5],
                [1.0, 0.2, -0.2],
                [1.2, 0.6, 0.3],
                [2.0, -0.4, 0.1],
            ],
            device=device,
        ),
        "scales_log": torch.tensor(
            [
                [-2.1, -1.7, -1.2],
                [-1.9, -1.4, -2.3],
                [-1.2, -2.0, -1.5],
                [-1.5, -1.1, -2.1],
                [-1.8, -1.3, -1.0],
            ],
            device=device,
        ),
        "quats": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.95, 0.1, 0.2, 0.05],
                [0.9, -0.2, 0.1, 0.3],
                [0.8, 0.4, -0.1, 0.2],
                [0.92, 0.05, 0.22, -0.1],
            ],
            device=device,
        ),
        "opacity_logit": torch.tensor([[-0.7], [-0.2], [0.1], [0.4], [-0.5]], device=device),
        "sh_dc": torch.arange(15, device=device, dtype=torch.float32).reshape(5, 3) * 0.03,
        "sh_rest": torch.arange(5 * 3 * 3, device=device, dtype=torch.float32).reshape(5, 3, 3) * 0.01,
    }
    if requires_grad:
        params = {key: value.requires_grad_(True) for key, value in params.items()}
    return params


def _projector_cfg(**kwargs: object) -> FunctionalParentProjectorConfig:
    values: dict[str, object] = {
        "max_scale_bg": 2.0,
        "max_scale_distant": 3.0,
        "max_scale_rigid": 1.5,
        "opacity_cap": 0.95,
        "tau_parent_scale_bg": 0.4,
        "tau_parent_scale_distant": 0.6,
        "tau_parent_scale_rigid": 0.5,
    }
    values.update(kwargs)
    return FunctionalParentProjectorConfig(**values)


def test_compute_child_projection_stats_matches_formula_and_is_attached() -> None:
    params = _params(torch.device("cpu"), requires_grad=True)
    assignment = _assignment(torch.tensor([0, 0, 1, 1, 2]))
    mass, tau_area, diag_cov = compute_child_projection_stats(
        scales_log=params["scales_log"],
        quats=params["quats"],
        opacity_logit=params["opacity_logit"],
        child_mass=assignment.child_mass,
        min_mass=1.0e-8,
        mass_mode="dynamic_tau_area",
    )
    scales = params["scales_log"].exp()
    expected_tau_area = F.softplus(params["opacity_logit"].reshape(-1)) * torch.topk(
        scales, k=2, dim=-1
    ).values.prod(dim=-1)
    assert torch.allclose(tau_area, expected_tau_area)
    assert torch.allclose(mass, expected_tau_area.clamp_min(1.0e-8))
    assert tuple(diag_cov.shape) == (5, 3)
    grads = torch.autograd.grad(
        mass.sum() + diag_cov.sum(),
        (params["scales_log"], params["quats"], params["opacity_logit"]),
    )
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert all(float(grad.abs().sum()) > 0.0 for grad in grads)


def test_legacy_runtime_cache_uses_shared_child_projection_stats() -> None:
    params = _params(torch.device("cpu"))
    assignment = _assignment(torch.tensor([0, 0, 1, 1, 2]))
    expected = compute_child_projection_stats(
        scales_log=params["scales_log"],
        quats=params["quats"],
        opacity_logit=params["opacity_logit"],
        child_mass=assignment.child_mass,
        min_mass=1.0e-8,
        mass_mode="dynamic_tau_area",
    )
    runtime = init_parent_branch_runtime(
        params=params,
        child_to_parent=assignment.child_to_parent,
        child_order=assignment.child_order,
        parent_start=assignment.parent_start,
        parent_count=assignment.parent_count,
        child_mass=assignment.child_mass,
        cfg={
            "backend": "torch_exact_diag",
            "mass_mode": "dynamic_tau_area",
            "min_child_mass": 1.0e-8,
            "min_scale": 1.0e-3,
            "max_scale": 2.0,
        },
    )
    for actual, value in zip(
        (runtime.child_cache.mass, runtime.child_cache.tau_area, runtime.child_cache.diag_cov),
        expected,
    ):
        assert torch.allclose(actual, value.detach(), atol=1.0e-6, rtol=1.0e-5)
        assert actual.requires_grad is False


def test_functional_projector_config_rejects_legacy_or_fallback_modes() -> None:
    with pytest.raises(ValueError, match="backend=cuda_exact_diag"):
        FunctionalParentProjectorConfig(backend="torch_exact_diag")
    with pytest.raises(ValueError, match="forbids CPU/Torch"):
        FunctionalParentProjectorConfig(allow_torch_fallback=True)
    with pytest.raises(ValueError, match="recompute_every_visit"):
        FunctionalParentProjectorConfig(recompute_every_visit=False)


def test_functional_pack_attachment_api_is_mutually_exclusive_and_strict() -> None:
    params = _params(torch.device("cpu"))
    assignment = _assignment(torch.tensor([0, 0, 1, 1, 2]))
    common = {
        "bg_params": params,
        "bg_assignment": assignment,
        "projector_cfg": _projector_cfg(),
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_functional_parent_pack(
            **common,
            attached=False,
            attached_by_branch={"bg": False},
        )
    with pytest.raises(ValueError, match="one of attached"):
        build_functional_parent_pack(**common)
    with pytest.raises(ValueError, match="missing present branches"):
        build_functional_parent_pack(**common, attached_by_branch={})
    with pytest.raises(TypeError, match="values must be booleans"):
        build_functional_parent_pack(
            **common,
            attached_by_branch={"bg": 1},
        )
    with pytest.raises(ValueError, match="unsupported branches"):
        build_functional_parent_pack(
            **common,
            attached_by_branch={"bg": False, "rigid": False},
        )


def _manual_branch(name: str, offset: float, rows: int) -> FunctionalParentBranch:
    means = (torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) + float(offset)).requires_grad_(True)
    params = {
        "means": means,
        "scales_log": torch.full((rows, 3), -1.0, requires_grad=True),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(rows, 4),
        "opacity_logit": torch.zeros((rows, 1), requires_grad=True),
        "sh_dc": torch.ones((rows, 3), requires_grad=True),
        "sh_rest": torch.ones((rows, 3, 3), requires_grad=True),
    }
    assignment = _assignment(torch.arange(rows), branch=name)
    projection = BigGSParentProjection(
        params=params,
        child_mass_sum=torch.ones((rows,), requires_grad=True),
        child_mass_mean=torch.ones((rows,), requires_grad=True),
    )
    return FunctionalParentBranch(
        assignment=assignment,
        projection=projection,
        child_stats_detached=FunctionalChildStats(
            mass=torch.ones((rows,)),
            tau_area=torch.ones((rows,)),
            diag_cov=torch.ones((rows, 3)),
        ),
        parent_mass_mean=projection.child_mass_mean.detach(),
        branch_name=name,
    )


def test_parent_lift_scene_is_constant_zero_ordered_and_geometry_isolated() -> None:
    bg = _manual_branch("bg", 0.0, 2)
    distant = _manual_branch("distant", 100.0, 1)
    rigid = _manual_branch("rigid_active", 200.0, 2)
    pack = FunctionalParentPack(bg=bg, distant=distant, rigid_active=rigid)
    scene = build_parent_lift_scene(pack)
    expected_means = torch.cat(
        [bg.projection.params["means"], distant.projection.params["means"], rigid.projection.params["means"]],
        dim=0,
    )
    assert torch.equal(scene["means"], expected_means.detach())
    assert torch.count_nonzero(scene["colors"]) == 0
    assert all(value.requires_grad is False and value.grad_fn is None for value in scene.values())

    frontend_feature = torch.ones((), requires_grad=True)
    loss = frontend_feature * (scene["opacities"].sum() + 1.0)
    child_grad, frontend_grad = torch.autograd.grad(
        loss,
        (bg.projection.params["opacity_logit"], frontend_feature),
        allow_unused=True,
    )
    assert child_grad is None
    assert frontend_grad is not None and float(frontend_grad) > 0.0
    assert bg.parent_mass_sum.requires_grad is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_functional_cuda_attached_forward_only_and_reference_parity() -> None:
    device = torch.device("cuda")
    params = _params(device, requires_grad=True)
    assignment = _assignment(torch.tensor([0, 0, 1, 1, 2], device=device))
    config = _projector_cfg()
    attached = build_functional_parent_pack(
        bg_params=params,
        bg_assignment=assignment,
        projector_cfg=config,
        attached=True,
    )
    first_visit = build_functional_parent_pack(
        bg_params=params,
        bg_assignment=assignment,
        projector_cfg=config,
        attached=False,
    )
    branch_attached = build_functional_parent_pack(
        bg_params=params,
        bg_assignment=assignment,
        projector_cfg=config,
        attached_by_branch={"bg": True},
    )
    ref = project_biggs_parent_diag_reference_tensors(
        means=params["means"],
        scales_log=params["scales_log"],
        quats=params["quats"],
        opacity_logit=params["opacity_logit"],
        sh_dc=params["sh_dc"],
        sh_rest=params["sh_rest"],
        child_mass=assignment.child_mass,
        child_to_parent=assignment.child_to_parent,
        parent_count=assignment.parent_count,
        min_scale=config.min_scale,
        max_scale=config.max_scale_bg,
        opacity_cap=config.opacity_cap,
        opacity_min=config.opacity_min,
        tau_parent_scale=config.tau_parent_scale_bg,
        eps=config.eps,
        min_mass=config.min_child_mass,
        mass_mode=config.mass_mode,
    )
    for index, key in enumerate(("means", "scales_log", "quats", "opacity_logit", "sh_dc", "sh_rest")):
        assert torch.allclose(attached.bg.projection.params[key], ref[index], atol=3.0e-5, rtol=3.0e-4)
        assert torch.allclose(branch_attached.bg.projection.params[key], ref[index], atol=3.0e-5, rtol=3.0e-4)
        assert torch.allclose(first_visit.bg.projection.params[key], ref[index], atol=3.0e-5, rtol=3.0e-4)
    assert attached.bg.projection.params["means"].grad_fn is not None
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in first_visit.bg.projection.params.values()
    )
    assert attached.bg.child_stats_detached.mass.requires_grad is False
    assert attached.bg.child_stats_detached.diag_cov.requires_grad is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_functional_cuda_projector_directional_derivative() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    params = _params(device)
    assignment = _assignment(torch.tensor([0, 0, 1, 1, 2], device=device))
    config = _projector_cfg(
        max_scale_bg=10.0,
        opacity_cap=0.99,
        tau_parent_scale_bg=0.2,
    )
    differentiable_keys = ("means", "scales_log", "opacity_logit")
    attached_params = {
        key: value.detach().clone().requires_grad_(key in differentiable_keys)
        for key, value in params.items()
    }
    directions = {
        key: torch.randn_like(attached_params[key]) * 0.1
        for key in differentiable_keys
    }
    output_weights = {
        "means": torch.randn((3, 3), device=device),
        "scales_log": torch.randn((3, 3), device=device),
        "opacity_logit": torch.randn((3, 1), device=device),
    }

    def objective(values: Dict[str, torch.Tensor]) -> torch.Tensor:
        branch = build_functional_parent_pack(
            bg_params=values,
            bg_assignment=assignment,
            projector_cfg=config,
            attached=True,
        ).bg
        return sum((branch.projection.params[key] * weight).sum() for key, weight in output_weights.items())

    loss = objective(attached_params)
    grads = torch.autograd.grad(loss, tuple(attached_params[key] for key in differentiable_keys))
    derivative_ad = sum((grad * directions[key]).sum() for key, grad in zip(differentiable_keys, grads))
    epsilon = 1.0e-3
    plus = {
        key: (
            value.detach() + epsilon * directions[key]
            if key in differentiable_keys
            else value.detach()
        )
        for key, value in attached_params.items()
    }
    minus = {
        key: (
            value.detach() - epsilon * directions[key]
            if key in differentiable_keys
            else value.detach()
        )
        for key, value in attached_params.items()
    }
    derivative_fd = (objective(plus) - objective(minus)) / (2.0 * epsilon)
    relative_error = (derivative_ad - derivative_fd).abs() / (1.0 + derivative_fd.abs())
    assert float(relative_error) <= 5.0e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_k2_earlier_delta_gradient_flows_through_exact_projector_and_v57_residual_backbone() -> None:
    """Prove the real Parent-only path without a source-render dependency.

    The deltas model visit-1 updater outputs consumed by visit 2.  The loss is
    downstream of the exact CUDA projector, detached legacy 17D codec, live 8D
    residual adapter, token builder, and far ParentSpatial decoder.
    """

    torch.manual_seed(34)
    device = torch.device("cuda")
    base = _params(device)
    assignment = _assignment(torch.tensor([0, 0, 1, 1, 2], device=device))
    means_delta = (torch.randn_like(base["means"]) * 0.01).requires_grad_(True)
    scales_delta = (torch.randn_like(base["scales_log"]) * 0.01).requires_grad_(True)
    opacity_delta = (torch.randn_like(base["opacity_logit"]) * 0.01).requires_grad_(True)
    live = {
        **base,
        "means": base["means"] + means_delta,
        "scales_log": base["scales_log"] + scales_delta,
        "opacity_logit": base["opacity_logit"] + opacity_delta,
    }
    branch = build_functional_parent_pack(
        bg_params=live,
        bg_assignment=assignment,
        projector_cfg=_projector_cfg(max_scale_bg=10.0, opacity_cap=0.99),
        attached=True,
    ).bg
    backbone = ParentSpatialBackbone(
        context_dim=4,
        event_dim=8,
        token_dim=8,
        param_codec_mode="legacy17d_plus_geometry8d_residual",
        param_codec_detach_params=False,
        param_codec_detach_support=True,
        ptv3_detach_coords=True,
        near_heads=2,
        use_xcpe=False,
    ).to(device)
    assert backbone.geometry_residual_adapter is not None
    # Simulate the first optimizer update: v57 intentionally zero-initializes
    # this projection, so geometry Jacobians become observable after it learns.
    with torch.no_grad():
        backbone.geometry_residual_adapter.output_proj.weight.normal_(mean=0.0, std=0.1)

    parent_input = ParentStructInput(
        parent_context=torch.zeros((branch.num_parents, 4), device=device),
        support=branch.parent_mass_mean,
        valid=torch.ones(branch.num_parents, dtype=torch.bool, device=device),
        coords=branch.projection.params["means"].detach(),
        branch_id=torch.zeros(branch.num_parents, dtype=torch.long, device=device),
        params_for_embed=branch.projection.params,
        split_0=branch.num_parents,
        split_1=0,
        geometry_branch_id=torch.zeros(branch.num_parents, dtype=torch.long, device=device),
        geometry_alpha=1.0,
    )
    encoded = backbone.encode_far(
        parent_input,
        aabb_min=torch.full((3,), -10.0, device=device),
        aabb_max=torch.full((3,), 10.0, device=device),
    )
    upstream = torch.randn_like(encoded.event)
    grads = torch.autograd.grad(
        (encoded.event * upstream).sum(),
        (means_delta, scales_delta, opacity_delta),
        allow_unused=True,
    )
    for grad in grads:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_functional_pack_projects_complete_rigid_active_route() -> None:
    device = torch.device("cuda")
    bg_params = _params(device, requires_grad=True)
    bg_assignment = _assignment(torch.tensor([0, 0, 1, 1, 2], device=device))
    rigid_params = {key: value[:3] for key, value in bg_params.items()}
    rigid_assignment = BigGSRigidActiveAssignment(
        fine_S=torch.tensor([4, 1, 3], device=device),
        child_to_active_parent_S=torch.tensor([0, 1, 0], device=device),
        active_parent_global=torch.tensor([2, 5], device=device),
        active_parent_count=torch.tensor([2, 1], device=device),
        active_parent_start=torch.tensor([0, 2], device=device),
        active_child_order_S=torch.tensor([0, 2, 1], device=device),
        child_mass_S=torch.tensor([1.0, 2.0, 3.0], device=device),
        parent_inside_mask=torch.tensor([True, False], device=device),
        child_inside_mask_S=torch.tensor([True, False, True], device=device),
    )
    pack = build_functional_parent_pack(
        bg_params=bg_params,
        bg_assignment=bg_assignment,
        rigid_active_params=rigid_params,
        rigid_active_assignment=rigid_assignment,
        projector_cfg=_projector_cfg(),
        attached_by_branch={"bg": True, "rigid_active": False},
    )
    assert pack.distant is None
    assert pack.rigid_active is not None
    assert pack.rigid_active.branch_name == "rigid_active"
    assert pack.rigid_active.num_children == 3
    assert pack.rigid_active.num_parents == 2
    assert tuple(branch.branch_name for branch in pack.iter_branches()) == ("bg", "rigid_active")
    assert pack.bg.projection.params["means"].requires_grad
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in pack.rigid_active.projection.params.values()
    )
