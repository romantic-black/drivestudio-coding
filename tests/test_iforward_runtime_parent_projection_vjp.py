from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.iforward.biggs_parent_projector import BigGSParentProjection
from models.iforward.biggs_parent_projector_diag import project_biggs_parent_diag_reference_tensors
from models.iforward.biggs_parent_stats import (
    init_parent_branch_runtime,
    projection_from_runtime,
    refresh_parent_branch_runtime_exact,
)
from models.iforward.parent_spatial_backbone import ParentSpatialBackbone
from models.iforward.runtime_parent_projection_vjp import (
    ParentVJPDriftPolicy,
    RuntimeParentVJPDriftCollector,
    parent_projection_feedback,
    runtime_exact_drift,
)


_PARAM_KEYS = ("means", "scales_log", "quats", "opacity_logit", "sh_dc", "sh_rest")


def _cfg(**extra):
    cfg = {
        "backend": "torch_exact_diag",
        "covariance_mode": "diagonal",
        "mass_mode": "dynamic_tau_area",
        "min_scale": 1.0e-3,
        "max_scale": 2.0,
        "opacity_cap": 0.9,
        "opacity_min": 1.0e-6,
        "tau_parent_scale": 0.5,
        "eps": 1.0e-6,
        "min_child_mass": 1.0e-8,
        "child_cache_dtype": "float32",
    }
    cfg.update(extra)
    return cfg


def _child_params(*, requires_grad: bool = True, dtype: torch.dtype = torch.float32):
    params = {
        "means": torch.tensor(
            [[0.0, 0.1, 0.2], [0.3, -0.1, 0.5], [1.0, 0.2, -0.2], [1.2, 0.6, 0.3]],
            dtype=dtype,
        ),
        "scales_log": torch.tensor(
            [[-2.1, -1.7, -1.2], [-1.9, -1.4, -2.3], [-1.2, -2.0, -1.5], [-1.5, -1.1, -2.1]],
            dtype=dtype,
        ),
        "quats": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.95, 0.1, 0.2, 0.05], [0.9, -0.2, 0.1, 0.3], [0.8, 0.4, -0.1, 0.2]],
            dtype=dtype,
        ),
        "opacity_logit": torch.tensor([[-0.7], [-0.2], [0.1], [0.4]], dtype=dtype),
        "sh_dc": torch.arange(12, dtype=dtype).reshape(4, 3) * 0.03,
        "sh_rest": torch.arange(4 * 2 * 3, dtype=dtype).reshape(4, 2, 3) * 0.01,
    }
    if requires_grad:
        for value in params.values():
            value.requires_grad_(True)
    return params


def _assignment(dtype: torch.dtype = torch.float32):
    return {
        "child_to_parent": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "parent_count": torch.tensor([2, 2], dtype=torch.long),
        "child_mass": torch.tensor([1.0, 2.0, 1.5, 0.5], dtype=dtype),
        "child_order": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "parent_start": torch.tensor([0, 2], dtype=torch.long),
    }


def _runtime(params, assignment, *, cfg=None):
    runtime = init_parent_branch_runtime(
        params=params,
        child_to_parent=assignment["child_to_parent"],
        parent_count=assignment["parent_count"],
        child_mass=assignment["child_mass"],
        cfg=_cfg() if cfg is None else cfg,
        child_order=assignment["child_order"],
        parent_start=assignment["parent_start"],
        max_scale=2.0,
        assignment_signature="bg",
    )
    return runtime, projection_from_runtime(runtime)


def _reference(params, assignment, *, cfg=None):
    cfg = _cfg() if cfg is None else cfg
    return project_biggs_parent_diag_reference_tensors(
        means=params["means"],
        scales_log=params["scales_log"],
        quats=params["quats"],
        opacity_logit=params["opacity_logit"],
        sh_dc=params["sh_dc"],
        sh_rest=params["sh_rest"],
        child_mass=assignment["child_mass"],
        child_to_parent=assignment["child_to_parent"],
        parent_count=assignment["parent_count"],
        min_scale=float(cfg["min_scale"]),
        max_scale=float(cfg["max_scale"]),
        opacity_cap=float(cfg["opacity_cap"]),
        opacity_min=float(cfg["opacity_min"]),
        tau_parent_scale=float(cfg["tau_parent_scale"]),
        eps=float(cfg["eps"]),
        min_mass=float(cfg["min_child_mass"]),
        mass_mode=str(cfg["mass_mode"]),
    )


def _feedback(projection, params, assignment, **kwargs):
    return parent_projection_feedback(
        projection,
        child_params=params,
        child_mass=assignment["child_mass"],
        child_to_parent=assignment["child_to_parent"],
        parent_count=assignment["parent_count"],
        projector_cfg=kwargs.pop("projector_cfg", _cfg()),
        alpha=kwargs.pop("alpha", 1.0),
        branch=kwargs.pop("branch", "bg"),
        max_scale=2.0,
        **kwargs,
    )


def _weighted_loss(projection, weights):
    return sum((projection.params[key] * weights[key]).sum() for key in _PARAM_KEYS)


def test_runtime_parent_feedback_preserves_forward_and_matches_exact_vjp() -> None:
    torch.manual_seed(17)
    params = _child_params()
    assignment = _assignment()
    _, runtime_projection = _runtime(params, assignment)
    collector = RuntimeParentVJPDriftCollector()
    feedback = _feedback(runtime_projection, params, assignment, drift_collector=collector)

    weights = {key: torch.randn_like(value) for key, value in runtime_projection.params.items()}
    for key in _PARAM_KEYS:
        torch.testing.assert_close(feedback.params[key], runtime_projection.params[key], rtol=0.0, atol=0.0)
        assert feedback.params[key].requires_grad
    assert feedback.child_mass_sum is runtime_projection.child_mass_sum
    assert feedback.child_mass_mean is runtime_projection.child_mass_mean

    actual = torch.autograd.grad(_weighted_loss(feedback, weights), tuple(params.values()), allow_unused=True)
    ref_params = {key: value.detach().clone().requires_grad_(True) for key, value in params.items()}
    exact = _reference(ref_params, assignment)
    expected_loss = sum((value * weights[key]).sum() for key, value in zip(_PARAM_KEYS, exact[:6]))
    expected = torch.autograd.grad(expected_loss, tuple(ref_params.values()), allow_unused=True)
    for actual_grad, expected_grad in zip(actual, expected):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2.0e-5, atol=2.0e-6)

    report = collector.latest("bg")
    assert report is not None
    assert float(report.max_rel_error) < 1.0e-4
    assert float(report.effective_alpha) == pytest.approx(1.0)
    assert float(report.refresh_required) == 0.0


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("mass_mode", ["dynamic_tau_area", "static_assignment_mass"])
def test_runtime_parent_feedback_alpha_scales_exact_vjp(alpha: float, mass_mode: str) -> None:
    cfg = _cfg(mass_mode=mass_mode)
    params = _child_params()
    assignment = _assignment()
    _, runtime_projection = _runtime(params, assignment, cfg=cfg)
    feedback = _feedback(runtime_projection, params, assignment, projector_cfg=cfg, alpha=alpha)
    loss = (
        feedback.params["means"].square().sum()
        + feedback.params["scales_log"].square().sum()
        + feedback.params["opacity_logit"].square().sum()
        + feedback.params["sh_dc"].square().sum()
        + feedback.params["sh_rest"].square().sum()
    )
    actual = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)

    ref_params = {key: value.detach().clone().requires_grad_(True) for key, value in params.items()}
    exact = _reference(ref_params, assignment, cfg=cfg)
    expected_loss = sum(value.square().sum() for index, value in enumerate(exact[:6]) if index != 2)
    expected = torch.autograd.grad(expected_loss, tuple(ref_params.values()), allow_unused=True)
    for actual_grad, expected_grad in zip(actual, expected):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad * alpha, rtol=3.0e-5, atol=3.0e-6)


def test_runtime_parent_feedback_drift_reduces_and_skips_vjp() -> None:
    params = _child_params()
    assignment = _assignment()
    _, runtime_projection = _runtime(params, assignment)
    base_params = dict(runtime_projection.params)
    base_params["means"] = base_params["means"] + 0.2
    drifted = replace(runtime_projection, params=base_params)
    policy = ParentVJPDriftPolicy(
        warn_threshold=0.0,
        skip_vjp_threshold=1.0,
        exact_refresh_threshold=1.0,
    )
    collector = RuntimeParentVJPDriftCollector()
    feedback = _feedback(
        drifted,
        params,
        assignment,
        drift_policy=policy,
        drift_collector=collector,
    )
    actual = torch.autograd.grad(feedback.params["means"].sum(), params["means"], retain_graph=False)[0]
    report = collector.latest("bg")
    assert report is not None
    assert 0.0 < float(report.effective_alpha) < 1.0

    ref_means = params["means"].detach().clone().requires_grad_(True)
    ref_params = {**params, "means": ref_means}
    exact = _reference(ref_params, assignment)
    expected = torch.autograd.grad(exact[0].sum(), ref_means)[0]
    torch.testing.assert_close(actual, expected * report.effective_alpha, rtol=2.0e-5, atol=2.0e-6)

    skip_params = dict(runtime_projection.params)
    skip_params["means"] = skip_params["means"] + 100.0
    skip_projection = replace(runtime_projection, params=skip_params)
    skip_collector = RuntimeParentVJPDriftCollector()
    skipped = _feedback(
        skip_projection,
        params,
        assignment,
        drift_policy=ParentVJPDriftPolicy(
            warn_threshold=1.0e-4,
            skip_vjp_threshold=1.0e-3,
            exact_refresh_threshold=2.0e-3,
        ),
        drift_collector=skip_collector,
    )
    skipped_grad = torch.autograd.grad(skipped.params["means"].sum(), params["means"])[0]
    assert torch.count_nonzero(skipped_grad) == 0
    skip_report = skip_collector.latest("bg")
    assert skip_report is not None
    assert float(skip_report.vjp_skipped) == 1.0
    assert float(skip_report.refresh_required) == 1.0


def test_runtime_exact_drift_and_refresh_are_graph_free() -> None:
    params = _child_params()
    assignment = _assignment()
    runtime, projection = _runtime(params, assignment)
    report = runtime_exact_drift(
        projection,
        child_params=params,
        child_mass=assignment["child_mass"],
        child_to_parent=assignment["child_to_parent"],
        parent_count=assignment["parent_count"],
        projector_cfg=_cfg(),
        alpha=0.3,
        drift_policy=ParentVJPDriftPolicy(),
        branch="bg",
        max_scale=2.0,
    )
    assert float(report.max_rel_error) < 1.0e-4
    assert float(report.effective_alpha) == pytest.approx(0.3)

    refreshed = refresh_parent_branch_runtime_exact(
        runtime=runtime,
        params=params,
        child_to_parent=assignment["child_to_parent"],
        parent_count=assignment["parent_count"],
        child_mass=assignment["child_mass"],
        cfg=_cfg(),
        child_order=assignment["child_order"],
        parent_start=assignment["parent_start"],
        max_scale=2.0,
    )
    assert refreshed.assignment_signature == "bg"
    tensors = (
        *refreshed.params.values(),
        refreshed.stats.weight_sum,
        refreshed.stats.weighted_mean_sum,
        refreshed.stats.weighted_second_sum,
        refreshed.child_cache.mass,
        refreshed.child_cache.diag_cov,
    )
    assert all(not value.requires_grad and value.grad_fn is None for value in tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_runtime_parent_feedback_cuda_amp_recomputes_vjp_in_fp32(dtype: torch.dtype) -> None:
    params_cpu = _child_params(requires_grad=False, dtype=dtype)
    params = {
        key: value.cuda().detach().requires_grad_(True)
        for key, value in params_cpu.items()
    }
    assignment = {key: value.cuda() for key, value in _assignment(dtype=dtype).items()}
    _, runtime_projection = _runtime(params, assignment)
    collector = RuntimeParentVJPDriftCollector()
    policy = ParentVJPDriftPolicy(
        warn_threshold=0.1,
        skip_vjp_threshold=0.5,
        exact_refresh_threshold=1.0,
    )
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        feedback = _feedback(
            runtime_projection,
            params,
            assignment,
            alpha=0.3,
            drift_policy=policy,
            drift_collector=collector,
        )
        loss = (
            feedback.params["means"].square().sum()
            + feedback.params["scales_log"].square().sum()
            + feedback.params["opacity_logit"].square().sum()
            + feedback.params["sh_dc"].square().sum()
            + feedback.params["sh_rest"].square().sum()
        )
    actual = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)

    ref_params = {
        key: value.detach().float().requires_grad_(True)
        for key, value in params.items()
    }
    ref_assignment = {
        key: value.float() if key == "child_mass" else value
        for key, value in assignment.items()
    }
    exact = _reference(ref_params, ref_assignment)
    expected_loss = sum(value.square().sum() for index, value in enumerate(exact[:6]) if index != 2)
    expected = torch.autograd.grad(expected_loss, tuple(ref_params.values()), allow_unused=True)
    report = collector.latest("bg")
    assert report is not None
    assert float(report.effective_alpha) == pytest.approx(0.3)
    tolerance = 2.0e-3 if dtype == torch.float16 else 3.0e-5
    for actual_grad, expected_grad in zip(actual, expected):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(
                actual_grad.float(),
                expected_grad * 0.3,
                rtol=tolerance,
                atol=tolerance,
            )


def test_runtime_parent_feedback_rejects_unsupported_or_attached_runtime() -> None:
    params = _child_params()
    assignment = _assignment()
    _, projection = _runtime(params, assignment)
    with pytest.raises(ValueError, match="only supports bg/distant"):
        _feedback(projection, params, assignment, branch="rigid")
    with pytest.raises(ValueError, match="exact diagonal covariance"):
        _feedback(projection, params, assignment, projector_cfg=_cfg(covariance_mode="full_eigh"))

    attached_params = dict(projection.params)
    attached_params["means"] = attached_params["means"].detach().requires_grad_(True)
    with pytest.raises(RuntimeError, match="runtime must remain graph-free"):
        _feedback(replace(projection, params=attached_params), params, assignment)

    not_runtime = BigGSParentProjection(
        params=dict(projection.params),
        child_mass_sum=projection.child_mass_sum,
        child_mass_mean=projection.child_mass_mean,
    )
    with pytest.raises(ValueError, match="incremental runtime projection"):
        _feedback(not_runtime, params, assignment)


def test_parent_spatial_backbone_configures_codec_detach_without_shape_change() -> None:
    legacy = ParentSpatialBackbone(context_dim=8, event_dim=8, token_dim=8, param_support_dim=6, near_heads=1)
    feedback = ParentSpatialBackbone(
        context_dim=8,
        event_dim=8,
        token_dim=8,
        param_support_dim=6,
        param_codec_detach_params=False,
        param_codec_detach_support=True,
        near_heads=1,
    )
    assert legacy.param_support_codec.detach_params is True
    assert legacy.param_support_codec.detach_support is True
    assert feedback.param_support_codec.detach_params is False
    assert feedback.param_support_codec.detach_support is True
    assert set(legacy.state_dict()) == set(feedback.state_dict())
    for key in legacy.state_dict():
        assert tuple(legacy.state_dict()[key].shape) == tuple(feedback.state_dict()[key].shape)


def test_parent_vjp_drift_policy_validates_threshold_order() -> None:
    with pytest.raises(ValueError, match="must satisfy"):
        ParentVJPDriftPolicy(warn_threshold=0.1, skip_vjp_threshold=0.01, exact_refresh_threshold=1.0)
    with pytest.raises(ValueError, match="check_interval"):
        ParentVJPDriftPolicy(check_interval=-1)
