from __future__ import annotations

import pytest
import torch

from models.iforward.biggs_parent_projector import project_biggs_parents
from models.iforward.biggs_parent_projector_diag import project_biggs_parent_diag_reference_tensors
from models.iforward.biggs_state import BigGSBranchAssignment


class _Branch:
    def __init__(self, *, means, scales_log, quats, opacity_logit, sh_dc, sh_rest) -> None:
        self.means = means
        self.scales_log = scales_log
        self.quats = quats
        self.opacity_logit = opacity_logit
        self.sh_dc = sh_dc
        self.sh_rest = sh_rest


def _assignment(child_to_parent: torch.Tensor, *, child_mass: torch.Tensor | None = None) -> BigGSBranchAssignment:
    child_to_parent = child_to_parent.long()
    n = int(child_to_parent.numel())
    m = int(child_to_parent.max().item() + 1) if n else 0
    order = []
    starts = torch.zeros((m,), dtype=torch.long, device=child_to_parent.device)
    counts = torch.zeros((m,), dtype=torch.long, device=child_to_parent.device)
    for parent in range(m):
        starts[parent] = len(order)
        rows = torch.nonzero(child_to_parent == parent, as_tuple=False).reshape(-1)
        counts[parent] = int(rows.numel())
        order.extend(int(x) for x in rows.tolist())
    if child_mass is None:
        child_mass = torch.ones((n,), dtype=torch.float32, device=child_to_parent.device)
    return BigGSBranchAssignment(
        branch="bg",
        child_to_parent=child_to_parent,
        child_order=torch.tensor(order, dtype=torch.long, device=child_to_parent.device),
        parent_start=starts,
        parent_count=counts,
        child_mass=child_mass.to(device=child_to_parent.device),
        num_children=n,
        num_parents=m,
    )


def _inputs(device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[_Branch, BigGSBranchAssignment]:
    means = torch.tensor(
        [
            [0.0, 0.1, 0.2],
            [0.3, -0.1, 0.5],
            [1.0, 0.2, -0.2],
            [1.2, 0.6, 0.3],
            [2.0, -0.4, 0.1],
        ],
        device=device,
        dtype=dtype,
    )
    scales_log = torch.tensor(
        [
            [-2.1, -1.7, -1.2],
            [-1.9, -1.4, -2.3],
            [-1.2, -2.0, -1.5],
            [-1.5, -1.1, -2.1],
            [-1.8, -1.3, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.1, 0.2, 0.05],
            [0.9, -0.2, 0.1, 0.3],
            [0.8, 0.4, -0.1, 0.2],
            [0.92, 0.05, 0.22, -0.1],
        ],
        device=device,
        dtype=dtype,
    )
    opacity_logit = torch.tensor([[-0.7], [-0.2], [0.1], [0.4], [-0.5]], device=device, dtype=dtype)
    sh_dc = torch.arange(15, device=device, dtype=dtype).reshape(5, 3) * 0.03
    sh_rest = torch.arange(5 * 3 * 3, device=device, dtype=dtype).reshape(5, 3, 3) * 0.01
    assign = _assignment(
        torch.tensor([0, 0, 1, 1, 2], device=device),
        child_mass=torch.tensor([1.0, 2.0, 1.5, 0.5, 3.0], device=device, dtype=dtype),
    )
    return _Branch(
        means=means,
        scales_log=scales_log,
        quats=quats,
        opacity_logit=opacity_logit,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
    ), assign


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
        "finite_check": True,
    }
    cfg.update(extra)
    return cfg


def test_biggs_diag_reference_matches_projector_dispatch_and_identity_quat() -> None:
    branch, assign = _inputs(torch.device("cpu"))
    proj = project_biggs_parents(branch=branch, assignment=assign, cfg=_cfg(), max_scale=2.0)
    ref = project_biggs_parent_diag_reference_tensors(
        means=branch.means,
        scales_log=branch.scales_log,
        quats=branch.quats,
        opacity_logit=branch.opacity_logit,
        sh_dc=branch.sh_dc,
        sh_rest=branch.sh_rest,
        child_mass=assign.child_mass,
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        min_scale=1.0e-3,
        max_scale=2.0,
        opacity_cap=0.9,
        opacity_min=1.0e-6,
        tau_parent_scale=0.5,
        eps=1.0e-6,
        min_mass=1.0e-8,
        mass_mode="dynamic_tau_area",
    )
    keys = ["means", "scales_log", "quats", "opacity_logit", "sh_dc", "sh_rest"]
    for key, expected in zip(keys, ref[:6]):
        assert torch.allclose(proj.params[key], expected, atol=1.0e-6, rtol=1.0e-5)
    assert torch.allclose(proj.params["quats"], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, 4))
    assert proj.aux_stats["projector_backend_id"] == 1.0


def test_biggs_diag_dynamic_mass_backpropagates_opacity_weighted_mean() -> None:
    branch, assign = _inputs(torch.device("cpu"))
    for tensor in (branch.means, branch.scales_log, branch.quats, branch.opacity_logit, branch.sh_dc, branch.sh_rest):
        tensor.requires_grad_(True)
    proj = project_biggs_parents(branch=branch, assignment=assign, cfg=_cfg(), max_scale=2.0)
    loss = proj.params["means"].sum() + proj.params["sh_dc"].sum()
    loss.backward()
    assert branch.opacity_logit.grad is not None
    assert float(branch.opacity_logit.grad.abs().sum().item()) > 0.0
    assert branch.scales_log.grad is not None
    assert float(branch.scales_log.grad.abs().sum().item()) > 0.0


def test_biggs_diag_static_mass_does_not_route_mean_grad_through_opacity() -> None:
    branch, assign = _inputs(torch.device("cpu"))
    for tensor in (branch.means, branch.scales_log, branch.quats, branch.opacity_logit, branch.sh_dc, branch.sh_rest):
        tensor.requires_grad_(True)
    proj = project_biggs_parents(
        branch=branch,
        assignment=assign,
        cfg=_cfg(mass_mode="static_assignment_mass"),
        max_scale=2.0,
    )
    loss = proj.params["means"].sum() + proj.params["sh_dc"].sum()
    loss.backward()
    assert branch.opacity_logit.grad is None or float(branch.opacity_logit.grad.abs().sum().item()) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_biggs_diag_cuda_forward_matches_reference() -> None:
    branch, assign = _inputs(torch.device("cuda"))
    cfg = _cfg(backend="cuda_exact_diag", allow_torch_fallback=False, finite_check=False)
    proj = project_biggs_parents(branch=branch, assignment=assign, cfg=cfg, max_scale=2.0)
    ref = project_biggs_parent_diag_reference_tensors(
        means=branch.means,
        scales_log=branch.scales_log,
        quats=branch.quats,
        opacity_logit=branch.opacity_logit,
        sh_dc=branch.sh_dc,
        sh_rest=branch.sh_rest,
        child_mass=assign.child_mass,
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        min_scale=1.0e-3,
        max_scale=2.0,
        opacity_cap=0.9,
        opacity_min=1.0e-6,
        tau_parent_scale=0.5,
        eps=1.0e-6,
        min_mass=1.0e-8,
        mass_mode="dynamic_tau_area",
    )
    for actual, expected in zip(
        (
            proj.params["means"],
            proj.params["scales_log"],
            proj.params["quats"],
            proj.params["opacity_logit"],
            proj.params["sh_dc"],
            proj.params["sh_rest"],
            proj.child_mass_sum,
            proj.child_mass_mean,
        ),
        ref,
    ):
        assert torch.allclose(actual, expected, atol=5.0e-5, rtol=5.0e-4)
    assert proj.aux_stats["projector_backend_id"] == 2.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_biggs_diag_cuda_backward_gradcheck() -> None:
    from models.iforward.cuda_parent_projector import project_biggs_parent_diag_cuda_tensors

    branch, assign = _inputs(torch.device("cuda"), dtype=torch.float64)
    tensors = (
        branch.means.detach().requires_grad_(True),
        branch.scales_log.detach().requires_grad_(True),
        branch.quats.detach().requires_grad_(True),
        branch.opacity_logit.detach().requires_grad_(True),
        branch.sh_dc.detach().requires_grad_(True),
        branch.sh_rest.detach().requires_grad_(True),
    )

    def fn(means, scales_log, quats, opacity_logit, sh_dc, sh_rest):
        out = project_biggs_parent_diag_cuda_tensors(
            means=means,
            scales_log=scales_log,
            quats=quats,
            opacity_logit=opacity_logit,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
            child_mass=assign.child_mass.to(dtype=torch.float64),
            child_order=assign.child_order,
            parent_start=assign.parent_start,
            parent_count=assign.parent_count,
            min_scale=1.0e-4,
            max_scale=3.0,
            opacity_cap=0.95,
            opacity_min=1.0e-6,
            tau_parent_scale=0.4,
            eps=1.0e-8,
            min_mass=1.0e-8,
            mass_mode="dynamic_tau_area",
        )
        return out[0], out[1], out[3], out[4], out[5], out[6]

    assert torch.autograd.gradcheck(fn, tensors, eps=1.0e-4, atol=2.0e-3, rtol=2.0e-2)
