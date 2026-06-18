from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.iforward.biggs_assignment import build_biggs_branch_assignment, build_biggs_assignments
from models.iforward.biggs_event_decoder import BigGSToFineEventDecoder
from models.iforward.biggs_parent_projector import _canonicalize_quat, project_biggs_active_rigid_parents, project_biggs_parents
from models.iforward.biggs_parent_stats import init_parent_branch_runtime, update_parent_branch_runtime
from models.iforward.biggs_state import BigGSBranchAssignment, BigGSRigidActiveAssignment, IForwardBigGSState
from models.iforward.state import IForwardState
from models.iforward.trainer import IForwardTrainer
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0 import LocalGSState, Stage6PosteriorUpdater, Stage6StructInput
from models.streetforward.stage6_0.event_encoder import EventPack


def _quat(n: int, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    out = torch.zeros((int(n), 4), device=device)
    out[:, 0] = 1.0
    return out


def _node_bg(
    n: int,
    *,
    means: torch.Tensor | None = None,
    device: torch.device = torch.device("cpu"),
    sh_bases: int = 2,
) -> NodeStateBackground:
    if means is None:
        means = torch.arange(int(n), device=device, dtype=torch.float32).reshape(-1, 1).repeat(1, 3) * 0.1
    else:
        means = means.to(device=device, dtype=torch.float32)
        n = int(means.shape[0])
    return NodeStateBackground(
        means=means,
        scales_log=torch.full((n, 3), -2.0, device=device),
        quats=_quat(n, device=device),
        opacity_logit=torch.full((n, 1), -1.5, device=device),
        sh_dc=torch.linspace(0.0, 1.0, max(n * 3, 1), device=device).reshape(n, 3),
        sh_rest=torch.zeros((n, int(sh_bases), 3), device=device),
    )


def _node_distant(n: int, *, device: torch.device = torch.device("cpu")) -> NodeStateDistant:
    bg = _node_bg(n, device=device)
    return NodeStateDistant(**bg.__dict__)


def _node_rigid(
    n: int,
    *,
    means: torch.Tensor | None = None,
    point_ids: torch.Tensor | None = None,
    device: torch.device = torch.device("cpu"),
    sh_bases: int = 2,
) -> NodeStateRigid:
    bg = _node_bg(n, means=means, device=device, sh_bases=sh_bases)
    if point_ids is None:
        point_ids = torch.zeros((int(n), 1), dtype=torch.long, device=device)
    return NodeStateRigid(
        means=bg.means,
        scales_log=bg.scales_log,
        quats=bg.quats,
        opacity_logit=bg.opacity_logit,
        sh_dc=bg.sh_dc,
        sh_rest=bg.sh_rest,
        point_ids=point_ids.to(device=device, dtype=torch.long),
        instances_quats=_quat(1, device=device).reshape(1, 1, 4),
        instances_trans=torch.zeros((1, 1, 3), device=device),
        instances_fv=torch.ones((1, 1), dtype=torch.bool, device=device),
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )


def _manual_assignment(
    child_to_parent: list[int],
    *,
    child_mass: list[float] | None = None,
    branch: str = "bg",
    device: torch.device = torch.device("cpu"),
) -> BigGSBranchAssignment:
    ctp = torch.tensor(child_to_parent, dtype=torch.long, device=device)
    n = int(ctp.numel())
    m = int(ctp.max().item() + 1) if n else 0
    order = []
    starts = torch.zeros((m,), dtype=torch.long, device=device)
    counts = torch.zeros((m,), dtype=torch.long, device=device)
    for p in range(m):
        starts[p] = len(order)
        rows = torch.nonzero(ctp == p, as_tuple=False).reshape(-1)
        counts[p] = int(rows.numel())
        order.extend(int(x) for x in rows.tolist())
    mass = torch.tensor(child_mass if child_mass is not None else [1.0] * n, dtype=torch.float32, device=device)
    return BigGSBranchAssignment(
        branch=branch,
        child_to_parent=ctp,
        child_order=torch.tensor(order, dtype=torch.long, device=device),
        parent_start=starts,
        parent_count=counts,
        child_mass=mass,
        num_children=n,
        num_parents=m,
    )


def _params_from_branch(branch: object) -> dict[str, torch.Tensor]:
    return {
        "means": branch.means,
        "scales_log": branch.scales_log,
        "quats": branch.quats,
        "opacity_logit": branch.opacity_logit,
        "sh_dc": branch.sh_dc,
        "sh_rest": branch.sh_rest,
    }


def test_biggs_assignment_caps_empty_singleton_instance_and_state_to_detach() -> None:
    bg = _node_bg(
        5,
        means=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.01, 0.0, 0.0],
            ]
        ),
    )
    assign = build_biggs_branch_assignment(
        branch="bg",
        means=bg.means,
        scales_log=bg.scales_log,
        opacity_logit=bg.opacity_logit,
        cfg={"voxel_size": 2.0, "max_children_per_parent": 2, "mass_init": "uniform"},
    )
    assert int(assign.parent_count.max().item()) <= 2
    assert int(assign.num_children) == 5

    rigid = _node_rigid(
        4,
        means=torch.zeros((4, 3)),
        point_ids=torch.tensor([[0], [1], [0], [1]], dtype=torch.long),
    )
    _, _, rigid_assign = build_biggs_assignments(
        bg=bg,
        distant=None,
        rigid=rigid,
        assignment_cfg={"rigid": {"voxel_size": 10.0, "max_children_per_parent": 8}},
    )
    assert rigid_assign is not None
    for parent_idx in range(int(rigid_assign.num_parents)):
        start = int(rigid_assign.parent_start[parent_idx].item())
        count = int(rigid_assign.parent_count[parent_idx].item())
        rows = rigid_assign.child_order[start : start + count]
        assert int(rigid.point_ids[rows, 0].unique().numel()) == 1

    empty = build_biggs_branch_assignment(
        branch="bg",
        means=torch.zeros((0, 3)),
        scales_log=torch.zeros((0, 3)),
        opacity_logit=torch.zeros((0, 1)),
        cfg={},
    )
    assert empty.num_children == 0 and empty.num_parents == 0
    singleton = build_biggs_branch_assignment(
        branch="bg",
        means=torch.zeros((1, 3)),
        scales_log=torch.zeros((1, 3)),
        opacity_logit=torch.zeros((1, 1)),
        cfg={},
    )
    state = IForwardBigGSState(bg=singleton)
    detached = state.detach()
    assert detached.bg is not None
    assert detached.bg.child_to_parent.device.type == "cpu"
    assert detached.bg.child_to_parent.data_ptr() != singleton.child_to_parent.data_ptr()
    assert state.to(device=torch.device("cpu")).bg is not None


def test_biggs_vectorized_assignment_covers_children_caps_and_keeps_rigid_instances() -> None:
    means = torch.tensor(
        [
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [1.01, 0.0, 0.0],
            [1.02, 0.0, 0.0],
            [1.03, 0.0, 0.0],
        ]
    )
    bg = _node_bg(6, means=means)
    cfg = {
        "builder": "vectorized_sort_segment",
        "sort_children": "none",
        "voxel_size": 2.0,
        "max_children_per_parent": 2,
        "mass_init": "uniform",
    }
    assign = build_biggs_branch_assignment(
        branch="bg",
        means=bg.means,
        scales_log=bg.scales_log,
        opacity_logit=bg.opacity_logit,
        cfg=cfg,
    )
    assert int(assign.num_children) == 6
    assert sorted(int(x) for x in assign.child_order.tolist()) == list(range(6))
    assert int(assign.parent_count.max().item()) <= 2
    assert int(assign.parent_count.sum().item()) == 6

    rigid = _node_rigid(
        6,
        means=torch.zeros((6, 3)),
        point_ids=torch.tensor([[7], [8], [7], [8], [7], [8]], dtype=torch.long),
    )
    rigid_assign = build_biggs_branch_assignment(
        branch="rigid",
        means=rigid.means,
        scales_log=rigid.scales_log,
        opacity_logit=rigid.opacity_logit,
        cfg=cfg,
        object_id=rigid.point_ids[:, 0],
    )
    for parent_idx in range(int(rigid_assign.num_parents)):
        start = int(rigid_assign.parent_start[parent_idx].item())
        count = int(rigid_assign.parent_count[parent_idx].item())
        rows = rigid_assign.child_order[start : start + count]
        assert int(rigid.point_ids[rows, 0].unique().numel()) == 1


def test_biggs_vectorized_assignment_fast_fails_unsafe_radius() -> None:
    bg = _node_bg(2)
    with pytest.raises(ValueError, match="voxel-size radius control"):
        build_biggs_branch_assignment(
            branch="bg",
            means=bg.means,
            scales_log=bg.scales_log,
            opacity_logit=bg.opacity_logit,
            cfg={
                "builder": "vectorized_sort_segment",
                "sort_children": "none",
                "voxel_size": 1.0,
                "max_parent_radius": 1.0,
                "max_children_per_parent": 2,
            },
        )


def test_biggs_whdd_basis_weighted_zero_mean_and_singleton_zero() -> None:
    means = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assign = build_biggs_branch_assignment(
        branch="bg",
        means=means,
        scales_log=torch.full((3, 3), -2.0),
        opacity_logit=torch.zeros((3, 1)),
        cfg={
            "builder": "vectorized_sort_segment",
            "voxel_size": 10.0,
            "max_children_per_parent": 2,
            "sort_children": "none",
            "build_whdd_basis": True,
            "whdd_basis": {"dtype": "float32", "min_std": 1.0e-4},
        },
    )
    assert assign.child_basis is not None
    for parent in range(int(assign.num_parents)):
        rows = torch.nonzero(assign.child_to_parent == parent, as_tuple=False).reshape(-1)
        basis = assign.child_basis.index_select(0, rows).float()
        weights = assign.child_mass.index_select(0, rows).reshape(-1, 1)
        mean = (basis * weights).sum(dim=0) / weights.sum().clamp_min(1.0e-8)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1.0e-5)
        if int(rows.numel()) == 1:
            assert torch.allclose(basis, torch.zeros_like(basis), atol=1.0e-6)


def test_biggs_whdd_basis_weighted_orthonormal_for_full_rank_parent() -> None:
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assign = build_biggs_branch_assignment(
        branch="bg",
        means=means,
        scales_log=torch.full((4, 3), -2.0),
        opacity_logit=torch.zeros((4, 1)),
        cfg={
            "builder": "vectorized_sort_segment",
            "voxel_size": 10.0,
            "max_children_per_parent": 8,
            "sort_children": "none",
            "build_whdd_basis": True,
            "whdd_basis": {"dtype": "float32", "min_std": 1.0e-4},
        },
    )
    assert assign.child_basis is not None
    assert int(assign.num_parents) == 1
    weights = assign.child_mass.reshape(-1, 1)
    basis = assign.child_basis.float()
    mean = (basis * weights).sum(dim=0) / weights.sum().clamp_min(1.0e-8)
    gram = basis.T @ (basis * weights) / weights.sum().clamp_min(1.0e-8)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1.0e-5)
    assert torch.allclose(gram, torch.eye(3), atol=1.0e-5, rtol=1.0e-5)


def _runtime_for_biggs_assignment_cache() -> MinimalStreetForwardStage6_0:
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.device = torch.device("cpu")
    runtime.sh_degree = 1
    runtime.stage2_0_biggs_assignment_cfg = {
        "builder": "vectorized_sort_segment",
        "sort_children": "none",
        "cache_scope": "scene_segment_topology",
        "ignore_episode_id": True,
        "cache_max_items": 2,
        "cache_device_copy": True,
        "mass_init": "uniform",
        "bg": {"voxel_size": 0.25, "max_children_per_parent": 2, "max_parent_radius": 0.5},
        "rigid": {"voxel_size": 0.25, "max_children_per_parent": 2, "max_parent_radius": 0.5},
    }
    runtime.stage2_0_biggs_assignment_cache_scope = "scene_segment_topology"
    runtime.stage2_0_biggs_assignment_ignore_episode_id = True
    runtime.stage2_0_biggs_assignment_cache_max_items = 2
    runtime.stage2_0_biggs_assignment_cache_device_copy = True
    runtime._stage2_0_biggs_assignment_cache = OrderedDict()
    runtime._stage2_0_biggs_assignment_device_cache = OrderedDict()
    return runtime


def test_biggs_runtime_assignment_cache_hits_across_episode_and_misses_topology_or_cfg() -> None:
    runtime = _runtime_for_biggs_assignment_cache()
    bg = _node_bg(4)
    rigid = _node_rigid(2, point_ids=torch.tensor([[1], [1]], dtype=torch.long))
    first_cpu, first_dev, first_stats = runtime._stage2_0_get_or_build_biggs_state_for_observe(
        existing=None,
        batch={},
        bg=bg,
        distant=None,
        rigid=rigid,
        ids_override=(11, 22, 1),
    )
    assert first_stats["iforward/biggs/assignment_cache_hit"] == 0.0
    assert first_cpu.episode_id == -1
    assert first_dev.bg is not None
    assert first_stats["iforward/biggs/assignment_cache_size"] == 1.0

    second_cpu, _, second_stats = runtime._stage2_0_get_or_build_biggs_state_for_observe(
        existing=None,
        batch={},
        bg=bg,
        distant=None,
        rigid=rigid,
        ids_override=(11, 22, 2),
    )
    assert second_stats["iforward/biggs/assignment_cache_hit"] == 1.0
    assert second_stats["iforward/biggs/assignment_build_ms"] == 0.0
    assert int(second_cpu.bg.num_parents) == int(first_cpu.bg.num_parents)

    bg_changed = _node_bg(5)
    _, _, changed_stats = runtime._stage2_0_get_or_build_biggs_state_for_observe(
        existing=None,
        batch={},
        bg=bg_changed,
        distant=None,
        rigid=rigid,
        ids_override=(11, 22, 3),
    )
    assert changed_stats["iforward/biggs/assignment_cache_hit"] == 0.0

    runtime.stage2_0_biggs_assignment_cfg = dict(runtime.stage2_0_biggs_assignment_cfg)
    runtime.stage2_0_biggs_assignment_cfg["min_child_mass"] = 1.0e-6
    _, _, cfg_stats = runtime._stage2_0_get_or_build_biggs_state_for_observe(
        existing=None,
        batch={},
        bg=bg,
        distant=None,
        rigid=rigid,
        ids_override=(11, 22, 4),
    )
    assert cfg_stats["iforward/biggs/assignment_cache_hit"] == 0.0


def test_biggs_runtime_assignment_cache_lru_eviction() -> None:
    runtime = _runtime_for_biggs_assignment_cache()
    runtime.stage2_0_biggs_assignment_cache_max_items = 1
    bg = _node_bg(4)
    for segment_id in (1, 2):
        runtime._stage2_0_get_or_build_biggs_state_for_observe(
            existing=None,
            batch={},
            bg=bg,
            distant=None,
            rigid=None,
            ids_override=(10, segment_id, 0),
        )
    assert len(runtime._stage2_0_biggs_assignment_cache) == 1
    _, _, stats = runtime._stage2_0_get_or_build_biggs_state_for_observe(
        existing=None,
        batch={},
        bg=bg,
        distant=None,
        rigid=None,
        ids_override=(10, 1, 2),
    )
    assert stats["iforward/biggs/assignment_cache_hit"] == 0.0


def test_biggs_parent_projector_weighted_mean_clamps_opacity_and_sh_shape() -> None:
    branch = _node_bg(
        2,
        means=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        sh_bases=3,
    )
    branch.opacity_logit[:] = 5.0
    assign = _manual_assignment([0, 0], child_mass=[1.0, 3.0])
    proj = project_biggs_parents(
        branch=branch,
        assignment=assign,
        cfg={"min_scale": 0.05, "opacity_cap": 0.7},
        max_scale=0.5,
    )
    assert torch.allclose(proj.params["means"][0], torch.tensor([1.5, 0.0, 0.0]), atol=1.0e-5)
    assert torch.isfinite(proj.params["scales_log"]).all()
    assert bool((torch.exp(proj.params["scales_log"]) <= 0.5 + 1.0e-6).all().item())
    assert float(torch.sigmoid(proj.params["opacity_logit"])[0, 0].item()) <= 0.70001
    assert tuple(proj.params["sh_rest"].shape) == (1, 3, 3)
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    runtime.stage2_0_biggs_projector_cfg = {"max_scale_bg": 0.5}
    stats = runtime._stage2_0_biggs_projection_stats(
        prefix="iforward/biggs/bg",
        projection=proj,
        parent_count=assign.parent_count,
        max_scale=0.5,
    )
    assert stats["iforward/biggs/bg/parent_scale_clip_ratio"] > 0.0


def test_biggs_parent_quat_canonicalizes_positive_w() -> None:
    q = torch.tensor([[-1.0, 0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, 0.5]])
    out = _canonicalize_quat(q)
    assert bool((out[:, 0] >= 0.0).all().item())
    assert torch.allclose(out.norm(dim=-1), torch.ones((2,)), atol=1.0e-6)


def test_biggs_active_rigid_projection_uses_active_child_subset() -> None:
    proj = project_biggs_active_rigid_parents(
        means_world_S=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        quats_world_S=_quat(2),
        scales_log_S=torch.full((2, 3), -2.0),
        opacity_logit_S=torch.zeros((2, 1)),
        sh_dc_S=torch.zeros((2, 3)),
        sh_rest_S=torch.zeros((2, 2, 3)),
        child_to_active_parent_S=torch.tensor([0, 1], dtype=torch.long),
        child_mass_S=torch.ones((2,)),
        active_parent_count=torch.tensor([1, 1], dtype=torch.long),
        cfg={"min_scale": 0.01},
        max_scale=1.0,
    )
    assert tuple(proj.params["means"].shape) == (2, 3)
    assert torch.allclose(proj.params["means"][0], torch.tensor([0.0, 0.0, 0.0]), atol=1.0e-6)
    assert torch.allclose(proj.params["means"][1], torch.tensor([10.0, 0.0, 0.0]), atol=1.0e-6)
    assert not torch.allclose(proj.params["means"][0], proj.params["means"][1])


def _decoder_fixture(event_dim: int = 4) -> tuple[LocalGSState, dict[str, object], EventPack]:
    bg = _node_bg(3)
    distant = _node_distant(2)
    rigid = _node_rigid(3)
    local_state = LocalGSState.from_node_states(bg=bg, distant=distant, rigid=rigid, hidden_dim=3)
    assign_bg = _manual_assignment([0, 0, 1], child_mass=[1.0, 3.0, 1.0], branch="bg")
    assign_bg = replace(
        assign_bg,
        child_basis=torch.tensor([[-3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    assign_distant = _manual_assignment([0, 0], branch="distant")
    assign_distant = replace(
        assign_distant,
        child_basis=torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    active = BigGSRigidActiveAssignment(
        fine_S=torch.tensor([2, 0, 1], dtype=torch.long),
        child_to_active_parent_S=torch.tensor([0, 1, 0], dtype=torch.long),
        active_parent_global=torch.tensor([0, 1], dtype=torch.long),
        active_parent_count=torch.tensor([2, 1], dtype=torch.long),
        active_parent_start=torch.tensor([0, 2], dtype=torch.long),
        active_child_order_S=torch.tensor([0, 2, 1], dtype=torch.long),
        child_mass_S=torch.tensor([1.0, 1.0, 2.0]),
        parent_inside_mask=torch.tensor([True, False]),
        child_inside_mask_S=torch.tensor([True, False, True]),
        child_basis_S=torch.tensor([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    route = SimpleNamespace(
        S=active.fine_S,
        S_in=active.fine_S[active.child_inside_mask_S],
        S_out=active.fine_S[~active.child_inside_mask_S],
        inside_mask_S=active.child_inside_mask_S,
        means_world_S=local_state.rigid.means.index_select(0, active.fine_S),
        quats_world_S=local_state.rigid.quats.index_select(0, active.fine_S),
    )
    parent_bg = _node_bg(2)
    parent_distant = _node_distant(1)
    parent_rigid = _node_bg(2)
    measurement: dict[str, object] = {
        "route": route,
        "assign_bg": assign_bg,
        "assign_distant": assign_distant,
        "assign_rigid_active": active,
        "parent_params_bg": _params_from_branch(parent_bg),
        "parent_params_distant": _params_from_branch(parent_distant),
        "parent_params_rigid_active": _params_from_branch(parent_rigid),
        "parent_mass_mean_bg": torch.tensor([2.0, 1.0]),
        "parent_mass_mean_distant": torch.tensor([1.0]),
        "parent_mass_mean_rigid_active": torch.tensor([1.5, 1.0]),
    }
    parent_event = EventPack(
        event_bg=torch.arange(2 * event_dim, dtype=torch.float32).reshape(2, event_dim),
        event_distant=torch.full((1, event_dim), 10.0),
        event_rigid=(torch.arange(2 * event_dim, dtype=torch.float32).reshape(2, event_dim) + 20.0),
        support_bg=torch.tensor([0.5, 0.25]),
        support_distant=torch.tensor([0.75]),
        support_rigid=torch.tensor([1.0, 0.1]),
        valid_bg=torch.tensor([True, False]),
        valid_distant=torch.tensor([True]),
        valid_rigid=torch.tensor([True, False]),
        obs_code_bg=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        obs_code_distant=torch.tensor([[0.2, 0.3]]),
        obs_code_rigid=torch.tensor([[0.4, 0.5], [0.6, 0.7]]),
        route=SimpleNamespace(S=torch.arange(2), inside_mask_S=torch.tensor([True, False])),
    )
    return local_state, measurement, parent_event


@pytest.mark.parametrize("mode", ["broadcast", "residual_mlp", "low_rank_basis", "whdd_fixed_basis"])
def test_biggs_child_decoder_modes_shapes_and_zero_init_broadcast(mode: str) -> None:
    local_state, measurement, parent_event = _decoder_fixture(event_dim=4)
    decoder = BigGSToFineEventDecoder(
        event_dim=4,
        mode=mode,
        rank=2,
        hidden_dim=8,
        zero_init_last=True,
        residual_scale_learnable=False,
    )
    out = decoder(parent_event_pack=parent_event, local_state=local_state, measurement=measurement)
    assert tuple(out.event_bg.shape) == (3, 4)
    assert tuple(out.event_distant.shape) == (2, 4)
    assert tuple(out.event_rigid.shape) == (3, 4)
    assert out.route is measurement["route"]
    expected_bg = parent_event.event_bg.index_select(0, measurement["assign_bg"].child_to_parent)
    expected_rigid = parent_event.event_rigid.index_select(0, measurement["assign_rigid_active"].child_to_active_parent_S)
    assert torch.allclose(out.event_bg, expected_bg, atol=1.0e-6)
    assert torch.allclose(out.event_rigid, expected_rigid, atol=1.0e-6)
    assert torch.equal(out.valid_bg, torch.tensor([True, True, False]))
    assert torch.equal(out.valid_rigid, torch.tensor([True, False, True]))


def test_biggs_whdd_weighted_mean_preserves_parent_event() -> None:
    local_state, measurement, parent_event = _decoder_fixture(event_dim=4)
    decoder = BigGSToFineEventDecoder(
        event_dim=4,
        mode="whdd_fixed_basis",
        rank=3,
        hidden_dim=8,
        zero_init_last=False,
        residual_scale_init=1.0,
        residual_scale_learnable=False,
    )
    out = decoder(parent_event_pack=parent_event, local_state=local_state, measurement=measurement)
    assign = measurement["assign_bg"]
    parent_bg = parent_event.event_bg.index_select(0, assign.child_to_parent)
    residual = out.event_bg - parent_bg
    weights = assign.child_mass.reshape(-1, 1)
    parent0_mean = (residual[:2] * weights[:2]).sum(dim=0) / weights[:2].sum()
    assert torch.allclose(parent0_mean, torch.zeros_like(parent0_mean), atol=1.0e-5)


def test_biggs_compact_whdd_projects_parent64_to_fine16() -> None:
    local_state, measurement, parent_event = _decoder_fixture(event_dim=64)
    decoder = BigGSToFineEventDecoder(
        event_dim=64,
        parent_event_dim=64,
        fine_event_dim=16,
        mode="whdd_compact_fixed_basis",
        rank=3,
        hidden_dim=32,
        zero_init_last=True,
        residual_scale_init=1.0,
        residual_scale_learnable=False,
    )
    out = decoder(parent_event_pack=parent_event, local_state=local_state, measurement=measurement)
    assert tuple(out.event_bg.shape) == (3, 16)
    assert tuple(out.event_distant.shape) == (2, 16)
    assert tuple(out.event_rigid.shape) == (3, 16)

    loss = out.event_bg.square().mean() + out.event_rigid.square().mean()
    loss.backward()
    base_grad = sum(float(p.grad.detach().abs().sum().item()) for p in decoder.base_proj.parameters() if p.grad is not None)
    detail_grad = sum(float(p.grad.detach().abs().sum().item()) for p in decoder.detail_head.parameters() if p.grad is not None)
    assert base_grad > 0.0
    assert detail_grad > 0.0


def test_biggs_low_rank_decoder_mean_preserves_residual() -> None:
    local_state, measurement, parent_event = _decoder_fixture(event_dim=4)
    decoder = BigGSToFineEventDecoder(
        event_dim=4,
        mode="low_rank_basis",
        rank=2,
        hidden_dim=8,
        zero_init_last=False,
        residual_scale_init=1.0,
        residual_scale_learnable=False,
    )
    out = decoder(parent_event_pack=parent_event, local_state=local_state, measurement=measurement)
    assign = measurement["assign_bg"]
    parent_bg = parent_event.event_bg.index_select(0, assign.child_to_parent)
    residual = out.event_bg - parent_bg
    weights = assign.child_mass.reshape(-1, 1)
    parent0_mean = (residual[:2] * weights[:2]).sum(dim=0) / weights[:2].sum()
    assert torch.allclose(parent0_mean, torch.zeros_like(parent0_mean), atol=1.0e-5)


def test_biggs_parent_stats_incremental_matches_exact_refresh() -> None:
    old = _node_bg(5)
    new = _node_bg(5)
    new.means = new.means + torch.linspace(0.0, 0.04, 5).reshape(-1, 1)
    new.opacity_logit = new.opacity_logit + torch.linspace(-0.02, 0.03, 5).reshape(-1, 1)
    new.sh_dc = new.sh_dc + 0.01
    assign = _manual_assignment([0, 0, 1, 1, 2], child_mass=[1.0, 2.0, 1.0, 1.5, 1.0])
    cfg = {
        "mass_mode": "dynamic_tau_area",
        "min_scale": 1.0e-3,
        "max_scale": 2.0,
        "opacity_cap": 0.9,
        "opacity_min": 1.0e-6,
        "tau_parent_scale": 0.5,
        "eps": 1.0e-6,
        "min_child_mass": 1.0e-8,
    }
    runtime0 = init_parent_branch_runtime(
        params=_params_from_branch(old),
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=2.0,
    )
    runtime1 = update_parent_branch_runtime(
        runtime=runtime0,
        old_params=_params_from_branch(old),
        new_params=_params_from_branch(new),
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=2.0,
    )
    exact = init_parent_branch_runtime(
        params=_params_from_branch(new),
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=2.0,
    )
    for key in ("means", "scales_log", "opacity_logit", "sh_dc", "sh_rest"):
        assert torch.allclose(runtime1.params[key], exact.params[key], atol=2.0e-6, rtol=1.0e-5)


def test_biggs_parent_stats_incremental_matches_exact_with_scale_opacity_mass_shift() -> None:
    old = _node_bg(6)
    new = _node_bg(6)
    new.means = new.means + torch.linspace(-0.03, 0.04, 6).reshape(-1, 1)
    new.scales_log = new.scales_log + torch.tensor(
        [
            [0.03, -0.01, 0.02],
            [-0.02, 0.04, 0.01],
            [0.01, 0.02, -0.03],
            [0.04, -0.02, -0.01],
            [-0.03, 0.01, 0.03],
            [0.02, 0.03, -0.02],
        ],
        dtype=new.scales_log.dtype,
    )
    new.opacity_logit = new.opacity_logit + torch.linspace(-0.2, 0.25, 6).reshape(-1, 1)
    new.quats = torch.nn.functional.normalize(
        old.quats
        + torch.tensor(
            [
                [0.0, 0.02, -0.01, 0.00],
                [0.0, -0.01, 0.03, 0.01],
                [0.0, 0.01, 0.00, -0.02],
                [0.0, -0.02, -0.01, 0.02],
                [0.0, 0.03, 0.01, -0.01],
                [0.0, -0.01, 0.02, 0.03],
            ],
            dtype=old.quats.dtype,
        ),
        dim=-1,
    )
    new.sh_dc = new.sh_dc + torch.linspace(0.0, 0.02, 6).reshape(-1, 1)
    assign = _manual_assignment([0, 0, 0, 1, 1, 1], child_mass=[1.0, 2.0, 0.8, 1.0, 1.5, 0.7])
    cfg = {
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
    runtime0 = init_parent_branch_runtime(
        params=_params_from_branch(old),
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=2.0,
    )
    runtime1 = update_parent_branch_runtime(
        runtime=runtime0,
        old_params=_params_from_branch(old),
        new_params=_params_from_branch(new),
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=2.0,
    )
    exact = init_parent_branch_runtime(
        params=_params_from_branch(new),
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=2.0,
    )
    for key in ("means", "scales_log", "opacity_logit", "sh_dc", "sh_rest"):
        assert torch.allclose(runtime1.params[key], exact.params[key], atol=2.0e-6, rtol=1.0e-5)


def test_biggs_parent_stats_incremental_matches_exact_after_multiple_updates() -> None:
    states = [_node_bg(6) for _ in range(4)]
    for idx, state in enumerate(states[1:], start=1):
        base = states[idx - 1]
        state.means = base.means + (0.01 * idx) * torch.linspace(-1.0, 1.0, 6).reshape(-1, 1)
        state.scales_log = base.scales_log + (0.02 * idx) * torch.tensor(
            [[1.0, -0.5, 0.25], [-0.25, 0.5, 1.0], [0.5, 1.0, -0.5], [-1.0, 0.25, 0.5], [0.25, -1.0, 0.5], [0.5, 0.25, -1.0]],
            dtype=base.scales_log.dtype,
        )
        state.opacity_logit = base.opacity_logit + (0.05 * idx) * torch.linspace(-1.0, 1.0, 6).reshape(-1, 1)
        state.sh_dc = base.sh_dc + 0.005 * idx
    assign = _manual_assignment([0, 0, 0, 1, 1, 1], child_mass=[1.0, 2.0, 0.8, 1.0, 1.5, 0.7])
    cfg = {
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
    runtime = init_parent_branch_runtime(
        params=_params_from_branch(states[0]),
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=2.0,
    )
    for old, new in zip(states[:-1], states[1:]):
        runtime = update_parent_branch_runtime(
            runtime=runtime,
            old_params=_params_from_branch(old),
            new_params=_params_from_branch(new),
            child_to_parent=assign.child_to_parent,
            parent_count=assign.parent_count,
            child_mass=assign.child_mass,
            cfg=cfg,
            max_scale=2.0,
        )
    exact = init_parent_branch_runtime(
        params=_params_from_branch(states[-1]),
        child_to_parent=assign.child_to_parent,
        parent_count=assign.parent_count,
        child_mass=assign.child_mass,
        cfg=cfg,
        max_scale=2.0,
    )
    for key in ("means", "scales_log", "opacity_logit", "sh_dc", "sh_rest"):
        assert torch.allclose(runtime.params[key], exact.params[key], atol=3.0e-6, rtol=1.0e-5)


def test_biggs_low_rank_zero_init_has_nonzero_gradient() -> None:
    local_state, measurement, parent_event = _decoder_fixture(event_dim=4)
    decoder = BigGSToFineEventDecoder(
        event_dim=4,
        mode="low_rank_basis",
        rank=2,
        hidden_dim=8,
        zero_init_last=True,
        residual_scale_init=1.0,
        residual_scale_learnable=False,
    )
    out = decoder(parent_event_pack=parent_event, local_state=local_state, measurement=measurement)
    loss = out.event_bg.square().sum() + out.event_rigid.square().sum()
    loss.backward()
    coeff_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in decoder.coeff_mlp.parameters()
        if p.grad is not None
    )
    basis_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in decoder.basis_mlp.parameters()
        if p.grad is not None
    )
    assert coeff_grad > 0.0
    assert basis_grad == 0.0


def test_biggs_child_code_inputs_detached_by_default() -> None:
    local_state, measurement, parent_event = _decoder_fixture(event_dim=4)
    decoder = BigGSToFineEventDecoder(
        event_dim=4,
        mode="low_rank_basis",
        rank=2,
        hidden_dim=8,
        zero_init_last=False,
        residual_scale_init=1.0,
        residual_scale_learnable=False,
        detach_child_code_inputs=True,
    )
    out = decoder(parent_event_pack=parent_event, local_state=local_state, measurement=measurement)
    out.event_bg.sum().backward()
    assert local_state.bg.means.grad is None
    assert local_state.bg.scales_log.grad is None
    assert local_state.bg.opacity_logit.grad is None


def test_biggs_child_code_uses_parent_local_frame() -> None:
    decoder = BigGSToFineEventDecoder(
        event_dim=4,
        mode="low_rank_basis",
        hidden_dim=8,
        child_code_parent_local_frame=True,
    )
    sqrt_half = 2.0 ** -0.5
    child_params = {
        "means": torch.tensor([[0.0, 1.0, 0.0]]),
        "scales_log": torch.zeros((1, 3)),
        "quats": _quat(1),
        "opacity_logit": torch.zeros((1, 1)),
        "sh_dc": torch.zeros((1, 3)),
        "sh_rest": torch.zeros((1, 2, 3)),
    }
    parent_params = {
        "means": torch.zeros((1, 3)),
        "scales_log": torch.zeros((1, 3)),
        "quats": torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]]),
        "opacity_logit": torch.zeros((1, 1)),
        "sh_dc": torch.zeros((1, 3)),
        "sh_rest": torch.zeros((1, 2, 3)),
    }
    code = decoder._child_code(
        child_params=child_params,
        parent_params=parent_params,
        parent_id=torch.tensor([0]),
        child_mass=torch.ones((1,)),
        parent_count=torch.ones((1,), dtype=torch.long),
        parent_mass_mean=torch.ones((1,)),
        branch_id=0,
        route_flag=None,
    )
    assert torch.allclose(code[0, :3], torch.tensor([1.0, 0.0, 0.0]), atol=1.0e-5)


class _ParentDecoder(nn.Module):
    def forward(
        self,
        *,
        near_in: Stage6StructInput,
        far_in: Stage6StructInput,
        route: object,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        near_batch_offsets: torch.Tensor | None = None,
        far_batch_offsets: torch.Tensor | None = None,
    ) -> EventPack:
        _ = aabb_min, aabb_max, near_batch_offsets, far_batch_offsets
        c = int(near_in.feat_2d.shape[1])
        n_bg = int(near_in.split_0)
        n_ri = int(near_in.split_1)
        n_d = int(far_in.split_0)
        n_ro = int(far_in.split_1)
        event_bg = near_in.feat_2d[:n_bg, :c]
        event_distant = far_in.feat_2d[:n_d, :c] if n_d > 0 else None
        event_rigid = None
        support_rigid = valid_rigid = obs_rigid = None
        if int(route.S.numel()) > 0:
            inside = route.inside_mask_S.to(dtype=torch.bool)
            event_rigid = event_bg.new_zeros((int(route.S.numel()), c))
            support_rigid = near_in.acc_w.new_zeros((int(route.S.numel()),))
            valid_rigid = torch.zeros((int(route.S.numel()),), dtype=torch.bool, device=event_bg.device)
            obs_rigid = near_in.obs_code.new_zeros((int(route.S.numel()), 2))
            if n_ri > 0:
                event_rigid[inside] = near_in.feat_2d[n_bg : n_bg + n_ri, :c]
                support_rigid[inside] = near_in.acc_w[n_bg : n_bg + n_ri]
                valid_rigid[inside] = True
                obs_rigid[inside] = near_in.obs_code[n_bg : n_bg + n_ri]
            if n_ro > 0:
                event_rigid[~inside] = far_in.feat_2d[n_d : n_d + n_ro, :c]
                support_rigid[~inside] = far_in.acc_w[n_d : n_d + n_ro]
                valid_rigid[~inside] = True
                obs_rigid[~inside] = far_in.obs_code[n_d : n_d + n_ro]
        return EventPack(
            event_bg=event_bg,
            event_distant=event_distant,
            event_rigid=event_rigid,
            support_bg=near_in.acc_w[:n_bg],
            support_distant=far_in.acc_w[:n_d] if n_d > 0 else None,
            support_rigid=support_rigid,
            valid_bg=torch.ones((n_bg,), dtype=torch.bool, device=event_bg.device),
            valid_distant=torch.ones((n_d,), dtype=torch.bool, device=event_bg.device) if n_d > 0 else None,
            valid_rigid=valid_rigid,
            obs_code_bg=near_in.obs_code[:n_bg],
            obs_code_distant=far_in.obs_code[:n_d] if n_d > 0 else None,
            obs_code_rigid=obs_rigid,
            route=route,
        )


def _runtime_for_event_builder(event_dim: int = 4) -> MinimalStreetForwardStage6_0:
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.device = torch.device("cpu")
    runtime.stage6_detach_v4_outputs = False
    runtime.stage6_feat_2d_dim = int(event_dim)
    runtime.stage6_near_debug_check_spconv_order = False
    runtime.bg_src_backproject_support_min = 0.0
    runtime.rigid_src_backproject_support_min = 0.0
    runtime.distant_src_backproject_support_min = 0.0
    runtime.bbx_min = torch.full((3,), -10.0)
    runtime.bbx_max = torch.full((3,), 10.0)
    runtime.stage6_struct_event_decoder = _ParentDecoder()
    runtime.biggs_child_decoder = BigGSToFineEventDecoder(event_dim=event_dim, mode="broadcast")
    runtime._phase_b_skip_distant_event = lambda: False
    runtime._mem_debug = lambda *args, **kwargs: None
    runtime._build_struct_batch_offsets = lambda struct_in, device: torch.tensor(
        [int(struct_in.coords.shape[0])], dtype=torch.long, device=device
    )
    return runtime


def test_stage2_0_biggs_event_builder_returns_fine_event_and_updater_consumes() -> None:
    local_state, measurement, _ = _decoder_fixture(event_dim=4)
    measurement.update(
        {
            "biggs_enabled": True,
            "source_frame_idx": 0,
            "parent_feat_2d_bg": torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
            "parent_acc_w_bg": torch.tensor([1.0, 0.5]),
            "parent_obs_bg": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "parent_coords_bg": measurement["parent_params_bg"]["means"],
            "parent_feat_2d_distant": torch.tensor([[3.0, 0.0, 0.0, 0.0]]),
            "parent_acc_w_distant": torch.tensor([0.75]),
            "parent_obs_distant": torch.tensor([[0.5, 0.6]]),
            "parent_coords_distant": measurement["parent_params_distant"]["means"],
            "parent_feat_2d_rigid_S": torch.tensor([[4.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]),
            "parent_acc_w_rigid_S": torch.tensor([0.9, 0.8]),
            "parent_obs_rigid_S": torch.tensor([[0.7, 0.8], [0.9, 1.0]]),
            "parent_coords_rigid_S": measurement["parent_params_rigid_active"]["means"],
            "iforward/biggs/num_parent_total": 5.0,
        }
    )
    runtime = _runtime_for_event_builder(event_dim=4)
    event = runtime._build_stage6_event_from_measurement(local_state=local_state, measurement=measurement)
    assert event.route is measurement["route"]
    assert tuple(event.event_bg.shape) == (3, 4)
    assert tuple(event.event_distant.shape) == (2, 4)
    assert tuple(event.event_rigid.shape) == (3, 4)
    updater = Stage6PosteriorUpdater(event_dim=4, ctx_dim=4, hidden_dim=8, stage_hidden_dim=3, sh_degree=1)
    delta, _ = updater(event=event, ctx_current=None, ctx_vsm=None)
    assert tuple(delta.bg.means.shape) == (3, 3)
    assert tuple(delta.rigid.means.shape) == (3, 3)


def test_stage2_0_zero_hidden_updater_applies_to_local_state() -> None:
    bg = _node_bg(2, sh_bases=0)
    local_state = LocalGSState.from_node_states(bg=bg, distant=None, rigid=None, hidden_dim=0)
    assert tuple(local_state.bg.hidden.shape) == (2, 0)
    updater = Stage6PosteriorUpdater(
        event_dim=16,
        ctx_dim=16,
        hidden_dim=32,
        stage_hidden_dim=0,
        sh_degree=0,
        accept_vsm_ctx=False,
        output_hidden=False,
        output_confidence=False,
        output_noop=True,
    )
    event = EventPack(event_bg=torch.randn(2, 16))
    delta, _ = updater(event=event, ctx_current=None, ctx_vsm=None)
    assert tuple(delta.bg.hidden.shape) == (2, 0)
    assert tuple(delta.bg.confidence.shape) == (2, 0)
    assert tuple(delta.bg.noop.shape) == (2, 1)
    next_state = local_state.apply_delta(delta)
    assert tuple(next_state.bg.hidden.shape) == (2, 0)


def test_stage6_event_builder_disabled_path_uses_legacy_struct_event() -> None:
    runtime = _runtime_for_event_builder(event_dim=4)
    near = Stage6StructInput(
        feat_2d=torch.ones((1, 4)),
        acc_w=torch.ones((1,)),
        obs_code=torch.zeros((1, 2)),
        coords=torch.zeros((1, 3)),
        branch_id=torch.zeros((1,), dtype=torch.long),
        params_for_embed=_params_from_branch(_node_bg(1)),
        split_0=1,
        split_1=0,
        meta={},
    )
    far = Stage6StructInput(
        feat_2d=torch.zeros((0, 4)),
        acc_w=torch.zeros((0,)),
        obs_code=torch.zeros((0, 2)),
        coords=torch.zeros((0, 3)),
        branch_id=torch.zeros((0,), dtype=torch.long),
        params_for_embed={k: v[:0] for k, v in near.params_for_embed.items()},
        split_0=0,
        split_1=0,
        meta={},
    )
    runtime._local_rigid_node_state = lambda local_state: None
    runtime._build_stage6_struct_input_near = lambda **kwargs: near
    runtime._build_stage6_struct_input_far = lambda **kwargs: far
    local_state = LocalGSState.from_node_states(bg=_node_bg(1), distant=None, rigid=None, hidden_dim=3)
    route = SimpleNamespace(S=torch.zeros((0,), dtype=torch.long), inside_mask_S=torch.zeros((0,), dtype=torch.bool))
    event = runtime._build_stage6_event_from_measurement(
        local_state=local_state,
        measurement={"route": route, "source_frame_idx": 0, "feat_2d_bg": torch.ones((1, 4))},
    )
    assert tuple(event.event_bg.shape) == (1, 4)
    assert event.route is route


def test_stage2_0_parent_scene_for_cnn_false_renders_fine_scene_lifts_parent() -> None:
    device = torch.device("cpu")
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.device = device
    runtime.sh_degree = 1
    runtime.stage2_0_biggs_assignment_cfg = {"bg": {"voxel_size": 10.0, "max_children_per_parent": 2}}
    runtime.stage2_0_biggs_projector_cfg = {"min_scale": 0.01}
    runtime.stage2_0_biggs_observe_cfg = {"parent_scene_for_cnn": False}
    runtime.stage2_0_biggs_return_debug_stats = False
    runtime._stage5_4_obs_code_all = None
    runtime._mem_debug = lambda *args, **kwargs: None
    runtime._source_subset = lambda batch, indices: (
        [object()],
        [torch.zeros((3, 4, 4), device=device)],
        [torch.zeros((1, 4, 4), dtype=torch.bool, device=device)],
        [torch.zeros((1, 4, 4), dtype=torch.bool, device=device)],
    )
    render_rows = []
    lift_rows = []

    def _render(**kwargs):
        render_rows.append(int(kwargs["gaussians_scene"]["means"].shape[0]))
        return {
            "features_2d": torch.zeros((1, 4, 4, 4), device=device),
            "source_pair_valid_mask": torch.ones((1,), dtype=torch.bool, device=device),
        }

    def _backproject(**kwargs):
        m = int(kwargs["gaussians_scene"]["means"].shape[0])
        lift_rows.append(m)
        runtime._stage5_4_obs_code_all = torch.zeros((m, 2), device=device)
        return torch.ones((m, 4), device=device), torch.ones((m,), device=device)

    runtime._render_source_scene_only_for_cnn = _render
    runtime._backproject_scene_features_multi_camera = _backproject
    local_state = LocalGSState.from_node_states(bg=_node_bg(4, device=device, sh_bases=3), distant=None, rigid=None, hidden_dim=3)
    measurement = runtime._observe_stage2_0_biggs_measurement(
        local_state=local_state,
        batch={"scene_id": 1, "segment_id": 1},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=None,
    )
    assert render_rows == [4]
    assert lift_rows == [2]
    assert tuple(measurement["parent_feat_2d_bg"].shape) == (2, 4)
    assert measurement["iforward/biggs/parent_scene_for_cnn"] == 0.0


class _DetachOnly:
    def detach(self) -> "_DetachOnly":
        return self


def test_iforward_state_detach_keeps_biggs_assignment() -> None:
    local_state = LocalGSState.from_node_states(bg=_node_bg(1), distant=None, rigid=None, hidden_dim=3)
    biggs = IForwardBigGSState(bg=_manual_assignment([0]))
    state = IForwardState(
        local_gs=local_state,
        memory=_DetachOnly(),
        history=_DetachOnly(),
        scene_id=1,
        segment_id=2,
        episode_id=3,
        biggs_state=biggs,
    )
    out = state.detach_for_next_rollout()
    assert out.biggs_state is not None and out.biggs_state.bg is not None
    assert int(out.biggs_state.bg.num_children) == 1
    assert out.biggs_state.bg.child_to_parent.data_ptr() != biggs.bg.child_to_parent.data_ptr()


class _BranchMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point = nn.Linear(2, 2)
        self.fuse = nn.Linear(2, 2)


class _Memory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bg = _BranchMemory()


class _Updater(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(2, 2)
        self.vsm_ctx_adapter = nn.Linear(2, 2)


class _Runtime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage6_posterior_updater = _Updater()
        self.stage6_struct_event_decoder = nn.Linear(2, 2)
        self.biggs_child_decoder = BigGSToFineEventDecoder(event_dim=2, mode="low_rank_basis", hidden_dim=4)
        self.stage6_measurement_trainable_param_names = set()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory = _Memory()
        self.phase_a_runtime = _Runtime()


class _Stage2NoMemoryModel(nn.Module):
    is_stage2_0_biggs_parent_lifting = True

    def __init__(self) -> None:
        super().__init__()
        self.phase_a_runtime = _Runtime()


def _iforward_trainer_cfg(*, train_biggs: bool = True) -> dict[str, object]:
    return {
        "model": {
            "iforward": {
                "version": "v1",
                "trainability": {
                    "train_memory": True,
                    "train_memory_fuse": True,
                    "train_vsm_ctx_adapter": True,
                    "unfreeze_updater_base_after_step": 0,
                    "train_stage6_struct_decoder": True,
                    "unfreeze_struct_decoder_after_step": 0,
                    "train_biggs_child_decoder": bool(train_biggs),
                }
            }
        },
        "optimizer": {
            "type": "adamw",
            "lr": {
                "default": 1.0e-4,
                "stage6_struct_decoder": 1.0e-5,
                "stage6_posterior_updater_base": 1.0e-5,
                "biggs_child_decoder": 7.0e-5,
            },
            "weight_decay": 0.0,
        },
    }


def _group(trainer: IForwardTrainer, name: str) -> dict[str, object]:
    for group in trainer.optimizer.param_groups:
        if group.get("name") == name:
            return group
    raise AssertionError(f"missing optimizer group {name}")


def test_iforward_trainer_biggs_child_decoder_group_and_trainability() -> None:
    trainer = IForwardTrainer(config=_iforward_trainer_cfg(train_biggs=True), device=torch.device("cpu"), model=_Model())
    group = _group(trainer, "biggs_child_decoder")
    assert float(group["lr"]) == 7.0e-5
    assert any(p.requires_grad for p in group["params"])

    frozen = IForwardTrainer(config=_iforward_trainer_cfg(train_biggs=False), device=torch.device("cpu"), model=_Model())
    frozen_group = _group(frozen, "biggs_child_decoder")
    assert float(frozen_group["lr"]) == 0.0
    assert not any(p.requires_grad for p in frozen_group["params"])


def test_stage2_0_biggs_trainer_groups_do_not_require_fine_memory() -> None:
    cfg = _iforward_trainer_cfg(train_biggs=True)
    cfg["model"]["iforward"]["version"] = "stage2_0_biggs_parent_lifting"
    trainer = IForwardTrainer(config=cfg, device=torch.device("cpu"), model=_Stage2NoMemoryModel())
    names = {str(group.get("name")) for group in trainer.optimizer.param_groups}
    assert "memory" not in names
    assert "memory_fuse" not in names
    assert "biggs_child_decoder" in names


def test_stage2_0_cuda_exact_diag_version_uses_biggs_optimizer_groups() -> None:
    cfg = _iforward_trainer_cfg(train_biggs=True)
    cfg["model"]["iforward"]["version"] = "stage2_0_biggs_cuda_exact_diagonal_projector"
    trainer = IForwardTrainer(config=cfg, device=torch.device("cpu"), model=_Model())
    names = {str(group.get("name")) for group in trainer.optimizer.param_groups}
    assert "memory" not in names
    assert "memory_fuse" not in names
    assert "biggs_child_decoder" in names


@pytest.mark.parametrize(
    ("iforward_overrides", "message"),
    [
        ({"biggs": {"enable": False}}, "biggs.enable=true"),
        ({"history_gate": {"enable": True}}, "history_gate.enable=false"),
        ({"history_gate_v2": {"enable": True}}, "history_gate_v2.enable=false"),
        ({"adc_lite": {"enable": True}}, "adc_lite.enable=false"),
        ({"biggs": {"enable": True, "observe": {"parent_scene_for_lifting": False}}}, "parent_scene_for_lifting=true"),
        (
            {"biggs": {"enable": True, "child_observation_skip": {"enable": True, "trainable": True}}},
            "forbids trainable child_observation_skip",
        ),
    ],
)
def test_stage2_0_biggs_config_conflicts_fast_fail(iforward_overrides: dict[str, object], message: str) -> None:
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    iforward = {
        "version": "stage2_0_biggs_parent_lifting",
        "biggs": {"enable": True},
    }
    for key, value in iforward_overrides.items():
        if isinstance(value, dict) and isinstance(iforward.get(key), dict):
            merged = dict(iforward[key])
            merged.update(value)
            iforward[key] = merged
        else:
            iforward[key] = value
    cfg = {
        "model": {
            "stage": "6_0",
            "phase": "phase_A_block_local_unroll",
            "iforward": iforward,
        }
    }
    with pytest.raises(ValueError, match=message):
        runtime._validate_stage6_0_phase_a_config(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stage2_0_biggs_observe_cuda_parent_shape() -> None:
    device = torch.device("cuda")
    n = 128
    c = 6
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.device = device
    runtime.sh_degree = 1
    runtime.stage2_0_biggs_assignment_cfg = {"bg": {"voxel_size": 0.01, "max_children_per_parent": 1}}
    runtime.stage2_0_biggs_projector_cfg = {}
    runtime._stage5_4_obs_code_all = None
    runtime._source_subset = lambda batch, indices: (
        [object()],
        [torch.zeros((3, 4, 4), device=device)],
        [torch.zeros((1, 4, 4), dtype=torch.bool, device=device)],
        [torch.zeros((1, 4, 4), dtype=torch.bool, device=device)],
    )
    runtime._render_source_scene_only_for_cnn = lambda **kwargs: {
        "features_2d": torch.zeros((1, c, 4, 4), device=device),
        "source_pair_valid_mask": torch.ones((1,), dtype=torch.bool, device=device),
    }

    def _backproject(**kwargs):
        m = int(kwargs["gaussians_scene"]["means"].shape[0])
        runtime._stage5_4_obs_code_all = torch.zeros((m, 2), device=device)
        return torch.ones((m, c), device=device), torch.ones((m,), device=device)

    runtime._backproject_scene_features_multi_camera = _backproject
    local_state = LocalGSState.from_node_states(bg=_node_bg(n, device=device), distant=None, rigid=None, hidden_dim=3)
    measurement = runtime._observe_stage2_0_biggs_measurement(
        local_state=local_state,
        batch={"scene_id": 1, "segment_id": 1},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=None,
    )
    assert tuple(measurement["parent_feat_2d_bg"].shape) == (n, c)
    assert tuple(measurement["parent_acc_w_bg"].shape) == (n,)
    assert tuple(measurement["parent_obs_bg"].shape) == (n, 2)
