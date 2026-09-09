from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from models.iforward.observation_feedback import ObservationFeedbackPolicy
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.node_states import NodeStateBackground, NodeStateRigid
from models.streetforward.stage6_0 import LocalGSState


def _quat(n: int, *, device: torch.device) -> torch.Tensor:
    quat = torch.zeros((int(n), 4), dtype=torch.float32, device=device)
    quat[:, 0] = 1.0
    return quat


def _node_bg(n: int, *, device: torch.device, sh_bases: int = 3) -> NodeStateBackground:
    means = torch.arange(int(n), dtype=torch.float32, device=device).reshape(-1, 1)
    means = torch.cat((means * 0.12, means * 0.03, means * -0.02), dim=1)
    return NodeStateBackground(
        means=means,
        scales_log=torch.full((n, 3), -2.0, dtype=torch.float32, device=device),
        quats=_quat(n, device=device),
        opacity_logit=torch.full((n, 1), -1.5, dtype=torch.float32, device=device),
        sh_dc=torch.linspace(-0.25, 0.25, max(n * 3, 1), device=device).reshape(n, 3),
        sh_rest=torch.zeros((n, int(sh_bases), 3), dtype=torch.float32, device=device),
    )


def _node_rigid(n: int, *, device: torch.device, sh_bases: int = 3) -> NodeStateRigid:
    bg = _node_bg(n, device=device, sh_bases=sh_bases)
    return NodeStateRigid(
        means=bg.means + torch.tensor([0.2, 0.1, 0.0], device=device),
        scales_log=bg.scales_log,
        quats=bg.quats,
        opacity_logit=bg.opacity_logit,
        sh_dc=bg.sh_dc,
        sh_rest=bg.sh_rest,
        point_ids=torch.zeros((n, 1), dtype=torch.long, device=device),
        instances_quats=_quat(1, device=device).reshape(1, 1, 4),
        instances_trans=torch.zeros((1, 1, 3), dtype=torch.float32, device=device),
        instances_fv=torch.ones((1, 1), dtype=torch.bool, device=device),
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )


class _RaiseIfCalled(nn.Module):
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover - failure path
        raise AssertionError("the functional direct-lift observe path must not call sparse parent modules")


def _stage3_4_feedback_policy(
    *,
    source_feedback: bool,
    source_alpha: float = 1.0,
    functional_parent_alpha: float = 1.0,
) -> ObservationFeedbackPolicy:
    return ObservationFeedbackPolicy.from_config(
        {
            "enable": True,
            "scope": "within_rollout",
            "schedule": {"origin": "global_step", "activation_step": 0},
            "modes": {
                "repeat_refine": "trainable_checkpointed",
                "shuffled_coverage": "trainable_checkpointed",
                "high_block_repair": "trainable_checkpointed",
            },
            "source_render": {
                "enable": bool(source_feedback),
                "renderer_mode": "differentiable_rgb",
                "checkpoint_scope": "full_dynamic_observation",
                "absgrad": False,
                "alpha_schedule": [[0, float(source_alpha)]],
            },
            "functional_parent": {
                "enable": True,
                "branches": ["bg", "distant", "rigid_active"],
                "start_after_model_updates": 1,
                "alpha_schedule": [[0, float(functional_parent_alpha)]],
            },
            "parent_projection": {"enable": False, "alpha_schedule": [[0, 0.0]]},
            "relation": {"enable": False, "alpha_schedule": [[0, 0.0]]},
            "scalar_anchor": {"geometry_grad": False},
            "discrete_routing_grad": False,
            "rollout_boundary_grad": False,
            "debug": {
                "grad_probe_interval": 0,
                "forward_parity_interval": 0,
                "log_feedback_memory": False,
            },
        }
    )


def _make_stage3_4_runtime(
    device: torch.device,
    *,
    source_feedback: bool = False,
) -> tuple[MinimalStreetForwardStage6_0, dict[str, list[Any]]]:
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.device = device
    runtime.sh_degree = 1
    runtime.bbx_min = torch.full((3,), -10.0, dtype=torch.float32, device=device)
    runtime.bbx_max = torch.full((3,), 10.0, dtype=torch.float32, device=device)
    runtime.stage3_4_functional_parentgs_enabled = True
    runtime.stage2_0_biggs_assignment_cfg = {
        "builder": "vectorized_sort_segment",
        "sort_children": "none",
        "mass_init": "uniform",
        "bg": {"voxel_size": 10.0, "max_children_per_parent": 2, "max_parent_radius": 20.0},
        "rigid": {"voxel_size": 10.0, "max_children_per_parent": 2, "max_parent_radius": 20.0},
    }
    runtime.stage2_0_biggs_assignment_cache_scope = "scene_segment_topology"
    runtime.stage2_0_biggs_assignment_ignore_episode_id = True
    runtime.stage2_0_biggs_assignment_cache_max_items = 2
    runtime.stage2_0_biggs_assignment_cache_device_copy = True
    runtime.stage2_0_biggs_projector_cfg = {
        "backend": "cuda_exact_diag",
        "covariance_mode": "diagonal",
        "mass_mode": "dynamic_tau_area",
        "recompute_every_visit": True,
        "grad_to_local_state": True,
        "allow_cpu_fallback": False,
        "allow_torch_fallback": False,
        "min_scale": 0.01,
        "max_scale_bg": 1.0,
        "max_scale_distant": 3.0,
        "max_scale_rigid": 1.0,
        "opacity_cap": 0.9,
        "opacity_min": 1.0e-6,
        "tau_parent_scale_bg": 0.5,
        "tau_parent_scale_distant": 0.7,
        "tau_parent_scale_rigid": 0.5,
        "eps": 1.0e-6,
        "min_child_mass": 1.0e-8,
        "finite_check": True,
        "stats_interval": 1,
    }
    runtime.stage2_0_biggs_parent_state_cfg = {"mode": "functional_per_visit"}
    runtime.stage2_0_biggs_observe_cfg = {
        "parent_scene_for_cnn": False,
        "return_debug_stats": False,
    }
    runtime.stage2_0_biggs_lifting_cfg = {}
    runtime.stage2_0_biggs_return_debug_stats = False
    runtime.iforward_repair_training_cfg = {}
    runtime.stage3_0_enabled = True
    runtime.stage3_0_global_step = 0
    runtime.stage3_0_lifting_cfg = {
        "type": "full_sparse_gather",
        "scalar_anchor_backend": "projected_meta",
        "context_dim": 4,
        "detail_dim": 2,
        "detach_geometry": True,
        "gather_aux_interval": 0,
        "memory_aux_interval": 0,
        "return_stage3_debug_tensors": False,
        "parent": {
            "type": "functional_parent_direct_lift",
            "color_mode": "constant_zero",
            "geometry_grad": False,
        },
        "scalar_anchor": {"support_threshold": {"child": 1.0e-4, "parent": 1.0e-4}},
        "child_gather": {
            "type": "support_center",
            "backend": "pytorch",
            "center_by_parent": True,
            "fixed_center_chunk_size": 32,
        },
    }
    runtime.stage3_parent_lifting_type = "functional_parent_direct_lift"
    runtime.stage3_parent_query = _RaiseIfCalled().to(device)
    runtime.stage3_parent_gather = _RaiseIfCalled().to(device)
    runtime.stage3_parent_context_fusion = _RaiseIfCalled().to(device)
    runtime._stage5_4_obs_code_all = None
    runtime._mem_debug = lambda *args, **kwargs: None
    runtime._register_observation_feedback_grad_probe = lambda *args, **kwargs: None

    runtime.observation_feedback_policy = _stage3_4_feedback_policy(
        source_feedback=source_feedback
    )
    runtime._observation_feedback_probe_modes_current = set()
    runtime._observation_feedback_probe_modes_seen = set()
    runtime._observation_feedback_force_probe_current = False
    runtime._observation_feedback_parity_steps_seen = set()

    captures: dict[str, list[Any]] = {
        "render": [],
        "features": [],
        "backproject": [],
        "meta": [],
    }
    runtime._source_subset = lambda batch, indices: (
        [object()],
        [torch.zeros((3, 4, 4), dtype=torch.float32, device=device)],
        [torch.zeros((1, 4, 4), dtype=torch.bool, device=device)],
        [torch.zeros((1, 4, 4), dtype=torch.bool, device=device)],
    )

    def _render_source_scene_only_for_cnn(**kwargs: Any) -> dict[str, torch.Tensor]:
        captures["render"].append(kwargs)
        features = torch.arange(4 * 4 * 4, dtype=torch.float32, device=device).reshape(1, 4, 4, 4)
        features = features.detach().clone().requires_grad_(True)
        if bool(kwargs.get("feedback_enabled", False)):
            # Model the intended checkpointed frontend dependency without invoking a renderer.
            features = features + kwargs["gaussians_scene"]["means"].sum() * 1.0e-3
        captures["features"].append(features)
        return {
            "features_2d": features,
            "fwhr_detail_2d": torch.ones((1, 4, 4, 2), dtype=torch.float32, device=device),
            "source_pair_valid_mask": torch.ones((1, 4, 4), dtype=torch.bool, device=device),
        }

    runtime._render_source_scene_only_for_cnn = _render_source_scene_only_for_cnn

    def _meta_builder(**kwargs: Any) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        captures["meta"].append(kwargs)
        for value in kwargs["gaussians"].values():
            if torch.is_tensor(value):
                _assert_graph_free(value)
        n = int(kwargs["gaussians"]["means"].shape[0])
        row = torch.arange(n, dtype=torch.float32, device=device)
        uv = torch.stack((row.remainder(3) + 0.5, row.div(3, rounding_mode="floor").remainder(3) + 0.5), dim=-1)
        return {
            "means2d": uv,
            "gaussian_ids": torch.arange(n, dtype=torch.long, device=device),
            "camera_ids": torch.zeros((n,), dtype=torch.long, device=device),
            "opacities": torch.ones((n,), dtype=torch.float32, device=device),
            "depths": torch.ones((n,), dtype=torch.float32, device=device),
            "radii": torch.ones((n,), dtype=torch.float32, device=device),
            "conics": torch.ones((n, 3), dtype=torch.float32, device=device),
        }, {"packed_rows": float(n)}

    runtime.alpha_t_extractor_v4 = SimpleNamespace(_build_multi_camera_meta_from_views=_meta_builder)

    def _parent_backproject(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        captures["backproject"].append(kwargs)
        scene = kwargs["gaussians_scene"]
        for name in ("means", "scales", "quats", "opacities", "colors"):
            value = scene[name]
            assert not value.requires_grad
            assert value.grad_fn is None
        assert torch.count_nonzero(scene["colors"]) == 0
        features = kwargs["features_2d"]
        pooled = features.mean(dim=(0, 1, 2))
        n = int(scene["means"].shape[0])
        return pooled.reshape(1, -1).expand(n, -1), torch.ones((n,), dtype=torch.float32, device=device)

    runtime._backproject_scene_features_multi_camera = _parent_backproject
    return runtime, captures


def _make_local_state(device: torch.device, *, with_rigid: bool) -> LocalGSState:
    return LocalGSState.from_node_states(
        bg=_node_bg(4, device=device),
        distant=None,
        rigid=_node_rigid(4, device=device) if with_rigid else None,
        hidden_dim=3,
    )


def _assert_graph_free(value: torch.Tensor) -> None:
    assert not value.requires_grad
    assert value.grad_fn is None


def _assert_assignment_graph_free(assignment: Any) -> None:
    for value in assignment.__dict__.values():
        if torch.is_tensor(value):
            _assert_graph_free(value)


def _assert_no_legacy_parent_runtime_diagnostics(measurement: dict[str, Any]) -> None:
    assert "biggs_parent_runtime" not in measurement
    for key in measurement:
        key_text = str(key)
        assert "runtime_update" not in key_text
        assert "incremental_update" not in key_text
        assert "exact_refresh" not in key_text
        assert "/drift" not in key_text
        if "parent_runtime" in key_text:
            assert key_text == "iforward/stage3_4/parent_runtime_enabled"


_CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def test_observe_wrapper_never_reopens_external_or_validation_no_grad() -> None:
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.stage2_0_biggs_enabled = True
    runtime.stage6_source_evidence_grad_mode = "checkpointed_full"
    runtime.observation_feedback_policy = _stage3_4_feedback_policy(source_feedback=True)
    observed_grad_modes: list[bool] = []

    def capture(**_: Any) -> dict[str, bool]:
        observed_grad_modes.append(bool(torch.is_grad_enabled()))
        return {"grad_enabled": bool(torch.is_grad_enabled())}

    runtime._observe_stage2_0_biggs_measurement = capture  # type: ignore[method-assign]
    kwargs = {
        "local_state": None,
        "batch": {},
        "source_indices": [],
        "source_frame_idx": 0,
        "visit_meta": {
            "global_step": 15000,
            "train_2d_mode": "trainable_checkpointed",
            "model_update_count": 1,
            "has_update_ancestor": True,
        },
    }
    with torch.no_grad():
        assert runtime._observe_v4_measurement(**kwargs)["grad_enabled"] is False
    validation_kwargs = dict(kwargs)
    validation_kwargs["visit_meta"] = {
        **kwargs["visit_meta"],
        "validation_render_only": True,
    }
    assert runtime._observe_v4_measurement(**validation_kwargs)["grad_enabled"] is False
    assert observed_grad_modes == [False, False]


@_CUDA_ONLY
def test_stage3_4_observe_k2_functional_pack_live_geometry_and_lift_isolation() -> None:
    device = torch.device("cuda")
    runtime, captures = _make_stage3_4_runtime(device)
    initial = _make_local_state(device, with_rigid=True)

    first = runtime._observe_stage2_0_biggs_measurement(
        local_state=initial,
        batch={"scene_id": 7, "segment_id": 11},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=None,
        visit_meta={
            "global_step": 0,
            "train_2d_mode": "trainable_checkpointed",
            "model_update_count": 0,
            "has_update_ancestor": False,
        },
    )

    assert "functional_parent_pack" in first
    assert "functional_parent_assignments" in first
    _assert_no_legacy_parent_runtime_diagnostics(first)
    first_pack = first["functional_parent_pack"]
    assert [branch.branch_name for branch in first_pack.iter_branches()] == ["bg", "rigid_active"]
    for branch in first_pack.iter_branches():
        for tensor in branch.projection.params.values():
            _assert_graph_free(tensor)
        _assert_graph_free(branch.child_stats_detached.mass)
        _assert_graph_free(branch.child_stats_detached.tau_area)
        _assert_graph_free(branch.child_stats_detached.diag_cov)
        _assert_graph_free(branch.parent_mass_mean)
        _assert_assignment_graph_free(branch.assignment)

    _assert_graph_free(first["parent_coords_bg"])
    _assert_graph_free(first["parent_coords_rigid_S"])
    _assert_graph_free(first["parent_acc_w_bg"])
    _assert_graph_free(first["parent_acc_w_rigid_S"])
    _assert_graph_free(first["parent_mass_mean_bg"])
    _assert_graph_free(first["parent_mass_mean_rigid_active"])
    _assert_assignment_graph_free(first["assign_bg"])
    _assert_assignment_graph_free(first["assign_rigid_active"])
    _assert_assignment_graph_free(first["functional_parent_assignments"].bg)
    _assert_assignment_graph_free(first["functional_parent_assignments"].rigid_active)
    for value in first["route"].__dict__.values():
        if torch.is_tensor(value):
            _assert_graph_free(value)

    expected_sentinels = {
        "iforward/stage3_4/enabled": 1.0,
        "iforward/stage3_4/functional_parent_enabled": 1.0,
        "iforward/stage3_4/parent_runtime_enabled": 0.0,
        "iforward/stage3_4/surrogate_vjp_enabled": 0.0,
        "iforward/stage3_4/relation_feedback_enabled": 0.0,
        "iforward/stage3_4/parent_lift_geometry_grad": 0.0,
        "feedback/parent_lift/boundary_assertion_passed": 1.0,
        "iforward/stage3_4/functional_parent_direct_lift_enabled": 1.0,
        "iforward/stage3_4/lift_geometry_grad_enabled": 0.0,
        "iforward/stage3_4/first_visit_forward_only": 1.0,
        "iforward/stage3_4/has_update_ancestor": 0.0,
        "feedback/functional_parent/geometry_alpha": 1.0,
        "feedback/functional_parent/grad_active": 0.0,
        "feedback/functional_parent/forward_only": 1.0,
        "feedback/functional_parent/validation_render_only": 0.0,
    }
    for key, expected in expected_sentinels.items():
        assert first[key] == expected
    for branch_name in ("bg", "rigid_active"):
        for suffix in (
            "num_children",
            "num_parents",
            "project_ms",
            "lift_ms",
            "parent_scale_clamp_ratio",
            "parent_opacity_cap_ratio",
            "parent_support_mean",
        ):
            assert f"iforward/stage3_4/{branch_name}/{suffix}" in first
    bg_means_delta = torch.full_like(initial.bg.means, 0.01, requires_grad=True)
    bg_scales_delta = torch.full_like(initial.bg.scales_log, 0.005, requires_grad=True)
    bg_opacity_delta = torch.full_like(initial.bg.opacity_logit, 0.01, requires_grad=True)
    assert initial.rigid is not None
    rigid_means_delta = torch.full_like(initial.rigid.means, -0.01, requires_grad=True)
    updated = LocalGSState(
        bg=replace(
            initial.bg,
            means=initial.bg.means + bg_means_delta,
            scales_log=initial.bg.scales_log + bg_scales_delta,
            opacity_logit=initial.bg.opacity_logit + bg_opacity_delta,
        ),
        distant=None,
        rigid=replace(initial.rigid, means=initial.rigid.means + rigid_means_delta),
        rigid_template=initial.rigid_template,
    )
    second = runtime._observe_stage2_0_biggs_measurement(
        local_state=updated,
        batch={"scene_id": 7, "segment_id": 11},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=first["biggs_state"],
        visit_meta={
            "global_step": 0,
            "train_2d_mode": "trainable_checkpointed",
            "model_update_count": 1,
            "has_update_ancestor": True,
        },
    )

    _assert_no_legacy_parent_runtime_diagnostics(second)
    assert second["iforward/stage3_4/first_visit_forward_only"] == 0.0
    assert second["iforward/stage3_4/has_update_ancestor"] == 1.0
    assert second["feedback/functional_parent/grad_active"] == 1.0
    assert second["feedback/functional_parent/forward_only"] == 0.0
    assert second["iforward/biggs/assignment_cache_hit"] == 1.0
    second_pack = second["functional_parent_pack"]
    for key in ("means", "scales_log", "quats", "opacity_logit"):
        assert second_pack.bg.projection.params[key].requires_grad
    assert second_pack.rigid_active is not None
    for key in ("means", "scales_log", "quats", "opacity_logit"):
        assert second_pack.rigid_active.projection.params[key].requires_grad

    geometry_loss = (
        second["parent_params_bg"]["means"].square().sum()
        + second["parent_params_bg"]["scales_log"].square().sum()
        + second["parent_params_bg"]["opacity_logit"].square().sum()
        + second["parent_params_rigid_active"]["means"].square().sum()
    )
    geometry_grads = torch.autograd.grad(
        geometry_loss,
        (bg_means_delta, bg_scales_delta, bg_opacity_delta, rigid_means_delta),
        retain_graph=True,
        allow_unused=True,
    )
    for grad in geometry_grads:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0

    # Parent lifting is differentiable to the frontend feature map, but its detached
    # scene must not introduce a second geometry path back to live LocalGS.
    lift_loss = second["parent_feat_2d_bg"].sum() + second["parent_feat_2d_rigid_S"].sum()
    lift_grads = torch.autograd.grad(
        lift_loss,
        (captures["features"][-1], updated.bg.means, updated.rigid.means),
        retain_graph=True,
        allow_unused=True,
    )
    assert lift_grads[0] is not None and torch.count_nonzero(lift_grads[0]) > 0
    assert lift_grads[1] is None
    assert lift_grads[2] is None

    for key in (
        "parent_coords_bg",
        "parent_coords_rigid_S",
        "parent_acc_w_bg",
        "parent_acc_w_rigid_S",
        "parent_mass_mean_bg",
        "parent_mass_mean_rigid_active",
    ):
        _assert_graph_free(second[key])
    for branch in second_pack.iter_branches():
        _assert_graph_free(branch.child_stats_detached.mass)
        _assert_graph_free(branch.child_stats_detached.diag_cov)
        _assert_assignment_graph_free(branch.assignment)
    _assert_assignment_graph_free(second["functional_parent_assignments"].bg)
    _assert_assignment_graph_free(second["functional_parent_assignments"].rigid_active)
    assert len(captures["backproject"]) == 2


@_CUDA_ONLY
def test_stage3_4_branch_sparse_update_keeps_unchanged_rigid_forward_only() -> None:
    """A global update ancestor does not imply every carried branch has a graph."""

    device = torch.device("cuda")
    runtime, _ = _make_stage3_4_runtime(device)
    initial = _make_local_state(device, with_rigid=True)
    first = runtime._observe_stage2_0_biggs_measurement(
        local_state=initial,
        batch={"scene_id": 37, "segment_id": 41},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=None,
        visit_meta={
            "global_step": 15000,
            "train_2d_mode": "trainable_checkpointed",
            "model_update_count": 0,
            "has_update_ancestor": False,
        },
    )

    # Model the next rollout boundary followed by a bg-only updater delta.
    bg_delta = torch.full_like(initial.bg.means, 0.01, requires_grad=True)
    carried = LocalGSState(
        bg=replace(
            initial.bg,
            means=initial.bg.means.detach().clone() + bg_delta,
            scales_log=initial.bg.scales_log.detach().clone(),
            quats=initial.bg.quats.detach().clone(),
            opacity_logit=initial.bg.opacity_logit.detach().clone(),
            sh_dc=initial.bg.sh_dc.detach().clone(),
            sh_rest=initial.bg.sh_rest.detach().clone(),
            hidden=initial.bg.hidden.detach().clone(),
        ),
        distant=None,
        rigid=replace(
            initial.rigid,
            means=initial.rigid.means.detach().clone(),
            scales_log=initial.rigid.scales_log.detach().clone(),
            quats=initial.rigid.quats.detach().clone(),
            opacity_logit=initial.rigid.opacity_logit.detach().clone(),
            sh_dc=initial.rigid.sh_dc.detach().clone(),
            sh_rest=initial.rigid.sh_rest.detach().clone(),
            hidden=initial.rigid.hidden.detach().clone(),
        ),
        rigid_template=initial.rigid_template.detach_clone(),
    )
    second = runtime._observe_stage2_0_biggs_measurement(
        local_state=carried,
        batch={"scene_id": 37, "segment_id": 41},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=first["biggs_state"],
        visit_meta={
            "global_step": 15000,
            "train_2d_mode": "trainable_checkpointed",
            "model_update_count": 1,
            "has_update_ancestor": True,
        },
    )

    assert second["functional_parent_grad_active"] is True
    assert second["functional_parent_attached_by_branch"] == {
        "bg": True,
        "distant": False,
        "rigid_active": False,
    }
    assert second["feedback/functional_parent/branch/bg/attached"] == 1.0
    assert second["feedback/functional_parent/branch/rigid_active/attached"] == 0.0
    assert second["functional_parent_pack"].bg.projection.params["means"].requires_grad
    rigid_branch = second["functional_parent_pack"].rigid_active
    assert rigid_branch is not None
    for tensor in rigid_branch.projection.params.values():
        _assert_graph_free(tensor)

    grad = torch.autograd.grad(
        second["parent_params_bg"]["means"].square().sum(),
        bg_delta,
    )[0]
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


@_CUDA_ONLY
def test_stage3_4_source_feedback_requires_an_update_ancestor() -> None:
    device = torch.device("cuda")
    runtime, captures = _make_stage3_4_runtime(device, source_feedback=True)
    local_state = _make_local_state(device, with_rigid=False)
    common_meta = {
        "global_step": 0,
        "train_2d_mode": "trainable_checkpointed",
    }

    first = runtime._observe_stage2_0_biggs_measurement(
        local_state=local_state,
        batch={"scene_id": 9, "segment_id": 13},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=None,
        visit_meta={**common_meta, "model_update_count": 0, "has_update_ancestor": False},
    )
    assert first["iforward/feedback/render_enabled"] == 0.0
    assert captures["render"][0]["feedback_enabled"] is False
    for value in captures["render"][0]["gaussians_scene"].values():
        if torch.is_tensor(value):
            _assert_graph_free(value)
    first_grad = torch.autograd.grad(
        first["parent_feat_2d_bg"].sum(),
        local_state.bg.means,
        allow_unused=True,
    )[0]
    assert first_grad is None

    second = runtime._observe_stage2_0_biggs_measurement(
        local_state=local_state,
        batch={"scene_id": 9, "segment_id": 13},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=first["biggs_state"],
        visit_meta={**common_meta, "model_update_count": 1, "has_update_ancestor": True},
    )
    assert second["iforward/feedback/render_enabled"] == 1.0
    assert captures["render"][1]["feedback_enabled"] is True
    assert captures["render"][1]["checkpoint_dynamic"] is True
    second_grad = torch.autograd.grad(
        second["parent_feat_2d_bg"].sum(),
        local_state.bg.means,
        allow_unused=True,
    )[0]
    assert second_grad is not None
    assert torch.isfinite(second_grad).all()
    assert torch.count_nonzero(second_grad) > 0


@_CUDA_ONLY
@pytest.mark.parametrize(
    ("source_active", "parent_active"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_stage3_4_source_and_functional_parent_gates_are_independent(
    source_active: bool,
    parent_active: bool,
) -> None:
    device = torch.device("cuda")
    runtime, captures = _make_stage3_4_runtime(
        device,
        source_feedback=source_active,
    )
    runtime.observation_feedback_policy = _stage3_4_feedback_policy(
        source_feedback=source_active,
        source_alpha=1.0 if source_active else 0.0,
        functional_parent_alpha=1.0 if parent_active else 0.0,
    )
    local_state = _make_local_state(device, with_rigid=False)
    measurement = runtime._observe_stage2_0_biggs_measurement(
        local_state=local_state,
        batch={"scene_id": 29, "segment_id": 31},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=None,
        visit_meta={
            "global_step": 0,
            "train_2d_mode": "trainable_checkpointed",
            "model_update_count": 1,
            "has_update_ancestor": True,
        },
    )

    assert bool(measurement["iforward/feedback/render_enabled"]) is source_active
    assert bool(measurement["functional_parent_grad_active"]) is parent_active
    assert (
        measurement["functional_parent_pack"].bg.projection.params["means"].requires_grad
        is parent_active
    )
    assert bool(captures["render"][0]["feedback_enabled"]) is source_active
    source_scene_means = captures["render"][0]["gaussians_scene"]["means"]
    assert bool(source_scene_means.requires_grad) is source_active


@_CUDA_ONLY
def test_stage3_4_source_feedback_alpha_zero_is_forward_only_after_update() -> None:
    device = torch.device("cuda")
    runtime, captures = _make_stage3_4_runtime(device, source_feedback=True)
    cfg = {
        "enable": True,
        "scope": "within_rollout",
        "schedule": {"origin": "global_step", "activation_step": 0},
        "modes": {
            "repeat_refine": "trainable_checkpointed",
            "shuffled_coverage": "trainable_checkpointed",
            "high_block_repair": "trainable_checkpointed",
        },
        "source_render": {
            "enable": True,
            "renderer_mode": "differentiable_rgb",
            "checkpoint_scope": "full_dynamic_observation",
            "absgrad": False,
            "alpha_schedule": [[0, 0.0]],
        },
        "parent_projection": {"enable": False, "alpha_schedule": [[0, 0.0]]},
        "relation": {"enable": False, "alpha_schedule": [[0, 0.0]]},
        "scalar_anchor": {"geometry_grad": False},
        "discrete_routing_grad": False,
        "rollout_boundary_grad": False,
    }
    runtime.observation_feedback_policy = ObservationFeedbackPolicy.from_config(cfg)
    local_state = _make_local_state(device, with_rigid=False)
    measurement = runtime._observe_stage2_0_biggs_measurement(
        local_state=local_state,
        batch={"scene_id": 9, "segment_id": 13},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=None,
        visit_meta={
            "global_step": 0,
            "train_2d_mode": "trainable_checkpointed",
            "model_update_count": 1,
            "has_update_ancestor": True,
        },
    )
    assert measurement["iforward/feedback/render_enabled"] == 0.0
    assert captures["render"][0]["feedback_enabled"] is False
    for value in captures["render"][0]["gaussians_scene"].values():
        if torch.is_tensor(value):
            _assert_graph_free(value)


@_CUDA_ONLY
def test_stage3_4_k3_no_grad_and_validation_render_only_stay_forward_only() -> None:
    device = torch.device("cuda")
    runtime, _ = _make_stage3_4_runtime(device, source_feedback=True)
    local_state = _make_local_state(device, with_rigid=True)
    common_meta = {
        "global_step": 15000,
        "train_2d_mode": "trainable_checkpointed",
    }

    with torch.no_grad():
        first = runtime._observe_stage2_0_biggs_measurement(
            local_state=local_state,
            batch={"scene_id": 19, "segment_id": 23},
            source_indices=[0],
            source_frame_idx=0,
            biggs_state=None,
            visit_meta={
                **common_meta,
                "model_update_count": 0,
                "has_update_ancestor": False,
            },
        )
        second = runtime._observe_stage2_0_biggs_measurement(
            local_state=local_state,
            batch={"scene_id": 19, "segment_id": 23},
            source_indices=[0],
            source_frame_idx=0,
            biggs_state=first["biggs_state"],
            visit_meta={
                **common_meta,
                "model_update_count": 1,
                "has_update_ancestor": True,
            },
        )

    third = runtime._observe_stage2_0_biggs_measurement(
        local_state=local_state,
        batch={"scene_id": 19, "segment_id": 23},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=second["biggs_state"],
        visit_meta={
            **common_meta,
            "model_update_count": 2,
            "has_update_ancestor": True,
            "validation_render_only": True,
        },
    )

    for measurement in (first, second, third):
        assert measurement["feedback/functional_parent/geometry_alpha"] == 1.0
        assert measurement["feedback/functional_parent/grad_active"] == 0.0
        assert measurement["feedback/functional_parent/forward_only"] == 1.0
        assert measurement["iforward/feedback/render_enabled"] == 0.0
        for branch in measurement["functional_parent_pack"].iter_branches():
            for tensor in branch.projection.params.values():
                _assert_graph_free(tensor)
    assert third["feedback/functional_parent/validation_render_only"] == 1.0
