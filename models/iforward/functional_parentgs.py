from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Mapping, Optional, Union

import torch

from .biggs_parent_projector import (
    BigGSParentProjection,
    project_biggs_active_rigid_parents,
    project_biggs_parents,
)
from .biggs_parent_projector_diag import compute_child_projection_stats
from .biggs_state import BigGSBranchAssignment, BigGSRigidActiveAssignment
from .utils import cfg_get


_PARENT_PARAM_KEYS = (
    "means",
    "scales_log",
    "quats",
    "opacity_logit",
    "sh_dc",
    "sh_rest",
)


@dataclass(frozen=True)
class FunctionalParentProjectorConfig:
    """Strict Stage 3.4 configuration for per-visit ParentGS projection."""

    backend: str = "cuda_exact_diag"
    covariance_mode: str = "diagonal"
    mass_mode: str = "dynamic_tau_area"
    recompute_every_visit: bool = True
    grad_to_local_state: bool = True
    allow_cpu_fallback: bool = False
    allow_torch_fallback: bool = False
    allow_forward_only: bool = False
    allow_surrogate_runtime_vjp: bool = False
    min_scale: float = 1.0e-3
    max_scale_bg: float = 0.60
    max_scale_distant: float = 3.0
    max_scale_rigid: float = 0.45
    opacity_cap: float = 0.90
    opacity_min: float = 1.0e-6
    tau_parent_scale_bg: float = 0.5
    tau_parent_scale_distant: float = 0.7
    tau_parent_scale_rigid: float = 0.5
    eps: float = 1.0e-6
    min_child_mass: float = 1.0e-8
    finite_check: bool = True

    def __post_init__(self) -> None:
        backend = str(self.backend).lower()
        if backend not in {"cuda_exact_diag", "cuda_exact_diagonal"}:
            raise ValueError(
                "functional ParentGS requires parent_projector.backend=cuda_exact_diag, "
                f"got {self.backend!r}"
            )
        covariance_mode = str(self.covariance_mode).lower()
        if covariance_mode not in {"diag", "diagonal", "exact_diag", "exact_diagonal"}:
            raise ValueError(
                "functional ParentGS requires parent_projector.covariance_mode=diagonal, "
                f"got {self.covariance_mode!r}"
            )
        if str(self.mass_mode).lower() != "dynamic_tau_area":
            raise ValueError(
                "functional ParentGS requires parent_projector.mass_mode=dynamic_tau_area, "
                f"got {self.mass_mode!r}"
            )
        if not bool(self.recompute_every_visit):
            raise ValueError("functional ParentGS requires recompute_every_visit=true")
        if not bool(self.grad_to_local_state):
            raise ValueError("functional ParentGS requires grad_to_local_state=true")
        if (
            bool(self.allow_cpu_fallback)
            or bool(self.allow_torch_fallback)
            or bool(self.allow_forward_only)
            or bool(self.allow_surrogate_runtime_vjp)
        ):
            raise ValueError(
                "functional ParentGS forbids CPU/Torch projector fallback, "
                "forward-only fallback, and surrogate runtime VJP"
            )
        if float(self.min_scale) <= 0.0 or float(self.eps) <= 0.0 or float(self.min_child_mass) <= 0.0:
            raise ValueError("functional ParentGS scale/epsilon/mass floors must be positive")
        for branch_name, max_scale in (
            ("bg", self.max_scale_bg),
            ("distant", self.max_scale_distant),
            ("rigid", self.max_scale_rigid),
        ):
            if float(max_scale) < float(self.min_scale):
                raise ValueError(
                    f"functional ParentGS max_scale_{branch_name} must be >= min_scale"
                )
        if not 0.0 < float(self.opacity_min) < float(self.opacity_cap) < 1.0:
            raise ValueError("functional ParentGS opacity bounds must satisfy 0 < min < cap < 1")

    @classmethod
    def from_config(cls, cfg: Any) -> "FunctionalParentProjectorConfig":
        if isinstance(cfg, cls):
            return cfg

        def strict_bool(key: str, default: bool) -> bool:
            value = cfg_get(cfg, key, default)
            if type(value) is not bool:
                raise ValueError(
                    f"functional ParentGS parent_projector.{key} must be a boolean, "
                    f"got {value!r}"
                )
            return value

        return cls(
            backend=str(cfg_get(cfg, "backend", "cuda_exact_diag")),
            covariance_mode=str(cfg_get(cfg, "covariance_mode", "diagonal")),
            mass_mode=str(cfg_get(cfg, "mass_mode", "dynamic_tau_area")),
            recompute_every_visit=strict_bool("recompute_every_visit", True),
            grad_to_local_state=strict_bool("grad_to_local_state", True),
            allow_cpu_fallback=strict_bool("allow_cpu_fallback", False),
            allow_torch_fallback=strict_bool("allow_torch_fallback", False),
            allow_forward_only=strict_bool("allow_forward_only", False),
            allow_surrogate_runtime_vjp=strict_bool("allow_surrogate_runtime_vjp", False),
            min_scale=float(cfg_get(cfg, "min_scale", 1.0e-3)),
            max_scale_bg=float(cfg_get(cfg, "max_scale_bg", cfg_get(cfg, "max_scale", 0.60))),
            max_scale_distant=float(cfg_get(cfg, "max_scale_distant", cfg_get(cfg, "max_scale", 3.0))),
            max_scale_rigid=float(cfg_get(cfg, "max_scale_rigid", cfg_get(cfg, "max_scale", 0.45))),
            opacity_cap=float(cfg_get(cfg, "opacity_cap", 0.90)),
            opacity_min=float(cfg_get(cfg, "opacity_min", 1.0e-6)),
            tau_parent_scale_bg=float(
                cfg_get(cfg, "tau_parent_scale_bg", cfg_get(cfg, "tau_parent_scale", 0.5))
            ),
            tau_parent_scale_distant=float(
                cfg_get(cfg, "tau_parent_scale_distant", cfg_get(cfg, "tau_parent_scale", 0.7))
            ),
            tau_parent_scale_rigid=float(
                cfg_get(cfg, "tau_parent_scale_rigid", cfg_get(cfg, "tau_parent_scale", 0.5))
            ),
            eps=float(cfg_get(cfg, "eps", cfg_get(cfg, "covariance_eps", 1.0e-6))),
            min_child_mass=float(cfg_get(cfg, "min_child_mass", 1.0e-8)),
            finite_check=strict_bool("finite_check", True),
        )

    @staticmethod
    def _canonical_branch_name(branch_name: str) -> str:
        name = str(branch_name).lower()
        if name == "rigid_active":
            return "rigid"
        if name not in {"bg", "distant", "rigid"}:
            raise ValueError(f"unsupported functional ParentGS branch={branch_name!r}")
        return name

    def max_scale_for(self, branch_name: str) -> float:
        name = self._canonical_branch_name(branch_name)
        return float(getattr(self, f"max_scale_{name}"))

    def tau_parent_scale_for(self, branch_name: str) -> float:
        name = self._canonical_branch_name(branch_name)
        return float(getattr(self, f"tau_parent_scale_{name}"))

    def projector_dict(self, *, branch_name: str, attached: bool) -> Dict[str, Any]:
        backend = str(self.backend).lower()
        if not bool(attached):
            backend = "cuda_exact_diag_forward_only"
        return {
            "backend": backend,
            "covariance_mode": "diagonal",
            "mass_mode": str(self.mass_mode),
            "allow_cpu_fallback": False,
            "allow_torch_fallback": False,
            "min_scale": float(self.min_scale),
            "max_scale": self.max_scale_for(branch_name),
            "opacity_cap": float(self.opacity_cap),
            "opacity_min": float(self.opacity_min),
            "tau_parent_scale": self.tau_parent_scale_for(branch_name),
            "eps": float(self.eps),
            "min_child_mass": float(self.min_child_mass),
            "finite_check": bool(self.finite_check),
        }


def validate_stage3_4_functional_parentgs_config(iforward_cfg: Any) -> None:
    """Validate the complete raw Stage 3.4 routing/gradient contract.

    This validator intentionally avoids Python truthiness for configuration
    booleans: values such as ``"false"`` are invalid instead of silently being
    treated as true.  Both IForwardModel and its private Stage6 runtime call it
    before constructing Stage 3.4 modules.
    """

    failures: list[str] = []

    def require(condition: bool, path: str, expected: str, actual: Any) -> None:
        if not bool(condition):
            failures.append(f"{path} must be {expected}, got {actual!r}")

    def require_bool(node: Any, key: str, expected: bool, path: str) -> None:
        value = cfg_get(node, key, None)
        require(
            type(value) is bool and value is expected,
            path,
            str(expected).lower(),
            value,
        )

    version = str(cfg_get(iforward_cfg, "version", ""))
    variant = str(cfg_get(iforward_cfg, "training_variant", ""))
    require(
        version == "stage3_4_functional_parentgs_lift",
        "model.iforward.version",
        "'stage3_4_functional_parentgs_lift'",
        version,
    )
    require(
        variant == "stage3_4_functional_parentgs_lift",
        "model.iforward.training_variant",
        "'stage3_4_functional_parentgs_lift'",
        variant,
    )

    feedback_cfg = cfg_get(iforward_cfg, "observation_feedback", {}) or {}
    schedule_cfg = cfg_get(feedback_cfg, "schedule", {}) or {}
    modes_cfg = cfg_get(feedback_cfg, "modes", {}) or {}
    source_cfg = cfg_get(feedback_cfg, "source_render", {}) or {}
    functional_feedback_cfg = cfg_get(feedback_cfg, "functional_parent", {}) or {}
    parent_feedback_cfg = cfg_get(feedback_cfg, "parent_projection", {}) or {}
    relation_feedback_cfg = cfg_get(feedback_cfg, "relation", {}) or {}
    scalar_anchor_cfg = cfg_get(feedback_cfg, "scalar_anchor", {}) or {}
    require_bool(feedback_cfg, "enable", True, "model.iforward.observation_feedback.enable")
    require(
        str(cfg_get(schedule_cfg, "origin", "")) == "activation_step",
        "model.iforward.observation_feedback.schedule.origin",
        "'activation_step'",
        cfg_get(schedule_cfg, "origin", None),
    )
    activation_step = cfg_get(schedule_cfg, "activation_step", None)
    require(
        type(activation_step) is int and activation_step == 0,
        "model.iforward.observation_feedback.schedule.activation_step",
        "integer 0",
        activation_step,
    )
    expected_modes = {
        "repeat_refine": "trainable_checkpointed",
        "shuffled_coverage": "trainable_checkpointed",
        "high_block_repair": "frozen_input_grad_checkpointed",
    }
    for name, expected_mode in expected_modes.items():
        value = str(cfg_get(modes_cfg, name, ""))
        require(
            value == expected_mode,
            f"model.iforward.observation_feedback.modes.{name}",
            repr(expected_mode),
            value,
        )
    require_bool(source_cfg, "enable", True, "model.iforward.observation_feedback.source_render.enable")
    require(
        str(cfg_get(source_cfg, "checkpoint_scope", "")) == "full_dynamic_observation",
        "model.iforward.observation_feedback.source_render.checkpoint_scope",
        "'full_dynamic_observation'",
        cfg_get(source_cfg, "checkpoint_scope", None),
    )
    alpha_schedule = cfg_get(source_cfg, "alpha_schedule", None)
    require(
        alpha_schedule is not None
        and not isinstance(alpha_schedule, (str, bytes))
        and hasattr(alpha_schedule, "__len__")
        and len(alpha_schedule) > 0,
        "model.iforward.observation_feedback.source_render.alpha_schedule",
        "a non-empty schedule",
        alpha_schedule,
    )
    require_bool(
        functional_feedback_cfg,
        "enable",
        True,
        "model.iforward.observation_feedback.functional_parent.enable",
    )
    functional_branches = cfg_get(functional_feedback_cfg, "branches", None)
    try:
        normalized_functional_branches = tuple(functional_branches)
    except TypeError:
        normalized_functional_branches = ()
    require(
        normalized_functional_branches == ("bg", "distant", "rigid_active"),
        "model.iforward.observation_feedback.functional_parent.branches",
        "['bg', 'distant', 'rigid_active']",
        functional_branches,
    )
    start_after_model_updates = cfg_get(
        functional_feedback_cfg,
        "start_after_model_updates",
        None,
    )
    require(
        type(start_after_model_updates) is int and start_after_model_updates == 1,
        "model.iforward.observation_feedback.functional_parent.start_after_model_updates",
        "integer 1",
        start_after_model_updates,
    )
    functional_alpha_schedule = cfg_get(functional_feedback_cfg, "alpha_schedule", None)
    expected_functional_alpha_schedule = (
        (0, 0.0),
        (1000, 0.10),
        (3000, 0.25),
        (8000, 0.50),
        (15000, 1.0),
    )
    try:
        normalized_functional_alpha_schedule = tuple(
            (int(point[0]), float(point[1])) for point in functional_alpha_schedule
        )
    except (TypeError, ValueError, IndexError):
        normalized_functional_alpha_schedule = ()
    require(
        normalized_functional_alpha_schedule == expected_functional_alpha_schedule,
        "model.iforward.observation_feedback.functional_parent.alpha_schedule",
        repr([list(point) for point in expected_functional_alpha_schedule]),
        functional_alpha_schedule,
    )
    require_bool(
        parent_feedback_cfg,
        "enable",
        False,
        "model.iforward.observation_feedback.parent_projection.enable",
    )
    require(
        str(cfg_get(parent_feedback_cfg, "forward_mode", "")) == "functional_per_visit",
        "model.iforward.observation_feedback.parent_projection.forward_mode",
        "'functional_per_visit'",
        cfg_get(parent_feedback_cfg, "forward_mode", None),
    )
    require(
        str(cfg_get(parent_feedback_cfg, "backward_mode", "")) == "disabled",
        "model.iforward.observation_feedback.parent_projection.backward_mode",
        "'disabled'",
        cfg_get(parent_feedback_cfg, "backward_mode", None),
    )
    require_bool(
        relation_feedback_cfg,
        "enable",
        False,
        "model.iforward.observation_feedback.relation.enable",
    )
    for key in (
        "differentiable_diag_cov",
        "checkpoint",
        "grad_to_child_geometry",
        "grad_to_parent_geometry",
        "grad_to_child_code",
        "grad_to_parent_event",
        "grad_to_support",
    ):
        require_bool(
            relation_feedback_cfg,
            key,
            False,
            f"model.iforward.observation_feedback.relation.{key}",
        )
    require_bool(
        scalar_anchor_cfg,
        "geometry_grad",
        False,
        "model.iforward.observation_feedback.scalar_anchor.geometry_grad",
    )
    require_bool(
        feedback_cfg,
        "discrete_routing_grad",
        False,
        "model.iforward.observation_feedback.discrete_routing_grad",
    )
    require_bool(
        feedback_cfg,
        "rollout_boundary_grad",
        False,
        "model.iforward.observation_feedback.rollout_boundary_grad",
    )

    lifting_cfg = cfg_get(iforward_cfg, "lifting", {}) or {}
    parent_lift_cfg = cfg_get(lifting_cfg, "parent", {}) or {}
    require_bool(lifting_cfg, "detach_geometry", True, "model.iforward.lifting.detach_geometry")
    require(
        str(cfg_get(parent_lift_cfg, "type", "")) == "functional_parent_direct_lift",
        "model.iforward.lifting.parent.type",
        "'functional_parent_direct_lift'",
        cfg_get(parent_lift_cfg, "type", None),
    )
    require_bool(
        parent_lift_cfg,
        "geometry_grad",
        False,
        "model.iforward.lifting.parent.geometry_grad",
    )
    require(
        str(cfg_get(parent_lift_cfg, "color_mode", "")) == "constant_zero",
        "model.iforward.lifting.parent.color_mode",
        "'constant_zero'",
        cfg_get(parent_lift_cfg, "color_mode", None),
    )

    biggs_cfg = cfg_get(iforward_cfg, "biggs", {}) or {}
    projector_cfg = cfg_get(biggs_cfg, "parent_projector", {}) or {}
    try:
        FunctionalParentProjectorConfig.from_config(projector_cfg)
    except (TypeError, ValueError) as exc:
        failures.append(str(exc))
    require(
        str(cfg_get(projector_cfg, "grad_mode", "")) == "functional_autograd",
        "model.iforward.biggs.parent_projector.grad_mode",
        "'functional_autograd'",
        cfg_get(projector_cfg, "grad_mode", None),
    )
    for key in (
        "recompute_every_visit",
        "grad_to_local_state",
    ):
        require_bool(
            projector_cfg,
            key,
            True,
            f"model.iforward.biggs.parent_projector.{key}",
        )
    for key in (
        "allow_cpu_fallback",
        "allow_torch_fallback",
        "allow_forward_only",
        "allow_surrogate_runtime_vjp",
    ):
        require_bool(
            projector_cfg,
            key,
            False,
            f"model.iforward.biggs.parent_projector.{key}",
        )

    parent_state_cfg = cfg_get(biggs_cfg, "parent_state", {}) or {}
    require(
        str(cfg_get(parent_state_cfg, "mode", "")) == "functional_per_visit",
        "model.iforward.biggs.parent_state.mode",
        "'functional_per_visit'",
        cfg_get(parent_state_cfg, "mode", None),
    )
    require_bool(
        parent_state_cfg,
        "persistent_geometry",
        False,
        "model.iforward.biggs.parent_state.persistent_geometry",
    )
    require_bool(
        parent_state_cfg,
        "incremental_update",
        False,
        "model.iforward.biggs.parent_state.incremental_update",
    )
    require(
        str(cfg_get(parent_state_cfg, "exact_refresh_policy", "")) == "none",
        "model.iforward.biggs.parent_state.exact_refresh_policy",
        "'none'",
        cfg_get(parent_state_cfg, "exact_refresh_policy", None),
    )

    gradient_cfg = cfg_get(biggs_cfg, "gradient_contract", {}) or {}
    expected_gradient_contract = {
        "param_codec_geometry": True,
        "lifting_geometry": False,
        "ptv3_coords": False,
        "assignment": False,
        "relation_child_geometry": False,
        "relation_parent_geometry": False,
    }
    for key, expected in expected_gradient_contract.items():
        require_bool(
            gradient_cfg,
            key,
            expected,
            f"model.iforward.biggs.gradient_contract.{key}",
        )

    decoder_cfg = cfg_get(biggs_cfg, "child_decoder", {}) or {}
    require(
        str(cfg_get(decoder_cfg, "relation_source", "")) == "functional_detached_stats",
        "model.iforward.biggs.child_decoder.relation_source",
        "'functional_detached_stats'",
        cfg_get(decoder_cfg, "relation_source", None),
    )
    for key in (
        "detach_relation_inputs",
        "detach_child_code_inputs",
        "detach_child_params",
        "detach_parent_params",
    ):
        require_bool(
            decoder_cfg,
            key,
            True,
            f"model.iforward.biggs.child_decoder.{key}",
        )

    parent_spatial_cfg = cfg_get(iforward_cfg, "parent_spatial", {}) or {}
    codec_cfg = cfg_get(parent_spatial_cfg, "param_codec", {}) or {}
    ptv3_cfg = cfg_get(parent_spatial_cfg, "ptv3", {}) or {}
    require(
        str(cfg_get(codec_cfg, "mode", "")) == "legacy17d_plus_geometry8d_residual",
        "model.iforward.parent_spatial.param_codec.mode",
        "'legacy17d_plus_geometry8d_residual'",
        cfg_get(codec_cfg, "mode", None),
    )
    require(
        str(cfg_get(codec_cfg, "schema", ""))
        == "legacy17d_plus_geometry8d_residual_v1",
        "model.iforward.parent_spatial.param_codec.schema",
        "'legacy17d_plus_geometry8d_residual_v1'",
        cfg_get(codec_cfg, "schema", None),
    )
    require_bool(
        codec_cfg,
        "grad_to_parent_params",
        True,
        "model.iforward.parent_spatial.param_codec.grad_to_parent_params",
    )
    require_bool(
        codec_cfg,
        "detach_legacy_params",
        True,
        "model.iforward.parent_spatial.param_codec.detach_legacy_params",
    )
    require_bool(
        codec_cfg,
        "detach_support",
        True,
        "model.iforward.parent_spatial.param_codec.detach_support",
    )
    require_bool(
        ptv3_cfg,
        "detach_coords",
        True,
        "model.iforward.parent_spatial.ptv3.detach_coords",
    )

    gdkv_cfg = cfg_get(iforward_cfg, "parent_optimizer_memory", {}) or {}
    require_bool(
        gdkv_cfg,
        "enable",
        True,
        "model.iforward.parent_optimizer_memory.enable",
    )
    require(
        str(cfg_get(gdkv_cfg, "type", "")) == "lowrank_gated_delta_kv",
        "model.iforward.parent_optimizer_memory.type",
        "'lowrank_gated_delta_kv'",
        cfg_get(gdkv_cfg, "type", None),
    )
    require(
        str(cfg_get(gdkv_cfg, "reset_scope", "")) == "episode",
        "model.iforward.parent_optimizer_memory.reset_scope",
        "'episode'",
        cfg_get(gdkv_cfg, "reset_scope", None),
    )
    require(
        str(cfg_get(gdkv_cfg, "detach_scope", "")) == "rollout_boundary",
        "model.iforward.parent_optimizer_memory.detach_scope",
        "'rollout_boundary'",
        cfg_get(gdkv_cfg, "detach_scope", None),
    )
    parent_mamba_cfg = cfg_get(iforward_cfg, "parent_optimizer_mamba", {}) or {}
    require_bool(
        parent_mamba_cfg,
        "enable",
        False,
        "model.iforward.parent_optimizer_mamba.enable",
    )

    if failures:
        raise ValueError("Stage 3.4 functional ParentGS contract violation: " + "; ".join(failures))


@dataclass(frozen=True)
class FunctionalChildStats:
    mass: torch.Tensor
    tau_area: torch.Tensor
    diag_cov: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("mass", self.mass),
            ("tau_area", self.tau_area),
            ("diag_cov", self.diag_cov),
        ):
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError(f"functional ParentGS relation stat {name} must be detached")


FunctionalAssignment = Union[BigGSBranchAssignment, BigGSRigidActiveAssignment]


@dataclass(frozen=True)
class FunctionalParentBranch:
    assignment: FunctionalAssignment
    projection: BigGSParentProjection
    child_stats_detached: FunctionalChildStats
    parent_mass_mean: torch.Tensor
    branch_name: str

    def __post_init__(self) -> None:
        if self.parent_mass_mean.requires_grad or self.parent_mass_mean.grad_fn is not None:
            raise ValueError("functional ParentGS parent_mass_mean support must be detached")

    @property
    def parent_mass_sum(self) -> torch.Tensor:
        """Detached Parent mass for geometry-only relational decoding."""

        return self.projection.child_mass_sum.detach()

    @property
    def num_children(self) -> int:
        return int(self.child_stats_detached.mass.shape[0])

    @property
    def num_parents(self) -> int:
        return int(self.projection.num_parents)


@dataclass(frozen=True)
class FunctionalParentPack:
    bg: FunctionalParentBranch
    distant: Optional[FunctionalParentBranch] = None
    rigid_active: Optional[FunctionalParentBranch] = None

    def iter_branches(self) -> Iterator[FunctionalParentBranch]:
        """Yield present branches in the Stage 3.4 ABI order."""

        yield self.bg
        if self.distant is not None:
            yield self.distant
        if self.rigid_active is not None:
            yield self.rigid_active


def _validate_child_params(child_params: Mapping[str, torch.Tensor]) -> None:
    missing = [key for key in _PARENT_PARAM_KEYS if key not in child_params]
    if missing:
        raise KeyError(f"functional ParentGS child params missing keys: {missing}")
    n = int(child_params["means"].shape[0])
    for key in _PARENT_PARAM_KEYS:
        value = child_params[key]
        if not torch.is_tensor(value):
            raise TypeError(f"functional ParentGS child param {key} must be a tensor")
        if int(value.shape[0]) != n:
            raise ValueError(
                f"functional ParentGS child param {key} length mismatch: "
                f"{int(value.shape[0])} != {n}"
            )


def _optional_detached(value: Optional[torch.Tensor], *, device: torch.device) -> Optional[torch.Tensor]:
    return None if value is None else value.detach().to(device=device)


def _detach_assignment_to(
    assignment: FunctionalAssignment,
    *,
    device: torch.device,
) -> FunctionalAssignment:
    if isinstance(assignment, BigGSBranchAssignment):
        return BigGSBranchAssignment(
            branch=str(assignment.branch),
            child_to_parent=assignment.child_to_parent.detach().to(device=device),
            child_order=assignment.child_order.detach().to(device=device),
            parent_start=assignment.parent_start.detach().to(device=device),
            parent_count=assignment.parent_count.detach().to(device=device),
            child_mass=assignment.child_mass.detach().to(device=device),
            num_children=int(assignment.num_children),
            num_parents=int(assignment.num_parents),
            object_id=_optional_detached(assignment.object_id, device=device),
            parent_object_id=_optional_detached(assignment.parent_object_id, device=device),
            child_basis=_optional_detached(assignment.child_basis, device=device),
            basis_valid=_optional_detached(assignment.basis_valid, device=device),
            basis_weight_sum=_optional_detached(assignment.basis_weight_sum, device=device),
            basis_version=int(assignment.basis_version),
        )
    if isinstance(assignment, BigGSRigidActiveAssignment):
        return BigGSRigidActiveAssignment(
            fine_S=assignment.fine_S.detach().to(device=device),
            child_to_active_parent_S=assignment.child_to_active_parent_S.detach().to(device=device),
            active_parent_global=assignment.active_parent_global.detach().to(device=device),
            active_parent_count=assignment.active_parent_count.detach().to(device=device),
            active_parent_start=assignment.active_parent_start.detach().to(device=device),
            active_child_order_S=assignment.active_child_order_S.detach().to(device=device),
            child_mass_S=assignment.child_mass_S.detach().to(device=device),
            parent_inside_mask=assignment.parent_inside_mask.detach().to(device=device),
            child_inside_mask_S=assignment.child_inside_mask_S.detach().to(device=device),
            child_basis_S=_optional_detached(assignment.child_basis_S, device=device),
            basis_valid=_optional_detached(assignment.basis_valid, device=device),
            basis_weight_sum=_optional_detached(assignment.basis_weight_sum, device=device),
        )
    raise TypeError(f"unsupported functional ParentGS assignment type: {type(assignment)!r}")


def _assignment_projection_args(
    assignment: FunctionalAssignment,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(assignment, BigGSBranchAssignment):
        return (
            assignment.child_to_parent,
            assignment.child_order,
            assignment.parent_start,
            assignment.parent_count,
            assignment.child_mass,
        )
    return (
        assignment.child_to_active_parent_S,
        assignment.active_child_order_S,
        assignment.active_parent_start,
        assignment.active_parent_count,
        assignment.child_mass_S,
    )


def build_functional_parent_branch(
    *,
    child_params: Dict[str, torch.Tensor],
    assignment: FunctionalAssignment,
    projector_cfg: FunctionalParentProjectorConfig | Any,
    branch_name: str,
    attached: bool = True,
) -> FunctionalParentBranch:
    """Recompute one ephemeral ParentGS branch from live LocalGS tensors."""

    _validate_child_params(child_params)
    config = FunctionalParentProjectorConfig.from_config(projector_cfg)
    canonical_name = config._canonical_branch_name(branch_name)
    stored_name = "rigid_active" if canonical_name == "rigid" else canonical_name
    ref = child_params["means"]
    assignment_detached = _detach_assignment_to(assignment, device=ref.device)
    params_for_projection = {
        key: (value if bool(attached) else value.detach())
        for key, value in child_params.items()
    }
    child_to_parent, child_order, parent_start, parent_count, child_mass = _assignment_projection_args(
        assignment_detached
    )
    if int(child_to_parent.numel()) != int(ref.shape[0]):
        raise ValueError(
            f"functional ParentGS {stored_name} assignment has {int(child_to_parent.numel())} "
            f"children for {int(ref.shape[0])} child params"
        )

    backend_cfg = config.projector_dict(branch_name=canonical_name, attached=bool(attached))
    max_scale = config.max_scale_for(canonical_name)
    project_t0 = time.perf_counter()
    if isinstance(assignment_detached, BigGSBranchAssignment):
        projection = project_biggs_parents(
            branch=SimpleNamespace(**params_for_projection),
            assignment=assignment_detached,
            cfg=backend_cfg,
            max_scale=max_scale,
        )
    else:
        projection = project_biggs_active_rigid_parents(
            means_world_S=params_for_projection["means"],
            quats_world_S=params_for_projection["quats"],
            scales_log_S=params_for_projection["scales_log"],
            opacity_logit_S=params_for_projection["opacity_logit"],
            sh_dc_S=params_for_projection["sh_dc"],
            sh_rest_S=params_for_projection["sh_rest"],
            child_to_active_parent_S=child_to_parent,
            child_mass_S=child_mass,
            active_parent_count=parent_count,
            cfg=backend_cfg,
            max_scale=max_scale,
            active_child_order_S=child_order,
            active_parent_start=parent_start,
        )
    projection.aux_stats["functional_project_ms"] = float(
        (time.perf_counter() - project_t0) * 1000.0
    )

    # Relation geometry is metadata, not a secondary backward path.  Compute it
    # from an explicitly detached view so the boundary never constructs a
    # temporary graph from live LocalGS tensors.
    with torch.no_grad():
        mass, tau_area, diag_cov = compute_child_projection_stats(
            scales_log=params_for_projection["scales_log"].detach(),
            quats=params_for_projection["quats"].detach(),
            opacity_logit=params_for_projection["opacity_logit"].detach(),
            child_mass=child_mass.detach(),
            min_mass=float(config.min_child_mass),
            mass_mode=str(config.mass_mode),
        )
    child_stats = FunctionalChildStats(
        mass=mass,
        tau_area=tau_area,
        diag_cov=diag_cov,
    )
    if not bool(attached):
        projection_tensors = (*projection.params.values(), projection.child_mass_sum, projection.child_mass_mean)
        if any(value.requires_grad or value.grad_fn is not None for value in projection_tensors):
            raise RuntimeError("functional ParentGS forward-only visit produced attached Parent tensors")
    return FunctionalParentBranch(
        assignment=assignment_detached,
        projection=projection,
        child_stats_detached=child_stats,
        parent_mass_mean=projection.child_mass_mean.detach(),
        branch_name=stored_name,
    )


def _optional_functional_branch(
    *,
    child_params: Optional[Dict[str, torch.Tensor]],
    assignment: Optional[FunctionalAssignment],
    projector_cfg: FunctionalParentProjectorConfig,
    branch_name: str,
    attached: bool,
) -> Optional[FunctionalParentBranch]:
    if child_params is None and assignment is None:
        return None
    if child_params is None or assignment is None:
        raise ValueError(
            f"functional ParentGS optional branch {branch_name} requires both child params and assignment"
        )
    return build_functional_parent_branch(
        child_params=child_params,
        assignment=assignment,
        projector_cfg=projector_cfg,
        branch_name=branch_name,
        attached=attached,
    )


def build_functional_parent_pack(
    *,
    bg_params: Dict[str, torch.Tensor],
    bg_assignment: BigGSBranchAssignment,
    projector_cfg: FunctionalParentProjectorConfig | Any,
    attached: Optional[bool] = None,
    attached_by_branch: Optional[Mapping[str, bool]] = None,
    distant_params: Optional[Dict[str, torch.Tensor]] = None,
    distant_assignment: Optional[BigGSBranchAssignment] = None,
    rigid_active_params: Optional[Dict[str, torch.Tensor]] = None,
    rigid_active_assignment: Optional[BigGSRigidActiveAssignment] = None,
) -> FunctionalParentPack:
    """Build the current measurement's bg -> distant -> rigid ParentGS pack."""

    if attached is not None and attached_by_branch is not None:
        raise ValueError("attached and attached_by_branch are mutually exclusive")
    if attached is None and attached_by_branch is None:
        raise ValueError("one of attached or attached_by_branch must be provided")
    if attached is not None:
        if type(attached) is not bool:
            raise TypeError("attached must be a boolean")
        branch_attachment = {
            "bg": bool(attached),
            "distant": bool(attached),
            "rigid_active": bool(attached),
        }
    else:
        if not isinstance(attached_by_branch, Mapping):
            raise TypeError("attached_by_branch must be a mapping")
        unknown = sorted(
            str(key)
            for key in set(attached_by_branch) - {"bg", "distant", "rigid_active"}
        )
        if unknown:
            raise ValueError(
                "attached_by_branch contains unsupported branches: " + ", ".join(unknown)
            )
        required = {"bg"}
        if distant_params is not None or distant_assignment is not None:
            required.add("distant")
        if rigid_active_params is not None or rigid_active_assignment is not None:
            required.add("rigid_active")
        missing = sorted(required - set(attached_by_branch))
        if missing:
            raise ValueError(
                "attached_by_branch is missing present branches: " + ", ".join(missing)
            )
        invalid = sorted(
            str(key)
            for key, value in attached_by_branch.items()
            if type(value) is not bool
        )
        if invalid:
            raise TypeError(
                "attached_by_branch values must be booleans; invalid: " + ", ".join(invalid)
            )
        branch_attachment = {
            name: bool(attached_by_branch.get(name, False))
            for name in ("bg", "distant", "rigid_active")
        }

    config = FunctionalParentProjectorConfig.from_config(projector_cfg)
    bg = build_functional_parent_branch(
        child_params=bg_params,
        assignment=bg_assignment,
        projector_cfg=config,
        branch_name="bg",
        attached=branch_attachment["bg"],
    )
    distant = _optional_functional_branch(
        child_params=distant_params,
        assignment=distant_assignment,
        projector_cfg=config,
        branch_name="distant",
        attached=branch_attachment["distant"],
    )
    rigid_active = _optional_functional_branch(
        child_params=rigid_active_params,
        assignment=rigid_active_assignment,
        projector_cfg=config,
        branch_name="rigid_active",
        attached=branch_attachment["rigid_active"],
    )
    return FunctionalParentPack(bg=bg, distant=distant, rigid_active=rigid_active)


def detach_parent_params_for_lifting(
    params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Create the explicit stop-gradient view used by ParentGS lifting."""

    detached: Dict[str, torch.Tensor] = {}
    for key, value in params.items():
        if not torch.is_tensor(value):
            raise TypeError(f"functional ParentGS parent param {key} must be a tensor")
        detached[key] = value.detach()
    if any(value.requires_grad or value.grad_fn is not None for value in detached.values()):
        raise RuntimeError("functional ParentGS lifting params must be graph-free")
    return detached


def build_parent_lift_scene(
    pack: FunctionalParentPack,
    color_mode: str = "constant_zero",
) -> Dict[str, torch.Tensor]:
    """Build the graph-isolated ParentGS scene used for feature lifting."""

    if str(color_mode).lower() != "constant_zero":
        raise ValueError(
            "functional ParentGS lifting only supports color_mode=constant_zero, "
            f"got {color_mode!r}"
        )
    parts: list[Dict[str, torch.Tensor]] = []
    color_channels: Optional[int] = None
    for branch in pack.iter_branches():
        params = detach_parent_params_for_lifting(branch.projection.params)
        branch_color_channels = int(params["sh_rest"].shape[1]) + 1
        if color_channels is None:
            color_channels = branch_color_channels
        elif branch_color_channels != color_channels:
            raise ValueError("functional ParentGS branches must use the same SH basis count")
        parts.append(
            {
                "means": params["means"],
                "scales": torch.exp(params["scales_log"]),
                "quats": params["quats"],
                "opacities": torch.sigmoid(params["opacity_logit"]).reshape(-1),
                "colors": params["means"].new_zeros(
                    (int(params["means"].shape[0]), branch_color_channels, 3)
                ),
            }
        )
    if not parts:  # pragma: no cover - FunctionalParentPack always has bg.
        raise ValueError("functional ParentGS lifting requires a bg branch")
    scene = {
        key: torch.cat([part[key] for part in parts], dim=0)
        for key in ("means", "scales", "quats", "opacities", "colors")
    }
    if any(value.requires_grad or value.grad_fn is not None for value in scene.values()):
        raise RuntimeError("functional ParentGS lifting scene geometry must be detached")
    return scene


__all__ = [
    "FunctionalChildStats",
    "FunctionalParentBranch",
    "FunctionalParentPack",
    "FunctionalParentProjectorConfig",
    "build_functional_parent_branch",
    "build_functional_parent_pack",
    "build_parent_lift_scene",
    "detach_parent_params_for_lifting",
    "validate_stage3_4_functional_parentgs_config",
]
