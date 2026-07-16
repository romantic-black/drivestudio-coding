from __future__ import annotations

import math
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import torch
from torch import nn


class FeedbackMode(str, Enum):
    """Per-rollout 2D frontend/observation autograd policy."""

    TRAINABLE = "trainable"
    FROZEN_NO_GRAD = "frozen_no_grad"
    AUTO = "auto"
    TRAINABLE_CHECKPOINTED = "trainable_checkpointed"
    FROZEN_INPUT_GRAD_CHECKPOINTED = "frozen_input_grad_checkpointed"

    @classmethod
    def parse(cls, value: Any, *, path: str = "feedback_mode") -> "FeedbackMode":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string, got {type(value).__name__}")
        try:
            return cls(value)
        except ValueError as exc:
            supported = ", ".join(mode.value for mode in cls)
            raise ValueError(f"unsupported {path}={value!r}; expected one of: {supported}") from exc

    @property
    def checkpointed(self) -> bool:
        return self in {self.TRAINABLE_CHECKPOINTED, self.FROZEN_INPUT_GRAD_CHECKPOINTED}

    @property
    def parameter_grad_enabled(self) -> bool:
        return self in {self.TRAINABLE, self.AUTO, self.TRAINABLE_CHECKPOINTED}

    @property
    def input_grad_enabled(self) -> bool:
        return self is not self.FROZEN_NO_GRAD

    @property
    def freezes_frontend_parameters(self) -> bool:
        return self in {self.FROZEN_NO_GRAD, self.FROZEN_INPUT_GRAD_CHECKPOINTED}


@dataclass(frozen=True)
class FeedbackAlphaSchedule:
    """Piecewise-linear feedback-Jacobian scale with endpoint clamping."""

    points: tuple[tuple[int, float], ...] = ((0, 0.0),)

    def __post_init__(self) -> None:
        normalized: list[tuple[int, float]] = []
        previous_step = -1
        for idx, point in enumerate(tuple(self.points)):
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise TypeError(f"alpha_schedule[{idx}] must be a [step, alpha] pair")
            step, alpha = point
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError(f"alpha_schedule[{idx}][0] must be an integer step")
            if step < 0:
                raise ValueError(f"alpha_schedule[{idx}][0] must be >= 0")
            if step <= previous_step:
                raise ValueError("alpha_schedule steps must be strictly increasing")
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
                raise TypeError(f"alpha_schedule[{idx}][1] must be a number")
            alpha_f = float(alpha)
            if not math.isfinite(alpha_f) or not 0.0 <= alpha_f <= 1.0:
                raise ValueError(f"alpha_schedule[{idx}][1] must be finite and in [0, 1]")
            normalized.append((int(step), alpha_f))
            previous_step = int(step)
        if not normalized:
            raise ValueError("alpha_schedule must contain at least one point")
        object.__setattr__(self, "points", tuple(normalized))

    @classmethod
    def from_config(cls, raw: Any, *, path: str = "alpha_schedule") -> "FeedbackAlphaSchedule":
        if isinstance(raw, cls):
            return raw
        if raw is None or isinstance(raw, (str, bytes, Mapping)):
            raise TypeError(f"{path} must be a sequence of [step, alpha] pairs")
        try:
            values = list(raw)
        except TypeError as exc:
            raise TypeError(f"{path} must be a sequence of [step, alpha] pairs") from exc
        points: list[tuple[int, float]] = []
        for idx, value in enumerate(values):
            try:
                pair = list(value)
            except TypeError as exc:
                raise TypeError(f"{path}[{idx}] must be a [step, alpha] pair") from exc
            if len(pair) != 2:
                raise ValueError(f"{path}[{idx}] must contain exactly two values")
            step, alpha = pair
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError(f"{path}[{idx}][0] must be an integer step")
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
                raise TypeError(f"{path}[{idx}][1] must be a number")
            points.append((int(step), float(alpha)))
        try:
            return cls(tuple(points))
        except (TypeError, ValueError) as exc:
            raise type(exc)(str(exc).replace("alpha_schedule", path)) from exc

    def value_at(self, step: int) -> float:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("feedback schedule step must be an integer")
        if step <= self.points[0][0]:
            return float(self.points[0][1])
        if step >= self.points[-1][0]:
            return float(self.points[-1][1])
        for (left_step, left_alpha), (right_step, right_alpha) in zip(self.points, self.points[1:]):
            if left_step <= step <= right_step:
                ratio = float(step - left_step) / float(right_step - left_step)
                return float(left_alpha + ratio * (right_alpha - left_alpha))
        raise RuntimeError("unreachable feedback alpha schedule interval")

    def __call__(self, step: int) -> float:
        return self.value_at(step)


def scale_feedback(tensor: torch.Tensor, alpha: float | torch.Tensor) -> torch.Tensor:
    """Keep the forward value identical while scaling only its local Jacobian."""

    if not torch.is_tensor(tensor):
        raise TypeError(f"scale_feedback tensor must be torch.Tensor, got {type(tensor).__name__}")
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            raise ValueError("scale_feedback alpha tensor must contain exactly one value")
        if alpha.device != tensor.device:
            raise ValueError("scale_feedback alpha tensor must be on the same device as tensor")
        if alpha.requires_grad:
            raise ValueError("scale_feedback alpha is a policy value and must not require gradients")
        alpha_float = float(alpha.detach().to(device="cpu", dtype=torch.float64).item())
        if not math.isfinite(alpha_float) or not 0.0 <= alpha_float <= 1.0:
            raise ValueError("scale_feedback alpha must be finite and in [0, 1]")
        alpha_value = alpha.to(dtype=tensor.dtype)
    else:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("scale_feedback alpha must be a number or scalar tensor")
        alpha_float = float(alpha)
        if not math.isfinite(alpha_float) or not 0.0 <= alpha_float <= 1.0:
            raise ValueError("scale_feedback alpha must be finite and in [0, 1]")
        alpha_value = alpha_float
    detached = tensor.detach()
    return detached + alpha_value * (tensor - detached)


def _mapping(raw: Any, *, path: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping) or hasattr(raw, "items"):
        try:
            return {str(key): value for key, value in raw.items()}
        except Exception as exc:
            raise TypeError(f"{path} must be a mapping") from exc
    raise TypeError(f"{path} must be a mapping, got {type(raw).__name__}")


def _section(raw: Any, *, path: str, allowed: set[str]) -> dict[str, Any]:
    values = _mapping(raw, path=path)
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"{path} contains unsupported keys: {', '.join(unknown)}")
    return values


def _bool(values: Mapping[str, Any], key: str, default: bool, *, path: str) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{path}.{key} must be a boolean")
    return value


def _str(values: Mapping[str, Any], key: str, default: str, *, path: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"{path}.{key} must be a string")
    return value


def _int(values: Mapping[str, Any], key: str, default: int, *, path: str, minimum: int = 0) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}.{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{path}.{key} must be >= {minimum}")
    return int(value)


def _float(values: Mapping[str, Any], key: str, default: float, *, path: str) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}.{key} must be a number")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"{path}.{key} must be finite")
    return value_f


def _branches(values: Mapping[str, Any], *, path: str) -> tuple[str, ...]:
    raw = values.get("branches", ("bg", "distant"))
    if isinstance(raw, (str, bytes, Mapping)):
        raise TypeError(f"{path}.branches must be a sequence")
    try:
        branches = tuple(str(value) for value in raw)
    except TypeError as exc:
        raise TypeError(f"{path}.branches must be a sequence") from exc
    if not branches or len(set(branches)) != len(branches):
        raise ValueError(f"{path}.branches must be non-empty and contain no duplicates")
    unsupported = sorted(set(branches) - {"bg", "distant"})
    if unsupported:
        raise ValueError(f"{path}.branches only supports bg/distant, got: {', '.join(unsupported)}")
    return branches


@dataclass(frozen=True)
class SourceRenderFeedbackPolicy:
    enable: bool = False
    renderer_mode: str = "differentiable_rgb"
    checkpoint_scope: str = "full_dynamic_observation"
    absgrad: bool = False
    alpha_schedule: FeedbackAlphaSchedule = field(default_factory=FeedbackAlphaSchedule)


@dataclass(frozen=True)
class ParentProjectionDriftPolicy:
    check_interval: int = 500
    warn_threshold: float = 1.0e-3
    skip_vjp_threshold: float = 5.0e-3
    exact_refresh_threshold: float = 1.0e-2


@dataclass(frozen=True)
class ParentProjectionFeedbackPolicy:
    enable: bool = False
    branches: tuple[str, ...] = ("bg", "distant")
    forward_mode: str = "incremental_runtime"
    backward_mode: str = "exact_diag_recompute_surrogate_vjp"
    alpha_schedule: FeedbackAlphaSchedule = field(default_factory=FeedbackAlphaSchedule)
    drift: ParentProjectionDriftPolicy = field(default_factory=ParentProjectionDriftPolicy)


@dataclass(frozen=True)
class RelationFeedbackPolicy:
    enable: bool = False
    branches: tuple[str, ...] = ("bg", "distant")
    differentiable_diag_cov: bool = True
    checkpoint: bool = True
    grad_to_child_geometry: bool = True
    grad_to_parent_geometry: bool = True
    grad_to_child_code: bool = False
    grad_to_parent_event: bool = True
    grad_to_support: bool = False
    alpha_schedule: FeedbackAlphaSchedule = field(default_factory=FeedbackAlphaSchedule)


@dataclass(frozen=True)
class ObservationFeedbackDebugPolicy:
    grad_probe_interval: int = 500
    forward_parity_interval: int = 1000
    log_feedback_memory: bool = True


@dataclass(frozen=True)
class ObservationFeedbackSchedulePolicy:
    """Clock used by all feedback-alpha schedules.

    ``global_step`` preserves legacy configurations.  ``activation_step``
    starts the schedule at zero when feedback is activated in an existing
    training run (for example when migrating a Stage3.2 checkpoint).
    """

    origin: str = "global_step"
    activation_step: int = 0

    def local_step(self, global_step: int, *, activation_step: int | None = None) -> int:
        if isinstance(global_step, bool) or not isinstance(global_step, int):
            raise TypeError("feedback global_step must be an integer")
        if self.origin == "global_step":
            return max(0, int(global_step))
        origin = int(self.activation_step if activation_step is None else activation_step)
        if origin < 0:
            raise ValueError("feedback activation_step must be >= 0")
        return max(0, int(global_step) - origin)


_DISTRIBUTIONS = ("repeat_refine", "shuffled_coverage", "high_block_repair")
_FEEDBACK_MODES = {
    FeedbackMode.TRAINABLE_CHECKPOINTED,
    FeedbackMode.FROZEN_INPUT_GRAD_CHECKPOINTED,
    FeedbackMode.FROZEN_NO_GRAD,
}


@dataclass(frozen=True)
class ObservationFeedbackPolicy:
    enable: bool = False
    scope: str = "within_rollout"
    schedule: ObservationFeedbackSchedulePolicy = field(default_factory=ObservationFeedbackSchedulePolicy)
    modes: Mapping[str, FeedbackMode] = field(default_factory=dict)
    source_render: SourceRenderFeedbackPolicy = field(default_factory=SourceRenderFeedbackPolicy)
    parent_projection: ParentProjectionFeedbackPolicy = field(default_factory=ParentProjectionFeedbackPolicy)
    relation: RelationFeedbackPolicy = field(default_factory=RelationFeedbackPolicy)
    scalar_anchor_geometry_grad: bool = False
    discrete_routing_grad: bool = False
    rollout_boundary_grad: bool = False
    debug: ObservationFeedbackDebugPolicy = field(default_factory=ObservationFeedbackDebugPolicy)

    @classmethod
    def from_config(cls, raw: Any) -> "ObservationFeedbackPolicy":
        outer = _mapping(raw, path="config")
        if "model" in outer:
            model_cfg = _mapping(outer["model"], path="config.model")
            iforward_cfg = _mapping(model_cfg.get("iforward", {}), path="config.model.iforward")
            raw = iforward_cfg.get("observation_feedback", None)
        elif "observation_feedback" in outer:
            raw = outer["observation_feedback"]
        else:
            feedback_keys = {
                "enable",
                "scope",
                "schedule",
                "modes",
                "source_render",
                "parent_projection",
                "relation",
                "scalar_anchor",
                "discrete_routing_grad",
                "rollout_boundary_grad",
                "debug",
            }
            raw = outer if bool(set(outer) & feedback_keys) else {}
        root = _section(
            raw,
            path="observation_feedback",
            allowed={
                "enable",
                "scope",
                "schedule",
                "modes",
                "source_render",
                "parent_projection",
                "relation",
                "scalar_anchor",
                "discrete_routing_grad",
                "rollout_boundary_grad",
                "debug",
            },
        )
        enabled = _bool(root, "enable", False, path="observation_feedback")
        scope = _str(root, "scope", "within_rollout", path="observation_feedback")
        if scope != "within_rollout":
            raise ValueError("observation_feedback.scope only supports 'within_rollout'")

        schedule_raw = _section(
            root.get("schedule", {}),
            path="observation_feedback.schedule",
            allowed={"origin", "activation_step"},
        )
        schedule_origin = _str(
            schedule_raw,
            "origin",
            "global_step",
            path="observation_feedback.schedule",
        )
        if schedule_origin not in {"global_step", "activation_step"}:
            raise ValueError(
                "observation_feedback.schedule.origin must be 'global_step' or 'activation_step'"
            )
        schedule = ObservationFeedbackSchedulePolicy(
            origin=schedule_origin,
            activation_step=_int(
                schedule_raw,
                "activation_step",
                0,
                path="observation_feedback.schedule",
                minimum=0,
            ),
        )

        modes_raw = _mapping(root.get("modes", {}), path="observation_feedback.modes")
        unknown_distributions = sorted(set(modes_raw) - set(_DISTRIBUTIONS))
        if unknown_distributions:
            raise ValueError(
                "observation_feedback.modes contains unsupported distributions: " + ", ".join(unknown_distributions)
            )
        if enabled and set(modes_raw) != set(_DISTRIBUTIONS):
            missing = sorted(set(_DISTRIBUTIONS) - set(modes_raw))
            raise ValueError("observation_feedback.modes must define all distributions; missing: " + ", ".join(missing))
        modes = {
            name: FeedbackMode.parse(value, path=f"observation_feedback.modes.{name}")
            for name, value in modes_raw.items()
        }
        if enabled:
            unsupported_modes = {name: mode for name, mode in modes.items() if mode not in _FEEDBACK_MODES}
            if unsupported_modes:
                rendered = ", ".join(f"{name}={mode.value}" for name, mode in unsupported_modes.items())
                raise ValueError(f"observation feedback requires explicit checkpoint/no-grad modes, got: {rendered}")

        source_raw = _section(
            root.get("source_render", {}),
            path="observation_feedback.source_render",
            allowed={"enable", "renderer_mode", "checkpoint_scope", "absgrad", "alpha_schedule"},
        )
        source = SourceRenderFeedbackPolicy(
            enable=_bool(source_raw, "enable", False, path="observation_feedback.source_render"),
            renderer_mode=_str(
                source_raw, "renderer_mode", "differentiable_rgb", path="observation_feedback.source_render"
            ),
            checkpoint_scope=_str(
                source_raw,
                "checkpoint_scope",
                "full_dynamic_observation",
                path="observation_feedback.source_render",
            ),
            absgrad=_bool(source_raw, "absgrad", False, path="observation_feedback.source_render"),
            alpha_schedule=FeedbackAlphaSchedule.from_config(
                source_raw.get("alpha_schedule", [[0, 0.0]]),
                path="observation_feedback.source_render.alpha_schedule",
            ),
        )
        if source.enable:
            if source.renderer_mode != "differentiable_rgb":
                raise ValueError("source_render.renderer_mode must be 'differentiable_rgb' when enabled")
            if source.checkpoint_scope != "full_dynamic_observation":
                raise ValueError("source_render.checkpoint_scope must be 'full_dynamic_observation' when enabled")
            if source.absgrad:
                raise ValueError("source_render.absgrad must remain false for observation feedback")

        parent_raw = _section(
            root.get("parent_projection", {}),
            path="observation_feedback.parent_projection",
            allowed={"enable", "branches", "forward_mode", "backward_mode", "alpha_schedule", "drift"},
        )
        drift_raw = _section(
            parent_raw.get("drift", {}),
            path="observation_feedback.parent_projection.drift",
            allowed={"check_interval", "warn_threshold", "skip_vjp_threshold", "exact_refresh_threshold"},
        )
        drift = ParentProjectionDriftPolicy(
            check_interval=_int(
                drift_raw,
                "check_interval",
                500,
                path="observation_feedback.parent_projection.drift",
                minimum=1,
            ),
            warn_threshold=_float(
                drift_raw, "warn_threshold", 1.0e-3, path="observation_feedback.parent_projection.drift"
            ),
            skip_vjp_threshold=_float(
                drift_raw, "skip_vjp_threshold", 5.0e-3, path="observation_feedback.parent_projection.drift"
            ),
            exact_refresh_threshold=_float(
                drift_raw,
                "exact_refresh_threshold",
                1.0e-2,
                path="observation_feedback.parent_projection.drift",
            ),
        )
        if not 0.0 <= drift.warn_threshold < drift.skip_vjp_threshold <= drift.exact_refresh_threshold:
            raise ValueError(
                "parent_projection drift thresholds require 0 <= warn < skip_vjp <= exact_refresh"
            )
        parent = ParentProjectionFeedbackPolicy(
            enable=_bool(parent_raw, "enable", False, path="observation_feedback.parent_projection"),
            branches=_branches(parent_raw, path="observation_feedback.parent_projection"),
            forward_mode=_str(
                parent_raw,
                "forward_mode",
                "incremental_runtime",
                path="observation_feedback.parent_projection",
            ),
            backward_mode=_str(
                parent_raw,
                "backward_mode",
                "exact_diag_recompute_surrogate_vjp",
                path="observation_feedback.parent_projection",
            ),
            alpha_schedule=FeedbackAlphaSchedule.from_config(
                parent_raw.get("alpha_schedule", [[0, 0.0]]),
                path="observation_feedback.parent_projection.alpha_schedule",
            ),
            drift=drift,
        )
        if parent.enable:
            if parent.forward_mode != "incremental_runtime":
                raise ValueError("parent_projection.forward_mode must be 'incremental_runtime' when enabled")
            if parent.backward_mode != "exact_diag_recompute_surrogate_vjp":
                raise ValueError(
                    "parent_projection.backward_mode must be 'exact_diag_recompute_surrogate_vjp' when enabled"
                )

        relation_raw = _section(
            root.get("relation", {}),
            path="observation_feedback.relation",
            allowed={
                "enable",
                "branches",
                "differentiable_diag_cov",
                "checkpoint",
                "grad_to_child_geometry",
                "grad_to_parent_geometry",
                "grad_to_child_code",
                "grad_to_parent_event",
                "grad_to_support",
                "alpha_schedule",
            },
        )
        relation = RelationFeedbackPolicy(
            enable=_bool(relation_raw, "enable", False, path="observation_feedback.relation"),
            branches=_branches(relation_raw, path="observation_feedback.relation"),
            differentiable_diag_cov=_bool(
                relation_raw, "differentiable_diag_cov", True, path="observation_feedback.relation"
            ),
            checkpoint=_bool(relation_raw, "checkpoint", True, path="observation_feedback.relation"),
            grad_to_child_geometry=_bool(
                relation_raw, "grad_to_child_geometry", True, path="observation_feedback.relation"
            ),
            grad_to_parent_geometry=_bool(
                relation_raw, "grad_to_parent_geometry", True, path="observation_feedback.relation"
            ),
            grad_to_child_code=_bool(
                relation_raw, "grad_to_child_code", False, path="observation_feedback.relation"
            ),
            grad_to_parent_event=_bool(
                relation_raw, "grad_to_parent_event", True, path="observation_feedback.relation"
            ),
            grad_to_support=_bool(relation_raw, "grad_to_support", False, path="observation_feedback.relation"),
            alpha_schedule=FeedbackAlphaSchedule.from_config(
                relation_raw.get("alpha_schedule", [[0, 0.0]]),
                path="observation_feedback.relation.alpha_schedule",
            ),
        )
        if relation.enable:
            if not relation.differentiable_diag_cov:
                raise ValueError("relation.differentiable_diag_cov must be true when relation feedback is enabled")
            if relation.grad_to_support:
                raise ValueError("relation.grad_to_support=true is not supported")
            if relation.grad_to_parent_geometry and not parent.enable:
                raise ValueError("relation.grad_to_parent_geometry=true requires parent_projection.enable=true")

        scalar_raw = _section(
            root.get("scalar_anchor", {}),
            path="observation_feedback.scalar_anchor",
            allowed={"geometry_grad"},
        )
        scalar_geometry_grad = _bool(
            scalar_raw, "geometry_grad", False, path="observation_feedback.scalar_anchor"
        )
        discrete_routing_grad = _bool(
            root, "discrete_routing_grad", False, path="observation_feedback"
        )
        rollout_boundary_grad = _bool(
            root, "rollout_boundary_grad", False, path="observation_feedback"
        )
        if scalar_geometry_grad:
            raise ValueError("scalar_anchor.geometry_grad=true is not supported")
        if discrete_routing_grad:
            raise ValueError("observation_feedback.discrete_routing_grad=true is not supported")
        if rollout_boundary_grad:
            raise ValueError("observation_feedback.rollout_boundary_grad=true is not supported")

        debug_raw = _section(
            root.get("debug", {}),
            path="observation_feedback.debug",
            allowed={"grad_probe_interval", "forward_parity_interval", "log_feedback_memory"},
        )
        debug = ObservationFeedbackDebugPolicy(
            grad_probe_interval=_int(
                debug_raw, "grad_probe_interval", 500, path="observation_feedback.debug", minimum=0
            ),
            forward_parity_interval=_int(
                debug_raw, "forward_parity_interval", 1000, path="observation_feedback.debug", minimum=0
            ),
            log_feedback_memory=_bool(
                debug_raw, "log_feedback_memory", True, path="observation_feedback.debug"
            ),
        )

        return cls(
            enable=enabled,
            scope=scope,
            schedule=schedule,
            modes=dict(modes),
            source_render=source,
            parent_projection=parent,
            relation=relation,
            scalar_anchor_geometry_grad=scalar_geometry_grad,
            discrete_routing_grad=discrete_routing_grad,
            rollout_boundary_grad=rollout_boundary_grad,
            debug=debug,
        )

    def mode_for(self, distribution_type: str) -> FeedbackMode:
        try:
            return self.modes[str(distribution_type)]
        except KeyError as exc:
            raise KeyError(f"no observation feedback mode configured for distribution {distribution_type!r}") from exc

    def mode_for_visit(self, visit_meta: Mapping[str, Any]) -> FeedbackMode:
        if not isinstance(visit_meta, Mapping):
            raise TypeError("visit_meta must be a mapping")
        eval_mode_raw = visit_meta.get("observation_feedback_eval_mode", None)
        if eval_mode_raw is not None:
            eval_mode = FeedbackMode.parse(
                eval_mode_raw, path="visit_meta.observation_feedback_eval_mode"
            )
            if eval_mode is not FeedbackMode.FROZEN_NO_GRAD:
                raise ValueError(
                    "observation_feedback_eval_mode must be 'frozen_no_grad'; "
                    f"got {eval_mode.value!r}"
                )
            return eval_mode
        nested_raw = visit_meta.get("iforward_stage3_2", {})
        nested = _mapping(nested_raw, path="visit_meta.iforward_stage3_2") if nested_raw is not None else {}
        distribution = str(visit_meta.get("distribution_type", nested.get("distribution_type", "")) or "")
        raw_mode = visit_meta.get("train_2d_mode", nested.get("train_2d_mode", None))
        if raw_mode is None:
            if not distribution:
                raise ValueError("visit_meta must contain train_2d_mode or distribution_type")
            return self.mode_for(distribution)
        mode = FeedbackMode.parse(raw_mode, path="visit_meta.train_2d_mode")
        if distribution:
            expected = self.mode_for(distribution)
            if mode is not expected:
                raise ValueError(
                    f"visit_meta train_2d_mode={mode.value!r} does not match configured "
                    f"{distribution!r} mode={expected.value!r}"
                )
        return mode

    def validate_scheduler_modes(self, scheduler_modes: Mapping[str, str]) -> None:
        actual = {str(name): FeedbackMode.parse(value, path=f"scheduler.train_2d_policy.{name}") for name, value in scheduler_modes.items()}
        expected = dict(self.modes)
        if actual != expected:
            actual_text = {name: mode.value for name, mode in actual.items()}
            expected_text = {name: mode.value for name, mode in expected.items()}
            raise ValueError(
                "scheduler train_2d_policy must exactly match observation_feedback.modes; "
                f"scheduler={actual_text}, feedback={expected_text}"
            )

    def alpha_for(self, component: str, step: int) -> float:
        if not self.enable:
            return 0.0
        if component == "source_render":
            return self.source_render.alpha_schedule(step) if self.source_render.enable else 0.0
        if component == "parent_projection":
            return self.parent_projection.alpha_schedule(step) if self.parent_projection.enable else 0.0
        if component == "relation":
            return self.relation.alpha_schedule(step) if self.relation.enable else 0.0
        raise KeyError(f"unknown observation feedback component {component!r}")

    def schedule_step(self, global_step: int, *, activation_step: int | None = None) -> int:
        return self.schedule.local_step(int(global_step), activation_step=activation_step)

    def source_alpha(self, step: int) -> float:
        return self.alpha_for("source_render", step)

    def parent_alpha(self, step: int) -> float:
        return self.alpha_for("parent_projection", step)

    def relation_alpha(self, step: int) -> float:
        return self.alpha_for("relation", step)

    @property
    def any_continuous_feedback_enabled(self) -> bool:
        return bool(
            self.enable
            and (self.source_render.enable or self.parent_projection.enable or self.relation.enable)
        )


class FrontendParameterModeScope(AbstractContextManager["FrontendParameterModeScope"]):
    """Temporarily freeze the exact frontend parameter set across forward/backward/step."""

    def __init__(
        self,
        module: nn.Module,
        parameter_names: Iterable[str],
        mode: FeedbackMode | str,
    ) -> None:
        self.mode = FeedbackMode.parse(mode)
        if not isinstance(module, nn.Module):
            raise TypeError(f"frontend module must be torch.nn.Module, got {type(module).__name__}")
        available = dict(module.named_parameters())
        unique: list[nn.Parameter] = []
        normalized_names: list[str] = []
        seen: set[int] = set()
        missing: list[str] = []
        for raw_name in parameter_names:
            name = str(raw_name)
            parameter = available.get(name)
            if parameter is None:
                missing.append(name)
                continue
            if id(parameter) not in seen:
                seen.add(id(parameter))
                unique.append(parameter)
                normalized_names.append(name)
        if missing:
            raise ValueError("frontend parameter_names are not present in module: " + ", ".join(sorted(missing)))
        self.module = module
        self.parameter_names = tuple(normalized_names)
        self.parameters = tuple(unique)
        self._original_requires_grad: tuple[bool, ...] | None = None

    @property
    def active(self) -> bool:
        return self._original_requires_grad is not None

    def __enter__(self) -> "FrontendParameterModeScope":
        if self.active:
            raise RuntimeError("FrontendParameterModeScope cannot be entered more than once")
        self._original_requires_grad = tuple(bool(parameter.requires_grad) for parameter in self.parameters)
        if self.mode.freezes_frontend_parameters:
            for parameter in self.parameters:
                parameter.requires_grad_(False)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        original = self._original_requires_grad
        if original is None:
            raise RuntimeError("FrontendParameterModeScope exited before it was entered")
        try:
            for parameter, requires_grad in zip(self.parameters, original):
                parameter.requires_grad_(requires_grad)
        finally:
            self._original_requires_grad = None
        return None


__all__ = [
    "FeedbackAlphaSchedule",
    "FeedbackMode",
    "FrontendParameterModeScope",
    "ObservationFeedbackDebugPolicy",
    "ObservationFeedbackPolicy",
    "ObservationFeedbackSchedulePolicy",
    "ParentProjectionDriftPolicy",
    "ParentProjectionFeedbackPolicy",
    "RelationFeedbackPolicy",
    "SourceRenderFeedbackPolicy",
    "scale_feedback",
]
