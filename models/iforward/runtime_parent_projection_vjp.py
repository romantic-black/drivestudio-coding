from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Mapping, Optional, Sequence

import torch
from torch.autograd.function import once_differentiable

from .biggs_parent_projector import BigGSParentProjection
from .biggs_parent_projector_diag import (
    mass_mode_to_id,
    project_biggs_parent_diag_reference_tensors,
)
from .utils import cfg_get


_PARAM_KEYS = (
    "means",
    "scales_log",
    "quats",
    "opacity_logit",
    "sh_dc",
    "sh_rest",
)
_GUARDED_OUTPUT_INDICES = (0, 1, 3, 4, 5)


@dataclass(frozen=True)
class ParentVJPDriftPolicy:
    """Reliability policy for the incremental-forward/exact-backward bridge."""

    check_interval: int = 500
    warn_threshold: float = 1.0e-3
    skip_vjp_threshold: float = 5.0e-3
    exact_refresh_threshold: float = 1.0e-2
    rms_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        if int(self.check_interval) < 0:
            raise ValueError("parent VJP drift check_interval must be >= 0")
        warn = float(self.warn_threshold)
        skip = float(self.skip_vjp_threshold)
        refresh = float(self.exact_refresh_threshold)
        if not (0.0 <= warn < skip <= refresh):
            raise ValueError(
                "parent VJP drift thresholds must satisfy "
                "0 <= warn_threshold < skip_vjp_threshold <= exact_refresh_threshold"
            )
        if float(self.rms_floor) <= 0.0:
            raise ValueError("parent VJP drift rms_floor must be > 0")

    @classmethod
    def from_config(cls, cfg: Any) -> "ParentVJPDriftPolicy":
        if cfg is None:
            return cls()
        return cls(
            check_interval=int(cfg_get(cfg, "check_interval", 500)),
            warn_threshold=float(cfg_get(cfg, "warn_threshold", 1.0e-3)),
            skip_vjp_threshold=float(cfg_get(cfg, "skip_vjp_threshold", 5.0e-3)),
            exact_refresh_threshold=float(cfg_get(cfg, "exact_refresh_threshold", 1.0e-2)),
            rms_floor=float(cfg_get(cfg, "rms_floor", 1.0e-6)),
        )

    def should_sample(self, step: int) -> bool:
        return int(self.check_interval) > 0 and int(step) >= 0 and int(step) % int(self.check_interval) == 0


@dataclass(frozen=True)
class RuntimeParentProjectionDrift:
    means_rel_error: torch.Tensor
    scales_rel_error: torch.Tensor
    quats_rel_error: torch.Tensor
    opacity_rel_error: torch.Tensor
    sh_dc_rel_error: torch.Tensor
    sh_rest_rel_error: torch.Tensor
    sh_rel_error: torch.Tensor
    max_rel_error: torch.Tensor
    nominal_alpha: torch.Tensor
    effective_alpha: torch.Tensor
    vjp_skipped: torch.Tensor
    refresh_required: torch.Tensor

    def metric_tensors(self, *, prefix: str) -> Dict[str, torch.Tensor]:
        prefix_s = str(prefix).rstrip("/")
        return {
            f"{prefix_s}/means_rel_error": self.means_rel_error,
            f"{prefix_s}/scales_rel_error": self.scales_rel_error,
            f"{prefix_s}/quats_rel_error": self.quats_rel_error,
            f"{prefix_s}/opacity_rel_error": self.opacity_rel_error,
            f"{prefix_s}/sh_dc_rel_error": self.sh_dc_rel_error,
            f"{prefix_s}/sh_rest_rel_error": self.sh_rest_rel_error,
            f"{prefix_s}/sh_rel_error": self.sh_rel_error,
            f"{prefix_s}/max_rel_error": self.max_rel_error,
            f"{prefix_s}/nominal_alpha": self.nominal_alpha,
            f"{prefix_s}/effective_alpha": self.effective_alpha,
            f"{prefix_s}/vjp_skipped": self.vjp_skipped,
            f"{prefix_s}/refresh_required": self.refresh_required,
        }

    def requires_exact_refresh(self) -> bool:
        """Synchronize only at a configured sampled-forward refresh decision."""

        return bool(self.refresh_required.detach().cpu().item())


class RuntimeParentVJPDriftCollector:
    """Stores detached scalar reports produced during custom backward."""

    def __init__(self) -> None:
        self._records: DefaultDict[str, list[RuntimeParentProjectionDrift]] = defaultdict(list)

    def record(self, branch: str, report: RuntimeParentProjectionDrift) -> None:
        branch_l = _validate_branch(branch)
        self._records[branch_l].append(report)

    def records(self, branch: str) -> tuple[RuntimeParentProjectionDrift, ...]:
        return tuple(self._records.get(_validate_branch(branch), ()))

    def latest(self, branch: str) -> Optional[RuntimeParentProjectionDrift]:
        records = self._records.get(_validate_branch(branch), ())
        return records[-1] if records else None

    def clear(self) -> None:
        self._records.clear()


@dataclass(frozen=True)
class _ReferenceProjectorConfig:
    min_scale: float
    max_scale: float
    opacity_cap: float
    opacity_min: float
    tau_parent_scale: float
    eps: float
    min_mass: float
    mass_mode: str

    @classmethod
    def from_config(cls, cfg: Any, *, max_scale: Optional[float]) -> "_ReferenceProjectorConfig":
        covariance_mode = str(cfg_get(cfg, "covariance_mode", "diagonal")).lower()
        if covariance_mode not in {"diag", "diagonal", "exact_diag", "exact_diagonal"}:
            raise ValueError(
                "parent surrogate VJP requires exact diagonal covariance, "
                f"got covariance_mode={covariance_mode!r}"
            )
        mass_mode = str(cfg_get(cfg, "mass_mode", "dynamic_tau_area"))
        mass_mode_to_id(mass_mode)
        out = cls(
            min_scale=float(cfg_get(cfg, "min_scale", 1.0e-3)),
            max_scale=float(max_scale if max_scale is not None else cfg_get(cfg, "max_scale", 10.0)),
            opacity_cap=float(cfg_get(cfg, "opacity_cap", 0.98)),
            opacity_min=float(cfg_get(cfg, "opacity_min", 1.0e-6)),
            tau_parent_scale=float(cfg_get(cfg, "tau_parent_scale", 1.0)),
            eps=float(cfg_get(cfg, "eps", cfg_get(cfg, "covariance_eps", 1.0e-6))),
            min_mass=float(cfg_get(cfg, "min_child_mass", 1.0e-8)),
            mass_mode=mass_mode,
        )
        if out.min_scale <= 0.0 or out.max_scale < out.min_scale:
            raise ValueError("parent surrogate VJP requires 0 < min_scale <= max_scale")
        if not (0.0 < out.opacity_min < out.opacity_cap < 1.0):
            raise ValueError("parent surrogate VJP requires 0 < opacity_min < opacity_cap < 1")
        if out.tau_parent_scale < 0.0:
            raise ValueError("parent surrogate VJP tau_parent_scale must be >= 0")
        if out.eps <= 0.0 or out.min_mass <= 0.0:
            raise ValueError("parent surrogate VJP eps and min_child_mass must be > 0")
        return out


def _validate_branch(branch: str) -> str:
    branch_l = str(branch).lower()
    if branch_l not in {"bg", "distant"}:
        raise ValueError(f"parent surrogate VJP only supports bg/distant, got branch={branch!r}")
    return branch_l


def _validate_param_dict(params: Mapping[str, torch.Tensor], *, label: str) -> tuple[torch.Tensor, ...]:
    missing = [key for key in _PARAM_KEYS if key not in params]
    if missing:
        raise KeyError(f"{label} is missing parent projection tensors: {missing}")
    tensors = tuple(params[key] for key in _PARAM_KEYS)
    if any(not torch.is_tensor(value) for value in tensors):
        raise TypeError(f"{label} parent projection values must all be tensors")
    ref = tensors[0]
    if not torch.is_floating_point(ref):
        raise TypeError(f"{label}.means must be floating point")
    n = int(ref.shape[0])
    expected_tail = ((3,), (3,), (4,), (1,), (3,))
    for key, value, tail in zip(_PARAM_KEYS[:5], tensors[:5], expected_tail):
        if value.device != ref.device:
            raise ValueError(f"{label}.{key} device {value.device} != means device {ref.device}")
        if not torch.is_floating_point(value):
            raise TypeError(f"{label}.{key} must be floating point")
        if int(value.shape[0]) != n or tuple(value.shape[1:]) != tail:
            raise ValueError(f"{label}.{key} has invalid shape {tuple(value.shape)}")
    sh_rest = tensors[5]
    if sh_rest.device != ref.device or not torch.is_floating_point(sh_rest):
        raise ValueError(f"{label}.sh_rest must be a floating tensor on {ref.device}")
    if sh_rest.dim() != 3 or int(sh_rest.shape[0]) != n or int(sh_rest.shape[-1]) != 3:
        raise ValueError(f"{label}.sh_rest has invalid shape {tuple(sh_rest.shape)}")
    return tensors


def _validate_runtime_projection(projection: BigGSParentProjection) -> tuple[torch.Tensor, ...]:
    if not isinstance(projection, BigGSParentProjection):
        raise TypeError("runtime_projection must be BigGSParentProjection")
    if float(projection.aux_stats.get("parent_runtime_incremental", 0.0)) != 1.0:
        raise ValueError("parent surrogate VJP requires an incremental runtime projection")
    if float(projection.aux_stats.get("projector_covariance_mode_id", -1.0)) != 1.0:
        raise ValueError("parent surrogate VJP requires runtime covariance_mode=diagonal")
    tensors = _validate_param_dict(projection.params, label="runtime_projection.params")
    for key, value in zip(_PARAM_KEYS, tensors):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(
                "incremental parent runtime must remain graph-free before surrogate VJP; "
                f"runtime_projection.params[{key!r}] is attached"
            )
    for label, value in (
        ("child_mass_sum", projection.child_mass_sum),
        ("child_mass_mean", projection.child_mass_mean),
    ):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(f"incremental parent runtime {label} must remain graph-free")
    return tensors


def _validate_assignment(
    *,
    child_params: Sequence[torch.Tensor],
    child_mass: torch.Tensor,
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
) -> None:
    n = int(child_params[0].shape[0])
    if not torch.is_floating_point(child_mass):
        raise TypeError("parent surrogate VJP child_mass must be floating point")
    if child_to_parent.dtype != torch.long or parent_count.dtype != torch.long:
        raise TypeError("parent surrogate VJP child_to_parent and parent_count must use torch.long")
    for label, value in (
        ("child_mass", child_mass),
        ("child_to_parent", child_to_parent),
        ("parent_count", parent_count),
    ):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(f"parent surrogate VJP discrete/support input {label} must be graph-free")
    if int(child_mass.numel()) != n or int(child_to_parent.numel()) != n:
        raise ValueError(
            "parent surrogate VJP assignment length mismatch: "
            f"children={n}, child_mass={int(child_mass.numel())}, "
            f"child_to_parent={int(child_to_parent.numel())}"
        )
    counts = parent_count.detach().to(dtype=torch.long).reshape(-1)
    if bool((counts < 0).any().item()) or int(counts.sum().item()) != n:
        raise ValueError("parent surrogate VJP parent_count must be non-negative and sum to num_children")
    if n > 0:
        pid = child_to_parent.detach().to(dtype=torch.long).reshape(-1)
        m = int(counts.numel())
        if m == 0 or bool(((pid < 0) | (pid >= m)).any().item()):
            raise ValueError("parent surrogate VJP child_to_parent contains an invalid parent index")
        actual_counts = torch.bincount(pid, minlength=m)
        if not torch.equal(actual_counts.to(device=counts.device), counts):
            raise ValueError("parent surrogate VJP parent_count does not match child_to_parent")


def _autocast_disabled(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _reference_outputs_fp32(
    child_params: Sequence[torch.Tensor],
    *,
    child_mass: torch.Tensor,
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
    config: _ReferenceProjectorConfig,
) -> tuple[torch.Tensor, ...]:
    means = child_params[0]
    with _autocast_disabled(means.device):
        return project_biggs_parent_diag_reference_tensors(
            means=child_params[0].float(),
            scales_log=child_params[1].float(),
            quats=child_params[2].float(),
            opacity_logit=child_params[3].float(),
            sh_dc=child_params[4].float(),
            sh_rest=child_params[5].float(),
            child_mass=child_mass.detach().to(device=means.device, dtype=torch.float32),
            child_to_parent=child_to_parent.detach().to(device=means.device, dtype=torch.long),
            parent_count=parent_count.detach().to(device=means.device, dtype=torch.long),
            min_scale=float(config.min_scale),
            max_scale=float(config.max_scale),
            opacity_cap=float(config.opacity_cap),
            opacity_min=float(config.opacity_min),
            tau_parent_scale=float(config.tau_parent_scale),
            eps=float(config.eps),
            min_mass=float(config.min_mass),
            mass_mode=str(config.mass_mode),
        )


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor, *, floor: float) -> torch.Tensor:
    if tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(
            "parent surrogate VJP drift shape mismatch: "
            f"runtime={tuple(actual.shape)} exact={tuple(expected.shape)}"
        )
    actual_f = actual.detach().to(device=expected.device, dtype=torch.float32)
    expected_f = expected.detach().float()
    if int(actual_f.numel()) == 0:
        return expected_f.new_zeros(())
    error_rms = torch.sqrt(torch.mean((actual_f - expected_f).square()))
    actual_rms = torch.sqrt(torch.mean(actual_f.square()))
    expected_rms = torch.sqrt(torch.mean(expected_f.square()))
    denominator = torch.maximum(torch.maximum(actual_rms, expected_rms), expected_f.new_tensor(float(floor)))
    return (error_rms / denominator).detach()


def _drift_from_outputs(
    runtime_outputs: Sequence[torch.Tensor],
    exact_outputs: Sequence[torch.Tensor],
    *,
    nominal_alpha: torch.Tensor,
    policy: ParentVJPDriftPolicy,
) -> RuntimeParentProjectionDrift:
    errors = [
        _relative_rms(runtime_outputs[index], exact_outputs[index], floor=float(policy.rms_floor))
        for index in range(6)
    ]
    sh_rel = torch.maximum(errors[4], errors[5])
    guarded = torch.stack((errors[0], errors[1], errors[3], errors[4], errors[5])).max()
    warn = guarded.new_tensor(float(policy.warn_threshold))
    skip = guarded.new_tensor(float(policy.skip_vjp_threshold))
    transition = ((skip - guarded) / (skip - warn)).clamp(0.0, 1.0)
    guard_scale = torch.where(guarded <= warn, torch.ones_like(guarded), transition)
    guard_scale = torch.where(guarded >= skip, torch.zeros_like(guard_scale), guard_scale)
    nominal = nominal_alpha.detach().to(device=guarded.device, dtype=torch.float32).reshape(())
    effective = (nominal * guard_scale).detach()
    return RuntimeParentProjectionDrift(
        means_rel_error=errors[0],
        scales_rel_error=errors[1],
        quats_rel_error=errors[2],
        opacity_rel_error=errors[3],
        sh_dc_rel_error=errors[4],
        sh_rest_rel_error=errors[5],
        sh_rel_error=sh_rel,
        max_rel_error=guarded,
        nominal_alpha=nominal,
        effective_alpha=effective,
        vjp_skipped=(guarded >= skip).to(dtype=torch.float32).detach(),
        refresh_required=(guarded >= guarded.new_tensor(float(policy.exact_refresh_threshold)))
        .to(dtype=torch.float32)
        .detach(),
    )


def _record_report(
    collector: Optional[RuntimeParentVJPDriftCollector],
    *,
    branch: str,
    report: RuntimeParentProjectionDrift,
) -> None:
    if collector is not None:
        collector.record(branch, report)


class RuntimeParentProjectionVJPFn(torch.autograd.Function):
    """Identity runtime forward with an FP32 exact-diagonal surrogate VJP."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        runtime_means: torch.Tensor,
        runtime_scales_log: torch.Tensor,
        runtime_quats: torch.Tensor,
        runtime_opacity_logit: torch.Tensor,
        runtime_sh_dc: torch.Tensor,
        runtime_sh_rest: torch.Tensor,
        child_means: torch.Tensor,
        child_scales_log: torch.Tensor,
        child_quats: torch.Tensor,
        child_opacity_logit: torch.Tensor,
        child_sh_dc: torch.Tensor,
        child_sh_rest: torch.Tensor,
        child_mass: torch.Tensor,
        child_to_parent: torch.Tensor,
        parent_count: torch.Tensor,
        nominal_alpha: torch.Tensor,
        reference_config: _ReferenceProjectorConfig,
        drift_policy: ParentVJPDriftPolicy,
        drift_collector: Optional[RuntimeParentVJPDriftCollector],
        branch: str,
    ) -> tuple[torch.Tensor, ...]:
        runtime_outputs = (
            runtime_means,
            runtime_scales_log,
            runtime_quats,
            runtime_opacity_logit,
            runtime_sh_dc,
            runtime_sh_rest,
        )
        child_params = (
            child_means,
            child_scales_log,
            child_quats,
            child_opacity_logit,
            child_sh_dc,
            child_sh_rest,
        )
        ctx.save_for_backward(
            *(value.detach() for value in runtime_outputs),
            *(value.detach() for value in child_params),
            child_mass.detach(),
            child_to_parent.detach(),
            parent_count.detach(),
            nominal_alpha.detach(),
        )
        ctx.reference_config = reference_config
        ctx.drift_policy = drift_policy
        ctx.drift_collector = drift_collector
        ctx.branch = branch
        return tuple(value.detach() for value in runtime_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grad_runtime_outputs: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        saved = ctx.saved_tensors
        runtime_outputs = saved[:6]
        child_values = saved[6:12]
        child_mass, child_to_parent, parent_count, nominal_alpha = saved[12:16]
        child_needs_grad = tuple(bool(value) for value in ctx.needs_input_grad[6:12])
        child_recompute = tuple(
            value.detach().to(dtype=torch.float32).requires_grad_(need_grad)
            for value, need_grad in zip(child_values, child_needs_grad)
        )

        child_grads: list[Optional[torch.Tensor]] = [None] * 6
        with torch.autograd.profiler.record_function(
            "iforward/feedback/parent_exact_vjp_recompute"
        ):
            with torch.enable_grad():
                exact_outputs = _reference_outputs_fp32(
                    child_recompute,
                    child_mass=child_mass,
                    child_to_parent=child_to_parent,
                    parent_count=parent_count,
                    config=ctx.reference_config,
                )
                report = _drift_from_outputs(
                    runtime_outputs,
                    exact_outputs,
                    nominal_alpha=nominal_alpha,
                    policy=ctx.drift_policy,
                )
                _record_report(ctx.drift_collector, branch=ctx.branch, report=report)

                selected_outputs: list[torch.Tensor] = []
                selected_grad_outputs: list[torch.Tensor] = []
                for index in _GUARDED_OUTPUT_INDICES:
                    grad_output = grad_runtime_outputs[index]
                    exact_output = exact_outputs[index]
                    if grad_output is None or not exact_output.requires_grad:
                        continue
                    selected_outputs.append(exact_output)
                    selected_grad_outputs.append(
                        grad_output.detach().to(device=exact_output.device, dtype=torch.float32)
                        * report.effective_alpha
                    )
                active_indices = [index for index, need_grad in enumerate(child_needs_grad) if need_grad]
                if selected_outputs and active_indices:
                    active_inputs = [child_recompute[index] for index in active_indices]
                    recomputed_grads = torch.autograd.grad(
                        tuple(selected_outputs),
                        tuple(active_inputs),
                        grad_outputs=tuple(selected_grad_outputs),
                        allow_unused=True,
                        create_graph=False,
                    )
                    for index, grad in zip(active_indices, recomputed_grads):
                        child_grads[index] = None if grad is None else grad.to(dtype=child_values[index].dtype)

        # Six runtime inputs, six child inputs, then eight non-differentiable
        # tensor/config/control inputs.
        return (None,) * 6 + tuple(child_grads) + (None,) * 8


def _alpha_tensor(alpha: float | torch.Tensor, *, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(alpha):
        if alpha.requires_grad:
            raise ValueError("parent surrogate VJP alpha must not require gradients")
        if int(alpha.numel()) != 1:
            raise ValueError("parent surrogate VJP alpha must be scalar")
        out = alpha.detach().to(device=ref.device, dtype=torch.float32).reshape(())
    else:
        out = ref.new_tensor(float(alpha), dtype=torch.float32)
    if not bool(torch.isfinite(out).item()) or not (0.0 <= float(out.item()) <= 1.0):
        raise ValueError("parent surrogate VJP alpha must be finite and in [0, 1]")
    return out


def parent_projection_feedback(
    runtime_projection: BigGSParentProjection,
    *,
    child_params: Mapping[str, torch.Tensor],
    child_mass: torch.Tensor,
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
    projector_cfg: Any,
    alpha: float | torch.Tensor,
    drift_policy: ParentVJPDriftPolicy | Any | None = None,
    drift_collector: Optional[RuntimeParentVJPDriftCollector] = None,
    branch: str,
    max_scale: Optional[float] = None,
) -> BigGSParentProjection:
    """Attach an exact-diagonal surrogate VJP without changing runtime values."""

    branch_l = _validate_branch(branch)
    runtime_outputs = _validate_runtime_projection(runtime_projection)
    child_tensors = _validate_param_dict(child_params, label="child_params")
    _validate_assignment(
        child_params=child_tensors,
        child_mass=child_mass,
        child_to_parent=child_to_parent,
        parent_count=parent_count,
    )
    if int(runtime_outputs[0].shape[0]) != int(parent_count.numel()):
        raise ValueError("runtime parent row count does not match parent_count")
    if int(runtime_outputs[5].shape[1]) != int(child_tensors[5].shape[1]):
        raise ValueError("runtime and child sh_rest basis counts do not match")
    reference_config = _ReferenceProjectorConfig.from_config(projector_cfg, max_scale=max_scale)
    policy = (
        drift_policy
        if isinstance(drift_policy, ParentVJPDriftPolicy)
        else ParentVJPDriftPolicy.from_config(drift_policy)
    )
    if drift_collector is not None and not isinstance(drift_collector, RuntimeParentVJPDriftCollector):
        raise TypeError("drift_collector must be RuntimeParentVJPDriftCollector or None")
    alpha_t = _alpha_tensor(alpha, ref=child_tensors[0])
    n = int(child_tensors[0].shape[0])
    m = int(parent_count.numel())
    if n == 0 or m == 0:
        aux = dict(runtime_projection.aux_stats)
        aux.update({"parent_vjp_enabled": 1.0, "parent_vjp_empty_bypass": 1.0, "parent_vjp_alpha": float(alpha_t.item())})
        return BigGSParentProjection(
            params=dict(runtime_projection.params),
            child_mass_sum=runtime_projection.child_mass_sum,
            child_mass_mean=runtime_projection.child_mass_mean,
            aux_stats=aux,
        )

    attached_outputs = RuntimeParentProjectionVJPFn.apply(
        *runtime_outputs,
        *child_tensors,
        child_mass,
        child_to_parent,
        parent_count,
        alpha_t,
        reference_config,
        policy,
        drift_collector,
        branch_l,
    )
    aux = dict(runtime_projection.aux_stats)
    aux.update(
        {
            "parent_vjp_enabled": 1.0,
            "parent_vjp_empty_bypass": 0.0,
            "parent_vjp_alpha": float(alpha_t.item()),
            "parent_vjp_warn_threshold": float(policy.warn_threshold),
            "parent_vjp_skip_threshold": float(policy.skip_vjp_threshold),
            "parent_vjp_refresh_threshold": float(policy.exact_refresh_threshold),
        }
    )
    return BigGSParentProjection(
        params=dict(zip(_PARAM_KEYS, attached_outputs)),
        child_mass_sum=runtime_projection.child_mass_sum,
        child_mass_mean=runtime_projection.child_mass_mean,
        aux_stats=aux,
    )


@torch.no_grad()
def runtime_exact_drift(
    runtime_projection: BigGSParentProjection,
    *,
    child_params: Mapping[str, torch.Tensor],
    child_mass: torch.Tensor,
    child_to_parent: torch.Tensor,
    parent_count: torch.Tensor,
    projector_cfg: Any,
    alpha: float | torch.Tensor,
    drift_policy: ParentVJPDriftPolicy | Any | None = None,
    branch: str,
    max_scale: Optional[float] = None,
) -> RuntimeParentProjectionDrift:
    """Sample runtime-vs-exact drift outside backward for refresh decisions."""

    _validate_branch(branch)
    runtime_outputs = _validate_runtime_projection(runtime_projection)
    child_tensors = _validate_param_dict(child_params, label="child_params")
    _validate_assignment(
        child_params=child_tensors,
        child_mass=child_mass,
        child_to_parent=child_to_parent,
        parent_count=parent_count,
    )
    if int(runtime_outputs[0].shape[0]) != int(parent_count.numel()):
        raise ValueError("runtime parent row count does not match parent_count")
    if int(runtime_outputs[5].shape[1]) != int(child_tensors[5].shape[1]):
        raise ValueError("runtime and child sh_rest basis counts do not match")
    reference_config = _ReferenceProjectorConfig.from_config(projector_cfg, max_scale=max_scale)
    policy = (
        drift_policy
        if isinstance(drift_policy, ParentVJPDriftPolicy)
        else ParentVJPDriftPolicy.from_config(drift_policy)
    )
    alpha_t = _alpha_tensor(alpha, ref=child_tensors[0])
    exact_outputs = _reference_outputs_fp32(
        tuple(value.detach() for value in child_tensors),
        child_mass=child_mass,
        child_to_parent=child_to_parent,
        parent_count=parent_count,
        config=reference_config,
    )
    return _drift_from_outputs(
        runtime_outputs,
        exact_outputs,
        nominal_alpha=alpha_t,
        policy=policy,
    )


__all__ = [
    "ParentVJPDriftPolicy",
    "RuntimeParentProjectionDrift",
    "RuntimeParentProjectionVJPFn",
    "RuntimeParentVJPDriftCollector",
    "parent_projection_feedback",
    "runtime_exact_drift",
]
