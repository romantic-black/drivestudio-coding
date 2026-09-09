from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from models.iforward.runtime.event import UpdateEvent
from models.iforward.versions import is_stage3_4_iforward_version


CONTRACT_SCHEMA = "iforward_validation_contract_v1"
STAGE34_LEGACY_METRIC_PATTERNS = (
    "iforward/biggs/exact_refresh",
    "iforward/biggs/incremental_update",
    "iforward/biggs/drift_",
    "iforward/biggs/time_parent_runtime",
    "iforward/biggs/parent_runtime_update",
    "runtime_parent_projection_vjp",
    "iforward/feedback/parent_vjp",
)


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    value = getattr(node, key, default)
    return default if value is None else value


def _stage34_identity(cfg: Any) -> tuple[bool, str, str, str]:
    iforward_cfg = _cfg_get(_cfg_get(cfg, "model", {}) or {}, "iforward", {}) or {}
    version = str(_cfg_get(iforward_cfg, "version", "") or "")
    variant = str(_cfg_get(iforward_cfg, "training_variant", "") or "")
    parent_spatial = _cfg_get(iforward_cfg, "parent_spatial", {}) or {}
    codec = _cfg_get(parent_spatial, "param_codec", {}) or {}
    schema = str(_cfg_get(codec, "schema", _cfg_get(codec, "schema_version", "")) or "")
    return bool(is_stage3_4_iforward_version(version)), version, variant, schema


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_nonfinite_paths(item, path))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]"
            out.extend(_nonfinite_paths(item, path))
        return out
    if isinstance(value, (int, bool)) or value is None:
        return []
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix or "<root>"]
    return []


def _event_metric_values(event: Any, base_key: str) -> list[float]:
    metrics = dict(getattr(event, "metrics", {}) or {})
    values: list[float] = []
    for key, value in metrics.items():
        name = str(key)
        if name == base_key or name in {
            f"{base_key}_last",
            f"{base_key}_mean",
            f"{base_key}_max",
        }:
            if isinstance(value, (int, float, bool)):
                values.append(float(value))
    return values


def _metric_keys(traces: Sequence[Any]) -> set[str]:
    keys: set[str] = set()
    for trace in traces:
        for event in list(getattr(trace, "events", []) or []):
            keys.update(str(key) for key in dict(getattr(event, "metrics", {}) or {}))
            keys.update(str(key) for key in dict(getattr(event, "state_health", {}) or {}))
        keys.update(str(key) for key in dict(getattr(trace, "summary", {}) or {}))
    return keys


def _check(passed: bool, *, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": _plain(actual), "expected": _plain(expected)}


def build_validation_contract(
    *,
    output_dir: str | Path,
    cfg: Any,
    plans: Sequence[Any],
    traces: Sequence[Any],
    plan_dirs: Sequence[str | Path],
    parameter_versions_before: Mapping[str, int],
    parameter_versions_after: Mapping[str, int],
    checkpoint_payload: Mapping[str, Any] | None = None,
    runtime_error: str = "",
) -> dict[str, Any]:
    output = Path(output_dir)
    stage34, version, variant, codec_schema = _stage34_identity(cfg)
    checkpoint = dict(checkpoint_payload or {})
    completed = len(traces)
    expected = len(plans)

    missing_artifacts: list[str] = []
    for idx, plan in enumerate(plans):
        if idx >= len(plan_dirs):
            missing_artifacts.append(f"plan[{idx}]:directory")
            continue
        plan_dir = Path(plan_dirs[idx])
        required = {"plan.json", "trace.jsonl", "summary.json", "html_summary.json", "index.html"}
        required.update(str(name) for name in tuple(getattr(plan, "expected_outputs", ()) or ()))
        for name in sorted(required):
            path = plan_dir / name
            if not path.is_file() or int(path.stat().st_size) <= 0:
                missing_artifacts.append(str(path))
    top_index = output / "index.html"
    if not top_index.is_file() or int(top_index.stat().st_size) <= 0:
        missing_artifacts.append(str(top_index))

    nonfinite: list[str] = []
    for trace_idx, trace in enumerate(traces):
        nonfinite.extend(
            f"trace[{trace_idx}].{path}"
            for path in _nonfinite_paths(_plain(trace))
        )

    changed_parameters = {
        name: [parameter_versions_before.get(name), parameter_versions_after.get(name)]
        for name in sorted(set(parameter_versions_before) | set(parameter_versions_after))
        if parameter_versions_before.get(name) != parameter_versions_after.get(name)
    }

    protocol_names_completed = [
        str(getattr(getattr(plan, "episode", None), "protocol_name", ""))
        for plan in plans[:completed]
    ]
    protocol_families_completed = {
        name.split("/", 1)[0] for name in protocol_names_completed if name
    }
    val_cfg = _cfg_get(cfg, "iforward_validation_v4", {}) or {}
    protocol_cfg = _cfg_get(val_cfg, "protocols", {}) or {}
    configured_protocol_families = {
        str(name)
        for name in (
            "assimilation_timeline",
            "repair_before_after",
            "order_robustness",
            "repeat_stability",
            "memory_ablation",
        )
        if bool(_cfg_get(protocol_cfg, str(name), False))
    }
    expected_memory_modes = {
        str(getattr(event, "memory_mode", ""))
        for plan in plans
        for event in tuple(getattr(plan, "events", ()) or ())
        if str(getattr(event, "memory_mode", ""))
    }
    completed_memory_modes = {
        str(getattr(event, "memory_mode", ""))
        for trace in traces
        for event in list(getattr(trace, "events", []) or [])
        if str(getattr(event, "memory_mode", ""))
    }

    k2_updates: list[dict[str, Any]] = []
    for plan_idx, (plan, trace) in enumerate(zip(plans, traces)):
        trace_by_id = {
            str(getattr(event, "event_id", "")): event
            for event in list(getattr(trace, "events", []) or [])
        }
        for event in tuple(getattr(plan, "events", ()) or ()):
            if not isinstance(event, UpdateEvent):
                continue
            rollout_plan = getattr(event, "rollout_plan", None)
            steps = list(getattr(rollout_plan, "steps", []) or [])
            inner_k = int(getattr(rollout_plan, "actual_inner_K", len(steps)) or len(steps))
            if inner_k < 2:
                continue
            trace_event = trace_by_id.get(str(getattr(event, "event_id", "")))
            update_values = (
                _event_metric_values(trace_event, "iforward/stage3_4/model_update_count")
                if trace_event is not None
                else []
            )
            k2_updates.append(
                {
                    "plan_idx": int(plan_idx),
                    "event_id": str(getattr(event, "event_id", "")),
                    "inner_k": int(inner_k),
                    "max_model_update_count": max(update_values) if update_values else None,
                }
            )

    stage34_runtime_events = [
        event
        for trace in traces
        for event in list(getattr(trace, "events", []) or [])
        if str(getattr(event, "event_kind", "")) not in {
            "reset_state",
            "snapshot_state",
            "restore_state",
            "set_memory_mode",
        }
    ]
    missing_grad_metric: list[str] = []
    nonzero_grad: list[str] = []
    missing_forward_metric: list[str] = []
    nonforward_only: list[str] = []
    for event in stage34_runtime_events:
        event_id = str(getattr(event, "event_id", ""))
        grad_values = _event_metric_values(event, "feedback/functional_parent/grad_active")
        if not grad_values:
            missing_grad_metric.append(event_id)
        elif any(abs(value) > 1.0e-8 for value in grad_values):
            nonzero_grad.append(event_id)
        forward_values = _event_metric_values(event, "feedback/functional_parent/forward_only")
        if not forward_values:
            missing_forward_metric.append(event_id)
        elif any(abs(value - 1.0) > 1.0e-8 for value in forward_values):
            nonforward_only.append(event_id)

    metric_keys = _metric_keys(traces)
    legacy_keys = sorted(
        key
        for key in metric_keys
        if any(pattern in key for pattern in STAGE34_LEGACY_METRIC_PATTERNS)
    )

    checks: dict[str, dict[str, Any]] = {
        "runtime_completed": _check(not runtime_error, actual=runtime_error, expected=""),
        "all_plans_completed": _check(completed == expected and expected > 0, actual=completed, expected=expected),
        "all_plan_artifacts_present": _check(not missing_artifacts, actual=missing_artifacts, expected=[]),
        "all_trace_numbers_finite": _check(not nonfinite, actual=nonfinite, expected=[]),
        "model_parameters_unchanged": _check(not changed_parameters, actual=changed_parameters, expected={}),
        "configured_protocols_completed": _check(
            configured_protocol_families.issubset(protocol_families_completed),
            actual=sorted(protocol_families_completed),
            expected=sorted(configured_protocol_families),
        ),
        "requested_memory_modes_completed": _check(
            expected_memory_modes.issubset(completed_memory_modes),
            actual=sorted(completed_memory_modes),
            expected=sorted(expected_memory_modes),
        ),
    }

    if stage34:
        saved_version = str(checkpoint.get("iforward_version", "") or "")
        saved_variant = str(checkpoint.get("training_variant", "") or "")
        saved_schema = str(checkpoint.get("parent_codec_schema", "") or "")
        local_update_values = [
            value
            for event in stage34_runtime_events
            for value in _event_metric_values(event, "iforward/stage3_4/model_update_count")
        ]
        gdkv_update_values = [
            value
            for event in stage34_runtime_events
            for key in (
                "iforward/parent_optimizer_gdkv/global_update_step",
                "iforward/parent_optimizer_gdkv/write",
            )
            for value in _event_metric_values(event, key)
        ]
        checks.update(
            {
                "native_stage34_checkpoint": _check(
                    bool(checkpoint)
                    and saved_version == version
                    and saved_variant == variant
                    and bool(codec_schema)
                    and saved_schema == codec_schema,
                    actual={
                        "iforward_version": saved_version,
                        "training_variant": saved_variant,
                        "parent_codec_schema": saved_schema,
                    },
                    expected={
                        "iforward_version": version,
                        "training_variant": variant,
                        "parent_codec_schema": codec_schema,
                    },
                ),
                "k2_update_ancestor_observed": _check(
                    any(
                        item["max_model_update_count"] is not None
                        and float(item["max_model_update_count"]) > 0.0
                        for item in k2_updates
                    ),
                    actual=k2_updates,
                    expected="at least one K>=2 update event with model_update_count>0",
                ),
                "causal_localgs_and_gdkv_advanced": _check(
                    any(value > 0.0 for value in local_update_values)
                    and any(value > 0.0 for value in gdkv_update_values),
                    actual={
                        "max_model_update_count": max(local_update_values) if local_update_values else None,
                        "max_gdkv_update": max(gdkv_update_values) if gdkv_update_values else None,
                    },
                    expected={
                        "max_model_update_count": "> 0",
                        "max_gdkv_update": "> 0",
                    },
                ),
                "functional_parent_grad_inactive": _check(
                    not missing_grad_metric and not nonzero_grad and bool(stage34_runtime_events),
                    actual={"missing": missing_grad_metric, "nonzero": nonzero_grad},
                    expected={"missing": [], "nonzero": []},
                ),
                "functional_parent_forward_only": _check(
                    not missing_forward_metric and not nonforward_only and bool(stage34_runtime_events),
                    actual={"missing": missing_forward_metric, "not_forward_only": nonforward_only},
                    expected={"missing": [], "not_forward_only": []},
                ),
                "legacy_parent_runtime_metrics_absent": _check(
                    not legacy_keys,
                    actual=legacy_keys,
                    expected=[],
                ),
            }
        )

    failures = [name for name, result in checks.items() if not bool(result.get("passed", False))]
    protocol_names = sorted(set(protocol_names_completed))
    memory_modes = sorted(completed_memory_modes)
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": "passed" if not failures else "failed",
        "stage3_4_required": bool(stage34),
        "identity": {
            "iforward_version": version,
            "training_variant": variant,
            "parent_codec_schema": codec_schema,
        },
        "plan_count_expected": int(expected),
        "plan_count_completed": int(completed),
        "event_count": int(sum(len(list(getattr(trace, "events", []) or [])) for trace in traces)),
        "protocols_completed": protocol_names,
        "memory_modes_completed": memory_modes,
        "checks": checks,
        "failures": failures,
    }


def write_validation_contract(contract: Mapping[str, Any], output_dir: str | Path) -> str:
    path = Path(output_dir) / "validation_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_plain(contract), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    return str(path)


def assert_validation_contract(contract: Mapping[str, Any]) -> None:
    failures = [str(name) for name in list(contract.get("failures", []) or [])]
    if failures:
        raise RuntimeError(
            "IForward validation contract failed: " + ", ".join(failures)
        )


__all__ = [
    "CONTRACT_SCHEMA",
    "STAGE34_LEGACY_METRIC_PATTERNS",
    "assert_validation_contract",
    "build_validation_contract",
    "write_validation_contract",
]
