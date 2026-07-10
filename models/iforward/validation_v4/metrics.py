from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: Iterable[Any]) -> float:
    vals = [_finite_float(x, float("nan")) for x in values]
    vals = [x for x in vals if math.isfinite(x)]
    return float(sum(vals) / max(1, len(vals))) if vals else 0.0


def _min(values: Iterable[Any]) -> float:
    vals = [_finite_float(x, float("nan")) for x in values]
    vals = [x for x in vals if math.isfinite(x)]
    return float(min(vals)) if vals else 0.0


def _std(values: Iterable[Any]) -> float:
    vals = [_finite_float(x, float("nan")) for x in values]
    vals = [x for x in vals if math.isfinite(x)]
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def summarize_event_traces(events: Iterable[Any]) -> dict[str, Any]:
    events_l = list(events)
    rows = []
    for event in events_l:
        metadata = dict(getattr(event, "metadata", {}) or {})
        stage32 = dict(metadata.get("iforward_stage3_2", {}) or {})
        row = {
            "protocol": str(getattr(event, "protocol", "")),
            "mode": str(getattr(event, "memory_mode", "")),
            "validation_rollout_kind": str(getattr(event, "event_kind", "")),
            "scheduler_phase": str(getattr(event, "scheduler_phase", "")),
            "distribution_type": str(stage32.get("distribution_type", metadata.get("distribution_type", ""))),
            "episode_stage": str(stage32.get("episode_stage", metadata.get("episode_stage", ""))),
            "order_type": str(stage32.get("order_type", metadata.get("order_type", ""))),
            "train_2d_mode": str(stage32.get("train_2d_mode", metadata.get("train_2d_mode", ""))),
            "current_psnr": _finite_float(getattr(event, "metrics", {}).get("current_psnr", 0.0)),
            "history_rollout_psnr": _finite_float(getattr(event, "metrics", {}).get("history_rollout_psnr", 0.0)),
            "current_loss": _finite_float(getattr(event, "metrics", {}).get("current", getattr(event, "metrics", {}).get("current_loss", 0.0))),
        }
        rows.append(row)
    summary = summarize_legacy_rows(rows)
    summary["uncertainty_calibration"] = _uncertainty_calibration_summary(events_l)
    summary["uncertainty_state"] = _uncertainty_state_summary(events_l)
    return summary


def _uncertainty_calibration_summary(events: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for event in events:
        metadata = dict(getattr(event, "metadata", {}) or {})
        stage32 = dict(metadata.get("iforward_stage3_2", {}) or {})
        distribution = str(stage32.get("distribution_type", metadata.get("distribution_type", "")))
        phase = str(getattr(event, "scheduler_phase", ""))
        metrics = dict(getattr(event, "metrics", {}) or {})
        for role in ("current", "in_rollout_history"):
            prefix = f"{role}/"
            values = {
                key[len(prefix) :]: _finite_float(value)
                for key, value in metrics.items()
                if str(key).startswith(prefix)
                and any(
                    token in str(key)
                    for token in (
                        "error_uncertainty_pearson",
                        "error_uncertainty_spearman",
                        "ause",
                        "risk_coverage_",
                    )
                )
            }
            if values:
                report_role = "repair" if phase == "repair" else role
                grouped.setdefault((distribution, phase, report_role), []).append(values)
    out = []
    for (distribution, phase, role), rows in sorted(grouped.items()):
        keys = sorted({key for row in rows for key in row})
        out.append(
            {
                "distribution_type": distribution,
                "scheduler_phase": phase,
                "role": role,
                "num_rows": int(len(rows)),
                **{key: _mean(row.get(key) for row in rows if key in row) for key in keys},
            }
        )
    return out


def _uncertainty_state_summary(events: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    state_keys = (
        "sigma_mean",
        "sigma_p10",
        "sigma_p50",
        "sigma_p90",
        "logvar_min",
        "logvar_max",
        "clamp_min_ratio",
        "clamp_max_ratio",
    )
    for event in events:
        metadata = dict(getattr(event, "metadata", {}) or {})
        stage32 = dict(metadata.get("iforward_stage3_2", {}) or {})
        distribution = str(stage32.get("distribution_type", metadata.get("distribution_type", "")))
        phase = str(getattr(event, "scheduler_phase", ""))
        metrics = dict(getattr(event, "metrics", {}) or {})
        for branch in ("bg", "distant", "rigid"):
            prefix = f"uncertainty/{branch}/"
            values = {
                key: _finite_float(metrics[f"{prefix}{key}"])
                for key in state_keys
                if f"{prefix}{key}" in metrics
            }
            if values:
                grouped.setdefault((distribution, phase, branch), []).append(values)
    out = []
    for (distribution, phase, branch), rows in sorted(grouped.items()):
        keys = sorted({key for row in rows for key in row})
        out.append(
            {
                "distribution_type": distribution,
                "scheduler_phase": phase,
                "branch": branch,
                "num_rows": int(len(rows)),
                **{key: _mean(row.get(key) for row in rows if key in row) for key in keys},
            }
        )
    return out


def summarize_legacy_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_l = [dict(row) for row in rows]
    protocol_summaries: list[dict[str, Any]] = []
    for key in sorted({(str(row.get("protocol", "")), str(row.get("mode", ""))) for row in rows_l}):
        protocol, mode = key
        group = [row for row in rows_l if str(row.get("protocol", "")) == protocol and str(row.get("mode", "")) == mode]
        if not group:
            continue
        current_psnrs = [_finite_float(row.get("current_psnr", 0.0)) for row in group]
        history_psnrs = [_finite_float(row.get("history_rollout_psnr", row.get("retention_psnr", 0.0))) for row in group]
        repair_rows = [
            row
            for row in group
            if str(row.get("scheduler_phase", "")) == "repair"
            or str(row.get("validation_rollout_kind", "")).startswith("repair")
        ]
        repeat_rows = [row for row in group if "repeat" in str(row.get("validation_rollout_kind", ""))]
        protocol_summaries.append(
            {
                "protocol": protocol,
                "mode": mode,
                "num_rows": int(len(group)),
                "current_psnr_mean": _mean(current_psnrs),
                "current_psnr_worst": _min(current_psnrs),
                "history_retention_auc": _mean(history_psnrs),
                "repair_mean": _mean(row.get("current_psnr", 0.0) for row in repair_rows),
                "repair_worst": _min(row.get("current_psnr", 0.0) for row in repair_rows),
                "order_permutation_std": _std(row.get("current_psnr", 0.0) for row in repair_rows),
                "repeat_stability_std": _std(row.get("current_psnr", 0.0) for row in repeat_rows),
            }
        )
    memory = _memory_summary(protocol_summaries)
    state_health = _state_health_summary(rows_l)
    return {
        "num_rows": int(len(rows_l)),
        "protocols": protocol_summaries,
        "memory_ablation": memory,
        "state_health": state_health,
    }


def _memory_summary(protocols: list[dict[str, Any]]) -> dict[str, float]:
    memory_rows = [row for row in protocols if "memory_ablation" in str(row.get("protocol", ""))]
    by_mode = {str(row.get("mode", "")): row for row in memory_rows}
    full = by_mode.get("full")
    off = by_mode.get("memory_off") or by_mode.get("mamba_off")
    shuffle = by_mode.get("memory_shuffle_state") or by_mode.get("mamba_shuffle_state")
    shuffle_rw = by_mode.get("memory_shuffle_read_write_state") or by_mode.get("mamba_shuffle_read_write_state")
    wrong_key = by_mode.get("memory_wrong_parent_key_fixed") or by_mode.get("mamba_wrong_parent_key_fixed")
    freeze_after_prefill = by_mode.get("memory_freeze_after_prefill") or by_mode.get("mamba_freeze_after_prefill")
    out: dict[str, float] = {}
    if full and off:
        out["memory_gain_retention"] = _finite_float(full.get("history_retention_auc")) - _finite_float(
            off.get("history_retention_auc")
        )
        out["memory_gain_current"] = _finite_float(full.get("current_psnr_mean")) - _finite_float(
            off.get("current_psnr_mean")
        )
    if full and shuffle:
        out["memory_shuffle_gap"] = _finite_float(full.get("history_retention_auc")) - _finite_float(
            shuffle.get("history_retention_auc")
        )
    if full and shuffle_rw:
        out["memory_shuffle_read_write_gap"] = _finite_float(full.get("history_retention_auc")) - _finite_float(
            shuffle_rw.get("history_retention_auc")
        )
    if full and wrong_key:
        out["memory_wrong_parent_key_gap"] = _finite_float(full.get("history_retention_auc")) - _finite_float(
            wrong_key.get("history_retention_auc")
        )
    if full and freeze_after_prefill:
        out["memory_freeze_after_prefill_gap"] = _finite_float(full.get("history_retention_auc")) - _finite_float(
            freeze_after_prefill.get("history_retention_auc")
        )
    return out


def _state_health_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "scale_abnormal_ratio",
        "opacity_abnormal_ratio",
        "nan_count",
        "inf_count",
        "gdkv_state_rms_max",
        "gdkv_ctx_rms_max",
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = [row.get(key) for row in rows if key in row]
        if vals:
            out[f"{key}/mean"] = _mean(vals)
            out[f"{key}/max"] = max(_finite_float(x) for x in vals)
    return out


__all__ = ["summarize_event_traces", "summarize_legacy_rows"]
