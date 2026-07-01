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
    rows = []
    for event in events:
        row = {
            "protocol": str(getattr(event, "protocol", "")),
            "mode": str(getattr(event, "memory_mode", "")),
            "validation_rollout_kind": str(getattr(event, "event_kind", "")),
            "scheduler_phase": str(getattr(event, "scheduler_phase", "")),
            "current_psnr": _finite_float(getattr(event, "metrics", {}).get("current_psnr", 0.0)),
            "history_rollout_psnr": _finite_float(getattr(event, "metrics", {}).get("history_rollout_psnr", 0.0)),
            "current_loss": _finite_float(getattr(event, "metrics", {}).get("current", getattr(event, "metrics", {}).get("current_loss", 0.0))),
        }
        rows.append(row)
    return summarize_legacy_rows(rows)


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
