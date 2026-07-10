from __future__ import annotations

import dataclasses
import html
import json
from pathlib import Path
from typing import Any, Iterable

from .metrics import summarize_event_traces, summarize_legacy_rows


def export_html_report(trace: Any, output_dir: str | Path, *, title: str = "IForward Validation v4") -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for event in list(getattr(trace, "events", []) or []):
        row = dataclasses.asdict(event) if dataclasses.is_dataclass(event) else dict(event)
        metrics = dict(row.get("metrics", {}) or {})
        row.update(
            {
                "current_psnr": metrics.get("current_psnr", 0.0),
                "history_rollout_psnr": metrics.get("history_rollout_psnr", 0.0),
                "mode": row.get("memory_mode", ""),
                "validation_rollout_kind": row.get("event_kind", ""),
            }
        )
        metadata = dict(row.get("metadata", {}) or {})
        stage32 = dict(metadata.get("iforward_stage3_2", {}) or {})
        if stage32:
            row.update(
                {
                    "distribution_type": str(stage32.get("distribution_type", "")),
                    "episode_stage": str(stage32.get("episode_stage", "")),
                    "order_type": str(stage32.get("order_type", "")),
                    "train_2d_mode": str(stage32.get("train_2d_mode", "")),
                    "stage3_2_K": stage32.get("K", 0),
                    "repair_visited_ratio": stage32.get("repair_visited_ratio", 0.0),
                }
            )
        rows.append(row)
    summary = summarize_event_traces(list(getattr(trace, "events", []) or []))
    _write_json(output / "html_summary.json", summary)
    html_text = _render_html(
        title=title,
        summary=summary,
        rows=rows,
        raw_links=_raw_links(output, ["plan.json", "trace.jsonl", "summary.json", "html_summary.json", "parent_diagnostics_summary.csv"]),
    )
    path = output / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return str(path)


def export_legacy_rows_html_report(rows: Iterable[dict[str, Any]], output_dir: str | Path, *, title: str = "IForward Validation v4") -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_l = [dict(row) for row in rows]
    summary = summarize_legacy_rows(rows_l)
    _write_json(output / "summary.json", summary)
    html_text = _render_html(title=title, summary=summary, rows=rows_l, raw_links=["summary.json"])
    path = output / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return str(path)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _render_html(*, title: str, summary: dict[str, Any], rows: list[dict[str, Any]], raw_links: list[str]) -> str:
    protocols = list(summary.get("protocols", []) or [])
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f4;color:#202124}",
        "h1,h2{margin:0 0 12px} section{margin:0 0 24px}",
        "table{border-collapse:collapse;width:100%;background:white} th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:13px}",
        "th{background:#ecebe6}.status{display:inline-block;padding:3px 8px;border-radius:4px;background:#e8f5e9}.warn{background:#fff8e1}.fail{background:#ffebee}",
        "img{max-width:100%;height:auto}.links a{margin-right:12px}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<section><h2>Run Summary</h2>",
        f"<p>Rows: {int(summary.get('num_rows', len(rows)))}</p>",
        "<p class='links'>" + " ".join(f"<a href='{html.escape(link)}'>{html.escape(link)}</a>" for link in raw_links) + "</p>",
        "</section>",
        "<section><h2>Traffic-light Status</h2>",
        _status_block(protocols, summary),
        "</section>",
        "<section><h2>Metrics Table</h2>",
        _table(protocols),
        "</section>",
        "<section><h2>Uncertainty Calibration</h2>",
        _table(list(summary.get("uncertainty_calibration", []) or [])),
        "</section>",
        "<section><h2>Uncertainty State by Branch</h2>",
        _table(list(summary.get("uncertainty_state", []) or [])),
        "</section>",
        "<section><h2>Raw Trace</h2>",
        _table(rows[:200]),
        "</section>",
        "<section><h2>Parent Diagnostics</h2>",
        _parent_artifact_table(rows),
        "</section>",
        "<section><h2>Uncertainty Diagnostics</h2>",
        _uncertainty_artifact_table(rows),
        "</section>",
        "</body></html>",
    ]
    return "\n".join(body)


def _raw_links(output: Path, candidates: list[str]) -> list[str]:
    return [name for name in candidates if (output / name).exists()]


def _status_block(protocols: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    current = _mean_field(protocols, "current_psnr_mean")
    history = _mean_field(protocols, "history_retention_auc")
    repair = _mean_field(protocols, "repair_mean")
    memory = (summary.get("memory_ablation", {}) or {}).get("memory_gain_retention", 0.0)
    parts = [
        ("current update", current, 18.0),
        ("history retention", history, 16.0),
        ("repair", repair, 16.0),
        ("memory ablation", memory, 0.0),
    ]
    spans = []
    for label, value, threshold in parts:
        cls = "status" if float(value) >= float(threshold) else "status warn"
        spans.append(f"<span class='{cls}'>{html.escape(label)}: {float(value):.3f}</span>")
    return "<p>" + " ".join(spans) + "</p>"


def _mean_field(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, 0.0)))
        except (TypeError, ValueError):
            pass
    return float(sum(vals) / max(1, len(vals))) if vals else 0.0


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and key not in {"metadata"}:
                keys.append(key)
        if len(keys) >= 12:
            break
    keys = keys[:12]
    lines = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(k))}</th>" for k in keys) + "</tr></thead><tbody>"]
    for row in rows:
        vals = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)[:180]
            vals.append(f"<td>{html.escape(str(value))}</td>")
        lines.append("<tr>" + "".join(vals) + "</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _parent_artifact_table(rows: list[dict[str, Any]]) -> str:
    compact = []
    for row in rows:
        artifacts = dict(row.get("artifacts", {}) or {})
        parent_csv = artifacts.get("parent_topk_csv", "")
        if not parent_csv:
            continue
        metadata = dict(row.get("metadata", {}) or {})
        parent_meta = dict(metadata.get("parent_diagnostics", {}) or {})
        compact.append(
            {
                "event_id": row.get("event_id", ""),
                "protocol": row.get("protocol", ""),
                "memory_mode": row.get("memory_mode", ""),
                "num_rows": parent_meta.get("num_rows", 0),
                "max_impact_score": parent_meta.get("max_impact_score", 0.0),
                "max_delta_norm_rms": parent_meta.get("max_delta_norm_rms", 0.0),
                "parent_topk_csv": parent_csv,
                "parent_summary_json": artifacts.get("parent_summary_json", ""),
            }
        )
    if not compact:
        return "<p>No parent diagnostics were recorded.</p>"
    lines = [
        "<table><thead><tr>"
        "<th>event_id</th><th>protocol</th><th>memory_mode</th><th>rows</th>"
        "<th>max impact</th><th>max delta rms</th><th>top-K</th><th>summary</th>"
        "</tr></thead><tbody>"
    ]
    for row in compact[:200]:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['event_id']))}</td>"
            f"<td>{html.escape(str(row['protocol']))}</td>"
            f"<td>{html.escape(str(row['memory_mode']))}</td>"
            f"<td>{html.escape(str(row['num_rows']))}</td>"
            f"<td>{float(row['max_impact_score']):.6f}</td>"
            f"<td>{float(row['max_delta_norm_rms']):.6f}</td>"
            f"<td><a href='{html.escape(str(row['parent_topk_csv']))}'>csv</a></td>"
            f"<td><a href='{html.escape(str(row['parent_summary_json']))}'>json</a></td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _uncertainty_artifact_table(rows: list[dict[str, Any]]) -> str:
    compact = []
    for row in rows:
        artifacts = dict(row.get("artifacts", {}) or {})
        for idx in range(2):
            grid = artifacts.get(f"uncertainty_grid_{idx}", "")
            if not grid:
                continue
            compact.append(
                {
                    "event_id": row.get("event_id", ""),
                    "protocol": row.get("protocol", ""),
                    "view": idx,
                    "grid": grid,
                    "before_after": artifacts.get(f"before_after_grid_{idx}", ""),
                    "confidence_bins": artifacts.get(f"confidence_bins_{idx}", ""),
                }
            )
    if not compact:
        return "<p>No uncertainty diagnostics were recorded.</p>"
    lines = [
        "<table><thead><tr><th>event</th><th>protocol</th><th>view</th>"
        "<th>GT / RGB / error / sigma / aleatoric / disagreement / alpha</th>"
        "<th>before/after deltas</th><th>confidence bins</th></tr></thead><tbody>"
    ]
    for row in compact[:200]:
        before_after = (
            f"<a href='{html.escape(str(row['before_after']))}'>image</a>"
            if row["before_after"]
            else ""
        )
        bins = (
            f"<a href='{html.escape(str(row['confidence_bins']))}'>csv</a>"
            if row["confidence_bins"]
            else ""
        )
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['event_id']))}</td>"
            f"<td>{html.escape(str(row['protocol']))}</td>"
            f"<td>{int(row['view'])}</td>"
            f"<td><a href='{html.escape(str(row['grid']))}'><img src='{html.escape(str(row['grid']))}'></a></td>"
            f"<td>{before_after}</td><td>{bins}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


__all__ = ["export_html_report", "export_legacy_rows_html_report"]
