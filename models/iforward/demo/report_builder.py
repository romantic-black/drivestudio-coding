from __future__ import annotations

import dataclasses
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from datasets.iforward_stage2_3.scheduler import Stage23Scheduler

from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.plan import EpisodePlan
from models.iforward.runtime.runner import IForwardRunner, RunnerOptions
from models.iforward.runtime.trace import EpisodeTrace, TraceRecorder
from models.iforward.validation_v4.html_exporter import export_html_report
from models.iforward.validation_v4.metrics import summarize_event_traces


@dataclass
class DemoRunResult:
    recipe: str
    output_dir: str
    index_html: str
    traces: list[EpisodeTrace]
    summary: dict[str, Any]


def build_demo_report(
    *,
    recipe: str,
    plans: Sequence[EpisodePlan],
    model: Any,
    scheduler: Stage23Scheduler,
    output_dir: str | Path,
    device: torch.device | str,
    trigger_step: int = 0,
    convert_batch_to_minimal_format: Callable[[dict[str, Any], torch.device, int], dict[str, Any]] | None = None,
) -> DemoRunResult:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plans_dir = root / "plans"
    runs_dir = root / "recipe_runs"
    plans_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    traces: list[EpisodeTrace] = []
    all_events = []
    run_links: list[dict[str, str]] = []
    adapter = Stage3SchedulerAdapter(scheduler)
    runner = IForwardRunner(model, adapter, convert_batch_to_minimal_format)
    for idx, plan in enumerate(list(plans)):
        safe_recipe = str(recipe).replace("/", "_")
        run_name = f"{safe_recipe}_{idx:02d}_{plan.plan_id}"
        run_dir = runs_dir / run_name
        recorder = TraceRecorder(run_dir, record_images=True)
        trace = runner.run(
            plan,
            recorder,
            RunnerOptions.for_mode("demo", device=str(device), trigger_step=int(trigger_step)),
        )
        traces.append(trace)
        all_events.extend(trace.events)
        plan_path = plans_dir / f"{safe_recipe}_{idx:02d}_{plan.plan_id}.json"
        _write_json(plan_path, plan.to_json_dict())
        html_path = export_html_report(trace, run_dir, title=f"IForward Demo {plan.episode.protocol_name}")
        run_links.append(
            {
                "plan_id": str(plan.plan_id),
                "protocol": str(plan.episode.protocol_name),
                "run_dir": str(run_dir.relative_to(root)),
                "html": str(Path(html_path).relative_to(root)),
                "plan": str(plan_path.relative_to(root)),
            }
        )

    summary = summarize_event_traces(all_events)
    summary["recipe"] = str(recipe)
    summary["num_plans"] = int(len(plans))
    summary["questions"] = _demo_question_answers(summary, all_events)
    _write_json(root / "summary.json", summary)
    _write_json(root / "plan.json", {"recipe": str(recipe), "plans": run_links})
    _write_trace_jsonl(root / "trace.jsonl", all_events)
    index_path = root / "index.html"
    index_path.write_text(_render_demo_index(recipe=str(recipe), summary=summary, run_links=run_links), encoding="utf-8")
    return DemoRunResult(
        recipe=str(recipe),
        output_dir=str(root),
        index_html=str(index_path),
        traces=traces,
        summary=summary,
    )


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_trace_jsonl(path: Path, events: Sequence[Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            row = dataclasses.asdict(event) if dataclasses.is_dataclass(event) else dict(event)
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _demo_question_answers(summary: dict[str, Any], events: Sequence[Any]) -> dict[str, Any]:
    protocols = list(summary.get("protocols", []) or [])
    current = _mean_field(protocols, "current_psnr_mean")
    history = _mean_field(protocols, "history_retention_auc")
    repair = _mean_field(protocols, "repair_mean")
    memory = dict(summary.get("memory_ablation", {}) or {})
    health = dict(summary.get("state_health", {}) or {})
    before = _event_metric(events, "before_repair", "current_psnr")
    after = _event_metric(events, "after_repair", "current_psnr")
    return {
        "current_frame_improved": "observed" if current > 0.0 else "not measured",
        "history_retention": history,
        "repair_before_after_gain": (after - before) if before is not None and after is not None else None,
        "memory_gain_retention": memory.get("memory_gain_retention"),
        "repeat_stability": "not run",
        "state_health": health or "not measured",
    }


def _event_metric(events: Sequence[Any], marker: str, metric: str) -> float | None:
    vals = []
    for event in events:
        meta = dict(getattr(event, "metadata", {}) or {})
        if str(meta.get("demo_stage", "")) != str(marker):
            continue
        metrics = dict(getattr(event, "metrics", {}) or {})
        if metric in metrics:
            vals.append(float(metrics[metric]))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _mean_field(rows: Sequence[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, 0.0)))
        except (TypeError, ValueError):
            pass
    return float(sum(vals) / max(1, len(vals))) if vals else 0.0


def _render_demo_index(*, recipe: str, summary: dict[str, Any], run_links: list[dict[str, str]]) -> str:
    questions = dict(summary.get("questions", {}) or {})
    protocols = list(summary.get("protocols", []) or [])
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>IForward Demo {html.escape(recipe)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f4;color:#202124}",
        "section{margin:0 0 24px} table{border-collapse:collapse;width:100%;background:#fff}",
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:13px} th{background:#ecebe6}",
        ".answer{display:inline-block;margin:4px 8px 4px 0;padding:4px 8px;border-radius:4px;background:#e8f5e9}",
        "</style></head><body>",
        f"<h1>IForward Demo {html.escape(recipe)}</h1>",
        "<section><h2>Questions</h2>",
        "".join(f"<span class='answer'>{html.escape(str(k))}: {html.escape(str(v))}</span>" for k, v in questions.items()),
        "</section>",
        "<section><h2>Runs</h2>",
        _table(run_links),
        "</section>",
        "<section><h2>Metrics</h2>",
        _table(protocols),
        "</section>",
        "<section><h2>Raw Links</h2><p><a href='summary.json'>summary.json</a> <a href='trace.jsonl'>trace.jsonl</a> <a href='plan.json'>plan.json</a></p></section>",
        "</body></html>",
    ]
    return "\n".join(parts)


def _table(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
        if len(keys) >= 10:
            break
    keys = keys[:10]
    out = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(k))}</th>" for k in keys) + "</tr></thead><tbody>"]
    for row in rows:
        out.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys)
            + "</tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


__all__ = ["DemoRunResult", "build_demo_report"]
