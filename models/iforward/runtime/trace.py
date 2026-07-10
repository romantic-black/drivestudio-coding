from __future__ import annotations

import dataclasses
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from models.iforward.diagnostics import build_parent_diagnostics

from .artifact_store import ArtifactStore


@dataclass
class EventTrace:
    plan_id: str
    event_id: str
    event_kind: str
    event_idx: int
    protocol: str
    memory_mode: str
    scheduler_phase: str = ""
    rollout_phase: str = ""
    input_positions: list[int] = field(default_factory=list)
    history_positions: list[int] = field(default_factory=list)
    repair_positions: list[int] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    state_health: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrace:
    plan_id: str
    protocol: str
    scene_id: int
    segment_id: int
    events: list[EventTrace] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)


class TraceRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        record_images: bool = True,
        record_parent_diagnostics: bool = True,
        parent_topk: int = 16,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(self.output_dir)
        self.record_images = bool(record_images)
        self.record_parent_diagnostics = bool(record_parent_diagnostics)
        self.parent_topk = int(parent_topk)
        self.trace_path = self.output_dir / "trace.jsonl"
        self._fh: Optional[Any] = None
        self._trace: Optional[EpisodeTrace] = None
        self._parent_rows: list[dict[str, Any]] = []

    def begin_plan(self, plan: Any) -> EpisodeTrace:
        self.artifacts.save_json("plan.json", plan.to_json_dict())
        self._fh = open(self.trace_path, "w", encoding="utf-8")
        episode = plan.episode
        self._trace = EpisodeTrace(
            plan_id=str(plan.plan_id),
            protocol=str(getattr(episode, "protocol_name", "") or plan.metadata.get("protocol", "")),
            scene_id=int(getattr(episode, "scene_id", -1)),
            segment_id=int(getattr(episode, "segment_id", -1)),
        )
        return self._trace

    def record_update(
        self,
        event: Any,
        out: Any,
        state: Any,
        *,
        event_idx: int,
        memory_mode: str,
        previous_state: Any = None,
    ) -> EventTrace:
        return self._record_event(event, out, state, event_idx=event_idx, memory_mode=memory_mode, previous_state=previous_state)

    def record_probe(
        self,
        event: Any,
        out: Any,
        state: Any,
        *,
        event_idx: int,
        memory_mode: str,
        previous_state: Any = None,
    ) -> EventTrace:
        return self._record_event(event, out, state, event_idx=event_idx, memory_mode=memory_mode, previous_state=previous_state)

    def record_control(self, event: Any, state: Any, *, event_idx: int, memory_mode: str) -> EventTrace:
        row = EventTrace(
            plan_id=self._trace.plan_id if self._trace is not None else "",
            event_id=str(event.event_id),
            event_kind=str(event.kind),
            event_idx=int(event_idx),
            protocol=self._trace.protocol if self._trace is not None else "",
            memory_mode=str(memory_mode),
            metadata=dict(getattr(event, "metadata", {}) or {}),
        )
        self._append(row)
        return row

    def end_plan(self, trace: Optional[EpisodeTrace] = None) -> EpisodeTrace:
        out = trace or self._trace
        if out is None:
            raise RuntimeError("TraceRecorder.end_plan called before begin_plan")
        out.summary = _summarize_events(out.events)
        parent_summary_path = self._write_parent_summary()
        if parent_summary_path:
            out.summary["parent_diagnostics_rows"] = float(len(self._parent_rows))
        self.artifacts.save_json("summary.json", {"plan_id": out.plan_id, "summary": out.summary})
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return out

    def finalize(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("TraceRecorder.finalize called before begin_plan")
        return self.end_plan(self._trace)

    def _record_event(
        self,
        event: Any,
        out: Any,
        state: Any,
        *,
        event_idx: int,
        memory_mode: str,
        previous_state: Any = None,
    ) -> EventTrace:
        resolved = getattr(out, "resolved", None)
        meta = dict(getattr(resolved, "meta", {}) or {}) if resolved is not None else {}
        event_meta = dict(getattr(event, "metadata", {}) or {})
        artifacts = self._artifacts_from_output(event, out) if self.record_images else {}
        parent_meta, parent_artifacts = self._parent_diagnostics_from_state(
            event=event,
            event_idx=event_idx,
            previous_state=previous_state,
            next_state=getattr(out, "next_state", state),
        )
        artifacts.update(parent_artifacts)
        row = EventTrace(
            plan_id=self._trace.plan_id if self._trace is not None else "",
            event_id=str(event.event_id),
            event_kind=str(event.kind),
            event_idx=int(event_idx),
            protocol=self._trace.protocol if self._trace is not None else "",
            memory_mode=str(memory_mode),
            scheduler_phase=str(meta.get("scheduler_phase", event_meta.get("scheduler_phase", ""))),
            rollout_phase=str(meta.get("rollout_phase", event_meta.get("rollout_phase", ""))),
            input_positions=[int(x) for x in list(meta.get("rollout_positions", getattr(event, "input_positions", [])) or [])],
            history_positions=[int(x) for x in list(meta.get("history_positions", []) or [])],
            repair_positions=[int(x) for x in list(meta.get("repair_positions", getattr(event, "repair_positions", [])) or [])],
            metrics=_metrics_from_output(out),
            state_health=_state_health_from_output(out),
            artifacts=artifacts,
            metadata={
                **event_meta,
                "ablation": str(getattr(out, "stats", {}).get("ablation", "")),
                **({"parent_diagnostics": parent_meta} if parent_meta else {}),
            },
        )
        self._append(row)
        return row

    def _append(self, row: EventTrace) -> None:
        if self._trace is not None:
            self._trace.events.append(row)
        if self._fh is not None:
            self._fh.write(json.dumps(dataclasses.asdict(row), sort_keys=True) + "\n")
            self._fh.flush()

    def _artifacts_from_output(self, event: Any, out: Any) -> dict[str, str]:
        pred_rgbs = list(getattr(out, "pred_rgbs", []) or [])
        gt_images = list(getattr(out, "gt_images", []) or [])
        if not pred_rgbs or not gt_images:
            return {}
        artifacts: dict[str, str] = {}
        max_pairs = min(2, len(pred_rgbs), len(gt_images))
        for idx in range(max_pairs):
            pred = pred_rgbs[idx]
            gt = gt_images[idx]
            error = (torch.as_tensor(pred).detach().float().cpu() - torch.as_tensor(gt).detach().float().cpu()).abs()
            name = f"{str(event.event_id)}_{idx:02d}.png"
            artifacts[f"grid_{idx}"] = self.artifacts.save_grid(name, [gt, pred, error.clamp(0.0, 1.0)])
        return artifacts

    def _parent_diagnostics_from_state(
        self,
        *,
        event: Any,
        event_idx: int,
        previous_state: Any,
        next_state: Any,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not self.record_parent_diagnostics:
            return {}, {}
        result = build_parent_diagnostics(
            previous_state=previous_state,
            next_state=next_state,
            topk=int(self.parent_topk),
        )
        if not result.rows:
            return result.summary if result.summary.get("num_rows", 0) else {}, {}
        event_id = str(getattr(event, "event_id", f"event_{int(event_idx):03d}")).replace("/", "_")
        parent_dir = self.output_dir / "parent_diagnostics"
        parent_dir.mkdir(parents=True, exist_ok=True)
        csv_path = parent_dir / f"{event_id}_topk.csv"
        json_path = parent_dir / f"{event_id}_summary.json"
        rows = [
            {
                "plan_id": self._trace.plan_id if self._trace is not None else "",
                "event_id": str(getattr(event, "event_id", "")),
                "event_idx": int(event_idx),
                **dict(row),
            }
            for row in result.rows
        ]
        _write_csv(csv_path, rows)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result.summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
        self._parent_rows.extend(rows)
        return (
            {
                "version": str(result.summary.get("version", "")),
                "num_rows": int(result.summary.get("num_rows", 0) or 0),
                "max_impact_score": float(result.summary.get("max_impact_score", 0.0) or 0.0),
                "max_delta_norm_rms": float(result.summary.get("max_delta_norm_rms", 0.0) or 0.0),
            },
            {
                "parent_topk_csv": self.artifacts.relpath(csv_path),
                "parent_summary_json": self.artifacts.relpath(json_path),
            },
        )

    def _write_parent_summary(self) -> str:
        if not self._parent_rows:
            return ""
        path = self.output_dir / "parent_diagnostics_summary.csv"
        _write_csv(path, self._parent_rows)
        return self.artifacts.relpath(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _scalar(value: Any) -> Optional[float]:
    if torch.is_tensor(value):
        if int(value.numel()) == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    if isinstance(value, (int, float, bool)):
        return float(value)
    return None


def _metrics_from_output(out: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for source in (getattr(out, "stats", {}) or {}, getattr(out, "losses", {}) or {}):
        for key, value in dict(source).items():
            scalar = _scalar(value)
            if scalar is not None:
                metrics[str(key)] = float(scalar)
    loss = _scalar(getattr(out, "loss", None))
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def _state_health_from_output(out: Any) -> dict[str, float]:
    metrics = _metrics_from_output(out)
    health_keys = (
        "nan",
        "inf",
        "rms",
        "delta_norm",
        "abnormal",
        "opacity",
        "scale",
        "parent_optimizer_gdkv",
        "parent_optimizer_memory",
    )
    return {key: value for key, value in metrics.items() if any(token in key for token in health_keys)}


def _summarize_events(events: list[EventTrace]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for event in events:
        for key, value in event.metrics.items():
            if isinstance(value, (int, float)):
                values.setdefault(str(key), []).append(float(value))
    return {f"{key}/mean": float(sum(vals) / max(1, len(vals))) for key, vals in values.items() if vals}


__all__ = ["EpisodeTrace", "EventTrace", "TraceRecorder"]
