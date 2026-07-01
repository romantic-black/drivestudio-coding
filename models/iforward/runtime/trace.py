from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

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
    def __init__(self, output_dir: str | Path, *, record_images: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(self.output_dir)
        self.record_images = bool(record_images)
        self.trace_path = self.output_dir / "trace.jsonl"
        self._fh: Optional[Any] = None
        self._trace: Optional[EpisodeTrace] = None

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

    def record_update(self, event: Any, out: Any, state: Any, *, event_idx: int, memory_mode: str) -> EventTrace:
        return self._record_event(event, out, state, event_idx=event_idx, memory_mode=memory_mode)

    def record_probe(self, event: Any, out: Any, state: Any, *, event_idx: int, memory_mode: str) -> EventTrace:
        return self._record_event(event, out, state, event_idx=event_idx, memory_mode=memory_mode)

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
        self.artifacts.save_json("summary.json", {"plan_id": out.plan_id, "summary": out.summary})
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return out

    def finalize(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("TraceRecorder.finalize called before begin_plan")
        return self.end_plan(self._trace)

    def _record_event(self, event: Any, out: Any, state: Any, *, event_idx: int, memory_mode: str) -> EventTrace:
        resolved = getattr(out, "resolved", None)
        meta = dict(getattr(resolved, "meta", {}) or {}) if resolved is not None else {}
        event_meta = dict(getattr(event, "metadata", {}) or {})
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
            artifacts=self._artifacts_from_output(event, out) if self.record_images else {},
            metadata={**event_meta, "ablation": str(getattr(out, "stats", {}).get("ablation", ""))},
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
