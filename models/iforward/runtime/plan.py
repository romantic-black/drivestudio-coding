from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from datasets.iforward_stage2_3.schema import RolloutPlanV3, Stage23StepPlan
from datasets.train_scheduler_iforward import IForwardFinalSupervisionPlan

from .event import ControlEvent, EpisodeSpec, ProbeEvent, RuntimeEvent, UpdateEvent


def _plain(value: Any) -> Any:
    if isinstance(value, RolloutPlanV3):
        return {"__rollout_plan_v3__": _plain(dataclasses.asdict(value))}
    if dataclasses.is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [_plain(x) for x in value]
    if isinstance(value, list):
        return [_plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def rollout_plan_v3_to_json_dict(plan: RolloutPlanV3) -> dict[str, Any]:
    return dict(_plain(plan)["__rollout_plan_v3__"])


def rollout_plan_v3_from_json_dict(data: dict[str, Any]) -> RolloutPlanV3:
    values = dict(data)
    steps = values.get("steps", []) or []
    values["steps"] = [Stage23StepPlan(**dict(step)) for step in steps]
    final = values.get("final_supervision", {}) or {}
    values["final_supervision"] = IForwardFinalSupervisionPlan(**dict(final))
    return RolloutPlanV3(**values)


def _maybe_rollout_from_plain(value: Any) -> Any:
    if isinstance(value, dict) and "__rollout_plan_v3__" in value:
        return rollout_plan_v3_from_json_dict(dict(value["__rollout_plan_v3__"]))
    return value


def _event_to_json_dict(event: RuntimeEvent) -> dict[str, Any]:
    out = {
        "__event_type__": type(event).__name__,
    }
    for field_info in dataclasses.fields(event):
        out[field_info.name] = _plain(getattr(event, field_info.name))
    return out


def _event_from_json_dict(data: dict[str, Any]) -> RuntimeEvent:
    payload = dict(data)
    event_type = str(payload.pop("__event_type__", ""))
    cls: type[RuntimeEvent]
    if event_type == "UpdateEvent":
        cls = UpdateEvent
    elif event_type == "ProbeEvent":
        cls = ProbeEvent
    elif event_type == "ControlEvent":
        cls = ControlEvent
    else:
        raise ValueError(f"unsupported EpisodePlan event type {event_type!r}")
    for key in ("rollout_plan",):
        if key in payload:
            payload[key] = _maybe_rollout_from_plain(payload[key])
    for key in ("input_positions", "repeat_budgets", "target_positions", "target_frame_ids", "target_cams", "roles"):
        if key in payload and isinstance(payload[key], list):
            payload[key] = tuple(payload[key])
    return cls(**payload)


def _episode_from_json_dict(data: dict[str, Any]) -> EpisodeSpec:
    payload = dict(data)
    for key in ("frame_ids", "frame_positions", "cam_ids"):
        if key in payload and isinstance(payload[key], list):
            payload[key] = tuple(int(x) for x in payload[key])
    return EpisodeSpec(**payload)


def stable_plan_id(data: dict[str, Any]) -> str:
    payload = dict(data)
    payload.pop("plan_id", None)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EpisodePlan:
    plan_id: str
    version: str
    episode: EpisodeSpec
    events: tuple[RuntimeEvent, ...]
    expected_outputs: tuple[str, ...] = ()
    deterministic: bool = True
    source: Literal["scheduler_adapter", "validation_recipe", "demo_recipe", "manual", "replay"] = "manual"
    created_at_step: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        data = {
            "plan_id": str(self.plan_id),
            "version": str(self.version),
            "episode": _plain(self.episode),
            "events": [_event_to_json_dict(event) for event in self.events],
            "expected_outputs": [str(x) for x in self.expected_outputs],
            "deterministic": bool(self.deterministic),
            "source": str(self.source),
            "created_at_step": int(self.created_at_step),
            "metadata": _plain(self.metadata),
        }
        if not data["plan_id"]:
            data["plan_id"] = stable_plan_id(data)
        return data

    def with_stable_plan_id(self) -> "EpisodePlan":
        return dataclasses.replace(self, plan_id=stable_plan_id(self.to_json_dict()))

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> "EpisodePlan":
        payload = dict(data)
        episode = _episode_from_json_dict(dict(payload["episode"]))
        events = tuple(_event_from_json_dict(dict(event)) for event in list(payload.get("events", []) or []))
        expected_outputs = tuple(str(x) for x in list(payload.get("expected_outputs", []) or []))
        return EpisodePlan(
            plan_id=str(payload.get("plan_id", "")),
            version=str(payload.get("version", "")),
            episode=episode,
            events=events,
            expected_outputs=expected_outputs,
            deterministic=bool(payload.get("deterministic", True)),
            source=str(payload.get("source", "manual")),  # type: ignore[arg-type]
            created_at_step=int(payload.get("created_at_step", -1)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


__all__ = [
    "EpisodePlan",
    "rollout_plan_v3_from_json_dict",
    "rollout_plan_v3_to_json_dict",
    "stable_plan_id",
]
