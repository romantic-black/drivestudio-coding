from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import torch

from .event import ControlEvent, ProbeEvent, UpdateEvent, memory_mode_to_forward_ablation, normalize_memory_mode
from .eval_guard import (
    FROZEN_FEEDBACK_RUNNER_MODES,
    apply_frozen_feedback_eval_metadata,
    assert_parameter_versions_unchanged,
    parameter_version_snapshot,
)
from .plan import EpisodePlan
from .state_snapshot import RuntimeSnapshot, clone_runtime_state


@dataclass
class RunnerOptions:
    mode: Literal["train", "validate", "demo", "replay"]
    allow_grad: bool = False
    compute_loss: bool = False
    optimizer_step: bool = False
    update_state: bool = True
    record_images: bool = True
    record_debug_tensors: bool = False
    device: str = "cuda"
    trigger_step: int = -1

    @staticmethod
    def for_mode(mode: str, *, device: str = "cuda", trigger_step: int = -1) -> "RunnerOptions":
        if str(mode) == "train":
            return RunnerOptions(
                mode="train",
                allow_grad=True,
                compute_loss=True,
                optimizer_step=True,
                record_images=False,
                device=str(device),
                trigger_step=int(trigger_step),
            )
        if str(mode) in {"validate", "demo", "replay"}:
            return RunnerOptions(mode=str(mode), device=str(device), trigger_step=int(trigger_step))  # type: ignore[arg-type]
        raise ValueError(f"unsupported IForwardRunner mode {mode!r}")


class IForwardRunner:
    def __init__(
        self,
        model: Any,
        scheduler_adapter: Any = None,
        convert_batch_to_minimal_format: Optional[Callable[[dict[str, Any], torch.device, int], dict[str, Any]]] = None,
    ) -> None:
        self.model = model
        self.scheduler_adapter = scheduler_adapter
        self.convert_batch_to_minimal_format = convert_batch_to_minimal_format

    def run(self, plan: EpisodePlan, recorder: Any, options: RunnerOptions) -> Any:
        eval_versions = (
            parameter_version_snapshot(self.model)
            if str(options.mode) in FROZEN_FEEDBACK_RUNNER_MODES
            else None
        )
        state = None
        memory_mode = "full"
        snapshots: dict[str, RuntimeSnapshot] = {}
        trace = recorder.begin_plan(plan)
        for idx, event in enumerate(plan.events):
            if isinstance(event, ControlEvent):
                state, memory_mode = self._run_control(event, state, memory_mode, snapshots)
                recorder.record_control(event, state, event_idx=idx, memory_mode=memory_mode)
            elif isinstance(event, UpdateEvent):
                event_memory_mode = normalize_memory_mode(event.memory_mode or memory_mode)
                previous_state = state
                out = self._run_update(event, state, event_memory_mode, options)
                state = self._next_state(out) if bool(options.update_state) else state
                recorder.record_update(
                    event,
                    out,
                    state,
                    event_idx=idx,
                    memory_mode=event_memory_mode,
                    previous_state=previous_state,
                )
                memory_mode = event_memory_mode
            elif isinstance(event, ProbeEvent):
                out = self._run_probe(event, state, memory_mode, options)
                recorder.record_probe(event, out, state, event_idx=idx, memory_mode=memory_mode, previous_state=state)
            else:
                raise TypeError(type(event))
        result = recorder.end_plan(trace)
        if eval_versions is not None:
            assert_parameter_versions_unchanged(self.model, eval_versions)
        return result

    def _run_control(
        self,
        event: ControlEvent,
        state: Any,
        memory_mode: str,
        snapshots: dict[str, RuntimeSnapshot],
    ) -> tuple[Any, str]:
        if event.kind == "reset_state":
            return None, memory_mode
        if event.kind == "snapshot_state":
            name = str(event.name or event.event_id)
            snapshots[name] = RuntimeSnapshot(name=name, carried_state=clone_runtime_state(state), metadata=dict(event.metadata or {}))
            return state, memory_mode
        if event.kind == "restore_state":
            name = str(event.name or event.event_id)
            if name not in snapshots:
                raise KeyError(f"runtime snapshot {name!r} does not exist")
            return clone_runtime_state(snapshots[name].carried_state), memory_mode
        if event.kind == "set_memory_mode":
            return state, normalize_memory_mode(event.memory_mode)
        raise ValueError(f"unsupported ControlEvent kind {event.kind!r}")

    def _run_update(self, event: UpdateEvent, carried_state: Any, memory_mode: str, options: RunnerOptions) -> Any:
        if self.scheduler_adapter is None:
            raise ValueError("IForwardRunner requires scheduler_adapter for UpdateEvent")
        raw = self.scheduler_adapter.batch_from_rollout_plan(event.rollout_plan)
        batch = self._convert(raw, options)
        ablation = memory_mode_to_forward_ablation(memory_mode)
        allow_grad = bool(options.allow_grad) and str(options.mode) not in FROZEN_FEEDBACK_RUNNER_MODES
        with torch.set_grad_enabled(allow_grad):
            return self.model.forward_rollout(batch, carried_state=carried_state, ablation=ablation)

    def _run_probe(self, event: ProbeEvent, carried_state: Any, memory_mode: str, options: RunnerOptions) -> Any:
        if self.scheduler_adapter is None:
            raise ValueError("IForwardRunner requires scheduler_adapter for ProbeEvent")
        plan = self.scheduler_adapter.build_render_probe_plan(event)
        raw = self.scheduler_adapter.batch_from_rollout_plan(plan)
        batch = self._convert(raw, options)
        ablation = memory_mode_to_forward_ablation(memory_mode)
        with torch.no_grad():
            return self.model.forward_rollout(batch, carried_state=carried_state, ablation=ablation)

    def _convert(self, raw: dict[str, Any], options: RunnerOptions) -> dict[str, Any]:
        if isinstance(raw.get("_iforward"), dict):
            raw["_iforward"]["validation_force_history_render"] = True
        if callable(self.convert_batch_to_minimal_format):
            batch = self.convert_batch_to_minimal_format(
                raw, torch.device(options.device), int(options.trigger_step)
            )
        else:
            raw["global_step"] = int(options.trigger_step)
            batch = raw
        if str(options.mode) in FROZEN_FEEDBACK_RUNNER_MODES:
            # Keep distribution metadata intact while making evaluation's
            # graph-free feedback policy explicit in both scheduler ABI slots.
            apply_frozen_feedback_eval_metadata(batch)
        return batch

    @staticmethod
    def _next_state(out: Any) -> Any:
        state = getattr(out, "next_state", None)
        return clone_runtime_state(state) if state is not None else None


__all__ = ["IForwardRunner", "RunnerOptions"]
