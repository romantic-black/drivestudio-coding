from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from models.iforward.resolver import (
    IFORWARD_STAGE2_2_SCHEDULER_VERSION,
    IForwardBatchResolver,
    IForwardResolvedBatch,
)


def _repeat0_steps(resolved: IForwardResolvedBatch) -> Tuple[Any, ...]:
    return tuple(step for step in resolved.steps if int(step.repeat_idx) == 0)


def _unique_preserve(values: Iterable[int]) -> Tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for raw in values:
        value = int(raw)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


class Stage22BatchResolver(IForwardBatchResolver):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(expected_scheduler_version=IFORWARD_STAGE2_2_SCHEDULER_VERSION, **kwargs)

    def resolve(self, batch: Dict[str, Any]) -> IForwardResolvedBatch:
        resolved = super().resolve(batch)
        if str(resolved.scheduler_version) != IFORWARD_STAGE2_2_SCHEDULER_VERSION:
            raise ValueError(
                f"Stage2_2 resolver requires scheduler_version={IFORWARD_STAGE2_2_SCHEDULER_VERSION!r}, "
                f"got {resolved.scheduler_version!r}"
            )
        self._validate_stage2_2(resolved)
        return resolved

    def _validate_stage2_2(self, resolved: IForwardResolvedBatch) -> None:
        if not resolved.steps:
            raise ValueError("Stage2_2 requires non-empty steps")
        phases = {str(step.scheduler_phase) for step in resolved.steps}
        if len(phases) != 1:
            raise ValueError(f"Stage2_2 rollout must have one scheduler_phase, got {sorted(phases)!r}")
        phase = next(iter(phases))
        if phase not in {"bootstrap", "causal", "repair", "stress"}:
            raise ValueError(f"Stage2_2 invalid scheduler_phase={phase!r}")
        for idx, step in enumerate(resolved.steps):
            if int(step.sequence_pos) < 0 or int(step.sequence_pos) >= 10:
                raise ValueError(f"Stage2_2 steps[{idx}] sequence_pos must be in [0, 9]")
            if bool(step.temporal_commit) and not bool(step.is_block_exit):
                raise ValueError("Stage2_2 temporal_commit is only allowed on block exit")
            if bool(step.physical_time_advance) and not bool(step.temporal_commit):
                raise ValueError("Stage2_2 physical_time_advance requires temporal_commit")
            if int(step.timestamp_us) < 0:
                raise ValueError("Stage2_2 timestamp_us must be non-negative")
        if phase == "bootstrap":
            self._validate_bootstrap(resolved)
        elif phase == "causal":
            self._validate_causal(resolved)
        elif phase == "repair":
            self._validate_repair(resolved)
        else:
            self._validate_stress(resolved)

    def _validate_bootstrap(self, resolved: IForwardResolvedBatch) -> None:
        positions = _unique_preserve(step.sequence_pos for step in _repeat0_steps(resolved))
        if len(positions) != 1:
            raise ValueError("Stage2_2 bootstrap must visit exactly one raw frame")
        if any(bool(step.temporal_read) for step in resolved.steps):
            raise ValueError("Stage2_2 bootstrap must not read temporal memory")
        if any(bool(step.temporal_commit) for step in resolved.steps):
            raise ValueError("Stage2_2 bootstrap must not commit temporal memory")
        if any(bool(step.commit_observation_memory) for step in resolved.steps):
            raise ValueError("Stage2_2 bootstrap must not commit observation memory")
        if any(bool(step.update_optimizer_memory) for step in resolved.steps):
            raise ValueError("Stage2_2 bootstrap must not update optimizer memory")

    def _validate_causal(self, resolved: IForwardResolvedBatch) -> None:
        repeat0 = _repeat0_steps(resolved)
        positions = _unique_preserve(step.sequence_pos for step in repeat0)
        if len(positions) != 2:
            raise ValueError("Stage2_2 causal rollout must be B2")
        if any(int(step.repeats_per_block) != 4 for step in resolved.steps):
            raise ValueError("Stage2_2 causal rollout must be R4")
        if int(resolved.inner_K) != 8:
            raise ValueError("Stage2_2 causal rollout must have inner_K=8")
        timestamps = [int(step.timestamp_us) for step in repeat0]
        if any(int(b) <= int(a) for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("Stage2_2 causal timestamps must be strictly increasing by raw frame visit")
        for step in resolved.steps:
            if str(step.visit_kind) != "causal_first":
                raise ValueError("Stage2_2 causal requires visit_kind=causal_first")
            if bool(step.temporal_read) is not True:
                raise ValueError("Stage2_2 causal requires temporal_read=true")
            if bool(step.temporal_commit) != bool(step.is_block_exit):
                raise ValueError("Stage2_2 causal temporal_commit must match block exit")
            if bool(step.physical_time_advance) != bool(step.is_block_exit):
                raise ValueError("Stage2_2 causal physical_time_advance must match block exit")
            if bool(step.update_optimizer_memory) != bool(step.is_block_exit):
                raise ValueError("Stage2_2 causal update_optimizer_memory must match block exit")
            if bool(step.commit_observation_memory) != bool(step.is_block_enter):
                raise ValueError("Stage2_2 causal commit_observation_memory must match block enter")

    def _validate_repair(self, resolved: IForwardResolvedBatch) -> None:
        repeat0 = _repeat0_steps(resolved)
        positions = _unique_preserve(step.sequence_pos for step in repeat0)
        if len(positions) < 1 or len(positions) > 10:
            raise ValueError("Stage2_2 repair must visit 1 to 10 raw-frame positions")
        if len(positions) != len(repeat0):
            raise ValueError("Stage2_2 repair must visit unique raw-frame positions")
        if any(int(step.repeats_per_block) != 1 for step in resolved.steps):
            raise ValueError("Stage2_2 repair rollout must be R1")
        if int(resolved.inner_K) != int(len(positions)):
            raise ValueError("Stage2_2 repair inner_K must equal visited raw-frame count")
        for step in resolved.steps:
            if str(step.visit_kind) != "repair":
                raise ValueError("Stage2_2 repair requires visit_kind=repair")
            if not bool(step.temporal_read):
                raise ValueError("Stage2_2 repair still reads temporal memory")
            if bool(step.temporal_commit) or bool(step.physical_time_advance):
                raise ValueError("Stage2_2 repair must not commit temporal or advance timestamp")
            if bool(step.commit_observation_memory) or bool(step.update_optimizer_memory):
                raise ValueError("Stage2_2 repair must not commit observation or optimizer memory")
            if not bool(step.repair_no_commit):
                raise ValueError("Stage2_2 repair step must carry repair_no_commit=true")

    def _validate_stress(self, resolved: IForwardResolvedBatch) -> None:
        positions = _unique_preserve(step.sequence_pos for step in _repeat0_steps(resolved))
        if len(positions) < 1:
            raise ValueError("Stage2_2 stress validation must visit at least one raw-frame position")
        for step in resolved.steps:
            if str(step.visit_kind) != "stress":
                raise ValueError("Stage2_2 stress validation requires visit_kind=stress")
            if not bool(step.temporal_read):
                raise ValueError("Stage2_2 stress validation reads temporal memory")
            if bool(step.temporal_commit) or bool(step.physical_time_advance):
                raise ValueError("Stage2_2 stress validation must not commit temporal or advance timestamp")
            if bool(step.commit_observation_memory) or bool(step.update_optimizer_memory):
                raise ValueError("Stage2_2 stress validation must not commit observation or optimizer memory")
            if bool(step.frame_gap) or abs(float(step.delta_t_sec)) > 0.0:
                raise ValueError("Stage2_2 stress validation must use zero physical time")


__all__ = ["Stage22BatchResolver"]
