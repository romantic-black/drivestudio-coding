from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from .resolver import IFORWARD_SEQUENCE10_SCHEDULER_VERSION, IForwardBatchResolver, IForwardResolvedBatch
from .sequence10_batch import (
    SEQUENCE10_PHASE_BOOTSTRAP,
    SEQUENCE10_PHASE_CAUSAL,
    SEQUENCE10_PHASE_REPAIR,
    SEQUENCE10_VALID_PHASES,
    SEQUENCE10_VALID_VISITS,
    SEQUENCE10_VISIT_BOOTSTRAP,
    SEQUENCE10_VISIT_CAUSAL_FIRST,
    SEQUENCE10_VISIT_REPAIR,
)


def _block_exit_steps(resolved: IForwardResolvedBatch) -> Tuple[Any, ...]:
    return tuple(step for step in resolved.steps if bool(step.is_block_exit))


def _repeat0_steps(resolved: IForwardResolvedBatch) -> Tuple[Any, ...]:
    return tuple(step for step in resolved.steps if int(step.repeat_idx) == 0)


def _unique_preserve_order(values: Iterable[int]) -> Tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for raw in values:
        value = int(raw)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


class IForwardSequence10Resolver(IForwardBatchResolver):
    """Resolver for the explicit Sequence10 training protocol."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(expected_scheduler_version=IFORWARD_SEQUENCE10_SCHEDULER_VERSION, **kwargs)

    def resolve(self, batch: Dict[str, Any]) -> IForwardResolvedBatch:
        resolved = super().resolve(batch)
        if str(resolved.scheduler_version) != IFORWARD_SEQUENCE10_SCHEDULER_VERSION:
            raise ValueError(
                f"Sequence10 resolver requires scheduler_version={IFORWARD_SEQUENCE10_SCHEDULER_VERSION!r}, "
                f"got {resolved.scheduler_version!r}"
            )
        self._validate_sequence10(resolved)
        return resolved

    def _validate_sequence10(self, resolved: IForwardResolvedBatch) -> None:
        if int(resolved.inner_K) > 10:
            raise ValueError("Sequence10 inner_K must be <= 10")
        if not resolved.steps:
            raise ValueError("Sequence10 requires non-empty steps")
        phases = {str(step.scheduler_phase) for step in resolved.steps}
        if len(phases) != 1:
            raise ValueError(f"Sequence10 rollout must have one scheduler_phase, got {sorted(phases)!r}")
        phase = next(iter(phases))
        if phase not in SEQUENCE10_VALID_PHASES:
            raise ValueError(f"Sequence10 invalid scheduler_phase={phase!r}")
        for idx, step in enumerate(resolved.steps):
            if int(step.sequence_pos) < 0 or int(step.sequence_pos) >= 10:
                raise ValueError(f"Sequence10 steps[{idx}] sequence_pos must be in [0, 9]")
            if str(step.visit_kind) not in SEQUENCE10_VALID_VISITS:
                raise ValueError(f"Sequence10 steps[{idx}] invalid visit_kind={step.visit_kind!r}")
            if int(step.frame_gap) not in (0, 1, 2):
                raise ValueError(f"Sequence10 steps[{idx}] frame_gap must be 0, 1, or 2")
            if bool(step.temporal_commit) and not bool(step.is_block_exit):
                raise ValueError("Sequence10 temporal_commit is only allowed on block exit")
        if phase == SEQUENCE10_PHASE_BOOTSTRAP:
            self._validate_bootstrap(resolved)
        elif phase == SEQUENCE10_PHASE_CAUSAL:
            self._validate_causal(resolved)
        elif phase == SEQUENCE10_PHASE_REPAIR:
            self._validate_repair(resolved)

    def _validate_bootstrap(self, resolved: IForwardResolvedBatch) -> None:
        positions = _unique_preserve_order(step.sequence_pos for step in _repeat0_steps(resolved))
        if len(positions) != 1:
            raise ValueError("Sequence10 bootstrap must visit exactly one block")
        repeats = {int(step.repeats_per_block) for step in resolved.steps}
        if repeats - {4, 6, 8}:
            raise ValueError("Sequence10 bootstrap repeats_per_block must be one of 4, 6, 8")
        if any(str(step.visit_kind) != SEQUENCE10_VISIT_BOOTSTRAP for step in resolved.steps):
            raise ValueError("Sequence10 bootstrap requires visit_kind=bootstrap")
        if any(bool(step.temporal_read) for step in resolved.steps):
            raise ValueError("Sequence10 bootstrap must not read temporal memory")
        if any(bool(step.temporal_commit) for step in resolved.steps):
            raise ValueError("Sequence10 bootstrap must not commit temporal memory")
        if any(bool(step.commit_observation_memory) for step in resolved.steps):
            raise ValueError("Sequence10 bootstrap must not commit observation memory")
        if any(bool(step.update_optimizer_memory) for step in resolved.steps):
            raise ValueError("Sequence10 bootstrap must not update optimizer memory")
        if resolved.history_rollout_target_indices:
            raise ValueError("Sequence10 bootstrap must not produce history refs")

    def _validate_causal(self, resolved: IForwardResolvedBatch) -> None:
        positions = _unique_preserve_order(step.sequence_pos for step in _repeat0_steps(resolved))
        if len(positions) != 2:
            raise ValueError("Sequence10 causal rollout must be B2")
        if positions[1] != positions[0] + 1:
            raise ValueError("Sequence10 causal positions must be contiguous")
        if any(int(step.repeats_per_block) != 4 for step in resolved.steps):
            raise ValueError("Sequence10 causal rollout must be R4")
        if int(resolved.inner_K) != 8:
            raise ValueError("Sequence10 causal rollout must have inner_K=8")
        if any(str(step.visit_kind) != SEQUENCE10_VISIT_CAUSAL_FIRST for step in resolved.steps):
            raise ValueError("Sequence10 causal requires visit_kind=causal_first")
        for step in resolved.steps:
            if not bool(step.temporal_read):
                raise ValueError("Sequence10 causal requires temporal_read=true")
            if bool(step.temporal_commit) != bool(step.is_block_exit):
                raise ValueError("Sequence10 causal temporal_commit must match block exit")
            if bool(step.update_optimizer_memory) != bool(step.is_block_exit):
                raise ValueError("Sequence10 causal optimizer memory update must match block exit")
            if bool(step.commit_observation_memory) != bool(step.is_block_enter):
                raise ValueError("Sequence10 causal observation commit must match block enter")
            if not bool(step.physical_time_advance):
                raise ValueError("Sequence10 causal requires physical_time_advance=true")
        if len(_block_exit_steps(resolved)) != 2:
            raise ValueError("Sequence10 causal must have two block exits")

    def _validate_repair(self, resolved: IForwardResolvedBatch) -> None:
        positions = _unique_preserve_order(step.sequence_pos for step in _repeat0_steps(resolved))
        if len(positions) != 10:
            raise ValueError("Sequence10 repair rollout must visit all 10 unique blocks")
        if tuple(positions) == tuple(range(10)):
            raise ValueError("Sequence10 repair permutation must be non-identity")
        if any(int(step.repeats_per_block) != 1 for step in resolved.steps):
            raise ValueError("Sequence10 repair rollout must be R1")
        if int(resolved.inner_K) != 10:
            raise ValueError("Sequence10 repair rollout must have inner_K=10")
        if any(str(step.visit_kind) != SEQUENCE10_VISIT_REPAIR for step in resolved.steps):
            raise ValueError("Sequence10 repair requires visit_kind=repair")
        for step in resolved.steps:
            if not bool(step.temporal_read):
                raise ValueError("Sequence10 repair still reads temporal memory")
            if bool(step.temporal_commit):
                raise ValueError("Sequence10 repair must not commit temporal memory")
            if bool(step.update_optimizer_memory):
                raise ValueError("Sequence10 repair must not update optimizer memory")
            if bool(step.commit_observation_memory):
                raise ValueError("Sequence10 repair must not commit observation memory")
            if bool(step.physical_time_advance):
                raise ValueError("Sequence10 repair must not advance physical time")
            if int(step.frame_gap) != 0:
                raise ValueError("Sequence10 repair frame_gap must be 0")


__all__ = ["IForwardSequence10Resolver"]
