from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from models.iforward.resolver import (
    IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS,
    IFORWARD_STAGE2_3_SCHEDULER_VERSION,
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


def _phase_max_inner_k(resolved: IForwardResolvedBatch, *, default: int) -> int:
    meta = dict(resolved.meta or {})
    raw = meta.get("phase_max_inner_k", None)
    if raw is None:
        request_meta = dict(meta.get("request_meta", {}) or {})
        raw = dict(request_meta.get("iforward_stage2_3", {}) or {}).get("phase_max_inner_k", None)
    if raw is None:
        return int(default)
    return int(raw)


class Stage23BatchResolver(IForwardBatchResolver):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(expected_scheduler_version=IFORWARD_STAGE2_3_SCHEDULER_VERSION, **kwargs)

    def resolve(self, batch: Dict[str, Any]) -> IForwardResolvedBatch:
        resolved = super().resolve(batch)
        if str(resolved.scheduler_version) not in IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS:
            raise ValueError(
                f"Stage2_3 resolver requires optimizer-sequence scheduler_version in "
                f"{sorted(IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS)!r}, "
                f"got {resolved.scheduler_version!r}"
            )
        self._validate_stage2_3(resolved)
        return resolved

    def _validate_common_step_metadata(self, resolved: IForwardResolvedBatch) -> None:
        for idx, step in enumerate(resolved.steps):
            if int(step.sequence_pos) < 0 or int(step.sequence_pos) >= int(resolved.window_end + 1):
                raise ValueError(f"Stage2_3 steps[{idx}] sequence_pos out of sequence range")
            if int(step.repeat_budget) < 1:
                raise ValueError("Stage2_3 repeat_budget must be positive")
            if int(step.repeat_idx) >= int(step.repeat_budget):
                raise ValueError("Stage2_3 repeat_idx must be < repeat_budget")
            if int(step.global_update_idx_in_episode) < 0:
                raise ValueError("Stage2_3 requires global_update_idx_in_episode")
            if int(step.source_keyframe_idx) < 0:
                raise ValueError("Stage2_3 requires source_keyframe_idx")
            if int(step.global_update_idx_in_episode) > 0 and int(step.previous_visit_sequence_pos) < 0:
                raise ValueError("Stage2_3 requires previous_visit_sequence_pos after the first episode update")
            if int(step.repeat_idx) > 0:
                if int(step.frame_gap) != 0 or abs(float(step.delta_t_sec)) > 1.0e-12:
                    raise ValueError("Stage2_3 repeated same-frame updates must have zero frame_gap and delta_t_sec")
                if int(step.previous_visit_sequence_pos) != int(step.sequence_pos):
                    raise ValueError("Stage2_3 repeated same-frame updates must point previous_visit_sequence_pos to same frame")
            if int(step.physical_frame_gap_abs) != abs(int(step.frame_gap)):
                raise ValueError("Stage2_3 physical_frame_gap_abs must equal abs(frame_gap)")

    def _validate_rollout_metadata(self, resolved: IForwardResolvedBatch) -> None:
        meta = dict(resolved.meta or {})
        repeat0 = tuple(int(step.sequence_pos) for step in _repeat0_steps(resolved))
        rollout_positions_raw = meta.get("rollout_positions", None)
        if rollout_positions_raw is None:
            raise ValueError("Stage2_3 requires rollout_positions metadata")
        rollout_positions = tuple(int(x) for x in list(rollout_positions_raw or []))
        if rollout_positions != repeat0:
            raise ValueError("Stage2_3 rollout_positions must match repeat-0 step positions")
        episode_positions = tuple(int(x) for x in list(meta.get("episode_positions", []) or []))
        if not episode_positions:
            raise ValueError("Stage2_3 requires episode_positions metadata")
        missing = [int(pos) for pos in repeat0 if int(pos) not in set(episode_positions)]
        if missing:
            raise ValueError(f"Stage2_3 rollout positions not present in episode_positions: {missing[:8]}")
        repeat_budgets = tuple(int(x) for x in list(meta.get("repeat_budgets", []) or []))
        expected_budgets = tuple(int(step.repeat_budget) for step in _repeat0_steps(resolved))
        if repeat_budgets != expected_budgets:
            raise ValueError("Stage2_3 repeat_budgets metadata must match repeat-0 step repeat_budget")
        frame_gaps = tuple(int(x) for x in list(meta.get("frame_gaps", []) or []))
        expected_gaps = tuple(int(step.frame_gap) for step in resolved.steps)
        if frame_gaps and frame_gaps != expected_gaps:
            raise ValueError("Stage2_3 frame_gaps metadata must match per-step frame_gap values")
        if "repair_round_idx" not in meta:
            raise ValueError("Stage2_3 requires repair_round_idx metadata")
        if "repair_pattern_name" not in meta:
            raise ValueError("Stage2_3 requires repair_pattern_name metadata")

    def _validate_stage2_3(self, resolved: IForwardResolvedBatch) -> None:
        if not resolved.steps:
            raise ValueError("Stage2_3 requires non-empty steps")
        phases = {str(step.scheduler_phase) for step in resolved.steps}
        if len(phases) != 1:
            raise ValueError(f"Stage2_3 rollout must have one scheduler_phase, got {sorted(phases)!r}")
        phase = next(iter(phases))
        if phase not in {"bootstrap", "assimilation", "repair", "repeat_stability", "final_all"}:
            raise ValueError(f"Stage2_3 invalid scheduler_phase={phase!r}")
        self._validate_rollout_metadata(resolved)
        self._validate_common_step_metadata(resolved)
        if phase == "bootstrap":
            self._validate_bootstrap(resolved)
        elif phase == "assimilation":
            self._validate_assimilation(resolved)
        elif phase == "repair":
            self._validate_repair(resolved)
        elif phase == "repeat_stability":
            self._validate_repeat_stability(resolved)
        else:
            self._validate_final_all(resolved)

    def _validate_bootstrap(self, resolved: IForwardResolvedBatch) -> None:
        positions = _unique_preserve(step.sequence_pos for step in _repeat0_steps(resolved))
        if len(positions) != 1:
            raise ValueError("Stage2_3 bootstrap must visit exactly one frame")
        for step in resolved.steps:
            if str(step.visit_kind) != "bootstrap":
                raise ValueError("Stage2_3 bootstrap requires visit_kind=bootstrap")
            if bool(step.optimizer_memory_read) or bool(step.optimizer_memory_write):
                raise ValueError("Stage2_3 bootstrap must not read/write optimizer memory")
            if bool(step.update_optimizer_memory) or bool(step.temporal_commit):
                raise ValueError("Stage2_3 bootstrap must not update optimizer memory")
            if bool(step.commit_observation_memory):
                raise ValueError("Stage2_3 bootstrap must use fresh state without observation commit")

    def _validate_assimilation(self, resolved: IForwardResolvedBatch) -> None:
        positions = _unique_preserve(step.sequence_pos for step in _repeat0_steps(resolved))
        if len(positions) < 1 or len(positions) > 2:
            raise ValueError("Stage2_3 assimilation rollout must be B1 or B2")
        max_inner_k = _phase_max_inner_k(resolved, default=12)
        if int(resolved.inner_K) > int(max_inner_k):
            raise ValueError(f"Stage2_3 assimilation inner_K must be <= phase_max_inner_k ({max_inner_k})")
        for step in resolved.steps:
            if str(step.visit_kind) not in {"assimilate", "assimilation"}:
                raise ValueError("Stage2_3 assimilation requires assimilate visit_kind")
            if not bool(step.optimizer_memory_read) or not bool(step.optimizer_memory_write):
                raise ValueError("Stage2_3 assimilation reads/writes optimizer memory every repeat")
            if bool(step.update_optimizer_memory) != bool(step.optimizer_memory_write):
                raise ValueError("Stage2_3 update_optimizer_memory must mirror optimizer_memory_write")

    def _validate_repair(self, resolved: IForwardResolvedBatch) -> None:
        positions = _unique_preserve(step.sequence_pos for step in _repeat0_steps(resolved))
        if len(positions) < 1:
            raise ValueError("Stage2_3 repair must visit at least one frame")
        max_inner_k = _phase_max_inner_k(resolved, default=12)
        if int(resolved.inner_K) > int(max_inner_k):
            raise ValueError(f"Stage2_3 repair inner_K must be <= phase_max_inner_k ({max_inner_k})")
        write_values = [bool(step.optimizer_memory_write) for step in resolved.steps]
        if not any(write_values):
            raise ValueError("Stage2_3 repair must write optimizer memory on at least one repeat")
        for idx, step in enumerate(resolved.steps):
            if str(step.visit_kind) != "repair":
                raise ValueError("Stage2_3 repair requires visit_kind=repair")
            if not bool(step.optimizer_memory_read):
                raise ValueError("Stage2_3 repair reads optimizer memory every repeat")
            if bool(step.update_optimizer_memory) != bool(step.optimizer_memory_write):
                raise ValueError("Stage2_3 repair update flag must mirror optimizer_memory_write")
            if bool(step.physical_time_advance):
                raise ValueError("Stage2_3 repair must not advance physical time")
            if not bool(step.optimizer_memory_write) and idx != len(resolved.steps) - 1:
                raise ValueError("Stage2_3 repair may only skip the final write")

    def _validate_repeat_stability(self, resolved: IForwardResolvedBatch) -> None:
        if int(resolved.inner_K) < 1:
            raise ValueError("Stage2_3 repeat_stability requires inner_K >= 1")
        for step in resolved.steps:
            if str(step.visit_kind) != "repeat_stability":
                raise ValueError("Stage2_3 repeat_stability requires visit_kind=repeat_stability")

    def _validate_final_all(self, resolved: IForwardResolvedBatch) -> None:
        if int(resolved.inner_K) != 1 or len(resolved.steps) != 1:
            raise ValueError("Stage2_3 final_all validation must use one render-only step")
        for step in resolved.steps:
            if str(step.visit_kind) != "final_all":
                raise ValueError("Stage2_3 final_all requires visit_kind=final_all")
            if not bool(step.validation_render_only):
                raise ValueError("Stage2_3 final_all requires validation_render_only=true")
            if bool(step.optimizer_memory_read) or bool(step.optimizer_memory_write):
                raise ValueError("Stage2_3 final_all must not read/write optimizer memory")
            if bool(step.update_optimizer_memory) or bool(step.temporal_commit):
                raise ValueError("Stage2_3 final_all must not update optimizer memory")
            if bool(step.commit_observation_memory):
                raise ValueError("Stage2_3 final_all must not commit observation memory")
            if bool(step.physical_time_advance):
                raise ValueError("Stage2_3 final_all must not advance physical time")


__all__ = ["Stage23BatchResolver"]
