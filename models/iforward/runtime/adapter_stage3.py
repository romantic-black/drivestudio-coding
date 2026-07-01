from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from datasets.iforward_stage2_3.schema import EpisodePlanV3, RolloutPlanV3
from datasets.iforward_stage2_3.scheduler import Stage23Scheduler

from .event import EpisodeSpec, ProbeEvent, UpdateEvent, normalize_memory_mode
from .plan import EpisodePlan


class Stage3SchedulerAdapter:
    def __init__(self, scheduler: Stage23Scheduler) -> None:
        self.scheduler = scheduler

    def sample_train_plan(self, step: int) -> EpisodePlan:
        state = self.scheduler.state_dict()
        try:
            batch = self.scheduler.next_batch()
            meta = dict(batch.get("_iforward", {}) or {})
            rollout = self._rollout_from_meta(meta)
            episode = EpisodeSpec(
                scene_id=int(meta.get("scene_id", -1)),
                segment_id=int(meta.get("segment_id", -1)),
                sequence_id=int(meta.get("sequence_id", -1)),
                frame_ids=tuple(int(x) for x in list(meta.get("sequence_source_frame_indices", []) or [])),
                frame_positions=tuple(int(x) for x in list(meta.get("episode_positions", []) or [])),
                cam_ids=tuple(range(int(meta.get("num_cams", 0) or 0))),
                protocol_name="train",
                metadata={"scheduler_meta": meta},
            )
            event = self._event_from_rollout_plan(rollout, event_idx=0, protocol_name="train") if rollout is not None else None
            if event is None:
                raise ValueError("Stage3SchedulerAdapter.sample_train_plan could not recover rollout_plan from batch")
            return EpisodePlan(
                plan_id="",
                version="iforward_episode_plan_v1",
                episode=episode,
                events=(event,),
                source="scheduler_adapter",
                created_at_step=int(step),
            ).with_stable_plan_id()
        finally:
            self.scheduler.load_state_dict(state)

    def plan_from_episode_v3(
        self,
        episode_v3: EpisodePlanV3,
        protocol_name: str,
        *,
        memory_mode: str = "full",
        source: str = "validation_recipe",
        created_at_step: int = -1,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodePlan:
        events = tuple(
            self._event_from_rollout_plan(
                rollout,
                event_idx=idx,
                protocol_name=str(protocol_name),
                memory_mode=memory_mode,
            )
            for idx, rollout in enumerate(tuple(episode_v3.rollouts))
        )
        episode = EpisodeSpec(
            scene_id=int(episode_v3.scene_id),
            segment_id=int(episode_v3.segment_id),
            sequence_id=int(episode_v3.sequence_id),
            frame_ids=tuple(int(x) for x in tuple(episode_v3.frame_set)),
            frame_positions=tuple(range(len(tuple(episode_v3.frame_set)))),
            cam_ids=tuple(range(int(getattr(events[0].rollout_plan, "num_cams", 0) or 0))) if events else (),
            protocol_name=str(protocol_name),
            metadata={**dict(episode_v3.metadata or {}), **dict(metadata or {})},
        )
        return EpisodePlan(
            plan_id="",
            version="iforward_episode_plan_v1",
            episode=episode,
            events=events,
            expected_outputs=("trace.jsonl", "summary.json"),
            deterministic=True,
            source=str(source),  # type: ignore[arg-type]
            created_at_step=int(created_at_step),
            metadata={"protocol": str(protocol_name), **dict(metadata or {})},
        ).with_stable_plan_id()

    def batch_from_rollout_plan(self, rollout_plan: RolloutPlanV3) -> dict[str, Any]:
        return self.scheduler._batch_from_plan(rollout_plan)

    def build_render_probe_plan(self, event: ProbeEvent) -> RolloutPlanV3:
        plan = getattr(event, "rollout_plan", None)
        if plan is None:
            plan = (getattr(event, "metadata", {}) or {}).get("rollout_plan")
        if plan is None:
            raise ValueError("ProbeEvent requires a render-only rollout_plan for the P0/P1 runtime")
        return self.force_render_only(plan)

    @staticmethod
    def force_render_only(plan: RolloutPlanV3) -> RolloutPlanV3:
        steps = []
        for step in list(getattr(plan, "steps", []) or []):
            values = dict(step.__dict__)
            values.update(
                {
                    "commit_observation_memory": False,
                    "update_optimizer_memory": False,
                    "record_update_norm": False,
                    "commit_support_on_exit": False,
                    "commit_residual_on_exit": False,
                    "temporal_read": False,
                    "temporal_commit": False,
                    "optimizer_memory_read": False,
                    "optimizer_memory_write": False,
                    "visit_memory_mask": False,
                    "physical_time_advance": False,
                    "validation_render_only": True,
                }
            )
            steps.append(type(step)(**values))
        return dataclasses.replace(
            plan,
            steps=steps,
            inner_K=len(steps),
            requested_inner_K=len(steps),
            actual_inner_K=len(steps),
            temporal_read_count=0,
            temporal_commit_count=0,
            optimizer_memory_read_count=0,
            optimizer_memory_write_count=0,
            observation_commit_count=0,
            scheduler_phase="render_probe",
            rollout_phase="render_probe",
        )

    def _event_from_rollout_plan(
        self,
        rollout_plan: RolloutPlanV3,
        *,
        event_idx: int,
        protocol_name: str,
        memory_mode: str = "full",
    ) -> UpdateEvent:
        phase = str(getattr(rollout_plan, "scheduler_phase", "") or "assimilation")
        return UpdateEvent(
            event_id=f"{str(protocol_name)}_rollout_{int(event_idx):03d}",
            kind="repair_update" if phase == "repair" else "observe_update",
            rollout_plan=rollout_plan,
            phase=phase,  # type: ignore[arg-type]
            input_positions=tuple(int(x) for x in list(getattr(rollout_plan, "rollout_positions", []) or [])),
            repeat_budgets=tuple(int(x) for x in list(getattr(rollout_plan, "repeat_budgets", []) or [])),
            blocks_per_rollout=int(getattr(rollout_plan, "blocks_per_rollout", 0)),
            repeats_per_block=int(getattr(rollout_plan, "repeats_per_block", 0)),
            memory_read=bool(int(getattr(rollout_plan, "optimizer_memory_read_count", 0)) > 0),
            memory_write=bool(int(getattr(rollout_plan, "optimizer_memory_write_count", 0)) > 0),
            observation_commit=bool(int(getattr(rollout_plan, "observation_commit_count", 0)) > 0),
            repair_training=bool(phase == "repair"),
            memory_mode=normalize_memory_mode(memory_mode),
            metadata=self._metadata_from_rollout(rollout_plan),
        )

    @staticmethod
    def _metadata_from_rollout(rollout_plan: RolloutPlanV3) -> dict[str, Any]:
        return {
            "scene_id": int(getattr(rollout_plan, "scene_id", -1)),
            "segment_id": int(getattr(rollout_plan, "segment_id", -1)),
            "sequence_id": int(getattr(rollout_plan, "sequence_id", -1)),
            "scheduler_phase": str(getattr(rollout_plan, "scheduler_phase", "")),
            "rollout_phase": str(getattr(rollout_plan, "rollout_phase", "")),
            "sequence_length": int(getattr(rollout_plan, "sequence_length", 0)),
            "rollout_positions": [int(x) for x in list(getattr(rollout_plan, "rollout_positions", []) or [])],
            "history_positions": [int(x) for x in list(getattr(rollout_plan, "history_positions", []) or [])],
            "repair_positions": [int(x) for x in list(getattr(rollout_plan, "repair_positions", []) or [])],
            "repeat_budgets": [int(x) for x in list(getattr(rollout_plan, "repeat_budgets", []) or [])],
            "optimizer_memory_read_count": int(getattr(rollout_plan, "optimizer_memory_read_count", 0)),
            "optimizer_memory_write_count": int(getattr(rollout_plan, "optimizer_memory_write_count", 0)),
            "observation_commit_count": int(getattr(rollout_plan, "observation_commit_count", 0)),
        }

    @staticmethod
    def _rollout_from_meta(meta: dict[str, Any]) -> RolloutPlanV3 | None:
        rollout = meta.get("rollout_plan")
        return rollout if isinstance(rollout, RolloutPlanV3) else None


__all__ = ["Stage3SchedulerAdapter"]
