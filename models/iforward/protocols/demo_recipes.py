from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

from datasets.iforward_stage2_3.scheduler import Stage23Scheduler
from datasets.iforward_stage2_3.validation_runner import _manual_stage2_3_plan, _rows_for_episode

from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.event import ControlEvent, ProbeEvent, normalize_memory_mode
from models.iforward.runtime.plan import EpisodePlan


@dataclass(frozen=True)
class DemoRecipeResult:
    recipe: str
    plans: tuple[EpisodePlan, ...]


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    if hasattr(node, key):
        value = getattr(node, key)
        return default if value is None else value
    return default


def _to_plain_cfg(cfg: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return dict(OmegaConf.to_container(cfg, resolve=False))
    except Exception:
        pass
    return copy.deepcopy(dict(cfg or {}))


def iforward_demo_cfg(cfg: Any) -> dict[str, Any]:
    raw = _cfg_get(cfg, "iforward_demo", {}) or {}
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "default_recipe": str(_cfg_get(raw, "default_recipe", "repair_showcase_20") or "repair_showcase_20"),
        "output_dir": str(_cfg_get(raw, "output_dir", "") or ""),
        "seed": int(_cfg_get(raw, "seed", 20260701)),
        "memory_ablation": [
            normalize_memory_mode(x)
            for x in list(
                _cfg_get(
                    raw,
                    "memory_ablation",
                    [
                        "full",
                        "memory_off",
                        "memory_read_write",
                        "memory_freeze_write",
                        "memory_shuffle_state",
                        "memory_shuffle_read_write_state",
                        "memory_freeze_after_prefill",
                        "memory_wrong_parent_key_fixed",
                    ],
                )
                or []
            )
        ],
    }


def make_demo_scheduler(
    *,
    cfg: Any,
    dataset: Any,
    scene_id: int,
    segment_id: int,
    seed: int,
    target_frames: int = 24,
    min_frames: int = 8,
    allow_short: bool = True,
) -> Stage23Scheduler:
    cfg_plain = _to_plain_cfg(cfg)
    if bool(_cfg_get(_cfg_get(cfg_plain, "scheduler_stage3_2", {}), "enable", False)):
        sched_key = "scheduler_stage3_2"
    elif bool(_cfg_get(_cfg_get(cfg_plain, "scheduler_stage3_0", {}), "enable", False)):
        sched_key = "scheduler_stage3_0"
    else:
        sched_key = "scheduler_v3"
    sched = dict(_cfg_get(cfg_plain, sched_key, {}) or {})
    sequence = dict(_cfg_get(sched, "sequence", {}) or {})
    sequence.update(
        {
            "min_frames": int(min_frames),
            "max_frames": int(target_frames),
            "frame_count_schedule": [
                {
                    "start_step": 0,
                    "target_frames": int(target_frames),
                    "min_frames": int(min_frames),
                    "allow_short": bool(allow_short),
                }
            ],
        }
    )
    sched["sequence"] = sequence
    repair = dict(_cfg_get(sched, "repair", {}) or {})
    repair["enable"] = False
    sched["repair"] = repair
    producer = dict(_cfg_get(sched, "producer", {}) or {})
    producer["enable"] = False
    sched["producer"] = producer
    cfg_plain[sched_key] = sched
    return Stage23Scheduler(
        dataset=dataset,
        cfg=cfg_plain,
        producer_cfg=producer,
        fixed_scene_id=int(scene_id),
        fixed_segment_id=int(segment_id),
        seed=int(seed),
        fail_fast=False,
    )


def build_demo_v0_plans(
    *,
    cfg: Any,
    dataset: Any,
    recipe: str,
    scene_id: int,
    segment_id: int,
    seed: int = 20260701,
    memory_ablation: Sequence[str] | None = None,
) -> DemoRecipeResult:
    recipe_name = str(recipe)
    target_frames = 20 if recipe_name == "repair_showcase_20" else 24
    scheduler = make_demo_scheduler(
        cfg=cfg,
        dataset=dataset,
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        seed=int(seed),
        target_frames=int(target_frames),
    )
    scheduler.global_step = max(int(scheduler.global_step), int(_cfg_get(scheduler.bootstrap_cfg, "end_step", 0)))
    required: set[str] = set()
    if recipe_name in {"distributional_episode_showcase", "shuffle_vs_chronological_showcase"}:
        required.update({"repeat_refine", "shuffled_coverage"})
    if recipe_name in {"distributional_episode_showcase", "repair_tail_showcase", "memory_ablation_distribution_showcase"}:
        required.add("high_block_repair")
    episode = _build_episode_matching(scheduler, required)
    adapter = Stage3SchedulerAdapter(scheduler)
    if recipe_name in {
        "repair_showcase_20",
        "repair_showcase_24",
        "distributional_episode_showcase",
        "repair_tail_showcase",
        "shuffle_vs_chronological_showcase",
    }:
        return DemoRecipeResult(
            recipe=recipe_name,
            plans=(
                _repair_showcase_plan(
                    adapter=adapter,
                    scheduler=scheduler,
                    episode=episode,
                    protocol_name=f"{recipe_name}/scene{int(scene_id)}/segment{int(segment_id)}",
                    recipe_name=recipe_name,
                ),
            ),
        )
    if recipe_name in {"memory_ablation_showcase", "memory_ablation_distribution_showcase"}:
        modes = [normalize_memory_mode(x) for x in list(memory_ablation or iforward_demo_cfg(cfg)["memory_ablation"])]
        plans = []
        for mode in modes:
            protocol_name = f"{recipe_name}/{mode}/scene{int(scene_id)}/segment{int(segment_id)}"
            if mode == "memory_freeze_after_prefill":
                plans.append(
                    _freeze_after_prefill_demo_plan(
                        adapter=adapter,
                        scheduler=scheduler,
                        episode=episode,
                        protocol_name=protocol_name,
                        recipe_name=recipe_name,
                    )
                )
            else:
                plans.append(
                    adapter.plan_from_episode_v3(
                        episode,
                        protocol_name,
                        memory_mode=mode,
                        source="demo_recipe",
                        metadata={"recipe": recipe_name, "memory_mode": mode, "scene_id": int(scene_id), "segment_id": int(segment_id)},
                    )
                )
        return DemoRecipeResult(recipe=recipe_name, plans=tuple(plans))
    raise ValueError(f"unsupported IForward demo recipe {recipe_name!r}")


def _episode_distribution_types(episode: Any) -> set[str]:
    out: set[str] = set()
    for rollout in tuple(getattr(episode, "rollouts", ()) or ()):
        request_meta = dict(getattr(rollout, "request_meta", {}) or {})
        stage32 = dict(request_meta.get("iforward_stage3_2", {}) or {})
        name = str(stage32.get("distribution_type", ""))
        if name:
            out.add(name)
    return out


def _build_episode_matching(scheduler: Stage23Scheduler, required: set[str]) -> Any:
    if not required or str(getattr(scheduler, "scheduler_version", "")) != "stage3_2_distributional_episode_v1":
        return scheduler._build_episode()
    last = None
    for _ in range(32):
        episode = scheduler._build_episode()
        last = episode
        if required.issubset(_episode_distribution_types(episode)):
            return episode
    return last if last is not None else scheduler._build_episode()


def _repair_showcase_plan(
    *,
    adapter: Stage3SchedulerAdapter,
    scheduler: Stage23Scheduler,
    episode: Any,
    protocol_name: str,
    recipe_name: str = "repair_showcase_20",
) -> EpisodePlan:
    rows = _rows_for_episode(scheduler, episode)
    events: list[Any] = [
        adapter._event_from_rollout_plan(rollout, event_idx=idx, protocol_name=protocol_name)
        for idx, rollout in enumerate(tuple(episode.rollouts))
    ]
    events.append(ControlEvent(event_id=f"{protocol_name}_snapshot_before_repair", kind="snapshot_state", name="before_repair"))
    final_positions = list(range(int(rows.shape[0])))
    probe_plan = _manual_stage2_3_plan(
        scheduler=scheduler,
        episode=episode,
        rows=rows,
        positions=[int(final_positions[-1]) if final_positions else 0],
        repeat_budgets=[1],
        phase="final_all",
        visit_kind="final_all",
        rollout_idx=len(tuple(episode.rollouts)),
        rollouts_per_episode=len(tuple(episode.rollouts)) + 2,
        target_positions=final_positions,
        validation_render_only=True,
    )
    events.append(
        ProbeEvent(
            event_id=f"{protocol_name}_probe_before_repair",
            kind="render_probe",
            target_positions=tuple(final_positions),
            rollout_plan=probe_plan,
            metadata={"demo_stage": "before_repair"},
        )
    )
    base_positions = list(range(int(rows.shape[0])))
    scheduler.rng.shuffle(base_positions)
    repair_positions = base_positions[: min(6, len(base_positions))]
    repair_plan = _manual_stage2_3_plan(
        scheduler=scheduler,
        episode=episode,
        rows=rows,
        positions=repair_positions,
        repeat_budgets=[1 for _ in repair_positions],
        phase="repair",
        visit_kind="repair",
        rollout_idx=len(tuple(episode.rollouts)) + 1,
        rollouts_per_episode=len(tuple(episode.rollouts)) + 3,
        repair_round_idx=0,
        repair_pattern_name="B6R1",
    )
    events.append(adapter._event_from_rollout_plan(repair_plan, event_idx=len(events), protocol_name=protocol_name))
    events.append(
        ProbeEvent(
            event_id=f"{protocol_name}_probe_after_repair",
            kind="render_probe",
            target_positions=tuple(final_positions),
            rollout_plan=probe_plan,
            metadata={"demo_stage": "after_repair"},
        )
    )
    base = adapter.plan_from_episode_v3(
        episode,
        protocol_name,
        source="demo_recipe",
        metadata={"recipe": str(recipe_name), "scene_id": int(episode.scene_id), "segment_id": int(episode.segment_id)},
    )
    return EpisodePlan(
        plan_id="",
        version=base.version,
        episode=base.episode,
        events=tuple(events),
        expected_outputs=("trace.jsonl", "summary.json", "index.html"),
        deterministic=True,
        source="demo_recipe",
        created_at_step=-1,
        metadata={**dict(base.metadata), "recipe": str(recipe_name)},
    ).with_stable_plan_id()


def _freeze_after_prefill_demo_plan(
    *,
    adapter: Stage3SchedulerAdapter,
    scheduler: Stage23Scheduler,
    episode: Any,
    protocol_name: str,
    recipe_name: str,
) -> EpisodePlan:
    rows = _rows_for_episode(scheduler, episode)
    events: list[Any] = [
        adapter._event_from_rollout_plan(rollout, event_idx=idx, protocol_name=protocol_name, memory_mode="full")
        for idx, rollout in enumerate(tuple(episode.rollouts))
    ]
    final_positions = list(range(int(rows.shape[0])))
    target_pos = int(final_positions[-1]) if final_positions else 0
    freeze_plan = _manual_stage2_3_plan(
        scheduler=scheduler,
        episode=episode,
        rows=rows,
        positions=[target_pos],
        repeat_budgets=[1],
        phase="assimilation",
        visit_kind="assimilation",
        rollout_idx=len(tuple(episode.rollouts)),
        rollouts_per_episode=len(tuple(episode.rollouts)) + 1,
        target_positions=final_positions,
    )
    events.append(
        adapter._event_from_rollout_plan(
            freeze_plan,
            event_idx=len(events),
            protocol_name=protocol_name,
            memory_mode="memory_freeze_after_prefill",
        )
    )
    base = adapter.plan_from_episode_v3(
        episode,
        protocol_name,
        source="demo_recipe",
        metadata={"recipe": str(recipe_name), "memory_mode": "memory_freeze_after_prefill"},
    )
    return EpisodePlan(
        plan_id="",
        version=base.version,
        episode=base.episode,
        events=tuple(events),
        expected_outputs=("trace.jsonl", "summary.json", "index.html"),
        deterministic=True,
        source="demo_recipe",
        created_at_step=-1,
        metadata={**dict(base.metadata), "recipe": str(recipe_name), "memory_mode": "memory_freeze_after_prefill"},
    ).with_stable_plan_id()


__all__ = [
    "DemoRecipeResult",
    "build_demo_v0_plans",
    "iforward_demo_cfg",
    "make_demo_scheduler",
]
