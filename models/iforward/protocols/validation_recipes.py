from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from datasets.iforward_stage2_3.scheduler import Stage23Scheduler
from datasets.iforward_stage2_3.validation_runner import _manual_stage2_3_plan, _rows_for_episode

from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.event import ControlEvent, ProbeEvent, normalize_memory_mode
from models.iforward.runtime.plan import EpisodePlan


@dataclass(frozen=True)
class FrameSetSpec:
    name: str
    target_frames: int
    min_frames: int
    allow_short: bool = False


DEFAULT_FRAME_SETS = {
    "seq10": FrameSetSpec(name="seq10", target_frames=10, min_frames=10, allow_short=False),
    "seq24": FrameSetSpec(name="seq24", target_frames=24, min_frames=8, allow_short=True),
}


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


def iforward_validation_v4_cfg(cfg: Any) -> dict[str, Any]:
    raw = _cfg_get(cfg, "iforward_validation_v4", {}) or {}
    protocols_raw = _cfg_get(raw, "protocols", {}) or {}
    if not protocols_raw:
        protocols_raw = {
            "assimilation_timeline": True,
            "repair_before_after": True,
            "order_robustness": True,
            "repeat_stability": True,
            "memory_ablation": True,
            "state_health": True,
        }
    frame_sets_raw = _cfg_get(raw, "frame_sets", ["seq10", "seq24"]) or ["seq10", "seq24"]
    frame_sets = _parse_frame_sets(frame_sets_raw)
    memory_modes = [
        normalize_memory_mode(x)
        for x in list(_cfg_get(raw, "memory_ablation", ["full", "memory_off", "memory_read_write", "memory_freeze_write", "memory_shuffle_state"]) or [])
    ]
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "interval_steps": int(_cfg_get(raw, "interval_steps", 5000)),
        "run_at_train_start": bool(_cfg_get(raw, "run_at_train_start", False)),
        "max_entries_debug": int(_cfg_get(raw, "max_entries_debug", _cfg_get(raw, "max_entries", 2))),
        "seed": int(_cfg_get(raw, "seed", 20260701)),
        "frame_sets": frame_sets,
        "repair_permutations": int(_cfg_get(raw, "repair_permutations", 3)),
        "repeat_stability": [int(x) for x in list(_cfg_get(raw, "repeat_stability", [8, 16]) or [])],
        "memory_ablation": memory_modes,
        "protocols": dict(protocols_raw),
        "report": dict(_cfg_get(raw, "report", {}) or {}),
    }


def build_validation_v4_plans(
    *,
    cfg: Any,
    dataset: Any,
    max_entries: int | None = None,
    frame_sets: Sequence[str] | None = None,
    repair_permutations: int | None = None,
    memory_ablation: Sequence[str] | None = None,
) -> list[EpisodePlan]:
    val = iforward_validation_v4_cfg(cfg)
    selected_frame_sets = _select_frame_sets(val["frame_sets"], frame_sets)
    max_entries_i = int(max_entries if max_entries is not None else val["max_entries_debug"])
    repair_perms = int(repair_permutations if repair_permutations is not None else val["repair_permutations"])
    memory_modes = [normalize_memory_mode(x) for x in list(memory_ablation or val["memory_ablation"] or ["full"])]
    protocols = dict(val["protocols"])
    plans: list[EpisodePlan] = []
    for fs in selected_frame_sets:
        for entry_idx in range(max_entries_i):
            scheduler = _make_scheduler_for_frame_set(cfg, dataset, fs=fs, seed=int(val["seed"]) + int(entry_idx))
            scheduler.global_step = max(int(scheduler.global_step), int(_cfg_get(scheduler.bootstrap_cfg, "end_step", 0)))
            episode = scheduler._build_episode()
            adapter = Stage3SchedulerAdapter(scheduler)
            if bool(protocols.get("assimilation_timeline", True)):
                plans.append(
                    adapter.plan_from_episode_v3(
                        episode,
                        f"assimilation_timeline/{fs.name}/entry{entry_idx}",
                        memory_mode="full",
                        metadata={"frame_set": fs.name, "entry_idx": int(entry_idx)},
                    )
                )
            if bool(protocols.get("memory_ablation", True)):
                for mode in memory_modes:
                    plans.append(
                        adapter.plan_from_episode_v3(
                            episode,
                            f"memory_ablation/{fs.name}/{mode}/entry{entry_idx}",
                            memory_mode=mode,
                            metadata={"frame_set": fs.name, "entry_idx": int(entry_idx), "memory_mode": mode},
                        )
                    )
            if bool(protocols.get("repair_before_after", True)):
                plans.append(
                    _repair_plan(
                        adapter=adapter,
                        scheduler=scheduler,
                        episode=episode,
                        protocol_name=f"repair_before_after/{fs.name}/entry{entry_idx}",
                        repair_permutations=1,
                        pattern_name="B6R1",
                    )
                )
            if bool(protocols.get("order_robustness", True)):
                plans.append(
                    _repair_plan(
                        adapter=adapter,
                        scheduler=scheduler,
                        episode=episode,
                        protocol_name=f"order_robustness/{fs.name}/entry{entry_idx}",
                        repair_permutations=repair_perms,
                        pattern_name="order_perm",
                    )
                )
            if bool(protocols.get("repeat_stability", True)):
                plans.append(
                    _repeat_stability_plan(
                        adapter=adapter,
                        scheduler=scheduler,
                        episode=episode,
                        protocol_name=f"repeat_stability/{fs.name}/entry{entry_idx}",
                        repeats=val["repeat_stability"],
                    )
                )
    return plans


def _parse_frame_sets(raw: Any) -> list[FrameSetSpec]:
    out: list[FrameSetSpec] = []
    for item in list(raw if not isinstance(raw, str) else [raw]):
        if isinstance(item, str):
            out.append(DEFAULT_FRAME_SETS[str(item)])
        else:
            name = str(_cfg_get(item, "name", _cfg_get(item, "id", f"seq{_cfg_get(item, 'target_frames', 10)}")))
            out.append(
                FrameSetSpec(
                    name=name,
                    target_frames=int(_cfg_get(item, "target_frames", 10)),
                    min_frames=int(_cfg_get(item, "min_frames", _cfg_get(item, "target_frames", 10))),
                    allow_short=bool(_cfg_get(item, "allow_short", False)),
                )
            )
    return out


def _select_frame_sets(frame_sets: list[FrameSetSpec], selected: Sequence[str] | None) -> list[FrameSetSpec]:
    if not selected:
        return list(frame_sets)
    wanted = {str(x) for x in selected}
    return [fs for fs in frame_sets if fs.name in wanted]


def _make_scheduler_for_frame_set(cfg: Any, dataset: Any, *, fs: FrameSetSpec, seed: int) -> Stage23Scheduler:
    cfg_plain = _to_plain_cfg(cfg)
    sched_key = "scheduler_stage3_0" if bool(_cfg_get(_cfg_get(cfg_plain, "scheduler_stage3_0", {}), "enable", False)) else "scheduler_v3"
    sched = dict(_cfg_get(cfg_plain, sched_key, {}) or {})
    sequence = dict(_cfg_get(sched, "sequence", {}) or {})
    sequence.update(
        {
            "min_frames": int(fs.min_frames),
            "max_frames": int(fs.target_frames),
            "frame_count_schedule": [
                {
                    "start_step": 0,
                    "target_frames": int(fs.target_frames),
                    "min_frames": int(fs.min_frames),
                    "allow_short": bool(fs.allow_short),
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
    return Stage23Scheduler(dataset=dataset, cfg=cfg_plain, producer_cfg=producer, seed=int(seed), fail_fast=False)


def _repair_plan(
    *,
    adapter: Stage3SchedulerAdapter,
    scheduler: Stage23Scheduler,
    episode: Any,
    protocol_name: str,
    repair_permutations: int,
    pattern_name: str,
) -> EpisodePlan:
    rows = _rows_for_episode(scheduler, episode)
    events = [adapter._event_from_rollout_plan(rollout, event_idx=idx, protocol_name=protocol_name) for idx, rollout in enumerate(episode.rollouts)]
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
        rollouts_per_episode=len(tuple(episode.rollouts)) + 1,
        target_positions=final_positions,
        validation_render_only=True,
    )
    events.append(
        ProbeEvent(
            event_id=f"{protocol_name}_probe_before_repair",
            kind="render_probe",
            target_positions=tuple(final_positions),
            rollout_plan=probe_plan,
        )
    )
    base_positions = list(range(int(rows.shape[0])))
    for perm_idx in range(max(1, int(repair_permutations))):
        positions = list(base_positions)
        scheduler.rng.shuffle(positions)
        positions = positions[: min(6, len(positions))]
        repair_plan = _manual_stage2_3_plan(
            scheduler=scheduler,
            episode=episode,
            rows=rows,
            positions=positions,
            repeat_budgets=[1 for _ in positions],
            phase="repair",
            visit_kind="repair",
            rollout_idx=len(tuple(episode.rollouts)) + 1 + int(perm_idx),
            rollouts_per_episode=len(tuple(episode.rollouts)) + 1 + max(1, int(repair_permutations)),
            repair_round_idx=int(perm_idx),
            repair_pattern_name=str(pattern_name),
        )
        events.append(adapter._event_from_rollout_plan(repair_plan, event_idx=len(events), protocol_name=protocol_name))
    events.append(
        ProbeEvent(
            event_id=f"{protocol_name}_probe_after_repair",
            kind="render_probe",
            target_positions=tuple(final_positions),
            rollout_plan=probe_plan,
        )
    )
    return _plan_from_events(adapter, episode, protocol_name, tuple(events))


def _repeat_stability_plan(
    *,
    adapter: Stage3SchedulerAdapter,
    scheduler: Stage23Scheduler,
    episode: Any,
    protocol_name: str,
    repeats: Sequence[int],
) -> EpisodePlan:
    rows = _rows_for_episode(scheduler, episode)
    events = [adapter._event_from_rollout_plan(rollout, event_idx=idx, protocol_name=protocol_name) for idx, rollout in enumerate(episode.rollouts)]
    events.append(ControlEvent(event_id=f"{protocol_name}_snapshot_repeat_base", kind="snapshot_state", name="repeat_base"))
    target_pos = max(0, int(rows.shape[0]) - 1)
    for repeat_count in list(repeats or [8, 16]):
        events.append(ControlEvent(event_id=f"{protocol_name}_restore_R{int(repeat_count)}", kind="restore_state", name="repeat_base"))
        plan = _manual_stage2_3_plan(
            scheduler=scheduler,
            episode=episode,
            rows=rows,
            positions=[int(target_pos)],
            repeat_budgets=[int(repeat_count)],
            phase="repeat_stability",
            visit_kind="repeat_stability",
            rollout_idx=len(tuple(episode.rollouts)) + int(repeat_count),
            rollouts_per_episode=len(tuple(episode.rollouts)) + 1,
        )
        events.append(adapter._event_from_rollout_plan(plan, event_idx=len(events), protocol_name=protocol_name))
    return _plan_from_events(adapter, episode, protocol_name, tuple(events))


def _plan_from_events(adapter: Stage3SchedulerAdapter, episode: Any, protocol_name: str, events: tuple[Any, ...]) -> EpisodePlan:
    base = adapter.plan_from_episode_v3(episode, protocol_name, metadata={"protocol": protocol_name})
    return EpisodePlan(
        plan_id="",
        version=base.version,
        episode=base.episode,
        events=events,
        expected_outputs=base.expected_outputs,
        deterministic=True,
        source="validation_recipe",
        created_at_step=-1,
        metadata=dict(base.metadata),
    ).with_stable_plan_id()


__all__ = ["FrameSetSpec", "build_validation_v4_plans", "iforward_validation_v4_cfg"]
