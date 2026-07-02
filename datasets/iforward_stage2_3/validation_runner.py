from __future__ import annotations

import dataclasses
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .scheduler import Stage23Scheduler
from .schema import STAGE23_CURRENT_ROLE, make_final_supervision_v3


DEFAULT_STAGE23_VALIDATION_PROTOCOLS = [
    "Assimilation-Causal",
    "Assimilation-Causal-FinalAll",
    "Repair-B6R1",
    "Repair-B8R1",
    "Repair-B6R2",
    "Repair-B10",
    "Repeat Stability",
    "Order Robustness",
]

DEFAULT_STAGE23_MAMBA_ABLATION_MODES = [
    "full",
    "mamba_off",
    "mamba_read_only",
    "mamba_read_write",
    "mamba_shuffle_state",
    "mamba_freeze_write",
]


def _mode_name(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _mamba_ablation_mode_name(value: Any) -> str:
    name = _mode_name(value)
    aliases = {
        "off": "mamba_off",
        "read_only": "mamba_read_only",
        "read_write": "mamba_read_write",
        "shuffled": "mamba_shuffle_state",
        "shuffle_memory": "mamba_shuffle_state",
        "bypass_memory": "mamba_off",
        "freeze_write": "mamba_freeze_write",
    }
    return str(aliases.get(name, name))


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        out = node.get(key, default)
        return default if out is None else out
    if hasattr(node, key):
        out = getattr(node, key)
        return default if out is None else out
    return default


def _emit_status(status_writer: Optional[Callable[[Dict[str, Any]], None]], row: Dict[str, Any]) -> None:
    if callable(status_writer):
        status_writer(dict(row))


def _scheduler_key_version(cfg: Any) -> Tuple[str, str]:
    sched_cfg = _cfg_get(cfg, "scheduler_stage3_0", None)
    if sched_cfg is not None and bool(_cfg_get(sched_cfg, "enable", False)):
        return "scheduler_stage3_0", str(_cfg_get(sched_cfg, "version", ""))
    sched_cfg = _cfg_get(cfg, "scheduler_v3", {}) or {}
    return "scheduler_v3", str(_cfg_get(sched_cfg, "version", ""))


def stage2_3_validation_cfg(cfg: Any) -> Dict[str, Any]:
    stage3_raw = _cfg_get(cfg, "scheduler_stage3_0_validation", None)
    legacy_raw = _cfg_get(cfg, "validation_v3", None)
    if stage3_raw is not None and legacy_raw is not None:
        if bool(_cfg_get(stage3_raw, "enable", False)) and bool(_cfg_get(legacy_raw, "enable", False)):
            raise ValueError(
                "scheduler_stage3_0_validation and validation_v3 must not both be enabled"
            )
    raw = (
        stage3_raw
        if stage3_raw is not None and bool(_cfg_get(stage3_raw, "enable", False))
        else (legacy_raw or stage3_raw or {})
    )
    protocols_raw = _cfg_get(raw, "protocols", DEFAULT_STAGE23_VALIDATION_PROTOCOLS)
    protocols_is_map = hasattr(protocols_raw, "items")
    if isinstance(protocols_raw, str):
        protocols = [protocols_raw]
    elif protocols_is_map:
        protocols = []
        if bool(protocols_raw.get("assimilation", False)):
            protocols.append("Assimilation-Causal")
            protocols.append("Assimilation-Causal-FinalAll")
        if protocols_raw.get("repair_permutations", None) is not None:
            protocols.extend(["Repair-B6R1", "Repair-B8R1", "Repair-B6R2", "Repair-B10", "Order Robustness"])
        if protocols_raw.get("repeat_stability", None) is not None:
            protocols.append("Repeat Stability")
        if protocols_raw.get("mamba_ablation", None) is not None:
            protocols.append("Mamba Ablation")
        if not protocols:
            protocols = list(DEFAULT_STAGE23_VALIDATION_PROTOCOLS)
    else:
        protocols = [str(x) for x in list(protocols_raw or DEFAULT_STAGE23_VALIDATION_PROTOCOLS)]
    modes_raw = _cfg_get(raw, "modes", ["full"])
    modes = [_mode_name(x) for x in list(modes_raw if not isinstance(modes_raw, str) else [modes_raw]) or ["full"]]
    ablation_cfg = _cfg_get(raw, "mamba_ablation", {}) or {}
    protocol_ablation_modes = protocols_raw.get("mamba_ablation", None) if protocols_is_map else None
    if protocol_ablation_modes is not None and not isinstance(protocol_ablation_modes, bool):
        ablation_modes = [_mamba_ablation_mode_name(x) for x in list(protocol_ablation_modes or [])]
    else:
        ablation_modes = [
            _mamba_ablation_mode_name(x)
            for x in list(_cfg_get(ablation_cfg, "modes", DEFAULT_STAGE23_MAMBA_ABLATION_MODES) or [])
        ]
    repeat_raw = _cfg_get(raw, "repeat_stability_repeats", None)
    if repeat_raw is None and protocols_is_map and protocols_raw.get("repeat_stability", None) is not None:
        repeat_raw = protocols_raw.get("repeat_stability")
    repair_perm_raw = _cfg_get(raw, "order_robustness_permutations", None)
    if repair_perm_raw is None and protocols_is_map and protocols_raw.get("repair_permutations", None) is not None:
        repair_perm_raw = protocols_raw.get("repair_permutations")
    if repeat_raw is None or isinstance(repeat_raw, bool):
        repeat_values = [4, 8, 16, 32]
    elif isinstance(repeat_raw, (int, float)):
        repeat_values = [int(repeat_raw)]
    else:
        repeat_values = list(repeat_raw or [4, 8, 16, 32])
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "run_at_train_start": bool(_cfg_get(raw, "run_at_train_start", True)),
        "interval_steps": int(_cfg_get(raw, "interval_steps", 20000)),
        "max_entries": int(_cfg_get(raw, "max_entries", 4)),
        "protocols": protocols,
        "seed": int(_cfg_get(raw, "seed", 0)),
        "modes": modes,
        "repeat_stability_repeats": [int(x) for x in repeat_values],
        "repeat_stability_position": int(_cfg_get(raw, "repeat_stability_position", -1)),
        "order_robustness_permutations": int(repair_perm_raw if repair_perm_raw is not None else 3),
        "mamba_ablation_modes": ablation_modes,
        "tensorboard_images": dict(_cfg_get(raw, "tensorboard_images", {}) or {}),
    }


def build_stage2_3_validation_manifest(*, scheduler: Stage23Scheduler, max_entries: int = 4) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    state = scheduler.state_dict()
    try:
        scheduler.global_step = max(int(scheduler.global_step), int(_cfg_get(scheduler.bootstrap_cfg, "end_step", 5000)))
        for _ in range(int(max_entries)):
            batch = scheduler.next_batch()
            meta = dict(batch.get("_iforward", {}) or {})
            entries.append(
                {
                    "scene_id": int(meta.get("scene_id", -1)),
                    "segment_id": int(meta.get("segment_id", -1)),
                    "sequence_id": int(meta.get("sequence_id", -1)),
                    "scheduler_phase": str(meta.get("scheduler_phase", "")),
                    "sequence_positions": [int(x) for x in list(meta.get("sequence_positions", []) or [])],
                    "episode_positions": [int(x) for x in list(meta.get("episode_positions", []) or [])],
                    "rollout_positions": [int(x) for x in list(meta.get("rollout_positions", []) or [])],
                }
            )
    finally:
        scheduler.load_state_dict(state)
    if not entries:
        raise ValueError("Stage2_3 validation manifest is empty")
    return entries


def run_stage2_3_validation_manifest_only(*, cfg: Any, dataset: Any, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
    sched_cfg = _cfg_get(cfg, "scheduler_stage3_0", None)
    if sched_cfg is None or not bool(_cfg_get(sched_cfg, "enable", False)):
        sched_cfg = _cfg_get(cfg, "scheduler_v3", {}) or {}
    producer_cfg = dict(_cfg_get(sched_cfg, "producer", {}) or {})
    producer_cfg["enable"] = False
    scheduler = Stage23Scheduler(dataset=dataset, cfg=cfg, producer_cfg=producer_cfg, fail_fast=False)
    val = stage2_3_validation_cfg(cfg)
    limit = int(max_entries if max_entries is not None else val.get("max_entries", 4))
    return build_stage2_3_validation_manifest(scheduler=scheduler, max_entries=limit)


def _detach_next_state(out: Any) -> Any:
    state = getattr(out, "next_state", None)
    return _clone_state_for_validation(state)


def _clone_state_for_validation(state: Any) -> Any:
    detach = getattr(state, "detach_for_next_rollout", None)
    return detach() if callable(detach) else state


def _should_carry(out: Any) -> bool:
    resolved = getattr(out, "resolved", None)
    return bool(getattr(resolved, "carry_scene_state_after_rollout", False)) and not bool(
        getattr(resolved, "episode_end_after_rollout", False)
    )


def _run_plan(
    *,
    scheduler: Stage23Scheduler,
    plan: Any,
    model: Any,
    carried_state: Any,
    mode: str,
    device: torch.device,
    trigger_step: int,
    convert_batch_to_minimal_format: Optional[Callable[[Dict[str, Any], torch.device, int], Dict[str, Any]]],
) -> Any:
    raw = scheduler._batch_from_plan(plan)
    if isinstance(raw.get("_iforward"), dict):
        raw["_iforward"]["validation_force_history_render"] = True
    batch = (
        convert_batch_to_minimal_format(raw, device, int(trigger_step))
        if callable(convert_batch_to_minimal_format)
        else raw
    )
    return model.forward_rollout(batch, carried_state=carried_state, ablation=str(mode))


def _scalar_tensor_value(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item()) if int(value.numel()) else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _row_from_output(*, out: Any, protocol: str, rollout_idx: int, mode: str, trigger_step: int) -> Dict[str, Any]:
    stats = dict(getattr(out, "stats", {}) or {})
    losses = dict(getattr(out, "losses", {}) or {})
    resolved = getattr(out, "resolved", None)
    meta = dict(getattr(resolved, "meta", {}) or {}) if resolved is not None else {}
    request_meta = dict(meta.get("request_meta", {}) or {})
    stage23_meta = dict(request_meta.get("iforward_stage2_3", {}) or {})
    stage32_meta = dict(request_meta.get("iforward_stage3_2", {}) or {})
    row = {
        "step": int(trigger_step),
        "trigger_step": int(trigger_step),
        "split": "iforward_stage2_3_validation",
        "protocol": str(protocol),
        "mode": str(mode),
        "scene_id": int(getattr(resolved, "scene_id", meta.get("scene_id", -1))) if resolved is not None else -1,
        "segment_id": int(getattr(resolved, "segment_id", meta.get("segment_id", -1))) if resolved is not None else -1,
        "rollout_idx": int(rollout_idx),
        "scheduler_phase": str(meta.get("scheduler_phase", "")),
        "rollout_phase": str(meta.get("rollout_phase", "")),
        "sequence_id": int(meta.get("sequence_id", -1)),
        "sequence_length": int(meta.get("sequence_length", 0)),
        "rollout_positions": [int(x) for x in list(meta.get("rollout_positions", meta.get("sequence_positions", [])) or [])],
        "history_positions": [int(x) for x in list(meta.get("history_positions", []) or [])],
        "repair_positions": [int(x) for x in list(meta.get("repair_positions", []) or [])],
        "final_all_positions": [int(x) for x in list(stage23_meta.get("final_all_positions", []) or [])],
        "final_all_frame_indices": [int(x) for x in list(stage23_meta.get("final_all_frame_indices", []) or [])],
        "repeat_budgets": [int(x) for x in list(meta.get("repeat_budgets", []) or [])],
        "frame_gaps": [int(x) for x in list(meta.get("frame_gaps", []) or [])],
        "repair_round_idx": int(meta.get("repair_round_idx", -1)),
        "repair_pattern_name": str(meta.get("repair_pattern_name", "")),
        "loss": _scalar_tensor_value(getattr(out, "loss", 0.0)),
        "current_loss": _scalar_tensor_value(losses.get("current", 0.0)),
        "history_loss": _scalar_tensor_value(losses.get("in_rollout_history", 0.0)),
        "history_damage": _scalar_tensor_value(losses.get("history_damage", 0.0)),
        "current_psnr": float(stats.get("current_psnr", 0.0)),
        "history_rollout_psnr": float(stats.get("history_rollout_psnr", 0.0)),
        "stage2_3_best_damage_loss": float(stats.get("stage2_3/best_damage_loss", 0.0)),
        "stage2_3_best_damage_p90": float(stats.get("stage2_3/best_damage_p90", 0.0)),
        "stage2_3_best_damage_max": float(stats.get("stage2_3/best_damage_max", 0.0)),
        "stage2_3_bank_valid_count": float(stats.get("stage2_3/bank_valid_count", 0.0)),
        "stage2_3_bank_update_count": float(stats.get("stage2_3/bank_update_count", 0.0)),
    }
    if stage32_meta:
        row.update(
            {
                "distribution_type": str(stage32_meta.get("distribution_type", "")),
                "distribution_type_id": int(stage32_meta.get("distribution_type_id", 0) or 0),
                "episode_stage": str(stage32_meta.get("episode_stage", "")),
                "episode_stage_id": int(stage32_meta.get("episode_stage_id", 0) or 0),
                "order_type": str(stage32_meta.get("order_type", "")),
                "order_type_id": int(stage32_meta.get("order_type_id", 0) or 0),
                "train_2d_mode": str(stage32_meta.get("train_2d_mode", "")),
                "train_2d_mode_id": int(stage32_meta.get("train_2d_mode_id", 0) or 0),
                "stage3_2_B": int(stage32_meta.get("B", 0) or 0),
                "stage3_2_K": int(stage32_meta.get("K", 0) or 0),
                "stage3_2_maxK": int(stage32_meta.get("maxK", 0) or 0),
                "stage3_2_R_mean": float(stage32_meta.get("R_mean", 0.0) or 0.0),
                "visited_ratio_before": float(stage32_meta.get("visited_ratio_before", 0.0) or 0.0),
                "visited_ratio_after": float(stage32_meta.get("visited_ratio_after", 0.0) or 0.0),
                "repair_visited_ratio": float(stage32_meta.get("repair_visited_ratio", 0.0) or 0.0),
                "curriculum_phase_name": str(stage32_meta.get("curriculum_phase_name", "")),
                "curriculum_phase_id": int(stage32_meta.get("curriculum_phase_id", 0) or 0),
            }
        )
    return row


def _rows_for_episode(scheduler: Stage23Scheduler, episode: Any) -> np.ndarray:
    segment_row = int((getattr(episode, "metadata", {}) or {}).get("segment_row", -1))
    if segment_row < 0:
        raise ValueError("Stage2_3 validation episode missing segment_row metadata")
    all_rows = scheduler.index.frames_for_segment_row(segment_row)
    by_frame = {int(row["frame_idx"]): row for row in all_rows}
    selected = [by_frame[int(frame)] for frame in tuple(getattr(episode, "frame_set", ()))]
    return np.asarray(selected, dtype=all_rows.dtype)


def _visit_state_after_plans(plans: Sequence[Any], rows: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int], Dict[str, Any], int]:
    visit_counts: Dict[int, int] = {}
    last_visit_step_by_pos: Dict[int, int] = {}
    last_visit_context: Dict[str, Any] = {}
    step_offset = 0
    for plan in plans:
        for step in list(getattr(plan, "steps", []) or []):
            pos = int(getattr(step, "sequence_pos", -1))
            if pos < 0 or pos >= int(rows.shape[0]):
                continue
            global_idx = int(getattr(step, "global_update_idx_in_episode", step_offset))
            if int(getattr(step, "repeat_idx", 0)) == 0:
                visit_counts[pos] = int(visit_counts.get(pos, 0)) + 1
            last_visit_step_by_pos[pos] = int(global_idx)
            row = rows[pos]
            last_visit_context = {
                "sequence_pos": int(pos),
                "frame_idx": int(row["frame_idx"]),
                "timestamp_us": int(row["timestamp_us"]),
                "ego_translation": np.asarray(row["ego_translation"], dtype=np.float32),
                "ego_yaw": float(row["ego_yaw"]),
                "global_update_idx": int(global_idx),
            }
            step_offset = max(step_offset, int(global_idx) + 1)
    return visit_counts, last_visit_step_by_pos, last_visit_context, int(step_offset)


def _manual_stage2_3_plan(
    *,
    scheduler: Stage23Scheduler,
    episode: Any,
    rows: np.ndarray,
    positions: Sequence[int],
    repeat_budgets: Sequence[int],
    phase: str,
    visit_kind: str,
    rollout_idx: int,
    rollouts_per_episode: int,
    repair_round_idx: int = -1,
    repair_pattern_name: str = "",
    target_positions: Optional[Sequence[int]] = None,
    validation_render_only: bool = False,
) -> Any:
    visit_counts, last_visit_step_by_pos, last_visit_context, step_offset = _visit_state_after_plans(
        list(getattr(episode, "rollouts", []) or []),
        rows,
    )
    selected = [int(x) for x in positions]
    history = [p for p in range(int(rows.shape[0])) if int(p) not in set(selected)]
    plan = scheduler._rollout_from_positions(
        rows=rows,
        scene_id=int(getattr(episode, "scene_id")),
        segment_id=int(getattr(episode, "segment_id")),
        sequence_id=int(getattr(episode, "sequence_id")),
        positions=selected,
        repeat_budgets=[int(x) for x in repeat_budgets],
        rollout_idx=int(rollout_idx),
        rollouts_per_episode=int(rollouts_per_episode),
        phase=str(phase),
        visit_kind=str(visit_kind),
        history_positions=history,
        repair_positions=selected if str(phase) == "repair" else [],
        repair_enabled=bool(str(phase) == "repair"),
        repair_hash=int(sum((idx + 1) * (pos + 1) for idx, pos in enumerate(selected))) if str(phase) == "repair" else -1,
        episode_step_offset=int(step_offset),
        visit_counts=dict(visit_counts),
        last_visit_step_by_pos=dict(last_visit_step_by_pos),
        is_last_rollout=True,
        last_visit_context=dict(last_visit_context),
        repair_round_idx=int(repair_round_idx),
        repair_pattern_name=str(repair_pattern_name),
    )
    if bool(validation_render_only):
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
        plan = dataclasses.replace(
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
        )
    if target_positions is not None:
        target_positions = [int(x) for x in target_positions]
        target_frames, target_refs = scheduler._ref_rows_for_positions(rows, target_positions)
        final = make_final_supervision_v3(
            refs=list(target_refs),
            roles=[STAGE23_CURRENT_ROLE for _ in target_refs],
            current_frames=list(target_frames),
            current_refs=list(target_refs),
            history_frames=[],
            history_refs=[],
        )
        request_meta = dict(getattr(plan, "request_meta", {}) or {})
        nested = dict(request_meta.get("iforward_stage2_3", {}) or {})
        nested["final_all_positions"] = list(target_positions)
        nested["final_all_frame_indices"] = list(target_frames)
        request_meta["iforward_stage2_3"] = nested
        plan = dataclasses.replace(
            plan,
            input_frame_indices=list(target_frames),
            delivery_frame_indices=list(target_frames),
            final_supervision=final,
            target_refs_flat=list(target_refs),
            target_roles_flat=[STAGE23_CURRENT_ROLE for _ in target_refs],
            request_meta=request_meta,
        )
    return plan


def _run_causal_episode(
    *,
    scheduler: Stage23Scheduler,
    episode: Any,
    model: Any,
    protocol: str,
    mode: str,
    device: torch.device,
    trigger_step: int,
    rows_out: List[Dict[str, Any]],
    convert_batch_to_minimal_format: Optional[Callable[[Dict[str, Any], torch.device, int], Dict[str, Any]]],
    writer: Optional[Any] = None,
    val_cfg: Optional[Dict[str, Any]] = None,
    tb_counters: Optional[Dict[str, int]] = None,
) -> Tuple[Any, List[Any]]:
    carried_state = None
    outs: List[Any] = []
    for rollout_idx, plan in enumerate(list(getattr(episode, "rollouts", []) or [])):
        out = _run_plan(
            scheduler=scheduler,
            plan=plan,
            model=model,
            carried_state=carried_state,
            mode=str(mode),
            device=device,
            trigger_step=int(trigger_step),
            convert_batch_to_minimal_format=convert_batch_to_minimal_format,
        )
        _maybe_write_tb_images(
            writer=writer,
            out=out,
            protocol=str(protocol),
            mode=str(mode),
            rollout_idx=int(rollout_idx),
            trigger_step=int(trigger_step),
            val_cfg=dict(val_cfg or {}),
            counters=tb_counters if tb_counters is not None else {},
        )
        rows_out.append(_row_from_output(out=out, protocol=str(protocol), rollout_idx=int(rollout_idx), mode=str(mode), trigger_step=int(trigger_step)))
        outs.append(out)
        carried_state = _detach_next_state(out) if _should_carry(out) else None
    if outs:
        carried_state = _detach_next_state(outs[-1])
    return carried_state, outs


def _repair_shape(protocol: str, sequence_length: int) -> Tuple[int, int, str]:
    text = str(protocol).replace("Repair-", "").upper()
    if text == "B10":
        return int(sequence_length), 1, "B10"
    if "R" in text:
        left, right = text.split("R", 1)
        return min(int(left.replace("B", "")), int(sequence_length)), max(1, int(right)), text
    return min(6, int(sequence_length)), 1, "B6R1"


def _append_summary(rows: List[Dict[str, Any]], *, protocol: str, mode: str, trigger_step: int, kind: str, source_rows: Sequence[Dict[str, Any]]) -> None:
    if not source_rows:
        return
    psnrs = [float(row.get("current_psnr", 0.0)) for row in source_rows]
    losses = [float(row.get("current_loss", 0.0)) for row in source_rows]
    rows.append(
        {
            "step": int(trigger_step),
            "trigger_step": int(trigger_step),
            "split": "iforward_stage2_3_validation",
            "protocol": str(protocol),
            "mode": str(mode),
            "rollout_idx": -1,
            "validation_rollout_kind": str(kind),
            "retention_auc": float(sum(psnrs) / max(1, len(psnrs))),
            "repair_mean": float(sum(psnrs) / max(1, len(psnrs))),
            "repair_worst": float(min(psnrs)) if psnrs else 0.0,
            "repair_mean_loss": float(sum(losses) / max(1, len(losses))),
            "permutation_std": float(statistics.pstdev(psnrs)) if len(psnrs) > 1 else 0.0,
        }
    )


def _retention_value(row: Dict[str, Any]) -> Tuple[float, float, bool]:
    history_positions = list(row.get("history_positions", []) or [])
    final_all_positions = list(row.get("final_all_positions", []) or [])
    has_history = bool(history_positions or final_all_positions or str(row.get("scheduler_phase", "")) == "final_all")
    psnr = float(row.get("history_rollout_psnr", 0.0))
    if not np.isfinite(psnr) or psnr <= 0.0:
        psnr = float(row.get("current_psnr", 0.0)) if bool(final_all_positions) else 0.0
    loss = float(row.get("history_loss", 0.0))
    if (not np.isfinite(loss) or loss <= 0.0) and bool(final_all_positions):
        loss = float(row.get("current_loss", 0.0))
    valid = bool(has_history and np.isfinite(psnr) and psnr > 0.0)
    return psnr, loss, valid


def _append_retention_curve(
    rows: List[Dict[str, Any]],
    *,
    protocol: str,
    mode: str,
    trigger_step: int,
    source_rows: Sequence[Dict[str, Any]],
) -> None:
    points: List[Dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        psnr, loss, valid = _retention_value(dict(row))
        if not valid:
            continue
        point = {
            "step": int(trigger_step),
            "trigger_step": int(trigger_step),
            "split": "iforward_stage2_3_validation",
            "protocol": str(protocol),
            "mode": str(mode),
            "rollout_idx": int(row.get("rollout_idx", idx)),
            "scheduler_phase": str(row.get("scheduler_phase", "")),
            "validation_rollout_kind": "retention_curve_point",
            "retention_curve_idx": int(len(points)),
            "retention_seen_count": int(len(list(row.get("history_positions", []) or []))),
            "retention_eval_count": int(
                len(list(row.get("final_all_positions", []) or []))
                or len(list(row.get("history_positions", []) or []))
            ),
            "retention_psnr": float(psnr),
            "retention_loss": float(loss),
            "retention_auc": float(psnr),
        }
        points.append(point)
    rows.extend(points)
    if not points:
        return
    psnrs = [float(point["retention_psnr"]) for point in points]
    losses = [float(point["retention_loss"]) for point in points]
    rows.append(
        {
            "step": int(trigger_step),
            "trigger_step": int(trigger_step),
            "split": "iforward_stage2_3_validation",
            "protocol": str(protocol),
            "mode": str(mode),
            "rollout_idx": -1,
            "validation_rollout_kind": "retention_curve_summary",
            "retention_points": int(len(points)),
            "retention_auc": float(sum(psnrs) / max(1, len(psnrs))),
            "retention_worst": float(min(psnrs)),
            "retention_loss_mean": float(sum(losses) / max(1, len(losses))),
        }
    )


def _image_chw(value: Any) -> Optional[torch.Tensor]:
    if not torch.is_tensor(value):
        return None
    image = value.detach().float().cpu()
    while int(image.ndim) > 3:
        image = image[0]
    if int(image.ndim) != 3:
        return None
    if int(image.shape[0]) in {1, 3}:
        out = image
    elif int(image.shape[-1]) in {1, 3}:
        out = image.permute(2, 0, 1)
    else:
        return None
    return out.clamp(0.0, 1.0)


def _maybe_write_tb_images(
    *,
    writer: Optional[Any],
    out: Any,
    protocol: str,
    mode: str,
    rollout_idx: int,
    trigger_step: int,
    val_cfg: Dict[str, Any],
    counters: Dict[str, int],
) -> None:
    if writer is None:
        return
    tb_cfg = dict(val_cfg.get("tensorboard_images", {}) or {})
    if not bool(tb_cfg.get("enable", True)):
        return
    add_image = getattr(writer, "add_image", None)
    if not callable(add_image):
        return
    max_per_role = int(tb_cfg.get("max_images_per_role", 2))
    pred_rgbs = list(getattr(out, "pred_rgbs", []) or [])
    gt_images = list(getattr(out, "gt_images", []) or [])
    roles = [str(x) for x in list(getattr(out, "image_roles", []) or [])]
    for idx, (pred_raw, gt_raw) in enumerate(zip(pred_rgbs, gt_images)):
        role = roles[idx] if idx < len(roles) else "target"
        key = f"{protocol}/{mode}/{role}"
        count = int(counters.get(key, 0))
        if count >= int(max_per_role):
            continue
        pred = _image_chw(pred_raw)
        gt = _image_chw(gt_raw)
        if pred is None or gt is None:
            continue
        if pred.shape[-2:] != gt.shape[-2:]:
            continue
        panel = torch.cat([gt, pred], dim=-1)
        tag = f"iforward_stage2_3_validation/{protocol}/{mode}/{role}/rollout_{int(rollout_idx)}_{count}"
        add_image(tag, panel, int(trigger_step))
        counters[key] = count + 1


def _make_scheduler(cfg: Any, dataset: Any, *, seed: int) -> Stage23Scheduler:
    sched_cfg = _cfg_get(cfg, "scheduler_stage3_0", None)
    if sched_cfg is None or not bool(_cfg_get(sched_cfg, "enable", False)):
        sched_cfg = _cfg_get(cfg, "scheduler_v3", {}) or {}
    repair_cfg = dict(_cfg_get(sched_cfg, "repair", {}) or {})
    repair_cfg["enable"] = False
    producer_cfg = dict(_cfg_get(sched_cfg, "producer", {}) or {})
    producer_cfg["enable"] = False
    scheduler = Stage23Scheduler(
        dataset=dataset,
        cfg=cfg,
        repair_cfg=repair_cfg,
        producer_cfg=producer_cfg,
        seed=int(seed),
        fail_fast=False,
    )
    scheduler.global_step = max(int(scheduler.global_step), int(_cfg_get(scheduler.bootstrap_cfg, "end_step", 5000)))
    return scheduler


def run_stage2_3_validation(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: Any,
    trigger_step: int = 0,
    modes: Optional[List[str]] = None,
    convert_batch_to_minimal_format: Optional[Callable[[Dict[str, Any], torch.device, int], Dict[str, Any]]] = None,
    writer: Optional[Any] = None,
    status_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    val = stage2_3_validation_cfg(cfg)
    protocols = list(val.get("protocols", DEFAULT_STAGE23_VALIDATION_PROTOCOLS))
    base_modes = list(modes or val.get("modes", ["full"]) or ["full"])
    scheduler_key, scheduler_version = _scheduler_key_version(cfg)
    planned_protocol_count = 0
    for protocol in protocols:
        planned_protocol_count += len(
            list(val.get("mamba_ablation_modes", DEFAULT_STAGE23_MAMBA_ABLATION_MODES))
            if str(protocol) == "Mamba Ablation"
            else base_modes
        )
    _emit_status(
        status_writer,
        {
            "status": "manifest_built",
            "max_entries": int(val.get("max_entries", 4)),
            "scheduler_key": str(scheduler_key),
            "scheduler_version": str(scheduler_version),
            "planned_protocol_count": int(planned_protocol_count),
            "protocols": [str(x) for x in protocols],
            "modes": [str(x) for x in base_modes],
        },
    )
    device_obj = torch.device(device)
    was_training = bool(getattr(model, "training", False))
    rows: List[Dict[str, Any]] = []
    tb_counters: Dict[str, int] = {}
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            for entry_idx in range(int(val.get("max_entries", 4))):
                for protocol in protocols:
                    protocol_modes = list(base_modes)
                    if str(protocol) == "Mamba Ablation":
                        protocol_modes = list(val.get("mamba_ablation_modes", DEFAULT_STAGE23_MAMBA_ABLATION_MODES))
                    for mode in protocol_modes:
                        protocol_rows_before = int(len(rows))
                        _emit_status(
                            status_writer,
                            {
                                "status": "protocol_start",
                                "entry_idx": int(entry_idx),
                                "protocol": str(protocol),
                                "mode": str(mode),
                            },
                        )
                        scheduler = _make_scheduler(cfg, dataset, seed=int(val.get("seed", 0)) + int(entry_idx))
                        episode = scheduler._build_episode()
                        frame_rows = _rows_for_episode(scheduler, episode)
                        if str(protocol) in {"Assimilation-Causal", "Mamba Ablation"}:
                            causal_rows: List[Dict[str, Any]] = []
                            _run_causal_episode(
                                scheduler=scheduler,
                                episode=episode,
                                model=model,
                                protocol=str(protocol),
                                mode=str(mode),
                                device=device_obj,
                                trigger_step=int(trigger_step),
                                rows_out=causal_rows,
                                convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                                writer=writer,
                                val_cfg=val,
                                tb_counters=tb_counters,
                            )
                            rows.extend(causal_rows)
                            _append_summary(rows, protocol=str(protocol), mode=str(mode), trigger_step=int(trigger_step), kind="causal_summary", source_rows=causal_rows)
                            _append_retention_curve(rows, protocol=str(protocol), mode=str(mode), trigger_step=int(trigger_step), source_rows=causal_rows)
                            _emit_status(
                                status_writer,
                                {
                                    "status": "protocol_done",
                                    "entry_idx": int(entry_idx),
                                    "protocol": str(protocol),
                                    "mode": str(mode),
                                    "rows_emitted": int(len(rows) - protocol_rows_before),
                                },
                            )
                            continue

                        if str(protocol) == "Assimilation-Causal-FinalAll":
                            causal_rows = []
                            causal_state, _ = _run_causal_episode(
                                scheduler=scheduler,
                                episode=episode,
                                model=model,
                                protocol=str(protocol),
                                mode=str(mode),
                                device=device_obj,
                                trigger_step=int(trigger_step),
                                rows_out=causal_rows,
                                convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                                writer=writer,
                                val_cfg=val,
                                tb_counters=tb_counters,
                            )
                            rows.extend(causal_rows)
                            final_positions = list(range(int(frame_rows.shape[0])))
                            final_source_pos = final_positions[-1] if final_positions else 0
                            plan = _manual_stage2_3_plan(
                                scheduler=scheduler,
                                episode=episode,
                                rows=frame_rows,
                                positions=[int(final_source_pos)],
                                repeat_budgets=[1],
                                phase="final_all",
                                visit_kind="final_all",
                                rollout_idx=len(getattr(episode, "rollouts", []) or []),
                                rollouts_per_episode=len(getattr(episode, "rollouts", []) or []) + 1,
                                target_positions=final_positions,
                                validation_render_only=True,
                            )
                            out = _run_plan(
                                scheduler=scheduler,
                                plan=plan,
                                model=model,
                                carried_state=_clone_state_for_validation(causal_state),
                                mode=str(mode),
                                device=device_obj,
                                trigger_step=int(trigger_step),
                                convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                            )
                            _maybe_write_tb_images(
                                writer=writer,
                                out=out,
                                protocol=str(protocol),
                                mode=str(mode),
                                rollout_idx=int(plan.rollout_idx_in_episode),
                                trigger_step=int(trigger_step),
                                val_cfg=val,
                                counters=tb_counters,
                            )
                            row = _row_from_output(
                                out=out,
                                protocol=str(protocol),
                                rollout_idx=int(plan.rollout_idx_in_episode),
                                mode=str(mode),
                                trigger_step=int(trigger_step),
                            )
                            row.update(
                                {
                                    "validation_rollout_kind": "final_all",
                                    "retention_auc": float(row.get("current_psnr", 0.0)),
                                    "final_all_eval_count": float(len(final_positions)),
                                }
                            )
                            rows.append(row)
                            _append_summary(
                                rows,
                                protocol=str(protocol),
                                mode=str(mode),
                                trigger_step=int(trigger_step),
                                kind="final_all_summary",
                                source_rows=[row],
                            )
                            _append_retention_curve(
                                rows,
                                protocol=str(protocol),
                                mode=str(mode),
                                trigger_step=int(trigger_step),
                                source_rows=[*causal_rows, row],
                            )
                            _emit_status(
                                status_writer,
                                {
                                    "status": "protocol_done",
                                    "entry_idx": int(entry_idx),
                                    "protocol": str(protocol),
                                    "mode": str(mode),
                                    "rows_emitted": int(len(rows) - protocol_rows_before),
                                },
                            )
                            continue

                        causal_state, _ = _run_causal_episode(
                            scheduler=scheduler,
                            episode=episode,
                            model=model,
                            protocol=str(protocol),
                            mode=str(mode),
                            device=device_obj,
                            trigger_step=int(trigger_step),
                            rows_out=rows,
                            convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                            writer=writer,
                            val_cfg=val,
                            tb_counters=tb_counters,
                        )

                        if str(protocol).startswith("Repair-"):
                            frames, repeats, pattern_name = _repair_shape(str(protocol), int(frame_rows.shape[0]))
                            repair_rows: List[Dict[str, Any]] = []
                            permutations = max(1, int(val.get("order_robustness_permutations", 3)))
                            base_positions = list(range(int(frame_rows.shape[0])))
                            for perm_idx in range(int(permutations)):
                                positions = list(base_positions)
                                scheduler.rng.shuffle(positions)
                                positions = positions[: int(frames)]
                                plan = _manual_stage2_3_plan(
                                    scheduler=scheduler,
                                    episode=episode,
                                    rows=frame_rows,
                                    positions=positions,
                                    repeat_budgets=[int(repeats) for _ in positions],
                                    phase="repair",
                                    visit_kind="repair",
                                    rollout_idx=len(getattr(episode, "rollouts", []) or []) + int(perm_idx),
                                    rollouts_per_episode=len(getattr(episode, "rollouts", []) or []) + int(permutations),
                                    repair_round_idx=int(perm_idx),
                                    repair_pattern_name=str(pattern_name),
                                )
                                out = _run_plan(
                                    scheduler=scheduler,
                                    plan=plan,
                                    model=model,
                                    carried_state=_clone_state_for_validation(causal_state),
                                    mode=str(mode),
                                    device=device_obj,
                                    trigger_step=int(trigger_step),
                                    convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                                )
                                _maybe_write_tb_images(
                                    writer=writer,
                                    out=out,
                                    protocol=str(protocol),
                                    mode=str(mode),
                                    rollout_idx=int(plan.rollout_idx_in_episode),
                                    trigger_step=int(trigger_step),
                                    val_cfg=val,
                                    counters=tb_counters,
                                )
                                row = _row_from_output(
                                    out=out,
                                    protocol=str(protocol),
                                    rollout_idx=int(plan.rollout_idx_in_episode),
                                    mode=str(mode),
                                    trigger_step=int(trigger_step),
                                )
                                row.update(
                                    {
                                        "validation_rollout_kind": f"repair_{str(pattern_name).lower()}",
                                        "repair_perm_idx": int(perm_idx),
                                        "repair_positions": [int(x) for x in positions],
                                        "repair_visit_count": float(len(positions)),
                                        "repair_mean": float(row.get("current_psnr", 0.0)),
                                        "repair_worst": float(row.get("current_psnr", 0.0)),
                                        "retention_auc": float(row.get("history_rollout_psnr", row.get("current_psnr", 0.0))),
                                    }
                                )
                                rows.append(row)
                                repair_rows.append(row)
                            _append_summary(rows, protocol=str(protocol), mode=str(mode), trigger_step=int(trigger_step), kind="repair_summary", source_rows=repair_rows)
                            _append_retention_curve(
                                rows,
                                protocol=str(protocol),
                                mode=str(mode),
                                trigger_step=int(trigger_step),
                                source_rows=repair_rows,
                            )
                            _emit_status(
                                status_writer,
                                {
                                    "status": "protocol_done",
                                    "entry_idx": int(entry_idx),
                                    "protocol": str(protocol),
                                    "mode": str(mode),
                                    "rows_emitted": int(len(rows) - protocol_rows_before),
                                },
                            )
                            continue

                        if str(protocol) == "Repeat Stability":
                            target_pos = int(val.get("repeat_stability_position", -1))
                            if target_pos < 0:
                                target_pos = int(frame_rows.shape[0]) - 1
                            target_pos = max(0, min(int(frame_rows.shape[0]) - 1, int(target_pos)))
                            repeat_rows: List[Dict[str, Any]] = []
                            for repeat_count in list(val.get("repeat_stability_repeats", [4, 8, 16, 32])):
                                plan = _manual_stage2_3_plan(
                                    scheduler=scheduler,
                                    episode=episode,
                                    rows=frame_rows,
                                    positions=[int(target_pos)],
                                    repeat_budgets=[int(repeat_count)],
                                    phase="repeat_stability",
                                    visit_kind="repeat_stability",
                                    rollout_idx=len(getattr(episode, "rollouts", []) or []) + int(repeat_count),
                                    rollouts_per_episode=len(getattr(episode, "rollouts", []) or []) + 1,
                                )
                                out = _run_plan(
                                    scheduler=scheduler,
                                    plan=plan,
                                    model=model,
                                    carried_state=_clone_state_for_validation(causal_state),
                                    mode=str(mode),
                                    device=device_obj,
                                    trigger_step=int(trigger_step),
                                    convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                                )
                                _maybe_write_tb_images(
                                    writer=writer,
                                    out=out,
                                    protocol=str(protocol),
                                    mode=str(mode),
                                    rollout_idx=int(repeat_count),
                                    trigger_step=int(trigger_step),
                                    val_cfg=val,
                                    counters=tb_counters,
                                )
                                row = _row_from_output(out=out, protocol=str(protocol), rollout_idx=int(repeat_count), mode=str(mode), trigger_step=int(trigger_step))
                                row.update(
                                    {
                                        "validation_rollout_kind": "repeat_stability",
                                        "repeat_stability_repeats": int(repeat_count),
                                        "repeat_stability_position": int(target_pos),
                                    }
                                )
                                rows.append(row)
                                repeat_rows.append(row)
                            _append_summary(rows, protocol=str(protocol), mode=str(mode), trigger_step=int(trigger_step), kind="repeat_stability_summary", source_rows=repeat_rows)
                            _emit_status(
                                status_writer,
                                {
                                    "status": "protocol_done",
                                    "entry_idx": int(entry_idx),
                                    "protocol": str(protocol),
                                    "mode": str(mode),
                                    "rows_emitted": int(len(rows) - protocol_rows_before),
                                },
                            )
                            continue

                        if str(protocol) == "Order Robustness":
                            perm_rows: List[Dict[str, Any]] = []
                            base_positions = list(range(int(frame_rows.shape[0])))
                            for perm_idx in range(max(1, int(val.get("order_robustness_permutations", 3)))):
                                positions = list(base_positions)
                                scheduler.rng.shuffle(positions)
                                plan = _manual_stage2_3_plan(
                                    scheduler=scheduler,
                                    episode=episode,
                                    rows=frame_rows,
                                    positions=positions,
                                    repeat_budgets=[1 for _ in positions],
                                    phase="repair",
                                    visit_kind="repair",
                                    rollout_idx=len(getattr(episode, "rollouts", []) or []) + int(perm_idx),
                                    rollouts_per_episode=len(getattr(episode, "rollouts", []) or []) + int(perm_idx) + 1,
                                    repair_round_idx=int(perm_idx),
                                    repair_pattern_name="order_perm",
                                )
                                out = _run_plan(
                                    scheduler=scheduler,
                                    plan=plan,
                                    model=model,
                                    carried_state=_clone_state_for_validation(causal_state),
                                    mode=str(mode),
                                    device=device_obj,
                                    trigger_step=int(trigger_step),
                                    convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                                )
                                _maybe_write_tb_images(
                                    writer=writer,
                                    out=out,
                                    protocol=str(protocol),
                                    mode=str(mode),
                                    rollout_idx=int(perm_idx),
                                    trigger_step=int(trigger_step),
                                    val_cfg=val,
                                    counters=tb_counters,
                                )
                                row = _row_from_output(out=out, protocol=str(protocol), rollout_idx=int(perm_idx), mode=str(mode), trigger_step=int(trigger_step))
                                row.update(
                                    {
                                        "validation_rollout_kind": "order_robustness_repair",
                                        "order_perm_idx": int(perm_idx),
                                        "repair_positions": [int(x) for x in positions],
                                        "repair_visit_count": float(len(positions)),
                                    }
                                )
                                rows.append(row)
                                perm_rows.append(row)
                            _append_summary(rows, protocol=str(protocol), mode=str(mode), trigger_step=int(trigger_step), kind="order_robustness_summary", source_rows=perm_rows)
                        _emit_status(
                            status_writer,
                            {
                                "status": "protocol_done",
                                "entry_idx": int(entry_idx),
                                "protocol": str(protocol),
                                "mode": str(mode),
                                "rows_emitted": int(len(rows) - protocol_rows_before),
                            },
                        )
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    if not rows:
        raise ValueError("Stage2_3 validation produced no rows")
    return rows


__all__ = [
    "build_stage2_3_validation_manifest",
    "run_stage2_3_validation",
    "run_stage2_3_validation_manifest_only",
    "stage2_3_validation_cfg",
]
