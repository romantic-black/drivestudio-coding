from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .validation_manifest import STAGE22_VALIDATION_PROTOCOLS, load_or_build_stage2_2_validation_manifest
from .index_loader import load_stage2_2_index
from .scheduler import Stage22Scheduler


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


def stage2_2_validation_cfg(cfg: Any) -> Dict[str, Any]:
    raw = _cfg_get(cfg, "iforward_stage2_2_validation", {}) or {}
    protocols_raw = _cfg_get(raw, "protocols", STAGE22_VALIDATION_PROTOCOLS)
    if isinstance(protocols_raw, str):
        protocols = [protocols_raw]
    else:
        protocols = [str(x) for x in list(protocols_raw or STAGE22_VALIDATION_PROTOCOLS)]
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "run_at_train_start": bool(_cfg_get(raw, "run_at_train_start", True)),
        "interval_steps": int(_cfg_get(raw, "interval_steps", 20000)),
        "max_entries": int(_cfg_get(raw, "max_entries", 8)),
        "protocols": protocols,
        "manifest_path": str(_cfg_get(raw, "manifest_path", "")),
        "seed": int(_cfg_get(raw, "seed", 0)),
        "modes": [str(x) for x in list(_cfg_get(raw, "modes", ["full"]) or ["full"])],
        "repeat_stability_repeats": [
            int(x) for x in list(_cfg_get(raw, "repeat_stability_repeats", [4, 8, 16, 32]) or [4, 8, 16, 32])
        ],
        "repeat_stability_position": int(_cfg_get(raw, "repeat_stability_position", 9)),
        "order_robustness_permutations": int(_cfg_get(raw, "order_robustness_permutations", 3)),
    }


def run_stage2_2_validation_manifest_only(*, cfg: Any) -> List[Dict[str, Any]]:
    sched = _cfg_get(cfg, "scheduler_stage2_2", {}) or {}
    val = stage2_2_validation_cfg(cfg)
    index_dir = str(_cfg_get(sched, "index_dir", ""))
    if not index_dir:
        raise ValueError("Stage2_2 validation requires scheduler_stage2_2.index_dir")
    manifest = load_or_build_stage2_2_validation_manifest(
        index_dir=index_dir,
        manifest_path=str(val.get("manifest_path", "") or ""),
        protocols=list(val.get("protocols", [])),
        max_entries=int(val.get("max_entries", 8)),
    )
    entries = list(manifest.get("entries", []) or [])
    if not entries:
        raise ValueError("Stage2_2 validation manifest is empty")
    return [dict(x) for x in entries]


def _entry_window(index: Any, entry: Dict[str, Any]) -> Any:
    rows = index.windows[
        (index.windows["segment_row"] == int(entry["segment_row"]))
        & (index.windows["start_local_frame"] == int(entry["start_local_frame"]))
        & (index.windows["pattern_id"] == int(entry.get("pattern_id", 0)))
    ]
    if int(rows.shape[0]) == 0:
        raise ValueError(f"Stage2_2 validation manifest window not found: {entry}")
    return rows[0]


def _row_from_output(*, out: Any, entry: Dict[str, Any], rollout_idx: int, mode: str, trigger_step: int) -> Dict[str, Any]:
    stats = dict(getattr(out, "stats", {}) or {})
    losses = dict(getattr(out, "losses", {}) or {})
    resolved = getattr(out, "resolved", None)
    return {
        "step": int(trigger_step),
        "split": "iforward_stage2_2_validation",
        "protocol": str(entry.get("protocol", "")),
        "mode": str(mode),
        "scene_id": int(entry.get("scene_id", -1)),
        "segment_id": int(entry.get("segment_id", -1)),
        "rollout_idx": int(rollout_idx),
        "scheduler_phase": str(getattr(resolved, "meta", {}).get("scheduler_phase", "")) if resolved is not None else "",
        "sequence_protocol": str(entry.get("window_protocol", "")),
        "pattern_id": int(entry.get("pattern_id", 0)),
        "loss": float(getattr(out, "loss", torch.zeros(())).detach().float().item()) if torch.is_tensor(getattr(out, "loss", None)) else 0.0,
        "current_loss": float(losses.get("current", torch.zeros(())).detach().float().item()) if torch.is_tensor(losses.get("current")) else 0.0,
        "history_loss": float(losses.get("in_rollout_history", torch.zeros(())).detach().float().item()) if torch.is_tensor(losses.get("in_rollout_history")) else 0.0,
        "history_damage": float(losses.get("history_damage", torch.zeros(())).detach().float().item()) if torch.is_tensor(losses.get("history_damage")) else 0.0,
        "current_psnr": float(stats.get("current_psnr", 0.0)),
        "history_rollout_psnr": float(stats.get("history_rollout_psnr", 0.0)),
        "stage2_2_best_damage_loss": float(stats.get("stage2_2/best_damage_loss", 0.0)),
        "stage2_2_best_damage_p90": float(stats.get("stage2_2/best_damage_p90", 0.0)),
        "stage2_2_best_damage_max": float(stats.get("stage2_2/best_damage_max", 0.0)),
        "stage2_2_bank_valid_count": float(stats.get("stage2_2/bank_valid_count", 0.0)),
        "stage2_2_bank_update_count": float(stats.get("stage2_2/bank_update_count", 0.0)),
    }


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
    scheduler: Stage22Scheduler,
    plan: Any,
    model: Any,
    carried_state: Any,
    mode: str,
    device: torch.device,
    trigger_step: int,
    convert_batch_to_minimal_format: Optional[Callable[[Dict[str, Any], torch.device, int], Dict[str, Any]]],
) -> Any:
    raw = scheduler._batch_from_plan(plan)
    batch = (
        convert_batch_to_minimal_format(raw, device, int(trigger_step))
        if callable(convert_batch_to_minimal_format)
        else raw
    )
    return model.forward_rollout(batch, carried_state=carried_state, ablation=str(mode))


def _manual_stage2_2_plan(
    *,
    scheduler: Stage22Scheduler,
    entry: Dict[str, Any],
    window: Any,
    index: Any,
    protocol: str,
    positions: Sequence[int],
    phase: str,
    visit_kind: str,
    repeats_per_block: int,
    rollout_idx: int,
    rollouts_per_episode: int,
    repair_positions: Sequence[int] = (),
    repair_hash: int = -1,
) -> Any:
    segment_row = int(window["segment_row"])
    seg = index.segments[int(segment_row)]
    frame_rows = scheduler._frame_rows_for_window(window, protocol)
    return scheduler._rollout_from_positions(
        frame_rows=frame_rows,
        scene_id=int(seg["scene_id"]),
        segment_id=int(seg["segment_id"]),
        sequence_id=int((int(entry.get("segment_row", 0)) + 1) * 100000 + int(entry.get("start_local_frame", 0))),
        protocol=str(protocol),
        positions=[int(x) for x in positions],
        rollout_idx=int(rollout_idx),
        rollouts_per_episode=int(rollouts_per_episode),
        phase=str(phase),
        visit_kind=str(visit_kind),
        repeats_per_block=int(repeats_per_block),
        history_positions=[],
        repair_positions=[int(x) for x in repair_positions],
        repair_enabled=bool(repair_positions),
        repair_hash=int(repair_hash),
        episode_step_offset=0,
        previous_physical_pos=None,
    )


def _resolved_sequence_positions(out: Any) -> List[int]:
    meta = getattr(getattr(out, "resolved", None), "meta", {}) or {}
    return [int(x) for x in list(meta.get("sequence_positions", []) or [])]


def _append_final_all10_row(
    *,
    rows: List[Dict[str, Any]],
    out: Any,
    entry: Dict[str, Any],
    mode: str,
    trigger_step: int,
    rollout_idx: int,
    label: str,
) -> Dict[str, Any]:
    row = _row_from_output(out=out, entry=entry, rollout_idx=int(rollout_idx), mode=str(mode), trigger_step=int(trigger_step))
    row.update(
        {
            "validation_rollout_kind": str(label),
            "repair_visit_count": 10.0,
            "repair_mean_psnr": float(row.get("current_psnr", 0.0)),
            "repair_mean_loss": float(row.get("current_loss", 0.0)),
            "all10_mean_psnr": float(row.get("current_psnr", 0.0)),
            "all10_mean_loss": float(row.get("current_loss", 0.0)),
            "best_to_final_damage_p90": float(row.get("stage2_2_best_damage_p90", 0.0)),
            "best_to_final_damage_max": float(row.get("stage2_2_best_damage_max", 0.0)),
            "retention_auc": float(row.get("current_psnr", 0.0)),
        }
    )
    rows.append(row)
    return row


def _run_causal_episode(
    *,
    scheduler: Stage22Scheduler,
    episode: Any,
    model: Any,
    entry: Dict[str, Any],
    mode: str,
    device: torch.device,
    trigger_step: int,
    rows: List[Dict[str, Any]],
    convert_batch_to_minimal_format: Optional[Callable[[Dict[str, Any], torch.device, int], Dict[str, Any]]],
) -> Tuple[Any, List[Any]]:
    carried_state = None
    outs: List[Any] = []
    for rollout_idx, plan in enumerate(list(episode.rollouts)[:5]):
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
        rows.append(_row_from_output(out=out, entry=entry, rollout_idx=int(rollout_idx), mode=str(mode), trigger_step=int(trigger_step)))
        outs.append(out)
        carried_state = _detach_next_state(out) if _should_carry(out) else None
    if outs:
        carried_state = _detach_next_state(outs[-1])
    return carried_state, outs


def run_stage2_2_validation(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: torch.device,
    trigger_step: int = 0,
    modes: Optional[List[str]] = None,
    convert_batch_to_minimal_format: Optional[Callable[[Dict[str, Any], torch.device, int], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    val = stage2_2_validation_cfg(cfg)
    sched = _cfg_get(cfg, "scheduler_stage2_2", {}) or {}
    index_dir = str(_cfg_get(sched, "index_dir", ""))
    if not index_dir:
        raise ValueError("Stage2_2 validation requires scheduler_stage2_2.index_dir")
    manifest = load_or_build_stage2_2_validation_manifest(
        index_dir=index_dir,
        manifest_path=str(val.get("manifest_path", "") or ""),
        protocols=list(val.get("protocols", [])),
        max_entries=int(val.get("max_entries", 8)),
    )
    entries = [dict(x) for x in list(manifest.get("entries", []) or [])]
    if not entries:
        raise ValueError("Stage2_2 validation manifest is empty")
    index = load_stage2_2_index(index_dir)
    modes = list(modes or ["full"])
    was_training = bool(model.training)
    rows: List[Dict[str, Any]] = []
    model.eval()
    try:
        with torch.no_grad():
            for entry in entries:
                protocol = str(entry.get("protocol", ""))
                window_protocol = str(entry.get("window_protocol", "D1"))
                for mode in modes:
                    scheduler = Stage22Scheduler(
                        dataset=dataset,
                        cfg=cfg,
                        index=index,
                        protocol_cfg={"weights": {window_protocol: 1.0}},
                        bootstrap_cfg={"end_step": 0, "repeat_choices": [{"repeats": 8, "prob": 1.0}]},
                        repair_cfg={"start_step": 0, "prob": 1.0 if "Repair" in protocol else 0.0},
                        seed=int(val.get("seed", 0)),
                        fail_fast=False,
                    )
                    window = _entry_window(index, entry)
                    episode = scheduler.build_episode_for_window(
                        window=window,
                        protocol=window_protocol,
                        repair_enabled=("Repair" in protocol),
                    )
                    if protocol == "Repeat Stability":
                        carried_state, _ = _run_causal_episode(
                            scheduler=scheduler,
                            episode=episode,
                            model=model,
                            entry=entry,
                            mode=str(mode),
                            device=device,
                            trigger_step=int(trigger_step),
                            rows=rows,
                            convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                        )
                        target_pos = max(0, min(9, int(val.get("repeat_stability_position", 9))))
                        repeat_rows: List[Dict[str, Any]] = []
                        for repeat_count in list(val.get("repeat_stability_repeats", [4, 8, 16, 32])):
                            plan = _manual_stage2_2_plan(
                                scheduler=scheduler,
                                entry=entry,
                                window=window,
                                index=index,
                                protocol=window_protocol,
                                positions=[int(target_pos)],
                                phase="stress",
                                visit_kind="stress",
                                repeats_per_block=int(repeat_count),
                                rollout_idx=1,
                                rollouts_per_episode=2,
                            )
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
                            row = _row_from_output(
                                out=out,
                                entry=entry,
                                rollout_idx=int(repeat_count),
                                mode=str(mode),
                                trigger_step=int(trigger_step),
                            )
                            row.update(
                                {
                                    "validation_rollout_kind": "repeat_stability",
                                    "repeat_stability_repeats": int(repeat_count),
                                    "repeat_stability_position": int(target_pos),
                                }
                            )
                            rows.append(row)
                            repeat_rows.append(row)
                            carried_state = _detach_next_state(out)
                        if repeat_rows:
                            psnrs = [float(r.get("current_psnr", 0.0)) for r in repeat_rows]
                            losses = [float(r.get("current_loss", 0.0)) for r in repeat_rows]
                            rows.append(
                                {
                                    "step": int(trigger_step),
                                    "split": "iforward_stage2_2_validation",
                                    "protocol": str(protocol),
                                    "mode": str(mode),
                                    "scene_id": int(entry.get("scene_id", -1)),
                                    "segment_id": int(entry.get("segment_id", -1)),
                                    "rollout_idx": -1,
                                    "validation_rollout_kind": "repeat_stability_summary",
                                    "repeat_stability_position": int(target_pos),
                                    "psnr_R4": float(repeat_rows[0].get("current_psnr", 0.0)),
                                    "psnr_R8": float(repeat_rows[1].get("current_psnr", 0.0)) if len(repeat_rows) > 1 else 0.0,
                                    "psnr_R16": float(repeat_rows[2].get("current_psnr", 0.0)) if len(repeat_rows) > 2 else 0.0,
                                    "psnr_R32": float(repeat_rows[3].get("current_psnr", 0.0)) if len(repeat_rows) > 3 else 0.0,
                                    "R4_to_R32_drop": float(psnrs[0] - psnrs[-1]),
                                    "loss_monotonic_violation": float(
                                        any(float(b) > float(a) for a, b in zip(losses, losses[1:]))
                                    ),
                                }
                            )
                        continue

                    if protocol == "Order Robustness":
                        causal_state, _ = _run_causal_episode(
                            scheduler=scheduler,
                            episode=episode,
                            model=model,
                            entry=entry,
                            mode=str(mode),
                            device=device,
                            trigger_step=int(trigger_step),
                            rows=rows,
                            convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                        )
                        perm_rows: List[Dict[str, Any]] = []
                        for perm_idx in range(max(1, int(val.get("order_robustness_permutations", 3)))):
                            perm = scheduler._repair_permutation()
                            repair_hash = int(sum((idx + 1) * int(pos + 1) for idx, pos in enumerate(perm)))
                            plan = _manual_stage2_2_plan(
                                scheduler=scheduler,
                                entry=entry,
                                window=window,
                                index=index,
                                protocol=window_protocol,
                                positions=list(perm),
                                phase="repair",
                                visit_kind="repair",
                                repeats_per_block=1,
                                rollout_idx=5 + int(perm_idx),
                                rollouts_per_episode=6 + int(perm_idx),
                                repair_positions=list(perm),
                                repair_hash=int(repair_hash),
                            )
                            out = _run_plan(
                                scheduler=scheduler,
                                plan=plan,
                                model=model,
                                carried_state=_clone_state_for_validation(causal_state),
                                mode=str(mode),
                                device=device,
                                trigger_step=int(trigger_step),
                                convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                            )
                            row = _row_from_output(
                                out=out,
                                entry=entry,
                                rollout_idx=5 + int(perm_idx),
                                mode=str(mode),
                                trigger_step=int(trigger_step),
                            )
                            row.update(
                                {
                                    "validation_rollout_kind": "order_robustness_repair",
                                    "order_perm_idx": int(perm_idx),
                                    "repair_visit_count": float(len(perm)),
                                    "repair_positions": [int(x) for x in perm],
                                    "repair_permutation_hash": int(repair_hash),
                                    "final_current_mean": float(row.get("current_psnr", 0.0)),
                                    "final_all10_mean": float(row.get("current_psnr", 0.0)),
                                }
                            )
                            rows.append(row)
                            perm_rows.append(row)
                        if perm_rows:
                            psnrs = [float(r.get("current_psnr", 0.0)) for r in perm_rows]
                            rows.append(
                                {
                                    "step": int(trigger_step),
                                    "split": "iforward_stage2_2_validation",
                                    "protocol": str(protocol),
                                    "mode": str(mode),
                                    "scene_id": int(entry.get("scene_id", -1)),
                                    "segment_id": int(entry.get("segment_id", -1)),
                                    "rollout_idx": -1,
                                    "validation_rollout_kind": "order_robustness_summary",
                                    "repair_visit_count": float(len(perm_rows[0].get("repair_positions", []) or [])),
                                    "final_current_mean": float(sum(psnrs) / max(1, len(psnrs))),
                                    "final_all10_mean": float(sum(psnrs) / max(1, len(psnrs))),
                                    "final_all10_std_across_perm": float(statistics.pstdev(psnrs)) if len(psnrs) > 1 else 0.0,
                                    "worst_perm_psnr": float(min(psnrs)),
                                }
                            )
                        continue

                    carried_state = None
                    last_out = None
                    for rollout_idx, plan in enumerate(list(episode.rollouts)):
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
                        row = _row_from_output(
                            out=out,
                            entry=entry,
                            rollout_idx=int(rollout_idx),
                            mode=str(mode),
                            trigger_step=int(trigger_step),
                        )
                        if str(getattr(getattr(out, "resolved", None), "meta", {}).get("scheduler_phase", "")) == "repair":
                            repair_positions = _resolved_sequence_positions(out)
                            repair_count = int(len(repair_positions) or 10)
                            row["validation_rollout_kind"] = (
                                "repair_all10" if int(repair_count) == 10 else f"repair_b{int(repair_count)}r1"
                            )
                            row["repair_visit_count"] = float(repair_count)
                            row["repair_mean_psnr"] = float(row.get("current_psnr", 0.0))
                            row["repair_mean_loss"] = float(row.get("current_loss", 0.0))
                            row["all10_mean_psnr"] = float(row.get("current_psnr", 0.0))
                            row["all10_mean_loss"] = float(row.get("current_loss", 0.0))
                        rows.append(row)
                        last_out = out
                        carried_state = _detach_next_state(out) if _should_carry(out) else None
                    if "Causal" in protocol and last_out is not None:
                        final_state = _detach_next_state(last_out)
                        plan = _manual_stage2_2_plan(
                            scheduler=scheduler,
                            entry=entry,
                            window=window,
                            index=index,
                            protocol=window_protocol,
                            positions=list(range(10)),
                            phase="repair",
                            visit_kind="repair",
                            repeats_per_block=1,
                            rollout_idx=10,
                            rollouts_per_episode=11,
                            repair_positions=list(range(10)),
                            repair_hash=0,
                        )
                        out = _run_plan(
                            scheduler=scheduler,
                            plan=plan,
                            model=model,
                            carried_state=final_state,
                            mode=str(mode),
                            device=device,
                            trigger_step=int(trigger_step),
                            convert_batch_to_minimal_format=convert_batch_to_minimal_format,
                        )
                        _append_final_all10_row(
                            rows=rows,
                            out=out,
                            entry=entry,
                            mode=str(mode),
                            trigger_step=int(trigger_step),
                            rollout_idx=10,
                            label=f"{protocol}-FinalAll10",
                        )
    finally:
        model.train(was_training)
    return rows


__all__ = ["run_stage2_2_validation", "run_stage2_2_validation_manifest_only", "stage2_2_validation_cfg"]
