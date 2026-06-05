from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from datasets.iforward_random_window_scheduler import IForwardRandomWindowScheduler


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


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def random_window_validation_cfg(cfg: Any) -> Dict[str, Any]:
    raw = _cfg_get(cfg, "iforward_random_window_validation", {}) or {}
    tb_raw = _cfg_get(raw, "tensorboard_images", {}) or {}
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "run_at_train_start": bool(_cfg_get(raw, "run_at_train_start", True)),
        "interval_steps": int(_cfg_get(raw, "interval_steps", 1000)),
        "segments_per_scene": int(_cfg_get(raw, "segments_per_scene", 1)),
        "rollouts_per_segment": int(_cfg_get(raw, "rollouts_per_segment", 8)),
        "seed": int(_cfg_get(raw, "seed", 20260604)),
        "carry_state_across_rollouts": bool(_cfg_get(raw, "carry_state_across_rollouts", True)),
        "reset_state_at_segment_begin": bool(_cfg_get(raw, "reset_state_at_segment_begin", True)),
        "tensorboard_images_enable": bool(_cfg_get(tb_raw, "enable", False)),
        "tensorboard_images_rollout_indices": [int(x) for x in list(_cfg_get(tb_raw, "rollout_indices", [0, 1, 3, 7]) or [])],
        "tensorboard_images_max_per_role": int(_cfg_get(tb_raw, "max_images_per_role", 2)),
    }


def fixed_random_window_starts(*, num_blocks: int, rollouts: int, seed: int, scene_id: int, segment_id: int) -> List[int]:
    if int(num_blocks) < 4:
        return []
    valid = list(range(0, int(num_blocks) - 4 + 1))
    rng = random.Random(int(seed) + int(scene_id) * 10007 + int(segment_id) * 1009)
    return [int(rng.choice(valid)) for _ in range(int(rollouts))]


def first_valid_random_window_eval_segments(cfg: Any, dataset: Any) -> List[Tuple[int, int]]:
    val_cfg = random_window_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return []
    data_cfg = _cfg_get(cfg, "data", {}) or {}
    eval_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "eval_scene_ids", []) or [])]
    out: List[Tuple[int, int]] = []
    for scene_id in eval_scene_ids:
        found = 0
        for segment_id in sorted(int(x) for x in list(dataset.list_segment_ids(int(scene_id)) or [])):
            sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
            keyframes = [int(x) for x in list(getattr(sidx, "keyframe_indices", []) or [])]
            if len(keyframes) < 4:
                continue
            out.append((int(scene_id), int(segment_id)))
            found += 1
            if found >= int(val_cfg["segments_per_scene"]):
                break
    return out


def make_random_window_validation_scheduler(cfg: Any, dataset: Any, scene_id: int, segment_id: int, starts: Sequence[int]) -> Any:
    sched = _cfg_get(cfg, "scheduler_iforward_random_window", {}) or {}
    traversal_cfg = dict(_cfg_get(sched, "traversal", {}) or {})
    traversal_cfg.update(
        {
            "fixed_scene_id": int(scene_id),
            "fixed_segment_id": int(segment_id),
            "scene_order": "ascending",
            "segment_order": "ascending",
            "seed": int(_cfg_get(traversal_cfg, "seed", 41) or 41),
        }
    )
    preload_cfg = dict(_cfg_get(sched, "preload", {}) or {})
    preload_cfg["emit_hints"] = False
    episode_cfg = dict(_cfg_get(sched, "episode", {}) or {})
    episode_cfg["rollouts_per_episode"] = int(len(starts))
    rollout_cfg = dict(_cfg_get(sched, "rollout", {}) or {})
    rollout_cfg["window_policy"] = "fixed_random_with_replacement"
    return IForwardRandomWindowScheduler(
        dataset=dataset,
        traversal_cfg=traversal_cfg,
        segment_cfg=dict(_cfg_get(sched, "segment", {}) or {}),
        episode_cfg=episode_cfg,
        rollout_cfg=rollout_cfg,
        evidence_cfg=dict(_cfg_get(sched, "evidence", {}) or {}),
        supervision_cfg=dict(_cfg_get(sched, "supervision", {}) or {}),
        memory_cfg=dict(_cfg_get(sched, "memory", {}) or {}),
        loss_timing_cfg=dict(_cfg_get(sched, "loss_timing", {}) or {}),
        preload_cfg=preload_cfg,
        include_test=False,
        fixed_scene_id=int(scene_id),
        fixed_segment_id=int(segment_id),
        seed=int(_cfg_get(traversal_cfg, "seed", 41) or 41),
        fail_fast=True,
        fixed_window_starts=[int(x) for x in starts],
    )


def write_random_window_validation_tb_images(
    *,
    writer: Any,
    out: Any,
    step: int,
    scene_id: int,
    segment_id: int,
    rollout_idx: int,
    max_images_per_role: int,
) -> None:
    if writer is None or int(max_images_per_role) <= 0:
        return
    pred_rgbs = list(getattr(out, "pred_rgbs", []) or [])
    gt_images = list(getattr(out, "gt_images", []) or [])
    image_refs = list(getattr(out, "image_refs", []) or [])
    image_roles = list(getattr(out, "image_roles", []) or [])
    counts: Dict[str, int] = {}
    for idx, (pred_raw, gt_raw) in enumerate(zip(pred_rgbs, gt_images)):
        role = str(image_roles[idx] if idx < len(image_roles) else "view").strip().lower().replace("/", "_")
        count = int(counts.get(role, 0))
        if count >= int(max_images_per_role):
            continue
        counts[role] = count + 1
        pred = torch.clamp(pred_raw.detach().float().cpu(), 0.0, 1.0)
        gt = torch.clamp(gt_raw.detach().float().cpu(), 0.0, 1.0)
        if pred.dim() != 3 or gt.dim() != 3:
            continue
        err = (pred - gt).abs()
        max_err = float(err.max().item()) if err.numel() else 0.0
        if max_err > 0.0:
            err = err / max_err
        ref = image_refs[idx] if idx < len(image_refs) else (-1, -1)
        tag = (
            f"iforward_random_window_validation/images/"
            f"scene_{int(scene_id):03d}_segment_{int(segment_id):03d}/rollout_{int(rollout_idx):02d}/"
            f"{role}/view_{count}_f{int(ref[0]):05d}_c{int(ref[1])}"
        )
        writer.add_image(f"{tag}/pred", pred.permute(2, 0, 1), int(step))
        writer.add_image(f"{tag}/gt", gt.permute(2, 0, 1), int(step))
        writer.add_image(f"{tag}/error", err.permute(2, 0, 1), int(step))


def write_random_window_validation_rows(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: torch.device,
    trigger_step: int,
    trigger_train_episode_counter: int,
    metrics_fh: Any,
    writer: Any,
    convert_batch_to_minimal_format: Any,
    write_metrics_history: Any,
    **_: Any,
) -> None:
    val_cfg = random_window_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return
    segments = first_valid_random_window_eval_segments(cfg, dataset)
    if not segments:
        row = {
            "step": int(trigger_step),
            "split": "iforward_random_window_validation_global",
            "trigger_step": int(trigger_step),
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "num_rollouts": 0,
            "status": "no_valid_eval_segments",
        }
        if metrics_fh is not None:
            write_metrics_history(metrics_fh, row)
        return

    was_training = bool(model.training)
    saved_cache = dict(getattr(model, "_state_cache", {}) or {})
    reset_bridge = getattr(model, "_reset_bridge_runtime_node_state", None)
    rows: List[Dict[str, Any]] = []
    if hasattr(model, "reset_iforward_state_cache"):
        model.reset_iforward_state_cache()
    if callable(reset_bridge):
        reset_bridge()
    model.eval()
    try:
        with torch.no_grad():
            for scene_id, segment_id in segments:
                sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
                keyframes = [int(x) for x in list(getattr(sidx, "keyframe_indices", []) or [])]
                starts = fixed_random_window_starts(
                    num_blocks=len(keyframes),
                    rollouts=int(val_cfg["rollouts_per_segment"]),
                    seed=int(val_cfg["seed"]),
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                )
                if not starts:
                    continue
                if bool(val_cfg["reset_state_at_segment_begin"]):
                    if hasattr(model, "reset_iforward_state_cache"):
                        model.reset_iforward_state_cache()
                    if callable(reset_bridge):
                        reset_bridge()
                scheduler = make_random_window_validation_scheduler(cfg, dataset, int(scene_id), int(segment_id), starts)
                carried = None
                previous_by_window: Dict[int, Dict[str, float]] = {}
                for rollout_idx in range(len(starts)):
                    raw_batch = scheduler.next_batch()
                    target = raw_batch.get("target") or {}
                    num_targets = int(target["image"].shape[0])
                    minimal = convert_batch_to_minimal_format(
                        raw_batch,
                        device,
                        num_targets=num_targets,
                        include_source_for_2d=True,
                        view_selection=None,
                    )
                    minimal["global_step"] = int(trigger_step)
                    out = model.forward_rollout(minimal, carried_state=carried, ablation="full")
                    stats = dict(out.stats or {})
                    losses = {name: _safe_float(value.detach().item()) for name, value in out.losses.items()}
                    window_hash = int(stats.get("window_hash", -1))
                    prev = previous_by_window.get(int(window_hash))
                    current_delta = float("nan")
                    history_delta = float("nan")
                    nearby_delta = float("nan")
                    if prev is not None:
                        current_delta = _safe_float(stats.get("current_latest_psnr")) - _safe_float(prev.get("current_latest_psnr"))
                        history_delta = _safe_float(stats.get("in_rollout_history_psnr")) - _safe_float(prev.get("in_rollout_history_psnr"))
                        nearby_delta = _safe_float(stats.get("nearby_psnr")) - _safe_float(prev.get("nearby_psnr"))
                    previous_by_window[int(window_hash)] = {
                        "current_latest_psnr": _safe_float(stats.get("current_latest_psnr")),
                        "in_rollout_history_psnr": _safe_float(stats.get("in_rollout_history_psnr")),
                        "nearby_psnr": _safe_float(stats.get("nearby_psnr")),
                    }
                    row = {
                        "step": int(trigger_step),
                        "split": "iforward_random_window_validation",
                        "trigger_step": int(trigger_step),
                        "trigger_train_episode_counter": int(trigger_train_episode_counter),
                        "scene_id": int(scene_id),
                        "segment_id": int(segment_id),
                        "rollout_idx": int(rollout_idx),
                        "window_start": int(stats.get("window_start", starts[int(rollout_idx)])),
                        "window_hash": int(window_hash),
                        "is_repeated_window": bool(stats.get("is_repeated_window", False)),
                        "loss_total": _safe_float(out.loss.detach().item()),
                        "loss_current_latest": losses.get("current_latest", losses.get("current", float("nan"))),
                        "loss_in_rollout_history": losses.get("in_rollout_history", float("nan")),
                        "loss_short_window_history": losses.get("short_window_history", float("nan")),
                        "loss_nearby": losses.get("nearby", float("nan")),
                        "current_latest_psnr": _safe_float(stats.get("current_latest_psnr")),
                        "in_rollout_history_psnr": _safe_float(stats.get("in_rollout_history_psnr")),
                        "short_window_history_psnr": _safe_float(stats.get("short_window_history_psnr")),
                        "nearby_psnr": _safe_float(stats.get("nearby_psnr")),
                        "revisit_current_psnr_delta": current_delta,
                        "revisit_history_psnr_delta": history_delta,
                        "revisit_nearby_psnr_delta": nearby_delta,
                    }
                    rows.append(row)
                    if metrics_fh is not None:
                        write_metrics_history(metrics_fh, row)
                    if writer is not None:
                        tag = f"iforward_random_window_validation/scene_{int(scene_id):03d}_segment_{int(segment_id):03d}/rollout_{int(rollout_idx):02d}"
                        for name in ("current_latest_psnr", "in_rollout_history_psnr", "short_window_history_psnr", "nearby_psnr", "loss_total"):
                            writer.add_scalar(f"{tag}/{name}", float(row[name]), int(trigger_step))
                        writer.add_scalar(f"{tag}/window_start", float(row["window_start"]), int(trigger_step))
                        writer.add_scalar(f"{tag}/is_repeated_window", float(1.0 if row["is_repeated_window"] else 0.0), int(trigger_step))
                        if (
                            bool(val_cfg["tensorboard_images_enable"])
                            and int(rollout_idx) in set(int(x) for x in val_cfg["tensorboard_images_rollout_indices"])
                        ):
                            write_random_window_validation_tb_images(
                                writer=writer,
                                out=out,
                                step=int(trigger_step),
                                scene_id=int(scene_id),
                                segment_id=int(segment_id),
                                rollout_idx=int(rollout_idx),
                                max_images_per_role=int(val_cfg["tensorboard_images_max_per_role"]),
                            )
                    if bool(val_cfg["carry_state_across_rollouts"]) and bool(out.resolved.carry_scene_state_after_rollout):
                        carried = out.next_state.detach_for_next_rollout()
                    else:
                        carried = None
                if hasattr(model, "reset_iforward_state_cache"):
                    model.reset_iforward_state_cache()
                if callable(reset_bridge):
                    reset_bridge()
    finally:
        if hasattr(model, "_state_cache"):
            model._state_cache = saved_cache
        if callable(reset_bridge):
            reset_bridge()
        model.train(was_training)

    if not rows:
        return
    last4 = rows[-4:]
    final_row = rows[-1]
    revisit_rows = [r for r in rows if math.isfinite(float(r["revisit_current_psnr_delta"]))]
    global_row = {
        "step": int(trigger_step),
        "split": "iforward_random_window_validation_global",
        "trigger_step": int(trigger_step),
        "trigger_train_episode_counter": int(trigger_train_episode_counter),
        "num_rollouts": int(len(rows)),
        "all_rollouts_current_latest_psnr_mean": _mean([r["current_latest_psnr"] for r in rows]),
        "all_rollouts_history_psnr_mean": _mean([r["in_rollout_history_psnr"] for r in rows]),
        "all_rollouts_nearby_psnr_mean": _mean([r["nearby_psnr"] for r in rows]),
        "last4_current_latest_psnr_mean": _mean([r["current_latest_psnr"] for r in last4]),
        "last4_history_psnr_mean": _mean([r["in_rollout_history_psnr"] for r in last4]),
        "last4_nearby_psnr_mean": _mean([r["nearby_psnr"] for r in last4]),
        "final_rollout_current_latest_psnr": float(final_row["current_latest_psnr"]),
        "final_rollout_history_psnr": float(final_row["in_rollout_history_psnr"]),
        "final_rollout_nearby_psnr": float(final_row["nearby_psnr"]),
        "revisit_current_psnr_delta_mean": _mean([r["revisit_current_psnr_delta"] for r in revisit_rows]),
        "revisit_history_psnr_delta_mean": _mean([r["revisit_history_psnr_delta"] for r in revisit_rows]),
        "revisit_nearby_psnr_delta_mean": _mean([r["revisit_nearby_psnr_delta"] for r in revisit_rows]),
    }
    if metrics_fh is not None:
        write_metrics_history(metrics_fh, global_row)
    if writer is not None:
        for key, value in global_row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(f"iforward_random_window_validation/global/{key}", float(value), int(trigger_step))
        flush = getattr(writer, "flush", None)
        if callable(flush):
            flush()


__all__ = [
    "fixed_random_window_starts",
    "first_valid_random_window_eval_segments",
    "make_random_window_validation_scheduler",
    "random_window_validation_cfg",
    "write_random_window_validation_rows",
]
