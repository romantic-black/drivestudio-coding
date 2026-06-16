from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from datasets.train_scheduler_iforward import (
    IFORWARD_V4_SCHEDULER_VERSION,
    TrainSchedulerIForward,
)

ImageRef = Tuple[int, int]


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


def _stable_shape_hash(name: str) -> int:
    value = 0
    for ch in str(name):
        value = (value * 131 + ord(ch)) & 0x7FFFFFFF
    return int(value)


def _shape_dict(raw: Any) -> Dict[str, Any]:
    item = dict(raw or {})
    blocks = int(_cfg_get(item, "blocks_per_rollout", 0) or 0)
    repeats = int(_cfg_get(item, "repeats_per_block", 0) or 0)
    if blocks < 1 or repeats < 1:
        raise ValueError("iforward_coverage_validation.shapes entries require blocks_per_rollout/repeats_per_block >= 1")
    name = str(_cfg_get(item, "name", f"r{repeats}b{blocks}"))
    return {"name": name, "blocks_per_rollout": blocks, "repeats_per_block": repeats, "prob": 1.0}


def iforward_coverage_validation_cfg(cfg: Any) -> Dict[str, Any]:
    raw = _cfg_get(cfg, "iforward_coverage_validation", {}) or {}
    episode = dict(_cfg_get(raw, "episode", {}) or {})
    rollout = dict(_cfg_get(raw, "rollout", {}) or {})
    tb_images = dict(_cfg_get(raw, "tensorboard_images", {}) or {})
    shapes_raw = list(_cfg_get(raw, "shapes", []) or [])
    if not shapes_raw:
        shapes_raw = [
            {"name": "r8b1", "blocks_per_rollout": 1, "repeats_per_block": 8},
            {"name": "r4b2", "blocks_per_rollout": 2, "repeats_per_block": 4},
            {"name": "r2b4", "blocks_per_rollout": 4, "repeats_per_block": 2},
        ]
    target_rs = [int(x) for x in list(_cfg_get(raw, "target_repeats_per_block", [8]) or [])]
    if not target_rs:
        raise ValueError("iforward_coverage_validation.target_repeats_per_block must not be empty")
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "run_at_train_start": bool(_cfg_get(raw, "run_at_train_start", True)),
        "interval_steps": int(_cfg_get(raw, "interval_steps", 2000)),
        "seed": int(_cfg_get(raw, "seed", 20260614)),
        "segments_per_scene": int(_cfg_get(raw, "segments_per_scene", 1)),
        "max_segments_total": int(_cfg_get(raw, "max_segments_total", 1)),
        "blocks_per_episode": int(_cfg_get(episode, "blocks_per_episode", 10)),
        "episode_stride": int(_cfg_get(episode, "episode_stride", _cfg_get(episode, "blocks_per_episode", 10))),
        "allow_short_last_episode": bool(_cfg_get(episode, "allow_short_last_episode", False)),
        "min_blocks_per_episode": int(_cfg_get(episode, "min_blocks_per_episode", 4)),
        "start_offset": int(_cfg_get(rollout, "start_offset", 0)),
        "tail_policy": str(_cfg_get(rollout, "tail_policy", "circular_fill")),
        "max_inner_K": int(_cfg_get(rollout, "max_inner_K", 8)),
        "shapes": [_shape_dict(x) for x in shapes_raw],
        "target_repeats_per_block": target_rs,
        "supervision": copy.deepcopy(dict(_cfg_get(raw, "supervision", {}) or {})),
        "final_eval": copy.deepcopy(dict(_cfg_get(raw, "final_eval", {}) or {})),
        "tensorboard_images_enable": bool(_cfg_get(tb_images, "enable", False)),
        "tensorboard_images_max_per_role": int(_cfg_get(tb_images, "max_images_per_role", 2)),
    }


def select_coverage_validation_segments(cfg: Any, dataset: Any) -> List[Tuple[int, int]]:
    val_cfg = iforward_coverage_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return []
    eval_scene_ids = [
        int(x)
        for x in list(_cfg_get(_cfg_get(cfg, "data", {}) or {}, "eval_scene_ids", []) or [])
    ]
    out: List[Tuple[int, int]] = []
    for scene_id in eval_scene_ids:
        found = 0
        for segment_id in sorted(int(x) for x in list(dataset.list_segment_ids(int(scene_id)) or [])):
            sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
            keyframes = [int(x) for x in list(getattr(sidx, "keyframe_indices", []) or [])]
            if len(keyframes) < int(val_cfg["blocks_per_episode"]):
                if not bool(val_cfg["allow_short_last_episode"]) or len(keyframes) < int(val_cfg["min_blocks_per_episode"]):
                    continue
            out.append((int(scene_id), int(segment_id)))
            found += 1
            if found >= int(val_cfg["segments_per_scene"]):
                break
            if int(val_cfg["max_segments_total"]) > 0 and len(out) >= int(val_cfg["max_segments_total"]):
                break
        if int(val_cfg["max_segments_total"]) > 0 and len(out) >= int(val_cfg["max_segments_total"]):
            break
    return out[: int(val_cfg["max_segments_total"])] if int(val_cfg["max_segments_total"]) > 0 else out


def _validation_supervision_cfg(raw: Mapping[str, Any]) -> Dict[str, Any]:
    current = dict(_cfg_get(raw, "current", {}) or {})
    history = dict(_cfg_get(raw, "history_replay", {}) or {})
    nearby = dict(_cfg_get(raw, "nearby", {}) or {})
    final_eval = dict(_cfg_get(raw, "final_eval", {}) or {})
    current.setdefault("enable", True)
    current.setdefault("role_name", "final_current_recon")
    current.setdefault("frame_policy", "all_rollout_input_frames")
    current.setdefault("camera_policy", "all_cams")
    history.setdefault("enable", True)
    history.setdefault("role_name", "final_history_replay")
    history.setdefault("sampling_policy", "previous_visited_blocks")
    history.setdefault("exclude_current_blocks", True)
    history.setdefault("camera_policy", "all_cams")
    history.setdefault("max_frames_per_rollout", 8)
    nearby.setdefault("enable", True)
    nearby.setdefault("role_name", "final_nearby_rollout")
    nearby.setdefault("scope", "current_rollout_random_block")
    nearby.setdefault("policy", "random_unsupervised_frame_in_current_rollout_block")
    nearby.setdefault("validation_sampling_policy", "fixed_once_per_block")
    nearby.setdefault("frames_per_rollout", 1)
    nearby.setdefault("camera_policy", "all_cams")
    nearby.setdefault("max_refs_per_rollout", 3)
    nearby.setdefault("add_to_evidence", False)
    final_eval.setdefault("enable", True)
    final_eval.setdefault("attach_to_last_rollout", True)
    final_eval.setdefault("recon_all_blocks", True)
    final_eval.setdefault("nearby_nvs_all_blocks", True)
    final_eval.setdefault("roles_zero_loss", True)
    final_eval.setdefault("require_per_ref_psnr", True)
    return {
        "current": current,
        "history_replay": history,
        "nearby": nearby,
        "final_eval": final_eval,
    }


def build_coverage_validation_scheduler(
    *,
    cfg: Any,
    dataset: Any,
    scene_id: int,
    segment_id: int,
    shape: Mapping[str, Any],
    target_repeats_per_block: int,
) -> TrainSchedulerIForward:
    val_cfg = iforward_coverage_validation_cfg(cfg)
    shape_item = _shape_dict(shape)
    seed = (
        int(val_cfg["seed"])
        + int(scene_id) * 10007
        + int(segment_id) * 1009
        + _stable_shape_hash(str(shape_item["name"])) * 131
        + int(target_repeats_per_block) * 17
    )
    supervision_raw = copy.deepcopy(dict(val_cfg.get("supervision", {}) or {}))
    if val_cfg.get("final_eval"):
        supervision_raw["final_eval"] = copy.deepcopy(dict(val_cfg["final_eval"]))
    return TrainSchedulerIForward(
        dataset=dataset,
        episode_cfg={
            "source_mode": "keyframes",
            "blocks_per_episode": int(val_cfg["blocks_per_episode"]),
            "episode_stride": int(val_cfg["episode_stride"]),
            "allow_short_last_episode": bool(val_cfg["allow_short_last_episode"]),
            "min_blocks_per_episode": int(val_cfg["min_blocks_per_episode"]),
            "target_repeats_per_block": int(target_repeats_per_block),
            "block_source_frame_policy": "random_within_keyframe_once_per_episode",
            "reset_scene_state_policy": "episode_begin",
        },
        rollout_cfg={
            "shape_sample_scope": "episode",
            "block_selection_policy": "ordered_cyclic_start",
            "start_offset_policy": "fixed",
            "start_offset": int(val_cfg["start_offset"]),
            "tail_policy": str(val_cfg["tail_policy"]),
            "delivery_order_policy": "rollout_order",
            "max_inner_K": int(val_cfg["max_inner_K"]),
            "detach_graph_after_rollout": True,
            "shapes": [shape_item],
            "shapes_schedule": [],
            "fixed_shape_name": str(shape_item["name"]),
        },
        traversal_cfg={
            "fixed_scene_id": int(scene_id),
            "fixed_segment_id": int(segment_id),
            "scene_order": "ascending",
            "segment_order": "ascending",
            "traversal_mode": "episode_serial",
            "seed": int(seed),
        },
        evidence_cfg={"camera_policy": "all_cams", "allow_camera_dropout": False, "mask_policy": "non_sky_non_egocar"},
        supervision_cfg=_validation_supervision_cfg(supervision_raw),
        memory_cfg={
            "observation_commit_policy": "first_repeat_only",
            "optimizer_memory_update_policy": "every_repeat",
            "reset_policy": "episode_begin",
            "carry_policy": "across_rollouts_until_episode_end",
        },
        loss_timing_cfg={"policy": "rollout_final_only", "intermediate_step_loss": False},
        leakage_check_cfg={"enable": True, "forbid_test_refs_in_train": True},
        preload_cfg={"emit_hints": False},
        include_test=False,
        fixed_scene_id=int(scene_id),
        fixed_segment_id=int(segment_id),
        seed=int(seed),
        version=IFORWARD_V4_SCHEDULER_VERSION,
        fail_fast=True,
    )


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            out.append(value_f)
    return out


def _mean(values: Iterable[float]) -> float:
    vals = _finite(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return float(vals[0])
    pos = max(0.0, min(1.0, float(q))) * float(len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _min(values: Iterable[float]) -> float:
    vals = _finite(values)
    return float(min(vals)) if vals else float("nan")


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_f = torch.nan_to_num(pred.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
    gt_f = torch.nan_to_num(gt.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
    mse = torch.mean((pred_f - gt_f) ** 2).item() if pred_f.numel() and gt_f.numel() else float("nan")
    if not math.isfinite(float(mse)):
        return float("nan")
    if float(mse) <= 1.0e-12:
        return 100.0
    return float(-10.0 * math.log10(max(float(mse), 1.0e-12)))


def per_ref_psnr(out: Any) -> List[Dict[str, Any]]:
    pred_rgbs = list(getattr(out, "pred_rgbs", []) or [])
    gt_images = list(getattr(out, "gt_images", []) or [])
    image_refs = list(getattr(out, "image_refs", []) or [])
    image_roles = list(getattr(out, "image_roles", []) or [])
    n = min(len(pred_rgbs), len(gt_images), len(image_refs), len(image_roles))
    rows: List[Dict[str, Any]] = []
    for idx in range(int(n)):
        ref = image_refs[idx]
        try:
            ref_t = (int(ref[0]), int(ref[1]))  # type: ignore[index]
        except Exception:
            ref_t = (-1, -1)
        rows.append(
            {
                "ref": ref_t,
                "role": str(image_roles[idx]),
                "psnr": _psnr(pred_rgbs[idx], gt_images[idx]),
            }
        )
    return rows


def _role_psnr(rows: Sequence[Mapping[str, Any]], role: str) -> float:
    return _mean(float(row.get("psnr", float("nan"))) for row in rows if str(row.get("role")) == str(role))


def _normalize_int_map(raw: Any) -> Dict[int, int]:
    if not isinstance(raw, Mapping):
        return {}
    return {int(k): int(v) for k, v in raw.items()}


def _block_psnr_for_frames(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    frame_by_block: Mapping[int, int],
) -> Dict[int, float]:
    by_block: Dict[int, List[float]] = {int(k): [] for k in frame_by_block.keys()}
    frame_to_blocks: Dict[int, List[int]] = {}
    for block_id, frame_idx in frame_by_block.items():
        frame_to_blocks.setdefault(int(frame_idx), []).append(int(block_id))
    for row in rows:
        if str(row.get("role")) != str(role):
            continue
        ref = row.get("ref", (-1, -1))
        try:
            frame_idx = int(ref[0])  # type: ignore[index]
        except Exception:
            continue
        for block_id in frame_to_blocks.get(int(frame_idx), []):
            by_block.setdefault(int(block_id), []).append(float(row.get("psnr", float("nan"))))
    return {int(block_id): _mean(vals) for block_id, vals in by_block.items()}


def _loss_value(out: Any, name: str, fallback: str = "") -> float:
    losses = dict(getattr(out, "losses", {}) or {})
    value = losses.get(name, losses.get(fallback, None))
    if value is None:
        return float("nan")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return _safe_float(value)


def _total_loss_value(out: Any) -> float:
    value = getattr(out, "loss", None)
    if value is None:
        return float("nan")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return _safe_float(value)


def _write_row(write_metrics_history: Any, metrics_fh: Any, row: Dict[str, Any]) -> None:
    if metrics_fh is None:
        return
    if callable(write_metrics_history):
        write_metrics_history(metrics_fh, row)
    elif hasattr(metrics_fh, "append"):
        metrics_fh.append(dict(row))
    else:
        metrics_fh.write(str(row) + "\n")


def _model_state_owners(model: Any) -> List[Any]:
    owners = [model]
    module = getattr(model, "module", None)
    if module is not None and module is not model:
        owners.append(module)
    return owners


def _replace_state_caches_for_validation(model: Any) -> List[Tuple[Any, Any]]:
    saved: List[Tuple[Any, Any]] = []
    for owner in _model_state_owners(model):
        if not hasattr(owner, "_state_cache"):
            continue
        saved.append((owner, getattr(owner, "_state_cache")))
        setattr(owner, "_state_cache", {})
    return saved


def _restore_state_caches(saved: Sequence[Tuple[Any, Any]]) -> None:
    for owner, cache in saved:
        setattr(owner, "_state_cache", cache)


def _reset_validation_state_cache_only(model: Any) -> None:
    for owner in _model_state_owners(model):
        if hasattr(owner, "_state_cache"):
            getattr(owner, "_state_cache").clear()


def _reset_bridge_runtime_only(model: Any) -> None:
    for owner in _model_state_owners(model):
        reset_bridge = getattr(owner, "_reset_bridge_runtime_node_state", None)
        if callable(reset_bridge):
            reset_bridge()


def _reset_validation_runtime(model: Any) -> None:
    _reset_validation_state_cache_only(model)
    _reset_bridge_runtime_only(model)


def _detach_next_state(out: Any) -> Any:
    next_state = getattr(out, "next_state", None)
    detach = getattr(next_state, "detach_for_next_rollout", None)
    return detach() if callable(detach) else next_state


def write_iforward_coverage_validation_rows(
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
    val_cfg = iforward_coverage_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return
    segments = select_coverage_validation_segments(cfg, dataset)
    if not segments:
        row = {
            "step": int(trigger_step),
            "split": "iforward_coverage_validation_global",
            "trigger_step": int(trigger_step),
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "num_segments": 0,
            "status": "no_valid_eval_segments",
            "hgv2_validation_mode": "no_grad_direct",
        }
        _write_row(write_metrics_history, metrics_fh, row)
        return

    was_training = bool(getattr(model, "training", False))
    saved_caches = _replace_state_caches_for_validation(model)
    model.eval()
    final_rows: List[Dict[str, Any]] = []
    try:
        _reset_bridge_runtime_only(model)
        with torch.no_grad():
            for scene_id, segment_id in segments:
                for shape in list(val_cfg["shapes"]):
                    for target_r in list(val_cfg["target_repeats_per_block"]):
                        scheduler = build_coverage_validation_scheduler(
                            cfg=cfg,
                            dataset=dataset,
                            scene_id=int(scene_id),
                            segment_id=int(segment_id),
                            shape=shape,
                            target_repeats_per_block=int(target_r),
                        )
                        carried_state = None
                        block_last_visit_psnr: Dict[int, float] = {}
                        block_best_visit_psnr: Dict[int, float] = {}
                        final_row: Optional[Dict[str, Any]] = None
                        _reset_validation_runtime(model)
                        while True:
                            raw_batch = scheduler.next_batch()
                            ifwd = dict(raw_batch.get("_iforward", {}) or {})
                            meta = dict(raw_batch.get("request_meta", ifwd.get("request_meta", {})) or {})
                            target = raw_batch.get("target", {}) or {}
                            num_targets = None
                            image = target.get("image") if isinstance(target, Mapping) else None
                            if hasattr(image, "shape") and len(image.shape) > 0:
                                num_targets = int(image.shape[0])
                            minimal_batch = convert_batch_to_minimal_format(
                                raw_batch,
                                device,
                                num_targets=num_targets,
                                include_source_for_2d=True,
                                view_selection=None,
                            )
                            minimal_batch["global_step"] = int(trigger_step)
                            out = model.forward_rollout(minimal_batch, carried_state=carried_state, ablation="full")
                            rows = per_ref_psnr(out)
                            window_blocks = [int(x) for x in list(ifwd.get("window_block_ids", []) or [])]
                            input_frames = [int(x) for x in list(ifwd.get("input_frame_indices", []) or [])]
                            current_map = {int(block_id): int(frame_idx) for block_id, frame_idx in zip(window_blocks, input_frames)}
                            current_block_psnr = _block_psnr_for_frames(
                                rows,
                                role="current_latest",
                                frame_by_block=current_map,
                            )
                            for block_id, psnr_value in current_block_psnr.items():
                                if not math.isfinite(float(psnr_value)):
                                    continue
                                block_last_visit_psnr[int(block_id)] = float(psnr_value)
                                prev = block_best_visit_psnr.get(int(block_id), float("-inf"))
                                block_best_visit_psnr[int(block_id)] = max(float(prev), float(psnr_value))

                            stats = dict(getattr(out, "stats", {}) or {})
                            rollout_row = {
                                "step": int(trigger_step),
                                "split": "iforward_coverage_validation_rollout",
                                "trigger_step": int(trigger_step),
                                "trigger_train_episode_counter": int(trigger_train_episode_counter),
                                "scene_id": int(scene_id),
                                "segment_id": int(segment_id),
                                "shape_name": str(shape["name"]),
                                "target_repeats_per_block": int(target_r),
                                "rollout_idx": int(ifwd.get("rollout_idx_in_episode", 0)),
                                "rollouts_per_episode": int(ifwd.get("rollouts_per_episode", 1)),
                                "window_block_ids": window_blocks,
                                "is_wraparound_rollout": bool(ifwd.get("is_wraparound_rollout", meta.get("is_wraparound_rollout", False))),
                                "block_repeat_count_min_after": int(meta.get("block_repeat_count_min_after", 0)),
                                "block_repeat_count_max_after": int(meta.get("block_repeat_count_max_after", 0)),
                                "loss_total": _total_loss_value(out),
                                "loss_current": _loss_value(out, "current", "current_latest"),
                                "loss_history": _loss_value(out, "in_rollout_history", "history"),
                                "loss_nearby": _loss_value(out, "nearby"),
                                "current_psnr": _role_psnr(rows, "current_latest"),
                                "history_psnr": _role_psnr(rows, "history_rollout"),
                                "nearby_psnr": _role_psnr(rows, "nearby"),
                                "current_ssim": _safe_float(stats.get("current_ssim")),
                                "history_ssim": _safe_float(stats.get("history_rollout_ssim", stats.get("in_rollout_history_ssim"))),
                                "nearby_ssim": _safe_float(stats.get("nearby_ssim")),
                                "hgv2_validation_mode": "no_grad_direct",
                            }
                            _write_row(write_metrics_history, metrics_fh, rollout_row)

                            resolved = getattr(out, "resolved", None)
                            episode_end = bool(getattr(resolved, "episode_end_after_rollout", ifwd.get("episode_end_after_rollout", False)))
                            carry_after = bool(getattr(resolved, "carry_scene_state_after_rollout", ifwd.get("carry_scene_state_after_rollout", False)))
                            if episode_end:
                                block_frame_map = _normalize_int_map(meta.get("block_frame_map", ifwd.get("request_meta", {}).get("block_frame_map", {})))
                                final_recon = _block_psnr_for_frames(
                                    rows,
                                    role="eval_recon_all_blocks",
                                    frame_by_block=block_frame_map,
                                )
                                nearby_frame_map = _normalize_int_map(meta.get("final_eval_nearby_nvs_frame_map", {}))
                                final_nearby = _block_psnr_for_frames(
                                    rows,
                                    role="eval_nearby_nvs_all_blocks",
                                    frame_by_block=nearby_frame_map,
                                )
                                recon_values = list(final_recon.values())
                                nearby_values = list(final_nearby.values())
                                last_drops = [
                                    float(block_last_visit_psnr[int(block_id)]) - float(final_recon[int(block_id)])
                                    for block_id in final_recon.keys()
                                    if int(block_id) in block_last_visit_psnr
                                    and math.isfinite(float(final_recon[int(block_id)]))
                                ]
                                best_drops = [
                                    float(block_best_visit_psnr[int(block_id)]) - float(final_recon[int(block_id)])
                                    for block_id in final_recon.keys()
                                    if int(block_id) in block_best_visit_psnr
                                    and math.isfinite(float(final_recon[int(block_id)]))
                                ]
                                final_row = {
                                    "step": int(trigger_step),
                                    "split": "iforward_coverage_validation_final",
                                    "trigger_step": int(trigger_step),
                                    "trigger_train_episode_counter": int(trigger_train_episode_counter),
                                    "scene_id": int(scene_id),
                                    "segment_id": int(segment_id),
                                    "shape_name": str(shape["name"]),
                                    "target_repeats_per_block": int(target_r),
                                    "num_blocks": int(ifwd.get("episode_num_blocks", len(block_frame_map))),
                                    "num_rollouts": int(ifwd.get("rollouts_per_episode", 1)),
                                    "final_recon_psnr_mean": _mean(recon_values),
                                    "final_recon_psnr_p10": _percentile(recon_values, 0.10),
                                    "final_recon_psnr_min": _min(recon_values),
                                    "final_nearby_nvs_psnr_mean": _mean(nearby_values),
                                    "final_nearby_nvs_psnr_p10": _percentile(nearby_values, 0.10),
                                    "final_nearby_nvs_psnr_min": _min(nearby_values),
                                    "final_recon_ssim_mean": _safe_float(stats.get("eval_recon_all_blocks_ssim")),
                                    "final_nearby_nvs_ssim_mean": _safe_float(stats.get("eval_nearby_nvs_all_blocks_ssim")),
                                    "forget_last_to_final_drop_mean": _mean(last_drops),
                                    "forget_last_to_final_drop_p90": _percentile(last_drops, 0.90),
                                    "forget_last_to_final_drop_max": float(max(_finite(last_drops))) if _finite(last_drops) else float("nan"),
                                    "forget_best_to_final_drop_mean": _mean(best_drops),
                                    "forget_best_to_final_drop_p90": _percentile(best_drops, 0.90),
                                    "forget_best_to_final_drop_max": float(max(_finite(best_drops))) if _finite(best_drops) else float("nan"),
                                    "coverage_exact": bool(meta.get("coverage_exact", False)),
                                    "coverage_reaches_target": bool(meta.get("coverage_reaches_target", False)),
                                    "coverage_exact_target": bool(meta.get("coverage_exact_target", False)),
                                    "coverage_exact_achieved": bool(meta.get("coverage_exact_achieved", meta.get("coverage_exact", False))),
                                    "achieved_repeats_per_block": int(meta.get("achieved_repeats_per_block", 0)),
                                    "hgv2_validation_mode": "no_grad_direct",
                                    "block_repeat_count_min": int(meta.get("block_repeat_count_min_after", 0)),
                                    "block_repeat_count_max": int(meta.get("block_repeat_count_max_after", 0)),
                                }
                                _write_row(write_metrics_history, metrics_fh, final_row)
                                final_rows.append(dict(final_row))
                            if bool(carry_after) and not bool(episode_end):
                                carried_state = _detach_next_state(out)
                            else:
                                carried_state = None
                                _reset_validation_runtime(model)
                            if bool(episode_end):
                                break
                        if final_row is None:
                            raise RuntimeError("coverage validation scheduler ended without a final row")
    finally:
        _reset_bridge_runtime_only(model)
        _restore_state_caches(saved_caches)
        train = getattr(model, "train", None)
        if callable(train):
            train(was_training)

    global_row: Dict[str, Any] = {
        "step": int(trigger_step),
        "split": "iforward_coverage_validation_global",
        "trigger_step": int(trigger_step),
        "trigger_train_episode_counter": int(trigger_train_episode_counter),
        "num_segments": int(len(segments)),
        "num_final_rows": int(len(final_rows)),
        "hgv2_validation_mode": "no_grad_direct",
    }
    for shape in list(val_cfg["shapes"]):
        for target_r in list(val_cfg["target_repeats_per_block"]):
            prefix = f"{shape['name']}_R{int(target_r)}"
            rows = [
                row
                for row in final_rows
                if str(row.get("shape_name")) == str(shape["name"])
                and int(row.get("target_repeats_per_block", -1)) == int(target_r)
            ]
            global_row[f"{prefix}_recon_psnr_mean"] = _mean(row.get("final_recon_psnr_mean", float("nan")) for row in rows)
            global_row[f"{prefix}_nearby_nvs_psnr_mean"] = _mean(
                row.get("final_nearby_nvs_psnr_mean", float("nan")) for row in rows
            )
            global_row[f"{prefix}_forget_p90"] = _mean(
                row.get("forget_best_to_final_drop_p90", float("nan")) for row in rows
            )
            if writer is not None:
                tag = f"iforward_coverage_validation/{shape['name']}/R{int(target_r)}"
                writer.add_scalar(f"{tag}/recon_psnr_mean", float(global_row[f"{prefix}_recon_psnr_mean"]), int(trigger_step))
                writer.add_scalar(
                    f"{tag}/nearby_nvs_psnr_mean",
                    float(global_row[f"{prefix}_nearby_nvs_psnr_mean"]),
                    int(trigger_step),
                )
                writer.add_scalar(f"{tag}/forget_p90", float(global_row[f"{prefix}_forget_p90"]), int(trigger_step))
    _write_row(write_metrics_history, metrics_fh, global_row)
    if writer is not None:
        flush = getattr(writer, "flush", None)
        if callable(flush):
            flush()


__all__ = [
    "build_coverage_validation_scheduler",
    "iforward_coverage_validation_cfg",
    "per_ref_psnr",
    "select_coverage_validation_segments",
    "write_iforward_coverage_validation_rows",
]
