"""
IForward multi-scene training entry.

Uses the existing V4 dataset materializer and scheduler_iforward batch contract,
but builds an independent IForward trainer.
"""

from __future__ import annotations

import copy
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from datasets.train_scheduler_iforward import TrainSchedulerIForward
from models.iforward import IForwardTrainer
from tools.train_minimal_streetforward_stage4_3_iforward_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_iforward_from_cfg,
    resolve_fixed_scene_segment_iforward,
)


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


def build_iforward_trainer_from_cfg(config: Any, device: torch.device) -> IForwardTrainer:
    return IForwardTrainer(config=config, device=device)


def checkpoint_prefix_iforward_from_cfg(cfg: Any) -> str:
    model_cfg = _cfg_get(cfg, "model", {}) or {}
    iforward_cfg = _cfg_get(model_cfg, "iforward", {}) or {}
    version = str(_cfg_get(iforward_cfg, "version", "v1"))
    return f"iforward_{version}"


def _config_file_from_argv(argv: List[str], default_config: str) -> str:
    for idx, arg in enumerate(argv):
        if arg == "--config_file" and idx + 1 < len(argv):
            return str(argv[idx + 1])
        if arg.startswith("--config_file="):
            return str(arg.split("=", 1)[1])
    return str(default_config)


def _route_random_window_entrypoint_if_needed(default_config: str) -> bool:
    config_file = _config_file_from_argv(list(sys.argv), default_config)
    cfg = OmegaConf.load(config_file)
    random_cfg = _cfg_get(cfg, "scheduler_iforward_random_window", None)
    legacy_cfg = _cfg_get(cfg, "scheduler_iforward", None)
    random_enabled = random_cfg is not None and bool(_cfg_get(random_cfg, "enable", True))
    if random_enabled and legacy_cfg is None:
        from tools import train_iforward_random_window

        train_iforward_random_window.main()
        return True
    if legacy_cfg is None:
        raise ValueError(
            "tools/train_iforward.py requires legacy scheduler_iforward. "
            "For scheduler_iforward_random_window configs, use tools/train_iforward_random_window.py."
        )
    return False


def _iforward_validation_cfg(cfg: Any) -> Dict[str, Any]:
    raw = _cfg_get(cfg, "iforward_validation", {}) or {}
    tb_images_raw = _cfg_get(raw, "tensorboard_images", {}) or {}
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "run_at_train_start": bool(_cfg_get(raw, "run_at_train_start", True)),
        "interval_steps": int(_cfg_get(raw, "interval_steps", 1000)),
        "segments_per_scene": int(_cfg_get(raw, "segments_per_scene", 1)),
        "rollouts_per_segment": int(_cfg_get(raw, "rollouts_per_segment", 1)),
        "tensorboard_images_enable": bool(_cfg_get(tb_images_raw, "enable", True)),
        "tensorboard_images_max_per_role": int(_cfg_get(tb_images_raw, "max_images_per_role", 2)),
    }


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: List[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_image_role(role: Any) -> str:
    text = str(role or "view").strip().lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(out).strip("_") or "view"


def _tb_hwc01(img: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(torch.nan_to_num(img.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if x.dim() != 3:
        raise ValueError(f"expected image tensor [H,W,C], got {tuple(x.shape)}")
    if int(x.shape[-1]) == 1:
        x = x.expand(*x.shape[:-1], 3)
    if int(x.shape[-1]) != 3:
        raise ValueError(f"expected image tensor channel dim=3, got {tuple(x.shape)}")
    return x


def _write_iforward_validation_tb_images(
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
    n = min(len(pred_rgbs), len(gt_images))
    role_counts: Dict[str, int] = {}
    for idx in range(int(n)):
        role = _safe_image_role(image_roles[idx] if idx < len(image_roles) else "view")
        count = int(role_counts.get(role, 0))
        if count >= int(max_images_per_role):
            continue
        role_counts[role] = count + 1
        ref = image_refs[idx] if idx < len(image_refs) else (-1, -1)
        try:
            frame_idx = int(ref[0])  # type: ignore[index]
            cam_idx = int(ref[1])  # type: ignore[index]
        except Exception:
            frame_idx = -1
            cam_idx = -1
        try:
            pred = _tb_hwc01(pred_rgbs[idx])
            gt = _tb_hwc01(gt_images[idx])
            err = (pred - gt).abs()
            max_err = float(err.max().item()) if err.numel() else 0.0
            if max_err > 0.0:
                err = err / max_err
            tag = (
                f"iforward_validation/images/"
                f"scene_{int(scene_id):03d}_segment_{int(segment_id):03d}/"
                f"{role}/rollout_{int(rollout_idx)}/view_{count}_f{frame_idx:05d}_c{cam_idx}"
            )
            writer.add_image(f"{tag}/pred", pred.permute(2, 0, 1), int(step))
            writer.add_image(f"{tag}/gt", gt.permute(2, 0, 1), int(step))
            writer.add_image(f"{tag}/error", err.permute(2, 0, 1), int(step))
        except Exception as exc:
            base.logger.warning(
                "Failed to write IForward validation TensorBoard image scene=%s segment=%s role=%s idx=%s: %s",
                int(scene_id),
                int(segment_id),
                role,
                int(idx),
                exc,
            )


def _first_valid_iforward_eval_segments(cfg: Any, dataset: Any) -> List[Tuple[int, int]]:
    val_cfg = _iforward_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return []
    eval_scene_ids = [int(x) for x in list(_cfg_get(_cfg_get(cfg, "data", {}) or {}, "eval_scene_ids", []) or [])]
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


def _make_validation_scheduler(cfg: Any, dataset: Any, scene_id: int, segment_id: int) -> TrainSchedulerIForward:
    sched = _cfg_get(cfg, "scheduler_iforward", {}) or {}
    episode_cfg = copy.deepcopy(dict(_cfg_get(sched, "episode", {}) or {}))
    episode_cfg.update(
        {
            "blocks_per_episode": 4,
            "episode_stride": 4,
            "allow_short_last_episode": False,
            "min_blocks_per_episode": 4,
        }
    )
    rollout_cfg = copy.deepcopy(dict(_cfg_get(sched, "rollout", {}) or {}))
    rollout_cfg.update(
        {
            "allow_short_final_rollout": False,
            "min_blocks_per_rollout": 4,
            "avoid_single_block_tail": True,
            "shapes": [
                {
                    "name": "b4_r2",
                    "blocks_per_rollout": 4,
                    "repeats_per_block": 2,
                    "prob": 1.0,
                }
            ],
            "shapes_schedule": [],
        }
    )
    traversal_cfg = copy.deepcopy(dict(_cfg_get(sched, "traversal", {}) or {}))
    traversal_cfg.update(
        {
            "fixed_scene_id": int(scene_id),
            "fixed_segment_id": int(segment_id),
            "scene_order": "ascending",
            "segment_order": "ascending",
            "traversal_mode": "episode_serial",
            "seed": 0,
        }
    )
    preload_cfg = copy.deepcopy(dict(_cfg_get(sched, "preload", {}) or {}))
    preload_cfg["emit_hints"] = False
    return TrainSchedulerIForward(
        dataset=dataset,
        episode_cfg=episode_cfg,
        rollout_cfg=rollout_cfg,
        traversal_cfg=traversal_cfg,
        evidence_cfg=copy.deepcopy(dict(_cfg_get(sched, "evidence", {}) or {})),
        supervision_cfg=copy.deepcopy(dict(_cfg_get(sched, "supervision", {}) or {})),
        memory_cfg=copy.deepcopy(dict(_cfg_get(sched, "memory", {}) or {})),
        loss_timing_cfg=copy.deepcopy(dict(_cfg_get(sched, "loss_timing", {}) or {})),
        leakage_check_cfg=copy.deepcopy(dict(_cfg_get(sched, "leakage_check", {}) or {})),
        preload_cfg=preload_cfg,
        include_test=False,
        fixed_scene_id=int(scene_id),
        fixed_segment_id=int(segment_id),
        seed=0,
        version="iforward_v1",
        fail_fast=True,
    )


def _write_iforward_validation_rows(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: torch.device,
    trigger_step: int,
    trigger_train_episode_counter: int,
    metrics_fh: Any,
    writer: Any,
    **_: Any,
) -> None:
    val_cfg = _iforward_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return
    segments = _first_valid_iforward_eval_segments(cfg, dataset)
    if not segments:
        row = {
            "step": int(trigger_step),
            "split": "iforward_validation_global",
            "trigger_step": int(trigger_step),
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "num_rollouts": 0,
            "status": "no_valid_eval_segments",
        }
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, row)
        return

    was_training = bool(model.training)
    saved_cache = dict(getattr(model, "_state_cache", {}) or {})
    rows: List[Dict[str, Any]] = []
    if hasattr(model, "reset_iforward_state_cache"):
        model.reset_iforward_state_cache()
    reset_bridge = getattr(model, "_reset_bridge_runtime_node_state", None)
    if callable(reset_bridge):
        reset_bridge()
    model.eval()
    try:
        with torch.no_grad():
            for scene_id, segment_id in segments:
                scheduler = _make_validation_scheduler(cfg, dataset, int(scene_id), int(segment_id))
                for rollout_idx in range(int(val_cfg["rollouts_per_segment"])):
                    raw_batch = scheduler.next_batch()
                    target = raw_batch.get("target") or {}
                    num_targets = int(target["image"].shape[0])
                    minimal_batch = base.convert_batch_to_minimal_format(
                        raw_batch,
                        device,
                        num_targets=num_targets,
                        include_source_for_2d=True,
                        view_selection=None,
                    )
                    minimal_batch["global_step"] = int(trigger_step)
                    out = model.forward_rollout(minimal_batch, carried_state=None, ablation="full")
                    stats = dict(out.stats or {})
                    losses = {name: _safe_float(value.detach().item()) for name, value in out.losses.items()}
                    row = {
                        "step": int(trigger_step),
                        "split": "iforward_validation",
                        "trigger_step": int(trigger_step),
                        "trigger_train_episode_counter": int(trigger_train_episode_counter),
                        "scene_id": int(scene_id),
                        "segment_id": int(segment_id),
                        "rollout_idx": int(rollout_idx),
                        "rollout_shape": "b4_r2",
                        "inference_only": True,
                        "loss": _safe_float(out.loss.detach().item()),
                        "current_loss": losses.get("current", float("nan")),
                        "history_rollout_loss": losses.get("in_rollout_history", float("nan")),
                        "nearby_loss": losses.get("nearby", float("nan")),
                        "current_psnr": _safe_float(stats.get("current_psnr")),
                        "history_rollout_psnr": _safe_float(stats.get("history_rollout_psnr")),
                        "nearby_psnr": _safe_float(stats.get("nearby_psnr")),
                        "current_valid_ratio": _safe_float(stats.get("current_valid_ratio")),
                        "history_rollout_valid_ratio": _safe_float(stats.get("in_rollout_history_valid_ratio")),
                        "nearby_valid_ratio": _safe_float(stats.get("nearby_valid_ratio")),
                        "current_num_refs": _safe_float(stats.get("current_num_refs")),
                        "history_rollout_num_refs": _safe_float(stats.get("history_rollout_num_refs")),
                        "nearby_num_refs": _safe_float(stats.get("nearby_num_refs")),
                    }
                    rows.append(row)
                    if metrics_fh is not None:
                        base._write_metrics_history(metrics_fh, row)
                    if writer is not None:
                        tag = f"iforward_validation/scene_{int(scene_id):03d}_segment_{int(segment_id):03d}"
                        writer.add_scalar(f"{tag}/current_psnr", float(row["current_psnr"]), int(trigger_step))
                        writer.add_scalar(f"{tag}/history_rollout_psnr", float(row["history_rollout_psnr"]), int(trigger_step))
                        writer.add_scalar(f"{tag}/nearby_psnr", float(row["nearby_psnr"]), int(trigger_step))
                        if bool(val_cfg["tensorboard_images_enable"]):
                            _write_iforward_validation_tb_images(
                                writer=writer,
                                out=out,
                                step=int(trigger_step),
                                scene_id=int(scene_id),
                                segment_id=int(segment_id),
                                rollout_idx=int(rollout_idx),
                                max_images_per_role=int(val_cfg["tensorboard_images_max_per_role"]),
                            )
                    if callable(reset_bridge):
                        reset_bridge()
                    if hasattr(model, "reset_iforward_state_cache"):
                        model.reset_iforward_state_cache()
    finally:
        if hasattr(model, "_state_cache"):
            model._state_cache = saved_cache
        if callable(reset_bridge):
            reset_bridge()
        model.train(was_training)

    if rows:
        global_row = {
            "step": int(trigger_step),
            "split": "iforward_validation_global",
            "trigger_step": int(trigger_step),
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "num_rollouts": int(len(rows)),
            "current_psnr": _mean([float(r["current_psnr"]) for r in rows]),
            "history_rollout_psnr": _mean([float(r["history_rollout_psnr"]) for r in rows]),
            "nearby_psnr": _mean([float(r["nearby_psnr"]) for r in rows]),
            "current_valid_ratio": _mean([float(r["current_valid_ratio"]) for r in rows]),
            "history_rollout_valid_ratio": _mean([float(r["history_rollout_valid_ratio"]) for r in rows]),
            "nearby_valid_ratio": _mean([float(r["nearby_valid_ratio"]) for r in rows]),
        }
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, global_row)
        if writer is not None:
            writer.add_scalar("iforward_validation/global/current_psnr", float(global_row["current_psnr"]), int(trigger_step))
            writer.add_scalar(
                "iforward_validation/global/history_rollout_psnr",
                float(global_row["history_rollout_psnr"]),
                int(trigger_step),
            )
            writer.add_scalar("iforward_validation/global/nearby_psnr", float(global_row["nearby_psnr"]), int(trigger_step))
            flush = getattr(writer, "flush", None)
            if callable(flush):
                flush()


def _iforward_train_start_hook(**kwargs: Any) -> None:
    cfg = kwargs["cfg"]
    val_cfg = _iforward_validation_cfg(cfg)
    if bool(val_cfg["enable"]) and bool(val_cfg["run_at_train_start"]):
        _write_iforward_validation_rows(**kwargs)


def _iforward_step_end_hook(**kwargs: Any) -> None:
    cfg = kwargs["cfg"]
    val_cfg = _iforward_validation_cfg(cfg)
    interval = int(val_cfg["interval_steps"])
    step = int(kwargs.get("trigger_step", 0))
    if not bool(val_cfg["enable"]) or interval <= 0:
        return
    if step < 0 or (step + 1) % int(interval) != 0:
        return
    _write_iforward_validation_rows(**kwargs)


def main() -> None:
    default_config = "configs/iforward/iforward_base.yaml"
    if not any(arg == "--config_file" or arg.startswith("--config_file=") for arg in sys.argv):
        sys.argv.extend(["--config_file", default_config])
    if _route_random_window_entrypoint_if_needed(default_config):
        return
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_iforward_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_iforward
    base.TRAINER_CLASS = build_iforward_trainer_from_cfg
    base.MinimalStreetForwardStage4_3 = build_iforward_trainer_from_cfg
    base.TRAIN_START_HOOK = _iforward_train_start_hook
    base.STEP_END_HOOK = _iforward_step_end_hook
    base.CKPT_PREFIX = "iforward_v1"
    base.CHECKPOINT_PREFIX_RESOLVER = checkpoint_prefix_iforward_from_cfg
    base.DEFAULT_CONFIG_FILE = default_config
    base.ALLOW_ONE_SEGMENT = False
    base.main()


if __name__ == "__main__":
    main()
