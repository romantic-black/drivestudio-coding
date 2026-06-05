"""
Minimal StreetForward multi-scene training loop（shared by stage wrappers）。

默认配置与 Trainer 类可由上层入口脚本覆盖（例如 stage4_3/4_4/4_5 + scheduler_v7 包装）。

日志里的 `scene_id` 与数据配置中的 `train_scene_ids` / 预处理时的 `scene_idx` 一致；`scene_dir` 为三位补零目录名（如 scene_id=5 -> 005），与 nuScenes 等导出目录 `str(scene_idx).zfill(3)` 对齐。

保存到 `images/train/` 的文件名片段：`sc{文件夹名}` 对应磁盘场景目录；`v{k}` 为**第 k 个 target 视图**（不是 scene）；`f`/`c` 为当前 batch 内时间步与相机槽位；若配置了 `data.pixel_source.cameras`，则追加 `_nuscam{id}` 为该槽位对应的 nuScenes 全局相机编号（与 drivestudio 注释中 CAM_* 编号一致）。

默认配置：
  conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding \\
    python tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py \\
      --config_file configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml

scheduler_v4.overlap（见 docs/trainers/TrainScheduler_V4_Overlap_Pointcloud_TopK.md）：
  - mode: none — 贪心选 target keyframe。
  - mode: pointcloud_topk — 需 overlap.point_sample_size、candidate_frame_policy: middle、score_type: nab_over_na、overlap_min，且 dataset.pointcloud 可用；score > overlap_min 的候选中随机选 extra，不足则 top-k 回退。
"""

from __future__ import annotations

import argparse
import gc
import inspect
import io
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO, Tuple

from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _normalize_omp_num_threads

_normalize_omp_num_threads()

import numpy as np
import torch
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.validation_scheduler_v7 import ValidationEpisodeSpecV7, build_validation_episode_specs_v7
from models.streetforward.metrics import compute_ssim_loss_masked
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3, RuntimePolicy
from tools.streetforward_validation_v7_config import ValidationV7Config, parse_validation_v7_config
from tools.train_minimal_streetforward_stage1_1 import (
    _compute_metrics,
    _open_metrics_history,
    _save_image_triplet,
    _write_metrics_history,
    convert_batch_to_minimal_format,
    setup,
)
from tools.train_minimal_streetforward_stage4_1 import (
    _diagnose_step,
    _parse_diagnostics_cfg,
    _parse_perf_cfg,
    _percentile,
    _save_diagnostic_renders,
)
from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import (
    _build_scheduler_node_sync,
    _load_init_checkpoint,
    _resolve_init_checkpoint_cfg,
)
from tools.train_minimal_streetforward_stage4_3_v4_common import (
    build_multi_scene_dataset_v3,
    build_train_scheduler_from_cfg,
    parse_include_test,
    resolve_fixed_scene_segment,
)
from tools.upload_to_vika import upload_experiment_summary
from utils.minimal_batch_view_selection import parse_view_selection
from utils.streetforward_baseline import set_deterministic_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

logger = logging.getLogger(__name__)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
CKPT_PREFIX = "minimal_sf_stage4_3_multi_scene_v4"
CHECKPOINT_PREFIX_RESOLVER = None
TRAINER_CLASS = MinimalStreetForwardStage4_3
DEFAULT_CONFIG_FILE = "configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml"
ALLOW_ONE_SEGMENT = False
ALLOW_OPTIONAL_ONE_SEGMENT = False
EPISODE_END_HOOK = None
TRAIN_START_HOOK = None
STEP_END_HOOK = None


def _scene_dir_str(scene_id: Any) -> str:
    """Zero-padded scene folder name (e.g. 5 -> '005'), matching preprocessed data dirs like nuScenes export."""
    try:
        s = int(scene_id)
    except (TypeError, ValueError):
        return "unknown"
    if s < 0:
        return "unknown"
    return f"{s:03d}"


def _metric_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _metric_int(value: Any, default: int = -1) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _metric_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _metric_list_int(value: Any) -> List[int]:
    if value is None:
        return []
    try:
        return [int(x) for x in list(value)]
    except (TypeError, ValueError):
        return []


def _is_iforward_random_window_result(result: Dict[str, Any]) -> bool:
    return str(result.get("iforward/scheduler_version", "")) == "random_window_v1"


def _copy_metric(
    row: Dict[str, Any],
    dst_key: str,
    result: Dict[str, Any],
    src_key: str,
    *,
    default: Optional[float] = None,
) -> None:
    row[dst_key] = _metric_float(result.get(src_key), default)


def _build_iforward_random_window_train_step_row(
    *,
    step: int,
    minimal_batch: Dict[str, Any],
    scheduler_info: Dict[str, Any],
    step_events: List[Dict[str, Any]],
    result: Dict[str, Any],
    loss_val: float,
    num_views: int,
    step_time_ms: float,
    batch_fetch_ms: float,
    batch_convert_ms: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "step": int(step),
        "split": "train_step",
        "scheduler_version": str(result.get("iforward/scheduler_version", "random_window_v1")),
        "scene_id": _metric_int(result.get("iforward/scene_id", minimal_batch.get("scene_id", -1))),
        "segment_id": _metric_int(result.get("iforward/segment_id", minimal_batch.get("segment_id", -1))),
        "epoch_idx": _metric_int(scheduler_info.get("epoch_idx", -1)),
        "global_step": _metric_int(scheduler_info.get("global_step", step)),
        "episode_id": _metric_int(result.get("iforward/episode_id", scheduler_info.get("episode_id", -1))),
        "rollout_id_global": _metric_int(result.get("iforward/rollout_id_global", -1)),
        "rollout_idx": _metric_int(result.get("iforward/rollout_idx_in_episode", -1)),
        "rollouts_per_episode": _metric_int(result.get("iforward/rollouts_per_episode", -1)),
        "window_start": _metric_int(result.get("iforward/window_start", -1)),
        "window_end": _metric_int(result.get("iforward/window_end", -1)),
        "window_hash": _metric_int(result.get("iforward/window_hash", -1)),
        "window_block_ids": _metric_list_int(result.get("iforward/window_block_ids")),
        "window_revisit_count": _metric_int(result.get("iforward/window_revisit_count", 0), 0),
        "unique_windows_seen": _metric_int(result.get("iforward/unique_windows_seen", 0), 0),
        "is_repeated_window": _metric_bool(result.get("iforward/is_repeated_window", False)),
        "reset_before_rollout": _metric_bool(result.get("iforward/reset_scene_state_before_rollout", False)),
        "carry_after_rollout": _metric_bool(result.get("iforward/carry_scene_state_after_rollout", False)),
        "episode_end_after_rollout": _metric_bool(result.get("iforward/episode_end_after_rollout", False)),
        "blocks_per_rollout": _metric_int(result.get("iforward/blocks_per_rollout", 4), 4),
        "repeats_per_block": _metric_int(result.get("iforward/repeats_per_block", 2), 2),
        "inner_K": _metric_int(result.get("iforward/inner_K", 8), 8),
        "num_views": int(num_views),
        "num_source_views": _metric_int(result.get("num_source_views", result.get("iforward/num_source_views", 0)), 0),
        "num_targets": _metric_int(result.get("num_targets", result.get("iforward/num_targets", 0)), 0),
        "num_gaussians_bg": _metric_int(result.get("num_gaussians_bg", result.get("iforward/num_gaussians_bg", 0)), 0),
        "num_gaussians_distant": _metric_int(
            result.get("num_gaussians_distant", result.get("iforward/num_gaussians_distant", 0)),
            0,
        ),
        "num_gaussians_rigid": _metric_int(
            result.get("num_gaussians_rigid", result.get("iforward/num_gaussians_rigid", 0)),
            0,
        ),
        "num_gaussians_sky": _metric_int(result.get("num_gaussians_sky", result.get("iforward/num_gaussians_sky", 0)), 0),
        "state_cache_size": _metric_int(result.get("iforward/state_cache_size", 0), 0),
        "stale_state_cache_entries_cleared": _metric_int(
            result.get("iforward/stale_state_cache_entries_cleared", 0),
            0,
        ),
        "history_entries_before": _metric_int(result.get("iforward/history_entries_before", 0), 0),
        "history_entries_after": _metric_int(result.get("iforward/history_entries_after", 0), 0),
        "history_entries": _metric_int(result.get("iforward/history_entries", 0), 0),
        "memory_entries_before": _metric_int(result.get("iforward/memory_entries_before", 0), 0),
        "memory_entries_after": _metric_int(result.get("iforward/memory_entries_after", 0), 0),
        "grad_norm_unclipped": _metric_float(result.get("iforward/grad_norm_unclipped")),
        "grad_norm_after_clip": _metric_float(result.get("iforward/grad_norm_after_clip")),
        "grad_clip_applied": _metric_bool(result.get("iforward/grad_clip_applied", False)),
        "step_time_ms": float(step_time_ms),
        "batch_fetch_ms": float(batch_fetch_ms),
        "batch_convert_ms": float(batch_convert_ms),
        "resolve_ms": _metric_float(result.get("resolve_ms"), 0.0),
        "forward_ms": _metric_float(result.get("forward_ms"), 0.0),
        "backward_ms": _metric_float(result.get("backward_ms"), 0.0),
        "optimizer_ms": _metric_float(result.get("optimizer_ms"), 0.0),
        "state_cache_ms": _metric_float(result.get("state_cache_ms"), 0.0),
        "logging_pack_ms": _metric_float(result.get("logging_pack_ms"), 0.0),
        "step_event_types": [str(ev.get("type", "")) for ev in step_events],
    }
    _copy_metric(row, "loss_total", result, "iforward/loss_total", default=float(loss_val))
    for name in (
        "current_latest",
        "in_rollout_history",
        "short_window_history",
        "nearby",
        "delta_regularization",
    ):
        _copy_metric(row, f"loss_{name}", result, f"iforward/loss_{name}")
    for role in ("current_latest", "in_rollout_history", "short_window_history", "nearby"):
        _copy_metric(row, f"{role}_psnr", result, f"iforward/{role}_psnr")
        _copy_metric(row, f"{role}_ssim", result, f"iforward/{role}_ssim")
        _copy_metric(row, f"{role}_l1", result, f"iforward/{role}_l1")
        _copy_metric(row, f"{role}_valid_ratio", result, f"iforward/{role}_valid_ratio")
        _copy_metric(row, f"{role}_num_refs", result, f"iforward/{role}_num_refs")
    for role in ("current", "history", "nearby"):
        _copy_metric(row, f"revisit_{role}_psnr_delta", result, f"iforward/revisit/{role}_psnr_delta")
    for branch in ("bg_point", "distant_point", "rigid_point"):
        _copy_metric(row, f"memory_{branch}_seen_ratio", result, f"iforward/memory_tokens/{branch}_seen_ratio")
        _copy_metric(row, f"memory_{branch}_seen", result, f"iforward/memory_tokens/{branch}_seen")
    return row


def _build_iforward_random_window_diagnostics_row(
    *,
    step: int,
    result: Dict[str, Any],
    scheduler_info: Dict[str, Any],
    diag_row: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "step": int(step),
        "split": "train_step_diagnostics",
        "scheduler_version": str(result.get("iforward/scheduler_version", "random_window_v1")),
        "scene_id": _metric_int(result.get("iforward/scene_id", -1)),
        "segment_id": _metric_int(result.get("iforward/segment_id", -1)),
        "episode_id": _metric_int(result.get("iforward/episode_id", -1)),
        "rollout_idx": _metric_int(result.get("iforward/rollout_idx_in_episode", -1)),
        "window_hash": _metric_int(result.get("iforward/window_hash", -1)),
        "global_step": _metric_int(scheduler_info.get("global_step", step)),
    }
    prefixes = (
        "iforward/optimizer/",
        "iforward/grad/",
        "iforward/adapter/",
        "iforward/delta_reg/",
        "iforward/loss_weight/",
        "iforward/memory_tokens/",
        "iforward/runtime_node_state_reset_before/",
        "iforward/runtime_node_state_reset_after/",
    )
    exact_keys = {
        "iforward/runtime_node_state_reset_before",
        "iforward/runtime_node_state_reset_after",
        "iforward/runtime_node_state_reset_before/before_bg",
        "iforward/runtime_node_state_reset_after/after_bg",
        "iforward/observe_ms",
        "iforward/event_ms",
        "iforward/memory_ms",
        "iforward/update_ms",
        "iforward/delta_reg_ms",
        "iforward/final_render_ms",
        "grad_norm_ms",
    }
    for key, value in result.items():
        if key.startswith("_"):
            continue
        if key not in exact_keys and not any(str(key).startswith(prefix) for prefix in prefixes):
            continue
        if isinstance(value, bool):
            row[key] = bool(value)
        elif isinstance(value, int):
            row[key] = int(value)
        elif isinstance(value, float):
            value_f = float(value)
            if np.isfinite(value_f):
                row[key] = value_f
    for key, value in diag_row.items():
        if isinstance(value, bool):
            row[key] = bool(value)
        elif isinstance(value, (int, float)):
            value_f = float(value)
            if np.isfinite(value_f):
                row[key] = value_f
    return row


def _write_scalar_row_to_tensorboard(writer: Any, prefix: str, row: Dict[str, Any], step: int) -> None:
    if writer is None:
        return
    skip = {"step", "split", "scheduler_version", "window_block_ids", "step_event_types"}
    for key, value in row.items():
        if key in skip or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            writer.add_scalar(f"{prefix}/{key}", float(value), int(step))


def _checkpoint_prefix_for_cfg(cfg: Any) -> str:
    resolver = globals().get("CHECKPOINT_PREFIX_RESOLVER", None)
    if callable(resolver):
        return str(resolver(cfg))
    return str(CKPT_PREFIX)


def _checkpoint_step(payload: Dict[str, Any], default: int = 0) -> int:
    for key in ("global_step", "step", "iteration", "iter"):
        value = payload.get(key)
        if value is not None:
            return int(value)
    lr_info = payload.get("lr_scheduler")
    if isinstance(lr_info, dict) and lr_info.get("global_step") is not None:
        return int(lr_info["global_step"])
    opt_state = payload.get("optimizer_state_dict")
    if isinstance(opt_state, dict) and opt_state.get("_sf_global_step") is not None:
        return int(opt_state["_sf_global_step"])
    return int(default)


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        logger.warning("Resume checkpoint has no rng_state; sampling will not be bitwise-continuous.")
        return
    try:
        if "python_random" in state:
            random.setstate(state["python_random"])
        if "numpy_random" in state:
            np.random.set_state(state["numpy_random"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"].detach().cpu())
        cuda_states = state.get("torch_cuda_all")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.detach().cpu() for x in list(cuda_states)])
        logger.info("Restored RNG state from resume checkpoint.")
    except Exception:
        logger.exception("Failed to restore RNG state from resume checkpoint; continuing with current RNG state.")


def _restore_scheduler_state_from_checkpoint(scheduler: Any, payload: Dict[str, Any]) -> bool:
    state = payload.get("scheduler_state")
    if not isinstance(state, dict):
        logger.warning("Resume checkpoint has no scheduler_state; scheduler traversal will restart from a fresh plan.")
        return False
    loader = getattr(scheduler, "load_state_dict", None)
    if not callable(loader):
        logger.warning("Scheduler %s does not implement load_state_dict; cannot restore scheduler_state.", type(scheduler).__name__)
        return False
    loader(state)
    logger.info(
        "Restored scheduler_state version=%s global_step=%s epoch_idx=%s.",
        state.get("scheduler_version"),
        state.get("global_step"),
        state.get("epoch_idx"),
    )
    return True


def _apply_scheduler_start_step(scheduler: Any, start_step: int) -> None:
    if int(start_step) <= 0:
        return
    if hasattr(scheduler, "global_step"):
        setattr(scheduler, "global_step", int(start_step))
    prefetch = getattr(scheduler, "_prefetch_v9_plan_for_current_state", None)
    if callable(prefetch):
        prefetch()
    logger.info("Applied training.start_step=%s to scheduler global_step.", int(start_step))


def _resolve_resume_checkpoint_cfg(cfg: Any, args: argparse.Namespace) -> str:
    cli_path = str(getattr(args, "resume_checkpoint", "") or "")
    training_cfg = cfg.get("training") or {}
    cfg_path = str(training_cfg.get("resume_checkpoint", "") or "")
    return cli_path or cfg_path


def _resolve_start_step(cfg: Any, args: argparse.Namespace, resume_payload: Optional[Dict[str, Any]]) -> int:
    cli_start = getattr(args, "start_step", None)
    if cli_start is not None:
        return int(cli_start)
    training_cfg = cfg.get("training") or {}
    cfg_start = int(training_cfg.get("start_step", 0) or 0)
    if cfg_start > 0:
        return int(cfg_start)
    if resume_payload is not None:
        return int(_checkpoint_step(resume_payload)) + 1
    return 0


def _load_resume_checkpoint(path: str, model: Any) -> Dict[str, Any]:
    if not path:
        return {}
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Resume checkpoint must be a dict payload, got {type(ckpt)}: {path}")
    sd = ckpt.get("model_state_dict")
    if sd is None:
        raise ValueError(f"Resume checkpoint missing model_state_dict: {path}")
    model.load_state_dict(sd, strict=True)
    od = ckpt.get("optimizer_state_dict")
    if od is None:
        raise ValueError(f"Resume checkpoint missing optimizer_state_dict: {path}")
    if hasattr(model, "load_optimizer_state_from_checkpoint"):
        loaded = bool(model.load_optimizer_state_from_checkpoint(ckpt))
        if not loaded:
            raise ValueError(f"Resume checkpoint optimizer state is incompatible with current model/config: {path}")
    else:
        model.optimizer.load_state_dict(od)
    runtime_loader = getattr(model, "load_runtime_state_from_checkpoint", None)
    if callable(runtime_loader):
        runtime_loader(ckpt)
    logger.info(
        "Loaded resume_checkpoint from %s (saved_step=%s, global_step=%s)",
        path,
        ckpt.get("step"),
        ckpt.get("global_step"),
    )
    return ckpt


def _checkpoint_runtime_extra(
    *,
    model: Any,
    scheduler: Any,
    train_episode_counter: int,
    step: int,
    start_step: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "train_loop": {
            "start_step": int(start_step),
            "saved_at_step": int(step),
            "train_episode_counter": int(train_episode_counter),
        },
        "rng_state": _capture_rng_state(),
    }
    state_fn = getattr(scheduler, "state_dict", None)
    if callable(state_fn):
        payload["scheduler_state"] = state_fn()
    model_runtime_fn = getattr(model, "build_runtime_checkpoint_extra", None)
    if callable(model_runtime_fn):
        payload.update(model_runtime_fn())
    return payload


def _scene_folder_label_from_batch(raw_batch: Dict[str, Any], scene_id_fallback: Any) -> str:
    """Prefer dataset batch ``scene_folder_name`` (e.g. ``005``); else derive from ``scene_id``."""
    sfn = raw_batch.get("scene_folder_name")
    if sfn is None:
        return _scene_dir_str(scene_id_fallback)
    if isinstance(sfn, torch.Tensor):
        if sfn.numel() == 0:
            return _scene_dir_str(scene_id_fallback)
        return f"{int(sfn.view(-1)[0].item()):03d}"
    return str(sfn).strip()


def _nuscenes_cam_id_suffix(pixel_camera_ids: List[int], cam_slot_idx: int) -> str:
    """``cam_slot_idx`` is the index into ``data.pixel_source.cameras`` (same as batch ``cam_indices``)."""
    ci = int(cam_slot_idx)
    if 0 <= ci < len(pixel_camera_ids):
        return f"_nuscam{int(pixel_camera_ids[ci])}"
    return ""


def _save_train_monitor_triplets(
    *,
    step: int,
    pred_rgbs: List[torch.Tensor],
    gt_images: List[torch.Tensor],
    raw_batch: Dict[str, Any],
    log_dir: str,
    block_idx_global: int,
    scene_id_fallback: Any,
    pixel_camera_ids: List[int],
) -> None:
    if len(pred_rgbs) == 0 or len(gt_images) == 0:
        return
    out_dir = os.path.join(log_dir, "images", "train")
    tgt_meta = raw_batch.get("target") or {}
    fi_t = tgt_meta.get("frame_indices")
    ci_t = tgt_meta.get("cam_indices")
    sc_lab = _scene_folder_label_from_batch(raw_batch, scene_id_fallback)
    safe_block_idx = int(block_idx_global)
    if safe_block_idx < 0:
        safe_block_idx = int(step)
    for v in range(min(len(pred_rgbs), len(gt_images))):
        if fi_t is not None and ci_t is not None and int(fi_t.shape[0]) > v and int(ci_t.shape[0]) > v:
            f_lab = int(fi_t[v].item())
            c_lab = int(ci_t[v].item())
            nusc_suf = _nuscenes_cam_id_suffix(pixel_camera_ids, c_lab)
            vsuf = f"b{safe_block_idx:06d}_sc{sc_lab}_v{v}_f{f_lab:05d}_c{c_lab}{nusc_suf}"
        else:
            vsuf = f"b{safe_block_idx:06d}_sc{sc_lab}_view{v}"
        _save_image_triplet(
            step,
            pred_rgbs[v],
            gt_images[v],
            out_dir,
            view_suffix=vsuf,
            save_error=False,
        )


def _safe_image_role(role: Any) -> str:
    text = str(role or "view").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "view"


def _hwc01_for_tb(img: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(torch.nan_to_num(img.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if x.dim() != 3:
        raise ValueError(f"expected image tensor [H,W,C], got {tuple(x.shape)}")
    if int(x.shape[-1]) == 1:
        x = x.expand(*x.shape[:-1], 3)
    if int(x.shape[-1]) != 3:
        raise ValueError(f"expected image tensor channel dim=3, got {tuple(x.shape)}")
    return x


def _save_iforward_train_images(
    *,
    step: int,
    pred_rgbs: List[torch.Tensor],
    gt_images: List[torch.Tensor],
    image_refs: List[Any],
    image_roles: List[Any],
    raw_batch: Dict[str, Any],
    log_dir: str,
    block_idx_global: int,
    scene_id_fallback: Any,
    pixel_camera_ids: List[int],
    writer: Optional[Any] = None,
    max_tb_images: int = 12,
) -> None:
    if len(pred_rgbs) == 0 or len(gt_images) == 0:
        return
    out_dir = os.path.join(log_dir, "images", "iforward_train")
    sc_lab = _scene_folder_label_from_batch(raw_batch, scene_id_fallback)
    safe_block_idx = int(block_idx_global)
    if safe_block_idx < 0:
        safe_block_idx = int(step)
    n = min(len(pred_rgbs), len(gt_images))
    for v in range(n):
        role = _safe_image_role(image_roles[v] if v < len(image_roles) else "view")
        ref = image_refs[v] if v < len(image_refs) else None
        try:
            f_lab = int(ref[0])  # type: ignore[index]
            c_lab = int(ref[1])  # type: ignore[index]
            nusc_suf = _nuscenes_cam_id_suffix(pixel_camera_ids, c_lab)
            vsuf = f"b{safe_block_idx:06d}_sc{sc_lab}_v{v}_{role}_f{f_lab:05d}_c{c_lab}{nusc_suf}"
        except Exception:
            vsuf = f"b{safe_block_idx:06d}_sc{sc_lab}_v{v}_{role}"
        _save_image_triplet(
            step,
            pred_rgbs[v],
            gt_images[v],
            out_dir,
            view_suffix=vsuf,
            save_error=True,
        )
        if writer is None or int(v) >= int(max_tb_images):
            continue
        try:
            pred = _hwc01_for_tb(pred_rgbs[v])
            gt = _hwc01_for_tb(gt_images[v])
            err = (pred - gt).abs()
            max_err = float(err.max().item()) if err.numel() else 0.0
            if max_err > 0.0:
                err = err / max_err
            tag_base = f"iforward_train/{role}/view{int(v)}"
            writer.add_image(f"{tag_base}/pred", pred.permute(2, 0, 1), int(step))
            writer.add_image(f"{tag_base}/gt", gt.permute(2, 0, 1), int(step))
            writer.add_image(f"{tag_base}/error", err.permute(2, 0, 1), int(step))
        except Exception as exc:
            logger.warning("Failed to write IForward TensorBoard image view=%s role=%s: %s", int(v), role, exc)


def _save_train_images_for_result(
    *,
    step: int,
    result: Dict[str, Any],
    pred_rgbs: List[torch.Tensor],
    gt_images: List[torch.Tensor],
    raw_batch: Dict[str, Any],
    log_dir: str,
    block_idx_global: int,
    scene_id_fallback: Any,
    pixel_camera_ids: List[int],
    writer: Optional[Any] = None,
) -> None:
    image_refs = result.get("image_refs")
    image_roles = result.get("image_roles")
    if isinstance(image_refs, list) and isinstance(image_roles, list):
        _save_iforward_train_images(
            step=step,
            pred_rgbs=pred_rgbs,
            gt_images=gt_images,
            image_refs=image_refs,
            image_roles=image_roles,
            raw_batch=raw_batch,
            log_dir=log_dir,
            block_idx_global=block_idx_global,
            scene_id_fallback=scene_id_fallback,
            pixel_camera_ids=pixel_camera_ids,
            writer=writer,
        )
        return
    _save_train_monitor_triplets(
        step=step,
        pred_rgbs=pred_rgbs,
        gt_images=gt_images,
        raw_batch=raw_batch,
        log_dir=log_dir,
        block_idx_global=block_idx_global,
        scene_id_fallback=scene_id_fallback,
        pixel_camera_ids=pixel_camera_ids,
    )


def _map_to_01(map_tensor: torch.Tensor) -> torch.Tensor:
    x = map_tensor.detach().float().cpu()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.dim() == 3 and int(x.shape[-1]) == 1:
        x = x.squeeze(-1)
    if x.numel() == 0:
        return x
    mn = float(x.min().item())
    mx = float(x.max().item())
    if mx > mn:
        return (x - mn) / (mx - mn)
    return x * 0.0


def _save_stage5_5_aux_debug_maps(
    *,
    step: int,
    aux_debug_maps: Optional[Dict[str, Any]],
    raw_batch: Dict[str, Any],
    log_dir: str,
    block_idx_global: int,
    scene_id_fallback: Any,
    writer: Optional[Any] = None,
) -> None:
    if not isinstance(aux_debug_maps, dict) or int(block_idx_global) < 1:
        return

    out_dir = os.path.join(log_dir, "images", "train_aux_uncertainty")
    os.makedirs(out_dir, exist_ok=True)
    sc_lab = _scene_folder_label_from_batch(raw_batch, scene_id_fallback)
    frame_idx = int(aux_debug_maps.get("frame_idx", -1))
    target_index = int(aux_debug_maps.get("target_index", 0))
    prefix = (
        f"step{int(step):06d}_b{int(block_idx_global):06d}_"
        f"sc{sc_lab}_aux{target_index}_f{frame_idx:05d}"
    )

    def _save_rgb(name: str, tensor: torch.Tensor) -> None:
        img = torch.clamp(tensor.detach().float().cpu(), 0.0, 1.0)
        arr = (img.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        try:
            from PIL import Image
        except ImportError:
            np.save(path.replace(".png", ".npy"), arr)
            return
        Image.fromarray(arr).save(path)

    def _save_signed_rgb(name: str, tensor: torch.Tensor, *, scale: float = 0.5) -> Optional[torch.Tensor]:
        x = tensor.detach().float().cpu()
        if x.dim() != 3 or int(x.shape[-1]) != 3:
            return None
        s = max(float(scale), 1.0e-6)
        img = (0.5 + 0.5 * x.clamp(-s, s) / s).clamp(0.0, 1.0)
        _save_rgb(name, img)
        return img

    def _save_map(name: str, tensor: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        img = _map_to_01(tensor) if normalize else torch.clamp(tensor.detach().float().cpu(), 0.0, 1.0)
        arr = (img.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        try:
            from PIL import Image
        except ImportError:
            np.save(path.replace(".png", ".npy"), arr)
            return img
        Image.fromarray(arr).save(path)
        return img

    gt = aux_debug_maps.get("gt")
    render = aux_debug_maps.get("render")
    if torch.is_tensor(gt):
        _save_rgb("gt", gt)
    if torch.is_tensor(render):
        _save_rgb("nearby_render", render)

    pred_rgb_residual = aux_debug_maps.get("pred_rgb_residual")
    gt_rgb_residual = aux_debug_maps.get("gt_rgb_residual")
    residual_vis_scale = max(float(aux_debug_maps.get("rgb_residual_max", 0.5) or 0.5), 1.0e-6)
    pred_residual_vis: Optional[torch.Tensor] = None
    gt_residual_vis: Optional[torch.Tensor] = None
    render_plus_pred_residual: Optional[torch.Tensor] = None
    render_plus_gt_residual: Optional[torch.Tensor] = None
    if torch.is_tensor(pred_rgb_residual):
        pred_residual_vis = _save_signed_rgb(
            "pred_rgb_residual_signed",
            pred_rgb_residual,
            scale=residual_vis_scale,
        )
    if torch.is_tensor(gt_rgb_residual):
        gt_residual_vis = _save_signed_rgb(
            "gt_rgb_residual_signed",
            gt_rgb_residual,
            scale=residual_vis_scale,
        )
    if torch.is_tensor(render) and torch.is_tensor(pred_rgb_residual):
        render_cpu = render.detach().float().cpu()
        residual_cpu = pred_rgb_residual.detach().float().cpu()
        if tuple(render_cpu.shape) == tuple(residual_cpu.shape):
            render_plus_pred_residual = torch.clamp(render_cpu + residual_cpu, 0.0, 1.0)
            _save_rgb("nearby_render_plus_pred_rgb_residual", render_plus_pred_residual)
    if torch.is_tensor(render) and torch.is_tensor(gt_rgb_residual):
        render_cpu = render.detach().float().cpu()
        residual_cpu = gt_rgb_residual.detach().float().cpu()
        if tuple(render_cpu.shape) == tuple(residual_cpu.shape):
            render_plus_gt_residual = torch.clamp(render_cpu + residual_cpu, 0.0, 1.0)
            _save_rgb("nearby_render_plus_gt_rgb_residual", render_plus_gt_residual)

    image_maps: Dict[str, torch.Tensor] = {}
    for name, normalize in (
        ("abs_error", True),
        ("pred_error", True),
        ("support", True),
        ("loss_mask", False),
        ("error_confidence", False),
        ("support_confidence", False),
        ("confidence", False),
        ("bridge_mask", False),
        ("bridge_weight", False),
    ):
        val = aux_debug_maps.get(name)
        if torch.is_tensor(val):
            image_maps[name] = _save_map(name, val, normalize=bool(normalize))

    if writer is None:
        return
    if torch.is_tensor(render):
        writer.add_image(
            "train_aux_uncertainty/nearby_render",
            torch.clamp(render.detach().float().cpu(), 0.0, 1.0).permute(2, 0, 1),
            int(step),
        )
    if torch.is_tensor(gt):
        writer.add_image(
            "train_aux_uncertainty/gt",
            torch.clamp(gt.detach().float().cpu(), 0.0, 1.0).permute(2, 0, 1),
            int(step),
        )
    if torch.is_tensor(render_plus_pred_residual):
        writer.add_image(
            "train_aux_uncertainty/nearby_render_plus_pred_rgb_residual",
            render_plus_pred_residual.permute(2, 0, 1),
            int(step),
        )
    if torch.is_tensor(render_plus_gt_residual):
        writer.add_image(
            "train_aux_uncertainty/nearby_render_plus_gt_rgb_residual",
            render_plus_gt_residual.permute(2, 0, 1),
            int(step),
        )
    if torch.is_tensor(pred_residual_vis):
        writer.add_image(
            "train_aux_uncertainty/pred_rgb_residual_signed",
            pred_residual_vis.permute(2, 0, 1),
            int(step),
        )
    if torch.is_tensor(gt_residual_vis):
        writer.add_image(
            "train_aux_uncertainty/gt_rgb_residual_signed",
            gt_residual_vis.permute(2, 0, 1),
            int(step),
        )
    for name, img in image_maps.items():
        writer.add_image(f"train_aux_uncertainty/{name}", img.unsqueeze(0), int(step))


def _stage5_6_debug_view_suffix(
    *,
    block_idx_global: int,
    scene_label: str,
    image_pack: Dict[str, Any],
    pixel_camera_ids: List[int],
    prefix: str,
) -> str:
    target_index = int(image_pack.get("target_index", 0))
    frame_idx = int(image_pack.get("frame_idx", -1))
    cam_idx = int(image_pack.get("cam_idx", -1))
    role_prefix = str(image_pack.get("role", prefix)).strip() or str(prefix)
    if cam_idx >= 0:
        nusc_suf = _nuscenes_cam_id_suffix(pixel_camera_ids, cam_idx)
        return (
            f"b{int(block_idx_global):06d}_sc{scene_label}_{role_prefix}{target_index}_"
            f"f{frame_idx:05d}_c{cam_idx}{nusc_suf}"
        )
    return f"b{int(block_idx_global):06d}_sc{scene_label}_{role_prefix}{target_index}_f{frame_idx:05d}"


def _save_stage5_6_debug_images(
    *,
    step: int,
    result: Dict[str, Any],
    raw_batch: Dict[str, Any],
    log_dir: str,
    block_idx_global: int,
    scene_id_fallback: Any,
    pixel_camera_ids: List[int],
) -> None:
    if int(block_idx_global) < 1:
        return
    sc_lab = _scene_folder_label_from_batch(raw_batch, scene_id_fallback)

    nearby_images = result.get("_stage5_6_nearby_debug_images")
    if isinstance(nearby_images, list):
        out_train = os.path.join(log_dir, "images", "train")
        for item in nearby_images:
            if not isinstance(item, dict):
                continue
            pred = item.get("pred")
            gt = item.get("gt")
            if not torch.is_tensor(pred) or not torch.is_tensor(gt):
                continue
            vsuf = _stage5_6_debug_view_suffix(
                block_idx_global=int(block_idx_global),
                scene_label=sc_lab,
                image_pack=item,
                pixel_camera_ids=pixel_camera_ids,
                prefix="nearby",
            )
            _save_image_triplet(
                int(step),
                pred,
                gt,
                out_train,
                view_suffix=vsuf,
                save_error=False,
            )

    error_images = result.get("_stage5_6_error_debug_images")
    if not isinstance(error_images, list):
        return
    out_error = os.path.join(log_dir, "images", "error")
    os.makedirs(out_error, exist_ok=True)

    def _save_rgb(path: str, tensor: torch.Tensor) -> None:
        img = torch.clamp(tensor.detach().float().cpu(), 0.0, 1.0)
        arr = (img.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        try:
            from PIL import Image
        except ImportError:
            np.save(path.replace(".png", ".npy"), arr)
            return
        Image.fromarray(arr).save(path)

    def _save_map(path: str, tensor: torch.Tensor) -> None:
        img = torch.clamp(tensor.detach().float().cpu(), 0.0, 1.0)
        if img.dim() == 3 and int(img.shape[-1]) == 1:
            img = img.squeeze(-1)
        arr = (img.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        try:
            from PIL import Image
        except ImportError:
            np.save(path.replace(".png", ".npy"), arr)
            return
        Image.fromarray(arr).save(path)

    for item in error_images:
        if not isinstance(item, dict):
            continue
        render = item.get("render")
        pred_error = item.get("pred_error")
        actual_error = item.get("actual_error")
        if not torch.is_tensor(render) or not torch.is_tensor(pred_error) or not torch.is_tensor(actual_error):
            continue
        prefix = "step%06d_%s" % (
            int(step),
            _stage5_6_debug_view_suffix(
                block_idx_global=int(block_idx_global),
                scene_label=sc_lab,
                image_pack=item,
                pixel_camera_ids=pixel_camera_ids,
                prefix="nearby",
            ),
        )
        _save_rgb(os.path.join(out_error, f"{prefix}_render.png"), render)
        _save_map(os.path.join(out_error, f"{prefix}_error.png"), pred_error)
        _save_map(os.path.join(out_error, f"{prefix}_actual_error.png"), actual_error)


def _stage5_6_debug_images_due(
    *,
    image_trigger_mode: str,
    step: int,
    scheduler_info: Dict[str, Any],
    step_events: List[Dict[str, Any]],
    image_trigger_interval_steps: int,
    image_interval_blocks_equiv: int,
) -> bool:
    if image_trigger_mode == "raw_step_interval":
        scheduler_global_step = int(scheduler_info.get("global_step", int(step) + 1))
        return scheduler_global_step > 0 and scheduler_global_step % int(image_trigger_interval_steps) == 0
    if image_trigger_mode == "episode_end":
        if not any(ev.get("type") == "episode_end" for ev in step_events):
            return False
        completed_blocks = int(scheduler_info.get("block_idx_global", -1)) + 1
        return completed_blocks > 0 and completed_blocks % int(image_interval_blocks_equiv) == 0
    if image_trigger_mode == "block_end":
        for ev in step_events:
            if ev.get("type") != "block_end":
                continue
            block_idx_global = int(ev.get("block_idx_global", 0))
            if block_idx_global >= 1 and (block_idx_global - 1) % int(image_interval_blocks_equiv) == 0:
                return True
    return False


def _build_scheduler_node_sync_v8_fallback(
    cfg: Any,
    scheduler_info: Dict[str, Any],
    step_events: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    sv8 = cfg.get("scheduler_v8")
    if sv8 is None:
        return None
    execution = sv8.get("execution") if hasattr(sv8, "get") else None
    block_order = str(scheduler_info.get("block_order", "block_major")).strip()
    if execution is not None and hasattr(execution, "get"):
        reset_policy = str(
            execution.get(
                "reset_policy",
                "episode_end" if block_order == "step_major" else "block_end",
            )
        ).strip()
    else:
        reset_policy = "episode_end" if block_order == "step_major" else "block_end"
    if reset_policy not in ("block_end", "episode_end", "never"):
        raise ValueError(
            "scheduler_v8.execution.reset_policy must be one of ['block_end', 'episode_end', 'never']"
        )
    if block_order == "step_major" and reset_policy == "block_end":
        raise ValueError(
            "scheduler_v8.execution.reset_policy=block_end is incompatible with execution.block_order=step_major; "
            "use reset_policy=episode_end or never."
        )
    if reset_policy == "block_end":
        should_reset = any(ev.get("type") == "block_end" for ev in step_events)
    elif reset_policy == "episode_end":
        should_reset = any(ev.get("type") == "episode_end" for ev in step_events)
    else:
        should_reset = False
    U = int(scheduler_info.get("U", 1))
    seg = int(scheduler_info.get("segment_local_step", 0))
    if U < 1:
        raise ValueError("scheduler_v8 scheduler_info.U must be >= 1 for model_node_state sync.")
    return {
        "U": int(U),
        "segment_local_step": int(seg),
        "reset_after_block": bool(should_reset),
        "reset_policy": str(reset_policy),
    }


def _build_scheduler_node_sync_v9_fallback(
    cfg: Any,
    scheduler_info: Dict[str, Any],
    step_events: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    sv9 = cfg.get("scheduler_v9")
    if sv9 is None or not bool(sv9.get("enable", False)):
        return None
    execution = sv9.get("execution") if hasattr(sv9, "get") else None
    block_order = str(scheduler_info.get("block_order", "step_major")).strip()
    if execution is not None and hasattr(execution, "get"):
        reset_policy = str(execution.get("reset_policy", "episode_end")).strip()
    else:
        reset_policy = "episode_end"
    if reset_policy not in ("episode_end", "never"):
        raise ValueError("scheduler_v9.execution.reset_policy must be one of ['episode_end', 'never']")
    should_reset = any(ev.get("type") == "episode_end" for ev in step_events) if reset_policy == "episode_end" else False
    U = int(scheduler_info.get("U", 1))
    seg = int(scheduler_info.get("segment_local_step", 0))
    if U < 1:
        raise ValueError("scheduler_v9 scheduler_info.U must be >= 1 for model_node_state sync.")
    return {
        "U": int(U),
        "segment_local_step": int(seg),
        "reset_after_block": bool(should_reset),
        "reset_policy": str(reset_policy),
        "scheduler_version": "v9",
        "block_order": str(block_order),
    }


def _build_scheduler_node_sync_long_phase_b_fallback(
    cfg: Any,
    scheduler_info: Dict[str, Any],
    step_events: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    slong = cfg.get("scheduler_long_phase_b") if hasattr(cfg, "get") else None
    if slong is None or not bool(slong.get("enable", False)):
        return None
    if str(scheduler_info.get("scheduler_version", "")) != "long_v1":
        return None
    execution = slong.get("execution") if hasattr(slong, "get") else None
    reset_policy = (
        str(execution.get("reset_policy", "episode_end")).strip()
        if execution is not None and hasattr(execution, "get")
        else "episode_end"
    )
    if reset_policy not in ("episode_end", "never"):
        raise ValueError("scheduler_long_phase_b.execution.reset_policy must be one of ['episode_end', 'never']")
    should_reset = any(ev.get("type") == "episode_end" for ev in step_events) if reset_policy == "episode_end" else False
    return {
        "U": int(scheduler_info.get("U", 1)),
        "segment_local_step": int(scheduler_info.get("segment_local_step", 0)),
        "reset_after_block": bool(should_reset),
        "reset_policy": str(reset_policy),
        "scheduler_version": "long_v1",
        "block_order": str(scheduler_info.get("block_order", "long_rollout")),
    }


def _node_state_cache_sizes(model: Any) -> Dict[str, int]:
    return {
        "bg": int(len(getattr(model, "node_states_bg", {}) or {})),
        "distant": int(len(getattr(model, "node_states_distant", {}) or {})),
        "rigid": int(len(getattr(model, "node_states_rigid", {}) or {})),
        "sky": int(len(getattr(model, "node_states_sky", {}) or {})),
    }


def _reset_model_node_state_and_release_cuda(
    model: Any,
    *,
    reason: str,
    step: int,
    scheduler_info: Optional[Dict[str, Any]] = None,
    log_reset: bool = True,
) -> Dict[str, Any]:
    before = _node_state_cache_sizes(model)
    model.reset_node_state()
    empty_cache = str(os.environ.get("STAGE6_EMPTY_CACHE_ON_RESET", "")).lower() in {"1", "true", "yes", "on"}
    if empty_cache:
        gc.collect()
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = _node_state_cache_sizes(model)
    scheduler_info = scheduler_info or {}
    if bool(log_reset):
        logger.info(
            "NODE_STATE_RESET step=%s reason=%s scene_id=%s scene_dir=%s segment=%s "
            "before=%s after=%s cuda_empty_cache=%s",
            int(step),
            str(reason),
            scheduler_info.get("scene_id", -1),
            _scene_dir_str(scheduler_info.get("scene_id", -1)),
            scheduler_info.get("segment_id", -1),
            before,
            after,
            bool(empty_cache and torch.cuda.is_available()),
        )
    return {
        "before": before,
        "after": after,
        "cuda_empty_cache": bool(empty_cache and torch.cuda.is_available()),
    }


def _drop_result_tensor_payloads(result: Optional[Dict[str, Any]]) -> None:
    if result is None:
        return
    for key, value in list(result.items()):
        if torch.is_tensor(value):
            if value.dim() == 0:
                continue
            result[key] = None
        elif isinstance(value, list) and any(torch.is_tensor(x) for x in value):
            result[key] = []
        elif isinstance(value, tuple) and any(torch.is_tensor(x) for x in value):
            result[key] = ()
        elif isinstance(value, dict) and any(torch.is_tensor(x) for x in value.values()):
            result[key] = {}


def _record_cuda_storage(
    tensor: Any,
    storages: Dict[int, int],
    *,
    meta: Optional[Dict[int, str]] = None,
) -> int:
    if not torch.is_tensor(tensor) or not tensor.is_cuda:
        return 0
    try:
        storage = tensor.untyped_storage()
        ptr = int(storage.data_ptr())
        nbytes = int(storage.nbytes())
    except Exception:
        ptr = int(tensor.data_ptr())
        nbytes = int(tensor.numel() * tensor.element_size())
    if ptr in storages:
        return 0
    storages[ptr] = int(nbytes)
    if meta is not None:
        try:
            shape = tuple(int(x) for x in tensor.shape)
        except Exception:
            shape = ()
        meta[ptr] = (
            f"{float(nbytes) / (1024.0 ** 2):.1f}MiB "
            f"shape={shape} dtype={str(tensor.dtype)} grad={bool(getattr(tensor, 'requires_grad', False))}"
        )
    return int(nbytes)


def _collect_cuda_storages_from_object(
    obj: Any,
    storages: Dict[int, int],
    *,
    max_depth: int = 6,
    max_items: int = 200000,
) -> int:
    seen: set[int] = set()
    visited = 0

    def walk(value: Any, depth: int) -> None:
        nonlocal visited
        if depth < 0 or visited >= max_items:
            return
        if torch.is_tensor(value):
            _record_cuda_storage(value, storages)
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        oid = id(value)
        if oid in seen:
            return
        seen.add(oid)
        visited += 1
        if isinstance(value, dict):
            for child in value.values():
                walk(child, depth - 1)
            return
        if isinstance(value, (list, tuple, set, deque)):
            for child in value:
                walk(child, depth - 1)
            return
        slots = getattr(value, "__slots__", None)
        if slots:
            for name in slots:
                if hasattr(value, name):
                    walk(getattr(value, name), depth - 1)
        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, dict):
            for child in attrs.values():
                walk(child, depth - 1)

    walk(obj, int(max_depth))
    return int(visited)


def _cuda_storage_bytes_from_tensors(tensors: Any) -> int:
    storages: Dict[int, int] = {}
    for tensor in list(tensors):
        _record_cuda_storage(tensor, storages)
    return int(sum(storages.values()))


def _optimizer_cuda_state_bytes(optimizer: Any) -> int:
    storages: Dict[int, int] = {}
    state = getattr(optimizer, "state", None)
    if isinstance(state, dict):
        for value in state.values():
            _collect_cuda_storages_from_object(value, storages, max_depth=4)
    return int(sum(storages.values()))


def _dataset_cuda_cache_bytes(dataset: Any) -> int:
    storages: Dict[int, int] = {}
    for name in (
        "_view_pack_cache",
        "_scene_asset_cache",
        "_segment_static_cache",
        "_image_meta_cache",
        "_egocar_mask_cache",
        "_preload_manager",
    ):
        if hasattr(dataset, name):
            _collect_cuda_storages_from_object(getattr(dataset, name), storages, max_depth=5)
    return int(sum(storages.values()))


def _stage6_runtime_cuda_cache_summary(model: Any) -> Dict[str, int]:
    runtime = getattr(getattr(getattr(model, "model", None), "bridge", None), "runtime", None)
    if runtime is None:
        return {
            "cuda_component/stage6_runtime_node_cache_bytes": 0,
            "cuda_component/stage6_runtime_node_cache_bg": 0,
            "cuda_component/stage6_runtime_node_cache_distant": 0,
            "cuda_component/stage6_runtime_node_cache_rigid": 0,
            "cuda_component/stage6_runtime_node_cache_sky": 0,
        }
    storages: Dict[int, int] = {}
    counts: Dict[str, int] = {}
    for branch, name in (
        ("bg", "node_states_bg"),
        ("distant", "node_states_distant"),
        ("rigid", "node_states_rigid"),
        ("sky", "node_states_sky"),
        ("h_bg", "h_cache_bg"),
        ("h_distant", "h_cache_distant"),
        ("h_rigid", "h_cache_rigid"),
        ("h_sky", "h_cache_sky"),
    ):
        cache = getattr(runtime, name, None)
        counts[branch] = int(len(cache) if hasattr(cache, "__len__") else 0)
        _collect_cuda_storages_from_object(cache, storages, max_depth=6)
    return {
        "cuda_component/stage6_runtime_node_cache_bytes": int(sum(storages.values())),
        "cuda_component/stage6_runtime_node_cache_bg": int(counts.get("bg", 0)),
        "cuda_component/stage6_runtime_node_cache_distant": int(counts.get("distant", 0)),
        "cuda_component/stage6_runtime_node_cache_rigid": int(counts.get("rigid", 0)),
        "cuda_component/stage6_runtime_node_cache_sky": int(counts.get("sky", 0)),
        "cuda_component/stage6_runtime_h_cache_bg": int(counts.get("h_bg", 0)),
        "cuda_component/stage6_runtime_h_cache_distant": int(counts.get("h_distant", 0)),
        "cuda_component/stage6_runtime_h_cache_rigid": int(counts.get("h_rigid", 0)),
        "cuda_component/stage6_runtime_h_cache_sky": int(counts.get("h_sky", 0)),
    }


def _cuda_live_tensor_summary(*, topk: int = 12) -> Dict[str, Any]:
    gc.collect()
    storages: Dict[int, int] = {}
    meta: Dict[int, str] = {}
    tensor_objects = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor_objects += 1
                _record_cuda_storage(obj, storages, meta=meta)
        except Exception:
            continue
    top_items = sorted(storages.items(), key=lambda kv: kv[1], reverse=True)[: max(0, int(topk))]
    return {
        "cuda_live/tensor_objects": int(tensor_objects),
        "cuda_live/unique_storages": int(len(storages)),
        "cuda_live/storage_bytes": int(sum(storages.values())),
        "cuda_live/top_storages": " | ".join(meta.get(ptr, f"{nbytes}B") for ptr, nbytes in top_items),
    }


def _cuda_component_memory_summary(
    *,
    model: Any,
    dataset: Any,
    include_live: bool,
    topk: int,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    buffers = list(model.buffers()) if hasattr(model, "buffers") else []
    state_cache = getattr(model, "_state_cache", None)
    state_cache_storages: Dict[int, int] = {}
    if state_cache is not None:
        _collect_cuda_storages_from_object(state_cache, state_cache_storages, max_depth=8)
    summary: Dict[str, Any] = {
        "cuda_component/model_param_bytes": int(_cuda_storage_bytes_from_tensors(params)),
        "cuda_component/model_buffer_bytes": int(_cuda_storage_bytes_from_tensors(buffers)),
        "cuda_component/optimizer_state_bytes": int(_optimizer_cuda_state_bytes(getattr(model, "optimizer", None))),
        "cuda_component/iforward_state_cache_bytes": int(sum(state_cache_storages.values())),
        "cuda_component/dataset_cache_bytes": int(_dataset_cuda_cache_bytes(dataset)),
    }
    summary.update(_stage6_runtime_cuda_cache_summary(model))
    if include_live:
        summary.update(_cuda_live_tensor_summary(topk=int(topk)))
    return summary


@dataclass(frozen=True)
class _BatchRequestValidationV7:
    scene_id: int
    segment_id: int
    source_image_ref: Tuple[int, int]
    target_image_refs: List[Tuple[int, int]]
    source_image_refs: Optional[List[Tuple[int, int]]] = None
    include_test: bool = False
    test_image_refs: Optional[List[Tuple[int, int]]] = None


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _compute_masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask_hw: torch.Tensor) -> Optional[float]:
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"masked PSNR shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
    if mask_hw.shape != pred.shape[:2]:
        raise ValueError(
            f"masked PSNR mask shape mismatch: mask={tuple(mask_hw.shape)} pred_hw={tuple(pred.shape[:2])}"
        )
    mask = mask_hw.to(pred.device).float()
    valid = float(mask.sum().item())
    if valid <= 0.0:
        return None
    diff2 = ((torch.clamp(pred, 0.0, 1.0) - torch.clamp(gt, 0.0, 1.0)) ** 2) * mask.unsqueeze(-1)
    mse = diff2.sum() / (mask.sum() * 3.0)
    mse_val = float(mse.item())
    if mse_val <= 0.0:
        return float("inf")
    return float(-10.0 * np.log10(mse_val))


def _compute_masked_ssim(pred: torch.Tensor, gt: torch.Tensor, mask_hw: torch.Tensor) -> Optional[float]:
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"masked SSIM shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
    if mask_hw.shape != pred.shape[:2]:
        raise ValueError(
            f"masked SSIM mask shape mismatch: mask={tuple(mask_hw.shape)} pred_hw={tuple(pred.shape[:2])}"
        )
    valid = float(mask_hw.float().sum().item())
    if valid <= 0.0:
        return None
    loss = compute_ssim_loss_masked(
        pred,
        gt,
        valid_mask=mask_hw.to(pred.device).float(),
        sky_mask=None,
        data_range=1.0,
    )
    return float((1.0 - float(loss.item())))


def _compute_masked_lpips(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask_hw: torch.Tensor,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
) -> Optional[float]:
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"masked LPIPS shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
    if mask_hw.shape != pred.shape[:2]:
        raise ValueError(
            f"masked LPIPS mask shape mismatch: mask={tuple(mask_hw.shape)} pred_hw={tuple(pred.shape[:2])}"
        )
    mask = mask_hw.to(pred.device).float().clamp(0.0, 1.0)
    valid = float(mask.sum().item())
    if valid <= 0.0:
        return None
    pred_c = torch.clamp(pred, 0.0, 1.0)
    gt_c = torch.clamp(gt, 0.0, 1.0)
    mask_3 = mask.unsqueeze(-1)
    pred_masked = pred_c * mask_3 + gt_c * (1.0 - mask_3)
    lp = lpips_metric(
        pred_masked.permute(2, 0, 1).unsqueeze(0),
        gt_c.permute(2, 0, 1).unsqueeze(0),
    )
    return float(lp.item())


def _snapshot_train_checkpoint_bytes(model: Any) -> bytes:
    if not hasattr(model, "optimizer") or model.optimizer is None:
        raise ValueError("validation segment_finetune_train requires model.optimizer")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
    }
    if hasattr(model, "build_light_checkpoint_extra"):
        payload.update(model.build_light_checkpoint_extra(step=int(getattr(model.optimizer, "global_step", 0))))
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _restore_train_checkpoint_bytes(model: Any, ckpt_bytes: bytes, device: torch.device) -> None:
    if not hasattr(model, "optimizer") or model.optimizer is None:
        raise ValueError("validation segment_finetune_train requires model.optimizer")
    payload = torch.load(io.BytesIO(ckpt_bytes), map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if hasattr(model, "load_optimizer_state_from_checkpoint"):
        loaded = bool(model.load_optimizer_state_from_checkpoint(payload))
        if not loaded and payload.get("optimizer_state_dict") is not None:
            logger.warning("Skipped optimizer restore in _restore_train_checkpoint_bytes due to signature mismatch.")
    else:
        model.optimizer.load_state_dict(payload["optimizer_state_dict"])


def _iter_episode_block_indices(
    *,
    blocks_per_episode: int,
    steps_per_block: int,
    block_order: str,
    step_major_switch_interval_steps: int = 1,
) -> List[int]:
    if block_order == "block_major":
        return [
            int(b)
            for b in range(int(blocks_per_episode))
            for _ in range(int(steps_per_block))
        ]
    if block_order == "step_major":
        switch_every = int(step_major_switch_interval_steps)
        if switch_every < 1:
            raise ValueError("step_major_switch_interval_steps must be >= 1")
        out: List[int] = []
        for round_base in range(0, int(steps_per_block), int(switch_every)):
            chunk = int(min(int(switch_every), int(steps_per_block) - int(round_base)))
            for b in range(int(blocks_per_episode)):
                out.extend([int(b)] * int(chunk))
        return out
    raise ValueError(f"unsupported block_order={block_order!r}")


def _run_validation_v7_round(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    specs: List[ValidationEpisodeSpecV7],
    validation_cfg: ValidationV7Config,
    device: torch.device,
    trigger_train_episode_counter: int,
    trigger_step: int,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: SSIM,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    metrics_fh: Optional[TextIO],
    writer: Optional[Any] = None,
) -> None:
    if len(specs) == 0:
        logger.warning("validation_v7 enabled but no valid episode specs from eval_scene_ids")
        return
    logger.info(
        "VALIDATION_V7_BEGIN trigger_episode_counter=%s trigger_step=%s num_specs=%s",
        int(trigger_train_episode_counter),
        int(trigger_step),
        int(len(specs)),
    )
    val_root = os.path.join(cfg.log_dir, str(validation_cfg.save_dir))
    os.makedirs(val_root, exist_ok=True)
    validation_mode = str(validation_cfg.mode)
    steps_per_block = int(validation_cfg.steps_per_block)
    validation_block_order = str(validation_cfg.block_order)
    step_major_switch_interval_steps = int(validation_cfg.step_major_switch_interval_steps)
    use_train_finetune = validation_mode == "segment_finetune_train"
    if steps_per_block < 1:
        raise ValueError(f"validation_v7.block.steps_per_block must be >= 1, got {steps_per_block}")
    if validation_block_order not in ("block_major", "step_major"):
        raise ValueError(
            "validation_v7.execution.block_order must be one of ['block_major', 'step_major'], "
            f"got {validation_block_order!r}"
        )
    if step_major_switch_interval_steps < 1:
        raise ValueError(
            "validation_v7.execution.step_major_switch_interval_steps must be >= 1, "
            f"got {step_major_switch_interval_steps}"
        )

    infer_policy = RuntimePolicy(
        do_backward=False,
        do_optimizer_step=False,
        update_hidden_cache=True,
        writeback_node_state=True,
        reset_node_state_after_block=False,
    )
    train_policy = RuntimePolicy(
        do_backward=True,
        do_optimizer_step=True,
        update_hidden_cache=True,
        writeback_node_state=True,
        reset_node_state_after_block=False,
    )
    base_ckpt_bytes: Optional[bytes] = None
    if use_train_finetune:
        base_ckpt_bytes = _snapshot_train_checkpoint_bytes(model)
    train_step_supports_runtime_policy = "runtime_policy" in inspect.signature(model.train_step).parameters
    infer_step_supports_runtime_policy = (
        "runtime_policy" in inspect.signature(model.inference_step_from_train_batch).parameters
    )

    all_episode_rows: List[Dict[str, Any]] = []
    try:
        for spec in specs:
            if use_train_finetune:
                _restore_train_checkpoint_bytes(model, base_ckpt_bytes, device)

            _reset_model_node_state_and_release_cuda(
                model,
                reason="validation_v7_episode_begin",
                step=int(trigger_step),
                scheduler_info={
                    "scene_id": int(spec.scene_id),
                    "segment_id": int(spec.segment_id),
                },
                log_reset=bool((cfg.get("logging") or {}).get("log_node_state_reset", True)),
            )
            validation_local_step = 0
            last_minimal: Optional[Dict[str, Any]] = None
            block_payloads: List[Dict[str, Any]] = []
            for block_idx_in_episode, block_frames in enumerate(spec.block_windows):
                src_frame = int(block_frames[0])
                source_ref = (int(src_frame), 0)
                source_refs = [(int(src_frame), int(cam_id)) for cam_id in range(int(spec.num_cams))]
                target_refs: List[Tuple[int, int]] = []
                for frame_idx in block_frames:
                    for cam_id in range(int(spec.num_cams)):
                        target_refs.append((int(frame_idx), int(cam_id)))
                req = _BatchRequestValidationV7(
                    scene_id=int(spec.scene_id),
                    segment_id=int(spec.segment_id),
                    source_image_ref=source_ref,
                    source_image_refs=source_refs,
                    target_image_refs=target_refs,
                    include_test=False,
                    test_image_refs=None,
                )
                raw_batch = dataset.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
                minimal_batch = convert_batch_to_minimal_format(
                    raw_batch,
                    device,
                    num_targets=int(raw_batch["target"]["image"].shape[0]),
                    include_source_for_2d=True,
                    view_selection=None,
                )
                block_payloads.append(
                    {
                        "block_idx": int(block_idx_in_episode),
                        "source_frame": int(src_frame),
                        "target_frames": [int(x) for x in block_frames],
                        "minimal_batch": minimal_batch,
                        "losses": [],
                    }
                )

            visit_order = _iter_episode_block_indices(
                blocks_per_episode=int(len(block_payloads)),
                steps_per_block=int(steps_per_block),
                block_order=str(validation_block_order),
                step_major_switch_interval_steps=int(step_major_switch_interval_steps),
            )
            for visit_idx, block_idx_in_episode in enumerate(visit_order):
                payload = block_payloads[int(block_idx_in_episode)]
                minimal_batch = payload["minimal_batch"]
                scheduler_node_sync = {
                    "U": 1,
                    "segment_local_step": int(validation_local_step + 1),
                    "reset_after_block": False,
                }
                if use_train_finetune:
                    train_step_kwargs: Dict[str, Any] = {
                        "batch": minimal_batch,
                        "step": None,
                        "profile_phase_timing": False,
                        "sync_cuda_timing": False,
                        "scheduler_node_sync": scheduler_node_sync,
                    }
                    if train_step_supports_runtime_policy:
                        train_step_kwargs["runtime_policy"] = train_policy
                    step_result = model.train_step(**train_step_kwargs)
                else:
                    infer_step_kwargs: Dict[str, Any] = {
                        "batch": minimal_batch,
                        "step": None,
                        "scheduler_node_sync": scheduler_node_sync,
                    }
                    if infer_step_supports_runtime_policy:
                        infer_step_kwargs["runtime_policy"] = infer_policy
                    step_result = model.inference_step_from_train_batch(**infer_step_kwargs)
                payload["losses"].append(float(step_result.get("loss", 0.0)))
                validation_local_step += 1
                last_minimal = minimal_batch
                logger.info(
                    "VALIDATION_V7_BLOCK_VISIT mode=%s block_order=%s scene_id=%s segment_id=%s "
                    "step_major_switch_interval_steps=%s block=%s visit=%s/%s source_frame=%s target_frames=%s loss=%.6f",
                    validation_mode,
                    validation_block_order,
                    int(spec.scene_id),
                    int(spec.segment_id),
                    int(step_major_switch_interval_steps),
                    int(block_idx_in_episode),
                    int(visit_idx + 1),
                    int(len(visit_order)),
                    int(payload["source_frame"]),
                    [int(x) for x in payload["target_frames"]],
                    float(payload["losses"][-1]),
                )

            for payload in block_payloads:
                block_loss_values = [float(x) for x in payload["losses"]]
                logger.info(
                    "VALIDATION_V7_BLOCK_SUMMARY mode=%s block_order=%s scene_id=%s segment_id=%s block=%s "
                    "step_major_switch_interval_steps=%s steps=%s source_frame=%s target_frames=%s mean_loss=%.6f",
                    validation_mode,
                    validation_block_order,
                    int(spec.scene_id),
                    int(spec.segment_id),
                    int(payload["block_idx"]),
                    int(step_major_switch_interval_steps),
                    int(len(block_loss_values)),
                    int(payload["source_frame"]),
                    [int(x) for x in payload["target_frames"]],
                    float(np.mean(block_loss_values)) if block_loss_values else 0.0,
                )

            if last_minimal is None:
                continue

            eval_req = _BatchRequestValidationV7(
                scene_id=int(spec.scene_id),
                segment_id=int(spec.segment_id),
                source_image_ref=(int(spec.frame_chain[0]), 0),
                source_image_refs=[(int(spec.frame_chain[0]), int(cam_id)) for cam_id in range(int(spec.num_cams))],
                target_image_refs=[(int(r[0]), int(r[1])) for r in spec.eval_image_refs],
                include_test=False,
                test_image_refs=None,
            )
            raw_eval = dataset.get_segment_batch_from_image_refs(eval_req, enforce_target0_equals_source=False)
            minimal_eval = convert_batch_to_minimal_format(
                raw_eval,
                device,
                num_targets=int(raw_eval["target"]["image"].shape[0]),
                include_source_for_2d=True,
                view_selection=None,
            )

            gs_state = model.export_3dgs_state(
                last_minimal,
                include_hidden=True,
                rigid_export_frame_idx=int(last_minimal["source_frame_idx"]),
            )
            scene_state = {
                "base_batch": minimal_eval,
                "gs_state": gs_state,
            }
            preds = model.render_views_from_scene_state(scene_state, minimal_eval["targets"])
            gts = [t["gt_image"] for t in minimal_eval["targets"]]
            if len(preds) != len(gts):
                raise ValueError(
                    f"validation render size mismatch: pred={len(preds)} gt={len(gts)} scene={spec.scene_id} seg={spec.segment_id}"
                )

            expected_views = int((len(spec.frame_chain)) * int(spec.num_cams))
            if len(preds) != expected_views:
                raise ValueError(
                    f"validation expected {(len(spec.frame_chain))}x{spec.num_cams} views={expected_views}, got={len(preds)}"
                )

            per_view_rows: List[Dict[str, Any]] = []
            psnr_vals: List[float] = []
            ssim_vals: List[float] = []
            lpips_vals: List[float] = []
            psnr_non_sky_vals: List[float] = []
            ssim_non_sky_vals: List[float] = []
            lpips_non_sky_vals: List[float] = []
            psnr_sky_vals: List[float] = []
            ssim_sky_vals: List[float] = []
            sky_coverage_vals: List[float] = []

            seg_dir = os.path.join(
                val_root,
                f"scene_{int(spec.scene_id):03d}",
                f"segment_{int(spec.segment_id):03d}",
                f"episode_start_{int(spec.episode_start_keyframe_pos):03d}",
            )
            render_dir = os.path.join(seg_dir, "renders")
            os.makedirs(render_dir, exist_ok=True)

            for idx, (pred, gt, tgt) in enumerate(zip(preds, gts, minimal_eval["targets"])):
                fallback_ref = eval_req.target_image_refs[int(idx)]
                m = _compute_metrics(
                    pred_rgb=pred,
                    gt_rgb=gt,
                    psnr_metric=psnr_metric,
                    ssim_metric=ssim_metric,
                    lpips_metric=lpips_metric,
                    compute_psnr=True,
                    compute_heavy=True,
                )
                psnr_vals.append(float(m["psnr"]))
                ssim_vals.append(float(m["ssim"]))
                lpips_vals.append(float(m["lpips"]))
                row = {
                    "index": int(idx),
                    "frame_idx": int(tgt.get("frame_idx", int(fallback_ref[0]))),
                    "cam_idx": int(tgt.get("cam_idx", int(fallback_ref[1]))),
                    "psnr": float(m["psnr"]),
                    "ssim": float(m["ssim"]),
                    "lpips": float(m["lpips"]),
                }

                if bool(validation_cfg.use_sky_mask_regions):
                    sky_mask = tgt.get("sky_mask")
                    if sky_mask is None and bool(validation_cfg.require_sky_mask):
                        raise ValueError(
                            "validation_v7.metrics.require_sky_mask=true but target missing sky_mask "
                            f"(scene={int(spec.scene_id)} segment={int(spec.segment_id)} idx={int(idx)})"
                        )
                    if sky_mask is not None:
                        sm = sky_mask.to(device).float()
                        if sm.dim() == 3:
                            sm = sm.squeeze(-1)
                        if sm.shape != gt.shape[:2]:
                            raise ValueError(
                                "validation sky_mask shape mismatch: "
                                f"sky_mask={tuple(sm.shape)} gt_hw={tuple(gt.shape[:2])}"
                            )
                        min_valid = int(validation_cfg.min_valid_pixels_per_region)
                        non_sky = (1.0 - sm).clamp(0.0, 1.0)
                        sky = sm.clamp(0.0, 1.0)
                        non_sky_count = int((non_sky > 0.5).sum().item())
                        sky_count = int((sky > 0.5).sum().item())
                        row["sky_mask_coverage"] = float(sm.mean().item())
                        row["non_sky_pixel_count"] = int(non_sky_count)
                        row["sky_pixel_count"] = int(sky_count)
                        sky_coverage_vals.append(float(row["sky_mask_coverage"]))

                        if non_sky_count >= min_valid:
                            psnr_non = _compute_masked_psnr(pred, gt, non_sky)
                            ssim_non = _compute_masked_ssim(pred, gt, non_sky)
                            lpips_non = _compute_masked_lpips(pred, gt, non_sky, lpips_metric)
                            if psnr_non is not None:
                                row["psnr_non_sky"] = float(psnr_non)
                                psnr_non_sky_vals.append(float(psnr_non))
                            if ssim_non is not None:
                                row["ssim_non_sky"] = float(ssim_non)
                                ssim_non_sky_vals.append(float(ssim_non))
                            if lpips_non is not None:
                                row["lpips_non_sky"] = float(lpips_non)
                                lpips_non_sky_vals.append(float(lpips_non))
                        if sky_count >= min_valid:
                            psnr_s = _compute_masked_psnr(pred, gt, sky)
                            ssim_s = _compute_masked_ssim(pred, gt, sky)
                            if psnr_s is not None:
                                row["psnr_sky"] = float(psnr_s)
                                psnr_sky_vals.append(float(psnr_s))
                            if ssim_s is not None:
                                row["ssim_sky"] = float(ssim_s)
                                ssim_sky_vals.append(float(ssim_s))

                per_view_rows.append(row)
                if validation_cfg.save_images:
                    _save_image_triplet(
                        int(trigger_step),
                        pred,
                        gt,
                        render_dir,
                        view_suffix=f"val_sc{int(spec.scene_id):03d}_seg{int(spec.segment_id):03d}_v{idx}",
                        save_error=False,
                    )

            episode_row = {
                "split": "validation_v7",
                "mode": validation_mode,
                "block_order": validation_block_order,
                "trigger_train_episode_counter": int(trigger_train_episode_counter),
                "trigger_step": int(trigger_step),
                "scene_id": int(spec.scene_id),
                "segment_id": int(spec.segment_id),
                "episode_start_keyframe_pos": int(spec.episode_start_keyframe_pos),
                "num_views": int(len(per_view_rows)),
                "psnr_full": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
                "ssim_full": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
                "lpips_full": float(np.mean(lpips_vals)) if lpips_vals else 0.0,
                "psnr_non_sky": _safe_mean(psnr_non_sky_vals),
                "ssim_non_sky": _safe_mean(ssim_non_sky_vals),
                "lpips_non_sky": _safe_mean(lpips_non_sky_vals),
                "psnr_sky": _safe_mean(psnr_sky_vals),
                "ssim_sky": _safe_mean(ssim_sky_vals),
                "sky_mask_coverage": _safe_mean(sky_coverage_vals),
                "num_views_non_sky_metric": int(len(psnr_non_sky_vals)),
                "num_views_sky_metric": int(len(psnr_sky_vals)),
                "views_formula": f"({len(spec.frame_chain)})x{int(spec.num_cams)}",
            }
            episode_row["psnr"] = float(episode_row["psnr_full"])
            episode_row["ssim"] = float(episode_row["ssim_full"])
            episode_row["lpips"] = float(episode_row["lpips_full"])
            episode_row["metric_scope"] = "full_image"
            if bool(validation_cfg.use_sky_mask_regions) and int(episode_row["num_views_non_sky_metric"]) > 0:
                episode_row["psnr"] = float(episode_row["psnr_non_sky"])
                episode_row["ssim"] = float(episode_row["ssim_non_sky"])
                episode_row["lpips"] = float(episode_row["lpips_non_sky"])
                episode_row["metric_scope"] = "non_sky"
            all_episode_rows.append(episode_row)

            with open(os.path.join(seg_dir, "per_view_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(per_view_rows, f, indent=2)
            with open(os.path.join(seg_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(episode_row, f, indent=2)
            if metrics_fh is not None:
                _write_metrics_history(metrics_fh, episode_row)
            if writer is not None:
                tb_step = max(int(trigger_step), 0)
                sid = int(spec.scene_id)
                seg = int(spec.segment_id)
                writer.add_scalar(
                    f"validation_v7/episode/psnr/scene_{sid:03d}_segment_{seg:03d}",
                    float(episode_row["psnr"]),
                    tb_step,
                )
                writer.add_scalar(
                    f"validation_v7/episode/ssim/scene_{sid:03d}_segment_{seg:03d}",
                    float(episode_row["ssim"]),
                    tb_step,
                )
                writer.add_scalar(
                    f"validation_v7/episode/lpips/scene_{sid:03d}_segment_{seg:03d}",
                    float(episode_row["lpips"]),
                    tb_step,
                )
                writer.add_scalar(
                    f"validation_v7/episode/num_views/scene_{sid:03d}_segment_{seg:03d}",
                    float(episode_row["num_views"]),
                    tb_step,
                )
                if bool(validation_cfg.use_sky_mask_regions):
                    writer.add_scalar(
                        f"validation_v7/episode/psnr_non_sky/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["psnr_non_sky"]),
                        tb_step,
                    )
                    writer.add_scalar(
                        f"validation_v7/episode/ssim_non_sky/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["ssim_non_sky"]),
                        tb_step,
                    )
                    writer.add_scalar(
                        f"validation_v7/episode/lpips_non_sky/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["lpips_non_sky"]),
                        tb_step,
                    )
                    writer.add_scalar(
                        f"validation_v7/episode/psnr_sky/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["psnr_sky"]),
                        tb_step,
                    )
                    writer.add_scalar(
                        f"validation_v7/episode/ssim_sky/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["ssim_sky"]),
                        tb_step,
                    )
                    writer.add_scalar(
                        f"validation_v7/episode/sky_mask_coverage/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["sky_mask_coverage"]),
                        tb_step,
                    )
                    writer.add_scalar(
                        f"validation_v7/episode/psnr_full/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["psnr_full"]),
                        tb_step,
                    )
                    writer.add_scalar(
                        f"validation_v7/episode/ssim_full/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["ssim_full"]),
                        tb_step,
                    )
                    writer.add_scalar(
                        f"validation_v7/episode/lpips_full/scene_{sid:03d}_segment_{seg:03d}",
                        float(episode_row["lpips_full"]),
                        tb_step,
                    )

            logger.info(
                "VALIDATION_V7_EPISODE_END mode=%s scene_id=%s segment_id=%s episode_start=%s num_views=%s "
                "metric_scope=%s psnr=%.4f ssim=%.4f lpips=%.4f",
                validation_mode,
                int(spec.scene_id),
                int(spec.segment_id),
                int(spec.episode_start_keyframe_pos),
                int(episode_row["num_views"]),
                str(episode_row["metric_scope"]),
                float(episode_row["psnr"]),
                float(episode_row["ssim"]),
                float(episode_row["lpips"]),
            )
    finally:
        if use_train_finetune and base_ckpt_bytes is not None:
            _restore_train_checkpoint_bytes(model, base_ckpt_bytes, device)
            model.train()

    if len(all_episode_rows) > 0:
        scene_to_rows: Dict[int, List[Dict[str, Any]]] = {}
        for row in all_episode_rows:
            sid = int(row["scene_id"])
            scene_to_rows.setdefault(sid, []).append(row)
        scene_agg = {
            str(int(sid)): {
                "num_episodes": int(len(rows)),
                "psnr": float(np.mean([float(r["psnr"]) for r in rows])),
                "ssim": float(np.mean([float(r["ssim"]) for r in rows])),
                "lpips": float(np.mean([float(r["lpips"]) for r in rows])),
                "psnr_full": float(np.mean([float(r["psnr_full"]) for r in rows])),
                "ssim_full": float(np.mean([float(r["ssim_full"]) for r in rows])),
                "lpips_full": float(np.mean([float(r["lpips_full"]) for r in rows])),
            }
            for sid, rows in scene_to_rows.items()
        }
        global_summary = {
            "split": "validation_v7_global",
            "mode": validation_mode,
            "block_order": validation_block_order,
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "trigger_step": int(trigger_step),
            "num_episodes": int(len(all_episode_rows)),
            "psnr": float(np.mean([float(r["psnr"]) for r in all_episode_rows])),
            "ssim": float(np.mean([float(r["ssim"]) for r in all_episode_rows])),
            "lpips": float(np.mean([float(r["lpips"]) for r in all_episode_rows])),
            "psnr_full": float(np.mean([float(r["psnr_full"]) for r in all_episode_rows])),
            "ssim_full": float(np.mean([float(r["ssim_full"]) for r in all_episode_rows])),
            "lpips_full": float(np.mean([float(r["lpips_full"]) for r in all_episode_rows])),
            "psnr_non_sky": _safe_mean([float(r["psnr_non_sky"]) for r in all_episode_rows]),
            "ssim_non_sky": _safe_mean([float(r["ssim_non_sky"]) for r in all_episode_rows]),
            "lpips_non_sky": _safe_mean([float(r["lpips_non_sky"]) for r in all_episode_rows]),
            "psnr_sky": _safe_mean([float(r["psnr_sky"]) for r in all_episode_rows]),
            "ssim_sky": _safe_mean([float(r["ssim_sky"]) for r in all_episode_rows]),
            "sky_mask_coverage": _safe_mean([float(r["sky_mask_coverage"]) for r in all_episode_rows]),
            "metric_scope": (
                "non_sky"
                if bool(validation_cfg.use_sky_mask_regions)
                and any(str(r.get("metric_scope", "")) == "non_sky" for r in all_episode_rows)
                else "full_image"
            ),
            "per_scene": scene_agg,
        }
        with open(
            os.path.join(val_root, f"summary_trigger_ep{int(trigger_train_episode_counter):06d}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(global_summary, f, indent=2)
        if metrics_fh is not None:
            _write_metrics_history(metrics_fh, global_summary)
        if writer is not None:
            tb_step = max(int(trigger_step), 0)
            ep_counter = int(trigger_train_episode_counter)
            writer.add_scalar("validation_v7/global/psnr", float(global_summary["psnr"]), tb_step)
            writer.add_scalar("validation_v7/global/ssim", float(global_summary["ssim"]), tb_step)
            writer.add_scalar("validation_v7/global/lpips", float(global_summary["lpips"]), tb_step)
            writer.add_scalar("validation_v7/global/num_episodes", float(global_summary["num_episodes"]), tb_step)
            if bool(validation_cfg.use_sky_mask_regions):
                writer.add_scalar("validation_v7/global/psnr_non_sky", float(global_summary["psnr_non_sky"]), tb_step)
                writer.add_scalar("validation_v7/global/ssim_non_sky", float(global_summary["ssim_non_sky"]), tb_step)
                writer.add_scalar("validation_v7/global/lpips_non_sky", float(global_summary["lpips_non_sky"]), tb_step)
                writer.add_scalar("validation_v7/global/psnr_sky", float(global_summary["psnr_sky"]), tb_step)
                writer.add_scalar("validation_v7/global/ssim_sky", float(global_summary["ssim_sky"]), tb_step)
                writer.add_scalar(
                    "validation_v7/global/sky_mask_coverage",
                    float(global_summary["sky_mask_coverage"]),
                    tb_step,
                )
                writer.add_scalar("validation_v7/global/psnr_full", float(global_summary["psnr_full"]), tb_step)
                writer.add_scalar("validation_v7/global/ssim_full", float(global_summary["ssim_full"]), tb_step)
                writer.add_scalar("validation_v7/global/lpips_full", float(global_summary["lpips_full"]), tb_step)
            writer.add_scalar(
                "validation_v7/global_by_train_episode/psnr",
                float(global_summary["psnr"]),
                ep_counter,
            )
            writer.add_scalar(
                "validation_v7/global_by_train_episode/ssim",
                float(global_summary["ssim"]),
                ep_counter,
            )
            writer.add_scalar(
                "validation_v7/global_by_train_episode/lpips",
                float(global_summary["lpips"]),
                ep_counter,
            )
        logger.info(
            "VALIDATION_V7_END trigger_episode_counter=%s num_episodes=%s metric_scope=%s psnr=%.4f ssim=%.4f lpips=%.4f",
            int(trigger_train_episode_counter),
            int(global_summary["num_episodes"]),
            str(global_summary["metric_scope"]),
            float(global_summary["psnr"]),
            float(global_summary["ssim"]),
            float(global_summary["lpips"]),
        )
    _reset_model_node_state_and_release_cuda(
        model,
        reason="validation_v7_end",
        step=int(trigger_step),
        scheduler_info=None,
        log_reset=bool((cfg.get("logging") or {}).get("log_node_state_reset", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="Path to config YAML.",
    )
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override. If unset, use training.seed from config (fallback: 42).",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Optional .pt from this script; loads model_state_dict (and optimizer unless --init_weights_only).",
    )
    parser.add_argument(
        "--init_weights_only",
        action="store_true",
        help="With --init_checkpoint, only restore model weights (fresh Adam state).",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Resume training from a checkpoint, restoring model and optimizer state.",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=None,
        help="Override training.start_step; interpreted as the first loop step to run.",
    )
    parser.add_argument("opts", nargs="*", help="Override config")
    args = parser.parse_args()

    cfg = setup(args)
    logging_cfg = cfg.get("logging") or {}
    if bool(logging_cfg.get("disable_stage6_mem_debug_env", False)):
        os.environ["STAGE6_MEM_DEBUG"] = "0"
    cfg.data.train_scene_ids = [int(x) for x in list(cfg.data.train_scene_ids)]
    cfg.data.eval_scene_ids = [int(x) for x in list(cfg.data.get("eval_scene_ids", []) or [])]
    if parse_view_selection(cfg.training.get("view_selection")) is not None:
        raise ValueError(
            "multi_scene training does not support training.view_selection.mode=explicit; "
            "remove view_selection from the config (dataset already samples keyframes per batch)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("RUN start time=%s device=%s", current_time, device)

    train_cfg_for_seed = cfg.get("training") or {}
    cfg_seed_raw = train_cfg_for_seed.get("seed")
    if args.seed is not None:
        resolved_seed = int(args.seed)
        if cfg_seed_raw is not None and int(cfg_seed_raw) != resolved_seed:
            logger.info(
                "Seed override: CLI --seed=%s overrides training.seed=%s",
                resolved_seed,
                int(cfg_seed_raw),
            )
    elif cfg_seed_raw is not None:
        resolved_seed = int(cfg_seed_raw)
    else:
        resolved_seed = 42
        logger.warning("No seed provided by CLI or config.training.seed; fallback to 42.")

    set_deterministic_seed(resolved_seed)
    logger.info("Seed: %s", resolved_seed)
    validation_v7_cfg = parse_validation_v7_config(cfg)
    losses_cfg = cfg.get("losses") or {}
    photometric_cfg = losses_cfg.get("photometric", {}) or {}
    mask_cfg = losses_cfg.get("mask", {}) or {}
    train_monitor_use_non_sky_region = bool(photometric_cfg.get("exclude_sky_region", False))
    train_monitor_require_sky_mask = bool(mask_cfg.get("require_sky_mask", train_monitor_use_non_sky_region))
    train_monitor_min_valid_pixels = int(validation_v7_cfg.min_valid_pixels_per_region)
    if train_monitor_min_valid_pixels < 1:
        raise ValueError("train monitor requires min_valid_pixels_per_region >= 1")
    if train_monitor_use_non_sky_region:
        logger.info(
            "train monitor metric scope=non_sky (losses.photometric.exclude_sky_region=true, "
            "require_sky_mask=%s, min_valid_pixels=%s)",
            bool(train_monitor_require_sky_mask),
            int(train_monitor_min_valid_pixels),
        )

    allow_one_segment = bool(globals().get("ALLOW_ONE_SEGMENT", False))
    allow_optional_one_segment = bool(globals().get("ALLOW_OPTIONAL_ONE_SEGMENT", False))
    allow_single_segment_mode = bool(allow_one_segment or allow_optional_one_segment)
    if cfg.get("one_segment") is not None and not allow_single_segment_mode:
        raise ValueError(
            "multi_scene training: remove `one_segment` from config; "
            f"use {DEFAULT_CONFIG_FILE}."
        )
    train_ids = list(cfg.data.train_scene_ids)
    if len(train_ids) < 2 and not allow_single_segment_mode:
        raise ValueError("multi_scene training requires len(data.train_scene_ids) >= 2")
    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment(cfg)
    if (fixed_scene_id is not None or fixed_segment_id is not None) and not allow_single_segment_mode:
        raise ValueError(
            "multi_scene training requires scheduler traversal fixed_scene_id and fixed_segment_id to be null "
            "(unset one_segment and traversal overrides)."
        )
    if allow_one_segment and not allow_optional_one_segment and (fixed_scene_id is None or fixed_segment_id is None):
        raise ValueError(
            "one-segment training requires one_segment.scene_id/segment_id or "
            "scheduler traversal fixed_scene_id/fixed_segment_id."
        )

    logger.info(
        "Building MultiSceneDatasetV3; multi_scene train_scene_ids=%s include_test=%s",
        train_ids,
        parse_include_test(cfg),
    )
    if cfg.get("test") is not None and bool(cfg.test.get("enable", False)):
        raise ValueError(
            "Formal testing is not supported in training script. "
            "Use the corresponding tools/test_minimal_streetforward_* entry instead."
        )
    dataset = build_multi_scene_dataset_v3(cfg, device)
    dataset.initialize()
    scheduler = build_train_scheduler_from_cfg(cfg, dataset)
    validation_specs: List[ValidationEpisodeSpecV7] = []
    train_episode_counter = 0
    if bool(validation_v7_cfg.eval_enable):
        if cfg.get("scheduler_v7") is None or not bool(cfg.scheduler_v7.get("enable", False)):
            raise ValueError("validation_v7 requires scheduler_v7.enable=true")
        sv7_block = cfg.scheduler_v7.get("block")
        if sv7_block is None:
            raise ValueError("validation_v7 requires scheduler_v7.block")
        sv7_ep = cfg.scheduler_v7.get("episode")
        if sv7_ep is None:
            raise ValueError("validation_v7 requires scheduler_v7.episode")
        sv7_execution = cfg.scheduler_v7.get("execution") or {}
        scheduler_steps_per_block = int(sv7_block["steps_per_block"])
        scheduler_blocks_per_episode = int(sv7_ep["blocks_per_episode"])
        scheduler_total_target_frames = int(sv7_ep["total_target_frames"])
        scheduler_block_order = str(sv7_execution.get("block_order", "block_major"))
        if scheduler_block_order not in ("block_major", "step_major"):
            raise ValueError(
                "scheduler_v7.execution.block_order must be one of ['block_major', 'step_major']"
            )
        scheduler_step_major_switch_interval_steps = int(sv7_execution.get("step_major_switch_interval_steps", 1))
        if scheduler_step_major_switch_interval_steps < 1:
            raise ValueError("scheduler_v7.execution.step_major_switch_interval_steps must be >= 1")

        validation_blocks_per_episode = (
            int(validation_v7_cfg.blocks_per_episode)
            if validation_v7_cfg.blocks_per_episode is not None
            else int(scheduler_blocks_per_episode)
        )
        validation_total_target_frames = (
            int(validation_v7_cfg.total_target_frames)
            if validation_v7_cfg.total_target_frames is not None
            else int(scheduler_total_target_frames)
        )

        if str(validation_v7_cfg.mode) == "segment_finetune_train":
            mismatches: List[str] = []
            if int(validation_v7_cfg.steps_per_block) != int(scheduler_steps_per_block):
                mismatches.append(
                    "block.steps_per_block "
                    f"({int(validation_v7_cfg.steps_per_block)} != {int(scheduler_steps_per_block)})"
                )
            if int(validation_blocks_per_episode) != int(scheduler_blocks_per_episode):
                mismatches.append(
                    "episode.blocks_per_episode "
                    f"({int(validation_blocks_per_episode)} != {int(scheduler_blocks_per_episode)})"
                )
            if int(validation_total_target_frames) != int(scheduler_total_target_frames):
                mismatches.append(
                    "episode.total_target_frames "
                    f"({int(validation_total_target_frames)} != {int(scheduler_total_target_frames)})"
                )
            if str(validation_v7_cfg.block_order) != str(scheduler_block_order):
                mismatches.append(
                    "execution.block_order "
                    f"({str(validation_v7_cfg.block_order)!r} != {str(scheduler_block_order)!r})"
                )
            if int(validation_v7_cfg.step_major_switch_interval_steps) != int(
                scheduler_step_major_switch_interval_steps
            ):
                mismatches.append(
                    "execution.step_major_switch_interval_steps "
                    f"({int(validation_v7_cfg.step_major_switch_interval_steps)} != "
                    f"{int(scheduler_step_major_switch_interval_steps)})"
                )
            if len(mismatches) > 0:
                raise ValueError(
                    "validation_v7.mode=segment_finetune_train requires validation_v7 to match scheduler_v7. "
                    f"Mismatches: {', '.join(mismatches)}"
                )

        validation_specs = build_validation_episode_specs_v7(
            dataset=dataset,
            eval_scene_ids=[int(x) for x in validation_v7_cfg.eval_scene_ids],
            blocks_per_episode=int(validation_blocks_per_episode),
            total_target_frames=int(validation_total_target_frames),
        )
        logger.info(
            "validation_v7 enabled: eval_scenes=%s specs=%s mode=%s block_order=%s reset_policy=%s "
            "step_major_switch_interval_steps=%s steps_per_block=%s blocks_per_episode=%s total_target_frames=%s "
            "validate_every_n_episodes=%s run_at_train_start=%s",
            [int(x) for x in validation_v7_cfg.eval_scene_ids],
            int(len(validation_specs)),
            str(validation_v7_cfg.mode),
            str(validation_v7_cfg.block_order),
            str(validation_v7_cfg.reset_policy),
            int(validation_v7_cfg.step_major_switch_interval_steps),
            int(validation_v7_cfg.steps_per_block),
            int(validation_blocks_per_episode),
            int(validation_total_target_frames),
            int(validation_v7_cfg.validate_every_n_episodes),
            bool(validation_v7_cfg.run_at_train_start),
        )
        if len(validation_specs) == 0:
            raise ValueError("validation_v7 enabled but no valid validation specs can be built")
        if bool(validation_v7_cfg.persist_across_training):
            for spec in validation_specs:
                if hasattr(dataset, "build_preload_hint") and hasattr(dataset, "submit_preload_hint"):
                    hint = dataset.build_preload_hint(
                        scene_id=int(spec.scene_id),
                        segment_id=int(spec.segment_id),
                        future_image_refs=[tuple(x) for x in spec.eval_image_refs],
                        scope="episode_chain_exact",
                    )
                    dataset.submit_preload_hint(
                        hint=hint,
                        hint_scope="episode_chain_exact",
                        epoch_idx=0,
                        global_step=0,
                        block_idx_global=0,
                        include_test=False,
                    )

    sv5 = cfg.get("scheduler_v5")
    if sv5 is not None and bool(sv5.get("enable", False)):
        ts5 = sv5.get("target_sampling") or {}
        logger.info(
            "TrainSchedulerV5 target sampling: total_target_frames=%s include_source_frame=%s policy=%s neighbor_ring=%s",
            ts5.get("total_target_frames"),
            ts5.get("include_source_frame"),
            ts5.get("policy"),
            ts5.get("neighbor_ring"),
        )
    else:
        ov = cfg.get("scheduler_v4", {}).get("overlap") or {}
        logger.info(
            "TrainScheduler overlap: mode=%s point_sample_size=%s candidate_frame_policy=%s score_type=%s topk=%s",
            ov.get("mode"),
            ov.get("point_sample_size"),
            ov.get("candidate_frame_policy"),
            ov.get("score_type"),
            ov.get("topk"),
        )
    pd = cfg.data.get("preload") if cfg.data is not None else None
    logger.info(
        "Dataset preload overlap: episode_superset=%s next_block_exact=%s",
        (pd or {}).get("warm_overlap_pairs_episode_superset"),
        (pd or {}).get("warm_overlap_pairs_next_block_exact"),
    )

    sv3_mns = cfg.get("scheduler_v3", {}).get("model_node_state") if cfg.get("scheduler_v3") else None
    if sv3_mns and bool(sv3_mns.get("sync_with_scheduler")):
        rp = sv3_mns.get("reset_policy", "auto(block_major->block_end, step_major->episode_end)")
        logger.info(
            "scheduler_v3.model_node_state.sync_with_scheduler=true: NodeState write-back when "
            "segment_local_step %% U == 0; reset_node_state() controlled by reset_policy=%s. "
            "model.update_node_state_interval / reset_node_state_interval are ignored.",
            rp,
        )
    sv9 = cfg.get("scheduler_v9")
    if sv9 is not None and bool(sv9.get("enable", False)):
        sv9_execution = sv9.get("execution") or {}
        logger.info(
            "scheduler_v9 node_state sync active: reset_node_state() controlled by execution.reset_policy=%s.",
            str(sv9_execution.get("reset_policy", "episode_end")),
        )

    max_iterations = int(args.max_steps or cfg.training.get("max_iterations", 1000))
    resume_checkpoint = _resolve_resume_checkpoint_cfg(cfg, args)
    log_interval = cfg.training.get("log_interval", 50)
    save_every = cfg.training.get("save_checkpoint_freq", 500)
    enable_psnr = bool(cfg.eval.get("enable_psnr", True))
    train_step_metrics_interval = int(cfg.logging.get("train_step_metrics_interval", 0))
    if train_step_metrics_interval < 0:
        raise ValueError("logging.train_step_metrics_interval must be >= 0")
    random_window_diagnostics_interval = int(
        cfg.logging.get("random_window_diagnostics_interval", max(int(train_step_metrics_interval), 100))
    )
    if random_window_diagnostics_interval < 0:
        raise ValueError("logging.random_window_diagnostics_interval must be >= 0")
    log_node_state_reset = bool(cfg.logging.get("log_node_state_reset", True))
    write_node_state_reset_metrics = bool(cfg.logging.get("write_node_state_reset_metrics", True))
    node_state_reset_cuda_memory = bool(cfg.logging.get("node_state_reset_cuda_memory", False))
    train_monitor_cfg = cfg.logging.get("train_monitor") or {}
    train_monitor_enable_heavy_metrics = bool(train_monitor_cfg.get("enable_heavy_metrics", True))
    train_monitor_include_per_view_metrics = bool(train_monitor_cfg.get("include_per_view_metrics", True))
    train_monitor_include_extra_result_metrics = bool(train_monitor_cfg.get("include_extra_result_metrics", True))
    train_monitor_enable_low_psnr_image_dump = bool(train_monitor_cfg.get("enable_low_psnr_image_dump", True))
    save_train_views_psnr_below: Optional[float]
    _raw_psnr_below = cfg.eval.get("save_train_views_psnr_below", None)
    if _raw_psnr_below is None:
        save_train_views_psnr_below = None
    else:
        save_train_views_psnr_below = float(_raw_psnr_below)
        if save_train_views_psnr_below <= 0:
            raise ValueError("eval.save_train_views_psnr_below must be > 0 when set")
        if not enable_psnr:
            raise ValueError("eval.save_train_views_psnr_below requires eval.enable_psnr=true")
    if not train_monitor_enable_low_psnr_image_dump and save_train_views_psnr_below is not None:
        logger.info(
            "train_monitor.enable_low_psnr_image_dump=false: ignore eval.save_train_views_psnr_below=%.4f",
            float(save_train_views_psnr_below),
        )
        save_train_views_psnr_below = None
    if not train_monitor_enable_heavy_metrics and save_train_views_psnr_below is not None:
        logger.info(
            "train_monitor.enable_heavy_metrics=false: disable low-PSNR image dump "
            "(eval.save_train_views_psnr_below=%.4f ignored).",
            float(save_train_views_psnr_below),
        )
        save_train_views_psnr_below = None
    logger.info(
        "Train monitor switches: heavy_metrics=%s include_per_view_metrics=%s "
        "include_extra_result_metrics=%s low_psnr_image_dump=%s train_step_metrics_interval=%s",
        bool(train_monitor_enable_heavy_metrics),
        bool(train_monitor_include_per_view_metrics),
        bool(train_monitor_include_extra_result_metrics),
        bool(train_monitor_enable_low_psnr_image_dump),
        int(train_step_metrics_interval),
    )
    logger.info(
        "Random-window metrics: diagnostics_interval=%s",
        int(random_window_diagnostics_interval),
    )
    enable_jsonl_metrics = bool(cfg.logging.get("enable_jsonl_metrics", True))
    metrics_history_append = bool(cfg.logging.get("metrics_history_append", True))
    if resume_checkpoint and not metrics_history_append:
        logger.warning(
            "Resume is active but logging.metrics_history_append=false; using the same log_dir would overwrite metrics_history.jsonl."
        )
    image_trigger_cfg = cfg.logging.get("image_trigger") or {}
    if image_trigger_cfg:
        image_trigger_mode = str(image_trigger_cfg.get("mode", "raw_step_interval")).strip()
        image_interval_blocks_equiv = int(
            image_trigger_cfg.get(
                "interval_blocks_equiv",
                cfg.logging.get("image_interval_blocks", 1),
            )
        )
    else:
        if "image_interval_blocks" not in cfg.logging:
            raise ValueError(
                "logging.image_interval_blocks is required when logging.image_trigger is unset."
            )
        image_trigger_mode = "block_end"
        image_interval_blocks_equiv = int(cfg.logging["image_interval_blocks"])
    if image_trigger_mode not in ("block_end", "raw_step_interval", "episode_end"):
        raise ValueError(
            "logging.image_trigger.mode must be one of ['block_end', 'raw_step_interval', 'episode_end']"
        )
    if image_interval_blocks_equiv < 1:
        raise ValueError(f"logging image trigger interval must be >= 1, got {image_interval_blocks_equiv}")
    scheduler_steps_per_block = int(getattr(scheduler, "steps_per_block", 1))
    if scheduler_steps_per_block < 1:
        scheduler_steps_per_block = 1
    image_trigger_interval_steps = int(image_interval_blocks_equiv * scheduler_steps_per_block)
    logger.info(
        "Train image trigger: mode=%s interval_blocks_equiv=%s steps_per_block=%s interval_steps=%s",
        image_trigger_mode,
        int(image_interval_blocks_equiv),
        int(scheduler_steps_per_block),
        int(image_trigger_interval_steps),
    )
    if image_trigger_mode == "episode_end":
        logger.info(
            "Train image trigger episode_end gate: save when completed_blocks %% interval_blocks_equiv == 0"
        )
    low_psnr_train_images_subdir: Optional[str] = None
    if save_train_views_psnr_below is not None:
        if "low_psnr_train_images_subdir" not in cfg.logging:
            raise ValueError(
                "logging.low_psnr_train_images_subdir is required when eval.save_train_views_psnr_below is set"
            )
        low_psnr_train_images_subdir = str(cfg.logging["low_psnr_train_images_subdir"]).strip()
        if not low_psnr_train_images_subdir:
            raise ValueError("logging.low_psnr_train_images_subdir must be non-empty")
        logger.info(
            "eval.save_train_views_psnr_below=%.4f (if any view PSNR < threshold, save all views' pred/gt to log_dir/images/%s/)",
            float(save_train_views_psnr_below),
            low_psnr_train_images_subdir,
        )
    use_tensorboard = bool(cfg.logging.get("use_tensorboard", False))
    diag_cfg = _parse_diagnostics_cfg(cfg)
    perf_cfg = _parse_perf_cfg(cfg)
    perf_raw_cfg = cfg.logging.get("performance") or {}
    perf_empty_cache_interval_steps = int(perf_raw_cfg.get("empty_cache_interval_steps", 0))
    if perf_empty_cache_interval_steps < 0:
        raise ValueError("logging.performance.empty_cache_interval_steps must be >= 0")
    perf_cleanup_metrics_interval_steps = int(
        perf_raw_cfg.get("cleanup_metrics_interval_steps", train_step_metrics_interval)
    )
    if perf_cleanup_metrics_interval_steps < 0:
        raise ValueError("logging.performance.cleanup_metrics_interval_steps must be >= 0")
    perf_live_tensor_summary_interval_steps = int(perf_raw_cfg.get("live_tensor_summary_interval_steps", 0))
    if perf_live_tensor_summary_interval_steps < 0:
        raise ValueError("logging.performance.live_tensor_summary_interval_steps must be >= 0")
    perf_live_tensor_summary_topk = int(perf_raw_cfg.get("live_tensor_summary_topk", 12))
    if perf_live_tensor_summary_topk < 0:
        raise ValueError("logging.performance.live_tensor_summary_topk must be >= 0")

    pixel_camera_ids: List[int] = []
    if cfg.data is not None and cfg.data.get("pixel_source") is not None:
        pcams = cfg.data.pixel_source.get("cameras")
        if pcams is not None:
            pixel_camera_ids = [int(x) for x in list(pcams)]

    trainer_cls = TRAINER_CLASS
    model = trainer_cls(config=cfg, device=device)
    required_validation_methods = [
        "inference_step_from_train_batch",
        "export_3dgs_state",
        "render_views_from_scene_state",
        "reset_node_state",
    ]
    if bool(validation_v7_cfg.eval_enable):
        if str(validation_v7_cfg.mode) == "segment_finetune_train":
            required_validation_methods.append("train_step")
        missing = [m for m in required_validation_methods if not hasattr(model, m)]
        if missing:
            raise ValueError(
                f"validation_v7 requires trainer methods {required_validation_methods}, "
                f"but {trainer_cls.__name__} is missing: {missing}. "
                "Disable validation_v7.eval_enable or implement the missing APIs."
            )
        if str(validation_v7_cfg.mode) == "segment_finetune_train":
            if not hasattr(model, "optimizer") or model.optimizer is None:
                raise ValueError(
                    "validation_v7.mode=segment_finetune_train requires trainer.optimizer to exist."
                )
    model.train()
    init_checkpoint, init_weights_only, require_export_type = _resolve_init_checkpoint_cfg(cfg, args)
    if resume_checkpoint and init_checkpoint:
        raise ValueError(
            "--resume_checkpoint / training.resume_checkpoint cannot be combined with init_checkpoint. "
            "Use resume for continuation or init_checkpoint for warm-start initialization."
        )
    resume_payload: Optional[Dict[str, Any]] = None
    if resume_checkpoint:
        resume_payload = _load_resume_checkpoint(resume_checkpoint, model)
    else:
        _load_init_checkpoint(
            init_checkpoint,
            model,
            device,
            weights_only=init_weights_only,
            require_export_type=require_export_type,
        )
    start_step = _resolve_start_step(cfg, args, resume_payload)
    if start_step < 0:
        raise ValueError(f"training.start_step must be >= 0, got {int(start_step)}")
    if resume_payload is not None:
        expected_resume_step = int(_checkpoint_step(resume_payload)) + 1
        if int(start_step) != int(expected_resume_step):
            logger.warning(
                "Resume start_step=%s differs from checkpoint step + 1 (%s); "
                "this is a manual override and may not be a strict continuation.",
                int(start_step),
                int(expected_resume_step),
            )
        restored_scheduler = _restore_scheduler_state_from_checkpoint(scheduler, resume_payload)
        if not restored_scheduler:
            _apply_scheduler_start_step(scheduler, start_step)
        _restore_rng_state(resume_payload.get("rng_state"))
        train_loop_state = resume_payload.get("train_loop")
        if isinstance(train_loop_state, dict) and train_loop_state.get("train_episode_counter") is not None:
            train_episode_counter = int(train_loop_state["train_episode_counter"])
        else:
            logger.warning(
                "Resume checkpoint has no train_loop.train_episode_counter; validation episode counters restart at 0."
            )
    else:
        _apply_scheduler_start_step(scheduler, start_step)
    if max_iterations > 0 and int(start_step) >= int(max_iterations):
        raise ValueError(
            f"start_step={int(start_step)} is >= training.max_iterations/--max_steps={int(max_iterations)}. "
            "max_iterations is interpreted as an absolute exclusive end step."
        )
    logger.info(
        "Training loop step range: start_step=%s max_iterations=%s resume_checkpoint=%s",
        int(start_step),
        int(max_iterations),
        bool(resume_checkpoint),
    )

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    metrics_fh: Optional[TextIO] = None
    writer: Optional[Any] = None
    result: Optional[Dict[str, Any]] = None
    last_train_loss = float("nan")
    total_steps = 0
    sum_num_gaussians_bg = 0.0
    sum_num_gaussians_distant = 0.0
    sum_num_gaussians_rigid = 0.0
    sum_num_gaussians_sky = 0.0
    sum_step_time_ms = 0.0
    step_time_ms_hist: List[float] = []
    peak_mem_bytes = 0
    peak_mem_reserved_bytes = 0
    diag_window: deque = deque(maxlen=max(diag_cfg.get("window_size", 0), 1))
    minimal_batch: Dict[str, Any] = {}
    block_loss_accum: Dict[int, Dict[str, Any]] = {}
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    history_cfg = model_cfg.get("history_memory", {}) if hasattr(model_cfg, "get") else {}
    model_stage = str(model_cfg.get("stage", ""))
    enable_block_exit_record = bool(model_stage in {"5_2", "5_3"}) and bool(
        str(history_cfg.get("record_on", "")) == "block_exit"
    )
    episode_end_hook = globals().get("EPISODE_END_HOOK")
    train_start_hook = globals().get("TRAIN_START_HOOK")
    step_end_hook = globals().get("STEP_END_HOOK")

    try:
        metrics_fh = _open_metrics_history(
            cfg.log_dir,
            enable_jsonl_metrics,
            append=metrics_history_append,
        )
        if use_tensorboard and SummaryWriter is not None:
            tb_dir = os.path.join(cfg.log_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
        elif use_tensorboard and SummaryWriter is None:
            logger.warning("logging.use_tensorboard=true but torch.utils.tensorboard is unavailable; TensorBoard disabled.")
        if bool(validation_v7_cfg.eval_enable and validation_v7_cfg.run_at_train_start):
            _run_validation_v7_round(
                cfg=cfg,
                dataset=dataset,
                model=model,
                specs=validation_specs,
                validation_cfg=validation_v7_cfg,
                device=device,
                trigger_train_episode_counter=0,
                trigger_step=int(start_step) - 1,
                psnr_metric=psnr_metric,
                ssim_metric=ssim_metric,
                lpips_metric=lpips_metric,
                metrics_fh=metrics_fh,
                writer=writer,
            )
        if callable(train_start_hook):
            train_start_hook(
                cfg=cfg,
                dataset=dataset,
                model=model,
                device=device,
                trigger_train_episode_counter=0,
                trigger_step=int(start_step) - 1,
                psnr_metric=psnr_metric,
                ssim_metric=ssim_metric,
                lpips_metric=lpips_metric,
                metrics_fh=metrics_fh,
                writer=writer,
            )

        for step in range(int(start_step), int(max_iterations)):
            iter_t0 = time.perf_counter()
            fetch_t0 = time.perf_counter()
            raw_batch = scheduler.next_batch()
            fetch_t1 = time.perf_counter()
            batch_fetch_ms = float((fetch_t1 - fetch_t0) * 1000.0)
            scheduler_info = raw_batch.get("_scheduler_v4_aligned_info")
            if scheduler_info is None:
                scheduler_info = scheduler.get_current_info()

            step_events = scheduler.pop_events()
            validation_due_episode_counters: List[int] = []
            hook_due_episode_counters: List[int] = []
            for ev in step_events:
                if ev.get("type") == "episode_end":
                    train_episode_counter += 1
                    if callable(episode_end_hook):
                        hook_due_episode_counters.append(int(train_episode_counter))
                    if bool(validation_v7_cfg.eval_enable):
                        if train_episode_counter % int(validation_v7_cfg.validate_every_n_episodes) == 0:
                            validation_due_episode_counters.append(int(train_episode_counter))
            scheduler_node_sync = _build_scheduler_node_sync(cfg, scheduler_info, step_events)
            if scheduler_node_sync is None:
                scheduler_node_sync = _build_scheduler_node_sync_long_phase_b_fallback(cfg, scheduler_info, step_events)
            if scheduler_node_sync is None:
                scheduler_node_sync = _build_scheduler_node_sync_v9_fallback(cfg, scheduler_info, step_events)
            if scheduler_node_sync is None:
                scheduler_node_sync = _build_scheduler_node_sync_v8_fallback(cfg, scheduler_info, step_events)
            defer_node_state_reset_for_block_exit_record = False
            defer_node_state_reset_for_episode_hook = False
            if (
                enable_block_exit_record
                and scheduler_node_sync is not None
                and bool(scheduler_node_sync.get("reset_after_block", False))
            ):
                # Stage5_2/Stage5_3 block-exit record pass reads current runtime node states/histories.
                # Defer scheduler-triggered reset to after record_block_history in this step.
                scheduler_node_sync = dict(scheduler_node_sync)
                scheduler_node_sync["reset_after_block"] = False
                defer_node_state_reset_for_block_exit_record = True
            if (
                callable(episode_end_hook)
                and scheduler_node_sync is not None
                and bool(scheduler_node_sync.get("reset_after_block", False))
                and any(ev.get("type") == "episode_end" for ev in step_events)
            ):
                scheduler_node_sync = dict(scheduler_node_sync)
                scheduler_node_sync["reset_after_block"] = False
                defer_node_state_reset_for_episode_hook = True
            for ev in step_events:
                if ev.get("type") == "segment_begin":
                    logger.info(
                        "SEGMENT_BEGIN epoch=%s global_step=%s scene_id=%s scene_dir=%s segment=%s U=%s segment_budget_u=%s segment_step_budget=%s updates_per_block=%s",
                        ev.get("epoch_idx"),
                        ev.get("global_step"),
                        ev.get("scene_id"),
                        _scene_dir_str(ev.get("scene_id", -1)),
                        ev.get("segment_id"),
                        ev.get("U"),
                        ev.get("segment_budget_u"),
                        ev.get("segment_step_budget"),
                        ev.get("updates_per_block"),
                    )
                elif ev.get("type") == "reset_event":
                    logger.info(
                        "RESET global_step=%s scene_id=%s scene_dir=%s segment=%s reset_episode_idx=%s reason=%s window=%s num_pairs=%s",
                        ev.get("global_step"),
                        ev.get("scene_id"),
                        _scene_dir_str(ev.get("scene_id", -1)),
                        ev.get("segment_id"),
                        ev.get("reset_episode_idx"),
                        ev.get("reason"),
                        ev.get("window_keyframes"),
                        ev.get("num_pairs"),
                    )
                elif ev.get("type") == "overlap_select":
                    logger.info(
                        "OVERLAP_SELECT global_step=%s scene_id=%s scene_dir=%s segment=%s overlap_mode=%s hits=%s misses=%s "
                        "pair_compute_miss_ms_total=%.2f pair_eval_wall_ms_total=%.2f",
                        ev.get("global_step"),
                        ev.get("scene_id"),
                        _scene_dir_str(ev.get("scene_id", -1)),
                        ev.get("segment_id"),
                        ev.get("overlap_mode"),
                        ev.get("cache_hits"),
                        ev.get("cache_misses"),
                        float(ev.get("pair_compute_miss_time_ms_total") or 0.0),
                        float(ev.get("pair_eval_wall_time_ms_total") or 0.0),
                    )
                elif ev.get("type") == "block_begin":
                    extra_bb = ""
                    if ev.get("overlap_mode") == "pointcloud_topk" and ev.get("selected_target_scores") is not None:
                        extra_bb = " overlap_scores=%s" % (ev.get("selected_target_scores"),)
                    logger.info(
                        "BLOCK_BEGIN global_step=%s scene_id=%s scene_dir=%s segment=%s block_seg=%s block_global=%s U=%s K_u_nom=%s K_u_eff=%s K_steps_eff=%s source_kf=%s source_frame=%s source_image_ref=%s overlap_mode=%s%s",
                        ev.get("global_step"),
                        ev.get("scene_id"),
                        _scene_dir_str(ev.get("scene_id", -1)),
                        ev.get("segment_id"),
                        ev.get("block_idx_in_segment"),
                        ev.get("block_idx_global"),
                        ev.get("U"),
                        ev.get("K_u_nominal"),
                        ev.get("K_u_effective"),
                        ev.get("K_steps_effective"),
                        ev.get("source_keyframe_idx"),
                        ev.get("source_frame_idx"),
                        ev.get("source_image_ref"),
                        ev.get("overlap_mode"),
                        extra_bb,
                    )
                elif ev.get("type") == "block_exit":
                    logger.info(
                        "BLOCK_EXIT global_step=%s scene_id=%s scene_dir=%s segment=%s block_seg=%s block_global=%s "
                        "source_frame=%s source_image_ref=%s target_refs=%s num_updates_in_block=%s",
                        ev.get("global_step"),
                        ev.get("scene_id"),
                        _scene_dir_str(ev.get("scene_id", -1)),
                        ev.get("segment_id"),
                        ev.get("block_idx_in_segment"),
                        ev.get("block_idx_global"),
                        ev.get("source_frame_idx"),
                        ev.get("source_image_ref"),
                        len(ev.get("target_image_refs") or []),
                        ev.get("num_updates_in_block"),
                    )

            tgt = raw_batch.get("target")
            if not isinstance(tgt, dict) or tgt.get("image") is None:
                raise ValueError("dataset batch must contain target.image")
            num_target_views = int(tgt["image"].shape[0])
            convert_t0 = time.perf_counter()
            minimal_batch = convert_batch_to_minimal_format(
                raw_batch,
                device,
                num_targets=num_target_views,
                include_source_for_2d=True,
                view_selection=None,
            )
            convert_t1 = time.perf_counter()
            batch_convert_ms = float((convert_t1 - convert_t0) * 1000.0)
            minimal_batch["_stage5_6_collect_debug_images"] = _stage5_6_debug_images_due(
                image_trigger_mode=image_trigger_mode,
                step=int(step),
                scheduler_info=scheduler_info,
                step_events=step_events,
                image_trigger_interval_steps=int(image_trigger_interval_steps),
                image_interval_blocks_equiv=int(image_interval_blocks_equiv),
            )

            step_t0 = time.perf_counter()
            if perf_cfg["enable"] and torch.cuda.is_available():
                torch.cuda.synchronize()
                step_t0 = time.perf_counter()
            if perf_cfg["enable"] and perf_cfg["cuda_memory"] and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            result = model.train_step(
                minimal_batch,
                step=step,
                profile_phase_timing=bool(perf_cfg["enable"] and perf_cfg["phase_timing"]),
                sync_cuda_timing=bool(perf_cfg["enable"] and perf_cfg["phase_timing"]),
                scheduler_node_sync=scheduler_node_sync,
            )
            if perf_cfg["enable"] and torch.cuda.is_available():
                torch.cuda.synchronize()
            step_t1 = time.perf_counter()
            step_time_ms = float((step_t1 - step_t0) * 1000.0)
            sum_step_time_ms += step_time_ms
            step_time_ms_hist.append(step_time_ms)
            if perf_cfg["enable"] and perf_cfg["cuda_memory"] and torch.cuda.is_available():
                peak_mem_bytes = int(max(peak_mem_bytes, int(torch.cuda.max_memory_allocated())))
                peak_mem_reserved_bytes = int(max(peak_mem_reserved_bytes, int(torch.cuda.max_memory_reserved())))

            if result is None:
                raise ValueError("train_step returned None")
            if defer_node_state_reset_for_block_exit_record or defer_node_state_reset_for_episode_hook:
                result = dict(result)
                result["node_state_sync_reset"] = True
                result["node_state_sync_reset_deferred"] = True
            if enable_block_exit_record:
                if not hasattr(model, "record_block_history"):
                    raise ValueError(
                        "Stage5_2/Stage5_3 record_on=block_exit requires model.record_block_history."
                    )
                block_exit_events = [ev for ev in step_events if str(ev.get("type", "")) == "block_exit"]
                for ev in block_exit_events:
                    rec_metrics = model.record_block_history(minimal_batch, ev)
                    logger.info(
                        "HISTORY_RECORD global_step=%s scene_id=%s scene_dir=%s segment=%s block_global=%s "
                        "record_views=%s num_views=%s source_refs=%s target_refs=%s "
                        "bg_s=%.6f bg_e=%.6f distant_s=%.6f distant_e=%.6f rigid_s=%.6f rigid_e=%.6f",
                        ev.get("global_step"),
                        ev.get("scene_id"),
                        _scene_dir_str(ev.get("scene_id", -1)),
                        ev.get("segment_id"),
                        ev.get("block_idx_global"),
                        "source_image_refs"
                        if float(rec_metrics.get("stage5_2_record_use_source_views", 0.0)) > 0.5
                        else "target_image_refs",
                        int(rec_metrics.get("stage5_2_record_num_views", 0.0)),
                        int(rec_metrics.get("stage5_2_record_num_source_refs", 0.0)),
                        int(rec_metrics.get("stage5_2_record_num_target_refs", 0.0)),
                        float(rec_metrics.get("stage5_2_history_bg_support_mean", 0.0)),
                        float(rec_metrics.get("stage5_2_history_bg_error_mean", 0.0)),
                        float(rec_metrics.get("stage5_2_history_distant_support_mean", 0.0)),
                        float(rec_metrics.get("stage5_2_history_distant_error_mean", 0.0)),
                        float(rec_metrics.get("stage5_2_history_rigid_support_mean", 0.0)),
                        float(rec_metrics.get("stage5_2_history_rigid_error_mean", 0.0)),
                    )
                    if writer is not None:
                        tb_step = int(ev.get("global_step", step))
                        writer.add_scalar(
                            "train/history/bg_support_mean",
                            float(rec_metrics.get("stage5_2_history_bg_support_mean", 0.0)),
                            tb_step,
                        )
                        writer.add_scalar(
                            "train/history/bg_error_mean",
                            float(rec_metrics.get("stage5_2_history_bg_error_mean", 0.0)),
                            tb_step,
                        )
                        writer.add_scalar(
                            "train/history/distant_support_mean",
                            float(rec_metrics.get("stage5_2_history_distant_support_mean", 0.0)),
                            tb_step,
                        )
                        writer.add_scalar(
                            "train/history/distant_error_mean",
                            float(rec_metrics.get("stage5_2_history_distant_error_mean", 0.0)),
                            tb_step,
                        )
                        writer.add_scalar(
                            "train/history/rigid_support_mean",
                            float(rec_metrics.get("stage5_2_history_rigid_support_mean", 0.0)),
                            tb_step,
                        )
                        writer.add_scalar(
                            "train/history/rigid_error_mean",
                            float(rec_metrics.get("stage5_2_history_rigid_error_mean", 0.0)),
                            tb_step,
                        )
                if defer_node_state_reset_for_block_exit_record and not defer_node_state_reset_for_episode_hook:
                    _reset_model_node_state_and_release_cuda(
                        model,
                        reason="deferred_block_exit_record",
                        step=int(step),
                        scheduler_info=scheduler_info,
                        log_reset=bool(log_node_state_reset),
                    )
            loss_val = float(result["loss"])
            last_train_loss = float(loss_val)
            pred_rgbs = result["pred_rgbs"]
            gt_images = result["gt_images"]
            num_views = len(pred_rgbs)
            total_steps += 1
            sum_num_gaussians_bg += int(result.get("num_gaussians_bg", 0))
            sum_num_gaussians_distant += int(result.get("num_gaussians_distant", 0))
            sum_num_gaussians_rigid += int(result.get("num_gaussians_rigid", 0))
            sum_num_gaussians_sky += int(result.get("num_gaussians_sky", 0))

            current_block_idx_global = int(scheduler_info.get("block_idx_global", -1))
            if current_block_idx_global >= 0:
                block_acc = block_loss_accum.setdefault(
                    int(current_block_idx_global),
                    {
                        "loss_sum": 0.0,
                        "loss_count": 0,
                        "scene_id": int(scheduler_info.get("scene_id", -1)),
                        "segment_id": int(scheduler_info.get("segment_id", -1)),
                        "episode_idx_global": int(scheduler_info.get("episode_idx_global", -1)),
                        "block_idx_in_episode": int(scheduler_info.get("block_idx_in_episode", -1)),
                    },
                )
                block_acc["loss_sum"] = float(block_acc.get("loss_sum", 0.0)) + float(loss_val)
                block_acc["loss_count"] = int(block_acc.get("loss_count", 0)) + 1

            if step % log_interval == 0:
                logger.info(
                    "Step %s: loss=%.6f views=%d rigid_update=%d bg_update=%d distant_update=%d sky_update=%d onepass=%d",
                    step,
                    loss_val,
                    num_views,
                    int(result.get("num_rigid_update", 0)),
                    int(result.get("num_bg_update", 0)),
                    int(result.get("num_distant_update", 0)),
                    int(result.get("num_sky_update", 0)),
                    int(result.get("src_backproject_pass_count", 0)),
                )
                if perf_cfg["enable"]:
                    logger.info(
                        "Perf step=%s step_time_ms=%.2f forward_ms=%.2f backward_ms=%.2f optimizer_ms=%.2f",
                        step,
                        step_time_ms,
                        float(result.get("forward_ms", 0.0)),
                        float(result.get("backward_ms", 0.0)),
                        float(result.get("optimizer_ms", 0.0)),
                    )

            diag_row: Dict[str, Any] = {}
            if diag_cfg["enable"]:
                diag_window.append({"loss": loss_val, "step_time_ms": step_time_ms})
                if step % max(diag_cfg["interval"], 1) == 0:
                    diag_row = _diagnose_step(list(diag_window))

            if train_step_metrics_interval > 0 and step % train_step_metrics_interval == 0:
                if _is_iforward_random_window_result(result):
                    train_step_row = _build_iforward_random_window_train_step_row(
                        step=int(step),
                        minimal_batch=minimal_batch,
                        scheduler_info=scheduler_info,
                        step_events=step_events,
                        result=result,
                        loss_val=float(loss_val),
                        num_views=int(num_views),
                        step_time_ms=float(step_time_ms),
                        batch_fetch_ms=float(batch_fetch_ms),
                        batch_convert_ms=float(batch_convert_ms),
                    )
                    if perf_cfg["enable"] and perf_cfg["cuda_memory"] and torch.cuda.is_available():
                        train_step_row["mem_allocated_bytes"] = int(torch.cuda.memory_allocated())
                        train_step_row["mem_reserved_bytes"] = int(torch.cuda.memory_reserved())
                        train_step_row["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
                        train_step_row["peak_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                    _write_metrics_history(metrics_fh, train_step_row)
                    if writer is not None:
                        writer.add_scalar("train/loss", float(loss_val), step)
                        writer.add_scalar("train/perf/step_time_ms", float(step_time_ms), step)
                        _write_scalar_row_to_tensorboard(writer, "train_step", train_step_row, step)
                    if (
                        random_window_diagnostics_interval > 0
                        and step % int(random_window_diagnostics_interval) == 0
                    ):
                        random_window_diag_row = _build_iforward_random_window_diagnostics_row(
                            step=int(step),
                            result=result,
                            scheduler_info=scheduler_info,
                            diag_row=diag_row,
                        )
                        _write_metrics_history(metrics_fh, random_window_diag_row)
                        _write_scalar_row_to_tensorboard(
                            writer,
                            "train_step_diagnostics",
                            random_window_diag_row,
                            step,
                        )
                else:
                    train_step_row = {
                        "step": int(step),
                        "split": "train_step",
                        "scene_id": int(minimal_batch.get("scene_id", -1)),
                        "scene_dir": _scene_dir_str(minimal_batch.get("scene_id", -1)),
                        "segment_id": int(minimal_batch.get("segment_id", -1)),
                        "epoch_idx": int(scheduler_info.get("epoch_idx", -1)),
                        "global_step": int(scheduler_info.get("global_step", -1)),
                        "segment_local_step": int(scheduler_info.get("segment_local_step", -1)),
                        "segment_step_budget": int(scheduler_info.get("segment_step_budget", -1)),
                        "block_idx_in_segment": int(scheduler_info.get("block_idx_in_segment", -1)),
                        "block_idx_global": int(scheduler_info.get("block_idx_global", -1)),
                        "episode_idx_global": int(scheduler_info.get("episode_idx_global", -1)),
                        "source_frame_idx": int(scheduler_info.get("source_frame_idx", -1)),
                        "source_keyframe_idx": int(scheduler_info.get("source_keyframe_idx", -1)),
                        "source_image_ref": list(scheduler_info.get("source_image_ref", (-1, -1))),
                        "target_image_refs": [list(x) for x in scheduler_info.get("target_image_refs", [])],
                        "step_event_types": [str(ev.get("type", "")) for ev in step_events],
                        "U": int(scheduler_info.get("U", -1)),
                        "K_u_nominal": int(scheduler_info.get("K_u_nominal", -1)),
                        "K_u_effective": int(scheduler_info.get("K_u_effective", -1)),
                        "K_steps_effective": int(scheduler_info.get("K_steps_effective", -1)),
                        "K_steps": int(scheduler_info.get("K_steps", -1)),
                        "R_steps": int(scheduler_info.get("R_steps", -1)),
                        "T_steps": int(scheduler_info.get("T_steps", -1)),
                        "loss": float(loss_val),
                        "num_views": int(num_views),
                        "num_source_views": int(result.get("num_source_views", 0)),
                        "num_targets": int(result.get("num_targets", 0)),
                        "num_query_targets": int(result.get("num_query_targets", 0)),
                        "num_gaussians_bg": int(result.get("num_gaussians_bg", 0)),
                        "num_gaussians_distant": int(result.get("num_gaussians_distant", 0)),
                        "num_gaussians_rigid": int(result.get("num_gaussians_rigid", 0)),
                        "num_gaussians_sky": int(result.get("num_gaussians_sky", 0)),
                        "step_time_ms": float(step_time_ms),
                        "batch_fetch_ms": float(batch_fetch_ms),
                        "batch_convert_ms": float(batch_convert_ms),
                        "forward_ms": float(result.get("forward_ms", 0.0)),
                        "backward_ms": float(result.get("backward_ms", 0.0)),
                        "optimizer_ms": float(result.get("optimizer_ms", 0.0)),
                        "node_state_sync_update": bool(result.get("node_state_sync_update", False)),
                        "node_state_sync_reset": bool(result.get("node_state_sync_reset", False)),
                    }
                    for k, v in result.items():
                        if k.startswith("_") or k in train_step_row or k in {"pred_rgbs", "gt_images", "image_refs", "image_roles"}:
                            continue
                        if isinstance(v, bool):
                            train_step_row[k] = bool(v)
                        elif isinstance(v, int):
                            train_step_row[k] = int(v)
                        elif isinstance(v, float):
                            train_step_row[k] = float(v)
                        elif isinstance(v, str) and k in {"stage6/phase", "shape_name", "iforward/scheduler_version"}:
                            train_step_row[k] = str(v)
                        elif k == "iforward/window_block_ids" and isinstance(v, (list, tuple)):
                            train_step_row[k] = [int(x) for x in v]
                    if diag_row:
                        train_step_row.update(diag_row)
                    if perf_cfg["enable"] and perf_cfg["cuda_memory"] and torch.cuda.is_available():
                        train_step_row["mem_allocated_bytes"] = int(torch.cuda.memory_allocated())
                        train_step_row["mem_reserved_bytes"] = int(torch.cuda.memory_reserved())
                        train_step_row["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
                        train_step_row["peak_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                    _write_metrics_history(metrics_fh, train_step_row)
                    if writer is not None:
                        writer.add_scalar("train/loss", float(loss_val), step)
                        writer.add_scalar("train/perf/step_time_ms", float(step_time_ms), step)
                        for k, v in train_step_row.items():
                            if k in {"step", "split", "scene_dir", "source_image_ref", "target_image_refs", "step_event_types"}:
                                continue
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                writer.add_scalar(f"train_step/{k}", float(v), step)

            if diag_cfg["enable"] and diag_cfg["save_branch_renders"] and step % max(diag_cfg["interval"], 1) == 0:
                _save_diagnostic_renders(model, minimal_batch, step, cfg.log_dir)

            if image_trigger_mode == "raw_step_interval":
                scheduler_global_step = int(scheduler_info.get("global_step", step + 1))
                if scheduler_global_step == 0 or scheduler_global_step % int(image_trigger_interval_steps) == 0:
                    _save_train_images_for_result(
                        step=int(step),
                        result=result,
                        pred_rgbs=pred_rgbs,
                        gt_images=gt_images,
                        raw_batch=raw_batch,
                        log_dir=str(cfg.log_dir),
                        block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                        scene_id_fallback=scheduler_info.get("scene_id", -1),
                        pixel_camera_ids=pixel_camera_ids,
                        writer=writer,
                    )
                    _save_stage5_5_aux_debug_maps(
                        step=int(step),
                        aux_debug_maps=result.get("_stage5_5_aux_debug_maps"),
                        raw_batch=raw_batch,
                        log_dir=str(cfg.log_dir),
                        block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                        scene_id_fallback=scheduler_info.get("scene_id", -1),
                        writer=writer,
                    )
                    _save_stage5_6_debug_images(
                        step=int(step),
                        result=result,
                        raw_batch=raw_batch,
                        log_dir=str(cfg.log_dir),
                        block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                        scene_id_fallback=scheduler_info.get("scene_id", -1),
                        pixel_camera_ids=pixel_camera_ids,
                    )
            if image_trigger_mode == "episode_end":
                if any(ev.get("type") == "episode_end" for ev in step_events):
                    completed_blocks = int(scheduler_info.get("block_idx_global", -1)) + 1
                    if completed_blocks > 0 and completed_blocks % int(image_interval_blocks_equiv) == 0:
                        _save_train_images_for_result(
                            step=int(step),
                            result=result,
                            pred_rgbs=pred_rgbs,
                            gt_images=gt_images,
                            raw_batch=raw_batch,
                            log_dir=str(cfg.log_dir),
                            block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                            scene_id_fallback=scheduler_info.get("scene_id", -1),
                            pixel_camera_ids=pixel_camera_ids,
                            writer=writer,
                        )
                        _save_stage5_5_aux_debug_maps(
                            step=int(step),
                            aux_debug_maps=result.get("_stage5_5_aux_debug_maps"),
                            raw_batch=raw_batch,
                            log_dir=str(cfg.log_dir),
                            block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                            scene_id_fallback=scheduler_info.get("scene_id", -1),
                            writer=writer,
                        )
                        _save_stage5_6_debug_images(
                            step=int(step),
                            result=result,
                            raw_batch=raw_batch,
                            log_dir=str(cfg.log_dir),
                            block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                            scene_id_fallback=scheduler_info.get("scene_id", -1),
                            pixel_camera_ids=pixel_camera_ids,
                        )

            block_end_monitor_ms = 0.0
            for ev in step_events:
                if ev.get("type") != "block_end":
                    continue
                block_end_t0 = time.perf_counter()

                block_idx_global = int(ev.get("block_idx_global", 0))
                if image_trigger_mode == "block_end":
                    if block_idx_global >= 1 and (block_idx_global - 1) % int(image_interval_blocks_equiv) == 0:
                        _save_train_images_for_result(
                            step=int(step),
                            result=result,
                            pred_rgbs=pred_rgbs,
                            gt_images=gt_images,
                            raw_batch=raw_batch,
                            log_dir=str(cfg.log_dir),
                            block_idx_global=int(block_idx_global),
                            scene_id_fallback=ev.get("scene_id", -1),
                            pixel_camera_ids=pixel_camera_ids,
                            writer=writer,
                        )
                        _save_stage5_5_aux_debug_maps(
                            step=int(step),
                            aux_debug_maps=result.get("_stage5_5_aux_debug_maps"),
                            raw_batch=raw_batch,
                            log_dir=str(cfg.log_dir),
                            block_idx_global=int(block_idx_global),
                            scene_id_fallback=ev.get("scene_id", -1),
                            writer=writer,
                        )
                        _save_stage5_6_debug_images(
                            step=int(step),
                            result=result,
                            raw_batch=raw_batch,
                            log_dir=str(cfg.log_dir),
                            block_idx_global=int(block_idx_global),
                            scene_id_fallback=ev.get("scene_id", -1),
                            pixel_camera_ids=pixel_camera_ids,
                        )

                acc = block_loss_accum.pop(int(block_idx_global), None)
                mean_loss: Optional[float] = None
                if acc is not None and int(acc.get("loss_count", 0)) > 0:
                    mean_loss = float(acc["loss_sum"]) / float(acc["loss_count"])
                # Monitoring metrics are logging-only; detach to avoid extra autograd graph/memory.
                pred_rgbs_eval = [p.detach() for p in pred_rgbs]
                gt_images_eval = [g.detach() for g in gt_images]
                mse_full_vals = [
                    float(
                        torch.mean(
                            (torch.clamp(pred_rgbs_eval[v], 0.0, 1.0) - torch.clamp(gt_images_eval[v], 0.0, 1.0)) ** 2
                        ).item()
                    )
                    for v in range(num_views)
                ]
                mse_full_val = float(np.mean(mse_full_vals))
                mse_primary_vals = list(mse_full_vals)
                mse_non_sky_vals: List[float] = []

                metric_vals: Dict[str, float] = {}
                metric_scope = "full_image"
                non_sky_metric_views_light = 0
                if train_monitor_use_non_sky_region:
                    for v in range(num_views):
                        tgt_view = minimal_batch["targets"][v]
                        sky_mask = tgt_view.get("sky_mask")
                        if sky_mask is None and train_monitor_require_sky_mask:
                            raise ValueError(
                                "train monitor non-sky metrics require target['sky_mask'] "
                                f"(view={int(v)}, scene={int(minimal_batch.get('scene_id', -1))}, "
                                f"segment={int(minimal_batch.get('segment_id', -1))})."
                            )
                        if sky_mask is None:
                            continue
                        sm = sky_mask.to(device).float()
                        if sm.dim() == 3:
                            sm = sm.squeeze(-1)
                        if sm.shape != gt_images_eval[v].shape[:2]:
                            raise ValueError(
                                "train monitor sky_mask shape mismatch: "
                                f"sky_mask={tuple(sm.shape)} gt_hw={tuple(gt_images_eval[v].shape[:2])}"
                            )
                        non_sky_mask = (1.0 - sm).clamp(0.0, 1.0)
                        if int((non_sky_mask > 0.5).sum().item()) < train_monitor_min_valid_pixels:
                            continue
                        pred_c = torch.clamp(pred_rgbs_eval[v], 0.0, 1.0)
                        gt_c = torch.clamp(gt_images_eval[v], 0.0, 1.0)
                        w3 = non_sky_mask.unsqueeze(-1)
                        denom = float((non_sky_mask.sum() * 3.0).item())
                        if denom <= 0.0:
                            continue
                        mse_non_sky = float((((pred_c - gt_c) ** 2) * w3).sum().item() / denom)
                        mse_primary_vals[v] = mse_non_sky
                        mse_non_sky_vals.append(mse_non_sky)
                        non_sky_metric_views_light += 1
                    if non_sky_metric_views_light > 0:
                        metric_scope = "non_sky"

                mse_val = float(np.mean(mse_primary_vals))
                if enable_psnr:
                    psnr_light_list = [float(-10.0 * np.log10(max(float(m), 1.0e-12))) for m in mse_primary_vals]
                    metric_vals["psnr_mean"] = float(np.mean(psnr_light_list)) if psnr_light_list else 0.0
                    if train_monitor_include_per_view_metrics:
                        for v, psnr_v in enumerate(psnr_light_list):
                            metric_vals[f"psnr_view{v}"] = float(psnr_v)
                    if train_monitor_use_non_sky_region:
                        psnr_full_light = [float(-10.0 * np.log10(max(float(m), 1.0e-12))) for m in mse_full_vals]
                        metric_vals["psnr_full_mean"] = float(np.mean(psnr_full_light)) if psnr_full_light else 0.0
                        metric_vals["mse_full_mean"] = float(mse_full_val)
                        metric_vals["mse_non_sky_mean"] = _safe_mean(mse_non_sky_vals)
                        metric_vals["num_views_non_sky_metric"] = float(non_sky_metric_views_light)
                if enable_psnr and train_monitor_enable_heavy_metrics:
                    psnr_full_list: List[float] = []
                    ssim_full_list: List[float] = []
                    lpips_full_list: List[float] = []
                    psnr_primary_list: List[float] = []
                    ssim_primary_list: List[float] = []
                    lpips_primary_list: List[float] = []
                    psnr_non_sky_list: List[float] = []
                    ssim_non_sky_list: List[float] = []
                    lpips_non_sky_list: List[float] = []
                    non_sky_metric_views = 0
                    with torch.no_grad():
                        for v in range(num_views):
                            v_vals = _compute_metrics(
                                pred_rgb=pred_rgbs_eval[v],
                                gt_rgb=gt_images_eval[v],
                                psnr_metric=psnr_metric,
                                ssim_metric=ssim_metric,
                                lpips_metric=lpips_metric,
                                compute_psnr=True,
                                compute_heavy=True,
                            )
                            psnr_full = float(v_vals["psnr"])
                            ssim_full = float(v_vals["ssim"])
                            lpips_full = float(v_vals["lpips"])
                            psnr_full_list.append(psnr_full)
                            ssim_full_list.append(ssim_full)
                            lpips_full_list.append(lpips_full)

                            psnr_primary = psnr_full
                            ssim_primary = ssim_full
                            lpips_primary = lpips_full
                            if train_monitor_use_non_sky_region:
                                tgt_view = minimal_batch["targets"][v]
                                sky_mask = tgt_view.get("sky_mask")
                                if sky_mask is None and train_monitor_require_sky_mask:
                                    raise ValueError(
                                        "train monitor non-sky metrics require target['sky_mask'] "
                                        f"(view={int(v)}, scene={int(minimal_batch.get('scene_id', -1))}, "
                                        f"segment={int(minimal_batch.get('segment_id', -1))})."
                                    )
                                if sky_mask is not None:
                                    sm = sky_mask.to(device).float()
                                    if sm.dim() == 3:
                                        sm = sm.squeeze(-1)
                                    if sm.shape != gt_images_eval[v].shape[:2]:
                                        raise ValueError(
                                            "train monitor sky_mask shape mismatch: "
                                            f"sky_mask={tuple(sm.shape)} gt_hw={tuple(gt_images_eval[v].shape[:2])}"
                                        )
                                    non_sky_mask = (1.0 - sm).clamp(0.0, 1.0)
                                    if int((non_sky_mask > 0.5).sum().item()) >= train_monitor_min_valid_pixels:
                                        psnr_non = _compute_masked_psnr(pred_rgbs_eval[v], gt_images_eval[v], non_sky_mask)
                                        ssim_non = _compute_masked_ssim(pred_rgbs_eval[v], gt_images_eval[v], non_sky_mask)
                                        lpips_non = _compute_masked_lpips(
                                            pred_rgbs_eval[v],
                                            gt_images_eval[v],
                                            non_sky_mask,
                                            lpips_metric,
                                        )
                                        if psnr_non is not None:
                                            psnr_primary = float(psnr_non)
                                            psnr_non_sky_list.append(float(psnr_non))
                                        if ssim_non is not None:
                                            ssim_primary = float(ssim_non)
                                            ssim_non_sky_list.append(float(ssim_non))
                                        if lpips_non is not None:
                                            lpips_primary = float(lpips_non)
                                            lpips_non_sky_list.append(float(lpips_non))
                                        non_sky_metric_views += 1

                            psnr_primary_list.append(psnr_primary)
                            ssim_primary_list.append(ssim_primary)
                            lpips_primary_list.append(lpips_primary)
                            if train_monitor_include_per_view_metrics:
                                metric_vals[f"psnr_view{v}"] = float(psnr_primary)
                                metric_vals[f"psnr_full_view{v}"] = float(psnr_full)

                    metric_vals["psnr_mean"] = float(np.mean(psnr_primary_list)) if psnr_primary_list else 0.0
                    metric_vals["ssim_mean"] = float(np.mean(ssim_primary_list)) if ssim_primary_list else 0.0
                    metric_vals["lpips_mean"] = float(np.mean(lpips_primary_list)) if lpips_primary_list else 0.0
                    metric_vals["psnr_full_mean"] = float(np.mean(psnr_full_list)) if psnr_full_list else 0.0
                    metric_vals["ssim_full_mean"] = float(np.mean(ssim_full_list)) if ssim_full_list else 0.0
                    metric_vals["lpips_full_mean"] = float(np.mean(lpips_full_list)) if lpips_full_list else 0.0
                    metric_vals["psnr_non_sky_mean"] = _safe_mean(psnr_non_sky_list)
                    metric_vals["ssim_non_sky_mean"] = _safe_mean(ssim_non_sky_list)
                    metric_vals["lpips_non_sky_mean"] = _safe_mean(lpips_non_sky_list)
                    metric_vals["num_views_non_sky_metric"] = float(non_sky_metric_views)
                    if train_monitor_use_non_sky_region and non_sky_metric_views > 0:
                        metric_scope = "non_sky"

                    if (
                        save_train_views_psnr_below is not None
                        and low_psnr_train_images_subdir is not None
                        and train_monitor_enable_low_psnr_image_dump
                    ):
                        out_low = os.path.join(cfg.log_dir, "images", low_psnr_train_images_subdir)
                        tgt_meta = raw_batch.get("target") or {}
                        fi_t = tgt_meta.get("frame_indices")
                        ci_t = tgt_meta.get("cam_indices")
                        block_idx_global = int(ev.get("block_idx_global", 0))
                        sdir = _scene_folder_label_from_batch(raw_batch, ev.get("scene_id"))
                        thr = float(save_train_views_psnr_below)
                        n_psnr = min(num_views, len(psnr_primary_list))
                        if any(float(psnr_primary_list[v]) < thr for v in range(n_psnr)):
                            for v in range(num_views):
                                if v >= len(psnr_primary_list):
                                    break
                                if fi_t is not None and ci_t is not None and int(fi_t.shape[0]) > v and int(ci_t.shape[0]) > v:
                                    f_lab = int(fi_t[v].item())
                                    c_lab = int(ci_t[v].item())
                                    nusc_suf = _nuscenes_cam_id_suffix(pixel_camera_ids, c_lab)
                                    vsuf = (
                                        f"b{block_idx_global:06d}_sc{sdir}_v{v}_f{f_lab:05d}_c{c_lab}{nusc_suf}"
                                        f"_psnr{float(psnr_primary_list[v]):.2f}"
                                    )
                                else:
                                    vsuf = (
                                        f"b{block_idx_global:06d}_sc{sdir}_v{v}_psnr{float(psnr_primary_list[v]):.2f}"
                                    )
                                _save_image_triplet(
                                    step,
                                    pred_rgbs_eval[v],
                                    gt_images_eval[v],
                                    out_low,
                                    view_suffix=vsuf,
                                    save_error=False,
                                )

                psnr_log = (
                    f"{float(metric_vals['psnr_mean']):.2f}"
                    if enable_psnr and "psnr_mean" in metric_vals
                    else "n/a"
                )
                mean_loss_log = "n/a" if mean_loss is None else f"{float(mean_loss):.6f}"
                logger.info(
                    "BLOCK_END global_step=%s scene_id=%s scene_dir=%s segment=%s block_seg=%s block_global=%s "
                    "mean_loss=%s mse=%.6e metric_scope=%s psnr_mean=%s onepass=%d",
                    ev.get("global_step"),
                    ev.get("scene_id"),
                    _scene_dir_str(ev.get("scene_id", -1)),
                    ev.get("segment_id"),
                    ev.get("block_idx_in_segment"),
                    ev.get("block_idx_global"),
                    mean_loss_log,
                    mse_val,
                    metric_scope,
                    psnr_log,
                    int(result.get("src_backproject_pass_count", 0)),
                )

                def _optional_int_result(key: str) -> Optional[int]:
                    if key not in result or result.get(key) is None:
                        return None
                    return int(result[key])

                def _optional_float_result(key: str) -> Optional[float]:
                    if key not in result or result.get(key) is None:
                        return None
                    return float(result[key])

                row = {
                    "step": int(step),
                    "split": "train_monitor",
                    "scene_id": int(minimal_batch.get("scene_id", -1)),
                    "scene_dir": _scene_dir_str(minimal_batch.get("scene_id", -1)),
                    "segment_id": int(minimal_batch.get("segment_id", -1)),
                    "epoch_idx": int(scheduler_info.get("epoch_idx", -1)),
                    "global_step": int(scheduler_info.get("global_step", -1)),
                    "segment_local_step": int(scheduler_info.get("segment_local_step", -1)),
                    "segment_step_budget": int(scheduler_info.get("segment_step_budget", -1)),
                    "block_idx_in_segment": int(scheduler_info.get("block_idx_in_segment", -1)),
                    "block_idx_global": int(scheduler_info.get("block_idx_global", -1)),
                    "source_frame_idx": int(scheduler_info.get("source_frame_idx", -1)),
                    "source_keyframe_idx": int(scheduler_info.get("source_keyframe_idx", -1)),
                    "source_image_ref": list(scheduler_info.get("source_image_ref", (-1, -1))),
                    "target_image_refs": [list(x) for x in scheduler_info.get("target_image_refs", [])],
                    "U": int(scheduler_info.get("U", -1)),
                    "K_u_nominal": int(scheduler_info.get("K_u_nominal", -1)),
                    "K_u_effective": int(scheduler_info.get("K_u_effective", -1)),
                    "K_steps_effective": int(scheduler_info.get("K_steps_effective", -1)),
                    "K_steps": int(scheduler_info.get("K_steps", -1)),
                    "R_steps": int(scheduler_info.get("R_steps", -1)),
                    "T_steps": int(scheduler_info.get("T_steps", -1)),
                    "loss": float(loss_val),
                    "mean_loss_in_block": float(mean_loss) if mean_loss is not None else None,
                    "loss_l1": float(result.get("loss_l1", 0.0)),
                    "loss_ssim": float(result.get("loss_ssim", 0.0)),
                    "loss_mask": float(result.get("loss_mask", 0.0)),
                    "loss_opacity_entropy": float(result.get("loss_opacity_entropy", 0.0)),
                    "num_rigid_src_feat_valid": _optional_int_result("num_rigid_src_feat_valid"),
                    "num_rigid_update": _optional_int_result("num_rigid_update"),
                    "num_target_frames": int(result.get("num_target_frames", 0)),
                    "loss_effective_frames": int(result.get("loss_effective_frames", 0)),
                    "num_source_views": int(result.get("num_source_views", 0)),
                    "num_targets": int(result.get("num_targets", 0)),
                    "num_rigid_valid_src": _optional_int_result("num_rigid_valid_src"),
                    "rigid_valid_ratio": _optional_float_result("rigid_valid_ratio"),
                    "rigid_update_ratio": _optional_float_result("rigid_update_ratio"),
                    "rigid_update_among_feat_valid": _optional_float_result("rigid_update_among_feat_valid"),
                    "writeback_rigid_ratio": _optional_float_result("writeback_rigid_ratio"),
                    "num_bg_src_feat_valid": int(result.get("num_bg_src_feat_valid", 0)),
                    "num_bg_update": int(result.get("num_bg_update", 0)),
                    "bg_update_ratio": float(result.get("bg_update_ratio", 0.0)),
                    "num_distant_src_feat_valid": int(result.get("num_distant_src_feat_valid", 0)),
                    "num_distant_update": int(result.get("num_distant_update", 0)),
                    "distant_update_ratio": float(result.get("distant_update_ratio", 0.0)),
                    "writeback_bg_ratio": float(result.get("writeback_bg_ratio", 0.0)),
                    "writeback_distant_ratio": float(result.get("writeback_distant_ratio", 0.0)),
                    "num_gaussians_bg": _optional_int_result("num_gaussians_bg"),
                    "num_gaussians_distant": _optional_int_result("num_gaussians_distant"),
                    "num_gaussians_rigid": _optional_int_result("num_gaussians_rigid"),
                    "num_gaussians_sky": int(result.get("num_gaussians_sky", 0)),
                    "num_sky_src_feat_valid": int(result.get("num_sky_src_feat_valid", 0)),
                    "num_sky_update": int(result.get("num_sky_update", 0)),
                    "sky_update_ratio": float(result.get("sky_update_ratio", 0.0)),
                    "src_backproject_pass_count": int(result.get("src_backproject_pass_count", 0)),
                    "hidden_norm_bg_mean": float(result.get("hidden_norm_bg_mean", 0.0)),
                    "hidden_norm_distant_mean": float(result.get("hidden_norm_distant_mean", 0.0)),
                    "hidden_norm_rigid_mean": _optional_float_result("hidden_norm_rigid_mean"),
                    "hidden_norm_sky_mean": float(result.get("hidden_norm_sky_mean", 0.0)),
                    "grad_norm_bg": float(result.get("grad_norm_bg", 0.0)),
                    "grad_norm_distant": float(result.get("grad_norm_distant", 0.0)),
                    "grad_norm_rigid": _optional_float_result("grad_norm_rigid"),
                    "grad_norm_sky": float(result.get("grad_norm_sky", 0.0)),
                    "step_time_ms": float(step_time_ms),
                    "forward_ms": float(result.get("forward_ms", 0.0)),
                    "backward_ms": float(result.get("backward_ms", 0.0)),
                    "optimizer_ms": float(result.get("optimizer_ms", 0.0)),
                    "mse": float(mse_val),
                    "metric_scope": metric_scope,
                    "node_state_sync_update": bool(result.get("node_state_sync_update", False)),
                    "node_state_sync_reset": bool(result.get("node_state_sync_reset", False)),
                }
                extra_metric_prefixes = ("bg_", "rigid_", "distant_", "scene_", "perf_")
                if train_monitor_include_extra_result_metrics:
                    for k, v in result.items():
                        if not k.startswith(extra_metric_prefixes):
                            continue
                        if k in row:
                            continue
                        if isinstance(v, bool):
                            row[k] = bool(v)
                        elif isinstance(v, (int, float)):
                            row[k] = float(v)
                # Always persist optimizer/lr/loss namespace scalars for run diagnosis,
                # independent of include_extra_result_metrics.
                always_scalar_prefixes = (
                    "optimizer/",
                    "lr/",
                    "loss/",
                    "error_pred/",
                    "feedback/",
                    "branch/",
                    "phaseA/",
                    "phase_a/",
                    "phase_b/",
                    "stage6/",
                    "state/",
                    "memory/",
                    "mask/",
                    "node_state_",
                    "grad/",
                    "monitor/aux_",
                    "perf/aux_",
                )
                always_scalar_keys = {
                    "num_gaussians_bg",
                    "num_gaussians_distant",
                    "num_gaussians_rigid",
                }
                for k, v in result.items():
                    if k.startswith("_") or k in row:
                        continue
                    if k not in always_scalar_keys and not any(k.startswith(pf) for pf in always_scalar_prefixes):
                        continue
                    if isinstance(v, bool):
                        row[k] = bool(v)
                    elif isinstance(v, (int, float)):
                        row[k] = float(v)
                row["loss_mask_ratio"] = float(row["loss_mask"] / max(float(loss_val), 1e-8))
                row.update(metric_vals)
                if diag_row:
                    row.update(diag_row)
                if perf_cfg["enable"] and perf_cfg["cuda_memory"] and torch.cuda.is_available():
                    row["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
                    row["peak_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved())

                _write_metrics_history(metrics_fh, row)
                if writer is not None:
                    writer.add_scalar("train/loss", float(loss_val), step)
                    if mean_loss is not None:
                        writer.add_scalar("train/mean_loss_in_block", float(mean_loss), step)
                    writer.add_scalar("train/mse", float(mse_val), step)
                    for scalar_key in (
                        "num_bg_update",
                        "num_distant_update",
                        "num_gaussians_bg",
                        "num_gaussians_distant",
                        "num_gaussians_rigid",
                        "num_gaussians_sky",
                        "src_backproject_pass_count",
                    ):
                        if scalar_key in result and isinstance(result[scalar_key], (int, float)):
                            writer.add_scalar(f"train/{scalar_key}", float(result[scalar_key]), step)
                    for k, v in metric_vals.items():
                        writer.add_scalar(f"train/{k}", float(v), step)
                    if train_monitor_include_extra_result_metrics:
                        for k, v in result.items():
                            if not k.startswith(extra_metric_prefixes):
                                continue
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f"train/{k}", float(v), step)
                    for k, v in result.items():
                        if k.startswith("_") or (
                            k not in always_scalar_keys and not any(k.startswith(pf) for pf in always_scalar_prefixes)
                        ):
                            continue
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f"train/{k}", float(v), step)
                    writer.add_scalar("train/perf/step_time_ms", float(step_time_ms), step)
                block_end_monitor_ms += float((time.perf_counter() - block_end_t0) * 1000.0)

            validation_ms = 0.0
            if callable(episode_end_hook) and len(hook_due_episode_counters) > 0:
                val_t0 = time.perf_counter()
                for ep_counter in hook_due_episode_counters:
                    episode_end_hook(
                        cfg=cfg,
                        dataset=dataset,
                        model=model,
                        device=device,
                        trigger_train_episode_counter=int(ep_counter),
                        trigger_step=int(step),
                        minimal_batch=minimal_batch,
                        scheduler_info=scheduler_info,
                        step_events=step_events,
                        psnr_metric=psnr_metric,
                        ssim_metric=ssim_metric,
                        lpips_metric=lpips_metric,
                        metrics_fh=metrics_fh,
                        writer=writer,
                    )
                validation_ms += float((time.perf_counter() - val_t0) * 1000.0)
            if bool(validation_v7_cfg.eval_enable) and len(validation_due_episode_counters) > 0:
                val_t0 = time.perf_counter()
                for ep_counter in validation_due_episode_counters:
                    _run_validation_v7_round(
                        cfg=cfg,
                        dataset=dataset,
                        model=model,
                        specs=validation_specs,
                        validation_cfg=validation_v7_cfg,
                        device=device,
                        trigger_train_episode_counter=int(ep_counter),
                        trigger_step=int(step),
                        psnr_metric=psnr_metric,
                        ssim_metric=ssim_metric,
                        lpips_metric=lpips_metric,
                        metrics_fh=metrics_fh,
                        writer=writer,
                    )
                validation_ms += float((time.perf_counter() - val_t0) * 1000.0)
            if callable(step_end_hook):
                val_t0 = time.perf_counter()
                step_end_hook(
                    cfg=cfg,
                    dataset=dataset,
                    model=model,
                    device=device,
                    trigger_train_episode_counter=int(train_episode_counter),
                    trigger_step=int(step),
                    minimal_batch=minimal_batch,
                    scheduler_info=scheduler_info,
                    step_events=step_events,
                    psnr_metric=psnr_metric,
                    ssim_metric=ssim_metric,
                    lpips_metric=lpips_metric,
                    metrics_fh=metrics_fh,
                    writer=writer,
                )
                validation_ms += float((time.perf_counter() - val_t0) * 1000.0)
            if defer_node_state_reset_for_episode_hook:
                reset_info = _reset_model_node_state_and_release_cuda(
                    model,
                    reason="deferred_episode_hook",
                    step=int(step),
                    scheduler_info=scheduler_info,
                    log_reset=bool(log_node_state_reset),
                )
                reset_before = dict(reset_info.get("before") or {})
                reset_after = dict(reset_info.get("after") or {})
                reset_row: Dict[str, Any] = {
                    "step": int(step),
                    "split": "node_state_reset",
                    "reason": "deferred_episode_hook",
                    "scene_id": int(scheduler_info.get("scene_id", -1)),
                    "scene_dir": _scene_dir_str(scheduler_info.get("scene_id", -1)),
                    "segment_id": int(scheduler_info.get("segment_id", -1)),
                    "epoch_idx": int(scheduler_info.get("epoch_idx", -1)),
                    "global_step": int(scheduler_info.get("global_step", -1)),
                    "segment_local_step": int(scheduler_info.get("segment_local_step", -1)),
                    "step_event_types": [str(ev.get("type", "")) for ev in step_events],
                    "node_state_cache_segments_bg_before": int(reset_before.get("bg", 0)),
                    "node_state_cache_segments_distant_before": int(reset_before.get("distant", 0)),
                    "node_state_cache_segments_rigid_before": int(reset_before.get("rigid", 0)),
                    "node_state_cache_segments_sky_before": int(reset_before.get("sky", 0)),
                    "node_state_cache_segments_bg_after": int(reset_after.get("bg", 0)),
                    "node_state_cache_segments_distant_after": int(reset_after.get("distant", 0)),
                    "node_state_cache_segments_rigid_after": int(reset_after.get("rigid", 0)),
                    "node_state_cache_segments_sky_after": int(reset_after.get("sky", 0)),
                    "cuda_empty_cache": bool(reset_info.get("cuda_empty_cache", False)),
                }
                if bool(node_state_reset_cuda_memory) and torch.cuda.is_available():
                    reset_row["memory/allocated_gb"] = float(torch.cuda.memory_allocated() / (1024.0 ** 3))
                    reset_row["memory/reserved_gb"] = float(torch.cuda.memory_reserved() / (1024.0 ** 3))
                    reset_row["memory/peak_gb"] = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
                if bool(write_node_state_reset_metrics):
                    _write_metrics_history(metrics_fh, reset_row)

            _drop_result_tensor_payloads(result)
            pred_rgbs = []
            gt_images = []
            if "pred_rgbs_eval" in locals():
                pred_rgbs_eval = []
            if "gt_images_eval" in locals():
                gt_images_eval = []
            if "train_step_row" in locals():
                train_step_row = {}
            if "row" in locals():
                row = {}
            if "metric_vals" in locals():
                metric_vals = {}
            if "diag_row" in locals():
                diag_row = {}
            raw_batch = {}
            minimal_batch = {}
            result = None
            did_empty_cache = False
            if (
                perf_empty_cache_interval_steps > 0
                and step % int(perf_empty_cache_interval_steps) == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
                did_empty_cache = True
            if (
                perf_cleanup_metrics_interval_steps > 0
                and step % perf_cleanup_metrics_interval_steps == 0
                and perf_cfg["enable"]
                and perf_cfg["cuda_memory"]
                and torch.cuda.is_available()
            ):
                iter_wall_pre_checkpoint_ms = float((time.perf_counter() - iter_t0) * 1000.0)
                residual_pre_checkpoint_ms = max(
                    0.0,
                    float(iter_wall_pre_checkpoint_ms)
                    - float(step_time_ms)
                    - float(batch_fetch_ms)
                    - float(batch_convert_ms)
                    - float(block_end_monitor_ms)
                    - float(validation_ms),
                )
                cleanup_row = {
                    "step": int(step),
                    "split": "step_cleanup",
                    "scene_id": int(scheduler_info.get("scene_id", -1)),
                    "scene_dir": _scene_dir_str(scheduler_info.get("scene_id", -1)),
                    "segment_id": int(scheduler_info.get("segment_id", -1)),
                    "global_step": int(scheduler_info.get("global_step", -1)),
                    "iter_wall_pre_checkpoint_ms": float(iter_wall_pre_checkpoint_ms),
                    "train_step_ms": float(step_time_ms),
                    "batch_fetch_ms": float(batch_fetch_ms),
                    "batch_convert_ms": float(batch_convert_ms),
                    "block_end_monitor_ms": float(block_end_monitor_ms),
                    "validation_ms": float(validation_ms),
                    "residual_non_train_pre_checkpoint_ms": float(residual_pre_checkpoint_ms),
                    "mem_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "mem_reserved_bytes": int(torch.cuda.memory_reserved()),
                    "peak_mem_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_mem_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "cuda_empty_cache": bool(did_empty_cache),
                }
                if (
                    perf_live_tensor_summary_interval_steps > 0
                    and step % perf_live_tensor_summary_interval_steps == 0
                ):
                    cleanup_row.update(
                        _cuda_component_memory_summary(
                            model=model,
                            dataset=dataset,
                            include_live=True,
                            topk=int(perf_live_tensor_summary_topk),
                        )
                    )
                _write_metrics_history(metrics_fh, cleanup_row)

            checkpoint_ms = 0.0
            if save_every and step > 0 and step % save_every == 0:
                ckpt_t0 = time.perf_counter()
                ckpt_prefix = _checkpoint_prefix_for_cfg(cfg)
                ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"{ckpt_prefix}_step{step}.pt")
                ckpt_payload = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                }
                if hasattr(model, "build_light_checkpoint_extra"):
                    ckpt_payload.update(model.build_light_checkpoint_extra(step=int(step)))
                ckpt_payload.update(
                    _checkpoint_runtime_extra(
                        model=model,
                        scheduler=scheduler,
                        train_episode_counter=int(train_episode_counter),
                        step=int(step),
                        start_step=int(start_step),
                    )
                )
                torch.save(
                    ckpt_payload,
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)
                checkpoint_ms = float((time.perf_counter() - ckpt_t0) * 1000.0)

            iter_wall_ms = float((time.perf_counter() - iter_t0) * 1000.0)
            residual_non_train_ms = float(
                iter_wall_ms
                - step_time_ms
                - batch_fetch_ms
                - batch_convert_ms
                - block_end_monitor_ms
                - validation_ms
                - checkpoint_ms
            )
            if perf_cfg["enable"] and step % log_interval == 0:
                logger.info(
                    "Perf iter=%s iter_wall_ms=%.2f train_step_ms=%.2f fetch_ms=%.2f convert_ms=%.2f block_end_ms=%.2f validation_ms=%.2f checkpoint_ms=%.2f residual_non_train_ms=%.2f",
                    step,
                    iter_wall_ms,
                    step_time_ms,
                    batch_fetch_ms,
                    batch_convert_ms,
                    block_end_monitor_ms,
                    validation_ms,
                    checkpoint_ms,
                    residual_non_train_ms,
                )

        summary = {
            "start_step": int(start_step),
            "final_step": int(max_iterations - 1),
            "train": {"loss": float(last_train_loss)},
            "gs_stats": {
                "avg_num_gaussians_bg": sum_num_gaussians_bg / max(total_steps, 1),
                "avg_num_gaussians_rigid": sum_num_gaussians_rigid / max(total_steps, 1),
                "avg_num_gaussians_distant": sum_num_gaussians_distant / max(total_steps, 1),
                "avg_num_gaussians_sky": sum_num_gaussians_sky / max(total_steps, 1),
            },
            "profiling": {
                "avg_step_time_ms": float(sum_step_time_ms / max(total_steps, 1)),
                "p50_step_time_ms": _percentile(step_time_ms_hist, 50.0),
                "p95_step_time_ms": _percentile(step_time_ms_hist, 95.0),
                "peak_mem_bytes": int(peak_mem_bytes),
                "peak_mem_reserved_bytes": int(peak_mem_reserved_bytes),
            },
        }
        with open(os.path.join(cfg.log_dir, "metrics_final.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        try:
            upload_experiment_summary(cfg.log_dir, summary)
        except Exception:
            logger.exception("Vika upload failed for log_dir=%s", cfg.log_dir)
    finally:
        if metrics_fh is not None:
            metrics_fh.close()
        if writer is not None:
            try:
                tb_dir = os.path.join(cfg.log_dir, "tb")
                os.makedirs(tb_dir, exist_ok=True)
                writer.flush()
                writer.close()
            except OSError as exc:
                logger.warning("TensorBoard writer close/flush failed (log_dir may have been removed): %s", exc)
        if hasattr(dataset, "shutdown_preload"):
            dataset.shutdown_preload()

    final_ckpt = os.path.join(cfg.log_dir, "checkpoints", f"{_checkpoint_prefix_for_cfg(cfg)}_final.pt")
    final_payload = {
        "step": max_iterations - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
    }
    if hasattr(model, "build_light_checkpoint_extra"):
        final_payload.update(model.build_light_checkpoint_extra(step=int(max_iterations - 1)))
    final_payload.update(
        _checkpoint_runtime_extra(
            model=model,
            scheduler=scheduler,
            train_episode_counter=int(train_episode_counter),
            step=int(max_iterations - 1),
            start_step=int(start_step),
        )
    )
    torch.save(
        final_payload,
        final_ckpt,
    )
    logger.info("Saved final checkpoint to %s", final_ckpt)


if __name__ == "__main__":
    main()
