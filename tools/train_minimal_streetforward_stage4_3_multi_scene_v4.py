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
import inspect
import io
import json
import logging
import os
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
from tools.streetforward_test_export import save_3dgs_state
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
TRAINER_CLASS = MinimalStreetForwardStage4_3
DEFAULT_CONFIG_FILE = "configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml"


def _scene_dir_str(scene_id: Any) -> str:
    """Zero-padded scene folder name (e.g. 5 -> '005'), matching preprocessed data dirs like nuScenes export."""
    try:
        s = int(scene_id)
    except (TypeError, ValueError):
        return "unknown"
    if s < 0:
        return "unknown"
    return f"{s:03d}"


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
    if int(block_idx_global) < 1:
        return
    out_dir = os.path.join(log_dir, "images", "train")
    tgt_meta = raw_batch.get("target") or {}
    fi_t = tgt_meta.get("frame_indices")
    ci_t = tgt_meta.get("cam_indices")
    sc_lab = _scene_folder_label_from_batch(raw_batch, scene_id_fallback)
    for v in range(len(pred_rgbs)):
        if fi_t is not None and ci_t is not None and int(fi_t.shape[0]) > v and int(ci_t.shape[0]) > v:
            f_lab = int(fi_t[v].item())
            c_lab = int(ci_t[v].item())
            nusc_suf = _nuscenes_cam_id_suffix(pixel_camera_ids, c_lab)
            vsuf = f"b{int(block_idx_global):06d}_sc{sc_lab}_v{v}_f{f_lab:05d}_c{c_lab}{nusc_suf}"
        else:
            vsuf = f"b{int(block_idx_global):06d}_sc{sc_lab}_view{v}"
        _save_image_triplet(
            step,
            pred_rgbs[v],
            gt_images[v],
            out_dir,
            view_suffix=vsuf,
            save_error=False,
        )


_EXPORT_3DGS_SCENE_TRIGGER_MODES = ("block_end", "raw_step_interval", "episode_end")


def _parse_export_3dgs_scene_cfg(
    cfg: Any, scheduler_steps_per_block: int
) -> Optional[Dict[str, Any]]:
    """Parse logging.export_3dgs_scene.

    Returns None when the sub-block is missing or enable=False (zero-IO no-op).
    Otherwise validates fields fast-fail and returns a dict with keys:
      enable, interval_blocks, interval_steps, subdir, trigger, include_hidden.
    """
    logging_cfg = cfg.get("logging") if hasattr(cfg, "get") else None
    if logging_cfg is None:
        return None
    raw = logging_cfg.get("export_3dgs_scene") if hasattr(logging_cfg, "get") else None
    if raw is None:
        return None
    if not hasattr(raw, "get"):
        raise ValueError("logging.export_3dgs_scene must be a mapping when present.")
    if "enable" not in raw:
        raise ValueError("logging.export_3dgs_scene.enable is required when the sub-block is present.")
    enable = bool(raw.get("enable"))
    if not enable:
        return None
    if "interval_blocks" not in raw:
        raise ValueError("logging.export_3dgs_scene.interval_blocks is required when enable=true.")
    interval_blocks = int(raw.get("interval_blocks"))
    if interval_blocks < 1:
        raise ValueError(
            f"logging.export_3dgs_scene.interval_blocks must be >= 1, got {interval_blocks}"
        )
    if "subdir" not in raw:
        raise ValueError("logging.export_3dgs_scene.subdir is required when enable=true.")
    subdir = str(raw.get("subdir")).strip()
    if not subdir:
        raise ValueError("logging.export_3dgs_scene.subdir must be non-empty.")
    if "trigger" not in raw:
        raise ValueError("logging.export_3dgs_scene.trigger is required when enable=true.")
    trigger = str(raw.get("trigger")).strip()
    if trigger not in _EXPORT_3DGS_SCENE_TRIGGER_MODES:
        raise ValueError(
            "logging.export_3dgs_scene.trigger must be one of "
            f"{list(_EXPORT_3DGS_SCENE_TRIGGER_MODES)}, got {trigger!r}"
        )
    if "include_hidden" not in raw:
        raise ValueError("logging.export_3dgs_scene.include_hidden is required when enable=true.")
    include_hidden_raw = raw.get("include_hidden")
    if not isinstance(include_hidden_raw, bool):
        raise ValueError(
            "logging.export_3dgs_scene.include_hidden must be a bool, "
            f"got {type(include_hidden_raw).__name__}"
        )
    steps_per_block = int(scheduler_steps_per_block)
    if steps_per_block < 1:
        steps_per_block = 1
    return {
        "enable": True,
        "interval_blocks": int(interval_blocks),
        "interval_steps": int(interval_blocks * steps_per_block),
        "subdir": subdir,
        "trigger": trigger,
        "include_hidden": bool(include_hidden_raw),
    }


def _save_3dgs_scene_snapshot(
    *,
    model: Any,
    minimal_batch: Dict[str, Any],
    log_dir: str,
    subdir: str,
    block_idx_global: int,
    step: int,
    include_hidden: bool,
) -> None:
    """Export the current optimized 3DGS scene state and save to log_dir/subdir/."""
    src_frame = minimal_batch.get("source_frame_idx")
    if src_frame is None:
        raise ValueError(
            "export_3dgs_scene requires minimal_batch['source_frame_idx'] to align rigid branch."
        )
    scene_id = int(minimal_batch.get("scene_id", -1))
    segment_id = int(minimal_batch.get("segment_id", -1))
    out_dir = os.path.join(str(log_dir), str(subdir))
    os.makedirs(out_dir, exist_ok=True)
    fname = (
        f"scene_{scene_id:03d}_seg_{segment_id:03d}"
        f"_block_{int(block_idx_global):06d}_step_{int(step):08d}.pt"
    )
    out_path = os.path.join(out_dir, fname)
    save_t0 = time.perf_counter()
    state = model.export_3dgs_state(
        minimal_batch,
        include_hidden=bool(include_hidden),
        rigid_export_frame_idx=int(src_frame),
    )
    save_3dgs_state(out_path, state)
    save_ms = float((time.perf_counter() - save_t0) * 1000.0)
    logger.info(
        "export_3dgs_scene saved: path=%s scene_id=%s segment_id=%s block=%s step=%s elapsed_ms=%.2f",
        out_path,
        scene_id,
        segment_id,
        int(block_idx_global),
        int(step),
        save_ms,
    )


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

            model.reset_node_state()
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
    model.reset_node_state()


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
    parser.add_argument("opts", nargs="*", help="Override config")
    args = parser.parse_args()

    cfg = setup(args)
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

    if cfg.get("one_segment") is not None:
        raise ValueError(
            "multi_scene training: remove `one_segment` from config; "
            f"use {DEFAULT_CONFIG_FILE}."
        )
    train_ids = list(cfg.data.train_scene_ids)
    if len(train_ids) < 2:
        raise ValueError("multi_scene training requires len(data.train_scene_ids) >= 2")
    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment(cfg)
    if fixed_scene_id is not None or fixed_segment_id is not None:
        raise ValueError(
            "multi_scene training requires scheduler traversal fixed_scene_id and fixed_segment_id to be null "
            "(unset one_segment and traversal overrides)."
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

    max_iterations = args.max_steps or cfg.training.get("max_iterations", 1000)
    log_interval = cfg.training.get("log_interval", 50)
    save_every = cfg.training.get("save_checkpoint_freq", 500)
    enable_psnr = bool(cfg.eval.get("enable_psnr", True))
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
        "include_extra_result_metrics=%s low_psnr_image_dump=%s",
        bool(train_monitor_enable_heavy_metrics),
        bool(train_monitor_include_per_view_metrics),
        bool(train_monitor_include_extra_result_metrics),
        bool(train_monitor_enable_low_psnr_image_dump),
    )
    enable_jsonl_metrics = bool(cfg.logging.get("enable_jsonl_metrics", True))
    metrics_history_append = bool(cfg.logging.get("metrics_history_append", True))
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
    export_3dgs_scene_cfg = _parse_export_3dgs_scene_cfg(cfg, scheduler_steps_per_block)
    if export_3dgs_scene_cfg is not None:
        logger.info(
            "export_3dgs_scene enabled: trigger=%s interval_blocks=%s interval_steps=%s subdir=%s include_hidden=%s",
            export_3dgs_scene_cfg["trigger"],
            int(export_3dgs_scene_cfg["interval_blocks"]),
            int(export_3dgs_scene_cfg["interval_steps"]),
            export_3dgs_scene_cfg["subdir"],
            bool(export_3dgs_scene_cfg["include_hidden"]),
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
    _load_init_checkpoint(
        args.init_checkpoint,
        model,
        device,
        weights_only=bool(args.init_weights_only),
    )

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    metrics_fh: Optional[TextIO] = None
    writer: Optional[Any] = None
    result: Optional[Dict[str, Any]] = None
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
        if bool(validation_v7_cfg.eval_enable and validation_v7_cfg.run_at_train_start):
            _run_validation_v7_round(
                cfg=cfg,
                dataset=dataset,
                model=model,
                specs=validation_specs,
                validation_cfg=validation_v7_cfg,
                device=device,
                trigger_train_episode_counter=0,
                trigger_step=-1,
                psnr_metric=psnr_metric,
                ssim_metric=ssim_metric,
                lpips_metric=lpips_metric,
                metrics_fh=metrics_fh,
                writer=writer,
            )

        for step in range(max_iterations):
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
            for ev in step_events:
                if ev.get("type") == "episode_end":
                    train_episode_counter += 1
                    if bool(validation_v7_cfg.eval_enable):
                        if train_episode_counter % int(validation_v7_cfg.validate_every_n_episodes) == 0:
                            validation_due_episode_counters.append(int(train_episode_counter))
            scheduler_node_sync = _build_scheduler_node_sync(cfg, scheduler_info, step_events)
            if scheduler_node_sync is None:
                scheduler_node_sync = _build_scheduler_node_sync_v8_fallback(cfg, scheduler_info, step_events)
            defer_node_state_reset_for_block_exit_record = False
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
                if defer_node_state_reset_for_block_exit_record:
                    model.reset_node_state()
            stage5_5_block_exit_monitor: Dict[str, Any] = {}
            if model_stage == "5_5" and hasattr(model, "record_block_history"):
                for ev in step_events:
                    if str(ev.get("type", "")) != "block_exit":
                        continue
                    rec_metrics = model.record_block_history(minimal_batch, ev)
                    for k, v in rec_metrics.items():
                        sk = str(k)
                        if isinstance(v, bool):
                            stage5_5_block_exit_monitor[sk] = bool(v)
                        elif isinstance(v, (int, float)):
                            stage5_5_block_exit_monitor[sk] = float(v)
            loss_val = float(result["loss"])
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

            if diag_cfg["enable"] and diag_cfg["save_branch_renders"] and step % max(diag_cfg["interval"], 1) == 0:
                _save_diagnostic_renders(model, minimal_batch, step, cfg.log_dir)

            if image_trigger_mode == "raw_step_interval":
                scheduler_global_step = int(scheduler_info.get("global_step", step + 1))
                if scheduler_global_step > 0 and scheduler_global_step % int(image_trigger_interval_steps) == 0:
                    _save_train_monitor_triplets(
                        step=int(step),
                        pred_rgbs=pred_rgbs,
                        gt_images=gt_images,
                        raw_batch=raw_batch,
                        log_dir=str(cfg.log_dir),
                        block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                        scene_id_fallback=scheduler_info.get("scene_id", -1),
                        pixel_camera_ids=pixel_camera_ids,
                    )
            if (
                export_3dgs_scene_cfg is not None
                and export_3dgs_scene_cfg["trigger"] == "raw_step_interval"
            ):
                scheduler_global_step_export = int(scheduler_info.get("global_step", step + 1))
                if (
                    scheduler_global_step_export > 0
                    and scheduler_global_step_export % int(export_3dgs_scene_cfg["interval_steps"]) == 0
                ):
                    _save_3dgs_scene_snapshot(
                        model=model,
                        minimal_batch=minimal_batch,
                        log_dir=str(cfg.log_dir),
                        subdir=str(export_3dgs_scene_cfg["subdir"]),
                        block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                        step=int(step),
                        include_hidden=bool(export_3dgs_scene_cfg["include_hidden"]),
                    )
            if image_trigger_mode == "episode_end":
                if any(ev.get("type") == "episode_end" for ev in step_events):
                    completed_blocks = int(scheduler_info.get("block_idx_global", -1)) + 1
                    if completed_blocks > 0 and completed_blocks % int(image_interval_blocks_equiv) == 0:
                        _save_train_monitor_triplets(
                            step=int(step),
                            pred_rgbs=pred_rgbs,
                            gt_images=gt_images,
                            raw_batch=raw_batch,
                            log_dir=str(cfg.log_dir),
                            block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                            scene_id_fallback=scheduler_info.get("scene_id", -1),
                            pixel_camera_ids=pixel_camera_ids,
                        )
            if (
                export_3dgs_scene_cfg is not None
                and export_3dgs_scene_cfg["trigger"] == "episode_end"
            ):
                if any(ev.get("type") == "episode_end" for ev in step_events):
                    completed_blocks_export = int(scheduler_info.get("block_idx_global", -1)) + 1
                    if (
                        completed_blocks_export > 0
                        and completed_blocks_export % int(export_3dgs_scene_cfg["interval_blocks"]) == 0
                    ):
                        _save_3dgs_scene_snapshot(
                            model=model,
                            minimal_batch=minimal_batch,
                            log_dir=str(cfg.log_dir),
                            subdir=str(export_3dgs_scene_cfg["subdir"]),
                            block_idx_global=int(scheduler_info.get("block_idx_global", 0)),
                            step=int(step),
                            include_hidden=bool(export_3dgs_scene_cfg["include_hidden"]),
                        )

            block_end_monitor_ms = 0.0
            for ev in step_events:
                if ev.get("type") != "block_end":
                    continue
                block_end_t0 = time.perf_counter()

                block_idx_global = int(ev.get("block_idx_global", 0))
                if image_trigger_mode == "block_end":
                    if block_idx_global >= 1 and (block_idx_global - 1) % int(image_interval_blocks_equiv) == 0:
                        _save_train_monitor_triplets(
                            step=int(step),
                            pred_rgbs=pred_rgbs,
                            gt_images=gt_images,
                            raw_batch=raw_batch,
                            log_dir=str(cfg.log_dir),
                            block_idx_global=int(block_idx_global),
                            scene_id_fallback=ev.get("scene_id", -1),
                            pixel_camera_ids=pixel_camera_ids,
                        )
                if (
                    export_3dgs_scene_cfg is not None
                    and export_3dgs_scene_cfg["trigger"] == "block_end"
                ):
                    if (
                        block_idx_global >= 1
                        and (block_idx_global - 1) % int(export_3dgs_scene_cfg["interval_blocks"]) == 0
                    ):
                        _save_3dgs_scene_snapshot(
                            model=model,
                            minimal_batch=minimal_batch,
                            log_dir=str(cfg.log_dir),
                            subdir=str(export_3dgs_scene_cfg["subdir"]),
                            block_idx_global=int(block_idx_global),
                            step=int(step),
                            include_hidden=bool(export_3dgs_scene_cfg["include_hidden"]),
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
                    "num_rigid_src_feat_valid": int(result.get("num_rigid_src_feat_valid", 0)),
                    "num_rigid_update": int(result.get("num_rigid_update", 0)),
                    "num_target_frames": int(result.get("num_target_frames", 0)),
                    "loss_effective_frames": int(result.get("loss_effective_frames", 0)),
                    "num_source_views": int(result.get("num_source_views", 0)),
                    "num_targets": int(result.get("num_targets", 0)),
                    "num_rigid_valid_src": int(result.get("num_rigid_valid_src", 0)),
                    "rigid_valid_ratio": float(result.get("rigid_valid_ratio", 0.0)),
                    "rigid_update_ratio": float(result.get("rigid_update_ratio", 0.0)),
                    "rigid_update_among_feat_valid": float(result.get("rigid_update_among_feat_valid", 0.0)),
                    "writeback_rigid_ratio": float(result.get("writeback_rigid_ratio", 0.0)),
                    "num_bg_src_feat_valid": int(result.get("num_bg_src_feat_valid", 0)),
                    "num_bg_update": int(result.get("num_bg_update", 0)),
                    "bg_update_ratio": float(result.get("bg_update_ratio", 0.0)),
                    "num_distant_src_feat_valid": int(result.get("num_distant_src_feat_valid", 0)),
                    "num_distant_update": int(result.get("num_distant_update", 0)),
                    "distant_update_ratio": float(result.get("distant_update_ratio", 0.0)),
                    "writeback_bg_ratio": float(result.get("writeback_bg_ratio", 0.0)),
                    "writeback_distant_ratio": float(result.get("writeback_distant_ratio", 0.0)),
                    "num_gaussians_sky": int(result.get("num_gaussians_sky", 0)),
                    "num_sky_src_feat_valid": int(result.get("num_sky_src_feat_valid", 0)),
                    "num_sky_update": int(result.get("num_sky_update", 0)),
                    "sky_update_ratio": float(result.get("sky_update_ratio", 0.0)),
                    "src_backproject_pass_count": int(result.get("src_backproject_pass_count", 0)),
                    "hidden_norm_bg_mean": float(result.get("hidden_norm_bg_mean", 0.0)),
                    "hidden_norm_distant_mean": float(result.get("hidden_norm_distant_mean", 0.0)),
                    "hidden_norm_rigid_mean": float(result.get("hidden_norm_rigid_mean", 0.0)),
                    "hidden_norm_sky_mean": float(result.get("hidden_norm_sky_mean", 0.0)),
                    "grad_norm_bg": float(result.get("grad_norm_bg", 0.0)),
                    "grad_norm_distant": float(result.get("grad_norm_distant", 0.0)),
                    "grad_norm_rigid": float(result.get("grad_norm_rigid", 0.0)),
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
                    "stage5_5_",
                    "scheduler_v9/",
                    "history/",
                )
                for k, v in result.items():
                    if k.startswith("_") or k in row:
                        continue
                    if not any(k.startswith(pf) for pf in always_scalar_prefixes):
                        continue
                    if isinstance(v, bool):
                        row[k] = bool(v)
                    elif isinstance(v, (int, float)):
                        row[k] = float(v)
                if isinstance(result.get("stage5_5_role"), str):
                    row["stage5_5_role"] = str(result["stage5_5_role"])
                row.update(stage5_5_block_exit_monitor)
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
                    writer.add_scalar("train/num_bg_update", int(result.get("num_bg_update", 0)), step)
                    writer.add_scalar("train/num_distant_update", int(result.get("num_distant_update", 0)), step)
                    writer.add_scalar("train/num_gaussians_sky", int(result.get("num_gaussians_sky", 0)), step)
                    writer.add_scalar("train/src_backproject_pass_count", int(result.get("src_backproject_pass_count", 0)), step)
                    for k, v in metric_vals.items():
                        writer.add_scalar(f"train/{k}", float(v), step)
                    if train_monitor_include_extra_result_metrics:
                        for k, v in result.items():
                            if not k.startswith(extra_metric_prefixes):
                                continue
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f"train/{k}", float(v), step)
                    writer.add_scalar("train/perf/step_time_ms", float(step_time_ms), step)
                block_end_monitor_ms += float((time.perf_counter() - block_end_t0) * 1000.0)

            validation_ms = 0.0
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
                validation_ms = float((time.perf_counter() - val_t0) * 1000.0)

            checkpoint_ms = 0.0
            if save_every and step > 0 and step % save_every == 0:
                ckpt_t0 = time.perf_counter()
                ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_step{step}.pt")
                ckpt_payload = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                }
                if hasattr(model, "build_light_checkpoint_extra"):
                    ckpt_payload.update(model.build_light_checkpoint_extra(step=int(step)))
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
            "final_step": int(max_iterations - 1),
            "train": {"loss": float(result["loss"]) if result is not None else float("nan")},
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

    final_ckpt = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_final.pt")
    final_payload = {
        "step": max_iterations - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
    }
    if hasattr(model, "build_light_checkpoint_extra"):
        final_payload.update(model.build_light_checkpoint_extra(step=int(max_iterations - 1)))
    torch.save(
        final_payload,
        final_ckpt,
    )
    logger.info("Saved final checkpoint to %s", final_ckpt)


if __name__ == "__main__":
    main()
