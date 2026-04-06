"""
Minimal StreetForward Stage 4.3 — 多场景训练：TrainSchedulerV4 + MultiSceneDatasetV3（image-ref batch）。

按 epoch 打乱 `data.train_scene_ids` 顺序，对每个场景打乱 segment 顺序遍历（见 `TrainSchedulerV4._init_epoch_segment_pair_iterator`）。

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
import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, TextIO

from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _normalize_omp_num_threads

_normalize_omp_num_threads()

import numpy as np
import torch
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3
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
    build_train_scheduler_v4_from_cfg,
    parse_include_test_v4,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
    if parse_view_selection(cfg.training.get("view_selection")) is not None:
        raise ValueError(
            "multi_scene v4 does not support training.view_selection.mode=explicit; "
            "remove view_selection from the config (dataset already samples keyframes per batch)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("RUN start time=%s device=%s", current_time, device)

    set_deterministic_seed(args.seed)
    logger.info("Seed: %s", args.seed)

    if cfg.get("one_segment") is not None:
        raise ValueError(
            "multi_scene v4: remove `one_segment` from config; "
            "use configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml."
        )
    train_ids = list(cfg.data.train_scene_ids)
    if len(train_ids) < 2:
        raise ValueError("multi_scene v4 requires len(data.train_scene_ids) >= 2")
    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment(cfg)
    if fixed_scene_id is not None or fixed_segment_id is not None:
        raise ValueError(
            "multi_scene v4 requires scheduler_v4.traversal.fixed_scene_id and fixed_segment_id to be null "
            "(unset one_segment and traversal overrides)."
        )

    logger.info(
        "Building MultiSceneDatasetV3; multi_scene train_scene_ids=%s include_test=%s",
        train_ids,
        parse_include_test_v4(cfg),
    )
    if cfg.get("test") is not None and bool(cfg.test.get("enable", False)):
        raise ValueError(
            "Formal testing is not supported in training script. "
            "Use tools/test_minimal_streetforward_stage4_3.py instead."
        )
    dataset = build_multi_scene_dataset_v3(cfg, device)
    dataset.initialize()
    scheduler = build_train_scheduler_v4_from_cfg(cfg, dataset)

    ov = cfg.get("scheduler_v4", {}).get("overlap") or {}
    logger.info(
        "TrainSchedulerV4 overlap: mode=%s point_sample_size=%s candidate_frame_policy=%s score_type=%s topk=%s",
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
        logger.info(
            "scheduler_v3.model_node_state.sync_with_scheduler=true: NodeState write-back when "
            "segment_local_step %% U == 0; reset_node_state() after each block_end. "
            "model.update_node_state_interval / reset_node_state_interval are ignored."
        )

    max_iterations = args.max_steps or cfg.training.get("max_iterations", 1000)
    log_interval = cfg.training.get("log_interval", 50)
    save_every = cfg.training.get("save_checkpoint_freq", 500)
    enable_psnr = bool(cfg.eval.get("enable_psnr", True))
    run_test_at_end = bool(cfg.eval.get("run_test_at_end", False))
    if run_test_at_end:
        logger.warning(
            "eval.run_test_at_end=true is deprecated in training script and will be ignored; "
            "use tools/test_minimal_streetforward_stage4_3.py for formal testing."
        )
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
    enable_jsonl_metrics = bool(cfg.logging.get("enable_jsonl_metrics", True))
    if "image_interval_blocks" not in cfg.logging:
        raise ValueError(
            "logging.image_interval_blocks is required (train images are saved every N TrainSchedulerV4 blocks, not every N steps)."
        )
    image_interval_blocks = int(cfg.logging["image_interval_blocks"])
    if image_interval_blocks < 1:
        raise ValueError(f"logging.image_interval_blocks must be >= 1, got {image_interval_blocks}")
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

    model = MinimalStreetForwardStage4_3(config=cfg, device=device)
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
    block_accum: Dict[str, Any] = {"loss_sum": 0.0, "count": 0, "start_step": 0, "event": None}

    try:
        metrics_fh = _open_metrics_history(cfg.log_dir, enable_jsonl_metrics)
        if use_tensorboard and SummaryWriter is not None:
            tb_dir = os.path.join(cfg.log_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)

        for step in range(max_iterations):
            raw_batch = scheduler.next_batch()
            scheduler_info = raw_batch.get("_scheduler_v4_aligned_info")
            if scheduler_info is None:
                scheduler_info = scheduler.get_current_info()

            step_events = scheduler.pop_events()
            scheduler_node_sync = _build_scheduler_node_sync(cfg, scheduler_info, step_events)
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
                    block_accum = {"loss_sum": 0.0, "count": 0, "start_step": int(step), "event": ev}

            tgt = raw_batch.get("target")
            if not isinstance(tgt, dict) or tgt.get("image") is None:
                raise ValueError("dataset batch must contain target.image")
            num_target_views = int(tgt["image"].shape[0])
            minimal_batch = convert_batch_to_minimal_format(
                raw_batch,
                device,
                num_targets=num_target_views,
                include_source_for_2d=True,
                view_selection=None,
            )

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
            step_t1 = time.perf_counter()
            step_time_ms = float((step_t1 - step_t0) * 1000.0)
            sum_step_time_ms += step_time_ms
            step_time_ms_hist.append(step_time_ms)
            if perf_cfg["enable"] and perf_cfg["cuda_memory"] and torch.cuda.is_available():
                peak_mem_bytes = int(max(peak_mem_bytes, int(torch.cuda.max_memory_allocated())))
                peak_mem_reserved_bytes = int(max(peak_mem_reserved_bytes, int(torch.cuda.max_memory_reserved())))

            if result is None:
                raise ValueError("train_step returned None")
            loss_val = float(result["loss"])
            pred_rgbs = result["pred_rgbs"]
            gt_images = result["gt_images"]
            num_views = len(pred_rgbs)
            total_steps += 1
            sum_num_gaussians_bg += int(result.get("num_gaussians_bg", 0))
            sum_num_gaussians_distant += int(result.get("num_gaussians_distant", 0))
            sum_num_gaussians_rigid += int(result.get("num_gaussians_rigid", 0))
            sum_num_gaussians_sky += int(result.get("num_gaussians_sky", 0))

            block_accum["loss_sum"] = float(block_accum.get("loss_sum", 0.0)) + float(loss_val)
            block_accum["count"] = int(block_accum.get("count", 0)) + 1

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

            for ev in step_events:
                if ev.get("type") != "block_end":
                    continue

                block_idx_global = int(ev.get("block_idx_global", 0))
                if block_idx_global >= 1 and (block_idx_global - 1) % image_interval_blocks == 0:
                    out_dir = os.path.join(cfg.log_dir, "images", "train")
                    tgt_meta = raw_batch.get("target") or {}
                    fi_t = tgt_meta.get("frame_indices")
                    ci_t = tgt_meta.get("cam_indices")
                    sc_lab = _scene_folder_label_from_batch(raw_batch, ev.get("scene_id"))
                    for v in range(num_views):
                        if fi_t is not None and ci_t is not None and int(fi_t.shape[0]) > v and int(ci_t.shape[0]) > v:
                            f_lab = int(fi_t[v].item())
                            c_lab = int(ci_t[v].item())
                            nusc_suf = _nuscenes_cam_id_suffix(pixel_camera_ids, c_lab)
                            vsuf = (
                                f"b{block_idx_global:06d}_sc{sc_lab}_v{v}_f{f_lab:05d}_c{c_lab}{nusc_suf}"
                            )
                        else:
                            vsuf = f"b{block_idx_global:06d}_sc{sc_lab}_view{v}"
                        _save_image_triplet(
                            step,
                            pred_rgbs[v],
                            gt_images[v],
                            out_dir,
                            view_suffix=vsuf,
                            save_error=False,
                        )

                mean_loss = float(block_accum.get("loss_sum", 0.0)) / max(int(block_accum.get("count", 0)), 1)
                mse_vals = [
                    float(
                        torch.mean((torch.clamp(pred_rgbs[v], 0.0, 1.0) - torch.clamp(gt_images[v], 0.0, 1.0)) ** 2).item()
                    )
                    for v in range(num_views)
                ]
                mse_val = float(np.mean(mse_vals))

                metric_vals: Dict[str, float] = {}
                if enable_psnr:
                    psnr_list: List[float] = []
                    ssim_list: List[float] = []
                    lpips_list: List[float] = []
                    for v in range(num_views):
                        v_vals = _compute_metrics(
                            pred_rgb=pred_rgbs[v],
                            gt_rgb=gt_images[v],
                            psnr_metric=psnr_metric,
                            ssim_metric=ssim_metric,
                            lpips_metric=lpips_metric,
                            compute_psnr=True,
                            compute_heavy=True,
                        )
                        psnr_list.append(v_vals["psnr"])
                        ssim_list.append(v_vals["ssim"])
                        lpips_list.append(v_vals["lpips"])
                        metric_vals[f"psnr_view{v}"] = float(v_vals["psnr"])
                    metric_vals["psnr_mean"] = float(np.mean(psnr_list)) if psnr_list else 0.0
                    metric_vals["ssim_mean"] = float(np.mean(ssim_list)) if ssim_list else 0.0
                    metric_vals["lpips_mean"] = float(np.mean(lpips_list)) if lpips_list else 0.0

                    if save_train_views_psnr_below is not None and low_psnr_train_images_subdir is not None:
                        out_low = os.path.join(cfg.log_dir, "images", low_psnr_train_images_subdir)
                        tgt_meta = raw_batch.get("target") or {}
                        fi_t = tgt_meta.get("frame_indices")
                        ci_t = tgt_meta.get("cam_indices")
                        block_idx_global = int(ev.get("block_idx_global", 0))
                        sdir = _scene_folder_label_from_batch(raw_batch, ev.get("scene_id"))
                        thr = float(save_train_views_psnr_below)
                        n_psnr = min(num_views, len(psnr_list))
                        if any(float(psnr_list[v]) < thr for v in range(n_psnr)):
                            for v in range(num_views):
                                if v >= len(psnr_list):
                                    break
                                if fi_t is not None and ci_t is not None and int(fi_t.shape[0]) > v and int(ci_t.shape[0]) > v:
                                    f_lab = int(fi_t[v].item())
                                    c_lab = int(ci_t[v].item())
                                    nusc_suf = _nuscenes_cam_id_suffix(pixel_camera_ids, c_lab)
                                    vsuf = (
                                        f"b{block_idx_global:06d}_sc{sdir}_v{v}_f{f_lab:05d}_c{c_lab}{nusc_suf}"
                                        f"_psnr{float(psnr_list[v]):.2f}"
                                    )
                                else:
                                    vsuf = (
                                        f"b{block_idx_global:06d}_sc{sdir}_v{v}_psnr{float(psnr_list[v]):.2f}"
                                    )
                                _save_image_triplet(
                                    step,
                                    pred_rgbs[v],
                                    gt_images[v],
                                    out_low,
                                    view_suffix=vsuf,
                                    save_error=False,
                                )

                psnr_log = (
                    f"{float(metric_vals['psnr_mean']):.2f}"
                    if enable_psnr and "psnr_mean" in metric_vals
                    else "n/a"
                )
                logger.info(
                    "BLOCK_END global_step=%s scene_id=%s scene_dir=%s segment=%s block_seg=%s block_global=%s mean_loss=%.6f mse=%.6e psnr_mean=%s onepass=%d",
                    ev.get("global_step"),
                    ev.get("scene_id"),
                    _scene_dir_str(ev.get("scene_id", -1)),
                    ev.get("segment_id"),
                    ev.get("block_idx_in_segment"),
                    ev.get("block_idx_global"),
                    mean_loss,
                    mse_val,
                    psnr_log,
                    int(result.get("src_backproject_pass_count", 0)),
                )

                row = {
                    "step": int(step),
                    "split": "train",
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
                    "mean_loss_in_block": float(mean_loss),
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
                    "node_state_sync_update": bool(result.get("node_state_sync_update", False)),
                    "node_state_sync_reset": bool(result.get("node_state_sync_reset", False)),
                }
                for k, v in result.items():
                    if (
                        k.startswith("bg_offset_")
                        or k.startswith("rigid_offset_")
                        or k.startswith("perf_")
                    ):
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
                    writer.add_scalar("train/mean_loss_in_block", float(mean_loss), step)
                    writer.add_scalar("train/mse", float(mse_val), step)
                    writer.add_scalar("train/num_bg_update", int(result.get("num_bg_update", 0)), step)
                    writer.add_scalar("train/num_distant_update", int(result.get("num_distant_update", 0)), step)
                    writer.add_scalar("train/num_gaussians_sky", int(result.get("num_gaussians_sky", 0)), step)
                    writer.add_scalar("train/src_backproject_pass_count", int(result.get("src_backproject_pass_count", 0)), step)
                    for k, v in metric_vals.items():
                        writer.add_scalar(f"train/{k}", float(v), step)
                    writer.add_scalar("train/perf/step_time_ms", float(step_time_ms), step)

            if save_every and step > 0 and step % save_every == 0:
                ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_step{step}.pt")
                torch.save(
                    {"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": model.optimizer.state_dict()},
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)

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
    torch.save(
        {"step": max_iterations - 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": model.optimizer.state_dict()},
        final_ckpt,
    )
    logger.info("Saved final checkpoint to %s", final_ckpt)


if __name__ == "__main__":
    main()

