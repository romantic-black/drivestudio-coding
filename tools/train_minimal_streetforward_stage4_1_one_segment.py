"""
Minimal StreetForward Stage 4.1 — 单段训练：Scheduler v2 + MultiSceneDatasetV2。

不使用 overfit .pt；配置见 configs/minimal_streetforward_stage4_1_one_segment_v2.yaml。

  conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding \\
    python tools/train_minimal_streetforward_stage4_1_one_segment.py \\
      --config_file configs/minimal_streetforward_stage4_1_one_segment_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, TextIO

# Match stage4_1 OMP normalization
def _normalize_omp_num_threads(*, fallback: int = 8) -> None:
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return
    raw_str = str(raw).strip()
    try:
        value = int(raw_str)
    except (TypeError, ValueError):
        os.environ["OMP_NUM_THREADS"] = str(fallback)
        logging.getLogger(__name__).warning("Invalid OMP_NUM_THREADS=%r; fallback to %d.", raw_str, fallback)
        return
    if value <= 0:
        os.environ["OMP_NUM_THREADS"] = str(fallback)
        logging.getLogger(__name__).warning("Non-positive OMP_NUM_THREADS=%d; fallback to %d.", value, fallback)


_normalize_omp_num_threads()

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.multi_scene_dataset_v2 import MultiSceneDatasetV2, TrainSchedulerV2
from models.streetforward.minimal_trainer_stage4_1 import MinimalStreetForwardStage4_1
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
    _merge_bg_distant_rigid_for_eval,
    _parse_diagnostics_cfg,
    _parse_perf_cfg,
    _percentile,
    _save_diagnostic_renders,
    _save_single_gray_image,
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
CKPT_PREFIX = "minimal_sf_stage4_1_one_segment"


def _build_multi_scene_dataset(cfg: Any, device: torch.device) -> MultiSceneDatasetV2:
    ds_cfg = cfg.dataset
    data_cfg = cfg.data
    pc = OmegaConf.to_container(ds_cfg.pointcloud, resolve=True)
    kfc = ds_cfg.get("keyframe_split_config")
    if kfc is not None:
        kfc = OmegaConf.to_container(kfc, resolve=True)
    return MultiSceneDatasetV2(
        data_cfg=data_cfg,
        train_scene_ids=list(data_cfg.train_scene_ids),
        eval_scene_ids=list(data_cfg.get("eval_scene_ids", [])),
        num_source_keyframes=int(ds_cfg.num_source_keyframes),
        num_target_keyframes=int(ds_cfg.num_target_keyframes),
        segment_overlap_ratio=float(ds_cfg.segment_overlap_ratio),
        keyframe_split_config=kfc,
        min_keyframes_per_scene=int(ds_cfg.min_keyframes_per_scene),
        min_keyframes_per_segment=int(ds_cfg.min_keyframes_per_segment),
        device=device,
        preload_scene_count=int(ds_cfg.preload_scene_count),
        segment_aabb=ds_cfg.segment_aabb,
        pointcloud_config=pc,
    )


def _parse_one_segment_cfg(cfg: Any) -> tuple[int, int, bool]:
    os_cfg = cfg.get("one_segment")
    if os_cfg is None:
        raise ValueError("config must define one_segment with scene_id and segment_id")
    if os_cfg.get("scene_id") is None or os_cfg.get("segment_id") is None:
        raise ValueError("one_segment.scene_id and one_segment.segment_id are required")
    return int(os_cfg.scene_id), int(os_cfg.segment_id), bool(os_cfg.get("include_test", True))


def _build_train_scheduler_v2(
    cfg: Any,
    dataset: MultiSceneDatasetV2,
    scene_id: int,
    segment_id: int,
    include_test: bool,
) -> TrainSchedulerV2:
    scheduler_cfg = cfg.get("scheduler_v2")
    if scheduler_cfg is None:
        raise ValueError("config must define scheduler_v2 for one-segment v2 training")
    if scheduler_cfg.get("target_include_source") is not True:
        raise ValueError("scheduler_v2.target_include_source must be true")
    return dataset.create_train_scheduler_v2(
        alpha_steps_per_keyframe=float(scheduler_cfg.alpha_steps_per_keyframe),
        min_steps_per_segment=int(scheduler_cfg.min_steps_per_segment),
        max_steps_per_segment=int(scheduler_cfg.max_steps_per_segment),
        source_hold_steps=int(scheduler_cfg.source_hold_steps),
        num_target_frames_total=int(scheduler_cfg.num_target_frames_total),
        target_include_source=bool(scheduler_cfg.target_include_source),
        include_test=bool(include_test),
        fixed_scene_id=int(scene_id),
        fixed_segment_id=int(segment_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Minimal StreetForward Stage 4.1 (one segment, dataloader)")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/minimal_streetforward_stage4_1_one_segment_v2.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--project", type=str, default="minimal_sf")
    parser.add_argument("--run_name", type=str, default="one_segment")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training.max_iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("opts", nargs="*", help="Override config")
    args = parser.parse_args()

    cfg = setup(args)
    if parse_view_selection(cfg.training.get("view_selection")) is not None:
        raise ValueError(
            "one_segment script does not support training.view_selection.mode=explicit; "
            "remove view_selection from the config (dataset already samples keyframes per batch)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("RUN start time=%s device=%s", current_time, device)

    set_deterministic_seed(args.seed)
    logger.info("Seed: %s", args.seed)

    scene_id, segment_id, include_test = _parse_one_segment_cfg(cfg)
    train_ids = list(cfg.data.train_scene_ids)
    if scene_id not in train_ids:
        raise ValueError(f"one_segment.scene_id={scene_id} must appear in data.train_scene_ids={train_ids}")

    logger.info(
        "Building MultiSceneDatasetV2; training one segment scene_id=%s segment_id=%s include_test=%s",
        scene_id,
        segment_id,
        include_test,
    )
    dataset = _build_multi_scene_dataset(cfg, device)
    dataset.initialize()
    scheduler = _build_train_scheduler_v2(cfg, dataset, scene_id, segment_id, include_test)

    max_iterations = args.max_steps or cfg.training.get("max_iterations", 1000)
    log_interval = cfg.training.get("log_interval", 50)
    save_every = cfg.training.get("save_checkpoint_freq", 500)
    metric_interval = int(cfg.eval.get("metric_interval", 10))
    heavy_metric_interval = int(cfg.eval.get("heavy_metric_interval", 50))
    enable_psnr = bool(cfg.eval.get("enable_psnr", True))
    run_test_at_end = bool(cfg.eval.get("run_test_at_end", True))
    enable_jsonl_metrics = bool(cfg.logging.get("enable_jsonl_metrics", True))
    image_interval = int(cfg.logging.get("image_interval", 50))
    use_tensorboard = bool(cfg.logging.get("use_tensorboard", False))
    diag_cfg = _parse_diagnostics_cfg(cfg)
    perf_cfg = _parse_perf_cfg(cfg)

    model = MinimalStreetForwardStage4_1(config=cfg, device=device)
    model.train()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    metrics_fh: Optional[TextIO] = None
    writer: Optional[Any] = None
    result: Dict[str, Any] = {}
    total_steps = 0
    sum_num_gaussians_bg = 0.0
    sum_num_gaussians_distant = 0.0
    sum_num_gaussians_rigid = 0.0
    sum_step_time_ms = 0.0
    step_time_ms_hist: List[float] = []
    peak_mem_bytes = 0
    peak_mem_reserved_bytes = 0
    diag_window: deque = deque(maxlen=max(diag_cfg.get("window_size", 0), 1))
    minimal_batch: Dict[str, Any] = {}

    try:
        metrics_fh = _open_metrics_history(cfg.log_dir, enable_jsonl_metrics)
        if use_tensorboard and SummaryWriter is not None:
            writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "tb"))

        for step in range(max_iterations):
            raw_batch = scheduler.next_batch()
            scheduler_info = scheduler.get_current_info()
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
            )
            step_t1 = time.perf_counter()
            step_time_ms = float((step_t1 - step_t0) * 1000.0)
            sum_step_time_ms += step_time_ms
            step_time_ms_hist.append(step_time_ms)
            if perf_cfg["enable"] and perf_cfg["cuda_memory"] and torch.cuda.is_available():
                peak_mem_bytes = int(max(peak_mem_bytes, int(torch.cuda.max_memory_allocated())))
                peak_mem_reserved_bytes = int(max(peak_mem_reserved_bytes, int(torch.cuda.max_memory_reserved())))
            loss_val = float(result["loss"])
            pred_rgbs = result["pred_rgbs"]
            gt_images = result["gt_images"]
            num_views = len(pred_rgbs)
            total_steps += 1
            sum_num_gaussians_bg += int(result.get("num_gaussians_bg", 0))
            sum_num_gaussians_distant += int(result.get("num_gaussians_distant", 0))
            sum_num_gaussians_rigid += int(result.get("num_gaussians_rigid", 0))

            if step % log_interval == 0:
                logger.info(
                    "Step %s: loss=%.6f (num_views=%d target_frames=%d rigid_update=%d feat_valid=%d)",
                    step,
                    loss_val,
                    num_views,
                    int(result.get("num_target_frames", 0)),
                    int(result.get("num_rigid_update", 0)),
                    int(result.get("num_rigid_src_feat_valid", 0)),
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

            want_psnr = enable_psnr and (step % metric_interval == 0)
            want_heavy = heavy_metric_interval > 0 and (step % heavy_metric_interval == 0)
            row = {
                "step": int(step),
                "split": "train",
                "scene_id": int(minimal_batch.get("scene_id", -1)),
                "segment_id": int(minimal_batch.get("segment_id", -1)),
                "epoch_idx": int(scheduler_info.get("epoch_idx", -1)),
                "global_step": int(scheduler_info.get("global_step", -1)),
                "segment_local_step": int(scheduler_info.get("segment_local_step", -1)),
                "segment_step_budget": int(scheduler_info.get("segment_step_budget", -1)),
                "source_frame_idx": int(scheduler_info.get("source_frame_idx", -1)),
                "source_block_step": int(scheduler_info.get("source_block_step", -1)),
                "loss": loss_val,
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
                "hidden_norm_bg_mean": float(result.get("hidden_norm_bg_mean", 0.0)),
                "hidden_norm_distant_mean": float(result.get("hidden_norm_distant_mean", 0.0)),
                "hidden_norm_rigid_mean": float(result.get("hidden_norm_rigid_mean", 0.0)),
                "grad_norm_bg": float(result.get("grad_norm_bg", 0.0)),
                "grad_norm_distant": float(result.get("grad_norm_distant", 0.0)),
                "grad_norm_rigid": float(result.get("grad_norm_rigid", 0.0)),
                "step_time_ms": step_time_ms,
                "forward_ms": float(result.get("forward_ms", 0.0)),
                "backward_ms": float(result.get("backward_ms", 0.0)),
                "optimizer_ms": float(result.get("optimizer_ms", 0.0)),
            }
            for k, v in result.items():
                if k.startswith("bg_offset_") or k.startswith("rigid_offset_"):
                    row[k] = float(v)
            row["loss_mask_ratio"] = float(row["loss_mask"] / max(loss_val, 1e-8))
            frame_loss_map = result.get("frame_loss_map", {})
            for fid, fl in frame_loss_map.items():
                row[f"loss_frame_{fid}"] = float(fl)
            if perf_cfg["enable"] and perf_cfg["cuda_memory"] and torch.cuda.is_available():
                row["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
                row["peak_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved())

            if want_psnr or want_heavy:
                mse_vals = [
                    float(
                        torch.mean(
                            (torch.clamp(pred_rgbs[v], 0.0, 1.0) - torch.clamp(gt_images[v], 0.0, 1.0)) ** 2
                        ).item()
                    )
                    for v in range(num_views)
                ]
                mse_val = float(np.mean(mse_vals))
                metric_vals: Dict[str, float] = {"psnr_mean": 0.0, "ssim_mean": 0.0, "lpips_mean": 0.0}
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
                        compute_psnr=want_psnr,
                        compute_heavy=want_heavy,
                    )
                    if "psnr" in v_vals:
                        psnr_list.append(v_vals["psnr"])
                        metric_vals[f"psnr_view{v}"] = v_vals["psnr"]
                    if "ssim" in v_vals:
                        ssim_list.append(v_vals["ssim"])
                    if "lpips" in v_vals:
                        lpips_list.append(v_vals["lpips"])
                if psnr_list:
                    metric_vals["psnr_mean"] = float(np.mean(psnr_list))
                if ssim_list:
                    metric_vals["ssim_mean"] = float(np.mean(ssim_list))
                if lpips_list:
                    metric_vals["lpips_mean"] = float(np.mean(lpips_list))
                log_parts = [
                    f"METRIC step={step} split=train loss={loss_val:.6f} mse={mse_val:.6e}",
                    f"psnr_mean={metric_vals.get('psnr_mean', 0):.2f}",
                ]
                for v in range(num_views):
                    if f"psnr_view{v}" in metric_vals:
                        log_parts.append(f"psnr_view{v}={metric_vals[f'psnr_view{v}']:.2f}")
                if "ssim_mean" in metric_vals and metric_vals["ssim_mean"] != 0:
                    log_parts.append(f"ssim_mean={metric_vals['ssim_mean']:.4f}")
                if "lpips_mean" in metric_vals and metric_vals["lpips_mean"] != 0:
                    log_parts.append(f"lpips_mean={metric_vals['lpips_mean']:.4f}")
                log_parts.append(
                    "rigid_valid="
                    f"{int(result.get('num_rigid_valid_src', 0))}/"
                    f"{int(result.get('num_gaussians_rigid', 0))}"
                )
                log_parts.append(f"rigid_feat_valid={int(result.get('num_rigid_src_feat_valid', 0))}")
                log_parts.append(f"rigid_update={int(result.get('num_rigid_update', 0))}")
                log_parts.append(f"loss_frames={int(result.get('loss_effective_frames', 0))}")
                logger.info(" ".join(log_parts))
                row["mse"] = mse_val
                row.update(metric_vals)
                if writer is not None:
                    writer.add_scalar("train/loss", loss_val, step)
                    writer.add_scalar("train/mse", mse_val, step)
                    for k, v in metric_vals.items():
                        writer.add_scalar(f"train/{k}", v, step)
                    writer.add_scalar("train/num_rigid_update", int(result.get("num_rigid_update", 0)), step)
                    writer.add_scalar("train/num_rigid_src_feat_valid", int(result.get("num_rigid_src_feat_valid", 0)), step)
            if diag_cfg["enable"]:
                diag_window.append(row)
                if step % max(diag_cfg["interval"], 1) == 0:
                    diag = _diagnose_step(list(diag_window))
                else:
                    diag = {"diag_tags": [], "diag_scores": {}, "diag_reason": "interval_not_reached"}
                row.update(diag)
            _write_metrics_history(metrics_fh, row)
            if writer is not None:
                writer.add_scalar("train/perf/step_time_ms", step_time_ms, step)
                writer.add_scalar("train/perf/forward_ms", float(result.get("forward_ms", 0.0)), step)
                writer.add_scalar("train/perf/backward_ms", float(result.get("backward_ms", 0.0)), step)
                writer.add_scalar("train/perf/optimizer_ms", float(result.get("optimizer_ms", 0.0)), step)
                writer.add_scalar("train/model/grad_norm_bg", float(result.get("grad_norm_bg", 0.0)), step)
                writer.add_scalar("train/model/grad_norm_distant", float(result.get("grad_norm_distant", 0.0)), step)
                writer.add_scalar("train/model/grad_norm_rigid", float(result.get("grad_norm_rigid", 0.0)), step)
                writer.add_scalar("train/model/hidden_norm_bg_mean", float(result.get("hidden_norm_bg_mean", 0.0)), step)
                writer.add_scalar("train/model/hidden_norm_distant_mean", float(result.get("hidden_norm_distant_mean", 0.0)), step)
                writer.add_scalar("train/model/hidden_norm_rigid_mean", float(result.get("hidden_norm_rigid_mean", 0.0)), step)
                if diag_cfg["enable"]:
                    for diag_name, diag_score in row.get("diag_scores", {}).items():
                        writer.add_scalar(f"train/diag/{diag_name}_score", float(diag_score), step)

            if step % image_interval == 0:
                out_dir = os.path.join(cfg.log_dir, "images", "train")
                for v in range(num_views):
                    _save_image_triplet(step, pred_rgbs[v], gt_images[v], out_dir, view_suffix=f"view{v}")
            if diag_cfg["enable"] and diag_cfg["save_branch_renders"] and step % max(diag_cfg["interval"], 1) == 0:
                _save_diagnostic_renders(model, minimal_batch, step, cfg.log_dir)

            if save_every and step > 0 and step % save_every == 0:
                ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_step{step}.pt")
                torch.save(
                    {"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": model.optimizer.state_dict()},
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)

        if run_test_at_end and minimal_batch.get("test_views"):
            prev_mode = model.training
            model.eval()
            with torch.no_grad():
                out = model.forward(minimal_batch)
                test_frame_indices = minimal_batch.get("test_frame_indices")
                default_fi = int(minimal_batch["targets"][0]["frame_idx"]) if minimal_batch.get("targets") else 0
                psnr_list = []
                ssim_list = []
                lpips_list = []
                for i, (view, gt) in enumerate(
                    zip(minimal_batch.get("test_views", []), minimal_batch.get("test_images", []))
                ):
                    fi = default_fi
                    if test_frame_indices is not None and i < len(test_frame_indices):
                        fi = int(test_frame_indices[i])
                    merged = _merge_bg_distant_rigid_for_eval(model, out, fi)
                    h, w = int(gt.shape[0]), int(gt.shape[1])
                    pred, _ = model._render_single_view(merged, view, h, w)
                    vals = _compute_metrics(pred, gt, psnr_metric, ssim_metric, lpips_metric, True, True)
                    psnr_list.append(vals["psnr"])
                    ssim_list.append(vals["ssim"])
                    lpips_list.append(vals["lpips"])
                if psnr_list:
                    summary = {
                        "final_step": int(max_iterations - 1),
                        "train": {"loss": float(result["loss"])},
                        "test": {
                            "psnr": float(np.mean(psnr_list)),
                            "ssim": float(np.mean(ssim_list)),
                            "lpips": float(np.mean(lpips_list)),
                            "num_test_views": int(len(psnr_list)),
                        },
                        "gs_stats": {
                            "avg_num_gaussians_bg": sum_num_gaussians_bg / max(total_steps, 1),
                            "avg_num_gaussians_rigid": sum_num_gaussians_rigid / max(total_steps, 1),
                            "avg_num_gaussians_distant": sum_num_gaussians_distant / max(total_steps, 1),
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
            if prev_mode:
                model.train()
    finally:
        if metrics_fh is not None:
            metrics_fh.close()
        if writer is not None:
            writer.close()

    final_ckpt = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_final.pt")
    torch.save(
        {"step": max_iterations - 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": model.optimizer.state_dict()},
        final_ckpt,
    )
    logger.info("Saved final checkpoint to %s", final_ckpt)


if __name__ == "__main__":
    main()
