"""
Training script for Minimal StreetForward Stage 2.1 (multi-target with proxy).

- Same multi-target as 2.0; uses proxy params and per-view loss.backward() then
  _backward_to_render_params. Compare with 2.0 for consistency.
- Logs per-view and mean PSNR/SSIM/LPIPS; saves per-view pred/gt/error images.

Use with overfit batch (with multiple targets):
  python tools/train_minimal_streetforward_stage2_1.py --config_file configs/minimal_streetforward_stage2_1.yaml \\
    overfit_batch_path=./data/overfit_batches/scene0_seg0_batch.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.streetforward.minimal_trainer_stage2_1 import MinimalStreetForwardStage2_1
from utils.logging import setup_logging
from utils.streetforward_baseline import set_deterministic_seed

from tools.train_minimal_streetforward_stage1_1 import (
    _compute_metrics,
    _open_metrics_history,
    _save_image_triplet,
    _write_metrics_history,
    convert_batch_to_minimal_format,
    setup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

logger = logging.getLogger(__name__)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

CKPT_PREFIX = "minimal_sf_stage2_1"


def main():
    parser = argparse.ArgumentParser(
        description="Train Minimal StreetForward Stage 2.1 (multi-target with proxy)"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/minimal_streetforward_stage2_1.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--project", type=str, default="minimal_sf")
    parser.add_argument("--run_name", type=str, default="overfit")
    parser.add_argument("--overfit_batch_path", type=str, default=None, help="Path to .pt overfit batch")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training.max_iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("opts", nargs="*", help="Override config, e.g. overfit_batch_path=path/to/batch.pt")
    args = parser.parse_args()

    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("RUN start time=%s device=%s", current_time, device)

    set_deterministic_seed(args.seed)
    logger.info("Seed: %s", args.seed)

    overfit_path = getattr(args, "overfit_batch_path", None) or cfg.get("overfit_batch_path")
    if not overfit_path or not os.path.isfile(overfit_path):
        raise FileNotFoundError(
            "Overfit batch required. Set --overfit_batch_path or config overfit_batch_path."
        )
    logger.info(
        "RUN config_path=%s log_dir=%s overfit_batch_path=%s",
        args.config_file,
        cfg.log_dir,
        overfit_path,
    )
    logger.info("Loading overfit batch from %s", overfit_path)
    from tools.overfit_one_batch import load_batch

    raw_batch = load_batch(overfit_path)
    num_targets = cfg.training.get("num_targets", 3)
    minimal_batch = convert_batch_to_minimal_format(raw_batch, device, num_targets=num_targets)
    logger.info("Using num_targets=%d (batch has %d targets)", num_targets, len(minimal_batch["targets"]))

    logger.info("Building MinimalStreetForwardStage2_1...")
    model = MinimalStreetForwardStage2_1(config=cfg, device=device)
    model.train()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

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

    metrics_fh: Optional[TextIO] = None
    writer: Optional["SummaryWriter"] = None
    result: Dict[str, Any] = {}
    try:
        metrics_fh = _open_metrics_history(cfg.log_dir, enable_jsonl_metrics)

        if use_tensorboard and SummaryWriter is not None:
            tb_dir = os.path.join(cfg.log_dir, "tb")
            writer = SummaryWriter(log_dir=tb_dir)

        logger.info(
            "Training for %s steps (log every %s, save every %s, metric_interval=%s, heavy_metric_interval=%s)",
            max_iterations,
            log_interval,
            save_every,
            metric_interval,
            heavy_metric_interval,
        )

        for step in range(max_iterations):
            result = model.train_step(minimal_batch, step=step)
            loss_val = float(result["loss"])
            pred_rgbs = result["pred_rgbs"]
            gt_images = result["gt_images"]
            num_views = len(pred_rgbs)

            if step % log_interval == 0:
                logger.info("Step %s: loss=%.6f (num_views=%d)", step, loss_val, num_views)

            want_psnr = enable_psnr and (step % metric_interval == 0)
            want_heavy = heavy_metric_interval > 0 and (step % heavy_metric_interval == 0)
            if want_psnr or want_heavy:
                mse_list = []
                for v in range(num_views):
                    mse_list.append(
                        float(
                            torch.mean(
                                (torch.clamp(pred_rgbs[v], 0.0, 1.0) - torch.clamp(gt_images[v], 0.0, 1.0))
                                ** 2
                            ).item()
                        )
                    )
                mse_val = float(np.mean(mse_list))

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
                        metric_vals[f"ssim_view{v}"] = v_vals["ssim"]
                    if "lpips" in v_vals:
                        lpips_list.append(v_vals["lpips"])
                        metric_vals[f"lpips_view{v}"] = v_vals["lpips"]
                if psnr_list:
                    metric_vals["psnr_mean"] = float(np.mean(psnr_list))
                if ssim_list:
                    metric_vals["ssim_mean"] = float(np.mean(ssim_list))
                if lpips_list:
                    metric_vals["lpips_mean"] = float(np.mean(lpips_list))

                log_parts = [
                    f"METRIC step={step} split=train loss_l1={loss_val:.6f} mse={mse_val:.6e}",
                    f"psnr_mean={metric_vals.get('psnr_mean', 0):.2f}",
                ]
                for v in range(num_views):
                    if f"psnr_view{v}" in metric_vals:
                        log_parts.append(f"psnr_view{v}={metric_vals[f'psnr_view{v}']:.2f}")
                if "ssim_mean" in metric_vals and metric_vals["ssim_mean"] != 0:
                    log_parts.append(f"ssim_mean={metric_vals['ssim_mean']:.4f}")
                if "lpips_mean" in metric_vals and metric_vals["lpips_mean"] != 0:
                    log_parts.append(f"lpips_mean={metric_vals['lpips_mean']:.4f}")
                logger.info(" ".join(log_parts))

                record = {
                    "step": int(step),
                    "split": "train",
                    "loss_l1": loss_val,
                    "mse": mse_val,
                    "num_views": num_views,
                    **metric_vals,
                }
                _write_metrics_history(metrics_fh, record)

                if writer is not None:
                    writer.add_scalar("train/loss_l1", loss_val, step)
                    writer.add_scalar("train/mse", mse_val, step)
                    writer.add_scalar("train/num_views", num_views, step)
                    for k, v in metric_vals.items():
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f"train/{k}", v, step)

            if step % image_interval == 0:
                images_dir = os.path.join(cfg.log_dir, "images", "train")
                for v in range(num_views):
                    _save_image_triplet(
                        step,
                        pred_rgbs[v],
                        gt_images[v],
                        images_dir,
                        view_suffix=f"view{v}",
                    )

                if writer is not None and num_views > 0:
                    pred_clamped = torch.clamp(pred_rgbs[0].detach().cpu(), 0.0, 1.0)
                    gt_clamped = torch.clamp(gt_images[0].detach().cpu(), 0.0, 1.0)
                    error = (pred_clamped - gt_clamped).abs()
                    if error.numel() > 0:
                        max_val = float(error.max().item())
                        if max_val > 0:
                            error = error / max_val
                    writer.add_image("train/pred_view0", pred_clamped.permute(2, 0, 1), step)
                    writer.add_image("train/gt_view0", gt_clamped.permute(2, 0, 1), step)
                    writer.add_image("train/error_view0", error.permute(2, 0, 1), step)

            if save_every and step > 0 and step % save_every == 0:
                ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_step{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model.optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)

        test_metrics: Optional[Dict[str, float]] = None
        if run_test_at_end and minimal_batch.get("test_views"):
            prev_mode = model.training
            model.eval()
            with torch.no_grad():
                out = model.forward(minimal_batch)
                render_params = out["render_params"]

                psnr_list: List[float] = []
                ssim_list: List[float] = []
                lpips_list: List[float] = []

                test_views = minimal_batch.get("test_views", [])
                test_images = minimal_batch.get("test_images", [])
                for view, gt in zip(test_views, test_images):
                    h, w = int(gt.shape[0]), int(gt.shape[1])
                    pred, _ = model._render_single_view(render_params, view, h, w)
                    vals = _compute_metrics(
                        pred_rgb=pred,
                        gt_rgb=gt,
                        psnr_metric=psnr_metric,
                        ssim_metric=ssim_metric,
                        lpips_metric=lpips_metric,
                        compute_psnr=True,
                        compute_heavy=True,
                    )
                    psnr_list.append(vals["psnr"])
                    ssim_list.append(vals["ssim"])
                    lpips_list.append(vals["lpips"])

                if psnr_list:
                    test_metrics = {
                        "psnr": float(np.mean(psnr_list)),
                        "ssim": float(np.mean(ssim_list)),
                        "lpips": float(np.mean(lpips_list)),
                        "num_test_views": int(len(psnr_list)),
                    }
                    logger.info(
                        "METRIC final split=test psnr=%.2f ssim=%.4f lpips=%.4f num_test_views=%d",
                        test_metrics["psnr"],
                        test_metrics["ssim"],
                        test_metrics["lpips"],
                        test_metrics["num_test_views"],
                    )
                    if writer is not None:
                        writer.add_scalar("test/psnr", test_metrics["psnr"], max_iterations - 1)
                        writer.add_scalar("test/ssim", test_metrics["ssim"], max_iterations - 1)
                        writer.add_scalar("test/lpips", test_metrics["lpips"], max_iterations - 1)

                    metrics_final_path = os.path.join(cfg.log_dir, "metrics_final.json")
                    with open(metrics_final_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "final_step": int(max_iterations - 1),
                                "train": {"loss_l1": float(result["loss"])},
                                "test": test_metrics,
                            },
                            f,
                            indent=2,
                        )
                    logger.info("Saved metrics_final.json to %s", metrics_final_path)

            if prev_mode:
                model.train()
    finally:
        if metrics_fh is not None:
            metrics_fh.close()
        if writer is not None:
            writer.close()

    logger.info("Done. Final loss: %.6f", result.get("loss", 0.0))
    final_ckpt = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_final.pt")
    torch.save(
        {
            "step": max_iterations - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
        },
        final_ckpt,
    )
    logger.info("Saved final checkpoint to %s", final_ckpt)


if __name__ == "__main__":
    main()
