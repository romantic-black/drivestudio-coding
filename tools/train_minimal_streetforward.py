"""
Training script for Minimal StreetForward (Stage 0).

- Input: 3D RGB point cloud only + single target (view + gt_image); no source.
- Pipeline: point cloud -> 3D features -> 3DGS head -> single-view render -> L1 loss.

Use with overfit batch:
  python tools/train_minimal_streetforward.py --config_file configs/minimal_streetforward.yaml \\
    overfit_batch_path=./data/overfit_batches/scene0_seg0_batch.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.streetforward.minimal_trainer import MinimalStreetForward
from utils.logging import setup_logging
from utils.streetforward_baseline import set_deterministic_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

logger = logging.getLogger(__name__)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _open_metrics_history(
    log_dir: str,
    enable_jsonl: bool,
) -> Optional[TextIO]:
    if not enable_jsonl:
        return None
    metrics_path = os.path.join(log_dir, "metrics_history.jsonl")
    return open(metrics_path, "a", encoding="utf-8")


def _write_metrics_history(
    fh: Optional[TextIO],
    record: Dict,
) -> None:
    if fh is None:
        return
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def _save_image_triplet(
    step: int,
    pred_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    out_dir: str,
) -> None:
    """Save pred / gt / error images to out_dir as PNG."""
    os.makedirs(out_dir, exist_ok=True)
    pred = torch.clamp(pred_rgb.detach().cpu(), 0.0, 1.0)
    gt = torch.clamp(gt_image.detach().cpu(), 0.0, 1.0)
    error = (pred - gt).abs()
    if error.numel() > 0:
        max_val = float(error.max().item())
        if max_val > 0:
            error = error / max_val
    for name, img in [
        ("pred", pred),
        ("gt", gt),
        ("error", error),
    ]:
        img_np = (img.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        filename = os.path.join(out_dir, f"step{step:06d}_{name}.png")
        try:
            from PIL import Image
        except ImportError:
            # Fallback: save via numpy if PIL is unavailable
            np.save(filename.replace(".png", ".npy"), img_np)
            continue
        Image.fromarray(img_np).save(filename)


def _hwc01_to_nchw01(img: torch.Tensor) -> torch.Tensor:
    """[H,W,3] -> [1,3,H,W] in [0,1]."""
    if img.dim() != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape={tuple(img.shape)}")
    return img.permute(2, 0, 1).unsqueeze(0)


def _compute_metrics(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: SSIM,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    compute_psnr: bool,
    compute_heavy: bool,
) -> Dict[str, float]:
    """
    Compute metrics using modules aligned with models/trainers/base.py.
    pred_rgb / gt_rgb: [H,W,3] on any device, assumed in [0,1] (will be clamped).
    """
    pred = torch.clamp(pred_rgb, 0.0, 1.0)
    gt = torch.clamp(gt_rgb, 0.0, 1.0)

    out: Dict[str, float] = {}

    pred_nchw = _hwc01_to_nchw01(pred)
    gt_nchw = _hwc01_to_nchw01(gt)

    if compute_psnr:
        psnr_metric.reset()
        out["psnr"] = float(psnr_metric(pred_nchw, gt_nchw).item())

    if compute_heavy:
        out["ssim"] = float(ssim_metric(pred_nchw, gt_nchw).item())

        lpips_metric.reset()
        out["lpips"] = float(lpips_metric(pred_nchw, gt_nchw).item())

    return out


def convert_batch_to_minimal_format(batch: Dict, device: torch.device) -> Dict:
    """
    Convert a raw overfit/dataset batch to MinimalStreetForward format.

    - pointcloud: only static part (background), as dict with "background" [N, 6].
    - targets: exactly one element { "frame_idx", "view", "gt_image" }.
    - No source views or images.
    """
    scene_id = batch.get("scene_id")
    segment_id = batch.get("segment_id")
    if torch.is_tensor(scene_id):
        scene_id = scene_id.item()
    if torch.is_tensor(segment_id):
        segment_id = segment_id.item() if segment_id.numel() == 1 else int(segment_id[0].item())

    pointcloud = batch.get("pointcloud")
    if pointcloud is None:
        raise ValueError("batch must contain 'pointcloud'")
    if isinstance(pointcloud, dict):
        background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
        pointcloud_minimal = {"background": background}
    else:
        pointcloud_minimal = pointcloud

    target_data = batch.get("target", batch.get("targets"))
    if target_data is None:
        raise ValueError("batch must contain 'target' or 'targets'")

    if isinstance(target_data, dict):
        num_target = target_data["image"].shape[0]
        target_views = []
        gt_images = []
        for i in range(num_target):
            view = type("View", (), {
                "camtoworlds": target_data["extrinsics"][i].to(device),
                "Ks": target_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
            })()
            target_views.append(view)
            gt_images.append(target_data["image"][i].to(device))
        frame_indices = target_data.get("frame_indices")
        targets = [
            {
                "frame_idx": int(frame_indices[i]) if frame_indices is not None else 0,
                "view": target_views[i],
                "gt_image": gt_images[i],
            }
            for i in range(num_target)
        ]
    else:
        targets = target_data

    if not targets:
        raise ValueError("At least one target required for minimal trainer")
    targets_minimal = [targets[0]]

    # Optional test split: build test_views/test_images if batch contains "test"
    test_views: List[Any] = []
    test_images: List[torch.Tensor] = []
    test_data = batch.get("test")
    if isinstance(test_data, dict) and "image" in test_data and test_data["image"].numel() > 0:
        num_test = int(test_data["image"].shape[0])
        for i in range(num_test):
            view = type(
                "View",
                (),
                {
                    "camtoworlds": test_data["extrinsics"][i].to(device),
                    "Ks": test_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
                },
            )()
            test_views.append(view)
            test_images.append(test_data["image"][i].to(device))

    return {
        "scene_id": scene_id,
        "segment_id": segment_id,
        "pointcloud": pointcloud_minimal,
        "targets": targets_minimal,
        "test_views": test_views,
        "test_images": test_images,
    }


def setup(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config_file)
    if getattr(args, "opts", None):
        cli = OmegaConf.from_cli(args.opts)
        cfg = OmegaConf.merge(cfg, cli)

    if "data" not in cfg:
        cfg.data = {}
    if "model" not in cfg:
        raise ValueError("config must contain 'model'")
    if "optimizer" not in cfg:
        cfg.optimizer = {"lr": 1e-3, "eps": 1e-15, "weight_decay": 0.0}

    # Eval / logging defaults for Minimal Stage 0
    if "eval" not in cfg:
        cfg.eval = {}
    if "enable_psnr" not in cfg.eval:
        cfg.eval.enable_psnr = True
    if "metric_interval" not in cfg.eval:
        cfg.eval.metric_interval = 10
    if "heavy_metric_interval" not in cfg.eval:
        cfg.eval.heavy_metric_interval = 50
    if "run_test_at_end" not in cfg.eval:
        cfg.eval.run_test_at_end = True

    if "logging" not in cfg:
        cfg.logging = {}
    if "image_interval" not in cfg.logging:
        cfg.logging.image_interval = 50
    if "enable_jsonl_metrics" not in cfg.logging:
        cfg.logging.enable_jsonl_metrics = True
    if "use_tensorboard" not in cfg.logging:
        cfg.logging.use_tensorboard = False

    log_dir = os.path.join(
        getattr(args, "output_root", "outputs"),
        getattr(args, "project", "minimal_sf"),
        getattr(args, "run_name", "overfit"),
    )
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for sub in ("images", "checkpoints"):
        os.makedirs(os.path.join(log_dir, sub), exist_ok=True)

    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train Minimal StreetForward (overfit one batch)")
    parser.add_argument("--config_file", type=str, required=True, help="Path to config YAML")
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
    minimal_batch = convert_batch_to_minimal_format(raw_batch, device)

    logger.info("Building MinimalStreetForward...")
    model = MinimalStreetForward(config=cfg, device=device)
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
            result = model.train_step(minimal_batch)
            loss_val = float(result["loss"])
            pred_rgb = result["pred_rgb"]
            gt_image = result["gt_image"]

            if step % log_interval == 0:
                logger.info("Step %s: loss=%.6f", step, loss_val)

            want_psnr = enable_psnr and (step % metric_interval == 0)
            want_heavy = heavy_metric_interval > 0 and (step % heavy_metric_interval == 0)
            if want_psnr or want_heavy:
                mse_val = float(
                    torch.mean(
                        (torch.clamp(pred_rgb, 0.0, 1.0) - torch.clamp(gt_image, 0.0, 1.0))
                        ** 2
                    ).item()
                )
                metric_vals = _compute_metrics(
                    pred_rgb=pred_rgb,
                    gt_rgb=gt_image,
                    psnr_metric=psnr_metric,
                    ssim_metric=ssim_metric,
                    lpips_metric=lpips_metric,
                    compute_psnr=want_psnr,
                    compute_heavy=want_heavy,
                )

                log_parts = [
                    f"METRIC step={step} split=train loss_l1={loss_val:.6f} mse={mse_val:.6e}"
                ]
                if "psnr" in metric_vals:
                    log_parts.append(f"psnr={metric_vals['psnr']:.2f}")
                if "ssim" in metric_vals:
                    log_parts.append(f"ssim={metric_vals['ssim']:.4f}")
                if "lpips" in metric_vals:
                    log_parts.append(f"lpips={metric_vals['lpips']:.4f}")
                logger.info(" ".join(log_parts))

                record = {
                    "step": int(step),
                    "split": "train",
                    "loss_l1": loss_val,
                    "mse": mse_val,
                    **metric_vals,
                }
                _write_metrics_history(metrics_fh, record)

                if writer is not None:
                    writer.add_scalar("train/loss_l1", loss_val, step)
                    writer.add_scalar("train/mse", mse_val, step)
                    if "psnr" in metric_vals:
                        writer.add_scalar("train/psnr", metric_vals["psnr"], step)
                    if "ssim" in metric_vals:
                        writer.add_scalar("train/ssim", metric_vals["ssim"], step)
                    if "lpips" in metric_vals:
                        writer.add_scalar("train/lpips", metric_vals["lpips"], step)

            if step % image_interval == 0:
                images_dir = os.path.join(cfg.log_dir, "images", "train")
                _save_image_triplet(step, pred_rgb, gt_image, images_dir)

                if writer is not None:
                    pred_clamped = torch.clamp(pred_rgb.detach().cpu(), 0.0, 1.0)
                    gt_clamped = torch.clamp(gt_image.detach().cpu(), 0.0, 1.0)
                    error = (pred_clamped - gt_clamped).abs()
                    if error.numel() > 0:
                        max_val = float(error.max().item())
                        if max_val > 0:
                            error = error / max_val
                    writer.add_image(
                        "train/pred", pred_clamped.permute(2, 0, 1), step
                    )
                    writer.add_image(
                        "train/gt", gt_clamped.permute(2, 0, 1), step
                    )
                    writer.add_image(
                        "train/error", error.permute(2, 0, 1), step
                    )

            if save_every and step > 0 and step % save_every == 0:
                ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"minimal_sf_step{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model.optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)

        # Phase 4: evaluate test views if present
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

    logger.info("Done. Final loss: %.6f", result["loss"])
    final_ckpt = os.path.join(cfg.log_dir, "checkpoints", "minimal_sf_final.pt")
    torch.save({
        "step": max_iterations - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
    }, final_ckpt)
    logger.info("Saved final checkpoint to %s", final_ckpt)


if __name__ == "__main__":
    main()
