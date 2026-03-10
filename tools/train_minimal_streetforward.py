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
import logging
import os
import time
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf

from models.streetforward.minimal_trainer import MinimalStreetForward
from utils.logging import setup_logging
from utils.streetforward_baseline import set_deterministic_seed

logger = logging.getLogger(__name__)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


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

    return {
        "scene_id": scene_id,
        "segment_id": segment_id,
        "pointcloud": pointcloud_minimal,
        "targets": targets_minimal,
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
    logger.info("Device: %s", device)

    set_deterministic_seed(args.seed)
    logger.info("Seed: %s", args.seed)

    overfit_path = getattr(args, "overfit_batch_path", None) or cfg.get("overfit_batch_path")
    if not overfit_path or not os.path.isfile(overfit_path):
        raise FileNotFoundError(
            "Overfit batch required. Set --overfit_batch_path or config overfit_batch_path."
        )
    logger.info("Loading overfit batch from %s", overfit_path)
    from tools.overfit_one_batch import load_batch
    raw_batch = load_batch(overfit_path)
    minimal_batch = convert_batch_to_minimal_format(raw_batch, device)

    logger.info("Building MinimalStreetForward...")
    model = MinimalStreetForward(config=cfg, device=device)
    model.train()

    max_iterations = args.max_steps or cfg.training.get("max_iterations", 1000)
    log_interval = cfg.training.get("log_interval", 50)
    save_every = cfg.training.get("save_checkpoint_freq", 500)

    logger.info("Training for %s steps (log every %s, save every %s)", max_iterations, log_interval, save_every)

    for step in range(max_iterations):
        result = model.train_step(minimal_batch)
        loss_val = result["loss"]
        if step % log_interval == 0:
            logger.info("Step %s: loss=%.6f", step, loss_val)
        if save_every and step > 0 and step % save_every == 0:
            ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"minimal_sf_step{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
            }, ckpt_path)
            logger.info("Saved checkpoint to %s", ckpt_path)

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
