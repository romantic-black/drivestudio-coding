from __future__ import annotations

import argparse
import logging
import os

import torch
from omegaconf import OmegaConf

from models.streetforward.sky_branch import MinimalSkyBranchTrainer
from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v8_from_cfg,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone SkyBranch on frozen StreetForward scene renders")
    parser.add_argument("--config_file", default="configs/minimal_skybranch_v0.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config_file)
    if args.opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_multi_scene_dataset_v4(cfg, device)
    scheduler = build_train_scheduler_v8_from_cfg(cfg, dataset)
    trainer = MinimalSkyBranchTrainer(cfg, device=device)
    max_steps = int(args.max_steps or cfg.training.get("max_iterations", 60000))
    log_interval = int(cfg.training.get("log_interval", 100))
    save_every = int(cfg.training.get("save_checkpoint_freq", 1000))
    ckpt_dir = os.path.join(str(cfg.get("log_dir", "outputs/skybranch")), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for step in range(1, max_steps + 1):
        raw_batch = scheduler.next_batch()
        minimal = convert_batch_to_minimal_format(
            raw_batch,
            device=device,
            num_targets=int(raw_batch["target"]["image"].shape[0]),
            include_source_for_2d=True,
            view_selection=cfg.training.get("view_selection"),
        )
        sync = raw_batch.get("_scheduler_v8_aligned_info")
        logs = trainer.train_step(minimal, step=step, scheduler_node_sync=sync)
        if step % log_interval == 0:
            logger.info("step=%d loss=%.6f composite_psnr=%.3f sky_psnr=%.3f", step, logs["loss"], logs.get("composite_psnr", 0.0), logs.get("sky_psnr", 0.0))
        if save_every > 0 and step % save_every == 0:
            trainer.save_checkpoint(os.path.join(ckpt_dir, f"skybranch_resume_step_{step:06d}.pth"), kind="resume")
            trainer.save_checkpoint(os.path.join(ckpt_dir, f"skybranch_model_step_{step:06d}.pth"), kind="model")
    trainer.save_checkpoint(os.path.join(ckpt_dir, "skybranch_resume_final.pth"), kind="resume")
    trainer.save_checkpoint(os.path.join(ckpt_dir, "skybranch_model_final.pth"), kind="model")


if __name__ == "__main__":
    main()
