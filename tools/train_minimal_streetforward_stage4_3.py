"""
Training script for Minimal StreetForward Stage 4.3.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict, Optional, TextIO


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

import torch

from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3
from tools.train_minimal_streetforward_stage1_1 import (
    _open_metrics_history,
    _write_metrics_history,
    convert_batch_to_minimal_format,
    setup,
)
from utils.minimal_batch_view_selection import parse_view_selection
from utils.streetforward_baseline import set_deterministic_seed

logger = logging.getLogger(__name__)
CKPT_PREFIX = "minimal_sf_stage4_3"


def main():
    parser = argparse.ArgumentParser(description="Train Minimal StreetForward Stage 4.3")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/minimal_streetforward_stage4_3.yaml",
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
    set_deterministic_seed(args.seed)

    overfit_path = getattr(args, "overfit_batch_path", None) or cfg.get("overfit_batch_path")
    if not overfit_path or not os.path.isfile(overfit_path):
        raise FileNotFoundError("Overfit batch required. Set --overfit_batch_path or config overfit_batch_path.")
    logger.info("RUN config_path=%s log_dir=%s overfit_batch_path=%s", args.config_file, cfg.log_dir, overfit_path)
    from tools.overfit_one_batch import load_batch

    raw_batch = load_batch(overfit_path)
    view_sel = cfg.training.get("view_selection")
    explicit = parse_view_selection(view_sel)
    num_targets = None if explicit is not None else cfg.training.get("num_targets", 1)
    minimal_batch = convert_batch_to_minimal_format(
        raw_batch,
        device,
        num_targets=num_targets,
        include_source_for_2d=True,
        view_selection=view_sel,
    )

    model = MinimalStreetForwardStage4_3(config=cfg, device=device)
    model.train()

    max_iterations = args.max_steps or cfg.training.get("max_iterations", 1000)
    log_interval = cfg.training.get("log_interval", 50)
    save_every = cfg.training.get("save_checkpoint_freq", 500)
    enable_jsonl_metrics = bool(cfg.logging.get("enable_jsonl_metrics", True))

    metrics_fh: Optional[TextIO] = None
    try:
        metrics_fh = _open_metrics_history(cfg.log_dir, enable_jsonl_metrics)
        for step in range(max_iterations):
            t0 = time.perf_counter()
            result = model.train_step(minimal_batch, step=step, profile_phase_timing=False, sync_cuda_timing=False)
            step_time_ms = float((time.perf_counter() - t0) * 1000.0)
            loss_val = float(result["loss"])

            if step % log_interval == 0:
                logger.info(
                    "Step %s: loss=%.6f rigid_update=%d bg_update=%d distant_update=%d onepass=%d",
                    step,
                    loss_val,
                    int(result.get("num_rigid_update", 0)),
                    int(result.get("num_bg_update", 0)),
                    int(result.get("num_distant_update", 0)),
                    int(result.get("src_backproject_pass_count", 0)),
                )

            row: Dict[str, Any] = {
                "step": int(step),
                "split": "train",
                "scene_id": int(minimal_batch.get("scene_id", -1)),
                "segment_id": int(minimal_batch.get("segment_id", -1)),
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
                "num_bg_src_feat_valid": int(result.get("num_bg_src_feat_valid", 0)),
                "num_bg_update": int(result.get("num_bg_update", 0)),
                "bg_update_ratio": float(result.get("bg_update_ratio", 0.0)),
                "num_distant_src_feat_valid": int(result.get("num_distant_src_feat_valid", 0)),
                "num_distant_update": int(result.get("num_distant_update", 0)),
                "distant_update_ratio": float(result.get("distant_update_ratio", 0.0)),
                "writeback_bg_ratio": float(result.get("writeback_bg_ratio", 0.0)),
                "writeback_distant_ratio": float(result.get("writeback_distant_ratio", 0.0)),
                "src_backproject_pass_count": int(result.get("src_backproject_pass_count", 0)),
                "hidden_norm_bg_mean": float(result.get("hidden_norm_bg_mean", 0.0)),
                "hidden_norm_distant_mean": float(result.get("hidden_norm_distant_mean", 0.0)),
                "hidden_norm_rigid_mean": float(result.get("hidden_norm_rigid_mean", 0.0)),
                "grad_norm_bg": float(result.get("grad_norm_bg", 0.0)),
                "grad_norm_distant": float(result.get("grad_norm_distant", 0.0)),
                "grad_norm_rigid": float(result.get("grad_norm_rigid", 0.0)),
                "step_time_ms": step_time_ms,
            }
            for k, v in result.items():
                if (
                    k.startswith("bg_offset_")
                    or k.startswith("rigid_offset_")
                    or k.startswith("perf_")
                ):
                    row[k] = float(v)
            _write_metrics_history(metrics_fh, row)

            if save_every and step > 0 and step % save_every == 0:
                ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_step{step}.pt")
                torch.save(
                    {"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": model.optimizer.state_dict()},
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)
    finally:
        if metrics_fh is not None:
            metrics_fh.close()

    final_ckpt = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_final.pt")
    torch.save(
        {"step": max_iterations - 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": model.optimizer.state_dict()},
        final_ckpt,
    )
    logger.info("Saved final checkpoint to %s", final_ckpt)


if __name__ == "__main__":
    main()

