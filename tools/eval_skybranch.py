from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

from models.streetforward.sky_branch import MinimalSkyBranchTrainer
from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v8_from_cfg,
)

logger = logging.getLogger(__name__)


def _mean(rows: list[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return float(sum(vals) / max(len(vals), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate standalone SkyBranch with opacity blender")
    parser.add_argument("--config_file", default="configs/minimal_skybranch_v0.yaml")
    parser.add_argument("--checkpoint", required=True, help="SkyBranch model checkpoint; runtime sky state is ignored.")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--output", default=None)
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
    trainer.load_model_checkpoint(args.checkpoint, strict=True)
    trainer.sky_branch.reset_runtime_state()
    rows: list[Dict[str, Any]] = []
    for step in range(1, int(args.max_steps) + 1):
        raw_batch = scheduler.next_batch()
        minimal = convert_batch_to_minimal_format(
            raw_batch,
            device=device,
            num_targets=int(raw_batch["target"]["image"].shape[0]),
            include_source_for_2d=True,
            view_selection=cfg.training.get("view_selection"),
        )
        sync = raw_batch.get("_scheduler_v8_aligned_info")
        with torch.no_grad():
            scene_pack = trainer.scene_provider.render_batch(minimal, scheduler_node_sync=sync, update_scene_state=True)
            out = trainer.sky_branch.forward_scene_batch(minimal, scene_pack, writeback=False)
        trainer.sky_branch.commit_forward_output(out)
        trainer.scene_provider.apply_pending_reset()
        logs = {"loss": float(out.loss.detach().item())}
        logs.update({k: float(v.detach().item()) if torch.is_tensor(v) else float(v) for k, v in out.logs.items()})
        rows.append(logs)
        logger.info(
            "eval_rollout_step=%d composite_psnr=%.3f sky_psnr=%.3f",
            step,
            logs.get("composite_psnr", 0.0),
            logs.get("sky_psnr", 0.0),
        )
    summary = {
        "num_steps": len(rows),
        "composite_psnr": _mean(rows, "composite_psnr"),
        "sky_psnr": _mean(rows, "sky_psnr"),
        "non_sky_psnr": _mean(rows, "non_sky_psnr"),
    }
    output = args.output or os.path.join(str(cfg.get("log_dir", "outputs/skybranch")), "skybranch_eval_summary.json")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", output)


if __name__ == "__main__":
    main()
