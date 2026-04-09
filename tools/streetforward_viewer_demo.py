from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import viser

from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3
from tools.streetforward_demo_controller import StreetForwardDemoController
from tools.streetforward_viewer import StreetForwardViewer
from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import (
    _load_init_checkpoint,
    _normalize_omp_num_threads,
)
from tools.train_minimal_streetforward_stage4_3_v4_common import (
    build_multi_scene_dataset_v3,
    build_train_scheduler_v4_from_cfg,
)
from tools.train_minimal_streetforward_stage1_1 import setup

_normalize_omp_num_threads()
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--viewer_port",
        type=int,
        default=8080,
        help="Web viewer port.",
    )
    parser.add_argument(
        "--viewer_host",
        type=str,
        default="0.0.0.0",
        help="Web viewer host.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Optional model checkpoint to initialize Stage4.3 model.",
    )
    parser.add_argument(
        "--init_weights_only",
        action="store_true",
        help="Load model weights only (fresh optimizer state).",
    )
    parser.add_argument("opts", nargs="*", help="OmegaConf overrides")
    args = parser.parse_args()

    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("StreetForward demo: device=%s", device)

    dataset = build_multi_scene_dataset_v3(cfg, device)
    dataset.initialize()
    scheduler = build_train_scheduler_v4_from_cfg(cfg, dataset)

    trainer = MinimalStreetForwardStage4_3(config=cfg, device=device)
    trainer.train()
    _load_init_checkpoint(
        args.init_checkpoint,
        trainer,
        device,
        weights_only=bool(args.init_weights_only),
    )

    controller = StreetForwardDemoController(
        cfg=cfg,
        dataset=dataset,
        scheduler=scheduler,
        trainer=trainer,
        device=device,
    )
    controller.prime_first_snapshot()

    server = viser.ViserServer(host=args.viewer_host, port=int(args.viewer_port), verbose=True)
    _viewer = StreetForwardViewer(
        server=server,
        controller=controller,
        output_dir=Path(cfg.log_dir) / "viewer_demo",
    )
    logger.info("StreetForward viewer is running at http://%s:%d", args.viewer_host, int(args.viewer_port))

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("StreetForward viewer demo stopped.")


if __name__ == "__main__":
    main()

