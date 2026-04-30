from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict

import torch
import viser

from models.streetforward.minimal_trainer_stage5_0 import MinimalStreetForwardStage5_0
from models.streetforward.minimal_trainer_stage5_2 import MinimalStreetForwardStage5_2
from models.streetforward.minimal_trainer_stage5_3 import MinimalStreetForwardStage5_3
from models.streetforward.minimal_trainer_stage5_3_production import (
    MinimalStreetForwardStage5_3_Production,
)
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.minimal_trainer_stage5_4_production import (
    MinimalStreetForwardStage5_4_Production,
)
from tools.streetforward_stage5_demo_controller import Stage5DemoController
from tools.streetforward_stage5_viewer import StreetForwardStage5Viewer
from tools.train_minimal_streetforward_stage1_1 import setup
from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _normalize_omp_num_threads
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4_for_demo,
)
from tools.streetforward_stage5_demo_scheduler import build_stage5_demo_scheduler_from_cfg

logger = logging.getLogger(__name__)
_normalize_omp_num_threads()


def _select_trainer(stage: str, cfg: Any):
    stage_norm = str(stage).strip()
    model_cfg = cfg.get("model") if cfg is not None else None
    use_production = bool(model_cfg.get("production_training", False)) if model_cfg is not None else False
    if stage_norm == "5_0":
        return MinimalStreetForwardStage5_0
    if stage_norm == "5_2":
        return MinimalStreetForwardStage5_2
    if stage_norm == "5_3":
        if use_production:
            return MinimalStreetForwardStage5_3_Production
        return MinimalStreetForwardStage5_3
    if stage_norm == "5_4":
        if use_production:
            return MinimalStreetForwardStage5_4_Production
        return MinimalStreetForwardStage5_4
    raise ValueError(f"Unsupported stage={stage_norm!r}, expected one of: 5_0, 5_2, 5_3, 5_4")


def _patch_cfg_for_demo(cfg: Any, args: argparse.Namespace) -> None:
    cfg.model.stage = str(args.stage)
    if cfg.get("scheduler_v8") is not None:
        cfg.scheduler_v8.enable = False
    if cfg.get("data") is not None:
        if cfg.data.get("preload") is None:
            cfg.data.preload = {}
        cfg.data.preload.enable = False
    if cfg.get("demo") is None:
        cfg.demo = {}
    if cfg.demo.get("scheduler") is None:
        cfg.demo.scheduler = {}
    cfg.demo.scheduler.initial_scene_id = None if args.scene_id is None else int(args.scene_id)
    cfg.demo.scheduler.initial_segment_id = None if args.segment_id is None else int(args.segment_id)
    if args.scene_id is not None:
        cfg.data.train_scene_ids = [int(args.scene_id)]
        cfg.data.eval_scene_ids = []


def _checkpoint_network_only_filter(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    skip_prefixes = (
        "optimizer",
        "node_states_",
        "h_cache_",
        "stage5_2_history_",
        "stage5_2_block_support_",
    )
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if not str(key).startswith(skip_prefixes):
            out[key] = value
    return out


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, mode: str) -> None:
    if not ckpt_path:
        logger.warning("No --ckpt provided; demo will run from random init weights.")
        return
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict):
        state = raw.get("model_state_dict")
        if state is None:
            state = raw.get("model")
        if state is None and all(isinstance(k, str) for k in raw.keys()):
            state = raw
    else:
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")
    if state is None:
        raise ValueError(f"Checkpoint missing model state_dict: {ckpt_path}")
    if str(mode) == "network_only":
        state = _checkpoint_network_only_filter(state)
    elif str(mode) != "full_state":
        raise ValueError(f"Unsupported ckpt_load_mode={mode!r}.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded checkpoint mode=%s path=%s missing=%d unexpected=%d",
        mode,
        ckpt_path,
        len(missing),
        len(unexpected),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/demo_minimal_streetforward_stage5_viewer.yaml",
        help="Path to demo config YAML.",
    )
    parser.add_argument("--stage", type=str, default="5_3", help="Stage trainer variant: 5_0 / 5_2 / 5_3 / 5_4.")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path.")
    parser.add_argument("--scene_id", type=int, default=None, help="Initial scene id for demo traversal.")
    parser.add_argument("--segment_id", type=int, default=None, help="Fixed segment id (optional).")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viewer host.")
    parser.add_argument("--port", type=int, default=8080, help="Viewer port.")
    parser.add_argument("--device", type=str, default="", help="Explicit torch device, e.g. cuda:0 / cpu.")
    parser.add_argument("--ckpt_load_mode", type=str, default="network_only", help="network_only or full_state.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer, execute step loop only.")
    parser.add_argument("--max_steps", type=int, default=0, help="Headless max steps, 0 means no limit.")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("opts", nargs="*", help="OmegaConf overrides")
    args = parser.parse_args()

    cfg = setup(args)
    _patch_cfg_for_demo(cfg, args)
    device = torch.device(args.device) if str(args.device).strip() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("Stage5 demo currently requires CUDA device because gsplat rasterization is CUDA-only.")
    logger.info("Stage5 demo device=%s stage=%s", device, args.stage)

    dataset = build_multi_scene_dataset_v4_for_demo(cfg, device)
    if getattr(dataset, "_preload_manager", None) is not None:
        raise RuntimeError("Demo dataset must disable preload, but _preload_manager is not None.")
    if getattr(dataset, "_preload_rtcfg", None) is not None:
        raise RuntimeError("Demo dataset must disable preload, but _preload_rtcfg is not None.")
    logger.info(
        "Demo preload check: cfg.data.preload.enable=%s preload_manager=%s preload_rtcfg=%s",
        bool((cfg.get("data") or {}).get("preload", {}).get("enable", False)),
        "none" if getattr(dataset, "_preload_manager", None) is None else "present",
        "none" if getattr(dataset, "_preload_rtcfg", None) is None else "present",
    )
    dataset.initialize()
    scheduler = build_stage5_demo_scheduler_from_cfg(cfg, dataset)
    trainer_cls = _select_trainer(args.stage, cfg)
    trainer = trainer_cls(config=cfg, device=device).to(device)
    _load_checkpoint(trainer, args.ckpt, mode=str(args.ckpt_load_mode))
    demo_cfg = cfg.get("demo") or {}
    demo_mode = str(demo_cfg.get("mode", "frozen_recurrent_inference")).strip().lower()
    if demo_mode in ("segment_finetune_train", "validation_v8_segment_finetune_train"):
        for p in trainer.parameters():
            p.requires_grad_(True)
        trainer.train()
        logger.info("Stage5 demo mode=%s (train+infer path enabled)", demo_mode)
    else:
        for p in trainer.parameters():
            p.requires_grad_(False)
        trainer.eval()
        logger.info("Stage5 demo mode=%s (frozen infer path enabled)", demo_mode)

    controller = Stage5DemoController(
        cfg=cfg,
        dataset=dataset,
        scheduler=scheduler,
        trainer=trainer,
        device=device,
        stage=args.stage,
    )
    controller.prime()

    if bool(args.headless):
        max_steps = int(args.max_steps)
        step = 0
        while True:
            controller.step_once()
            step += 1
            if max_steps > 0 and step >= max_steps:
                break
        logger.info("Headless demo finished at step=%d", step)
        return

    server = viser.ViserServer(host=str(args.host), port=int(args.port), verbose=True)
    _viewer = StreetForwardStage5Viewer(
        server=server,
        controller=controller,
        output_dir=Path(cfg.log_dir) / "stage5_demo_viewer",
    )
    logger.info("Stage5 viewer is running at http://%s:%d", args.host, int(args.port))
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
