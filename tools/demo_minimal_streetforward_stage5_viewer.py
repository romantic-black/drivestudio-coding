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
from tools.streetforward_stage5_demo_controller import Stage5DemoController
from tools.streetforward_stage5_viewer import StreetForwardStage5Viewer
from tools.train_minimal_streetforward_stage1_1 import setup
from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _normalize_omp_num_threads
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v8_from_cfg,
)

logger = logging.getLogger(__name__)
_normalize_omp_num_threads()


def _select_trainer(stage: str):
    stage_norm = str(stage).strip()
    if stage_norm == "5_0":
        return MinimalStreetForwardStage5_0
    if stage_norm == "5_2":
        return MinimalStreetForwardStage5_2
    if stage_norm == "5_3":
        return MinimalStreetForwardStage5_3
    raise ValueError(f"Unsupported stage={stage_norm!r}, expected one of: 5_0, 5_2, 5_3")


def _patch_cfg_for_demo(cfg: Any, args: argparse.Namespace) -> None:
    if cfg.get("scheduler_v8") is None or bool(cfg.scheduler_v8.get("enable")) is not True:
        raise ValueError("Demo requires scheduler_v8.enable=true.")
    target_policy = str(cfg.scheduler_v8.episode.get("target_policy", ""))
    reset_policy = str(cfg.scheduler_v8.execution.get("reset_policy", ""))
    if target_policy != "visited_episode_frames":
        raise ValueError("Demo requires scheduler_v8.episode.target_policy=visited_episode_frames.")
    if reset_policy != "episode_end":
        raise ValueError("Demo requires scheduler_v8.execution.reset_policy=episode_end.")

    cfg.model.stage = str(args.stage)
    cfg.scheduler_v8.traversal.fixed_scene_id = int(args.scene_id)
    if args.segment_id is None:
        cfg.scheduler_v8.traversal.fixed_segment_id = None
    else:
        cfg.scheduler_v8.traversal.fixed_segment_id = int(args.segment_id)
    if int(args.scene_id) not in [int(x) for x in cfg.data.train_scene_ids]:
        cfg.data.train_scene_ids.append(int(args.scene_id))


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
    parser.add_argument("--stage", type=str, default="5_2", help="Stage trainer variant: 5_0 / 5_2 / 5_3.")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path.")
    parser.add_argument("--scene_id", type=int, required=True, help="Fixed scene id for demo traversal.")
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

    dataset = build_multi_scene_dataset_v4(cfg, device)
    dataset.initialize()
    scheduler = build_train_scheduler_v8_from_cfg(cfg, dataset)
    trainer_cls = _select_trainer(args.stage)
    trainer = trainer_cls(config=cfg, device=device).to(device)
    _load_checkpoint(trainer, args.ckpt, mode=str(args.ckpt_load_mode))
    for p in trainer.parameters():
        p.requires_grad_(False)
    trainer.eval()

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

