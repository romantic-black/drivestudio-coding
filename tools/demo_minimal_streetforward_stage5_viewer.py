from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import viser
from omegaconf import OmegaConf

from models.streetforward.minimal_trainer_stage5_6_production import (
    MinimalStreetForwardStage5_6_Production,
)
from streetforward_eval.stage5_6_runtime import configure_segment_finetune_optimizer
from tools.streetforward_stage5_demo_controller import Stage5DemoController
from tools.streetforward_stage5_viewer import StreetForwardStage5Viewer
from tools.train_minimal_streetforward_stage1_1 import current_time
from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _normalize_omp_num_threads
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4_for_demo,
)
from tools.streetforward_stage5_demo_scheduler import build_stage5_demo_scheduler_from_cfg
from utils.logging import setup_logging

logger = logging.getLogger(__name__)
_normalize_omp_num_threads()


def _select_trainer(stage: str, cfg: Any):
    stage_norm = str(stage).strip()
    model_cfg = cfg.get("model") if cfg is not None else None
    use_production = bool(model_cfg.get("production_training", False)) if model_cfg is not None else False
    if stage_norm != "5_6" or not use_production:
        raise ValueError("Stage5 demo viewer refactor only supports model.stage='5_6' with production_training=true.")
    return MinimalStreetForwardStage5_6_Production


def _load_demo_cfg(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load(args.config_file)
    base_cfg_file = cfg.get("base_config_file")
    if base_cfg_file:
        base = OmegaConf.load(str(base_cfg_file))
        cfg = OmegaConf.merge(base, cfg)
    if getattr(args, "opts", None):
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    if cfg.get("logging") is None:
        cfg.logging = {}
    log_dir_override = cfg.logging.get("log_dir")
    if log_dir_override is not None and str(log_dir_override).strip():
        log_dir = os.path.abspath(str(log_dir_override).strip())
    else:
        output_root = str(cfg.logging.get("output_root", getattr(args, "output_root", "outputs"))).strip()
        project = str(cfg.logging.get("project", "minimal_sf_stage5_6_demo_viewer")).strip()
        run_name = str(cfg.get("output_name", "stage5_6_demo_viewer")).strip()
        log_dir = os.path.join(output_root, project, run_name)
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for sub in ("images", "checkpoints", "tb"):
        os.makedirs(os.path.join(log_dir, sub), exist_ok=True)
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    return cfg


def _patch_cfg_for_demo(cfg: Any, args: argparse.Namespace) -> None:
    cfg.model.stage = "5_6"
    cfg.model.production_training = True
    if cfg.get("data") is not None:
        if cfg.data.get("preload") is None:
            cfg.data.preload = {}
        cfg.data.preload.enable = False
    if cfg.get("demo") is None:
        cfg.demo = {}
    if cfg.demo.get("scheduler") is None:
        cfg.demo.scheduler = {}
    cfg.demo.scheduler.type = "eval_v8_stage5_6"
    cfg.demo.scheduler.initial_scene_id = None if args.scene_id is None else int(args.scene_id)
    cfg.demo.scheduler.initial_segment_id = None if args.segment_id is None else int(args.segment_id)
    if args.sequence_start_pos is not None:
        cfg.demo.scheduler.initial_sequence_start_pos = int(args.sequence_start_pos)
    if args.scene_id is not None:
        cfg.demo.scheduler.scene_ids = [int(args.scene_id)]
        cfg.data.train_scene_ids = [int(args.scene_id)]
        cfg.data.eval_scene_ids = []


def _checkpoint_network_only_filter(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    skip_prefixes = (
        "optimizer",
        "node_states_",
        "h_cache_",
        # Keep trainable module params such as stage5_2_history_proj.*.
        # Only drop runtime caches/checkpoint-only state buckets.
        "stage5_2_history_bg",
        "stage5_2_history_distant",
        "stage5_2_history_rigid",
        "stage5_2_block_support_bg",
        "stage5_2_block_support_distant",
        "stage5_2_block_support_rigid",
    )
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if not str(key).startswith(skip_prefixes):
            out[key] = value
    return out


def _checkpoint_step(payload: Dict[str, Any]) -> int:
    for key in ("global_step", "step", "iteration", "iter"):
        if isinstance(payload, dict) and payload.get(key) is not None:
            try:
                return int(payload.get(key))
            except Exception:
                pass
    lr_info = payload.get("lr_scheduler") if isinstance(payload, dict) else None
    if isinstance(lr_info, dict) and lr_info.get("global_step") is not None:
        try:
            return int(lr_info.get("global_step"))
        except Exception:
            pass
    opt_state = payload.get("optimizer_state_dict") if isinstance(payload, dict) else None
    if isinstance(opt_state, dict) and opt_state.get("_sf_global_step") is not None:
        try:
            return int(opt_state.get("_sf_global_step"))
        except Exception:
            pass
    return 0


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, *, mode: str, strict: bool) -> int:
    if not ckpt_path:
        raise ValueError("Stage5_6 demo viewer requires a checkpoint path.")
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict):
        ckpt_step = _checkpoint_step(raw)
        state = raw.get("model_state_dict")
        if state is None:
            state = raw.get("model")
        if state is None:
            state = raw.get("state_dict")
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
    incompatible = model.load_state_dict(state, strict=bool(strict))
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    logger.info(
        "Loaded checkpoint mode=%s strict=%s path=%s missing=%d unexpected=%d",
        mode,
        bool(strict),
        ckpt_path,
        len(missing),
        len(unexpected),
    )
    setattr(model, "global_step", int(ckpt_step))
    opt = getattr(model, "optimizer", None)
    if opt is not None and hasattr(opt, "global_step"):
        opt.global_step = int(ckpt_step)
    return int(ckpt_step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/viewer/demo_minimal_streetforward_stage5_6_viewer.yaml",
        help="Path to demo config YAML.",
    )
    parser.add_argument("--stage", type=str, default="5_6", help="Only 5_6 is supported by this viewer path.")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path.")
    parser.add_argument("--scene_id", type=int, default=None, help="Initial scene id for demo traversal.")
    parser.add_argument("--segment_id", type=int, default=None, help="Fixed segment id (optional).")
    parser.add_argument("--sequence_start_pos", type=int, default=None, help="Initial eval window start position.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viewer host.")
    parser.add_argument("--port", type=int, default=8080, help="Viewer port.")
    parser.add_argument("--device", type=str, default="", help="Explicit torch device, e.g. cuda:0 / cpu.")
    parser.add_argument("--ckpt_load_mode", type=str, default="", help="network_only or full_state; config default is used when empty.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer, execute step loop only.")
    parser.add_argument("--max_steps", type=int, default=0, help="Headless max steps, 0 means no limit.")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("opts", nargs="*", help="OmegaConf overrides")
    args = parser.parse_args()

    cfg = _load_demo_cfg(args)
    _patch_cfg_for_demo(cfg, args)
    logger.info("Patched demo config:\n%s", OmegaConf.to_yaml(cfg))
    OmegaConf.save(config=cfg, f=os.path.join(str(cfg.log_dir), "config.yaml"))
    if str(args.stage).strip() != "5_6":
        raise ValueError("--stage must be 5_6; this refactored viewer is Stage5_6-only.")
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
    scheduler = build_stage5_demo_scheduler_from_cfg(cfg, dataset, device=device)
    trainer_cls = _select_trainer(args.stage, cfg)
    trainer = trainer_cls(config=cfg, device=device).to(device)
    if hasattr(trainer, "bind_eval_dataset"):
        trainer.bind_eval_dataset(dataset)
    else:
        setattr(trainer, "_bound_dataset", dataset)
    demo_checkpoint = (cfg.get("demo") or {}).get("checkpoint") or {}
    batch_eval_checkpoint = (cfg.get("batch_eval") or {}).get("checkpoint") or {}
    ckpt_path = str(args.ckpt or demo_checkpoint.get("path") or batch_eval_checkpoint.get("path") or "")
    ckpt_mode = str(args.ckpt_load_mode or demo_checkpoint.get("load_mode", "full_state"))
    ckpt_strict = bool(demo_checkpoint.get("strict", batch_eval_checkpoint.get("strict", True)))
    ckpt_step = _load_checkpoint(trainer, ckpt_path, mode=ckpt_mode, strict=ckpt_strict)
    demo_cfg = cfg.get("demo") or {}
    demo_mode = str(demo_cfg.get("mode", "frozen_recurrent_inference")).strip().lower()
    if demo_mode in ("segment_finetune_train", "validation_v8_segment_finetune_train"):
        finetune_cfg = {}
        if cfg.get("batch_eval") is not None and cfg.batch_eval.get("finetune") is not None:
            finetune_cfg.update(dict(OmegaConf.to_container(cfg.batch_eval.finetune, resolve=True)))
        if demo_cfg.get("finetune") is not None:
            finetune_cfg.update(dict(OmegaConf.to_container(demo_cfg.finetune, resolve=True)))
        configure_segment_finetune_optimizer(
            trainer,
            finetune_cfg=finetune_cfg,
            start_step=int(ckpt_step),
            log_prefix="demo_viewer",
        )
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
