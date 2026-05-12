from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.streetforward.minimal_trainer_stage5_6_production import (
    MinimalStreetForwardStage5_6_Production,
)
from streetforward_eval.stage5_6_runtime import configure_segment_finetune_optimizer
from tools.streetforward_stage5_demo_controller import Stage5DemoController
from tools.streetforward_stage5_demo_scheduler import build_stage5_demo_scheduler_from_cfg
from tools.streetforward_stage5_demo_video import Stage5DemoVideoExporter, derive_input_offsets
from tools.train_minimal_streetforward_stage1_1 import current_time
from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _normalize_omp_num_threads
from tools.train_minimal_streetforward_stage4_3_v8_common import build_multi_scene_dataset_v4_for_demo
from utils.logging import setup_logging

logger = logging.getLogger(__name__)
_normalize_omp_num_threads()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        from omegaconf import ListConfig

        if isinstance(value, ListConfig):
            return list(value)
    except Exception:
        pass
    return [value]


def _plain_dict(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return dict(OmegaConf.to_container(node, resolve=True))
    if isinstance(node, dict):
        return dict(node)
    return {}


def _checkpoint_network_only_filter(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    skip_prefixes = (
        "optimizer",
        "node_states_",
        "h_cache_",
        "stage5_2_history_bg",
        "stage5_2_history_distant",
        "stage5_2_history_rigid",
        "stage5_2_block_support_bg",
        "stage5_2_block_support_distant",
        "stage5_2_block_support_rigid",
    )
    return {k: v for k, v in state_dict.items() if not str(k).startswith(skip_prefixes)}


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
        raise ValueError("Stage5_6 demo video requires a checkpoint path.")
    raw = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")
    ckpt_step = _checkpoint_step(raw)
    state = raw.get("model_state_dict") or raw.get("model") or raw.get("state_dict")
    if state is None and all(isinstance(k, str) for k in raw.keys()):
        state = raw
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


def _collapse_sky_runtime_to_single_state(sky_branch: Any) -> Optional[Any]:
    states = getattr(sky_branch, "node_states_sky", None)
    if not isinstance(states, dict) or len(states) == 0:
        return None
    key = sorted(states.keys(), key=lambda x: str(x))[0]
    state = states[key]
    states.clear()
    states[key] = state
    h_cache = getattr(sky_branch, "h_cache_sky", None)
    if isinstance(h_cache, dict):
        h = h_cache.get(key)
        h_cache.clear()
        if h is not None:
            h_cache[key] = h
    return state


def _load_video_sky_branch_from_cfg(cfg: Any, device: torch.device) -> Optional[Any]:
    video_cfg = cfg.get("video") or {}
    sky_cfg = video_cfg.get("sky") or {}
    if not bool(sky_cfg.get("enable", False)):
        return None
    config_file = str(sky_cfg.get("config_file") or "")
    checkpoint = str(sky_cfg.get("checkpoint") or sky_cfg.get("path") or "")
    if not config_file:
        raise ValueError("video.sky.config_file is required when video.sky.enable=true")
    if not checkpoint:
        raise ValueError("video.sky.checkpoint is required when video.sky.enable=true")

    from models.streetforward.sky_branch import SkyBranchV0

    sky_model_cfg = OmegaConf.load(config_file)
    sky_branch = SkyBranchV0(sky_model_cfg, device=device).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"SkyBranch checkpoint must be dict-like, got {type(payload).__name__}")
    state = payload.get("sky_branch_state_dict") or payload.get("model_state_dict") or payload.get("state_dict") or payload
    if not isinstance(state, dict):
        raise TypeError(f"SkyBranch checkpoint state must be dict-like, got {type(state).__name__}")
    strict = bool(sky_cfg.get("strict", True))
    incompatible = sky_branch.load_state_dict(state, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if bool(sky_cfg.get("freeze_params", True)):
        for p in sky_branch.parameters():
            p.requires_grad_(False)
    sky_branch.eval()

    loaded_runtime = False
    if bool(sky_cfg.get("load_runtime_state", True)):
        if payload.get("node_states_sky"):
            sky_branch.load_runtime_state_dict(payload)
            loaded_runtime = True
        elif bool(sky_cfg.get("require_runtime_state", True)):
            raise ValueError(
                "video.sky.load_runtime_state=true but checkpoint has no node_states_sky. "
                "Use a skybranch_resume checkpoint or set video.sky.require_runtime_state=false."
            )
    else:
        sky_branch.reset_runtime_state()
    single_state = _collapse_sky_runtime_to_single_state(sky_branch) if bool(sky_cfg.get("reuse_single_state", True)) else None
    logger.info(
        "Loaded frozen SkyBranch for demo video checkpoint=%s strict=%s missing=%d unexpected=%d "
        "runtime_state=%s single_state=%s",
        checkpoint,
        strict,
        len(missing),
        len(unexpected),
        loaded_runtime,
        single_state is not None,
    )
    return sky_branch


def _load_cfg(args: argparse.Namespace) -> Any:
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
        project = str(cfg.logging.get("project", "minimal_sf_stage5_6_demo_video")).strip()
        run_name = str(cfg.get("output_name", "stage5_6_demo_video")).strip()
        log_dir = os.path.join(output_root, project, run_name)
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    return cfg


def _camera_preset(mode: str, cfg: Any) -> Tuple[List[int], List[str]]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "front":
        return [0], ["front"]
    if mode_norm in ("three_front", "front_three", "front_triplet"):
        return [0, 1, 2], ["front", "front_left", "front_right"]
    if mode_norm != "custom":
        raise ValueError("video.camera_mode must be one of: front, three_front, custom")
    cameras_cfg = (cfg.get("video") or {}).get("cameras") or {}
    ids = [int(x) for x in _as_list(cameras_cfg.get("ids", [0]))]
    names = [str(x) for x in _as_list(cameras_cfg.get("names", []))]
    if not names:
        names = [f"cam{int(x)}" for x in ids]
    if len(ids) != len(names):
        raise ValueError("video.cameras.ids and video.cameras.names length mismatch")
    return ids, names


def _set_camera_cfg(cfg: Any, ids: List[int], names: List[str]) -> None:
    payload = {"ids": [int(x) for x in ids], "names": [str(x) for x in names]}
    if cfg.get("video") is None:
        cfg.video = {}
    cfg.video.cameras = payload
    if cfg.get("demo") is None:
        cfg.demo = {}
    cfg.demo.cameras = payload
    cfg.demo.update_cameras = payload
    if cfg.demo.get("scheduler") is None:
        cfg.demo.scheduler = {}
    cfg.demo.scheduler.cameras = payload
    cfg.demo.scheduler.update_cameras = payload
    if cfg.get("batch_eval") is None:
        cfg.batch_eval = {}
    cfg.batch_eval.cameras = payload
    cfg.batch_eval.update_cameras = payload


def _set_update_camera_cfg(cfg: Any, ids: List[int], names: Optional[List[str]] = None) -> None:
    payload = {"ids": [int(x) for x in ids]}
    if names is not None and len(names) > 0:
        if len(names) != len(ids):
            raise ValueError("update camera ids/names length mismatch")
        payload["names"] = [str(x) for x in names]
    if cfg.get("demo") is None:
        cfg.demo = {}
    cfg.demo.update_cameras = payload
    if cfg.demo.get("scheduler") is None:
        cfg.demo.scheduler = {}
    cfg.demo.scheduler.update_cameras = payload
    if cfg.get("batch_eval") is None:
        cfg.batch_eval = {}
    cfg.batch_eval.update_cameras = payload


def _configured_scene_ids(cfg: Any) -> List[int]:
    scheduler_cfg = (cfg.get("demo") or {}).get("scheduler") or {}
    batch_dataset_cfg = (cfg.get("batch_eval") or {}).get("dataset") or {}
    raw = scheduler_cfg.get("scene_ids")
    if raw is None:
        raw = batch_dataset_cfg.get("scene_ids")
    if raw is None and cfg.get("data") is not None:
        raw = cfg.data.get("train_scene_ids")
    return [int(x) for x in _as_list(raw)]


def _train_frames_for_segment(dataset: Any, scene_id: int, segment_id: int) -> List[int]:
    sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
    frames = [int(x) for x in sorted(sidx.frame_indices)]
    train_frame_set = getattr(sidx, "train_frame_set", None)
    if train_frame_set is not None:
        train_set = set(int(x) for x in train_frame_set)
        frames = [int(f) for f in frames if int(f) in train_set]
    return frames


def _resolve_initial_scope_from_cfg(cfg: Any, dataset: Any) -> Tuple[int, int]:
    scheduler_cfg = (cfg.get("demo") or {}).get("scheduler") or {}
    batch_dataset_cfg = (cfg.get("batch_eval") or {}).get("dataset") or {}
    start_at = batch_dataset_cfg.get("start_at") or {}
    scene_ids = _configured_scene_ids(cfg)
    raw_scene = scheduler_cfg.get("initial_scene_id")
    if raw_scene is None and hasattr(start_at, "get"):
        raw_scene = start_at.get("scene_id")
    if raw_scene is None and not scene_ids:
        raise ValueError("demo.scheduler.scene_ids or batch_eval.dataset.scene_ids must be set for demo video")
    scene_id = int(raw_scene if raw_scene is not None else scene_ids[0])
    raw_segment = scheduler_cfg.get("initial_segment_id")
    if raw_segment is None and hasattr(start_at, "get"):
        raw_segment = start_at.get("segment_id")
    if raw_segment is None:
        seg_ids = [int(x) for x in dataset.list_segment_ids(int(scene_id))]
        if not seg_ids:
            raise ValueError(f"scene_id={scene_id} has no registered segments")
        segment_id = int(seg_ids[0])
    else:
        segment_id = int(raw_segment)
    return int(scene_id), int(segment_id)


def _patch_initial_sequence_start_from_cfg(cfg: Any, dataset: Any) -> None:
    if cfg.get("demo") is None:
        cfg.demo = {}
    if cfg.demo.get("scheduler") is None:
        cfg.demo.scheduler = {}
    scheduler_cfg = cfg.demo.scheduler
    if scheduler_cfg.get("initial_sequence_start_pos") is not None:
        return
    batch_dataset_cfg = (cfg.get("batch_eval") or {}).get("dataset") or {}
    start_at = batch_dataset_cfg.get("start_at") or {}
    if not hasattr(start_at, "get"):
        return
    raw_scene = start_at.get("scene_id")
    raw_segment = start_at.get("segment_id")
    if raw_scene is None or raw_segment is None:
        return
    start_scene = int(raw_scene)
    start_segment = int(raw_segment)
    configured_initial_scene = scheduler_cfg.get("initial_scene_id")
    configured_initial_segment = scheduler_cfg.get("initial_segment_id")
    if configured_initial_scene is not None and int(configured_initial_scene) != int(start_scene):
        return
    if configured_initial_segment is not None and int(configured_initial_segment) != int(start_segment):
        return
    if start_at.get("sequence_start_pos") is not None:
        start_pos = int(start_at.get("sequence_start_pos"))
    elif start_at.get("frame_id") is not None:
        frame_id = int(start_at.get("frame_id"))
        frames = _train_frames_for_segment(dataset, start_scene, start_segment)
        if frame_id not in set(frames):
            preview = frames[:5] + (["..."] if len(frames) > 10 else []) + frames[-5:]
            raise ValueError(
                "batch_eval.dataset.start_at.frame_id is not in train frames: "
                f"scene={start_scene} segment={start_segment} frame_id={frame_id} "
                f"available_count={len(frames)} available_preview={preview}"
            )
        start_pos = [int(x) for x in frames].index(int(frame_id))
    else:
        return
    scheduler_cfg.initial_scene_id = int(start_scene)
    scheduler_cfg.initial_segment_id = int(start_segment)
    scheduler_cfg.initial_sequence_start_pos = int(start_pos)
    logger.info(
        "Demo video initial start resolved from batch_eval.dataset.start_at: scene_id=%d segment_id=%d start_pos=%d",
        int(start_scene),
        int(start_segment),
        int(start_pos),
    )


def _initialize_dataset_for_video(cfg: Any, dataset: Any) -> None:
    video_cfg = cfg.get("video") or {}
    init_scope = str(video_cfg.get("dataset_init_scope", "active_segment")).strip().lower()
    if init_scope in ("all", "full", "training_assets"):
        dataset.initialize()
        return
    if init_scope in ("selected_segments", "video_segments", "configured_segments"):
        scene_ids = _configured_scene_ids(cfg)
        if not scene_ids:
            scene_ids = [int(x) for x in dataset.list_training_scene_ids()]
        count = 0
        for scene_id in scene_ids:
            for segment_id in dataset.list_segment_ids(int(scene_id)):
                dataset.get_segment_index(int(scene_id), int(segment_id))
                count += 1
        setattr(dataset, "_initialized", True)
        logger.info(
            "Demo video dataset initialized for selected segments: scenes=%s segments=%d",
            [int(x) for x in scene_ids],
            int(count),
        )
        return
    if init_scope not in ("active_segment", "current_segment", "demo_segment"):
        raise ValueError("video.dataset_init_scope must be active_segment, selected_segments, or all")
    scene_id, segment_id = _resolve_initial_scope_from_cfg(cfg, dataset)
    # Avoid MultiSceneDatasetV4.initialize(), which validates every segment in
    # data.train_scene_ids. For demo video we only need the selected segment.
    dataset.get_segment_index(int(scene_id), int(segment_id))
    setattr(dataset, "_initialized", True)
    logger.info(
        "Demo video dataset initialized only for active segment: scene_id=%d segment_id=%d",
        int(scene_id),
        int(segment_id),
    )


def _patch_cfg_for_video(cfg: Any, args: argparse.Namespace) -> None:
    if cfg.get("model") is None:
        cfg.model = {}
    cfg.model.stage = "5_6"
    cfg.model.production_training = True
    if cfg.get("data") is None:
        cfg.data = {}
    if cfg.data.get("preload") is None:
        cfg.data.preload = {}
    cfg.data.preload.enable = False
    if cfg.get("demo") is None:
        cfg.demo = {}
    if cfg.demo.get("scheduler") is None:
        cfg.demo.scheduler = {}
    cfg.demo.scheduler.type = "eval_v8_stage5_6"
    cfg.demo.scheduler.name = str(cfg.demo.scheduler.get("name", "stage5_6_demo_video_eval_v8"))

    video_cfg = cfg.get("video") or {}
    recon_cfg = video_cfg.get("reconstruction") or {}
    window_size = int(recon_cfg.get("window_size", cfg.demo.scheduler.get("sequence_length", 8)))
    input_offsets = derive_input_offsets(
        window_size=int(window_size),
        input_gap_frames=int(recon_cfg.get("input_gap_frames", 1)),
        explicit=recon_cfg.get("input_offsets"),
    )
    window_stride = int(recon_cfg.get("window_stride", window_size))
    steps_per_input = int(recon_cfg.get("steps_per_input", cfg.demo.scheduler.get("steps_per_input", 16)))
    cfg.demo.scheduler.sequence_length = int(window_size)
    cfg.demo.scheduler.input_offsets = [int(x) for x in input_offsets]
    cfg.demo.scheduler.eval_offsets = "all"
    cfg.demo.scheduler.stride = int(window_stride)
    cfg.demo.scheduler.require_full_window = bool(recon_cfg.get("require_full_window", False))
    cfg.demo.scheduler.window_policy = str(recon_cfg.get("window_policy", "sliding"))
    cfg.demo.scheduler.steps_per_input = int(steps_per_input)
    cfg.demo.scheduler.block_order = str(recon_cfg.get("block_order", cfg.demo.scheduler.get("block_order", "block_major")))
    cfg.demo.scheduler.step_major_switch_interval_steps = int(
        recon_cfg.get(
            "step_major_switch_interval_steps",
            cfg.demo.scheduler.get("step_major_switch_interval_steps", 1),
        )
    )
    cfg.demo.scheduler.max_target_frames_including_source = int(
        recon_cfg.get(
            "max_target_frames_including_source",
            cfg.demo.scheduler.get("max_target_frames_including_source", len(input_offsets)),
        )
    )
    cfg.demo.scheduler.wrap_scene = bool(cfg.demo.scheduler.get("wrap_scene", True))
    cfg.demo.scheduler.wrap_segment = bool(cfg.demo.scheduler.get("wrap_segment", True))
    cfg.demo.scheduler.wrap_episode = bool(cfg.demo.scheduler.get("wrap_episode", True))

    if cfg.get("batch_eval") is None:
        cfg.batch_eval = {}
    if cfg.batch_eval.get("dataset") is None:
        cfg.batch_eval.dataset = {}
    cfg.batch_eval.dataset.stride = int(window_stride)
    cfg.batch_eval.dataset.require_full_window = bool(cfg.demo.scheduler.require_full_window)
    cfg.batch_eval.dataset.window_policy = str(cfg.demo.scheduler.window_policy)
    configured_scene_ids = _configured_scene_ids(cfg)
    if configured_scene_ids:
        cfg.demo.scheduler.scene_ids = [int(x) for x in configured_scene_ids]
        cfg.batch_eval.dataset.scene_ids = [int(x) for x in configured_scene_ids]
        cfg.data.train_scene_ids = [int(x) for x in configured_scene_ids]
        cfg.data.eval_scene_ids = []

    if args.scene_id is not None:
        cfg.demo.scheduler.scene_ids = [int(args.scene_id)]
        cfg.demo.scheduler.initial_scene_id = int(args.scene_id)
        cfg.batch_eval.dataset.scene_ids = [int(args.scene_id)]
        cfg.data.train_scene_ids = [int(args.scene_id)]
        cfg.data.eval_scene_ids = []
    if args.segment_id is not None:
        cfg.demo.scheduler.initial_segment_id = int(args.segment_id)
    if args.sequence_start_pos is not None:
        cfg.demo.scheduler.initial_sequence_start_pos = int(args.sequence_start_pos)

    camera_mode = str(args.camera_mode or video_cfg.get("camera_mode", "three_front"))
    camera_ids, camera_names = _camera_preset(camera_mode, cfg)
    _set_camera_cfg(cfg, camera_ids, camera_names)
    video_cfg = cfg.get("video") or {}
    update_cameras_cfg = video_cfg.get("update_cameras")
    if update_cameras_cfg is not None:
        update_ids = [int(x) for x in _as_list(update_cameras_cfg.get("ids", camera_ids))]
        update_names = [str(x) for x in _as_list(update_cameras_cfg.get("names", []))]
        _set_update_camera_cfg(cfg, update_ids, update_names if update_names else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/viewer/demo_minimal_streetforward_stage5_6_video.yaml",
        help="Path to Stage5_6 demo video config YAML.",
    )
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path override.")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Alias for --ckpt, kept for overfit render scripts.")
    parser.add_argument("--scene_id", type=int, default=None, help="Scene id override.")
    parser.add_argument("--segment_id", type=int, default=None, help="Segment id override.")
    parser.add_argument("--sequence_start_pos", type=int, default=None, help="Initial sequence start override.")
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="",
        help="front, three_front, or custom. Empty uses video.camera_mode from config.",
    )
    parser.add_argument("--device", type=str, default="", help="Explicit torch device, e.g. cuda:0 / cpu.")
    parser.add_argument("--ckpt_load_mode", type=str, default="", help="network_only or full_state; config default is used when empty.")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("opts", nargs="*", help="OmegaConf overrides")
    args = parser.parse_args()

    cfg = _load_cfg(args)
    _patch_cfg_for_video(cfg, args)
    OmegaConf.save(config=cfg, f=os.path.join(str(cfg.log_dir), "config.yaml"))
    logger.info("Patched Stage5_6 video config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device(args.device) if str(args.device).strip() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("Stage5_6 demo video requires CUDA because gsplat rasterization is CUDA-only.")

    dataset = build_multi_scene_dataset_v4_for_demo(cfg, device)
    _initialize_dataset_for_video(cfg, dataset)
    _patch_initial_sequence_start_from_cfg(cfg, dataset)
    OmegaConf.save(config=cfg, f=os.path.join(str(cfg.log_dir), "config.yaml"))
    scheduler = build_stage5_demo_scheduler_from_cfg(cfg, dataset, device=device)
    trainer = MinimalStreetForwardStage5_6_Production(config=cfg, device=device).to(device)
    if hasattr(trainer, "bind_eval_dataset"):
        trainer.bind_eval_dataset(dataset)
    else:
        setattr(trainer, "_bound_dataset", dataset)

    demo_checkpoint = (cfg.get("demo") or {}).get("checkpoint") or {}
    batch_eval_checkpoint = (cfg.get("batch_eval") or {}).get("checkpoint") or {}
    ckpt_path = str(args.ckpt or args.init_checkpoint or demo_checkpoint.get("path") or batch_eval_checkpoint.get("path") or "")
    ckpt_mode = str(args.ckpt_load_mode or demo_checkpoint.get("load_mode", "full_state"))
    ckpt_strict = bool(demo_checkpoint.get("strict", batch_eval_checkpoint.get("strict", True)))
    ckpt_step = _load_checkpoint(trainer, ckpt_path, mode=ckpt_mode, strict=ckpt_strict)

    demo_cfg = cfg.get("demo") or {}
    demo_mode = str(demo_cfg.get("mode", "segment_finetune_train")).strip().lower()
    if demo_mode in ("segment_finetune_train", "validation_v8_segment_finetune_train"):
        finetune_cfg: Dict[str, Any] = {}
        if cfg.get("batch_eval") is not None and cfg.batch_eval.get("finetune") is not None:
            finetune_cfg.update(_plain_dict(cfg.batch_eval.finetune))
        if demo_cfg.get("finetune") is not None:
            finetune_cfg.update(_plain_dict(demo_cfg.finetune))
        configure_segment_finetune_optimizer(
            trainer,
            finetune_cfg=finetune_cfg,
            start_step=int(ckpt_step),
            log_prefix="stage5_6_demo_video",
        )
        trainer.train()
    else:
        for p in trainer.parameters():
            p.requires_grad_(False)
        trainer.eval()
    sky_branch = _load_video_sky_branch_from_cfg(cfg, device)

    controller = Stage5DemoController(
        cfg=cfg,
        dataset=dataset,
        scheduler=scheduler,
        trainer=trainer,
        device=device,
        stage="5_6",
    )
    controller.prime()

    video_cfg = cfg.get("video") or {}
    output_cfg = video_cfg.get("output") or {}
    out_dir_cfg = output_cfg.get("dir")
    out_dir = Path(str(out_dir_cfg)) if out_dir_cfg is not None and str(out_dir_cfg).strip() else Path(str(cfg.log_dir)) / "videos"
    exporter = Stage5DemoVideoExporter(
        cfg=cfg,
        dataset=dataset,
        controller=controller,
        device=device,
        output_dir=out_dir,
        sky_branch=sky_branch,
    )
    metadata = exporter.export()
    logger.info("Stage5_6 demo video complete: %s", metadata.get("videos"))
    logger.info("Metadata: %s", metadata.get("metadata_path"))


if __name__ == "__main__":
    main()
