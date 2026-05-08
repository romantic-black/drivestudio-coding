from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf

from models.streetforward.sky_branch import MinimalSkyBranchTrainer
from tools.train_minimal_streetforward_stage1_1 import (
    _open_metrics_history,
    _write_metrics_history,
    convert_batch_to_minimal_format,
)
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v8_from_cfg,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _build_scheduler_node_sync_v8(cfg: Any, scheduler_info: Dict[str, Any], step_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    sv8 = _cfg_get(cfg, "scheduler_v8")
    if sv8 is None:
        return None
    execution = _cfg_get(sv8, "execution", {}) or {}
    block_order = str(scheduler_info.get("block_order", "block_major")).strip()
    reset_policy = str(_cfg_get(execution, "reset_policy", "episode_end" if block_order == "step_major" else "block_end")).strip()
    if reset_policy not in ("block_end", "episode_end", "never"):
        raise ValueError("scheduler_v8.execution.reset_policy must be one of ['block_end', 'episode_end', 'never']")
    if block_order == "step_major" and reset_policy == "block_end":
        raise ValueError("scheduler_v8.execution.reset_policy=block_end is incompatible with step_major block order.")
    if reset_policy == "block_end":
        should_reset = any(ev.get("type") == "block_end" for ev in step_events)
    elif reset_policy == "episode_end":
        should_reset = any(ev.get("type") == "episode_end" for ev in step_events)
    else:
        should_reset = False
    return {
        "U": int(scheduler_info.get("U", 1)),
        "segment_local_step": int(scheduler_info.get("segment_local_step", 0)),
        "reset_after_block": bool(should_reset),
        "reset_policy": str(reset_policy),
    }


def _scene_dir_str(scene_id: Any) -> str:
    try:
        value = int(scene_id)
    except (TypeError, ValueError):
        return "unknown"
    return "unknown" if value < 0 else f"{value:03d}"


def _event_for_log(step_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for event_type in ("episode_end", "block_end", "block_exit"):
        for ev in step_events:
            if ev.get("type") == event_type:
                return ev
    return None


def _log_scheduler_events(step_events: List[Dict[str, Any]]) -> None:
    for ev in step_events:
        ev_type = str(ev.get("type", ""))
        if ev_type == "segment_begin":
            logger.info(
                "SEGMENT_BEGIN global_step=%s scene_id=%s scene_dir=%s segment=%s U=%s segment_step_budget=%s updates_per_block=%s",
                ev.get("global_step"),
                ev.get("scene_id"),
                _scene_dir_str(ev.get("scene_id", -1)),
                ev.get("segment_id"),
                ev.get("U"),
                ev.get("segment_step_budget"),
                ev.get("updates_per_block"),
            )
        elif ev_type == "reset_event":
            logger.info(
                "RESET global_step=%s scene_id=%s scene_dir=%s segment=%s reason=%s window=%s num_pairs=%s",
                ev.get("global_step"),
                ev.get("scene_id"),
                _scene_dir_str(ev.get("scene_id", -1)),
                ev.get("segment_id"),
                ev.get("reason"),
                ev.get("window_keyframes"),
                ev.get("num_pairs"),
            )
        elif ev_type == "overlap_select":
            logger.info(
                "OVERLAP_SELECT global_step=%s scene_id=%s scene_dir=%s segment=%s overlap_mode=%s hits=%s misses=%s",
                ev.get("global_step"),
                ev.get("scene_id"),
                _scene_dir_str(ev.get("scene_id", -1)),
                ev.get("segment_id"),
                ev.get("overlap_mode"),
                ev.get("cache_hits"),
                ev.get("cache_misses"),
            )
        elif ev_type == "block_begin":
            logger.info(
                "BLOCK_BEGIN global_step=%s scene_id=%s scene_dir=%s segment=%s block_seg=%s block_global=%s U=%s "
                "K_u_eff=%s K_steps_eff=%s source_frame=%s source_image_ref=%s overlap_mode=%s",
                ev.get("global_step"),
                ev.get("scene_id"),
                _scene_dir_str(ev.get("scene_id", -1)),
                ev.get("segment_id"),
                ev.get("block_idx_in_segment"),
                ev.get("block_idx_global"),
                ev.get("U"),
                ev.get("K_u_effective"),
                ev.get("K_steps_effective"),
                ev.get("source_frame_idx"),
                ev.get("source_image_ref"),
                ev.get("overlap_mode"),
            )
        elif ev_type == "block_exit":
            logger.info(
                "BLOCK_EXIT global_step=%s scene_id=%s scene_dir=%s segment=%s block_seg=%s block_global=%s "
                "source_frame=%s source_image_ref=%s target_refs=%s num_updates_in_block=%s",
                ev.get("global_step"),
                ev.get("scene_id"),
                _scene_dir_str(ev.get("scene_id", -1)),
                ev.get("segment_id"),
                ev.get("block_idx_in_segment"),
                ev.get("block_idx_global"),
                ev.get("source_frame_idx"),
                ev.get("source_image_ref"),
                len(ev.get("target_image_refs") or []),
                ev.get("num_updates_in_block"),
            )


def _format_log_message(step: int, logs: Dict[str, Any], scheduler_info: Dict[str, Any], step_events: List[Dict[str, Any]]) -> str:
    ev = _event_for_log(step_events)
    kind = str(ev.get("type")) if ev is not None else "step"
    src = ev if ev is not None else scheduler_info
    parts = [
        f"{kind.upper()}",
        f"step={step}",
        f"scene={src.get('scene_id', scheduler_info.get('scene_id', -1))}",
        f"segment={src.get('segment_id', scheduler_info.get('segment_id', -1))}",
        f"episode={src.get('episode_idx_global', scheduler_info.get('episode_idx_global', -1))}",
        f"block={src.get('block_idx_in_episode', scheduler_info.get('block_idx_in_episode', -1))}",
        f"global_block={src.get('block_idx_global', scheduler_info.get('block_idx_global', -1))}",
        f"loss={float(logs.get('loss', 0.0)):.6f}",
        f"composite_psnr={float(logs.get('composite_psnr', 0.0)):.3f}",
        f"sky_psnr={float(logs.get('sky_psnr', 0.0)):.3f}",
    ]
    if "cuda_alloc_gb" in logs:
        parts.extend(
            [
                f"cuda_alloc={float(logs.get('cuda_alloc_gb', 0.0)):.2f}GB",
                f"cuda_reserved={float(logs.get('cuda_reserved_gb', 0.0)):.2f}GB",
                f"cuda_peak={float(logs.get('cuda_peak_alloc_gb', 0.0)):.2f}GB",
            ]
        )
    return " ".join(parts)


def _to_hwc01(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().float().cpu()
    if y.dim() == 4 and int(y.shape[0]) == 1:
        y = y[0]
    if y.dim() == 3 and int(y.shape[0]) in {1, 3} and int(y.shape[-1]) not in {1, 3}:
        y = y.permute(1, 2, 0)
    if y.dim() == 2:
        y = y.unsqueeze(-1)
    if y.dim() != 3:
        raise ValueError(f"Expected image tensor [H,W,C], got {tuple(y.shape)}")
    if int(y.shape[-1]) == 1:
        y = y.expand(-1, -1, 3)
    return y.clamp(0.0, 1.0)


def _save_png(path: str, image: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = (_to_hwc01(image).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    try:
        from PIL import Image
    except ImportError:
        torch.save(torch.from_numpy(arr), path + ".pt")
        return
    Image.fromarray(arr).save(path)


def _save_skybranch_debug_images(
    *,
    step: int,
    trainer: MinimalSkyBranchTrainer,
    minimal_batch: Dict[str, Any],
    scheduler_info: Dict[str, Any],
    log_dir: str,
) -> None:
    out = trainer.last_forward_output
    scene_pack = trainer.last_scene_pack
    if out is None or scene_pack is None:
        return
    block_idx = int(scheduler_info.get("block_idx_global", 0))
    scene_id = int(scheduler_info.get("scene_id", minimal_batch.get("scene_id", -1)))
    segment_id = int(scheduler_info.get("segment_id", minimal_batch.get("segment_id", -1)))
    base = os.path.join(log_dir, "images", "train_monitor")
    targets = list(minimal_batch.get("targets") or [])
    for v, target in enumerate(targets):
        suffix = f"step{step:06d}_b{block_idx:06d}_sc{_scene_dir_str(scene_id)}_seg{segment_id}_v{v}"
        gt = target["gt_image"].to(out.comp_rgb.device)
        sky_mask = target.get("sky_mask")
        err = (out.comp_rgb[v].detach() - gt.detach()).abs()
        _save_png(os.path.join(base, f"{suffix}_comp.png"), out.comp_rgb[v])
        _save_png(os.path.join(base, f"{suffix}_gt.png"), gt)
        _save_png(os.path.join(base, f"{suffix}_scene.png"), scene_pack.target_rgb[v])
        _save_png(os.path.join(base, f"{suffix}_sky.png"), out.sky_rgb[v])
        _save_png(os.path.join(base, f"{suffix}_error.png"), err / err.max().clamp_min(1.0e-6))
        _save_png(os.path.join(base, f"{suffix}_scene_alpha.png"), scene_pack.target_alpha[v])
        _save_png(os.path.join(base, f"{suffix}_sky_alpha.png"), out.sky_alpha[v])
        if sky_mask is not None:
            _save_png(os.path.join(base, f"{suffix}_sky_mask.png"), sky_mask)


def _image_due(step: int, cfg: Any, scheduler_info: Dict[str, Any], step_events: List[Dict[str, Any]]) -> bool:
    logging_cfg = _cfg_get(cfg, "logging", {}) or {}
    mode = str(_cfg_get(logging_cfg, "image_trigger", "block_end")).strip()
    if mode in {"none", "disabled", "false"}:
        return False
    interval_blocks = max(int(_cfg_get(logging_cfg, "image_interval_blocks", 50)), 1)
    steps_per_block = int(_cfg_get(_cfg_get(_cfg_get(cfg, "scheduler_v8", {}) or {}, "block", {}) or {}, "steps_per_block", 1))
    interval_steps = max(int(_cfg_get(logging_cfg, "image_interval_steps", interval_blocks * max(steps_per_block, 1))), 1)
    if mode in {"raw_step_interval", "step_interval"}:
        return step > 0 and step % interval_steps == 0
    if mode == "episode_end":
        if not any(ev.get("type") == "episode_end" for ev in step_events):
            return False
        completed_blocks = int(scheduler_info.get("block_idx_global", -1)) + 1
        return completed_blocks > 0 and completed_blocks % interval_blocks == 0
    if mode == "block_end":
        for ev in step_events:
            if ev.get("type") != "block_end":
                continue
            block_idx = int(ev.get("block_idx_global", scheduler_info.get("block_idx_global", -1)))
            if block_idx >= 1 and (block_idx - 1) % interval_blocks == 0:
                return True
        return False
    raise ValueError("logging.image_trigger must be one of ['block_end', 'episode_end', 'raw_step_interval', 'step_interval', 'none'].")


def _metrics_row(step: int, logs: Dict[str, Any], scheduler_info: Dict[str, Any], step_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "step": int(step),
        "split": "train_monitor",
        "scene_id": int(scheduler_info.get("scene_id", -1)),
        "scene_dir": _scene_dir_str(scheduler_info.get("scene_id", -1)),
        "segment_id": int(scheduler_info.get("segment_id", -1)),
        "global_step": int(scheduler_info.get("global_step", step)),
        "segment_local_step": int(scheduler_info.get("segment_local_step", -1)),
        "block_idx_in_segment": int(scheduler_info.get("block_idx_in_segment", -1)),
        "block_idx_in_episode": int(scheduler_info.get("block_idx_in_episode", -1)),
        "block_idx_global": int(scheduler_info.get("block_idx_global", -1)),
        "episode_idx_global": int(scheduler_info.get("episode_idx_global", -1)),
        "source_frame_idx": int(scheduler_info.get("source_frame_idx", -1)),
        "source_image_ref": list(scheduler_info.get("source_image_ref", (-1, -1))),
        "target_image_refs": [list(x) for x in scheduler_info.get("target_image_refs", [])],
        "U": int(scheduler_info.get("U", -1)),
        "K_u_effective": int(scheduler_info.get("K_u_effective", -1)),
        "K_steps_effective": int(scheduler_info.get("K_steps_effective", -1)),
        "events": [str(ev.get("type", "")) for ev in step_events],
    }
    for k, v in logs.items():
        if isinstance(v, bool):
            row[k] = bool(v)
        elif isinstance(v, (int, float)):
            row[k] = float(v)
    return row


def _setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    file_path = os.path.join(log_dir, "train.log")
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_path for h in root.handlers):
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        root.addHandler(fh)


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
    log_dir = str(cfg.get("log_dir", "outputs/skybranch"))
    _setup_logging(log_dir)
    logger.info("SkyBranch config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_multi_scene_dataset_v4(cfg, device)
    scheduler = build_train_scheduler_v8_from_cfg(cfg, dataset)
    trainer = MinimalSkyBranchTrainer(cfg, device=device)
    max_steps = int(args.max_steps or cfg.training.get("max_iterations", 60000))
    log_interval = int(cfg.training.get("log_interval", 100))
    log_on = str(cfg.training.get("log_on", "block_end")).strip()
    save_every = int(cfg.training.get("save_checkpoint_freq", 1000))
    cleanup_cfg = cfg.training.get("cleanup", {}) or {}
    empty_cache_after_step = bool(cleanup_cfg.get("empty_cache_after_step", False))
    logging_cfg = cfg.get("logging", {}) or {}
    enable_jsonl_metrics = bool(logging_cfg.get("enable_jsonl_metrics", True))
    metrics_history_append = bool(logging_cfg.get("metrics_history_append", True))
    use_tensorboard = bool(logging_cfg.get("use_tensorboard", True))
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    block_accum: Dict[str, List[float]] = defaultdict(list)
    metrics_fh = _open_metrics_history(log_dir, enable_jsonl_metrics, append=metrics_history_append)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb")) if use_tensorboard and SummaryWriter is not None else None
    try:
        for step in range(1, max_steps + 1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            raw_batch = scheduler.next_batch()
            scheduler_info = raw_batch.get("_scheduler_v8_aligned_info")
            if scheduler_info is None:
                raise ValueError("SkyBranch v8 training requires batch['_scheduler_v8_aligned_info'] from scheduler.next_batch().")
            scheduler_info = dict(scheduler_info or {})
            step_events = scheduler.pop_events() if hasattr(scheduler, "pop_events") else []
            _log_scheduler_events(step_events)
            scheduler_node_sync = _build_scheduler_node_sync_v8(cfg, scheduler_info, step_events)
            minimal = convert_batch_to_minimal_format(
                raw_batch,
                device=device,
                num_targets=int(raw_batch["target"]["image"].shape[0]),
                include_source_for_2d=True,
                view_selection=cfg.training.get("view_selection"),
            )
            logs = trainer.train_step(minimal, step=step, scheduler_node_sync=scheduler_node_sync)
            if torch.cuda.is_available():
                logs["cuda_alloc_gb"] = float(torch.cuda.memory_allocated() / (1024.0**3))
                logs["cuda_reserved_gb"] = float(torch.cuda.memory_reserved() / (1024.0**3))
                logs["cuda_peak_alloc_gb"] = float(torch.cuda.max_memory_allocated() / (1024.0**3))
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    block_accum[k].append(float(v))

            if _image_due(step, cfg, scheduler_info, step_events):
                _save_skybranch_debug_images(
                    step=step,
                    trainer=trainer,
                    minimal_batch=minimal,
                    scheduler_info=scheduler_info,
                    log_dir=log_dir,
                )

            has_block_end = any(ev.get("type") == "block_end" for ev in step_events)
            has_block_exit = any(ev.get("type") == "block_exit" for ev in step_events)
            has_episode_event = any(ev.get("type") == "episode_end" for ev in step_events)
            should_log = False
            if log_on == "step_interval":
                should_log = log_interval > 0 and step % log_interval == 0
            elif log_on == "episode_end":
                should_log = has_episode_event
            elif log_on == "block_end":
                should_log = has_block_end or has_episode_event
            elif log_on == "block_exit":
                should_log = has_block_exit or has_block_end or has_episode_event
            elif log_on == "both":
                should_log = (log_interval > 0 and step % log_interval == 0) or has_block_end or has_episode_event
            else:
                raise ValueError("training.log_on must be one of ['block_end', 'block_exit', 'episode_end', 'step_interval', 'both']")
            if should_log:
                log_payload = dict(logs)
                if block_accum:
                    for k, vals in block_accum.items():
                        if vals:
                            log_payload[f"{k}_mean_since_log"] = float(sum(vals) / len(vals))
                    block_accum.clear()
                logger.info(_format_log_message(step, log_payload, scheduler_info, step_events))
                row = _metrics_row(step, log_payload, scheduler_info, step_events)
                _write_metrics_history(metrics_fh, row)
                if writer is not None:
                    for k, v in log_payload.items():
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f"train/{k}", float(v), step)
                    writer.flush()
            if save_every > 0 and step % save_every == 0:
                trainer.save_checkpoint(os.path.join(ckpt_dir, f"skybranch_resume_step_{step:06d}.pth"), kind="resume")
                trainer.save_checkpoint(os.path.join(ckpt_dir, f"skybranch_model_step_{step:06d}.pth"), kind="model")
            if empty_cache_after_step and torch.cuda.is_available():
                torch.cuda.empty_cache()
        trainer.save_checkpoint(os.path.join(ckpt_dir, "skybranch_resume_final.pth"), kind="resume")
        trainer.save_checkpoint(os.path.join(ckpt_dir, "skybranch_model_final.pth"), kind="model")
    finally:
        if writer is not None:
            writer.close()
        if metrics_fh is not None:
            metrics_fh.close()


if __name__ == "__main__":
    main()
