"""
Preflight sweep for StreetForward: run forward-only checks across a number of batches to surface
volume/mask/target anomalies before committing to long training runs.

Uses minimal imports (no models.trainers) to avoid heavy deps (nerfstudio, etc.).
"""

import argparse
import json
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from datasets.multi_scene_dataset import MultiSceneDataset
from utils.logging import setup_logging
from utils.streetforward_baseline import convert_batch_to_streetforward_format


logger = logging.getLogger(__name__)


def _preflight_setup(args: argparse.Namespace):
    """Minimal config setup (avoids importing train_streetforward)."""
    cfg = OmegaConf.load(args.config_file)
    args_from_cli = OmegaConf.from_cli(args.opts) if args.opts else {}
    dataset_preset = None
    if "data" in args_from_cli and hasattr(args_from_cli.data, "dataset_preset"):
        dataset_preset = args_from_cli.data.dataset_preset
    if dataset_preset is None and "data" in cfg and hasattr(cfg.data, "get"):
        dataset_preset = cfg.data.get("dataset_preset")
    if dataset_preset is None and "data" in cfg and hasattr(cfg.data, "dataset"):
        dataset_preset = cfg.data.get("dataset")  # fallback
    dataset_cfg_key = dataset_preset
    if dataset_cfg_key:
        candidate = os.path.join("configs", "datasets", f"{dataset_cfg_key}.yaml")
        if os.path.exists(candidate):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(candidate))
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    return cfg


def _load_streetforward_trainer():
    """Load StreetForwardTrainer without models.trainers (avoids nerfstudio, etc.)."""
    import importlib.util
    import sys
    path = Path(__file__).resolve().parent.parent / "models" / "trainers" / "streetforward.py"
    spec = importlib.util.spec_from_file_location("models.trainers.streetforward", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    if "models.trainers" not in sys.modules:
        sys.modules["models.trainers"] = ModuleType("models.trainers")
    sys.modules["models.trainers.streetforward"] = mod
    spec.loader.exec_module(mod)
    return mod.StreetForwardTrainer


def _to_int(val: Any) -> int:
    if isinstance(val, torch.Tensor):
        return int(val.item())
    return int(val)


def _summarize_batch(
    trainer,
    sf_batch: Dict,
    max_dense_elements: int = None,
) -> Dict[str, Any]:
    device = trainer.device
    sf_batch["scene_id"] = _to_int(sf_batch["scene_id"][0] if isinstance(sf_batch["scene_id"], torch.Tensor) else sf_batch["scene_id"])
    sf_batch["segment_id"] = _to_int(sf_batch["segment_id"][0] if isinstance(sf_batch["segment_id"], torch.Tensor) else sf_batch["segment_id"])
    source_frame_idx = _to_int(sf_batch.get("source_frame_idx", 0))

    trainer._reset_sentinel_cache()
    trainer._update_runtime_flags()

    key, node_state_bg, node_state_rigid, node_state_distant = trainer._get_or_init_node_states(sf_batch)
    targets = trainer._parse_targets(sf_batch)
    masks = trainer._precompute_rigid_masks(node_state_rigid, source_frame_idx, targets)

    with torch.no_grad():
        try:
            trainer._build_3d_feature_volume(
                node_state_bg=node_state_bg,
                node_state_rigid=node_state_rigid,
                source_frame_idx=source_frame_idx,
                mask_src_rigid=masks.mask_src_rigid,
                idx_src_rigid=masks.idx_src_rigid,
            )
        except Exception as exc:
            return {
                "scene_id": key[0],
                "segment_id": key[1],
                "error": str(exc),
            }

    record: Dict[str, Any] = {
        "scene_id": key[0],
        "segment_id": key[1],
        "num_targets": len(targets),
        "N_bg": int(node_state_bg.means.shape[0]),
        "N_rigid": int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
        "N_distant": int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0,
        "mask_update_rigid_mean": float(masks.mask_update_rigid.float().mean().item()) if masks.mask_update_rigid is not None and masks.mask_update_rigid.numel() > 0 else None,
        "idx_tgt_rigid_counts": [int(idx.numel()) for idx in masks.idx_tgt_rigid] if masks.idx_tgt_rigid else [],
        "vol_dim_prod": trainer._last_vol_dim_prod,
        "dense_elements_est": trainer._last_dense_elements_est,
    }

    alerts: List[str] = []
    if record["num_targets"] == 0:
        alerts.append("no_targets")
    if record["mask_update_rigid_mean"] is not None and record["mask_update_rigid_mean"] == 0.0 and record["N_rigid"] > 0:
        alerts.append("rigid_gate_zero")
    # Only compare dense_elements_est to max_dense_elements (vol_dim_prod and dense_elements have different scales)
    max_dense = max_dense_elements or getattr(trainer, "sentinel_max_dense_elements", None)
    if max_dense is not None and record["dense_elements_est"] is not None and record["dense_elements_est"] > max_dense:
        alerts.append(f"dense_elements_gt_{max_dense}")
    record["alerts"] = alerts
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StreetForward preflight sweep (forward-only sanity checks).")
    parser.add_argument("--config_file", type=str, required=True, help="Path to StreetForward config file.")
    parser.add_argument("--output_root", type=str, default="./logs", help="Output root for config/logs (shared with train).")
    parser.add_argument("--project", type=str, default="streetforward", help="Project name for output directory.")
    parser.add_argument("--run_name", type=str, default="preflight", help="Run name; defaults to 'preflight'.")
    parser.add_argument("--max_batches", type=int, default=128, help="Number of batches to sweep.")
    parser.add_argument("--max_dense_elements", type=int, default=None, help="Hard limit for dense volume elements; overrides config.")
    parser.add_argument("--export_path", type=str, default=None, help="Optional JSON export path. Defaults to <log_dir>/preflight_report.json.")
    parser.add_argument("--log_interval", type=int, default=10, help="How often to log progress.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Override config options, same as train_streetforward.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    setup_logging(output=None, level=logging.INFO)
    cfg = _preflight_setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[preflight] device={device}")

    StreetForwardTrainer = _load_streetforward_trainer()
    dataset = MultiSceneDataset(
        data_cfg=cfg.data,
        train_scene_ids=cfg.data.train_scene_ids,
        eval_scene_ids=cfg.data.eval_scene_ids,
        num_source_keyframes=cfg.dataset.num_source_keyframes,
        num_target_keyframes=cfg.dataset.num_target_keyframes,
        segment_overlap_ratio=cfg.dataset.segment_overlap_ratio,
        keyframe_split_config=cfg.dataset.get("keyframe_split_config", None),
        min_keyframes_per_scene=cfg.dataset.min_keyframes_per_scene,
        min_keyframes_per_segment=cfg.dataset.min_keyframes_per_segment,
        device=device,
        preload_scene_count=cfg.dataset.get("preload_scene_count", 3),
        segment_aabb=cfg.dataset.segment_aabb,
        pointcloud_config=cfg.dataset.get("pointcloud", None),
    )

    trainer = StreetForwardTrainer(config=cfg, device=device)
    trainer.eval()
    if args.max_dense_elements is not None:
        trainer.sentinel_max_dense_elements = int(args.max_dense_elements)
    max_batches = int(args.max_batches)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

    records: List[Dict[str, Any]] = []
    for step in range(max_batches):
        batch = dataset.sample_random_batch()
        sf_batch = convert_batch_to_streetforward_format(batch, device)
        record = _summarize_batch(trainer, sf_batch, max_dense_elements=args.max_dense_elements)
        records.append(record)
        if step % args.log_interval == 0:
            logger.info(
                f"[preflight] step {step}/{max_batches} "
                f"N_bg={record.get('N_bg')} N_rigid={record.get('N_rigid')} "
                f"vol={record.get('vol_dim_prod')} alerts={record.get('alerts')}"
            )

    summary = {
        "max_batches": max_batches,
        "num_alert_batches": sum(1 for r in records if r.get("alerts")),
        "num_error_batches": sum(1 for r in records if r.get("error")),
        "alerts": [r for r in records if r.get("alerts")],
        "errors": [r for r in records if r.get("error")],
        "max_vol_dim_prod": max((r.get("vol_dim_prod") or 0) for r in records),
        "max_dense_elements_est": max((r.get("dense_elements_est") or 0) for r in records),
    }

    export_path = args.export_path or os.path.join(cfg.log_dir, "preflight_report.json")
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)

    logger.info(f"[preflight] completed {max_batches} batches; report saved to {export_path}")
    if summary["num_alert_batches"] > 0:
        logger.warning(f"[preflight] found {summary['num_alert_batches']} batches with alerts.")


if __name__ == "__main__":
    main(parse_args())
