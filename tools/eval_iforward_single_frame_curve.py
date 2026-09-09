from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from streetforward_eval.metrics import _masked_metrics, _to_hw_mask, _to_hwc_rgb
from tools.train_iforward import build_iforward_trainer_from_cfg
from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format
from tools.train_minimal_streetforward_stage4_3_v8_common import build_multi_scene_dataset_v4_for_demo


LOGGER = logging.getLogger("eval_iforward_single_frame_curve")
ImageRef = Tuple[int, int]
DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _parse_budgets(values: Sequence[int]) -> List[int]:
    budgets = [int(value) for value in values]
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("budgets must be a non-empty list of positive integers")
    if len(set(budgets)) != len(budgets):
        raise ValueError(f"budgets must be unique, got {budgets}")
    return budgets


def _checkpoint_step(payload: Mapping[str, Any]) -> int:
    for key in ("step", "global_step", "iteration", "iter"):
        if payload.get(key) is not None:
            return int(payload[key])
    return 0


def _load_checkpoint(model: Any, checkpoint_path: str) -> Tuple[Dict[str, Any], int]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a dict payload: {checkpoint_path}")
    state = payload.get("model_state_dict")
    if state is None:
        raise ValueError(f"checkpoint is missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    feedback_loader = getattr(model, "load_feedback_schedule_state_from_checkpoint", None)
    if callable(feedback_loader):
        feedback_loader(payload)
    runtime_loader = getattr(model, "load_runtime_state_from_checkpoint", None)
    if callable(runtime_loader):
        runtime_loader(payload)
    return payload, _checkpoint_step(payload)


def _make_iforward_metadata(
    *,
    scene_id: int,
    segment_id: int,
    frame_id: int,
    source_keyframe_idx: int,
    num_cams: int,
    budget: int,
    scheduler_version: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    refs: List[ImageRef] = [(int(frame_id), cam_id) for cam_id in range(int(num_cams))]
    denom = float(max(int(budget) - 1, 1))
    steps: List[Dict[str, Any]] = []
    for repeat_idx in range(int(budget)):
        is_enter = repeat_idx == 0
        is_exit = repeat_idx == int(budget) - 1
        steps.append(
            {
                "step_idx": int(repeat_idx),
                "block_id": 0,
                "episode_block_idx": 0,
                "rollout_block_rank": 0,
                "repeat_idx": int(repeat_idx),
                "repeats_per_block": int(budget),
                "is_block_enter": bool(is_enter),
                "is_block_exit": bool(is_exit),
                "is_frame_exit": bool(is_exit),
                "source_keyframe_idx": int(source_keyframe_idx),
                "source_frame_idx": int(frame_id),
                "evidence_refs": list(refs),
                "source_indices": list(range(int(num_cams))),
                "commit_observation_memory": bool(is_enter),
                "update_optimizer_memory": True,
                "detach_before_step": False,
                "detach_after_step": False,
                "allow_step_render_loss": False,
                "step_loss_refs": [],
                "rollout_pos_code": float(repeat_idx) / denom,
                "frame_pos_code": 0.0,
                "repeat_pos_code": float(repeat_idx) / denom,
                "episode_visit_idx": 0,
                "rollout_visit_idx": 0,
                "optimizer_step_idx_in_episode": int(repeat_idx),
                "record_update_norm": True,
                "commit_support_on_exit": bool(is_exit),
                "commit_residual_on_exit": bool(is_exit),
                "window_hash": 0,
                "window_revisit_count": 0,
                "block_visit_count_before": 0,
                "block_visit_count_after": 1,
            }
        )

    target_roles = ["final_current_recon"] * len(refs)
    final_supervision = {
        "refs": list(refs),
        "roles": list(target_roles),
        "current_input_frames": [int(frame_id)],
        "current_frames": [int(frame_id)],
        "current_refs": list(refs),
        "current_ref_count": len(refs),
    }
    ifwd = {
        "scheduler_version": str(scheduler_version),
        "model_family": "IForward",
        "scene_id": int(scene_id),
        "segment_id": int(segment_id),
        "episode_id": 0,
        "rollout_id_global": 0,
        "rollout_idx_in_episode": 0,
        "rollouts_per_episode": 1,
        "inner_K": int(budget),
        "blocks_per_rollout": 1,
        "repeats_per_block": int(budget),
        "steps": steps,
        "input_frame_indices": [int(frame_id)],
        "evidence_refs_flat": list(refs),
        "target_refs_flat": list(refs),
        "target_roles_flat": list(target_roles),
        "final_supervision": final_supervision,
        "reset_scene_state_before_rollout": True,
        "carry_scene_state_after_rollout": False,
        "episode_end_after_rollout": True,
        "detach_graph_after_rollout": True,
        "window_start": 0,
        "window_end": 1,
        "window_block_ids": [0],
        "window_hash": 0,
        "window_revisit_count": 0,
    }
    request_meta = {
        "assembly_mode": "image_ref_iforward_v1",
        "scheduler_version": str(scheduler_version),
        "model_family": "IForward",
        "loss_timing_policy": "rollout_final_only",
        "scene_id": int(scene_id),
        "segment_id": int(segment_id),
        "episode_id": 0,
        "rollout_id_global": 0,
        "rollout_idx_in_episode": 0,
        "rollouts_per_episode": 1,
        "inner_K": int(budget),
        "shape_name": f"r{int(budget)}b1_eval",
        "blocks_per_rollout": 1,
        "repeats_per_block": int(budget),
        "source_image_refs": list(refs),
        "target_image_refs": list(refs),
        "target_image_roles": list(target_roles),
        "final_supervision": final_supervision,
    }
    return ifwd, request_meta


def _materialize_batch(
    *,
    dataset: Any,
    device: torch.device,
    scene_id: int,
    segment_id: int,
    frame_id: int,
    budget: int,
    global_step: int,
    scheduler_version: str,
) -> Dict[str, Any]:
    sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
    if int(frame_id) not in set(int(value) for value in sidx.train_frame_set):
        raise ValueError(
            f"frame {frame_id} is not a train frame for scene={scene_id} segment={segment_id}; "
            f"available={list(sidx.frame_indices)}"
        )
    num_cams = int(sidx.num_cams)
    refs: List[ImageRef] = [(int(frame_id), cam_id) for cam_id in range(num_cams)]
    raw = dataset._assemble_segment_batch_from_image_refs(
        int(scene_id),
        int(segment_id),
        source_image_refs=refs,
        target_image_refs=refs,
        aux_image_refs=[],
        include_test=False,
        test_image_refs=None,
        enforce_target0_equals_source=True,
        target_ref_purpose="train",
    )
    ifwd, request_meta = _make_iforward_metadata(
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        frame_id=int(frame_id),
        source_keyframe_idx=int(sidx.frame_to_keyframe[int(frame_id)]),
        num_cams=num_cams,
        budget=int(budget),
        scheduler_version=str(scheduler_version),
    )
    raw["_iforward"] = ifwd
    raw["request_meta"] = request_meta
    minimal = convert_batch_to_minimal_format(
        raw,
        device,
        num_targets=len(refs),
        include_source_for_2d=True,
        view_selection=None,
    )
    minimal["global_step"] = int(global_step)
    return minimal


def _target_by_ref(batch: Mapping[str, Any]) -> Dict[ImageRef, Mapping[str, Any]]:
    result: Dict[ImageRef, Mapping[str, Any]] = {}
    for target in list(batch.get("targets") or []):
        ref = (int(target.get("frame_idx", -1)), int(target.get("cam_idx", -1)))
        result[ref] = target
    return result


def _pooled_psnr(items: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]) -> Tuple[float, int]:
    squared_error = 0.0
    valid_pixels = 0
    for pred, gt, sky_mask in items:
        valid = torch.ones(pred.shape[:2], device=pred.device, dtype=torch.bool)
        if sky_mask is not None:
            valid &= sky_mask <= 0.5
        count = int(valid.sum().item())
        squared_error += float(((pred - gt).pow(2) * valid.unsqueeze(-1)).sum().item())
        valid_pixels += count
    if valid_pixels < 1:
        return float("nan"), 0
    mse = squared_error / float(valid_pixels * 3)
    return float(-10.0 * math.log10(mse + 1.0e-12)), int(valid_pixels)


def _save_rgb(path: Path, tensor: torch.Tensor) -> None:
    image = _to_hwc_rgb(tensor).clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
    Image.fromarray(image).save(path)


def _evaluate_budget(
    *,
    model: Any,
    batch: Dict[str, Any],
    budget: int,
    image_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    reset = getattr(model, "reset_iforward_state_cache", None)
    if callable(reset):
        reset()
    reset_bridge = getattr(model, "_reset_bridge_runtime_node_state", None)
    if callable(reset_bridge):
        reset_bridge()
    with torch.inference_mode():
        output = model.forward_rollout(batch, carried_state=None)

    targets = _target_by_ref(batch)
    per_view: List[Dict[str, Any]] = []
    pooled_non_sky: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    pooled_full: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    for pred_raw, gt_raw, ref_raw, role in zip(
        output.pred_rgbs,
        output.gt_images,
        output.image_refs,
        output.image_roles,
    ):
        ref = (int(ref_raw[0]), int(ref_raw[1]))
        if str(role) not in {"final_current_recon", "current_latest"}:
            continue
        target = targets[ref]
        pred = _to_hwc_rgb(pred_raw).float().clamp(0.0, 1.0)
        gt = _to_hwc_rgb(gt_raw).float().clamp(0.0, 1.0)
        sky = _to_hw_mask(target.get("sky_mask"))
        ego = _to_hw_mask(target.get("egocar_mask"))
        if sky is not None:
            sky = sky.to(device=pred.device)
        if ego is not None:
            ego = ego.to(device=pred.device)
        metrics = _masked_metrics(
            pred=pred,
            gt=gt,
            sky_mask=sky,
            egocar_mask=ego,
            primary_mask="non_sky",
            min_valid_pixels=1,
        )
        per_view.append(
            {
                "optimization_steps": int(budget),
                "frame_id": int(ref[0]),
                "cam_id": int(ref[1]),
                "role": str(role),
                **metrics,
            }
        )
        pooled_non_sky.append((pred, gt, sky))
        pooled_full.append((pred, gt, None))
        _save_rgb(image_dir / f"steps_{int(budget):03d}_cam_{int(ref[1])}_pred.png", pred)
        if int(budget) == 1:
            _save_rgb(image_dir / f"cam_{int(ref[1])}_gt.png", gt)

    if not per_view:
        raise RuntimeError(
            f"budget={budget} produced no current-frame reconstruction images; "
            f"image_refs={list(output.image_refs)} image_roles={list(output.image_roles)} "
            f"pred_count={len(output.pred_rgbs)}"
        )
    pooled_psnr, valid_pixels = _pooled_psnr(pooled_non_sky)
    pooled_psnr_full, full_pixels = _pooled_psnr(pooled_full)
    mean_view_psnr = float(np.mean([float(row["psnr"]) for row in per_view]))
    mean_view_psnr_full = float(np.mean([float(row["psnr_full"]) for row in per_view]))
    summary = {
        "optimization_steps": int(budget),
        "psnr": mean_view_psnr,
        "psnr_full": mean_view_psnr_full,
        "pooled_psnr": float(pooled_psnr),
        "pooled_psnr_full": float(pooled_psnr_full),
        "valid_pixels": int(valid_pixels),
        "full_pixels": int(full_pixels),
        "num_views": len(per_view),
        "rollout_loss": float(output.loss.detach().float().item()),
    }
    return summary, per_view


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_curve_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; skipping PSNR curve plot")
        return
    steps = [int(row["optimization_steps"]) for row in rows]
    psnr = [float(row["psnr"]) for row in rows]
    psnr_full = [float(row["psnr_full"]) for row in rows]
    peak_idx = int(np.nanargmax(np.asarray(psnr)))
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(steps, psnr, marker="o", linewidth=2.0, label="non-sky PSNR")
    axis.plot(steps, psnr_full, marker="o", linewidth=1.4, label="full-image PSNR")
    axis.scatter([steps[peak_idx]], [psnr[peak_idx]], color="tab:red", zorder=4)
    axis.annotate(
        f"peak: {steps[peak_idx]} steps, {psnr[peak_idx]:.2f} dB",
        (steps[peak_idx], psnr[peak_idx]),
        xytext=(8, -20),
        textcoords="offset points",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(steps, labels=[str(value) for value in steps])
    axis.set_xlabel("RAFT-like optimization steps")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("IForward single-frame optimization curve")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an IForward checkpoint on one frame at repeat budgets 1..256")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--segment_id", type=int, default=0)
    parser.add_argument("--frame_id", type=int, default=34)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    budgets = _parse_budgets(args.budgets)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = OmegaConf.load(args.config_file)
    cfg.data.train_scene_ids = [int(args.scene_id)]
    cfg.data.eval_scene_ids = []
    device = torch.device(str(args.device))
    LOGGER.info("initializing dataset assets from %s", cfg.data.assets.root)
    dataset = build_multi_scene_dataset_v4_for_demo(cfg, device)
    dataset.initialize()

    LOGGER.info("building IForward model and loading %s", args.checkpoint)
    model = build_iforward_trainer_from_cfg(cfg, device)
    payload, checkpoint_step = _load_checkpoint(model, str(args.checkpoint))
    model.eval()
    scheduler_version = str(cfg.scheduler_iforward.version)

    summaries: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    for budget in budgets:
        batch = _materialize_batch(
            dataset=dataset,
            device=device,
            scene_id=int(args.scene_id),
            segment_id=int(args.segment_id),
            frame_id=int(args.frame_id),
            budget=int(budget),
            global_step=int(checkpoint_step),
            scheduler_version=scheduler_version,
        )
        summary, view_rows = _evaluate_budget(
            model=model,
            batch=batch,
            budget=int(budget),
            image_dir=image_dir,
        )
        summaries.append(summary)
        per_view_rows.extend(view_rows)
        LOGGER.info(
            "steps=%d mean_view_psnr=%.4f mean_view_psnr_full=%.4f pooled_psnr=%.4f",
            int(budget),
            float(summary["psnr"]),
            float(summary["psnr_full"]),
            float(summary["pooled_psnr"]),
        )

    _write_csv(output_dir / "psnr_by_optimization_steps.csv", summaries)
    _write_csv(output_dir / "metrics_per_view.csv", per_view_rows)
    _write_curve_plot(output_dir / "psnr_curve.png", summaries)
    manifest = {
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_iforward_version": str(payload.get("iforward_version", cfg.model.iforward.version)),
        "scheduler_version": scheduler_version,
        "asset_root": str(cfg.data.assets.root),
        "scene_id": int(args.scene_id),
        "segment_id": int(args.segment_id),
        "frame_id": int(args.frame_id),
        "camera_ids": list(range(int(dataset.get_segment_index(args.scene_id, args.segment_id).num_cams))),
        "budgets": budgets,
        "budget_semantics": "independent rollout from checkpoint state; one frame/block; N RAFT-like recurrent updates",
        "primary_metric": "arithmetic mean of per-camera RGB PSNR over non-sky pixels (matches StreetForward curve)",
        "seed": int(args.seed),
        "device": str(device),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote evaluation outputs to %s", output_dir)


if __name__ == "__main__":
    main()
