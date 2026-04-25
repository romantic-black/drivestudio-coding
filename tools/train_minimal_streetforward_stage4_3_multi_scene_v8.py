"""
Stage 4.3 multi-scene training entry for V4 dataset + V8 scheduler.

Thin wrapper over the stable multi-scene v4 training loop:
- swap dataset builder to MultiSceneDatasetV4
- swap scheduler builder to TrainSchedulerV8
- swap validation config/spec parser to validation_v8
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import torch
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from datasets.validation_scheduler_v8 import (
    ValidationEpisodeSpecV8,
    build_validation_episode_specs_v8,
)
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3
from tools.streetforward_validation_v8_config import ValidationV8Config, parse_validation_v8_config
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v8_from_cfg,
    resolve_fixed_scene_segment_v8,
)


def _parse_validation_v8_config_with_context(cfg: Any) -> ValidationV8Config:
    out = parse_validation_v8_config(cfg)
    base._validation_v8_cfg_runtime = out
    return out


def _build_validation_specs_v8_proxy(
    *,
    dataset: Any,
    eval_scene_ids: List[int],
    blocks_per_episode: int,
    total_target_frames: int,
) -> List[ValidationEpisodeSpecV8]:
    vcfg: Optional[ValidationV8Config] = getattr(base, "_validation_v8_cfg_runtime", None)
    steps_per_block = int(getattr(vcfg, "steps_per_block", 1))
    block_order = str(getattr(vcfg, "block_order", "block_major"))
    switch_steps = int(getattr(vcfg, "step_major_switch_interval_steps", 1))
    return build_validation_episode_specs_v8(
        dataset=dataset,
        eval_scene_ids=[int(x) for x in eval_scene_ids],
        blocks_per_episode=int(blocks_per_episode),
        total_target_frames=int(total_target_frames),
        steps_per_block=int(steps_per_block),
        block_order=str(block_order),
        step_major_switch_interval_steps=int(switch_steps),
    )


def _run_validation_v8_round(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    specs: List[ValidationEpisodeSpecV8],
    validation_cfg: ValidationV8Config,
    device: torch.device,
    trigger_train_episode_counter: int,
    trigger_step: int,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: SSIM,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    metrics_fh: Optional[TextIO],
    writer: Optional[Any] = None,
) -> None:
    if len(specs) == 0:
        base.logger.warning("validation_v8 enabled but no valid episode specs from eval_scene_ids")
        return
    base.logger.info(
        "VALIDATION_V8_BEGIN trigger_episode_counter=%s trigger_step=%s num_specs=%s",
        int(trigger_train_episode_counter),
        int(trigger_step),
        int(len(specs)),
    )
    val_root = os.path.join(cfg.log_dir, str(validation_cfg.save_dir))
    os.makedirs(val_root, exist_ok=True)
    validation_mode = str(validation_cfg.mode)
    use_train_finetune = validation_mode == "segment_finetune_train"
    infer_policy = base.RuntimePolicy(
        do_backward=False,
        do_optimizer_step=False,
        update_hidden_cache=True,
        writeback_node_state=True,
        reset_node_state_after_block=False,
    )
    train_policy = base.RuntimePolicy(
        do_backward=True,
        do_optimizer_step=True,
        update_hidden_cache=True,
        writeback_node_state=True,
        reset_node_state_after_block=False,
    )
    base_ckpt_bytes: Optional[bytes] = None
    if use_train_finetune:
        base_ckpt_bytes = base._snapshot_train_checkpoint_bytes(model)
    train_step_supports_runtime_policy = "runtime_policy" in inspect.signature(model.train_step).parameters
    infer_step_supports_runtime_policy = (
        "runtime_policy" in inspect.signature(model.inference_step_from_train_batch).parameters
    )

    def _collect_metrics_for_step(step_result: Dict[str, Any], minimal_batch: Dict[str, Any]) -> Dict[str, float]:
        preds = list(step_result.get("pred_rgbs") or [])
        gts = list(step_result.get("gt_images") or [])
        if len(preds) == 0 or len(preds) != len(gts):
            return {
                "num_views": float(min(len(preds), len(gts))),
                "psnr": 0.0,
                "ssim": 0.0,
                "lpips": 0.0,
                "psnr_full": 0.0,
                "ssim_full": 0.0,
                "lpips_full": 0.0,
                "psnr_non_sky": 0.0,
                "ssim_non_sky": 0.0,
                "lpips_non_sky": 0.0,
                "psnr_sky": 0.0,
                "ssim_sky": 0.0,
                "sky_mask_coverage": 0.0,
                "num_views_non_sky_metric": 0.0,
                "num_views_sky_metric": 0.0,
                "metric_scope": "full_image",
            }

        psnr_full_vals: List[float] = []
        ssim_full_vals: List[float] = []
        lpips_full_vals: List[float] = []
        psnr_non_sky_vals: List[float] = []
        ssim_non_sky_vals: List[float] = []
        lpips_non_sky_vals: List[float] = []
        psnr_sky_vals: List[float] = []
        ssim_sky_vals: List[float] = []
        sky_coverage_vals: List[float] = []
        targets = list(minimal_batch.get("targets") or [])
        min_valid = int(validation_cfg.min_valid_pixels_per_region)

        for idx, (pred, gt) in enumerate(zip(preds, gts)):
            m = base._compute_metrics(
                pred_rgb=pred,
                gt_rgb=gt,
                psnr_metric=psnr_metric,
                ssim_metric=ssim_metric,
                lpips_metric=lpips_metric,
                compute_psnr=True,
                compute_heavy=True,
            )
            psnr_full_vals.append(float(m["psnr"]))
            ssim_full_vals.append(float(m["ssim"]))
            lpips_full_vals.append(float(m["lpips"]))

            if not bool(validation_cfg.use_sky_mask_regions):
                continue
            tgt = targets[int(idx)] if int(idx) < len(targets) else {}
            sky_mask = tgt.get("sky_mask")
            if sky_mask is None and bool(validation_cfg.require_sky_mask):
                raise ValueError(
                    "validation_v8.metrics.require_sky_mask=true but target missing sky_mask "
                    f"(idx={int(idx)})"
                )
            if sky_mask is None:
                continue
            sm = sky_mask.to(device).float()
            if sm.dim() == 3:
                sm = sm.squeeze(-1)
            if sm.shape != gt.shape[:2]:
                raise ValueError(
                    "validation_v8 sky_mask shape mismatch: "
                    f"sky_mask={tuple(sm.shape)} gt_hw={tuple(gt.shape[:2])}"
                )
            non_sky = (1.0 - sm).clamp(0.0, 1.0)
            sky = sm.clamp(0.0, 1.0)
            non_sky_count = int((non_sky > 0.5).sum().item())
            sky_count = int((sky > 0.5).sum().item())
            sky_coverage_vals.append(float(sm.mean().item()))
            if non_sky_count >= min_valid:
                psnr_non = base._compute_masked_psnr(pred, gt, non_sky)
                ssim_non = base._compute_masked_ssim(pred, gt, non_sky)
                lpips_non = base._compute_masked_lpips(pred, gt, non_sky, lpips_metric)
                if psnr_non is not None:
                    psnr_non_sky_vals.append(float(psnr_non))
                if ssim_non is not None:
                    ssim_non_sky_vals.append(float(ssim_non))
                if lpips_non is not None:
                    lpips_non_sky_vals.append(float(lpips_non))
            if sky_count >= min_valid:
                psnr_sky = base._compute_masked_psnr(pred, gt, sky)
                ssim_sky = base._compute_masked_ssim(pred, gt, sky)
                if psnr_sky is not None:
                    psnr_sky_vals.append(float(psnr_sky))
                if ssim_sky is not None:
                    ssim_sky_vals.append(float(ssim_sky))

        out = {
            "num_views": float(len(preds)),
            "psnr_full": float(np.mean(psnr_full_vals)) if psnr_full_vals else 0.0,
            "ssim_full": float(np.mean(ssim_full_vals)) if ssim_full_vals else 0.0,
            "lpips_full": float(np.mean(lpips_full_vals)) if lpips_full_vals else 0.0,
            "psnr_non_sky": base._safe_mean(psnr_non_sky_vals),
            "ssim_non_sky": base._safe_mean(ssim_non_sky_vals),
            "lpips_non_sky": base._safe_mean(lpips_non_sky_vals),
            "psnr_sky": base._safe_mean(psnr_sky_vals),
            "ssim_sky": base._safe_mean(ssim_sky_vals),
            "sky_mask_coverage": base._safe_mean(sky_coverage_vals),
            "num_views_non_sky_metric": float(len(psnr_non_sky_vals)),
            "num_views_sky_metric": float(len(psnr_sky_vals)),
            "metric_scope": "full_image",
        }
        out["psnr"] = out["psnr_full"]
        out["ssim"] = out["ssim_full"]
        out["lpips"] = out["lpips_full"]
        if bool(validation_cfg.use_sky_mask_regions) and int(out["num_views_non_sky_metric"]) > 0:
            out["psnr"] = out["psnr_non_sky"]
            out["ssim"] = out["ssim_non_sky"]
            out["lpips"] = out["lpips_non_sky"]
            out["metric_scope"] = "non_sky"
        return out

    all_episode_rows: List[Dict[str, Any]] = []
    for spec in specs:
        if use_train_finetune:
            base._restore_train_checkpoint_bytes(model, base_ckpt_bytes, device)
        model.reset_node_state()
        validation_local_step = 0
        visit_rows: List[Dict[str, Any]] = []
        for visit_idx, (block_idx_in_episode, block_frames) in enumerate(
            zip(spec.block_visit_order, spec.visit_target_windows)
        ):
            src_frame = int(block_frames[0])
            source_ref = (int(src_frame), 0)
            source_refs = [(int(src_frame), int(cam_id)) for cam_id in range(int(spec.num_cams))]
            target_refs = [
                (int(frame_idx), int(cam_id))
                for frame_idx in block_frames
                for cam_id in range(int(spec.num_cams))
            ]
            req = base._BatchRequestValidationV7(
                scene_id=int(spec.scene_id),
                segment_id=int(spec.segment_id),
                source_image_ref=source_ref,
                source_image_refs=source_refs,
                target_image_refs=target_refs,
                include_test=False,
                test_image_refs=None,
            )
            raw_batch = dataset.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
            minimal_batch = base.convert_batch_to_minimal_format(
                raw_batch,
                device,
                num_targets=int(raw_batch["target"]["image"].shape[0]),
                include_source_for_2d=True,
                view_selection=None,
            )
            scheduler_node_sync = {
                "U": 1,
                "segment_local_step": int(validation_local_step + 1),
                "reset_after_block": False,
            }
            if use_train_finetune:
                kwargs: Dict[str, Any] = {
                    "batch": minimal_batch,
                    "step": None,
                    "profile_phase_timing": False,
                    "sync_cuda_timing": False,
                    "scheduler_node_sync": scheduler_node_sync,
                }
                if train_step_supports_runtime_policy:
                    kwargs["runtime_policy"] = train_policy
                step_result = model.train_step(**kwargs)
            else:
                kwargs = {
                    "batch": minimal_batch,
                    "step": None,
                    "scheduler_node_sync": scheduler_node_sync,
                }
                if infer_step_supports_runtime_policy:
                    kwargs["runtime_policy"] = infer_policy
                step_result = model.inference_step_from_train_batch(**kwargs)
            validation_local_step += 1
            visit_metrics = _collect_metrics_for_step(step_result, minimal_batch)
            visit_row = {
                "split": "validation_v8_visit",
                "mode": validation_mode,
                "block_order": str(validation_cfg.block_order),
                "trigger_train_episode_counter": int(trigger_train_episode_counter),
                "trigger_step": int(trigger_step),
                "scene_id": int(spec.scene_id),
                "segment_id": int(spec.segment_id),
                "episode_start_keyframe_pos": int(spec.episode_start_keyframe_pos),
                "visit_idx": int(visit_idx + 1),
                "num_visits_total": int(len(spec.block_visit_order)),
                "block_idx_in_episode": int(block_idx_in_episode),
                "source_frame": int(src_frame),
                "target_frames": [int(x) for x in block_frames],
                "loss": float(step_result.get("loss", 0.0)),
                **visit_metrics,
            }
            visit_rows.append(visit_row)
            if metrics_fh is not None:
                base._write_metrics_history(metrics_fh, visit_row)
            if writer is not None:
                tb_step = max(int(trigger_step), 0)
                writer.add_scalar("validation_v8/visit/loss", float(visit_row["loss"]), tb_step)
                writer.add_scalar("validation_v8/visit/psnr", float(visit_row["psnr"]), tb_step)
                writer.add_scalar("validation_v8/visit/ssim", float(visit_row["ssim"]), tb_step)
                writer.add_scalar("validation_v8/visit/lpips", float(visit_row["lpips"]), tb_step)
            base.logger.info(
                "VALIDATION_V8_BLOCK_VISIT mode=%s block_order=%s scene_id=%s segment_id=%s "
                "block=%s visit=%s/%s source_frame=%s target_frames=%s loss=%.6f",
                validation_mode,
                str(validation_cfg.block_order),
                int(spec.scene_id),
                int(spec.segment_id),
                int(block_idx_in_episode),
                int(visit_idx + 1),
                int(len(spec.block_visit_order)),
                int(src_frame),
                [int(x) for x in block_frames],
                float(step_result.get("loss", 0.0)),
            )
        if not visit_rows:
            continue

        episode_row = {
            "split": "validation_v8",
            "mode": validation_mode,
            "block_order": str(validation_cfg.block_order),
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "trigger_step": int(trigger_step),
            "scene_id": int(spec.scene_id),
            "segment_id": int(spec.segment_id),
            "episode_start_keyframe_pos": int(spec.episode_start_keyframe_pos),
            "num_visits": int(len(visit_rows)),
            "num_views": float(np.mean([float(r["num_views"]) for r in visit_rows])),
            "loss": float(np.mean([float(r["loss"]) for r in visit_rows])),
            "psnr": float(np.mean([float(r["psnr"]) for r in visit_rows])),
            "ssim": float(np.mean([float(r["ssim"]) for r in visit_rows])),
            "lpips": float(np.mean([float(r["lpips"]) for r in visit_rows])),
            "psnr_full": float(np.mean([float(r["psnr_full"]) for r in visit_rows])),
            "ssim_full": float(np.mean([float(r["ssim_full"]) for r in visit_rows])),
            "lpips_full": float(np.mean([float(r["lpips_full"]) for r in visit_rows])),
            "psnr_non_sky": float(np.mean([float(r["psnr_non_sky"]) for r in visit_rows])),
            "ssim_non_sky": float(np.mean([float(r["ssim_non_sky"]) for r in visit_rows])),
            "lpips_non_sky": float(np.mean([float(r["lpips_non_sky"]) for r in visit_rows])),
            "psnr_sky": float(np.mean([float(r["psnr_sky"]) for r in visit_rows])),
            "ssim_sky": float(np.mean([float(r["ssim_sky"]) for r in visit_rows])),
            "sky_mask_coverage": float(np.mean([float(r["sky_mask_coverage"]) for r in visit_rows])),
            "metric_scope": (
                "non_sky"
                if bool(validation_cfg.use_sky_mask_regions)
                and any(str(r.get("metric_scope", "")) == "non_sky" for r in visit_rows)
                else "full_image"
            ),
        }
        all_episode_rows.append(episode_row)
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, episode_row)
        if writer is not None:
            tb_step = max(int(trigger_step), 0)
            sid = int(spec.scene_id)
            seg = int(spec.segment_id)
            writer.add_scalar(
                f"validation_v8/episode/psnr/scene_{sid:03d}_segment_{seg:03d}",
                float(episode_row["psnr"]),
                tb_step,
            )
            writer.add_scalar(
                f"validation_v8/episode/ssim/scene_{sid:03d}_segment_{seg:03d}",
                float(episode_row["ssim"]),
                tb_step,
            )
            writer.add_scalar(
                f"validation_v8/episode/lpips/scene_{sid:03d}_segment_{seg:03d}",
                float(episode_row["lpips"]),
                tb_step,
            )
            writer.add_scalar(
                f"validation_v8/episode/loss/scene_{sid:03d}_segment_{seg:03d}",
                float(episode_row["loss"]),
                tb_step,
            )
        spec_dir = os.path.join(
            val_root,
            f"scene_{int(spec.scene_id):03d}",
            f"segment_{int(spec.segment_id):03d}",
            f"episode_start_{int(spec.episode_start_keyframe_pos):03d}",
        )
        os.makedirs(spec_dir, exist_ok=True)
        with open(os.path.join(spec_dir, "summary_v8.json"), "w", encoding="utf-8") as f:
            json.dump(episode_row, f, indent=2)

    if len(all_episode_rows) > 0:
        global_row = {
            "split": "validation_v8_global",
            "mode": validation_mode,
            "block_order": str(validation_cfg.block_order),
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "trigger_step": int(trigger_step),
            "num_episodes": int(len(all_episode_rows)),
            "loss": float(np.mean([float(r["loss"]) for r in all_episode_rows])),
            "psnr": float(np.mean([float(r["psnr"]) for r in all_episode_rows])),
            "ssim": float(np.mean([float(r["ssim"]) for r in all_episode_rows])),
            "lpips": float(np.mean([float(r["lpips"]) for r in all_episode_rows])),
            "psnr_full": float(np.mean([float(r["psnr_full"]) for r in all_episode_rows])),
            "ssim_full": float(np.mean([float(r["ssim_full"]) for r in all_episode_rows])),
            "lpips_full": float(np.mean([float(r["lpips_full"]) for r in all_episode_rows])),
            "psnr_non_sky": float(np.mean([float(r["psnr_non_sky"]) for r in all_episode_rows])),
            "ssim_non_sky": float(np.mean([float(r["ssim_non_sky"]) for r in all_episode_rows])),
            "lpips_non_sky": float(np.mean([float(r["lpips_non_sky"]) for r in all_episode_rows])),
            "psnr_sky": float(np.mean([float(r["psnr_sky"]) for r in all_episode_rows])),
            "ssim_sky": float(np.mean([float(r["ssim_sky"]) for r in all_episode_rows])),
            "sky_mask_coverage": float(np.mean([float(r["sky_mask_coverage"]) for r in all_episode_rows])),
            "metric_scope": (
                "non_sky"
                if bool(validation_cfg.use_sky_mask_regions)
                and any(str(r.get("metric_scope", "")) == "non_sky" for r in all_episode_rows)
                else "full_image"
            ),
        }
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, global_row)
        with open(
            os.path.join(val_root, f"summary_trigger_ep{int(trigger_train_episode_counter):06d}_v8.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(global_row, f, indent=2)
        if writer is not None:
            tb_step = max(int(trigger_step), 0)
            writer.add_scalar("validation_v8/global/loss", float(global_row["loss"]), tb_step)
            writer.add_scalar("validation_v8/global/psnr", float(global_row["psnr"]), tb_step)
            writer.add_scalar("validation_v8/global/ssim", float(global_row["ssim"]), tb_step)
            writer.add_scalar("validation_v8/global/lpips", float(global_row["lpips"]), tb_step)


def _setup_v8(args: Any) -> Any:
    cfg = _ORIG_SETUP(args)
    if cfg.get("scheduler_v8") is not None and cfg.get("scheduler_v7") is None:
        cfg["scheduler_v7"] = cfg.get("scheduler_v8")
    if cfg.get("validation_v8") is not None and cfg.get("validation_v7") is None:
        cfg["validation_v7"] = cfg.get("validation_v8")
    return cfg


_ORIG_SETUP = base.setup


def main() -> None:
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                "configs/minimal_streetforward_stage4_4_multi_scene_v8.yaml",
            ]
        )
    base.setup = _setup_v8
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_v8_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v8
    base.parse_validation_v7_config = _parse_validation_v8_config_with_context
    base.build_validation_episode_specs_v7 = _build_validation_specs_v8_proxy
    base._run_validation_v7_round = _run_validation_v8_round
    if (
        getattr(base, "TRAINER_CLASS", None) is None
        or getattr(base.TRAINER_CLASS, "__name__", "") == "MinimalStreetForwardStage4_3"
    ):
        base.TRAINER_CLASS = MinimalStreetForwardStage4_3
    if str(getattr(base, "CKPT_PREFIX", "")) == "minimal_sf_stage4_3_multi_scene_v4":
        base.CKPT_PREFIX = "minimal_sf_stage4_3_multi_scene_v8"
    if str(getattr(base, "DEFAULT_CONFIG_FILE", "")) == "configs/minimal_streetforward_stage4_4_multi_scene_v4.yaml":
        base.DEFAULT_CONFIG_FILE = "configs/minimal_streetforward_stage4_4_multi_scene_v8.yaml"
    base.main()


if __name__ == "__main__":
    main()
