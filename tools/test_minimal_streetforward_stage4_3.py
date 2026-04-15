from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.multi_scene_dataset_v3 import EvalRequestV3
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3, RuntimePolicy
from tools.streetforward_test_config import (
    ensure_dataset_initialized_for_test,
    validate_dataset_test_split_or_raise,
    validate_test_config,
)
from tools.streetforward_test_export import save_3dgs_state, save_test_summary
from tools.train_minimal_streetforward_stage1_1 import _compute_metrics, _save_image_triplet, convert_batch_to_minimal_format, setup
from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _build_scheduler_node_sync, _load_init_checkpoint
from tools.train_minimal_streetforward_stage4_3_v4_common import build_multi_scene_dataset_v3, build_train_scheduler_v4_from_cfg
from utils.streetforward_baseline import set_deterministic_seed

logger = logging.getLogger(__name__)


def _scene_seg_dir(root: str, scene_id: int, segment_id: int, mode: str) -> str:
    d = os.path.join(root, "test", f"scene_{int(scene_id):03d}", f"segment_{int(segment_id):03d}", mode)
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "renders"), exist_ok=True)
    return d


def _metric_better(metric: str, new_val: float, old_val: Optional[float]) -> bool:
    if old_val is None:
        return True
    if metric == "lpips":
        return float(new_val) < float(old_val)
    return float(new_val) > float(old_val)


def _log_block_interval_train_psnr_and_images(
    *,
    step: int,
    block_idx: int,
    scene_id: int,
    segment_id: int,
    mode_label: str,
    result: Dict[str, Any],
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: SSIM,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    seg_dir: str,
    save_images: bool,
) -> None:
    """When logging.image_interval_blocks fires: log train-batch PSNR and optionally save pred/gt/error PNGs."""
    preds = result.get("pred_rgbs")
    gts = result.get("gt_images")
    if preds is None or gts is None:
        return
    pred_list = list(preds) if not isinstance(preds, list) else preds
    gt_list = list(gts) if not isinstance(gts, list) else gts
    if len(pred_list) == 0 or len(gt_list) == 0:
        return
    psnr_vals: List[float] = []
    for pred, gt in zip(pred_list, gt_list):
        vals = _compute_metrics(pred, gt, psnr_metric, ssim_metric, lpips_metric, True, False)
        psnr_vals.append(float(vals["psnr"]))
    mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    logger.info(
        "%s block_interval scene=%s segment=%s block=%s step=%s train_psnr_mean=%.4f train_psnr_per_view=%s",
        mode_label,
        scene_id,
        segment_id,
        block_idx,
        step,
        mean_psnr,
        [round(x, 4) for x in psnr_vals],
    )
    if save_images:
        out_dir = os.path.join(seg_dir, "renders", "block_interval")
        os.makedirs(out_dir, exist_ok=True)
        for idx, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            _save_image_triplet(
                step,
                pred,
                gt,
                out_dir,
                view_suffix=f"{mode_label}_b{block_idx}_v{idx}",
                save_error=True,
            )


def _build_minimal_eval_from_refs(
    dataset: Any,
    scene_id: int,
    segment_id: int,
    source_ref: Tuple[int, int],
    test_refs: List[Tuple[int, int]],
    device: torch.device,
) -> Dict[str, Any]:
    req = EvalRequestV3(
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        source_image_ref=tuple(source_ref),
        eval_image_refs=[tuple(r) for r in test_refs],
    )
    raw_eval = dataset.get_segment_eval_batch_from_image_refs(req)
    raw = dict(raw_eval)
    raw["target"] = raw_eval["eval"]
    return convert_batch_to_minimal_format(
        raw,
        device,
        num_targets=int(raw["target"]["image"].shape[0]),
        include_source_for_2d=True,
        view_selection=None,
    )


def _eval_on_test_refs(
    model: MinimalStreetForwardStage4_3,
    dataset: Any,
    scene_id: int,
    segment_id: int,
    source_ref: Tuple[int, int],
    test_refs: List[Tuple[int, int]],
    device: torch.device,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: SSIM,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    minimal_eval = _build_minimal_eval_from_refs(dataset, scene_id, segment_id, source_ref, test_refs, device)
    prev_mode = model.training
    model.eval()
    with torch.no_grad():
        out = model.forward(minimal_eval)
    if prev_mode:
        model.train()
    per_view: List[Dict[str, Any]] = []
    psnr_list: List[float] = []
    ssim_list: List[float] = []
    lpips_list: List[float] = []
    for i, (pred, gt, tgt) in enumerate(zip(out["pred_rgbs"], out["gt_images"], minimal_eval["targets"])):
        vals = _compute_metrics(pred, gt, psnr_metric, ssim_metric, lpips_metric, True, True)
        psnr_list.append(float(vals["psnr"]))
        ssim_list.append(float(vals["ssim"]))
        lpips_list.append(float(vals["lpips"]))
        per_view.append(
            {
                "index": int(i),
                "frame_idx": int(tgt["frame_idx"]),
                "psnr": float(vals["psnr"]),
                "ssim": float(vals["ssim"]),
                "lpips": float(vals["lpips"]),
            }
        )
    summary = {
        "psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "lpips": float(np.mean(lpips_list)) if lpips_list else 0.0,
        "num_views": int(len(psnr_list)),
    }
    return summary, per_view


def _override_cfg_for_fixed_segment(cfg: Any, scene_id: int, segment_id: int) -> Any:
    cfg_cp = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if cfg_cp.get("scheduler_v4") is None:
        raise ValueError("scheduler_v4 is required")
    if cfg_cp.scheduler_v4.get("traversal") is None:
        cfg_cp.scheduler_v4.traversal = {}
    cfg_cp.scheduler_v4.traversal.fixed_scene_id = int(scene_id)
    cfg_cp.scheduler_v4.traversal.fixed_segment_id = int(segment_id)
    if cfg_cp.get("multi_scene") is None:
        cfg_cp.multi_scene = {}
    cfg_cp.multi_scene.include_test = False
    return cfg_cp


def _source_ref_by_protocol(
    cfg: Any,
    scheduler_info: Dict[str, Any],
    minimal_batch: Dict[str, Any],
) -> Tuple[int, int]:
    source_ref = scheduler_info.get("source_image_ref")
    if source_ref is not None:
        return (int(source_ref[0]), int(source_ref[1]))
    # Fallback should be rare; keep deterministic.
    return (int(minimal_batch["source_frame_idx"]), 0)


def run_adapt_supervised(
    cfg: Any,
    dataset: Any,
    model: MinimalStreetForwardStage4_3,
    device: torch.device,
    test_cfg: Dict[str, Any],
) -> None:
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    log_interval_blocks = int(cfg.logging.get("image_interval_blocks", 1))
    if log_interval_blocks < 1:
        raise ValueError("logging.image_interval_blocks must be >= 1 for test runner logs")
    eval_scene_ids = [int(x) for x in test_cfg["eval_scene_ids"]]
    fixed_scene_id = cfg.test.runner.get("fixed_scene_id")
    fixed_segment_id = cfg.test.runner.get("fixed_segment_id")
    if fixed_scene_id is not None:
        eval_scene_ids = [int(fixed_scene_id)]

    for scene_id in eval_scene_ids:
        scene_data = dataset.get_scene(int(scene_id))
        if scene_data is None:
            logger.warning("Skip scene_id=%s because scene cannot be loaded", scene_id)
            continue
        seg_ids = list(range(len(scene_data["segments"])))
        if fixed_segment_id is not None:
            seg_ids = [int(fixed_segment_id)]
        max_seg = int(test_cfg["max_segments_per_scene"])
        used = 0
        for segment_id in seg_ids:
            test_refs = dataset.resolve_test_image_refs_deterministic(int(scene_id), int(segment_id))
            if len(test_refs) < int(test_cfg["min_test_views_per_segment"]):
                logger.info(
                    "Skip scene=%s segment=%s: test refs %s < min_test_views_per_segment %s",
                    scene_id,
                    segment_id,
                    len(test_refs),
                    int(test_cfg["min_test_views_per_segment"]),
                )
                continue
            if max_seg > 0 and used >= max_seg:
                break
            used += 1
            seg_dir = _scene_seg_dir(cfg.log_dir, scene_id, segment_id, "adapt_supervised")
            logger.info("Adapt-supervised test start scene=%s segment=%s", scene_id, segment_id)
            logger.info(
                "Adapt-supervised segment settings scene=%s segment=%s test_views=%s validate_every_blocks=%s max_steps=%s",
                scene_id,
                segment_id,
                len(test_refs),
                int(cfg.test.adapt_supervised.validate_every_blocks),
                int(cfg.test.adapt_supervised.max_steps_per_segment),
            )

            if bool(cfg.test.adapt_supervised.reset_runtime_state_each_segment):
                model.reset_node_state()

            seg_cfg = _override_cfg_for_fixed_segment(cfg, int(scene_id), int(segment_id))
            scheduler = build_train_scheduler_v4_from_cfg(seg_cfg, dataset)
            best_metric_name = str(cfg.test.adapt_supervised.keep_best_by)
            best_metric_val: Optional[float] = None
            best_eval_summary: Optional[Dict[str, Any]] = None
            best_eval_per_view: List[Dict[str, Any]] = []
            best_step = -1
            block_counter = 0
            validations_without_improve = 0
            init_saved = False
            final_minimal: Optional[Dict[str, Any]] = None

            for step in range(int(cfg.test.adapt_supervised.max_steps_per_segment)):
                raw_batch = scheduler.next_batch()
                scheduler_info = raw_batch.get("_scheduler_v4_aligned_info")
                if scheduler_info is None:
                    scheduler_info = scheduler.get_current_info()
                step_events = scheduler.pop_events()
                scheduler_node_sync = _build_scheduler_node_sync(seg_cfg, scheduler_info, step_events)
                minimal_batch = convert_batch_to_minimal_format(
                    raw_batch,
                    device,
                    num_targets=int(raw_batch["target"]["image"].shape[0]),
                    include_source_for_2d=True,
                    view_selection=None,
                )
                final_minimal = minimal_batch
                if not init_saved and bool(cfg.test.export.save_3dgs_init):
                    model.ensure_runtime_state_from_batch(minimal_batch)
                    init_state = model.export_3dgs_state(minimal_batch, rigid_export_frame_idx=int(minimal_batch["source_frame_idx"]))
                    save_3dgs_state(os.path.join(seg_dir, "3dgs_init.pt"), init_state)
                    init_saved = True
                result = model.train_step(
                    minimal_batch,
                    step=step,
                    profile_phase_timing=False,
                    sync_cuda_timing=False,
                    scheduler_node_sync=scheduler_node_sync,
                )

                stop_segment = False
                for ev in step_events:
                    if ev.get("type") != "block_end":
                        continue
                    block_counter += 1
                    if block_counter % log_interval_blocks == 0:
                        _log_block_interval_train_psnr_and_images(
                            step=step,
                            block_idx=block_counter,
                            scene_id=int(scene_id),
                            segment_id=int(segment_id),
                            mode_label="adapt",
                            result=result,
                            psnr_metric=psnr_metric,
                            ssim_metric=ssim_metric,
                            lpips_metric=lpips_metric,
                            seg_dir=seg_dir,
                            save_images=bool(cfg.test.export.save_rendered_images),
                        )
                        logger.info(
                            "Adapt block_interval scene=%s segment=%s block=%s step=%s source_ref=%s loss=%.6f",
                            scene_id,
                            segment_id,
                            block_counter,
                            step,
                            scheduler_info.get("source_image_ref"),
                            float(result.get("loss", 0.0)),
                        )
                    if block_counter % int(cfg.test.adapt_supervised.validate_every_blocks) != 0:
                        continue
                    source_ref = _source_ref_by_protocol(cfg, scheduler_info, minimal_batch)
                    eval_summary, eval_per_view = _eval_on_test_refs(
                        model,
                        dataset,
                        int(scene_id),
                        int(segment_id),
                        source_ref,  # type: ignore[arg-type]
                        [tuple(r) for r in test_refs],
                        device,
                        psnr_metric,
                        ssim_metric,
                        lpips_metric,
                    )
                    cur_metric = float(eval_summary[best_metric_name])
                    logger.info(
                        "Adapt eval scene=%s segment=%s step=%s block=%s source_ref=%s metric(psnr=%.4f ssim=%.4f lpips=%.4f)",
                        scene_id,
                        segment_id,
                        step,
                        block_counter,
                        source_ref,
                        float(eval_summary["psnr"]),
                        float(eval_summary["ssim"]),
                        float(eval_summary["lpips"]),
                    )
                    if _metric_better(best_metric_name, cur_metric, best_metric_val):
                        best_metric_val = cur_metric
                        best_eval_summary = dict(eval_summary)
                        best_eval_per_view = list(eval_per_view)
                        best_step = int(step)
                        validations_without_improve = 0
                        logger.info(
                            "Adapt best updated scene=%s segment=%s step=%s best_%s=%.6f",
                            scene_id,
                            segment_id,
                            best_step,
                            best_metric_name,
                            float(cur_metric),
                        )
                        if bool(cfg.test.export.save_3dgs_best):
                            best_state = model.export_3dgs_state(
                                minimal_batch, rigid_export_frame_idx=int(minimal_batch["source_frame_idx"])
                            )
                            save_3dgs_state(os.path.join(seg_dir, "3dgs_best.pt"), best_state)
                        if bool(cfg.test.export.save_rendered_images):
                            render_dir = os.path.join(seg_dir, "renders")
                            for idx, (pred, gt) in enumerate(zip(result["pred_rgbs"], result["gt_images"])):
                                _save_image_triplet(step, pred, gt, render_dir, view_suffix=f"adapt_train_v{idx}", save_error=True)
                    else:
                        validations_without_improve += 1
                    if int(cfg.test.adapt_supervised.early_stop_patience) > 0 and validations_without_improve >= int(
                        cfg.test.adapt_supervised.early_stop_patience
                    ):
                        logger.info(
                            "Early stop scene=%s segment=%s at step=%s (patience=%s)",
                            scene_id,
                            segment_id,
                            step,
                            int(cfg.test.adapt_supervised.early_stop_patience),
                        )
                        stop_segment = True
                        break
                if stop_segment:
                    break

            if final_minimal is None:
                continue
            final_state = model.export_3dgs_state(final_minimal, rigid_export_frame_idx=int(final_minimal["source_frame_idx"]))
            if bool(cfg.test.export.save_3dgs_final):
                save_3dgs_state(os.path.join(seg_dir, "3dgs_final.pt"), final_state)
            summary = {
                "mode": "adapt_supervised",
                "split": "test_adapt_supervised",
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "best_step": int(best_step),
                "best_metric": best_metric_name,
                "best_metric_value": float(best_metric_val) if best_metric_val is not None else None,
                "best_eval": best_eval_summary,
            }
            save_test_summary(os.path.join(seg_dir, "summary.json"), summary)
            logger.info(
                "Adapt-supervised segment done scene=%s segment=%s best_step=%s best_%s=%s",
                scene_id,
                segment_id,
                best_step,
                best_metric_name,
                "None" if best_metric_val is None else f"{float(best_metric_val):.6f}",
            )
            if bool(cfg.test.export.save_per_view_metrics_json):
                with open(os.path.join(seg_dir, "per_view_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(best_eval_per_view, f, indent=2)


def run_inference_only(
    cfg: Any,
    dataset: Any,
    model: MinimalStreetForwardStage4_3,
    device: torch.device,
    test_cfg: Dict[str, Any],
) -> None:
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    log_interval_blocks = int(cfg.logging.get("image_interval_blocks", 1))
    if log_interval_blocks < 1:
        raise ValueError("logging.image_interval_blocks must be >= 1 for test runner logs")
    eval_scene_ids = [int(x) for x in test_cfg["eval_scene_ids"]]
    fixed_scene_id = cfg.test.runner.get("fixed_scene_id")
    fixed_segment_id = cfg.test.runner.get("fixed_segment_id")
    if fixed_scene_id is not None:
        eval_scene_ids = [int(fixed_scene_id)]

    infer_policy = RuntimePolicy(
        do_backward=False,
        do_optimizer_step=False,
        update_hidden_cache=bool(cfg.test.inference_only.allow_hidden_cache_update),
        writeback_node_state=bool(cfg.test.inference_only.allow_node_state_writeback),
        reset_node_state_after_block=True,
    )
    if not bool(cfg.test.inference_only.allow_hidden_cache_update) and not bool(cfg.test.inference_only.allow_node_state_writeback):
        logger.warning(
            "inference_only is configured with allow_hidden_cache_update=false and "
            "allow_node_state_writeback=false; runtime state will barely change from checkpoint."
        )

    for scene_id in eval_scene_ids:
        scene_data = dataset.get_scene(int(scene_id))
        if scene_data is None:
            logger.warning("Skip scene_id=%s because scene cannot be loaded", scene_id)
            continue
        seg_ids = list(range(len(scene_data["segments"])))
        if fixed_segment_id is not None:
            seg_ids = [int(fixed_segment_id)]
        max_seg = int(test_cfg["max_segments_per_scene"])
        used = 0
        for segment_id in seg_ids:
            test_refs = dataset.resolve_test_image_refs_deterministic(int(scene_id), int(segment_id))
            if len(test_refs) < int(test_cfg["min_test_views_per_segment"]):
                logger.info(
                    "Skip scene=%s segment=%s: test refs %s < min_test_views_per_segment %s",
                    scene_id,
                    segment_id,
                    len(test_refs),
                    int(test_cfg["min_test_views_per_segment"]),
                )
                continue
            if max_seg > 0 and used >= max_seg:
                break
            used += 1
            seg_dir = _scene_seg_dir(cfg.log_dir, scene_id, segment_id, "inference_only")
            logger.info("Inference-only episode test start scene=%s segment=%s", scene_id, segment_id)
            logger.info(
                "Inference-only segment settings scene=%s segment=%s test_views=%s max_episodes_per_segment=%s "
                "allow_hidden_cache_update=%s allow_node_state_writeback=%s",
                scene_id,
                segment_id,
                len(test_refs),
                int(cfg.test.inference_only.max_episodes_per_segment),
                bool(cfg.test.inference_only.allow_hidden_cache_update),
                bool(cfg.test.inference_only.allow_node_state_writeback),
            )

            seg_cfg = _override_cfg_for_fixed_segment(cfg, int(scene_id), int(segment_id))
            scheduler = build_train_scheduler_v4_from_cfg(seg_cfg, dataset)
            max_episodes_per_segment = int(cfg.test.inference_only.max_episodes_per_segment)

            final_minimal: Optional[Dict[str, Any]] = None
            init_saved = False
            step = 0
            segment_done = False
            per_episode: List[Dict[str, Any]] = []
            per_episode_per_view: List[Dict[str, Any]] = []
            block_end_count = 0

            while not segment_done:
                raw_batch = scheduler.next_batch()
                scheduler_info = raw_batch.get("_scheduler_v4_aligned_info")
                if scheduler_info is None:
                    scheduler_info = scheduler.get_current_info()
                step_events = scheduler.pop_events()
                scheduler_node_sync = _build_scheduler_node_sync(seg_cfg, scheduler_info, step_events)
                minimal_batch = convert_batch_to_minimal_format(
                    raw_batch,
                    device,
                    num_targets=int(raw_batch["target"]["image"].shape[0]),
                    include_source_for_2d=True,
                    view_selection=None,
                )
                final_minimal = minimal_batch
                if not init_saved and bool(cfg.test.export.save_3dgs_init):
                    model.ensure_runtime_state_from_batch(minimal_batch)
                    init_state = model.export_3dgs_state(
                        minimal_batch,
                        rigid_export_frame_idx=int(minimal_batch["source_frame_idx"]),
                    )
                    save_3dgs_state(os.path.join(seg_dir, "3dgs_init.pt"), init_state)
                    init_saved = True

                infer_step = step
                infer_result = model.inference_step_from_train_batch(
                    minimal_batch,
                    step=infer_step,
                    scheduler_node_sync=scheduler_node_sync,
                    runtime_policy=infer_policy,
                )
                step += 1

                for ev in step_events:
                    if ev.get("type") == "block_end":
                        block_end_count += 1
                        if block_end_count % log_interval_blocks == 0:
                            _log_block_interval_train_psnr_and_images(
                                step=infer_step,
                                block_idx=block_end_count,
                                scene_id=int(scene_id),
                                segment_id=int(segment_id),
                                mode_label="infer",
                                result=infer_result,
                                psnr_metric=psnr_metric,
                                ssim_metric=ssim_metric,
                                lpips_metric=lpips_metric,
                                seg_dir=seg_dir,
                                save_images=bool(cfg.test.export.save_rendered_images),
                            )
                            logger.info(
                                "Inference block_interval scene=%s segment=%s block=%s step=%s source_ref=%s pseudo_loss=%.6f",
                                scene_id,
                                segment_id,
                                block_end_count,
                                infer_step,
                                scheduler_info.get("source_image_ref"),
                                float(infer_result.get("loss", 0.0)),
                            )
                    if ev.get("type") != "episode_end":
                        continue
                    source_ref = _source_ref_by_protocol(cfg, scheduler_info, minimal_batch)
                    ep_summary, ep_per_view = _eval_on_test_refs(
                        model,
                        dataset,
                        int(scene_id),
                        int(segment_id),
                        source_ref,  # type: ignore[arg-type]
                        [tuple(r) for r in test_refs],
                        device,
                        psnr_metric,
                        ssim_metric,
                        lpips_metric,
                    )
                    episode_idx = int(ev.get("reset_episode_idx", len(per_episode) + 1))
                    per_episode.append(
                        {
                            "episode_idx": episode_idx,
                            "source_ref": [int(source_ref[0]), int(source_ref[1])],
                            "num_views": int(ep_summary["num_views"]),
                            "psnr": float(ep_summary["psnr"]),
                            "ssim": float(ep_summary["ssim"]),
                            "lpips": float(ep_summary["lpips"]),
                        }
                    )
                    logger.info(
                        "Inference episode eval scene=%s segment=%s episode=%s source_ref=%s "
                        "psnr=%.4f ssim=%.4f lpips=%.4f num_views=%s",
                        scene_id,
                        segment_id,
                        episode_idx,
                        source_ref,
                        float(ep_summary["psnr"]),
                        float(ep_summary["ssim"]),
                        float(ep_summary["lpips"]),
                        int(ep_summary["num_views"]),
                    )
                    if bool(cfg.test.inference_only.save_per_episode_per_view_metrics_json):
                        for row in ep_per_view:
                            out_row = dict(row)
                            out_row["episode_idx"] = int(episode_idx)
                            per_episode_per_view.append(out_row)
                    if max_episodes_per_segment > 0 and len(per_episode) >= max_episodes_per_segment:
                        segment_done = True
                        break

                if segment_done:
                    break
                if any(ev.get("type") == "segment_end" for ev in step_events):
                    segment_done = True

            if final_minimal is None:
                continue
            if len(per_episode) == 0:
                raise ValueError(
                    f"inference_only requires at least one episode_end evaluation "
                    f"(scene={scene_id} segment={segment_id})"
                )

            summary = {
                "mode": "inference_only",
                "split": "test_inference_only",
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "aggregate_across_episodes": "mean",
                "num_episodes": int(len(per_episode)),
                "num_views_per_episode": int(per_episode[0]["num_views"]) if per_episode else 0,
                "psnr": float(np.mean([float(x["psnr"]) for x in per_episode])),
                "ssim": float(np.mean([float(x["ssim"]) for x in per_episode])),
                "lpips": float(np.mean([float(x["lpips"]) for x in per_episode])),
            }
            save_test_summary(os.path.join(seg_dir, "summary.json"), summary)
            logger.info(
                "Inference-only segment done scene=%s segment=%s episodes=%s block_ends=%s "
                "mean(psnr=%.4f ssim=%.4f lpips=%.4f)",
                scene_id,
                segment_id,
                int(len(per_episode)),
                int(block_end_count),
                float(summary["psnr"]),
                float(summary["ssim"]),
                float(summary["lpips"]),
            )
            if bool(cfg.test.inference_only.save_per_episode_metrics_json):
                with open(os.path.join(seg_dir, "per_episode_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(per_episode, f, indent=2)
            if bool(cfg.test.inference_only.save_per_episode_per_view_metrics_json):
                with open(os.path.join(seg_dir, "per_episode_per_view_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(per_episode_per_view, f, indent=2)

            final_state = model.export_3dgs_state(
                final_minimal,
                rigid_export_frame_idx=int(final_minimal["source_frame_idx"]),
            )
            if bool(cfg.test.export.save_3dgs_final):
                save_3dgs_state(os.path.join(seg_dir, "3dgs_final.pt"), final_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/minimal_streetforward_stage4_3_multi_scene_v4_test.yaml",
    )
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--init_weights_only", action="store_true")
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()

    cfg = setup(args)
    test_cfg = validate_test_config(cfg)
    if int(args.max_steps) > 0:
        cfg.test.adapt_supervised.max_steps_per_segment = int(args.max_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_deterministic_seed(int(args.seed))
    dataset = build_multi_scene_dataset_v3(cfg, device)
    ensure_dataset_initialized_for_test(dataset, cfg)
    validate_dataset_test_split_or_raise(dataset, test_cfg)

    model = MinimalStreetForwardStage4_3(config=cfg, device=device)
    model.train()
    _load_init_checkpoint(args.init_checkpoint, model, device, weights_only=bool(args.init_weights_only))

    mode = str(cfg.test.mode)
    if mode in ("adapt_supervised", "both") and bool(cfg.test.adapt_supervised.enable):
        run_adapt_supervised(cfg, dataset, model, device, test_cfg)
    if mode in ("inference_only", "both") and bool(cfg.test.inference_only.enable):
        both_cfg = cfg.test.get("both") or {}
        if mode == "both" and bool(both_cfg.get("reload_init_before_inference", True)):
            if not args.init_checkpoint:
                raise ValueError(
                    "mode=both with test.both.reload_init_before_inference=true requires --init_checkpoint."
                )
            model.reset_node_state()
            _load_init_checkpoint(
                args.init_checkpoint,
                model,
                device,
                weights_only=True,
            )
        run_inference_only(cfg, dataset, model, device, test_cfg)

    if hasattr(dataset, "shutdown_preload"):
        dataset.shutdown_preload()


if __name__ == "__main__":
    main()

