from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch

from tools.train_minimal_streetforward_stage1_1 import (
    _save_image_triplet,
    _write_metrics_history,
    convert_batch_to_minimal_format,
)
from tools.train_minimal_streetforward_stage4_3_v7_common import (
    parse_include_test,
    validate_train_scene_for_fixed,
)
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    resolve_fixed_scene_segment_v8,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BatchRequest:
    scene_id: int
    segment_id: int
    source_image_ref: Tuple[int, int]
    target_image_refs: List[Tuple[int, int]]
    source_image_refs: Optional[List[Tuple[int, int]]] = None
    include_test: bool = False
    test_image_refs: Optional[List[Tuple[int, int]]] = None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        out = cfg.get(key, default)
        return default if out is None else out
    return getattr(cfg, key, default)


def resolve_fixed_scene_segment_v8_one_segment(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    one_segment = _cfg_get(cfg, "one_segment", None)
    if one_segment is not None:
        scene_id = _cfg_get(one_segment, "scene_id", None)
        segment_id = _cfg_get(one_segment, "segment_id", None)
        if scene_id is not None or segment_id is not None:
            if scene_id is None or segment_id is None:
                raise ValueError("one_segment must define both scene_id and segment_id")
            return int(scene_id), int(segment_id)
    return resolve_fixed_scene_segment_v8(cfg)


def build_train_scheduler_v8_one_segment_overfit_from_cfg(cfg: Any, dataset: Any) -> Any:
    sv8 = _cfg_get(cfg, "scheduler_v8", None)
    if sv8 is None:
        raise ValueError("config must define scheduler_v8")
    if _cfg_get(sv8, "enable") is not True:
        raise ValueError("scheduler_v8.enable must be true")
    block = _cfg_get(sv8, "block", None)
    ep = _cfg_get(sv8, "episode", None)
    trav = _cfg_get(sv8, "traversal", None)
    preload = _cfg_get(sv8, "preload", None)
    execution = _cfg_get(sv8, "execution", {}) or {}
    if block is None or ep is None or trav is None or preload is None:
        raise ValueError("scheduler_v8 must define block/episode/traversal/preload")

    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_v8_one_segment(cfg)
    if fixed_scene_id is None or fixed_segment_id is None:
        raise ValueError("one-segment overfit requires a fixed scene_id and segment_id")
    validate_train_scene_for_fixed(cfg, fixed_scene_id)

    sidx = dataset.get_segment_index(int(fixed_scene_id), int(fixed_segment_id))
    frame_indices = [int(x) for x in list(getattr(sidx, "frame_indices", []))]
    if len(frame_indices) == 0:
        raise ValueError(
            f"fixed segment has no train frame_indices: scene={fixed_scene_id} segment={fixed_segment_id}"
        )

    block_order = str(_cfg_get(execution, "block_order", "block_major"))
    if block_order not in ("block_major", "step_major"):
        raise ValueError("scheduler_v8.execution.block_order must be one of ['block_major', 'step_major']")
    step_major_switch_interval_steps = int(_cfg_get(execution, "step_major_switch_interval_steps", 1))
    if step_major_switch_interval_steps < 1:
        raise ValueError("scheduler_v8.execution.step_major_switch_interval_steps must be >= 1")
    reset_policy = str(_cfg_get(execution, "reset_policy", "episode_end"))
    target_policy = str(_cfg_get(ep, "target_policy", "visited_episode_frames"))
    history_target_policy = str(_cfg_get(ep, "history_target_policy", "random_visited"))
    block_source_frame_policy = str(_cfg_get(ep, "block_source_frame_policy", "fixed_once_per_episode"))
    total_target_frames = int(_cfg_get(ep, "total_target_frames", min(3, len(frame_indices))))
    if total_target_frames < 1:
        raise ValueError("scheduler_v8.episode.total_target_frames must be >= 1")
    if total_target_frames > len(frame_indices):
        raise ValueError(
            "scheduler_v8.episode.total_target_frames must be <= fixed segment frame count "
            f"({total_target_frames} > {len(frame_indices)})"
        )

    return dataset.create_train_scheduler_v8(
        steps_per_block=int(_cfg_get(block, "steps_per_block")),
        blocks_per_episode=int(len(frame_indices)),
        total_target_frames=int(total_target_frames),
        include_source_frame=bool(_cfg_get(ep, "include_source_frame", True)),
        frame_within_keyframe_policy=str(_cfg_get(ep, "frame_within_keyframe_policy", "middle_frame")),
        min_keyframes_required_policy=str(_cfg_get(ep, "min_keyframes_required_policy", "skip_if_less_than_window")),
        traversal_mode=str(_cfg_get(trav, "mode", "linear_scene_segment")),
        switch_after_episode=bool(_cfg_get(trav, "switch_after_episode", True)),
        segment_order=str(_cfg_get(trav, "segment_order", "ascending")),
        scene_order=str(_cfg_get(trav, "scene_order", "ascending")),
        include_test=parse_include_test(cfg),
        fixed_scene_id=int(fixed_scene_id),
        fixed_segment_id=int(fixed_segment_id),
        emit_preload_hints=bool(_cfg_get(preload, "emit_hints", False)),
        warm_next_block_exact=bool(_cfg_get(preload, "warm_next_block_exact", False)),
        warm_next_episode_chain=bool(_cfg_get(preload, "warm_next_episode_chain", False)),
        block_order=block_order,
        step_major_switch_interval_steps=int(step_major_switch_interval_steps),
        target_policy=target_policy,
        history_target_policy=history_target_policy,
        reset_policy=reset_policy,
        near_random_supervision_cfg=_cfg_get(sv8, "near_random_supervision", {}) or {},
        aux_feature_splat_targets_cfg=_cfg_get(sv8, "aux_feature_splat_targets", {}) or {},
        block_source_frame_policy=block_source_frame_policy,
        episode_source_mode="segment_frames",
    )


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_c = torch.clamp(pred.detach().float(), 0.0, 1.0)
    gt_c = torch.clamp(gt.detach().float(), 0.0, 1.0)
    mse = float(torch.mean((pred_c - gt_c) ** 2).item())
    if mse <= 0.0:
        return float("inf")
    return float(-10.0 * np.log10(max(mse, 1.0e-12)))


def _masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask_hw: torch.Tensor) -> Optional[float]:
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"masked PSNR shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
    if mask_hw.shape != pred.shape[:2]:
        raise ValueError(
            f"masked PSNR mask shape mismatch: mask={tuple(mask_hw.shape)} pred_hw={tuple(pred.shape[:2])}"
        )
    mask = mask_hw.to(pred.device).float().clamp(0.0, 1.0)
    if float(mask.sum().item()) <= 0.0:
        return None
    pred_c = torch.clamp(pred.detach().float(), 0.0, 1.0)
    gt_c = torch.clamp(gt.detach().float(), 0.0, 1.0)
    diff2 = ((pred_c - gt_c) ** 2) * mask.unsqueeze(-1)
    mse = float((diff2.sum() / (mask.sum() * 3.0)).item())
    if mse <= 0.0:
        return float("inf")
    return float(-10.0 * np.log10(max(mse, 1.0e-12)))


def _node_state_to_cpu_dict(state: Any) -> Dict[str, torch.Tensor]:
    keys = ("means", "scales_log", "quats", "opacity_logit", "sh_dc", "sh_rest")
    return {k: getattr(state, k).detach().cpu() for k in keys if hasattr(state, k)}


def _rigid_state_to_cpu_dict(state: Any) -> Dict[str, Any]:
    out = _node_state_to_cpu_dict(state)
    for k in ("point_ids", "instances_quats", "instances_trans", "instances_fv"):
        if hasattr(state, k):
            out[k] = getattr(state, k).detach().cpu()
    if hasattr(state, "instance_ids"):
        out["instance_ids"] = list(getattr(state, "instance_ids"))
    if hasattr(state, "frame_ids"):
        out["frame_ids"] = list(getattr(state, "frame_ids"))
    if hasattr(state, "cur_frame"):
        out["cur_frame"] = int(getattr(state, "cur_frame"))
    return out


def _runtime_node_state_payload(model: Any, scene_id: int, segment_id: int) -> Dict[str, Any]:
    key = (int(scene_id), int(segment_id))
    label = f"scene_{int(scene_id)}_segment_{int(segment_id)}"
    payload: Dict[str, Any] = {
        "node_states": {},
        "node_states_distant": {},
        "node_states_rigid": {},
        "h_cache_bg": {},
        "h_cache_distant": {},
        "h_cache_rigid": {},
    }
    for attr, out_key, packer in (
        ("node_states_bg", "node_states", _node_state_to_cpu_dict),
        ("node_states", "node_states", _node_state_to_cpu_dict),
        ("node_states_distant", "node_states_distant", _node_state_to_cpu_dict),
        ("node_states_rigid", "node_states_rigid", _rigid_state_to_cpu_dict),
    ):
        states = getattr(model, attr, None)
        if not isinstance(states, dict) or key not in states or states[key] is None:
            continue
        payload[out_key][label] = packer(states[key])
    for attr, out_key in (
        ("h_cache_bg", "h_cache_bg"),
        ("h_cache_distant", "h_cache_distant"),
        ("h_cache_rigid", "h_cache_rigid"),
    ):
        cache = getattr(model, attr, None)
        if isinstance(cache, dict) and key in cache and torch.is_tensor(cache[key]):
            payload[out_key][label] = cache[key].detach().cpu()
    return payload


class OverfitSegmentEpisodeEvaluator:
    def __init__(self) -> None:
        self.best_psnr = float("-inf")

    def __call__(
        self,
        *,
        cfg: Any,
        dataset: Any,
        model: Any,
        device: torch.device,
        trigger_train_episode_counter: int,
        trigger_step: int,
        minimal_batch: Dict[str, Any],
        scheduler_info: Dict[str, Any],
        step_events: List[Dict[str, Any]],
        psnr_metric: Any,
        ssim_metric: Any,
        lpips_metric: Any,
        metrics_fh: Optional[TextIO],
        writer: Optional[Any],
    ) -> None:
        _ = (step_events, psnr_metric, ssim_metric, lpips_metric)
        raw = _cfg_get(cfg, "overfit_segment_eval", {}) or {}
        if not bool(_cfg_get(raw, "enable", False)):
            return
        trigger = _cfg_get(raw, "trigger", {}) or {}
        every = int(_cfg_get(trigger, "validate_every_n_episodes", 1))
        if every < 1:
            raise ValueError("overfit_segment_eval.trigger.validate_every_n_episodes must be >= 1")
        if int(trigger_train_episode_counter) % every != 0:
            return

        scene_id = int(scheduler_info.get("scene_id", minimal_batch.get("scene_id", -1)))
        segment_id = int(scheduler_info.get("segment_id", minimal_batch.get("segment_id", -1)))
        if scene_id < 0 or segment_id < 0:
            raise ValueError("overfit_segment_eval cannot resolve current scene_id/segment_id")
        sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
        eval_refs = [(int(x[0]), int(x[1])) for x in list(getattr(sidx, "train_image_refs", ())) if len(x) >= 2]
        if len(eval_refs) == 0:
            eval_refs = [
                (int(frame_idx), int(cam_idx))
                for frame_idx in list(getattr(sidx, "frame_indices", []))
                for cam_idx in range(int(sidx.num_cams))
            ]
        if len(eval_refs) == 0:
            raise ValueError("overfit_segment_eval found no segment train images")

        render_cfg = _cfg_get(raw, "render", {}) or {}
        save_images = bool(_cfg_get(render_cfg, "save_images", False))
        save_dir = str(_cfg_get(render_cfg, "save_dir", "validation/overfit_segment"))
        chunk_size = int(_cfg_get(render_cfg, "chunk_size_images", 0))
        if chunk_size <= 0:
            chunk_size = len(eval_refs)
        if not save_dir.strip():
            raise ValueError("overfit_segment_eval.render.save_dir must be non-empty")
        metrics_cfg = _cfg_get(raw, "metrics", {}) or {}
        metric_scope = str(_cfg_get(metrics_cfg, "scope", "auto")).strip().lower()
        if metric_scope == "auto":
            losses_cfg = _cfg_get(cfg, "losses", {}) or {}
            photometric_cfg = _cfg_get(losses_cfg, "photometric", {}) or {}
            metric_scope = "non_sky" if bool(_cfg_get(photometric_cfg, "exclude_sky_region", False)) else "full_image"
        if metric_scope not in ("full_image", "non_sky"):
            raise ValueError("overfit_segment_eval.metrics.scope must be one of ['auto', 'full_image', 'non_sky']")
        min_valid_pixels = int(_cfg_get(metrics_cfg, "min_valid_pixels_per_region", 32))
        if min_valid_pixels < 1:
            raise ValueError("overfit_segment_eval.metrics.min_valid_pixels_per_region must be >= 1")
        require_sky_mask = bool(_cfg_get(metrics_cfg, "require_sky_mask", metric_scope == "non_sky"))
        root = os.path.join(str(cfg.log_dir), save_dir)
        out_dir = os.path.join(
            root,
            f"scene_{int(scene_id):03d}",
            f"segment_{int(segment_id):03d}",
            f"episode_{int(trigger_train_episode_counter):06d}",
        )
        if save_images:
            os.makedirs(os.path.join(out_dir, "renders"), exist_ok=True)
        else:
            os.makedirs(out_dir, exist_ok=True)

        source_frame = int(minimal_batch.get("source_frame_idx", eval_refs[0][0]))
        source_refs = [(int(source_frame), int(cam_idx)) for cam_idx in range(int(sidx.num_cams))]
        source_ref = (int(source_frame), 0)
        rigid_ref = int(minimal_batch.get("source_frame_idx", source_frame))
        gs_state = model.export_3dgs_state(
            minimal_batch,
            include_hidden=True,
            rigid_export_frame_idx=int(rigid_ref),
        )

        per_view_rows: List[Dict[str, Any]] = []
        psnr_vals: List[float] = []
        psnr_full_vals: List[float] = []
        psnr_non_sky_vals: List[float] = []
        sky_coverage_vals: List[float] = []
        prev_mode = model.training
        try:
            model.eval()
            for chunk_start in range(0, len(eval_refs), chunk_size):
                chunk_refs = eval_refs[chunk_start : chunk_start + chunk_size]
                chunk_ref_set = {tuple(x) for x in chunk_refs}
                missing_source_refs = [tuple(x) for x in source_refs if tuple(x) not in chunk_ref_set]
                render_refs = [tuple(x) for x in missing_source_refs] + [tuple(x) for x in chunk_refs]
                req = _BatchRequest(
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    source_image_ref=source_ref,
                    source_image_refs=source_refs,
                    target_image_refs=[tuple(x) for x in render_refs],
                    include_test=False,
                    test_image_refs=None,
                )
                raw_eval = dataset.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=False)
                minimal_eval = convert_batch_to_minimal_format(
                    raw_eval,
                    device,
                    num_targets=int(raw_eval["target"]["image"].shape[0]),
                    include_source_for_2d=True,
                    view_selection=None,
                )
                preds = model.render_views_from_scene_state(
                    {"base_batch": minimal_eval, "gs_state": gs_state},
                    list(minimal_eval.get("targets") or []),
                )
                targets = list(minimal_eval.get("targets") or [])
                if len(preds) != len(targets):
                    raise ValueError(
                        f"overfit_segment_eval render size mismatch: pred={len(preds)} targets={len(targets)}"
                    )
                metric_offset = int(len(missing_source_refs))
                metric_preds = preds[metric_offset:]
                metric_targets = targets[metric_offset:]
                if len(metric_preds) != len(chunk_refs) or len(metric_targets) != len(chunk_refs):
                    raise ValueError(
                        "overfit_segment_eval metric render size mismatch: "
                        f"pred={len(metric_preds)} targets={len(metric_targets)} refs={len(chunk_refs)}"
                    )
                for local_idx, (pred, tgt) in enumerate(zip(metric_preds, metric_targets)):
                    global_idx = int(chunk_start + local_idx)
                    gt = tgt["gt_image"]
                    if gt.dim() == 4:
                        gt = gt.squeeze(0)
                    psnr_full_v = _psnr(pred, gt)
                    psnr_non_sky_v: Optional[float] = None
                    sky_coverage: Optional[float] = None
                    sky_mask = tgt.get("sky_mask")
                    if sky_mask is None and require_sky_mask:
                        raise ValueError(
                            "overfit_segment_eval.metrics.require_sky_mask=true but target missing sky_mask "
                            f"(frame={chunk_refs[local_idx][0]} cam={chunk_refs[local_idx][1]})"
                        )
                    if sky_mask is not None:
                        sm = sky_mask.to(device).float()
                        if sm.dim() == 3:
                            sm = sm.squeeze(-1)
                        if sm.shape != gt.shape[:2]:
                            raise ValueError(
                                "overfit_segment_eval sky_mask shape mismatch: "
                                f"sky_mask={tuple(sm.shape)} gt_hw={tuple(gt.shape[:2])}"
                            )
                        sky_coverage = float(sm.mean().item())
                        non_sky = (1.0 - sm).clamp(0.0, 1.0)
                        if int((non_sky > 0.5).sum().item()) >= int(min_valid_pixels):
                            psnr_non_sky_v = _masked_psnr(pred, gt, non_sky)
                    if metric_scope == "non_sky":
                        if psnr_non_sky_v is None:
                            raise ValueError(
                                "overfit_segment_eval metrics scope=non_sky but no valid non-sky pixels for "
                                f"frame={chunk_refs[local_idx][0]} cam={chunk_refs[local_idx][1]}"
                            )
                        psnr_v = float(psnr_non_sky_v)
                    else:
                        psnr_v = float(psnr_full_v)
                    psnr_vals.append(float(psnr_v))
                    psnr_full_vals.append(float(psnr_full_v))
                    if psnr_non_sky_v is not None:
                        psnr_non_sky_vals.append(float(psnr_non_sky_v))
                    if sky_coverage is not None:
                        sky_coverage_vals.append(float(sky_coverage))
                    frame_idx = int(tgt.get("frame_idx", chunk_refs[local_idx][0]))
                    cam_idx = int(tgt.get("cam_idx", chunk_refs[local_idx][1]))
                    row = {
                        "index": int(global_idx),
                        "frame_idx": int(frame_idx),
                        "cam_idx": int(cam_idx),
                        "psnr": float(psnr_v),
                        "psnr_full": float(psnr_full_v),
                        "psnr_non_sky": float(psnr_non_sky_v) if psnr_non_sky_v is not None else None,
                        "sky_mask_coverage": float(sky_coverage) if sky_coverage is not None else None,
                        "metric_scope": str(metric_scope),
                    }
                    per_view_rows.append(row)
                    if save_images:
                        _save_image_triplet(
                            int(trigger_step),
                            pred,
                            gt,
                            os.path.join(out_dir, "renders"),
                            view_suffix=f"seg_eval_v{global_idx:04d}_f{frame_idx:05d}_c{cam_idx}",
                            save_error=False,
                        )
        finally:
            if prev_mode:
                model.train()

        mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
        mean_psnr_full = float(np.mean(psnr_full_vals)) if psnr_full_vals else 0.0
        mean_psnr_non_sky = float(np.mean(psnr_non_sky_vals)) if psnr_non_sky_vals else 0.0
        best_before = float(self.best_psnr)
        row = {
            "split": "overfit_segment_eval",
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "trigger_step": int(trigger_step),
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "num_images": int(len(per_view_rows)),
            "psnr": float(mean_psnr),
            "psnr_full": float(mean_psnr_full),
            "psnr_non_sky": float(mean_psnr_non_sky),
            "num_images_non_sky_metric": int(len(psnr_non_sky_vals)),
            "sky_mask_coverage": float(np.mean(sky_coverage_vals)) if sky_coverage_vals else 0.0,
            "metric_scope": str(metric_scope),
            "node_state_exported": False,
            "node_state_export_path": None,
            "best_psnr_before": best_before if bool(np.isfinite(best_before)) else None,
        }

        export_cfg = _cfg_get(raw, "export_node_state", {}) or {}
        if bool(_cfg_get(export_cfg, "enable", False)):
            min_psnr = float(_cfg_get(export_cfg, "min_psnr", 0.0))
            if float(mean_psnr) > min_psnr and float(mean_psnr) > float(self.best_psnr):
                export_dir = os.path.join(str(cfg.log_dir), str(_cfg_get(export_cfg, "save_dir", "node_state_exports")))
                os.makedirs(export_dir, exist_ok=True)
                export_path = os.path.join(
                    export_dir,
                    (
                        f"node_state_scene{int(scene_id):03d}_segment{int(segment_id):03d}_"
                        f"episode{int(trigger_train_episode_counter):06d}_"
                        f"step{int(trigger_step):08d}_psnr{float(mean_psnr):.4f}.pt"
                    ),
                )
                payload = {
                    "format": "streetforward_stage5_6_overfit_node_state_v1",
                    "scene_id": int(scene_id),
                    "segment_id": int(segment_id),
                    "trigger_train_episode_counter": int(trigger_train_episode_counter),
                    "trigger_step": int(trigger_step),
                    "psnr": float(mean_psnr),
                    "previous_best_psnr": float(self.best_psnr),
                    "threshold_psnr": float(min_psnr),
                    "gs_state": gs_state,
                    "runtime_node_state": _runtime_node_state_payload(model, scene_id, segment_id),
                }
                torch.save(payload, export_path)
                self.best_psnr = float(mean_psnr)
                row["node_state_exported"] = True
                row["node_state_export_path"] = export_path
                logger.info(
                    "OVERFIT_SEGMENT_NODE_STATE_EXPORT scene_id=%s segment_id=%s episode=%s step=%s psnr=%.4f path=%s",
                    int(scene_id),
                    int(segment_id),
                    int(trigger_train_episode_counter),
                    int(trigger_step),
                    float(mean_psnr),
                    export_path,
                )

        with open(os.path.join(out_dir, "per_view_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(per_view_rows, f, indent=2)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        if metrics_fh is not None:
            _write_metrics_history(metrics_fh, row)
        if writer is not None:
            writer.add_scalar("overfit_segment_eval/psnr", float(row["psnr"]), int(trigger_step))
            writer.add_scalar("overfit_segment_eval/psnr_full", float(row["psnr_full"]), int(trigger_step))
            writer.add_scalar("overfit_segment_eval/psnr_non_sky", float(row["psnr_non_sky"]), int(trigger_step))
            writer.add_scalar(
                "overfit_segment_eval/num_images",
                float(row["num_images"]),
                int(trigger_step),
            )
        logger.info(
            "OVERFIT_SEGMENT_EVAL_END episode=%s step=%s scene_id=%s segment_id=%s images=%s "
            "metric_scope=%s psnr=%.4f psnr_full=%.4f psnr_non_sky=%.4f exported=%s",
            int(trigger_train_episode_counter),
            int(trigger_step),
            int(scene_id),
            int(segment_id),
            int(row["num_images"]),
            str(row["metric_scope"]),
            float(row["psnr"]),
            float(row["psnr_full"]),
            float(row["psnr_non_sky"]),
            bool(row["node_state_exported"]),
        )


__all__ = [
    "OverfitSegmentEpisodeEvaluator",
    "build_multi_scene_dataset_v4",
    "build_train_scheduler_v8_one_segment_overfit_from_cfg",
    "resolve_fixed_scene_segment_v8_one_segment",
]
