"""
Training script for Minimal StreetForward Stage 1.1 (NodeState + single target + GRU).

- Same pipeline as Stage 1 script: overfit one batch, same config/eval/logging.
- Model: MinimalStreetForwardStage1_1 (NodeStateBackground + GRU-style offset prediction).

Use with overfit batch:
  python tools/train_minimal_streetforward_stage1_1.py --config_file configs/minimal_streetforward_stage1_1.yaml \\
    overfit_batch_path=./data/overfit_batches/scene0_seg0_batch.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.streetforward.minimal_trainer_stage1_1 import MinimalStreetForwardStage1_1
from utils.logging import setup_logging
from utils.minimal_batch_view_selection import (
    build_explicit_minimal_batch_parts,
    build_explicit_targets_only,
    find_row,
    parse_view_selection,
)
from utils.streetforward_baseline import set_deterministic_seed
from tools.upload_to_vika import upload_experiment_summary

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

logger = logging.getLogger(__name__)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

# Checkpoint filename prefix for Stage 1.1
CKPT_PREFIX = "minimal_sf_stage1_1"


def _open_metrics_history(
    log_dir: str,
    enable_jsonl: bool,
    *,
    append: bool = True,
) -> Optional[TextIO]:
    if not enable_jsonl:
        return None
    metrics_path = os.path.join(log_dir, "metrics_history.jsonl")
    mode = "a" if append else "w"
    return open(metrics_path, mode, encoding="utf-8")


def _write_metrics_history(
    fh: Optional[TextIO],
    record: Dict,
) -> None:
    if fh is None:
        return
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def _save_image_triplet(
    step: int,
    pred_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    out_dir: str,
    view_suffix: Optional[str] = None,
    *,
    save_error: bool = True,
) -> None:
    """Save pred / gt / (optional) error images to out_dir as PNG. If view_suffix (e.g. 'view0'), use step{step:06d}_{view_suffix}_{name}.png."""
    os.makedirs(out_dir, exist_ok=True)
    pred = torch.clamp(pred_rgb.detach().cpu(), 0.0, 1.0)
    gt = torch.clamp(gt_image.detach().cpu(), 0.0, 1.0)
    error = (pred - gt).abs()
    if save_error and error.numel() > 0:
        max_val = float(error.max().item())
        if max_val > 0:
            error = error / max_val
    name_prefix = f"step{step:06d}_{view_suffix}_" if view_suffix else f"step{step:06d}_"
    triple: List[Tuple[str, torch.Tensor]] = [("pred", pred), ("gt", gt)]
    if save_error:
        triple.append(("error", error))
    for name, img in triple:
        img_np = (img.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        filename = os.path.join(out_dir, f"{name_prefix}{name}.png")
        try:
            from PIL import Image
        except ImportError:
            np.save(filename.replace(".png", ".npy"), img_np)
            continue
        Image.fromarray(img_np).save(filename)


def _hwc01_to_nchw01(img: torch.Tensor) -> torch.Tensor:
    """[H,W,3] -> [1,3,H,W] in [0,1]."""
    if img.dim() != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape={tuple(img.shape)}")
    return img.permute(2, 0, 1).unsqueeze(0)


def _compute_metrics(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: SSIM,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    compute_psnr: bool,
    compute_heavy: bool,
) -> Dict[str, float]:
    pred = torch.clamp(pred_rgb, 0.0, 1.0)
    gt = torch.clamp(gt_rgb, 0.0, 1.0)
    out: Dict[str, float] = {}
    pred_nchw = _hwc01_to_nchw01(pred)
    gt_nchw = _hwc01_to_nchw01(gt)
    if compute_psnr:
        psnr_metric.reset()
        out["psnr"] = float(psnr_metric(pred_nchw, gt_nchw).item())
    if compute_heavy:
        out["ssim"] = float(ssim_metric(pred_nchw, gt_nchw).item())
        lpips_metric.reset()
        out["lpips"] = float(lpips_metric(pred_nchw, gt_nchw).item())
    return out


def _build_minimal_source_stack_from_dataset_source(
    source_data: Dict,
    device: torch.device,
) -> Dict[str, Any]:
    """Build source_views / source_images (and optional per-view masks) from batch['source'] (MultiSceneDataset)."""
    images = source_data["image"]
    n = int(images.shape[0])
    extrinsics = source_data["extrinsics"]
    intrinsics = source_data["intrinsics"]
    frame_indices = source_data.get("frame_indices")
    source_views: List[Any] = []
    source_images: List[torch.Tensor] = []
    for i in range(n):
        view = type(
            "View",
            (),
            {
                "camtoworlds": extrinsics[i].to(device),
                "Ks": intrinsics[i][:3, :3].unsqueeze(0).to(device),
            },
        )()
        source_views.append(view)
        source_images.append(images[i].to(device))
    sfi = 0
    if frame_indices is not None and int(frame_indices.shape[0]) > 0:
        sfi = int(frame_indices[0].item())
    out: Dict[str, Any] = {
        "source_views": source_views,
        "source_images": source_images,
        "source_frame_idx": sfi,
    }
    src_sky = source_data.get("sky_mask")
    src_vd = source_data.get("viewdirs")
    src_ego = source_data.get("egocar_mask")
    src_dyn = source_data.get("dynamic_mask")
    if src_sky is not None and src_sky.shape[0] >= n:
        src_sky_list = [src_sky[i].to(device) for i in range(n)]
        out["source_sky_masks"] = src_sky_list
        out["source_sky_mask"] = src_sky_list
    if src_vd is not None and src_vd.shape[0] >= n:
        out["source_viewdirs"] = [src_vd[i].to(device) for i in range(n)]
    if src_ego is not None and src_ego.shape[0] >= n:
        src_ego_list = [src_ego[i].to(device) for i in range(n)]
        out["source_egocar_masks"] = src_ego_list
        out["source_egocar_mask"] = src_ego_list
    if src_dyn is not None and src_dyn.shape[0] >= n:
        src_dyn_list = [src_dyn[i].to(device) for i in range(n)]
        out["source_dynamic_masks"] = src_dyn_list
        out["source_dynamic_mask"] = src_dyn_list
    return out


def _finalize_test_frame_indices_for_minimal(
    test_frame_indices: List[int],
    targets_minimal: List[Dict],
) -> List[int]:
    """Replace sentinel -1 with first training target frame when batch['test'] omits frame_indices."""
    if not test_frame_indices:
        return test_frame_indices
    fb = int(targets_minimal[0]["frame_idx"]) if targets_minimal else 0
    return [fb if x < 0 else x for x in test_frame_indices]


def _role_dict_to_minimal_targets(
    role_data: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    from datasets.base.pixel_source import get_rays

    num_target = int(role_data["image"].shape[0])
    target_views = []
    gt_images = []
    viewdirs_list: List[Optional[torch.Tensor]] = [None] * num_target
    target_viewdirs = role_data.get("viewdirs")
    for i in range(num_target):
        view = SimpleNamespace(
            camtoworlds=role_data["extrinsics"][i].to(device),
            Ks=role_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
        )
        target_views.append(view)
        gt_images.append(role_data["image"][i].to(device))
        if target_viewdirs is not None:
            viewdirs_list[i] = target_viewdirs[i].to(device)
        else:
            gt = role_data["image"][i]
            h, w = int(gt.shape[0]), int(gt.shape[1])
            c2w = role_data["extrinsics"][i]
            intrinsic = role_data["intrinsics"][i][:3, :3]
            if c2w.dim() == 2:
                c2w = c2w.unsqueeze(0)
            if intrinsic.dim() == 2:
                intrinsic = intrinsic.unsqueeze(0)
            y_coords = torch.arange(h, device=device, dtype=torch.float32)
            x_coords = torch.arange(w, device=device, dtype=torch.float32)
            x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")
            _, viewdirs, _ = get_rays(
                x_grid.flatten(), y_grid.flatten(),
                c2w.to(device), intrinsic.to(device),
            )
            viewdirs_list[i] = viewdirs.reshape(h, w, 3)
    frame_indices = role_data.get("frame_indices")
    cam_indices = role_data.get("cam_indices")
    sky_mask = role_data.get("sky_mask")
    egocar_mask = role_data.get("egocar_mask")
    dynamic_mask = role_data.get("dynamic_mask")
    targets = [
        {
            "frame_idx": int(frame_indices[i]) if frame_indices is not None else 0,
            **({"cam_idx": int(cam_indices[i])} if cam_indices is not None else {}),
            "view": target_views[i],
            "gt_image": gt_images[i],
            **({"sky_mask": sky_mask[i].to(device)} if sky_mask is not None else {}),
            **({"egocar_mask": egocar_mask[i].to(device)} if egocar_mask is not None else {}),
            **({"dynamic_mask": dynamic_mask[i].to(device)} if dynamic_mask is not None else {}),
            **({"viewdirs": viewdirs_list[i]} if viewdirs_list[i] is not None else {}),
        }
        for i in range(num_target)
    ]
    return targets


def convert_batch_to_minimal_format(
    batch: Dict,
    device: torch.device,
    num_targets: Optional[int] = None,
    include_source_for_2d: bool = False,
    view_selection: Optional[Any] = None,
) -> Dict:
    """Convert raw overfit/dataset batch to minimal format.

    When num_targets is None, keeps single target (targets_minimal = [targets[0]]) for Stage 1.1.
    When num_targets is set (e.g. 3), targets_minimal = targets[:num_targets]; uses all if fewer available.
    When include_source_for_2d is True, adds source_views, source_images, source_frame_idx from the first target.

    When training.view_selection.mode == explicit (pass view_selection), selects rows by (frame_idx, cam_id)
    from batch['source'] / batch['target']; num_targets must be None; include_source_for_2d must be True.
    """
    scene_id = batch.get("scene_id")
    segment_id = batch.get("segment_id")
    if torch.is_tensor(scene_id):
        scene_id = scene_id.item()
    if torch.is_tensor(segment_id):
        segment_id = segment_id.item() if segment_id.numel() == 1 else int(segment_id[0].item())

    pointcloud = batch.get("pointcloud")
    if pointcloud is None:
        raise ValueError("batch must contain 'pointcloud'")
    knn_init_batch = batch.get("knn_init")
    knn_struct_neighbors_batch = batch.get("knn_struct_neighbors")
    request_meta_batch = batch.get("request_meta")
    passthrough: Dict[str, Any] = {}
    if request_meta_batch is not None:
        if isinstance(request_meta_batch, dict):
            passthrough["request_meta"] = dict(request_meta_batch)
        else:
            passthrough["request_meta"] = request_meta_batch
    for k in (
        "_scheduler_v4_aligned_info",
        "_scheduler_v7_aligned_info",
        "_scheduler_v8_aligned_info",
        "_scheduler_v9_aligned_info",
        "_scheduler_v9",
        "_scheduler_long_phase_b",
        "_iforward",
        "_iforward_plan",
        "_iforward_runtime_maps",
        "rollout_plan",
    ):
        if batch.get(k) is not None:
            passthrough[k] = batch.get(k)
    if isinstance(pointcloud, dict):
        # Keep all available branches (background/dynamic/...) so downstream
        # trainers (e.g. stage4 rigid) can access dynamic pointclouds.
        pointcloud_minimal = dict(pointcloud)
        if "background" not in pointcloud_minimal:
            pointcloud_minimal["background"] = np.zeros((0, 6), dtype=np.float32)
    else:
        pointcloud_minimal = pointcloud

    explicit = parse_view_selection(view_selection)
    if explicit is not None:
        if num_targets is not None:
            raise ValueError(
                "training.num_targets conflicts with training.view_selection.mode=explicit; "
                "remove num_targets from config when using explicit observation lists."
            )
        test_views: List[Any] = []
        test_images: List[torch.Tensor] = []
        test_frame_indices: List[int] = []
        test_data = batch.get("test")
        if isinstance(test_data, dict) and "image" in test_data and test_data["image"].numel() > 0:
            num_test = int(test_data["image"].shape[0])
            fi = test_data.get("frame_indices")
            for i in range(num_test):
                view = type(
                    "View",
                    (),
                    {
                        "camtoworlds": test_data["extrinsics"][i].to(device),
                        "Ks": test_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
                    },
                )()
                test_views.append(view)
                test_images.append(test_data["image"][i].to(device))
                if fi is not None and hasattr(fi, "shape") and int(fi.shape[0]) > i:
                    test_frame_indices.append(int(fi[i].item()))
                else:
                    test_frame_indices.append(-1)
        if include_source_for_2d:
            targets_minimal, source_views, source_images, source_frame_idx = build_explicit_minimal_batch_parts(
                batch, device, explicit
            )
            source_sky_masks: List[torch.Tensor] = []
            source_viewdirs: List[torch.Tensor] = []
            source_egocar_masks: List[torch.Tensor] = []
            source_dynamic_masks: List[torch.Tensor] = []
            source_data = batch.get("source")
            if not isinstance(source_data, dict):
                raise ValueError("explicit view_selection requires batch['source'] dict for source-only Stage4 contract.")
            sfi = source_data.get("frame_indices")
            sci = source_data.get("cam_indices")
            if sfi is None or sci is None:
                raise ValueError("batch['source'] must contain frame_indices and cam_indices.")
            for frame_idx, cam_id in explicit.source_refs:
                row = find_row(sfi, sci, frame_idx, cam_id, role="batch['source']")
                src_sky = source_data.get("sky_mask")
                src_vd = source_data.get("viewdirs")
                src_ego = source_data.get("egocar_mask")
                src_dyn = source_data.get("dynamic_mask")
                if src_sky is not None:
                    source_sky_masks.append(src_sky[row].to(device))
                if src_vd is not None:
                    source_viewdirs.append(src_vd[row].to(device))
                if src_ego is not None:
                    source_egocar_masks.append(src_ego[row].to(device))
                if src_dyn is not None:
                    source_dynamic_masks.append(src_dyn[row].to(device))
            if test_frame_indices:
                test_frame_indices = _finalize_test_frame_indices_for_minimal(test_frame_indices, targets_minimal)
            return {
                "scene_id": scene_id,
                "segment_id": segment_id,
                "pointcloud": pointcloud_minimal,
                "targets": targets_minimal,
                "test_views": test_views,
                "test_images": test_images,
                **({"test_frame_indices": test_frame_indices} if test_frame_indices else {}),
                "source_views": source_views,
                "source_images": source_images,
                "source_frame_idx": int(source_frame_idx),
                **({"dynamic_info": batch.get("dynamic_info")} if batch.get("dynamic_info") is not None else {}),
                **({"knn_init": knn_init_batch} if knn_init_batch is not None else {}),
                **(
                    {"knn_struct_neighbors": knn_struct_neighbors_batch}
                    if isinstance(knn_struct_neighbors_batch, dict)
                    else {}
                ),
                **({"source_sky_masks": source_sky_masks} if source_sky_masks else {}),
                **({"source_sky_mask": source_sky_masks} if source_sky_masks else {}),
                **({"source_viewdirs": source_viewdirs} if source_viewdirs else {}),
                **({"source_egocar_masks": source_egocar_masks} if source_egocar_masks else {}),
                **({"source_egocar_mask": source_egocar_masks} if source_egocar_masks else {}),
                **({"source_dynamic_masks": source_dynamic_masks} if source_dynamic_masks else {}),
                **({"source_dynamic_mask": source_dynamic_masks} if source_dynamic_masks else {}),
                **passthrough,
            }
        targets_minimal = build_explicit_targets_only(batch, device, explicit)
        if test_frame_indices:
            test_frame_indices = _finalize_test_frame_indices_for_minimal(test_frame_indices, targets_minimal)
        return {
            "scene_id": scene_id,
            "segment_id": segment_id,
            "pointcloud": pointcloud_minimal,
            "targets": targets_minimal,
            "test_views": test_views,
            "test_images": test_images,
            **({"test_frame_indices": test_frame_indices} if test_frame_indices else {}),
            **({"dynamic_info": batch.get("dynamic_info")} if batch.get("dynamic_info") is not None else {}),
            **({"knn_init": knn_init_batch} if knn_init_batch is not None else {}),
            **(
                {"knn_struct_neighbors": knn_struct_neighbors_batch}
                if isinstance(knn_struct_neighbors_batch, dict)
                else {}
            ),
            **passthrough,
        }

    target_data = batch.get("target", batch.get("targets"))
    if target_data is None:
        raise ValueError("batch must contain 'target' or 'targets'")

    if isinstance(target_data, dict):
        targets = _role_dict_to_minimal_targets(target_data, device=device)
    else:
        targets = target_data

    if not targets:
        raise ValueError("At least one target required for minimal trainer")
    if num_targets is not None:
        targets_minimal = targets[:num_targets]
    else:
        targets_minimal = [targets[0]]

    test_views: List[Any] = []
    test_images: List[torch.Tensor] = []
    test_frame_indices: List[int] = []
    test_data = batch.get("test")
    if isinstance(test_data, dict) and "image" in test_data and test_data["image"].numel() > 0:
        num_test = int(test_data["image"].shape[0])
        fi = test_data.get("frame_indices")
        for i in range(num_test):
            view = type(
                "View",
                (),
                {
                    "camtoworlds": test_data["extrinsics"][i].to(device),
                    "Ks": test_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
                },
            )()
            test_views.append(view)
            test_images.append(test_data["image"][i].to(device))
            if fi is not None and hasattr(fi, "shape") and int(fi.shape[0]) > i:
                test_frame_indices.append(int(fi[i].item()))
            else:
                test_frame_indices.append(-1)
    if test_frame_indices:
        test_frame_indices = _finalize_test_frame_indices_for_minimal(test_frame_indices, targets_minimal)

    result: Dict = {
        "scene_id": scene_id,
        "segment_id": segment_id,
        "pointcloud": pointcloud_minimal,
        "targets": targets_minimal,
        "test_views": test_views,
        "test_images": test_images,
        **({"test_frame_indices": test_frame_indices} if test_frame_indices else {}),
    }
    if "dynamic_info" in batch and batch.get("dynamic_info") is not None:
        result["dynamic_info"] = batch["dynamic_info"]
    if knn_init_batch is not None:
        result["knn_init"] = knn_init_batch
    if isinstance(knn_struct_neighbors_batch, dict):
        result["knn_struct_neighbors"] = knn_struct_neighbors_batch
    aux_target_data = batch.get("aux_target")
    if isinstance(aux_target_data, dict) and int(aux_target_data.get("image", torch.zeros((0,))).shape[0]) > 0:
        result["aux_targets"] = _role_dict_to_minimal_targets(aux_target_data, device=device)
    query_label_data = batch.get("query_label")
    if isinstance(query_label_data, dict) and int(query_label_data.get("image", torch.zeros((0,))).shape[0]) > 0:
        result["query_targets"] = _role_dict_to_minimal_targets(query_label_data, device=device)
    result.update(passthrough)
    if include_source_for_2d:
        source_data = batch.get("source")
        if isinstance(source_data, dict) and source_data.get("image") is not None:
            n_src = int(source_data["image"].shape[0])
            if n_src > 0:
                result.update(_build_minimal_source_stack_from_dataset_source(source_data, device))
            else:
                first = targets_minimal[0]
                result["source_views"] = [first["view"]]
                result["source_images"] = [
                    first["gt_image"].to(device) if first["gt_image"].device != device else first["gt_image"]
                ]
                result["source_frame_idx"] = int(first.get("frame_idx", 0))
        else:
            first = targets_minimal[0]
            result["source_views"] = [first["view"]]
            result["source_images"] = [
                first["gt_image"].to(device) if first["gt_image"].device != device else first["gt_image"]
            ]
            result["source_frame_idx"] = int(first.get("frame_idx", 0))
            if isinstance(source_data, dict):
                src_sky = source_data.get("sky_mask")
                src_vd = source_data.get("viewdirs")
                src_ego = source_data.get("egocar_mask")
                src_dyn = source_data.get("dynamic_mask")
                if src_sky is not None and src_sky.shape[0] > 0:
                    src_sky_list = [src_sky[0].to(device)]
                    result["source_sky_masks"] = src_sky_list
                    result["source_sky_mask"] = src_sky_list
                if src_vd is not None and src_vd.shape[0] > 0:
                    result["source_viewdirs"] = [src_vd[0].to(device)]
                if src_ego is not None and src_ego.shape[0] > 0:
                    src_ego_list = [src_ego[0].to(device)]
                    result["source_egocar_masks"] = src_ego_list
                    result["source_egocar_mask"] = src_ego_list
                if src_dyn is not None and src_dyn.shape[0] > 0:
                    src_dyn_list = [src_dyn[0].to(device)]
                    result["source_dynamic_masks"] = src_dyn_list
                    result["source_dynamic_mask"] = src_dyn_list
    return result


def setup(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config_file)
    if getattr(args, "opts", None):
        cli = OmegaConf.from_cli(args.opts)
        cfg = OmegaConf.merge(cfg, cli)

    if "data" not in cfg:
        cfg.data = {}
    if "model" not in cfg:
        raise ValueError("config must contain 'model'")
    if "optimizer" not in cfg:
        cfg.optimizer = {"lr": 1e-3, "eps": 1e-15, "weight_decay": 0.0}

    if "eval" not in cfg:
        cfg.eval = {}
    if "enable_psnr" not in cfg.eval:
        cfg.eval.enable_psnr = True
    if "metric_interval" not in cfg.eval:
        cfg.eval.metric_interval = 10
    if "heavy_metric_interval" not in cfg.eval:
        cfg.eval.heavy_metric_interval = 50
    if "run_test_at_end" not in cfg.eval:
        cfg.eval.run_test_at_end = True

    if "logging" not in cfg:
        cfg.logging = {}
    if "image_interval" not in cfg.logging:
        cfg.logging.image_interval = 50
    if "enable_jsonl_metrics" not in cfg.logging:
        cfg.logging.enable_jsonl_metrics = True
    if "metrics_history_append" not in cfg.logging:
        cfg.logging.metrics_history_append = True
    if "use_tensorboard" not in cfg.logging:
        cfg.logging.use_tensorboard = False

    run_name = cfg.get("output_name", getattr(args, "run_name", "overfit"))
    logging_cfg = cfg.logging if cfg.get("logging") is not None else {}
    log_dir_override = logging_cfg.get("log_dir")
    if log_dir_override is not None:
        log_dir_override = str(log_dir_override).strip()
    if log_dir_override:
        log_dir = os.path.abspath(log_dir_override)
    else:
        output_root = str(logging_cfg.get("output_root", getattr(args, "output_root", "outputs"))).strip()
        project = str(logging_cfg.get("project", getattr(args, "project", "minimal_sf"))).strip()
        if not output_root:
            raise ValueError("logging.output_root must be non-empty when logging.log_dir is not set")
        if not project:
            raise ValueError("logging.project must be non-empty when logging.log_dir is not set")
        log_dir = os.path.join(output_root, project, run_name)
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for sub in ("images", "checkpoints", "tb"):
        os.makedirs(os.path.join(log_dir, sub), exist_ok=True)

    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train Minimal StreetForward Stage 1.1 (overfit one batch)")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/minimal_streetforward_stage1_1.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--project", type=str, default="minimal_sf")
    parser.add_argument("--run_name", type=str, default="overfit")
    parser.add_argument("--overfit_batch_path", type=str, default=None, help="Path to .pt overfit batch")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training.max_iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("opts", nargs="*", help="Override config, e.g. overfit_batch_path=path/to/batch.pt")
    args = parser.parse_args()

    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("RUN start time=%s device=%s", current_time, device)

    set_deterministic_seed(args.seed)
    logger.info("Seed: %s", args.seed)

    overfit_path = getattr(args, "overfit_batch_path", None) or cfg.get("overfit_batch_path")
    if not overfit_path or not os.path.isfile(overfit_path):
        raise FileNotFoundError(
            "Overfit batch required. Set --overfit_batch_path or config overfit_batch_path."
        )
    logger.info(
        "RUN config_path=%s log_dir=%s overfit_batch_path=%s",
        args.config_file,
        cfg.log_dir,
        overfit_path,
    )
    logger.info("Loading overfit batch from %s", overfit_path)
    from tools.overfit_one_batch import load_batch
    raw_batch = load_batch(overfit_path)
    view_sel = cfg.training.get("view_selection")
    explicit = parse_view_selection(view_sel)
    num_targets = None if explicit is not None else cfg.training.get("num_targets")
    minimal_batch = convert_batch_to_minimal_format(
        raw_batch,
        device,
        num_targets=num_targets,
        include_source_for_2d=False,
        view_selection=view_sel,
    )

    logger.info("Building MinimalStreetForwardStage1_1...")
    model = MinimalStreetForwardStage1_1(config=cfg, device=device)
    model.train()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    max_iterations = args.max_steps or cfg.training.get("max_iterations", 1000)
    log_interval = cfg.training.get("log_interval", 50)
    save_every = cfg.training.get("save_checkpoint_freq", 500)

    metric_interval = int(cfg.eval.get("metric_interval", 10))
    heavy_metric_interval = int(cfg.eval.get("heavy_metric_interval", 50))
    enable_psnr = bool(cfg.eval.get("enable_psnr", True))
    run_test_at_end = bool(cfg.eval.get("run_test_at_end", True))
    enable_jsonl_metrics = bool(cfg.logging.get("enable_jsonl_metrics", True))
    metrics_history_append = bool(cfg.logging.get("metrics_history_append", True))
    image_interval = int(cfg.logging.get("image_interval", 50))
    use_tensorboard = bool(cfg.logging.get("use_tensorboard", False))

    metrics_fh: Optional[TextIO] = None
    writer: Optional["SummaryWriter"] = None
    result: Dict[str, Any] = {}

    # Step-level profiling accumulators
    total_steps = 0
    sum_step_time_ms = 0.0
    peak_mem_bytes = 0
    peak_mem_reserved_bytes = 0
    try:
        metrics_fh = _open_metrics_history(
            cfg.log_dir,
            enable_jsonl_metrics,
            append=metrics_history_append,
        )

        if use_tensorboard and SummaryWriter is not None:
            tb_dir = os.path.join(cfg.log_dir, "tb")
            writer = SummaryWriter(log_dir=tb_dir)

        logger.info(
            "Training for %s steps (log every %s, save every %s, metric_interval=%s, heavy_metric_interval=%s)",
            max_iterations,
            log_interval,
            save_every,
            metric_interval,
            heavy_metric_interval,
        )

        for step in range(max_iterations):
            step_start_wall = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            result = model.train_step(minimal_batch, step=step)

            step_end_wall = time.time()
            step_wall_time_ms = (step_end_wall - step_start_wall) * 1000.0
            total_steps += 1
            sum_step_time_ms += step_wall_time_ms
            if torch.cuda.is_available():
                step_mem = torch.cuda.max_memory_allocated()
                step_mem_reserved = torch.cuda.max_memory_reserved()
                if step_mem > peak_mem_bytes:
                    peak_mem_bytes = int(step_mem)
                if step_mem_reserved > peak_mem_reserved_bytes:
                    peak_mem_reserved_bytes = int(step_mem_reserved)
            loss_val = float(result["loss"])
            pred_rgb = result["pred_rgb"]
            gt_image = result["gt_image"]

            if step % log_interval == 0:
                logger.info(
                    "Step %s: loss=%.6f step_time_ms=%.2f",
                    step,
                    loss_val,
                    step_wall_time_ms,
                )

            want_psnr = enable_psnr and (step % metric_interval == 0)
            want_heavy = heavy_metric_interval > 0 and (step % heavy_metric_interval == 0)
            if want_psnr or want_heavy:
                mse_val = float(
                    torch.mean(
                        (torch.clamp(pred_rgb, 0.0, 1.0) - torch.clamp(gt_image, 0.0, 1.0))
                        ** 2
                    ).item()
                )
                metric_vals = _compute_metrics(
                    pred_rgb=pred_rgb,
                    gt_rgb=gt_image,
                    psnr_metric=psnr_metric,
                    ssim_metric=ssim_metric,
                    lpips_metric=lpips_metric,
                    compute_psnr=want_psnr,
                    compute_heavy=want_heavy,
                )

                log_parts = [
                    f"METRIC step={step} split=train loss_l1={loss_val:.6f} mse={mse_val:.6e}"
                ]
                if "psnr" in metric_vals:
                    log_parts.append(f"psnr={metric_vals['psnr']:.2f}")
                if "ssim" in metric_vals:
                    log_parts.append(f"ssim={metric_vals['ssim']:.4f}")
                if "lpips" in metric_vals:
                    log_parts.append(f"lpips={metric_vals['lpips']:.4f}")
                logger.info(" ".join(log_parts))

                record = {
                    "step": int(step),
                    "split": "train",
                    "loss_l1": loss_val,
                    "mse": mse_val,
                    **metric_vals,
                }
                _write_metrics_history(metrics_fh, record)

                if writer is not None:
                    writer.add_scalar("train/loss_l1", loss_val, step)
                    writer.add_scalar("train/mse", mse_val, step)
                    if "psnr" in metric_vals:
                        writer.add_scalar("train/psnr", metric_vals["psnr"], step)
                    if "ssim" in metric_vals:
                        writer.add_scalar("train/ssim", metric_vals["ssim"], step)
                    if "lpips" in metric_vals:
                        writer.add_scalar("train/lpips", metric_vals["lpips"], step)

            if step % image_interval == 0:
                images_dir = os.path.join(cfg.log_dir, "images", "train")
                _save_image_triplet(step, pred_rgb, gt_image, images_dir)

                if writer is not None:
                    pred_clamped = torch.clamp(pred_rgb.detach().cpu(), 0.0, 1.0)
                    gt_clamped = torch.clamp(gt_image.detach().cpu(), 0.0, 1.0)
                    error = (pred_clamped - gt_clamped).abs()
                    if error.numel() > 0:
                        max_val = float(error.max().item())
                        if max_val > 0:
                            error = error / max_val
                    writer.add_image("train/pred", pred_clamped.permute(2, 0, 1), step)
                    writer.add_image("train/gt", gt_clamped.permute(2, 0, 1), step)
                    writer.add_image("train/error", error.permute(2, 0, 1), step)

            if save_every and step > 0 and step % save_every == 0:
                ckpt_path = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_step{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model.optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint to %s", ckpt_path)

        test_metrics: Optional[Dict[str, float]] = None
        if run_test_at_end and minimal_batch.get("test_views"):
            prev_mode = model.training
            model.eval()
            with torch.no_grad():
                out = model.forward(minimal_batch)
                render_params = out["render_params"]

                psnr_list: List[float] = []
                ssim_list: List[float] = []
                lpips_list: List[float] = []

                test_views = minimal_batch.get("test_views", [])
                test_images = minimal_batch.get("test_images", [])
                for view, gt in zip(test_views, test_images):
                    h, w = int(gt.shape[0]), int(gt.shape[1])
                    pred, _ = model._render_single_view(render_params, view, h, w)
                    vals = _compute_metrics(
                        pred_rgb=pred,
                        gt_rgb=gt,
                        psnr_metric=psnr_metric,
                        ssim_metric=ssim_metric,
                        lpips_metric=lpips_metric,
                        compute_psnr=True,
                        compute_heavy=True,
                    )
                    psnr_list.append(vals["psnr"])
                    ssim_list.append(vals["ssim"])
                    lpips_list.append(vals["lpips"])

                if psnr_list:
                    test_metrics = {
                        "psnr": float(np.mean(psnr_list)),
                        "ssim": float(np.mean(ssim_list)),
                        "lpips": float(np.mean(lpips_list)),
                        "num_test_views": int(len(psnr_list)),
                    }
                    logger.info(
                        "METRIC final split=test psnr=%.2f ssim=%.4f lpips=%.4f num_test_views=%d",
                        test_metrics["psnr"],
                        test_metrics["ssim"],
                        test_metrics["lpips"],
                        test_metrics["num_test_views"],
                    )
                    if writer is not None:
                        writer.add_scalar("test/psnr", test_metrics["psnr"], max_iterations - 1)
                        writer.add_scalar("test/ssim", test_metrics["ssim"], max_iterations - 1)
                        writer.add_scalar("test/lpips", test_metrics["lpips"], max_iterations - 1)

                    avg_step_time_ms = (
                        sum_step_time_ms / max(total_steps, 1) if total_steps > 0 else 0.0
                    )
                    summary = {
                        "final_step": int(max_iterations - 1),
                        "train": {"loss_l1": float(result["loss"])},
                        "test": test_metrics,
                        "profiling": {
                            "avg_step_time_ms": avg_step_time_ms,
                            "peak_mem_bytes": peak_mem_bytes,
                            "peak_mem_reserved_bytes": peak_mem_reserved_bytes,
                        },
                    }
                    metrics_final_path = os.path.join(cfg.log_dir, "metrics_final.json")
                    with open(metrics_final_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)
                    logger.info(
                        "Saved metrics_final.json to %s (avg_step_time_ms=%.2f, peak_mem_bytes=%d)",
                        metrics_final_path,
                        avg_step_time_ms,
                        peak_mem_bytes,
                    )
                    # Try uploading summary to Vika (no-op if env or vika.py missing)
                    try:
                        upload_experiment_summary(cfg.log_dir, summary)
                    except Exception:
                        logger.exception("Vika upload failed for log_dir=%s", cfg.log_dir)

            if prev_mode:
                model.train()
    finally:
        if metrics_fh is not None:
            metrics_fh.close()
        if writer is not None:
            writer.close()

    logger.info("Done. Final loss: %.6f", result.get("loss", 0.0))
    final_ckpt = os.path.join(cfg.log_dir, "checkpoints", f"{CKPT_PREFIX}_final.pt")
    torch.save({
        "step": max_iterations - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
    }, final_ckpt)
    logger.info("Saved final checkpoint to %s", final_ckpt)


if __name__ == "__main__":
    main()
