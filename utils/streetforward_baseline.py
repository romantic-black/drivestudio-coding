"""
Utilities to record and validate StreetForward Golden Baselines.

The helpers here are shared by the CLI recorder and pytest regression tests.
They intentionally avoid stubs/mocks and operate on the real trainer + dataset
pipeline described in docs/trainers/StreetForward_Golden_Baseline_Design.md.
"""

from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset
    from models.trainers.streetforward import (
        NodeStateBackground,
        NodeStateDistant,
        NodeStateRigid,
        StreetForwardTrainer,
    )


# --- Determinism helpers ----------------------------------------------------


def set_deterministic_seed(seed: int) -> None:
    """Set all major RNG seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- Config & construction helpers -----------------------------------------


def load_config(config_path: str) -> OmegaConf:
    """
    Load an OmegaConf config and (best‑effort) merge dataset preset if present.

    Training configs often rely on `data.dataset` to pick a dataset preset
    (e.g. configs/datasets/nuscenes/*.yaml). We merge it only when a matching
    file exists; otherwise we keep the original config untouched.
    """
    cfg = OmegaConf.load(config_path)
    dataset_type = None
    if "data" in cfg and hasattr(cfg.data, "get"):
        dataset_type = cfg.data.get("dataset")
    if dataset_type:
        preset_path = Path("configs") / "datasets" / f"{dataset_type}.yaml"
        if preset_path.exists():
            cfg = OmegaConf.merge(cfg, OmegaConf.load(str(preset_path)))
    return cfg


def build_dataset(cfg: OmegaConf, device: torch.device):
    """Create MultiSceneDataset using the same parameters as the training script."""
    from datasets.multi_scene_dataset import MultiSceneDataset  # Lazy import to avoid hard cv2 dependency during import time
    if "data" not in cfg or "dataset" not in cfg:
        raise ValueError("Config must contain both data and dataset sections.")
    ds_cfg = cfg.dataset
    return MultiSceneDataset(
        data_cfg=cfg.data,
        train_scene_ids=cfg.data.train_scene_ids,
        eval_scene_ids=cfg.data.eval_scene_ids,
        num_source_keyframes=ds_cfg.num_source_keyframes,
        num_target_keyframes=ds_cfg.num_target_keyframes,
        segment_overlap_ratio=ds_cfg.segment_overlap_ratio,
        keyframe_split_config=ds_cfg.get("keyframe_split_config", None),
        min_keyframes_per_scene=ds_cfg.min_keyframes_per_scene,
        min_keyframes_per_segment=ds_cfg.min_keyframes_per_segment,
        device=device,
        preload_scene_count=ds_cfg.get("preload_scene_count", 3),
        segment_aabb=ds_cfg.segment_aabb,
        segment_input_aabb=ds_cfg.segment_input_aabb,
        pointcloud_config=ds_cfg.get("pointcloud", None),
    )


def convert_batch_to_streetforward_format(batch: Dict, device: torch.device) -> Dict:
    """
    Convert a MultiSceneDataset batch into StreetForwardTrainer's expected format.

    This mirrors tools/train_streetforward.py but lives here to avoid test/CLI
    import path issues.
    """
    scene_id = batch["scene_id"]
    segment_id = batch["segment_id"]
    if isinstance(segment_id, int):
        segment_id = torch.tensor([segment_id], dtype=torch.long)

    pointcloud = batch.get("pointcloud")
    if pointcloud is None:
        raise ValueError("pointcloud is required but not found in batch")
    dynamic_info = batch.get("dynamic_info")

    target_data = batch["target"]
    target_views = []
    gt_images = []
    targets = []

    num_target_images = target_data["image"].shape[0]
    for i in range(num_target_images):
        view = type(
            "View",
            (),
            {
                "camtoworlds": target_data["extrinsics"][i].to(device),
                "Ks": target_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
            },
        )()
        target_views.append(view)

        gt_image = target_data["image"][i].to(device)
        gt_images.append(gt_image)
        frame_indices = target_data.get("frame_indices")
        frame_idx = int(frame_indices[i]) if frame_indices is not None else 0
        sky_src = target_data.get("sky_mask")
        sky_mask = sky_src[i].to(device) if sky_src is not None else None
        target_entry = {
            "frame_idx": frame_idx,
            "view": view,
            "gt_image": gt_image,
        }
        if sky_mask is not None:
            target_entry["sky_mask"] = sky_mask
        targets.append(target_entry)

    source_views = []
    src_images = []
    source_frame_idx = None
    if "source" in batch:
        source_data = batch["source"]
        num_source_images = source_data["image"].shape[0]
        for i in range(num_source_images):
            view = type(
                "View",
                (),
                {
                    "camtoworlds": source_data["extrinsics"][i].to(device),
                    "Ks": source_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
                },
            )()
            source_views.append(view)
            src_image = source_data["image"][i].to(device)
            src_images.append(src_image)
            if source_frame_idx is None:
                frame_indices = source_data.get("frame_indices")
                if frame_indices is not None:
                    source_frame_idx = int(frame_indices[i])

    test_views = []
    test_images = []
    if "test" in batch:
        test_data = batch["test"]
        num_test_images = test_data["image"].shape[0]
        for i in range(num_test_images):
            view = type(
                "View",
                (),
                {
                    "camtoworlds": test_data["extrinsics"][i].to(device),
                    "Ks": test_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
                },
            )()
            test_views.append(view)
            test_image = test_data["image"][i].to(device)
            test_images.append(test_image)

    return {
        "scene_id": scene_id.to(device)
        if isinstance(scene_id, torch.Tensor)
        else torch.tensor([scene_id], dtype=torch.long).to(device),
        "segment_id": segment_id.to(device)
        if isinstance(segment_id, torch.Tensor)
        else torch.tensor([segment_id], dtype=torch.long).to(device),
        "pointcloud": pointcloud,
        "dynamic_info": dynamic_info,
        "target_views": target_views,
        "gt_images": gt_images,
        "targets": targets,
        "source_frame_idx": source_frame_idx if source_frame_idx is not None else 0,
        "source_views": source_views,
        "src_images": src_images,
        "test_views": test_views,
        "test_images": test_images,
    }


# --- Batch plan & cache helpers ---------------------------------------------


def batch_plan_from_dataset(
    dataset,
    max_scenes: int = 2,
    segments_per_scene: int = 2,
    batches_per_segment: int = 2,
) -> List[Tuple[int, int, int]]:
    """
    Simple deterministic plan: take first N scenes, first M segments per scene,
    and record K batches per (scene, segment).
    """
    if not getattr(dataset, "_initialized", False):
        dataset.initialize()
    plan: List[Tuple[int, int, int]] = []
    for sid in dataset.train_scene_ids[:max_scenes]:
        scene_data = dataset.get_scene(sid)
        if scene_data is None:
            continue
        num_segments = len(scene_data.get("segments", []))
        seg_ids = list(range(min(num_segments, segments_per_scene)))
        for seg_id in seg_ids:
            plan.append((sid, seg_id, batches_per_segment))
    return plan


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(x) for x in obj)
    return obj


def harvest_batch_cache(
    cfg: OmegaConf,
    device: torch.device,
    seed: int,
    plan: List[Tuple[int, int, int]],
    output_path: Path,
    include_test: bool = False,
) -> Dict:
    """
    Collect batches according to a plan and save them for reuse.
    """
    set_deterministic_seed(seed)
    dataset = build_dataset(cfg, device)
    if not getattr(dataset, "_initialized", False):
        dataset.initialize()

    batches: List[Dict] = []
    scene_segment_sequence: List[Tuple[int, int]] = []
    for scene_id, segment_id, num_batches in plan:
        for _ in range(num_batches):
            batch = dataset.get_segment_batch(scene_id, segment_id, include_test=include_test)
            batches.append(_to_cpu(batch))
            scene_segment_sequence.append((scene_id, segment_id))

    meta = {
        "seed": seed,
        "plan": plan,
        "scene_segment_sequence": scene_segment_sequence,
        "config_path": cfg.get("config_path", None),
        "include_test": include_test,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "batches": batches}, output_path)
    return meta


def load_batch_cache(cache_path: Path) -> Tuple[Dict, List[Dict]]:
    data = torch.load(cache_path, map_location="cpu")
    return data.get("meta", {}), data.get("batches", [])


# --- Summary helpers -------------------------------------------------------


def _tensor_stats(t: torch.Tensor) -> Dict:
    if t is None or t.numel() == 0:
        return {"shape": [0]}
    if t.requires_grad:
        t = t.detach()
    return {
        "shape": list(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "norm": float(t.norm().item()),
    }


def feature_tensor_summary(t: Optional[torch.Tensor]) -> Optional[Dict]:
    """Summary for 2D feature tensors [N, C] for baseline value alignment."""
    if t is None or t.numel() == 0:
        return None
    return _tensor_stats(t)


def node_state_background_summary(ns: Optional[Any]) -> Optional[Dict]:
    if ns is None:
        return None
    return {
        "num_points": int(ns.means.shape[0]),
        "means": _tensor_stats(ns.means),
        "scales_log": _tensor_stats(ns.scales_log),
        "opacity_logit": _tensor_stats(ns.opacity_logit),
        "sh_dc": _tensor_stats(ns.sh_dc),
        "sh_rest": _tensor_stats(ns.sh_rest),
    }


def node_state_rigid_summary(ns: Optional[Any]) -> Optional[Dict]:
    if ns is None:
        return None
    point_ids = ns.point_ids.squeeze(-1)
    hist = (
        torch.bincount(point_ids, minlength=len(ns.instance_ids)).tolist()
        if point_ids.numel() > 0
        else []
    )
    return {
        "num_points": int(ns.means.shape[0]),
        "point_ids_hist": hist,
        "means": _tensor_stats(ns.means),
        "scales_log": _tensor_stats(ns.scales_log),
        "opacity_logit": _tensor_stats(ns.opacity_logit),
        "sh_dc": _tensor_stats(ns.sh_dc),
        "sh_rest": _tensor_stats(ns.sh_rest),
        "instances_quats": _tensor_stats(ns.instances_quats),
        "instances_trans": _tensor_stats(ns.instances_trans),
    }


def node_state_distant_summary(ns: Optional[Any]) -> Optional[Dict]:
    if ns is None:
        return None
    return {
        "num_points": int(ns.means.shape[0]),
        "means": _tensor_stats(ns.means),
        "scales_log": _tensor_stats(ns.scales_log),
        "opacity_logit": _tensor_stats(ns.opacity_logit),
        "sh_dc": _tensor_stats(ns.sh_dc),
        "sh_rest": _tensor_stats(ns.sh_rest),
    }


def offsets_summary(offsets: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict]:
    if offsets is None:
        return None
    return {k: _tensor_stats(v) for k, v in offsets.items() if v is not None}


def grad_norms_summary(trainer: Any) -> Dict[str, float]:
    """Global L2 grad norm per key module."""
    def _module_norm(mod: Optional[torch.nn.Module]) -> float:
        if mod is None:
            return 0.0
        sq = []
        for p in mod.parameters():
            if p.grad is not None:
                sq.append(p.grad.detach().pow(2).sum().item())
        return float(np.sqrt(sum(sq))) if sq else 0.0

    return {
        "sparse_conv": _module_norm(getattr(trainer, "sparse_conv", None)),
        "mlp_offset_pos": _module_norm(getattr(trainer, "mlp_offset_pos", None)),
        "mlp_conv": _module_norm(getattr(trainer, "mlp_conv", None)),
        "mlp_opacity": _module_norm(getattr(trainer, "mlp_opacity", None)),
        "gaussion_decoder": _module_norm(getattr(trainer, "gaussion_decoder", None)),
    }


# --- Recording --------------------------------------------------------------


@dataclass
class BaselineStep:
    step: int
    scene_id: int
    segment_id: int
    total_loss: float
    num_targets: int
    num_bg: int
    num_rigid: int
    num_distant: int
    node_state_bg_summary: Optional[Dict]
    node_state_rigid_summary: Optional[Dict]
    node_state_distant_summary: Optional[Dict]
    offset_bg_summary: Optional[Dict]
    offset_rigid_summary: Optional[Dict]
    offset_distant_summary: Optional[Dict]
    feat_3d_bg_summary: Optional[Dict]
    feat_3d_rigid_summary: Optional[Dict]
    feat_3d_distant_summary: Optional[Dict]
    feat_2d_bg_summary: Optional[Dict]
    feat_2d_rigid_summary: Optional[Dict]
    feat_2d_distant_summary: Optional[Dict]
    feat_bg_input_summary: Optional[Dict]
    feat_rigid_input_summary: Optional[Dict]
    feat_distant_input_summary: Optional[Dict]
    grad_norms: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "scene_id": self.scene_id,
            "segment_id": self.segment_id,
            "total_loss": self.total_loss,
            "num_targets": self.num_targets,
            "num_bg": self.num_bg,
            "num_rigid": self.num_rigid,
            "num_distant": self.num_distant,
            "node_state_bg_summary": self.node_state_bg_summary,
            "node_state_rigid_summary": self.node_state_rigid_summary,
            "node_state_distant_summary": self.node_state_distant_summary,
            "offset_bg_summary": self.offset_bg_summary,
            "offset_rigid_summary": self.offset_rigid_summary,
            "offset_distant_summary": self.offset_distant_summary,
            "feat_3d_bg_summary": self.feat_3d_bg_summary,
            "feat_3d_rigid_summary": self.feat_3d_rigid_summary,
            "feat_3d_distant_summary": self.feat_3d_distant_summary,
            "feat_2d_bg_summary": self.feat_2d_bg_summary,
            "feat_2d_rigid_summary": self.feat_2d_rigid_summary,
            "feat_2d_distant_summary": self.feat_2d_distant_summary,
            "feat_bg_input_summary": self.feat_bg_input_summary,
            "feat_rigid_input_summary": self.feat_rigid_input_summary,
            "feat_distant_input_summary": self.feat_distant_input_summary,
            "grad_norms": self.grad_norms,
        }


def record_step(
    trainer: StreetForwardTrainer,
    batch: Dict,
    result: Dict,
    step_idx: int,
) -> BaselineStep:
    scene_id = int(batch["scene_id"].item()) if isinstance(batch["scene_id"], torch.Tensor) else int(batch["scene_id"])
    segment_id = int(batch["segment_id"].item()) if isinstance(batch["segment_id"], torch.Tensor) else int(batch["segment_id"])

    node_state_bg = result.get("node_state")
    node_state_rigid = result.get("node_state_rigid")
    node_state_distant = result.get("node_state_distant")

    offsets_bg = getattr(trainer, "_last_offsets_bg", None)
    offsets_rigid = getattr(trainer, "_last_offsets_rigid", None)
    offsets_distant = getattr(trainer, "_last_offsets_distant", None)

    feat_3d_bg = getattr(trainer, "_last_feat_3d_bg", None)
    feat_3d_rigid = getattr(trainer, "_last_feat_3d_rigid", None)
    feat_3d_distant = getattr(trainer, "_last_feat_3d_distant", None)
    feat_2d_bg = getattr(trainer, "_last_feat_2d_bg", None)
    feat_2d_rigid = getattr(trainer, "_last_feat_2d_rigid", None)
    feat_2d_distant = getattr(trainer, "_last_feat_2d_distant", None)
    feat_bg_input = getattr(trainer, "_last_feat_bg_input", None)
    feat_rigid_input = getattr(trainer, "_last_feat_rigid_input", None)
    feat_distant_input = getattr(trainer, "_last_feat_distant_input", None)

    total_loss = result.get("total_loss", torch.tensor(0.0))
    if isinstance(total_loss, torch.Tensor):
        total_loss = float(total_loss.detach().item())

    num_targets = len(batch.get("targets", []))

    return BaselineStep(
        step=step_idx,
        scene_id=scene_id,
        segment_id=segment_id,
        total_loss=total_loss,
        num_targets=num_targets,
        num_bg=int(node_state_bg.means.shape[0]) if node_state_bg is not None else 0,
        num_rigid=int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
        num_distant=int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0,
        node_state_bg_summary=node_state_background_summary(node_state_bg),
        node_state_rigid_summary=node_state_rigid_summary(node_state_rigid),
        node_state_distant_summary=node_state_distant_summary(node_state_distant),
        offset_bg_summary=offsets_summary(offsets_bg),
        offset_rigid_summary=offsets_summary(offsets_rigid),
        offset_distant_summary=offsets_summary(offsets_distant),
        feat_3d_bg_summary=feature_tensor_summary(feat_3d_bg),
        feat_3d_rigid_summary=feature_tensor_summary(feat_3d_rigid),
        feat_3d_distant_summary=feature_tensor_summary(feat_3d_distant),
        feat_2d_bg_summary=feature_tensor_summary(feat_2d_bg),
        feat_2d_rigid_summary=feature_tensor_summary(feat_2d_rigid),
        feat_2d_distant_summary=feature_tensor_summary(feat_2d_distant),
        feat_bg_input_summary=feature_tensor_summary(feat_bg_input),
        feat_rigid_input_summary=feature_tensor_summary(feat_rigid_input),
        feat_distant_input_summary=feature_tensor_summary(feat_distant_input),
        grad_norms=grad_norms_summary(trainer),
    )


def default_baseline_meta(
    cfg: OmegaConf,
    seed: int,
    num_steps: int,
    device: torch.device,
    scene_segment_sequence: List[Tuple[int, int]],
    *,
    scheduler_kwargs: Optional[Dict] = None,
    batch_cache_path: Optional[str] = None,
    batch_plan: Optional[List[Tuple[int, int, int]]] = None,
) -> Dict:
    return {
        "config_path": cfg.get("config_path", None),
        "seed": seed,
        "num_steps": num_steps,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "scene_segment_sequence": scene_segment_sequence,
        "scheduler": scheduler_kwargs or {},
        "batch_cache_path": batch_cache_path,
        "batch_plan": batch_plan,
    }


def save_baseline(baseline: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)


def load_baseline(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --- Comparison -------------------------------------------------------------


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare_step(
    baseline_step: Dict,
    current_step: Dict,
    *,
    loss_rtol: float = 5e-2,
    loss_atol: float = 1e-5,
    stat_rtol: float = 3e-1,
    stat_atol: float = 1e-3,
    grad_rtol: float = 1e-1,
) -> Tuple[bool, str]:
    if baseline_step["scene_id"] != current_step["scene_id"] or baseline_step["segment_id"] != current_step["segment_id"]:
        return False, f"Scene/segment mismatch at step {baseline_step['step']}: expected {(baseline_step['scene_id'], baseline_step['segment_id'])}, got {(current_step['scene_id'], current_step['segment_id'])}"

    if not _close(baseline_step["total_loss"], current_step["total_loss"], loss_rtol, loss_atol):
        return False, f"total_loss mismatch at step {baseline_step['step']}: baseline={baseline_step['total_loss']}, current={current_step['total_loss']}"

    def _compare_summary(base: Optional[Dict], cur: Optional[Dict], name: str, rtol: float, atol: float) -> Tuple[bool, str]:
        if base is None and cur is None:
            return True, ""
        if (base is None) != (cur is None):
            return False, f"{name} presence mismatch"
        for k, v in base.items():
            if isinstance(v, (int, float)):
                if not _close(float(v), float(cur.get(k, 0.0)), rtol, atol):
                    return False, f"{name}.{k} mismatch: {v} vs {cur.get(k, 0.0)}"
            elif isinstance(v, dict):
                cur_sub = cur.get(k, {})
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        if not _close(float(vv), float(cur_sub.get(kk, 0.0)), rtol, atol):
                            return False, f"{name}.{k}.{kk} mismatch: {vv} vs {cur_sub.get(kk, 0.0)}"
                    elif kk == "shape" and isinstance(vv, list):
                        if list(cur_sub.get(kk, [])) != vv:
                            return False, f"{name}.{k}.shape mismatch: {vv} vs {cur_sub.get(kk, [])}"
        return True, ""

    ok, msg = _compare_summary(baseline_step.get("node_state_bg_summary"), current_step.get("node_state_bg_summary"), "bg", stat_rtol, stat_atol)
    if not ok:
        return ok, msg
    ok, msg = _compare_summary(baseline_step.get("node_state_rigid_summary"), current_step.get("node_state_rigid_summary"), "rigid", stat_rtol, stat_atol)
    if not ok:
        return ok, msg
    ok, msg = _compare_summary(baseline_step.get("node_state_distant_summary"), current_step.get("node_state_distant_summary"), "distant", stat_rtol, stat_atol)
    if not ok:
        return ok, msg

    # Value alignment for offset summaries (skip when baseline lacks key for backward compatibility)
    for _offset_key, _name in (
        ("offset_bg_summary", "offset_bg"),
        ("offset_rigid_summary", "offset_rigid"),
        ("offset_distant_summary", "offset_distant"),
    ):
        _base_off = baseline_step.get(_offset_key)
        if _base_off is None:
            continue
        _cur_off = current_step.get(_offset_key)
        if _cur_off is None:
            return False, f"{_offset_key} missing in current step (baseline has it)"
        ok, msg = _compare_summary(_base_off, _cur_off, _name, stat_rtol, stat_atol)
        if not ok:
            return ok, msg

    # Value alignment for all feature summaries (skip when baseline lacks key for backward compatibility)
    _FEAT_SUMMARY_KEYS = (
        "feat_3d_bg_summary", "feat_3d_rigid_summary", "feat_3d_distant_summary",
        "feat_2d_bg_summary", "feat_2d_rigid_summary", "feat_2d_distant_summary",
        "feat_bg_input_summary", "feat_rigid_input_summary", "feat_distant_input_summary",
    )
    for key in _FEAT_SUMMARY_KEYS:
        base_feat = baseline_step.get(key)
        if base_feat is None:
            continue
        cur_feat = current_step.get(key)
        if cur_feat is None:
            return False, f"{key} missing in current step (baseline has it)"
        ok, msg = _compare_summary(base_feat, cur_feat, key.replace("_summary", ""), stat_rtol, stat_atol)
        if not ok:
            return ok, msg

    for key, base_val in baseline_step.get("grad_norms", {}).items():
        cur_val = current_step.get("grad_norms", {}).get(key, 0.0)
        if not _close(float(base_val), float(cur_val), grad_rtol, 0.0):
            return False, f"grad_norms[{key}] mismatch: {base_val} vs {cur_val}"

    return True, ""
