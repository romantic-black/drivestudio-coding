import os
import json
import argparse
from typing import Optional
import torch
import numpy as np
from omegaconf import OmegaConf


def _save_mask_images(batch: dict, save_dir: str) -> None:
    raise RuntimeError(
        "Point-coverage mask has been removed from the codebase; mask dumping is no longer supported."
    )


def _save_target_images(batch: dict, save_dir: str) -> None:
    """Save each target image as RGB PNG for inspection."""
    target = batch.get("target")
    if target is None:
        return
    images = target.get("image")
    if images is None:
        return
    images_dir = os.path.join(save_dir, "target_images")
    os.makedirs(images_dir, exist_ok=True)
    for i in range(images.shape[0]):
        img = images[i]
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        path = os.path.join(images_dir, f"view_{i:03d}.png")
        try:
            from PIL import Image
            Image.fromarray(img, mode="RGB").save(path)
        except ImportError:
            np.save(path.replace(".png", ".npy"), img)
    print(f"Saved {images.shape[0]} target images to {images_dir}")

def _print_pointcloud_stats(
    batch: dict,
    *,
    near_max_points: Optional[int] = None,
    distant_max_points: Optional[int] = None,
) -> None:
    pc = batch.get("pointcloud")
    if pc is None:
        print("[batch] pointcloud: <missing>")
        return

    if not isinstance(pc, dict):
        pts = getattr(pc, "points", None)
        n = int(pts.shape[0]) if pts is not None and hasattr(pts, "shape") else -1
        print(f"[batch] pointcloud: non-dict type={type(pc)} points={n}")
        return

    bg = pc.get("background", None)
    bg_n = int(bg.shape[0]) if bg is not None and hasattr(bg, "shape") else 0
    dyn = pc.get("dynamic", {}) or {}
    dyn_instances = len(dyn) if isinstance(dyn, dict) else 0
    dyn_points = 0
    if isinstance(dyn, dict):
        for _, arr in dyn.items():
            if arr is not None and hasattr(arr, "shape"):
                dyn_points += int(arr.shape[0])

    print(f"[batch] pointcloud.background: N={bg_n}")
    print(f"[batch] pointcloud.dynamic: instances={dyn_instances} total_points={dyn_points}")

    # StreetForward/MultiScene split logic: segment_aabb (near) vs outside (distant), in seg0 coords.
    seg_aabb = batch.get("aabb", None)
    if bg is None or not hasattr(bg, "shape") or bg_n == 0 or seg_aabb is None:
        return

    try:
        aabb_np = seg_aabb.detach().cpu().numpy() if torch.is_tensor(seg_aabb) else np.asarray(seg_aabb)
        crop_min = aabb_np[0].astype(np.float32)
        crop_max = aabb_np[1].astype(np.float32)
        bg_np = bg.detach().cpu().numpy() if torch.is_tensor(bg) else np.asarray(bg)
        xyz = bg_np[:, :3].astype(np.float32)
        in_crop = ((xyz >= crop_min) & (xyz <= crop_max)).all(axis=1)
        near_n = int(in_crop.sum())
        distant_n = int((~in_crop).sum())
        print(f"[batch] pointcloud split by segment_aabb: near={near_n} distant={distant_n}")
    except Exception as e:
        print(f"[batch] pointcloud split failed: {e}")
        return

    if near_max_points is not None or distant_max_points is not None:
        near_after = min(near_n, int(near_max_points)) if near_max_points is not None else near_n
        distant_after = min(distant_n, int(distant_max_points)) if distant_max_points is not None else distant_n
        print(
            f"[batch] caps: near_max_points={near_max_points} distant_max_points={distant_max_points} "
            f"=> expected_after_cap near={near_after} distant={distant_after}"
        )


def capture_batch(cfg, scene_id: int, segment_id: int, save_dir: str):
    """提取单个 batch 并持久化到磁盘"""
    # NOTE: Heavy dataset deps (e.g. pytorch3d) are imported lazily so that
    # `load_batch()` can be used in lightweight environments.
    from datasets.multi_scene_dataset import MultiSceneDataset

    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Initializing Dataset for scene {scene_id}...")
    # 1. 最小化初始化 Dataset
    dataset = MultiSceneDataset(
        data_cfg=cfg.data,
        train_scene_ids=[scene_id],
        eval_scene_ids=[],
        num_source_keyframes=cfg.dataset.num_source_keyframes,
        num_target_keyframes=cfg.dataset.num_target_keyframes,
        segment_overlap_ratio=cfg.dataset.get("segment_overlap_ratio", 0.2),
        keyframe_split_config=cfg.dataset.get("keyframe_split_config", None),
        min_keyframes_per_scene=cfg.dataset.get("min_keyframes_per_scene", 2),
        min_keyframes_per_segment=cfg.dataset.get("min_keyframes_per_segment", 2),
        device=torch.device("cpu"),
        preload_scene_count=1,
        segment_aabb=cfg.dataset.segment_aabb,
        pointcloud_config=cfg.dataset.get("pointcloud", None)
    )
    dataset.initialize()
    
    # 2. 获取 Batch
    print(f"Capturing batch for scene {scene_id}, segment {segment_id}...")
    batch = dataset.get_segment_batch(scene_id=scene_id, segment_id=segment_id, include_test=True)
    pc_cfg = cfg.dataset.get("pointcloud", {}) if hasattr(cfg, "dataset") else {}
    near_cap = pc_cfg.get("near_max_points", None) if hasattr(pc_cfg, "get") else None
    distant_cap = pc_cfg.get("distant_max_points", None) if hasattr(pc_cfg, "get") else None
    _print_pointcloud_stats(batch, near_max_points=near_cap, distant_max_points=distant_cap)
    
    # 3. 核心数据持久化
    batch_path = os.path.join(save_dir, f"scene{scene_id}_seg{segment_id}_batch.pt")
    torch.save(batch, batch_path)
    
    # 4. 保存易读的元数据 (Meta Info)
    meta_info = {
        "scene_id": int(batch['scene_id'].item() if torch.is_tensor(batch['scene_id']) else batch['scene_id']),
        "segment_id": batch['segment_id'],
        "keyframe_info": batch.get('keyframe_info', {}),
        "has_pointcloud": 'pointcloud' in batch,
        "has_test_views": 'test' in batch
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta_info, f, indent=4)

    # 5. 保存所有 target 图像便于查看
    _save_target_images(batch, save_dir)

    print(f"Batch successfully saved to {batch_path}")
    print(f"Metadata saved to {os.path.join(save_dir, 'meta.json')}")
    return batch_path


def load_batch(batch_path: str):
    """快速读取存盘的 Batch"""
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Overfit batch not found at {batch_path}")
    batch = torch.load(batch_path, map_location="cpu", weights_only=False)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a single batch for overfitting")
    parser.add_argument("--config_file", type=str, required=True, help="Path to config file")
    args, extra_args = parser.parse_known_args()

    # Load configuration
    cfg = OmegaConf.load(args.config_file)
    
    # Ensure dataset preset/type is properly handled if not directly in config
    if "data" in cfg and "dataset" not in cfg.data and "dataset_preset" in cfg.data:
        dataset_preset = cfg.data.dataset_preset
        preset_path = os.path.join("configs", "datasets", f"{dataset_preset}.yaml")
        if os.path.exists(preset_path):
            dataset_cfg = OmegaConf.load(preset_path)
            cfg = OmegaConf.merge(cfg, dataset_cfg)
            cfg.data["dataset_preset"] = dataset_preset
    
    # Setup capture parameters
    capture_cfg = cfg.get("capture", {})
    scene_id = capture_cfg.get("scene_id", 0)
    segment_id = capture_cfg.get("segment_id", 0)
    save_dir = capture_cfg.get("save_dir", "./data/overfit_batches")
    
    capture_batch(cfg, scene_id, segment_id, save_dir)
