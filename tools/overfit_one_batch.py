import os
import json
import argparse
import torch
from omegaconf import OmegaConf


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
        segment_input_aabb=cfg.dataset.segment_input_aabb,
        pointcloud_config=cfg.dataset.get("pointcloud", None)
    )
    dataset.initialize()
    
    # 2. 获取 Batch
    print(f"Capturing batch for scene {scene_id}, segment {segment_id}...")
    batch = dataset.get_segment_batch(scene_id=scene_id, segment_id=segment_id, include_test=True)
    
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
