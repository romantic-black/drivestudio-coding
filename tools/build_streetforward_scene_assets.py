from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.streetforward_assets import StreetForwardAssetStore
from tools.streetforward_export_require_full_config import (
    require_full_training_config_for_asset_export,
)
from tools.train_minimal_streetforward_stage4_3_v4_common import build_multi_scene_dataset_v3


def _flatten_segments(scene_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    segments = list(scene_data.get("segments", []))
    segment_ids = []
    seg_frame_flat = []
    seg_frame_offsets = [0]
    seg_kf_flat = []
    seg_kf_offsets = [0]
    for sid, seg in enumerate(segments):
        segment_ids.append(int(sid))
        frames = [int(x) for x in seg.get("frame_indices", [])]
        kfs = [int(x) for x in seg.get("keyframe_indices", [])]
        seg_frame_flat.extend(frames)
        seg_kf_flat.extend(kfs)
        seg_frame_offsets.append(len(seg_frame_flat))
        seg_kf_offsets.append(len(seg_kf_flat))
    return {
        "segment_ids": np.asarray(segment_ids, dtype=np.int32),
        "segment_frame_indices_flat": np.asarray(seg_frame_flat, dtype=np.int32),
        "segment_frame_offsets": np.asarray(seg_frame_offsets, dtype=np.int64),
        "segment_keyframe_indices_flat": np.asarray(seg_kf_flat, dtype=np.int32),
        "segment_keyframe_offsets": np.asarray(seg_kf_offsets, dtype=np.int64),
    }


def _flatten_keyframe_segments(scene_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    keyframe_segments = scene_data.get("keyframe_segments", [])
    keyframe_indices = np.asarray(list(range(len(keyframe_segments))), dtype=np.int32)
    flat = []
    offsets = [0]
    for seg in keyframe_segments:
        vals = [int(x) for x in seg]
        flat.extend(vals)
        offsets.append(len(flat))
    return {
        "keyframe_indices": keyframe_indices,
        "keyframe_to_frames_flat": np.asarray(flat, dtype=np.int32),
        "keyframe_to_frames_offsets": np.asarray(offsets, dtype=np.int64),
    }


def _to_4x4_np(mat: Any) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 3):
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = arr
        return out
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = arr
        return out
    raise ValueError(f"Unsupported matrix shape for 4x4 conversion: {arr.shape}")


def _build_image_table_rows(dataset, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    scene_dataset = scene_data["dataset"]
    pixel_source = scene_dataset.pixel_source
    if pixel_source is None:
        raise ValueError("scene_dataset.pixel_source is required for image_table export")
    train_set = set(int(x) for x in scene_data["train_frame_indices"])
    test_set = set(int(x) for x in scene_data["test_frame_indices"])
    rows: List[Dict[str, Any]] = []
    seen = set()
    num_frames = int(scene_dataset.num_img_timesteps)
    for cam_id in list(pixel_source.camera_list):
        cam_data = pixel_source.camera_data[int(cam_id)]
        for frame_idx in range(num_frames):
            key = (int(frame_idx), int(cam_id))
            if key in seen:
                raise ValueError(f"Duplicate image table key detected: {key}")
            seen.add(key)
            c2w = _to_4x4_np(cam_data.cam_to_worlds[frame_idx])
            intr = _to_4x4_np(cam_data.intrinsics[frame_idx])
            image_path = (
                str(cam_data.img_filepaths[frame_idx])
                if hasattr(cam_data, "img_filepaths") and len(cam_data.img_filepaths) > frame_idx
                else ""
            )
            depth_path = (
                str(cam_data.depth_filepaths[frame_idx])
                if hasattr(cam_data, "depth_filepaths") and len(cam_data.depth_filepaths) > frame_idx
                else ""
            )
            sky_mask_path = (
                str(cam_data.sky_mask_filepaths[frame_idx])
                if hasattr(cam_data, "sky_mask_filepaths") and len(cam_data.sky_mask_filepaths) > frame_idx
                else ""
            )
            dynamic_mask_path = (
                str(cam_data.dynamic_mask_filepaths[frame_idx])
                if hasattr(cam_data, "dynamic_mask_filepaths")
                and len(cam_data.dynamic_mask_filepaths) > frame_idx
                else ""
            )
            for p in (image_path, depth_path, sky_mask_path, dynamic_mask_path):
                if p and not os.path.exists(p):
                    raise ValueError(
                        f"image_table export path missing: scene={scene_data.get('scene_id', 'unknown')} "
                        f"frame={frame_idx} cam={cam_id} path={p}"
                    )
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "cam_id": int(cam_id),
                    "img_idx": int(frame_idx * int(scene_dataset.num_cams) + int(cam_id)),
                    "is_train": bool(frame_idx in train_set),
                    "is_test": bool(frame_idx in test_set),
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "sky_mask_path": sky_mask_path,
                    "dynamic_mask_path": dynamic_mask_path,
                    "height": int(cam_data.HEIGHT),
                    "width": int(cam_data.WIDTH),
                    "intrinsic_4x4_flat": intr.reshape(-1).astype(np.float32).tolist(),
                    "camera_to_world_flat": c2w.reshape(-1).astype(np.float32).tolist(),
                }
            )
    return rows


def _build_scene_asset(store: StreetForwardAssetStore, dataset, scene_id: int) -> str:
    scene_data = dataset._ensure_scene_loaded(int(scene_id))
    if scene_data is None:
        raise ValueError(f"Scene {scene_id} cannot be loaded")
    scene_dataset = scene_data["dataset"]
    dataset_name = str(dataset.data_cfg.get("dataset"))

    split_cfg = {
        "test_image_stride": int(getattr(dataset.data_cfg.pixel_source, "test_image_stride", 0)),
        "max_test_images": int(getattr(dataset.data_cfg.pixel_source, "max_test_images", 0)),
        "segment_overlap_ratio": float(dataset.segment_overlap_ratio),
        "keyframe_split_config": dict(dataset.keyframe_split_config),
        "min_keyframes_per_scene": int(dataset.min_keyframes_per_scene),
        "min_keyframes_per_segment": int(dataset.min_keyframes_per_segment),
    }
    scene_index_arrays = {
        "scene_id": np.asarray([int(scene_id)], dtype=np.int32),
        "train_frame_indices": np.asarray(scene_data["train_frame_indices"], dtype=np.int32),
        "test_frame_indices": np.asarray(scene_data["test_frame_indices"], dtype=np.int32),
        **_flatten_keyframe_segments(scene_data),
        **_flatten_segments(scene_data),
    }
    image_table_rows = _build_image_table_rows(dataset, scene_data)
    scene_name = f"{int(scene_id):06d}"
    return store.export_scene_asset(
        dataset=dataset_name,
        scene_id=int(scene_id),
        scene_name=scene_name,
        num_frames=int(scene_data["num_frames"]),
        num_cams=int(scene_dataset.num_cams),
        split_config=split_cfg,
        scene_index_arrays=scene_index_arrays,
        image_table_rows=image_table_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build StreetForward scene assets")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--scene_id", type=int, default=None)
    parser.add_argument("--all_train_scenes", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    require_full_training_config_for_asset_export(cfg)
    if cfg.data.get("assets") is not None:
        cfg.data.assets.enable = False
    device = torch.device("cpu")
    dataset = build_multi_scene_dataset_v3(cfg, device=device)
    dataset.initialize()

    assets_cfg = cfg.data.get("assets")
    if assets_cfg is None or assets_cfg.get("root") is None:
        raise ValueError("data.assets.root is required for export scripts.")
    store = StreetForwardAssetStore(str(assets_cfg.root), missing_policy="error")

    if args.all_train_scenes:
        scene_ids: List[int] = [int(x) for x in cfg.data.train_scene_ids]
    elif args.scene_id is not None:
        scene_ids = [int(args.scene_id)]
    else:
        raise ValueError("Provide either --scene_id or --all_train_scenes")

    for scene_id in scene_ids:
        asset_id = _build_scene_asset(store, dataset, scene_id)
        print(f"[scene-asset] scene_id={scene_id} asset_id={asset_id}")


if __name__ == "__main__":
    main()
