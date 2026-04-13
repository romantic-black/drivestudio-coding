from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.streetforward_assets import StreetForwardAssetStore
from tools.build_streetforward_scene_assets import _build_scene_asset
from tools.streetforward_export_require_full_config import (
    require_full_training_config_for_asset_export,
)
from tools.train_minimal_streetforward_stage4_3_v4_common import build_multi_scene_dataset_v3


def _dynamic_tracks_from_runtime(
    dataset,
    scene_dataset,
    frame_indices: Sequence[int],
    pointcloud: Dict[str, Any],
    world_to_seg0: torch.Tensor,
) -> Dict[str, np.ndarray]:
    mapping = pointcloud.get("instance_mapping")
    exclude: Optional[set[int]] = None
    meta = pointcloud.get("metadata")
    if isinstance(meta, dict):
        raw = meta.get("static_instance_intids")
        if raw:
            exclude = {int(x) for x in raw}
    dynamic_info = dataset._build_dynamic_info(
        scene_dataset=scene_dataset,
        frame_indices=[int(x) for x in frame_indices],
        instance_mapping=mapping,
        world_to_seg0=world_to_seg0,
        exclude_instance_intids=exclude,
    )
    frame_list = sorted(int(x) for x in frame_indices)
    intids: List[int] = sorted(int(k) for k in pointcloud.get("dynamic", {}).keys())
    quats = np.zeros((len(frame_list), len(intids), 4), dtype=np.float32)
    trans = np.zeros((len(frame_list), len(intids), 3), dtype=np.float32)
    fv = np.zeros((len(frame_list), len(intids)), dtype=np.uint8)
    intid_to_col = {intid: i for i, intid in enumerate(intids)}
    if dynamic_info is not None:
        for row, frame_idx in enumerate(frame_list):
            fobj = dynamic_info.get(int(frame_idx), {})
            instances = fobj.get("instances", {})
            for intid, state in instances.items():
                col = intid_to_col.get(int(intid))
                if col is None:
                    continue
                quats[row, col] = np.asarray(state["quat"], dtype=np.float32)
                trans[row, col] = np.asarray(state["trans"], dtype=np.float32)
                fv[row, col] = 1
    return {
        "frame_indices": np.asarray(frame_list, dtype=np.int32),
        "instance_intids": np.asarray(intids, dtype=np.int32),
        "instances_quats": quats,
        "instances_trans": trans,
        "instances_fv": fv,
        "static_instance_intids": np.asarray(sorted(exclude or []), dtype=np.int32),
    }


def _build_segment_asset(
    store: StreetForwardAssetStore,
    dataset,
    *,
    scene_id: int,
    segment_id: int,
    parent_scene_asset_id: str,
) -> str:
    scene_data = dataset._ensure_scene_loaded(int(scene_id))
    if scene_data is None:
        raise ValueError(f"Scene {scene_id} cannot be loaded")
    scene_dataset = scene_data["dataset"]
    sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
    segment = scene_data["segments"][int(segment_id)]
    segment_first_pose, world_to_seg0, seg0_frame_idx, seg0_source = dataset._ensure_segment_pose_cached(
        int(scene_id), int(segment_id), scene_dataset, segment
    )
    pointcloud = dataset._ensure_segment_pointcloud_cached(int(scene_id), int(segment_id), segment_first_pose)
    if not isinstance(pointcloud, dict):
        raise ValueError(
            f"pointcloud missing for scene={scene_id} segment={segment_id}; export requires initialized pointcloud."
        )
    if "background" not in pointcloud:
        raise ValueError("pointcloud.background is required")
    frame_union = sorted(set(sidx.frame_indices) | set(sidx.test_frame_indices))
    dynamic_tracks = _dynamic_tracks_from_runtime(
        dataset,
        scene_dataset,
        frame_indices=frame_union,
        pointcloud=pointcloud,
        world_to_seg0=world_to_seg0,
    )

    dataset_name = str(dataset.data_cfg.get("dataset"))
    train_refs = np.asarray(sidx.train_image_refs or [], dtype=np.int32).reshape(-1, 2)
    test_refs = np.asarray(sidx.test_image_refs or [], dtype=np.int32).reshape(-1, 2)
    stats = {
        "num_train_frames": int(len(sidx.frame_indices)),
        "num_test_frames": int(len(sidx.test_frame_indices)),
        "num_keyframes": int(len(sidx.keyframe_indices)),
        "background_points": int(np.asarray(pointcloud["background"]).shape[0]),
        "dynamic_instances": int(len(pointcloud.get("dynamic", {}))),
        "dynamic_points": int(
            sum(int(np.asarray(v).shape[0]) for v in pointcloud.get("dynamic", {}).values())
        ),
    }
    return store.export_segment_asset(
        dataset=dataset_name,
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        parent_scene_asset_id=str(parent_scene_asset_id),
        segment_index_payload={
            "num_cams": int(sidx.num_cams),
            "frame_indices": [int(x) for x in sidx.frame_indices],
            "test_frame_indices": [int(x) for x in sidx.test_frame_indices],
            "keyframe_indices": [int(x) for x in sidx.keyframe_indices],
            "keyframe_to_frames": {int(k): [int(x) for x in v] for k, v in sidx.keyframe_to_frames.items()},
            "frame_to_keyframe": {int(k): int(v) for k, v in sidx.frame_to_keyframe.items()},
            "segment_first_frame_idx": int(sidx.segment_first_frame_idx),
            "train_image_refs": train_refs,
            "test_image_refs": test_refs,
        },
        segment_pose_payload={
            "segment_first_pose_world": segment_first_pose.detach().cpu().numpy(),
            "world_to_seg0": world_to_seg0.detach().cpu().numpy(),
            "segment_first_frame_idx": int(seg0_frame_idx),
            "segment_pose_source": str(seg0_source),
        },
        pointcloud_payload=pointcloud,
        dynamic_tracks_payload=dynamic_tracks,
        segment_aabb=dataset.segment_aabb_np,
        pointcloud_config_normalized=dict(dataset.pointcloud_config),
        stats=stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build StreetForward segment assets")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--scene_id", type=int, default=None)
    parser.add_argument("--segment_id", type=int, default=None)
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
        scene_ids = [int(x) for x in cfg.data.train_scene_ids]
    elif args.scene_id is not None:
        scene_ids = [int(args.scene_id)]
    else:
        raise ValueError("Provide either --scene_id or --all_train_scenes")

    for scene_id in scene_ids:
        dataset_name = str(dataset.data_cfg.get("dataset"))
        try:
            scene_handle = store.get_scene_asset(dataset_name, int(scene_id))
            scene_asset_id = str(scene_handle.load_manifest()["asset_id"])
        except Exception:
            scene_asset_id = _build_scene_asset(store, dataset, scene_id)
        scene_data = dataset._ensure_scene_loaded(int(scene_id))
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segment_ids = (
            [int(args.segment_id)]
            if args.segment_id is not None and args.scene_id is not None
            else list(range(len(scene_data.get("segments", []))))
        )
        for seg_id in segment_ids:
            if store.has_segment_asset(dataset_name, int(scene_id), int(seg_id)):
                existing = store.get_segment_asset(dataset_name, int(scene_id), int(seg_id))
                existing_id = str(existing.load_manifest()["asset_id"])
                print(
                    f"[segment-asset] scene_id={scene_id} segment_id={seg_id} asset_id={existing_id} "
                    f"(existing=true parent_scene_asset_id={scene_asset_id})"
                )
                continue
            asset_id = _build_segment_asset(
                store,
                dataset,
                scene_id=int(scene_id),
                segment_id=int(seg_id),
                parent_scene_asset_id=scene_asset_id,
            )
            print(
                f"[segment-asset] scene_id={scene_id} segment_id={seg_id} asset_id={asset_id} "
                f"(parent_scene_asset_id={scene_asset_id})"
            )


if __name__ == "__main__":
    main()
