import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from .base import RGBPointCloudGenerator

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)


class LiDARRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    LiDAR RGB point cloud generator.

    For each frame, loads LiDAR points, colors them using multi-camera images,
    and separates static/dynamic points using instance boxes.
    """

    def __init__(
        self,
        sparsity: Literal["Drop90", "Drop80", "Drop50", "Drop25", "full"] = "full",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            sparsity=sparsity,
            filter_sky=False,
            depth_consistency=False,
            downscale=1,
            device=device,
        )

    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
        segment_first_pose=None,
    ) -> Dict:
        scene_data = dataset.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")

        world_to_seg0 = self._compute_world_to_seg0(segment_first_pose)
        segment = scene_data["segments"][segment_id]
        frame_indices = sorted(list(set(segment["frame_indices"])))
        frame_indices = self._apply_sparsity_filter(frame_indices)
        if len(frame_indices) == 0:
            raise ValueError("No frames selected after sparsity filtering")

        instance_mapping, instances_by_frame = self._get_instances_for_segment(
            dataset, scene_id, segment_id, frame_indices, world_to_seg0
        )
        frame_to_instances = {
            frame_idx: instances_by_frame[i]
            for i, frame_idx in enumerate(frame_indices)
        }

        all_backgrounds: List[np.ndarray] = []
        all_dynamic_objects: Dict[int, List[np.ndarray]] = defaultdict(list)
        _agent_first_frame_logged = False

        for frame_idx in frame_indices:
            points_world, points_vehicle = self._load_lidar_points(
                dataset, scene_id, frame_idx
            )
            if points_world is None or len(points_world) == 0:
                continue

            points_vehicle_rgb, points_world_rgb = self._colorize_lidar_points(
                dataset, scene_id, frame_idx, points_vehicle, points_world
            )

            if points_world_rgb is None:
                continue

            points_world_frame = points_world_rgb[:, :3]
            colors_frame = points_world_rgb[:, 3:]
            points_world_frame = self._transform_points_np(points_world_frame, world_to_seg0)

            frame_instances = frame_to_instances.get(frame_idx, [])
            background_frame, dynamic_frame = self._separate_static_dynamic(
                points_world_frame, colors_frame, frame_instances
            )
            all_backgrounds.append(background_frame)
            for intid, pts in dynamic_frame.items():
                all_dynamic_objects[intid].append(pts)

        background = (
            np.concatenate(all_backgrounds, axis=0)
            if len(all_backgrounds) > 0
            else np.zeros((0, 6), dtype=np.float32)
        )
        # #region agent log
        try:
            import json
            _bg = background[:, :3]
            _log = {"sessionId": "9dda96", "runId": "run1", "hypothesisId": "H1_H3", "location": "lidar.py:generate_pointcloud", "message": "final background extent", "data": {"background_count": int(len(background)), "background_seg0_min": _bg.min(axis=0).tolist() if len(_bg) > 0 else None, "background_seg0_max": _bg.max(axis=0).tolist() if len(_bg) > 0 else None}, "timestamp": __import__("time").time() * 1000}
            with open("/root/drivestudio-coding/.cursor/debug-9dda96.log", "a") as _f:
                _f.write(json.dumps(_log, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion
        dynamic_objects = {
            intid: np.concatenate(points_list, axis=0)
            for intid, points_list in all_dynamic_objects.items()
            if len(points_list) > 0
        }

        # Single-stage filtering; distant/near 划分交给上层根据 segment_aabb 完成
        if len(background) > 0:
            background_pts, background_colors = background[:, :3], background[:, 3:]
            background_pts, background_colors = self.filter_pointcloud(
                background_pts, background_colors, strict=True
            )
            background = (
                np.concatenate([background_pts, background_colors], axis=1)
                if len(background_pts) > 0
                else np.zeros((0, 6), dtype=np.float32)
            )

        metadata = {
            "type": "lidar",
            "frame_indices": frame_indices,
            "frames_used": len(all_backgrounds),
            "sparsity": self.sparsity,
        }

        return {
            "background": background,
            "dynamic": dynamic_objects,
            "instance_mapping": instance_mapping,
            "metadata": metadata,
        }

    def _apply_sparsity_filter(self, frame_indices: List[int]) -> List[int]:
        if self.sparsity == "full":
            return frame_indices

        filtered: List[int] = []
        for frame_pos, frame_idx in enumerate(frame_indices):
            if self.sparsity == "Drop50":
                if frame_pos % 4 in (2, 3):
                    continue
            elif self.sparsity == "Drop80":
                if frame_pos % 5 != 0:
                    continue
            elif self.sparsity == "Drop25":
                if frame_pos % 4 == 2:
                    continue
            elif self.sparsity == "Drop90":
                if frame_pos % 10 != 0:
                    continue
            filtered.append(frame_idx)
        return filtered

    def _load_lidar_points(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        frame_idx: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        scene_data = dataset.get_scene(scene_id)
        if scene_data is None:
            return None, None

        scene_dataset = scene_data["dataset"]
        lidar_source = getattr(scene_dataset, "lidar_source", None)
        if lidar_source is None:
            return None, None

        required_attrs = ["origins", "directions", "ranges", "timesteps"]
        if not all(hasattr(lidar_source, attr) for attr in required_attrs):
            return None, None

        frame_indices_tensor = torch.tensor(
            [frame_idx], dtype=lidar_source.timesteps.dtype, device=lidar_source.timesteps.device
        )
        mask = torch.isin(lidar_source.timesteps, frame_indices_tensor)
        if not mask.any():
            return None, None

        origins = lidar_source.origins[mask]
        directions = lidar_source.directions[mask]
        ranges = lidar_source.ranges[mask]
        points_world = origins + directions * ranges
        points_world_np = points_world.cpu().numpy().astype(np.float32)

        points_vehicle_np = None
        if (
            lidar_source.lidar_to_worlds is not None
            and frame_idx < lidar_source.lidar_to_worlds.shape[0]
        ):
            T_vw = lidar_source.lidar_to_worlds[frame_idx].cpu().numpy()
            T_wv = np.linalg.inv(T_vw)
            points_world_homo = np.concatenate(
                [points_world_np, np.ones((points_world_np.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            points_vehicle = (T_wv @ points_world_homo.T).T[:, :3]
            points_vehicle_np = points_vehicle.astype(np.float32)

        return points_world_np, points_vehicle_np

    def _colorize_lidar_points(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        frame_idx: int,
        points_vehicle: Optional[np.ndarray],
        points_world: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if points_vehicle is None and points_world is None:
            return None, None

        scene_data = dataset.get_scene(scene_id)
        if scene_data is None:
            return None, None
        scene_dataset = scene_data["dataset"]
        pixel_source = scene_dataset.pixel_source
        lidar_source = getattr(scene_dataset, "lidar_source", None)

        if points_vehicle is None and points_world is not None and lidar_source is not None:
            # Derive vehicle coords from world if needed
            if frame_idx < lidar_source.lidar_to_worlds.shape[0]:
                T_vw = lidar_source.lidar_to_worlds[frame_idx].cpu().numpy()
                T_wv = np.linalg.inv(T_vw)
                points_world_homo = np.concatenate(
                    [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1
                )
                points_vehicle = (T_wv @ points_world_homo.T).T[:, :3].astype(np.float32)

        if points_world is None and points_vehicle is not None and lidar_source is not None:
            if frame_idx < lidar_source.lidar_to_worlds.shape[0]:
                T_vw = lidar_source.lidar_to_worlds[frame_idx].cpu().numpy()
                points_vehicle_homo = np.concatenate(
                    [points_vehicle, np.ones((points_vehicle.shape[0], 1), dtype=np.float32)], axis=1
                )
                points_world = (T_vw @ points_vehicle_homo.T).T[:, :3].astype(np.float32)

        if points_vehicle is None or points_world is None:
            return None, None

        colors = np.zeros((points_vehicle.shape[0], 3), dtype=np.float32)
        colored_mask = np.zeros(points_vehicle.shape[0], dtype=bool)

        # Use lidar_to_worlds as ego pose
        T_vw = None
        if lidar_source is not None and lidar_source.lidar_to_worlds is not None:
            if frame_idx < lidar_source.lidar_to_worlds.shape[0]:
                T_vw = lidar_source.lidar_to_worlds[frame_idx].cpu().numpy()
        if T_vw is None:
            T_vw = np.eye(4, dtype=np.float32)

        for cam_id in pixel_source.camera_list:
            unique_cam_idx = pixel_source.camera_data[cam_id].unique_cam_idx
            img_idx = frame_idx * pixel_source.num_cams + unique_cam_idx
            try:
                image_infos, cam_infos = pixel_source.get_image(img_idx)
            except Exception as exc:
                logger.debug(
                    "Skip colorizing frame %s cam %s due to load failure: %s",
                    frame_idx,
                    cam_id,
                    exc,
                )
                continue

            image = image_infos["pixels"]
            image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            H, W = image_np.shape[:2]

            intrinsic = cam_infos["intrinsics"]
            intrinsic_np = intrinsic.cpu().numpy() if isinstance(intrinsic, torch.Tensor) else intrinsic
            if intrinsic_np.shape[0] == 4:
                intrinsic_np = intrinsic_np[:3, :3]
            extrinsic = cam_infos["camera_to_world"]
            extrinsic_np = extrinsic.cpu().numpy() if isinstance(extrinsic, torch.Tensor) else extrinsic
            T_cw = np.linalg.inv(extrinsic_np)

            points_world_homo = np.concatenate(
                [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1
            )
            points_cam = (T_cw @ points_world_homo.T).T[:, :3]
            z = points_cam[:, 2]
            valid = z > 1e-6
            if not np.any(valid):
                continue

            # #region agent log
            _n = z.shape[0]
            _n_z_le0 = int(np.sum(z <= 0))
            _n_z_tiny = int(np.sum((z > 0) & (z <= 1e-6)))
            if _n_z_le0 > 0 or _n_z_tiny > 0:
                try:
                    _f = open("/root/drivestudio-coding/.cursor/debug.log", "a")
                    _f.write('{"id":"lidar_z","timestamp":' + str(int(__import__("time").time() * 1000)) + ',"location":"lidar.py:325","message":"z invalid counts","data":{"n_total":' + str(_n) + ',"n_z_le0":' + str(_n_z_le0) + ',"n_z_tiny":' + str(_n_z_tiny) + '},"hypothesisId":"H1"}\n')
                    _f.close()
                except Exception:
                    pass
            # #endregion

            uv = (intrinsic_np @ points_cam.T).T
            uv_xy = uv[:, :2]

            # Only perform perspective divide on valid points to avoid divide-by-zero
            uv = np.zeros_like(uv_xy, dtype=np.float32)
            uv[valid] = uv_xy[valid] / z[valid, None]
            # #region agent log
            _uv_nan = int(np.isnan(uv).any(axis=1).sum())
            _uv_inf = int(np.isinf(uv).any(axis=1).sum())
            if _uv_nan > 0 or _uv_inf > 0:
                try:
                    _f = open("/root/drivestudio-coding/.cursor/debug.log", "a")
                    _f.write('{"id":"lidar_uv","timestamp":' + str(int(__import__("time").time() * 1000)) + ',"location":"lidar.py:uv","message":"uv nan/inf after divide","data":{"n_uv_nan":' + str(_uv_nan) + ',"n_uv_inf":' + str(_uv_inf) + '},"hypothesisId":"H2"}\n')
                    _f.close()
                except Exception:
                    pass
            # #endregion
            u = np.round(uv[:, 0]).astype(int)
            v = np.round(uv[:, 1]).astype(int)
            in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not np.any(in_img):
                continue

            sample_idx = np.where(in_img & (~colored_mask))[0]
            if len(sample_idx) == 0:
                sample_idx = np.where(in_img)[0]
            if len(sample_idx) == 0:
                continue

            sampled_colors = image_np[v[sample_idx], u[sample_idx]]
            # 修复：更准确地判断颜色范围，避免重复乘以255
            # 如果最大值 <= 1.0 + 1e-3，说明是 [0, 1] 范围，需要乘以 255
            # 但如果最大值 > 1.0 + 1e-3，说明已经是 [0, 255] 范围，不需要乘以 255
            if sampled_colors.max() <= 1.0 + 1e-3:
                sampled_colors = sampled_colors * 255.0
            else:
                # 确保颜色值在 [0, 255] 范围内（防止超出范围的值）
                sampled_colors = np.clip(sampled_colors, 0.0, 255.0)
            colors[sample_idx] = sampled_colors.astype(np.float32)
            colored_mask[sample_idx] = True

        points_vehicle_rgb = np.concatenate([points_vehicle, colors], axis=1)
        points_world_rgb = np.concatenate([points_world, colors], axis=1)
        return points_vehicle_rgb.astype(np.float32), points_world_rgb.astype(np.float32)

    def _get_instances_for_segment(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
        frame_indices: List[int],
        world_to_seg0: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[int, int], List[List[Dict]]]:
        # Reuse monocular logic to stay in sync（不再传递 crop_aabb/input_aabb；AABB 由上层基于 segment_aabb 控制）
        from .monocular import MonocularRGBPointCloudGenerator

        helper = MonocularRGBPointCloudGenerator(
            chosen_cam_ids=[],
            sparsity="full",
            filter_sky=False,
            depth_consistency=False,
            downscale=1,
            dynamic_filter=False,
            device=self.device,
        )
        return helper._get_instances_for_segment(
            dataset, scene_id, segment_id, frame_indices, world_to_seg0
        )
