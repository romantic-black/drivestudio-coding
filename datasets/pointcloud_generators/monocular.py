import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import torch

from .base import RGBPointCloudGenerator
from .motion_utils import compute_static_instance_intids

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)


class MonocularRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    RGB point cloud generator that back-projects monocular depth from MultiSceneDataset.
    Generates static background points by back-projecting monocular depth.

    Dynamic handling (fast-fail):
    - Pixel-domain dynamic filtering is still used to keep background clean.
    - Optional dynamic recovery extracts per-instance points by 3D bbox from
      monocular back-projected points.
    """

    def __init__(
        self,
        chosen_cam_ids: List[int] = [0],
        sparsity: Literal["Drop90", "Drop80", "Drop50", "Drop25", "full"] = "full",
        filter_sky: bool = True,
        depth_consistency: bool = True,
        downscale: int = 2,
        dynamic_filter: bool = True,
        dynamic_recovery_enable: bool = False,
        dynamic_recovery_bbox_expand_xyz_m: Optional[List[float]] = None,
        dynamic_recovery_max_points_per_instance: Optional[int] = None,
        dynamic_recovery_assignment: Literal["first_hit", "nearest_center"] = "first_hit",
        static_instance_motion_enable: bool = False,
        static_instance_motion_traj_length_thresh_m: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            sparsity=sparsity,
            filter_sky=filter_sky,
            depth_consistency=depth_consistency,
            downscale=downscale,
            device=device,
        )
        self.chosen_cam_ids = chosen_cam_ids
        self.dynamic_filter = bool(dynamic_filter)
        # Dynamic mask is fixed to coarse dynamic masks to simplify configuration.
        self.dynamic_mask_key = "dynamic_masks"
        self.dynamic_recovery_enable = bool(dynamic_recovery_enable)
        self.dynamic_recovery_assignment = str(dynamic_recovery_assignment)
        self.static_instance_motion_enable = bool(static_instance_motion_enable)
        self.static_instance_motion_traj_length_thresh_m = static_instance_motion_traj_length_thresh_m
        if self.static_instance_motion_enable and self.static_instance_motion_traj_length_thresh_m is None:
            raise ValueError(
                "MonocularRGBPointCloudGenerator: static_instance_motion_enable=true requires "
                "static_instance_motion_traj_length_thresh_m."
            )
        if self.dynamic_recovery_assignment not in ("first_hit", "nearest_center"):
            raise ValueError(
                "dynamic_recovery_assignment must be one of ['first_hit', 'nearest_center']."
            )
        if self.dynamic_recovery_enable:
            if dynamic_recovery_bbox_expand_xyz_m is None:
                raise ValueError(
                    "dynamic_recovery_enable=true requires dynamic_recovery_bbox_expand_xyz_m."
                )
            bbox_expand = np.asarray(dynamic_recovery_bbox_expand_xyz_m, dtype=np.float32).reshape(-1)
            if bbox_expand.shape[0] != 3:
                raise ValueError(
                    "dynamic_recovery_bbox_expand_xyz_m must contain 3 values: [dx, dy, dz]."
                )
            if np.any(bbox_expand < 0):
                raise ValueError("dynamic_recovery_bbox_expand_xyz_m must be non-negative.")
            if dynamic_recovery_max_points_per_instance is None:
                raise ValueError(
                    "dynamic_recovery_enable=true requires dynamic_recovery_max_points_per_instance."
                )
            self.dynamic_recovery_max_points_per_instance = int(dynamic_recovery_max_points_per_instance)
            if self.dynamic_recovery_max_points_per_instance <= 0:
                raise ValueError("dynamic_recovery_max_points_per_instance must be > 0.")
            self.dynamic_recovery_bbox_expand_xyz_m = bbox_expand
        else:
            self.dynamic_recovery_bbox_expand_xyz_m = np.zeros((3,), dtype=np.float32)
            self.dynamic_recovery_max_points_per_instance = None

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

        scene_dataset = scene_data["dataset"]
        pixel_source = getattr(scene_dataset, "pixel_source", None)
        static_instance_intids: Set[int] = set()
        if self.static_instance_motion_enable:
            static_instance_intids = compute_static_instance_intids(
                pixel_source,
                frame_indices,
                float(self.static_instance_motion_traj_length_thresh_m),
            )

        frame_data_by_camera: Dict[int, List[Tuple[int, Dict]]] = {
            cam_id: [] for cam_id in self.chosen_cam_ids
        }
        for frame_idx in frame_indices:
            for cam_id in self.chosen_cam_ids:
                frame_data = self._load_frame_data(dataset, scene_id, frame_idx, cam_id)
                if frame_data is not None:
                    frame_data_by_camera[cam_id].append((frame_idx, frame_data))

        # Find first valid frame to set image size
        H = W = None
        for _, frames in frame_data_by_camera.items():
            if frames:
                sample_img = frames[0][1]["image"]
                H, W = sample_img.shape[:2]
                break
        if H is None or W is None:
            raise ValueError("No valid frame data loaded for monocular generator")

        downscale_mask = None
        if self.downscale != 1:
            downscale_mask = np.zeros((H, W), dtype=bool)
            downscale_mask[:: self.downscale, :: self.downscale] = True

        consistency_masks_by_camera: Dict[int, List[np.ndarray]] = {}
        sorted_frame_data_by_camera: Dict[int, List[Tuple[int, Dict]]] = {}
        for cam_id, frames in frame_data_by_camera.items():
            frames_sorted = sorted(frames, key=lambda x: x[0])
            sorted_frame_data_by_camera[cam_id] = frames_sorted
            frame_data_list = [fd for _, fd in frames_sorted]
            if len(frame_data_list) == 0:
                consistency_masks_by_camera[cam_id] = []
                continue
            if self.depth_consistency:
                consistency_masks_by_camera[cam_id] = self._depth_consistency_check(
                    frame_data_list, H, W
                )
            else:
                consistency_masks_by_camera[cam_id] = [
                    np.ones((H, W), dtype=bool) for _ in frame_data_list
                ]

        all_backgrounds: List[np.ndarray] = []
        all_dynamic_objects: Dict[int, List[np.ndarray]] = defaultdict(list)
        dynamic_recovery_frame_count = 0
        dynamic_recovered_points_before_cap = 0

        use_instance_policy = bool(self.dynamic_recovery_enable or self.static_instance_motion_enable)
        if use_instance_policy:
            instance_mapping, instances_by_frame = self._get_instances_for_segment(
                dataset, scene_id, segment_id, frame_indices, world_to_seg0
            )
            frame_to_instances = {
                frame_idx: instances_by_frame[i] for i, frame_idx in enumerate(frame_indices)
            }
        else:
            instance_mapping = {}
            frame_to_instances = {}

        for frame_idx in frame_indices:
            frame_points_seg0: List[np.ndarray] = []
            frame_colors: List[np.ndarray] = []
            frame_points_seg0_for_dynamic: List[np.ndarray] = []
            frame_colors_for_dynamic: List[np.ndarray] = []
            frame_instances = frame_to_instances.get(frame_idx, [])
            moving_instances, _ = self._split_instances_by_static_intids(
                frame_instances,
                static_instance_intids,
            )
            use_instance_background_policy = len(frame_instances) > 0

            for cam_id, frames_sorted in sorted_frame_data_by_camera.items():
                if not frames_sorted:
                    continue
                masks = consistency_masks_by_camera[cam_id]
                for order, (fi, frame_data) in enumerate(frames_sorted):
                    if fi != frame_idx:
                        continue
                    consistency_mask = masks[order] if order < len(masks) else None
                    if use_instance_background_policy:
                        points_w_all, colors_all, pixels_yx = self._generate_points_from_frame_data(
                            frame_data,
                            consistency_mask,
                            downscale_mask,
                            apply_dynamic_filter=False,
                            include_pixels=True,
                        )
                        if points_w_all is None or len(points_w_all) == 0:
                            continue
                        points_seg0_all = self._transform_points_np(points_w_all, world_to_seg0)
                        bg_points, bg_colors = self._filter_background_points_with_instance_policy(
                            points_seg0_all,
                            colors_all,
                            pixels_yx,
                            frame_data,
                            frame_instances,
                            static_instance_intids,
                        )
                        if bg_points is not None and len(bg_points) > 0:
                            frame_points_seg0.append(bg_points)
                            frame_colors.append(bg_colors)
                        if self.dynamic_recovery_enable:
                            frame_points_seg0_for_dynamic.append(points_seg0_all)
                            frame_colors_for_dynamic.append(colors_all)
                    else:
                        points_w, colors = self._generate_points_from_frame_data(
                            frame_data,
                            consistency_mask,
                            downscale_mask,
                            apply_dynamic_filter=True,
                        )
                        if points_w is not None and len(points_w) > 0:
                            frame_points_seg0.append(self._transform_points_np(points_w, world_to_seg0))
                            frame_colors.append(colors)
                        if self.dynamic_recovery_enable:
                            points_w_dyn, colors_dyn = self._generate_points_from_frame_data(
                                frame_data,
                                consistency_mask,
                                downscale_mask,
                                apply_dynamic_filter=False,
                            )
                            if points_w_dyn is not None and len(points_w_dyn) > 0:
                                frame_points_seg0_for_dynamic.append(
                                    self._transform_points_np(points_w_dyn, world_to_seg0)
                                )
                                frame_colors_for_dynamic.append(colors_dyn)

            if len(frame_points_seg0) > 0:
                points_seg0 = np.concatenate(frame_points_seg0, axis=0)
                colors = np.concatenate(frame_colors, axis=0)
                if len(points_seg0) > 0:
                    background_frame = np.concatenate([points_seg0, colors], axis=1).astype(
                        np.float32, copy=False
                    )
                    all_backgrounds.append(background_frame)

            if self.dynamic_recovery_enable and len(frame_points_seg0_for_dynamic) > 0:
                dynamic_points_seg0 = np.concatenate(frame_points_seg0_for_dynamic, axis=0)
                dynamic_colors = np.concatenate(frame_colors_for_dynamic, axis=0)
                dynamic_frame, frame_recovered_before_cap = self._recover_dynamic_points_by_3d_bbox(
                    dynamic_points_seg0,
                    dynamic_colors,
                    moving_instances,
                )
                if len(dynamic_frame) > 0:
                    dynamic_recovery_frame_count += 1
                dynamic_recovered_points_before_cap += frame_recovered_before_cap
                for intid, points_local_rgb in dynamic_frame.items():
                    all_dynamic_objects[intid].append(points_local_rgb)

        background = (
            np.concatenate(all_backgrounds, axis=0)
            if len(all_backgrounds) > 0
            else np.zeros((0, 6), dtype=np.float32)
        )
        dynamic_objects = self._finalize_dynamic_objects_with_cap(all_dynamic_objects)
        # Single-stage filtering without inside/outside split; distant/near划分交给上层根据 segment_aabb 完成
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
            "type": "monocular",
            "frame_indices": frame_indices,
            "frames_used": len(all_backgrounds),
            "sparsity": self.sparsity,
            "dynamic_filter": self.dynamic_filter,
            "dynamic_mask_key": self.dynamic_mask_key,
            "dynamic_recovery_enable": self.dynamic_recovery_enable,
            "dynamic_recovery_assignment": self.dynamic_recovery_assignment,
            "dynamic_recovery_bbox_expand_xyz_m": self.dynamic_recovery_bbox_expand_xyz_m.tolist(),
            "dynamic_recovery_max_points_per_instance": self.dynamic_recovery_max_points_per_instance,
            "dynamic_recovery_frames_with_points": dynamic_recovery_frame_count,
            "dynamic_recovered_instances": len(dynamic_objects),
            "dynamic_recovered_points_before_cap": int(dynamic_recovered_points_before_cap),
            "dynamic_recovered_points_total": int(
                sum(int(points.shape[0]) for points in dynamic_objects.values())
            ),
            "static_instance_motion_enable": self.static_instance_motion_enable,
            "static_instance_intids": sorted(int(x) for x in static_instance_intids),
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

    def _load_frame_data(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        frame_idx: int,
        cam_id: int,
    ) -> Optional[Dict]:
        scene_data = dataset.get_scene(scene_id)
        if scene_data is None:
            return None
        scene_dataset = scene_data["dataset"]
        pixel_source = scene_dataset.pixel_source

        if cam_id not in pixel_source.camera_list:
            logger.warning(
                "Camera ID %s not found in camera_list %s", cam_id, pixel_source.camera_list
            )
            return None

        unique_cam_idx = pixel_source.camera_data[cam_id].unique_cam_idx
        img_idx = frame_idx * pixel_source.num_cams + unique_cam_idx

        try:
            image_infos, cam_infos = pixel_source.get_image(img_idx)
        except Exception as exc:
            logger.warning(
                "Failed to load image for scene %s frame %s cam %s: %s",
                scene_id,
                frame_idx,
                cam_id,
                exc,
            )
            return None

        camera_data = pixel_source.camera_data[cam_id]
        depth = None
        try:
            if hasattr(camera_data, "depth_maps") and camera_data.depth_maps is not None:
                depth = camera_data.depth_maps[frame_idx]
            elif hasattr(camera_data, "lidar_depth_maps") and camera_data.lidar_depth_maps is not None:
                depth = camera_data.lidar_depth_maps[frame_idx]
            elif "depth_map" in image_infos:
                depth = image_infos["depth_map"]
            elif "lidar_depth_map" in image_infos:
                depth = image_infos["lidar_depth_map"]
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Depth unavailable for frame %s cam %s: %s", frame_idx, cam_id, exc)

        if depth is None:
            return None

        sky_mask = image_infos.get("sky_masks")
        if self.filter_sky and sky_mask is None:
            raise ValueError(
                "Monocular pointcloud requires image_infos['sky_masks'] when filter_sky is enabled."
            )

        dynamic_mask = image_infos.get(self.dynamic_mask_key)
        if self.dynamic_filter and dynamic_mask is None:
            raise ValueError(
                f"Monocular pointcloud requires image_infos['{self.dynamic_mask_key}'] "
                f"when dynamic_filter is enabled."
            )

        return {
            "image": image_infos["pixels"],
            "depth": depth,
            "extrinsic": cam_infos["camera_to_world"],
            "intrinsic": cam_infos["intrinsics"],
            "sky_mask": sky_mask,
            "dynamic_mask": dynamic_mask,
        }

    def _depth_consistency_check(
        self, frame_data_list: List[Dict], H: int, W: int
    ) -> List[np.ndarray]:
        if len(frame_data_list) == 0:
            return []

        masks: List[np.ndarray] = []
        last_depth = None
        for idx, frame_data in enumerate(frame_data_list):
            depth = frame_data["depth"]
            depth_np = depth.cpu().numpy() if isinstance(depth, torch.Tensor) else depth
            if depth_np is None:
                masks.append(np.ones((H, W), dtype=bool))
                continue

            if idx == 0 or last_depth is None:
                masks.append(np.ones((H, W), dtype=bool))
                last_depth = depth_np.copy()
                continue

            c2w = frame_data["extrinsic"]
            last_c2w = frame_data_list[idx - 1]["extrinsic"]
            K = frame_data["intrinsic"]
            last_K = frame_data_list[idx - 1]["intrinsic"]

            c2w = c2w.cpu().numpy() if isinstance(c2w, torch.Tensor) else c2w
            last_c2w = (
                last_c2w.cpu().numpy() if isinstance(last_c2w, torch.Tensor) else last_c2w
            )
            K = K.cpu().numpy() if isinstance(K, torch.Tensor) else K
            last_K = last_K.cpu().numpy() if isinstance(last_K, torch.Tensor) else last_K
            if K.shape[0] == 4:
                K = K[:3, :3]
            if last_K.shape[0] == 4:
                last_K = last_K[:3, :3]

            x = np.arange(0, W)
            y = np.arange(0, H)
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack([xx.ravel(), yy.ravel()]).T

            cx, cy = K[0, 2], K[1, 2]
            fx, fy = K[0, 0], K[1, 1]
            depth_flat = depth_np.ravel()
            x_cam = (pixels[:, 0] - cx) * depth_flat / fx
            y_cam = (pixels[:, 1] - cy) * depth_flat / fy
            z_cam = depth_flat
            coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)

            trans_mat = np.linalg.inv(last_c2w) @ c2w
            coordinates_homo = np.column_stack([coordinates, np.ones(len(coordinates))])
            last_coordinates = (trans_mat @ coordinates_homo.T).T

            last_cx, last_cy = last_K[0, 2], last_K[1, 2]
            last_fx, last_fy = last_K[0, 0], last_K[1, 1]
            last_x = (last_fx * last_coordinates[:, 0] + last_cx * last_coordinates[:, 2]) / last_coordinates[
                :, 2
            ]
            last_y = (last_fy * last_coordinates[:, 1] + last_cy * last_coordinates[:, 2]) / last_coordinates[
                :, 2
            ]
            valid_mask = (
                (last_x >= 0)
                & (last_x < W)
                & (last_y >= 0)
                & (last_y < H)
                & (last_coordinates[:, 2] > 0)
            )

            depth_mask = np.ones(H * W, dtype=bool)
            if np.any(valid_mask):
                last_pixels_int = np.stack([last_x[valid_mask], last_y[valid_mask]], axis=1).astype(int)
                last_pixels_int[:, 0] = np.clip(last_pixels_int[:, 0], 0, W - 1)
                last_pixels_int[:, 1] = np.clip(last_pixels_int[:, 1], 0, H - 1)
                depth_diff = np.abs(
                    depth_flat[valid_mask]
                    - last_depth[last_pixels_int[:, 1], last_pixels_int[:, 0]]
                )
                depth_mask[valid_mask] = depth_diff < depth_diff.mean()

            masks.append(depth_mask.reshape(H, W))
            last_depth = depth_np.copy()

        return masks

    def _generate_points_from_frame_data(
        self,
        frame_data: Dict,
        consistency_mask: Optional[np.ndarray],
        downscale_mask: Optional[np.ndarray],
        *,
        apply_dynamic_filter: bool,
        include_pixels: bool = False,
    ):
        rgb = frame_data["image"]
        depth = frame_data["depth"]
        extrinsic = frame_data["extrinsic"]
        intrinsic = frame_data["intrinsic"]
        sky_mask = frame_data.get("sky_mask")
        dynamic_mask = frame_data.get("dynamic_mask")

        rgb_np = rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
        depth_np = depth.cpu().numpy() if isinstance(depth, torch.Tensor) else depth
        extrinsic_np = extrinsic.cpu().numpy() if isinstance(extrinsic, torch.Tensor) else extrinsic
        intrinsic_np = intrinsic.cpu().numpy() if isinstance(intrinsic, torch.Tensor) else intrinsic
        if intrinsic_np.shape[0] == 4:
            intrinsic_np = intrinsic_np[:3, :3]

        H, W = depth_np.shape

        # sky_mask from MultiSceneDataset is canonical **1=sky, 0=non-sky** (float).
        if sky_mask is not None:
            if isinstance(sky_mask, torch.Tensor):
                sky_mask = sky_mask.cpu().numpy()
            is_sky = sky_mask > 0.5
            if self.filter_sky:
                keep_mask = ~is_sky
            else:
                keep_mask = np.ones((H, W), dtype=bool)
        else:
            keep_mask = np.ones((H, W), dtype=bool)

        # dynamic_mask is canonical **1=dynamic(ignore), 0=static(keep)** (float/bool).
        if self.dynamic_filter and apply_dynamic_filter:
            if dynamic_mask is None:
                raise ValueError(
                    "dynamic_filter is enabled but dynamic_mask is missing in frame_data."
                )
            if isinstance(dynamic_mask, torch.Tensor):
                dynamic_mask = dynamic_mask.cpu().numpy()
            is_dynamic = dynamic_mask > 0.5
            keep_dynamic_mask = ~is_dynamic
        else:
            keep_dynamic_mask = np.ones((H, W), dtype=bool)

        if consistency_mask is None:
            consistency_mask = np.ones((H, W), dtype=bool)

        if downscale_mask is not None:
            final_mask = consistency_mask & keep_mask & keep_dynamic_mask & downscale_mask
        else:
            final_mask = consistency_mask & keep_mask & keep_dynamic_mask

        kept = np.argwhere(final_mask)
        if len(kept) == 0:
            if include_pixels:
                return None, None, None
            return None, None

        depth_values = depth_np[kept[:, 0], kept[:, 1]]
        rgb_values = rgb_np[kept[:, 0], kept[:, 1]]
        valid_depth_mask = np.isfinite(depth_values) & (depth_values > 0)
        if not np.any(valid_depth_mask):
            if include_pixels:
                return None, None, None
            return None, None

        depth_values = depth_values[valid_depth_mask]
        rgb_values = rgb_values[valid_depth_mask]
        kept_valid = kept[valid_depth_mask]

        pixel_coords = kept_valid[:, [1, 0]].astype(np.float32)
        x_cam = (pixel_coords[:, 0] - intrinsic_np[0, 2]) * depth_values / intrinsic_np[0, 0]
        y_cam = (pixel_coords[:, 1] - intrinsic_np[1, 2]) * depth_values / intrinsic_np[1, 1]
        z_cam = depth_values
        coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)
        valid_coords_mask = np.isfinite(coordinates).all(axis=1)
        if not np.any(valid_coords_mask):
            if include_pixels:
                return None, None, None
            return None, None

        coordinates = coordinates[valid_coords_mask]
        rgb_values = rgb_values[valid_coords_mask]
        kept_valid = kept_valid[valid_coords_mask]
        coordinates_homo = np.column_stack([coordinates, np.ones(len(coordinates))])
        worlds = (extrinsic_np @ coordinates_homo.T).T[:, :3]

        valid_worlds_mask = np.isfinite(worlds).all(axis=1)
        if not np.any(valid_worlds_mask):
            if include_pixels:
                return None, None, None
            return None, None

        worlds = worlds[valid_worlds_mask]
        rgb_values = rgb_values[valid_worlds_mask]
        kept_valid = kept_valid[valid_worlds_mask]

        # Ensure colors are in [0, 255]
        if rgb_values.max() <= 1.0 + 1e-3:
            rgb_values = rgb_values * 255.0
        rgb_values = rgb_values.astype(np.float32)

        if include_pixels:
            return worlds.astype(np.float32), rgb_values, kept_valid.astype(np.int64, copy=False)
        return worlds.astype(np.float32), rgb_values

    @staticmethod
    def _split_instances_by_static_intids(
        instances: List[Dict],
        static_instance_intids: Set[int],
    ) -> Tuple[List[Dict], List[Dict]]:
        static_ids = {int(x) for x in static_instance_intids}
        moving: List[Dict] = []
        stationary: List[Dict] = []
        for instance in instances:
            if int(instance["intid"]) in static_ids:
                stationary.append(instance)
            else:
                moving.append(instance)
        return moving, stationary

    def _points_inside_any_instance_mask(
        self,
        points_world: np.ndarray,
        instances: List[Dict],
    ) -> np.ndarray:
        n = int(points_world.shape[0])
        mask_any = np.zeros((n,), dtype=bool)
        if n == 0 or len(instances) == 0:
            return mask_any

        points_world = points_world.astype(np.float32, copy=False)
        points_h = np.concatenate(
            [points_world, np.ones((n, 1), dtype=np.float32)],
            axis=1,
        )
        for instance in instances:
            T_ow = np.asarray(instance["T_ow"], dtype=np.float32)
            size_lwh = np.asarray(instance["size_lwh"], dtype=np.float32)
            if T_ow.shape == (3, 4):
                T_ow = np.vstack([T_ow, np.array([0, 0, 0, 1], dtype=np.float32)])
            if T_ow.shape != (4, 4):
                continue
            if size_lwh.shape != (3,):
                size_lwh = size_lwh.reshape(3)
            try:
                T_wo = np.linalg.inv(T_ow)
            except np.linalg.LinAlgError:
                continue

            local_all = (T_wo @ points_h.T).T[:, :3]
            half = 0.5 * size_lwh + self.dynamic_recovery_bbox_expand_xyz_m
            mask_any |= (np.abs(local_all) <= (half[None, :] + 1e-6)).all(axis=1)
        return mask_any

    def _filter_background_points_with_instance_policy(
        self,
        points_world: np.ndarray,
        colors: np.ndarray,
        pixels_yx: Optional[np.ndarray],
        frame_data: Dict,
        instances: List[Dict],
        static_instance_intids: Set[int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if points_world is None or len(points_world) == 0:
            return None, None

        moving_instances, stationary_instances = self._split_instances_by_static_intids(
            instances,
            static_instance_intids,
        )
        moving_bbox_mask = self._points_inside_any_instance_mask(points_world, moving_instances)
        stationary_bbox_mask = self._points_inside_any_instance_mask(points_world, stationary_instances)
        keep = ~moving_bbox_mask

        if self.dynamic_filter:
            dynamic_mask = frame_data.get("dynamic_mask")
            if dynamic_mask is None:
                raise ValueError(
                    "dynamic_filter is enabled but dynamic_mask is missing in frame_data."
                )
            if pixels_yx is None:
                raise ValueError("pixels_yx is required for instance-aware dynamic mask filtering.")
            if isinstance(dynamic_mask, torch.Tensor):
                dynamic_mask = dynamic_mask.cpu().numpy()
            pixels_yx = np.asarray(pixels_yx, dtype=np.int64)
            is_dynamic_pixel = dynamic_mask[pixels_yx[:, 0], pixels_yx[:, 1]] > 0.5
            keep &= (~is_dynamic_pixel) | stationary_bbox_mask

        if not np.any(keep):
            return None, None
        return (
            points_world[keep].astype(np.float32, copy=False),
            colors[keep].astype(np.float32, copy=False),
        )

    def _stride_limit_points6(self, points6: np.ndarray, max_count: Optional[int]) -> np.ndarray:
        if max_count is None:
            return points6
        n = int(points6.shape[0])
        if n <= int(max_count):
            return points6
        step = max(1, n // int(max_count))
        idx = np.arange(0, n, step, dtype=np.int64)
        if idx.shape[0] > int(max_count):
            idx = idx[: int(max_count)]
        return points6[idx]

    def _recover_dynamic_points_by_3d_bbox(
        self,
        points_world: np.ndarray,
        colors: np.ndarray,
        instances: List[Dict],
    ) -> Tuple[Dict[int, np.ndarray], int]:
        if len(points_world) == 0 or len(instances) == 0:
            return {}, 0

        points_world = points_world.astype(np.float32, copy=False)
        colors = colors.astype(np.float32, copy=False)
        n = int(points_world.shape[0])
        points_h = np.concatenate(
            [points_world, np.ones((n, 1), dtype=np.float32)],
            axis=1,
        )

        dynamic_dict: Dict[int, np.ndarray] = {}
        recovered_before_cap = 0

        if self.dynamic_recovery_assignment == "nearest_center":
            best_intid = np.full((n,), -1, dtype=np.int64)
            best_dist = np.full((n,), np.inf, dtype=np.float32)
            best_local = np.zeros((n, 3), dtype=np.float32)

            for instance in instances:
                intid = int(instance["intid"])
                T_ow = np.asarray(instance["T_ow"], dtype=np.float32)
                size_lwh = np.asarray(instance["size_lwh"], dtype=np.float32)
                if T_ow.shape == (3, 4):
                    T_ow = np.vstack([T_ow, np.array([0, 0, 0, 1], dtype=np.float32)])
                if T_ow.shape != (4, 4):
                    continue
                if size_lwh.shape != (3,):
                    size_lwh = size_lwh.reshape(3)
                try:
                    T_wo = np.linalg.inv(T_ow)
                except np.linalg.LinAlgError:
                    continue

                local_all = (T_wo @ points_h.T).T[:, :3]
                half = 0.5 * size_lwh + self.dynamic_recovery_bbox_expand_xyz_m
                inside = (np.abs(local_all) <= (half[None, :] + 1e-6)).all(axis=1)
                if not np.any(inside):
                    continue
                dist = np.linalg.norm(local_all, axis=1).astype(np.float32)
                better = inside & (dist < best_dist)
                if not np.any(better):
                    continue
                best_dist[better] = dist[better]
                best_intid[better] = intid
                best_local[better] = local_all[better].astype(np.float32, copy=False)

            valid = best_intid >= 0
            if not np.any(valid):
                return {}, 0
            recovered_before_cap = int(valid.sum())
            unique_intids = np.unique(best_intid[valid]).tolist()
            for intid in unique_intids:
                mask = best_intid == intid
                local_points = best_local[mask]
                selected_colors = colors[mask].astype(np.float32, copy=False)
                points6 = np.concatenate([local_points, selected_colors], axis=1)
                points6 = self._stride_limit_points6(points6, self.dynamic_recovery_max_points_per_instance)
                if points6.shape[0] > 0:
                    dynamic_dict[int(intid)] = points6.astype(np.float32, copy=False)
            return dynamic_dict, recovered_before_cap

        assigned = np.zeros(n, dtype=bool)
        for instance in instances:
            intid = int(instance["intid"])
            T_ow = np.asarray(instance["T_ow"], dtype=np.float32)
            size_lwh = np.asarray(instance["size_lwh"], dtype=np.float32)
            if T_ow.shape == (3, 4):
                T_ow = np.vstack([T_ow, np.array([0, 0, 0, 1], dtype=np.float32)])
            if T_ow.shape != (4, 4):
                continue
            if size_lwh.shape != (3,):
                size_lwh = size_lwh.reshape(3)

            try:
                T_wo = np.linalg.inv(T_ow)
            except np.linalg.LinAlgError:
                continue

            if self.dynamic_recovery_assignment == "first_hit":
                candidate_idx = np.where(~assigned)[0]
            else:
                candidate_idx = np.arange(n, dtype=np.int64)
            if candidate_idx.shape[0] == 0:
                continue

            local_all = (T_wo @ points_h[candidate_idx].T).T[:, :3]
            half = 0.5 * size_lwh + self.dynamic_recovery_bbox_expand_xyz_m
            inside = (np.abs(local_all) <= (half[None, :] + 1e-6)).all(axis=1)
            if not np.any(inside):
                continue

            local_points = local_all[inside].astype(np.float32, copy=False)
            selected_idx = candidate_idx[inside]
            selected_colors = colors[selected_idx].astype(np.float32, copy=False)
            points6 = np.concatenate([local_points, selected_colors], axis=1)
            recovered_before_cap += int(points6.shape[0])
            points6 = self._stride_limit_points6(points6, self.dynamic_recovery_max_points_per_instance)
            if points6.shape[0] > 0:
                dynamic_dict[intid] = points6.astype(np.float32, copy=False)
                if self.dynamic_recovery_assignment == "first_hit":
                    assigned[selected_idx] = True

        return dynamic_dict, recovered_before_cap

    def _finalize_dynamic_objects_with_cap(
        self,
        all_dynamic_objects: Dict[int, List[np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        if len(all_dynamic_objects) == 0:
            return {}

        finalized: Dict[int, np.ndarray] = {}
        for intid, points_list in all_dynamic_objects.items():
            if len(points_list) == 0:
                continue
            merged = np.concatenate(points_list, axis=0).astype(np.float32, copy=False)
            merged = self._stride_limit_points6(
                merged,
                self.dynamic_recovery_max_points_per_instance,
            )
            finalized[intid] = merged
        return finalized

    def _get_instances_for_segment(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
        frame_indices: List[int],
        world_to_seg0: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[int, int], List[List[Dict]]]:
        scene_data = dataset.get_scene(scene_id)
        if scene_data is None:
            return {}, [[] for _ in frame_indices]

        scene_dataset = scene_data["dataset"]
        pixel_source = scene_dataset.pixel_source
        if pixel_source.instances_pose is None:
            return {}, [[] for _ in frame_indices]

        instances_pose = pixel_source.instances_pose
        instances_size = pixel_source.instances_size
        per_frame_instance_mask = pixel_source.per_frame_instance_mask
        instances_true_id = pixel_source.instances_true_id

        instances_pose_np = (
            instances_pose.cpu().numpy()
            if isinstance(instances_pose, torch.Tensor)
            else instances_pose
        )
        instances_size_np = (
            instances_size.cpu().numpy()
            if isinstance(instances_size, torch.Tensor)
            else instances_size
        )
        per_frame_instance_mask_np = (
            per_frame_instance_mask.cpu().numpy()
            if isinstance(per_frame_instance_mask, torch.Tensor)
            else per_frame_instance_mask
        )
        instances_true_id_np = (
            instances_true_id.cpu().numpy()
            if isinstance(instances_true_id, torch.Tensor)
            else instances_true_id
        )

        num_instances = instances_pose_np.shape[1]
        instance_mapping = {int(instances_true_id_np[i]): int(i) for i in range(num_instances)}

        instances_by_frame: List[List[Dict]] = []
        for frame_idx in frame_indices:
            frame_instances: List[Dict] = []
            for ins_id in range(num_instances):
                if (
                    frame_idx >= per_frame_instance_mask_np.shape[0]
                    or not per_frame_instance_mask_np[frame_idx, ins_id]
                ):
                    continue
                frame_instances.append(
                    {
                        "intid": int(ins_id),
                        "original_id": int(instances_true_id_np[ins_id]),
                        "T_ow": instances_pose_np[frame_idx, ins_id],
                        "size_lwh": instances_size_np[ins_id],
                    }
                )
            instances_by_frame.append(
                self._transform_instances_to_seg0(frame_instances, world_to_seg0)
            )

        return instance_mapping, instances_by_frame
