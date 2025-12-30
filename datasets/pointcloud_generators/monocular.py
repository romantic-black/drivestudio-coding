import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from .base import RGBPointCloudGenerator

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)


class MonocularRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    RGB point cloud generator that back-projects monocular depth from MultiSceneDataset.
    Splits points into static background (world coords) and dynamic objects (local coords).
    """

    def __init__(
        self,
        chosen_cam_ids: List[int] = [0],
        sparsity: Literal["Drop90", "Drop80", "Drop50", "Drop25", "full"] = "full",
        filter_sky: bool = True,
        depth_consistency: bool = True,
        use_bbx: bool = True,
        downscale: int = 2,
        crop_aabb: Optional[np.ndarray] = None,
        input_aabb: Optional[np.ndarray] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            sparsity=sparsity,
            filter_sky=filter_sky,
            depth_consistency=depth_consistency,
            use_bbx=use_bbx,
            downscale=downscale,
            crop_aabb=crop_aabb,
            input_aabb=input_aabb,
            device=device,
        )
        self.chosen_cam_ids = chosen_cam_ids

    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> Dict:
        scene_data = dataset.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")

        segment = scene_data["segments"][segment_id]
        frame_indices = sorted(list(set(segment["frame_indices"])))
        frame_indices = self._apply_sparsity_filter(frame_indices)
        if len(frame_indices) == 0:
            raise ValueError("No frames selected after sparsity filtering")

        instance_mapping, instances_by_frame = self._get_instances_for_segment(
            dataset, scene_id, segment_id, frame_indices
        )
        frame_to_instances = {
            frame_idx: instances_by_frame[i]
            for i, frame_idx in enumerate(frame_indices)
        }

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

        for frame_idx in frame_indices:
            frame_points_world: List[np.ndarray] = []
            frame_colors: List[np.ndarray] = []

            for cam_id, frames_sorted in sorted_frame_data_by_camera.items():
                if not frames_sorted:
                    continue
                masks = consistency_masks_by_camera[cam_id]
                for order, (fi, frame_data) in enumerate(frames_sorted):
                    if fi != frame_idx:
                        continue
                    consistency_mask = masks[order] if order < len(masks) else None
                    points_w, colors = self._generate_points_from_frame_data(
                        frame_data, consistency_mask, downscale_mask
                    )
                    if points_w is not None and len(points_w) > 0:
                        frame_points_world.append(points_w)
                        frame_colors.append(colors)

            if len(frame_points_world) == 0:
                continue

            points_world = np.concatenate(frame_points_world, axis=0)
            colors = np.concatenate(frame_colors, axis=0)

            if self.use_bbx:
                crop_min, crop_max = self.get_crop_aabb()
                points_world, colors = self.crop_pointcloud(
                    crop_min, crop_max, points_world, colors
                )

            if len(points_world) == 0:
                continue

            frame_instances = frame_to_instances.get(frame_idx, [])
            background_frame, dynamic_frame = self._separate_static_dynamic(
                points_world, colors, frame_instances
            )
            all_backgrounds.append(background_frame)
            for intid, pts in dynamic_frame.items():
                all_dynamic_objects[intid].append(pts)

        background = (
            np.concatenate(all_backgrounds, axis=0)
            if len(all_backgrounds) > 0
            else np.zeros((0, 6), dtype=np.float32)
        )
        dynamic_objects = {
            intid: np.concatenate(points_list, axis=0)
            for intid, points_list in all_dynamic_objects.items()
            if len(points_list) > 0
        }

        if self.use_bbx:
            input_min, input_max = self.get_input_aabb()
            background_pts, background_colors = (
                background[:, :3],
                background[:, 3:],
            )
            background_pts, background_colors, outside_pts, outside_colors = (
                self.split_pointcloud(
                    input_min, input_max, background_pts, background_colors
                )
            )
            inside_pts, inside_colors = self.filter_pointcloud(
                background_pts, background_colors, use_bbx=True
            )
            outside_pts, outside_colors = self.filter_pointcloud(
                outside_pts, outside_colors, use_bbx=False
            )
            background = np.concatenate(
                [
                    np.concatenate([inside_pts, inside_colors], axis=1)
                    if len(inside_pts) > 0
                    else np.zeros((0, 6), dtype=np.float32),
                    np.concatenate([outside_pts, outside_colors], axis=1)
                    if len(outside_pts) > 0
                    else np.zeros((0, 6), dtype=np.float32),
                ],
                axis=0,
            )
        else:
            if len(background) > 0:
                pts, cols = background[:, :3], background[:, 3:]
                pts, cols = self.filter_pointcloud(pts, cols, use_bbx=False)
                background = (
                    np.concatenate([pts, cols], axis=1)
                    if len(pts) > 0
                    else np.zeros((0, 6), dtype=np.float32)
                )

        metadata = {
            "type": "monocular",
            "frame_indices": frame_indices,
            "frames_used": len(all_backgrounds),
            "sparsity": self.sparsity,
        }

        return {
            "background": background,
            "dynamic_objects": dynamic_objects,
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

        sky_mask = image_infos.get("sky_mask")

        return {
            "image": image_infos["pixels"],
            "depth": depth,
            "extrinsic": cam_infos["camera_to_world"],
            "intrinsic": cam_infos["intrinsics"],
            "sky_mask": sky_mask,
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
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rgb = frame_data["image"]
        depth = frame_data["depth"]
        extrinsic = frame_data["extrinsic"]
        intrinsic = frame_data["intrinsic"]
        sky_mask = frame_data.get("sky_mask")

        rgb_np = rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
        depth_np = depth.cpu().numpy() if isinstance(depth, torch.Tensor) else depth
        extrinsic_np = extrinsic.cpu().numpy() if isinstance(extrinsic, torch.Tensor) else extrinsic
        intrinsic_np = intrinsic.cpu().numpy() if isinstance(intrinsic, torch.Tensor) else intrinsic
        if intrinsic_np.shape[0] == 4:
            intrinsic_np = intrinsic_np[:3, :3]

        H, W = depth_np.shape

        if sky_mask is not None:
            if isinstance(sky_mask, torch.Tensor):
                sky_mask = sky_mask.cpu().numpy()
            sky_mask = sky_mask.astype(bool)
            if self.filter_sky:
                sky_mask = ~sky_mask  # True means keep
            else:
                sky_mask = np.ones((H, W), dtype=bool)
        else:
            sky_mask = np.ones((H, W), dtype=bool)

        if consistency_mask is None:
            consistency_mask = np.ones((H, W), dtype=bool)

        if downscale_mask is not None:
            final_mask = consistency_mask & sky_mask & downscale_mask
        else:
            final_mask = consistency_mask & sky_mask

        kept = np.argwhere(final_mask)
        if len(kept) == 0:
            return None, None

        depth_values = depth_np[kept[:, 0], kept[:, 1]]
        rgb_values = rgb_np[kept[:, 0], kept[:, 1]]
        valid_depth_mask = np.isfinite(depth_values) & (depth_values > 0)
        if not np.any(valid_depth_mask):
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
            return None, None

        coordinates = coordinates[valid_coords_mask]
        rgb_values = rgb_values[valid_coords_mask]
        coordinates_homo = np.column_stack([coordinates, np.ones(len(coordinates))])
        worlds = (extrinsic_np @ coordinates_homo.T).T[:, :3]

        valid_worlds_mask = np.isfinite(worlds).all(axis=1)
        if not np.any(valid_worlds_mask):
            return None, None

        worlds = worlds[valid_worlds_mask]
        rgb_values = rgb_values[valid_worlds_mask]

        # Ensure colors are in [0, 255]
        if rgb_values.max() <= 1.0 + 1e-3:
            rgb_values = rgb_values * 255.0
        rgb_values = rgb_values.astype(np.float32)

        return worlds.astype(np.float32), rgb_values

    def _get_instances_for_segment(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
        frame_indices: List[int],
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
            instances_by_frame.append(frame_instances)

        return instance_mapping, instances_by_frame
