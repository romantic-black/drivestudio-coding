import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import open3d as o3d

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)


class RGBPointCloudGenerator(ABC):
    """
    Base class for RGB point cloud generators.

    Subclasses should return a dictionary with background points, dynamic objects,
    and an instance id mapping. Colors are stored in float32 with range [0, 255].
    """

    def __init__(
        self,
        sparsity: Literal["Drop90", "Drop80", "Drop50", "Drop25", "full"] = "full",
        filter_sky: bool = True,
        depth_consistency: bool = True,
        downscale: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        self.sparsity = sparsity
        self.filter_sky = filter_sky
        self.depth_consistency = depth_consistency
        self.downscale = downscale
        self.device = device
    
    def _convert_pose_to_numpy(self, pose) -> Optional[np.ndarray]:
        """Convert pose (torch/np/list) to 4x4 numpy array."""
        if pose is None:
            return None
        if isinstance(pose, torch.Tensor):
            pose_np = pose.detach().cpu().float().numpy()
        else:
            pose_np = np.asarray(pose, dtype=np.float32)
        if pose_np.shape == (3, 4):
            pose_np = np.vstack([pose_np, np.array([0, 0, 0, 1], dtype=np.float32)])
        if pose_np.shape != (4, 4):
            logger.warning(f"Unexpected pose shape {pose_np.shape}; expected (4,4).")
            return None
        return pose_np.astype(np.float32)

    def _compute_world_to_seg0(self, segment_first_pose) -> Optional[np.ndarray]:
        """Compute world->segment0 transform from a segment-first pose."""
        pose_np = self._convert_pose_to_numpy(segment_first_pose)
        if pose_np is None:
            raise ValueError("segment_first_pose is required to compute world->seg0 transform.")
        try:
            return np.linalg.inv(pose_np)
        except np.linalg.LinAlgError:
            raise ValueError("Failed to invert segment_first_pose; matrix may be singular.")

    def _transform_points_np(
        self,
        points: np.ndarray,
        world_to_seg0: Optional[np.ndarray],
    ) -> np.ndarray:
        """Transform points with a 4x4 matrix (world->segment0)."""
        if world_to_seg0 is None or len(points) == 0:
            return points
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
        )
        transformed = (world_to_seg0 @ points_h.T).T[:, :3]
        return transformed.astype(np.float32)

    def _transform_instances_to_seg0(
        self,
        instances: List[Dict],
        world_to_seg0: Optional[np.ndarray],
    ) -> List[Dict]:
        """Deep-copy instance list and transform T_ow into segment-first coords."""
        if world_to_seg0 is None:
            return instances
        transformed: List[Dict] = []
        for inst in instances:
            inst_copy = dict(inst)
            T_ow = np.asarray(inst_copy["T_ow"], dtype=np.float32)
            if T_ow.shape == (3, 4):
                T_ow = np.vstack([T_ow, np.array([0, 0, 0, 1], dtype=np.float32)])
            if T_ow.shape != (4, 4):
                logger.debug(f"Skip transforming instance with unexpected pose shape {T_ow.shape}")
                transformed.append(inst_copy)
                continue
            inst_copy["T_ow"] = (world_to_seg0 @ T_ow).astype(np.float32)
            transformed.append(inst_copy)
        return transformed

    @abstractmethod
    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
        segment_first_pose=None,
    ) -> Dict:
        """
        Generate RGB point cloud for a segment.

        Expected keys:
            - background: np.ndarray [N, 6] (world xyz + rgb)
            - dynamic: Dict[int, np.ndarray] ({intid: [M, 6]} in local coords)
            - instance_mapping: Dict[int, int] (original id -> intid)
            - metadata: Dict (optional extras)
        
        Args:
            dataset: MultiSceneDataset instance
            scene_id: Scene identifier
            segment_id: Segment identifier
            segment_first_pose: Optional 4x4 pose of the segment's first frame in original world coords.
        """
        raise NotImplementedError

    def crop_pointcloud(
        self,
        crop_min: Optional[np.ndarray],
        crop_max: Optional[np.ndarray],
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if crop_min is None or crop_max is None or len(points) == 0:
            return points, colors

        mask = (
            (points[:, 0] > crop_min[0])
            & (points[:, 0] < crop_max[0])
            & (points[:, 1] > crop_min[1])
            & (points[:, 1] < crop_max[1])
            & (points[:, 2] > crop_min[2])
            & (points[:, 2] < crop_max[2])
        )
        return points[mask], colors[mask]

    def split_pointcloud(
        self,
        input_min: Optional[np.ndarray],
        input_max: Optional[np.ndarray],
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if input_min is None or input_max is None or len(points) == 0:
            return points, colors, np.zeros((0, 3), dtype=np.float32), np.zeros(
                (0, 3), dtype=np.float32
            )

        mask = (
            (points[:, 0] > input_min[0])
            & (points[:, 0] < input_max[0])
            & (points[:, 1] > input_min[1])
            & (points[:, 1] < input_max[1])
            & (points[:, 2] > input_min[2])
            & (points[:, 2] < input_max[2])
        )
        inside_points, inside_colors = points[mask], colors[mask]
        outside_points, outside_colors = points[~mask], colors[~mask]
        return inside_points, inside_colors, outside_points, outside_colors

    def filter_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        *,
        strict: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter point cloud (statistical filter and uniform downsampling).
        
        Args:
            points: Point coordinates [N, 3]
            colors: Point colors [N, 3] in range [0, 255]
            strict: If True, use stricter filtering for inside points (uniform downsample every_k=2).
                    If False, use looser filtering for outside points (uniform downsample every_k=4).
        
        Returns:
            Filtered points and colors
        """
        if len(points) == 0:
            return points, colors

        # Remove NaNs and Infs
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        if len(points) == 0:
            return points, colors

        # Convert to Open3D PointCloud for filtering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Open3D expects colors in range [0, 1]
        colors_normalized = colors / 255.0 if colors.max() > 1.0 + 1e-3 else colors
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

        # Statistical outlier removal
        if strict:
            # Inside points: stricter filtering
            #cl, ind = pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.5)
            #pcd = pcd.select_by_index(ind)
            # Uniform downsampling
            pcd = pcd.uniform_down_sample(every_k_points=2)
        else:
            # Outside points: looser filtering
            #cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            #pcd = pcd.select_by_index(ind)
            # Uniform downsampling
            pcd = pcd.uniform_down_sample(every_k_points=4)

        # Convert back to numpy arrays
        filtered_points = np.asarray(pcd.points).astype(np.float32)
        filtered_colors = np.asarray(pcd.colors).astype(np.float32)
        # Convert colors back to [0, 255] range
        if filtered_colors.max() <= 1.0 + 1e-3:
            filtered_colors = filtered_colors * 255.0

        return filtered_points, filtered_colors

    def _separate_static_dynamic(
        self,
        points_world: np.ndarray,
        colors: np.ndarray,
        instances: List[Dict],
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Split a single-frame point cloud into static background and dynamic objects.
        Dynamic points are converted to local coordinates.
        """
        if len(points_world) == 0:
            return (
                np.zeros((0, 6), dtype=np.float32),
                {},
            )

        points_world = points_world.astype(np.float32)
        colors = colors.astype(np.float32)
        N = points_world.shape[0]
        any_obj_mask = np.zeros(N, dtype=bool)
        dynamic_points_dict: Dict[int, np.ndarray] = {}

        for instance in instances:
            intid = int(instance["intid"])
            T_ow = np.asarray(instance["T_ow"], dtype=np.float32)
            size_lwh = np.asarray(instance["size_lwh"], dtype=np.float32)

            T_wo = np.linalg.inv(T_ow)
            points_homo = np.concatenate(
                [points_world, np.ones((N, 1), dtype=np.float32)], axis=1
            )
            points_local = (T_wo @ points_homo.T).T[:, :3]

            half = size_lwh / 2.0
            mask = (np.abs(points_local) <= (half + 1e-6)).all(axis=1)
            if not np.any(mask):
                continue

            dynamic_points_local = points_local[mask]
            dynamic_colors = colors[mask]
            dynamic_points_dict[intid] = np.concatenate(
                [dynamic_points_local, dynamic_colors], axis=1
            )
            any_obj_mask |= mask

        background_points = points_world[~any_obj_mask]
        background_colors = colors[~any_obj_mask]
        background = (
            np.concatenate([background_points, background_colors], axis=1)
            if len(background_points) > 0
            else np.zeros((0, 6), dtype=np.float32)
        )

        return background, dynamic_points_dict
