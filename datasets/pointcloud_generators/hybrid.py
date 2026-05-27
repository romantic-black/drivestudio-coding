import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import open3d as o3d

from .base import RGBPointCloudGenerator
from .dynamic_balance import (
    cap_dynamic_points_by_intid,
    compute_volume_balanced_point_caps,
    dynamic_point_balance_enabled,
    merge_instance_volume_maps,
    normalize_dynamic_point_balance_cfg,
    volume_map_from_metadata,
    volume_map_to_jsonable,
)
from .lidar import LiDARRGBPointCloudGenerator
from .monocular import MonocularRGBPointCloudGenerator

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)


class HybridRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    混合RGB点云生成器，结合LiDAR和单目深度点云。
    
    策略：
    1. 首先从LiDAR生成稳定的基础点云
    2. 然后从单目深度补充细节点云
    3. 融合两者（背景融合；动态默认仅使用LiDAR）
    """

    def __init__(
        self,
        # LiDAR生成器参数
        lidar_sparsity: Literal["Drop90", "Drop80", "Drop50", "Drop25", "full"] = "full",
        # 单目生成器参数
        monocular_chosen_cam_ids: List[int] = [0],
        monocular_sparsity: Literal["Drop90", "Drop80", "Drop50", "Drop25", "full"] = "full",
        monocular_filter_sky: bool = True,
        monocular_depth_consistency: bool = True,
        monocular_downscale: int = 2,
        monocular_dynamic_recovery_bbox_expand_xyz_m: Optional[List[float]] = None,
        monocular_dynamic_recovery_max_points_per_instance: Optional[int] = None,
        # 融合参数
        near_max_points: Optional[int] = None,
        distant_max_points: Optional[int] = None,
        static_instance_motion_enable: bool = False,
        static_instance_motion_traj_length_thresh_m: Optional[float] = None,
        dynamic_point_balance: Optional[Dict[str, Any]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        # 使用通用参数初始化基类（不再持有 crop_aabb/input_aabb，AABB 由上层控制）
        super().__init__(
            sparsity="full",  # 基类的sparsity不会被使用
            filter_sky=False,  # 基类的filter_sky不会被使用
            depth_consistency=False,  # 基类的depth_consistency不会被使用
            downscale=1,  # 基类的downscale不会被使用
            device=device,
        )

        self.dynamic_point_balance = normalize_dynamic_point_balance_cfg(dynamic_point_balance)
        self.dynamic_max_points_per_instance = (
            int(monocular_dynamic_recovery_max_points_per_instance)
            if monocular_dynamic_recovery_max_points_per_instance is not None
            else None
        )
        if (
            dynamic_point_balance_enabled(self.dynamic_point_balance)
            and self.dynamic_max_points_per_instance is None
        ):
            raise ValueError(
                "HybridRGBPointCloudGenerator: dynamic_point_balance.enable=true requires "
                "monocular_dynamic_recovery_max_points_per_instance."
            )

        # 创建LiDAR生成器（不再传递 crop_aabb/input_aabb；由上层使用 segment_aabb 划分近/远景）
        self.lidar_generator = LiDARRGBPointCloudGenerator(
            sparsity=lidar_sparsity,
            device=device,
            static_instance_motion_enable=static_instance_motion_enable,
            static_instance_motion_traj_length_thresh_m=static_instance_motion_traj_length_thresh_m,
            dynamic_bbox_expand_xyz_m=monocular_dynamic_recovery_bbox_expand_xyz_m,
            dynamic_max_points_per_instance=self.dynamic_max_points_per_instance,
            dynamic_point_balance=self.dynamic_point_balance,
        )

        # 创建单目生成器
        self.monocular_generator = MonocularRGBPointCloudGenerator(
            chosen_cam_ids=monocular_chosen_cam_ids,
            sparsity=monocular_sparsity,
            filter_sky=monocular_filter_sky,
            depth_consistency=monocular_depth_consistency,
            downscale=monocular_downscale,
            dynamic_filter=True,
            dynamic_recovery_enable=True,
            dynamic_recovery_bbox_expand_xyz_m=monocular_dynamic_recovery_bbox_expand_xyz_m,
            dynamic_recovery_max_points_per_instance=monocular_dynamic_recovery_max_points_per_instance,
            dynamic_recovery_assignment="first_hit",
            static_instance_motion_enable=static_instance_motion_enable,
            static_instance_motion_traj_length_thresh_m=static_instance_motion_traj_length_thresh_m,
            dynamic_point_balance=self.dynamic_point_balance,
            device=device,
        )

        # 存储融合参数
        self.near_max_points = int(near_max_points) if near_max_points is not None else None
        self.distant_max_points = int(distant_max_points) if distant_max_points is not None else None
        # Fast-fail simplification: hybrid uses deterministic behavior
        # - background fusion: merge
        # - dynamic fusion: fuse (lidar + monocular)
        # - no extra dynamic downsample here
        self.fusion_strategy: Literal["merge"] = "merge"
        self.dynamic_source: Literal["fuse"] = "fuse"
        self.downsample_dynamic = False

    def _stride_downsample(self, pts6: np.ndarray, max_count: Optional[int]) -> np.ndarray:
        if max_count is None or max_count <= 0:
            return pts6
        n = int(pts6.shape[0])
        if n <= max_count:
            return pts6
        step = max(1, n // int(max_count))
        idx = np.arange(0, n, step, dtype=np.int64)
        if idx.shape[0] > max_count:
            idx = idx[: int(max_count)]
        return pts6[idx]

    def _apply_segment_aabb_caps(self, dataset: "MultiSceneDataset", background: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply near/distant caps by splitting with dataset.segment_aabb (seg0 coords).
        Returns (background_after, stats_dict).
        """
        if background is None or background.shape[0] == 0:
            return background, {"background_n": 0}
        if self.near_max_points is None and self.distant_max_points is None:
            return background, {"background_n": int(background.shape[0]), "caps_disabled": True}

        seg_aabb = getattr(dataset, "segment_aabb_np", None)
        if seg_aabb is None:
            seg_aabb_t = getattr(dataset, "segment_aabb", None)
            if seg_aabb_t is not None:
                seg_aabb = seg_aabb_t.detach().cpu().numpy() if torch.is_tensor(seg_aabb_t) else np.asarray(seg_aabb_t)
        if seg_aabb is None:
            return background, {"background_n": int(background.shape[0]), "caps_disabled": True, "reason": "missing segment_aabb"}

        seg_aabb = np.asarray(seg_aabb, dtype=np.float32).reshape(2, 3)
        crop_min = seg_aabb[0]
        crop_max = seg_aabb[1]
        xyz = background[:, :3].astype(np.float32, copy=False)
        in_crop = ((xyz >= crop_min[None, :]) & (xyz <= crop_max[None, :])).all(axis=1)
        near = background[in_crop]
        distant = background[~in_crop]

        near_before = int(near.shape[0])
        distant_before = int(distant.shape[0])
        near_after_arr = self._stride_downsample(near, self.near_max_points)
        distant_after_arr = self._stride_downsample(distant, self.distant_max_points)
        out = np.concatenate([near_after_arr, distant_after_arr], axis=0).astype(np.float32, copy=False)

        stats = {
            "background_before": int(background.shape[0]),
            "near_before": near_before,
            "distant_before": distant_before,
            "near_cap": self.near_max_points,
            "distant_cap": self.distant_max_points,
            "near_after": int(near_after_arr.shape[0]),
            "distant_after": int(distant_after_arr.shape[0]),
            "background_after": int(out.shape[0]),
        }
        return out, stats

    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
        segment_first_pose=None,
    ) -> Dict:
        """
        生成混合RGB点云。
        
        流程：
        1. 使用LiDAR生成器生成基础点云
        2. 使用单目生成器生成补充点云
        3. 融合两种点云（背景融合；动态默认仅使用LiDAR）
        4. 返回统一格式的结果
        """
        lidar_result = None
        monocular_result = None
        lidar_error = None
        monocular_error = None

        # 尝试生成LiDAR点云
        try:
            lidar_result = self.lidar_generator.generate_pointcloud(
                dataset, scene_id, segment_id, segment_first_pose=segment_first_pose
            )
        except Exception as e:
            lidar_error = e
            logger.warning(
                f"LiDAR generator failed for scene {scene_id}, segment {segment_id}: {e}"
            )

        # 尝试生成单目点云
        try:
            monocular_result = self.monocular_generator.generate_pointcloud(
                dataset, scene_id, segment_id, segment_first_pose=segment_first_pose
            )
        except Exception as e:
            monocular_error = e
            logger.warning(
                f"Monocular generator failed for scene {scene_id}, segment {segment_id}: {e}"
            )

        # 错误处理：如果两者都失败，抛出错误
        if lidar_result is None and monocular_result is None:
            error_msg = (
                f"Both LiDAR and monocular generators failed for scene {scene_id}, "
                f"segment {segment_id}."
            )
            if lidar_error:
                error_msg += f" LiDAR error: {lidar_error}"
            if monocular_error:
                error_msg += f" Monocular error: {monocular_error}"
            raise RuntimeError(error_msg)

        # 如果只有一个成功，使用该结果（但标记为混合类型）
        if lidar_result is None:
            logger.warning(
                f"Using monocular-only result for scene {scene_id}, segment {segment_id}"
            )
            result = monocular_result.copy()
            result["metadata"]["type"] = "hybrid_monocular_fallback"
            return result

        if monocular_result is None:
            logger.warning(
                f"Using LiDAR-only result for scene {scene_id}, segment {segment_id}"
            )
            result = lidar_result.copy()
            result["metadata"]["type"] = "hybrid_lidar_fallback"
            return result

        # 提取背景和动态点云
        lidar_background = lidar_result["background"]  # [N1, 6]
        lidar_dynamic = lidar_result["dynamic"]  # Dict[int, np.ndarray]
        lidar_instance_mapping = lidar_result["instance_mapping"]
        lidar_meta = lidar_result.get("metadata") or {}

        monocular_background = monocular_result["background"]  # [N2, 6]
        monocular_dynamic = monocular_result["dynamic"]  # Dict[int, np.ndarray]
        monocular_instance_mapping = monocular_result["instance_mapping"]
        monocular_meta = monocular_result.get("metadata") or {}

        static_instance_intids = {
            int(x)
            for x in list(lidar_meta.get("static_instance_intids", []) or [])
            + list(monocular_meta.get("static_instance_intids", []) or [])
        }
        if len(static_instance_intids) > 0:
            lidar_dynamic = {
                int(k): v for k, v in lidar_dynamic.items() if int(k) not in static_instance_intids
            }
            monocular_dynamic = {
                int(k): v
                for k, v in monocular_dynamic.items()
                if int(k) not in static_instance_intids
            }
        dynamic_instance_volumes_m3 = merge_instance_volume_maps(
            volume_map_from_metadata(lidar_meta),
            volume_map_from_metadata(monocular_meta),
        )
        dynamic_point_caps = compute_volume_balanced_point_caps(
            self.dynamic_max_points_per_instance,
            dynamic_instance_volumes_m3,
            self.dynamic_point_balance,
        )

        # 统一实例映射（使用LiDAR的映射，因为它是更稳定的来源）
        instance_mapping = lidar_instance_mapping
        # if lidar_instance_mapping != monocular_instance_mapping:
        #     logger.warning(
        #         f"Instance mappings differ between LiDAR and monocular generators. "
        #         f"Using LiDAR mapping."
        #     )

        # 融合背景点云
        fused_background = self._fuse_background_points(lidar_background, monocular_background)
        fused_background, cap_stats = self._apply_segment_aabb_caps(dataset, fused_background)

        # 融合动态对象点云
        fused_dynamic = self._fuse_dynamic_objects(lidar_dynamic, monocular_dynamic)
        dynamic_count_before_balance = sum(int(points.shape[0]) for points in fused_dynamic.values())
        if dynamic_point_balance_enabled(self.dynamic_point_balance):
            fused_dynamic = cap_dynamic_points_by_intid(
                fused_dynamic,
                cap_by_intid=dynamic_point_caps,
                default_cap=self.dynamic_max_points_per_instance,
            )

        # 构建元数据
        dynamic_count = sum(len(points) for points in fused_dynamic.values())
        metadata = {
            "type": "hybrid",
            "lidar_count": len(lidar_background),
            "monocular_count": len(monocular_background),
            "fused_background_count": len(fused_background),
            "dynamic_count": dynamic_count,
            "fusion_strategy": self.fusion_strategy,
            "dynamic_source": self.dynamic_source,
            "lidar_frames_used": lidar_meta.get("frames_used", 0),
            "monocular_frames_used": monocular_result["metadata"].get("frames_used", 0),
            "static_instance_motion_enable": lidar_meta.get(
                "static_instance_motion_enable",
                monocular_meta.get("static_instance_motion_enable", False),
            ),
            "static_instance_intids": sorted(static_instance_intids),
            "dynamic_count_before_balance": int(dynamic_count_before_balance),
            "dynamic_point_balance": dict(self.dynamic_point_balance),
            "dynamic_instance_volumes_m3": volume_map_to_jsonable(dynamic_instance_volumes_m3),
            "dynamic_instance_point_caps": {
                str(int(k)): int(v)
                for k, v in sorted(dynamic_point_caps.items(), key=lambda kv: int(kv[0]))
            },
        }

        return {
            "background": fused_background,
            "dynamic": fused_dynamic,
            "instance_mapping": instance_mapping,
            "metadata": metadata,
        }

    def _fuse_background_points(
        self,
        lidar_background: np.ndarray,
        monocular_background: np.ndarray,
    ) -> np.ndarray:
        """
        融合背景点云。
        
        策略：固定 merge（直接拼接 LiDAR + monocular）。
        """
        # 处理空点云情况
        if len(lidar_background) == 0 and len(monocular_background) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        if len(lidar_background) == 0:
            # 只有单目点云
            return monocular_background

        if len(monocular_background) == 0:
            # 只有LiDAR点云
            return lidar_background

        return np.concatenate([lidar_background, monocular_background], axis=0)

    def _fuse_dynamic_objects(
        self,
        lidar_dynamic: Dict[int, np.ndarray],
        monocular_dynamic: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        动态对象点云融合策略：固定 fuse（按实例ID合并 lidar + monocular）。
        """
        result = lidar_dynamic.copy()
        for intid, monocular_points in monocular_dynamic.items():
            if intid in result:
                result[intid] = np.concatenate([result[intid], monocular_points], axis=0)
            else:
                result[intid] = monocular_points

        return result

    def _estimate_density(
        self,
        points: np.ndarray,
        k: int = 10,
    ) -> float:
        """
        估计点云的平均密度（每个点的k近邻平均距离的倒数）。
        """
        if len(points) == 0:
            return 0.0

        if len(points) == 1:
            return 1.0

        points_xyz = points[:, :3] if points.shape[1] > 3 else points
        k = min(k, len(points) - 1)
        if k < 1:
            return 0.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        avg_distances = []
        for i in range(len(points_xyz)):
            [_, _, dist] = pcd_tree.search_knn_vector_3d(points_xyz[i], k + 1)
            # 排除自身
            avg_dist = np.mean(np.sqrt(dist[1:]))
            avg_distances.append(avg_dist)

        mean_avg_dist = np.mean(avg_distances)
        # 密度 = 1 / 平均距离
        density = 1.0 / (mean_avg_dist + 1e-6)
        return density
