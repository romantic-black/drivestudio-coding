import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import open3d as o3d

from .base import RGBPointCloudGenerator
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
        # 融合参数
        near_max_points: Optional[int] = None,
        distant_max_points: Optional[int] = None,
        fusion_strategy: Literal["merge", "lidar_first", "adaptive"] = "adaptive",
        dynamic_source: Literal["lidar_only", "fuse"] = "lidar_only",
        downsample_dynamic: bool = False,
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

        # 创建LiDAR生成器（不再传递 crop_aabb/input_aabb；由上层使用 segment_aabb 划分近/远景）
        self.lidar_generator = LiDARRGBPointCloudGenerator(
            sparsity=lidar_sparsity,
            device=device,
        )

        # 创建单目生成器
        self.monocular_generator = MonocularRGBPointCloudGenerator(
            chosen_cam_ids=monocular_chosen_cam_ids,
            sparsity=monocular_sparsity,
            filter_sky=monocular_filter_sky,
            depth_consistency=monocular_depth_consistency,
            downscale=monocular_downscale,
            device=device,
        )

        # 存储融合参数
        self.near_max_points = int(near_max_points) if near_max_points is not None else None
        self.distant_max_points = int(distant_max_points) if distant_max_points is not None else None
        self.fusion_strategy = fusion_strategy
        self.dynamic_source = dynamic_source
        self.downsample_dynamic = downsample_dynamic

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

        monocular_background = monocular_result["background"]  # [N2, 6]
        monocular_dynamic = monocular_result["dynamic"]  # Dict[int, np.ndarray]
        monocular_instance_mapping = monocular_result["instance_mapping"]

        # 统一实例映射（使用LiDAR的映射，因为它是更稳定的来源）
        instance_mapping = lidar_instance_mapping
        if lidar_instance_mapping != monocular_instance_mapping:
            logger.warning(
                f"Instance mappings differ between LiDAR and monocular generators. "
                f"Using LiDAR mapping."
            )

        # 融合背景点云
        fused_background = self._fuse_background_points(
            lidar_background, monocular_background, lidar_dynamic
        )
        fused_background, cap_stats = self._apply_segment_aabb_caps(dataset, fused_background)

        # 融合动态对象点云
        fused_dynamic = self._fuse_dynamic_objects(lidar_dynamic, monocular_dynamic)

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
            "lidar_frames_used": lidar_result["metadata"].get("frames_used", 0),
            "monocular_frames_used": monocular_result["metadata"].get("frames_used", 0),
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
        lidar_dynamic: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        融合背景点云。
        
        策略：
        - 先按fusion_strategy融合（merge / lidar_first / adaptive）
        - adaptive 策略在“没有点数预算”时退化为 merge（即返回全量点云）
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

        # 根据融合策略融合点云
        if self.fusion_strategy == "merge":
            # 简单合并
            return np.concatenate([lidar_background, monocular_background], axis=0)

        elif self.fusion_strategy == "lidar_first":
            # “lidar_first” 在没有点数预算的前提下等价于 merge（保留全量LiDAR和单目点）
            return np.concatenate([lidar_background, monocular_background], axis=0)

        elif self.fusion_strategy == "adaptive":
            # 没有点数预算时，自适应策略没有意义，退化为 merge
            return np.concatenate([lidar_background, monocular_background], axis=0)

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _fuse_dynamic_objects(
        self,
        lidar_dynamic: Dict[int, np.ndarray],
        monocular_dynamic: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        动态对象点云融合策略。
        
        - dynamic_source="lidar_only": 直接返回lidar_dynamic
        - dynamic_source="fuse": 按实例ID合并lidar_dynamic与monocular_dynamic（实验性）
        """
        if self.dynamic_source == "lidar_only":
            result = lidar_dynamic.copy()
        elif self.dynamic_source == "fuse":
            # 按实例ID合并
            result = lidar_dynamic.copy()
            for intid, monocular_points in monocular_dynamic.items():
                if intid in result:
                    # 合并同一实例的点云
                    result[intid] = np.concatenate([result[intid], monocular_points], axis=0)
                else:
                    # 添加新实例
                    result[intid] = monocular_points
        else:
            raise ValueError(f"Unknown dynamic_source: {self.dynamic_source}")

        # 如果启用下采样，对动态点云进行下采样
        if self.downsample_dynamic:
            for intid in result:
                # 对每个动态对象进行下采样（保持点数比例）
                # 这里使用简单的均匀下采样
                points = result[intid]
                if len(points) > 1000:  # 只对较大的点云下采样
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    # 修复：更准确地判断颜色范围，避免错误的归一化/反归一化
                    # 如果最大值 > 1.0 + 1e-3，说明是 [0, 255] 范围，需要归一化到 [0, 1] 供 Open3D 使用
                    # 如果最大值 <= 1.0 + 1e-3，说明已经是 [0, 1] 范围，不需要归一化
                    points_colors = points[:, 3:]
                    if points_colors.max() > 1.0 + 1e-3:
                        # 已经是 [0, 255] 范围，归一化到 [0, 1]
                        colors_normalized = points_colors / 255.0
                    else:
                        # 已经是 [0, 1] 范围，不需要归一化
                        colors_normalized = points_colors
                    pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
                    # 下采样到原来的50%
                    every_k = max(1, len(points) // (len(points) // 2))
                    pcd = pcd.uniform_down_sample(every_k_points=every_k)
                    filtered_points = np.asarray(pcd.points).astype(np.float32)
                    filtered_colors = np.asarray(pcd.colors).astype(np.float32)
                    # 修复：确保颜色值转换回 [0, 255] 范围
                    # Open3D 返回的颜色值应该在 [0, 1] 范围内
                    # 如果最大值 <= 1.0 + 1e-3，说明是 [0, 1] 范围，需要乘以 255 转换到 [0, 255]
                    # 如果最大值 > 1.0 + 1e-3，说明已经是 [0, 255] 范围（不应该发生，但为了安全起见处理）
                    if filtered_colors.max() <= 1.0 + 1e-3:
                        filtered_colors = filtered_colors * 255.0
                    else:
                        # 如果已经是 [0, 255] 范围，确保值在有效范围内
                        filtered_colors = np.clip(filtered_colors, 0.0, 255.0)
                    result[intid] = np.concatenate([filtered_points, filtered_colors], axis=1)

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
