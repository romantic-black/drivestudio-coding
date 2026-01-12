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
    3. 融合两者并限制**静态背景**点数在max_points以下（动态点不计入）
    """

    def __init__(
        self,
        # LiDAR生成器参数
        lidar_sparsity: Literal["Drop90", "Drop80", "Drop50", "Drop25", "full"] = "full",
        lidar_use_bbx: bool = True,
        # 单目生成器参数
        monocular_chosen_cam_ids: List[int] = [0],
        monocular_sparsity: Literal["Drop90", "Drop80", "Drop50", "Drop25", "full"] = "full",
        monocular_filter_sky: bool = True,
        monocular_depth_consistency: bool = True,
        monocular_use_bbx: bool = True,
        monocular_downscale: int = 2,
        # 融合参数
        max_points: int = 500000,
        fusion_strategy: Literal["merge", "lidar_first", "adaptive"] = "adaptive",
        dynamic_source: Literal["lidar_only", "fuse"] = "lidar_only",
        downsample_dynamic: bool = False,
        count_dynamic_in_max_points: bool = False,
        background_downsample_method: Literal["uniform", "density", "distance"] = "uniform",
        # 通用参数
        crop_aabb: Optional[np.ndarray] = None,
        input_aabb: Optional[np.ndarray] = None,
        device: torch.device = torch.device("cpu"),
    ):
        # 使用通用参数初始化基类
        super().__init__(
            sparsity="full",  # 基类的sparsity不会被使用
            filter_sky=False,  # 基类的filter_sky不会被使用
            depth_consistency=False,  # 基类的depth_consistency不会被使用
            use_bbx=True,  # 基类的use_bbx不会被使用
            downscale=1,  # 基类的downscale不会被使用
            crop_aabb=crop_aabb,
            input_aabb=input_aabb,
            device=device,
        )

        # 创建LiDAR生成器
        self.lidar_generator = LiDARRGBPointCloudGenerator(
            sparsity=lidar_sparsity,
            use_bbx=lidar_use_bbx,
            crop_aabb=crop_aabb,
            input_aabb=input_aabb,
            device=device,
        )

        # 创建单目生成器
        self.monocular_generator = MonocularRGBPointCloudGenerator(
            chosen_cam_ids=monocular_chosen_cam_ids,
            sparsity=monocular_sparsity,
            filter_sky=monocular_filter_sky,
            depth_consistency=monocular_depth_consistency,
            use_bbx=monocular_use_bbx,
            downscale=monocular_downscale,
            crop_aabb=crop_aabb,
            input_aabb=input_aabb,
            device=device,
        )

        # 存储融合参数
        self.max_points = max_points
        self.fusion_strategy = fusion_strategy
        self.dynamic_source = dynamic_source
        self.downsample_dynamic = downsample_dynamic
        self.count_dynamic_in_max_points = count_dynamic_in_max_points
        self.background_downsample_method = background_downsample_method

    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> Dict:
        """
        生成混合RGB点云。
        
        流程：
        1. 使用LiDAR生成器生成基础点云
        2. 使用单目生成器生成补充点云
        3. 融合两种点云（背景融合；动态默认仅使用LiDAR）
        4. 对背景点云应用点数限制（默认限制到max_points；动态默认不下采样且不计入max_points）
        5. 返回统一格式的结果
        """
        lidar_result = None
        monocular_result = None
        lidar_error = None
        monocular_error = None

        # 尝试生成LiDAR点云
        try:
            lidar_result = self.lidar_generator.generate_pointcloud(
                dataset, scene_id, segment_id
            )
        except Exception as e:
            lidar_error = e
            logger.warning(
                f"LiDAR generator failed for scene {scene_id}, segment {segment_id}: {e}"
            )

        # 尝试生成单目点云
        try:
            monocular_result = self.monocular_generator.generate_pointcloud(
                dataset, scene_id, segment_id
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
            "max_points": self.max_points,
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
        - 再对背景点云下采样到 background_budget
        """
        # 计算背景点数预算
        if self.count_dynamic_in_max_points:
            dynamic_points = sum(len(v) for v in lidar_dynamic.values())
            background_budget = max(0, self.max_points - dynamic_points)
        else:
            background_budget = self.max_points

        # 处理空点云情况
        if len(lidar_background) == 0 and len(monocular_background) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        if len(lidar_background) == 0:
            # 只有单目点云
            return self._limit_point_count(
                monocular_background, background_budget, self.background_downsample_method
            )

        if len(monocular_background) == 0:
            # 只有LiDAR点云
            return self._limit_point_count(
                lidar_background, background_budget, self.background_downsample_method
            )

        # 根据融合策略融合点云
        if self.fusion_strategy == "merge":
            # 简单合并
            merged = np.concatenate([lidar_background, monocular_background], axis=0)
            return self._limit_point_count(
                merged, background_budget, self.background_downsample_method
            )

        elif self.fusion_strategy == "lidar_first":
            # 优先保留所有LiDAR点，剩余配额分配给单目点
            lidar_count = len(lidar_background)
            if lidar_count >= background_budget:
                # LiDAR点已经足够，只保留LiDAR点
                return self._limit_point_count(
                    lidar_background, background_budget, self.background_downsample_method
                )
            else:
                # 保留所有LiDAR点，剩余配额给单目点
                remaining_budget = background_budget - lidar_count
                selected_monocular = self._limit_point_count(
                    monocular_background, remaining_budget, self.background_downsample_method
                )
                return np.concatenate([lidar_background, selected_monocular], axis=0)

        elif self.fusion_strategy == "adaptive":
            # 自适应策略：优先保留LiDAR，单目点补充稀疏区域
            # 首先保留所有LiDAR点（如果不超过预算）
            lidar_count = len(lidar_background)
            if lidar_count >= background_budget:
                return self._limit_point_count(
                    lidar_background, background_budget, self.background_downsample_method
                )

            # 使用单目点补充稀疏区域
            remaining_budget = background_budget - lidar_count
            complementary_monocular = self._select_complementary_points(
                lidar_background[:, :3],  # 参考点（LiDAR）
                monocular_background,  # 候选点（单目）
                remaining_budget,
            )
            return np.concatenate([lidar_background, complementary_monocular], axis=0)

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
                    colors_normalized = (
                        points[:, 3:] / 255.0
                        if points[:, 3:].max() > 1.0 + 1e-3
                        else points[:, 3:]
                    )
                    pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
                    # 下采样到原来的50%
                    every_k = max(1, len(points) // (len(points) // 2))
                    pcd = pcd.uniform_down_sample(every_k_points=every_k)
                    filtered_points = np.asarray(pcd.points).astype(np.float32)
                    filtered_colors = np.asarray(pcd.colors).astype(np.float32)
                    if filtered_colors.max() <= 1.0 + 1e-3:
                        filtered_colors = filtered_colors * 255.0
                    result[intid] = np.concatenate([filtered_points, filtered_colors], axis=1)

        return result

    def _limit_point_count(
        self,
        points: np.ndarray,
        target_count: int,
        method: Literal["uniform", "density", "distance"] = "uniform",
    ) -> np.ndarray:
        """
        限制点云数量（用于背景点云）。
        
        方法：
        - "uniform": 均匀下采样
        - "density": 基于密度的下采样（保留稀疏区域）
        - "distance": 基于距离的过滤（移除过近的点）
        """
        if len(points) == 0:
            return points

        if len(points) <= target_count:
            return points

        points_xyz = points[:, :3]
        points_rgb = points[:, 3:]

        if method == "uniform":
            # 使用Open3D的均匀下采样
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz)
            colors_normalized = (
                points_rgb / 255.0 if points_rgb.max() > 1.0 + 1e-3 else points_rgb
            )
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

            # 计算下采样步长
            every_k = max(1, len(points) // target_count)
            pcd = pcd.uniform_down_sample(every_k_points=every_k)

            # 如果还是太多，再次下采样
            # 添加循环保护，防止无限循环
            max_iterations = 10  # 防止无限循环
            iteration = 0
            while len(pcd.points) > target_count * 1.1 and iteration < max_iterations:
                every_k = max(1, len(pcd.points) // target_count)
                if every_k == 1:
                    # every_k=1 时不会减少点数，直接跳出循环，后续使用随机采样
                    logger.debug(
                        f"Downsampling: every_k=1 detected (points={len(pcd.points)}, "
                        f"target={target_count}). Breaking loop, will use random sampling."
                    )
                    break
                
                logger.debug(
                    f"Downsampling iteration {iteration}: {len(pcd.points)} points -> "
                    f"target {target_count}, every_k={every_k}"
                )
                pcd = pcd.uniform_down_sample(every_k_points=every_k)
                iteration += 1
            
            if iteration >= max_iterations:
                logger.warning(
                    f"Downsampling reached max iterations ({max_iterations}). "
                    f"Current points: {len(pcd.points)}, target: {target_count}. "
                    f"Will use random sampling to reach target."
                )

            filtered_points = np.asarray(pcd.points).astype(np.float32)
            filtered_colors = np.asarray(pcd.colors).astype(np.float32)
            if filtered_colors.max() <= 1.0 + 1e-3:
                filtered_colors = filtered_colors * 255.0

            result = np.concatenate([filtered_points, filtered_colors], axis=1)

            # 如果还是超过目标，随机采样
            if len(result) > target_count:
                indices = np.random.choice(len(result), target_count, replace=False)
                result = result[indices]

            return result

        elif method == "density":
            # 基于密度的下采样：保留稀疏区域
            # 使用KDTree计算每个点的k近邻平均距离作为密度指标
            k = min(10, len(points) - 1)
            if k < 1:
                return points

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            densities = []
            for i in range(len(points_xyz)):
                [_, idx, dist] = pcd_tree.search_knn_vector_3d(points_xyz[i], k + 1)
                # 排除自身，计算平均距离（距离越大，密度越低）
                avg_dist = np.mean(np.sqrt(dist[1:]))  # 跳过第一个（自身）
                densities.append(avg_dist)

            densities = np.array(densities)
            # 选择密度最低的点（稀疏区域）
            # 按密度排序，选择前target_count个最稀疏的点
            sparse_indices = np.argsort(densities)[::-1][:target_count]
            return points[sparse_indices]

        elif method == "distance":
            # 基于距离的过滤：移除过近的点
            # 使用KDTree找到距离太近的点对，移除其中一个
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            # 估计合适的距离阈值（基于目标点数）
            # 使用体素下采样来估计
            voxel_size = np.max(points_xyz.max(axis=0) - points_xyz.min(axis=0)) / (
                target_count ** (1.0 / 3.0)
            )

            # 使用体素下采样
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            # 如果还是太多，继续下采样
            while len(pcd.points) > target_count * 1.1:
                voxel_size *= 1.2
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            filtered_points = np.asarray(pcd.points).astype(np.float32)

            # 从原始点云中找到对应的颜色
            # 对于每个下采样后的点，找到最近的原点并取其颜色
            filtered_colors = np.zeros((len(filtered_points), 3), dtype=np.float32)
            for i, pt in enumerate(filtered_points):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
                filtered_colors[i] = points_rgb[idx[0]]

            result = np.concatenate([filtered_points, filtered_colors], axis=1)

            # 如果还是超过目标，随机采样
            if len(result) > target_count:
                indices = np.random.choice(len(result), target_count, replace=False)
                result = result[indices]

            return result

        else:
            raise ValueError(f"Unknown downsampling method: {method}")

    def _select_complementary_points(
        self,
        reference_points: np.ndarray,
        candidate_points: np.ndarray,
        count: int,
    ) -> np.ndarray:
        """
        从候选点中选择与参考点距离较远的点，用于补充空区域。
        
        使用KNN或距离计算找到距离参考点较远的候选点。
        """
        if len(candidate_points) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        if count <= 0:
            return np.zeros((0, 6), dtype=np.float32)

        if len(reference_points) == 0:
            # 没有参考点，直接返回候选点（限制数量）
            if len(candidate_points) <= count:
                return candidate_points
            indices = np.random.choice(len(candidate_points), count, replace=False)
            return candidate_points[indices]

        # 计算每个候选点到最近参考点的距离
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(reference_points)
        ref_tree = o3d.geometry.KDTreeFlann(ref_pcd)

        candidate_xyz = candidate_points[:, :3]
        distances = []
        for pt in candidate_xyz:
            [_, _, dist] = ref_tree.search_knn_vector_3d(pt, 1)
            distances.append(np.sqrt(dist[0]))

        distances = np.array(distances)
        # 选择距离最远的点（补充稀疏区域）
        far_indices = np.argsort(distances)[::-1][:count]
        return candidate_points[far_indices]

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
