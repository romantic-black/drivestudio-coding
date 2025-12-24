"""
RGB Point Cloud Generator

This module provides point cloud generation functionality for MultiSceneDataset.
"""

import logging
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import open3d as o3d

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)

# Default bounding box for nuScenes
X_MIN, X_MAX = -20, 20
Y_MIN, Y_MAX = -20, 4.8
Z_MIN, Z_MAX = -20, 70


class RGBPointCloudGenerator(ABC):
    """
    RGB 点云生成器基类。
    
    核心功能：
    1. 定义点云生成的抽象接口
    2. 提供通用的辅助方法（边界框、裁剪、滤波等）
    3. 支持多种点云生成策略（单目、立体等）
    """
    
    def __init__(
        self,
        sparsity: Literal['Drop90', 'Drop80', 'Drop50', 'Drop25', 'full'] = 'full',
        filter_sky: bool = True,
        depth_consistency: bool = True,
        use_bbx: bool = True,
        downscale: int = 2,
        bbx_min: Optional[np.ndarray] = None,  # [3] - 自定义边界框最小值
        bbx_max: Optional[np.ndarray] = None,   # [3] - 自定义边界框最大值
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            sparsity: 稀疏度级别（'Drop90', 'Drop80', 'Drop50', 'Drop25', 'full'）
            filter_sky: 是否过滤天空区域
            depth_consistency: 是否进行深度一致性检查
            use_bbx: 是否使用边界框裁剪
            downscale: 点云生成时的下采样倍数
            bbx_min: 自定义边界框最小值（如果为None，使用默认值）
            bbx_max: 自定义边界框最大值（如果为None，使用默认值）
            device: 设备（用于深度图处理）
        """
        self.sparsity = sparsity
        self.filter_sky = filter_sky
        self.depth_consistency = depth_consistency
        self.use_bbx = use_bbx
        self.downscale = downscale
        self.device = device
        
        # Custom bounding box or use defaults
        if bbx_min is not None and bbx_max is not None:
            self.bbx_min = np.array(bbx_min)
            self.bbx_max = np.array(bbx_max)
        else:
            self.bbx_min = None
            self.bbx_max = None
    
    @abstractmethod
    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> o3d.geometry.PointCloud:
        """
        为指定场景和段生成 RGB 点云。
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            pointcloud: Open3D 点云对象，包含位置和颜色
        """
        pass
    
    def get_bbx(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取边界框范围。
        
        Returns:
            bbx_min: [3] - 边界框最小值
            bbx_max: [3] - 边界框最大值
        """
        if self.bbx_min is not None and self.bbx_max is not None:
            return self.bbx_min.copy(), self.bbx_max.copy()
        else:
            # Use default bounding box
            return np.array([X_MIN, Y_MIN, Z_MIN]), np.array([X_MAX, Y_MAX, Z_MAX])
    
    def crop_pointcloud(
        self,
        bbx_min: np.ndarray,
        bbx_max: np.ndarray,
        points: np.ndarray,  # [N, 3]
        colors: np.ndarray,  # [N, 3]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        裁剪点云到边界框。
        
        Args:
            bbx_min: [3] - 边界框最小值
            bbx_max: [3] - 边界框最大值
            points: [N, 3] - 点云位置
            colors: [N, 3] - 点云颜色
            
        Returns:
            cropped_points: [M, 3] - 裁剪后的点云位置
            cropped_colors: [M, 3] - 裁剪后的点云颜色
        """
        mask = (
            (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) &
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) &
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2] + 50)  # Extended Z for background
        )
        return points[mask], colors[mask]
    
    def split_pointcloud(
        self,
        bbx_min: np.ndarray,
        bbx_max: np.ndarray,
        points: np.ndarray,  # [N, 3]
        colors: np.ndarray,  # [N, 3]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        将点云分割为边界框内部和外部两部分。
        
        Args:
            bbx_min: [3] - 边界框最小值
            bbx_max: [3] - 边界框最大值
            points: [N, 3] - 点云位置
            colors: [N, 3] - 点云颜色
            
        Returns:
            inside_points: [M1, 3] - 内部点云位置
            inside_colors: [M1, 3] - 内部点云颜色
            outside_points: [M2, 3] - 外部点云位置
            outside_colors: [M2, 3] - 外部点云颜色
        """
        mask = (
            (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) &
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) &
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])
        )
        inside_points, inside_colors = points[mask], colors[mask]
        outside_points, outside_colors = points[~mask], colors[~mask]
        return inside_points, inside_colors, outside_points, outside_colors
    
    def filter_pointcloud(
        self,
        pointcloud: o3d.geometry.PointCloud,
        use_bbx: bool = True,
    ) -> o3d.geometry.PointCloud:
        """
        对点云进行滤波（统计滤波和均匀下采样）。
        
        Args:
            pointcloud: Open3D 点云对象
            use_bbx: 是否使用边界框（影响滤波参数）
            
        Returns:
            filtered_pointcloud: 滤波后的点云
        """
        if use_bbx:
            # 内部点云使用更严格的滤波参数
            cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.5)
            pointcloud = pointcloud.select_by_index(ind)
            pointcloud = pointcloud.uniform_down_sample(every_k_points=2)
        else:
            # 全局滤波
            cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pointcloud = pointcloud.select_by_index(ind)
            pointcloud = pointcloud.uniform_down_sample(every_k_points=5)
        
        return pointcloud


class MonocularRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    单目 RGB 点云生成器。
    
    从 MultiSceneDataset 的段中生成单目深度点云。
    支持从段内所有帧（或按稀疏度过滤后的帧）生成点云。
    """
    
    def __init__(
        self,
        chosen_cam_ids: List[int] = [0],  # 选择使用的相机ID列表
        sparsity: Literal['Drop90', 'Drop80', 'Drop50', 'Drop25', 'full'] = 'full',
        filter_sky: bool = True,
        depth_consistency: bool = True,
        use_bbx: bool = True,
        downscale: int = 2,
        bbx_min: Optional[np.ndarray] = None,
        bbx_max: Optional[np.ndarray] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            chosen_cam_ids: 选择使用的相机ID列表（例如 [0] 表示只使用前置摄像头）
            sparsity: 稀疏度级别
            filter_sky: 是否过滤天空区域
            depth_consistency: 是否进行深度一致性检查
            use_bbx: 是否使用边界框裁剪
            downscale: 点云生成时的下采样倍数
            bbx_min: 自定义边界框最小值
            bbx_max: 自定义边界框最大值
            device: 设备
        """
        super().__init__(
            sparsity=sparsity,
            filter_sky=filter_sky,
            depth_consistency=depth_consistency,
            use_bbx=use_bbx,
            downscale=downscale,
            bbx_min=bbx_min,
            bbx_max=bbx_max,
            device=device,
        )
        self.chosen_cam_ids = chosen_cam_ids
    
    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> o3d.geometry.PointCloud:
        """
        为指定场景和段生成 RGB 点云。
        
        流程：
        1. 获取段内所有帧索引
        2. 根据稀疏度过滤帧
        3. 加载所有选中帧的 RGB 图像、深度图、外参、内参
        4. 应用深度一致性检查（如果启用）
        5. 生成点云（反投影、变换、累积）
        6. 应用边界框裁剪（如果启用）
        7. 滤波和下采样
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            pointcloud: Open3D 点云对象
        """
        # 1. 获取段内所有帧索引
        frame_indices = self._get_segment_frames(dataset, scene_id, segment_id)
        if len(frame_indices) == 0:
            raise ValueError(f"Segment {segment_id} in scene {scene_id} has no frames")
        
        # 2. 根据稀疏度过滤帧
        filtered_frame_indices = self._apply_sparsity_filter(frame_indices)
        if len(filtered_frame_indices) == 0:
            raise ValueError(f"No frames selected after sparsity filtering")
        
        # 3. 按相机分组加载帧数据
        frame_data_by_camera = {cam_id: [] for cam_id in self.chosen_cam_ids}
        for frame_idx in filtered_frame_indices:
            for cam_id in self.chosen_cam_ids:
                frame_data = self._load_frame_data(dataset, scene_id, frame_idx, cam_id)
                if frame_data is not None:
                    frame_data_by_camera[cam_id].append((frame_idx, frame_data))
        
        # 检查是否有有效数据
        total_frames = sum(len(frames) for frames in frame_data_by_camera.values())
        if total_frames == 0:
            raise ValueError("No valid frame data loaded")
        
        # 获取图像尺寸（从第一个有效帧）
        first_cam_id = next(iter([cam_id for cam_id in self.chosen_cam_ids if len(frame_data_by_camera[cam_id]) > 0]))
        H, W = frame_data_by_camera[first_cam_id][0][1]['rgb'].shape[:2]
        
        # 4. 对每个相机分别进行深度一致性检查
        consistency_masks_by_camera = {}
        frame_data_list_by_camera = {}
        for cam_id in self.chosen_cam_ids:
            frames = frame_data_by_camera[cam_id]
            if len(frames) == 0:
                continue
            
            # 按帧索引排序
            frames_sorted = sorted(frames, key=lambda x: x[0])
            frame_data_list = [fd for _, fd in frames_sorted]
            frame_data_list_by_camera[cam_id] = frame_data_list
            
            # 对每个相机单独进行深度一致性检查
            if self.depth_consistency:
                consistency_masks_by_camera[cam_id] = self._depth_consistency_check(frame_data_list, H, W)
            else:
                consistency_masks_by_camera[cam_id] = [np.ones((H, W), dtype=bool) for _ in frame_data_list]
        
        # 5. 生成点云（合并所有相机的数据）
        accumulated_pointcloud = self._generate_pointcloud_from_frames_by_camera(
            frame_data_list_by_camera, consistency_masks_by_camera, H, W
        )
        
        # 转换为 Open3D 点云
        points = accumulated_pointcloud[:, :3]  # [N, 3]
        colors = accumulated_pointcloud[:, 3:6]  # [N, 3]
        
        # 确保颜色在 [0, 1] 范围内
        colors = np.clip(colors, 0.0, 1.0)
        
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        # 6. 应用边界框裁剪（如果启用）
        if self.use_bbx:
            bbx_min, bbx_max = self.get_bbx()
            inside_points, inside_colors, outside_points, outside_colors = self.split_pointcloud(
                bbx_min, bbx_max, points, colors
            )
            
            # 分别滤波内部和外部点云
            if len(inside_points) > 0:
                inside_pcd = o3d.geometry.PointCloud()
                inside_pcd.points = o3d.utility.Vector3dVector(inside_points)
                inside_pcd.colors = o3d.utility.Vector3dVector(inside_colors)
                inside_pcd = self.filter_pointcloud(inside_pcd, use_bbx=True)
                
                if len(outside_points) > 0:
                    outside_pcd = o3d.geometry.PointCloud()
                    outside_pcd.points = o3d.utility.Vector3dVector(outside_points)
                    outside_pcd.colors = o3d.utility.Vector3dVector(outside_colors)
                    outside_pcd = self.filter_pointcloud(outside_pcd, use_bbx=False)
                    
                    # 合并内部和外部点云
                    pointcloud = inside_pcd + outside_pcd
                else:
                    pointcloud = inside_pcd
            elif len(outside_points) > 0:
                outside_pcd = o3d.geometry.PointCloud()
                outside_pcd.points = o3d.utility.Vector3dVector(outside_points)
                outside_pcd.colors = o3d.utility.Vector3dVector(outside_colors)
                pointcloud = self.filter_pointcloud(outside_pcd, use_bbx=False)
        else:
            # 全局滤波
            pointcloud = self.filter_pointcloud(pointcloud, use_bbx=False)
        
        return pointcloud
    
    def _get_segment_frames(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> List[int]:
        """
        获取段内所有帧索引。
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID
            
        Returns:
            frame_indices: 段内所有帧索引列表
        """
        return dataset.get_segment_frames(scene_id, segment_id)
    
    def _apply_sparsity_filter(
        self,
        frame_indices: List[int],
    ) -> List[int]:
        """
        根据稀疏度级别过滤帧。
        
        Args:
            frame_indices: 原始帧索引列表
            
        Returns:
            filtered_frame_indices: 过滤后的帧索引列表
        """
        if self.sparsity == 'full':
            return frame_indices
        
        # 按位置过滤（保持原始顺序）
        filtered = []
        for frame_pos, frame_idx in enumerate(frame_indices):
            if self.sparsity == "Drop50":
                if frame_pos % 4 == 2 or frame_pos % 4 == 3:
                    continue  # 保留50%的帧
            elif self.sparsity == 'Drop80':
                if frame_pos % 5 != 0:  # 保留20%的帧
                    continue
            elif self.sparsity == 'Drop25':
                if frame_pos % 4 == 2:  # 保留75%的帧
                    continue
            elif self.sparsity == 'Drop90':
                if frame_pos % 10 != 0:  # 保留10%的帧
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
        """
        加载指定帧和相机的数据。
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            frame_idx: 帧索引
            cam_id: 相机ID（在 camera_list 中的索引）
            
        Returns:
            Dict包含：
                - 'rgb': np.ndarray [H, W, 3] - RGB图像（归一化到[0,1]）
                - 'depth': np.ndarray [H, W] - 深度图
                - 'extrinsic': np.ndarray [4, 4] - 外参（cam_to_world）
                - 'intrinsic': np.ndarray [3, 3] - 内参（3x3矩阵）
            None 如果数据加载失败
        """
        try:
            frame_data = dataset.get_frame_data(scene_id, frame_idx, cam_id)
            
            # 转换为numpy数组
            rgb = frame_data['image'].cpu().numpy()  # [H, W, 3]
            depth = frame_data['depth'].cpu().numpy()  # [H, W]
            extrinsic = frame_data['extrinsic'].cpu().numpy()  # [4, 4]
            
            # 转换内参为3x3（如果原本是4x4）
            intrinsic = frame_data['intrinsic'].cpu().numpy()  # [3, 3] or [4, 4]
            if intrinsic.shape == (4, 4):
                intrinsic = intrinsic[:3, :3]
            
            # 获取天空掩码（如果存在）
            sky_mask = frame_data.get('sky_mask')
            if sky_mask is not None:
                # 保持为 Tensor，稍后在生成点云时转换为 numpy
                pass
            
            return {
                'rgb': rgb,
                'depth': depth,
                'extrinsic': extrinsic,
                'intrinsic': intrinsic,
                'sky_mask': sky_mask,  # Tensor [H, W] or None
            }
        except Exception as e:
            logger.warning(f"Failed to load frame data for scene {scene_id}, frame {frame_idx}, cam {cam_id}: {e}")
            return None
    
    def _depth_consistency_check(
        self,
        frame_data_list: List[Dict],
        H: int,
        W: int,
    ) -> List[np.ndarray]:
        """
        检查连续帧之间的深度一致性。
        
        Args:
            frame_data_list: 帧数据列表（按时间顺序）
            H: 图像高度
            W: 图像宽度
            
        Returns:
            consistency_masks: List[np.ndarray] - 每个帧的一致性掩码 [H, W]
        """
        if not self.depth_consistency:
            return [np.ones((H, W), dtype=bool) for _ in frame_data_list]
        
        depth_masks = []
        last_depth = None
        
        for i, frame_data in enumerate(frame_data_list):
            depth = frame_data['depth']  # [H, W]
            
            if i == 0:
                # 第一帧假设正确
                last_depth = depth.copy()
                depth_masks.append(np.ones((H, W), dtype=bool))
                continue
            
            # 获取当前帧和上一帧的外参和内参
            c2w = frame_data['extrinsic']  # [4, 4]
            last_c2w = frame_data_list[i-1]['extrinsic']  # [4, 4]
            K = frame_data['intrinsic']  # [3, 3] - 当前帧内参
            last_K = frame_data_list[i-1]['intrinsic']  # [3, 3] - 上一帧内参
            
            # 反投影当前帧的深度到3D点（使用当前帧内参）
            x = np.arange(0, W)
            y = np.arange(0, H)
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack([xx.ravel(), yy.ravel()]).T  # [H*W, 2]
            
            cx, cy = K[0, 2], K[1, 2]
            fx, fy = K[0, 0], K[1, 1]
            
            x_cam = (pixels[:, 0] - cx) * depth.ravel() / fx
            y_cam = (pixels[:, 1] - cy) * depth.ravel() / fy
            z_cam = depth.ravel()
            coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)  # [H*W, 3]
            
            # 变换到上一帧的坐标系
            trans_mat = np.linalg.inv(last_c2w) @ c2w
            coordinates_homo = np.column_stack([coordinates, np.ones(len(coordinates))])  # [H*W, 4]
            last_coordinates = (trans_mat @ coordinates_homo.T).T  # [H*W, 4]
            
            # 投影到上一帧的图像平面（使用上一帧内参）
            last_cx, last_cy = last_K[0, 2], last_K[1, 2]
            last_fx, last_fy = last_K[0, 0], last_K[1, 1]
            last_x = (last_fx * last_coordinates[:, 0] + last_cx * last_coordinates[:, 2]) / last_coordinates[:, 2]
            last_y = (last_fy * last_coordinates[:, 1] + last_cy * last_coordinates[:, 2]) / last_coordinates[:, 2]
            last_pixels = np.vstack([last_x, last_y]).T  # [H*W, 2]
            
            # 检查投影位置是否在图像范围内
            valid_mask = (
                (last_pixels[:, 0] >= 0) & (last_pixels[:, 0] < W) &
                (last_pixels[:, 1] >= 0) & (last_pixels[:, 1] < H) &
                (last_coordinates[:, 2] > 0)  # 深度为正
            )
            
            # 计算深度差异
            depth_mask = np.ones(H * W, dtype=bool)
            if np.any(valid_mask):
                last_pixels_int = last_pixels[valid_mask].astype(int)
                last_pixels_int[:, 0] = np.clip(last_pixels_int[:, 0], 0, W - 1)
                last_pixels_int[:, 1] = np.clip(last_pixels_int[:, 1], 0, H - 1)
                
                depth_diff = np.abs(
                    depth.ravel()[valid_mask] -
                    last_depth[last_pixels_int[:, 1], last_pixels_int[:, 0]]
                )
                
                # 深度差异小于平均值的点认为是有效的
                depth_mask[valid_mask] = depth_diff < depth_diff.mean()
            
            depth_mask = depth_mask.reshape(H, W)
            depth_masks.append(depth_mask)
            
            # 更新上一帧的深度
            last_depth = depth.copy()
        
        return depth_masks
    
    def _generate_pointcloud_from_frames_by_camera(
        self,
        frame_data_list_by_camera: Dict[int, List[Dict]],
        consistency_masks_by_camera: Dict[int, List[np.ndarray]],
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        从按相机分组的帧数据生成点云。
        
        Args:
            frame_data_list_by_camera: 按相机分组的帧数据字典 {cam_id: [frame_data, ...]}
            consistency_masks_by_camera: 按相机分组的一致性掩码字典 {cam_id: [mask, ...]}
            H: 图像高度
            W: 图像宽度
            
        Returns:
            pointcloud: [N, 6] - 点云数据（前3列是位置，后3列是颜色）
        """
        color_pointclouds = []
        
        # 初始化下采样掩码
        if self.downscale != 1:
            downscale_mask = np.zeros((H, W), dtype=bool)
            downscale_mask[::self.downscale, ::self.downscale] = True
        else:
            downscale_mask = None
        
        # 遍历每个相机的帧数据
        for cam_id, frame_data_list in frame_data_list_by_camera.items():
            consistency_masks = consistency_masks_by_camera[cam_id]
            
            # 遍历该相机的所有帧
            for i, frame_data in enumerate(frame_data_list):
                rgb = frame_data['rgb']  # [H, W, 3]
                depth = frame_data['depth']  # [H, W]
                extrinsic = frame_data['extrinsic']  # [4, 4]
                intrinsic = frame_data['intrinsic']  # [3, 3]
                
                # 应用一致性掩码
                consistency_mask = consistency_masks[i]  # [H, W]
                
                # 应用天空过滤（如果启用）
                sky_mask = frame_data.get('sky_mask')
                if sky_mask is not None:
                    # 转换为 numpy 数组
                    if isinstance(sky_mask, torch.Tensor):
                        sky_mask = sky_mask.cpu().numpy()
                    if self.filter_sky:
                        # 天空掩码为 True 表示天空区域，需要取反（保留非天空区域）
                        sky_mask = sky_mask.astype(bool)
                    else:
                        sky_mask = np.ones((H, W), dtype=bool)
                else:
                    # 如果没有天空掩码，根据 filter_sky 决定
                    if self.filter_sky:
                        # 如果启用天空过滤但没有掩码，发出警告但继续处理
                        logger.warning(f"No sky mask available for camera {cam_id}, frame {i}, skipping sky filtering")
                        sky_mask = np.ones((H, W), dtype=bool)
                    else:
                        sky_mask = np.ones((H, W), dtype=bool)
                
                # 应用下采样掩码
                if downscale_mask is not None:
                    final_mask = consistency_mask & sky_mask & downscale_mask
                else:
                    final_mask = consistency_mask & sky_mask
                
                # 提取有效像素
                kept = np.argwhere(final_mask)
                if len(kept) == 0:
                    continue
                
                depth_values = depth[kept[:, 0], kept[:, 1]]
                rgb_values = rgb[kept[:, 0], kept[:, 1]]
                
                # 过滤无效深度值
                valid_depth_mask = np.isfinite(depth_values) & (depth_values > 0)
                if not np.any(valid_depth_mask):
                    continue
                
                depth_values = depth_values[valid_depth_mask]
                rgb_values = rgb_values[valid_depth_mask]
                kept_valid = kept[valid_depth_mask]
                
                # 反投影到相机坐标系
                pixel_coords = kept_valid[:, [1, 0]]  # [x, y] 格式
                x_cam = (pixel_coords[:, 0] - intrinsic[0, 2]) * depth_values / intrinsic[0, 0]
                y_cam = (pixel_coords[:, 1] - intrinsic[1, 2]) * depth_values / intrinsic[1, 1]
                z_cam = depth_values
                coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)  # [N, 3]
                
                # 过滤NaN/inf坐标
                valid_coords_mask = np.isfinite(coordinates).all(axis=1)
                if not np.any(valid_coords_mask):
                    continue
                
                coordinates = coordinates[valid_coords_mask]
                rgb_values = rgb_values[valid_coords_mask]
                coordinates_homo = np.column_stack([coordinates, np.ones(len(coordinates))])  # [N, 4]
                
                # 变换到世界坐标系
                worlds = (extrinsic @ coordinates_homo.T).T  # [N, 4]
                worlds = worlds[:, :3]  # [N, 3]
                
                # 过滤NaN/inf世界坐标
                valid_worlds_mask = np.isfinite(worlds).all(axis=1)
                if not np.any(valid_worlds_mask):
                    continue
                
                worlds = worlds[valid_worlds_mask]
                rgb_values = rgb_values[valid_worlds_mask]
                
                # 累积点云块
                point_cloud_chunk = np.concatenate([worlds, rgb_values], axis=-1)  # [N, 6]
                color_pointclouds.append(point_cloud_chunk)
        
        # 合并所有点云块
        if len(color_pointclouds) == 0:
            raise ValueError("No valid point cloud generated")
        
        accumulated_pointcloud = np.concatenate(color_pointclouds, axis=0)  # [M, 6]
        
        # 最终过滤：移除剩余的NaN/inf值
        valid_mask = np.isfinite(accumulated_pointcloud[:, :3]).all(axis=1)
        accumulated_pointcloud = accumulated_pointcloud[valid_mask]
        
        return accumulated_pointcloud
    
    def _generate_pointcloud_from_frames(
        self,
        frame_data_list: List[Dict],
        consistency_masks: List[np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        从帧数据生成点云。
        
        Args:
            frame_data_list: 帧数据列表
            consistency_masks: 一致性掩码列表
            H: 图像高度
            W: 图像宽度
            
        Returns:
            pointcloud: [N, 6] - 点云数据（前3列是位置，后3列是颜色）
        """
        color_pointclouds = []
        
        # 初始化下采样掩码
        if self.downscale != 1:
            downscale_mask = np.zeros((H, W), dtype=bool)
            downscale_mask[::self.downscale, ::self.downscale] = True
        else:
            downscale_mask = None
        
        # 遍历所有帧数据
        for i, frame_data in enumerate(frame_data_list):
            rgb = frame_data['rgb']  # [H, W, 3]
            depth = frame_data['depth']  # [H, W]
            extrinsic = frame_data['extrinsic']  # [4, 4]
            intrinsic = frame_data['intrinsic']  # [3, 3]
            
            # 应用一致性掩码
            consistency_mask = consistency_masks[i]  # [H, W]
            
            # 应用天空过滤（如果启用）
            if self.filter_sky:
                # TODO: 从 MultiSceneDataset 获取天空掩码
                # 目前先跳过
                sky_mask = np.ones((H, W), dtype=bool)
            else:
                sky_mask = np.ones((H, W), dtype=bool)
            
            # 应用下采样掩码
            if downscale_mask is not None:
                final_mask = consistency_mask & sky_mask & downscale_mask
            else:
                final_mask = consistency_mask & sky_mask
            
            # 提取有效像素
            kept = np.argwhere(final_mask)
            if len(kept) == 0:
                continue
            
            depth_values = depth[kept[:, 0], kept[:, 1]]
            rgb_values = rgb[kept[:, 0], kept[:, 1]]
            
            # 过滤无效深度值
            valid_depth_mask = np.isfinite(depth_values) & (depth_values > 0)
            if not np.any(valid_depth_mask):
                continue
            
            depth_values = depth_values[valid_depth_mask]
            rgb_values = rgb_values[valid_depth_mask]
            kept_valid = kept[valid_depth_mask]
            
            # 反投影到相机坐标系
            pixel_coords = kept_valid[:, [1, 0]]  # [x, y] 格式
            x_cam = (pixel_coords[:, 0] - intrinsic[0, 2]) * depth_values / intrinsic[0, 0]
            y_cam = (pixel_coords[:, 1] - intrinsic[1, 2]) * depth_values / intrinsic[1, 1]
            z_cam = depth_values
            coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)  # [N, 3]
            
            # 过滤NaN/inf坐标
            valid_coords_mask = np.isfinite(coordinates).all(axis=1)
            if not np.any(valid_coords_mask):
                continue
            
            coordinates = coordinates[valid_coords_mask]
            rgb_values = rgb_values[valid_coords_mask]
            coordinates_homo = np.column_stack([coordinates, np.ones(len(coordinates))])  # [N, 4]
            
            # 变换到世界坐标系
            worlds = (extrinsic @ coordinates_homo.T).T  # [N, 4]
            worlds = worlds[:, :3]  # [N, 3]
            
            # 过滤NaN/inf世界坐标
            valid_worlds_mask = np.isfinite(worlds).all(axis=1)
            if not np.any(valid_worlds_mask):
                continue
            
            worlds = worlds[valid_worlds_mask]
            rgb_values = rgb_values[valid_worlds_mask]
            
            # 累积点云块
            point_cloud_chunk = np.concatenate([worlds, rgb_values], axis=-1)  # [N, 6]
            color_pointclouds.append(point_cloud_chunk)
        
        # 合并所有点云块
        if len(color_pointclouds) == 0:
            raise ValueError("No valid point cloud generated")
        
        accumulated_pointcloud = np.concatenate(color_pointclouds, axis=0)  # [M, 6]
        
        # 最终过滤：移除剩余的NaN/inf值
        valid_mask = np.isfinite(accumulated_pointcloud[:, :3]).all(axis=1)
        accumulated_pointcloud = accumulated_pointcloud[valid_mask]
        
        return accumulated_pointcloud

