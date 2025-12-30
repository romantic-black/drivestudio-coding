"""
RGB Point Cloud Generator

This module provides point cloud generation functionality for MultiSceneDataset.
"""

import logging
import sys
import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import open3d as o3d

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)


@dataclass
class StaticPointCloud:
    """
    静态点云数据结构。
    
    核心设计：
    1. 点云按帧组织（可选）
    2. 点云使用世界坐标系
    3. 可以跨帧累积
    """
    # 按帧组织的静态点云列表
    # frame_points[i] = (N, 6) 世界坐标 + RGB
    frame_points: List[np.ndarray]
    
    def get_merged_points(self) -> np.ndarray:
        """
        合并所有帧的静态点云。
        
        Returns:
            points: (N, 6) - 世界坐标 + RGB
        """
        if len(self.frame_points) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        
        return np.concatenate(self.frame_points, axis=0)
    
    def get_frame_points(
        self,
        frame_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        获取指定帧的静态点云。
        
        Args:
            frame_indices: 帧索引列表（如果为None，返回所有帧的点）
            
        Returns:
            points: (N, 6) - 世界坐标 + RGB
        """
        if frame_indices is None:
            return self.get_merged_points()
        
        points_list = []
        for frame_idx in frame_indices:
            if 0 <= frame_idx < len(self.frame_points):
                points_list.append(self.frame_points[frame_idx])
        
        if len(points_list) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        
        return np.concatenate(points_list, axis=0)


@dataclass
class DynamicPointCloud:
    """
    动态点云数据结构。
    
    核心设计：
    1. 点云按实例ID组织
    2. 每个实例的点云按帧索引组织
    3. 点云使用物体局部坐标系
    4. 包含实例的位姿和尺寸信息
    """
    # 实例ID映射：原始ID -> 连续int ID（从1开始）
    instance_id_mapping: Dict[int, int]  # waymoid2intid
    
    # 动态点云：intid2inboxpoints[intid][frame_idx] = (N, 6) 局部坐标 + RGB
    # 格式：[x_local, y_local, z_local, r, g, b]
    points_by_instance: Dict[int, Dict[int, np.ndarray]]  # intid2inboxpoints
    
    # 实例信息：每个实例的位姿和尺寸
    # instances_info[intid] = {
    #     "poses": np.ndarray,  # (num_frames, 4, 4) - Object->World 变换
    #     "size": np.ndarray,   # (3,) - 边界框尺寸 [l, w, h]
    #     "frame_info": np.ndarray,  # (num_frames,) - 每帧是否出现
    # }
    instances_info: Dict[int, Dict[str, np.ndarray]]
    
    def get_instance_points(
        self,
        instance_id: int,
        frame_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        获取指定实例的点云（局部坐标）。
        
        Args:
            instance_id: 实例ID（连续int ID）
            frame_indices: 帧索引列表（如果为None，返回所有帧的点）
            
        Returns:
            points: (N, 6) - 局部坐标 + RGB
        """
        if instance_id not in self.points_by_instance:
            return np.zeros((0, 6), dtype=np.float32)
        
        frame_dict = self.points_by_instance[instance_id]
        if frame_indices is None:
            frame_indices = list(frame_dict.keys())
        
        points_list = []
        for frame_idx in frame_indices:
            if frame_idx in frame_dict:
                points_list.append(frame_dict[frame_idx])
        
        if len(points_list) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        
        return np.concatenate(points_list, axis=0)
    
    def transform_to_world(
        self,
        instance_id: int,
        frame_idx: int,
    ) -> np.ndarray:
        """
        将指定实例的点云变换到世界坐标系。
        
        Args:
            instance_id: 实例ID（连续int ID）
            frame_idx: 帧索引
            
        Returns:
            points_world: (N, 6) - 世界坐标 + RGB
        """
        if instance_id not in self.points_by_instance:
            return np.zeros((0, 6), dtype=np.float32)
        
        if frame_idx not in self.points_by_instance[instance_id]:
            return np.zeros((0, 6), dtype=np.float32)
        
        points_local = self.points_by_instance[instance_id][frame_idx]  # (N, 6)
        
        if instance_id not in self.instances_info:
            return points_local
        
        pose = self.instances_info[instance_id]["poses"][frame_idx]  # (4, 4)
        T_ow = pose  # Object->World
        
        # 变换到世界坐标
        points_local_xyz = points_local[:, :3]  # (N, 3)
        points_local_homo = np.concatenate([
            points_local_xyz,
            np.ones((points_local_xyz.shape[0], 1), dtype=np.float32)
        ], axis=1)  # (N, 4)
        
        points_world_xyz = (T_ow @ points_local_homo.T).T[:, :3]  # (N, 3)
        points_world = np.concatenate([
            points_world_xyz,
            points_local[:, 3:6]  # RGB
        ], axis=1)  # (N, 6)
        
        return points_world


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
        crop_aabb: np.ndarray = None,  # [2, 3] - 裁剪边界框 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        input_aabb: np.ndarray = None,  # [2, 3] - 输入边界框（用于分割和滤波）[[x_min, y_min, z_min], [x_max, y_max, z_max]]
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            sparsity: 稀疏度级别（'Drop90', 'Drop80', 'Drop50', 'Drop25', 'full'）
            filter_sky: 是否过滤天空区域
            depth_consistency: 是否进行深度一致性检查
            use_bbx: 是否使用边界框裁剪
            downscale: 点云生成时的下采样倍数
            crop_aabb: 裁剪边界框，shape [2, 3]，格式 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                      用于裁剪时移除超出边界框的点云
            input_aabb: 输入边界框，shape [2, 3]，格式 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                       用于分割和滤波时区分内部和外部点云
            device: 设备（用于深度图处理）
        """
        self.sparsity = sparsity
        self.filter_sky = filter_sky
        self.depth_consistency = depth_consistency
        self.use_bbx = use_bbx
        self.downscale = downscale
        self.device = device
        
        # Validate and store crop_aabb
        if crop_aabb is None:
            raise ValueError("crop_aabb must be provided (shape [2, 3])")
        crop_aabb = np.array(crop_aabb)
        if crop_aabb.shape != (2, 3):
            raise ValueError(f"crop_aabb must have shape [2, 3], got {crop_aabb.shape}")
        if not np.all(crop_aabb[0] < crop_aabb[1]):
            raise ValueError("crop_aabb min must be less than max for all dimensions")
        self.crop_aabb = crop_aabb
        
        # Validate and store input_aabb
        if input_aabb is None:
            raise ValueError("input_aabb must be provided (shape [2, 3])")
        input_aabb = np.array(input_aabb)
        if input_aabb.shape != (2, 3):
            raise ValueError(f"input_aabb must have shape [2, 3], got {input_aabb.shape}")
        if not np.all(input_aabb[0] < input_aabb[1]):
            raise ValueError("input_aabb min must be less than max for all dimensions")
        self.input_aabb = input_aabb
    
    @abstractmethod
    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> o3d.geometry.PointCloud:
        """
        为指定场景和段生成 RGB 点云（基类接口，返回合并后的点云）。
        
        此方法用于向后兼容，实际实现应调用 generate_pointcloud_with_static_dynamic()。
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            pointcloud: Open3D 点云对象，包含位置和颜色
        """
        pass
    
    @abstractmethod
    def generate_pointcloud_with_static_dynamic(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> Tuple[StaticPointCloud, DynamicPointCloud]:
        """
        为指定场景和段生成 RGB 点云（包含静动态分割）。
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            static_pc: StaticPointCloud - 静态点云
            dynamic_pc: DynamicPointCloud - 动态点云
        """
        pass
    
    def get_crop_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取裁剪边界框范围。
        
        Returns:
            crop_min: [3] - 裁剪边界框最小值
            crop_max: [3] - 裁剪边界框最大值
        """
        return self.crop_aabb[0].copy(), self.crop_aabb[1].copy()
    
    def get_input_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取输入边界框范围（用于分割和滤波）。
        
        Returns:
            input_min: [3] - 输入边界框最小值
            input_max: [3] - 输入边界框最大值
        """
        return self.input_aabb[0].copy(), self.input_aabb[1].copy()
    
    def crop_pointcloud(
        self,
        crop_min: np.ndarray,
        crop_max: np.ndarray,
        points: np.ndarray,  # [N, 3]
        colors: np.ndarray,  # [N, 3]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        裁剪点云到边界框（移除超出边界框的点）。
        
        Args:
            crop_min: [3] - 裁剪边界框最小值
            crop_max: [3] - 裁剪边界框最大值
            points: [N, 3] - 点云位置
            colors: [N, 3] - 点云颜色
            
        Returns:
            cropped_points: [M, 3] - 裁剪后的点云位置
            cropped_colors: [M, 3] - 裁剪后的点云颜色
        """
        mask = (
            (points[:, 0] > crop_min[0]) & (points[:, 0] < crop_max[0]) &
            (points[:, 1] > crop_min[1]) & (points[:, 1] < crop_max[1]) &
            (points[:, 2] > crop_min[2]) & (points[:, 2] < crop_max[2])
        )
        return points[mask], colors[mask]
    
    def split_pointcloud(
        self,
        input_min: np.ndarray,
        input_max: np.ndarray,
        points: np.ndarray,  # [N, 3]
        colors: np.ndarray,  # [N, 3]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        将点云分割为边界框内部和外部两部分。
        
        Args:
            input_min: [3] - 输入边界框最小值
            input_max: [3] - 输入边界框最大值
            points: [N, 3] - 点云位置
            colors: [N, 3] - 点云颜色
            
        Returns:
            inside_points: [M1, 3] - 内部点云位置
            inside_colors: [M1, 3] - 内部点云颜色
            outside_points: [M2, 3] - 外部点云位置
            outside_colors: [M2, 3] - 外部点云颜色
        """
        mask = (
            (points[:, 0] > input_min[0]) & (points[:, 0] < input_max[0]) &
            (points[:, 1] > input_min[1]) & (points[:, 1] < input_max[1]) &
            (points[:, 2] > input_min[2]) & (points[:, 2] < input_max[2])
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
        crop_aabb: np.ndarray = None,  # [2, 3] - 裁剪边界框
        input_aabb: np.ndarray = None,  # [2, 3] - 输入边界框
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
            crop_aabb: 裁剪边界框，shape [2, 3]，格式 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            input_aabb: 输入边界框，shape [2, 3]，格式 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            device: 设备
        """
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
        
        # 5. 使用新接口生成点云（包含静动态分割）
        static_pc, _ = self.generate_pointcloud_with_static_dynamic(
            dataset, scene_id, segment_id
        )
        
        # 合并所有帧的静态点
        all_points = static_pc.get_merged_points()
        if all_points.shape[0] == 0:
            # 返回空点云
            pointcloud = o3d.geometry.PointCloud()
            return pointcloud
        
        points = all_points[:, :3]  # [N, 3]
        colors = all_points[:, 3:6]  # [N, 3]
        
        # 确保颜色在 [0, 1] 范围内
        colors = np.clip(colors, 0.0, 1.0)
        
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        # 6. 应用边界框裁剪（如果启用）
        if self.use_bbx:
            crop_min, crop_max = self.get_crop_aabb()
            input_min, input_max = self.get_input_aabb()
            # 先裁剪：使用 crop_aabb 移除超出边界框的点
            points, colors = self.crop_pointcloud(crop_min, crop_max, points, colors)
            # 再分割：使用 input_aabb 分割为内部和外部点云
            inside_points, inside_colors, outside_points, outside_colors = self.split_pointcloud(
                input_min, input_max, points, colors
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
    
    def generate_pointcloud_with_static_dynamic(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> Tuple[StaticPointCloud, DynamicPointCloud]:
        """
        为指定场景和段生成 RGB 点云（包含静动态分割）。
        
        流程：
        1. 从单目深度图生成点云（世界坐标）
        2. 使用实例信息分割静动态点
        3. 静态点保存为世界坐标
        4. 动态点转换为物体局部坐标
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            static_pc: StaticPointCloud - 静态点云
            dynamic_pc: DynamicPointCloud - 动态点云
        """
        # 1. 获取段内所有帧索引
        frame_indices = self._get_segment_frames(dataset, scene_id, segment_id)
        if len(frame_indices) == 0:
            raise ValueError(f"Segment {segment_id} in scene {scene_id} has no frames")
        
        # 2. 根据稀疏度过滤帧
        filtered_frame_indices = self._apply_sparsity_filter(frame_indices)
        if len(filtered_frame_indices) == 0:
            raise ValueError(f"No frames selected after sparsity filtering")
        
        # 3. 获取场景数据
        scene_data = dataset.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")
        
        scene_dataset = scene_data['dataset']
        
        # 4. 预加载实例信息
        waymoid2intid_global, id2framePoseSize, frame_instances = self._load_instances_info(
            scene_data, scene_id
        )
        
        # 5. 按相机分组加载帧数据
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
        
        # 6. 对每个相机分别进行深度一致性检查
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
        
        # 7. 生成点云（按帧组织）
        static_frame_points = []
        dynamic_points_by_instance = {}
        
        # 创建帧索引到段内索引的映射
        frame_idx_to_segment_idx = {frame_idx: i for i, frame_idx in enumerate(filtered_frame_indices)}
        
        # 遍历每个相机的帧数据
        for cam_id, frame_data_list in frame_data_list_by_camera.items():
            consistency_masks = consistency_masks_by_camera[cam_id]
            
            # 遍历该相机的所有帧
            for frame_data_idx, frame_data in enumerate(frame_data_list):
                # 获取原始帧索引（需要从 frame_data_by_camera 中获取）
                # 由于已经排序，我们需要找到对应的原始帧索引
                sorted_frames = sorted(frame_data_by_camera[cam_id], key=lambda x: x[0])
                frame_idx = sorted_frames[frame_data_idx][0]
                segment_idx = frame_idx_to_segment_idx.get(frame_idx, frame_data_idx)
                
                rgb = frame_data['rgb']  # [H, W, 3]
                depth = frame_data['depth']  # [H, W]
                extrinsic = frame_data['extrinsic']  # [4, 4]
                intrinsic = frame_data['intrinsic']  # [3, 3]
                
                # 应用一致性掩码
                consistency_mask = consistency_masks[frame_data_idx]  # [H, W]
                
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
                        logger.warning(f"No sky mask available for camera {cam_id}, frame {frame_idx}, skipping sky filtering")
                        sky_mask = np.ones((H, W), dtype=bool)
                    else:
                        sky_mask = np.ones((H, W), dtype=bool)
                
                # 应用下采样掩码
                if self.downscale != 1:
                    downscale_mask = np.zeros((H, W), dtype=bool)
                    downscale_mask[::self.downscale, ::self.downscale] = True
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
                
                # 组合世界坐标 + RGB
                pts_wrgb = np.concatenate([worlds, rgb_values], axis=-1)  # [N, 6]
                
                # 8. 获取当前帧的实例列表
                waymoid2intid, inst_list = self._get_instances_for_frame(
                    waymoid2intid_global, id2framePoseSize, frame_instances, frame_idx, scene_dataset
                )
                
                # 9. 分割静动态点
                bg_points, dynamic_points = self._split_static_dynamic(pts_wrgb, inst_list)
                
                # 10. 保存静态背景点（按段内索引组织）
                if segment_idx >= len(static_frame_points):
                    # 扩展列表以容纳该索引
                    static_frame_points.extend([np.zeros((0, 6), dtype=np.float32)] * (segment_idx + 1 - len(static_frame_points)))
                static_frame_points[segment_idx] = np.concatenate([
                    static_frame_points[segment_idx],
                    bg_points.astype(np.float32)
                ], axis=0) if static_frame_points[segment_idx].shape[0] > 0 else bg_points.astype(np.float32)
                
                # 11. 保存动态物体点（按实例ID和段内索引）
                for intid, po_rgb in dynamic_points.items():
                    if intid not in dynamic_points_by_instance:
                        dynamic_points_by_instance[intid] = {}
                    if segment_idx not in dynamic_points_by_instance[intid]:
                        dynamic_points_by_instance[intid][segment_idx] = []
                    dynamic_points_by_instance[intid][segment_idx].append(po_rgb.astype(np.float32))
        
        # 12. 合并同一实例同一帧的多块点云
        for intid in dynamic_points_by_instance:
            for segment_idx in dynamic_points_by_instance[intid]:
                if isinstance(dynamic_points_by_instance[intid][segment_idx], list):
                    if len(dynamic_points_by_instance[intid][segment_idx]) > 0:
                        dynamic_points_by_instance[intid][segment_idx] = np.concatenate(
                            dynamic_points_by_instance[intid][segment_idx], axis=0
                        )
                    else:
                        dynamic_points_by_instance[intid][segment_idx] = np.zeros((0, 6), dtype=np.float32)
        
        # 13. 构建 instances_info
        instances_info = {}
        num_frames = len(filtered_frame_indices)
        
        # 遍历所有实例ID
        for intid in dynamic_points_by_instance.keys():
            # 找到对应的原始ID (sid)
            sid = None
            for orig_sid, mapped_intid in waymoid2intid_global.items():
                if mapped_intid == intid:
                    sid = orig_sid
                    break
            
            if sid is None or sid not in id2framePoseSize:
                continue
            
            # 构建该实例的位姿数组和帧信息
            poses_list = []
            frame_info_list = []
            
            for segment_idx, frame_idx in enumerate(filtered_frame_indices):
                absolute_frame_idx = self._get_absolute_frame_idx(scene_dataset, frame_idx)
                
                if absolute_frame_idx in id2framePoseSize[sid]:
                    T_ow, size = id2framePoseSize[sid][absolute_frame_idx]
                    poses_list.append(T_ow)
                    frame_info_list.append(True)
                else:
                    # 该帧没有该实例，使用单位矩阵作为占位符
                    poses_list.append(np.eye(4, dtype=np.float32))
                    frame_info_list.append(False)
            
            instances_info[intid] = {
                "poses": np.stack(poses_list, axis=0),  # (num_frames, 4, 4)
                "size": id2framePoseSize[sid][list(id2framePoseSize[sid].keys())[0]][1],  # (3,) - 使用第一帧的尺寸
                "frame_info": np.array(frame_info_list, dtype=bool),  # (num_frames,)
            }
        
        # 14. 构建并返回新数据结构
        static_pc = StaticPointCloud(frame_points=static_frame_points)
        dynamic_pc = DynamicPointCloud(
            instance_id_mapping=waymoid2intid_global if waymoid2intid_global else {},
            points_by_instance=dynamic_points_by_instance,
            instances_info=instances_info,
        )
        
        return static_pc, dynamic_pc
    
    def _get_absolute_frame_idx(
        self,
        scene_dataset,
        frame_idx: int,
    ) -> int:
        """
        将相对帧索引转换为绝对帧号。
        
        Args:
            scene_dataset: 场景数据集实例
            frame_idx: 相对帧索引（MultiSceneDataset 使用的索引）
            
        Returns:
            absolute_frame_idx: 绝对帧号（用于 timesteps 和实例 JSON 查找）
        """
        # 从 scene_dataset 获取 start_timestep
        # 如果 scene_dataset 是 Mock 对象或没有 start_timestep 属性，默认为 0
        try:
            start_timestep = getattr(scene_dataset, 'start_timestep', 0)
            # 确保 start_timestep 是整数类型（避免 Mock 对象）
            if not isinstance(start_timestep, (int, np.integer)):
                start_timestep = 0
        except (AttributeError, TypeError):
            start_timestep = 0
        return int(start_timestep) + frame_idx
    
    def _load_instances_info(
        self,
        scene_data: Dict,
        scene_id: int,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]], Dict]:
        """
        从场景目录加载实例信息。
        
        Args:
            scene_data: 场景数据字典
            scene_id: 场景ID（用于缓存键）
            
        Returns:
            (waymoid2intid, id2framePoseSize, frame_instances) 元组
        """
        # 使用缓存（如果存在）
        if not hasattr(self, '_instances_cache'):
            self._instances_cache = {}
        
        cache_key = scene_id
        if cache_key in self._instances_cache:
            return self._instances_cache[cache_key]
        
        try:
            # 尝试从场景数据获取场景目录路径
            scene_dataset = scene_data['dataset']
            scene_dir = None
            
            # 尝试多种方式获取场景目录
            if hasattr(scene_dataset, 'data_cfg'):
                data_cfg = scene_dataset.data_cfg
                if hasattr(data_cfg, 'data_root') and hasattr(data_cfg, 'scene_idx'):
                    # 构建场景目录路径
                    scene_dir = os.path.join(data_cfg.data_root, f"{data_cfg.scene_idx:03d}")
            
            if scene_dir is None or not os.path.isdir(scene_dir):
                # 如果没有找到场景目录，返回空实例信息
                result = ({}, {}, {})
                self._instances_cache[cache_key] = result
                return result
            
            # 读取实例文件
            info_path = os.path.join(scene_dir, "instances", "instances_info.json")
            frame_path = os.path.join(scene_dir, "instances", "frame_instances.json")
            
            if not (os.path.exists(info_path) and os.path.exists(frame_path)):
                # 如果文件不存在，返回空实例信息
                result = ({}, {}, {})
                self._instances_cache[cache_key] = result
                return result
            
            with open(info_path, "r") as f:
                instances_info = json.load(f)  # keys: "0","1",...
            with open(frame_path, "r") as f:
                frame_instances = json.load(f)  # keys: "0","1",... -> [ids]
            
            # 将 instances_info 预处理为：每个 id -> {frame_idx: (T_ow, size)}
            id2framePoseSize = {}
            for sid_str, rec in instances_info.items():
                sid = int(sid_str)
                frames = rec["frame_annotations"]["frame_idx"]
                poses = rec["frame_annotations"]["obj_to_world"]
                sizes = rec["frame_annotations"]["box_size"]
                mapping = {}
                for fi, pose, sz in zip(frames, poses, sizes):
                    T_ow = np.array(pose, dtype=np.float32).reshape(4, 4)  # Object->World
                    sz = np.array(sz, dtype=np.float32).reshape(3,)  # [l,w,h]
                    mapping[int(fi)] = (T_ow, sz)
                id2framePoseSize[sid] = mapping
            
            # 构建稳定的 int id（1..M），保持与"旧代码的 waymoid2intid"风格一致
            all_ids = sorted([int(k) for k in instances_info.keys()])
            waymoid2intid = {sid: i+1 for i, sid in enumerate(all_ids)}  # 外部可用：原始（简化）id -> 连续 int
            
            result = (waymoid2intid, id2framePoseSize, frame_instances)
            self._instances_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Failed to load instances info for scene {scene_id}: {e}")
            result = ({}, {}, {})
            self._instances_cache[cache_key] = result
            return result
    
    def _get_instances_for_frame(
        self,
        waymoid2intid: Dict[int, int],
        id2framePoseSize: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        frame_instances: Dict,
        frame_idx: int,
        scene_dataset=None,
    ) -> Tuple[Dict[int, int], List[Tuple[int, np.ndarray, np.ndarray]]]:
        """
        获取指定帧的实例列表。
        
        Args:
            waymoid2intid: 实例ID映射
            id2framePoseSize: 实例ID到帧位姿和尺寸的映射（键是绝对帧号）
            frame_instances: 帧到实例ID列表的映射（键是绝对帧号的字符串）
            frame_idx: 相对帧索引（MultiSceneDataset 使用的索引）
            scene_dataset: 场景数据集实例（用于获取 start_timestep）
            
        Returns:
            (waymoid2intid, inst_list) - 实例ID映射和实例列表
        """
        out = []
        if not frame_instances:
            return waymoid2intid, out
        
        # 将相对帧索引转换为绝对帧号
        if scene_dataset is not None:
            absolute_frame_idx = self._get_absolute_frame_idx(scene_dataset, frame_idx)
        else:
            # 如果没有提供 scene_dataset，假设 frame_idx 已经是绝对帧号（向后兼容）
            absolute_frame_idx = frame_idx
        
        # 使用绝对帧号作为键
        key = str(absolute_frame_idx)
        if key not in frame_instances:
            return waymoid2intid, out
        
        for sid in frame_instances[key]:
            # sid 已是简化 id（int）
            sid = int(sid)
            # id2framePoseSize 的键也是绝对帧号
            if sid in id2framePoseSize and absolute_frame_idx in id2framePoseSize[sid]:
                T_ow, sz = id2framePoseSize[sid][absolute_frame_idx]
                intid = waymoid2intid[sid]
                out.append((intid, T_ow, sz))
        
        return waymoid2intid, out
    
    def _split_static_dynamic(
        self,
        pts_wrgb: np.ndarray,
        inst_list: List[Tuple[int, np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        将点云分割为静态背景和动态物体。
        
        Args:
            pts_wrgb: (N, 6) - 世界坐标点 + RGB
            inst_list: List[Tuple[int, np.ndarray, np.ndarray]] - 
                每项为 (intid, T_ow(4x4), size(3,))
            
        Returns:
            bg_points: (M, 6) - 静态背景点（世界坐标 + RGB）
            dynamic_points: Dict[int, np.ndarray] - 
                dynamic_points[intid] = (K, 6) 局部坐标 + RGB
        """
        # 1. 初始化掩码
        any_obj_mask = np.zeros((pts_wrgb.shape[0],), dtype=bool)
        dynamic_points = {}
        
        # 2. 遍历每个实例
        for (intid, T_ow, size_lwh) in inst_list:
            # World->Object
            T_wo = np.linalg.inv(T_ow)
            
            # 计算每个点在物体局部的坐标
            pw = pts_wrgb[:, :3]
            pw_h = np.concatenate([pw, np.ones((pw.shape[0], 1), dtype=np.float32)], axis=1)
            po = (T_wo @ pw_h.T).T[:, :3]  # (N, 3)
            
            # 检查点是否在边界框内
            half = size_lwh.astype(np.float32) / 2.0
            m = (np.abs(po) <= (half + 1e-6)).all(axis=1)  # in-box mask
            
            if not np.any(m):
                continue
            
            # 提取局部坐标 + RGB
            po_rgb = np.concatenate([po[m], pts_wrgb[m, 3:]], axis=1).astype(np.float32)
            any_obj_mask |= m
            
            # 保存到字典
            if intid not in dynamic_points:
                dynamic_points[intid] = []
            dynamic_points[intid].append(po_rgb)
        
        # 3. 合并同一实例的多块点云
        for intid in dynamic_points:
            if len(dynamic_points[intid]) > 0:
                dynamic_points[intid] = np.concatenate(dynamic_points[intid], axis=0)
        
        # 4. 提取静态背景点
        bg_points = pts_wrgb[~any_obj_mask]
        
        return bg_points, dynamic_points
    
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


class LiDARRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    基于 LiDAR 的 RGB 点云生成器。
    
    从 MultiSceneDataset 的段中加载 LiDAR 点云，通过多相机投影获取 RGB 颜色，
    并分割静态背景和动态物体。
    
    参考 tools/project_lidar.py 的实现。
    """
    
    def __init__(
        self,
        chosen_cam_ids: List[int] = None,
        camera_priority: Optional[List[int]] = None,
        resomult: float = 0.5,
        dataset: str = "waymo",
        crop_aabb: np.ndarray = None,
        input_aabb: np.ndarray = None,
        use_bbx: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            chosen_cam_ids: 选择使用的相机ID列表（默认根据数据集类型设置）
            camera_priority: 相机优先级（如果为None，使用数据集默认优先级）
            resomult: 图像分辨率缩放倍数（默认 0.5）
            dataset: 数据集类型（waymo/kitti/nuscenes/argoverse，默认 "waymo"）
            crop_aabb: 裁剪边界框，shape [2, 3]，格式 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            input_aabb: 输入边界框，shape [2, 3]，格式 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            use_bbx: 是否使用边界框裁剪
            device: 设备
        """
        super().__init__(
            sparsity='full',  # LiDAR 生成器不使用稀疏度
            filter_sky=False,  # LiDAR 生成器不使用天空过滤
            depth_consistency=False,  # LiDAR 生成器不使用深度一致性
            use_bbx=use_bbx,
            downscale=1,  # LiDAR 生成器不使用下采样
            crop_aabb=crop_aabb,
            input_aabb=input_aabb,
            device=device,
        )
        
        if chosen_cam_ids is None:
            if dataset.lower() == "waymo":
                self.chosen_cam_ids = [0, 1, 2, 3, 4]
            elif dataset.lower() == "nuscenes":
                self.chosen_cam_ids = [0, 1, 2, 3, 4, 5]
            elif dataset.lower() == "argoverse":
                self.chosen_cam_ids = [0, 5, 6, 1, 2, 3, 4]
            elif dataset.lower() == "kitti":
                self.chosen_cam_ids = [0, 1]
            else:
                self.chosen_cam_ids = [0]
        else:
            self.chosen_cam_ids = chosen_cam_ids
        
        self.resomult = resomult
        self.dataset = dataset.lower()
        
        # 设置相机优先级
        if camera_priority is not None:
            self.camera_priority = camera_priority
        else:
            if self.dataset == "nuscenes":
                self.camera_priority = [0, 1, 2, 3, 4, 5]
            elif self.dataset == "argoverse":
                self.camera_priority = [0, 5, 6, 1, 2, 3, 4]
            elif self.dataset == "waymo":
                self.camera_priority = [0, 1, 2, 3, 4]
            elif self.dataset == "kitti":
                self.camera_priority = [0, 1]
            else:
                self.camera_priority = self.chosen_cam_ids
        
        # 实例信息缓存
        self._instances_cache = {}
    
    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> o3d.geometry.PointCloud:
        """
        为指定场景和段生成 RGB 点云（基类接口实现）。
        
        此方法返回合并后的静态点云，符合基类接口要求。
        如需获取静动态分割结果，请使用 generate_pointcloud_with_static_dynamic()。
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            pointcloud: Open3D 点云对象，包含位置和颜色（仅静态背景点）
        """
        static_pc, _ = self.generate_pointcloud_with_static_dynamic(
            dataset, scene_id, segment_id
        )
        
        # 合并所有帧的静态点
        all_points = static_pc.get_merged_points()
        if all_points.shape[0] == 0:
            # 返回空点云
            pointcloud = o3d.geometry.PointCloud()
            return pointcloud
        
        points = all_points[:, :3]  # [N, 3]
        colors = all_points[:, 3:6]  # [N, 3]
        
        # 确保颜色在 [0, 1] 范围内
        colors = np.clip(colors, 0.0, 1.0)
        
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        return pointcloud
    
    def generate_pointcloud_with_static_dynamic(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> Tuple[StaticPointCloud, DynamicPointCloud]:
        """
        为指定场景和段生成 RGB 点云（包含静动态分割）。
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            static_pc: StaticPointCloud - 静态点云
            dynamic_pc: DynamicPointCloud - 动态点云
        """
        # 1. 获取段内所有帧索引
        frame_indices = dataset.get_segment_frames(scene_id, segment_id)
        if len(frame_indices) == 0:
            raise ValueError(f"Segment {segment_id} in scene {scene_id} has no frames")
        
        # 2. 获取场景数据
        scene_data = dataset.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")
        
        scene_dataset = scene_data['dataset']
        
        # 3. 预加载实例信息
        waymoid2intid_global, id2framePoseSize, frame_instances = self._load_instances_info(
            scene_data, scene_id
        )
        
        # 4. 初始化输出
        frame_points = []
        intid2inboxpoints = {}
        
        # 5. 遍历每帧
        for i, frame_idx in enumerate(frame_indices):
            # 5.1 加载 LiDAR 点云（世界坐标系）
            # 注意：SceneLidarSource 的点已经是世界坐标，不需要额外变换
            pts_w = self._load_lidar_points_world(scene_dataset, frame_idx)
            if pts_w is None or pts_w.shape[0] == 0:
                # 如果该帧没有 LiDAR 点，添加空的背景点列表
                frame_points.append(np.zeros((0, 6), dtype=np.float32))
                continue
            
            # 5.2 RGB 着色（点已经是世界坐标，不需要变换）
            pts_wrgb = self._colorize_points_world(
                dataset, scene_id, frame_idx, pts_w, scene_data
            )
            
            # 5.3 获取当前帧的实例列表（使用相对帧索引，方法内部会转换为绝对帧号）
            waymoid2intid, inst_list = self._get_instances_for_frame(
                waymoid2intid_global, id2framePoseSize, frame_instances, frame_idx, scene_dataset
            )
            
            # 5.4 分割静动态点
            bg_points, dynamic_points = self._split_static_dynamic(pts_wrgb, inst_list)
            
            # 5.5 保存静态背景点
            frame_points.append(bg_points.astype(np.float32))
            
            # 5.6 保存动态物体点（按实例ID和段内索引）
            # 注意：查找实例时使用全局帧号 frame_idx，但保存时使用段内索引 i
            # 这样与 project_lidar.py 的实现保持一致，便于后续渲染时使用 FrameSpec 选择帧
            for intid, po_rgb in dynamic_points.items():
                if intid not in intid2inboxpoints:
                    intid2inboxpoints[intid] = {}
                # 使用段内索引 i 作为键（与 project_lidar.py 一致）
                intid2inboxpoints[intid][i] = po_rgb.astype(np.float32)
        
        # 6. 构建 instances_info
        instances_info = {}
        num_frames = len(frame_indices)
        
        # 遍历所有实例ID
        for intid in intid2inboxpoints.keys():
            # 找到对应的原始ID (sid)
            sid = None
            for orig_sid, mapped_intid in waymoid2intid_global.items():
                if mapped_intid == intid:
                    sid = orig_sid
                    break
            
            if sid is None or sid not in id2framePoseSize:
                continue
            
            # 构建该实例的位姿数组和帧信息
            poses_list = []
            frame_info_list = []
            
            for i, frame_idx in enumerate(frame_indices):
                absolute_frame_idx = self._get_absolute_frame_idx(scene_dataset, frame_idx)
                
                if absolute_frame_idx in id2framePoseSize[sid]:
                    T_ow, size = id2framePoseSize[sid][absolute_frame_idx]
                    poses_list.append(T_ow)
                    frame_info_list.append(True)
                else:
                    # 该帧没有该实例，使用单位矩阵作为占位符
                    poses_list.append(np.eye(4, dtype=np.float32))
                    frame_info_list.append(False)
            
            instances_info[intid] = {
                "poses": np.stack(poses_list, axis=0),  # (num_frames, 4, 4)
                "size": id2framePoseSize[sid][list(id2framePoseSize[sid].keys())[0]][1],  # (3,) - 使用第一帧的尺寸
                "frame_info": np.array(frame_info_list, dtype=bool),  # (num_frames,)
            }
        
        # 7. 构建并返回新数据结构
        static_pc = StaticPointCloud(frame_points=frame_points)
        dynamic_pc = DynamicPointCloud(
            instance_id_mapping=waymoid2intid_global if waymoid2intid_global else {},
            points_by_instance=intid2inboxpoints,
            instances_info=instances_info,
        )
        
        return static_pc, dynamic_pc
    
    def _get_absolute_frame_idx(
        self,
        scene_dataset,
        frame_idx: int,
    ) -> int:
        """
        将相对帧索引转换为绝对帧号。
        
        Args:
            scene_dataset: 场景数据集实例
            frame_idx: 相对帧索引（MultiSceneDataset 使用的索引）
            
        Returns:
            absolute_frame_idx: 绝对帧号（用于 timesteps 和实例 JSON 查找）
        """
        # 从 scene_dataset 获取 start_timestep
        # 如果 scene_dataset 是 Mock 对象或没有 start_timestep 属性，默认为 0
        try:
            start_timestep = getattr(scene_dataset, 'start_timestep', 0)
            # 确保 start_timestep 是整数类型（避免 Mock 对象）
            if not isinstance(start_timestep, (int, np.integer)):
                start_timestep = 0
        except (AttributeError, TypeError):
            start_timestep = 0
        return int(start_timestep) + frame_idx
    
    def _load_lidar_points_world(
        self,
        scene_dataset,
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """
        从场景数据加载指定帧的 LiDAR 点云（世界坐标系）。
        
        注意：SceneLidarSource.origins/directions/ranges 已经是世界坐标
        （根据基类 get_aabb() 的注释："we assume the lidar points are already in the world coordinate system"）
        
        Args:
            scene_dataset: 场景数据集实例
            frame_idx: 相对帧索引（MultiSceneDataset 使用的索引）
            
        Returns:
            pts_w: (N, 3) - 世界坐标系下的点云，如果失败返回 None
        """
        try:
            lidar_source = scene_dataset.lidar_source
            if lidar_source is None:
                return None
            
            # 检查是否有 timesteps
            if not hasattr(lidar_source, 'timesteps') or lidar_source.timesteps is None:
                return None
            
            # 将相对帧索引转换为绝对帧号
            absolute_frame_idx = self._get_absolute_frame_idx(scene_dataset, frame_idx)
            
            # 筛选对应帧的点（使用绝对帧号）
            timesteps = lidar_source.timesteps
            if isinstance(timesteps, torch.Tensor):
                mask = (timesteps == absolute_frame_idx)
            else:
                mask = np.array(timesteps) == absolute_frame_idx
            
            if not mask.any():
                return None
            
            # 获取点坐标
            # SceneLidarSource.origins + directions * ranges 已经是世界坐标
            if hasattr(lidar_source, 'points') and lidar_source.points is not None:
                # 如果有 points 属性，直接使用（列 3:6 对应 points）
                points = lidar_source.points[mask]
                if isinstance(points, torch.Tensor):
                    points = points.cpu().numpy()
                # 如果 points 是多列的，取列 3:6（对应 x, y, z）
                if points.shape[1] >= 6:
                    pts_w = points[:, 3:6].astype(np.float32)
                else:
                    pts_w = points[:, :3].astype(np.float32)
            else:
                # 从 origins + directions * ranges 计算（已经是世界坐标）
                origins = lidar_source.origins[mask]
                directions = lidar_source.directions[mask]
                ranges = lidar_source.ranges[mask]
                
                if isinstance(origins, torch.Tensor):
                    origins = origins.cpu().numpy()
                    directions = directions.cpu().numpy()
                    ranges = ranges.cpu().numpy()
                
                # 处理 ranges 的形状
                if ranges.ndim == 1:
                    ranges = ranges.reshape(-1, 1)
                
                # 计算点坐标（已经是世界坐标）
                pts_w = (origins + directions * ranges).astype(np.float32)
            
            return pts_w
        except Exception as e:
            logger.warning(f"Failed to load LiDAR points for frame {frame_idx}: {e}")
            return None
    
    def _get_ego_pose(
        self,
        scene_dataset,
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """
        从场景数据获取指定帧的车辆位姿（Vehicle->World 的 4x4 变换矩阵）。
        
        注意：由于 SceneLidarSource 的点已经是世界坐标，此方法主要用于兼容性。
        实际上，点云不需要额外的位姿变换。
        
        Args:
            scene_dataset: 场景数据集实例
            frame_idx: 相对帧索引（MultiSceneDataset 使用的索引）
            
        Returns:
            T_vw: (4, 4) - 车辆到世界的变换矩阵，如果失败返回 None
        """
        try:
            # 从 lidar_source.lidar_to_worlds 获取
            if hasattr(scene_dataset, 'lidar_source') and scene_dataset.lidar_source is not None:
                lidar_source = scene_dataset.lidar_source
                
                # 检查是否有 lidar_to_worlds
                if hasattr(lidar_source, 'lidar_to_worlds') and lidar_source.lidar_to_worlds is not None:
                    lidar_to_worlds = lidar_source.lidar_to_worlds
                    
                    # 将相对帧索引转换为绝对帧号
                    absolute_frame_idx = self._get_absolute_frame_idx(scene_dataset, frame_idx)
                    
                    # lidar_to_worlds 的形状可能是 (num_timesteps, 4, 4) 或 (num_points, 4, 4)
                    # 需要根据 timesteps 找到对应帧的变换
                    if hasattr(lidar_source, 'timesteps') and lidar_source.timesteps is not None:
                        timesteps = lidar_source.timesteps
                        if isinstance(timesteps, torch.Tensor):
                            timesteps = timesteps.cpu().numpy()
                        
                        # 找到该帧的第一个点的索引（使用绝对帧号）
                        frame_mask = (timesteps == absolute_frame_idx)
                        if not np.any(frame_mask):
                            logger.warning(f"No lidar points found for frame {frame_idx} (absolute: {absolute_frame_idx})")
                            return None
                        
                        # 获取第一个匹配点的索引
                        first_idx = np.where(frame_mask)[0][0]
                        
                        # 安全地获取 lidar_to_worlds 的形状和访问
                        try:
                            # 尝试获取形状
                            if isinstance(lidar_to_worlds, torch.Tensor):
                                lidar_shape = lidar_to_worlds.shape
                            elif hasattr(lidar_to_worlds, 'shape'):
                                lidar_shape = lidar_to_worlds.shape
                            else:
                                # 尝试转换为 numpy 数组
                                lidar_array = np.array(lidar_to_worlds)
                                lidar_shape = lidar_array.shape
                                lidar_to_worlds = lidar_array
                        except (AttributeError, TypeError, ValueError):
                            # 如果无法获取形状，尝试直接索引
                            try:
                                if hasattr(lidar_to_worlds, '__getitem__'):
                                    T_lw = lidar_to_worlds[first_idx]
                                else:
                                    logger.warning(f"Cannot access lidar_to_worlds for frame {frame_idx}")
                                    return None
                            except (TypeError, IndexError, KeyError):
                                logger.warning(f"Cannot access lidar_to_worlds for frame {frame_idx}")
                                return None
                        else:
                            # 如果 lidar_to_worlds 是按点存储的，直接取第一个点
                            if len(lidar_shape) == 3 and lidar_shape[0] > first_idx:
                                T_lw = lidar_to_worlds[first_idx]
                            # 如果 lidar_to_worlds 是按时间步存储的，需要找到对应的时间步索引
                            elif len(lidar_shape) == 3 and absolute_frame_idx < lidar_shape[0]:
                                # 可能是按时间步索引存储（使用绝对帧号）
                                T_lw = lidar_to_worlds[absolute_frame_idx]
                            else:
                                # 尝试直接使用 first_idx 作为索引（按点存储）
                                try:
                                    if first_idx < lidar_shape[0]:
                                        T_lw = lidar_to_worlds[first_idx]
                                    else:
                                        logger.warning(f"Frame {frame_idx} (absolute: {absolute_frame_idx}) out of range for lidar_to_worlds")
                                        return None
                                except (TypeError, IndexError, KeyError):
                                    logger.warning(f"Cannot access lidar_to_worlds for frame {frame_idx}")
                                    return None
                        
                        # 转换为 numpy
                        if isinstance(T_lw, torch.Tensor):
                            T_lw = T_lw.cpu().numpy()
                        
                        # lidar_to_worlds 是 LiDAR->World，对于大多数数据集，LiDAR 坐标系 = 车辆坐标系
                        # 所以 T_lw 可以直接作为 T_vw
                        return T_lw.astype(np.float32)
            
            # 如果都没有，返回 None
            logger.warning(f"Failed to get ego pose for frame {frame_idx}: lidar_source or lidar_to_worlds not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to get ego pose for frame {frame_idx}: {e}")
            return None
    
    def _load_instances_info(
        self,
        scene_data: Dict,
        scene_id: int,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]], Dict]:
        """
        从场景目录加载实例信息。
        
        Args:
            scene_data: 场景数据字典
            scene_id: 场景ID（用于缓存键）
            
        Returns:
            (waymoid2intid, id2framePoseSize, frame_instances) 元组
        """
        # 使用缓存
        cache_key = scene_id
        if cache_key in self._instances_cache:
            return self._instances_cache[cache_key]
        
        try:
            # 尝试从场景数据获取场景目录路径
            scene_dataset = scene_data['dataset']
            scene_dir = None
            
            # 尝试多种方式获取场景目录
            if hasattr(scene_dataset, 'data_cfg'):
                data_cfg = scene_dataset.data_cfg
                if hasattr(data_cfg, 'data_root') and hasattr(data_cfg, 'scene_idx'):
                    # 构建场景目录路径
                    scene_dir = os.path.join(data_cfg.data_root, f"{data_cfg.scene_idx:03d}")
            
            if scene_dir is None or not os.path.isdir(scene_dir):
                # 如果没有找到场景目录，返回空实例信息
                result = ({}, {}, {})
                self._instances_cache[cache_key] = result
                return result
            
            # 读取实例文件
            info_path = os.path.join(scene_dir, "instances", "instances_info.json")
            frame_path = os.path.join(scene_dir, "instances", "frame_instances.json")
            
            if not (os.path.exists(info_path) and os.path.exists(frame_path)):
                # 如果文件不存在，返回空实例信息
                result = ({}, {}, {})
                self._instances_cache[cache_key] = result
                return result
            
            with open(info_path, "r") as f:
                instances_info = json.load(f)  # keys: "0","1",...
            with open(frame_path, "r") as f:
                frame_instances = json.load(f)  # keys: "0","1",... -> [ids]
            
            # 将 instances_info 预处理为：每个 id -> {frame_idx: (T_ow, size)}
            id2framePoseSize = {}
            for sid_str, rec in instances_info.items():
                sid = int(sid_str)
                frames = rec["frame_annotations"]["frame_idx"]
                poses = rec["frame_annotations"]["obj_to_world"]
                sizes = rec["frame_annotations"]["box_size"]
                mapping = {}
                for fi, pose, sz in zip(frames, poses, sizes):
                    T_ow = np.array(pose, dtype=np.float32).reshape(4, 4)  # Object->World
                    sz = np.array(sz, dtype=np.float32).reshape(3,)  # [l,w,h]
                    mapping[int(fi)] = (T_ow, sz)
                id2framePoseSize[sid] = mapping
            
            # 构建稳定的 int id（1..M），保持与"旧代码的 waymoid2intid"风格一致
            all_ids = sorted([int(k) for k in instances_info.keys()])
            waymoid2intid = {sid: i+1 for i, sid in enumerate(all_ids)}  # 外部可用：原始（简化）id -> 连续 int
            
            result = (waymoid2intid, id2framePoseSize, frame_instances)
            self._instances_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Failed to load instances info for scene {scene_id}: {e}")
            result = ({}, {}, {})
            self._instances_cache[cache_key] = result
            return result
    
    def _get_instances_for_frame(
        self,
        waymoid2intid: Dict[int, int],
        id2framePoseSize: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        frame_instances: Dict,
        frame_idx: int,
        scene_dataset=None,
    ) -> Tuple[Dict[int, int], List[Tuple[int, np.ndarray, np.ndarray]]]:
        """
        获取指定帧的实例列表。
        
        Args:
            waymoid2intid: 实例ID映射
            id2framePoseSize: 实例ID到帧位姿和尺寸的映射（键是绝对帧号）
            frame_instances: 帧到实例ID列表的映射（键是绝对帧号的字符串）
            frame_idx: 相对帧索引（MultiSceneDataset 使用的索引）
            scene_dataset: 场景数据集实例（用于获取 start_timestep）
            
        Returns:
            (waymoid2intid, inst_list) - 实例ID映射和实例列表
        """
        out = []
        if not frame_instances:
            return waymoid2intid, out
        
        # 将相对帧索引转换为绝对帧号
        if scene_dataset is not None:
            absolute_frame_idx = self._get_absolute_frame_idx(scene_dataset, frame_idx)
        else:
            # 如果没有提供 scene_dataset，假设 frame_idx 已经是绝对帧号（向后兼容）
            absolute_frame_idx = frame_idx
        
        # 使用绝对帧号作为键
        key = str(absolute_frame_idx)
        if key not in frame_instances:
            return waymoid2intid, out
        
        for sid in frame_instances[key]:
            # sid 已是简化 id（int）
            sid = int(sid)
            # id2framePoseSize 的键也是绝对帧号
            if sid in id2framePoseSize and absolute_frame_idx in id2framePoseSize[sid]:
                T_ow, sz = id2framePoseSize[sid][absolute_frame_idx]
                intid = waymoid2intid[sid]
                out.append((intid, T_ow, sz))
        
        return waymoid2intid, out
    
    def _get_opencv2dataset_matrix(self) -> np.ndarray:
        """
        返回 OpenCV(右-下-前) -> 数据集相机坐标 的 4x4 齐次变换矩阵。
        
        Returns:
            (4, 4) 变换矩阵
        """
        if self.dataset == "waymo":
            return np.array(
                [[0, 0, 1, 0],
                 [-1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 0, 1]], dtype=np.float32)
        elif self.dataset in ("kitti", "nuscenes", "argoverse"):
            return np.eye(4, dtype=np.float32)
        else:
            # 兜底：维持之前的行为（Waymo）
            return np.eye(4, dtype=np.float32)
    
    def _project_points_to_image(
        self,
        points_w: np.ndarray,
        T_cw: np.ndarray,
        K: np.ndarray,
        img_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将世界坐标点投影到图像平面。
        
        Args:
            points_w: (N, 3) - 世界坐标点
            T_cw: (4, 4) - 世界到相机的变换矩阵
            K: (3, 3) - 相机内参
            img_size: (W, H) - 图像尺寸
            
        Returns:
            (uv, dists, indices) - 像素坐标（整数）、距离、原始索引
        """
        W, H = img_size
        pts_h = np.concatenate([points_w, np.ones((points_w.shape[0], 1), np.float32)], 1)
        pc = (T_cw @ pts_h.T).T[:, :3]  # World->Camera(OpenCV)
        z = pc[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return (np.zeros((0, 2), np.int32), np.zeros((0, 1), np.float32), np.zeros((0,), np.int64))
        pc = pc[valid]
        uv = (K @ pc.T).T
        uv = uv[:, :2] / pc[:, 2:3]
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[in_img], v[in_img]
        d = np.linalg.norm(pc[in_img], axis=1, keepdims=True).astype(np.float32)
        return np.stack([u, v], 1), d, np.where(valid)[0][in_img]
    
    def _colorize_points_world(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        frame_idx: int,
        pts_w: np.ndarray,
        scene_data: Dict,
    ) -> np.ndarray:
        """
        用该帧多相机图给世界坐标系点上色。
        
        注意：输入点云已经是世界坐标（SceneLidarSource.origins/directions/ranges 已经是世界坐标），
        不需要额外的坐标变换。
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            frame_idx: 相对帧索引（MultiSceneDataset 使用的索引）
            pts_w: (N, 3) - 世界坐标系下的点云
            scene_data: 场景数据字典
            
        Returns:
            pts_wrgb: (N, 6) - 世界坐标点云 + RGB
        """
        # 1. 初始化 RGB（全零，使用 float32 保持精度）
        rgb = np.zeros((pts_w.shape[0], 3), dtype=np.float32)
        
        # 3. 按优先级遍历相机，投影并着色
        for cam_id in self.camera_priority:
            if cam_id not in self.chosen_cam_ids:
                continue
            
            try:
                # 获取图像和相机参数
                frame_data = dataset.get_frame_data(scene_id, frame_idx, cam_id)
            except Exception as e:
                logger.debug(f"Failed to load frame data for cam {cam_id}, frame {frame_idx}: {e}")
                continue
            
            # 转换为numpy数组
            img = frame_data['image'].cpu().numpy()  # [H, W, 3]
            extrinsic = frame_data['extrinsic'].cpu().numpy()  # [4, 4] - cam_to_world
            
            # 转换内参为3x3
            intrinsic = frame_data['intrinsic'].cpu().numpy()  # [3, 3] or [4, 4]
            if intrinsic.shape == (4, 4):
                intrinsic = intrinsic[:3, :3]
            
            H0, W0 = img.shape[:2]
            W = int(round(W0 * self.resomult))
            H = int(round(H0 * self.resomult))
            if W <= 0 or H <= 0:
                continue
            
            # 调整内参（根据 resomult）
            K_scaled = intrinsic.copy()
            K_scaled[0, 0] *= self.resomult
            K_scaled[1, 1] *= self.resomult
            K_scaled[0, 2] *= self.resomult
            K_scaled[1, 2] *= self.resomult
            
            # 计算世界到相机的变换
            # MultiSceneDataset.get_frame_data() 返回的 extrinsic 已经是 OpenCV 相机系的 cam_to_world
            # 所以直接取逆即可，不需要额外的坐标系转换
            T_cw = np.linalg.inv(extrinsic)  # World->Camera(OpenCV)
            
            # 投影到图像平面（pts_w 已经是世界坐标）
            uv, d, indices = self._project_points_to_image(pts_w, T_cw, K_scaled, (W, H))
            if uv.shape[0] == 0:
                continue
            
            # 缩放图像并采样颜色
            import cv2
            img_small = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            bgr = img_small[uv[:, 1], uv[:, 0]]  # float32 [0, 1]
            # 直接赋值，保持 float32 类型和 [0, 1] 范围
            rgb[indices] = bgr[:, ::-1]  # BGR->RGB 覆盖
        
        # 2. 组合结果（rgb 已经是 float32）
        pts_wrgb = np.concatenate([pts_w, rgb], axis=1)
        return pts_wrgb
    
    def _split_static_dynamic(
        self,
        pts_wrgb: np.ndarray,
        inst_list: List[Tuple[int, np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        将点云分割为静态背景和动态物体。
        
        Args:
            pts_wrgb: (N, 6) - 世界坐标点 + RGB
            inst_list: List[Tuple[int, np.ndarray, np.ndarray]] - 
                每项为 (intid, T_ow(4x4), size(3,))
            
        Returns:
            bg_points: (M, 6) - 静态背景点（世界坐标 + RGB）
            dynamic_points: Dict[int, np.ndarray] - 
                dynamic_points[intid] = (K, 6) 局部坐标 + RGB
        """
        # 1. 初始化掩码
        any_obj_mask = np.zeros((pts_wrgb.shape[0],), dtype=bool)
        dynamic_points = {}
        
        # 2. 遍历每个实例
        for (intid, T_ow, size_lwh) in inst_list:
            # World->Object
            T_wo = np.linalg.inv(T_ow)
            
            # 计算每个点在物体局部的坐标
            pw = pts_wrgb[:, :3]
            pw_h = np.concatenate([pw, np.ones((pw.shape[0], 1), dtype=np.float32)], axis=1)
            po = (T_wo @ pw_h.T).T[:, :3]  # (N, 3)
            
            # 检查点是否在边界框内
            half = size_lwh.astype(np.float32) / 2.0
            m = (np.abs(po) <= (half + 1e-6)).all(axis=1)  # in-box mask
            
            if not np.any(m):
                continue
            
            # 提取局部坐标 + RGB
            po_rgb = np.concatenate([po[m], pts_wrgb[m, 3:]], axis=1).astype(np.float32)
            any_obj_mask |= m
            
            # 保存到字典
            if intid not in dynamic_points:
                dynamic_points[intid] = []
            dynamic_points[intid].append(po_rgb)
        
        # 3. 合并同一实例的多块点云
        for intid in dynamic_points:
            if len(dynamic_points[intid]) > 0:
                dynamic_points[intid] = np.concatenate(dynamic_points[intid], axis=0)
        
        # 4. 提取静态背景点
        bg_points = pts_wrgb[~any_obj_mask]
        
        return bg_points, dynamic_points

