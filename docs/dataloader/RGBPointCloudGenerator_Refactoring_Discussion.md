# RGB 点云生成器重构讨论文档

## 概述

本文档讨论 RGB 点云生成器系统的重构方案，重点关注：

1. **静动态点云分离**：从基类开始重构，要求生成的点云分为静态和动态两部分，需要构思动态点云的表示形式

---

## 问题 1：静动态点云分离重构

### 1.1 现状分析

#### 当前实现

**LiDARRGBPointCloudGenerator** 已经实现了静动态分割：

- 静态点：保存为世界坐标 `(x_w, y_w, z_w, r, g, b)`
- 动态点：保存为物体局部坐标 `(x_o, y_o, z_o, r, g, b)`，按实例ID和帧索引组织

**MonocularRGBPointCloudGenerator** 目前只生成静态点云：

- 从单目深度图生成点云
- 没有区分静态和动态部分

**driving_dataset.py 的 get_init_objects 方法** 展示了如何从 LiDAR 点云中提取动态物体：

- 遍历所有帧和实例
- 将 LiDAR 点转换到物体坐标系
- 使用边界框过滤点
- 累积多帧的点云
- 返回物体局部坐标的点云

#### 问题

1. **基类接口不统一**：`RGBPointCloudGenerator.generate_pointcloud()` 只返回单个点云，不支持静动态分离
2. **MonocularRGBPointCloudGenerator 缺少动态物体支持**：无法从单目深度图生成动态物体点云
3. **动态点云表示形式不统一**：不同生成器可能使用不同的表示形式

### 1.2 重构方案

#### 方案 A：基类接口扩展（推荐）

**核心思想**：在基类中定义统一的静动态点云接口，所有子类都实现此接口。

**接口设计**：

```python
class RGBPointCloudGenerator(ABC):
    """
    RGB 点云生成器基类。
  
    核心功能：
    1. 定义点云生成的抽象接口（支持静动态分离）
    2. 提供通用的辅助方法（边界框、裁剪、滤波等）
    3. 支持多种点云生成策略（单目、LiDAR、融合等）
    """
  
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
    ) -> Tuple[
        StaticPointCloud,      # 静态点云
        DynamicPointCloud,     # 动态点云
    ]:
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
```

**动态点云表示形式设计**：

```python
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
```

**实现示例**：

```python
class MonocularRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    单目 RGB 点云生成器（支持静动态分割）。
    """
  
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
        """
        # 1. 生成基础点云（世界坐标）
        frame_points_list = []
        for frame_idx in self._get_segment_frames(dataset, scene_id, segment_id):
            # 从单目深度图生成点云
            points_world = self._generate_pointcloud_from_frame(
                dataset, scene_id, frame_idx
            )  # (N, 6) 世界坐标 + RGB
      
            frame_points_list.append(points_world)
  
        # 2. 获取实例信息
        scene_data = dataset.get_scene(scene_id)
        waymoid2intid, id2framePoseSize, frame_instances = self._load_instances_info(
            scene_data, scene_id
        )
  
        # 3. 分割静动态点
        static_frame_points = []
        dynamic_points_by_instance = {}
        instances_info = {}
  
        frame_indices = self._get_segment_frames(dataset, scene_id, segment_id)
        for i, frame_idx in enumerate(frame_indices):
            points_world = frame_points_list[i]  # (N, 6)
      
            # 获取当前帧的实例列表
            _, inst_list = self._get_instances_for_frame(
                waymoid2intid, id2framePoseSize, frame_instances, frame_idx, scene_data['dataset']
            )
      
            # 分割静动态点
            bg_points, dynamic_points = self._split_static_dynamic(
                points_world, inst_list
            )
      
            static_frame_points.append(bg_points)
      
            # 保存动态点
            for intid, po_rgb in dynamic_points.items():
                if intid not in dynamic_points_by_instance:
                    dynamic_points_by_instance[intid] = {}
                dynamic_points_by_instance[intid][i] = po_rgb
  
        # 4. 构建实例信息
        for intid in dynamic_points_by_instance:
            # 从 id2framePoseSize 获取位姿和尺寸
            # 这里需要根据实际的实例ID映射来获取
            instances_info[intid] = {
                "poses": ...,  # 从 id2framePoseSize 获取
                "size": ...,   # 从 id2framePoseSize 获取
                "frame_info": ...,  # 从 frame_instances 获取
            }
  
        # 5. 返回结果
        static_pc = StaticPointCloud(frame_points=static_frame_points)
        dynamic_pc = DynamicPointCloud(
            instance_id_mapping=waymoid2intid,
            points_by_instance=dynamic_points_by_instance,
            instances_info=instances_info,
        )
  
        return static_pc, dynamic_pc
```
