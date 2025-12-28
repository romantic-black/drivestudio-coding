# RGB 点云生成器设计文档（支持静动态分割）

## 概述

本文档设计基于 LiDAR 的 RGB 点云生成器系统，参考 `tools/project_lidar.py` 的实现，支持从 LiDAR 点云生成 RGB 点云，并能够分割静态背景和动态物体。该系统从 `MultiSceneDataset` 的段中获取数据，生成可用于训练的 RGB 点云。

**核心特性**：

1. 从 LiDAR 点云开始，通过多相机投影获取 RGB 颜色
2. 使用实例信息分割静态背景和动态物体
3. 静态点保存为世界坐标，动态点保存为物体局部坐标
4. 支持多帧融合和动态物体优先渲染

---

## 核心概念

### 1. 数据来源

**从 MultiSceneDataset 获取数据**：

- LiDAR 点云：通过 `MultiSceneDataset` 获取车辆坐标系下的点云
- 相机图像：通过 `MultiSceneDataset` 获取 RGB 图像
- 相机参数：通过 `cam_infos` 获取内参、外参
- 实例信息：通过 `instances_info` 和 `frame_instances` 获取动态物体信息
- 车辆位姿：通过 `ego_pose` 获取车辆到世界的变换

### 2. 点云生成流程

参考 `tools/project_lidar.py`，点云生成流程包括：

1. **LiDAR 点云加载**：从段内所有帧加载 LiDAR 点云（车辆坐标系）
2. **RGB 着色**：将点云投影到多个相机图像上，获取 RGB 颜色
3. **坐标变换**：将车辆坐标系的点变换到世界坐标系
4. **静动态分割**：
   - 使用实例边界框判断点是否属于动态物体
   - 静态点：保存为世界坐标 + RGB
   - 动态点：转换为物体局部坐标 + RGB
5. **多帧融合**：合并多帧的点云（可选）
6. **后处理**：
   - 边界框裁剪（可选）
   - 统计滤波
   - 均匀下采样

### 3. 静动态分割原理

**静态背景点**：

- 不属于任何实例边界框的点
- 保存为世界坐标 `(x_w, y_w, z_w, r, g, b)`
- 可以跨帧累积，形成静态场景点云

**动态物体点**：

- 属于某个实例边界框的点
- 转换为物体局部坐标：`p_local = (T_ow^-1 * p_world)[:3]`
- 保存为局部坐标 `(x_o, y_o, z_o, r, g, b)`
- 按实例 ID 和帧索引组织：`intid2inboxpoints[intid][frame_idx]`

**优势**：

- 动态物体点使用局部坐标，可以独立变换和渲染
- 支持动态物体优先渲染（先渲染动态，再渲染静态）
- 支持多帧融合（静态点可以跨帧累积，动态点可以按帧选择）

---

## 类设计

### RGBPointCloudGenerator（基类）

```python
class RGBPointCloudGenerator(ABC):
    """
    RGB 点云生成器基类。
  
    核心功能：
    1. 定义点云生成的抽象接口
    2. 提供通用的辅助方法（边界框、裁剪、滤波等）
    3. 支持多种点云生成策略（单目深度、LiDAR 等）
    """
  
    def __init__(
        self,
        use_bbx: bool = True,
        bbx_min: Optional[np.ndarray] = None,  # [3] - 自定义边界框最小值
        bbx_max: Optional[np.ndarray] = None,   # [3] - 自定义边界框最大值
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            use_bbx: 是否使用边界框裁剪
            bbx_min: 自定义边界框最小值（如果为None，使用默认值）
            bbx_max: 自定义边界框最大值（如果为None，使用默认值）
            device: 设备（用于深度图处理）
        """
        pass
  
    @abstractmethod
    def generate_pointcloud(
        self,
        dataset: MultiSceneDataset,
        scene_id: int,
        segment_id: int,
    ) -> Tuple[
        List[np.ndarray],  # frame_points: 静态背景点列表
        Dict[int, int],    # waymoid2intid: 实例ID映射
        Dict[int, Dict[int, np.ndarray]],  # intid2inboxpoints: 动态物体点
    ]:
        """
        为指定场景和段生成 RGB 点云（包含静动态分割）。
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
          
        Returns:
            frame_points: List[np.ndarray] - 每项为 (N, 6) 世界坐标背景点 + RGB
            waymoid2intid: Dict[int, int] - 原始实例ID -> 连续int ID（从1开始）
            intid2inboxpoints: Dict[int, Dict[int, np.ndarray]] - 
                intid2inboxpoints[intid][frame_idx] = (N, 6) 局部坐标 + RGB
        """
        pass
  
    def get_bbx(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取边界框范围。
      
        Returns:
            bbx_min: [3] - 边界框最小值
            bbx_max: [3] - 边界框最大值
        """
        pass
  
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
        pass
  
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
        pass
```

### LiDARRGBPointCloudGenerator（子类）

```python
class LiDARRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    基于 LiDAR 的 RGB 点云生成器。
  
    从 MultiSceneDataset 的段中加载 LiDAR 点云，通过多相机投影获取 RGB 颜色，
    并分割静态背景和动态物体。
  
    参考 tools/project_lidar.py 的实现。
    """
  
    def __init__(
        self,
        chosen_cam_ids: List[int] = [0, 1, 2, 3, 4],  # 选择使用的相机ID列表
        camera_priority: Optional[List[int]] = None,  # 相机优先级（用于RGB着色）
        resomult: float = 0.5,  # 图像分辨率缩放倍数
        use_bbx: bool = True,
        bbx_min: Optional[np.ndarray] = None,
        bbx_max: Optional[np.ndarray] = None,
        device: torch.device = torch.device("cpu"),
        dataset: str = "waymo",  # 数据集类型（waymo/kitti/nuscenes/argoverse）
    ):
        """
        Args:
            chosen_cam_ids: 选择使用的相机ID列表（用于RGB着色）
            camera_priority: 相机优先级（如果为None，使用数据集默认优先级）
            resomult: 图像分辨率缩放倍数（用于加速投影计算）
            use_bbx: 是否使用边界框裁剪
            bbx_min: 自定义边界框最小值
            bbx_max: 自定义边界框最大值
            device: 设备
            dataset: 数据集类型（影响坐标变换和相机优先级）
        """
        super().__init__(
            use_bbx=use_bbx,
            bbx_min=bbx_min,
            bbx_max=bbx_max,
            device=device,
        )
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
                self.camera_priority = chosen_cam_ids
  
    def generate_pointcloud(
        self,
        dataset: MultiSceneDataset,
        scene_id: int,
        segment_id: int,
    ) -> Tuple[
        List[np.ndarray],
        Dict[int, int],
        Dict[int, Dict[int, np.ndarray]],
    ]:
        """
        为指定场景和段生成 RGB 点云（包含静动态分割）。
      
        流程：
        1. 获取段内所有帧索引
        2. 对每帧：
           a. 加载 LiDAR 点云（车辆坐标系）
           b. 通过多相机投影获取 RGB 颜色
           c. 变换到世界坐标系
           d. 使用实例信息分割静动态点
           e. 静态点保存为世界坐标，动态点保存为局部坐标
        3. 返回 frame_points, waymoid2intid, intid2inboxpoints
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
          
        Returns:
            frame_points: List[np.ndarray] - 每项为 (N, 6) 世界坐标背景点 + RGB
            waymoid2intid: Dict[int, int] - 原始实例ID -> 连续int ID（从1开始）
            intid2inboxpoints: Dict[int, Dict[int, np.ndarray]] - 
                intid2inboxpoints[intid][frame_idx] = (N, 6) 局部坐标 + RGB
        """
        pass
  
    def _get_segment_frames(
        self,
        dataset: MultiSceneDataset,
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
        pass
  
    def _load_lidar_points_vehicle(
        self,
        dataset: MultiSceneDataset,
        scene_id: int,
        frame_idx: int,
    ) -> np.ndarray:
        """
        加载指定帧的 LiDAR 点云（车辆坐标系）。
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            frame_idx: 帧索引
          
        Returns:
            pts_v: (N, 3) - 车辆坐标系下的点云
        """
        pass
  
    def _colorize_points_vehicle(
        self,
        dataset: MultiSceneDataset,
        scene_id: int,
        frame_idx: int,
        pts_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        用该帧多相机图给车辆坐标系点上色；并输出其世界坐标副本。
      
        参考 project_lidar.py 的 colorize_points_vehicle()。
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            frame_idx: 帧索引
            pts_v: (N, 3) - 车辆坐标系下的点云
          
        Returns:
            pts_vrgb: (N, 6) - 车辆坐标 + RGB(float)
            pts_wrgb: (N, 6) - 世界坐标 + RGB(float)
        """
        pass
  
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
            uv: (M, 2) - 投影后的像素坐标（整数）
            dists: (M, 1) - 点到相机的距离
            indices: (M,) - 有效点的原始索引
        """
        pass
  
    def _load_instances_for_frame(
        self,
        dataset: MultiSceneDataset,
        scene_id: int,
        frame_idx: int,
    ) -> Tuple[Dict[int, int], List[Tuple[int, np.ndarray, np.ndarray]]]:
        """
        加载指定帧的实例信息。
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            frame_idx: 帧索引
          
        Returns:
            waymoid2intid: Dict[int, int] - 原始实例ID -> 连续int ID（从1开始）
            inst_list: List[Tuple[int, np.ndarray, np.ndarray]] - 
                每项为 (intid, T_ow(4x4), size(3,))
        """
        pass
  
    def _split_static_dynamic(
        self,
        pts_wrgb: np.ndarray,
        inst_list: List[Tuple[int, np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        将点云分割为静态背景和动态物体。
      
        参考 project_lidar.py 的 build_frame_points_and_objects() 中的分割逻辑。
      
        Args:
            pts_wrgb: (N, 6) - 世界坐标点 + RGB
            inst_list: List[Tuple[int, np.ndarray, np.ndarray]] - 
                每项为 (intid, T_ow(4x4), size(3,))
          
        Returns:
            bg_points: (M, 6) - 静态背景点（世界坐标 + RGB）
            dynamic_points: Dict[int, np.ndarray] - 
                dynamic_points[intid] = (K, 6) 局部坐标 + RGB
        """
        pass
```

---

## MultiSceneDataset 扩展

### 新增方法

```python
class MultiSceneDataset:
    # ... 现有方法 ...
  
    def get_lidar_points(
        self,
        scene_id: int,
        frame_idx: int,
    ) -> np.ndarray:
        """
        获取指定帧的 LiDAR 点云（车辆坐标系）。
      
        Args:
            scene_id: 场景ID
            frame_idx: 帧索引
          
        Returns:
            points: (N, 3) - 车辆坐标系下的点云
        """
        pass
  
    def get_ego_pose(
        self,
        scene_id: int,
        frame_idx: int,
    ) -> np.ndarray:
        """
        获取指定帧的车辆位姿（Vehicle->World 的 4x4 变换矩阵）。
      
        Args:
            scene_id: 场景ID
            frame_idx: 帧索引
          
        Returns:
            T_vw: (4, 4) - 车辆到世界的变换矩阵
        """
        pass
  
    def get_instances_info(
        self,
        scene_id: int,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]]:
        """
        获取场景的实例信息。
      
        Args:
            scene_id: 场景ID
          
        Returns:
            waymoid2intid: Dict[int, int] - 原始实例ID -> 连续int ID（从1开始）
            id2framePoseSize: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] - 
                id2framePoseSize[sid][frame_idx] = (T_ow(4x4), size(3,))
        """
        pass
  
    def get_frame_instances(
        self,
        scene_id: int,
        frame_idx: int,
    ) -> List[int]:
        """
        获取指定帧的实例ID列表。
      
        Args:
            scene_id: 场景ID
            frame_idx: 帧索引
          
        Returns:
            instance_ids: List[int] - 该帧出现的实例ID列表
        """
        pass
```

---

## 实现细节

### 1. RGB 着色流程

**参考 `project_lidar.py` 的 `colorize_points_vehicle()`**：

```python
def _colorize_points_vehicle(
    self,
    dataset: MultiSceneDataset,
    scene_id: int,
    frame_idx: int,
    pts_v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用该帧多相机图给车辆坐标系点上色；并输出其世界坐标副本。
    """
    # 1. 获取车辆位姿（Vehicle->World）
    T_vw = dataset.get_ego_pose(scene_id, frame_idx)
  
    # 2. 变换到世界坐标系
    pts_w = (T_vw[:3, :3] @ pts_v.T + T_vw[:3, 3:4]).T
  
    # 3. 初始化 RGB（全零）
    rgb = np.zeros((pts_v.shape[0], 3), dtype=np.uint8)
  
    # 4. 按优先级遍历相机，投影并着色
    for cam_id in self.camera_priority:
        if cam_id not in self.chosen_cam_ids:
            continue
      
        # 获取图像和内参
        frame_data = dataset.get_frame_data(scene_id, frame_idx, cam_id)
        if frame_data is None:
            continue
      
        img = frame_data['image'].cpu().numpy()  # [H, W, 3]
        K = frame_data['intrinsic'][:3, :3].cpu().numpy()  # [3, 3]
        extrinsic = frame_data['extrinsic'].cpu().numpy()  # [4, 4] - cam_to_world
      
        # 调整内参（根据 resomult）
        H0, W0 = img.shape[:2]
        W = int(round(W0 * self.resomult))
        H = int(round(H0 * self.resomult))
        K_scaled = K.copy()
        K_scaled[0, 0] *= self.resomult
        K_scaled[1, 1] *= self.resomult
        K_scaled[0, 2] *= self.resomult
        K_scaled[1, 2] *= self.resomult
      
        # 计算世界到相机的变换
        T_cw = np.linalg.inv(extrinsic)  # World->Camera
      
        # 投影到图像平面
        uv, dists, indices = self._project_points_to_image(
            pts_w, T_cw, K_scaled, (W, H)
        )
      
        if uv.shape[0] == 0:
            continue
      
        # 缩放图像并采样颜色
        img_small = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        bgr = img_small[uv[:, 1], uv[:, 0]]
        rgb[indices] = bgr[:, ::-1]  # BGR->RGB 覆盖
  
    # 5. 组合结果
    pts_vrgb = np.concatenate([pts_v, rgb.astype(np.float32)], axis=1)
    pts_wrgb = np.concatenate([pts_w, rgb.astype(np.float32)], axis=1)
    return pts_vrgb, pts_wrgb
```

### 2. 静动态分割流程

**参考 `project_lidar.py` 的 `build_frame_points_and_objects()`**：

```python
def _split_static_dynamic(
    self,
    pts_wrgb: np.ndarray,
    inst_list: List[Tuple[int, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    将点云分割为静态背景和动态物体。
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
```

### 3. 点云生成主循环

```python
def generate_pointcloud(
    self,
    dataset: MultiSceneDataset,
    scene_id: int,
    segment_id: int,
) -> Tuple[
    List[np.ndarray],
    Dict[int, int],
    Dict[int, Dict[int, np.ndarray]],
]:
    """
    为指定场景和段生成 RGB 点云（包含静动态分割）。
    """
    # 1. 获取段内所有帧索引
    frame_indices = self._get_segment_frames(dataset, scene_id, segment_id)
  
    # 2. 预加载实例信息
    waymoid2intid_global, id2framePoseSize = dataset.get_instances_info(scene_id)
  
    # 3. 初始化输出
    frame_points = []
    intid2inboxpoints = {}
  
    # 4. 遍历每帧
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Generating pointcloud")):
        # 4.1 加载 LiDAR 点云（车辆坐标系）
        pts_v = self._load_lidar_points_vehicle(dataset, scene_id, frame_idx)
      
        # 4.2 RGB 着色并变换到世界坐标
        pts_vrgb, pts_wrgb = self._colorize_points_vehicle(
            dataset, scene_id, frame_idx, pts_v
        )
      
        # 4.3 获取当前帧的实例列表
        waymoid2intid, inst_list = self._load_instances_for_frame(
            dataset, scene_id, frame_idx
        )
      
        # 4.4 分割静动态点
        bg_points, dynamic_points = self._split_static_dynamic(pts_wrgb, inst_list)
      
        # 4.5 保存静态背景点
        frame_points.append(bg_points.astype(np.float32))
      
        # 4.6 保存动态物体点（按实例ID和帧索引）
        for intid, po_rgb in dynamic_points.items():
            if intid not in intid2inboxpoints:
                intid2inboxpoints[intid] = {}
            intid2inboxpoints[intid][i] = po_rgb.astype(np.float32)
  
    # 5. 返回结果
    waymoid2intid_out = waymoid2intid_global if waymoid2intid_global else {}
    return frame_points, waymoid2intid_out, intid2inboxpoints
```

---

## 使用示例

### 1. 基本使用

```python
# 1. 创建数据集
dataset = MultiSceneDataset(
    data_cfg=data_cfg,
    train_scene_ids=[0, 1, 2],
    eval_scene_ids=[3, 4],
    # ... 其他参数
)

# 2. 创建点云生成器
pointcloud_generator = LiDARRGBPointCloudGenerator(
    chosen_cam_ids=[0, 1, 2, 3, 4],  # 使用所有相机
    resomult=0.5,  # 图像分辨率缩放
    use_bbx=True,
    dataset="waymo",
)

# 3. 为指定场景和段生成点云
scene_id = 0
segment_id = 0
frame_points, waymoid2intid, intid2inboxpoints = pointcloud_generator.generate_pointcloud(
    dataset=dataset,
    scene_id=scene_id,
    segment_id=segment_id,
)

# 4. 使用结果
# frame_points: List[np.ndarray] - 每项为 (N, 6) 静态背景点（世界坐标 + RGB）
# waymoid2intid: Dict[int, int] - 实例ID映射
# intid2inboxpoints: Dict[int, Dict[int, np.ndarray]] - 动态物体点（局部坐标 + RGB）

# 5. 保存静态点云（合并所有帧）
static_points = np.concatenate(frame_points, axis=0)
static_pcd = o3d.geometry.PointCloud()
static_pcd.points = o3d.utility.Vector3dVector(static_points[:, :3])
static_pcd.colors = o3d.utility.Vector3dVector(static_points[:, 3:6] / 255.0)
o3d.io.write_point_cloud(f"scene_{scene_id}_segment_{segment_id}_static.ply", static_pcd)
```

### 2. 渲染动态物体（参考 project_lidar.py）

```python
def render_dynamic_objects(
    frame_points: List[np.ndarray],
    intid2inboxpoints: Dict[int, Dict[int, np.ndarray]],
    waymoid2intid: Dict[int, int],
    frame_idx: int,
    T_cw_virtual: np.ndarray,
    K: np.ndarray,
    img_size: Tuple[int, int],
    bg_spec: FrameSpec = None,
    dyn_spec: FrameSpec = None,
) -> np.ndarray:
    """
    渲染动态物体（先动态，后背景）。
  
    参考 project_lidar.py 的 render_with_dynamics()。
    """
    W, H = img_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.ones((H, W, 3), dtype=np.uint8)
  
    n_total = len(frame_points)
    bg_spec = bg_spec or FrameSpec("pm", K=4, S=1)
    dyn_spec = dyn_spec or FrameSpec("pm", K=1, S=1)
  
    # 1. 合并背景帧
    bg_indices = bg_spec.select(n_total, frame_idx)
    merged_bg = np.concatenate([frame_points[i] for i in bg_indices], axis=0)
  
    # 2. 收集动态物体点（按 dyn_spec 指定的帧集合）
    dyn_indices = dyn_spec.select(n_total, frame_idx)
    objs_world_rgb = []
    objs_depth = []
  
    for intid, frame_dict in intid2inboxpoints.items():
        # 汇总该实例在 dyn_indices 里出现的局部点
        buf = []
        for fi in dyn_indices:
            if fi in frame_dict:
                buf.append(frame_dict[fi])
        if not buf:
            continue
      
        pinbox = np.concatenate(buf, axis=0)  # (K, 6) [x_local, y_local, z_local, r, g, b]
      
        # 局部点 -> 世界（需要获取当前帧的 T_ow）
        # 这里简化处理，实际需要从 dataset 获取 T_ow
        # T_ow = get_instance_pose(intid, frame_idx)
        # R_o = T_ow[:3, :3]
        # t_o = T_ow[:3, 3]
        # pw = (R_o @ pinbox[:, :3].T).T + t_o[None, :]
        # rgb = pinbox[:, 3:].astype(np.float32)
        # objs_world_rgb.append(np.concatenate([pw, rgb], axis=1))
      
        # 排序依据：局部 xy 的最小半径
        min_r = np.linalg.norm(pinbox[:, :2], axis=1).min() if pinbox.shape[0] > 0 else 1e9
        objs_depth.append(min_r)
  
    # 3. 先画动态（按深度排序）
    # ... 投影和渲染逻辑 ...
  
    # 4. 再画背景
    if merged_bg.shape[0] > 0:
        uv, d, idx = project_points_to_image(merged_bg[:, :3], T_cw_virtual, K, (W, H))
        if uv.shape[0] > 0:
            uv_f, _, col = filter_duplicates(uv, d, merged_bg[idx, 3:])
            temp = np.zeros_like(img)
            temp[uv_f[:, 1], uv_f[:, 0]] = col.astype(np.uint8)
            img += temp * mask
  
    return img
```

---

## 与 project_lidar.py 的对应关系

| project_lidar.py                     | LiDARRGBPointCloudGenerator      | 说明                 |
| ------------------------------------ | -------------------------------- | -------------------- |
| `load_lidar_points_vehicle()`      | `_load_lidar_points_vehicle()` | 加载 LiDAR 点云      |
| `colorize_points_vehicle()`        | `_colorize_points_vehicle()`   | RGB 着色             |
| `build_frame_points_and_objects()` | `generate_pointcloud()`        | 生成点云并分割静动态 |
| `get_instances_for_frame()`        | `_load_instances_for_frame()`  | 获取实例信息         |
| `render_with_dynamics()`           | （外部函数）                     | 渲染动态物体         |

---

## 关键设计决策

### 1. 坐标系统

- **静态点**：使用世界坐标，便于跨帧累积和场景重建
- **动态点**：使用物体局部坐标，便于独立变换和渲染

### 2. 数据组织

- **frame_points**：按帧组织的静态点列表，每帧独立
- **intid2inboxpoints**：按实例ID和帧索引组织的动态点字典

### 3. RGB 着色策略

- 按相机优先级遍历，后覆盖前（优先级高的相机颜色覆盖优先级低的）
- 支持多相机融合，提高点云着色覆盖率

### 4. 边界框检查

- 使用轴对齐边界框（AABB）判断点是否属于实例
- 局部坐标下检查：`|p_local| <= size/2`

---

## 反直觉检查清单

### 1. 数据获取检查

- [ ] **LiDAR 点云格式正确**：点云在车辆坐标系下，形状为 (N, 3)
- [ ] **车辆位姿正确**：T_vw 是 Vehicle->World 的变换
- [ ] **相机外参正确**：extrinsic 是 cam_to_world，需要取逆得到 world_to_cam
- [ ] **实例信息完整**：每个实例都有 T_ow 和 size

### 2. RGB 着色检查

- [ ] **投影正确**：点云正确投影到图像平面
- [ ] **颜色采样正确**：从缩放后的图像采样颜色
- [ ] **相机优先级生效**：优先级高的相机颜色覆盖优先级低的
- [ ] **分辨率缩放正确**：内参和图像都按 resomult 缩放

### 3. 静动态分割检查

- [ ] **坐标变换正确**：World->Object 变换正确
- [ ] **边界框检查正确**：点在局部坐标下正确判断是否在框内
- [ ] **掩码正确**：静态点和动态点不重叠
- [ ] **局部坐标正确**：动态点的局部坐标相对于物体中心

### 4. 数据组织检查

- [ ] **frame_points 格式正确**：每项为 (N, 6) 世界坐标 + RGB
- [ ] **intid2inboxpoints 格式正确**：intid2inboxpoints[intid][frame_idx] = (N, 6) 局部坐标 + RGB
- [ ] **waymoid2intid 映射正确**：原始ID正确映射到连续int ID（从1开始）

---

## 总结

基于 LiDAR 的 RGB 点云生成器设计遵循以下原则：

1. **数据复用**：从 `MultiSceneDataset` 获取数据，避免重复加载
2. **静动态分割**：使用实例信息分割静态背景和动态物体
3. **坐标系统**：静态点用世界坐标，动态点用局部坐标
4. **多相机融合**：通过多相机投影提高 RGB 着色覆盖率
5. **灵活渲染**：支持动态物体优先渲染和多帧融合

该设计允许在不修改 `MultiSceneDataset` 核心功能的情况下，实现从 LiDAR 点云生成 RGB 点云并分割静动态物体的功能，同时保持与 `project_lidar.py` 的兼容性。
