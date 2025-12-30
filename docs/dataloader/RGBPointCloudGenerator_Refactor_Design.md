# RGB 点云生成器重构设计文档

## 概述

本文档设计 RGB 点云生成器系统的重构方案，主要目标是：

1. 支持静态背景和动态物体的分离输出
2. 基于 3D bbox 分割点云
3. 新增 LiDAR 点云生成器
4. 重构代码结构，将不同生成器分离到独立文件
5. 所有数据从 MultiSceneDataset 获取，不直接读取文件

---

## 设计目标

### 1. 功能目标

- **静态背景点云**：不属于任何动态物体的点，使用世界坐标
- **动态物体点云**：属于动态物体的点，使用局部坐标（相对于物体坐标系）
- **实例映射**：维护原始实例 ID 到连续整数 ID 的映射
- **数据源统一**：所有数据通过 MultiSceneDataset 获取，不直接读取文件系统

### 2. 架构目标

- **模块化设计**：基类和子类分离到独立文件
- **接口统一**：所有生成器继承同一基类，使用统一的接口
- **可扩展性**：易于添加新的点云生成策略（如立体视觉、多传感器融合等）

---

## 核心概念

### 1. 静态与动态点云

参考 `tools/project_lidar.py` 的 `build_frame_points_and_objects` 函数：

- **frame_points**：背景点云列表，每个元素是 `[N, 6]` 的 numpy 数组（前3列是世界坐标，后3列是RGB颜色）
- **intid2inboxpoints**：动态物体点云字典，格式为 `{intid: {frame_idx: [N, 6]}}`，其中点云使用局部坐标（相对于物体坐标系）
- **waymoid2intid**：原始实例 ID 到连续整数 ID 的映射字典

### 2. 坐标系统

- **世界坐标**：用于静态背景点云，所有点共享同一世界坐标系
- **局部坐标**：用于动态物体点云，每个物体有独立的坐标系
  - 定义：`p_local = (T_ow^-1 @ p_world)[:3]`，其中 `T_ow` 是物体到世界的变换矩阵
  - 局部坐标系原点在物体的中心，轴方向与物体方向对齐

### 3. 3D Bbox 判断

使用轴对齐边界框（AABB）判断点是否属于动态物体：

- 输入：点的世界坐标 `p_world`、物体的变换矩阵 `T_ow`（Object->World）、物体尺寸 `size_lwh`（长度、宽度、高度）
- 步骤：
  1. 将点变换到物体局部坐标系：`p_local = (T_ow^-1 @ p_world)[:3]`
  2. 计算局部坐标的半边长：`half = size_lwh / 2.0`
  3. 判断点是否在框内：`mask = (abs(p_local) <= (half + epsilon)).all(axis=1)`

---

## 类设计

### 1. RGBPointCloudGenerator（基类）

**文件位置**：`datasets/pointcloud_generators/base.py`

```python
class RGBPointCloudGenerator(ABC):
    """
    RGB 点云生成器基类。
  
    核心功能：
    1. 定义点云生成的抽象接口
    2. 提供通用的辅助方法（边界框、裁剪、滤波等）
    3. 支持多种点云生成策略（单目、LiDAR、立体等）
    """
  
    def __init__(
        self,
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
            sparsity: 稀疏度级别
            filter_sky: 是否过滤天空区域
            depth_consistency: 是否进行深度一致性检查
            use_bbx: 是否使用边界框裁剪
            downscale: 点云生成时的下采样倍数
            crop_aabb: 裁剪边界框，shape [2, 3]
            input_aabb: 输入边界框，shape [2, 3]
            device: 设备
        """
        pass
  
    @abstractmethod
    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> Dict:
        """
        为指定场景和段生成 RGB 点云（包含静态背景和动态物体）。
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
          
        Returns:
            Dict包含：
                - 'background': np.ndarray [N, 6] - 静态背景点云（世界坐标 + RGB）
                  前3列为世界坐标，后3列为RGB颜色（float32，范围[0,255]）
                - 'dynamic_objects': Dict[int, np.ndarray] - 动态物体点云字典
                  {intid: [M, 6]}，每个点云使用局部坐标 + RGB
                  前3列为局部坐标，后3列为RGB颜色（float32，范围[0,255]）
                - 'instance_mapping': Dict[int, int] - 原始实例ID到连续整数ID的映射
                  {original_id: intid}
                - 'metadata': Dict - 其他元数据（可选）
        """
        pass
  
    def get_crop_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取裁剪边界框范围。"""
        pass
  
    def get_input_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取输入边界框范围。"""
        pass
  
    def crop_pointcloud(self, ...):
        """裁剪点云到边界框。"""
        pass
  
    def split_pointcloud(self, ...):
        """将点云分割为边界框内部和外部两部分。"""
        pass
  
    def filter_pointcloud(self, ...):
        """对点云进行滤波。"""
        pass
  
    def _separate_static_dynamic(
        self,
        points_world: np.ndarray,  # [N, 3] - 世界坐标点云
        colors: np.ndarray,  # [N, 3] - RGB颜色
        instances: List[Dict],  # 实例列表，每个元素包含 {'intid', 'T_ow', 'size_lwh'}
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        将点云分割为静态背景和动态物体。
      
        Args:
            points_world: [N, 3] - 世界坐标点云
            colors: [N, 3] - RGB颜色（float32，范围[0,255]）
            instances: 实例列表，每个元素包含：
                - 'intid': int - 连续整数ID
                - 'T_ow': np.ndarray [4, 4] - 物体到世界的变换矩阵
                - 'size_lwh': np.ndarray [3] - 物体尺寸（长度、宽度、高度）
      
        Returns:
            background: [M1, 6] - 静态背景点云（世界坐标 + RGB）
                前3列为世界坐标，后3列为RGB颜色
            dynamic_objects: Dict[int, np.ndarray] - 动态物体点云字典 {intid: [M2, 6]}
                每个点云使用局部坐标 + RGB，前3列为局部坐标，后3列为RGB颜色
        """
        pass
```

### 2. MonocularRGBPointCloudGenerator（单目深度生成器）

**文件位置**：`datasets/pointcloud_generators/monocular.py`

```python
class MonocularRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    单目 RGB 点云生成器。
  
    从 MultiSceneDataset 的段中生成单目深度点云。
    支持从段内所有帧（或按稀疏度过滤后的帧）生成点云。
    支持静态背景和动态物体的分离。
    """
  
    def __init__(
        self,
        chosen_cam_ids: List[int] = [0],
        sparsity: Literal['Drop90', 'Drop80', 'Drop50', 'Drop25', 'full'] = 'full',
        filter_sky: bool = True,
        depth_consistency: bool = True,
        use_bbx: bool = True,
        downscale: int = 2,
        crop_aabb: np.ndarray = None,
        input_aabb: np.ndarray = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            chosen_cam_ids: 选择使用的相机ID列表
            sparsity: 稀疏度级别
            filter_sky: 是否过滤天空区域
            depth_consistency: 是否进行深度一致性检查
            use_bbx: 是否使用边界框裁剪
            downscale: 点云生成时的下采样倍数
            crop_aabb: 裁剪边界框
            input_aabb: 输入边界框
            device: 设备
        """
        super().__init__(...)
        self.chosen_cam_ids = chosen_cam_ids
  
    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> Dict:
        """
        为指定场景和段生成 RGB 点云（包含静态背景和动态物体）。
      
        流程：
        1. 获取段内所有帧索引
        2. 根据稀疏度过滤帧
        3. 加载所有选中帧的 RGB 图像、深度图、外参、内参、天空掩码
        4. 对每个相机分别应用深度一致性检查（如果启用）
        5. **逐帧生成点云并分离**：
           a. 对每一帧：
              - 生成该帧的点云（反投影、变换），合并所有相机的数据
              - 获取该帧的实例信息（T_ow, size_lwh）
              - 使用该帧的实例 bbox 分离该帧的静态背景和动态物体点云
           b. 将所有帧的静态背景点云合并
           c. 将所有帧的动态物体点云按 intid 合并
        6. 应用边界框裁剪（如果启用）
        7. 滤波和下采样
        
        **重要**：必须逐帧分离，不能将所有帧点云合并后再用首帧（或任何单帧）的bbox分离，
        因为动态物体会移动，多帧累积后用单帧bbox会导致大量动态点被错误分类。
      
        Returns:
            Dict包含：
                - 'background': np.ndarray [N, 6] - 静态背景点云（世界坐标 + RGB）
                  前3列为世界坐标，后3列为RGB颜色（float32，范围[0,255]）
                - 'dynamic_objects': Dict[int, np.ndarray] - 动态物体点云字典
                  {intid: [M, 6]}，每个点云使用局部坐标 + RGB
                  前3列为局部坐标，后3列为RGB颜色（float32，范围[0,255]）
                - 'instance_mapping': Dict[int, int] - 实例ID映射
                - 'metadata': Dict - 元数据
        """
        pass
  
    def _load_frame_data(self, ...):
        """加载指定帧和相机的数据。"""
        pass
  
    def _depth_consistency_check(self, ...):
        """检查连续帧之间的深度一致性。"""
        pass
  
    def _generate_pointcloud_from_frames_by_camera(self, ...):
        """从按相机分组的帧数据生成点云。"""
        pass
  
    def _get_instances_for_segment(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
        frame_indices: List[int],
    ) -> Tuple[Dict[int, int], List[Dict]]:
        """
        从 MultiSceneDataset 获取段内所有帧的动态物体实例信息。
      
        实现方式：
        - 通过 dataset.get_scene(scene_id) 获取场景数据
        - 通过 scene_data['dataset'] 获取 DrivingDataset 实例
        - 根据数据集类型，从文件系统读取或通过数据集 API 获取实例信息
        - 参考"从 MultiSceneDataset 获取数据"章节的详细说明
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID
            frame_indices: 段内帧索引列表
      
        Returns:
            instance_mapping: Dict[int, int] - 原始实例ID到连续整数ID的映射
            instances_by_frame: List[List[Dict]] - 每个帧的实例列表
                每个元素是列表，包含该帧的所有实例，每个实例包含：
                - 'original_id': int - 原始实例ID
                - 'intid': int - 连续整数ID（数组索引，从0开始）
                - 'T_ow': np.ndarray [4, 4] - 物体到世界的变换矩阵
                - 'size_lwh': np.ndarray [3] - 物体尺寸（长度、宽度、高度）
        """
        pass
```

### 3. LiDARRGBPointCloudGenerator（LiDAR 生成器）

**文件位置**：`datasets/pointcloud_generators/lidar.py`

```python
class LiDARRGBPointCloudGenerator(RGBPointCloudGenerator):
    """
    LiDAR RGB 点云生成器。
  
    从 MultiSceneDataset 的段中生成 LiDAR 点云。
    参考 tools/project_lidar.py 的 build_frame_points_and_objects 函数。
    支持静态背景和动态物体的分离。
    """
  
    def __init__(
        self,
        sparsity: Literal['Drop90', 'Drop80', 'Drop50', 'Drop25', 'full'] = 'full',
        use_bbx: bool = True,
        crop_aabb: np.ndarray = None,
        input_aabb: np.ndarray = None,
        resomult: float = 0.5,  # 图像分辨率倍数（用于给LiDAR点上色）
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            sparsity: 稀疏度级别（用于过滤帧）
            use_bbx: 是否使用边界框裁剪
            crop_aabb: 裁剪边界框
            input_aabb: 输入边界框
            resomult: 图像分辨率倍数（用于从图像给LiDAR点上色）
            device: 设备
        """
        super().__init__(
            sparsity=sparsity,
            filter_sky=False,  # LiDAR 不需要天空过滤
            depth_consistency=False,  # LiDAR 不需要深度一致性检查
            use_bbx=use_bbx,
            downscale=1,  # LiDAR 不需要下采样
            crop_aabb=crop_aabb,
            input_aabb=input_aabb,
            device=device,
        )
        self.resomult = resomult
  
    def generate_pointcloud(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        segment_id: int,
    ) -> Dict:
        """
        为指定场景和段生成 LiDAR RGB 点云（包含静态背景和动态物体）。
      
        流程：
        1. 获取段内所有帧索引
        2. 根据稀疏度过滤帧
        3. **逐帧处理点云并分离**：
           a. 对每一帧：
              - 加载该帧的 LiDAR 点云（世界坐标和车辆坐标）
                * lidar_source 中的点已经是世界坐标（通过 lidar_to_worlds 变换）
                * 如果需要车辆坐标，从世界坐标转换：points_vehicle = (T_wv @ points_world_homo)[:3]
              - 加载该帧的 RGB 图像、外参、内参
              - 给 LiDAR 点上色（从图像投影）
                * 使用车辆坐标点云进行上色流程
                * 将车辆坐标变换到世界坐标用于投影
              - 获取该帧的实例信息（T_ow, size_lwh）
              - 使用该帧的实例 bbox 分离该帧的静态背景和动态物体点云（使用世界坐标点云）
              - 将动态物体点云转换到局部坐标
           b. 将所有帧的静态背景点云合并
           c. 将所有帧的动态物体点云按 intid 合并
        4. 应用边界框裁剪（如果启用）
        5. 滤波和下采样
        
        **重要**：必须逐帧分离，不能将所有帧点云合并后再用首帧（或任何单帧）的bbox分离，
        因为动态物体会移动，多帧累积后用单帧bbox会导致大量动态点被错误分类。
      
        Returns:
            Dict包含：
                - 'background': np.ndarray [N, 6] - 静态背景点云（世界坐标 + RGB）
                  前3列为世界坐标，后3列为RGB颜色（float32，范围[0,255]）
                - 'dynamic_objects': Dict[int, np.ndarray] - 动态物体点云字典
                  {intid: [M, 6]}，每个点云使用局部坐标 + RGB
                  前3列为局部坐标，后3列为RGB颜色（float32，范围[0,255]）
                - 'instance_mapping': Dict[int, int] - 实例ID映射
                - 'metadata': Dict - 元数据
        """
        pass
  
    def _load_lidar_points(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 MultiSceneDataset 加载指定帧的 LiDAR 点云（世界坐标和车辆坐标）。
      
        实现方式：
        - 通过 dataset.get_scene(scene_id) 获取场景数据
        - 通过 scene_data['dataset'].lidar_source 获取 LiDAR 数据
        - 使用 timesteps 筛选对应 frame_idx 的点
        - 计算 3D 坐标：points_world = origins + directions * ranges（已经是世界坐标）
        - 如果需要车辆坐标，使用 ego pose 转换：points_vehicle = (T_vw^-1 @ points_world)[:3]
        - 参考"从 MultiSceneDataset 获取数据"章节的详细说明
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            frame_idx: 帧索引
      
        Returns:
            points_world: [N, 3] - LiDAR 点云（世界坐标）
            points_vehicle: [N, 3] - LiDAR 点云（车辆坐标）
        """
        pass
  
    def _colorize_lidar_points(
        self,
        dataset: "MultiSceneDataset",
        scene_id: int,
        frame_idx: int,
        points_vehicle: np.ndarray,  # [N, 3] - 车辆坐标点云
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        用该帧多相机图给车辆坐标系点上色，并输出其世界坐标副本。
      
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            frame_idx: 帧索引
            points_vehicle: [N, 3] - 车辆坐标点云
      
        Returns:
            points_vehicle_rgb: [N, 6] - 车辆坐标 + RGB
            points_world_rgb: [N, 6] - 世界坐标 + RGB
        """
        pass
  
    def _get_instances_for_segment(self, ...):
        """
        从 MultiSceneDataset 获取段内所有帧的动态物体实例信息。
        
        实现方式：参考 MonocularRGBPointCloudGenerator 的同名方法。
        参考"从 MultiSceneDataset 获取数据"章节的详细说明。
        """
        pass
```

---

## 从 MultiSceneDataset 获取数据

**重要**：理论上不需要扩展 MultiSceneDataset 的接口，所有数据都可以通过现有接口获取。

### 1. 获取场景和段信息

```python
# 获取场景数据（包含 DrivingDataset 实例）
scene_data = dataset.get_scene(scene_id)
scene_dataset = scene_data['dataset']  # DrivingDataset 实例

# 获取段信息
segment = scene_data['segments'][segment_id]
frame_indices = segment['frame_indices']  # 段内所有帧索引
```

### 2. 获取帧数据（图像、深度、相机参数）

通过 `scene_dataset.pixel_source.get_image()` 获取：

**重要**：生成 `img_idx` 时必须使用相机的 `unique_cam_idx`（在 camera_list 中的枚举序号），
而不是原始的 `cam_id`。`pixel_source.parse_img_idx` 期待的是 `unique_cam_idx`（0..num_cams-1）。
如果 `camera_list` 不是从0连续递增（例如 [0, 2, 4] 或使用字符串ID），直接使用 `cam_id` 会导致错误的 frame_idx/相机解析。

```python
def _get_frame_data_from_dataset(
    scene_dataset: DrivingDataset,
    frame_idx: int,
    cam_id: int,  # 原始相机ID（来自 chosen_cam_ids）
) -> Dict:
    """
    从 DrivingDataset 获取指定帧和相机的数据。
    
    Args:
        scene_dataset: DrivingDataset 实例
        frame_idx: 帧索引
        cam_id: 原始相机ID（例如 chosen_cam_ids 中的值，可能是 0, 2, 4 等）
    
    Returns:
        Dict包含图像、深度、相机参数等信息
    """
    pixel_source = scene_dataset.pixel_source
    
    # 重要：需要找到 cam_id 在 camera_list 中的位置（unique_cam_idx）
    # camera_list 可能不是从0连续递增（例如 [0, 2, 4]），需要找到索引位置
    try:
        cam_list_idx = pixel_source.camera_list.index(cam_id)
    except ValueError:
        raise ValueError(f"Camera ID {cam_id} not found in camera_list: {pixel_source.camera_list}")
    
    # unique_cam_idx 是相机在 camera_list 中的索引位置（0, 1, 2, ...）
    unique_cam_idx = pixel_source.camera_data[cam_id].unique_cam_idx
    
    # 计算图像索引（使用 unique_cam_idx，不是 cam_list_idx）
    # pixel_source.parse_img_idx 期待的是 unique_cam_idx（0..num_cams-1）
    img_idx = frame_idx * pixel_source.num_cams + unique_cam_idx
    
    # 获取图像和相机信息
    image_infos, cam_infos = pixel_source.get_image(img_idx)
    
    # 获取深度图（从 camera_data）
    depth = None
    try:
        camera_data = pixel_source.camera_data[cam_id]
        if hasattr(camera_data, 'depth_maps') and camera_data.depth_maps is not None:
            depth = camera_data.depth_maps[frame_idx]
        elif hasattr(camera_data, 'lidar_depth_maps') and camera_data.lidar_depth_maps is not None:
            depth = camera_data.lidar_depth_maps[frame_idx]
    except (IndexError, KeyError, AttributeError):
        pass
    
    # 获取天空掩码（如果可用）
    sky_mask = image_infos.get('sky_mask')
    
    return {
        'image': image_infos['pixels'],  # Tensor [H, W, 3]
        'extrinsic': cam_infos['camera_to_world'],  # Tensor [4, 4]
        'intrinsic': cam_infos['intrinsics'],  # Tensor [3, 3] or [4, 4]
        'depth': depth,  # Tensor [H, W] or None
        'sky_mask': sky_mask,  # Tensor [H, W] or None
    }
```

### 3. 获取 LiDAR 点云

通过 `scene_dataset.lidar_source` 获取：

```python
def _get_lidar_points_from_dataset(
    scene_dataset: DrivingDataset,
    frame_idx: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 DrivingDataset 获取指定帧的 LiDAR 点云（世界坐标和车辆坐标）。
    
    注意：lidar_source 中的点已经是世界坐标系（通过 lidar_to_worlds 变换）。
    """
    if scene_dataset.lidar_source is None:
        return None, None
    
    lidar_source = scene_dataset.lidar_source
    
    # 检查 LiDAR 数据是否已加载
    if not (hasattr(lidar_source, 'origins') and 
            hasattr(lidar_source, 'directions') and 
            hasattr(lidar_source, 'ranges') and 
            hasattr(lidar_source, 'timesteps')):
        return None, None
    
    # 筛选对应 frame_idx 的点
    frame_indices_tensor = torch.tensor(
        [frame_idx], 
        dtype=lidar_source.timesteps.dtype, 
        device=lidar_source.timesteps.device
    )
    mask = torch.isin(lidar_source.timesteps, frame_indices_tensor)
    
    if not mask.any():
        return None, None
    
    # 获取该帧的 LiDAR 点
    origins = lidar_source.origins[mask]
    directions = lidar_source.directions[mask]
    ranges = lidar_source.ranges[mask]
    
    # 计算 3D 坐标：points_world = origins + directions * ranges
    # 注意：这些点已经是世界坐标系（因为 origins 已经通过 lidar_to_worlds 变换）
    points_world = origins + directions * ranges  # Tensor [N, 3] - 世界坐标
    points_world_np = points_world.cpu().numpy().astype(np.float32)
    
    # 如果需要车辆坐标，从世界坐标转换
    points_vehicle_np = None
    if lidar_source.lidar_to_worlds is not None and frame_idx < lidar_source.lidar_to_worlds.shape[0]:
        T_vw = lidar_source.lidar_to_worlds[frame_idx].cpu().numpy()  # Vehicle->World
        T_wv = np.linalg.inv(T_vw)  # World->Vehicle
        
        # 转换为齐次坐标
        points_world_homo = np.concatenate([
            points_world_np,
            np.ones((points_world_np.shape[0], 1), dtype=np.float32)
        ], axis=1)  # [N, 4]
        
        # 转换到车辆坐标系
        points_vehicle = (T_wv @ points_world_homo.T).T[:, :3]  # [N, 3]
        points_vehicle_np = points_vehicle.astype(np.float32)
    
    return points_world_np, points_vehicle_np
```

**坐标系说明**：
- `lidar_source` 中的点**已经是世界坐标系**（通过 `lidar_to_worlds` 变换）
- `origins + directions * ranges` 计算出的点在世界坐标系下
- 如果需要车辆坐标系的点（如用于给点上色），需要从世界坐标转换：
  - `T_vw = lidar_source.lidar_to_worlds[frame_idx]`（Vehicle->World）
  - `T_wv = np.linalg.inv(T_vw)`（World->Vehicle）
  - `points_vehicle = (T_wv @ points_world_homo)[:3]`

### 4. 获取 Ego Vehicle Pose

**重要**：经过代码检查，`DrivingDataset.lidar_source` **直接提供 ego vehicle pose**！

**直接获取方式**：

```python
def _get_ego_pose_from_lidar_source(
    scene_dataset: DrivingDataset,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """
    从 lidar_source 获取 ego vehicle pose（Vehicle->World 变换矩阵）。
    
    注意：
    - 对于 nuScenes，lidar 坐标系就是 ego vehicle 坐标系，所以 lidar_to_world 就是 ego pose
    - 对于 Waymo/KITTI 等，lidar_to_world 也是从 ego pose 计算的，等价于 ego pose
    - lidar_to_worlds 可能相对于第一帧做了对齐（相对于第一帧的 ego pose）
    
    Args:
        scene_dataset: DrivingDataset 实例
        frame_idx: 帧索引
    
    Returns:
        T_vw: np.ndarray [4, 4] - Vehicle->World 变换矩阵，或 None 如果不可用
    """
    if scene_dataset.lidar_source is None:
        return None
    
    lidar_source = scene_dataset.lidar_source
    
    # 检查 lidar_to_worlds 是否已加载
    if lidar_source.lidar_to_worlds is None:
        return None
    
    # 检查 frame_idx 是否有效
    if frame_idx < 0 or frame_idx >= lidar_source.lidar_to_worlds.shape[0]:
        return None
    
    # 获取该帧的 ego pose
    T_vw = lidar_source.lidar_to_worlds[frame_idx].cpu().numpy()  # [4, 4] - Vehicle->World
    
    return T_vw.astype(np.float32)
```

**关键属性说明**：

- `lidar_source.lidar_to_worlds`: Tensor [num_frames, 4, 4]
  - LiDAR 到世界的变换矩阵（LiDAR->World）
  - **对于 nuScenes**：LiDAR 坐标系就是 ego vehicle 坐标系，所以 `lidar_to_world` 就是 ego pose
  - **对于 Waymo/KITTI 等**：`lidar_to_world` 是从 `ego_pose` 文件计算的，等价于 ego pose
  - 可能相对于第一帧做了对齐（相对于第一帧的 ego pose 进行变换）

**注意**：
- 这是**推荐的方式**，因为直接从接口获取，不需要访问文件系统
- 某些场景可能没有 LiDAR 数据，`lidar_source` 可能为 None，需要检查
- `lidar_to_worlds` 的形状是 `[num_frames, 4, 4]`，需要确保 `frame_idx` 在有效范围内

### 5. 获取动态物体实例信息

**重要**：经过代码检查，`DrivingDataset.pixel_source` **直接提供动态物体实例信息**！

**直接获取方式**：

```python
def _get_dynamic_instances_from_pixel_source(
    scene_dataset: DrivingDataset,
    frame_indices: List[int],
) -> Tuple[Dict[int, int], List[List[Dict]]]:
    """
    从 pixel_source 直接获取动态物体实例信息。
    
    Args:
        scene_dataset: DrivingDataset 实例
        frame_indices: 帧索引列表
    
    Returns:
        instance_mapping: Dict[int, int] - 原始实例ID到连续整数ID的映射
            {original_id: intid}，intid 从 0 开始（注意：不是从1开始，因为这是数组索引）
        instances_by_frame: List[List[Dict]] - 每个帧的实例列表
            每个元素是该帧的所有实例列表，每个实例包含：
            - 'intid': int - 连续整数ID（数组索引，从0开始）
            - 'original_id': int - 原始实例ID（从 instances_true_id 获取）
            - 'T_ow': Tensor [4, 4] - 物体到世界的变换矩阵（Object->World）
            - 'size_lwh': Tensor [3] - 物体尺寸（长度、宽度、高度）
    """
    pixel_source = scene_dataset.pixel_source
    
    # 检查是否有实例信息
    if pixel_source.instances_pose is None:
        return {}, [[] for _ in frame_indices]
    
    # 获取实例信息
    # instances_pose: Tensor [frame_num, instance_num, 4, 4] - Object->World
    # instances_size: Tensor [instance_num, 3] - 每个实例的尺寸（长度、宽度、高度）
    # per_frame_instance_mask: Tensor [frame_num, instance_num] - 每帧每个实例是否活跃（bool）
    # instances_true_id: Tensor [instance_num] - 原始实例ID
    
    instances_pose = pixel_source.instances_pose  # [frame_num, instance_num, 4, 4]
    instances_size = pixel_source.instances_size  # [instance_num, 3]
    per_frame_instance_mask = pixel_source.per_frame_instance_mask  # [frame_num, instance_num]
    instances_true_id = pixel_source.instances_true_id  # [instance_num]
    
    num_instances = instances_pose.shape[1]
    
    # 构建 instance_mapping: 原始ID -> 连续整数ID（数组索引，从0开始）
    # 注意：这里使用数组索引作为 intid，如果需要从1开始，需要 +1
    instance_mapping = {
        int(instances_true_id[i].item()): i 
        for i in range(num_instances)
    }
    
    # 构建 instances_by_frame
    instances_by_frame = []
    for frame_idx in frame_indices:
        frame_instances = []
        for ins_id in range(num_instances):
            # 检查该实例在该帧是否活跃
            if not per_frame_instance_mask[frame_idx, ins_id]:
                continue
            
            frame_instances.append({
                'intid': ins_id,  # 连续整数ID（数组索引，从0开始）
                'original_id': int(instances_true_id[ins_id].item()),  # 原始实例ID
                'T_ow': instances_pose[frame_idx, ins_id].cpu().numpy(),  # [4, 4] - Object->World
                'size_lwh': instances_size[ins_id].cpu().numpy(),  # [3] - 长度、宽度、高度
            })
        instances_by_frame.append(frame_instances)
    
    return instance_mapping, instances_by_frame
```

**关键属性说明**：

- `pixel_source.instances_pose`: Tensor [frame_num, instance_num, 4, 4]
  - 每个实例在每帧的位姿（Object->World 变换矩阵）
  - 已经过坐标对齐（相对于第一帧的 ego pose）

- `pixel_source.instances_size`: Tensor [instance_num, 3]
  - 每个实例的尺寸（长度、宽度、高度）
  - 注意：这是平均尺寸（如果同一实例在不同帧有不同尺寸，会取平均）

- `pixel_source.per_frame_instance_mask`: Tensor [frame_num, instance_num]
  - 每帧每个实例是否活跃（bool）
  - 用于判断某个实例在某一帧是否存在

- `pixel_source.instances_true_id`: Tensor [instance_num]
  - 原始实例ID（在数据集中的真实ID）
  - 用于构建 instance_mapping

**注意**：
- intid 是数组索引，从 0 开始
- 如果需要从 1 开始（与 project_lidar.py 一致），需要在使用时 +1
- 某些场景可能没有动态物体标注，`instances_pose` 可能为 None，需要检查

### 6. 数据获取总结

所有数据都可以通过现有接口获取，不需要扩展 MultiSceneDataset：

| 数据 | 获取方式 | 接口/属性 |
|------|----------|-----------|
| 场景数据 | `dataset.get_scene(scene_id)` | MultiSceneDataset |
| 帧数据（图像、深度、相机参数） | `scene_dataset.pixel_source.get_image(img_idx)` | DrivingDataset.pixel_source |
| LiDAR 点云 | `scene_dataset.lidar_source.origins/directions/ranges/timesteps` | DrivingDataset.lidar_source |
| Ego Pose | **直接从 lidar_source 获取** | `scene_dataset.lidar_source.lidar_to_worlds[frame_idx]` |
| 动态物体实例 | **直接从 pixel_source 获取** | `pixel_source.instances_pose/size/per_frame_instance_mask/instances_true_id` |

**注意事项**：

1. **坐标系一致性**：确保所有数据使用相同的坐标系约定
2. **数据可用性检查**：在使用前检查数据是否可用（如 `lidar_source is None`、`instances_pose is None`）
3. **错误处理**：如果某些数据不可用，生成器应能够优雅处理（如只生成静态背景点云）
4. **Ego Pose 获取**：直接从 `lidar_source.lidar_to_worlds[frame_idx]` 获取（对于 nuScenes，lidar 坐标系就是 ego 坐标系；对于其他数据集，lidar_to_world 也是从 ego pose 计算的）
5. **实例ID映射**：注意 `pixel_source` 中的 intid 是数组索引（从0开始），如果需要从1开始需要转换
6. **数据集适配**：不同数据集的数据格式可能不同，需要适配层统一接口

---

## 实现细节

### 1. 静态动态分离算法

**重要**：必须逐帧分离，不能将所有帧点云合并后再用单帧bbox分离。

```python
def _separate_static_dynamic(
    self,
    points_world: np.ndarray,  # [N, 3] - 单帧的世界坐标点云
    colors: np.ndarray,  # [N, 3] - 单帧的RGB颜色
    instances: List[Dict],  # 该帧的实例列表，每个元素包含 {'intid', 'T_ow', 'size_lwh'}
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    将单帧点云分割为静态背景和动态物体。
    
    注意：此函数处理单帧点云，多帧累积的点云需要逐帧调用此函数后再合并结果。
    """
    N = points_world.shape[0]
    any_obj_mask = np.zeros(N, dtype=bool)
    dynamic_points_dict = {}
  
    # 遍历该帧的所有实例
    for instance in instances:
        intid = instance['intid']
        T_ow = instance['T_ow']  # [4, 4] - Object->World（该帧的位姿）
        size_lwh = instance['size_lwh']  # [3] - length, width, height
      
        # World->Object
        T_wo = np.linalg.inv(T_ow)
      
        # 将点变换到物体局部坐标系
        points_homo = np.concatenate([
            points_world,
            np.ones((N, 1), dtype=np.float32)
        ], axis=1)  # [N, 4]
        points_local = (T_wo @ points_homo.T).T[:, :3]  # [N, 3]
      
        # 判断点是否在 bbox 内
        half = size_lwh.astype(np.float32) / 2.0
        mask = (np.abs(points_local) <= (half + 1e-6)).all(axis=1)  # [N]
      
        if not np.any(mask):
            continue
      
        # 保存动态物体点云（局部坐标 + RGB）
        # 注意：如果一个点同时在多个实例的 bbox 内，它会被分配给遍历顺序中的第一个实例
        # 因为使用 any_obj_mask |= mask，一旦被标记就不会再分配给其他实例
        dynamic_points_local = points_local[mask]  # [M, 3]
        dynamic_colors = colors[mask]  # [M, 3]
        dynamic_points_dict[intid] = np.concatenate([dynamic_points_local, dynamic_colors], axis=1)  # [M, 6]
      
        # 更新全局掩码（标记已被分配的点）
        any_obj_mask |= mask
  
    # 静态背景点云（世界坐标 + RGB）
    background_points = points_world[~any_obj_mask]  # [M1, 3]
    background_colors = colors[~any_obj_mask]  # [M1, 3]
    background = np.concatenate([background_points, background_colors], axis=1)  # [M1, 6]
  
    return background, dynamic_points_dict
```

**多帧累积的处理方式**：

```python
# 伪代码：逐帧处理
all_backgrounds = []  # List[np.ndarray]
all_dynamic_objects = {}  # Dict[int, List[np.ndarray]]

for frame_idx in frame_indices:
    # 1. 生成该帧的点云
    points_world_frame, colors_frame = generate_pointcloud_for_frame(frame_idx)
    
    # 2. 获取该帧的实例信息
    frame_instances = instances_by_frame[frame_idx]  # List[Dict]
    
    # 3. 使用该帧的实例bbox分离该帧的点云
    background_frame, dynamic_objects_frame = _separate_static_dynamic(
        points_world_frame, colors_frame, frame_instances
    )
    
    # 4. 累积结果
    all_backgrounds.append(background_frame)
    for intid, points in dynamic_objects_frame.items():
        if intid not in all_dynamic_objects:
            all_dynamic_objects[intid] = []
        all_dynamic_objects[intid].append(points)

# 5. 合并所有帧的结果
background = np.concatenate(all_backgrounds, axis=0)  # [N_total, 6]
dynamic_objects = {
    intid: np.concatenate(points_list, axis=0)
    for intid, points_list in all_dynamic_objects.items()
}
```

### 2. 局部坐标转换

动态物体点云使用局部坐标，转换公式：

```python
# 世界坐标 -> 局部坐标
T_wo = np.linalg.inv(T_ow)  # World->Object
p_local = (T_wo @ [p_world, 1])[:3]  # 齐次坐标变换后取前3维

# 局部坐标 -> 世界坐标（如果需要）
p_world = (T_ow @ [p_local, 1])[:3]
```

### 3. LiDAR 点上色流程

参考 `tools/project_lidar.py` 的 `colorize_points_vehicle` 函数：

1. 获取车辆坐标点云（从世界坐标转换：`points_vehicle = (T_wv @ points_world_homo)[:3]`）
2. 将车辆坐标点云变换到世界坐标系（使用 ego pose：`points_world = (T_vw @ points_vehicle_homo)[:3]`）
3. 按相机优先级遍历所有相机
4. 对每个相机：
   - 加载图像和内参、外参
   - 将世界坐标点投影到图像平面
   - 从图像中采样颜色
   - 覆盖之前的颜色（使用 Z-buffer 或距离最近的点）

**注意**：
- `lidar_source` 中的点已经是世界坐标，但如果需要给点上色，通常需要车辆坐标作为中间步骤
- 对于给点上色，通常使用车辆坐标，然后转换为世界坐标用于投影

### 4. 点云合并策略

- **静态背景**：所有帧的背景点云合并为一个点云
- **动态物体**：每个实例的点云按帧分别存储，或合并所有帧的点云（取决于使用场景）

**注意**：
- 如果一个点同时属于多个实例的 bbox，它会被分配给遍历顺序中的第一个实例
- 这是因为使用 `any_obj_mask |= mask` 标记，一旦被标记为动态物体，就不会再分配给其他实例

---

## 文件结构

```
datasets/pointcloud_generators/
├── __init__.py                 # 导出所有生成器类
├── base.py                     # RGBPointCloudGenerator 基类
├── monocular.py                # MonocularRGBPointCloudGenerator
└── lidar.py                    # LiDARRGBPointCloudGenerator
```

### __init__.py

```python
from .base import RGBPointCloudGenerator
from .monocular import MonocularRGBPointCloudGenerator
from .lidar import LiDARRGBPointCloudGenerator

__all__ = [
    'RGBPointCloudGenerator',
    'MonocularRGBPointCloudGenerator',
    'LiDARRGBPointCloudGenerator',
]
```

---

## 使用示例

### 1. 基本使用（单目）

```python
from datasets.pointcloud_generators import MonocularRGBPointCloudGenerator

# 创建生成器
generator = MonocularRGBPointCloudGenerator(
    chosen_cam_ids=[0],
    sparsity='Drop50',
    filter_sky=True,
    depth_consistency=True,
    use_bbx=True,
    downscale=2,
    crop_aabb=np.array([[-20, -20, -5], [20, 5, 20]]),
    input_aabb=np.array([[-20, -20, -5], [20, 5, 20]]),
)

# 生成点云
result = generator.generate_pointcloud(
    dataset=dataset,
    scene_id=0,
    segment_id=0,
)

# 获取结果
background_points = result['background']  # np.ndarray [N, 6] - 世界坐标 + RGB
dynamic_objects = result['dynamic_objects']  # Dict[int, np.ndarray] - {intid: [M, 6]}
instance_mapping = result['instance_mapping']  # Dict[int, int]

# 保存点云（示例：保存为 .ply 格式，需要先转换为 o3d 格式，或使用其他方式保存）
# 方式1：使用 numpy 保存为 .npy 格式
np.save("background.npy", background_points)
for intid, points in dynamic_objects.items():
    np.save(f"dynamic_{intid}.npy", points)

# 方式2：如果需要保存为 .ply 格式，可以使用 open3d 转换
# import open3d as o3d
# background_pcd = o3d.geometry.PointCloud()
# background_pcd.points = o3d.utility.Vector3dVector(background_points[:, :3])
# background_pcd.colors = o3d.utility.Vector3dVector(background_points[:, 3:6] / 255.0)
# o3d.io.write_point_cloud("background.ply", background_pcd)
```

### 2. 基本使用（LiDAR）

```python
from datasets.pointcloud_generators import LiDARRGBPointCloudGenerator

# 创建生成器
generator = LiDARRGBPointCloudGenerator(
    sparsity='full',
    use_bbx=True,
    crop_aabb=np.array([[-20, -20, -5], [20, 5, 20]]),
    input_aabb=np.array([[-20, -20, -5], [20, 5, 20]]),
    resomult=0.5,
)

# 生成点云
result = generator.generate_pointcloud(
    dataset=dataset,
    scene_id=0,
    segment_id=0,
)

# 使用结果（同上）
```

### 3. 在调度器中使用

```python
from datasets.pointcloud_generators import MonocularRGBPointCloudGenerator

# 创建生成器
generator = MonocularRGBPointCloudGenerator(...)

# 在调度器中生成点云
scheduler = dataset.create_scheduler(...)

# 为当前段生成点云
scene_id = scheduler.get_current_info()['scene_id']
segment_id = scheduler.get_current_info()['segment_id_in_scene']

result = generator.generate_pointcloud(
    dataset=dataset,
    scene_id=scene_id,
    segment_id=segment_id,
)
```

---

## 迁移策略

### 1. 向后兼容

- 不要考虑向后兼容

### 2. 代码迁移步骤

1. **第一步**：创建新的基类文件 `base.py`，实现基础功能
2. **第二步**：创建 `monocular.py`，迁移并重构 `MonocularRGBPointCloudGenerator`，添加静态动态分离
3. **第三步**：创建 `lidar.py`，实现 `LiDARRGBPointCloudGenerator`，添加静态动态分离
4. **第四步**：更新所有使用点云生成器的代码，适配新的返回格式
5. **第五步**：删除旧的 `rgb_pointcloud_generator.py` 文件

**注意**：不需要扩展 MultiSceneDataset 的接口，所有数据都通过现有接口获取。

---

## 反直觉检查清单

### 1. 坐标系统检查

- [ ] **世界坐标正确**：静态背景点云使用世界坐标
- [ ] **局部坐标正确**：动态物体点云使用局部坐标（相对于物体坐标系）
- [ ] **坐标转换正确**：局部坐标 = (T_ow^-1 @ p_world)[:3]
- [ ] **坐标系一致性**：所有点云使用相同的坐标系约定

### 2. 3D Bbox 检查

- [ ] **bbox 判断正确**：使用 AABB 判断点是否在物体框内
- [ ] **尺寸定义正确**：size_lwh 是长度、宽度、高度（不是半边长）
- [ ] **容差处理正确**：使用 epsilon（如 1e-6）处理浮点误差
- [ ] **变换矩阵正确**：T_ow 是 Object->World，需要取逆得到 World->Object

### 3. 实例信息检查

        - [ ] **实例映射正确**：原始ID到连续整数ID的映射正确（pixel_source 中 intid 从0开始，如需要从1开始需转换）
- [ ] **实例数据完整**：每个实例包含 T_ow 和 size_lwh
- [ ] **空实例处理**：如果场景没有动态物体，正确返回空的 dynamic_objects 字典
- [ ] **多帧实例处理**：同一实例在不同帧可能有不同的 T_ow 和 size_lwh
- [ ] **逐帧分离正确**：必须逐帧分离点云，不能将所有帧点云合并后用单帧bbox分离（动态物体会移动）
- [ ] **img_idx 生成正确**：使用相机的 `unique_cam_idx`（枚举序号），而不是原始的 `cam_id`

### 4. 数据获取检查

- [ ] **所有数据从 MultiSceneDataset 获取**：不直接读取文件系统，通过现有接口获取
- [ ] **场景数据获取**：通过 `dataset.get_scene(scene_id)` 获取，访问 `scene_data['dataset']` 获取 DrivingDataset 实例
- [ ] **LiDAR 点云获取**：通过 `scene_dataset.lidar_source` 获取（origins, directions, ranges, timesteps）
- [ ] **图像数据获取**：通过 `scene_dataset.pixel_source.get_image(img_idx)` 获取
- [ ] **实例信息获取**：根据数据集类型，从文件系统读取或通过数据集 API 获取（见"从 MultiSceneDataset 获取数据"章节）
- [ ] **ego pose 获取**：直接从 `lidar_source.lidar_to_worlds[frame_idx]` 获取（对于 nuScenes 等数据集，lidar 坐标系就是 ego 坐标系）
- [ ] **动态物体实例获取**：直接从 `pixel_source.instances_pose/size/per_frame_instance_mask/instances_true_id` 获取
- [ ] **实例ID映射**：注意 intid 是数组索引（从0开始），如需要从1开始需转换

### 5. 点云生成检查

- [ ] **静态背景点云正确**：不属于任何动态物体的点
- [ ] **动态物体点云正确**：属于动态物体的点使用局部坐标
- [ ] **点云不重复**：一个点只属于静态背景或一个动态物体，不重复
- [ ] **颜色正确**：点云颜色与原始图像一致

### 6. 边界框和滤波检查

- [ ] **裁剪边界框正确**：使用 crop_aabb 裁剪点云
- [ ] **输入边界框正确**：使用 input_aabb 分割和滤波
- [ ] **滤波参数合理**：静态和动态点云使用合适的滤波参数

---

## 总结

本次重构的主要改进：

1. **功能扩展**：支持静态背景和动态物体的分离输出
2. **代码组织**：基类和子类分离到独立文件，提高可维护性
3. **新增生成器**：添加 LiDARRGBPointCloudGenerator，支持从 LiDAR 生成点云
4. **数据源统一**：所有数据通过 MultiSceneDataset 获取，不直接读取文件
5. **接口统一**：所有生成器使用统一的接口，返回包含静态和动态点云的字典

该设计遵循以下原则：

- **模块化**：每个生成器独立文件，易于维护和扩展
- **可扩展性**：易于添加新的点云生成策略
- **数据一致性**：所有数据通过统一接口获取，确保一致性
- **向后兼容**：尽可能保持向后兼容，降低迁移成本
