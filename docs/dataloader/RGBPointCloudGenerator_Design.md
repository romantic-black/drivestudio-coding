# RGB 点云生成器设计文档

## 概述

本文档设计 RGB 点云生成器系统，用于从 `MultiSceneDataset` 的段中生成 RGB 单目点云。该系统基于 `notebooks/nuscenes_pcd_generation.ipynb` 中的点云生成流程，但改为从 `MultiSceneDataset` 获取数据，而不是直接从场景文件夹读取。

---

## 核心概念

### 1. 数据来源

**原始方式（notebook）**：
- 从场景文件夹直接读取文件（`images/`, `depth/`, `extrinsics/`, `intrinsics/`）
- 文件命名格式：`{frame_idx:03d}_{cam_id}.jpg/npy/txt`

**新方式（MultiSceneDataset）**：
- 从 `MultiSceneDataset` 获取段内帧数据
- 通过 `scene_data['dataset'].pixel_source.get_image(img_idx)` 获取图像和相机信息
- 通过 `scene_data['segments'][segment_id]['frame_indices']` 获取段内所有帧索引
- 通过 `scene_data['segments'][segment_id]['aabb']` 获取段的 AABB（如果配置了固定段AABB，所有段使用相同的AABB）
- 通过 `scene_data['dataset'].get_aabb()` 获取场景 AABB（用于场景级别的操作）

### 2. 点云生成流程

参考 `notebooks/nuscenes_pcd_generation.ipynb`，点云生成流程包括：

1. **数据准备**：获取段内所有帧的 RGB 图像、深度图、外参、内参
2. **深度预处理**：使用 `depth_utils.process_depth_for_use()` 将深度图插值到原始尺寸
3. **掩码应用**：
   - 深度一致性检查（可选）
   - 天空过滤（可选）
   - 下采样掩码（可选）
4. **点云生成**：
   - 反投影深度图到相机坐标系
   - 变换到世界坐标系
   - 累积所有帧的点云
5. **后处理**：
   - 边界框裁剪（可选）
   - 统计滤波
   - 均匀下采样

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
        pass
    
    @abstractmethod
    def generate_pointcloud(
        self,
        dataset: MultiSceneDataset,
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

### MonocularRGBPointCloudGenerator（子类）

```python
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
        dataset: MultiSceneDataset,
        scene_id: int,
        segment_id: int,
    ) -> o3d.geometry.PointCloud:
        """
        为指定场景和段生成 RGB 点云。
        
        流程：
        1. 获取段内所有帧索引
        2. 根据稀疏度过滤帧
        3. 按相机分组加载所有选中帧的 RGB 图像、深度图、外参、内参、天空掩码
        4. 对每个相机分别应用深度一致性检查（如果启用）
           - 关键：确保深度一致性检查只在同一相机的连续帧之间进行
        5. 生成点云（反投影、变换、累积），合并所有相机的数据
        6. 应用边界框裁剪（如果启用）
        7. 滤波和下采样
        
        Args:
            dataset: MultiSceneDataset 实例
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            pointcloud: Open3D 点云对象
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
        pass
    
    def _load_frame_data(
        self,
        dataset: MultiSceneDataset,
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
        pass
    
    def _depth_consistency_check(
        self,
        frame_data_list: List[Dict],
        H: int,
        W: int,
    ) -> List[np.ndarray]:
        """
        检查连续帧之间的深度一致性。
        
        关键修复：投影到上一帧时使用上一帧的内参（last_K），而不是当前帧的内参。
        这确保了当不同帧的内参不同时，投影计算仍然正确。
        
        Args:
            frame_data_list: 帧数据列表（按时间顺序，必须是同一相机的帧）
            H: 图像高度
            W: 图像宽度
            
        Returns:
            consistency_masks: List[np.ndarray] - 每个帧的一致性掩码 [H, W]
        """
        pass
    
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
        pass
    
    def _generate_pointcloud_from_frames(
        self,
        frame_data_list: List[Dict],
        consistency_masks: List[np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        从帧数据生成点云（保留用于向后兼容，实际使用 _generate_pointcloud_from_frames_by_camera）。
        
        Args:
            frame_data_list: 帧数据列表
            consistency_masks: 一致性掩码列表
            H: 图像高度
            W: 图像宽度
            
        Returns:
            pointcloud: [N, 6] - 点云数据（前3列是位置，后3列是颜色）
        """
        pass
```

---

## MultiSceneDataset 扩展

### 新增方法

```python
class MultiSceneDataset:
    # ... 现有方法 ...
    
    def get_segment_frames(
        self,
        scene_id: int,
        segment_id: int,
    ) -> List[int]:
        """
        获取段内所有帧索引。
        
        Args:
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            frame_indices: 段内所有帧索引列表（已排序、去重）
        """
        pass
    
    def get_frame_data(
        self,
        scene_id: int,
        frame_idx: int,
        cam_idx: int,
    ) -> Dict:
        """
        获取指定帧和相机的数据。
        
        Args:
            scene_id: 场景ID
            frame_idx: 帧索引
            cam_idx: 相机索引（在 camera_list 中的索引）
            
        Returns:
            Dict包含：
                - 'image': Tensor [H, W, 3] - RGB图像
                - 'extrinsic': Tensor [4, 4] - 外参（cam_to_world）
                - 'intrinsic': Tensor [4, 4] - 内参（4x4矩阵）
                - 'depth': Tensor [H, W] - 深度图（如果可用）
                - 'sky_mask': Tensor [H, W] - 天空掩码（如果可用，True表示天空区域）
        """
        pass
```

---

## MultiSceneDatasetScheduler 扩展

### 新增方法

```python
class MultiSceneDatasetScheduler:
    # ... 现有方法 ...
    
    def generate_segment_pointcloud(
        self,
        pointcloud_generator: RGBPointCloudGenerator,
        scene_id: Optional[int] = None,
        segment_id: Optional[int] = None,
    ) -> o3d.geometry.PointCloud:
        """
        为当前段（或指定段）生成点云。
        
        Args:
            pointcloud_generator: 点云生成器实例
            scene_id: 场景ID（如果为None，使用当前场景）
            segment_id: 段ID（如果为None，使用当前段）
            
        Returns:
            pointcloud: Open3D 点云对象
        """
        pass
    
    def generate_all_segment_pointclouds(
        self,
        pointcloud_generator: RGBPointCloudGenerator,
        scene_id: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[int, o3d.geometry.PointCloud]:
        """
        为场景的所有段生成点云。
        
        Args:
            pointcloud_generator: 点云生成器实例
            scene_id: 场景ID（如果为None，使用当前场景）
            save_dir: 保存目录（如果为None，不保存）
            
        Returns:
            Dict[segment_id, pointcloud]: 每个段的点云字典
        """
        pass
```

---

## 实现细节

### 1. 数据获取流程

**从 MultiSceneDataset 获取段内帧数据**：

```python
def _get_segment_frames(
    self,
    dataset: MultiSceneDataset,
    scene_id: int,
    segment_id: int,
) -> List[int]:
    """
    获取段内所有帧索引。
    """
    # 1. 获取场景数据
    scene_data = dataset.get_scene(scene_id)
    if scene_data is None:
        raise ValueError(f"Scene {scene_id} not found")
    
    # 2. 获取段信息
    segment = scene_data['segments'][segment_id]
    
    # 3. 返回段内所有帧索引（已排序、去重）
    return segment['frame_indices']
```

**加载帧数据**：

```python
def _load_frame_data(
    self,
    dataset: MultiSceneDataset,
    scene_id: int,
    frame_idx: int,
    cam_id: int,
) -> Optional[Dict]:
    """
    加载指定帧和相机的数据。
    """
    # 1. 通过 dataset.get_frame_data() 获取数据
    try:
        frame_data = dataset.get_frame_data(scene_id, frame_idx, cam_id)
    except Exception as e:
        logger.warning(f"Failed to load frame data for scene {scene_id}, frame {frame_idx}, cam {cam_id}: {e}")
        return None
    
    # 2. 转换为numpy数组
    rgb = frame_data['image'].cpu().numpy()  # [H, W, 3]
    depth = frame_data['depth'].cpu().numpy()  # [H, W]
    extrinsic = frame_data['extrinsic'].cpu().numpy()  # [4, 4]
    
    # 3. 转换内参为3x3（如果原本是4x4）
    intrinsic = frame_data['intrinsic'].cpu().numpy()  # [3, 3] or [4, 4]
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    
    # 4. 获取天空掩码（如果存在）
    sky_mask = frame_data.get('sky_mask')  # Tensor [H, W] or None
    
    return {
        'rgb': rgb,
        'depth': depth,
        'extrinsic': extrinsic,
        'intrinsic': intrinsic,
        'sky_mask': sky_mask,  # Tensor [H, W] or None
    }
```

### 2. 深度预处理

**使用 depth_utils 预处理深度图**：

```python
def _preprocess_depth(
    self,
    depth: np.ndarray,
    target_shape: Tuple[int, int],
    depth_file_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    预处理深度图（插值到目标尺寸）。
    
    如果 depth_file_path 提供，使用 depth_utils 加载和预处理。
    否则，直接使用提供的深度图（可能需要插值）。
    """
    if depth_file_path is not None:
        # 使用 depth_utils 加载和预处理
        from depth_utils import process_depth_for_use
        depth_processed, metadata = process_depth_for_use(
            depth_file_path,
            target_shape=target_shape,
        )
        return depth_processed, metadata
    else:
        # 直接使用提供的深度图（可能需要插值）
        if depth.shape != target_shape:
            # 使用双线性插值
            depth_processed = cv2.resize(
                depth,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            depth_processed = depth
        
        return depth_processed, {'ori_shape': depth.shape}
```

### 3. 点云生成主循环

**按相机分组处理，然后生成点云**：

```python
def generate_pointcloud(
    self,
    dataset: MultiSceneDataset,
    scene_id: int,
    segment_id: int,
) -> o3d.geometry.PointCloud:
    """
    为指定场景和段生成 RGB 点云。
    """
    # 1. 获取段内所有帧索引
    frame_indices = self._get_segment_frames(dataset, scene_id, segment_id)
    
    # 2. 根据稀疏度过滤帧
    filtered_frame_indices = self._apply_sparsity_filter(frame_indices)
    
    # 3. 按相机分组加载帧数据（关键：避免跨相机的一致性检查）
    frame_data_by_camera = {cam_id: [] for cam_id in self.chosen_cam_ids}
    for frame_idx in filtered_frame_indices:
        for cam_id in self.chosen_cam_ids:
            frame_data = self._load_frame_data(dataset, scene_id, frame_idx, cam_id)
            if frame_data is not None:
                frame_data_by_camera[cam_id].append((frame_idx, frame_data))
    
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
    
    # ... 后续处理（边界框裁剪、滤波等）
```

**从按相机分组的帧数据生成点云**：

```python
def _generate_pointcloud_from_frames_by_camera(
    self,
    frame_data_list_by_camera: Dict[int, List[Dict]],
    consistency_masks_by_camera: Dict[int, List[np.ndarray]],
    H: int,
    W: int,
) -> np.ndarray:
    """
    从按相机分组的帧数据生成点云。
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
                    sky_mask = ~sky_mask.astype(bool)
                else:
                    sky_mask = np.ones((H, W), dtype=bool)
            else:
                # 如果没有天空掩码，根据 filter_sky 决定
                if self.filter_sky:
                    logger.warning(f"No sky mask available for camera {cam_id}, frame {i}, skipping sky filtering")
                    sky_mask = np.ones((H, W), dtype=bool)
                else:
                    sky_mask = np.ones((H, W), dtype=bool)
            
            # 应用下采样掩码
            if downscale_mask is not None:
                final_mask = consistency_mask & sky_mask & downscale_mask
            else:
                final_mask = consistency_mask & sky_mask
            
            # ... 后续处理（提取像素、反投影、变换等）
    
    # 合并所有点云块
    accumulated_pointcloud = np.concatenate(color_pointclouds, axis=0)
    return accumulated_pointcloud
```

### 4. 深度一致性检查

**参考 notebook 的深度一致性检查**：

```python
def _depth_consistency_check(
    self,
    frame_data_list: List[Dict],
    H: int,
    W: int,
) -> List[np.ndarray]:
    """
    检查连续帧之间的深度一致性。
    """
    if not self.depth_consistency:
        return [np.ones((H, W), dtype=bool) for _ in frame_data_list]
    
    depth_masks = []
    
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
        last_K = frame_data_list[i-1]['intrinsic']  # [3, 3] - 上一帧内参（关键修复）
        
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
        
        # 投影到上一帧的图像平面（使用上一帧内参 - 关键修复）
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
pointcloud_generator = MonocularRGBPointCloudGenerator(
    chosen_cam_ids=[0],  # 只使用前置摄像头
    sparsity='Drop50',
    filter_sky=True,
    depth_consistency=True,
    use_bbx=True,
    downscale=2,
)

# 3. 为指定场景和段生成点云
scene_id = 0
segment_id = 0
pointcloud = pointcloud_generator.generate_pointcloud(
    dataset=dataset,
    scene_id=scene_id,
    segment_id=segment_id,
)

# 4. 保存点云
import open3d as o3d
o3d.io.write_point_cloud(f"scene_{scene_id}_segment_{segment_id}.ply", pointcloud)
```

### 2. 使用调度器生成点云

```python
# 1. 创建调度器
scheduler = dataset.create_scheduler(
    batches_per_segment=20,
    segment_order="sequential",
    scene_order="random",
)

# 2. 为当前段生成点云
pointcloud = scheduler.generate_segment_pointcloud(
    pointcloud_generator=pointcloud_generator,
)

# 3. 为场景的所有段生成点云
all_pointclouds = scheduler.generate_all_segment_pointclouds(
    pointcloud_generator=pointcloud_generator,
    scene_id=0,
    save_dir="pointclouds/scene_0",
)
```

### 3. 批量生成点云

```python
# 为所有训练场景的所有段生成点云
for scene_id in train_scene_ids:
    scene_data = dataset.get_scene(scene_id)
    if scene_data is None:
        continue
    
    for segment_id in range(len(scene_data['segments'])):
        try:
            pointcloud = pointcloud_generator.generate_pointcloud(
                dataset=dataset,
                scene_id=scene_id,
                segment_id=segment_id,
            )
            
            # 保存点云
            save_path = f"pointclouds/scene_{scene_id}/segment_{segment_id}.ply"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            o3d.io.write_point_cloud(save_path, pointcloud)
            
            print(f"Generated pointcloud for scene {scene_id}, segment {segment_id}")
        except Exception as e:
            print(f"Failed to generate pointcloud for scene {scene_id}, segment {segment_id}: {e}")
```

---

## 与原始 Notebook 的差异

### 1. 数据来源

| 方面 | Notebook | MultiSceneDataset |
|------|----------|-------------------|
| 图像 | 从 `images/` 文件夹读取 | 从 `pixel_source.get_image()` 获取 |
| 深度图 | 从 `depth/` 文件夹读取 | 从 `_get_depth()` 获取 |
| 外参 | 从 `extrinsics/` 文件夹读取 | 从 `cam_infos['camera_to_world']` 获取 |
| 内参 | 从 `intrinsics/` 文件夹读取 | 从 `cam_infos['intrinsics']` 获取 |
| 帧索引 | 从文件名解析 | 从 `segment['frame_indices']` 获取 |

### 2. 深度预处理

- **Notebook**：使用 `depth_utils.process_depth_for_use()` 从文件加载和预处理
- **MultiSceneDataset**：深度图可能已经加载到内存，需要检查是否需要预处理

### 3. 天空掩码

- **Notebook**：从 `sky_masks/` 文件夹读取
- **MultiSceneDataset**：从 `image_infos['sky_masks']` 获取（如果 `load_sky_mask=True`）
  - 在 `get_frame_data()` 中返回天空掩码
  - 在点云生成时，如果 `filter_sky=True`，取反天空掩码（保留非天空区域）
  - 如果天空掩码不可用，记录警告但继续处理

### 4. 坐标对齐

- **Notebook**：使用第一帧第一相机的pose作为参考坐标系
- **MultiSceneDataset**：外参可能已经对齐，需要确认

### 5. 段 AABB 配置

- **MultiSceneDataset**：支持全局固定的段 AABB 配置（`fixed_segment_aabb`）
  - 如果配置了 `fixed_segment_aabb`，所有段使用此固定 AABB
  - 否则，每个段基于段内帧的 lidar 数据计算独立的 AABB
  - 固定 AABB 格式：`[2, 3]`，其中 `aabb[0]` 是 `[x_min, y_min, z_min]`，`aabb[1]` 是 `[x_max, y_max, z_max]`
  - 坐标系：x=front, y=left, z=up（与 `_compute_segment_aabb` 一致）
  - 点云生成器可以通过 `scene_data['segments'][segment_id]['aabb']` 获取段的 AABB

---

## 反直觉检查清单

### 1. 数据获取检查

- [ ] **帧索引正确**：段内帧索引与场景数据一致
- [ ] **相机ID映射正确**：`chosen_cam_ids` 映射到 `camera_list` 中的索引
- [ ] **图像索引计算正确**：`img_idx = frame_idx * num_cams + cam_idx`
- [ ] **深度图可用**：所有帧都有对应的深度图（或占位符）

### 2. 深度预处理检查

- [ ] **深度图尺寸正确**：深度图已插值到原始图像尺寸
- [ ] **深度值范围合理**：深度值在合理范围内（> 0，< 200m）
- [ ] **深度图格式正确**：深度图为 numpy 数组，形状为 [H, W]

### 3. 点云生成检查

- [ ] **反投影正确**：相机坐标系点云正确生成
- [ ] **坐标变换正确**：世界坐标系点云正确生成
- [ ] **点云累积正确**：所有帧的点云正确累积
- [ ] **颜色正确**：点云颜色与RGB图像一致
- [ ] **按相机分组**：多相机时，深度一致性检查只在同一相机的连续帧之间进行
- [ ] **深度一致性内参**：投影到上一帧时使用上一帧的内参，而不是当前帧的内参
- [ ] **天空掩码应用**：如果启用天空过滤，正确应用天空掩码（取反以保留非天空区域）

### 4. 后处理检查

- [ ] **边界框裁剪正确**：点云正确裁剪到边界框
- [ ] **滤波参数合理**：统计滤波和均匀下采样参数合理
- [ ] **点云数量合理**：最终点云数量在合理范围内

### 5. 性能检查

- [ ] **内存占用合理**：不会因为加载过多帧导致内存溢出
- [ ] **处理速度合理**：点云生成速度在可接受范围内
- [ ] **缓存机制**：合理利用 MultiSceneDataset 的缓存机制

---

## 关键修复说明

### 1. 深度一致性检查的内参修复

**问题**：原始实现使用当前帧的内参投影到上一帧，导致内参不同时投影错误。

**修复**：投影到上一帧时使用上一帧的内参（`last_K`），确保投影计算正确。

```python
# 修复前（错误）
last_x = (fx * last_coordinates[:, 0] + cx * last_coordinates[:, 2]) / last_coordinates[:, 2]

# 修复后（正确）
last_K = frame_data_list[i-1]['intrinsic']  # 上一帧内参
last_fx, last_fy = last_K[0, 0], last_K[1, 1]
last_cx, last_cy = last_K[0, 2], last_K[1, 2]
last_x = (last_fx * last_coordinates[:, 0] + last_cx * last_coordinates[:, 2]) / last_coordinates[:, 2]
```

### 2. 帧和相机交错问题修复

**问题**：多相机时，帧和相机交错排列，导致深度一致性检查在跨相机之间进行，使有效像素被错误地无效化。

**修复**：按相机分组处理，每个相机单独进行深度一致性检查。

```python
# 修复前（错误）
frame_data_list = []  # 交错排列：[cam0_frame0, cam1_frame0, cam0_frame1, ...]
consistency_masks = self._depth_consistency_check(frame_data_list, H, W)  # 跨相机检查

# 修复后（正确）
frame_data_by_camera = {cam_id: [] for cam_id in self.chosen_cam_ids}  # 按相机分组
for cam_id in self.chosen_cam_ids:
    consistency_masks_by_camera[cam_id] = self._depth_consistency_check(frame_data_list, H, W)  # 单相机检查
```

### 3. 天空掩码实现

**问题**：天空掩码始终为 True，`filter_sky` 标志实际上不起作用。

**修复**：
- 在 `get_frame_data()` 中从 `image_infos['sky_masks']` 获取天空掩码
- 在点云生成时，如果 `filter_sky=True`，取反天空掩码（保留非天空区域）
- 如果天空掩码不可用，记录警告但继续处理

```python
# 修复前（错误）
if self.filter_sky:
    sky_mask = np.ones((H, W), dtype=bool)  # 始终为 True，不起作用

# 修复后（正确）
sky_mask = frame_data.get('sky_mask')
if sky_mask is not None:
    if self.filter_sky:
        sky_mask = ~sky_mask.astype(bool)  # 取反，保留非天空区域
```

## 总结

RGB 点云生成器系统的设计遵循以下原则：

1. **抽象接口**：基类定义通用接口，子类实现具体策略
2. **数据复用**：从 `MultiSceneDataset` 获取数据，避免重复加载
3. **流程一致**：保持与 notebook 的点云生成流程一致
4. **灵活配置**：支持多种稀疏度、滤波选项等
5. **易于扩展**：支持未来添加其他点云生成策略（如立体视觉）
6. **正确性保证**：修复了深度一致性检查的内参错误、帧和相机交错问题，以及天空掩码的实现

该设计允许在不修改 `MultiSceneDataset` 核心功能的情况下，实现从段中生成 RGB 点云的功能，同时保持与原始 notebook 流程的兼容性。关键修复确保了多相机场景下的正确性和天空掩码的有效应用。

