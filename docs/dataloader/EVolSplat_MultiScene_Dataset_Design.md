# EVolSplat 多场景数据集设计文档

## 概述

本文档设计了一个多场景处理类 `EvolSplatDataset`，用于 EVolSplat 的 feed-forward 3DGS 训练。该类基于 Drivestudio 场景文件夹结构，尽量复用 `driving_dataset.py` 的单场景读取方法，避免使用 nerfstudio 的 config 系统。

---

## 核心概念回顾

### NeRF、3DGS 与 Feed-forward 3DGS

- **NeRF (Neural Radiance Fields)**: 基于神经网络的隐式场景表示，通过体积渲染生成新视角图像
- **3DGS (3D Gaussian Splatting)**: 基于显式 3D 高斯点的场景表示，通过光栅化渲染
- **Pre-scene 3DGS**: 每个场景独立训练，直接优化场景特定的高斯参数
- **Feed-forward 3DGS**: 通过神经网络直接预测高斯参数，支持零样本泛化到新场景

### EVolSplat 与标准 3DGS 的区别

| 特性 | 标准 3DGS (Splatfacto) | EVolSplat |
|------|----------------------|-----------|
| **训练方式** | Per-scene 优化 | Feed-forward 预测 |
| **参数优化** | 直接优化 Gaussian 参数 | 优化神经网络参数 |
| **场景支持** | 单场景训练 | 多场景联合训练 |
| **泛化能力** | 场景特定 | 零样本泛化 |
| **推理速度** | 需要优化迭代 | 单次前向传播 |

### Feed-forward Dataloader 的关键特性

1. **多场景数据组织**: 管理多个场景的数据，支持随机场景采样
2. **Source-Target 图像对**: 每次训练迭代提供 source 图像（用于特征提取）和 target 图像（用于监督）
3. **点云初始化**: 每个场景的 3D 点云独立存储，用于初始化高斯点
4. **最近邻选择**: 根据相机姿态自动选择最相关的 source 图像
5. **场景 ID 索引**: 通过 `scene_id` 在模型内部索引不同场景的点云和参数

---

## Drivestudio 场景文件夹结构

```
scene_dir/ (例如: /path/to/trainval/000)
├── images/                    # RGB图像目录
│   ├── 000_0.jpg (或 .png)   # 格式: {frame_idx:03d}_{cam_id}.jpg
│   ├── 000_1.jpg
│   ├── 001_0.jpg
│   └── ...
├── depth/                     # 深度图目录
│   ├── 000_0.npy             # 格式: {frame_idx:03d}_{cam_id}.npy
│   ├── 000_0_meta.npz        # 元数据文件（包含原始尺寸、内参等）
│   ├── 000_1.npy
│   ├── 000_1_meta.npz
│   └── ...
├── extrinsics/                # 相机外参（cam_to_world变换矩阵）
│   ├── 000_0.txt             # 格式: {frame_idx:03d}_{cam_id}.txt (4x4矩阵)
│   ├── 000_1.txt
│   └── ...
├── intrinsics/                # 相机内参（每个相机固定）
│   ├── 0.txt                 # 格式: {cam_id}.txt (fx, fy, cx, cy)
│   ├── 1.txt
│   └── ...
├── sky_masks/                 # 天空掩码（可选）
│   ├── 000_0.png             # 格式: {frame_idx:03d}_{cam_id}.png (0=sky, 255=non-sky)
│   └── ...
├── transforms.json           # 场景变换信息（可选）
└── [sparsity_dir]/           # 点云输出目录（如Drop50, Drop80等）
    └── 0.ply                 # 生成的点云文件
```

### 文件命名规则

- **图像**: `{frame_idx:03d}_{cam_id}.jpg` 或 `.png`
- **深度图**: `{frame_idx:03d}_{cam_id}.npy` (对应元数据: `{frame_idx:03d}_{cam_id}_meta.npz`)
- **外参**: `{frame_idx:03d}_{cam_id}.txt` (4x4矩阵，每行一个值)
- **内参**: `{cam_id}.txt` (fx, fy, cx, cy，每行一个值)

---

## 设计目标

1. **复用单场景读取**: 尽量调用 `DrivingDataset` 的单场景读取方法，避免重复实现
2. **多场景管理**: 支持管理多个场景，每个场景使用独立的 `DrivingDataset` 实例
3. **Feed-forward 格式**: 输出符合 EVolSplat 要求的 source-target 图像对格式
4. **点云加载**: 支持从 PLY 文件加载每个场景的点云
5. **最近邻选择**: 根据相机姿态选择最近的 source 图像
6. **场景采样**: 支持随机场景采样和 FPS 采样策略

---

## 类设计

### EvolSplatDataset

```python
class EvolSplatDataset:
    """
    多场景数据集类，用于 EVolSplat feed-forward 3DGS 训练。
    
    设计原则：
    1. 尽量复用 DrivingDataset 的单场景读取方法
    2. 管理多个场景，每个场景使用独立的 DrivingDataset 实例
    3. 提供符合 EVolSplat 要求的数据格式
    """
    
    def __init__(
        self,
        scene_dirs: List[str],
        data_cfg_template: OmegaConf,
        num_source_image: int = 3,
        train_cameras_sampling_strategy: Literal["random", "fps"] = "random",
        load_3D_points: bool = True,
        pcd_dir_name: str = "Drop50",
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            scene_dirs: 场景目录列表，例如 ["/path/to/000", "/path/to/001", ...]
            data_cfg_template: DrivingDataset 的配置模板（OmegaConf）
            num_source_image: 每个 target 图像使用的 source 图像数量（默认3）
            train_cameras_sampling_strategy: 相机采样策略（"random" 或 "fps"）
            load_3D_points: 是否加载点云（默认True）
            pcd_dir_name: 点云目录名称（默认"Drop50"）
            device: 设备（默认CPU）
        """
        pass
    
    def __len__(self) -> int:
        """返回总图像数量（所有场景的总和）"""
        pass
    
    def get_scene_info(self) -> Dict[int, Dict]:
        """返回场景信息字典，包含每个场景的图像数量、点云路径等"""
        pass
    
    def get_batch(
        self,
        scene_id: int,
        target_image_idx: int,
        num_source_image: Optional[int] = None,
    ) -> Dict:
        """
        获取指定场景的 source-target 图像对批次。
        
        Args:
            scene_id: 场景ID（0-based）
            target_image_idx: 目标图像在该场景中的索引（0-based）
            num_source_image: source 图像数量（如果为None，使用初始化时的值）
            
        Returns:
            Dict包含：
                - 'scene_id': Tensor[1] - 场景ID
                - 'source': {
                    'image': Tensor[V, H, W, 3],
                    'extrinsics': Tensor[V, 4, 4],
                    'intrinsics': Tensor[V, 4, 4],
                    'depth': Tensor[V, H, W],
                    'source_id': Tensor[V]
                  }
                - 'target': {
                    'image': Tensor[1, H, W, 3],
                    'extrinsics': Tensor[1, 4, 4],
                    'intrinsics': Tensor[1, 4, 4],
                    'target_id': int
                  }
        """
        pass
    
    def get_point_cloud(self, scene_id: int) -> Optional[Dict[str, Tensor]]:
        """
        获取指定场景的点云。
        
        Args:
            scene_id: 场景ID（0-based）
            
        Returns:
            Dict包含：
                - 'points3D_xyz': Tensor[N, 3]
                - 'points3D_rgb': Tensor[N, 3]
            如果点云不存在，返回 None
        """
        pass
    
    def sample_random_batch(self) -> Dict:
        """
        随机采样一个批次（随机选择场景和target图像）。
        
        Returns:
            与 get_batch() 相同的格式
        """
        pass
```

---

## 接口设计

### 1. 初始化接口

```python
# 伪代码
def __init__(
    self,
    scene_dirs: List[str],
    data_cfg_template: OmegaConf,
    num_source_image: int = 3,
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random",
    load_3D_points: bool = True,
    pcd_dir_name: str = "Drop50",
    device: torch.device = torch.device("cpu"),
):
    """
    初始化步骤：
    1. 验证所有场景目录存在
    2. 为每个场景创建 DrivingDataset 实例
    3. 计算每个场景的图像数量（用于索引映射）
    4. 加载点云（如果启用）
    5. 构建全局图像索引到场景ID的映射
    """
    
    # 1. 验证场景目录
    self.scene_dirs = scene_dirs
    self.num_scenes = len(scene_dirs)
    for scene_dir in scene_dirs:
        assert os.path.exists(scene_dir), f"Scene directory not found: {scene_dir}"
    
    # 2. 为每个场景创建 DrivingDataset 实例
    self.scene_datasets = []
    for scene_idx, scene_dir in enumerate(scene_dirs):
        # 修改 data_cfg_template 的 scene_idx 和 data_root
        scene_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg_template))
        scene_cfg.data_root = os.path.dirname(scene_dir)  # 父目录
        scene_cfg.scene_idx = os.path.basename(scene_dir)  # 场景名称
        
        # 创建 DrivingDataset 实例
        scene_dataset = DrivingDataset(scene_cfg)
        self.scene_datasets.append(scene_dataset)
    
    # 3. 计算每个场景的图像数量
    self.num_images_per_scene = []
    self.scene_start_indices = [0]  # 全局图像索引的起始位置
    for scene_dataset in self.scene_datasets:
        num_images = scene_dataset.pixel_source.num_imgs
        self.num_images_per_scene.append(num_images)
        self.scene_start_indices.append(
            self.scene_start_indices[-1] + num_images
        )
    
    # 4. 加载点云（如果启用）
    self.point_clouds = []
    if load_3D_points:
        for scene_idx, scene_dir in enumerate(scene_dirs):
            pcd_path = self._find_point_cloud_path(scene_dir, pcd_dir_name)
            if pcd_path and os.path.exists(pcd_path):
                pcd = self._load_point_cloud(pcd_path)
                self.point_clouds.append(pcd)
            else:
                self.point_clouds.append(None)
                logger.warning(f"Point cloud not found for scene {scene_idx}: {pcd_path}")
    
    # 5. 存储配置
    self.num_source_image = num_source_image
    self.train_cameras_sampling_strategy = train_cameras_sampling_strategy
    self.device = device
```

### 2. 获取批次接口

```python
# 伪代码
def get_batch(
    self,
    scene_id: int,
    target_image_idx: int,
    num_source_image: Optional[int] = None,
) -> Dict:
    """
    获取批次的核心逻辑：
    1. 从指定场景的 DrivingDataset 获取 target 图像
    2. 获取 target 图像的相机参数
    3. 选择最近的 source 图像（基于相机姿态）
    4. 从 DrivingDataset 获取 source 图像和参数
    5. 加载深度图（如果可用）
    6. 组装成 EVolSplat 格式的批次
    """
    
    if num_source_image is None:
        num_source_image = self.num_source_image
    
    # 1. 获取场景数据集
    scene_dataset = self.scene_datasets[scene_id]
    
    # 2. 获取 target 图像信息
    # 注意：target_image_idx 是场景内的索引，需要转换为全局索引
    target_global_idx = self.scene_start_indices[scene_id] + target_image_idx
    
    # 从 DrivingDataset 获取图像和相机信息
    # 使用 pixel_source.get_image() 方法
    target_image_infos, target_cam_infos = scene_dataset.pixel_source.get_image(target_image_idx)
    
    target_image = target_image_infos['pixels']  # Tensor[H, W, 3]
    target_extrinsic = target_cam_infos['camera_to_world']  # Tensor[4, 4]
    target_intrinsic = target_cam_infos['intrinsics']  # Tensor[3, 3]
    
    # 3. 选择最近的 source 图像
    source_indices = self._select_source_images(
        scene_dataset=scene_dataset,
        target_image_idx=target_image_idx,
        target_extrinsic=target_extrinsic,
        num_source_image=num_source_image,
    )
    
    # 4. 获取 source 图像和参数
    source_images = []
    source_extrinsics = []
    source_intrinsics = []
    source_depths = []
    
    for source_idx in source_indices:
        # 使用 pixel_source.get_image() 方法
        source_image_infos, source_cam_infos = scene_dataset.pixel_source.get_image(source_idx)
        
        source_images.append(source_image_infos['pixels'])  # Tensor[H, W, 3]
        source_extrinsics.append(source_cam_infos['camera_to_world'])  # Tensor[4, 4]
        source_intrinsics.append(source_cam_infos['intrinsics'])  # Tensor[3, 3]
        
        # 加载深度图（优先使用 DrivingDataset 的深度图）
        depth = self._get_depth_from_driving_dataset(scene_dataset, source_idx)
        if depth is None:
            # 如果 DrivingDataset 没有深度图，尝试从文件直接加载
            depth = self._load_depth_from_file(scene_id, source_idx)
        source_depths.append(depth)
    
    # 5. 组装批次
    batch = {
        'scene_id': torch.tensor([scene_id], dtype=torch.long),
        'source': {
            'image': torch.stack(source_images, dim=0),  # [V, H, W, 3]
            'extrinsics': torch.stack(source_extrinsics, dim=0),  # [V, 4, 4]
            'intrinsics': torch.stack(source_intrinsics, dim=0),  # [V, 3, 3] 或 [V, 4, 4]
            'depth': torch.stack(source_depths, dim=0) if source_depths[0] is not None else None,  # [V, H, W]
            'source_id': torch.tensor(source_indices, dtype=torch.long),
        },
        'target': {
            'image': target_image.unsqueeze(0),  # [1, H, W, 3]
            'extrinsics': target_extrinsic.unsqueeze(0),  # [1, 4, 4]
            'intrinsics': target_intrinsic.unsqueeze(0),  # [1, 3, 3] 或 [1, 4, 4]
            'target_id': target_global_idx,
        }
    }
    
    return batch
```

### 3. Source 图像选择接口

```python
# 伪代码
def _select_source_images(
    self,
    scene_dataset: DrivingDataset,
    target_image_idx: int,
    target_extrinsic: Tensor,
    num_source_image: int,
) -> List[int]:
    """
    根据相机姿态选择最近的 source 图像。
    
    策略：
    1. 计算 target 相机位置（从 extrinsic 提取）
    2. 遍历场景中所有其他图像，计算相机位置
    3. 计算欧氏距离
    4. 排除 target 图像本身
    5. 选择距离最近的 num_source_image 张图像
    """
    
    # 1. 提取 target 相机位置
    target_cam_pos = target_extrinsic[:3, 3]  # [3]
    
    # 2. 获取场景中所有图像的相机位置
    all_cam_positions = []
    all_image_indices = []
    
    num_images = scene_dataset.pixel_source.num_imgs
    for img_idx in range(num_images):
        if img_idx == target_image_idx:
            continue  # 排除 target 本身
        
        # 使用 pixel_source.get_image() 方法
        _, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
        cam_pos = cam_infos['camera_to_world'][:3, 3]  # 提取相机位置
        all_cam_positions.append(cam_pos)
        all_image_indices.append(img_idx)
    
    if len(all_cam_positions) == 0:
        # 如果没有其他图像，返回空列表（或重复 target）
        return []
    
    # 3. 计算距离
    all_cam_positions = torch.stack(all_cam_positions, dim=0)  # [N, 3]
    distances = torch.norm(
        all_cam_positions - target_cam_pos.unsqueeze(0),
        dim=1
    )  # [N]
    
    # 4. 选择最近的 num_source_image 张图像
    _, nearest_indices = torch.topk(
        distances,
        k=min(num_source_image, len(all_image_indices)),
        largest=False
    )
    
    selected_indices = [all_image_indices[i] for i in nearest_indices.cpu().numpy()]
    
    return selected_indices
```

### 4. 点云加载接口

```python
# 伪代码
def _find_point_cloud_path(
    self,
    scene_dir: str,
    pcd_dir_name: str,
) -> Optional[str]:
    """
    查找点云文件路径。
    
    搜索顺序：
    1. scene_dir/ply/{pcd_dir_name}/0.ply
    2. scene_dir/{pcd_dir_name}/0.ply
    3. scene_dir/ply/0.ply
    """
    
    possible_paths = [
        os.path.join(scene_dir, 'ply', pcd_dir_name, '0.ply'),
        os.path.join(scene_dir, pcd_dir_name, '0.ply'),
        os.path.join(scene_dir, 'ply', '0.ply'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def _load_point_cloud(self, pcd_path: str) -> Dict[str, Tensor]:
    """
    从 PLY 文件加载点云。
    
    Returns:
        Dict包含：
            - 'points3D_xyz': Tensor[N, 3]
            - 'points3D_rgb': Tensor[N, 3] (0-255)
    """
    
    import open3d as o3d
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        return None
    
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    
    # 如果颜色在 [0, 1] 范围，转换为 [0, 255]
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    
    return {
        'points3D_xyz': torch.from_numpy(points),
        'points3D_rgb': torch.from_numpy(colors),
    }
```

### 5. 随机采样接口

```python
# 伪代码
def sample_random_batch(self) -> Dict:
    """
    随机采样一个批次。
    
    流程：
    1. 随机选择场景ID
    2. 从该场景中随机选择 target 图像索引
    3. 调用 get_batch() 获取批次
    """
    
    # 随机选择场景
    scene_id = torch.randint(0, self.num_scenes, (1,)).item()
    
    # 随机选择 target 图像
    num_images = self.num_images_per_scene[scene_id]
    target_image_idx = torch.randint(0, num_images, (1,)).item()
    
    # 获取批次
    batch = self.get_batch(scene_id, target_image_idx)
    
    return batch
```

---

## 与 DrivingDataset 的集成

### 复用 DrivingDataset 的方法

`EvolSplatDataset` 通过以下方式复用 `DrivingDataset` 的功能：

1. **场景数据加载**: 每个场景使用独立的 `DrivingDataset` 实例
2. **图像访问**: 通过 `scene_dataset.pixel_source` 访问图像和相机参数
3. **深度图访问**: 如果 `DrivingDataset` 支持，直接使用其深度图加载方法
4. **相机参数**: 使用 `DrivingDataset` 的相机参数（内参、外参）

### 需要扩展的接口

根据代码分析，`DrivingDataset` 和 `ScenePixelSource` 提供以下接口：

1. **`pixel_source.get_image(img_idx)`**: 返回 `(image_infos, cam_infos)` 元组
   - `image_infos`: 包含 `pixels` (图像), `origins`, `viewdirs` 等
   - `cam_infos`: 包含 `camera_to_world`, `intrinsics`, `height`, `width` 等
2. **`pixel_source.parse_img_idx(img_idx)`**: 解析图像索引为 `(cam_idx, frame_idx)`
3. **`pixel_source.camera_data[cam_id].get_image(frame_idx)`**: 从特定相机获取图像
4. **深度图**: 通过 `camera_data[cam_id].lidar_depth_maps[frame_idx]` 访问

### 辅助方法

```python
# 伪代码
def _get_image_info_from_driving_dataset(
    self,
    scene_dataset: DrivingDataset,
    image_idx: int,
) -> Dict:
    """
    从 DrivingDataset 获取图像信息。
    
    使用 pixel_source.get_image() 方法，该方法返回 (image_infos, cam_infos)。
    """
    
    pixel_source = scene_dataset.pixel_source
    
    # 使用 pixel_source.get_image() 方法
    image_infos, cam_infos = pixel_source.get_image(image_idx)
    
    # 提取所需信息
    image = image_infos['pixels']  # Tensor[H, W, 3]
    extrinsic = cam_infos['camera_to_world']  # Tensor[4, 4]
    intrinsic = cam_infos['intrinsics']  # Tensor[3, 3]
    
    return {
        'image': image,
        'extrinsic': extrinsic,
        'intrinsic': intrinsic,
        'height': cam_infos['height'],
        'width': cam_infos['width'],
    }

def _get_depth_from_driving_dataset(
    self,
    scene_dataset: DrivingDataset,
    image_idx: int,
) -> Optional[Tensor]:
    """
    从 DrivingDataset 获取深度图。
    
    通过 pixel_source.parse_img_idx() 解析图像索引，
    然后从 camera_data 获取深度图。
    """
    
    pixel_source = scene_dataset.pixel_source
    
    # 解析图像索引
    cam_idx, frame_idx = pixel_source.parse_img_idx(image_idx)
    
    # 找到对应的 cam_id
    cam_id = None
    for cid in pixel_source.camera_list:
        if pixel_source.camera_data[cid].unique_cam_idx == cam_idx:
            cam_id = cid
            break
    
    if cam_id is None:
        return None
    
    # 获取深度图
    camera_data = pixel_source.camera_data[cam_id]
    if camera_data.lidar_depth_maps is not None:
        depth = camera_data.lidar_depth_maps[frame_idx]  # Tensor[H, W]
        return depth
    
    return None

def _load_depth_from_file(
    self,
    scene_id: int,
    image_idx: int,
) -> Optional[Tensor]:
    """
    从文件直接加载深度图。
    
    根据 Drivestudio 场景文件夹结构：
    - depth/{frame_idx:03d}_{cam_id}.npy
    """
    
    scene_dir = self.scene_dirs[scene_id]
    scene_dataset = self.scene_datasets[scene_id]
    
    # 使用 pixel_source.parse_img_idx() 解析图像索引
    cam_idx, frame_idx = scene_dataset.pixel_source.parse_img_idx(image_idx)
    
    # 找到对应的 cam_id（在 camera_list 中的索引）
    cam_id = None
    for cid in scene_dataset.pixel_source.camera_list:
        if scene_dataset.pixel_source.camera_data[cid].unique_cam_idx == cam_idx:
            cam_id = cid
            break
    
    if cam_id is None:
        return None
    
    depth_file = os.path.join(
        scene_dir,
        'depth',
        f'{frame_idx:03d}_{cam_id}.npy'
    )
    
    if not os.path.exists(depth_file):
        return None
    
    # 加载深度图
    depth = np.load(depth_file)
    
    # 如果需要预处理（插值到原始尺寸），使用 depth_utils
    # 获取目标尺寸
    _, cam_infos = scene_dataset.pixel_source.get_image(image_idx)
    H, W = cam_infos['height'].item(), cam_infos['width'].item()
    
    # 如果深度图尺寸不匹配，使用 depth_utils 进行插值
    if depth.shape != (H, W):
        try:
            from depth_utils import process_depth_for_use
            depth, _ = process_depth_for_use(depth_file, target_shape=(H, W))
        except ImportError:
            # 如果 depth_utils 不可用，使用简单的插值
            import cv2
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    
    depth = torch.from_numpy(depth).float()
    
    return depth
```

---

## 数据格式转换

### EVolSplat 批次格式

```python
batch = {
    'scene_id': Tensor[1],  # 场景ID
    'source': {
        'image': Tensor[V, H, W, 3],      # V 张 source 图像
        'extrinsics': Tensor[V, 4, 4],    # source 相机外参
        'intrinsics': Tensor[V, 4, 4],    # source 相机内参（需要转换为4x4）
        'depth': Tensor[V, H, W],         # source 深度图
        'source_id': Tensor[V]            # source 图像全局索引
    },
    'target': {
        'image': Tensor[1, H, W, 3],      # target 图像
        'extrinsics': Tensor[1, 4, 4],     # target 相机外参
        'intrinsics': Tensor[1, 4, 4],    # target 相机内参（需要转换为4x4）
        'target_id': int                  # target 图像全局索引
    }
}
```

### 内参格式转换

`DrivingDataset` 可能使用 3x3 内参矩阵，而 EVolSplat 需要 4x4 格式：

```python
# 伪代码
def _convert_intrinsic_to_4x4(self, intrinsic: Tensor) -> Tensor:
    """
    将 3x3 内参矩阵转换为 4x4 格式。
    
    Args:
        intrinsic: Tensor[3, 3] 或 Tensor[4, 4]
        
    Returns:
        Tensor[4, 4]
    """
    
    if intrinsic.shape == (4, 4):
        return intrinsic
    
    assert intrinsic.shape == (3, 3), f"Unexpected intrinsic shape: {intrinsic.shape}"
    
    intrinsic_4x4 = torch.eye(4, dtype=intrinsic.dtype, device=intrinsic.device)
    intrinsic_4x4[:3, :3] = intrinsic
    
    return intrinsic_4x4
```

---

## 反直觉检查清单

### 1. 索引映射问题

**问题**: 全局图像索引与场景内索引的转换可能出错。

**检查**:
- 验证 `scene_start_indices` 的计算是否正确
- 验证 `target_global_idx = scene_start_indices[scene_id] + target_image_idx` 是否正确
- 检查场景边界是否处理正确

**示例**:
```python
# 场景0有100张图像，场景1有150张图像
# scene_start_indices = [0, 100, 250]
# 场景1的第50张图像的全局索引应该是 100 + 50 = 150
```

### 2. 相机位置计算问题

**问题**: 从外参矩阵提取相机位置时，可能使用了错误的列。

**检查**:
- 验证 `target_cam_pos = target_extrinsic[:3, 3]` 是否正确
- 检查坐标系是否一致（OpenCV vs OpenGL）
- 验证距离计算是否使用欧氏距离

### 3. Source 图像选择问题

**问题**: 选择的 source 图像可能不是最近的，或者数量不足。

**检查**:
- 验证 `_select_source_images()` 是否正确排除 target 图像
- 检查当可用图像数量少于 `num_source_image` 时的处理
- 验证距离计算的正确性

### 4. 深度图加载问题

**问题**: 深度图可能不存在或尺寸不匹配。

**检查**:
- 验证深度图文件路径的构建是否正确
- 检查深度图尺寸是否与图像尺寸匹配
- 验证深度图是否需要预处理（插值）

### 5. 点云加载问题

**问题**: 点云文件可能不存在或格式不正确。

**检查**:
- 验证点云文件路径的搜索顺序
- 检查点云颜色格式（[0, 1] vs [0, 255]）
- 验证点云坐标系统是否与场景一致

### 6. 批次格式问题

**问题**: 批次格式可能不符合 EVolSplat 的要求。

**检查**:
- 验证所有张量的维度是否正确
- 检查内参是否转换为 4x4 格式
- 验证数据类型（float32, int64等）

### 7. 设备问题

**问题**: 张量可能在不同设备上（CPU vs GPU）。

**检查**:
- 确保所有张量都在正确的设备上
- 验证 `device` 参数是否正确传递

### 8. 场景数量问题

**问题**: 场景数量可能为0或索引越界。

**检查**:
- 验证 `num_scenes > 0`
- 检查 `scene_id` 是否在有效范围内
- 验证每个场景的图像数量是否 > 0

### 9. parse_img_idx 和 unique_cam_idx 的使用问题

**问题**: `parse_img_idx()` 返回的是 `unique_cam_idx`，不是 `cam_id`，需要正确映射。

**检查**:
- 验证 `parse_img_idx()` 的返回值：`(unique_cam_idx, frame_idx)`
- 验证 `unique_cam_idx` 与 `cam_id` 的映射关系
- 检查 `camera_list` 中的 `cam_id` 与 `unique_cam_idx` 的对应关系

**示例**:
```python
# pixel_source.camera_list = [0, 1, 2, 3, 4, 5]  # cam_id 列表
# pixel_source.camera_data[0].unique_cam_idx = 0
# pixel_source.camera_data[1].unique_cam_idx = 1
# ...

# 当 parse_img_idx(10) 返回 (4, 1) 时：
# - unique_cam_idx = 4
# - frame_idx = 1
# - 需要找到 cam_id，使得 camera_data[cam_id].unique_cam_idx == 4
# - 在这个例子中，cam_id = 4
```

### 10. 图像索引与帧索引的混淆

**问题**: `image_idx` 是全局图像索引（所有相机的图像），而 `frame_idx` 是帧索引（时间步）。

**检查**:
- 理解 `num_imgs = num_cams * num_frames`
- 验证 `image_idx = frame_idx * num_cams + cam_idx` 的关系
- 检查在计算 source 图像时是否混淆了图像索引和帧索引

### 11. 深度图尺寸与图像尺寸的匹配

**问题**: 深度图可能不是原始图像尺寸，需要插值。

**检查**:
- 验证深度图尺寸是否与图像尺寸匹配
- 如果使用 `depth_utils.process_depth_for_use()`，确保正确传递目标尺寸
- 检查深度图元数据文件（`_meta.npz`）是否包含原始尺寸信息

### 12. 坐标系一致性

**问题**: 不同组件可能使用不同的坐标系（OpenCV vs OpenGL）。

**检查**:
- 验证 `camera_to_world` 矩阵的坐标系约定
- 检查点云坐标系统是否与场景坐标系统一致
- 验证深度图反投影时的坐标系转换是否正确

---

## 使用示例

```python
# 1. 准备场景目录列表
scene_dirs = [
    "/path/to/trainval/000",
    "/path/to/trainval/001",
    "/path/to/trainval/002",
    # ... 更多场景
]

# 2. 准备配置模板
data_cfg_template = OmegaConf.create({
    "dataset": "NuScenes",
    "data_root": "/path/to/trainval",
    "pixel_source": {
        "type": "datasets.nuscenes.nuscenes_sourceloader.NuScenesPixelSource",
        # ... 其他配置
    },
    "lidar_source": {
        "load_lidar": False,  # EVolSplat 不需要 LiDAR
    },
})

# 3. 创建数据集
dataset = EvolSplatDataset(
    scene_dirs=scene_dirs,
    data_cfg_template=data_cfg_template,
    num_source_image=3,
    train_cameras_sampling_strategy="random",
    load_3D_points=True,
    pcd_dir_name="Drop50",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# 4. 获取随机批次
batch = dataset.sample_random_batch()

# 5. 获取指定场景的批次
batch = dataset.get_batch(scene_id=0, target_image_idx=10)

# 6. 获取点云
point_cloud = dataset.get_point_cloud(scene_id=0)
```

---

## 总结

`EvolSplatDataset` 的设计遵循以下原则：

1. **复用优先**: 尽量使用 `DrivingDataset` 的现有功能，避免重复实现
2. **多场景管理**: 通过场景ID索引管理多个场景
3. **格式兼容**: 输出符合 EVolSplat 要求的数据格式
4. **灵活扩展**: 支持不同的采样策略和配置选项

该设计允许在不修改 `DrivingDataset` 核心代码的情况下，实现多场景 feed-forward 训练的数据加载。

