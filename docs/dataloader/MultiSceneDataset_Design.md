# MultiSceneDataset 设计文档

## 概述

本文档设计 `MultiSceneDataset` 类，用于在 Drivestudio 中实现 EVolSplat 的 feed-forward 3DGS 训练，支持静动态解耦和动态物体处理。该类基于 Drivestudio 的场景文件夹结构和 `SceneDataset` 接口，实现多场景、多段、关键帧机制的数据加载。

---

## 核心概念

### 1. 数据层次结构

```
场景 (Scene)
  └── 段 (Segment) - 按照场景 AABB 限制分割，包含多个关键帧
      └── 关键帧 (Keyframe) - 按照距离分割的小段，至少包含一帧
          └── 帧 (Frame) - 时间步，包含多张图像（多相机）
              └── 图像 (Image) - 单张图像，由 (frame_idx, cam_id) 标识
```

**关系说明**：
- **场景**：一个完整的驾驶场景，对应一个场景文件夹（如 `/path/to/trainval/000`）
- **段**：场景按照 AABB 限制分割成的子区域，用于构建独立的 3DGS 场景
- **关键帧**：按照轨迹距离分割的小段，一段关键帧至少包含一帧
- **帧**：时间步，同一时刻所有相机的图像集合
- **图像**：单张图像，由 `(frame_idx, cam_id)` 唯一标识

### 2. 索引系统

**图像索引 (img_idx)**：
- 全局图像索引：`img_idx = frame_idx * num_cams + cam_idx`
- 用于 `pixel_source.get_image(img_idx)` 访问图像

**帧索引 (frame_idx)**：
- 时间步索引，范围 `[0, num_frames)`
- 同一帧的所有相机图像共享相同的 `frame_idx`

**相机索引 (cam_idx)**：
- 相机在 `camera_list` 中的索引
- 范围 `[0, num_cams)`

### 3. Source 和 Target 的定义

**Source**：
- 用于特征提取的图像集合
- 每次使用 3 个关键帧，每个关键帧选择 1 帧
- 共 3 帧 × 6 相机 = 18 张图像

**Target**：
- 用于监督学习的图像集合
- 包含 source 的 3 帧 + 另外 3 个关键帧（每个关键帧选择 1 帧）
- 共 6 帧 × 6 相机 = 36 张图像
- **注意**：Target 包含 source，不要求独立

---

## 类设计

### MultiSceneDataset

```python
class MultiSceneDataset:
    """
    多场景数据集类，用于 EVolSplat feed-forward 3DGS 训练。
    
    核心功能：
    1. 管理多个场景，支持训练/评估场景划分
    2. 基于关键帧机制分割场景为多个段
    3. 在段内随机选择 source 和 target 关键帧
    4. 打包数据为 EVolSplat 格式的 batch
    """
    
    def __init__(
        self,
        data_cfg: OmegaConf,
        train_scene_ids: List[int],
        eval_scene_ids: List[int],
        num_source_keyframes: int = 3,
        num_target_keyframes: int = 6,  # 包含 source 的 3 帧 + 另外 3 帧
        segment_overlap_ratio: float = 0.2,  # 段与段之间的重叠比例
        keyframe_split_config: Dict = None,  # 关键帧分割配置
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            data_cfg: Drivestudio 数据配置（OmegaConf）
            train_scene_ids: 训练场景ID列表
            eval_scene_ids: 评估场景ID列表
            num_source_keyframes: Source 使用的关键帧数量（默认3）
            num_target_keyframes: Target 使用的关键帧数量（默认6，包含source）
            segment_overlap_ratio: 段与段之间的重叠比例（默认0.2）
            keyframe_split_config: 关键帧分割配置
                - num_splits: 关键帧分割数量（0表示自动）
                - min_count: 每个关键帧段的最小帧数（默认1）
                - min_length: 每个关键帧段的最小长度（默认0）
            min_keyframes_per_scene: 场景的最小关键帧数量，不满足则跳过（默认10）
            min_keyframes_per_segment: 段的最小关键帧数量，不满足则跳过（默认6）
            device: 设备（默认CPU）
        """
        pass
    
    def get_scene(self, scene_id: int) -> Dict:
        """
        获取指定场景的数据和信息。
        
        Args:
            scene_id: 场景ID（全局索引，不是场景文件夹名称）
            
        Returns:
            Dict包含：
                - 'dataset': DrivingDataset 实例
                - 'segments': List[Dict] - 段信息列表
                - 'keyframes': List[List[int]] - 每个段的关键帧列表
                - 'num_frames': int - 场景总帧数
                - 'num_cams': int - 场景相机数量
        """
        pass
    
    def get_segment_batch(
        self,
        scene_id: int,
        segment_id: int,
    ) -> Dict:
        """
        获取指定场景和段的训练批次。
        
        Args:
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            Dict包含：
                - 'scene_id': Tensor[1] - 场景ID
                - 'segment_id': int - 段ID
                - 'source': {
                    'image': Tensor[18, H, W, 3],  # 3帧 × 6相机
                    'extrinsics': Tensor[18, 4, 4],
                    'intrinsics': Tensor[18, 4, 4],
                    'depth': Tensor[18, H, W],
                    'frame_indices': Tensor[18],  # 帧索引
                    'cam_indices': Tensor[18],    # 相机索引
                }
                - 'target': {
                    'image': Tensor[36, H, W, 3],  # 6帧 × 6相机
                    'extrinsics': Tensor[36, 4, 4],
                    'intrinsics': Tensor[36, 4, 4],
                    'depth': Tensor[36, H, W],
                    'frame_indices': Tensor[36],
                    'cam_indices': Tensor[36],
                }
        """
        pass
    
    def sample_random_batch(self) -> Dict:
        """
        随机采样一个训练批次。
        
        Returns:
            与 get_segment_batch() 相同的格式
        """
        pass
```

---

## 实现细节

### 1. 初始化流程

```python
def __init__(self, ...):
    # 1. 存储配置
    self.data_cfg = data_cfg
    self.train_scene_ids = train_scene_ids
    self.eval_scene_ids = eval_scene_ids
    self.num_source_keyframes = num_source_keyframes
    self.num_target_keyframes = num_target_keyframes
    self.segment_overlap_ratio = segment_overlap_ratio
    self.device = device
    
    # 2. 初始化关键帧分割配置
    self.keyframe_split_config = keyframe_split_config or {
        'num_splits': 0,  # 自动确定
        'min_count': 1,
        'min_length': 0.0,
    }
    self.min_keyframes_per_scene = min_keyframes_per_scene
    self.min_keyframes_per_segment = min_keyframes_per_segment
    
    # 3. 加载所有训练场景（跳过不适合的场景）
    self.train_scenes = {}
    for scene_id in train_scene_ids:
        scene_data = self._load_scene(scene_id)
        if scene_data is not None:  # 场景适合训练
            self.train_scenes[scene_id] = scene_data
        else:
            logger.warning(f"Skipping training scene {scene_id} (not suitable)")
    
    # 4. 加载所有评估场景（跳过不适合的场景）
    self.eval_scenes = {}
    for scene_id in eval_scene_ids:
        scene_data = self._load_scene(scene_id)
        if scene_data is not None:  # 场景适合评估
            self.eval_scenes[scene_id] = scene_data
        else:
            logger.warning(f"Skipping eval scene {scene_id} (not suitable)")
    
    # 5. 构建场景到段的映射
    self._build_segment_mapping()
```

### 2. 场景加载

```python
def _load_scene(self, scene_id: int) -> Dict:
    """
    加载单个场景的数据。
    
    流程：
    1. 创建 DrivingDataset 实例
    2. 获取场景的轨迹（用于关键帧分割）
    3. 分割关键帧
    4. 分割段（基于 AABB 限制）
    5. 返回场景信息
    """
    # 1. 创建场景配置
    scene_cfg = OmegaConf.create(OmegaConf.to_container(self.data_cfg))
    scene_cfg.scene_idx = scene_id
    
    # 2. 创建 DrivingDataset 实例
    scene_dataset = DrivingDataset(scene_cfg)
    
    # 3. 获取场景轨迹（使用前相机的轨迹）
    trajectory = self._get_scene_trajectory(scene_dataset)
    
    # 4. 分割关键帧
    keyframe_segments, keyframe_ranges = self._split_keyframes(trajectory)
    
    # 5. 检查场景是否适合训练（关键帧数量是否足够）
    if not self._is_scene_suitable(keyframe_segments):
        logger.warning(f"Scene {scene_id} is not suitable for training (insufficient keyframes), skipping...")
        return None  # 返回 None 表示场景不适合
    
    # 6. 分割段（基于 AABB 限制和关键帧距离）
    segments = self._split_segments(
        scene_dataset=scene_dataset,
        keyframe_segments=keyframe_segments,
        keyframe_ranges=keyframe_ranges,
        overlap_ratio=self.segment_overlap_ratio,
    )
    
    if len(segments) == 0:
        logger.warning(f"Scene {scene_id} has no valid segments after filtering, skipping...")
        return None
    
    return {
        'dataset': scene_dataset,
        'trajectory': trajectory,
        'keyframe_segments': keyframe_segments,
        'keyframe_ranges': keyframe_ranges,
        'segments': segments,
        'num_frames': scene_dataset.num_img_timesteps,
        'num_cams': scene_dataset.num_cams,
    }

def _get_scene_trajectory(self, scene_dataset: DrivingDataset) -> Tensor:
    """
    获取场景的轨迹（相机变换矩阵）。
    
    使用 DrivingDataset 的 get_novel_render_traj 方法获取前相机的轨迹。
    
    Returns:
        trajectory: Tensor[num_frames, 4, 4] - 相机变换矩阵
    """
    # 使用 DrivingDataset 的 get_novel_render_traj 方法
    # 获取前相机的轨迹（"front_center_interp"）
    num_frames = scene_dataset.num_img_timesteps
    traj_dict = scene_dataset.get_novel_render_traj(["front_center_interp"], num_frames)
    trajectory = traj_dict["front_center_interp"]  # Tensor[num_frames, 4, 4]
    
    return trajectory

def _is_scene_suitable(
    self,
    keyframe_segments: List[List[int]],
) -> bool:
    """
    检查场景是否适合训练。
    
    判断标准：
    - 关键帧数量是否足够（>= min_keyframes_per_scene）
    
    Args:
        keyframe_segments: 关键帧段列表
        
    Returns:
        bool: True 表示场景适合，False 表示不适合
    """
    num_keyframes = len(keyframe_segments)
    
    if num_keyframes < self.min_keyframes_per_scene:
        return False
    
    return True
```

### 3. 关键帧分割

```python
def _split_keyframes(
    self,
    trajectory: Tensor,  # [num_frames, 4, 4]
) -> Tuple[List[List[int]], Tensor]:
    """
    按照距离分割关键帧。
    
    使用用户提供的 split_trajectory 函数。
    
    Returns:
        keyframe_segments: List[List[int]] - 每个关键帧段包含的帧索引列表
        keyframe_ranges: Tensor[num_keyframes, 2] - 每个关键帧段的距离范围
    """
    keyframe_segments, keyframe_ranges = split_trajectory(
        trajectory=trajectory,
        num_splits=self.keyframe_split_config['num_splits'],
        min_count=self.keyframe_split_config['min_count'],
        min_length=self.keyframe_split_config['min_length'],
    )
    
    return keyframe_segments, keyframe_ranges
```

### 4. 段分割

```python
def _split_segments(
    self,
    scene_dataset: DrivingDataset,
    keyframe_segments: List[List[int]],
    keyframe_ranges: Tensor,  # [num_keyframes, 2] - 每个关键帧段的距离范围
    overlap_ratio: float,
) -> List[Dict]:
    """
    按照场景 AABB 限制分割段。
    
    策略：
    1. 获取场景的 AABB 和轨迹
    2. 计算关键帧的合计距离
    3. 将关键帧按照距离和 AABB 长度分组为段
    4. 过滤掉关键帧数量不足的段
    
    注意：
    - 段分割不需要那么精确，关键帧的合计距离对比AABB的长度即可
    - 段内设置一个最低关键帧数量限制，不满足则跳过
    - 段与段可以部分重合（overlap_ratio）
    
    Args:
        scene_dataset: 场景数据集
        keyframe_segments: 关键帧段列表
        keyframe_ranges: 关键帧段的距离范围 [num_keyframes, 2]
        overlap_ratio: 段与段之间的重叠比例
    
    Returns:
        segments: List[Dict] - 每个段包含：
            - 'segment_id': int - 段ID
            - 'keyframe_indices': List[int] - 该段包含的关键帧索引（全局关键帧索引）
            - 'frame_indices': List[int] - 该段包含的所有帧索引（去重后的帧索引列表）
            - 'aabb': Tensor[2, 3] - 段的 AABB 边界（使用场景 AABB）
    """
    # 1. 获取场景 AABB
    scene_aabb = scene_dataset.get_aabb()  # [2, 3]
    scene_size = scene_aabb[1] - scene_aabb[0]  # [3]
    
    # 2. 计算场景 AABB 的主要方向长度（通常是 x 或 y 方向，取决于行驶方向）
    # 使用最大的维度作为主要长度
    aabb_length = scene_size.max().item()  # 标量
    
    # 3. 计算关键帧的合计距离
    # keyframe_ranges 的每一行是 [start_distance, end_distance]
    # 计算每个关键帧段的长度
    keyframe_lengths = keyframe_ranges[:, 1] - keyframe_ranges[:, 0]  # [num_keyframes]
    total_keyframe_distance = keyframe_lengths.sum().item()  # 所有关键帧段的合计距离
    
    # 4. 根据距离和 AABB 长度确定段的数量
    # 如果关键帧合计距离远小于 AABB 长度，说明车辆移动距离短，可能只有1个段
    # 如果关键帧合计距离接近 AABB 长度，可以分成多个段
    if total_keyframe_distance < aabb_length * 0.3:
        # 车辆移动距离短，只创建一个段
        num_segments = 1
    else:
        # 根据关键帧数量和距离确定段数
        # 每个段至少需要 min_keyframes_per_segment 个关键帧
        max_segments = len(keyframe_segments) // self.min_keyframes_per_segment
        num_segments = max(1, min(max_segments, int(total_keyframe_distance / aabb_length * 2)))
    
    # 5. 将关键帧按照距离分组为段
    segments = []
    segment_id = 0
    
    if num_segments == 1:
        # 只有一个段，包含所有关键帧
        all_frames = []
        for kf_seg in keyframe_segments:
            all_frames.extend(kf_seg)
        
        segments.append({
            'segment_id': segment_id,
            'keyframe_indices': list(range(len(keyframe_segments))),
            'frame_indices': sorted(list(set(all_frames))),
            'aabb': scene_aabb,
        })
    else:
        # 多个段，按照累积距离分组
        segment_distance = total_keyframe_distance / num_segments
        overlap_distance = segment_distance * overlap_ratio
        
        current_segment_kf_indices = []
        current_segment_frames = set()
        current_distance = 0.0
        segment_start_distance = 0.0
        
        for kf_idx in range(len(keyframe_segments)):
            kf_length = keyframe_lengths[kf_idx].item()
            kf_center_distance = (keyframe_ranges[kf_idx, 0] + keyframe_ranges[kf_idx, 1]) / 2.0
            
            # 检查是否应该开始新段
            if (len(current_segment_kf_indices) > 0 and 
                kf_center_distance - segment_start_distance > segment_distance + overlap_distance):
                # 当前段的关键帧数量是否足够
                if len(current_segment_kf_indices) >= self.min_keyframes_per_segment:
                    segments.append({
                        'segment_id': segment_id,
                        'keyframe_indices': current_segment_kf_indices.copy(),
                        'frame_indices': sorted(list(current_segment_frames)),
                        'aabb': scene_aabb,  # 使用场景 AABB
                    })
                    segment_id += 1
                
                # 开始新段（考虑重叠）
                segment_start_distance = kf_center_distance - overlap_distance
                current_segment_kf_indices = []
                current_segment_frames = set()
            
            # 将关键帧添加到当前段
            current_segment_kf_indices.append(kf_idx)
            current_segment_frames.update(keyframe_segments[kf_idx])
        
        # 处理最后一个段
        if len(current_segment_kf_indices) >= self.min_keyframes_per_segment:
            segments.append({
                'segment_id': segment_id,
                'keyframe_indices': current_segment_kf_indices,
                'frame_indices': sorted(list(current_segment_frames)),
                'aabb': scene_aabb,
            })
    
    # 6. 过滤掉关键帧数量不足的段（双重检查）
    valid_segments = [
        seg for seg in segments
        if len(seg['keyframe_indices']) >= self.min_keyframes_per_segment
    ]
    
    return valid_segments
```

### 5. Source 和 Target 选择

```python
def _select_source_and_target_keyframes(
    self,
    segment: Dict,
    num_source_keyframes: int,
    num_target_keyframes: int,
) -> Tuple[List[int], List[int]]:
    """
    在段内随机选择 source 和 target 关键帧。
    
    策略：
    1. 随机选择 num_source_keyframes 个关键帧作为 source
    2. 从剩余关键帧中随机选择 (num_target_keyframes - num_source_keyframes) 个作为额外的 target
    3. Target 包含 source 的所有关键帧
    
    Returns:
        source_keyframe_indices: List[int] - Source 关键帧索引列表
        target_keyframe_indices: List[int] - Target 关键帧索引列表（包含 source）
    """
    available_keyframes = segment['keyframe_indices']
    
    if len(available_keyframes) < num_source_keyframes:
        # 如果可用关键帧不足，重复使用
        source_keyframe_indices = available_keyframes * (num_source_keyframes // len(available_keyframes) + 1)
        source_keyframe_indices = source_keyframe_indices[:num_source_keyframes]
    else:
        # 随机选择 source 关键帧
        source_keyframe_indices = random.sample(available_keyframes, num_source_keyframes)
    
    # 计算需要额外选择的 target 关键帧数量
    num_extra_target_keyframes = num_target_keyframes - num_source_keyframes
    
    # 从剩余关键帧中选择额外的 target 关键帧
    remaining_keyframes = [kf for kf in available_keyframes if kf not in source_keyframe_indices]
    
    if len(remaining_keyframes) < num_extra_target_keyframes:
        # 如果剩余关键帧不足，重复使用
        extra_target_keyframes = remaining_keyframes * (num_extra_target_keyframes // len(remaining_keyframes) + 1)
        extra_target_keyframes = extra_target_keyframes[:num_extra_target_keyframes]
    else:
        # 随机选择额外的 target 关键帧
        extra_target_keyframes = random.sample(remaining_keyframes, num_extra_target_keyframes)
    
    # Target 包含 source 的所有关键帧
    target_keyframe_indices = source_keyframe_indices + extra_target_keyframes
    
    return source_keyframe_indices, target_keyframe_indices
```

### 6. 从关键帧选择帧

```python
def _select_frame_from_keyframe(
    self,
    keyframe_segment: List[int],  # 关键帧段包含的帧索引列表
) -> int:
    """
    从关键帧段中随机选择一帧。
    
    Args:
        keyframe_segment: 关键帧段包含的帧索引列表
        
    Returns:
        frame_idx: 选中的帧索引
    """
    if len(keyframe_segment) == 0:
        raise ValueError("Keyframe segment is empty")
    
    # 随机选择一帧
    frame_idx = random.choice(keyframe_segment)
    
    return frame_idx
```

### 7. 批次打包

```python
def get_segment_batch(
    self,
    scene_id: int,
    segment_id: int,
) -> Dict:
    """
    获取指定场景和段的训练批次。
    """
    # 1. 获取场景和段信息
    scene_data = self.train_scenes[scene_id]
    segment = scene_data['segments'][segment_id]
    scene_dataset = scene_data['dataset']
    
    # 2. 选择 source 和 target 关键帧
    source_keyframe_indices, target_keyframe_indices = self._select_source_and_target_keyframes(
        segment=segment,
        num_source_keyframes=self.num_source_keyframes,
        num_target_keyframes=self.num_target_keyframes,
    )
    
    # 3. 从每个关键帧中选择一帧
    source_frame_indices = []
    for kf_idx in source_keyframe_indices:
        keyframe_segment = scene_data['keyframe_segments'][kf_idx]
        frame_idx = self._select_frame_from_keyframe(keyframe_segment)
        source_frame_indices.append(frame_idx)
    
    target_frame_indices = []
    for kf_idx in target_keyframe_indices:
        keyframe_segment = scene_data['keyframe_segments'][kf_idx]
        frame_idx = self._select_frame_from_keyframe(keyframe_segment)
        target_frame_indices.append(frame_idx)
    
    # 4. 加载 source 图像（3帧 × 6相机 = 18张）
    source_images = []
    source_extrinsics = []
    source_intrinsics = []
    source_depths = []
    source_frame_idxs = []
    source_cam_idxs = []
    
    for frame_idx in source_frame_indices:
        for cam_idx in range(scene_dataset.num_cams):
            img_idx = frame_idx * scene_dataset.num_cams + cam_idx
            image_infos, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
            
            source_images.append(image_infos['pixels'])  # [H, W, 3]
            source_extrinsics.append(cam_infos['camera_to_world'])  # [4, 4]
            
            # 转换内参为 4x4
            intrinsic_3x3 = cam_infos['intrinsics']  # [3, 3]
            intrinsic_4x4 = self._convert_intrinsic_to_4x4(intrinsic_3x3)
            source_intrinsics.append(intrinsic_4x4)
            
            # 获取深度图
            depth = self._get_depth(scene_dataset, frame_idx, cam_idx)
            if depth is None:
                # 如果深度图不存在，创建占位符
                H, W = image_infos['pixels'].shape[:2]
                depth = torch.ones(H, W, dtype=torch.float32, device=self.device) * 10.0
            source_depths.append(depth)
            
            source_frame_idxs.append(frame_idx)
            source_cam_idxs.append(cam_idx)
    
    # 5. 加载 target 图像（6帧 × 6相机 = 36张）
    target_images = []
    target_extrinsics = []
    target_intrinsics = []
    target_depths = []
    target_frame_idxs = []
    target_cam_idxs = []
    
    for frame_idx in target_frame_indices:
        for cam_idx in range(scene_dataset.num_cams):
            img_idx = frame_idx * scene_dataset.num_cams + cam_idx
            image_infos, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
            
            target_images.append(image_infos['pixels'])
            target_extrinsics.append(cam_infos['camera_to_world'])
            
            intrinsic_3x3 = cam_infos['intrinsics']
            intrinsic_4x4 = self._convert_intrinsic_to_4x4(intrinsic_3x3)
            target_intrinsics.append(intrinsic_4x4)
            
            depth = self._get_depth(scene_dataset, frame_idx, cam_idx)
            if depth is None:
                # 如果深度图不存在，创建占位符
                H, W = image_infos['pixels'].shape[:2]
                depth = torch.ones(H, W, dtype=torch.float32, device=self.device) * 10.0
            target_depths.append(depth)
            
            target_frame_idxs.append(frame_idx)
            target_cam_idxs.append(cam_idx)
    
    # 6. 组装批次
    batch = {
        'scene_id': torch.tensor([scene_id], dtype=torch.long),
        'segment_id': segment_id,
        
        'source': {
            'image': torch.stack(source_images, dim=0),  # [18, H, W, 3]
            'extrinsics': torch.stack(source_extrinsics, dim=0),  # [18, 4, 4]
            'intrinsics': torch.stack(source_intrinsics, dim=0),  # [18, 4, 4]
            'depth': torch.stack(source_depths, dim=0),  # [18, H, W]
            'frame_indices': torch.tensor(source_frame_idxs, dtype=torch.long),  # [18]
            'cam_indices': torch.tensor(source_cam_idxs, dtype=torch.long),  # [18]
        },
        
        'target': {
            'image': torch.stack(target_images, dim=0),  # [36, H, W, 3]
            'extrinsics': torch.stack(target_extrinsics, dim=0),  # [36, 4, 4]
            'intrinsics': torch.stack(target_intrinsics, dim=0),  # [36, 4, 4]
            'depth': torch.stack(target_depths, dim=0),  # [36, H, W]
            'frame_indices': torch.tensor(target_frame_idxs, dtype=torch.long),  # [36]
            'cam_indices': torch.tensor(target_cam_idxs, dtype=torch.long),  # [36]
        }
    }
    
    return batch

def _get_depth(
    self,
    scene_dataset: DrivingDataset,
    frame_idx: int,
    cam_idx: int,
) -> Optional[Tensor]:
    """
    获取指定帧和相机的深度图。
    
    优先从 DrivingDataset 的 lidar_depth_maps 获取，
    如果不存在，尝试从文件直接加载。
    
    Returns:
        depth: Tensor[H, W] 或 None
    """
    # 方法1：从 DrivingDataset 的 lidar_depth_maps 获取
    try:
        pixel_source = scene_dataset.pixel_source
        cam_id = pixel_source.camera_list[cam_idx]
        camera_data = pixel_source.camera_data[cam_id]
        
        if camera_data.lidar_depth_maps is not None:
            depth = camera_data.lidar_depth_maps[frame_idx]  # Tensor[H, W]
            return depth
    except (IndexError, KeyError, AttributeError):
        pass
    
    # 方法2：从文件直接加载（根据 Drivestudio 场景文件夹结构）
    try:
        scene_dir = scene_dataset.data_path
        depth_file = os.path.join(
            scene_dir,
            'depth',
            f'{frame_idx:03d}_{cam_idx}.npy'
        )
        
        if os.path.exists(depth_file):
            depth = np.load(depth_file)
            
            # 获取目标尺寸
            img_idx = frame_idx * scene_dataset.num_cams + cam_idx
            _, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
            H, W = cam_infos['height'].item(), cam_infos['width'].item()
            
            # 如果深度图尺寸不匹配，进行插值
            if depth.shape != (H, W):
                import cv2
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            
            depth = torch.from_numpy(depth).float().to(self.device)
            return depth
    except (FileNotFoundError, OSError, ValueError):
        pass
    
    return None

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

## 关键帧分割函数

```python
def split_trajectory(trajectory, num_splits=0, min_count=1, min_length=0):
    """
    Split trajectory into segments.
    
    Args:
        trajectory (torch.Tensor): Trajectory tensor of shape [frame_num, 4, 4].
        num_splits (int): Number of splits. If 0, the function will automatically determine the number of splits.
        min_count (int): Minimum number of occurence of each split.
        min_length (float): Minimum length of each split.
        
    Returns:
        segments (list): List of segments, each segment is a list of frame indices.
        ranges (torch.Tensor): Tensor of shape [num_splits, 2], each row is a range [start, end].
    """
    positions = trajectory[:, :3, 3].cpu()
    
    delta_positions = positions[1:] - positions[:-1]  # 相邻帧的位置差，形状为[frame_num - 1, 3]
    distances = torch.norm(delta_positions, dim=1)    # 相邻帧之间的距离，形状为[frame_num - 1]
    cumulative_distances = torch.cat([torch.tensor([0.0], device=distances.device), torch.cumsum(distances, dim=0)])  # 累积距离，形状为[frame_num]
    total_distance = cumulative_distances[-1]
    
    # 初始化区段数量为帧数
    frame_num = positions.shape[0]
    max_segments = frame_num
    
    # 自适应计算最大可行的区段数量
    if num_splits == 0:
        for n in range(max_segments, 0, -1):
            # 计算每个区段的边界距离
            segment_boundaries = torch.linspace(0, total_distance, steps=n + 1)
            
            # 使用bucketize函数确定每帧所属的区段索引
            segment_indices = torch.bucketize(cumulative_distances, segment_boundaries, right=False) - 1
            segment_indices = torch.clamp(segment_indices, min=0, max=n - 1)  # 确保索引在有效范围内
            
            # 统计每个区段的帧数
            counts = torch.bincount(segment_indices, minlength=n)
            
            # 检查是否所有区段都有至少一帧且长度满足最小长度
            segment_lengths = segment_boundaries[1:] - segment_boundaries[:-1]
            if torch.all(counts >= min_count) and torch.all(segment_lengths >= min_length):
                # 找到了最大的n，使得每个区段至少有一帧且长度满足最小长度
                num_splits = n
                break
    
    segment_length = total_distance / num_splits
    
    segment_indices = (cumulative_distances / segment_length).long()
    segment_indices = torch.clamp(segment_indices, max=num_splits-1)
    segment_indices = segment_indices
    segments = [[] for _ in range(num_splits)]
    boundaries = torch.linspace(0, total_distance, steps=num_splits + 1)
    start, end = boundaries[:-1], boundaries[1:]
    ranges = torch.stack([start, end], dim=1)
    for i in range(num_splits):
        indices = torch.where(segment_indices == i)[0].tolist()
        segments[i] = indices
    
    return segments, ranges
```

---

## 反直觉检查清单

### 1. 索引系统检查

- [ ] **图像索引计算正确**：`img_idx = frame_idx * num_cams + cam_idx`
- [ ] **帧索引范围正确**：`frame_idx in [0, num_frames)`
- [ ] **相机索引范围正确**：`cam_idx in [0, num_cams)`
- [ ] **关键帧索引正确**：关键帧索引是段内的索引，不是全局索引
- [ ] **段索引正确**：段索引是场景内的索引，不是全局索引

### 2. Source 和 Target 数量检查

- [ ] **Source 图像数量**：`num_source_keyframes * num_cams = 3 * 6 = 18`
- [ ] **Target 图像数量**：`num_target_keyframes * num_cams = 6 * 6 = 36`
- [ ] **Target 包含 Source**：Target 的关键帧列表包含 Source 的所有关键帧
- [ ] **关键帧选择不重复**：Source 和额外的 Target 关键帧不重复（但 Target 包含 Source）

### 3. 关键帧和段的关系检查

- [ ] **关键帧段至少包含一帧**：每个关键帧段至少有一帧
- [ ] **段包含多个关键帧**：每个段包含多个关键帧
- [ ] **段与段可以重叠**：段与段之间可以部分重合（overlap_ratio）
- [ ] **关键帧分割基于距离**：使用 `split_trajectory` 函数按距离分割
- [ ] **段分割基于 AABB**：段分割考虑场景的 AABB 限制

### 4. 数据加载检查

- [ ] **图像尺寸一致**：所有 source 图像尺寸一致，所有 target 图像尺寸一致
- [ ] **内参格式正确**：内参已转换为 4x4 格式
- [ ] **深度图存在**：所有图像都有对应的深度图
- [ ] **设备一致性**：所有张量在同一设备上
- [ ] **数据类型正确**：图像为 float32，索引为 int64

### 5. 场景和段的关系检查

- [ ] **场景加载正确**：每个场景使用独立的 `DrivingDataset` 实例
- [ ] **段信息完整**：每个段包含关键帧索引、帧索引、AABB 等信息
- [ ] **段索引唯一**：同一场景内的段索引唯一
- [ ] **场景索引正确**：场景索引是全局索引，对应 `train_scene_ids` 或 `eval_scene_ids`

### 6. 随机选择检查

- [ ] **Source 关键帧随机选择**：每次调用 `get_segment_batch` 时随机选择
- [ ] **Target 关键帧随机选择**：从剩余关键帧中随机选择额外的 target 关键帧
- [ ] **帧选择随机**：从关键帧段中随机选择一帧
- [ ] **边界情况处理**：当可用关键帧不足时，正确处理（重复使用或减少数量）

### 7. 批次格式检查

- [ ] **批次格式符合 EVolSplat 要求**：包含 scene_id, source, target
- [ ] **Source 维度正确**：`[18, H, W, 3]` 或 `[num_source_keyframes * num_cams, H, W, 3]`
- [ ] **Target 维度正确**：`[36, H, W, 3]` 或 `[num_target_keyframes * num_cams, H, W, 3]`
- [ ] **包含元数据**：frame_indices, cam_indices 等信息正确

### 8. 轨迹获取检查

- [ ] **轨迹来源正确**：使用前相机的轨迹（或指定相机）
- [ ] **轨迹格式正确**：`[num_frames, 4, 4]` 的变换矩阵
- [ ] **轨迹完整性**：包含所有训练帧的轨迹

### 9. AABB 和段分割检查

- [ ] **AABB 获取正确**：使用 `scene_dataset.get_aabb()` 获取场景 AABB
- [ ] **段分割合理**：段的大小和数量合理
- [ ] **重叠处理正确**：段与段之间的重叠比例正确应用

### 10. 与 EVolSplat Offset 机制的兼容性

- [ ] **场景 ID 传递正确**：batch 中的 scene_id 可以正确索引模型的 offset
- [ ] **段 ID 处理**：段 ID 可能需要映射到场景级别的 offset（如果 offset 是场景级别的）
- [ ] **Offset 共享**：同一场景的不同段可能共享 offset（需要根据实际需求决定）

---

## 使用示例

```python
# 1. 准备配置
data_cfg = OmegaConf.create({
    "dataset": "NuScenes",
    "data_root": "/path/to/trainval",
    "pixel_source": {
        "type": "datasets.nuscenes.nuscenes_sourceloader.NuScenesPixelSource",
        # ... 其他配置
    },
    "lidar_source": {
        "load_lidar": False,
    },
})

# 2. 创建数据集
dataset = MultiSceneDataset(
    data_cfg=data_cfg,
    train_scene_ids=[0, 1, 2, 3, 4],
    eval_scene_ids=[5, 6],
    num_source_keyframes=3,
    num_target_keyframes=6,
    segment_overlap_ratio=0.2,
    keyframe_split_config={
        'num_splits': 0,  # 自动确定
        'min_count': 1,
        'min_length': 0.0,
    },
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# 3. 获取随机批次
batch = dataset.sample_random_batch()

# 4. 获取指定场景和段的批次
batch = dataset.get_segment_batch(scene_id=0, segment_id=2)

# 5. 获取场景信息
scene_info = dataset.get_scene(scene_id=0)
print(f"场景 {scene_id} 有 {len(scene_info['segments'])} 个段")
```

---

## 与 EVolSplat 的集成

### Offset 机制考虑

根据 `EVolSplat_Offset_Training_Mechanism.md`，offset 是场景级别的，每个场景有一个独立的 `self.offset[scene_id]`。

**设计决策**：
- **场景级别的 Offset**：每个场景维护独立的 offset
- **段共享 Offset**：同一场景的不同段共享同一个 offset（因为 offset 是场景级别的）
- **Offset 累积更新**：每次训练迭代都会更新 offset，下次迭代使用更新后的值

**实现建议**：
```python
# 在 trainer 中
# scene_id 从 batch 中获取
scene_id = batch['scene_id'].item()

# 模型内部使用 scene_id 索引 offset
# self.offset[scene_id] 是场景级别的，所有该场景的段共享
```

### Batch 格式适配

EVolSplat 的 `get_outputs` 方法期望的 batch 格式：
```python
batch = {
    'scene_id': Tensor[1],
    'source': {
        'image': Tensor[V, H, W, 3],
        'extrinsics': Tensor[V, 4, 4],
        'intrinsics': Tensor[V, 4, 4],
        'depth': Tensor[V, H, W],
    },
    'target': {
        'image': Tensor[1, H, W, 3],  # 注意：EVolSplat 期望 1 张 target
        'extrinsics': Tensor[1, 4, 4],
        'intrinsics': Tensor[1, 4, 4],
    }
}
```

**适配策略**：

**方案1：在 trainer 中循环处理每张 target 图像（推荐）**
```python
# 在 trainer 中
batch = dataset.get_segment_batch(scene_id, segment_id)

# 对每张 target 图像分别计算损失
for target_idx in range(batch['target']['image'].shape[0]):
    # 创建单张 target 的 batch
    single_target_batch = {
        'scene_id': batch['scene_id'],
        'source': batch['source'],
        'target': {
            'image': batch['target']['image'][target_idx:target_idx+1],  # [1, H, W, 3]
            'extrinsics': batch['target']['extrinsics'][target_idx:target_idx+1],
            'intrinsics': batch['target']['intrinsics'][target_idx:target_idx+1],
        }
    }
    
    # 前向传播和损失计算
    outputs = model.get_outputs(camera, single_target_batch)
    loss = compute_loss(outputs, single_target_batch)
    loss.backward()
```

**方案2：修改 EVolSplat 的 `get_outputs` 方法支持多张 target**
- 需要修改 `evolsplat.py` 中的 `get_outputs` 方法
- 可能需要修改损失计算逻辑
- 不推荐，因为会修改 EVolSplat 的核心代码

**方案3：在数据集层面提供单张 target 的接口**
```python
def get_segment_batch_single_target(
    self,
    scene_id: int,
    segment_id: int,
    target_idx: int = None,  # 如果为 None，随机选择
) -> Dict:
    """
    获取指定场景和段的训练批次，但只包含一张 target 图像。
    
    如果 target_idx 为 None，从 target 关键帧中随机选择一张。
    """
    # 获取完整的 batch
    full_batch = self.get_segment_batch(scene_id, segment_id)
    
    # 选择一张 target 图像
    if target_idx is None:
        target_idx = random.randint(0, full_batch['target']['image'].shape[0] - 1)
    
    # 创建单张 target 的 batch
    batch = {
        'scene_id': full_batch['scene_id'],
        'source': full_batch['source'],
        'target': {
            'image': full_batch['target']['image'][target_idx:target_idx+1],
            'extrinsics': full_batch['target']['extrinsics'][target_idx:target_idx+1],
            'intrinsics': full_batch['target']['intrinsics'][target_idx:target_idx+1],
            'frame_indices': full_batch['target']['frame_indices'][target_idx:target_idx+1],
            'cam_indices': full_batch['target']['cam_indices'][target_idx:target_idx+1],
        }
    }
    
    return batch
```

**推荐使用方案1或方案3**，因为它们不需要修改 EVolSplat 的核心代码。

---

## 总结

`MultiSceneDataset` 的设计遵循以下原则：

1. **层次化数据组织**：场景 → 段 → 关键帧 → 帧 → 图像
2. **关键帧机制**：按照距离分割关键帧，支持动态场景处理
3. **段分割机制**：按照 AABB 限制分割段，支持大场景处理
4. **灵活的 Source/Target 选择**：在段内随机选择关键帧和帧
5. **EVolSplat 兼容**：输出格式符合 EVolSplat 的要求
6. **Drivestudio 集成**：复用 `DrivingDataset` 和 `ScenePixelSource` 的接口

该设计允许在不修改 Drivestudio 核心代码的情况下，实现支持动态物体的 feed-forward 3DGS 训练。

