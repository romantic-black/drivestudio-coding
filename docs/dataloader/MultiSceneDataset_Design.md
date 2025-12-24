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
- 每次使用 `num_source_keyframes` 个关键帧（默认 3），每个关键帧选择 1 帧
- 共 `num_source_keyframes × num_cams` 张图像（相机数量是动态的，从场景数据中获取）

**Target**：
- 用于监督学习的图像集合
- 包含 source 的所有关键帧 + 另外 `(num_target_keyframes - num_source_keyframes)` 个关键帧（每个关键帧选择 1 帧）
- 共 `num_target_keyframes × num_cams` 张图像
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
        keyframe_split_config: Optional[Dict] = None,  # 关键帧分割配置
        min_keyframes_per_scene: int = 10,  # 场景的最小关键帧数量
        min_keyframes_per_segment: int = 6,  # 段的最小关键帧数量
        device: torch.device = torch.device("cpu"),
        preload_scene_count: int = 3,  # 预加载场景数量
        fixed_segment_aabb: Optional[Tensor] = None,  # 全局固定的段AABB（可选）
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
            preload_scene_count: 预加载场景数量（默认3），用于控制内存占用
            fixed_segment_aabb: 可选的全局固定段AABB。如果提供，所有段将使用此固定AABB，
                而不是从lidar数据计算。形状：[2, 3]，其中 aabb[0] 是 [x_min, y_min, z_min]，
                aabb[1] 是 [x_max, y_max, z_max]。坐标系：x=front, y=left, z=up
                （与 _compute_segment_aabb 一致）。如果为 None，则使用从lidar数据计算的AABB。
        """
        pass
    
    def initialize(self):
        """
        初始化训练队列和预加载初始场景。
        
        此方法：
        1. 初始化训练队列（验证并添加初始场景）
        2. 预加载初始场景
        
        这是可选的 - 数据集会在第一次使用时自动初始化，
        但显式调用此方法可以提前检测错误。
        """
        pass
    
    def mark_scene_completed(self, scene_id: int):
        """
        标记场景训练完成并切换到下一个场景。
        
        此方法：
        1. 验证 scene_id 是否匹配当前场景
        2. 切换到队列中的下一个场景
        3. 卸载已完成的场景
        4. 预加载下一个场景（如果可用）
        
        Args:
            scene_id: 已完成的场景ID
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
                - 'keyframe_info': Dict - 关键帧信息（用于调试/显示）
                    - 'segment_keyframes': List[int] - 段内所有关键帧索引
                    - 'source_keyframes': List[int] - 选择的source关键帧索引
                    - 'target_keyframes': List[int] - 选择的target关键帧索引（包含source）
                - 'source': {
                    'image': Tensor[num_source_keyframes * num_cams, H, W, 3],
                    'extrinsics': Tensor[num_source_keyframes * num_cams, 4, 4],
                    'intrinsics': Tensor[num_source_keyframes * num_cams, 4, 4],
                    'depth': Tensor[num_source_keyframes * num_cams, H, W],
                    'frame_indices': Tensor[num_source_keyframes * num_cams],  # 帧索引
                    'cam_indices': Tensor[num_source_keyframes * num_cams],    # 相机索引
                    'keyframe_indices': Tensor[num_source_keyframes],  # 关键帧索引
                }
                - 'target': {
                    'image': Tensor[num_target_keyframes * num_cams, H, W, 3],
                    'extrinsics': Tensor[num_target_keyframes * num_cams, 4, 4],
                    'intrinsics': Tensor[num_target_keyframes * num_cams, 4, 4],
                    'depth': Tensor[num_target_keyframes * num_cams, H, W],
                    'frame_indices': Tensor[num_target_keyframes * num_cams],
                    'cam_indices': Tensor[num_target_keyframes * num_cams],
                    'keyframe_indices': Tensor[num_target_keyframes],  # 关键帧索引
                }
        """
        pass
    
    def get_current_scene_id(self) -> Optional[int]:
        """
        获取当前训练场景ID。
        
        Returns:
            当前场景ID或None（如果没有可用场景）
        """
        pass
    
    def sample_random_batch(self) -> Dict:
        """
        随机采样一个训练批次。
        
        Returns:
            与 get_segment_batch() 相同的格式
        """
        pass
    
    def create_scheduler(
        self,
        batches_per_segment: int = 20,
        segment_order: str = "random",
        scene_order: str = "random",
        shuffle_segments: bool = True,
        preload_next_scene: bool = True,
    ) -> 'MultiSceneDatasetScheduler':
        """
        创建调度器实例，用于管理场景和段的遍历顺序。
        
        Args:
            batches_per_segment: 每个段遍历的batch数量（默认20）
            segment_order: 段遍历顺序（"random"或"sequential"，默认"random"）
            scene_order: 场景遍历顺序（"random"或"sequential"，默认"random"）
            shuffle_segments: 是否在每个场景内打乱段顺序（默认True）
            preload_next_scene: 是否在最后一个段开始训练时预加载下一个场景（默认True）
            
        Returns:
            MultiSceneDatasetScheduler实例
        """
        pass
```

### MultiSceneDatasetScheduler

```python
class MultiSceneDatasetScheduler:
    """
    调度器类，用于管理场景和段的遍历顺序。
    
    核心功能：
    1. 管理段内batch计数
    2. 自动切换段（当达到batches_per_segment时）
    3. 自动切换场景（当所有段遍历完成时）
    4. 后台线程预加载场景（类似 torch DataLoader 的 worker 线程）
    5. 确保训练队列始终保持满状态
    6. 场景切换时阻塞等待（如果场景未加载完成）
    """
    
    def __init__(
        self,
        dataset: MultiSceneDataset,
        batches_per_segment: int = 20,
        segment_order: str = "random",
        scene_order: str = "random",
        shuffle_segments: bool = True,
        preload_next_scene: bool = True,
    ):
        """
        Args:
            dataset: MultiSceneDataset实例
            batches_per_segment: 每个段遍历的batch数量（默认20）
            segment_order: 段遍历顺序（"random"或"sequential"，默认"random"）
            scene_order: 场景遍历顺序（"random"或"sequential"，默认"random"）
            shuffle_segments: 是否在每个场景内打乱段顺序（默认True）
            preload_next_scene: 是否在最后一个段开始训练时预加载下一个场景（默认True）
        """
        pass
    
    def next_batch(self) -> Dict:
        """
        获取下一个训练batch。
        
        自动管理：
        1. 段内batch计数
        2. 段切换（当达到batches_per_segment时）
        3. 场景切换（当所有段遍历完成时）
        4. 场景预加载（当最后一个段开始训练时）
        
        Returns:
            与 get_segment_batch() 相同的格式
            
        Raises:
            StopIteration: 当所有场景都已处理完成时
        """
        pass
    
    def reset(self):
        """重置调度器状态"""
        pass
    
    def shutdown(self):
        """
        停止后台线程并清理资源。
        
        应该在不再使用调度器时调用，以确保后台线程正确停止。
        """
        pass
    
    def get_current_info(self) -> Dict:
        """
        获取当前调度器状态信息。
        
        Returns:
            Dict包含：
                - 'scene_id': 当前场景ID
                - 'segment_id': 当前段ID（在scene_segment_order中的索引）
                - 'segment_id_in_scene': 场景内的实际段ID
                - 'batch_count': 当前段内的batch计数
                - 'batches_per_segment': 每个段的batch数量
        """
        pass
```

---

## 实现细节

### 1. 初始化流程

**延迟加载机制**：数据集采用延迟加载和预加载机制，不在初始化时加载所有场景，而是按需加载，以节省内存。

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
    self.preload_scene_count = preload_scene_count
    
    # 3. 初始化场景候选池（未验证的场景ID）
    self.scene_candidate_pool = train_scene_ids.copy()
    random.shuffle(self.scene_candidate_pool)  # 打乱以增加随机性
    
    # 4. 初始化训练队列（已验证的场景ID，按训练顺序）
    self.scene_training_queue = []
    
    # 5. 初始化场景缓存（已加载的场景数据，最多 preload_scene_count + 1 个场景）
    self.train_scenes_cache = {}
    
    # 6. 初始化评估场景（按需加载，可以保留所有）
    self.eval_scenes = {}
    
    # 7. 初始化当前场景索引
    self.current_scene_index = 0
    
    # 8. 初始化无效场景ID集合（已验证但不适合的场景）
    self.invalid_scene_ids = set()
    
    # 9. 跟踪是否已初始化
    self._initialized = False

def initialize(self):
    """
    初始化训练队列和预加载初始场景。
    
    此方法：
    1. 初始化训练队列（验证并添加初始场景）
    2. 预加载初始场景
    
    这是可选的 - 数据集会在第一次使用时自动初始化，
    但显式调用此方法可以提前检测错误。
    """
    if self._initialized:
        return
    
    # 确保训练队列有足够的场景
    self._ensure_training_queue_ready()
    
    if len(self.scene_training_queue) == 0:
        logger.warning("No valid training scenes found after validation")
        return
    
    # 预加载初始场景
    self._preload_scenes()
    
    self._initialized = True
```

**场景管理机制**：
- **候选池 (scene_candidate_pool)**：未验证的场景ID列表，用于填充训练队列
- **训练队列 (scene_training_queue)**：已验证且适合训练的场景ID列表，按训练顺序排列
- **场景缓存 (train_scenes_cache)**：已加载的场景数据，最多保留 `preload_scene_count + 1` 个场景
- **评估场景 (eval_scenes)**：评估场景数据，按需加载，可以保留所有
- **无效场景集合 (invalid_scene_ids)**：已验证但不适合训练的场景ID集合

**场景生命周期管理**：
1. **场景验证**：`_validate_and_add_to_queue()` 方法验证场景是否适合训练
   - 加载场景并检查关键帧数量
   - 如果适合，添加到训练队列
   - 如果不适合，标记为无效场景
2. **队列维护**：`_ensure_training_queue_ready()` 方法确保训练队列有足够的场景
   - 目标队列大小：`preload_scene_count + 1`
   - 从候选池中验证并添加场景
   - 如果候选池为空，从原始场景ID中重新填充
   - **线程安全**：所有队列操作都使用锁保护
3. **场景预加载**：`_preload_scenes()` 方法预加载即将使用的场景
   - 加载当前场景（如果未加载）
   - 预加载接下来的 `preload_scene_count` 个场景
   - 如果场景加载失败，从队列中移除
4. **场景切换**：`_switch_to_next_scene()` 和 `mark_scene_completed()` 方法管理场景切换
   - 卸载当前场景
   - 更新当前场景索引
   - 确保队列有足够的场景
   - 预加载下一个场景
   - **线程安全**：所有缓存操作都使用锁保护

**后台线程预加载机制**（MultiSceneDatasetScheduler）：
1. **后台线程**：调度器启动一个守护线程，持续运行直到调度器被销毁
   - 线程名称：`ScenePreloadWorker`
   - 守护线程：主线程退出时自动退出
2. **预加载任务队列**：使用 `queue.Queue` 在主线程和后台线程之间通信
   - 主线程发送预加载任务（场景ID）
   - 后台线程接收任务并执行场景加载
3. **场景加载状态**：使用 `threading.Event` 跟踪每个场景的加载状态
   - 场景加载完成后，设置对应的 Event
   - 主线程在场景切换时，如果场景未加载，阻塞等待 Event
4. **队列持续填充**：后台线程持续监控训练队列状态
   - 当队列不满时，自动调用 `_ensure_training_queue_ready()` 填充队列
   - 确保队列始终保持满状态（至少 `preload_scene_count + 1` 个场景）
5. **预加载策略**：
   - **主动预加载**：在最后一个 segment 开始时，主线程通过任务队列触发下一个场景的预加载
   - **被动填充**：后台线程持续监控队列，自动预加载队列中的下一个场景
6. **阻塞等待机制**：场景切换时，如果下一个场景未加载完成
   - 主线程阻塞等待，直到场景加载完成
   - 使用 `Event.wait()` 实现阻塞等待
   - 避免在场景未准备好时继续训练

### 2. 场景加载和验证

**场景验证流程**：
```python
def _validate_and_add_to_queue(self, scene_id: int) -> bool:
    """
    验证场景并添加到训练队列（如果适合）。
    
    此方法执行轻量级验证，通过加载场景并检查是否适合。
    如果适合，添加到队列。
    
    Args:
        scene_id: 要验证的场景ID
        
    Returns:
        bool: True 如果场景适合并已添加到队列，False 否则
    """
    # 跳过如果已在队列中或无效
    if scene_id in self.scene_training_queue:
        return True
    if scene_id in self.invalid_scene_ids:
        return False
    
    # 尝试加载并准备场景（这会进行完整验证）
    scene_data = self._load_and_prepare_scene(scene_id)
    
    if scene_data is not None:
        # 场景适合，添加到队列
        self.scene_training_queue.append(scene_id)
        # 不保留在缓存中，将在需要时加载
        # 清理加载的数据以节省内存
        if 'dataset' in scene_data:
            dataset = scene_data['dataset']
            if hasattr(dataset, 'cleanup'):
                dataset.cleanup()
            if hasattr(dataset, 'pixel_source') and hasattr(dataset.pixel_source, 'cleanup'):
                dataset.pixel_source.cleanup()
        del scene_data
        return True
    else:
        # 场景不适合，标记为无效
        self.invalid_scene_ids.add(scene_id)
        return False
```

**场景加载流程**：

```python
def _load_and_prepare_scene(self, scene_id: int) -> Optional[Dict]:
    """
    加载场景并完成所有预处理。
    
    此方法加载场景并执行所有必要的预处理：
    - 场景加载（DrivingDataset）
    - 轨迹提取
    - 关键帧分割
    - 场景适合性检查
    - 段分割
    
    Args:
        scene_id: 要加载的场景ID
        
    Returns:
        场景数据字典或None（如果场景不适合）
    """
    return self._load_scene(scene_id)

def _load_scene(self, scene_id: int) -> Optional[Dict]:
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
    4. 为每个段计算独立的 AABB（基于段内帧的lidar数据，或使用固定AABB如果配置了）
    5. 过滤掉关键帧数量不足的段
    
    注意：
    - 段分割不需要那么精确，关键帧的合计距离对比AABB的长度即可
    - 段内设置一个最低关键帧数量限制，不满足则跳过
    - 段与段可以部分重合（overlap_ratio）
    - **每个段使用独立的AABB**：
      - 如果配置了 `fixed_segment_aabb`，所有段使用此固定AABB
      - 否则，基于段内帧的lidar数据计算，而不是使用场景AABB
    
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
            - 'aabb': Tensor[2, 3] - 段的 AABB 边界（基于段内帧的lidar数据计算）
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
        # 使用更激进的公式：如果距离接近AABB长度，创建更多段
        # 按距离比例缩放，如果距离 >= 0.3 * aabb_length，至少创建2个段
        distance_ratio = total_keyframe_distance / aabb_length
        num_segments_by_distance = max(2, int(distance_ratio * 3))  # 更长距离创建更多段
        num_segments = max(1, min(max_segments, num_segments_by_distance))
    
    # 5. 将关键帧按照距离分组为段（支持重叠）
    segments = []
    segment_id = 0
    
    if num_segments == 1:
        # 只有一个段，包含所有关键帧
        all_frames = []
        for kf_seg in keyframe_segments:
            all_frames.extend(kf_seg)
        
        frame_indices = sorted(list(set(all_frames)))
        # 使用固定AABB（如果配置了），否则计算段的AABB（基于段内帧的lidar数据）
        if self.fixed_segment_aabb is not None:
            segment_aabb = self.fixed_segment_aabb
        else:
            segment_aabb = self._compute_segment_aabb(scene_dataset, frame_indices)
        
        segments.append({
            'segment_id': segment_id,
            'keyframe_indices': list(range(len(keyframe_segments))),
            'frame_indices': frame_indices,
            'aabb': segment_aabb,
        })
    else:
        # 多个段，支持重叠
        # 计算段长度和步长
        segment_distance = total_keyframe_distance / num_segments
        # 限制overlap_ratio最大为0.5，避免过度重叠
        overlap_ratio_clamped = min(overlap_ratio, 0.5)
        step_distance = segment_distance * (1 - overlap_ratio_clamped)
        
        # 计算可以生成多少个重叠段
        max_start_distance = total_keyframe_distance - segment_distance
        if step_distance > 0:
            num_overlap_segments = int(max_start_distance / step_distance) + 1
        else:
            # 如果step_distance为0（overlap_ratio = 1），只生成一个段
            num_overlap_segments = 1
        
        # 多次遍历，生成重叠段
        for seg_idx in range(num_overlap_segments):
            segment_start_distance = seg_idx * step_distance
            segment_end_distance = segment_start_distance + segment_distance
            
            # 收集该段内的keyframe
            current_segment_kf_indices = []
            current_segment_frames = set()
            
            for kf_idx in range(len(keyframe_segments)):
                kf_center_distance = (keyframe_ranges[kf_idx, 0] + keyframe_ranges[kf_idx, 1]) / 2.0
                
                # 检查keyframe是否在该段的距离范围内
                if segment_start_distance <= kf_center_distance < segment_end_distance:
                    current_segment_kf_indices.append(kf_idx)
                    current_segment_frames.update(keyframe_segments[kf_idx])
            
            # 只添加关键帧数量足够的段
            if len(current_segment_kf_indices) >= self.min_keyframes_per_segment:
                frame_indices = sorted(list(current_segment_frames))
                # 使用固定AABB（如果配置了），否则计算段的AABB（基于段内帧的lidar数据）
                if self.fixed_segment_aabb is not None:
                    segment_aabb = self.fixed_segment_aabb
                else:
                    segment_aabb = self._compute_segment_aabb(scene_dataset, frame_indices)
                
                segments.append({
                    'segment_id': segment_id,
                    'keyframe_indices': current_segment_kf_indices,
                    'frame_indices': frame_indices,
                    'aabb': segment_aabb,
                })
                segment_id += 1
    
    # 6. 过滤掉关键帧数量不足的段（双重检查）
    valid_segments = [
        seg for seg in segments
        if len(seg['keyframe_indices']) >= self.min_keyframes_per_segment
    ]
    
    return valid_segments
```

### 4.1 段AABB计算

每个段的AABB是基于段内帧的lidar数据独立计算的，而不是使用场景级别的AABB。这确保了每个段都有反映其实际数据范围的边界框。

```python
def _compute_segment_aabb(
    self,
    scene_dataset: DrivingDataset,
    frame_indices: List[int],
) -> Tensor:
    """
    计算段的AABB边界。
    
    使用段内帧的lidar数据，参考 lidar_source.get_aabb 的方式计算。
    
    流程：
    1. 从 scene_dataset.lidar_source 获取段内帧的lidar数据
    2. 筛选段内帧的lidar点（使用 timesteps 匹配 frame_indices）
    3. 计算lidar点的3D坐标：lidar_pts = origins + directions * ranges
    4. 下采样lidar点（使用 lidar_downsample_factor）
    5. 使用分位数计算AABB边界：
       - aabb_min = torch.quantile(lidar_pts, percentile, dim=0)
       - aabb_max = torch.quantile(lidar_pts, 1 - percentile, dim=0)
    6. 调整高度：如果 aabb_max[-1] < 20，设置为 20.0
    
    注意：
    - 如果lidar_source不存在或段内没有lidar数据，回退到场景AABB
    - 配置参数从 lidar_source.data_cfg 获取：
      - lidar_downsample_factor: 下采样因子（默认4）
      - lidar_percentile: 分位数（默认0.02）
    
    Args:
        scene_dataset: 场景数据集实例
        frame_indices: 段内帧索引列表
        
    Returns:
        aabb: Tensor[2, 3] - 段的AABB边界 [min, max]
    """
    # 检查lidar_source是否存在
    if scene_dataset.lidar_source is None:
        logger.warning("Lidar source not available, falling back to scene AABB")
        return scene_dataset.get_aabb()
    
    lidar_source = scene_dataset.lidar_source
    
    # 检查lidar数据是否已加载
    assert (
        lidar_source.origins is not None
        and lidar_source.directions is not None
        and lidar_source.ranges is not None
        and lidar_source.timesteps is not None
    ), "Lidar points not loaded, cannot compute segment AABB."
    
    # 筛选段内帧的lidar点
    frame_indices_tensor = torch.tensor(frame_indices, dtype=lidar_source.timesteps.dtype, device=lidar_source.timesteps.device)
    mask = torch.isin(lidar_source.timesteps, frame_indices_tensor)
    
    if not mask.any():
        logger.warning(f"No lidar points found for frame indices {frame_indices}, falling back to scene AABB")
        return scene_dataset.get_aabb()
    
    # 获取段内帧的lidar点
    segment_origins = lidar_source.origins[mask]
    segment_directions = lidar_source.directions[mask]
    segment_ranges = lidar_source.ranges[mask]
    
    # 计算lidar点的3D坐标
    lidar_pts = segment_origins + segment_directions * segment_ranges
    
    # 下采样lidar点
    downsample_factor = lidar_source.data_cfg.get('lidar_downsample_factor', 4)
    if downsample_factor > 1 and len(lidar_pts) > downsample_factor:
        lidar_pts = lidar_pts[
            torch.randperm(len(lidar_pts))[
                : int(len(lidar_pts) / downsample_factor)
            ]
        ]
    
    # 使用分位数计算AABB
    percentile = lidar_source.data_cfg.get('lidar_percentile', 0.02)
    aabb_min = torch.quantile(lidar_pts, percentile, dim=0)
    aabb_max = torch.quantile(lidar_pts, 1 - percentile, dim=0)
    
    # 清理临时变量
    del lidar_pts
    torch.cuda.empty_cache()
    
    # 通常lidar的高度非常小，所以稍微增加AABB的高度
    if aabb_max[-1] < 20:
        aabb_max[-1] = 20.0
    
    # 组合为 [min, max] 格式
    aabb = torch.stack([aabb_min, aabb_max], dim=0)  # [2, 3]
    
    return aabb
```

**段AABB计算的优势**：
- **精确性**：每个段的AABB反映该段内实际数据范围，而不是整个场景的范围
- **适应性**：不同段可能有不同的空间分布，独立AABB能更好地适应这种变化
- **效率**：使用段内lidar数据计算，避免使用整个场景的lidar数据

**与场景AABB的关系**：
- 段AABB通常包含在场景AABB内，但可能不完全一致（由于分位数计算和下采样）
- 如果段内没有lidar数据或lidar_source不可用，会回退到场景AABB

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
    
    if len(remaining_keyframes) == 0:
        # 所有关键帧都被选为source，重复使用source关键帧作为target
        extra_target_keyframes = source_keyframe_indices * (num_extra_target_keyframes // len(source_keyframe_indices) + 1)
        extra_target_keyframes = extra_target_keyframes[:num_extra_target_keyframes]
    elif len(remaining_keyframes) < num_extra_target_keyframes:
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

### 7. 线程安全机制

**MultiSceneDataset 的线程安全**：
- 使用 `threading.RLock()` 保护队列和缓存操作
- 需要加锁的方法：
  - `_ensure_training_queue_ready()` - 队列操作
  - `_validate_and_add_to_queue()` - 队列操作
  - `_ensure_scene_loaded()` - 缓存操作
  - `_unload_scene()` - 缓存操作
  - `mark_scene_completed()` - 场景切换（涉及队列和缓存）
  - `get_current_scene_id()` - 队列读取
- 注意：`_load_and_prepare_scene()` 是耗时的 I/O 操作，在后台线程中执行，不需要加锁

**MultiSceneDatasetScheduler 的线程同步**：
- 使用 `queue.Queue` 在主线程和后台线程之间通信
- 使用 `threading.Event` 跟踪场景加载状态
- 使用 `threading.RLock()` 保护加载状态字典
- 后台线程持续运行，确保队列满和场景预加载

### 8. 场景缓存管理

**场景缓存策略**：
```python
def _ensure_scene_loaded(self, scene_id: int) -> Optional[Dict]:
    """
    确保指定场景已加载到缓存中。
    
    如果场景已在缓存中，返回它。
    如果不在，使用 _load_and_prepare_scene 加载它。
    如果缓存已满，卸载一个非当前场景。
    
    Args:
        scene_id: 要确保加载的场景ID
        
    Returns:
        场景数据字典或None（如果场景无法加载）
    """
    # 检查是否已在缓存中
    if scene_id in self.train_scenes_cache:
        return self.train_scenes_cache[scene_id]
    
    # 检查是否是评估场景
    if scene_id in self.eval_scene_ids:
        if scene_id not in self.eval_scenes:
            # 加载评估场景
            scene_data = self._load_and_prepare_scene(scene_id)
            if scene_data is not None:
                self.eval_scenes[scene_id] = scene_data
            else:
                return None
        return self.eval_scenes[scene_id]
    
    # 是训练场景，检查缓存大小
    max_cache_size = self.preload_scene_count + 1
    
    # 如果缓存已满，卸载一个非当前场景
    if len(self.train_scenes_cache) >= max_cache_size:
        # 找到一个要卸载的场景（优先非当前场景）
        current_scene_id = self.get_current_scene_id()
        for cached_scene_id in list(self.train_scenes_cache.keys()):
            if cached_scene_id != current_scene_id:
                self._unload_scene(cached_scene_id)
                break
        # 如果仍然满，卸载任何场景
        if len(self.train_scenes_cache) >= max_cache_size:
            scene_to_unload = list(self.train_scenes_cache.keys())[0]
            self._unload_scene(scene_to_unload)
    
    # 加载场景
    scene_data = self._load_and_prepare_scene(scene_id)
    if scene_data is not None:
        self.train_scenes_cache[scene_id] = scene_data
        return scene_data
    else:
        return None

def _unload_scene(self, scene_id: int):
    """
    从缓存中卸载场景并释放内存。
    
    Args:
        scene_id: 要卸载的场景ID
    """
    if scene_id in self.train_scenes_cache:
        scene_data = self.train_scenes_cache[scene_id]
        
        # 清理数据集资源（如果它有清理方法）
        if 'dataset' in scene_data:
            dataset = scene_data['dataset']
            if hasattr(dataset, 'cleanup'):
                dataset.cleanup()
            if hasattr(dataset, 'pixel_source') and hasattr(dataset.pixel_source, 'cleanup'):
                dataset.pixel_source.cleanup()
        
        # 从缓存中移除
        del self.train_scenes_cache[scene_id]
        logger.info(f"Scene {scene_id} unloaded from cache")
```

### 9. 批次打包

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
                # 如果深度图不存在，创建占位符（值为10.0）
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
                # 如果深度图不存在，创建占位符（值为10.0）
                H, W = image_infos['pixels'].shape[:2]
                depth = torch.ones(H, W, dtype=torch.float32, device=self.device) * 10.0
            target_depths.append(depth)
            
            target_frame_idxs.append(frame_idx)
            target_cam_idxs.append(cam_idx)
    
    # 6. 组装批次
    batch = {
        'scene_id': torch.tensor([scene_id], dtype=torch.long),
        'segment_id': segment_id,
        
        # 关键帧信息（用于调试/显示）
        'keyframe_info': {
            'segment_keyframes': segment['keyframe_indices'],  # 段内所有关键帧
            'source_keyframes': source_keyframe_indices,  # 选择的source关键帧索引
            'target_keyframes': target_keyframe_indices,  # 选择的target关键帧索引（包含source）
        },
        
        'source': {
            'image': torch.stack(source_images, dim=0),  # [num_source_keyframes * num_cams, H, W, 3]
            'extrinsics': torch.stack(source_extrinsics, dim=0),  # [num_source_keyframes * num_cams, 4, 4]
            'intrinsics': torch.stack(source_intrinsics, dim=0),  # [num_source_keyframes * num_cams, 4, 4]
            'depth': torch.stack(source_depths, dim=0),  # [num_source_keyframes * num_cams, H, W]
            'frame_indices': torch.tensor(source_frame_idxs, dtype=torch.long),  # [num_source_keyframes * num_cams]
            'cam_indices': torch.tensor(source_cam_idxs, dtype=torch.long),  # [num_source_keyframes * num_cams]
            'keyframe_indices': torch.tensor(source_keyframe_indices, dtype=torch.long),  # [num_source_keyframes]
        },
        
        'target': {
            'image': torch.stack(target_images, dim=0),  # [num_target_keyframes * num_cams, H, W, 3]
            'extrinsics': torch.stack(target_extrinsics, dim=0),  # [num_target_keyframes * num_cams, 4, 4]
            'intrinsics': torch.stack(target_intrinsics, dim=0),  # [num_target_keyframes * num_cams, 4, 4]
            'depth': torch.stack(target_depths, dim=0),  # [num_target_keyframes * num_cams, H, W]
            'frame_indices': torch.tensor(target_frame_idxs, dtype=torch.long),  # [num_target_keyframes * num_cams]
            'cam_indices': torch.tensor(target_cam_idxs, dtype=torch.long),  # [num_target_keyframes * num_cams]
            'keyframe_indices': torch.tensor(target_keyframe_indices, dtype=torch.long),  # [num_target_keyframes]
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
    
    优先级：
    1. 从 camera_data.depth_maps 获取（从文件加载的深度图）
    2. 从 camera_data.lidar_depth_maps 获取（从LiDAR投影得到的深度图）
    3. 如果都不存在，返回None（调用者会创建占位符，值为10.0）
    
    Returns:
        depth: Tensor[H, W] 或 None
    """
    try:
        pixel_source = scene_dataset.pixel_source
        cam_id = pixel_source.camera_list[cam_idx]
        camera_data = pixel_source.camera_data[cam_id]
        
        # 方法1：从 depth_maps 获取（从文件加载）
        if hasattr(camera_data, 'depth_maps') and camera_data.depth_maps is not None:
            depth = camera_data.depth_maps[frame_idx]  # Tensor[H, W]
            return depth.to(self.device)
        
        # 方法2：从 lidar_depth_maps 获取（从LiDAR投影）
        if camera_data.lidar_depth_maps is not None:
            depth = camera_data.lidar_depth_maps[frame_idx]  # Tensor[H, W]
            return depth.to(self.device)
    except (IndexError, KeyError, AttributeError) as e:
        logger.warning(f"Failed to get depth map for camera {cam_idx}, frame {frame_idx}: {e}")
    
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

- [ ] **Source 图像数量**：`num_source_keyframes * num_cams`（相机数量是动态的，从场景数据中获取）
- [ ] **Target 图像数量**：`num_target_keyframes * num_cams`
- [ ] **Target 包含 Source**：Target 的关键帧列表包含 Source 的所有关键帧
- [ ] **关键帧选择不重复**：Source 和额外的 Target 关键帧不重复（但 Target 包含 Source）
- [ ] **批次包含 keyframe_info**：批次中包含 `keyframe_info` 字段，用于调试和显示

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
- [ ] **Source 维度正确**：`[num_source_keyframes * num_cams, H, W, 3]`
- [ ] **Target 维度正确**：`[num_target_keyframes * num_cams, H, W, 3]`
- [ ] **包含元数据**：frame_indices, cam_indices, keyframe_indices 等信息正确
- [ ] **包含 keyframe_info**：批次中包含 keyframe_info 字段，包含段内所有关键帧、source关键帧、target关键帧信息

### 8. 轨迹获取检查

- [ ] **轨迹来源正确**：使用前相机的轨迹（或指定相机）
- [ ] **轨迹格式正确**：`[num_frames, 4, 4]` 的变换矩阵
- [ ] **轨迹完整性**：包含所有训练帧的轨迹

### 9. AABB 和段分割检查

- [ ] **AABB 获取正确**：使用 `scene_dataset.get_aabb()` 获取场景 AABB
- [ ] **段分割合理**：段的大小和数量合理
- [ ] **重叠处理正确**：段与段之间的重叠比例正确应用
- [ ] **固定AABB使用正确**：如果配置了 `fixed_segment_aabb`，所有段使用此固定AABB
- [ ] **固定AABB格式正确**：固定AABB的形状为 [2, 3]，min < max
- [ ] **固定AABB坐标系正确**：固定AABB使用与 `_compute_segment_aabb` 相同的坐标系（x=front, y=left, z=up）

### 10. 与 EVolSplat Offset 机制的兼容性

- [ ] **场景 ID 传递正确**：batch 中的 scene_id 可以正确索引模型的 offset
- [ ] **段 ID 处理**：段 ID 可能需要映射到场景级别的 offset（如果 offset 是场景级别的）
- [ ] **Offset 共享**：同一场景的不同段可能共享 offset（需要根据实际需求决定）

### 11. 调度器检查

- [ ] **段内batch计数正确**：每个段固定遍历 `batches_per_segment` 次
- [ ] **段切换正确**：当达到 `batches_per_segment` 时，自动切换到下一个段
- [ ] **场景切换正确**：当所有段遍历完成时，自动切换到下一个场景
- [ ] **段遍历顺序正确**：根据 `segment_order` 配置正确遍历（random或sequential）
- [ ] **场景预加载正确**：在最后一个段开始训练时预加载下一个场景
- [ ] **边界情况处理**：当所有场景遍历完成时，正确抛出 `StopIteration` 异常
- [ ] **重叠段生成正确**：段之间有正确的overlap（基于距离，不是keyframe数量）
- [ ] **重叠段数量合理**：重叠段数量不会过多（overlap_ratio限制在0.5以内）

### 12. 后台线程检查

- [ ] **后台线程启动**：调度器创建时自动启动后台线程
- [ ] **后台线程停止**：调用 `shutdown()` 时正确停止后台线程
- [ ] **队列持续填充**：后台线程确保训练队列始终保持满状态
- [ ] **场景预加载**：后台线程自动预加载队列中的下一个场景
- [ ] **阻塞等待**：场景切换时，如果场景未加载，主线程阻塞等待
- [ ] **线程安全**：所有共享状态访问都使用锁保护
- [ ] **资源清理**：调度器销毁时正确清理线程资源

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

# 2. 创建数据集（使用固定段AABB）
fixed_aabb = torch.tensor([
    [-20.0, -20.0, -5.0],  # [x_min, y_min, z_min]
    [20.0, 4.8, 20.0]      # [x_max, y_max, z_max]
])
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
    min_keyframes_per_scene=10,
    min_keyframes_per_segment=6,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    preload_scene_count=3,  # 预加载3个场景
    fixed_segment_aabb=fixed_aabb,  # 使用固定段AABB（可选）
)

# 3. 初始化数据集（可选，会在第一次使用时自动初始化）
dataset.initialize()

# 4. 获取当前场景ID
current_scene_id = dataset.get_current_scene_id()
print(f"当前训练场景: {current_scene_id}")

# 5. 获取随机批次
batch = dataset.sample_random_batch()

# 6. 获取指定场景和段的批次
batch = dataset.get_segment_batch(scene_id=0, segment_id=2)

# 7. 获取场景信息
scene_info = dataset.get_scene(scene_id=0)
print(f"场景 {scene_id} 有 {len(scene_info['segments'])} 个段")

# 8. 在训练循环中使用（方式1：使用sample_random_batch）
for iteration in range(100):
    batch = dataset.sample_random_batch()
    scene_id = batch['scene_id'].item()
    segment_id = batch['segment_id']
    
    # 使用批次进行训练
    # loss = model(batch)
    # loss.backward()
    # optimizer.step()
    
    # 如果场景训练完成，标记并切换到下一个场景
    # dataset.mark_scene_completed(scene_id)

# 9. 在训练循环中使用（方式2：使用调度器，推荐）
scheduler = dataset.create_scheduler(
    batches_per_segment=20,
    segment_order="random",
    scene_order="random",
    shuffle_segments=True,
    preload_next_scene=True,
)

try:
    for iteration in range(100):
        try:
            batch = scheduler.next_batch()
            
            # 使用批次进行训练
            # loss = model(batch)
            # loss.backward()
            # optimizer.step()
            
            # 获取当前状态信息（可选）
            if iteration % 100 == 0:
                info = scheduler.get_current_info()
                print(f"Iteration {iteration}: scene_id={info['scene_id']}, "
                      f"segment_id={info['segment_id_in_scene']}, "
                      f"batch_count={info['batch_count']}/{info['batches_per_segment']}")
        except StopIteration:
            # 所有场景遍历完成
            print("All scenes have been processed")
            break
finally:
    # 确保清理后台线程
    scheduler.shutdown()
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
5. **延迟加载和预加载机制**：按需加载场景，控制内存占用，最多同时缓存 `preload_scene_count + 1` 个训练场景
6. **场景队列管理**：使用候选池、训练队列、场景缓存等机制管理场景生命周期
7. **后台线程预加载**：类似 torch DataLoader 的 worker 线程，持续预加载场景，确保训练队列满
8. **线程安全**：所有队列和缓存操作都使用锁保护，支持多线程场景加载
9. **阻塞等待机制**：场景切换时，如果场景未加载完成，主线程阻塞等待，确保数据就绪
10. **EVolSplat 兼容**：输出格式符合 EVolSplat 的要求，包含 keyframe_info 等元数据
11. **Drivestudio 集成**：复用 `DrivingDataset` 和 `ScenePixelSource` 的接口
12. **动态相机数量**：支持不同场景有不同数量的相机

该设计允许在不修改 Drivestudio 核心代码的情况下，实现支持动态物体的 feed-forward 3DGS 训练，同时通过延迟加载和预加载机制有效控制内存占用。

