# MultiSceneDataset 使用文档

## 概述

`MultiSceneDataset` 是 Drivestudio 中用于 EVolSplat feed-forward 3DGS 训练的多场景数据集类。它支持多场景管理、基于关键帧的场景分割、段级别的数据组织，以及灵活的 source/target 图像对生成。

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

### 2. Source 和 Target 的定义

**Source**：
- 用于特征提取的图像集合
- 每次使用 `num_source_keyframes` 个关键帧（默认 3），每个关键帧选择 1 帧
- 共 `num_source_keyframes × num_cams` 张图像

**Target**：
- 用于监督学习的图像集合
- 包含 source 的所有关键帧 + 另外 `(num_target_keyframes - num_source_keyframes)` 个关键帧（每个关键帧选择 1 帧）
- 共 `num_target_keyframes × num_cams` 张图像
- **注意**：Target 包含 source，不要求独立

### 3. 训练/测试帧分离

**帧分离机制**：
- 根据 `test_image_stride` 配置分离训练帧和测试帧
- 如果 `test_image_stride = 0`：所有帧同时用于训练和测试
- 如果 `test_image_stride > 0`：每隔 `test_image_stride` 帧被标记为测试帧，其余为训练帧
- **关键帧分割只使用训练帧**：确保训练和测试数据的分离

**测试帧限制**：
- 通过 `max_test_images` 配置限制每个段中使用的测试帧数量
- 如果测试帧数量超过 `max_test_images`，随机采样指定数量的测试帧

---

## 坐标系定义

### 自动驾驶坐标系

MultiSceneDataset 使用的坐标系定义如下：

- **x 轴**：左右方向（左为负，右为正）
- **y 轴**：上下方向（上为负，下为正）
- **z 轴**：后前方向（后为负，前为正）

**AABB 格式**：
```python
aabb = torch.tensor([
    [x_min, y_min, z_min],  # 最小值
    [x_max, y_max, z_max]   # 最大值
])
```

StreetForward 方案 A 约定：`crop_aabb` / `input_aabb` 定义在 **segment 第一帧坐标系**。数据侧会将点云（背景/远景的“世界坐标”）及 source/target/test 相机外参统一转换到该坐标系；动态点云保持局部坐标不变。

### StreetForward 资产坐标契约

`build_streetforward_scene_assets.py` / `build_streetforward_segment_assets.py` 导出的资产统一使用 **seg0 camera/OpenCV** 坐标系：

- **x**：向右为正
- **y**：向下为正
- **z**：向前为正

Waymo 原生坐标为 `x` 前、`y` 左、`z` 上；Waymo sourceloader 会先按自身 `OPENCV2DATASET` 约定构造 camera-to-world，再由 StreetForward 的 `world_to_seg0 @ camera_to_world / pointcloud / dynamic_tracks` 路径归一到上述 seg0 camera/OpenCV 坐标。训练和推理不应再对 Waymo 做额外坐标分支。

Waymo 资产导出默认使用前向 3 相机 `[0, 1, 2]`。当前处理后的 Waymo 样例数据通常只有这 3 个相机有 depth `.npy`；若 `load_depth_maps: true` 且选择 side camera `3/4`，导出前 preflight 会直接报错。`dataset.segment_aabb` 也必须继续写 seg0 camera/OpenCV 坐标，推荐 Waymo 默认范围为：

```yaml
dataset:
  segment_aabb:
    - [-20.0, -10.0, -5.0]
    - [20.0, 4.8, 80.0]
```

**示例**：
```python
# 典型的段 AABB 范围
fixed_aabb = torch.tensor([
    [-20.0, -20.0, -5.0],  # [x_min, y_min, z_min]
    [20.0, 4.8, 20.0]      # [x_max, y_max, z_max]
])
```

### 相机坐标系

- **外参 (extrinsics)**：`camera_to_world` 变换矩阵，形状 `[4, 4]`
- **内参 (intrinsics)**：相机内参矩阵，形状 `[4, 4]`（从 3x3 自动转换）

---

## 数据范围与格式

### 图像数据

- **数值范围**：`[0.0, 1.0]`（已归一化，原始图像除以 255）
- **数据类型**：`torch.float32`
- **形状**：`[H, W, 3]`（通道在最后）
- **颜色通道**：RGB 顺序

### 深度数据

- **数值范围**：实际深度值（米），如果深度图不存在则使用占位符值 `10.0`
- **数据类型**：`torch.float32`
- **形状**：`[H, W]`
- **优先级**：
  1. 从 `camera_data.depth_maps` 获取（从文件加载的深度图）
  2. 从 `camera_data.lidar_depth_maps` 获取（从 LiDAR 投影得到的深度图）
  3. 如果都不存在，创建占位符（值为 `10.0`）

### 相机参数

- **外参 (extrinsics)**：
  - 形状：`[4, 4]`
  - 类型：`torch.float32`
  - 格式：`camera_to_world` 变换矩阵

- **内参 (intrinsics)**：
  - 形状：`[4, 4]`（自动从 3x3 转换）
  - 类型：`torch.float32`
  - 格式：标准相机内参矩阵

### 点云数据（可选）

如果配置了点云生成器，批次中会包含点云数据：

- **背景点云**：
  - 格式：`np.ndarray [N, 6]`
  - 内容：`[x, y, z, r, g, b]`（世界坐标系）

- **动态物体点云**：
  - 格式：`Dict[int, np.ndarray]`，键为 `instance_id`
  - 内容：每个实例的点云 `[M_i, 6]`，格式为 `[x, y, z, r, g, b]`（局部坐标系）

### 动态物体信息（可选）

如果点云包含动态物体且 `pixel_source` 支持，批次中会包含动态物体信息：

- **格式**：`Dict[int, Dict]`，按 `frame_idx` 索引
- **内容**：每个帧包含：
  ```python
  {
      "instances": {
          instance_id: {
              "quat": List[4],  # [w, x, y, z] 四元数（wxyz格式）
              "trans": List[3],  # [x, y, z] 平移向量
          }
      }
  }
  ```

---

## 文件夹结构

### 场景文件夹结构

每个场景对应一个文件夹，标准结构如下：

```
scene_dir/ (例如: /path/to/trainval/000)
├── images/                    # RGB图像目录
│   ├── 000_0.jpg (或 .png)   # 格式: {frame_idx:03d}_{cam_id}.jpg
│   ├── 000_1.jpg
│   ├── 001_0.jpg
│   └── ...
├── depth/                     # 深度图目录（可选）
│   ├── 000_0.npy             # 格式: {frame_idx:03d}_{cam_id}.npy
│   ├── 000_0_meta.npz        # 元数据文件（包含原始尺寸、内参等）
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
│   ├── 000_0.png             # 格式: {frame_idx:03d}_{cam_id}.png (像素0=sky,255=non-sky; 加载后转为0/1 float)
│   └── ...
├── lidar/                     # LiDAR数据（可选，用于AABB计算和点云生成）
│   ├── 000.bin               # 格式: {frame_idx:03d}.bin
│   └── ...
└── transforms.json           # 场景变换信息（可选）
```

### 文件命名规则

- **图像**：`{frame_idx:03d}_{cam_id}.jpg` 或 `.png`
- **深度图**：`{frame_idx:03d}_{cam_id}.npy`（对应元数据：`{frame_idx:03d}_{cam_id}_meta.npz`）
- **外参**：`{frame_idx:03d}_{cam_id}.txt`（4x4矩阵，每行一个值）
- **内参**：`{cam_id}.txt`（fx, fy, cx, cy）
- **天空掩码**：`{frame_idx:03d}_{cam_id}.png`（0=天空，255=非天空）

---

## 输出结构

### Batch 格式

`get_segment_batch()` 方法返回的批次格式如下：

```python
batch = {
    # 场景和段标识
    'scene_id': Tensor[1],           # 场景ID（long类型）
    'segment_id': int,                # 段ID（场景内索引）
    'aabb': Tensor[2, 3],             # 段 AABB [min, max]，坐标系为 segment 第一帧 (seg0)，与 extrinsics/点云一致
    
    # 关键帧信息（用于调试/显示）
    'keyframe_info': {
        'segment_keyframes': List[int],      # 段内所有关键帧索引
        'source_keyframes': List[int],       # 选择的source关键帧索引
        'target_keyframes': List[int],        # 选择的target关键帧索引（包含source）
    },
    
    # Source 数据
    'source': {
        'image': Tensor[V_s, H, W, 3],           # V_s = num_source_keyframes * num_cams
        'extrinsics': Tensor[V_s, 4, 4],
        'intrinsics': Tensor[V_s, 4, 4],
        'depth': Tensor[V_s, H, W],
        'frame_indices': Tensor[V_s],             # 帧索引
        'cam_indices': Tensor[V_s],               # 相机索引
        'keyframe_indices': Tensor[num_source_keyframes],  # 关键帧索引
        'viewdirs': Tensor[V_s, H, W, 3],        # 可选，射线方向
        'sky_mask': Tensor[V_s, H, W],           # 可选，**1=天空，0=非天空**（float）
        'egocar_mask': Tensor[V_s, H, W],        # 可选，自车mask（1=需要忽略的自车区域，0=有效区域）
    },
    
    # Target 数据
    'target': {
        'image': Tensor[V_t, H, W, 3],           # V_t = num_target_keyframes * num_cams
        'extrinsics': Tensor[V_t, 4, 4],
        'intrinsics': Tensor[V_t, 4, 4],
        'depth': Tensor[V_t, H, W],
        'frame_indices': Tensor[V_t],             # 帧索引
        'cam_indices': Tensor[V_t],               # 相机索引
        'keyframe_indices': Tensor[num_target_keyframes],  # 关键帧索引
        'viewdirs': Tensor[V_t, H, W, 3],        # 可选，射线方向（用于 Sky 渲染）
        'sky_mask': Tensor[V_t, H, W],           # 可选，**1=天空，0=非天空**（float），见下「sky_mask」
        'egocar_mask': Tensor[V_t, H, W],        # 可选，自车mask（1=需要忽略的自车区域，0=有效区域）
    },
    
    # 测试视图（可选，如果 include_test=True 且段内包含测试帧）
    'test': {
        'image': Tensor[num_test_images, H, W, 3],
        'extrinsics': Tensor[num_test_images, 4, 4],
        'intrinsics': Tensor[num_test_images, 4, 4],
        'depth': Tensor[num_test_images, H, W],
        'frame_indices': Tensor[num_test_images],
        'cam_indices': Tensor[num_test_images],
        'viewdirs': Tensor[num_test_images, H, W, 3],   # 可选
        'sky_mask': Tensor[num_test_images, H, W],      # 可选
        'egocar_mask': Tensor[num_test_images, H, W],   # 可选
    },
    
    # 点云数据（配置了点云生成器时必有）
    'pointcloud': {
        'background': np.ndarray [N, 6],           # [x, y, z, r, g, b]
        'dynamic': Dict[int, np.ndarray],          # {instance_id: [M_i, 6]}
    },
    
    # 动态物体信息（可选，如果点云包含动态物体）
    'dynamic_info': Dict[int, Dict],              # 按 frame_idx 索引
}
```


### viewdirs 与 sky_mask（可选）

当底层 `pixel_source.get_image()` 返回的 `image_infos` 中包含对应键时，MultiSceneDataset 会在组装 batch 时收集并填入：

- **viewdirs**：射线方向，形状 `[V, H, W, 3]`，来自 `image_infos['viewdirs']`（与 `get_rays` 约定一致，世界系单位向量）。用于 Sky 渲染等。**Stage 3.1（天空）要求 target 提供 viewdirs**，需使用会返回 `viewdirs` 的 pixel_source（如 `datasets/base/pixel_source.py` 中实现）。
- **sky_mask**：天空掩码，形状 `[V, H, W]`，来自 `image_infos['sky_masks']`，在 **batch 内**统一为 float 0/1，语义为 **1=天空，0=非天空**（与名称一致）。  
  - 在 `data` 配置中设置 **`sky_mask_semantics`**（当 `pixel_source.load_sky_mask: true` 时必填）：`one_is_non_sky` 表示 loader 中「1=非天空」（常见 PNG：0=天空、255=非天空 经 `>0` 后），数据集在组装时做 `1-x` 归一化；`one_is_sky` 表示 loader 中已为「1=天空」，不再取反。详见 [Sky_Mask_Semantics_One_Is_Sky_Refactor.md](Sky_Mask_Semantics_One_Is_Sky_Refactor.md)。

若某张图像的 `get_image()` 未返回 `viewdirs` 或 `sky_masks`，该视角在 stack 时以占位（viewdirs 为零向量、**sky_mask 为 0 表示全图非天空**）保持形状一致。使用 Stage 3.1 时请确保 pixel_source 为所有 target 图像提供 viewdirs。

**Stage 3.1 contract（强约束）**：

- `target['viewdirs']` 必须与 `target['image']` / `target['gt_image']` **同分辨率**（相同的 H/W）。
- `viewdirs[...,3]` 必须是 **单位向量**（与 `get_rays()` 一致）。
- `MinimalStreetForwardStage3_1` 在 trainer 侧会对分辨率不一致 **直接报错**（不再在 trainer 内插值 resize viewdirs）。如需兼容旧缓存 batch，应在 batch 转换阶段用 `get_rays()` 在目标分辨率重算并写入。

### 数据维度说明

- **V_s**：Source 图像数量 = `num_source_keyframes × num_cams`
- **V_t**：Target 图像数量 = `num_target_keyframes × num_cams`
- **H, W**：图像高度和宽度（从场景数据中获取）
- **num_cams**：场景的相机数量（不同场景可能不同）
- **num_test_images**：测试图像数量（取决于段内测试帧数量和 `max_test_images` 配置）

---

## 索引系统

### 图像索引 (img_idx)

- **全局图像索引**：`img_idx = frame_idx * num_cams + cam_idx`
- **用途**：用于 `pixel_source.get_image(img_idx)` 访问图像

### 帧索引 (frame_idx)

- **时间步索引**，范围 `[0, num_frames)`
- 同一帧的所有相机图像共享相同的 `frame_idx`
- 根据 `test_image_stride` 分为训练帧和测试帧

### 相机索引 (cam_idx)

- 相机在 `camera_list` 中的索引
- 范围 `[0, num_cams)`

### 关键帧索引 (keyframe_idx)

- 关键帧在段内的索引（不是全局索引）
- 用于标识段内的关键帧位置

---

## 初始化参数

### 必需参数

- **data_cfg**：Drivestudio 数据配置（OmegaConf）
- **train_scene_ids**：训练场景ID列表
- **eval_scene_ids**：评估场景ID列表

### 可选参数

- **num_source_keyframes**：Source 使用的关键帧数量（默认 3）
- **num_target_keyframes**：Target 使用的关键帧数量（默认 6，包含 source）
- **segment_overlap_ratio**：段与段之间的重叠比例（默认 0.2）
- **keyframe_split_config**：关键帧分割配置
  - `num_splits`：关键帧分割数量（0表示自动）
  - `min_count`：每个关键帧段的最小帧数（默认 1）
  - `min_length`：每个关键帧段的最小长度（默认 0）
- **min_keyframes_per_scene**：场景的最小关键帧数量，不满足则跳过（默认 10）
- **min_keyframes_per_segment**：段的最小关键帧数量，不满足则跳过（默认 6）
- **device**：设备（默认 CPU）
- **preload_scene_count**：预加载场景数量（默认 3），用于控制内存占用
- **segment_aabb**：**必需**，段级 AABB（**segment 第一帧坐标系 seg0**）
  - 形状：`[2, 3]`；格式：`[[x_min, y_min, z_min], [x_max, y_max, z_max]]`
  - 用途：作为 **batch['aabb']**、点云硬裁剪 `crop_aabb`、以及模型 `bbx_min/bbx_max` 的唯一来源
- **segment_input_aabb**：**必需**，用于点云 inside/outside 分流与分层过滤（seg0 系）
  - 形状同上
- **pointcloud_config**：**必需**，点云生成器配置（不再包含 AABB 与 use_bbx 开关）
  - `type`：生成器类型（"monocular" / "lidar" / "hybrid"）
  - 其余参数如 `sparsity/filter_sky/depth_consistency/downscale/...` 仍可配置
  - `type=monocular` 时，`dynamic_filter` 为必填，动态mask键固定为 `dynamic_masks`；可选 `dynamic_recovery` 子配置启用 3D bbox 动态点回收
- `type=hybrid` 时，单目动态过滤固定开启并使用 `dynamic_masks`，需提供 `monocular_dynamic_recovery_bbox_expand_xyz_m` / `monocular_dynamic_recovery_max_points_per_instance`

---

## 主要方法

### 初始化

```python
dataset = MultiSceneDataset(
    data_cfg=data_cfg,
    train_scene_ids=[0, 1, 2, 3, 4],
    eval_scene_ids=[5, 6],
    num_source_keyframes=3,
    num_target_keyframes=6,
    # ... 其他参数
)

# 可选：显式初始化（会在第一次使用时自动初始化）
dataset.initialize()
```

### 获取批次

```python
# 方式1：获取指定场景和段的批次
batch = dataset.get_segment_batch(scene_id=0, segment_id=2, include_test=True)

# 方式2：随机采样批次（训练场景）
batch = dataset.sample_random_batch(eval=False, include_test=False)

# 方式3：随机采样批次（评估场景）
eval_batch = dataset.sample_random_batch(eval=True, include_test=False)
```

### 获取场景信息

```python
# 获取场景数据
scene_info = dataset.get_scene(scene_id=0)
print(f"场景有 {len(scene_info['segments'])} 个段")
print(f"场景有 {scene_info['num_frames']} 帧")
print(f"场景有 {scene_info['num_cams']} 个相机")

# 获取段内所有帧索引
frame_indices = dataset.get_segment_frames(scene_id=0, segment_id=0)

# 获取指定帧和相机的数据
frame_data = dataset.get_frame_data(scene_id=0, frame_idx=0, cam_idx=0)
# frame_data 包含：'image', 'extrinsic', 'intrinsic', 'depth', 'sky_mask'
```

### 使用调度器（推荐）

> Scheduler v2（`MultiSceneDatasetV2`）请参考：`docs/dataloader/MultiSceneDataset_V2_Usage.md` 与 `docs/trainers/StreetForward_Scheduler_V2_Usage.md`。本节示例是 legacy `create_scheduler()`。

```python
# 创建调度器
scheduler = dataset.create_scheduler(
    batches_per_segment=20,
    segment_order="random",
    scene_order="random",
    shuffle_segments=True,
    preload_next_scene=True,
    include_test=False,
)

# 在训练循环中使用
try:
    for iteration in range(1000):
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
    print("All scenes have been processed")
finally:
    # 确保清理后台线程
    scheduler.shutdown()
```

---

## 场景管理机制

### 延迟加载和预加载

- **延迟加载**：场景不在初始化时加载，而是按需加载
- **预加载机制**：最多同时缓存 `preload_scene_count + 1` 个训练场景
- **场景切换**：使用 `mark_scene_completed()` 标记场景训练完成并切换到下一个场景

### 场景队列

- **候选池**：未验证的场景ID列表，用于填充训练队列
- **训练队列**：已验证且适合训练的场景ID列表，按训练顺序排列
- **场景缓存**：已加载的场景数据，最多保留 `preload_scene_count + 1` 个场景
- **评估场景**：评估场景数据，按需加载，可以保留所有

---

## 段分割机制

### 段分割策略

1. **基于轨迹距离与参考长度**：段数由关键帧总轨迹距离与「参考 AABB 长度」决定；参考长度来自 pointcloud 的 **crop_aabb**（与训练时点云裁剪一致，seg0 系）。
2. **段重叠**：段与段之间可以部分重合（`segment_overlap_ratio`）。
3. **段内无独立 AABB 字段**：每个 segment 不再存储 `aabb`；训练/渲染使用的 AABB 由 **batch['aabb']** 提供（见下文），坐标系为 segment 第一帧 (seg0)。

### Batch 中的 AABB（seg0 系）

- **batch['aabb']**：形状 `[2, 3]`，为 **segment 第一帧坐标系 (seg0)** 下的 AABB，与 batch 内外参、点云、dynamic_info 一致。\n+- **取值**：固定等于 `dataset.segment_aabb`（唯一来源）。Trainer 应使用 `batch['aabb']` 作为场景框，保证与当前 batch 同系。

---

## 关键帧分割机制

### 分割方法

- **基于距离**：使用 `split_trajectory` 函数按轨迹距离分割关键帧
- **自动确定**：如果 `num_splits=0`，自动确定关键帧数量
- **最小限制**：每个关键帧段至少包含 `min_count` 帧，长度至少 `min_length`

### 轨迹获取

- **来源**：使用前相机的轨迹（`front_center_interp`）
- **格式**：`[num_frames, 4, 4]` 的变换矩阵
- **仅使用训练帧**：关键帧分割只基于训练帧，不包含测试帧

---

## 点云生成（可选）

### 配置点云生成器

```python
pointcloud_config = {
    'type': 'monocular',
    'chosen_cam_ids': [0, 1, 2],
    'sparsity': 'full',
    'filter_sky': True,
    'depth_consistency': True,
    'downscale': 2,
    'dynamic_filter': True,
    'dynamic_recovery': {
        'enable': True,
        'bbox_expand_xyz_m': [0.3, 0.2, 0.5],
        'max_points_per_instance': 3000,
    },
}

dataset = MultiSceneDataset(
    # ... 其他参数
    segment_aabb=[[-20, -20, -20], [20, 4.8, 70]],
    segment_input_aabb=[[-20, -20, -20], [20, 4.8, 120]],
    pointcloud_config=pointcloud_config,
)
```

### 使用点云生成器

```python
# 如果配置了点云生成器，批次中会自动包含点云数据
batch = dataset.get_segment_batch(scene_id=0, segment_id=0)
if 'pointcloud' in batch:
    background_pcd = batch['pointcloud']['background']  # [N, 6]
    dynamic_pcds = batch['pointcloud']['dynamic']       # Dict[int, np.ndarray]

# 使用调度器生成点云
if dataset.pointcloud_generator is not None:
    # 为当前段生成点云
    pointcloud = scheduler.generate_segment_pointcloud(
        pointcloud_generator=dataset.pointcloud_generator,
    )
    
    # 为场景的所有段生成点云
    all_pointclouds = scheduler.generate_all_segment_pointclouds(
        pointcloud_generator=dataset.pointcloud_generator,
        scene_id=0,
        save_dir="/path/to/save/pointclouds",  # 可选
    )
```

### Segment 级点云缓存（性能关键）

- `pointcloud` 是 **segment 级静态对象**（对同一 `(scene_id, segment_id)` 不随 batch 内 source/target 采样变化）。
- `MultiSceneDataset` 会按 `(scene_id, segment_id)` 缓存点云；同一段后续 `get_segment_batch()` 默认复用缓存，不再重复调用生成器。
- 场景从缓存卸载（`_unload_scene`）时，会同步清理该场景对应的 segment 点云缓存以释放内存。
- 因此 one-segment 训练中，首个 batch 可能较慢（首次建云），后续 batch 不应再承担重复建云开销。

---

## 注意事项

### 数据范围

- **图像**：已归一化到 `[0.0, 1.0]`，不需要再次归一化
- **深度**：实际深度值（米），如果不存在则使用占位符 `10.0`

### 坐标系

- **AABB**：使用 x=左右, y=上下（负数为上）, z=后前 坐标系
- **相机外参**：`camera_to_world` 变换矩阵
- **点云坐标**：背景/远景点云使用 segment 第一帧的世界坐标系，动态物体点云使用局部坐标系；相机外参和 `dynamic_info` 也会对齐到 segment 第一帧

### 内存管理

- **场景缓存**：最多同时缓存 `preload_scene_count + 1` 个训练场景
- **点云缓存**：按 `(scene_id, segment_id)` 维护，随场景卸载同步清理
- **场景切换**：使用 `mark_scene_completed()` 及时释放已完成场景的内存
- **评估场景**：评估场景可以保留所有，不会自动卸载

### 线程安全

- **调度器**：使用后台线程预加载场景，确保训练队列满
- **场景切换**：场景切换时会阻塞等待，直到场景加载完成
- **资源清理**：使用 `scheduler.shutdown()` 确保后台线程正确停止

---

## 常见问题

### Q: 图像数据范围是多少？

A: 图像已归一化到 `[0.0, 1.0]`（原始图像除以 255），数据类型为 `torch.float32`。

### Q: 深度图不存在怎么办？

A: 如果深度图不存在，会自动创建占位符深度图，值为 `10.0` 米。

### Q: 如何获取段内所有帧？

A: 使用 `dataset.get_segment_frames(scene_id, segment_id)` 方法。

### Q: 如何访问单个帧的数据？

A: 使用 `dataset.get_frame_data(scene_id, frame_idx, cam_idx)` 方法。

### Q: 不同场景的相机数量可以不同吗？

A: 可以，`num_cams` 是从场景数据中动态获取的。

### Q: 如何控制测试帧数量？

A: 通过配置 `pixel_source.max_test_images` 限制每个段中使用的测试帧数量。

### Q: 段 AABB 是如何计算的？batch 里的 aabb 是什么系？

A: 段分割不再为每个段计算/存储 AABB。Batch 中的 **batch['aabb']** 为 **segment 第一帧坐标系 (seg0)**，固定来自 `dataset.segment_aabb`，与 extrinsics、点云一致。

---

## 参考

- [MultiSceneDataset 设计文档](./MultiSceneDataset_Design.md)：详细的设计说明和实现细节
- [DrivingDataset 文档](../README.md)：底层数据集接口说明
- [点云生成器文档](../pointcloud_generators/PointCloud_Generators.md)：点云生成器使用说明
