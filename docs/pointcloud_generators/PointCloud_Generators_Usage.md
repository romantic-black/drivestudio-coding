# Point Cloud Generators 使用文档

## 概述

`datasets/pointcloud_generators` 模块提供了从不同数据源生成RGB点云的功能，支持从单目相机、LiDAR数据以及两者的混合生成带有RGB颜色信息的点云。这些生成器与 `MultiSceneDataset` 集成，可以在训练过程中自动生成点云数据。

---

## 快速开始

### 基本使用

在 `MultiSceneDataset` 中配置点云生成器：

```python
from datasets.multi_scene_dataset import MultiSceneDataset
from omegaconf import OmegaConf

# 配置数据
data_cfg = OmegaConf.create({
    "dataset": "NuScenes",
    "data_root": "/path/to/trainval",
    "pixel_source": {
        "type": "datasets.nuscenes.nuscenes_sourceloader.NuScenesPixelSource",
        # ... 其他配置
    },
})

# 配置点云生成器
pointcloud_config = {
    'type': 'monocular',  # 或 'lidar' 或 'hybrid'
    'chosen_cam_ids': [0, 1, 2],
    'sparsity': 'full',
    'filter_sky': True,
    'depth_consistency': True,
    'downscale': 2,
    'crop_aabb': [[-20, -20, -20], [20, 4.8, 70]],
    'input_aabb': [[-20, -20, -20], [20, 4.8, 120]],
}

# 创建数据集（会自动创建点云生成器）
dataset = MultiSceneDataset(
    data_cfg=data_cfg,
    train_scene_ids=[0, 1, 2],
    eval_scene_ids=[3, 4],
    pointcloud_config=pointcloud_config,  # 传入配置
)

# 获取批次（会自动生成点云）
batch = dataset.get_segment_batch(scene_id=0, segment_id=0)

# 访问点云数据
if 'pointcloud' in batch:
    background_pcd = batch['pointcloud']['background']  # [N, 6] - 背景点云
    dynamic_pcds = batch['pointcloud']['dynamic']         # Dict[int, np.ndarray] - 动态对象点云
```

---

## 生成器类型

### 1. MonocularRGBPointCloudGenerator（单目生成器）

从单目深度图生成RGB点云，通过深度图反投影生成3D点。

**适用场景**：
- 有深度图数据（从文件加载或LiDAR投影）
- 需要高密度的点云
- 需要深度一致性检查

**配置示例**：

```python
pointcloud_config = {
    'type': 'monocular',
    'chosen_cam_ids': [0, 1, 2],  # 使用的相机ID列表
    'sparsity': 'full',            # 帧稀疏度：Drop90/Drop80/Drop50/Drop25/full
    'filter_sky': True,            # 是否过滤天空区域
    'depth_consistency': True,     # 是否进行深度一致性检查
    'downscale': 2,                # 图像下采样比例
    'dynamic_filter': True,        # 必须为 True：先用动态mask清理背景
    # 可选：基于3D bbox回收动态点（推荐）
    'dynamic_recovery': {
        'enable': True,
        'bbox_expand_xyz_m': [0.3, 0.2, 0.5],
        'max_points_per_instance': 3000,
        'assignment': 'first_hit',  # first_hit / nearest_center
    },
    'crop_aabb': [[-20, -20, -20], [20, 4.8, 70]],  # 裁剪AABB
    'input_aabb': [[-20, -20, -20], [20, 4.8, 120]],  # 输入区域AABB
}
```

### 2. LiDARRGBPointCloudGenerator（LiDAR生成器）

从LiDAR数据生成RGB点云，使用多相机图像为LiDAR点着色。

**适用场景**：
- 有LiDAR数据
- 需要稳定的点云（LiDAR点云更稳定）
- 不需要深度一致性检查

**配置示例**：

```python
pointcloud_config = {
    'type': 'lidar',
    'sparsity': 'full',            # 帧稀疏度
    'crop_aabb': [[-20, -20, -20], [20, 4.8, 70]],
    'input_aabb': [[-20, -20, -20], [20, 4.8, 120]],
}
```

**注意**：LiDAR生成器不支持 `filter_sky`、`depth_consistency` 和 `downscale` 参数。

### 3. HybridRGBPointCloudGenerator（混合生成器）

结合LiDAR和单目深度点云，提供多种融合策略。

**适用场景**：
- 同时有LiDAR和深度图数据
- 需要平衡点云密度和质量
- 需要在训练侧控制近/远景初始化点数（通过 `near_max_points` / `distant_max_points`）

**配置示例**：

```python
pointcloud_config = {
    'type': 'hybrid',
    # LiDAR生成器参数
    'lidar_sparsity': 'full',
    # 单目生成器参数
    'monocular_chosen_cam_ids': [0, 1, 2],
    'monocular_sparsity': 'full',
    'monocular_filter_sky': True,
    'monocular_depth_consistency': True,
    'monocular_downscale': 2,
    'monocular_dynamic_recovery_bbox_expand_xyz_m': [0.3, 0.2, 0.5],
    'monocular_dynamic_recovery_max_points_per_instance': 3000,
    # 通用参数
    'crop_aabb': [[-20, -20, -20], [20, 4.8, 70]],
    'input_aabb': [[-20, -20, -20], [20, 4.8, 120]],
}
```

---

## 参数详解

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sparsity` | Literal | "full" | 帧稀疏度：Drop90/Drop80/Drop50/Drop25/full |
| `filter_sky` | bool | True | 是否过滤天空区域（仅单目生成器）；`frame_data['sky_mask']` 为 **1=天空**（与 MultiSceneDataset 一致） |
| `depth_consistency` | bool | True | 是否进行深度一致性检查（仅单目生成器） |
| `downscale` | int | 2 | 图像下采样比例（仅单目生成器） |
| `crop_aabb` | np.ndarray | None | 裁剪用AABB `[[x_min, y_min, z_min], [x_max, y_max, z_max]]` |
| `input_aabb` | np.ndarray | None | 输入区域AABB `[[x_min, y_min, z_min], [x_max, y_max, z_max]]` |

### 单目生成器专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chosen_cam_ids` | List[int] | [0] | 使用的相机ID列表 |
| `dynamic_filter` | bool | True | 是否启用像素级动态mask过滤（当前要求为 `True`）。 |
| `dynamic_recovery.enable` | bool | False | 是否启用基于3D bbox的动态点回收。 |
| `dynamic_recovery.bbox_expand_xyz_m` | List[float] | None | 启用回收时必填，bbox扩张米值 `[dx,dy,dz]`。 |
| `dynamic_recovery.max_points_per_instance` | int | None | 启用回收时必填，单实例点数上限。 |
| `dynamic_recovery.assignment` | Literal | "first_hit" | 重叠bbox分配策略：`first_hit`/`nearest_center`。 |

> 动态mask键固定为 `dynamic_masks`，不再暴露配置项。

### 混合生成器专用参数

| 参数类别 | 参数 | 类型 | 默认值 | 说明 |
|---------|------|------|--------|------|
| **LiDAR参数** | `lidar_sparsity` | Literal | "full" | LiDAR帧稀疏度 |
| **单目参数** | `monocular_chosen_cam_ids` | List[int] | [0] | 单目使用的相机ID |
| | `monocular_sparsity` | Literal | "full" | 单目帧稀疏度 |
| | `monocular_filter_sky` | bool | True | 单目是否过滤天空 |
| | `monocular_depth_consistency` | bool | True | 单目是否深度一致性检查 |
| | `monocular_downscale` | int | 2 | 单目下采样比例 |
| | `monocular_dynamic_recovery_enable` | bool | True | 固定开启单目3D bbox动态点回收（无需配置）。 |
| | `monocular_dynamic_recovery_bbox_expand_xyz_m` | List[float] | None | 回收启用时必填，bbox扩张米值 |
| | `monocular_dynamic_recovery_max_points_per_instance` | int | None | 回收启用时必填，单实例上限 |
| | `monocular_dynamic_recovery_assignment` | Literal | "first_hit" | 固定 `first_hit`（不再暴露配置）。 |
| **融合参数** | - | - | - | 融合策略固定：背景 merge，动态 fuse；不再暴露额外配置。 |

### MultiScene：`static_instance_motion`（仅 `lidar` / `hybrid` 内 LiDAR 分支）

在 [`datasets/multi_scene_dataset.py`](../../datasets/multi_scene_dataset.py) 的 `dataset.pointcloud` 下可配置（与 `DrivingDataset.get_init_objects(..., only_moving=True)` 的轨迹阈值语义对齐）：

| 字段 | 说明 |
|------|------|
| `static_instance_motion.enable` | 是否启用：在 **段内** `frame_indices` 上根据 `instances_pose` 平移轨迹累计长度判定静止实例。 |
| `static_instance_motion.traj_length_thresh_m` | 当 `enable=true` **必填**：累计相邻帧位移范数之和 **≤** 该阈值的实例视为静止。 |

行为：

- **LiDAR 分割**：静止实例不再从 background 抠入 `dynamic`（[`_separate_static_dynamic`](../../datasets/pointcloud_generators/base.py) 跳过对应 intid），框内点留在 **background**（世界坐标）。
- **`dynamic_info`**：[`_build_dynamic_info`](../../datasets/multi_scene_dataset.py) 会排除 `metadata.static_instance_intids` 中的 intid，与点云 `dynamic` 字典一致。
- **单目**：若启用 `dynamic_recovery.enable=true`，会输出 per-instance `dynamic`；否则仍仅有背景。`hybrid` 默认以 LiDAR 动态为主。

若设置 `static_instance_motion` 块，则必须包含 `enable`；`enable=true` 时必须包含 `traj_length_thresh_m`（fast-fail）。

---

## 稀疏度选项

| 选项 | 保留比例 | 采样规则 | 说明 |
|------|---------|---------|------|
| `"full"` | 100% | 使用所有帧 | 最大点云密度 |
| `"Drop25"` | 75% | 每4帧丢弃第3帧 | 轻微稀疏化 |
| `"Drop50"` | 50% | 每4帧保留前2帧 | 中等稀疏化 |
| `"Drop80"` | 20% | 每5帧只保留第1帧 | 高度稀疏化 |
| `"Drop90"` | 10% | 每10帧只保留第1帧 | 极高稀疏化 |

---

## 融合策略（混合生成器）

### merge

简单合并两种点云。

**特点**：
- 保留所有LiDAR点和单目点
- 点云密度最大

**适用场景**：需要最大点云密度

### lidar_first

优先保留所有LiDAR点，剩余配额分配给单目点。

**特点**：
- 优先保留所有LiDAR点
- 单目点补充细节

**适用场景**：LiDAR质量高，单目补充细节

### adaptive（推荐）

LiDAR优先，单目补充稀疏区域。

**特点**：
- 优先保留所有LiDAR点（如果不超过预算）
- 单目点补充LiDAR稀疏区域
- 平衡密度和细节

**适用场景**：平衡点云密度和质量

---

## 下采样方法（混合生成器）

### uniform

均匀下采样。

**特点**：
- 简单快速
- 保持均匀分布
- 可能丢失细节

### density

基于密度的下采样，保留稀疏区域。

**特点**：
- 保留细节
- 计算较慢
- 适合需要保留细节的场景

### distance

基于距离的过滤，体素下采样。

**特点**：
- 保持空间分布
- 中等速度
- 平衡质量和速度

---

## 坐标系定义

### AABB 坐标系

- **x 轴**：左右方向（左为负，右为正）
- **y 轴**：上下方向（上为负，下为正）
- **z 轴**：后前方向（后为负，前为正）

**AABB 格式**：
```python
crop_aabb = np.array([
    [x_min, y_min, z_min],  # 最小值
    [x_max, y_max, z_max]   # 最大值
])
```

**示例**：
```python
# 典型的裁剪AABB
crop_aabb = [[-20, -20, -20], [20, 4.8, 70]]

# 典型的输入AABB（通常比crop_aabb更大）
input_aabb = [[-20, -20, -20], [20, 4.8, 120]]
```

### 点云坐标系

- **背景点云**：世界坐标系（全局固定参考系）
- **动态对象点云**：对象局部坐标系（相对于对象中心）

---

## 输出格式

### 标准输出格式

```python
{
    "background": np.ndarray,              # [N, 6] 背景点云 [x, y, z, r, g, b]
    "dynamic": Dict[int, np.ndarray],      # {intid: [M_i, 6]} 动态对象点云
    "instance_mapping": Dict[int, int],    # {original_id: intid}
    "metadata": Dict                       # 元数据
}
```

### 点云数据格式

- **形状**：`[N, 6]` 或 `[M_i, 6]`
- **数据类型**：`np.float32`
- **内容**：`[x, y, z, r, g, b]`
  - `[x, y, z]`：3D坐标（float32）
  - `[r, g, b]`：RGB颜色（float32，范围 `[0, 255]`）

### 元数据 (metadata)

```python
metadata = {
    "type": "monocular" | "lidar" | "hybrid",  # 生成器类型
    "frame_indices": List[int],                 # 使用的帧索引
    "frames_used": int,                         # 实际使用的帧数
    "sparsity": str,                            # 稀疏度设置
    # 混合生成器额外字段
    "lidar_count": int,                         # LiDAR点数量
    "monocular_count": int,                     # 单目点数量
    "fused_background_count": int,             # 融合后背景点数
    "dynamic_count": int,                        # 动态点总数
    "fusion_strategy": str,                     # 融合策略
}
```

---

## 使用示例

### 示例1：单目生成器

```python
from datasets.multi_scene_dataset import MultiSceneDataset
from omegaconf import OmegaConf

# 配置
data_cfg = OmegaConf.create({
    "dataset": "NuScenes",
    "data_root": "/path/to/trainval",
    "pixel_source": {
        "type": "datasets.nuscenes.nuscenes_sourceloader.NuScenesPixelSource",
        # ... 其他配置
    },
})

pointcloud_config = {
    'type': 'monocular',
    'chosen_cam_ids': [0, 1, 2, 3, 4, 5],  # 使用所有6个相机
    'sparsity': 'Drop50',                   # 使用50%的帧
    'filter_sky': True,
    'depth_consistency': True,
    'downscale': 2,
    'dynamic_filter': True,
    'dynamic_recovery': {
        'enable': True,
        'bbox_expand_xyz_m': [0.3, 0.2, 0.5],
        'max_points_per_instance': 3000,
    },
    'crop_aabb': [[-20, -20, -20], [20, 4.8, 70]],
    'input_aabb': [[-20, -20, -20], [20, 4.8, 120]],
}

dataset = MultiSceneDataset(
    data_cfg=data_cfg,
    train_scene_ids=[0, 1, 2],
    eval_scene_ids=[3, 4],
    pointcloud_config=pointcloud_config,
)

# 获取批次
batch = dataset.get_segment_batch(scene_id=0, segment_id=0)

# 访问点云
if 'pointcloud' in batch:
    background = batch['pointcloud']['background']  # [N, 6]
    dynamic = batch['pointcloud']['dynamic']       # Dict[int, np.ndarray]
    metadata = batch['pointcloud'].get('metadata', {})
    
    print(f"背景点数: {len(background)}")
    print(f"动态对象数: {len(dynamic)}")
    print(f"生成器类型: {metadata.get('type', 'unknown')}")
```

### 示例2：LiDAR生成器

```python
pointcloud_config = {
    'type': 'lidar',
    'sparsity': 'full',
    'crop_aabb': [[-20, -20, -20], [20, 4.8, 70]],
    'input_aabb': [[-20, -20, -20], [20, 4.8, 120]],
}

dataset = MultiSceneDataset(
    data_cfg=data_cfg,
    train_scene_ids=[0, 1, 2],
    eval_scene_ids=[3, 4],
    pointcloud_config=pointcloud_config,
)
```

### 示例3：混合生成器（推荐）

```python
pointcloud_config = {
    'type': 'hybrid',
    # LiDAR参数
    'lidar_sparsity': 'full',
    # 单目参数
    'monocular_chosen_cam_ids': [0, 1, 2],
    'monocular_sparsity': 'Drop50',
    'monocular_filter_sky': True,
    'monocular_depth_consistency': True,
    'monocular_downscale': 2,
    'monocular_dynamic_recovery_bbox_expand_xyz_m': [0.3, 0.2, 0.5],
    'monocular_dynamic_recovery_max_points_per_instance': 3000,
    # 通用参数
    'crop_aabb': [[-20, -20, -20], [20, 4.8, 70]],
    'input_aabb': [[-20, -20, -20], [20, 4.8, 120]],
}

dataset = MultiSceneDataset(
    data_cfg=data_cfg,
    train_scene_ids=[0, 1, 2],
    eval_scene_ids=[3, 4],
    pointcloud_config=pointcloud_config,
)
```

### 示例4：直接使用生成器

```python
from datasets.pointcloud_generators import MonocularRGBPointCloudGenerator
import torch

# 直接创建生成器
generator = MonocularRGBPointCloudGenerator(
    chosen_cam_ids=[0, 1, 2],
    sparsity='full',
    filter_sky=True,
    depth_consistency=True,
    downscale=2,
    dynamic_filter=True,
    dynamic_recovery_enable=True,
    dynamic_recovery_bbox_expand_xyz_m=[0.3, 0.2, 0.5],
    dynamic_recovery_max_points_per_instance=3000,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# 使用生成器
result = generator.generate_pointcloud(
    dataset=dataset,
    scene_id=0,
    segment_id=0,
)

# 访问结果
background = result['background']      # [N, 6]
dynamic = result['dynamic']            # Dict[int, np.ndarray]
instance_mapping = result['instance_mapping']  # Dict[int, int]
metadata = result['metadata']         # Dict
```

### 示例5：使用调度器生成点云

```python
# 创建调度器
scheduler = dataset.create_scheduler(
    batches_per_segment=20,
    segment_order="random",
    scene_order="random",
)

# 为当前段生成点云
if dataset.pointcloud_generator is not None:
    pointcloud = scheduler.generate_segment_pointcloud(
        pointcloud_generator=dataset.pointcloud_generator,
        scene_id=None,  # 使用当前场景
        segment_id=None,  # 使用当前段
    )
    
    print(f"背景点数: {len(pointcloud['background'])}")
    print(f"动态对象数: {len(pointcloud['dynamic'])}")

# 为场景的所有段生成点云
all_pointclouds = scheduler.generate_all_segment_pointclouds(
    pointcloud_generator=dataset.pointcloud_generator,
    scene_id=0,
    save_dir="/path/to/save/pointclouds",  # 可选：保存点云
)

# 访问结果
for segment_id, pointcloud in all_pointclouds.items():
    print(f"段 {segment_id}: 背景点数={len(pointcloud['background'])}, "
          f"动态对象数={len(pointcloud['dynamic'])}")
```

---

## 配置文件示例

### YAML 配置

```yaml
# configs/evolsplat/multi_scene.yaml
data:
  # ... 其他配置
  
  # Point cloud generation configuration
  pointcloud:
    type: hybrid  # monocular, lidar, or hybrid
    # LiDAR参数
    lidar_sparsity: full
    # 单目参数
    monocular_chosen_cam_ids: [0, 1, 2]
    monocular_sparsity: Drop50
    monocular_filter_sky: true
    monocular_depth_consistency: true
    monocular_downscale: 2
    monocular_dynamic_recovery_bbox_expand_xyz_m: [0.3, 0.2, 0.5]
    monocular_dynamic_recovery_max_points_per_instance: 3000
    # AABB配置
    crop_aabb: [[-20, -20, -20], [20, 4.8, 70]]
    input_aabb: [[-20, -20, -20], [20, 4.8, 120]]
```

---

## 数据要求

### MultiSceneDataset 接口要求

点云生成器需要 `MultiSceneDataset` 提供以下接口：

| 方法/属性 | 说明 |
|----------|------|
| `get_scene(scene_id)` | 获取场景数据 |
| `get_segment_frames(scene_id, segment_id)` | 获取段内所有帧索引 |
| `get_frame_data(scene_id, frame_idx, cam_idx)` | 获取指定帧和相机的数据 |

### 像素源 (pixel_source) 要求

| 属性/方法 | 类型 | 说明 |
|----------|------|------|
| `camera_list` | List[int] | 可用相机ID列表 |
| `camera_data[cam_id]` | Dict | 相机数据（包含depth_maps等） |
| `get_image(img_idx)` | Tuple | 返回(image_infos, cam_infos) |
| `instances_pose` | Tensor/Array | 实例姿态 [F, N, 4, 4] |
| `instances_size` | Tensor/Array | 实例尺寸 [N, 3] |
| `per_frame_instance_mask` | Tensor/Array | 每帧实例掩码 [F, N] |
| `instances_true_id` | Tensor/Array | 实例原始ID [N] |

### LiDAR源 (lidar_source) 要求（仅LiDAR生成器）

| 属性 | 类型 | 说明 |
|------|------|------|
| `origins` | Tensor | 射线起点 [M, 3] |
| `directions` | Tensor | 射线方向 [M, 3] |
| `ranges` | Tensor | 射线距离 [M] |
| `timesteps` | Tensor | 时间戳/帧索引 [M] |
| `lidar_to_worlds` | Tensor | 车辆到世界变换 [F, 4, 4] |

---

## 注意事项

### 1. 数据源要求

- **单目生成器**：需要深度图（`depth_maps` 或 `lidar_depth_maps`）
- **LiDAR生成器**：需要 `lidar_source` 属性
- **混合生成器**：需要两者都可用（至少一个）

### 2. 相机参数

- 相机内外参必须正确
- 变换矩阵必须符合坐标系约定

### 3. 实例信息

- 必须包含正确的变换矩阵（`T_ow`）和尺寸（`size_lwh`）
- `per_frame_instance_mask` 必须正确标记每帧的实例

### 4. 颜色值处理

- 颜色值会自动在 `[0, 1]` 和 `[0, 255]` 之间转换
- 最终输出为 `[0, 255]` 范围，float32类型

### 5. 过滤参数调整

- 过滤参数对最终点云质量影响较大
- 需要根据数据特点（密度、噪声水平）调整
- 内部点（input_aabb内）使用严格过滤，外部点使用宽松过滤

### 6. 内存管理

- 背景点云规模由数据本身决定；训练侧可通过 `near_max_points` / `distant_max_points` 控制初始化点数
- 大量帧/多相机时注意内存使用

### 7. 错误处理

- 混合生成器在单个生成器失败时会fallback到另一个
- 两者都失败时会抛出详细错误信息

### 8. 坐标系约定

- **背景点云**：世界坐标系（全局固定参考系）
- **动态对象点云**：对象局部坐标系（相对于对象中心）
- **AABB**：使用 x=左右, y=上下（负数为上）, z=后前 坐标系

---

## 常见问题

### Q: 如何选择生成器类型？

A: 
- **单目生成器**：有深度图数据，需要高密度点云
- **LiDAR生成器**：有LiDAR数据，需要稳定的点云
- **混合生成器**：同时有LiDAR和深度图，需要平衡密度和质量（推荐）

### Q: 如何控制点云大小？

A: 
- 使用 `near_max_points` / `distant_max_points` 控制近/远景初始化点数
- 调整 `sparsity` 参数减少使用的帧数
- 调整 `downscale` 参数（单目生成器）

### Q: 如何选择融合策略？

A: 
- **merge**：需要最大点云密度
- **lidar_first**：LiDAR质量高，单目补充细节
- **adaptive**：平衡密度和质量（推荐）

### Q: 深度一致性检查的作用？

A: 
- 过滤帧间不一致的深度值
- 提高点云质量
- 仅单目生成器支持

### Q: 天空过滤的作用？

A: 
- 过滤天空区域（通常没有深度信息）
- 减少无效点
- 仅单目生成器支持

### Q: 如何保存点云？

A: 
```python
import numpy as np

# 保存背景点云
background = result['background']
np.save('background.npy', background)

# 保存动态对象点云
for intid, points in result['dynamic'].items():
    np.save(f'dynamic_{intid}.npy', points)
```

---

## 参考

- [MultiSceneDataset 使用文档](../dataloader/MultiSceneDataset_Usage.md)：数据集使用说明
- [Point Cloud Generators 技术文档](./PointCloud_Generators.md)：详细的技术文档和实现细节
