# Point Cloud Generators 模块文档

## 概述

`datasets/pointcloud_generators` 模块提供了从不同数据源生成RGB点云的功能。该模块包含一个抽象基类和三个具体实现，用于从单目相机、LiDAR数据以及两者的混合生成带有RGB颜色信息的点云。

## 模块结构

```
pointcloud_generators/
├── __init__.py          # 模块导出
├── base.py              # 抽象基类 RGBPointCloudGenerator
├── monocular.py         # 单目相机点云生成器
├── lidar.py             # LiDAR点云生成器
└── hybrid.py            # 混合点云生成器
```

## 类对比表

| 特性 | RGBPointCloudGenerator (基类) | MonocularRGBPointCloudGenerator | LiDARRGBPointCloudGenerator | HybridRGBPointCloudGenerator |
|------|------------------------------|--------------------------------|----------------------------|----------------------------|
| **数据源** | 抽象基类 | 单目深度图 | LiDAR点云 | LiDAR + 单目深度图 |
| **颜色来源** | - | RGB图像 | 多相机RGB图像 | 多相机RGB图像 |
| **点云生成方式** | - | 深度图反投影 | 直接使用LiDAR点 | 融合两种点云 |
| **支持多相机** | - | ✅ (chosen_cam_ids) | ✅ (自动使用所有相机) | ✅ (通过子生成器) |
| **深度一致性检查** | 可配置 | ✅ (可选) | ❌ | 通过单目生成器 |
| **天空过滤** | 可配置 | ✅ (可选) | ❌ | 通过单目生成器 |
| **下采样** | 可配置 | ✅ (downscale参数) | ❌ (downscale=1) | 通过单目生成器 |
| **稀疏度过滤** | ✅ | ✅ | ✅ | ✅ (分别配置) |
| **AABB裁剪** | ✅ | ✅ | ✅ | ✅ |
| **统计过滤** | ✅ | ✅ | ✅ | ✅ |
| **动态对象分离** | ✅ | ✅ | ✅ | ✅ |
| **融合策略** | - | - | - | merge/lidar_first/adaptive |
| **点数限制** | - | - | - | ✅ (max_points) |

## 核心类详解

### RGBPointCloudGenerator (基类)

抽象基类，定义了所有点云生成器的通用接口和工具方法。

#### 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sparsity` | Literal | "full" | 帧稀疏度：Drop90/Drop80/Drop50/Drop25/full |
| `filter_sky` | bool | True | 是否过滤天空区域 |
| `depth_consistency` | bool | True | 是否进行深度一致性检查 |
| `downscale` | int | 2 | 图像下采样比例 |
| `crop_aabb` | np.ndarray | None | 裁剪用AABB [min, max] |
| `input_aabb` | np.ndarray | None | 输入区域AABB [min, max] |

#### 核心工具方法

| 方法 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `crop_pointcloud` | AABB裁剪 | 点坐标、颜色、AABB边界 | 裁剪后的点和颜色 |
| `split_pointcloud` | 点云分割 | 点坐标、颜色、AABB边界 | 内部点和外部点 |
| `filter_pointcloud` | 统计过滤+下采样 | 点坐标、颜色、过滤强度 | 过滤后的点和颜色 |
| `_separate_static_dynamic` | 分离静态/动态 | 世界坐标点、颜色、实例信息 | 背景点云、动态对象点云 |

### MonocularRGBPointCloudGenerator

从单目深度图生成RGB点云，通过深度图反投影生成3D点。

#### 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chosen_cam_ids` | List[int] | [0] | 使用的相机ID列表 |
| `sparsity` | Literal | "full" | 帧稀疏度 |
| `filter_sky` | bool | True | 是否过滤天空 |
| `depth_consistency` | bool | True | 是否进行深度一致性检查 |
| `downscale` | int | 2 | 图像下采样比例 |

#### 核心方法

| 方法 | 功能 |
|------|------|
| `_load_frame_data` | 加载单帧数据（图像、深度、相机参数） |
| `_depth_consistency_check` | 帧间深度一致性检查，生成一致性掩码 |
| `_generate_points_from_frame_data` | 从单帧数据反投影生成3D点 |
| `_get_instances_for_segment` | 获取片段中的实例信息 |

### LiDARRGBPointCloudGenerator

从LiDAR数据生成RGB点云，使用多相机图像为LiDAR点着色。

#### 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sparsity` | Literal | "full" | 帧稀疏度 |
| `filter_sky` | - | False | 固定为False（不支持） |
| `depth_consistency` | - | False | 固定为False（不支持） |
| `downscale` | - | 1 | 固定为1（不支持） |

#### 核心方法

| 方法 | 功能 |
|------|------|
| `_load_lidar_points` | 从LiDAR源加载点云（世界坐标+车辆坐标） |
| `_colorize_lidar_points` | 使用多相机图像为LiDAR点着色 |
| `_get_instances_for_segment` | 获取片段中的实例信息（复用单目逻辑） |

### HybridRGBPointCloudGenerator

混合生成器，结合LiDAR和单目深度点云，提供多种融合策略。

#### 关键参数

| 参数类别 | 参数 | 类型 | 默认值 | 说明 |
|---------|------|------|--------|------|
| **LiDAR参数** | `lidar_sparsity` | Literal | "full" | LiDAR帧稀疏度 |
| **单目参数** | `monocular_chosen_cam_ids` | List[int] | [0] | 单目使用的相机ID |
| | `monocular_sparsity` | Literal | "full" | 单目帧稀疏度 |
| | `monocular_filter_sky` | bool | True | 单目是否过滤天空 |
| | `monocular_depth_consistency` | bool | True | 单目是否深度一致性检查 |
| | `monocular_downscale` | int | 2 | 单目下采样比例 |
| **融合参数** | `max_points` | int | 2000000 | 背景点云总点数预算（近景+远景），用于生成阶段的全局限制 |
| | `near_max_points` | Optional[int] | - | 近景背景点数上限（segment_aabb 内，仅 StreetForward/MultiScene 路径使用） |
| | `distant_max_points` | Optional[int] | - | 远景背景点数上限（segment_aabb 外，仅 StreetForward/MultiScene 路径使用） |
| | `fusion_strategy` | Literal | "adaptive" | 融合策略：merge/lidar_first/adaptive |
| | `dynamic_source` | Literal | "lidar_only" | 动态点来源：lidar_only/fuse |
| | `background_downsample_method` | Literal | "uniform" | 下采样方法：uniform/density/distance |

#### 核心方法

| 方法 | 功能 |
|------|------|
| `_fuse_background_points` | 融合背景点云（按策略） |
| `_fuse_dynamic_objects` | 融合动态对象点云 |
| `_limit_point_count` | 限制点云数量（多种下采样方法） |
| `_select_complementary_points` | 选择补充点（用于adaptive策略） |

## 生成流程

### 单目生成器流程

```
1. 加载场景和片段数据
   ↓
2. 应用稀疏度过滤（选择帧）
   ↓
3. 获取实例信息（动态对象）
   ↓
4. 按相机分组加载帧数据
   ├─ 图像 (RGB)
   ├─ 深度图
   ├─ 相机内外参
   └─ 天空掩码（可选）
   ↓
5. 深度一致性检查（可选）
   ├─ 将当前帧点投影到上一帧
   ├─ 比较深度差异
   └─ 生成一致性掩码
   ↓
6. 逐帧生成点云
   ├─ 应用掩码（天空、一致性、下采样）
   ├─ 深度图反投影到3D（相机坐标系）
   ├─ 转换到世界坐标系
   └─ 提取RGB颜色
   ↓
7. 分离静态背景和动态对象
   ├─ 背景点：保留世界坐标
   └─ 动态点：转换到对象局部坐标
   ↓
8. 应用裁剪和过滤
   ├─ AABB裁剪（如果启用）
   ├─ 分离内部/外部点
   └─ 统计过滤（不同强度）
   ↓
9. 合并所有帧的点云
   ↓
10. 返回结果
```

### LiDAR生成器流程

```
1. 加载场景和片段数据
   ↓
2. 应用稀疏度过滤（选择帧）
   ↓
3. 获取实例信息（动态对象）
   ↓
4. 逐帧处理
   ├─ 加载LiDAR点云
   │  ├─ 世界坐标系点
   │  └─ 车辆坐标系点
   ├─ 多相机着色
   │  ├─ 遍历所有相机
   │  ├─ 将点投影到相机像素坐标
   │  ├─ 从图像采样颜色
   │  └─ 优先未着色点，覆盖已着色点
   ├─ AABB裁剪（如果启用）
   └─ 分离静态背景和动态对象
   ↓
5. 合并所有帧的点云
   ↓
6. 应用过滤
   ├─ 分离内部/外部点
   └─ 统计过滤（不同强度）
   ↓
7. 返回结果
```

### 混合生成器流程

```
1. 并行生成两种点云
   ├─ LiDAR生成器 → lidar_result
   └─ 单目生成器 → monocular_result
   ↓
2. 错误处理
   ├─ 两者都失败 → 抛出错误
   ├─ 仅LiDAR成功 → 返回LiDAR结果（标记为fallback）
   └─ 仅单目成功 → 返回单目结果（标记为fallback）
   ↓
3. 融合背景点云
   ├─ 计算背景点数预算（考虑动态点）
   ├─ 按融合策略融合
   │  ├─ merge: 简单合并
   │  ├─ lidar_first: 优先保留所有LiDAR点
   │  └─ adaptive: LiDAR优先，单目补充稀疏区域
   └─ 应用点数限制（下采样到max_points）
   ↓
4. 融合动态对象点云
   ├─ dynamic_source="lidar_only": 仅使用LiDAR
   └─ dynamic_source="fuse": 合并两种来源
   ↓
5. 动态点下采样（可选）
   ↓
6. 返回融合结果
```

## 关键组件

### 1. 坐标系转换系统

| 坐标系 | 用途 | 转换关系 |
|--------|------|----------|
| **世界坐标系 (world)** | 数据源坐标系（LiDAR/相机/标注实例位姿通常在该系） | 基准坐标系 |
| **Segment 第一帧坐标系 (seg0)** | 训练/裁剪/渲染使用的局部参考系（以 segment 第一帧为原点） | \(T_{s0w}=\mathrm{inv}(T_{ws0})\)，world→seg0 |
| **车辆坐标系 (vehicle)** | 相对于车辆位置，LiDAR原始数据 | T_vw: 车辆→世界 |
| **相机坐标系 (camera)** | 相对于相机位置，用于反投影 | T_cw: 相机→世界 |
| **对象局部坐标系 (local)** | 相对于动态对象中心，动态点云使用 | T_ow: 对象→世界 |

**关键转换**：
- 深度图反投影：像素坐标 → 相机坐标 → 世界坐标
- LiDAR着色：世界坐标 → 相机坐标 → 像素坐标
- **Segment 对齐**：世界坐标 → seg0（用于 background 点云、实例位姿、AABB 裁剪）
- 动态对象分离：世界坐标/seg0 → 对象局部坐标（动态点云输出为 local）

### 2. 掩码系统

| 掩码类型 | 生成器 | 用途 | 生成方式 |
|---------|--------|------|----------|
| **天空掩码** | Monocular | 过滤天空区域 | 从image_infos获取 |
| **深度一致性掩码** | Monocular | 过滤不一致的深度 | 帧间深度比较 |
| **下采样掩码** | Monocular | 降低点云密度 | 按downscale参数生成 |
| **AABB掩码** | 所有 | 裁剪点云范围 | 基于crop_aabb/input_aabb |

### 3. 过滤系统

#### 分层过滤策略

```
原始点云
  ↓
AABB裁剪 (crop_aabb)
  ↓
静态/动态分离
  ↓
AABB分割 (input_aabb)
  ├─ 内部点 → 严格过滤 (nb_neighbors=35, std_ratio=1.5, every_k=2)
  └─ 外部点 → 宽松过滤 (nb_neighbors=20, std_ratio=2.0, every_k=5)
  ↓
最终点云
```

#### 统计过滤参数

| 参数 | 内部点（严格） | 外部点（宽松） | 说明 |
|------|---------------|---------------|------|
| `nb_neighbors` | 35 | 20 | 统计离群点检测的邻居数 |
| `std_ratio` | 1.5 | 2.0 | 标准差比率阈值 |
| `every_k` | 2 | 5 | 均匀下采样步长 |

### 4. 融合系统（混合生成器）

#### 融合策略对比

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **merge** | 简单合并两种点云 | 需要最大点云密度 |
| **lidar_first** | 优先保留所有LiDAR点，剩余配额给单目 | LiDAR质量高，单目补充细节 |
| **adaptive** | LiDAR优先，单目补充稀疏区域 | 平衡密度和细节 |

#### 下采样方法对比

| 方法 | 原理 | 特点 |
|------|------|------|
| **uniform** | 均匀下采样 | 简单快速，保持均匀分布 |
| **density** | 基于密度，保留稀疏区域 | 保留细节，计算较慢 |
| **distance** | 基于距离，体素下采样 | 保持空间分布，中等速度 |

### 5. 实例管理系统

#### 实例信息结构

```python
{
    "intid": int,              # 内部ID（用于索引）
    "original_id": int,        # 原始ID（来自标注数据）
    "T_ow": np.ndarray,        # 对象到世界的变换矩阵 [4, 4]
    "size_lwh": np.ndarray,    # 对象尺寸 [length, width, height]
}
```

#### 实例映射

- `instance_mapping`: `{original_id: intid}` - 原始ID到内部ID的映射
- 用于保持与标注数据的兼容性
- 混合生成器使用LiDAR的映射（更稳定）

## 关键数据

### 输入数据

#### MultiSceneDataset接口要求

| 数据源 | 必需属性/方法 | 说明 |
|--------|--------------|------|
| **场景数据** | `get_scene(scene_id)` | 获取场景数据 |
| **片段数据** | `segments[segment_id]` | 包含frame_indices |
| **像素源** | `pixel_source` | 图像和相机数据 |
| **LiDAR源** | `lidar_source` | LiDAR点云数据（仅LiDAR生成器） |

#### 像素源 (pixel_source) 要求

| 属性/方法 | 类型 | 说明 |
|----------|------|------|
| `camera_list` | List[int] | 可用相机ID列表 |
| `camera_data[cam_id]` | Dict | 相机数据（包含depth_maps等） |
| `get_image(img_idx)` | Tuple | 返回(image_infos, cam_infos) |
| `instances_pose` | Tensor/Array | 实例姿态 [F, N, 4, 4] |
| `instances_size` | Tensor/Array | 实例尺寸 [N, 3] |
| `per_frame_instance_mask` | Tensor/Array | 每帧实例掩码 [F, N] |
| `instances_true_id` | Tensor/Array | 实例原始ID [N] |

#### LiDAR源 (lidar_source) 要求

| 属性 | 类型 | 说明 |
|------|------|------|
| `origins` | Tensor | 射线起点 [M, 3] |
| `directions` | Tensor | 射线方向 [M, 3] |
| `ranges` | Tensor | 射线距离 [M] |
| `timesteps` | Tensor | 时间戳/帧索引 [M] |
| `lidar_to_worlds` | Tensor | 车辆到世界变换 [F, 4, 4] |

### 输出数据

#### 标准输出格式

```python
{
    "background": np.ndarray,              # [N, 6] 背景点云
    "dynamic": Dict[int, np.ndarray],      # {intid: [M, 6]} 动态对象点云
    "instance_mapping": Dict[int, int],    # {original_id: intid}
    "metadata": Dict                       # 元数据
}
```

#### 与 MultiSceneDataset 的衔接

当点云由 **MultiSceneDataset** 在 `get_segment_batch` 中生成并放入 batch 时，训练 batch 中会包含：

- `batch['pointcloud']['background']`：seg0 系下的背景点云；
- 可选的 `batch['pointcloud']['dynamic']` 与 `batch['dynamic_info']`：动态物体点云及其时序姿态信息。


#### 点云数据格式

| 维度 | 说明 | 范围/类型 |
|------|------|-----------|
| `[N, 6]` | N个点，每点6个值 | float32 |
| `[x, y, z]` | 3D坐标 | float32 |
| `[r, g, b]` | RGB颜色 | float32, [0, 255] |

**坐标系约定**：
- 背景点云：**segment 第一帧坐标系 (seg0)**（world→seg0 对齐后输出）
- 动态对象点云：对象局部坐标系

#### segment_first_pose（重要）

生成器在 `generate_pointcloud(..., segment_first_pose=...)` 中会通过 `segment_first_pose` 计算 `world_to_seg0`，以保证：\n
- `crop_aabb` / `input_aabb` 在 seg0 系下生效；\n
- 点云 background 与 batch 内相机外参、dynamic_info 使用同一 seg0 坐标系。\n
\n
因此在 StreetForward/MultiSceneDataset 路径中，**segment_first_pose 必须由数据集侧传入**；缺失会导致无法构建 `world_to_seg0` 或坐标系不一致。

#### 元数据 (metadata)

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | str | 生成器类型：monocular/lidar/hybrid |
| `frame_indices` | List[int] | 使用的帧索引 |
| `frames_used` | int | 实际使用的帧数 |
| `sparsity` | str | 稀疏度设置 |
| `lidar_count` | int | LiDAR点数量（仅hybrid） |
| `monocular_count` | int | 单目点数量（仅hybrid） |
| `fused_background_count` | int | 融合后背景点数（仅hybrid） |
| `dynamic_count` | int | 动态点总数（仅hybrid） |
| `fusion_strategy` | str | 融合策略（仅hybrid） |
| `max_points` | int | 最大点数（仅hybrid） |

### 中间数据

#### 帧数据 (frame_data) - 单目生成器

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | Tensor/Array | RGB图像 [H, W, 3] |
| `depth` | Tensor/Array | 深度图 [H, W] |
| `extrinsic` | Tensor/Array | 相机到世界变换 [4, 4] |
| `intrinsic` | Tensor/Array | 相机内参 [3, 3] 或 [4, 4] |
| `sky_mask` | Tensor/Array | 天空掩码 [H, W] (可选) |

#### 一致性掩码 - 单目生成器

- 形状：`[H, W]`，bool类型
- 生成方式：将当前帧点投影到上一帧，比较深度差异
- 阈值：深度差异 < 平均深度差异

#### LiDAR点云数据

| 数据 | 类型 | 说明 |
|------|------|------|
| `points_world` | np.ndarray | 世界坐标点 [N, 3] |
| `points_vehicle` | np.ndarray | 车辆坐标点 [N, 3] |
| `points_world_rgb` | np.ndarray | 带颜色的世界坐标点 [N, 6] |
| `points_vehicle_rgb` | np.ndarray | 带颜色的车辆坐标点 [N, 6] |

## 稀疏度选项详解

| 选项 | 保留比例 | 采样规则 | 说明 |
|------|---------|---------|------|
| `"full"` | 100% | 使用所有帧 | 最大点云密度 |
| `"Drop25"` | 75% | 每4帧丢弃第3帧 | 轻微稀疏化 |
| `"Drop50"` | 50% | 每4帧保留前2帧 | 中等稀疏化 |
| `"Drop80"` | 20% | 每5帧只保留第1帧 | 高度稀疏化 |
| `"Drop90"` | 10% | 每10帧只保留第1帧 | 极高稀疏化 |

## 设计决策

### 1. 坐标系分离策略

- **背景点云使用世界坐标系**：便于全局场景重建和融合
- **动态对象点云使用局部坐标系**：便于对象级别的处理和变换
- **转换时机**：在分离静态/动态时进行坐标转换

### 2. 分层过滤策略

- **先裁剪后过滤**：AABB裁剪 → 静态/动态分离 → 统计过滤
- **内外不同强度**：内部点（input_aabb内）使用严格过滤，外部点使用宽松过滤
- **原因**：内部点需要高质量，外部点保留更多信息

### 3. 颜色处理策略

- **内部存储**：`[0, 255]` 范围，float32类型
- **Open3D交互**：临时转换为 `[0, 1]` 范围
- **自动检测**：根据最大值自动判断是否需要转换

### 4. 融合策略设计

- **LiDAR优先**：LiDAR点云更稳定，优先保留
- **单目补充**：单目点云补充细节和稀疏区域
- **点数控制**：通过max_points限制背景点云大小，避免内存问题
- **动态点独立**：动态点默认不计入max_points，保持对象完整性

## 依赖项

| 库 | 用途 |
|----|------|
| `numpy` | 数值计算和数组操作 |
| `torch` | 张量操作（与数据集交互） |
| `open3d` | 点云处理和过滤（统计过滤、下采样） |
| `datasets.multi_scene_dataset.MultiSceneDataset` | 数据集接口 |

## 注意事项

1. **数据源要求**：
   - 单目生成器需要深度图（depth_maps或lidar_depth_maps）
   - LiDAR生成器需要lidar_source属性
   - 混合生成器需要两者都可用（至少一个）

2. **相机参数**：
   - 相机内外参必须正确
   - 变换矩阵必须符合坐标系约定

3. **实例信息**：
   - 必须包含正确的变换矩阵（T_ow）和尺寸（size_lwh）
   - per_frame_instance_mask必须正确标记每帧的实例

4. **颜色值处理**：
   - 颜色值会自动在 `[0, 1]` 和 `[0, 255]` 之间转换
   - 最终输出为 `[0, 255]` 范围

5. **过滤参数调整**：
   - 过滤参数对最终点云质量影响较大
   - 需要根据数据特点（密度、噪声水平）调整
   - 内部点使用严格过滤，外部点使用宽松过滤

6. **内存管理**：
   - 混合生成器通过max_points限制背景点云大小
   - 动态点默认不计入max_points限制
   - 大量帧时注意内存使用

7. **错误处理**：
   - 混合生成器在单个生成器失败时会fallback到另一个
   - 两者都失败时会抛出详细错误信息
