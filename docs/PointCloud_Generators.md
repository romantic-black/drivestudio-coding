# Point Cloud Generators 模块文档

## 概述

`datasets/pointcloud_generators` 模块提供了从不同数据源生成RGB点云的功能。该模块包含一个抽象基类和两个具体实现，用于从单目相机和LiDAR数据生成带有RGB颜色信息的点云。

## 模块结构

```
pointcloud_generators/
├── __init__.py          # 模块导出
├── base.py              # 抽象基类 RGBPointCloudGenerator
├── monocular.py         # 单目相机点云生成器
└── lidar.py             # LiDAR点云生成器
```

## 核心类

### RGBPointCloudGenerator (基类)

位于 `base.py`，是所有点云生成器的抽象基类。

#### 初始化参数

- `sparsity`: 点云稀疏度选项，可选值：`"Drop90"`, `"Drop80"`, `"Drop50"`, `"Drop25"`, `"full"`，默认为 `"full"`
- `filter_sky`: 是否过滤天空区域，默认为 `True`
- `depth_consistency`: 是否进行深度一致性检查，默认为 `True`
- `use_bbx`: 是否使用边界框进行过滤，默认为 `True`
- `downscale`: 下采样比例，默认为 `2`
- `crop_aabb`: 裁剪用的轴对齐包围盒 (AABB)，格式为 `[min, max]`，可选
- `input_aabb`: 输入区域用的AABB，格式为 `[min, max]`，可选
- `device`: PyTorch设备，默认为 `cpu`

#### 抽象方法

##### `generate_pointcloud(dataset, scene_id, segment_id) -> Dict`

生成RGB点云的核心方法，子类必须实现。

**输入参数：**
- `dataset`: `MultiSceneDataset` 实例
- `scene_id`: 场景ID
- `segment_id`: 片段ID

**返回格式：**
```python
{
    "background": np.ndarray,        # [N, 6] 背景点云 (世界坐标系 xyz + rgb)
    "dynamic_objects": Dict[int, np.ndarray],  # {intid: [M, 6]} 动态对象点云 (局部坐标系 xyz + rgb)
    "instance_mapping": Dict[int, int],        # {原始id: intid} 实例ID映射
    "metadata": Dict                           # 可选的元数据
}
```

**点云格式说明：**
- 背景点云使用世界坐标系
- 动态对象点云使用对象局部坐标系
- 颜色值范围：[0, 255]，类型为 `float32`

#### 工具方法

##### `crop_pointcloud(crop_min, crop_max, points, colors) -> Tuple[np.ndarray, np.ndarray]`

根据AABB裁剪点云，返回裁剪后的点和颜色。

##### `split_pointcloud(input_min, input_max, points, colors) -> Tuple`

将点云分为AABB内部和外部两部分，返回：
- `inside_points`, `inside_colors`: AABB内的点
- `outside_points`, `outside_colors`: AABB外的点

##### `filter_pointcloud(points, colors, use_bbx=True) -> Tuple[np.ndarray, np.ndarray]`

对点云进行统计离群点移除和均匀下采样：
- 移除NaN和Inf值
- 使用Open3D进行统计离群点移除
- 均匀下采样

**过滤参数：**
- `use_bbx=True`: 内部点使用更严格的过滤（nb_neighbors=35, std_ratio=1.5, every_k=2）
- `use_bbx=False`: 外部点使用较宽松的过滤（nb_neighbors=20, std_ratio=2.0, every_k=5）

##### `_separate_static_dynamic(points_world, colors, instances) -> Tuple`

将点云分离为静态背景和动态对象：
- 背景点：保留在世界坐标系
- 动态点：转换为对象的局部坐标系

### MonocularRGBPointCloudGenerator

位于 `monocular.py`，从单目深度图生成RGB点云。

#### 特点

- 支持多相机选择（`chosen_cam_ids`）
- 通过深度图反投影生成3D点
- 支持深度一致性检查
- 支持天空区域过滤
- 支持稀疏帧采样

#### 主要方法

##### `generate_pointcloud()`

生成流程：
1. 加载场景和片段数据
2. 应用稀疏度过滤选择帧
3. 获取实例信息（动态对象）
4. 为每个选定的相机加载帧数据（图像、深度、相机参数）
5. 进行深度一致性检查（可选）
6. 从每个帧生成点云：
   - 反投影深度图到3D空间
   - 应用天空掩码和一致性掩码
   - 应用下采样
7. 分离静态背景和动态对象
8. 应用裁剪和过滤
9. 合并所有帧的点云

##### `_depth_consistency_check(frame_data_list, H, W) -> List[np.ndarray]`

进行帧间深度一致性检查：
- 将当前帧的点投影到上一帧
- 比较深度值的差异
- 返回一致性掩码

##### `_generate_points_from_frame_data(frame_data, consistency_mask, downscale_mask) -> Tuple`

从单帧数据生成3D点：
1. 提取RGB、深度、相机内外参
2. 应用各种掩码（天空、一致性、下采样）
3. 反投影像素到3D空间（相机坐标系 -> 世界坐标系）
4. 返回世界坐标点和RGB颜色

##### `_get_instances_for_segment() -> Tuple`

获取片段中所有帧的实例信息，返回实例映射和每帧的实例列表。

### LiDARRGBPointCloudGenerator

位于 `lidar.py`，从LiDAR数据生成RGB点云。

#### 特点

- 直接从LiDAR点云数据生成
- 使用多相机图像为LiDAR点着色
- 不支持天空过滤（`filter_sky=False`）
- 不支持深度一致性检查（`depth_consistency=False`）
- 不支持下采样（`downscale=1`）

#### 主要方法

##### `generate_pointcloud()`

生成流程：
1. 加载场景和片段数据
2. 应用稀疏度过滤
3. 获取实例信息
4. 对每个帧：
   - 加载LiDAR点云（世界坐标系和车辆坐标系）
   - 使用多相机图像为点着色
   - 裁剪点云（如果启用）
   - 分离静态和动态对象
5. 合并所有帧的点云
6. 应用过滤（内部/外部点分别处理）

##### `_load_lidar_points(dataset, scene_id, frame_idx) -> Tuple`

从LiDAR源加载点云：
- 需要 `lidar_source` 具有属性：`origins`, `directions`, `ranges`, `timesteps`
- 返回世界坐标系和车辆坐标系的点

##### `_colorize_lidar_points(dataset, scene_id, frame_idx, points_vehicle, points_world) -> Tuple`

使用多相机图像为LiDAR点着色：
1. 遍历所有相机
2. 将点投影到相机像素坐标
3. 从图像中采样颜色
4. 优先使用未着色的点，如果所有相机都处理过，则覆盖
5. 返回带颜色的点（车辆坐标系和世界坐标系）

## 稀疏度选项说明

- `"full"`: 使用所有帧
- `"Drop25"`: 每4帧中丢弃第3帧（保留75%）
- `"Drop50"`: 每4帧中保留前2帧（保留50%）
- `"Drop80"`: 每5帧中只保留第1帧（保留20%）
- `"Drop90"`: 每10帧中只保留第1帧（保留10%）

## 数据格式约定

### 坐标系

- **世界坐标系 (world)**: 全局固定坐标系
- **车辆坐标系 (vehicle)**: 相对于车辆的位置
- **相机坐标系 (camera)**: 相对于相机的坐标系
- **对象局部坐标系 (local)**: 相对于动态对象中心的坐标系

### 点云格式

- 形状：`[N, 6]`，其中6个维度为 `[x, y, z, r, g, b]`
- 数据类型：`np.float32`
- 颜色范围：`[0, 255]`
- 背景点云：世界坐标系
- 动态对象点云：对象局部坐标系

### 实例信息格式

```python
{
    "intid": int,              # 内部ID
    "original_id": int,        # 原始ID
    "T_ow": np.ndarray,        # 对象到世界的变换矩阵 [4, 4]
    "size_lwh": np.ndarray,    # 对象尺寸 [length, width, height]
}
```

## 使用示例

### 单目相机生成器

```python
from datasets.pointcloud_generators import MonocularRGBPointCloudGenerator

generator = MonocularRGBPointCloudGenerator(
    chosen_cam_ids=[0, 1],          # 使用相机0和1
    sparsity="Drop50",              # 保留50%的帧
    filter_sky=True,                # 过滤天空
    depth_consistency=True,         # 启用深度一致性检查
    use_bbx=True,                   # 使用边界框过滤
    downscale=2,                    # 2倍下采样
)

result = generator.generate_pointcloud(
    dataset=my_dataset,
    scene_id=0,
    segment_id=0,
)

background = result["background"]          # [N, 6]
dynamic_objects = result["dynamic_objects"]  # {intid: [M, 6]}
instance_mapping = result["instance_mapping"]
metadata = result["metadata"]
```

### LiDAR生成器

```python
from datasets.pointcloud_generators import LiDARRGBPointCloudGenerator

generator = LiDARRGBPointCloudGenerator(
    sparsity="full",
    use_bbx=True,
    crop_aabb=np.array([[-50, -50, -5], [50, 50, 5]]),  # 裁剪范围
    input_aabb=np.array([[-30, -30, -3], [30, 30, 3]]), # 输入范围
)

result = generator.generate_pointcloud(
    dataset=my_dataset,
    scene_id=0,
    segment_id=0,
)
```

## 关键设计决策

1. **坐标系分离**：背景点使用世界坐标系便于全局重建，动态对象使用局部坐标系便于对象级别的处理

2. **分层过滤**：先裁剪（AABB），再分离静态/动态，最后统计过滤，确保不同区域使用不同的过滤强度

3. **颜色归一化**：内部处理时保持 `[0, 255]` 范围，与Open3D交互时临时转换为 `[0, 1]`

4. **多帧融合**：通过稀疏度选项支持多帧融合，提高点云密度和覆盖范围

5. **实例映射**：维护原始ID到内部ID的映射，保持与标注数据的兼容性

## 依赖项

- `numpy`: 数值计算
- `torch`: 张量操作
- `open3d`: 点云处理和过滤
- `datasets.multi_scene_dataset.MultiSceneDataset`: 数据集接口

## 注意事项

1. 确保数据集具有相应的数据源（深度图或LiDAR数据）
2. 相机参数和变换矩阵必须正确
3. 实例信息必须包含正确的变换矩阵和尺寸
4. 颜色值会自动在 `[0, 1]` 和 `[0, 255]` 之间转换
5. 过滤参数对最终点云质量影响较大，需要根据数据特点调整
