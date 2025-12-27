# 点云边界框处理方式对比分析

本文档对比分析了 EVolSplat 预处理代码和当前代码库中，对单目点云的边界框（bounding box）处理和外部点的处理方式。

## 重构说明

当前代码库已重构为使用 `crop_aabb` 和 `input_aabb` 参数：
- `crop_aabb`: 用于裁剪时移除超出边界框的点云（shape: [2, 3]，格式：[[x_min, y_min, z_min], [x_max, y_max, z_max]]）
- `input_aabb`: 用于滤波时区分内部和外部点云（shape: [2, 3]，格式相同）

这两个参数都是必需的，不再使用固定的边界框常量。

## 1. EVolSplat 预处理代码中的处理方式

### 1.1 边界框定义

在 `third_party/EVolSplat/preprocess/read_dataset/generate_nuscenes_pcd.py` 中，默认边界框定义为：

```python
X_MIN, X_MAX = -20, 20
Y_MIN, Y_MAX = -20, 4.8
Z_MIN, Z_MAX = -20, 70
```

### 1.2 点云裁剪与分割流程

EVolSplat 预处理代码的处理流程如下：

#### 步骤 1: 裁剪点云（保留背景）

```python
def crop_pointcloud(self, bbx_min, bbx_max, points, color):
    """Crop point cloud to bounding box."""
    mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
        (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
        (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2] + 50)  # Extended Z for background
    
    return points[mask], color[mask]
```

**关键点**：
- Z 方向扩展 50 米用于保留背景点云
- 这是**第一次裁剪**，目的是保留远距离背景

#### 步骤 2: 分割为内部和外部点云

```python
def split_pointcloud(self, bbx_min, bbx_max, points, color):
    """Split point cloud into inside and outside bounding box."""
    mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
        (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
        (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])  # 注意：这里没有扩展
    
    inside_pnt, inside_rgb = points[mask], color[mask]
    outside_pnt, outside_rgb = points[~mask], color[~mask]
    return inside_pnt, inside_rgb, outside_pnt, outside_rgb
```

**关键点**：
- 使用**严格的边界框**（Z 方向不扩展）进行分割
- 分割后的外部点云包含：
  - 在 X、Y 方向超出边界框的点
  - 在 Z 方向超出边界框但仍在 `bbx_max[2] + 50` 范围内的背景点

#### 步骤 3: 分别滤波内部和外部点云

```python
# 内部点云：更严格的滤波
inside_pointcloud = o3d.geometry.PointCloud()
inside_pointcloud.points = o3d.utility.Vector3dVector(inside_pnt[:, :3])
inside_pointcloud.colors = o3d.utility.Vector3dVector(inside_rgb)
cl, ind = inside_pointcloud.remove_statistical_outlier(
    nb_neighbors=35, std_ratio=1.5
)
inside_pointcloud = inside_pointcloud.select_by_index(ind)

# 外部点云：较宽松的滤波
outside_pointcloud = o3d.geometry.PointCloud()
outside_pointcloud.points = o3d.utility.Vector3dVector(outside_pnt[:, :3])
outside_pointcloud.colors = o3d.utility.Vector3dVector(outside_rgb)
cl, ind = outside_pointcloud.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=2.0
)
outside_pointcloud = outside_pointcloud.select_by_index(ind)
```

**滤波参数对比**：

| 区域 | nb_neighbors | std_ratio | 说明 |
|------|--------------|-----------|------|
| 内部点云 | 35 | 1.5 | 更严格，保留更多细节 |
| 外部点云 | 20 | 2.0 | 更宽松，允许更多噪声 |

#### 步骤 4: 合并并下采样

```python
combined_pointcloud = inside_pointcloud + outside_pointcloud
combined_pointcloud = combined_pointcloud.uniform_down_sample(every_k_points=2)
```

**关键点**：
- 合并后的点云进行均匀下采样，每 2 个点保留 1 个

### 1.3 处理逻辑总结

EVolSplat 预处理代码的处理逻辑：

1. **先裁剪**：使用扩展的 Z 边界（+50）保留背景
2. **再分割**：使用严格边界框分割为内部和外部
3. **分别滤波**：内部严格，外部宽松
4. **合并下采样**：统一下采样率（every_k_points=2）

## 2. 当前代码库中的处理方式

### 2.1 边界框定义

在 `datasets/pointcloud_generators/rgb_pointcloud_generator.py` 中，默认边界框定义为：

```python
X_MIN, X_MAX = -20, 20
Y_MIN, Y_MAX = -20, 4.8
Z_MIN, Z_MAX = -20, 70
```

与 EVolSplat 预处理代码**完全一致**。

### 2.2 点云裁剪与分割流程

当前代码库的处理流程如下：

#### 步骤 1: 裁剪点云

```python
def crop_pointcloud(
    self,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    points: np.ndarray,  # [N, 3]
    colors: np.ndarray,  # [N, 3]
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop point cloud to bounding box (remove points outside this box)."""
    mask = (
        (points[:, 0] > crop_min[0]) & (points[:, 0] < crop_max[0]) &
        (points[:, 1] > crop_min[1]) & (points[:, 1] < crop_max[1]) &
        (points[:, 2] > crop_min[2]) & (points[:, 2] < crop_max[2])  # No Z extension
    )
    return points[mask], colors[mask]
```

**重构说明**：
- 移除了 Z 方向自动扩展（+50）逻辑
- 使用 `crop_aabb` 进行严格裁剪
- 在 `generate_pointcloud` 方法中，先调用 `crop_pointcloud` 裁剪点云，然后再调用 `split_pointcloud` 分割

#### 步骤 2: 分割为内部和外部点云

```python
def split_pointcloud(
    self,
    input_min: np.ndarray,
    input_max: np.ndarray,
    points: np.ndarray,  # [N, 3]
    colors: np.ndarray,  # [N, 3]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split point cloud into inside and outside bounding box."""
    mask = (
        (points[:, 0] > input_min[0]) & (points[:, 0] < input_max[0]) &
        (points[:, 1] > input_min[1]) & (points[:, 1] < input_max[1]) &
        (points[:, 2] > input_min[2]) & (points[:, 2] < input_max[2])  # 严格边界框
    )
    inside_points, inside_colors = points[mask], colors[mask]
    outside_points, outside_colors = points[~mask], colors[~mask]
    return inside_points, inside_colors, outside_points, outside_colors
```

**重构说明**：
- 使用 `input_aabb` 进行分割
- 分割后的外部点云包含在 `crop_aabb` 内但不在 `input_aabb` 内的点

#### 步骤 3: 分别滤波内部和外部点云

```python
def filter_pointcloud(
    self,
    pointcloud: o3d.geometry.PointCloud,
    use_bbx: bool = True,
) -> o3d.geometry.PointCloud:
    """Filter point cloud (statistical filter and uniform downsampling)."""
    if use_bbx:
        # 内部点云使用更严格的滤波参数
        cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.5)
        pointcloud = pointcloud.select_by_index(ind)
        pointcloud = pointcloud.uniform_down_sample(every_k_points=2)
    else:
        # 全局滤波
        cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pointcloud = pointcloud.select_by_index(ind)
        pointcloud = pointcloud.uniform_down_sample(every_k_points=5)
```

**关键差异**：

| 区域 | nb_neighbors | std_ratio | every_k_points |
|------|--------------|-----------|----------------|
| 内部点云 | 35 | 1.5 | 2 |
| 外部点云 | 20 | 2.0 | 5 |

**注意**：当前代码库中，外部点云的下采样率是 **every_k_points=5**，而 EVolSplat 预处理代码是 **every_k_points=2**（合并后统一下采样）。

#### 步骤 4: 合并点云

```python
# 在 generate_pointcloud 方法中
# 先裁剪：使用 crop_aabb 移除超出边界框的点
crop_min, crop_max = self.get_crop_aabb()
input_min, input_max = self.get_input_aabb()
points, colors = self.crop_pointcloud(crop_min, crop_max, points, colors)
# 再分割：使用 input_aabb 分割为内部和外部点云
inside_points, inside_colors, outside_points, outside_colors = self.split_pointcloud(
    input_min, input_max, points, colors
)

if len(inside_points) > 0:
    inside_pcd = o3d.geometry.PointCloud()
    inside_pcd.points = o3d.utility.Vector3dVector(inside_points)
    inside_pcd.colors = o3d.utility.Vector3dVector(inside_colors)
    inside_pcd = self.filter_pointcloud(inside_pcd, use_bbx=True)
    
    if len(outside_points) > 0:
        outside_pcd = o3d.geometry.PointCloud()
        outside_pcd.points = o3d.utility.Vector3dVector(outside_points)
        outside_pcd.colors = o3d.utility.Vector3dVector(outside_colors)
        outside_pcd = self.filter_pointcloud(outside_pcd, use_bbx=False)
        
        # 合并内部和外部点云
        pointcloud = inside_pcd + outside_pcd
    else:
        pointcloud = inside_pcd
```

**关键点**：
- **先裁剪**：使用扩展的 Z 边界（+50）移除超出扩展边界框的点
- **再分割**：使用严格边界框分割为内部和外部
- 内部点云使用 `use_bbx=True`（严格滤波 + 下采样率 2）
- 外部点云使用 `use_bbx=False`（宽松滤波 + 下采样率 5）
- **合并后不再进行统一下采样**

### 2.3 处理逻辑总结

当前代码库的处理逻辑：

1. **先裁剪**：使用 `crop_aabb` 移除超出边界框的点（不再有 Z 扩展）
2. **再分割**：使用 `input_aabb` 分割为内部和外部
3. **分别滤波和下采样**：
   - 内部：严格滤波 + 下采样率 2
   - 外部：宽松滤波 + 下采样率 5
4. **合并**：直接合并，不再统一下采样

**重构说明**：
- 使用 `crop_aabb` 和 `input_aabb` 参数，不再使用固定的边界框常量
- 移除了 Z 方向自动扩展逻辑，完全由用户指定的 `crop_aabb` 决定
- 两个边界框可以不同，`crop_aabb` 用于裁剪，`input_aabb` 用于分割和滤波

## 3. 主要差异对比

### 3.1 下采样策略差异

| 项目 | EVolSplat 预处理 | 当前代码库 |
|------|------------------|------------|
| 内部点云下采样 | every_k_points=2 | every_k_points=2 |
| 外部点云下采样 | 无（合并后统一） | every_k_points=5 |
| 合并后下采样 | every_k_points=2 | 无 |

**影响**：
- EVolSplat：内部和外部点云密度一致（都是 2）
- 当前代码库：外部点云密度更低（5），可能丢失更多背景细节

### 3.2 滤波参数一致性

| 参数 | EVolSplat 预处理 | 当前代码库 | 一致性 |
|------|------------------|------------|--------|
| 内部 nb_neighbors | 35 | 35 | ✅ 一致 |
| 内部 std_ratio | 1.5 | 1.5 | ✅ 一致 |
| 外部 nb_neighbors | 20 | 20 | ✅ 一致 |
| 外部 std_ratio | 2.0 | 2.0 | ✅ 一致 |

**结论**：滤波参数**完全一致**。

### 3.3 边界框定义一致性

| 参数 | EVolSplat 预处理 | 当前代码库 | 一致性 |
|------|------------------|------------|--------|
| X_MIN, X_MAX | -20, 20 | -20, 20 | ✅ 一致 |
| Y_MIN, Y_MAX | -20, 4.8 | -20, 4.8 | ✅ 一致 |
| Z_MIN, Z_MAX | -20, 70 | -20, 70 | ✅ 一致 |
| Z 扩展 | +50 | +50 | ✅ 一致 |

**结论**：边界框定义**完全一致**。

## 4. 在训练脚本中的使用

在 `tools/train_evolsplat.py` 中，点云生成器的配置如下：

```python
pointcloud_generator = MonocularRGBPointCloudGenerator(
    chosen_cam_ids=cfg.data.pixel_source.cameras,
    sparsity=cfg.data.pointcloud.get("sparsity", "full"),
    filter_sky=cfg.data.pointcloud.get("filter_sky", True),
    depth_consistency=cfg.data.pointcloud.get("depth_consistency", True),
    use_bbx=cfg.data.pointcloud.get("use_bbx", True),
    downscale=cfg.data.pointcloud.get("downscale", 2),
    crop_aabb=np.array(cfg.data.pointcloud.crop_aabb),
    input_aabb=np.array(cfg.data.pointcloud.input_aabb),
    device=device,
)
```

**关键点**：
- pointcloud 配置存储在 `cfg.data.pointcloud` 下（dataset 配置中）
- `use_bbx=True`：启用边界框处理
- 边界框处理逻辑在 `MonocularRGBPointCloudGenerator.generate_pointcloud()` 中执行
- 训练脚本本身不直接处理边界框，而是通过配置传递给点云生成器

**配置结构**：
```yaml
data:
  # ... other data config ...
  pointcloud:
    sparsity: full
    filter_sky: true
    depth_consistency: true
    use_bbx: true
    downscale: 2
    crop_aabb: [[-20, -20, -20], [20, 4.8, 70]]
    input_aabb: [[-20, -20, -20], [20, 4.8, 70]]
```

## 5. 建议与改进

### 5.1 下采样策略统一

**建议**：考虑将外部点云的下采样率调整为与 EVolSplat 预处理代码一致，或者合并后统一下采样。

**理由**：
- 保持与 EVolSplat 预处理代码的一致性
- 避免外部点云密度过低导致背景细节丢失

**可选方案**：

1. **方案 A**：外部点云也使用 `every_k_points=2`，合并后不再下采样
   ```python
   # 外部点云下采样改为 2
   outside_pcd = self.filter_pointcloud(outside_pcd, use_bbx=False)
   # 但需要修改 filter_pointcloud，使 use_bbx=False 时也使用 every_k_points=2
   ```

2. **方案 B**：外部点云不下采样，合并后统一下采样（与 EVolSplat 一致）
   ```python
   # 外部点云只滤波，不下采样
   outside_pcd = self.filter_pointcloud_without_downsample(outside_pcd, use_bbx=False)
   # 合并后统一下采样
   pointcloud = (inside_pcd + outside_pcd).uniform_down_sample(every_k_points=2)
   ```

### 5.2 代码可读性改进

**建议**：在 `filter_pointcloud` 方法中，明确区分内部和外部点云的滤波参数，而不是通过 `use_bbx` 标志来区分。

**理由**：
- 提高代码可读性
- 便于后续调整参数

## 6. 总结

### 6.1 一致性

- ✅ **边界框定义**：完全一致
- ✅ **滤波参数**：完全一致
- ✅ **处理流程**：基本一致（裁剪 → 分割 → 滤波 → 合并）

### 6.2 差异

- ✅ **裁剪流程**：已修复，现在与 EVolSplat 一致（先裁剪再分割）
- ⚠️ **下采样策略**：外部点云下采样率不同（5 vs 2）
- ⚠️ **合并后处理**：EVolSplat 有统一下采样，当前代码库没有

### 6.3 影响

- 当前代码库的外部点云密度较低，可能影响背景细节的保留
- 建议统一下采样策略以保持一致性

