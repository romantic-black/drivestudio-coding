# `sample_within_window` 方法详解

## 概述

`sample_within_window` 是 EVolsplat 训练流程中的核心方法之一，用于从多视角源图像中采样局部窗口内的 2D 特征。该方法将 3D 点投影到多个源视图的 2D 图像平面，并在投影点周围采样一个局部窗口（local window）内的 RGB 特征和可见性信息。

**核心功能**：

- 将 3D 点投影到多个源视图的 2D 图像平面
- 在每个投影点周围采样局部窗口内的 RGB 特征
- 计算可见性图（visibility map）用于遮挡感知的图像渲染（IBR）
- 生成有效掩码（valid mask）用于过滤无效采样

**在训练流程中的位置**：

- 调用位置：`EVolsplatTrainer.extract_shared_features()` 方法中
- 执行时机：每个训练步骤开始时，只执行一次（所有 target view 共享）
- 作用：为后续的 MLP 预测提供 2D 图像特征输入

---

## 方法签名

```python
def sample_within_window(
    self,
    xyz: torch.Tensor,                    # [n_samples, 3] - 3D 点坐标
    train_imgs: torch.Tensor,             # [n_views, c, h, w] - 源图像
    train_cameras: torch.Tensor,          # [n_views, 4, 4] - 相机外参（OpenGL 坐标系）
    train_intrinsics: torch.Tensor,       # [n_views, 4, 4] - 相机内参
    source_depth: Optional[torch.Tensor], # [n_views, h, w] - 源图像深度图（可选）
    local_radius: int = 2,                # 局部窗口半径
    depth_delta: float = 0.2,             # 深度差异阈值（用于可见性判断）
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回值：
        rgb_feat_sampled: [n_samples, n_views, local_h*local_w, 3] - 采样的 RGB 特征
        valid_mask: [n_samples, n_views, local_h*local_w] - 有效掩码
        visibility_map: [n_samples, n_views, local_h*local_w, 1] - 可见性图
    """
```

---

## 流程图

```
输入
├─ xyz: [N, 3] - 3D 点坐标
├─ train_imgs: [V, C, H, W] - V 个源视图图像
├─ train_cameras: [V, 4, 4] - 相机外参
├─ train_intrinsics: [V, 4, 4] - 相机内参
├─ source_depth: [V, H, W] - 深度图（可选）
└─ local_radius: R - 窗口半径

步骤 1: 生成窗口网格
├─ local_h = local_w = 2*R + 1
├─ generate_window_grid(-R, R, -R, R, local_h, local_w)
└─ window_grid: [local_h*local_w, 2] → [V, local_h*local_w, 2]

步骤 2: 3D 点投影
├─ compute_projections(xyz, train_cameras, train_intrinsics)
├─ pixel_locations: [V, N, 2] - 投影像素坐标
├─ mask_in_front: [V, N] - 点在相机前方的掩码
└─ project_depth: [V, N] - 投影深度值

步骤 3: 可见性计算（如果提供 source_depth）
├─ 从 source_depth 采样深度值
├─ 计算深度差异: visibility_map = project_depth - retrieved_depth
└─ visibility_map: [V, N] → [V, N, local_h*local_w] → [N, V, local_h*local_w, 1]

步骤 4: 构建局部窗口采样坐标
├─ pixel_locations: [V, N, 2]
├─ window_grid: [V, local_h*local_w, 2]
├─ 广播相加: pixel_locations[:, :, None, :] + window_grid[:, None, :, :]
└─ pixel_locations: [V, N, local_h*local_w, 2] → [V, N*local_h*local_w, 2]

步骤 5: 边界和可见性掩码扩展
├─ mask_in_front: [V, N] → [V, N, local_h*local_w] → [V, N*local_h*local_w]
└─ inbound: [V, N*local_h*local_w] - 边界检查

步骤 6: RGB 特征采样
├─ normalize(pixel_locations) → [-1, 1] 归一化坐标
├─ F.grid_sample(train_imgs, normalized_coords)
└─ rgb_sampled: [N*local_h*local_w, V, 3] → [N, V, local_h*local_w, 3]

步骤 7: 生成有效掩码
├─ valid_mask = inbound & mask_in_front
└─ valid_mask: [N, V, local_h*local_w]

输出
├─ rgb_feat_sampled: [N, V, local_h*local_w, 3]
├─ valid_mask: [N, V, local_h*local_w]
└─ visibility_map: [N, V, local_h*local_w, 1]
```

---

## 关键数据信息

### 输入数据

| 参数                 | 形状                   | 类型                    | 说明                                       |
| -------------------- | ---------------------- | ----------------------- | ------------------------------------------ |
| `xyz`              | `[n_samples, 3]`     | `torch.Tensor`        | 3D 点坐标（世界坐标系）                    |
| `train_imgs`       | `[n_views, c, h, w]` | `torch.Tensor`        | 源视图图像，通常 `c=3`（RGB）            |
| `train_cameras`    | `[n_views, 4, 4]`    | `torch.Tensor`        | 相机外参（OpenGL 坐标系，camera-to-world） |
| `train_intrinsics` | `[n_views, 4, 4]`    | `torch.Tensor`        | 相机内参矩阵                               |
| `source_depth`     | `[n_views, h, w]`    | `torch.Tensor` (可选) | 源视图深度图，用于遮挡感知                 |
| `local_radius`     | `int`                | `int`                 | 局部窗口半径，通常为 1 或 2                |

### 中间变量

| 变量名                         | 形状                                                                       | 说明                                   |
| ------------------------------ | -------------------------------------------------------------------------- | -------------------------------------- |
| `n_views`                    | `int`                                                                    | 源视图数量                             |
| `n_samples`                  | `int`                                                                    | 3D 点数量                              |
| `local_h = local_w`          | `int`                                                                    | 窗口高度/宽度 =`2*local_radius + 1`  |
| `window_grid`                | `[local_h*local_w, 2]` → `[n_views, local_h*local_w, 2]`              | 窗口网格偏移                           |
| `pixel_locations`            | `[n_views, n_samples, 2]` → `[n_views, n_samples*local_h*local_w, 2]` | 投影像素坐标                           |
| `mask_in_front`              | `[n_views, n_samples]` → `[n_views, n_samples*local_h*local_w]`       | 点在相机前方的掩码                     |
| `project_depth`              | `[n_views, n_samples]`                                                   | 投影深度值                             |
| `normalized_pixel_locations` | `[n_views, 1, n_samples*local_h*local_w, 2]`                             | 归一化像素坐标（用于 `grid_sample`） |
| `inbound`                    | `[n_views, n_samples*local_h*local_w]`                                   | 边界检查掩码                           |

### 输出数据

| 返回值               | 形状                                         | 说明                       |
| -------------------- | -------------------------------------------- | -------------------------- |
| `rgb_feat_sampled` | `[n_samples, n_views, local_h*local_w, 3]` | 采样的 RGB 特征            |
| `valid_mask`       | `[n_samples, n_views, local_h*local_w]`    | 有效掩码（边界内且在前方） |
| `visibility_map`   | `[n_samples, n_views, local_h*local_w, 1]` | 可见性图（深度差异）       |

### 维度计算示例

假设：

- `n_samples = 10000`（10K 个 3D 点）
- `n_views = 9`（9 个源视图）
- `local_radius = 1`（窗口半径为 1）
- `c = 3`（RGB 通道）

则：

- `local_h = local_w = 2*1 + 1 = 3`
- `window_size = 3*3 = 9`
- `rgb_feat_sampled`: `[10000, 9, 9, 3]` = 2.43M 个元素
- `valid_mask`: `[10000, 9, 9]` = 810K 个元素
- `visibility_map`: `[10000, 9, 9, 1]` = 810K 个元素

**在 EVolsplat 中的后续处理**：

```python
# 拼接特征和可见性图
sampled_feat = torch.concat([rgb_feat_sampled, visibility_map], dim=-1)
# sampled_feat: [N, V, 9, 4]  (3 RGB + 1 visibility)

# 重塑为特征向量
sampled_feat = sampled_feat.reshape(N, V * 9 * 4)
# sampled_feat: [N, 324]  (9 views * 9 pixels * 4 channels)
```

---

## 关键组件

### 1. `generate_window_grid()` - 生成窗口网格

**功能**：生成局部窗口的网格坐标偏移。

**实现**：

```python
def generate_window_grid(self, h_min, h_max, w_min, w_max, len_h, len_w, device):
    x, y = torch.meshgrid([
        torch.linspace(w_min, w_max, len_w, device=device),
        torch.linspace(h_min, h_max, len_h, device=device),
    ])
    grid = torch.stack((x, y), -1).transpose(0, 1).float()
    return grid  # [H, W, 2]
```

**示例**（`local_radius=1`）：

```
窗口网格（3x3）：
[-1, -1]  [0, -1]  [1, -1]
[-1,  0]  [0,  0]  [1,  0]
[-1,  1]  [0,  1]  [1,  1]
```

**作用**：定义每个投影点周围需要采样的像素位置偏移。

### 2. `compute_projections()` - 3D 点投影

**功能**：将 3D 点投影到多个视图的 2D 图像平面。

**实现流程**：

1. **坐标系转换**：OpenGL 坐标系（`train_cameras`）转换为投影坐标系
2. **投影计算**：
   ```python
   # 齐次坐标
   xyz_h = [xyz, 1]  # [N, 4]

   # 投影变换
   projections = intrinsics @ inv(extrinsics) @ xyz_h.T
   # [V, 4, N]

   # 像素坐标
   pixel_locations = projections[:, :2, :] / projections[:, 2:3, :]
   # [V, N, 2]
   ```
3. **可见性检查**：`mask_in_front = projections[:, 2, :] > 0`（深度 > 0 表示在相机前方）

**输出**：

- `pixel_locations`: `[n_views, n_samples, 2]` - 投影像素坐标
- `mask_in_front`: `[n_views, n_samples]` - 点在相机前方的布尔掩码
- `project_depth`: `[n_views, n_samples]` - 投影深度值

### 3. 可见性图计算（遮挡感知）

**功能**：使用深度图计算遮挡关系，生成可见性图。

**实现**（当 `source_depth` 不为 `None` 时）：

```python
# 从深度图采样深度值
depths_sampled = F.grid_sample(
    source_depth,  # [V, 1, H, W]
    normalized_pixel_locations,  # [V, 1, N, 2]
    align_corners=False
)  # [V, 1, 1, N]

# 计算深度差异
visibility_map = project_depth - retrieved_depth
# visibility_map: [V, N]
# 正值表示点在前方（可见），负值表示被遮挡
```

**可见性图含义**：

- **正值**：投影深度 > 采样深度，点在遮挡物前方（可见）
- **负值**：投影深度 < 采样深度，点被遮挡（不可见）
- **零值**：深度一致（边界情况）

**扩展**：将每个点的可见性值复制到窗口内的所有像素：

```python
visibility_map = visibility_map.unsqueeze(-1).repeat(1, 1, local_h*local_w)
# [V, N, local_h*local_w]
```

### 4. `normalize()` - 像素坐标归一化

**功能**：将像素坐标归一化到 `[-1, 1]` 范围，用于 `F.grid_sample()`。

**实现**：

```python
def normalize(self, pixel_locations, h, w):
    resize_factor = torch.tensor([w - 1.0, h - 1.0]).to(pixel_locations.device)
    normalized = 2 * pixel_locations / resize_factor - 1.0
    return normalized  # [V, N, 2]
```

**坐标映射**：

- `(0, 0)` → `(-1, -1)`（左上角）
- `(w-1, h-1)` → `(1, 1)`（右下角）

### 5. `F.grid_sample()` - 双线性插值采样

**功能**：从图像中采样 RGB 特征（使用双线性插值）。

**输入**：

- `train_imgs`: `[V, C, H, W]` - 源图像
- `normalized_pixel_locations`: `[V, 1, N*local_h*local_w, 2]` - 归一化坐标

**输出**：

- `rgbs_sampled`: `[V, C, 1, N*local_h*local_w]` - 采样的 RGB 特征

**特点**：

- 使用双线性插值，支持亚像素精度采样
- 自动处理边界（`padding_mode="zeros"`）

### 6. `inbound()` - 边界检查

**功能**：检查像素坐标是否在图像边界内。

**实现**：

```python
def inbound(self, pixel_locations, h, w):
    return (
        (pixel_locations[..., 0] <= w - 1.0) &
        (pixel_locations[..., 0] >= 0) &
        (pixel_locations[..., 1] <= h - 1.0) &
        (pixel_locations[..., 1] >= 0)
    )
```

**作用**：过滤掉超出图像边界的采样点。

### 7. 有效掩码生成

**功能**：组合边界检查和前方检查，生成最终的有效掩码。

**实现**：

```python
# 边界检查
inbound = self.inbound(pixel_locations, h, w)  # [V, N*local_h*local_w]

# 前方检查（已扩展）
mask_in_front = mask_in_front.reshape(V, N*local_h*local_w)  # [V, N*local_h*local_w]

# 组合掩码
mask = (inbound & mask_in_front).float()  # [V, N*local_h*local_w]

# 重塑并转置
valid_mask = mask.permute(1, 0)[..., None]  # [N*local_h*local_w, V, 1]
valid_mask = valid_mask.reshape(N, V, local_h*local_w)  # [N, V, local_h*local_w]
```

**有效掩码含义**：

- `1.0`：采样点有效（在边界内且点在相机前方）
- `0.0`：采样点无效（超出边界或被遮挡）

---

## 在 EVolsplat 训练流程中的作用

### 1. 调用位置

```python
# models/trainers/evolsplat.py - extract_shared_features()
sampled_feat, valid_mask, vis_map = self.projector.sample_within_window(
    xyz=means,  # [N, 3] - 3D 点坐标
    train_imgs=source_images.squeeze(0),  # [V, c, h, w]
    train_cameras=source_extrinsics,  # [V, 4, 4]
    train_intrinsics=source_intrinsics,  # [V, 4, 4]
    source_depth=source_depth,  # [V, h, w]
    local_radius=self.local_radius,  # 通常为 1
)
```

### 2. 后续处理

**步骤 1：拼接特征和可见性图**

```python
# sampled_feat: [N, V, (2R+1)^2, 3]
# vis_map: [N, V, (2R+1)^2, 1]
sampled_feat = torch.concat([sampled_feat, vis_map], dim=-1)
# sampled_feat: [N, V, (2R+1)^2, 4]
```

**步骤 2：重塑为特征向量**

```python
# 计算特征维度
feature_dim_in = 4 * num_views * (2 * local_radius + 1) ** 2
# 例如: 4 * 9 * 9 = 324 (当 num_views=9, R=1 时)

# 重塑
sampled_feat = sampled_feat.reshape(N, feature_dim_in)
# sampled_feat: [N, 324]
```

**步骤 3：投影掩码过滤**

```python
# 过滤有效点（至少被 local_radius^2+1 个视角看到）
projection_mask = valid_mask.sum(dim=1) > (local_radius**2 + 1)
# projection_mask: [N] - bool

# 裁剪到有效点
means_crop = means[projection_mask]  # [num_valid_points, 3]
sampled_color = sampled_feat[projection_mask]  # [num_valid_points, 324]
```

### 3. 输入到 MLP 解码器

```python
# render_for_target_view() 中
# 计算观察方向和距离
ob_view = means_crop - camera_position  # [num_valid_points, 3]
ob_dist = ob_view.norm(dim=1, keepdim=True)  # [num_valid_points, 1]
ob_view = ob_view / ob_dist  # [num_valid_points, 3]

# 拼接特征
input_feature = torch.cat([sampled_color, ob_dist, ob_view], dim=-1)
# input_feature: [num_valid_points, 324 + 1 + 3] = [num_valid_points, 328]

# 预测 SH 系数（颜色）
sh = self.gaussion_decoder(input_feature)
# sh: [num_valid_points, feature_dim_out]
```

### 4. 为什么需要局部窗口采样？

**原因 1：鲁棒性**

- 单个像素的 RGB 值可能受噪声影响
- 局部窗口采样提供空间上下文，提高特征鲁棒性

**原因 2：亚像素精度**

- 投影点可能不在整数像素位置
- 窗口采样结合双线性插值，提供亚像素精度的特征

**原因 3：遮挡处理**

- 窗口内的不同像素可能具有不同的可见性
- 可见性图提供遮挡信息，帮助模型学习遮挡关系

### 5. 窗口大小的影响

**`local_radius = 1`**（窗口大小 3x3 = 9 像素）：

- **优点**：计算量小，特征维度适中
- **缺点**：空间上下文有限
- **适用场景**：大多数场景，默认配置

**`local_radius = 2`**（窗口大小 5x5 = 25 像素）：

- **优点**：更大的空间上下文，更鲁棒
- **缺点**：计算量大，特征维度高（`4 * V * 25`）
- **适用场景**：复杂场景，需要更多上下文

**特征维度计算**：

```python
feature_dim_in = 4 * num_views * (2 * local_radius + 1) ** 2

# local_radius = 1: 4 * V * 9 = 36V
# local_radius = 2: 4 * V * 25 = 100V
```

---

## 关键设计决策

### 1. 为什么使用窗口采样而不是单点采样？

**单点采样**（`local_radius = 0`）：

- 只采样投影点本身的 RGB 值
- 特征维度：`4 * V`（例如 `4 * 9 = 36`）
- **问题**：对投影误差敏感，缺乏空间上下文

**窗口采样**（`local_radius >= 1`）：

- 采样投影点周围窗口内的 RGB 值
- 特征维度：`4 * V * (2R+1)^2`（例如 `4 * 9 * 9 = 324`）
- **优势**：提供空间上下文，对投影误差更鲁棒

### 2. 为什么需要可见性图？

**无可见性图**（`source_depth = None`）：

- 所有点被视为可见：`visibility_map = ones`
- **问题**：无法处理遮挡，可能导致渲染错误

**有可见性图**（`source_depth` 提供）：

- 使用深度差异判断遮挡：`visibility_map = project_depth - retrieved_depth`
- **优势**：遮挡感知，提高渲染质量

### 3. 为什么在 `extract_shared_features()` 中只执行一次？

**原因**：

- 3D 点坐标（`means`）和源视图图像在所有 target view 之间共享
- 投影结果和采样特征不依赖于 target view
- 因此可以只计算一次，所有 target view 复用

**节省计算**：

- 假设有 6 个 target views
- 如果每个 target view 都重新计算，需要 6 倍的计算量
- 共享特征提取可以节省约 83% 的计算时间

### 4. 投影掩码过滤的作用

**过滤条件**：

```python
projection_mask = valid_mask.sum(dim=1) > (local_radius**2 + 1)
```

**含义**：

- 只保留至少被 `local_radius^2 + 1` 个视角看到的点
- 例如 `local_radius = 1`：至少被 2 个视角看到
- 例如 `local_radius = 2`：至少被 5 个视角看到

**作用**：

- 过滤掉只在少数视角可见的点（可能是噪声或边界点）
- 提高训练稳定性
- 减少无效计算

---

## 常见问题

### Q1: 为什么 `pixel_locations` 需要 reshape？

**A**: 因为需要为每个投影点采样窗口内的多个像素，所以需要将 `[V, N, 2]` 扩展为 `[V, N*local_h*local_w, 2]`。

### Q2: 可见性图的值范围是多少？

**A**: 可见性图的值是深度差异（`project_depth - retrieved_depth`），没有固定的范围。正值表示可见，负值表示被遮挡。在实际使用中，通常需要归一化或使用阈值。

### Q3: 如果 `source_depth` 为 `None`，可见性图是什么？

**A**: 如果 `source_depth` 为 `None`，可见性图被设置为全 1：

```python
visibility_map = torch.ones_like(project_depth)
```

表示所有点都被视为可见。

### Q4: 窗口采样是否支持不同大小的图像？

**A**: 是的，`normalize()` 方法会根据图像尺寸（`h, w`）自动归一化坐标，因此支持不同大小的图像。

### Q5: 为什么使用 `F.grid_sample()` 而不是直接索引？

**A**: `F.grid_sample()` 支持：

1. **双线性插值**：提供亚像素精度的采样
2. **自动边界处理**：超出边界的坐标自动填充零值
3. **批处理**：可以高效地处理多个视图和多个点

直接索引无法提供这些功能。

---

## 总结

`sample_within_window` 方法是 EVolsplat 训练流程中的关键组件，它：

1. **将 3D 点投影到多视角 2D 图像平面**
2. **在投影点周围采样局部窗口内的 RGB 特征**
3. **计算可见性图用于遮挡感知**
4. **生成有效掩码用于过滤无效采样**

该方法为后续的 MLP 预测提供了丰富的 2D 图像特征，是 feed-forward 3DGS 训练的基础。理解该方法的工作原理对于调试和优化 EVolsplat 训练流程至关重要。
