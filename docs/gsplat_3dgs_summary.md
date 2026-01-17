# gsplat 3DGS 实现总结

本文档总结 `third_party/gsplat/gsplat` 中 3D Gaussian Splatting (3DGS) 的核心实现，包括输入输出、关键数据、关键组件和算法流程。

**注意**：本文档仅涵盖 3DGS，不包含 2DGS 相关内容。

---

## 目录

1. [概述](#概述)
2. [核心函数接口](#核心函数接口)
3. [输入输出](#输入输出)
4. [关键数据结构](#关键数据结构)
5. [关键组件](#关键组件)
6. [算法流程](#算法流程)
7. [数学公式](#数学公式)

---

## 概述

gsplat 是一个用于 CUDA 加速的 3D 高斯点光栅化库，基于论文 "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields" 实现，但具有更高的效率和更多特性。

### 核心特性

- **高效内存使用**：相比官方实现，训练内存占用可减少 4 倍
- **快速训练**：在 MipNeRF 360 数据集上训练时间可减少 15%
- **批处理渲染**：支持一次渲染多张图像
- **N-D 特征渲染**：支持任意维度的特征渲染
- **深度渲染**：支持深度图渲染
- **稀疏梯度**：支持稀疏梯度存储以节省内存
- **多 GPU 分布式**：支持多 GPU 分布式光栅化
- **相机畸变支持**：支持 pinhole、ortho、fisheye、ftheta 等相机模型
- **3DGUT 支持**：支持非线性相机投影和滚动快门效果

---

## 核心函数接口

### `rasterization()` 函数

**位置**：`gsplat/rendering.py`

**功能**：将一组 3D 高斯点（N 个）光栅化到一批图像平面（C 个）

```python
def rasterization(
    means: Tensor,      # [..., N, 3]
    quats: Tensor,      # [..., N, 4]
    scales: Tensor,     # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,     # [..., (C,) N, D] or [..., (C,) N, K, 3]
    viewmats: Tensor,   # [..., C, 4, 4]
    Ks: Tensor,         # [..., C, 3, 3]
    width: int,
    height: int,
    ...
) -> Tuple[Tensor, Tensor, Dict]:
```

**返回**：
- `render_colors`: 渲染的颜色图像 [..., C, height, width, X]
- `render_alphas`: 渲染的 alpha 通道 [..., C, height, width, 1]
- `meta`: 包含光栅化中间结果的字典

---

## 输入输出

### 输入参数

#### 必需参数

| 参数 | 形状 | 描述 |
|------|------|------|
| `means` | `[..., N, 3]` | 3D 高斯点的中心位置（世界坐标系） |
| `quats` | `[..., N, 4]` | 四元数（wxyz 约定），表示旋转，不需要归一化 |
| `scales` | `[..., N, 3]` | 缩放向量，表示 3D 高斯点在三个主轴上的尺度 |
| `opacities` | `[..., N]` | 不透明度，取值范围 [0, 1] |
| `colors` | `[..., (C,) N, D]` 或 `[..., (C,) N, K, 3]` | 颜色/特征。如果 `sh_degree=None`，则为后激活颜色值；否则为 SH 系数 |
| `viewmats` | `[..., C, 4, 4]` | 世界坐标系到相机坐标系的变换矩阵 |
| `Ks` | `[..., C, 3, 3]` | 相机内参矩阵 |
| `width` | `int` | 图像宽度（像素） |
| `height` | `int` | 图像高度（像素） |

#### 可选参数（重要）

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `sh_degree` | `int \| None` | `None` | Spherical Harmonics 阶数。如果为 `None`，`colors` 为后激活颜色；否则为 SH 系数 |
| `packed` | `bool` | `True` | 是否使用打包模式（更省内存但可能稍慢） |
| `tile_size` | `int` | `16` | 瓦片大小（像素），用于光栅化 |
| `render_mode` | `str` | `"RGB"` | 渲染模式：`"RGB"`, `"D"`, `"ED"`, `"RGB+D"`, `"RGB+ED"` |
| `sparse_grad` | `bool` | `False` | 是否使用稀疏梯度（COO 格式） |
| `absgrad` | `bool` | `False` | 是否计算投影 2D 均值的绝对梯度 |
| `return_transmittances` | `bool` | `False` | 是否在 `meta` 中返回每像素的传输率（transmittances） |
| `rasterize_mode` | `str` | `"classic"` | 光栅化模式：`"classic"` 或 `"antialiased"` |
| `radius_clip` | `float` | `0.0` | 2D 半径小于等于此值的高斯点会被跳过（像素单位） |
| `eps2d` | `float` | `0.3` | 添加到投影 2D 协方差矩阵特征值的 epsilon |
| `camera_model` | `str` | `"pinhole"` | 相机模型：`"pinhole"`, `"ortho"`, `"fisheye"`, `"ftheta"` |
| `with_ut` | `bool` | `False` | 是否使用 Unscented Transform (UT) 进行投影（3DGUT） |
| `with_eval3d` | `bool` | `False` | 是否在 3D 世界空间计算高斯响应（而非 2D 图像空间） |
| `distributed` | `bool` | `False` | 是否使用多 GPU 分布式渲染 |

### 输出

#### `render_colors`
- **形状**：`[..., C, height, width, X]`
- **描述**：渲染的颜色/特征图像
- **X 的取值**：
  - `render_mode="RGB"`: X = D（颜色通道数）
  - `render_mode="D"` 或 `"ED"`: X = 1（深度通道）
  - `render_mode="RGB+D"` 或 `"RGB+ED"`: X = D + 1（颜色 + 深度）

#### `render_alphas`
- **形状**：`[..., C, height, width, 1]`
- **描述**：渲染的 alpha 通道（累积不透明度）

#### `meta` 字典
包含光栅化的中间结果，主要包括：

| 键 | 形状/类型 | 描述 |
|----|----------|------|
| `batch_ids` | `Tensor \| None` | 批次 ID（packed 模式）。形状：`[nnz]` 或 `None` |
| `camera_ids` | `Tensor \| None` | 相机 ID（packed 模式）。形状：`[nnz]` 或 `None` |
| `gaussian_ids` | `Tensor \| None` | 高斯点 ID（packed 模式）。形状：`[nnz]` 或 `None` |
| `radii` | `Tensor` | 2D 半径（边界框半径，像素单位）。形状：packed: `[nnz, 2]`, 非packed: `[..., C, N, 2]` |
| `means2d` | `Tensor` | 投影后的 2D 均值（图像坐标）。形状：packed: `[nnz, 2]`, 非packed: `[..., C, N, 2]` |
| `depths` | `Tensor` | 深度值（相机坐标系 z 坐标）。形状：packed: `[nnz]`, 非packed: `[..., C, N]` |
| `conics` | `Tensor` | 投影后的 2D 协方差的 conic 表示 `[a, b, c]`。形状：packed: `[nnz, 3]`, 非packed: `[..., C, N, 3]` |
| `opacities` | `Tensor` | 不透明度。形状：packed: `[nnz]`, 非packed: `[..., C, N]` |
| `transmittances` | `Tensor \| None` | 每像素的传输率（当 `return_transmittances=True` 时）。形状：`[..., C, height, width]` 或 `None` |
| `tile_width` | `int` | 瓦片宽度（瓦片数量） |
| `tile_height` | `int` | 瓦片高度（瓦片数量） |
| `tiles_per_gauss` | `Tensor` | 每个高斯点覆盖的瓦片数。形状：`[..., N]` 或 `[nnz]` |
| `isect_ids` | `Tensor` | 交集 ID（高斯点与瓦片的交集索引）。形状：`[n_isects]` |
| `flatten_ids` | `Tensor` | 扁平化 ID（用于光栅化的高斯点索引）。形状：`[n_isects]` |
| `isect_offsets` | `Tensor` | 交集偏移量（每个瓦片的交集起始位置）。形状：`[..., C, tile_height, tile_width]` |
| `width` | `int` | 图像宽度（像素） |
| `height` | `int` | 图像高度（像素） |
| `tile_size` | `int` | 瓦片大小（像素，默认 16） |
| `n_batches` | `int` | 批次数量（如果有） |
| `n_cameras` | `int` | 相机数量 |

**注意**：
- **packed 模式**：只存储可见的高斯点（`radii > 0`），形状为 `[nnz, ...]`，其中 `nnz` 为可见高斯点数量
- **非 packed 模式**：存储所有高斯点，形状为 `[..., C, N, ...]`，但只有 `radii > 0` 的元素有效
- **`transmittances`**：当 `return_transmittances=True` 时，`meta["transmittances"]` 包含每像素的传输率（传输率为光线穿透所有高斯点后的剩余能量，`T = Π_i (1 - α_i)`）。如果未在光栅化时计算，会从 `render_alphas` 计算：`transmittances = 1.0 - render_alphas[..., 0]`

---

## 关键数据结构

### 1. 3D 高斯点参数化

#### 均值（Mean）
- **表示**：`means` - `[..., N, 3]`
- **含义**：3D 高斯点在世界坐标系中的中心位置 `μ ∈ ℝ³`

#### 协方差矩阵（Covariance）
- **参数化**：通过四元数 `q` 和缩放向量 `s` 表示
- **公式**：`Σ = R * S * S^T * R^T`
  - `R` 是旋转矩阵（由四元数 `q` 计算）
  - `S = diag(s)` 是对角缩放矩阵
- **存储**：
  - `quats`: `[..., N, 4]` - 四元数（wxyz）
  - `scales`: `[..., N, 3]` - 缩放向量
  - 可选：`covars`: `[..., N, 3, 3]` - 预计算的协方差矩阵（上三角形式为 `[..., N, 6]`）

#### 不透明度（Opacity）
- **表示**：`opacities` - `[..., N]`
- **取值范围**：`[0, 1]`

#### 颜色/特征（Color/Features）
- **表示方式 1**：后激活颜色值
  - 形状：`[..., N, D]` 或 `[..., C, N, D]`
  - `D` 为特征维度
- **表示方式 2**：Spherical Harmonics 系数
  - 形状：`[..., N, K, 3]` 或 `[..., C, N, K, 3]`
  - `K = (sh_degree + 1)²` 为 SH 基函数数量
  - 渲染时根据视角方向动态计算颜色

### 2. 投影后的 2D 数据结构

#### 2D 均值（`means2d`）
- **形状**：`[..., N, 2]` 或 `[nnz, 2]`（packed 模式）
- **含义**：投影到图像平面的 2D 中心位置 `μ' ∈ ℝ²`

#### 2D 协方差（`conics`）
- **形状**：`[..., N, 3]` 或 `[nnz, 3]`（packed 模式）
- **表示**：使用 conic 矩阵的上三角元素 `[a, b, c]` 表示 2D 协方差
  ```
  Σ' = [a b]
       [b c]
  ```

#### 深度（`depths`）
- **形状**：`[..., N]` 或 `[nnz]`（packed 模式）
- **含义**：相机坐标系下的深度值 `z`

#### 半径（`radii`）
- **形状**：`[..., N, 2]` 或 `[nnz, 2]`（packed 模式）
- **含义**：投影后 2D 高斯的边界框半径（像素单位），用于瓦片交集计算

### 3. 瓦片数据结构

#### 瓦片网格
- **瓦片数量**：`tile_width × tile_height`
- **瓦片大小**：`tile_size × tile_size` 像素（默认 16×16）

#### 交集信息
- **`isect_ids`**：高斯点与瓦片的交集 ID
  - 形状：`[n_isects]`
  - 含义：标识每个交集属于哪个图像/瓦片
  
- **`flatten_ids`**：扁平化的高斯点索引
  - 形状：`[n_isects]`
  - 含义：标识每个交集对应的高斯点（在 packed 模式下为 `nnz` 范围内的索引，在非 packed 模式下为 `N` 范围内的索引）
  
- **`isect_offsets`**：每个瓦片的交集起始偏移量
  - 形状：`[..., C, tile_height, tile_width]`
  - 含义：标识每个瓦片的第一个交集的索引位置，用于快速定位该瓦片内的所有高斯点
  
- **`tiles_per_gauss`**：每个高斯点覆盖的瓦片数量
  - 形状：`[..., N]`（非 packed）或 `[nnz]`（packed）
  - 含义：每个高斯点在图像平面上的投影覆盖了多少个瓦片

---

## 关键组件

### 1. 投影模块（Projection）

#### 功能
将 3D 高斯点投影到 2D 图像平面

#### 关键步骤

1. **世界坐标到相机坐标**
   - 输入：`means`（世界坐标）、`viewmats`（变换矩阵）
   - 输出：`means_c`（相机坐标）
   - 变换：`μ_c = W * μ + t`，其中 `[W | t]` 为世界到相机的变换

2. **3D 协方差转换**
   - 输入：`quats`、`scales`（或 `covars`）
   - 计算：`Σ = R * S * S^T * R^T`
   - 转换到相机空间：`Σ_c = W * Σ * W^T`

3. **透视投影**
   - 使用投影雅可比矩阵近似：
     ```
     J = [fx/z   0    -fx*x/z²]
         [0      fy/z -fy*y/z²]
         [0      0    0       ]
     ```
   - 投影后的 2D 协方差：
     ```
     Σ' = J * Σ_c * J^T
     ```

#### 实现函数
- `fully_fused_projection()`: 标准投影
- `fully_fused_projection_with_ut()`: 使用 Unscented Transform 的投影（3DGUT）

### 2. 排序模块（Sorting）

#### 功能
对高斯点进行深度排序，以便从前到后进行 alpha 合成

#### 关键步骤

1. **瓦片交集计算**
   - 计算每个高斯点与哪些瓦片相交
   - 使用 `radii` 确定高斯点的边界框

2. **深度排序**
   - 在每个瓦片内，按深度 `z` 递增排序
   - 使用基数排序（radix sort）高效实现

3. **扁平化索引**
   - 生成 `flatten_ids` 用于后续光栅化

#### 实现函数
- `isect_tiles()`: 计算高斯点与瓦片的交集
- `isect_offset_encode()`: 编码交集偏移量

### 3. 光栅化模块（Rasterization）

#### 功能
将排序后的高斯点渲染到像素

#### 关键步骤

1. **高斯权重计算**
   - 对于像素 `p = (x, y)` 和高斯点 `i`：
     ```
     Δ = p - μ'_i
     σ = 0.5 * (conic_i[0] * Δx² + conic_i[2] * Δy² + 2 * conic_i[1] * Δx * Δy)
     ```

2. **Alpha 合成**
   - 每个高斯点的贡献：
     ```
     α_i = min(0.999, opacity_i * exp(-σ))
     ```
   - 从前到后累积：
     ```
     C(p) = Σ_i c_i * α_i * T_i
     T_i = Π_{j<i} (1 - α_j)
     ```
   - `T_i` 是传输率（transmittance）

3. **并行化**
   - 每个线程块处理一个瓦片
   - 每个线程处理瓦片内的一个像素
   - 使用批处理（batch processing）高效处理多个高斯点

#### 实现函数
- `rasterize_to_pixels()`: 标准光栅化
- `rasterize_to_pixels_eval3d()`: 3D 空间评估模式
- `rasterize_to_pixels_3dgs_fwd()`: CUDA 内核实现

### 4. Spherical Harmonics 模块

#### 功能
根据视角方向动态计算高斯点颜色

#### 关键步骤

1. **计算视角方向**
   - 从相机位置到高斯点中心的方向向量

2. **评估 SH 基函数**
   - 使用 SH 阶数 `sh_degree` 计算基函数值

3. **颜色计算**
   - `c = Σ_k SH_k(dir) * coeffs_k`

#### 实现函数
- `spherical_harmonics()`: 计算 SH 基函数并评估颜色

### 5. Transmittances 计算模块

#### 功能
计算并返回每像素的传输率（transmittance）

#### 关键概念

**传输率（Transmittance）**：表示光线穿透所有高斯点后的剩余能量，即光线未被遮挡的比例。

#### 计算方法

1. **在光栅化时计算**（当 `return_transmittances=True` 时）：
   - 在 CUDA 内核中直接计算并返回每个像素的传输率
   - 传输率 `T` 在 alpha 合成过程中累积：`T = Π_i (1 - α_i)`
   - 形状：`[..., C, height, width]`

2. **从 render_alphas 计算**（如果未在光栅化时计算）：
   ```python
   transmittances = 1.0 - render_alphas[..., 0]
   ```
   - 这是因为 `render_alphas` 存储的是累积不透明度 `A = 1 - T`
   - 形状：`[..., C, height, width]`

#### 使用场景

- **避免重复计算**：如果后续需要传输率信息（例如用于计算损失），可以通过 `return_transmittances=True` 避免从 `render_alphas` 重新计算
- **存储效率**：在光栅化时计算可以直接写入输出，避免额外的内存分配和计算

#### 存储位置

传输率存储在 `meta["transmittances"]` 中，只有当 `return_transmittances=True` 时才会包含。

---

## 算法流程

### 整体流程

```
输入：3D 高斯点参数 + 相机参数
  ↓
[1] 投影阶段（Projection）
  ├─ 3D 高斯 → 相机坐标系
  ├─ 计算 3D 协方差（从 quats + scales）
  ├─ 透视投影 → 2D 高斯
  └─ 计算 2D 半径（用于瓦片交集）
  ↓
[2] 颜色计算（可选）
  └─ 如果使用 SH：根据视角计算颜色
  ↓
[3] 排序阶段（Sorting）
  ├─ 计算高斯点与瓦片的交集
  ├─ 在每个瓦片内按深度排序
  └─ 生成扁平化索引
  ↓
[4] 光栅化阶段（Rasterization）
  ├─ 对每个瓦片的每个像素：
  │   ├─ 从前到后遍历排序后的高斯点
  │   ├─ 计算高斯权重（conic 形式）
  │   ├─ 计算 alpha（不透明度 * exp(-权重)）
  │   ├─ Alpha 合成颜色
  │   └─ 累积传输率 T（如果 return_transmittances=True）
  └─ 输出最终图像
  ↓
[5] 后处理（可选）
  ├─ 如果 return_transmittances=True：
  │   └─ 将 transmittances 添加到 meta
  └─ 如果 render_mode 包含 "ED"：
      └─ 归一化深度值（深度 / alpha）
  ↓
输出：渲染图像 + alpha 通道 + meta 信息（可能包含 transmittances）
```

### 详细步骤

#### 步骤 1：投影（Projection）

1. **变换到相机坐标系**
   ```python
   means_c = transform_world_to_camera(means, viewmats)
   covar_c = transform_covar_to_camera(covar, viewmats)
   ```

2. **计算 3D 协方差矩阵**
   ```python
   R = quaternion_to_rotation_matrix(quats)
   S = diag(scales)
   covar = R * S * S^T * R^T
   ```

3. **透视投影**
   ```python
   means2d = project_point(means_c, Ks)
   covar2d = project_covariance(covar_c, means_c, Ks)  # 使用雅可比近似
   ```

4. **计算 2D 半径**
   ```python
   radii = compute_2d_radius(covar2d)
   ```

#### 步骤 2：排序（Sorting）

1. **瓦片交集**
   ```python
   # 对每个高斯点，计算它覆盖的瓦片
   tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
       means2d, radii, depths, tile_size, ...
   )
   ```

2. **深度排序**
   ```python
   # 在每个瓦片内，按深度 z 递增排序
   # 使用基数排序高效实现
   sorted_indices = radix_sort_by_depth(depths, isect_ids, ...)
   ```

3. **编码偏移量**
   ```python
   isect_offsets = isect_offset_encode(isect_ids, ...)
   ```

#### 步骤 3：光栅化（Rasterization）

1. **对每个瓦片**：
   ```python
   for tile in tiles:
       # 获取该瓦片的高斯点列表（已按深度排序）
       gaussians = get_gaussians_for_tile(tile, isect_offsets, flatten_ids)
       
       # 对瓦片内的每个像素
       for pixel in tile:
           T = 1.0  # 初始传输率
           color = 0.0  # 初始颜色
           
           # 从前到后遍历高斯点
           for gaussian in gaussians:
               # 计算高斯权重
               delta = pixel - gaussian.means2d
               sigma = 0.5 * (conic[0] * delta.x² + 
                             conic[2] * delta.y² + 
                             2 * conic[1] * delta.x * delta.y)
               
               # 计算 alpha
               alpha = min(0.999, opacity * exp(-sigma))
               
               # Alpha 合成
               visibility = alpha * T
               color += gaussian.color * visibility
               T *= (1.0 - alpha)
               
               # 早停：如果传输率很小，不再处理后续高斯点
               if T < 1e-4:
                   break
           
           output[pixel] = color
           output_alpha[pixel] = 1.0 - T
           
           # 如果 return_transmittances=True，同时存储传输率
           if return_transmittances:
               output_transmittances[pixel] = T
   ```

### 关键优化

1. **打包模式（Packed Mode）**
   - 只存储可见的高斯点（`radii > 0`）
   - 使用稀疏张量存储，节省内存

2. **瓦片化（Tiling）**
   - 将图像划分为 16×16 的瓦片
   - 只处理与每个瓦片相交的高斯点
   - 减少计算量

3. **并行化**
   - GPU 并行：每个线程块处理一个瓦片
   - 批处理：每个线程同时处理多个高斯点
   - 多 GPU：分布式渲染大规模场景

4. **早停机制**
   - 当传输率 `T < 1e-4` 时停止处理后续高斯点
   - 当 alpha 太小（`< 1/255`）时跳过高斯点

---

## 数学公式

### 1. 3D 协方差矩阵参数化

给定四元数 `q`（wxyz 约定）和缩放向量 `s = [s_x, s_y, s_z]`：

1. **四元数转旋转矩阵**：
   ```
   R = quaternion_to_rotation_matrix(q)
   ```

2. **缩放矩阵**：
   ```
   S = diag(s) = [s_x  0   0  ]
                 [0    s_y 0  ]
                 [0    0   s_z]
   ```

3. **协方差矩阵**：
   ```
   Σ = R * S * S^T * R^T = R * diag(s²) * R^T
   ```

### 2. 投影到 2D

#### 点的投影
相机坐标系中的点 `[x_c, y_c, z_c]` 投影到图像平面：
```
[u]   [fx/zc * xc + cx]
[v] = [fy/zc * yc + cy]
```

#### 协方差的投影
使用投影雅可比矩阵 `J` 近似：
```
J = [fx/zc   0     -fx*xc/zc²]
    [0       fy/zc -fy*yc/zc²]
    [0       0     0          ]

Σ' = J * Σ_c * J^T
```

### 3. 2D 高斯权重计算

对于像素 `p = (x, y)` 和投影后的 2D 高斯 `(μ', Σ')`：

**距离计算**：
```
Δ = p - μ' = [x - μ'_x, y - μ'_y]
```

**Conic 形式**（协方差矩阵的逆）：
```
Σ'⁻¹ = [a b]
       [b c]
```

**权重**：
```
σ = 0.5 * (a * Δx² + c * Δy² + 2 * b * Δx * Δy)
```

### 4. Alpha 合成

**单个高斯点的贡献**：
```
α_i = min(0.999, opacity_i * exp(-σ_i))
```

**传输率（前 i-1 个高斯点的累积）**：
```
T_i = Π_{j=0}^{i-1} (1 - α_j)
```

**像素颜色**：
```
C(p) = Σ_i c_i * α_i * T_i
      = c₀ * α₀ * 1.0 + 
        c₁ * α₁ * (1 - α₀) + 
        c₂ * α₂ * (1 - α₀) * (1 - α₁) + 
        ...
```

**Alpha 通道**：
```
A(p) = 1 - T_final = 1 - Π_i (1 - α_i)
```

**传输率（Transmittance）**：
```
T(p) = T_final = Π_i (1 - α_i)
```

传输率表示光线穿透所有高斯点后的剩余能量。当 `return_transmittances=True` 时，传输率会被存储在 `meta["transmittances"]` 中，形状为 `[..., C, height, width]`。

### 5. Spherical Harmonics 颜色

给定视角方向 `dir` 和 SH 系数 `coeffs[k]`（k 为基函数索引）：

**SH 基函数评估**：
```
SH_k(dir) = Y_l^m(θ, φ)  # 球谐函数
```

**颜色计算**：
```
c = Σ_{k=0}^{K-1} SH_k(dir) * coeffs[k]
```

其中 `K = (sh_degree + 1)²` 是 SH 基函数的数量。

---

## 参考资料

- **论文**：[3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **gsplat 文档**：`third_party/gsplat/docs/source/apis/rasterization.rst`
- **实现代码**：`third_party/gsplat/gsplat/rendering.py`

---

## 总结

gsplat 的 3DGS 实现通过以下关键技术实现了高效渲染：

1. **高效参数化**：使用四元数 + 缩放向量表示 3D 协方差，比直接存储 3×3 矩阵更紧凑
2. **瓦片化光栅化**：将图像划分为瓦片，只处理相关的高斯点，减少计算量
3. **深度排序**：使用基数排序高效地对高斯点进行深度排序
4. **打包模式**：只存储可见的高斯点，节省内存
5. **并行化**：充分利用 GPU 并行计算能力，支持批处理和分布式渲染

这些优化使得 gsplat 在保持渲染质量的同时，实现了更高的训练和推理效率。
