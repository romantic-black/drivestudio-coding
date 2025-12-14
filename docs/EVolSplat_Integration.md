# EVolSplat 集成说明

本文档详细讲解提交 `c6f1828471e73edb6edc2b545aec02f0df048424` 中关于 EVolSplat 初始化的改动内容。

## 提交概述

该提交实现了 EVolSplat 框架的初始化集成，将 EVolSplat 的核心组件整合到 drivestudio 系统中。主要改动包括：

- 新增 EVolSplat 模型组件模块
- 新增 NuScenes 直接数据加载器
- 新增 EVolSplat 训练器
- 新增配置文件

## 文件结构

本次提交共添加了 8 个新文件，总计 1751 行代码：

```
configs/evolsplat_nuscenes.yaml            (134 行)
datasets/nuscenes/nuscenes_directloader.py (381 行)
models/evol_splat/__init__.py              (32 行)
models/evol_splat/mlp_decoders.py          (131 行)
models/evol_splat/projection.py            (272 行)
models/evol_splat/sparse_conv.py           (166 行)
models/evol_splat/utils.py                 (54 行)
models/trainers/evol_splat_trainer.py      (581 行)
```

## 详细改动说明

### 1. 配置文件：`configs/evolsplat_nuscenes.yaml`

该配置文件定义了 EVolSplat 在 NuScenes 数据集上的训练配置。

#### 主要配置项：

**训练器配置 (trainer)**
- `type`: 使用 `EVolSplatTrainer` 作为训练器
- `optim.num_iters`: 训练迭代次数（30000）
- `render`: 渲染相关配置（近远平面、抗锯齿等）
- `losses`: 损失函数权重配置
  - RGB 损失权重：0.8
  - SSIM 损失权重：0.2
  - Mask 损失权重：0.05
  - Depth 损失权重：0.01
  - Affine 正则化权重：0.00001

**EVolSplat 特定配置 (evol_splat)**
- `sparse_conv_outdim`: 稀疏卷积输出维度（8）
- `voxel_size`: 体素大小（0.1）
- `local_radius`: 2D 特征采样的局部窗口半径（1）
- `num_neighbour_select`: 使用的相邻视图数量（4）
- `sh_degree`: 球谐函数度数（1）
- `offset_max`: 点精化的最大偏移量（0.1）
- `freeze_volume`: 是否在初始化后冻结体积特征（false）

**优化器配置**
- 为 EVolSplat 的各个组件（gaussion_decoder、mlp_conv、mlp_opacity、mlp_offset、sparse_conv）配置了独立的学习率（默认 0.001）

### 2. 数据加载器：`datasets/nuscenes/nuscenes_directloader.py`

实现了直接从 NuScenes devkit 读取数据的加载器，无需预处理步骤。

#### 核心类：

**`NuScenesDirectCameraData`**
- 继承自 `CameraData`
- 直接从 NuScenes API 获取图像和标定数据
- 使用 `sample_data_token` 而不是文件路径来标识图像
- 自动处理相机到世界坐标系的转换和对齐

**`NuScenesDirectPixelSource`**
- 继承自 `ScenePixelSource`
- 管理多个相机的数据加载
- 使用第一个选定帧的前置相机作为世界坐标系原点
- 支持对象加载（从 NuScenes 标注中提取）

**`NuScenesDirectLiDARSource`**
- 继承自 `SceneLidarSource`
- 直接从 NuScenes 原始点云文件读取 LiDAR 数据
- 将点云转换为射线格式（origins, directions, ranges）

#### 关键功能：

1. **坐标系统一**：使用 `reference_inv` 将所有传感器数据对齐到统一的坐标系
2. **时间戳管理**：自动注册和归一化时间戳
3. **相机标定**：自动缩放内参以适应图像尺寸变化

### 3. EVolSplat 模型组件

#### 3.1 `models/evol_splat/__init__.py`

模块入口文件，导出所有 EVolSplat 组件：

- `SparseCostRegNet`: 稀疏成本正则化网络
- `construct_sparse_tensor`: 构建稀疏张量
- `sparse_to_dense_volume`: 稀疏到密集体积转换
- `Projector`: 2D 特征投影器
- MLP 解码器创建函数：`create_gaussion_decoder`, `create_mlp_conv`, `create_mlp_opacity`, `create_mlp_offset`
- 工具函数：`interpolate_features`, `get_grid_coords`

#### 3.2 `models/evol_splat/mlp_decoders.py`

定义了用于生成高斯参数的 MLP 解码器。

**主要函数：**

- `create_gaussion_decoder()`: 创建高斯外观解码器
  - 输入：特征维度 + 4（ob_dist 和 ob_view）
  - 输出：球谐函数系数（3 * num_sh_bases）
  - 结构：3 层，每层 128 维

- `create_mlp_conv()`: 创建尺度和旋转预测 MLP
  - 输出：3 维尺度 + 4 维四元数旋转
  - 结构：2 层，每层 64 维，使用 Tanh 激活

- `create_mlp_opacity()`: 创建不透明度预测 MLP
  - 输出：1 维不透明度
  - 结构：2 层，每层 64 维

- `create_mlp_offset()`: 创建 3D 偏移预测 MLP
  - 输出：3 维偏移量
  - 结构：2 层，每层 64 维，输出使用 Tanh 激活

**依赖处理：**
- 尝试从 `nerfstudio` 导入 MLP，如果失败则使用简单的回退实现

#### 3.3 `models/evol_splat/projection.py`

实现了 2D 图像特征提取的投影器。

**`Projector` 类：**

- `inbound()`: 检查像素位置是否在有效范围内
- `normalize()`: 将像素位置归一化到 [-1, 1] 范围
- `compute_projections()`: 将 3D 点投影到相机
  - 输入：3D 点坐标、相机外参、内参
  - 输出：像素位置、可见性掩码、深度值
- `sample_within_window()`: 在局部窗口内采样 2D 特征
  - 从多个相邻视图采样特征
  - 使用双线性插值获取特征值

#### 3.4 `models/evol_splat/sparse_conv.py`

实现了稀疏卷积网络用于 3D 体积特征提取。

**核心组件：**

- `BasicSparseConvolutionBlock`: 基础稀疏卷积块
  - 包含稀疏卷积、批归一化和 ReLU 激活
- `BasicSparseDeconvolutionBlock`: 基础稀疏反卷积块
  - 用于上采样特征
- `SparseCostRegNet`: 稀疏成本正则化网络
  - 编码器-解码器结构
  - 使用跳跃连接保留细节
  - 输出维度可配置（默认 8 维）

**工具函数：**

- `construct_sparse_tensor()`: 从原始坐标和特征构建稀疏张量
  - 体素化点云
  - 量化坐标到离散网格
- `sparse_to_dense_volume()`: 将稀疏张量转换为密集体积
  - 用于后续的三线性插值

#### 3.5 `models/evol_splat/utils.py`

提供工具函数：

- `interpolate_features()`: 从 3D 体积中使用三线性插值提取特征
  - 使用 `torch.nn.functional.grid_sample`
- `get_grid_coords()`: 将世界坐标转换为网格坐标
  - 用于体积插值的坐标归一化

### 4. 训练器：`models/trainers/evol_splat_trainer.py`

核心训练器类，整合了 EVolSplat 组件和 drivestudio 渲染系统。

#### `EVolSplatTrainer` 类

继承自 `MultiTrainer`，实现了动态生成 3DGS 参数的训练流程。

**设计理念：**
- EVolSplat 生成 3DGS 点云参数（而非直接生成图像）
- 参数转换为 drivestudio 格式
- 使用 drivestudio 的渲染系统进行最终渲染

**核心方法：**

1. **`__init__()`**: 初始化
   - 加载 EVolSplat 配置
   - 初始化所有组件（Projector、SparseCostRegNet、MLP 解码器）
   - 设置边界框和体素参数

2. **`get_param_groups()`**: 获取参数组
   - 为 EVolSplat 的各个组件创建参数组
   - 用于优化器初始化

3. **`initialize_optimizer()`**: 初始化优化器
   - 为每个参数组设置独立的学习率
   - 支持学习率调度器

4. **`init_gaussians_from_dataset()`**: 从数据集初始化种子点
   - 占位接口，需要用户实现
   - 应设置 `self.means`, `self.anchor_feats`, `self.scales`, `self.offset`

5. **`forward()`**: 前向传播
   - 生成高斯参数
   - 调用父类方法进行渲染

6. **`collect_gaussians()`**: 收集高斯参数
   - 覆盖父类方法
   - 返回 EVolSplat 生成的高斯参数

7. **`_generate_gaussians_from_features()`**: 核心生成逻辑
   - **步骤 1**：使用稀疏卷积提取 3D 特征
     - 构建稀疏张量
     - 通过 SparseCostRegNet 提取特征
     - 转换为密集体积
   - **步骤 2**：使用投影器提取 2D 特征
     - 从多个相邻视图采样特征
     - 过滤有效投影点
   - **步骤 3**：插值 3D 特征
     - 使用三线性插值从体积中提取特征
   - **步骤 4**：计算视图相关特征
     - 计算观察方向和距离
   - **步骤 5**：使用 MLP 解码器生成高斯参数
     - 颜色（球谐函数系数）
     - 尺度和旋转
     - 不透明度
     - 偏移量（用于点精化）

8. **`_convert_to_drivestudio_format()`**: 格式转换
   - 将 EVolSplat 格式转换为 drivestudio 的 `dataclass_gs` 格式
   - 处理尺度、旋转、不透明度的转换
   - 评估球谐函数得到 RGB 颜色

**数据流：**

```
输入图像和相机参数
    ↓
转换为 EVolSplat 批次格式
    ↓
提取 3D 特征（稀疏卷积）
    ↓
提取 2D 特征（投影器）
    ↓
插值 3D 特征
    ↓
生成高斯参数（MLP 解码器）
    ↓
转换为 drivestudio 格式
    ↓
渲染
```

## 关键设计决策

### 1. 模块化设计

EVolSplat 组件被设计为独立的模块，可以单独使用或组合使用：
- 稀疏卷积网络可以独立提取 3D 特征
- 投影器可以独立进行 2D 特征采样
- MLP 解码器可以独立生成高斯参数

### 2. 格式转换层

在 EVolSplat 和 drivestudio 之间添加了格式转换层：
- `_convert_to_evolsplat_batch()`: drivestudio → EVolSplat
- `_convert_to_drivestudio_format()`: EVolSplat → drivestudio

这允许两个系统保持相对独立，便于维护和扩展。

### 3. 占位接口

某些方法被设计为占位接口，需要用户根据具体数据格式实现：
- `init_gaussians_from_dataset()`: 初始化种子点
- `_convert_to_evolsplat_batch()`: 数据格式转换

这种设计提供了灵活性，允许用户根据具体需求定制实现。

### 4. 体积缓存

实现了体积特征缓存机制：
- `freeze_volume` 选项允许冻结体积特征
- 在训练过程中可以复用已计算的体积特征，提高效率

## 使用说明

### 1. 配置训练

使用 `configs/evolsplat_nuscenes.yaml` 配置文件启动训练：

```bash
python tools/train.py --config configs/evolsplat_nuscenes.yaml
```

### 2. 数据准备

使用新的直接加载器，无需预处理：

```python
# 在配置文件中指定
dataset: nuscenes/6cams_direct
```

### 3. 实现占位接口

需要实现以下方法：

**初始化种子点：**
```python
def init_gaussians_from_dataset(self, dataset):
    # 从数据集获取初始点云
    self.means = ...  # [N, 3]
    self.anchor_feats = ...  # [N, 3] RGB
    self.scales = ...  # [N, 3] log scale
    self.offset = ...  # [N, 3] zeros
```

**数据格式转换：**
```python
def _convert_to_evolsplat_batch(self, image_infos, camera_infos):
    # 转换为 EVolSplat 批次格式
    return {
        'source': {
            'image': ...,  # [N_views, C, H, W]
            'extrinsics': ...,  # [N_views, 4, 4]
            'intrinsics': ...,  # [N_views, 4, 4]
        },
        'target': {
            'image': ...,  # [H, W, C]
            'intrinsics': ...,  # [4, 4]
        }
    }
```

## 依赖项

### 必需依赖

- `torch`: PyTorch 框架
- `nuscenes-devkit`: NuScenes 数据集开发工具包
- `spconv`: 稀疏卷积库（用于 `SparseCostRegNet`）
- `gsplat`: 高斯点渲染库
- `einops`: 张量操作库

### 可选依赖

- `nerfstudio`: 如果可用，会使用其 MLP 实现；否则使用回退实现

## 注意事项

1. **内存使用**：稀疏卷积和体积特征可能占用大量内存，建议使用适当的体素大小和分辨率设置。

2. **训练稳定性**：EVolSplat 使用 MLP 解码器生成参数，可能需要仔细调整学习率和损失权重。

3. **数据格式**：确保数据格式转换正确，特别是相机坐标系（OpenCV vs OpenGL）的转换。

4. **点云初始化**：初始点云的质量对最终结果有重要影响，建议使用高质量的 LiDAR 点云或密集重建点云。

## 未来改进方向

1. **完整的实现**：实现占位接口，提供完整的数据加载和转换流程。

2. **性能优化**：优化稀疏卷积和体积插值的性能，支持更大的场景。

3. **更多数据集支持**：扩展直接加载器以支持其他数据集（KITTI、Waymo 等）。

4. **训练策略**：实现更高级的训练策略，如渐进式训练、自适应分辨率等。

## 相关文件

- 配置文件：`configs/evolsplat_nuscenes.yaml`
- 数据加载器：`datasets/nuscenes/nuscenes_directloader.py`
- 模型组件：`models/evol_splat/`
- 训练器：`models/trainers/evol_splat_trainer.py`

## 参考

- EVolSplat 原始论文和代码
- drivestudio 项目文档
- NuScenes 数据集文档

