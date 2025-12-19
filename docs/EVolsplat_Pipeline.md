# EVolSplat Pipeline 和 Model 对比分析

## 概述

本文档对比分析 EVolSplat 的 feed-forward 3DGS 方法与标准 nerfstudio per-scene 3DGS 方法在 Pipeline 和 Model 层面的实现差异。EVolSplat 通过复用 nerfstudio 的基础架构，实现了支持多场景训练和零样本泛化的 feed-forward 模型。

---

## 核心概念

### NeRF、3DGS 与 nerfstudio

- **NeRF (Neural Radiance Fields)**: 基于神经网络的隐式场景表示，通过体积渲染生成新视角图像
- **3DGS (3D Gaussian Splatting)**: 基于显式 3D 高斯点的场景表示，通过光栅化渲染
- **Per-scene 3DGS**: 每个场景独立训练，直接优化场景特定的高斯参数
- **Feed-forward 3DGS**: 通过神经网络直接预测高斯参数，支持零样本泛化到新场景
- **nerfstudio**: 模块化的 NeRF/3DGS 训练框架，提供 Pipeline、Model、DataManager 等抽象接口

### EVolSplat 与标准 3DGS 的核心区别

| 特性               | 标准 3DGS (Splatfacto) | EVolSplat         |
| ------------------ | ---------------------- | ----------------- |
| **训练方式** | Per-scene 优化         | Feed-forward 预测 |
| **参数优化** | 直接优化 Gaussian 参数 | 优化神经网络参数  |
| **场景支持** | 单场景训练             | 多场景联合训练    |
| **泛化能力** | 场景特定               | 零样本泛化        |
| **推理速度** | 需要优化迭代           | 单次前向传播      |

---

## Pipeline 对比

### 1. Pipeline 类结构对比

| 特性                  | 标准 nerfstudio                               | EVolSplat                  |
| --------------------- | --------------------------------------------- | -------------------------- |
| **Pipeline 类** | `VanillaPipeline`                           | `VanillaPipeline` (复用) |
| **配置文件**    | `VanillaPipelineConfig`                     | `VanillaPipelineConfig`  |
| **DataManager** | `VanillaDataManager` / `SplatDatamanager` | `SplatDatamanager`       |
| **Model**       | `SplatfactoModel`                           | `EvolSplatModel`         |
| **初始化方式**  | 单场景点云                                    | 多场景点云列表             |

### 2. Pipeline 初始化流程对比

#### 标准 3DGS Pipeline 初始化

```
VanillaPipeline.__init__()
    ↓
1. 设置 DataManager
   - datamanager.setup()
   - 解析单场景数据
   ↓
2. 提取 seed_points
   - 从 metadata['points3D_xyz'] 获取点云
   - 格式: (points, colors) Tuple
   ↓
3. 初始化 Model
   - model.setup(seed_points=seed_points)
   - SplatfactoModel 接收单场景点云
```

#### EVolSplat Pipeline 初始化

```
VanillaPipeline.__init__()
    ↓
1. 设置 DataManager
   - SplatDatamanager.setup()
   - 解析多场景数据
   ↓
2. 提取 seed_points
   - 从 metadata['input_pnt'] 获取点云列表
   - 格式: List[Dict{'points3D_xyz', 'points3D_rgb'}]
   ↓
3. 初始化 Model
   - model.setup(seed_points=seed_points)
   - EvolSplatModel 接收多场景点云列表
```

### 3. Pipeline 核心方法对比

| 方法                                          | 标准 Pipeline                  | EVolSplat Pipeline                              |
| --------------------------------------------- | ------------------------------ | ----------------------------------------------- |
| **get_train_loss_dict()**               | 返回 `RayBundle` + `batch` | 返回 `Cameras` + `batch` (含 source/target) |
| **get_eval_loss_dict()**                | 单图像评估                     | 单图像评估（支持多场景）                        |
| **get_eval_image_metrics_and_images()** | 标准评估                       | 支持多场景评估                                  |
| **get_param_groups()**                  | 返回 Gaussian 参数组           | 返回 MLP 网络参数组                             |

### 4. 训练数据流对比

#### 标准 3DGS 数据流

```
DataManager.next_train(step)
    ↓
RayBundle + Batch
    ├─ RayBundle.origins: [N_rays, 3]
    ├─ RayBundle.directions: [N_rays, 3]
    └─ batch['image']: [H, W, 3]
    ↓
Model.get_outputs(ray_bundle)
    ↓
渲染输出
```

#### EVolSplat 数据流

```
SplatDatamanager.next_train(step)
    ↓
Cameras + Batch
    ├─ camera: Cameras[1]
    ├─ batch['scene_id']: Tensor[1]
    ├─ batch['source']: {
    │     'image': [V, H, W, 3],
    │     'extrinsics': [V, 4, 4],
    │     'intrinsics': [V, 4, 4],
    │     'depth': [V, H, W]
    │   }
    └─ batch['target']: {
          'image': [1, H, W, 3],
          'extrinsics': [1, 4, 4],
          'intrinsics': [1, 4, 4]
        }
    ↓
EvolSplatModel.get_outputs(camera, batch)
    ↓
渲染输出
```

---

## Model 对比

### 1. Model 类结构对比

| 特性                 | SplatfactoModel                        | EvolSplatModel                     |
| -------------------- | -------------------------------------- | ---------------------------------- |
| **继承**       | `Model`                              | `Model`                          |
| **初始化参数** | `seed_points: Tuple[Tensor, Tensor]` | `seed_points: List[Dict]`        |
| **参数存储**   | `ParameterDict` (可优化)             | `List[Tensor]` (固定) + MLP 网络 |
| **场景数量**   | 1                                      | `num_scenes` (多场景)            |

### 2. Model 初始化对比

#### SplatfactoModel 初始化

**关键组件**:

- `gauss_params`: `ParameterDict` 包含可优化的 Gaussian 参数
  - `means`: `Parameter[N, 3]` - 位置
  - `scales`: `Parameter[N, 3]` - 尺度
  - `quats`: `Parameter[N, 4]` - 旋转（四元数）
  - `features_dc`: `Parameter[N, 3]` - 颜色（DC项）
  - `features_rest`: `Parameter[N, SH-1, 3]` - 颜色（高阶SH）
  - `opacities`: `Parameter[N, 1]` - 不透明度

**参数优化**:

- 所有 Gaussian 参数直接参与梯度更新
- 通过优化器直接优化每个高斯点的参数

#### EvolSplatModel 初始化

**关键组件**:

- `means`: `List[Tensor[N_i, 3]]` - 每个场景的初始位置（固定）
- `anchor_feats`: `List[Tensor[N_i, 3]]` - 每个场景的锚点特征（固定）
- `scales`: `List[Tensor[N_i, 3]]` - 每个场景的初始尺度（固定）
- `offset`: `List[Tensor[N_i, 3]]` - 每个场景的位置偏移（迭代更新）

**神经网络组件**:

- `sparse_conv`: `SparseCostRegNet` - 3D 稀疏卷积网络（提取体积特征）
- `mlp_offset`: `MLP` - 预测位置偏移
- `mlp_conv`: `MLP` - 预测尺度和旋转
- `mlp_opacity`: `MLP` - 预测不透明度
- `gaussion_decoder`: `MLP` - 预测颜色（SH系数）
- `projector`: `Projector` - 2D 特征投影器

**参数优化**:

- 只优化神经网络参数（MLP 和 sparse_conv）
- Gaussian 参数通过神经网络预测，不直接优化

### 3. Model.get_outputs() 方法对比

#### SplatfactoModel.get_outputs()

**流程**:

```
输入: RayBundle
    ↓
1. 获取 Gaussian 参数（直接从 ParameterDict）
   - means, scales, quats, opacities, features
    ↓
2. 相机优化（可选）
   - camera_optimizer(camera)
    ↓
3. 光栅化渲染
   - rasterization(means, quats, scales, opacities, colors)
    ↓
输出: {'rgb', 'depth', 'accumulation'}
```

**关键数据**:

- 所有参数直接从 `self.gauss_params` 读取
- 参数在训练过程中通过梯度更新

#### EvolSplatModel.get_outputs()

**流程**:

```
输入: Camera + Batch (含 scene_id, source, target)
    ↓
1. 根据 scene_id 加载场景数据
   - means = self.means[scene_id]
   - scales = self.scales[scene_id]
   - offset = self.offset[scene_id]
   - anchors_feat = self.anchor_feats[scene_id]
    ↓
2. 构建 3D 特征体积（如果未冻结）
   - construct_sparse_tensor(means, anchors_feat)
   - sparse_conv() → dense_volume
    ↓
3. 提取 2D 特征（从 source 图像）
   - projector.sample_within_window(source_images, ...)
   - 多视角特征聚合
    ↓
4. 预测 Gaussian 参数
   ├─ 位置: means_crop + mlp_offset(feat_3d)
   ├─ 尺度+旋转: mlp_conv(feat_3d + view_dir)
   ├─ 不透明度: mlp_opacity(feat_3d + view_dir)
   └─ 颜色: gaussion_decoder(sampled_feat + view_dir)
    ↓
5. 光栅化渲染
   - rasterization(means_crop, quats, scales, opacities, colors)
    ↓
6. 背景渲染（可选）
   - bg_field() → background_rgb
    ↓
输出: {'rgb', 'depth', 'accumulation', 'background'}
```

**关键数据**:

- 场景特定数据从 `List` 中索引获取
- Gaussian 参数通过神经网络预测
- `offset` 通过保存-加载机制迭代更新（不参与梯度）

### 4. Model.get_param_groups() 对比

#### SplatfactoModel 参数组

```python
{
    'means': [Parameter[N, 3]],
    'scales': [Parameter[N, 3]],
    'quats': [Parameter[N, 4]],
    'features_dc': [Parameter[N, 3]],
    'features_rest': [Parameter[N, SH-1, 3]],
    'opacities': [Parameter[N, 1]],
    'camera_opt': [相机优化器参数]
}
```

**特点**:

- 参数数量 = N（高斯点数量）
- 每个高斯点有独立的可优化参数
- 参数数量随场景复杂度变化

#### EvolSplatModel 参数组

```python
{
    'sparse_conv': [稀疏卷积网络参数],
    'mlp_conv': [MLP 参数（尺度+旋转）],
    'mlp_opacity': [MLP 参数（不透明度）],
    'mlp_offset': [MLP 参数（位置偏移）],
    'gaussianDecoder': [MLP 参数（颜色）],
    'background_model': [背景模型参数]
}
```

**特点**:

- 参数数量固定（网络结构决定）
- 不依赖场景中的高斯点数量
- 支持多场景共享参数

### 5. 训练机制对比

#### SplatfactoModel 训练机制

**优化目标**:

- 直接优化每个高斯点的参数
- 通过 densification 和 pruning 动态调整高斯点数量

**训练流程**:

```
Step N:
1. 前向传播
   - 使用当前 Gaussian 参数渲染
2. 计算损失
   - L1 + SSIM
3. 反向传播
   - 更新所有 Gaussian 参数
4. Densification/Pruning（周期性）
   - 根据梯度信息添加/删除高斯点
```

**关键特性**:

- 参数直接参与梯度更新
- 支持动态结构调整（densification）
- 每个场景需要独立训练

#### EvolSplatModel 训练机制

**优化目标**:

- 优化神经网络参数（MLP 和 sparse_conv）
- 学习从多视角图像到 Gaussian 参数的映射

**训练流程**:

```
Step N:
1. 随机选择场景 scene_id
2. 前向传播
   - 加载场景点云和 offset
   - 提取 2D/3D 特征
   - 预测 Gaussian 参数
   - 渲染图像
3. 计算损失
   - L1 + SSIM + Entropy
4. 反向传播
   - 更新 MLP 网络参数
   - 更新 sparse_conv 参数
   - offset 被 detach，不参与梯度
5. 保存 offset（迭代更新）
   - self.offset[scene_id] = offset_crop.detach()
```

**关键特性**:

- 只优化网络参数，不直接优化 Gaussian 参数
- offset 通过保存-加载机制迭代更新
- 支持多场景联合训练

---

## 关键数据流对比

### 1. 输入数据对比

| 数据项             | SplatfactoModel        | EvolSplatModel                 |
| ------------------ | ---------------------- | ------------------------------ |
| **输入类型** | `RayBundle`          | `Camera` + `Dict`          |
| **场景数据** | 单场景，嵌入在模型中   | 多场景，通过 `scene_id` 索引 |
| **图像数据** | 单张图像（用于监督）   | Source 图像组 + Target 图像    |
| **点云数据** | 初始化时使用，后续优化 | 每个场景固定存储，用于特征提取 |

### 2. 中间特征对比

| 特征类型           | SplatfactoModel              | EvolSplatModel                 |
| ------------------ | ---------------------------- | ------------------------------ |
| **3D 特征**  | 无（直接使用 Gaussian 参数） | 稀疏卷积提取的体积特征         |
| **2D 特征**  | 无                           | 从 source 图像投影的多视角特征 |
| **视图特征** | 无                           | 相机方向 + 距离                |
