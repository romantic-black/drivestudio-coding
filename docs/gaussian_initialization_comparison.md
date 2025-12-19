# 3D Gaussian Splatting 初始化流程对比

## 3DGS方法概述

3D Gaussian Splatting (3DGS) 是一种基于3D高斯点的神经渲染方法。每个高斯点包含以下参数：
- **位置 (means)**: 3D空间坐标 `[N, 3]`
- **尺度 (scales)**: 高斯椭球的三个轴长度 `[N, 3]`，通常以log形式存储
- **旋转 (quats)**: 四元数表示的旋转 `[N, 4]`
- **不透明度 (opacity)**: 每个点的可见性 `[N, 1]`，通常以logit形式存储
- **颜色 (features)**: 球谐函数(SH)系数表示的颜色 `[N, K, 3]`，其中K取决于SH度数

## 初始化流程对比

### 1. 数据源与点云获取

| 对比项 | scene_graph.py `init_gaussians_from_dataset` | EVolsplat 初始化流程 |
|--------|-----------------------------------------------|----------------------|
| **数据源** | 多源混合：LiDAR点云、随机采样点、实例分割点云 | 单目深度图 |
| **点云获取方式** | 1. Background: `dataset.get_lidar_samples()` 或 `uniform_sample_sphere()`<br>2. RigidNodes/DeformableNodes: `dataset.get_init_objects()`<br>3. SMPLNodes: `dataset.get_init_smpl_objects()` | `init_mono_points_from_dataset()` → `NuScenesMonoPCDGenerator` |
| **点云类型** | 分类别处理：Background、RigidNodes、DeformableNodes、SMPLNodes | 单一点云集合 |
| **预处理** | 1. 可见性检查 (`check_pts_visibility`)<br>2. 过滤实例框内点 (`filter_pts_in_boxes`)<br>3. 天空过滤（可选） | 1. 深度一致性检查<br>2. 天空过滤<br>3. 边界框裁剪<br>4. 稀疏度过滤 |

### 2. 高斯参数初始化

| 对比项 | scene_graph.py | EVolsplat |
|--------|----------------|-----------|
| **调用方法** | `model.create_from_pcd(init_means, init_colors)` | `populate_modules()` 或 `init_mono_points_from_dataset()` → `_convert_to_tensor()` |
| **位置 (means)** | 直接使用点云坐标：`Parameter(init_means)` | 从点云读取：`self.means = torch.from_numpy(points)` |
| **颜色初始化** | RGB → SH转换：<br>- `RGB2SH(init_colors)`<br>- 初始化SH系数：`features_dc` 和 `features_rest` | 直接存储RGB：<br>- `self.anchor_feats = torch.from_numpy(colors)`<br>- 后续通过MLP预测SH系数 |
| **尺度 (scales)** | KNN计算：<br>- `k_nearest_sklearn(means, 3)`<br>- `scales = log(avg_dist.repeat(1, 3))` | 两种方式：<br>1. KNN计算（populate_modules）：`log(avg_dist.repeat(1, 3))`<br>2. 固定初始值（init_mono）：`log(ones * initial_scale)` |
| **旋转 (quats)** | 随机初始化：`random_quat_tensor(num_points)` | 不直接初始化，通过MLP预测 |
| **不透明度 (opacity)** | 固定初始值：`logit(0.1 * ones)` | 不直接初始化，通过MLP预测 |
| **偏移 (offset)** | 不初始化 | 初始化为零：`torch.zeros_like(means)` |

### 3. 关键组件

| 组件 | scene_graph.py | EVolsplat |
|------|----------------|-----------|
| **点云处理** | `DrivingDataset` 类方法：<br>- `get_lidar_samples()`<br>- `get_init_objects()`<br>- `check_pts_visibility()`<br>- `filter_pts_in_boxes()` | `NuScenesMonoPCDGenerator` 类：<br>- 深度图加载与处理<br>- 点云累积<br>- 深度一致性检查 |
| **高斯模型** | 分类别模型：<br>- `VanillaGaussians` (Background)<br>- `RigidNodes`<br>- `DeformableNodes`<br>- `SMPLNodes` | 单一模型：<br>- `EvolSplatModel` |
| **参数存储** | 直接存储为可训练参数：<br>- `Parameter(means)`<br>- `Parameter(scales)`<br>- `Parameter(quats)`<br>- `Parameter(opacity)` | 存储为普通tensor，后续通过MLP预测：<br>- `self.means`<br>- `self.anchor_feats`<br>- `self.scales`<br>- `self.offset` |
| **MLP解码器** | 无（直接优化参数） | 有：<br>- `gaussion_decoder` (颜色/SH)<br>- `mlp_conv` (尺度+旋转)<br>- `mlp_opacity` (不透明度)<br>- `mlp_offset` (位置偏移) |

### 4. 数据流对比

#### scene_graph.py 数据流

```
数据集 (DrivingDataset)
    ↓
[点云获取]
├─ Background: LiDAR采样 / 随机采样 → 可见性检查 → 过滤实例框
├─ RigidNodes: 实例分割点云
├─ DeformableNodes: 实例分割点云
└─ SMPLNodes: SMPL模型点云
    ↓
[分类别初始化]
├─ Background → create_from_pcd()
│   ├─ means = init_means
│   ├─ scales = log(KNN距离)
│   ├─ quats = 随机四元数
│   ├─ opacity = logit(0.1)
│   └─ features = RGB2SH(colors)
├─ RigidNodes → create_from_pcd(instance_pts_dict)
└─ DeformableNodes/SMPLNodes → create_from_pcd(instance_pts_dict)
    ↓
[可训练参数]
Parameter(means, scales, quats, opacity, features)
```

#### EVolsplat 数据流

```
数据集 (DrivingDataset)
    ↓
[单目深度图处理]
init_mono_points_from_dataset()
    ↓
NuScenesMonoPCDGenerator
├─ 加载深度图
├─ 深度一致性检查
├─ 天空过滤
├─ 边界框裁剪
└─ 点云累积
    ↓
[转换为Tensor]
_convert_to_tensor()
├─ means = 点云坐标
├─ anchor_feats = RGB颜色
├─ scales = log(固定值) 或 log(KNN距离)
└─ offset = zeros
    ↓
[存储为普通Tensor]
self.means, self.anchor_feats, self.scales, self.offset
    ↓
[训练时通过MLP预测]
forward() → _generate_gaussians_from_features()
├─ 3D特征提取 (SparseCostRegNet)
├─ 2D特征采样 (Projector)
├─ 特征融合
└─ MLP解码器预测：
    ├─ 颜色 (SH系数)
    ├─ 尺度 + 旋转
    ├─ 不透明度
    └─ 位置偏移
```

### 5. 主要差异总结

| 维度 | scene_graph.py | EVolsplat |
|------|----------------|-----------|
| **初始化策略** | 直接初始化所有高斯参数为可训练参数 | 仅初始化种子点，参数通过MLP动态预测 |
| **点云来源** | 多源：LiDAR + 随机采样 + 实例分割 | 单一：单目深度图 |
| **参数优化** | 直接优化高斯参数 | 优化MLP权重，间接影响高斯参数 |
| **场景表示** | 分类别表示（背景、刚体、可变形、人体） | 统一表示 |
| **颜色处理** | 初始化时转换为SH系数 | 存储RGB，训练时通过MLP预测SH |
| **旋转/不透明度** | 初始化时设置 | 完全通过MLP预测 |
| **位置更新** | 直接优化means | 通过offset预测更新means |

### 6. 适用场景

| 方法 | 适用场景 |
|------|----------|
| **scene_graph.py** | - 需要细粒度场景分解（背景/物体/人体）<br>- 有LiDAR数据或实例分割结果<br>- 需要直接控制高斯参数优化<br>- 多类别场景表示 |
| **EVolsplat** | - 仅有单目图像和深度图<br>- 需要端到端学习场景表示<br>- 统一场景表示即可<br>- 通过特征学习优化渲染质量 |

