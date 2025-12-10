# 项目设计文档

## 概述

本项目旨在整合 **drivestudio** 和 **EVolSplat** 两个框架，创建一个统一的3D高斯场景重建系统。项目需要：

1. 复用 EVolSplat 的核心算法组件
2. 使用 drivestudio 的渲染系统（rigid node + background node 混合渲染）
3. 添加新的 nuScenes 数据加载器（直接读取，无需预处理）
4. 统一训练和评估流程

## 项目架构设计

### 1. 目录结构建议

```
drivestudio-coding/
├── datasets/                    # 数据集加载器
│   ├── nuscenes/
│   │   ├── nuscenes_loader.py   # 新的直接读取loader（整合preprocess逻辑）
│   │   └── nuscenes_sourceloader.py  # 现有loader（保留作为参考）
│   ├── base/                    # 基础数据类
│   └── ...
├── models/
│   ├── gaussians/               # 高斯表示
│   │   ├── vanilla.py          # 基础高斯（drivestudio）
│   │   └── ...
│   ├── nodes/                   # 节点类型
│   │   ├── rigid.py            # Rigid节点（drivestudio）
│   │   └── ...
│   ├── evol_splat/             # EVolSplat相关组件（新建）
│   │   ├── __init__.py
│   │   ├── encoder.py          # 从EVolSplat复用的编码器
│   │   ├── transformer.py     # 从EVolSplat复用的transformer
│   │   └── volume.py           # 从EVolSplat复用的volume相关
│   ├── trainers/               # 训练器
│   │   ├── base.py             # 基础训练器（drivestudio）
│   │   ├── scene_graph.py      # 场景图训练器（drivestudio）
│   │   └── evol_splat_trainer.py  # EVolSplat训练器（新建，整合）
│   └── ...
├── third_party/
│   └── EVolSplat/              # EVolSplat原始代码（保留作为参考）
│       └── nerfstudio/
├── configs/
│   ├── datasets/
│   │   └── nuscenes/
│   │       └── 6cams.yaml      # nuScenes配置
│   └── ...
├── tools/
│   ├── train.py                # 训练脚本（drivestudio）
│   └── eval.py                 # 评估脚本（drivestudio）
└── docs/
    └── PROJECT_DESIGN.md        # 本文档
```

### 2. 核心设计决策

#### 2.1 数据加载器设计

**目标**：创建新的 nuScenes 数据加载器，直接读取原始数据，无需预处理步骤。

**设计思路**：

- 参考 `nuscenes_preprocess.py` 的数据处理逻辑
- 在 `nuscenes_loader.py` 中直接实现数据读取和转换
- 输出格式与 `nuscenes_sourceloader.py` 兼容
- 配置参数参考 `6cams.yaml`

**实现要点**：

```python
# datasets/nuscenes/nuscenes_loader.py
class NuScenesDirectLoader:
    """
    直接从nuScenes原始数据读取，无需预处理
    整合了nuscenes_preprocess.py的逻辑
    """
    def __init__(self, data_root, split, scene_idx, ...):
        # 直接使用nuscenes SDK加载数据
        self.nusc = NuScenes(version=split, dataroot=data_root)
      
    def load_images(self):
        # 直接读取原始图像，不依赖processed文件夹
      
    def load_calibrations(self):
        # 直接从nuscenes API获取相机内外参
      
    def load_lidar(self):
        # 直接从nuscenes API获取LiDAR数据
      
    def load_objects(self):
        # 直接从nuscenes API获取对象标注
```

**优势**：

- 无需预处理步骤，减少存储空间
- 更灵活的数据处理
- 与现有sourceloader接口兼容

#### 2.2 EVolSplat 组件复用

**目标**：复用 EVolSplat 的核心算法组件，同时使用 drivestudio 的渲染系统。

**设计思路**：

- 将 EVolSplat 的核心组件提取到 `models/evol_splat/` 目录
- 保持 EVolSplat 的编码器、transformer、volume 等组件
- 替换渲染部分，使用 drivestudio 的渲染系统

**需要复用的组件**：

1. **编码器** (`third_party/EVolSplat/nerfstudio/Encoder/`)

   - 点云编码
   - 特征提取
2. **Transformer** (`third_party/EVolSplat/nerfstudio/transformer/`)

   - 时序建模
   - 特征融合
3. **Volume相关** (`third_party/EVolSplat/nerfstudio/field_components/`)

   - 体积表示
   - 稀疏卷积

**集成方式**：

```python
# models/evol_splat/__init__.py
from .encoder import EVolSplatEncoder
from .transformer import EVolSplatTransformer
from .volume import EVolSplatVolume

# models/trainers/evol_splat_trainer.py
from models.evol_splat import EVolSplatEncoder, EVolSplatTransformer
from models.trainers.scene_graph import MultiTrainer  # 复用渲染系统

class EVolSplatTrainer(MultiTrainer):
    def __init__(self, ...):
        super().__init__(...)
        # 使用EVolSplat的编码器和transformer
        self.encoder = EVolSplatEncoder(...)
        self.transformer = EVolSplatTransformer(...)
        # 使用drivestudio的渲染系统
        # (继承自MultiTrainer)
```

#### 2.3 渲染系统选择

**决策**：使用 drivestudio 的渲染系统（rigid node + background node 混合渲染）

**理由**：

1. **更灵活的场景表示**：

   - Background node：静态场景
   - Rigid node：刚体对象（车辆等）
   - 支持多节点混合渲染
2. **更好的动态场景处理**：

   - 支持实例级别的刚体变换
   - 支持SMPL节点（人体）
   - 支持可变形节点
3. **与EVolSplat的兼容性**：

   - EVolSplat的输出（高斯参数）可以直接用于drivestudio的渲染
   - 只需要适配数据格式

**渲染流程**：

```python
# models/trainers/scene_graph.py (已存在)
class MultiTrainer:
    def forward(self, image_infos, camera_infos):
        # 1. 收集各节点的Gaussians
        gs = self.collect_gaussians(...)
      
        # 2. 使用drivestudio的渲染系统
        outputs = self.render_gaussians(gs, ...)
      
        # 3. 混合渲染（Background + RigidNodes + ...）
        return outputs
```

#### 2.4 训练和评估代码选择

**决策**：使用 drivestudio 的训练和评估代码作为基础

**理由**：

1. **更完整的训练流程**：

   - 支持多节点训练
   - 支持场景图结构
   - 支持各种正则化损失
2. **更好的评估功能**：

   - 支持PSNR、SSIM、LPIPS等指标
   - 支持分节点渲染可视化
   - 支持视频生成
3. **易于扩展**：

   - 可以轻松添加EVolSplat特定的损失函数
   - 可以集成EVolSplat的训练策略

**集成方式**：

```python
# tools/train.py (修改)
# 保持drivestudio的训练框架
# 在trainer初始化时选择使用EVolSplatTrainer或MultiTrainer

def setup(args):
    cfg = OmegaConf.load(args.config_file)
  
    # 根据配置选择trainer
    if cfg.model.type == "evolsplat":
        trainer = EVolSplatTrainer(...)
    else:
        trainer = MultiTrainer(...)
  
    return trainer, dataset, cfg
```

## 3. 实现步骤

### Phase 1: 数据加载器

1. ✅ 分析 `nuscenes_preprocess.py` 的数据处理逻辑
2. ✅ 分析 `nuscenes_sourceloader.py` 的接口要求
3. ⬜ 实现 `nuscenes_loader.py`，直接读取原始数据
4. ⬜ 测试数据加载器，确保输出格式兼容

### Phase 2: EVolSplat组件提取

1. ⬜ 分析 EVolSplat 的核心组件
2. ⬜ 提取编码器、transformer、volume等组件到 `models/evol_splat/`
3. ⬜ 适配接口，使其与drivestudio兼容
4. ⬜ 编写单元测试

### Phase 3: 训练器集成

1. ⬜ 创建 `EVolSplatTrainer`，继承 `MultiTrainer`
2. ⬜ 集成EVolSplat的编码器和transformer
3. ⬜ 保持drivestudio的渲染系统
4. ⬜ 添加EVolSplat特定的损失函数（如需要）

### Phase 4: 配置和测试

1. ⬜ 创建EVolSplat的训练配置
2. ⬜ 测试端到端训练流程
3. ⬜ 验证渲染质量
4. ⬜ 性能优化

## 4. 关键接口设计

### 4.1 数据加载器接口

```python
class NuScenesDirectLoader:
    """直接读取nuScenes数据，无需预处理"""
  
    def __init__(
        self,
        data_root: str,           # nuScenes原始数据路径
        split: str,               # 'v1.0-mini', 'v1.0-trainval', etc.
        scene_idx: int,           # 场景索引
        start_timestep: int = 0,   # 起始时间步
        end_timestep: int = -1,    # 结束时间步（-1表示到最后）
        cameras: List[int] = [0, 1, 2, 3, 4, 5],  # 使用的相机
        interpolate_N: int = 0,    # 插值帧数（0表示不插值）
    ):
        pass
  
    def load_cameras(self) -> Dict[str, CameraData]:
        """加载相机数据"""
        pass
  
    def load_lidar(self) -> SceneLidarSource:
        """加载LiDAR数据"""
        pass
  
    def load_objects(self) -> Dict:
        """加载对象标注"""
        pass
```

### 4.2 EVolSplat组件接口

```python
class EVolSplatEncoder(nn.Module):
    """EVolSplat编码器"""
  
    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """编码点云和特征"""
        pass

class EVolSplatTransformer(nn.Module):
    """EVolSplat Transformer"""
  
    def forward(self, encoded_features: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """时序特征融合"""
        pass

class EVolSplatVolume(nn.Module):
    """EVolSplat体积表示"""
  
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成体积表示"""
        pass
```

### 4.3 训练器接口

```python
class EVolSplatTrainer(MultiTrainer):
    """整合EVolSplat和drivestudio的训练器"""
  
    def __init__(
        self,
        evol_splat_config: OmegaConf,  # EVolSplat配置
        **kwargs  # MultiTrainer的其他参数
    ):
        super().__init__(**kwargs)
        # 初始化EVolSplat组件
        self.encoder = EVolSplatEncoder(...)
        self.transformer = EVolSplatTransformer(...)
        # 渲染系统继承自MultiTrainer
  
    def forward(self, image_infos, camera_infos):
        # 1. 使用EVolSplat编码和transformer处理
        # 2. 生成Gaussians
        # 3. 使用drivestudio渲染系统渲染
        pass
```

## 5. 配置示例

### 5.1 nuScenes数据配置

```yaml
# configs/datasets/nuscenes/6cams_direct.yaml
data:
  data_root: data/nuscenes/raw  # 原始数据路径
  dataset: nuscenes
  split: v1.0-mini
  scene_idx: 0
  start_timestep: 0
  end_timestep: -1
  pixel_source:
    type: datasets.nuscenes.nuscenes_loader.NuScenesDirectLoader
    cameras: [0, 1, 2, 3, 4, 5]
    interpolate_N: 0  # 不插值，使用原始2Hz数据
    # ... 其他参数
```

### 5.2 EVolSplat训练配置

```yaml
# configs/evolsplat_nuscenes.yaml
model:
  type: models.trainers.evol_splat_trainer.EVolSplatTrainer
  
  # EVolSplat组件配置
  evol_splat:
    encoder:
      # 编码器配置
    transformer:
      # Transformer配置
    volume:
      # Volume配置
  
  # drivestudio渲染配置
  Background:
    type: models.gaussians.VanillaGaussians
    # ...
  RigidNodes:
    type: models.nodes.RigidNodes
    # ...
```

## 6. 注意事项

### 6.1 数据格式兼容性

- 确保新的loader输出格式与现有sourceloader兼容
- 注意坐标系转换（OpenCV vs nuScenes）
- 注意时间戳对齐

### 6.2 依赖管理

- EVolSplat可能依赖特定版本的nerfstudio
- 需要处理依赖冲突
- 考虑使用条件导入

### 6.3 性能考虑

- 直接读取原始数据可能比预处理慢
- 考虑添加缓存机制
- 优化数据加载流程

### 6.4 代码维护

- 保持EVolSplat原始代码在third_party中作为参考
- 提取的组件要有清晰的文档
- 保持与原始实现的兼容性

## 7. 总结

本项目采用**混合架构**：

- **数据层**：新的直接读取loader（整合preprocess逻辑）
- **模型层**：EVolSplat组件 + drivestudio渲染系统
- **训练层**：drivestudio训练框架 + EVolSplat组件集成

这种设计的优势：

1. ✅ 充分利用两个框架的优势
2. ✅ 保持代码模块化和可维护性
3. ✅ 易于扩展和修改
4. ✅ 避免重复实现
