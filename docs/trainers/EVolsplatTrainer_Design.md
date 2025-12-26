# EVolsplatTrainer 设计文档

## 概述

本文档设计 `EVolsplatTrainer` 类，用于实现基于 EVolsplat 的 feed-forward 3DGS 训练流程。该类从 `MultiSceneDataset` 获取数据，使用 RGB 点云初始化 3DGS node，并支持多段、多场景的训练。

**核心特点**：
- **不继承 MultiTrainer**：因为 EVolsplat 是 feed-forward 3DGS（通过 MLP 预测参数），而 MultiTrainer 是 pre-scene 3DGS（直接优化参数），差异较大
- **使用 VanillaGaussians node 形式**：复用 `models/gaussians/vanilla.py` 的 node 结构，但以 feed-forward 方式使用
- **RGB 点云初始化**：从 `RGBPointCloudGenerator` 获取点云，转换为 3DGS node
- **混合初始化策略**：颜色、旋转、不透明度使用 `scene_graph.py` 的初始化方式，尺度、偏移使用 EVolsplat 的初始化方式
- **多 target 训练**：支持对多张 target 图像进行渲染和反向传播，最后一起 step
- **多次反向传播**：必须使用多次反向传播（遗弃计算图）的方式计算累计梯度，以节省显存

---

## 核心概念

### 1. 训练流程

```
MultiSceneDataset
    ↓
[判断是否新段]
    ├─ 是 → 重置 node
    │   ├─ RGB 点云生成 (RGBPointCloudGenerator)
    │   ├─ 将 RGB 点云初始化为 VanillaGaussians node
    │   └─ 初始化同一段内必要的维持信息 (offset, frozen_volume)
    └─ 否 → 继续使用现有 node
    ↓
[读取 batch]
    ├─ source: [num_source_keyframes * num_cams, H, W, 3]
    └─ target: [num_target_keyframes * num_cams, H, W, 3]
    ↓
[训练循环]
    ├─ optimizer.zero_grad()
    ├─ [共享特征提取]（只执行一次）
    │   ├─ 构建3D特征体积（construct_sparse_tensor + sparse_conv）
    │   ├─ 2D特征采样（projector.sample_within_window）
    │   └─ 准备共享状态（dense_volume, sampled_feat等）
    ├─ [对每个 target view 循环]
    │   ├─ ob_view 和 ob_dist 计算
    │   ├─ MLP预测参数（颜色、尺度、旋转、不透明度、offset）
    │   ├─ render_for_target_view(target_view, shared_state)
    │   ├─ render_for_target_view_background(target_view, shared_state)
    │   ├─ 图像合起来
    │   ├─ loss_v = loss_fn(outs_v, gt_v) / len(target_views)
    │   └─ loss_v.backward()（不 retain_graph，遗弃计算图）
    ├─ [梯度裁剪]（可选）
    │   └─ clip_grad_norm_(model.parameters(), max_norm)
    └─ optimizer.step()（所有 target 一起更新）
    ↓
[更新 offset]
    └─ offset_crop.detach() → 更新当前段的 node.offset
    ↓
[评估]
    └─ 使用评估数据集进行渲染和指标计算
```

### 2. Node 初始化策略

**混合初始化策略**（参考 `docs/gaussian_initialization_comparison.md`）：

| 参数 | 初始化方式 | 来源 | 说明 |
|------|-----------|------|------|
| **位置 (means)** | 直接使用点云坐标 | RGB 点云 | 存储为普通 tensor，不参与梯度 |
| **颜色 (features)** | RGB → SH 转换，初始化 `features_dc` 和 `features_rest` | scene_graph.py | 作为 anchor_feats 存储，后续通过 MLP 预测 |
| **旋转 (quats)** | 随机四元数 | scene_graph.py | 仅用于初始化，实际通过 MLP 预测 |
| **不透明度 (opacity)** | `logit(0.1 * ones)` | scene_graph.py | 仅用于初始化，实际通过 MLP 预测 |
| **尺度 (scales)** | `log(KNN距离)` 或 `log(固定值)` | EVolsplat | 作为初始尺度存储，实际通过 MLP 预测增量 |
| **偏移 (offset)** | `zeros` | EVolsplat | 段级别管理，每次迭代更新 |

**注意**：
- 先只考虑静态场景，动态场景（RigidNodes、DeformableNodes、SMPLNodes）后续扩展
- 虽然使用 VanillaGaussians node 形式，但参数不是直接优化的 Parameter，而是通过 MLP 预测的

---

## 类设计

### 1. EVolsplatTrainer 类结构

#### 1.1 类定义

```python
class EVolsplatTrainer(nn.Module):
    """
    EVolsplat feed-forward 3DGS trainer.
    
    核心功能：
    1. 管理多场景、多段的训练流程
    2. 使用 RGB 点云初始化 3DGS node
    3. 通过 MLP 预测高斯参数（feed-forward）
    4. 支持多 target 图像训练
    5. 段级别的 offset 管理
    """
```

#### 1.2 初始化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `dataset` | `MultiSceneDataset` | - | 多场景数据集 |
| `pointcloud_generator` | `RGBPointCloudGenerator` | - | RGB 点云生成器 |
| `config` | `OmegaConf` | - | 训练配置 |
| `device` | `torch.device` | `cuda` | 训练设备 |
| `log_dir` | `str` | `"./logs"` | 日志目录 |

#### 1.3 核心属性

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `nodes` | `Dict[Tuple[int, int], VanillaGaussians]` | 段级别的 node 字典，key 为 (scene_id, segment_id) |
| `offset_cache` | `Dict[Tuple[int, int], torch.Tensor]` | 段级别的 offset 缓存，key 为 (scene_id, segment_id) |
| `frozen_volume_cache` | `Dict[Tuple[int, int], torch.Tensor]` | 段级别的冻结特征体积缓存 |
| `sparse_conv` | `SparseCostRegNet` | 3D 稀疏卷积网络 |
| `projector` | `Projector` | 2D 特征投影器 |
| `gaussion_decoder` | `MLP` | 高斯颜色解码器（预测 SH 系数） |
| `mlp_conv` | `MLP` | 尺度+旋转预测 MLP |
| `mlp_opacity` | `MLP` | 不透明度预测 MLP |
| `mlp_offset` | `MLP` | 位置偏移预测 MLP |
| `bg_field` | `MLP` | 背景场 MLP（可选） |
| `optimizer` | `torch.optim.Optimizer` | 优化器 |
| `scaler` | `torch.cuda.amp.GradScaler` | 混合精度梯度缩放器（可选） |
| `step` | `int` | 当前训练步数 |

#### 1.4 配置参数

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model.sparseConv_outdim` | `int` | 32 | 稀疏卷积输出维度 |
| `model.local_radius` | `int` | 1 | 局部半径 |
| `model.offset_max` | `float` | 0.1 | offset 最大范围 |
| `model.num_neighbour_select` | `int` | 4 | 邻居选择数量 |
| `model.sh_degree` | `int` | 1 | 球谐函数度数 |
| `model.voxel_size` | `float` | 0.1 | 体素大小 |
| `model.freeze_volume` | `bool` | `False` | 是否冻结特征体积 |
| `model.enable_background` | `bool` | `True` | 是否启用背景模型 |
| `training.max_iterations` | `int` | 30000 | 最大训练迭代数 |
| `training.gradient_clip_val` | `float` | `None` | 梯度裁剪值（None 表示不裁剪） |
| `training.use_mixed_precision` | `bool` | `False` | 是否使用混合精度训练 |
| `training.save_checkpoint_freq` | `int` | 5000 | 检查点保存频率 |
| `training.save_checkpoint_dir` | `str` | `"./checkpoints"` | 检查点保存目录 |
| `loss.ssim_lambda` | `float` | 0.2 | SSIM 损失权重 |
| `loss.entropy_loss` | `float` | 0.1 | 熵损失权重 |

---

## 核心方法设计

### 1. 初始化方法

#### 1.1 `__init__()`

**功能**：初始化 trainer，创建 MLP 网络和优化器。

**关键步骤**：
1. 初始化 MLP 网络（sparse_conv, projector, gaussion_decoder, mlp_conv, mlp_opacity, mlp_offset）
2. 初始化优化器
3. 初始化混合精度缩放器（如果启用）
4. 初始化节点字典和缓存

#### 1.2 `init_node_from_pointcloud()`

**功能**：从 RGB 点云初始化 VanillaGaussians node。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `scene_id` | `int` | 场景ID |
| `segment_id` | `int` | 段ID |
| `pointcloud` | `o3d.geometry.PointCloud` | RGB 点云 |

**返回值**：`VanillaGaussians` node

**流程**：
1. 从点云提取位置和颜色
2. 初始化 VanillaGaussians node（使用混合初始化策略）
3. 计算初始尺度（KNN）
4. 初始化 offset 为零
5. 存储到 `self.nodes[(scene_id, segment_id)]`

### 2. 训练方法

#### 2.1 `train_step()`

**功能**：执行一个训练步骤。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `batch` | `Dict` | 训练批次，包含 source 和 target 数据 |

**返回值**：`Dict[str, torch.Tensor]` 损失字典

**流程**：
1. 检查是否为新段，如果是则初始化 node
2. 获取当前段的 node 和 offset
3. 执行共享特征提取（只执行一次）
4. 对每个 target view 循环：
   - 计算 ob_view 和 ob_dist
   - 通过 MLP 预测参数
   - 渲染图像
   - 计算损失并反向传播（不 retain_graph）
5. 梯度裁剪（如果启用）
6. 优化器 step
7. 更新 offset（detach 后更新 node）

#### 2.2 `extract_shared_features()`

**功能**：提取共享特征（只执行一次，所有 target 共享）。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `batch` | `Dict` | 训练批次 |
| `node` | `VanillaGaussians` | 当前段的 node |
| `offset` | `torch.Tensor` | 当前段的 offset |

**返回值**：`Dict` 共享状态（dense_volume, sampled_feat, valid_mask 等）

**流程**：
1. 构建稀疏张量（construct_sparse_tensor）
2. 稀疏卷积（sparse_conv）
3. 转换为密集体积（sparse_to_dense_volume）
4. 2D 特征采样（projector.sample_within_window）
5. 返回共享状态

#### 2.3 `render_for_target_view()`

**功能**：为单个 target view 渲染图像。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `target_view` | `Dict` | target view 数据（图像、相机参数等） |
| `shared_state` | `Dict` | 共享特征状态 |
| `node` | `VanillaGaussians` | 当前段的 node |
| `offset` | `torch.Tensor` | 当前段的 offset |

**返回值**：`Dict[str, torch.Tensor]` 渲染结果（rgb, depth, accumulation 等）

**流程**：
1. 使用共享状态计算 ob_view 和 ob_dist
2. 通过 MLP 预测参数（颜色、尺度、旋转、不透明度、offset）
3. 更新位置（means + offset）
4. 光栅化渲染
5. 渲染背景（如果启用）
6. 合成最终图像

### 3. 损失计算

#### 3.1 `compute_loss()`

**功能**：计算损失。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `outputs` | `Dict[str, torch.Tensor]` | 渲染输出 |
| `gt_image` | `torch.Tensor` | 真实图像 |

**返回值**：`Dict[str, torch.Tensor]` 损失字典

**损失项**：

| 损失项 | 公式 | 权重 |
|--------|------|------|
| **L1 损失** | `\|pred_rgb - gt_rgb\|_1` | `1 - ssim_lambda` |
| **SSIM 损失** | `1 - SSIM(pred_rgb, gt_rgb)` | `ssim_lambda` |
| **熵损失** | `-accumulation * log(accumulation) - (1-accumulation) * log(1-accumulation)` | `entropy_loss`（每10步） |

### 4. Offset 管理

#### 4.1 `update_offset()`

**功能**：更新当前段的 offset。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `scene_id` | `int` | 场景ID |
| `segment_id` | `int` | 段ID |
| `offset_crop` | `torch.Tensor` | 预测的 offset（带梯度） |
| `projection_mask` | `torch.Tensor` | 投影掩码 |

**流程**：
1. `offset_crop = offset_crop.detach().cpu()`
2. 更新 `self.offset_cache[(scene_id, segment_id)][projection_mask] = offset_crop`
3. 更新 `self.nodes[(scene_id, segment_id)].offset = offset_crop`（如果 node 有 offset 属性）

**注意**：
- Offset 必须 detach，不参与梯度
- 更新到当前段的 node，而不是直接更新 `self.offset[scene_id][segment_id]`
- Offset 是段级别的，每个段独立管理

### 5. 检查点管理

#### 5.1 `save_checkpoint()`

**功能**：保存检查点。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `step` | `int` | 当前步数 |
| `is_final` | `bool` | 是否为最终检查点 |

**保存内容**：

| 内容 | 说明 |
|------|------|
| `step` | 当前训练步数 |
| `model_state_dict` | 模型状态（MLP 网络） |
| `optimizer_state_dict` | 优化器状态 |
| `scaler_state_dict` | 混合精度缩放器状态（如果启用） |
| `nodes_state_dict` | 节点状态（means, scales, anchor_feats 等） |
| `offset_cache` | Offset 缓存 |
| `frozen_volume_cache` | 冻结特征体积缓存（如果启用） |
| `config` | 训练配置 |

**文件命名**：
- 普通检查点：`checkpoint_step_{step:06d}.pth`
- 最终检查点：`checkpoint_final.pth`

#### 5.2 `load_checkpoint()`

**功能**：加载检查点。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `checkpoint_path` | `str` | 检查点路径 |
| `load_only_model` | `bool` | 是否只加载模型（不加载优化器等） |

**加载流程**：
1. 加载检查点文件
2. 恢复模型状态
3. 恢复优化器状态（如果 `load_only_model=False`）
4. 恢复混合精度缩放器状态（如果启用且 `load_only_model=False`）
5. 恢复节点状态和缓存
6. 返回恢复的步数

### 6. 梯度裁剪

#### 6.1 `clip_gradients()`

**功能**：裁剪梯度。

**参数**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `max_norm` | `float` | 最大梯度范数 |

**实现**：
```python
torch.nn.utils.clip_grad_norm_(self.get_trainable_parameters(), max_norm)
```

**调用时机**：在 `train_step()` 中，所有 target 反向传播完成后，`optimizer.step()` 之前。

**注意事项**：
- 只裁剪 MLP 网络的梯度，不裁剪 node 的梯度（node 参数不参与训练）
- 如果使用混合精度，需要在 `scaler.scale()` 之后裁剪

### 7. 混合精度训练

#### 7.1 混合精度支持

**启用条件**：`config.training.use_mixed_precision = True`

**实现方式**：
1. 使用 `torch.cuda.amp.autocast()` 包装前向传播
2. 使用 `torch.cuda.amp.GradScaler` 缩放梯度
3. 在反向传播前缩放损失：`scaler.scale(loss).backward()`
4. 在优化器 step 前缩放梯度：`scaler.step(optimizer)`
5. 更新缩放器：`scaler.update()`

**关键代码结构**：
```python
with torch.cuda.amp.autocast():
    # 前向传播
    outputs = self.render_for_target_view(...)
    loss = self.compute_loss(outputs, gt_image)

# 反向传播
scaler.scale(loss).backward()

# 梯度裁剪（在 scaler.unscale() 之后）
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(..., max_norm)

# 优化器 step
scaler.step(optimizer)
scaler.update()
```

**注意事项**：
- 光栅化操作（rasterization）可能不支持混合精度，需要测试
- 如果出现数值不稳定，可以降低缩放器的初始 scale 或使用动态缩放

---

## 可行性分析

### 1. 多次反向传播（遗弃计算图）

**可行性**：✅ **已证实可行**

**实现方式**：
- 对每个 target view 分别计算损失并调用 `loss.backward()`（不设置 `retain_graph=True`）
- 每次反向传播后，计算图被释放，节省显存
- 梯度自动累积到参数中
- 最后统一调用 `optimizer.step()`

**注意事项**：
- 必须确保每次 `backward()` 后不保留计算图
- 梯度裁剪需要在所有反向传播完成后进行

### 2. Offset 更新机制

**可行性**：✅ **可行**

**实现方式**：
- Offset 通过 MLP 预测，但必须 detach
- 更新到当前段的 node，而不是全局的 `self.offset[scene_id][segment_id]`
- 每个段独立管理 offset

**与 EVolsplat 原实现的差异**：
- 原实现：`self.offset[scene_id][projection_mask] = offset_crop.detach().cpu()`
- 新实现：`self.nodes[(scene_id, segment_id)].offset = offset_crop.detach().cpu()`

**优势**：
- 段级别的 offset 管理更清晰
- 便于检查点和恢复
- 支持多段并行训练（未来扩展）

### 3. 检查点保存

**可行性**：✅ **可行**

**关键点**：
- 需要保存 MLP 网络状态
- 需要保存节点状态（means, scales, anchor_feats 等）
- 需要保存 offset 缓存
- 需要保存优化器状态（用于恢复训练）
- 需要保存混合精度缩放器状态（如果启用）

**潜在问题**：
- 节点数量可能很大，检查点文件可能很大
- 解决方案：只保存必要的节点状态，或使用压缩

### 4. 梯度裁剪

**可行性**：✅ **可行**

**实现方式**：
- 使用 `torch.nn.utils.clip_grad_norm_()`
- 只裁剪 MLP 网络的梯度
- 如果使用混合精度，需要在 `scaler.unscale()` 之后裁剪

**注意事项**：
- 梯度裁剪值需要根据模型大小调整
- 过小的裁剪值可能影响训练效果
- 建议从 1.0 开始尝试

### 5. 混合精度训练

**可行性**：⚠️ **需要测试**

**潜在问题**：
1. **光栅化操作兼容性**：`gsplat` 库的光栅化操作可能不支持混合精度
   - 解决方案：在光栅化前将相关张量转换为 float32
2. **数值稳定性**：某些操作（如 exp, log）在 float16 下可能不稳定
   - 解决方案：使用 `torch.cuda.amp.autocast()` 的 `enabled` 参数控制特定操作
3. **性能提升**：混合精度训练可能不会带来显著的性能提升（取决于瓶颈）
   - 解决方案：先实现，根据实际效果决定是否启用

**方案**：
- 先实现基础训练流程

---

## 反直觉检查

### 1. Offset 更新时机

**直觉**：Offset 应该在每次反向传播后更新。

**实际情况**：
- Offset 不参与梯度，所以不需要在反向传播后更新
- Offset 应该在前向传播中预测后立即更新（detach）
- 更新后的 offset 用于下次迭代的特征计算

**结论**：✅ **设计正确**

### 2. 共享特征提取

**直觉**：每个 target view 都需要重新提取特征。

**实际情况**：
- 3D 特征体积和 2D 特征采样只依赖于 source 图像和当前段的 node
- 所有 target view 共享相同的 source 图像和 node
- 因此可以只提取一次共享特征，所有 target view 复用

**结论**：✅ **设计正确，可以节省计算**

### 3. 多次反向传播的梯度累积

**直觉**：多次反向传播会导致梯度被覆盖。

**实际情况**：
- PyTorch 的梯度是累积的，每次 `backward()` 会将梯度累加到参数的 `.grad` 属性
- 只要不调用 `optimizer.zero_grad()`，梯度就会累积
- 因此多次反向传播可以正确累积梯度

**结论**：✅ **设计正确**

### 4. Node 的参数管理

**直觉**：VanillaGaussians node 的参数应该参与训练。

**实际情况**：
- 在 feed-forward 3DGS 中，node 的参数（means, scales 等）不是直接优化的
- 这些参数只是初始值，实际渲染时通过 MLP 预测
- 因此 node 的参数不需要是 `Parameter`，可以是普通 tensor

**结论**：✅ **设计正确**

### 5. 段级别的 Offset 管理

**直觉**：Offset 应该是场景级别的，所有段共享。

**实际情况**：
- 每个段有独立的点云和 node
- 每个段的几何结构可能不同
- 因此每个段应该有独立的 offset

**结论**：✅ **设计正确**

---

## 总结

`EVolsplatTrainer` 的设计遵循以下原则：

1. **不继承 MultiTrainer**：因为 EVolsplat 是 feed-forward 3DGS，与 pre-scene 3DGS 差异较大
2. **使用 VanillaGaussians node 形式**：复用现有的 node 结构，但以 feed-forward 方式使用
3. **混合初始化策略**：结合 scene_graph.py 和 EVolsplat 的初始化方式
4. **多 Target 训练**：支持对多张 target 图像进行训练
5. **段级别管理**：支持多段、多场景的训练流程
6. **多次反向传播**：使用遗弃计算图的方式节省显存
7. **Offset 段级别管理**：每个段独立管理 offset，更新到 node 而不是全局缓存

该设计允许在不修改 EVolsplat 核心代码的情况下，实现基于 RGB 点云的 feed-forward 3DGS 训练，同时支持多段、多场景的训练流程。检查点保存、梯度裁剪、混合精度训练等高级功能均已考虑并设计，但混合精度训练需要实际测试验证可行性。
