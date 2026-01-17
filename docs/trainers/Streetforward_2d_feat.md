# StreetForward 2D+3D 特征融合方案设计

本文档描述在 StreetForwardTrainer 中引入 2D 语义特征（通过 CNN 从 source 帧提取）并与现有 3D 特征融合的设计方案。

**状态**：✅ 已实现并集成到主训练流程中

**相关文档**：
- 流程文档：`docs/trainers/StreetForward_Flow.md` - 包含完整的训练流程和 2D 特征融合步骤
- 问题分析：`docs/trainers/StreetForward_2DFeat_Issues.md` - 实现过程中的关键问题和解决方案

## 目录
1. [方案概述](#方案概述)
2. [核心设计原则](#核心设计原则)
3. [架构设计](#架构设计)
4. [2D 特征提取与反投影](#2d-特征提取与反投影)
5. [特征融合与梯度流](#特征融合与梯度流)
6. [与现有机制的集成](#与现有机制的集成)
7. [关键实现细节](#关键实现细节)

---

## 方案概述

### 目标
在保持 StreetForward 现有训练范式（proxy 梯度累积、单次反向传播）的前提下，引入 source 帧的 2D 语义特征，与 3D 体积特征融合，提升偏移量预测的准确性。

### 核心思路
- **Source 帧固定**：只对 source 时间戳的多相机图像提取 2D 特征，避免动态一致性问题
- **αT 加权反投影**：使用渲染时的 αT 贡献权重将 2D 特征反投影到每个高斯点
- **梯度兼容**：确保 CNN 梯度能通过现有 proxy 机制回传，不破坏训练范式

---

## 核心设计原则

### 1. 时间戳一致性
- **Source 帧 = 1 个时间戳**：所有 2D 特征来自同一时刻，保证动态物体状态一致
- **Source 图像 = 多张**：同一时刻的多相机（front/left/right/...）都要提取特征
- **Target 帧 = 多个**：可以跨时间戳，但 2D 特征不参与 target 帧处理

### 2. 梯度代理机制兼容
- **CNN 只跑一次**：在 inner-iteration 开始时对 source 图像跑一次 CNN
- **梯度来自所有 target**：通过 proxy 机制，所有 target 的损失梯度最终回传到 CNN
- **单次回灌**：保持现有的 `autograd.backward(render_tensors, proxy_grads)` 机制

### 3. αT 权重处理
- **αT 作为采样权重**：`w = w.detach()`（stop-grad），只用于特征聚合
- **避免高阶梯度**：不通过 αT 权重优化 offsets，避免通过 rasterizer 的二阶耦合
- **可见性对齐**：使用与 RGB 渲染相同的排序/截断规则，保证语义聚合与渲染可见性一致

### 4. 动静态分离
- **统一反投影**：对合并后的（bg + rigid 变换到 source 帧）高斯进行反投影
- **分离存储**：反投影后按原始索引拆分为 `feat_2d_bg` 和 `feat_2d_rigid`
- **独立融合**：静态和动态分别进行 2D+3D 特征融合

---

## 架构设计

### 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│              StreetForwardTrainer (with 2D+3D Features)       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │ NodeStateBg      │         │ NodeStateRigid   │          │
│  │ (Detached)       │         │ (Detached)       │          │
│  └──────────────────┘         └──────────────────┘          │
│           │                              │                  │
│           └──────────┬───────────────────┘                  │
│                      │                                      │
│           ┌──────────▼──────────┐                          │
│           │  train_iter()        │                          │
│           │  (1 source + N targets)                        │
│           └──────────┬───────────┘                          │
│                      │                                      │
│    ┌─────────────────┼─────────────────┐                   │
│    │                 │                 │                    │
│    ▼                 ▼                 ▼                    │
│ ┌─────────┐   ┌──────────┐   ┌──────────┐                 │
│ │Transform│   │ 2D CNN   │   │ 3D Vol   │                 │
│ │to Source│   │ Extract  │   │ Builder  │                 │
│ └─────────┘   └──────────┘   └──────────┘                 │
│     │              │                 │                      │
│     └──────┬───────┴────────┬────────┘                     │
│            │                │                              │
│     ┌──────▼────────────────▼──────┐                      │
│     │  αT Backprojection            │                      │
│     │  (scatter-add aggregation)     │                      │
│     └──────┬───────────────────────┘                      │
│            │                                                │
│     ┌──────▼───────────────────────┐                       │
│     │  2D+3D Feature Fusion        │                       │
│     │  concat([feat_3d, feat_2d, vis])                     │
│     └──────┬────────────────────────┘                       │
│            │                                                │
│     ┌──────▼────────────────────────┐                      │
│     │  MLP Offsets Prediction       │                      │
│     │  (existing MLPs with wider input)                    │
│     └──────┬─────────────────────────┘                      │
│            │                                                │
│     ┌──────▼─────────────────────────┐                     │
│     │  Proxy Params + Multi-target   │                     │
│     │  Gradient Accumulation         │                     │
│     └────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 关键模块

1. **2D CNN 特征提取器**：对 source 时间戳的所有相机图像提取语义特征
2. **αT 反投影模块**：使用渲染权重将 2D 特征聚合到高斯点
3. **特征融合模块**：将 2D 和 3D 特征拼接，输入到 MLP
4. **梯度回传链路**：确保 CNN 梯度能通过 proxy 机制回传

---

## 2D 特征提取与反投影

### A. CNN 特征提取

**实现**（`ImageFeatureExtractor`）：

**输入**：
- `source_images: List[Tensor]`：source 时间戳的多相机图像列表
  - 每个图像格式：`[H, W, 3]` 或 `[1, H, W, 3]`
  - 所有图像来自同一时间戳，保证动态物体状态一致

**网络结构**：
- **Backbone**：标准 UNet 架构（编码器-解码器 + 跳跃连接）
  - **编码器路径**：4 层下采样（MaxPool + DoubleConv），通道数逐步增加（32 → 64 → 128 → 256）
  - **解码器路径**：4 层上采样（Upsample + Concatenate + DoubleConv），通道数逐步减少
  - **跳跃连接**：将编码器每层的特征图连接到解码器对应层，保留细节信息
  - **输出**：特征图空间分辨率与输入一致（或按 `feature_downscale` 缩放）
  - **特点**：使用 BatchNorm 和 ReLU 激活，支持梯度回传

**输出**：
- `features_2d: List[Tensor]` - 每个视图的特征图 `[C2, H, W]`
  - `C2`：特征通道数（默认 16 或 32，由 `feat_2d_channels` 配置）
  - `H, W`

**关键设计**：
- CNN 在 inner-iteration 开始时只跑一次
- 特征分辨率与后续反投影渲染分辨率对齐（通过 `get_feature_resolution()` 计算）
- 支持批处理：多个图像通过 `torch.cat` 合并后一起处理，提高效率

### B. αT 权重获取

**实现**（`AlphaTWeightExtractor`）：

**αT 权重计算流程**：

1. **渲染获取 meta 信息**：
   - 在 source 帧下，将 rigid 变换到世界坐标，对每个视图进行渲染（`render_mode="RGB"`）
   - 从渲染元数据（`meta`）中提取：
     - `flatten_ids`：高斯 id 列表 `[n_isects]`
     - `isect_offsets`：每个 tile 的起始/结束索引 `[tile_h, tile_w]`
     - `conics`：高斯 conic 矩阵 `[n_isects, 3]`
     - `means2d`：2D 投影位置 `[n_isects, 2]`
     - `opacities`：不透明度 `[n_isects]`

2. **使用 `rasterize_to_indices_in_range` 获取 αT 权重**：
   
   使用 gsplat 新增的 `rasterize_to_indices_in_range` 函数，设置 `return_weights=True` 可直接获取每个高斯点-像素对的 αT 权重：

   ```python
   from gsplat.cuda._wrapper import rasterize_to_indices_in_range
   
   # 初始化传输率：使用全一矩阵（表示初始时所有像素完全透明）
   device = meta["means2d"].device
   transmittances = torch.ones((height, width), device=device, dtype=meta["means2d"].dtype)
   
   # 调用 rasterize_to_indices_in_range 获取索引和权重
   gaussian_ids, pixel_ids, image_ids, weights = rasterize_to_indices_in_range(
       range_start=0,
       range_end=1e10,  # 处理所有高斯点
       transmittances=transmittances,  # [H, W]
       means2d=meta["means2d"],  # [n_isects, 2]
       conics=meta["conics"],  # [n_isects, 3]
       opacities=meta["opacities"],  # [n_isects]
       image_width=width,
       image_height=height,
       tile_size=16,
       isect_offsets=meta["isect_offsets"],  # [tile_h, tile_w]
       flatten_ids=meta["flatten_ids"],  # [n_isects]
       return_weights=True,  # 关键：返回权重
   )
   # 返回：
   # - gaussian_ids: [M] - 高斯点索引（原始索引，非 packed）
   # - pixel_ids: [M] - 像素索引（row-major，一维：y * width + x）
   # - image_ids: [M] - 图像索引（单视图时为 0）
   # - weights: [M] - αT 权重 w_{k,p,v} = T * alpha
   ```

   **关键点**：
   - `weights` 即为所需的 αT 权重：`w_{k,p,v} = α_{k,p,v} · T_{p,v}^{<k}`
   - 所有计算在 GPU 上完成，无需 CPU-GPU 数据传输
   - 返回的索引格式为稀疏表示，只包含权重 > threshold 的高斯点-像素对

3. **多视图处理**：
   - 对每个 source 视图重复上述步骤，得到每个视图的 `(gaussian_ids_v, pixel_ids_v, weights_v)`
   - 合并所有视图的结果，为后续特征聚合做准备


### C. Per-Gaussian 2D 特征聚合

**反投影公式**：

对于每个高斯 k，聚合所有 source 视图的特征：

```
num_k += Σ_{p,v} w_{k,p,v} · F2D_v(p)
den_k += Σ_{p,v} w_{k,p,v}
f2d_k = num_k / (den_k + ε)
```

**GPU 加速实现流程**：

1. **特征采样**（GPU 操作）：
   
   从 2D 特征图中采样每个像素的特征值，使用双线性插值：

   ```python
   def sample_features_at_pixels(
       features_2d: torch.Tensor,  # [V, H_feat, W_feat, C2] - V 个视图的特征图
       pixel_ids: torch.Tensor,  # [M] - 像素索引（row-major）
       view_ids: torch.Tensor,  # [M] - 视图索引
       height: int, width: int,  # 特征图尺寸
   ) -> torch.Tensor:
       """
       从 2D 特征图中采样像素特征（GPU 操作）。
       
       Returns:
           sampled_features: [M, C2] - 采样后的特征
       """
       device = features_2d.device
       V, H_feat, W_feat, C2 = features_2d.shape
       
       # 将 pixel_ids 转换为归一化坐标 (x, y) ∈ [0, 1]
       pixel_coords = torch.zeros(len(pixel_ids), 2, device=device)
       pixel_coords[:, 0] = (pixel_ids % width) / width  # x 归一化
       pixel_coords[:, 1] = (pixel_ids // width) / height  # y 归一化
       
       # 转换为 grid_sample 格式：坐标范围 [-1, 1]
       pixel_coords_norm = pixel_coords * 2.0 - 1.0  # [M, 2]
       
       # 为每个视图分别采样（避免循环，使用向量化操作）
       sampled_features = []
       for v in range(V):
           mask = (view_ids == v)  # [M]
           if mask.sum() == 0:
               continue
           
           # 该视图的特征图：[1, C2, H_feat, W_feat] (channels_first)
           feat_v = features_2d[v].permute(2, 0, 1).unsqueeze(0)  # [1, C2, H_feat, W_feat]
           
           # 该视图的像素坐标：[1, 1, n_mask, 2]
           coords_v = pixel_coords_norm[mask].unsqueeze(0).unsqueeze(1)
           
           # grid_sample 双线性插值采样（GPU 加速）
           sampled_v = torch.nn.functional.grid_sample(
               feat_v,
               coords_v,
               mode="bilinear",
               align_corners=True,
               padding_mode="zeros",
           )  # [1, C2, 1, n_mask]
           
           sampled_v = sampled_v.squeeze(0).squeeze(2).T  # [n_mask, C2]
           sampled_features.append((mask, sampled_v))
       
       # 合并结果
       result = torch.zeros(len(pixel_ids), C2, device=device)
       for mask, feat in sampled_features:
           result[mask] = feat
       
       return result  # [M, C2]
   ```

2. **Scatter-Add 聚合**（GPU 操作）：
   
   使用 `torch.scatter_add` 进行加权聚合，所有计算在 GPU 上完成：

   ```python
   def aggregate_features_per_gaussian(
       sampled_features: torch.Tensor,  # [M, C2] - 采样后的特征
       weights: torch.Tensor,  # [M] - αT 权重
       gaussian_ids: torch.Tensor,  # [M] - 高斯点索引（原始索引，0 到 N-1）
       num_gaussians: int,  # N - 总高斯点数量
       eps: float = 1e-8,
   ) -> torch.Tensor:
       """
       将 2D 特征按 αT 权重聚合到每个高斯点（GPU 操作）。
       
       公式：
           num_k = Σ_{p,v} w_{k,p,v} · F2D_v(p)
           den_k = Σ_{p,v} w_{k,p,v}
           f2d_k = num_k / (den_k + ε)
       
       Returns:
           aggregated_features: [N, C2] - 聚合后的特征
       """
       device = sampled_features.device
       C2 = sampled_features.shape[1]
       
       # 加权特征：w · F2D
       weighted_features = sampled_features * weights.unsqueeze(-1)  # [M, C2]
       
       # Scatter-add 聚合分子：num_k = Σ w · F2D
       num = torch.zeros(num_gaussians, C2, device=device)  # [N, C2]
       num = num.scatter_add_(
           0, 
           gaussian_ids.unsqueeze(-1).expand(-1, C2), 
           weighted_features
       )
       
       # Scatter-add 聚合分母：den_k = Σ w
       den = torch.zeros(num_gaussians, device=device)  # [N]
       den = den.scatter_add_(0, gaussian_ids, weights)
       
       # 归一化：f2d_k = num_k / (den_k + ε)
       aggregated_features = num / (den.unsqueeze(-1) + eps)  # [N, C2]
       
       return aggregated_features
   ```

3. **完整聚合流程**（GPU 操作）：
   
   将上述步骤组合，处理所有视图：

   ```python
   def backproject_features_alpha_t(
       features_2d_list: List[torch.Tensor],  # [H_feat, W_feat, C2] × V
       gaussian_ids_list: List[torch.Tensor],  # [M_v] × V
       pixel_ids_list: List[torch.Tensor],  # [M_v] × V
       weights_list: List[torch.Tensor],  # [M_v] × V
       num_gaussians: int,
       height: int, width: int,
   ) -> torch.Tensor:
       """
       完整的 2D 特征反投影流程（全部在 GPU 上完成）。
       
       Returns:
           feat_2d_aggregated: [N, C2] - 聚合后的 2D 特征
       """
       V = len(features_2d_list)
       device = features_2d_list[0].device
       
       # 合并所有视图的索引和权重
       all_gaussian_ids = torch.cat(gaussian_ids_list, dim=0)  # [M_total]
       all_pixel_ids = torch.cat(pixel_ids_list, dim=0)  # [M_total]
       all_weights = torch.cat(weights_list, dim=0)  # [M_total]
       all_view_ids = torch.cat([
           torch.full((len(ids),), v, device=device) 
           for v, ids in enumerate(gaussian_ids_list)
       ], dim=0)  # [M_total]
       
       # 合并特征图：[V, H_feat, W_feat, C2]
       features_2d_batch = torch.stack(features_2d_list, dim=0)  # [V, H_feat, W_feat, C2]
       
       # 采样特征（GPU）
       sampled_features = sample_features_at_pixels(
           features_2d_batch, all_pixel_ids, all_view_ids, height, width
       )  # [M_total, C2]
       
       # 聚合特征（GPU）
       feat_2d_aggregated = aggregate_features_per_gaussian(
           sampled_features, all_weights, all_gaussian_ids, num_gaussians
       )  # [N, C2]
       
       return feat_2d_aggregated
   ```

**关键优化点**：
- **全部 GPU 操作**：特征采样（`grid_sample`）和聚合（`scatter_add`）都在 GPU 上完成，避免 CPU-GPU 数据传输
- **向量化处理**：使用 PyTorch 的向量化操作，避免 Python 循环
- **内存效率**：使用稀疏索引格式，只处理权重 > threshold 的高斯点-像素对
- **数值稳定性**：在归一化时添加小的 epsilon 避免除零



### D. 动静态分离


**分离阶段**：
- 反投影后，按原始索引拆分：
  - `feat_2d_bg = f2d[:N_bg]`
  - `feat_2d_rigid = f2d[N_bg:N_bg+N_rigid]`
- 保持高斯 id 编号稳定，避免特征串点

---

## 特征融合与梯度流

### 特征融合

**输入**：
- `feat_3d_crop_bg`：`[N_bg, C3]` - 静态背景的 3D 特征（C3 = outdim，默认 32）
- `feat_3d_crop_rigid`：`[N_rigid, C3]` - 动态物体的 3D 特征
- `feat_2d_bg`：`[N_bg, C2]` - 静态背景的 2D 特征（C2 = 16/32）
- `feat_2d_rigid`：`[N_rigid, C2]` - 动态物体的 2D 特征
- `vis_bg`：`[N_bg]` - 静态背景的可见性
- `vis_rigid`：`[N_rigid]` - 动态物体的可见性

**融合方式**（简单拼接）：
```python
feat_bg = concat([feat_3d_crop_bg, feat_2d_bg, vis_bg.unsqueeze(-1)])  # [N_bg, C3+C2+1]
feat_rigid = concat([feat_3d_crop_rigid, feat_2d_rigid, vis_rigid.unsqueeze(-1)])  # [N_rigid, C3+C2+1]
```

**MLP 输入维度调整**：
- 现有 MLP 输入维度：`outdim`（默认 32）
- 新输入维度：`C3 + C2 + 1`（例如：32 + 16 + 1 = 49）
- 需要调整 MLP 第一层：`nn.Linear(C3 + C2 + 1, 64)`

### 梯度流设计

**关键路径**：

```
Target Loss (多个target累积)
  ↓
Proxy Params (proxies_bg, proxies_rigid)
  ↓ (autograd.backward)
Render Params (means_r, scales_r, ...)
  ↓
Offsets (offset_pos, offset_scales, ...)
  ↓
MLP Heads (mlp_offset_pos, mlp_conv, ...)
  ↑
Fused Features (feat_3d + feat_2d + vis)
  ↑                    ↑
3D Volume         2D Backprojection
  ↑                    ↑
sparse_conv       CNN (F2D[v])
```

**梯度回传机制**：

1. **CNN 前向**：只在 inner-iteration 开始时跑一次，得到 `F2D[v]`
2. **反投影聚合**：使用 `scatter_add` 聚合特征（线性算子，可微）
3. **特征融合**：拼接 2D 和 3D 特征
4. **MLP 预测**：输入融合特征，输出 offsets
5. **Proxy 累积**：多个 target 的损失梯度累积到 proxies
6. **单次回灌**：`autograd.backward(render_tensors, proxy_grads)` 将梯度回传到：
   - MLP 参数
   - 3D sparse_conv 参数
   - **2D 反投影聚合算子**（scatter_add 的梯度）
   - **CNN 参数**

**关键点**：
- αT 权重 stop-grad，不参与梯度计算
- 反投影聚合是线性算子，梯度能正常回传
- CNN 梯度来自所有 target 的累积，但 CNN 只跑一次
