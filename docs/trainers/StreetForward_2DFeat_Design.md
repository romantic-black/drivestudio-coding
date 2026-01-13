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
- **Backbone**：ResNet18（默认）或其他 ResNet 变体
  - 使用 `layer1`（`feature_resolution >= 0.25`）
  - 可选 `layer2`（`feature_resolution <= 0.125`）
  - 可选 `layer3`（`feature_resolution <= 0.0625`）
  - Fallback：如果 torchvision 不可用，使用轻量级 CNN
- **投影层**：`nn.Conv2d(backbone_out_dim, out_channels, kernel_size=1)`
- **插值**：如果 backbone 输出分辨率与目标不一致，使用双线性插值对齐

**输出**：
- `features_2d: List[Tensor]` - 每个视图的特征图 `[C2, Hf, Wf]`
  - `C2`：特征通道数（默认 16 或 32，由 `feat_2d_channels` 配置）
  - `Hf, Wf`：下采样后的特征分辨率（由 `feat_2d_resolution` 配置，默认 0.25，即原图的 1/4）
  - 计算方式：`Hf = round(H * feat_2d_resolution)`，`Wf = round(W * feat_2d_resolution)`

**关键设计**：
- CNN 在 inner-iteration 开始时只跑一次
- 特征分辨率与后续反投影渲染分辨率对齐（通过 `get_feature_resolution()` 计算）
- 支持批处理：多个图像通过 `torch.cat` 合并后一起处理，提高效率

### B. αT 权重获取

**实现**（`AlphaTWeightExtractor`）：

**渲染设置**：
- 使用**低分辨率渲染**（分辨率 = `(Hf, Wf)`）
- 自动缩放相机内参：`K'[:, 0, 0] *= scale_w`，`K'[:, 1, 1] *= scale_h`，`K'[:, 0, 2] *= scale_w`，`K'[:, 1, 2] *= scale_h`
  - `scale_h = target_height / orig_h`，`scale_w = target_width / orig_w`
- 在 rasterizer 中指定 `width=Wf, height=Hf`
- 保证像素坐标与特征图坐标一致

**αT 权重计算流程**：
1. 对每个视图进行低分辨率渲染（`render_mode="RGB"`，`sparse_grad=False`）
2. 从渲染元数据（`meta`）中提取：
   - `flatten_ids`：高斯 id 列表
   - `isect_offsets`：每个 tile 的起始/结束索引
   - `conics`：高斯 conic 矩阵
   - `means2d`：2D 投影位置
   - `opacities`：不透明度
3. 对每个 tile 内的像素：
   - 计算每个高斯的贡献：`weight = exp(-0.5 * sigma_term) * opacity * T`
   - 使用 `torch.topk` 选择 top-K 贡献最大的高斯
   - 存储到 `idx_map[Hf, Wf, K]` 和 `w_map[Hf, Wf, K]`

**输出接口**：
- `gaussian_indices: List[Tensor]` - 每个视图的高斯索引 `[Hf, Wf, K]`（无效填 -1）
- `alpha_t_weights: List[Tensor]` - 每个视图的 αT 权重 `[Hf, Wf, K]`（已 detach）

**关键约束**：
- αT 权重必须与 RGB 渲染使用相同的排序/截断/early-stop 规则
- **权重 stop-grad**：`w = w.detach()`，避免高阶梯度耦合
- 渲染在 `torch.no_grad()` 上下文中进行，只提取权重，不参与梯度计算

### C. Per-Gaussian 2D 特征聚合

**反投影公式**：

对于每个高斯 k，聚合所有 source 视图的特征：

```
num_k += Σ_{p,v} w_{k,p,v} · F2D_v(p)
den_k += Σ_{p,v} w_{k,p,v}
f2d_k = num_k / (den_k + ε)
```

**实现方式**（index_add）：
1. 将特征图展平：`Fpix[v] = permute(F2D[v], (1, 2, 0)).reshape(-1, C2) ∈ R^{P×C2}`，其中 `P = Hf×Wf`
   - 注意：特征图格式为 `[C2, Hf, Wf]`，需要先 permute 再 reshape
2. 对每个视图 v：
   - `idx_flat ∈ Z^{P×K}`：像素到高斯的映射（从 `[Hf, Wf, K]` reshape 得到）
   - `w_flat ∈ R^{P×K}`：对应的权重（从 `[Hf, Wf, K]` reshape 得到）
   - 对每个 top-K 位置 k：
     - 提取有效项：`valid = (idx_flat[:, k] >= 0)`
     - 使用 `torch.index_add` 累加：`num.index_add(0, idx_valid, w_valid * Fpix_valid)`，`den.index_add(0, idx_valid, w_valid)`
3. 跨视图融合：所有视图的 `num` 和 `den` 累加（在循环中自动完成）
4. 归一化：`f2d = num / (den.unsqueeze(-1) + ε)`

**边界处理**：
- `idx == -1` 的无效项需要 mask 掉
- `den_k < eps` 的高斯：
  - 设置 `f2d_k = 0`
  - 输出可见性标记 `vis_k = 0`（或 `clamp(den_k, 0, 1)`）

**可见性输出**：
- `vis_k`：表示高斯被看到的程度，可作为额外特征输入 MLP
- 建议使用 `vis_k = clamp(den_k, 0, 1)` 或 `log(den_k + 1)`

### D. 动静态分离

**合并阶段**：
- 在 source 帧下，将 rigid 变换到世界坐标
- 合并 bg 和 rigid 的高斯参数（用于反投影渲染）

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

---

## 与现有机制的集成

### 在 train_iter 中的插入位置

**推荐顺序**（最小改动）：

1. **获取/初始化 NodeState**（已有）
2. **设置 source_frame_idx**（已有）
3. **变换 rigid 到 source 帧世界坐标**（已有）
4. **合并 bg + rigid 高斯**（已有）
5. **🆕 提取 2D 特征**：
   - 使用 `_prepare_source_views_for_2d_features()` 准备 source 视图和图像
   - 对 source 时间戳的所有相机图像跑 CNN（`image_feature_extractor`）
   - 得到 `features_2d: List[Tensor]`，每个元素为 `[C2, Hf, Wf]` 格式
6. **🆕 αT 反投影**：
   - 使用合并后的高斯在 source views 上低分辨率渲染
   - 获取 `idx[v]` 和 `w[v]`
   - scatter-add 聚合得到 `f2d`（per-Gaussian）
   - 拆分为 `feat_2d_bg` 和 `feat_2d_rigid`
7. **构建 3D volume**（已有）
8. **插值得到 feat_3d**（已有）
9. **🆕 特征融合**：
   - `feat_bg = concat([feat_3d_crop_bg, feat_2d_bg, vis_bg])`
   - `feat_rigid = concat([feat_3d_crop_rigid, feat_2d_rigid, vis_rigid])`
10. **预测 offsets**（已有，但 MLP 输入维度需调整）
11. **后续流程**（proxy + 多 target backward + 回灌，已有）

### 与 Proxy 机制的兼容性

**完全兼容**：
- CNN 和反投影在 proxy 创建之前完成
- 2D 特征参与 offsets 预测，offsets 参与 render_params 计算
- render_params 创建 proxies，梯度链路完整
- 多 target 的梯度累积到 proxies，单次回灌将梯度传回 CNN

**无需修改**：
- Proxy 创建机制
- 多 target 梯度累积
- 单次回灌机制

---

## 关键实现细节

### 1. 分辨率对齐

**问题**：反投影渲染分辨率必须与 CNN 特征图分辨率一致

**解决方案**：
- CNN 输出特征图分辨率：`(Hf, Wf)`
- 反投影渲染分辨率：`(Hf, Wf)`
- 使用缩放后的相机内参：`K' = K * scale`，其中 `scale = (Hf/H, Wf/W)`
- 在 rasterizer 中指定 `image_size=(Hf, Wf)`

### 2. Index-Add 实现

**实际实现**（`Feature2DBackprojector`）：
- 使用 `torch.index_add`（非原地操作，支持梯度回传）
- `num` 和 `den` 累加使用 `fp32`（`accum_dtype`），避免精度损失
- 最终输出转换为输入 dtype（保持与输入特征图一致）
- 处理 `idx == -1` 的无效项（通过 `valid = idx_k >= 0` mask 掉）
- 处理 `den ≈ 0` 的情况（通过 `eps` 避免除零，输出 `feat_2d = 0` 和 `vis = 0`）

**梯度兼容性**：
- `torch.index_add` 是可微操作，梯度能正常回传到 `feat_valid`
- 累加器 `num` 和 `den` 虽然是新创建的，但通过 `index_add` 与输入特征建立了计算图连接
- 最终特征 `feat_all = num / (den + eps)` 的梯度能回传到 CNN

**实际实现结构**（`Feature2DBackprojector`）：
```python
# 初始化累加器（使用 fp32 精度避免累加误差）
device = features_2d[0].device
dtype = features_2d[0].dtype
accum_dtype = torch.float32 if dtype.is_floating_point else dtype
num = torch.zeros(N_total, C2, device=device, dtype=accum_dtype)
den = torch.zeros(N_total, device=device, dtype=accum_dtype)

# 对每个视图
for feat_map, idx_map, w_map in zip(features_2d, gaussian_indices, alpha_t_weights):
    # 展平特征图：从 [C2, Hf, Wf] 转换为 [P, C2]
    feat_flat = feat_map.permute(1, 2, 0).reshape(-1, C2).to(accum_dtype)  # [P, C2]
    idx_flat = idx_map.reshape(-1, K).long()  # [P, K]
    w_flat = w_map.reshape(-1, K).to(accum_dtype)  # [P, K]
    
    # 对每个 top-K 位置
    for k in range(K):
        idx_k = idx_flat[:, k]  # [P]
        valid = idx_k >= 0  # 过滤无效项（idx == -1）
        if not valid.any():
            continue
        
        idx_valid = idx_k[valid]  # [P_valid]
        w_valid = w_flat[valid, k]  # [P_valid]
        feat_valid = feat_flat[valid]  # [P_valid, C2]
        
        # 加权特征
        weighted = feat_valid * w_valid.unsqueeze(-1)  # [P_valid, C2]
        
        # 使用 index_add 累加（可微操作）
        num = torch.index_add(num, 0, idx_valid, weighted)
        den = torch.index_add(den, 0, idx_valid, w_valid)

# 归一化并转换回原始 dtype
eps = 1e-8
feat_all = (num / (den.unsqueeze(-1) + eps)).to(dtype)  # [N_total, C2]
vis_all = torch.clamp(den, 0.0, 1.0).to(dtype)  # [N_total]

# 按索引拆分静态和动态
feat_2d_bg = feat_all[bg_indices]  # [N_bg, C2]
feat_2d_rigid = feat_all[rigid_indices]  # [N_rigid, C2]
vis_bg = vis_all[bg_indices]  # [N_bg]
vis_rigid = vis_all[rigid_indices]  # [N_rigid]
```

**关键实现细节**：
- 使用 `torch.index_add` 而不是 `scatter_add_`（原地操作），确保梯度能正常回传
- 累加器使用 `fp32` 精度（`accum_dtype`），最终输出转换为输入 dtype
- 特征图从 `[C2, Hf, Wf]` 格式通过 `permute(1, 2, 0)` 转换为 `[Hf, Wf, C2]`，再 reshape 为 `[P, C2]`
- 对每个 top-K 位置分别处理，避免无效索引影响累加

### 3. Top-K 选择

**建议值**：
- 初始实现：`K = 8` 或 `16`
- 语义反投影对长尾贡献不敏感，K 不需要太大
- 可根据显存和速度调整

### 4. MLP 输入维度调整

**现有结构**：
```python
mlp_offset_pos = nn.Sequential(
    nn.Linear(outdim, 64),  # outdim = 32
    ...
)
```

**新结构**：
```python
feat_dim = outdim + C2 + 1  # 例如：32 + 16 + 1 = 49
mlp_offset_pos = nn.Sequential(
    nn.Linear(feat_dim, 64),  # 输入维度扩展
    ...
)
```

**其他 MLP**（`mlp_conv`, `mlp_opacity`, `gaussion_decoder`）同样需要调整第一层输入维度。

### 5. 高斯 ID 稳定性

**问题**：bg 和 rigid 合并后，需要保证拆分时索引对应正确

**解决方案**：
- 合并时记录索引映射：`bg_indices = range(N_bg)`，`rigid_indices = range(N_bg, N_bg+N_rigid)`
- 反投影时使用合并后的高斯 id
- 拆分时按原始索引：`feat_2d_bg = f2d[bg_indices]`，`feat_2d_rigid = f2d[rigid_indices]`

### 6. Source 视图准备

**实现**（`_prepare_source_views_for_2d_features`）：

**功能**：
- 从 batch 中提取 source 帧的多相机视图和图像
- 支持多种 batch 格式（`source_views`、`source_images`、`targets` 等）

**处理流程**：
1. 检查 batch 中是否包含 source 相关信息
2. 提取 source 帧的所有相机视图（同一时间戳）
3. 提取对应的图像数据
4. 返回 `source_views: List[View]` 和 `source_images: List[Tensor]`

**边界情况**：
- 如果没有 source 视图/图像，返回空列表
- 训练器会检测到空列表，使用零特征作为 fallback

### 7. 显存优化

**潜在瓶颈**：
- CNN 特征图：`V × C2 × Hf × Wf`（V 个视图）
- Top-K 索引和权重：`V × Hf × Wf × K`
- Index-add 累加器：`N_total × C2`（fp32 精度）

**优化建议**：
- 使用半精度（fp16）存储 CNN 特征（前向用 fp32，存储用 fp16）
- 分批处理多个视图（如果 V 很大）
- 及时释放中间变量（CNN 特征图、索引、权重）
- 累加器使用 fp32 但只在累加阶段，最终输出转换回输入 dtype

---

## 总结

### 核心优势

1. **时间一致性**：只对 source 帧提取 2D 特征，避免动态一致性问题
2. **梯度兼容**：CNN 梯度通过现有 proxy 机制回传，不破坏训练范式
3. **显存高效**：CNN 只跑一次，不随 target 数量增加显存
4. **实现简单**：特征融合使用简单拼接，易于实现和调试

### 关键约束

1. **αT 权重 stop-grad**：避免高阶梯度耦合
2. **分辨率对齐**：反投影渲染分辨率必须与特征图一致
3. **可见性对齐**：αT 权重必须与 RGB 渲染使用相同规则
4. **索引稳定性**：合并/拆分时保证高斯 id 对应正确

### 后续优化方向

1. **特征融合方式**：从简单拼接升级到 cross-attention（如果简单拼接效果不够）
2. **多尺度特征**：使用 CNN 的多层特征（FPN 风格）
3. **时序一致性**：如果 source 有多个时间戳，考虑时序融合
4. **自适应权重**：αT 权重可以学习（但需要谨慎处理梯度）

---

## 参考文献

- StreetForward 原始设计：`docs/FeedForward_3DGS_Design.md`
- StreetForward 流程文档：`docs/trainers/StreetForward_Flow.md` - 包含完整的训练流程和 2D 特征融合步骤
- 实现代码：
  - `models/trainers/streetforward.py` - 主训练器
  - `models/feature_extractors/image_feature_extractor.py` - 2D 特征提取器
  - `models/feature_extractors/alpha_t_extractor.py` - αT 权重提取器
  - `models/feature_extractors/feature_2d_backprojector.py` - 2D 特征反投影器
  - `models/feature_extractors/feature_fusion.py` - 特征融合模块
- 配置文件：`configs/streetforward/multi_scene.yaml`
- 问题分析：`docs/trainers/StreetForward_2DFeat_Issues.md` - 实现过程中的关键问题和解决方案
