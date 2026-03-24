# StreetForward Trainer 流程图与数据结构说明

本文档详细梳理了 `StreetForwardTrainer` 的训练流程、数据结构和关键组件。

> **模块结构（Phase 2 重构后）**：训练逻辑分布在 `models/streetforward/` 下：
> - `trainer.py`：主编排、`train_iter`、`_train_inner_iteration`、`_parse_targets`、`_get_source_frame_idx`
> - `node_state_mixin.py`：`_get_or_init_node_states`、`_precompute_rigid_masks`、`_update_node_states`、Rigid 变换
> - `feature_volume_mixin.py`：`_build_3d_feature_volume`、`_compute_and_fuse_features`
> - `offsets_mixin.py`：`_predict_offsets`、`_predict_offsets_gru`、`_compute_render_params_for_inner_iter`、`_render_params_from_offsets`
> - `proxy_rendering_mixin.py`：`_create_proxy_params`、`_render_targets_and_accumulate_loss`、`_backward_to_render_params`、`_merge_params_with_rigid_subset`
> - `metrics.py`：`compute_psnr`、`compute_ssim`、`compute_lpips`、`evaluate_test_views`
>
> **Minimal Stage 3.3（旁路实验管线）**：在 `MinimalStreetForwardStage3_2` 之上实现 bg / distant 配置与网络解耦，入口为 `tools/train_minimal_streetforward_stage3_3.py`，实现见 `models/streetforward/minimal_trainer_stage3_3.py`；设计背景见 [StreetForward_Stage3_3_Design.md](StreetForward_Stage3_3_Design.md)。包导出在 `models/streetforward/__init__.py` 中通过 lazy import 暴露 `MinimalStreetForwardStage3_3`。

## 目录
1. [整体架构](#整体架构)
2. [训练流程](#训练流程)
3. [3D特征体积构建详细流程](#3d特征体积构建详细流程) ⭐
4. [数据结构详解](#数据结构详解)
5. [关键组件说明](#关键组件说明)
6. [梯度反向传播机制](#梯度反向传播机制)
7. [天空分支（Stage 3.1）](#天空分支stage-31)
8. [Minimal Stage 3.3（bg/distant 解耦）](#minimal-stage-3-3-bg-distant)

---

## 整体架构

StreetForwardTrainer 实现了基于代理（Proxy）的多视角梯度累积的前馈式 3D Gaussian Splatting 训练器，支持静态背景、动态物体和背景远景的联合训练。

### 核心设计理念

- **多 NodeState 架构**：每个 `(scene_id, segment_id)` 维护多个 `NodeState`：
  - `NodeStateBackground`：存储静态背景的高斯参数（世界坐标系）
  - `NodeStateRigid`：存储动态物体的高斯参数（局部坐标系）
  - `NodeStateDistant`：存储背景远景的高斯参数（世界坐标系，可选）
- **前馈预测**：通过 3D 特征体积（可选融合 2D 特征）预测偏移量（offsets），静态和动态物体共享相同的 MLP 网络
- **GRU-style Hidden Fusion（当前实现）**：训练时将「点的融合特征」与「NodeState 参数 embedding」通过 GRU 风格更新得到 `h_new`，再用 `h_new` 作为偏移头输入预测 offsets；并在 `train_iter` 级别缓存 `h_cache_{bg,rigid,distant}`，实现跨 iter 的状态记忆（每次 `train_iter` 起始处对 `h_old` 做 `detach()` 截断跨 iter 梯度）
- **2D/3D 特征融合**：可选地从源视图提取 2D 图像特征，与 3D 特征融合以增强表示能力
- **代理参数渲染**：使用代理参数进行渲染，实现多视角梯度累积
- **单次反向传播**：每个迭代只进行一次反向传播
- **帧变换机制**：动态物体在不同帧间通过 RigidNodes 变换，支持时间一致性
- **可见性掩码**：动态物体使用可见性掩码屏蔽不可见点的偏移量

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│        StreetForwardTrainer (with Dynamic Objects + 2D Feat) │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐│
│  │ NodeStateBg      │  │ NodeStateRigid   │  │NodeStateDist││
│  │ (Detached)       │  │ (Detached)       │  │ (Detached)  ││
│  │ - means          │  │ - means (local)  │  │ - means     ││
│  │ - scales_log     │  │ - scales_log     │  │ - scales_log││
│  │ - quats          │  │ - quats          │  │ - quats     ││
│  │ - opacity_logit  │  │ - opacity_logit  │  │ - opacity   ││
│  │ - sh_dc          │  │ - sh_dc          │  │ - sh_dc     ││
│  │ - sh_rest        │  │ - sh_rest        │  │ - sh_rest   ││
│  └──────────────────┘  │ - point_ids      │  └─────────────┘│
│           │            │ - instances_*    │                 │
│           │            └──────────────────┘                 │
│           │                    │                             │
│           └──────────┬──────────┘                             │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │  train_iter()        │                           │
│           │  (1 source + N targets)                           │
│           └──────────┬───────────┘                           │
│                      │                                       │
│    ┌─────────────────┼─────────────────┐                    │
│    │                 │                 │                     │
│    ▼                 ▼                 ▼                     │
│ ┌─────────┐   ┌──────────┐   ┌──────────┐                  │
│ │Transform│   │ 3D Vol   │   │ 2D Feat  │                  │
│ │to Source│──▶│ Builder   │   │ Extract  │                  │
│ └─────────┘   └─────┬────┘   └─────┬────┘                  │
│                     │              │                          │
│                     └──────┬───────┘                         │
│                            ▼                                 │
│                     ┌──────────────┐                         │
│                     │ Feature Fusion│                        │
│                     │ (3D + 2D)    │                         │
│                     └──────┬───────┘                         │
│                            ▼                                 │
│                     ┌──────────────┐                         │
│                     │ Offsets Predict│                        │
│                     └──────┬───────┘                         │
│                            │                                 │
│  ┌──────────────────────────────────────────────┐          │
│  │  For each target frame:                      │          │
│  │  1. Transform RigidNodes to target frame      │          │
│  │  2. Merge Bg + Rigid + Distant params       │          │
│  │  3. Create proxy params                      │          │
│  │  4. Render & accumulate gradients             │          │
│  └──────────────────────────────────────────────┘          │
│                                                               │
│  ┌──────────────────────────────────────────────┐          │
│  │  Neural Networks                              │          │
│  │  - sparse_conv: 3D特征提取                    │          │
│  │  - image_feature_extractor: 2D特征提取       │          │
│  │  - feature_fusion: 2D/3D特征融合              │          │
│  │  - mlp_offset_pos: 位置偏移预测               │          │
│  │  - mlp_conv: 尺度与旋转偏移预测               │          │
│  │  - mlp_opacity: 不透明度偏移预测              │          │
│  │  - gaussion_decoder: SH系数偏移预测           │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## 训练流程

### 主训练循环流程图

```mermaid
graph TD
    A[开始: train_iter] --> B[_get_or_init_node_states]
    B --> C[_parse_targets]
    C --> D{是否有targets?}
    D -->|否| E[返回零损失]
    D -->|是| F[清零梯度]
    F --> F1[从 h_cache 取 h_old 并 detach]
    F1 --> G[开始inner_iterations循环]
    G --> H[_train_inner_iteration]
    H --> H1[预计算RigidMasks]
    H1 --> H2[构建3D特征+2D融合]
    H2 --> H3[GRU-style 预测偏移量+gate(rigid)]
    H3 --> H4[计算渲染参数]
    H4 --> H5[创建代理参数]
    H5 --> H6[多target渲染+损失]
    H6 --> H7[反向传播到渲染参数]
    H7 --> I{是否apply_update?}
    I -->|是| J[优化器更新]
    I -->|否| K[跳过更新]
    J --> L{是否update_state?}
    K --> L
    L -->|是| M[_update_node_states]
    L -->|否| N[保持原状态]
    M --> O{是否还有inner_iter?}
    N --> O
    O -->|是| G
    O -->|否| P[保存NodeState + 写回 h_cache 并返回]
```

**_train_inner_iteration 内部步骤（对应方法）：**
1. `_get_source_frame_idx` → `_precompute_rigid_masks`（返回 `RigidMasks`）
2. `_build_3d_feature_volume`（传入 `mask_src_rigid`, `idx_src_rigid`）
3. `_compute_and_fuse_features`（2D 特征 + 融合）
4. `_build_params_for_embed` + `_predict_offsets_gru`（GRU-style 融合 + `mask_update_rigid` gate，仅 rigid 需要）
5. `_compute_render_params_for_inner_iter`
6. `_create_proxy_params`
7. `_render_targets_and_accumulate_loss`（含 `_merge_params_with_rigid_subset` 支持 rigid 子集）
8. `_backward_to_render_params`

### 详细步骤说明

#### 1. 初始化阶段 (`_get_or_init_node_states`)

**输入数据：**
- `batch["scene_id"]`: 场景ID
- `batch["segment_id"]`: 片段ID
- `batch["pointcloud"]`: 点云数据（字典格式，包含 `background` 和可选的 `dynamic`）
- `batch["dynamic_info"]`: 动态物体信息（可选，如果点云包含动态物体）

**处理流程：**
```
点云数据 → 先按 input_aabb 过滤 → 再按 bbx(crop) 切分前景/远景 →
  前景(在bbx内): 提取坐标和颜色 → 计算初始尺度 → 生成随机四元数 → 创建NodeStateBackground
  动态: 提取坐标和颜色 → 计算初始尺度 → 生成随机四元数 → 创建NodeStateRigid（局部坐标）
  远景(在bbx外但仍在input_aabb内): 提取坐标和颜色 → 计算初始尺度 → 生成随机四元数 → 创建NodeStateDistant（世界坐标，可选）
```

**关键操作：**
- **静态背景**：使用 k-NN 计算邻居距离，初始化尺度；将 RGB 颜色转换为球谐函数（SH）的 DC 分量；所有参数初始化为分离（detached）状态
- **动态物体**：从 `pointcloud["dynamic"]` 获取各实例的点云（局部坐标）；从 `dynamic_info` 初始化 `instances_quats` 和 `instances_trans`；记录 `point_ids` 以关联每个点到实例；初始化 `instances_fv` 记录可见性
- **背景远景**：从 `pointcloud["distant"]` 获取远景点云（世界坐标，可选）；使用与静态背景相同的初始化方法

#### 2. 预计算 Rigid 掩码 (`_precompute_rigid_masks`)

**归属：** NodeStateMixin

**输入：** `node_state_rigid`, `source_frame_idx`, `targets`

**返回：** `RigidMasks` 数据类，包含：
- `mask_src_rigid`: source 帧有有效位姿且可见的点
- `mask_tgt_rigid`: 各 target 帧的可见点掩码列表
- `mask_update_rigid`: `mask_src_rigid & mask_any_tgt_rigid`，用于 gate 偏移量
- `idx_tgt_rigid`: 各 target 帧可见点的索引列表，用于渲染时只合并可见 rigid 子集
- `idx_src_rigid`: source 帧可见点索引，供 `_build_3d_feature_volume` 使用

**说明：** 调用方必须在 `_build_3d_feature_volume` 前调用此方法，并传入 `mask_src_rigid` 与 `idx_src_rigid`。

#### 3. 3D 特征体积构建 (`_build_3d_feature_volume`)

**归属：** FeatureVolumeMixin

**步骤：**
```
1. 接收 mask_src_rigid, idx_src_rigid（来自 _precompute_rigid_masks）
2. 设置 RigidNodes.cur_frame = source_frame_idx
3. 获取静态背景点云（世界坐标）
   NodeStateBg.means [N_bg, 3]
4. 变换动态物体到 source 帧的世界坐标
   NodeStateRigid.means [N_rigid, 3] (局部) 
     → _transform_rigid_to_world() 
     → means_rigid_world [N_rigid, 3] (世界坐标)
5. 计算动态物体可见性掩码
   rigid_visible_mask [N_rigid] - 基于 instances_fv
   rigid_in_crop_mask [N_rigid] - 基于边界框检查
6. 合并静态和动态点云（使用 idx_src_rigid 筛选可见且在 crop 内的动态点）
   means_all = cat([means_bg, means_rigid_world[effective_mask]]) [N_total, 3]
   anchor_rgb_all = cat([anchor_rgb_bg, anchor_rgb_rigid[effective_mask]]) [N_total, 3]
7. 构建统一的 3D 特征体积
   construct_sparse_tensor() 
     → sparse_feat [M, 3] (RGB特征，M ≤ N_total)
     → vol_dim [3] (体积维度)
     → valid_coords [M, 3] (有效坐标)
   sparse_conv() 
     → feat_3d [M, outdim] (3D特征)
   sparse_to_dense_volume() 
     → dense_volume [1, C, D, H, W]
8. 分别为静态和动态点插值特征
   feat_3d_crop_bg [N_bg, outdim]
   feat_3d_crop_rigid [N_rigid, outdim] (不可见点特征为零)
9. 删除密集体积以释放内存
   del dense_volume
```

**数据维度说明：**
- `sparse_feat`: `[M, 3]` - M个唯一体素的RGB特征（M ≤ N_total，因为可能有重复体素）
- `feat_3d`: `[M, outdim]` - 经过稀疏卷积后的3D特征（默认outdim=32）
- `dense_volume`: `[1, C, D, H, W]` - 密集化的3D特征体积（使用后立即删除）
- `feat_3d_crop_bg`: `[N_bg, outdim]` - 静态背景点的3D特征
- `feat_3d_crop_rigid`: `[N_rigid, outdim]` - 动态物体点的3D特征（不可见点为零）
- `rigid_visible_mask`: `[N_rigid]` - 动态物体可见性掩码（bool）
- `rigid_in_crop_mask`: `[N_rigid]` - 动态物体是否在crop内的掩码（bool）

#### 3.1. 2D 特征计算与融合 (`_compute_and_fuse_features`) - 可选

**归属：** FeatureVolumeMixin

**说明：** 封装 2D 特征计算（`_compute_2d_features_all`）与融合（`_fuse_features`），返回 `feat_bg_input`, `feat_rigid_input`, `feat_distant_input` 及原始 2D 特征。

#### 3.2. 2D 特征计算 (`_compute_2d_features_all`) - 可选

**条件：** `use_2d_features == True` 且提供了 `source_views` 和 `src_images`

**步骤：**
```
1. 准备高斯参数（合并静态和动态，变换到source帧）
   _prepare_gaussians_for_source()
     → gaussians (合并后的高斯参数)
     → num_bg, num_rigid (数量统计)
2. 第一阶段：渲染RGB图像（供CNN使用）
   alpha_t_extractor.render_rgb_only()
     → rendered_rgbs [V, H, W, 3]
3. 构建多通道输入
   multi_channel_input = cat([image_batch, rendered_rgbs], dim=-1)
     → [V, H, W, 6] (RGB + 渲染RGB)
4. CNN特征提取
   image_feature_extractor(multi_channel_input)
     → features_2d [V, H_feat, W_feat, C]
5. 第二阶段：流式渲染并反投影
   alpha_t_extractor.render_and_backproject_streaming()
     → feat_2d_all [N_total, C_2d]
6. 分离静态和动态特征
   feat_2d_bg = feat_2d_all[:num_bg]
   feat_2d_rigid = feat_2d_all[num_bg:]
   feat_2d_rigid *= rigid_visible_mask (应用可见性掩码)
```

**数据维度说明：**
- `features_2d`: `[V, H_feat, W_feat, C]` - 2D图像特征（V个视图）
- `feat_2d_bg`: `[N_bg, C_2d]` - 静态背景点的2D特征
- `feat_2d_rigid`: `[N_rigid, C_2d]` - 动态物体点的2D特征（不可见点为零）
- `feat_2d_distant`: `[N_distant, C_2d]` - 背景远景点的2D特征（如果存在）

#### 3.3. 特征融合 (`_fuse_features`) - 可选

**条件：** `use_2d_features == True` 且 `feature_fusion` 已初始化

**步骤：**
```
对于每个点类型（bg, rigid, distant）：
  feat_input = feature_fusion.fuse(feat_3d, feat_2d, visibility)
    → [N, C_fused] (融合后的特征)
```

**融合策略：**
- 使用 `FeatureFusion` 模块（通常是门控融合或MLP融合）
- `visibility` 用于加权2D特征的贡献
- 如果 `feat_2d` 为 `None`，直接返回 `feat_3d`

#### 4. 特征插值

**步骤：**
```
NodeState.means 
  ↓
get_grid_coords() 
  → grid_coords [N, 3] (归一化网格坐标，范围[-1, 1])
  ↓
interpolate_features() 
  → feat_3d_crop [N, outdim] (每个点对应的3D特征)
```

**关键函数：**
- `get_grid_coords()`: 将世界坐标转换为体积网格的归一化坐标
- `interpolate_features()`: 使用双线性插值从密集体积中提取每个点的特征

#### 5. 偏移量预测与 Gate (`_predict_and_gate_offsets`)

**归属：** OffsetsMixin

**说明（当前实现优先）**：训练主路径已切换到 **GRU-style** 的 `_predict_offsets_gru()`（特征 + 参数 embedding + GRU 更新 → offsets）。旧的 `_predict_and_gate_offsets()` 仍保留（封装 `_predict_offsets` 并对 rigid 做 gate），但不再是 `StreetForwardTrainer._train_inner_iteration` 的主调用路径。

#### 5.1. 偏移量预测 (`_predict_offsets`)

**输入：**
- `feat_bg_input`: `[N_bg, C_input]` - 静态背景点的融合特征（如果启用2D特征则为融合特征，否则为3D特征）
- `feat_rigid_input`: `[N_rigid, C_input]` - 动态物体点的融合特征
- `feat_distant_input`: `[N_distant, C_input]` - 背景远景点的融合特征（如果存在）

**处理：**
- 静态、动态和远景使用**相同的 MLP 网络**预测偏移量
- 在 source 帧下，静态和动态都是确定的，偏移量是共同预测的，不区别对待
- 如果启用了2D特征，输入特征维度可能大于3D特征维度（`C_input > outdim`）

#### 5.2. 动态物体偏移量 Gate（`_predict_and_gate_offsets` 内）

**目的：** 使用 `mask_update_rigid` 屏蔽无监督 rigid 点的偏移量

**处理：**
```
gate = mask_update_rigid.unsqueeze(-1).detach()  # [Nr, 1]
offsets_rigid_world["offset_pos"] *= gate
offsets_rigid_world["offset_scales"] *= gate
offsets_rigid_world["offset_quat"] *= gate
offsets_rigid_world["offset_opacity"] *= gate
offsets_rigid_world["offset_sh"] *= gate
```

**关键点：**
- `mask_update_rigid = mask_src_rigid & mask_any_tgt_rigid`，仅在 source 和至少一个 target 可见时更新
- 无监督点的偏移量被 gate 成 0，确保这些点不会被更新
- **注意**：GRU-style 主路径中，`offset_quat` 对无监督点会被置为 **identity quaternion**（不是乘 0）

**输出：**
```python
{
    "offset_pos": [N, 3],        # 位置偏移（经过tanh限制）
    "offset_scales": [N, 3],     # 尺度对数偏移（经过tanh限制）
    "offset_quat": [N, 4],       # 四元数偏移（wxyz格式，从轴角转换）
    "offset_opacity": [N, 1],    # 不透明度对数偏移（经过tanh限制）
    "offset_sh": [N, 3*num_sh],  # SH系数偏移（包含DC和rest，分别限制）
}
```

**MLP 网络结构：**
- `mlp_offset_pos`: `outdim → 64 → 32 → 3` (位置偏移)
- `mlp_conv`: `outdim → 64 → 32 → 6` (3个尺度对数偏移 + 3个轴角偏移)
- `mlp_opacity`: `outdim → 64 → 32 → 1` (不透明度对数偏移)
- `gaussion_decoder`: `outdim → 64 → 32 → 3*num_sh` (SH系数偏移，包含DC和rest)

#### 5.1.1. GRU-style 偏移量预测（训练主路径：`_predict_offsets_gru`）⭐

**目的：** 将「融合特征」与「当前 NodeState 参数」融合后再预测 offsets，并维护每类点（bg/rigid/distant）的隐藏状态缓存 \(h\)。

**关键输入：**
- `feat`: `[N, C_fused]`（3D 或 3D+2D 融合特征）
- `params_for_embed`: 来自 `_build_params_for_embed()`，包含 `means/quats/scales_log/opacity_logit/sh_dc/sh_rest`
  - 对 rigid：会先将 `means/quats` 变换到 source 帧世界坐标对齐（便于跨类型统一归一化）
- `h_old`: `[N, H]`（来自 `h_cache_*`，在 `train_iter` 起始处 `detach()` 截断跨 iter 梯度）
- `mask_update_rigid`（仅 rigid 传入）：`[N_rigid]` bool

**参数 embedding（固定 17 维 param_vec）**：
- `means_norm(3)`: 以 `bbx_min/bbx_max` 归一化到 `[-1, 1]`
- `rot6d(6)`: quaternion → rot6d（旋转矩阵前两列）
- `scales_norm(3)`: `scales_log` clamp 后做 layer_norm
- `opacity_norm(1)`: `tanh(opacity_logit)`
- `sh_dc(3)`: 原值
- `sh_rest_energy(1)`: `||sh_rest||_2`（高阶能量标量）

**GRU-style 更新与 offsets：**
1. `param_embed = LayerNorm(MLP(param_vec))`
2. `x = concat([feat, param_embed])`
3. 使用 update/reset/candidate 线性层得到 `h_new`
4. `head_input = gru_to_head(h_new)`（若 hidden_dim != head_dim 会线性投影）
5. `offsets = _predict_offsets(head_input)`（复用原偏移头）

**Rigid gate（训练关键约束）**：
- gate 的目标：仅允许 source & 任意 target 可见的 rigid 点更新  
  `mask_update_rigid = mask_src_rigid & mask_any_tgt_rigid`
- 对 offsets：
  - `offset_pos/offset_scales/offset_opacity/offset_sh`：乘以 `gate`（并 `detach()`）
  - `offset_quat`：对 gate==False 的点使用 **identity quaternion** `[1,0,0,0]`（`torch.where`），而不是数值乘零
- 对隐藏状态：`h_new = h_old*(1-gate) + h_new*gate`，保证无监督点 hidden 不漂移

**空特征边界情况：**
- 当 `feat` 为空时返回“零更新” offsets（`offset_quat` 返回 identity），并保持 `h_new = h_old`（rigid 仍会按 gate 保持一致性）。

**偏移量预测流程：**

1. **位置偏移** (`offset_pos`)：
   ```python
   offset_pos_raw = mlp_offset_pos(feat_3d_crop)  # [N, 3]
   offset_pos = offset_max * tanh(offset_pos_raw)  # 限制在 [-offset_max, offset_max]
   ```

2. **尺度与旋转偏移** (`offset_scales`, `offset_quat`)：
   ```python
   scales_and_omega = mlp_conv(feat_3d_crop)  # [N, 6]
   offset_scales_raw, offset_omega_raw = split([3, 3])  # 分别提取尺度和轴角
   offset_scales = scale_max * tanh(offset_scales_raw)  # 限制在 [-scale_max, scale_max]
   offset_omega = omega_max * tanh(offset_omega_raw)    # 限制在 [-omega_max, omega_max]
   offset_quat = _axis_angle_to_quat(offset_omega)      # 转换为四元数
   ```

3. **不透明度偏移** (`offset_opacity`)：
   ```python
   offset_opacity_raw = mlp_opacity(feat_3d_crop)  # [N, 1]
   offset_opacity = opacity_max * tanh(offset_opacity_raw)  # 限制在 [-opacity_max, opacity_max]
   ```

4. **SH系数偏移** (`offset_sh`)：
   ```python
   sh_raw = gaussion_decoder(feat_3d_crop)  # [N, 3*num_sh]
   sh_dc_raw = sh_raw[:, :3]                # DC分量
   sh_rest_raw = sh_raw[:, 3:]              # rest分量
   offset_sh_dc = sh_dc_max * tanh(sh_dc_raw)      # 限制在 [-sh_dc_max, sh_dc_max]
   offset_sh_rest = sh_rest_max * tanh(sh_rest_raw)  # 限制在 [-sh_rest_max, sh_rest_max]
   offset_sh = concat([offset_sh_dc, offset_sh_rest], dim=-1)  # [N, 3*num_sh]
   ```

**约束参数（默认值）：**
- `offset_max`: 0.1 (米) - 位置偏移上限
- `scale_max`: 0.1 (对数域) - 尺度偏移上限
- `omega_max`: 0.1 (弧度，约5.7°) - 旋转偏移上限
- `opacity_max`: 0.1 (logit域) - 不透明度偏移上限
- `sh_dc_max`: 0.1 - SH DC偏移上限
- `sh_rest_max`: 0.05 - SH rest偏移上限（通常比DC更小，因为rest分量通常更小）

**轴角到四元数转换** (`_axis_angle_to_quat`)：
- **目的**：将轴角表示转换为四元数，提供比直接预测四元数更平滑的梯度
- **实现**：使用无分支的 sinc 结构，避免阈值附近的不连续性
- **公式**：
  ```python
  theta = ||omega||  # [N, 1] - 旋转角度
  half_theta = theta * 0.5
  sinc_half = sin(half_theta) / (theta + eps)  # 避免除零，提供平滑梯度
  xyz = omega * sinc_half  # [N, 3] - 四元数的xyz分量
  w = cos(half_theta)      # [N, 1] - 四元数的w分量
  quat = [w, xyz]          # [N, 4] - wxyz格式
  ```
- **优势**：
  - 当 `theta → 0` 时，`sinc_half → 0.5`，因此 `xyz → omega/2`（正确的小角度近似）
  - 避免了直接除法可能导致的数值不稳定
  - 提供连续可微的梯度流

**偏移头初始化**：
- 所有偏移预测头的最后一层（输出层）被初始化为零权重和零偏置
- 这确保训练开始时预测的偏移量接近零，避免初始阶段的大幅跳跃
- 初始化代码：`nn.init.zeros_(layer.weight)` 和 `nn.init.zeros_(layer.bias)`

#### 6. 渲染参数计算 (`_compute_render_params_for_inner_iter` / `_render_params_from_offsets`)

**归属：** OffsetsMixin

**说明：** 训练时使用 `_compute_render_params_for_inner_iter`，内部对 rigid 做 `_transform_offsets_world_to_local` 后调用 `_render_params_from_offsets`。评估时使用 `_compute_render_params`（ProxyRenderingMixin）。

**计算过程：**
```
静态背景:
  NodeStateBg (分离) + OffsetsBg (可微) × Eta → Render ParamsBg (可微，世界坐标)

动态物体:
  NodeStateRigid (分离) + OffsetsRigid (可微，局部坐标) × Eta → Render ParamsRigid (可微，局部坐标)
  注意：偏移量需要从世界坐标变换到局部坐标（_transform_offsets_world_to_local）
```

**关键点：**
- 静态背景的渲染参数是**世界坐标**
- 动态物体的渲染参数是**局部坐标**（需要在渲染前变换到目标帧）

**具体计算（应用步长因子 eta）：**

1. **位置参数** (`means_r`)：
   ```python
   means_r = node_state.means + eta_means * offset_pos  # [N, 3]
   ```
   - **注意**：不在此处进行 clamp，以保持梯度流
   - 只有在写回 `NodeState` 时才进行 clamp（限制在 `[bbx_min, bbx_max]` 范围内）

2. **尺度参数** (`scales_log_r`, `scales_r`)：
   ```python
   scales_log_r = node_state.scales_log + eta_scales * offset_scales  # [N, 3]
   scales_r = exp(scales_log_r)  # [N, 3] - 转换到线性域
   ```

3. **旋转参数** (`quats_r`)：
   ```python
   quats_r = normalize(quat_multiply(node_state.quats, offset_quat))  # [N, 4]
   ```
   - 使用四元数乘法组合旋转：`q_result = q_node * q_offset`
   - 然后归一化以确保单位四元数

4. **不透明度参数** (`opacity_logit_r`, `opacities_r`)：
   ```python
   opacity_logit_r = node_state.opacity_logit + eta_opacity * offset_opacity  # [N, 1]
   opacities_r = sigmoid(opacity_logit_r).squeeze(-1)  # [N] - 转换到[0,1]范围
   ```

5. **SH颜色参数** (`sh_dc_r`, `sh_rest_r`, `colors_r`)：
   ```python
   # 提取DC和rest分量
   offset_sh_dc = offset_sh[:, :3]  # [N, 3]
   sh_rest_flat = offset_sh[:, 3:]   # [N, 3*(num_sh-1)]
   sh_rest_offset = sh_rest_flat.view(N, num_sh-1, 3)  # [N, num_sh-1, 3]
   
   # 应用偏移
   sh_dc_r = node_state.sh_dc + eta_sh_dc * offset_sh_dc  # [N, 3]
   sh_rest_r = node_state.sh_rest + eta_sh_rest * sh_rest_offset  # [N, num_sh-1, 3]
   
   # 组合成完整的SH系数
   colors_r = cat([sh_dc_r[:, None, :], sh_rest_r], dim=1)  # [N, num_sh, 3]
   ```

**步长因子（Eta）**：
- `eta_means`: 位置步长因子（默认1.0）
- `eta_scales`: 尺度步长因子（默认1.0）
- `eta_opacity`: 不透明度步长因子（默认1.0）
- `eta_sh_dc`: SH DC步长因子（默认1.0）
- `eta_sh_rest`: SH rest步长因子（默认1.0）
- **作用**：允许精细控制不同参数类型的更新幅度，通常在训练过程中保持固定
- **设计考虑**：通过调整这些因子，可以平衡不同参数类型的更新速度，避免某些参数更新过快导致训练不稳定

**输出字典结构：**
```python
{
    "means_r": [N, 3],           # 渲染用的位置（可微，未clamp）
    "scales_log_r": [N, 3],      # 渲染用的尺度对数（可微）
    "scales_r": [N, 3],          # 渲染用的尺度（exp(scales_log_r)）
    "quats_r": [N, 4],           # 渲染用的四元数（归一化，可微）
    "opacity_logit_r": [N, 1],   # 渲染用的不透明度对数（可微）
    "opacities_r": [N],          # 渲染用的不透明度（sigmoid(opacity_logit_r)）
    "sh_dc_r": [N, 3],           # 渲染用的SH DC分量（可微）
    "sh_rest_r": [N, num_sh-1, 3], # 渲染用的SH高阶分量（可微）
    "colors_r": [N, num_sh, 3],  # 完整的SH系数（用于渲染）
}
```

**关键设计点**：
1. **梯度流保护**：`means_r` 在计算时不进行 clamp，确保梯度可以正常反向传播
2. **数值稳定性**：使用对数域（log域）进行尺度更新，避免指数爆炸
3. **旋转组合**：通过四元数乘法组合旋转，确保旋转的连续性和可微性
4. **参数分离**：SH的DC和rest分量分别应用步长因子，允许独立控制

#### 7. 代理参数创建 (`_create_proxy_params`)

**归属：** ProxyRenderingMixin

**目的：** 创建可微的代理参数，用于多视角梯度累积

**操作：**
```python
proxy = render_param.detach().requires_grad_(True)
```

**关键点：**
- 代理参数从渲染参数中分离（detach），但重新启用梯度
- 这样可以在多个视角上累积梯度，然后一次性反向传播到渲染参数

#### 8. 多 Target 帧渲染与损失计算 (`_render_targets_and_accumulate_loss`)

**归属：** ProxyRenderingMixin

**循环结构：**
```python
# 代理参数在 _train_inner_iteration 中创建，传入此方法
for view_idx, target in enumerate(targets):
    target_frame_idx = target["frame_idx"]
    view = target["view"]
    gt_img = target["gt_image"]
    
    # 1. 设置 RigidNodes.cur_frame = target_frame_idx
    # 2. 按 idx_tgt_rigid[view_idx] 取 rigid 子集，变换到 target 帧世界坐标
    if view_idx < len(masks.idx_tgt_rigid) and masks.idx_tgt_rigid[view_idx].numel() > 0:
        idx = masks.idx_tgt_rigid[view_idx]
        rigid_means_local_subset = proxies_rigid["means_p"][idx]
        # ... 取子集后 _transform_rigid_to_world / _transform_rigid_quats_to_world
    
    # 3. 合并参数：有可见 rigid 时用 _merge_params_with_rigid_subset，否则用 _merge_all_params
    if rigid_subset["means"].numel() > 0:
        merged_* = _merge_params_with_rigid_subset(proxies_bg, proxies_distant, rigid_subset)
    else:
        merged_* = _merge_all_params(...)
    
    # 4. 渲染、计算损失、loss.backward()
    rgb, acc = _render_single_view(merged_params, view, height, width)
    loss = compute_loss(rgb, gt_img) / view_count
    loss.backward()
```

**损失函数：**
- `L2 Loss`: `mean((pred_rgb - gt_image) ** 2)`
- 每个 target 帧的损失除以 target 帧数量，实现平均

**关键设计点：**
- **Rigid 子集渲染**：仅渲染 `idx_tgt_rigid[view_idx]` 指定的可见点，使用 `_merge_params_with_rigid_subset` 合并
- **代理参数共享**：`proxies_bg` 和 `proxies_rigid` 在所有 target 帧中共享，梯度自动累积
- **可微变换**：坐标变换保持梯度连接，不使用 detach

#### 9. 梯度反向传播 (`_backward_to_render_params`)

**归属：** ProxyRenderingMixin

#### 9.1. 梯度反向传播机制

**两步反向传播：**

**第一步：** 从代理参数到渲染参数（分别处理静态、动态和远景）
```python
# 收集所有渲染参数和对应的代理梯度
render_tensors = [
    render_params_bg["means_r"],
    render_params_bg["scales_r"],
    render_params_bg["quats_r"],
    render_params_bg["opacities_r"],
    render_params_bg["colors_r"],
]
grad_tensors = [
    _grad_or_zero(proxies_bg["means_p"], "bg.means"),
    _grad_or_zero(proxies_bg["scales_p"], "bg.scales"),
    _grad_or_zero(proxies_bg["quats_p"], "bg.quats"),
    _grad_or_zero(proxies_bg["opacities_p"], "bg.opacities"),
    _grad_or_zero(proxies_bg["colors_p"], "bg.colors"),
]

# 动态物体（如果存在）
if render_params_rigid is not None and proxies_rigid is not None:
    render_tensors += [
        render_params_rigid["means_r"],
        render_params_rigid["scales_r"],
        render_params_rigid["quats_r"],
        render_params_rigid["opacities_r"],
        render_params_rigid["colors_r"],
    ]
    grad_tensors += [
        _grad_or_zero(proxies_rigid["means_p"], "rigid.means"),
        _grad_or_zero(proxies_rigid["scales_p"], "rigid.scales"),
        _grad_or_zero(proxies_rigid["quats_p"], "rigid.quats"),
        _grad_or_zero(proxies_rigid["opacities_p"], "rigid.opacities"),
        _grad_or_zero(proxies_rigid["colors_p"], "rigid.colors"),
    ]

# 背景远景（如果存在）
if render_params_distant is not None and proxies_distant is not None:
    render_tensors += [
        render_params_distant["means_r"],
        render_params_distant["scales_r"],
        render_params_distant["quats_r"],
        render_params_distant["opacities_r"],
        render_params_distant["colors_r"],
    ]
    grad_tensors += [
        _grad_or_zero(proxies_distant["means_p"], "distant.means"),
        _grad_or_zero(proxies_distant["scales_p"], "distant.scales"),
        _grad_or_zero(proxies_distant["quats_p"], "distant.quats"),
        _grad_or_zero(proxies_distant["opacities_p"], "distant.opacities"),
        _grad_or_zero(proxies_distant["colors_p"], "distant.colors"),
    ]

# 一次性反向传播
torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)
```

**第二步：** 从渲染参数到网络参数（自动）
- 通过 `offset_*` 参数链
- 最终更新所有 MLP、sparse_conv 和 feature_fusion 的参数
- **注意**：静态、动态和远景使用相同的 MLP 网络，梯度会自动合并

**grad_warned 状态**：`_grad_or_zero` 使用 `self._proxy_grad_warned` 避免重复打印 "Proxy gradient is None" 的 warning。

#### 10. 状态更新 (`_update_node_states`)

**归属：** NodeStateMixin

**条件：** `update_state == True`

**操作：**
```python
with torch.no_grad():
    # 更新静态背景 NodeState
    means_clamped = torch.clamp(
        render_params_bg["means_r"].detach(),
        min=self.bbx_min,
        max=self.bbx_max
    )
    node_state_bg.means.copy_(means_clamped)
    node_state_bg.scales_log.copy_(render_params_bg["scales_log_r"].detach())
    node_state_bg.quats.copy_(render_params_bg["quats_r"].detach())
    node_state_bg.opacity_logit.copy_(render_params_bg["opacity_logit_r"].detach())
    node_state_bg.sh_dc.copy_(render_params_bg["sh_dc_r"].detach())
    node_state_bg.sh_rest.copy_(render_params_bg["sh_rest_r"].detach())
    
    # 更新动态物体 NodeState（局部坐标）
    if node_state_rigid is not None and render_params_rigid is not None:
        node_state_rigid.means.copy_(render_params_rigid["means_r"].detach())
        node_state_rigid.scales_log.copy_(render_params_rigid["scales_log_r"].detach())
        node_state_rigid.quats.copy_(render_params_rigid["quats_r"].detach())
        node_state_rigid.opacity_logit.copy_(render_params_rigid["opacity_logit_r"].detach())
        node_state_rigid.sh_dc.copy_(render_params_rigid["sh_dc_r"].detach())
        node_state_rigid.sh_rest.copy_(render_params_rigid["sh_rest_r"].detach())
    
    # 更新背景远景 NodeState（如果存在）
    if node_state_distant is not None and render_params_distant is not None:
        means_distant = torch.clamp(
            render_params_distant["means_r"].detach(),
            min=self.input_aabb_min,
            max=self.input_aabb_max,
        )
        node_state_distant.means.copy_(means_distant)
        node_state_distant.scales_log.copy_(render_params_distant["scales_log_r"].detach())
        node_state_distant.quats.copy_(render_params_distant["quats_r"].detach())
        node_state_distant.opacity_logit.copy_(render_params_distant["opacity_logit_r"].detach())
        node_state_distant.sh_dc.copy_(render_params_distant["sh_dc_r"].detach())
        node_state_distant.sh_rest.copy_(render_params_distant["sh_rest_r"].detach())
```

**注意：** 
- 所有更新都是分离的（detached），保持 NodeState 作为缓冲区
- 静态背景的 `means` 在写回时进行 clamp，限制在边界框范围内（`bbx_min` 到 `bbx_max`），但在反向传播时保持不限制以保持梯度流
- 动态物体的 `means` 是局部坐标，不需要 clamp（边界框限制在变换到世界坐标时处理）
- 背景远景的 `means` 在写回时进行 clamp，限制在输入AABB范围内（`input_aabb_min` 到 `input_aabb_max`）

---

## 3D特征体积构建详细流程

本节深入讲解 3D 特征体积构建的详细实现，这是整个训练流程中的核心部分。

### 代码片段

**归属：** `models/streetforward/feature_volume_mixin.py` 的 `_build_3d_feature_volume`

```python
# 合并 means_all, anchor_rgb_all 后（来自静态+动态点云）
sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
    raw_coords=means_all.clone(),
    feats=anchor_rgb_all,
    Bbx_max=self.bbx_max,
    Bbx_min=self.bbx_min,
    voxel_size=self.voxel_size,
    device=self.device,
)
feat_3d = self.sparse_conv(sparse_feat)
dense_volume = self.sparse_to_dense_volume(
    sparse_tensor=feat_3d,
    coords=valid_coords,
    vol_dim=vol_dim,
).unsqueeze(dim=0)
dense_volume = dense_volume.permute(0, 4, 3, 2, 1)  # [1, C, D, H, W]

# 分别为 bg 和 rigid 点计算 grid_coords 并插值
grid_coords_bg = self.get_grid_coords(means_bg, self.bbx_min, vol_dim, self.voxel_size)
feat_3d_crop_bg = self.interpolate_features(grid_coords_bg, dense_volume)
# rigid 同理，使用 means_rigid_world
```

### 输入数据准备

在执行这段代码之前，需要准备以下数据：

#### 1. `means_s` - 分离的位置参数

```python
means_s = node_state.means  # [N, 3]
```

**数据说明：**
- **来源：** 从 `NodeState` 中获取，代表 N 个 Gaussian 的中心位置
- **类型：** `torch.Tensor`
- **形状：** `[N, 3]`，其中 N 是点的数量，3 表示 (x, y, z) 坐标
- **特性：** **分离的（detached）**，不参与梯度计算，作为稳定的参考点
- **坐标系：** 世界坐标系

**示例值：**
```python
means_s = tensor([[10.5, 2.3, 25.1],
                  [11.2, 2.4, 25.3],
                  ...])
```

#### 2. `anchor_rgb` - RGB 颜色特征

```python
anchor_rgb = _sh_to_rgb(node_state.sh_dc)  # [N, 3]
```

**数据说明：**
- **来源：** 从 `NodeState.sh_dc` 转换而来，`sh_dc` 是球谐函数的 DC（直流）分量
- **转换函数：** `_sh_to_rgb()`
  ```python
  c0 = 0.28209479177387814  # SH基函数的归一化常数
  rgb = sh * c0 + 0.5
  ```
- **形状：** `[N, 3]`，表示 N 个点的 RGB 颜色
- **值域：** RGB 值通常在 [0, 1] 范围内
- **用途：** 作为 3D 特征体积的初始特征（用于稀疏卷积）

**数据流：**
```
NodeState.sh_dc [N, 3] (SH DC分量，可负值)
  ↓ _sh_to_rgb()
anchor_rgb [N, 3] (RGB颜色，范围[0,1])
```

#### 3. `self.bbx_min` 和 `self.bbx_max` - 边界框

```python
bbx_min = tensor([-20.0, -20.0, -20.0])  # 默认值
bbx_max = tensor([20.0, 4.8, 70.0])      # 默认值
```

**数据说明：**
- **类型：** `torch.Tensor`
- **形状：** `[3]`，表示 (x_min, y_min, z_min) 或 (x_max, y_max, z_max)
- **用途：** 定义 3D 特征体积的空间范围
- **坐标系：** 世界坐标系

#### 4. `self.voxel_size` - 体素大小

```python
voxel_size = 0.1  # 默认值（米）
```

**数据说明：**
- **类型：** `float`
- **含义：** 每个体素的物理尺寸（单位：米）
- **用途：** 将连续空间离散化为体素网格

---

### 步骤1：构建稀疏张量 (`construct_sparse_tensor`)

#### 函数调用

```python
sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
    raw_coords=means_s.clone(),  # [N, 3]
    feats=anchor_rgb,             # [N, 3]
    Bbx_max=self.bbx_max,         # [3]
    Bbx_min=self.bbx_min,         # [3]
    voxel_size=self.voxel_size,   # float
)
```

**关键点：**
- 使用 `means_s.clone()` 而不是 `means_s`，确保不修改原始数据

#### 函数实现（nerfstudio 版本）

**主要流程：**

1. **提取边界框值（转换为CPU标量）**
   ```python
   X_MIN = Bbx_min[0].cpu().item()
   X_MAX = Bbx_max[0].cpu().item()
   Y_MIN = Bbx_min[1].cpu().item()
   Y_MAX = Bbx_max[1].cpu().item()
   Z_MIN = Bbx_min[2].cpu().item()
   Z_MAX = Bbx_max[2].cpu().item()
   ```

2. **转换为numpy数组（需要detach）**
   ```python
   if isinstance(raw_coords, torch.Tensor):
       raw_coords = raw_coords.detach().cpu().numpy()
   if isinstance(feats, torch.Tensor):
       feats = feats.detach().cpu().numpy()
   ```
   **注意：** 必须 `detach()`，因为这些张量可能带有梯度信息

3. **计算体积维度**
   ```python
   bbx_max = np.array([X_MAX, Y_MAX, Z_MAX])
   bbx_min = np.array([X_MIN, Y_MIN, Z_MIN])
   vol_dim = (bbx_max - bbx_min) / voxel_size  # 例如: [400, 248, 900]
   vol_dim = vol_dim.astype(int).tolist()      # [X, Y, Z] 格式
   ```
   **示例计算：**
   ```python
   # 假设 bbx_max = [20, 4.8, 70], bbx_min = [-20, -20, -20], voxel_size = 0.1
   vol_dim = ([20, 4.8, 70] - [-20, -20, -20]) / 0.1
           = [40, 24.8, 90] / 0.1
           = [400, 248, 900]  # X=400, Y=248, Z=900
   ```

4. **将坐标相对于边界框原点**
   ```python
   raw_coords -= np.array([X_MIN, Y_MIN, Z_MIN]).astype(int)
   ```
   **示例：**
   ```python
   # 原始坐标 [10.5, 2.3, 25.1]
   # 减去 bbx_min [-20, -20, -20]
   # 得到 [30.5, 22.3, 45.1]
   ```

5. **体素化（voxelization）- 关键步骤！**
   ```python
   coords, indices = sparse_quantize(raw_coords, voxel_size, return_index=True)
   ```
   **功能：** 将多个点映射到同一个体素，返回唯一的体素坐标和索引
   
   **工作原理：**
   ```python
   # 伪代码示例
   voxel_coords = floor(raw_coords / voxel_size)  # 离散化
   unique_coords, indices = unique(voxel_coords, return_inverse=True)
   ```
   
   **示例：**
   ```python
   # 假设 voxel_size = 0.1
   raw_coords = [[10.53, 2.34, 25.17],  # 点1
                 [10.56, 2.35, 25.18],  # 点2 - 与点1在同一体素
                 [11.23, 2.41, 25.32]]  # 点3
   
   # 体素化后
   coords = [[105, 23, 251],  # 体素1 (点1和点2合并)
             [112, 24, 253]]  # 体素2 (点3)
   indices = [0, 0, 1]  # 点1和点2都映射到体素0，点3映射到体素1
   ```
   
   **结果：** M ≤ N，因为可能有多个点映射到同一体素

6. **转换为torch张量并添加batch维度**
   ```python
   coords = torch.tensor(coords, dtype=torch.int).cuda()  # [M, 3]
   zeros = torch.zeros(coords.shape[0], 1).cuda()         # [M, 1]
   coords = torch.cat((zeros, coords), dim=1).to(torch.int32)  # [M, 4] - [B, X, Y, Z]
   ```
   **格式说明：** `[B, X, Y, Z]`，其中 `B=0` 表示batch维度

7. **根据索引选择特征**
   ```python
   feats = torch.tensor(feats[indices], dtype=torch.float).cuda()  # [M, 3]
   ```
   **注意：** 体素化后，如果多个点映射到同一体素，特征会被选择（通常使用第一个点的特征）

8. **创建SparseTensor**
   ```python
   sparse_feat = SparseTensor(feats, coords=coords)
   ```
   **数据结构：**
   - `sparse_feat.feats`: `[M, 3]` - RGB特征
   - `sparse_feat.coords`: `[M, 4]` - 体素坐标 `[B, X, Y, Z]`

9. **返回结果**
   ```python
   return sparse_feat, vol_dim, coords[:, 1:]  # coords[:, 1:] = [X, Y, Z]
   ```

#### 输出数据

##### 1. `sparse_feat` - 稀疏特征张量

**类型：** `SparseTensor` (torchsparse库)

**结构：**
- `sparse_feat.feats`: `[M, 3]` - RGB特征
- `sparse_feat.coords`: `[M, 4]` - 体素坐标 `[B, X, Y, Z]`

**数据说明：**
- `M` 是唯一体素的数量（M ≤ N，因为可能有重复体素）
- 特征维度为 3（RGB颜色）
- 只存储有数据的体素，节省内存

##### 2. `vol_dim` - 体积维度

**类型：** `List[int]` 或 `torch.Tensor`

**格式：** `[X, Y, Z]` / `[W, H, D]`（permute 后与坐标轴一致）

**计算：**
```python
vol_dim = (bbx_max - bbx_min) / voxel_size
```

**示例值：**
```python
vol_dim = [400, 248, 900]  # X=400, Y=248, Z=900（即 W×H×D）
```

##### 3. `valid_coords` - 有效坐标

**类型：** `torch.Tensor`

**形状：** `[M, 3]` - `[X, Y, Z]` 格式（去掉了batch维度）

**用途：** 后续用于将稀疏特征转换回密集体积

---

### 步骤2：稀疏卷积 (`sparse_conv`)

#### 函数调用

```python
feat_3d = self.sparse_conv(sparse_feat)  # SparseTensor → SparseTensor
```

#### 网络结构（SparseCostRegNet）

**输入：** `SparseTensor` with features `[M, 3]` (RGB)

**网络架构：**

```
输入: [M, 3] RGB特征
  ↓
conv0: BasicSparseConvolutionBlock(3 → outdim)  # outdim默认32
  → [M, 32]
  ↓
下采样路径:
conv1: BasicSparseConvolutionBlock(32 → 16, stride=2)
  → [M1, 16] (体素数量减少)
  ↓
conv2: BasicSparseConvolutionBlock(16 → 16)
  → [M1, 16]
  ↓
conv3: BasicSparseConvolutionBlock(16 → 32, stride=2)
  → [M2, 32] (体素数量进一步减少)
  ↓
conv4: BasicSparseConvolutionBlock(32 → 32)
  → [M2, 32]
  ↓
conv5: BasicSparseConvolutionBlock(32 → 64, stride=2)
  → [M3, 64]
  ↓
conv6: BasicSparseConvolutionBlock(64 → 64)
  → [M3, 64]
  ↓
上采样路径（带残差连接）:
conv7: BasicSparseDeconvolutionBlock(64 → 32, stride=2)
  → [M2, 32] + conv4的残差
  ↓
conv9: BasicSparseDeconvolutionBlock(32 → 16, stride=2)
  → [M1, 16]
  ↓
conv11: BasicSparseDeconvolutionBlock(16 → outdim, stride=2)
  → [M, outdim] (恢复到原始体素数量)
```

**关键点：**
- 使用**稀疏卷积**，只在有数据的体素上计算，高效
- **U-Net 风格**结构：下采样提取特征，上采样恢复分辨率
- **残差连接**：`conv4 + conv7`，保留细节信息
- 最终输出与输入具有**相同的体素坐标**（相同数量的体素）

#### 输出数据

**`feat_3d` - 3D特征**

**类型：** `SparseTensor`

**结构：**
- `feat_3d.feats`: `[M, outdim]` - 3D特征（默认outdim=32）
- `feat_3d.coords`: `[M, 4]` - 体素坐标（与输入相同）

**数据说明：**
- 特征维度从 3（RGB）扩展到 outdim（默认32）
- 每个体素现在有更丰富的特征表示

---

### 步骤3：稀疏转密集 (`sparse_to_dense_volume`)

#### 函数调用

```python
dense_volume = self.sparse_to_dense_volume(
    sparse_tensor=feat_3d,      # SparseTensor [M, outdim]
    coords=valid_coords,        # [M, 3]
    vol_dim=vol_dim,            # [X, Y, Z]
).unsqueeze(dim=0)              # [1, D, H, W, C]
```

#### 函数实现（nerfstudio 版本）

```python
def sparse_to_dense_volume(sparse_tensor, coords, vol_dim, default_val=0):
    c = sparse_tensor.shape[-1]  # outdim (例如32)
    coords = coords.to(torch.int64)
    
    # 1. 限制坐标在有效范围内（防止越界）
    # vol_dim 是 [X, Y, Z] 格式，coords 也是 [X, Y, Z] 格式
    coords[:, 0] = coords[:, 0].clamp(0, vol_dim[0] - 1)  # X维度
    coords[:, 1] = coords[:, 1].clamp(0, vol_dim[1] - 1)  # Y维度
    coords[:, 2] = coords[:, 2].clamp(0, vol_dim[2] - 1)  # Z维度
    
    # 2. 创建密集体积（全部初始化为default_val）
    device = sparse_tensor.device
    dense = torch.full(
        [vol_dim[0], vol_dim[1], vol_dim[2], c],  # [X, Y, Z, C]
        float(default_val),
        device=device
    )
    
    # 3. 将稀疏特征填入对应位置
    dense[coords[:, 0], coords[:, 1], coords[:, 2]] = sparse_tensor
    # coords[:, 0]是X索引，coords[:, 1]是Y索引，coords[:, 2]是Z索引
    
    return dense  # [X, Y, Z, C]
```

#### 关键操作详解

##### 1. 索引操作

```python
dense[coords[:, 0], coords[:, 1], coords[:, 2]] = sparse_tensor
```

**工作原理：**
- 使用**高级索引**（advanced indexing）
- `coords[:, 0]` 是 D 维度的索引数组
- `coords[:, 1]` 是 H 维度的索引数组
- `coords[:, 2]` 是 W 维度的索引数组

**示例：**
```python
# 假设 coords = [[10, 20, 30], [15, 25, 35]]
# sparse_tensor = [[f1_0, f1_1, ...], [f2_0, f2_1, ...]]

# 等价于：
dense[10, 20, 30] = [f1_0, f1_1, ...]  # 将特征填入体素(10,20,30)
dense[15, 25, 35] = [f2_0, f2_1, ...]  # 将特征填入体素(15,25,35)

# 其他位置保持 default_val (0)
```

##### 2. `unsqueeze(dim=0)`

```python
dense_volume = dense.unsqueeze(dim=0)  # [X, Y, Z, C] → [1, X, Y, Z, C]
```

**目的：** 添加 batch 维度，便于后续操作

**注意：** `sparse_to_dense_volume` 返回的格式是 `[X, Y, Z, C]`，其中 `vol_dim` 是 `[X, Y, Z]` 格式（对应世界坐标系的 X、Y、Z 轴）。在后续的 `permute` 操作中，会将其转换为 `grid_sample` 需要的 `[1, C, Z, Y, X]` = `[1, C, D, H, W]` 格式。

#### 输出数据

**`dense_volume` - 密集体积**

**形状（unsqueeze 后）：** `[1, X, Y, Z, C]`

**数据说明：**
- 大部分位置为 `default_val`（通常为 0）
- 只有 `coords` 指定的位置有实际特征值
- 这是一个**稀疏的密集表示**（sparse dense representation）
- 在后续的 `permute` 操作中，会转换为 `[1, C, Z, Y, X]` = `[1, C, D, H, W]` 格式以匹配 `grid_sample` 的要求

---

### 步骤4：维度重排 (`permute`)

#### 函数调用

```python
dense_volume = dense_volume.permute(0, 4, 3, 2, 1)
# [1, D, H, W, C] → [1, C, Z, Y, X] = [1, C, D, H, W]
```

#### 维度变换详解

**原始维度：** `[1, D, H, W, C]`
- `0`: batch维度 (1)
- `1`: D (深度，对应Z轴)
- `2`: H (高度，对应Y轴)
- `3`: W (宽度，对应X轴)
- `4`: C (特征通道)

**目标维度：** `[1, C, Z, Y, X]` = `[1, C, D, H, W]`
- `0`: batch维度 (1)
- `1`: C (特征通道)
- `2`: Z (深度，对应D)
- `3`: Y (高度，对应H)
- `4`: X (宽度，对应W)

**维度映射：**
```
索引映射: 0→0, 4→1, 3→2, 2→3, 1→4
```

**为什么需要这个变换？**

PyTorch 的 `grid_sample` 函数（5D版本）期望输入格式为 `[B, C, D, H, W]`，其中：
- `D` 是深度维度（Z轴）
- `H` 是高度维度（Y轴）
- `W` 是宽度维度（X轴）

`sparse_to_dense_volume` 返回的格式是 `[D, H, W, C]`，经过 `unsqueeze(0)` 后变成 `[1, D, H, W, C]`。通过 `permute(0, 4, 3, 2, 1)` 变换，我们将特征通道移到第二个维度，并确保维度顺序与 `grid_sample` 的要求匹配：`[1, C, D, H, W]`。

---

### 步骤5：计算网格坐标 (`get_grid_coords`)

#### 函数调用

```python
grid_coords = self.get_grid_coords(
    means_s,           # [N, 3] - 原始点坐标（世界坐标系）
    self.bbx_min,      # [3] - 边界框最小值
    vol_dim,           # [X, Y, Z] - 体积维度
    self.voxel_size,   # float - 体素大小
)
```

#### 函数实现

```python
def get_grid_coords(
    self, position_w: torch.Tensor, bbx_min: torch.Tensor, vol_dim, voxel_size: float
) -> torch.Tensor:
    # 1. 将坐标相对于边界框原点
    pts = position_w - bbx_min.to(position_w.device)  # [N, 3]
    
    # 2. 转换为体素索引（浮点数索引）
    x_index = pts[..., 0] / voxel_size  # [N] - W方向索引
    y_index = pts[..., 1] / voxel_size  # [N] - H方向索引
    z_index = pts[..., 2] / voxel_size  # [N] - D方向索引
    
    # 3. 确保vol_dim是torch.Tensor
    if isinstance(vol_dim, (list, tuple)):
        vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
    elif not isinstance(vol_dim, torch.Tensor):
        vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
    else:
        vol_dim = vol_dim.to(position_w.device).float()
    
    # 4. 归一化到[-1, 1]范围（grid_sample的要求）；vol_dim 为 [X, Y, Z]
    x_norm = x_index / (vol_dim[0] - 1).clamp(min=1.0) * 2 - 1  # [N]
    y_norm = y_index / (vol_dim[1] - 1).clamp(min=1.0) * 2 - 1  # [N]
    z_norm = z_index / (vol_dim[2] - 1).clamp(min=1.0) * 2 - 1  # [N]
    
    # 6. 堆叠成坐标张量
    grid_coords = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # [N, 3]
    return grid_coords
```

#### 关键操作详解

##### 1. 坐标变换

**步骤1：相对坐标**
```python
pts = position_w - bbx_min
# 例如: [10.5, 2.3, 25.1] - [-20.0, -20.0, -20.0] = [30.5, 22.3, 45.1]
```

**步骤2：体素索引**
```python
x_index = pts[..., 0] / voxel_size  # W方向
y_index = pts[..., 1] / voxel_size  # H方向
z_index = pts[..., 2] / voxel_size  # D方向
# 例如: 30.5 / 0.1 = 305
```

**步骤3：归一化**
```python
x_norm = (x_index / (vol_dim[0] - 1)) * 2 - 1
# 例如: (305 / (vol_dim[0] - 1)) * 2 - 1，在[-1, 1]范围内
```

**归一化公式：**
```
normalized = (index / (dim - 1)) * 2 - 1
```

**边界情况：**
- `index = 0` → `normalized = -1`
- `index = dim-1` → `normalized = 1`
- `index = (dim-1)/2` → `normalized = 0`

#### 输出数据

**`grid_coords` - 归一化网格坐标**

**形状：** `[N, 3]`

**格式：** `[x_norm, y_norm, z_norm]`，每个值在 `[-1, 1]` 范围内

**坐标系：**
- `x_norm`: W方向（宽度）
- `y_norm`: H方向（高度）
- `z_norm`: D方向（深度）

**注意：** 这里使用的是 `[x, y, z]` 顺序，对应 `[W, H, D]` 维度。

---

### 步骤6：特征插值 (`interpolate_features`)

#### 函数调用

```python
feat_3d_crop = self.interpolate_features(grid_coords, dense_volume)
# grid_coords: [N, 3]
# dense_volume: [1, C, W, D, H] (经过permute后)
# 输出: [N, C]
```

#### 函数实现

```python
def interpolate_features(
    self, grid_coords: torch.Tensor, feature_volume: torch.Tensor
) -> torch.Tensor:
    # 1. 扩展grid_coords维度以匹配grid_sample的要求
    # grid_sample需要: [B, N, 1, 1, 3] 格式
    grid_coords_expanded = grid_coords[None, None, None, ...]  # [1, 1, 1, N, 3]
    
    # 2. 使用三线性插值从体积中提取特征
    feature = torch.nn.functional.grid_sample(
        feature_volume,           # [1, C, W, D, H] - 输入体积
        grid_coords_expanded,     # [1, 1, 1, N, 3] - 采样坐标
        mode="bilinear",          # 双线性插值（3D中实际是三线性插值）
        align_corners=True,       # 对齐角点（与归一化坐标对应）
        padding_mode="zeros",     # 边界外填充0
    )
    # 输出: [1, C, 1, 1, N]
    
    # 3. 重塑并转置
    return feature[0, :, 0, 0, :].T  # [1, C, 1, 1, N] → [C, N] → [N, C]
```

#### 关键操作详解

##### 1. `grid_sample` - 网格采样

**函数签名：**
```python
torch.nn.functional.grid_sample(
    input,      # [B, C, D_in, H_in, W_in] - 输入体积
    grid,       # [B, D_out, H_out, W_out, 3] - 采样网格
    mode,       # "bilinear" | "nearest"
    align_corners,  # True | False
    padding_mode,   # "zeros" | "border" | "reflection"
)
```

**在我们的代码中：**
- `input`: `[1, C, W, D, H]` - 特征体积
- `grid`: `[1, 1, 1, N, 3]` - 采样坐标（每个点一个坐标）

**插值模式：**
- `mode="bilinear"`: 在3D空间中使用**三线性插值**
- `align_corners=True`: 确保 `-1` 和 `1` 对应体积的边界

**工作原理：**
1. 对于每个 `grid_coords[i] = [x, y, z]`
2. 在 `feature_volume` 中找到对应的位置
3. 使用周围8个体素的加权平均计算插值特征

**数学公式（三线性插值）：**
```
对于坐标 (x, y, z)，找到8个相邻体素：
- (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0)
- (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)

插值特征 = Σ(权重_i × 特征_i)
权重基于距离计算
```

##### 2. 维度变换

**输入：** `grid_coords` - `[N, 3]`

**扩展：** `grid_coords_expanded = grid_coords[None, None, None, ...]`
- `[N, 3]` → `[1, 1, 1, N, 3]`
- 添加了 batch 和空间维度

**输出：** `feature` - `[1, C, 1, 1, N]`
- 第一个 `1`: batch维度
- `C`: 特征通道
- `1, 1`: 空间维度（因为我们只采样N个点）
- `N`: 点的数量

**最终输出：** `feature[0, :, 0, 0, :].T`
- `[0, :, 0, 0, :]` → `[C, N]` (去掉batch和空间维度)
- `.T` → `[N, C]` (转置)

#### 输出数据

**`feat_3d_crop` - 每个点的3D特征**

**形状：** `[N, C]` 其中 `C = outdim` (默认32)

**数据说明：**
- 每个原始点（`means_s`）现在都有一个对应的3D特征
- 特征通过三线性插值从密集体积中提取
- 这些特征将用于后续的偏移量预测

---

### 完整数据流总结

```
输入:
  means_s: [N, 3] - 点位置（世界坐标）
  anchor_rgb: [N, 3] - RGB特征
  bbx_min: [3] - 边界框最小值
  bbx_max: [3] - 边界框最大值
  voxel_size: float - 体素大小

步骤1: construct_sparse_tensor
  → sparse_feat: SparseTensor([M, 3], [M, 4]) - 稀疏特征
  → vol_dim: [X, Y, Z] - 体积维度
  → valid_coords: [M, 3] - 有效体素坐标

步骤2: sparse_conv
  → feat_3d: SparseTensor([M, C], [M, 4]) - 3D特征（C=32）

步骤3: sparse_to_dense_volume
  → dense_volume: [X, Y, Z, C] (其中 vol_dim 是 [X, Y, Z])
  → unsqueeze: [1, X, Y, Z, C]

步骤4: permute
  → dense_volume: [1, C, Z, Y, X] = [1, C, D, H, W]

步骤5: get_grid_coords
  → grid_coords: [N, 3] - 归一化坐标 [-1, 1]

步骤6: interpolate_features
  → feat_3d_crop: [N, C] - 每个点的3D特征

输出:
  feat_3d_crop: [N, C] - 用于后续偏移量预测
```

### 关键设计点

1. **稀疏到密集的转换：** 先使用稀疏卷积（高效），再转换为密集体积（便于插值）
2. **体素化：** 多个点可能映射到同一体素，减少计算量
3. **特征提取：** 通过稀疏卷积提取丰富的3D特征表示
4. **特征插值：** 使用三线性插值从体积中提取每个点的特征
5. **内存效率：** 稀疏表示节省内存，只在需要时转换为密集表示

---

## 数据结构详解

### 1. NodeStateBackground

**定义：**
```python
@dataclass
class NodeStateBackground:
    means: torch.Tensor          # [N_bg, 3] - Gaussian中心位置（世界坐标）
    scales_log: torch.Tensor     # [N_bg, 3] - 尺度的对数（3个轴）
    quats: torch.Tensor          # [N_bg, 4] - 旋转四元数（wxyz格式）
    opacity_logit: torch.Tensor  # [N_bg, 1] - 不透明度的logit值
    sh_dc: torch.Tensor          # [N_bg, 3] - 球谐函数DC分量（RGB）
    sh_rest: torch.Tensor        # [N_bg, num_sh-1, 3] - 球谐函数高阶分量
```

**特性：**
- 所有张量都是分离的（detached），不参与梯度计算
- 每个 `(scene_id, segment_id)` 对应一个 NodeStateBackground
- 存储在 `self.node_states: Dict[Tuple[int, int], NodeStateBackground]` 中（兼容旧代码，实际是 Background）

**初始化：**
- `means`: 从静态背景点云坐标初始化（世界坐标）
- `scales_log`: 基于 k-NN 距离计算（`log(clamp(avg_dist, min=1e-3))`）
- `quats`: 随机生成单位四元数
- `opacity_logit`: 初始化为 `logit(0.1)`
- `sh_dc`: 从点云颜色转换（`(rgb - 0.5) / c0`）
- `sh_rest`: 初始化为零

### 1.1. NodeStateRigid

**定义：**
```python
@dataclass
class NodeStateRigid:
    means: torch.Tensor          # [N_rigid, 3] - Gaussian中心位置（局部坐标）
    scales_log: torch.Tensor     # [N_rigid, 3] - 尺度的对数（3个轴）
    quats: torch.Tensor          # [N_rigid, 4] - 旋转四元数（wxyz格式，局部旋转）
    opacity_logit: torch.Tensor  # [N_rigid, 1] - 不透明度的logit值
    sh_dc: torch.Tensor          # [N_rigid, 3] - 球谐函数DC分量（RGB）
    sh_rest: torch.Tensor        # [N_rigid, num_sh-1, 3] - 球谐函数高阶分量
    point_ids: torch.Tensor      # [N_rigid, 1] - 每个点属于哪个实例
    instances_quats: torch.Tensor # [num_frames, num_instances, 4] - 实例旋转（wxyz格式）
    instances_trans: torch.Tensor # [num_frames, num_instances, 3] - 实例平移
    instances_fv: torch.Tensor   # [num_frames, num_instances] - 实例可见性（bool）
    instance_ids: List[int]      # 实例ID列表
    frame_ids: List[int]         # 帧ID列表（用于索引 instances_*）
    cur_frame: int               # 当前帧索引（用于变换）
```

**特性：**
- 所有张量都是分离的（detached），不参与梯度计算
- 每个 `(scene_id, segment_id)` 对应一个 NodeStateRigid（如果存在动态物体）
- 存储在 `self.node_states_rigid: Dict[Tuple[int, int], Optional[NodeStateRigid]]` 中
- 如果场景没有动态物体，值为 `None`

**初始化：**
- `means`: 从动态物体点云坐标初始化（局部坐标）
- `scales_log`: 基于 k-NN 距离计算
- `quats`: 随机生成单位四元数（局部旋转）
- `opacity_logit`: 初始化为 `logit(0.1)`
- `sh_dc`: 从点云颜色转换
- `sh_rest`: 初始化为零
- `point_ids`: 从点云生成时记录每个点属于哪个实例
- `instances_quats`: 从 `dynamic_info` 初始化，包含所有帧的实例旋转
- `instances_trans`: 从 `dynamic_info` 初始化，包含所有帧的实例平移
- `instances_fv`: 从 `dynamic_info` 初始化，记录每个帧每个实例的可见性

### 1.2. NodeStateDistant

**定义：**
```python
@dataclass
class NodeStateDistant:
    means: torch.Tensor          # [N_distant, 3] - Gaussian中心位置（世界坐标）
    scales_log: torch.Tensor     # [N_distant, 3] - 尺度的对数（3个轴）
    quats: torch.Tensor          # [N_distant, 4] - 旋转四元数（wxyz格式）
    opacity_logit: torch.Tensor  # [N_distant, 1] - 不透明度的logit值
    sh_dc: torch.Tensor          # [N_distant, 3] - 球谐函数DC分量（RGB）
    sh_rest: torch.Tensor        # [N_distant, num_sh-1, 3] - 球谐函数高阶分量
```

**特性：**
- 所有张量都是分离的（detached），不参与梯度计算
- 每个 `(scene_id, segment_id)` 对应一个 NodeStateDistant（如果启用背景远景）
- 存储在 `self.node_states_distant: Dict[Tuple[int, int], Optional[NodeStateDistant]]` 中
- 如果场景没有背景远景，值为 `None`
- 用于表示远离主要场景的背景元素（如天空、远山等）

**初始化：**
- `means`: 从背景远景点云坐标初始化（世界坐标）
- `scales_log`: 基于 k-NN 距离计算
- `quats`: 随机生成单位四元数
- `opacity_logit`: 初始化为 `logit(0.1)`
- `sh_dc`: 从点云颜色转换
- `sh_rest`: 初始化为零

### 2. Batch 输入数据

**结构：**
```python
batch = {
    "scene_id": int,                    # 场景ID
    "segment_id": int,                  # 片段ID
    "source_frame_idx": int,            # Source 帧索引（场景全局 frame_idx）
    "pointcloud": Union[dict, object],   # 点云数据
    "dynamic_info": Optional[Dict],      # 动态物体信息（可选）
    "targets": List[Dict],              # Target 帧列表（推荐格式）
    "target_views": List[View],         # 目标视角列表（兼容旧格式）
    "gt_images": List[torch.Tensor],    # 真实图像列表（兼容旧格式）
    "test_views": Optional[List[View]], # 测试视角列表（可选，用于评估）
    "test_images": Optional[List[torch.Tensor]],  # 测试图像列表（可选，用于评估）
}
```

**点云格式：**
- **字典格式（推荐）：**
  ```python
  {
      "background": np.ndarray,  # [N_bg, 6] - 静态背景 [x, y, z, r, g, b]（世界坐标）
      "dynamic": Dict[int, np.ndarray],  # {instance_id: [N_i, 6]} - 动态物体点云（局部坐标）
  }
  ```
- **对象格式：** 需有 `points` 和 `colors` 属性

**动态物体信息格式：**
```python
dynamic_info = {
    frame_idx: {
        "instances": {
            instance_id: {
                "quat": List[4],  # [w, x, y, z] 四元数（wxyz格式）
                "trans": List[3],  # [x, y, z] 平移向量
            }
        }
    }
}
```

**Target 格式（推荐）：**
```python
targets = [
    {
        "frame_idx": int,           # 场景全局 frame_idx
        "view": View,               # 相机视角
        "gt_image": torch.Tensor,   # [H, W, 3] 真实图像
    },
    ...
]
```

**View 对象：**
- `camtoworlds`: `[4, 4]` 或 `[B, 4, 4]` - 相机到世界变换矩阵
- `Ks` 或 `K`: `[3, 3]` 或 `[B, 3, 3]` - 相机内参矩阵

### 3. 中间数据流

#### 3D 特征体积构建阶段

| 变量名 | 形状 | 说明 |
|--------|------|------|
| `means_bg` | `[N_bg, 3]` | 从NodeStateBg获取的静态背景位置（分离，世界坐标） |
| `means_rigid_local` | `[N_rigid, 3]` | 从NodeStateRigid获取的动态物体位置（分离，局部坐标） |
| `means_rigid_world` | `[N_rigid, 3]` | 变换到source帧的动态物体位置（分离，世界坐标） |
| `means_all` | `[N_total, 3]` | 合并后的所有点位置（N_total = N_bg + N_rigid） |
| `anchor_rgb_bg` | `[N_bg, 3]` | 从SH DC分量转换的静态背景RGB |
| `anchor_rgb_rigid` | `[N_rigid, 3]` | 从SH DC分量转换的动态物体RGB |
| `anchor_rgb_all` | `[N_total, 3]` | 合并后的所有点RGB |
| `sparse_feat` | `[M, 3]` | 稀疏特征（RGB，M ≤ N_total） |
| `vol_dim` | `[3]` | 体积维度 `[X, Y, Z]` / `[W, H, D]`（permute 后） |
| `valid_coords` | `[M, 3]` | 有效体素坐标 |
| `feat_3d` | `[M, outdim]` | 稀疏卷积后的3D特征 |
| `dense_volume` | `[1, C, D, H, W]` | 密集化的3D特征体积 |
| `feat_3d_crop_bg` | `[N_bg, outdim]` | 静态背景点的3D特征 |
| `feat_3d_crop_rigid` | `[N_rigid, outdim]` | 动态物体点的3D特征（不可见点为零） |
| `rigid_visible_mask` | `[N_rigid]` | 动态物体可见性掩码（bool，可选） |
| `rigid_in_crop_mask` | `[N_rigid]` | 动态物体是否在crop内的掩码（bool，可选） |
| `feat_2d_bg` | `[N_bg, C_2d]` | 静态背景点的2D特征（可选） |
| `feat_2d_rigid` | `[N_rigid, C_2d]` | 动态物体点的2D特征（可选，不可见点为零） |
| `feat_2d_distant` | `[N_distant, C_2d]` | 背景远景点的2D特征（可选） |
| `feat_bg_input` | `[N_bg, C_input]` | 静态背景点的融合特征（如果启用2D特征则为融合特征，否则为3D特征） |
| `feat_rigid_input` | `[N_rigid, C_input]` | 动态物体点的融合特征 |
| `feat_distant_input` | `[N_distant, C_input]` | 背景远景点的融合特征（如果存在） |

#### 偏移量预测阶段

| 变量名 | 形状 | 说明 |
|--------|------|------|
| `grid_coords_bg` | `[N_bg, 3]` | 静态背景点的归一化网格坐标 `[-1, 1]` |
| `grid_coords_rigid` | `[N_rigid, 3]` | 动态物体点的归一化网格坐标 `[-1, 1]` |
| `feat_3d_crop_bg` | `[N_bg, outdim]` | 静态背景点插值得到的3D特征（默认outdim=32） |
| `feat_3d_crop_rigid` | `[N_rigid, outdim]` | 动态物体点插值得到的3D特征（默认outdim=32） |
| `offsets_bg` | Dict | 静态背景的偏移量字典 |
| `offsets_rigid_world` | Dict | 动态物体的偏移量字典（世界坐标，已应用可见性掩码） |
| `offsets_rigid_local` | Dict | 动态物体的偏移量字典（局部坐标，从世界坐标变换而来） |
| `offsets_distant` | Dict | 背景远景的偏移量字典（如果存在） |
| `offset_pos` | `[N, 3]` | 位置偏移（`offset_max * tanh(mlp_output)`，范围`[-offset_max, offset_max]`） |
| `offset_scales_raw` | `[N, 3]` | 尺度偏移原始输出（从`mlp_conv`的前3维） |
| `offset_scales` | `[N, 3]` | 尺度对数偏移（`scale_max * tanh(offset_scales_raw)`，范围`[-scale_max, scale_max]`） |
| `offset_omega_raw` | `[N, 3]` | 轴角偏移原始输出（从`mlp_conv`的后3维） |
| `offset_omega` | `[N, 3]` | 轴角偏移（`omega_max * tanh(offset_omega_raw)`，范围`[-omega_max, omega_max]`，单位：弧度） |
| `offset_quat` | `[N, 4]` | 四元数偏移（从轴角转换，wxyz格式，单位四元数） |
| `offset_opacity` | `[N, 1]` | 不透明度对数偏移（`opacity_max * tanh(mlp_output)`，范围`[-opacity_max, opacity_max]`） |
| `sh_raw` | `[N, 3*num_sh]` | SH系数原始输出（从`gaussion_decoder`） |
| `sh_dc_raw` | `[N, 3]` | SH DC原始输出（`sh_raw`的前3维） |
| `sh_rest_raw` | `[N, 3*(num_sh-1)]` | SH rest原始输出（`sh_raw`的后`3*(num_sh-1)`维） |
| `offset_sh_dc` | `[N, 3]` | SH DC偏移（`sh_dc_max * tanh(sh_dc_raw)`，范围`[-sh_dc_max, sh_dc_max]`） |
| `offset_sh_rest` | `[N, 3*(num_sh-1)]` | SH rest偏移（`sh_rest_max * tanh(sh_rest_raw)`，范围`[-sh_rest_max, sh_rest_max]`） |
| `offset_sh` | `[N, 3*num_sh]` | SH系数偏移（扁平化，`concat([offset_sh_dc, offset_sh_rest], dim=-1)`） |

#### 渲染参数阶段

| 变量名 | 形状 | 说明 |
|--------|------|------|
| `render_params_bg` | Dict | 静态背景的渲染参数字典（世界坐标） |
| `render_params_rigid` | Dict | 动态物体的渲染参数字典（局部坐标） |
| `render_params_distant` | Dict | 背景远景的渲染参数字典（世界坐标，如果存在） |
| `means_r_bg` | `[N_bg, 3]` | 静态背景渲染用的位置（世界坐标，可微，不在此处clamp） |
| `means_r_rigid` | `[N_rigid, 3]` | 动态物体渲染用的位置（局部坐标，可微） |
| `means_r` | `[N, 3]` | 渲染用的位置（`node_state.means + eta_means * offset_pos`，可微，不在此处clamp） |
| `scales_log_r` | `[N, 3]` | 渲染用的尺度对数（`node_state.scales_log + eta_scales * offset_scales`，可微） |
| `scales_r` | `[N, 3]` | 渲染用的尺度（`exp(scales_log_r)`，转换到线性域） |
| `quats_r` | `[N, 4]` | 渲染用的四元数（`normalize(quat_multiply(node_state.quats, offset_quat))`，归一化，可微） |
| `opacity_logit_r` | `[N, 1]` | 渲染用的不透明度对数（`node_state.opacity_logit + eta_opacity * offset_opacity`，可微） |
| `opacities_r` | `[N]` | 渲染用的不透明度（`sigmoid(opacity_logit_r).squeeze(-1)`，范围[0,1]） |
| `offset_sh_dc` | `[N, 3]` | SH DC偏移（从`offset_sh[:, :3]`提取） |
| `sh_rest_flat` | `[N, 3*(num_sh-1)]` | SH rest偏移（扁平化，从`offset_sh[:, 3:]`提取） |
| `sh_rest_offset` | `[N, num_sh-1, 3]` | SH rest偏移（重塑为`[N, num_sh-1, 3]`） |
| `sh_dc_r` | `[N, 3]` | 渲染用的SH DC分量（`node_state.sh_dc + eta_sh_dc * offset_sh_dc`，可微） |
| `sh_rest_r` | `[N, num_sh-1, 3]` | 渲染用的SH高阶分量（`node_state.sh_rest + eta_sh_rest * sh_rest_offset`，可微） |
| `colors_r` | `[N, num_sh, 3]` | 完整的SH系数（`cat([sh_dc_r[:, None, :], sh_rest_r], dim=1)`，用于渲染） |

#### 代理参数阶段

| 变量名 | 形状 | 说明 |
|--------|------|------|
| `proxies_bg` | Dict | 静态背景的代理参数字典（世界坐标，所有target帧共享） |
| `proxies_rigid` | Dict | 动态物体的代理参数字典（局部坐标，所有target帧共享） |
| `proxies_distant` | Dict | 背景远景的代理参数字典（世界坐标，所有target帧共享，如果存在） |
| `means_p_bg` | `[N_bg, 3]` | 静态背景代理位置（分离但可微，世界坐标） |
| `means_p_rigid` | `[N_rigid, 3]` | 动态物体代理位置（分离但可微，局部坐标） |
| `means_rigid_world` | `[N_rigid, 3]` | 变换到target帧的动态物体位置（可微，世界坐标） |
| `quats_rigid_world` | `[N_rigid, 4]` | 变换到target帧的动态物体旋转（可微，世界坐标） |
| `merged_means` | `[N_total, 3]` | 合并后的位置（`cat([means_p_bg, means_rigid_world, means_p_distant])`） |
| `merged_quats` | `[N_total, 4]` | 合并后的旋转（`cat([quats_p_bg, quats_rigid_world, quats_p_distant])`） |
| `merged_scales` | `[N_total, 3]` | 合并后的尺度（`cat([scales_p_bg, scales_p_rigid, scales_p_distant])`） |
| `merged_opacities` | `[N_total]` | 合并后的不透明度（`cat([opacities_p_bg, opacities_rigid, opacities_p_distant])`，动态物体已应用可见性掩码） |
| `merged_colors` | `[N_total, num_sh, 3]` | 合并后的颜色（`cat([colors_p_bg, colors_p_rigid, colors_p_distant])`） |
| `means_p` | `[N, 3]` | 代理位置（分离但可微） |
| `scales_p` | `[N, 3]` | 代理尺度（分离但可微） |
| `quats_p` | `[N, 4]` | 代理四元数（分离但可微） |
| `opacities_p` | `[N]` | 代理不透明度（分离但可微） |
| `colors_p` | `[N, num_sh, 3]` | 代理颜色（分离但可微） |

#### 渲染输出阶段

| 变量名 | 形状 | 说明 |
|--------|------|------|
| `render` | `[1, H, W, 4]` | 渲染结果（RGB + alpha） |
| `alpha` | `[1, H, W]` | 累积不透明度 |
| `rgb` | `[H, W, 3]` | RGB图像 |
| `loss` | `scalar` | 单个视角的损失 |
| `total_loss_val` | `float` | 所有视角的累积损失（标量） |
| `test_metrics` | `Optional[Dict[str, float]]` | 测试视图评估指标（如果进行了评估） |

---

### 4. 评估指标 (`models/streetforward/metrics.py`)

**评估流程：** `_evaluate_test_views` 构造 `render_fn = lambda v, h, w: self._render_single_view(render_params, v, h, w)`，调用 `metrics.evaluate_test_views(render_fn, test_views, test_images, device, lpips_model)`，通过回调避免 metrics 依赖 trainer 类型。

#### PSNR (Peak Signal-to-Noise Ratio)

**实现：** `metrics.compute_psnr(pred, gt)`

**计算方法：**
```python
mse = torch.mean((pred - gt) ** 2)
psnr = -10 * torch.log10(mse)  # 如果 mse <= 0，返回 inf
```

**含义：** 衡量预测图像和真实图像之间的像素级差异，值越高越好（通常范围：20-40 dB）

#### SSIM (Structural Similarity Index)

**实现：** `metrics.compute_ssim(pred, gt)`，使用 `pytorch_msssim` 库

**含义：** 衡量预测图像和真实图像之间的结构相似性，值越高越好（范围：0-1）

**注意：** 如果库不可用，返回 `NaN`

#### LPIPS (Learned Perceptual Image Patch Similarity)

**实现：** `metrics.compute_lpips(pred, gt, lpips_model, device)`，使用 `lpips` 库，AlexNet 作为特征提取器

**含义：** 基于感知的相似性度量，值越低越好（通常范围：0-1）

**注意：** 如果库不可用，返回 `NaN`

---

## 关键组件说明

### 1. 偏移量预测网络 (Offset Prediction Networks)

**作用：** 从3D特征预测Gaussian参数的偏移量

**网络结构：**

#### `mlp_offset_pos` - 位置偏移预测
- **结构：** `outdim → 64 → 32 → 3`
- **输出：** 位置偏移原始值 `[N, 3]`
- **处理：** `offset_pos = offset_max * tanh(mlp_output)`
- **初始化：** 最后一层初始化为零（输出接近零偏移）

#### `mlp_conv` - 尺度与旋转偏移预测
- **结构：** `outdim → 64 → 32 → 6`
- **输出：** 尺度偏移（前3维）+ 轴角偏移（后3维）
- **处理：**
  - `offset_scales = scale_max * tanh(scales_raw)`
  - `offset_omega = omega_max * tanh(omega_raw)`
  - `offset_quat = _axis_angle_to_quat(offset_omega)`
- **初始化：** 最后一层初始化为零（输出接近零偏移）

#### `mlp_opacity` - 不透明度偏移预测
- **结构：** `outdim → 64 → 32 → 1`
- **输出：** 不透明度对数偏移原始值 `[N, 1]`
- **处理：** `offset_opacity = opacity_max * tanh(mlp_output)`
- **初始化：** 最后一层初始化为零（输出接近零偏移）

#### `gaussion_decoder` - SH系数偏移预测
- **结构：** `outdim → 64 → 32 → 3*num_sh`
- **输出：** SH系数偏移原始值（包含DC和rest）
- **处理：**
  - `offset_sh_dc = sh_dc_max * tanh(sh_dc_raw)`
  - `offset_sh_rest = sh_rest_max * tanh(sh_rest_raw)`
  - `offset_sh = concat([offset_sh_dc, offset_sh_rest])`
- **初始化：** 最后一层初始化为零（输出接近零偏移）

**共同特性：**
- 所有网络使用 `ReLU` 激活函数（中间层）
- 所有网络的最后一层**无激活函数**（直接输出原始值）
- 所有偏移量通过 `tanh` 函数限制在指定范围内
- 初始化策略确保训练开始时预测接近零偏移，避免初始阶段的大幅跳跃

### 2. 稀疏卷积网络 (SparseConv)

**作用：** 从稀疏点云特征构建3D特征表示

**实现：**
- **实现：** `models.evol_splat.SparseCostRegNet`
- **错误处理：** 如果 `SparseCostRegNet` 不可用且未提供自定义 `sparse_conv`，会抛出 `ImportError`

**输入/输出：**
- 输入：`[N, 3]` - RGB特征
- 输出：`[N, outdim]` - 3D特征（默认outdim=32）

### 3. 2D 特征提取与融合（可选）

#### 3.1. 图像特征提取器 (ImageFeatureExtractor)

**作用：** 从源视图图像中提取2D特征

**实现：**
- **实现：** `models.feature_extractors.ImageFeatureExtractor`
- **输入：** `[V, H, W, 6]` - 多通道输入（RGB + 渲染RGB）
- **输出：** `[V, H_feat, W_feat, C]` - 2D图像特征

**条件：** `use_2d_features == True` 且提供了 `source_views` 和 `src_images`

#### 3.2. Alpha-T 权重提取器 (AlphaTWeightExtractor)

**作用：** 执行双轮渲染：先渲染RGB供CNN使用，再流式渲染提取权重并反投影

**实现：**
- **实现：** `models.feature_extractors.AlphaTWeightExtractor`
- **功能：**
  1. `render_rgb_only()`: 渲染RGB图像（供CNN使用）
  2. `render_and_backproject_streaming()`: 流式渲染并反投影2D特征到3D点

#### 3.3. 特征反投影器 (FeatureBackprojector)

**作用：** 将2D图像特征反投影到3D点

**实现：**
- **实现：** `models.feature_extractors.FeatureBackprojector`
- **输入：** 2D特征和渲染权重
- **输出：** `[N, C_2d]` - 每个3D点的2D特征

#### 3.4. 特征融合模块 (FeatureFusion)

**作用：** 融合3D特征和2D特征

**实现：**
- **实现：** `models.feature_extractors.FeatureFusion`
- **输入：**
  - `feat_3d`: `[N, C_3d]` - 3D特征
  - `feat_2d`: `[N, C_2d]` - 2D特征
  - `visibility`: `[N]` - 可见性权重
- **输出：** `[N, C_fused]` - 融合后的特征

**融合策略：**
- 通常是门控融合或MLP融合
- 使用 `visibility` 加权2D特征的贡献
- 如果 `feat_2d` 为 `None`，直接返回 `feat_3d`

### 3. 体积构建函数

#### `construct_sparse_tensor`
- **实现：** `models.evol_splat.construct_sparse_tensor`
- **功能：** 将点云坐标和特征转换为稀疏张量格式
- **返回：** `(sparse_feat, vol_dim, valid_coords)`
- **错误处理：** 如果函数不可用且未提供自定义 `construct_sparse_tensor_fn`，会抛出 `ImportError`

#### `sparse_to_dense_volume`
- **实现：** `models.evol_splat.sparse_to_dense_volume`
- **功能：** 将稀疏特征转换为密集体积
- **输入：** `[N, C]` 稀疏特征
- **输出：** `[D, H, W, C]` 密集体积
- **错误处理：** 如果函数不可用且未提供自定义 `sparse_to_dense_volume_fn`，会抛出 `ImportError`

### 4.5. 参数合并 (ProxyRenderingMixin)

#### `_merge_all_params`
- **功能：** 合并 bg、rigid、distant 的完整代理参数（rigid 为完整 `means_rigid_world` 等）
- **适用：** 无 rigid 或 rigid 无可见点时

#### `_merge_params_with_rigid_subset`
- **功能：** 合并 bg、rigid 子集、distant，支持只渲染 `idx_tgt_rigid[view_idx]` 指定的可见 rigid 点
- **输入：** `proxies_bg`, `proxies_distant`, `rigid_subset`（含 means, quats, scales, opacities, colors）
- **适用：** 有可见 rigid 点时

### 5. 渲染器 (Renderer)

**实现：**
- **实现：** `gsplat.rendering.rasterization`
- **错误处理：** 如果 `gsplat` 不可用且未提供自定义 `renderer`，会抛出 `ImportError`

**输入参数：**
- `means`: `[N, 3]` - Gaussian中心
- `quats`: `[N, 4]` - 旋转四元数（wxyz）
- `scales`: `[N, 3]` - 尺度
- `opacities`: `[N]` - 不透明度
- `colors`: `[N, num_sh, 3]` - SH系数
- `viewmats`: `[1, 4, 4]` - 视图矩阵（世界到相机）
- `Ks`: `[1, 3, 3]` - 相机内参

**输出：**
- `render`: `[1, H, W, 4]` - RGB + alpha
- `alpha`: `[1, H, W]` - 累积不透明度

### 6. 辅助函数

#### 四元数操作
- `_random_quat_tensor()`: 生成随机单位四元数（wxyz格式）
- `_quat_multiply()`: 四元数乘法（用于组合旋转）
- `_normalize_quat()`: 四元数归一化（确保单位四元数）
- `_axis_angle_to_quat()`: 轴角到四元数转换（使用无分支sinc结构，提供平滑梯度）

#### 球谐函数转换
- `_rgb_to_sh()`: RGB → SH DC分量
- `_sh_to_rgb()`: SH DC分量 → RGB
- `_num_sh_bases()`: 计算SH基函数数量

#### 坐标转换
- `get_viewmat()`: 相机到世界 → 世界到相机（视图矩阵）
- `get_grid_coords()`: 世界坐标 → 体积网格归一化坐标（包含clamp操作，确保坐标在边界框内）
- `_transform_rigid_to_world()`: 将动态物体从局部坐标变换到世界坐标
- `_transform_rigid_quats_to_world()`: 将动态物体旋转从局部坐标变换到世界坐标
- `_transform_offsets_world_to_local()`: 将偏移量从世界坐标变换到局部坐标

#### 距离计算
- `_pairwise_neighbor_distances()`: 使用sklearn的k-NN计算邻居距离（内存高效）

#### 可见性处理
- `_mask_rigid_offsets()`: 使用可见性掩码屏蔽不可见动态物体的偏移量
- `_resolve_rigid_frame_idx()`: 解析动态物体的帧索引（处理帧ID映射）

---

## 梯度反向传播机制

### 代理参数机制

**设计目的：** 实现多视角梯度累积，同时避免重复构建计算图

**工作流程：**

```
1. 创建代理参数（从渲染参数分离，但启用梯度）
   render_params → proxies (detach + requires_grad=True)

2. 多视角渲染与梯度累积
   for view in target_views:
       render(proxies) → loss → loss.backward()
       # 梯度累积到 proxies.grad

3. 反向传播到渲染参数
   autograd.backward(render_tensors, proxy_grads)
   # 将代理梯度传播到渲染参数

4. 自动反向传播到网络参数
   render_params ← offsets ← MLPs ← sparse_conv
   # PyTorch自动计算梯度链
```

### 梯度流图

```
gt_image
  ↓
loss (L2)
  ↓
rgb (renderer输出)
  ↓
proxies (代理参数)
  ├─ means_p.grad
  ├─ scales_p.grad
  ├─ quats_p.grad
  ├─ opacities_p.grad
  └─ colors_p.grad
  ↓ (autograd.backward)
render_params (渲染参数)
  ├─ means_r
  ├─ scales_r
  ├─ quats_r
  ├─ opacities_r
  └─ colors_r
  ↓ (自动反向传播)
offsets (偏移量)
  ├─ offset_pos ← mlp_offset_pos
  ├─ offset_scales ← mlp_conv
  ├─ offset_quat ← mlp_conv
  ├─ offset_opacity ← mlp_opacity
  └─ offset_sh ← gaussion_decoder
  ↓
feat_3d_crop (3D特征)
  ↓
dense_volume
  ↓
feat_3d ← sparse_conv
  ↓
sparse_feat
  ↓
网络参数更新
```

### 关键设计点

1. **NodeState 分离：** NodeState 始终保持分离状态，不参与梯度计算，作为稳定的参数缓冲区

2. **代理参数桥接：** 代理参数作为 NodeState 和可微计算图之间的桥梁，实现梯度传递

3. **单次反向传播：** 每个 inner_iteration 只进行一次完整的反向传播，避免内存累积

4. **梯度累积：** 多个视角的梯度在代理参数上累积，然后一次性反向传播

---

## 配置参数

### Model 配置

```python
model:
  # 偏移量的物理上限（通常固定）
  offset_max: 0.1          # 位置偏移上限（米）
  scale_max: 0.1           # 尺度偏移上限（对数域）
  omega_max: 0.1           # 旋转偏移上限（弧度，约5.7°）
  opacity_max: 0.1         # 不透明度偏移上限（logit域）
  sh_dc_max: 0.1           # SH DC偏移上限
  sh_rest_max: 0.05        # SH rest偏移上限（通常更小）
  
  # 步长因子（控制偏移量幅度，通常固定）
  eta_means: 1.0           # 位置步长因子
  eta_scales: 1.0          # 尺度步长因子
  eta_opacity: 1.0         # 不透明度步长因子
  eta_sh_dc: 1.0           # SH DC步长因子
  eta_sh_rest: 1.0         # SH rest步长因子
  
  # 其他模型参数
  sh_degree: 1             # 球谐函数度数
  voxel_size: 0.1          # 体素大小
  max_iterations: 1        # 内部迭代次数
  bbx_min: [-20.0, -20.0, -20.0]  # 边界框最小值
  bbx_max: [20.0, 4.8, 70.0]      # 边界框最大值
  sparseConv_outdim: 32    # 稀疏卷积输出维度
  use_2d_features: False  # 是否启用2D特征融合
  feat_2d_channels: 16    # 2D CNN 输出通道数（启用 use_2d_features 时生效）
  feat_2d_downscale: 1    # 2D 特征下采样倍率（启用 use_2d_features 时生效）

  # GRU-style offsets（训练主路径）
  param_embed_dim: 32              # 参数 embedding 维度（默认等于 fused_in_dim）
  offset_gru_hidden_dim: 32        # GRU hidden 维度（默认等于 fused_in_dim）
  offset_gru_use_reset_gate: True  # 是否启用 reset gate（类似标准 GRU）

  input_aabb_min: [...]    # 输入AABB最小值（用于背景远景）
  input_aabb_max: [...]    # 输入AABB最大值（用于背景远景）
```

### Checkpoint（与流程相关的补充）

- **会保存的额外状态（当前实现）**：
  - `h_cache_bg / h_cache_rigid / h_cache_distant`：GRU-style offsets 的隐藏状态缓存
  - `mlp_params_embed / param_embed_norm / gru_update / gru_candidate / gru_reset / gru_to_head`：GRU-style 相关模块（用于从 checkpoint 恢复训练连续性）

### Optimizer 配置

```python
optimizer:
  lr: 1e-3                       # 学习率
  eps: 1e-15                     # Adam epsilon
  weight_decay: 0.0              # 权重衰减
```

### 其他配置

```python
log_images: False                # 是否保存渲染图像（节省GPU内存）
```

---

## 总结

StreetForwardTrainer 通过以下关键机制实现了高效的前馈式 3DGS 训练：

1. **分离的 NodeState：** 作为稳定的参数缓冲区，避免梯度干扰（支持 Background、Rigid 和 Distant 三种类型）
2. **3D 特征体积：** 通过稀疏卷积构建空间特征表示
3. **2D/3D 特征融合：** 可选地从源视图提取2D特征，与3D特征融合以增强表示能力
4. **偏移量预测：** 使用 GRU-style hidden fusion（特征 + 参数 embedding）生成 offsets，偏移头仍复用 MLP；并对 rigid 通过 `mask_update_rigid` 做 gate 以保证仅受监督点更新
5. **代理参数机制：** 实现多视角梯度累积
6. **单次反向传播：** 每个迭代只进行一次完整的梯度更新
7. **内存优化：** 密集体积在使用后立即删除，避免内存累积

**Minimal Stage 3.3** 在精简 bg+distant 管线上将分支配置（`model.branches`）、distant 的 2D-only 特征与独立偏移头、以及依赖 `sky_mask` 的复合损失单独成节说明；见上文 [Minimal Stage 3.3（bg/distant 解耦）](#minimal-stage-3-3-bg-distant)。

这种设计既保证了训练效率，又实现了多视角监督的有效利用，同时支持静态背景、动态物体和背景远景的联合训练。

---

## 天空分支（Stage 3.1）

本节将 [StreetForward_Sky_Model_Design.md](StreetForward_Sky_Model_Design.md) 的核心要点并入本文档，用于说明 **Stage 3.1（天空）** 如何与 StreetForward 的渲染与训练流程对接。

### 1. 输入与数据契约（viewdirs / sky_mask）

- **viewdirs**：
  - 来源：`datasets/base/pixel_source.py:get_rays()`（MultiSceneDataset 在组 batch 时从 `image_infos['viewdirs']` 收集）。
  - 形状：每张图像提供 `[H, W, 3]`（batch 中常为 `[V, H, W, 3]`）。
  - 语义：世界坐标系（seg0），单位向量。
  - **强约束**：viewdirs 必须与对应 `gt_image`/渲染分辨率一致；若分辨率不一致，应在数据侧/转换阶段用 `get_rays` 重算，而不是在 trainer 内插值 resize。

- **sky_mask（可选）**：
  - 形状：`[H, W]`（batch `[V, H, W]`）。
  - 约定：**`1=天空`，`0=非天空`**（float 0/1）；由 `MultiSceneDataset` 根据 `data.sky_mask_semantics` 从 loader 归一化。
  - 用途：可用于 loss 加权（例如仅天空区域更强监督 sky）；non-sky 区域权重为 `1 - sky_mask`。

### 2. 天空模型接口与坐标系约定

- **接口**：
  - 输入：`image_infos = {'viewdirs': viewdirs}`，其中 `viewdirs` shape 可为 `(H,W,3)` 或 batched `(B,H,W,3)`。
  - 输出：`rgb_sky`，shape 与 viewdirs 前缀一致，RGB 建议值域 `[0,1]`（例如 sigmoid）。

- **坐标系**（与 `get_rays` / MultiSceneDataset 对齐）：
  - 世界系：`+X=右，-Y=上，+Z=前`；viewdirs 为该系单位向量。
  - 若使用 cubemap + nvdiffrast `dr.texture`：需按 OpenGL cubemap convention 采样方向。
    - 常用变换：`to_opengl: (x,y,z) -> (x, z, -y)`（与 EnvLight 一致），再进行 cubemap 采样。

### 3. 合成公式与训练（单次 backward）

Stage 3.1 的核心是把天空作为“未被高斯遮挡区域”的补全项：

- 高斯渲染得到：`rgb_gaussians` 与 `opacity`（累积不透明度）
- 天空渲染得到：`rgb_sky`（仅依赖 viewdirs）
- 合成：
  - `rgb_composite = rgb_gaussians + rgb_sky * (1 - opacity)`

训练时对 `rgb_composite` 与 `gt_image` 计算 loss（如 L1/L2），并执行一次 `loss.backward()`，梯度同时更新：

- 高斯分支（proxy → render_params → offsets → 网络参数）
- 天空分支（sky_model 参数）

### 4. 多视角（multi-view）实现建议

当一次迭代包含多个 target views 时，建议将 sky 采样 **batch 化**：

- stack target viewdirs 为 `(T, H, W, 3)`
- 一次 sky forward 得到 `(T, H, W, 3)`
- 与 `(T, H, W, 3)` 的 `pred_rgbs` / `(T, H, W)` 的 `opacity` 一次性合成

这样可减少重复的 `dr.texture` launch 与 Python 循环开销。

---

<a id="minimal-stage-3-3-bg-distant"></a>

## Minimal Stage 3.3（bg/distant 解耦）

本节说明 **Minimal** 管线中与主 `StreetForwardTrainer` 不同的 Stage 3.3 机制：在 Stage 3.2 的 GRU-style 偏移、proxy 多视角渲染与天空合成之上，对 **背景（bg）** 与 **远景（distant）** 做配置与预测头分离；更完整的设计动机与分阶段计划见 [StreetForward_Stage3_3_Design.md](StreetForward_Stage3_3_Design.md)。

### 1. 与主文档的关系

- **主文档上文**：描述完整 `StreetForwardTrainer`（含 rigid、inner_iteration、双阶段 backward 等）。
- **Stage 3.3 Minimal**：仅含 **bg + distant**（无 rigid 分支），`forward` 路径继承 `MinimalStreetForwardStage3_2`（含 `train_step`、天空 `_composite_sky*` 等），在特征与 offsets 上对 distant 单独处理。
- **配置**：示例见仓库内 `configs/minimal_streetforward_stage3_3.yaml`。

### 2. 配置：`model.branches.{bg,distant}`（fast-fail）

初始化时 **必须** 同时存在 `model.branches.bg` 与 `model.branches.distant`，且每个分支必须包含子键 `init`、`limits`、`eta`、`mlp`、`freeze_means`；缺失直接报错，不做静默默认补齐。

每个分支结构要点：

| 块 | 含义 |
|----|------|
| `freeze_means` | 若为 true，渲染位置上功能冻结：`means_r = means + eta * offset_pos` 中的 offset 项乘零，仍保留计算图连接以便 proxy 反传。 |
| `init` | `opacity_init`；`scale_init.mode` 为 `isotropic`（`isotropic_log_value` 填 `scales_log`）或 `knn`（`knn_k`、`knn_log_scale_bias`）；**bg/distant 的 quat 均固定为单位四元数**，无配置项。 |
| `limits` | 对应各偏移的 `tanh` 上限（原单的 `offset_max`、`scale_max`、`omega_max` 等），按分支分别读取。 |
| `eta` | 各 Gaussian 参数相对 NodeState 的步长（`means` / `scales` / `opacity` / `sh_dc` / `sh_rest`）。 |
| `mlp` | `hidden_dim`、`use_3d_feat`、`use_2d_feat`、`freeze_quat`（distant 常为 true：旋转偏移在实现上等效为轴角乘零再转四元数）。 |

**与继承代码的衔接**：Stage 3.3 将 **bg** 的 `limits` / `eta` 写回 `self.offset_max`、`self.eta_means` 等属性，供父类里仍按「单套标量」工作的共享逻辑默认使用 bg 分支数值。

### 3. NodeState 初始化与分割

- 方法：`_get_or_init_node_states_bg_distant(batch)`。
- 点云来自 `batch["pointcloud"]`（字典时取 `background` 的 `xyz + rgb`）。
- 用 **`segment_aabb`（与 trainer 内 `bbx_min` / `bbx_max` 一致）** 划分：
  - **框内** → `NodeStateBackground`（`branches.bg.init` 初始化尺度与 opacity）。
  - **框外** → 若有剩余点则 `NodeStateDistant`（`branches.distant.init`）；否则 distant 为 `None`。
- 框内无点则直接报错（fast-fail）。

### 4. 特征与 offsets 前向（bg vs distant）

**背景（bg）**

1. `_build_3d_features(means_bg, anchor_rgb_bg)` 得到 3D 点特征。
2. `_prepare_gaussians_bg_distant` + `_compute_2d_features_bg_distant` 得到 `feat_2d_bg`（及 distant 的 2D）。
3. `feat_bg_input = _fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)`（与 Stage 3.2 一致的 3D+2D 融合）。
4. `_predict_offsets_gru(feat_bg_input, params_bg, h_old_bg, mask_update_rigid=None)` → **共用** Stage 3.2 的偏移 MLP 头。
5. `_render_params_from_offsets_bg(node_state_bg, offsets)`，**eta 全部来自 `branches.bg.eta`**。

**远景（distant）**

- 设计前提：distant 不在可靠 3D 体积支撑内，**不走 3D 特征拼接**；仅当 `num_distant > 0` 且 `feat_2d_distant` 有效时参与更新。
- `feat_2d_distant` 经 **`distant_feat_proj`**（`Linear`: 2D 通道维 → `fused_in_dim`）与 GRU 输入维对齐。
- `_predict_offsets_gru_distant(feat_distant_input, params_distant, h_old_distant)`：
  - **GRU 子模块**（`mlp_params_embed`、`param_embed_norm`、`gru_*`、`gru_to_head`）与 bg **共用参数**；
  - **偏移头独立**：`mlp_offset_pos_distant`、`mlp_conv_distant`、`mlp_opacity_distant`、`gaussion_decoder_distant`，`tanh` 上限来自 **`branches.distant.limits`**。
- `_render_params_from_offsets_distant` 使用 **`branches.distant.eta`**。

空特征时 distant 返回零位移类 offset、`offset_quat` 为 identity，`h_new = h_old`（与主 trainer 空特征约定一致）。

### 5. 渲染、proxy 与损失

- 训练时：`proxies_bg`、可选 `proxies_distant` 经 `_merge_params_bg_distant` 合并后多视角渲染；天空合成逻辑继承 Stage 3.2。
- **损失（训练态）**：每个 target **必须** 提供 `sky_mask`（`1`=天空，`0`=非天空）；否则报错。在有效像素掩码上计算：
  - RGB：`loss_w_l1 * L1 + loss_w_ssim * SSIM`（masked）；
  - Mask：将累积不透明度与 `(1 - sky_mask)` 在有效区域上做 BCE 类监督（权重 `loss_w_mask`）；
  - 可选：`loss_w_opacity_entropy *` 不透明度熵（masked）。
- 返回字典除 `loss` 外包含 `loss_l1`、`loss_ssim`、`loss_rgb`、`loss_mask`、`loss_opacity_entropy` 等分项，便于日志与消融。

### 6. 训练脚本与 checkpoint

- **脚本**：`tools/train_minimal_streetforward_stage3_3.py` — 默认配置 `configs/minimal_streetforward_stage3_3.yaml`，依赖 overfit batch（`--overfit_batch_path` 或 config），`convert_batch_to_minimal_format(..., include_source_for_2d=True)`，视图选择逻辑与 Stage 3.2 脚本一致。
- **Checkpoint**：`__init__` 在注册 distant 头与 `distant_feat_proj` 后 **重建 Adam**，状态 dict 含新增键；与 Stage 3.2 旧权重 **不保证 strict 兼容**，需按设计文档做部分 warm-start 或重训。

### 7. 配置示例（缩略）

完整字段以 `configs/minimal_streetforward_stage3_3.yaml` 为准；结构形如：

```yaml
model:
  branches:
    bg:
      freeze_means: false
      init:
        scale_init: { mode: isotropic, isotropic_log_value: -2.30, knn_k: 3, knn_log_scale_bias: 0.0 }
        opacity_init: 0.1
      limits: { offset_max: 0.1, scale_max: 0.1, omega_max: 0.1, opacity_max: 0.1, sh_dc_max: 0.1, sh_rest_max: 0.05 }
      eta: { means: 1.0, scales: 1.0, opacity: 1.0, sh_dc: 1.0, sh_rest: 1.0 }
      mlp: { hidden_dim: 64, use_3d_feat: true, use_2d_feat: true, freeze_quat: false }
    distant:
      freeze_means: true
      init: { ... }
      limits: { ... }
      eta: { ... }
      mlp: { hidden_dim: 64, use_3d_feat: false, use_2d_feat: true, freeze_quat: true }
```
