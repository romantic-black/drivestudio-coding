# 最简 StreetForward 模型设计计划

本文档讨论**最简 StreetForward 模型**的设计与实现计划，用于在 Overfit One Batch 场景下逐步排查现有完整模型的错误。设计参考 [Overfit One Batch 设计](dataloader/Overfit_One_Batch_Design.md)、[StreetForward 流程说明](StreetForward_Flow.md)，以及 [EVolSplat 模型](third_party/EVolSplat/nerfstudio/models/evolsplat.py) 的 3DGS 输出与渲染方式。

---

## 1. 目标与动机

- **目标**：在数据与模型都极简的前提下，先打通「点云 → 3D 特征 → 3DGS 参数 → 渲染 → 单 target 监督」的整条链路，再逐步加回 source、多 target、动态物体、2D 特征等，便于定位问题是在「最简链路」还是「扩展逻辑」。
- **与 Overfit One Batch 的关系**：沿用 Overfit One Batch 的 **Capture / Load Batch** 与 `convert_batch_to_streetforward_format`，但在最简模式下**只使用**其中的：单 target 视角、单 target 图像、以及 3D RGB 点云；不取 source。

---

## 2. 最简模型定义

### 2.1 输入

| 项目 | 说明 |
|------|------|
| **点云** | **仅 3D RGB 点云**：`[N, 6]`（x, y, z, r, g, b），世界坐标，单一段内静态点。不区分子类型（无 background / rigid / distant 拆分）。 |
| **Target** | **仅一张**：一个 target 视角（`View`） + 一张 GT 图像 `gt_image`。 |
| **Source** | **不取**：无 source 帧、无 source 图像、无 2D 特征。 |

即：**输入 = 点云 (N,6) + 单 target (view + gt_image)**。

### 2.2 中间：3D 特征部分

与现有 StreetForward 的 3D 分支一致，只做「点云 → 逐点 3D 特征」：

1. **稀疏体素化**  
   - 输入：点云坐标 `means [N, 3]`、点云颜色 `anchor_rgb [N, 3]`（由 RGB 归一化到 [0,1] 或与现有 `_sh_to_rgb` 一致）。  
   - `construct_sparse_tensor(means, anchor_rgb, bbx_min, bbx_max, voxel_size)` → `sparse_feat`, `vol_dim`, `valid_coords`。

2. **稀疏卷积**  
   - `feat_3d = sparse_conv(sparse_feat)` → 体素级特征 `[M, outdim]`。

3. **稀疏 → 密集 + 插值**  
   - `sparse_to_dense_volume` 得到 `dense_volume`；  
   - 对每个点的世界坐标用 `get_grid_coords` 得到归一化网格坐标，再 `interpolate_features` 得到 **逐点 3D 特征** `feat_3d_crop [N, outdim]`。

**输出**：每个点一个 3D 特征向量，无 2D、无 GRU、无 NodeState 更新，仅此一段前向。

### 2.3 Head：3DGS 参数预测与渲染

Head 负责从**逐点 3D 特征**直接得到 3DGS 渲染所需参数，设计上参考 EVolSplat 的 MLP 头与 gsplat 渲染接口。

#### 2.3.1 参数与 EVolSplat 对应关系（简要）

| 3DGS 参数 | EVolSplat 做法（参考） | 最简 StreetForward 建议 |
|----------|------------------------|---------------------------|
| **位置** | `means` 初始为点云坐标，用 `mlp_offset(feat_3d)` 得到 offset，`means_crop = means + offset_max * tanh(mlp_offset)` | 同：点云坐标 + 可学习/受限的 offset（或先固定 offset=0 做更简版）。 |
| **尺度** | k-NN 初始 `scales_log`，再 `mlp_conv` 出 scale 增量；最终 `scales = exp(scales_log + scale_crop)` | 同：点云初始化 `scales_log`，Head 只预测 scale 增量（或直接预测 scales_log）。 |
| **旋转** | `mlp_conv` 输出 4 维，归一化为四元数 `quats_crop` | 同：Head 输出 4 维 → 归一化四元数。 |
| **不透明度** | `mlp_opacity(feat_3d)` → `opacities = sigmoid(opacities_crop)` | 同：Head 输出 1 维 logit → sigmoid。 |
| **颜色** | `gaussion_decoder(concat(2d_feat, ob_dist, ob_view))` 输出 SH；EVolSplat 还用了 2D 特征 | 最简：**仅用 3D 特征**，如 `gaussion_decoder(feat_3d)` 或 `feat_3d + 可选 ob_dist/ob_view`，输出 SH DC + rest。 |

#### 2.3.2 最简 Head 结构建议

- **输入**：`feat_3d_crop [N, outdim]`
- **输出**（与现有 StreetForward / EVolSplat 对齐）：  
  - `offset_pos` → `means_r = means + eta * offset_pos`  
  - `scales_log_delta`（或 scales 直接）→ `scales_r = exp(scales_log_init + scales_log_delta)`  
  - `quats_r`：4 维 → 归一化四元数  
  - `opacity_logit_r` → `opacities_r = sigmoid(opacity_logit_r)`  
  - `sh_dc`, `sh_rest` → `colors_r`（SH 系数，供 gsplat 渲染）

- **渲染**：与 EVolSplat 一致，调用 `gsplat.rendering.rasterization`（或现有 StreetForward 的渲染封装），传入 `means_r`, `scales_r`, `quats_r`, `opacities_r`, `colors_r` 以及 target 的 `viewmat`、内参、高宽。

这样 Head 与 EVolSplat 的 3DGS 输出与渲染接口一致，便于复用和对比。

### 2.4 训练与数据流（最简）

- **前向**：  
  `点云(means, rgb)` → 3D 特征模块 → `feat_3d_crop` → Head → 3DGS 参数 → 对**唯一 target** 渲染 → `pred_rgb`。  
- **监督**：  
  `loss = L(pred_rgb, gt_image)`（如 L1/L2 + 可选 SSIM），**仅此一个 target，无 source，无多视角累积**。  
- **反向**：  
  单次 `loss.backward()`，梯度经渲染 → Head → 3D 特征（稀疏卷积 + 体素化/插值）。

无需：NodeState、代理参数、多 target 循环、GRU、2D 特征、Rigid/Distant。

---

## 3. 与 Overfit One Batch 的衔接

- **Batch 来源**：仍用 Overfit One Batch 的 **Capture** 得到 `batch.pt` + `meta.json`。  
- **加载与转换**：用 **Load Batch** + `convert_batch_to_streetforward_format(batch, device)` 得到 `View` 与张量。  
- **最简取数**：  
  - 点云：只取 `batch["pointcloud"]` 中**静态部分**（如 `pointcloud["background"]`），转成 `[N, 6]`（xyz + rgb）。  
  - Target：只取 **一个** target，例如 `targets[0]` 的 `view` 与 `gt_image`；不取 `source_frame_idx`、不取 source 视图。  
- **Config**：  
  - 可沿用 `overfit_one_batch_template.yaml`，但在最简模式下设置 `num_source_keyframes: 0`（或不读 source），`num_target_keyframes: 1`，且 dataloader/trainer 只喂一张 target。

这样 Overfit One Batch 的流程不变，只是「用哪几帧、用不用点云子集」的约束在最简模型中收紧。

---

## 4. 逐步排查策略

按「先最简，再加一点」的方式排查现有模型错误：

1. **Stage 0：最简链路**  
   - 输入：仅 3D RGB 点云 + 单 target。  
   - 中间：仅 3D 特征（体素化 → sparse_conv → 插值 → 逐点 feat）。  
   - Head：仅 MLP 头 → 3DGS 参数 → 单 view 渲染。  
   - 验证：Overfit 一个 batch 能否收敛（loss 下降、PSNR 提升）。若这里不通，问题在：点云格式、3D 特征、Head 或渲染接口。

2. **Stage 1：加 NodeState + 单 target**  
   - 点云 → 初始化 NodeStateBackground（仅静态），仍无 Rigid/Distant。  
   - 3D 特征仍只从点云/NodeState 位置与颜色来；Head 改为「NodeState + offset」形式（与现有 StreetForward 一致）。  
   - 仍单 target、无 source。  
   - 验证：与 Stage 0 行为是否一致、能否 overfit。

2.1. **Stage 1.1：加 GRU-style 偏移量预测**  
   - 在 Stage 1 基础上，用「特征 + 参数 embedding + GRU → offsets」替代「feat → 直接 offsets」（与 [StreetForward_Flow](StreetForward_Flow.md) §5.1.1 一致）。  
   - 维护 h_cache_bg，step 间 detach；无 Rigid，无 mask_update_rigid。  
   - 详见 [Minimal_StreetForward_Next_Steps_Stage1_1_GRU](Minimal_StreetForward_Next_Steps_Stage1_1_GRU.md)。

3. **Stage 2：加多 target（在 Stage 1.1 基础上）**  
   - **Stage 2.0**：仅多 target（如 3 个），同一套 render_params 多 view 渲染，loss 取均后单次 backward；无代理。  
   - **Stage 2.1**：多 target + 代理参数与多视角梯度累积（参考 StreetForward_Flow）；与 2.0 同数据对比以保证一致性。  
   - 仍无 source、无 2D 特征。  
   - 方案讨论见 [Minimal_StreetForward_Next_Steps_Stage2_MultiTarget](Minimal_StreetForward_Next_Steps_Stage2_MultiTarget.md)。

4. **Stage 3：加 source + 2D 特征（可选）**  
   - 加入 source 视图与 2D 特征提取、融合。  
   - 验证：与现有 StreetForward 的 2D 分支差异。

5. **Stage 4：加动态物体（Rigid/Distant）**  
   - 恢复 pointcloud 的 dynamic/distant、RigidMasks、GRU-style 等。  
   - 验证：动态与静态是否同时收敛、有无漏梯度或错误 mask。

每一步都可在同一 Overfit One Batch 上跑，便于对比与定位。

---

## 5. 实现要点小结

| 模块 | 最简实现 |
|------|----------|
| **输入** | 仅 3D RGB 点云 + 单 target (view + gt_image)；不取 source。 |
| **3D 特征** | 与现有 `_build_3d_feature_volume` 一致：体素化 → sparse_conv → dense → 插值 → `feat_3d_crop [N, outdim]`。 |
| **Head** | 参考 EVolSplat：由 `feat_3d_crop`（+ 可选 ob_dist/ob_view）预测 offset_pos、scales、quats、opacity、SH；输出 3DGS 参数并调用 gsplat 渲染。 |
| **训练** | 单 target 渲染 → 单 loss → 单次 backward；无 NodeState 更新、无代理、无 GRU、无 2D。 |
| **数据** | Overfit One Batch 的 batch；只取 pointcloud 静态部分 + 一个 target。 |

---

## 6. 参考文件

- [Overfit One Batch 设计](../dataloader/Overfit_One_Batch_Design.md)  
- [StreetForward 流程图与数据结构](StreetForward_Flow.md)  
- [EVolSplat 模型](../../third_party/EVolSplat/nerfstudio/models/evolsplat.py)（3D 特征、MLP 头、rasterization 调用）

本文档可作为「最简 StreetForward」的实现与排查 checklist，后续若增加配置项或脚本入口，可在此补充。

---

## 7. 实现入口（已实现）

| 项目 | 路径 |
|------|------|
| **模型** | `models/streetforward/minimal_trainer.py` — `MinimalStreetForward` |
| **训练脚本** | `tools/train_minimal_streetforward.py` |
| **配置** | `configs/minimal_streetforward.yaml` |

**运行示例**（需先 Capture 得到 overfit batch）：

```bash
# 1. Capture 一个 batch（若尚未有）
python tools/overfit_one_batch.py --config_file configs/overfit_one_batch_template.yaml

# 2. 最简训练（单 target，无 source）
python tools/train_minimal_streetforward.py --config_file configs/minimal_streetforward.yaml \
  overfit_batch_path=./data/overfit_batches/scene0_seg0_batch.pt
```

依赖与完整 StreetForward 一致：`gsplat`、`models.evol_splat`（含 `torchsparse`）等。
