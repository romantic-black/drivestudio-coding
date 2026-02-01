# StreetForward Golden Baseline 测试覆盖范围总结

本文档基于 `tools/record_streetforward_golden_baseline.py`、`utils/streetforward_baseline.py`、`tests/test_streetforward_golden_baseline.py`、`tests/test_streetforward_feature_alignment.py` 与 `docs/trainers/StreetForward_Flow.md`，总结当前测试的覆盖范围、**测试方式**，以及特征值对齐方案（含对所有特征进行值对齐的约定）。

---

## 1. 相关代码与文档索引

| 角色 | 路径 | 作用 |
|------|------|------|
| 录制脚本 | `tools/record_streetforward_golden_baseline.py` | 按固定 seed/config 跑 N 步 `train_iter`，输出 golden JSON（meta + per_step） |
| 工具库 | `utils/streetforward_baseline.py` | 配置/数据集/批次转换、batch 计划与 cache、BaselineStep 录制与 **compare_step** 比较 |
| Golden 回归测试 | `tests/test_streetforward_golden_baseline.py` | 加载 golden、按相同参数重放录制、逐步 **值比较** |
| 特征对齐单元测试 | `tests/test_streetforward_feature_alignment.py` | 点序与可见性对齐、3D/2D 输出形状与掩码（stub 依赖） |
| 流程与机制 | `docs/trainers/StreetForward_Flow.md` | Trainer 训练流程、3D 体积、NodeState、代理参数、梯度反传等 |
| 特征对齐方案 | `docs/trainers/StreetForward_Feature_Alignment.md` | 点序约定、2D/3D/其他特征索引对应及**值对齐**约定 |

---

## 2. 当前测试一览与测试方式

### 2.1 Golden Baseline 回归测试（端到端）

| 测试 | 文件 | 测试方式 | 说明 |
|------|------|----------|------|
| `test_streetforward_golden_baseline` | `test_streetforward_golden_baseline.py` | **端到端重放 + 逐步值比较** | 用 golden 的 meta 恢复 config/seed/device/scheduler/batch_cache，再跑一遍 `run_recording`，对每一步调用 `compare_step(baseline_step, current_step)`，断言步数一致且每步通过。依赖：cv2、真实 config/data_root、可选 batch cache、与 baseline 一致的 device。 |

**比较内容（当前）**：由 `compare_step` 完成，**仅对以下项做值对齐**：

- `scene_id`、`segment_id`（严格相等）
- `total_loss`（rtol/atol）
- `node_state_bg_summary`、`node_state_rigid_summary`、`node_state_distant_summary`（shape、标量统计 min/max/mean/std/norm；Rigid 含 point_ids_hist）
- `grad_norms`（各模块 L2 梯度范数，grad_rtol）

**未参与比较的录制内容**：`offset_bg_summary`、`offset_rigid_summary`、`offset_distant_summary` 已写入 JSON，但 **`compare_step` 未比较**。3D/2D/融合特征当前未录制、也未比较。

### 2.2 特征对齐单元测试（stub 依赖）

| 测试 | 文件 | 测试方式 | 说明 |
|------|------|----------|------|
| `test_prepare_all_gaussians_orders_bg_rigid_distant` | `test_streetforward_feature_alignment.py` | **单元测试**：真实 Trainer + 真实 NodeState，无真实渲染/稀疏卷积 | 调用 `_prepare_all_gaussians`，断言返回的 `num_bg/num_rigid/num_distant` 与点数和 `gaussians["means"]` 的合并顺序（bg → rigid → distant）一致。 |
| `test_compute_2d_features_all_respects_alignment_and_visibility` | 同上 | **单元测试**：Dummy 2D 提取器 + Dummy AlphaT（按 gaussian 索引填 0,1,2,...） | 调用 `_compute_2d_features_all`，断言 `feat_2d_bg/feat_2d_rigid/feat_2d_distant` 形状与拆分一致，且 rigid 的不可见点被置零（值对齐可见性掩码）。 |
| `test_build_3d_feature_volume_masks_invisible_and_out_of_crop` | 同上 | **单元测试**：Dummy 稀疏卷积/体积（常数特征） | 调用 `_build_3d_feature_volume`，断言 `rigid_visible_mask`、`rigid_in_crop_mask` 正确，且不在 crop 或不可见的 rigid 点对应 3D 特征为零。 |

上述单元测试通过 stub 替换 gsplat、torchsparse、真实 CNN，在无 GPU/无数据下验证**点序与可见性对齐**及 3D/2D 输出的**形状与掩码行为**，不验证具体数值与真实特征值。

### 2.3 测试方式小结

| 方式 | 含义 | 用例 |
|------|------|------|
| **端到端重放** | 相同 config/seed/batch 再跑一遍录制，对每步做数值比较 | `test_streetforward_golden_baseline` |
| **单元测试（stub）** | 用 dummy 替换重依赖，只测接口与逻辑（顺序、形状、掩码） | `test_prepare_all_gaussians_*`、`test_compute_2d_features_all_*`、`test_build_3d_feature_volume_*` |

---

## 3. StreetForward_Flow 机制覆盖情况

| Flow 机制 | 覆盖方式 |
|-----------|----------|
| 多 NodeState（Bg/Rigid/Distant） | Golden：每步比较 node_state_*_summary；单元：`_prepare_all_gaussians` 顺序 |
| 3D 特征体积与插值 | Golden：通过 node_state 与 grad_norms 间接；单元：`_build_3d_feature_volume` 形状与 visible/crop 掩码 |
| 2D 特征与反投影 | 单元：`_compute_2d_features_all` 形状与 rigid 可见性置零；Golden 未录 2D 特征值 |
| 融合与偏移预测 | Golden：grad_norms 覆盖 MLP；未比较 offset 汇总与融合特征值 |
| 梯度反传 | Golden：grad_norms 每步比较 |
| 调度与数据 | Golden：meta 中 scheduler_kwargs/batch_cache_path 保证同批数据 |

---

## 4. 对所有特征进行值对齐的方案

### 4.1 原则

- **值对齐**：在回归测试中，baseline 与 current 的**同一 step、同一点类型**下，所有参与训练的特征与派生量应在**数值上一致**（或落在约定容差内），以便发现实现或依赖变更导致的漂移。
- **特征范围**：包括 3D 特征（feat_3d_crop_*）、2D 特征（feat_2d_*）、融合特征（feat_*_input）、偏移量（offset_*）、以及现有的 NodeState 汇总与 grad_norms。

### 4.2 当前已做值对齐的项

- **Golden 每步**：`scene_id`、`segment_id`、`total_loss`、`node_state_bg_summary`、`node_state_rigid_summary`、`node_state_distant_summary`、`grad_norms`。
- **特征对齐单元测试**：通过 dummy 输出与掩码检查 2D/3D 的**形状与可见性**对齐，不比较具体标量值。

### 4.3 应对所有特征做值对齐的约定

1. **Offset 汇总**  
   - 录制中已有 `offset_bg_summary`、`offset_rigid_summary`、`offset_distant_summary`（与 node_state 同形式的统计：shape、min/max/mean/std/norm）。  
   - **约定**：在 `compare_step` 中增加对上述三项的比较（使用与 node_state 相同的 stat_rtol/stat_atol 或单独 offset 容差），使偏移量参与值对齐。

2. **3D / 2D / 融合特征（可选扩展）**  
   - 若需对“特征本身”做值对齐，可在录制时增加每步、每类型的**特征汇总**（如 `feat_3d_crop_bg_summary`、`feat_2d_bg_summary`、`feat_bg_input_summary` 等，与现有 summary 同格式），并在 `compare_step` 中比较。  
   - **约定**：新增特征流或融合方式时，若会影响 MLP 输入或梯度，应纳入录制与比较（至少以 summary 形式），保证所有特征参与值对齐。

3. **实现检查**  
   - `compare_step` 中已比较的：scene/segment、loss、node_state_*_summary、grad_norms。  
   - 待补：offset_*_summary。  
   - 可选：feat_3d/feat_2d/fused 的 summary（在录制端与比较端均支持后再启用）。

### 4.4 与索引对齐的关系

- **索引/点序对齐**（见 `StreetForward_Feature_Alignment.md`）：保证同一“点”在不同特征张量中对应同一索引，融合时按索引逐点组合。  
- **值对齐**：在索引对齐的前提下，在回归测试中比较这些特征的**数值**（或汇总统计），确保实现变更不会悄悄改变输出。  
- 二者结合：先满足点序与拆分一致，再在 golden 中对所有已录制特征（含 offset，及可选的 3D/2D/fused summary）做值对齐比较。

---

## 5. 已使用的工具链

- **配置**：`load_config(config_path)`，并合并 dataset preset（若存在）
- **数据**：Golden 测试中若 meta 提供 `batch_cache_path` 且文件存在则从 cache 读 batch，否则 `build_dataset` + scheduler
- **批次格式**：`convert_batch_to_streetforward_format` 将 DataLoader batch 转为 Trainer 输入格式
- **录制**：`record_step` 产出 `BaselineStep`（含 node_state/offset/grad_norms），`save_baseline` 写 JSON
- **比较**：`compare_step` 使用固定 rtol/atol 比较上述字段（待加入 offset_*_summary）

---

## 6. 显式跳过或未覆盖的场景

- 无 cv2、无 golden 文件、无 config、无 data_root、CUDA baseline 在仅 CPU 环境、需要 batch cache 但缺失 → Golden 整测 skip  
- 仅一种调度配置（meta 中的 scheduler_kwargs），未单独测 random/shuffle/preload  
- 未测：`apply_update=False` / `update_state=False`  
- 未测：`use_2d_features=True` 的 golden 重放（单元测已覆盖 2D 对齐与可见性）  
- 未测：`harvest_if_missing` 录制路径  
- 工具函数（`load_config`、`compare_step`、`convert_batch_to_streetforward_format` 等）无独立单元测试

---

## 7. 总结表

| 维度 | 当前状态 | 测试方式 | 建议 |
|------|----------|----------|------|
| 端到端确定性 | 已覆盖 | 重放 + compare_step | 保持 |
| NodeState / grad_norms 值对齐 | 已覆盖 | compare_step | 保持 |
| Offset 值对齐 | 已录制、未比较 | - | **在 compare_step 中增加 offset_*_summary** |
| 3D/2D/融合特征值对齐 | 未录制、未比较 | - | 可选：录 summary 并比较 |
| 点序与可见性对齐 | 已覆盖 | 单元测试（stub） | 保持 |
| 2D golden 重放 | 未覆盖 | - | 若正式支持 2D，可单独 golden |
| 工具函数单元测试 | 无 | - | 建议增加 compare_step、load_config 等 |

整体上，当前测试通过**端到端重放**与**特征对齐单元测试**覆盖了主训练路径与点序/可见性；通过对**所有特征进行值对齐**（先补 offset，再视需要补 3D/2D/fused summary），可进一步保证任意特征改动都可被回归发现。
