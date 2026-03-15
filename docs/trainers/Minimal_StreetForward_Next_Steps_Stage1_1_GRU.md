# Minimal StreetForward 下一步方案：Stage 1.1（GRU）

本文档在 Stage 0、Stage 1 已完成的前提下，讨论 **Stage 1.1**：在最小复现中接入与完整 StreetForward 一致的 **GRU-style 偏移量预测**（[StreetForward_Flow.md 5.1.1](StreetForward_Flow.md)），便于后续与多 target、2D 特征、Rigid 等扩展对齐。

参考：[Minimal_StreetForward_Design_Plan.md](Minimal_StreetForward_Design_Plan.md)、[StreetForward_Flow.md](StreetForward_Flow.md)、`models/streetforward/minimal_trainer_stage1.py`、`models/streetforward/offsets_mixin.py`、`models/streetforward/trainer.py`。

---

## 1. 当前状态与目标

| 阶段 | 状态 | 说明 |
|------|------|------|
| Stage 0 | 已完成 | 点云 → 3D 特征 → MLP Head → 3DGS 参数 → 单 target 渲染 |
| Stage 1 | 已完成 | NodeStateBackground + 单 target；Head = NodeState + offsets，可选 NodeState write-back |
| **Stage 1.1** | **待实现** | 在 Stage 1 基础上，用 **GRU-style** 替代「feat_3d_crop → 直接 _predict_offsets」 |
| Stage 2 | 未做 | 多 target + 代理参数 + 梯度累积 |
| Stage 3+ | 未做 | source + 2D 特征、Rigid/Distant 等 |

**Stage 1.1 目标**：实现与 [StreetForward_Flow.md §5.1.1](StreetForward_Flow.md) 一致的「特征 + 参数 embedding + GRU 更新 → offsets」，并维护 h 缓存；最简场景下**仅背景点、单 target**，无 Rigid gate。

---

## 2. GRU-style 逻辑回顾（StreetForward_Flow §5.1.1）

以下与 `StreetForward_Flow.md` 363–396 及 `offsets_mixin._predict_offsets_gru` 对齐。

### 2.1 输入

- `feat`: `[N, C_fused]` — 当前为 3D 特征 `feat_3d_crop`，即 `[N, outdim]`。
- `params_for_embed`: 来自 `_build_params_for_embed()`，包含 `means / quats / scales_log / opacity_logit / sh_dc / sh_rest`（NodeState 字段）。
- `h_old`: `[N, H]` — 来自 `h_cache_bg`，在每 train step 起始处 `detach()`，截断跨 step 梯度。
- `mask_update_rigid`：Stage 1.1 **不传**（仅背景，全部点更新）。

### 2.2 参数 embedding（17 维 param_vec）

`_normalize_params_for_embed(params_for_embed)` 得到 `param_vec [N, 17]`：

| 维度 | 内容 | 做法 |
|------|------|------|
| 3 | means_norm | 以 `bbx_min/bbx_max` 归一化到 `[-1, 1]` |
| 6 | rot6d | quaternion → 旋转矩阵前两列 |
| 3 | scales_norm | `scales_log` clamp(-10,10) 后 LayerNorm |
| 1 | opacity_norm | `tanh(opacity_logit)` |
| 3 | sh_dc | 原值 |
| 1 | sh_rest_energy | `\|sh_rest\|_2` |

实现见 `offsets_mixin._normalize_params_for_embed`；Stage 1.1 可直接复用或拷贝该逻辑（仅依赖 `bbx_min/bbx_max` 与 NodeState 字段）。

### 2.3 GRU 更新与 offsets

1. `param_embed = LayerNorm(MLP(param_vec))`，即 `param_embed_norm(mlp_params_embed(param_vec))`。
2. `x = concat([feat, param_embed])`，维度 `[N, C_fused + param_embed_dim]`。
3. `hx = concat([h_old, x])`，输入 GRU 门：
   - `z = sigmoid(gru_update(hx))`（update gate）
   - 若启用 reset gate：`r = sigmoid(gru_reset(hx))`，`h_cand = tanh(gru_candidate(concat(r*h_old, x)))`
   - 否则：`h_cand = tanh(gru_candidate(hx))`
   - `h_new = (1-z)*h_old + z*h_cand`
4. `head_input = gru_to_head(h_new)`（若 `hidden_dim != head_dim` 则线性投影到原 head 输入维）。
5. `offsets = _predict_offsets(head_input)`（与现有一致：同一组 mlp_offset_pos / mlp_conv / mlp_opacity / gaussion_decoder）。

Stage 1.1 无 Rigid：不执行「对 offsets 乘 gate、对 offset_quat 置 identity、h_new 按 gate 混合」；即始终 `mask_update_rigid=None`。

### 2.4 h 缓存与 reset

- 每个 step 开始时用 `h_old = h_cache_bg[key].detach()`（或未命中时零初始化）。
- Step 结束后 `h_cache_bg[key] = h_new.detach()`。
- 当 `(scene_id, segment_id)` 或点云数量变化时，应对 h 做 reset（与完整版 `_get_or_init_hidden` 一致：按 key + num_points + 可选 signature 判断是否清零）。

---

## 3. Stage 1.1 实现要点

### 3.1 新增/复用的模块

| 模块 | 说明 | 参考 |
|------|------|------|
| `param_embed_input_dim` | 固定 17 | trainer 164 行 |
| `param_embed_dim` | 配置或默认 `outdim` | trainer 166 行 |
| `offset_gru_hidden_dim` | 配置或默认与 `param_embed_dim` 一致 | trainer 167 行 |
| `offset_gru_use_reset_gate` | 是否使用 reset gate | trainer 184–186 行 |
| `mlp_params_embed` | 17 → param_embed_dim → param_embed_dim | trainer 171–174 行 |
| `param_embed_norm` | LayerNorm(param_embed_dim) | trainer 176 行 |
| `gru_update` | (param_embed_dim + outdim + hidden_dim) → hidden_dim | trainer 179–180 行 |
| `gru_candidate` | 同上 | trainer 181 行 |
| `gru_reset` | 同上或 None | trainer 183–186 行 |
| `gru_to_head` | hidden_dim → outdim（或 Identity 若相等） | trainer 189–192 行 |
| `h_cache_bg` | Dict[(scene_id, segment_id), Tensor [N, H]] | trainer 344、798 行 |
| `_h_cache_signatures` | 可选，用于 cache 失效判断 | trainer 347、581–594 行 |

Head（mlp_offset_pos / mlp_conv / mlp_opacity / gaussion_decoder）**输入维为 outdim**，与 Stage 1 一致；GRU 分支通过 `gru_to_head(h_new)` 映射回 outdim 再进 Head。

### 3.2 需要实现的方法

- **`_normalize_params_for_embed(params)`**  
  输入为 `_build_params_for_embed` 返回的 dict，输出 `[N, 17]`。可直接从 `offsets_mixin` 拷贝，仅依赖 `bbx_min/bbx_max` 与 NodeState 字段名。

- **`_build_params_for_embed(node_state_bg, coord_space="world")`**  
  Stage 1.1 只有 `NodeStateBackground`，无 rigid，因此无需 `frame_idx`；直接返回 `means, scales_log, quats, opacity_logit, sh_dc, sh_rest` 的 dict。可与 `offsets_mixin._build_params_for_embed` 对齐（去掉 rigid 分支即可）。

- **`_predict_offsets_gru(feat, params_for_embed, h_old, mask_update_rigid=None)`**  
  与 `offsets_mixin._predict_offsets_gru` 一致：param_vec → param_embed → concat feat → GRU → head_input → _predict_offsets。Stage 1.1 调用时始终传 `mask_update_rigid=None`；空特征边界处理（返回零 offsets + identity quat、h_new=h_old）可一并保留。

- **`_get_or_init_hidden(cache, key, num_points, node_state=..., node_type="bg")`**  
  与 trainer 中逻辑一致：取不到或 shape/signature 变化时零初始化并写回 cache，返回 `h.detach()`。

### 3.3 前向与训练流程变更（相对 Stage 1）

- **forward / train_step**  
  - 取或初始化 NodeState（与 Stage 1 相同）。  
  - 构建 3D 特征 `feat_3d_crop`。  
  - `params_bg = _build_params_for_embed(node_state_bg, coord_space="world")`。  
  - `h_old = _get_or_init_hidden(h_cache_bg, key, N, node_state_bg, "bg")`。  
  - `offsets, h_new = _predict_offsets_gru(feat_3d_crop, params_bg, h_old, mask_update_rigid=None)`。  
  - `render_params = _render_params_from_offsets(node_state_bg, offsets)`，渲染、loss、backward 同 Stage 1。  
  - Step 结束后 `h_cache_bg[key] = h_new.detach()`；若存在 NodeState write-back，逻辑同 Stage 1。

- **reset_node_state**  
  除清空 `node_states_bg` 外，清空 `h_cache_bg`（及可选 `_h_cache_signatures`），与完整版 node_state_mixin 行为一致。

### 3.4 配置项建议

在现有 Stage 1 的 model 配置下增加（可选，便于与完整版对齐）：

```yaml
# 可选，与 StreetForward 一致
param_embed_dim: 32
offset_gru_hidden_dim: 32
offset_gru_use_reset_gate: true
```

若省略则默认与 `outdim`（如 32）一致，便于最小改动验证。

---

## 4. 与完整 StreetForward 的差异（Stage 1.1）

| 项目 | 完整版 | Stage 1.1 |
|------|--------|-----------|
| 特征输入 | 3D 或 3D+2D 融合 | 仅 3D `feat_3d_crop` |
| params_for_embed | bg/rigid/distant，rigid 需 world 变换 | 仅 NodeStateBackground，无 frame_idx |
| mask_update_rigid | rigid 传 gate | 不传（None） |
| h_cache | bg / rigid / distant 三套 | 仅 `h_cache_bg` |
| inner_iterations | 可能 >1 | 通常 1（单 target 单步） |

其余（17 维 param_vec、GRU 公式、offset 头结构、渲染参数计算）与完整版一致，便于日后合并或对比。

---

## 5. 验证建议

1. **对齐性**：固定随机种子，Stage 1.1 在「零初始化 h、单 step、不写回 NodeState」下，与 Stage 1（同一 batch）首步 loss 可略有差异（因 GRU 输入多了 param_embed），但不应出现 NaN；若将 `param_embed` 和 GRU 输出置零，可退化为与 Stage 1 相近行为（需保证 gru_to_head 零初值或等价）。
2. **Overfit 一个 batch**：与 Stage 1 相同设置下，Stage 1.1 应能 overfit（loss 下降、PSNR 提升），且 h 随 step 更新（可做简单统计或可视化验证）。
3. **reset_node_state**：调用后再次 forward，应使用重新初始化的 h（全零），行为与「新 segment」一致。

---

## 6. 参考文件与代码位置

| 内容 | 位置 |
|------|------|
| GRU 流程说明 | [StreetForward_Flow.md](StreetForward_Flow.md) §5.1.1（363–396 行） |
| 参数 embedding、GRU 公式、空特征与 Rigid gate | `models/streetforward/offsets_mixin.py`：`_normalize_params_for_embed`、`_build_params_for_embed`、`_predict_offsets_gru` |
| GRU 模块与 h_cache 初始化/更新 | `models/streetforward/trainer.py`：param_embed/GRU 定义（163–192 行）、`_get_or_init_hidden`（567–595 行）、`h_cache_bg` 读写（707–709、778–779、798 行）、`_train_inner_iteration` 内 GRU 调用（934–946 行） |
| Stage 1 当前前向与 Head | `models/streetforward/minimal_trainer_stage1.py`：`_predict_offsets`、`_render_params_from_offsets`、`forward` / `train_step` |
| 设计阶段划分 | [Minimal_StreetForward_Design_Plan.md](Minimal_StreetForward_Design_Plan.md) §4 |

---

## 7. 小结

Stage 1.1 在保持「NodeStateBackground + 单 target、无 source、无 2D、无 Rigid」的前提下，引入与 [StreetForward_Flow.md §5.1.1](StreetForward_Flow.md) 一致的 GRU-style 偏移量预测：**17 维 param_vec → param_embed → 与 feat 拼接 → GRU 更新 h → head_input → 原有 _predict_offsets**，并维护 `h_cache_bg` 与 step 间 detach。实现时可直接复用或拷贝 `offsets_mixin` 的 `_normalize_params_for_embed`、`_build_params_for_embed`（仅 bg）、`_predict_offsets_gru`，在 minimal_trainer_stage1 中增加 GRU 相关模块与 h 缓存即可。完成后可为 Stage 2（多 target）及后续 Rigid/2D 扩展提供与完整版一致的基础路径。
