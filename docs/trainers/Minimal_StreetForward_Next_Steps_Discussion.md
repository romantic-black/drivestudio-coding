# Minimal StreetForward 下一步方案讨论

本文档在 Stage 0 已实现的前提下，讨论 **Stage 1：加 NodeState + 单 target** 的方案与实现要点。参考：

- [StreetForward 流程图与数据结构](StreetForward_Flow.md)
- [Minimal StreetForward 设计计划](Minimal_StreetForward_Design_Plan.md)（尤其是 107–113 行 Stage 1 定义）
- 训练脚本与模型：`tools/train_minimal_streetforward.py`、`models/streetforward/minimal_trainer.py`

---

## 1. 现状简述

### Stage 0（已实现）

- **输入**：3D RGB 点云（仅 static/background）+ 单 target（view + gt_image），无 source。
- **数据流**：  
  `点云(means, anchor_rgb)` → `_build_3d_features`（体素化 → sparse_conv → dense → 插值）→ `feat_3d_crop` → `_predict_render_params`（Head）→ 3DGS 参数 → 单 view 渲染 → L1 loss。
- **Head 形式**：**无 NodeState**。每步前向都用「点云坐标 + 点云颜色」重新算初始值（scales_log_init、opacity_logit_init、sh_dc_init 等），Head 只预测 **offset**，渲染参数 = 初始值 + eta × offset。  
  即：`means_r = means + eta*offset_pos`，`scales_log_r = scales_log_init + eta*offset_scales`，quats 用纯 offset 四元数等。
- **训练**：单 target、单 loss、单次 backward；无 NodeState 更新、无 proxy、无 GRU、无 2D 特征。

### 设计计划中的 Stage 1（目标）

引用 [Minimal_StreetForward_Design_Plan.md](Minimal_StreetForward_Design_Plan.md) 107–113 行：

- 点云 → **初始化 NodeStateBackground（仅静态）**，仍无 Rigid/Distant。
- 3D 特征仍只从**点云/NodeState 位置与颜色**来；**Head 改为「NodeState + offset」形式**（与现有 StreetForward 一致）。
- 仍**单 target、无 source**。
- **验证**：与 Stage 0 行为是否一致、能否 overfit。

---

## 2. Stage 1 与 Stage 0 的核心差异

| 维度 | Stage 0 | Stage 1（目标） |
|------|--------|------------------|
| **状态存储** | 无；每步从点云重新算“初始值” | 有 NodeStateBackground，首次从点云初始化，之后可持久化 |
| **Head 输入** | feat_3d_crop + 每步重算的 init（means, scales_log_init, opacity_logit_init, sh_dc_init） | feat_3d_crop + **NodeState 参数**（或仅 feat，见下节） |
| **渲染参数** | init + eta × offset（init 来自点云） | **NodeState + eta × offset**（与 [StreetForward_Flow](StreetForward_Flow.md) 中 `_render_params_from_offsets` 一致） |
| **状态更新** | 无 | 可选：每步/每隔步将渲染参数写回 NodeState（`_update_node_states` 的简化版） |
| **3D 特征来源** | 点云 means + anchor_rgb | 仍为点云/NodeState 的 means + 颜色（NodeState 首次即从点云来，一致） |

因此 Stage 1 的本质是：**引入持久化的 NodeState，并把「base」从「每步点云重算」改为「NodeState」，Head 仍预测 offset，渲染 = NodeState + offset**。

---

## 3. 与 StreetForward_Flow 的对应关系

[StreetForward_Flow.md](StreetForward_Flow.md) 中完整流程要点：

- **NodeState**：`_get_or_init_node_states` 从 pointcloud 得到 NodeStateBackground（及可选的 Rigid/Distant）；NodeState 全部 detached，作为缓冲区。
- **3D 特征**：`_build_3d_feature_volume` 用 NodeState 的 means（及 rigid 变换到世界坐标）和 anchor_rgb（如 `_sh_to_rgb(node_state.sh_dc)`）建体积，再插值得到 feat_3d_crop。
- **Head**：完整版用 **GRU-style**（`_predict_offsets_gru`）：feat + 参数 embedding + h_old → h_new，再对 h_new 做 offset 头；并带 h_cache、rigid gate 等。
- **渲染参数**：`_compute_render_params_for_inner_iter` / `_render_params_from_offsets`：  
  `means_r = node_state.means + eta_means * offset_pos`，同理 scales、quats、opacity、SH。
- **代理与多 target**：多 target 时用 proxy 做梯度累积；单 target 时不需要 proxy。

Stage 1 的取舍建议：

- **必须对齐**：  
  - 用 **NodeStateBackground** 存 means / scales_log / quats / opacity_logit / sh_dc / sh_rest；  
  - 渲染参数 = **NodeState + eta × offset**（即与 `_render_params_from_offsets` 的公式一致）；  
  - 3D 特征仍由「NodeState 的 means + 颜色（如 _sh_to_rgb(sh_dc)）」构建，这样与完整版一致，且首次与 Stage 0 同源（点云）。
- **可暂不实现**：  
  - **GRU**：Stage 1 可仍用「feat_3d_crop → offset 头」的简单路径（与 Stage 0 相同的 MLP 头），不引入 h_cache、param_embed、GRU。这样更容易验证「只换 NodeState + offset 形式」是否保持 overfit 行为。  
  - **Proxy**：单 target 不需要。  
  - **update_state**：可先做成可选（如配置项 `update_node_state: true/false`），便于对比「每步写回 NodeState」与「不写回、仅用初始 NodeState + offset」的差异。

---

## 4. 实现方案讨论

### 4.1 复用与新增

- **NodeState 初始化**：  
  可直接复用 `NodeStateMixin._init_node_state_background_only(scene_id, segment_id, pointcloud)`（或等价的 `_get_or_init_node_states` 仅背景路径），得到 `NodeStateBackground`。  
  需注意 Minimal 目前没有 Mixin 结构，可以：  
  - 在 `MinimalStreetForward` 里内联一份「仅背景」的初始化逻辑（与 node_state_mixin 保持一致），或  
  - 抽一个共享函数/小模块，供 Minimal 与完整 Trainer 共用，避免重复与行为不一致。
- **3D 特征**：  
  - 输入改为：`means = node_state_bg.means`，`anchor_rgb = _sh_to_rgb(node_state_bg.sh_dc)`。  
  - 其余与 Stage 0 相同：`_build_3d_features(means, anchor_rgb)` → `feat_3d_crop`。  
  这样「第一次」与 Stage 0 完全一致（因为 NodeState 刚从点云初始化）；之后若开启 update_state，means/sh_dc 会变，特征也随之变。
- **Head 与渲染参数**：  
  - **方案 A（推荐）**：Head 仍只吃 `feat_3d_crop`，输出 offset；渲染参数用 **NodeState 作为 base**：  
    `means_r = node_state_bg.means + eta_means * offset_pos`，  
    `scales_log_r = node_state_bg.scales_log + eta_scales * offset_scales`，  
    `quats_r = normalize(quat_multiply(node_state_bg.quats, offset_quat))`，  
    opacity / SH 同理。  
    与 [StreetForward_Flow](StreetForward_Flow.md) 中 `_render_params_from_offsets` 一致，且与 Stage 0 的**数学形式**一致（base 从点云 init 改为 NodeState）。
  - **方案 B**：为与完整版完全一致，再引入 GRU（feat + param_embed + h → offset）。Stage 1 可不先做，留给后续 Stage。
- **update_state（写回 NodeState）**：  
  - 若 `update_node_state=True`，每步（或每 K 步）在 backward 之后，用当前 `render_params_*` 的 detach 写回 `node_state_bg`（means 需 clamp 到 bbx），与 `_update_node_states` 中 bg 部分一致。  
  - 若 `update_node_state=False`，NodeState 始终为初始点云状态，仅「NodeState + offset」参与训练；可用来对比 Stage 0（无 NodeState）与 Stage 1（有 NodeState、不写回）的差异。

### 4.2 配置与入口

- **配置**：  
  - 在 minimal 的 config 中增加 `model.update_node_state: true/false`（或 `training.update_node_state`），默认建议先 `false`，便于与 Stage 0 对齐验证。  
  - 其他（bbx、voxel_size、eta、offset_max 等）与 Stage 0 保持一致，便于对比。
- **入口**：  
  - 仍使用 `tools/train_minimal_streetforward.py`，通过 config 或环境区分 Stage 0 / Stage 1；或新增 `minimal_streetforward_stage1.yaml`，对应「NodeState + 单 target」的模型/逻辑。

### 4.3 数据与脚本

- **Batch 格式**：与 Stage 0 相同，继续使用 `convert_batch_to_minimal_format`：pointcloud（仅 background）+ 单 target。  
- **Overfit batch**：与 Stage 0 共用同一 overfit batch（如 `scene0_seg0_batch.pt`），便于直接对比 loss/PSNR。

---

## 5. 验证标准（与设计计划一致）

1. **与 Stage 0 行为一致**  
   - 同一 overfit batch、相同 step 数、相同 seed：  
     - Stage 1 在 `update_node_state=False` 时，**第一步**的渲染结果应与 Stage 0 第一步一致（因为 NodeState 初始 = 点云，3D 特征与 base 一致，Head 结构相同）。  
   - 若第一步一致，则差异仅来自「base 是否持久化、是否写回」；可再比较若干 step 后的 loss 曲线是否接近。
2. **能否 overfit**  
   - Stage 1（先不写回）：loss 应能下降、PSNR 能提升，与 Stage 0 量级接近。  
   - Stage 1（写回）：overfit 同一 batch，最终 loss/PSNR 应至少不差于不写回，且 NodeState 在写回后应逐渐收敛到合理形状（可选：可视化或简单统计 means 变化）。

若上述两点满足，可以为 Stage 1 正确接入 NodeState + offset 形式，为 Stage 2（多 target + proxy）打基础。

---

## 6. 任务拆解（建议顺序）

1. **在 Minimal 中引入 NodeStateBackground**  
   - 从 pointcloud 初始化（复用或对齐 `_init_node_state_background_only`）；  
   - 以 `(scene_id, segment_id)` 为 key 缓存，仅背景、无 Rigid/Distant。
2. **3D 特征改为从 NodeState 取**  
   - means = node_state_bg.means，anchor_rgb = _sh_to_rgb(node_state_bg.sh_dc)；  
   - 保持现有体素化 → sparse_conv → 插值 流程。
3. **Head 改为 NodeState + offset**  
   - 渲染参数 = NodeState + eta × offset（与 `_render_params_from_offsets` 一致）；  
   - offset 仍由现有 MLP 从 feat_3d_crop 预测（不引入 GRU）。
4. **可选：update_state**  
   - 配置控制是否在每步后将渲染参数写回 NodeState（means clamp 到 bbx）；  
   - 默认 false，便于与 Stage 0 对比。
5. **验证**  
   - 同 batch、同 seed：Step 0 与 Stage 0 一致；  
   - 若干 step 后 overfit 曲线与 Stage 0 接近；  
   - 若开启 update_state，检查写回后训练稳定且 overfit 仍成立。

---

## 7. 参考文件小结

| 文档/代码 | 用途 |
|-----------|------|
| [StreetForward_Flow.md](StreetForward_Flow.md) | NodeState、3D 特征、_render_params_from_offsets、GRU/proxy 流程 |
| [Minimal_StreetForward_Design_Plan.md](Minimal_StreetForward_Design_Plan.md) 107–113 | Stage 1 定义与验证要求 |
| `tools/train_minimal_streetforward.py` | 当前 Stage 0 训练入口与 batch 转换 |
| `models/streetforward/minimal_trainer.py` | Stage 0 模型：无 NodeState，Head 为 init+offset |
| `models/streetforward/node_state_mixin.py` | NodeStateBackground 初始化、_get_or_init_node_states |
| `models/streetforward/offsets_mixin.py` | _predict_offsets、_render_params_from_offsets（及 GRU 版） |

本文档可作为 Stage 1 实现与评审的讨论基础；若后续引入 GRU 或多 target，可在此基础上追加小节。
