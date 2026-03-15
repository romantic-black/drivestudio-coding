# Minimal StreetForward Stage 1 实现方案：NodeState + 单 Target（独立新模型）

本文讨论 `docs/trainers/Minimal_StreetForward_Design_Plan.md` 中 Stage 1（L108-L112） 的实现落地方案，并 **参考** `docs/trainers/StreetForward_Flow.md` 的数据流与命名习惯。

> **硬性约束**：Stage 1 必须新建一个与 `models/streetforward/minimal_trainer.py` 中 `MinimalStreetForward` **完全独立** 的模型（新文件、新类名、新状态缓存），不要在 `MinimalStreetForward` 上打补丁式演进，避免 Stage 0 与 Stage 1 互相污染。

---

## 1. Stage 1 目标与非目标

### 目标（对应 Design Plan Stage 1）

- **点云 → 初始化 NodeStateBackground（仅静态）**
  - 只支持 background 点云；仍 **不引入** Rigid / Distant（动态与远景一律不做）。
- **3D 特征来源不变**
  - 3D 特征仍来自体素化 + sparse conv + 插值；
  - 颜色/位置来自 NodeState（初始化时来自点云，更新后来自 NodeState 写回）。
- **Head 形式切换为 “NodeState + offsets → render params”**
  - 与 `StreetForward_Flow.md` 中的 “NodeState（detached buffer） + offsets（可微）→ render params（可微）→ render” 对齐。
- **训练仍单 target**
  - 无 source、无 proxy、多视角梯度累积（Stage 2 才加）。
- **验证**
  - 与 Stage 0 行为一致（在禁用 NodeState 更新时）；
  - 能 overfit one batch（启用 NodeState 更新时）。

### 非目标（明确不做）

- 不实现：`RigidMasks`、`NodeStateRigid`、`NodeStateDistant`、刚体变换、GRU-style hidden fusion、2D features、proxy rendering、多 target 梯度累积、eval pipeline 的完整对齐。

---

## 2. 新模型命名与文件建议（必须独立）

建议新建文件与类（任选其一，但需与 Stage 0 分离）：

- **文件**：`models/streetforward/minimal_trainer_stage1.py`
- **类**：`MinimalStreetForwardNodeStateSingleTarget`（或 `MinimalStreetForwardStage1`）

并新增对应 trainer/脚本入口（是否马上实现由后续任务决定）：

- `tools/train_minimal_streetforward_stage1.py`
- `configs/minimal_streetforward_stage1.yaml`

这样 Stage 0 的 `MinimalStreetForward` 保持“无状态、无 NodeState”的纯 baseline，Stage 1 作为独立闭环更容易对照与回退。

---

## 3. 参考 StreetForward 的关键切片（Stage 1 只取其中最小闭环）

从 `StreetForward_Flow.md` 中抽取 Stage 1 需要复用的概念与步骤：

- **NodeStateBackground**：作为 **detached buffer** 持有 `means/scales_log/quats/opacity_logit/sh_dc/sh_rest`
- **3D feature volume**：以 `NodeState.means` 与 `sh_dc → anchor_rgb` 构建体素特征并插值到每点
- **offsets 预测**：MLP head 从 per-point feature 预测 offsets（pos/scale/quat/opacity/SH）
- **render params**：`node_state + eta * offsets`，得到用于渲染的可微参数
- **渲染与 loss**：单 target view
- **写回 NodeState（可选开关）**：`with torch.no_grad(): node_state <- render_params.detach()`，并对 means 做 clamp（与 StreetForward 一致）

Stage 1 不需要：proxy params、multi target、rigid 子集合并、rigid world/local 转换。

---

## 4. 数据结构设计（Stage 1 的最小 NodeState）

### 4.1 NodeStateBackground 数据类

直接沿用 StreetForward 的字段集合（保证后续 Stage 2/4 演进不需要改字段）：

- `means: [N, 3]`（world）
- `scales_log: [N, 3]`
- `quats: [N, 4]`（wxyz）
- `opacity_logit: [N, 1]`
- `sh_dc: [N, 3]`
- `sh_rest: [N, num_sh-1, 3]`

### 4.2 NodeState 缓存 Key

建议与 StreetForward 一致，按 `(scene_id, segment_id)` 缓存：

- `self.node_states_bg: Dict[Tuple[int,int], NodeStateBackground]`

Stage 1 单 batch overfit 时通常只会命中一个 key，但保持该结构有利于后续扩展。

---

## 5. 端到端前向/训练流程（单 Target、无 Proxy）

本节给出 Stage 1 的最小可运行训练闭环（对齐 StreetForward 的命名，但删去无关分支）。

### 5.1 `train_iter(batch)`（推荐对齐的顶层 API）

输入 batch（与 Stage 0 保持一致的最小子集）：

- `scene_id: int`
- `segment_id: int`
- `pointcloud: dict | object`（只读取 static/background）
- `targets: List[{"view": View, "gt_image": Tensor, "frame_idx": int?}]`
  - Stage 1 **只取** `targets[0]`

流程：

1. `_get_or_init_node_state_bg(scene_id, segment_id, pointcloud)`
2. `target = targets[0]`
3. `_build_3d_feature_volume(node_state_bg)` → `feat_3d_crop_bg: [N, C]`
4. `_predict_offsets(feat_3d_crop_bg)` → offsets dict
5. `_render_params_from_offsets(node_state_bg, offsets)` → render params dict
6. `_render_single_view(render_params, target["view"], H, W)` → `rgb`
7. `loss = mse(rgb, target["gt_image"])`
8. `loss.backward()` → `optimizer.step()`
9. 若 `update_state=True`：`_update_node_state_bg(node_state_bg, render_params)`

### 5.2 与 Stage 0 “行为一致性” 的关键开关

为了验证 Stage 1 只是在结构上引入 NodeState，而不是改了数值逻辑，建议增加两个运行模式：

- **模式 A（对齐 Stage 0）**：`update_state=False`
  - NodeState 只用于承载初始点云参数，不发生写回；
  - 这时 render params 始终是“初始点云 + offsets”，应当与 Stage 0 的“点云参数 + offsets”在数值上尽量一致（允许初始化细节差异，但整体收敛曲线应相近）。
- **模式 B（真正 Stage 1）**：`update_state=True`
  - 每步把渲染参数写回 NodeState，形成 “stateful refinement”。

---

## 6. 关键实现细节（避免踩坑）

### 6.1 初始化：pointcloud → NodeStateBackground

建议复用 Stage 0 已经跑通的初始化逻辑（同样的尺度初始化、SH 转换、quat 初始化等），但输出改为 NodeState 字段：

- `means`：点坐标（float32）
- `sh_dc/sh_rest`：由 rgb 转 SH；`sh_rest` 初始化 0
- `scales_log`：基于 kNN 距离估计并取 log（与 StreetForward 一致）
- `opacity_logit`：常数初始化（例如对应 0.1）
- `quats`：随机单位四元数或全 identity（二者都可，但需在 Stage 0/1 对齐时固定策略）

> 注意：Stage 1 的 “行为一致性” 强依赖初始化细节一致。若 Stage 0 使用了某套初始化（例如随机 quat），Stage 1 应尽量复制同样策略，否则对齐验证会被初始化噪声干扰。

### 6.2 特征体积构建：NodeState → anchor_rgb → voxelize

对齐 `StreetForward_Flow.md` 的数据流：

- `anchor_rgb = _sh_to_rgb(node_state_bg.sh_dc)`（或初始化时直接保留 rgb，但推荐按 StreetForward 走 sh_dc）
- `construct_sparse_tensor(raw_coords=node_state_bg.means, feats=anchor_rgb, ...)`
- `sparse_conv → sparse_to_dense_volume → grid_sample` 得到 `feat_3d_crop_bg`

### 6.3 offsets head：输入维度与 Stage 0 一致

Stage 1 暂不引入 param-embedding 或 GRU hidden fusion，因此 offsets head 可保持 Stage 0 的 MLP 结构（EVolSplat-style）：

- `mlp_offset_pos: C → 3`
- `mlp_conv: C → 6`（scale delta + axis-angle）
- `mlp_opacity: C → 1`
- `gaussion_decoder: C → 3*num_sh`

并保持“最后一层 zero-init”以确保训练初期 offsets 接近 0（参考 `StreetForward_Flow.md` 的偏移头初始化说明）。

### 6.4 render params：NodeState + eta * offsets

保持与 `StreetForward_Flow.md` 的语义一致（这里只存在 bg 分支）：

- `means_r = node_state.means + eta_means * offset_pos`（渲染时不 clamp）
- `scales_log_r = node_state.scales_log + eta_scales * offset_scales`
- `scales_r = exp(scales_log_r)`
- `quats_r = normalize(quat_multiply(node_state.quats, offset_quat))`
- `opacity_logit_r = node_state.opacity_logit + eta_opacity * offset_opacity`
- `opacities_r = sigmoid(opacity_logit_r).squeeze(-1)`
- `sh_dc_r, sh_rest_r` 按 DC/rest 分拆后分别加偏移
- `colors_r = cat([sh_dc_r[:,None,:], sh_rest_r], dim=1)`（供渲染）

### 6.5 NodeState 写回：必须 no_grad + detach + clamp

对齐 StreetForward 的要点：

- `with torch.no_grad():`
  - `node_state.means.copy_(clamp(render_params["means_r"].detach(), bbx_min, bbx_max))`
  - 其他字段 `copy_(render_params[field].detach())`

写回时 clamp，渲染时不 clamp（保护梯度），是 StreetForward 已验证过的一条关键经验。

---

## 7. 验证计划（Stage 1 的 Definition of Done）

### 7.1 与 Stage 0 的一致性验证（update_state=False）

- **设置**：相同 overfit batch、相同随机种子、相同网络初始化策略（尤其是 quat 初始化）
- **期望**：
  - loss/PSNR 曲线与 Stage 0 近似
  - 单步前向渲染的数值差异在可解释范围（允许微小差异，但不应出现明显发散）

### 7.2 overfit 验证（update_state=True）

- **设置**：与 Stage 0 相同 batch
- **期望**：
  - loss 稳定下降、PSNR 上升
  - NodeState 参数（means/scales/opacity 等）随迭代发生合理变化（可选打印统计量：均值/方差/最大步长）

### 7.3 失败定位顺序（建议）

1. 先跑 `update_state=False` 对齐 Stage 0（排除初始化/渲染/特征差异）
2. 再打开 `update_state=True`（若发散，多半是写回 clamp/尺度/opacity 的数值范围问题）

---

## 8. 与后续 Stage 的接口预留（但不实现）

Stage 2（多 target + proxy）会把 “render + backward” 的形态从单 view 扩展到多 view 梯度累积，但 Stage 1 只要保证以下接口边界清晰即可：

- `_render_params_from_offsets(node_state_bg, offsets_bg) -> render_params_bg`
- `_create_proxy_params(render_params_bg)`（Stage 1 可以没有；Stage 2 再加）
- `_render_targets_and_accumulate_loss(proxies_bg, targets)`（Stage 2 再加）
- `_backward_to_render_params(...)`（Stage 2 再加）

这样 Stage 1 的代码结构会天然贴近 `StreetForward_Flow.md`，但实现复杂度仍保持最小。

---

## 9. 参考

- `docs/trainers/Minimal_StreetForward_Design_Plan.md`（Stage 1：L108-L112）
- `docs/trainers/StreetForward_Flow.md`（NodeState / offsets / render params / update 的权威数据流）
- `models/streetforward/minimal_trainer.py`（Stage 0 已实现 baseline，供一致性对照）

