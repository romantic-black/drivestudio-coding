# StreetForward Stage 5_6：Nearby Error Pred（Post-GRU State）修订方案与向后兼容

## 文档目标

本文分两层次说明：

1. **目标方案（Recommended）**：将 nearby error prediction 的输入从「source lift 后的 2D feature」切换为「**更新后的 node state（优先 post-GRU hidden）**」，并停止使用 source-support 作为 splat mask；同步收紧 error head / fuser 的输入语义。
2. **向后兼容（Legacy）**：保留现有配置语义与代码路径（`post_update_node_feature` + `_feat_2d_*` + source-support gate），便于对照实验与老配置复现实验。

文中「**已实现**」「**计划中**」会显式标注；若实现与本文不一致，以代码为准，并应回写本文「实现状态」小节。

---

## 0. 与 Frame-Parallel Nearby Feedback 的关系

Stage 5_6 已落地 **「slot = nearby frame」** 的并行/串行框架（单帧内 all-cams 并行 splat / error head / V4 lift，多帧串行写 per-frame cache，下一 step 再 flatten-K fuser）。该框架与本文的 **error splat 特征从哪来** 正交：

- Frame-parallel 只规定 **按帧批处理视图** 与 **cache 拓扑**；
- 本文规定 **splatted 特征张量的语义**：应从 **当前 Gaussian state（post-update）** 来，而非 source-lifted f2d。

---

## 1. 语义修正：error pred 在学什么

監督目标本质是：

```text
e_gt = |render_after_update - gt_nearby|
```

因此 error predictor 看到的输入也应当描述 **同一套「update 之后的 state」** 在 nearby view 上的表现，而不是「source 上看到什么 / 可靠性如何」。

**不应再进入 error pred 特征通路的内容（目标方案）**

- source support / accumulation weight 作为可靠性提示；
- gate、history statistic、或其它「source 观测质量」先验；
- 用 source-support mask **禁止未在 source 看见的 node 参与 splat**（会把问题偷换成「二次传播 source f2d」）。

**仍建议保留的内容**

- nearby splat 后的 **渲染 RGB / alpha**：它们是 **当前 state 的显式渲染结果**，与 `e_gt` 同源，适合做条件上下文（与 support/gate 不同类）。

---

## 2. 主方案：`input_feature: post_gru_hidden`

### 2.1 数据流（目标）

```text
current updated node state (post-GRU hidden h_next)
    -> ErrorSplatProjector (branch-specific recommended)
    -> splat z_err to nearby views (all-cams in-frame)
    -> error head predicts (scalar error map, liftable latent)
    -> V4 multi-camera lift -> per-frame node cache
    -> next step flatten-K fuser -> struct path
```

与「post_struct / pre-GRU 上下文」对照：

```text
post_struct:
    update 语境 / decoder token，语义上偏「这一轮准备怎么更新」

post_gru_hidden:
    recurrent state，更接近「这一轮 update 之后 node 内部状态」
```

主实验建议使用 **post-GRU**，与 `e_gt` 对齐；`post_struct` 仅作为 **ablation**（见 §8）。

### 2.2 专用中间张量（建议字段名）

在 `out` 中增加 error splat 专用特征，避免复用 `_feat_2d_bg / _feat_2d_distant / _feat_2d_rigid_*`：

```python
out["_err_splat_feat_bg"]       # [N_bg, D_err]
out["_err_splat_feat_distant"]  # [N_dist, D_err]
out["_err_splat_feat_rigid"]    # [N_rigid, D_err]
```

其中 `h_bg_next`, `h_distant_next`, `h_rigid_next` 为各 branch GRU（或等价模块）输出的 hidden，`D_err` 由配置 `error_splat_dim` 指定。

### 2.3 `ErrorSplatProjector` 与梯度策略

**分支无关 vs 分支独立**

- 可共享一个 `Linear` 栈：`ErrorSplatProjector(in_dim=H_gru, out_dim=D_err)`；
- **更推荐**三套：`err_splat_proj_bg / err_splat_proj_distant / err_splat_proj_rigid`，因三支动态范围与更新语义不同。

**detach 边界（训练 error 路径但不搅动主 updater）**

```python
z_err = projector(h_next.detach())   # ✅ 推荐：error loss 训练 projector + error head，不向 GRU hidden 反传

# ❌ 若写作 projector(h_next).detach()，等价于 projector 常数段，不利于学习 splat 表征
```

即：**detach 点在 hidden 入口**；projector 输出 **不要**整张量 detach（否则 projector 拿不到梯度）。

`splat / renderer` 前对 **几何**仍可 `detach_geometry`（现状配置），error head 的 render 上下文也可 `detach_render_context`，与 projector 梯度策略独立。

### 2.4 实现易错点：**禁止在 splat 前对 projected feature 再 `.detach()`**

这是后续 PR 里最容易漏掉的一点。若已实现：

```python
h_in = h_next.detach()
z_err = projector(h_in)
```

则 **必须把 `z_err`（未 detach）送进 splat**，让梯度能流到 **projector 参数**；**不能**写成：

```python
feat_tilde, support_all = splat(node_features=z_err.detach(), ...)  # ❌ projector 梯度全断
```

与「旧路径里 `node_pack["features"].detach()` 一把 detach」区分开：Legacy 可把 lifted f2d 当常数喂给 error path；**新路径下 projector 是学习对象，splat 是算子，几何/alpha_weight 仍可 detach，feature 向量本身不可再在 splat 入口 detach。**

---

## 3. `node_mask`：`renderable` 的严格定义（不是「全 ones」）

现状（legacy）：`_splat_node_features_to_views` 将 `node_mask` 乘到 feature 与 splat normalization channel；若 mask 来自 `acc_w > τ`，则「source 没看见」的点 feature 恒为 0。

目标方案：**mask = 语义上的「允许参与 error splat 的 node」**，其含义是 **active & finite & branch_valid & frame_valid**，**不是**「source support 够不够」，也**不是** naive 的「全分支全 ones」：

```text
renderable :=
    active（未被 prune / 未被标记 disabled / 仍属于当前训练图与当前 branch）
    & finite_params（均值/尺度/不透明等无量纲或物理参数无 NaN/Inf）
    & opacity_not_degenerate（例如低于某地板可视为无效，具体阈值与主 renderer 一致）
    & scale_not_degenerate
    & branch_valid（确实属于 bg / distant / rigid 当前 segment 与合并队列）
    & frame_valid_for_rigid（对 rigid：在目标 auxiliary frame 上存在合法位姿 / 可被 `_build_rigid_world_for_aux_frame` 视作有效并入 world merge 的子集）
```

建议配置命名上把 `node_mask_policy: renderable` **文档化成上述合取式**，实现时逐项落地；刚性分支必须与 **目标 nearby frame** 对齐（「在该 frame 无有效 pose」的点应关掉），背景/远端亦应排除已知死点/非法参数而非无条件 `torch.ones`。

**说明**：splat 里的 normalization「support channel」仍可保留——它只是 rasterization 的加权归一中间量，**不是** reliability 特征，也不必进入 error head 或 fuser。

---

## 4. `_build_feedback_node_pack` 重写要点（目标方案）

Legacy：从 `_feat_2d_*`（及 rigid 上与 `route.S` 对齐的 source-observed 子集填零逻辑）拼装。

目标方案拼装逻辑概要：

```text
feat_bg       <- out["_err_splat_feat_bg"]
feat_distant  <- out["_err_splat_feat_distant"]
feat_rigid    <- out["_err_splat_feat_rigid"]

rigid_world, rigid_order = _build_rigid_world_for_aux_frame(out, target_frame_idx)

# bg:   全量 + renderable mask
# rigid: 按 rigid_order 从「全量 rigid err feature」取样，而不是按 route.S 的子集对齐 source f2d
# distant: 若存在 distant render branch，同上
merged_render = _tensor_merge_bg_rigid_distant_world(render_bg, rigid_world, render_distant)

return {
  "render": merged_render,
  "features": merged_features,
  "mask": merged_renderable_masks,
  ...
}
```

**Rigid 关键差异**：改用 post-GRU 全向量后，rigid **不应再**沿用「仅用 route.S 可观测 rigid 才把 feature 填入、其余填零」的 `_feat_2d_rigid_S` 语义；应按 **aux frame 的有效 rigid index 顺序**取 `feat_rigid_all[rigid_order]`。

---

## 5. Error head 输入保持「干净」（目标方案）

仅拼接：

```text
splatted post-GRU-derived feature   # [V, D_err, H, W]

+ pred_rgb_ctx   （可选，detach_render_context 可配）
+ pred_alpha_ctx （可选）
```

不拼接：source support、feedback support、concat valid_mask 进 head、branch embed、gate、history stats 等。

`Stage5_6ErrorPredictHead` 结构与「预测标量 error + liftable latent」目标可保持不变。

---

## 5.1 Fuser **注入位置**与语义：post-GRU source vs pre-struct  sink（轻微不对称）

主方案数据流在时间上是：

```text
本步：post_update（含 post-GRU hidden）
  -> error pred / lift -> 写入 per-frame feedback cache

下步：读 cache -> flatten-K fuser -> 注入点通常在 before_struct_decoder 的 feat_2d 路径
```

因此存在**刻意允许**的一层语义不对称：

- **来源**：feedback 描述的 latent 是从 **更新后状态**（与 `e_gt = |render_after_update - gt_nearby|`）一致的空间里 lift 回来的 **nearby error descriptor**；
- **去向**：它被 adapter 映射进 **下一步的 pre-struct 更新输入（feat_2d 侧 tokens）**，**不是**对 GRU hidden 做向量残差。

务必在文档与代码注释中写清：

```text
feedback latent ≠ hidden residual
feedback latent ≈ 「nearby failure pattern」经 fuser 转成「下一轮 struct/updater 可用的 2D-side 调制」
```

避免后续误设为「直接把 feedback 加到 `h_next`」。

### Fusion 档位（Ablation roadmap）

| 档位 | 注入点 | 备注 |
|------|--------|------|
| **A（P0 推荐）** | `before_struct_decoder` | 与当前 Stage5_6 一致，改动面最小、也相对最稳 |
| **B（可选）** | `before_gru` / 与 **post_struct** token 对齐的更早融合 | 更强的「update 语境」耦合，工程与语义都要单独验证 |
| **C（不推荐作默认）** | 直接进入 **hidden state** | 容易与主 recurrence 争抢自由度，仅在强假设下做小范围 ablation |

**P0** 仍以 **档位 A** 为默认叙事；B/C 仅作对照，不在首版铺开。

---

## 6. Feedback fuser：去掉 support 类输入（目标方案）

目标方案推荐关闭：

```yaml
nearby_error_feedback:
  fusion:
    input_current_source_support: false
    input_feedback_support: false
```

Flatten-K fuser 的输入简化为：**当前点特征 + 各 slot 的 feedback latent / error / valid**。

可选更激进做法：`valid` 仅用于 **`delta *= valid_any` 的输出门控**，不拼进 MLP（实现时二选一并在配置中写明）。

Legacy：若旧实验依赖 support 调制 residual，仍可保留两项为 `true`（向后兼容）。

### 6.1 与 `feedback_lift.support_min` 的边界（易混点）

下列两件事**不要混为一谈**：

| 说法 | 对错 |
|------|------|
| error head / fuser **不拼接** raster support、source support、feedback support 当作特征 | ✅ 正确（目标方案） |
| cache 写入时 **不使用** lift 返回的 accumulated weight、`support_min` 判断 node 侧 feedback 是否可信 | ❌ **错误** |

**`feedback_lift.support_min` 必须保留**：V4 lift 输出的 per-node **geometry coverage / accumulated weight** 是「该像素的 error 证据是否真正投回到这个 Gaussian」的判据，属于 **cache pack 的 valid 构造**，**不是** error predictor 的输入特征。  
node 级 `valid`（及可选的 `valid_ratio` 统计）应继续由 **`support > support_min`**（及必要时的其它 lift 侧规则）驱动；仅当 **不把该标量再 feed 进 head/fuser MLP** 时，才与「不加 support 特征」一致。

---

## 7. 配置草案（YAML，目标方案示意）

以下为 **建议**默认；具体键名以实现为准，但若与本文分歧应在 PR 中说明。

```yaml
nearby_error_feedback:
  error_pred:
    enable: true
    target_role: near_random

    # 主方案：post-GRU；Legacy 见 §9
    input_feature: post_gru_hidden
    error_splat_dim: 48
    detach_input_hidden: true
    detach_projected_feature: false   # projector 输出应可反传到 projector 参数

    splat_to: nearby_view
    # renderable = active & finite & branch_valid & frame_valid（见 §3，非 naive ones）
    node_mask_policy: renderable      # Legacy: source_support_threshold

    detach_geometry: true
    detach_alpha_weights: true
    detach_render_context: true

    use_render_rgb: true
    use_render_alpha: true

    use_source_support_input: false
    concat_log_support: false
    concat_valid_mask: false

    head_type: lite_unet
    hidden_dim: 64
    error_feat_dim: 16
    error_max: 0.7

    max_frames_per_step: 2
    every_n_steps: 1

  fusion:
    type: flatten_frame_slots
    num_slots: 2
    input_current_source_support: false
    input_feedback_support: false
    input_feedback_age: false
```

`feedback_lift.support_min` **必须保留**：用作 **lift 写入 cache 时 node-level valid**（几何覆盖阈值），见 §6.1；**不作为** error head / fuser 的特征输入。

---

## 8. Ablation：`input_feature: post_struct` —— 「哪个 post_struct」必须写死

StreetForward 的 struct 路径 **near / far / rigid 三套并不天然同构**，不能假设存在一个统一的 `post_struct_tensor`：

| Branch | 典型形态（随实现而异） |
|--------|-------------------------|
| **Near（bg / rigid_in）** | xCPE / sparse 3D conv 等结构化 token |
| **Far（distant / rigid_out）** | **MLP** 支路上的点级 token |
| **Rigid（routed）** | source-frame routed decoder + **local-world / auxiliary frame** 变换下的表达 |

因此对 **post_struct ablation**，实现上必须：

1. **分别从三条支路取其「struct decoder 输出、且已对齐到点后」的表征**（名字可为 `post_struct_bg`, `post_struct_distant`, `post_struct_rigid` 或与 `out` 中现有字段对齐）；  
2. **禁止**假定三者 **shape / 语义** 天然一致；  
3. 统一送进 splat 前，一律经 **三支独立 projector** 压到同一 `error_splat_dim`（与 post-GRU 分支一致：`D_err` 可配置）；

```python
z_err_bg       = proj_err_bg(post_struct_bg)           # [N_bg, D_err]
z_err_distant  = proj_err_distant(post_struct_distant)  # [N_dist, D_err]
z_err_rigid    = proj_err_rigid(post_struct_rigid)     # [N_rigid, D_err]
```

其中 `proj_err_*` 的 **输入维**由各支路 token 维度决定（near xCPE 与 far MLP、rigid routed 三路通常互不相同），**输出维**统一为 `D_err`。配置里写 `error_splat_dim: 64` 时，语义是「**三路输出维在 projector 后对齐到 64**」，而不是「post_struct 原始就是 64 维一张量」。

优劣简述：

- **更稳**：对历史递归依赖更少；
- **语义偏移**：表征偏「结构化 update 语境」，与 `render_after_update` 的 error 对齐度弱于 **post_gru_hidden**；
- **工程更重**：必须维护 **三套 struct 取样 + 三套 projector**，与 §2 中 post-GRU 的三分支 projector 同理。

主实验仍以 **post_gru_hidden** 为默认叙事；post_struct 为 **对齐维度的三路 ablation**，不是单一 tensor。

---

## 9. 向后兼容（Legacy）配置与语义

下列约定用于 **不破坏既有配置与 checkpoint 叙述**；代码实现应采用 **显式分支**（例如读取 `error_pred.input_feature`），默认值可逐步切换但须文档化。

### 9.1 `input_feature: post_update_node_feature`（Legacy，现行主线）

语义保持与早期 Stage 5_6 一致：

- splat 使用 `_feat_2d_bg / _feat_2d_distant / _feat_2d_rigid_S`（或 fusion 改写后的 `_stage5_6_last_fused_features` 中的同名 role）；
- `node_mask` 由 **source backproject support**（如 `acc_w > τ`）推导；
- rigid 侧 **允许** 继续用 `route.S` 将 source-observed 行对齐到全 rigid（未观测位置填零）——这是为 **source-lifted f2d** 设计的；
- error head 仍 **仅**使用 splat feature +（可选）render RGB/alpha，**不**要求重新引入已关闭的 `use_source_support_input` 等。

### 9.2 `input_feature: post_gru_hidden`（目标方案）

- 使用 §2.2 的 `_err_splat_feat_*` 与 §2.3 / §2.4 的 projector 与梯度边界；
- `node_mask_policy: renderable`（§3 严格定义）；
- rigid 按 §4 从全量 `feat_rigid` + `rigid_order` 取样。

### 9.3 `input_feature: post_struct`（Ablation）

- 使用 §8 的三路 **`post_struct_*` + `proj_err_*`**，统一到 `error_splat_dim`；
- 其余 splat / error head / lift / cache valid（`support_min`）流程与 post-GRU 目标方案一致。

### 9.4 `max_targets_per_step`（Legacy 图像数上限）

Frame-parallel 版本以 **`max_frames_per_step`** 为主；若仅提供旧键 `max_targets_per_step`，实现可保留兼容映射，例如：

```text
max_frames_per_step = max(1, max_targets_per_step // num_cams)
```

（仅当未显式配置 `max_frames_per_step` 时生效。）

### 9.5 Cache / Fusion 模式（已实现约束）

现行实现要求：

- `nearby_error_feedback.cache.mode == frame_bank`；
- `fusion.type == flatten_frame_slots`；
- `fusion.input_feedback_age == false`。

Legacy **实验若在旧分支仍可跑 overwrite 单槽 cache**，应在代码中单独维护或文档标注「frozen revision」，避免静默行为变化。

---

## 10. 训练阶段建议（与现有 warmup 对齐）

与现有 **`pred_error_only_steps` / `fusion_start_step` / `fusion_warmup_steps`** 可组合为三阶段叙事：

**Stage A（0 ~ pred_error_only）**

- 训练：`ErrorSplatProjector` + `error_head`；
- fusion scale = 0；
- cache 可按策略允许写入（与现配置一致）。

**Stage B（fusion 打开并 warmup）**

- `fusion_scale: 0 -> 1`；
- 继续训练 projector + head + **fusers**；
- 保持 `h_next.detach()` 再 projector，避免 auxiliary error loss 直连扰动 GRU。

**Stage C（可选）**

- `detach_input_hidden: false`：仅在 error pred 足够稳定且需更强耦合时试用；默认不建议开局启用。

---

## 11. 实现检查清单（给后续 PR）

- [ ] 在 forward 末尾或 GRU update 之后填充 `_err_splat_feat_*`（post-GRU）或等价的三路 post_struct 取样（§8）；
- [ ] `_build_feedback_node_pack` 按 `input_feature` 分支（Legacy vs post_gru vs post_struct）；
- [ ] **§2.4**：splat 的 `node_features` 传入 **projector 输出且未 detach**；几何 `detach_geometry` / `detach_alpha_weights` 仍可开；
- [ ] §3：**renderable mask** 按 active & finite & branch_valid & frame_valid 实现，**禁止**无脑全 `ones`；
- [ ] rigid：post_gru / post_struct 分支取消仅靠 `route.S` 对齐 source f2d 的填零策略（Legacy 除外）；
- [ ] §6.1：**cache pack 的 `valid` 仍由 lift `support` 与 `support_min` 决定**，与 fuser「不输入 support 特征」同时成立；
- [ ] §5.1：注释/文档标明 feedback 注入 **feat_2d adapter**，不是 **hidden residual**；默认 fusion 仍为 **档位 A**；
- [ ] 单测 / 冒烟：K、V、**projector 梯度存在**、`h_next` **无** error loss 梯度（当 `detach_input_hidden: true`）。

---

## 12. 一句话总结

**目标语义**：Nearby error prediction 归因于 **当前（更新后）Gaussian state 在 nearby 的表现**；用 **post-GRU hidden → 分支 projector → splat（feature 不因 splat 再 detach）**，配合 **严格 renderable mask**、**不加 support 的干净 error head**，以及 **仍以 lift support + `support_min` 构造 cache valid**；fuser 在 **P0 仍建议 before_struct_decoder**，把反馈当作 **nearby error descriptor → 下一轮 pre-struct 输入**，而非 hidden 残差。

**向后兼容**：保留 **`post_update_node_feature`** 路径，继续使用 **source-lifted `_feat_2d_*` + source-support mask + 旧 rigid 对齐逻辑**，便于复现与消融对照。

**post_struct 消融**：**三路 struct 表征 + 三路 projector**，统一到 **`error_splat_dim`**，不假设单一同构 tensor。
