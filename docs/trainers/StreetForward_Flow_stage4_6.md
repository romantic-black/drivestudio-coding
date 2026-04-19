# StreetForward Flow（Stage4_6 Canonical）

本文以 `models/streetforward/minimal_trainer_stage4_6.py` 的 `MinimalStreetForwardStage4_6` 为唯一基准，描述 Stage4_6 训练/推理主流程、数据契约与关键不变量。

> 参考实现：`minimal_trainer_stage4_4.py`、`minimal_trainer_stage4_5.py`、`minimal_trainer_stage4_6.py`、`models/feature_extractors/alpha_t_extractor_v3.py`。

---

## 1. Stage4_6 核心定位

Stage4_6 是在 Stage4_5（no-sky）基础上的 routed rigid 版本：

- 移除 rigid 专属 decoder/head（不再有 `mlp_offset_pos_rigid` 等模块）。
- rigid 仍是 dynamic node（local 坐标存储）。
- source 帧把 rigid 点先变到 world，再按 `segment_aabb` 路由：
  - inside -> 走 `bg` 共享 heads
  - outside -> 走 `distant` 共享 heads
- source 2D 仍使用 `AlphaTWeightExtractorV3` 的 multi-camera fused 回投（one-pass）。
- 损失保持 Stage4_5 语义：photometric 仅在 non-sky 区域计算。

---

## 2. 继承关系与兼容层

类继承链（关键部分）：

- `MinimalStreetForwardStage4_6`
  - `MinimalStreetForwardStage4_5BaseNoRigidHead`
    - `MinimalStreetForwardStage4_5`
      - `MinimalStreetForwardStage4_2`

`MinimalStreetForwardStage4_5BaseNoRigidHead` 的职责：

- 用兼容配置先走一遍 Stage4_5 初始化（补齐 rigid legacy 字段，仅用于兼容初始化流程）。
- 初始化后删除 rigid 专属模块并重建 optimizer。
- 再用原始 Stage4_6 配置覆盖运行时语义（避免配置漂移）。

---

## 3. 配置 fast-fail 规则（Stage4_6）

`_validate_stage4_6_config` 与 `_parse_rigid_routed_cfg` 的硬约束：

- 禁止 `model.sky` 与 `model.branches.sky`（Stage4_6 是 no-sky）。
- `model.branches.rigid` 禁止出现：
  - `mlp`
  - `limits`
  - `freeze_means`
  - `freeze_quat`
- `model.rigid_routed` 必须满足：
  - `route_space: source_frame_world`
  - `route_aabb: segment_aabb`
  - `inside_decoder: bg`
  - `outside_decoder: distant`
- `update_means` 与 `update_quat` 显式受配控制。

这类约束不满足会直接 `raise ValueError`，属于 fast-fail 设计。

---

## 4. Source 2D 一次回投（one-pass fused）

Stage4_6 复用 Stage4_5 的 scene-only 2D 管线，并保持 one-pass：

1. 构建 source 下的 `gaussians_scene = bg + distant + rigid@S(world)`。
2. `_render_source_scene_only_for_cnn` 生成 `[image, render]` 拼接输入。
3. `AlphaTWeightExtractorV3.render_and_backproject_streaming_fused_multi_camera(...)` 执行 fused 回投。
4. 一次性拿到 `feat_2d_all / acc_w_all`，再按顺序切分：
   - `bg`
   - `distant`
   - `rigid_S`

关键点：

- Stage4_6 仍依赖 V3 fused 多相机 CUDA 路径。
- `source_pair_valid_mask`（sky/egocar mask 组合）会下沉到 fused kernel。
- 统计口径中 `src_backproject_pass_count = 1`。

---

## 5. Routed rigid 主流程

### 5.1 索引定义

- `S`：source 帧可见 rigid 点（全局 rigid 下标子集）。
- `S_in`：`S` 中位于 `segment_aabb` 内部的点。
- `S_out`：`S` 中位于 `segment_aabb` 外部的点。
- `U`：参与更新的 rigid 点（`mask_src_feat_valid_rigid & mask_any_tgt_rigid`）。
- `U_in`：`U` 中 routed to `bg` 的子集。
- `U_out`：`U` 中 routed to `distant` 的子集。

### 5.2 路由规则

`_route_rigid_source_points`：

- 把 `node_state_rigid.means/quats` 从 local 变换到 source-frame world。
- 使用 `bbx_min <= means_world <= bbx_max` 做 inside 判断。
- 输出 `RigidRoute` 数据结构，包含全套索引与 world 变换结果。

### 5.3 共享 head 更新

- 对 `U_in`：
  - 融合 `feat_3d_rigid_in + feat_2d_rigid_S`。
  - 走 bg heads（`mlp_offset_pos` / `mlp_conv` / `mlp_opacity` / `gaussion_decoder`）。
- 对 `U_out`：
  - 走 distant 路线（2D 投影后用 distant heads）。
- 两路得到 world offsets 后，统一回到 rigid local 参数空间写渲染参数。

这一步由 `_render_params_from_routed_offsets_rigid_local` 完成，内部包含：

- `world -> local` 坐标反变换
- `quat` world/local 双向变换
- `eta` 应用（来自 `model.branches.rigid.eta`）

---

## 6. 损失语义（继承 Stage4_5）

训练阶段对每个 target view：

- `valid_loss_mask`：有效监督区域。
- `sky_mask`：必须提供（`require_sky_mask_for_loss=true`）。
- photometric（L1 + SSIM）只在 `valid_non_sky_mask = valid_loss_mask * (1 - sky_mask)` 上计算。
- mask BCE 与 opacity entropy 保持 no-sky 管线一致。

聚合方式：

- 先 view 内求 `total_i`。
- 再 frame 内平均。
- 最后对 frame 平均得到总 loss。

---

## 7. 写回与状态缓存

### 7.1 Hidden cache

`h_cache_bg`、`h_cache_distant`、`h_cache_rigid` 延续 Stage4_5 机制。

### 7.2 NodeState 写回

Stage4_6 覆盖 `_writeback_node_states_from_out`，关键差异：

- bg/distant 仍走各自 subset 写回逻辑。
- rigid 必须使用 `_rigid_writeback_idx`（即 `U_all`）与 `render_params_rigid_local` 对齐写回。
- 不调用 Stage4_5 的 rigid 写回路径，避免 routed 下标错位风险（测试已覆盖）。

---

## 8. 日志与可观测指标（新增重点）

Stage4_6 在输出中新增 routed 相关指标：

- `rigid_route_num_S`
- `rigid_route_num_in`
- `rigid_route_num_out`
- `rigid_route_ratio_in`
- `rigid_route_ratio_out`
- `rigid_in_update_count`
- `rigid_out_update_count`
- `rigid_in_acc_w_mean`
- `rigid_out_acc_w_mean`
- `rigid_writeback_count`

并保留兼容指标：

- `grad_norm_rigid_legacy`
- `rigid_grad_norm_routed_to_bg_shared`
- `rigid_grad_norm_routed_to_distant_shared`

---

## 9. 与 Stage4_4 / 4_5 / 4_6 对比

| 维度 | Stage4_4 | Stage4_5 | Stage4_6 |
|---|---|---|---|
| sky node | 有（sky GS） | 无 | 无 |
| source backproject | scene + sky（双 pass） | scene-only（one-pass） | scene-only（one-pass） |
| rigid head | 独立 | 独立 | 移除（路由到 bg/distant 共享） |
| rigid 更新空间 | local 存储，常规 rigid path | local 存储，常规 rigid path | local 存储，source-world 路由决策 |
| photometric 区域 | 依阶段配置 | non-sky | non-sky |

---

## 10. 训练入口与配置

推荐入口：

- 脚本：`tools/train_minimal_streetforward_stage4_6_multi_scene_v7.py`
- 配置：`configs/minimal_streetforward_stage4_6_multi_scene_v7.yaml`

该入口会显式绑定：

- V4 dataset 构建器
- V7 scheduler
- Trainer class = `MinimalStreetForwardStage4_6`

---

## 11. 测试覆盖建议（现有单测）

`tests/test_minimal_stage4_6.py` 已覆盖关键行为：

- forbidden rigid fields fast-fail
- compat 配置注入
- route split 正确性（`S_in/S_out`）
- subset 写回索引对齐（`U`）
- 只 `U_in` / 只 `U_out` 时打包行为
- 覆盖 `_writeback_node_states_from_out` 不回落父类 rigid 逻辑
- source mask plural key 优先与 legacy fallback

---

## 12. 实现不变量（维护时请保持）

1. Stage4_6 不得恢复 rigid 专属 heads。
2. routed 规则必须在 source-frame world + segment_aabb 语义下执行。
3. `U_all` 与 `render_params_rigid_local_U` 行数必须严格对齐。
4. no-sky photometric 语义（non-sky-only）不可回退。
5. source 2D 回投统计口径保持 one-pass（`src_backproject_pass_count=1`）。

---

## 13. 一句话流程总览

`source views -> one-pass fused 2D backproject -> rigid source-world routing (in/out) -> shared heads predict offsets -> rigid world/local roundtrip writeback -> multi-target render -> non-sky loss -> proxy backward + node state sync`。

