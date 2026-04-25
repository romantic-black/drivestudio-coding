# StreetForward Stage5_2 设计方案（Full-Branch Memory-Gated Routed Update）

> 参考实现与约束来源：
> - `models/streetforward/minimal_trainer_stage5_0.py`
> - `models/feature_extractors/alpha_t_extractor_v3.py`
> - `datasets/train_scheduler_v8.py`
> - `datasets/validation_scheduler_v8.py`
> - `datasets/multi_scene_dataset_v4.py`

---

## 0. 一句话定义

`Stage5_2 = Stage5_0 + full-branch history memory + update gate + near/far 分离解码。`

严格边界：

- 不恢复 rigid-specific head。
- 不把 distant 强塞进 near xCPE spconv。
- 保持 Stage4_6 路由语义：`rigid_in -> bg heads`，`rigid_out -> distant heads`。

---

## 1. 设计定位与现状对齐

### 1.1 Stage5_0 已有能力

`Stage5_0` 在 `minimal_trainer_stage5_0.py` 已明确 fast-fail：

- `struct_decoder.type == xcpe`
- `scope == bg_rigid_in`
- `include_distant == false`
- `include_rigid_out == false`
- `output_role == gru_input`
- `clamp_grid_coord == false`

并且 `bg + rigid_in` 进入 `StreetForwardXCPEDecoder`，切分后只替换 GRU 输入，不改 routed heads 语义。

### 1.2 Scheduler V8 已有约束

`train_scheduler_v8.py` 已强约束：

- `target_policy == visited_episode_frames`
- `total_target_frames <= blocks_per_episode`
- `include_source_frame == true`
- `reset_policy == episode_end`

`validation_scheduler_v8.py` 也按 visited 语义构造 visit target windows。

因此 Stage5_2 的关键不在 2D 主干重写，而在“全分支记忆与更新控制”。

---

## 2. Stage5_2 核心结构

建议 forward 拆为五层：

1. one-pass routed 2D evidence
2. branch-specific token building
3. near/far structural decoding
4. memory-conditioned keep/update gate
5. routed GRU + shared heads update

数据流：

```text
source images
  -> one-pass 2D lifting (bg / distant / rigid_S)
  -> rigid route split (rigid_in / rigid_out)
  -> near decoder:  (bg + rigid_in)  -> xCPE
  -> far decoder:   (distant + rigid_out) -> lightweight MLP
  -> history memory (support/error/update_norm EMA)
  -> point scalar gate
  -> shared heads writeback
```

---

## 3. 分支语义（必须保持）

### 3.1 Near branch

- `bg` + `rigid_in`
- 使用 xCPE（point-preserving）
- `rigid_in` 仍走 bg shared heads

### 3.2 Far branch

- `distant` + `rigid_out`
- 不进 xCPE
- 使用 lightweight token MLP
- `rigid_out` 仍走 distant shared heads

### 3.3 Rigid memory 存储原则

`history_rigid` 仅按 rigid local row 存一套，不拆 in/out 两套。  
同一 rigid point 在不同 source frame 可切换 in/out，但身份不变。

---

## 4. History memory 设计

每个分支一套：

- `support_ema: [N, 1]`
- `error_ema: [N, 1]`
- `update_norm_ema: [N, 1]`
- `initialized: [N, 1]`

覆盖：

- `history_bg`
- `history_distant`
- `history_rigid`

---

## 5. Record pass 与触发时机

### 5.1 记录视图来源

record pass 使用当前 batch 的：

- `batch["request_meta"]["target_image_refs"]`

`MultiSceneDatasetV4` 当前 batch 已稳定提供 `source_image_refs` 与 `target_image_refs`，可直接复用。

### 5.2 触发点建议

推荐语义：`block_exit`。  
原因：`step_major` 下 block 会 re-enter，只在 `block_end` 更新会偏晚。

落地优先级：

1. 推荐：在 `TrainSchedulerV8` 切块前新增 `block_exit` 事件。
2. 最小侵入：训练 loop 通过 `(scene_id, segment_id, block_idx_global, source_image_ref)` 变化检测 block switch，并在切换时对上一块做 record。

### 5.3 Record pass 约束

- `torch.no_grad()`
- 不 `backward`
- 不 `optimizer.step`
- 不修改 node state 参数
- 只写 history EMA

### 5.4 Record pass 必须独立于 2D feature lifting

record pass 不能直接复用 `_compute_2d_features_all_branches_once_routed()`。  
原因：该函数语义是 source views 的 CNN feature backprojection，不是 target residual 回投。

Stage5_2 建议新增独立接口：

- `_compute_record_support_error_all_branches_once_routed(...)`

输入：

- routed node states
- `target_views` / `target_images`
- dynamic info
- render size

输出（建议统一全量 row-space）：

- `support_bg, error_bg`（`[N_bg,1]`）
- `support_distant, error_distant`（`[N_distant,1]`）
- `support_rigid, error_rigid`（`[N_rigid,1]`）

其中 rigid 建议直接输出 `[N_rigid,1]`，避免把 source route (`S_in/S_out`) 与 target frame 的可见性混淆。

---

## 6. 记录量定义

### 6.1 support

基于 alpha-T 累计权重：

- `s_i = sum(w_ivp)`
- `w_ivp = T_ivp * alpha_ivp`
- squash 建议 `log1p(s_i)`

### 6.2 error

先渲染 target views，再算像素 residual（v1 用 RGB L1），最后用 alpha-T 权重回投到点：

- `e_i = sum(w * E) / (sum(w) + eps)`

### 6.3 update_norm

v1 只记录 means 更新量（与验证偏移不足直接相关）：

- 从 train forward 实际 writeback 的 `delta_means` 统计
- 仅对本步写入点更新 `update_norm_ema`

---

## 7. EMA 更新规则

- support：始终更新（但建议 visible/invisible 使用不同 beta）
- error：仅 `support > tau` 时更新（避免不可见点被误写为 0）
- update_norm：仅对被写入点更新

建议：

- `support_beta_visible = 0.90`
- `support_beta_invisible = 0.98`
- `error_beta = 0.90`
- `update_norm_beta = 0.95`
- `error_eps = 1e-6`

如果第一版不想拆太细，统一 `ema_beta` 也建议从 `0.95` 起步；在 `step_major_switch_interval_steps` 较小、record 频繁时，`0.90` 往往衰减过快。

---

## 8. Gate 设计（v1）

`g_i` 为 point scalar update gate：

- 大 -> 更允许更新
- 小 -> 更保留旧状态

### 8.1 Gate 输入必须包含 initialized（可选再加 visible_now）

仅输入 `error_ema` 会有冷启动歧义：未观测点的 `error_ema=0` 会被误判为“误差低”。  
因此 `initialized` 必须显式进入 gate。

建议：

- `history_raw = cat([support_ema, error_ema, update_norm_ema, initialized.float()], dim=-1)`
- 可选追加 `visible_now = (acc_w > support_min).float()`

即 v1 gate history 至少 4 维，推荐 5 维（含 `visible_now`）。

### 8.2 gate 与 mask_update 必须严格绑定

实现时不允许只用 `g_min` 直接更新全量点；`mask_update=false` 的点必须完全不变。

最终有效 gate：

- `effective_gate = gate * mask_update.float().unsqueeze(-1)`

应用：

- `delta_theta = effective_gate * delta_theta_hat`
- `h_new = (1 - effective_gate) * h_old + effective_gate * h_candidate`

对 `mask_update=false` 的位置建议增加断言：

- `assert allclose(h_new[~mask], h_old[~mask])`

为避免初期锁死：

- `g = g_min + (1-g_min) * sigmoid(a)`
- `min_gate = 0.05`
- `init_bias = 2.0`（初始接近 Stage5_0 行为）

v1 gate 输入建议（先保持简洁但不丢关键信号）：

- current_feat
- hidden_state
- history_embed（必须覆盖 initialized，建议再覆盖 visible_now）
- param_embed
- branch_embed

### 8.3 GRU candidate 与 gate 应用顺序

需要明确两层语义，避免与 `_predict_offsets_gru_with_heads(..., mask_update=...)` 内部逻辑重复冲突：

1. `_predict_candidate_offsets_gru_with_heads(...)` 产生 `delta_hat` 与 `h_candidate`
2. `_apply_mask_and_gate(...)` 用 `effective_gate` 得到最终 `delta/h_new`

若短期不拆函数，至少保证：

- `h_candidate = h_new_from_predict`
- 再执行 `effective_gate` 融合
- 对 `mask_update=false` 做不变性断言

---

## 9. Near/Far 解码器

### 9.1 Near decoder（复用 Stage5_0 xCPE）

scope: `bg + rigid_in`

token 组成：

- 2D feat
- support embed
- param embed
- branch embed
- history embed

branch id：

- `0 = bg`
- `1 = rigid_in`

### 9.2 Far decoder（新增 lightweight MLP）

scope: `distant + rigid_out`

第一版结构：

- `Linear(token_dim, hidden)`
- `GELU`
- `LayerNorm`
- `Linear(hidden, fused_in_dim)`

branch id：

- `2 = distant`
- `3 = rigid_out`

v1 不引入 attention/KNN/pooling。

---

## 10. Trainer 结构改造建议

新增 trainer：

- `models/streetforward/minimal_trainer_stage5_2.py`

建议继承（关键修正）：

- `MinimalStreetForwardStage5_2(MinimalStreetForwardStage4_6)`

不建议直接走 `Stage5_0.__init__`，因为其 `_validate_stage5_0_config` 强制：

- `model.stage == 5_0`
- `struct_decoder.type == xcpe`
- `scope == bg_rigid_in`
- `include_distant == false`
- `include_rigid_out == false`

与 Stage5_2 的 `routed_near_far/full_routed/include_distant/include_rigid_out` 目标配置冲突。

推荐初始化流程：

1. `_validate_stage5_2_config(config)`
2. `super().__init__(...)`（Stage4_6）
3. `_init_stage5_2_modules(config)`
4. `_rebuild_optimizer_after_stage5_modules()`

建议新增/覆盖：

- `_validate_stage5_2_config`
- `_init_stage5_2_modules`
- `_compute_full_routed_gru_inputs`
- `_record_block_history`
- `_apply_history_update`
- `_apply_update_gate`

建议新增 dataclass：

- `FullRoutedGRUInputs`
  - `feat_bg_input`
  - `feat_distant_input`
  - `feat_rigid_in_input_all`
  - `feat_rigid_out_input_all`
  - `gate_bg/distant/rigid_in/rigid_out`
  - `aux`

---

## 11. 配置草案（v1）

```yaml
model:
  stage: "5_2"
  struct_decoder:
    enable: true
    type: "routed_near_far"
    scope: "full_routed"
    output_role: "gru_input"
    point_preserving: true
    include_bg: true
    include_distant: true
    include_rigid_in: true
    include_rigid_out: true
    near:
      type: "xcpe"
      branches: ["bg", "rigid_in"]
      sparse_backend: "spconv"
      clamp_grid_coord: false
    far:
      type: "mlp"
      branches: ["distant", "rigid_out"]
  history_memory:
    enable: true
    record_on: "block_exit"
    record_views: "source_image_refs"
    support_beta_visible: 0.90
    support_beta_invisible: 0.98
    error_beta: 0.90
    update_norm_beta: 0.95
  update_gate:
    enable: true
    min_gate: 0.05
    init_bias: 2.0
    warmup_steps: 1000
    require_initialized_in_input: true
    include_visible_now: true
    bind_with_mask_update: true
```

---

## 12. 分阶段实验策略

### 12.1 Stage5_2-a（先打通 full branch 输入）

- 开 far MLP
- 关 history
- 关 gate

关注：

- distant/rigid_out update count
- distant loss 是否稳定

### 12.2 Stage5_2-b（只记录 memory）

- 开 history record
- forward 不使用 history
- 关 gate

关注：

- support/error/update_norm 统计是否符合可见性逻辑

### 12.3 Stage5_2-c（开启 gate）

- history 参与 forward
- 开 gate（带 warmup）

关注：

- gate 均值与低值比例
- validation offset norm
- PSNR / SSIM / LPIPS

---

## 13. 关键观测指标（必须）

- near/far 点计数：`bg, rigid_in, distant, rigid_out`
- gate 均值：`gate_bg/distant/rigid_in/rigid_out`
- history 均值：`support/error/update_norm`（三分支）
- record pass 次数、record view 数
- rigid_in 与 rigid_out 更新是否失衡

---

## 14. Fast-Fail 清单（Stage5_2）

1. `model.stage == 5_2`
2. `rigid_routed.inside_decoder == bg`
3. `rigid_routed.outside_decoder == distant`
4. 禁止 rigid-specific heads 配置出现
5. `struct_decoder.type == routed_near_far`
6. `near.branches == [bg, rigid_in]`
7. `far.branches == [distant, rigid_out]`
8. `far.type == mlp`（v1）
9. 禁止 distant 进入 near xCPE
10. 禁止 `clamp_grid_coord == true`
11. `history_memory.record_on == block_exit`（v1）
12. `scheduler_v8.target_policy == visited_episode_frames`
13. gate 输入必须包含 `initialized`（可选再含 `visible_now`）
14. gate 应用必须与 `mask_update` 相乘，`mask_update=false` 点严格不变

任一不满足立即报错，不做隐式降级。

---

## 15. 实施优先级建议

1. 先完成 `Stage5_2-a` 跑通（确保 full branch 输入链路正确）。
2. 再接 `record pass`（仅记录，不参与 forward）。
3. 最后引入 gate 并做 warmup。

该顺序可最大化 fast-fail 能力，减少一次性引入过多变量导致的定位成本。

---

## 16. 结论

Stage5_2 不追求“所有点统一进同一个结构网络”，而是追求“所有分支都具备历史质量判断与更新控制”。  
它在保持 Stage4_6 路由语义与 Stage5_0 near xCPE 优势的同时，把 distant/rigid_out 纳入可控的 recurrent 更新框架，目标直指“训练块内表现好但验证偏移不足”的问题。
