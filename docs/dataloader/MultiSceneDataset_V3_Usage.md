# MultiSceneDataset V3 Usage

本文档说明 `MultiSceneDatasetV3` / `TrainSchedulerV3` 的设计与使用方式。  
v3 在 v2 基础上进一步将 scheduler 与 node-state 时间尺度联动正式化，核心目标是 **统一时间基准、确定性派生宏观调度、强化 block 级可解释日志**。

相关文档：

- `docs/dataloader/MultiSceneDataset_Usage.md`
- `docs/dataloader/MultiSceneDataset_V2_Usage.md`
- `docs/trainers/StreetForward_Scheduler_V2_Design.md`
- `docs/trainers/StreetForward_Scheduler_V2_Usage.md`

---

## 1. 设计目标

Scheduler v3 的目标是把训练调度从“只管理 source/target 采样”升级为：

> 统一管理 source block、target 刷新、node state update/reset、segment budget、epoch 与评估节奏的多时间尺度调度器。

需要满足：

1. 时间尺度统一：所有调度量围绕 `U = state_write_interval_steps` 联动。
2. block 语义明确：source 固定、target 按规则刷新、state 在固定时间尺度内演化。
3. 宏观调度确定性：`S/K/R` 可随 segment 变化，但必须是派生量，不做随机噪声。
4. 日志可解释：每个 source block 输出 begin/end 日志。
5. 配置可复用：one-segment、single-scene、multi-scene 共用一套机制。

---

## 2. 核心原则

### 2.1 `S/K/R` 是派生量，不是随机量

建议保留随机性的部分：

- scene / segment 顺序（可 shuffle）
- source keyframe / source frame 采样
- extra target 采样

不建议随机波动的宏观 protocol 量：

- `steps_per_segment`
- `source_hold_steps`
- `reset_interval`

原因：避免训练 protocol 漂移、实验不可比、loss 抖动来源不可解释。

### 2.2 所有时间量优先定义在 update units

设：

- `U = state_write_interval_steps`

在 update units 中定义：

- `S_u`：每 segment 预算（update ticks）
- `K_u`：每 source block 长度（update ticks）
- `R_u`：reset episode 长度（update ticks）
- `T_u`：target hold 周期（update ticks）

映射到 raw steps：

- `S = S_u * U`
- `K = K_u * U`
- `R = R_u * U`
- `T = T_u * U`

当 `U=2` 且 update-unit 参数不变时，`S/K/R/T` raw-step 自动 `x2`。

### 2.3 默认 reset 与 source block 对齐

baseline 建议：

- `R_u = K_u`
- `reset.policy = align_with_source_block`

这使每个 source block 都具备一致的“state 起点 -> 演化 -> 结束”语义，最稳定且最易 debug。

---

## 3. 时间尺度层次

v3 明确区分 5 层：

1. **raw step**：`forward/backward/optimizer.step`
2. **update tick**：每 `U` 个 raw step 触发一次 state write-back
3. **target block**：target 在 `T_u` 个 update ticks 内保持不变
4. **source block**：source 在 `K_u` 个 update ticks 内固定
5. **reset episode**：state 在 `R_u` 个 update ticks 内连续演化

segment budget 为 `S_u` 个 update ticks，并优先对齐完整 source blocks。

---

## 4. 派生量定义（可实现）

### 4.1 基础配置量（人工配置）

- `time_base.state_write_interval_steps` (`U`)
- `segment_budget.alpha_updates_per_keyframe`
- `segment_budget.min_updates_per_segment`
- `segment_budget.max_updates_per_segment`
- `source_block.target_source_blocks_per_segment`
- `source_block.min_updates_per_source_block`
- `source_block.max_updates_per_source_block`
- `reset.policy`
- `reset.reset_blocks_per_episode`
- `target_sampling.target_hold_updates`

### 4.2 segment 预算 `S_u`

```python
S_u_raw = clamp(
    round(alpha_updates_per_keyframe * num_keyframes),
    min_updates_per_segment,
    max_updates_per_segment,
)
```

### 4.3 source block 长度 `K_u`

```python
K_u_raw = round(S_u_raw / target_source_blocks_per_segment)
K_u = clamp(K_u_raw, min_updates_per_source_block, max_updates_per_source_block)
```

### 4.4 block 数与最终预算

为尽量使用完整 block：

```python
B_seg = max(1, ceil(S_u_raw / K_u))
S_u_final = B_seg * K_u
```

### 4.5 reset 周期 `R_u`

默认：

```python
R_u = K_u
```

一般式：

```python
R_u = reset_blocks_per_episode * K_u
```

### 4.6 target hold 周期 `T_u`

```python
T_u = target_hold_updates
T = T_u * U
```

默认建议 `T_u = 1`，即 target 刷新节奏与 update tick 对齐。

---

## 5. 采样策略

### 5.1 source：`shuffled_keyframe_cycle`

对每个 segment、每个 epoch：

1. 取 segment 全部 keyframes
2. 打乱形成循环列表
3. 每个 source block 依次取下一个 keyframe
4. keyframe 用尽后重新打乱继续

在选中的 keyframe 内均匀随机采样 1 个 frame 作为 source。

优势：覆盖更均匀、实现简单、无需引入复杂 overlap/pose 统计。

### 5.2 target：包含 source + extra

规则：

- target 必须包含 source frame（建议放 `target[0]`）
- extra target 优先从非 source keyframe 无放回选择
- 不足时允许有放回补齐
- 每个选中 keyframe 内均匀随机采样 1 个 frame
- target 在一个 target block 内固定，每 `T_u` update ticks 刷新一次

### 5.3 有效 target 数允许下降

短 segment 可允许：

```python
effective_target_frames_total <= min(config_total_target_frames, num_keyframes)
```

避免对小样本 segment 强行凑满导致采样异常。

---

## 6. Reset 策略

### 6.1 `align_with_source_block`（默认）

```python
R_u = K_u
```

适用于 one-segment overfit 与 early debug，最稳、最可解释。

### 6.2 `every_n_source_blocks`

```python
R_u = reset_blocks_per_episode * K_u  # reset_blocks_per_episode > 1
```

用于在 baseline 稳定后测试 memory carry-over 收益。

### 6.3 `segment_only`

仅 segment 开始 reset，一般不建议作为 baseline 默认策略。

---

## 7. 日志设计（每个 block 必打）

建议支持 console + JSONL + TensorBoard，其中 JSONL 为主格式。

### 7.1 segment begin log（每 segment 一次）

建议字段：

- `epoch_idx`
- `global_step_begin`
- `scene_id`
- `segment_id`
- `num_keyframes`
- `S_u_raw`
- `S_u_final`
- `S_raw_steps`
- `U`
- `K_u`
- `K_steps`
- `B_seg`
- `R_u`
- `R_steps`
- `T_u`
- `T_steps`
- `reset_policy`
- `mode`

### 7.2 source block begin log（每 block 一次）

建议字段：

- `epoch_idx`
- `global_step`
- `scene_id`
- `segment_id`
- `block_idx_in_segment`
- `block_idx_global`
- `source_keyframe_idx`
- `source_frame_idx`
- `block_update_span`
- `block_raw_step_span`
- `reset_episode_idx`
- `state_reset_this_block`
- `num_target_frames_total`
- `effective_target_frames_total`

### 7.3 source block end log（每 block 一次）

建议字段：

- `epoch_idx`
- `global_step`
- `scene_id`
- `segment_id`
- `block_idx_in_segment`
- `source_keyframe_idx`
- `source_frame_idx`
- `mean_loss`
- `mean_rgb_loss`
- `mean_ssim_loss`
- `mean_mask_loss`
- `mean_psnr`
- `num_target_refreshes_in_block`
- `num_state_updates_in_block`
- `num_optimizer_steps_in_block`
- `elapsed_time_sec`
- `diagnostics_summary`

### 7.4 reset event log（发生 reset 时）

建议字段：

- `global_step`
- `scene_id`
- `segment_id`
- `reset_episode_idx`
- `reason`（`segment_enter` / `source_block_boundary` / `policy_every_n_blocks`）

---

## 8. Scheduler v3 主流程

### 8.1 Build epoch plan

对每个 segment：

1. 读取 `num_keyframes`
2. 计算 `S_u_raw`
3. 计算 `K_u`
4. 计算 `B_seg`
5. 计算 `S_u_final = B_seg * K_u`
6. 计算 `R_u`
7. 形成 plan item

### 8.2 Enter segment

1. 初始化 segment state
2. 构造 `source_keyframe_cycle`
3. 输出 `segment_begin_log`
4. 按策略 reset
5. 进入第一个 source block

### 8.3 Enter source block

1. 从 cycle 取 source keyframe
2. 在 keyframe 内采 source frame
3. 按策略判断是否 reset
4. 采样 target（source + extras）
5. 输出 `block_begin_log`

### 8.4 Block 内执行

每个 block 执行 `K_u` 个 update ticks；每个 tick 包含 `U` 个 raw steps：

- 每 `T_u` ticks 刷新 target
- 每个 raw step 正常优化
- 每个 update tick 末尾做 state write-back

### 8.5 Exit block / segment

- block 结束：聚合统计并输出 `block_end_log`
- segment 的 `B_seg` 个 blocks 结束后切换下一 segment

---

## 9. 推荐配置模板（v3）

```yaml
scheduler_v3:
  enable: true

  time_base:
    state_write_interval_steps: 1
    use_update_units: true

  segment_budget:
    alpha_updates_per_keyframe: 4
    min_updates_per_segment: 12
    max_updates_per_segment: 48
    align_budget_to_full_blocks: true

  source_block:
    target_source_blocks_per_segment: 4
    min_updates_per_source_block: 2
    max_updates_per_source_block: 12
    source_keyframe_policy: shuffled_cycle
    source_frame_policy: uniform_in_keyframe

  target_sampling:
    include_source: true
    total_target_frames: 3
    allow_effective_target_drop: true
    target_hold_updates: 1
    extra_target_keyframe_policy: uniform_without_replacement
    extra_target_frame_policy: uniform_in_keyframe

  reset:
    policy: align_with_source_block
    reset_blocks_per_episode: 1
    reset_at_segment_start: true
    reset_at_segment_end: false

  ordering:
    shuffle_scenes_each_epoch: true
    shuffle_segments_each_epoch: true

  logging:
    console: true
    jsonl: true
    tensorboard: true
    log_segment_begin: true
    log_block_begin: true
    log_block_end: true
    log_reset_events: true
```

---

## 10. Stage4.1 one-segment baseline 建议值

针对 one-segment overfit / stability monitor，建议：

- `state_write_interval_steps: 1`
- `target_source_blocks_per_segment: 4`
- `min_updates_per_source_block: 2`
- `max_updates_per_source_block: 8`
- `target_hold_updates: 1`
- `reset.policy: align_with_source_block`
- `reset.reset_blocks_per_episode: 1`

语义：

- `U=1`：每步 write-back
- 每 segment 每 epoch 大约 4 个 source blocks
- 每个 source block 开始 reset
- target 每个 update tick 刷新（此时等价每步刷新）

---

## 11. `U=2` 的自动联动示例

若只改：

```yaml
state_write_interval_steps: 2
```

且其它 update-unit 参数不变，则自动得到：

- `S = 2 * S_u`
- `K = 2 * K_u`
- `R = 2 * R_u`
- `T = 2 * T_u`

例如 `K_u=4, R_u=4, T_u=1` 时，raw-step 变为 `K=8, R=8, T=2`。  
即每 2 步 update state、每 8 步换 source/reset、每 2 步刷新 target。

---

## 12. v2 -> v3 关键升级

v2 已具备 epoch、source hold、target 包含 source 等能力；v3 的关键升级是：

1. 以 `U` 作为统一时间基准。
2. 所有关键调度量先定义在 update units。
3. `S/K/R/T` 全部作为确定性派生量。
4. block/reset/logging 边界与 state update 对齐。
5. 更适合 stateful 训练行为分析与归因。

---

## 13. 落地约束（实现时必须满足）

1. 日志必须可复原宏观调度：至少输出 `U/S_u/K_u/R_u/T_u` 及对应 raw-step 值。
2. 不要在 epoch 内随机抖动 `S/K/R`。
3. block 日志必须以 source block 为单位。
4. reset 事件必须显式记录。
5. target 刷新边界必须与 update tick 对齐。
6. segment budget 优先对齐完整 block。

---

## 14. 一句话方案

Scheduler v3 的主线是：  
以 `U = state_write_interval_steps` 为统一时间基准，将 segment budget `S`、source block 长度 `K`、reset 周期 `R`、target hold 周期 `T` 全部定义在 update units，再映射到 raw steps；其中 `S/K/R` 随 segment 确定性变化，不做随机化；每个 source block 必须输出 begin/end 日志。

