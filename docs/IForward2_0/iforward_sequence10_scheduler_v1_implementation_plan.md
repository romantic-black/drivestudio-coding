# IForward Sequence10 Scheduler v1 详细实现方案

## 一个面向 10 帧短序列重建的全新、简化 scheduler

目标版本：`iforward_sequence10_v1`  
适用模型：IForward Stage 2.1（FWHR + Parent PTv3 + Parent Temporal Mamba + GRLD）  
基线代码：`drivestudio_stage6_refactor_context_20260622_v33`  
训练方式：从零训练，不依赖 checkpoint  

---

# 0. 核心结论

当前 scheduler 同时承担：

```text
scene/segment traversal
4-block episode 构造
window selection
shape sampling
blocks/repeats 预算绑定
history ref sampling
memory commit policy
coverage 计数
tail/circular fill
多个 scheduler version 兼容
```

这造成三个问题：

```text
1. sequence length、repeat budget、BPTT 长度和监督范围被 shape 名字耦合。
2. 第一次时序观测和随机优化重访没有明确区分。
3. training / validation 都复用随机 scheduler，导致 shape 或协议可能没有真正逐项执行。
```

新 scheduler 不再围绕：

```text
b1_r8 / b2_r4 / b3_r3 / r8b1 / r4b2 / r2b4
```

构建训练任务，而只保留两个训练模式：

```text
A. Bootstrap：单帧 updater 学习
B. Sequence10：10 帧 causal assimilation + 可选 randomized repair
```

Sequence10 episode 的固定结构：

```text
选择同一 segment 内 10 个唯一 keyframe blocks
stride = 1 或 2

Causal pass：
    5 个 rollout
    每个 rollout 包含 2 个连续 block
    每个 block repeat 4 次
    即固定 B2×R4，inner_K=8
    chronological
    temporal read=true
    temporal commit=true

Optional repair pass：
    1 个 rollout
    随机排列全部 10 个 block
    每个 block repeat 1 次
    即 B10×R1，inner_K=10
    temporal read=true
    temporal commit=false
```

新 scheduler 的核心原则：

```text
1. segment、episode、keyframe block 定义保持不变。
2. sequence 是主训练维度；repeat 只是单帧计算预算。
3. 第一次观测按真实时间顺序进入 Temporal Mamba。
4. 随机 repair 只优化 GS state，不污染 Temporal Mamba。
5. 10 帧内 history supervision 使用全部已见帧。
6. rollout 边界 detach graph，但不 reset GS / Temporal state。
7. validation 使用固定 manifest 和显式协议 runner，不复用随机训练 scheduler。
```

---

# 1. 当前 scheduler 为什么应被替换

## 1.1 Shape 同时编码了多个不同概念

当前：

```text
b1_r8
b2_r4
b3_r3
```

同时决定：

```text
- 一个 rollout 中有几帧
- 每帧优化几次
- graph 中跨几帧有梯度
- 总 inner_K
- history refs 的生成时机
- temporal commit 次数
```

但这些不是同一个维度。

例如：

```text
b1_r8：主要训练单帧迭代收敛
b2_r4：主要训练两帧 temporal credit assignment
b3_r3：同时降低每帧优化预算并增加时序跨度
```

将这些 shape 直接混合，使模型在每个 optimizer step 面对不同任务，却没有明确课程目标。

## 1.2 当前 scheduler 的 policy 开关过多

当前配置包含：

```text
block_selection_policy
start_offset_policy
delivery_order_policy
tail_policy
allow_short_final_rollout
avoid_single_block_tail
fixed_shape_names
shapes
shapes_schedule
target_repeats_per_block
rollouts_per_episode
blocks_per_episode
```

这些开关能表达很多组合，但也允许产生含义不清或互相冲突的组合。

新 scheduler 将不再暴露这些低层 policy。

## 1.3 当前 Temporal commit 无法表达 repair

Stage 2.1 当前在每个 block exit 无条件 commit parent temporal state。

随机重访若继续使用同一路径，会把：

```text
随机优化顺序
```

错误写成：

```text
真实物理时间
```

新 scheduler 必须显式产生：

```text
temporal_read
temporal_commit
physical_time_advance
visit_kind
```

而不是从 repeat_idx / block_exit 隐式推导。

## 1.4 当前 validation 与 scheduler 耦合过深

将多个 validation shape 放入 scheduler sampling pool，并不能保证它们都执行；不同协议还可能共享已经被前一协议修改的 state。

新 validation 必须：

```text
固定 manifest
显式循环协议
每个协议独立 reset / clone state
```

---

# 2. 目标与非目标

## 2.1 目标

```text
1. 从零训练一个稳定的 tied-weight iterative updater。
2. 单帧至少具备 R4 的可靠 current reconstruction。
3. 在 10 帧 chronological 输入中保持 persistent GS 和 Temporal state。
4. 学会在随机重访顺序下修复整个 10 帧 scene state。
5. history loss 覆盖全部 10 帧，而不是只覆盖最近 1~2 帧。
6. validation 能独立测量 causal、repair、order robustness 和 repeat stability。
7. scheduler 本身易读、可复现、可恢复、可测试。
```

## 2.2 非目标

```text
- 本版本不训练 30 帧中序列。
- 不支持任意无序照片集模式。
- 不做 r8b1/r4b2/r2b4 训练混合。
- 不做 circular fill。
- 不允许重复 block 凑足 10 帧。
- 不在 repair pass 中更新 Temporal Mamba。
- 不实现自适应收敛停止。
- 不以 teacher 或 per-scene optimization 作为训练组成。
```

---

# 3. 新 scheduler 的概念模型

新 scheduler 只包含四个对象：

```text
SequenceSpec
EpisodePlan
RolloutPlan
VisitSpec
```

## 3.1 SequenceSpec

表示从一个 segment 选择出的固定 10 帧序列：

```python
@dataclass(frozen=True)
class Sequence10Spec:
    scene_id: int
    segment_id: int
    sequence_id: int
    stride: int                    # 1 or 2
    start_block_pos: int
    block_ids: tuple[int, ...]     # len=10, unique
    keyframe_indices: tuple[int, ...]
    source_frame_indices: tuple[int, ...]
    source_refs_by_pos: tuple[tuple[ImageRef, ...], ...]
```

要求：

```text
len(block_ids) == 10
所有 block 唯一
block position 差严格等于 stride
无 wraparound
无 circular fill
```

## 3.2 VisitSpec

一个 block 的一次逻辑访问：

```python
@dataclass(frozen=True)
class Sequence10VisitSpec:
    sequence_pos: int
    block_id: int
    source_frame_idx: int
    source_keyframe_idx: int
    evidence_refs: tuple[ImageRef, ...]

    visit_kind: str
    # bootstrap | causal_first | repair

    repeat_budget: int
    frame_gap: int

    temporal_read: bool
    temporal_commit: bool
    optimizer_memory_commit: bool
    observation_commit: bool
    physical_time_advance: bool

    is_first_visit: bool
```

## 3.3 RolloutPlan

```python
@dataclass(frozen=True)
class Sequence10RolloutPlan:
    phase: str
    # bootstrap | causal | repair

    rollout_idx_in_episode: int
    visits: tuple[Sequence10VisitSpec, ...]

    current_positions: tuple[int, ...]
    history_positions: tuple[int, ...]
    full_scene_positions: tuple[int, ...]

    reset_state_before: bool
    carry_state_after: bool
    detach_graph_after: bool
    episode_end_after: bool

    repair_permutation_hash: int = -1
```

## 3.4 EpisodePlan

```python
@dataclass
class Sequence10EpisodePlan:
    episode_id: int
    mode: str              # bootstrap | sequence10
    sequence: Sequence10Spec | None
    rollouts: list[Sequence10RolloutPlan]
    next_rollout_idx: int
    rng_snapshot: object
```

---

# 4. Curriculum

只保留三个 step 阈值：

```text
bootstrap_end_step
repair_start_step
history_damage_start_step
```

## 4.1 Phase A：Single-frame Bootstrap

默认：

```text
step 0 ~ 4999
```

每个 episode 只有一个 rollout：

```text
1 block
repeat budget 随机：R4 / R6 / R8
```

推荐概率：

```text
R4: 0.60
R6: 0.30
R8: 0.10
```

行为：

```text
reset state before rollout
current-only supervision
Temporal state允许初始化/读写，但没有跨帧价值
无 history refs
```

目的：

```text
- 学会可靠 current update
- 学会不同 iteration budget
- 避免从零直接面对 10 帧状态保持
```

## 4.2 Phase B：Sequence10 Causal

默认：

```text
step 5000 ~ 14999
```

每个 episode：

```text
10 unique blocks
5 causal rollouts
每 rollout = B2×R4
```

序列：

```text
[0,1]
[2,3]
[4,5]
[6,7]
[8,9]
```

每个 block：

```text
first visit
chronological
Temporal read=true
Temporal commit=true at block exit
physical_time_advance=true
```

History：

```text
每个 rollout final监督当前2帧
以及此前全部已见帧
```

## 4.3 Phase C：Sequence10 + Repair

默认：

```text
step 15000+
```

Causal pass 不变。

每个 episode 以概率 `repair_probability` 加一个 repair rollout：

```text
B10×R1
随机排列全部10帧
inner_K=10
```

推荐：

```text
repair_probability = 0.5
```

原因：

```text
50% causal-only：模型不能依赖后续一定有人修复
50% causal+repair：训练顺序鲁棒性和全局修复
```

Repair：

```text
temporal_read=true
temporal_commit=false
optimizer_memory_commit=false
observation_commit=false
physical_time_advance=false
visit_kind=repair
```

GS state 仍然按随机顺序更新；Temporal state 保持 causal pass 结束时的值。

---

# 5. 10 帧序列采样

## 5.1 Stride

```yaml
sequence:
  length: 10
  stride_probs:
    1: 0.7
    2: 0.3
```

Stride 1 要求 segment 至少有：

```text
10 个可用 block
```

Stride 2 要求至少有：

```text
19 个 block position
```

## 5.2 Eligibility index

Scheduler 初始化时预计算：

```python
eligible[(scene_id, segment_id, stride)] = [valid_start_positions]
```

预计算只读取 segment index，不加载完整图像/点云。

若随机选择的 stride 不可用：

```text
优先 fallback 到另一个 stride
两个 stride 都不可用则跳过该 segment
```

不允许：

```text
wraparound
重复 block
short episode
circular fill
```

## 5.3 Source frame

保持现有已验证语义：

```text
每个 block 在 episode begin 选择一个 source frame
causal 和 repair 重访使用同一个 source frame
```

配置：

```yaml
source_frame_policy: random_within_keyframe_once_per_episode
```

这样 repair 测试的是 update order，而不是同时引入新的帧内采样变化。

---

# 6. Causal Rollout 生成

固定 5 个 rollout：

```text
rollout 0: positions [0,1]
rollout 1: positions [2,3]
rollout 2: positions [4,5]
rollout 3: positions [6,7]
rollout 4: positions [8,9]
```

每个 visit：

```text
repeat_budget = 4
```

每 rollout：

```text
blocks = 2
inner_K = 8
```

Step expansion：

```python
for rollout_block_rank, visit in enumerate(visits):
    for repeat_idx in range(4):
        emit step:
            block_enter = repeat_idx == 0
            block_exit = repeat_idx == 3
            observation_commit = repeat_idx == 0
            temporal_commit = repeat_idx == 3
            optimizer_memory_commit = repeat_idx == 3
```

注意：

```text
temporal_commit 是 VisitSpec 语义；
最终是否 commit 应由 model 显式读取该字段，不能再由 block_exit 隐式决定。
```

---

# 7. Repair Rollout 生成

## 7.1 Permutation

```python
permutation = rng.sample(range(10), 10)
```

约束：

```text
不能等于 chronological identity
所有位置唯一
```

可选 fail-fast：

```text
若 permutation 与 identity 相同则重新采样
```

## 7.2 Step expansion

每个 block：

```text
repeat_budget = 1
```

一个 rollout：

```text
blocks = 10
inner_K = 10
```

每 visit：

```text
is_first_visit = false
visit_kind = repair
temporal_read = true
temporal_commit = false
observation_commit = false
optimizer_memory_commit = false
physical_time_advance = false
```

## 7.3 Repair supervision

Final supervision：

```text
完整 10 帧全部作为 repair_all10
```

同时计算 episode-level damage：

```text
final frame loss vs causal pass 中该帧的 best detached loss
```

---

# 8. State 与 detach 语义

## 8.1 Episode begin

```text
reset LocalGSState
reset ParentTemporalState
reset parent runtime
reset history best-loss bank
```

## 8.2 Rollout boundary

```text
detach LocalGSState graph
保持数值

detach ParentTemporalState graph
保持数值

不 reset
```

## 8.3 Episode end

```text
丢弃 LocalGSState
丢弃 ParentTemporalState
丢弃 history bank
切换 scene/segment
```

## 8.4 Model 必须修改的 commit 逻辑

当前 Stage 2.1 在 `is_block_exit` 时无条件 commit Temporal Mamba。

必须改为：

```python
if is_block_exit and step.temporal_commit:
    parent_temporal_state = parent_temporal.commit(...)
```

无论 causal 还是 repair，block cache 都必须在 block exit 清空：

```python
if is_block_exit:
    if step.temporal_commit:
        commit(...)
    stage2_1_parent_block_cache = {}
```

Resolver 不再强制：

```text
update_optimizer_memory == is_block_exit
```

而是验证显式 flag 与 visit kind：

```text
causal_first：block exit必须 temporal_commit=true
repair：所有step必须 temporal_commit=false
```

---

# 9. 时间与访问类型输入

为了让模型区分：

```text
stride1 / stride2
first observation / repair revisit
```

新增轻量 embedding：

```text
frame_gap_embed: 4D
visit_kind_embed: 4D
```

输入 Parent Token Builder：

```text
frame_gap ∈ {0,1,2}
visit_kind ∈ {bootstrap, causal_first, repair}
```

Repair 的：

```text
frame_gap = 0
physical_time_advance = false
```

不要把随机 permutation 中相邻位置的 index 差作为 frame gap。

---

# 10. Supervision 与 History Loss

## 10.1 Bootstrap

```text
L = L_current + L_delta_reg
```

无 history fetch / render。

## 10.2 Causal rollout

假设当前 rollout 是 positions `[2j, 2j+1]`。

```text
current_positions = [2j, 2j+1]
history_positions = [0, ..., 2j-1]
```

Loss：

\[
L = L_{current}
  + \lambda_h L_{history}
  + L_{delta\_reg}.
\]

其中：

```text
L_current：当前两个 block
L_history：全部以前已见 block，不包括当前两个
```

History weight：

```yaml
history:
  target_weight: 0.5
  warmup_start_step: 5000
  warmup_steps: 10000
```

## 10.3 Repair rollout

```text
L_repair_all10 = mean(loss(frame_i), i=0..9)
```

Damage：

\[
L_{damage}
=
\frac{1}{10}\sum_i
\max(0, L_i^{final} - L_i^{best} - m).
\]

建议：

```yaml
damage:
  weight: 0.15
  margin: 0.002
  warmup_start_step: 15000
  warmup_steps: 10000
```

## 10.4 Best-loss bank

Episode 内维护：

```python
best_loss_by_pos: Tensor[10]
best_psnr_by_pos: Tensor[10]
seen_mask: BoolTensor[10]
```

更新时 detach：

```python
best_loss[pos] = min(best_loss[pos], current_loss.detach())
```

不将 bank 放入 optimizer state 或 checkpoint。

## 10.5 计算优化

Causal rollout history 最多 8 帧，用户当前目标只有 10 帧，因此 P0 可以全部监督。

但必须 grouped render：

```text
一次按所有 history refs grouped multiview render
```

并记录：

```text
history_num_frames
history_num_refs
history_render_ms
```

Bootstrap 阶段必须完全跳过 history fetch/render。

---

# 11. 新配置结构

新 scheduler 仅保留下列配置，不再支持旧 shape policy。

```yaml
scheduler_iforward:
  enable: true
  version: iforward_sequence10_v1
  fail_fast: true

  traversal:
    seed: 41
    scene_order: shuffle_per_epoch
    segment_order: shuffle_per_epoch
    traversal_mode: scene_round_robin_episode
    forbid_consecutive_same_scene: true

  curriculum:
    bootstrap_end_step: 5000
    repair_start_step: 15000

  bootstrap:
    repeat_budgets:
      - {repeats: 4, prob: 0.60}
      - {repeats: 6, prob: 0.30}
      - {repeats: 8, prob: 0.10}

  sequence:
    length: 10
    stride_probs:
      1: 0.70
      2: 0.30
    fallback_to_available_stride: true
    source_frame_policy: random_within_keyframe_once_per_episode
    require_unique_blocks: true
    allow_wraparound: false
    allow_short_sequence: false

  causal:
    blocks_per_rollout: 2
    repeats_per_block: 4
    temporal_read: true
    temporal_commit: true
    physical_time_advance: true

  repair:
    enable: true
    probability: 0.50
    blocks_per_rollout: 10
    repeats_per_block: 1
    order: random_permutation
    temporal_read: true
    temporal_commit: false
    physical_time_advance: false

  state:
    reset_at_episode_begin: true
    carry_across_rollouts: true
    detach_after_rollout: true
    reset_at_episode_end: true

  supervision:
    bootstrap_current_only: true
    causal_current: current_chunk
    causal_history: all_seen_excluding_current
    repair: all_10

  history_loss:
    target_weight: 0.5
    warmup_start_step: 5000
    warmup_steps: 10000

  damage_loss:
    enable: true
    target_weight: 0.15
    margin: 0.002
    warmup_start_step: 15000
    warmup_steps: 10000

  preload:
    warm_current_rollout: true
    warm_next_rollout: true
    warm_full_sequence_async: true
```

旧配置禁止项：

```text
blocks_per_episode
rollouts_per_episode
shapes
shapes_schedule
fixed_shape_names
target_repeats_per_block
block_selection_policy
start_offset_policy
tail_policy
circular_fill
avoid_single_block_tail
```

若新版本配置出现这些字段，fail-fast。

---

# 12. 代码结构

## 12.1 新文件

```text
datasets/train_scheduler_iforward_sequence10.py
models/iforward/sequence10_resolver.py
models/iforward/sequence10_batch.py
models/iforward/sequence10_history_bank.py
datasets/iforward_sequence10_validation.py
```

## 12.2 不直接重写训练器的兼容策略

第一版使用 adapter：

```text
Sequence10RolloutPlan
    -> Legacy IForwardRolloutPlan-compatible batch meta
    -> 现有 model/trainer
```

但新增字段必须保留：

```text
visit_kind
temporal_read
temporal_commit
physical_time_advance
frame_gap
sequence_pos
sequence_id
scheduler_phase
```

不能在 adapter 中丢弃。

## 12.3 Scheduler 分支入口

`TrainSchedulerIForward.__new__` 或训练工具中：

```python
if version == "iforward_sequence10_v1":
    scheduler = TrainSchedulerIForwardSequence10(...)
else:
    scheduler = TrainSchedulerIForward(...)
```

不继续向旧 scheduler 添加 `if version == ...` 分支。

## 12.4 Resolver

新增：

```python
IForwardSequence10Resolver
```

它只验证新协议，不包含旧 v1/v3/v4 policy。

---

# 13. Resume / state_dict

Scheduler checkpoint 必须保存：

```text
global_step
epoch_idx
episode_id_next
rollout_id_global
scene traversal queue
segment traversal queue
RNG state
current episode plan
current rollout index
sequence source frame choices
repair enabled flag
repair permutation
```

Resume 后下一个 batch 必须与未中断运行完全一致。

禁止在 resume 时重新采样 repair permutation 或 source frame。

---

# 14. Preload 与 Fetch

10 帧 episode 一旦生成，全部 refs 已知。

Preload：

```text
episode begin：异步 enqueue 10帧 image/meta/segment refs
当前 rollout：确保 current refs ready
每 rollout结束：warm next rollout refs
repair：复用已加载10帧
```

重要：

```text
preload hint 构建不能同步 resolve 完整 segment bundle
```

只传轻量 asset identifiers，由 worker 加载。

日志：

```text
preload_episode_submit_ms
preload_current_wait_ms
preload_next_hit_ratio
batch_fetch_ms
sequence_cache_hit_ratio
```

---

# 15. 日志系统

## 15.1 每 rollout scheduler row

新增 split：

```text
iforward_sequence10_scheduler
```

字段：

```text
step
epoch_idx
episode_id
scene_id
segment_id
sequence_id
scheduler_phase
stride
rollout_phase
rollout_idx_in_episode
sequence_positions
block_ids
source_frame_indices
visit_kinds
repeat_budgets
inner_K
history_positions
history_num_frames
history_num_refs
temporal_read_count
temporal_commit_count
physical_time_advance_count
repair_enabled
repair_permutation_hash
reset_before
carry_after
detach_after
episode_end
```

## 15.2 每 train step

```text
scheduler/phase_id
scheduler/sequence_length
scheduler/stride
scheduler/rollout_phase_id
scheduler/sequence_pos_min/max
scheduler/unique_blocks_seen
scheduler/temporal_commit_count
scheduler/history_num_frames
scheduler/history_num_refs
scheduler/repair_enabled
```

Loss：

```text
loss/current
loss/history_all_seen
loss/repair_all10
loss/damage
loss/delta_reg
```

Temporal：

```text
parent_temporal/seen_ratio_bg
parent_temporal/seen_ratio_distant
parent_temporal/seen_ratio_rigid
parent_temporal/commit_count
parent_temporal/commit_skipped_repair
parent_temporal/adapter_to_spatial_ratio
```

## 15.3 Episode end summary

新增 split：

```text
iforward_sequence10_episode
```

字段：

```text
10 unique blocks verified
causal_rollouts_completed = 5
repair_applied
causal_final_all10_psnr
repair_final_all10_psnr
repair_gain
first_frame_final_psnr
forget_p90
episode_total_inner_K
episode_history_render_count
episode_elapsed_ms
```

---

# 16. Validation 系统

Validation 不使用训练 scheduler sampling。

## 16.1 Fixed manifest

新增 JSON manifest：

```json
{
  "version": "sequence10_manifest_v1",
  "items": [
    {
      "scene_id": 0,
      "segment_id": 0,
      "start_block_pos": 3,
      "stride": 1,
      "source_frame_offsets": [0,0,0,0,0,0,0,0,0,0]
    }
  ]
}
```

Manifest 初始化后固定，所有 checkpoint 使用同一序列。

## 16.2 协议

### SingleFrame-K8

```text
独立 reset
一个 block
R8
```

用于监控 current 质量是否退化。

### S10-D1-Causal

```text
10帧 stride1
5×B2R4
无 repair
```

### S10-D2-Causal

同上，stride2。

### S10-D1-Repair

```text
先 causal
clone/detach causal final state
执行固定 permutation B10R1
```

### S10-D2-Repair

同上，stride2。

### Repeat Stability（低频）

```text
从固定 state 选择1帧
R4/R8/R16/R32
Temporal commit=false
```

## 16.3 Validation state isolation

每个协议必须：

```text
reset model state cache
reset bridge runtime state
reset ParentTemporal state
使用独立 LocalGSState
```

不同 repair permutation 不允许串行共享 state。

## 16.4 指标

Current：

```text
current_psnr_by_position[0..9]
current_ssim_by_position
current_l1_by_position
```

Final all10：

```text
mean_psnr
p10_psnr
min_psnr
first_frame_psnr
last_frame_psnr
```

Retention：

```text
best_to_final_drop_mean
best_to_final_drop_p90
last_seen_to_final_drop_p90
retention_auc
quality_vs_age_curve
```

Repair：

```text
repair_gain_mean
repair_gain_p10
repair_harm_p90
permutation_std
```

Repeat stability：

```text
R4_to_R8_delta
R4_to_R16_delta
R4_to_R32_delta
best_to_R32_drop
monotonic_violation_count
```

高频：

```text
masked_edge_l1
masked_laplacian_l1
high_frequency_energy_ratio
```

## 16.5 Validation 频率

```yaml
validation:
  single_frame:
    interval_steps: 1000
  sequence10_quick:
    interval_steps: 2500
    num_sequences: 2
  sequence10_full:
    interval_steps: 10000
    num_sequences: 8
  repeat_stability:
    interval_steps: 10000
```

---

# 17. 测试方案

## 17.1 配置测试

```text
test_sequence10_rejects_legacy_shape_fields
test_sequence10_requires_length_10
test_sequence10_requires_causal_b2r4
test_sequence10_requires_repair_b10r1
test_sequence10_max_inner_k_10
```

## 17.2 Sequence selection

```text
test_stride1_selects_10_unique_blocks
test_stride2_selects_positions_with_gap_2
test_no_circular_fill
test_short_segment_is_skipped
test_stride_fallback
test_source_frame_fixed_across_repair
```

## 17.3 Causal rollout

```text
test_causal_episode_has_five_rollouts
test_each_causal_rollout_has_two_blocks_four_repeats
test_causal_positions_cover_0_to_9_once
test_causal_temporal_commit_only_at_block_exit
test_causal_physical_time_advance_true
```

## 17.4 Repair rollout

```text
test_repair_permutation_unique_and_non_identity
test_repair_has_10_blocks_one_repeat
test_repair_temporal_commit_false
test_repair_observation_commit_false
test_repair_physical_time_advance_false
test_repair_supervises_all10
```

## 17.5 State lifecycle

```text
test_state_reset_only_episode_begin
test_state_carried_across_causal_rollouts
test_graph_detached_but_values_preserved
test_temporal_state_unchanged_by_repair
test_episode_end_clears_state
```

## 17.6 History

```text
test_bootstrap_fetches_no_history
test_causal_history_is_all_seen_excluding_current
test_history_has_no_duplicate_refs
test_repair_uses_all10
test_best_loss_bank_is_detached
```

## 17.7 Resume

```text
test_scheduler_resume_reproduces_next_rollout
test_resume_preserves_repair_permutation
test_resume_preserves_source_frame_choices
```

## 17.8 Validation

```text
test_validation_manifest_is_deterministic
test_protocols_use_independent_state
test_causal_and_repair_do_not_share_mutated_state
test_repeat_stability_never_commits_temporal_memory
test_metrics_include_all10_frames
```

## 17.9 Model integration

```text
test_parent_temporal_commit_respects_step_flag
test_repair_block_exit_does_not_commit
test_frame_gap_embedding_changes_stride1_vs_stride2
test_visit_kind_embedding_changes_causal_vs_repair
test_invalid_child_update_hard_zero_remains_enabled
```

---

# 18. Fail-fast 条件

Scheduler 启动时：

```text
- legacy shape fields 出现：error
- sequence length != 10：error
- causal inner_K > 12：error
- repair inner_K > 12：error
- circular fill requested：error
- duplicate block in sequence：error
- repair temporal_commit=true：error
- causal temporal_commit=false：error
- source frame在repair中改变：error
```

Runtime：

```text
- episode block未达到10：skip并记录，不补重复
- history ref跨scene/segment：error
- repair permutation有重复：error
- Temporal state在repair后发生version变化：error
```

---

# 19. Migration

## 19.1 保留旧 scheduler

旧：

```text
iforward_v1
iforward_v3_random_window
iforward_v4_coverage_ordered
iforward_stage2_1_parent_temporal
```

继续保留用于复现实验，但不再扩展。

## 19.2 新版本独立入口

```text
iforward_sequence10_v1
```

不继承旧 scheduler 类。

## 19.3 Batch 兼容

第一版通过 adapter 输出现有 batch meta；模型完成 `temporal_commit` 等显式字段支持后，逐步移除 legacy 字段。

---

# 20. 实施阶段

## P0：Scheduler 核心

```text
SequenceSpec / VisitSpec / RolloutPlan / EpisodePlan
eligibility index
bootstrap
causal B2R4
state_dict/resume
```

## P1：Repair 与显式 memory flags

```text
B10R1 random repair
model temporal_commit flag
visit_kind/frame_gap embedding
```

## P2：History supervision

```text
all-seen causal history
all10 repair loss
best-loss damage bank
```

## P3：Validation

```text
fixed manifest
S10-D1/D2 causal + repair
repeat stability
retention curves
```

## P4：Preload 与性能

```text
full sequence async hint
history grouped render
scheduler timing/log cleanup
```

---

# 21. 验收标准

Scheduler correctness：

```text
每个 Sequence10 episode：
    恰好10个unique blocks
    恰好5个causal rollouts
    每causal block恰好R4
    repair最多1次
    repair不修改Temporal state
```

训练：

```text
bootstrap current K8 不低于当前Stage2.1 baseline
S10-D1 causal final all10质量稳定上升
history loss实际覆盖远端frame0/1
```

短序列：

```text
S10-D1 causal forget p90 显著低于旧4-block r8b1
S10-D2不出现明显崩溃
repair gain为正
repair permutation std逐步下降
```

稳定性：

```text
R4 -> R16无明显质量崩溃
R4 -> R32 best-to-final drop受控
```

工程：

```text
scheduler代码不包含旧version分支
配置不再包含shape policy矩阵
resume确定性通过
validation协议全部独立执行
```

---

# 22. 文献依据

1. RAFT：共享权重 recurrent update operator，支持迭代 refinement。  
   https://arxiv.org/abs/2003.12039
2. DROID-SLAM：通过 recurrent update 与 differentiable optimization 处理视频序列状态。  
   https://arxiv.org/abs/2108.10869
3. CUT3R：persistent state 随新观测持续更新，并采用从短序列到更长序列的训练方式。  
   https://arxiv.org/abs/2501.12387
4. Unbiasing Truncated BPTT：固定截断长度对长期依赖存在偏差，支持将 state 生命周期与 graph 长度分离。  
   https://arxiv.org/abs/1705.08209
5. Adaptive TBPTT：截断长度过短或过长都可能影响优化效率，支持明确控制 rollout graph 长度。  
   https://arxiv.org/abs/1905.07473

---

# 最终建议

新 scheduler 应当非常明确：

```text
训练前期：一个 block，学习如何迭代更新。
训练后期：一个 episode 就是一条固定10帧序列。

先用 chronological B2R4 吸收10帧并写 Temporal state；
再以一定概率用 random B10R1 修复 GS state，但不写 Temporal state；
所有已见历史都参与监督；
每个 rollout 后 detach graph，但整个 episode 不 reset state。
```

它不再尝试用一个通用 shape 系统表达所有任务，而是直接表达 IForward 当前唯一需要解决的训练问题：

```text
在10帧真实时序中建立、保持并修复同一个可持续更新的GS scene state。
```
