# IForward 2_3：Parent Optimizer Mamba 与 Scheduler v3 详细实现方案

基线代码：`drivestudio_stage6_refactor_context_20260625_v35`  
目标版本名：`iforward_2_3_scheduler_v3_optimizer_mamba`  
兼容策略：**不保留 Stage 2_2 Temporal Mamba 的状态语义兼容；不兼容旧 checkpoint。**  
核心目标：把 IForward 明确定义为 **learned iterative optimizer**，将 Mamba 从“物理时间记忆”改为“parent-level 优化器记忆”，并重构 scheduler，使其训练“任意访问顺序下持续优化同一 GS state 且保持历史渲染”的能力。

---

# 0. 核心结论

Stage 2_2 当前把 Mamba 当成：

```text
Parent Temporal Mamba
= 真实时间帧序列记忆
```

但 IForward 的实际目标更像：

```text
Parent Optimizer Mamba
= episode 内的 parent-level learned optimizer hidden state
```

因此 Stage 2_3 改为：

```text
每一个 optimizer update repeat：
    read Mamba
    compute update
    apply delta
    write Mamba
```

而不是：

```text
每 repeat read
每 frame exit commit 一次
```

新的 Mamba state 表示：

```text
这个 episode 内优化器已经做过什么、哪些 parent 刚被修、哪些区域容易破坏、历史渲染应如何保持。
```

不是：

```text
真实物理时间中的 scene dynamics。
```

因此 frame 输入顺序可以是：

```text
chronological causal order
random repair order
repeat stress order
```

只要 visit metadata 告诉模型当前 update 属于哪一类。

---

# 1. 新语义：Parent Optimizer Mamba

## 1.1 旧语义的问题

Stage 2_2 使用：

```text
Causal：按时间顺序写 Mamba
Repair：只读 Mamba，不写 Mamba
```

这等价于：

```text
Mamba = temporal context
```

它能帮助当前 frame 利用过去 frame 的 parent-level context，但它没有直接学习：

```text
我刚刚优化过 frame7；
我上一步破坏了 frame1；
某个 parent 多次迭代后已经接近稳定；
repair 的随机访问顺序下如何维持全局状态。
```

这与 IForward 的 learned optimizer 本质不一致。

## 1.2 新语义

Stage 2_3 中：

```text
Mamba = Parent Optimizer Memory
```

其输入序列是：

```text
optimizer update sequence
```

而不是：

```text
physical frame sequence
```

因此以下顺序都合法：

```text
f0 repeat0 -> f0 repeat1 -> f0 repeat2 -> f0 repeat3
f1 repeat0 -> f1 repeat1 -> ...
repair f7 -> repair f2 -> repair f9
repeat stress f3 R8
```

Mamba 本身是选择性状态空间序列模型，状态会随输入选择性保留/遗忘信息。只要序列语义一致，它不要求该序列必须是物理时间序列。

## 1.3 State 生命周期

```text
episode begin:
    reset OptimizerMambaState

每个 update repeat:
    preview/read state
    run parent event / GRLD / posterior
    apply GS delta
    write/update state

rollout boundary:
    detach graph
    keep state value

episode end:
    discard state
```

即：

```text
state persistence scope = episode
state update granularity = repeat
BPTT scope = rollout
```

---

# 2. 总体模型架构

```text
raw frame observation
    ↓
FWHR fine-GS lifting
    ├─ parent_context [M,48]
    └─ child_detail [N,8]

Parent Token Builder
    ↓
Parent PTv3 spatial encoder
    ↓
parent_spatial_event [M,64]

Parent Optimizer Mamba
    ├─ read optimizer state
    ├─ fuse optimizer context with spatial event
    └─ parent_event [M,64]

GRLD
    ↓
child_event [N,16]

Posterior Updater
    ↓
GS delta

Apply Delta
    ↓
updated LocalGSState

Parent Optimizer Mamba write
    input = parent_spatial_event + update_summary + visit metadata
```

Important split:

```text
FWHR child_detail:
    current image evidence
    transient
    not stored directly in Mamba

Parent Optimizer Mamba:
    optimizer hidden state
    parent-level
    episode-local
```

---

# 3. Parent Optimizer Mamba V3

## 3.1 文件重构

新增：

```text
models/iforward/stage2_3/optimizer_memory_schema.py
models/iforward/stage2_3/parent_optimizer_mamba.py
models/iforward/stage2_3/optimizer_visit_embedding.py
models/iforward/stage2_3/optimizer_write_token.py
models/iforward/stage2_3/episode_history_bank_v3.py
models/iforward/stage2_3/sequence_loss_v3.py
```

废弃或停止用于 Stage 2_3：

```text
parent_temporal_mamba_v2.py
parent_temporal_state_v2.py
parent_temporal_keys_v2.py
TemporalMotionEmbedding 的 physical-time-only 语义
```

可以复用底层 `StreamingMambaCell`，但 state schema 和读写语义必须重写。

---

## 3.2 State schema

```python
@dataclass
class DenseOptimizerState:
    conv_state: Tensor
    ssm_state: Tensor
    seen: BoolTensor
    update_count: IntTensor
    last_visit_step: IntTensor
    last_frame_id: IntTensor
    last_visit_kind: IntTensor

@dataclass
class KeyedOptimizerState:
    keys: LongTensor
    conv_state: Tensor
    ssm_state: Tensor
    seen: BoolTensor
    update_count: IntTensor
    last_visit_step: IntTensor
    last_frame_id: IntTensor
    last_visit_kind: IntTensor

@dataclass
class ParentOptimizerMambaState:
    bg: DenseOptimizerState
    distant: DenseOptimizerState
    rigid: KeyedOptimizerState
    global_update_step: int
```

Keying:

```text
BG：dense parent id
Distant：dense parent id
Rigid：hash(instance_id, global_parent_id)
```

Rigid duplicate active rows:

```text
preview：同一 key 共享 state；输出复制回 active rows
write：按 support 加权聚合 token 后写一次
```

---

## 3.3 每 repeat 的读写 API

```python
ctx, seen = optimizer_mamba.preview(
    parent_spatial_event,
    state,
    parent_keys,
    visit_meta,
)

parent_event = optimizer_mamba.fuse(
    parent_spatial_event,
    ctx,
    seen,
    visit_meta,
)

# after delta is computed and applied
new_state = optimizer_mamba.write(
    write_token,
    state,
    parent_keys,
    support,
    valid,
    visit_meta,
)
```

必须保证：

```text
preview 不修改 state
write 修改 state
每 repeat 最多 write 一次
episode end 丢弃 state
```

---

## 3.4 Visit metadata

每个 repeat 必须传入：

```python
@dataclass(frozen=True)
class VisitMeta:
    visit_kind: Literal[
        "bootstrap",
        "assimilate",
        "repair",
        "repeat_stability",
    ]
    frame_id: int
    keyframe_id: int
    sequence_pos: int
    timestamp_us: int
    frame_gap_from_previous_visit: int
    time_since_same_frame_visit: float
    visit_count_for_frame: int
    repeat_idx: int
    repeat_budget: int
    global_update_idx_in_episode: int
    is_first_visit_of_frame: bool
    is_last_update_of_episode: bool
```

Stage 2_3 不再把 `frame_gap` 当成物理 temporal transition 的唯一依据，而是作为 optimizer condition。

---

## 3.5 Visit embedding

```python
class OptimizerVisitEmbedding(nn.Module):
    visit_kind_embed: Embedding
    repeat_idx_embed: Embedding
    repeat_budget_embed: Embedding
    frame_visit_count_embed: Embedding
    sequence_pos_embed: Embedding
    frame_gap_embed: Embedding or Fourier
    global_update_fourier: Fourier + MLP
```

推荐 first version:

```text
visit_kind_dim         8
repeat_idx_dim         8
repeat_budget_dim      8
visit_count_dim        8
sequence_pos_dim       8
frame_gap_dim          8
output_dim            32
```

---

## 3.6 Write token

不要只写 parent event。应写入：

```text
parent_spatial_event
support / valid
branch embedding
visit embedding
update summary
```

Update summary 第一版：

```text
parent_mean_delta_norm
parent_opacity_delta_norm
parent_sh_delta_norm
parent_scale_delta_norm
parent_noop_mean
parent_confidence_mean
```

这些从 child delta scatter 到 parent：

```python
summary_parent = scatter_weighted_mean(child_summary, child_to_parent, child_mass)
```

不写入：

```text
child_detail [N,8]
raw image feature
full child delta tensor
```

避免 Mamba 变成第二套场景缓存。

---

## 3.7 Fusion

```python
optimizer_delta = Adapter(ctx)  # 32 -> 64

fusion_gate = sigmoid(
    branch_gate
    + visit_gate(visit_kind)
    + support_gate(log1p(support))
)

parent_event = LayerNorm(
    parent_spatial_event
    + seen * fusion_gate * optimizer_delta
)
```

Bootstrap:

```text
Mamba disabled
seen false
```

Assimilation/Repair:

```text
Mamba enabled
read/write every repeat
```

---

# 4. Scheduler v3 总体设计

Stage 2_3 scheduler 不再叫 temporal scheduler。它应叫：

```text
Optimizer Episode Scheduler
```

Episode:

```text
one segment
one LocalGSState
one ParentOptimizerMambaState
one sampled raw-frame set
```

Episode 内部分两大部分：

```text
Part 1：Assimilation
    B 少，R 多
    建立 scene state
    训练 repeat-level optimizer memory

Part 2：Repair
    B 大，R 小
    随机重访 frame
    训练 order-robust repair 和 history keeping
```

---

# 5. Bootstrap 阶段

## 5.1 目标

```text
训练基础 updater
训练 FWHR / GRLD / Posterior
训练 B1 repeat 稳定性
不训练 Mamba
减少冷启动 fetch
```

## 5.2 结构

不是每 step 换一个 scene，而是：

```text
Bootstrap asset pack:
    same segment
    4 independent frames
    each fresh GS
    shared loaded asset/cache
```

Example:

```text
bootstrap episode:
    rollout 0: frame a, fresh GS, R4
    rollout 1: frame b, fresh GS, R6
    rollout 2: frame c, fresh GS, R8
    rollout 3: frame d, fresh GS, R4
```

每个 rollout:

```text
temporal_read=false
temporal_write=false
reset LocalGSState from segment initial GS
```

但 dataset/cache:

```text
不重新加载 segment static asset
```

## 5.3 Repeat distribution

```yaml
bootstrap:
  end_step: 5000
  frames_per_asset_pack: 4
  repeat_distribution:
    4: 0.50
    6: 0.30
    8: 0.20
```

---

# 6. Sequence sampling

Stage 2_3 不必固定 D1/D2/I123。新的采样是：

```text
同一 segment 内随机取 n 个 raw frames
满足空间/时间约束
```

## 6.1 n distribution

```yaml
sequence:
  min_frames: 6
  max_frames: 10
  frame_count_distribution:
    8: 0.30
    9: 0.30
    10: 0.40
```

如果 segment frame 不足 8，但 >=6，可作为 short-sequence sample。

不足 6：

```text
只用于 bootstrap，不用于 sequence training
```

## 6.2 Sampling constraints

```text
unique_keyframe_count >= 3
frame_span >= 8
frame_span <= 30
time_span <= configured max
all required cameras available
same segment
```

## 6.3 Ordering

For Assimilation:

```text
80% chronological order
20% local-perturbed chronological order
```

Local perturbation:

```text
sort by time
then swap adjacent pairs with small probability
```

Do not use full random order in assimilation.

For Repair:

```text
full random permutation of selected frames
```

---

# 7. Assimilation stage

## 7.1 Goal

```text
吸收多个观测
训练每 repeat read/write Mamba
每帧有足够 repeat 达到较好 current quality
保持历史状态
```

## 7.2 Rollout structure

```text
2 frames per rollout
R sampled per frame
inner_K <= 12
```

Repeat pair table:

```text
[R4, R4] prob 0.30  inner_K 8
[R4, R6] prob 0.15  inner_K 10
[R6, R4] prob 0.15  inner_K 10
[R6, R6] prob 0.15  inner_K 12
[R4, R8] prob 0.10  inner_K 12
[R8, R4] prob 0.10  inner_K 12
[R5, R5] prob 0.05  inner_K 10
```

If selected frame count is odd, final rollout is:

```text
B1 × R sampled from {4,6,8}
```

## 7.3 Per repeat behavior

For every repeat:

```text
read Optimizer Mamba
run model update
apply delta
write Optimizer Mamba
```

No special frame-exit commit exists anymore.

---

# 8. Repair stage

## 8.1 Goal

```text
随机重访历史帧
修复被破坏的渲染
训练 optimizer memory 不依赖固定顺序
```

## 8.2 Repair schedule

```yaml
repair:
  start_step: 15000
  probability_schedule:
    - [15000, 0.10]
    - [18000, 0.25]
    - [22000, 0.50]
  rounds_distribution:
    1: 0.70
    2: 0.30
```

## 8.3 Per round sampling

```yaml
repair:
  round_patterns:
    - name: B6R1
      frames: 6
      repeats: 1
      prob: 0.35
    - name: B8R1
      frames: 8
      repeats: 1
      prob: 0.45
    - name: B6R2
      frames: 6
      repeats: 2
      prob: 0.20
```

Hard cap:

```text
inner_K <= 12
```

## 8.4 Multi-round repair

If 2 rounds:

```text
round 1: random subset
round 2: biased to cover frames not visited in round 1
```

Example for 10 frames:

```text
round1: [7,1,9,3,0,6]
round2: [2,4,5,8,1,7]
```

## 8.5 Mamba behavior

Repair is no longer special from Mamba view:

```text
every repeat read/write same Optimizer Mamba
```

But visit metadata says:

```text
visit_kind=repair
physical_time_advance=false
```

Last update in episode:

```text
write optional false
```

This is a small optimization only.

---

# 9. Loss design

## 9.1 Assimilation rollout loss

```text
current frames:
    L1 + SSIM

history frames already seen but not current:
    L1-only or L1 + low-cost SSIM subset

best damage:
    L1-only
```

## 9.2 Repair rollout loss

```text
visited frames:
    L1 + SSIM

unvisited frames:
    L1-only retention

best damage:
    L1-only against episode best bank
```

## 9.3 Episode bank

For each sampled frame:

```text
best_loss
best_psnr
last_loss
visit_count
last_update_step
```

Bank update after every rollout final.

Damage:

```python
damage = relu(current_loss - best_loss - margin)
```

Do not render twice. Use final render per-ref loss.

---

# 10. Scheduler data schema

```python
@dataclass(frozen=True)
class FrameVisit:
    frame_idx: int
    keyframe_idx: int
    timestamp_us: int
    sequence_pos: int
    visit_kind: str  # bootstrap / assimilate / repair
    repeat_budget: int
    repeat_idx_start: int
    is_first_visit_of_frame: bool
    visit_count_for_frame: int
    order_index: int
```

```python
@dataclass(frozen=True)
class RolloutPlanV3:
    phase: str  # bootstrap / assimilation / repair
    visits: tuple[FrameVisit, ...]
    current_positions: tuple[int, ...]
    history_positions: tuple[int, ...]
    inner_K: int
    detach_after_rollout: bool = True
```

```python
@dataclass(frozen=True)
class EpisodePlanV3:
    scene_id: int
    segment_id: int
    frame_set: tuple[int, ...]
    keyframe_set: tuple[int, ...]
    sampled_order: tuple[int, ...]
    rollouts: tuple[RolloutPlanV3, ...]
    episode_hash: int
```

---

# 11. Scheduler implementation

New package:

```text
datasets/iforward_stage2_3/
    index_builder.py
    index_loader.py
    bootstrap_sampler.py
    sequence_sampler.py
    repair_sampler.py
    episode_producer.py
    scheduler.py
    resolver.py
    validation_runner.py
```

No backward compatibility with Stage2_2 scheduler.

## 11.1 Index

Keep Stage2_2 raw-frame index idea, but add:

```text
frame candidates per segment
unique keyframe count lookup
frame span lookup
random sequence pool per segment
```

Precompute candidate frame sets if needed:

```text
segment -> list of valid frame_set seeds
```

But avoid enumerating all combinations for long segments.

Use bounded rejection sampling:

```text
try 64 random samples per segment
fallback to precomputed small pool
```

## 11.2 Traversal

```text
scene round-robin
segment round-robin within scene
```

Sequence sampling inside segment:

```text
random n frames satisfying constraints
```

Bootstrap uses independent sampler:

```text
scene balanced -> segment balanced -> frame random
```

---

# 12. Thread and preload design

Use same idea from Stage2_2 but adjust for v3:

```text
EpisodeProducer thread builds EpisodePlanV3
preload current rollout
preload next rollout
preload full frame set metadata
preload next episode segment static
```

Bootstrap asset pack:

```text
load one segment once
sample 4 independent frames
reuse segment static / DINO cache / parent assignment cache
```

Expected effect:

```text
first rollout fetch lower
bootstrap no longer cold-switch every step
repair frames already hot from assimilation
```

---

# 13. Validation v3

Validation must reflect optimizer-memory semantics.

## 13.1 Assimilation validation

```text
sample n=10 frames
chronological
R table same as training or fixed R4
no repair
render all frames
```

## 13.2 Repair validation

From same assimilation final state:

```text
clone state
run permutation A
clone state
run permutation B
clone state
run permutation C
```

Metrics:

```text
mean PSNR
worst permutation PSNR
std across permutations
unvisited-frame regression
```

## 13.3 Repeat validation

From same state clone:

```text
R4 / R8 / R16 / R32
```

Each R starts from identical state.

## 13.4 Mamba ablation validation

```text
Mamba off
Mamba read-only
Mamba read/write every repeat
Mamba shuffled state
```

This directly tests whether optimizer memory is useful.

---

# 14. Logging

Must log:

```text
optimizer_mamba/read_count
optimizer_mamba/write_count
optimizer_mamba/state_norm
optimizer_mamba/context_to_spatial_ratio
optimizer_mamba/write_token_delta_norm
optimizer_mamba/repeat_idx_distribution
optimizer_mamba/visit_kind_distribution
```

Scheduler:

```text
episode/frame_count
episode/unique_keyframe_count
episode/frame_span
episode/assimilation_rollouts
episode/repair_rounds
rollout/phase
rollout/inner_K
rollout/repeat_budgets
rollout/current_count
rollout/history_count
```

Loss:

```text
loss/current_raw
loss/history_raw
loss/history_weighted
loss/best_damage_raw
loss/fixed_objective
```

Memory:

```text
mem/after_forward
mem/after_final_render
mem/after_loss
mem/before_backward
mem/peak
```

---

# 15. Tests

## Mamba semantics

```text
test_every_repeat_writes_state
test_rollout_boundary_detach_keeps_state
test_episode_reset_clears_state
test_repair_writes_optimizer_state
test_visit_kind_changes_context
test_repeat_idx_changes_context
test_write_token_uses_delta_summary
```

## Scheduler

```text
test_bootstrap_fresh_gs_but_reused_asset_pack
test_sequence_sampling_constraints
test_assimilation_inner_k_le_12
test_repair_inner_k_le_12
test_repair_random_permutation
test_multi_round_repair_covers_unvisited_frames
test_scheduler_scene_segment_balancing
```

## Loss

```text
test_unvisited_repair_frames_get_l1_retention
test_best_damage_uses_l1_only
test_no_second_render_for_per_pos
test_bank_updates_after_every_rollout
```

## Validation

```text
test_repair_permutations_use_cloned_state
test_repeat_validation_clones_initial_state
test_mamba_ablation_modes
test_order_robustness_metrics
```

---

# 16. Main risks

## 16.1 Mamba becomes scene cache

Mitigation:

```text
state small, e.g. 32
parent-level only
no child detail write
episode reset
permutation validation
state shuffle ablation
```

## 16.2 Repeat writing amplifies early mistakes

Mitigation:

```text
write token includes noop/confidence/delta norm
low support parent write masked
grad clipping
small Mamba fusion gate
```

## 16.3 Repair overfits order

Mitigation:

```text
many permutations
permutation std metric
worst permutation objective
```

## 16.4 Memory pressure

Mitigation:

```text
inner_K <= 12
repair L1-only for unvisited/damage
chunk final render if needed
FWHR fused CUDA later
```

---

# 17. Config draft

```yaml
iforward_2_3:
  optimizer_mamba:
    enable: true
    name: parent_optimizer_mamba
    read_policy: every_repeat
    write_policy: every_repeat
    state_dim: 8
    context_dim: 32
    fusion_gate_init: 0.05
    write_token:
      include_parent_event: true
      include_support: true
      include_visit_kind: true
      include_repeat_idx: true
      include_delta_summary: true
      include_noop_confidence: true

scheduler_v3:
  bootstrap:
    end_step: 5000
    frames_per_asset_pack: 4
    fresh_gs_each_frame: true
    mamba: off
    repeats:
      4: 0.50
      6: 0.30
      8: 0.20

  sequence:
    min_frames: 6
    max_frames: 10
    frame_count_distribution:
      8: 0.30
      9: 0.30
      10: 0.40
    min_unique_keyframes: 3
    min_frame_span: 8
    max_frame_span: 30

  assimilation:
    start_step: 5000
    frames_per_rollout: 2
    repeat_pair_table:
      - repeats: [4,4]
        prob: 0.30
      - repeats: [4,6]
        prob: 0.15
      - repeats: [6,4]
        prob: 0.15
      - repeats: [6,6]
        prob: 0.15
      - repeats: [4,8]
        prob: 0.10
      - repeats: [8,4]
        prob: 0.10
      - repeats: [5,5]
        prob: 0.05
    max_inner_k: 12

  repair:
    start_step: 15000
    probability_schedule:
      - [15000, 0.10]
      - [18000, 0.25]
      - [22000, 0.50]
    rounds_distribution:
      1: 0.70
      2: 0.30
    patterns:
      - name: B6R1
        frames: 6
        repeats: 1
        prob: 0.35
      - name: B8R1
        frames: 8
        repeats: 1
        prob: 0.45
      - name: B6R2
        frames: 6
        repeats: 2
        prob: 0.20
    max_inner_k: 12
```

---

# 18. Implementation stages

## Stage A

```text
Rename Temporal Mamba -> Optimizer Mamba
Change write policy to every repeat
Add visit/repeat embeddings
Keep existing scheduler
Smoke test
```

## Stage B

```text
Implement scheduler v3 assimilation
Disable repair
Train 5k-10k
```

## Stage C

```text
Enable one-round repair
Add unvisited L1 retention
Train 10k-20k
```

## Stage D

```text
Enable multi-round repair
Add validation ablations
```

---

# Final statement

IForward 2.3 的关键修正是：

```text
Mamba 不再表示物理时间，而表示优化过程。
```

Scheduler v3 的关键修正是：

```text
训练不再强调“真实时间顺序是否正确”，而强调“任意优化访问顺序下，GS state是否持续改善且历史不被破坏”。
```

因此新的基本循环是：

```text
read optimizer memory
read current frame evidence
update GS
write optimizer memory
```

每一个 repeat 都执行这一循环。这个定义比当前 Stage 2.2 更符合 IForward 作为 learned iterative optimizer 的本质。
