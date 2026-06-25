# IForward Stage 2_2：Raw-Frame Stream10 Scheduler 与 Temporal Mamba V2 全量重构方案

基线：`drivestudio_stage6_refactor_context_20260623_v34`  
目标版本：`iforward_stage2_2_stream10_rawframe`  
兼容策略：**不兼容 Stage 2_1 scheduler、resolver、Temporal Mamba state、配置或 checkpoint**  
核心目标：在不改变 segment/GS asset 范围的前提下，以 raw-frame observation 重建真正可覆盖多场景的 10 帧时序训练与验证系统，并将 Temporal Mamba 改造成支持非均匀时间间隔、稳定 parent identity 和严格时钟语义的状态模块。

---

# 0. 最终架构决策

Stage 2_2 固定使用四层语义：

```text
Scene
  └─ Segment
       └─ Episode：同一 segment 内的 10 个 raw-frame observations
            └─ Observation block：一个 raw frame + 所有输入 camera + R 次 iterative update
```

其中：

```text
Segment：
    保持不变；继续决定初始 GS、AABB、点密度、parent assignment 和 episode reset 范围。

Keyframe：
    不再是 Sequence10 的时间步；仅作为空间覆盖标签、采样约束和日志元数据。

Observation block：
    改为实际 raw frame；是 Temporal Mamba 的一个真实时间步。

Episode：
    同一 segment 内 10 个按时间排序的 raw-frame observations；
    LocalGSState 与 ParentTemporalState 在整个 episode 内持续保留。
```

主训练协议：

```text
Bootstrap：
    全训练集单帧，R∈{4,6,8}。

Causal Stream10：
    10 个 raw frames；5 个 B2×R4 rollout；chronological；每 frame commit 一次 Mamba。

Optional Repair：
    causal 完成后，B10×R1 随机重访；更新 GS；读取但不写 Temporal Mamba。
```

主 sequence protocol：

```text
D1：固定 raw-frame gap 1，概率 0.30
D2：固定 raw-frame gap 2，概率 0.40
I123：相邻 gap 从 {1,2,3} 有序采样，概率 0.30
```

不在 Stage 2_2 主训练中使用 uniform gap 1–5。Gap 4/5 只作为后期 stress validation。

---

# 1. 为什么必须重构，而不是继续修补 Sequence10 V1

Sequence10 V1 把 keyframe 当作 block，要求同一 segment 至少存在 10/19 个 keyframe，导致当前 asset 的 Sequence10 覆盖坍缩：

```text
train stride1：2 scenes / 2 segments
train stride2：0 scenes
validation：0 valid sequences
```

根因不是 scene 总帧数不足，而是 segment 的空间划分与 keyframe 数量不匹配。Segment 又决定 GS 初始资产和密度，因此不能为了 scheduler 任意扩大。

当前 asset 在 raw frame 下具有足够覆盖：

```text
D1：132 train scenes / 959 segments
D2：129 train scenes / 712 segments
D3：106 train scenes / 351 segments
```

因此 Stage 2_2 的核心不是修改 segment，而是把时序 observation 从 keyframe block 解耦。

本次不保留旧实现的原因：

```text
- 旧 scheduler 同时支持多个互相冲突的 block/shape/stride 语义；
- 大量配置字段存在但未真正控制行为；
- eligibility、bootstrap、validation 和 preload 相互耦合；
- frame_gap 仅支持 0/1/2；
- Temporal Mamba state 缺乏每-key时间戳；
- 旧 resolver 依赖 keyframe block 的角色构造；
- 兼容 shim 会继续隐藏错误协议。
```

Stage 2_2 建议建立独立 package，并从训练入口移除旧 scheduler 注册。

---

# 2. 目录与模块重构

新增 package：

```text
datasets/iforward_stage2_2/
    schema.py
    index_format.py
    index_builder.py
    index_loader.py
    protocol_sampler.py
    traversal.py
    episode_producer.py
    scheduler.py
    resolver.py
    preload.py
    validation_manifest.py
    validation_runner.py

models/iforward/stage2_2/
    temporal_schema.py
    temporal_motion_embedding.py
    parent_temporal_state_v2.py
    parent_temporal_mamba_v2.py
    parent_temporal_keys_v2.py
    episode_history_bank.py
    sequence_loss.py

configs/iforward/
    iforward_stage2_2_stream10_rawframe.yaml
```

从 Stage 2_2 训练入口彻底删除：

```text
TrainSchedulerIForward
TrainSchedulerIForwardSequence10
sequence10_resolver.py
sequence10_batch.py
sequence10_history_bank.py
旧 sequence10 validation manifest-only 路径
旧离散 frame_gap Embedding(3)
旧 scheduler_iforward version 分支
旧 shape matrix / block_source / strides 配置
```

旧 Stage 2.0/2.1 checkpoint 不加载；Stage 2.2 从 0 开始训练。

---

# 3. 数据预处理索引

Scheduler 运行时禁止：

```text
- 扫描所有 scene/segment；
- Python 循环构造所有窗口；
- 同步 resolve segment bundle；
- 动态检查 keyframe 覆盖；
- 为每个 episode 重建 frame→keyframe 映射。
```

这些工作必须在预处理阶段完成。

## 3.1 Index Builder

新增命令：

```bash
python tools/build_iforward_stage2_2_index.py \
  --config configs/iforward/iforward_stage2_2_stream10_rawframe.yaml
```

输出目录：

```text
<asset_root>/iforward_stage2_2_index/<fingerprint>/
    metadata.json
    segments.npy
    frames.npy
    d1_windows.npy
    d2_windows.npy
    irregular_patterns.npy
    irregular_windows.npy
    bootstrap_frames.npy
    train_scene_table.npy
    eval_scene_table.npy
```

使用 NumPy contiguous arrays / mmap，不保存大量 Python object。

## 3.2 Fingerprint

Index fingerprint 必须覆盖：

```text
asset registry hash
segment manifest version
scene split
camera ids
raw frame availability
frame→keyframe mapping
timestamp/ego-pose hash
sequence length
protocol definitions
unique-keyframe thresholds
```

运行时 fingerprint 不匹配必须 fail-fast，禁止静默重建或回退。

## 3.3 Frame Table

每个 raw frame 行保存：

```text
scene_id             int32
segment_id           int32
frame_idx             int32
keyframe_idx          int32
timestamp_us          int64
ego_translation       float32[3]
ego_yaw               float32
is_train              uint8
available_camera_mask uint8
```

Segment table 保存 frame CSR offset：

```text
scene_id
segment_id
frame_start
frame_count
keyframe_count
asset_id_hash
```

## 3.4 Protocol Window Index

### D1/D2

直接预计算：

```text
(segment_row, start_local_frame)
```

且提前过滤：

```text
D1 unique_keyframes >= 2
D2 unique_keyframes >= 3
所有需要 camera 可用
严格位于同一 segment
```

### I123

避免枚举 3^9 个 gap 序列。预生成固定 pattern bank：

```text
128 个长度 9 的 gap pattern
P(1)=0.25, P(2)=0.50, P(3)=0.25
限制总 span 在 12～24 raw frames
```

预处理阶段对每个 `(segment, pattern)` 计算合法 start，过滤：

```text
unique_keyframes >= 4
所有 frame/camera 可用
```

使用 CSR 保存：

```text
pattern_segment_offsets
valid_start_indices
```

预计数百万 int32，内存仅十几 MB，优于运行时 reject sampling。

## 3.5 Bootstrap Index

Bootstrap 必须完全独立于 Sequence10 eligibility。

```text
bootstrap_frames.npy = 所有 train scene / segment 中的合法 raw frame
```

采样层级：

```text
scene uniform -> segment uniform -> frame uniform
```

避免再次只覆盖少数 Sequence10 合法 segment。

---

# 4. Scheduler Schema

不沿用旧 batch dict 的隐式字段。新增不可变 dataclass。

```python
@dataclass(frozen=True)
class ObservationSpec:
    sequence_pos: int
    frame_idx: int
    keyframe_idx: int
    timestamp_us: int
    delta_t_sec: float
    frame_gap: int
    ego_delta_translation: tuple[float, float, float]
    ego_delta_yaw: float

    visit_kind: Literal["bootstrap", "causal", "repair"]
    repeat_budget: int

    temporal_read: bool
    temporal_commit: bool
    physical_time_advance: bool
    observation_commit: bool
    update_optimizer_memory: bool


@dataclass(frozen=True)
class RolloutPlan:
    phase: Literal["bootstrap", "causal", "repair"]
    observations: tuple[ObservationSpec, ...]
    current_positions: tuple[int, ...]
    history_positions: tuple[int, ...]
    detach_after_rollout: bool


@dataclass(frozen=True)
class EpisodePlan:
    episode_id: int
    scene_id: int
    segment_id: int
    protocol: Literal["D1", "D2", "I123"]
    observations: tuple[ObservationSpec, ...]  # length 10
    unique_keyframe_count: int
    frame_span: int
    time_span_sec: float
    sequence_hash: int
    rollouts: tuple[RolloutPlan, ...]
```

所有字段在 episode 创建时确定，model forward 内禁止修改 scheduler 语义。

---

# 5. 场景与协议采样

## 5.1 Hierarchical Fair Traversal

禁止从所有 windows 中 uniform choice，因为长 segment 和 D1 会被过采样。

实现三级公平队列：

```text
SceneRoundRobin
    -> ProtocolDeficitSampler
        -> SegmentRoundRobin(protocol)
            -> WindowRandomChoice
```

### Scene queue

每 epoch：

```text
对有任一可用 protocol 的 scene shuffle 一次
每 scene 使用一次后移到队尾
禁止连续同 scene（只有一个可用 scene 时除外）
```

### Protocol deficit

目标概率：

```text
D1 0.30
D2 0.40
I123 0.30
```

维护累计 deficit：

```python
deficit[p] += target_prob[p]
选择当前 scene 中可用且 deficit 最大的 protocol
deficit[selected] -= 1
```

这样不会因为 D1 windows 数量多而吞掉 D2/I123。

### Segment queue

每 `(scene, protocol)` 保持独立 shuffled segment queue；segment 使用后轮转。

### Window

在已选 segment/protocol 的预计算 window 列表中 O(1) 随机选择。

## 5.2 Coverage Fail-fast

启动时打印并验证：

```text
configured_scene_count
asset_scene_count
eligible_scene_count_by_protocol
eligible_segment_count_by_protocol
window_count_by_protocol
unique_keyframe_count histogram
frame_span histogram
time_span histogram
```

正式训练建议阈值：

```text
D1 train scenes >= 120
D2 train scenes >= 120
I123 train scenes >= 80
D1/D2 eval scenes >= 8
```

不足直接终止训练。

---

# 6. Episode 与 Rollout 固定协议

Stage 2.2 不再存在 shape matrix。

## 6.1 Bootstrap

```text
step 0～5000
1 raw frame / episode
R 从 {4,6,8} 采样：0.6/0.3/0.1
Temporal Mamba 完全 bypass
history 关闭
```

Bootstrap 中禁止无效 memory commit：

```text
temporal_read=false
temporal_commit=false
physical_time_advance=false
```

## 6.2 Causal Stream10

```text
10 raw frames
chronological
5 rollouts
每 rollout 2 frames
每 frame R=4
inner_K=8
```

固定划分：

```text
[0,1] [2,3] [4,5] [6,7] [8,9]
```

每 raw frame：

```text
repeat0：缓存 first-observation commit token
repeat0..3：读取 previous temporal state，不写
block exit：commit 一次
```

Temporal state 在整个 episode 内保持，rollout 边界仅 detach graph。

## 6.3 Repair

默认 `start_step=15000`，episode 概率 0.5。

```text
causal 结束后
随机非恒等 permutation
B10×R1
一个 rollout
```

Repair observation：

```text
temporal_read=true
temporal_commit=false
physical_time_advance=false
observation_commit=false
update_optimizer_memory=false
visit_kind=repair
```

Repair 只优化 LocalGSState，不污染真实时间 memory。

Stage 2.2 初版不加入 B5×R2 或更多 repair shape，避免重新产生 shape 系统。

---

# 7. Resolver 与 Batch Assembly

新增 `Stage22Resolver`，只接受 Stage 2.2 schema。

## 7.1 Evidence refs

每 observation：

```text
source frame × all configured cameras
```

Current supervision：

```text
当前 rollout 中全部 observation frames × all cameras
```

History supervision：

```text
已见且不在 current_positions 中的所有 positions
最多 10 frames / 30 refs
```

Repair supervision：

```text
全部10 frames / 30 refs
```

## 7.2 Leakage

Hard asserts：

```text
所有 refs 同 scene/segment
train 不含 test refs
current 覆盖所有输入 frames
history 与 current positions 不重叠
repair permutation 覆盖0..9且无重复
causal timestamps 严格递增
repair temporal_commit 全 false
```

## 7.3 Batch Metadata

Dataset batch 必须携带：

```text
stage2_2_episode_id
sequence_hash
protocol
rollout_phase
sequence_positions
frame_indices
keyframe_indices
timestamps_us
delta_t_sec
frame_gaps
ego_motion
visit_kinds
temporal_read_mask
temporal_commit_mask
physical_time_advance_mask
```

不再根据旧 `block_idx` 推断这些语义。

---

# 8. Temporal Mamba V2

旧 Temporal Mamba 的主要限制：

```text
- frame gap 仅支持 0/1/2；
- 没有真实 timestamp；
- 没有每-parent last-seen time；
- repair/causal 依赖外部隐式约定；
- rigid duplicate key 聚合可能不按 support 加权。
```

Stage 2.2 全量替换。

## 8.1 State Schema

```python
@dataclass
class DenseTemporalStateV2:
    conv_state: Tensor
    ssm_state: Tensor
    seen: BoolTensor
    last_timestamp_sec: Tensor

@dataclass
class KeyedTemporalStateV2:
    keys: LongTensor
    conv_state: Tensor
    ssm_state: Tensor
    seen: BoolTensor
    last_timestamp_sec: Tensor

@dataclass
class ParentTemporalStateV2:
    bg: DenseTemporalStateV2
    distant: DenseTemporalStateV2
    rigid: KeyedTemporalStateV2
```

状态在 episode 开始清空；rollout 边界 detach；episode 内不 reset。

## 8.2 Parent Identity

```text
BG：assignment-local parent id，dense
Distant：assignment-local parent id，dense
Rigid：hash(instance_id, global_parent_id)，keyed
```

Rigid near/out 多 active row 共享同一个 key 时：

```text
preview：共享相同历史 state
commit：按 support 加权聚合 observation token 后只写一次
```

## 8.3 Temporal Motion Embedding

删除 `nn.Embedding(3)`。

每个 parent 构造：

```text
sequence_delta_t_sec
key_delta_t_sec = current_time - last_seen_time[key]
log1p(frame_gap)
ego_delta_translation xyz
ego_translation_norm
sin(ego_delta_yaw)
cos(ego_delta_yaw)
visit_kind embedding
branch embedding
seen flag
```

编码：

```python
continuous = FourierFeatures(raw_time_motion)
time_motion_embed = MLP(continuous)
```

Repair 时：

```text
sequence_delta_t=0
key_delta_t=0
visit_kind=repair
physical_time_advance=false
```

## 8.4 API

```python
ctx, seen = temporal.preview(
    spatial_event,
    state,
    parent_keys,
    timestamp_sec,
    motion_meta,
    visit_kind,
)

new_state = temporal.commit(
    first_repeat_spatial_event,
    state,
    parent_keys,
    timestamp_sec,
    support,
    valid,
)
```

`preview()` 必须无状态修改；`commit()` 每 causal frame 最多一次。

## 8.5 Fusion

```python
temporal_delta = adapter(ctx)
gate = sigmoid(branch_gate + reliability_gate)
parent_event = spatial_event + seen * gate * temporal_delta
parent_event = LayerNorm(parent_event)
```

Unseen parent 的 temporal contribution 必须严格为零。

记录：

```text
temporal_to_spatial_norm_ratio
seen_ratio
commit_valid_ratio
key_delta_t_mean/p95
state_memory_mb
```

---

# 9. History Bank 与 Loss

## 9.1 EpisodeHistoryBankV2

每 sequence position 存储：

```text
seen
best_detached_loss
best_detached_psnr
last_detached_loss
last_detached_psnr
last_visit_rollout
```

每 causal rollout final：

```text
计算 current + all-seen history 的 per-position loss
更新 best/last bank
```

Repair final：

```text
评估全部10 positions
与 causal 后 best bank 比较
```

## 9.2 Loss

```math
L = L_current
  + λ_h L_history
  + λ_d L_best_damage
  + L_delta_reg
```

Role 独立归一化：

```text
L_current = current refs mean
L_history = history refs mean
```

Best damage：

```math
L_best_damage = mean_i relu(L_i - stopgrad(best_i) - margin)
```

推荐 schedule：

```text
0～5k：history 0
5k～15k：history 0 -> 0.5
15k～25k：repair启用，damage 0 -> 0.25
```

监控必须额外记录固定目标：

```text
fixed_monitor = current + 0.5*history + 0.25*damage
```

避免 warmup 导致 total loss 上升被误判。

---

# 10. 异步 Episode Producer 与 Preload

Scheduler 主线程的 `next()` 目标是 O(1) queue pop。

## 10.1 Episode Producer

```python
class Stage22EpisodeProducer:
    queue_depth = 32
    background thread
```

线程负责：

```text
scene/segment/protocol traversal
window选择
EpisodePlan构建
sequence hash
rollout plan构建
lightweight preload hint提交
```

它只读取 mmap index 和轻量 dataset metadata，不执行：

```text
segment bundle resolve
图像解码
Torch/CUDA操作
```

Queue 空时允许同步构建一次，但记录严重告警。

## 10.2 利用现有 Preload Manager

必须调用：

```python
dataset.build_preload_hint_light(...)
dataset.submit_preload_hint(...)
```

禁止调用同步 `_resolve_segment_bundle()` 的重 hint 构建。

Episode 已知全部10帧，可一次提交三层优先级：

```text
P0：当前 rollout exact refs，view meta + view pack
P1：下一 rollout exact refs，view meta + view pack
P2：余下 episode chain，view meta only
P3：下一 episode segment static
```

当前 `AssetPreloadManagerV2` 已具备：

```text
单独 worker thread
priority heap
dedupe
queue cap
stale/drop stats
scene meta / segment static / view meta / view pack warming
```

Stage 2.2 只扩展 hint scope 与优先级，不重新实现资源加载线程。

## 10.3 Prefetch Backpressure

```text
plan queue depth：32
preload max pending：沿用 manager 配置
若 preload queue >80%：只提交 current/next，停止 episode-chain
若 plan queue <4：producer 提升计划生成优先级
```

## 10.4 Scheduler State/Resume

Checkpoint 保存：

```text
epoch id
scene queue order/cursor
segment queue order/cursor
protocol deficits
RNG state
episode counter
index fingerprint
```

不保存 background queue 内容。Resume 时清空 queue，并根据 state deterministic refill。

---

# 11. 配置系统

Stage 2.2 使用全新根配置，不解析旧 scheduler 字段。

```yaml
scheduler_stage2_2:
  enable: true
  index:
    root: auto
    require_prebuilt: true
    mmap: true

  traversal:
    scene_order: shuffled_round_robin
    segment_order: shuffled_round_robin
    forbid_consecutive_scene: true

  bootstrap:
    end_step: 5000
    repeat_distribution:
      4: 0.6
      6: 0.3
      8: 0.1

  sequence:
    length: 10
    protocols:
      D1:
        probability: 0.30
        gap: 1
        min_unique_keyframes: 2
      D2:
        probability: 0.40
        gap: 2
        min_unique_keyframes: 3
      I123:
        probability: 0.30
        gap_values: [1, 2, 3]
        gap_probabilities: [0.25, 0.50, 0.25]
        min_unique_keyframes: 4
        min_frame_span: 12
        max_frame_span: 24

  causal:
    start_step: 5000
    frames_per_rollout: 2
    repeats_per_frame: 4
    temporal_commit: true

  repair:
    enable: true
    start_step: 15000
    episode_probability: 0.5
    repeats_per_frame: 1
    temporal_commit: false

  history:
    all_seen: true
    max_frames: 10

  producer:
    queue_depth: 32
    synchronous_fallback: true

  preload:
    current_rollout: view_pack
    next_rollout: view_pack
    remaining_episode: view_meta
    next_episode: segment_static
```

Unknown/legacy keys直接报错。

Temporal config：

```yaml
model:
  iforward:
    version: stage2_2
    parent_temporal_mamba_v2:
      event_dim: 64
      ctx_dim: 32
      model_dim: 32
      state_dim: 8
      conv_kernel: 2
      timestamp_mode: per_parent_last_seen
      time_motion_embedding:
        fourier_bands: 6
        output_dim: 32
        use_delta_t: true
        use_frame_gap: true
        use_ego_translation: true
        use_ego_yaw: true
        use_visit_kind: true
      commit:
        support_min: 0.001
        hard_valid: true
        token: first_repeat_spatial_event
```

---

# 12. 日志系统

## 12.1 Startup Coverage

```text
stage2_2/index/fingerprint
stage2_2/coverage/train_scenes_D1/D2/I123
stage2_2/coverage/train_segments_D1/D2/I123
stage2_2/coverage/train_windows_D1/D2/I123
stage2_2/coverage/eval_*
stage2_2/coverage/unique_keyframe_hist
```

## 12.2 Per Episode

强制 episode 事件日志，不依赖固定 step interval：

```text
scene_id / segment_id / episode_id
protocol
frame_indices
keyframe_indices
timestamps
gaps
unique_keyframe_count
frame_span
time_span
ego_motion_span
repair_enabled
sequence_hash
```

## 12.3 Per Rollout

```text
phase
positions
inner_K
current_ref_count
history_ref_count
temporal_commit_count
plan_queue_depth
plan_wait_ms
preload_submit_ms
batch_fetch_ms
```

强制记录：

```text
first causal rollout
last causal rollout
repair rollout
episode end
```

## 12.4 Loss

```text
loss/current_raw
loss/history_raw
loss/history_weighted
loss/best_damage_raw
loss/best_damage_weighted
loss/total_optimization
loss/fixed_weight_monitor
```

## 12.5 Temporal

```text
temporal/seen_ratio_by_branch
temporal/commit_rows_by_branch
temporal/commit_count_per_frame
temporal/delta_t_key_mean_p95
temporal/context_norm
temporal/spatial_norm
temporal/context_to_spatial_ratio
temporal/state_memory_mb
```

## 12.6 Scheduler Performance

```text
scheduler/next_ms
scheduler/queue_wait_ms
scheduler/queue_depth
scheduler/sync_fallback_count
scheduler/episode_build_ms
scheduler/index_lookup_ms
scheduler/protocol_actual_ratio
preload/tasks_completed/failed/dropped
preload/latency_ms
```

---

# 13. Validation 系统

预处理阶段生成固定 manifest：

```text
S10-D1-Causal
S10-D2-Causal
S10-I123-Causal
S10-D1-Repair
S10-D2-Repair
S10-Repeat-Stability
S10-Order-Robustness
```

每 protocol 独立 reset state，不共享 causal result，除非该 repair protocol 明确从其 causal checkpoint fork。

## 13.1 Causal

```text
10 raw frames
R4 per frame
chronological
无 repair
```

## 13.2 Repair

从相同 causal final state clone：

```text
3 个固定 random permutation
B10R1
no temporal commit
```

## 13.3 Repeat Stability

固定一帧和相同初始 state：

```text
R4 / R8 / R16 / R32
no temporal commit
```

## 13.4 Metrics

```text
final all10 PSNR mean/p10/min
first-frame final PSNR
best-to-final forget p90
retention AUC
current quality curve
repair gain
repair worst-frame regression
permutation std
R4->R32 quality drop
dynamic/static region metrics
```

快速 validation：

```text
每1000 step，D1/D2各2 sequences
```

完整 validation：

```text
每5000 step，D1/D2/I123各8～16 sequences，scene balanced
```

Manifest 无合法 sequence 必须在训练开始前 fail-fast。

---

# 14. 测试系统

## Index

```text
test_index_fingerprint_changes_on_asset_or_protocol_change
test_d1_d2_windows_match_bruteforce
test_irregular_patterns_chronological
test_unique_keyframe_filters
test_bootstrap_index_not_restricted_by_sequence_eligibility
test_mmap_roundtrip
```

## Traversal

```text
test_scene_round_robin
test_no_consecutive_scene_when_possible
test_segment_round_robin
test_protocol_ratio_deficit_sampler
test_window_count_does_not_bias_scene_probability
test_resume_determinism
```

## Scheduler

```text
test_episode_has_10_unique_raw_frames
test_all_frames_same_segment
test_causal_is_chronological
test_d1_d2_gap_exact
test_i123_gap_range
test_causal_rollout_is_5x_b2r4
test_repair_is_b10r1_permutation
test_repair_no_temporal_commit
test_bootstrap_no_temporal_state
```

## Temporal Mamba V2

```text
test_preview_no_state_mutation
test_commit_once_per_raw_frame
test_repeat_count_does_not_advance_time
test_per_key_last_seen_delta_t
test_unseen_context_zero
test_repair_no_timestamp_update
test_rigid_duplicate_keys_support_weighted
test_rollout_detach_preserves_values
test_episode_reset_clears_state
```

## History

```text
test_bank_update_best_per_position
test_all_seen_history_positions
test_repair_best_damage
test_role_normalization
test_history_warmup_skip_before_start
```

## Preload/Thread

```text
test_episode_producer_queue_order
test_queue_bounded
test_worker_exception_propagates
test_sync_fallback_counter
test_light_hint_does_not_resolve_segment_sync
test_preload_dedupe
test_resume_refills_deterministically
test_shutdown_no_deadlock
```

## Validation

```text
test_manifest_protocol_coverage
test_protocol_state_isolation
test_repair_forks_same_causal_state
test_repeat_stability_no_commit
test_metrics_per_position
```

---

# 15. 性能验收

预处理后 runtime scheduler 目标：

```text
index mmap startup < 2 s
scheduler next p50 < 0.2 ms
scheduler next p99 < 2 ms
plan queue miss < 0.1%
sync fallback < 1 / 1000 episodes
scheduler CPU占总step时间 < 0.5%
```

Preload 目标：

```text
batch_fetch p50 < 60 ms
batch_fetch p95 < 180 ms
cold episode first-rollout p95 < 300 ms
preload failed task = 0
```

计划生成线程不得创建 Torch tensor，不得触碰 CUDA。

---

# 16. 实施阶段

## Phase 1：Index 与 Schema

```text
- 新 index builder/loader
- raw frame protocol覆盖检查
- 新 immutable plan schema
- 删除旧 scheduler 注册
```

## Phase 2：Scheduler/Resolver/Preload

```text
- scene/segment/protocol fair traversal
- producer queue
- lightweight hints
- bootstrap + causal
- repair暂时关闭
```

## Phase 3：Temporal Mamba V2

```text
- state schema
- per-key timestamp
- motion/time embedding
- preview/commit
- rigid support-weighted keys
```

## Phase 4：History Bank 与 Repair

```text
- all-seen history
- best damage bank
- B10R1 repair
```

## Phase 5：Validation

```text
- fixed manifests
- causal/repair/stability/order protocols
```

## Phase 6：性能优化

```text
- queue/backpressure tuning
- prefetch priorities
- index compaction
- optional FWHR fused CUDA（仅显存或profile需要时）
```

---

# 17. 正式训练前的硬性验收

```text
1. Bootstrap 覆盖 >=120 train scenes。
2. D1/D2 Sequence10 各覆盖 >=120 train scenes。
3. Eval D1/D2 各覆盖 >=8 scenes。
4. Scheduler 不出现连续同scene（可避免时）。
5. D1/D2/I123 actual protocol ratio 在目标±3%。
6. Causal 每frame Temporal commit恰好1次。
7. Repair commit次数严格为0。
8. Sequence10 validation 能在train start运行。
9. Fixed monitor loss与current/history raw曲线可独立查看。
10. 所有Stage2_2测试通过，无旧scheduler兼容路径。
```

---

# 18. 最终判断

Stage 2_2 不应继续尝试让 keyframe 数量适配 10-frame 时序训练。正确的抽象是：

```text
Segment控制空间资产；
Keyframe描述空间覆盖；
Raw frame是时序observation；
Episode是同一segment内的10帧状态生命周期。
```

Temporal Mamba V2 必须理解真实时间间隔和 parent 的 last-seen 时间，而不是只接收离散 gap 1/2。

Scheduler 的性能问题应通过：

```text
预构建mmap索引
后台EpisodePlan producer
现有AssetPreloadManagerV2
lightweight hint
有界队列与backpressure
```

解决，而不是在训练主线程持续扫描和组装。
