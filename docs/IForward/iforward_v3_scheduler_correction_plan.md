# IForward v3 Scheduler 改正方案文档

版本：v3 scheduler correction draft  
目标结构：**episode reset + random-window rollout supervision + block + repeat**  
模型主线：**TimeAwarePointGRU + 5_4-style EMA history + final writeback history gate**

---

## 0. 核心结论

IForward v3 的 scheduler 不应该简单回到严格 chronological rollout，也不应该直接沿用当前 v15 的 `scheduler_iforward_v1` 或旧的 `random_window_v1`。新的 v3 scheduler 应该明确采用：

```text
segment
  → episode      # reset LocalGS / PointGRU / EMA history / visited target buffer
      → rollout  # random window, may repeat, one training batch, rollout-final supervision
          → block    # one source frame / source keyframe visit
              → repeat   # repeated update on same block
```

关键改正是：

```text
1. rollout 使用 random window with replacement，可以重复访问同一窗口。
2. episode 内 state 继续 carry，但语义必须从“时间递推”改成“同一 segment 的迭代式重建优化”。
3. history 不是 chronological past，而是 episode 内 previous visited frames。
4. TimeAwarePointGRU 的 dt 不能再用 source_frame_idx 差值，应改用 visit_clock / update_clock。
5. current 监督语义改为 rollout 内所有 input frames。
6. history 监督来自当前 rollout 之前已经 visited 的 frames，且默认排除 current refs，避免 role 冲突。
7. nearby 从当前 rollout 内随机 block 采一个未监督过的随机 frame。
8. EMA history 维持 5_4 风格：support / residual 在 block_exit commit；update_norm 每 repeat 更新。
```

因此，新 scheduler 最准确的名字建议是：

```text
scheduler_iforward.version = iforward_v3_random_window
```

不要继续叫 `iforward_v1`，因为 current/history 语义、random-window replacement、explicit block events、history replay target 都已经变了。

---

## 1. 设计边界

### 1.1 当前 v3 只解决 scheduler，不引入新模块

本方案只考虑：

```text
TimeAwarePointGRU
HistoryGate
EMA history
random-window scheduler
```

不考虑：

```text
Mamba
memory-xCPE
新的 history ledger
长序列全局 memory
复杂 replay buffer
```

### 1.2 v3 的随机窗口不是时间序列滤波

这是最重要的语义修正。

如果 rollout 可以 random window 并且可以重复，那么 episode 内的 state carry 不能再解释成：

```text
frame t → frame t+1 → frame t+2 的物理时间递推
```

而应该解释成：

```text
对同一个 local segment / local GS state 做多次随机 block visit 的优化过程。
```

也就是说：

```text
history = 在当前 episode 内已经被优化 / 验证过的 visited frames。
```

而不是：

```text
history = 时间上早于当前 source_frame_idx 的 frames。
```

这个定义允许：

```text
先 visit block 5，再 visit block 2。
```

这并不是“未来影响过去”，而是“同一 segment 的优化器用一个随机约束继续更新 3DGS”。

因此，scheduler 必须提供 **visit clock**，模型 memory 也必须使用 visit clock，而不是 source frame index 差值来做 time-aware decay。

---

## 2. 训练阶段与 rollout shape schedule

用户确认的最大 inner iteration 数是 8，因此所有 rollout shape 都满足：

```text
blocks_per_rollout * repeats_per_block <= 8
```

本方案采用用户 notation：

```text
rXbY = repeats_per_block = X, blocks_per_rollout = Y
```

对应当前代码 notation：

```text
r8b1 == b1_r8
r4b2 == b2_r4
r2b4 == b4_r2
```

### 2.1 推荐 shape schedule

| 训练阶段 | step 范围 | shape | inner_K | 目标 |
|---|---:|---|---:|---|
| Phase A | 0 - 20k | r2b1 / r4b1 / r6b1 / r8b1 | 2 / 4 / 6 / 8 | 先训练单帧迭代优化能力，同时让 history gate / EMA 管线运行起来 |
| Phase B | 20k - 40k | r8b1 / r4b2 | 8 / 8 | 从单帧优化过渡到短窗口两帧稳定 |
| Phase C | 40k 以后 | r8b1 / r4b2 / r2b4 | 8 / 8 / 8 | 保持单帧强能力，同时训练 2-block 与 4-block history stability |

如果总训练预算是 60k，Phase C 就是最后 20k。  
如果你说的“后 4w 轮”指的是 20k 之后的全部 40k，那么可以直接让 Phase C 从 20k 开始，跳过 Phase B。代码上建议保留两个 milestone，方便 ablation。

### 2.2 推荐初始概率

```yaml
scheduler_iforward:
  rollout:
    max_inner_K: 8
    shapes:
      # 0 - 20k: single-block iterative training
      - {name: r2b1, blocks_per_rollout: 1, repeats_per_block: 2, prob: 0.20}
      - {name: r4b1, blocks_per_rollout: 1, repeats_per_block: 4, prob: 0.30}
      - {name: r6b1, blocks_per_rollout: 1, repeats_per_block: 6, prob: 0.25}
      - {name: r8b1, blocks_per_rollout: 1, repeats_per_block: 8, prob: 0.25}

    shapes_schedule:
      - start_step: 20000
        shapes:
          - {name: r8b1, blocks_per_rollout: 1, repeats_per_block: 8, prob: 0.50}
          - {name: r4b2, blocks_per_rollout: 2, repeats_per_block: 4, prob: 0.50}

      - start_step: 40000
        shapes:
          - {name: r8b1, blocks_per_rollout: 1, repeats_per_block: 8, prob: 0.30}
          - {name: r4b2, blocks_per_rollout: 2, repeats_per_block: 4, prob: 0.40}
          - {name: r2b4, blocks_per_rollout: 4, repeats_per_block: 2, prob: 0.30}
```

这个 schedule 的意图是：

```text
前期不要过早把问题变成多帧平均；
中期开始训练“更新当前帧但不破坏已 visited frames”；
后期加入 4-block rollout，检查 history gate 能不能保持更长窗口稳定。
```

---

## 3. 推荐 scheduler 配置

### 3.1 主配置

```yaml
scheduler_iforward:
  enable: true
  version: iforward_v3_random_window
  fail_fast: true

  traversal:
    traversal_mode: episode_serial
    scene_order: shuffle_per_epoch
    segment_order: shuffle_per_epoch
    fixed_scene_id: null
    fixed_segment_id: null
    seed: null

  episode:
    source_mode: keyframes
    blocks_per_episode: 8
    episode_stride: 8
    allow_short_last_episode: false
    min_blocks_per_episode: 8

    # One episode owns one local-GS / memory / EMA-history lifetime.
    reset_scene_state_policy: episode_begin
    reset_optimizer_memory_policy: episode_begin
    reset_history_ema_policy: episode_begin
    reset_visited_target_buffer_policy: episode_begin

    # Random-window training needs a rollout budget.
    # Episode ends by rollout budget, not by block_cursor.
    rollouts_per_episode: 8

    # Each block visit samples the actual source frame from the keyframe group.
    # This allows repeated visits to the same keyframe with different frames.
    block_source_frame_policy: random_within_keyframe_per_visit

  rollout:
    window_policy: random_with_replacement
    delivery_order_policy: chronological_inside_window
    detach_graph_after_rollout: true
    allow_short_final_rollout: false
    min_blocks_per_rollout: 1
    max_inner_K: 8

    shapes:
      - {name: r2b1, blocks_per_rollout: 1, repeats_per_block: 2, prob: 0.20}
      - {name: r4b1, blocks_per_rollout: 1, repeats_per_block: 4, prob: 0.30}
      - {name: r6b1, blocks_per_rollout: 1, repeats_per_block: 6, prob: 0.25}
      - {name: r8b1, blocks_per_rollout: 1, repeats_per_block: 8, prob: 0.25}

    shapes_schedule:
      - start_step: 20000
        shapes:
          - {name: r8b1, blocks_per_rollout: 1, repeats_per_block: 8, prob: 0.50}
          - {name: r4b2, blocks_per_rollout: 2, repeats_per_block: 4, prob: 0.50}
      - start_step: 40000
        shapes:
          - {name: r8b1, blocks_per_rollout: 1, repeats_per_block: 8, prob: 0.30}
          - {name: r4b2, blocks_per_rollout: 2, repeats_per_block: 4, prob: 0.40}
          - {name: r2b4, blocks_per_rollout: 4, repeats_per_block: 2, prob: 0.30}

  evidence:
    camera_policy: all_cams
    allow_camera_dropout: false
    mask_policy: non_sky_non_egocar

  memory:
    observation_commit_policy: first_repeat_only
    optimizer_memory_update_policy: every_repeat
    reset_policy: episode_begin
    carry_policy: across_rollouts_until_episode_end

  loss_timing:
    policy: rollout_final_only
    intermediate_step_loss: false

  supervision:
    current:
      enable: true
      role_name: final_current_recon
      frame_policy: all_rollout_input_frames
      camera_policy: all_cams
      required: true

    history_replay:
      enable: true
      role_name: final_history_replay
      source: episode_visited_before_rollout
      frame_policy: visited_frames_excluding_current
      sample_policy: uniform_keep_order
      max_frames_per_rollout: 8
      camera_policy: all_cams
      required: false

    nearby:
      enable: true
      role_name: final_nearby_rollout
      scope: current_rollout_random_block
      policy: random_unsupervised_frame_in_random_rollout_block
      frames_per_rollout: 1
      insufficient_policy: use_available_or_skip_if_none
      camera_policy: all_cams
      max_refs_per_rollout: 24
      add_to_evidence: false
      mask_policy: non_sky_non_egocar
```

### 3.2 和当前 v15 配置的区别

| 项目 | v15 当前 `iforward_v3_gru_history_gate.yaml` | v3 改正后 |
|---|---|---|
| scheduler version | `iforward_v1` | `iforward_v3_random_window` |
| random window | `random_start_contiguous`，但不是 true replacement | `random_with_replacement` |
| shape | `b4_r2`, `b6_r1` | 用户指定的 `r2b1/r4b1/r6b1/r8b1 → r8b1/r4b2 → r8b1/r4b2/r2b4` |
| current 语义 | 代码里仍会拆成 latest/current + in-rollout-history | current = rollout 内所有 input frames |
| history target | 当前 v1 scheduler 没有真正 history target | 来自 episode 内 previous visited frames，先采 frame 再展开 all cams |
| nearby | rollout keyframe span 的 non-input | 当前 rollout 内随机 block 的 unsupervised frame |
| block exit | model 里用 next source_frame 推断 | scheduler 显式给 `is_block_exit` |
| GRU dt | 当前代码用 `source_frame_idx - last_frame_idx` | 改成 `visit_clock - last_visit_clock` |
| episode end | block cursor 到尾 | `rollouts_per_episode` 达到预算 |
| repeated window | 当前 v1 尝试避免 used starts | 明确允许重复，并记录 revisit count |

---

## 4. Scheduler 事件语义

### 4.1 Episode begin

episode begin 时必须 reset：

```text
LocalGS state
TimeAwarePointGRU state
EMA history state
scheduler visited-target buffer
window revisit counters
block visit counters
episode visit clock
```

代码层面：

```python
plan.reset_scene_state_before_rollout = rollout_idx_in_episode == 0
```

trainer 看到这个 flag 后：

```python
clear state_cache[(scene, segment, episode)]
reset bridge runtime node state
```

### 4.2 Rollout begin

每个 rollout：

```text
1. sample active rollout shape by global_step schedule
2. sample random contiguous window with replacement
3. sample source frame for each block visit
4. build current targets from current rollout input frames
5. build history targets from episode.visited_frames before this rollout, then expand to all cams
6. build nearby target from one random block in current rollout
7. build explicit step plan
```

### 4.3 Block enter

每个 block 的 repeat 0：

```text
commit_observation_memory = true
is_block_enter = true
```

模型行为：

```python
if step.is_block_enter:
    history_ema.record_block_support_snapshot(event, local_state)
```

注意：support 只 snapshot 一次，不随 repeat 多次刷新。

### 4.4 Repeat update

每个 repeat 都执行：

```text
observe
build event
PointGRU read
predict delta_hat
history gate
gate delta
apply delta
record update_norm EMA
PointGRU write
```

其中：

```python
if step.record_update_norm:
    history_ema.record_update_norm(delta_applied.detach())
```

必须使用 gate 之后、expand rigid 之后、实际 apply 的 delta。

### 4.5 Block exit

每个 block 的最后一个 repeat：

```text
is_block_exit = true
is_frame_exit = true
```

模型行为：

```python
if step.commit_residual_on_exit:
    residual_pack = compute_block_residual_history(...)
    history_ema.commit_residual(residual_pack.detach())

if step.commit_support_on_exit:
    history_ema.commit_block_support(detach=True)
```

`step.is_block_exit` 只描述 block event。v3 主路径中，是否 commit support/residual 必须由 scheduler 发出的 `commit_support_on_exit` 和 `commit_residual_on_exit` 决定；`is_block_exit` 只作为旧 resolver 兼容 fallback。

scheduler 行为：

```text
把本 block 的 evidence refs 加入 episode.visited_refs，用于后续 rollout 的 history supervision。
```

注意：scheduler 的 visited target buffer 与模型里的 EMA history 是两个不同概念。

| 名称 | 位置 | 作用 | 更新时刻 |
|---|---|---|---|
| EMA history | model state | 给 history gate 提供 per-point 保护统计 | block_exit / every repeat update_norm |
| visited target buffer | scheduler state | 给 final loss 提供 previous visited frame targets | rollout emission 后记录 current evidence refs |

### 4.6 Rollout exit

rollout final supervision：

```text
current loss: 当前 rollout 所有 input frames
history loss: 当前 rollout 之前已经 visited 的 frames
nearby loss: 当前 rollout 内随机 block 中未监督过的 random frame
```

然后：

```python
loss.backward()
optimizer.step()
next_state = out.next_state.detach_for_next_rollout()
```

如果不是 episode end：

```python
state_cache[(scene, segment, episode)] = next_state
```

如果 episode end：

```python
pop state_cache
reset runtime node state
```

---

## 5. StepPlan / RolloutPlan 数据结构修改

### 5.1 新 StepPlan

当前 `IForwardStepPlan` 缺少 `block_id / is_block_exit`，model 只能通过 next frame 推断 block exit。v3 必须显式加入。

```python
@dataclass(frozen=True)
class IForwardV3StepPlan:
    step_idx: int

    # identity
    episode_block_idx: int        # block id inside episode window
    block_id: int                 # alias of episode_block_idx, for random-window compatibility
    rollout_block_rank: int       # 0..blocks_per_rollout-1
    repeat_idx: int
    repeats_per_block: int

    # visit clocks
    episode_visit_idx: int        # monotonically increasing block-visit clock inside episode
    rollout_visit_idx: int        # block rank inside rollout
    optimizer_step_idx_in_episode: int

    # source
    source_keyframe_idx: int
    source_frame_idx: int
    evidence_refs: List[ImageRef]
    evidence_frame_indices: List[int]
    evidence_cam_indices: List[int]

    # events
    is_block_enter: bool
    is_block_exit: bool
    is_frame_exit: bool
    commit_observation_memory: bool
    update_optimizer_memory: bool
    record_update_norm: bool
    commit_support_on_exit: bool
    commit_residual_on_exit: bool

    # random-window metadata
    window_start: int
    window_end: int
    window_hash: int
    window_revisit_count: int
    is_repeated_window: bool
    block_visit_count_before: int
    block_visit_count_after: int

    # position codes
    rollout_pos_code: float
    frame_pos_code: float
    repeat_pos_code: float
```

最重要的是：

```python
is_block_exit = repeat_idx == repeats_per_block - 1
```

不要再在 model 里用：

```python
next_step.source_frame_idx != step.source_frame_idx
```

来推断 frame exit。

### 5.2 新 FinalSupervisionPlan

```python
@dataclass(frozen=True)
class IForwardV3FinalSupervisionPlan:
    refs: List[ImageRef]
    roles: List[str]

    current_frames: List[int]
    current_refs: List[ImageRef]

    history_frames: List[int]
    history_refs: List[ImageRef]
    history_ref_count_before_dedupe: int
    history_skipped: bool
    history_skip_reason: str

    nearby_frames: List[int]
    nearby_refs: List[ImageRef]
    nearby_block_id: int
    nearby_skipped: bool
    nearby_skip_reason: str

    current_ref_count: int
    history_ref_count: int
    nearby_ref_count: int
```

### 5.3 新 RolloutPlan

```python
@dataclass(frozen=True)
class IForwardV3RolloutPlan:
    scheduler_version: str
    model_family: str

    scene_id: int
    segment_id: int
    episode_id: int
    rollout_id_global: int
    rollout_idx_in_episode: int
    rollouts_per_episode: int

    keyframe_window: List[int]
    episode_num_blocks: int

    shape_name: str
    blocks_per_rollout: int
    repeats_per_block: int
    inner_K: int

    window_policy: str
    window_start: int
    window_end: int
    window_block_ids: List[int]
    window_keyframe_indices: List[int]
    window_frame_indices: List[int]
    window_hash: int
    window_revisit_count: int
    unique_windows_seen: int
    is_repeated_window: bool

    steps: List[IForwardV3StepPlan]
    final_supervision: IForwardV3FinalSupervisionPlan

    reset_scene_state_before_rollout: bool
    carry_scene_state_after_rollout: bool
    episode_end_after_rollout: bool
    detach_graph_after_rollout: bool

    evidence_refs_flat: List[ImageRef]
    target_refs_flat: List[ImageRef]
    target_roles_flat: List[str]

    request_meta: Dict[str, Any]
    leakage_check: Dict[str, Any]
```

---

## 6. Random-window sampling 规则

### 6.1 Window with replacement

v3 要求 window 可以重复，所以不要使用当前 v1 的逻辑：

```python
used_starts = episode.used_rollout_starts
available = starts not in used_starts
if not available:
    available = all starts
```

这不是 true replacement。

v3 应该使用：

```python
def sample_window_start(episode, shape, rng):
    max_start = len(episode.keyframe_window) - shape.blocks_per_rollout
    return rng.randint(0, max_start)
```

然后记录：

```python
window_hash = stable_window_hash(scene_id, segment_id, window_block_ids)
prev_count = episode.window_counts.get(window_hash, 0)
episode.window_counts[window_hash] = prev_count + 1
is_repeated_window = prev_count > 0
window_revisit_count = prev_count
```

### 6.2 Episode end 不再由 block_cursor 决定

random-window replacement 下，`block_cursor` 不再有意义。episode end 应该由 rollout budget 决定：

```python
episode_end = rollout_idx_in_episode + 1 >= rollouts_per_episode
```

对应：

```python
reset_before = rollout_idx_in_episode == 0
carry_after = not episode_end
```

### 6.3 Delivery order

window 内仍然按 block id 从小到大交付：

```text
window_start, window_start+1, ..., window_end-1
```

这保持局部 temporal coherence，但 episode 内 rollout 之间允许随机。

---

## 7. Supervision 目标构建

### 7.1 Current：当前 rollout 内所有 input frames

新的 current 语义：

```text
current = 当前 rollout 涉及的所有 source frames，all cams。
```

不再是 latest frame。

```python
current_refs = refs_for_frames(num_cams, input_frames)
```

对应 role：

```text
final_current_recon
```

### 7.2 History：当前 rollout 前已经 visited 的 frames

history target 来自 scheduler episode state：

```python
episode.visited_refs
```

构建规则：

```python
history_candidates = episode.visited_refs_before_this_rollout
history_candidates = history_candidates - set(current_refs)
```

默认排除 current refs，原因是当前 resolver / target role 系统不应该让一个 ref 同时拥有 current 和 history 两个 role。重复 window 的情况下，current loss 已经覆盖这些 frames；是否是 revisit 通过 metadata 和 metrics 记录。

建议 sampling policy：

```text
all_until_cap_then_recent_uniform_mix
```

伪代码：

```python
def sample_history_refs(visited_refs, current_refs, max_frames, num_cams, rng):
    current_set = set(current_refs)
    candidates = [ref for ref in visited_refs if ref not in current_set]
    frames = unique_frames_keep_order(candidates)

    if max_frames <= 0 or len(frames) <= max_frames:
        selected_frames = frames
    else:
        recent_n = max_frames // 2
        uniform_n = max_frames - recent_n
        recent = frames[-recent_n:]
        pool = frames[:-recent_n]
        uniform = rng.sample(pool, k=min(uniform_n, len(pool)))
        selected_frames = sorted(set(uniform + recent), key=lambda f: frames.index(f))

    return refs_for_frames(num_cams, selected_frames)
```

Role：

```text
final_history_replay
```

如果没有 previous visited frames：

```text
history_skipped = true
history loss = 0
```

### 7.3 Nearby：当前 rollout 内随机 block 中未监督过的随机 frame

用户定义的 nearby 是：

```text
当前 rollout 内随机 block 中没监督过的随机帧。
```

因此，不应该从整个 segment 或 rollout span 任意选；应该先选当前 window 内的一个 block，再从该 block 对应 keyframe 的候选 train frames 里采样。

伪代码：

```python
def sample_nearby_for_rollout(window_block_ids, keyframe_window, input_frames, current_refs, history_refs, rng):
    supervised_frames = set(frame for frame, _ in current_refs + history_refs)
    candidate_blocks = shuffled(window_block_ids)

    for block_id in candidate_blocks:
        keyframe_idx = keyframe_window[block_id]
        candidates = keyframe_train_frames(keyframe_idx)
        candidates = [f for f in candidates if f not in supervised_frames]
        if candidates:
            nearby_frame = rng.choice(candidates)
            return [nearby_frame], refs_for_frames(num_cams, [nearby_frame]), block_id

    return [], [], -1
```

Role：

```text
final_nearby_rollout
```

Leakage rule：

```text
nearby must not overlap evidence refs
nearby must not overlap current refs
nearby must not overlap history refs
nearby must come from one of current rollout's blocks
```

---

## 8. 5_4-style EMA history 记录规则

v3 继续使用原 EMA，不引入 ledger。

### 8.1 Support

Support 表示当前 block 的观测证据。记录规则：

```text
repeat 0 / block_enter:
    record support snapshot

block_exit:
    commit support EMA, detach
```

不要每个 repeat 都更新 support EMA。否则单个 block 的多次 repeat 会被误认为多次 visit。

推荐模型流程：

```python
if step.is_block_enter:
    history_ema.record_block_support_snapshot(event, local_state)

...

if step.is_block_exit:
    history_ema.commit_block_support(...)
```

### 8.2 Residual

Residual 表示当前 block 在完成所有 repeat 后的 post-update 拟合质量。

记录规则：

```text
block_exit after apply_delta:
    render current source frame
    backproject residual to points
    commit residual EMA, detach
```

不要在 repeat 中间状态记录 residual。

### 8.3 Update norm

Update norm 表示实际写入幅度，必须每次 repeat 更新。

记录规则：

```text
after gated delta apply, every repeat:
    record update_norm EMA
```

必须用：

```python
delta_applied = gate(delta_hat)
```

不能用：

```python
delta_hat
```

否则 gate 阻止掉的更新也会被当成真实破坏风险。

### 8.4 Detach 规则

所有 history EMA 写入都应该 detach / no_grad：

```python
with torch.no_grad():
    history_ema.update(...)
```

History 是 gate 的 state，不应该让反向梯度穿过过去 block 的 EMA 写入。

---

## 9. TimeAwarePointGRU 的 scheduler 相关修正

当前 v15 的 `IForwardTimeAwarePointGRU` 使用：

```python
dt = source_frame_idx - last_frame_idx
dt = clamp_min(dt, 0)
```

在 random-window with replacement 下，这是错误的。因为 episode 内可能出现：

```text
rollout 0: block 5
rollout 1: block 2
```

如果继续用 source frame 差值，负 dt 会被 clamp 成 0，导致 GRU 误以为两个 observation 没有时间间隔。

### 9.1 改正原则

GRU decay 使用：

```text
visit_clock dt
```

不是：

```text
source_frame_idx dt
```

即：

```python
dt_visit = step.episode_visit_idx - state.last_visit_idx
```

source frame gap 可以作为额外 feature，但不能作为 decay clock。

### 9.2 State 字段修改

当前：

```python
IForwardGRUBranchState:
    h
    seen
    last_frame_idx
```

改成：

```python
IForwardGRUBranchState:
    h
    seen
    last_visit_idx
    last_source_frame_idx   # optional feature/debug, not decay clock
```

### 9.3 Read / write 修改

Read：

```python
dt = episode_visit_idx - last_visit_idx
dt = where(seen, clamp(dt, 0, dt_clip), 0)
h_prior = h_seen * exp(-rate * dt)
```

Write：

```python
last_visit_idx = where(write_mask, episode_visit_idx, last_visit_idx)
last_source_frame_idx = where(write_mask, source_frame_idx, last_source_frame_idx)
```

### 9.4 StepContext 修改

`IForwardMemoryStepContext` 应增加：

```python
episode_visit_idx: int
optimizer_step_idx_in_episode: int
block_id: int
block_visit_count_before: int
window_revisit_count: int
is_block_enter: bool
is_block_exit: bool
```

---

## 10. Resolver 修改

当前 resolver 有一个旧语义：

```python
latest_input_frame_idx = input_frame_indices[-1]
current_latest_target_indices = current refs for latest frame
history_rollout_target_indices = current refs for input_frame_indices[:-1]
```

v3 要改掉这个语义。

### 10.1 新 role mapping

```text
current_target_indices = role == final_current_recon
history_target_indices = role == final_history_replay
nearby_target_indices  = role == final_nearby_rollout
```

为兼容当前 model，可以临时映射：

```python
resolved.current_latest_target_indices = resolved.current_target_indices
resolved.history_rollout_target_indices = resolved.history_target_indices
```

但文档和日志里应该逐渐改名：

```text
current_latest → current
in_rollout_history → history
history_rollout → history
```

### 10.2 不再自动把 input_frames[:-1] 当 history

这段逻辑必须删除或只作为 legacy fallback：

```python
history_frames = input_frame_indices[:-1]
history_rollout_indices = current_indices whose frame in history_frames
```

v3 中：

```python
history_rollout_indices = role_indices[history_role]
```

### 10.3 Current coverage validation

改成：

```python
expected_current_refs = refs_for_frames(num_cams, input_frame_indices)
actual_current_refs = refs with role final_current_recon
assert actual_current_refs == expected_current_refs
```

### 10.4 Nearby leakage validation

新增：

```python
nearby_refs & evidence_refs == empty
nearby_refs & current_refs == empty
nearby_refs & history_refs == empty
nearby block id in current window_block_ids
```

### 10.5 History validation

新增：

```python
history_refs subset of visited_refs_before_rollout
history_refs disjoint current_refs
history_refs disjoint nearby_refs
```

---

## 11. Model forward 修改

### 11.1 不再推断 block_exit

当前 v15 里：

```python
next_step = resolved.steps[step_pos + 1] if step_pos + 1 < len(resolved.steps) else None
is_frame_exit = next_step is None or int(next_step.source_frame_idx) != int(step.source_frame_idx)
```

v3 改成：

```python
is_block_exit = bool(step.is_block_exit)
is_frame_exit = bool(step.is_frame_exit)
```

兼容 fallback 可以保留，但主线必须依赖 scheduler event。

### 11.2 Current loss 使用所有 current target

`_render_final_losses()` 当前代码用：

```python
current_indices = list(resolved.current_latest_target_indices)
```

v3 可以暂时让 resolver 把 `current_latest_target_indices` alias 到 all current。更干净的改法：

```python
current_indices = list(resolved.current_target_indices)
```

### 11.3 History loss 使用 scheduler-provided history target

```python
history_indices = list(resolved.history_target_indices)
```

临时兼容：

```python
history_indices = list(resolved.history_rollout_target_indices)
```

但其语义已经不是 in-rollout previous frames，而是 previous visited frames before rollout。

### 11.4 StepContext 使用 visit clock

```python
step_context = IForwardMemoryStepContext(
    step_idx=step.step_idx,
    source_frame_idx=step.source_frame_idx,
    episode_visit_idx=step.episode_visit_idx,
    optimizer_step_idx_in_episode=step.optimizer_step_idx_in_episode,
    block_id=step.block_id,
    commit_observation_memory=step.commit_observation_memory,
    update_optimizer_memory=step.update_optimizer_memory,
    is_block_enter=step.is_block_enter,
    is_block_exit=step.is_block_exit,
    is_frame_exit=step.is_frame_exit,
    repeat_pos_code=step.repeat_pos_code,
    frame_pos_code=step.frame_pos_code,
    rollout_pos_code=step.rollout_pos_code,
    global_step=global_step,
)
```

### 11.5 GRU write invariant

TimeAwarePointGRU 的 decay clock 是 `episode_visit_idx`，不是 `source_frame_idx`。如果 write 阶段把 read 得到的 decayed `h_prior` 持久写回 state，即使该 row 的 `write_mask=False`，也必须同步推进该 row 的 `last_visit_idx`：

```python
if write_mask[row]:
    h[row] = h_candidate[row]
    last_visit_idx[row] = step.episode_visit_idx
    last_source_frame_idx[row] = step.source_frame_idx
elif seen_or_touched[row]:
    h[row] = h_prior[row]
    last_visit_idx[row] = step.episode_visit_idx
    # last_source_frame_idx unchanged
```

否则下一次 read 会再次覆盖同一段 elapsed visit interval，导致 double-decay。`last_source_frame_idx` 只保留 debug / feature metadata，不参与 dt 计算。

---

## 12. Scheduler episode state

v3 scheduler 的 `_current_episode` 应该包含：

```python
episode = {
    "scene_id": int,
    "segment_id": int,
    "episode_id": int,
    "episode_start_keyframe_pos": int,
    "keyframe_window": List[int],
    "num_cams": int,

    "rollout_idx_in_episode": int,
    "rollouts_per_episode": int,

    # random-window state
    "window_counts": Dict[int, int],
    "unique_windows_seen": int,

    # visited target buffer for history supervision
    "visited_refs": List[ImageRef],
    "visited_frames": List[int],
    "visited_frame_set": Set[int],
    "visited_ref_set": Set[ImageRef],

    # block visit stats
    "block_visit_counts": Dict[int, int],
    "episode_visit_idx": int,
    "optimizer_step_idx_in_episode": int,

    # deterministic RNG state for resume / peek
    "episode_rng_state": rng.getstate(),
}
```

### 12.1 visited_refs 更新时机

为了保证当前 rollout 的 history targets 只来自“rollout 之前 visited”的 frames：

```text
build plan 时先读取 visited_frames_before_rollout；
构建 current/history/nearby targets；
plan emission 后，再把 current evidence refs 加入 episode.visited_refs，并按首次出现顺序加入 episode.visited_frames。
```

也就是说：

```python
history_frames = sample_from(episode.visited_frames)
history_refs = expand_all_cams(history_frames)
plan = build_rollout_plan(...)
episode.visited_refs += plan.evidence_refs_flat
```

这样当前 rollout 内的 frames 不会被误当成 history target。

history replay 语义是 frame-level：先从已经 visited 的 frame 中排除 current frames，再采样 frame，最后展开该 frame 的所有 camera refs。除非显式做 ablation，否则不应该独立采样单个 `(frame_idx, cam_idx)` ref。

### 12.2 repeated current 的处理

如果 current refs 已经在 visited_refs 里：

```python
current_revisit_refs = set(current_refs) & set(visited_refs_before_rollout)
```

不要把它们再加入 history role；只记录 metadata：

```python
current_revisit_ref_count
current_revisit_frame_count
is_current_revisit
```

这对分析 repeated-window stability 很重要。

---

## 13. 训练流程示例

假设 episode keyframes：

```text
K0 K1 K2 K3 K4 K5 K6 K7
```

`rollouts_per_episode = 4`，Phase C shape 可能采到：

```text
rollout 0: r8b1, window [5]
rollout 1: r4b2, window [2,3]
rollout 2: r2b4, window [3,4,5,6]
rollout 3: r8b1, window [5]    # repeated window/block
```

### Rollout 0

| block | repeat | current | history | nearby |
|---|---:|---|---|---|
| K5 | 8 | frame sampled from K5 | empty | random unsupervised frame from K5 |

block_exit 后：

```text
visited_refs += K5 source refs
EMA support/residual commit for K5
update_norm recorded every repeat
```

### Rollout 1

| block | repeat | current | history | nearby |
|---|---:|---|---|---|
| K2, K3 | 4 each | K2 frame + K3 frame | previous K5 refs | random unsupervised frame from K2 or K3 |

block_exit 后：

```text
visited_refs += K2/K3 source refs
```

### Rollout 2

| block | repeat | current | history | nearby |
|---|---:|---|---|---|
| K3, K4, K5, K6 | 2 each | all 4 current frames | previous K5/K2/K3 refs excluding current exact refs | random unsupervised frame from one of K3/K4/K5/K6 |

如果 K5 当前 frame 与 Rollout 0 的 K5 frame 是同一个 exact frame：

```text
它只作为 current，不重复作为 history。
```

### Rollout 3

Repeated window / repeated block：

```text
window_revisit_count > 0
block_visit_count_before for K5 > 0
current_revisit_ref_count may be > 0
```

此时可以比较 repeated-window 的 current PSNR / history PSNR delta，用来诊断是否越优化越糊。

---

## 14. 与 5_4 的对应关系

| 5_4 概念 | IForward v3 对应 |
|---|---|
| episode reset | episode_begin reset LocalGS / memory / history |
| block visit | random-window 内的一个 block |
| block_exit history commit | v3 block last repeat 后 commit support/residual |
| update_norm after update | v3 every repeat after gated delta apply |
| visited_episode_frames target | v3 history target = previous visited refs before rollout |
| history gate final writeback | v3 `delta_hat → history_gate → apply_delta` |
| scheduler 可以 revisit | v3 random-window with replacement + window_revisit_count |

核心一致点：

```text
一个 block visit 完成后，它才成为 history；
后续 update 必须通过 history gate 保护已经 visited 的内容。
```

差异点：

```text
5_4 更接近 block/episode 顺序训练；
IForward v3 使用 random-window rollout-final supervision，rollout 是反传单位。
```

---

## 15. 与当前 v15 代码的具体改动清单

### P0：新建或替换 scheduler class

建议新建：

```text
datasets/train_scheduler_iforward_v3.py
```

不要直接复用旧 `iforward_random_window_scheduler.py`，因为它有硬编码：

```text
blocks_per_rollout == 4
repeats_per_block == 2
input_frame_indices exactly 4
current_latest / in_rollout_history 旧语义
```

也不要直接复用当前 `train_scheduler_iforward.py` 的 `random_start_contiguous`，因为它：

```text
不是真 replacement；
没有 previous visited history target；
没有 explicit is_block_exit；
current 语义仍是旧 resolver 语义；
block_cursor 与 random window 混在一起。
```

最小可行做法：

```text
以 train_scheduler_iforward.py 为基础，加入 v3 mode；
以 iforward_random_window_scheduler.py 的 window_hash / revisit_count / is_frame_exit 为参考；
删除 random_start 的 used_start avoidance；
加入 visited_refs buffer 和 history role。
```

### P0：StepPlan 加 explicit block events

必须加入：

```text
block_id
is_block_enter
is_block_exit
is_frame_exit
episode_visit_idx
optimizer_step_idx_in_episode
window_hash
window_revisit_count
block_visit_count_before
```

### P0：Resolver 改 current/history 角色

必须改：

```text
current = role final_current_recon 的所有 target refs
history = role final_history_replay 的 target refs
nearby = role final_nearby_rollout 的 target refs
```

不要再把 latest frame 才当 current。

### P0：Model 使用 explicit is_block_exit

必须改：

```python
is_frame_exit = bool(step.is_frame_exit)
```

不要用 next source frame 推断。

### P0：GRU dt clock 改成 visit clock

必须改：

```text
last_frame_idx → last_visit_idx
source_frame_idx 差值不用于 decay
```

否则 random-window repeat 会污染 PointGRU memory。

### P1：Target role 与 loss stats rename

短期可以兼容旧 key：

```text
current_latest_target_indices = current_target_indices
history_rollout_target_indices = history_target_indices
```

但日志建议改成：

```text
iforward/current_psnr
iforward/history_psnr
iforward/nearby_psnr
```

而不是：

```text
current_latest_psnr
in_rollout_history_psnr
```

### P1：Scheduler state_dict 支持 resume

`state_dict()` 必须包含：

```text
_current_episode.visited_refs
_current_episode.window_counts
_current_episode.block_visit_counts
_current_episode.episode_visit_idx
_current_episode.optimizer_step_idx_in_episode
_current_episode.episode_rng_state
```

### P1：Leakage check 更新

新增检查：

```text
current covers all input frames
history comes only from prior visited refs
history excludes current refs
nearby excludes current/history/evidence refs
nearby comes from current rollout block
role count matches ref count
```

---

## 16. 伪代码：核心 scheduler

```python
class TrainSchedulerIForwardV3:
    def next_batch(self):
        episode = self._ensure_episode()
        shape = self._sample_shape(global_step=self.global_step)
        plan = self._build_random_window_rollout(episode, shape)

        batch = self._batch_from_plan(plan)

        # Important: update scheduler visited buffer after plan target construction.
        self._commit_rollout_visits_to_episode(episode, plan)

        episode["rollout_idx_in_episode"] += 1
        self.global_step += 1
        self._rollout_id_global += 1

        if plan.episode_end_after_rollout:
            self._emit_episode_end(...)
            self._current_episode = None

        return batch
```

```python
def _build_random_window_rollout(self, episode, shape):
    rng = self._episode_rng(episode)

    start = rng.randint(0, len(episode.keyframe_window) - shape.blocks_per_rollout)
    block_ids = list(range(start, start + shape.blocks_per_rollout))
    window_hash = stable_window_hash(..., block_ids)
    revisit_count = episode.window_counts.get(window_hash, 0)

    input_frames = []
    for block_id in block_ids:
        keyframe_idx = episode.keyframe_window[block_id]
        frame_idx = sample_frame_for_keyframe(keyframe_idx, rng)
        input_frames.append(frame_idx)

    current_refs = refs_for_frames(num_cams, input_frames)
    history_refs = sample_history_refs(
        visited_refs=episode.visited_refs,
        current_refs=current_refs,
        max_frames=cfg.history.max_frames_per_rollout,
        rng=rng,
    )
    nearby_refs = sample_nearby_from_current_window(
        block_ids=block_ids,
        input_frames=input_frames,
        supervised_refs=current_refs + history_refs,
        rng=rng,
    )

    steps = []
    for block_rank, block_id in enumerate(block_ids):
        block_visit_before = episode.block_visit_counts.get(block_id, 0)
        episode_visit_idx_for_block = episode.episode_visit_idx + block_rank
        for repeat_idx in range(shape.repeats_per_block):
            step_idx = len(steps)
            steps.append(IForwardV3StepPlan(
                step_idx=step_idx,
                block_id=block_id,
                episode_block_idx=block_id,
                rollout_block_rank=block_rank,
                repeat_idx=repeat_idx,
                repeats_per_block=shape.repeats_per_block,
                episode_visit_idx=episode_visit_idx_for_block,
                optimizer_step_idx_in_episode=episode.optimizer_step_idx_in_episode + step_idx,
                is_block_enter=(repeat_idx == 0),
                is_block_exit=(repeat_idx == shape.repeats_per_block - 1),
                is_frame_exit=(repeat_idx == shape.repeats_per_block - 1),
                commit_observation_memory=(repeat_idx == 0),
                update_optimizer_memory=True,
                record_update_norm=True,
                commit_support_on_exit=(repeat_idx == shape.repeats_per_block - 1),
                commit_residual_on_exit=(repeat_idx == shape.repeats_per_block - 1),
                window_hash=window_hash,
                window_revisit_count=revisit_count,
                is_repeated_window=(revisit_count > 0),
                block_visit_count_before=block_visit_before,
                block_visit_count_after=block_visit_before + 1,
                ...
            ))

    episode_end = episode.rollout_idx_in_episode + 1 >= episode.rollouts_per_episode

    return IForwardV3RolloutPlan(...)
```

```python
def _commit_rollout_visits_to_episode(self, episode, plan):
    # Window counts.
    episode.window_counts[plan.window_hash] = episode.window_counts.get(plan.window_hash, 0) + 1

    # Visited target buffer.
    for ref in plan.evidence_refs_flat:
        if ref not in episode.visited_ref_set:
            episode.visited_refs.append(ref)
            episode.visited_ref_set.add(ref)

    # Block visit counts.
    for block_id in plan.window_block_ids:
        episode.block_visit_counts[block_id] = episode.block_visit_counts.get(block_id, 0) + 1

    # Clocks.
    episode.episode_visit_idx += plan.blocks_per_rollout
    episode.optimizer_step_idx_in_episode += plan.inner_K
```

---

## 17. 单元测试计划

### 17.1 Shape schedule

```text
test_v3_shapes_step0_single_block:
  global_step=0
  active shapes are r2b1/r4b1/r6b1/r8b1
  all blocks_per_rollout == 1
  inner_K <= 8
```

```text
test_v3_shapes_step20000_two_block:
  active shapes are r8b1/r4b2
  inner_K == 8
```

```text
test_v3_shapes_step40000_four_block:
  active shapes are r8b1/r4b2/r2b4
  inner_K == 8
```

### 17.2 Random window replacement

```text
test_v3_random_window_can_repeat:
  fixed seed, rollouts_per_episode > number of possible starts
  repeated window appears
  window_revisit_count increments
  is_repeated_window true after first repeat
```

### 17.3 Episode end by rollout budget

```text
test_v3_episode_end_by_rollouts_per_episode:
  random window mode
  block_cursor is unused
  episode_end_after_rollout true only when rollout_idx + 1 == rollouts_per_episode
```

### 17.4 Current target coverage

```text
test_v3_current_covers_all_rollout_input_frames:
  for b4_r2
  current refs == all cams of 4 input frames
```

### 17.5 History target from prior visited only

```text
test_v3_history_empty_first_rollout:
  rollout_idx=0
  history refs empty
```

```text
test_v3_history_uses_previous_rollout_frames_all_cams:
  after rollout 0
  rollout 1 history frames subset of rollout0 visited_frames
  each selected history frame expands to all cams
  history refs disjoint current refs
```

```text
test_v3_history_max_frames_and_max_refs_fallback:
  max_frames_per_rollout caps selected frames
  if max_frames_per_rollout is absent, max_refs_per_rollout // num_cams caps selected frames
```

### 17.6 Nearby target

```text
test_v3_nearby_from_current_window_block:
  nearby frame comes from keyframe_to_frames of one block in current window
  nearby not in current/history/evidence refs
```

### 17.7 Explicit block exit

```text
test_v3_step_events:
  for r4b2:
    repeat 0: is_block_enter true, is_block_exit false
    repeat 1/2: both false
    repeat 3: is_block_exit true
  two blocks produce two block_exit events
```

### 17.8 GRU visit-clock

```text
test_v3_gru_dt_uses_visit_clock:
  rollout 0 uses source_frame_idx 100
  rollout 1 uses source_frame_idx 50
  dt_visit > 0
  dt_source would be negative but is not used for decay
```

### 17.9 Resume

```text
test_v3_scheduler_state_dict_resume:
  after N rollouts
  save/load scheduler state
  next rollout plan matches exactly
  visited_frames/visited_refs/window_counts/block_visit_counts/RNG preserved
```

### 17.10 Scheduler flag obedience

```text
test_v3_model_obeys_commit_flags:
  record_update_norm=false does not update update_norm EMA
  commit_residual_on_exit=false does not render/commit residual
  commit_support_on_exit=false leaves pending block support uncommitted
```

---

## 18. Stateful consumption requirement

IForward v3 random-window scheduler is stateful within an episode:

```text
visited_frames
visited_refs
window_counts
block_visit_counts
episode_visit_idx
optimizer_step_idx_in_episode
episode_rng_state
```

Training must consume emitted rollout batches in order. Multi-worker or asynchronous dataloading must not reorder stateful rollouts or create independent scheduler instances with diverging episode state. Early v3 training should keep the scheduler stream serial, or move rollout emission into the main training loop where order is explicit.

---

## 19. 最终执行顺序

建议实现顺序：

```text
P0.1  新建 TrainSchedulerIForwardV3，支持 variable shapes + schedule + random replacement。
P0.2  增加 explicit StepPlan events：block_id / is_block_enter / is_block_exit / visit_clock。
P0.3  加入 scheduler visited_frames/visited_refs buffer，构建 current/history/nearby target roles。
P0.4  修改 resolver：current=all rollout input frames；history=final_history_replay role。
P0.5  修改 model：scheduler flags 是 commit 事件真源；is_block_exit 只做兼容 fallback。
P0.6  修改 TimeAwarePointGRU：dt 使用 episode_visit_idx，write 持久 decay 时推进 last_visit_idx。
P1.1  增加 YAML schedule 与 logging。
P1.2  增加 leakage check 与 tests。
P2    根据训练结果再调整 history sampling cap / loss weight / rollouts_per_episode。
```

---

## 20. 最重要的 invariant

v3 scheduler 必须始终满足：

```text
1. len(steps) == blocks_per_rollout * repeats_per_block <= 8
2. current targets == all cams of all rollout input frames
3. history targets come from episode visited frames before this rollout, expanded to all cams
4. history targets disjoint current targets
5. nearby targets disjoint evidence/current/history targets
6. block_exit is explicit and equals last repeat of each block
7. support/residual EMA commits only when scheduler commit flags are true
8. update_norm records only when scheduler record_update_norm is true
9. GRU decay clock is visit_clock, not source_frame_idx
10. persisted decayed h_prior advances last_visit_idx, avoiding double-decay
11. episode end is controlled by rollouts_per_episode in random-window mode
12. stateful rollout batches are consumed serially
```

这些 invariant 如果任何一条被破坏，v3 的 history gate 实验都会变得不可解释。

---

## 21. 一句话总结

IForward v3 的 scheduler 应该是：

```text
5_4-style block-exit history
+
random-window-with-replacement rollout supervision
+
current=all rollout frames
+
history=previous visited frames
+
nearby=current-window unsupervised frame
+
visit-clock TimeAwarePointGRU
```

随机窗口可以保留，甚至可以重复；但必须承认它不是 chronological sequence training，而是同一 local segment 上的随机约束优化。只要 scheduler 显式记录 block visit / block exit / visited history / visit clock，history gate 的语义就仍然可以保持 5_4 风格。
