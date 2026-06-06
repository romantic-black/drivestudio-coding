# IForward v3 实现方案文档：TimeAwarePointGRU + EMA History Gate

版本：v3_gru_history_gate  
目标：在当前 IForward / Stage6 事件骨架上，恢复类似 StreetForward 5_4 的 block-level history gate 稳定性，并用 TimeAwarePointGRU 替换当前 tiny point-Mamba 作为轻量 optimizer memory。  
范围：只实现 **TimeAwarePointGRU + EMA history gate**。不引入新 ledger，不引入 memory-xCPE，不引入 Mamba，不把 random-window / revisit 放进 v3 主线。

---

## 0. 结论

IForward v3 的主线应当是：

```text
Stage6 V4 observe / 2D lifting / event xCPE
    ↓
Stage6 EventPack
    ↓
TimeAwarePointGRU.read(...)        # 读 optimizer memory，不写 history EMA
    ↓
Stage6PosteriorUpdater             # 预测 delta_hat
    ↓
EMA HistoryGate                    # 写回前 gate
    ↓
gated delta apply
    ↓
update_norm EMA                    # 每个 repeat 后更新
    ↓
TimeAwarePointGRU.write_after_update(...)
    ↓
block_exit: support EMA + residual EMA detach 更新
```

关键原则：

1. **IForward v3 不把 repeat 当成 history 事件。** repeat 只是当前 block 内的 inner iteration。
2. **support / residual 仍然按 5_4 语义在 block_exit 更新。** 当前帧内部多次迭代时不刷新这些 history。
3. **只有 update_norm 是 repeat 内更新。** 因为它描述真实写回幅度，会直接影响当前 block 后续 repeat 的保护强度。
4. **history gate 必须放在最终写回前。** 它不是 posterior 内部 noop gate，也不是 GRU 的隐式能力。
5. **GRU 是 optimizer memory，不是 history memory。** EMA history 仍然是 gate 的主要稳定来源。
6. **v3 不考虑 Mamba / memory-xCPE / new ledger / random-window 主线。** 先恢复可解释、可验证的稳定机制。
7. **只要 LocalGS / GRU / EMA state 跨 rollout carry，scheduler 必须是 chronological next_contiguous。** `random_start_contiguous + carried state` 会破坏 block-clocked history 语义。

---

## 1. 当前代码基线与改动点

当前 IForward 主流程在：

```text
models/iforward/model.py
```

核心循环是：

```python
measurement = bridge.observe(...)
event = bridge.build_event(...)
ctx_memory = memory(...)
local_state, delta, update_aux = bridge.apply_update(...)
```

其中：

```text
bridge.apply_update(...)
    → runtime._apply_event_update(...)
        → stage6_posterior_updater(...)
        → expand rigid delta
        → apply branch scope
        → local_state.apply_delta(delta)
```

v3 必须拆开这个黑盒更新，否则 history gate 没有正确插入点。

v3 最小必要改动：

```text
models/iforward/
  ├── gru_memory.py          # 新增 TimeAwarePointGRU
  ├── history_ema.py         # 新增 EMA history state 与 block-exit update
  ├── history_gate.py        # 新增 attribute-wise history gate
  ├── delta_ops.py           # 新增 BranchDelta / DeltaPack gate 与 norm 工具
  ├── state.py               # 扩展 IForwardState，挂载 history_ema
  ├── bridge.py              # 拆 predict_delta / finalize_delta / apply_delta_only / record_block_residual
  └── model.py               # 新增 version=v3_gru_history_gate 分支
```

不做：

```text
models/iforward/point_mamba_memory.py        # v3 不使用
models/iforward/local_conflict_xcpe.py       # v3 不使用
new long-sequence history ledger             # v3 不做
new Mamba block                              # v3 不做
```

---

## 2. v3 forward 结构

### 2.1 总体 forward

```python
for step in resolved.steps:
    measurement = bridge.observe(
        local_state=local_state,
        batch=batch,
        source_indices=step.source_indices,
        source_frame_idx=step.source_frame_idx,
    )

    event = bridge.build_event(
        local_state=local_state,
        measurement=measurement,
    )

    is_block_enter = bool(step.is_block_enter)
    is_block_exit = bool(step.is_block_exit)

    if is_block_enter:
        history_ema.record_block_support_snapshot(event=event)

    ctx_gru, gru_prepared, gru_aux = point_gru.read(
        event=event,
        local_state=local_state,
        state=gru_state,
        step_context=step_context,
    )

    hist_view = history_ema.view_for_event(event=event, local_state=local_state)

    delta_raw, posterior_aux = bridge.predict_delta(
        local_state=local_state,
        event=event,
        ctx_memory=ctx_gru,
    )

    delta_raw = bridge.apply_branch_scope_event_rows(delta_raw)

    gate, gate_aux = history_gate(
        event=event,
        local_state=local_state,
        delta_raw=delta_raw,
        ctx_gru=ctx_gru,
        history_view=hist_view,
    )

    delta_gated_raw = apply_history_gate_to_delta_raw(
        delta_raw,
        gate,
        event=event,
    )

    delta_full = bridge.expand_rigid_delta(
        delta_gated_raw,
        event=event,
        local_state=local_state,
    )

    local_state = bridge.apply_delta_only(
        local_state=local_state,
        delta=delta_full,
    )

    history_ema.record_update_norm_after_apply(delta_full)

    gru_state, gru_write_aux = point_gru.write_after_update(
        prepared=gru_prepared,
        event=event,
        delta_raw=delta_gated_raw,
        gate=gate,
        step_context=step_context,
    )

    if is_block_exit:
        residual_pack = bridge.compute_block_residual_history(
            local_state=local_state,
            batch=batch,
            source_indices=step.source_indices,
            source_frame_idx=step.source_frame_idx,
        )
        history_ema.commit_block_exit_residual(residual_pack)
        history_ema.commit_block_exit_support()
        history_ema.clear_block_pending()
```

### 2.2 关键顺序说明

#### 为什么 support 在 repeat0 snapshot，但 block_exit commit？

因为 support 是当前观测证据，不应该被当前帧的 K 次 repeat 放大。v3 使用 5_4 风格：

```text
repeat0 observe 后：记录本 block 的 support snapshot
block_exit：detach 后把 snapshot 写入 support EMA
```

在当前 block 的 repeat1、repeat2、... 中，gate 可以看到：

```text
旧 history EMA
当前 event.support / event.valid / obs_code
每步更新后的 update_norm EMA
```

但不会看到已经被当前帧 repeat 多次污染后的 support EMA。

#### 为什么 residual 只在 block_exit？

residual 表示当前 block 完成优化后的重建误差。它不应该记录中间迭代状态。

```text
block_exit post-update render source refs
    → RGB residual
    → backproject to GS points
    → update error_fast / error_slow EMA
```

#### 为什么 update_norm 每 repeat 更新？

update_norm 表示实际写回风险。如果当前 block 内前几个 repeat 已经写了很大 delta，后续 repeat 的 gate 应该立即知道这一点并收缩。

必须使用：

```text
gated delta, after history gate, after branch scope
```

不能使用：

```text
delta_hat before gate
```

---

## 3. Scheduler 语义

v3 不重建大 scheduler，但需要把现有 IForward scheduler 的事件语义显式化。

### 3.1 层次

```text
segment
  episode
    rollout
      block       # 一个 source frame / keyframe visit
        repeat    # 当前 block 内的 inner iteration
```

### 3.2 Scheduler hard rule

v3 mainline 是 block-clocked、chronological、stateful scheduler：

```text
episode:
  block0 -> block1 -> block2 -> ...
```

当 `memory.carry_policy=across_rollouts_until_episode_end` 时必须满足：

```text
rollout.block_selection_policy = next_contiguous
rollout.delivery_order_policy  = chronological
```

`random_start_contiguous` 只允许在 state reset per window 或单 rollout episode 下作为未来 ablation 使用。v3 主线禁止：

```text
random_start_contiguous + carried LocalGS / GRU / EMA state
```

### 3.3 StepPlan 必要字段

当前 `IForwardResolvedStep` 已经有：

```python
step_idx
source_frame_idx
repeat_idx
rollout_block_rank
source_indices
evidence_refs
commit_observation_memory
update_optimizer_memory
rollout_pos_code
frame_pos_code
repeat_pos_code
```

v3 必须由 scheduler 输出并由 resolver 保留：

```python
block_id: int
episode_block_idx: int
rollout_block_rank: int
repeat_idx: int
repeats_per_block: int
is_block_enter: bool       # repeat_idx == 0
is_block_exit: bool        # repeat_idx == repeats_per_block - 1
```

构建规则：

```python
is_block_enter = int(step.repeat_idx) == 0
is_block_exit = int(step.repeat_idx) == int(step.repeats_per_block) - 1
```

`source_frame_idx` 变化只能作为 legacy batch compatibility fallback，不能作为 v3 block boundary 主判据。fallback 优先级：

```text
explicit is_block_exit
  -> block_id / episode_block_idx changes
  -> source_frame_idx changes
```

这样同一个 `source_frame_idx` 的不同 block visit 不会被误合并。

### 3.4 v3 调度规则

```text
commit_observation_memory = True  only repeat_idx == 0
update_optimizer_memory   = True  every repeat
record_support_snapshot   = True  only repeat_idx == 0
record_update_norm        = True  every repeat after apply
record_residual           = True  only block_exit after apply
commit_support_ema        = True  only block_exit
commit_residual_ema       = True  only block_exit
```

### 3.5 与 5_4 的关系

v3 的 history clock 更接近 5_4：

```text
5_4 block_exit / visit history record
    ↓
v3 block_exit support/residual EMA commit
```

v3 与 5_4 的唯一重要差别：

```text
update_norm EMA 在当前 block 内每个 repeat 后更新
```

这是必要的，因为 IForward 当前 block 内有多次写回。

---

## 4. State 设计

### 4.1 IForwardState 扩展

当前：

```python
@dataclass
class IForwardState:
    local_gs: LocalGSState
    memory: Any
    history: IForwardShortWindowHistory
    scene_id: int
    segment_id: int
    episode_id: int
```

v3 增加：

```python
history_ema: Optional[IForwardHistoryEMAState] = None
```

建议：

```python
@dataclass
class IForwardState:
    local_gs: LocalGSState
    memory: Any
    history: IForwardShortWindowHistory
    scene_id: int
    segment_id: int
    episode_id: int
    history_ema: Optional[IForwardHistoryEMAState] = None
    node_state_bg: Optional[NodeStateBackground] = None
    node_state_distant: Optional[NodeStateDistant] = None
    node_state_rigid: Optional[NodeStateRigid] = None
```

`detach_for_next_rollout()`：

```python
return IForwardState(
    local_gs=detach_local_gs_state(self.local_gs),
    memory=self.memory.detach(),
    history=self.history.detach(),
    history_ema=None if self.history_ema is None else self.history_ema.detach(),
    ...
)
```

### 4.2 TimeAwarePointGRU state

新增文件：

```text
models/iforward/gru_memory.py
```

核心 dataclass：

```python
@dataclass
class IForwardGRUBranchState:
    h: torch.Tensor               # [N, H]
    seen: torch.Tensor            # [N] bool
    last_frame_idx: torch.Tensor  # [N] int64

    def detach(self): ...

@dataclass
class IForwardGRUMemoryState:
    bg: IForwardGRUBranchState
    distant: IForwardGRUBranchState
    rigid: IForwardGRUBranchState

    @classmethod
    def empty(cls): ...
    def detach(self): ...
    def count_tokens(self): ...
```

初始化时按 LocalGSState 行数创建：

```python
IForwardGRUBranchState(
    h=torch.zeros(N, hidden_dim),
    seen=torch.zeros(N, dtype=torch.bool),
    last_frame_idx=torch.full((N,), -1, dtype=torch.long),
)
```

### 4.3 History EMA state

新增文件：

```text
models/iforward/history_ema.py
```

```python
@dataclass
class IForwardHistoryBranchEMA:
    support_fast: torch.Tensor       # [N,1]
    error_fast: torch.Tensor         # [N,1]
    update_norm_fast: torch.Tensor   # [N,1]
    support_slow: torch.Tensor       # [N,1]
    error_slow: torch.Tensor         # [N,1]
    update_norm_slow: torch.Tensor   # [N,1]
    initialized: torch.Tensor        # [N,1]

    # pending block support, not committed until block_exit
    block_support_sum: torch.Tensor    # [N,1]
    block_present_count: torch.Tensor  # [N,1]
    block_visible_count: torch.Tensor  # [N,1]

@dataclass
class IForwardHistoryEMAState:
    bg: IForwardHistoryBranchEMA
    distant: Optional[IForwardHistoryBranchEMA]
    rigid: Optional[IForwardHistoryBranchEMA]
```

`initialized` 的语义：

```text
该 point 至少在某个 committed block 中 visible，或 residual 被有效记录。
```

---

## 5. TimeAwarePointGRU 设计

### 5.1 定位

TimeAwarePointGRU 是 optimizer memory，不是 history gate 的替代品。

它只提供：

```text
当前点过去 optimizer context 的 compact hidden state
```

它不负责：

```text
判断是否允许写回
保护历史帧
记录 residual / support history
```

### 5.2 read/write 分离

v3 推荐 read/write 分离，避免 memory 在同一步中先写当前 event 再参与当前 delta 预测。

```python
ctx, prepared, aux = point_gru.read(event, state, step_context)
...
state_next, aux = point_gru.write_after_update(prepared, event, delta_raw, gate, step_context)
```

#### read

```python
h_old = state.h[rows]
dt = current_frame_idx - state.last_frame_idx
seen = state.seen

# unseen rows use zero hidden and dt=0
h_old = torch.where(seen[:, None], h_old, zeros)
dt = torch.where(seen, dt.clamp_min(0), zeros)

# time-aware decay
rate = softplus(decay_log_rate)          # [H] or scalar
gamma = exp(-rate * clamp(dt, 0, dt_clip))
h_prior = h_old * gamma

ctx = read_proj(layer_norm(h_prior))     # [N, ctx_dim]
```

同一个 source frame 内多次 repeat：

```text
dt = 0
```

跨 frame：

```text
dt > 0
```

这比普通 GRU 更适合 source frame 不规则间隔。

#### write_after_update

write token 使用当前 event + 实际写回信息：

```python
write_token = cat([
    event_x,
    obs_code,
    log1p(support_now),
    valid_now,
    repeat_pos_code,
    frame_pos_code,
    rollout_pos_code,
    gate_means,
    gate_scales,
    gate_quat,
    gate_opacity,
    gate_sh,
    delta_norm_means,
    delta_norm_attr,
])

h_candidate = GRUCell(write_token, h_prior)
h_new = where(write_mask[:, None], h_candidate, h_prior)
```

P0 `delta_norm_attr` 可以只保留 means norm，保持 5_4 风格：

```python
delta_norm_means = norm(delta.means, dim=-1, keepdim=True)
```

### 5.3 write mask

```python
write_mask = step.update_optimizer_memory
write_mask &= valid_now
write_mask &= support_now >= hard_support_min_optimizer
```

配置：

```yaml
model:
  iforward:
    point_gru:
      hidden_dim: 48
      ctx_dim: 48
      dt_clip: 32
      hard_valid_required: true
      hard_support_min_optimizer: 0.0
      write_policy: every_repeat
```

注意：GRU write 是 optimizer memory write，不是 EMA history write，因此允许 every repeat。

### 5.4 branch 对齐

```text
bg       : full bg rows
Distant  : full distant rows
Rigid    : event.rigid rows align with route.S, state.rigid 是 full rigid rows
```

Rigid read/write：

```python
rows = event.route.S
h_prior = state.rigid.h[rows]
...
state.rigid.h[rows] = h_new
```

不要给未出现在 `route.S` 的 rigid rows 写 GRU。

---

## 6. EMA History 设计

### 6.1 History raw fields

v3 使用原始 EMA 结构，不上 ledger。

每个 point 存：

```text
support_fast
error_fast
update_norm_fast
support_slow
error_slow
update_norm_slow
initialized
```

gate 输入时再拼当前观测：

```text
visible_now
log1p(support_now)
valid_now
obs_code_now[0:2]
```

v3 `history_raw` 推荐 12 维：

```python
history_raw = cat([
    support_fast,         # 1
    error_fast,           # 1
    update_norm_fast,     # 1
    support_slow,         # 1
    error_slow,           # 1
    update_norm_slow,     # 1
    initialized,          # 1
    visible_now,          # 1
    log1p(support_now),   # 1
    valid_now,            # 1
    obs_code_now[:, 0:2], # 2
], dim=-1)                # total 12
```

这与 5_4 的 12 维 history embed 兼容精神一致，但不使用 view_transient。

### 6.2 support EMA

记录时刻：

```text
repeat_idx == 0：snapshot 当前 support 到 block pending
block_exit：commit pending support 到 EMA
```

P0 不在 repeat1..K 中更新 support。

```python
def record_block_support_snapshot(event):
    support = log1p(event.support.detach()).reshape(N, 1)
    present = active_rows
    visible = valid & (support > support_min)
    block_support_sum += support
    block_present_count += present
    block_visible_count += visible
```

commit：

```python
has_present = block_present_count > 0
support_cur = block_support_sum / block_present_count.clamp_min(1)
visible = has_present & (block_visible_count > 0) & (support_cur > support_min)
invisible = has_present & ~visible

support_fast = where(
    visible,
    beta_fast_visible * support_fast + (1 - beta_fast_visible) * support_cur,
    where(invisible, beta_fast_invisible * support_fast, support_fast),
)

support_slow = where(
    visible,
    beta_slow_visible * support_slow + (1 - beta_slow_visible) * support_cur,
    where(invisible, beta_slow_invisible * support_slow, support_slow),
)

initialized = max(initialized, visible)
```

也就是说：present + visible rows 写入当前 support；present 但 invisible rows 按 invisible beta 衰减旧 support；not-present rows 不变。否则 invisible beta 永远不会真正生效。

Rigid support snapshot 必须 scatter 到 full rigid rows：

```python
support_full = zeros([num_rigid, 1])
present_full = zeros([num_rigid, 1])
visible_full = zeros([num_rigid, 1])
support_full.index_add_(0, event.route.S, log1p(event.support_rigid))
present_full.index_add_(0, event.route.S, ones_like(event.support_rigid))
visible_full.index_add_(0, event.route.S, valid_rigid.float())
```

### 6.3 residual EMA

记录时刻：

```text
block_exit after gated apply
```

流程：

```text
local_state post-update
    ↓
render current block source refs
    ↓
abs(pred_rgb - gt_rgb).mean(channel)
    ↓
backproject residual to gaussians
    ↓
error EMA update where visible
```

EMA：

```python
error_fast = where(
    visible,
    beta_error_fast * error_fast + (1 - beta_error_fast) * error_cur,
    error_fast,
)

error_slow = where(
    visible,
    beta_error_slow * error_slow + (1 - beta_error_slow) * error_cur,
    error_slow,
)

initialized = max(initialized, visible)
```

P0 residual 只处理当前 block 的 source refs，不采样 old anchors。

### 6.4 update_norm EMA

记录时刻：

```text
every repeat after apply_delta
```

必须使用 gated delta：

```python
update_norm_cur = norm(delta_full.means.detach(), dim=-1, keepdim=True)
```

按 5_4 风格，P0 只用 means norm：

```python
written = update_norm_cur > 0
update_norm_fast = where(written, beta_fast * old + (1 - beta_fast) * update_norm_cur, old)
update_norm_slow = where(written, beta_slow * old + (1 - beta_slow) * update_norm_cur, old)
```

为什么不先统计 SH / opacity：

```text
v3 目标是最小恢复 5_4 稳定机制。appearance update_norm 可以作为 v3.1 ablation，不进入 v3 P0。
```

---

## 7. HistoryGate 设计

新增文件：

```text
models/iforward/history_gate.py
```

### 7.1 输出

```python
@dataclass
class IForwardAttributeGate:
    means: torch.Tensor       # [N,1]
    scales: torch.Tensor      # [N,1]
    quat: torch.Tensor        # [N,1]
    opacity: torch.Tensor     # [N,1]
    sh: torch.Tensor          # [N,1]
    hidden: torch.Tensor      # [N,1]
```

### 7.2 输入

每个 branch 的 gate 输入：

```python
gate_input = cat([
    event_x,              # Stage6 event, includes current obs / support / params indirectly
    ctx_gru,              # TimeAwarePointGRU prior context
    history_embed,        # EMA history + current visible / support / obs_code
    branch_embed,
], dim=-1)
```

P0 不再额外引入 normalized params，因为 Stage6 event 已经包含 ParamObsCodec / param embedding 语义。

### 7.3 Gate 公式

```python
gate_logits = gate_mlp(gate_input) + branch_bias
raw = sigmoid(gate_logits)
gate = min_gate + (1 - min_gate) * raw
```

然后执行三个约束：

#### 1. cold-open

```python
if cold_open_uninitialized:
    gate = where(initialized > 0, gate, ones_like(gate))
```

原因：单帧阶段或新点未初始化时，不应该被 history gate 锁死。

#### 2. hard mask binding

```python
mask_update = valid_now & (support_now >= support_min)
gate = gate * mask_update.float()
```

这一步必须在 cold-open 之后执行。

注意：

```text
min_gate 不允许绕过 mask_update。
mask_update == False 时，所有 delta 必须严格为 0。
```

#### 3. hidden gate

沿用 5_4 的 weighted-sum 风格：

```python
g_hidden = (
    w_means   * g_means
  + w_scales  * g_scales
  + w_quat    * g_quat
  + w_opacity * g_opacity
  + w_sh      * g_sh
)
```

### 7.4 branch id

建议：

```text
0 = bg
1 = distant
2 = rigid
```

不再区分 `rigid_in / rigid_out` 的 gate branch id。原因：Stage6 event 已经经过 routed near/far decoder，rigid event rows 已经包含 inside/outside 差异；v3 P0 不需要把 5_4 的 4-branch gate 完整搬回。

如果后续发现 rigid_in/out gate 差异很重要，v3.1 再加 `rigid_route_code`。

### 7.5 初始化建议

```yaml
history_gate:
  enable: true
  hidden_dim: 64
  history_embed_dim: 16
  cold_open_uninitialized: true
  bind_with_mask_update: true
  min_gate:
    means: 0.02
    scales: 0.02
    quat: 0.005
    opacity: 0.03
    sh: 0.03
  init_bias:
    means: -1.20
    scales: -1.40
    quat: -1.80
    opacity: -0.40
    sh: 0.00
  hidden_gate:
    weights:
      means: 0.20
      scales: 0.00
      quat: 0.00
      opacity: 0.30
      sh: 0.50
  branch_bias:
    bg:      {means:  0.0, scales:  0.0, quat: -0.2, opacity: 0.0, sh: 0.1}
    distant: {means: -1.0, scales: -0.3, quat: -1.0, opacity: 0.0, sh: 0.0}
    rigid:   {means: -0.2, scales: -0.2, quat: -0.3, opacity: 0.1, sh: 0.2}
```

---

## 8. Delta 操作

新增文件：

```text
models/iforward/delta_ops.py
```

### 8.1 gate raw delta

```python
def gate_branch_delta(delta: BranchDelta, gate: IForwardAttributeGate) -> BranchDelta:
    return BranchDelta(
        means=delta.means * gate.means,
        scales_log=delta.scales_log * gate.scales,
        quat_axis_angle=delta.quat_axis_angle * gate.quat,
        opacity_logit=delta.opacity_logit * gate.opacity,
        sh=delta.sh * gate.sh,
        hidden=delta.hidden * gate.hidden,
        confidence=delta.confidence,
        noop=delta.noop,
    )
```

### 8.2 rigid gate 位置

Rigid 的 `delta_raw.rigid` 与 `event.event_rigid` 对齐，即 rows = `route.S`。

所以 gate 应在 expand 前：

```text
delta_raw.rigid [len(route.S)]
    ↓ gate with history.rigid[route.S]
    ↓ expand to full rigid rows
```

不要先 expand 再用 event-rigid gate，否则需要构造 full-row event / ctx / support，容易出错。

### 8.3 apply order

推荐：

```text
posterior delta_raw
    ↓
branch scope on event rows
    ↓
history gate on event rows
    ↓
rigid expand to full rows
    ↓
local_state.apply_delta
    ↓
constrain local state
```

---

## 9. Bridge API 改造

当前 `bridge.apply_update()` 太粗，需要拆成 v3 使用的四个方法。

### 9.1 predict_delta

```python
def predict_delta(
    self,
    *,
    local_state: LocalGSState,
    event: EventPack,
    ctx_memory: Optional[ContextPack],
) -> Tuple[DeltaPack, Dict[str, Any]]:
    delta, aux = self.runtime.stage6_posterior_updater(
        event=event,
        ctx_current=None,
        ctx_vsm=ctx_memory,
    )
    return delta, {**event.aux, **aux}
```

### 9.2 apply_branch_scope_event_rows

```python
def apply_branch_scope_event_rows(self, delta: DeltaPack) -> DeltaPack:
    # runtime._apply_branch_scope works row-wise and does not require rigid to be full rows.
    return self.runtime._apply_branch_scope(delta)
```

### 9.3 expand_rigid_delta

```python
def expand_rigid_delta(self, *, delta: DeltaPack, event: EventPack, local_state: LocalGSState) -> DeltaPack:
    if delta.rigid is None or local_state.rigid is None:
        return delta
    route = event.route
    return DeltaPack(
        bg=delta.bg,
        distant=delta.distant,
        rigid=self.runtime._expand_branch_delta(
            delta.rigid,
            indices=route.S,
            total=int(local_state.rigid.means.shape[0]),
        ),
        aux=delta.aux,
    )
```

### 9.4 apply_delta_only

```python
def apply_delta_only(self, *, local_state: LocalGSState, delta: DeltaPack) -> LocalGSState:
    next_state = local_state.apply_delta(delta)
    return self.runtime._constrain_local_state_after_delta(next_state)
```

### 9.5 compute_block_residual_history

新增：

```python
def compute_block_residual_history(
    self,
    *,
    local_state: LocalGSState,
    batch: Dict[str, Any],
    source_indices: List[int],
    source_frame_idx: int,
) -> IForwardResidualPack:
    ...
```

实现策略：

1. 从 `batch["source_views"]`、`batch["source_images"]`、mask 中取 `source_indices` 子集。
2. 使用 Stage6 runtime 的 `_render_params_for_frame(local_state, frame_idx)` 得到当前帧 render params。
3. 对每个 source view render RGB。
4. 计算 masked residual：

```python
residual = abs(pred_rgb - gt_rgb).mean(dim=-1, keepdim=True)
```

5. 使用 runtime 已有 backproject 工具把 residual backproject 到当前 render params 的 gaussian rows。
6. 拆回 bg / distant / rigid full rows。

P0 可以直接借鉴 Stage5_3 的 `_compute_record_support_error_all_branches_once_routed()` 逻辑，但输入从 NodeState 改成 LocalGSState。

返回：

```python
@dataclass
class IForwardResidualPack:
    error_bg: torch.Tensor          # [N_bg,1]
    support_bg: torch.Tensor        # [N_bg,1]
    error_distant: Optional[Tensor]
    support_distant: Optional[Tensor]
    error_rigid: Optional[Tensor]   # full rigid rows [N_rigid,1]
    support_rigid: Optional[Tensor]
```

---

## 10. Model 初始化与版本分支

### 10.1 version

新增：

```yaml
model:
  iforward:
    version: v3_gru_history_gate
```

`IForwardModel.__init__`：

```python
self.is_v3_gru_history_gate = self.iforward_version == "v3_gru_history_gate"
```

### 10.2 module init

```python
if self.is_v3_gru_history_gate:
    self.point_gru = IForwardTimeAwarePointGRU(
        event_dim=event_dim,
        hidden_dim=cfg.point_gru.hidden_dim,
        ctx_dim=cfg.point_gru.ctx_dim,
        dt_clip=cfg.point_gru.dt_clip,
        hard_support_min_optimizer=cfg.point_gru.hard_support_min_optimizer,
    )
    self.history_gate = IForwardHistoryGate(
        event_dim=event_dim,
        ctx_dim=cfg.point_gru.ctx_dim,
        history_embed_dim=cfg.history_gate.history_embed_dim,
        hidden_dim=cfg.history_gate.hidden_dim,
        ...
    )
    self.memory = None
```

不要实例化：

```python
IForwardPointMambaMemory
IForwardLocalConflictXcpe
```

### 10.3 state init

```python
if self.is_v3_gru_history_gate:
    memory_state = IForwardGRUMemoryState.from_local_state(
        local_state,
        hidden_dim=self.point_gru.hidden_dim,
    )
    history_ema = IForwardHistoryEMAState.from_local_state(local_state)
else:
    ...
```

`IForwardState(...)` 加上：

```python
history_ema=history_ema
```

### 10.4 forward_rollout 分支

当前：

```python
if self.is_v6_point_mamba_xcpe:
    memory_state, ctx_memory, memory_aux = self._build_v6_context(...)
else:
    memory_state, ctx_memory, memory_aux, short_entries = self.memory(...)

local_state, delta, update_aux = self.bridge.apply_update(...)
```

v3：

```python
if self.is_v3_gru_history_gate:
    local_state, memory_state, history_ema, delta, aux = self._v3_step(...)
elif self.is_v6_point_mamba_xcpe:
    ...
else:
    ...
```

---

## 11. v3 配置建议

新增配置文件：

```text
configs/iforward/iforward_v3_gru_history_gate.yaml
```

核心配置：

```yaml
output_name: iforward_v3_gru_history_gate

scheduler_iforward:
  enable: true
  episode:
    blocks_per_episode: 1
    episode_stride: 1
    min_blocks_per_episode: 1
    block_source_frame_policy: random_within_keyframe_once_per_episode
    reset_scene_state_policy: episode_begin
  rollout:
    block_selection_policy: next_contiguous
    delivery_order_policy: chronological
    min_blocks_per_rollout: 1
    allow_short_final_rollout: false
    detach_graph_after_rollout: true
    shapes:
      - {name: b1_r4, blocks_per_rollout: 1, repeats_per_block: 4, prob: 1.0}
  memory:
    reset_policy: episode_begin
    carry_policy: across_rollouts_until_episode_end

model:
  stage: "6_0"
  phase: "phase_A_block_local_unroll"
  iforward:
    version: v3_gru_history_gate

    point_gru:
      enable: true
      hidden_dim: 48
      ctx_dim: 48
      dt_clip: 32
      hard_valid_required: true
      hard_support_min_optimizer: 0.0
      write_policy: every_repeat
      read_write_split: true
      branch_weights: separate

    history_memory:
      enable: true
      record_on: block_exit
      record_views: source_image_refs
      support:
        fast_ema_beta_visible: 0.35
        fast_ema_beta_invisible: 0.60
        slow_ema_beta_visible: 0.90
        slow_ema_beta_invisible: 0.95
      residual:
        fast_error_beta: 0.35
        slow_error_beta: 0.90
        error_eps: 1.0e-6
      update:
        fast_ema_beta: 0.45
        slow_ema_beta: 0.92
        apply_in_eval: true
        norm_source: means_only

    history_gate:
      enable: true
      type: attribute_5
      hidden_dim: 64
      history_embed_dim: 16
      cold_open_uninitialized: true
      bind_with_mask_update: true
      min_gate:
        means: 0.02
        scales: 0.02
        quat: 0.005
        opacity: 0.03
        sh: 0.03
      init_bias:
        means: -1.20
        scales: -1.40
        quat: -1.80
        opacity: -0.40
        sh: 0.00
      branch_bias:
        bg:      {means:  0.0, scales:  0.0, quat: -0.2, opacity: 0.0, sh: 0.1}
        distant: {means: -1.0, scales: -0.3, quat: -1.0, opacity: 0.0, sh: 0.0}
        rigid:   {means: -0.2, scales: -0.2, quat: -0.3, opacity: 0.1, sh: 0.2}
      hidden_gate:
        mode: weighted_sum
        weights:
          means: 0.20
          scales: 0.00
          quat: 0.00
          opacity: 0.30
          sh: 0.50

    short_window_history:
      max_entries: 24
      max_memory_entries: 0

    loss:
      current: {weight: 1.0}
      nearby: {weight: 0.20}
      in_rollout_history: {weight: 1.0}
      short_window_history: {weight: 1.0}
      delta_regularization: {weight: 1.0}

    trainability:
      train_point_gru: true
      train_history_gate: true
      train_measurement_frontend: true
      train_stage6_struct_decoder: true
      train_stage6_posterior_updater_base: true
      train_vsm_ctx_adapter: true
      train_point_mamba: false
      train_local_conflict_xcpe: false
```

Stage6 posterior updater 的 vsm ctx dim 必须与 GRU ctx dim 对齐：

```yaml
model:
  stage6_0:
    posterior_updater:
      phase_b_hooks:
        accept_vsm_ctx: true
        vsm_ctx_dim: 48
```

注意：不要启用 Stage6 runtime 内部的 `model.history_memory` / `model.update_gate`。v3 的 history gate 在 IForward 外层实现。

---

## 12. 训练策略

v3 使用同一 forward graph，不再区分单帧 A 结构和短序列 B 结构。

### 12.1 Phase 1：单帧能力

```yaml
scheduler_iforward:
  episode:
    blocks_per_episode: 1
    episode_stride: 1
  rollout:
    block_selection_policy: next_contiguous
    delivery_order_policy: chronological
    min_blocks_per_rollout: 1
    shapes:
      - {name: b1_r4, blocks_per_rollout: 1, repeats_per_block: 4, prob: 1.0}
  memory:
    carry_policy: across_rollouts_until_episode_end
```

loss：

```text
current: 1.0
nearby: 0.0-0.1
in_rollout_history: 0
short_window_history: 0
```

此时：

```text
initialized = 0
cold_open_uninitialized = true
```

所以 history gate 不会锁死单帧优化。

block_exit 仍然记录：

```text
support EMA
residual EMA
update_norm EMA
```

这样进入短序列阶段时 history 管线已经被训练过。

### 12.2 Phase 2：短序列稳定性

```yaml
blocks_per_rollout: 2 -> 4
repeats_per_block: 2 -> 4
```

loss：

```text
current: 1.0
in_rollout_history: 0.5 -> 1.0
short_window_history: 0.25 -> 1.0
nearby: 0.1 -> 0.2
```

重点监控：

```text
history frame PSNR 是否随新 block 下降
gate means 是否在不可见/弱 support 点明显收缩
update_norm_fast 是否在同一 block 内抑制连续大写回
当前帧是否因为 gate 过强而无法优化
```

### 12.3 Phase 3：稍长 chronological episode

v3 mainline 继续保持 chronological block serial，只增加 episode / rollout 长度：

```yaml
scheduler_iforward:
  episode:
    blocks_per_episode: 8
    episode_stride: 8
  rollout:
    block_selection_policy: next_contiguous
    delivery_order_policy: chronological
    min_blocks_per_rollout: 4
    shapes:
      - {name: b4_r2, blocks_per_rollout: 4, repeats_per_block: 2, prob: 0.7}
      - {name: b4_r1, blocks_per_rollout: 4, repeats_per_block: 1, prob: 0.3}
```

`random-window / revisit` 是 future ablation，不进入 v3 carried-state training path。未来如果启用，必须先引入显式 revisit semantics，并明确 LocalGS / GRU / EMA state 是否 reset、probe-only，或以特殊 revisit mode 更新。

---

## 13. Optimizer 参数组

新增参数组：

```text
iforward_point_gru
iforward_history_gate
```

建议 lr：

```yaml
optimizer:
  lr:
    iforward_point_gru: 1.0e-4
    iforward_history_gate: 1.0e-4
    stage6_posterior_updater_base: 1.0e-5 ~ 5.0e-5
    stage6_posterior_updater_vsm_ctx_adapter: 1.0e-4
    stage6_struct_decoder: 5.0e-5 ~ 1.0e-4
    measurement_frontend: 1.0e-5 ~ 5.0e-5
```

不应出现：

```text
point_mamba
local_conflict_xcpe
cell_mamba
global_mamba
```

---

## 14. Logging 与诊断

### 14.1 per-step stats

```text
v3/gru/bg_ctx_norm
v3/gru/distant_ctx_norm
v3/gru/rigid_ctx_norm
v3/gru/bg_write_ratio
v3/gru/rigid_write_ratio
v3/gru/bg_dt_mean
v3/gru/rigid_dt_mean

v3/gate/bg_means_mean
v3/gate/bg_sh_mean
v3/gate/rigid_means_mean
v3/gate/distant_means_mean
v3/gate/bg_mask_update_ratio
v3/gate/rigid_mask_update_ratio

v3/history/bg_update_fast_mean
v3/history/bg_update_slow_mean
v3/history/rigid_update_fast_mean
```

### 14.2 block-exit stats

```text
v3/history/bg_support_fast_mean
v3/history/bg_support_slow_mean
v3/history/bg_error_fast_mean
v3/history/bg_error_slow_mean
v3/history/bg_initialized_ratio

v3/history/distant_support_fast_mean
v3/history/rigid_support_fast_mean
v3/history/rigid_initialized_ratio
```

### 14.3 稳定性指标

沿用现有 final render stats，并重点看：

```text
psnr_gap/current_minus_rollout_history
psnr_gap/current_minus_short_history
history_rollout_psnr
short_window_history_psnr
```

新增：

```text
v3/stability/update_to_gate_corr
v3/stability/gate_low_support_mean
v3/stability/gate_visible_mean
```

---

## 15. 单元测试计划

新增测试：

```text
tests/test_iforward_v3_gru_state.py
tests/test_iforward_v3_history_ema.py
tests/test_iforward_v3_history_gate.py
tests/test_iforward_v3_rollout.py
```

### 15.1 GRU state

测试：

```text
empty state row count correct
read unseen rows returns zero/finite ctx
same frame repeat dt = 0
new frame dt > 0
write_mask false rows unchanged
rigid route.S only writes selected rows
state.detach removes graph
```

### 15.2 History EMA

测试：

```text
support snapshot repeat0 后 pending 改变，但 support_fast 不变
block_exit 后 support_fast 改变
repeat1/repeat2 不更新 support/error
update_norm 每次 after apply 改变
residual 只在 block_exit 改变 error_fast/error_slow
rigid support/residual scatter full rows 正确
```

### 15.3 History gate

测试：

```text
mask_update false -> all gate effective zero
cold_open initialized=0 且 mask_update=true -> gate 接近 1
initialized=1 -> learned gate 生效
min_gate 不绕过 mask_update
rigid gate row数等于 len(route.S)
empty branch 返回 empty gate
```

### 15.4 Rollout integration

测试：

```text
version=v3_gru_history_gate 可跑一个 B1R2 rollout
block_exit 后 history_ema initialized_ratio > 0
per_step 有 gate/gru/update_norm stats
final loss finite
next_state.detach_for_next_rollout 可用
```

---

## 16. 实现顺序

### P0：结构拆分

1. `bridge.predict_delta()`
2. `bridge.apply_branch_scope_event_rows()`
3. `bridge.expand_rigid_delta()`
4. `bridge.apply_delta_only()`
5. `delta_ops.py`

验收：v1/v6 旧路径不变；v3 能用新拆分路径完成一次 update。

### P1：History EMA

1. `history_ema.py` dataclass
2. support snapshot / commit
3. update_norm EMA
4. residual pack API stub

验收：不用 gate，只记录 stats，rollout 能跑通。

### P2：HistoryGate

1. `history_gate.py`
2. branch gate forward
3. mask_update hard binding
4. delta gate

验收：history_gate_only 能稳定跑，gate stats 正常。

### P3：TimeAwarePointGRU

1. `gru_memory.py`
2. read / write_after_update
3. rigid route.S scatter
4. ContextPack 输出

验收：GRU + gate 路径跑通。

### P4：block_exit residual

1. `bridge.compute_block_residual_history()`
2. backproject residual to branches
3. error EMA commit

验收：block_exit 后 error_fast/error_slow 更新，history gate 输入包含 error。

### P5：config / trainer / optimizer group

1. `configs/iforward/iforward_v3_gru_history_gate.yaml`
2. optimizer group 增加 point_gru/history_gate
3. logging 汇总
4. tests

---

## 17. 主要风险与处理

### 风险 1：gate 过强导致当前帧优化变差

处理：

```text
cold_open_uninitialized=true
opacity/sh gate init bias 不要太保守
single-frame phase 先训练 updater + gate
```

### 风险 2：support 只取 repeat0，当前 block 后期 geometry 变化后 support 变旧

这是有意选择。support 是观测证据，不是优化状态。如果 repeat 内频繁刷新 support，会重新引入 repeat-level history 污染。

### 风险 3：update_norm 只用 means，无法感知 appearance blur

v3 P0 按 5_4 复原，只用 means norm。若 history 仍有 appearance blur，v3.1 再加入：

```text
opacity update norm
SH update norm
combined update norm
```

### 风险 4：GRU write_after_update 太弱，ctx 初期为零

这是可接受的。posterior updater 已经直接看到 event，GRU 只是 optimizer memory。单帧阶段不应依赖 GRU。

### 风险 5：residual backproject 计算昂贵

P0 只在 block_exit 对当前 source refs 做，不对 short-window history 做 residual 更新。

---

## 18. 最终 v3 定义

IForward v3 是：

```text
Stage6 event backbone
+ TimeAwarePointGRU optimizer memory
+ original EMA history statistics
+ final writeback attribute history gate
+ block_exit support/residual commit
+ repeat-level update_norm EMA
```

IForward v3 不是：

```text
Mamba version
new long-sequence memory ledger
new scheduler architecture
memory-xCPE version
Stage5_4 full trainer restore
```

一句话总结：

**v3 应该把历史语义重新拉回 5_4：support 和 residual 只在 block_exit 以 detach EMA 方式更新；当前 block 内只有真实写回幅度 update_norm 每个 repeat 更新；TimeAwarePointGRU 只提供 optimizer context；所有 delta 最终必须通过 history gate 才能写回。**
