# Gate-Suppressed Update ADC-Lite 实现方案

版本：IForward v3 / v19 后续方案  
目标：替换当前多项加权 ADC score，使用 **history gate / HGV2 抑制的更新量** 作为 clone 选择依据。  
约束：不使用 learnable selector head；只考虑 bg branch；只做 clone；episode-local；从头训练，不 resume checkpoint。

---

## 0. 核心结论

当前 fixed-score ADC-Lite 的问题不是 clone 工程本身，而是 score 选择机制没有命中真正的 current-history conflict。已有日志显示，即使把 conflict 权重调高，clone 仍然更像被 scale / abs-grad 驱动，而不是被 current-history conflict 驱动；clone 后 current 可提升，但 history/nearby 没有稳定改善。

Gate-Suppressed Update ADC-Lite 的核心改动是：

```text
不要用 abs_grad_current + abs_grad_history + scale + conflict 的多项加权分数。

直接使用：
  当前 updater 想写入，但 history gate / HGV2 抑制掉的更新量。

如果一个点的候选 delta 很大，而 gate 把它关小，说明这个点正在承担 current 与 history 的冲突。
这种点才最应该 clone。
```

数学上：

```text
S_i = RMS_attr( normalize_attr( ||(1 - g_i_attr) * Δ_i_attr|| ) )
```

其中：

```text
Δ_i_attr = history gate 前的 candidate delta
g_i_attr = history gate / HGV2 后的 learned gate，不能把 hard invalid mask 当成 gate
S_i      = bg point i 的 clone score
```

HGV2 不作为 ADC 的额外 score 项。HGV2 的作用是影响 gate；ADC 通过 gate 的抑制量自然接收 HGV2 信号。

---

## 1. 非目标

本方案不做：

```text
1. 不使用 learnable selector head。
2. 不做 dynamic persistent ADC。
3. 不改 distant / rigid。
4. 不做 split。
5. 不调 clone jitter / opacity split / scale init 作为主实验变量。
6. 不使用 current-history gradient weighted score。
7. 不使用 validation 中不带 ADC 的旧指标作为结论。
```

---

## 2. 与现有 v19 ADC-Lite 的区别

| 项目 | 当前 v19 fixed-score ADC | 新 Gate-Suppressed ADC |
|---|---|---|
| score | `abs_grad_current + 0.5 history + scale + conflict` | `||(1-gate)*delta||` |
| 是否多项调权 | 是 | 否 |
| 是否依赖 HGV2 | 弱/间接 | 直接通过 gate 体现 |
| 是否需要 history gradient | 是 | 否 |
| validation 下能否构建 bank | 当前 fixed-score 依赖 loss grad，`no_grad` 下不可用 | 可以，因为只用 forward 中 delta/gate |
| clone 语义 | 大点 / 高梯度点 / 弱 conflict | 当前想改但 history gate 不让改的点 |

---

## 3. 模块新增与改名

建议保留当前文件：

```text
models/iforward/adc_lite.py
```

新增数据结构：

```python
@dataclass
class IForwardADCSuppressionBank:
    valid: bool
    source_rollout_id: int
    source_episode_id: int
    score: Tensor                 # [N_bg]
    score_max: Tensor             # [N_bg]
    score_sum: Tensor             # [N_bg]
    score_count: Tensor           # [N_bg]
    candidate_mask: Tensor        # [N_bg]
    parent_gate_mean: Tensor      # [N_bg]
    parent_delta_demand: Tensor   # [N_bg]
    parent_support_mean: Tensor   # [N_bg]
    score_p90: Tensor
    score_p99: Tensor
    score_topk_mean: Tensor
```

也可以复用 `IForwardADCBank`，但建议另设 `score_type="gate_suppressed_update"`，避免和 fixed-score 混淆。

---

## 4. 需要记录什么

在每个 repeat 的 **history gate 之后、delta apply 之前** 记录 bg 点的 suppressed update。

当前流程大致是：

```text
delta_raw = posterior_updater(...)
delta_scoped = apply_branch_scope_event_rows(delta_raw)
gate_pack = history_gate(...)
delta_gated_event = gate_delta_pack(delta_scoped, gate_pack)
```

新增：

```text
suppression_score = compute_gate_suppressed_score(delta_scoped.bg, gate_pack.bg, mask_update_bg)
accumulator.accumulate(bg_indices, suppression_score)
```

注意：score 必须用 learned gate，而不能直接用 `effective_gate = gate * mask_update` 后的结果。否则 invalid / unsupported 点会因为 hard mask=0 被误判成“强抑制”。

如果当前代码只暴露 effective gate，则第一版必须这样处理：

```python
score = where(mask_update_bg, ||(1 - effective_gate) * delta||, 0)
```

但更推荐修改 `history_gate.py`，同时返回：

```text
gate_raw      # min_gate + sigmoid logits，尚未乘 mask_update
gate_effective
mask_update
```

ADC score 使用：

```text
(1 - gate_raw) * delta
```

candidate filter 使用：

```text
mask_update
```

---

## 5. Attribute 处理

只考虑 bg，但 bg 内所有 updatable 属性都参与 score：

```text
means
scales_log
quat_axis_angle
opacity_logit
sh
```

不做语义加权。为了避免量纲不一致，使用 per-attribute normalization。

### 5.1 每个属性的 suppressed norm

```python
def attr_norm(x):
    # x: [N, C]
    return sqrt(mean(x ** 2, dim=-1))

supp_attr = attr_norm((1 - gate_attr) * delta_attr)
demand_attr = attr_norm(delta_attr)
```

### 5.2 normalization

第一版使用 rollout 内 percentile normalization：

```python
denom_attr = percentile(demand_attr[mask_update], 95).clamp_min(eps)
supp_attr_normed = supp_attr / denom_attr
supp_attr_normed = clamp(supp_attr_normed, 0, score_clip)
```

这样不需要人工调 attr 权重。

### 5.3 合并属性

推荐 RMS：

```python
score_i = sqrt(mean_attr(supp_attr_normed_i ** 2))
```

也可以记录 `max_attr` 作为日志，但 selection 用 RMS。

---

## 6. Rollout 内 accumulator

新增 transient accumulator，不进入 `IForwardState`，只在当前 forward_rollout 内存在。

```python
class GateSuppressedADCAccumulator:
    score_sum: Tensor [N_bg]
    score_max: Tensor [N_bg]
    score_count: Tensor [N_bg]
    gate_sum: Tensor [N_bg]
    delta_demand_sum: Tensor [N_bg]
    support_sum: Tensor [N_bg]

    def accumulate(indices, score, gate_mean, delta_demand, support, mask):
        score_sum[indices] += score
        score_max[indices] = max(score_max[indices], score)
        score_count[indices] += 1
        gate_sum[indices] += gate_mean
        delta_demand_sum[indices] += delta_demand
        support_sum[indices] += support
```

Selection score 推荐用：

```text
score = score_max
```

原因：

```text
一个点只要在某个 repeat 中被强烈 gate 抑制，就可能需要 clone；
用 sum 会让 repeats 多的 rollout 天然分数更高。
```

同时日志保留 `score_sum / score_count`。

---

## 7. Rollout end 构建 ADC bank

在 `forward_rollout` 末尾，代替当前 `build_adc_lite_bank_from_losses(...)`：

```python
next_adc_bank = build_gate_suppressed_adc_bank(
    accumulator=adc_suppression_accumulator,
    local_state=local_state,
    history_ema=history_ema,
    adc_meta=state.adc_meta,
    cfg=adc_lite_cfg,
    rollout_id=resolved.rollout_id_global,
    episode_id=resolved.episode_id,
)
```

这个函数不需要 autograd，不需要 history loss gradient，所以 validation 也可使用。

### 7.1 candidate filter

Bank candidate 只做 hard filter，不加权：

```text
bg only
not cloned child
not boundary
not cooldown
history initialized
score_count >= min_count
score >= min_score
alpha > alpha_min
scale > scale_min
finite score
```

可选：

```text
current support > min_support
```

但是 support 只能作为 filter，不进入 score。

### 7.2 默认 filter 建议

```yaml
candidate:
  require_history: true
  min_count: 2
  min_score_percentile: 99.0
  alpha_min: 0.005
  scale_min: 0.0001
  exclude_clones_as_parent: true
  exclude_boundary_parents: true
```

---

## 8. Next rollout start 应用 clone

沿用当前：

```text
apply_at: rollout_start
```

流程：

```python
if state.adc_bank is not None and state.adc_bank.valid:
    parents = topk(state.adc_bank.score[candidate_mask], budget)
    clone_bg_parents(parents)
    state.adc_bank = None
else:
    no clone
```

必须保证：

```text
ADC bank 只消费一次。
```

当前 v19 `apply_bg_clone_episode_local()` 已经在读取 bank 后设置 `state.adc_bank = None`，这个行为必须保留。

---

## 9. Clone 策略不调参

按用户要求，本方案不调整 clone/几何自由度。保留当前 clone 策略：

```yaml
clone:
  opacity_split: alpha_preserving
  mean_jitter_std_scale: 0.05
  scale_init: parent
  quat_init: parent
  sh_init: parent
  local_hidden_init: parent
  gru_memory_init: zero_unseen
  history_ema_init: cold_open
  hgv2_gradient_init: zero_invalid
```

注意：`alpha_preserving` 必须保留，避免 clone 后 alpha mass 翻倍。

---

## 10. HGV2 的使用

HGV2 要引入，但不作为 ADC 的独立 score 项。

推荐配置：

```yaml
history_gate_v2:
  enable: true
```

ADC score 依赖 HGV2 的方式是：

```text
HGV2 → 改变 history gate 输出 → 改变 suppressed update → 改变 ADC score
```

不要新增 learnable ADC head，也不要将 HGV2 gradient feature 直接拼进 ADC score。

---

## 11. Scheduler 与状态生命周期

### 11.1 训练主生命周期

当前 scheduler：

```text
episode_begin
  rollout 0
  rollout 1
  ...
  rollout 7
episode_end
```

IForwardState carry 规则：

```text
rollout end 且 carry_scene_state_after_rollout=True 且 episode_end=False:
  trainer 保存 out.next_state.detach_for_next_rollout()

否则:
  丢弃 state_cache，并 reset runtime node_state
```

`adc_bank` 与 `adc_meta` 都在 `IForwardState.detach_for_next_rollout()` 中 detach。当前 v19 已经这样处理，必须保留。

### 11.2 adc_bank 生命周期

```text
rollout t start:
  使用来自 rollout t-1 end 的 adc_bank
  如果 clone，则消费并清空 adc_bank

rollout t forward:
  accumulator transient 记录 suppressed update

rollout t end:
  构建 next_adc_bank

rollout t+1 start:
  消费 next_adc_bank
```

因此：

```text
adc_bank 最多跨 1 个 rollout；
adc_bank 不跨 episode；
adc_bank 不在 validation/training 之间共享；
adc_bank 不写入 Stage6 node_state；
adc_bank 不写入 assets。
```

### 11.3 adc_meta 生命周期

`adc_meta` 跟随 episode 内 state carry：

```text
episode begin:
  adc_meta = None
  ensure_adc_meta_for_state() 初始化 original_bg_count、parent_index、birth_rollout、cooldown

每次 clone:
  更新 parent_index / birth_rollout_id / cooldown_until_rollout
  num_bg_clones_created_episode += num_new

rollout carry:
  detach 保存

episode end:
  整个 IForwardState 被丢弃，adc_meta 释放
```

### 11.4 accumulator 生命周期

```text
只存在于当前 forward_rollout 内；
rollout end 后构建 bank；
然后释放；
不进入 state_cache。
```

---

## 12. 释放与扩展一致性

当 clone 发生时，必须同步扩展：

```text
local_gs.bg
point_gru.bg
history_ema.bg
history_gradient_bank.bg / HGV2 bank
adc_meta
```

当前 v19 的 clone 已经包含对 GRU / history EMA / HGV2 gradient bank 的扩展语义，`hgv2_gradient_init: zero_invalid` 必须保留。

新增 Gate-Suppressed ADC bank 不需要扩展，因为它在 clone 之前被消费并清空。下一轮新的 bank 会按 clone 后的 bg size 重建。

---

## 13. 从头训练策略

用户要求不 resume checkpoint，因此训练从 step 0 开始：

```yaml
training:
  start_step: 0
  resume_checkpoint: null
```

但是不建议从 step 0 就允许 clone。此前从头训练 ADC 表现偏弱，说明基础 updater/history gate 未成型时，clone 会增加训练噪声。

因此推荐：

```yaml
adc_lite:
  start_step: 40000
```

也就是说：

```text
从头训练模型；
但 ADC clone 在 40k 后才启用。
```

这不违反“不 resume ckpt”。它只是 warmup ADC。

为什么 40k：当前 scheduler 40k 后进入 `r8b1/r4b2`，多 block rollout 开始出现；这是 ADC 需要解决 current-history conflict 的阶段。

如果想更激进：

```yaml
adc_lite:
  start_step: 20000
```

但第一版建议 40k。

---

## 14. 是否限制到 multi-block rollout

推荐第一版：

```yaml
enable_policy:
  min_blocks_per_rollout: 2
```

原因：之前日志显示单 block r8b1 中 clone 容易变成 current fitting 增容，而不是 history conflict 解决。

因此：

```text
r8b1: 不 clone，但仍可记录 suppressed score 日志；
r4b2 / r2b4: clone；
```

如果你希望从头训练中观察 r8b1，也可以只 logging，不 apply。

---

## 15. 推荐配置草案

```yaml
model:
  iforward:
    history_gate_v2:
      enable: true

    adc_lite:
      enable: true
      version: gate_suppressed_update_v1

      scope:
        branch: bg
        operation: clone
        lifetime: episode_local
        apply_at: rollout_start
        build_bank_at: rollout_end

      start_step: 40000

      enable_policy:
        min_blocks_per_rollout: 2
        log_only_before_start: true

      score:
        type: gate_suppressed_update
        attr_normalize:
          mode: rollout_percentile
          percentile: 95.0
          eps: 1.0e-8
        attr_merge: rms
        repeat_merge: max
        score_clip: 10.0

      candidate:
        require_history: true
        require_support: true
        min_count: 2
        min_score_percentile: 99.0
        alpha_min: 0.005
        scale_min: 0.0001
        exclude_clones_as_parent: true
        exclude_boundary_parents: true
        boundary_margin_eps: 0.0001

      budget:
        max_new_points_per_rollout: 1000
        max_new_points_per_episode: 4000
        max_total_bg_points_episode: 204000
        cooldown_rollouts: 2

      clone:
        opacity_split: alpha_preserving
        mean_jitter_std_scale: 0.05
        aabb_eps: 1.0e-5
        scale_init: parent
        quat_init: parent
        sh_init: parent
        local_hidden_init: parent
        gru_memory_init: zero_unseen
        history_ema_init: cold_open
        hgv2_gradient_init: zero_invalid

      logging:
        enable: true
        log_suppression_score: true
        log_parent_stats: true
        log_clone_counts: true
        log_alpha_mass_check: true
```

---

## 16. Validation 设计

当前旧 validation 的问题是：它虽然调用 `forward_rollout`，但 fixed-score ADC bank 依赖 loss gradient；validation 在 `torch.no_grad()` 下不能构建该 bank。因此旧 validation 没有真正测到 ADC。

Gate-Suppressed ADC 解决了这个问题，因为 bank 只依赖 forward 中的 delta/gate，不依赖 autograd gradient。

### 16.1 新 validation 目标

Validation 必须回答：

```text
同样的 eval rollout 序列中，有 ADC clone 与无 ADC clone 的差异是什么？
```

### 16.2 双模式 validation

新增两个 mode：

```text
full_adc
no_adc
```

两者使用同一个 eval segment、同一组 rollout batches、同一个 initial state。

流程：

```python
for mode in ["full_adc", "no_adc"]:
    carried_state = None
    for rollout_batch in fixed_eval_rollouts:
        out = model.forward_rollout(batch, carried_state=carried_state, ablation=mode)
        carried_state = out.next_state.detach_for_next_rollout() if carry else None
```

`no_adc` ablation 只关闭 clone apply，但仍可选择是否记录 suppression score。建议：

```text
no_adc: 不 build bank，不 apply clone。
full_adc: build bank + apply clone。
```

### 16.3 Validation rollout 数量

ADC 至少需要：

```text
rollout t build bank
rollout t+1 apply bank
```

因此 `rollouts_per_segment` 不能太少。

推荐：

```yaml
iforward_validation:
  rollouts_per_segment: 5
```

固定形状建议：

```text
rollout0: r8b1  warmup，不看 ADC
rollout1: r4b2  build suppression bank
rollout2: r4b2  apply ADC
rollout3: r2b4  apply ADC
rollout4: r4b2  apply ADC
```

如果当前 scheduler validation 只支持 `fixed_shape_names=[r8b1,r4b2,r2b4]`，需要扩成 5 个固定 rollout，或者允许循环使用。

### 16.4 Validation 指标

保留原指标：

```text
current_psnr
history_rollout_psnr
nearby_psnr
current-history gap
```

新增 ADC 指标：

```text
val_adc/full/adc_applied_ratio
val_adc/full/num_cloned_mean
val_adc/full/bg_count_after_mean
val_adc/full/suppression_parent_score_mean
val_adc/full/clone_rollout_current_psnr
val_adc/full/clone_rollout_history_psnr
val_adc/full/clone_rollout_nearby_psnr
val_adc/no_adc/current_psnr
val_adc/delta/full_minus_noadc_current
val_adc/delta/full_minus_noadc_history
val_adc/delta/full_minus_noadc_nearby
```

必须按 `adc_applied` 分组，否则 rollout0/无 bank rollout 会稀释结论。

### 16.5 Validation 状态隔离

Validation 必须继续：

```text
保存训练 state_cache
清空 validation state_cache
validation 内部独立 carry
结束后恢复训练 state_cache
reset runtime node_state
```

当前 `tools/train_iforward.py` 已经有这种隔离逻辑，必须保留。

---

## 17. 代码改动清单

### P0：新增 suppressed update accumulator

文件：`models/iforward/adc_lite.py`

新增：

```python
GateSuppressedADCAccumulator
compute_gate_suppressed_score(...)
build_gate_suppressed_adc_bank(...)
```

### P0：history gate 返回 raw gate / mask_update

文件：`models/iforward/history_gate.py`

需要确保 ADC 能拿到：

```text
gate_raw
mask_update
```

如果暂时不能改，就用 effective gate 并强制：

```text
score = 0 when mask_update=False
```

### P0：forward_rollout 中记录 suppression score

文件：`models/iforward/model.py`

在每个 repeat 的 gate 之后、delta apply 之前：

```python
if adc_enabled:
    adc_accumulator.accumulate_from_bg_delta_gate(...)
```

rollout end：

```python
next_adc_bank = build_gate_suppressed_adc_bank(accumulator, ...)
```

替换旧的 `build_adc_lite_bank_from_losses`。

### P0：ADC validation ablation

文件：`models/iforward/model.py` 或 `validation.py`

新增 ablation：

```text
no_adc
```

在 `no_adc` 下：

```text
skip apply_bg_clone_episode_local
skip build adc bank
```

### P1：validation full_adc vs no_adc

文件：`tools/train_iforward.py`

新增 ADC validation 对比，或扩展当前 validation rows：

```text
mode=full_adc / no_adc
```

### P1：日志

新增训练日志：

```text
adc_suppressed/score_mean
adc_suppressed/score_topk_mean
adc_suppressed/parent_score_mean
adc_suppressed/parent_gate_mean
adc_suppressed/parent_delta_demand_mean
adc_suppressed/selected_parent_suppression_rank_percentile
adc_suppressed/num_cloned_this_rollout
adc_suppressed/num_cloned_episode
adc_suppressed/bg_count_before
adc_suppressed/bg_count_after
```

---

## 18. 实验设置

用户要求从头训练：

```yaml
training:
  start_step: 0
  resume_checkpoint: null
```

建议输出名：

```text
experiment008_gate_suppressed_adc_bg_clone_fromscratch_20w15w
```

训练长度：

```text
至少 60k，最好 100k。
```

因为 ADC 从 40k 开启，60k 才能看到 20k step ADC 行为。

如果资源有限：

```text
先跑到 50k，重点看 40k–50k。
```

---

## 19. 成功标准

和 baseline `experiment002` 对比。

### 19.1 40k–50k 总体

```text
history PSNR >= baseline + 0.2 dB
nearby PSNR >= baseline
current PSNR >= baseline - 0.2 dB
current-history gap 缩小，但不是靠 current 大幅下降
```

### 19.2 r4b2

```text
history / nearby 提升；
current 不明显下降；
clone rollout 不再比 non-clone rollout 差。
```

### 19.3 ADC 自身

```text
selected_parent_suppression_rank_percentile 高；
parent_gate_mean 低于 all_gate_mean；
parent_delta_demand_mean 高于 all_delta_demand_mean；
clone 不再由 scale rank 主导。
```

---

## 20. 失败判据

如果出现以下情况，说明 score 仍不对：

```text
1. clone rollout history 低于 non-clone rollout；
2. current 大涨但 history/nearby 下降；
3. selected_parent_suppression_rank 不高；
4. parent_gate_mean 不低；
5. ADC 开启后 gap 变大；
6. 40k–50k r4b2 current/history/nearby 没有任何正向变化。
```

---

## 21. 总结

Gate-Suppressed Update ADC 的核心是：

```text
不要再人为组合 scale、gradient、conflict。

直接 clone 那些：
  当前 updater 想写，
  但 history gate / HGV2 不让写的 bg points。
```

它的优势是：

```text
1. score 单一、直观；
2. 不需要 learnable head；
3. 与 HGV2 自然耦合；
4. validation 中可运行 ADC，因为不依赖 autograd gradient；
5. 不需要调 clone 几何参数；
6. 更直接对应 current-history 容量冲突。
```

