# Geometry-Grounded ADC-Lite 实现方案文档

目标版本：IForward v3 / v3.1 兼容实现  
实验目标：Experiment A，20w/15w baseline + bg-only clone-only episode-local ADC-Lite  
约束：只做 bg；只做 clone；只在 episode 内存在；不删除、不修改 HGV2；不做 rigid/distant；不做 persistent densification；不改 renderer。

---

## 0. 设计结论

当前 HGV2 能提供 history safety 信号，但它只能调节 gate，不能增加新的局部表达自由度。Geometry-Grounded ADC-Lite 的目标是补上 IForward 当前缺少的最小 densification 能力：

```text
在 rollout 边界，根据 current/history 梯度冲突和几何尺度信号，选择 bg 中少量 Gaussian 作为 parent；
在 episode-local LocalGSState 中 clone 出 child Gaussian；
child 继承 parent 的几何/外观，但 history/memory 冷启动；
后续 rollout 中 child 可以承担 current/detail，parent 保留 history。
```

第一版只做 **bg clone-only episode-local**，不做 split、不做 persistent 写回、不做 distant/rigid。

核心 score 固定为用户指定公式：

```text
S_adc =
  abs_grad_current
+ 0.5 * abs_grad_history
+ 0.5 * scale_or_screen_radius
+ 1.0 * current_history_conflict
```

这不是普通随机增点，也不是完整 3DGS ADC。它是 IForward 训练环境里的 **rollout-boundary capacity allocation**。

---

## 1. 与现有系统的关系

### 1.1 起点

实验 A 从 `experiment002` / 20w/15w baseline 出发：

```text
assets.root = streetforward_assets_pt20w_15w
near_max_points = 200000
distant_max_points = 150000
scheduler_iforward.version = iforward_v3_random_window
blocks_per_episode = 8
rollouts_per_episode = 8
window_policy = random_with_replacement
history_replay.enable = true
```

建议直接从已有 20w/15w baseline checkpoint resume，优先从 40k 附近开始，因为 40k 后开始出现 `r8b1/r4b2`，更能测试 current-history conflict。若资源允许也可以从 0 训练，但不是第一优先级。

### 1.2 HGV2 策略

本方案不删除、不修改 HGV2 代码和配置。

- 如果从 `experiment002` baseline 启动：HGV2 保持 baseline 状态，也就是不开 HGV2。
- 如果之后要和 HGV2 联合测试：直接沿用已有 HGV2 配置，不因 ADC-Lite 调 HGV2 参数。

因此 Experiment A 的唯一新增变量是：

```text
bg clone-only episode-local ADC-Lite
```

不把 HGV2 调整混入 A，避免归因不清。

---

## 2. 为什么是 bg-only / clone-only / episode-local

### 2.1 bg-only

第一版只处理 `LocalGSState.bg`。

原因：

```text
1. 当前门/墙/道路/建筑模糊主要落在 bg 解释区域；
2. distant 几何通常冻结或弱更新，appearance/opacity clone 容易制造远景雾；
3. rigid branch 有 point_ids、instance transform、route.S、template 等复杂状态，第一版不适合 surgery；
4. bg LocalBranchState 最稳定，clone 工程复杂度最低。
```

### 2.2 clone-only

第一版不做 split，只做 clone。

原因：

```text
1. clone 不改变 parent 几何，风险低；
2. 它主要提供额外 appearance/opacity/detail 自由度；
3. 对门/墙/道路这种“形状大致对、观感错”的问题更直接；
4. split 需要主轴方向、scale 收缩、位置偏移，更容易出几何噪声。
```

### 2.3 episode-local

clone 只存在于当前 IForward episode 内。

```text
episode begin: 从 baseline LocalGS 初始化；
rollout start: 根据上一 rollout ADC bank clone 少量 bg points；
rollout 间: clone 随 IForwardState carry；
episode end: IForwardState 清空，clone 消失。
```

不写回 assets，不写回 Stage6 NodeState，不扩长期缓存。

---

## 3. 数据结构新增

新增文件：

```text
models/iforward/adc_lite.py
```

### 3.1 ADCBank

```python
@dataclass
class IForwardADCBank:
    valid: bool
    source_rollout_id: int
    source_episode_id: int
    source_num_current_refs: int
    source_num_history_refs: int

    # bg full-row score tensors, shape [N_bg]
    score: torch.Tensor
    abs_grad_current: torch.Tensor
    abs_grad_history: torch.Tensor
    scale_score: torch.Tensor
    conflict_score: torch.Tensor

    # parent candidate mask, shape [N_bg]
    candidate_mask: torch.Tensor

    # optional diagnostics
    score_topk_mean: torch.Tensor
    score_p90: torch.Tensor
    score_p99: torch.Tensor
```

只存 bg full rows。dtype 建议：

```text
score / components: fp16 or bf16
candidate_mask: bool
```

### 3.2 ADCStateMeta

用于 episode-local clone 管理：

```python
@dataclass
class IForwardADCStateMeta:
    original_bg_count: int
    num_bg_clones_created_episode: int
    parent_index: Optional[torch.Tensor]     # [N_bg_total], -1 for original
    birth_rollout_id: Optional[torch.Tensor] # [N_bg_total], -1 for original
    cooldown_until_rollout: Optional[torch.Tensor]
```

第一版可以简化：只记录 `original_bg_count` 和 `num_bg_clones_created_episode`。更完整的 parent/cooldown 便于防止 clone-of-clone。

### 3.3 IForwardState 扩展

在 `models/iforward/state.py` 中扩展：

```python
@dataclass
class IForwardState:
    local_gs: LocalGSState
    memory: ...
    history: ...
    history_ema: ...
    history_gradient_bank: Optional[HistoryGradientBank] = None
    adc_bank: Optional[IForwardADCBank] = None
    adc_meta: Optional[IForwardADCStateMeta] = None
```

`detach_for_next_rollout()` 要 detach adc bank/meta。

---

## 4. Score 构建

### 4.1 构建时机

在 rollout end，final losses 计算后，构建下一 rollout 用的 ADC bank：

```text
1. 已经有 final local_state；
2. 已经有 L_current / L_history / L_nearby；
3. 不额外 render；
4. 从已有 loss graph 对 final_local_state.bg 求 gradient。
```

插入点：`models/iforward/model.py`，在 final losses 之后、构造 `next_state` 之前。

### 4.2 需要的梯度

```python
g_current = autograd.grad(
    loss_current,
    bg_params,
    retain_graph=True,
    create_graph=False,
    allow_unused=True,
)

g_history = autograd.grad(
    loss_in_rollout_history,
    bg_params,
    retain_graph=True,
    create_graph=False,
    allow_unused=True,
)
```

如果 `loss_in_rollout_history == 0` 或 history refs 为空：

```text
abs_grad_history = 0
current_history_conflict = 0
```

此时 ADC 仍可由 current gradient + scale score 触发，但第一版建议只在 history 存在时启用 clone，避免把它变成纯 current densification。

推荐配置：

```yaml
require_history_for_clone: true
```

### 4.3 abs_grad_current

对 bg 每个点计算属性梯度 norm：

```text
abs_grad_current_i =
    w_means   * ||g_current.means_i||_2
  + w_scales  * ||g_current.scales_log_i||_2
  + w_opacity * |g_current.opacity_logit_i|
  + w_sh      * mean_abs_or_l2(g_current.sh_i)
```

默认：

```yaml
grad_attr_weights:
  means: 1.0
  scales: 0.5
  opacity: 0.75
  sh: 0.75
```

然后做 robust normalize：

```python
abs_grad_current_norm = clip01(
    abs_grad_current / (percentile(abs_grad_current, 99) + eps)
)
```

### 4.4 abs_grad_history

同上：

```text
abs_grad_history_i = norm of ∂L_history / ∂bg_i
```

也做 percentile normalize。

### 4.5 scale_or_screen_radius

第一版优先使用 scale proxy，不依赖 renderer 提供 screen radius：

```python
scale_raw_i = max(exp(scales_log_i))
scale_score_i = clip01(scale_raw_i / (percentile(scale_raw, 95) + eps))
```

如果后续 renderer 能输出 screen radius，可替换或融合：

```python
scale_or_screen_radius = max(scale_score, screen_radius_score)
```

第一版：

```text
scale_or_screen_radius = scale_score
```

### 4.6 current_history_conflict

使用 signed gradients 判断 current 与 history 是否竞争。

对每个点拼接 selected attr gradient 向量：

```text
g_c_i = concat(g_current.means, g_current.scales, g_current.opacity, g_current.sh)
g_h_i = concat(g_history.means, g_history.scales, g_history.opacity, g_history.sh)
```

计算：

```python
cos_i = dot(g_c_i, g_h_i) / (||g_c_i|| * ||g_h_i|| + eps)
conflict_score_i = sqrt(abs_grad_current_norm_i * abs_grad_history_norm_i) * relu(-cos_i)
```

含义：

```text
current 和 history 梯度都大，并且方向相反 → 高 conflict。
```

若 history 无梯度：

```text
conflict_score = 0
```

### 4.7 最终 S_adc

严格使用指定公式：

```python
S_adc = (
    abs_grad_current_norm
  + 0.5 * abs_grad_history_norm
  + 0.5 * scale_score
  + 1.0 * conflict_score
)
```

再乘 candidate mask：

```python
S_adc = S_adc * candidate_mask.float()
```

---

## 5. Candidate mask

只考虑 bg parent，且默认不让 clone 再当 parent。

```python
candidate_mask = torch.ones(N_bg, dtype=bool)
candidate_mask &= torch.arange(N_bg) < adc_meta.original_bg_count
candidate_mask &= torch.isfinite(S_adc)
candidate_mask &= opacity_alpha > alpha_min
candidate_mask &= scale_raw > scale_min
```

建议：

```yaml
candidate:
  alpha_min: 0.005
  scale_min: 1.0e-4
  exclude_clones_as_parent: true
  require_history: true
```

如不要求 history，则可能变成 current-only densification。实验 A 建议 `require_history=true`。

---

## 6. Clone 策略

### 6.1 clone 时机

在下一个 rollout 开始、进入 repeat loop 前执行：

```text
state = carried_state or init_state
if adc_lite.enable and state.adc_bank.valid:
    state = apply_bg_clone_episode_local(state, resolved)
then run normal rollout
```

不能在 repeat 中间 clone。不能在 render 中动态 clone。

### 6.2 选择 parent

```python
scores = adc_bank.score
scores[~candidate_mask] = -inf
parent_idx = topk(scores, k=max_new_points_per_rollout)
```

预算：

```yaml
budget:
  max_new_points_per_rollout: 2000
  max_new_points_per_episode: 8000
  max_total_bg_points_episode: 208000
```

若当前 bg 已达到 cap：不 clone。

### 6.3 clone attributes

对每个 parent 创建一个 child。

#### means

```python
child.means = parent.means + jitter
```

第一版 jitter 使用 deterministic hash noise：

```python
std = mean(exp(parent.scales_log)) * mean_jitter_std_scale
jitter ~ Normal(0, std)
```

配置：

```yaml
mean_jitter_std_scale: 0.05
```

若担心几何噪声，可先设 0.0 做完全重合 clone。但完全重合 clone 只有在后续 delta 区分后才有用，通常 0.03–0.05 更好。

#### scales / quat

```python
child.scales_log = parent.scales_log
child.quats = parent.quats
```

#### opacity：alpha-preserving split

不要直接复制 opacity，也不要简单 logit(alpha/2) 作为唯一选择。为了让两个相同 alpha 合成后尽量接近原 alpha，使用：

```python
alpha_old = sigmoid(parent.opacity_logit)
alpha_each = 1 - sqrt(1 - alpha_old)
parent.opacity_logit = logit(alpha_each)
child.opacity_logit = logit(alpha_each)
```

因为两个相同 alpha 的合成近似：

```text
alpha_combined = 1 - (1 - alpha_each)^2 = alpha_old
```

这比简单 alpha/2 更保守地保持渲染密度。

#### SH

```python
child.sh = parent.sh
```

#### hidden/local latent

如果 bg branch 有 hidden：

```python
child.hidden = parent.hidden
```

但 optimizer memory / GRU state 不复制，见下文。

---

## 7. 关联状态扩展

新增 bg clone 后，必须同步扩展以下状态，否则 shape mismatch。

### 7.1 LocalGSState.bg

扩展：

```text
means
scales_log
quats
opacity_logit
sh_dc / sh_rest 或 sh
hidden
```

具体字段以 `LocalBranchState` 为准。

### 7.2 PointGRU memory

新增 child rows：

```text
hidden = 0
seen = false
last_seen_visit = current_visit 或 -1
uncertainty = initial
```

不要复制 parent GRU memory。原因：child 是新自由度，应 cold-start，让当前 rollout 决定它的 optimizer memory。

### 7.3 History EMA

新增 child rows：

```text
initialized = false
support_fast/slow = 0
error_fast/slow = 0
update_norm_fast/slow = 0
pending buffers = 0
```

不要复制 parent history。原因：parent 保存历史，child 用于 current/detail repair。

### 7.4 HistoryGradientBank / HGV2

不修改 HGV2 代码逻辑，但如果当前 state 存在 history_gradient_bank，需要对新增 child rows 扩展 invalid/zero gradient：

```text
grad_child = 0
valid_child = false
```

这样 HGV2 若开启，也不会把 parent gradient 错用到 child。

### 7.5 ADC bank/meta

当前 ADC bank 用完后可以保留作日志，但 clone 后不应重复使用同一 bank。

建议：

```text
apply clone 后，state.adc_bank = None
```

避免同一个 high score parent 在多个 rollout 被重复 clone。

---

## 8. 与 HGV2 的兼容性

本实验不删除、不修改 HGV2。

ADC-Lite 与 HGV2 关系：

```text
ADC-Lite: 创建 episode-local 新 bg child，解决 capacity allocation；
HGV2: 调整 history gate，提供 history safety；
二者正交。
```

Experiment A 从 `experiment002` baseline 开始，因此 HGV2 保持 baseline 状态。若未来想测试 ADC+HGV2，则加载 `experiment004_hgv2_gradient_prior_20w15w` 配置，保持 HGV2 原样，只添加 ADC-Lite。

---

## 9. 代码插入点

### 9.1 新增模块

```text
models/iforward/adc_lite.py
```

主要 API：

```python
def build_adc_lite_bank_from_losses(
    *,
    loss_current: torch.Tensor,
    loss_history: Optional[torch.Tensor],
    final_local_state: LocalGSState,
    cfg: Mapping[str, Any],
    rollout_id: int,
    episode_id: int,
    num_current_refs: int,
    num_history_refs: int,
) -> Optional[IForwardADCBank]:
    ...


def apply_bg_clone_episode_local(
    *,
    state: IForwardState,
    cfg: Mapping[str, Any],
    rollout_id: int,
    device: torch.device,
) -> Tuple[IForwardState, Dict[str, float]]:
    ...
```

### 9.2 IForwardState 扩展

```text
models/iforward/state.py
```

新增字段：

```python
adc_bank: Optional[IForwardADCBank] = None
adc_meta: Optional[IForwardADCStateMeta] = None
```

`detach_for_next_rollout()` 里 detach。

### 9.3 forward_rollout 开始处应用 clone

`models/iforward/model.py`，拿到 state 后、rollout loop 前：

```python
adc_stats = {}
if self.adc_lite_enable:
    state, adc_stats = apply_bg_clone_episode_local(
        state=state,
        cfg=self.adc_lite_cfg,
        rollout_id=int(resolved.rollout_id_global),
        device=self.device,
    )
local_state = state.local_gs
```

注意：clone 发生在 observe/event 前。这样本 rollout 的 render、2D lifting、updater 都能看到 child。

### 9.4 rollout end 构建 ADC bank

在 final losses 计算后、next_state 构建前：

```python
next_adc_bank = None
if self.adc_lite_enable:
    next_adc_bank = build_adc_lite_bank_from_losses(
        loss_current=final_losses["current"],
        loss_history=final_losses.get("in_rollout_history"),
        final_local_state=local_state,
        cfg=self.adc_lite_cfg,
        rollout_id=int(resolved.rollout_id_global),
        episode_id=int(resolved.episode_id),
        num_current_refs=len(resolved.current_target_indices),
        num_history_refs=len(resolved.history_rollout_target_indices),
    )
```

构建 next_state：

```python
next_state = IForwardState(
    local_gs=local_state,
    memory=next_memory,
    history_ema=next_history_ema,
    history_gradient_bank=next_hgv2_bank,
    adc_bank=next_adc_bank,
    adc_meta=adc_meta,
)
```

### 9.5 trainer 不需要改 scheduler

ADC-Lite 跟随 IForwardState carry。episode reset 时 state cache 清空，clone 自动消失。

---

## 10. 配置草案：Experiment A

新配置建议命名：

```text
configs/iforward/iforward_v3_gg_adc_lite_bg_clone_20w15w.yaml
```

继承 `experiment002` / 20w15w baseline，添加：

```yaml
output_name: experiment007_gg_adc_lite_bg_clone_20w15w

model:
  iforward:
    adc_lite:
      enable: true
      version: gg_adc_lite_v1

      scope:
        branch: bg
        operation: clone
        lifetime: episode_local
        apply_at: rollout_start
        build_bank_at: rollout_end

      require_history_for_clone: true

      score:
        formula: fixed_v1
        weights:
          abs_grad_current: 1.0
          abs_grad_history: 0.5
          scale_or_screen_radius: 0.5
          current_history_conflict: 1.0

        grad_attr_weights:
          means: 1.0
          scales: 0.5
          opacity: 0.75
          sh: 0.75

        normalize:
          mode: percentile
          percentile: 99.0
          eps: 1.0e-8

        scale_proxy:
          mode: max_exp_scale
          percentile: 95.0

        conflict:
          mode: signed_gradient_cosine
          eps: 1.0e-8

      candidate:
        exclude_clones_as_parent: true
        alpha_min: 0.005
        scale_min: 1.0e-4
        min_score: 0.0

      budget:
        max_new_points_per_rollout: 2000
        max_new_points_per_episode: 8000
        max_total_bg_points_episode: 208000
        cooldown_rollouts: 2

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

      logging:
        enable: true
        log_topk: true
        log_score_components: true
        log_clone_counts: true
        log_alpha_mass_check: true
```

HGV2 section not changed. If baseline has no HGV2, it remains off.

---

## 11. 必须新增日志

```text
iforward/adc_lite/enabled
iforward/adc_lite/bank_valid
iforward/adc_lite/applied
iforward/adc_lite/num_cloned_this_rollout
iforward/adc_lite/num_cloned_episode
iforward/adc_lite/bg_count_before
iforward/adc_lite/bg_count_after

iforward/adc_lite/score/topk_mean
iforward/adc_lite/score/p90
iforward/adc_lite/score/p99
iforward/adc_lite/score/abs_grad_current_topk_mean
iforward/adc_lite/score/abs_grad_history_topk_mean
iforward/adc_lite/score/scale_topk_mean
iforward/adc_lite/score/conflict_topk_mean

iforward/adc_lite/parent_alpha_mean
iforward/adc_lite/child_alpha_mean
iforward/adc_lite/alpha_combined_error_mean

iforward/adc_lite/parent_score_mean
iforward/adc_lite/parent_scale_mean
iforward/adc_lite/parent_conflict_mean
```

关键评估指标仍然是：

```text
current_psnr
history_rollout_psnr
nearby_psnr
current_minus_history_gap
loss_current
loss_history
loss_nearby
peak_mem_bytes
step_time_ms
```

---

## 12. 实验 A 运行方案

### 12.1 推荐启动点

最省资源：

```text
从 experiment002 / 20w15w baseline 的 40k checkpoint resume。
```

原因：

```text
40k 后出现 r8b1/r4b2，current-history conflict 更明显；
ADC-Lite 更容易在 r4b2 中表现出价值；
不需要重复早期单帧 warmup。
```

### 12.2 训练长度

第一阶段：

```text
5k step sanity + effect check
```

若稳定：

```text
继续到 10k–20k step
```

### 12.3 成功标准

与 baseline 同 step 比较：

```text
current PSNR: 不下降超过 0.3 dB，最好上升；
history PSNR: +0.2 dB 以上；
nearby PSNR: +0.2 dB 以上；
current-history gap: 缩小，但不是靠 current 下降；
视觉上门/墙/道路灰雾减少，边缘更实；
peak_mem 增加可控，< +2GB；
step_time 增加 < 15%。
```

### 12.4 失败标准

```text
alpha 变厚 / 发雾；
current PSNR 下降 > 0.8 dB；
history 不升；
nearby 下降；
clone 后 bg_count 接近 cap 但无收益；
step_time 或 peak_mem 超预算。
```

---

## 13. 单元测试计划

新增：

```text
tests/test_iforward_adc_lite.py
```

### 13.1 Score formula test

构造已知梯度，验证：

```text
S = c + 0.5h + 0.5s + conflict
```

### 13.2 Alpha-preserving clone test

给定 parent alpha，clone 后检查：

```text
1 - (1 - alpha_parent_new) * (1 - alpha_child) ≈ alpha_old
```

### 13.3 LocalGS bg shape extension test

clone K 个点后：

```text
means/scales/quats/opacity/sh/hidden rows 都 +K
```

### 13.4 Memory/history extension test

clone K 个点后：

```text
GRU bg memory rows +K, child unseen；
history EMA rows +K, child initialized=false；
HGV2 bank rows +K, child valid=false。
```

### 13.5 Episode reset test

episode end 后：

```text
IForwardState cache cleared；
next episode bg count returns to original_bg_count。
```

### 13.6 No repeated clone from same bank test

apply clone 后：

```text
state.adc_bank is None
```

### 13.7 Budget cap test

设置 cap：

```text
max_new_points_per_rollout=10
max_new_points_per_episode=20
```

验证不会超过预算。

---

## 14. 风险与规避

### 14.1 Alpha 变厚 / 雾化

风险：clone 增加 opacity mass。

规避：

```text
alpha-preserving split；
限制 max_new_points_per_rollout；
监控 alpha_combined_error；
必要时降低 child alpha 到 alpha_each * 0.8。
```

### 14.2 clone 无法分化

风险：child 和 parent 太像，后续更新仍同步。

规避：

```text
mean_jitter_std_scale=0.05；
child history cold-open；
child GRU zero_unseen；
child 可获得不同 event/support 后自然分化。
```

### 14.3 只提升 current，不改善 history

风险：ADC 退化成 current densification。

规避：

```text
require_history_for_clone=true；
score 中 history/conflict 占 1.5 总权重；
只在 history refs 存在时 build valid bank。
```

### 14.4 内存增加

预算控制：

```text
每 rollout +2000 bg；
每 episode +8000；
cap 208k；
相比 200k 增加 4%。
```

### 14.5 clone-of-clone 爆炸

规避：

```text
exclude_clones_as_parent=true；
apply clone 后清空 bank；
cooldown_rollouts=2。
```

---

## 15. 与完整 GGS / ADC 的关系

本方案不是完整 Geometry-Grounded Gaussian Splatting，也不是标准 persistent ADC。

它只吸收两个思想：

```text
1. Gaussian 是几何实体，不只是 color token；
2. 高 residual / 高 gradient / 大 scale / current-history conflict 的 Gaussian 需要额外自由度。
```

第一版 geometry grounding 只用 `scale_or_screen_radius`。若 Experiment A 成功，再考虑增加 depth inconsistency：

```text
score += w_depth * depth_consistency_conflict
```

但 A 不做 depth。

---

## 16. 最终预期

如果 hypothesis 成立，Experiment A 应该表现为：

```text
r4b2 中 current/history/nearby 同时改善；
r8b1 current 不明显下降；
视觉上局部门/墙/道路更实，灰雾减少；
HGV2 不参与时也能看到 capacity 增益。
```

如果 A 无效，说明当前主要问题不是缺少 bg 局部自由度，而更可能是：

```text
1. updater/backbone 仍然不够强；
2. appearance/opacity update 机制有问题；
3. distant/rigid 才是主要模糊源；
4. clone 位置/opacity 策略不对，需要 split 或 depth grounding。
```

---

## 17. 实施优先级

```text
P0.1  添加 adc_lite.py 数据结构和 score builder
P0.2  添加 bg clone episode-local surgery
P0.3  扩展 IForwardState / memory / history_ema 状态
P0.4  集成 forward_rollout start/end
P0.5  添加日志
P0.6  添加单元测试
P1    跑 Experiment A：20w/15w baseline 40k resume，5k step
P2    视结果决定是否加入 depth/screen-radius 或 split
```
