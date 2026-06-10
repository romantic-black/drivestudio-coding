# IForward History Gate v2：Lagged History-Gradient Prior Gate 实现方案

版本：2026-06-10 draft  
目标：替代当前独立 HSP probe / projection，把已有 history loss 的梯度转化为 history gate 的输入和监督信号。

---

## 0. 结论

当前 HSP 的主要问题不是“没有信号”，而是执行形态不对：

```text
HSP probe / projection 是第二套历史保护机制；
history gate 是第一套历史保护机制；
两者并列会重复、难调、昂贵。
```

History Gate v2 的设计是：

```text
不再额外 probe；
不再额外 projection；
不再创建第二个 gate；
在 rollout end 复用已有 L_history 的梯度；
把这个梯度作为下一 rollout 每个 repeat 的 history gate 输入；
再用一个轻量 auxiliary loss 训练 gate 在 harmful direction 上关小。
```

也就是：

```text
执行器仍然只有 history_gate；
新增的是 history_gradient_prior。
```

---

## 1. 对“固定 history gradient 作为下一 rollout 输入”的客观评估

### 1.1 可行性

可行，但它不是精确梯度，而是 lagged prior。

```text
Rollout t end：
  从 L_history(final_state_t) 中提取 ∂L_history / ∂local_G_final_t。

Rollout t+1：
  把这个 gradient 固定作为 prior，给每个 repeat 的 history gate 使用。
```

这和在线精确约束不同。它的语义是：

```text
上一 rollout 的 history loss 告诉我：
哪些点、哪些属性方向在最近历史视角中敏感；
下一 rollout 如果又要沿着这些方向更新，就应该更谨慎。
```

它能有效的前提是：

```text
1. 同一 episode 内 local_state 连续 carry；
2. random-window revisit 会重复触达相似的点；
3. history gradient 在相邻 rollout 间有足够相关性；
4. gradient 只作为 gate input / supervision，不直接强制投影。
```

当前 IForward v3 满足前两个条件：scheduler 是 episode 内 carry 的 random-window rollout，history replay 是 rollout final target 的一部分。

### 1.2 主要隐患

#### 隐患 A：stale gradient

上一 rollout 的 gradient 对应的是：

```text
上一 rollout 结束时的 local_state；
上一 rollout 的 history refs；
上一 rollout 的 target distribution。
```

下一 rollout 的 source/current/history 都可能不同。

因此不能把 gradient 当成真理，只能当成 prior。

第一版处理：

```text
不做跨 episode 保存；
episode reset 时清空；
如果当前 rollout 没有 history loss，则不生成新 bank；
下一 rollout 无 valid bank 时 history gate 退化为 v1。
```

不引入 age 作为核心机制，但保留 logging：

```text
hgv2/bank_valid
hgv2/bank_rollout_gap
```

#### 隐患 B：可能误伤当前修复

如果一个历史帧本来就错，旧 gradient 可能阻止当前帧修复它。

第一版不做复杂 repair policy，但 gate 仍然可以看到原有 history EMA：

```text
history_error_fast / slow
support_fast / slow
initialized
current support
```

因此 HGV2 不应该 analytic cap，而应该作为 learned gate input，让模型自己结合 history EMA 决定是否使用 gradient prior。

#### 隐患 C：直接把 gradient 当执行器会重复 history gate

不做：

```python
gate = gate * analytic_cap(cos_damage)
```

第一版只做：

```text
gradient feature input + gate auxiliary loss
```

这样 history gate 是唯一执行器。

#### 隐患 D：几何梯度容易锁死

用户要求一开始保存所有属性 gradient。这个可以做，但不等于所有属性都强约束。

第一版策略：

```text
所有属性都保存：means / scales / quat / opacity / sh。
所有属性都输入 gate。
aux loss 全属性启用，但几何 attr weight 更低。
不使用 hard analytic cap。
```

推荐权重：

```yaml
means: 0.25
scales: 0.25
quat: 0.10
opacity: 1.00
sh: 1.00
```

这样能满足“全保存、少实验次数”的需求，同时降低几何锁死风险。

---

## 2. 与 HSP 的关系

当前 HSP 做的是：

```text
额外选 history probe refs；
额外 render probe；
额外 autograd.grad；
得到 damage signal；
可选 projection / damage loss。
```

History Gate v2 改成：

```text
直接复用已有 rollout final 的 L_history；
从 L_history 对 final local_G 的梯度中提取 history sensitivity；
保存到 gradient bank；
下一 rollout 使用。
```

因此：

```yaml
history_safe_projection:
  enable: false
```

新增：

```yaml
history_gate_v2:
  enable: true
  source: previous_rollout_history_gradient
```

---

## 3. 总体流程

### 3.1 Rollout t forward

使用上一 rollout 存好的 gradient bank：

```text
history_gradient_bank_{t-1}
```

每个 repeat：

```python
delta_candidate = posterior_updater(event, ctx)
delta_scoped = apply_branch_scope_event_rows(delta_candidate)

grad_prior = history_gradient_bank.gather(event.rows)
damage_features = compute_damage_features(grad_prior, delta_scoped)

gate = history_gate_v2(
    event=event,
    history_ema=history_ema,
    memory_ctx=ctx,
    damage_features=damage_features,
)

delta_applied = gate * delta_scoped
local_state = apply_delta(local_state, expand_rigid(delta_applied))
```

关键点：

```text
当前 rollout 内不更新 gradient bank；
每个 repeat 使用固定 bank；
bank 只作为 gate 输入，不直接改 delta。
```

### 3.2 Rollout t end

正常计算：

```text
L_current
L_history
L_nearby
L_delta_reg
L_gate_grad_aux
```

如果本 rollout 有 history loss：

```python
g = torch.autograd.grad(
    L_history,
    final_local_state_params,
    retain_graph=True,
    create_graph=False,
    allow_unused=True,
)
```

然后：

```python
g = detach(g)
g = normalize_and_pack_full_state_gradient(g)
next_state.history_gradient_bank = g
```

如果没有 history loss：

```text
next_state.history_gradient_bank = invalid / empty
```

不要保留更老 bank，避免 stale gradient。

---

## 4. 数据结构

新增文件：

```text
models/iforward/history_gradient_bank.py
```

### 4.1 GradientBankAttr

```python
@dataclass
class GradientBankAttr:
    direction: torch.Tensor  # normalized gradient direction, fp16/fp32
    log_norm: torch.Tensor   # per-row log grad norm, fp16/fp32
    valid: torch.Tensor      # [N], bool
```

### 4.2 HistoryGradientBank

```python
@dataclass
class HistoryGradientBank:
    bg_means: GradientBankAttr
    bg_scales: GradientBankAttr
    bg_quat: GradientBankAttr
    bg_opacity: GradientBankAttr
    bg_sh: GradientBankAttr

    distant_means: GradientBankAttr
    distant_scales: GradientBankAttr
    distant_quat: GradientBankAttr
    distant_opacity: GradientBankAttr
    distant_sh: GradientBankAttr

    rigid_means: GradientBankAttr
    rigid_scales: GradientBankAttr
    rigid_quat: GradientBankAttr
    rigid_opacity: GradientBankAttr
    rigid_sh: GradientBankAttr

    valid: bool
    source_rollout_id: int
    source_history_loss: float
    source_history_num_refs: int
```

### 4.3 Shape 约定

```text
means:   [N, 3]
scales:  [N, 3]
quat:    [N, 4]  # gradient wrt stored quaternion state
opacity: [N, 1]
sh:      [N, C]
```

对于 `quat`，delta 是 axis-angle，gradient 是 quaternion-state gradient。使用时需要把 axis-angle candidate 转成 induced quaternion delta：

```python
q_delta_param = apply_axis_angle(q_old, delta_axis_angle) - q_old
cos_damage_quat = cosine(grad_quat, q_delta_param)
```

---

## 5. IForwardState 修改

在 `models/iforward/state.py` 中：

```python
@dataclass
class IForwardState:
    local_gs: LocalGSState
    memory: Optional[IForwardMemoryState]
    history: IForwardShortWindowHistory
    history_ema: Optional[IForwardHistoryEMAState]
    history_gradient_bank: Optional[HistoryGradientBank]
    ...
```

`detach_for_next_rollout()` 需要：

```python
history_gradient_bank = self.history_gradient_bank.detach()
```

episode reset / new state 初始化：

```python
history_gradient_bank = None
```

---

## 6. Rollout end 记录 gradient bank

### 6.1 需要保留 final local state leaf tensors

在 v3 forward 中，final local_state 是可微的。rollout final render 已经基于 final local_state 计算 `loss_in_rollout_history`。

新增函数：

```python
def build_history_gradient_bank_from_loss(
    loss_history: torch.Tensor,
    final_local_state: LocalGSState,
    rollout_id: int,
    history_num_refs: int,
    cfg: Dict,
) -> Optional[HistoryGradientBank]:
    ...
```

### 6.2 取梯度

```python
params = [
    final_local_state.bg.means,
    final_local_state.bg.scales_log,
    final_local_state.bg.quats,
    final_local_state.bg.opacity_logit,
    final_local_state.bg.sh,
    final_local_state.distant.means,
    final_local_state.distant.scales_log,
    final_local_state.distant.quats,
    final_local_state.distant.opacity_logit,
    final_local_state.distant.sh,
    final_local_state.rigid.means,
    final_local_state.rigid.scales_log,
    final_local_state.rigid.quats,
    final_local_state.rigid.opacity_logit,
    final_local_state.rigid.sh,
]

grads = torch.autograd.grad(
    loss_history,
    params,
    retain_graph=True,
    create_graph=False,
    allow_unused=True,
)
```

注意：

```text
不额外 render；
不 create_graph；
grad detach；
只从已有 L_history 图上取一次梯度。
```

### 6.3 normalize

每一行：

```python
norm = grad.flatten(1).norm(dim=-1, keepdim=True)
direction = grad / norm.clamp_min(eps)
valid = isfinite(norm) & (norm > min_norm)
log_norm = log(norm + eps)
```

建议：

```yaml
history_gate_v2:
  bank:
    dtype: fp16
    min_grad_norm: 1.0e-8
    normalize: per_row
```

---

## 7. 每个 repeat 使用 gradient bank

### 7.1 gather full-state bank to event rows

新增：

```python
def gather_history_grad_bank_for_event(
    bank: HistoryGradientBank,
    event: Stage6EventPack,
    local_state: LocalGSState,
) -> HistoryGradientEventPack:
    ...
```

对于 bg/distant：

```text
直接对应 full rows。
```

对于 rigid：

```text
event.rigid 是 route.S 子集；
必须用当前 event.route.S 从 full-state bank gather。
```

不要存 event-row gradient。否则下一 rollout/下一 repeat route.S 变化会错位。

### 7.2 delta 转换

每个 attr 的 candidate delta：

```text
means: delta.means
scales: delta.scales_log
opacity: delta.opacity_logit
sh: delta.sh
quat: apply_axis_angle(q_old, delta.quat_axis_angle) - q_old
```

### 7.3 damage features

对每个 branch row、每个 attr：

```python
delta_norm = norm(delta_attr)
cos = dot(grad_dir, delta_attr) / (delta_norm + eps)
pos = relu(cos)
neg = relu(-cos)
valid = bank_attr.valid & (delta_norm > eps)
```

其中 `grad_dir` 已经 per-row normalized，所以不需要再除以 grad norm。

输入 feature：

```text
cos_damage_attr
pos_damage_attr
neg_damage_attr
grad_log_norm_attr
bank_valid_attr
```

一开始全部属性都生成：

```text
means / scales / quat / opacity / sh
```

---

## 8. HistoryGateV2 网络修改

当前 history gate v1 输入大致包含：

```text
event feature
param/history embed
memory ctx
branch embed
step embed
```

v2 增加：

```text
gradient prior feature
```

### 8.1 Gradient feature encoding

每个 row 拼接：

```text
for each attr in [means, scales, quat, opacity, sh]:
  cos
  pos
  neg
  log_norm
  valid
```

共：

```text
5 attrs × 5 dims = 25 dims
```

新增 MLP：

```python
self.grad_prior_embed = nn.Sequential(
    nn.Linear(25, grad_embed_dim),
    nn.LayerNorm(grad_embed_dim),
    nn.GELU(),
    nn.Linear(grad_embed_dim, grad_embed_dim),
)
```

推荐：

```yaml
history_gate_v2:
  grad_embed_dim: 16
```

最终 gate 输入：

```python
gate_input = concat([
    original_gate_input,
    grad_prior_embed,
])
```

### 8.2 初始化

为了不破坏 v1 行为：

```python
last linear layer for grad_prior path = zero init
```

或者：

```python
grad_prior_scale = learnable scalar initialized to 0.0
```

推荐第二种：

```python
self.grad_prior_scale = nn.Parameter(torch.tensor(0.0))
gate_input = original + sigmoid_or_tanh_scale * grad_embed
```

第一阶段让模型自己决定是否使用 gradient prior。

---

## 9. Auxiliary loss：训练 gate 用 gradient prior

仅把 gradient prior 当输入可能学习很慢。新增 gate 辅助损失：

```python
harm = relu(cos_damage - tau)
L_close = mean(stopgrad(harm) * gate)
```

但只关 harmful gate 容易使 gate 全关。加一个轻量 safe-open：

```python
safe = relu(-cos_damage - tau_safe)
L_open = mean(stopgrad(safe) * (1 - gate))
```

总损失：

```python
L_hgv2 =
    Σ_attr w_attr * L_close_attr
  + safe_open_weight * Σ_attr w_attr * L_open_attr
```

推荐第一版：

```yaml
history_gate_v2:
  auxiliary_loss:
    enable: true
    close_weight: 0.02
    safe_open_weight: 0.002
    tau_cos: 0.05
    tau_safe: 0.10
    detach_damage: true
    train_gate_only: true
    attr_weights:
      means: 0.25
      scales: 0.25
      quat: 0.10
      opacity: 1.00
      sh: 1.00
```

实现细节：

```text
damage feature detach；
delta detach；
grad detach；
gate 不 detach；
该 auxiliary loss 主要训练 history_gate 参数。
```

如果想严格只训练 gate，可以在计算该 loss 时只允许 gate 参数得到梯度；但通常 `harm.detach() * gate` 已经只通过 gate 分支回传。

---

## 10. 配置草案

```yaml
model:
  iforward:
    history_safe_projection:
      enable: false

    history_gate_v2:
      enable: true
      source: previous_rollout_history_gradient

      bank:
        update_on: rollout_end
        clear_on_episode_reset: true
        if_no_history_loss: invalidate
        storage: full_state
        dtype: fp16
        normalize: per_row
        min_grad_norm: 1.0e-8

      attrs:
        means: true
        scales: true
        quat: true
        opacity: true
        sh: true

      features:
        include_cos: true
        include_pos: true
        include_neg: true
        include_log_norm: true
        include_valid: true
        grad_embed_dim: 16
        grad_prior_scale_init: 0.0

      auxiliary_loss:
        enable: true
        close_weight: 0.02
        safe_open_weight: 0.002
        tau_cos: 0.05
        tau_safe: 0.10
        detach_damage: true
        attr_weights:
          means: 0.25
          scales: 0.25
          quat: 0.10
          opacity: 1.00
          sh: 1.00

      logging:
        enable: true
```

---

## 11. 代码插入点

### 11.1 新增文件

```text
models/iforward/history_gradient_bank.py
models/iforward/history_gate_v2_features.py
```

### 11.2 修改 `state.py`

添加：

```python
history_gradient_bank: Optional[HistoryGradientBank]
```

并在 detach / reset 中处理。

### 11.3 修改 `model.py`

在每个 repeat：

```python
if self.history_gate_v2_enabled and state.history_gradient_bank is not None:
    grad_event = gather_history_grad_bank_for_event(...)
    grad_features = compute_grad_damage_features(grad_event, delta_scoped, local_state, event)
else:
    grad_features = zero_grad_features_like_event(...)

gate = self.history_gate(..., grad_features=grad_features)
```

在 rollout final loss 后：

```python
if cfg.history_gate_v2.enable and loss_in_rollout_history > 0:
    next_grad_bank = build_history_gradient_bank_from_loss(
        loss_history=loss_in_rollout_history,
        final_local_state=local_state,
        rollout_id=rollout_id,
        history_num_refs=history_num_refs,
    )
else:
    next_grad_bank = None

next_state.history_gradient_bank = next_grad_bank
```

### 11.4 修改 `history_gate.py`

增加可选输入：

```python
grad_features: Optional[IForwardHistoryGradientFeaturePack] = None
```

如果 None：

```text
行为等同 v1。
```

如果有：

```text
concat grad_prior_embed。
```

---

## 12. 日志

必须加：

```text
iforward/hgv2/bank_valid
iforward/hgv2/bank_source_history_loss
iforward/hgv2/bank_source_history_num_refs

iforward/hgv2/damage_pos_ratio/means
iforward/hgv2/damage_pos_ratio/scales
iforward/hgv2/damage_pos_ratio/quat
iforward/hgv2/damage_pos_ratio/opacity
iforward/hgv2/damage_pos_ratio/sh

iforward/hgv2/gate_harmful_mean/means
iforward/hgv2/gate_harmful_mean/scales
iforward/hgv2/gate_harmful_mean/quat
iforward/hgv2/gate_harmful_mean/opacity
iforward/hgv2/gate_harmful_mean/sh

iforward/hgv2/gate_safe_mean/means
iforward/hgv2/gate_safe_mean/scales
iforward/hgv2/gate_safe_mean/quat
iforward/hgv2/gate_safe_mean/opacity
iforward/hgv2/gate_safe_mean/sh

iforward/loss_hgv2_gate
iforward/hgv2/grad_prior_scale
```

成功信号：

```text
gate_harmful_mean 下降；
gate_safe_mean 不明显下降；
history PSNR / nearby PSNR 提升；
current-history gap 缩小；
current PSNR 下降小于 0.3~0.5 dB。
```

失败信号：

```text
所有 gate 都下降；
current PSNR 明显下降；
history/nearby 不提升；
grad_prior_scale 长期接近 0，说明模型不用 gradient prior。
```

---

## 13. 单元测试

新增：

```text
tests/test_iforward_history_gradient_bank.py
tests/test_iforward_history_gate_v2.py
```

测试内容：

1. no history loss -> bank invalid。
2. history loss -> bank shapes 正确。
3. detach 后 bank 不带 graph。
4. rigid branch 当前 route.S gather 不错位。
5. zero bank -> history gate v2 与 v1 输出一致。
6. harmful cos 高时 auxiliary loss 对 gate 有梯度。
7. safe cos 负时 safe-open loss 对 gate 有反向约束。
8. quat induced delta shape 和 finite 检查。

---

## 14. 推荐实验

### 实验 A：HGV2-all-attrs，主实验

从当前 20w/15w 直接训练，不使用ckpt
```yaml
history_safe_projection:
  enable: false
history_gate_v2:
  enable: true
  attrs:
    means: true
    scales: true
    quat: true
    opacity: true
    sh: true
  auxiliary_loss:
    close_weight: 0.02
    safe_open_weight: 0.002
```

训练 5k step 先看趋势。

### 实验 B：HGV2 feature-only，对照

```yaml
history_gate_v2:
  enable: true
  auxiliary_loss:
    enable: false
```

如果 A 好 B 不好，说明 auxiliary loss 关键。  
如果 B 也好，说明 gradient prior 作为输入就足够。  
如果 A/B 都不好，说明 stale prior 或 gate 学习路径无效。

资源紧张时只跑 A，B 可以 1k sanity。

---

## 15. 最终判断

固定 history gradient 作为下一 rollout 的 gate 输入是可行的，但必须理解为：

```text
lagged gradient prior，而不是 exact gradient。
```

它可能提升性能的原因是：

```text
history gate v1 只知道历史统计；
history gate v2 额外知道“上一 rollout 的 history loss 对哪些属性方向敏感”。
```

它的主要隐患是 stale / noisy / geometry lock。第一版通过：

```text
full-state storage;
episode reset 清空;
per-row normalization;
feature input 而非 hard cap;
全属性保存但几何低权重;
safe-open loss 防止 gate 全关;
```

来控制风险。

这比当前 HSP 更贴近 IForward 环境：

```text
不额外 probe；
不额外 render；
不额外 projection；
复用已有 history loss；
让 history gate 成为唯一执行器。
```
