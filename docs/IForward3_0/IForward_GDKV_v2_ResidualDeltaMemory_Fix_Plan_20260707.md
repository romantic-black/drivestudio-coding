# IForward GDKV v2：Residual Delta Memory 修正方案

日期：2026-07-07  
适用代码：`drivestudio_stage6_refactor_context_20260707_v46`  
目标：直接修复 Low-rank Gated Delta KV memory state/ctx 长期贴近 RMS 上限的问题，使 memory 从“被 clamp 截断的高能量上下文”变回“可编辑、可收敛、可区分 parent identity 的优化器记忆”。

---

## 0. 执行结论

当前 IForward GDKV 的问题不是普通超参问题。它的根因是：

```text
当前 update rule 允许 write gate 长期大于 erase gate；
在 repeated write 场景下，memory 的 fixed point 会被 write/erase ratio 放大；
decay 又接近 1，repair/repeat_stability 甚至等于 1；
于是 state/ctx 长期被推到 state_rms_max / ctx_rms_max，最后靠 clamp 截断。
```

当前实现：

```python
S_decay = S_old * decay
old = S_decay.T @ (erase * k)
S = S_decay - outer(k, old) + outer(k, write * v)
S = rms_clamp(S, state_rms_max)
```

这个形式看起来像 Gated DeltaNet-2，但在 IForward 的使用条件下有固定点放大风险。若简化为 scalar erase/write 且同一个 key/value 反复写：

```text
y_{t+1} = (1 - erase) * y_t + write * v
fixed point: y* = (write / erase) * v
```

日志中常见：

```text
write_gate_mean ≈ 0.85 - 0.88
erase_gate_mean ≈ 0.23
write / erase ≈ 3.7
```

因此 memory 被推向 `3-4x value` 后贴着 RMS clamp，是符合公式的，不是偶然。

**最重要的判断：**

```text
Step1 diagnostics 是必须的；
Step2 单纯 write<=erase 可以止血，但不是最干净解；
Step3 “改成 residual delta write”方向正确，但要注意：当前代码已经是 GDN2 风格 residual edit；真正需要改成以同一个 read address 为基准、fixed point 为 v 的 residual correction。
```

最终推荐实现：

```text
GDKV v2 = residual-delta-corrective write + optional cleanup erase + pre/post clamp diagnostics
```

核心公式：

```python
S0 = decay * S

# optional cleanup path, weak and separately gated
S_clean = S0 - cleanup_strength * outer(e, S0.T @ e)

old = S_clean.T @ k
residual = v - old
alpha = alpha_gate(token, residual)       # bounded, surprise-aware
S_new = S_clean + outer(k, alpha * residual)
S_new = emergency_rms_clamp(S_new)
```

这个更新有三个关键性质：

```text
1. 如果 memory 已经读出 v，则 residual≈0，不再继续写；
2. fixed point 是 v，不是 (write/erase)*v；
3. stale cleanup 与 corrective write 解耦，避免把 erase/write ratio 变成幅度放大器。
```

---

## 1. 当前实现的问题定位

### 1.1 当前代码路径

文件：

```text
models/iforward/stage2_3/parent_optimizer_gated_delta_kv.py
```

当前 `LowRankGatedDeltaKVCell.write()` 的核心逻辑：

```python
s_old = state.kv_state.to(dtype=torch.float32)
k = rms_unit(key_proj(token)) / sqrt(K)
v = rms_clamp(value_proj(token), value_rms_max)
erase = sigmoid(erase_proj(token)) * erase_gate_max
write = sigmoid(write_proj(token)) * write_gate_max
decay = decay_min + (1 - decay_min) * sigmoid(decay_proj(token))

s_decay = s_old * decay[:, :, None]
old = einsum('nkv,nk->nv', s_decay, erase * k)
s_erased = s_decay - outer(k, old)
s_write = s_erased + outer(k, write * v)
s_new = rms_clamp(s_write, state_rms_max)
```

### 1.2 它和 Gated DeltaNet-2 的关系

Gated DeltaNet-2 的 recurrence 可以写成：

```text
S_t = (I - k_t (b_t ⊙ k_t)^T) D_t S_{t-1} + k_t (w_t ⊙ v_t)^T
```

它的思想是：

```text
b_t：key-side erase gate
w_t：value-side write gate
D_t：channel-wise decay
```

当前 IForward 实现基本复用了这个形式。问题不是“公式完全错了”，而是 **IForward 的使用场景与 GDN2 原始使用场景不同**：

| 项目 | 原 GDN2 | IForward 当前 GDKV |
|---|---|---|
| 输入 | token sequence | parent optimizer write token |
| 状态重置 | layer sequence / cache | episode 内每 repeat 写 |
| decay | learned log-decay，通常 < 1 | assimilation≈0.99，repair=1 |
| 输出 | RMSNorm + SiLU gate + output projection | ctx rms_clamp + adapter + fusion gate |
| memory size | multi-head large K/V | single low-rank K=16,V=32 |
| training objective | sequence modeling | 3DGS iterative state editing |
| writes | 每 token，但语言 token 有局部模式 | 每 update，可能同一 parent 反复写相近信息 |

因此，即使公式来自 GDN2，当前 IForward 仍会出现 saturation。

### 1.3 为什么 fusion gate 小不能防止 state 写满

配置：

```yaml
fusion:
  gate_init:
    assimilate: 0.05
    repair: 0.03
```

这个 gate 控制的是：

```text
ctx -> adapter -> parent_event 的融合强度
```

不是：

```text
GDKV state 的写入强度
```

GDKV 内部 write/erase projection bias 当前是 0：

```text
erase ≈ sigmoid(0) = 0.5
write ≈ sigmoid(0) = 0.5
```

训练后日志显示 write 更高、erase 更低，于是 state 会持续增大。fusion gate 小只会让模型一开始读出的 ctx 对 event 影响小，不会阻止 state 写满。

---

## 2. Step1-3 是否正确，是否能彻底解决？

### Step1：加 diagnostics ——必须做

必须新增以下指标，否则无法区分问题发生在 state、ctx、fusion 还是 identity binding：

```text
pre_clamp_state_rms_mean/max
post_clamp_state_rms_mean/max
state_clamp_scale_mean/min
state_clamp_ratio
pre_clamp_ctx_rms_mean/max
post_clamp_ctx_rms_mean/max
ctx_clamp_ratio
old_rms_mean/max
value_rms_mean/max
residual_rms_mean/max
write_gate_mean/max
erase_gate_mean/max
write_erase_ratio_mean/max
alpha_mean/max
cleanup_strength_mean/max
memory_contribution_rms
parent_event_rms_before/after
contribution_ratio
fusion_gate_mean by visit_kind/distribution
```

没有这些指标，继续调 gate/cap 基本是盲调。

### Step2：write <= erase ——能止血，但不彻底

一种简单修法：

```python
erase = sigmoid(erase_raw)
write = erase_value_proxy * sigmoid(write_raw)
```

或 scalar 简化：

```text
write <= erase
```

这样 scalar fixed point 不会超过 `v`：

```text
y* = (write / erase) * v <= v
```

但它仍然有问题：

```text
1. key-side erase 和 value-side write 维度不同，严格绑定不自然；
2. 它只是限制幅度，不保证重复写已记忆信息时写入为 0；
3. stale association 在其他 key 上仍不能被清理。
```

所以它是止血，不是最终解。

### Step3：改成 residual delta write ——方向正确，但必须改对

如果只是说“用 residual delta”，容易误以为当前代码还不是 residual。实际上，当前 GDN2 形式已经包含：

```text
write target - erase read
```

但由于 read address 是 `erase * k`，write target 是 `write * v`，fixed point 变成 `write/erase * v`。

真正适合 IForward 的 residual delta 应该是：

```python
old = S.T @ k
residual = v - old
S = decay * S + outer(k, alpha * residual)
```

这保证：

```text
重复写同一 key/value 时，old -> v，residual -> 0，写入自动停止。
```

如果还想保留 erase，则应把 erase 作为 **cleanup path**，而不是作为 main write residual 的 read 缩放因子。

### 是否彻底解决？

它能彻底解决当前最直接的问题：

```text
write/erase ratio 放大导致 state/ctx 长期贴 clamp。
```

但它不能保证解决所有性能问题：

```text
不能解决 K=16,V=32 的容量限制；
不能解决 parent feature 太粗导致 key 区分度弱；
不能解决 shuffled_coverage 本身难；
不能保证 current 到 25 dB；
不能自动让 full vs shuffle_state gap 大幅拉开。
```

所以更精确的结论是：

> residual-delta-corrective write 是当前 memory saturation 的必要修复，但不是 IForward 整体性能的充分条件。

---

## 3. 直接解决当前问题的实现方案：GDKV v2

### 3.1 新增 update rule

在 `LowRankGatedDeltaKVCell` 增加：

```python
update_rule: str = 'gdn2_legacy' | 'balanced_residual_delta_v1'
```

默认新配置使用：

```yaml
gated_delta_kv:
  update_rule: balanced_residual_delta_v1
```

保留 legacy 便于 ablation。

### 3.2 新增 projection heads

当前 heads：

```python
key_proj
value_proj
erase_proj
write_proj
decay_proj
q_proj
```

新增：

```python
alpha_proj          # residual write strength, scalar or value_dim
cleanup_key_proj    # optional independent erase address
cleanup_proj        # cleanup strength
surprise_proj       # optional residual/surprise modulation
```

最小实现可以不加 `cleanup_key_proj`，先用当前 key 的弱 cleanup：

```python
e = k
```

推荐实现加 `cleanup_key_proj`，因为 stale information 可能不在当前 write key 上。

### 3.3 新 update 公式

#### 3.3.1 decay

```python
S0 = S_old * decay[:, :, None]
```

decay 建议：

```yaml
decay_min:
  assimilation: 0.98
  repair: 0.995
  repeat_stability: 0.995
```

不要再让 repair=1.0 作为默认。

#### 3.3.2 optional cleanup erase

```python
e_raw = cleanup_key_proj(token)        # [N,K]
e = rms_unit(e_raw) / sqrt(K)
cleanup = sigmoid(cleanup_proj(token) + cleanup_bias) * cleanup_max
old_e = S0.T @ e
S_clean = S0 - outer(e, cleanup * old_e)
```

建议初始：

```yaml
cleanup_max: 0.2
cleanup_init: 0.02
```

第一版可以只对 repair/shuffled 开启 cleanup，对 repeat 弱一些：

```yaml
cleanup_max_by_kind:
  assimilation: 0.1
  repair: 0.2
  repeat_stability: 0.1
```

#### 3.3.3 corrective residual write

```python
k = rms_unit(key_proj(token)) / sqrt(K)
v = rms_clamp(value_proj(token), value_rms_max)
old = S_clean.T @ k
residual = v - old
```

write strength：

```python
alpha = sigmoid(alpha_proj(token) + alpha_bias) * alpha_max
```

可选 surprise gate：

```python
surprise = rms(residual).detach()
surprise_gate = clamp(surprise / surprise_target, 0, 1)
alpha = alpha * surprise_gate
```

写入：

```python
S_write = S_clean + outer(k, alpha * residual)
```

#### 3.3.4 emergency clamp

```python
pre_rms = rms(S_write)
S_new = rms_clamp(S_write, state_rms_max)
clamp_scale = state_rms_max / pre_rms
```

但目标是：

```text
clamp_ratio < 5%
post_state_rms_mean << state_rms_max
```

如果 clamp 仍然长期触发，说明容量或 identity 仍有问题。

---

## 4. 配置草案

```yaml
model:
  iforward:
    parent_optimizer_memory:
      gated_delta_kv:
        update_rule: balanced_residual_delta_v1

        # dimensions unchanged
        K: 16
        V: 32
        query_rms_unit: true
        key_rms_unit: true
        value_rms_max: 2.0

        # keep current caps first; expect less frequent clamp
        state_rms_max: 4.0
        ctx_rms_max: 4.0

        # residual write
        alpha_mode: value_channel       # scalar | value_channel
        alpha_max: 1.0
        alpha_init: 0.10                # bias logit(0.1)
        surprise_gating: true
        surprise_target_rms: 1.0
        min_alpha_on_unseen: 0.5

        # cleanup erase
        cleanup_enable: true
        cleanup_key: learned            # current_key | learned
        cleanup_max: 0.2
        cleanup_init: 0.02
        cleanup_by_kind:
          assimilation: 0.10
          repair: 0.20
          repeat_stability: 0.10

        # decay
        decay_min:
          bootstrap: 1.0
          assimilate: 0.98
          assimilation: 0.98
          repair: 0.995
          repeat_stability: 0.995

        # legacy fields kept for backward compatibility but unused in v2 main write
        erase_gate_max: 1.0
        write_gate_max: 1.0
```

---

## 5. 代码修改点

### 5.1 `LowRankGatedDeltaKVCell.__init__`

新增 args：

```python
update_rule: str = 'gdn2_legacy'
alpha_mode: str = 'value_channel'
alpha_max: float = 1.0
alpha_init: float = 0.1
cleanup_enable: bool = True
cleanup_key: str = 'learned'
cleanup_max: float = 0.2
cleanup_init: float = 0.02
surprise_gating: bool = True
surprise_target_rms: float = 1.0
min_alpha_on_unseen: float = 0.5
```

新增 modules：

```python
if alpha_mode == 'scalar':
    self.alpha_proj = nn.Linear(token_dim, 1)
else:
    self.alpha_proj = nn.Linear(token_dim, value_dim)

self.cleanup_key_proj = nn.Linear(token_dim, key_dim)
self.cleanup_proj = nn.Linear(token_dim, 1)
```

初始化：

```python
alpha_bias = logit(alpha_init)
cleanup_bias = logit(cleanup_init / cleanup_max)
```

### 5.2 `write()` dispatch

```python
if self.update_rule == 'gdn2_legacy':
    return self._write_gdn2_legacy(...)
elif self.update_rule == 'balanced_residual_delta_v1':
    return self._write_residual_delta(...)
else:
    raise ValueError(...)
```

### 5.3 `_write_residual_delta()` 伪代码

```python
S_old = state.kv_state.float()
seen_old = state.seen.bool()
mask = write_mask.bool()

k = norm_key(key_proj(token))
v = rms_clamp(value_proj(token).float(), value_rms_max)

decay = compute_decay(visit_meta)
S0 = S_old * decay[:, :, None]

if cleanup_enable:
    if cleanup_key == 'learned':
        e = norm_key(cleanup_key_proj(token))
    else:
        e = k
    cleanup = sigmoid(cleanup_proj(token) + cleanup_bias) * cleanup_max_by_kind
    old_e = einsum('nkv,nk->nv', S0, e)
    S_clean = S0 - einsum('nk,nv->nkv', e, cleanup * old_e)
else:
    S_clean = S0

old = einsum('nkv,nk->nv', S_clean, k)
residual = v - old

alpha = sigmoid(alpha_proj(token) + alpha_bias) * alpha_max
if surprise_gating:
    surprise = rms(residual).detach()
    sg = clamp(surprise / surprise_target_rms, 0, 1)
    alpha = alpha * sg

# unseen rows need stronger first write
alpha = where(seen_old[:, None], alpha, maximum(alpha, min_alpha_on_unseen))

S_write = S_clean + outer(k, alpha * residual)
pre_rms = state_rms(S_write)
S_new = rms_clamp(S_write, state_rms_max)

S_final = where(mask, S_new, S_old)
seen = seen_old | mask
```

### 5.4 aux stats

必须记录：

```text
pre_state_rms_mean/max
post_state_rms_mean/max
state_clamp_scale_mean/min
state_clamp_ratio
old_rms_mean/max
value_rms_mean/max
residual_rms_mean/max
alpha_mean/max
cleanup_mean/max
cleanup_old_rms_mean/max
surprise_gate_mean/max
```

不要只记录 post-clamp state_rms，否则看不出是否仍然被 clamp 截断。

---

## 6. 兼容与迁移

### 6.1 checkpoint 兼容

state shape 不变：

```text
kv_state: [N,K,V]
```

因此旧 checkpoint 的 optimizer memory state 可以继续加载。

但 module 参数新增：

```text
alpha_proj
cleanup_key_proj
cleanup_proj
```

如果从旧 checkpoint resume，需要：

```text
strict=False
skip missing new heads
initialize new heads as above
```

建议在切换到 v2 时：

```text
reset optimizer state for parent_optimizer_gdkv parameters
```

因为新 heads 没有 Adam moments，旧 heads 的 Adam moment 也可能不匹配新 update dynamics。

### 6.2 ablation 保留

保留 legacy：

```yaml
update_rule: gdn2_legacy
```

新主线：

```yaml
update_rule: balanced_residual_delta_v1
```

validation 必须同时跑：

```text
full
memory_off
memory_shuffle_state
```

关键看：

```text
full-off 是否保持；
full-shuffle gap 是否变大；
state clamp ratio 是否下降；
shuffled_coverage current 是否上升。
```

---

## 7. 测试计划

### 7.1 Synthetic fixed-point test

构造单 parent、单 key、单 value，重复写 100 次。

Legacy 预期：

```text
state_rms 接近上限；
read ctx 可能接近 write/erase 放大后的 value；
clamp_ratio 高。
```

Residual v2 预期：

```text
old -> v；
residual -> 0；
state_rms 收敛且低于上限；
clamp_ratio 约 0。
```

### 7.2 Conflicting values test

同一 key 写 v1，再写 v2。

预期：

```text
read 从 v1 平滑转向 v2；
不会同时保留两者导致 state_rms 打满。
```

### 7.3 Many-key interference test

随机 64 个 key/value 写入 K=16,V=32 memory。

比较：

```text
legacy vs v2 的 clamp_ratio
retrieval MSE
full-shuffle sensitivity
```

### 7.4 IForward 500-step smoke

配置：

```text
resume existing checkpoint
update_rule=balanced_residual_delta_v1
metrics interval: gdkv_aux_interval=100
```

通过条件：

```text
无 NaN/Inf；
state_clamp_ratio < 10%；
pre_state_rms_mean 不长期贴 max；
loss 正常下降。
```

### 7.5 5k resume validation

从 50k 或 60k checkpoint resume 5k：

```text
repeat/shuffle/repair distributions 都覆盖；
validation_v4 memory_ablation 完整跑。
```

关键指标：

```text
full-off current/history gap 不下降；
full-shuffle gap 上升；
shuffled_coverage current 不下降，最好上升；
state/ctx clamp ratio 显著下降。
```

---

## 8. 成功标准

GDKV v2 成功不是只看 PSNR，而是先看 memory behavior：

```text
1. state_clamp_ratio 从长期高值降到 <5-10%；
2. ctx_clamp_ratio 从长期高值降到 <10%；
3. residual_rms 随 repeated write 下降；
4. full-off gap 保持或增大；
5. full-shuffle gap 增大，说明 identity binding 增强；
6. shuffled_coverage current 不低于 legacy，最好提升；
7. 无 NaN/Inf，无 GDKV drift。
```

如果只看到：

```text
state_rms 下降，但 full-off gap 也消失
```

说明 memory 被削弱，不是修好了。

---

## 9. 为什么这是“直接解决”而不是调参

调参方案：

```text
write_gate_max 降低
state_rms_max 提高
decay 降低
```

这些只能改变速度或容量，不能改变 fixed point。

GDKV v2 改的是固定点：

```text
legacy fixed point: y* = (write / erase) * v
v2 fixed point:     y* = v
```

所以它直接解决：

```text
重复写入相同或相近信息时 state 继续增大的问题。
```

这才是当前 memory saturation 的根因修复。

---

## 10. 下一步选择

推荐先做最小 v2：

```text
update_rule=balanced_residual_delta_v1
cleanup_enable=false
surprise_gating=true
alpha_mode=value_channel
alpha_init=0.1
repair_decay_min=0.995
```

如果 clamp 降了但 stale interference 仍明显，再开 cleanup：

```text
cleanup_enable=true
cleanup_key=learned
cleanup_max=0.2
cleanup_init=0.02
```

也就是说分两阶段：

```text
v2a: corrective residual write
v2b: corrective residual write + cleanup erase
```

先不要同时扩 K/V 或提高 RMS cap。等确认 fixed point 修好后，再决定是否扩容。

---

## 11. 后续 GPT 上下文块

```text
当前 IForward GDKV legacy update 是：
S_decay = decay * S
old = S_decay^T (erase*k)
S = S_decay - outer(k, old) + outer(k, write*v)
S = rms_clamp(S)
该形式是 GDN2-like，但在 scalar 分析下 repeated write fixed point 为 (write/erase)*v。
日志显示 write≈0.85, erase≈0.23, decay≈0.99/1.0，所以 memory state/ctx 长期贴 RMS cap。
注意 fusion gate init 0.05/0.03 不是 write gate，它只控制 memory ctx 融合到 parent_event 的比例。
真正修复方向是 GDKV v2：old = S^T k, residual = v-old, S += outer(k, alpha*residual)，必要时先做 weak cleanup erase。
目标是 fixed point = v，重复写已记住信息时 residual -> 0。
必须新增 pre/post clamp、residual_rms、write/erase ratio、contribution_ratio 等诊断。
不要先盲目调 write_gate_max 或 state_rms_max，也不要先扩 K/V。
```
