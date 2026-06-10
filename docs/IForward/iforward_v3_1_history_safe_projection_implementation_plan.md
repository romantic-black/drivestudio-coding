# IForward v3.1：History-Safe Projected Update 实现方案

目标：在不增加点、不动态 densify、不做随机 dropout、不改变 render 贡献的前提下，直接约束 IForward 当前迭代的 delta 不沿着破坏历史视角的方向写入。

当前基线：IForward v3 已经是 `scheduler_iforward.version=iforward_v3_random_window`，episode 内 random-window rollout、current/history/nearby 监督均存在；history replay 开启并以 all-cams visited frames 作为历史目标；TimeAwarePointGRU、EMA history、history gate 均启用。当前配置仍是 fixed point set，不进行 dynamic split/prune。IForward v3 的 `stage6_0.local_rollout.writeback_policy=none`，状态由 IForwardState 在 episode 内 carry。当前 v3 配置使用 16/32 宽度，posterior updater 输入 `input_vsm_ctx=true`，history gate、GRU、history replay 共同影响 delta 写入。

---

## 1. 核心判断

当前问题不应再优先建模成：

```text
哪些点应该随机冻结？
是否需要 reserve/densification？
是否要控制点对当前视角贡献？
```

更直接的建模是：

```text
当前 posterior updater 给出的 delta 是否会提升当前帧 loss，但同时增大历史帧 loss？
```

history gate 只能控制 delta 幅度：

```text
delta_final = gate * delta
```

它无法区分：

```text
安全方向：改善 current，不损害 history
危险方向：改善 current，但损害 history
```

因此 v3.1 增加 **History-Safe Projected Update**：

```text
先预测 delta；
再用历史视角的一阶梯度判断这个 delta 是否会让历史 loss 上升；
如果会上升，就把 delta 中破坏历史的一阶方向分量投影掉。
```

---

## 2. 数学定义

局部 GS 状态记为：

```text
θ
```

当前模型在某个 repeat 内预测：

```text
Δθ
```

对一个历史 probe target 集合 `H`，历史 loss：

```text
L_H(θ)
```

历史梯度：

```text
g_H = ∂L_H(θ) / ∂θ
```

一阶近似：

```text
L_H(θ + Δθ) - L_H(θ) ≈ <g_H, Δθ>
```

如果：

```text
<g_H, Δθ> > τ
```

说明当前更新会增加历史 loss。投影：

```text
Δθ_safe = Δθ - s * max(0, <g_H, Δθ> - τ) / (||g_H||² + ε) * g_H
```

其中：

```text
s: projection strength，训练中 warmup，从 0 到 0.5/1.0
τ: 容忍阈值，可分属性设置，第一版用 0
ε: 数值稳定项
```

如果 `dot <= τ`，不改 delta。

---

## 3. 为什么不做二阶梯度

history probe 的梯度只用于修正 delta，不让其反传到 renderer/probe 网络路径。

实现原则：

```python
with torch.enable_grad():
    proxy_state = detach_clone_local_state(local_state, requires_grad=True)
    hist_loss = bridge.render_loss(proxy_state, history_probe_indices)
    g_hist = torch.autograd.grad(
        hist_loss,
        probe_state_tensors,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )

g_hist = detach(g_hist)
```

然后：

```python
delta_safe = project(delta_gated, g_hist.detach())
```

`delta_safe` 仍然对 `delta_gated` 可微；网络参数通过后续 current/history/nearby render loss 训练。不会形成二阶梯度。

---

## 4. IForward 当前代码插入点

当前 v3 update 流程在 `models/iforward/model.py::forward_rollout` 中大致是：

```python
delta_raw = bridge.predict_delta(...)
delta_scoped = bridge.apply_branch_scope_event_rows(delta_raw)
gate_pack = history_gate(...)
delta_gated_event = gate_delta_pack(delta_scoped, gate_pack)
delta = bridge.expand_rigid_delta(delta=delta_gated_event, event=event, local_state=local_state)
local_state = bridge.apply_delta_only(local_state=local_state, delta=delta)
history_ema.record_update_norm(delta=delta, ...)
point_gru.write_after_update(delta_raw=delta_gated_event, ...)
```

v3.1 插入位置：

```python
delta_gated_event = gate_delta_pack(delta_scoped, gate_pack)

# 新增：history-safe projection，仍保持 event-row delta，方便 GRU/write/update_norm 对齐
if hsp is not None:
    delta_safe_event, hsp_aux, hsp_loss = hsp(
        local_state=local_state,
        event=event,
        delta_event=delta_gated_event,
        resolved=resolved,
        batch=batch,
        step=step,
        step_context=step_context,
        history_ema=history_ema,
        probe_cache=block_probe_cache,
    )
else:
    delta_safe_event = delta_gated_event

delta = bridge.expand_rigid_delta(delta=delta_safe_event, event=event, local_state=local_state)
local_state = bridge.apply_delta_only(local_state=local_state, delta=delta)
history_ema.record_update_norm(delta=delta, ...)
point_gru.write_after_update(delta_raw=delta_safe_event, ...)
```

关键：

```text
projection 应作用在 gated delta 后；
update_norm 应记录 projected full delta；
GRU write 应看到 projected event-row delta；
否则 GRU 会记录一个实际没有写入的危险 delta。
```

---

## 5. 新增模块文件

建议新增：

```text
models/iforward/history_safe_projection.py
```

导出：

```python
class IForwardHistorySafeProjection(nn.Module):
    def forward(
        self,
        *,
        local_state: LocalGSState,
        event: EventPack,
        delta_event: DeltaPack,
        resolved: IForwardResolvedBatch,
        batch: Dict[str, Any],
        step: IForwardResolvedStep,
        step_context: IForwardMemoryStepContext,
        history_ema: IForwardHistoryEMAState,
        bridge: IForwardStage6Bridge,
        probe_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DeltaPack, Dict[str, float], torch.Tensor]:
        ...
```

辅助 dataclass：

```python
@dataclass
class HistoryGradBranch:
    means: Optional[torch.Tensor]
    scales_log: Optional[torch.Tensor]
    opacity_logit: Optional[torch.Tensor]
    sh: Optional[torch.Tensor]

@dataclass
class HistoryGradPack:
    bg: HistoryGradBranch
    distant: Optional[HistoryGradBranch]
    rigid: Optional[HistoryGradBranch]
```

第一版不投影 quat / hidden：

```text
quat_axis_angle: disabled
hidden: disabled
```

原因：当前 render 梯度直接对应 `state.quats`，而 updater 输出是 `quat_axis_angle`，两者不是同一参数化。第一版先投影 shape 一致的属性：

```text
means
scales_log
opacity_logit
sh
```

---

## 6. Probe target 选择

IForward v3 resolver 已经区分 target roles：

```text
final_current_recon
final_history_replay
final_nearby_rollout
```

HSP 只使用 `resolved.history_rollout_target_indices`。

第一版 probe 策略：

```yaml
probe:
  source: resolved_history_rollout_targets
  frames_per_block: 1
  cams_per_frame: 1
  policy: deterministic_recent_or_hash
  fallback_policy: skip_if_no_history
```

实现：

```python
history_indices = list(resolved.history_rollout_target_indices)
if not history_indices:
    return delta_event, aux_skip, zero_loss

# 按 frame 分组
frame_to_indices = group_by_frame(resolved.target_refs[idx] for idx in history_indices)

# 选 1 个 frame：默认 recent，也可以 hash(block_uid)
probe_frame = select_frame(frame_to_indices, step)

# 选 1 个 camera：默认 hash 选；后续可用 all cams
probe_indices = select_cams(frame_to_indices[probe_frame], cams_per_frame=1)
```

不要一开始用所有 history refs。当前 history replay 最多 8 个 frame × 3 cams，直接全用会显著增加显存和时间。

---

## 7. Probe 计算频率

第一版不建议每 repeat 都重新 render/backward history probe。

推荐：

```yaml
probe_frequency: block_enter
reuse_within_block: true
```

含义：

```text
block_enter / repeat_idx=0 时计算一次 history gradient；
同一 block 内 r2/r4/r6/r8 的后续 repeat 复用该 g_hist。
```

虽然后续 repeat 中 local_state 已变化，梯度会 stale，但这是可接受的工程折中。K<=8，且目标是抑制明显历史有害方向，不需要精确优化二阶轨迹。

可选高级模式：

```yaml
probe_frequency: every_n_repeats
n: 2
```

不建议初始使用。

---

## 8. Gradient probe state

需要新增工具：

```python
def make_probe_local_state(local_state, *, attrs) -> Tuple[LocalGSState, List[torch.Tensor], Dict[str, TensorRef]]:
    ...
```

实现要点：

```text
1. detach + clone 当前 local_state；
2. 对需要投影的属性设置 requires_grad=True；
3. 对不投影属性 detach，不要求梯度；
4. rigid_template detach_clone；
5. 返回 tensor list，用于 autograd.grad。
```

第一版属性：

```yaml
attrs:
  means: true
  scales: true
  opacity: true
  sh: true
  quat: false
```

branch 中 SH 的映射：

```python
# state side
sh_dc:   [N, 3]
sh_rest: [N, B-1, 3]

# delta side
delta.sh: [N, B*3]
```

需要把 `grad(sh_dc)` 和 `grad(sh_rest)` 合并成与 `delta.sh` 同 shape：

```python
grad_sh = torch.cat([
    grad_sh_dc[:, None, :],
    grad_sh_rest,
], dim=1).reshape(N, -1)
```

如果 `sh_rest` 不存在或阶数为 0，则只用 DC。

---

## 9. Full-row gradient 转 event-row gradient

为了让 projection 后的 delta 仍能传给 GRU write，HSP 应在 event-row delta 上操作，而不是 full-row delta 后再反向映射。

映射规则：

```text
bg:      event rows 与 local_state.bg 全量行对齐

distant: event rows 与 local_state.distant 全量行对齐

rigid:   event.rigid 是 route.S 子集，需要从 full rigid grad 中 gather route.S
```

新增函数：

```python
def grad_pack_to_event_rows(
    grad_full: HistoryGradPack,
    event: EventPack,
    local_state: LocalGSState,
) -> HistoryGradPack:
    ...
```

rigid gather：

```python
idx = event.route.S.to(torch.long)
grad_event.rigid.means = grad_full.rigid.means[idx]
...
```

如果某 branch delta 是 None，对应 grad 也 None。

---

## 10. 投影实现

核心函数：

```python
def project_attr(delta_attr, grad_attr, *, strength, tau, eps):
    # delta_attr: [N, ...]
    # grad_attr:  [N, ...]
    d = delta_attr.reshape(N, -1)
    g = grad_attr.reshape(N, -1).detach()

    dot = (d * g).sum(dim=-1, keepdim=True)          # [N,1]
    norm2 = (g * g).sum(dim=-1, keepdim=True) + eps  # [N,1]
    violation = torch.relu(dot - tau)
    coeff = strength * violation / norm2
    d_safe = d - coeff * g
    return d_safe.reshape_as(delta_attr), aux
```

投影是 row-wise，每个 point 独立。

### 10.1 Damage loss

为了训练 updater 不再提出危险 delta，加入：

```python
damage = relu(dot / sqrt(norm2 + eps) - tau_norm)
loss_damage = mean(damage ** 2)
```

推荐用 normalized damage，而不是 raw dot，避免不同属性量纲差异太大。

```python
normalized_dot = dot / torch.sqrt(norm2 + eps)
```

第一版属性权重：

```yaml
attr_weights:
  means: 1.0
  scales: 0.5
  opacity: 0.7
  sh: 0.7
  quat: 0.0
```

### 10.2 Projection strength warmup

```python
strength = linear_warmup(global_step, start_step, warmup_steps, start_value, end_value)
```

第一版：

```text
0 -> 0.5，5000 step
```

不建议一开始直接 1.0。

---

## 11. 保护权重 protect_weight

不是所有历史都应强保护。历史本来 residual 高时，强投影会锁死错误。

使用 EMA history error 生成保护权重：

```python
protect = sigmoid((error_good_threshold - error_slow) / temp)
```

```text
history error 低 → protect 接近 1
history error 高 → protect 接近 0
```

投影系数：

```python
coeff = strength * protect * violation / norm2
```

第一版也可以关闭 protect_weight，统一 `protect=1`，但更推荐开启，因为它与当前 EMA history 设计匹配。

配置：

```yaml
protect_weight:
  enable: true
  source: history_ema_error_slow
  error_good_threshold:
    bg: 0.08
    distant: 0.08
    rigid: 0.10
  temp: 0.03
  min_weight: 0.0
  max_weight: 1.0
```

---

## 12. 与 history gate 的关系

顺序：

```text
posterior delta
→ branch scope / clamp
→ history gate
→ HSP projection
→ expand rigid
→ apply delta
```

原因：

```text
history gate 先决定允许写多少；
HSP 再决定允许沿哪个方向写。
```

不要替代 history gate。两者关系：

```text
history gate = scalar / attribute write magnitude control
HSP          = direction-level history safety control
```

---

## 13. 与 current/history/nearby final loss 的关系

HSP 不删除原 final losses。

最终 loss：

```python
loss = (
    current_weight * current_loss
  + history_weight * history_loss
  + nearby_weight * nearby_loss
  + delta_reg_weight * delta_reg
  + hsp_damage_weight * hsp_damage_loss
)
```

`hsp_damage_loss` 是额外 loss，仅来自 pre-projection delta 与 detached history grad 的冲突。

---

## 14. 需要修改的文件

### 14.1 `models/iforward/history_safe_projection.py`

新增完整模块。

包含：

```text
IForwardHistorySafeProjection
HistoryGradPack / HistoryGradBranch
make_probe_local_state
gather_probe_tensors
grad_state_to_grad_pack
grad_pack_to_event_rows
project_delta_pack
select_history_probe_indices
```

### 14.2 `models/iforward/model.py`

新增初始化：

```python
hsp_cfg = cfg_get(iforward_cfg, "history_safe_projection", {}) or {}
if self.is_v3_gru_history_gate and bool(cfg_get(hsp_cfg, "enable", False)):
    self.history_safe_projection = IForwardHistorySafeProjection(...)
else:
    self.history_safe_projection = None
```

新增 loss weight：

```python
self.hsp_damage_loss_weight = float(cfg_get(..., "damage_loss.weight", 0.0))
```

forward loop 中新增 block probe cache：

```python
block_hsp_cache = None
```

在 update 里插入：

```python
if self.history_safe_projection is not None:
    delta_safe_event, hsp_aux, hsp_loss = self.history_safe_projection(...)
    delta_gated_event = delta_safe_event
    hsp_losses.append(hsp_loss)
    update_aux.update(hsp_aux)
```

最终 loss 中加入：

```python
if hsp_losses:
    hsp_damage_loss = torch.stack(hsp_losses).mean()
else:
    hsp_damage_loss = local_state.bg.means.new_tensor(0.0)
loss = loss + self.hsp_damage_loss_weight * hsp_damage_loss
stats["hsp_damage_loss"] = float(hsp_damage_loss.detach())
```

### 14.3 `models/iforward/bridge.py`

可以复用现有：

```python
render_loss(local_state, batch, target_indices, mask_policy)
```

为了避免在 HSP 模块里访问 runtime 细节，可新增薄封装：

```python
def history_probe_loss(
    self,
    *,
    local_state: LocalGSState,
    batch: Dict[str, Any],
    target_indices: List[int],
    mask_policy: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    return self.render_loss(...)
```

可选：增加 `pred_rgbs_out=None, gt_images_out=None`，不要保存 probe 图片，减少显存。

### 14.4 `models/iforward/__init__.py`

导出：

```python
IForwardHistorySafeProjection
```

### 14.5 配置文件

新增：

```yaml
model:
  iforward:
    history_safe_projection:
      enable: true
      mode: damage_loss_only  # log_only / damage_loss_only / project_delta
      apply_after_history_gate: true

      probe:
        source: resolved_history_rollout_targets
        frequency: block_enter
        reuse_within_block: true
        frames_per_block: 1
        cams_per_frame: 1
        policy: deterministic_recent_or_hash
        fallback_policy: skip_if_no_history
        mask_policy: non_sky_non_egocar
        create_graph: false

      attrs:
        means: true
        scales: true
        opacity: true
        sh: true
        quat: false
        hidden: false

      projection:
        strength:
          start_step: 0
          warmup_steps: 5000
          start_value: 0.0
          end_value: 0.5
        eps: 1.0e-8
        tau_norm:
          means: 0.0
          scales: 0.0
          opacity: 0.0
          sh: 0.0

      protect_weight:
        enable: true
        source: history_ema_error_slow
        error_good_threshold:
          bg: 0.08
          distant: 0.08
          rigid: 0.10
        temp: 0.03
        min_weight: 0.0
        max_weight: 1.0

      damage_loss:
        enable: true
        weight: 0.05
        normalized: true
        attr_weights:
          means: 1.0
          scales: 0.5
          opacity: 0.7
          sh: 0.7
```

---

## 15. 日志指标

必须新增：

```text
iforward/hsp/enabled
iforward/hsp/probe_num_refs
iforward/hsp/probe_loss
iforward/hsp/probe_psnr
iforward/hsp/skipped_no_history

iforward/hsp/damage_loss
iforward/hsp/damage_pos_ratio/means
iforward/hsp/damage_pos_ratio/scales
iforward/hsp/damage_pos_ratio/opacity
iforward/hsp/damage_pos_ratio/sh

iforward/hsp/damage_norm_mean/means
iforward/hsp/damage_norm_mean/scales
iforward/hsp/damage_norm_mean/opacity
iforward/hsp/damage_norm_mean/sh

iforward/hsp/projection_strength
iforward/hsp/projection_norm_ratio/means
iforward/hsp/projection_norm_ratio/scales
iforward/hsp/projection_norm_ratio/opacity
iforward/hsp/projection_norm_ratio/sh

iforward/hsp/delta_norm_before/opacity
iforward/hsp/delta_norm_after/opacity
iforward/hsp/delta_norm_before/sh
iforward/hsp/delta_norm_after/sh
```

核心判据：

```text
如果 opacity/sh damage_pos_ratio 高，说明外观更新确实在破坏历史；
如果 projection_norm_ratio 高但 current PSNR 不大降，说明过去有很多有害 delta；
如果 history PSNR / nearby PSNR 提升且 current-history gap 缩小，说明 HSP 命中问题。
```

---

## 16. 数值与稳定性注意事项

1. `autograd.grad` 必须 `create_graph=False`。
2. `g_hist` 必须 detach。
3. probe render 不保存 pred image 到 tensorboard。
4. probe target 没有 history 时，HSP identity，damage loss 为 0。
5. projection 默认不处理 quaternion。
6. 对 dot/norm2 做 finite check；NaN 时 fallback identity 并记录 `hsp_skipped_nonfinite=1`。
7. strength warmup 必须有，第一版不要直接 strength=1。
8. HSP 可能增加显存；第一版必须 `frames_per_block=1, cams_per_frame=1, frequency=block_enter`。
9. damage loss 的 weight 从 0.05 开始，不要直接 0.2+。
10. 如果 current PSNR 下降超过 1 dB，而 history/nearby 不提升，应立即停实验。

---

## 17. 单元测试

### 17.1 投影数学测试

构造：

```python
g = [1,0]
d = [1,1]
```

projection 后：

```python
d_safe = [0,1]
```

如果：

```python
d = [-1,1]
```

dot<0，不应改变。

### 17.2 attr shape 测试

确保：

```text
bg/distant/rigid 的 means/scales/opacity/sh projection 后 shape 不变。
```

### 17.3 rigid route.S gather 测试

构造 rigid full rows N=10，event route.S=[2,5,8]，确认 grad_event.rigid 只 gather 对应行。

### 17.4 no-history fallback 测试

`resolved.history_rollout_target_indices=[]` 时：

```text
delta_safe == delta_input
loss_damage == 0
aux.skipped_no_history == 1
```

### 17.5 differentiability 测试

`delta_safe.sum().backward()` 后 posterior updater 参数应有梯度；history probe state 不应持有网络梯度。

---

## 18. 集成测试

1. 跑 2 个 batch，mode=log_only，确认不改变 loss 曲线、日志有 damage。
2. 跑 100 step，mode=damage_loss_only，确认没有 NaN，显存可控。
3. 跑 100 step，mode=project_delta，strength end=0.1，确认 update_norm 下降、history PSNR 不变差。
4. 与 baseline 同 seed 对比 1k step。

---

## 19. 与当前配置的关系

当前 experiment002 / experiment003 已经满足：

```text
scheduler_iforward.version = iforward_v3_random_window
history_replay.enable = true
current = all_rollout_input_frames
history = final_history_replay
nearby = final_nearby_rollout
history_memory.record_on = block_exit
update.record_on = every_repeat
```

因此 HSP 不需要重做 scheduler。只需要利用 `resolved.history_rollout_target_indices` 作为 probe 来源。

当前 20/15 资产显存更低，适合作为 HSP 初始实验；30/20 current 更好但显存更高，不适合先上 probe-gradient。

---

## 20. 最小实现路线

```text
P0.1 新增 history_safe_projection.py 的投影数学和 no-history fallback
P0.2 在 model.py 初始化 HSP 模块，插入 update path
P0.3 实现 probe target selection，只从 resolved.history_rollout_target_indices 取 1 frame × 1 cam
P0.4 实现 grad probe，仅对 means/scales/opacity/sh 求梯度
P0.5 实现 event-row grad mapping，尤其 rigid route.S
P0.6 加 damage loss 和日志
P1   开启 projection strength warmup
P2   如有效，再扩展 cams_per_frame=3 或 every_n_repeats
```

---

# 两个最有必要的训练实验

你资源有限，不建议跑太多 ablation。下面两个最关键。

## 实验 1：HSP-Damage-Only 诊断训练

目的：验证当前 delta 是否确实存在大量 history-harmful 方向，并训练 updater 主动减少这种方向。这个实验不投影 delta，风险低。

### 配置

基线：使用 20/15 资产的 experiment002 配置继续。原因是 20/15 显存低，HSP probe 会额外消耗显存；experiment002 已经使用 20/15、history replay、v3 random window、history warmup、较高 LR，是更合适的起点。

建议从已有 checkpoint resume：

```text
优先：40k 左右 checkpoint
次选：20k 左右 checkpoint
```

因为 40k 后开始出现 `r4b2`，能测试当前帧更新对前后 block/history 的影响。如果只能从 0 开始，至少跑到 10k，但诊断力度会弱。

新增配置：

```yaml
model:
  iforward:
    history_safe_projection:
      enable: true
      mode: damage_loss_only
      probe:
        frequency: block_enter
        reuse_within_block: true
        frames_per_block: 1
        cams_per_frame: 1
        policy: deterministic_recent_or_hash
        fallback_policy: skip_if_no_history
        create_graph: false
      attrs:
        means: true
        scales: true
        opacity: true
        sh: true
        quat: false
      damage_loss:
        enable: true
        weight: 0.05
        normalized: true
        attr_weights:
          means: 1.0
          scales: 0.5
          opacity: 0.7
          sh: 0.7
      projection:
        strength:
          start_value: 0.0
          end_value: 0.0
```

训练长度：

```text
5k steps 最小；10k steps 更好。
```

### 成功判据

继续推进到实验 2 的条件：

```text
1. hsp_damage_pos_ratio/opacity 或 /sh 明显 > 0.15；
2. damage_loss 随训练下降；
3. current PSNR 下降 < 0.3 dB；
4. history/nearby 不恶化，最好略升；
5. 没有明显显存爆炸，peak_mem 增幅可接受。
```

如果：

```text
damage_pos_ratio 接近 0
```

说明“当前 delta 方向破坏历史”不是主要问题，HSP 方向应暂停。

如果：

```text
damage_pos_ratio 很高，但 damage-only 无法降低
```

说明需要 projection，而不是只靠 loss。

---

## 实验 2：HSP-Soft-Projection 外观优先

目的：直接测试投影是否能改善历史/nearby 观感，尤其门、墙、道路这类 appearance/opacity 模糊。

### 配置

从实验 1 的最好 checkpoint resume，或者从同一个 40k baseline resume。

第一版只强投影 appearance 和弱投影 geometry：

```yaml
model:
  iforward:
    history_safe_projection:
      enable: true
      mode: project_delta
      probe:
        frequency: block_enter
        reuse_within_block: true
        frames_per_block: 1
        cams_per_frame: 1
        policy: deterministic_recent_or_hash
        fallback_policy: skip_if_no_history
        create_graph: false
      attrs:
        means: true
        scales: true
        opacity: true
        sh: true
        quat: false
      projection:
        strength:
          start_step: 0
          warmup_steps: 3000
          start_value: 0.0
          end_value: 0.5
        attr_strength_scale:
          means: 0.25
          scales: 0.5
          opacity: 1.0
          sh: 1.0
        tau_norm:
          means: 0.0
          scales: 0.0
          opacity: 0.0
          sh: 0.0
      damage_loss:
        enable: true
        weight: 0.05
        normalized: true
```

训练长度：

```text
10k steps 建议；最少 5k。
```

### 成功判据

认为 HSP 有效的条件：

```text
1. history PSNR +0.3 dB 以上，或 current-history gap 缩小 0.5 dB 以上；
2. nearby PSNR 不下降，最好提升；
3. current PSNR 下降 < 0.5 dB；
4. 图像上灰雾、门/墙 appearance 错误、拖影减少；
5. projection_norm_ratio/opacity 和 /sh 有实际数值，不是 0；
6. opacity/sh damage_pos_ratio 比实验开始下降。
```

立即停止的条件：

```text
1. current PSNR 下降 > 1 dB；
2. history/nearby 没有提升；
3. projection_norm_ratio 长期 > 0.7，说明 projection 过强；
4. 出现 geometry 锁死或明显空洞。
```

### 为什么实验 2 不直接投影 means=1.0？

因为你现在的图像问题更像 appearance/opacity/scale 的错误分配，而不是纯几何崩坏。全强度 means 投影容易导致当前帧修不动，先从 opacity/sh 方向验证更稳。

---

# 结论

如果只能做两个训练实验：

```text
实验 1：damage_loss_only，确认当前 delta 是否真的在一阶上破坏历史。
实验 2：soft projection，优先投影 opacity/sh，验证能否减少历史模糊且不明显牺牲 current。
```

不建议先跑：

```text
random point dropout
reserve Gaussian pool
dynamic densification
contribution gate
full-strength all-attr projection
```

这些都更重、更容易引入新的变量。v3.1 的第一目标应该是回答一个最直接的问题：

```text
当前 IForward 预测的 delta，是否沿着会增大 history loss 的方向写入？
如果是，把这个方向投影掉，图像是否更稳定、更少模糊？
```
