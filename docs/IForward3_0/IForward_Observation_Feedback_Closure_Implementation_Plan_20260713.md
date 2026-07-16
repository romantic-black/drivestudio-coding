# IForward 观测反馈闭环 / Hybrid Recompute VJP 完整实现方案

日期：2026-07-13  
适用代码：`drivestudio_stage6_refactor_context_20260712_v52`  
相关审计：`IForward_Stage3_2_Scheduler_Rollout_Gradient_Propagation_Audit_20260713.md`  
目标：在保持 Stage 3.2 高 block、长 rollout、低显存训练能力的前提下，恢复 **单 rollout 内** 的 `LocalGS → 后续观测/BigGS → event/GDKV → child update → LocalGS` 连续反馈链。

---

## 0. 执行结论

Codex 方案的核心判断正确：

```text
冻结 2D 参数 ≠ 禁止对 2D 输入求梯度；
persistent BigGS runtime 不保图 ≠ 后向不能重算连续量 VJP；
连续动力学可以可微，离散拓扑与路由继续 stop-gradient。
```

但原方案必须做五项关键修正：

1. **当前 source render 本身就在 `torch.no_grad()` 内，而且 RGB/alpha 显式 `.detach()`。** 只把 `detach_source_render_for_cnn=false` 不会恢复反馈；必须新增真正可微的 source render 路径。
2. **只 checkpoint residual CNN 不够。** 如果 source render 可微，而 render graph 每个 K 都保存，K=15 显存仍可能显著上升。应 checkpoint 整个动态观测分支：
   `LocalGS → source render → residual CNN → fusion/detail`。
3. **runtime parent 的“精确反向”严格说是代理 VJP。** 前向使用 incremental runtime 参数，后向用 exact diagonal projector 重算。除非 runtime 前向与 exact projector 数值一致，否则它不是 runtime 函数本身的精确导数。必须记录 runtime-vs-exact drift，并设置 refresh/skip 策略。
4. **仅恢复 parent projector VJP 仍不够。** `Stage6ParentParamSupportCodec` 当前默认 `detach_params=true`，会在 parent 参数进入 parent event 前再次截断梯度。
5. **GRLD 对 bg/distant 的 covariance relation 仍可能走 detached cache。** 打开 relation detach 前，必须给 bg/distant child params 补充由 `scales_log + quats` 现场计算的 differentiable `diag_cov`。

推荐分三阶段实现：

```text
Phase A：可微 source-render / 2D observation feedback，保留 anchor geometry detach；
Phase B：runtime parent forward + exact recompute surrogate VJP，并打开 parent param codec；
Phase C：GRLD relation/child-param feedback，补 differentiable diag_cov。
```

第一轮只解决 **单 rollout 内闭环**。继续保留：

```text
rollout boundary detach；
GDKV state 在 rollout boundary detach；
离散 assignment / sorting / index / valid mask / threshold detach；
DINO 静态特征 no-grad/cache；
scalar-anchor geometry detach。
```

---

## 1. Codex 方案正确性审查

| Codex 主张 | 审查结论 | 必要修正 |
|---|---|---|
| 冻结 2D 参数与输入梯度解耦 | 正确 | 不能用整个 2D forward 的 `torch.no_grad()`；需在 forward+backward 生命周期内冻结参数，但保留 input autograd |
| source render 不 detach | 正确但不完整 | 当前 renderer 自身 `no_grad + detach`；必须新增 differentiable renderer API |
| residual CNN checkpoint | 方向正确但范围不足 | checkpoint 整个 dynamic observation branch，而不只是 CNN |
| persistent parent runtime 前向，backward exact VJP | 可行 | 属于 surrogate VJP；必须 drift/parity guard，不能无条件称“精确 runtime backward” |
| 连续量开放、离散量 detach | 正确 | 第一版只开 source-render continuous path；parent/relation 分阶段 |
| parent param codec / GRLD relation 解除 detach | 正确 | parent codec 与 relation 必须分别控制；bg/distant `diag_cov` 需改成可微现场计算 |
| scalar anchor geometry gradient | 暂不需要 | 第一、二阶段保持 detach，避免 UV/support/valid 离散边界引入噪声 |
| gradient bridge alpha 渐进开启 | 正确 | render、parent-VJP、relation 使用独立 alpha，不能共用一个开关 |
| 维持 maxK=15 | 可行但需验证 | activation 可由 checkpoint 控制，时间会增加；必须做 K=15 峰值显存与 wall-time 实测 |

总体评价：

```text
方向正确，可实施；
原方案约 70%-80% 正确；
真正工程风险集中在 source render checkpoint 边界、runtime surrogate VJP 一致性、
以及 relation covariance 仍被 detached cache 截断。
```

---

## 2. 当前 child → parent 实现

### 2.1 Fine LocalGS / child state

LocalGS 中 bg、distant、rigid child 保存真实 fine Gaussian 状态：

```text
means
scales_log
quats
opacity_logit
sh_dc
sh_rest
```

posterior updater 每个 inner update 输出 delta，并写回新的 LocalGS。

### 2.2 离散 child-to-parent assignment

当前 BigGS 使用 branch-aware voxel/cap assignment：

```text
child_to_parent
child_order
parent_start
parent_count
```

这些是离散拓扑，应继续 stop-gradient。模型不需要对 voxel assignment、排序和索引求导。

### 2.3 Child continuous params → parent sufficient statistics

exact diagonal projector 使用 child continuous parameters 计算：

```text
mass = tau(opacity) × projected area(scales) 或 static assignment mass
parent_mean = Σ mass_i * mean_i / Σ mass_i
parent_diag_cov = Σ mass_i * (child_diag_cov_i + spatial_offset_i²) / Σ mass_i
parent_scale = sqrt(parent_diag_cov)
parent_opacity = 1 - exp(-tau_parent)
parent_SH = mass-weighted child SH
```

因此在连续量上，下列 child 参数都能影响 parent：

```text
means
scales_log
quats
opacity_logit
sh_dc
sh_rest
```

### 2.4 当前 incremental parent runtime

为降低每个 inner step 的开销，当前实现维护 incremental sufficient-stat runtime：

```text
weight_sum
weighted mean/second moment
opacity/tau stats
SH sums
parent param cache
```

当前 runtime 初始化/更新主要在 `no_grad` 下执行；`projection_from_runtime()` 返回 detached clone。因此 runtime 是高效数值状态，但不是 autograd graph。

### 2.5 Parent 2D lifting 与 parent event

parent params 与 2D context/support 进入 parent spatial backbone：

```text
parent params + support/context
  → Stage6ParentParamSupportCodec
  → parent token / PTV3 / spatial backbone
  → parent event
  → GDKV preview/read/write
```

当前 `Stage6ParentParamSupportCodec(detach_params=True, detach_support=True)`，所以 parent continuous params 即使由可微 projector 产生，也会在这里再次截断。

---

## 3. 当前 parent → child 实现

### 3.1 Parent event 广播到 child

parent event 经 `child_to_parent` index/broadcast 到对应 fine child。索引本身离散，保持 detach；但 parent event tensor 对 decoder/updater 保持可微。

### 3.2 GRLD relation feature

当前 Gaussian relation 主要包括：

```text
r_xyz      = (child_mean - parent_mean) / parent_scale
r_cov      = log(child_diag_cov) - log(parent_diag_cov)
r_mass     = log(child_mass) - log(parent_mean_mass)
r_opacity  = child_opacity - parent_opacity
r_sh       = child/parent SH difference or energy difference
```

relation 与 parent event 共同产生 child residual/fine event，并使用 mean-preserve 约束减少 parent group 内平均漂移。

### 3.3 当前 GRLD 梯度断点

当前配置：

```yaml
detach_relation_inputs: true
detach_child_code_inputs: true
detach_child_params: true
detach_parent_params: true
```

此外，bg/distant `_branch_params()` 没有放入 differentiable `diag_cov`；relation codec 会回退使用 `child_cache.diag_cov`，该 cache 是 detached。于是即使把 `detach_relation_inputs=false`，bg/distant scale/quat 的 relation feedback 仍然不完整。

### 3.4 Child detail 与 posterior update

child 还从 2D detail map 进行 support-center gather，然后与 child event 一起进入 posterior updater：

```text
child event
+ child 2D detail
+ branch info
→ posterior updater
→ Gaussian delta
→ next LocalGS
```

当前 child detail gather 的 feature backward 可以保留，但 geometry/UV/support 路由第一阶段继续 detach。

---

## 4. 当前完整 stop-gradient 地图

| 链路 | 当前断点 | 结果 |
|---|---|---|
| LocalGS → source render | `render_rgb_only()` 整体 `torch.no_grad()`；RGB/alpha `.detach()` | future observation loss 无法回到 earlier LocalGS |
| source render → CNN | `detach_source_render_for_cnn=true` | 即使 renderer 可微仍会再次断链 |
| repair 2D branch | 整个 `_render_source_scene_only_for_cnn()` 在 `torch.no_grad()` 中 | 冻结参数与输入梯度被错误绑定 |
| LocalGS → observe NodeStates | `to_node_states_detached_view()` | BigGS observe 看不到 LocalGS 的连续 Jacobian |
| LocalGS → parent runtime | runtime init/update `no_grad`；runtime projection detached | parent event 不对 child Gaussian params 回传 |
| parent params → parent event | parent codec `detach_params=true` | parent projector VJP 即使存在也被截断 |
| scalar anchor geometry | `detach_geometry=true` | UV/support/采样位置不对 geometry 求导 |
| child/parent params → GRLD relation | 多个 detach flag | child update 无法经 relation 回到 LocalGS continuous params |
| bg/distant covariance relation | detached `child_cache.diag_cov` | scale/quat relation feedback缺失 |
| rollout boundary | `next_state.detach_for_next_rollout()` | 跨 rollout activation credit 被截断；本方案暂不修改 |

结论：

```text
当前不是一个 detach 点，而是多层 detach 串联；
必须按反馈链逐层恢复，不能只改一个 config flag。
```

---

## 5. 目标反馈闭环

目标单 rollout 内闭环：

```text
LocalGS_t
  ├─ differentiable source render ─→ source RGB/residual
  │                                   ↓
  │                     checkpointed residual CNN/fusion
  │                                   ↓
  │                        2D context / detail features
  │                                   ↓
  ├─ child→parent runtime forward ─→ parent context/event
  │        backward exact recompute surrogate VJP ↑
  │                                   ↓
  │                            GDKV memory/event
  │                                   ↓
  └──────────── GRLD relation ← parent/child continuous params
                                      ↓
                                 child event
                                      ↓
                              posterior updater
                                      ↓
                                  LocalGS_t+1
                                      ↓
                         later observation / final loss
```

基本边界：

```text
连续量可微：
  means/scales/quats/opacity/SH
  source-render RGB
  residual CNN input
  2D gathered feature values
  parent continuous params
  relation features
  event/GDKV/updater/delta

离散量继续 detach：
  assignment / child_to_parent
  sorting / parent_start / parent_count
  valid masks / support threshold decisions
  index / topology
  DINO cache and static image feature extraction
  history bank baselines
  rollout/episode carry boundary
```

---

## 6. 新反馈策略配置

新增：

```yaml
model:
  iforward:
    observation_feedback:
      enable: true
      scope: within_rollout

      modes:
        repeat_refine: trainable_checkpointed
        shuffled_coverage: trainable_checkpointed
        high_block_repair: frozen_input_grad_checkpointed

      source_render:
        enable: true
        renderer_mode: differentiable_rgb
        checkpoint_scope: full_dynamic_observation
        absgrad: false
        alpha_schedule:
          - [0, 0.0]
          - [1000, 0.10]
          - [3000, 0.25]
          - [8000, 0.50]
          - [15000, 1.0]

      parent_projection:
        enable: false
        branches: [bg, distant]
        forward_mode: incremental_runtime
        backward_mode: exact_diag_recompute_surrogate_vjp
        alpha_schedule:
          - [0, 0.0]
          - [3000, 0.05]
          - [8000, 0.15]
          - [15000, 0.30]
        drift:
          check_interval: 500
          warn_threshold: 1.0e-3
          skip_vjp_threshold: 5.0e-3
          exact_refresh_threshold: 1.0e-2

      relation:
        enable: false
        branches: [bg, distant]
        differentiable_diag_cov: true
        checkpoint: true
        alpha_schedule:
          - [0, 0.0]
          - [3000, 0.05]
          - [8000, 0.15]
          - [15000, 0.30]

      scalar_anchor:
        geometry_grad: false
      discrete_routing_grad: false
      rollout_boundary_grad: false

      debug:
        grad_probe_interval: 500
        forward_parity_interval: 1000
        log_feedback_memory: true
```

推荐第一轮只打开 `source_render.enable=true`；parent/relation 仍 false。

---

## 7. 2D 三模式设计

不要再只有 `trainable / frozen_no_grad` 两类。

| 模式 | 2D 参数梯度 | source-render 输入梯度 | activation 策略 | 用途 |
|---|---:|---:|---|---|
| `trainable_checkpointed` | 有 | 有 | checkpoint full dynamic branch | repeat/shuffle，训练 2D 同时保反馈 |
| `frozen_input_grad_checkpointed` | 无 | 有 | checkpoint full dynamic branch | high-block repair，推荐 |
| `frozen_no_grad` | 无 | 无 | no-grad | validation / 极限显存 fallback |

关键原则：

```text
frozen_input_grad_checkpointed:
  residual CNN / fusion / detail 参数不产生 grad；
  但其输入 source-render RGB 保持 requires_grad；
  backward 重算 CNN，从而把梯度传回 LocalGS。
```

### 7.1 参数冻结生命周期

推荐由 `IForwardTrainer.train_step()` 管理整个 forward+backward 生命周期：

```python
with model.frontend_parameter_mode(mode):
    out = model.forward_rollout(...)
    backward(out.loss)
    optimizer_step()
```

`frozen_input_grad_checkpointed` 时：

```text
在 forward 前把 residual frontend/fusion/detail 参数 requires_grad=False；
保持到 backward/checkpoint recompute 完成；
optimizer step 后恢复原 requires_grad；
输入仍可求梯度。
```

不能在 model forward 返回前就恢复，否则 checkpoint backward 重算时参数状态会改变。

若未来使用 DDP static graph，不允许动态切 `requires_grad`，则使用 `torch.func.functional_call` + detached parameter mapping 作为替代实现。

---

## 8. Phase A：可微 source-render + checkpointed dynamic observation

### 8.1 新 differentiable renderer API

修改：

```text
models/feature_extractors/alpha_t_extractor.py
```

新增：

```python
def render_rgb_feedback(
    gaussians,
    cameras,
    height,
    width,
    *,
    return_acc=False,
    viewmats_override=None,
):
    # 不使用 torch.no_grad()
    # 不对 RGB 调用 detach()
    # absgrad=False，避免无用 densification absgrad state
    # camera tensors/constants 不求梯度
    # 只保留 RGB graph；alpha 若只用于 mask/debug 可 detach
```

或者把原函数改为显式模式：

```python
render_rgb_only(..., grad_mode='none' | 'gaussian_params')
```

禁止隐式根据全局 grad 状态猜测。

当前 `_extract_rgb()` 也会 `.detach()`，feedback 模式不能复用该 detach 版本。

### 8.2 为什么 checkpoint 必须包含 source render

只 checkpoint CNN 时，gsplat source render 的中间图仍会为每个 inner K 保留。高 block K=15 时，显存可能再次线性增长。

推荐 checkpoint 闭包：

```text
输入：
  LocalGS continuous tensors
  static source images
  cached DINO features
  camera/view matrices

闭包：
  fine-scene tensor assembly
  differentiable gsplat RGB render
  residual U-Net
  DINO/residual fusion
  detail head

输出：
  features_2d
  fwhr_detail_2d
  optional stage3_dino_native_2d
```

使用：

```python
torch.utils.checkpoint.checkpoint(
    dynamic_observation_fn,
    *tensor_inputs,
    use_reentrant=False,
    preserve_rng_state=False,  # 前提是分支无 dropout/随机 op
)
```

PyTorch 官方建议显式使用 `use_reentrant=False`；该模式对嵌套结构、无 input-grad 的首步等情况更兼容。

### 8.3 Checkpoint 闭包必须纯函数

不能在 closure 内做：

```text
DINO cache insert/lookup mutation
perf accumulator 累加
TraceRecorder 写入
`.item()` 同步日志
全局 step 变化
随机数据采样
runtime state mutation
```

这些操作必须在 closure 外完成。否则 backward 重算会重复 side effect 或产生 forward/recompute 不一致。

### 8.4 DINO 与静态特征

DINO/static image features 继续：

```text
no-grad
cache
在 checkpoint 外计算或读取
作为常量 tensor 输入 closure
```

只有 dynamic residual branch 对 LocalGS 反馈。

### 8.5 Gradient scaling

source render 输出后：

```python
def scale_feedback(x, alpha):
    return x.detach() + alpha * (x - x.detach())
```

forward 数值恒等于原 `x`；backward 乘 alpha。

不要通过缩小 forward feature 数值实现渐进开启，否则会同时改变模型输入分布。

---

## 9. Phase B：Runtime parent forward + exact recompute surrogate VJP

### 9.1 为什么不直接让 runtime 持图

incremental runtime 是为高 K 设计的。若每次 parent sufficient stats 都保留完整 autograd graph：

```text
显存随 K 增长；
state cache 变成 graph-carry；
parent stats scatter/index graph 大；
repair K=15 可能失去现有优势。
```

因此前向继续使用 detached persistent runtime。

### 9.2 新自定义 autograd bridge

新增：

```text
models/iforward/runtime_parent_projection_vjp.py
```

核心：

```python
class RuntimeParentProjectionVJPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *runtime_parent_outputs, *child_params, assignment, config, alpha):
        # forward 返回当前 incremental runtime 数值，完全不改变现有前向
        # 保存 child continuous tensors + static assignment
        return tuple(x.detach() for x in runtime_parent_outputs)

    @staticmethod
    def backward(ctx, *grad_parent_outputs):
        # 重新创建 requires_grad child params
        # 调用 project_biggs_parent_diag_reference_tensors / CUDA recompute-autograd path
        # 用 grad_parent_outputs 做 VJP
        # 返回 alpha * child grads
```

现有 `_BigGSParentProjectDiagFn` 已经证明：CUDA exact forward 可以在 backward 中用 reference projector 重算 VJP。本方案复用同一 reference 公式。

### 9.3 这是 surrogate VJP，不应伪装为 exact runtime gradient

forward：

```text
incremental runtime projection
```

backward：

```text
exact diagonal projection derivative
```

只有当两者 forward 数值足够接近时，代理梯度才可信。

必须记录：

```text
parent_vjp/bg/means_rel_error
parent_vjp/bg/scales_rel_error
parent_vjp/bg/opacity_rel_error
parent_vjp/bg/sh_rel_error
parent_vjp/distant/*
```

策略：

```text
error < warn_threshold：正常使用 VJP；
warn < error < skip_threshold：使用但告警/降低 alpha；
error > skip_threshold：本次 parent VJP 置零；
error > refresh_threshold：触发 exact runtime refresh。
```

### 9.4 Parent codec 必须同时打开

修改：

```text
models/iforward/parent_spatial_backbone.py
```

把：

```python
Stage6ParentParamSupportCodec(detach_params=True, detach_support=True)
```

改为配置驱动：

```python
detach_params = not feedback_policy.parent_projection.enable
detach_support = True
```

第一版 support 仍 detach；只让 continuous parent params 进入 parent event gradient。

### 9.5 Branch 范围

先做：

```text
bg
distant
```

rigid 涉及 canonical/world transform、active rigid routing，放到后续单独实现。

---

## 10. Phase C：GRLD relation feedback

### 10.1 独立控制四类 detach

不要一次把所有 flag 变 false。新增细粒度 policy：

```yaml
relation:
  grad_to_child_geometry: true
  grad_to_parent_geometry: true
  grad_to_child_code: false
  grad_to_parent_event: true
  grad_to_support: false
```

初始只开：

```text
child/parent continuous geometry relation
```

child code input 可后续单独 ablation。

### 10.2 bg/distant differentiable diag_cov

修改：

```text
models/iforward/biggs_event_decoder.py
```

对 bg/distant `_branch_params()` 补：

```python
child_params['diag_cov'] = _diag_cov_from_scales_quats(
    child_params['scales_log'],
    child_params['quats'],
)
```

否则 relation codec 会继续使用 detached `child_cache.diag_cov`，scale/quat feedback 仍然断开。

### 10.3 Relation feedback scaling

在 relation 输入处：

```python
child_param_fb = scale_feedback(child_param, alpha_relation)
parent_param_fb = scale_feedback(parent_param, alpha_relation)
```

parent event 主链本来可微，不应额外 detach。

### 10.4 Checkpoint relation/decode

若 GRLD relation+decode activation 在 K=15 明显增加，可对每个 branch checkpoint：

```text
relation build
relation normalization
low-rank child residual decode
```

但需确认 fused CUDA decoder backward 可重复执行且 closure 无 side effect。

---

## 11. Scalar anchor / sparse gather 策略

第一、二、三阶段均保持：

```yaml
scalar_anchor.detach_geometry: true
```

允许：

```text
feature value backward
source-render RGB backward
parent continuous param backward
relation continuous param backward
```

继续禁止：

```text
UV location gradient
support threshold/valid mask gradient
assignment/index gradient
```

理由：

```text
source render 已经提供 means/scales/quats/opacity/SH 的几何反馈；
不需要同时引入 projected support/UV 的不稳定离散边界。
```

只有前三阶段效果不足时，才单独评估 anchor geometry gradient。

---

## 12. Trainer / lifecycle 改造

修改：

```text
models/iforward/trainer.py
models/iforward/model.py
models/streetforward/minimal_trainer_stage6_0.py
```

### 12.1 训练步骤生命周期

```python
feedback_mode = resolve_feedback_mode(batch.rollout_plan)

with model.feedback_runtime_scope(feedback_mode):
    out = model.forward_rollout(...)
    loss.backward()
    grad_stats = ...
    optimizer.step()
```

scope 覆盖 forward 和 backward，确保 checkpoint 重算时参数 freeze 状态不改变。

### 12.2 State 边界

本方案不改：

```text
optimizer.step 后 next_state.detach_for_next_rollout()
```

也不把 runtime graph 写入 persistent state。所有反馈只在当前 rollout autograd graph 内存在。

### 12.3 AMP

建议：

```text
source gsplat render：FP32 第一版；
residual CNN/fusion：继续 AMP；
parent exact VJP：FP32；
GRLD fused CUDA：按当前 FP32 island；
loss/delta apply：FP32。
```

checkpoint backward 重算必须使用与 forward 相同的 autocast context。使用 non-reentrant checkpoint `context_fn` 或在 closure 内显式进入同一 AMP policy。

---

## 13. 代码改动清单

### 13.1 新文件

```text
models/iforward/observation_feedback.py
  ObservationFeedbackPolicy
  FeedbackMode
  FeedbackAlphaSchedule
  scale_feedback
  FrontendParameterModeScope

models/iforward/runtime_parent_projection_vjp.py
  RuntimeParentProjectionVJPFn
  runtime_exact_drift
  parent_projection_feedback
```

### 13.2 修改文件

```text
models/feature_extractors/alpha_t_extractor.py
  新增 differentiable source render API

models/streetforward/minimal_trainer_stage4_5.py
  将 source render / residual / DINO / fusion 拆成纯 dynamic branch
  删除 feedback 模式下的 scene_rgb detach

models/streetforward/minimal_trainer_stage6_0.py
  新三种 2D mode
  checkpoint full dynamic observation branch
  feedback policy 路由

models/iforward/biggs_parent_stats.py
  runtime forward 输出接 surrogate VJP bridge
  drift/exact refresh API

models/iforward/parent_spatial_backbone.py
  param codec detach 配置化

models/iforward/biggs_event_decoder.py
  bg/distant differentiable diag_cov
  relation feedback policy

models/iforward/biggs_relational_decoder.py
  relation detach 改成 policy/alpha

models/iforward/trainer.py
  frontend freeze scope 跨 forward/backward
  feedback grad metrics

configs/iforward/iforward_stage3_3_observation_feedback.yaml
  新 feedback 配置
```

---

## 14. 必须新增的指标

### 14.1 反馈开关与 alpha

```text
iforward/feedback/mode_id
iforward/feedback/render_enabled
iforward/feedback/render_alpha
iforward/feedback/parent_vjp_enabled
iforward/feedback/parent_vjp_alpha
iforward/feedback/relation_enabled
iforward/feedback/relation_alpha
```

### 14.2 梯度可达性

低频 probe：

```text
feedback/grad/source_rgb
feedback/grad/local_before_observe/means
feedback/grad/local_before_observe/scales
feedback/grad/local_before_observe/opacity
feedback/grad/parent_runtime_output
feedback/grad/relation_child_params
feedback/grad/relation_parent_params
```

### 14.3 2D 冻结真实性

```text
feedback/2d_param_grad_count
feedback/2d_param_grad_norm
feedback/source_render_input_grad_norm
```

期望：

```text
frozen_input_grad_checkpointed：
  2d_param_grad_count = 0
  source_render_input_grad_norm > 0
  earlier LocalGS grad > 0
```

### 14.4 Parent VJP 一致性

```text
feedback/parent_vjp/runtime_exact_rel_error/*
feedback/parent_vjp/skipped
feedback/parent_vjp/exact_refresh
feedback/parent_vjp/grad_norm/*
```

### 14.5 成本

```text
feedback/source_render_forward_ms
feedback/source_render_recompute_ms
feedback/cnn_recompute_ms
feedback/parent_vjp_recompute_ms
feedback/relation_recompute_ms
feedback/peak_allocated_mb
feedback/step_time_ms
```

---

## 15. 单元测试与梯度测试

### 15.1 Forward parity

```text
feedback enable=true, alpha=0
```

必须满足：

```text
所有 forward tensor 与 baseline 一致；
loss / PSNR / event / next LocalGS 一致；
仅 metadata/metrics 可不同。
```

### 15.2 Source render gradient

构造两 inner steps：

```text
LocalGS0 -> update1 -> LocalGS1 -> observe2 -> final loss
```

断言：

```text
baseline：update1 delta 不从 observe2 获得梯度；
render feedback：update1 delta 获得非零梯度。
```

### 15.3 Frozen input-grad test

```text
2D parameters grad is None/zero；
source RGB grad nonzero；
LocalGS means/scales/opacity grad nonzero。
```

### 15.4 Checkpoint parity

对同一小 batch：

```text
eager differentiable observation
checkpointed differentiable observation
```

比较 forward 和 gradients。

### 15.5 Parent surrogate VJP

小规模 child/parent：

```text
runtime forward == exact forward 情况下
surrogate VJP 与 exact projector VJP 对齐。
```

并运行 finite-difference / gradcheck reference 测试。

### 15.6 Drift guard

人工扰动 runtime cache：

```text
small drift -> VJP active
medium drift -> alpha reduced/warn
large drift -> skip VJP/exact refresh
```

### 15.7 GRLD diag_cov test

断言 bg/distant：

```text
relation loss 对 scales_log / quats 有非零梯度；
关闭 relation feedback 时梯度为零。
```

### 15.8 K=15 memory test

四种模式对比：

```text
baseline frozen_no_grad
render feedback eager
render feedback checkpointed
render + parent VJP checkpointed
```

检查 peak memory 与 step time。

---

## 16. 消融设计

### 主消融

| 组别 | Source-render feedback | Parent surrogate VJP | Relation feedback | 目的 |
|---|---:|---:|---:|---|
| A | 否 | 否 | 否 | 当前基线 |
| B | 是 | 否 | 否 | 判断 LocalGS→render→CNN 反馈贡献 |
| C | 否 | 是 | 否 | 判断 child→parent continuous feedback贡献 |
| D | 是 | 是 | 否 | 完整 observation/parent 闭环 |
| E | 是 | 是 | 是 | 完整 continuous relation 闭环 |

### 每组必须看

```text
repeat K 增加时 current/history 曲线
shuffled coverage current/history
seq10/seq20 assimilation
seq10/seq20 repair/order
memory full/off/shuffle-state gap
K=15 peak memory / step time
```

特别关注此前异常：

```text
repeat K 越大 current 越高、history 越低。
```

成功闭环后，目标不是让 current 下降，而是让 history 随 K 的下降幅度明显缩小。

---

## 17. 推荐实施顺序

### Milestone 0：基线与 probe

先固定当前 checkpoint 和 batch，加入无行为改变的 gradient probe。

通过条件：

```text
能稳定复现 source-render/parent/relation 三类梯度为零；
能复现 same-rollout update 主链仍可微。
```

### Milestone 1：Differentiable source render（不 checkpoint，小 K）

仅 K=2/4 smoke：

```text
去 no_grad / detach；
alpha_render=0/0.1；
确认梯度到 earlier LocalGS。
```

### Milestone 2：Full dynamic observation checkpoint

目标：

```text
K=15 可运行；
2D frozen params无梯度；
LocalGS feedback梯度存在；
显存接近 baseline frozen-no-grad。
```

### Milestone 3：Parent surrogate VJP

先 bg/distant，alpha 小，drift guard 开启。

### Milestone 4：Parent codec + relation feedback

补 differentiable diag_cov，逐步打开 relation。

### Milestone 5：长训

只在 A/B/C/D 小实验明确收益后，进入 5k/10k 对照训练。

---

## 18. 预期成本与可行性

### 显存

```text
checkpoint full dynamic observation：
  不保存 source-render/CNN activation；
  主要额外保存 LocalGS input tensor references 与 checkpoint metadata；
  预计能保留 K=15 量级能力。
```

### 时间

```text
backward 时重跑 source render + residual CNN/fusion；
高 block repair step 时间一定上升；
这是用计算换显存与正确梯度。
```

建议接受标准：

```text
B 组 K=15 peak allocated 不超过 baseline frozen-no-grad 的 1.15 倍；
B 组 wall-time 增长控制在可接受范围并由实测决定；
若 wall-time过高，可设置 feedback_horizon=4/8 作为 fallback，
但主实验先尝试 full within-rollout closure。
```

### Parent VJP

parent exact reference recompute 相比 source render/CNN 通常更轻，但 scatter/SH 维度仍有成本；按 branch 和 interval profile。

---

## 19. 风险与防护

### 风险 1：长链梯度爆炸/消失

防护：

```text
独立 alpha schedule；
现有 grad clip；
记录 earlier-update grad by distance；
必要时 feedback_horizon。
```

### 风险 2：Checkpoint forward/recompute 不一致

防护：

```text
pure closure；
DINO/cache/logging移到外部；
AMP context一致；
无 dropout时 preserve_rng_state=false。
```

### 风险 3：Surrogate parent VJP 偏差

防护：

```text
drift check；
skip/refresh；
与 forward_exact_checkpointed reference 对照。
```

### 风险 4：2D 参数意外获得梯度

防护：

```text
trainer-scope freeze；
2d_param_grad_count assertion；
optimizer group grad audit。
```

### 风险 5：打开 relation 后 scale/quat 仍无梯度

防护：

```text
强制 bg/distant child_params 包含 differentiable diag_cov；
独立单元测试 scales/quats grad。
```

### 风险 6：前向质量变化

所有 bridge 使用 forward-identity：

```text
alpha=0 或 alpha>0 均不改变 forward value；
只改变 backward Jacobian。
```

---

## 20. 验收标准

### 正确性

```text
[ ] alpha=0 forward 与 baseline 对齐
[ ] frozen 2D 参数 grad=0
[ ] source-render input 与 earlier LocalGS grad>0
[ ] parent VJP drift 在阈值内
[ ] relation scales/quats grad probe 通过
[ ] discrete topology/index grad 始终关闭
[ ] rollout boundary 仍 detach
```

### 稳定性

```text
[ ] K=15 无 NaN/Inf
[ ] optimizer step skip=0
[ ] grad clip 不长期饱和
[ ] GDKV/BigGS drift 不恶化
```

### 性能

```text
[ ] checkpointed 模式能维持 K=15
[ ] peak memory 满足项目预算
[ ] source/CNN recompute 时间被单独量化
```

### 模型效果

至少观察：

```text
repeat K 增加时 history 下降显著减缓；
shuffled coverage current/history 提升；
seq20 repair/order 改善；
不以牺牲 repeat current 为代价。
```

---

## 21. 最终推荐

最优第一步不是一次性打开所有 detach，而是：

```text
1. 新增 differentiable source renderer；
2. 将 LocalGS→render→residual CNN→fusion/detail 作为完整 checkpoint 单元；
3. high-block repair 改为 frozen_input_grad_checkpointed；
4. 保持 scalar anchor geometry、parent runtime、GRLD relation 暂时 detach；
5. 先验证 B 组是否改善 history/shuffle；
6. 再依次增加 parent surrogate VJP 和 relation feedback。
```

这一方案直接针对当前最有解释力的现象：

```text
同 block repeat 能优化 current；
跨 block future observation 无法把 credit 传回 earlier LocalGS；
history 随 repeat K 增加而下降。
```

它在保留高 K 设计的同时，恢复真正需要的 continuous observation feedback，是当前最切中根因、同时风险可控的实现路线。

---

## 22. 参考资料

- PyTorch activation checkpointing：`torch.utils.checkpoint`，推荐显式 `use_reentrant=False`；checkpoint 通过 backward 重算 forward 片段换取显存。
- PyTorch custom autograd Function：用于实现 runtime-forward / recompute-backward VJP bridge。
- gsplat rasterization API：rasterizer 支持对 Gaussian 参数反向传播；相机内参目前不可微，本方案不需要相机梯度。
