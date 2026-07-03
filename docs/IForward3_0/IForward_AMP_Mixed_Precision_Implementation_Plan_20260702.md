# IForward AMP / Mixed Precision 总体实现方案

日期：2026-07-02  
适用代码：`drivestudio_stage6_refactor_context_20260702_v44`  
适用主线：IForward 3_1 / 3_2，Low-rank Gated Delta KV，distributional episode scheduler  
目标：使用 AMP 降低显存并提升速度，同时保留几何、渲染、loss、BigGS parent state 等数值敏感路径的 FP32 稳定性。

---

## 0. 执行结论

原始状态是 `training.amp.enable=false`；当前实现已在 Stage3.1 / Stage3.2 主配置默认开启 AMP 与 Phase 5 bf16 storage。当前 3_1 / 3_2 模型已经是：

```text
stage3_1_lowrank_gated_delta_kv_lift
parent_optimizer_memory = lowrank_gated_delta_kv
parent.type = legacy_direct_lift
child_gather = support_center, num_taps=1
repair_training: repair 阶段 freeze_2d_frontend + no_grad_2d_forward
```

因此 AMP 应该分层接入，而不是一刀切：

```text
1. Global Trainer AMP：用 torch.amp.autocast 包住 forward/loss，用 GradScaler 支持 fp16。
2. FP32 islands：保留几何投影、gsplat raster、scalar anchor、BigGS sufficient stats、loss reduction、grad clipping 等关键路径。
3. AMP-safe zones：2D frontend、parent 2D feature lifting、parent spatial/PTv3、GDKV MLP、child decoder MLP、posterior updater MLP。
4. Parent 2D feature lifting：允许低精度，因为 parent 本来是粗略估计；child detail / render / geometry 不跟随放宽。
5. 分阶段启用：先 bf16 autocast + FP32 state；稳定后再考虑 fp16 + scaler 或 GDKV state/cache bf16。
```

推荐默认版：

```yaml
training:
  amp:
    enable: true
    dtype: auto       # bf16 if supported else fp16
    grad_scaler: auto # fp16 true, bf16 false
    render_fp32: true
    geometry_fp32: true
    loss_fp32: true
    parent_lift_amp: true
    child_gather_amp: false
    storage:
      features_2d_cache_dtype: bf16
      parent_context_cache_dtype: bf16
    gdkv_state_dtype: bf16
    child_detail_output_dtype: bf16
```

核心态度：**AMP 优先用于 activation-heavy 的神经网络路径和 parent coarse feature 路径；几何、render、loss、BigGS parent stats、sparse gather kernel 和 GRLD child decoder 继续保持 FP32。**

---

## 1. 背景与动机

IForward 3_2 引入 distributional episode 后，K / rollout length / repair high-block 都可能增大，显存和速度压力会明显上升。AMP 的收益主要来自：

```text
2D residual U-Net / DINO feature 后处理 activations
parent spatial/PTv3
GDKV read/write projections
BigGS child decoder / posterior updater MLP
parent / child feature tensor cache 的可选低精度存储
```

但 IForward 不适合全局强制 half：

```text
3DGS geometry, covariance, conic, depth, alpha/transmittance 对数值敏感；
Stage3 cuda_scalar_anchor 当前要求 float32；
Stage3 CUDA sparse gather 当前要求 float32 feature_map / uv / weights；
BigGS parent sufficient stats 配置就是 float32；
loss/SSIM/mask/PSNR/reduction 应保持 float32；
optimizer step 与 master weights 仍应 FP32。
```

PyTorch AMP 官方推荐的典型训练方式是用 `torch.autocast` 包住 forward / loss，并用 `torch.amp.GradScaler` 处理 fp16 梯度缩放；bf16 通常不需要 GradScaler，因为指数范围更大。文档实现应使用新的 `torch.amp.autocast("cuda", ...)` 和 `torch.amp.GradScaler("cuda", ...)` API，而不是旧的 `torch.cuda.amp.*`。

---

## 2. 当前代码现状

### 2.1 Trainer 现状

`models/iforward/trainer.py` 当前训练流程是：

```text
resolve batch
state cache lookup
optimizer.zero_grad
model.forward_rollout
loss.backward
clip_grad_norm_
optimizer.step
state cache update
metrics logging
```

AMP 尚未接入：

```text
没有 torch.autocast 包围 forward_rollout；
没有 GradScaler；
grad clipping 在 backward 后直接执行；
training.amp.enable 当前为 false。
```

因此 Trainer 是 AMP P0 改动点。

### 2.2 Stage3 / BigGS 现状

当前 IForward 3_1 / 3_2 主线：

```text
parent.type = legacy_direct_lift
child_gather = support_center, num_taps=1
parent_optimizer_memory = lowrank_gated_delta_kv
BigGS parent_state.stats_dtype = float32
BigGS child_cache_dtype = float32
```

这意味着：

```text
parent 2D lifting 可以降低精度；
child fine detail 与 render/loss 暂时不建议降低精度；
BigGS parent stats 必须继续 FP32；
GDKV 计算可 autocast，但 state 第一版建议 FP32。
```

### 2.3 当前硬约束

当前代码中存在若干 dtype 强约束：

```text
Stage3 cuda_scalar_anchor requires float32 means2d / conics / opacities / depths。
Stage3 CUDA sparse gather requires CUDA float32 feature_map。
Stage3 CUDA sparse gather requires float32 uv and weights。
```

因此 AMP 不能简单地让 autocast 自动覆盖全部 Stage3。必须显式设置 FP32 islands，并在必要处 cast。

---

## 3. AMP 精度分区策略

### 3.1 Zone A：必须 FP32

以下模块第一版必须保持 FP32：

| 模块 | 原因 |
|---|---|
| 3DGS means/scales/quats/opacities/SH master state | 累积更新、几何参数、optimizer master weights 需稳定 |
| camera intrinsics/extrinsics/projection math | 小误差会放大到 2D uv / conic / alpha |
| gsplat rasterization / alpha compositing / depth | 直接影响 render loss 与可见性 |
| Stage3 scalar_anchor CUDA | 当前 kernel 要求 float32 |
| Stage3 uv / weights / valid / depths / conics | anchor/gather 索引和权重不应低精度 |
| BigGS parent sufficient stats | 当前配置 stats_dtype=float32，累积统计应稳定 |
| parent projector / exact refresh / drift check | 几何统计与验证路径 |
| loss reductions：RGB/SSIM/mask/history_damage/delta_reg | 防止小 loss、mask 权重、SSIM 数值抖动 |
| grad norm / grad clipping | GradScaler 下必须先 unscale，再 clip |
| optimizer step | AdamW state 与 master params 保持 FP32 |

### 3.2 Zone B：建议 autocast

以下模块可以进入 autocast：

| 模块 | 建议 dtype | 说明 |
|---|---|---|
| residual U-Net / 2D frontend trainable部分 | bf16/fp16 | activation-heavy，收益大 |
| DINO adapter / fusion neck 若启用 | bf16/fp16 | DINO backbone冻结，cache 已经支持 float16 |
| parent 2D feature lifting | bf16/fp16 | parent 是粗略估计，用户明确不要求高精度 |
| parent token builder | bf16/fp16 | MLP/LayerNorm 可 autocast，必要 reduce fp32 |
| parent PTv3 / sparse conv / xCPE | bf16/fp16 | 需实际 smoke；spconv 支持情况需检测 |
| GDKV projections / read adapter | bf16/fp16 | state 第一版 FP32，投影计算可混精 |
| BigGS child decoder MLP / GRLD | bf16 优先 | relation normalize/reduction 可局部 fp32 |
| posterior updater MLP | bf16/fp16 | 输出 delta 后 clamp 前可转 fp32 |
| metrics-only norm / aux stats | FP32 on detach | 不影响训练图，可以 detach.float() |

### 3.3 Zone C：可选低精度存储

以下不是第一版默认，但可以作为 Phase 2：

```text
features_2d/context_2d cache dtype = bf16/fp16
detail_2d cache dtype = bf16 only, 默认先不动
GDKV state dtype = bf16
GDKV ctx dtype = bf16
parent event cache dtype = bf16
```

建议顺序：

```text
先 activation autocast；
再 parent feature cache bf16；
最后才考虑 GDKV state bf16。
```

---

## 4. 全局 AMP 配置设计

新增/扩展：

```yaml
training:
  amp:
    enable: true
    dtype: auto          # auto / bf16 / fp16 / fp32
    autocast_device: cuda
    grad_scaler: auto    # auto / true / false
    init_scale: 65536.0
    growth_factor: 2.0
    backoff_factor: 0.5
    growth_interval: 2000
    cache_enabled: true

    fp32_islands:
      geometry: true
      render: true
      scalar_anchor: true
      sparse_gather_cuda: true
      parent_state_stats: true
      loss: true
      grad_clip: true

    stage3:
      parent_lift_amp: true
      parent_lift_dtype: amp   # amp / bf16 / fp16 / fp32
      child_gather_amp: false
      child_detail_output_dtype: bf16
      scalar_anchor_force_fp32: true
      cuda_sparse_gather_force_fp32_kernel: true

    storage:
      features_2d_cache_dtype: bf16
      parent_context_cache_dtype: bf16

    memory:
      gdkv_compute_amp: true
      gdkv_state_dtype: bf16
      gdkv_aux_stats_fp32: true

    render:
      force_fp32: true
      loss_force_fp32: true

    debug:
      log_dtype_summary: true
      log_scaler: true
      log_amp_memory: true
      finite_check_interval: 1000
      compare_fp32_interval: 0
```

### 4.1 dtype 选择

```python
def resolve_amp_dtype(cfg):
    if not torch.cuda.is_available():
        return None
    name = str(cfg.training.amp.dtype).lower()
    if name == 'auto':
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp16', 'float16', '16'}:
        return torch.float16
    return None
```

建议默认：

```text
A100/H100/支持 bf16 的卡：bf16 mixed；
不支持 bf16 的卡：fp16 mixed + GradScaler；
debug/validation 对照：fp32。
```

---

## 5. Trainer AMP 实现

### 5.1 新增 AmpPolicy

新增文件：

```text
models/iforward/amp_policy.py
```

核心类：

```python
@dataclass
class AmpPolicy:
    enabled: bool
    dtype: Optional[torch.dtype]
    use_grad_scaler: bool
    cache_enabled: bool = True

    def autocast(self):
        if not self.enabled or self.dtype is None or not torch.cuda.is_available():
            return nullcontext()
        return torch.amp.autocast('cuda', dtype=self.dtype, enabled=True, cache_enabled=self.cache_enabled)

    def fp32(self):
        if not torch.cuda.is_available():
            return nullcontext()
        return torch.amp.autocast('cuda', enabled=False)
```

### 5.2 Trainer 初始化

`IForwardTrainer.__init__`：

```python
self.amp_policy = build_amp_policy(config)
self.grad_scaler = torch.amp.GradScaler(
    'cuda',
    enabled=bool(self.amp_policy.enabled and self.amp_policy.use_grad_scaler),
    init_scale=..., growth_factor=..., backoff_factor=..., growth_interval=...,
)
```

bf16：

```text
use_grad_scaler = false
```

fp16：

```text
use_grad_scaler = true
```

### 5.3 train_step 改法

当前：

```python
out = self.model.forward_rollout(...)
loss = out.loss
loss.backward()
clip_grad_norm_(...)
optimizer.step()
```

改为：

```python
with self.amp_policy.autocast():
    out = self.model.forward_rollout(batch, carried_state=carried, ablation=ablation)
    loss = out.loss

if self.grad_scaler.is_enabled():
    self.grad_scaler.scale(loss).backward()
    self.grad_scaler.unscale_(self.optimizer)
    grad_norm = clip_grad_norm_(...)
    self.grad_scaler.step(self.optimizer)
    self.grad_scaler.update()
else:
    loss.backward()
    grad_norm = clip_grad_norm_(...)
    self.optimizer.step()

self.optimizer.zero_grad(set_to_none=True)
```

重要顺序：

```text
fp16 scaler.scale(loss).backward
scaler.unscale_(optimizer)
finite grad norm check
clip_grad_norm_
scaler.step(optimizer)
scaler.update()
```

不要在未 unscale 的梯度上做 grad clipping。

### 5.4 skipped step 处理

fp16 下 GradScaler 可能因为 Inf/NaN 跳过 optimizer step。IForward 有 carried state，必须处理：

```text
如果 optimizer step 被 scaler 跳过：
  不应该把 out.next_state 写入 state_cache；
  应该清理或保持旧 carried state；
  metrics 记录 amp/optimizer_step_skipped=1。
```

否则会出现：

```text
模型参数没更新，但 runtime state 前进了
```

这是 IForward 特有问题。

建议实现：

```python
scale_before = scaler.get_scale()
scaler.step(optimizer)
scaler.update()
step_skipped = scaler.get_scale() < scale_before

if step_skipped:
    do_not_cache_next_state()
else:
    cache_next_state_normally()
```

bf16 无 scaler 时不需要。

---

## 6. Model / Runtime AMP 策略

### 6.1 forward_rollout 总原则

Trainer 的 autocast 包住整个 forward，但模型内部要对 FP32 islands 显式关闭 autocast：

```python
with amp_policy.fp32():
    scalar_anchor = run_scalar_anchor_fp32(...)
```

由于 `IForwardModel` 不一定直接持有 trainer，可将 amp config 放入 model：

```python
self.amp_policy = build_amp_policy(config, inference_only=True)
```

或者用轻量 helper：

```python
def amp_fp32_context():
    return torch.amp.autocast('cuda', enabled=False) if torch.cuda.is_available() else nullcontext()
```

### 6.2 LocalGSState / Gaussian params

保持 FP32 master state：

```text
local_state.bg.means/scales/quats/opacities/sh: FP32
local_state.distant: FP32
local_state.rigid: FP32
posterior delta 输出可 autocast，但 apply 前 cast FP32
```

推荐：

```python
delta = delta.to(dtype=torch.float32)
local_state = local_state.apply_delta(delta)
```

### 6.3 Render / loss

第一版 render 全部 FP32：

```python
with amp_policy.fp32():
    pred = render(local_state.float(), cameras.float())
    losses = compute_losses(pred.float(), gt.float(), masks.float())
```

虽然这会降低 AMP 收益，但避免 render/loss 指标漂移。后续可单独做 render autocast ablation。

---

## 7. Stage3 / 2D lifting AMP 策略

### 7.1 2D frontend

推荐放入 autocast：

```python
with amp_policy.autocast():
    features_2d, detail_2d = image_feature_extractor(...)
```

输出策略：

```text
features_2d/context_2d: AMP dtype allowed
parent_lift input: AMP dtype allowed
child_gather input: first version cast to FP32 kernel path
```

DINO cache 当前已经有 `dtype: float16` 配置，说明低精度 2D cache 已经是项目内已有方向。

### 7.2 scalar anchor

必须 FP32：

```python
with amp_policy.fp32():
    anchor = scalar_anchor(
        means2d.float(), conics.float(), opacities.float(), depths.float(), ...
    )
```

原因：当前 CUDA scalar anchor kernel 要求 float32。

### 7.3 parent 2D feature lifting

这是本方案的重点优化对象。

用户判断：parent 本来就是粗略估计，因此 parent 的 2D feature lifting 不需要高精度。

策略：

```text
parent feature map / parent context 可以用 AMP dtype；
parent lift output 可以是 bf16/fp16；
parent spatial / token builder 可继续 autocast；
必要时在进入 FP32-only kernel 前局部 cast。
```

当前主线 parent 是：

```text
parent.type = legacy_direct_lift
feature_source = features_2d
```

因此第一版实现：

```python
if amp.parent_lift_amp:
    context_2d_parent = context_2d.to(amp_dtype)
    parent_context = legacy_parent_lift(context_2d_parent, ...)
    parent_context = parent_context.to(amp_dtype)
else:
    parent_context = legacy_parent_lift(context_2d.float(), ...)
```

如果未来切到 `parent.type=sparse_gather`，当前 CUDA sparse gather 要求 float32。两种选择：

```text
A. 为 parent sparse gather 使用 PyTorch/AMP backend，允许 half/bf16 feature_map；
B. CUDA kernel 输入仍 float32，但 parent gather output 立刻 cast 到 amp dtype。
```

第一版推荐 A 作为 parent-only 可选路径，因为 parent 不需要高精度；child 不走这个路径。

### 7.4 child gather

gather 计算默认不 AMP，输出 storage 可 bf16：

```yaml
child_gather_amp: false
child_detail_output_dtype: bf16
```

原因：

```text
child detail 直接影响 fine Gaussian update；
当前 CUDA sparse gather 要求 float32 feature_map / uv / weights；
child support_center 单点 gather 本来已经是低显存路径，不应优先牺牲精度。
```

可选 ablation：

```text
child detail input bf16，kernel 内部/accumulate FP32，output bf16/fp32 ablation。
```

---

## 8. BigGS / Parent State AMP 策略

### 8.1 BigGS assignment / parent_projector

保持 FP32：

```text
assignment topology int/long;
parent projector geometry FP32;
mass/opacity/tau/scale stats FP32。
```

### 8.2 parent_state incremental sufficient stats

保持 FP32：

```yaml
parent_state:
  stats_dtype: float32
  child_cache_dtype: float32
```

不建议第一版改成 bf16。原因：

```text
incremental sufficient stats 是跨 repeat/rollout 累积；
一旦低精度累积偏移，会影响 parent event 和后续 memory。
```

### 8.3 parent spatial / PTV3

可以 autocast：

```python
with amp_policy.autocast():
    parent_event = parent_spatial(parent_context, parent_params, support, ...)
```

其中 support / valid / stats 可 FP32 输入，但 MLP/attention 输出可 AMP。

---

## 9. GDKV AMP 策略

当前 GDKV：

```text
K=16,V=32
query/key RMS unit
value_rms_max=2
ctx_rms_max=4
state_rms_max=4
```

推荐默认版：

```yaml
memory:
  gdkv_compute_amp: true
  gdkv_state_dtype: bf16
  gdkv_aux_stats_fp32: true
```

实现：

```python
with amp_policy.autocast():
    q/k/v/gates = projections(token)

# state update either fp32 or controlled dtype
S = state.kv_state.float()
q = q.float() if rms sensitive else q
k = k.float() if rms sensitive else k
v = v.to(dtype=amp_dtype)
S_new = gated_delta_update(S, k, v, gates)
S_new = rms_clamp(S_new.float()).to(state_dtype)
```

理由：

```text
projection MLP 受益于 AMP；
state update / RMS clamp / stats 仍以 FP32 计算；
只在写回 kv_state storage 时 cast 到 gdkv_state_dtype。
```

回退 ablation：

```yaml
gdkv_state_dtype: fp32
```

但必须增加：

```text
full vs bf16-state validation；
gdkv_state_rms_max / ctx_rms_max 对照；
memory ablation gain 对照。
```

---

## 10. Posterior updater / delta AMP 策略

posterior updater MLP 可以 autocast，但输出 delta 应在 clamp/apply 前转 FP32：

```python
with amp_policy.autocast():
    delta_raw = posterior_updater(event, child_detail, ...)

with amp_policy.fp32():
    delta = clamp_delta(delta_raw.float())
    local_state = apply_delta(local_state.float(), delta)
```

branch clamps 保持 FP32，尤其 distant scale 已经打开：

```yaml
branch_clamps:
  distant:
    means_max_step_m: 0.08
    scales_log_max_step: 0.04
    quat_axis_angle_max_step_rad: 0.0
```

---

## 11. Validation / Demo AMP

Validation 也应支持 AMP，但默认：

```text
autocast on;
no GradScaler;
render/loss/probe FP32；
metrics FP32；
image dump FP32 clamp 后 CPU。
```

新增 CLI：

```bash
python -m tools.iforward_validate_v4 ... \
  training.amp.enable=true \
  training.amp.dtype=bf16
```

Demo report 应记录：

```text
amp/enabled
amp/dtype
amp/render_fp32
amp/parent_lift_amp
amp/child_gather_amp
```

---

## 12. Metrics 与 Debug

新增 metrics：

```text
amp/enabled
amp/dtype_id                # fp32=0, fp16=1, bf16=2
amp/grad_scaler_enabled
amp/grad_scale
amp/grad_scale_growth_tracker
amp/optimizer_step_skipped
amp/autocast_forward_enabled
amp/render_fp32
amp/geometry_fp32
amp/parent_lift_amp
amp/child_gather_amp
amp/gdkv_state_dtype_id
```

dtype summary：

```text
amp/dtype/features_2d
amp/dtype/detail_2d
amp/dtype/parent_context
amp/dtype/parent_event
amp/dtype/gdkv_state
amp/dtype/pred_rgb
amp/dtype/loss
```

memory summary：

```text
perf/cuda/after_forward_allocated_mb
perf/cuda/after_backward_allocated_mb
perf/cuda/peak_allocated_mb
amp/memory_saving_vs_fp32_mb
```

finite checks：

```text
amp/nonfinite_loss_count
amp/nonfinite_grad_count
amp/nonfinite_pred_count
amp/nonfinite_state_count
```

---

## 13. 测试计划

### 13.1 Unit tests

```text
test_amp_policy_resolve_dtype
test_trainer_amp_fp16_scaler_unscale_before_clip
test_trainer_amp_step_skip_does_not_cache_next_state
test_stage3_scalar_anchor_force_fp32_under_autocast
test_cuda_sparse_gather_force_fp32_kernel_under_autocast
test_parent_lift_amp_output_dtype
test_loss_force_fp32
```

### 13.2 Smoke tests

#### A. 200-step bf16 smoke

```bash
python -m tools.train_iforward \
  --config_file configs/iforward/iforward_stage3_2_distributional_episode_gdkv.yaml \
  training.max_iterations=200 \
  training.amp.enable=true \
  training.amp.dtype=bf16 \
  logging.log_dir=/root/autodl-tmp/outputs/amp_bf16_smoke
```

通过标准：

```text
无 NaN/Inf；
loss 正常下降；
GDKV read/write 正常；
state cache 正常；
peak memory 低于 FP32 baseline。
```

#### B. 200-step fp16 smoke

```bash
training.amp.dtype=fp16
training.amp.grad_scaler=true
```

通过标准：

```text
scaler 正常更新；
没有长期 step_skipped；
grad clip 在 unscale 后执行。
```

#### C. fixed-plan FP32 vs AMP 对照

用同一个 EpisodePlan：

```text
FP32 run
AMP run
```

比较：

```text
loss_total delta
current_psnr delta
history_psnr delta
repair_mean delta
GDKV ctx/state rms delta
```

建议阈值：

```text
bf16: PSNR delta < 0.10-0.20 dB
fp16: PSNR delta < 0.20-0.35 dB
```

### 13.3 Long smoke

```text
2k step bf16
5k step bf16
```

检查：

```text
AMP memory gain；
throughput gain；
validation metrics drift；
parent_lift_amp 是否改变 current/history。
```

---

## 14. 分阶段落地路线

### Phase 0：配置与 AmpPolicy

输出：

```text
models/iforward/amp_policy.py
training.amp config parser
metrics dtype id helper
```

不改 forward。

### Phase 1：Trainer AMP

改：

```text
IForwardTrainer.__init__
IForwardTrainer.train_step
```

支持：

```text
autocast forward/loss
GradScaler fp16
unscale before grad clip
skip step 不写 state cache
AMP metrics
```

### Phase 2：FP32 islands

给以下路径包 `autocast(enabled=False)`：

```text
scalar_anchor
cuda sparse gather kernel wrapper
render/loss
BigGS parent projector/stats update
loss reductions
```

### Phase 3：Parent lift AMP

启用：

```text
parent_lift_amp=true
parent_lift_dtype=amp
```

确保：

```text
parent_context dtype = amp dtype
parent_event 可 autocast
scalar_anchor 仍 FP32
child_gather 仍 FP32
```

### Phase 4：GDKV / decoder / updater autocast 精细化

确保：

```text
GDKV projection autocast
GDKV state FP32
child decoder MLP autocast
posterior updater MLP autocast
apply_delta FP32
```

### Phase 5：Optional storage AMP

Stage3.1 / Stage3.2 主配置默认开启：

```text
features_2d cache bf16
parent_context cache bf16
GDKV state bf16
child detail bf16 ablation
```

计算敏感路径继续保持 FP32：geometry/render/loss/scalar anchor/sparse gather kernel/BigGS parent stats/GRLD child decoder。

当前 runner 使用单 `--config_file` + trailing `opts`，所以 ablation 直接用 dotlist 覆盖：

```bash
# all_fp32
training.amp.storage.features_2d_cache_dtype=fp32 \
training.amp.storage.parent_context_cache_dtype=fp32 \
training.amp.memory.gdkv_state_dtype=fp32 \
training.amp.stage3.child_detail_output_dtype=fp32

# features_only
training.amp.storage.features_2d_cache_dtype=bf16 \
training.amp.storage.parent_context_cache_dtype=fp32 \
training.amp.memory.gdkv_state_dtype=fp32 \
training.amp.stage3.child_detail_output_dtype=fp32

# parent_context_only
training.amp.storage.features_2d_cache_dtype=fp32 \
training.amp.storage.parent_context_cache_dtype=bf16 \
training.amp.memory.gdkv_state_dtype=fp32 \
training.amp.stage3.child_detail_output_dtype=fp32

# gdkv_state_only
training.amp.storage.features_2d_cache_dtype=fp32 \
training.amp.storage.parent_context_cache_dtype=fp32 \
training.amp.memory.gdkv_state_dtype=bf16 \
training.amp.stage3.child_detail_output_dtype=fp32

# child_detail_only
training.amp.storage.features_2d_cache_dtype=fp32 \
training.amp.storage.parent_context_cache_dtype=fp32 \
training.amp.memory.gdkv_state_dtype=fp32 \
training.amp.stage3.child_detail_output_dtype=bf16

# all_bf16
training.amp.storage.features_2d_cache_dtype=bf16 \
training.amp.storage.parent_context_cache_dtype=bf16 \
training.amp.memory.gdkv_state_dtype=bf16 \
training.amp.stage3.child_detail_output_dtype=bf16
```

---

## 15. 风险与处理

### 风险 1：fp16 underflow / overflow

处理：

```text
默认 bf16；
fp16 必须 GradScaler；
loss/grad_norm/clip FP32；
step skip 不更新 carried state。
```

### 风险 2：CUDA kernel dtype assert

处理：

```text
scalar_anchor_force_fp32=true；
cuda_sparse_gather_force_fp32_kernel=true；
对输入显式 float()。
```

### 风险 3：AMP 破坏 render metrics

处理：

```text
render_fp32=true；
loss_force_fp32=true；
validation fixed-plan 对照。
```

### 风险 4：parent lift AMP 改变模型行为

处理：

```text
只 parent_lift_amp，不 child_gather_amp；
对比 parent_context_rms / parent_event_norm / validation；
如影响过大，改为 parent_lift_dtype=bf16 或 fp32。
```

### 风险 5：GDKV state 低精度导致 memory 质量下降

处理：

```text
默认 gdkv_state_dtype=bf16；
如数值风险偏高，用 all_fp32 或 gdkv_state_only 反向 ablation 回退。
```

---

## 16. 推荐默认配置

```yaml
training:
  amp:
    enable: true
    dtype: auto
    grad_scaler: auto
    cache_enabled: true
    fp32_islands:
      geometry: true
      render: true
      scalar_anchor: true
      sparse_gather_cuda: true
      parent_state_stats: true
      loss: true
      grad_clip: true
    stage3:
      parent_lift_amp: true
      parent_lift_dtype: amp
      child_gather_amp: false
      child_detail_output_dtype: bf16
      scalar_anchor_force_fp32: true
      cuda_sparse_gather_force_fp32_kernel: true
    storage:
      features_2d_cache_dtype: bf16
      parent_context_cache_dtype: bf16
    memory:
      gdkv_compute_amp: true
      gdkv_state_dtype: bf16
      gdkv_aux_stats_fp32: true
    render:
      force_fp32: true
      loss_force_fp32: true
    debug:
      log_dtype_summary: true
      log_scaler: true
      log_amp_memory: true
      finite_check_interval: 1000
```

默认不推荐：

```yaml
child_gather_amp: true
render.force_fp32: false
geometry_fp32: false
```

---

## 17. 后续 GPT 上下文块

```text
IForward AMP 目标：降低显存/提升速度，但保持几何和 render 稳定。
当前 Stage3.1/Stage3.2 主配置默认 training.amp.enable=true，需要保持 trainer-level torch.amp.autocast + fp16 GradScaler 路径可回退。
默认 dtype=auto：bf16 if supported else fp16。bf16 不用 GradScaler，fp16 用 GradScaler。
Trainer 必须在 scaler.unscale_(optimizer) 后再 grad clip；如果 scaler skip optimizer step，则不能把 out.next_state 写入 state_cache。
FP32 islands：gsplat render/projection, Stage3 scalar_anchor, CUDA sparse gather kernel, BigGS parent stats, loss reductions, grad clip, optimizer step。
AMP zones：2D frontend, parent 2D feature lifting, parent spatial/PTv3, GDKV projections, child decoder MLP, posterior updater MLP。
用户明确认为 parent 2D feature lifting 不需要高精度，因为 parent 是粗估计，因此 parent_lift_amp=true 是第一优先优化点。
child_gather 计算保持 FP32，因为 child detail 直接影响 fine update，且 CUDA sparse gather 当前要求 float32 feature_map/uv/weights；child detail 输出 storage 默认 bf16。
GDKV compute 可 AMP，但 read/write/update/RMS/stats 仍 FP32；kv_state 写回 storage 默认 bf16，可用 all_fp32 dotlist 回退。
render/loss 强制 FP32，保证 validation/PSNR 可信。
```
