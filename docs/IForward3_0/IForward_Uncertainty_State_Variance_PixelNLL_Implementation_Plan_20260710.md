# IForward Uncertainty V1 完整实现方案

**主题：Uncertainty-state delta prediction + Per-Gaussian appearance variance + Pixel heteroscedastic NLL**  
日期：2026-07-10  
适用代码：`drivestudio_stage6_refactor_context_20260710_v48`  
适用渲染器：`gsplat_source_20260703`  
适用模型：IForward Stage 3.2 / GDKV / BigGS parent-child / AMP

---

## 0. 执行结论

本方案只实现三个闭环组件：

```text
1. Per-Gaussian appearance uncertainty state
   每个 child Gaussian 持久保存一个 scalar log-variance。

2. Uncertainty-state delta prediction
   posterior updater 每次迭代预测 delta_logvar，并像 means/scales/SH 一样更新状态。

3. Pixel uncertainty rendering + heteroscedastic NLL
   用与 RGB 相同的 alpha/transmittance responsibility 将 Gaussian variance 渲染到图像空间，
   再用 Gaussian NLL 训练。
```

推荐第一版采用：

```text
correctness-first two-pass rendering
+ decoupled uncertainty gradients
+ raw L1/SSIM anchor
+ history_damage 不做 uncertainty attenuation
```

第一版不实现：

```text
不传播 means/scales/quat 参数 covariance；
不让 uncertainty 直接控制 GDKV write；
不让 uncertainty 直接缩放 Gaussian delta；
不做 parent centroid confidence filtering；
不替换 current/history/repair 主协议；
不移除原始 unweighted PSNR/SSIM/L1 评估。
```

核心原则：

> 先证明预测 uncertainty 与真实 residual / destructive update 有校准关系，再让 confidence 参与 memory 或 update gate。不能一开始就让模型通过提高 uncertainty 隐藏困难区域。

---

## 1. 概念与数学定义

### 1.1 3DGS 几何 covariance 与预测 uncertainty 不同

已有 3D Gaussian 空间 covariance：

\[
\Sigma_i^{\mathrm{geom}}
=
R_i\operatorname{diag}(s_i^2)R_i^\top
\]

表示 Gaussian 的空间形状、朝向与覆盖范围。

本方案新增 appearance / observation uncertainty：

\[
s_i = \log v_i^{\mathrm{app}},
\qquad
v_i^{\mathrm{app}}=\exp(s_i),
\]

表示这个 Gaussian 的 RGB appearance 对当前训练分布有多不可靠。两者不能混用。

### 1.2 Gaussian heteroscedastic NLL

像素残差：

\[
\mathbf r_p=\mathbf d_p-\mathbf d_p^*.
\]

若 RGB channel 使用 mean reduction，推荐：

\[
\mathcal L_{\mathrm{NLL},p}
=
\frac{1}{2}e^{-s_p}
\operatorname{mean}_c(r_{p,c}^2)
+
\frac{1}{2}s_p.
\]

这里：

```text
s_p = pixel log-variance
exp(-s_p) = residual precision / adaptive weight
0.5*s_p = 防止模型无限增大 variance
```

若使用 channel sum，则 log-variance 项应乘 RGB 维数 `D/2=3/2`；本方案用 channel mean，保持 loss scale 与当前 L1/SSIM 更接近。

### 1.3 Laplace NLL 仅作为后续 ablation

\[
\mathcal L_{\mathrm{Laplace},p}
=
\frac{\sqrt{2}}{\sigma_p}
\operatorname{mean}_c|r_{p,c}|
+
\log\sigma_p.
\]

第一版选择 Gaussian NLL，因为 per-Gaussian variance 可以通过二阶矩和全方差公式严格传播；Laplace scale 的 mixture 没有同样干净的闭式传播。

---

## 2. 总体数据流

```text
LocalGSState
  branch.appearance_logvar [N,1]
       |
       v
Stage6PosteriorUpdater
  head_uncertainty_delta -> delta_logvar [N,1]
       |
       v
LocalGSState.apply_delta
  logvar_next = clamp(logvar + delta_logvar)
       |
       v
Uncertainty rasterizer
  RGB mean pass
  detached second-moment / variance pass
       |
       v
pixel variance / logvar map
       |
       v
heteroscedastic Gaussian NLL
  + raw L1 anchor
  + SSIM anchor
```

---

## 3. Part A：Per-Gaussian Appearance Variance State

### 3.1 State 字段

每个 child Gaussian 增加一个标量：

```python
appearance_logvar: Tensor  # [N,1], FP32
```

分别存在：

```text
bg.appearance_logvar
distant.appearance_logvar
rigid.appearance_logvar
```

推荐 scalar 而不是 RGB 三通道：

```text
显存低；
校准更稳定；
避免 uncertainty head 通过通道间自由度作弊；
可以在 pixel 端使用 RGB channel mean NLL。
```

### 3.2 需要修改的 dataclass

#### `models/streetforward/node_states.py`

在以下 dataclass 末尾增加 optional field：

```python
appearance_logvar: Optional[torch.Tensor] = None
```

对象：

```text
NodeStateBackground
NodeStateDistant
NodeStateRigid
```

`NodeStateSky` 第一版不加；sky 不在当前 non-sky render loss 主路径内。

#### `models/streetforward/stage6_0/local_gs_state.py`

`LocalBranchState` 增加：

```python
appearance_logvar: torch.Tensor
```

以下方法都必须覆盖新字段：

```text
from_tensors
iter_tensors
to
_apply_branch
to_node_states_detached
to_node_states_grad
```

### 3.3 初始化

旧 asset / checkpoint 没有 uncertainty 字段时，按 branch 初始化：

```yaml
uncertainty:
  state:
    init_sigma:
      bg: 0.08
      distant: 0.12
      rigid: 0.10
```

\[
s_{0,b}=2\log\sigma_{0,b}.
\]

推荐：

```text
bg:      log(0.08^2)  ≈ -5.05
distant: log(0.12^2)  ≈ -4.24
rigid:   log(0.10^2)  ≈ -4.61
```

不建议初始 sigma 太小，否则刚开始 NLL precision 过高，错误监督会产生巨大梯度。

### 3.4 范围

```yaml
sigma_min: 0.01
sigma_max: 0.50
```

对应：

\[
s_{\min}=2\log(0.01)\approx-9.21,
\qquad
s_{\max}=2\log(0.5)\approx-1.386.
\]

State update 后始终：

```python
appearance_logvar = appearance_logvar.clamp(logvar_min, logvar_max)
```

### 3.5 持久化与兼容

需要同步修改所有显式重建 `LocalBranchState` / `NodeState*` 的位置：

```text
models/streetforward/stage6_0/local_gs_state.py
models/streetforward/minimal_trainer_stage6_0.py
models/iforward/history_safe_projection.py
models/iforward/adc_lite.py
```

以及：

```text
phase_b detach/freeze
TBPTT cache
state snapshot
validation state snapshot
checkpoint serialization
rigid local/world subset routing
```

旧 checkpoint load：

```text
若 appearance_logvar 缺失，不报 strict error；按 branch prior 初始化。
```

---

## 4. Part B：Uncertainty-State Delta Prediction

### 4.1 不复用现有 `BranchDelta.confidence`

当前 `BranchDelta` 已有：

```python
confidence: Tensor
noop: Tensor
```

但 `confidence` 是更新附属预测，当前没有进入持久 state；它不能直接当作 log-variance delta。

新增独立字段：

```python
appearance_logvar_delta: Tensor  # [N,1]
```

修改：

```text
BranchDelta
DeltaPack cast/expand/fill helpers
branch scope
optimizer write token optional summary
all BranchDelta constructors
```

### 4.2 Posterior updater head

在 `Stage6PosteriorUpdater` 增加：

```python
self.head_appearance_logvar_delta = nn.Linear(hidden_dim, 1)
```

初始化：

```python
nn.init.zeros_(head.weight)
nn.init.zeros_(head.bias)
```

确保加载旧 checkpoint 后第一步不会突然改变 variance。

### 4.3 预测公式

推荐第一版：

\[
\Delta s_i
=
\eta_s
\tanh(h_s(\operatorname{stopgrad}(h_i))).
\]

即：

```python
raw = self.head_appearance_logvar_delta(h.detach())
delta_logvar = max_logvar_step * torch.tanh(raw)
```

为什么默认 detach trunk input：

```text
uncertainty calibration loss 不应反向把主 event representation 训练成“预测自己会错”；
先让 uncertainty head 学 residual scale；
mean/update 主干仍由 raw RGB loss 和 NLL mean path 训练。
```

后续可 ablation：

```yaml
uncertainty.updater.detach_input: false
```

### 4.4 是否乘主 `noop` gate

推荐默认：

```yaml
gate_by_main_noop: false
```

理由：

```text
即使 geometry/appearance delta 是 no-op，新观测仍可能改变“这个 Gaussian 是否可靠”的估计；
uncertainty state 与 Gaussian update state 不完全同义。
```

但必须用有效性 mask：

```python
delta_logvar *= appearance_valid.float()
```

没有观测到的 Gaussian 不更新 uncertainty。

### 4.5 Delta clamp

```yaml
max_logvar_step:
  bg: 0.08
  distant: 0.10
  rigid: 0.10
```

每次 sigma 的乘法变化上界约：

\[
\sigma_{t+1}/\sigma_t
=
\exp(\Delta s/2).
\]

`delta s=0.10` 时，单次 sigma 只变化约 5.1%，适合 iterative updater。

### 4.6 可选 prior pull

长时间没有可靠观测时，过度自信可能持续。可加入很小 prior pull：

\[
s_{t+1}
=
\operatorname{clip}
\left[
 s_t+\Delta s_t
 +\lambda_{prior}(s_0-s_t)
\right].
\]

默认：

```yaml
prior_pull: 0.001
```

第一版也可设 0，先减少变量。

### 4.7 Branch scope

新增：

```yaml
branch_scope:
  bg:
    update_appearance_logvar: true
  distant:
    update_appearance_logvar: true
  rigid:
    update_appearance_logvar: true
```

`Stage6PosteriorUpdater._SCOPE_KEYS` 加：

```python
"appearance_logvar_delta": "update_appearance_logvar"
```

### 4.8 GDKV write token

第一版不把 uncertainty 强行加入 GDKV read/write gate，但可以把 detach summary 加到 write token：

```text
mean(logvar_current)
mean(delta_logvar)
mean(pixel_calibration_error) 可选
```

配置默认：

```yaml
write_token.include_uncertainty_summary: false
```

等 uncertainty 校准有效后再打开。

---

## 5. Part C：Per-Gaussian Variance Rasterization

### 5.1 目标

RGB mean 仍使用现有 3DGS render：

\[
\boldsymbol\mu_p=\sum_iw_{pi}\boldsymbol\mu_i.
\]

每个 Gaussian：

\[
\mathbf c_i
\sim
\mathcal N(\boldsymbol\mu_i,v_i I_3).
\]

总像素 scalar variance：

\[
v_p
=
\underbrace{\sum_iw_{pi}v_i}_{\text{aleatoric}}
+
\underbrace{\sum_iw_{pi}\operatorname{mean}_c(\mu_{i,c}^2)
-\operatorname{mean}_c(\mu_{p,c}^2)}_{\text{visible splat disagreement}}.
\]

这是 law of total variance 的 scalar RGB 平均形式。

### 5.2 gsplat 可行性

当前 gsplat `rasterization()` 在 `sh_degree=None` 时支持任意 D-channel feature，并提示 `D>32` 才明显变慢。本方案最多需要 2 或 5 channel，接口足够。

### 5.3 第一版：双 pass correctness-first

#### Pass 1：原 RGB render

保持现有：

```python
rgb, alpha = self._render_single_view(...)
```

所有 geometry/opacity/SH 梯度与当前实现一致。

#### Pass 2：uncertainty moment render

使用 detach 的：

```text
means
scales
quats
opacities
view-dependent RGB mean
```

只让 `appearance_logvar` 接收梯度。

对每个 view 手工评估 SH：

```python
rgb_i = spherical_harmonics(sh_degree, dirs, sh_coeffs)
rgb_i = torch.clamp_min(rgb_i + 0.5, 0.0)
```

构造 2-channel feature：

```python
var_i = appearance_logvar.exp()
m2_i = rgb_i.detach().square().mean(dim=-1, keepdim=True) + var_i
feature_i = torch.cat([m2_i, var_i], dim=-1)  # [N,2]
```

使用相同 view / geometry / opacity，调用：

```python
rasterization(
    means=means.detach(),
    quats=quats.detach(),
    scales=scales.detach(),
    opacities=opacities.detach(),
    colors=feature_i,
    sh_degree=None,
    ...
)
```

输出：

```text
moment2_render = rendered[...,0]
aleatoric_render = rendered[...,1]
```

像素 variance：

```python
rgb_energy = rgb.detach().square().mean(dim=-1)
var_total = (moment2_render - rgb_energy).clamp_min(var_floor)
var_total = var_total.clamp_max(var_max)
logvar_pixel = var_total.log()
```

这样：

```text
uncertainty calibration loss 只更新 appearance_logvar / uncertainty head；
不会通过 variance 通道改 geometry / opacity / SH；
主 RGB render 完全不变。
```

### 5.4 Alpha / background 处理

现有 render 无显式 background，天空被 mask；仍需处理低 alpha：

```python
var_pixel = var_total + (1.0 - alpha.clamp(0,1)) * background_var
```

推荐：

```yaml
background_sigma: 0.10
alpha_valid_min: 0.01
```

低 alpha 像素仍按现有 valid mask 处理；不要因为 alpha 低就直接丢失所有监督。

### 5.5 第二版：单 pass 5-channel 优化

双 pass 验证正确后，可替换为单 pass：

```text
channels 0:3 = RGB mean
channel 3    = mean(RGB_i^2) + variance_i
channel 4    = variance_i
```

手工评估 per-view SH，并调用一次 `rasterization(sh_degree=None)`：

```python
features = torch.cat([rgb_i, m2_i, var_i], dim=-1)  # [C,N,5]
```

输出直接得到 RGB / second moment / aleatoric variance。

启用前必须做 parity：

```text
single-pass RGB 与原 SH raster RGB PSNR > 60 dB；
max absolute RGB error < 1e-5 或设定的 CUDA 容差；
RGB gradients 与原路径一致。
```

第一版不建议直接上单 pass，避免把 renderer parity 与 uncertainty 机制混在一起。

### 5.6 RenderBundle

新增：

```python
@dataclass
class UncertaintyRenderBundle:
    rgb: Tensor           # [H,W,3]
    alpha: Tensor         # [H,W]
    variance: Tensor      # [H,W]
    logvar: Tensor        # [H,W]
    aleatoric_variance: Tensor  # [H,W]
    disagreement_variance: Tensor  # [H,W]
```

将 `_render_target`、`_render_targets_grouped_by_frame` 从 tuple 扩展为 bundle；兼容旧调用时可提供 `bundle.rgb, bundle.alpha`。

---

## 6. Pixel Heteroscedastic NLL

### 6.1 新 loss 文件

建议新增：

```text
models/iforward/uncertainty_losses.py
```

核心函数：

```python
def masked_gaussian_rgb_nll(
    pred_rgb,
    gt_rgb,
    pixel_logvar,
    mask,
    detach_weight=True,
    decoupled=True,
): ...
```

### 6.2 Decoupled NLL

推荐：

```python
resid2 = (pred_rgb - gt_rgb).square().mean(dim=-1)
precision = torch.exp(-pixel_logvar)

loss_mean = 0.5 * precision.detach() * resid2
loss_unc = 0.5 * precision * resid2.detach() + 0.5 * pixel_logvar
loss = loss_mean + uncertainty_calibration_weight * loss_unc
```

作用：

```text
loss_mean：uncertainty 作为固定权重训练 RGB/GS mean；
loss_unc：residual 作为固定目标校准 uncertainty；
避免 uncertainty 与 RGB 主干互相串通。
```

### 6.3 与原 loss 组合

保留 raw anchor：

\[
L_{rgb}
=
\lambda_{nll}L_{NLL}
+
\lambda_{l1}L_{L1,raw}
+
\lambda_{ssim}L_{DSSIM,raw}.
\]

推荐初始：

```yaml
nll_weight: 0.50
uncertainty_calibration_weight: 0.10
raw_l1_anchor_weight: 0.25
raw_ssim_anchor_weight: keep_existing
```

不要一开始完全替换原 masked RGB loss。

### 6.4 对不同 loss role 的应用

| role | NLL | raw anchor | uncertainty attenuation |
|---|---:|---:|---:|
| current | 是 | 是 | 允许 |
| in_rollout_history | 是 | 是 | 有 weight floor |
| repair photometric | 是 | 是 | 允许，但记录 before/after uncertainty |
| history_damage | 否 | 保持现有 relative raw loss | 禁止 |
| validation PSNR/SSIM | 否 | 使用 raw metric | 禁止 |

`history_damage` 不允许 uncertainty attenuation，因为模型不能通过提高 variance 逃避“当前 update 破坏历史”的约束。

### 6.5 History weight floor

如果对 history NLL 使用 uncertainty，增加 precision floor：

```python
weight = w_min + (1 - w_min) * normalized_precision.detach()
```

推荐：

```yaml
history_precision_floor: 0.30
repair_precision_floor: 0.30
```

第一版也可以 current/history/repair 全部只用 NLL + raw anchor，不额外 gate；floor 作为第二阶段。

---

## 7. 文件级修改清单

### 7.1 State / delta

```text
models/streetforward/node_states.py
models/streetforward/stage6_0/local_gs_state.py
models/streetforward/stage6_0/posterior_updater.py
models/streetforward/minimal_trainer_stage6_0.py
models/iforward/delta_ops.py
models/iforward/history_safe_projection.py
models/iforward/adc_lite.py
models/iforward/stage2_3/optimizer_write_token.py
```

### 7.2 Rendering

```text
models/streetforward/minimal_trainer_stage6_0.py
models/iforward/bridge.py
models/iforward/uncertainty_renderer.py      # 新增
```

### 7.3 Loss / config

```text
models/iforward/uncertainty_losses.py        # 新增
models/iforward/model.py
models/iforward/trainer.py
configs/iforward/iforward_stage3_3_uncertainty_v1.yaml
```

### 7.4 Validation / demo

```text
models/iforward/runtime/trace.py
models/iforward/validation_v4/metrics.py
models/iforward/validation_v4/html_exporter.py
models/iforward/demo/report_builder.py
```

---

## 8. 完整配置草案

```yaml
model:
  iforward:
    version: stage3_3_uncertainty_v1

    uncertainty:
      enable: true
      representation: log_variance
      channels: 1
      dtype: fp32

      state:
        init_sigma:
          bg: 0.08
          distant: 0.12
          rigid: 0.10
        sigma_min: 0.01
        sigma_max: 0.50
        prior_pull: 0.0

      updater:
        enable: true
        detach_input: true
        gate_by_appearance_valid: true
        gate_by_main_noop: false
        max_logvar_step:
          bg: 0.08
          distant: 0.10
          rigid: 0.10
        zero_init_head: true

      rasterizer:
        mode: two_pass_detached_moments
        variance_mode: total_variance_scalar
        detach_geometry: true
        detach_opacity: true
        detach_mean_color: true
        background_sigma: 0.10
        variance_floor: 1.0e-4
        variance_max: 0.25
        alpha_valid_min: 0.01
        grouped_multiview: true

      loss:
        distribution: gaussian
        channel_reduction: mean
        nll_weight: 0.50
        calibration_weight: 0.10
        raw_l1_anchor_weight: 0.25
        use_existing_ssim_anchor: true
        decoupled: true
        current_enable: true
        history_enable: true
        repair_enable: true
        history_damage_enable: false
        history_precision_floor: 0.30
        repair_precision_floor: 0.30

      logging:
        interval: 200
        calibration_interval: 1000
        image_interval: 5000
```

---

## 9. Checkpoint 与迁移

### 9.1 旧 checkpoint -> 新模型

允许 missing keys：

```text
appearance_logvar state
head_appearance_logvar_delta.*
uncertainty loss buffers
```

迁移时：

```text
旧 Gaussian 参数正常加载；
appearance_logvar 按 branch prior 初始化；
uncertainty head zero-init；
optimizer state 对新参数初始化为空。
```

### 9.2 新 checkpoint -> uncertainty disabled

保留字段但不使用，或提供 state strip 工具。不要让旧代码 strict load 新 checkpoint。

### 9.3 State version

在 checkpoint / run manifest 记录：

```text
local_gs_state_schema_version = 2
uncertainty_state_version = appearance_logvar_v1
uncertainty_raster_version = detached_moments_v1
```

---

## 10. AMP 策略

第一版：

```text
appearance_logvar state FP32
uncertainty head 可在 autocast 内计算，但 delta cast FP32
variance rasterizer FP32 island
pixel NLL FP32
calibration metrics FP32
```

原因：variance、log、exp、NLL precision 对范围更敏感；当前 render/loss 本来就是 FP32 island。

后续仅对 uncertainty head MLP 使用 BF16；state/raster/loss 不降精度。

---

## 11. 日志与校准指标

### 11.1 Per-branch state

```text
uncertainty/{branch}/sigma_mean
uncertainty/{branch}/sigma_p10/p50/p90
uncertainty/{branch}/logvar_min/max
uncertainty/{branch}/delta_logvar_mean/abs_mean/max
uncertainty/{branch}/clamp_min_ratio
uncertainty/{branch}/clamp_max_ratio
```

### 11.2 Pixel uncertainty

```text
uncertainty/pixel/variance_mean
uncertainty/pixel/aleatoric_mean
uncertainty/pixel/disagreement_mean
uncertainty/pixel/logvar_p10/p50/p90
uncertainty/pixel/error_uncertainty_corr
```

### 11.3 Loss

```text
loss_uncertainty_nll
loss_uncertainty_mean_path
loss_uncertainty_calibration
loss_raw_l1_anchor
loss_raw_ssim_anchor
```

### 11.4 必须同时保留 raw 指标

```text
raw PSNR
raw SSIM
raw L1
raw LPIPS if enabled
```

不能用 uncertainty-weighted PSNR 作为主效果指标。

---

## 12. Calibration 验证

### 12.1 Error-uncertainty correlation

计算：

```text
Pearson / Spearman correlation(pixel variance, squared residual)
```

按：

```text
repeat_refine
shuffled_coverage
high_block_repair
bg/distant/rigid
current/history/repair
```

分组。

### 12.2 Risk-coverage curve

按 uncertainty 从低到高保留：

```text
100%, 80%, 60%, 40%, 20%
```

保留区域 raw error 应单调下降。

### 12.3 Sparsification / AUSE

比较：

```text
按预测 uncertainty 删除像素
vs
按真实 residual oracle 删除像素
```

### 12.4 Before/after repair

记录：

```text
error_before / error_after
uncertainty_before / uncertainty_after
```

理想：repair 成功区域 error 和 uncertainty 同时下降。

---

## 13. Demo 可视化

每个 selected view 输出：

```text
GT
render RGB
absolute error
pixel sigma
aleatoric variance
disagreement variance
alpha
error-after minus error-before
uncertainty-after minus uncertainty-before
```

增加 confidence bin 表：

| sigma bin | pixel count | raw L1 | raw PSNR | destructive update ratio |
|---|---:|---:|---:|---:|

如果 uncertainty 有效：

```text
sigma 越高，raw error 越高；
低 sigma 区域 update 成功率更高；
高 sigma 区域更集中在遮挡、动态、边界和低 support 区域。
```

---

## 14. 单元测试

新增：

```text
tests/test_iforward_uncertainty_state.py
tests/test_iforward_uncertainty_delta.py
tests/test_iforward_uncertainty_raster.py
tests/test_iforward_uncertainty_nll.py
```

### 14.1 State test

```text
旧 NodeState 无 uncertainty -> 正确初始化；
state to/detach/clone/carry 保留 appearance_logvar；
rigid subset/world transform 保留 row 对齐；
checkpoint migration 正常。
```

### 14.2 Delta test

```text
head zero-init 时 delta=0；
delta bounded；
invalid row delta=0；
state clamp 正确；
uncertainty 不受 main noop 默认影响。
```

### 14.3 Raster toy test

两个 Gaussian、固定 alpha：

```text
相同颜色、zero variance -> disagreement≈0；
不同颜色 -> total variance 增大；
variance 增大 -> pixel variance 单调增大；
双 pass moment 与 CPU reference 一致。
```

### 14.4 Gradient routing test

```text
RGB pass 对 geometry/opacity/SH 有梯度；
variance pass 对 geometry/opacity/mean color 无梯度；
variance pass 对 appearance_logvar 有梯度；
decoupled mean path 不更新 uncertainty；
calibration path 不更新 RGB mean branch。
```

### 14.5 NLL test

```text
固定 residual 下，最优 variance 接近 residual energy；
logvar penalty 阻止无限 variance；
mask reduction 正确；
全 invalid mask 不产生 NaN。
```

---

## 15. Smoke / ablation 计划

### Phase 0：state-only

```text
新增 state 和 delta head；
uncertainty loss weight=0；
验证训练与旧模型 parity。
```

标准：

```text
RGB PSNR 变化 <0.05 dB；
delta_logvar 初始≈0；
checkpoint resume 正常。
```

### Phase 1：calibration-only

```text
variance raster + uncertainty calibration loss；
mean path仍用旧 raw loss；
uncertainty 不参与 RGB gradient weighting。
```

目标：先证明 uncertainty 与 error 相关。

### Phase 2：decoupled NLL

```text
启用 NLL mean weighting；
保留 raw L1/SSIM anchor。
```

### Phase 3：role ablation

```text
current-only NLL
history-only NLL
repair-only NLL
current+history+repair NLL
```

`history_damage` 始终 raw。

### Phase 4：renderer optimization

```text
双 pass -> 单 pass 5-channel；
做 RGB/gradient parity 后再切主线。
```

---

## 16. 成功标准

### 16.1 数值与校准

```text
无 NaN/Inf；
uncertainty clamp ratio 不长期贴边；
error-uncertainty Spearman > 0.3 作为初期目标；
risk-coverage 单调；
raw PSNR 不因 hiding 假提升。
```

### 16.2 效果

重点看：

```text
shuffled_coverage current 是否提升；
seq20 repair/order final probe 是否改善；
history 不下降；
uncertainty 高的区域是否与动态/遮挡/边界一致。
```

第一阶段合理目标：

```text
shuffled_coverage current +0.3~0.7 dB；
seq20 repair/order +0.2~0.5 dB；
raw seq10/seq20 assimilation 不下降超过0.1 dB。
```

---

## 17. 风险与控制

| 风险 | 表现 | 控制 |
|---|---|---|
| uncertainty hiding | sigma 全部变大、weighted loss 降但 raw PSNR 不涨 | raw anchor、clamp、decoupled gradient、raw metrics |
| dead zone | 高 uncertainty 区域永远不学 | precision floor；第一版不 gate GDKV/delta |
| dynamic region 被永久忽略 | rigid/dynamic 质量下降 | branch-specific prior；不 hard drop |
| opacity/geometry 操纵 variance | alpha 变化代替校准 | variance pass detach geometry/opacity |
| state 过度自信 | sigma 长期贴 min | prior、sigma_min、calibration metric |
| render 成本上升 | step time 明显增加 | correctness-first 双 pass后做5-channel融合 |

---

## 18. 推荐实施顺序

```text
P0  State schema + checkpoint migration
P1  Delta logvar head + bounded state update
P2  Detached two-pass moment rasterizer
P3  Calibration-only loss + metrics/demo
P4  Decoupled Gaussian NLL + raw anchor
P5  current/history/repair role ablation
P6  Single-pass 5-channel optimization
P7  仅在校准有效后考虑 confidence -> GDKV/update gate
```

---

## 19. 关键实现决策总结

```text
State：每 child Gaussian 一个 scalar FP32 log-variance。
Delta：独立 zero-init delta_logvar head，不复用现有 confidence。
Update：小步 bounded additive update，默认不乘 main noop。
Renderer：第一版双 pass，variance pass detach geometry/opacity/color。
Variance：total variance = per-GS aleatoric + visible splat disagreement。
Loss：decoupled Gaussian NLL + raw L1/SSIM anchor。
History damage：保持 raw relative damage，不做 uncertainty attenuation。
Metrics：主效果始终使用 unweighted PSNR/SSIM/L1。
```

---

## 20. 相关工作

- Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*, arXiv:1703.04977：input-dependent aleatoric uncertainty 与 learned loss attenuation。
- Tran & Kosecka, *VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM*, arXiv:2603.09673：per-splat appearance variance、alpha compositing 与 law-of-total-variance pixel uncertainty。
- Wang et al., *Uncertainty-Aware 4D Gaussian Splatting for Monocular Occluded Human Rendering*, arXiv:2602.06343：per-primitive uncertainty、pixel uncertainty rasterization 与 heteroscedastic robust supervision。
- Han & Dumery, *View-Dependent Uncertainty Estimation of 3D Gaussian Splatting*, arXiv:2504.07370：view-dependent per-Gaussian uncertainty 的后续扩展方向。

---

## 21. 后续 GPT 上下文块

```text
项目要实现 IForward Uncertainty V1，只有三部分：
1) per-Gaussian scalar appearance_logvar persistent state；
2) posterior updater 预测 bounded delta_logvar；
3) uncertainty rasterization + Gaussian pixel NLL。
不要复用 BranchDelta.confidence；新增 appearance_logvar_delta。
第一版使用双 pass：原 RGB render 保持不变；第二个 detached moment pass 只对 logvar 回传。
像素 total variance = alpha-composited per-GS variance + visible splat color disagreement。
使用 decoupled NLL：mean path 使用 detach(logvar)，calibration path使用detach(residual)。
保留 raw L1/SSIM anchor；history_damage 不使用 uncertainty attenuation。
所有 raw PSNR/SSIM/L1 继续作为主评估。
验证校准后，才考虑 confidence 控制 GDKV write 或 delta magnitude。
```
