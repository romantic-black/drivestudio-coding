# IForward Stage 2_0：CUDA Exact Diagonal Parent Projector 实现方案

版本：stage2_0_biggs_cuda_exact_diagonal_projector  
目标：把当前 PyTorch parent projection 替换为 **CUDA forward/backward exact diagonal projector**，同时修正 stage 2_0 中与 IForward 理念不一致的训练链路问题。

---

## 0. 关键结论

当前 `eig` 的目的只是：

```text
full parent covariance [M,3,3]
    -> eig
    -> parent scale + parent quat
```

但 stage 2_0 的 parent GS 主要有两个作用：

```text
1. 给 2D lifting 提供相对准确的 alpha/T 权重；
2. 把多个 fine GS 的 observation 压缩成 parent observation token。
```

因此不需要 full covariance + oriented ellipsoid。主线应改成：

```text
all branches use diagonal covariance
parent quat = identity / axis-aligned
parent scale = exact diagonal moment matching
```

也就是：

```text
删除 eig；
保留 exact weighted moment；
用 CUDA forward/backward 实现 exact diagonal projector。
```

这里的 **exact** 指：对选定的 diagonal covariance 数学定义精确实现 forward/backward，而不是 PyTorch 近似、cache scale/quat、fixed voxel 或 no-grad projection。

---

## 1. 当前实现为什么慢

当前 `models/iforward/biggs_parent_projector.py` 的 `_project_params_to_parents()` 走的是 PyTorch full covariance path：

```python
scales = torch.exp(params["scales_log"])
rot = _quat_to_rotmat(_normalize_quat(params["quats"]))
cov_child = rot @ torch.diag_embed(scales.square()) @ rot.transpose(-1, -2)
centered = means - out_means.index_select(0, pid)
cov_terms = cov_child + centered.unsqueeze(-1) @ centered.unsqueeze(-2)
cov = means.new_zeros((m, 3, 3))
cov.index_add_(0, pid, cov_terms * mass[:, None, None])
eigvals, eigvecs = torch.linalg.eigh(cov)
```

慢点有四类：

```text
1. 构造 [N,3,3] rot / diag / cov_child / cov_terms 临时张量；
2. 多次 index_add，kernel launch 多，内存带宽压力高；
3. torch.linalg.eigh([M,3,3]) 对大量小矩阵很不划算；
4. projection stats / finite check 中的 .item() / .all() / quantile 会造成 CUDA sync。
```

但如果采用 diagonal covariance，核心计算可以变成：

```text
每个 parent block 读取自己的 children；
在一个 CUDA block 内 reduce：mass、mean、diag second moment、opacity tau-area、SH；
直接输出 parent means / scales_log / identity quats / opacity_logit / SH。
```

不需要 `eig`，也不需要 `[N,3,3]` 大临时张量。

---

## 2. 新 projector 的数学定义

### 2.1 输入

对每个 fine child GS：

```text
μ_i              : [3]
s_i_log          : [3]
q_i              : [4], wxyz
opacity_logit_i  : [1]
sh_dc_i          : [3]
sh_rest_i        : [B,3]
parent_id_i      : int
```

assignment 固定，只作为 routing，不参与梯度：

```text
child_order      : [N]
parent_start     : [M]
parent_count     : [M]
```

### 2.2 动态 mass

Stage 2_0 既然要求 projector forward/backward，应避免只用 assignment 构建时的 detached `child_mass`。主线使用动态 mass：

```text
scale_i = exp(s_i_log)
opacity_i = sigmoid(opacity_logit_i)
tau_i = softplus(opacity_logit_i)
area_i = top2(scale_i).prod()
m_i = clamp_min(tau_i * area_i, min_mass)
```

说明：

```text
tau_i = -log(1 - sigmoid(logit_i)) = softplus(logit_i)
```

这比显式 `sigmoid -> log1p` 更稳定。

可选配置：

```yaml
parent_projector:
  mass_mode: dynamic_tau_area   # dynamic_tau_area | static_assignment_mass
```

但默认应为：

```text
dynamic_tau_area
```

因为只有这样 parent mean/scale/opacity/SH 的权重才会随 fine opacity/scale 学习而变化。

### 2.3 parent mean

```text
W_k = Σ_{i∈C_k} m_i
A_k = Σ_{i∈C_k} m_i μ_i
μ_k = A_k / W_k
```

### 2.4 child world diagonal covariance

虽然 parent 使用 diagonal covariance，但 child 自身仍然可以有 quat。不要忽略 child quat。对 child covariance：

```text
Σ_i = R(q_i) diag(scale_i^2) R(q_i)^T
```

只取世界坐标对角线：

```text
d_i.x = R00^2 sx^2 + R01^2 sy^2 + R02^2 sz^2
d_i.y = R10^2 sx^2 + R11^2 sy^2 + R12^2 sz^2
d_i.z = R20^2 sx^2 + R21^2 sy^2 + R22^2 sz^2
```

这里不需要构造完整 3×3 matrix。

### 2.5 parent diagonal covariance

用 second moment 形式，避免两遍 centered outer：

```text
B_k = Σ_i m_i (d_i + μ_i^2)
var_k = B_k / W_k - μ_k^2
var_k = clamp(var_k + eps, min_scale^2, max_scale_branch^2)
scale_k = sqrt(var_k)
scales_log_k = log(scale_k)
```

parent quat：

```text
quat_k = [1, 0, 0, 0]
```

这表示 parent 是 world-axis aligned Gaussian。

### 2.6 parent opacity

使用 optical-thickness area aggregation，但避免 hard cap 梯度归零。建议使用 soft cap：

```text
U_k = Σ_i tau_i * area_i
area_parent_k = top2(scale_k).prod()
tau_parent_k = tau_parent_scale_branch * U_k / (area_parent_k + eps)
opacity_parent_k = opacity_cap * (1 - exp(-tau_parent_k))
opacity_parent_k = clamp(opacity_parent_k, opacity_min, opacity_cap - eps)
opacity_logit_parent_k = logit(opacity_parent_k)
```

不要用：

```text
opacity_parent = clamp(1 - exp(-tau), max=cap)
```

因为 hard cap 会让 saturation 区域梯度变成 0。当前 parent opacity saturation 已经偏高，所以 soft cap 更合适。

### 2.7 parent SH

```text
sh_dc_k = Σ_i m_i sh_dc_i / W_k
sh_rest_k = Σ_i m_i sh_rest_i / W_k
```

---

## 3. CUDA op 设计

### 3.1 推荐位置

不建议把这个 op 放进 gsplat 主逻辑，因为它是 IForward-specific parent construction，不是 rasterizer。建议新增：

```text
models/iforward/csrc/biggs_parent_projector_ext.cpp
models/iforward/csrc/biggs_parent_projector_diag.cu
models/iforward/cuda_parent_projector.py
```

如果现有工程没有独立 CUDA extension build，可以临时挂到 gsplat extension，但命名必须私有化：

```text
gsplat.cuda._wrapper.biggs_parent_project_diag_forward
gsplat.cuda._wrapper.biggs_parent_project_diag_backward
```

长期建议迁回 StreetForward / IForward extension。

### 3.2 Python API

新增 autograd Function：

```python
class _BigGSParentProjectDiagFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means,              # [N,3]
        scales_log,         # [N,3]
        quats,              # [N,4], wxyz
        opacity_logit,      # [N,1]
        sh_dc,              # [N,3]
        sh_rest,            # [N,B,3]
        child_order,        # [N]
        parent_start,       # [M]
        parent_count,       # [M]
        min_scale: float,
        max_scale: float,
        opacity_cap: float,
        opacity_min: float,
        tau_parent_scale: float,
        eps: float,
        mass_mode: int,
    ):
        ...

    @staticmethod
    def backward(ctx, *grad_outputs):
        ...
```

输出：

```text
parent_means          [M,3]
parent_scales_log     [M,3]
parent_quats          [M,4] identity
parent_opacity_logit  [M,1]
parent_sh_dc          [M,3]
parent_sh_rest        [M,B,3]
child_mass_sum        [M]
child_mass_mean       [M]
```

封装函数：

```python
def project_biggs_parents_cuda_exact_diag(
    *,
    params: Dict[str, Tensor],
    assignment: BigGSBranchAssignment,
    cfg: Any,
    max_scale: float,
) -> BigGSParentProjection:
    ...
```

替换现有 PyTorch path：

```python
if cfg.parent_projector.backend == "cuda_exact_diag":
    return project_biggs_parents_cuda_exact_diag(...)
else:
    return project_biggs_parents_torch(...)
```

---

## 4. CUDA forward kernel

### 4.1 Kernel layout

因为 assignment 有 cap：

```text
bg max children ≈ 32
distant max children ≈ 64
rigid max children ≈ 32
```

最适合：

```text
grid.x = M parents
block.x = 64 or 128 threads
one CUDA block per parent
```

每个 parent：

```cpp
int p = blockIdx.x;
int start = parent_start[p];
int count = parent_count[p];
for local = threadIdx.x; local < count; local += blockDim.x:
    i = child_order[start + local];
    compute child mass, diag_cov, sums
block_reduce sums
thread 0 writes parent outputs
```

### 4.2 Forward accumulators

每个 block 需要 reduce：

```text
W                         scalar
A[3]                      Σ m μ
B[3]                      Σ m (diag_cov + μ^2)
U                         Σ tau * area
SH_DC[3]                  Σ m sh_dc
SH_REST[B,3]              Σ m sh_rest
```

对于 SH：

- `sh_degree=1` 时 `B=3`，非常小，可以直接在同一个 kernel 中 loop；
- 若未来 `B` 变大，可以拆成第二个 kernel 或用 vectorized loop。

### 4.3 Top2 area

child area：

```cpp
scale = exp(scales_log)
area_child = product_top2(scale.x, scale.y, scale.z)
```

parent area：

```cpp
area_parent = product_top2(parent_scale.x, parent_scale.y, parent_scale.z)
```

同时保存 top2 index，供 backward 使用：

```text
child_top2_idx optional recompute in backward
parent_top2_idx saved [M,2] or recompute from parent_scale
```

建议 backward 重新计算 child top2，保存 parent scale 即可。

### 4.4 Saved tensors

为了 backward，不要保存巨大 `[N,3,3]`。保存：

```text
inputs:
    means, scales_log, quats, opacity_logit, sh_dc, sh_rest
    child_order, parent_start, parent_count
outputs / small buffers:
    parent_means [M,3]
    parent_scales [M,3]
    parent_opacity [M]
    mass_sum [M]
    maybe parent_area [M]
    maybe tau_area_sum [M]
```

`tau_area_sum` 可在 backward 中重算，但保存它可以减少一次 parent reduce。推荐保存：

```text
mass_sum
parent_means
parent_scales
parent_area
tau_area_sum
```

---

## 5. Backward 设计

### 5.1 Backward 总体结构

不要在 backward 中一个 kernel 完成所有事。推荐两段：

```text
Kernel B1: parent_adjoint_kernel
    输入 grad_parent_outputs 和 saved parent buffers
    输出每个 parent 的 accumulator gradients：
        gW[M]
        gA[M,3]
        gB[M,3]
        gU[M]
        gC_dc[M,3]
        gC_rest[M,B,3]

Kernel B2: child_backward_kernel
    one thread per child or one block per parent
    读取 child input + parent adjoints
    输出 grad child params：
        grad_means[N,3]
        grad_scales_log[N,3]
        grad_quats[N,4]
        grad_opacity_logit[N,1]
        grad_sh_dc[N,3]
        grad_sh_rest[N,B,3]
```

B2 不需要 atomic，因为每个 child 只写自己的 gradient。

### 5.2 Parent output 到 accumulators 的梯度

定义：

```text
mean = A / W
var = B / W - mean^2
scale = sqrt(var_clamped)
scales_log = log(scale)
sh = C / W
```

从 `grad_scales_log` 得到：

```text
g_scale = grad_scales_log / scale
if var outside clamp range: g_var = 0
else g_var = g_scale * 0.5 / scale
```

`var = B/W - mean^2`：

```text
gB += g_var / W
gmean += -2 * g_var * mean
gW += -sum(g_var * B) / W^2
```

`mean = A/W`：

```text
gA += gmean / W
gW += -dot(gmean, A) / W^2
```

`sh = C/W`：

```text
gC += gsh / W
gW += -dot(gsh, C) / W^2
```

Opacity：

```text
opacity = opacity_cap * (1 - exp(-tau_parent))
logit = log(opacity / (1 - opacity))
```

从 `grad_opacity_logit`：

```text
g_opacity = grad_logit / (opacity * (1 - opacity))
g_tau_parent = g_opacity * opacity_cap * exp(-tau_parent)
```

`tau_parent = tau_parent_scale * U / area_parent`：

```text
gU += g_tau_parent * tau_parent_scale / area_parent
g_area_parent += -g_tau_parent * tau_parent_scale * U / area_parent^2
```

`area_parent = top2(scale).prod()`：

```text
if top2 = (a,b):
    g_scale[a] += g_area_parent * scale[b]
    g_scale[b] += g_area_parent * scale[a]
```

再把这部分加回 `g_var`。

### 5.3 Accumulators 到 child 的梯度

Child 对 accumulators 的贡献：

```text
W += m
A += m * μ
B += m * (diag_cov + μ^2)
U += tau * area
C += m * sh
```

因此：

```text
g_m += gW
      + dot(gA, μ)
      + dot(gB, diag_cov + μ^2)
      + dot(gC_dc, sh_dc)
      + dot(gC_rest, sh_rest)

g_μ += m * gA + m * 2 * μ * gB

g_diag_cov += m * gB

g_sh += m * gC

g_tau += gU * area

g_area += gU * tau
```

如果 `mass_mode=dynamic_tau_area`：

```text
m = tau * area
```

则：

```text
g_tau += g_m * area
g_area += g_m * tau
```

如果 `mass_mode=static_assignment_mass`：

```text
g_m 不回传到 child opacity/scale，只用于日志或 optional grad_child_mass。
```

### 5.4 area 到 child scale 的梯度

```text
area = product_top2(scale)
scale = exp(scales_log)
```

如果 top2 是 `(a,b)`：

```text
g_scale[a] += g_area * scale[b]
g_scale[b] += g_area * scale[a]
g_scales_log[j] += g_scale[j] * scale[j]
```

### 5.5 diag_cov 到 scale / quat 的梯度

Child diagonal covariance：

```text
d_a = Σ_j R[a,j]^2 * scale_j^2
```

Given `g_d[a]`：

```text
g_scale_j += Σ_a g_d[a] * 2 * R[a,j]^2 * scale_j
```

转到 log-scale：

```text
g_scales_log_j += g_scale_j * scale_j
```

对 rotation matrix：

```text
g_R[a,j] += g_d[a] * 2 * R[a,j] * scale_j^2
```

然后 `R = quat_to_rotmat(normalize(q))`，backward 需要：

```text
g_q_norm = dR/dq_norm^T * g_R
g_q_raw = normalize_backward(q_raw, g_q_norm)
```

### 5.6 quat backward 实现建议

不要在文档层面用数值差分。必须手写解析 backward。

沿用项目约定 `q=(w,x,y,z)`，rotation matrix：

```text
R00 = 1 - 2(y^2 + z^2)
R01 = 2(xy - wz)
R02 = 2(xz + wy)
R10 = 2(xy + wz)
R11 = 1 - 2(x^2 + z^2)
R12 = 2(yz - wx)
R20 = 2(xz - wy)
R21 = 2(yz + wx)
R22 = 1 - 2(x^2 + y^2)
```

根据 `g_R` 累积到 normalized quat：

```text
gw += -2*z*gR01 +  2*y*gR02 + 2*z*gR10 - 2*x*gR12 - 2*y*gR20 + 2*x*gR21

gx +=  2*y*gR01 +  2*z*gR02 + 2*y*gR10 - 4*x*gR11 - 2*w*gR12 + 2*z*gR20 + 2*w*gR21 - 4*x*gR22

gy += -4*y*gR00 + 2*x*gR01 + 2*w*gR02 + 2*x*gR10 + 2*z*gR12 - 2*w*gR20 + 2*z*gR21 - 4*y*gR22

gz += -4*z*gR00 - 2*w*gR01 + 2*x*gR02 + 2*w*gR10 - 4*z*gR11 + 2*y*gR12 + 2*x*gR20 + 2*y*gR21
```

然后 normalization backward：

```text
q_norm = q / (||q|| + eps)

g_q = g_q_norm / norm - q * dot(g_q_norm, q) / norm^3
```

要注意：如果 forward 使用 `norm + eps`，backward 也要完全一致。

---

## 6. Forward/backward 和训练链路的必要修正

仅实现 projector backward 还不够。当前 stage 2_0 里有几个会切断梯度的点，必须一起修。

### 6.1 当前配置问题

当前配置是类似：

```yaml
model:
  iforward:
    trainability:
      train_measurement_frontend: false
  stage6_0:
    base_measurement:
      source_evidence_grad_mode: no_grad_v4
      train_2d_frontend: false
      train_residual_unet: false
      train_fusion_neck: false
      train_v4_lift: false
      detach_v4_outputs: true
```

这与“2D 部分需要训练”的目标冲突。

应改成：

```yaml
model:
  iforward:
    trainability:
      train_measurement_frontend: true
  stage6_0:
    base_measurement:
      source_evidence_grad_mode: grad_v4
      train_2d_frontend: true
      train_residual_unet: true
      train_fusion_neck: true
      train_dinov2: false
      train_v4_lift: false        # 先不训 raster/lift 参数；如果后续 V4 权重端有 backward 再开
      detach_v4_outputs: false
```

### 6.2 `_render_source_scene_only_for_cnn` 当前 detach 了 parent render

当前实现里有：

```python
scene_rgb_batch = torch.stack(scene_rgbs, dim=0).detach()
```

如果希望 parent render residual 对 parent params / projector 有梯度，必须改成配置控制：

```python
scene_rgb_batch = torch.stack(scene_rgbs, dim=0)
if self.stage6_detach_source_render_for_cnn:
    scene_rgb_batch = scene_rgb_batch.detach()
```

新增配置：

```yaml
model:
  stage6_0:
    base_measurement:
      detach_source_render_for_cnn: false
```

### 6.3 BigGS observe 当前使用 detached local state

当前 `_observe_stage2_0_biggs_measurement()` 开头使用：

```python
bg_m, distant_m, rigid_m = self._local_to_node_states_detached(local_state)
```

如果 projector backward 要连到 fine local state / previous delta，需要新增非 detach path：

```python
bg_m, distant_m, rigid_m = self._local_to_node_states(
    local_state,
    detach=not self.stage2_0_biggs_projector_grad_to_local_state,
)
```

新增配置：

```yaml
model:
  iforward:
    biggs:
      parent_projector:
        grad_to_local_state: true
```

注意：persistent writeback 仍然必须 detach。只是 observe/event 内部不能 detach。

### 6.4 Parent encoder / child decoder 中不要 detach parent params

当前配置有：

```yaml
param_obs_codec:
  detach_params: true
child_decoder:
  detach_child_code_inputs: true
  detach_child_params: true
  detach_parent_params: true
```

如果想让 projector backward 生效，应改成：

```yaml
model:
  stage6_0:
    struct_event_decoder:
      param_obs_codec:
        detach_params: false
        detach_obs_code: false       # obs 本身无 parent param grad，可仍 true；先 false 方便检查
        detach_acc_w: false          # acc_w 目前 non-diff；如果 V4 不回权重梯度，true/false 无影响

  iforward:
    biggs:
      child_decoder:
        detach_child_code_inputs: false
        detach_child_params: false
        detach_parent_params: false
```

如果训练不稳定，可以只保留 parent params 梯度：

```yaml
child_decoder:
  detach_child_params: true
  detach_parent_params: false
```

### 6.5 AlphaT V4 当前只对 features_2d 有 backward

`AlphaTWeightExtractorV4` 的 autograd Function 当前 backward 返回：

```text
grad_feat2d
```

但对：

```text
means2d, conics, opacities
```

返回 `None`。

这意味着：

```text
parent projection backward 不会从 alpha/T lifting 权重本身收到梯度；
它只能从 parent render residual、parent param embedding、child decoder code 等路径收到梯度。
```

如果目标是“让 parent alpha/T 权重本身可学习”，还必须新增：

```text
V4 rasterize/backproject backward wrt means2d / conics / opacities
    -> gsplat projection backward
    -> parent means/scales/quats/opacities
    -> CUDA parent projector backward
    -> fine GS params
```

这是独立大工程。Stage 2_0 CUDA projector 可以先实现，但 validation 必须检查实际梯度来源：

```text
grad/parent_projector_from_param_embed
grad/parent_projector_from_source_render
grad/parent_projector_from_lifting_weights
```

第三项在当前 V4 下应为 0。

---

## 7. 配置建议

主线配置：

```yaml
model:
  iforward:
    biggs:
      parent_projector:
        backend: cuda_exact_diag
        covariance_mode: diagonal
        branches: [bg, distant, rigid]
        mass_mode: dynamic_tau_area
        grad_to_local_state: true
        min_scale: 0.001
        max_scale_bg: 0.6
        max_scale_distant: 3.0
        max_scale_rigid: 0.45
        opacity_cap: 0.9
        opacity_min: 1.0e-6
        tau_parent_scale_bg: 0.5
        tau_parent_scale_distant: 0.7
        tau_parent_scale_rigid: 0.5
        eps: 1.0e-6
        finite_check_interval: 100
        profile_cuda_events: true

      child_decoder:
        mode: low_rank_basis
        detach_child_code_inputs: false
        detach_child_params: false
        detach_parent_params: false

  stage6_0:
    base_measurement:
      source_evidence_grad_mode: grad_v4
      detach_v4_outputs: false
      detach_source_render_for_cnn: false
      train_2d_frontend: true
      train_residual_unet: true
      train_fusion_neck: true
      train_dinov2: false
      train_v4_lift: false

    struct_event_decoder:
      param_obs_codec:
        detach_params: false
        detach_obs_code: false
        detach_acc_w: false
```

Optimizer：

```yaml
optimizer:
  lr:
    measurement_frontend: 1.0e-5
    stage6_measurement_frontend_residual_unet: 1.0e-5
    stage6_measurement_frontend_fusion_neck: 1.0e-5
    stage6_struct_decoder: 1.0e-5
    biggs_child_decoder: 1.0e-4
    stage6_posterior_updater_base: 1.0e-5
```

---

## 8. 集成改动点

### 8.1 `biggs_parent_projector.py`

新增：

```python
def _project_params_to_parents_cuda_exact_diag(...):
    ...
```

`_project_params_to_parents()` 中分支：

```python
backend = str(cfg_get(cfg, "backend", "torch_full_eigh"))
covariance_mode = str(cfg_get(cfg, "covariance_mode", "diagonal"))

if backend == "cuda_exact_diag":
    if covariance_mode != "diagonal":
        raise ValueError("cuda_exact_diag requires covariance_mode=diagonal")
    return _project_params_to_parents_cuda_exact_diag(...)
```

保留旧 PyTorch full eig path 作为 fallback / debug：

```yaml
parent_projector:
  backend: torch_full_eigh
```

### 8.2 `minimal_trainer_stage6_0.py`

新增非 detach node conversion：

```python
def _local_to_node_states(self, local_state, *, detach: bool):
    if detach:
        return self._local_to_node_states_detached(local_state)
    return local_state.to(device=self.device).to_node_states_grad()
```

需要在 `LocalGSState` 里实现：

```python
def to_node_states_grad(self):
    return NodeStateBackground(... tensors without detach/clone ...), ...
```

Stage2 observe 使用：

```python
detach_local = not bool(cfg_get(projector_cfg, "grad_to_local_state", False))
bg_m, distant_m, rigid_m = self._local_to_node_states(local_state, detach=detach_local)
```

### 8.3 `LocalGSState`

新增：

```python
def to_node_states_grad(self):
    # no detach, no clone for floating tensors
    # template tensors such as point_ids / instance transforms can remain detached/static
```

Rigid 注意：

```text
point_ids / instance_ids / frame_ids 是 metadata，不参与 grad；
instances_quats / instances_trans 如果未来训练 object pose，可不 detach；当前可保持 template static。
```

### 8.4 V4 lifting gradient contract

当前可以先只保证：

```text
features_2d receives grad
```

如果要让 parent alpha/T 权重端可训练，必须另开：

```text
AlphaT V4 backward wrt means2d/conics/opacities
```

这个不属于 projector 本身，但必须在整体检查中记录为 blocker。

---

## 9. 测试方案

### 9.1 Forward 对齐测试

新增：

```text
tests/test_iforward_biggs_cuda_projector.py
```

测试：

```python
test_cuda_diag_forward_matches_torch_reference()
```

Torch reference 使用同样 diagonal math，不用旧 full eig：

```text
parent means / scales_log / opacity_logit / sh_dc / sh_rest
relative error < 1e-4 float32
```

### 9.2 Backward gradcheck

使用小 N/M：

```text
N=32, M=5, B=3, dtype=float64
```

测试：

```python
torch.autograd.gradcheck(fn, inputs, eps=1e-4, atol=1e-3, rtol=1e-2)
```

需要注意 top2/clip 的不可导点。测试输入避免：

```text
scale_x == scale_y
var 接近 min/max clamp
opacity 接近 cap
```

### 9.3 Integration gradient test

测试 projector backward 真实接入：

```python
loss = parent_params["means"].sum() + parent_params["scales_log"].sum() + parent_params["opacity_logit"].sum()
loss.backward()
assert local_state.bg.means.grad is not None
assert local_state.bg.scales_log.grad is not None
assert local_state.bg.opacity_logit.grad is not None
```

再测 Stage2 event：

```python
measurement = runtime._observe_stage2_0_biggs_measurement(... grad mode ...)
event = runtime._build_stage2_0_biggs_event_from_measurement(...)
loss = event.event_bg.square().mean()
loss.backward()
assert grad exists on measurement frontend or local state, according to config
```

### 9.4 Performance test

对真实规模：

```text
N_bg=300k
N_distant=200k
M_total=25k-50k
B=3
```

记录：

```text
projector/forward_ms
projector/backward_ms
projector/max_memory_allocated
```

目标：

```text
forward < 30-80 ms
backward < 60-150 ms
```

实际上取决于 SH bases、child_count cap 和 GPU，但应显著低于 PyTorch full eig path。

---

## 10. IForward 理念整体检查

### 10.1 当前最大问题：训练链路还是偏 no-grad

Stage 2_0 的理念是：

```text
parent GS 承接 2D evidence；
parent event 做压缩后的 3D reasoning；
fine event decode 后由原 updater 写回 fine GS。
```

如果 2D frontend 和 parent evidence 全部 no-grad，模型学到的主要是：

```text
parent_event_encoder + child_decoder + posterior_updater
```

这会削弱 IForward “iterative forward optimizer” 的核心：2D evidence 与 update 之间应可共同适配。必须打开 measurement frontend 训练。

### 10.2 Parent projection backward 不等于 alpha/T 权重可训练

即使 CUDA projector 有 backward，如果 V4 backproject 只对 `features_2d` 回传梯度，parent α/T 权重本身仍然不可训练。

所以阶段划分应明确：

```text
2_0a:
    CUDA exact diagonal projector forward/backward
    2D frontend trainable
    parent params through render/param embedding/decoder 有梯度

2_0b:
    V4 alpha/T backward wrt means2d/conics/opacities
    parent weights 本身可训练
```

不要把 2_0a 的成功误判成 alpha/T 权重端已经被训练。

### 10.3 Diagonal parent 与 child local frame 一致

既然所有 parent 都使用 diagonal covariance，parent quat 应该固定 identity。child decoder 的 `parent_local_frame` 也应对应 world-axis frame。否则使用无意义的 parent quat 会引入噪声。

建议：

```yaml
child_decoder:
  child_code_parent_local_frame: false
```

或者保留 true，但 parent quat identity，效果等价。

### 10.4 scheduler 仍需修

Stage 2_0 当前仍不应在同一 scene 连续跑多个 episode。应改：

```yaml
scheduler_iforward:
  traversal:
    traversal_mode: scene_round_robin_episode
    forbid_consecutive_same_scene: true
```

同时 validation rollout shape 不应硬编码为 `b4_r2`，要与 stage 2_0 单帧目标对齐：

```yaml
iforward_validation:
  rollout_shapes:
    - name: b1_r4
      blocks_per_rollout: 1
      repeats_per_block: 4
      prob: 1.0
```

### 10.5 assignment cache 仍需修

CUDA projector 解决的是 per-repeat projection 慢，不解决 episode_begin assignment rebuild。assignment cache 仍应使用：

```text
scene_id + segment_id + topology_signature + assignment_cfg_hash
```

不能绑定 episode_id。

---

## 11. 实现优先级

### P0：数学与配置修正

```text
1. 全 branch diagonal covariance；
2. parent quat identity；
3. dynamic_tau_area mass；
4. 关闭 hard opacity clamp，改 soft cap；
5. 打开 measurement frontend grad；
6. 去掉 BigGS observe 的 local_state detach；
7. parent encoder / child decoder 参数 detach 改为可配置，主线 false。
```

### P1：CUDA exact diagonal projector

```text
1. C++/CUDA extension；
2. forward one-block-per-parent reduce；
3. backward parent-adjoint + child-gradient 两段 kernel；
4. gradcheck；
5. 对齐 torch diagonal reference。
```

### P2：训练链路验证

```text
1. 验证 measurement frontend 有非零 grad；
2. 验证 local_state/fine GS params 有 projector 方向 grad；
3. 验证 projector backward 实际参与 loss；
4. 单帧 before/after validation。
```

### P3：如果需要 alpha/T 权重端可训练

```text
1. 扩展 V4 backward wrt means2d/conics/opacities；
2. 接通 gsplat 3D->2D projection backward；
3. 检查梯度和显存。
```

---

## 12. 最终推荐主线

最终 stage 2_0 应改为：

```text
fine local state, grad-enabled
    ↓
fixed assignment, cached by scene/segment/topology
    ↓
CUDA exact diagonal parent projector, forward/backward
    ↓
parent GS scene, axis-aligned, soft opacity cap
    ↓
trainable source 2D frontend
    ↓
parent alpha/T lifting [M,C]
    ↓
parent AnchorTokenBuilder + xCPE
    ↓
child low-rank decoder, no trainable child 2D skip
    ↓
original posterior updater
    ↓
fine delta + original render supervision
```

这条路线符合 IForward 的核心理念：

```text
2D evidence 进入 parent；
3D reasoning 在低 token parent 上完成；
fine GS 只接收 decoded event 和 delta；
训练梯度不被 no-grad / detach 切断。
```

最重要的是：

```text
删除 eig 不等于放弃精确性；
而是把“精确 full covariance parent”改成“精确 diagonal parent”。
```

对于 stage 2_0 的 parent lifting/pooling 目标，这个取舍是正确的。
