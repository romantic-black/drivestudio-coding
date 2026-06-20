# IForward Stage 2_0：FWHR-Lift 详细实现方案

## Fine-Weighted Hierarchical Residual Lifting

基线代码：`drivestudio_stage6_refactor_context_20260620_v30`  
建议版本名：`stage2_0_fwhr_lift_grld_dinov2base`  
方案范围：单帧/current 优化阶段；不引入 teacher；不启用 history gate、ADC 或 parent-level memory。

---

# 0. 决策摘要

本方案对 Stage 2_0 作一个明确的架构修正：

> **Fine GS 负责真实 visibility、alpha/T 与高频当前帧 evidence；Parent GS 负责 pooling、3D token 压缩和重型上下文建模。**

不再要求一个大 parent GS 同时近似一组 fine GS 的：

```text
深度排序
遮挡边界
细粒度 footprint
局部图像纹理
```

主链路改为：

```text
当前 fine GS state
    │
    ├─ fine GS render -> current residual image
    │
    └─ fine GS alpha/T rasterization（单次）
          │
          ├─ 直接按 child_to_parent 累加 48D context
          │      -> parent_context [M,48]
          │
          └─ 按 fine child 累加 8D residual detail
                 -> child_detail [N,8]
                 -> parent 内 support-weighted centering
                 -> child_detail_residual [N,8]

parent_context [M,48]
    -> AnchorTokenBuilder
    -> parent xCPE / future PTv3 / Mamba
    -> parent_event [M,64]

parent_event + child Gaussian relation
    -> GRLD
    -> fine_geometry_event [N,16]

child_detail_residual [N,8]
    -> lightweight detail adapter
    -> appearance_detail_hidden [N,32]

fine_geometry_event
    -> posterior shared trunk
    ├─ geometry heads：means / scale / quat
    └─ appearance heads：opacity / SH
         额外接收 appearance_detail_hidden
```

本阶段不做：

```text
- 不做 parent GS alpha/T lifting。
- 不做 parent + child 两套 lifting。
- 不生成完整 child [N,48] feature。
- 不引入 teacher。
- 不让 child detail 写入 parent memory。
- 不让 child detail 默认影响 means / scale / quat。
- 不打开 alpha/T geometry Jacobian。
```

---

# 1. 为什么需要 FWHR-Lift

## 1.1 当前 parent-only lifting 的信息瓶颈

当前 parent-only observation 是：

```text
parent GS rasterization
    -> parent feature [M,48]
    -> parent event
    -> GRLD 根据 child 几何属性猜测 fine event
```

即使 GRLD 已经读取：

```text
child-parent relative xyz
relative diagonal covariance
relative optical mass
relative opacity
relative SH
```

它仍然没有观测到：

```text
当前图像中 child 实际覆盖区域的 residual、edge 和 texture。
```

多个 child 的当前图像 evidence 压成一个 parent feature 后，该信息不可逆。增加 GRLD rank 或 hidden dim 只能增强基于统计先验的猜测能力，不能恢复任意当前帧高频。

## 1.2 不能简单恢复完整 child lifting

完整 child feature：

```text
child_feature [N,48]
```

虽然能够恢复高频，但会重新带来：

```text
- N×48 feature 激活显存；
- N 级重型 3D reasoning 的诱惑；
- current-only shortcut 过强；
- parent token 压缩收益被部分抵消。
```

FWHR-Lift 的目标是保留 child 当前帧信息中的最小必要部分：

```text
parent：完整 48D context
child：仅 8D 高频 residual
```

## 1.3 核心解耦

FWHR 将两个角色解耦：

| 角色 | primitive | 说明 |
|---|---|---|
| visibility carrier | fine GS | 保留真实 alpha/T、遮挡、深度排序和 footprint |
| reasoning token | parent GS | 保存压缩几何、parent event、未来 PTv3/Mamba token |

这意味着 parent 不必再是一个足够准确的 render replacement，只需是一个稳定、可解释的 3D optimizer anchor。

---

# 2. 数学定义

设当前 source view 集合为 `v=1...V`，像素为 `x`，fine child GS 为 `i`，parent assignment 为 `p(i)`。

## 2.1 Fine GS 贡献权重

对每个 view/pixel/child：

```text
w_i^v(x) = T_i^v(x) · α_i^v(x)
```

这里必须使用当前 fine GS 的真实：

```text
means
scale
quat
opacity
深度排序
```

geometry 仍保持 stop-gradient；只对 image feature 回传梯度。

## 2.2 Image context feature

图像 context：

```text
c^v(x) = concat(residual32^v(x), dino16^v(x)) ∈ R^48
```

Parent context 直接由 fine contribution 聚合：

```text
C_p =
    Σ_{i:p(i)=p} Σ_v Σ_x w_i^v(x) c^v(x)
    ------------------------------------------------
    Σ_{i:p(i)=p} Σ_v Σ_x w_i^v(x) + eps
```

输出：

```text
parent_context [M,48]
```

这不是：

```text
先生成 child_context [N,48]，再 scatter pooling。
```

而是在 raster CUDA kernel 内直接写入 parent rows。

## 2.3 Child detail feature

仅从 residual branch 生成低维 detail image feature：

```text
d^v(x) = DetailHead2D(residual32^v(x)) ∈ R^8
```

Fine child detail：

```text
z_i =
    Σ_v Σ_x w_i^v(x) d^v(x)
    ---------------------------
    Σ_v Σ_x w_i^v(x) + eps
```

输出：

```text
child_detail [N,8]
child_detail_weight [N]
```

## 2.4 Parent 内 detail centering

定义当前帧 observation weight：

```text
s_i = child_detail_weight_i
```

对 parent `p`：

```text
z̄_p = Σ_{i:p(i)=p} s_i z_i / (Σ_i s_i + eps)
```

Child residual：

```text
r_i = z_i - z̄_{p(i)}
```

不可见 child：

```text
s_i <= detail_support_min -> r_i = 0
```

由此严格满足：

```text
Σ_i s_i r_i ≈ 0
```

其意义是：

```text
parent context 保存 group common component；
child detail 只保存 parent 内的 current-image difference。
```

## 2.5 Parent observation code

现有 obs code 定义依赖每个 view 的 feature weight：

```text
rho = Σ_v weight_v
rho_code = log(1 + rho)
overlap = (rho - max_v weight_v) / (rho + eps)
```

FWHR 中应按 fine contribution直接生成 parent per-view weight：

```text
parent_weight_view[v,p]
    = Σ_{i:p(i)=p} weight_view[v,i]
```

再使用完全相同的公式生成：

```text
parent_obs_code [M,2]
```

这样 parent obs code 表示整个 child group 在多相机中的真实可见性，而不是 parent Gaussian approximation 的可见性。

---

# 3. Image feature extractor 改造

## 3.1 输出结构

新增数据结构：

```python
@dataclass
class FWHRImageFeatures:
    context: torch.Tensor   # [V,Hf,Wf,48]
    detail: torch.Tensor    # [V,Hf,Wf,8]
    aux: Dict[str, Any]
```

Extractor：

```text
GT RGB + fine current render RGB
    -> ResidualUNet-GN
    -> residual32

GT RGB
    -> frozen DINOv2-Base
    -> cached dino16

context48 = concat(residual32, dino16)
detail8   = ResidualDetailHead(residual32)
```

## 3.2 Residual U-Net：BatchNorm 改为 GroupNorm

当前 `ImageFeatureExtractor.DoubleConv` 使用：

```text
Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
```

当前有效 image batch 主要是相机数，并且 render residual 分布随 repeat 和模型更新变化，BatchNorm running stats 容易造成 train/eval mismatch。

必须改成可配置 normalization，主线默认 GroupNorm：

```python
def make_2d_norm(channels: int, norm: str, groups: int = 8):
    if norm == "groupnorm":
        g = min(groups, channels)
        while channels % g != 0:
            g -= 1
        return nn.GroupNorm(g, channels)
    if norm == "batchnorm":
        return nn.BatchNorm2d(channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(...)
```

配置：

```yaml
residual_unet:
  norm: groupnorm
  norm_groups: 8
```

不保留 validation 时 BN train-mode 这种临时行为。

## 3.3 DetailHead2D

建议：

```python
class ResidualDetailHead(nn.Module):
    def __init__(self, in_dim=32, out_dim=8):
        self.net = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=1, bias=True),
        )
```

注意：

```text
- DetailHead 只读取 residual32；
- 不读取 DINO；
- 不做 residual+DINO image fusion；
- DetailHead 是 feature compression，不是上下文建模。
```

初始化：

```text
Conv1x1 使用小 Kaiming 初始化；
不要全零初始化，否则高频支路启动过慢。
```

稳定性通过 posterior 中的 appearance gate 控制，而不是把 image detail 初始化为全零。

## 3.4 DINO cache

继续使用：

```text
DINOv2-Base
16 channels
frozen backbone + frozen adapter
GPU L1 cache, FP16
CPU L2 默认关闭
```

必须保证 miss 路径也在写入 GPU cache 前 cast 到：

```text
cache dtype = FP16
```

主线配置：

```yaml
dino:
  cache:
    dtype: float16
    cpu_max_items: 0
    gpu_max_items: 2
```

## 3.5 新 extractor API

建议新增：

```text
models/feature_extractors/fwhr_image_features.py
```

主类：

```python
class FWHRImageFeatureExtractor(nn.Module):
    def forward(
        self,
        images_6ch,
        *,
        cached_dino=None,
    ) -> FWHRImageFeatures:
        ...
```

或者扩展当前 `DINOv2ResidualConcatExtractor`：

```python
forward_fwhr(...) -> FWHRImageFeatures
```

不建议让普通 `forward()` 有时返回 Tensor、有时返回 dataclass；新增显式 API 更安全。

---

# 4. Fine scene 构造与 parent 映射

## 4.1 Source render 改为 fine scene

当前 parent-only 路径使用 parent scene：

```text
parent scene -> source render -> residual CNN
parent scene -> alpha/T lifting
```

FWHR 必须改为：

```text
fine scene -> source render -> residual CNN
fine scene -> alpha/T hierarchical lifting
```

原因：

```text
residual 必须描述 GT 与当前真正被更新的 fine state 的误差；
不能描述 GT 与 parent approximation 的误差。
```

配置语义改成：

```yaml
biggs:
  observe:
    rendering_scene: fine
    lifting_scene: fine
    parent_scene_for_cnn: false
    parent_scene_for_lifting: false
```

旧 boolean 建议保留一个版本周期并 fail-fast 检查一致性，之后删除。

## 4.2 Active fine scene row order

建议统一拼接：

```text
fine_scene = [bg, distant, rigid_S]
```

定义：

```text
N_bg
N_distant
N_rigid_S
N_total
```

CUDA packed global child ID 对应这个 active row space。

## 4.3 Global parent row order

Parent event scene：

```text
parent_scene = [parent_bg, parent_distant, parent_rigid_active]
```

定义 parent offset：

```text
parent_offset_bg       = 0
parent_offset_distant  = M_bg
parent_offset_rigid    = M_bg + M_distant
```

构建 active fine `child_to_parent_global`：

```python
bg_parent_global = assignment_bg.child_to_parent

distant_parent_global = (
    assignment_distant.child_to_parent + M_bg
)

rigid_parent_global = (
    rigid_active_assignment.child_to_active_parent_S
    + M_bg + M_distant
)

child_to_parent_global = cat(
    bg_parent_global,
    distant_parent_global,
    rigid_parent_global,
)
```

Shape：

```text
[N_total]
```

必须满足：

```text
0 <= child_to_parent_global < M_total
```

## 4.4 Rigid active assignment 向量化

不允许使用：

```text
Python per-child .item()
Python per-parent nonzero scan
```

使用：

```python
key = global_parent_id * 2 + inside_mask.long()
unique_key, inverse, counts = torch.unique(
    key,
    sorted=True,
    return_inverse=True,
    return_counts=True,
)

child_to_active_parent = inverse
active_parent_global = unique_key // 2
parent_inside_mask = (unique_key % 2).bool()
child_order = torch.argsort(child_to_active_parent, stable=True)
parent_start = torch.cumsum(counts, 0) - counts
```

这是 FWHR 热路径前必须完成的速度修正，因为 fine rasterization 会增加 observe 成本，不能继续保留 Python rigid bottleneck。

---

# 5. FWHR CUDA operator

## 5.1 新 operator 命名

建议：

```text
rasterize_and_backproject_hierarchical_residual_multi_camera
```

Python wrapper：

```text
gsplat/cuda/_wrapper.py
```

C++：

```text
gsplat/cuda/csrc/Rasterization.cpp
gsplat/cuda/csrc/Rasterization.h
```

CUDA：

```text
gsplat/cuda/csrc/RasterizeAndBackprojectFWHRMulti.cu
```

## 5.2 Forward API

```python
fwhr_forward(
    means2d,                         # packed fine [P,2]
    conics,                          # [P,3]
    opacities,                       # [P]
    tile_offsets,
    flatten_ids,
    packed_global_child_ids,         # [P]
    child_to_parent_global,          # [N]
    context_feat2d,                  # [V,Hf,Wf,48]
    detail_feat2d,                   # [V,Hf,Wf,8]
    pair_valid_mask,                 # optional [V,H,W]
    num_children=N,
    num_parents=M,
    image_width,
    image_height,
    tile_size,
    weight_threshold,
    obs_eps,
) -> (
    parent_context_sum,              # [M,48]
    parent_context_weight,           # [M]
    parent_support_weight,           # [M]
    parent_weight_view,              # [V,M]
    parent_obs_code,                 # [M,2]
    child_detail_sum,                # [N,8]
    child_detail_weight,             # [N]
    child_support_weight,            # [N] optional/debug
    pair_count_total,
    pair_count_threshold,
)
```

## 5.3 Forward pair logic

现有 raster kernel 中，每个有效 pixel-child pair 已经计算：

```text
child_id
pixel_id
view_id
weight = T * alpha
```

新增：

```cpp
const int64_t child_id = packed_global_child_ids[g_local];
const int64_t parent_id = child_to_parent_global[child_id];
```

采样一次 context/detail image feature：

```text
context_pixel[48]
detail_pixel[8]
```

累加：

```cpp
for c in 0..47:
    atomicAdd(parent_context_sum[parent_id,c], weight * context_pixel[c]);

atomicAdd(parent_context_weight[parent_id], weight);
atomicAdd(parent_weight_view[view_id,parent_id], weight);

for d in 0..7:
    atomicAdd(child_detail_sum[child_id,d], weight * detail_pixel[d]);

atomicAdd(child_detail_weight[child_id], weight);
```

`support_weight` 保持现有 threshold/support 语义，不能和 feature normalization weight 混淆。

## 5.4 Normalize 与 obs code

Kernel 后：

```text
parent_context = parent_context_sum / (parent_context_weight + eps)
child_detail   = child_detail_sum / (child_detail_weight + eps)
```

Parent obs：

```text
rho_p = Σ_v parent_weight_view[v,p]
max_rho_p = max_v parent_weight_view[v,p]

obs[p,0] = log1p(rho_p)
obs[p,1] = (rho_p - max_rho_p) / (rho_p + obs_eps)
```

## 5.5 Child detail centering kernel

新增第二个小 kernel：

```text
center_child_detail_by_parent
```

输入：

```text
child_detail [N,8]
child_detail_weight [N]
child_to_parent [N]
child_order / parent_start / parent_count
```

输出：

```text
child_detail_residual [N,8]
parent_detail_mean [M,8] optional debug
child_detail_valid [N]
```

推荐 one block per parent：

```text
1. block reduce Σ s_i z_i 和 Σ s_i
2. 计算 mean
3. 每 thread 写 z_i - mean
4. invalid child 写 zero
```

不使用 global atomics。

## 5.6 Backward API

训练语义：

```text
feature gradient：开启
geometry gradient：关闭
```

Backward 输入：

```text
grad_parent_context [M,48]
grad_child_detail [N,8]
```

输出：

```text
grad_context_feat2d [V,Hf,Wf,48]
grad_detail_feat2d [V,Hf,Wf,8]
```

不输出：

```text
grad means2d
grad conics
grad opacities
grad child_to_parent
```

## 5.7 Backward 公式

Parent context normalization：

```text
C_p = S_p / W_p
```

对一个 pair `i,x`：

```text
∂L/∂c(x) += w_i(x) / W_parent(i) · grad_C_parent(i)
```

Child detail：

```text
z_i = D_i / W_i
```

对一个 pair：

```text
∂L/∂d(x) += w_i(x) / W_i · grad_z_i
```

Centering kernel 的 backward：

```text
r_i = z_i - Σ_j π_j z_j
```

其中 support weight stop-gradient，因此：

```text
grad_z_i = grad_r_i - π_i Σ_j grad_r_j
```

建议 centering 自定义 autograd/CUDA，避免保存大 scatter graph。

## 5.8 Dtype

当前 gsplat fused op 要求 feature 输入 FP32。第一版保持：

```text
context/detail image feature -> cast FP32 for raster op
output -> cast 回原 dtype
```

后续性能版本再支持 FP16/BF16 feature accumulation，但 accumulation 应保持 FP32。

## 5.9 Contention 风险

Direct parent accumulation 会让一个 parent 下多个 child pair 写同一 `[parent,48]` row，存在 atomic contention。

P0 先实现直接 atomic，记录：

```text
pairs/sec
parent contention histogram
kernel duration
```

若明显慢，再做 P1：

```text
- tile/shared partial parent accumulation；
- warp aggregated atomic；
- context 48 分块；
- 两阶段 pair contribution sort/reduce。
```

不要在 P0 就引入复杂排序 kernel。

---

# 6. AlphaT extractor 集成

新增：

```text
models/feature_extractors/fwhr_lifting_extractor.py
```

数据结构：

```python
@dataclass
class FWHRLiftOutput:
    parent_context: torch.Tensor          # [M,48]
    parent_support: torch.Tensor          # [M]
    parent_obs_code: torch.Tensor         # [M,2]
    child_detail: torch.Tensor            # [N,8], centered
    child_detail_support: torch.Tensor    # [N]
    child_detail_valid: torch.Tensor      # [N] bool
    aux: Dict[str, float]
```

API：

```python
class FWHRWeightExtractor(nn.Module):
    def render_and_backproject_hierarchical(
        *,
        fine_gaussians,
        cameras,
        image_features: FWHRImageFeatures,
        child_to_parent,
        num_children,
        num_parents,
        source_pair_valid_mask,
    ) -> FWHRLiftOutput:
        ...
```

保留 AlphaT V4 作为：

```text
baseline/reference path
```

FWHR 不应通过两次调用 V4 来模拟，因为那会做两套 rasterization。

---

# 7. Measurement 数据结构改造

当前 measurement 主要包含 parent feature。新增：

```python
measurement = {
    # parent full context
    "parent_feat_2d_bg":       [M_bg,48],
    "parent_feat_2d_distant":  [M_d,48],
    "parent_feat_2d_rigid_S":  [M_rS,48],

    "parent_acc_w_bg": ...,
    "parent_obs_bg": ...,

    # child low-dim current image detail
    "child_detail_bg":       [N_bg,8],
    "child_detail_distant":  [N_d,8],
    "child_detail_rigid_S":  [N_rS,8],

    "child_detail_support_bg": ...,
    "child_detail_valid_bg": ...,

    # parent geometry/state remains
    "parent_params_*": ...,
    "parent_coords_*": ...,
    "parent_runtime": ...,
    "assignment": ...,
    "route": original fine route,
}
```

Child detail row order必须严格保持：

```text
bg -> local_state.bg
 distant -> local_state.distant
 rigid_S -> route.S
```

---

# 8. Parent encoder 与 GRLD

Parent encoder 本身不需要结构修改：

```text
parent_context [M,48]
parent support / obs
parent params
    -> Stage6StructEventDecoder
    -> parent_event [M,64]
```

GRLD 继续输出：

```text
fine_geometry_event [N,16]
```

GRLD relation normalization 与 canonical rigid relation 修正继续保留：

```yaml
child_decoder:
  relation_normalization: sibling_rms
  rigid_relation_space: canonical
```

FWHR 不替代 GRLD。二者职责：

| 模块 | 信息 |
|---|---|
| GRLD | parent 3D context + child Gaussian state relation |
| FWHR child detail | child 当前图像高频 residual |

---

# 9. Posterior updater 双流改造

## 9.1 目标

Child image detail 第一版只影响：

```text
opacity
SH
```

不默认影响：

```text
means
scales
quat
noop gate
```

这样既恢复当前帧高频，又限制 current-only shortcut 直接破坏几何。

## 9.2 新 DetailPack

```python
@dataclass
class AppearanceDetailPack:
    bg: torch.Tensor                    # [N_bg,8]
    distant: Optional[torch.Tensor]
    rigid: Optional[torch.Tensor]
    valid_bg: torch.Tensor
    valid_distant: Optional[torch.Tensor]
    valid_rigid: Optional[torch.Tensor]
```

## 9.3 Updater 结构

当前：

```text
event16 -> trunk -> hidden32 -> all heads
```

改成：

```text
fine_geometry_event16
    -> shared trunk
    -> h_geom [N,32]

child_detail8
    -> detail_adapter
    -> h_detail [N,32]

h_app = h_geom + detail_gate * h_detail
```

Heads：

```text
head_means(h_geom)
head_scales(h_geom)
head_quat(h_geom)
head_noop(h_geom)
head_hidden(h_geom)
head_confidence(h_geom)

head_opacity(h_app)
head_sh(h_app)
```

## 9.4 Detail adapter

```python
self.appearance_detail_adapter = nn.Sequential(
    nn.LayerNorm(8),
    nn.Linear(8, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),
)
```

最后一层建议小初始化：

```text
std = 1e-3
```

Branch gate：

```python
self.detail_gate_bg       = nn.Parameter(torch.tensor(0.10))
self.detail_gate_distant  = nn.Parameter(torch.tensor(0.10))
self.detail_gate_rigid    = nn.Parameter(torch.tensor(0.05))
```

使用有界形式：

```text
gate = sigmoid(raw_gate) * max_gate
```

推荐：

```text
max_gate = 1.0
```

## 9.5 Invalid child

```python
h_detail = h_detail * detail_valid[:, None]
```

不可见 child 退化成纯 GRLD event。

## 9.6 可选后续扩展

若 appearance-only 恢复纹理但边界位置仍不准确，可加入：

```text
means_detail_gate_max = 0.05
```

但必须作为单独 ablation，主线默认 0。

---

# 10. Parent state 与 repeat 生命周期

Parent stats incremental 路径保留：

```text
block enter：exact diagonal projector
repeat 内：child delta -> incremental parent stats
```

变化仅在 observation：

```text
旧：updated parent params -> parent raster/lifting
新：updated fine state -> fine raster/lifting
```

Parent params仍用于：

```text
parent token coords
parent param embedding
GRLD parent-child relation
未来 PTv3/Mamba token
```

每 repeat 顺序：

```text
1. 当前 fine state + current parent runtime
2. fine render for residual
3. FWHR fine alpha/T lifting
4. parent encoder
5. GRLD geometry event
6. child appearance detail
7. posterior update
8. fine apply_delta
9. 非最后 repeat：incremental parent stats update
```

最后 repeat 不更新 parent runtime。

---

# 11. 梯度边界

允许梯度：

```text
loss
 -> posterior updater
 -> GRLD
 -> parent encoder
 -> parent context feature
 -> residual UNet / DINO concat feature projection

loss
 -> opacity/SH heads
 -> appearance detail adapter
 -> child detail feature
 -> DetailHead2D
 -> residual UNet
```

停止梯度：

```text
alpha/T weights
fine means/scale/quat/opacity through lifting geometry
child_to_parent
parent stats/projector
DINO backbone / adapter
support/obs code
```

最终 fine render loss仍会通过：

```text
fine state recurrence
```

训练 earlier delta。

---

# 12. Validation 修正

## 12.1 Shape 独立评估

当前配置列出多个 shape，但 `rollouts_per_segment=1` 会导致只采样其中一个，而不是分别评估。

新增：

```yaml
iforward_validation:
  shape_eval_mode: independent_all
```

实现：

```python
for shape in configured_shapes:
    reset scene state
    create fixed-shape validation scheduler
    run rollout
    write metrics tagged by shape
```

不能让多个 shape 顺序共享 state。

## 12.2 主验证与 stress validation 分离

主质量验证：

```text
K=8
```

可选当前训练早期：

```text
K=4
```

长周期 stress：

```text
K=16 / 32 / 64
```

单独输出：

```text
iforward_stability_validation
```

不能使用 K64 图作为常规画质结论。

推荐：

```yaml
iforward_validation:
  rollout_shapes:
    - name: b1_r8
      repeats_per_block: 8

iforward_stability_validation:
  enable: true
  interval_steps: 10000
  shapes: [b1_r16, b1_r32, b1_r64]
```

## 12.3 Masked sky 诊断

当前 photometric loss 排除 sky，因此可视化需要同时输出：

```text
render_raw
render_masked_eval
mask_overlay
GT
```

指标：

```text
psnr_non_sky_non_egocar
ssim_non_sky_non_egocar
psnr_full_image_diag
masked_pixel_ratio
```

避免把未监督 sky 区域当作 current optimizer 的主要质量证据。

---

# 13. Scheduler 修正

FWHR 初始训练仍应控制 recurrent difficulty。

## 13.1 Episode

推荐保持当前修正后的：

```yaml
episode:
  blocks_per_episode: 4
  episode_stride: 4
  rollouts_per_episode: 4
```

避免一个 segment 连续 12 次 optimizer update。

## 13.2 K curriculum

FWHR 引入新 observation 后，不建议立即随机 K2～K10。

Phase A，0～5k：

```yaml
shapes:
  - b1_r4: 1.0
```

Phase B，5k～20k：

```text
K2: 0.15
K4: 0.45
K6: 0.30
K8: 0.10
```

Phase C，稳定后：

```text
K2～K10
```

暂不加入 K16 作为训练 shape。

## 13.3 Preload fetch spike

修复 preload hint 构造：

```text
scheduler 主线程只产生轻量 asset IDs / frame refs；
不在 build_preload_hint 阶段同步 resolve segment bundle。
```

新增：

```text
build_preload_hint_light
```

由 preload worker 完成实际 asset load。

---

# 14. 推荐配置草案

```yaml
model:
  version: stage2_0_fwhr_lift_grld_dinov2base
  feat_2d_channels: 48

  feature_extractor:
    type: fwhr_dinov2_residual

    residual_unet:
      in_channels: 6
      feat_channels: 32
      base_channels: 32
      depth: 3
      bilinear: true
      norm: groupnorm
      norm_groups: 8

    dino:
      model_name: vit_base_patch14_reg4_dinov2
      out_channels: 16
      freeze: true
      freeze_adapter: true
      cache:
        enable: true
        dtype: float16
        cpu_max_items: 0
        gpu_max_items: 2

    context:
      residual_channels: 32
      dino_channels: 16
      out_channels: 48
      mode: direct_concat

    detail:
      source: residual
      out_channels: 8
      norm: groupnorm
      activation: silu

  iforward:
    biggs:
      observe:
        rendering_scene: fine
        lifting_scene: fine
        parent_scene_for_cnn: false
        parent_scene_for_lifting: false

      assignment:
        build_whdd_basis: false
        cache_scope: scene_segment_topology

      parent_state:
        mode: incremental_sufficient_stats
        exact_refresh_policy: block_enter
        update_after_each_nonfinal_repeat: true

      lifting:
        type: fwhr
        context_channels: 48
        detail_channels: 8
        geometry_grad: false
        detail_centering: support_weighted_parent
        detail_support_min: 1.0e-4
        fused_cuda: true

      child_decoder:
        mode: gaussian_relational
        relation_normalization: sibling_rms
        rigid_relation_space: canonical

  stage6_0:
    struct_event_decoder:
      feat_2d_dim: 48
      event_dim: 64

    posterior_updater:
      event_dim: 16
      hidden_dim: 32
      appearance_detail:
        enable: true
        input_dim: 8
        inject_heads: [opacity, sh]
        gate_init:
          bg: 0.10
          distant: 0.10
          rigid: 0.05
        geometry_gate: 0.0

scheduler_iforward:
  episode:
    blocks_per_episode: 4
    episode_stride: 4
    rollouts_per_episode: 4

  rollout:
    fixed_shape_names: []
    shapes:
      - name: b1_r4
        blocks_per_rollout: 1
        repeats_per_block: 4
        prob: 1.0

iforward_validation:
  shape_eval_mode: independent_all
  rollout_shapes:
    - name: b1_r8
      blocks_per_rollout: 1
      repeats_per_block: 8
      prob: 1.0
```

---

# 15. 文件级改动清单

## 15.1 新增

```text
models/feature_extractors/fwhr_image_features.py
models/feature_extractors/fwhr_lifting_extractor.py
models/iforward/fwhr_detail.py

gsplat/cuda/csrc/RasterizeAndBackprojectFWHRMulti.cu
```

## 15.2 修改

```text
models/feature_extractors/image_feature_extractor.py
    - normalization configurable
    - default GroupNorm

models/feature_extractors/dinov2_residual_concat.py
    - expose residual32 / dino16
    - integrate detail8 head or delegate to FWHR extractor

models/streetforward/minimal_trainer_stage4_5.py
    - fine scene render for CNN
    - return FWHR image features

models/streetforward/minimal_trainer_stage6_0.py
    - construct fine active scene
    - build global child_to_parent
    - call FWHR lifting
    - build parent event + detail pack

models/iforward/biggs_event_decoder.py
    - return EventPack + AppearanceDetailPack

models/streetforward/stage6_0/posterior_updater.py
    - appearance detail adapter
    - opacity/SH dual-stream heads

gsplat/cuda/_wrapper.py
    - FWHR Python wrapper

gsplat/cuda/csrc/Rasterization.cpp
Rasterization.h
    - FWHR binding

tools/train_iforward.py
models/iforward/validation.py
    - independent-all shape validation

datasets/train_scheduler_iforward.py
    - light preload hint
```

---

# 16. 单元测试

## 16.1 CUDA forward correctness

Tiny scene reference：

```text
1. 使用现有 V4 生成 full child context [N,48]
2. PyTorch scatter pooling 成 parent context
3. 使用现有 V4 生成 child detail [N,8]
4. PyTorch weighted centering
5. 与 FWHR fused 输出比较
```

误差标准：

```text
FP32 max abs < 1e-4
mean abs < 1e-5
```

## 16.2 Backward correctness

对比：

```text
reference child lifting + scatter
vs
FWHR custom autograd
```

检查：

```text
grad context_feat2d
grad detail_feat2d
```

不得产生：

```text
grad fine geometry
```

## 16.3 Detail conservation

```text
Σ_i support_i * child_detail_residual_i ≈ 0
```

按 parent 检查最大误差。

## 16.4 Singleton / invisible parent

```text
single visible child -> centered detail = 0
all child invisible -> parent context/detail = 0, valid false
```

## 16.5 Branch row alignment

检查：

```text
bg rows
 distant rows
 rigid route.S rows
 parent global offsets
```

防止 detail 写到错误 child。

## 16.6 GroupNorm train/eval consistency

同一输入：

```text
model.train() no optimizer update
model.eval()
```

Residual U-Net 输出差异应仅来自非 normalization 随机项；当前无 dropout 时应基本一致。

## 16.7 Validation shape isolation

配置 K8/K16：

```text
两者必须各运行一次；
各自从 fresh state 开始；
metrics 中同时出现两个 shape tag。
```

---

# 17. 集成测试与 ablation

不使用 teacher。

建议顺序：

```text
A. current parent-only baseline
B. fine scene render + parent-only lifting
C. fine alpha/T -> parent context48 only
D. FWHR parent48 + child detail4
E. FWHR parent48 + child detail8
F. FWHR parent48 + child detail16
```

关键判断：

```text
B > A：parent render residual 是重要问题
C > B：parent alpha/T approximation 是重要问题
E > C：child-specific image evidence 是主要高频瓶颈
E ≈ F：8D 足够
```

Appearance injection ablation：

```text
1. opacity + SH
2. SH only
3. opacity only
4. all heads（只作为上界，不建议主线）
```

---

# 18. 日志

## FWHR operator

```text
iforward/fwhr/fine_num_gaussians
iforward/fwhr/num_parents
iforward/fwhr/context_channels
iforward/fwhr/detail_channels
iforward/fwhr/raster_ms
iforward/fwhr/center_ms
iforward/fwhr/backward_ms
iforward/fwhr/pairs_total
iforward/fwhr/pairs_after_threshold
```

## Parent context

```text
iforward/fwhr/parent_context_norm
iforward/fwhr/parent_support_mean
iforward/fwhr/parent_support_p95
iforward/fwhr/parent_valid_ratio
iforward/fwhr/parent_obs_overlap_mean
```

## Child detail

```text
iforward/fwhr/child_detail_norm
iforward/fwhr/child_detail_residual_norm
iforward/fwhr/child_detail_valid_ratio
iforward/fwhr/detail_weighted_mean_error
iforward/fwhr/detail_to_geometry_event_ratio
```

## Updater

```text
iforward/posterior/noop_mean_bg
iforward/posterior/noop_mean_distant
iforward/posterior/noop_mean_rigid
iforward/posterior/appearance_detail_gate_*
iforward/posterior/detail_hidden_norm_*
```

## Image feature branches

```text
iforward/image/residual32_rms
iforward/image/dino16_rms
iforward/image/detail8_rms
iforward/image/lifted_residual_part_rms
iforward/image/lifted_dino_part_rms
```

## Validation

```text
iforward_validation/shape
iforward_validation/current_psnr
iforward_validation/current_ssim
iforward_validation/masked_ratio
iforward_stability_validation/*
```

---

# 19. 性能与显存预期

以：

```text
N = 500k
M = 30k
```

FP16 feature storage：

```text
parent context48：30k × 48 × 2 ≈ 2.9 MB
child detail8：500k × 8 × 2 ≈ 8.0 MB
合计约 10.9 MB
```

完整 child48：

```text
500k × 48 × 2 ≈ 48 MB
```

FWHR feature output约为完整 child48 的 23%。

但 fine rasterization pair 数会高于 parent-only，所以 observe kernel 会变慢。验收重点是：

```text
质量恢复是否显著；
总 step time 是否仍明显低于 Stage 1；
显存是否满足后续 parent PTv3/Mamba。
```

---

# 20. 验收标准

## 正确性

```text
- FWHR parent context 与 fine child context scatter reference 对齐。
- Child detail weighted mean error < 1e-4。
- DINO/residual/detail 均有正确 channel shape。
- Geometry tensors无 lifting gradient。
- Residual U-Net / detail head / parent encoder / GRLD / updater梯度非零。
```

## 画质

Matched K8 validation：

```text
- 高频边缘和纹理明显优于 parent-only；
- SSIM 不再随合理 K 增加而明显下降；
- PSNR 不低于 parent-only；
- Train/eval 不再出现 BatchNorm 式大幅分布偏移。
```

## 性能

```text
- 不存在 child [N,48] 持久输出。
- 不存在 parent 第二次 rasterization。
- FWHR 输出 feature memory < full child48 的 30%。
- Rigid active assignment 无 Python per-child loop。
```

---

# 21. 实施顺序

## Phase 0：先修诊断

```text
1. Validation shape independent-all。
2. 主 validation 固定 K8。
3. Residual U-Net BatchNorm -> GroupNorm。
4. masked sky 可视化分离。
```

先用原 parent-only checkpoint复测，确认图2中 validation异常有多少来自 K64/BN。

## Phase 1：PyTorch/reference FWHR

在小规模 scene：

```text
现有 full child V4 lifting
+ PyTorch parent scatter
+ child detail centering
```

仅用于确认质量上界和接口；不作为正式热路径。

## Phase 2：FWHR CUDA forward/backward

```text
1. parent context direct accumulation
2. child detail direct accumulation
3. parent obs code
4. detail centering
5. feature-only backward
```

## Phase 3：Posterior appearance injection

```text
child detail8 -> hidden32
只接 opacity/SH heads
```

## Phase 4：训练 curriculum

```text
K4 warm start
再逐渐加入 K2/K6/K8
```

## Phase 5：性能优化

仅在 correctness/质量成立后优化：

```text
atomic contention
FP16 feature input
shared-memory parent reduction
CUDA centering融合
```

---

# 22. 最终架构定位

FWHR-Lift 不是退回 Stage 1。

Stage 1 的基本形态是：

```text
fine child full feature
+ fine child 3D reasoning
```

FWHR 是：

```text
fine child alpha/T evidence
    ├─ full context压缩到 parent
    └─ 仅低维高频残差留给 child

重型 3D reasoning仍在 parent
```

因此它保留了 Stage 2 的核心价值：

```text
低 token parent attention / PTv3 / Mamba
```

同时修复 parent-only observation 的不可逆高频信息损失。
