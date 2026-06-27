# IForward Stage 3_0：Full Sparse Gather Lift 详细实现方案

目标版本：`iforward_stage3_0_full_sparse_gather_lift`  
基线代码：`drivestudio_stage6_refactor_context_20260627_v36`  
策略：**不实现 QDG-Child 过渡版；直接替换当前 FWHR feature-scatter lifting。**

---

# 0. 决策摘要

当前 FWHR 仍然把 2D feature 当作可被 GS alpha/T 搬运的大通道特征：

```text
context_2d 48D + detail_2d 8D
    -> concat 56D
    -> fine GS alpha/T fused backproject
    -> child_feature_sum [N,56]
    -> parent_context [M,48] + child_detail [N,8]
```

这个设计质量稳定，但其本质仍是：

```text
对每个 repeat、每个可见 fine GS、每个像素贡献 pair 搬运 56D feature。
```

Stage 3_0 改为：

```text
Full Sparse Gather Lift
    gsplat 只生成 scalar anchor / support / visibility / projection stats
    parent 和 child 都作为 query，从 2D feature map 中 sparse gather
```

最终仍然输出当前下游需要的接口：

```text
parent_context [M,48]
child_detail   [N, 8]
parent_support [M]
child_valid    [N]
```

但来源变成：

```text
Parent context：
    parent query -> sparse gather context_2d

Child detail：
    child query -> sparse gather detail_2d
```

核心目标：

```text
1. 消除 child_feature_sum [N,56] 及其大规模 autograd graph。
2. 消除特征通道在 gsplat raster/backproject CUDA 中的大搬运。
3. 让 2D observation 与 Optimizer Mamba / repeat / repair 状态关联。
4. 保留 GS projection、visibility、support 作为强几何约束。
5. 不引入 N×patch full attention。
```

核心约束：

```text
- 不做 full patch↔GS attention。
- 不做 dense BEV grid。
- Stage 3_0 不再调用当前 FWHR child 56D feature backproject。
- gsplat 不对 context/detail feature 反传，只输出 scalar anchor stats。
- 2D frontend 仍可训练，梯度来自 sparse gather 的少量采样点。
- Alpha/T geometry 和 GS projection 默认 stop-gradient。
```

---

# 1. 当前代码中的瓶颈

## 1.1 FWHR 现状

当前 Stage 2_x 主要路径在：

```text
models/streetforward/minimal_trainer_stage6_0.py
    _stage2_0_fwhr_lift_from_fine_scene()

models/iforward/fwhr_lift.py
    aggregate_fwhr_child_lift()

models/feature_extractors/alpha_t_extractor_v4.py
    render_and_backproject_streaming_fused_multi_camera()
```

关键代码逻辑：

```python
feat_2d = torch.cat([context_2d, detail_2d], dim=-1)  # 56D

child_feature_sum, child_weight_sum_feature, child_support = \
    _backproject_scene_features_multi_camera(..., features_2d=feat_2d, return_raw_lift=True)

aggregate_fwhr_child_lift(
    child_feature_sum=[N,56],
    child_to_parent=[N],
)
```

这导致：

```text
1. 每个 repeat 都生成 [N,56] raw feature sums。
2. context 与 detail 被绑定在同一次 56D lift 中。
3. gsplat CUDA backward 需要把 grad_feat_sum 回传到 2D feature map。
4. K8/B2 等 recurrent graph 会保留多次大 feature lifting graph。
5. 在 48GB 显存上已经贴边，repair/final render/SSIM 很容易触发 OOM。
```

## 1.2 当前 gsplat wrapper 语义

当前 v4 wrapper 是：

```text
rasterize_and_backproject_multi_camera_obs_in_range
backproject_feature_grad_multi_camera_sharded_in_range
```

它的功能是：

```text
for each pixel/gaussian pair:
    w = T * alpha
    feat_sum[gaussian] += w * feat2d[pixel]
    weight_sum[gaussian] += w
```

这对 FWHR 很合适，但对 Stage 3_0 目标过重。Stage 3_0 需要的是：

```text
for each pixel/gaussian pair:
    只累计 scalar visibility / support / weighted uv / depth
```

而不是搬 56D value。

---

# 2. 文献启发与适配判断

Stage 3_0 借鉴的是 sparse query + reference point + deformable gather，而不是 full attention。

相关对应：

```text
DETR3D：
    3D query 通过相机矩阵投影到多视角2D特征采样。

Deformable DETR：
    query 只围绕 reference point attend 少量 sampling points。

Sparse4D：
    3D anchor 分配多个4D keypoints，投影到多view/multiscale/temporal image features sparse sampling。

PETR/PETRv2：
    image feature 中加入3D position-aware embedding，避免 sample 到的2D feature缺少几何语义。
```

对 IForward 的转译：

```text
parent GS / child GS = sparse 3D query
GS projection / support = reference anchor
context_2d / detail_2d = image value maps
Parent Optimizer Mamba = optimizer state conditioning
sparse gather = deformable sampling
```

不采用 full attention 的原因：

```text
N_child ≈ 5e5
patch 7×7 = 49
N×patch attention matrix 极大
且 full attention 仍需要 query feature，容易绕回 lifting。
```

---

# 3. Stage 3_0 总体数据流

完整 forward observe：

```text
Fine GS current state
    │
    ├─ RGB render source image
    │      -> residual image / 2D frontend
    │
    ├─ 2D frontend
    │      ├─ context_2d [V,Hf,Wf,48]
    │      └─ detail_2d  [V,Hf,Wf, 8]
    │
    ├─ gsplat scalar anchor pass
    │      ├─ child_anchor_stats
    │      └─ parent_anchor_stats
    │
    ├─ Parent sparse gather
    │      parent_query + context_2d + parent_anchor_stats
    │      -> parent_context [M,48]
    │
    ├─ Parent event
    │      parent_context + parent params + support
    │      -> Parent PTv3
    │      -> Parent Optimizer Mamba read/fusion
    │      -> parent_event [M,64]
    │
    ├─ Child sparse gather
    │      child_query(parent_event, child relation, anchor stats)
    │      + detail_2d
    │      -> child_detail [N,8]
    │
    ├─ GRLD
    │      parent_event + child relation
    │      -> child_event [N,16]
    │
    └─ posterior updater
           child_event + child_detail
           -> delta
```

下游接口保持接近当前：

```text
parent_feat_2d_bg / distant / rigid
parent_acc_w_bg / distant / rigid
child_detail_bg / distant / rigid
child_detail_valid_bg / distant / rigid
```

但其来源不再是 FWHR feature scatter。

---

# 4. 新核心模块

新增 package：

```text
models/iforward/stage3_0/
    __init__.py
    scalar_anchor.py
    sparse_gather_lift.py
    parent_query.py
    child_query.py
    geometry_pe.py
    sparse_grid_sample.py
    gather_outputs.py
    losses.py
```

新增 feature extractor wrapper：

```text
models/feature_extractors/stage3_sparse_gather_features.py
```

新增 alpha/gsplat wrapper：

```text
models/feature_extractors/alpha_t_extractor_v5_scalar_anchor.py
```

---

# 5. Scalar Anchor Pass

## 5.1 目标

gsplat 不再负责搬特征，而负责输出：

```text
child_anchor_stats
parent_anchor_stats
```

这些 stats 必须包含：

```python
@dataclass
class SparseAnchorStats:
    child_uv: Tensor              # [N,V,2]
    child_support: Tensor         # [N,V]
    child_valid: Tensor           # [N,V]
    child_depth: Tensor           # [N,V]
    child_radius: Tensor          # [N,V]
    child_conic: Tensor           # [N,V,3]
    child_ray: Tensor             # [N,V,3]

    parent_uv: Tensor             # [M,V,2]
    parent_support: Tensor        # [M,V]
    parent_valid: Tensor          # [M,V]
    parent_depth: Tensor          # [M,V]
    parent_radius: Tensor         # [M,V]
    parent_conic_approx: Tensor   # [M,V,3]

    child_support_total: Tensor   # [N]
    parent_support_total: Tensor  # [M]
```

其中 parent stats 由 fine child stats 按 assignment 聚合，而不是依赖 parent GS render 近似。

## 5.2 P0 CUDA接口

扩展当前 v4 CUDA wrapper，新增：

```python
rasterize_scalar_anchor_multi_camera_in_range(
    means2d,
    conics,
    opacities,
    depths,
    isect_offsets,
    flatten_ids,
    packed_global_gaussian_ids,
    child_to_parent,
    num_children,
    num_parents,
    pair_valid_mask,
    weight_threshold,
    return_parent=True,
)
```

输出：

```text
child_support_view      [N,V]
child_uv_sum_view       [N,V,2]
child_depth_sum_view    [N,V]
child_radius_view       [N,V]
child_conic_view        [N,V,3]
parent_support_view     [M,V]
parent_uv_sum_view      [M,V,2]
parent_depth_sum_view   [M,V]
parent_radius_view      [M,V]
```

Kernel 内：

```cpp
w = alpha * T
child = packed_global_gaussian_ids[g]
parent = child_to_parent[child]
view = camera_id

atomicAdd(child_support[child,view], w)
atomicAdd(child_uv_sum[child,view], w * pixel_uv)
atomicAdd(child_depth_sum[child,view], w * depth)

atomicAdd(parent_support[parent,view], w)
atomicAdd(parent_uv_sum[parent,view], w * pixel_uv)
atomicAdd(parent_depth_sum[parent,view], w * depth)
```

这些输出默认 non-differentiable：

```text
不回传到 means2d / conics / opacity / geometry
不回传到 image feature
```

只作为 gather anchor。

## 5.3 为什么需要 scalar pass

如果只用投影中心：

```text
uv = project(mean)
```

会缺少：

```text
occlusion
view可见性
有效support
parent内child可见分布
```

FWHR的优势在于 alpha/T support 准确；Stage 3_0 仍然保留这一点，只是不搬 feature channels。

## 5.4 P0实现可选降级

若 CUDA暂未完成，可临时：

```text
1. 使用已有 V4 meta 提供 child projected uv/conic。
2. view_support 使用投影可见性近似。
3. parent_support 由 child projected valid scatter 得到。
```

但该模式只能用于 smoke，不建议作为正式质量实验，因为没有真实 alpha/T occlusion。

---

# 6. Sparse Gather Lift

## 6.1 Parent Gather

输入：

```python
parent_query_input:
    parent gaussian params
    parent support / view support
    parent branch embedding
    optimizer memory preview / state summary
    visit metadata
```

输出 query：

```text
q_parent [M,Q]
```

采样：

```text
context_2d [V,H,W,48]
anchor uv [M,V,2]
```

Parent gather 预测：

```text
view logits      [M,V]
tap offsets      [M,V,Kp,2]
tap logits       [M,V,Kp]
feature gate     [M,48]
```

结果：

```python
parent_context[p] = Σ_v Σ_k softmax(logits[p,v,k]) *
                    sample(context_2d[v], uv[p,v] + offset[p,v,k])
```

推荐 P0：

```text
Kp = 5
center + four screen-axis taps
learned offset residual zero-init
```

## 6.2 Child Gather

输入：

```python
child_query_input:
    child-parent relative geometry
    parent_event[parent_id]
    parent optimizer event / support
    child scalar anchor stats
    visit metadata
```

输出：

```text
q_child [N,Q]
```

采样：

```text
detail_2d [V,H,W,8]
anchor uv [N,V,2]
```

输出：

```text
child_detail_raw [N,8]
```

然后进行 parent 内 centering：

```python
child_detail = child_detail_raw - weighted_mean_parent(child_detail_raw)
```

权重：

```text
child scalar support × gather confidence
```

## 6.3 Parent 与 Child 的时间顺序

Full Sparse Gather 必须拆成两段，避免循环依赖：

```text
1. Parent gather：只用 parent geometry/support/optimizer prior/visit meta。
2. Parent event：PTv3 + Optimizer Mamba read/fusion。
3. Child gather：使用 parent_event 和 child relation。
```

不要让 parent gather 依赖 parent_event，否则会形成循环。

---

# 7. Geometry PE

从 PETR/PETRv2 借鉴 position-aware feature 的思想，sampled 2D feature 不应裸用。

每个 sampled point 的 PE：

```text
view embedding
normalized uv
ray direction
depth
log radius
support
branch embedding
relative child-parent xyz/cov
visit kind / repeat idx
```

实现：

```python
value = sampled_feature + Linear(geometry_pe)
```

或：

```python
value = MLP(concat(sampled_feature, geometry_pe))
```

P0 推荐 additive PE，节省显存。

---

# 8. Sparse sampling实现

## 8.1 P0：chunked grid_sample

P0 使用 PyTorch：

```python
torch.nn.functional.grid_sample
```

但必须 chunk：

```text
parent chunk：32768 rows
child chunk：32768 / 65536 rows
```

避免一次生成：

```text
[N,V,K,8]
```

巨大中间 tensor。

## 8.2 P1：自定义 CUDA sparse bilinear gather

新增 op：

```text
sparse_bilinear_gather_forward
sparse_bilinear_gather_backward
```

输入：

```text
value_map [V,H,W,C]
uv        [R,V,K,2]
weights   [R,V,K]
valid     [R,V]
```

输出：

```text
out [R,C]
```

反向：

```text
grad_value_map
可选 grad_uv / grad_weights
```

P1 优先支持：

```text
grad_value_map
grad_weights
grad_offsets
```

但 geometry anchor本身仍 detach。

---

# 9. Query Heads 初始化

必须 zero/near-center init，避免训练初期乱采。

```python
offset_head.weight = 0
offset_head.bias = 0

weight_head.bias:
    center tap = +2
    side taps = 0

view logits:
    add log(view_support + eps)

gate_head.bias = 0
```

初始行为接近：

```text
从anchor中心根据support加权采样
```

然后逐渐学习 offset/tap/view 权重。

Offset 限制：

```python
offset = tanh(raw_offset) * radius * offset_scale
```

推荐：

```text
parent offset_scale = 0.5
child offset_scale  = 0.75
max pixel offset    = 8 px
```

---

# 10. 观察路径重构

当前：

```text
_compute_biggs_features...
    fine_scene render
    2D frontend
    FWHR feature lift
    parent_context + child_detail
```

Stage 3_0：

```text
_compute_stage3_0_sparse_gather_features(...)
    fine_scene render
    2D frontend -> context_2d/detail_2d
    scalar_anchor_pass(fine_scene, child_to_parent)
    parent_sparse_gather(context_2d, parent_anchor_stats, parent_query)
    parent_event_builder/PTv3/Mamba handled in model path
    child_sparse_gather(detail_2d, child_anchor_stats, parent_event)
```

由于 child gather 需要 parent_event，建议把 Stage 3_0 observe 拆成：

```text
observe_pre_parent:
    returns context_2d, detail_2d, anchor_stats, parent_context

model parent event:
    parent_context -> parent_event

observe_child_detail:
    parent_event + anchor_stats + detail_2d -> child_detail
```

如果改动太大，可将 child gather 放在 `biggs_event_decoder` 内部：

```text
BigGSEventDecoder.forward(..., detail_2d, anchor_stats)
```

但更干净的是新增 Stage3 measurement object。

---

# 11. 接口设计

```python
@dataclass
class Stage3SparseGatherMeasurement:
    parent_context_bg: Tensor
    parent_context_distant: Tensor
    parent_context_rigid: Tensor
    parent_support_bg: Tensor
    parent_support_distant: Tensor
    parent_support_rigid: Tensor

    anchor_stats: SparseAnchorStats
    detail_2d: Tensor
    context_2d: Tensor
    child_to_parent_global: Tensor

    route: Any
    biggs_state: IForwardBigGSState
    parent_runtime: BigGSParentRuntime
```

Child detail deferred output：

```python
@dataclass
class Stage3ChildDetailPack:
    child_detail_bg: Tensor
    child_detail_distant: Tensor
    child_detail_rigid: Tensor
    child_detail_valid_bg: Tensor
    child_detail_valid_distant: Tensor
    child_detail_valid_rigid: Tensor
    aux: Dict[str, float]
```

Model flow：

```python
measurement = bridge.observe_stage3_parent(...)
parent_event = parent_event_encoder(measurement.parent_context, ...)
child_detail = bridge.gather_stage3_child_detail(measurement, parent_event)
child_event = GRLD(parent_event, relation)
delta = posterior(child_event, child_detail)
```

---

# 12. 配置草案

```yaml
model:
  iforward:
    version: stage3_0_full_sparse_gather_lift

    lifting:
      type: full_sparse_gather
      disable_fwhr_feature_backproject: true
      scalar_anchor_backend: cuda_scalar_anchor
      context_dim: 48
      detail_dim: 8

      parent_gather:
        num_taps: 5
        offset_scale: 0.5
        max_offset_px: 8.0
        query_dim: 96
        use_optimizer_prior: true
        use_geometry_pe: true
        chunk_size: 32768

      child_gather:
        num_taps: 5
        offset_scale: 0.75
        max_offset_px: 8.0
        query_dim: 128
        use_parent_event: true
        use_geometry_pe: true
        center_by_parent: true
        chunk_size: 65536

      scalar_anchor:
        return_child_view_support: true
        return_parent_view_support: true
        support_threshold:
          parent: 1.0e-4
          child: 1.0e-4
        detach_geometry: true

      regularization:
        offset_l2: 1.0e-4
        offset_radius_bound: 1.0
        view_entropy_min_weight: 0.0
        tap_entropy_min_weight: 0.0

      debug:
        shadow_fwhr: false
        log_gather_stats: true
```

---

# 13. 训练阶段

## Stage 3_0-S0：fixed sparse gather smoke

```text
offset固定为0
tap只用center
view权重来自support
训练query gate和下游
```

目标：确认不使用 FWHR feature scatter 也能跑通。

## Stage 3_0-S1：train tap/view weights

```text
offset仍冻结
训练view/tap logits
```

目标：让模型学多视角权重。

## Stage 3_0-S2：train offsets

```text
打开offset_head
offset_scale从0.1 warmup到0.75
```

目标：学习 deformable correction。

## Stage 3_0-S3：full training

```text
接入 Scheduler v3 / IForward 2_3
训练完整 current/history/repair
```

---

# 14. Loss 与正则

主 loss 不变：

```text
current/history/damage/render losses
```

新增 gather 正则：

```text
offset_l2
out_of_bounds penalty
support-weight alignment
view entropy optional
```

建议初期只用：

```text
offset_l2 = 1e-4
out_of_bounds penalty = 1e-3
```

不要一开始加复杂 entropy，避免限制模型自适应选择 view。

---

# 15. Validation

必须新增 Stage3 gather-specific validation：

```text
1. current PSNR / history PSNR / repair gain
2. gather in-bound ratio
3. parent_context RMS
4. child_detail RMS
5. offset magnitude mean/p95
6. view entropy
7. tap entropy
8. valid child ratio
9. parent support valid ratio
10. peak memory / observe time
```

与 Stage2 FWHR 对比：

```text
Stage2 FWHR baseline
Stage3 fixed center gather
Stage3 learned view/tap
Stage3 learned offset
```

最重要验证：

```text
同等 scheduler 下：
    observe time下降
    peak memory下降
    PSNR不显著下降
```

---

# 16. 日志

新增：

```text
iforward/stage3/scalar_anchor_ms
iforward/stage3/parent_gather_ms
iforward/stage3/child_gather_ms
iforward/stage3/parent_num_taps
iforward/stage3/child_num_taps
iforward/stage3/parent_inbound_ratio
iforward/stage3/child_inbound_ratio
iforward/stage3/parent_offset_norm_mean/p95
iforward/stage3/child_offset_norm_mean/p95
iforward/stage3/view_entropy_parent/child
iforward/stage3/tap_entropy_parent/child
iforward/stage3/parent_context_rms
iforward/stage3/child_detail_rms
iforward/stage3/peak_alloc_after_anchor
iforward/stage3/peak_alloc_after_parent_gather
iforward/stage3/peak_alloc_after_child_gather
```

必须保留旧 FWHR 关键指标的对照名：

```text
iforward/fwhr/enabled = 0
iforward/stage3/enabled = 1
```

---

# 17. 测试计划

## Unit tests

```text
test_scalar_anchor_shapes
test_scalar_anchor_parent_sum_matches_child_sum
test_sparse_gather_center_equiv_grid_sample
test_sparse_gather_chunk_equivalence
test_offset_zero_init_center_sample
test_parent_child_split_shapes
test_child_detail_centering_weighted_zero_mean
test_geometry_detached
test_gather_grad_to_2d_features
test_no_fwhr_feature_backproject_called
```

## CUDA tests

```text
test_scalar_anchor_cuda_matches_reference_small_scene
test_scalar_anchor_masked_pixels
test_scalar_anchor_view_support_sum
test_sparse_cuda_gather_matches_torch
test_sparse_cuda_backward_feature_grad
```

## Integration tests

```text
test_stage3_observe_smoke_b1r4
test_stage3_scheduler_v3_smoke
test_stage3_repair_smoke
test_stage3_validation_smoke
test_stage3_memory_less_than_fwhr_reference
```

---

# 18. 迁移步骤

## Phase 1：新增模块但不接入训练

```text
- ScalarAnchor dataclass
- SparseGather modules
- PyTorch gather reference
- unit tests
```

## Phase 2：接入 parent gather，child仍旧FWHR detail

只用于 debug，不作为正式实验。

## Phase 3：接入 full sparse gather

```text
parent_context = sparse gather
child_detail = sparse gather
FWHR feature backproject disabled
```

## Phase 4：CUDA scalar anchor

替换投影近似/临时实现。

## Phase 5：CUDA sparse gather

若 PyTorch gather 成本仍高，再写 CUDA gather。

---

# 19. 风险与应对

## 19.1 Parent context 质量下降

Full sparse gather 最容易出问题的是 parent context。

应对：

```text
- parent K=5 起步，不用K=1
- support view prior 强约束
- offset zero-init
- parent query不直接大offset
- parent_context RMS 和 render PSNR 监控
```

## 19.2 Child detail 不稳定

应对：

```text
- child detail parent centering
- offset_l2
- valid support threshold
- detail gate 保守初始化
```

## 19.3 Query 乱采

应对：

```text
- offset tanh + radius bound
- out-of-bound penalty
- sample in-bound ratio fail-fast
```

## 19.4 scalar anchor 过慢

应对：

```text
- 不做feature channel搬运
- debug pair recount关闭
- view support只accumulate float scalars
- 后续CUDA优化tile-local reduce
```

---

# 20. 验收标准

相对 Stage2 FWHR baseline：

```text
observe time：下降 >= 20%
peak allocated：下降 >= 1.5GB
current PSNR：下降 <= 0.3dB
history PSNR：下降 <= 0.5dB
repair gain：不下降超过0.3dB
repeat stability：不恶化超过0.5dB
```

如果质量下降超过阈值：

```text
先增加 parent taps / geometry PE
再考虑 parent residual hybrid
不要回退到 child-only过渡方案
```

---

# 21. 最终判断

Stage 3_0 的正确目标不是：

```text
QDG-Child 只替换 child detail
```

而是：

```text
Full Sparse Gather Lift
    parent_context 和 child_detail 全部由 query-guided sparse gather 产生
    gsplat 只提供 scalar visibility / projection anchor
```

这才真正切断当前 FWHR 的核心负担：

```text
fine GS × pixel pair × 56D feature transport
```

同时保留 IForward 的核心思想：

```text
GS / parent token 是优化器状态
2D feature 是当前观测
Mamba / PTv3 / GRLD 作为 learned optimizer 的上下文
```

