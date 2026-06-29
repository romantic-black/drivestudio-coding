# IForward Stage 3_0：Full Sparse Gather Lift CUDA 与性能修复完整方案

基线代码：`drivestudio_stage6_refactor_context_20260628_v37`  
外部 CUDA 基线：`gsplat_source_20260616`  
目标版本：`iforward_stage3_0_full_sparse_gather_lift_cuda_v1`

---

# 0. 当前结论

当前 Full Sparse Gather Lift 的方向仍然成立，但现有 v37 实现不是最终形态。它的问题不是“sparse gather 理论上一定慢”，而是实现方式把原来 CUDA fused feature scatter 的隐式高效中间量，替换成了大量显式 PyTorch Tensor、Query MLP 和 `grid_sample` graph。

当前 metrics31 中，单步 R1 smoke 就已经出现：

```text
step_time_ms ≈ 2426 ms
forward_ms   ≈ 1663 ms
backward_ms  ≈ 708  ms
observe_ms   ≈ 526  ms
event_ms     ≈ 1040 ms
```

与 Stage2.3 R1 大致相比：

```text
observe 变快，但 event/gather 变慢更多；总 step 变慢。
```

核心原因：

```text
1. child query [N,96] 全量构建；
2. chunk_size=1024 造成数百次小 kernel / grid_sample；
3. 每个 chunk 重复 value_map permute/contiguous；
4. fixed-center warmup 仍然跑完整 query/head；
5. 无效 child / parent row 没有提前过滤；
6. projected_meta anchor 不含真实 alpha/T visibility，导致 valid/inbound 低；
7. PyTorch grid_sample 对 500k × views × taps 的场景不是理想执行路径。
```

所以后续方案分两层：

```text
A. 立即修 PyTorch 路径 P0，让它不再灾难性慢；
B. 实现 CUDA scalar anchor + CUDA sparse gather，真正替代 PyTorch gather。
```

---

# 1. 当前代码问题定位

## 1.1 当前路径

当前 Stage3 observe 主要路径在：

```text
models/streetforward/minimal_trainer_stage6_0.py
    _stage3_0_build_anchor_stats
    _stage3_0_parent_sparse_gather
    _stage3_0_gather_child_detail

models/iforward/stage3_0/scalar_anchor.py
models/iforward/stage3_0/sparse_gather_lift.py
models/iforward/stage3_0/sparse_grid_sample.py
```

Stage3 当前流程：

```text
2D frontend
    context_2d [V,H,W,48]
    detail_2d  [V,H,W,8]

scalar anchor backend = projected_meta
    child_uv / parent_uv / support / valid

parent_sparse_gather(context_2d)
    -> parent_context [M,48]

PTv3 + Optimizer Mamba + GRLD
    -> parent_event

child_sparse_gather(detail_2d)
    -> child_detail [N,8]
```

## 1.2 当前 metrics 暴露的问题

metrics31 中：

```text
scalar_anchor_ms ≈ 14.9 ms
meta_render_packed_total_ms ≈ 10.9 ms
meta_build_multi_meta_ms ≈ 10.9 ms
```

anchor 构建本身不是最大瓶颈。

真正异常是：

```text
event_ms ≈ 1040 ms
```

这说明 sparse gather/query 被算入 event 或 event-adjacent 路径后，成为主要新瓶颈。

同时，当前有效率偏低：

```text
parent_support_valid_ratio ≈ 0.55
child_support_valid_ratio  ≈ 0.46
child_bg_valid_ratio       ≈ 0.35
child_bg_inbound_ratio     ≈ 0.13
parent_bg_inbound_ratio    ≈ 0.20
```

也就是说大量 rows 其实不应该执行 gather。

---

# 2. 总体目标

Stage3_0 CUDA v1 的目标不是一步做到最优，而是做到：

```text
1. Full sparse gather 不再比 Stage2.3/FWHR 慢；
2. 显存不再高于 Stage2.3；
3. parent + child 都可以通过 sparse gather 提供观测；
4. 2D frontend 仍可训练；
5. geometry / alpha / anchor 默认 stop-gradient；
6. 后续可逐步替代 projected_meta 为 occlusion-aware scalar anchor。
```

量化验收：

```text
B1R1 smoke：
    step_time 不高于 Stage2.3 R1 +10%
    event_ms 降低至少 50%

B2R4 smoke：
    peak allocated 低于 Stage2.3/FWHR 或至少不高于 +0.5GB
    无 OOM

质量：
    current PSNR 相比 Stage2.3 下降 <=0.3 dB
    history PSNR 下降 <=0.5 dB
```

---

# 3. P0：不等 CUDA，先修 PyTorch 路径

这些修复必须先做，因为它们也会影响 CUDA 接口设计。

## 3.1 `value_map` 不允许每 chunk 重复 permute

当前风险点：

```python
value_nchw = value_map.permute(0, 3, 1, 2).contiguous()
```

如果在每个 chunk 内执行，child 500k、chunk 1024 时可能执行约 500 次。

修复：

```python
value_nchw = prepare_value_nchw(value_map)  # once per branch/map
for chunk:
    sampled = sparse_grid_sample_prepared(value_nchw, grid_chunk)
```

新增：

```text
models/iforward/stage3_0/sparse_grid_sample.py
    prepare_value_nchw
    sparse_grid_sample_prepared
```

## 3.2 valid row prefilter

当前：

```text
对所有 parent/child 运行 query + gather，最后无效行置零
```

修复：

```python
valid_rows = anchor.support_total >= threshold
query/gather 只对 valid_rows 执行
invalid rows 直接填 0
```

注意：

```text
valid row index 必须保留，以便 scatter 回完整 [N,C] / [M,C]
```

建议新增：

```python
SparseGatherRows(
    row_indices,
    anchor_uv_valid,
    support_valid,
    ...
)
```

## 3.3 query builder 必须 chunk 化

当前：

```python
q_child = ChildQueryBuilder(all_children)  # [N,96]
```

这会生成巨大的 MLP activation graph。

修复：

```python
for valid_child_chunk:
    q = child_query_builder(chunk)
    out = gather(q, chunk_anchor)
    scatter_to_output(chunk_indices, out)
```

不要在外部构建完整 `[N,query_dim]`。

Parent query 也可以 chunk，但优先级低于 child。

## 3.4 fixed-center fast path

当前 fixed-center 阶段：

```yaml
fixed_center_steps: 1000
```

但仍执行完整：

```text
query MLP
head MLP
offset/tap/view/gate heads
```

修复：

```python
if global_step < fixed_center_steps:
    return fixed_center_gather_fast(...)
```

fast path：

```text
center tap only
optional view prior fusion
no query MLP
no offset head
no gate head
no geometry MLP unless explicitly enabled
```

这对当前 smoke 阶段最重要。

## 3.5 chunk size 调整

在 P0 修完前，不要盲目增大 chunk。

P0 修完后测试：

```yaml
parent_gather.chunk_size: 8192
child_gather.chunk_size: 8192
```

再测试：

```yaml
16384
32768
```

最终目标是减少 kernel launch，同时不把 single-chunk activation 撑爆。

---

# 4. CUDA 部分总体设计

CUDA 分两块：

```text
A. Scalar Anchor CUDA：gsplat 只输出 geometry/visibility scalar，不搬 feature channel。
B. Sparse Gather CUDA：给定 anchor/offset/weight，从2D feature map采样并融合。
```

二者分开实现，方便定位和回退。

---

# 5. Scalar Anchor CUDA

## 5.1 现有 gsplat 参考

当前 `gsplat_source_20260616` 中有：

```text
gsplat/gsplat/cuda/csrc/RasterizeAndBackproject3DGSMulti.cu
```

它做：

```text
遍历 tile / pixel / gaussian intersection
计算 alpha 和 transmittance
vis = alpha * trans
对 feat_sum[g,c] atomicAdd(vis * feature)
对 weight_sum_feature[g] atomicAdd(vis)
对 weight_sum_support[g] atomicAdd(vis)
```

Stage3 scalar anchor 只保留：

```text
vis scalar accumulation
uv/depth/radius/conic/view support
```

不再采样 feature。

## 5.2 新增 CUDA 文件

建议新增：

```text
gsplat/gsplat/cuda/csrc/RasterizeScalarAnchor3DGSMulti.cu
gsplat/gsplat/cuda/csrc/RasterizeScalarAnchor3DGSMulti.h
```

并修改：

```text
gsplat/gsplat/cuda/csrc/Rasterization.cpp
gsplat/gsplat/cuda/csrc/Rasterization.h
gsplat/gsplat/cuda/ext.cpp
gsplat/gsplat/cuda/_wrapper.py
```

## 5.3 Forward 输入

```cpp
means2d                    [N_packed,2]
conics                     [N_packed,3]
opacities                  [N_packed]
depths                     [N_packed] 或 packed depths
radii                      [N_packed] 或 projected radii
isect_offsets              [V,tile_h,tile_w]
flatten_ids                [num_isects]
packed_global_gaussian_ids [N_packed]
pair_valid_mask            [V,H,W]
num_gaussians              int
num_views                  V
image_width, image_height
feat_width, feat_height    // 用于uv映射
support_threshold
return_view_support
```

## 5.4 Forward 输出

Child-level：

```text
support_sum         [N]
view_support        [N,V]
weighted_uv_sum     [N,V,2]
weighted_depth_sum  [N,V]
weighted_radius_sum [N,V]
valid_pair_count    [N,V]
```

Optional parent aggregation：

```text
child_to_parent     [N]
parent_support_sum  [M]
parent_view_support [M,V]
parent_uv_sum       [M,V,2]
parent_depth_sum    [M,V]
parent_radius_sum   [M,V]
```

建议 P1 先只输出 child-level，再在 Python scatter 到 parent。P2 再支持 kernel 内 parent aggregation。

## 5.5 Accumulation 公式

每个有效 pixel-gaussian pair：

```text
vis = alpha * transmittance
u_feat = pixel_u mapped to feature resolution
v_feat = pixel_v mapped to feature resolution
```

累积：

```cpp
atomicAdd(support[g], vis)
atomicAdd(view_support[g,v], vis)
atomicAdd(weighted_uv[g,v,0], vis * u_feat)
atomicAdd(weighted_uv[g,v,1], vis * v_feat)
atomicAdd(weighted_depth[g,v], vis * depth)
atomicAdd(weighted_radius[g,v], vis * radius)
```

归一化在 Python 或 CUDA 后处理：

```text
uv = weighted_uv / (view_support + eps)
depth = weighted_depth / (view_support + eps)
radius = weighted_radius / (view_support + eps)
```

## 5.6 Backward

默认无 backward：

```text
geometry / alpha / anchor stop-gradient
```

即：

```python
anchor tensors = anchor tensors.detach()
```

这与当前 `train_2d_detach_alpha` 语义一致。

## 5.7 性能优化

Scalar anchor 仍会遍历 raster pairs，但避免：

```text
channels loop
feature bilinear sample
feat_sum atomic per channel
feature backward kernel
```

相比当前 feature backproject，理论上更轻，尤其当 C=48/56 时。

---

# 6. Sparse Gather CUDA

## 6.1 目标

替代 PyTorch：

```text
grid_sample + weights + sum
```

避免：

```text
巨大 grid tensor
PyTorch grid_sample graph
chunk loop Python overhead
NCHW重复转换
```

## 6.2 新增 CUDA 文件

```text
gsplat/gsplat/cuda/csrc/SparseGather2DMulti.cu
gsplat/gsplat/cuda/csrc/SparseGather2DMulti.h
```

wrapper：

```text
gsplat/gsplat/cuda/_wrapper.py
    sparse_gather_2d_forward
    sparse_gather_2d_backward
```

PyTorch function：

```text
models/iforward/stage3_0/cuda_sparse_gather.py
```

## 6.3 Forward 输入

```text
feature_map    [V,H,W,C] float16/float32 contiguous HWC
uv             [R,V,K,2] float32 pixel/feature coords
weights        [R,V,K] float32
valid          [R,V,K] bool
row_indices    [R] long optional
```

其中 R 是 valid row 数，不是全 N。

输出：

```text
out [R,C]
inbound [R,V,K]
```

## 6.4 Forward kernel 映射

建议线程布局：

```text
blockIdx.x = row tile
blockIdx.y = channel tile
threadIdx.x = channel lane or row lane
```

初版简单实现：

```text
one block handles one row group and channel group
loop V*K
bilinear sample
weighted sum
```

对 C=8 和 C=48 分两种优化：

```text
C=8：一个 row block 处理所有 channels
C=48：channel tile 16 或 32
```

## 6.5 Backward 输出

需要梯度到：

```text
grad_feature_map [V,H,W,C]
grad_weights     [R,V,K]
grad_uv          [R,V,K,2]
```

因为：

```text
weights head需要训练
offset head需要训练
2D frontend需要训练
```

几何 anchor 本身 detach，但 learned offset 通过 uv 需要梯度。

## 6.6 Backward 公式

Bilinear sample：

```text
f(u,v) = Σ_ab w_ab(u,v) * F[pixel_ab]
```

给定：

```text
grad_out_c
sample_weight a
```

则：

```text
grad_feature[p] += a * w_ab * grad_out_c
grad_weight     += dot(sample_value, grad_out)
grad_uv         += a * dot(dF/du,dF/dv, grad_out)
```

`grad_feature` 使用 atomicAdd。

## 6.7 dtype

推荐：

```text
feature_map fp16/bf16 input
内部accumulate float32
out float32 或 input dtype可选
```

第一版全部 float32，确认正确后再支持 fp16。

---

# 7. Query/head CUDA 是否需要融合

第一版不建议把 Query MLP 融进 CUDA。MLP 留在 PyTorch，更容易训练和调试。

但必须 chunk query builder：

```python
for row_chunk:
    q = query_builder(rows)
    offsets, logits, gate = heads(q)
    out = sparse_gather_cuda(...)
```

避免全量 `[N,query_dim]`。

---

# 8. Stage3 model 结构修正

## 8.1 新数据结构

新增：

```python
@dataclass
class SparseGatherRows:
    row_indices: Tensor       # [R]
    uv: Tensor                # [R,V,2]
    support: Tensor           # [R,V]
    valid: Tensor             # [R,V]
    depth: Tensor             # [R,V]
    radius: Tensor            # [R,V]
```

新增：

```python
@dataclass
class SparseGatherOutput:
    full: Tensor              # [N,C] or [M,C]
    valid_mask: Tensor        # [N] or [M]
    confidence: Tensor        # [N] or [M]
    aux: Dict[str,float]
```

## 8.2 Parent gather 改法

```python
parent_rows = anchor.parent_valid_rows()
for chunk in parent_rows:
    q = ParentQueryBuilder(chunk)
    obs = ParentGatherCuda(context_2d, chunk_anchor, q)
    scatter parent_context[chunk] = obs
```

无效 parent：

```text
context = 0
confidence = 0
```

## 8.3 Child gather 改法

```python
child_rows = anchor.child_valid_rows()
for chunk in child_rows:
    q = ChildQueryBuilder(chunk)
    detail = ChildGatherCuda(detail_2d, chunk_anchor, q)
    scatter child_detail[chunk] = detail
```

然后：

```text
parent centering only over valid child rows
```

## 8.4 Fixed center fast path

```python
if global_step < fixed_center_steps:
    gather_center_only_cuda_or_prepared_grid_sample()
```

Fast path 不构造：

```text
query
head hidden
offset
view/tap logits
```

只使用：

```text
support-weighted view fusion
center/keypoint sample
```

---

# 9. Config 修改

```yaml
model:
  iforward:
    lifting:
      type: full_sparse_gather_cuda
      scalar_anchor_backend: cuda_scalar_anchor
      sparse_gather_backend: cuda
      detach_geometry: true

      scalar_anchor:
        support_threshold:
          bg: 1.0e-3
          distant: 1.0e-3
          rigid: 1.0e-3
        output_parent_aggregate: false

      parent_gather:
        chunk_size: 16384
        valid_row_filter: true
        fixed_center_fast_path: true
        backend: cuda

      child_gather:
        chunk_size: 16384
        valid_row_filter: true
        query_chunked: true
        fixed_center_fast_path: true
        backend: cuda

      training_schedule:
        fixed_center_steps: 1000
        train_weights_steps: 3000
        offset_warmup_steps: 5000
```

保留 debug fallback：

```yaml
fallback:
  allow_pytorch_gather: true
  compare_cuda_pytorch_every_n_steps: 1000
```

---

# 10. 日志新增

## Performance

```text
stage3/anchor_cuda_ms
stage3/anchor_normalize_ms
stage3/parent_query_ms
stage3/parent_gather_cuda_ms
stage3/child_query_ms
stage3/child_gather_cuda_ms
stage3/gather_total_ms
stage3/valid_filter_ms
stage3/fixed_fast_path_enabled
```

## Counts

```text
stage3/parent_rows_total
stage3/parent_rows_valid
stage3/child_rows_total
stage3/child_rows_valid
stage3/child_rows_valid_ratio
stage3/chunk_size_child
stage3/num_child_chunks
```

## Memory

```text
stage3/mem_before_anchor
stage3/mem_after_anchor
stage3/mem_after_parent_gather
stage3/mem_after_child_gather
stage3/mem_peak_gather
```

## Anchor quality

```text
stage3/anchor_support_mean
stage3/anchor_view_entropy
stage3/anchor_uv_oob_ratio
stage3/anchor_scalar_visibility_backend
```

---

# 11. 测试计划

## CUDA scalar anchor

```text
test_scalar_anchor_matches_feature_backproject_weight_sum_small_case
test_scalar_anchor_view_support_shape
test_scalar_anchor_no_feature_grad
test_scalar_anchor_parent_scatter_matches_python
test_scalar_anchor_pair_valid_mask
```

## CUDA sparse gather

```text
test_sparse_gather_forward_matches_grid_sample
test_sparse_gather_backward_feature_matches_grid_sample
test_sparse_gather_backward_uv_finite
test_sparse_gather_backward_weights_finite
test_sparse_gather_masked_invalid_zero
test_sparse_gather_fp16_forward_close
test_sparse_gather_chunk_equivalence
```

## Integration

```text
test_stage3_cuda_b1r1_smoke
test_stage3_cuda_b1r4_smoke
test_stage3_cuda_b2r4_smoke
test_stage3_fixed_center_skips_query
test_stage3_valid_filter_reduces_rows
test_stage3_no_N56_tensor_allocated
```

## Performance regression

固定小场景：

```text
assert step_time_stage3_cuda <= stage3_pytorch * 0.6
assert peak_mem_stage3_cuda <= stage3_pytorch - 1GB
```

---

# 12. 实施阶段

## Phase 1：PyTorch P0 修复

```text
- value_nchw 只转换一次；
- valid row prefilter；
- query chunking；
- fixed-center fast path；
- chunk_size调整到8192/16384。
```

目标：即使没有CUDA，也让 Stage3 不再灾难性慢。

## Phase 2：CUDA sparse gather

```text
- 实现 SparseGather2DMulti forward/backward；
- 接入 Parent/Child gather；
- 与 grid_sample 对齐。
```

目标：替代最重的 PyTorch grid_sample 路径。

## Phase 3：CUDA scalar anchor

```text
- 实现 RasterizeScalarAnchor3DGSMulti；
- 输出 child scalar support/uv/depth/radius；
- kernel 内同步输出 parent support/uv/depth/radius/conic raw sums；
- projected_meta 仅保留为显式 debug fallback。
```

目标：替代 projected_meta，并提供 occlusion-aware scalar visibility。

## Phase 4：联合优化

```text
- CUDA anchor + CUDA gather；
- row filter；
- fixed path；
- 非 fixed-center path profile；
- b1r1/b1r4/b1r8/b2r4 性能与显存 smoke；
- 不做长时间 validation/quality sweep。
```

## Phase 5：可选 parent aggregate in CUDA

已并入 Phase 3 的 scalar anchor CUDA kernel：

```text
child_to_parent aggregation -> parent raw sums
```

本阶段只补硬化项：

```text
- parent aggregate CUDA aux metrics；
- parent raw sums vs child raw sums Python aggregation parity test；
- 禁止正式 CUDA 配置静默退回 projected_meta/Python parent scatter。
```

---

# 13. 是否继续用 projected_meta

建议：

```text
projected_meta = fallback/debug only
cuda_scalar_anchor = 正式训练
```

原因：

```text
projected_meta不含真实transmittance/occlusion；
当前 inbound/valid ratio偏低；
容易造成 parent/child view权重不准。
```

不过 CUDA scalar anchor 实现前，projected_meta 可以继续用来验证 query/gather 流程。

---

# 14. 风险控制

## 14.1 质量下降

若 Full Sparse Parent 不稳定，加临时 hybrid：

```text
parent_context = sparse_parent_context + λ * no_grad_fwhr_shadow
```

只用于 debug，不作为长期方案。

## 14.2 CUDA gather 梯度不稳定

先禁用 offset：

```text
offset_scale = 0
只训练 weights/gate
```

确认质量后再开 offset。

## 14.3 显存仍高

降低：

```text
num_taps: 5 -> 3
child_chunk_size
parent_geometry_pe
SSIM refs
```

但优先保持 QDG 结构不变。

---

# 15. 最终推荐下一步

当前 Phase 1/2/3 已完成，Phase 5 parent aggregation 已并入 CUDA scalar anchor。剩余推荐顺序：

```text
1. 固定 b1r1 / b1r4 做 200 step 回归；
2. 强制 non-fixed path 做 b1r1 / b1r4 50 step profile；
3. 跑 b1r8 / b2r4 50 step capacity smoke；
4. 基于 profile 决定是否优化 anchor aux、decoder/render 或 AMP/checkpointing；
5. 性能稳定后再进入 validation/quality sweep。
```

不要在性能边界未确认前做长质量 sweep；否则无法区分质量问题和系统性性能/显存瓶颈。
