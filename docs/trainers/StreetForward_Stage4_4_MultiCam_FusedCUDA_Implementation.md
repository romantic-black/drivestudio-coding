# StreetForward Stage 4.4 Multi-Camera Fused CUDA 实施方案（修正版）

> 目标：只优化 CUDA kernel 层与其最小包装层，解决 Stage 4.4 多 source 下 `2D feat / fused backproject / backward` 过慢问题。  
> 基线：当前 `AlphaTWeightExtractorV3` 逐视角 `packed=True` 渲染 + 单视角 fused op + 全局累加归一化。

---

## 1. 目标与边界

### 1.1 本次要做

1. 把逐视角 Python 循环改为：一次 multi-camera packed meta + 一次 multi-camera fused forward/backward op。
2. backward 显式减少 `grad_feat2d` 的全局 atomic 写入。
3. forward 视情况减少 `feat_sum / weight_sum_feature / weight_sum_support` 的全局 atomic 写入。

### 1.2 本次不做

- 不改主公式：`feat_out = A / (B + eps)`。
- 不改 support 语义：`acc_w = S`。
- 不改 Stage4.2/4.3/4.4 one-pass split 语义，不改 trainer 主流程/loss/scheduler/writeback。
- 不引入 `V x N x C` 大缓存。

---

## 2. 当前瓶颈与修正共识

### 2.1 当前瓶颈来源（已确认）

- 当前多视角路径仍是逐视角调用 fused op，`V` 个视角就有 `V` 次关键 kernel launch。
- backward 对每个有效 pair、每个 channel、四邻点做 global `atomicAdd`，在大 `pairs_total` 下非常重。

### 2.2 必须写清的四个实现事实

1. **shared patch 只砍“全局 atomic 写”，不砍 pair 遍历本身**  
   `sigma/alpha/vis`、阈值判断、pair traversal、梯度读取都还在，收益上限必须由 profile 判断。
2. **shared patch flush 到全局仍必须 atomicAdd**  
   相邻 image tiles 的 feature patch 会重叠（双线性四邻点跨 tile 边界），不能普通写。
3. **patch 计算必须含 1 像素 halo**  
   patch 必须覆盖 tile 内所有像素对应的 `u0/u1/v0/v1`，不是简单整数映射框。
4. **shared memory 预算不稳定，必须有 fallback 或 channel tiling**  
   预算超限时自动回退到旧路径，或分 channel chunk（如 8/16 通道）。

---

## 3. 设计原则

1. **与 v3 数值语义等价**：  
   `A=Σ_v feat_sum_v`，`B=Σ_v weight_sum_feature_v`，`S=Σ_v weight_sum_support_v`，最终 `feat_out=A/(B+eps)`，`acc_w=S`。
2. **并行粒度下沉到 CUDA**：  
   `grid.z = cam_id`，而不是 Python 外层 `for cam in cameras`。
3. **profile 驱动决策**：  
   Phase B 必须先确认 backward 主要瓶颈在 global atomic，而不是 traversal/访存本身。

---

## 4. Multi-Camera Packed Meta Builder

### 4.1 输入/输出

输入：

- `means [N,3]`, `quats [N,4]`, `scales [N,3]`, `opacities [N]`
- `viewmats [V,4,4]`, `Ks [V,3,3]`
- `width`, `height`

输出（全视角合并 packed meta）：

- `means2d [nnz,2]`, `conics [nnz,3]`, `opacities [nnz]`
- `camera_ids [nnz]`（builder 阶段使用）
- `gaussian_ids [nnz]`（packed 全局高斯 id）
- `isect_offsets [V,tile_h,tile_w]`
- `flatten_ids [n_isects]`

### 4.2 参数对齐（必须）

为保持和当前 v3 单视角 `renderer(..., packed=True)` 可见集一致，builder 必须显式对齐：

- `near_plane`
- `far_plane`
- `tile_size`
- `radius_clip`（若依赖）
- `camera_model`
- `rasterize_mode="classic"`
- `sparse_grad=False`
- `absgrad=True`（若当前路径开启）

否则可能出现 “逻辑正确但数值不等价”。

### 4.3 `gaussian_ids` dtype/语义修正

- 新接口中不再沿用旧名 `packed_to_global_gaussian_ids`，改为 `packed_global_gaussian_ids`。
- 接口允许 `int32`（可选支持 `int64` 并内部统一），因为 packed projection 返回的 `gaussian_ids` 通常是全局索引（常见 int32）。

---

## 5. Multi-Camera Forward Op

### 5.1 wrapper 签名（修正后建议）

```python
@torch.no_grad()
def rasterize_and_backproject_multi_camera_in_range(
    range_start: int,
    range_end: int,
    means2d: torch.Tensor,                     # [nnz, 2], float32
    conics: torch.Tensor,                      # [nnz, 3], float32
    opacities: torch.Tensor,                   # [nnz], float32
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,               # [V, tile_h, tile_w], int32
    flatten_ids: torch.Tensor,                 # [n_isects], int32
    packed_global_gaussian_ids: torch.Tensor,  # [nnz], int32/int64
    feat2d: torch.Tensor,                      # [V, Hf, Wf, C], float32
    num_gaussians: int,
    weight_threshold: float,
    return_support: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

### 5.2 两个冗余输入的修正

1. **`camera_ids` 不进 kernel**  
   kernel 运行时 camera 由 `blockIdx.z` 唯一确定；`camera_ids` 仅用于 builder 构建多图像 intersections。
2. **`transmittances [V,H,W]` 不显式传入**  
   v1 语义固定起始 `trans=1.0f`，直接在 kernel 内初始化。未来需要 external transmittance 再加 optional 分支。

### 5.3 CUDA 映射

- `grid=(tile_h,tile_w,V)`
- `block=(tile_size,tile_size,1)`
- block 绑定 `(cam_id, tile_i, tile_j)`，逻辑复用单视角路径。

---

## 6. Multi-Camera Backward Op（重点）

### 6.1 wrapper 签名（修正后建议）

```python
@torch.no_grad()
def backproject_feature_grad_multi_camera_in_range(
    range_start: int,
    range_end: int,
    means2d: torch.Tensor,                     # [nnz, 2], float32
    conics: torch.Tensor,                      # [nnz, 3], float32
    opacities: torch.Tensor,                   # [nnz], float32
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,               # [V, tile_h, tile_w], int32
    flatten_ids: torch.Tensor,                 # [n_isects], int32
    packed_global_gaussian_ids: torch.Tensor,  # [nnz], int32/int64
    grad_feat_sum: torch.Tensor,               # [num_gaussians, C], float32
    feat_h: int,
    feat_w: int,
    channels: int,
    weight_threshold: float,
) -> torch.Tensor:                             # [V, Hf, Wf, C]
    ...
```

### 6.2 Phase A（最低风险）

- 先做 multi-camera backward 合并版。
- 仍允许沿用旧的“pair 级全局 atomic 写 `grad_feat2d`”。
- 目标：先去掉 Python per-view 循环和多次 launch。

### 6.3 Phase B（主优化）

引入 `shared-memory grad patch cache`：

1. 计算当前 tile 在 feature map 的 patch 范围（含 halo）。
2. 分配 `s_grad_patch[patch_h, patch_w, C]`。
3. 先累加到 shared patch。
4. block 结束后 atomic flush 到全局 `grad_feat2d[cam_id]`。

### 6.4 Phase B 实施硬规则（必须遵守）

1. **flush 仍是 atomicAdd**（不能普通 store）。
2. **patch 带 1 像素 halo**，覆盖所有 `u0/u1/v0/v1`。
3. **shared memory 预算先算后执行**：
   - `patch_h * patch_w * C * sizeof(float)`
   - `+ id_batch + xy_opacity_batch + conic_batch`
4. **预算超阈值自动分流**：
   - fallback 到旧路径，或
   - channel tiling（8/16 通道 chunk）。

---

## 7. forward 第二阶段优化（可选）

在 forward kernel 加 block-local Gaussian cache（shared hash/小字典）：

- batch 内先按 `g_global` 聚合
- batch 尾统一 flush 到全局
- cache 不足时 fallback

优先级低于 backward Phase B。

---

## 8. 文件改动清单

### 8.1 CUDA/C++

- `third_party/gsplat/gsplat/cuda/csrc/RasterizeAndBackproject3DGSMulti.h`
- `third_party/gsplat/gsplat/cuda/csrc/RasterizeAndBackproject3DGSMulti.cu`
- `third_party/gsplat/gsplat/cuda/csrc/BackprojectFeatureGrad3DGSMulti.cu`
- `third_party/gsplat/gsplat/cuda/ext.cpp`（pybind 导出）

### 8.2 Python wrapper

- `third_party/gsplat/gsplat/cuda/_wrapper.py`
  - `rasterize_and_backproject_multi_camera_in_range`
  - `backproject_feature_grad_multi_camera_in_range`

fast-fail 类型检查：

- `float32`: `means2d/conics/opacities/feat2d/grad_feat_sum`
- `int32`: `isect_offsets/flatten_ids`
- `int32/int64`: `packed_global_gaussian_ids`

### 8.3 Autograd / Extractor / Trainer

- `models/feature_extractors/alpha_t_extractor_v3.py`
  - `_RasterizeAndBackprojectFeatOnlyMultiCamFn`
  - `render_and_backproject_streaming_fused_multi_camera(...)`
- `models/streetforward/minimal_trainer_stage4_4.py`
  - 仅切换 extractor 内部路径，不改 one-pass split 主语义

---

## 9. 关键伪代码（修正后）

### 9.1 Forward

```python
meta = build_multi_camera_packed_meta(gaussians, cameras, H, W)
feat_sum, w_feat, w_sup, cnt_all, cnt_kept = rasterize_and_backproject_multi_camera_in_range(
    means2d=meta.means2d,
    conics=meta.conics,
    opacities=meta.opacities,
    isect_offsets=meta.isect_offsets,
    flatten_ids=meta.flatten_ids,
    packed_global_gaussian_ids=meta.gaussian_ids,
    feat2d=features_2d.float(),
    num_gaussians=N,
    weight_threshold=threshold,
    return_support=return_acc_w,
)
feat_out = feat_sum / (w_feat.unsqueeze(-1) + eps)
acc_w = w_sup
```

### 9.2 Backward

```python
grad_feat2d = backproject_feature_grad_multi_camera_in_range(
    means2d=saved.means2d,
    conics=saved.conics,
    opacities=saved.opacities,
    isect_offsets=saved.isect_offsets,
    flatten_ids=saved.flatten_ids,
    packed_global_gaussian_ids=saved.gaussian_ids,
    grad_feat_sum=grad_feat_sum.contiguous(),
    feat_h=ctx.feat_h,
    feat_w=ctx.feat_w,
    channels=ctx.channels,
    weight_threshold=ctx.weight_threshold,
)
```

---

## 10. 测试与验收

### 10.1 数值一致性

- 对同输入比较旧 v3 与新 multi-camera：
  - `feat_out` allclose
  - `acc_w` allclose
  - `pair_count_total/pair_count_threshold` 一致

### 10.2 backward 一致性

- 比较两版 `features_2d.grad`（旧 `_RasterizeAndBackprojectFeatOnlyFn` vs 新 multi-camera）。

### 10.3 Stage4.4 集成一致性

- `feat_2d_all.shape`、`acc_w_all.shape`
- one-pass split `[bg, distant, rigid_S, sky]` 不变
- trainer 语义（如 `src_backproject_pass_count`）不变

### 10.4 性能判定规则（新增）

- Phase B 前先 profile，确认 backward 瓶颈在 global atomic。
- 若主要瓶颈在 pair traversal/访存，则优先优化 traversal 路径，不盲目扩大 shared patch。

---

## 11. 分阶段落地

### Phase A

- multi-camera meta builder + multi-camera forward/backward（不含 shared patch）

### Phase B

- backward shared grad patch + halo + atomic flush + shared budget fallback/channel tiling

### Phase C

- forward block-local Gaussian cache / warp-aggregated atomics

---

## 12. 回滚策略

- 保留现有 v3 逐视角 fused 路径作为 fallback。
- 通过配置开关强制回退，不影响训练主线。

## 12.1 实装开关（当前代码）

- `model.use_fused_cuda_backproject_v4`：开启 v4 multi-camera fused 路径。
- `model.fused_cuda_backproject_v4_force_fallback`：强制回退到现有 v3 路径。

当前 `MinimalStreetForwardStage4_4` 的调用优先级为：

1. `v4`（若开启且不强制 fallback）
2. `v3`
3. `v2`
4. `v1`

## 12.2 已完成验证（当前）

- Python 侧语法检查通过：
  - `models/feature_extractors/alpha_t_extractor_v3.py`
  - `models/streetforward/minimal_trainer_stage4_4.py`
  - `third_party/gsplat/gsplat/cuda/_wrapper.py`
- 单测通过（`conda drivestudio-new` + `PYTHONPATH=/root/drivestudio-coding`）：
  - `tests/models/test_alpha_t_extractor_v3_multicam.py`
  - `tests/models/test_alpha_t_extractor_fused_v2.py`
  - `tests/models/test_alpha_t_extractor_v2_grad_alignment.py`
- CUDA wrapper smoke test通过：
  - `rasterize_and_backproject_multi_camera_in_range`
  - `backproject_feature_grad_multi_camera_in_range`

## 12.3 Phase C 说明（forward cache）

Phase C 属于 profile 驱动的可选项。当前先保留前向主路径语义不变，优先落地 Phase A/B（尤其 backward shared-patch + budget fallback）。  
若后续 profile 证明 forward 全局 atomic 仍是瓶颈，再开启 block-local Gaussian cache 实装。

---

## 13. 一句话目标

在不改变 v3 数学语义与 Stage4.4 one-pass 语义的前提下，把并行粒度从 Python `camera` 循环下沉到 CUDA `grid.z=image_id`，并以 profile 驱动的 shared patch 方案优先压缩 backward 全局 atomic 开销。
