# IForward Stage 2_0 v27：无 AMP 的显存与性能根本优化方案

版本基线：`stage2_0_biggs_incremental_whdd` / v27  
目标版本建议：`stage2_0_biggs_compact16_residualonly`

---

## 0. 决策摘要

本轮明确不使用 AMP，优先从结构上降低复杂度：

```text
1. 完全移除 Stage 2_0 的 DINO 分支及 DINO cache。
2. 删除单分支情况下冗余的 fusion neck。
3. 2D feature 由 48 维降到 16 维；residual U-Net 由 base48/depth4 缩到 base24/depth3。
4. parent event 与 fine event 解耦：parent 保留 64 维，WHDD 解码为 N 级 16 维 fine event。
5. fine posterior updater 改为 event16 -> hidden32 -> delta heads。
6. Stage 2_0 删除未被读取的 N 级 hidden state 和 confidence head，只保留 noop gate。
7. parent 压缩率不作为解决 U-Net 显存的手段；先维持约 16×，之后只做 24×/32× 质量实验。
8. 性能侧继续完成：FP32 parent stats、CUDA incremental stats、GPU vectorized rigid active assignment、fused WHDD、缩小 GPU assignment cache。
```

预期（工程估算，需实测）：

```text
peak allocated: 当前约 29–31GB -> 约 12–18GB
step time:      当前均值约 1.7s -> 约 0.7–1.1s
```

---

# 1. 为什么 parent 压缩率不会降低 residual U-Net / fusion 显存

当前顺序：

```text
GT RGB + parent render RGB
    -> residual U-Net / DINO / fusion，生成图像特征 [V,H,W,C]
    -> parent alpha/T lifting
    -> parent token [M,C]
```

U-Net/fusion 的复杂度取决于：

```text
相机数量 V
图像分辨率 H×W
CNN 通道数
网络深度
RAFT-like repeat K
```

不取决于：

```text
parent 数量 M
fine 数量 N
压缩率 N/M
```

因为 parent 数量只在图像特征已经生成之后，影响 raster/backprojection 的输出行数。

因此：

```text
16× -> 50× parent compression
```

可能降低：

```text
parent lifting 时间
parent xCPE / far MLP 时间
parent event activation
```

但几乎不降低：

```text
residual U-Net activation
DINO adapter activation
fusion activation
```

而 50× 会显著增加：

```text
parent footprint
深度层混合
alpha/T approximation error
child visibility 信息丢失
```

结论：不以 50× 解决 CNN 显存。P0 保持当前 grouping；P2 再测试 24×、32×，50×只作为速度上限实验。

---

# 2. 第一阶段：移除 DINO 和 fusion neck

## 2.1 新的数据流

当前：

```text
DINO(rgb) ----------------┐
                          ├-> fusion neck -> feat2d[48]
UNet(gt_rgb, render_rgb) -┘
```

改为：

```text
CompactResidualUNet(gt_rgb, parent_render_rgb)
    -> feat2d[16]
```

不再需要：

```text
DINO backbone
DINO adapter proj/fuse
DINO cache
FusionNeck2D
DINO optimizer/trainability 分组
```

不要从全项目删除 DINO 模块；只让 Stage 2_0 选择新的 extractor 类型，避免破坏 Stage 1 / 其他阶段。

## 2.2 P0 最小实现：复用现有 ImageFeatureExtractor

新增 extractor：

```python
class ResidualOnlyFeatureExtractor(nn.Module):
    def __init__(self, ...):
        self.residual_unet = ImageFeatureExtractor(
            in_channels=6,
            feat_channels=16,
            base_channels=24,
            feature_downscale=1,
            depth=3,
            bilinear=True,
        )

    def forward(self, images):
        return self.residual_unet(images)
```

配置：

```yaml
model:
  feat_2d_channels: 16
  feature_extractor:
    type: residual_only
    residual_unet:
      in_channels: 6
      feat_channels: 16
      base_channels: 24
      feature_downscale: 1
      depth: 3
      bilinear: true
```

删除 Stage 2 配置中的：

```yaml
dino: ...
fusion: ...
```

trainability：

```yaml
stage6_0:
  base_measurement:
    train_2d_frontend: true
    train_residual_unet: true
    train_fusion_neck: false
    train_dino_adapter: false
    train_dinov2: false
```

optimizer 删除：

```text
stage6_measurement_frontend_fusion_neck
DINO adapter 相关 group
```

保留：

```text
stage6_measurement_frontend_residual_unet
```

## 2.3 P1 进一步根本优化：半分辨率 Compact Residual Pyramid

若 P0 后显存仍高，再新增：

```text
ResidualPyramid2D
```

结构：

```text
input [V,6,H,W]
  -> stride-2 stem, 16ch                    [H/2,W/2]
  -> residual block 16
  -> down block 24                          [H/4,W/4]
  -> down block 32                          [H/8,W/8]
  -> top-down additive FPN
  -> output 16ch                            [H/2,W/2]
```

关键：

```text
- 不用完整 U-Net 对称 decoder
- skip 使用 add，不用 concat
- 输出分辨率 H/2×W/2
- alpha/T lifting 同步使用 feature resolution
```

配置：

```yaml
feature_extractor:
  type: residual_pyramid
  residual_pyramid:
    in_channels: 6
    stem_channels: 16
    stage_channels: [16, 24, 32]
    out_channels: 16
    output_downscale: 2
    skip_mode: add
```

半分辨率使图像 activation 和 feature lifting 像素量近似降为 1/4，是比提高 parent compression 更直接的机制优化。

---

# 3. 第二阶段：N 级 fine token 96 -> 16

## 3.1 不要把 parent 与 fine 使用同一宽度

使用 dual-width：

```text
parent event width: 64
fine event width:   16
posterior hidden:   32
```

原因：

```text
parent 数 M 约 3万，可以保留较强表达；
fine 数 N 约 50万，必须极窄。
```

若担心一次性改 parent 质量，第一轮可：

```text
parent width 96 不变
fine width 16
```

稳定后再把 parent 96 -> 64。

## 3.2 Struct decoder 参数

推荐主配置：

```yaml
model:
  sparseConv_outdim: 64
  feat_2d_channels: 16

stage6_0:
  struct_event_decoder:
    event_dim: 64
    feat_2d_dim: 16
    support_embed_dim: 4
    branch_embed_dim: 4
    token:
      token_dim: 64
    param_obs_codec:
      output_dim: 24
    near:
      num_blocks: 2
    far:
      hidden_dim: 64
      num_layers: 2
```

`event_dim == token_dim` 的约束继续满足。

## 3.3 WHDD 改为异宽解码

当前 WHDD 假设：

```text
parent event E -> fine event E
```

改为：

```text
parent event Ep=64 -> fine event Ef=16
```

公式：

```text
base_k   = BaseProj(parent_event_k)        [16]
detail_k = DetailHead(parent_event_k)      [R=3,16]

fine_event_i = base_parent(i)
             + gamma * Σ_r phi_i,r detail_parent(i),r
```

模块：

```python
class WHDDCompactDecoder(nn.Module):
    base_proj   = Linear(parent_event_dim=64, fine_event_dim=16)
    detail_head = Linear(64, rank * 16)
```

最后层 zero-init 的对象只应是 `detail_head`；`base_proj` 使用正常初始化或从旧 parent event 做蒸馏初始化。

配置：

```yaml
biggs:
  child_decoder:
    mode: whdd_compact_fixed_basis
    rank: 3
    parent_event_dim: 64
    fine_event_dim: 16
    residual_scale_init: 0.01
    fused_cuda: true
```

CUDA fused op 直接输出 `[N,16]`，禁止构造 `[N,3,16]`。

## 3.4 Fine posterior updater

改为：

```text
fine event 16
  -> Linear(16,32)
  -> LN/GELU
  -> Linear(32,32)
  -> delta heads
```

配置：

```yaml
posterior_updater:
  event_dim: 16
  hidden_dim: 32
  stage_hidden_dim: 0
  input_current_ctx: false
  input_vsm_ctx: false
  phase_b_hooks:
    accept_vsm_ctx: false
```

理论 trunk FLOPs 比：

```text
旧：96*96 + 96*96 = 18432 / point
新：16*32 + 32*32 = 1536 / point
约 12× 降低
```

N 级单个 feature tensor：

```text
[N,96] -> [N,16]
约 6× 降低
```

## 3.5 删除无用 fine hidden/confidence

Stage 2 当前 memory/VSM/current context 都关闭，`LocalGSState.hidden` 没有成为下一 repeat 的输入。

改动：

```text
- Stage6PosteriorUpdater 支持 stage_hidden_dim=0
- 不创建 head_hidden
- BranchDelta.hidden 改 Optional[Tensor] 或 [N,0]
- LocalGSState.apply_delta 在 hidden=None 时不更新 hidden
- Stage 2 local state 不分配 [N,96] hidden
```

`confidence` 当前只用于日志，若无 gate/ADC consumer：

```text
enable_confidence_head: false
```

保留 `noop`，因为它实际控制 delta gate。

配置：

```yaml
posterior_updater:
  output_hidden: false
  output_confidence: false
  output_noop: true
```

---

# 4. Parent 压缩率的具体建议

当前数据约：

```text
N fine ≈ 51万
M parent ≈ 3.15万
压缩约 16×
```

不要直接改到 50×。

分级实验：

```text
P0：保持当前约 16×，先完成 DINO移除 + fine16。
P1：24×。
P2：32×。
P3：50× 只做速度上限，不作为默认训练。
```

大致目标 M：

```text
16× -> 31–32k
24× -> 21k
32× -> 16k
50× -> 10k
```

调整顺序应是：

```text
1. target/max children
2. voxel size
3. projected footprint / depth conflict 限制仍必须保留
```

示例 24×：

```yaml
assignment:
  bg:
    voxel_size: 0.65
    target_children_per_parent: 24
    max_children_per_parent: 48
  distant:
    voxel_size: 4.0
    target_children_per_parent: 48
    max_children_per_parent: 96
  rigid:
    voxel_size: 0.38
    target_children_per_parent: 20
    max_children_per_parent: 40
```

不能只放大 voxel 而不监控：

```text
parent projected radius
parent depth span
alpha/T group error
opacity saturation
rigid visibility conflict
```

压缩率提高的预期收益主要是：

```text
lifting 45–50ms -> 可能降到 20–35ms
parent xCPE / far MLP 降低
parent event activation 降低
```

不会解决图像 CNN 的主要显存。

---

# 5. 性能侧必须同时落地的修正

## 5.1 Parent stats 禁止 float64

当前 `_stats_dtype()` 将 FP32 输入升级为 float64，必须改为：

```python
def _stats_dtype(ref):
    return torch.float32
```

配置：

```yaml
parent_state:
  stats_dtype: float32
  child_cache_dtype: float32
```

## 5.2 CUDA incremental stats

当前 PyTorch padded gather：

```text
[M,max_child,...]
```

必须替换为 one-block-per-parent CUDA reduce：

```text
old child cache + new child state
  -> ΔW/ΔA/ΔB/ΔU/ΔSH
  -> parent stats
  -> finalize parent params
```

最后 repeat 保持 skip。

## 5.3 Rigid active assignment GPU 向量化

禁止：

```text
Python per-child loop
.item()
Python dict grouping
per-parent nonzero
```

使用：

```python
key = global_parent_id * 2 + inside_flag
unique_key, inverse, counts = torch.unique(...)
order = torch.argsort(inverse)
starts = exclusive_cumsum(counts)
```

分开计时：

```text
rigid_active_assignment_ms
rigid_projector_cuda_ms
```

## 5.4 Fused WHDD compact16

必须启用：

```yaml
child_decoder:
  fused_cuda: true
```

输入：

```text
parent_base [M,16]
parent_detail [M,3,16]
child_basis [N,3]
child_to_parent [N]
```

输出：

```text
fine_event [N,16]
```

## 5.5 Assignment GPU cache 缩小

当前 64 items + device copy 会积累常驻显存。

改为：

```yaml
assignment:
  cache_max_items_cpu: 64
  cache_max_items_gpu: 2
  cache_device_copy: hot_only
```

若实现成本高，P0 直接：

```yaml
cache_max_items: 8
cache_device_copy: false
```

---

# 6. 推荐完整配置草案

```yaml
model:
  feat_2d_channels: 16
  sparseConv_outdim: 64

  feature_extractor:
    type: residual_only
    residual_unet:
      in_channels: 6
      feat_channels: 16
      base_channels: 24
      feature_downscale: 1
      depth: 3
      bilinear: true

  iforward:
    version: stage2_0_biggs_compact16_residualonly
    biggs:
      assignment:
        cache_scope: scene_segment_topology
        cache_max_items: 8
        cache_device_copy: false
        # 首轮保持原压缩率
        bg:
          voxel_size: 0.5
          target_children_per_parent: 16
          max_children_per_parent: 32
        distant:
          voxel_size: 3.0
          target_children_per_parent: 32
          max_children_per_parent: 64
        rigid:
          voxel_size: 0.3
          target_children_per_parent: 16
          max_children_per_parent: 32

      parent_state:
        mode: incremental_sufficient_stats
        stats_dtype: float32
        child_cache_dtype: float32
        update_backend: cuda_incremental_sufficient_stats

      child_decoder:
        mode: whdd_compact_fixed_basis
        rank: 3
        parent_event_dim: 64
        fine_event_dim: 16
        fused_cuda: true
        residual_scale_init: 0.01

  stage6_0:
    base_measurement:
      source_evidence_grad_mode: train_2d_detach_alpha
      train_2d_frontend: true
      train_residual_unet: true
      train_fusion_neck: false
      train_dino_adapter: false
      train_dinov2: false
      detach_v4_outputs: false
      detach_source_render_for_cnn: true

    struct_event_decoder:
      event_dim: 64
      feat_2d_dim: 16
      support_embed_dim: 4
      branch_embed_dim: 4
      token:
        token_dim: 64
      param_obs_codec:
        output_dim: 24
      near:
        num_blocks: 2
        voxel_size: 0.25
      far:
        hidden_dim: 64
        num_layers: 2

    posterior_updater:
      event_dim: 16
      hidden_dim: 32
      stage_hidden_dim: 0
      input_event: true
      input_current_ctx: false
      input_vsm_ctx: false
      output_hidden: false
      output_confidence: false
      output_noop: true
      phase_b_hooks:
        accept_vsm_ctx: false

training:
  amp:
    enable: false
```

---

# 7. 文件级改动

## 新增

```text
models/feature_extractors/residual_only.py
models/iforward/whdd_compact_decoder.py
models/iforward/csrc/whdd_compact_decode.cu
models/iforward/csrc/biggs_parent_stats_update.cu
```

## 修改

```text
models/feature_extractors/__init__.py
    注册 residual_only

models/streetforward/minimal_trainer_stage6_0.py
    支持无 DINO / 无 fusion frontend

models/streetforward/stage6_0/struct_event_decoder.py
    parent event 64 / feat2d 16

models/iforward/biggs_event_decoder.py
    或新增独立 compact WHDD

models/streetforward/stage6_0/posterior_updater.py
    stage_hidden_dim=0
    optional confidence/hidden heads

models/streetforward/stage6_0/local_gs_state.py
    Stage2 不分配/更新 hidden

models/iforward/biggs_parent_stats.py
    float32 stats
    CUDA update backend

models/iforward/biggs_assignment.py
    GPU cache policy
    rigid active vectorization helper
```

---

# 8. 分阶段实施与 ablation

## Phase A：DINO/fusion 删除

只改：

```text
DINO/fusion -> residual-only 48ch，其他维度不变
```

目的：验证删除 DINO 不明显损伤单帧效果，并测纯前端收益。

然后：

```text
residual output 48 -> 16
base48/depth4 -> base24/depth3
```

不要把两步合在第一次实验，否则质量变化无法归因。

## Phase B：fine event 16

先保持 parent event 96：

```text
parent96 -> WHDD -> fine16 -> posterior32
```

稳定后再：

```text
parent96 -> 64
```

## Phase C：删除 hidden/confidence

确认 Stage2 无 reader 后删除。

## Phase D：性能后端

```text
float32 stats
CUDA incremental
rigid assignment vectorization
fused WHDD
cache缩小
```

## Phase E：压缩率实验

```text
16× baseline
24×
32×
50× speed ceiling only
```

---

# 9. 测试与验收

固定同一 validation batch、K=4，至少比较：

```text
A：v27 原版
B：无DINO/无fusion，仍48维
C：residual16 + parent96/fine16
D：parent64/fine16 + hidden删除
E：D + CUDA性能修正
```

记录：

```text
peak allocated / reserved
每 repeat after_unet / after_parent_encoder / after_whdd / after_posterior
step / forward / backward / observe ms
parent lifting ms
WHDD ms
posterior ms
PSNR / SSIM / LPIPS
每 K 的 delta norm 与 PSNR
```

建议验收：

```text
显存：E <= 18GB，理想 <= 15GB
速度：E <= 1.1s/step，理想 <= 0.8s
质量：固定验证 PSNR 相比 A 下降 <= 0.3dB
迭代：K=1..4 效果总体改善，不依赖固定第4步突然恢复
```

若质量下降：

```text
优先恢复 parent event 64 -> 96；
其次 residual base24 -> 32；
不要先把 fine event 16 恢复到96。
```

fine16 是本轮最关键的结构性压缩，应尽量保留。
