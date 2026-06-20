# IForward Stage 2_0：GRLD Parent→Child Decoder 与 DINOv2-Base Residual-Feature Concat 详细设计方案

版本建议：`stage2_0_biggs_grld_dinov2base_concat48`  
基线代码：`drivestudio_stage6_refactor_context_20260619_v28`  
目标：替换当前 `whdd_compact_fixed_basis`，让 child 当前 Gaussian 属性参与 parent→child 解码；重新引入轻量 DINOv2-base 语义特征，但只在 image feature 层做直接拼接，避免过早 image-space 深融合。

---

## 0. 核心结论

当前 v28 的 Stage 2_0 已经把主链路压缩到：

```text
parent GS lifting [M,16]
  -> parent struct event [M,64]
  -> WHDD compact fixed basis
  -> fine event [N,16]
  -> posterior updater [N,16]
```

这个方向节省了显存，但当前 `WHDD compact fixed basis` 有两个核心缺陷：

```text
1. child 不接收自身当前 Gaussian 属性。
   当前 child detail 只来自 assignment-time fixed xyz basis。
   scale / opacity / SH / mass / covariance / current repeat state 都没有进入 decoder。

2. parent→child 关系没有被充分利用。
   当前 decoder 是固定空间 basis × parent detail，
   不是“当前 child Gaussian state 相对当前 parent Gaussian state”的动态关系映射。
```

因此建议引入：

```text
GRLD = Gaussian Relational Lifting Decoder
```

主公式：

```text
child_event_i
  = base(parent_event_p)
  + λ_branch · Σ_r q_i,r · gate_p,r · detail_p,r
```

其中：

```text
q_i = RelationProj(centered_relation(child_i, parent_p))
```

也就是：

```text
child 当前 Gaussian 属性
  -> parent-relative Gaussian relation
  -> sibling weighted-centering
  -> parent-conditioned low-rank relation operator
  -> fine event
```

DINOv2 部分建议改为：

```text
GT RGB -> frozen DINOv2-base -> 16ch DINO feature
GT RGB + parent render RGB -> trainable residual UNet -> 32ch residual feature
concat([residual32, dino16]) -> 48ch image feature
alpha/T lifting -> parent feature [M,48]
```

不做：

```text
image-space fusion neck
DINO-large
DINO feature 与 residual feature 的深层 image conv 融合
```

原因：后续要接 PTv3 / Mamba / parent 3D reasoning，真正强上下文建模应发生在 GS/parent-token 层，而不是过早在 image feature 层把 residual 和 semantic 特征揉在一起。

---

# 1. 设计约束与非目标

## 1.1 保留不变

```text
- parent GS 仍直接参与 alpha/T lifting。
- projector assignment 固定。
- parent projector forward-only，不训练 alpha/T geometry Jacobian。
- parent stats incremental update 与 block exact refresh 保留。
- posterior updater 暂时保持 fine-event 输入，不在本阶段改成 parent-level delta decoder。
- history gate / ADC 仍不进入本阶段。
```

## 1.2 本阶段要解决

```text
A. child 当前属性必须进入 parent→child decoder。
B. parent→child 关系应是动态 Gaussian relation，而非静态 xyz basis。
C. 解码器不能退回 N 级大 MLP。
D. DINOv2-base 作为语义低频补充重新引入，但不做 image-space 深融合。
```

## 1.3 本阶段不解决

```text
- 不训练 DINOv2 backbone。
- 不训练 DINO adapter/proj/fusion；DINO输出作为 frozen semantic feature。
- 不对 parent alpha/T geometry 做 backward。
- 不直接替换 posterior updater 为 parent-level updater。
- 不加入 view-dependent child relation，先预留接口。
```

---

# 2. 当前 WHDD 的问题定位

当前 `whdd_compact_fixed_basis` 逻辑可概括为：

```text
parent_event_p -> base_p [16]
parent_event_p -> detail_p [R=3,16]
child_basis_i  -> assignment-time xyz basis [3]

fine_event_i = base_p + γ · child_basis_i · detail_p
```

这个结构的优点是轻，但它把 child 的差异完全固定为：

```text
assignment-time spatial coordinate
```

而 IForward 需要的 child 差异应该来自：

```text
current child Gaussian state relative to current parent Gaussian state
```

例如两个 child 当前具有不同：

```text
opacity / optical thickness
scale / projected footprint
SH / appearance residual
relative mass contribution
relative covariance
```

当前 WHDD 无法感知这些差异，只能给出由静态 xyz basis 决定的 event variation。

因此不是简单把 rank 从 3 加到 8，而是要替换 detail coordinate。

---

# 3. GRLD 总体结构

## 3.1 数据流

```text
parent_event_p [M,64]
parent_params_p [M,*]
parent_stats_p  [M,*]
child_params_i  [N,*]
child_cache_i   [N,*]  # dynamic mass / diag_cov from parent stats update
child_to_parent [N]

        │
        ▼
Gaussian Relation Codec
        │
        ▼
relation_i [N,d_rel]
        │
        ▼
Dynamic Mass Weighted Centering per parent
        │
        ▼
relation_centered_i [N,d_rel]
        │
        ├─ RelationProj -> coeff_i [N,R]
        │
        └─ Weighted RMS/entropy -> sibling_summary_p [M,d_sum]

parent_event_p + SummaryProj(sibling_summary_p)
        │
        ├─ BaseHead   -> base_p [M,E]
        ├─ DetailHead -> detail_p [M,R,E]
        └─ GateHead   -> gate_p [M,R]

fused low-rank decode:
        fine_event_i = base_p + λ Σ_r coeff_i,r gate_p,r detail_p,r
```

推荐第一版：

```text
parent_event_dim = 64
fine_event_dim   = 16
relation_dim     = 12 或 16
rank             = 4
```

不要默认 R=8。R=4 已足以表达：

```text
xyz/cov/mass/appearance 的少量低阶变化
```

---

# 4. Gaussian Relation Codec

GRLD 的核心是 relation code。它不是普通 child MLP 输入，而是经过 Gaussian 语义归一化的 parent-relative residual。

所有 relation 输入默认：

```text
detach()
```

也就是 forward 使用当前 child 属性，但不重新打开 geometry Jacobian。

## 4.1 相对位置

### BG / distant

```text
r_xyz_i = (μ_i - μ_p) / (scale_p + eps)
```

由于 parent projector 已采用 diagonal covariance，parent local frame 就是世界轴向归一化。

### Rigid

rigid 应使用 object canonical frame：

```text
r_xyz_i = R_obj^-1 (μ_i - μ_p) / (scale_p + eps)
```

这样同一车辆旋转后 child relation 不随世界姿态变化。

第一版如果 route 层还没有提供 canonical parent/child params，则可先使用 world-space relation，并记录：

```text
iforward/grld/rigid_relation_world_mode = 1
```

后续修正。

## 4.2 相对 covariance / projected shape proxy

父节点使用 diagonal covariance，因此最自然的是比较世界 diagonal covariance。

child diagonal covariance：

```text
diagΣ_i = diag(R(q_i) diag(scale_i²) R(q_i)^T)
```

parent diagonal covariance：

```text
diagΣ_p = scale_p²
```

relation：

```text
r_cov_i = log(diagΣ_i + eps) - log(diagΣ_p + eps)
```

这里应直接复用：

```text
BigGSParentRuntime.child_cache.diag_cov
```

不要在 decoder 内重复计算 quat→diag_cov。

## 4.3 相对 optical mass

当前 child contribution cache 中应已有：

```text
m_i = τ_i · area_i
```

relation：

```text
mean_mass_p = W_p / n_p
r_mass_i = log(m_i + eps) - log(mean_mass_p + eps)
```

注意：这里必须使用 block runtime 中的动态 child mass，而不是 assignment 时的静态 child mass。

## 4.4 相对 opacity

```text
r_opacity_i = opacity_logit_i - opacity_logit_p
```

它和 mass 不同：

```text
mass = opacity × area
opacity = alpha blending 直接因素
```

因此单独保留。

## 4.5 相对 appearance

不建议直接把完整 SH 全塞入。当前 sh degree=1 时可以用：

```text
r_sh_dc_i = sh_dc_i - sh_dc_p             # 3 dims
r_sh_rest_energy_i = log(||sh_rest_i|| + eps) - log(||sh_rest_p|| + eps)  # 1 dim
```

更稳的版本：

```text
r_sh_proj_i = LinearNoBias((sh_i - sh_p).detach()) -> 4 dims
```

第一版推荐：

```text
r_sh_dc 3维 + r_sh_rest_energy 1维
```

## 4.6 推荐 relation 维度

第一版：

```text
z_i = [
  r_xyz,             # 3
  r_cov,             # 3
  r_mass,            # 1
  r_opacity,         # 1
  r_sh_dc,           # 3
  r_sh_rest_energy,  # 1
]
```

总维度：

```text
d_rel = 12
```

可选扩展：

```text
r_support_i = log(support_parent + eps) broadcast 后加入 1维
r_static_xyz_init_i 作为稳定位置 prior 3维
```

但第一版不要加入过多维度。

---

# 5. Dynamic Mass Weighted Centering

对 parent `p` 下 child `i`：

```text
π_i = m_i / Σ_j m_j
```

计算：

```text
z_mean_p = Σ_i π_i z_i
z_centered_i = z_i - z_mean_p
```

然后：

```text
q_i = RelationProj(z_centered_i)
```

`RelationProj` 必须：

```text
bias=False
```

因为：

```text
Σ_i π_i z_centered_i = 0
=> Σ_i π_i q_i = 0
```

后续若 parent-conditioned decode 对 child 是线性的，则：

```text
Σ_i π_i (fine_event_i - base_parent) = 0
```

这保证 parent event 是 children event 的 weighted coarse/DC component。

---

# 6. Sibling Summary

仅靠 parent stats 无法知道 group 内是否存在强烈分裂。建议给 parent event 加一个小的 sibling summary：

```text
s_p = [
  weighted_rms(z_centered),   # 12 dims
  mass_entropy,               # 1 dim
  log_child_count,            # 1 dim
]
```

其中：

```text
weighted_rms = sqrt(Σ_i π_i z_centered_i² + eps)
mass_entropy = -Σ_i π_i log(π_i + eps)
```

然后：

```text
h_p = parent_event_p + SummaryProj(s_p)
```

`SummaryProj` 最后一层 zero-init，避免初始化时破坏现有 parent event 分布。

---

# 7. Parent-Conditioned Relation Operator

## 7.1 Heads

```python
base_p   = BaseHead(h_p)                  # [M,E]
detail_p = DetailHead(h_p).view(M,R,E)    # [M,R,E]
gate_p   = 1.0 + tanh(GateHead(h_p))      # [M,R]
```

推荐：

```text
E = 16
R = 4
BaseHead: Linear(64,16)
DetailHead: LayerNorm(64) + Linear(64, R*16), zero-init
GateHead: LayerNorm(64) + Linear(64, R), zero-init
RelationProj: Linear(12,R,bias=False)
```

初始化时：

```text
DetailHead = 0
GateHead = 0
=> fine_event_i = base_p
```

这等价 parent broadcast，安全。

## 7.2 Decode formula

```text
fine_event_i = base_p + λ_branch · Σ_r q_i,r · gate_p,r · detail_p,r
```

branch scale：

```text
λ_bg       = learnable scalar, init 0.01
λ_distant  = learnable scalar, init 0.01
λ_rigid    = learnable scalar, init 0.01
```

这样每个 branch 可以不同速度地学习 child detail。

## 7.3 为什么不是普通 child MLP

普通 child MLP：

```text
child_code_i + parent_event_p -> MLP -> fine_event_i
```

会引入：

```text
O(N · hidden_dim²)
```

而 GRLD：

```text
RelationProj: O(N · d_rel · R)
Decode:       O(N · R · E)
Parent heads: O(M · 64 · R · E)
```

在：

```text
N≈500k, M≈30k, d_rel=12, R=4, E=16
```

计算量约：

```text
RelationProj: 500k * 12 * 4  ≈ 24M MAC
Decode:       500k * 4 * 16  ≈ 32M MAC
```

远低于 per-child 128 hidden MLP。

---

# 8. GRLD CUDA Fused Decode

不要构造：

```text
detail[parent_id] -> [N,R,E]
```

新增 fused op：

```python
fine_event = grld_decode(
    base,            # [M,E]
    detail,          # [M,R,E]
    gate,            # [M,R]
    coeff,           # [N,R]
    child_to_parent, # [N]
    branch_scale,
)
```

forward：

```text
out[i,e] = base[p,e]
         + branch_scale * Σ_r coeff[i,r] * gate[p,r] * detail[p,r,e]
```

backward：

```text
grad_base[p,e] += grad_out[i,e]

grad_detail[p,r,e] += branch_scale * coeff[i,r] * gate[p,r] * grad_out[i,e]

grad_gate[p,r] += branch_scale * coeff[i,r] * detail[p,r,e] * grad_out[i,e]

grad_coeff[i,r] = branch_scale * Σ_e gate[p,r] * detail[p,r,e] * grad_out[i,e]
```

`child_to_parent` 与 relation 输入不需要梯度。

为了避免 atomic，可以使用：

```text
child_order / parent_start / parent_count
```

做 one-block-per-parent backward reduction。

---

# 9. GRLD 文件与接口

新增文件：

```text
models/iforward/biggs_relational_decoder.py
models/iforward/csrc/grld_decode.cu
models/iforward/csrc/grld_decode_ext.cpp
models/iforward/cuda_grld_decode.py
```

主类：

```python
class GaussianRelationCodec(nn.Module):
    def build_relation(
        self,
        child_params,
        parent_params,
        child_cache,
        parent_stats,
        child_to_parent,
        branch,
        rigid_canonical_meta=None,
    ) -> Tensor:  # [N, d_rel]
        ...

class GaussianRelationalLiftingDecoder(nn.Module):
    def decode_branch(
        self,
        parent_event,
        parent_params,
        child_params,
        child_cache,
        parent_stats,
        child_to_parent,
        parent_start,
        parent_count,
        branch_id,
    ) -> Tensor:  # [N, E]
        ...
```

将当前：

```text
mode: whdd_compact_fixed_basis
```

替换为：

```text
mode: gaussian_relational
```

---

# 10. GRLD 配置草案

```yaml
model:
  iforward:
    biggs:
      child_decoder:
        mode: gaussian_relational
        parent_event_dim: 64
        fine_event_dim: 16
        relation_dim: 12
        rank: 4
        hidden_dim: 64
        fused_cuda: true

        detach_relation_inputs: true
        use_dynamic_child_cache: true
        use_static_assignment_mass: false

        relation:
          use_xyz: true
          use_diag_cov: true
          use_mass: true
          use_opacity: true
          use_sh_dc: true
          use_sh_rest_energy: true
          use_static_xyz_prior: false
          rigid_space: canonical_or_world_fallback

        sibling_centering:
          enable: true
          weight: dynamic_mass
          summary: true
          summary_zero_init: true
          entropy: true

        heads:
          base: linear
          detail: linear_zero_init
          gate: tanh_zero_init

        residual_scale:
          per_branch: true
          init_bg: 0.01
          init_distant: 0.01
          init_rigid: 0.01
          learnable: true
```

---

# 11. DINOv2-base Residual-Feature Concat

## 11.1 设计原则

重新引入 DINOv2，但不恢复旧的深融合结构。

旧结构问题：

```text
DINO feature + residual feature -> FusionNeck2D conv
```

这会在 image space 过早融合语义和 residual。后续如果要接 PTv3/Mamba 进行 parent-token级上下文建模，image-level deep fusion 可能把特征关系提前固化，降低后续 GS-level optimizer 可塑性。

新结构：

```text
residual branch: GT + parent render -> residual feature 32ch
DINO branch: GT RGB -> frozen DINOv2-base feature 16ch
直接 concat -> image feature 48ch
alpha/T lifting -> parent feature [M,48]
```

即：

```text
feature_2d = concat([residual32, dino16], dim=channel)
```

没有：

```text
FusionNeck2D
image-space conv after concat
cross attention
DINO-large
```

## 11.2 为什么用 DINOv2-base 而不是 large

```text
DINOv2-large 计算和显存压力高；
当前瓶颈已经在 parent lifting / recurrent updates / decoder；
DINO 只作为低频 semantic prior，base 足够。
```

输出通道只设：

```text
DINO = 16
Residual = 32
Total = 48
```

这与旧配置中的 `feat_2d_channels=48` 兼容，但比旧 DINO 48 + residual 48 再 fusion 的设计更轻。

## 11.3 新 extractor

新增：

```text
models/feature_extractors/dinov2_residual_concat.py
```

类：

```python
class DINOv2ResidualConcatExtractor(nn.Module):
    def __init__(
        dino_model_name="vit_base_patch14_reg4_dinov2",
        dino_freeze=True,
        dino_out_channels=16,
        residual_feat_channels=32,
        residual_base_channels=32,
        fusion_mode="concat",
    ):
        ...

    def forward(self, images, *, cached_dino=None):
        x6 = to_nchw_6(images)
        rgb = x6[:, :3]

        residual = residual_unet(x6)         # [B,H,W,32], trainable

        if cached_dino is None:
            dino = dino_adapter(rgb)         # [B,H,W,16], no_grad
        else:
            dino = cached_dino

        return torch.cat([residual, dino], dim=-1)  # [B,H,W,48]
```

注意：

```text
DINO branch 输出必须 no_grad / detached。
Residual branch 和后续 parent struct decoder 可训练。
```

## 11.4 DINO adapter 是否可训练

本阶段建议：

```text
DINO backbone: frozen
DINO adapter/proj/fuse: frozen
DINO feature cache: adapter output level
```

如果 adapter 可训练，就不能缓存 adapter output。为了避免复杂性，先全部冻结。

真正的 feature reweighting 留给：

```text
parent TokenBuilder / xCPE / later PTv3 / Mamba
```

这符合“不做 image-space 深融合”的目标。

## 11.5 DINO cache

DINO feature 只依赖：

```text
GT RGB
camera/image identity
DINO model fingerprint
feature resolution
```

与 parent render 无关。因此可复用之前的 lazy cache：

```text
L1 GPU hot cache
L2 CPU pinned FP16 cache
```

cache key：

```text
(scene_id, segment_id, source_frame_idx, camera_ids, image_ids,
 image_hw, feature_hw, dino_fingerprint)
```

DINO feature dtype：

```text
fp16 cache, forward 时 cast 到 residual dtype
```

## 11.6 配置草案

```yaml
model:
  feat_2d_channels: 48
  feature_extractor:
    type: dinov2_residual_concat

    residual_unet:
      in_channels: 6
      feat_channels: 32
      base_channels: 32
      feature_downscale: 1
      depth: 3
      bilinear: true

    dino:
      model_name: vit_base_patch14_reg4_dinov2
      pretrained: true
      freeze: true
      freeze_adapter: true
      out_channels: 16
      intermediate_layers: [4, 8, 11]
      pad_to_patch_multiple: 14
      cache:
        enable: true
        level: adapter_output
        dtype: float16
        cpu_pinned: true
        cpu_max_items: 64
        gpu_max_items: 2
        async_copy: true
        fail_if_trainable: true

    concat:
      order: [residual, dino]
      normalize_dino: fixed_layernorm
      trainable_fusion: false

  struct_decoder:
    feat_2d_channels: 48

  stage6_0:
    struct_event_decoder:
      feat_2d_dim: 48
      event_dim: 64
      token:
        token_dim: 64

    base_measurement:
      source_evidence_grad_mode: train_2d_detach_alpha
      train_2d_frontend: true
      train_residual_unet: true
      train_fusion_neck: false
      train_v4_lift: false
      train_dinov2: false
      detach_v4_outputs: false
      detach_source_render_for_cnn: true
```

`train_fusion_neck=false` 是因为本设计没有 fusion neck。

---

# 12. 与 GRLD 的接口关系

DINO concat 改的是：

```text
image feature -> parent lifting -> parent_event
```

GRLD 改的是：

```text
parent_event + child Gaussian relation -> fine_event
```

二者是互补的：

```text
DINO 给 parent_event 提供低频语义/材质上下文；
GRLD 根据 child 当前 Gaussian 状态把 parent_event 分配给 fine child。
```

不要让 DINO feature 直接进入 child relation。DINO 是 parent observation；child relation 是 Gaussian state geometry/appearance residual。二者在 parent-conditioned operator 中结合。

---

# 13. 实施文件清单

## GRLD

```text
models/iforward/biggs_relational_decoder.py
models/iforward/cuda_grld_decode.py
models/iforward/csrc/grld_decode.cu
models/iforward/csrc/grld_decode_ext.cpp
models/iforward/biggs_event_decoder.py        # 接入 mode=gaussian_relational 或替换旧 decoder
models/streetforward/minimal_trainer_stage6_0.py  # 传入 parent_runtime.child_cache / stats
```

## DINO concat

```text
models/feature_extractors/dinov2_residual_concat.py
models/feature_extractors/__init__.py
models/iforward/dino_feature_cache.py
models/streetforward/minimal_trainer_stage6_0.py
models/streetforward/minimal_trainer_stage4_5.py
configs/iforward/iforward_stage2_0_biggs_grld_dinov2base_concat.yaml
```

## 测试

```text
tests/test_iforward_grld_decoder.py
tests/test_iforward_grld_cuda_decode.py
tests/test_dinov2_residual_concat_extractor.py
tests/test_iforward_dino_cache.py
tests/test_iforward_stage2_0_grld_dino_concat.py
```

---

# 14. 训练与验证计划

## 14.1 Ablation 顺序

```text
A. v28 baseline: residual_only + WHDD compact fixed basis
B. residual_only + GRLD reference
C. residual_only + GRLD fused CUDA
D. DINO concat + GRLD reference
E. DINO concat + GRLD fused CUDA
```

不要同时改 DINO 与 decoder 后直接比较最终结果；需要中间 ablation。

## 14.2 关键指标

质量：

```text
current_psnr
current_ssim
current_l1
current_lpips
parent_render_psnr_diag
```

优化行为：

```text
child_delta_variance_within_parent
mean_delta_norm
opacity_delta_norm
sh_delta_norm
detail_to_base_ratio
```

性能：

```text
dino_cache_hit_l1/l2/miss
dino_backbone_ms
residual_unet_ms
parent_lifting_ms
grld_relation_ms
grld_decode_ms
peak_alloc_gb
```

结构守恒：

```text
grld/weighted_mean_error
grld/relation_centering_error
grld/dynamic_mass_nan_ratio
grld/relation_cov_norm
grld/relation_mass_norm
grld/relation_sh_norm
```

## 14.3 验收标准

GRLD：

```text
1. zero-init 时等价 parent broadcast。
2. 修改 child opacity / scale / SH，会改变 fine_event。
3. weighted mean preservation error < 1e-4 fp32 reference。
4. 不构造 [N,R,E] 中间量。
5. 充分训练后 PSNR 不低于 WHDD baseline；理想上更高。
```

DINO concat：

```text
1. cached/uncached DINO 输出在 fp16 tolerance 内一致。
2. residual_unet grad 非零。
3. DINO 参数 grad 为 None。
4. parent lifted feature channel = 48。
5. DINO cache hit 后每 repeat 不调用 DINO backbone。
```

---

# 15. 风险与处理

## 风险 1：GRLD relation 太强，导致 child detail 噪声变大

处理：

```text
- residual_scale init 0.005 或 0.01
- DetailHead zero-init
- gate 初始化为 1
- 增加 detail_to_base_ratio 监控
```

## 风险 2：DINO 语义支路压制 residual 支路

处理：

```text
- DINO 输出做 fixed LayerNorm
- residual/DINO 直接 concat，不做 image conv fusion
- parent TokenBuilder 内用 trainable projection 学习通道权重
- 记录 residual/dino lifted norm ratio
```

## 风险 3：relation 使用 dynamic mass 导致不稳定

处理：

```text
- mass clamp_min
- mass log 输入 clamp
- parent 内 singleton 直接 relation zero
- entropy / child_count 记录异常 group
```

## 风险 4：rigid canonical relation 暂时缺失

处理：

```text
P0: world relation fallback
P1: 接入 object canonical transform
P2: 对 rigid relation 做 canonical invariance test
```

---

# 16. 最终推荐配置片段

```yaml
model:
  version: stage2_0_biggs_grld_dinov2base_concat48
  feat_2d_channels: 48

  feature_extractor:
    type: dinov2_residual_concat
    residual_unet:
      in_channels: 6
      feat_channels: 32
      base_channels: 32
      feature_downscale: 1
      depth: 3
      bilinear: true
    dino:
      model_name: vit_base_patch14_reg4_dinov2
      pretrained: true
      freeze: true
      freeze_adapter: true
      out_channels: 16
      intermediate_layers: [4, 8, 11]
      pad_to_patch_multiple: 14
      cache:
        enable: true
        level: adapter_output
        dtype: float16
        cpu_pinned: true
        cpu_max_items: 64
        gpu_max_items: 2
        async_copy: true
        fail_if_trainable: true
    concat:
      order: [residual, dino]
      trainable_fusion: false
      normalize_dino: fixed_layernorm

  iforward:
    biggs:
      child_decoder:
        mode: gaussian_relational
        parent_event_dim: 64
        fine_event_dim: 16
        relation_dim: 12
        rank: 4
        fused_cuda: true
        detach_relation_inputs: true
        use_dynamic_child_cache: true
        residual_scale:
          per_branch: true
          init_bg: 0.01
          init_distant: 0.01
          init_rigid: 0.01
        relation:
          use_xyz: true
          use_diag_cov: true
          use_mass: true
          use_opacity: true
          use_sh_dc: true
          use_sh_rest_energy: true
          rigid_space: canonical_or_world_fallback
        sibling_centering:
          enable: true
          weight: dynamic_mass
          summary: true
          entropy: true

  stage6_0:
    base_measurement:
      source_evidence_grad_mode: train_2d_detach_alpha
      train_2d_frontend: true
      train_residual_unet: true
      train_fusion_neck: false
      train_v4_lift: false
      train_dinov2: false
      detach_v4_outputs: false
      detach_source_render_for_cnn: true
    struct_event_decoder:
      feat_2d_dim: 48
      event_dim: 64
    posterior_updater:
      event_dim: 16
      hidden_dim: 32
```

---

# 17. 总结

本方案不是把当前 WHDD 简单加宽，而是重新定义 parent→child 的信息分配方式：

```text
parent 提供当前 2D evidence 与区域级 context；
child 提供当前 Gaussian state relation；
sibling centering 保证 parent 是 coarse/DC component；
parent-conditioned low-rank operator 决定 relation 如何转成 fine event。
```

DINOv2-base 的重新引入也不应恢复旧式 image-space fusion，而应作为 frozen semantic channel 与 residual channel 直接 concat 后一起 lifting：

```text
residual 32 + DINO 16 = 48
```

这样上下文建模仍然留给 parent GS / PTv3 / Mamba，而不是过早在 image space 固化。
