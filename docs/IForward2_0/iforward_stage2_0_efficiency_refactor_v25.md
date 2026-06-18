# IForward Stage 2_0 效率重构详细实施方案

## Fixed Assignment + Forward-only Projector + Incremental Parent Stats + Frozen DINO Cache + R=3 WHDD

基线版本：`drivestudio_stage6_refactor_context_20260618_v25`  
目标版本建议：`stage2_0_biggs_incremental_whdd`  
文档状态：实施方案，尚不代表代码已经完成

---

## 0. 核心结论

本次改造只保留四个高价值变化：

```text
1. projector assignment 固定；parent projector 只做 forward。
   训练 2D residual UNet / fusion / parent encoder，
   不训练 alpha/T 对 parent/fine geometry 的 Jacobian。

2. 每个 block 开始做一次 exact diagonal parent refresh；
   block 内后续 repeat 通过 child state 变化增量维护 parent sufficient stats。

3. DINOv2 全分支冻结，按 source image 缓存静态 DINO feature；
   每个 repeat 只重算 parent render、residual UNet、fusion 和 parent lifting。

4. 删除当前 rank-8 动态 child MLP decoder；
   替换为 R=3 Weighted Hierarchical Detail Decoder（WHDD）：
   固定、加权零均值的 parent-local XYZ basis + parent-only detail head。
```

最终热路径：

```text
block enter
    ├─ 获取固定 assignment / fixed WHDD basis
    ├─ exact CUDA diagonal projector（forward-only）
    │      └─ parent sufficient stats + parent params + child contribution cache
    └─ 获取或生成 frozen DINO cache

repeat k
    ├─ parent params 渲染 source RGB
    ├─ residual UNet(parent RGB, GT RGB)
    ├─ fusion(cached DINO, residual feature)
    ├─ parent alpha/T 2D lifting [M, C]
    ├─ parent AnchorTokenBuilder + xCPE/far MLP
    ├─ R=3 WHDD：parent event -> fine event
    ├─ 原 posterior updater：fine event -> fine delta
    ├─ 原可导 LocalGSState.apply_delta
    └─ 若不是 block 最后一个 repeat：
          no-grad incremental parent stats update
          -> next parent params

block exit
    └─ 丢弃 block-local parent runtime；下一 block exact refresh
```

这套设计保留 IForward 的核心语义：

```text
固定当前帧的可学习 2D evidence，
在 K 次 repeat 中迭代更新 fine GS state；
parent 是 fine state 的低维 observation carrier / pooling state，
而不是需要跨 repeat 完整 BPTT 的可训练几何节点。
```

---

# 1. v25 当前实现检查

## 1.1 已经正确实现的部分

v25 已经具备：

```text
- assignment cache_scope = scene_segment_topology
- ignore_episode_id = true
- vectorized_sort_segment builder
- sort_children = none
- CUDA exact diagonal parent projector forward
- parent quat 固定 identity
- source_evidence_grad_mode = train_2d_detach_alpha
- residual UNet / fusion neck 可训练
- V4 lifting backward 只回传到 feat2d
- scheduler 已切换 scene_round_robin_episode
- validation rollout shape 已可配置
```

这些部分不应回退。

## 1.2 当前仍存在的主要问题

### 问题 A：projector 主路径仍然挂着昂贵的 autograd backward

当前文件：

```text
models/iforward/cuda_parent_projector.py
```

`_BigGSParentProjectDiagFn.forward()` 保存：

```text
means
scales_log
quats
opacity_logit
sh_dc
sh_rest
child_mass
child_order
parent_count
```

其 backward 不是 CUDA backward，而是重新调用：

```text
project_biggs_parent_diag_reference_tensors(...)
torch.autograd.grad(...)
```

这会造成：

```text
- 保存 N 级 child tensor
- backward 时重新构建完整 PyTorch projector 图
- 额外 index_add / quaternion / exp / topk / SH 聚合
- 与当前 V4 alpha/T geometry stop-gradient 语义不一致
```

当前 V4 fused lifting backward 只返回 `grad_feat2d`，对：

```text
means2d
conics
opacities
```

均返回 `None`。因此 parent projector backward 并不能训练 alpha/T geometry。

### 问题 B：每个 repeat 都重新从 fine GS 做完整 projection

当前路径在：

```text
MinimalStreetForwardStage6_0._observe_stage2_0_biggs_measurement()
```

每次 observe 都执行：

```text
project bg
project distant
build active rigid assignment
project active rigid
```

但同一 block 的 K 次 repeat：

```text
assignment 不变
source frame 不变
rigid active set / instance pose 不变
parent topology 不变
```

真正变化的是 fine GS 属性。因此应该维护 parent state，而不是反复“重新构建 parent”。

### 问题 C：frozen DINO 每 repeat 仍完整执行

当前：

```text
DINOv2UNetFusionExtractor.forward()
    residual_unet(x6)
    dino_adapter(rgb)
    fusion_neck(cat(...))
```

DINO backbone、DINO adapter 的 projection/fuse 当前都没有被 trainability 配置标记为可训练；可训练部分是：

```text
image_feature_extractor.residual_unet
image_feature_extractor.fusion_neck
```

因此 DINO adapter 的最终输出是 source RGB 的确定性静态特征，重复执行没有训练收益。

### 问题 D：rank-8 child decoder 在 N 级做了过重计算

当前：

```text
basis = basis_mlp(parent_event)          [M, R, E]
coeff = coeff_mlp(child_code)            [N, R]
coeff mean-centering                     scatter over N
basis.index_select(parent_id)            logical [N, R, E]
einsum                                   [N, E]
```

当前典型规模：

```text
N ≈ 50 万
R = 8
E = 96
```

其主要问题不是参数量，而是：

```text
- 每 child MLP
- 每 repeat coefficient scatter-centering
- parent basis 向 N 行 gather
- 大量 [N,R,E] 级中间访问/临时量
```

而已有监控显示 scaled child residual 相对 parent event 很小，说明 child decoder 更适合“共享 parent + 低阶固定细节模式”，而不是动态 rank-8 child network。

---

# 2. 目标架构与状态生命周期

必须区分四类对象：

| 对象 | 生命周期 | 是否可导 | 作用 |
|---|---:|---:|---|
| `BigGSAssignment` | scene/segment/topology | 否 | child-parent 拓扑 |
| `WHDDFixedBasis` | assignment 或 block（rigid active） | 否 | parent 到 child 的固定 detail 模式 |
| `BigGSParentRuntime` | block 内 K repeats | 否 | parent sufficient stats、params、child contribution cache |
| 2D feature / event / delta graph | 当前 rollout | 是 | 训练 2D frontend、parent encoder、WHDD detail head、posterior updater |

不要继续把这些对象都塞进 `IForwardBigGSState`。

建议：

```text
IForwardBigGSState
    只保存 assignment + static WHDD basis
    可跨 episode/runtime cache

BigGSBlockRuntime
    只保存在 rollout 的 Python 局部变量
    block enter 初始化
    block exit 丢弃
```

---

# 3. 第一步：固定 assignment + forward-only projector + 只训练 2D feature

## 3.1 assignment 固定语义

主线：

```text
assignment key =
(scene_id,
 segment_id,
 topology_signature,
 assignment_config_hash)
```

明确不包含：

```text
episode_id
current fine means
current opacity
current scales
```

因为 assignment 是拓扑，不是当前状态。

### topology signature 必须加强

v25 当前 cache key 对 bg/distant 主要使用 point count，对 rigid 额外使用 point ID hash。仅 point count 无法发现“同样数量但点顺序变化”的错误。

优先使用 dataset/asset 提供的稳定标识：

```text
scene_asset_version
segment_asset_version
bg_topology_id
distant_topology_id
rigid_point_id_hash
```

缺失时才 fallback：

```text
(num_points, stable point id checksum)
```

禁止用动态 means checksum，因为 fine means 会被 optimizer 更新。

## 3.2 projector 改成真正 forward-only

新增 backend：

```yaml
parent_projector:
  backend: cuda_exact_diag_forward_only
  grad_mode: stop_geometry
  grad_to_local_state: false
```

实现不再经过 `torch.autograd.Function`：

```python
@torch.no_grad()
def project_biggs_parent_diag_cuda_forward_only(...):
    outputs = ext.biggs_parent_project_diag_forward(...)
    return tuple(x.detach() for x in outputs)
```

保留原 reference autograd projector 仅用于单元测试：

```text
torch_exact_diag_reference
```

不允许训练配置误用。

## 3.3 明确梯度边界

主线计算图：

```text
parent params.detach()
    -> parent render RGB.detach()
    -> residual UNet / fusion（可训练）
    -> feat2d（可训练）
    -> V4 lifting
       backward 只到 feat2d
    -> parent encoder（可训练）
    -> WHDD（可训练 parent detail head）
    -> posterior updater（可训练）
    -> fine delta
    -> differentiable LocalGSState.apply_delta
    -> final fine render loss
```

禁止的梯度：

```text
loss
 -> lifted parent feature
 -> parent alpha/T geometry
 -> parent projector
 -> earlier fine geometry
```

允许的梯度：

```text
loss -> V4 lifted feature -> fusion/residual UNet
loss -> parent encoder
loss -> WHDD detail head
loss -> posterior updater
loss -> delta chain -> fine GS final render
```

## 3.4 配置修正

```yaml
model:
  iforward:
    biggs:
      parent_projector:
        backend: cuda_exact_diag_forward_only
        covariance_mode: diagonal
        mass_mode: dynamic_tau_area
        grad_mode: stop_geometry
        grad_to_local_state: false
        finite_check: false

  stage6_0:
    base_measurement:
      source_evidence_grad_mode: train_2d_detach_alpha
      train_2d_frontend: true
      train_residual_unet: true
      train_fusion_neck: true
      train_v4_lift: false
      train_dinov2: false
      detach_v4_outputs: false
      detach_source_render_for_cnn: true

    struct_event_decoder:
      param_obs_codec:
        detach_params: true
        detach_obs_code: true
        detach_acc_w: true
```

`detach_source_render_for_cnn=true` 不会阻止 residual UNet 学习。网络参数梯度不要求输入 RGB 自身可导。

## 3.5 去掉不必要的 detached clone

当前 `LocalGSState.to_node_states_detached()` 会对所有 fine tensor：

```python
x.detach().clone()
```

forward-only projector 不需要 clone。新增：

```python
def to_node_states_detached_view(self):
    # detach only, no clone
```

或直接让 exact refresh / stats update 接受 `LocalBranchState` tensor，避免转换成 NodeState。

这是必要优化，不是边缘优化。对 50 万点，重复 clone means/scales/quats/opacity/SH 会造成明显带宽和显存压力。

## 3.6 代码改动

```text
models/iforward/cuda_parent_projector.py
    + forward-only wrapper
    - 主训练路径的 _BigGSParentProjectDiagFn.backward

models/iforward/biggs_parent_projector.py
    + backend cuda_exact_diag_forward_only
    + enforce outputs.requires_grad == false

models/streetforward/stage6_0/local_gs_state.py
    + detached view API 或移除 NodeState 转换依赖

models/streetforward/minimal_trainer_stage6_0.py
    + fail-fast 检查 projector grad mode
```

## 3.7 必须测试

```text
test_projector_forward_only_matches_cuda_forward
    输出与当前 exact diagonal CUDA forward 对齐

test_projector_forward_only_has_no_grad_fn
    parent outputs.requires_grad == false

test_2d_frontend_still_gets_grad
    residual_unet / fusion grad_sum > 0

test_alpha_geometry_has_no_grad
    parent means/scales/opacity grad is None

test_posterior_and_whdd_get_grad
    decoder/updater grad_sum > 0
```

---

# 4. 第二步：parent sufficient stats 增量更新 + 每 block exact refresh

## 4.1 先说明一个重要事实

如果 incremental update 实现为：

```text
apply fine delta
再额外全量扫描所有 child
重算 new contribution
重算 old contribution
reduce 到 parent
```

它仍然是 `O(N)`，而且可能比当前 projector 更慢。

因此实施方案必须区分：

```text
A. correctness reference：独立 PyTorch/CUDA delta stats update
B. production fast path：缓存旧 child contribution，单次计算 new contribution，
   分组 reduce；不再重新计算完整 old projector
```

不能只实现 A 就宣称优化完成。

## 4.2 parent sufficient stats

对 child `i`：

```text
μ_i       = means
s_i       = exp(scales_log)
τ_i       = softplus(opacity_logit)
area_i    = top2(s_i).product
m_i       = max(τ_i * area_i, min_mass)
D_i       = diag(R(q_i) diag(s_i²) R(q_i)^T)
```

定义 child contribution：

```text
w_i   = m_i                         scalar
A_i   = m_i μ_i                     [3]
B_i   = m_i (D_i + μ_i²)            [3]
U_i   = τ_i area_i                  scalar
Cdc_i = m_i sh_dc_i                 [3]
Cr_i  = m_i sh_rest_i               [B,3]
```

parent stats：

```text
W   = Σ w_i
A   = Σ A_i
B   = Σ B_i
U   = Σ U_i
Cdc = Σ Cdc_i
Cr  = Σ Cr_i
```

finalize：

```text
parent_mean = A / W
parent_var  = B / W - parent_mean² + eps
parent_scale = sqrt(clamp(parent_var))
parent_quat = identity
parent_area = top2(parent_scale).product
parent_tau = tau_parent_scale * U / parent_area
parent_opacity = opacity_cap * (1 - exp(-parent_tau))
parent_SH = C / W
```

这与当前 exact diagonal projector 数学一致。

## 4.3 block runtime 数据结构

新增：

```python
@dataclass
class BigGSParentStats:
    weight_sum: Tensor          # [M], fp32
    weighted_mean_sum: Tensor   # [M,3], fp32
    weighted_second_sum: Tensor # [M,3], fp32
    tau_area_sum: Tensor        # [M], fp32
    weighted_sh_dc_sum: Tensor  # [M,3], fp32
    weighted_sh_rest_sum: Tensor# [M,B,3], fp32
    parent_count: Tensor        # [M]

@dataclass
class BigGSChildContributionCache:
    mass: Tensor                # [N], fp32/fp16
    tau_area: Tensor            # [N], fp32/fp16
    diag_cov: Tensor            # [N,3], fp32/fp16

@dataclass
class BigGSParentBranchRuntime:
    stats: BigGSParentStats
    params: Dict[str, Tensor]
    child_cache: BigGSChildContributionCache
    assignment_signature: str

@dataclass
class BigGSBlockRuntime:
    bg: BigGSParentBranchRuntime
    distant: Optional[BigGSParentBranchRuntime]
    rigid_active: Optional[BigGSParentBranchRuntime]
    rigid_active_assignment: Optional[BigGSRigidActiveAssignment]
    source_frame_idx: int
    block_id: int
    exact_refresh_count: int
    incremental_update_count: int
```

该 runtime：

```text
- 不进入 optimizer
- 不进入 autograd graph
- 不写入跨 block persistent state
- 不随 checkpoint 保存
```

assignment 仍在 `IForwardBigGSState`。

## 4.4 exact refresh

在 block enter：

```text
CUDA exact diagonal projector
    -> parent stats
    -> parent params
    -> child contribution cache
```

因此需要扩展当前 projector CUDA 输出：

```text
现有输出：parent params + mass sum/mean
新增输出：A, B, U, Cdc, Cr, child mass, child tau_area, child diag_cov
```

如果不希望一次返回过多，可拆：

```text
biggs_parent_stats_init_forward
biggs_parent_stats_finalize
```

推荐一个 init kernel 完成全部初始化，避免再次扫描 child。

## 4.5 incremental update

child state 更新：

```text
old_local_state
    --原可导 apply_delta-->
new_local_state
```

随后在 `torch.no_grad()` 中：

```text
new contribution = contribution(new child state)
Δstats = Σ(new contribution - old cached contribution)
new parent stats = old parent stats + Δstats
new parent params = finalize(new parent stats)
更新 child contribution cache
```

### 为什么缓存 `mass/tau_area/diag_cov`

旧 child state 本身仍存在，但若每次从旧 state 重算：

```text
exp(scales)
softplus(opacity)
quat -> diag covariance
```

会把 nonlinear 计算做两次。

缓存：

```text
old mass
old tau_area
old diag_cov
```

即可从旧 state 便宜恢复：

```text
old A = old mass * old mean
old B = old mass * (cached diag_cov + old mean²)
old C = old mass * old SH
```

新 contribution 只计算一次。

按 50 万 child 估算，缓存 5 个标量/child：

```text
mass 1 + tau_area 1 + diag_cov 3 = 5
fp32 ≈ 10 MB
fp16 ≈ 5 MB
```

相对当前约 40GB 峰值可接受，并显著小于缓存完整 `A/B/SH contribution`。

## 4.6 production CUDA kernel

新增：

```text
models/iforward/csrc/biggs_parent_stats_update.cu
models/iforward/csrc/biggs_parent_stats_ext.cpp
models/iforward/cuda_parent_stats.py
```

接口：

```python
new_stats, new_params, new_child_cache = update_parent_stats_forward_only(
    old_stats,
    old_child_state,
    new_child_state,
    old_child_cache,
    child_order,
    parent_start,
    parent_count,
    branch_update_mask,
    projector_constants,
)
```

建议 one block / parent：

```text
parent group 已连续存放在 child_order
每个 parent child cap 为 32/64
block 内 reduce，无全局 atomic
```

每个 child：

```text
读取 old mean/SH + old cache
读取 new state
计算 new mass/tau_area/diag_cov
形成 ΔW/ΔA/ΔB/ΔU/ΔC
block reduce
```

每个 parent：

```text
stats += Δstats
finalize params
```

### distant 专用 fast path

当前 distant 不更新：

```text
means
scales
quat
```

但 opacity 改变会改变 mass，进而改变 parent weighted mean/variance，所以仍需更新 W/A/B。

可以跳过：

```text
new diag_cov 计算
new scale/quaternion 计算
```

直接复用 cached diag_cov 和 old geometry，只计算新 opacity mass 与 SH。

## 4.7 rollout 流程改动

当前 `is_block_enter/is_block_exit` 在 observe 之后才计算。必须提前到每个 step 开头：

```python
for step_pos, step in ...:
    next_step = ...
    is_block_enter, is_block_exit = flags(step, next_step)

    if is_block_enter:
        parent_runtime = exact_refresh(...)

    measurement = bridge.observe(
        ...,
        biggs_parent_runtime=parent_runtime,
    )

    old_local_state = local_state
    local_state, delta, update_aux = bridge.apply_update(...)

    if not is_block_exit:
        parent_runtime = bridge.update_parent_runtime(
            runtime=parent_runtime,
            old_local_state=old_local_state,
            new_local_state=local_state,
            delta=delta,
        )
```

最后一个 repeat 不再更新 parent stats，因为没有下一次 parent observation：

```text
skip_incremental_on_block_exit = true
```

K=8 时：

```text
1 exact refresh + 7 incremental updates
```

而不是 8 次 exact projector。

## 4.8 rigid active 特殊处理

BG/distant：

```text
all children active
assignment/basis 可复用
```

Rigid：

```text
assignment 固定在 canonical object space
active S / near-out split 随 source frame 变化
```

因此：

```text
- block enter 构建 active rigid assignment
- exact refresh 使用 world-space active child
- block 内 pose/source frame 固定，active assignment 固定
- incremental update 维护 active world-space parent stats
- 下一个 block/source frame 必须重新 exact refresh
```

P0 可继续用现有 route helper生成 old/new world child tensor。P1 再缓存每 child instance transform，减少 route 重算。

## 4.9 数值漂移检查

即使公式精确，浮点加减仍可能累积误差。每若干 block 做 no-grad 对照：

```text
incremental params vs fresh exact projector params
```

日志：

```text
iforward/biggs/stats_drift_mean_xyz
iforward/biggs/stats_drift_scale_log
iforward/biggs/stats_drift_opacity_logit
iforward/biggs/stats_drift_sh
```

默认：

```yaml
parent_state:
  exact_refresh_policy: block_enter
  drift_check_interval_blocks: 100
  drift_fail_threshold: 1.0e-3
```

## 4.10 必须测试

```text
test_incremental_one_step_matches_exact_refresh
    old state + delta 后，incremental parent 与 exact reproject 对齐

test_incremental_multiple_steps_matches_exact_refresh
    连续 8 次随机小 delta

test_incremental_distant_opacity_reweights_geometry
    distant geometry不更新，但 opacity mass 变化时 parent mean/scale 应正确变化

test_incremental_rigid_active_matches_world_exact

test_block_exit_skips_unused_incremental_update

test_parent_runtime_has_no_autograd_graph
```

性能测试必须区分：

```text
exact_refresh_ms
incremental_update_ms
parent_finalize_ms
child_cache_bytes
```

验收条件不是“结构改了”，而是：

```text
incremental_update_ms 显著低于 exact_refresh_ms
```

若独立 incremental kernel 未达标，不进入主线，继续做 contribution cache / grouped CUDA 优化。

---

# 5. 第三步：缓存 frozen DINO feature

## 5.1 缓存层级

当前可训练部分：

```text
residual_unet
fusion_neck
```

DINO adapter 整体冻结。因此主线缓存：

```text
DINOv2BackboneAdapter 的最终 dino_feat [V,Hf,Wf,Cd]
```

而不是只缓存 backbone token。

启动时 fail-fast：

```python
assert all(not p.requires_grad for p in image_feature_extractor.dino_adapter.parameters())
```

若未来要训练 DINO adapter 的 projection/fuse，则切换为缓存 raw backbone intermediates；本阶段不做。

## 5.2 extractor API 重构

当前：

```python
DINOv2UNetFusionExtractor.forward(images)
```

改为：

```python
def extract_dino_feature(self, rgb, *, target_hw) -> Tensor:
    # no-grad, detached

def extract_residual_feature(self, x6) -> Tensor:
    # trainable

def fuse_features(self, dino_feat, residual_feat) -> Tensor:
    # trainable fusion neck

def forward(self, images, *, cached_dino=None):
    residual_feat = extract_residual_feature(images)
    if cached_dino is None:
        dino_feat = extract_dino_feature(images[:,:3], target_hw=...)
    else:
        dino_feat = cached_dino
    return fuse_features(dino_feat, residual_feat)
```

## 5.3 cache key

建议：

```text
(scene_id,
 segment_id,
 source_frame_idx,
 ordered_camera_ids,
 ordered_image_ids,
 image_hw,
 feature_hw,
 dino_model_fingerprint,
 preprocessing_fingerprint)
```

`dino_model_fingerprint` 包括：

```text
model_name
weights/checkpoint hash
intermediate_layers
pad_to_patch_multiple
out_channels
adapter state version
```

禁止只用 `source_indices`，因为它可能只是当前 batch 内局部下标。

## 5.4 两级 lazy cache

不是离线预处理，而是 runtime lazy cache：

```text
L1：当前 block GPU hot cache
    max_items = 1~4
    直接复用，无传输

L2：runtime CPU pinned FP16 LRU
    max_items = 32~128
    跨 episode/scene round-robin 复用
```

流程：

```text
查 L1 -> hit
否则查 L2 -> async H2D -> 放入 L1
否则运行 DINO -> detach/cast -> 写 L1/L2
```

配置：

```yaml
model:
  feature_extractor:
    dino:
      freeze: true
      cache:
        enable: true
        level: adapter_output
        dtype: float16
        cpu_pinned: true
        cpu_max_items: 64
        gpu_max_items: 2
        async_copy: true
        fail_if_trainable: true
```

## 5.5 和训练图的关系

cached DINO：

```text
requires_grad = false
```

但 fusion 仍可训练：

```text
cached DINO ─┐
             ├─ trainable fusion neck -> loss
residual UNet┘
```

所以缓存不会破坏：

```text
residual UNet gradient
fusion neck gradient
```

## 5.6 缓存失效条件

以下任一变化即 miss：

```text
source image / camera order
image resize / crop / augmentation
DINO weights
DINO intermediate layers
feature resolution
DINO adapter parameters
```

若 source image pipeline 存在随机 augmentation，则必须把 augmentation signature 放入 key；否则禁用跨 block cache，仅保留 block-local cache。

## 5.7 代码改动

```text
models/feature_extractors/dinov2_unet_fusion.py
    + split extract/fuse API

models/iforward/dino_feature_cache.py
    + L1/L2 cache class

models/streetforward/minimal_trainer_stage4_5.py
    + _render_source_scene_only_for_cnn(..., cached_dino=...)

models/streetforward/minimal_trainer_stage6_0.py
    + cache key builder
    + cache metrics
```

## 5.8 日志

```text
iforward/dino/cache_hit_l1
iforward/dino/cache_hit_l2
iforward/dino/cache_miss
iforward/dino/backbone_ms
iforward/dino/h2d_ms
iforward/dino/cache_cpu_mb
iforward/dino/cache_gpu_mb
iforward/dino/feature_dtype_id
```

## 5.9 必须测试

```text
test_cached_dino_matches_uncached

test_cached_dino_keeps_residual_and_fusion_grad

test_cached_dino_has_no_grad

test_dino_cache_key_changes_with_camera_order

test_dino_cache_invalidates_on_model_fingerprint

test_dino_cache_lru_eviction
```

---

# 6. 第四步：rank-8 child MLP -> R=3 WHDD fixed-basis decoder

## 6.1 设计依据

层级表示中，parent 应表示 children 的低频/common component，children 只需要表达相对 parent 的 detail。点云层级变换中的 approximation/DC + detail/AC 分解，以及 Hierarchical 3DGS 的 parent/children 平滑过渡，都支持这一结构。

Stage 2_0 不需要让每个 child 动态预测任意 rank-8 coefficient。更高效的形式是：

```text
parent event = common/DC
固定几何 basis = child detail coordinates
parent detail head = 当前观测下每个 detail mode 的向量
```

## 6.2 固定 R=3 basis

对 parent `k` 内 child `i`，用 assignment-time static mass：

```text
π_i = child_mass_i / Σ_j child_mass_j
```

### BG / distant

使用初始/asset world coordinate：

```text
center_k = Σ π_i x_i
std_k = sqrt(Σ π_i (x_i-center_k)² + eps)
φ_i = (x_i - center_k) / std_k
```

因此：

```text
Σ π_i φ_i = 0
```

R=3 对应：

```text
φ_x, φ_y, φ_z
```

若某一维退化：

```text
std < basis_min_std -> 该 mode 置 0
```

不做 per-parent eig/PCA；diagonal parent 本身已经定义 axis-aligned 三个变化方向。

### Rigid

global assignment 使用 canonical object coordinate 生成基础几何。

但是当前 source frame 只激活 `S`，并且 active assignment 还可能按 near/out 拆开同一 global parent。全局 basis 在 active subset 上不再保证零均值。

因此 rigid 使用：

```text
block enter：
    对 active rigid assignment 在 canonical child coordinates 上
    重新 weighted center / normalize
    得到 block-local fixed basis_S [N_S,3]
```

它只在 block enter 计算一次，不在 repeat 内重算。

## 6.3 assignment/state 扩展

```python
@dataclass
class BigGSBranchAssignment:
    ...
    child_basis: Optional[Tensor]       # [N,3]
    basis_valid: Optional[Tensor]       # [M,3]
    basis_weight_sum: Optional[Tensor]  # [M]
    basis_version: int = 1
```

Rigid active：

```python
@dataclass
class BigGSRigidActiveAssignment:
    ...
    child_basis_S: Optional[Tensor]     # [N_S,3]
    basis_valid: Optional[Tensor]       # [M_active,3]
```

存储：

```text
child_basis fp16 或 fp32
```

50 万点、R=3：

```text
fp16 ≈ 3 MB
fp32 ≈ 6 MB
```

主线先用 fp16；构建时保留 fp32 校验 mean error，再量化。

## 6.4 WHDD 公式

parent event：

```text
e_k ∈ R^E
```

parent-only detail head：

```text
D_k = DetailHead(e_k) ∈ R^(3×E)
```

推荐第一版：

```python
LayerNorm(E)
Linear(E, 3E)
```

最后一层 zero-init。

child event：

```text
e_i = e_parent(i)
      + γ_branch Σ_{r=1..3} φ_i,r D_parent(i),r
```

由于：

```text
Σ_i π_i φ_i,r = 0
```

自动得到：

```text
Σ_i π_i e_i = e_parent
```

不再需要每 repeat 的 scatter mean-centering。

## 6.5 删除的模块

删除主路径：

```text
child_code MLP
basis_mlp producing rank-8 dynamic bases
coeff_mlp
per-repeat coefficient scatter mean
child/parent geometry param gather
route flag / relative opacity / relative scale child code
```

WHDD 不接收 child trainable geometry，因此天然符合：

```text
不训练 alpha/T geometry Jacobian
```

## 6.6 新 decoder 类

建议新增而不是继续向旧类塞 mode：

```text
models/iforward/whdd_event_decoder.py
```

```python
class WeightedHierarchicalDetailDecoder(nn.Module):
    def __init__(self, event_dim=96, rank=3, ...): ...

    def decode_branch(
        parent_event,
        child_to_parent,
        child_basis,
        branch_id,
    ) -> fine_event:
        detail = self.detail_head(parent_event).view(M, 3, E)
        return whdd_decode(...)
```

support/valid/obs_code 继续 parent broadcast。

## 6.7 fused CUDA decoder

不能使用：

```python
parent_detail.index_select(0, parent_id)  # [N,3,E]
```

这会重新制造大临时量。

新增 fused op：

```python
fine_event = whdd_decode(
    parent_event,       # [M,E]
    parent_detail,      # [M,3,E]
    child_basis,        # [N,3]
    child_to_parent,    # [N]
    gamma,              # scalar or [3 branches]
)
```

forward：

```text
for child i, event channel e:
    p = parent_id[i]
    out[i,e] = parent_event[p,e]
             + gamma * (
                 phi[i,0] * detail[p,0,e]
               + phi[i,1] * detail[p,1,e]
               + phi[i,2] * detail[p,2,e])
```

backward：

```text
grad_parent_event[p,e] = Σ_i grad_out[i,e]

grad_parent_detail[p,r,e]
    = gamma Σ_i phi[i,r] grad_out[i,e]

grad_gamma
    = Σ_i,e grad_out[i,e] Σ_r phi[i,r] detail[p,r,e]
```

`child_basis` 默认无梯度。

利用 `child_order/parent_start/parent_count` 做 one block per parent backward，可避免 atomic。

## 6.8 配置

```yaml
model:
  iforward:
    biggs:
      assignment:
        build_whdd_basis: true
        whdd_basis:
          type: weighted_parent_local_xyz
          rank: 3
          dtype: float16
          min_std: 1.0e-4
          rigid_space: canonical
          active_rigid_recenter: true

      child_decoder:
        mode: whdd_fixed_basis
        rank: 3
        event_dim: 96
        detail_head:
          type: linear
          zero_init_last: true
        residual_scale_init: 1.0e-2
        residual_scale_learnable: true
        residual_scale_per_branch: true
        fused_cuda: true
        mean_preserve: true
```

## 6.9 初始化与 checkpoint 迁移

旧 checkpoint 包含：

```text
basis_mlp
coeff_mlp
residual_mlp
```

WHDD 无法严格转换。迁移策略：

```text
- 加载 parent struct decoder / posterior updater / 2D frontend
- 丢弃旧 child decoder weights
- WHDD detail head zero-init
- residual scale 可从旧 checkpoint scalar 复制，或初始化 0.01
```

训练前几百 step parent broadcast 占主导，WHDD detail 逐渐学习，不会突然破坏 posterior input 分布。

## 6.10 日志

```text
iforward/whdd/decode_ms
iforward/whdd/basis_weighted_mean_error_bg
iforward/whdd/basis_weighted_mean_error_distant
iforward/whdd/basis_weighted_mean_error_rigid
iforward/whdd/basis_valid_ratio
iforward/whdd/detail_norm
iforward/whdd/scaled_detail_norm
iforward/whdd/gamma_bg
iforward/whdd/gamma_distant
iforward/whdd/gamma_rigid
iforward/whdd/mean_preserve_error
```

## 6.11 必须测试

```text
test_whdd_basis_weighted_zero_mean

test_whdd_singleton_parent_has_zero_detail

test_whdd_degenerate_axis_is_masked

test_whdd_rigid_active_basis_zero_mean_after_split

test_whdd_forward_matches_reference

test_whdd_cuda_backward_matches_reference

test_whdd_no_nre_intermediate_allocation

test_whdd_zero_init_equals_parent_broadcast

test_whdd_weighted_mean_preserves_parent_event
```

---

# 7. 完整配置草案

```yaml
model:
  iforward:
    version: stage2_0_biggs_incremental_whdd

    biggs:
      enable: true

      assignment:
        method: branch_aware_voxel_cap
        cache_scope: scene_segment_topology
        ignore_episode_id: true
        cache_max_items: 64
        cache_device_copy: true
        builder: vectorized_sort_segment
        sort_children: none
        topology_signature_source: dataset_asset

        build_whdd_basis: true
        whdd_basis:
          type: weighted_parent_local_xyz
          rank: 3
          dtype: float16
          min_std: 1.0e-4
          rigid_space: canonical
          active_rigid_recenter: true

        mass_init: tau_area
        bg:
          voxel_size: 0.5
          target_children_per_parent: 16
          max_children_per_parent: 32
          max_parent_radius: 1.0
        distant:
          voxel_size: 3.0
          target_children_per_parent: 32
          max_children_per_parent: 64
          max_parent_radius: 6.0
        rigid:
          voxel_size: 0.3
          target_children_per_parent: 16
          max_children_per_parent: 32
          max_parent_radius: 0.6

      parent_projector:
        backend: cuda_exact_diag_forward_only
        covariance_mode: diagonal
        mass_mode: dynamic_tau_area
        grad_mode: stop_geometry
        grad_to_local_state: false
        stats_dtype: float32
        min_scale: 1.0e-3
        max_scale_bg: 0.60
        max_scale_distant: 3.0
        max_scale_rigid: 0.45
        opacity_cap: 0.90
        opacity_min: 1.0e-6
        tau_parent_scale_bg: 0.5
        tau_parent_scale_distant: 0.7
        tau_parent_scale_rigid: 0.5
        eps: 1.0e-6
        min_child_mass: 1.0e-8
        finite_check: false

      parent_state:
        mode: incremental_sufficient_stats
        exact_refresh_policy: block_enter
        update_after_each_nonfinal_repeat: true
        skip_update_on_block_exit: true
        child_cache_dtype: float32
        drift_check_interval_blocks: 100
        drift_fail_threshold: 1.0e-3

      observe:
        parent_scene_for_lifting: true
        parent_scene_for_cnn: true
        detach_parent_geometry: true
        return_debug_stats: false

      child_decoder:
        mode: whdd_fixed_basis
        rank: 3
        event_dim: 96
        detail_head:
          type: linear
          zero_init_last: true
        residual_scale_init: 1.0e-2
        residual_scale_learnable: true
        residual_scale_per_branch: true
        fused_cuda: true

  feature_extractor:
    dino:
      freeze: true
      cache:
        enable: true
        level: adapter_output
        dtype: float16
        cpu_pinned: true
        cpu_max_items: 64
        gpu_max_items: 2
        async_copy: true
        fail_if_trainable: true

  stage6_0:
    base_measurement:
      source_evidence_grad_mode: train_2d_detach_alpha
      train_2d_frontend: true
      train_residual_unet: true
      train_fusion_neck: true
      train_v4_lift: false
      train_dinov2: false
      detach_v4_outputs: false
      detach_source_render_for_cnn: true

    struct_event_decoder:
      param_obs_codec:
        detach_params: true
        detach_obs_code: true
        detach_acc_w: true
```

---

# 8. 文件级实施清单

## 必改

```text
models/iforward/biggs_state.py
    assignment 增加 WHDD basis
    新增 parent stats/runtime dataclass

models/iforward/biggs_assignment.py
    robust topology signature
    vectorized WHDD basis builder
    rigid canonical basis

models/iforward/cuda_parent_projector.py
    forward-only 主接口
    reference autograd 仅测试保留

models/iforward/csrc/biggs_parent_projector_diag.cu
    exact init 同时输出 sufficient stats / child contribution cache

models/iforward/biggs_parent_stats.py
models/iforward/cuda_parent_stats.py
models/iforward/csrc/biggs_parent_stats_update.cu
models/iforward/csrc/biggs_parent_stats_ext.cpp
    incremental update

models/feature_extractors/dinov2_unet_fusion.py
    split DINO/residual/fusion API

models/iforward/dino_feature_cache.py
    L1/L2 lazy cache

models/iforward/whdd_event_decoder.py
models/iforward/csrc/whdd_decode.cu
models/iforward/csrc/whdd_decode_ext.cpp
    R=3 decoder

models/streetforward/minimal_trainer_stage6_0.py
    observe 使用 parent runtime
    DINO cache 接入
    parent exact refresh / active rigid helper

models/iforward/model.py
    block flags 提前
    block runtime 生命周期
    update 后增量 parent stats
```

## 修改测试

```text
tests/test_iforward_biggs_stage2_0.py
tests/test_iforward_biggs_cuda_projector.py
```

## 新增测试

```text
tests/test_iforward_biggs_parent_stats.py
tests/test_iforward_dino_cache.py
tests/test_iforward_whdd_decoder.py
tests/test_iforward_stage2_0_incremental_rollout.py
```

---

# 9. 分阶段实施顺序

## Phase A：梯度边界和 forward-only projector

完成：

```text
- forward-only backend
- grad_to_local_state=false
- param codec detach
- 2D frontend grad 验证
- 无 detached clone 热路径
```

验收：

```text
forward 数值不变
2D frontend grad 非零
parent geometry grad 为零
峰值显存下降
```

## Phase B：parent stats exact init + correctness incremental

先完成 PyTorch/reference 和 CUDA correctness 版本。

验收：

```text
8 repeat 后 incremental 与 exact reproject 参数误差受控
```

## Phase C：production incremental fast path

完成：

```text
child contribution cache
one-block-per-parent CUDA reduce
block exit skip
rigid active runtime
```

验收：

```text
incremental_update_ms 显著小于 exact_refresh_ms
normal step 降速明确
```

## Phase D：DINO cache

验收：

```text
同一 source 的 DINO backbone call count 从 K 降到 1
residual/fusion grad 不变
```

## Phase E：WHDD R=3

先 reference，再 CUDA fused。

验收：

```text
零初始化等价 parent broadcast
weighted mean preservation
无 [N,R,E] 中间量
child decode time / memory 显著下降
单帧质量接近 rank-8 baseline
```

---

# 10. Validation 与性能对比矩阵

所有实验使用固定 validation batch、相同 seed、相同 K：

| 实验 | Projector | Parent state | DINO | Decoder |
|---|---|---|---|---|
| A | 当前 CUDA + backward | 每 repeat exact | 每 repeat | rank-8 |
| B | forward-only | 每 repeat exact | 每 repeat | rank-8 |
| C | forward-only | incremental | 每 repeat | rank-8 |
| D | forward-only | incremental | cached | rank-8 |
| E | forward-only | incremental | cached | WHDD R=3 |

记录：

```text
step_time_ms
observe_ms
exact_refresh_ms
incremental_update_ms
dino_backbone_ms
parent_render_cnn_ms
parent_lifting_ms
parent_encoder_ms
child_decode_ms
update_ms
peak_alloc_gb
peak_reserved_gb
PSNR / SSIM / LPIPS
means/scale/opacity/SH delta norms
```

关键验收：

```text
B：确认 forward-only 不损伤单帧质量和 2D 训练
C：确认 incremental 参数与 exact 对齐并真实提速
D：确认 DINO call count 和 observe time 显著下降
E：确认 decoder memory/time 显著下降，质量下降可接受
```

建议 stop criteria：

```text
- incremental update 若不比 exact refresh 快至少 2x，不进入主线
- DINO cached/uncached feature 不一致超过 fp16 tolerance，停止
- WHDD 训练充分后固定验证 PSNR 比 rank-8 低 >0.3dB，检查 grouping/basis，
  不要首先提高 rank；先检查 parent footprint 和 rigid active basis
```

---

# 11. 重要风险

## 风险 1：incremental 只是换位置，没有减少计算

规避：

```text
- 缓存 old child nonlinear contribution
- one-block-per-parent grouped reduce
- 最后 repeat 不更新
- distant branch specialization
```

## 风险 2：DINO cache 缓存了未来想训练的参数输出

规避：

```text
cache level=adapter_output 时 fail_if_trainable=true
```

## 风险 3：rigid active subset 破坏 WHDD 零均值

规避：

```text
block enter 对 active assignment 重新 center basis
```

## 风险 4：固定 XYZ basis 无法表达材质/opacity 子差异

主线先不增加复杂度。若固定验证显示 WHDD 明显不足，下一优先级不是恢复 child MLP，而是：

```text
仍保持 R=3，
将三个 basis 从纯 XYZ 改为 assignment-time 对
[relative_xyz, relative_scale, relative_tau] 的加权三维低秩基。
```

这仍是固定 basis，不恢复 per-child neural decoder。

---

# 12. 最终推荐

本轮不应该继续优化 projector backward，因为主训练语义不需要它。真正一致的设计是：

```text
parent geometry/state：forward-only recurrent observation state
2D feature：可训练
alpha/T geometry Jacobian：stop-gradient
fine delta recurrence：可导
```

并通过：

```text
fixed assignment
block exact refresh
incremental sufficient stats
frozen DINO cache
R=3 fixed WHDD
```

把 Stage 2_0 从“每 repeat 重建层级表示”改成“层级状态随 fine update 演化”。

---

# 参考工作

1. Bernhard Kerbl et al., **A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets**, 2024. https://arxiv.org/abs/2406.12080
2. Eduardo Pavez et al., **Region Adaptive Graph Fourier Transform for 3D Point Clouds**, 2020. https://arxiv.org/abs/2003.01866
3. Tao Lu et al., **Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering**, 2023. https://arxiv.org/abs/2312.00109
