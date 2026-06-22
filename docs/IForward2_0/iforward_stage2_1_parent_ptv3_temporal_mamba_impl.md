# IForward Stage 2_1：Parent PTv3 + Temporal Mamba 短序列优化详细实现方案

基线：`drivestudio_stage6_refactor_context_20260621_v31`  
建议版本名：`stage2_1_fwhr_parent_ptv3_temporal_mamba`  
训练方式：**从 0 开始训练，不加载 Stage 2_0 checkpoint**  
主目标：在保留 FWHR current-frame 清晰度的同时，显著降低多 block 顺序优化造成的历史破坏。

---

# 0. 核心决策

Stage 2_1 主结构：

```text
Fine GS FWHR lifting
    ├─ parent context [M,48]
    └─ child detail [N,8]

parent context + parent params + support/valid
    -> Parent Token Builder [M,64]
    -> Parent PTv3 Spatial Encoder [M,64]
    -> Parent Temporal Mamba read [M,32]
    -> temporal residual fusion
    -> parent_event [M,64]
    -> GRLD
    -> child geometry event [N,16]

child detail [N,8]
    -> attribute-specific detail adapters
    -> means / scales / opacity / SH heads

posterior updater
    -> fine GS delta
```

关键原则：

```text
1. PTv3 负责同一 block 内 parent token 的空间冲突与上下文。
2. Temporal Mamba 只沿真实 block 时间更新，不沿同帧 repeat 时间更新。
3. 同一 block 的 K 次 repeat 可以读取 memory，但只在 block exit commit 一次。
4. Child FWHR detail 是 current-only transient evidence，不进入 temporal memory。
5. 移除恒零 obs code；保留 support 和 valid。
6. 使用直接 history render loss，并加入 history damage hinge。
7. 第一版训练 scheduler 不混合 r8b1/r2b4；短序列主训练固定使用 r4b2。
8. History gate / ADC 在 Stage 2_1 第一版保持关闭，以隔离 PTv3 + Temporal Mamba 的作用。
9. FWHR hierarchical fused CUDA 暂缓，继续使用当前 child raw lift + parent pooling reference 路径。
```

---

# 1. 当前基线与 Stage 2_1 要解决的问题

当前 Stage 2_0 已经解决：

```text
- parent token 数压缩；
- fine Tα 提供 current-frame 高频 evidence；
- parent context + child detail 的 FWHR 信息路径；
- GRLD parent→child relation；
- parent sufficient stats incremental update；
- GroupNorm residual frontend；
- DINOv2-Base frozen context；
- appearance detail 注入。
```

但短序列仍存在：

```text
r8b1 best-to-final forgetting p90 ≈ 4 dB 级；
r4b2 即使有 block 重访，best-to-final forgetting 仍很大；
r2b4 表现较好主要来自持续 rehearsal，而不是历史保护。
```

根因不是 current evidence 不够，而是：

```text
1. parent token 只有 block-local xCPE，没有足够强的空间 conflict modeling；
2. 没有 parent-level temporal state；
3. history replay 在训练中未真正启用；
4. memory update 时钟没有区分 repeat 与 block；
5. current detail 可能驱动 geometry，但没有历史监督约束其长期影响。
```

---

# 2. 总体运行时序

## 2.1 两个时间轴

必须严格区分：

```text
repeat time：
    同一 block / 同一 source observation 下的 RAFT-like optimizer iteration。

block time：
    新 source frame / 新 block 到达，代表真实短序列时间推进。
```

Temporal Mamba 状态只沿 block time 更新。

错误语义：

```text
每 repeat 写一次 Mamba state
```

这会让 K=8 的单帧被当成 8 个时间步，memory 行为会依赖采样 K。

正确语义：

```text
block repeat 0：
    生成 first_observation_token，缓存为该 block 的 commit token。

block repeat 0...K-1：
    读取上一 block 的 memory；不修改 memory state。

block exit：
    用 first_observation_token commit 一次 temporal state。
```

## 2.2 完整 block 流程

```text
block enter
    1. exact refresh parent stats / parent params
    2. 生成或复用 PTv3 serialization
    3. 建立 temporal memory key mapping

repeat 0
    4. FWHR fine lifting
    5. parent token builder
    6. PTv3 spatial encode
    7. 缓存 first_spatial_event 作为 temporal commit token
    8. Temporal Mamba preview/read，state 不写
    9. temporal fusion -> parent event
   10. GRLD -> child event
   11. child detail attribute injection
   12. posterior delta / apply delta
   13. incremental parent stats update

repeat 1...K-1
    重复 4～13，但不替换 first_spatial_event，不写 Mamba state

block exit
   14. 使用 first_spatial_event 对 Mamba state commit 一次
   15. 写入 block history reference bank
   16. 若 rollout final，计算 current + history + damage loss
```

---

# 3. 移除恒零 obs code

## 3.1 当前问题

FWHR 当前 `parent_obs_mode=zero`，但代码仍：

```text
- 分配 parent_obs_code [M,2]；
- Stage6StructInput 携带 obs_code；
- ParamObsCodec 把恒零 obs 拼到输入；
- EventPack 保留 obs_code_bg/distant/rigid；
- 配置 require_obs_code=true。
```

恒零输入没有模型价值，也会使后续设计误以为已经具备可靠的多视角 observation statistics。

## 3.2 Stage 2_1 接口

Stage 2_1 不直接破坏 Stage 1 / Stage 2_0 旧路径。新增 parent 专用结构：

```python
@dataclass
class ParentStructInput:
    feat_2d: torch.Tensor            # [M,48]
    support: torch.Tensor            # [M]
    valid: torch.Tensor              # [M] bool
    coords: torch.Tensor             # [M,3]
    branch_id: torch.Tensor          # [M]
    params_for_embed: Dict[str, Tensor]
    split_0: int
    split_1: int
    meta: Dict[str, Any]

@dataclass
class ParentStructOutput:
    event: torch.Tensor              # [M,64]
    support: torch.Tensor
    valid: torch.Tensor
    aux: Dict[str, Any]
```

新增：

```text
Stage6ParentParamSupportCodec
```

输入：

```text
raw parent params 17D
support feature 2D：
    log1p(support)
    valid
branch embedding
```

不再输入 obs code。

## 3.3 Codec 维度

```text
raw params        17
support             2
branch embed        4
---------------------
input              23
output             24
```

配置：

```yaml
parent_token_codec:
  type: param_support
  support_dim: 2
  branch_embed_dim: 4
  output_dim: 24
  detach_params: true
  detach_support: true
```

## 3.4 EventPack 迁移

Stage 2_1 的 `EventPack` 不填充 obs code：

```python
EventPack(
    event_bg=...,
    event_distant=...,
    event_rigid=...,
    support_bg=...,
    support_distant=...,
    support_rigid=...,
    valid_bg=...,
    valid_distant=...,
    valid_rigid=...,
    obs_code_bg=None,
    obs_code_distant=None,
    obs_code_rigid=None,
)
```

旧字段可暂留为 Optional 兼容旧版本，但 Stage 2_1 路径：

```text
- 不分配 [M,2] zero tensor；
- 不写 measurement；
- 不送入 codec；
- 配置 require_obs_code=false。
```

## 3.5 Support / valid 的语义

```text
support：fine child contribution 聚合到 parent 的 visibility mass。
valid：support > branch-specific threshold。
```

Temporal memory commit 使用：

```python
write_mask = valid & (support >= temporal_support_min)
```

Stage 2_1 不需要旧 obs code；未来若需要多视角可靠性，新增独立 `evidence_stats`，不要复用旧接口。

---

# 4. Parent PTv3 Spatial Encoder

## 4.1 架构选择

PTv3 用于替换当前 near xCPE 主干，而不是堆在其后。

Stage 2_1 第一版路由：

```text
BG parent + rigid_inside parent
    -> shared PTv3 near encoder

distant parent + rigid_out parent
    -> 继续使用 far point MLP
```

理由：

```text
- near parent 数量最多，也最需要空间 conflict modeling；
- distant 的尺度和稀疏性与 near 差异很大；
- 第一版避免把 spatial backbone 与 temporal memory 两个变量同时扩大到所有 branch；
- temporal Mamba 仍覆盖 bg / distant / rigid 三个 branch。
```

后续可增加 `far_ptv3_lite`，不是 P0。

## 4.2 新文件

```text
models/iforward/parent_serialization.py
models/iforward/parent_ptv3.py
models/iforward/parent_spatial_backbone.py
```

## 4.3 Parent token builder

输入：

```text
parent context       48D
parent param/support 24D
support projection
branch embedding
```

输出：

```text
parent_token [M,64]
```

建议保持加法式 token builder，而不是 concat 后大 MLP：

```python
x = feat_proj(parent_context)
x = x + param_support_proj(param_support)
x = x + support_proj([log1p(support), valid])
x = x + branch_embed(branch_id)
x = LayerNorm(x)
```

这样更适合 PTv3 residual blocks。

## 4.4 Serialization

输入：

```text
coords [M,3]
batch_id [M]
```

量化：

```python
grid = floor((coords - aabb_min) / serialization_grid_size)
```

初版 order：

```text
z：Morton/Z order
z-trans：交换 xyz 轴后的 Morton order
```

每个 PTv3 block 交替使用：

```text
block 0 -> z
block 1 -> z-trans
```

同一 block 的 K repeats：

```text
复用 serialization order / inverse / patch indices。
```

每个新真实 block：

```text
根据 exact-refresh 后 parent coords 重新序列化。
```

## 4.5 Patch 构建

```text
patch_size = 64
```

对每个 batch/group：

```text
1. 根据 serialization order 排序；
2. pad 到 patch_size 倍数；
3. reshape [num_patches, patch_size, dim]；
4. 生成 padding mask；
5. attention 后 unpad + inverse reorder。
```

必须禁止不同 batch/scene 跨 patch。

当前一次只处理一个 scene/segment，可将 near path 视为一个 batch；branch embedding区分 bg和rigid。

## 4.6 PTv3 block

```python
class ParentPTv3Block(nn.Module):
    x = x + cpe_scale * CPE(x, coords)
    x = x + attn_scale * SerializedAttention(LN1(x), layout)
    x = x + mlp_scale * MLP(LN2(x))
```

建议：

```text
model_dim       64
num_heads        4
head_dim        16
patch_size      64
depth             2
mlp_ratio       4.0
dropout          0
drop_path        0～0.05
CPE sparse conv  3×3×3
LayerScale init  1e-3
RPE              disabled
```

使用：

```text
torch.nn.functional.scaled_dot_product_attention
```

若 FlashAttention backend 可用则自动使用；否则 fallback math/memory-efficient SDPA。

## 4.7 CPE

可复用当前 xCPE voxel layout：

```text
point token -> voxel scatter mean
3D sparse conv
voxel delta -> gather point
```

但每个 PTv3 block 只做一次 CPE，不再保留当前额外两层 xCPE decoder。

## 4.8 输出

```text
spatial_parent_event [M,64]
```

初始稳定性：

```text
- 从 scratch 训练；
- LayerScale 1e-3；
- token builder 和 event norm 正常初始化；
- 不使用 zero output adapter，避免 PTv3 首阶段无梯度。
```

## 4.9 PTv3 配置

```yaml
parent_spatial_backbone:
  type: ptv3_encoder_only
  scope: near_only
  input_dim: 64
  model_dim: 64
  depth: 2
  num_heads: 4
  patch_size: 64
  mlp_ratio: 4.0
  serialization_orders: [z, z_trans]
  serialization_grid_size: 0.25
  reuse_layout_within_block: true
  cpe:
    enable: true
    sparse_backend: spconv
    kernel_size: 3
    layer_scale_init: 0.001
  attention:
    use_sdpa: true
    enable_rpe: false
  drop_path: 0.02
```

---

# 5. Parent Temporal Mamba

## 5.1 目标

Temporal Mamba 建模：

```text
同一个 parent identity 在连续真实 block 中的 observation history。
```

它不做：

```text
- 同一帧 point serialization；
- 空间 token mixing；
- child detail memory；
- 每 repeat memory update。
```

## 5.2 新文件

```text
models/iforward/parent_temporal_state.py
models/iforward/parent_temporal_mamba.py
models/iforward/parent_temporal_keys.py
```

复用：

```text
StreamingMambaCell
DenseMambaState / KeyedMambaState 路由逻辑
```

但增加显式接口：

```python
preview(x, state, keys) -> context        # state 不写
commit(x, state, keys, write_mask) -> new_state
```

不要继续依赖 `write_mask=false` 的隐式语义，降低实现错误风险。

## 5.3 State 数据结构

```python
@dataclass
class ParentTemporalBranchState:
    dense: Optional[DenseMambaState] = None
    keyed: Optional[KeyedMambaState] = None

@dataclass
class ParentTemporalState:
    bg: ParentTemporalBranchState
    distant: ParentTemporalBranchState
    rigid: ParentTemporalBranchState
    last_committed_block_id: int = -1

    def detach(self, clone: bool = False): ...
```

加入：

```python
IForwardState.parent_temporal: Optional[ParentTemporalState]
```

生命周期：

```text
episode begin：reset
rollout 内：保持可导
rollout exit：detach（truncated BPTT）
episode end：discard
```

## 5.4 Parent keys

### BG

固定 assignment local parent id：

```text
key = bg_parent_id
```

使用 dense memory。

### Distant

```text
key = distant_parent_id
```

使用 dense memory。

### Rigid

必须使用稳定 global identity：

```text
key = hash(instance_id, global_parent_id)
```

不能使用：

```text
active_parent_row
```

因为 active row 会随 block、near/out split变化。

如果一个 global rigid parent 在同一 block 被拆成 near/out 两个 active row：

```text
preview：两行读取同一 key context；
commit：先按 key 做 weighted mean，再写一次。
```

## 5.5 Temporal input token

不使用 obs code。

```text
spatial parent event 64D
log1p(support)          1D
valid                   1D
branch embedding        4D
seen flag               1D（只用于fusion，不必输入cell）
```

先投影：

```python
temporal_input = Linear(70, 64)(concat(...))
```

Mamba：

```text
input_dim    64
model_dim    32
state_dim     8
conv_kernel   2
output_dim   32
```

## 5.6 Preview / read

每 repeat：

```python
ctx32, seen = temporal_mamba.preview(
    x=current_spatial_event,
    state=previous_block_state,
    keys=parent_keys,
)
```

对于 unseen parent：

```python
ctx32 = 0
```

不要让 zero-state Mamba 的 current-token transform冒充历史 context。

## 5.7 Commit

在 block repeat 0 缓存：

```python
first_repeat_commit_token = temporal_input
```

在 block exit：

```python
write_mask = valid & (support >= support_min_commit)
new_state = temporal_mamba.commit(
    x=first_repeat_commit_token,
    state=old_state,
    keys=parent_keys,
    write_mask=write_mask,
)
```

为何使用 first repeat：

```text
- commit语义与真实observation对应；
- 不依赖K；
- 避免把optimizer后验状态误当成新测量；
- K=4和K=8得到一致memory clock。
```

## 5.8 Temporal fusion

```python
temporal_delta = TemporalAdapter(ctx32)  # 32 -> 64
fusion_gate = sigmoid(gate_raw[branch])
parent_event = LayerNorm(spatial_event + fusion_gate * temporal_delta)
```

初始 gate：

```text
BG       0.05
Distant  0.05
Rigid    0.03
```

`TemporalAdapter` 最后一层使用小非零初始化 `std=1e-3`，不能 exact zero，否则 Mamba 初始无梯度。

## 5.9 配置

```yaml
parent_temporal_memory:
  enable: true
  type: streaming_mamba
  input_dim: 64
  model_dim: 32
  state_dim: 8
  conv_kernel: 2
  output_dim: 32

  branches:
    bg:
      enable: true
      storage: dense
    distant:
      enable: true
      storage: dense
    rigid:
      enable: true
      storage: keyed

  read_policy: every_repeat
  commit_policy: block_exit
  commit_token: first_repeat_spatial_event
  support_min_commit: 0.001
  hard_valid_required: true
  unseen_context: zero
  detach_after_rollout: true

  fusion:
    output_dim: 64
    adapter_init_std: 0.001
    gate_init:
      bg: 0.05
      distant: 0.05
      rigid: 0.03
    gate_max: 1.0
```

---

# 6. GRLD 与 FWHR Detail 的职责

## 6.1 GRLD

继续负责：

```text
parent spatial/temporal event
+ child current Gaussian relation
-> child coarse/geometry event [N,16]
```

Relation inputs仍 detach，不打开 geometry Jacobian。

## 6.2 Current child detail

FWHR detail仍为：

```text
current-only
8D
parent内weighted zero-mean
不写temporal memory
```

Stage 2_1 允许其影响：

```text
means
scales
opacity
SH
```

quat 第一版保持关闭或极低权重。

---

# 7. Attribute-specific Detail Injection

当前 updater 只让 detail 进入 opacity/SH。Stage 2_1 改为属性独立 gate。

## 7.1 Updater 结构

```python
h_geo = trunk(event)
d = detail_adapter(child_detail)

h_means  = h_geo + gate_means[branch]  * d
h_scales = h_geo + gate_scales[branch] * d
h_quat   = h_geo + gate_quat[branch]   * d
h_opacity= h_geo + gate_opacity[branch]* d
h_sh     = h_geo + gate_sh[branch]     * d
```

Head：

```text
means       <- h_means
scales      <- h_scales
quat        <- h_quat
opacity     <- h_opacity
SH          <- h_sh
noop        <- h_geo
confidence  <- h_geo
```

所有属性仍共享 noop gate：

```text
delta_attr *= (1 - noop)
```

## 7.2 初始 gate

建议：

```yaml
child_detail_injection:
  bg:
    means: 0.02
    scales: 0.01
    quat: 0.00
    opacity: 0.10
    sh: 0.10
  distant:
    means: 0.00
    scales: 0.00
    quat: 0.00
    opacity: 0.10
    sh: 0.10
  rigid:
    means: 0.03
    scales: 0.015
    quat: 0.005
    opacity: 0.08
    sh: 0.08
```

Gate max：

```text
means      0.15
scales     0.08
quat       0.03
opacity    1.00
SH         1.00
```

这些是 hidden injection gate，不是最终 delta 幅度；最终仍受现有 per-attribute clamp约束。

## 7.3 为什么 detail 不进入 PTv3/Mamba

```text
- detail代表当前图像高频；
- memory应保存parent级稳定上下文，而不是当前appearance噪声；
- 让detail进入temporal state会增加过拟合当前帧与颜色漂移风险；
- geometry detail通过受限gate直接作用于当前delta即可。
```

---

# 8. History Supervision

Stage 2_1 不依赖 history gate，先通过直接监督训练 PTv3 + Temporal Mamba。

## 8.1 History refs

训练 shape 固定为 `r4b2` 后：

```text
rollout 0: blocks [0,1]，无history
rollout 1: blocks [1,2]，history候选 [0]
rollout 2: blocks [2,3]，history候选 [0,1]
rollout 3: blocks [3,0]，history候选 [1,2]
```

采样：

```yaml
history_replay:
  enable: true
  sampling_policy: previous_visited_blocks
  exclude_current_blocks: true
  max_frames_per_rollout: 2
  camera_policy: all_cams
  max_refs_per_rollout: 6
```

## 8.2 Absolute history render loss

```math
L_hist = \frac{1}{|H|}\sum_{h\in H}
  [L1(R(G_{final},h),I_h)+\lambda_{ssim}L_{ssim}]
```

每个 role 独立按有效 refs 归一化，不允许 current refs 数量掩盖 history 权重。

## 8.3 History damage hinge

只用绝对 history loss 可能鼓励模型不断重优化旧帧，但不能直接区分“本 rollout 造成的破坏”。新增 damage loss。

在 rollout 开始、尚未处理当前 blocks 前，对同一组 history refs 计算 no-grad per-ref loss：

```text
l_before[h]
```

rollout final：

```text
l_after[h]
```

定义：

```math
L_damage = \frac{1}{|H|}\sum_h
  \max(0, l_{after,h}-stopgrad(l_{before,h})-m)
```

推荐 margin：

```text
m = 0.002
```

作用：

```text
- 允许小幅数值波动；
- 只惩罚当前rollout新增的damage；
- 历史帧变好时不处罚；
- absolute history loss防止no-op成为最优解。
```

## 8.4 Loss 总式

```math
L = L_current
  + \lambda_h L_hist
  + \lambda_d L_damage
  + \lambda_n L_nearby
  + L_delta_reg.
```

第一版：

```text
current        1.0
history abs    warmup 0 -> 0.5
history damage warmup 0 -> 0.25
nearby         0（P0）
delta reg      延用Stage2_0
```

## 8.5 Weight schedule

```yaml
loss:
  current:
    weight: 1.0

  in_rollout_history:
    weight: 0.5
    warmup:
      enable: true
      start_step: 5000
      steps: 15000
      start_factor: 0.0

  history_damage:
    enable: true
    weight: 0.25
    margin: 0.002
    warmup:
      enable: true
      start_step: 10000
      steps: 15000
      start_factor: 0.0

  nearby:
    weight: 0.0
```

## 8.6 实现改动

新增：

```python
@dataclass
class HistoryDamageProbe:
    target_indices: List[int]
    loss_before_per_ref: torch.Tensor
    valid_mask: torch.Tensor
```

在 `IForwardModel.forward()` rollout loop 前：

```python
history_probe = bridge.render_per_ref_loss_no_grad(
    local_state=local_state,
    target_indices=resolved.history_rollout_target_indices,
)
```

在 final render：

```python
history_after = bridge.render_per_ref_loss(...)
damage_loss = relu(history_after - history_probe.before - margin).mean()
```

`render_loss` 需要新增：

```text
return_per_ref_loss=true
```

避免只返回全局平均导致不同 history ref 之间相互抵消。

---

# 9. Scheduler：不做 r8b1/r2b4 episode-level 混合

## 9.1 训练 curriculum

从 0 训练，使用单一 shape phase，不进行 episode-level shape mixture。

### Phase 0：current bootstrap

```text
step 0～5k
shape：b1_r4
history refs可以生成，但loss weight=0
Temporal Mamba启用，但episode首block多数parent为unseen
```

目标：

```text
FWHR + PTv3 + GRLD current能力先稳定。
```

### Phase 1：短序列 warmup

```text
step 5k～20k
shape：r4b2
history weight从0增加到0.5
Temporal Mamba按block commit
```

### Phase 2：正式短序列

```text
step 20k+
shape：r4b2固定
history abs=0.5
damage=0.25
```

不混合：

```text
r8b1
r2b4
b3r3
```

它们仅作为 validation stress shapes。

## 9.2 Episode 设置

```yaml
scheduler_iforward:
  version: iforward_stage2_1_parent_temporal

  traversal:
    traversal_mode: scene_round_robin_episode
    forbid_consecutive_same_scene: true
    scene_order: shuffle_per_epoch
    segment_order: shuffle_per_epoch

  episode:
    blocks_per_episode: 4
    episode_stride: 4
    rollouts_per_episode: 4
    block_source_frame_policy: random_within_keyframe_once_per_episode
    reset_scene_state_policy: episode_begin

  rollout:
    shape_sample_scope: episode
    block_selection_policy: ordered_cyclic_start
    start_offset_policy: random_cyclic_offset
    delivery_order_policy: rollout_order
    tail_policy: circular_fill
    detach_graph_after_rollout: true
    max_inner_K: 8
```

`random_within_keyframe_once_per_episode` 很重要：同一个 block 重访时使用稳定 source frame，避免 temporal memory 同时承受“block identity相同但随机frame变化”的噪声。

## 9.3 Shape schedule

```yaml
shapes_schedule:
  - start_step: 0
    shapes:
      - name: b1_r4
        blocks_per_rollout: 1
        repeats_per_block: 4
        prob: 1.0

  - start_step: 5000
    shapes:
      - name: r4b2
        blocks_per_rollout: 2
        repeats_per_block: 4
        prob: 1.0
```

## 9.4 Memory 时钟

```yaml
memory:
  observation_commit_policy: first_repeat_only
  optimizer_memory_update_policy: block_exit
  reset_policy: episode_begin
  carry_policy: across_rollouts_until_episode_end
```

需要修改 scheduler/resolver，使 `update_optimizer_memory=true` 仅在 block exit，而不是每 repeat。

---

# 10. Validation 设计

## 10.1 Current quality validation

独立 reset：

```text
b1_r8
```

每 500 step，固定 10 eval scenes。

指标：

```text
PSNR
SSIM value / SSIM loss分开记录
LPIPS
edge L1
Laplacian L1
high-frequency energy ratio
```

## 10.2 Coverage / forgetting validation

独立 shape、独立 state reset：

```text
r8b1：最强sequential forgetting stress
r4b2：训练分布内主shape
r2b4：rehearsal upper reference
```

虽然训练不混合 r8b1/r2b4，但 validation 必须保留它们。

主指标：

```text
r8b1 best-to-final forget p90
r8b1 final all-block PSNR
r4b2 best-to-final forget p90
r4b2 history replay PSNR
r2b4只作为rehearsal参考，不作为主要结论
```

## 10.3 Long-gap retention test

固定流程：

```text
observe block0
observe block1
observe block2
observe block3
final eval block0
```

期间禁止再次看到 block0。

输出：

```text
block0 best PSNR
block0 final PSNR
drop
time_since_seen
```

这是对 temporal memory 最直接的测试。

## 10.4 Validation 频率

```yaml
current_validation:
  interval_steps: 500
  segments: 10
  shape: b1_r8

coverage_quick:
  interval_steps: 2000
  max_segments_total: 1

coverage_full:
  interval_steps: 10000
  max_segments_total: 8
```

## 10.5 Memory ablations

每次 full coverage 至少支持：

```text
A. full PTv3 + Temporal Mamba
B. temporal_read_off
C. temporal_write_off
D. temporal_state_zero
E. temporal_key_shuffle
F. PTv3 bypass
```

若 full 模型优于 B/C，才说明 temporal memory 真正有效。

---

# 11. 从 0 训练的初始化与优化器

## 11.1 初始化

```text
DINOv2：冻结预训练权重
Residual UNet：随机初始化
FWHR detail head：small nonzero init
Parent token builder：Xavier
PTv3：标准初始化 + LayerScale 1e-3
Temporal Mamba：标准Mamba初始化
Temporal adapter：std=1e-3
Temporal fusion gate：0.03～0.05
GRLD：保留small-nonzero detail初始化
Posterior updater：随机初始化
```

不加载旧 checkpoint。

## 11.2 学习率

```yaml
optimizer:
  type: adamw
  lr:
    default: 1.0e-4
    measurement_frontend_residual_unet: 5.0e-5
    fwhr_detail_head: 1.0e-4
    parent_token_builder: 1.0e-4
    parent_ptv3: 1.0e-4
    parent_temporal_mamba: 1.0e-4
    parent_temporal_adapter: 1.0e-4
    grld: 1.0e-4
    posterior_updater: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 1.0e-4
```

全局：

```text
warmup 1000 steps
cosine decay到初始LR的10%
grad clip 1.0
```

## 11.3 AMP

PTv3 attention、DINO feature和大部分 MLP 可使用 bf16；parent stats和GS state仍保持 fp32。

推荐：

```yaml
training:
  amp:
    enable: true
    dtype: bfloat16
```

如当前硬件/gsplat路径对 bf16 不稳定，先保持 fp32，不把 AMP 与架构正确性同时引入。

---

# 12. IForwardState 与 Runtime 扩展

```python
@dataclass
class IForwardState:
    local_gs: LocalGSState
    history: IForwardHistory
    biggs_state: Optional[IForwardBigGSState]
    parent_temporal: Optional[ParentTemporalState]
    ...
```

Block-local runtime：

```python
@dataclass
class ParentBlockRuntime:
    parent_stats: BigGSBlockRuntime
    ptv3_layout_near: Optional[ParentSerializedLayout]
    parent_keys_bg: Tensor
    parent_keys_distant: Tensor
    parent_keys_rigid: Tensor
    first_repeat_commit_bg: Optional[Tensor]
    first_repeat_commit_distant: Optional[Tensor]
    first_repeat_commit_rigid: Optional[Tensor]
```

该 runtime：

```text
- block enter初始化；
- K repeats复用；
- block exit后释放；
- 不写checkpoint。
```

Temporal state：

```text
- episode state；
- 跨rollout携带；
- rollout end detach；
- episode end清空。
```

---

# 13. 模型 forward 伪代码

```python
state = carried_state or init_state()
parent_temporal = state.parent_temporal or ParentTemporalState.empty()

history_probe = build_history_damage_probe_if_needed(state, resolved)
block_runtime = None

for step in resolved.steps:
    if step.is_block_enter:
        block_runtime = init_parent_block_runtime(
            local_state=local_state,
            biggs_assignment=state.biggs_state,
            temporal_state=parent_temporal,
        )

    measurement = observe_fwhr(...)

    parent_input = build_parent_struct_input_without_obs_code(measurement)
    parent_tokens = parent_token_builder(parent_input)

    spatial_event, ptv3_aux = parent_spatial_backbone(
        parent_tokens,
        coords=parent_input.coords,
        layout=block_runtime.ptv3_layout_near,
    )

    temporal_ctx, temporal_seen = parent_temporal_mamba.preview(
        spatial_event=spatial_event,
        state=parent_temporal,
        keys=block_runtime.parent_keys,
    )

    parent_event = fuse_spatial_temporal(
        spatial_event,
        temporal_ctx,
        temporal_seen,
    )

    if step.repeat_idx == 0:
        block_runtime.cache_commit_token(spatial_event, support, valid)

    child_event = grld(parent_event, child_relation)
    delta = posterior_updater(
        child_event,
        child_detail=measurement.child_detail,
        detail_valid=measurement.child_detail_valid,
    )

    old_state = local_state
    local_state = apply_delta(local_state, delta)

    if not step.is_block_exit:
        block_runtime.parent_stats = incremental_parent_update(old_state, local_state)

    if step.is_block_exit:
        parent_temporal = parent_temporal_mamba.commit(
            token=block_runtime.first_repeat_commit,
            state=parent_temporal,
            keys=block_runtime.parent_keys,
            write_mask=block_runtime.commit_valid,
        )
        block_runtime = None

loss_current = render_current(local_state)
loss_history = render_history(local_state)
loss_damage = history_damage(history_probe, loss_history_per_ref)
loss = combine_losses(...)

next_state.parent_temporal = detach_if_rollout_exit(parent_temporal)
```

---

# 14. 文件级改动清单

## 新增

```text
models/iforward/parent_serialization.py
models/iforward/parent_ptv3.py
models/iforward/parent_spatial_backbone.py
models/iforward/parent_temporal_state.py
models/iforward/parent_temporal_keys.py
models/iforward/parent_temporal_mamba.py
models/iforward/history_damage_loss.py
```

## 修改

```text
models/streetforward/stage6_0/struct_event_decoder.py
    + ParentStructInput / ParentStructOutput
    + Stage6ParentParamSupportCodec
    + PTv3 parent route
    - Stage2_1 obs-code dependency

models/streetforward/stage6_0/event_encoder.py
    obs_code fields Stage2_1不再使用

models/streetforward/stage6_0/posterior_updater.py
    + attribute-specific detail gates
    + means/scales/quat detail paths

models/iforward/fwhr_lift.py
    - parent_obs_code zero allocation

models/iforward/state.py
    + parent_temporal state

models/iforward/model.py
    + PTv3 block runtime
    + temporal preview/commit clock
    + history probe/damage loss

models/iforward/bridge.py
    + parent temporal APIs
    + per-ref render loss

datasets/train_scheduler_iforward.py
    + block_exit memory update semantics
    + fixed phase shape schedule
    + stable source frame per episode

datasets/iforward_coverage_validation.py
    + long-gap test
    + memory ablations
```

---

# 15. 测试计划

## 15.1 obs code 移除

```text
test_stage2_1_parent_token_has_no_obs_code
test_stage2_1_does_not_allocate_parent_obs_tensor
test_parent_support_codec_shape
test_legacy_stage2_0_obs_path_still_works
```

## 15.2 PTv3

```text
test_parent_serialization_roundtrip
test_serialized_attention_no_cross_batch
test_ptv3_reuses_layout_within_block
test_ptv3_rebuilds_layout_on_new_block
test_ptv3_empty_rigid_and_empty_distant
test_ptv3_forward_backward_finite
```

## 15.3 Temporal Mamba

```text
test_temporal_preview_does_not_modify_state
test_temporal_commit_updates_once_per_block
test_temporal_state_independent_of_repeat_count
test_unseen_parent_context_is_zero
test_rigid_global_key_stable_across_active_rows
test_duplicate_rigid_keys_commit_once
test_temporal_state_detaches_at_rollout_end
```

最重要的 clock test：

```text
相同block observation：
K=4与K=8最终commit后的parent temporal state必须近似一致。
```

## 15.4 History loss

```text
test_history_loss_zero_without_refs
test_history_damage_zero_when_history_improves
test_history_damage_positive_when_history_degrades
test_history_refs_exclude_current_blocks
test_history_per_ref_normalization
test_history_probe_no_grad
```

## 15.5 Attribute detail

```text
test_child_detail_changes_means_when_enabled
test_child_detail_does_not_change_quat_when_gate_zero
test_distant_detail_only_updates_enabled_attrs
test_detail_does_not_enter_temporal_commit_token
```

## 15.6 Scheduler

```text
test_stage2_1_phase0_is_b1_r4_only
test_stage2_1_phase1_is_r4b2_only
test_r4b2_ordered_cyclic_sequence
test_source_frame_stable_within_episode
test_memory_commit_only_at_block_exit
```

---

# 16. 日志设计

## PTv3

```text
iforward/ptv3/num_tokens
iforward/ptv3/num_patches
iforward/ptv3/padding_ratio
iforward/ptv3/serialization_ms
iforward/ptv3/cpe_ms
iforward/ptv3/attention_ms
iforward/ptv3/output_norm
iforward/ptv3/peak_alloc_mb
```

## Temporal Mamba

```text
iforward/parent_mamba/seen_ratio_bg
iforward/parent_mamba/seen_ratio_distant
iforward/parent_mamba/seen_ratio_rigid
iforward/parent_mamba/preview_rows
iforward/parent_mamba/commit_rows
iforward/parent_mamba/commit_count_this_block
iforward/parent_mamba/context_norm
iforward/parent_mamba/fusion_gate_bg
iforward/parent_mamba/fusion_gate_distant
iforward/parent_mamba/fusion_gate_rigid
iforward/parent_mamba/state_memory_mb
iforward/parent_mamba/preview_ms
iforward/parent_mamba/commit_ms
```

Hard assert：

```text
commit_count_this_block <= 1
```

## History

```text
iforward/history/num_refs
iforward/history/loss_abs
iforward/history/loss_damage
iforward/history/loss_before
iforward/history/loss_after
iforward/history/damaged_ref_ratio
iforward/history/psnr
iforward/history/weight_abs
iforward/history/weight_damage
```

## Attribute detail

```text
posterior/detail_gate_means_bg
posterior/detail_gate_scales_bg
posterior/detail_gate_quat_bg
posterior/detail_gate_opacity_bg
posterior/detail_gate_sh_bg
```

---

# 17. Validation 验收标准

## Current

相较 Stage 2_0 FWHR current baseline：

```text
K8 PSNR下降不超过0.2dB；
SSIM / edge metric不明显下降；
高频视觉不退回parent-only模糊。
```

## History

在 full coverage 8+ segments：

```text
r8b1 best-to-final forget p90：显著低于Stage2_0约4dB水平；
第一阶段目标：< 2.5dB；
后续目标：< 1.5dB。

r4b2 best-to-final forget p90：
第一阶段目标：< 2.0dB。

long-gap block0 drop：
目标低于无memory baseline至少30%。
```

## Memory causal evidence

必须满足：

```text
full > temporal_read_off
full > temporal_write_off
full > temporal_key_shuffle
```

否则不能声称 Temporal Mamba 有效。

---

# 18. 完整配置草案

```yaml
model:
  iforward:
    version: stage2_1_fwhr_parent_ptv3_temporal_mamba

    biggs:
      # FWHR、assignment、parent projector、incremental stats延用v31
      lifting:
        type: fwhr
        context_dim: 48
        detail_dim: 8
        geometry_grad: false
        parent_obs_mode: none

      child_decoder:
        mode: gaussian_relational
        rank: 4
        parent_event_dim: 64
        fine_event_dim: 16
        relation_normalization: sibling_rms
        relation_rms_floor: 0.05
        relation_clip: 5.0
        rigid_relation_space: canonical

    parent_spatial_backbone:
      type: ptv3_encoder_only
      scope: near_only
      model_dim: 64
      depth: 2
      num_heads: 4
      patch_size: 64
      mlp_ratio: 4.0
      serialization_orders: [z, z_trans]
      serialization_grid_size: 0.25
      reuse_layout_within_block: true
      cpe:
        enable: true
        sparse_backend: spconv
        kernel_size: 3
        layer_scale_init: 0.001
      attention:
        use_sdpa: true
        enable_rpe: false
      drop_path: 0.02

    parent_temporal_memory:
      enable: true
      type: streaming_mamba
      input_dim: 64
      model_dim: 32
      state_dim: 8
      conv_kernel: 2
      output_dim: 32
      read_policy: every_repeat
      commit_policy: block_exit
      commit_token: first_repeat_spatial_event
      support_min_commit: 0.001
      hard_valid_required: true
      unseen_context: zero
      detach_after_rollout: true
      branches:
        bg: {enable: true, storage: dense}
        distant: {enable: true, storage: dense}
        rigid: {enable: true, storage: keyed}
      fusion:
        adapter_init_std: 0.001
        gate_init: {bg: 0.05, distant: 0.05, rigid: 0.03}
        gate_max: 1.0

    loss:
      current: {weight: 1.0}
      in_rollout_history:
        weight: 0.5
        warmup:
          enable: true
          start_step: 5000
          steps: 15000
          start_factor: 0.0
      history_damage:
        enable: true
        weight: 0.25
        margin: 0.002
        warmup:
          enable: true
          start_step: 10000
          steps: 15000
          start_factor: 0.0
      nearby: {weight: 0.0}
      delta_regularization: {weight: 1.0}

  stage6_0:
    base_measurement:
      require_obs_code: false
      obs_code_dim: 0
      source_evidence_grad_mode: train_2d_detach_alpha

    parent_struct_event_decoder:
      enable: true
      feat_2d_dim: 48
      event_dim: 64
      param_support_codec:
        support_dim: 2
        branch_embed_dim: 4
        output_dim: 24
        detach_params: true
        detach_support: true

    posterior_updater:
      event_dim: 16
      hidden_dim: 32
      appearance_detail:
        enable: true
        detail_dim: 8
        attribute_gates:
          bg: {means: 0.02, scales: 0.01, quat: 0.0, opacity: 0.10, sh: 0.10}
          distant: {means: 0.0, scales: 0.0, quat: 0.0, opacity: 0.10, sh: 0.10}
          rigid: {means: 0.03, scales: 0.015, quat: 0.005, opacity: 0.08, sh: 0.08}

scheduler_iforward:
  version: iforward_stage2_1_parent_temporal
  traversal:
    traversal_mode: scene_round_robin_episode
    forbid_consecutive_same_scene: true
  episode:
    blocks_per_episode: 4
    episode_stride: 4
    rollouts_per_episode: 4
    block_source_frame_policy: random_within_keyframe_once_per_episode
    reset_scene_state_policy: episode_begin
  rollout:
    shape_sample_scope: episode
    block_selection_policy: ordered_cyclic_start
    start_offset_policy: random_cyclic_offset
    delivery_order_policy: rollout_order
    tail_policy: circular_fill
    detach_graph_after_rollout: true
    max_inner_K: 8
    shapes_schedule:
      - start_step: 0
        shapes:
          - {name: b1_r4, blocks_per_rollout: 1, repeats_per_block: 4, prob: 1.0}
      - start_step: 5000
        shapes:
          - {name: r4b2, blocks_per_rollout: 2, repeats_per_block: 4, prob: 1.0}
  memory:
    observation_commit_policy: first_repeat_only
    optimizer_memory_update_policy: block_exit
    reset_policy: episode_begin
    carry_policy: across_rollouts_until_episode_end
  supervision:
    current:
      enable: true
      frame_policy: all_rollout_input_frames
      camera_policy: all_cams
    history_replay:
      enable: true
      sampling_policy: previous_visited_blocks
      exclude_current_blocks: true
      max_frames_per_rollout: 2
      camera_policy: all_cams
      max_refs_per_rollout: 6
    nearby:
      enable: false

optimizer:
  type: adamw
  lr:
    default: 1.0e-4
    stage6_measurement_frontend_residual_unet: 5.0e-5
    fwhr_detail_head: 1.0e-4
    parent_token_builder: 1.0e-4
    parent_ptv3: 1.0e-4
    parent_temporal_mamba: 1.0e-4
    parent_temporal_adapter: 1.0e-4
    biggs_child_decoder: 1.0e-4
    stage6_posterior_updater_base: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 1.0e-4

training:
  resume_checkpoint: ''
  start_step: 0
  grad_clip: {enable: true, max_norm: 1.0}
```

---

# 19. 实施阶段

## Phase 1：接口与 current baseline

```text
- 移除Stage2_1 obs code；
- attribute-specific child detail gates；
- Parent Token Builder；
- PTv3 near encoder；
- 暂不启用history loss；
- b1_r4 from scratch smoke training。
```

验收：current K8质量与Stage2_0接近。

## Phase 2：Temporal Mamba clock

```text
- parent keys；
- preview/commit API；
- block-exit single commit；
- state carry/detach；
- no history loss先验证memory clock。
```

验收：K4/K8同block commit state近似一致。

## Phase 3：r4b2 + history abs

```text
- fixed r4b2；
- history replay refs；
- absolute history loss warmup。
```

## Phase 4：history damage

```text
- per-ref before/after probe；
- damage hinge；
- coverage/full validation。
```

## Phase 5：性能优化

只有在 PTv3 + r4b2 显存接近上限时，再实现 FWHR hierarchical fused CUDA。

---

# 20. 参考工作

```text
Point Transformer V3: Simpler, Faster, Stronger — arXiv:2312.10035
Mamba: Linear-Time Sequence Modeling with Selective State Spaces — arXiv:2312.00752
Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model — arXiv:2404.14966
MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models — arXiv:2405.14338
```

Stage 2_1 借鉴的是：

```text
PTv3 的 serialized spatial attention；
Mamba 的 selective recurrent state；
Mamba4D 的空间/时间解耦原则。
```

不是把 Mamba3D 与 PTv3 两个空间 backbone 完整串联。

---

# 21. 最终判断

Stage 2_1 的核心不是“模型更大”，而是三件事同时成立：

```text
PTv3：让当前parent之间交换空间上下文；
Temporal Mamba：让同一parent跨真实block保留状态；
History render + damage loss：明确告诉模型后续更新不能破坏过去。
```

只加入 PTv3 / Mamba、不加入正确时钟和 history loss，无法解决短序列遗忘；只加入 history loss、没有 parent temporal identity，也只能依赖反复 rehearsal。
