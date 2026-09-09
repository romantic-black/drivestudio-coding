# IForward Stage 3.4 v57：Functional ParentGS Lifting 实施方案

**版本日期：** 2026-07-16  
**实现身份：** `stage3_4_functional_parentgs_lift`  
**Parent codec schema：** `legacy17d_plus_geometry8d_residual_v1`  
**初始化基线：** IForward Stage 3.3 Observation Feedback，weights-only  
**执行手册：** [IForward Stage 3.4 Acceptance Runbook](./IForward_Stage3_4_Acceptance_Runbook_20260716.md)

> 本文是 v57 的唯一有效实现合同。v56 曾提出“用随机初始化的 13D
> geometry-only codec 替换旧 17D codec，并在迁移时跳过旧 codec”的方案；该方案已
> **废弃**，只能作为历史问题说明，不能再用于配置、checkpoint 迁移、测试或训练。

---

## 0. 决策摘要

Stage 3.4 采用 Functional ParentGS：LocalGS 是唯一真实持久的几何状态，ParentGS 在
每个 visit 从当前 LocalGS 和 graph-free assignment 临时重算。

v57 固定以下合同：

1. bg、distant、rigid-active 三个存在的 Parent branch 均可通过 geometry token 把梯度
   传回 live LocalGS；缺失的 optional branch 保持 `None`。
2. Functional Parent 使用独立 alpha 调度：

   ```text
   0.00 @ 0
   0.10 @ 1000
   0.25 @ 3000
   0.50 @ 8000
   1.00 @ 15000
   ```

3. 只有已有真实 updater ancestor、当前 feedback mode 允许 input gradient、当前处于
   grad-enabled 训练且 alpha 大于零时，Functional Parent projector 才 attached。
4. validation、demo、replay、`torch.no_grad()` 和 `validation_render_only` 始终逐 visit
   exact forward 重算 Parent，但使用 forward-only projector，不建立 LocalGS geometry 图。
5. Parent lifting、PTV3 coords/layout、assignment、support 和 relation geometry 均显式
   detach；Parent memory 继续遵循原 GDKV 生命周期。
6. Parent token 使用 detached Stage 3.3 legacy 17D 表示作为稳定基线，再叠加
   zero-init 8D geometry residual；不读取 quaternion 或 SH 建立新几何梯度。
7. Stage 3.3 → v57 只允许 weights-only 初始化；原生 v57 checkpoint 才能 strict resume。
   缺少 v57 schema 的旧 Stage 3.4 13D checkpoint 明确拒绝。
8. P0 交付包含自动测试、真实 Validation v4 quick/full 和 K=15 profile。1000-step
   matched B/C 只定义合同和延期命令模板，不在本轮实现或执行。

---

## 1. 状态与数据流

### 1.1 状态分层

Canonical LocalGS：

\[
G_t=\{G_t^{bg},G_t^{distant},G_t^{rigid}\}
\]

它是唯一被 updater delta 修改、参与最终渲染并在 rollout 内保留计算图的几何状态；
rollout boundary 显式 detach。

固定 graph-free assignment \(A\) 后，Functional ParentGS 为：

\[
P_t=\widetilde\Pi(G_t;A)
\]

ParentGS：

- 每 visit 重算；
- 不持久保存，不接受独立 delta；
- 不做 incremental update、drift refresh 或 surrogate VJP；
- forward 与 backward 使用同一个 exact diagonal projector；
- Parent memory 只用稳定 Parent ID 索引，不存储 Parent geometry。

### 1.2 单 visit 数据流

```text
live LocalGS G_t
  ├─ source-feedback live view ─ render ─ frontend feature ─┐
  │                                                        │
  └─ Functional-Parent live view                           │
       └─ exact per-visit projector ─ live Parent params   │
            ├─ detached legacy 17D codec ──────────────────┤
            ├─ alpha-gated live 8D residual ───────────────┤
            ├─ detach geometry ─ Parent 2D lifting ─ context
            ├─ detach means ─ PTV3 routing/layout          │
            └─ detach stats/params ─ relation              │
                                                           ▼
                                                PTV3 + GDKV + updater
                                                           │
                                                           ▼
                                                G_{t+1}=Apply(G_t, Δ_t)
```

source feedback 和 Functional Parent feedback 必须拥有独立的 live/detached view 与 gate。
禁止用一个全局 `grad_to_local_state` 布尔值把两条反馈路径重新耦合。

---

## 2. 梯度合同

| 路径 | v57 合同 |
|---|---|
| `LocalGS → source render → frontend → updater` | 按 source feedback 独立调度可微 |
| `LocalGS → exact Parent projector → 8D residual → updater` | bg/distant/rigid-active 均按 Functional Parent 独立调度可微 |
| `LocalGS → Parent lifting geometry` | 显式 detach，禁止梯度 |
| `features_2d → Parent lifting → updater` | 可微，frontend 可训练 |
| `Parent means → PTV3 coords/serialization` | 显式 detach |
| `assignment/order/layout cache → Parent` | graph-free |
| `support/valid → geometry residual` | 显式 detach |
| `relation child/Parent geometry → decoder` | 显式 detach |
| Parent memory within rollout | 保持现有 GDKV 合同 |
| LocalGS across rollout boundary | 显式 detach |

Parent geometry 对 updater 的唯一新梯度入口为：

\[
\frac{\partial \Delta_t}{\partial G_t}
\supset
\frac{\partial \Delta_t}{\partial E_{res}}
\frac{\partial E_{res}}{\partial P_t}
\frac{\partial P_t}{\partial G_t}
\]

quaternion 和 SH 不属于 v57 residual 的输入。Parent lifting 或 relation 不能提供第二条
geometry backward。

### 2.1 统一 Functional Parent gate

配置使用严格解析的独立 policy：

```yaml
model:
  iforward:
    observation_feedback:
      functional_parent:
        enable: true
        branches: [bg, distant, rigid_active]
        start_after_model_updates: 1
        alpha_schedule:
          - [0, 0.0]
          - [1000, 0.10]
          - [3000, 0.25]
          - [8000, 0.50]
          - [15000, 1.0]
```

每个 visit 只计算一次统一 gate，projector attachment、live codec input、断言和指标均使用
同一结果：

```python
functional_parent_grad_active = (
    stage34
    and policy.functional_parent.enable
    and has_update_ancestor
    and model_update_count >= policy.functional_parent.start_after_model_updates
    and feedback_mode.input_grad_enabled
    and torch.is_grad_enabled()
    and not validation_render_only
    and functional_parent_alpha > 0.0
)
```

branch gate 为：

```python
attached_by_branch[name] = (
    functional_parent_grad_active
    and name in policy.functional_parent.branches
    and has_live_geometry_graph(parent_params_by_branch[name])
)
```

三分支的“可微”是配置能力，不代表每个 visit 必须 attached：first visit、alpha=0、
`frozen_no_grad`、validation、缺失 branch，或 rollout boundary 后尚未被当前 rollout 的
updater 重新连接到计算图的 branch 均是 forward-only。`has_live_geometry_graph()` 只检查
means/scales/quaternion/opacity 中是否存在 live geometry graph，不以 SH 作为 attachment
证据。全局 gate 仍是唯一 policy gate；该逐分支检查只是把 attachment 与当前 branch
真实可用的图求交，不能把配置为 off 的 branch 重新打开。

### 2.2 alpha 的 Jacobian-only 语义

8D geometry vector 使用 forward-identical gate：

```python
g_used = g.detach() + alpha * (g - g.detach())
```

- alpha=0、0.25、1 的 forward 逐 tensor 相同；
- 对 Parent/LocalGS geometry 的 Jacobian 分别为 0、0.25、1 倍；
- alpha=0 时 residual adapter 的参数仍可训练，因为 adapter 仍读取 detached geometry；
- alpha 不作用于 legacy 17D baseline，也不改变 Parent projection/lifting 数值。

### 2.3 validation/no-grad 合同

IForwardRunner 的 `validate`、`demo`、`replay` 与 legacy validation runner 均注入同一个
frozen evaluation metadata：

```text
observation_feedback_eval_mode = frozen_no_grad
validation_render_only = true
allow_grad = false
```

rollout 必须把 `validation_render_only` 传播到每个 `visit_meta`。即使 validation 内真实
updater delta 已使 `model_update_count > 0`，Functional Parent 也只能 forward-only；不得在
observe 内部用 `torch.enable_grad()` 重新开启图。

### 2.4 source-feedback forward parity 与静态 DINO cache

低频 source-feedback parity probe 比较 feedback 和 detached-reference 两条前向。两条路径
必须复用同一个静态 DINO tensor，不能各自执行一次冻结 DINO 后再用放宽容差掩盖 AMP
非确定性：当实际 view 数不超过 `cnn_view_chunk_size` 时，两者均使用基础 cache key；只有
确实发生 view chunking 时，才共同使用 `(base_key, view_chunk, start, end, total)`。在共享
cache entry 后，`source_rgb`、`features_2d`、detail 和 valid mask 均继续执行严格 forward
parity；相对容差仅保留为后备诊断，不作为正常训练的预期路径。

---

## 3. Functional Parent 数据层

### 3.1 数据结构

```python
@dataclass(frozen=True)
class FunctionalChildStats:
    mass: torch.Tensor
    tau_area: torch.Tensor
    diag_cov: torch.Tensor


@dataclass(frozen=True)
class FunctionalParentBranch:
    assignment: BigGSBranchAssignment | BigGSRigidActiveAssignment
    projection: BigGSParentProjection
    child_stats_detached: FunctionalChildStats
    parent_mass_mean: torch.Tensor
    branch_name: str


@dataclass(frozen=True)
class FunctionalParentPack:
    bg: FunctionalParentBranch
    distant: FunctionalParentBranch | None
    rigid_active: FunctionalParentBranch | None
```

assignment 持久但 graph-free，由 `IForwardState.biggs_state` 按 scene/segment topology 缓存。
Functional Parent geometry 只存在于当前 measurement。

### 3.2 Pack 接口

```python
build_functional_parent_pack(
    ...,
    attached_by_branch: Mapping[str, bool] | None = None,
    attached: bool | None = None,
) -> FunctionalParentPack
```

- Stage 3.4 runtime 必须传 `attached_by_branch`；
- scalar `attached` 仅保留给旧测试/兼容调用；
- 两者互斥，且必须提供其中一个；
- present branch 必须在 mapping 中出现；
- branch 顺序固定为 bg → distant → rigid-active。

### 3.3 Exact projector 与 stats

P0 继续复用 `cuda_exact_diag` autograd projector及其 forward-only/reference 实现，不新增
exact-with-stats wrapper。`compute_child_projection_stats()` 是 dynamic-tau-area、tau-area
和 diagonal covariance 的共享公式来源。

Parent projection 保持当前 exact reference 的：

- `dynamic_tau_area` child mass；
- diagonal covariance；
- identity Parent quaternion；
- branch-specific scale clamp；
- tau-area Parent opacity及 cap/min；
- bg/distant/rigid branch tau scale。

GDKV delta summary 使用 assignment 中的静态 `child_mass`；live dynamic mass只供当前
projector/relation forward。geometry-only CUDA projector 和 detached-mass 消融均属于 P1。

### 3.4 两套 Parent view

```python
parent_params_live = functional_branch.projection.params

parent_params_for_lift = {
    key: value.detach()
    for key, value in parent_params_live.items()
}
```

lifting scene 使用 `constant_zero` color。调用边界必须断言 means、scales、quats、opacity
均 `requires_grad=False`。support、valid、coords、layout cache 和 relation stats也在调用边界
断言 graph-free。

measurement 固定包含：

- `functional_parent_pack`；
- `functional_parent_assignments`；
- 现有 `assign_*` 兼容键；
- per-branch current Parent params；
- detached coords、context、support；
- `functional_parent_grad_active` 与 branch attachment 状态。

Stage 3.4 model loop不得创建、读取或更新 `biggs_parent_runtime`，也不得执行 refresh、
incremental update、drift/VJP bridge或 runtime timing。

---

## 4. v57 Parent token codec

### 4.1 稳定的 legacy baseline

v57 保留原 `Stage6ParentParamSupportCodec`、其 module name
`parent_spatial_backbone.param_support_codec.*` 及后续
`parent_spatial_backbone.token_builder.param_support_proj.*`。

legacy codec：

- 仍使用 Stage 3.3 的 17D 参数/support 表示；
- Parent params 和 support 均 detach；
- 只用于保留 Stage 3.3 checkpoint 的 forward token 分布；
- 不承担 Functional Parent geometry gradient。

### 4.2 Zero-init 8D residual

新增 `Stage34ParentGeometryResidualAdapter`：

```text
normalized means                         3
normalized absolute mean log-size       1
normalized zero-sum log-shape            3
tanh(opacity_logit)                      1
                                        ─
                                        8

Linear(8, 24) → GELU → Linear(24, 24)
```

第二个 Linear 的 weight 和 bias 必须精确 zero-init。最终输入为：

```python
param_support = detached_legacy_17d_support + geometry_residual_8d
```

不增加可学习 beta，也不随机替换 legacy token。zero-init 保证从 Stage 3.3 weights-only
初始化时，首个 forward 与旧 token 分布一致。

### 4.3 8D geometry 定义

`geometry_branch_id` 使用稳定三值 schema：

```text
bg = 0
distant = 1
rigid = 2
```

它与现有 near/far `branch_id` 语义独立，且传入 codec 前 detach。

means 使用固定 AABB 归一化。scale 使用 exact projector 的固定 branch bounds：

```python
lo = log(min_scale)
hi = log(max_scale_for_geometry_branch)
center = (lo + hi) * 0.5
half_range = max((hi - lo) * 0.5, eps)

clamped = clamp(scales_log, lo, hi)
log_size = mean(clamped, dim=-1, keepdim=True)
log_size_norm = (log_size - center) / half_range
log_shape_norm = (clamped - log_size) / half_range
```

固定范围为：

```text
min_scale          1.0e-3
max_scale_bg       0.60
max_scale_distant  3.00
max_scale_rigid    0.45
```

该表示保留 absolute/uniform size shift，避免逐 Parent LayerNorm 抹除尺度信息。8D residual
完全不索引 quats、`sh_dc` 或 `sh_rest`；identity quaternion 不再占据恒定 rot6d 通道。

### 4.4 ParentStructInput 与 optimizer

```python
ParentStructInput(
    params_for_embed=parent_params_live,
    coords=parent_means.detach(),
    support=parent_support.detach(),
    valid=parent_valid.detach(),
    geometry_branch_id=geometry_branch_id.detach(),
    geometry_alpha=functional_parent_alpha,
)
```

`geometry_residual_adapter` 显式属于 `parent_token_builder` optimizer group，学习率沿用
`1.5e-4`。legacy codec和后续 projection按既有 optimizer语义恢复。

### 4.5 配置与 schema

```yaml
parent_spatial:
  param_codec:
    mode: legacy17d_plus_geometry8d_residual
    schema: legacy17d_plus_geometry8d_residual_v1
    output_dim: 24
    grad_to_parent_params: true
    detach_legacy_params: true
    detach_support: true
```

`geometry_only_stage3_4` / `geometry_only_13d_v1` 是 pre-v57 历史 schema，不是可选 fallback。

---

## 5. 配置、版本与 fail-fast

正式 YAML 从完整 Stage 3.3 配置派生，并固定：

```text
model.iforward.version = stage3_4_functional_parentgs_lift
model.iforward.training_variant = stage3_4_functional_parentgs_lift
parent lifting = functional_parent_direct_lift
parent projector = cuda_exact_diag
parent state = functional_per_visit
Parent runtime/VJP = off
Relation Feedback = off
lifting geometry gradient = off
PTV3 coords/assignment/support/relation geometry gradient = off
```

Stage 3.4 初始化必须逐项 fail-fast：

- Functional Parent policy启用且 branches 精确为 bg/distant/rigid-active；
- `start_after_model_updates == 1`；
- Functional Parent alpha精确为五点正式调度；
- exact projector每 visit重算且无 CPU/Torch/surrogate fallback；
- codec mode/schema为 v57 值，legacy params和support均 detach；
- Parent lifting为 constant-zero color且 geometry detach；
- old Parent projection feedback 与 relation feedback关闭；
- Parent runtime不构建、不传递、不更新。

run manifest 与 checkpoint payload必须包含：

```text
iforward_version
training_variant
parent_param_codec_mode
parent_codec_schema
```

---

## 6. Checkpoint 迁移与 resume

### 6.1 Stage 3.3 → v57

唯一支持的跨版本入口是：

```text
native Stage 3.3 checkpoint + --init_weights_only
```

迁移必须加载：

- 完整 `parent_spatial_backbone.param_support_codec.*` legacy codec；
- 完整 `parent_spatial_backbone.token_builder.param_support_proj.*`；
- GDKV/PTV3/updater/frontend及其余形状兼容权重。

只允许新 `parent_spatial_backbone.geometry_residual_adapter.*` 在 Stage 3.3 source 中缺失，
并必须验证 output projection仍为精确 zero-init。禁止通过 `initialization.skip_keys` 跳过
legacy codec或 token projection。

### 6.2 原生 v57 strict resume

原生 Stage 3.4 checkpoint只有在下列身份全部匹配时才允许 strict resume或 Validation v4：

```text
iforward_version == stage3_4_functional_parentgs_lift
training_variant == stage3_4_functional_parentgs_lift
parent_codec_schema == legacy17d_plus_geometry8d_residual_v1
```

### 6.3 明确拒绝的输入

- 旧 Stage 3.4 13D checkpoint；
- 缺少 `parent_codec_schema` 的 Stage 3.4 checkpoint；
- schema 为 `geometry_only_13d_v1` 的 checkpoint；
- Stage 3.3 checkpoint的 strict resume；
- 带 legacy runtime/VJP state但声称是 native v57 的 checkpoint。

错误信息必须指示用户从 native Stage 3.3 checkpoint做 weights-only 初始化，而不是静默
随机重置或宽松加载。

---

## 7. 诊断合同

每次 Stage 3.4 observe至少输出：

```text
feedback/functional_parent/geometry_alpha
feedback/functional_parent/grad_active
feedback/functional_parent/forward_only
feedback/functional_parent/validation_render_only
feedback/functional_parent/branch/bg/configured
feedback/functional_parent/branch/bg/attached
feedback/functional_parent/branch/distant/configured
feedback/functional_parent/branch/distant/attached
feedback/functional_parent/branch/rigid_active/configured
feedback/functional_parent/branch/rigid_active/attached
```

继续记录 branch projector/lift/support/clamp 与 first-visit/update-ancestor 信息。Stage 3.4
不得出现 legacy runtime、incremental update、drift、refresh或 surrogate VJP指标。

隔离指标使用明确的静态合同名：

```text
feedback/parent_lift/geometry_grad_configured_off
feedback/ptv3_coords/geometry_grad_configured_off
feedback/relation/geometry_grad_configured_off
```

禁止再用硬编码 `geometry_grad_detected=0` 充当梯度证据。真正的证据来自：

1. lifting/PTV3/relation调用边界的 `requires_grad=False` 断言；
2. 独立 `autograd.grad` 路径测试；
3. Functional Parent output hook 的路径专属梯度 probe。

---

## 8. 测试与 Validation

### 8.1 Unit / integration

必须覆盖：

- CUDA、forward-only、reference exact projector三方 forward parity；
- projector对 means/scales/quats/opacity 的 directional derivative；
- functional detached stats与共享公式一致；
- `attached_by_branch` 三分支、optional `None` 和完整 rigid world route；
- rollout boundary 后的 branch-sparse update：已更新 branch attached，未更新但仍 present 的
  branch exact forward-only，且不得因全局 gate 为真而触发断言；
- K=2/K=3 `torch.no_grad()`、`frozen_no_grad`、`validation_render_only` 全部 graph-free；
- source/Functional Parent四种开关组合，不共享错误的 detach gate；
- source-feedback parity 的 chunked/unchunked DINO cache-key 一致性；reference 不得触发第二次
  静态 DINO 前向，真实 resume parity tensor 必须严格一致；
- alpha=0/0.25/1 forward完全一致及0/0.25/1 Jacobian比例；
- alpha=0时 residual adapter参数仍有梯度；
- uniform scale shift在8D输入中可观测；quaternion/SH无 residual梯度；
- Parent lifting只回传 feature gradient；PTV3、support、relation隔离；
- assignment-only delta summary与legacy runtime入口 forward parity；
- rollout boundary和 first-visit gate重置；
- Stage 3.3 weights-only migration、v57 strict resume、旧13D拒绝；
- Runner validate/demo/replay及legacy validation统一 frozen metadata。

### 8.2 Post-fix smoke 与真实 Validation v4

先从已知 Stage 3.3 step39999 checkpoint做10-step weights-only smoke，产出原生 v57
checkpoint。然后按 runbook依次执行：

1. seq10 / full-memory quick validation；
2. seq10+seq20 / 全 memory ablation full validation；
3. `validation_contract.json` 强校验；
4. K=15 bounded performance regression。

Validation v4必须证明：

- 所有 plan、trace、summary和HTML存在且非空；
- 所有数值 finite，无 OOM/NaN/Inf；
- 至少一次 K≥2 update 的 `model_update_count > 0`；
- 所有 eval event `grad_active=0` 且 `forward_only=1`；
- model parameter version未变，但 causal LocalGS/GDKV state按协议推进；
- assimilation、repair、order、repeat和要求的 memory ablation均完成；
- checkpoint version/variant/schema匹配；
- 无 legacy runtime/VJP/drift/refresh键。

完整命令与输出目录见 Acceptance Runbook。

### 8.3 K=15 profile

baseline为 Stage 3.3 source-only，candidate为 v57 Stage 3.4。Profiler不得篡改生产配置的
alpha schedule；source和Functional Parent都保留正式五点调度。为了测量完整 Jacobian 的
worst-case训练图，profiler把 trainer logical global step固定在 `15000 + local_offset`，因此
两条正式调度自然取 alpha=1。每个 rollout的 first visit仍由 update-ancestor gate保持
forward-only，后续 visits使用完整 Jacobian。

这种做法既覆盖 K=15最重训练图，又不为 profiler放宽 Stage 3.4生产 fail-fast。验收阈值：

```text
candidate peak allocated CUDA memory <= 1.15 × baseline
candidate median synchronized step time <= 1.20 × baseline
post-cleanup retained growth <= 64 MiB
```

### 8.4 延后的 1000-step matched B/C

本轮只定义实验合同，不实现或执行 alpha=0 control。B/C必须使用同一 Stage 3.3
weights-only init、相同数据/seed/source schedule和完全相同 forward：

| Run | Functional Parent alpha | 目的 |
|---|---|---|
| B | 独立受控 alpha=0 ablation variant | exact Functional Parent forward，关闭 geometry Jacobian |
| C | 正式五点调度 | 只增加计划中的 geometry Jacobian |

生产 `stage3_4_functional_parentgs_lift` 必须继续严格要求正式五点调度，不能用普通 CLI
override绕过 fail-fast。正式运行 B/C 前，需另行新增有独立 version/variant/config/manifest
身份的 alpha=0 ablation；该 P1 实验入口不属于本轮 v57 P0。

不得再用“随机13D codec的 Stage3.4 对 Stage3.3”作为 matched 方法实验，因为 codec forward
分布不匹配，无法隔离 Functional Parent梯度效果。

---

## 9. P1 延后项

以下不阻塞 v57 P0：

- `mass_gradient_mode: detached_weights` 消融；
- geometry-only CUDA projector；
- soft assignment、full covariance或Parent mixture；
- Parent lifting geometry backward；
- relation geometry feedback；
- 1000-step matched B/C和长程质量实验。

只有 K=15 profile证明 SH输出/保存是主要瓶颈时，才评估 geometry-only CUDA projector。
P1不得改变 v57 checkpoint schema或恢复 incremental Parent runtime。

---

## 10. Definition of Done

- [ ] Stage 3.4 version/config/manifest均声明 v57 codec schema；
- [ ] bg、distant、rigid-active均受独立 Functional Parent policy控制；
- [ ] alpha五点调度、first-update gate与 no-grad validation gate一致；
- [ ] 每 visit exact重算 Parent，不创建 legacy runtime/VJP/relation bridge；
- [ ] detached legacy 17D + zero-init 8D residual forward兼容；
- [ ] scale absolute size可观测，quaternion/SH不进入 residual；
- [ ] lifting、PTV3、assignment、support、relation梯度边界通过；
- [ ] Stage 3.3 weights-only初始化成功且 legacy codec/projection真实加载；
- [ ] native v57 strict resume成功，旧13D Stage 3.4 checkpoint明确拒绝；
- [ ] 自动测试通过；
- [ ] post-fix smoke产出 native v57 checkpoint；
- [ ] Validation v4 quick/full contract通过；
- [ ] K=15无OOM/NaN/retained growth且显存/时间达标；
- [ ] 1000-step matched B/C只记录实验合同和延期模板，未误作为本轮完成门禁。

---

## 11. 最终原则

```text
真实图像反馈：
LocalGS → source render → frontend → updater

粗几何反馈：
LocalGS → exact Functional ParentGS → alpha-gated 8D residual → updater

稳定 token 基线：
Detached Parent params/support → legacy 17D codec

图像采样：
Detached ParentGS → 2D lifting → Parent context

明确停止：
Parent lifting geometry、PTV3 routing、assignment、support、relation geometry
```

Stage 3.4 v57 的核心是：同一个 exact projector负责 Parent forward/backward，三分支共享
清晰的训练能力和统一 no-grad gate；同时用 forward-compatible residual把新几何信用路径与
Stage 3.3 token分布、checkpoint迁移和 validation身份严格绑定。
