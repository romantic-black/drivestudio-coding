# StreetForward Stage 6_0 Phase A 重构总体方案

> 范围：本方案只覆盖三件事：
>
> 1. 定义 `RolloutPlan / Role / Batch` 协议，让 Phase A scheduler、dataset、trainer 围绕统一契约协作。
> 2. 第一阶段引入 legacy facade：训练入口不再直接暴露 Stage5_4 继承链，但内部 runtime 仍是 `MinimalStreetForwardStage6_0`。
> 3. 先迁移 Phase A legacy parity recipe，形成新架构里的最小可测闭环；真正移除继承等 measurement/renderer/event builder 完全抽出后再做。

---

## 0. 当前实现中已经存在的 Phase A 语义

当前 `MinimalStreetForwardStage6_0` 名义上继承 `MinimalStreetForwardStage5_4`，但类注释已经说明：Stage6 只复用 Stage5_4 的 V4 source measurement helpers 和 renderer，不执行 Stage5_3/5_4 recurrent update、history、gate、support EMA、train_step paths。这个信息出现在 `models/streetforward/minimal_trainer_stage6_0.py:57-64`。

Phase A 的真实 forward 语义已经比较明确：

```text
scheduler emits K-step local rollout
  ↓
resolver maps image refs to source/target tensor indices
  ↓
init LocalGSState from persistent node states
  ↓
for k in 0..K-1:
    observe evidence with V4 measurement
    build near/far Stage6StructInput
    routed struct event decoder produces EventPack
    posterior updater produces DeltaPack
    LocalGSState.apply_delta
    render block_loss targets
    render nearby_loss targets
    add delta regularization
  ↓
loss.backward
optimizer.step
optional detached writeback to persistent node states
```

这条路径在当前代码里的对应位置是：

- Phase A batch resolver：`models/streetforward/stage6_0/v9_role_resolver.py:205-303`
- V4 evidence measurement：`models/streetforward/minimal_trainer_stage6_0.py:1457-1541`
- measurement → struct event：`models/streetforward/minimal_trainer_stage6_0.py:1794-1831`
- event → posterior update：`models/streetforward/minimal_trainer_stage6_0.py:1833-1867`
- render loss：`models/streetforward/minimal_trainer_stage6_0.py:2004-2068`
- Phase A forward loop：`models/streetforward/minimal_trainer_stage6_0.py:2518-2629`
- train step + detached writeback：`models/streetforward/minimal_trainer_stage6_0.py:3510-3617`

---

## 1. Phase A 模型理念

Phase A 不是“长时序记忆训练”，也不是“query view prediction”。它的科研对象是：

> 给定同一个局部 block 的一次或多次观测，模型能否学习一个稳定、可微、受约束的 posterior update，使 Gaussian state 在局部渲染监督下逐步变好？

因此 Phase A 的核心不是 history memory，而是一个局部贝叶斯式更新闭环：

```text
prior local Gaussian state
  + evidence image measurement
  → structured observation event
  → bounded posterior delta
  → updated local Gaussian state
  → block / nearby render supervision
```

Phase A 的关键理念有五条。

### 1.1 Evidence 和 Loss 必须分离

Phase A 里 `evidence` 是 update-only，`block_loss` / `nearby_loss` 是 loss-only。`nearby_loss` 不能泄漏到 evidence。当前 resolver 已经有这个检查：`nearby_loss_refs` 与 `evidence_refs` 交集非空时直接报错。

这应该变成新协议的硬规则，而不是隐藏在 `request_meta` 字符串里。

### 1.2 V4 measurement 是观测前端，不是 Stage6 的父类

V4 measurement 的作用是把 source images 和当前 local state 转成每个 Gaussian point 的 2D feature、support weight、obs code。它是 Phase A 的感知前端。

Stage6 真正要学习的是：

```text
V4 measurement + current GS params → structured event → bounded delta
```

所以 Stage6 不应该继承 Stage5_4；Stage5_4 应该被包成一个组件，例如：

```text
Stage5V4MeasurementAdapter
Stage5RendererAdapter
Stage5NodeStateProviderAdapter
```

### 1.3 Phase A 的 rollout 是局部 inner-K unroll

当前 scheduler 在 Phase A 中采样 `inner_K`，并对同一个 source frame 重复做 K 次 evidence update。block loss 每步都有，nearby loss 默认只在 final step 出现。当前 scheduler 构造逻辑在 `datasets/train_scheduler_v9.py:832-887`。

新系统里这应该被命名为：

```text
PhaseALocalUnrollPlan
```

而不是继续叫 V9 ViewSetRollout。

### 1.4 Posterior delta 必须有物理边界

当前 `Stage6PosteriorUpdater` 对 means、scale、quat、opacity、SH、hidden 都用 clamp/tanh 限制步长，并且有 noop gate 和 confidence head。这个设计应该保留，因为它是让 iterative update 稳定的关键。

### 1.5 Persistent state 写回必须 detached

Phase A 训练可以在一个 block 结束后把 local state 写回 persistent node state，但必须 detached，不能让跨 batch 图意外连起来。当前 `LocalGSState.writeback_detached` 做了这个约束，见 `models/streetforward/stage6_0/local_gs_state.py:178-190`。

新系统中 writeback 应该是 runner 的策略，而不是 recipe 内部偷偷修改全局状态。

---

## 2. 目标架构

新的 Phase A 最小闭环应该长这样：

```text
PhaseAScheduler
  emits RolloutPlan
        ↓
RolloutPlanValidator
        ↓
ImageRefBatchAssembler
  materializes Batch
        ↓
PhaseABatchResolver
  returns ResolvedBatch
        ↓
PhaseARecipe.forward
  uses adapters/modules/losses
        ↓
TrainRunner
  backward / optimizer / detached writeback / logging
```

依赖方向必须固定：

```text
protocols  ← scheduler
protocols  ← assembler
protocols  ← resolver
protocols  ← recipe
recipe     ← adapters/modules/losses
runner     ← recipe interface
legacy     ← adapters only
```

禁止反向依赖：scheduler 不应该 import trainer；dataset 不应该知道 `MinimalStreetForwardStage6_0`；recipe 不应该读取 scheduler 私有 `_scheduler_v9` 字段。

---

## 3. 协议层设计

建议新增包：

```text
streetforward_core/
  protocols/
    refs.py
    roles.py
    rollout.py
    batch.py
    resolved.py
```

### 3.1 ImageRef

```python
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class ImageRef:
    frame_idx: int
    cam_idx: int

    @staticmethod
    def from_raw(x: tuple[int, int] | list[int]) -> "ImageRef":
        if len(x) != 2:
            raise ValueError(f"ImageRef requires length 2, got {x!r}")
        return ImageRef(frame_idx=int(x[0]), cam_idx=int(x[1]))

    def as_tuple(self) -> tuple[int, int]:
        return (int(self.frame_idx), int(self.cam_idx))
```

不要继续在不同模块里重复 `Tuple[int, int]` + `_as_ref`。

### 3.2 Role

```python
from enum import Enum

class Role(str, Enum):
    EVIDENCE = "evidence"
    BLOCK_LOSS = "block_loss"
    NEARBY_LOSS = "nearby_loss"

    # legacy only, Phase A forbids these
    PREFIX_LOSS = "prefix_loss"
    QUERY_LABEL = "query_label"
    AUX_LOSS = "aux_loss"
```

Phase A 的 role contract：

| Role | 在 source_views? | 在 targets? | 可 update? | 可 render loss? |
|---|---:|---:|---:|---:|
| `evidence` | yes | no | yes | no |
| `block_loss` | no | yes | no | yes |
| `nearby_loss` | no | yes | no | yes |
| `prefix_loss` | no | no | no | no |
| `query_label` | no | no | no | no |

### 3.3 RolloutPlan

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class RolloutStep:
    step_idx: int
    evidence_refs: tuple[ImageRef, ...]
    block_loss_refs: tuple[ImageRef, ...]
    nearby_loss_refs: tuple[ImageRef, ...] = ()

@dataclass(frozen=True)
class RolloutPlan:
    protocol_version: str
    phase: str
    scene_id: int
    segment_id: int
    episode_id: int
    num_cams: int
    inner_K: int
    steps: tuple[RolloutStep, ...]
    meta: dict[str, Any] = field(default_factory=dict)
```

Phase A 具体化：

```python
PHASE_A_NAME = "phase_A_block_local_unroll"

@dataclass(frozen=True)
class PhaseALocalUnrollPlan(RolloutPlan):
    source_keyframe_idx: int | None = None
    block_idx: int | None = None
```

### 3.4 Plan Validator

所有 planner 输出后必须先过 validator：

```python
def validate_phase_a_plan(plan: RolloutPlan) -> None:
    if plan.protocol_version != "sf.phase_a.v1":
        raise ValueError("Phase A requires protocol_version=sf.phase_a.v1")
    if plan.phase != PHASE_A_NAME:
        raise ValueError("Phase A plan has wrong phase")
    if plan.inner_K < 1:
        raise ValueError("Phase A requires inner_K >= 1")
    if len(plan.steps) != plan.inner_K:
        raise ValueError("len(steps) must equal inner_K")

    evidence: set[ImageRef] = set()
    nearby: set[ImageRef] = set()
    for step in plan.steps:
        if not step.evidence_refs:
            raise ValueError(f"step {step.step_idx} requires evidence_refs")
        evidence.update(step.evidence_refs)
        nearby.update(step.nearby_loss_refs)
    if evidence & nearby:
        raise ValueError("Phase A nearby_loss_refs leaked into evidence_refs")
```

这些规则对应当前 resolver 和 scheduler 的隐含规则，但应该提前到 plan 层 fail-fast。

---

## 4. Batch 协议设计

当前 batch 是 dict，里面混合：

```text
source_views
targets
request_meta
_scheduler_v9
_scheduler_v9_aligned_info
scene_id / segment_id
```

重构后不要立刻消灭 dict，因为 dataset / trainer 里有大量历史代码。建议分两层：

1. `RawBatch`: 兼容旧 dataset 输出，仍然是 `dict[str, Any]`。
2. `ResolvedPhaseABatch`: recipe 使用的强类型视图。

### 4.1 BatchAssembler

```python
class BatchAssembler(Protocol):
    def materialize(self, plan: RolloutPlan) -> dict[str, Any]:
        ...
```

Phase A assembler 的职责：

```text
input: PhaseALocalUnrollPlan
output RawBatch:
  source_views: evidence refs 去重后的图像 / view tensors
  targets: block_loss + nearby_loss refs 去重后的 target tensors
  request_meta: 只保留兼容字段
  rollout_plan: 原始强类型 plan 或 plan_id
```

Phase A dataset adapter 直接从 `PhaseALocalUnrollPlan` 物化 image refs：

```python
class PhaseAImageRefBatchAssembler:
    def __init__(self, dataset):
        self.dataset = dataset

    def materialize(self, plan: PhaseALocalUnrollPlan) -> dict[str, Any]:
        validate_phase_a_plan(plan)
        source_refs = dedupe(step.evidence_refs for step in plan.steps)
        target_refs, target_roles = dedupe_roles(plan.steps)
        batch = self.dataset._assemble_segment_batch_from_image_refs(
            plan.scene_id,
            plan.segment_id,
            source_refs,
            target_refs,
            include_test=False,
            test_image_refs=None,
            enforce_target0_equals_source=False,
            target_ref_purpose="train",
        )
        batch["request_meta"].update({
            "scheduler_version": "phase_a_core_v1",
            "scheduler_phase": plan.phase,
            "assembly_mode": "image_ref_v9",
            "target_image_roles": target_roles,
        })
        batch["rollout_plan"] = plan
        return batch
```

### 4.2 ResolvedPhaseABatch

```python
@dataclass(frozen=True)
class ResolvedPhaseABatch:
    raw: dict[str, Any]
    plan: PhaseALocalUnrollPlan

    source_index_by_ref: dict[ImageRef, int]
    target_index_by_ref: dict[ImageRef, int]

    evidence_source_indices_by_step: tuple[tuple[int, ...], ...]
    block_target_indices_by_step: tuple[tuple[int, ...], ...]
    nearby_target_indices_by_step: tuple[tuple[int, ...], ...]
```

Resolver 只做一件事：把 plan refs 映射到 batch tensor indices，并检查顺序、role、leakage。

```python
class PhaseABatchResolver:
    def resolve(self, batch: dict[str, Any]) -> ResolvedPhaseABatch:
        plan = batch.get("rollout_plan")
        if plan is None:
            plan = legacy_v9_batch_to_phase_a_plan(batch)
        validate_phase_a_plan(plan)
        ...
```

这一步替代当前 `resolve_v9_phase_a_batch(batch)` 直接解析 `request_meta` 的方式。

---

## 5. Scheduler 迁移方案

短期不要重写整个 `TrainSchedulerV9`。先加一个 adapter，让它输出新协议。

```python
class PhaseAScheduler(Protocol):
    def next_plan(self) -> PhaseALocalUnrollPlan:
        ...

class LegacyV9PhaseASchedulerAdapter:
    def __init__(self, legacy_v9):
        self.legacy_v9 = legacy_v9

    def next_plan(self) -> PhaseALocalUnrollPlan:
        st = self.legacy_v9.current_episode_state
        legacy_plan = self.legacy_v9._build_phase_a_block_unroll_plan(st)
        return convert_v9_phase_a_plan(legacy_plan)
```

`convert_v9_phase_a_plan` 只转换字段，不加载 tensor，不调用 trainer：

```python
def convert_v9_phase_a_plan(v9: ViewSetRolloutBatchV9) -> PhaseALocalUnrollPlan:
    steps = []
    for s in v9.steps:
        steps.append(RolloutStep(
            step_idx=int(s.step_idx),
            evidence_refs=tuple(ImageRef.from_raw(x) for x in s.evidence_refs),
            block_loss_refs=tuple(ImageRef.from_raw(x) for x in s.block_loss_refs),
            nearby_loss_refs=tuple(ImageRef.from_raw(x) for x in s.nearby_loss_refs),
        ))
    return PhaseALocalUnrollPlan(
        protocol_version="sf.phase_a.v1",
        phase="phase_A_block_local_unroll",
        scene_id=int(v9.scene_id),
        segment_id=int(v9.segment_id),
        episode_id=int(v9.episode_id),
        num_cams=int(v9.num_cams),
        inner_K=int(v9.inner_K),
        steps=tuple(steps),
        source_keyframe_idx=int(v9.steps[0].source_keyframe_idx),
        block_idx=int(v9.steps[0].block_idx),
        meta={
            "legacy_scheduler_version": "v9",
            "episode_start_keyframe_pos": int(v9.episode_start_keyframe_pos),
        },
    )
```

长期再把 `TrainSchedulerV9._build_phase_a_block_unroll_plan` 中 Phase A 相关逻辑迁出，形成真正的 `PhaseALocalUnrollScheduler`。

---

## 6. Stage5_4 继承改 adapter

当前 Stage6 继承 Stage5_4 的主要原因是复用：

```text
1. node state 初始化 / persistent cache
2. V4 measurement 前端
3. renderer
4. 部分 optimizer/checkpoint 工具
```

第一阶段的新训练入口是 `Stage6PhaseAFacadeTrainer`：外壳不继承 Stage5_4，但内部组合的 legacy runtime 仍是 `MinimalStreetForwardStage6_0`。真正的“Stage6 不再继承 Stage5_4”要等下面四类 adapter 完全接管后完成。

### 6.1 Component Interfaces

```python
class NodeStateProvider(Protocol):
    def get_or_init(self, batch: dict[str, Any]) -> tuple[NodeStateBackground, NodeStateRigid | None, NodeStateDistant | None]:
        ...

class MeasurementFrontend(Protocol):
    def observe(
        self,
        *,
        local_state: LocalGSState,
        batch: dict[str, Any],
        source_indices: list[int],
        source_frame_idx: int,
    ) -> dict[str, Any]:
        ...

class Renderer(Protocol):
    def render_loss(
        self,
        *,
        local_state: LocalGSState,
        batch: dict[str, Any],
        target_indices: list[int],
        mask_policy: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ...

class PersistentWriteback(Protocol):
    def writeback_detached(
        self,
        local_state: LocalGSState,
        node_state_bg: NodeStateBackground,
        node_state_distant: NodeStateDistant | None,
        node_state_rigid: NodeStateRigid | None,
    ) -> None:
        ...
```

### 6.2 Legacy Adapter

迁移早期可以保留一个 legacy holder，但它不再是 Stage6 的父类：

```python
class Stage5RuntimeAdapter:
    def __init__(self, stage5_runtime: MinimalStreetForwardStage5_4):
        self.stage5 = stage5_runtime

class Stage5V4MeasurementAdapter:
    def __init__(self, runtime: Stage5RuntimeAdapter):
        self.runtime = runtime

    def observe(self, *, local_state, batch, source_indices, source_frame_idx):
        return self.runtime.stage5._observe_v4_measurement(
            local_state=local_state,
            batch=batch,
            source_indices=source_indices,
            source_frame_idx=source_frame_idx,
        )
```

但更干净的方式是把 `_observe_v4_measurement` 从 `MinimalStreetForwardStage6_0` 中复制/移动到 adapter，并把它依赖的 helpers 也通过 Stage5 runtime 提供。

建议分两步：

#### Step A: facade adapter

```text
Stage6PhaseARecipe 仍调用 Stage6LegacyFacade.observe_v4_measurement
PhaseARecipe 仍通过 Stage6LegacyFacade 调用 legacy private helpers
MinimalStreetForwardStage6_0 保留为 legacy runtime、Phase B 入口和 parity reference
```

#### Step B: real adapter

```text
把 _observe_v4_measurement 及其依赖 helpers 移入 Stage5V4MeasurementAdapter
PhaseARecipe 不再通过 Stage6LegacyFacade 依赖 MinimalStreetForwardStage6_0 private methods
```

### 6.3 为什么必须这样做

继承的问题不是“代码风格不好”，而是语义错误：

```text
MinimalStreetForwardStage6_0 is-a MinimalStreetForwardStage5_4  ❌
Stage6PhaseA has-a Stage5V4MeasurementFrontend          ✅
Stage6PhaseA has-a Stage5RendererAdapter                ✅
```

Phase A 继承 Stage5_4 会让人误以为 Stage5 的 history/gate/recurrent update 仍在协议中。实际上当前类注释已经说这些都不执行。adapter 能把这种关系表达准确。

---

## 7. Phase A Recipe 设计

### 7.1 Recipe 输入输出

```python
@dataclass
class PhaseAForwardOutput:
    loss: torch.Tensor
    local_state: LocalGSState
    node_state_bg: NodeStateBackground
    node_state_distant: NodeStateDistant | None
    node_state_rigid: NodeStateRigid | None
    resolved: ResolvedPhaseABatch
    per_step: list[dict[str, float]]
    pred_rgbs: list[torch.Tensor]
    gt_images: list[torch.Tensor]
```

### 7.2 Recipe 类

```python
class PhaseARecipe(nn.Module):
    def __init__(
        self,
        *,
        node_state_provider: NodeStateProvider,
        measurement: MeasurementFrontend,
        event_builder: PhaseAEventBuilder,
        posterior_updater: Stage6PosteriorUpdater,
        renderer: Renderer,
        loss_cfg: PhaseALossConfig,
        resolver: PhaseABatchResolver,
    ):
        super().__init__()
        self.node_state_provider = node_state_provider
        self.measurement = measurement
        self.event_builder = event_builder
        self.posterior_updater = posterior_updater
        self.renderer = renderer
        self.loss_cfg = loss_cfg
        self.resolver = resolver

    def forward(self, batch: dict[str, Any]) -> PhaseAForwardOutput:
        resolved = self.resolver.resolve(batch)
        node_bg, node_rigid, node_distant = self.node_state_provider.get_or_init(batch)
        local_state = LocalGSState.from_node_states(
            bg=node_bg,
            distant=node_distant,
            rigid=node_rigid,
            hidden_dim=self.loss_cfg.stage_hidden_dim,
        )

        total_loss = local_state.bg.means.new_tensor(0.0)
        per_step = []
        pred_rgbs = []
        gt_images = []

        for k in range(resolved.plan.inner_K):
            evidence_refs = resolved.plan.steps[k].evidence_refs
            source_frame_idx = int(evidence_refs[0].frame_idx)

            measurement = self.measurement.observe(
                local_state=local_state,
                batch=batch,
                source_indices=list(resolved.evidence_source_indices_by_step[k]),
                source_frame_idx=source_frame_idx,
            )
            event = self.event_builder.build(local_state=local_state, measurement=measurement)
            delta, aux = self.posterior_updater(event=event, ctx_current=None, ctx_vsm=None)
            delta = self.event_builder.apply_branch_scope(delta)
            local_state = local_state.apply_delta(delta)
            local_state = self.event_builder.constrain_state(local_state)

            block_loss, block_stats = self.renderer.render_loss(
                local_state=local_state,
                batch=batch,
                target_indices=list(resolved.block_target_indices_by_step[k]),
                mask_policy=self.loss_cfg.block_mask_policy,
            )
            nearby_loss, nearby_stats = self.renderer.render_loss(
                local_state=local_state,
                batch=batch,
                target_indices=list(resolved.nearby_target_indices_by_step[k]),
                mask_policy=self.loss_cfg.nearby_mask_policy,
            )
            reg_loss, reg_stats = delta_regularization(delta, **self.loss_cfg.regularization_kwargs())

            near_w = self.loss_cfg.nearby_weight(global_step=batch.get("global_step", 0), k=k)
            step_w = self.loss_cfg.step_gamma ** (resolved.plan.inner_K - 1 - k)
            loss_k = step_w * (self.loss_cfg.block_weight * block_loss + near_w * nearby_loss + reg_loss)
            total_loss = total_loss + loss_k

            per_step.append(build_phase_a_step_metrics(k, block_loss, nearby_loss, block_stats, nearby_stats, reg_stats, aux))

        return PhaseAForwardOutput(
            loss=total_loss,
            local_state=local_state,
            node_state_bg=node_bg,
            node_state_distant=node_distant,
            node_state_rigid=node_rigid,
            resolved=resolved,
            per_step=per_step,
            pred_rgbs=pred_rgbs,
            gt_images=gt_images,
        )
```

### 7.3 Recipe 不应该做的事

`PhaseARecipe` 不应该：

```text
- 创建 scheduler
- 从 YAML 里到处 _cfg_get
- 决定 optimizer groups
- 写 TensorBoard / WandB
- 修改 persistent node state
- 处理 Phase B / VSM / query decoder
- 解析 _scheduler_v9 私有字段
```

这些职责分别属于 config builder、runner、logger、resolver、adapter。

---

## 8. PhaseAEventBuilder 设计

当前 `_build_stage6_struct_input_near`、`_build_stage6_struct_input_far`、`_build_stage6_event_from_measurement`、`_apply_branch_scope`、`_constrain_local_state_after_delta` 混在 trainer 中。

建议抽为：

```text
streetforward_core/modules/phase_a_event_builder.py
```

```python
class PhaseAEventBuilder(nn.Module):
    def __init__(
        self,
        *,
        struct_event_decoder: Stage6RoutedStructEventDecoder,
        branch_scope: BranchScopeConfig,
        aabb_provider: AABBProvider,
        detach_v4_outputs: bool,
        sh_degree: int,
    ):
        ...

    def build(self, *, local_state: LocalGSState, measurement: dict[str, Any]) -> EventPack:
        near_in = self.build_near_input(local_state, measurement)
        far_in = self.build_far_input(local_state, measurement)
        aabb_min, aabb_max = self.aabb_provider.get(...)
        event = self.struct_event_decoder(near_in=near_in, far_in=far_in, ...)
        return event

    def apply_branch_scope(self, delta: DeltaPack) -> DeltaPack:
        ...

    def constrain_state(self, state: LocalGSState) -> LocalGSState:
        ...
```

这个 builder 是 Phase A 的算法核心之一。它把“measurement feature”转成“可更新 GS 参数的结构化事件”。

---

## 9. 配置边界

Phase A 配置应该收敛成一个 schema，不再让 recipe 自己 `_cfg_get`。

```python
@dataclass(frozen=True)
class PhaseARecipeConfig:
    phase: Literal["phase_A_block_local_unroll"]
    mode: Literal["updater_only", "from_scratch"]
    hidden_dim: int
    event_dim: int
    feat_2d_dim: int
    detach_v4_outputs: bool
    train_2d_frontend: bool
    block_weight: float
    block_step_gamma: float
    block_mask_policy: str
    nearby_enable: bool
    nearby_weight: float
    nearby_warmup_steps: int
    nearby_final_step_only: bool
    nearby_mask_policy: str
    delta_l2_weight: float
    opacity_delta_l2_weight: float
    sh_delta_l2_weight: float
    scale_barrier_weight: float
    writeback_policy: Literal["block_end_detached", "none"]
```

配置校验必须集中在 builder 里：

```python
def validate_phase_a_config(cfg: FullConfig) -> PhaseARecipeConfig:
    require(cfg.model.stage == "6_0")
    require(cfg.model.phase == "phase_A_block_local_unroll")
    require(cfg.scheduler.phase == "phase_A_block_local_unroll")
    forbid(cfg.model.history_memory.enable)
    forbid(cfg.model.update_gate.enable)
    forbid(cfg.model.stage6_0.vsm.enable)
    forbid(cfg.model.stage6_0.query_decoder.enable)
    forbid(cfg.losses.phase_a.disabled.query_observation is False)
    forbid(cfg.losses.phase_a.disabled.prefix_render is False)
    ...
```

当前 `stage6_0_phase_a.yaml` 中的重要语义包括：

- `phase_a_mode: from_scratch`
- V4 fused measurement，允许训练 2D residual/fusion frontend，但不训练 Dinov2 和 V4 lift
- `local_rollout.writeback_policy: block_end_detached`
- struct event decoder 使用 token/param_obs_codec/near XcPE/far point MLP
- posterior updater 不接 current context / VSM context
- block render + final nearby render + delta regularization

对应配置位置可见 `configs/stage6_0_phase_a.yaml:321-420` 和 `configs/stage6_0_phase_a.yaml:619-644`。

---

## 10. TrainRunner 设计

`TrainRunner` 负责 optimizer、backward、grad check、writeback、logging。

```python
class TrainRunner:
    def train_step(self, batch: dict[str, Any], step: int) -> dict[str, Any]:
        batch = dict(batch)
        batch["global_step"] = int(step)

        self.recipe.train()
        self.optimizer.zero_grad(set_to_none=True)
        out = self.recipe(batch)
        out.loss.backward()

        grad_stats = self.grad_checker.check(self.recipe)
        grad_norm = self.grad_checker.clip_or_check(self.recipe)
        self.optimizer.step()

        if self.writeback_policy == "block_end_detached":
            self.writeback.writeback_detached(
                out.local_state,
                out.node_state_bg,
                out.node_state_distant,
                out.node_state_rigid,
            )

        self.optimizer.zero_grad(set_to_none=True)
        return self.metric_builder.phase_a_train_metrics(out, grad_stats, grad_norm)
```

这让 recipe 成为 pure forward，runner 成为训练控制器。

---

## 11. 日志指标协议

Phase A 必须稳定输出这些指标：

```text
phase_a/loss_total
phase_a/inner_K
phase_a/loss_block_final
phase_a/loss_nearby_final
phase_a/block_psnr_final
phase_a/nearby_psnr_final
phase_a/k{i}/loss_block
phase_a/k{i}/loss_nearby
phase_a/k{i}/block_valid_ratio
phase_a/k{i}/nearby_valid_ratio
grad/struct_event_decoder_near
grad/struct_event_decoder_far
grad/param_obs_codec
grad/posterior_updater
grad/measurement_frontend
state/num_bg
state/num_distant
state/num_rigid
```

旧 key 可以保留 alias 一段时间，比如 `phaseA/loss_total` → `phase_a/loss_total`，但新系统内部只使用 snake_case。

---

## 12. 测试计划

### 12.1 协议测试

```text
test_phase_a_plan_requires_inner_k_steps
test_phase_a_plan_requires_nonempty_evidence_each_step
test_phase_a_plan_rejects_nearby_evidence_overlap
test_phase_a_plan_rejects_prefix_or_query_roles
test_phase_a_plan_ref_order_is_stable
```

### 12.2 Scheduler adapter 测试

```text
test_legacy_v9_phase_a_adapter_converts_steps
test_adapter_preserves_inner_K
test_adapter_preserves_source_frame_all_cams_evidence
test_adapter_nearby_final_step_only
```

### 12.3 Batch resolver 测试

这些对应当前 `tests/test_minimal_stage6_0_phase_a.py:215-239` 的 resolver 行为：

```text
test_phase_a_resolver_maps_refs_to_source_indices
test_phase_a_resolver_maps_block_refs_to_target_indices
test_phase_a_resolver_maps_nearby_refs_to_target_indices
test_phase_a_resolver_rejects_target_role_mismatch
test_phase_a_resolver_rejects_missing_ref_in_batch
```

### 12.4 Adapter 测试

```text
test_stage5_v4_measurement_adapter_no_grad_mode
test_stage5_v4_measurement_adapter_from_scratch_grad_mode
test_stage5_renderer_adapter_batches_same_frame_cameras
test_node_state_provider_returns_stable_persistent_rows
```

### 12.5 Recipe smoke test

```text
test_phase_a_recipe_forward_returns_finite_loss
test_phase_a_recipe_backward_gives_struct_decoder_grad
test_phase_a_recipe_backward_gives_posterior_updater_grad
test_phase_a_recipe_updater_only_freezes_measurement_frontend
test_phase_a_recipe_from_scratch_trains_residual_unet_and_fusion_neck
test_phase_a_recipe_writeback_is_runner_only
```

### 12.6 Regression test

```text
test_new_phase_a_recipe_matches_legacy_loss_on_fake_batch
test_new_phase_a_recipe_matches_legacy_ref_mapping_on_v9_batch
test_new_phase_a_recipe_preserves_final_pred_rgbs_for_logging
```

---

## 13. 迁移路线

### PR-1: 建协议，不改训练行为

新增：

```text
streetforward_core/protocols/refs.py
streetforward_core/protocols/roles.py
streetforward_core/protocols/rollout.py
streetforward_core/protocols/batch.py
streetforward_core/protocols/validators.py
```

验收：协议测试通过；不改 legacy trainer。

### PR-2: V9 Phase A adapter

新增：

```text
streetforward_core/data/schedulers/legacy_v9_phase_a_adapter.py
streetforward_core/data/assemblers/v9_image_ref_batch_assembler.py
streetforward_core/data/resolvers/phase_a_resolver.py
```

验收：同一个 V9 plan 转出的 `PhaseALocalUnrollPlan` 与旧 `request_meta` 等价。

### PR-3: Stage5 runtime adapter

新增：

```text
streetforward_core/legacy/stage5_runtime.py
streetforward_core/legacy/stage5_v4_measurement_adapter.py
streetforward_core/legacy/stage5_renderer_adapter.py
streetforward_core/legacy/stage5_node_state_provider.py
```

验收：adapter 直接调用旧 V4 measurement / renderer，数值与旧 trainer 一致。

### PR-4: PhaseAEventBuilder

从 `MinimalStreetForwardStage6_0` 抽出：

```text
_build_stage6_struct_input_near
_build_stage6_struct_input_far
_build_stage6_event_from_measurement
_apply_branch_scope
_constrain_local_state_after_delta
```

验收：旧 tests 中 struct decoder / updater / clamp / branch scope 相关测试迁到 builder tests。

### PR-5: PhaseARecipe

新增：

```text
streetforward_core/recipes/phase_a_recipe.py
streetforward_core/train/runner.py
```

验收：fake batch 可以 forward/backward；关键参数组有梯度；冻结模块无梯度。

### PR-6: Legacy parity

同一个固定 batch 上跑：

```text
legacy MinimalStreetForwardStage6_0._forward_phase_a
new PhaseARecipe.forward
```

比较：

```text
loss_total
per-step block_loss / nearby_loss
delta reg stats
render stats
pred_rgbs 数量
```

允许浮点微小误差，不允许协议行为差异。

### PR-7: 训练入口切换

新的训练入口：

```bash
python -m streetforward_core.train.entry --config configs/stage6_0_phase_a.yaml
```

旧入口继续保留，但标记 legacy。

---

## 14. 文件最终形态建议

```text
streetforward_core/
  protocols/
    refs.py
    roles.py
    rollout.py
    validators.py
    batch.py
  data/
    schedulers/
      phase_a_scheduler.py
      legacy_v9_phase_a_adapter.py
    assemblers/
      image_ref_batch_assembler.py
      legacy_v9_batch_assembler.py
    resolvers/
      phase_a_resolver.py
  modules/
    phase_a_event_builder.py
    measurement_interfaces.py
    renderer_interfaces.py
  legacy/
    stage5_runtime.py
    stage5_v4_measurement_adapter.py
    stage5_renderer_adapter.py
    stage5_node_state_provider.py
  recipes/
    phase_a_recipe.py
  train/
    runner.py
    optimizer_builder.py
    grad_checker.py
    metric_builder.py
```

旧文件保留：

```text
models/streetforward/minimal_trainer_stage6_0.py
```

但逐步降级为：

```text
LegacyStage6Facade / parity reference only
```

---

## 15. 最小可交付版本

第一版不追求彻底重写，只追求三件事完成：

```text
1. scheduler 输出 PhaseALocalUnrollPlan
2. dataset 从 PhaseALocalUnrollPlan materialize batch
3. PhaseARecipe 不继承 Stage5_4，也不解析 _scheduler_v9
```

第一版可以仍然通过 adapter 调用旧 measurement 和 renderer。这已经足够切断最危险的继承语义，并建立 StreetForward 新架构的中心协议。

---

## 16. 成功标准

这次重构成功的标志不是代码少了多少，而是下面这些问题都有唯一答案：

```text
Q: Phase A 一个 batch 里哪些图像用于 update？
A: RolloutPlan.steps[k].evidence_refs。

Q: 哪些图像用于 block loss？
A: RolloutPlan.steps[k].block_loss_refs。

Q: nearby supervision 会不会泄漏进 evidence？
A: PlanValidator 和 Resolver 双重禁止。

Q: Stage6 为什么需要 Stage5？
A: 只需要 Stage5V4MeasurementAdapter 和 Stage5RendererAdapter，不是继承关系。

Q: Phase A 真正学习的模块是什么？
A: struct event decoder、posterior updater；from_scratch 模式还训练 2D residual/fusion frontend。

Q: persistent node state 何时更新？
A: Runner 根据 writeback_policy 在 optimizer.step 后 detached writeback。

Q: validation 是否复用同一 forward 逻辑？
A: 当前 `validate_v9_phase_a` 仍显式代理到 legacy validation path；recipe-backed validator 是下一阶段。
```

---

## 17. 一句话总结

Phase A 应该被重构为 StreetForward 的第一个“协议驱动 recipe”：scheduler 只声明局部观测-监督计划，dataset 只物化图像 refs，adapter 只提供 V4 measurement/rendering，PhaseARecipe 只学习 `measurement → struct event → posterior delta → local GS update → render loss` 这个科研闭环。

---

## 18. Phase B Long 当前主线

Phase B Long 不再和旧 `phase_B_viewset_rollout / scheduler_v9` 共用主配置。当前主线固定为：

```text
model.phase = 6_0_phase_b
scheduler_long_phase_b.enable = true
scheduler_long_phase_b.version = long_v1
validation_long_phase_b.enable = true
scheduler_v9.enable = false
validation_v9.enable = false
```

新的 Phase B Long batch 路径是：

```text
TrainSchedulerLongPhaseB
  emits PhaseBLongRolloutPlan
        ↓
validate_phase_b_long_plan
        ↓
PhaseBLongBatchAssembler
  materializes source/target ImageRef batch
        ↓
PhaseBLongBatchResolver
  returns ResolvedLongPhaseBBatch
        ↓
PhaseBLongRecipe / PhaseBLongTrainRunner
```

`PhaseBLongRolloutPlan` 的协议版本是 `sf.phase_b_long.v1`，scheduler 版本是 `long_v1`。允许的监督 role 只包括：

```text
final_history_recon
final_history_nvs
final_current_recon
final_current_nvs
```

`prefix_loss`、`query_label`、`block_loss`、`nearby_loss` 属于旧 Phase B/V9 语义，Long V1 resolver 和 validator 必须拒绝它们。Long V1 的科研边界是 `long VSM / offset decoder → final render loss`，不是 query decoder，也不是 Phase A 的大 K 版本。

Phase B Long 初始化只接受 `MinimalStreetForwardStage6_0.build_phase_b_export_checkpoint()` 生成的 `export_type=stage6_0_phase_a_for_phase_b` payload。普通 Phase A resume checkpoint 默认拒绝；`load_modules`、`freeze_after_load`、`train_new_modules` 不再作为配置 schema 暴露，因为当前加载、冻结和 trainability 策略是固定 contract。

当前 `Stage6PhaseBLongFacadeTrainer` 和 `PhaseBLongRecipe` 仍是 legacy-backed parity 层：外壳提供新 recipe/runner 入口，但内部仍通过 `MinimalStreetForwardStage6_0` 复用已有 state、renderer、optimizer/checkpoint 和 `_forward_6_0_phase_b_long`。真正把 FrozenPhaseAObserver、LongVSM、offset decoder 和 final render loss 完全迁出 legacy runtime，是下一阶段工作。
