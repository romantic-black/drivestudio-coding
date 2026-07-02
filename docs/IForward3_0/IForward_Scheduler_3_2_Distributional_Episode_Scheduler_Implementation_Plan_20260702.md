# IForward Scheduler 3_2 / Distributional Episode Scheduler 完整实现方案

日期：2026-07-02  
适用代码：`drivestudio_stage6_refactor_context_20260702_v43`  
适用模型：IForward 3_1 Low-rank Gated Delta KV 及后续 IForward 3_2 训练  
目标：在当前 Runtime / Scheduler / Validation / Demo 重构基础上，引入三分布 episode scheduler：`repeat_refine`、`shuffled_coverage`、`high_block_repair`。

---

## 0. 执行结论

当前结果已经足够说明：IForward 不能继续主要依赖 B1/B2 repeat-heavy assimilation 训练。模型在 repeat-heavy 训练下对 BnR1 / BnR2 repair 分布泛化很差；加入少量 repair / 乱序训练后 repair 指标快速提升，说明问题主要是 rollout distribution mismatch，而不是模型完全学不会。

Scheduler 3_2 的目标不是再做一个更复杂的 monolithic scheduler，而是在 v43 已经存在的 runtime 抽象上实现：

```text
RolloutDistributionSpec
  -> EpisodeRecipeSpec
  -> DistributionalEpisodeCompiler
  -> EpisodePlan / RolloutPlanV3
  -> IForwardRunner / TraceRecorder / ValidationReport / DemoReport
```

核心原则：

```text
1. 乱序必须进入主训练分布。
2. Repeat 保留，但降级为局部 refinement / stability regularizer。
3. Repair 使用高 block、低 repeat，作为 episode tail。
4. 单 episode 结构为：prelude(repeat_refine + shuffled_coverage) -> repair_tail(high_block_repair)。
5. Prelude 内 repeat 和 shuffle 不要永远固定顺序，推荐混合顺序。
6. Repair 优先从 prelude 已访问过的 positions 中采样，避免退化成 random assimilation。
7. 2D trainable 与 2D frozen/no-grad 对应不同 maxK；frozen/no-grad 时允许更高 K。
8. train / validation / debug demo 必须执行同一种 EpisodePlan 语义。
```

---

## 1. 当前代码基础

v43 已经具备以下基础设施。

### 1.1 Runtime data structures

目录：

```text
models/iforward/runtime/
  event.py
  plan.py
  runner.py
  adapter_stage3.py
  trace.py
  state_snapshot.py
  artifact_store.py
```

已有核心结构：

```python
EpisodeSpec
UpdateEvent
ProbeEvent
ControlEvent
EpisodePlan
IForwardRunner
TraceRecorder
Stage3SchedulerAdapter
```

这说明 Scheduler 3_2 不应该绕开 runtime，而应该将 distributional scheduler 输出编译成现有 `EpisodePlan` / `RolloutPlanV3`。

### 1.2 Existing Stage2_3 / Stage3_0 scheduler schema

当前 schema 已经有：

```python
Stage23StepPlan
RolloutPlanV3
EpisodePlanV3
```

`Stage23StepPlan` 已有字段：

```text
visit_kind
scheduler_phase
frame_gap
visit_order_gap
physical_frame_gap_abs
optimizer_memory_read/write
temporal_read/commit
physical_time_advance
validation_render_only
```

这些字段已经足够表达“visit order 与 physical order 不同”这一关键点。

### 1.3 Current model constraints

当前 3_1 模型约束：

```text
parent_optimizer_memory = lowrank_gated_delta_kv, K=16,V=32
parent_optimizer_mamba.enable = false
child_gather = support_center, num_taps=1
parent.type = legacy_direct_lift
repair_training.freeze_2d_frontend = true
repair_training.no_grad_2d_forward = true
```

因此 Scheduler 3_2 不应依赖增加 child query / multi-tap gather 来换效果。当前方向应从 rollout distribution 和 update sequence 上改。

---

## 2. 设计目标与非目标

### 2.1 目标

1. 用三种 rollout 分布训练 IForward：

```text
repeat_refine
shuffled_coverage
high_block_repair
```

2. 将乱序训练提前进入主训练阶段，而不是 30k 后才出现。
3. 保留 repeat，但限制为 B≤2，并随机 R / K budget。
4. 高 block repair 放在 episode tail，R 最大 2 或 3。
5. 支持 2D trainable 与 frozen/no-grad 两种 maxK 策略。
6. 让 train / validation / demo 使用同一 plan 语义。
7. 增加 debug metrics，使每个 rollout 能追踪 distribution_type、order_type、B、R、K、visited_ratio、repair_visited_ratio、train_2d_mode。

### 2.2 非目标

本轮不做：

```text
不改 GDKV K/V；
不改 child gather 查询数；
不改 parent lifting；
不把所有训练变成纯 repair；
不删除旧 scheduler_stage3_0；
不改变 dataset / asset loader；
不把 validation/demo 改成另一套 plan 语义；
不引入新的模型模块。
```

---

## 3. 三种 rollout 分布

## 3.1 Distribution A：repeat_refine

### 语义

`repeat_refine` 用于训练局部迭代能力：

```text
1. 同一 block 或两个 block 的多次 refinement；
2. repeat 后不发散；
3. current frame 精修；
4. fixed-point / no-op 行为。
```

它不负责主要 history robustness，也不负责 repair。

### 约束

```text
B 最大为 2。
R 随机。
K = sum(R_i) 受 max_k_repeat 限制。
```

### 推荐采样

不要再固定大量命名 pattern，如 B1R8 / B2R4。推荐：

```text
1. 先采 K budget；
2. 再采 B ∈ {1,2}；
3. 将 K 随机分配到 B 个 block；
4. R_i ≥ 1。
```

示例：

```text
K=8, B=1 -> [8]
K=8, B=2 -> [3,5] / [6,2] / [4,4]
K=10,B=2 -> [2,8] / [7,3]
```

### 推荐比例

```text
warmup: 30-40%
main:   15-25%
harden: 10-15%
```

---

## 3.2 Distribution B：shuffled_coverage

### 语义

`shuffled_coverage` 是 Scheduler 3_2 的主分布。它用于训练模型在不同输入顺序下合并多个观测约束：

```text
1. 多 block 覆盖；
2. 每个 block 少量 repeat；
3. 更新顺序不等于物理时间顺序；
4. memory 正常 read/write；
5. observation / parent_state / LocalGSState 正常更新。
```

它不是 repair，但应当 repair-like。

### 推荐范围

```text
B ∈ {3,4,6,8}
R ∈ {1,2}
```

若 2D trainable：

```text
maxK_shuffle ≈ 8 或 10
```

若 2D frozen/no-grad：

```text
maxK_shuffle ≈ 12 / 16
```

### 乱序类型

`shuffled_coverage` 不应只有一种 random。建议支持：

```text
local_shuffle:
  小窗口内打乱，保留局部时间邻近。

stratified_shuffle:
  从 episode 的不同时间段抽 block，再随机顺序访问。
  推荐作为主乱序方式。

global_shuffle:
  全局随机，最难，小比例用于 hardening。

reverse_or_pingpong:
  可选，用于验证时间方向依赖。
```

推荐初始比例：

```text
local_shuffle:      0.35
stratified_shuffle: 0.50
global_shuffle:     0.15
```

---

## 3.3 Distribution C：high_block_repair

### 语义

`high_block_repair` 是真正的 repair tail。它应当发生在 prelude 已经修改过 state 之后。

训练目标：

```text
1. 重新访问历史约束；
2. 修复先前 update 造成的破坏；
3. 训练 GDKV 在高 block 乱序访问下的 memory read/write；
4. 训练 state editor，而不是局部 repeat refiner。
```

### 推荐范围

```text
B ∈ {6,8,10,12}
R ∈ {1,2}
少量 R=3
```

2D frozen/no-grad 时可试：

```text
B ∈ {8,10,12,16}
R ∈ {1,2}
少量 B8R3 / B6R3
```

### Repair candidate policy

关键原则：

```text
repair 优先从 prelude 已访问过的 positions 中采样。
```

推荐策略：

```text
candidate_pool = visited_positions excluding current tail positions
if len(candidate_pool) < B:
    fill from unvisited_positions
```

这样 repair 真正在训练“修复先前迭代 / 兼容历史”，而不是退化成 random high-block assimilation。

---

## 4. Episode 结构

Scheduler 3_2 的 episode 分成两段：

```text
Episode
  ├── Prelude
  │     ├── repeat_refine rollouts
  │     └── shuffled_coverage rollouts
  └── RepairTail
        └── high_block_repair rollouts
```

### 4.1 Prelude

Prelude 的作用是制造一个真实迭代过程中的 state：

```text
state 不是 fresh；
LocalGSState 已经被多次编辑；
parent_state 已经更新；
GDKV memory 已经 read/write；
history_positions 有真实 visited context。
```

### 4.2 RepairTail

Repair tail 在 prelude 后执行：

```text
从 visited history 中采样 high-block positions；
随机顺序；
low repeat；
repair_training 可以启用 2D frozen/no-grad；
评估/训练 repair 和 history retention。
```

### 4.3 Prelude 内部顺序

不要永远固定：

```text
repeat -> shuffle -> repair
```

推荐：

```text
prelude 内 repeat 和 shuffle 按比例混合；
repair 固定在 tail。
```

原因：如果永远固定 repeat-then-shuffle-then-repair，模型可能过拟合新的 phase order。Prelude 内混合可以降低 schedule overfit。

---

## 5. Curriculum 设计

### 5.1 Phase 0：warmup

```text
start: 0
end:   3k 或 5k
```

目标：current update、GDKV write、基础 state update 站稳，但从一开始就见到少量乱序。

推荐：

```yaml
repeat_refine:      0.35
shuffled_coverage:  0.55
high_block_repair:  0.10
```

maxK：

```text
2D trainable:
  repeat 8
  shuffle 8
  repair 8
```

### 5.2 Phase 1：main

```text
start: 5k
end:   30k
```

目标：乱序覆盖成为主训练分布。

推荐：

```yaml
repeat_refine:      0.20
shuffled_coverage:  0.55
high_block_repair:  0.25
```

maxK：

```text
2D trainable:
  repeat 8
  shuffle 10
  repair 10 或 12

2D frozen/no-grad:
  repeat 12
  shuffle 12 或 16
  repair 16
```

### 5.3 Phase 2：hardening

```text
start: 30k
end: training end
```

目标：强化 high-block repair 和 order robustness，同时保留 shuffle coverage。

推荐：

```yaml
repeat_refine:      0.10
shuffled_coverage:  0.40
high_block_repair:  0.50
```

maxK：

```text
2D frozen/no-grad:
  repair 16 / 20 / 24
```

不建议 30k 后变成纯 repair。仍然保留 shuffle coverage，避免模型只会 explicit repair，不会持续合并新约束。

---

## 6. 2D training policy

Scheduler 3_2 应把 2D training policy 作为 rollout 级别条件。

### 6.1 Trainable mode

```text
2D frontend 有梯度；
显存高；
K 保守；
适合 current / feature adaptation。
```

推荐 maxK：

```text
repeat_refine:     8
shuffled_coverage: 8 或 10
high_block_repair: 8 或 12
```

### 6.2 Frozen / no-grad mode

```text
2D frontend no-grad；
显存低；
K 可以更大；
适合 high-block repair / order robustness。
```

推荐 maxK：

```text
repeat_refine:     12
shuffled_coverage: 12 或 16
high_block_repair: 16 / 20 / 24
```

### 6.3 第一版落地建议

第一版先保持现有模型行为：

```text
repair phase -> repair_training -> 2D frozen/no-grad
assimilation/shuffled phase -> 2D trainable
```

第二版再引入通用 `train_2d_mode`，允许 `shuffled_coverage_frozen`。

如果直接第一版引入 `train_2d_mode`，需要修改模型侧 `repair_training.kinds` 或新增更通用的 `frontend_training_policy`。这会扩大实现面。建议先用 phase/visit_kind 复用现有 repair_training 机制。

---

## 7. 配置设计

新增配置块：

```yaml
scheduler_stage3_2:
  enable: true
  version: distributional_episode_v1
  inherit_from: scheduler_stage3_0
  fail_fast: true
```

### 7.1 Distribution configs

```yaml
scheduler_stage3_2:
  distributions:
    repeat_refine:
      enable: true
      b_choices:
        1: 0.55
        2: 0.45
      k_budget:
        2: 0.10
        4: 0.30
        6: 0.30
        8: 0.20
        10: 0.10
      r_allocation: random_partition
      max_b: 2
      order:
        local: 0.7
        chronological: 0.3
      phase: assimilation
      visit_kind: repeat_refine

    shuffled_coverage:
      enable: true
      b_choices:
        3: 0.20
        4: 0.35
        6: 0.30
        8: 0.15
      r_choices:
        1: 0.70
        2: 0.30
      order:
        local_shuffle: 0.35
        stratified_shuffle: 0.50
        global_shuffle: 0.15
      phase: assimilation
      visit_kind: shuffled_coverage

    high_block_repair:
      enable: true
      b_choices:
        6: 0.25
        8: 0.35
        10: 0.25
        12: 0.15
      r_choices:
        1: 0.75
        2: 0.25
        3: 0.00
      candidate_policy: visited_preferred
      unvisited_fill: true
      random_order: true
      phase: repair
      visit_kind: repair
      last_update_write: false
```

### 7.2 Curriculum configs

```yaml
scheduler_stage3_2:
  curriculum:
    - name: warmup
      start_step: 0
      end_step: 5000
      sequence_target_frames: 10
      min_frames: 10
      allow_short: false
      weights:
        repeat_refine: 0.35
        shuffled_coverage: 0.55
        high_block_repair: 0.10
      max_k:
        train_2d:
          repeat_refine: 8
          shuffled_coverage: 8
          high_block_repair: 8
        frozen_2d:
          repeat_refine: 12
          shuffled_coverage: 12
          high_block_repair: 12

    - name: main
      start_step: 5000
      end_step: 30000
      sequence_target_frames: 16
      min_frames: 10
      allow_short: true
      weights:
        repeat_refine: 0.20
        shuffled_coverage: 0.55
        high_block_repair: 0.25
      max_k:
        train_2d:
          repeat_refine: 8
          shuffled_coverage: 10
          high_block_repair: 12
        frozen_2d:
          repeat_refine: 12
          shuffled_coverage: 16
          high_block_repair: 16

    - name: hardening
      start_step: 30000
      end_step: 60010
      sequence_target_frames: 24
      min_frames: 8
      allow_short: true
      weights:
        repeat_refine: 0.10
        shuffled_coverage: 0.40
        high_block_repair: 0.50
      max_k:
        train_2d:
          repeat_refine: 8
          shuffled_coverage: 10
          high_block_repair: 12
        frozen_2d:
          repeat_refine: 12
          shuffled_coverage: 16
          high_block_repair: 20
```

### 7.3 Episode recipe configs

```yaml
scheduler_stage3_2:
  episode_recipe:
    prelude:
      mix:
        repeat_refine: true
        shuffled_coverage: true
      order_policy: mixed_random
      min_rollouts: 2
      max_rollouts: 8
      cover_target_ratio: 0.65
    repair_tail:
      enable: true
      min_rollouts: 0
      max_rollouts: 4
      candidate_policy: visited_preferred
      require_prelude_visited: true
    train_2d_policy:
      repeat_refine: trainable
      shuffled_coverage: trainable
      high_block_repair: frozen_no_grad
```

---

## 8. 新增核心数据结构

建议放在：

```text
models/iforward/protocols/distributional_scheduler.py
```

或：

```text
datasets/iforward_stage2_3/distributional_episode.py
```

第一版若主要服务 training scheduler，可放在 dataset scheduler 侧；若要 train/validation/demo 共享，应放在 `models/iforward/protocols/`。

### 8.1 RolloutDistributionSpec

```python
@dataclass(frozen=True)
class RolloutDistributionSpec:
    name: str
    distribution_type: Literal['repeat_refine', 'shuffled_coverage', 'high_block_repair']
    phase: Literal['assimilation', 'repair']
    visit_kind: str
    b_choices: dict[int, float]
    r_choices: dict[int, float] | None = None
    k_budget: dict[int, float] | None = None
    max_b: int = 0
    max_k_train_2d: int = 8
    max_k_frozen_2d: int = 16
    order_weights: dict[str, float] = field(default_factory=dict)
    candidate_policy: str = 'sequential_unvisited'
    train_2d_mode: Literal['trainable', 'frozen_no_grad', 'auto'] = 'trainable'
    last_update_write: bool = True
```

### 8.2 CurriculumPhaseSpec

```python
@dataclass(frozen=True)
class CurriculumPhaseSpec:
    name: str
    start_step: int
    end_step: int
    sequence_target_frames: int
    min_frames: int
    allow_short: bool
    distribution_weights: dict[str, float]
    max_k_train_2d: dict[str, int]
    max_k_frozen_2d: dict[str, int]
```

### 8.3 EpisodeRecipeSpec

```python
@dataclass(frozen=True)
class EpisodeRecipeSpec:
    name: str
    prelude_distribution_names: tuple[str, ...]
    repair_distribution_names: tuple[str, ...]
    prelude_order_policy: Literal['fixed', 'mixed_random', 'stratified'] = 'mixed_random'
    repair_tail_policy: Literal['none', 'append_after_prelude'] = 'append_after_prelude'
    min_prelude_rollouts: int = 2
    max_prelude_rollouts: int = 8
    min_repair_rollouts: int = 0
    max_repair_rollouts: int = 4
    cover_target_ratio: float = 0.65
    repair_candidate_policy: str = 'visited_preferred'
```

### 8.4 DistributionalRolloutSample

Internal object before converting to RolloutPlanV3:

```python
@dataclass(frozen=True)
class DistributionalRolloutSample:
    distribution_type: str
    order_type: str
    phase: str
    visit_kind: str
    positions: tuple[int, ...]
    repeat_budgets: tuple[int, ...]
    requested_b: int
    requested_k: int
    train_2d_mode: str
    episode_stage: Literal['prelude', 'repair_tail']
    candidate_pool: str
    metadata: dict[str, Any]
```

---

## 9. Compiler 设计

新增：

```python
class DistributionalEpisodeCompiler:
    def __init__(self, scheduler: Stage23Scheduler, cfg: Any): ...
    def build_episode(self, *, step: int) -> EpisodePlanV3: ...
```

它复用 Stage23Scheduler 的现有能力：

```text
_sample_sequence_rows()
_rollout_from_positions()
_ref_rows_for_positions()
_make_steps()
```

第一版不要重写 batch resolver。

### 9.1 build_episode 总流程

```python
def build_episode(step):
    phase_cfg = curriculum.phase_for_step(step)
    rows = scheduler._sample_sequence_rows_with_phase_cfg(phase_cfg)
    context = EpisodeBuildContext(rows, visit_counts, last_visit_context, visited_positions)

    prelude_samples = sample_prelude(context, phase_cfg)
    repair_samples = sample_repair_tail(context, phase_cfg, prelude_samples)

    samples = prelude_samples + repair_samples
    rollouts = []
    for sample in samples:
        rollout = scheduler._rollout_from_positions(..., sample.positions, sample.repeat_budgets, ...)
        rollout = attach_distribution_metadata(rollout, sample)
        rollouts.append(rollout)
        context.update_after_rollout(rollout)

    return EpisodePlanV3(... rollouts=tuple(rollouts), metadata=episode_metadata)
```

### 9.2 Context state

```python
class EpisodeBuildContext:
    rows
    order_physical
    visit_counts: dict[int,int]
    visited_positions: set[int]
    prelude_positions: list[int]
    repair_positions: list[int]
    last_visit_step_by_pos: dict[int,int]
    last_visit_context: dict[str,Any]
    step_offset: int
```

The existing scheduler already has `visit_counts`, `last_visit_step_by_pos`, `last_visit_context`; reuse them.

---

## 10. Sampling algorithms

## 10.1 repeat_refine sampling

Inputs:

```text
available positions
visited/unvisited context
maxK
```

Algorithm:

```python
B = sample({1,2})
K = sample(k_budget)
K = min(K, maxK)
B = min(B, K)
positions = choose_repeat_positions(B)
repeat_budgets = random_partition(K, B, min_each=1)
order_type = sample(order_weights)
positions = apply_order(positions, order_type)
```

Candidate preference:

```text
early episode: unvisited preferred
later episode: mix visited and unvisited
```

Reason: repeat refine should learn both first-time current refinement and repeated refinement.

## 10.2 shuffled_coverage sampling

Algorithm:

```python
B = sample({3,4,6,8})
R = sample({1,2})
K = B * R
if K > maxK: downsample B or R
positions = choose_coverage_positions(B, policy='unvisited_preferred')
order_type = sample(local/stratified/global)
positions = apply_order(positions, order_type)
repeat_budgets = [R] * B
```

Candidate preference:

```text
unvisited preferred until episode cover ratio reached;
then allow visited mix.
```

## 10.3 high_block_repair sampling

Algorithm:

```python
B = sample({6,8,10,12,16})
R = sample({1,2,3})
K = B * R
if K > maxK_repair: reduce B/R
candidate_pool = visited_positions excluding current_positions
if len(candidate_pool) < B and unvisited_fill:
    candidate_pool += unvisited_positions
positions = sample_without_replacement(candidate_pool, B)
shuffle(positions)
repeat_budgets = [R] * B
```

Important:

```text
repair_positions in RolloutPlanV3 should reflect all repair tail positions for this episode.
history_positions for each repair rollout should include previously visited positions not in current repair positions.
```

---

## 11. Metadata / schema extension

`RolloutPlanV3` should gain optional fields, or these fields should be placed in `request_meta`. For stronger metrics and serialization, adding dataclass fields is cleaner.

Recommended fields:

```python
distribution_type: str = ''
episode_stage: str = ''          # prelude / repair_tail
order_type: str = ''             # chronological/local_shuffle/stratified_shuffle/global_shuffle
train_2d_mode: str = ''          # trainable/frozen_no_grad/auto
requested_distribution_b: int = 0
requested_distribution_r: int = 0
requested_distribution_k: int = 0
candidate_pool: str = ''         # unvisited_preferred/visited_preferred/mixed
visited_ratio_before: float = 0.0
visited_ratio_after: float = 0.0
repair_visited_ratio: float = 0.0
prelude_rollout_idx: int = -1
repair_tail_idx: int = -1
```

If not adding fields immediately, put into:

```python
rollout.request_meta['iforward_stage3_2'] = {...}
```

But for `metrics_history`, explicit fields are easier.

---

## 12. Integration with Stage3SchedulerAdapter / Runtime

### 12.1 Adapter should remain thin

`Stage3SchedulerAdapter` should not implement distribution logic. It should continue to convert `RolloutPlanV3` to `UpdateEvent`.

Add metadata passthrough:

```text
UpdateEvent.metadata should include distribution_type, episode_stage, order_type, train_2d_mode, B/R/K, visited ratios.
```

### 12.2 EpisodePlan version

Use:

```text
iforward_episode_plan_v2_distributional
```

or keep plan version:

```text
iforward_episode_plan_v1
```

and add:

```python
EpisodePlan.metadata['scheduler_version'] = 'stage3_2_distributional_episode_v1'
```

Recommended first version: keep `EpisodePlan.version` unchanged to avoid breaking replay tests, and put stage3_2 information in metadata.

---

## 13. Training integration

### 13.1 Config selection

In train builder:

```text
if scheduler_stage3_2.enable:
    use Stage23Scheduler with DistributionalEpisodeCompiler
elif scheduler_stage3_0.enable:
    use existing Stage23Scheduler
else:
    fallback old scheduler_v3
```

### 13.2 Minimal invasive path

Option A: subclass or wrapper

```python
class Stage32DistributionalScheduler(Stage23Scheduler):
    def _build_episode(self):
        if self.stage3_2_enabled:
            return self.distributional_compiler.build_episode(step=self.global_step)
        return super()._build_episode()
```

Option B: composition

```python
Stage23Scheduler owns DistributionalEpisodeCompiler;
_build_episode delegates when version startswith stage3_2.
```

Recommended: Option B. It preserves `Stage23Scheduler` traversal, producer, `_batch_from_plan`, state_dict/load_state_dict.

### 13.3 Producer compatibility

Producer/preload sees only `RolloutPlanV3`; no change if rollouts contain evidence_refs and target_refs.

Need ensure episode chain preload includes all prelude + repair tail refs.

---

## 14. 2D train/frozen integration details

First version:

```text
repeat_refine / shuffled_coverage: phase='assimilation', visit_kind='assimilate' or custom, 2D trainable
high_block_repair: phase='repair', visit_kind='repair', 2D frozen/no-grad via existing repair_training
```

If custom visit_kind is used:

```text
repeat_refine -> visit_kind='repeat_refine'
shuffled_coverage -> visit_kind='shuffled_coverage'
high_block_repair -> visit_kind='repair'
```

Then model-side `repair_training.kinds` should remain `['repair']` to only freeze 2D for repair tail.

Second version can add:

```yaml
frontend_training_policy:
  trainable_kinds: [repeat_refine, shuffled_coverage]
  frozen_no_grad_kinds: [repair, shuffled_coverage_frozen]
```

But do not block first version on this.

---

## 15. Validation v4 updates

Validation must mirror the new distributions.

### 15.1 Add protocols

```text
distribution_assimilation_timeline:
  runs a full distributional episode and reports by distribution_type.

shuffle_order_robustness:
  same frame set, same B/R/K, multiple orders.

repair_tail_before_after:
  snapshot before repair tail, probe final_all before and after.

repeat_refine_stability:
  B1/B2 repeat budgets 4/8/16.

memory_ablation_by_distribution:
  full/off/freeze/shuffle for repeat/shuffle/repair separately.

train2d_policy_stress:
  trainable vs frozen_no_grad maxK comparison.
```

### 15.2 Required report groups

Metrics should be grouped by:

```text
frame_set
protocol
distribution_type
episode_stage
order_type
memory_mode
train_2d_mode
B
R
K
```

### 15.3 Order robustness

For order robustness, use at least:

```yaml
repair_permutations: 3
```

Better:

```yaml
order_permutations: 5
```

Current `repair_permutations=1` cannot evaluate order robustness.

---

## 16. Debug demo updates

Add demo recipes:

```text
distributional_episode_showcase
shuffle_vs_chronological_showcase
repair_tail_showcase
memory_ablation_distribution_showcase
```

### 16.1 distributional_episode_showcase

Timeline:

```text
prelude rollout 0: repeat_refine
prelude rollout 1: shuffled_coverage
prelude rollout 2: shuffled_coverage
snapshot before repair
probe final_all before repair
repair rollout 0: high_block_repair
probe final_all after repair
```

### 16.2 Demo report must show

```text
rollout timeline
positions visited
order permutation
current PSNR before/after
history damage before/after
repair gain
GDKV read/write/state/ctx RMS
2D train/frozen state
images before/after repair
```

---

## 17. Metrics and logging

Add scheduler metrics:

```text
iforward/stage3_2/enabled
iforward/stage3_2/curriculum_phase_id
iforward/stage3_2/curriculum_phase_name
iforward/stage3_2/distribution_type_id
iforward/stage3_2/episode_stage_id
iforward/stage3_2/order_type_id
iforward/stage3_2/train_2d_mode_id
iforward/stage3_2/B
iforward/stage3_2/R_mean
iforward/stage3_2/K
iforward/stage3_2/maxK
iforward/stage3_2/visited_ratio_before
iforward/stage3_2/visited_ratio_after
iforward/stage3_2/repair_visited_ratio
iforward/stage3_2/prelude_repeat_count
iforward/stage3_2/prelude_shuffle_count
iforward/stage3_2/repair_tail_count
```

Keep old metrics:

```text
scheduler_phase
rollout_phase
shape_name
blocks_per_rollout
repeats_per_block
actual_inner_K
repair_flag
repair_pattern_name
optimizer_memory_read_count/write_count
observation_commit_count
```

Do not break old dashboard.

---

## 18. Unit tests

Add tests:

```text
tests/test_iforward_stage3_2_distributional_scheduler.py
tests/iforward/runtime/test_distributional_episode_plan.py
tests/iforward/demo/test_distributional_demo_recipes.py
tests/iforward/validation_v4/test_distributional_validation_recipes.py
```

### 18.1 Sampling tests

```text
repeat_refine never B>2.
repeat_refine K <= maxK.
shuffled_coverage B/R/K valid.
high_block_repair B/R/K valid.
repair candidate pool uses visited positions when available.
```

### 18.2 Episode structure tests

```text
Episode contains prelude rollouts before repair tail.
Prelude contains both repeat and shuffle under mixed recipe.
Repair tail rollouts have phase=repair.
2D policy metadata is correct.
```

### 18.3 Serialization tests

```text
EpisodePlan to_json/from_json keeps distribution metadata.
RolloutPlanV3 request_meta keeps stage3_2 block.
Plan replay produces identical events.
```

### 18.4 Runtime tests

```text
IForwardRunner can execute a handcrafted distributional episode in no-grad mode.
TraceRecorder records distribution_type/order_type.
Stage3SchedulerAdapter passes metadata to UpdateEvent.
```

### 18.5 Statistical tests

For 1000 sampled rollouts:

```text
observed distribution weights roughly match config.
order_type ratios roughly match config.
B/R/K constraints always respected.
```

---

## 19. Migration plan

### Phase 0: Read-only config parser

Implement parser and unit tests:

```text
parse scheduler_stage3_2
parse distributions
parse curriculum
validate maxK / B / R constraints
```

No training change.

### Phase 1: DistributionalEpisodeCompiler smoke

Implement compiler that builds `EpisodePlanV3` using existing `_rollout_from_positions()`.

Run tests only.

### Phase 2: Adapter / runtime metadata

Ensure `Stage3SchedulerAdapter` forwards metadata to `UpdateEvent` and TraceRecorder.

### Phase 3: Validation/demo recipes

Add distributional validation/demo recipes before training integration.

Goal: inspect plans and reports without optimizer step.

### Phase 4: Training integration behind flag

Enable:

```yaml
scheduler_stage3_2.enable=true
scheduler_stage3_0.enable=false or as fallback
```

First run:

```text
single scene / fixed segment / 200 steps
```

### Phase 5: Full training config

Add:

```text
configs/iforward/iforward_stage3_2_distributional_episode_gdkv.yaml
```

---

## 20. Acceptance criteria

Implementation is considered complete when:

```text
[ ] Stage3_2 config parses and validates.
[ ] repeat_refine never samples B>2.
[ ] shuffled_coverage appears before 30k and is majority in main phase.
[ ] high_block_repair appears before 30k with low probability and grows later.
[ ] repair tail prioritizes visited positions.
[ ] 2D train/frozen policy controls maxK as configured.
[ ] train/validation/demo all use EpisodePlan.
[ ] validation reports group metrics by distribution_type.
[ ] debug demo can show repeat -> shuffle -> repair timeline.
[ ] metrics_history contains stage3_2 distribution metrics.
[ ] old scheduler_stage3_0 can still run unchanged.
```

---

## 21. Risks and guardrails

### Risk 1: Current PSNR drops early

Cause:

```text
Too much shuffled/high-block training too early.
```

Guardrail:

```text
warmup keep repeat_refine 30-40%;
limit maxK with trainable 2D;
keep shuffled global_shuffle small.
```

### Risk 2: Repair tail becomes random assimilation

Cause:

```text
repair candidates sampled mostly from unvisited frames.
```

Guardrail:

```text
visited_preferred candidate policy;
log repair_visited_ratio;
fail_fast if repair_visited_ratio below threshold after prelude has enough visits.
```

### Risk 3: New fixed protocol overfit

Cause:

```text
Always repeat -> shuffle -> repair in same order.
```

Guardrail:

```text
Prelude mixed_random;
multiple recipe variants in validation;
order robustness protocol.
```

### Risk 4: 2D frozen high-K changes objective too much

Cause:

```text
Large K no-grad repair learns backend-only behavior.
```

Guardrail:

```text
Keep shuffled_coverage trainable in main phase;
only repair tail frozen/no-grad initially;
add train2d_policy_stress validation.
```

---

## 22. Recommended first config

For the first real run, use the balanced version:

```text
warmup 0-5k:
  repeat 35%, shuffle 55%, repair 10%

main 5k-30k:
  repeat 20%, shuffle 55%, repair 25%

hardening 30k+:
  repeat 10%, shuffle 40%, repair 50%
```

Use:

```text
Prelude order: mixed_random
Repair candidate: visited_preferred
Shuffle: stratified > local > global
2D: trainable for repeat/shuffle, frozen/no-grad for repair
```

This is the recommended Scheduler 3_2 baseline.

---

## 23. 后续 GPT 上下文块

```text
IForward Scheduler 3_2 是 distributional episode scheduler。
目标不是再做一个大 scheduler，而是在 v43 runtime 的 EpisodePlan/UpdateEvent/TraceRecorder 基础上，引入三种 rollout distribution：
1) repeat_refine: B<=2, R/K 随机，用于局部迭代/refinement/stability；
2) shuffled_coverage: B=3/4/6/8, R=1/2，顺序 local/stratified/global shuffle，是主训练分布；
3) high_block_repair: B=6/8/10/12/16, R<=2/3，放在 episode tail，优先采 prelude visited positions。
单 episode: prelude(repeat_refine + shuffled_coverage mixed_random) -> repair_tail(high_block_repair)。
2D trainable 与 frozen/no-grad 控制 maxK：trainable K 小，frozen/no-grad K 大。第一版只让 repair phase frozen/no-grad，repeat/shuffle 仍 trainable。
实现上增加 DistributionalEpisodeCompiler，复用 Stage23Scheduler._sample_sequence_rows 和 _rollout_from_positions，输出 RolloutPlanV3/EpisodePlanV3，再由 Stage3SchedulerAdapter 转为 runtime EpisodePlan。
必须记录 distribution_type, episode_stage, order_type, train_2d_mode, B/R/K, visited_ratio, repair_visited_ratio。
Validation/demo 必须使用同一 EpisodePlan，并按 distribution_type 分组报告。
```
