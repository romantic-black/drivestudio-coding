# IForward Runtime / Scheduler / Validation / Demo 重构详细实现方案

日期：2026-07-01  
适用代码：`drivestudio_stage6_refactor_context_20260701_v42`  
适用模型：IForward 3_1 Low-rank Gated Delta KV

---

## 0. 执行结论

当前 3_1 GDKV 模型已经证明：

```text
1. GDKV 替代 Mamba 后，训练稳定性明显改善；
2. 训练已跑到 35k 左右，无 NaN/Inf；
3. 34999 validation 已完成；
4. 30k 后 repair training 明显生效，Repair-B6R1/B8R1/B6R2 从 29999 的 14-16 dB 恢复到 22-23 dB；
5. GDKV full/read_write 明显好于 off/read_only/freeze_write；
6. 当前主要瓶颈不是模型结构，而是 scheduler / validation / demo 的证据链不完整。
```

因此下一步重构不应该继续改模型，而应该建立统一的 IForward runtime：

```text
Protocol Recipe -> EpisodePlan -> IForwardRunner -> TraceRecorder -> ValidationReport / DemoReport / TrainMetrics
```

核心原则：

```text
不做一个更大的 Scheduler v4；
把 scheduler 降级为 plan generator；
让训练、validation、demo 都执行同一种 EpisodePlan；
先做 validate-only / static report，再接训练 scheduler adapter，最后做交互 demo。
```

---

## 1. 当前代码基础与重构边界

### 1.1 当前已有基础

v42 代码里已经存在：

```text
configs/iforward/iforward_stage3_1_lowrank_gated_delta_kv.yaml
models/iforward/stage2_3/parent_optimizer_gated_delta_kv.py
models/iforward/stage2_3/optimizer_memory_schema.py
datasets/iforward_stage2_3/scheduler.py
datasets/iforward_stage2_3/schema.py
datasets/iforward_stage2_3/validation_runner.py
models/iforward/model.py
models/iforward/trainer.py
```

当前 scheduler schema 已经有：

```python
Stage23StepPlan
RolloutPlanV3
EpisodePlanV3
```

并且 `Stage23StepPlan` 已经包含：

```python
optimizer_memory_read: bool
optimizer_memory_write: bool
validation_render_only: bool
scheduler_phase: str
visit_kind: str
repeat_budget: int
```

当前 validation runner 已经能运行：

```text
Assimilation-Causal
Assimilation-Causal-FinalAll
Repair-B6R1
Repair-B8R1
Repair-B6R2
Repair-B10
Repeat Stability
Order Robustness
Mamba Ablation
```

所以第一版 runtime 不需要重写所有计划构造逻辑，而应复用现有 `RolloutPlanV3` / `_batch_from_plan()` / `model.forward_rollout()`。

### 1.2 不在本轮重构中做的事

本轮重构不要做：

```text
不改 GDKV K/V；
不改 RMS max；
不改 distant means / quat；
不把 30k 后训练强行改为纯 repair；
不重写 dataset / asset loader；
不做实时高性能 3DGS viewer；
不删除旧 scheduler；
不强制把当前训练路径迁移到新 runner 的第一版。
```

第一阶段目标是：**证据链闭环**，不是模型收益最大化。

---

## 2. 总体架构

目标目录：

```text
models/iforward/runtime/
  __init__.py
  event.py
  plan.py
  runner.py
  adapter_stage3.py
  trace.py
  state_snapshot.py
  artifact_store.py

models/iforward/protocols/
  __init__.py
  recipes.py
  train_recipes.py
  validation_recipes.py
  demo_recipes.py
  compiler.py

models/iforward/validation_v4/
  __init__.py
  metrics.py
  failure_modes.py
  report.py
  html_exporter.py
  video_exporter.py
  image_grid.py

models/iforward/demo/
  __init__.py
  report_builder.py
  actions.py
  server.py          # P1/P2 才做

tools/
  iforward_validate_v4.py
  iforward_demo_report.py
  iforward_plan_replay.py
```

数据流：

```text
Recipe / Adapter
    -> EpisodePlan
        -> IForwardRunner.run(plan)
            -> model.forward_rollout(batch, carried_state, ablation/memory_mode)
                -> TraceRecorder
                    -> jsonl / images / html / summary table
```

---

## 3. 核心数据结构

### 3.1 EpisodeSpec

文件：`models/iforward/runtime/event.py`

只描述“在哪个 scene/segment/frame set 上执行”，不包含训练或 validation 语义。

```python
@dataclass(frozen=True)
class EpisodeSpec:
    scene_id: int
    segment_id: int
    sequence_id: int
    frame_ids: tuple[int, ...]
    frame_positions: tuple[int, ...]
    cam_ids: tuple[int, ...]
    init_state: Literal['asset_fresh', 'checkpoint', 'snapshot'] = 'asset_fresh'
    seed: int = 0
    protocol_name: str = ''
    episode_uid: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.2 Event 类型

第一版不要做一个万能大 event。拆成三类。

#### UpdateEvent

用于真正更新 LocalGSState / parent_state / GDKV memory。

```python
@dataclass(frozen=True)
class UpdateEvent:
    event_id: str
    kind: Literal['observe_update', 'repair_update']
    rollout_plan: Any              # MVP: RolloutPlanV3
    phase: Literal['assimilation', 'repair']
    input_positions: tuple[int, ...]
    repeat_budgets: tuple[int, ...]
    blocks_per_rollout: int
    repeats_per_block: int
    memory_read: bool = True
    memory_write: bool = True
    observation_commit: bool = True
    parent_state_update: bool = True
    local_state_update: bool = True
    repair_training: bool = False
    memory_mode: str = 'full'
    tag: str = ''
```

MVP 中 `rollout_plan` 直接持有现有 `RolloutPlanV3`，这样可以复用 `scheduler._batch_from_plan()`，避免重写 resolver。

#### ProbeEvent

用于渲染与记录，不更新 state。

```python
@dataclass(frozen=True)
class ProbeEvent:
    event_id: str
    kind: Literal['render_probe']
    target_positions: tuple[int, ...]
    target_frame_ids: tuple[int, ...]
    target_cams: tuple[int, ...]
    roles: tuple[str, ...] = ('current', 'history')
    update_state: bool = False
    compute_loss: bool = False
    tag: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### ControlEvent

用于 reset / snapshot / restore / memory ablation。

```python
@dataclass(frozen=True)
class ControlEvent:
    event_id: str
    kind: Literal['reset_state', 'snapshot_state', 'restore_state', 'set_memory_mode']
    name: str = ''
    memory_mode: str = 'full'
    tag: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
```

第一版暂不做 `fork_state`，用 snapshot/restore 模拟分叉。

### 3.3 MemoryMode

内部命名不再用 mamba。

```python
MemoryMode = Literal[
    'full',
    'memory_off',
    'memory_read_only',
    'memory_read_write',
    'memory_shuffle_state',
    'memory_freeze_write',
]
```

兼容旧名称：

```python
LEGACY_MEMORY_MODE_ALIASES = {
    'mamba_off': 'memory_off',
    'mamba_read_only': 'memory_read_only',
    'mamba_read_write': 'memory_read_write',
    'mamba_shuffle_state': 'memory_shuffle_state',
    'mamba_freeze_write': 'memory_freeze_write',
}
```

### 3.4 EpisodePlan

文件：`models/iforward/runtime/plan.py`

```python
@dataclass(frozen=True)
class EpisodePlan:
    plan_id: str
    version: str
    episode: EpisodeSpec
    events: tuple[UpdateEvent | ProbeEvent | ControlEvent, ...]
    expected_outputs: tuple[str, ...] = ()
    deterministic: bool = True
    source: Literal['scheduler_adapter', 'validation_recipe', 'demo_recipe', 'manual'] = 'manual'
    created_at_step: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> 'EpisodePlan': ...
```

`plan_id` 必须稳定：

```python
plan_id = sha1(json.dumps(plan_without_plan_id, sort_keys=True)).hexdigest()[:16]
```

这样任意 validation/demo 失败都能用 plan_id 复现。

---

## 4. IForwardRunner 设计

文件：`models/iforward/runtime/runner.py`

### 4.1 RunnerOptions

```python
@dataclass
class RunnerOptions:
    mode: Literal['train', 'validate', 'demo', 'replay']
    allow_grad: bool = False
    compute_loss: bool = False
    optimizer_step: bool = False
    update_state: bool = True
    record_images: bool = True
    record_debug_tensors: bool = False
    device: str = 'cuda'
    trigger_step: int = -1
```

默认：

```text
train:    allow_grad=True,  compute_loss=True,  optimizer_step=True,  record_images=False
validate: allow_grad=False, compute_loss=False, optimizer_step=False, record_images=True
demo:     allow_grad=False, compute_loss=False, optimizer_step=False, record_images=True
```

### 4.2 Runner.run

```python
class IForwardRunner:
    def __init__(self, model, scheduler_adapter=None, convert_batch_to_minimal_format=None):
        self.model = model
        self.scheduler_adapter = scheduler_adapter
        self.convert_batch_to_minimal_format = convert_batch_to_minimal_format

    def run(self, plan: EpisodePlan, recorder: TraceRecorder, options: RunnerOptions) -> 'EpisodeTrace':
        state = None
        memory_mode = 'full'
        snapshots = {}
        trace = recorder.begin_plan(plan)

        for idx, event in enumerate(plan.events):
            if isinstance(event, ControlEvent):
                state, memory_mode = self._run_control(event, state, memory_mode, snapshots, recorder)
            elif isinstance(event, UpdateEvent):
                out = self._run_update(event, state, memory_mode, options)
                state = self._next_state(out)
                recorder.record_update(event, out, state)
            elif isinstance(event, ProbeEvent):
                out = self._run_probe(event, state, memory_mode, options)
                recorder.record_probe(event, out, state)
            else:
                raise TypeError(type(event))

        return recorder.end_plan(trace)
```

### 4.3 MVP 的关键选择：rollout-level event

第一版不要把每个 repeat / step 拆成事件。以 `RolloutPlanV3` 为 event 粒度：

```text
一个 UpdateEvent = 一个 RolloutPlanV3 = 调用一次 model.forward_rollout()
```

理由：

```text
1. 当前 model.forward_rollout 已经正确处理 inner_K / repeats / memory read-write；
2. validation_runner 已有 _run_plan() 逻辑；
3. step-level runner 会牵动 model 内部循环，风险太大；
4. demo/validation 第一版只需要 rollout-level trace。
```

后续如果需要逐 repeat 可视化，再在 model 内部增加 per-k trace hook，而不是一开始改 runner 粒度。

### 4.4 _run_update

```python
def _run_update(self, event, carried_state, memory_mode, options):
    raw = self.scheduler_adapter.batch_from_rollout_plan(event.rollout_plan)
    batch = self._convert(raw, options.device, options.trigger_step)
    ablation = self._resolve_ablation(event.memory_mode or memory_mode)
    with torch.set_grad_enabled(options.allow_grad):
        out = self.model.forward_rollout(batch, carried_state=carried_state, ablation=ablation)
    return out
```

这里继续用 `ablation` 参数兼容现有 `model.forward_rollout()`。

### 4.5 _run_probe

MVP 中 probe 也通过一个 `validation_render_only=True` 的 RolloutPlanV3 实现。可以复用 `validation_runner._manual_stage2_3_plan(..., validation_render_only=True)`。

```python
def _run_probe(self, event, carried_state, memory_mode, options):
    plan = self.scheduler_adapter.build_render_probe_plan(event)
    raw = self.scheduler_adapter.batch_from_rollout_plan(plan)
    batch = self._convert(raw, options.device, options.trigger_step)
    with torch.no_grad():
        out = self.model.forward_rollout(batch, carried_state=carried_state, ablation=memory_mode)
    return out
```

---

## 5. Scheduler Adapter

文件：`models/iforward/runtime/adapter_stage3.py`

### 5.1 目标

不是替换 `Stage23Scheduler`，而是包一层：

```python
Stage3SchedulerAdapter.sample_train_plan(step) -> EpisodePlan
Stage3SchedulerAdapter.batch_from_rollout_plan(plan) -> batch
Stage3SchedulerAdapter.build_manual_rollout(...) -> RolloutPlanV3
```

### 5.2 API

```python
class Stage3SchedulerAdapter:
    def __init__(self, scheduler: Stage23Scheduler):
        self.scheduler = scheduler

    def sample_train_plan(self, step: int) -> EpisodePlan:
        batch = self.scheduler.next_batch()
        meta = batch['_iforward']
        episode = self._episode_from_meta(meta)
        rollout_plan = meta.get('rollout_plan') or self._recover_rollout_plan(batch)
        event = self._event_from_rollout_plan(rollout_plan)
        return EpisodePlan(...)

    def plan_from_episode_v3(self, episode_v3: EpisodePlanV3, protocol_name: str) -> EpisodePlan:
        events = [self._event_from_rollout_plan(r) for r in episode_v3.rollouts]
        return EpisodePlan(...)

    def batch_from_rollout_plan(self, rollout_plan: RolloutPlanV3) -> dict:
        return self.scheduler._batch_from_plan(rollout_plan)
```

如果当前 batch 里没有直接带 `rollout_plan`，第一版 validation-only 可直接从 `EpisodePlanV3.rollouts` 建 plan，不强行支持 train batch 转 plan。训练路径放到 Phase 3。

### 5.3 adapter 输出字段

每个 UpdateEvent 的 metadata 至少包含：

```text
scene_id
segment_id
sequence_id
scheduler_phase
rollout_phase
sequence_length
rollout_positions
history_positions
repair_positions
repeat_budgets
optimizer_memory_read_count
optimizer_memory_write_count
observation_commit_count
```

---

## 6. TraceRecorder

文件：`models/iforward/runtime/trace.py`

### 6.1 Trace 数据结构

```python
@dataclass
class EventTrace:
    plan_id: str
    event_id: str
    event_kind: str
    event_idx: int
    protocol: str
    memory_mode: str
    scheduler_phase: str
    rollout_phase: str
    input_positions: list[int]
    history_positions: list[int]
    repair_positions: list[int]
    metrics: dict[str, float]
    state_health: dict[str, float]
    artifacts: dict[str, str]
    metadata: dict[str, Any]
```

```python
@dataclass
class EpisodeTrace:
    plan_id: str
    protocol: str
    scene_id: int
    segment_id: int
    events: list[EventTrace]
    summary: dict[str, float]
```

### 6.2 必须记录的 metrics

基础：

```text
current_psnr
current_l1
history_rollout_psnr
history_l1
loss_current
loss_history
loss_history_damage
```

Memory / GDKV：

```text
parent_optimizer_gdkv/read
parent_optimizer_gdkv/write
bg/distant/rigid_state_rms_mean/max
bg/distant/rigid_ctx_rms_mean/max
bg/distant/rigid_written
```

State health：

```text
delta_norm_bg/distant/rigid
fine_event_norm_bg/distant/rigid
scale_delta_norm_distant
opacity_delta_norm_distant
scale_abnormal_ratio
opacity_abnormal_ratio
nan_count
inf_count
```

Scheduler：

```text
scheduler_phase
rollout_phase
sequence_length
actual_inner_K
blocks_per_rollout
repeats_per_block
repair_flag
memory_read_count
memory_write_count
```

### 6.3 ArtifactStore

文件：`models/iforward/runtime/artifact_store.py`

```python
class ArtifactStore:
    def __init__(self, root: Path): ...
    def save_image(self, name: str, tensor_or_array) -> str: ...
    def save_grid(self, name: str, images: list) -> str: ...
    def save_video(self, name: str, frames: list) -> str: ...
    def save_json(self, name: str, obj: dict) -> str: ...
```

输出目录：

```text
<output_dir>/
  plan.json
  trace.jsonl
  summary.json
  images/
  grids/
  videos/
  index.html
```

---

## 7. State Snapshot

文件：`models/iforward/runtime/state_snapshot.py`

### 7.1 需要 snapshot 的内容

MVP 中 snapshot 只做 in-memory：

```python
@dataclass
class RuntimeSnapshot:
    name: str
    carried_state: Any
    metadata: dict[str, Any]
```

通过：

```python
snapshot = carried_state.detach_for_next_rollout()
```

不要第一版实现磁盘序列化。因为 LocalGSState / BigGS parent state / GDKV state 的完整保存可能牵涉大量 tensor 和设备迁移。

### 7.2 后续持久化

第二版再做：

```python
save_snapshot(snapshot, path)
load_snapshot(path)
```

持久化内容：

```text
LocalGSState
BigGS parent state
GDKV memory state
history bank
visited frame metadata
```

---

## 8. Validation v4 设计

文件：`models/iforward/validation_v4/`

### 8.1 配置

新增 config block：

```yaml
iforward_validation_v4:
  enable: true
  interval_steps: 5000
  run_at_train_start: false
  max_entries_debug: 2
  seed: 20260701
  frame_sets:
    - name: seq10
      target_frames: 10
      min_frames: 10
    - name: seq24
      target_frames: 24
      min_frames: 8
  repair_permutations: 3
  repeat_stability: [8, 16]
  memory_ablation:
    - full
    - memory_off
    - memory_read_write
    - memory_freeze_write
    - memory_shuffle_state
  protocols:
    assimilation_timeline: true
    history_retention: true
    repair_before_after: true
    order_robustness: true
    repeat_stability: true
    memory_ablation: true
    state_health: true
  report:
    html: true
    images: true
    videos: false
```

保留旧 `scheduler_stage3_0_validation` 作为 fallback，不强制删除。

### 8.2 Failure-mode driven protocols

#### A. Assimilation Timeline 10/24

Plan：

```text
reset_state
observe_update frame 0
render_probe current/history
observe_update frame 1
render_probe current/history
...
```

MVP 可用 rollout-level 近似：每个 rollout 后 probe。

Metrics：

```text
current_after_psnr
current_gain
history_auc
history_worst
retention_curve
```

#### B. History Retention Heatmap

输出矩阵：

```text
rows = history frames
cols = update events
value = PSNR / drop
```

Artifacts：

```text
history_retention_heatmap.png
history_retention.csv
```

#### C. Repair Before/After 10/24

Plan：

```text
reset_state
assimilation sequence
snapshot before_repair
render_probe history refs -> before
repair_update shuffled positions
render_probe same refs -> after
compare before/after
```

Metrics：

```text
repair_gain_mean = after_psnr - before_psnr
repair_gain_worst
repair_worse_ratio
repair_after_psnr
```

#### D. Order Robustness

Run same repair set with `P=3/5` permutations.

Metrics：

```text
permutation_mean
permutation_std
permutation_min
permutation_worst_drop
```

当前 3_1 的 `repair_permutations=1` 不足以判断 order robustness，因此 v4 必须默认为 3。

#### E. Repeat Stability

Plan：

```text
snapshot same pre-update state
restore -> repeat 8
restore -> repeat 16
render same refs
```

Metrics：

```text
repeat_R16_drop = psnr_R16 - psnr_R8
state_delta_growth
history_drop
```

#### F. Memory Ablation

统一改名：Memory Ablation。

Modes：

```text
full
memory_off
memory_read_write
memory_freeze_write
memory_shuffle_state
```

Metrics：

```text
memory_gain_retention = retention_full - retention_off
memory_gain_history = history_full - history_off
memory_shuffle_gap = retention_full - retention_shuffle
```

#### G. State Health

Metrics：

```text
scale_abnormal_ratio
opacity_abnormal_ratio
mean_delta_p95
scale_delta_p95
distant_scale_delta_p95
gdkv_state_rms_max
gdkv_ctx_rms_max
nan_or_inf_count
```

---

## 9. HTML Report 设计

文件：`models/iforward/validation_v4/html_exporter.py`

### 9.1 输出结构

```text
report_root/
  index.html
  summary.json
  plan.json
  trace.jsonl
  images/
  grids/
  videos/
  plots/
```

### 9.2 页面结构

`index.html` 包含：

```text
1. Run Summary
   checkpoint / step / config / plan_id / scene / segment

2. Traffic-light status
   current update: pass/warn/fail
   history retention: pass/warn/fail
   repair: pass/warn/fail
   order robustness: pass/warn/fail
   memory ablation: pass/warn/fail
   state health: pass/warn/fail

3. Metrics Table
   protocol-level summary

4. Assimilation Timeline
   before / after / GT / error grid

5. History Retention Heatmap

6. Repair Before/After
   before / after / delta error

7. Repeat Stability

8. Memory Ablation

9. State Health
   branch-wise charts

10. Raw Trace Links
```

### 9.3 不依赖 TensorBoard

HTML 报告必须可以单独打开，不依赖 TensorBoard。TensorBoard 仍可作为训练过程记录，但不是 demo/validation 主要载体。

---

## 10. validate_only 工具

文件：`tools/iforward_validate_v4.py`

### 10.1 CLI

```bash
python -m tools.iforward_validate_v4 \
  --config_file configs/iforward/iforward_stage3_1_lowrank_gated_delta_kv.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --output_dir /root/autodl-tmp/outputs/val_v4_step34999 \
  --scene_ids 0,1 \
  --max_entries 2 \
  --frame_sets seq10,seq24 \
  --repair_permutations 3 \
  --memory_ablation full,memory_off,memory_read_write,memory_freeze_write,memory_shuffle_state
```

### 10.2 实现步骤

```python
def main():
    cfg = load_config(...)
    dataset = build_dataset(cfg)
    model = build_model(cfg)
    load_checkpoint(model, checkpoint)
    scheduler = Stage23Scheduler(..., producer_enable=False)
    adapter = Stage3SchedulerAdapter(scheduler)
    recipes = build_validation_recipes(cfg.iforward_validation_v4)
    runner = IForwardRunner(model, adapter, convert_batch_to_minimal_format)
    recorder = TraceRecorder(output_dir)
    for plan in recipes.build_plans(dataset, adapter):
        runner.run(plan, recorder, RunnerOptions(mode='validate'))
    export_html(recorder.finalize())
```

### 10.3 第一版可复用 train_iforward 的构建逻辑

不要复制太多 build_model/build_dataset 代码。建议从 `tools/train_iforward.py` 或 common helper 中抽函数：

```python
build_iforward_runtime_from_cfg(cfg, checkpoint=None) -> RuntimeBundle
```

返回：

```python
@dataclass
class RuntimeBundle:
    cfg: Any
    dataset: Any
    model: Any
    scheduler: Stage23Scheduler
    convert_batch_to_minimal_format: Callable
    device: torch.device
```

---

## 11. Demo v0：静态报告

文件：`tools/iforward_demo_report.py`

### 11.1 目标

不做交互 viewer，先做“可解释状态变化”的静态 demo。

输入：

```bash
python -m tools.iforward_demo_report \
  --config_file ... \
  --checkpoint ... \
  --scene_id 0 \
  --segment_id 12 \
  --recipe repair_showcase_24 \
  --output_dir demo_report_scene0_segment12
```

输出：

```text
index.html
assimilation_timeline.mp4 或 image grid
history_retention_heatmap.png
repair_before_after_grid.png
repeat_stability_grid.png
memory_ablation_grid.png
state_health.json
trace.jsonl
plan.json
```

### 11.2 Demo recipes

`models/iforward/protocols/demo_recipes.py`

```python
repair_showcase_10
repair_showcase_24
memory_ablation_showcase
repeat_stability_showcase
history_failure_search
```

第一版只做：

```text
repair_showcase_24
memory_ablation_showcase
```

### 11.3 报告必须回答的问题

```text
当前帧有没有变好？
历史帧有没有坏？
repair 有没有修？
GDKV memory 到底贡献多少？
repeat 多了会不会崩？
distant scale 是否正常更新？
```

---

## 12. Interactive Demo v1 / v2

这不是当前第一步，但需要在架构中预留。

### 12.1 UI 分区

```text
左：3DGS / camera trajectory / parent nodes
中：GT / Pred / Error / Before-After image panel
右：IForward control panel
下：timeline + metrics curves + trace table
```

### 12.2 控制面板

```text
Load scene / segment
Reset state
Select frame/camera
Observe once
Repeat K
Batch assimilate selected frames
Repair selected history frames
Snapshot / restore
Memory mode: full/off/read_write/freeze_write/shuffle
Export HTML report
```

### 12.3 技术路线

```text
v1: Python server-side render + simple web UI
v2: Viser / Nerfstudio-style websocket viewer
v3: 若真的需要高 FPS，再考虑 SIBR / OpenGL / WebGPU
```

不要第一版追求实时自由漫游。IForward demo 的核心是状态变化可解释，而不是 viewer FPS。

---

## 13. 训练路径接入

训练接入放在 Phase 3，不是第一步。

### 13.1 Stage3SchedulerAdapter train mode

现有训练仍可以使用：

```python
batch = scheduler.next_batch()
out = model.forward_rollout(batch, carried_state=...)
```

Phase 3 后改成：

```python
plan = scheduler_adapter.sample_train_plan(step)
trace = runner.run(plan, mode='train')
loss = trace.loss
```

但是迁移时必须逐步校验：

```text
同 seed 下 scheduler_phase / rollout_phase 分布一致；
sequence_length 分布一致；
repair ratio 一致；
current/history loss 接近；
train throughput 不下降超过 10%。
```

### 13.2 保守迁移策略

先做 parallel trace：

```text
旧训练路径继续负责 forward/backward；
adapter 只把 batch meta 转成 EpisodePlan 并写 plan.json / trace.jsonl。
```

等验证一致后，才让 runner 接管 train forward。

---

## 14. 配置迁移

新增：

```yaml
iforward_runtime:
  enable: true
  plan_version: iforward_episode_plan_v1
  event_granularity: rollout
  legacy_mamba_alias: true
  save_plan_json: true
  save_trace_jsonl: true

iforward_validation_v4:
  enable: false
  run_at_train_start: false
  interval_steps: 5000
  max_entries_debug: 2
  frame_sets:
    - seq10
    - seq24
  repair_permutations: 3
  repeat_stability: [8, 16]
  memory_ablation:
    - full
    - memory_off
    - memory_read_write
    - memory_freeze_write
    - memory_shuffle_state
  report:
    html: true
    images: true
    videos: false

iforward_demo:
  enable: false
  default_recipe: repair_showcase_24
  output_dir: ''
```

旧配置继续保留：

```yaml
scheduler_stage3_0_validation:
  enable: true
```

迁移期不要同时自动跑两个 validation，除非手动指定。

---

## 15. 测试计划

### 15.1 Unit tests

新增：

```text
tests/iforward/runtime/test_plan_serialization.py
tests/iforward/runtime/test_memory_mode_alias.py
tests/iforward/runtime/test_trace_recorder.py
tests/iforward/runtime/test_stage3_adapter.py
tests/iforward/validation_v4/test_metrics.py
```

测试内容：

```text
EpisodePlan to_json/from_json roundtrip
plan_id stable
mamba_* alias -> memory_* alias
TraceRecorder writes valid jsonl
Stage3SchedulerAdapter converts EpisodePlanV3 -> EpisodePlan
retention/repair/memory metrics formula correct
```

### 15.2 Smoke tests

```bash
python -m tools.iforward_validate_v4 --max_entries 1 --frame_sets seq10 --protocols assimilation_timeline
python -m tools.iforward_validate_v4 --max_entries 1 --frame_sets seq24 --protocols repair_before_after --repair_permutations 3
python -m tools.iforward_demo_report --recipe repair_showcase_24 --scene_id 0 --segment_id <id>
```

通过标准：

```text
生成 index.html；
trace.jsonl 非空；
plan.json 可 replay；
图片 artifact 可打开；
metrics 中包含 current/history/repair/memory/state_health；
无 state 更新时 probe 不改变 carried_state。
```

### 15.3 Regression tests

对比旧 validation runner：

```text
Assimilation-Causal seq10 v4 retention_auc 与旧 runner 差异 < 0.05 dB；
Repair-B6R1 seq10 v4 repair_mean 与旧 runner 差异 < 0.05 dB；
Memory full/off 差值方向一致。
```

允许因为 image selection / probe timing 有小差异，但必须解释。

---

## 16. 分阶段实施路线

### Phase 0：当前状态固化

产物：

```text
34999 validation summary json
当前 v42 config copy
当前 metrics snapshot
```

目的：后续重构不能让 baseline 丢失。

验收：

```text
能复述当前关键指标：repair 22-23 dB，GDKV full-off retention gap ~1.0 dB。
```

### Phase 1：TraceRecorder + validate_only_v4 skeleton

实现：

```text
runtime/event.py
runtime/plan.py
runtime/trace.py
runtime/artifact_store.py
tools/iforward_validate_v4.py skeleton
```

只支持：

```text
读取旧 validation runner 输出 rows -> 生成 HTML-lite
```

验收：

```text
不跑新 runner，也能把 metrics_history 中 34999 生成可读 HTML report。
```

这是最快有用的一步。

### Phase 2：EpisodePlan + Runner MVP

实现：

```text
runtime/runner.py
runtime/adapter_stage3.py
ControlEvent reset/snapshot/restore
UpdateEvent observe/repair via RolloutPlanV3
ProbeEvent via validation_render_only plan
```

验收：

```text
validate_only_v4 能直接调用 model.forward_rollout；
能复现 seq10 Assimilation-Causal 和 Repair-B6R1。
```

### Phase 3：Validation v4 protocols

实现：

```text
validation_recipes.py
validation_v4/metrics.py
validation_v4/html_exporter.py
```

支持：

```text
seq10 + seq24
repair_permutations=3
memory_ablation renamed
state_health
```

验收：

```text
生成完整 HTML；
max_entries=2 能跑；
Order Robustness 有非零 std；
24-frame repair protocol 可运行。
```

### Phase 4：Demo v0 static report

实现：

```text
demo/report_builder.py
tools/iforward_demo_report.py
```

验收：

```text
给定 checkpoint + scene/segment，生成 repair_showcase_24 HTML；
能看 before/after、history heatmap、memory ablation。
```

### Phase 5：Scheduler train adapter

实现：

```text
Stage3SchedulerAdapter.sample_train_plan
train_iforward parallel trace mode
```

验收：

```text
训练行为不变；
每个 train_step 可保存 plan_id；
失败时可用 plan replay。
```

### Phase 6：Interactive demo v1

实现：

```text
demo/server.py
demo/actions.py
```

验收：

```text
能交互执行 observe/repair/snapshot/restore；
能导出当前 HTML report。
```

---

## 17. 优先级清单

### P0 必做

```text
TraceRecorder
validate_only_v4
HTML report
seq10 + seq24 validation
repair_permutations=3
memory_ablation rename alias
state_health metrics
```

### P1 应做

```text
EpisodePlan serialization
Runner MVP
Stage3SchedulerAdapter validation mode
Demo v0 static report
plan replay
```

### P2 后做

```text
train runner 接管
interactive demo server
state snapshot disk serialization
fork_state primitive
large validation
```

---

## 18. 风险与防护

| 风险 | 说明 | 防护 |
|---|---|---|
| 重构改变训练分布 | scheduler 改动可能让 3_1 正向结果不可比 | 先 adapter，parallel trace，不接管训练 |
| runner 粒度太细 | step-level event 会牵动 model.forward_rollout 内部 | MVP 使用 rollout-level event |
| validation 太大 | max_entries / permutations 太多会拖慢训练 | Debug v4 默认 max_entries=2, permutations=3 |
| demo 变成 viewer 工程 | 过早做 3D viewer 会拖慢科研判断 | 先做 static HTML/MP4 |
| mamba 命名污染 | 3_1 已经不是 Mamba | 内部统一 memory_*，外部兼容 legacy |
| plan 无法复现 | 随机 seed / asset / frame set 没保存 | plan.json 必须包含 seed、scene、segment、positions、protocol |
| State clone 太重 | snapshot 深拷贝 tensor 显存大 | MVP 只在 validation/demo 小规模使用 |

---

## 19. 完成标准

第一轮重构完成标准：

```text
[ ] tools/iforward_validate_v4.py 可运行
[ ] 能生成 HTML report
[ ] report 覆盖 seq10 / seq24
[ ] repair_permutations=3 生效
[ ] memory_ablation 使用 memory_* 新命名
[ ] 可输出 state_health
[ ] plan.json 和 trace.jsonl 可保存
[ ] 同一 plan 可 replay
[ ] 旧训练路径不受影响
[ ] 34999 checkpoint 可跑出与旧 validation 同方向的结果
```

---

## 20. 后续 GPT 上下文块

```text
当前项目：StreetForward / IForward 3_1。
当前模型：Low-rank Gated Delta KV，K=16,V=32，parent_optimizer_mamba disabled，parent_optimizer_memory enabled。
当前训练状态：34999 validation 已出现，repair validation 从 29999 的 14-16 dB 恢复到 22-23 dB，GDKV full/read_write 比 off/read_only/freeze_write retention 高约 1 dB。
下一步不是改模型，而是重构 scheduler/validation/demo 的证据链。
总体架构：Protocol Recipe -> EpisodePlan -> IForwardRunner -> TraceRecorder -> ValidationReport / DemoReport。
第一版不要重写训练 scheduler；先做 validate_only_v4 + TraceRecorder + HTML report。
Event 粒度第一版是 rollout-level，即一个 UpdateEvent 对应现有 RolloutPlanV3 并调用一次 model.forward_rollout。
必须新增 seq24 validation 和 repair_permutations=3。
内部命名从 mamba_ablation 改为 memory_ablation，但保留旧 alias。
Demo 第一版做 static HTML/MP4，不做实时 3D viewer。
```
