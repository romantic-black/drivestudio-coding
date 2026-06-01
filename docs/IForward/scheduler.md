# IForward Scheduler 详细设计方案

版本：IForward Scheduler V1  
目标模型：IForward — Short-Sequence Mamba Iterative Optimizer  
依赖数据集：`datasets/multi_scene_dataset_v4.py` 与 StreetForward 预处理资产

---

## 0. 设计结论

IForward scheduler 应该是全新的 scheduler，不继承 `stage6_0`、`phase_a`、`phase_b`、`TrainSchedulerV9` 或 `TrainSchedulerLongPhaseB` 的语义。它可以借鉴现有 keyframe / block / episode / segment / scene 的组织方式，但应该重新定义 rollout 作为 IForward 的最小训练 batch。

核心训练单位从原来的单 block 或 phase-B rollout 改成：

```text
scene
  └── segment
        └── episode                 # 场景状态 reset 的边界
              └── rollout           # 短序列；一次 optimizer step / backward 的边界
                    └── block       # 一个输入帧 / 一个 keyframe block
                          └── repeat / iter step
```

关键规则：

1. **rollout 是短序列**。一个 rollout 包含多个输入帧，输入帧按 scheduler 给定顺序交给模型。
2. **rollout 内不断梯度**。模型在同一个 rollout 内连续处理多个 frame 和 repeat step，中间不 detach，不 backward。
3. **rollout 完成后统一监督和反传**。final loss 使用该 rollout 内所有输入帧作为 current reconstruction supervision；nearby 使用短序列内非输入帧随机采样作为额外 supervision，无候选则 skip。
4. **rollout 完成后不 reset scene**。IForward 的 GS state 和 Mamba memory state 持续进入下一个 rollout。
5. **episode 结束才 reset scene**。episode 是场景状态生命周期边界。
6. **scheduler 只负责角色、顺序、reset/carry 信号和资产加载请求**，不管理实际 3DGS tensor 或 Mamba memory tensor。

---

## 1. 与 `MultiSceneDatasetV4` 的关系

### 1.1 必须继续使用 V4 资产系统

IForward scheduler 不重新实现数据加载，不重新读取原始 dataset，也不重新构建 pointcloud / dynamic track / KNN。它必须通过 `MultiSceneDatasetV4` 读取已经预处理好的 StreetForward assets。

`MultiSceneDatasetV4` 已经提供 IForward 需要的基础：

```text
SegmentIndexV4
├── scene_id
├── segment_id
├── num_cams
├── frame_indices
├── test_frame_indices
├── train_frame_set
├── test_frame_set
├── keyframe_indices
├── keyframe_to_frames
├── frame_to_keyframe
├── segment_first_frame_idx
├── train_image_refs
└── test_image_refs
```

以及通用 batch 组装能力：

```text
_assemble_segment_batch_from_image_refs(
    scene_id,
    segment_id,
    source_image_refs,
    target_image_refs,
    aux_image_refs,
    query_label_image_refs,
    include_test,
    test_image_refs,
    enforce_target0_equals_source,
    target_ref_purpose,
)
```

IForward 不应该绕过这个接口，因为 V4 组装 batch 时还会统一处理：

```text
image / depth / masks
intrinsics / extrinsics
world_to_seg0 transform
segment aabb
pointcloud
rigid dynamic_info
knn_init / knn_struct_neighbors
view pack cache
asset consistency validation
```

### 1.2 对 V4 的最小扩展

建议在 `datasets/multi_scene_dataset_v4.py` 中新增两个轻量入口：

```python
def create_train_scheduler_iforward(... ) -> TrainSchedulerIForward:
    from datasets.train_scheduler_iforward import TrainSchedulerIForward
    return TrainSchedulerIForward(dataset=self, ...)


def _assemble_segment_batch_from_iforward_request(
    self,
    *,
    scene_id: int,
    segment_id: int,
    plan: IForwardRolloutPlan,
    include_test: bool = False,
) -> Dict[str, Any]:
    ...
```

注意：

```text
TrainSchedulerIForward 独立放在 datasets/train_scheduler_iforward.py
MultiSceneDatasetV4 只增加 materializer，不承载 scheduler 状态机
IForward plan dataclass 可以在 train_scheduler_iforward.py 中定义
V4 侧只接受 plan 并按 refs 组装 batch
```

---

## 2. Scheduler 文件与命名

建议新增：

```text
datasets/train_scheduler_iforward.py
```

不要命名为：

```text
train_scheduler_v10.py
train_scheduler_stage6_1.py
train_scheduler_phase_b.py
```

原因：IForward 是新模型，不是 stage6_0 的后续 phase。

建议配置入口：

```yaml
scheduler_iforward:
  enable: true
  version: iforward_v1
```

建议 batch 内标识：

```text
batch["_iforward"]
batch["request_meta"]["scheduler_version"] = "iforward_v1"
batch["request_meta"]["model_family"] = "IForward"
```

不要复用：

```text
_scheduler_v9
phase_B
vsm_reset_policy
prefix_loss
query_label
```

---

## 3. 核心数据结构

### 3.1 ImageRef

```python
ImageRef = Tuple[int, int]  # (frame_idx, cam_idx)
```

---

### 3.2 RolloutShape

一个 shape 描述 rollout 内有多少输入帧，每个输入帧做多少次迭代。

```python
@dataclass(frozen=True)
class IForwardRolloutShape:
    name: str
    blocks_per_rollout: int       # rollout 内输入帧数
    repeats_per_block: int        # 每个输入帧 repeat / iterative steps
    prob: float = 1.0

    @property
    def inner_K(self) -> int:
        return self.blocks_per_rollout * self.repeats_per_block
```

示例：

```yaml
rollout:
  shapes:
    - {name: b2_r8, blocks_per_rollout: 2, repeats_per_block: 8, prob: 0.40}
    - {name: b3_r6, blocks_per_rollout: 3, repeats_per_block: 6, prob: 0.35}
    - {name: b4_r4, blocks_per_rollout: 4, repeats_per_block: 4, prob: 0.25}
```

---

### 3.3 StepPlan

StepPlan 是模型实际执行的最小调度单元。

```python
@dataclass(frozen=True)
class IForwardStepPlan:
    step_idx: int

    # episode / rollout location
    episode_block_idx: int
    rollout_block_rank: int
    repeat_idx: int

    # source frame
    source_keyframe_idx: int
    source_frame_idx: int

    # evidence views used by observation backbone
    evidence_refs: List[ImageRef]
    evidence_frame_indices: List[int]
    evidence_cam_indices: List[int]

    # memory / graph behavior
    commit_observation_memory: bool
    update_optimizer_memory: bool
    detach_before_step: bool
    detach_after_step: bool

    # loss behavior
    allow_step_render_loss: bool
    step_loss_refs: List[ImageRef]

    # useful normalized codes for model
    rollout_pos_code: float       # step_idx / (inner_K - 1)
    frame_pos_code: float         # rollout_block_rank / (num_input_blocks - 1)
    repeat_pos_code: float        # repeat_idx / (repeats_per_block - 1)
```

P0 规则：

```text
commit_observation_memory = True  only when repeat_idx == 0
update_optimizer_memory   = True  for every repeat
.detach_before_step       = False inside rollout
.detach_after_step        = False inside rollout
allow_step_render_loss    = False by default
step_loss_refs            = [] by default
```

解释：

- 对同一个输入帧 repeat 多次时，只有第一次 repeat 代表新观测写入 observation memory。
- 后续 repeat 是 optimizer refinement，不能伪装成新 frame evidence。
- rollout 内不 detach，保证短序列优化链路可反传。
- final loss 在 rollout 完成后统一计算。

---

### 3.4 FinalSupervisionPlan

```python
@dataclass(frozen=True)
class IForwardFinalSupervisionPlan:
    refs: List[ImageRef]
    roles: List[str]

    current_input_frames: List[int]
    nearby_frames: List[int]
    skipped_nearby: bool
    nearby_skip_reason: str

    current_ref_count: int
    nearby_ref_count: int
```

P0 roles：

```text
final_current_recon
final_nearby_rollout
```

含义：

```text
final_current_recon:
    rollout 内所有输入帧，默认 all cams，用于监督 IForward 吸收当前短序列观测。

final_nearby_rollout:
    rollout 短序列范围内的非输入帧随机采样，默认 all cams。
    这些帧不进入 evidence，不写 memory，只参与 final render loss。
```

---

### 3.5 RolloutPlan

```python
@dataclass(frozen=True)
class IForwardRolloutPlan:
    scheduler_version: str

    scene_id: int
    segment_id: int
    episode_id: int
    rollout_id_global: int
    rollout_idx_in_episode: int

    # episode structure
    episode_start_keyframe_pos: int
    keyframe_window: List[int]
    frame_chain: List[int]
    num_cams: int

    # rollout structure
    shape_name: str
    blocks_per_rollout: int
    repeats_per_block: int
    requested_blocks_per_rollout: int
    actual_blocks_per_rollout: int
    requested_inner_K: int
    actual_inner_K: int
    short_rollout: bool
    short_rollout_reason: str

    # selected input sequence
    episode_block_indices: List[int]
    input_keyframe_indices: List[int]
    input_frame_indices: List[int]
    delivery_frame_indices: List[int]
    delivery_order_policy: str

    # execution
    inner_K: int
    steps: List[IForwardStepPlan]

    # final supervision
    final_supervision: IForwardFinalSupervisionPlan

    # state lifecycle
    reset_scene_state_before_rollout: bool
    carry_scene_state_after_rollout: bool
    episode_end_after_rollout: bool
    detach_graph_after_rollout: bool

    # flattened refs for materialization
    evidence_refs_flat: List[ImageRef]
    target_refs_flat: List[ImageRef]
    target_roles_flat: List[str]

    # diagnostics / validation
    request_meta: Dict[str, Any]
    leakage_check: Dict[str, Any]
```

关键约定：

```text
scheduler global_step == rollout step count
inner_K == len(steps) == actual_blocks_per_rollout * repeats_per_block
```

---

## 4. 训练层级与状态机

### 4.1 Scene / Segment

scene 与 segment 来自 V4 asset registry。

scheduler epoch plan：

```text
for scene_id in ordered_scene_ids:
    for segment_id in ordered_segment_ids(scene_id):
        sidx = dataset.get_segment_index(scene_id, segment_id)
        build episode windows from sidx.keyframe_indices
```

推荐 P0 traversal：

```text
scene_order: shuffle_per_epoch
segment_order: shuffle_per_epoch or ascending
traversal_mode: episode_serial
```

不建议 P0 做 episode interleave。原因是 IForward 的状态包含 3DGS + Mamba memory，跨多个 episode 交错训练需要维护多个大状态缓存，复杂且容易出错。P0 应该一次只执行一个 episode 的所有 rollouts。

---

### 4.2 Episode

episode 是 scene state 的 reset 边界。

一个 episode 包含一个 keyframe window：

```text
keyframe_window = keyframe_indices[start : start + blocks_per_episode]
```

每个 keyframe block 选一个输入 frame，形成：

```text
frame_chain = [f0, f1, f2, ..., fE-1]
```

推荐 source frame 策略：

```yaml
episode:
  source_mode: keyframes
  blocks_per_episode: 8
  episode_stride: 8
  allow_short_last_episode: true
  block_source_frame_policy: random_within_keyframe_once_per_episode
```

`block_source_frame_policy` 可选：

```text
middle_in_keyframe_once_per_episode
random_within_keyframe_once_per_episode
random_within_keyframe_per_rollout_visit
```

P0 推荐：

```text
random_within_keyframe_once_per_episode
```

原因：同一个 episode 内每个 block 的输入 frame 稳定，方便跨 rollout carry state。

Episode 生命周期：

```text
episode begin:
    scheduler emits reset_scene_state_before_rollout=True on first rollout
    trainer initializes GS state and Mamba memory from segment assets / default memory

rollout 0:
    process short sequence
    backward
    keep scene state values
    detach graph

rollout 1:
    continue from previous GS + memory
    backward
    keep scene state values
    detach graph

...

episode end:
    scheduler emits episode_end_after_rollout=True
    trainer discards scene state / memory cache after optimizer step
```

---

### 4.3 Rollout

rollout 是 IForward 的一个 training batch。

一个 rollout 选择 episode 内连续或采样的若干 block：

```text
rollout_block_indices = [b0, b1, ..., bR-1]
input_frame_indices = [frame_chain[b] for b in rollout_block_indices]
```

P0 推荐使用 contiguous chronological chunks：

```yaml
rollout:
  block_selection_policy: next_contiguous
  delivery_order_policy: chronological
  allow_short_final_rollout: true
```

之后可以扩展：

```text
random_future_contiguous
random_without_replacement_then_chronological
reverse
random_delivery_order
curriculum_mixed_order
```

但 P0 不建议一开始做 random order，因为需要先验证 persistent optimizer 的基本稳定性。

---

### 4.4 Block

在 IForward scheduler 中，一个 block 表示一个输入 frame，而不是一次 optimizer step。

```text
block b_i
    source_keyframe_idx = keyframe_window[b_i]
    source_frame_idx    = frame_chain[b_i]
```

每个 block 会展开成多个 repeat step：

```text
for repeat_idx in range(repeats_per_block):
    emit IForwardStepPlan(...)
```

---

### 4.5 Repeat / Iter step

repeat 是模型的 RAFT-style iterative refinement step。

对于一个输入 frame：

```text
repeat 0:
    observe current GS using this frame evidence
    commit observation memory
    update optimizer memory
    predict delta
    apply delta

repeat 1..R-1:
    observe current GS again using same frame evidence
    do not commit new observation memory
    update optimizer memory as refinement trajectory
    predict delta
    apply delta
```

注意：

```text
repeat 不是新的 frame
repeat 不是新的 independent observation
repeat 不应该重复污染 long memory
```

---

## 5. Rollout 内训练策略

### 5.1 梯度策略

用户确定的基础策略：

```text
rollout 中间不断梯度
rollout 完成后反传
rollout 完成后不 reset 场景
episode 结束才 reset 场景
```

落地为：

```text
inside rollout:
    no detach
    no optimizer.step
    no state reset

at rollout boundary:
    compute final loss over rollout supervision refs
    backward
    optimizer.step
    commit final GS/memory values for next rollout
    detach graph from carried state

at episode boundary:
    discard carried GS/memory state
```

关键点：

```text
跨 rollout carry state，但不跨 rollout 保留 autograd graph
```

否则长 episode 会造成显存无限增长。也就是说：

```text
state values persist across rollouts
computational graph stops at rollout boundary
scene reset waits until episode end
```

---

### 5.2 Final-only loss

P0 loss timing：

```yaml
loss_timing:
  policy: rollout_final_only
  intermediate_step_loss: false
```

即模型完成整个短序列后，用 final GS state 渲染：

```text
所有 input frames
+ rollout-local nearby frames
```

然后统一反传。

这与 phase A 的 per-iteration loss 不同。IForward 的 P0 目标是让模型学习一个短序列 persistent optimizer，而不是每个 frame / repeat 后立即被单步 loss 拉住。

后续可以加：

```text
intermediate_current_loss
after_each_block_loss
iter_decay_loss
```

但 P0 先不做。

---

## 6. Evidence 采样

### 6.1 Evidence frame

每个 rollout input frame 都是 evidence frame。

```text
evidence input frames = delivery_frame_indices
```

这些 frame 会依次送给模型的 observation backbone。

---

### 6.2 Evidence camera policy

建议配置：

```yaml
evidence:
  camera_policy: all_cams
  cams_per_input: null
  fixed_cam_ids: null
  allow_camera_dropout: false
```

P0 推荐：

```text
all_cams
```

如果 dataset 是 3cam，则每个输入 frame 使用 3cam。之后可以 curriculum 到：

```text
random_subset
fixed_subset
round_robin_subset
1cam / 2cam evidence + heldout cam supervision
```

---

### 6.3 Evidence refs

对于一个输入 frame `f`：

```python
evidence_refs(f) = [(f, cam_id) for cam_id in selected_evidence_cams]
```

每个 StepPlan 都有自己的 `evidence_refs`，但 batch materialization 会将所有 rollout evidence refs flatten + dedupe 后放进：

```text
batch["source"]
```

同时保存 mapping：

```text
request_meta["iforward"]["source_ref_to_index"]
request_meta["iforward"]["steps"][k]["source_indices"]
```

模型每个 step 根据 `source_indices` 从 `batch["source"]` 中取当前 evidence views。

---

## 7. Final supervision 采样

### 7.1 Current input supervision

用户确定：

```text
rollout 完成后，以短序列内所有输入帧作为监督进行反传
```

因此 P0 必须满足：

```text
final_current_recon frames == input_frame_indices
```

camera policy 推荐：

```yaml
supervision:
  current:
    camera_policy: all_cams
```

即：

```python
current_refs = all_cams(input_frame_indices)
roles = ["final_current_recon"] * len(current_refs)
```

校验条件：

```text
每个 input frame 必须至少有一个 final_current_recon ref
P0 默认每个 input frame 必须覆盖 all cams
```

---

### 7.2 Nearby supervision

用户确定：

```text
nearby 选择短序列内，非输入帧的随机帧作为监督；无则 skip
```

因此 nearby 不再沿用 phase A 的 adjacent/same-keyframe final-step 逻辑，而是 rollout-local random non-input。

#### 7.2.1 Rollout-local nearby frame pool

推荐定义 `rollout_local_frame_pool`：

```text
rollout_keyframes = input_keyframe_indices
candidate frames = union(sidx.keyframe_to_frames[kf] for kf in rollout_keyframes)
```

然后过滤：

```text
frame in train_frame_set
frame not in input_frame_indices
frame has frame_to_keyframe mapping
```

P0 nearby scope：

```yaml
nearby:
  scope: rollout_keyframe_span
```

含义：短序列内 = 本 rollout 选中的 keyframe blocks 所覆盖的 frame 集合。

可选扩展：

```text
temporal_minmax_span:
    在 min(input_frame_indices) 到 max(input_frame_indices) 之间选非输入帧

same_keyframe_per_input:
    每个输入 keyframe 内单独选 nearby
```

但 P0 推荐 `rollout_keyframe_span`。

#### 7.2.2 Nearby 采样规则

```python
candidates = sorted(unique(candidate_frames - input_frames))
if len(candidates) == 0:
    nearby_frames = []
    skipped_nearby = True
    nearby_skip_reason = "no_non_input_frame_in_rollout"
else:
    nearby_frames = random.sample(candidates, min(frames_per_rollout, len(candidates)))
```

配置：

```yaml
nearby:
  enable: true
  frames_per_rollout: 1
  insufficient_policy: use_available_or_skip_if_none
  camera_policy: all_cams
  max_refs_per_rollout: 24
  add_to_evidence: false
  role_name: final_nearby_rollout
```

#### 7.2.3 Nearby refs

```python
nearby_refs = all_cams(nearby_frames)
roles = ["final_nearby_rollout"] * len(nearby_refs)
```

强约束：

```text
nearby refs 不允许进入 evidence_refs
nearby frames 不允许等于 input frames
nearby refs 只进入 target render loss
无 nearby candidates 时 skip，不报错
```

---

### 7.3 Optional future：episode history anchors

P0 不强制加入 previous-rollout history anchors，因为用户当前确定的训练策略是：

```text
rollout 完成后，以短序列内所有输入帧作为监督
nearby 只从短序列内非输入帧选择
```

但 IForward 长期目标是 history retention，因此 scheduler 应预留扩展位：

```yaml
history_replay:
  enable: false
```

未来可启用：

```text
从当前 episode 已处理过的 previous rollout input frames 采样 history anchors
作为 final_history_recon role
```

但这不是 P0 必须项。

---

## 8. Batch materialization 设计

### 8.1 Flatten refs

对于一个 `IForwardRolloutPlan`：

```python
evidence_refs_flat = dedupe_keep_order(flatten(step.evidence_refs for step in steps))
target_refs_flat, target_roles_flat = dedupe_refs_roles_keep_order(final_supervision.refs, final_supervision.roles)
```

注意：

```text
evidence refs 和 target refs 可以重叠
```

因为 input frames 既是 evidence，也可以是 current reconstruction supervision。

但：

```text
nearby refs 不允许与 evidence refs 重叠
```

---

### 8.2 V4 assembly

V4 materializer 调用：

```python
batch = self._assemble_segment_batch_from_image_refs(
    scene_id=plan.scene_id,
    segment_id=plan.segment_id,
    source_image_refs=plan.evidence_refs_flat,
    target_image_refs=plan.target_refs_flat,
    aux_image_refs=None,
    query_label_image_refs=None,
    include_test=include_test,
    test_image_refs=None,
    enforce_target0_equals_source=False,
    target_ref_purpose="train",
)
```

然后更新：

```python
batch["_iforward"] = dataclasses.asdict(plan)

batch["request_meta"].update({
    "scheduler_version": "iforward_v1",
    "model_family": "IForward",
    "scene_id": plan.scene_id,
    "segment_id": plan.segment_id,
    "episode_id": plan.episode_id,
    "rollout_id_global": plan.rollout_id_global,
    "rollout_idx_in_episode": plan.rollout_idx_in_episode,
    "inner_K": plan.inner_K,
    "source_image_refs": plan.evidence_refs_flat,
    "target_image_refs": plan.target_refs_flat,
    "target_image_roles": plan.target_roles_flat,
    "iforward": {...},
})
```

---

### 8.3 Source / target index mapping

模型不应该每次通过 tuple 查找 refs。materializer 应直接写 index mapping。

```python
source_ref_to_index = {
    ref: idx for idx, ref in enumerate(plan.evidence_refs_flat)
}
target_ref_to_index = {
    ref: idx for idx, ref in enumerate(plan.target_refs_flat)
}
```

每个 step 写：

```python
step_meta = {
    "step_idx": step.step_idx,
    "source_frame_idx": step.source_frame_idx,
    "evidence_refs": step.evidence_refs,
    "source_indices": [source_ref_to_index[ref] for ref in step.evidence_refs],
    "commit_observation_memory": step.commit_observation_memory,
    "update_optimizer_memory": step.update_optimizer_memory,
    "detach_before_step": step.detach_before_step,
    "detach_after_step": step.detach_after_step,
    "repeat_idx": step.repeat_idx,
    "episode_block_idx": step.episode_block_idx,
    "rollout_block_rank": step.rollout_block_rank,
}
```

Final supervision 写：

```python
final_supervision_meta = {
    "refs": plan.target_refs_flat,
    "roles": plan.target_roles_flat,
    "target_indices_by_role": {
        "final_current_recon": [...],
        "final_nearby_rollout": [...],
    },
    "current_input_frames": plan.final_supervision.current_input_frames,
    "nearby_frames": plan.final_supervision.nearby_frames,
    "skipped_nearby": plan.final_supervision.skipped_nearby,
}
```

---

### 8.4 Role groups

`request_meta["role_groups"]` 建议：

```python
[
  {
    "role": "evidence_input",
    "refs": evidence_refs_flat,
    "allow_update_evidence": True,
    "allow_render_loss": False,
    "allow_memory_write": True,
    "mask_policy": "non_sky_non_egocar",
  },
  {
    "role": "final_current_recon",
    "refs": current_refs,
    "allow_update_evidence": False,
    "allow_render_loss": True,
    "allow_memory_write": False,
    "mask_policy": "non_sky_non_egocar",
  },
  {
    "role": "final_nearby_rollout",
    "refs": nearby_refs,
    "allow_update_evidence": False,
    "allow_render_loss": True,
    "allow_memory_write": False,
    "mask_policy": "non_sky_non_egocar",
  },
]
```

---

## 9. Leakage / consistency checks

IForward scheduler 必须 fail-fast。

### 9.1 基础检查

```text
所有 refs 必须属于同一个 scene / segment
所有 train refs 必须通过 dataset.validate_image_ref(..., purpose="train")
禁止 test refs 出现在 train rollout
len(target_image_refs) == len(target_image_roles)
len(steps) == inner_K
inner_K == actual_blocks_per_rollout * repeats_per_block
```

### 9.2 Evidence checks

```text
evidence_refs_flat 非空
每个 step 的 evidence_refs 非空
每个 step 的 evidence_refs 必须来自单一 source_frame_idx
每个 input frame 至少出现一次 commit_observation_memory=True
同一 input frame 的 repeat 0 commit=True，其余 repeat commit=False
```

### 9.3 Current supervision checks

```text
final_current_recon frames == input_frame_indices set
P0 默认 final_current_recon 覆盖每个 input frame 的 all cams
```

### 9.4 Nearby checks

```text
nearby_frames ∩ input_frame_indices == empty
nearby_refs ∩ evidence_refs == empty
nearby refs 不进入 source_image_refs
无 nearby candidates 时 skipped_nearby=True 且不报错
```

### 9.5 State lifecycle checks

```text
reset_scene_state_before_rollout=True only for first rollout in episode
carry_scene_state_after_rollout=True for all rollouts except trainer may discard after episode end
episode_end_after_rollout=True only for last rollout in episode
inside rollout all detach_before_step/detach_after_step=False
```

---

## 10. Scheduler 状态推进

### 10.1 Global step

```text
global_step += 1 per rollout batch
```

不是 per repeat step。

---

### 10.2 Episode cursor

每个 episode state：

```python
current_episode_state = {
    "scene_id": int,
    "segment_id": int,
    "episode_id": int,
    "episode_start_keyframe_pos": int,
    "keyframe_window": List[int],
    "frame_chain": List[int],
    "block_cursor": int,
    "rollout_idx_in_episode": int,
    "rollout_id_global_base": int,
    "episode_num_blocks": int,
    "episode_end": bool,
}
```

---

### 10.3 next_batch 逻辑

```python
def next_batch(self):
    if no current episode:
        start_next_episode()

    plan = build_rollout_plan(current_episode_state)
    validate_plan(plan)
    batch = dataset._assemble_segment_batch_from_iforward_request(plan)

    advance_rollout_cursor(plan)
    global_step += 1

    if plan.episode_end_after_rollout:
        mark_episode_finished_after_batch()

    emit_preload_hints_for_next_rollout()
    return batch
```

重要：

```text
scheduler 可以在返回 batch 前推进 cursor
trainer 根据 batch.request_meta 处理 reset/carry
```

也可以采用 “commit after materialization” 方式，关键是 state_dict/load_state_dict 能恢复一致。

---

## 11. Rollout plan 构建算法

### 11.1 选择 rollout shape

```python
shape = random.choices(active_shapes, weights=[s.prob for s in active_shapes], k=1)[0]
```

支持 curriculum：

```yaml
rollout:
  shapes_schedule:
    - start_step: 0
      shapes:
        - {name: b2_r8, blocks_per_rollout: 2, repeats_per_block: 8, prob: 1.0}
    - start_step: 10000
      shapes:
        - {name: b2_r8, blocks_per_rollout: 2, repeats_per_block: 8, prob: 0.4}
        - {name: b3_r6, blocks_per_rollout: 3, repeats_per_block: 6, prob: 0.4}
        - {name: b4_r4, blocks_per_rollout: 4, repeats_per_block: 4, prob: 0.2}
```

---

### 11.2 选择 rollout blocks

P0：next contiguous。

```python
start = episode_state.block_cursor
end = min(start + shape.blocks_per_rollout, episode_num_blocks)
selected_blocks = list(range(start, end))
```

如果 short final rollout：

```python
if len(selected_blocks) < shape.blocks_per_rollout:
    if allow_short_final_rollout and len(selected_blocks) >= min_blocks_per_rollout:
        short_rollout = True
    else:
        end episode or raise
```

---

### 11.3 选择 input frames

```python
input_frames = [frame_chain[b] for b in selected_blocks]
input_keyframes = [keyframe_window[b] for b in selected_blocks]
```

---

### 11.4 输入交付顺序

P0：chronological。

```python
delivery_blocks = selected_blocks
```

扩展：

```text
reverse
random_per_rollout
random_but_supervision_chronological
```

注意：supervision frames 应覆盖 input set，不依赖 delivery order。

---

### 11.5 展开 steps

```python
steps = []
for rollout_rank, block_idx in enumerate(delivery_blocks):
    frame_idx = frame_chain[block_idx]
    keyframe_idx = keyframe_window[block_idx]
    evidence_refs = select_evidence_refs(frame_idx)

    for repeat_idx in range(repeats_per_block):
        step_idx = len(steps)
        steps.append(IForwardStepPlan(
            step_idx=step_idx,
            episode_block_idx=block_idx,
            rollout_block_rank=rollout_rank,
            repeat_idx=repeat_idx,
            source_keyframe_idx=keyframe_idx,
            source_frame_idx=frame_idx,
            evidence_refs=evidence_refs,
            evidence_frame_indices=[frame_idx],
            evidence_cam_indices=[cam for _, cam in evidence_refs],
            commit_observation_memory=(repeat_idx == 0),
            update_optimizer_memory=True,
            detach_before_step=False,
            detach_after_step=False,
            allow_step_render_loss=False,
            step_loss_refs=[],
            rollout_pos_code=step_idx / max(inner_K - 1, 1),
            frame_pos_code=rollout_rank / max(len(delivery_blocks) - 1, 1),
            repeat_pos_code=repeat_idx / max(repeats_per_block - 1, 1),
        ))
```

---

### 11.6 Final supervision

```python
current_refs = refs_for_frames(input_frames, camera_policy="all_cams")
nearby_frames = sample_rollout_nearby_frames(...)
nearby_refs = refs_for_frames(nearby_frames, camera_policy="all_cams")

final_refs = current_refs + nearby_refs
final_roles = ["final_current_recon"] * len(current_refs) \
            + ["final_nearby_rollout"] * len(nearby_refs)
```

Deduplicate target refs with role conflict check。

---

## 12. Preload 设计

P0 可以复用 V4 preload manager 已支持的 scope：

```text
v9_role_refs
```

因为 `AssetPreloadManagerV2` 已经把 `v9_role_refs` 当作 exact view-pack warmup。IForward P0 可以先调用：

```python
hint = dataset.build_preload_hint(
    scene_id=plan.scene_id,
    segment_id=plan.segment_id,
    future_image_refs=dedupe(plan.evidence_refs_flat + plan.target_refs_flat),
    scope="v9_role_refs",
)
dataset.submit_preload_hint(..., hint_scope="v9_role_refs", ...)
```

后续可以扩展 preload manager，加入：

```text
iforward_rollout_refs
iforward_next_rollout_refs
```

但 P0 不需要阻塞在 preload scope 重构上。

推荐 preload：

```yaml
preload:
  emit_hints: true
  warm_current_rollout_refs: true
  warm_next_rollout_refs: true
  warm_episode_chain: true
```

---

## 13. Trainer 与模型的 batch contract

IForward trainer 收到 batch 后，按以下约定执行。

### 13.1 Reset / carry

```python
meta = batch["request_meta"]["iforward"]

if meta["reset_scene_state_before_rollout"]:
    state = init_iforward_state_from_batch_assets(batch)
else:
    state = state_cache[(scene_id, segment_id, episode_id)]
```

rollout 结束后：

```python
loss.backward()
optimizer.step()

state = detach_state_values(state)

if meta["episode_end_after_rollout"]:
    delete state_cache[(scene_id, segment_id, episode_id)]
else:
    state_cache[(scene_id, segment_id, episode_id)] = state
```

P0 推荐只允许 episode serial，因此 state_cache 实际只需要一个 active state。

---

### 13.2 Model step loop

```python
state = current_state
steps = batch["request_meta"]["iforward"]["steps"]

for step in steps:
    src = select_source_views(batch["source"], step["source_indices"])

    observation = model.observe(state, src)
    event = model.encode_event(state, observation)
    mem_read, state.memory = model.memory_step(
        state,
        event,
        commit_observation=step["commit_observation_memory"],
        update_optimizer_memory=step["update_optimizer_memory"],
    )
    delta = model.predict_delta(state, event, mem_read)
    state.gs = model.apply_delta(state.gs, delta)

# final supervision only
loss = render_final_supervision(state, batch["target"], final_supervision_meta)
```

---

## 14. 配置草案

```yaml
scheduler_iforward:
  enable: true
  version: iforward_v1
  fail_fast: true

  traversal:
    fixed_scene_id: null
    fixed_segment_id: null
    scene_order: shuffle_per_epoch
    segment_order: shuffle_per_epoch
    traversal_mode: episode_serial

  episode:
    source_mode: keyframes
    blocks_per_episode: 8
    episode_stride: 8
    allow_short_last_episode: true
    min_blocks_per_episode: 2
    block_source_frame_policy: random_within_keyframe_once_per_episode
    reset_scene_state_policy: episode_begin

  rollout:
    block_selection_policy: next_contiguous
    delivery_order_policy: chronological
    allow_short_final_rollout: true
    min_blocks_per_rollout: 1
    detach_graph_after_rollout: true
    shapes:
      - {name: b2_r8, blocks_per_rollout: 2, repeats_per_block: 8, prob: 0.50}
      - {name: b3_r6, blocks_per_rollout: 3, repeats_per_block: 6, prob: 0.30}
      - {name: b4_r4, blocks_per_rollout: 4, repeats_per_block: 4, prob: 0.20}
    shapes_schedule: []

  evidence:
    camera_policy: all_cams
    cams_per_input: null
    fixed_cam_ids: null
    allow_camera_dropout: false
    mask_policy: non_sky_non_egocar

  memory:
    observation_commit_policy: first_repeat_only
    optimizer_memory_update_policy: every_repeat
    reset_policy: episode_begin
    carry_policy: across_rollouts_until_episode_end

  loss_timing:
    policy: rollout_final_only
    intermediate_step_loss: false

  supervision:
    current:
      enable: true
      role_name: final_current_recon
      frame_policy: all_input_frames
      camera_policy: all_cams
      required: true

    nearby:
      enable: true
      role_name: final_nearby_rollout
      scope: rollout_keyframe_span
      policy: random_non_input_frames
      frames_per_rollout: 1
      insufficient_policy: use_available_or_skip_if_none
      camera_policy: all_cams
      max_refs_per_rollout: 24
      add_to_evidence: false
      mask_policy: non_sky_non_egocar

    history_replay:
      enable: false

  leakage_check:
    enable: true
    same_scene_segment_required: true
    forbid_test_refs_in_train: true
    target_role_count_match_required: true
    nearby_not_in_evidence: true
    nearby_not_input_frame: true
    current_supervision_must_cover_all_inputs: true

  preload:
    emit_hints: true
    warm_current_rollout_refs: true
    warm_next_rollout_refs: true
    warm_episode_chain: true
    hint_scope_for_exact_refs: v9_role_refs
```

---

## 15. P0 / P1 实现路线

### P0：可训练闭环

必须完成：

```text
1. 新建 datasets/train_scheduler_iforward.py
2. 新建 IForward dataclasses
3. V4 增加 create_train_scheduler_iforward
4. V4 增加 _assemble_segment_batch_from_iforward_request
5. episode -> rollout -> block -> repeat 状态机
6. rollout-local current + nearby final supervision
7. batch request_meta index mapping
8. reset/carry/detach flags
9. fail-fast validation
10. preload hints 复用 v9_role_refs
```

P0 验证目标：

```text
一个 rollout batch 能正确加载所有 evidence / target refs
rollout 内 steps 顺序正确
current supervision 覆盖所有输入帧
nearby 只来自短序列内非输入帧，无则 skip
first rollout reset=true，后续 rollout reset=false
episode last rollout episode_end=true
```

---

### P1：训练质量增强

```text
1. rollout shape curriculum
2. evidence camera subset curriculum
3. random / reverse delivery order
4. optional intermediate loss
5. optional previous-rollout history replay
6. heldout cam supervision
7. scheduler validation protocol
```

---

### P2：长序列前置

```text
1. episode 内更多 rollouts
2. chunk overlap
3. previous episode warm-start experiments，默认 off
4. state cache checkpoint / restore
5. memory ablation-friendly metadata
```

---

## 16. 最终建议

IForward scheduler 的核心不是“再做一个 V9”，而是把训练单位改成真正的短序列 rollout：

```text
rollout 内连续优化，不 detach
rollout final 用所有输入帧监督
nearby 从 rollout-local 非输入帧随机监督
rollout 后 carry scene state
episode 结束 reset scene state
```

这与 IForward 模型的定义一致：

```text
IForward = persistent short-sequence 3DGS iterative optimizer
```

因此 scheduler 的设计必须从第一天就围绕 persistent optimizer，而不是围绕单帧 block 或 phase-B VSM 读写。
