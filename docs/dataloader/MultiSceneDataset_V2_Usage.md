# MultiSceneDataset V2 Usage

本文档说明 `MultiSceneDatasetV2` 的使用方式与行为约束。  
v2 面向 StreetForward Scheduler v2，强调 **采样与组 batch 解耦** 和 **fast-fail**。

相关设计文档：

- `docs/trainers/StreetForward_Scheduler_V2_Design.md`
- `docs/trainers/StreetForward_Scheduler_V2_Usage.md`

---

## 1. 核心变化（相对 v1）

1. 新增显式接口：

```python
batch = dataset.get_segment_batch_from_frames(
    scene_id=...,
    segment_id=...,
    source_frame_idx=...,
    target_frame_indices=[...],  # target[0] 必须是 source
    include_test=False,
)
```

2. 采样职责外移：

- dataset 只负责加载与组装。
- scheduler 负责 source/target 的采样策略与状态推进。

3. fast-fail 约束：

- `target_frame_indices` 不能为空。
- `target_frame_indices[0]` 必须等于 `source_frame_idx`。
- source/target frame 必须属于该 segment 的 `frame_indices`。

---

## 2. 训练侧典型调用

```python
from datasets.multi_scene_dataset_v2 import MultiSceneDatasetV2

dataset = MultiSceneDatasetV2(...)
dataset.initialize()

scheduler = dataset.create_train_scheduler_v2(
    alpha_steps_per_keyframe=4,
    min_steps_per_segment=12,
    max_steps_per_segment=48,
    source_hold_steps=4,
    num_target_frames_total=6,
    target_include_source=True,
    include_test=False,
    fixed_scene_id=1,      # one-segment 可指定
    fixed_segment_id=0,    # one-segment 可指定
)

for _ in range(100):
    batch = scheduler.next_batch()
```

---

## 3. Eval manifest 与评估调度

### 3.1 构建 deterministic manifest

```python
manifest = dataset.build_eval_manifest_v2(
    scene_ids=[5, 6],
    num_target_frames_total=6,
    max_test_frames_per_segment=4,
)
```

### 3.2 固定顺序评估

```python
eval_scheduler = dataset.create_eval_scheduler_v2(
    manifest=manifest,
    include_test=True,
)

try:
    while True:
        batch = eval_scheduler.next_batch()
        # evaluate(batch)
except StopIteration:
    pass
```

---

## 4. batch 结构兼容性说明

`get_segment_batch_from_frames()` 复用原有 batch 组装路径，返回结构与 `get_segment_batch()` 主体一致（包含 `source/target/test/pointcloud/dynamic_info` 等字段）。  
v2 的主要变化是 **采样语义**，不是 batch 字段命名。

---

## 5. 常见报错与处理

- `target_frame_indices[0] must equal source_frame_idx`  
  说明 scheduler 没有把 source frame 放到 target 第一位。

- `source_frame_idx is not in scene/segment`  
  说明 frame plan 与 segment 边界不一致。

- `No non-source keyframes available for extra target sampling`  
  说明当前段 keyframe 数太少，无法满足 `num_target_frames_total > 1`。

