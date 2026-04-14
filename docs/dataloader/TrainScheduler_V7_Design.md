# TrainSchedulerV7 设计文档（Episode Rolling Traversal）

## 文档目标

本文定义 `TrainSchedulerV7` 的主语义与实现方案。  
V7 不再是 V5/V6 的局部修补，而是调度主语义重构：

- 从 `U -> block -> episode -> segment -> scene` 迁移到 `raw_step -> block -> episode -> episode-level global traversal`
- 将 scene/segment 切换单位从“segment 完成”提前到“episode 完成”
- 在 episode 内引入确定性的 rolling block chain，保证 source 随时间单调推进

V7 依赖并复用现有 `MultiSceneDatasetV4` 与 `AssetPreloadManagerV2` 的资产接口，不推翻 V4 资产模型。

---

## 1. V7 新定位

### 1.1 训练主语义变更

从旧语义：

- `U -> block -> episode -> segment -> scene`

改为新语义：

- `raw_step`（trainer 内部计数）
- `-> Block B`
- `-> Episode E`
- `-> Episode-level global traversal`

关键变化：

1. `U` 从 scheduler 内部语义移除（scheduler 不再管理 step 与 U 的换算层）
2. scene/segment 切换粒度提前到 episode 结束，而非 segment 结束

### 1.2 目标导向

V7 的调度目标不再是“segment 内随机块训练 + 驻留摊薄切换成本”，而是：

- 在 segment 内按时间顺序推进
- 让 block 间形成滚动连续链
- 在多 scene 训练时按 episode 级别交错，避免单 segment 长时间独占

---

## 2. 核心定义与公式

定义：

- `T = total_target_frames`
- `E = blocks_per_episode`
- `W = episode_window_keyframes = E + T - 1`

当前主任务（`T=3`）下：

- `W = E + 2`

一个 episode 在某个 segment 中先抽取连续 keyframe 窗口：

- `[k0, k1, ..., k_{W-1}]`

然后在 episode 开始阶段，仅一次采样得到冻结 frame chain：

- `[f0, f1, ..., f_{W-1}]`

强约束：

- 同一个 episode 内，frame chain 必须固定不变
- 禁止在 block 内二次重采样同 keyframe

原因：需要保证“上一 block 的后续 target 成为下一 block 的 source”，避免随机重采样导致连续性被破坏。

---

## 3. Block 构造规则（Rolling Chain）

对 `b = 0 ... E-1`，第 `b` 个 block 定义为：

- `source_frame = f_b`
- `target_frames = [f_b, f_{b+1}, ..., f_{b+T-1}]`

由于 `W = E + T - 1`，`b=E-1` 时 target 仍不越界。

当 `T=3` 时：

- block 0: `[f0, f1, f2]`
- block 1: `[f1, f2, f3]`
- ...
- block E-1: `[f_{E-1}, f_E, f_{E+1}]`

这就是 V7 的 rolling 训练单元定义。

---

## 4. 为什么 E 在 V7 必须重新定义

V5/V6 中，episode 相关量由多个预算变量组合得到（如 `keyframes_per_episode`、`episodes_per_segment`、`updates_per_block`、`U`）。

V7 中应明确：

- `E = blocks_per_episode`

原因：

- V7 的核心是“episode 内滚动 block 数决定窗口长度”
- `W = E + T - 1` 直接由 block 数驱动
- 若继续沿用“E 是 keyframe 数或中间预算变量”的语义，会与 rolling chain 主语义冲突

---

## 5. 遍历策略（从头到尾覆盖）

### 5.1 单 segment 内 episode 生成

设 segment keyframe 序列为：

- `K = [k0, k1, ..., k_{N-1}]`

策略：

- `N < W`：直接跳过（`skip_if_less_than_window`）
- `N >= W`：按顺序生成 episode 窗口起点，不使用随机窗口默认策略

推荐起点序列：

- `start = 0, E, 2E, 3E, ...`

对应窗口：

- `K[start : start + W]`

尾部策略（tail-aligned）：

- `tail_start = N - W`
- 若最后一个规则起点不等于 `tail_start`，补一个尾对齐 episode

这保证：

- source coverage 连续推进
- 尾部不漏覆盖
- 仍保持固定窗口长度，不引入“变长 episode”复杂度

### 5.2 全局 scene/segment 切换策略

#### 模式 A：线性调试（固定 scene / segment）

- `traversal.mode = linear_scene_segment`
- 若给定 `fixed_scene_id`，scene 固定
- segment 按 `segment_id` 升序
- episode 起点按时间升序

适用：单场景从头到尾调试与可解释性验证。

#### 模式 B：多 scene 正式训练（推荐默认）

- `traversal.mode = round_robin_episode_interleave`
- 每个 `(scene, segment)` 维护独立 episode cursor
- 每次只执行一个 episode，结束即切到下一个 cursor

收益：

- segment 内保持时间顺序推进
- 全局训练避免长期停留单 segment
- 与现有资产缓存体系兼容（切换成本可接受）

---

## 6. 必须同步修改的训练语义

### 6.1 Scheduler 内移除 U 层

V7 不再维护以下状态：

- `state_write_interval_steps`
- `segment_budget_u`
- `segment_local_u`
- `u_in_block`

保留：

- `block.steps_per_block`

解释：

- 同一 block 可重复训练 `steps_per_block` 次
- 但 scheduler 不再表达 `U -> block` 的中间语义，仅表达样本调度推进

### 6.2 周期性 reset 机制降级

对 V7 的连续时间推进目标，固定间隔 reset（如 `reset_node_state_interval=10`）通常是冲突项。  
V7 语义下建议：

- segment 内不做机械周期 reset
- 在 segment 完成或 epoch 边界做 reset

否则会在尚未遍历完成时打断连续优化链。

---

## 7. 实现方案

### 7.1 新文件与工厂

新增：

- `datasets/train_scheduler_v7.py`

`MultiSceneDatasetV4` 侧新增工厂：

- `create_train_scheduler_v7(...)`

### 7.2 数据集接口依赖（沿用 V4）

V7 继续依赖已存在接口：

- `list_training_scene_ids()`
- `list_segment_ids(scene_id)`
- `get_segment_index(scene_id, segment_id)`
- `get_segment_batch_from_image_refs(...)`

无需推翻 `SegmentIndexV4`/`BatchRequestV4`/`SegmentStaticBundle`。

### 7.3 关键状态结构

```python
@dataclass(frozen=True)
class EpisodeCursorV7:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int


@dataclass(frozen=True)
class EpisodeRuntimeV7:
    scene_id: int
    segment_id: int
    keyframe_window: List[int]
    frame_chain: List[int]
    block_windows: List[List[int]]  # len == E, each len == T
```

运行时 `current_episode_state` 建议字段：

- `scene_id`
- `segment_id`
- `episode_idx_global`
- `episode_start_keyframe_pos`
- `keyframe_window`
- `frame_chain`
- `block_windows`
- `block_cursor`
- `block_repeat_step`
- `current_source_frame_idx`
- `current_target_frame_indices`

---

## 8. 配置设计（scheduler_v7）

```yaml
scheduler_v7:
  enable: true

  block:
    steps_per_block: 8

  episode:
    blocks_per_episode: 3
    total_target_frames: 3
    include_source_frame: true
    frame_within_keyframe_policy: random_once_per_episode
    min_keyframes_required_policy: skip_if_less_than_window

  traversal:
    mode: round_robin_episode_interleave   # or linear_scene_segment
    switch_after_episode: true
    fixed_scene_id: null
    fixed_segment_id: null
    segment_order: ascending
    scene_order: shuffle_per_epoch

  preload:
    emit_hints: true
    warm_next_block_exact: true
    warm_next_episode_chain: true

  include_test: false
```

派生量：

- `T = total_target_frames`
- `E = blocks_per_episode`
- `W = E + T - 1`

---

## 9. 关键函数设计

### 9.1 `_build_segment_episode_starts(num_keyframes, E, W)`

输入：segment 的 keyframe 数  
输出：episode 起点数组

规则：

- `num_keyframes < W` -> `[]`
- 默认起点：`0, E, 2E, ...`
- 若尾部未覆盖，追加 `tail = num_keyframes - W`

```python
def _build_segment_episode_starts(num_keyframes: int, E: int, W: int) -> List[int]:
    if num_keyframes < W:
        return []
    starts = list(range(0, num_keyframes - W + 1, E))
    tail = num_keyframes - W
    if starts[-1] != tail:
        starts.append(tail)
    return starts
```

### 9.2 `_start_episode(cursor)`

步骤：

1. 从 `SegmentIndexV4.keyframe_indices` 取连续 `W` 个 keyframe
2. 对每个 keyframe 按 `frame_within_keyframe_policy` 采样一次 frame
3. 冻结形成 `frame_chain`
4. 构建 `block_windows = [frame_chain[b:b+T] for b in range(E)]`

### 9.3 `_start_block()`

步骤：

1. `source_frame_idx = block_windows[block_cursor][0]`
2. `target_frame_indices = block_windows[block_cursor]`
3. 扩展到 all-cam image refs
4. 组装 `BatchRequestV4` 并调用 `get_segment_batch_from_image_refs(...)`

### 9.4 `next_batch()`

推进规则：

1. 若当前 block 的 `block_repeat_step < steps_per_block`，复用当前 block batch 语义
2. 否则推进 `block_cursor`
3. 若 episode 结束，切换到下一个 episode cursor（按 traversal 模式）
4. 若所有 cursor 耗尽，开始新 epoch 并重建 traversal 队列

---

## 10. Frame 选取策略

V7 首版仅保留简单策略：

- `random_once_per_episode`（默认）
- `middle_frame`

默认含义：

- 每个 keyframe 在 episode 开始时随机采样一帧
- 一个 episode 内固定，不随 block 重采样

---

## 11. Preload 方案（V7 增量）

保留已有能力：

- segment static 预热
- `next_block_exact` 预热

新增能力：

- `next_episode_chain` 预热（episode 级 frame_chain）

要求：

- 至少预热 image meta
- 可选预热 view pack（依据当前 preload cfg）
- 采用新 hint scope，不改 `AssetPreloadManagerV2` 线程模型

建议 hint scope：

- `next_block_exact`
- `next_episode_chain`

---

## 12. 与 V5/V6 差异总结

### 12.1 V5/V6

- segment 仍是主要驻留单位
- `U -> block -> episode -> segment` 多层预算
- episode 内 block 非严格 rolling chain
- 单 segment 内存在随机窗口/随机采样主导行为

### 12.2 V7

- episode 成为主要调度单位
- segment 内 episode 按时间顺序推进
- episode 内 block 为确定性的 rolling chain
- scheduler 不再管理 U 层
- episode 结束即可切换 scene/segment（按 traversal 模式）

---

## 13. 建议默认参数（T=3）

```yaml
scheduler_v7:
  enable: true

  block:
    steps_per_block: 6

  episode:
    blocks_per_episode: 3
    total_target_frames: 3
    include_source_frame: true
    frame_within_keyframe_policy: random_once_per_episode
    min_keyframes_required_policy: skip_if_less_than_window

  traversal:
    mode: round_robin_episode_interleave
    switch_after_episode: true
    segment_order: ascending
    scene_order: shuffle_per_epoch
    fixed_scene_id: null
    fixed_segment_id: null

  preload:
    emit_hints: true
    warm_next_block_exact: true
    warm_next_episode_chain: true

  include_test: false
```

对应派生量：

- `E = 3`
- `T = 3`
- `W = 5`

即：有效 segment 至少需要 5 个 keyframe。  
单 episode rolling block 为：

- `[f0, f1, f2]`
- `[f1, f2, f3]`
- `[f2, f3, f4]`

---

## 14. 迁移与落地顺序

建议按以下顺序落地，降低回归风险：

1. 新增 `train_scheduler_v7.py` 与基础单测（窗口构造、rolling block、tail 策略）
2. 在 `MultiSceneDatasetV4` 增加 `create_train_scheduler_v7(...)`
3. 训练入口新增 `scheduler_v7` 配置解析，保留 v6 兼容分支
4. preload hint 增加 `next_episode_chain` 路径
5. 联调固定 scene/segment 的线性模式，再切 round-robin 模式

---

## 15. 验收标准（DoD）

满足以下条件视为 V7 基线可用：

1. episode 内 block 严格满足 rolling chain，且 frame chain 在 episode 内不变
2. segment 内 episode 起点顺序推进，尾部按 tail-aligned 规则补齐
3. round-robin 模式下，episode 结束后发生 scene/segment 级交错
4. scheduler 状态中不再出现 U 预算字段
5. `MultiSceneDatasetV4` 资产路径与 batch 组装接口无需改写即可跑通
6. 预热日志可观测到 `next_block_exact` 与 `next_episode_chain` 两类 hint

---

## 一句话定义

`TrainSchedulerV7` 是一个以 episode 为调度主单位、以 rolling block 为训练单元、以全局 episode 交错为遍历策略的时序推进型 scheduler。

