# TrainSchedulerV8 设计文档（Visited Episode Frame Target）

## 文档目标

本文定义 `TrainSchedulerV8` 与 `ValidationV8` 的统一语义与落地方案。  
V8 保留 V7 的 episode-level traversal 与 `step_major` 执行机制，但移除 future rolling window target，改为 visited-episode-frame target。

核心结论：

- V7 的主要问题不是 `step_major`，而是 target 仍来自 future rolling window。
- V8 只在 episode 内使用 `E` 个 block source frames，不再预留 `E + T - 1` 的 future frames。
- V8 的 target 来自 `当前 source + 当前 episode 内已访问过的 block source frames`。

---

## 1. 一句话定义

```text
SchedulerV8 = SchedulerV7 的 episode-level traversal + step_major 执行方式
              但移除 future rolling target，
              改为 visited-episode-frame target。
```

对比：

```text
V7:
  episode_window_keyframes = E + T - 1
  block b target = [f_b, f_{b+1}, ..., f_{b+T-1}]

V8:
  episode_window_keyframes = E
  block b source = f_b
  block b target = source f_b + 当前 episode 内已经访问过的 block source frames
```

其中：

- `E = blocks_per_episode`
- `T = total_target_frames`（在 V8 中语义变为 `max_target_frames`）
- `W = episode_window_keyframes`

在 V8 中强制：

```text
W = E
```

---

## 2. 语义澄清：允许 `(s, s+1, s+2)` 的前提

以配置为例：

```yaml
block:
  steps_per_block: 8
episode:
  blocks_per_episode: 3
  total_target_frames: 3
execution:
  block_order: step_major
  step_major_switch_interval_steps: 4
  reset_policy: episode_end
```

若某个 episode 的 `frame_chain = [f0, f1, f2]`：

- 第一次访问 `b0`：target 只能是 `{f0}`
- 第一次访问 `b1`：target 可以是 `{f1, f0}`
- 第一次访问 `b2`：target 可以是 `{f2, f1, f0}`
- 第二轮回到 `b0` 时：因为 `b1/b2` 已访问，target 可扩展到 `{f0, f1, f2}`

关键点：

- 允许出现 `(s, s+1, s+2)`，前提是 `s+1/s+2` 已在当前 episode 内被访问过。
- 禁止的是“第一次访问 `s` 时直接监督未访问 future frame”。

---

## 3. 训练侧核心数据结构

建议新增：

```python
@dataclass(frozen=True)
class EpisodePlanV8:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int
    keyframe_window: List[int]      # len == E
    frame_chain: List[int]          # len == E
    num_cams: int
```

与 V7 的关键差异：

- 移除 `block_windows` 预构造。
- target 在运行时按 visited 状态动态生成。

运行时状态建议新增字段：

```python
current_episode_state = {
    "frame_chain": frame_chain,
    "visited_block_indices": set(),
    "block_first_visit_order": {},
    "current_source_frame_idx": -1,
    "current_target_frame_indices": [],
}
```

---

## 4. Episode 窗口长度修改

V7 现状：

```python
self.episode_window_keyframes = self.blocks_per_episode + self.total_target_frames - 1
```

V8 改为：

```python
self.episode_window_keyframes = self.blocks_per_episode
```

即：

```text
E = blocks_per_episode
T = max_target_frames
W = E
```

当 `E=3, T=3` 时，每个 episode 只采样 `3` 帧链（`[f0, f1, f2]`），不再采样 `5` 帧链。

---

## 5. Target 构造规则（V8）

建议函数：

```python
def _build_target_frames_for_block_v8(
    *,
    frame_chain: List[int],
    block_idx: int,
    visited_block_indices: set[int],
    max_target_frames: int,
) -> List[int]:
    source_frame = int(frame_chain[block_idx])
    if max_target_frames < 1:
        raise ValueError("max_target_frames must be >= 1")

    candidates = [int(b) for b in visited_block_indices if int(b) != int(block_idx)]
    prev_blocks = sorted([b for b in candidates if b < int(block_idx)], reverse=True)
    next_blocks = sorted([b for b in candidates if b > int(block_idx)])

    selected_blocks: List[int] = []
    for b in prev_blocks:
        if len(selected_blocks) >= int(max_target_frames) - 1:
            break
        selected_blocks.append(int(b))
    for b in next_blocks:
        if len(selected_blocks) >= int(max_target_frames) - 1:
            break
        selected_blocks.append(int(b))

    return [int(source_frame)] + [int(frame_chain[b]) for b in selected_blocks]
```

说明：

- source 永远放在 `target[0]`，与 `enforce_target0_equals_source=True` 对齐。
- 语义描述可用“history + source”，但传给 batch 的 `target_image_refs[0]` 必须等于 source。

---

## 6. `_select_block()` 行为变化

V8 的 `_select_block()` 关键流程：

1. 读取 `frame_chain` 与 `block_idx`。
2. 基于当前 `visited_block_indices` 构造 target。
3. 更新 `source_image_refs/target_image_refs`。
4. 最后再将当前 block 记为 visited（保证首次访问不会引入自己之后的未访问帧）。

参考伪代码：

```python
target_frames = self._build_target_frames_for_block_v8(
    frame_chain=frame_chain,
    block_idx=bcur,
    visited_block_indices=set(st["visited_block_indices"]),
    max_target_frames=int(self.total_target_frames),
)

# ... set current source/target refs ...

if bcur not in st["visited_block_indices"]:
    st["visited_block_indices"].add(int(bcur))
    st["block_first_visit_order"][int(bcur)] = int(st["episode_step_cursor"])
```

---

## 7. `block_major` 与 `step_major` 的统一解释

`block_major`：

```text
b0, b0, ..., b1, b1, ..., b2, b2, ...
```

target 递进为：

```text
b0: {f0}
b1: {f1, f0}
b2: {f2, f1, f0}
```

`step_major`：

```text
b0, b1, b2, b0, b1, b2, ...
```

target 递进为：

```text
第一次 b0: {f0}
第一次 b1: {f1, f0}
第一次 b2: {f2, f1, f0}
第二次 b0: {f0, f1, f2}
第二次 b1: {f1, f0, f2}
第二次 b2: {f2, f1, f0}
```

这保留 step-major 混合训练收益，同时不引入未访问 future 监督。

---

## 8. ValidationV8 设计

V7 验证函数 `build_validation_episode_specs_v7()` 当前语义为固定 `T=3` rolling window（`W = E + 2`）。  
V8 需改为与训练一致：

- `episode_window_keyframes = blocks_per_episode`
- 不再构造 `block_windows = frame_chain[b:b+T]`
- 改为按 visit order 构造 `visit_target_windows`

建议新增：

```python
@dataclass(frozen=True)
class ValidationEpisodeSpecV8:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int
    frame_chain: List[int]              # len == E
    block_visit_order: List[int]        # len == E * steps_per_block / switch-driven expansion
    visit_target_windows: List[List[int]]
    eval_image_refs: List[Tuple[int, int]]
    num_cams: int
```

建议函数：

```python
def build_visit_target_windows_v8(
    *,
    frame_chain: List[int],
    block_visit_order: List[int],
    max_target_frames: int,
) -> List[List[int]]:
    visited: set[int] = set()
    out: List[List[int]] = []
    for bcur in block_visit_order:
        source = int(frame_chain[int(bcur)])
        prev_blocks = sorted([b for b in visited if b < int(bcur)], reverse=True)
        next_blocks = sorted([b for b in visited if b > int(bcur)])
        selected: List[int] = []
        for b in prev_blocks:
            if len(selected) >= int(max_target_frames) - 1:
                break
            selected.append(int(b))
        for b in next_blocks:
            if len(selected) >= int(max_target_frames) - 1:
                break
            selected.append(int(b))
        out.append([int(source)] + [int(frame_chain[b]) for b in selected])
        visited.add(int(bcur))
    return out
```

`eval_image_refs` 只需覆盖 `frame_chain` 对应帧（`E` 帧），不再包含额外 future 帧。

---

## 9. Preload 策略调整

V7 中 `episode_chain_exact` 与 `warm_next_episode_chain` 可包含 `E + T - 1` 帧。  
V8 改为仅基于 `frame_chain[0:E]`。

推荐两阶段方案：

1. **稳定优先（先落地）**  
   `next_block_exact` 直接预热整个 episode chain（`E` 帧 × cams），避免 dynamic target 漏预热。
2. **性能优化（后续可选）**  
   再引入“模拟 visited 状态”的精细预热，只预热 next block 预计 target。

由于常见配置 `E=3`，先用稳定方案开销可接受。

---

## 10. 配置设计建议

训练：

```yaml
scheduler_v8:
  enable: true
  block:
    steps_per_block: 8
  episode:
    blocks_per_episode: 3
    total_target_frames: 3          # V8 语义: max_target_frames
    include_source_frame: true
    target_policy: visited_episode_frames
    frame_within_keyframe_policy: random_once_per_episode
    min_keyframes_required_policy: skip_if_less_than_window
  traversal:
    mode: round_robin_episode_interleave
    switch_after_episode: true
    fixed_scene_id: null
    fixed_segment_id: null
    segment_order: ascending
    scene_order: shuffle_per_epoch
  execution:
    block_order: step_major
    step_major_switch_interval_steps: 4
    reset_policy: episode_end
  preload:
    emit_hints: true
    warm_next_block_exact: true
    warm_next_episode_chain: true
```

验证：

```yaml
validation_v8:
  eval_enable: true
  mode: segment_finetune_train
  block:
    steps_per_block: 8
  episode:
    blocks_per_episode: 3
    total_target_frames: 3
    target_policy: visited_episode_frames
  execution:
    block_order: step_major
    step_major_switch_interval_steps: 4
    reset_policy: episode_end
  trigger:
    by: train_episode_interval
    validate_every_n_episodes: 100
    run_at_train_start: false
  episode_selection:
    policy: middle
  render:
    save_images: true
    save_dir: validation/episodes
  cache:
    persist_across_training: true
```

---

## 11. Fast-Fail 约束（建议强制）

V8 初始化建议增加以下硬约束：

```python
if total_target_frames < 1:
    raise ValueError("scheduler_v8.episode.total_target_frames must be >= 1")
if total_target_frames > blocks_per_episode:
    raise ValueError(
        "scheduler_v8 does not use future frames; total_target_frames must be <= blocks_per_episode"
    )
if target_policy != "visited_episode_frames":
    raise ValueError("scheduler_v8 only supports target_policy=visited_episode_frames")
if reset_policy != "episode_end":
    raise ValueError("scheduler_v8 requires execution.reset_policy=episode_end")
if not include_source_frame:
    raise ValueError("scheduler_v8 requires include_source_frame=true")
```

说明：

- `T > E` 在 V8 语义下不可满足（无 future frame 可补齐）。
- 保持 fast-fail，避免“静默降级 + 隐式默认行为”。

---

## 12. 事件与可观测性

建议在 V8 事件中追加以下字段，便于排查与可解释性：

- `target_policy: "visited_episode_frames"`
- `visited_block_indices`
- `block_first_visit_order`
- `target_frame_indices`
- `target_image_refs`

至少在 `block_begin`/`block_end` 中对齐输出。

---

## 13. MultiSceneDatasetV4 集成点

训练侧建议新增：

- `datasets/train_scheduler_v8.py`
- `MultiSceneDatasetV4.create_train_scheduler_v8(...)`

验证侧建议新增：

- `datasets/validation_scheduler_v8.py`
- `build_validation_episode_specs_v8(...)`

`BatchRequest` 约束继续沿用现有实现：

- `enforce_target0_equals_source=True`
- `target_image_refs[0] == source_image_ref`

---

## 14. 迁移顺序（推荐）

1. 复制 `train_scheduler_v7.py` 到 `train_scheduler_v8.py`。  
2. 将 `episode_window_keyframes` 从 `E + T - 1` 改为 `E`。  
3. 删除/停用 `block_windows` 固定 target 构造。  
4. 引入 `visited_block_indices` 与 `_build_target_frames_for_block_v8()`。  
5. 改写 `_select_block()` 为动态 target。  
6. 更新 `block_begin/block_end` 事件字段。  
7. 新增 `validation_scheduler_v8.py` 与 visit-level target plan。  
8. preload 先采用“整条 episode chain 预热”稳定方案。  
9. 在配置工厂接入 `scheduler_v8` / `validation_v8`。  

---

## 15. 最小测试矩阵

建议先覆盖两个主场景（`E=3, T=3`）：

1. `block_major`  
   期望 target 序列：
   - `[f0]`
   - `[f1, f0]`
   - `[f2, f1, f0]`

2. `step_major` + `switch_interval=4`  
   期望 visit-level target 序列：
   - `[f0]`
   - `[f1, f0]`
   - `[f2, f1, f0]`
   - `[f0, f1, f2]`
   - `[f1, f0, f2]`
   - `[f2, f1, f0]`

并额外断言：

- 每次 `target[0] == source`
- target 长度 `<= T`
- target 元素来自当前 episode 的 `frame_chain`
- 首次访问某 block 时不包含未访问 block frame

---

## 16. 结论

V8 的核心不是更换执行顺序，而是修正 target 语义：

- 保留 V7 的 episode traversal 与 `step_major` 混合训练收益；
- 移除 future rolling target；
- 将监督严格限定为“当前 source + 当前 episode 已访问帧”。

这能显著降低“模型被迫学习未观测 future”带来的训练目标错配风险，并保持实现可迁移、可验证、可观测。
