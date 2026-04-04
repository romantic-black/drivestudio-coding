# TrainSchedulerV4 实现方案

本文档描述 **TrainSchedulerV4** 在仓库中的**具体落地方案**：以 `MultiSceneDatasetV3` 为数据契约，新建调度器类与工厂、扩展配置与 minimal trainer 日志/测试。概念层目标与采样原则与 [`StreetForward_Scheduler_V4_Design.md`](StreetForward_Scheduler_V4_Design.md) 对齐；数据层细节以 [`MultiSceneDataset_V3_Design.md`](../dataloader/MultiSceneDataset_V3_Design.md) 与实现 [`datasets/multi_scene_dataset_v3.py`](../../datasets/multi_scene_dataset_v3.py) 为准。

---

## 1. 文档范围

| 内容 | 本文档 | 备注 |
|------|--------|------|
| 为何采用 image-ref 主语义 | ✓ | 与 V3 设计文档一致 |
| 状态机字段、`next_batch` 流程 | ✓ | 可直接对照实现 |
| 新建 `TrainSchedulerV4` 与 `create_train_scheduler_v4()` | ✓ | 建议代码位置见 §16 |
| 扩展 YAML / minimal 训练脚本 / 单测 | ✓ | 基于现有 v3 范式 |
| 纯概念叙述（无实现细节） | 部分 | 更完整的背景见 `StreetForward_Scheduler_V4_Design.md` |

---

## 2. 前提：`MultiSceneDatasetV3` 契约

V3 已将 **image-ref `(frame_idx, cam_id)`** 作为规范原语，并提供：

- `SegmentIndex`：`num_cams`、`frame_indices`、`keyframe_indices`、`keyframe_to_frames`、`frame_to_keyframe`、`segment_first_frame_idx` 等（调度器应**复用索引**，不重复维护映射）。
- `get_segment_index(scene_id, segment_id) -> SegmentIndex`
- `validate_image_ref(scene_id, segment_id, image_ref, purpose)`
- `get_segment_batch_from_image_refs(request: BatchRequestV3, enforce_target0_equals_source=True)`
- `get_or_compute_pair_score(...)`（当前多为扩展点；`mode="none"` 时可返回 `None`）
- `build_preload_hint(...)`：向 scene 队列 / 预加载层提供 `future_image_refs`、`unique_frame_indices` 等

**Canonical batch 路径**是 `get_segment_batch_from_image_refs(BatchRequestV3)`，见 `multi_scene_dataset_v3.py` 文件头说明。

### 2.1 为何 V4 不能主走 `get_segment_batch_from_frames()`

`get_segment_batch_from_frames()` 是 **legacy**：对每个训练 **frame** 会扩展到 **所有相机**，语义仍是「frame 驱动 + all-cam 展开」，与 V4 的 **single-src-image + 显式 target 图像集合** 不一致。V4 训练循环必须调用 `get_segment_batch_from_image_refs()`，以保证调度语义与 tensor 契约一致。

### 2.2 Batch 硬约束（与数据集实现对齐）

- `target_image_refs[0] == source_image_ref`（`enforce_target0_equals_source=True` 时由数据集侧强校验）。
- 训练用 refs 经 `validate_image_ref(..., purpose="train")`。
- 若 `include_test=True` 且提供 `test_image_refs`，须经 `purpose="test"` 校验。
- 请求统一封装为 `BatchRequestV3`。

---

## 3. 设计目标与非目标

### 3.1 目标

1. 调度主语义从 **frame 级** 升级为 **image-ref 级**（输出 `source_image_ref` / `target_image_refs`）。
2. Segment 内保留清晰、可解释的 **时间层级** `U → B → R → S → N`（与 TrainSchedulerV3 的「时间尺度 + 事件流」思想一致，但采样原语不同）。
3. 为 **pair overlap 评分**、**preload**、更复杂 keyframe 策略预留 **hook**，第一版不把复杂逻辑写死在主循环。
4. **沿用**现有 V3 的配置键风格、`_emit()` / `pop_events()` **事件类型**；计数与预算在**对外暴露**时与 TrainSchedulerV3 一致采用 **raw-step 域**（见 §11.1），并**扩展** image-ref 字段（`source_image_ref` / `target_image_refs`）。
5. 与 `SegmentIndex` / `build_preload_hint` / `get_or_compute_pair_score` 自然对接。

### 3.2 非目标（第一版）

- 不实现真实几何 overlap 计算；仅接 `get_or_compute_pair_score` 与可选 `PairScorer` 协议。
- 不把 preload 做成独立后台线程系统；仅 **emit** `build_preload_hint` 结果（数据集已有 scene queue / preload / unload）。
- 默认 **1 个 source image per block**（非「一个 block 多个 source image」）。

---

## 4. 时间层级（与 V3 记法兼容）

```text
N: Next Scene
  └── S: Segment
        └── R: Reset Episode
              └── B: Source Block
                    └── U: Update Unit
```

- **U（Update Unit）**：由 `state_write_interval_steps` 定义；每累计 `U` 个 **raw training step** 计为 1 个 update unit（与 TrainSchedulerV3 中 `U` 含义一致）。
- **B（Source Block）**：固定 `source_image_ref` 与 `target_image_refs`，连续执行 `updates_per_block` 个 U（每个 U 对应 `state_write_interval_steps` 个 raw step，见 §11.1）。
- **R（Reset Episode）**：一次 reset 起始；在**连续 keyframe 窗口**内对 `(keyframe, cam)` **无放回遍历**，每个 pair 对应一个 block；episode 内 node state / hidden cache 不按 step 重置（与 V3「reset 事件」语义对齐，具体以 trainer 配置为准）。
- **S（Segment）**：点云 / dynamic / seg0 边界与现有 `MultiSceneDatasetV3` segment 缓存一致。
- **N（Next Scene）**：scene 切换继续依赖数据集 `scene_training_queue` 等；V4 只负责当前 scene/segment 上的 image-ref 调度。

---

## 5. 核心采样原则

### 5.1 Source（必须先 pair 后 frame）

1. 在 episode 窗口内选定 `(keyframe_idx, cam_id)`（来自 §6 的 pair 列表）。
2. 从 `SegmentIndex.keyframe_to_frames[keyframe_idx]` 中随机选 `frame_idx`。
3. `source_image_ref = (frame_idx, cam_id)`。

顺序不可颠倒：保证 **image 级调度**与 **以 keyframe、camera 为先** 的可控覆盖。

### 5.2 Target（默认策略）

给定 `source_image_ref = (f_src, cam_src)`，`kf_src = frame_to_keyframe[f_src]`，`T_total = total_target_images`：

- `target_image_refs[0] = source_image_ref`。
- `T_extra = T_total - 1` 个 extras 建议策略：
  1. 优先在 episode 窗口内、**除 `kf_src` 外** 的 keyframe 上选候选；
  2. 默认 **同 cam = `cam_src`**；
  3. 候选 keyframe 按与 `kf_src` 的索引距离 **近到远** 排序；
  4. 每个候选 keyframe 随机取一帧 → `(frame_tgt, cam_src)`；
  5. 窗口不足则扩展到 **segment 全域** 其他 keyframe；
  6. 仍不足则允许 **replacement**，但约束 `keyframe != kf_src`。

后续若接入 overlap，只需替换「候选排序器」，主状态机不变。

### 5.3 Episode：连续 keyframe 窗口 + pair 洗牌

- `segment_keyframes = sidx.keyframe_indices`，`R_kf = keyframes_per_episode`。
- 若 `len(segment_keyframes) > R_kf`：在 `[0, len(segment_keyframes) - R_kf]` 内随机 `start`，`window = segment_keyframes[start : start + R_kf]`。
- 否则 `window = segment_keyframes`（全窗）。
- `pair_list = [(kf, cam) for kf in window for cam in range(sidx.num_cams)]`，再 `random.shuffle(pair_list)`，保证 episode 内 **无放回随机遍历**。

---

## 6. 内部状态（建议字段）

在 TrainSchedulerV3 的 segment 状态基础上，**显式纳入相机与 image-ref**：

```python
# 示意：内部可维护 U 域计数；对外 get_current_info / 日志见 §11.1（raw-step 域）
current_segment_state = {
    "scene_id": int,
    "segment_id": int,
    "segment_local_u": int,       # 内部：已完成 update units 数
    "segment_budget_u": int,      # 本 segment 的 U 预算（见 §8）
    "episode_idx_in_segment": int,
    "block_idx_in_episode": int,
    "block_idx_in_segment": int,
    "block_idx_global": int,
    "episode_window_keyframes": List[int],
    "pair_list": List[Tuple[int, int]],  # (keyframe_idx, cam_id)
    "pair_cursor": int,
    "source_keyframe_idx": int,
    "source_frame_idx": int,
    "source_cam_idx": int,
    "source_image_ref": Tuple[int, int],
    "target_image_refs": List[Tuple[int, int]],
    "u_in_block": int,
}
```

`get_current_info()` 应暴露 **raw-step 域**步数与预算（与 V3 对齐，§11.1），以及 `source_image_ref`、`target_image_refs`、`source_cam_idx` 等，供日志与 `model_node_state` 同步逻辑使用。

---

## 7. 调度流程概要

1. **进入 segment**：`get_segment_index` → 计算 `U_seg`（§8）→ `segment_begin` → 开启首个 episode。
2. **进入 episode**：`reset_episode_idx` 递增 → 采样 window → 构造 `pair_list` → `pair_cursor = 0` → 发 **`reset_event`**（§11.2，与 TrainSchedulerV3 事件类型名一致）。
3. **进入 block**：若 `pair_cursor` 耗尽则判断是否需要 **同 segment 内下一 episode**（受 `episodes_per_segment` **上界**与 §8 预算规则约束）；否则取 `pair`，采样 frame 与 targets → `u_in_block = 0` → `block_begin`；可选 `preload_hint`。
4. **`next_batch()`**：若需切换 epoch/segment 则先处理边界；否则 `get_segment_batch_from_image_refs(BatchRequestV3(...))`；递增 `global_step`；按 §11.1 更新 raw-step 与 U 域计数；满 `updates_per_block` 个 U 则 `block_end` 并进入下一 block；**segment 结束条件仅由预算决定**（§8），见下。

---

## 7.1 `segment_budget_u`、`updates_per_block`、`episodes_per_segment` 与 pair 规模的关系（必须实现的规则）

下列量同时出现在配置与状态中，**必须**在文档与实现中采用同一套优先级，否则会出现「预算优先」与「episode 数优先」两种分叉实现。

记：

- `U_seg`：本 segment 的 update-unit 预算（§8 中 `segment_budget_u`）。
- `U = state_write_interval_steps`：每个 update unit 含 `U` 个 raw training step。
- `R_kf = keyframes_per_episode`，`W = len(window)`（通常 `W = R_kf`，若 segment keyframe 不足则 `W < R_kf`）。
- 每个 episode：`num_pairs = W * num_cams`。
- 每个 block 消耗 `updates_per_block` 个 U（每个 U 含 `U` 个 raw step）。

**拍板规则（预算绝对优先）：**

1. **`segment_budget_u`（即 `U_seg`）是唯一硬结束条件**：当且仅当 `segment_local_u >= segment_budget_u` 时结束当前 segment，进入 `segment_end` 并推进 epoch plan。
2. **`episodes_per_segment` 是上界，不是必须跑满的配额**：表示在同一 segment 内**至多**开启的 episode 个数（从 0 计数到 `episodes_per_segment - 1`）。若在达到该上界之前已满足 `segment_local_u >= segment_budget_u`，则 **提前结束 segment**，不再开启新 episode。
3. **允许被预算截断**：
   - 最后一个 **episode** 可以只消费 `pair_list` 的**前缀**（`pair_cursor` 未走到 `num_pairs` 即因预算结束而终止）。
   - 最后一个 **block** 可以只消费不足 `updates_per_block` 个 U：若剩余 `segment_budget_u - segment_local_u`（以 U 计）小于 `updates_per_block`，则在累计满剩余 U 后 `block_end` 并 `segment_end`，**不要求**最后一个 block 凑满完整的 `updates_per_block`。
4. **预算必须能在「至多 `episodes_per_segment` 个 episode」内花完（推荐 fast-fail）**：记 `W_max = min(keyframes_per_episode, len(segment_keyframes))`，单 segment 内 **理论最大 U 消耗能力**（在永不提前因预算截断的前提下）约为 `episodes_per_segment * W_max * num_cams * updates_per_block`。若 `segment_budget_u` **大于**该上界，则存在「episode 次数用尽而预算仍未达标」的死区。第一版实现应在 **进入 segment 时** `raise ValueError`（或等价 fast-fail），要求用户增大 `episodes_per_segment` / `keyframes_per_episode` 或减小 `segment_budget_u`。**不**在第一版引入「无界重复 episode 直至预算耗尽」的隐式循环，以免与 `episodes_per_segment` 语义打架。

若未来需要支持「极小 `episodes_per_segment` + 大预算」，应改为把 `episodes_per_segment` 改为**由预算派生**的配置方案（见 §8 注），而不是在运行时默默超发 episode。

**与 TrainSchedulerV3 的对比**：V3 通过 `S_u_raw → K_u → B_seg → S_u_final` 把 segment 步数与 block 尺度**算死**在同一套推导里。V4 用 **U 域预算 + 可截断 episode/block** 换取 image-ref 调度灵活性；对外仍用 raw-step 域报告进度（§11.1），以便与现有 trainer 字段名对齐。

---

**Batch 构造必须是：**

```python
batch = dataset.get_segment_batch_from_image_refs(
    BatchRequestV3(
        scene_id=scene_id,
        segment_id=segment_id,
        source_image_ref=source_image_ref,
        target_image_refs=target_image_refs,
        include_test=include_test,
        test_image_refs=test_image_refs,
    ),
    enforce_target0_equals_source=True,
)
```

---

## 8. Segment 预算（建议）

与 V3「按 keyframe 数推导 segment 更新预算」一致，建议：

```python
U_seg = clamp(
    round(alpha_updates_per_keyframe * num_keyframes),
    min_updates_per_segment,
    max_updates_per_segment,
)
```

具体 `clamp` 与 epoch plan 中每项字段（如 V3 的 `S_u_final`、`B_seg`）是否仍写入 plan，可由实现选择与 trainer 日志对齐；**至少**需 `segment_budget_u`（即 `U_seg`）供 `segment_end` 判断。

**注（与 `episodes_per_segment` 的替代关系）**：若不想维护 §7.1 第 4 点的 fast-fail，可将 `episodes_per_segment` 改为**派生量**（由 `segment_budget_u`、window 规模与 `updates_per_block` 反推下限），使 episode 数与预算天然相容；第一版文档采用 **独立配置项 + 进入 segment 时校验** 的方案，便于与现有 YAML 心智对齐。

---

## 9. Overlap 与 Preload 扩展点

### 9.1 Pair 排序 / overlap

- 配置项如 `overlap.mode`：默认 `"none"`，候选按 `|kf_tgt - kf_src|` 排序。
- 非 none 时调用 `dataset.get_or_compute_pair_score(...)`，以 score 为主排序键、距离为次键。

可选 Protocol（实现放在调度模块或独立小文件）：

```python
class PairScorer(Protocol):
    def score(
        self,
        dataset: MultiSceneDatasetV3,
        scene_id: int,
        segment_id: int,
        src: ImageRef,
        tgt: ImageRef,
    ) -> Optional[float]: ...
```

### 9.2 Preload hint

在 **`reset_event`** 之后（新 episode 的 window / pair 已确定，可枚举未来可能访问的 frame）与 **`block_begin`** 之后（可枚举下一 block 的候选 refs）调用 `dataset.build_preload_hint(scene_id, segment_id, future_image_refs=[...])`，将结果放入 **`preload_hint` 事件**（见 §11.3），不强制数据集立即加载。

---

## 10. 配置设计（`scheduler_v4`）与工厂一一对应

第一版 **删掉尚未实现、否则易成「死配置」的键**（方案 A）：不在 YAML 里写 `allow_cross_cam_when_insufficient`、`overlap.use_for_*`、`cache_scores`、`preload.lookahead_*`、`traversal.shuffle_*`、`eval.include_test` 等；其中 **`include_test`** 由 minimal 脚本的既有逻辑（如 `one_segment` / `eval`）传入工厂，**不**重复出现在 `scheduler_v4` 里。

建议在 [`configs/minimal_streetforward_stage4_1_one_segment_v3_s1.yaml`](../../configs/minimal_streetforward_stage4_1_one_segment_v3_s1.yaml) 范式上**新增** `scheduler_v4`（或专供 V4 的 yaml）。**与 `create_train_scheduler_v4()` 可逐项映射**的示例如下（仅包含第一版会消费的键）：

```yaml
scheduler_v4:
  enable: true
  time_base:
    state_write_interval_steps: 1
  segment_budget:
    alpha_updates_per_keyframe: 8.0
    min_updates_per_segment: 24
    max_updates_per_segment: 160
  source_block:
    updates_per_block: 2
  reset_episode:
    keyframes_per_episode: 3
    episodes_per_segment: 2
    keyframe_window_policy: random_contiguous_window
    pair_order_policy: shuffle_without_replacement
  target_sampling:
    total_target_images: 4
    include_source: true
    extra_target_policy: same_cam_different_keyframe
    prefer_nearby_keyframes: true
    fallback_expand_to_segment: true
    fallback_with_replacement: true
  overlap:
    mode: none
  preload:
    emit_hints: true
  traversal:
    fixed_scene_id: null
    fixed_segment_id: null
```

**映射说明**：

- `enable`：由 trainer 读取；为 false 时不走 V4 分支；**不**传入工厂。
- `state_write_interval_steps`：同 V3 `scheduler_v3.time_base.state_write_interval_steps`（记为 `U`，raw step / update unit 换算见 §11.1）。
- `alpha_updates_per_keyframe` / `min_updates_per_segment` / `max_updates_per_segment`：推导 `segment_budget_u`（§8）。
- `updates_per_block`：block 内固定 source/target 时的 **U 个数**；V4 第一版**不**引入 V3 的 `target_hold_updates` / `freeze_target_within_block`。
- `keyframe_window_policy` / `pair_order_policy`：字符串枚举，第一版实现 `random_contiguous_window` 与 `shuffle_without_replacement`；未支持的取值应 **fast-fail**。
- 日后若重新引入被删键，必须**同时**加入 `create_train_scheduler_v4(...)` 参数与实现分支，禁止只在 YAML 中存在。

---

## 11. 事件与日志

保留 V3 的 **`_emit` / `pop_events`** 模式；训练脚本在 step 循环中 `pop_events()` 并写 logger / jsonl。

### 11.1 计数口径：对外以 raw-step 域为主（与 TrainSchedulerV3 对齐）

TrainSchedulerV3 的 `get_current_info()` 与常用日志字段以 **raw training step** 为「主刻度」：`global_step`、`segment_local_step`、`segment_step_budget`，以及已乘上 `U` 的 `K_steps` / `R_steps` / `T_steps` 等。

V4 **内部**可用 `segment_local_u`、`u_in_block`、`segment_budget_u`（U 域）推进状态机，但 **`get_current_info()` 与写入 jsonl 的主字段应与 V3 同口径**，避免现有可视化 / 对比脚本误读：

- `global_step`：累计 raw step（每次 `next_batch` 调用 +1 或与现有一致）。
- `segment_local_step`：当前 segment 内已执行的 **raw step** 数（或等价定义，须在实现注释中写死）。
- `segment_step_budget`：本 segment 的 raw-step 预算，建议 **`segment_budget_u * state_write_interval_steps`**（与 V3「步数预算 = U 域预算 × U」一致）。
- 可选：同时提供 `segment_local_u`、`segment_budget_u`、`U`（`state_write_interval_steps`）作为**辅助字段**，便于对照设计文档中的 U 域叙述；但**不得**用 U 域字段替代 `segment_local_step` / `segment_step_budget` 作为主口径，除非全仓库 trainer 与文档同步迁移。

**结论**：事件 **类型名**沿用 V3 习惯；**数值口径**以 raw-step 为主 —— 不是「沿用 V3 日志结构」的模糊说法，而是明确 **与 V3 一致的步数域约定**，并额外挂载 image-ref 与 U 域辅助字段。

### 11.2 `reset_event` 命名（与 TrainSchedulerV3 统一）

- **事件 `type` 一律使用 `reset_event`**，与 [`TrainSchedulerV3`](../../datasets/multi_scene_dataset_v2.py) 一致；**不使用** `reset_begin` 作为事件类型名（避免 trainer / 测试出现第三套分支）。
- V4 在每个 **reset episode 开始**（新 window、新 `pair_list`）发 `reset_event`，`reason` 建议为 **`"episode_begin"`**（与 V3 使用的 `"segment_enter"`、`"source_block_boundary"` 区分）。
- 文档叙述中的「reset episode 开始」均指上述 **`reset_event`**，不另设别名事件。

### 11.3 事件 payload 表

| 类型 | 关键 payload |
|------|----------------|
| `segment_begin` | `epoch_idx`, `global_step`, `scene_id`, `segment_id`, `num_keyframes`, `num_cams`, `segment_budget_u`, `segment_step_budget`（raw，建议）、`updates_per_block`, `keyframes_per_episode`, `episodes_per_segment`, `total_target_images`, `U` |
| `reset_event` | `reset_episode_idx`, `window_keyframes`, `num_pairs`, `reason`（如 `"episode_begin"`） |
| `block_begin` | `block_idx_in_episode`, `block_idx_in_segment`, `block_idx_global`, `source_keyframe_idx`, `source_frame_idx`, `source_cam_idx`, `source_image_ref`, `target_image_refs`, `U`, `updates_per_block` |
| `block_end` | `source_image_ref`, `num_updates_in_block`（U 域或同时给 raw，二选一时优先与 `get_current_info` 一致） |
| `preload_hint` | `hint`（`build_preload_hint` 返回值） |
| `segment_end` | 与 V3 类似；可含 `source_image_ref` 最后值等可选摘要 |

**Minimal trainer 扩展**：

- 在 [`train_minimal_streetforward_stage4_3_one_segment_v3.py`](../../tools/train_minimal_streetforward_stage4_3_one_segment_v3.py)（或并行新增 `*_v4.py`）中：`pop_events()` 分支对 `reset_event` 使用**与 v3 相同**的 `type` 判断；payload 通过 `reason` 区分 episode 与 V3 的 segment/block 边界。
- jsonl `scheduler_info`：在 V3 的 `source_frame_idx` / `source_keyframe_idx` 之外，增加 `source_image_ref` / `target_image_refs`；**主步数字段**仍用 `segment_local_step` / `segment_step_budget`（raw-step 域）。

---

## 12. 工厂接口

在 `MultiSceneDatasetV3` 上新增；**参数与 §10 YAML 可逐项对应**（外加 trainer 传入的 `include_test`），避免「配置里有、工厂没有」或 trainer 侧偷偷塞私有参数。

```python
def create_train_scheduler_v4(
    self,
    *,
    state_write_interval_steps: int,
    alpha_updates_per_keyframe: float,
    min_updates_per_segment: int,
    max_updates_per_segment: int,
    updates_per_block: int,
    keyframes_per_episode: int,
    episodes_per_segment: int,
    keyframe_window_policy: str,
    pair_order_policy: str,
    total_target_images: int,
    include_source: bool,
    extra_target_policy: str,
    prefer_nearby_keyframes: bool,
    fallback_expand_to_segment: bool,
    fallback_with_replacement: bool,
    overlap_mode: str,
    emit_preload_hints: bool,
    include_test: bool,
    fixed_scene_id: Optional[int],
    fixed_segment_id: Optional[int],
) -> "TrainSchedulerV4":
    ...
```

`_build_train_scheduler_v4(cfg, dataset_v3, scene_id, segment_id, include_test)`：从 `cfg["scheduler_v4"]` 解析上表参数，**缺键或未知枚举则 fast-fail**；`include_test` 来自与现有一致的 `one_segment` / eval 解析，**不**从 `scheduler_v4` 嵌套读取。

---

## 13. 类骨架（伪代码）

```python
class TrainSchedulerV4:
    def __init__(..., dataset: MultiSceneDatasetV3, ...):
        validate_config()
        self.dataset = dataset
        self.U = state_write_interval_steps
        # epoch_idx, global_step, epoch_plan, plan_cursor, current_segment_state,
        # _pending_events, _block_idx_global, _reset_episode_idx
        if not dataset._initialized:
            dataset.initialize()
        self.start_new_epoch()

    def pop_events(self) -> List[Dict[str, Any]]: ...
    def get_current_info(self) -> Dict[str, Any]: ...
    def next_batch(self) -> Dict[str, Any]:
        # 仅通过 get_segment_batch_from_image_refs + BatchRequestV3
        ...
```

详细分支逻辑见 §7 / §7.1；实现时注意 **epoch 结束**、**预算截断** 与 **§7.1 第 4 点 fast-fail**（避免 episode 上界与预算不相容）。

---

## 14. 测试设计

建议新增或扩展 [`tests/test_multi_scene_dataset_v3.py`](../../tests/test_multi_scene_dataset_v3.py)（或 `tests/test_train_scheduler_v4.py`），用 **mock `MultiSceneDatasetV3`** 断言调用路径与事件序列。

### 14.1 采样与索引

- `test_v4_source_sampling_is_pair_then_frame`：先 `(kf, cam)` 再 `frame`。
- `test_v4_episode_window_is_contiguous`：窗口连续。
- `test_v4_pair_list_shuffle_without_replacement`：episode 内 pair 无重复、长度正确。
- `test_v4_target0_equals_source`。
- `test_v4_targets_default_same_cam`：extras 默认 `cam_src`。
- `test_v4_targets_use_other_keyframes`：extras 不来自 `kf_src`。
- `test_v4_fallback_expand_to_segment` / `test_v4_fallback_with_replacement`。
- `test_v4_segment_budget_truncates_last_episode_and_block`：验证 §7.1 — `segment_local_u >= segment_budget_u` 时允许未完成整段 `pair_list` 或不足 `updates_per_block`。
- `test_v4_config_or_enter_segment_rejects_oversized_budget`：验证 §7.1 第 4 点 — `segment_budget_u` 大于 `episodes_per_segment * W_max * num_cams * updates_per_block` 时 fast-fail。

### 14.2 Batch 契约

- `test_v4_next_batch_calls_get_segment_batch_from_image_refs`。
- `test_v4_never_calls_get_segment_batch_from_frames`（mock 断言未调用 legacy）。
- `test_v4_batch_request_single_source_image_ref`。
- `test_v4_explicit_test_image_refs_validated`（`include_test=True` 时）。

### 14.3 事件

- `test_v4_emits_segment_begin_reset_event_block_begin`：`reset_event` 的 `type` 与 TrainSchedulerV3 一致，`reason=="episode_begin"`。
- `test_v4_emits_block_end_after_updates_per_block`。
- `test_v4_pop_events_clears_queue`。
- `test_v4_get_current_info_uses_raw_step_fields`：`segment_local_step` / `segment_step_budget` 与 §11.1 一致。

---

## 15. 与 TrainSchedulerV3 的差异（实现视角）

| 维度 | V3 | V4 |
|------|----|----|
| 数据集类 | `MultiSceneDatasetV2` + frame 级 batch 路径 | `MultiSceneDatasetV3` + `get_segment_batch_from_image_refs` |
| 主输出 | `source_frame_idx` / `target_frame_indices` | `source_image_ref` / `target_image_refs` |
| Block 内 target | 可有 `target_hold_updates`、`freeze_target_within_block` | 第一版固定 lists，无 hold/freeze |
| Episode / reset | `reset_policy` 等与 source block 对齐；`reset_event` | 连续 keyframe 窗口 + `(kf,cam)` 无放回；**仍发 `reset_event`**（`reason=episode_begin`） |
| 日志步数主口径 | raw-step：`segment_local_step` 等 | **同为 raw-step 主口径**（§11.1），另可加 U 域辅助字段 |

---

## 16. 仓库改动清单（建议顺序）

1. **`datasets/multi_scene_dataset_v3.py`**：实现 `TrainSchedulerV4` 类（或 `datasets/train_scheduler_v4.py` 仅从 v3 import 类型）、`create_train_scheduler_v4`。
2. **`tests/`**：§14 中单测 + mock。
3. **`configs/`**：在 `minimal_streetforward_stage4_1_one_segment_v3_s1.yaml` 上扩展或新增 `*_v4*.yaml`，写入 `scheduler_v4`。
4. **`tools/train_minimal_streetforward_stage4_1_one_segment_v3.py`**：增加 `_build_train_scheduler_v4` 与配置分支；或 **新建** `train_minimal_streetforward_stage4_3_one_segment_v4.py` 专走 V4，减少对现有 v3 实验的回归面。
5. **`tools/train_minimal_streetforward_stage4_3_one_segment_v3.py`**：当选用 V4 时切换 dataset/scheduler 与日志字段（若与 4 合并为单脚本，则用 `cfg.scheduler_v4.enable` 分支）。

---

## 17. 一句话定义

**TrainSchedulerV4**：在每个 segment 内以 **`segment_budget_u` 为硬结束条件**（`episodes_per_segment` 为上界，见 §7.1）；每个 **reset episode** 在**连续 keyframe 窗口**上构造 `(keyframe, cam)` 对并 **shuffle 无放回**；每个 **source block** 先固定 `(keyframe, cam)` 再随机 `frame` 得到 `source_image_ref`，再按「`target[0]=source`、extras 默认同 cam 异 keyframe」生成 `target_image_refs`；一律通过 **`MultiSceneDatasetV3.get_segment_batch_from_image_refs(BatchRequestV3)`** 取 batch；以 **`reset_event` / `block_begin` / …** 与 V3 **同名事件类型**暴露边界，**主步数口径为 raw-step**（§11.1），并附带 image-ref 字段。

---

## 18. 相关路径索引

- 数据与 batch：`datasets/multi_scene_dataset_v3.py`
- 设计说明：`docs/dataloader/MultiSceneDataset_V3_Design.md`
- 概念层 V4：`docs/trainers/StreetForward_Scheduler_V4_Design.md`
- 配置参考：`configs/minimal_streetforward_stage4_1_one_segment_v3_s1.yaml`
- Minimal 训练入口：`tools/train_minimal_streetforward_stage4_3_one_segment_v3.py`、`tools/train_minimal_streetforward_stage4_1_one_segment_v3.py`
- 现有 TrainSchedulerV3 实现：`datasets/multi_scene_dataset_v2.py`（`TrainSchedulerV3` 类）
