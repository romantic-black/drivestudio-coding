# TrainSchedulerV10 独立化与结构化 Step 重构计划

## 文档目标

本文在以下实现与文档的对照基础上，给出将 **TrainSchedulerV10** 从「V9 继承 + request_meta 补丁」演进为 **独立调度器 + 结构化 Step 生成** 的详细重构计划：

- 现状：`datasets/train_scheduler_v10.py`（继承 `TrainSchedulerV9`，在 `materialize_current_batch_without_advance` / `next_batch` 后打补丁）
- 基线：`datasets/train_scheduler_v9.py`（`teacher_prob` / `student_prob`、`role_sampling`、`targets.weights` 等）
- 消费端：`models/streetforward/minimal_trainer_stage6_0.py`（仍大量依赖 `stage6_role`、`train_target_*`、`scheduler_request_v10.live_teacher_bridge` 等）

设计取向与 [StreetForward_Stage6_0_Decoupled_Teacher_Student_Design.md](../trainers/StreetForward_Stage6_0_Decoupled_Teacher_Student_Design.md) 一致：域命名清晰、bridge 边界可验证、**probe 默认不进总 loss**。

---

## 1. 现状差距（Why refactor）

### 1.1 V10 当前仍是「V9 + 后处理」

`TrainSchedulerV10` 不拥有 episode/block 的语义核心逻辑，仅在父类产出 batch 后调用 `_patch_request_meta_to_v10`：

- 将 V9 的 `teacher_preserve` / `visited` / `near_random` 映射为 `teacher_anchor` / `history_visited` / `probe_near`
- 从扁平 `target_*` 拆出 `train_target_*` 与 `probe_target_*`
- 构造嵌套的 `scheduler_request_v10`（`teacher_obs`、`student_prop`、`live_teacher_bridge` 等）

**问题**：step 语义仍由 V9 的随机 role（`teacher_prob` / `student_prob`）与 `force_teacher_on_block_entry` 等隐式规则决定；V10 层无法保证「某一步一定是 student_anchor」这类**可重复、可配置**的程序语义。

### 1.2 Trainer 仍以「扁平 role + teacher/student 二值」驱动

`minimal_trainer_stage6_0.py` 中：

- `_get_stage6_role` 读 `request_meta.stage6_role` / `stage5_5_role`（teacher vs student）
- `_compute_2d_features_all_branches_once_routed` 在 student 分支用 `scheduler_request_v10.live_teacher_bridge` 决定是否重跑 teacher 2D；**teacher 路径上的 cache 更新**与 **live bridge 的 teacher 2D** 混在同一套 `role == "teacher"` 分支里，仅靠 meta 区分，缺少显式 `purpose`（见下文 5.2）
- `_build_target_view_weights` 在 v10/v9 下仍从 `train_target_image_roles` / `target_image_roles` 列表推权重

**问题**：Stage6 loss 域（`losses.stage6_0`）与 scheduler 的「谁在什么时候写 history / cache」没有单一真相来源；后续维护成本高。

### 1.3 配置面仍携带 V9 遗留键

Stage6_0 校验已禁止部分 preserve 命名，并对 `scheduler_v10.targets.weights` _train 域_ 要求全为 1.0、probe 为 0（非 explicit 阶段）；但 **V9 的 `role_sampling`、`near_random_supervision` 等仍可能存在于 YAML**，与「V10 独立配置」目标不一致。

---

## 2. V10 独立后的核心目标

V10 **不再**以「一个 source + 一组 target roles」为唯一输出契约。

V10 的权威输出应为 **`Stage6StepRequest`（或等价 dict / `request_meta` 顶层结构）**，显式包含：

| 区块 | 含义 |
|------|------|
| `teacher_obs` | 是否执行 teacher observation；**目的**（训练更新 vs live_bridge）；各写回开关 |
| `student_prop` | 是否执行 student propagation；bridge 模式（live / cache / none）；写回开关 |
| `supervision` | 各监督域：`self_teacher` / `self_student` / `teacher_anchor` / `history_visited` / `probe_near`（仅域开关与 refs，**不放真实 loss 权重**） |
| `history_record` | observed / runtime 的触发与 commit 策略 |
| `preload_hints` | 当前 step 所需的全部 frame / image refs（与 dataset preload 对齐） |

**核心从「role sampling」转为「结构化 step generation」**：每一步的 teacher/student/supervision 组合由 **step program** 决定，而非概率表。

---

## 3. 建议数据结构（实现阶段落盘）

以下为建议的 dataclass 草图（可与 `datasets/stage6_step_request.py` 一类模块同存）。

### 3.1 `ImageRef` 与 MultiScene V4 契约（写死，禁止含糊）

`datasets/multi_scene_dataset_v4.py` 已定义并与 `_load_image_meta` / `get_segment_batch_from_image_refs` 一致：

```python
ImageRef = Tuple[int, int]  # (frame_idx, cam_id)
```

- `scene_id`、`segment_id` **不**编入 tuple；它们出现在 `Stage6StepRequest` 顶层（及 `BatchRequestV4` / `request_meta`），与 `MultiSceneDatasetV4._assemble_segment_batch_from_image_refs(scene_id, segment_id, ...)` 的调用方式一致。
- `Stage6StepRequest`、一切 `preload_hints`、`supervision.*.image_refs` 中的引用 **必须** 使用同一 `ImageRef` alias；`_materialize_batch_from_request` 实现前应在类型层 `TypeAlias` 导出并单测「tuple 长度与语义」。

若未来需要 segment 内局部索引，应 **新增命名类型**（例如 `KeyframeLocalRef`），而不是重载 `ImageRef` 元组位置语义。

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

ImageRef = Tuple[int, int]  # (frame_idx, cam_id) — 与 multi_scene_dataset_v4.ImageRef 一致


class Stage6StepType(str, Enum):
    TEACHER_BOOTSTRAP = "teacher_bootstrap"
    STUDENT_SELF = "student_self"
    STUDENT_ANCHOR = "student_anchor"
    STUDENT_HISTORY = "student_history"
    TEACHER_REFRESH = "teacher_refresh"


@dataclass
class Stage6TeacherObsRequest:
    enable: bool
    frame_idx: Optional[int] = None
    image_refs: List[ImageRef] = field(default_factory=list)
    purpose: str = "train_update"  # train_update | live_bridge
    # 默认全 False：避免漏设 purpose 时误更新 cache/state/history。
    # teacher_bootstrap / teacher_refresh 的 builder 必须显式打开所需开关。
    update_state: bool = False
    update_teacher_prior_cache: bool = False
    update_observed_history: bool = False
    update_runtime_history: bool = False
    use_gt_input: bool = True


def validate_teacher_obs_invariants(obs: Stage6TeacherObsRequest) -> None:
    if obs.purpose == "live_bridge":
        if obs.update_state or obs.update_teacher_prior_cache or obs.update_observed_history or obs.update_runtime_history:
            raise ValueError(
                "teacher_obs purpose=live_bridge forbids any state/cache/history writes; "
                f"got update_state={obs.update_state}, cache={obs.update_teacher_prior_cache}, "
                f"observed={obs.update_observed_history}, runtime={obs.update_runtime_history}"
            )


@dataclass
class Stage6StudentPropRequest:
    enable: bool
    frame_idx: Optional[int] = None
    image_refs: List[ImageRef] = field(default_factory=list)
    bridge_mode: str = "live"  # live | cache | none
    require_live_bridge: bool = True
    update_state: bool = True
    update_runtime_history: bool = True
    update_observed_history: bool = False
    use_gt_input: bool = False


@dataclass
class Stage6SupervisionRequest:
    self_teacher_refs: List[ImageRef] = field(default_factory=list)
    self_student_refs: List[ImageRef] = field(default_factory=list)
    teacher_anchor_refs: List[ImageRef] = field(default_factory=list)
    history_visited_refs: List[ImageRef] = field(default_factory=list)
    probe_near_refs: List[ImageRef] = field(default_factory=list)

    enable_self_teacher: bool = False
    enable_self_student: bool = False
    enable_teacher_anchor: bool = False
    enable_history_visited: bool = False
    enable_probe_near: bool = False


@dataclass
class Stage6StepRequest:
    scheduler_version: str
    step_type: Stage6StepType
    scene_id: int
    segment_id: int
    block_idx: int
    step_idx_in_block: int
    global_scheduler_step: int

    teacher_obs: Stage6TeacherObsRequest
    student_prop: Stage6StudentPropRequest
    supervision: Stage6SupervisionRequest

    teacher_anchor_frame_idx: Optional[int] = None
    student_frame_idx: Optional[int] = None
    committed_history_frame_indices: List[int] = field(default_factory=list)
    probe_near_frame_indices: List[int] = field(default_factory=list)

    compat: Dict[str, object] = field(default_factory=dict)
```

**构造后校验**：`_build_step_request` 在返回前应对 `teacher_obs` 调用 `validate_teacher_obs_invariants`（或对等价 dict 做相同断言）。`purpose == "train_update"` 且 `enable` 时，由对应 step builder **显式**打开需要为 true 的 `update_*` 开关；不在 dataclass 默认值里「偷偷」为 true。

**与当前补丁式 meta 的映射关系**：

- 现有 `scheduler_request_v10.teacher_obs` / `live_teacher_bridge` 合并进 **`teacher_obs` + `purpose`**，避免「live bridge 被误当成 teacher 训练步」。
- 现有 `train_targets` / `probe_targets` 由 **`supervision` + `history_record` 策略** 直接生成，不再依赖 V9 先产全量再过滤。

---

## 4. EpisodePlanV10 与 BlockStateV10

### 4.1 不继承 V9 的 Episode 契约

`EpisodePlanV10` **不从 V9 继承**；可参考 `datasets/train_scheduler_v8.py` 中的 `EpisodePlanV8`（字段：`scene_id`、`segment_id`、`episode_start_keyframe_pos`、`keyframe_window`、`frame_chain`、`num_cams`），由 V10 **自行扩展** block 级规划字段：

```python
@dataclass
class EpisodePlanV10:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int
    keyframe_window: List[int]
    frame_chain: List[int]
    num_cams: int

    block_keyframe_indices: List[int]
    teacher_frame_by_block: Dict[int, int]
    student_candidates_by_block: Dict[int, List[int]]
    probe_near_candidates_by_block: Dict[int, List[int]]
```

**复用策略**：V7/V8 的 dataset 遍历、keyframe 窗口、`TrainSchedulerDatasetV7` 接口可继续复用；**episode 计划对象与 block 状态机**在 V10 模块内自建，避免拖入 `role_sampling_cfg`。

### 4.2 EpisodeStateV10（`committed_history_frame_indices` 的权威归属）

**`committed_history_frame_indices` 必须挂在 Episode 级状态上**，而不是「当前 block 内随时增长」的模糊列表，否则 `teacher_anchor` 与 `history_visited` 会再次混用同一帧。

语义（写死）：

- **定义**：`committed_history_frame_indices` = 已完成 block 的 **teacher anchor / teacher 观测帧**（即该 block 的 `teacher_frame_idx`），在 **block exit** 时一次性 commit。
- **当前 block**：该 block 的 `teacher_frame_idx` **只**作为 `teacher_anchor` 监督引用；**不得**进入本 block 内任意 step 的 `history_visited` 采样池。
- **下一 block 及之后**：上一 block（及更早）已 commit 的 teacher 帧才可出现在 `student_history` 的 `history_visited` 候选中。

推荐 `_commit_block` 行为：

```python
def _commit_block(self) -> None:
    if self.current_block_state.teacher_seen:
        self.episode_state.committed_history_frame_indices.append(
            int(self.current_block_state.teacher_frame_idx)
        )
    # 再推进 block_idx / 重置 BlockStateV10
```

`Stage6StepRequest.committed_history_frame_indices` 字段应为 **构造该 step 时 episode 状态的只读快照**（供 trainer / 日志对齐），与 `episode_state` 同源。

### 4.3 `student_history` 与空 committed 的策略（禁止 silent empty）

| 阶段 | 程序步为 `student_history` 且 `committed_history_frame_indices` 为空 |
|------|------------------------------------------------------------------------|
| **warmup** | 允许 **降级** 为 `student_self`（仍产生有效 self_student 监督）；须在 `request_meta` 中打标，例如 `scheduler_v10/fallback_no_history_applied=1` |
| **正式训练** | **禁止** silent empty：要么 `fast-fail`，要么 **不降 step 类型** 但 `enable_history_visited=false` 并递增 `skip_history_target_count` / 写入 perf，使监控明确显示「本步无 history 可训」——**不得**假装算了 history loss |

实现须在调度器或 trainer 侧统一约定一种行为并在文档与配置中写死（推荐正式阶段 fast-fail 或显式计数二选一，避免第三种「loss=0 无日志」）。

### 4.4 BlockStateV10

每个 block 维护本 block 的步进与 runtime 痕迹；**committed 列表不放在 block 内作为权威源**（权威在 `EpisodeStateV10`）。

```python
@dataclass
class EpisodeStateV10:
    committed_history_frame_indices: List[int] = field(default_factory=list)


@dataclass
class BlockStateV10:
    block_idx: int
    keyframe_idx: int
    teacher_frame_idx: int
    student_candidates: List[int]

    step_idx: int = 0
    teacher_seen: bool = False
    student_seen: bool = False

    runtime_updated_frames: List[int] = field(default_factory=list)

    last_teacher_frame_idx: Optional[int] = None
    last_student_frame_idx: Optional[int] = None
```

**与 V9 差异**：V9 的 `visited` 采样可混入 near/anchor；V10 要求 **`history_visited` 仅来自 `episode_state.committed_history_frame_indices`（block exit 后）**，采样时再排除当前 student source、`teacher_anchor` 帧、probe 帧（与用户规格及 Stage6 文档一致）。

**说明**：student 更新后的帧是否进入「history」由 **history_record / runtime** 策略单独定义（例如 runtime 列表用于诊断）；**不得**与 `committed_history_frame_indices` 混用，以免与 teacher-anchor commit 语义冲突。

---

## 5. Step program（取代 teacher_prob / student_prob）

### 5.1 配置形态

不再使用：

```yaml
role_sampling:
  teacher_prob: 0.4
  student_prob: 0.6
```

改为显式序列，例如每 block 12 step，`mode: fixed_cycle`：

```yaml
scheduler_v10:
  block:
    steps_per_block: 12
  step_program:
    mode: fixed_cycle
    sequence:
      - teacher_bootstrap
      - student_self
      - student_self
      - student_anchor
      - student_self
      - student_self
      - student_history
      - student_self
      - student_anchor
      - student_self
      - student_self
      - student_history
```

Warmup 可使用另一段 `sequence`（例如更多 `student_self`、更少 `student_history`），由 `stage6_0.phase` 或独立 `scheduler_v10.warmup.step_program` 选择（实现时 fast-fail：未定义则报错，避免静默默认）。

### 5.2 校验

- `len(sequence) == steps_per_block`（或允许 repeat 规则显式声明，二选一，避免歧义）
- `sequence` 中元素 ∈ `Stage6StepType` 值集
- `_validate_cfg` 拒绝遗留键：`role_sampling`、`teacher_prob`、`student_prob`、`teacher_preserve`、`student_preserve`、`near_random_supervision`、`targets.weights` 等（与独立 YAML 草案一致）

### 5.3 Step program 与数据不足时的 **显式 fallback**（主设计，非风险提示）

固定 `sequence` 在「无 student 候选 / 无 committed history / 无 probe 候选」时必须落到**确定**的替代行为，并打 **metrics**，禁止 silent empty supervision。

**配置（建议写死在 `scheduler_v10.frame_selection`）**：

```yaml
scheduler_v10:
  frame_selection:
    skip_student_if_single_source: true
    fallback_step_type_if_no_student: teacher_bootstrap   # 或 teacher_refresh（若已实现）
    fallback_step_type_if_no_committed_history: student_self
```

**规则表**：

| 情形 | 行为 |
|------|------|
| 程序步为 `student_self` / `student_anchor` 且无 student candidate | 将本步 **解析为** `fallback_step_type_if_no_student`（通常为 `teacher_bootstrap`）；递增 `scheduler_v10/fallback_no_student_count`（或写入 `request_meta` / perf 字典） |
| 程序步为 `student_history` 且 `committed_history_frame_indices` 为空 | **warmup**：降级为 `fallback_step_type_if_no_committed_history`（默认 `student_self`），递增 `scheduler_v10/fallback_no_history_count`；**正式**：见 §4.3（fast-fail 或显式 skip 计数，禁止 silent） |
| 配置开启 `probe_near` 但本步无候选 | **关闭** 本步 `supervision.probe_near.enable`；递增 `scheduler_v10/probe_near_empty_count` |

调度器在 `materialize_current_batch_without_advance` 出口应能汇总上述计数（或与 trainer 约定由一方单调递增），便于监控数据分布。

---

## 6. 各 StepType 语义（与 Request 字段对齐）

### 6.1 `teacher_bootstrap`

每个 block 的第一步应能由 program **强制**为：

- `teacher_obs.enable = true`，`purpose = "train_update"`；**builder 显式**将 `update_teacher_prior_cache` / `update_observed_history` / `update_runtime_history` / `update_state` 按 `history_record` 配置设为 `True`（因 `Stage6TeacherObsRequest` 字段默认全为 `False`，见 §3）
- `student_prop.enable = false`
- `supervision`：`self_teacher` 指向 teacher frame；`self_student` / `teacher_anchor` / `history_visited` 关闭；`probe_near` 可选且 **log_only**

### 6.2 `student_self`

- `student_prop.enable = true`；student frame = 与 keyframe 一致策略下选取的帧（**非 teacher 锚点**）
- `supervision.self_student`：student refs；anchor/history 关闭；probe 可选 log_only
- **Live bridge**：若 `bridge.student_steps_use_live_bridge` 为 true，则 `teacher_obs.enable = true` 但 **`purpose = "live_bridge"`**，且：
  - `update_state = false`
  - `update_teacher_prior_cache = false`
  - `update_observed_history` / `update_runtime_history` 按设计关闭或仅允许「bridge 专用」统计（与 trainer 中「仅重跑 teacher 2D / prior、不写 cache」对齐）

当前 trainer 在 `role == "student"` 时用 `live_teacher_bridge` 分支重跑 2D，在 `role == "teacher"` 时写 cache；重构后应以 **`teacher_obs.purpose`** 分支，而不是仅靠 `stage6_role`。

### 6.3 `student_anchor`

- `student_prop.enable = true`
- `supervision.self_student` + `supervision.teacher_anchor` 同时 enable
- 仅此（或显式列出的）step 类型计算 **`loss/teacher_anchor`** 域
- **渲染与状态顺序必须与 §6.5 一致**，否则 live_bridge 用的 teacher render 会被误复用为 anchor loss 的「post-student」监督，失去「学生更新是否破坏锚点」的语义。

### 6.4 `student_history`

- `supervision.history_visited`：**仅从 `episode_state.committed_history_frame_indices`（§4.2）采样**；排除当前 student source、`teacher_anchor` 帧、probe 帧
- 不包含 V9 式 `near_random` 混入 history；空池时的行为见 **§4.3** 与 **§5.3**

### 6.5 Student step 的状态顺序（pre-update / post-update）

同一 `student_anchor` step 内，**teacher anchor 视角**会被使用两次，语义不同，**禁止混用一次 render 结果**：

1. **A** — 在 **pre-student** 场景状态下，对 **teacher anchor** 视角 render（+ teacher GT），仅用于 **`teacher_obs.purpose == "live_bridge"`** 的 teacher 2D feature / prior，供 student UNet 输入；**不参与** `loss/teacher_anchor`。
2. **B** — 在 **pre-student** 状态下，对 **student source** 视角 render，作为 student 分支输入几何/颜色上下文（与现有 Stage6 前向一致部分）。
3. **C** — `student_unet` + lift + **更新 student 相关 state**（`student_prop.update_state` 为 true 时）。
4. **D** — 在 **post-student** 状态下，对 **student source** 再 render，计算 **`self_student`**（若本步 enable）。
5. **E** — 在 **post-student** 状态下，对 **teacher anchor** 再 render，计算 **`loss/teacher_anchor`**（仅此时启用 teacher_anchor 监督）。
6. **F** — 在 **post-student** 状态下，对 **history / probe** 视角 render，计算 history loss 或 **无梯度** probe 指标。

`student_self` / `student_history` 若无 anchor 分支，可省略 **E**；**A** 仍仅在 live_bridge 开启且本步需要 prior 时执行。

### 6.6 `teacher_refresh`（非 MVP）

长 block 或 cache stale 时可选：`teacher_obs` 的 `purpose=train_update` 且 builder 显式打开各 `update_*`、`student_prop` 关闭、`self_teacher` 监督。建议 **第一版不实现**，配置中不出现或 `enable: false` 且代码路径未注册。

---

## 7. `request_meta` 与 `compat`

### 7.1 权威结构

每个 batch 的 `request_meta` 应以 **扁平 + 嵌套** 组合为准（便于日志与旧代码过渡）：

- 顶层：`scheduler_version`、`step_type`、`scene_id`、`segment_id`、`block_idx`、`step_idx_in_block`、`global_scheduler_step`
- `teacher_obs` / `student_prop` / `supervision`：与各域一致；`supervision` 下每个 domain 对象含 `enable`、`image_refs`、`domain`。
- **`probe_near` 不得包含 `loss_weight`**（避免 scheduler 与 loss 耦合、防止 `loss += meta["supervision"]["probe_near"]["loss_weight"] * ...` 类写法 creeping）。应使用 `log_only: true` 与 **`trainable: false`**；trainer 在 `phase != explicit_near_training` 时对 `probe_near` 做 **fast-fail**：若 `trainable is not False` 则报错。
- **所有标量 loss 权重只允许存在于 `losses.stage6_0`**（与现有 Stage6_0 校验方向一致）。
- `history_record`：`observed_writer`、`runtime_writer`、`commit_policy`（如 `step_exit`）

`probe_near` 在 `request_meta` 中的推荐形状示例：

```json
"probe_near": {
  "enable": true,
  "image_refs": [],
  "domain": "probe_near",
  "log_only": true,
  "trainable": false
}
```

### 7.2 `compat` 块

仅为旧 dataloader / 中间层不炸：

```text
compat.source_image_refs
compat.target_image_refs
compat.target_image_roles
```

**Stage6 的 loss 聚合不应再依赖 compat 的扁平 role**；trainer 应改为读 `supervision.*.enable` + refs。

---

## 8. 独立 V10 配置草案（与 Stage6_0 校验协同）

建议新增顶层 `scheduler_v10`（或保留键名但 **实现与 V9 工厂脱钩**），包含：

- `block`、`episode`、`traversal`、`frame_selection`（**含 §5.3 的 fallback 键**）、`step_program`、`supervision`、`bridge`、`history_record`、`preload`

**删除或禁止**（由 `_validate_cfg` 硬拒绝）：`role_sampling`、`targets.weights`、`teacher_prob`、`student_prob`、`near_random_supervision`、`teacher_preserve`、`student_preserve` 等。

**与现有 `minimal_trainer_stage6_0._validate_stage6_0_config` 的衔接**：

- 当前校验仍读取 `scheduler_v10.targets.weights`；独立 V10 应 **移除该节** 或改为「若存在则报错」，loss 权重仅保留 `losses.stage6_0`。
- `probe_near`：**仅** `losses.stage6_0.probe.near.loss_weight`；`request_meta.supervision.probe_near` **不出现** `loss_weight` 字段（§7.1）。

---

## 9. 实现骨架（类与方法边界）

```python
class TrainSchedulerV10:
    def __init__(
        self,
        *,
        dataset,
        steps_per_block: int,
        blocks_per_episode: int,
        frame_selection_cfg,
        step_program_cfg,
        supervision_cfg,
        traversal_cfg,
        preload_cfg,
        fixed_scene_id=None,
        fixed_segment_id=None,
    ):
        ...

    def _validate_cfg(self) -> None: ...

    def start_new_episode(self) -> None:
        self.current_episode_plan = self._build_episode_plan_v10()
        self.current_block_state = self._init_block_state(block_idx=0)

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        step_type = self._current_step_type()
        req = self._build_step_request(step_type)
        batch = self._materialize_batch_from_request(req)
        batch["request_meta"] = self._request_to_meta(req)
        return batch

    def advance(self) -> None:
        ...
```

**数据装载**：`_materialize_batch_from_request` 可内部调用与 V7 dataset 一致的「按 ref 取 view」逻辑（从现有 V8/V9 私有方法抽取 **纯函数或 Mixin**，避免继续 `super().materialize_*`）。

---

## 10. Trainer 侧消费改造（`minimal_trainer_stage6_0.py`）

### 10.1 目标形态

```text
meta = batch["request_meta"]
sup = meta["supervision"]

if sup["self_teacher"]["enable"]:
    loss_self_teacher = ...
if sup["self_student"]["enable"]:
    loss_self_student = ...
...
if sup["probe_near"]["enable"]:
    with torch.no_grad():
        metric_probe_near = ...
```

**总 loss** 仅从 `losses.stage6_0` 读权重，且 **不得**将 `probe_near` 纳入加权和（除非 `phase=explicit_near_training` 且单独分支，与现有校验一致）。

### 10.2 与现有代码的对应关系

| 现状 | 目标 |
|------|------|
| `_get_stage6_role` + `stage6_role` | `student_prop.enable` + `teacher_obs.purpose` 组合驱动 forward 分支 |
| `live_teacher_bridge` dict | 并入 `teacher_obs`，`purpose == "live_bridge"` |
| `_build_target_view_weights` 读 `train_target_image_roles` | 按 `supervision` 各域分别收集 targets 与权重，或分域调用父类逻辑 |
| cache 更新在 `role == "teacher"` | 仅在 `teacher_obs.purpose == "train_update"` 且相应 `update_*` 为 true 时执行 |

### 10.3 迁移顺序建议

1. 在 scheduler 侧同时写出「新 `supervision` 块」与短期 **compat** 扁平字段（双写期）
2. Trainer 先读新块，缺失再 fallback 到旧字段（可选，缩短双写期可省略 fallback，直接 fast-fail）
3. 删除 V10 对 `TrainSchedulerV9` 的继承与 `_patch_request_meta_to_v10`
4. 收紧 YAML 与 `_validate_cfg`

---

## 11. 分阶段落地清单（建议）

| 阶段 | 内容 | 验收 |
|------|------|------|
| A | 新建 `stage6_step_types.py`（Enum + dataclass）+ `request_meta` 序列化/反序列化单测 | 不依赖 dataset 的纯单测 |
| B | 实现 `EpisodePlanV10` / `EpisodeStateV10` / `BlockStateV10` + `_commit_block`（§4.2）+ `_build_episode_plan_v10`（从 V8 抄 traversal/keyframe，不经过 V9 role） | 固定 seed 下 step_type 可复现；block 尾 committed 列表单调增长 |
| C | `_build_step_request` + `_materialize_batch_from_request` + `preload_hints` | batch 含完整 refs，dataloader 不报错 |
| D | Trainer 按 `supervision` + `teacher_obs.purpose` 改分支；总 loss 不含 probe | 与现 Stage6 指标对齐或文档说明差异 |
| E | 配置迁移 + 删除 V9 继承 + 清理 `scheduler_request_v10` 旧嵌套键名（或保留一层 alias 一个版本） | `_validate_cfg` 与 CI |

---

## 12. 风险与注意事项

- **Camera 采样**：V9 的 `camera_sampling_cfg` 若仍需，应作为 V10 的独立子配置接入 `ImageRef` 的第二维 `cam_id`，而非绑在 `role_sampling` 上。
- **单源 / 空 history / 空 probe**：主流程见 **§5.3** 与 **§4.3**，必须配 metrics，禁止 silent empty。
- **全局步数**：`global_scheduler_step` 与 optimizer `global_step` 对齐策略需在文档/日志中写清（沿用 aligned_info 或仅 request_meta）。

---

## 13. 参考文献（仓库内）

- `datasets/train_scheduler_v8.py` — `EpisodePlanV8`
- `datasets/train_scheduler_v9.py` — `TrainSchedulerV9` 构造参数与 `role_sampling`
- `datasets/train_scheduler_v10.py` — 当前兼容层实现
- `models/streetforward/minimal_trainer_stage6_0.py` — cache / live bridge / target 权重
- `docs/trainers/StreetForward_Stage6_0_Decoupled_Teacher_Student_Design.md` — Stage6 域与命名

本文档描述的是 **目标架构与迁移步骤**；具体 PR 可按第 11 节拆分为多个可审查变更集。
