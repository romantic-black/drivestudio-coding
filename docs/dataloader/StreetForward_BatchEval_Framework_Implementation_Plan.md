# StreetForward BatchEval：批量测试框架详细实现方案

## 1. 设计目标

本文档给出 StreetForward BatchEval 的落地实现计划。核心目标是建立一套**独立于训练 scheduler** 的批量测试系统，不复用 `TrainSchedulerV8` / `ValidationV8` 的 target 生成语义，而是显式定义评估协议：

- 输入帧集合 `input frames`
- 评估帧集合 `eval frames`
- 相机集合 `cameras`
- 每个输入帧的迭代步数 `steps_per_input`
- 每次迭代需要保存的全部评估视角

当前 V8 scheduler 的语义是 `visited_episode_frames`，并强制 `W = E`，目标帧来自当前 episode 的已访问 block，不适用于 STORM 风格协议（如输入 0/5/10/15，评估 0..19）。

BatchEval 应仅复用：

1. `segment_finetune_train` 的数据构造模式；
2. 模型已有无梯度推理/更新接口；
3. 模型内部 node state / hidden cache / history memory 更新逻辑；
4. 渲染、mask、metric 计算工具。

不复用训练 scheduler 的 episode/block/target 选择逻辑。

---

## 2. 与现有实现的边界（基于当前代码）

### 2.1 V8 语义边界（为何必须解耦）

`datasets/train_scheduler_v8.py` 中存在硬约束：

- `target_policy` 仅支持 `visited_episode_frames`
- `total_target_frames <= blocks_per_episode`
- target window 基于“当前 block + 已访问 block”动态生成

`datasets/validation_scheduler_v8.py` 也将 `eval_image_refs` 绑定到 `frame_chain` 展开：

- `frame_chain` 来自 keyframe window
- `eval_image_refs` 对 `frame_chain x all cams` 展开
- 不能直接表达 `input_offsets != eval_offsets` 的协议

因此 BatchEval 需要独立 episode/protocol builder。

### 2.2 Dataset 侧可复用能力

`datasets/multi_scene_dataset_v4.py` 已具备按 image refs 组 batch 的基础设施：

- `list_segment_ids(scene_id)`
- `get_segment_index(scene_id, segment_id)`
- `get_segment_batch_from_image_refs(BatchRequestV4, enforce_target0_equals_source=...)`
- `get_segment_eval_batch_from_image_refs(EvalRequestV4)`

同时 `_assemble_segment_batch_from_image_refs(...)` 已内建大量严格校验（空 refs、image ref 合法性、KNN 结构一致性、mask 对齐等），适合作为 BatchEval 的数据入口。

---

## 3. 推荐新增目录结构

```text
streetforward_eval/
  __init__.py

  protocols.py
  episode_builder.py
  batch_builder.py
  runner.py
  metrics.py
  snapshot_writer.py
  summary.py

tools/
  eval_streetforward_benchmark.py

configs/
  eval/
    streetforward_batcheval.yaml
    exp1_single_frame.yaml
    exp2_storm20_sparse4.yaml
    exp4_input_count.yaml
    exp5_block_size.yaml
```

职责：

- `protocols.py`：实验协议 dataclass 与校验
- `episode_builder.py`：从 scene/segment/window 生成 `TestEpisodeSpec`
- `batch_builder.py`：source refs + target refs -> 模型可用 batch
- `runner.py`：迭代更新、渲染、history 记录
- `metrics.py`：PSNR / SSIM / LPIPS / L1
- `snapshot_writer.py`：每迭代图像、meta、CSV、视频
- `summary.py`：episode/scene/checkpoint 聚合

---

## 4. 核心原则

### 4.1 训练 scheduler 与测试 protocol 解耦

BatchEval 协议以 frame 集合显式表达，不使用 V8 的 `frame_chain -> visited target window`：

```text
frame_ids       = 完整短序列（例：0..19）
input_frame_ids = 稀疏输入帧（例：0,5,10,15）
eval_frame_ids  = 评估帧集合（例：0..19）
```

### 4.2 每次迭代都渲染完整 eval views

StreetForward 是 iterative update，必须保留时序：

- `iter_000_pre`（初始状态）
- 每个 input frame 的每个 update step 渲染
- 最终状态

### 4.3 Update 与 Eval 的 target 严格分离（防评测泄漏）

每个迭代必须拆成两段，且使用**不同 target 集合**：

1. **Update step（允许更新状态）**
   - `source_image_refs = 当前 input_frame x cameras`
   - `update_target_image_refs = observed_frames x cameras`
   - `observed_frames` 仅包含到当前为止已输入帧（例如先 [0]，再 [0,5]，再 [0,5,10]）
   - 禁止包含任何未观测 eval frames

2. **Eval render step（只渲染，不更新状态）**
   - `render_target_image_refs = eval_frames x cameras`
   - 仅用于导出可视化与指标，不参与 updater 状态更新

这样才能保证 Exp2 真正对齐 STORM 语义：更新仅基于稀疏输入证据，完整 20 帧只用于评估输出。

---

## 5. 配置设计

## 5.1 总配置（建议）

```yaml
batch_eval:
  enable: true
  output_dir: outputs/batch_eval
  data_mode: segment_finetune_train

  checkpoint:
    path: outputs/xxx/checkpoints/step_600000.pth
    strict: true

  dataset:
    scene_ids: ${data.eval_scene_ids}
    segment_policy: all
    require_full_window: true
    stride: 20
    max_episodes_per_scene: null
    max_total_episodes: null
    window_policy: sliding

  cameras:
    names: ["front_left", "front", "front_right"]

  runtime:
    no_grad: true
    amp: true
    reset_state_per_episode: true
    update_node_state: true
    update_hidden_state: true
    update_view_transient: true

  history:
    update_step_norm_ema: true
    record_support_residual_on_input_exit: true
    record_each_step: false

  render:
    save_pre_update: true
    save_each_iter_views: true
    save_png: true
    save_video: true
    save_numpy: false
    save_depth_or_acc: false

  metrics:
    primary_mask: non_sky_non_ego
    report_full_image: true
    compute_psnr: true
    compute_ssim: true
    compute_lpips: true
    compute_l1: true
    min_valid_pixels: 32

  experiments:
    - name: exp1_single_frame
      sequence_length: 1
      input_offsets: [0]
      eval_offsets: [0]
      steps_per_input: 8

    - name: exp2_storm20_sparse4
      sequence_length: 20
      input_offsets: [0, 5, 10, 15]
      eval_offsets: all
      steps_per_input: 8
```

## 5.2~5.6 实验定义原则

- `exp1`：最小闭环，1 frame x 3 cams x all iterations
- `exp2`：20 帧稀疏输入 + 全帧评估（主协议）
- `exp3`：只换 checkpoint/variant，其他协议强一致
- `exp4`：输入帧数量实验，建议 `uniform-over-20`
- `exp5`：`fixed budget` 与 `native budget` 分表报告

---

## 6. 数据结构设计

## 6.1 `TestProtocolSpec`

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class TestProtocolSpec:
    name: str
    data_mode: str

    sequence_length: int
    input_offsets: list[int]
    eval_offsets: list[int] | Literal["all"]

    camera_names: list[str]
    camera_ids: list[int]

    steps_per_input: int
    save_pre_update: bool = True
    save_each_iter_views: bool = True

    metric_primary_mask: str = "non_sky_non_ego"
    report_full_image: bool = True

    input_count_label: str | None = None
    train_block_size_label: str | None = None
```

## 6.2 `TestEpisodeSpec`

```python
@dataclass(frozen=True)
class TestEpisodeSpec:
    exp_name: str
    scene_id: int
    segment_id: int

    sequence_start_pos: int
    frame_offsets: list[int]           # 0..sequence_length-1，相对索引（对齐 STORM）
    frame_ids: list[int]               # 数据集绝对 frame_idx（用于取数）

    input_offsets: list[int]           # 例如 [0,5,10,15]
    eval_offsets: list[int]            # 例如 [0..19]
    input_frame_ids: list[int]         # 由 input_offsets 映射得到的绝对 frame_idx
    eval_frame_ids: list[int]          # 由 eval_offsets 映射得到的绝对 frame_idx

    camera_ids: list[int]
    camera_names: list[str]

    input_image_refs: list[tuple[int, int]]
    eval_image_refs: list[tuple[int, int]]

    episode_uid: str
```

`episode_uid` 建议：

```python
f"scene{scene_id:03d}_seg{segment_id:03d}_start{sequence_start_pos:06d}"
```

## 6.3 `EvalStepRecord`

```python
@dataclass
class EvalStepRecord:
    exp_name: str
    episode_uid: str

    global_iter: int
    input_index: int | None
    input_frame_id: int | None
    local_step: int

    is_pre_update: bool
    is_final: bool

    source_image_refs: list[tuple[int, int]]
    eval_image_refs: list[tuple[int, int]]

    output_dir: str
```

---

## 7. Episode 生成器

接口：

```python
def build_test_episode_specs(
    *,
    dataset,
    scene_ids: list[int],
    protocol: TestProtocolSpec,
    segment_policy: str = "all",
    window_policy: str = "sliding",
    stride: int = 20,
    require_full_window: bool = True,
    max_episodes_per_scene: int | None = None,
    max_total_episodes: int | None = None,
) -> list[TestEpisodeSpec]:
    ...
```

关键点：

- 使用滑窗得到 episode 的 `frame_ids`（绝对 frame_idx）与 `frame_offsets`（0..L-1）
- `input_offsets` / `eval_offsets` 始终定义在 `frame_offsets` 上（STORM 对齐语义）
- 通过 `frame_ids[offset]` 映射为 `input_frame_ids` / `eval_frame_ids`
- 不经过 V8 keyframe target 逻辑
- 所有 offset 做边界 fast-fail，不加“隐式默认补救”

---

## 8. Batch 构造器（Update 与 Render 分离）

接口：

```python
def build_update_batch_from_refs(
    *,
    dataset,
    scene_id: int,
    segment_id: int,
    source_image_refs: list[tuple[int, int]],
    update_target_image_refs: list[tuple[int, int]],
    data_mode: str = "segment_finetune_train",
    device: str | torch.device = "cuda",
) -> dict:
    ...
```

实现建议：

1. 底层复用 `MultiSceneDatasetV4.get_segment_batch_from_image_refs(...)`
2. `update_target_image_refs` 仅允许来自 observed frames，禁止含未来帧
3. 在 wrapper 层补齐 runner 需要的显式字段（含 `request_meta.eval_protocol` 与 `request_meta.batch_role=update`）
4. 统一 source/target 数量和 frame 一致性校验

render-only 另建接口：

```python
def build_render_batch_from_refs(
    *,
    dataset,
    scene_id: int,
    segment_id: int,
    render_target_image_refs: list[tuple[int, int]],
    data_mode: str = "segment_finetune_train",
    device: str | torch.device = "cuda",
) -> dict:
    ...
```

`build_render_batch_from_refs` 仅提供渲染视角，不作为 updater 输入；`request_meta.batch_role=render_only`。

fast-fail 例子：

```python
assert len(source_image_refs) == 3
assert len(update_target_image_refs) > 0
source_frame_idx = source_image_refs[0][0]
assert all(fid == source_frame_idx for fid, _ in source_image_refs)
assert all(cam in camera_ids for _, cam in source_image_refs)
assert all(cam in camera_ids for _, cam in update_target_image_refs)
assert all(fid in observed_frame_ids for fid, _ in update_target_image_refs)
```

---

## 9. Runner 设计

主接口：

```python
class StreetForwardBatchEvalRunner:
    def __init__(self, model, dataset, protocol, writer, metric_acc, device): ...
    def run_episode(self, spec: TestEpisodeSpec) -> dict: ...
```

执行流程（最终版）：

```python
reset_eval_runtime_state()

render_current_state(eval_refs)  # iter_000_pre

observed_frame_ids = []

for input_frame_id in input_frame_ids:
    observed_frame_ids.append(input_frame_id)

    for local_step in range(1, steps_per_input + 1):
        # A) update-only
        update_batch = build_update_batch_from_refs(
            source_refs=input_frame_id x cameras,
            update_target_refs=observed_frame_ids x cameras,
        )
        model.eval_update_step(update_batch)

        # B) render-only
        render_batch = build_render_batch_from_refs(
            render_target_refs=eval_frame_ids x cameras,
        )
        render_out = model.eval_render_current_state(render_batch)
        save_metrics_and_snapshots(render_out)

    record_history_on_input_exit(source_refs)
```

关键约束：

- update step 绝不接收完整 eval refs
- 完整 eval refs 仅用于 render-only 路径
- render-only 路径不允许更新 node/hidden/history/view-transient

接口建议：

- `eval_update_step(update_batch, ...)`：允许状态更新
- `eval_render_current_state(render_batch, ...)`：只读当前状态做渲染

---

## 10. History memory 策略

推荐语义：

- **一个 input frame 视为一个 eval block**
- `steps_per_input` 为该 block 内 recurrent 更新步数
- block 完成后再做 support/residual history record

history 配置显式拆分为两类，避免语义冲突：

```yaml
history:
  update_step_norm_ema: true
  record_support_residual_on_input_exit: true
  record_each_step: false
```

执行约束：

- `update_step_norm_ema` 可在每个 update step 更新
- `record_support_residual_on_input_exit` 只在 block exit 调用一次
- 禁止同时走“每 step record + block exit record”双通道

---

## 11. Metrics 设计

单视角接口：

```python
def compute_view_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sky_mask: torch.Tensor | None,
    egocar_mask: torch.Tensor | None,
    *,
    primary_mask: str,
) -> dict:
    ...
```

主指标：

- `psnr_non_sky_non_ego`
- `ssim_non_sky_non_ego`
- `lpips_non_sky_non_ego`
- `l1_non_sky_non_ego`

附加指标：

- `psnr_full`, `ssim_full`, `lpips_full`, `l1_full`

分组统计（实验 2 建议）：

- `input` / `interp` / `extrap`
- camera 维度分组（`front`, `front_left`, `front_right`）

---

## 12. SnapshotWriter 设计

目录建议：

```text
outputs/batch_eval/
  exp2_storm20_sparse4/
    full/
      ckpt_step600000/
        meta.json
        protocol.yaml
        metrics_iter.csv
        metrics_final.csv
        summary.csv
        scene_xxx/segment_xxx/...
```

命名建议：

- pre：`iter_000_pre`
- 非 pre：`iter_{global_iter:03d}_input{input_frame_id:06d}_step{local_step:02d}`

每视角保存：

- `{cam_name}_pred.png`
- `{cam_name}_gt.png`
- `{cam_name}_error.png`

---

## 13. 主入口脚本

脚本：`tools/eval_streetforward_benchmark.py`

支持：

- 单实验、批量实验
- 单 checkpoint、多 checkpoint/variant
- CLI override（`--experiment/--experiments/--checkpoint/--output_dir`）

运行示例：

```bash
PYTHONPATH=/root/drivestudio-coding \
python tools/eval_streetforward_benchmark.py \
  --config_file configs/eval/streetforward_batcheval.yaml \
  --experiment exp2_storm20_sparse4 \
  --checkpoint outputs/xxx/checkpoints/step_600000.pth \
  --output_dir outputs/batch_eval/exp2
```

---

## 14. Fast-fail 检查清单

统一在 protocol resolve 阶段检查：

- `data_mode == "segment_finetune_train"`
- `sequence_length >= 1`
- `len(input_offsets) >= 1`
- `steps_per_input >= 1`
- 所有 offset 均在 `[0, sequence_length)` 内
- `camera_ids` 非空且与 `camera_names` 一一对应

专项检查：

- Exp2/3/4/5：`sequence_length == 20` 且 `eval_offsets == "all"`
- Exp3：所有 variant 共用同一 `episode_specs`
- Exp5 fixed budget：所有 checkpoint 共享相同 `steps_per_input`

---

## 15. 与现有代码的最小侵入改动

必须新增：

- `streetforward_eval/protocols.py`
- `streetforward_eval/episode_builder.py`
- `streetforward_eval/batch_builder.py`
- `streetforward_eval/runner.py`
- `streetforward_eval/metrics.py`
- `streetforward_eval/snapshot_writer.py`
- `tools/eval_streetforward_benchmark.py`

建议模型新增 wrapper（Stage5.3/5.4）：

- `eval_update_step(...)`：显式 update-only 入口
- `eval_render_current_state(...)`：真正 render-only 入口（不跑 updater 子图）
- `eval_record_block_history(...)`：统一 block-exit 记录入口

---

## 16. 分阶段实现计划

### Phase 1：最小可跑（Exp1）

范围：

- `protocols + episode_builder + batch_builder + runner(单 input) + metrics(PSNR/L1) + snapshot_writer(PNG)`

验收：

- 输出 `iter_000_pre` 与 `iter_001..iter_K`
- 生成 `metrics_iter.csv`

### Phase 2：STORM20 稀疏输入（Exp2）

范围：

- `eval_offsets=all`
- `20x3` 每步渲染
- `global_iter` 与 `input/interp/extrap` 分组
- `summary.csv`

验收：

- 可绘制 PSNR-over-iteration
- 可导出每步 `20x3` 视角结果

### Phase 3：多 checkpoint/ablation（Exp3）

范围：

- variant 循环
- 固定 episode 列表复用
- `summary_all_variants.csv`

验收：

- full/no_history/no_xcpe 等在**相同 episode 集合**上可比

### Phase 4：输入数量实验（Exp4）

范围：

- `input_offset_sets`
- `input_count_label`
- 按输入数量聚合

验收：

- input4/input6/input8/input10 同表对比

### Phase 5：训练 block_size 实验（Exp5）

范围：

- checkpoint group
- fixed budget & native budget 双评估

验收：

- 输出 Table A（fixed）与 Table B（native）

---

## 17. 关键实现选择（最终建议）

框架实现应坚持以下 5 条：

1. 不复用 V8 target 逻辑，只复用 dataset + model inference/update 能力；
2. `update targets` 只能来自 observed input frames，`eval targets` 只能用于 render-only；
3. 一个 input frame 视为一个 eval block，history 在 block exit 记录；
4. 每个 recurrent step 后都做一次 `render_current_state(eval_frames x 3 cams)`；
5. Exp3/4/5 只改变目标变量，其余 protocol 固定。

这能同时覆盖：

- 单帧重建能力
- 稀疏输入到 20 帧重建能力
- 消融公平比较
- 输入数量敏感性
- 训练 block_size 敏感性

并从根本上避免被训练 scheduler 的 visited-frame 语义约束。
