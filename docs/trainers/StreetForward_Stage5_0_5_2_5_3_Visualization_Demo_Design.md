# StreetForward Stage5_0 / 5_2 / 5_3 可视化验证 Demo 详细设计

## 1. 目标与范围

本文给出 `Stage5_0`、`Stage5_2`、`Stage5_3` 的统一可视化验证 demo 方案，目标是：

- 复用现有训练主链路中的 `DatasetV4 + TrainSchedulerV8 + MinimalTrainer`。
- 使用**冻结网络权重**做交互式时序推理，不进行反向传播和优化器更新。
- 严格遵循 `scheduler_v8` 的 source/target 生成与事件语义（尤其是 `visited_episode_frames`、`episode_end reset`）。
- Viewer 默认仅显示 `bg + distant`，但**不破坏**模型内部 full-routed 语义（`rigid_in`/`rigid_out` 仍参与推理）。

非目标：

- 不在 demo 中改动训练目标函数。
- 不在 demo 中引入新的采样顺序或绕开 scheduler 的手写顺序。
- 不在 demo 中做在线 finetune。

---

## 2. 现状基线与关键约束

### 2.1 当前代码可复用能力

- 已有数据与调度器构建能力：
  - `tools/train_minimal_streetforward_stage4_3_v8_common.py`
  - `build_multi_scene_dataset_v4`
  - `build_train_scheduler_v8_from_cfg`
- 已有 SchedulerV8 事件机制：
  - `datasets/train_scheduler_v8.py`
  - `next_batch()` + `pop_events()`，且会发出 `block_end`、`block_exit`、`reset_event` 等事件。
- 已有 block-exit history 训练链路：
  - `tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py` 中已经是 `block_exit -> model.record_block_history(...)`。
- 已有 viewer/controller 原型：
  - `tools/streetforward_demo_controller.py`
  - `tools/streetforward_viewer.py`
  - 适合作为新 demo 的 UI/控制器脚手架。

### 2.2 为什么不能直接用 `train_step()`

`Stage4_3` 基类 `train_step()` 明确包含：

- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

直接复用会导致参数继续更新，不符合“预训练权重推理验证”的实验语义。

### 2.3 为什么建议新增 `demo_infer_step()`（而不是直接复用现有 inference API）

现有 `inference_step_from_train_batch()` 虽然可做到 no-backward/no-optimizer，但它是通用接口，默认行为包含训练态路径与 scheduler 同步逻辑，且对 Stage5 系列的 memory/gate 细节没有“显式可控开关”。

demo 需要更强可控性：

- 显式控制 node_state/hidden/history 是否写回。
- 显式接收 scheduler 事件并按 block_exit 触发 record。
- 为 Stage5_2/5_3 分别处理 update memory 的差异语义。

---

## 3. Stage 差异矩阵（demo 设计必须显式覆盖）

| 维度 | Stage5_0 | Stage5_2 | Stage5_3 |
|---|---|---|---|
| struct decoder | xCPE，仅 `bg+rigid_in` | routed near/far，full-routed | routed near/far，full-routed |
| history memory | 无 | 必需，flat schema | 必需，nested schema（support/residual/update） |
| update gate | 无 | 必需 | 必需 |
| record_block_history | 无 | 有 | 有 |
| feature extractor | 继承 Stage4_6 路径 | 继承 Stage4_6 路径 | 强制 DINOv2+UNet fusion |
| scheduler 约束 | V8（配置） | 强约束 V8 visited+episode_end | 强约束 V8 visited+episode_end |

补充说明：

- `Stage5_2` 的 `forward()` 里 update_norm 统计仅在 `self.training` 下更新。
- `Stage5_3` 的 update memory 路径支持 `history_memory.update.apply_in_eval`（代码默认可为 true）。
- 因此 demo 中需要对两者分别处理，避免“看起来都在跑，但 memory 行为不一致”。

---

## 4. 总体架构设计

建议新增 4 个文件（兼容 5_0/5_2/5_3）：

```text
tools/demo_minimal_streetforward_stage5_viewer.py
tools/streetforward_stage5_demo_controller.py
tools/streetforward_stage5_viewer.py
configs/demo_minimal_streetforward_stage5_viewer.yaml
```

> 说明：统一入口比按 stage 拆 3 套脚本更易维护；通过 `model.stage` 或参数选择具体 trainer。

### 4.1 模块职责

- `demo_minimal_streetforward_stage5_viewer.py`
  - CLI 入口、加载配置、构建 dataset/scheduler/model、加载 ckpt、启动 viewer。
- `streetforward_stage5_demo_controller.py`
  - 管理 step/block 推进、事件处理、调用 `demo_infer_step()`、维护 UI 状态。
- `streetforward_stage5_viewer.py`
  - 提供 `Next Step / Next Block / Reset` 按钮与状态面板。
  - 只渲染 `bg + distant`。
- `demo_minimal_streetforward_stage5_viewer.yaml`
  - demo 专用配置，不污染训练配置。

---

## 5. CLI 与配置设计

## 5.1 CLI 参数

```bash
python tools/demo_minimal_streetforward_stage5_viewer.py \
  --config_file configs/demo_minimal_streetforward_stage5_viewer.yaml \
  --ckpt outputs/minimal_sf/stage5_2/checkpoints/step120000.pt \
  --stage 5_2 \
  --scene_id 0 \
  --segment_id 0 \
  --host 0.0.0.0 \
  --port 8080 \
  --device cuda \
  --ckpt_load_mode network_only \
  --max_steps 0
```

参数建议：

- `--config_file`
- `--ckpt`
- `--stage`：`5_0 | 5_2 | 5_3`
- `--scene_id`
- `--segment_id`（可选，空则遍历 scene 下 segment）
- `--host`、`--port`
- `--device`
- `--ckpt_load_mode`：`network_only | full_state`
- `--max_steps`（0 表示不限）
- `--headless`（用于无 viewer 快速验证）

## 5.2 demo 配置块（新增）

```yaml
demo:
  mode: frozen_recurrent_inference
  pause_on_start: true
  step_button_mode: raw_step
  display_branches: ["bg", "distant"]
  hide_rigid: true

  checkpoint:
    load_mode: network_only
    strict_network: false

  inference:
    update_node_state: true
    update_hidden_state: true
    update_history_memory: true
    record_block_history_on_block_exit: true
    no_optimizer_step: true
    no_backward: true

  viewer:
    host: "0.0.0.0"
    port: 8080
    viewer_res: 1280
```

fast-fail 约束：

- demo 启动时校验 `scheduler_v8.enable == true`。
- Stage5_2/5_3 校验 `target_policy == visited_episode_frames` 且 `reset_policy == episode_end`。
- `hide_rigid=true` 仅影响渲染层，若用户尝试关闭 rigid 路由参与推理，直接报错。

---

## 6. Scheduler 对齐策略（必须复用 V8）

核心原则：**顺序完全来自 `TrainSchedulerV8`**。

controller 单步：

1. `raw_batch = scheduler.next_batch()`
2. `events = scheduler.pop_events()`
3. batch 转 minimal（复用训练转换函数）
4. `model.demo_infer_step(...)`
5. 若事件含 `block_exit`（或配置选择 `block_end`），调用 `model.record_block_history(...)`
6. 刷新 viewer 状态并 rerender

推荐适配器：

```python
class SchedulerStepAdapter:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def next(self):
        batch = self.scheduler.next_batch()
        events = self.scheduler.pop_events() if hasattr(self.scheduler, "pop_events") else []
        return batch, events
```

---

## 7. Trainer 侧 API 设计

## 7.1 新增统一 demo 推理接口

在 `Stage5_0/5_2/5_3` 对应 trainer 中新增：

```python
@torch.no_grad()
def demo_infer_step(
    self,
    batch: Dict[str, Any],
    *,
    scheduler_events: Optional[list[Dict[str, Any]]] = None,
    update_node_state: bool = True,
    update_hidden_state: bool = True,
    update_history_memory: bool = True,
) -> Dict[str, Any]:
    ...
```

设计要点：

- 不做 backward / optimizer step。
- 保留 forward 的 full-routed 结构计算。
- node_state / hidden_cache 的写回逻辑与训练路径复用同一内部函数，避免两套逻辑漂移。
- 返回 stats 供 UI 展示（source/target refs、branch 更新计数、history 统计等）。

## 7.2 训练与 demo 共享的内部提交函数

将 `forward -> 写回`路径抽象为内部公共函数，例如：

- `_commit_forward_state_from_out(...)`
- `_commit_hidden_cache_from_out(...)`
- `_commit_step_history_from_out(...)`（仅 Stage5_2/5_3）

`train_step()` 与 `demo_infer_step()`都走同一提交函数，减少行为偏差。

---

## 8. Stage 级别细化

## 8.1 Stage5_0

无 history / gate，demo 最简单：

- 仅做 no-grad forward。
- 写回 node_state 与 hidden cache。
- 不调用 `record_block_history`（接口不存在时必须 fast-fail 分支跳过）。

## 8.2 Stage5_2

重点：

- 保留 full-routed 输入语义（near: `bg+rigid_in`，far: `distant+rigid_out`）。
- `record_block_history()` 建议在 `block_exit` 时调用，保持与训练一致。
- 若 `demo_infer_step()`使用 eval 态，需要显式处理 step update norm（否则 update memory 统计可能不完整）。

推荐方案：

- 在 demo 路径下显式触发 `_update_last_step_update_norm_from_out(out)`，而不是依赖 `self.training`。
- 或新增专用 flag（如 `_stage5_demo_update_history`）控制该行为。

## 8.3 Stage5_3

重点：

- 特征提取器是 DINOv2+UNet 融合，需确保 checkpoint 与配置匹配。
- history schema 为 nested，配置必须严格校验。
- update memory 支持 eval 路径开关（`apply_in_eval`），demo 中需显式记录当前开关状态到 UI/日志。

---

## 9. Viewer 设计（交互与渲染）

基于 `tools/streetforward_viewer.py` 扩展一个 Stage5 版本：

- 默认暂停，不自动循环。
- `Next Step`：执行 1 个 scheduler raw step。
- `Next Block`：执行到当前 block 结束。
- `Reset Scene State`：调用 `model.reset_node_state()` 后刷新当前快照。

状态面板建议字段：

- `Global Step`
- `Scene ID`
- `Segment ID`
- `Episode / Block / Segment Step`
- `Source Ref`
- `Target Refs`
- `Last Events`
- `history_*_support/error mean`（5_2/5_3）

### 9.1 只显示 `bg + distant` 的正确方式

必须遵循：

- 推理阶段保留 rigid 分支参与计算。
- 渲染阶段只拼接 `bg + distant` 参数。

不允许：

- 在 batch 组装或 forward 阶段删除 rigid（会破坏 routed 语义与 memory/gate 一致性）。

---

## 10. Batch 组装与兼容性要求

demo 必须复用训练时的 batch 转换函数（例如 `convert_batch_to_minimal_format`），保证以下字段一致：

- `source_views`
- `source_images`
- `targets`
- `source_frame_idx`
- `request_meta.source_image_refs`
- `request_meta.target_image_refs`

原因：

- `Stage5_2/5_3.record_block_history()` 对这些字段有硬依赖。
- 缺失字段应立即 fast-fail，而不是 silent fallback。

---

## 11. Checkpoint 加载策略

## 11.1 `network_only`（默认推荐）

用途：新 scene/segment 的纯推理演示。

- 仅加载网络参数。
- 跳过 `node_states_*`、`h_cache_*`、`history_*`、`optimizer*` 等 runtime 状态。

## 11.2 `full_state`

用途：同一训练上下文的状态复现。

- 加载完整状态，可能包含旧 scene/segment 运行态。
- 若目标 scene/segment 不一致，可能发生状态污染或 shape mismatch，需在启动时告警。

---

## 12. 事件驱动时序约定

`Next Step` 必须等价于 `scheduler.next_batch()` 一次推进。

以 `step_major + switch_interval=4` 为例，用户点击应观察到逐步切换，而非一次跑完整 block。

`Next Block` 可以做附加按钮，但不能替代主按钮语义。

---

## 13. 分阶段落地计划

1. **Headless 验证先行**
   - 跑 `--headless --max_steps 10`，验证 batch/forward/writeback/event 无异常。
2. **补 `demo_infer_step()`**
   - 先在 Stage5_2 实现，再平移到 5_0/5_3。
3. **渲染 helper**
   - 实现 `bg+distant-only` 渲染，不依赖 target camera。
4. **接 viewer**
   - 先静态显示初始状态，再接 `Next Step`。
5. **接 block_exit record**
   - 打通 `record_block_history` 指标展示。
6. **补鲁棒性**
   - busy 防重入、异常显示、状态恢复、max_steps 截断。

---

## 14. 验收标准

功能验收：

- 点击 `Next Step` 后，scheduler 对齐信息与 source/target refs 按 V8 规则变化。
- 模型参数不更新（可比较 step 前后参数 hash）。
- node_state 与 hidden cache 持续演化。
- `5_2/5_3` 在 block_exit 触发 history 记录，指标非空并可视化。
- viewer 仅显示 `bg + distant`，但 rigid 路由相关统计仍变化。

一致性验收：

- 与训练主循环同一 scene/segment/seed 下，事件顺序一致（允许浮点误差）。
- batch 关键字段结构一致，无 alias 缺失。

---

## 15. 测试建议（遵循项目约定）

运行测试时建议：

- 使用 `conda drivestudio-new`
- 设置 `PYTHONPATH=/root/drivestudio-coding`

示例：

```bash
conda run -n drivestudio-new \
  env PYTHONPATH=/root/drivestudio-coding \
  python tools/demo_minimal_streetforward_stage5_viewer.py \
  --config_file configs/demo_minimal_streetforward_stage5_viewer.yaml \
  --stage 5_2 \
  --ckpt /path/to/ckpt.pt \
  --scene_id 0 \
  --segment_id 0 \
  --headless \
  --max_steps 10
```

---

## 16. 风险点与 fast-fail 清单

- 误用 `train_step()` 导致参数更新：启动即校验 demo mode 并禁用 optimizer。
- 手写调度顺序：强制只走 `scheduler.next_batch()`。
- `record_block_history` 输入字段缺失：字段校验失败直接抛错。
- 通过删 rigid 来隐藏渲染：禁止；仅允许渲染层过滤。
- checkpoint runtime 状态污染：`network_only` 设为默认并提示风险。

---

## 17. 推荐实现结论

优先做“统一 Stage5 demo 框架”，在同一入口通过 `--stage` 分流 5_0/5_2/5_3，复用 DatasetV4 与 SchedulerV8，并新增显式 `demo_infer_step()`。

这样可同时满足：

- 与训练时序语义一致
- 可视化交互可控
- 不更新网络参数
- 可扩展到后续 Stage5_x

