# StreetForward Stage5_3 Production 训练实现计划（Fast-Fail + Lightweight Resume）

> 参考代码与配置：
> - `models/streetforward/minimal_trainer_stage5_3.py`
> - `configs/minimal_streetforward_stage5_3_multi_scene_v8.yaml`

---

## 0. 背景与结论

本计划将“增强现有 `Stage5_3` 训练逻辑”改为“新增 `Stage5_3_production` 训练实现”，目标是：

1. 不污染 `MinimalStreetForwardStage5_3` 的最终模型语义实现；
2. bad step 采用 **fast-fail**（直接失败，不做 skip）；
3. checkpoint 使用 **lightweight resume**（仅保存必要训练状态，不追求 bitwise continuation）。

核心原则：

```text
MinimalStreetForwardStage5_3 只负责模型语义
MinimalStreetForwardStage5_3_Production 负责工程化训练能力
```

---

## 1. 改动边界与非目标

### 1.1 改动边界

- 新增 production 训练包装类，继承 `MinimalStreetForwardStage5_3`；
- 新增 production 训练工具与 checkpoint 管理；
- 为 `TrainSchedulerV8` 增加 lightweight 状态导出/恢复接口；
- 新增 production 配置与启动脚本；
- 保持 Stage5_3 的 forward / history / gate / routed decoder 语义不变。

### 1.2 非目标（明确不做）

- 不在 `MinimalStreetForwardStage5_3` 内直接塞入 production 工程逻辑；
- 不做 bitwise resume；
- 不保存 runtime-heavy 状态：`h_cache`、history dict、block support、NodeState runtime、dataset/preload/cache、RNG。

---

## 2. 类与文件拆分方案

新增文件：

```text
models/streetforward/minimal_trainer_stage5_3_production.py
models/streetforward/production_training_utils.py
tools/train_minimal_streetforward_stage5_3_production_multi_scene_v8.py
tools/streetforward_checkpointing.py
configs/minimal_streetforward_stage5_3_production_multi_scene_v8.yaml
```

小改文件：

```text
models/streetforward/__init__.py
datasets/train_scheduler_v8.py
```

不改（保持语义稳定）：

```text
models/streetforward/minimal_trainer_stage5_3.py
```

---

## 3. Stage 命名兼容策略

`MinimalStreetForwardStage5_3` 当前会强校验 `model.stage == "5_3"`，生产方案采用以下推荐：

- 推荐：配置保持 `model.stage: "5_3"`，并增加 `model.production_training: true`；
- 不推荐：引入 `5_3_production` 并在子类做“伪装校验”。

推荐理由：

- 不修改父类校验逻辑，风险更低；
- 避免对 Stage5_3 主语义类引入额外分支。

---

## 4. Production 类设计

目标类：

```python
class MinimalStreetForwardStage5_3_Production(MinimalStreetForwardStage5_3):
    """
    Production training wrapper for Stage5_3.

    保持 Stage5_3 模型语义不变，仅替换训练工程能力：
    optimizer / train_step / checkpoint hooks / metrics
    """
```

建议覆盖能力：

1. `_rebuild_optimizer_after_stage5_modules()`  
   使用 production optimizer + scheduler + AMP scaler 构建策略。
2. `train_step(...)`  
   替换为 fast-fail 语义的 production train step。
3. `clear_runtime_state_for_lightweight_resume()`  
   清理 runtime 状态（h_cache/history/block_support），用于 lightweight 恢复后冷启动。

---

## 5. Bad Step 策略：Fast-Fail（不 skip）

Stage5_3 存在 history memory、gate、NodeState 与 block 统计耦合。bad step 若仅“跳过 optimizer step”会导致 runtime 与参数更新脱节，产生静默污染。

因此 production 统一采用：

```text
non-finite loss -> raise
non-finite grad -> raise
grad norm exceed hard threshold -> raise
AMP overflow -> raise（不 silent skip）
```

### 5.1 训练顺序（成功路径）

```text
zero_grad
forward (autocast)
loss finite check
backward (scaled / unscaled)
grad finite check
grad clip
optimizer step
scaler update
lr scheduler step
commit node / hidden / history
log metrics
```

### 5.2 关键约束

- 只有在 optimizer step 成功后，才允许 commit runtime 侧状态；
- 任意 bad step 直接终止，避免继续训练造成不可追踪污染。

---

## 6. Checkpoint 语义：Lightweight Resume

### 6.1 保存内容（必须）

```text
model_state_dict
optimizer_state_dict
lr_scheduler_state_dict
amp_scaler_state_dict
global_step / global_update_step / epoch_idx
TrainSchedulerV8 lightweight state
config / meta
```

### 6.2 不保存内容（明确）

```text
h_cache_bg / distant / rigid
history_memory（branch history dict）
block_support_acc（block 内临时统计）
NodeState runtime
dataset/preload/view caches
RNG state
```

恢复后上述状态全部重新初始化，这是设计语义，不是 bug。

---

## 7. 边界约束：只在 clean boundary 保存

为避免“恢复时位于 episode/block 中间，但历史统计已丢失”的语义不一致，lightweight checkpoint 要求：

```text
save_at: episode_end
allow_mid_episode_save: false
allow_mid_block_save: false
```

建议实现：

- 到达保存间隔时，若尚未到 episode_end，则延后到最近一次 episode_end 再落盘；
- 在保存函数中显式断言 `train_scheduler.is_at_episode_boundary()`。

---

## 8. TrainSchedulerV8 Lightweight 状态接口

在 `datasets/train_scheduler_v8.py` 增加：

1. `production_state_dict()`  
   导出训练位置相关最小状态（global/epoch/episode/block 游标、最小 segment runtime、traversal cursor）。
2. `load_production_state_dict(state)`  
   恢复轻量状态；`current_episode_state` 置空，恢复后从下一 episode 开始。
3. `is_at_episode_boundary()`  
   用于 checkpoint 保存时机断言。

恢复语义关键点：

```text
不回到旧 episode 中间
从 next clean episode 开始继续训练
```

---

## 9. Checkpoint 工具接口草案

新增 `tools/streetforward_checkpointing.py`：

- `save_stage5_3_production_lightweight_checkpoint(...)`
- `load_stage5_3_production_lightweight_checkpoint(...)`

保存函数职责：

- 校验 episode boundary；
- 打包 lightweight payload；
- 按 `keep_last_k` 进行保留策略。

加载函数职责：

- 加载模型与训练组件状态；
- 加载 scheduler lightweight 状态；
- 调用 `model.clear_runtime_state_for_lightweight_resume()`；
- 返回恢复后的 `global_step`。

---

## 10. Production 配置草案（关键字段）

```yaml
model:
  stage: "5_3"
  production_training: true

training:
  amp:
    enable: true
    dtype: fp16
    fail_on_overflow: true
  bad_step:
    policy: fast_fail
    fail_on_nonfinite_loss: true
    fail_on_nonfinite_grad: true
    fail_on_amp_overflow: true
    fail_on_grad_norm_gt: 100.0

checkpoint:
  type: lightweight
  save_at: episode_end
  allow_mid_episode_save: false
  allow_mid_block_save: false
  save_runtime_history: false
  save_h_cache: false
  save_node_state: false
  save_rng_state: false
```

与用户偏好一致：

- fast-fail；
- 非必要不增加默认兜底分支（配置缺失尽量直接报错）。

---

## 11. 恢复后的运行语义（Warm-Runtime Resume）

恢复流程：

```text
load model/optimizer/lr/scaler
load TrainSchedulerV8 lightweight state
align global step
clear h_cache
clear history_memory
clear block_support_acc
re-init NodeState from dataset/assets
continue from next episode boundary
```

预期现象：

- 恢复后前若干 episode 出现冷启动抖动（loss/metrics 短期波动）；
- 这是 lightweight resume 的预期行为。

建议日志显式标记：

```text
resume/runtime_reset = 1
resume/history_reset = 1
resume/node_state_reset = 1
```

---

## 12. 实施步骤（建议顺序）

1. 新建 production trainer 与 training utils（先跑通初始化与 train_step）；
2. 接入 fast-fail bad step 策略并添加关键指标日志；
3. 新增 checkpointing 工具并实现 lightweight save/load；
4. 扩展 `TrainSchedulerV8` lightweight state 接口；
5. 新增 production 配置与训练入口脚本；
6. 在 `__init__` 暴露新 trainer；
7. 进行 smoke test + resume test + boundary test。

---

## 13. 验收标准

### 13.1 功能验收

- 训练可用 `Stage5_3_production` 入口启动；
- bad step 出现时进程立即失败；
- checkpoint 仅含 lightweight 内容，文件体积显著低于 full runtime；
- resume 后从 clean episode boundary 继续训练。

### 13.2 一致性验收

- `MinimalStreetForwardStage5_3` 原有语义与配置校验保持不变；
- production 改动不影响原 `stage5_3` 研究/实验路径。

### 13.3 稳定性验收

- 连续多次 save/load 能稳定恢复训练；
- 无 mid-block/mid-episode 恢复路径；
- 日志中可追踪 resume 冷启动状态。

---

## 14. 最终结论

本方案将模型语义与训练工程能力解耦：

- `MinimalStreetForwardStage5_3` 保持“最终模型语义实现”；
- `MinimalStreetForwardStage5_3_Production` 承担“生产训练能力”；
- bad step 使用 fast-fail，避免 silent corruption；
- checkpoint 使用 lightweight resume，在可控成本下恢复训练进度。

该方案更符合当前目标：稳定迭代 production 训练能力，同时保持 Stage5_3 主实现干净、可维护。
