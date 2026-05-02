# StreetForward Stage6_0 Decoupled Teacher-Student 详细设计

## 文档目标

本文在对照以下现有实现的基础上，给出 `Stage6_0` 的可落地方案：

- `datasets/train_scheduler_v9.py`
- `configs/minimal_streetforward_stage5_5_multi_scene_v9.yaml`
- `models/streetforward/minimal_trainer_stage5_5.py`

目标是解决 `Stage5_5 + SchedulerV9` 中目标域混合、命名歧义和 bridge 边界不清的问题，形成可验证、可迁移、可 fast-fail 的 `Stage6_0` 设计。

---

## 0. 一句话定义

```text
Stage6_0 = Decoupled Teacher-Student StreetForward with Teacher Anchor and Live Bridge
```

语义定义：

```text
Teacher 负责真实观测；
Student 负责无 GT 输入的传播更新；
Bridge 负责 teacher prior 到 student 视角的投影（live/cached）；
History 负责长期状态；
Near 仅用于 probe（默认不参与训练反传）。
```

---

## 1. 现状问题（来自 Stage5_5/V9）

### 1.1 SchedulerV9 目标域仍是扁平 role 列表

当前 `train_scheduler_v9.py` 的目标构造中，student 分支会把目标混入：

- `student_source`
- `teacher_preserve`
- `visited`
- `near_random`（按配置打开）

这使得 `loss_total` 难以解释，且 role 语义容易漂移。

### 1.2 Stage5_5 命名与监督语义耦合

`minimal_trainer_stage5_5.py` 中权重 key 要求固定包含：

- `teacher_source`
- `student_source`
- `teacher_preserve`
- `visited`
- `near_random`

`teacher_preserve` 在语义上是 “student 更新后对 teacher 锚点的保持约束”，但名称容易被误读为 teacher 自身 preserve，或 student preserve。

### 1.3 Cache 已经 detach（这是正确的），但 live 可微路径缺失

`_stage5_5_update_teacher_prior_cache` 当前写入使用：

- `.detach().float()`

这是跨 step persistent cache 的正确做法；但在仅使用 cache prior 的情况下，student loss 无法改善当前 step 的 teacher prior 表征。

### 1.4 near_random 默认参与训练监督

`stage5_5_multi_scene_v9` 配置中 `near_random` 存在非零权重选项，导致 near 既作为诊断域又作为训练域，影响可解释性。

---

## 2. Stage6_0 设计原则

`Stage6_0` 固定四个监督/评估域：

- `self`
- `teacher_anchor`
- `history`
- `probe`

并强制边界：

1. 删除 `student_preserve/teacher_preserve/preserve` 的训练语义；
2. `near` 默认只做 probe；
3. `live_bridge` 必须 current-step 重跑 teacher 2D feature；
4. `persistent cache` 永远 detach；
5. `student valid_mask` 必须在 `student_unet` 输入前应用，且作为输入 channel。

---

## 3. 命名规范（最终版）

### 3.1 废弃命名

以下名称在 `Stage6_0` 中视为废弃：

- `student_preserve`
- `teacher_preserve`
- `preserve`
- `anchor_preserve`

兼容迁移仅允许：

```text
teacher_preserve -> teacher_anchor
```

但日志必须只输出新名字。

### 3.2 新命名

训练域命名统一为：

- `loss/self/teacher`
- `loss/self/student`
- `loss/teacher_anchor`
- `loss/history/visited`
- `probe/near/*`（仅 metric，默认不反传）

---

## 4. 监督域拆分

### 4.1 self

- `teacher_self`: teacher path 使用 `GT + render` 进行观测监督。
- `student_self`: student path 不看 GT，仅用 `render + projected_teacher_prior + history + valid_mask` 更新后监督。

### 4.2 teacher_anchor

只在 student step 之后计算：

```text
student update state
-> render teacher anchor frame
-> compute teacher_anchor loss
```

作用是防止 student 更新破坏 teacher 已观测结构。第一版建议 `weight=0.1`。

### 4.3 history

`history_targets` 仅来自 committed visited frames，不包含：

- near
- teacher_anchor
- student_source

### 4.4 probe

`near_neighbor/near_random` 默认仅记录指标：

- `l1`
- `psnr`
- `ssim`
- `lpips`

默认：

- `loss_weight = 0.0`
- `log_only = true`

---

## 5. Bridge 设计：Live vs Cache

### 5.1 Live bridge（current-step，可微）

Student step 中必须支持：

1. current-step 重跑 teacher 2D feature；
2. 投影到 student source 视角得到 prior/conf；
3. student loss 可回传到 `teacher_prior_adapter`；
4. 初期不回传 teacher backbone / dino。

建议梯度边界：

- `student_loss_to_teacher_prior_adapter: true`
- `student_loss_to_teacher_backbone: false`
- `student_loss_to_teacher_dino: false`

### 5.2 Cache bridge（跨 step，持久）

persistent cache 只做：

- fallback
- eval 推理
- 跨 step prior 提供
- coverage 诊断

写入必须保持 detach：

```python
cache.feat[...] = teacher_prior.detach().float()
cache.support[...] = support.detach().float()
cache.valid[...] = True
```

禁止跨 step 保留计算图。

---

## 6. Valid Mask 规则（Stage6_0 强制）

### 6.1 构建

建议：

```text
student_valid_mask =
  source_pair_valid_mask
  & camera_valid_mask
  & not_sky
  & not_egocar
```

若需要降采样到 feature 分辨率，强制 `nearest`。

### 6.2 输入前 mask

在 `student_unet` 前对输入硬 mask：

- `render_rgb *= valid_mask`
- `prior_map *= valid_mask`
- `prior_conf *= valid_mask`
- `history_context *= valid_mask`（如启用）

并把 `valid_mask` 作为输入 channel 拼接。

### 6.3 输出后再 mask

`feat2d_student *= valid_mask_feat`，再做 backprojection，形成双重防线。

---

## 7. 模块结构与继承关系

`Stage6_0` 建议新增文件：

```text
models/streetforward/minimal_trainer_stage6_0.py
models/streetforward/stage6_teacher.py
models/streetforward/stage6_student.py
models/streetforward/stage6_bridge.py
models/streetforward/stage6_losses.py
datasets/train_scheduler_v10.py
configs/minimal_streetforward_stage6_0_multi_scene_v10.yaml
tools/train_minimal_streetforward_stage6_0_multi_scene_v10.py
```

继承关系：

```python
class MinimalStreetForwardStage6_0(MinimalStreetForwardStage5_4):
    ...
```

不建议继承 `Stage5_5`，避免把 `teacher_preserve/near_random` 的历史耦合语义带入。

---

## 8. Stage6_0 Forward 流程

### 8.1 Teacher observe step

```text
teacher_module(gt, render, valid_mask, views, gaussians)
-> teacher_prior_live
-> cache_bridge.write_detached(...)
-> history.observed(teacher_only)
-> history.runtime(teacher_and_student)
-> loss/self/teacher
```

### 8.2 Student propagate step

```text
if live_bridge.enable:
    rerun teacher 2D current-step
    project live prior -> student source
else:
    read cache prior

build student_valid_mask
student_module(render, prior, conf, history, valid_mask)
feat2d_student *= valid_mask_feat
backproject_v4(pair_valid_mask=student_valid_mask)
update state
render + losses
```

loss 构成：

- `loss/self/student`
- `loss/teacher_anchor`
- `loss/history/visited`
- `probe/near/*`（metric only）

---

## 9. Loss 聚合（结构化输出）

Stage6_0 采用结构化 loss 输出，而非扁平 target role 列表：

```python
Stage6LossOutput(
    total_train=...,
    self_loss={"teacher": ..., "student": ...},
    teacher_anchor_loss=...,
    history_loss=...,
    probe_metrics={"near": ...},
)
```

总 loss：

```text
loss_total_train =
  w_self_teacher * L_self_teacher
  + w_self_student * L_self_student
  + w_teacher_anchor * L_teacher_anchor
  + w_history * L_history
```

不包含 `probe_near`。

---

## 10. SchedulerV10 设计

`Stage6_0` 建议启用 `TrainSchedulerV10`，输出结构化 request，而非单纯 role 采样：

- `teacher_obs`
- `student_prop`
- `teacher_anchor`
- `history_targets`
- `probe_near`

核心变化：

- 不再输出 `teacher_preserve_role_name`
- 不再输出 `student_preserve_role_name`
- 不再以 `near_random_weight` 混入总训练目标

---

## 11. History Memory 规则

保持 observed/runtime split，并强制 writer 边界：

- `observed_history.writer = teacher_only`
- `runtime_history.writer = teacher_and_student`

`near` 不写 observed history，也不进入 history targets。

---

## 12. 配置草案（推荐默认）

关键默认建议：

- `self.teacher_weight = 0.2`
- `self.student_weight = 1.0`
- `teacher_anchor.weight = 0.1`
- `history.visited_weight = 0.1`
- `probe.near.loss_weight = 0.0`
- `bridge.live.enable = true`
- `bridge.cache.detach_write = true`
- `student.valid_mask.apply_before_unet = true`
- `student.valid_mask.append_as_channel = true`

---

## 13. 日志规范

训练域日志：

- `loss/total_train`
- `loss/self/teacher`
- `loss/self/student`
- `loss/teacher_anchor`
- `loss/history/visited`

probe 指标：

- `probe/near/l1`
- `probe/near/psnr`
- `probe/near/ssim`
- `probe/near/lpips`

bridge 诊断：

- `bridge/live/enabled`
- `bridge/live/rerun_teacher_2d`
- `bridge/live/prior_conf_mean`
- `bridge/live/prior_conf_nonzero_ratio`
- `bridge/live/grad_teacher_adapter_norm`
- `bridge/live/grad_teacher_backbone_norm`
- `bridge/cache/fallback_ratio`

valid mask 诊断：

- `student_valid/mask_ratio`
- `student_valid/render_masked_mean`
- `student_valid/prior_masked_mean`
- `student_valid/output_invalid_abs_mean`

warmup 期要求：

- `grad_teacher_backbone_norm == 0`
- `grad_teacher_adapter_norm > 0`
- `output_invalid_abs_mean -> 0`

---

## 14. Fast-fail 规则

### 14.1 废弃 key 拦截

配置出现以下任一 key，直接报错（或仅做兼容映射）：

- `student_preserve`
- `teacher_preserve`
- `preserve`

兼容映射仅允许：

```text
teacher_preserve -> teacher_anchor
```

### 14.2 训练安全拦截

以下条件触发 fast-fail：

- `near.loss_weight != 0` 且未进入显式 near-train phase
- `live_bridge.enable=true` 且 `rerun_teacher_2d_current_step=false`
- `cache.detach_write=false`
- `student.valid_mask.apply_before_unet=false`
- `student.valid_mask.append_as_channel=false`
- warmup 阶段 `student_loss_to_teacher_backbone=true`

---

## 15. 训练阶段建议

### Phase A: Teacher warm start

- 从 `Stage5_4` checkpoint 初始化
- near 只 probe

### Phase B: Student + live adapter

- 训练 `student_unet + teacher_prior_adapter`
- 冻结 teacher backbone/dino
- `teacher_anchor` 先设 `0.0 ~ 0.05`

### Phase C: 加入 teacher_anchor

- `teacher_anchor = 0.1`
- `history = 0.0 ~ 0.05`

### Phase D: 加入 history

- `history = 0.1`
- `teacher_anchor = 0.1`

### Phase E: Full Stage6_0

- `self/student = 1.0`
- `self/teacher = 0.2`
- `teacher_anchor = 0.1`
- `history = 0.1`
- `probe/near = 0.0`

---

## 16. 评测规则

评测默认关闭 live 可微桥，使用 cache bridge：

```yaml
stage6_0:
  bridge:
    live:
      enable: false
    cache:
      enable: true
```

流程：

1. 输入帧：teacher observe，更新 state/cache/history；
2. 预测帧：student propagate（无 GT 输入）；
3. 统计 `input / non-input / all-frame` 分域指标。

---

## 17. 迁移清单（从 Stage5_5/V9 到 Stage6_0）

### 17.1 Scheduler

- 从 `TrainSchedulerV9` 迁移到 `TrainSchedulerV10`
- 从 `target_roles + target_weights` 迁移到结构化 request

### 17.2 Trainer

- 从 `teacher_preserve` 迁移为 `teacher_anchor`
- 引入 `live_bridge` current-step 重跑路径
- 保持 `cache_bridge.detach_write=true`
- 强制 student valid mask 输入前处理

### 17.3 Config

- 去除默认训练域中的 `near_random` 权重
- 增加 `stage6_0.bridge.live/cache` 显式开关
- 增加 `stage6_0.student.valid_mask.*` 强约束项

---

## 18. 结论

`Stage6_0` 的核心不是继续堆叠 role，而是建立清晰边界：

- `self` 负责学习当前传播；
- `teacher_anchor` 负责不破坏 teacher 已观测结构；
- `history` 负责长期保持；
- `probe_near` 负责诊断泛化而非默认训练。

在此边界下，`live_bridge` 与 `cache_bridge` 的职责可被清楚验证，日志可解释性明显高于 `Stage5_5 + SchedulerV9` 的扁平 role 方案。
