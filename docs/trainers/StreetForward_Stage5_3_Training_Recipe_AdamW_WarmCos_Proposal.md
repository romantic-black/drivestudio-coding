# StreetForward Stage5_3 训练 Recipe 改造方案（AdamW + Warmup Cosine）

## 结论

本次改动**不建议新建 `Stage5_4`**。  
这是训练系统改造，不是结构语义改造。

但 production 落地应采用**双类分层**，避免污染 Stage5_3 语义实现：

1. `MinimalStreetForwardStage5_3`：保留当前 minimal/final model 语义；
2. `MinimalStreetForwardStage5_3_Production`：继承 Stage5_3，承载 AdamW + warmup cosine + grad clip + fast-fail + production checkpoint；
3. 配置层仍保持 `model.stage: "5_3"`，通过 `model.production_training: true` + `training_recipe.name` 进入 production 训练路径。

---

## 背景与现状

基于当前代码与配置：

- `models/streetforward/minimal_trainer_stage5_3.py` 在 `__init__` 末尾执行 `_init_stage5_3_modules()` 后调用 `_rebuild_optimizer_after_stage5_modules()`；
- `_rebuild_optimizer_after_stage5_modules()` 当前是硬编码单组 `torch.optim.Adam`；
- `configs/minimal_streetforward_stage5_3_multi_scene_v8.yaml` 当前 optimizer 仍是：
  - `lr=1e-4`
  - `eps=1e-15`
  - `weight_decay=0.0`
- `minimal_trainer_stage5_2.py`、`minimal_trainer_stage5_0.py` 同样是同类 Adam 重建模式。

这说明 Stage5_3 已经具备“在模块初始化完成后统一重建 optimizer”的天然改造入口，不需要引入新的模型 stage。

---

## 为什么不新建 Stage5_4

### 1) 不是结构变更

Stage5_3 的结构约束不变：full-routed near/far、xCPE + MLP、history memory、update gate、DINOv2/UNet fusion。

### 2) 不改变参数键语义

AdamW/param groups/LR schedule 仅影响训练动态，不改变 `model.state_dict()` 参数键。  
旧 Stage5_3 权重的验证路径可保持一致。

### 3) 避免错误语义

若命名为 Stage5_4，容易让人误解为网络结构迭代，而实际只是训练 recipe 升级。  
更合理的是“Stage5_3 + recipe versioning”。

建议表达：

```yaml
model:
  stage: "5_3"

training_recipe:
  name: "stage5_3_adamw_warmcos_v1"
```

---

## 何时才需要 Stage5_4

仅当发生结构/语义变更时再升级 stage：

- 修改 xCPE / far branch 主体结构；
- 修改 gate 输入语义或维度；
- 修改 history memory buffer 语义或其 state_dict 键；
- 修改 forward 输出或 validation 推理逻辑。

以下不需要新 stage：

- optimizer/scheduler 调整；
- 训练配置调整；
- loss 权重调整；
- 冻结策略调整（例如 DINO freeze 开关）。

---

## 配置设计（兼容式 + production 强约束）

保留旧字段兼容，同时支持新配置：

```yaml
model:
  stage: "5_3"
  production_training: true

training_recipe:
  name: stage5_3_adamw_warmcos_v1

optimizer:
  type: adamw
  lr: 1.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 1.0e-4
  filter_frozen: true

  no_weight_decay:
    enable: true
    name_keywords: [".bias", "bias", "norm", "Norm", "ln", "LayerNorm", "embedding", "Embedding"]
    ndim_leq: 1

  groups:
    # ... dino / residual_unet / fusion_neck / struct_near_xcpe / struct_far_mlp / gate_history / recurrent_update / default

lr_scheduler:
  enable: true
  type: warmup_cosine
  interval: step
  warmup_steps: 3000
  total_steps: ${training.max_iterations}
  min_lr_ratio: 0.05
  warmup_start_ratio: 0.05

training:
  amp:
    enable: false
    dtype: fp16
  grad_clip:
    enable: true
    max_norm: 1.0
    norm_type: 2.0
  bad_step:
    policy: fast_fail
    fail_on_nonfinite_loss: true
    fail_on_nonfinite_grad: true
    fail_on_amp_overflow: false
    fail_on_grad_norm_gt: 100.0
```

AMP 策略：

- production 默认关闭 AMP；
- 不把 AMP 写入主 production recipe（避免引入额外训练不确定性）。

兼容策略分层：

- `Stage5_3`（非 production）可兼容 legacy Adam；
- `Stage5_3_Production` 必须：
  - `optimizer.type=adamw`
  - `lr_scheduler.type=warmup_cosine`
  - `training_recipe.name` 非空  
  任一不满足直接 fast-fail 报错，不允许 silent fallback。

---

## 实现建议

新增文件：

`models/streetforward/training_optim.py`

建议包含：

- `build_streetforward_optimizer(model, config)`
- `build_streetforward_lr_scheduler(optimizer, config, start_step=0)`
- `warmup_cosine_factor(...)`
- `StreetForwardWarmupCosineLR`
- `optimizer_group_signature(optimizer)`

新增 production trainer：

`models/streetforward/minimal_trainer_stage5_3_production.py`

建议职责：

- 继承 `MinimalStreetForwardStage5_3`；
- 覆盖 `_rebuild_optimizer_after_stage5_modules()`（仅生产版启用 AdamW/warmcos 强约束）；
- 覆盖训练 step（接入 grad clip / fast-fail / scheduler step / production 日志）；
- 覆盖或扩展 checkpoint 保存/恢复入口（production lightweight 语义）。

### 参数分组规则

1. 遍历 `model.named_parameters()`；
2. 可选过滤 `requires_grad=False`（默认开启）；
3. 按 prefix/contains 命中逻辑组；
4. 再按 decay/no_decay 拆子组；
5. param group 附带 `name` 和 `param_names` 便于日志与校验。

production 必须追加严格校验：

- 每个 trainable parameter **必须且只能**命中一个 logical group；
- 不允许重复命中；
- 不允许漏分组；
- `dino.freeze=true` 时 DINO 参数不得进入 optimizer；
- group 顺序必须固定（确保 `optimizer.state_dict()` 兼容性和可复现性）。

### no weight decay 规则

默认对以下参数禁用 weight decay：

- 参数维度 `ndim <= 1`
- 名称含 `bias/norm/layernorm/embedding` 等关键字

### DINO 冻结处理

当 `dino.freeze=true` 且参数 `requires_grad=False` 时不进入 optimizer，避免无效 state。

---

## LR 调度策略

采用**全局 optimizer step**，不要依赖 episode/block 状态。  
理由：TrainSchedulerV8 训练与验证的 block/step 配置可不同，且 traversal/reset 策略不应耦合学习率。

推荐 warmup + cosine 因子：

```python
def warmup_cosine_factor(step, warmup_steps, total_steps, min_lr_ratio, warmup_start_ratio):
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        t = step / max(1, warmup_steps)
        return warmup_start_ratio + t * (1.0 - warmup_start_ratio)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
```

训练循环顺序：

1. `loss.backward()`
2. `total_grad_norm = clip_grad_norm_(..., error_if_nonfinite=True)`
3. `if total_grad_norm > fail_on_grad_norm_gt: raise FloatingPointError`
4. `optimizer.step()`
5. `lr_scheduler.step()`

同时要求 production fast-fail：

- `loss` 非有限值：立即报错；
- grad 非有限值：通过 `clip_grad_norm_(error_if_nonfinite=True)` 立即报错；
- grad norm 超阈值：立即报错并中断训练。

---

## 代码接入点（避免污染 Stage5_3 语义类）

不建议直接改写 `minimal_trainer_stage5_3.py` 主类语义。  
推荐在 `minimal_trainer_stage5_3_production.py` 中覆盖：

- `_rebuild_optimizer_after_stage5_modules()`：
  - `self.optimizer = build_streetforward_optimizer(...)`
  - `self.lr_scheduler = build_streetforward_lr_scheduler(...)`
- `train_step()`（或对应训练循环入口）：
  - `loss.backward()`
  - grad clip + fast-fail
  - `optimizer.step()`
  - `lr_scheduler.step()`

说明：Stage5_3 已在 `_init_stage5_3_modules()` 之后重建 optimizer，production 子类复用该时机即可完整覆盖 feature extractor、struct decoder、gate/history 等参数。

---

## Checkpoint 方案（lightweight）

目标：可继续训练，不追求“完整运行态恢复”。

建议保存：

- 必须：`model.state_dict()`
- 建议：`optimizer.state_dict()`
- 必要元信息：`global_step`、optimizer 配置快照、optimizer group signature、lr scheduler 配置快照

可不保存：

- TrainSchedulerV8 runtime 状态
- dataset/preload cache
- history memory / NodeState cache（默认不保存，允许恢复后重新 warm-up）

---

## 兼容性策略

加载 checkpoint 时分三类：

1. **旧 model-only ckpt**：加载 model，optimizer/scheduler 用当前配置重建；
2. **旧 Adam 单组 ckpt**：若 group signature 不匹配，跳过 optimizer state，仅 model warm-start；
3. **新 AdamW 多组 ckpt**：signature 匹配时加载 optimizer state。

不建议强行映射旧 Adam state 到新多组 AdamW，风险高且收益有限。

---

## 日志与可观测性

训练启动时打印：

- optimizer 类型、可训练/冻结参数量；
- 每个 param group 的 name、参数量、lr、wd；
- scheduler 配置（warmup_steps/total_steps/min_lr_ratio/warmup_start_ratio）。

production 建议固定打印以下关键项（用于快速排障）：

- `optimizer/group/default/num_params`
- `optimizer/group/residual_unet/num_params`
- `optimizer/group/fusion_neck/num_params`
- `optimizer/group/struct_near_xcpe/num_params`
- `optimizer/group/struct_far_mlp/num_params`
- `optimizer/group/gate_history/num_params`
- `optimizer/group/recurrent_update/num_params`
- `optimizer/frozen/dino/num_params`
- `optimizer/unassigned_trainable_params`

当 `optimizer/unassigned_trainable_params > 0` 时，production 直接报错。

按 `log_interval` 输出关键 lr：

- `lr/default`
- `lr/struct_near_xcpe`
- `lr/fusion_neck`
- `lr/gate_history`
- `optimizer/global_step`

---

## 分阶段实施建议

### Phase 1（最小风险）

- 新增 `Stage5_3_Production`，不改或最小改动 `Stage5_3` 语义类；
- production 路径接入 helper + AdamW warmcos + grad clip + fast-fail；
- 旧 Stage5_3 配置保持可跑（legacy 行为仅限非 production）；
- 新增一份配置文件，不覆盖现有配置。

### Phase 2（可选统一）

- 将 Stage5_2 / Stage4_6 的 optimizer rebuild 也统一到 helper；
- 保持 legacy 配置行为不变。

---

## 最终建议

1. 不新建 Stage5_4；
2. 保留 `model.stage: "5_3"`，并新增 `Stage5_3_Production` 训练子类；
3. production 强制 `adamw + warmup_cosine + 非空 training_recipe.name`；
4. 训练循环必须包含 grad clip 与 nonfinite/bad-step fast-fail；
5. param group 必须严格“唯一命中、无漏分组、固定顺序、可观测可审计”；
6. lightweight checkpoint 满足可续训，不追求完整 runtime 还原。

该方案能在不改变 Stage5_3 结构语义的前提下，提供 production 训练需要的稳定性和可追责性。
