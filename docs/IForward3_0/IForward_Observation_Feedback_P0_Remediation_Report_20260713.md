# IForward Observation Feedback P0/P1 修复报告

日期：2026-07-13

## 结论

附件指出的两个 P0 均属实，现已修复：

1. feedback alpha 不再被 Stage3.2 的 global step 直接推到满强度；Stage3.3 使用独立 activation clock，并在 checkpoint 中保存和恢复。
2. `forward_parity_interval` 与 `log_feedback_memory` 已接入真实 runtime，不再是仅解析配置。真实 scene 131 / segment 1 CUDA 检查已执行。

同时完成了首次 feedback mode 强制梯度 probe、frozen repair 硬断言、NumPy read-only storage 修复、Stage3.3 training variant/checkpoint 命名，以及 ego-mask 的逐相机显式语义校验。

## Feedback schedule resume 语义

Stage3.3 配置：

```yaml
observation_feedback:
  schedule:
    origin: activation_step
    activation_step: 30000
```

统一计算：

```text
feedback_schedule_step = max(0, global_step - feedback_activation_global_step)
alpha = alpha_schedule(feedback_schedule_step)
```

因此 Stage3.2 step 30000 迁移到 Stage3.3 时从 alpha=0 开始。Stage3.3 checkpoint 新增：

- `feedback_activation_global_step`
- `feedback_schedule_step`
- schedule format/origin

Stage3.3 自身 resume 会严格恢复该状态；Stage3.2 或旧 checkpoint 没有该状态时使用配置 activation origin。origin 跨 resume 变化会直接报错。

训练指标新增：

- `iforward/feedback/activation_global_step`
- `iforward/feedback/schedule_step`

## Runtime forward parity

在每个 schedule interval 首次命中时，使用同一 LocalGS、图像、camera、mask 与 DINO 输入执行：

```text
attached feedback renderer/checkpoint frontend
vs
detached legacy renderer/frontend
```

检查并记录：

- source RGB
- `features_2d`
- `detail_2d`
- DINO native feature（存在时）
- source valid mask

这些是 Stage3 lifting 和后续 event/updater/loss 的完整观测输入；通过后下游继续使用同一份代码与同一实际 forward，不另造第二套 loss 实现。

真实 CUDA 首次检查发现 `features_2d` 并非 bitwise equal：source RGB、detail 和 mask 为严格零差，冻结 DINO 在 CUDA AMP 下重复执行的 context relative RMS 为 `0.00816`、max abs 为 `0.119`。因此 runtime 使用：

- renderer/detail/mask/FP32：`atol/rtol=1e-5` 严格检查；
- AMP `features_2d`：relative RMS `<=1e-2` 且 max abs `<=0.25`；
- 超阈值直接抛出 RuntimeError。

parity 与 feedback memory 指标通过既有 measurement stats 汇入训练日志。`log_feedback_memory=true` 现在记录 observation 开始/结束 allocated、reserved 及 delta。

## 首次 mode probe 与 repair 硬断言

每种 mode 第一次出现时强制注册 probe，之后仍按 interval 采样：

- `trainable_checkpointed`
- `frozen_input_grad_checkpointed`
- `frozen_no_grad`

对 `frozen_input_grad_checkpointed`：

- frontend parameter gradient count 必须为 0；
- 当 alpha>0 且 source probe 已注册时，source LocalGS gradient norm 必须大于 0；
- 违反任一条件立即终止训练。

真实 K=15 high-block repair 结果：frontend grad count `0`，source LocalGS grad norm `2.05e-4`，loss finite，optimizer skip `0`。

## 数据与版本修复

- depth/mask 读取改为 writable NumPy copy 后再 `torch.from_numpy`，消除未定义的 read-only tensor 风险。
- Stage3.3 增加 `training_variant: stage3_3_observation_feedback`。
- checkpoint 前缀改为 `iforward_stage3_3_observation_feedback`，run manifest 同时记录 model version 与 training variant。
- Stage3.3 保持 `require_egocar_mask_template=true`，并设置 `egocar_mask_absent_cameras: [0, 1, 2]`：这些相机确认不存在 ego 遮挡，缺少模板时按全零 mask 处理；其他未声明相机缺少模板仍会 fail-fast。

## 验证

- 相关 IForward、dataset、manifest 回归：`471 passed, 29 warnings`。
- 真实 CUDA observation parity：通过。
- 真实 CUDA K=15 frozen repair：通过。
- 全仓：`1034 passed, 2 skipped, 16 failed`；失败集中在既有 Stage4.4/Stage5.0/Stage5.2 与旧 StreetForward CPU/spconv 测试，不在本次改动路径。

## 未冒充已解决的项目

- Parent VJP 与 relation feedback 仍默认关闭；需要分别做长训练 drift/冲突实验。
- nominal rollout probability 与 effective K budget 的差异属于实验设计，应在 A/B 时按 K budget 对齐。
- DDP 动态冻结仍按既有策略 fail-fast；多 GPU 需要独立的 gradient gating 设计。
- alpha=0 的 legacy 快路径是性能优化，不影响 correctness，本次未启用。
- nuScenes cam 0/1/2 不需要补 ego-mask 资产；其“无遮挡”状态已在配置中显式声明。其他相机若新增到训练列表，仍需提供模板或明确声明无遮挡。

## 2026-07-14：10k validation 与 checkpoint 顺序修复

从头训练在完成 step 9999 后进入 `memory_freeze_after_prefill` validation 时，synthetic event
未携带训练 scheduler 的 `train_2d_mode/distribution_type`，被 observation feedback 严格校验拦截。
现已修复为：

- validate/demo/replay batch 显式携带独立的
  `observation_feedback_eval_mode=frozen_no_grad`；该字段不篡改 distribution metadata，且 policy 只接受只读模式。
- periodic checkpoint 按“已完成训练步数”判断：完成 `0..9999` 后保存 `step9999`，resume 从
  step 10000 开始。
- checkpoint 在所有周期 validation hook 前落盘；validation fail-fast 不再丢失整个 checkpoint interval。

验证：

- policy/runtime/checkpoint 与全部相关 validation 回归：`69 passed`。
- 真实 CUDA `memory_freeze_after_prefill` seq10/seq20：两条 plan 均 completed。
- 真实 CUDA 单步 `save_checkpoint_freq=1 + validation interval=1`：日志确认先保存
  `step0.pt`，再完整执行 validation，最后正常退出。
