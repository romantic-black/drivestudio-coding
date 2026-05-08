# StreetForward BatchEval: Stage5_6 + Sky + segment_finetune_train

本文档描述在 `tools/eval_streetforward_benchmark.py` + `streetforward_eval/runner.py` 的 BatchEval 框架中，如何兼容 Stage5_6 的 nearby-error 反馈、如何在每个 Stage5_6 step 后做一次 sky 更新，以及如何以 `segment_finetune_train` 模式仅微调 Stage5_6 主结构（不训练 error predictor、也不训练 sky branch）。

## 1. Stage5_6 nearby 兼容调度（adjacent as nearby）

### 目标

- **Stage5_6 新增**：`nearby_error_feedback` 会消费 “nearby 目标帧” 的角色（默认 `near_random`）来做额外 error 预测/反馈。
- **BatchEval 兼容**：当协议给定 input offsets（例如 `[1,3,5,7]`），在每个 input offset 的更新 step 中，使用其相邻 offset（\(-1,+1\)）对应的帧作为 nearby 候选。
- **跳过规则**：若相邻帧已经是 input / source / visited target（包括因 `max_target_frames_including_source` 被截断但仍属于 input 的帧），则跳过该相邻帧；候选不足时按 `allow_partial_nearby` 决定是否允许 0/1 个 nearby。

### 当前实现位置

- 入口：`streetforward_eval/runner.py`
  - `RunnerRuntimeConfig.stage5_6_enable_nearby_feedback=true`
  - `RunnerRuntimeConfig.stage5_6_nearby_policy=adjacent_non_input`
  - `RunnerRuntimeConfig.stage5_6_allow_partial_nearby`

### 行为示例

- input offsets = `[1,3,5,7]`，对 offset=1 的 block：
  - adjacent offsets = `[0,2]`
  - 若 `[0,2]` 都不属于 input offsets 且不在 source/visited targets，则作为 nearby。

## 2. “Stage5_6 一步优化后，Sky 一步优化”

### 目标

每次对 StreetForward（Stage5_6）做一次 update step 后，立即对 sky 状态做一次更新，使 sky 预测与最新的 scene 状态对齐。

### 设计说明

- sky branch 在 BatchEval 中通常 **参数冻结**（`freeze_params=true`），这里的一步“优化”指：
  - 通过 `SkyBranchV0.forward_scene_batch(..., writeback=True)` 更新 sky 的 **runtime state**（node state + hidden cache）
  - 不对 sky 网络参数做梯度更新（BatchEval 的 sky 路径默认 `torch.no_grad()`）

### 当前实现位置

- `streetforward_eval/runner.py`
  - `_run_stage5_6_update_step()` 完成 Stage5_6 update
  - 随后 `_run_sky_update_after_scene_step()` 做一次 sky state writeback

## 3. segment_finetune_train eval mode：只微调 5_6 主结构

### 目标

在 eval（BatchEval）期间做 “segment-level finetune”，但严格约束：

- **只训练 Stage5_6 主结构**（和正常训练路径一致）
- **不训练**：
  - Stage5_6 error predictor（`stage5_6_error_head*`）
  - Stage5_6 nearby feedback fuser（`stage5_6_*_fuser*`，除非显式开启）
  - SkyBranch（单独模型，不在同一个 optimizer 中）
- **scheduler / history / cache 行为与训练保持一致**：
  - aligned info（`_scheduler_v8_aligned_info`）必须包含 `episode_idx_global/block_idx_global/block_repeat_step/visited_block_indices/...`
  - history 的触发策略（`history_record_on_input_exit` / `history_record_each_step`）需与训练时的 `history_memory.record_on` 对齐
  - cache（Stage5_6 nearby_error_feedback.cache）依赖 `episode_idx_global/block_idx_global` 作为作用域键，且在 `record_block_history()` 后清理

### 实现概览

- 入口脚本：`tools/eval_streetforward_benchmark.py`
  - 当 `batch_eval.runtime.mode=segment_finetune_train` 时配置一个 AdamW optimizer，仅包含被选中的“主结构”参数集合
  - 通过前缀/contains 匹配选择主结构参数，并显式排除 `stage5_6_error_head*` 等分支

- runner：`streetforward_eval/runner.py`
  - 当 `stage5_6_enable_nearby_feedback=true` 时走“train-batch path”
  - `runtime.mode=segment_finetune_train` 时调用 `model.train_step()`，保证主结构更新逻辑与正常训练一致

## 4. Q1：segment_finetune_train 模式下，history 有没有监督？

结论：**有**（history 会被更新），但监督信号来自 **历史 input 对应的 target 视图**，不会引入“未来帧”的真值泄露。

解释（与代码行为对应）：

- history memory 的更新发生在 `eval_sparse_record_history()`（BatchEval）或训练中的 `record_block_history()` 路径；
- `history_memory.record_views` 通常配置为 `source_image_refs`，即 **输入帧的相机视图**；
- 因此 history 的 residual/support 统计，是根据这些输入视图上的 photometric residual 等信号更新的，本质上仍是“用输入监督输入的历史记忆”，不包含未来 eval offsets 的 GT。

## 5. 配置与运行（Stage5_6 + Sky）

建议使用配置：`configs/eval/streetforward_batcheval_stage5_6_sky.yaml`

- Stage5_6：
  - checkpoint: `/root/autodl-tmp/outputs/minimal_sf_stage5_6_multi_scene_v8_adamw_warmcos/experiment002_60w_40w_block16/checkpoints/minimal_sf_stage5_6_multi_scene_v8_step250000.pt`
  - base_config_file: `/root/autodl-tmp/outputs/minimal_sf_stage5_6_multi_scene_v8_adamw_warmcos/experiment002_60w_40w_block16/config.yaml`
- Sky：
  - checkpoint: `/root/autodl-tmp/outputs/skybranch_stage5_4_exp002/checkpoints/skybranch_resume_step_100000.pth`
  - config: `configs/skybranch_stage5_4_exp002.yaml`

运行环境：

```bash
conda activate drivestudio-new
export PYTHONPATH=/root/drivestudio-coding
python tools/eval_streetforward_benchmark.py --config_file configs/eval/streetforward_batcheval_stage5_6_sky.yaml
```

