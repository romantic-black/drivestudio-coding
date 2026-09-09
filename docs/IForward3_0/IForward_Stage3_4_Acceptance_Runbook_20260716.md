# IForward Stage 3.4 v57 Acceptance Runbook

本文是 `stage3_4_functional_parentgs_lift` v57 的可执行验收手册。正式门禁包括：

1. 自动测试；
2. 从 native Stage 3.3 checkpoint 做 weights-only 的10-step post-fix smoke；
3. 使用 smoke 产出的 native v57 checkpoint运行 Validation v4 quick/full；
4. K=15 bounded profile。

1000-step matched B/C 仅在本文最后保留实验协议和延期命令模板，**本轮不实现、不执行，
也不属于 P0 完成条件**。旧 pre-v57 13D Stage 3.4 checkpoint不能用于
smoke、validation或resume。

---

## 1. 固定环境与输入

所有命令使用 nuScenes scene `131`、segment `1`、seed `41` 和
`drivestudio-new` conda环境。

```bash
export ROOT=/root/drivestudio-coding
export CFG=$ROOT/configs/iforward/iforward_stage3_4_functional_parentgs_lift.yaml
export INIT_CKPT=/root/autodl-tmp/outputs/iforward_stage3_3_observation_feedback_from_scratch_60k_validation_fixed_20260714/checkpoints/iforward_stage3_3_observation_feedback_step39999.pt
export SMOKE_OUT=/root/autodl-tmp/outputs/iforward_stage3_4_v57_postfix_smoke_20260716
export QUICK_VAL_OUT=/root/autodl-tmp/outputs/iforward_stage3_4_v57_validation_v4_seq10
export FULL_VAL_OUT=/root/autodl-tmp/outputs/iforward_stage3_4_v57_validation_v4_full
export PROFILE_OUT=/root/autodl-tmp/outputs/iforward_stage3_4_v57_k15/profile.json

cd "$ROOT"
```

开始前检查输入 checkpoint存在：

```bash
test -s "$INIT_CKPT"
```

不要复用旧 smoke输出目录。若目录已经存在，换一个新目录名；本手册不授权删除已有
结果。

---

## 2. 自动测试门禁

```bash
conda run -n drivestudio-new --no-capture-output \
  env PYTHONPATH=$ROOT pytest -q \
  tests/test_iforward_functional_parentgs.py \
  tests/test_iforward_stage3_4_observe.py \
  tests/test_iforward_parent_ptv3_stage2_1.py \
  tests/test_iforward_stage3_4_checkpoint.py \
  tests/test_iforward_stage3_4_config.py \
  tests/test_iforward_validate_v4.py \
  tests/iforward/validation_v4/test_contract.py \
  tests/test_iforward_stage2_3_validation.py \
  tests/iforward/runtime/test_stage3_adapter.py \
  tests/test_iforward_rollout.py
```

必须覆盖并通过：

- exact CUDA/forward-only/reference parity与directional derivative；
- bg、distant、rigid-active三分支 attachment；
- alpha=0/0.25/1 forward parity和Jacobian比例；
- uniform scale shift可观测、quaternion/SH隔离；
- K=2/K=3 no-grad、`frozen_no_grad`、`validation_render_only`；
- source/Functional Parent独立gate；
- lifting/PTV3/support/relation边界隔离；
- Stage 3.3 weights-only迁移和zero-init residual；
- native v57 strict resume以及旧13D Stage 3.4拒绝；
- Validation v4 contract生成与失败传播。

任一测试失败时停止，不启动 smoke 或 Validation v4。

---

## 3. 10-step post-fix smoke

此运行不从0训练。它从 native Stage 3.3 step39999 checkpoint做 weights-only 初始化，
加载 detached legacy 17D codec与 downstream token projection，只初始化新的 zero-init 8D
geometry residual。

```bash
conda run -n drivestudio-new --no-capture-output \
  env PYTHONPATH=$ROOT python $ROOT/tools/train_iforward_one_segment.py \
  --config_file "$CFG" --max_steps 10 --seed 41 \
  --init_checkpoint "$INIT_CKPT" --init_weights_only \
  initialization.skip_keys=[] \
  training.save_checkpoint_freq=10 \
  scheduler_stage3_0.traversal.fixed_scene_id=131 \
  scheduler_stage3_0.traversal.fixed_segment_id=1 \
  scheduler_stage3_0.traversal.seed=41 \
  scheduler_stage3_0.producer.enable=false \
  'data.train_scene_ids=[131]' 'data.eval_scene_ids=[]' \
  data.pixel_source.require_egocar_mask_template=false \
  dataset.preload_scene_count=1 \
  scheduler_stage3_0_validation.enable=false \
  iforward_validation_v4.enable=false \
  eval.run_test_at_end=false \
  logging.metrics_history_append=false \
  logging.train_step_metrics_interval=1 \
  logging.scheduler_metrics_interval=1 \
  model.iforward.observation_feedback.debug.grad_probe_interval=1 \
  output_name=iforward_stage3_4_v57_postfix_smoke \
  logging.project=iforward_stage3_4_v57_postfix_smoke \
  logging.log_dir="$SMOKE_OUT"
```

最终 checkpoint：

```bash
export CKPT=$SMOKE_OUT/checkpoints/iforward_stage3_4_functional_parentgs_lift_final.pt
test -s "$CKPT"
```

smoke必须满足：

- checkpoint声明 `stage3_4_functional_parentgs_lift` version/variant；
- `parent_codec_schema=legacy17d_plus_geometry8d_residual_v1`；
- 迁移日志表明 legacy codec和 `param_support_proj`已加载，不是 skipped；
- 新 residual adapter在加载时为精确 zero-init，随后进入 optimizer；
- 无 OOM、NaN、Inf或 skipped optimizer step；
- 存在 bg，并按数据实际情况记录 distant/rigid-active；
- first visit forward-only；之后仅在统一 gate 为真且该 branch 当前确有 live geometry graph
  时 attached。rollout boundary 后未被当前 rollout updater 重新连接的 present branch 必须
  exact forward-only；
- source-feedback parity probe 的 feedback/reference 必须复用同一静态 DINO cache entry；
  chunked 与 unchunked 情况均不得因重复 DINO AMP 前向而依赖放宽容差；
- 无 legacy runtime/VJP/relation/drift/refresh指标。

不要把旧 Stage 3.4 smoke checkpoint复制或重命名成 `$CKPT`；Validation v4 会检查原生
version/variant/schema并拒绝它。

---

## 4. Validation v4 quick gate

先只运行 seq10、full memory、一个repair permutation，不保存图像：

```bash
conda run -n drivestudio-new --no-capture-output \
  env PYTHONPATH=$ROOT python $ROOT/tools/iforward_validate_v4.py \
  --config_file "$CFG" --checkpoint "$CKPT" \
  --output_dir "$QUICK_VAL_OUT" \
  --device cuda --max_entries 1 --frame_sets seq10 \
  --repair_permutations 1 --memory_ablation full \
  --image_policy none --trigger_step 15
```

该命令会在写出 `validation_contract.json` 后执行强校验，任一合同失败均返回非零。检查：

```bash
test -s "$QUICK_VAL_OUT/validation_contract.json"
test -s "$QUICK_VAL_OUT/index.html"

conda run -n drivestudio-new --no-capture-output \
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps(p,indent=2,ensure_ascii=False)); assert p["status"] == "passed"' \
  "$QUICK_VAL_OUT/validation_contract.json"
```

quick gate失败时停止，不运行 full validation。

---

## 5. Validation v4 full gate

完整协议使用 seq10/seq20、三个repair permutations及全部八种memory mode：

```bash
conda run -n drivestudio-new --no-capture-output \
  env PYTHONPATH=$ROOT python $ROOT/tools/iforward_validate_v4.py \
  --config_file "$CFG" --checkpoint "$CKPT" \
  --output_dir "$FULL_VAL_OUT" \
  --device cuda --max_entries 1 --frame_sets seq10,seq20 \
  --repair_permutations 3 \
  --memory_ablation full,memory_off,memory_read_write,memory_freeze_write,memory_shuffle_state,memory_shuffle_read_write_state,memory_freeze_after_prefill,memory_wrong_parent_key_fixed \
  --image_policy first_plan_only --trigger_step 9999
```

检查：

```bash
test -s "$FULL_VAL_OUT/validation_contract.json"
test -s "$FULL_VAL_OUT/index.html"

conda run -n drivestudio-new --no-capture-output \
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps(p,indent=2,ensure_ascii=False)); assert p["status"] == "passed"' \
  "$FULL_VAL_OUT/validation_contract.json"
```

`validation_contract.json` 必须确认：

- 计划数与完成数一致，所有 plan/trace/summary/HTML非空；
- 所有数值finite，无runtime exception；
- 至少存在一次 K≥2 update且 `model_update_count > 0`；
- 所有 Stage 3.4 eval event均有 `grad_active=0`、`forward_only=1`；
- model parameter version在validation前后完全一致；
- causal LocalGS/GDKV state按plan推进，不因 no-grad 被错误冻结；
- 配置启用的 assimilation timeline、repair before/after、order robustness 完成；
- 全部请求的memory modes完成；
- checkpoint的version、variant和codec schema与配置一致；
- 不存在 incremental Parent runtime、drift、refresh、runtime update或 surrogate VJP键。

Validation中 `model_update_count > 0` 与 `grad_active=0` 必须同时成立；这正是 v57
no-grad K≥2修复的关键验收条件。

---

## 6. K=15 bounded profile

先用 dry-run检查命令、固定数据和阈值，不触发数据/CUDA工作：

```bash
conda run -n drivestudio-new --no-capture-output \
  env PYTHONPATH=$ROOT python $ROOT/tools/profile_iforward_stage3_4_k15.py \
  --scene-id 131 --segment-id 1 --seed 41 --samples 3 --dry-run
```

正式 profile：

```bash
conda run -n drivestudio-new --no-capture-output \
  env PYTHONPATH=$ROOT python $ROOT/tools/profile_iforward_stage3_4_k15.py \
  --baseline-config $ROOT/configs/iforward/iforward_stage3_3_observation_feedback.yaml \
  --candidate-config "$CFG" \
  --scene-id 131 --segment-id 1 --seed 41 --samples 3 \
  --max-peak-ratio 1.15 \
  --max-time-ratio 1.20 \
  --max-retained-growth-mb 64 \
  --output-json "$PROFILE_OUT"
```

Profiler不篡改 source或Functional Parent alpha schedule，也不放宽生产配置的fail-fast。
它保留正式五点调度，并把 trainer logical global step设为 `15000 + local_offset`，使两条
调度自然处于 alpha=1。每个 rollout的 first visit仍因 update-ancestor gate保持
forward-only，visits 2..K使用完整 Jacobian。

命令在下列任一条件不满足时返回非零：

- baseline/candidate均为预期版本且 scheduler metadata signature一致；
- warmup为prelude，三个 measured rollout均为 repair-tail B5R3/K=15；
- 无 OOM/non-finite/skipped optimizer step；
- exactly one first visit forward-only，visits 2..K有 update ancestor；
- candidate三分支按present状态记录 configured/attached和project/lift/support/clamp指标；
- lifting/PTV3/relation boundary assertions通过；
- candidate无 legacy runtime/VJP/drift/refresh键；
- candidate peak allocated CUDA memory不超过 baseline `1.15x`；
- candidate median synchronized step time不超过 baseline `1.20x`；
- 非terminal rollout的 post-cleanup allocation spread不超过64 MiB。

检查输出：

```bash
test -s "$PROFILE_OUT"

conda run -n drivestudio-new --no-capture-output \
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps(p,indent=2)); assert p["comparison"]["accepted"]' \
  "$PROFILE_OUT"
```

只使用 synchronized top-level step time做性能门禁；per-branch `project_ms`仅作presence诊断。
显存门禁使用 PyTorch peak allocated memory，不使用driver reserved memory。

---

## 7. Deferred matched 1000-step B/C（本轮不实现、不执行）

### 7.1 实验合同

B/C都从同一个 native Stage 3.3 checkpoint做 weights-only初始化，并使用相同的 v57
forward合同、数据顺序、seed、source feedback、scheduler、codec和optimizer设置。B通过独立
ablation identity表达 alpha=0，唯一方法变量仍是 Functional Parent geometry Jacobian：

| Run | Functional Parent alpha | 含义 |
|---|---|---|
| B | 始终0 | exact Functional Parent forward存在，8D adapter仍训练，但不回传到Parent/LocalGS |
| C | 正式五点调度 | forward相同，按正式调度打开Parent/LocalGS geometry Jacobian |

不能用 Stage 3.3 source-only代替 B，也不能使用旧随机13D Stage 3.4 checkpoint。否则无法把
差异归因于 Functional Parent Jacobian。

### 7.2 前置条件

生产 `stage3_4_functional_parentgs_lift` 会严格拒绝非五点 Functional Parent schedule。
因此不能通过普通 OmegaConf/CLI override把生产 variant改成 alpha=0。

正式执行 B/C 前，必须另行新增一个受控 alpha=0 ablation：

- 独立 version、training variant和完整配置；
- 与 v57使用相同 exact Functional Parent forward和
  `legacy17d_plus_geometry8d_residual_v1` codec；
- 唯一方法差异是 Functional Parent geometry Jacobian恒为0；
- manifest/checkpoint明确标识 ablation，不能伪装成生产 v57；
- 有单测证明 B/C forward逐 tensor一致且 B 的Parent/LocalGS Jacobian为0。

该 ablation入口属于后续实验工作，本轮不新增。以下命令是待 `$B_CFG` 落地后的运行模板，
现在不能执行。

### 7.3 共享设置

```bash
export AB_ROOT=/root/autodl-tmp/outputs/iforward_stage3_4_v57_fixed_segment_bc
export B_CFG=/absolute/path/to/future_stage3_4_functional_parent_alpha0_ablation.yaml

COMMON_OVERRIDES=(
  initialization.skip_keys=[]
  logging.metrics_history_append=false
  logging.train_step_metrics_interval=1
  logging.scheduler_metrics_interval=1
  logging.performance.enable=true
  logging.performance.phase_timing=true
  logging.performance.cuda_memory=true
  training.save_checkpoint_freq=1000
  training.seed=41
  scheduler_stage3_0.traversal.fixed_scene_id=131
  scheduler_stage3_0.traversal.fixed_segment_id=1
  scheduler_stage3_0.traversal.seed=41
  scheduler_stage3_0.producer.enable=false
  'data.train_scene_ids=[131]'
  'data.eval_scene_ids=[]'
  data.pixel_source.require_egocar_mask_template=false
  dataset.preload_scene_count=1
  scheduler_stage3_0_validation.enable=false
  iforward_validation_v4.enable=false
  eval.run_test_at_end=false
  model.iforward.observation_feedback.schedule.activation_step=0
  model.iforward.observation_feedback.debug.grad_probe_interval=1
  model.iforward.observation_feedback.parent_projection.enable=false
  model.iforward.observation_feedback.relation.enable=false
)
```

### 7.4 B：未来 alpha=0 exact-forward control

```bash
conda run -n drivestudio-new --no-capture-output \
  env PYTHONPATH=$ROOT python $ROOT/tools/train_iforward_one_segment.py \
  --config_file "$B_CFG" --max_steps 1000 --seed 41 \
  --init_checkpoint "$INIT_CKPT" --init_weights_only \
  "${COMMON_OVERRIDES[@]}" \
  output_name=stage3_4_v57_parent_alpha0 \
  logging.project=stage3_4_v57_parent_alpha0 \
  logging.log_dir="$AB_ROOT/parent_alpha0"
```

### 7.5 C：正式五点调度

```bash
conda run -n drivestudio-new --no-capture-output \
  env PYTHONPATH=$ROOT python $ROOT/tools/train_iforward_one_segment.py \
  --config_file "$CFG" --max_steps 1000 --seed 41 \
  --init_checkpoint "$INIT_CKPT" --init_weights_only \
  "${COMMON_OVERRIDES[@]}" \
  output_name=stage3_4_v57_parent_alpha_production \
  logging.project=stage3_4_v57_parent_alpha_production \
  logging.log_dir="$AB_ROOT/parent_alpha_production"
```

> 这两条命令是延期实验模板，不在本轮运行。在 `$B_CFG` 及其独立身份、测试和manifest
> 落地前，B/C均不得启动；不得用普通 OmegaConf override篡改生产
> `functional_parent.alpha_schedule`。

完成 B/C 后，应在相同 post-warmup step范围比较 finite/isolation/runtime-key/memory/time，
并检查 current/history、repair retention、order robustness和 earlier-delta gradient。只有
`K>=3` rollout才要求distance-2 probe。1000 steps的质量指标是探索性结果，不替代长程实验。

---

## 8. 结果归档

所有P0门禁通过后，归档 smoke日志、native v57 checkpoint、quick/full Validation v4报告
和K=15 JSON；不包含原始数据集或旧checkpoint：

```bash
export ACCEPTANCE_ARCHIVE=/root/autodl-tmp/outputs/iforward_stage3_4_v57_acceptance_20260716.tar.zst

tar --zstd -cf "$ACCEPTANCE_ARCHIVE" \
  -C /root/autodl-tmp/outputs \
  "$(basename "$SMOKE_OUT")" \
  "$(basename "$QUICK_VAL_OUT")" \
  "$(basename "$FULL_VAL_OUT")" \
  iforward_stage3_4_v57_k15

test -s "$ACCEPTANCE_ARCHIVE"
sha256sum "$ACCEPTANCE_ARCHIVE" > "$ACCEPTANCE_ARCHIVE.sha256"
```

如果任一门禁失败，仍保留对应输出用于诊断，但不得把归档标记为 accepted。
