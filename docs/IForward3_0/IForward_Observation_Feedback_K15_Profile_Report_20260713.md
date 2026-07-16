# IForward 观测反馈闭环 CUDA K=15 实测报告

日期：2026-07-13  
设备：NVIDIA GeForce RTX 4090  
软件：PyTorch 2.0.0+cu118  
配置：`configs/iforward/iforward_stage3_3_observation_feedback.yaml`

## 1. 方法

使用生产 `MultiSceneDatasetV4 → Stage23Scheduler → IForwardTrainer.train_step` 链路，固定：

- scene 131、segment 1、seed 41；
- 第一个 rollout 为 repeat-refine K=2 prelude，用来建立相同 episode carry；
- 第二个 rollout 强制为 repair B5R3，即 K=15；
- prelude 的 source/parent/relation alpha 均为 0；
- 测量 rollout 的 source alpha=1.0，parent/relation alpha=0.3；
- repair 的 frontend 参数冻结；
- 每档都执行完整 forward、scaled backward、unscale/clip、optimizer step 和 rollout-boundary detach。

可复现工具：

```bash
conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding \
  python tools/profile_iforward_observation_feedback_k15.py \
  --variants baseline_frozen_no_grad,render_eager,render_checkpoint,render_parent_vjp,full_relation \
  --output-json /tmp/iforward_k15_all.json
```

## 2. 结果

| Variant | Peak allocated | 相对 baseline | Step time | 时间增量 | Source grad norm | Parent VJP reports | Finite | Optimizer skip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen-no-grad baseline | 19,596 MB | 1.000× | 2,408 ms | 0.0% | 0 | 0 | 是 | 0 |
| render eager | 44,716 MB | 2.282× | 2,655 ms | +10.3% | 2.054e-4 | 0 | 是 | 0 |
| render checkpoint | 21,401 MB | 1.092× | 2,798 ms | +16.2% | 2.043e-4 | 0 | 是 | 0 |
| render + parent VJP | 21,809 MB | 1.113× | 2,994 ms | +24.4% | 2.057e-4 | 14 | 是 | 0 |
| full relation | 22,090 MB | 1.127× | 3,052 ms | +26.8% | 2.054e-4 | 14 | 是 | 0 |

所有 frozen-input-grad 档的 2D frontend 参数梯度计数均为 0；source LocalGS 梯度均非零。完整 relation 档的 runtime 指标确认 relation feedback 已启用。

## 3. 验收判断

- K=15：通过，实测 B=5、R=3、K=15。
- NaN/Inf：通过，五档均 finite。
- optimizer skip：通过，五档均为 0。
- checkpoint 显存不超过 frozen-no-grad baseline 1.15×：通过。
  - source checkpoint：1.092×；
  - source + parent VJP：1.113×；
  - full relation：1.127×。
- frozen frontend 参数无梯度、LocalGS source 梯度非零：通过。
- eager 显存：2.282×，不满足长 rollout 预算；它仅作为对照，证明 full-dynamic checkpoint 是必要条件。

## 4. 说明

Stage3.3 默认分布的自然采样最大为 K=12；`max_inner_k_hard_cap=15` 是硬上限，不保证采到 15。本报告使用 profiler 局部覆盖为 B5R3，未改变正式 Stage3.3 配置的采样分布。

Parent VJP report 数为 14，而不是 15，是因为首个 visit 建立 exact runtime，后续 14 个 visit 才使用 incremental-runtime surrogate VJP；这符合当前 runtime 生命周期。

本次是固定 batch 的 smoke/profile，不替代长期训练或消融实验。
