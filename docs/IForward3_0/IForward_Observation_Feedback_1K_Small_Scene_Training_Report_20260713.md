# IForward Stage3.3 单场景 1000-step 训练报告

日期：2026-07-13

## 结论

scene 131 / segment 1 的真实生产训练已完成 step 0--999，正常退出并保存最终 checkpoint。未发现 NaN/Inf、OOM、checkpoint 重算错误、producer worker 错误或状态缓存因 optimizer skip 被丢弃。source-render feedback 在非零 alpha 下有明确的 LocalGS 输入梯度，repair 的冻结 2D frontend 行为正确。

本次发现的非阻断项：

1. CUDA caching allocator 的 reserved 高水位达到 18.03 GiB；最终活跃 allocated 仅 1.49 GiB，高水位在中后程保持平台，未呈现历史图泄漏特征。
2. 每 10 step 的 100 个详细梯度样本中有 3 次触发 `max_norm=1.5` 裁剪，未裁剪范数为 1.93、2.20、1.98；均为有限值且 optimizer 正常更新。
3. 数据加载器当时报告 cam 0/1/2 缺少 nuScenes egocar mask 模板。后续核实这三个相机不存在 ego 遮挡，因此缺少模板是预期数据语义；Stage3.3 现已将它们显式声明为无 occlusion，并物化为全零 mask。

## 运行范围

| 项目 | 值 |
|---|---:|
| 配置 | `iforward_stage3_3_observation_feedback.yaml` |
| scene / segment | 131 / 1 |
| step | 0--999，共 1000 step |
| source feedback alpha | 0.0 -> 0.0999 |
| parent / relation feedback | 默认关闭，未覆盖 |
| Gaussian 数量 | bg 300000；distant 41549；rigid 18998 |
| 总训练墙钟时间 | 约 21 分钟 |

使用专用单 segment 入口 `tools/train_iforward_one_segment.py`，其余 dataset、scheduler、trainer、optimizer、feedback 和 checkpoint 路径与生产 IForward 入口一致。

## 数值与收敛

| 指标 | 结果 |
|---|---:|
| 1000-step loss 均值 / 中位数 | 0.13524 / 0.12314 |
| loss 最小 / 最大 | 0.06768 / 0.30735 |
| 前 100 step loss 均值 | 0.21459 |
| 后 100 step loss 均值 | 0.10286 |
| 前后 100-step 均值下降 | 52.1% |
| 最终 step loss | 0.10583 |

分段 loss 均值持续下降：

| step 窗口 | loss 均值 |
|---|---:|
| 0--99 | 0.21459 |
| 100--199 | 0.16708 |
| 200--399 | 0.14375 |
| 400--599 | 0.12272 |
| 600--799 | 0.11379 |
| 800--999 | 0.10512 |

## Feedback 与 scheduler 核验

step 500、source alpha=0.05 的梯度 probe：

| probe | 梯度范数 |
|---|---:|
| source render 总输入 | 3.538e-4 |
| means | 1.940e-4 |
| scales | 2.701e-4 |
| quats | 5.633e-5 |
| opacities | 9.301e-5 |
| colors | 5.272e-5 |

这验证了真实训练中的 `earlier LocalGS -> later source render/CNN -> later loss` 梯度链。step 0、alpha=0 时对应 probe 为 0，符合 Jacobian 缩放语义。

记录到的 351 个 scheduler 详细行覆盖：

- distribution：shuffled coverage 158、repeat refine 120、high-block repair 73。
- K：2、3、4、6、8、10、12；本次 warmup 未调度到硬上限 K=15。
- mode：trainable checkpointed 278、frozen input-grad checkpointed 73。

每 10 step 的详细训练行中，5 个 frozen repair 样本的 2D 参数梯度计数和范数均为 0；95 个 trainable 样本的 2D 参数梯度均非零，说明冻结和恢复行为正确。

## 稳定性与性能

| 指标 | 结果 |
|---|---:|
| optimizer skip（100 个详细样本） | 0 |
| nonfinite gradient（100 个详细样本） | 0 |
| producer worker error | 0 |
| state cache drop on skip | 0 |
| 平均 / P50 / P95 step time | 1239.8 / 1284.8 / 2061.0 ms |
| 平均 forward / backward / optimizer | 626.7 / 567.5 / 3.6 ms |
| 最大 active CUDA memory | 17.79 GiB |
| 最大 reserved CUDA memory | 18.03 GiB |
| 训练结束 active / reserved | 1.49 / 18.03 GiB |

前 100 step 平均 1216.5 ms，后 100 step 平均 1260.8 ms。差异与后段调度到更多高 K/多 view rollout 一致；reserved 显存在约 step 500 后保持平台，没有持续增长证据。

## 产物

- 输出目录：`/root/autodl-tmp/outputs/iforward_stage3_3_observation_feedback_smoke1k_scene131_seg1_20260713`
- 最终 checkpoint：`checkpoints/iforward_stage3_1_lowrank_gated_delta_kv_lift_final.pt`
- checkpoint 校验：step 999、739 个 model state tensors、7 个 optimizer parameter groups，并包含 RNG、scheduler 和 train-loop 状态。
- 指标：`metrics_history.jsonl`、`metrics_final.json`
- 完整日志：`logs/log_2026-07-13_17-02-04.txt`、`console.log`

## 适用边界

这是固定单场景/单 segment 的 1000-step 稳定性与过拟合 smoke，不等价于多场景泛化实验。Stage3.3 默认关闭的 parent VJP 和 relation feedback、本配置 source alpha 的 0.25--1.0 区间、以及真实 K=15 长时训练未被本次运行覆盖；这些仍应由既有 K=15 profile 和后续定向训练分别验证。
