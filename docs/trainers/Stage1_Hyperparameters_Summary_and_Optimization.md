# Stage 1 超参总结与优化建议

本文档：1）汇总 Stage 1 实验涉及的**全部超参**；2）分析**是否有必要调整或需设计实验优化**的超参。

---

## 1. Stage 1 实验涉及的所有超参

### 1.1 配置文件中显式出现的超参（configs/minimal_streetforward_stage1.yaml）

| 类别 | 超参 | 当前值 | 说明 |
|------|------|--------|------|
| **输出** | `output_name` | `minimal_sf_stage1` | 日志/checkpoint 目录名 |
| **dataset** | `segment_aabb` | `[[-40,-20,-20],[40,4.8,70]]` | 场景 AABB（点云过滤与体素范围） |
| **dataset** | `segment_input_aabb` | 同上 | 与 full trainer 一致，Stage 1 未用 |
| **model** | `offset_max` | 0.1 | 位置偏移 tanh 后缩放上限（米） |
| **model** | `scale_max` | 0.1 | 尺度对数偏移上限 |
| **model** | `omega_max` | 0.1 | 轴角偏移上限（弧度） |
| **model** | `opacity_max` | 0.1 | 不透明度 logit 偏移上限 |
| **model** | `sh_dc_max` | 0.1 | SH DC 偏移上限 |
| **model** | `sh_rest_max` | 0.05 | SH rest 偏移上限 |
| **model** | `eta_means` | 1.0 | 位置步长因子 |
| **model** | `eta_scales` | 1.0 | 尺度步长因子 |
| **model** | `eta_opacity` | 1.0 | 不透明度步长因子 |
| **model** | `eta_sh_dc` | 1.0 | SH DC 步长因子 |
| **model** | `eta_sh_rest` | 1.0 | SH rest 步长因子 |
| **model** | `sh_degree` | 1 | 球谐阶数 |
| **model** | `voxel_size` | 0.2 | 体素大小（米），影响 3D 体积分辨率 |
| **model** | `sparseConv_outdim` | 32 | 稀疏卷积输出维度（= Head 输入维） |
| **model** | `update_node_state_interval` | 10 | 每 N 步写回 NodeState；0=不写回 |
| **optimizer** | `lr` | 1e-3 | 学习率 |
| **optimizer** | `eps` | 1e-15 | Adam eps |
| **optimizer** | `weight_decay` | 0.0 | 权重衰减 |
| **training** | `max_iterations` | 1000 | 训练步数 |
| **training** | `log_interval` | 50 | 打印 loss 间隔 |
| **training** | `save_checkpoint_freq` | 500 | 保存 checkpoint 间隔 |
| **training** | `seed` | 42 | 随机种子 |
| **eval** | `enable_psnr` | true | 是否算 PSNR |
| **eval** | `metric_interval` | 10 | 计算 PSNR 的 step 间隔 |
| **eval** | `heavy_metric_interval` | 50 | 计算 SSIM/LPIPS 的 step 间隔 |
| **eval** | `run_test_at_end` | true | 结束时是否跑 test views |
| **logging** | `image_interval` | 50 | 保存 pred/gt/error 图像的间隔 |
| **logging** | `enable_jsonl_metrics` | true | 是否写 metrics_history.jsonl |
| **logging** | `use_tensorboard` | true | 是否用 TensorBoard |

### 1.2 脚本默认值（未在 Stage 1 配置中写出，但会参与 run）

| 来源 | 超参 | 默认值 | 说明 |
|------|------|--------|------|
| train_minimal_streetforward_stage1.py | `--output_root` | `outputs` | 输出根目录 |
| train_minimal_streetforward_stage1.py | `--project` | `minimal_sf` | 二级目录名 |
| train_minimal_streetforward_stage1.py | `--run_name` | `overfit` | 被 config 的 `output_name` 覆盖时不用 |
| train_minimal_streetforward_stage1.py | `--seed` | 42 | 可与 config 的 training.seed 一致或覆盖 |
| train_minimal_streetforward_stage1.py | setup() 中 eval/logging 默认 | 见脚本 | 配置缺失时的补全 |

### 1.3 模型内部写死的超参（未暴露到 config）

| 位置 | 超参 | 值 | 说明 |
|------|------|-----|------|
| minimal_trainer_stage1.py | k-NN 的 k | 3 | `_pairwise_neighbor_distances(means, k=3)`，初始 scales_log |
| minimal_trainer_stage1.py | scales 最小距离 clamp | 1e-3 | 避免 log(0) |
| minimal_trainer_stage1.py | 初始 opacity 对应概率 | 0.1 | `torch.logit(torch.full(..., 0.1, ...))` |
| minimal_trainer_stage1.py | MLP 隐藏层维度 | 64, 32 | 所有 offset 头均为 outdim→64→32→输出 |
| minimal_trainer_stage1.py | 颜色 [0,255] 判断阈值 | 1.0+1e-3 | 点云颜色归一化判断 |

---

## 2. 是否有必要调整的超参？是否需要设计实验优化？

### 2.1 建议「可先不动」的超参（当前实验已证明有效）

- **offset_*_max / eta_***：当前 overfit 收敛良好（loss 从 ~0.18 降到 ~0.04，train PSNR ~23），说明偏移范围与步长因子在 Stage 1 单 target 设定下是合理的；与 StreetForward_Flow 一致，无需为 Stage 1 单独大改。
- **sh_degree**：1 与常见 3DGS 一致，Stage 1 无证据需要提高。
- **optimizer (lr, eps, weight_decay)**：Adam 1e-3 无 weight_decay 已稳定收敛，可维持。
- **seed**：复现用，不参与“优化”。
- **log_interval / metric_interval / image_interval / save_checkpoint_freq**：仅影响日志与存储，不改变训练效果。

### 2.2 值得关注或可做小范围对比的超参

| 超参 | 当前值 | 建议 | 理由 |
|------|--------|------|------|
| **update_node_state_interval** | 10 | 做 0 / 1 / 10 / 50 小网格对比 | 目前只验证了 10；0（不写回）vs 1（每步写回）可验证「写回必要性」与收敛速度/稳定性；10 vs 50 可看对收敛曲线和最终 PSNR 的影响。 |
| **voxel_size** | 0.2 | 可选：0.1 vs 0.2 对比 | 影响体素分辨率和显存；更小分辨率可能带来更细特征，但计算量上升；可在同一 batch 上跑短 run（如 200 step）看 loss 曲线差异。 |
| **sparseConv_outdim** | 32 | 可选：32 vs 64 对比 | 决定 3D 特征与 Head 容量；若 32 已能 overfit，不一定需要调；若后续 Stage 2/3 显存允许，可试 64 看是否更快收敛或更高上限。 |
| **max_iterations** | 1000 | 视目标定 | 当前 1000 步已明显 overfit；若要做「收敛到多少」的对比，可固定 500/1000/2000 做横向比较。 |

### 2.3 建议设计实验优化的超参（若有精力）

1. **update_node_state_interval**
   - **实验**：同一 config、同一 batch、固定 max_iterations（如 1000），只改 `update_node_state_interval`：0, 1, 5, 10, 20, 50。
   - **指标**：train loss_l1 / PSNR 曲线、最终 step 的 train PSNR、run 时间（若每步写回更贵可略记）。
   - **目的**：确认写回是否有收益、最佳间隔量级，并为 Stage 2 选默认间隔提供依据。

2. **voxel_size（可选）**
   - **实验**：voxel_size = 0.1 vs 0.2，其余一致，短 run（如 200 step）比较 loss 与 train PSNR。
   - **目的**：看更细体素是否在 Stage 1 单 target 上带来明显收益，以及显存/速度是否可接受。

3. **lr（可选）**
   - **实验**：lr = 5e-4, 1e-3, 2e-3，固定其他，看收敛速度与最终 train PSNR 是否敏感。
   - **目的**：确认 1e-3 是否已接近合理区间，避免后续 Stage 2/3 盲目调 lr。

### 2.4 暂不建议改动的部分

- **segment_aabb**：由数据与场景定义，不属于“调参”范畴；换数据会换 aabb。
- **offset_*_max / eta_***：与完整 StreetForward 对齐，且 Stage 1 已收敛良好，除非要做「更大偏移/更小步长」的消融，否则可保持。
- **MLP 结构（64→32）与 k=3**：当前未暴露为 config；若未来要系统优化，再考虑暴露 k 与 hidden_dim 并做小网格搜索。

---

## 3. 小结

| 问题 | 结论 |
|------|------|
| **Stage 1 涉及的所有超参** | 见第 1 节：配置中 30+ 项、脚本默认若干项、模型内写死 5 类（k=3、clamp 1e-3、初始 opacity 0.1、MLP 64/32、颜色阈值）。 |
| **是否有必要调整** | **有**：`update_node_state_interval` 建议做 0/1/10/50 等对比；**可选**：`voxel_size`、`sparseConv_outdim`、`lr`、`max_iterations` 视目标做小范围对比。 |
| **是否需要设计实验优化** | **建议至少做**：`update_node_state_interval` 的间隔对比实验；其余为可选，用于为 Stage 2/3 选默认值与理解敏感性。 |

当前 Stage 1 已证明「NodeState + offset + 按间隔写回」链路有效；通过上述少量对比实验，可以更清楚哪些超参对收敛与稳定性敏感，并为后续多 target、2D 特征、动态物体等阶段提供更稳妥的默认配置。
