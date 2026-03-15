# Stage 1 vs Stage 1.1 结果比较与后续方向

本文档在**相同数据、相同训练配置（步数/学习率/写回间隔等）**下，对比 Minimal StreetForward **Stage 1**（NodeState + 直接 offset 头）与 **Stage 1.1**（NodeState + GRU-style 偏移量预测 + h_cache），并参考常见深度学习实践讨论结论与后续该做的事。

- **Stage 1 输出**：`outputs/minimal_sf/minimal_sf_stage1`
- **Stage 1.1 输出**：`outputs/minimal_sf/minimal_sf_stage1_1`
- **对照实验原则**：同一 overfit batch、同一 seed=42、同一 1000 steps、同一 `update_node_state_interval=10`，仅模型是否含 GRU 不同。

---

## 1. 实验设置对照

| 项目 | Stage 1 | Stage 1.1 |
|------|---------|-----------|
| **模型** | NodeState + `feat_3d` → MLP → offsets | NodeState + `feat_3d` + param_embed → GRU → MLP → offsets |
| **额外模块** | 无 | param_embed(17→32)、GRU(update/candidate/reset)、gru_to_head、h_cache_bg |
| **参数量** | 较少 | 较多（param_embed + GRU + 原 head） |
| **数据 / 步数 / 写回** | 同 batch、1000 step、interval=10 | 同左 |
| **优化器** | Adam lr=1e-3 | 同左 |

二者**训练集、验证方式、随机种子、迭代次数**一致，满足对照实验的可比性要求。

---

## 2. 结果对比

### 2.1 最终指标（metrics_final.json）

| 指标 | Stage 1 | Stage 1.1 | 差异简要 |
|------|---------|-----------|----------|
| **Train loss_l1** | 0.0374 | 0.0410 | Stage 1 更低 |
| **Test PSNR** | 10.81 | 10.39 | Stage 1 略高 |
| **Test SSIM** | 0.236 | 0.231 | Stage 1 略高 |
| **Test LPIPS** | 0.840 | 0.845 | Stage 1 略优（越低越好） |

结论：在当前「单 target、1000 step、同一 lr」设定下，**Stage 1 在 train loss 与 test 指标上均略优于 Stage 1.1**；差异不大，但方向一致。

### 2.2 训练曲线（metrics_history.jsonl 抽样）

| Step | Stage 1 loss_l1 | Stage 1.1 loss_l1 | Stage 1 PSNR | Stage 1.1 PSNR |
|------|-----------------|-------------------|--------------|----------------|
| 0    | 0.176           | 0.176             | 13.19        | 13.19          |
| 100  | 0.112           | 0.098             | 15.97        | 17.65          |
| 200  | 0.065           | 0.076             | 20.21        | 19.18          |
| 500  | 0.044           | 0.052             | 22.29        | 21.53          |
| 800  | 0.039           | 0.044             | 22.94        | 22.46          |
| 990  | 0.037           | 0.041             | 23.24        | 22.86          |

- **初始点一致**（step 0 相同）：说明数据、初始化、单步前向在两种模型下对齐良好。
- **前 100 步**：Stage 1.1 一度 loss 更低、PSNR 更高（GRU 多参数，前期拟合快）。
- **中后期**：Stage 1 反超并保持更低的 loss、更高的 PSNR，最终收敛更好。

这符合常见现象：**更复杂模型在相同 step/lr 下未必更好**，往往需要更长训练或不同学习率/正则。

---

## 3. 从深度学习实践角度的解读

### 3.1 对照实验有效性

- **控制变量**：仅「是否使用 GRU + param_embed + h_cache」不同，其余（数据、步数、写回、seed）一致，便于归因。
- **多指标**：同时看 train loss、train PSNR、test PSNR/SSIM/LPIPS，避免单指标偶然性。
- **曲线而非单点**：看整条曲线可知 Stage 1.1 前期略优、后期被 Stage 1 超过，说明不是单纯“GRU 更差”，而是**当前训练配置更有利于简单模型**。

### 3.2 可能原因简述

1. **优化难度**  
   GRU + param_embed 增加参数与非线性和时序（h 的更新），在相同 lr、相同步数下可能尚未充分收敛；Stage 1 结构简单，更容易在 1000 step 内逼近该单视角的局部最优。

2. **学习率与步数**  
   常见做法是：**大模型 / 更复杂模块往往需要更长训练或略低 lr**。当前对二者使用完全相同的 lr 与 max_iterations，没有针对 Stage 1.1 做任何调参。

3. **h 的 detach**  
   Stage 1.1 每步对 `h_old` detach，梯度不跨 step 回传。这样设计与完整 StreetForward 一致，但会限制「跨步」的梯度流，在单 target、单步 loss 的设定下，GRU 的时序表达能力未必能发挥，反而多了一层非线性和优化负担。

4. **单 target 设定**  
   当前只有 1 个 target view、每步同一视角。GRU 的设计初衷之一是融合多步/多视角信息；在严格单视角 overfit 下，其相对 Stage 1 的边际收益可能很小，而多出来的容量需要更多数据或更长训练才能体现。

### 3.3 小结（最佳实践视角）

- **结论**：在**当前**配置下，Stage 1 略优于 Stage 1.1，但二者都能稳定 overfit、loss 下降正常，说明 **Stage 1.1 的 GRU 实现没有破坏训练稳定性**，只是尚未在「相同步数、相同 lr」下展现出优势。
- **不建议**仅凭此实验就放弃 GRU：更合理的做法是**先做针对性调参与更长训练**，再与 Stage 1 比较；若后续引入多 target（Stage 2），再评估 GRU 是否带来增益。

---

## 4. 后续该干啥（建议优先级）

### 4.1 若目标是把 Stage 1.1 调得至少不逊于 Stage 1

- **延长训练**：例如 2000–3000 step，观察 Stage 1.1 的 loss/PSNR 是否继续改善并反超 Stage 1。
- **学习率**：对 Stage 1.1 尝试略低 lr（如 5e-4）或使用 warmup/余弦退火，看最终 train/test 是否更稳、更好。
- **GRU 相关超参**：如 `param_embed_dim`、`offset_gru_hidden_dim`、`offset_gru_use_reset_gate`，做小规模网格或手调，记录与 Stage 1 的对比（同 step、同数据）。
- **写回间隔**：可尝试 `update_node_state_interval=1` 或 5，看对两种模型的影响是否不同（尤其是对 h 与 NodeState 的协同）。

### 4.2 若目标是与设计计划对齐、为多视角做准备

- **Stage 2（多 target）**：同一 batch 内多张 target、代理参数与多视角梯度累积（参考 [StreetForward_Flow](StreetForward_Flow.md)、[Minimal_StreetForward_Design_Plan](Minimal_StreetForward_Design_Plan.md)）。  
  - 在多 target 设定下**再比一次 Stage 1 vs Stage 1.1**，看 GRU 是否在「多视角/多步」下更有优势。
- **评估方式**：除最终 test PSNR/SSIM/LPIPS 外，可加「每 N 步在固定若干 view 上算 PSNR」，画成曲线，便于比较收敛速度与稳定性。

### 4.3 若目标是把 minimal 当调试/回归基线

- **固定实验协议**：例如「同一 batch、seed=42、1000 step、同一 eval 脚本」，把 Stage 1 与 Stage 1.1 的 metrics_final.json 和关键 step 的 metrics 纳入 CI 或回归文档，防止后续改动导致某一阶段明显变差。
- **文档化**：将本次对比的表格与曲线（或链接到 `metrics_history.jsonl`）写进 [Minimal_StreetForward_Design_Plan](Minimal_StreetForward_Design_Plan.md) 或本目录下的实验小结，便于以后复现与对比。

### 4.4 不建议立刻做的事

- **不要**仅因「Stage 1.1 最终略逊于 Stage 1」就删除 GRU 或回退到仅 Stage 1：当前实验未对 Stage 1.1 做任何调参，且多 target/多步场景尚未测。
- **不要**在未固定 seed 与数据的情况下做「零散改代码再跑一次」的对比：否则无法区分是模型结构还是随机性/数据导致的差异。

---

## 5. 简要结论表

| 问题 | 结论 |
|------|------|
| Stage 1 vs 1.1 谁更好？ | 在当前 1000 step、同一 lr 下，**Stage 1 略好**（train loss 与 test PSNR/SSIM/LPIPS 均略优）。 |
| Stage 1.1 是否训坏了？ | **没有**。二者从同一初始 loss 起步，Stage 1.1 收敛正常，仅最终略逊。 |
| 是否还值得保留 GRU？ | **值得**。需在「更长 step / 调 lr / 多 target」下再评估，再决定是否沿用或简化。 |
| 下一步优先做啥？ | （1）对 Stage 1.1 做**更长训练 + 学习率/写回间隔**小调参；（2）实现 **Stage 2 多 target**，再比 Stage 1 vs 1.1；（3）**固定实验协议**，把本次结果当回归基线。 |

---

## 6. 参考

- [Minimal_StreetForward_Design_Plan](Minimal_StreetForward_Design_Plan.md) — 阶段划分与验证目标  
- [Minimal_StreetForward_Next_Steps_Stage1_1_GRU](Minimal_StreetForward_Next_Steps_Stage1_1_GRU.md) — Stage 1.1 GRU 设计说明  
- [StreetForward_Flow](StreetForward_Flow.md) — 完整流程与 GRU/多 target 说明  
- Stage 1 实验结果分析：`outputs/minimal_sf/minimal_sf_stage1/实验结果分析_StreetForward功能验证.md`  
- 本次对比数据来源：`outputs/minimal_sf/minimal_sf_stage1/metrics_final.json`、`minimal_sf_stage1_1/metrics_final.json` 及二者 `metrics_history.jsonl`
