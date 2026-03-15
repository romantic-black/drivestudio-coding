# Minimal StreetForward 下一步方案：Stage 2 多 Target（2.0 无代理 / 2.1 代理参数）

本文档讨论 **Minimal StreetForward Stage 2** 的实现方案：在 **Stage 1.1 模型**（NodeStateBackground + GRU-style 偏移量 + 单 target）基础上，先支持**多 target**（如 3 个），再引入**代理参数**与多视角梯度累积，并与无代理版本对比以保证一致性。设计参考 [Minimal_StreetForward_Design_Plan](Minimal_StreetForward_Design_Plan.md) §3（Stage 2）、[StreetForward_Flow](StreetForward_Flow.md) 的代理渲染流程，以及 [minimal_trainer_stage1_1.py](../../models/streetforward/minimal_trainer_stage1_1.py) 与 [train_minimal_streetforward_stage1_1.py](../../tools/train_minimal_streetforward_stage1_1.py)。

---

## 1. 方案总览

### 1.1 基础模型

- **一律在 Stage 1.1 上扩展**：使用 [MinimalStreetForwardStage1_1](Minimal_StreetForward_Next_Steps_Stage1_1_GRU.md)（NodeStateBackground、3D 特征、GRU-style offsets、h_cache_bg、单 target 渲染）。
- 不基于 Stage 0（minimal_trainer.py），以保证 NodeState 与 GRU 行为一致，便于后续接 Stage 3（source + 2D）等。

### 1.2 分阶段实现

| 阶段 | 内容 | 目的 |
|------|------|------|
| **Stage 2.0** | 仅多 target（如 3 个），**不使用**代理参数 | 同一套 `render_params` 对多 view 渲染，loss 累加或取均再单次 `backward`，作为正确基线。 |
| **Stage 2.1** | 多 target + **使用**代理参数 + 多视角梯度累积 | 与 StreetForward_Flow 一致的多 view 梯度累积路径；与 2.0 在相同数据与 num_targets 下对比，保证一致性。 |

实现顺序：**先 2.0，再 2.1**；2.1 完成后用相同 overfit batch 与相同 num_targets（如 3）对比 2.0 与 2.1 的 loss/PSNR，验证代理路径等价或可接受差异。

---

## 2. Stage 2.0：多 Target，无代理参数

### 2.1 目标

- **输入**：与 Stage 1.1 相同（点云 + targets）；`targets` 为**多个**（如 3 个），每项 `{ "frame_idx", "view", "gt_image" }`。
- **前向**：与 Stage 1.1 完全相同的链路算**一套** `render_params`（NodeState → 3D 特征 → GRU → offsets → render_params）；用**同一套** `render_params` 对每个 target 渲染，得到 `pred_rgb_i`。
- **损失**：例如 `loss = (1/N) * sum_i L(pred_rgb_i, gt_image_i)`（L1），或 `loss = sum_i L(pred_rgb_i, gt_image_i)` 再 `loss.backward()`。
- **反向**：**单次** `loss.backward()`，梯度经各次渲染直接回到 `render_params` → offsets → 网络；无代理、无多次 backward。

### 2.2 实现要点

- 在 Stage 1.1 的 `forward` 上扩展（或新类继承 Stage 1.1）：
  - 仍只算一次 `render_params`（与单 target 逻辑一致）。
  - 对 `batch["targets"]` 循环：`pred_rgb_i, _ = _render_single_view(render_params, view_i, h_i, w_i)`，`loss_i = F.l1_loss(pred_rgb_i, gt_image_i)`。
  - 总 loss：`loss = loss_i.mean()` 或 `loss = loss_i.sum() / num_targets`（建议取均，便于与单 target 量级可比）。
- `train_step`：`optimizer.zero_grad()` → `forward(batch)` → `loss.backward()` → `optimizer.step()`；并保留 Stage 1.1 的 h_cache_bg 写回与 `update_node_state_interval` 的 NodeState 更新逻辑。

### 2.3 数据与配置

- Overfit batch 需提供**至少 3 个 target**（或配置 `num_targets`，如 3）；`convert_batch_to_minimal_format` 保留 `targets[:num_targets]`，不再只取 `targets[0]`。
- 配置建议：`num_targets: 3`（或从 batch 取满），`loss_per_view: true`（对每个 view 的 loss 取均再 backward）。

### 2.4 验证

- 多 target 上 loss 下降、平均 PSNR 上升。
- `num_targets=1` 时与 Stage 1.1 单 target 行为一致（相同 loss/曲线），便于回归。

---

## 3. Stage 2.1：多 Target + 代理参数 + 梯度累积

### 3.1 目标

- **输入**：与 2.0 相同（多 target，如 3 个）。
- **前向**：同样只算**一套** `render_params`（与 2.0 相同）；从 `render_params` 创建**代理参数**（`proxy = render_param.detach().requires_grad_(True)`）。
- **多 view 循环**：对每个 target 用**同一组代理**渲染 → `loss_i = L(pred_rgb_i, gt_image_i) / num_targets` → `loss_i.backward()`（梯度累积到代理），不在此循环内 `zero_grad`。
- **回传**：`_backward_to_render_params(render_params, proxies)`，将代理上的梯度回传到 `render_params`，再经计算图回传到 offsets → 网络。
- **优化**：单次 `optimizer.step()`；并保留 h_cache_bg、NodeState 更新（与 Stage 1.1 一致）。

### 3.2 与 StreetForward_Flow 的对应

- 代理创建：与 [proxy_rendering_mixin.py](../../models/streetforward/proxy_rendering_mixin.py) 中 `_create_proxy_params` 一致；Minimal 仅一组（bg），无 rigid/distant。
- 多 target 渲染与损失：每 target 用代理渲染，`loss_i / num_targets` 再 `loss_i.backward()`。
- 回传：`_backward_to_render_params`，用 `torch.autograd.backward(render_tensors, grad_tensors)` 将 proxy.grad 传到 render_params。

### 3.3 与 Stage 2.0 的一致性对比

- **同一 overfit batch、同一 num_targets（如 3）**下对比：
  - 2.0：无代理，`loss = mean(loss_i)`，一次 `loss.backward()`。
  - 2.1：代理路径，`loss_i/num_targets` 各 backward，再 `_backward_to_render_params`。
- 期望：两者 loss 曲线接近（至少同量级、同趋势）；若存在数值差异，需在文档中说明（例如梯度累积顺序、除法时机等），并确认 2.1 的梯度与 2.0 在数学上等价或可接受。

### 3.4 实现要点

- 代理键名：`_render_single_view` 当前接受 `render_params` 的键 `means_r`, `scales_r`, `quats_r`, `opacities_r`, `colors_r`；代理可用 `means_p` 等，在传入渲染时做一层薄封装（例如内部用 `params["means_p"] if "means_p" in params else params["means_r"]`），或统一为同一键名由调用方传入。
- 其余同 [原文档 §4](#4-实现要点)（代理创建、多 view 循环、回传、单次 step）。

---

## 4. 实现入口与文件

### 4.1 模型

- **Stage 2.0**：在 `minimal_trainer_stage1_1.py` 基础上扩展，或新建 `minimal_trainer_stage2_0.py`，类名如 `MinimalStreetForwardStage2_0`。仅重写/扩展 `forward`（多 target 渲染 + 损失聚合），`train_step` 保持单次 backward + h_cache 与 NodeState 更新。
- **Stage 2.1**：在 2.0 或 Stage 1.1 基础上，新建 `minimal_trainer_stage2_1.py`，类名如 `MinimalStreetForwardStage2_1`。在 `forward` 中：算出一套 `render_params` → 创建 proxy → 多 target 渲染并逐 view `loss_i.backward()` → 返回便于 `train_step` 调用的结构；`train_step` 内调用 `_backward_to_render_params` 再 `optimizer.step()`，并保留 h_cache、NodeState 更新。

### 4.2 训练脚本

- **Stage 2.0**：新建 `tools/train_minimal_streetforward_stage2_0.py`，复用 stage1_1 的 batch 加载与转换逻辑，但 **保留多 target**（如 `targets_minimal = targets[:num_targets]`，`num_targets` 默认 3）；使用 `MinimalStreetForwardStage2_0`。
- **Stage 2.1**：新建 `tools/train_minimal_streetforward_stage2_1.py`，同样多 target，使用 `MinimalStreetForwardStage2_1`；日志中可同时报告平均 loss 与各 view PSNR，便于与 2.0 对比。

### 4.3 配置

- `configs/minimal_streetforward_stage2_0.yaml`：在 stage1_1 配置基础上增加 `num_targets: 3`、`loss_per_view: true` 等。
- `configs/minimal_streetforward_stage2_1.yaml`：同上，用于 2.1；可与 2.0 共用同一 overfit batch 路径，便于对比。

### 4.4 数据

- Overfit batch 需包含**至少 3 个 target**（多 view 或多帧）；若 capture 当前只有单 target，需扩展 overfit one batch 的 capture/格式，支持多 target。

---

## 5. 参考文件

- [Minimal_StreetForward_Design_Plan](Minimal_StreetForward_Design_Plan.md) — Stage 2 定义
- [Minimal_StreetForward_Next_Steps_Stage1_1_GRU](Minimal_StreetForward_Next_Steps_Stage1_1_GRU.md) — Stage 1.1 模型说明
- [StreetForward_Flow](StreetForward_Flow.md) — 代理参数、多 target 渲染、梯度回传（§7–9）
- [models/streetforward/minimal_trainer_stage1_1.py](../../models/streetforward/minimal_trainer_stage1_1.py) — Stage 1.1 模型（Stage 2 的基类）
- [tools/train_minimal_streetforward_stage1_1.py](../../tools/train_minimal_streetforward_stage1_1.py) — Stage 1.1 训练脚本与 batch 转换
- [models/streetforward/proxy_rendering_mixin.py](../../models/streetforward/proxy_rendering_mixin.py) — 代理创建、多 target 渲染、回传接口

---

## 6. 小结

| 项目 | Stage 2.0 | Stage 2.1 |
|------|-----------|-----------|
| **基础模型** | Stage 1.1 | Stage 1.1 |
| **Target 数** | 多（如 3） | 多（如 3） |
| **代理参数** | 不使用 | 使用 |
| **反向** | 单次 loss.backward() | 每 view loss_i.backward() 累积到 proxy，再 _backward_to_render_params |
| **对比** | 基线 | 与 2.0 同数据对比，保证一致性 |

先实现 2.0 验证多 target 链路正确，再实现 2.1 并与 2.0 对比，确保代理路径与无代理结果一致或可解释差异。
