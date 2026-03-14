# StreetForward Golden Baseline 设计文档

本文档讨论在重构 `models/trainers/streetforward.py` 之前，如何建立并维护一个 **Golden Baseline（黄金基线）**，以确保重构前后行为等价。基线以 `notebooks/StreetForward_Demo.ipynb` 的**第八部分（训练循环演示）及其依赖**为基准，并遵循软件开发与深度学习的标准实践。不考虑 CI 集成与 stub/mock，测试需**保留多场景、多 segment、多 batch** 以覆盖真实数据流与状态演进。

---

## 目录

1. [目的与范围](#1-目的与范围)
2. [基准定义：第八部分及依赖](#2-基准定义第八部分及依赖)
3. [参考功能与流程](#3-参考功能与流程)
4. [需捕获的观测项](#4-需捕获的观测项)
5. [测试矩阵设计](#5-测试矩阵设计)
6. [确定性设计](#6-确定性设计)
7. [基线存储与格式](#7-基线存储与格式)
8. [比较策略与容差](#8-比较策略与容差)
9. [实现要点与边界](#9-实现要点与边界)

---

## 1. 目的与范围

### 1.1 背景

- `models/trainers/streetforward.py` 体量过大，需要拆分与模块化重构。
- 重构前必须确立**可复现、可比对**的 Golden Baseline，用于回归测试。
- 基线应覆盖「真实训练一步」的完整数据流（与 Notebook 第八部分一致），**不使用 stub**，以保证梯度、NodeState 更新、多 target 渲染等行为与生产一致。

### 1.2 目标

- **定义基线内容**：在固定种子、固定数据与固定配置下，对「多场景 × 多 segment × 多 batch」运行指定步数的 `train_iter`，并记录一组**标量与张量摘要**。
- **回归测试**：重构后，在相同输入与配置下重跑相同步骤，将新结果与基线比对；在约定容差内一致即视为通过。
- **覆盖关键路径**：涵盖静态背景、动态物体（如有）、多 target 视图、代理梯度回传、NodeState 读写等，与 [StreetForward_Flow.md](./StreetForward_Flow.md) 中描述的流程对应。

### 1.3 明确不在本方案内

- **CI 集成**：如何接入 CI、触发条件、环境等在本文中不展开。
- **Stub / Mock**：基线与回归测试使用**真实** `StreetForwardTrainer`、真实稀疏卷积与渲染管线（或与 Notebook 相同的依赖栈）
- **最小化测试集**：不追求「单 batch、单场景」的最小用例；需**保留多场景、多 segment、多 batch**，以检验跨 segment 的 NodeState 隔离、不同 batch 下的损失与梯度分布。

---

## 2. 基准定义：第八部分及依赖

### 2.1 第八部分在做什么

Notebook 第八部分「训练循环演示」的核心逻辑可概括为：

```
scheduler = dataset.create_scheduler(batches_per_segment=5, segment_order="random", ...)
for iteration in range(num_iterations):
    multi_scene_batch = scheduler.next_batch()   # 或 reset 后继续
    streetforward_batch = convert_batch_to_streetforward_format(multi_scene_batch, device)
    outputs = trainer.train_iter(
        batch=streetforward_batch,
        apply_update=True,
        update_state=True,
    )
    loss = outputs['total_loss'].item()
```

即：**来自 MultiSceneDataset 的 batch → 转为 StreetForward 格式 → 一次 `train_iter(apply_update=True, update_state=True)`**。每一 iter 会更新优化器与 NodeState，并依赖前一步的 NodeState（同一 (scene_id, segment_id) 会跨 batch 复用）。

### 2.2 依赖链（用于复现第八部分）

Golden Baseline 的「基准流程」应包含与第八部分相同的前置条件与调用链：

| 层级 | 内容 | 说明 |
|------|------|------|
| 环境与配置 | Part 1 导入；Part 2 的 `cfg`、`data_cfg`、`dataset_cfg`、`segment_aabb/segment_input_aabb` 等 | 决定 trainer 与 dataset 的构造参数 |
| 数据源 | Part 3–4：`MultiSceneDataset` 实例、pointcloud 生成方式 | 与 Part 8 使用同一 dataset / 同一配置文件 |
| Trainer | Part 6：`StreetForwardTrainer(cfg)`，含 model/optimizer 等 | 与 Part 8 使用同一配置与构建方式 |
| 数据驱动方式 | Part 8：`dataset.create_scheduler(...)` + `scheduler.next_batch()` | 决定「哪些 (scene, segment)」、每个 segment 多少个 batch、是否 shuffle |
| 格式与调用 | `convert_batch_to_streetforward_format(batch, device)` + `trainer.train_iter(..., apply_update=True, update_state=True)` | 与 Part 8 完全一致 |

基线脚本应能**复现上述依赖**（可从同一 yaml 与同一 dataset 构建），从而在相同「场景→段→batch 顺序」下得到可比对结果。

### 2.3 基准的「入口」与「步数」

- **入口**：以「第一次调用 `trainer.train_iter(...)` 且 `update_state=True`」为逻辑起点；之前可有单独的「dataset / scheduler / trainer 构造」步骤，这些步骤的随机性由全局种子控制（见 §6）。
- **步数**：建议对「基线录制」定义固定步数 \(N\)（例如 \(N=5\) 或 \(10\)），在**固定 scheduler 顺序**下连续执行 \(N\) 次 `next_batch → convert → train_iter`，并记录每步的观测项。回归时在同一顺序下再跑 \(N\) 步并比对。

---

## 3. 参考功能与流程

StreetForward 的语义与数据流以 [StreetForward_Flow.md](./StreetForward_Flow.md) 为准，Golden Baseline 需要覆盖其中与「单步 train_iter」相关的部分，主要包括：

1. **NodeState 的获取与初始化**  
   `_get_or_init_node_states(batch)`：对 (scene_id, segment_id) 返回或初始化 Background / Rigid / Distant NodeState；若为首次则该步会从 pointcloud 初始化。
2. **3D 特征体积**  
   静态 + 动态点合并 → 稀疏张量 → 稀疏卷积 → 密集体积 → 插值得到 per-point 特征；动态分支含可见性/ Crop 掩码。
3. **2D 特征与融合（若开启）**  
   源视图特征提取、反投影、与 3D 特征融合；Baseline 可在 `use_2d_features=False` 下先建基线，再单独为 2D 分支加一组配置。
4. **偏移预测与掩码**  
   同一组 MLP 预测 offset_pos / offset_scales / offset_quat / offset_opacity / offset_sh；对 rigid 做可见性掩码。
5. **渲染参数与代理**  
   从 NodeState + offsets 得到渲染参数，再 detach+requires_grad 得到 proxy；对多个 target 视图依次渲染、损失、backward，梯度在 proxy 上累积后再反传到渲染参数与网络。
6. **梯度回传与状态更新**  
   `apply_update=True` 时执行优化器步；`update_state=True` 时将当前步的渲染参数写回对应 NodeState（含 clamp 等）。

Baseline 的观测项设计应能反映上述环节的「数值与形状」是否在重构前后一致（见 §4）。

---

## 4. 需捕获的观测项

在每一步 `train_iter` 后记录以下内容，用于回归对比与调试。

### 4.1 标量与配置摘要（每步）

- `total_loss`：`outputs["total_loss"].item()`（或等价字段）。
- `scene_id`, `segment_id`：当前 batch 的 (scene, segment)。
- `num_targets`：`len(batch["targets"])` 或等价。
- 若存在：`num_bg`, `num_rigid`, `num_distant`（当前 step 参与计算的点的数量，可从 NodeState 或内部变量读取）。

这些用于快速判断「是否跑在同一 (scene, segment) 且 target 数一致」，以及 loss 曲线是否一致。

### 4.2 每步的 NodeState 摘要（不存完整张量时）

为控制基线文件大小，可只存「摘要统计」而非完整参数：

- **Background**（若存在）：  
  `means` 的 min/max/mean、L2 范数；`scales_log`、`opacity_logit`、`sh_dc` 的类似统计；`num_points`。
- **Rigid**（若存在）：  
  同上；另加 `instances_quats` / `instances_trans` 的逐实例摘要、`point_ids` 的分布（如每实例点数）。
- **Distant**（若存在）：  
  同 Background 的摘要方式。

若基线仅用于「回归通过/不通过」，可优先存标量摘要；若用于精细调试，可对少数 step 额外保存完整张量到单独文件（见 §7）。

### 4.3 梯度与偏移量摘要（可选但推荐）

- **Last offsets**：  
  若 trainer 暴露或可注入 hook，记录 `_last_offsets_bg`（及 rigid/distant 若有）中 `offset_pos` / `offset_scales` / `offset_quat` 等的 min/max/mean/范数。用于检查前向与梯度是否改变。
- **关键参数梯度范数**：  
  对 `sparse_conv`、`mlp_offset_pos`、`mlp_conv`、`mlp_opacity`、`gaussion_decoder` 等若干参数的梯度做 global norm 或 per-module norm，记标量。用于确认梯度连通性在重构后未被破坏。

若实现成本高，可先在基线中只存 `total_loss` 与 NodeState 摘要，后续再加梯度与 offset 摘要。

### 4.4 可选：中间张量指纹

在关键节点（如 3D 特征体积插值后、offset 预测后、合并渲染参数后）对张量做确定性指纹（如 shape + sum + std + 少数分位数），写入基线。回归时在同一位置计算指纹并比对。仅在对「数值完全可复现」有强需求时采用，并需注意实现位置与 baseline 版本绑定，避免频繁因实现细节变动而失效。

---

## 5. 测试矩阵设计

为保证「多场景、多 segment、多 batch」的覆盖，测试矩阵建议如下。

### 5.1 维度定义

- **场景 (scene)**：至少 2 个不同的 `scene_id`（例如来自 `train_scene_ids` 中不同 id）。用于检查 NodeState 按 (scene_id, segment_id) 隔离、无串扰。
- **Segment**：每个场景至少 2 个 segment。用于检查同一场景下不同段的 NodeState 独立、首次进入某段时初始化正确。
- **Batch**：每个 segment 至少 2 个 batch（即同一 (scene, segment) 被取到 ≥2 次）。用于检查同一段上多步 `train_iter` 的累计更新、loss 随步数变化、梯度与 NodeState 的递推关系。

因此，最小测试规模可设计为：

- 2 场景 × 2 segment/场景 × 2 batch/segment ⇒ 至少 8 次 `train_iter` 调用（具体顺序由 scheduler 决定，见 §6）。

实际录制基线时，可以是「固定 scheduler 下连续 \(N\) 步」，其中 \(N\ge 8\)，且这 \(N\) 步中要覆盖到上述场景/段/batch 组合；或显式枚举 (scene, segment)，对每个 (scene, segment) 跑若干固定 batch 数，再合并为一条基线序列。

### 5.2 数据与顺序的确定性

- 使用固定 `create_scheduler(..., segment_order="sequential", scene_order="sequential", shuffle_segments=False)`（或等价顺序配置）保证每次运行的 batch 顺序一致。
- 若无法完全顺序化，则需将「本次运行实际采用的 (scene_id, segment_id) 序列」记录下来，作为基线元数据的一部分；回归时用同一序列或同一 scheduler 配置复现。

### 5.3 场景与 segment 的选取

- 优先从已有 MultiSceneDataset 配置（与 Notebook 一致）中选取「确有足够 keyframe、能生成非空 pointcloud」的 scene 和 segment。
- 若某段仅有背景、无动态物体，则基线中该段对应的 rigid 相关观测可为「无」或占位；若有动态物体，则必须覆盖到含 rigid 的 (scene, segment)，以验证动态分支与可见性掩码。

---

## 6. 确定性设计

为保证「同一输入 → 同一基线」，需固定所有主要随机源与设备相关行为。

### 6.1 随机种子

- 在**任何**数据加载与模型前向之前，统一设置：
  - `torch.manual_seed(seed)`，`torch.cuda.manual_seed_all(seed)`
  - `np.random.seed(seed)`，`random.seed(seed)`（若用到）
- 基线文件与脚本中应记录采用的 `seed`（如 42）；回归时使用相同 seed。

### 6.2 数据顺序与 DataLoader

- 使用固定顺序的 scheduler（见 §5.2），且不在此流程中引入额外 shuffle。
- 若 MultiSceneDataset 内部有随机性（如采样 keyframe），需通过种子或固定配置关闭或固定，使「同一 (scene, segment) 多次运行」得到的 batch 内容一致。

### 6.3 设备与数据类型

- 明确基线是在 **CPU** 还是 **CUDA** 上录制。若 Baseline 在 CUDA 上，回归也应在 CUDA 上比对，避免 cuDNN 等非确定性导致的差异。
- 在 CUDA 上为追求可复现，可设置 `torch.backends.cudnn.deterministic=True` 与 `torch.backends.cudnn.benchmark=False`（权衡性能与确定性由项目决定）。
- 数据类型（float32/float16）、`voxel_size`、`bbx_min/max` 等与 Notebook/配置一致。

### 6.4 版本与环境信息

- 基线元数据中记录：PyTorch 版本、Python 版本、关键依赖（如 gsplat、torchsparse/evol_splat）版本，以及录制时的 git commit 或代码快照标识，便于回归环境对齐。

---

## 7. 基线存储与格式

### 7.1 存储位置与命名

- 建议目录：`docs/trainers/golden/` 或项目内约定的 `baselines/` 目录。
- 命名示例：`streetforward_golden_<config_name>_seed<seed>_steps<N>_<device>.json`（或 `.npz` / 其他便于版本管理的格式）。
- 若分多文件（如「标量 + 摘要」与「少量完整张量」分离），可采用同一前缀加后缀区分。

### 7.2 建议的基线文件内容结构

```text
{
  "meta": {
    "config_name": "streetforward_multi_scene",
    "seed": 42,
    "num_steps": 10,
    "device": "cuda",
    "pytorch_version": "2.x.x",
    "scene_segment_sequence": [[scene_id, segment_id], ...]  // 每步的 (scene, segment)
  },
  "per_step": [
    {
      "step": 0,
      "scene_id": 1,
      "segment_id": 0,
      "total_loss": 0.123,
      "num_targets": 18,
      "num_bg": 500000,
      "num_rigid": 66642,
      "node_state_bg_summary": { "means_min": [...], "means_max": [...], ... },
      "node_state_rigid_summary": { ... } | null,
      "offset_bg_summary": { "offset_pos_norm": ..., ... } | null,
      "grad_norms": { "sparse_conv": ..., "mlp_offset_pos": ..., ... } | null
    },
    ...
  ]
}
```

- `per_step` 与「固定 scheduler 顺序下的步」一一对应。
- 若某步没有 rigid/distant，对应 summary 为 `null` 或省略。
- 若使用张量指纹，可在每步下增加 `"fingerprints": { "feat_3d_crop_bg": [...], ... }`。

### 7.3 完整张量（可选）

若需要逐参数比对，可对选定 step（如 step 0 与最后一步）将 NodeState 的 `means`、`scales_log`、`quats`、`opacity_logit`、`sh_dc`、`sh_rest` 等以 `.pt` 或 `.npz` 形式单独保存，并在元数据中注明对应 step 与 (scene_id, segment_id)。回归脚本则加载这些文件并与当前 run 的对应张量做 diff。

---

## 8. 比较策略与容差

### 8.1 标量（loss、摘要统计）

- **total_loss**：  
  `relative_tolerance`（如 1e-2）与 `absolute_tolerance`（如 1e-5）结合：  
  `|a - b| <= atol + rtol * max(|a|, |b|)`。  
  首步与后续步的 scale 可能不同，可根据经验为 loss 单独设更大 rtol（如 5e-2）。
- **NodeState 摘要中的标量**：  
  建议 `atol=1e-4`、`rtol=1e-2` 起步；若某统计量易受数值环境影响，可单独放宽并记录原因。

### 8.2 梯度范数

- 若记录了 per-module 梯度范数：  
  使用相对容差为主（如 rtol=1e-1），因为小梯度在 atol 下容易误报；同时允许「两者均为 0 或均为 nan」的等价处理规则。

### 8.3 张量直接比对（若存在）

- 使用 `torch.allclose(a, b, rtol=..., atol=...)` 或等价接口；建议 atol=1e-5、rtol=1e-3 作为默认，对敏感参数可收紧。
- 若比对的是「完整 NodeState 张量」，需同时比对 shape 与 dtype，再比数值。

### 8.4 通过/不通过规则

- 所有参与比对的项都在相应容差内 → 回归通过。
- 任一项超差 → 回归不通过，并输出首条失败项与差值，便于排查。可选：对「仅 loss 超差」与「梯度/NodeState 超差」区分报告，便于判断是优化动态还是前向/梯度逻辑变化。

---

## 9. 实现要点与边界


### 9.2 配置与数据依赖

- 基线录制依赖「可用的 MultiSceneDataset + 至少 2 场景、每场景至少 2 segment、每 segment 至少 2 batch」；若当前仓库内无此类数据，需在文档或脚本中说明「最小数据要求」与示例配置（如指向与 Notebook 相同的 yaml 与 data root）。
- 若希望 CI 或无数据环境下也能做轻量回归，可再考虑「用固定小点云 + 固定虚拟相机生成的 batch」做第二套 baseline（与「真实 MultiSceneDataset 基线」分开），但本方案以「真实多场景/段/batch」为主。

### 9.3 可维护性

- 基线格式与字段应有简短说明（可在本设计 doc 或 README 中），并避免在未更新文档的情况下增删字段。
- 当 intentionally 改变训练逻辑或损失定义时，应**重新录制** Golden Baseline，并注明原因与代码变更；回归脚本的容差也可随版本迭代调整，并在变更说明中记录。

---

## 10. 总结

| 项目 | 内容 |
|------|------|
| **基准来源** | Notebook 第八部分：scheduler → next_batch → convert_batch_to_streetforward_format → train_iter(apply_update=True, update_state=True) |
| **依赖** | Part 1–2 配置、Part 3–4 的 MultiSceneDataset、Part 6 的 StreetForwardTrainer；与 [StreetForward_Flow.md](./StreetForward_Flow.md) 一致的功能语义 |
| **测试矩阵** | 多场景（≥2）× 多 segment（≥2/场景）× 多 batch（≥2/段），顺序固定、可复现 |
| **观测项** | 每步 total_loss、(scene_id, segment_id)、num_targets、NodeState 摘要、可选 offset/梯度范数或张量指纹 |
| **确定性** | 全局种子、固定 scheduler 顺序、明确设备与 dtype、记录环境与版本 |
| **存储** | 结构化 JSON（或类似）+ 可选的少量张量文件；放置于 `docs/trainers/golden/` 或项目约定目录 |
| **比较** | 标量用 atol/rtol；张量用 allclose；明确通过/不通过规则与报告方式 |
| **不在本文范围** | CI 集成方式、stub/mock；且不缩小为单场景/单 segment/单 batch |

按此设计实现并维护 Golden Baseline 后，即可在重构 `streetforward.py` 时，通过回归脚本对「多场景、多 segment、多 batch」下的训练一步行为做持续比对，从而在保证行为一致的前提下安全地进行模块拆分与接口调整。
