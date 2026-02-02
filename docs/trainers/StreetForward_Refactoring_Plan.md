# StreetForward 重构方案

本文档讨论 `models/trainers/streetforward.py` 的重构方案，旨在解决当前文件臃肿（~3809 行）、难以维护与扩展的问题。重构需遵循深度学习的标准实践，并与 [StreetForward_Golden_Baseline_Design.md](./StreetForward_Golden_Baseline_Design.md) 对齐，以确保行为等价。

---

## 目录

1. [问题分析](#1-问题分析)
2. [原始版本保留](#2-原始版本保留)
3. [重构目标与原则](#3-重构目标与原则)
4. [模块拆分方案](#4-模块拆分方案)
5. [依赖关系与接口设计](#5-依赖关系与接口设计)
6. [Golden Baseline 对齐策略](#6-golden-baseline-对齐策略)
7. [对照 StreetForward_Flow 的潜在问题](#7-对照-streetforward_flow-的潜在问题)
8. [实施阶段建议](#8-实施阶段建议)
9. [风险与缓解](#9-风险与缓解)

---

## 1. 问题分析

### 1.1 当前结构概览

| 类型 | 数量 | 行数估算 | 说明 |
|------|------|----------|------|
| 模块级工具函数 | ~15 | ~350 | `_rgb_to_sh`, `_quat_*`, `get_viewmat`, `_pairwise_neighbor_distances` 等 |
| 数据类 (dataclass) | 3 | ~130 | `NodeStateBackground`, `NodeStateRigid`, `NodeStateDistant` |
| `StreetForwardTrainer` 方法 | ~51 | ~3300+ | 含 `train_iter`（单方法 ~600 行）、`_build_3d_feature_volume`、checkpoint、evaluate 等 |

### 1.2 痛点

1. **单文件过大**：难以快速定位逻辑，代码审查成本高。
2. **职责混杂**：几何变换、特征构建、渲染、训练循环、checkpoint、TensorBoard 等全在一个类中。
3. **train_iter 过长**：主训练循环与大量内部逻辑耦合，难以单元测试或替换子步骤。
4. **扩展困难**：添加新特征（如新的 2D 特征、损失项、调度策略）需在巨大文件中修改多处。
5. **与深度学习标准实践不符**：主流框架（如 PyTorch Lightning、MMDetection）通常将 Model、Data、Trainer、Callback 分离。

### 1.3 参考流程与数据流

- **训练流程**：详见 [StreetForward_Flow.md](./StreetForward_Flow.md)，核心为：
  - `_get_or_init_node_states` → `_build_3d_feature_volume` → 2D 特征（可选）→ `_predict_offsets` → `_render_params_from_offsets` → `_create_proxy_params` → 多 target 渲染与损失 → 梯度回传 → 状态更新
- **数据格式**：MultiSceneDataset 的 batch 格式见 [MultiSceneDataset_Usage.md](../dataloader/MultiSceneDataset_Usage.md)；转换为 StreetForward 格式后由 `train_iter` 消费。

---

## 2. 原始版本保留

### 2.1 目的

在重构前保留 `models/trainers/streetforward.py` 的**原始版本**，用于：

- Golden Baseline 录制与回归比对（重构前后对比）
- 回归不通过时的调试参考
- 紧急回退或 A/B 测试

### 2.2 保留方式

| 方式 | 说明 |
|------|------|
| **Git Tag** | 在开始重构前打 tag（如 `streetforward-pre-refactor`），可随时 checkout 恢复 |
| **副本文件** | 复制为 `models/trainers/streetforward_original.py`，作为只读参考，**不参与正常导入** |
| **分支** | 在 `refactor` 分支上重构，`main` 或单独分支保留原实现 |

### 2.3 推荐做法

1. **重构开始前**：执行 `git tag streetforward-pre-refactor` 或 `git tag streetforward-v1-baseline`
2. **可选**：将 `streetforward.py` 复制为 `streetforward_original.py` 放入 `models/trainers/`，在文件头部注明「仅作参考，不导入」
3. **测试脚本**：Golden Baseline 回归脚本可通过环境变量或配置选择「原始版」或「重构版」，例如：
   ```python
   # 回归脚本示例
   USE_ORIGINAL = os.environ.get("STREETFORWARD_USE_ORIGINAL", "0") == "1"
   if USE_ORIGINAL:
       from models.trainers.streetforward_original import StreetForwardTrainer  # 若有副本
   else:
       from models.trainers.streetforward import StreetForwardTrainer
   ```
4. **向后兼容**：重构后 `models/trainers/streetforward.py` 作为薄包装导入新实现，对外接口保持不变，因此**不需要**在业务代码中切换导入路径；原始版本仅用于回归与调试。

### 2.4 注意事项

- 若使用副本文件 `streetforward_original.py`，需将其加入 `.gitignore` 的**排除项**或明确纳入版本管理，避免被忽略；或放在 `docs/trainers/reference/` 等目录作为快照
- 原始版本与重构版本应能在**相同配置、相同数据、相同种子**下产生可比对结果

---

## 3. 重构目标与原则

### 3.1 目标

1. **可维护性**：每个模块职责清晰、行数可控（建议单文件 < 500 行）。
2. **可测试性**：各子模块可独立单元测试，Golden Baseline 回归覆盖主流程。
3. **可扩展性**：新增 2D 特征、损失、调度策略等时，改动局部化。
4. **行为等价**：重构前后 `train_iter` 在相同输入下产生相同输出（在 Golden Baseline 容差内）。

### 3.2 原则

| 原则 | 说明 |
|------|------|
| 单一职责 | 每个模块只负责一类任务（如几何、特征、渲染、训练循环）。 |
| 依赖注入 | 可替换的组件（sparse_conv、renderer、loss）通过构造函数注入。 |
| 接口稳定 | `StreetForwardTrainer.train_iter(batch, apply_update, update_state)` 签名与返回格式保持不变。 |
| 渐进式重构 | 分阶段拆分，每阶段均通过 Golden Baseline 回归。 |
| 文档同步 | 与 StreetForward_Flow.md、Golden_Baseline_Design.md 保持一致。 |

---

## 4. 模块拆分方案

### 4.1 建议的目录结构

```
models/
├── streetforward/                    # 新建包
│   ├── __init__.py                   # 导出 StreetForwardTrainer, NodeState* 等
│   ├── math_utils.py                 # 四元数、SH、viewmat 等纯数学工具
│   ├── node_states.py                # NodeStateBackground, NodeStateRigid, NodeStateDistant
│   ├── feature_volume.py             # 3D 特征体积构建
│   ├── offsets.py                    # 偏移量预测与渲染参数计算
│   ├── proxy_rendering.py            # 代理参数、多 target 渲染、梯度回传
│   ├── trainer.py                    # StreetForwardTrainer 主类（编排逻辑）
│   └── checkpoint.py                 # 保存/加载 checkpoint、NodeState 序列化
├── trainers/
│   └── streetforward.py              # 向后兼容：from models.streetforward import *
```

### 4.2 模块职责划分

| 模块 | 职责 | 迁移内容 |
|------|------|----------|
| `math_utils.py` | 纯数学工具，无模型/设备依赖 | `_num_sh_bases`, `_rgb_to_sh`, `_sh_to_rgb`, `_random_quat_tensor`, `_quat_multiply`, `_quat_conjugate`, `_normalize_quat`, `_quat_to_rotmat`, `_axis_angle_to_quat`, `get_viewmat`, `_pairwise_neighbor_distances` |
| `node_states.py` | NodeState 定义与初始化 | `NodeStateBackground`, `NodeStateRigid`, `NodeStateDistant`, `_compute_initial_scales`, `_init_node_state_from_arrays`, `_init_node_from_pointcloud`, `_init_rigid_node_state_from_pcd`, `_extend_rigid_frames` |
| `feature_volume.py` | 3D 特征体积构建 | `_build_3d_feature_volume`, `get_grid_coords`, `interpolate_features`, `_prepare_gaussians_for_source`, `_prepare_all_gaussians`, `_compute_2d_features`, `_compute_2d_features_all`, `_fuse_features` |
| `offsets.py` | 偏移预测与渲染参数 | `_mask_rigid_offsets`, `_predict_offsets`, `_render_params_from_offsets`, `_transform_offsets_world_to_local` |
| `proxy_rendering.py` | 代理、合并、渲染、损失 | `_create_proxy_params`, `_merge_all_params`, `compute_loss`, `_compute_render_params`, `_render_single_view`，以及多 target 循环中的渲染与梯度回传逻辑 |
| `checkpoint.py` | 持久化 | `_node_state_to_dict`, `_node_state_from_dict`, `_node_state_rigid_to_dict`, `_node_state_rigid_from_dict`, `_node_state_distant_from_dict`, `save_checkpoint`, `load_checkpoint` |
| `trainer.py` | 编排与训练循环 | `StreetForwardTrainer.__init__`, `_get_or_init_node_states`, `_resolve_rigid_frame_idx`, `_per_point_pose_valid`, `_visible_mask_from_instances_fv`, `_transform_rigid_to_world`, `_transform_rigid_quats_to_world`, `train_iter`, `forward`, `evaluate`, `_evaluate_test_views`, `_compute_psnr`, `_compute_ssim`, `_compute_lpips`, `_log_to_tensorboard`, `close` |

### 4.3 依赖关系图

```
                    trainer.py
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
  node_states.py   feature_volume.py  offsets.py
        │                │                │
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                  proxy_rendering.py
                         │
                         ▼
                    checkpoint.py
                         │
                         ▼
                    math_utils.py
```

- `math_utils`：无内部依赖，可被任意模块使用。
- `node_states`：依赖 `math_utils`。
- `feature_volume`：依赖 `math_utils`, `node_states`。
- `offsets`：依赖 `math_utils`, `node_states`。
- `proxy_rendering`：依赖 `math_utils`, `node_states`, `offsets`。
- `checkpoint`：依赖 `node_states`。
- `trainer`：依赖所有子模块。

---

## 5. 依赖关系与接口设计

### 5.1 子模块接口（建议）

各子模块以「函数 + 可选小类」为主，减少与 `StreetForwardTrainer` 的紧耦合：

- **feature_volume**：提供 `build_3d_feature_volume(trainer, node_state_bg, node_state_rigid, ...)` 或封装为 `FeatureVolumeBuilder(trainer)`，接收 trainer 的 `sparse_conv`、`construct_sparse_tensor` 等。
- **offsets**：提供 `predict_offsets(mlps, feat_bg, feat_rigid, ...)` 和 `render_params_from_offsets(node_state, offsets, eta_*, ...)`。
- **proxy_rendering**：提供 `create_proxy_params`, `merge_all_params`, `render_and_accumulate_loss(targets, proxies, ...)`，或封装为 `ProxyRenderer(trainer)`。

`StreetForwardTrainer` 负责：
- 持有配置、设备、子模块实例；
- 调用各子模块完成 `train_iter` 的步骤编排；
- 管理 `node_states` 字典、优化器、TensorBoard。

### 5.2 深度学习标准实践参考

| 实践 | 应用方式 |
|------|----------|
| **Model / Trainer 分离** | `StreetForwardTrainer` 作为 Trainer，内部持有「模型组件」（sparse_conv、MLP、feature_extractor），但不把训练循环逻辑塞进模型类。 |
| **数据与训练解耦** | batch 格式由 `convert_batch_to_streetforward_format` 统一处理，Trainer 只消费标准格式。 |
| **Callback 机制** | TensorBoard、checkpoint 可抽象为 `TrainingCallback`，由 Trainer 在适当时机调用。 |
| **可配置损失** | `compute_loss` 可接受 `loss_fn` 参数，默认 L2，支持扩展 SSIM、LPIPS 等。 |
| **模块化特征** | 2D 特征、特征融合作为可选插件，通过 `use_2d_features` 与 `FeatureFusion` 注入。 |

---

## 6. Golden Baseline 对齐策略

### 6.1 回归测试要求

根据 [StreetForward_Golden_Baseline_Design.md](./StreetForward_Golden_Baseline_Design.md)：

- 基准：Notebook 第八部分流程：`scheduler.next_batch` → `convert_batch_to_streetforward_format` → `train_iter(apply_update=True, update_state=True)`。
- 测试矩阵：多场景（≥2）× 多 segment（≥2/场景）× 多 batch（≥2/段），顺序固定。
- 观测项：每步 `total_loss`、`(scene_id, segment_id)`、`num_targets`、NodeState 摘要、可选 offset/梯度范数。

### 6.2 重构中的对齐措施

1. **入口不变**：`StreetForwardTrainer(batch, apply_update, update_state)` 的调用方式与返回结构保持不变。
2. **分阶段验证**：每完成一个子模块迁移，运行 Golden Baseline 回归；全部迁移后，再运行完整回归。
3. **确定性**：迁移过程中不改变随机种子、scheduler 顺序、数据类型、设备行为。
4. **观测点一致**：`utils/streetforward_baseline.py` 中用于录制/比对的观测项，在重构后应从同一逻辑路径获取（如 `outputs["total_loss"]`、NodeState 的同一字段）。
5. **文档同步**：StreetForward_Flow.md 中若引用具体方法名，随重构更新为新模块下的函数/方法路径。

### 6.3 回归不通过时的排查

- 优先检查：新拆分的模块在输入相同时，输出张量是否与旧实现一致（shape、dtype、allclose）。
- 其次检查：`train_iter` 内部调用顺序、mask 传递、梯度回传路径是否与 StreetForward_Flow.md 一致。
- 最后检查：NodeState 更新逻辑、clamp 范围、eta 因子是否被错误修改。

---

## 7. 对照 StreetForward_Flow 的潜在问题

以下问题基于 [StreetForward_Flow.md](./StreetForward_Flow.md) 与当前实现的对照，重构时需特别注意，避免遗漏或错误拆分。

### 7.1 Mask 预计算逻辑（train_iter 内）

**Flow 描述**：使用 `rigid_visible_mask` 与 `rigid_in_crop_mask` 筛选动态点参与 3D 体积构建。

**实际实现**：`train_iter` 在 inner 循环开始时预计算一组更细粒度的 mask：

- `mask_src_rigid = pose_valid_src & visible_src`：source 帧有有效位姿且可见
- `mask_tgt_rigid = [pv & vis for pv, vis in zip(pose_valid_tgt, visible_tgt)]`：各 target 帧
- `mask_update_rigid = mask_src_rigid & mask_any_tgt_rigid`：仅在 source 和至少一个 target 可见时更新
- `idx_src_rigid`, `idx_tgt_rigid`：用于索引与 gate

**潜在问题**：重构时若把 `_build_3d_feature_volume` 迁到 `feature_volume.py`，必须保留 `mask_src_rigid` 与 `idx_src_rigid` 的传入；`mask_update_rigid` 在 offsets 阶段用于 gate rigid offsets。这些 mask 的计算依赖 `_per_point_pose_valid`、`_visible_mask_from_instances_fv`，应与 trainer 或 node_states 模块保持一致。拆分时**不要**把 mask 计算逻辑拆散到互不关联的模块。

### 7.2 动态物体偏移量的世界→局部变换

**Flow 描述**：动态物体的 offsets 在 source 帧世界坐标系预测，需经 `_transform_offsets_world_to_local` 转为局部坐标后再用于 `_render_params_from_offsets`。

**潜在问题**：`offsets.py` 中的 `_transform_offsets_world_to_local` 依赖 `node_state_rigid`（`instances_quats`、`instances_trans`、`cur_frame`）和 source 帧。拆分 offsets 模块时，接口需显式传入 `node_state_rigid` 及 source frame 信息，避免隐式依赖 trainer 内部状态。

### 7.3 两步梯度回传与 _grad_or_zero

**Flow 描述**：先用 `torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)` 将代理梯度反传到渲染参数，再依赖 PyTorch 自动链式传播到网络。

**实际实现**：`_grad_or_zero` 是 `train_iter` 内的嵌套函数，用于从代理参数取梯度（若为 None 则用零），再传给 `autograd.backward`。

**潜在问题**：若将梯度回传逻辑迁到 `proxy_rendering.py`，`_grad_or_zero` 须一并迁移，且需保证与代理/渲染参数的结构完全对应（bg、rigid、distant 及各自的 means、scales、quats、opacities、colors）。拆时注意不要破坏 `render_tensors` 与 `grad_tensors` 的一一对应顺序。

### 7.4 Distant 点云的来源

**Flow 描述**：`NodeStateDistant` 表示背景远景。

**实际实现**：distant 来自**静态背景点云中**落在 `input_aabb` 内但**不在** `crop_aabb`（bbx）内的点，即 `fg_points`（crop 内）与 `distant_points`（crop 外、input_aabb 内）的划分。**不是**从 `pointcloud["distant"]` 读取。

**潜在问题**：`node_states.py` 或 trainer 的初始化逻辑必须保持「background → fg/distant 按 AABB 划分」的语义，避免误以为存在独立的 `pointcloud["distant"]` 字段。

### 7.5 get_grid_coords 与 vol_dim 的依赖

**Flow 描述**：`get_grid_coords(means_s, bbx_min, vol_dim, voxel_size)` 将世界坐标转为归一化网格坐标；`vol_dim` 来自 `construct_sparse_tensor`。

**潜在问题**：`feature_volume.py` 中 `get_grid_coords` 和 `interpolate_features` 依赖 `vol_dim`，而 `vol_dim` 由 `construct_sparse_tensor` 返回。数据流为：`construct_sparse_tensor` → `vol_dim` → `sparse_conv` → `sparse_to_dense_volume` → `permute` → `get_grid_coords` → `interpolate_features`。拆分时需保证该顺序和维度约定（如 permute 后 `[1, C, D, H, W]` 供 `grid_sample`）不被破坏。

### 7.6 NodeState 类型别名

**实现**：`NodeState = NodeStateBackground`；`_node_state_from_dict` 返回 `NodeState`，实际为 `NodeStateBackground`。checkpoint 的序列化/反序列化依赖该约定。

**潜在问题**：迁移到 `node_states.py` 时保留 `NodeState` 别名，避免 checkpoint 加载逻辑出错。

### 7.7 调试日志 _debug_log

**实现**：`train_iter` 内存在 `_debug_log` 调用（如 inner 循环前后、`_build_3d_feature_volume` 之后），写入 `.cursor/debug.log`。

**潜在问题**：重构时若将 `train_iter` 拆成多步调用，需决定 `_debug_log` 保留在 trainer 的编排层，还是迁入子模块。建议保留在 trainer 层，便于统一控制日志粒度，避免子模块与调试逻辑耦合。

### 7.8 小结

| 问题 | 涉及模块 | 建议 |
|------|----------|------|
| mask 预计算与传递 | trainer, feature_volume, offsets | 在 trainer 中计算 mask，显式传给 feature_volume 和 offsets |
| offsets 世界→局部变换 | offsets | 接口显式传入 node_state_rigid 与 source frame |
| _grad_or_zero 与两步 backward | proxy_rendering | 迁移时保持与代理/渲染参数结构一致 |
| distant 来源 | node_states / trainer | 保持 fg/distant 按 AABB 划分，不假设 pointcloud["distant"] |
| get_grid_coords 与 vol_dim | feature_volume | 保持 construct_sparse_tensor → vol_dim → interpolate 数据流 |
| NodeState 别名 | node_states | 保留 `NodeState = NodeStateBackground` |
| _debug_log | trainer | 保留在 trainer 编排层 |

---

## 8. 实施阶段建议

### Phase 1：提取纯工具与数据类（低风险）

- 新建 `math_utils.py`、`node_states.py`。
- 将工具函数与 NodeState 迁移过去，`streetforward.py` 改为 `from .math_utils import ...` 等。
- 运行 Golden Baseline 回归 → 应无变化。

### Phase 2：提取特征体积与偏移模块（中风险）

- 新建 `feature_volume.py`、`offsets.py`。
- 将 `_build_3d_feature_volume`、`_predict_offsets`、`_render_params_from_offsets` 等迁移。
- Trainer 通过导入调用，或注入 `FeatureVolumeBuilder`、`OffsetPredictor` 实例。
- 每步迁移后运行回归。

### Phase 3：提取代理渲染与 checkpoint（中风险）

- 新建 `proxy_rendering.py`、`checkpoint.py`。
- 将多 target 渲染循环、梯度回传、checkpoint 逻辑迁移。
- `train_iter` 变为清晰的「编排代码」，调用各子模块。

### Phase 4：整理 trainer.py 与向后兼容

- 将剩余逻辑移入 `models/streetforward/trainer.py`。
- `models/trainers/streetforward.py` 保留为薄包装：`from models.streetforward import *`，保证现有 `from models.trainers.streetforward import StreetForwardTrainer` 仍然可用。

### Phase 5：可选增强

- 将 TensorBoard、checkpoint 抽象为 `TrainingCallback`。
- 将损失计算抽象为可配置的 `LossModule`。
- 为 `FeatureVolumeBuilder`、`OffsetPredictor` 等增加单元测试。

---

## 9. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 梯度流被破坏 | 迁移时保持 `requires_grad`、`detach`、`backward` 逻辑不变；回归测试包含梯度范数比对。 |
| NodeState 隔离错误 | 明确 `(scene_id, segment_id)` 为 key，不在迁移中改变缓存与清空逻辑。 |
| 性能回退 | 避免不必要的 `.clone()`、多余中间变量；必要时做 profile 对比。 |
| 导入循环 | 子模块间通过显式依赖顺序和 `TYPE_CHECKING` 避免循环导入。 |
| 第三方依赖变化 | 保持 `renderer`、`sparse_conv` 等通过构造函数注入，便于测试与替换。 |

---

## 10. 总结

| 项目 | 内容 |
|------|------|
| **目标** | 将 ~3809 行单文件拆分为职责清晰的子模块，提升可维护性与可扩展性。 |
| **原则** | 单一职责、依赖注入、接口稳定、渐进式重构、文档同步。 |
| **模块** | `math_utils`, `node_states`, `feature_volume`, `offsets`, `proxy_rendering`, `checkpoint`, `trainer`。 |
| **Golden Baseline** | 每阶段通过回归测试，入口与观测项保持一致。 |
| **实施** | 分 5 个阶段，从低风险工具与数据类开始，逐步迁移核心逻辑。 |

重构完成后，新增功能（如新的 2D 特征、损失、调度策略）可在对应模块中局部修改，并通过 Golden Baseline 保证主流程行为不变。
