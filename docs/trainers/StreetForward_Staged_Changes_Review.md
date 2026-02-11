# StreetForward 暂存区修改反直觉检查

本文档对当前暂存区（staged）的修改进行反直觉审查，对照 `StreetForward_Logging_System_Update_Plan.md` 与 `StreetForward_Flow.md`，检查是否存在偷工减料、错误实现或违背流程文档的行为。

> **2025-02-07 修复**：以下 P0/P1 问题已修复：permute、max_dense_elements、feature_fusion、preflight 阈值、train_streetforward 重复 TB 写入。

---

## 一、严重问题（必须修复）

### 1.1 【致命】feature_volume_mixin.py：permute 维度顺序错误 ✅ 已修复

**位置**：`feature_volume_mixin.py` 第 125 行

**暂存区修改**：
```python
dense_volume = dense_volume.permute(0, 4, 1, 2, 3)
```

**正确实现（StreetForward_Flow.md 第 761 行、proxy_rendering_mixin.py 第 380 行）**：
```python
dense_volume = dense_volume.permute(0, 4, 3, 2, 1)  # [1, C, D, H, W]
```

**分析**：
- `sparse_to_dense_volume` 返回 `[X, Y, Z, C]`，unsqueeze 后为 `[1, X, Y, Z, C]`
- `permute(0, 4, 3, 2, 1)` → `[1, C, Z, Y, X]` = `[1, C, D, H, W]`，与 `grid_sample` 期望的 `[B, C, D, H, W]` 一致
- `permute(0, 4, 1, 2, 3)` → `[1, C, X, Y, Z]`，等价于 `[1, C, W, H, D]`，**轴顺序错误**

**影响**：将导致特征插值与坐标错位，属于 Logging Plan 第二节「坐标维度/permute/grid_sample 错位」长尾隐患的直接引入，训练会收敛慢或崩溃。

**结论**：**违背 StreetForward_Flow.md**，必须改回 `permute(0, 4, 3, 2, 1)`。

---

### 1.2 【严重】config max_dense_elements 默认值过小 ✅ 已修复

**位置**：`configs/streetforward/multi_scene.yaml` 第 166 行

**暂存区修改**：
```yaml
max_dense_elements: 200000000
```

**分析**：
- 示例 `vol_dim≈[400,248,900]` 时，`vol_prod = 89,280,000`，`dense_elements = vol_prod * C ≈ 2.86e9`（C=32）
- `200000000`（2e8）远小于典型 dense 元素数，会导致几乎所有 batch 触发 `RuntimeError`
- Logging Plan 3.2 节：「`prod(vol_dim) > VMAX`：跳过 / 缩小 voxel_size…」—— 阈值应能覆盖正常训练

**建议**：将默认值改为 `null`（不限制）或 `5000000000` 量级；或明确此为「fail‑fast 调试用」，正式训练时提高或禁用。

---

## 二、潜在问题（需确认或修正）

### 2.1 feature_volume_mixin：effective_mask 逻辑变更

**暂存区修改**：当 `mask_src_rigid` 和 `idx_src_rigid` 未提供时，不再 `raise ValueError`，改为使用 `rigid_visible_mask` / `rigid_in_crop_mask` 作为 fallback。

**StreetForward_Flow.md 约定**：
> 调用方必须在 `_build_3d_feature_volume` 前调用此方法，并传入 `mask_src_rigid` 与 `idx_src_rigid`。

**分析**：训练路径 `_train_inner_iteration` 始终传入 `masks.mask_src_rigid` 与 `masks.idx_src_rigid`，fallback 可能仅用于评估路径。需确认：
- 评估路径（如 `_compute_render_params`）是否调用 `_build_3d_feature_volume` 且不传 mask
- 若存在此类路径，fallback 合理；否则可能掩盖错误调用

**建议**：在文档或注释中明确 fallback 的适用场景，或对训练路径保持「必须传入」的约定。

---

### 2.2 preflight_sweep：vol_dim_prod 与 max_dense_elements 阈值混用 ✅ 已修复

**位置**：`tools/preflight_sweep_streetforward.py` 中 alerts 逻辑

**暂存区逻辑**：
```python
if record["vol_dim_prod"] is not None and record["vol_dim_prod"] > threshold:
    alerts.append(f"vol_dim_prod_gt_{threshold}")
if record["dense_elements_est"] is not None and record["dense_elements_est"] > threshold:
    alerts.append(f"dense_elements_gt_{threshold}")
```

**分析**：`threshold` 来自 `max_dense_elements`（dense 元素数），却同时用于 `vol_dim_prod`。两者量级不同（`vol_dim_prod` 约 1e8，`dense_elements_est` 约 3e9），混用同一阈值语义不清。

**建议**：区分 `max_vol_dim_prod` 与 `max_dense_elements`，或仅对 `dense_elements_est` 使用 `max_dense_elements` 阈值。

---

### 2.3 分模块 grad norm 未包含 feature_fusion ✅ 已修复

**位置**：`trainer._compute_grad_norms_by_module`

**Logging Plan 7.3 节**：network grads 应包含 `sparse_conv`、各 head、**feature_fusion**、GRU 层。

**暂存区实现**：包含 `image_feature_extractor`，**未包含** `feature_fusion`。

**建议**：若 `feature_fusion` 存在且参与训练，应加入分模块 grad norm 统计。

---

## 三、与 Logging Plan 的符合度

### 3.1 已满足项

| Plan 要求 | 暂存区实现 | 符合 |
|----------|------------|------|
| 移除 debug 日志 | 移除 `_debug_log` 调用 | ✅ |
| strict_proxy_grad | `_strict_proxy_grad_active` + step 判断 | ✅ |
| Proxy grad None → raise（严格模式） | `_grad_or_zero` 中 strict 分支 | ✅ |
| 哨兵：num_targets, N_bg, N_rigid, N_distant | `_collect_sentinel_metrics` | ✅ |
| 哨兵：mask_update_rigid.mean, idx_tgt_rigid | 同上 | ✅ |
| 哨兵：vol_dim_prod, dense_elements_est | `_record_volume_stats` + `_last_*` | ✅ |
| 哨兵：proxy grad norms | `_augment_sentinel_with_grads` | ✅ |
| 哨兵：分模块 grad norm | `_compute_grad_norms_by_module` | ⚠️ 缺 feature_fusion |
| 哨兵：max_memory_allocated_gb | 同上 | ✅ |
| 哨兵：render_params 数值范围 | `_render_stats` (opacities, quat_norm_dev, means, scales_log) | ✅ |
| 哨兵：offsets 统计 | `_offset_stats` | ✅ |
| detect_anomaly | `_update_runtime_flags` 中 set_detect_anomaly | ✅ |
| NaN/Inf 检测 | `_check_for_nan_inf`、`_maybe_alert_on_sentinel` | ✅ |
| 空 targets → raise | `train_iter` 中 `len(targets)==0` 时 raise | ✅ |
| 空点云 → raise | `node_state_mixin` 中 `len(fg_points)==0` 时 raise | ✅ |
| VMAX 硬阈值（dense 元素数） | `_record_volume_stats` 中 max_dense_elements 检查 | ⚠️ 默认值过小 |
| preflight 脚本 | `tools/preflight_sweep_streetforward.py` | ✅ |
| 单元测试 | `tests/test_streetforward_logging.py` | ✅ |

### 3.2 未实现或部分实现的 Plan 项

| Plan 要求 | 状态 |
|----------|------|
| 10 个单元/集成测试（体积插值、Rigid 变换、Proxy 梯度、Gate、梯度覆盖率） | 仅实现 3 个测试（strict_proxy_grad、sentinel_metrics、record_volume_stats） |
| h_cache 对齐校验（点数/point_ids hash 变化时 reset） | 未实现 |
| 配置 `log_level` 控制 logger 级别 | 已添加 `log_level: info`，但未确认是否实际作用于各模块 logger |

---

## 四、与 StreetForward_Flow 的符合度

### 4.1 违背项

| 违背点 | 说明 |
|--------|------|
| **permute 维度** | 使用 `permute(0, 4, 1, 2, 3)` 替代文档规定的 `permute(0, 4, 3, 2, 1)`，破坏 `[1, C, D, H, W]` 约定 |

### 4.2 符合项

- 训练流程顺序未变：`_precompute_rigid_masks` → `_build_3d_feature_volume` → … → `_backward_to_render_params`
- 插桩位置与 Plan 9.4 一致：`_train_inner_iteration`、`_build_3d_feature_volume`、`_backward_to_render_params`、`_log_to_tensorboard`
- `_record_volume_stats` 在 `_build_3d_feature_volume` 内、dense_volume 使用后调用，符合「得到 vol_dim 后」的插桩点

---

## 五、其他细节检查

### 5.1 train_streetforward 重复写入 TensorBoard ✅ 已修复

**暂存区修改**：在 `step % log_interval == 0` 时，除 `metric_logger` 外，还执行：
```python
tb_writer.add_scalar("train/lr", current_lr, step)
tb_writer.add_scalar("train/grad_norm", grad_norm, step)
```

**分析**：`trainer._log_to_tensorboard` 已写入 `train/lr` 和 `train/grad_norm`，且 `tb_log_every` 控制频率。若 `log_interval != tb_log_every`，可能出现同一 step 的双重写入或不同 step 的重复。建议统一由 `_log_to_tensorboard` 负责，避免在 `train_streetforward` 中重复写入。

### 5.2 proxy_rendering_mixin 中的 sentinel 属性

`_backward_to_render_params` 使用 `getattr(self, "sentinel_alert_on_nan", False)` 等。`ProxyRenderingMixin` 混入 `StreetForwardTrainer`，这些属性由 trainer 初始化设置，调用时 `self` 为 trainer，逻辑正确。

### 5.3 preflight_sweep 的 import 路径

使用 `from models.trainers.streetforward import StreetForwardTrainer`，而 `models/trainers/streetforward.py` 从 `models.streetforward` 重导出，路径有效。

---

## 六、总结与行动项

### 必须修复（P0）✅ 已完成

1. **恢复 permute**：`feature_volume_mixin.py` 第 125 行已改回 `permute(0, 4, 3, 2, 1)`。
2. **调整 max_dense_elements 默认值**：config 中已改为 `null`。

### 建议修复（P1）✅ 已完成

1. 在 `_compute_grad_norms_by_module` 中加入 `feature_fusion`（若存在）。
2. 在 preflight 中仅对 `dense_elements_est` 使用 `max_dense_elements` 阈值。
3. 移除 `train_streetforward` 中重复的 TB 写入，由 `_log_to_tensorboard` 统一负责。

### 待补充（P2）

1. 补齐 Logging Plan 第六节的 10 个单元/集成测试。
2. 实现 h_cache 对齐校验（点数/point_ids 变化时 reset cache）。

---

## 七、反直觉检查清单

| 检查项 | 结果 |
|--------|------|
| 是否有改变核心算法/数据流的 unintentional 修改？ | ⚠️ **有**：permute 修改会破坏插值正确性 |
| 是否存在配置与实现语义不一致？ | ⚠️ max_dense_elements 默认值过小 |
| 是否引入与 Flow 文档不符的维度/坐标约定？ | ❌ **有**：permute 违背 Flow |
| 是否需要保留的 debug 能力被误删？ | ✅ 符合 Plan 要求，移除 debug 日志 |
| 哨兵是否能覆盖 Plan 要求的主要指标？ | ✅ 基本覆盖，缺 feature_fusion |
| 严格模式与异常处理是否正确接入？ | ✅ 接入正确 |
