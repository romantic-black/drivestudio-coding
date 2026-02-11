# StreetForward 日志系统更新：改动总结与用法指南

本文档总结当前暂存区的 StreetForward 日志系统更新，并说明如何配置与使用。

---

## 一、改动总览

### 1.1 涉及文件

| 文件 | 变更类型 |
|------|----------|
| `configs/streetforward/multi_scene.yaml` | 新增 sentinel、strict 模式、log_level；data 直接配置（无 dataset_preset） |
| `models/streetforward/feature_volume_mixin.py` | 移除 debug 日志；新增 `_record_volume_stats`、体积阈值检查 |
| `models/streetforward/node_state_mixin.py` | 空点云 raise |
| `models/streetforward/offsets_mixin.py` | 移除 debug 日志 |
| `models/streetforward/proxy_rendering_mixin.py` | 移除 debug 日志；strict_proxy_grad、NaN 检测 |
| `models/streetforward/trainer.py` | 哨兵指标、严格模式、分模块 grad norm、空 targets raise |
| `tools/train_streetforward.py` | 可选 dataset_preset 合并、log_level、ValueError 处理、metric_logger 扩展 |
| `tools/preflight_sweep_streetforward.py` | **新增**：训练前扫雷脚本 |
| `tests/test_streetforward_logging.py` | **新增**：哨兵与严格模式单元测试 |
| `docs/trainers/StreetForward_Logging_System_Update_Plan.md` | **新增**：方案设计文档 |
| `docs/trainers/StreetForward_Staged_Changes_Review.md` | **新增**：反直觉审查文档 |

### 1.2 核心功能

- **哨兵指标**：训练中记录 N_bg/N_rigid/N_distant、mask 比例、体积、梯度、显存等
- **严格模式**：前 N step 的 anomaly 检测、Proxy grad None → raise、NaN/Inf 检测
- **训练前扫雷**：`preflight_sweep_streetforward.py` 不训练、只前向统计
- **异常 fail-fast**：空 targets、空点云、体积超限直接 raise

---

## 二、配置说明

### 2.1 新增配置项（configs/streetforward/multi_scene.yaml）

```yaml
# data 字段直接配置 MultiSceneDataset（参考 docs/dataloader/MultiSceneDataset_Usage.md）
# pixel_source、lidar_source 等在 data 中完整配置，无需 dataset_preset

training:
  log_level: info                  # debug | info | warning
  
  # 严格模式（前 N step）
  strict_proxy_grad: false         # true 时 Proxy grad 为 None 直接 raise
  strict_proxy_grad_steps: 50     # 严格模式持续步数
  detect_anomaly_steps: 50        # autograd.set_detect_anomaly 持续步数
  
  sentinel:
    enabled: true                 # 是否启用哨兵指标
    log_every: 20                 # 每 N step 写入 TensorBoard
    alert_on_nan: true             # 任一哨兵指标 NaN/Inf 则 raise
    alert_on_grad_zero: false      # 梯度为 0 时 warning（不 raise）
    warn_on_proxy_grad_none: false # Proxy grad None 时 warning（不 raise）
    max_dense_elements: null       # 不限制；设整数值可 fail-fast 防 OOM（如 5e9）
```

### 2.2 配置推荐

| 场景 | 配置建议 |
|------|----------|
| **Canary run（试跑）** | `strict_proxy_grad: true`，`detect_anomaly_steps: 50`，`sentinel.enabled: true` |
| **正式训练** | `strict_proxy_grad: false`，`detect_anomaly_steps: 0`，`sentinel.enabled: true` |
| **Golden Baseline 回归** | 保持 `sentinel.enabled: true`（或 `false` 以加快回归），`strict_proxy_grad: false` |
| **显存紧张 / 防 OOM** | `sentinel.max_dense_elements: 5000000000`（约 5e9） |

---

## 三、用法

### 3.1 训练：train_streetforward.py

与之前相同，使用 config 启动：

```bash
python tools/train_streetforward.py --config_file configs/streetforward/multi_scene.yaml
```

**CLI 覆盖示例**：

```bash
# 覆盖 data 配置（如 data_root、相机等）
python tools/train_streetforward.py --config_file configs/streetforward/multi_scene.yaml data.data_root=/path/to/nuscenes

# 临时开启严格模式
python tools/train_streetforward.py --config_file configs/streetforward/multi_scene.yaml training.strict_proxy_grad=true

# 限制体积以 fail-fast
python tools/train_streetforward.py --config_file configs/streetforward/multi_scene.yaml training.sentinel.max_dense_elements=5000000000
```

**新增行为**：

- 空 targets 或空点云会 raise `ValueError`，不再静默返回零损失
- 每 `log_interval` 步在控制台输出 `metric_logger`（含 loss、lr、grad_norm）
- TensorBoard 中除 `train/total_loss`、`train/lr`、`train/grad_norm` 外，还有 `sentinel/*` 系列标量

### 3.2 训练前扫雷：preflight_sweep_streetforward.py

在正式长跑前，用 preflight 做一次前向扫雷，检查体积、mask、target 是否有异常：

```bash
python tools/preflight_sweep_streetforward.py --config_file configs/streetforward/multi_scene.yaml
```

**常用参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max_batches` | 128 | 扫雷 batch 数量 |
| `--max_dense_elements` | None | 硬上限，超过则标记 alert |
| `--export_path` | `<log_dir>/preflight_report.json` | 报告输出路径 |
| `--log_interval` | 10 | 每隔多少 batch 打印进度 |

**示例**：

```bash
# 扫 500 个 batch
python tools/preflight_sweep_streetforward.py --config_file configs/streetforward/multi_scene.yaml --max_batches 500

# 设定体积上限并输出到指定路径
python tools/preflight_sweep_streetforward.py --config_file configs/streetforward/multi_scene.yaml \
  --max_dense_elements 5000000000 --export_path ./preflight_report.json
```

**输出**：

- 控制台：每 `log_interval` 打印 `N_bg`、`N_rigid`、`vol_dim_prod`、`alerts`
- JSON 报告：`records`（每 batch 统计）、`summary`（alert/error 数量、最大体积等）

**Alert 类型**：

- `no_targets`：该 batch 无 target
- `rigid_gate_zero`：有 rigid 点但 `mask_update_rigid` 全为 0
- `dense_elements_gt_{N}`：dense 元素数超过 `max_dense_elements`

### 3.3 单元测试

```bash
python -m pytest tests/test_streetforward_logging.py -v
```

包含测试：

- `test_strict_proxy_grad_raises_on_none`：strict 模式下 Proxy grad None 会 raise
- `test_sentinel_metrics_cover_masks_and_volume`：哨兵覆盖 mask、体积等指标
- `test_record_volume_stats_respects_limit`：体积超限会 raise

---

## 四、TensorBoard 哨兵指标

启用 `sentinel.enabled: true` 后，每 `sentinel.log_every` 步写入以下标量（前缀 `sentinel/`）：

### 4.1 数据/分支

| 指标 | 说明 |
|------|------|
| `num_targets` | target 数量 |
| `N_bg`, `N_rigid`, `N_distant` | 各类点数 |
| `mask_update_rigid_mean` | rigid 可更新比例 |
| `mask_src_rigid_mean` | rigid 在 source 可见比例 |
| `idx_tgt_rigid_mean`, `idx_tgt_rigid_max` | 各 target 可见 rigid 数统计 |

### 4.2 体积/显存

| 指标 | 说明 |
|------|------|
| `vol_dim_prod` | prod(vol_dim) |
| `dense_elements_est` | 估算的 dense 元素数 |
| `max_memory_allocated_gb` | 峰值显存（GB） |

### 4.3 梯度

| 指标 | 说明 |
|------|------|
| `grad_norm_total` | 总梯度范数 |
| `grad_{module}` | 各模块 grad norm（sparse_conv、mlp_*、gru_* 等） |
| `proxy_grad_{name}` | 各 proxy 参数 grad norm（bg.means、rigid.quats 等） |

### 4.4 数值健康

| 指标 | 说明 |
|------|------|
| `bg_opacities_min/max` | 背景 opacities 范围 |
| `bg_quat_norm_dev` | 四元数模长偏离 1 的均值 |
| `bg_means_min/max` | 背景 means 范围 |
| `bg_scales_log_min/max` | 背景 scales_log 范围 |
| `bg_offset_pos_max` | 背景 offset_pos 最大值 |
| `rigid_*`, `distant_*` | 对应 rigid/distant 的同类指标 |

---

## 五、推荐训练流程

按 `StreetForward_Logging_System_Update_Plan.md` 的流程：

1. **Preflight sweep**：先跑 preflight，确认无大量 `no_targets`、`rigid_gate_zero`、体积异常
2. **Tiny overfit**：用 1 个 scene/segment 固定视角，跑几百步，确认 loss 明显下降
3. **Canary run**：开严格模式，随机 20~50 batch 跑 1~2 小时，观察哨兵与异常
4. **Full run**：关闭 anomaly、将 strict_proxy_grad 降为告警，保留哨兵做长期监控

---

## 六、与现有文档的衔接

- **StreetForward_Flow.md**：流程逻辑未变，只是在关键节点插桩记录哨兵
- **StreetForward_Formal_Training_Gap_Analysis.md**：本更新对应 2.5 节可观测性与调试缺口
- **StreetForward_Logging_System_Update_Plan.md**：设计文档，含长尾隐患与测试清单
- **StreetForward_Staged_Changes_Review.md**：反直觉审查，已修复 permute、max_dense_elements 等问题
