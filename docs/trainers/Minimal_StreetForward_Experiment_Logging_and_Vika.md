Minimal StreetForward 实验监控与 Vika 数据上报设计
========================================

本档基于 `minimal_trainer_stage1.py`、`minimal_trainer_stage1_1.py`、`minimal_trainer_stage2_0.py`、`minimal_trainer_stage2_1.py`、`minimal_trainer_stage2_2.py`、`minimal_trainer_stage3_2d.py` 的现有实现，规划：

- **训练过程中需要额外监控的关键信息（step 耗时、显存、GS 点数等）**
- **统一的实验数据（超参、结果、指标等）收集规范**
- **将实验结果上报到 Vika 云端表的字段设计与数据格式**

后续在实现时应遵循“fast-fail、非必要不加默认值”的原则：字段缺失或格式错误时尽早报错，而不是静默填默认值。

---

### 一、训练过程监控：step 耗时 / 显存 / GS 点数

#### 1.1 监控目标与粒度

监控分为两层：

- **Step 级别（核心）**：每个训练 step 的耗时和整体显存峰值，用于判断整体是否可训练、是否 OOM、是否需要减小 batch / GS 点数。
- **阶段 / 模块级别（建议）**：对一个 step 内的若干关键阶段做粗粒度拆分和计时，以便定位计算热点。

##### Step 级别建议监控字段

- **step_index**：当前全局 step 编号（int）
- **step_wall_time_ms**：单个 step 的总 wall-clock 时间（float, 毫秒）
- **step_data_time_ms**（可选）：data loader / batch 准备时间（float）
- **step_forward_time_ms**：`model.forward` 耗时（float）
- **step_backward_time_ms**：`loss.backward` + 反向传播耗时（float）
- **step_optimizer_time_ms**：`optimizer.step` 耗时（float）
- **step_misc_time_ms**（可选）：其它操作（日志、可视化、状态更新等）耗时（float）
- **step_peak_mem_bytes**：本 step 内 GPU 显存峰值（int）
- **step_peak_mem_reserved_bytes**：本 step 内 GPU 预留显存峰值（int）

这些字段可以在训练 loop 里通过 `time.time()` + `torch.cuda.max_memory_allocated()` / `torch.cuda.reset_peak_memory_stats()` 获取。

##### 阶段 / 模块级别（可选）监控字段

针对 Minimal StreetForward 的具体结构，可重点关注：

- **3D 特征卷积部分**（`_build_3d_features` / `self.sparse_conv`）：
  - `time_feat3d_ms`
  - `mem_feat3d_peak_bytes`
- **GRU + offsets head**：
  - `time_gru_head_ms`
  - `mem_gru_head_peak_bytes`
- **Rasterization / 渲染**（gsplat）：
  - `time_render_ms`
  - `mem_render_peak_bytes`
- **Stage 3 中 2D 分支相关模块**：
  - Image encoder: `time_image_feature_extractor_ms`
  - AlphaT/backproject: `time_backproject_ms`

这些监控不必每 step 都记录到日志或上传 Vika，可以：

- 在代码中实现统一的 `ProfilerContext`（例如基于 `time.time()` 与 `torch.cuda.max_memory_allocated()`），返回结构化结果；
- 在训练脚本中按一定 `log_interval` 将这些信息聚合成平均值，再写到日志或 Vika。

---

#### 1.2 显存统计的实现要点

在训练 loop 中，推荐的基本模式：

```python
start_time = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# data_time 视需求拆分

step_t0 = time.time()
out = model.train_step(batch, step=global_step)
step_t1 = time.time()

step_wall_time_ms = (step_t1 - start_time) * 1000.0
step_forward_backward_optimizer_ms = (step_t1 - step_t0) * 1000.0

if torch.cuda.is_available():
    step_peak_mem_bytes = torch.cuda.max_memory_allocated()
    step_peak_mem_reserved_bytes = torch.cuda.max_memory_reserved()
else:
    step_peak_mem_bytes = None
    step_peak_mem_reserved_bytes = None
```

更细粒度的模块监控，可以在 trainer 内部使用小的上下文管理器包装关键函数调用，例如 `_build_3d_features`、`_render_single_view`、Stage 3 的 2D 分支等：

```python
with profiler_context("feat3d"):
    feat_3d_crop = self._build_3d_features(means, anchor_rgb)
```

`profiler_context` 内部可以将时间 / 显存记录到 `self._last_profile` 或返回到 `train_step` 结果里，再由外层训练脚本做汇总。

---

#### 1.3 各 Node/GS 点数统计

Minimal StreetForward 当前有以下主要 NodeState：

- Stage 1 / 1.1 / 2.x：`NodeStateBackground`
- Stage 3：`NodeStateBackground` + `NodeStateDistant`

需要监控的关键数量：

- **num_gaussians_bg**：`node_state_bg.means.shape[0]`
- **num_gaussians_distant**（Stage 3）：若存在 `node_state_distant`，则为其 `means.shape[0]`，否则为 0
- **num_targets_per_batch**：`len(batch["targets"])`
- **num_source_views**（Stage 3）：`len(batch["source_views"])`

建议在各个 Stage 的 `train_step` 返回字典中附带这些统计，例如：

```python
return {
    "loss": loss.item(),
    "pred_rgbs": ...,
    "gt_images": ...,
    "num_gaussians_bg": int(node_state_bg.means.shape[0]),
    "num_gaussians_distant": int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0,
    "num_targets": len(batch["targets"]),
}
```

训练脚本在拿到 `train_step` 的输出后，可以配合 step 级别的耗时和显存，写入 Vika。

---

### 二、实验数据收集设计（超参 / 结果 / 指标）

#### 2.1 实验单元定义

Vika 中推荐的最小“实验单元”是 **一次完整的训练 run**，对应一次命令行启动，例如：

```bash
python tools/train_minimal_streetforward_stage3_2d.py \
  configs/minimal_streetforward_stage3_2d.yaml \
  --experiment_name minimal_sf_stage3_2d_v1
```

**一条 Vika 表记录 = 一个 experiment run**，每条记录包含：

- 本次 run 的配置（代码版本、配置文件、关键超参）；
- 训练过程的汇总结果（例如最优/最终指标、平均 step 耗时、显存峰值等）；
- 部分详细结构（例如 per-stage 的 GA 点数、分模块耗时等，视表头设计可以以 JSON 文本存储）。

---

#### 2.2 建议收集的数据类别

1. **标识信息**
   - `experiment_id`：短字符串，全局唯一（可由训练脚本生成 UUID 或基于时间戳）
   - `run_name`：人为可读名称（如 `minimal_sf_stage3_2d_v1`）
   - `stage`：训练使用的 Stage（例如 `"1"`, `"1_1"`, `"2_0"`, `"2_1"`, `"2_2"`, `"3_2d"`）
   - `trainer_class`：例如 `"MinimalStreetForwardStage3_2d"`
   - `config_path`：本次使用的配置路径（如 `configs/minimal_streetforward_stage3_2d.yaml`）
   - `code_version`：Git commit hash（通过 `git rev-parse HEAD` 获取）

2. **数据集与场景信息**
   - `dataset_name`：数据集名称 / 配置中的 `config.dataset.name`
   - `scene_id_list`：参与训练的 scene 列表（可为 JSON 字符串）
   - `segment_aabb`：`config.dataset.segment_aabb`（JSON）
   - `segment_input_aabb`（Stage 3）：`config.dataset.segment_input_aabb`（JSON）

3. **关键超参（来自 config）**
   - 优化器相关：
     - `optimizer_type`：如 `"Adam"`
     - `optimizer_lr`
     - `optimizer_eps`
     - `optimizer_weight_decay`
   - 模型结构相关（`config.model` 中）：
     - `sh_degree`
     - `voxel_size`
     - `sparseConv_outdim`
     - `param_embed_dim`（Stage 1.1 / 2.x / 3）
     - `offset_gru_hidden_dim`（Stage 1.1 / 2.x / 3）
     - `offset_gru_use_reset_gate`（bool）
     - `feat_2d_channels`（Stage 3）
     - `feat_2d_downscale`（Stage 3）
   - NodeState / offsets 相关：
     - `offset_max`, `scale_max`, `omega_max`, `opacity_max`, `sh_dc_max`, `sh_rest_max`
     - `eta_means`, `eta_scales`, `eta_opacity`, `eta_sh_dc`, `eta_sh_rest`
     - `update_node_state_interval`
   - 训练流程相关（通常在 config 或 CLI 中）：
     - `max_steps`
     - `log_interval`
     - `save_interval`
     - `eval_interval`
     - `batch_size`（如脚本中有）

   如果字段过多，可以在 Vika 中将“完整 config”以 JSON long text 的形式存入一个字段，例如 `config_json`，同时再将上述几个关键超参单独展开成列。

4. **训练过程统计（汇总）**
   - Step 统计（可以取平均/中位/最小等）：
     - `avg_step_time_ms`
     - `p50_step_time_ms`（可选）
     - `p95_step_time_ms`（可选）
     - `avg_forward_time_ms`
     - `avg_backward_time_ms`
     - `avg_optimizer_time_ms`
   - 显存统计：
     - `peak_mem_bytes`：整个训练过程中观测到的最大 `max_memory_allocated`
     - `peak_mem_reserved_bytes`
   - GS 点数统计：
     - `avg_num_gaussians_bg`
     - `avg_num_gaussians_distant`（Stage 3）
     - `avg_num_targets_per_batch`
     - `avg_num_source_views`（Stage 3）

   这些值可以在训练脚本中维护滑动平均或全局统计，在训练结束时一次性写入 Vika。

5. **指标与结果**

根据 Minimal StreetForward 的训练脚本，通常会在 validation/test 时计算：

- `best_train_loss` / `final_train_loss`
- `best_val_loss` / `final_val_loss`
- 如果有重建指标：
  - `val_psnr_mean`
  - `val_ssim_mean`
  - `val_lpips_mean`

若指标尚未完全实现，可先预留字段，暂时写入 `null` 或不写入（视 Vika 字段是否必填而定，建议 Vika 端先设为非必填）。

6. **元信息与备注**

- `start_time_utc` / `end_time_utc`（ISO8601 字符串）
- `total_wall_time_minutes`
- `host_name`
- `cuda_version`
- `pytorch_version`
- `gsplat_version`
- `notes`：文本备注（训练中观察到的问题、肉眼观感、TODO 等）

---

#### 2.3 实验记录示例（Vika 表头草案）

下表是推荐的 Vika 表头（可以根据实际使用情况增删）：

| 字段名                     | 类型        | 说明                                                                 |
|--------------------------|-----------|----------------------------------------------------------------------|
| experiment_id            | 文本      | 本次 run 的唯一 ID（例如 UUID）                                      |
| run_name                 | 文本      | 人类可读名称，如 `minimal_sf_stage3_2d_v1`                          |
| stage                    | 文本      | `"1"`, `"1_1"`, `"2_0"`, `"2_1"`, `"2_2"`, `"3_2d"` 等              |
| trainer_class            | 文本      | 使用的 trainer 类名                                                 |
| config_path              | 文本      | 配置文件路径                                                        |
| code_version             | 文本      | Git commit hash                                                     |
| dataset_name             | 文本      | 数据集名称                                                          |
| scene_id_list            | 长文本    | JSON 数组，列出所有 scene_id                                        |
| segment_aabb             | 长文本    | `[[min_x, min_y, min_z],[max_x, max_y, max_z]]` JSON               |
| segment_input_aabb       | 长文本    | 仅 Stage 3 使用                                                     |
| optimizer_type           | 文本      | 优化器名称                                                          |
| optimizer_lr             | 数字      | 学习率                                                              |
| optimizer_eps            | 数字      | eps                                                                 |
| optimizer_weight_decay   | 数字      | 权重衰减                                                            |
| sh_degree                | 数字      | SH 阶数                                                             |
| voxel_size               | 数字      | 体素大小                                                            |
| sparseConv_outdim        | 数字      | 3D 特征通道数                                                       |
| param_embed_dim          | 数字      | 参数 embedding 维度（如有）                                         |
| offset_gru_hidden_dim    | 数字      | GRU 隐藏维度                                                        |
| offset_gru_use_reset_gate| 复选框    | 是否使用 reset gate                                                 |
| feat_2d_channels         | 数字      | Stage 3 的 2D 特征通道数                                            |
| feat_2d_downscale        | 数字      | Stage 3 的 2D 特征下采样倍数                                       |
| offset_max               | 数字      | 位置 offset 上限                                                    |
| scale_max                | 数字      | scale offset 上限                                                   |
| omega_max                | 数字      | 旋转 offset 上限                                                    |
| opacity_max              | 数字      | 不透明度 offset 上限                                                |
| sh_dc_max                | 数字      | SH DC offset 上限                                                   |
| sh_rest_max              | 数字      | SH 其余系数 offset 上限                                             |
| eta_means                | 数字      | NodeState means 更新系数                                            |
| eta_scales               | 数字      | NodeState scales 更新系数                                           |
| eta_opacity              | 数字      | NodeState opacity 更新系数                                          |
| eta_sh_dc                | 数字      | NodeState sh_dc 更新系数                                            |
| eta_sh_rest              | 数字      | NodeState sh_rest 更新系数                                          |
| update_node_state_interval | 数字    | NodeState 写回间隔                                                  |
| max_steps                | 数字      | 最大训练步数                                                        |
| batch_size               | 数字      | batch 大小（如有）                                                  |
| avg_step_time_ms         | 数字      | 训练过程中 step 耗时平均值                                          |
| p50_step_time_ms         | 数字      | 中位数 step 耗时（可选）                                            |
| p95_step_time_ms         | 数字      | 95 分位 step 耗时（可选）                                           |
| avg_forward_time_ms      | 数字      | 平均 forward 耗时                                                   |
| avg_backward_time_ms     | 数字      | 平均 backward 耗时                                                  |
| avg_optimizer_time_ms    | 数字      | 平均 optimizer.step 耗时                                            |
| peak_mem_bytes           | 数字      | 整个训练过程的显存峰值（字节）                                      |
| peak_mem_reserved_bytes  | 数字      | 预留显存峰值（字节）                                                |
| avg_num_gaussians_bg     | 数字      | 平均背景 GS 点数                                                    |
| avg_num_gaussians_distant| 数字      | 平均远处 GS 点数（Stage 3）                                         |
| avg_num_targets_per_batch| 数字      | 平均每 batch 渲染的 target 数                                       |
| avg_num_source_views     | 数字      | 平均每 batch 源视角数（Stage 3）                                    |
| best_train_loss          | 数字      | 训练过程中最优的 train loss                                         |
| final_train_loss         | 数字      | 训练结束时的 train loss                                             |
| best_val_loss            | 数字      | 最优 val loss（如有验证）                                           |
| final_val_loss           | 数字      | 最终 val loss（如有验证）                                           |
| val_psnr_mean            | 数字      | 验证集 PSNR 均值（如有）                                            |
| val_ssim_mean            | 数字      | 验证集 SSIM 均值（如有）                                            |
| val_lpips_mean           | 数字      | 验证集 LPIPS 均值（如有）                                           |
| start_time_utc           | 日期时间  | 训练开始时间                                                        |
| end_time_utc             | 日期时间  | 训练结束时间                                                        |
| total_wall_time_minutes  | 数字      | 训练总时长（分钟）                                                  |
| host_name                | 文本      | 机器名                                                               |
| cuda_version             | 文本      | CUDA 版本                                                            |
| pytorch_version          | 文本      | PyTorch 版本                                                         |
| gsplat_version           | 文本      | gsplat 版本                                                          |
| config_json              | 长文本    | 完整 OmegaConf 配置的 JSON dump                                     |
| extra_profile_json       | 长文本    | 可选：存放更详细的分模块耗时 / 显存 JSON                            |
| notes                    | 长文本    | 人工备注                                                             |

---

### 三、Vika 上报方式设计（基于 vika.py）

本节基于官方 Python SDK [vika.py](https://github.com/vikadata/vika.py) 设计上报方式，直接在训练脚本或独立 Python 工具中调用 Vika API，不再依赖 vika.js。

#### 3.1 vika.py 基本用法回顾

参考 `vika.py` 仓库的示例，用法大致如下（略去错误处理） [`vika.py` 文档](https://github.com/vikadata/vika.py)：

```python
from vika import Vika
import os

client = Vika(os.environ["VIKA_TOKEN"])
dst = client.datasheet("datasheetId")  # 表格 ID 或 URL

def create_experiment_record(fields: dict) -> str:
    # 单条创建：fields 的 key 对应表头字段名
    record = dst.records.create(fields)
    # fast-fail：vika.py 出错会抛异常，直接让上层失败
    return record._id  # recordId
```

批量创建时可以使用 `bulk_create`：

```python
records = dst.records.bulk_create([
    {
        "experiment_id": "exp_20260316_120001_stage3_2d",
        "run_name": "minimal_sf_stage3_2d_debug",
        "stage": "3_2d",
        "trainer_class": "MinimalStreetForwardStage3_2d",
        "config_path": "configs/minimal_streetforward_stage3_2d.yaml",
        "code_version": "abcdef1234567890",
        "optimizer_lr": 5e-4,
        "avg_step_time_ms": 120.3,
        "peak_mem_bytes": 10485760000,
        "best_val_loss": 0.023,
        "notes": "first run on scene xxxx, looks good",
    },
])
```

在训练脚本中，可以在训练结束后构造一个 `fields` 字典（见下文 3.2 的 JSON 结构），直接传给 `dst.records.create(fields)` 或 `bulk_create`。

#### 3.2 Python 训练结果到 Vika 的数据结构

为简化实现，可以先在训练结束时构造一个 Python dict，并可选地 dump 成 JSON 以便调试；其 schema 与 Vika 字段一一对应。例如：

```json
{
  "experiment_id": "exp_20260316_120001_stage3_2d",
  "run_name": "minimal_sf_stage3_2d_debug",
  "stage": "3_2d",
  "trainer_class": "MinimalStreetForwardStage3_2d",
  "config_path": "configs/minimal_streetforward_stage3_2d.yaml",
  "code_version": "abcdef1234567890",
  "dataset_name": "xxx_dataset",
  "scene_id_list": [0, 1, 2],
  "segment_aabb": [[-1, -1, -1], [1, 1, 1]],
  "optimizer_type": "Adam",
  "optimizer_lr": 0.0005,
  "sh_degree": 1,
  "voxel_size": 0.1,
  "avg_step_time_ms": 120.3,
  "peak_mem_bytes": 10485760000,
  "best_train_loss": 0.02,
  "final_train_loss": 0.025,
  "start_time_utc": "2026-03-16T04:00:00Z",
  "end_time_utc": "2026-03-16T04:30:00Z",
  "total_wall_time_minutes": 30.0,
  "config_json": "{...原始 OmegaConf JSON...}",
  "extra_profile_json": "{...更详细的 profile 数据...}",
  "notes": "first stage3_2d run"
}
```

在 Python 中，直接把这个 dict 作为 `fields` 传给 vika.py：

```python
from vika import Vika

client = Vika(VIKA_TOKEN)
dst = client.datasheet("datasheetId")

with open("experiment_summary.json", "r") as f:
    fields = json.load(f)

dst.records.create(fields)
```

如果希望完全避免中间 JSON 文件，也可以在训练脚本中直接构造 `fields` dict（而不是写盘），然后调用 `dst.records.create(fields)`。

---

### 四、小结与后续工作

- **监控层面**：在训练 loop 中加入 step 耗时与显存统计，在各 Stage 的 `train_step` 输出中添加 `num_gaussians_*` 等信息；可选地对 3D 特征、渲染、2D 分支等模块做更细粒度 profiling。
- **数据侧**：以“实验 run”为单位设计 Vika 表头，覆盖标识、数据集、关键超参、训练过程统计、重建指标与元信息；冗长或结构化数据以 JSON 文本字段保存。
- **对接侧**：训练脚本内部或独立 Python 工具基于 vika.py 直接调用 Vika API，将 `fields` dict 写入表格，保持 fast-fail 策略。

后续可以在 `tools/train_minimal_streetforward_stage*_*.py` 中增加：

- 统一的 profile/metrics 聚合工具类；
- 将最终统计结果 dump 成与本设计兼容的 JSON；
- 视需要增加自动推送 Vika 的脚本或 CI 步骤。

