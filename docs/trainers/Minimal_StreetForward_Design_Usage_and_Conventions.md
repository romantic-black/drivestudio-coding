Minimal StreetForward 设计、用法与规范
===================================

本档面向 Minimal StreetForward 系列（Stage 1 / 1.1 / 2.0 / 2.1 / 2.2 / 3_2d），总结：

- **整体设计与各 Stage 职责**
- **训练脚本与使用方式**
- **日志、profiling 与 GS 统计规范**
- **实验记录与 Vika 上报规范（配合 vika.py 与 `configs/vika_minimal_sf.yaml`）**

语言以中文为主，代码与标识符保持英文。

---

### 一、设计概览：Minimal StreetForward 各 Stage 职责

Minimal 版本沿着 StreetForward 的完整设计拆成若干 Stage，循序渐进地增加功能与复杂度：

- **Stage 1 (`minimal_trainer_stage1.py`)**
  - 只包含 `NodeStateBackground`，使用点云背景初始化高斯点。
  - 单视角目标（single target），无 2D 分支、无远处节点。
  - 结构：`NodeStateBackground -> 3D sparse conv 特征 -> offsets head -> render_params -> 渲染 -> L1 损失`。

- **Stage 1.1 (`minimal_trainer_stage1_1.py`)**
  - 在 Stage 1 基础上，引入 GRU 风格 offset 预测：
    - 将 Node 参数（means/scales/quats/opacity/sh）编码为向量，与 3D 特征拼接喂入 GRU。
    - 维护 `h_cache_bg` 实现跨 step 的记忆。
  - 仍然是单目标单视角，主要验证 GRU + NodeState 的训练行为。

- **Stage 2.0 (`minimal_trainer_stage2_0.py`)**
  - Multi-target，无 proxy。
  - 一个 `render_params` 对所有目标视角共享，逐视角渲染，`loss = mean(loss_i)`，单次 backward。
  - 解决“多视角共享同一套高斯”的最小实现。

- **Stage 2.1 (`minimal_trainer_stage2_1.py`)**
  - Multi-target + proxy 梯度累积：
    - 对 `render_params` 创建可梯度 proxy tensors。
    - 每个视角单独 `loss_i.backward()`，在 proxy 上累积梯度。
    - 再通过 `_backward_to_render_params_bg` 把梯度推回真实 `render_params`。
  - 与 2.0 行为对齐，用于验证 proxy 方案的稳定性。

- **Stage 2.2 (`minimal_trainer_stage2_2.py`)**
  - Multi-target，无 proxy。
  - 在所有 target 尺寸一致时，通过 `_render_multi_view` 一次 gsplat 渲染所有视角，降低渲染开销。
  - 逻辑上仍然是 “shared render_params + mean loss”。

- **Stage 3 2D 分支 (`minimal_trainer_stage3_2d.py`)**
  - 在 Stage 2.1 上增加：
    - 2D 特征抽取与反投影（`ImageFeatureExtractor` + `AlphaTWeightExtractor` + `FeatureBackprojector` + `FeatureFusion`）。
    - 点云以输入 AABB 切分为 `NodeStateBackground` + `NodeStateDistant`，并合并渲染。
  - 训练时需要 source 视角与图像（通过 `convert_batch_to_minimal_format(..., include_source_for_2d=True)` 提供）。

整体上：

- **Stage 编号越大，功能越完整，越接近完整 StreetForward Flow**。
- Minimal 版本保持尽量少的组件，方便 debug、profiling 和实验记录。

---

### 二、训练脚本与基本用法

每个 Stage 对应一个 `tools/train_minimal_streetforward_stageX_Y.py` 脚本，均采用“单 batch 过拟合”的训练模式（`overfit_one_batch` 生成 `.pt` batch）。

通用参数（各脚本基本一致）：

- `--config_file`: OmegaConf 配置路径（如 `configs/minimal_streetforward_stage2_2.yaml`）。
- `--output_root`: 输出根目录，默认 `outputs`。
- `--project`: 项目名，默认 `minimal_sf`，用于拼接 `log_dir`。
- `--run_name`: run 名称，用作 `log_dir` 子目录。
- `--overfit_batch_path`: `.pt` batch 路径（必填，否则 fast-fail）。
- `--max_steps`: 覆盖 `cfg.training.max_iterations`。
- `--seed`: 随机种子。
- `opts`: 额外的 `key=value` 覆盖配置项。

示例（Stage 3_2d）：

```bash
python tools/train_minimal_streetforward_stage3_2d.py \
  --config_file configs/minimal_streetforward_stage3_2d.yaml \
  --run_name minimal_sf_stage3_2d_v1 \
  --overfit_batch_path data/overfit_batches/scene0_seg0_batch.pt \
  training.max_iterations=1000 training.log_interval=50
```

训练脚本内部通用流程：

1. `setup(args)`：
   - 加载 config，合并 CLI `opts`，设置 `cfg.log_dir`。
   - 创建 `images/`、`checkpoints/`、`metrics_history.jsonl` 等子目录。
   - 打印并存档最终配置（`config.yaml`）。
2. 加载 overfit batch，并通过 `convert_batch_to_minimal_format` 适配为 minimal 结构：
   - Stage 1 / 1.1：单 target。
   - Stage 2.x：多 target；Stage 3：多 target + source_views/source_images。
3. 构建对应 `MinimalStreetForwardStage*` 模型，搬到 `device`。
4. 初始化指标模块（PSNR / SSIM / LPIPS）和训练超参（步数、log / save / metric 周期等）。
5. **训练循环**：
   - 调用 `model.train_step(minimal_batch, step=step)`。
   - 记录 loss、PSNR/SSIM/LPIPS、保存图像与 checkpoint。
   - 统计 profiling、GS 点数等信息（见下一节）。
6. 结束时 optional test（若 batch 中包含 test views），写入 `metrics_final.json`。
7. 若配置了 Vika（见第 4 节），自动上传一次 run 的 summary 到 Vika。

---

### 三、日志、profiling 与 GS 统计规范

#### 3.1 Step 级 profiling

在 `train_minimal_streetforward_stage1_1.py` 中，训练循环已加入：

- 每个 step：
  - `step_start_wall = time.time()` / `step_end_wall = time.time()`。
  - 可用 GPU 时：`torch.cuda.reset_peak_memory_stats()` + `max_memory_allocated` / `max_memory_reserved`。
- 训练结束后：
  - 计算 `avg_step_time_ms`（所有 step 平均耗时）。
  - 记录全程 `peak_mem_bytes` 与 `peak_mem_reserved_bytes`。
  - 写入 `metrics_final.json` 的 `profiling` 字段。

其它 Stage 可按相同模式拓展（目前主要对 Stage 1.1 做了标准实现）。

#### 3.2 GS 点数与多视角统计

在各个 trainer 的 `train_step` 中返回了 GS 点数与视图统计，用于上层脚本汇总：

- Stage 1 / 1.1：
  - `num_gaussians_bg`: `int(node_state_bg.means.shape[0])`
- Stage 2.0 / 2.1：
  - `num_gaussians_bg`
  - `num_targets`: `len(batch["targets"])`
- Stage 3_2d：
  - `num_gaussians_bg`
  - `num_gaussians_distant`
  - `num_targets`
  - `num_source_views`

对应训练脚本中，会维护：

- `sum_num_gaussians_bg`、`sum_num_gaussians_distant`（Stage 3）、`sum_num_targets`、`sum_num_source_views` 与 `total_steps`。
- 在最终写入 `metrics_final.json` 时写入 `gs_stats`：
  - `avg_num_gaussians_bg`
  - `avg_num_gaussians_distant`（Stage 3）
  - `avg_num_targets_per_batch`
  - `avg_num_source_views`（Stage 3）

这些字段为后续实验分析与 Vika 上报提供统一的来源。

---

### 四、Vika 配置与上传规范

#### 4.1 安装与配置 vika.py

- 安装（已在当前环境中执行过）：

```bash
pip install --upgrade vika
```

- 配置文件：`configs/vika_minimal_sf.yaml`，示例内容：

```yaml
vika:
  enabled: true
  token_env: "VIKA_TOKEN"
  datasheet_id: "REPLACE_WITH_YOUR_DATASHEET_ID"
  view_id: null
  field_mapping: {}
```

建议做法：

- **Token 一律通过环境变量提供**，不直接写入配置文件：

```bash
export VIKA_TOKEN="your_api_token"
export VIKA_DATASHEET_ID="dstt3KGCKtp11fgK0t"
```

- 在配置文件中仅记录：
  - `enabled`：是否启用 Vika 上传。
  - `token_env`：读取 token 的环境变量名。
  - `datasheet_id`：目标表 id（占位符由使用者自行替换）。
  - `field_mapping`：若 Vika 表头与本地字段名不同，可在此做映射（当前实现未强制使用）。

#### 4.2 上传 helper：`tools/upload_to_vika.py`

实现要点：

- 使用 [`vika.py` 官方 SDK](https://github.com/vikadata/vika.py)，以 `Vika(token)` 实例化客户端。
- `_load_vika_config()`：
  - 从 `configs/vika_minimal_sf.yaml` 读取配置。
  - 若未存在或 `vika.enabled` 为 false，则跳过上传。
- `upload_experiment_summary(log_dir, summary_fields=None)`：
  - 若未传 `summary_fields`，则从 `log_dir/metrics_final.json` 读入 summary。
  - 从配置与环境变量中获取 token 与 `datasheet_id`：
    - 优先：`configs/vika_minimal_sf.yaml` 中的 `token_env` 与 `datasheet_id`。
    - 退化：环境变量 `VIKA_TOKEN` 和 `VIKA_DATASHEET_ID`。
  - 调用：

```python
client = Vika(token)
dst = client.datasheet(datasheet_id)
record = dst.records.create(fields)
```

  - Fast-fail 策略：
    - 未安装 `vika.py` 或缺失 token / datasheet_id：只打印 warning，不抛异常。
    - Vika API 运行时异常在训练脚本中捕获并做 `logger.exception`。

#### 4.3 训练脚本中的自动上传调用

在以下脚本末尾写 `metrics_final.json` 时，都已经集成了自动上传：

- `tools/train_minimal_streetforward_stage1_1.py`
- `tools/train_minimal_streetforward_stage2_0.py`
- `tools/train_minimal_streetforward_stage2_1.py`
- `tools/train_minimal_streetforward_stage2_2.py`
- `tools/train_minimal_streetforward_stage3_2d.py`

模式类似（以 2.0 为例）：

```python
summary = {
    "final_step": int(max_iterations - 1),
    "train": {"loss_l1": float(result["loss"])},
    "test": test_metrics,
    "gs_stats": {
        "avg_num_gaussians_bg": avg_num_gaussians_bg,
        "avg_num_targets_per_batch": avg_num_targets,
    },
}
metrics_final_path = os.path.join(cfg.log_dir, "metrics_final.json")
with open(metrics_final_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

try:
    upload_experiment_summary(cfg.log_dir, summary)
except Exception:
    logger.exception("Vika upload failed for log_dir=%s", cfg.log_dir)
```

也就是说：

- **每次训练 run 结束时**，会：
  - 将 summary 写入本地 JSON；
  - 尝试向 Vika 对应 datasheet 创建一条记录（字段即 summary dict 的键）。

如需严格对齐文档中的 Vika 表头，可以在 `summary` 结构中进一步展开字段名（例如添加 `experiment_id`、`run_name`、`code_version` 等），并在 `field_mapping` 中配置好映射。

---

### 五、编码与实验规范总结

- **代码与配置**
  - 代码与标识符统一使用英文，文档可以使用中文。
  - 遵循 fast-fail：关键字段缺失时尽早抛错；Vika 上传失败仅记录日志，不影响训练主流程。
- **Profiling 与统计**
  - 训练脚本负责 step 级 profiling、显存与 GS 点数的聚合。
  - Trainer 只返回必要的元数据，避免携带大张量回上层。
- **实验记录**
  - 每个训练 run 至少保证：
    - `metrics_history.jsonl`：逐 step 指标；
    - `metrics_final.json`：最终指标 + profiling + GS 聚合；
    - `config.yaml`：完整配置；
    - 可选 Vika 记录：由 `upload_experiment_summary` 创建。
- **Vika 集成**
  - 所有敏感信息（token）仅通过环境变量注入；
  - `configs/vika_minimal_sf.yaml` 只承载非敏感配置（开关 / datasheet id 等）；
  - `upload_to_vika.py` 集中处理上传逻辑，训练脚本只负责构造 summary 并调用。

这些约定使得 Minimal StreetForward 在保持结构简洁的同时，具备了良好的可观测性（时间 / 显存 / GS 数量）和可复现实验记录（本地 JSON + Vika 云端表）。在后续扩展完整 StreetForward Flow 时，可以直接沿用这套规范。 

