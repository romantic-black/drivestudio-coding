# StreetForward Stage 4.3 多场景 V4：正式测试（eval）实现方案

本文档在训练管线已基本跑通的前提下，讨论**测试 / 评估**部分的落地设计。依据：

- 数据与预加载：`datasets/multi_scene_dataset_v3.py`、`datasets/dataset_preload_manager.py`、`datasets/multi_scene_dataset.py`
- 训练入口与配置：`tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py`、`configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml`

---

## 1. 背景：为什么训练主脚本不适合承担「正式测试」

### 1.1 训练脚本的硬约束

`train_minimal_streetforward_stage4_3_multi_scene_v4.py` 当前：

- 要求 `len(data.train_scene_ids) >= 2`（多场景训练假设）。
- 要求 `scheduler_v4.traversal.fixed_scene_id` / `fixed_segment_id` 均为空（禁止单场景单段钉死遍历）。
- 保存的是 **PyTorch checkpoint**（`model_state_dict` + `optimizer_state_dict`），不是可独立复用的 **3D Gaussian Splatting（3DGS）场景状态**。

### 1.2 尾部 `run_test_at_end` 的局限

配置项 `eval.run_test_at_end` 在通用 `setup()` 中默认可为 `true`，但多场景 v4 配置中建议为 `false`。训练脚本**不再**将「最后一个 batch 的 test_views」作为正式 eval；若需完整测试请使用 [`tools/test_minimal_streetforward_stage4_3.py`](tools/test_minimal_streetforward_stage4_3.py)。历史上在训练循环**结束后**若最后一个 `minimal_batch` 带有 `test_views` 做附带评估的流程：

- 只覆盖**最后一个 batch** 的 `test_views`，不是完整 eval suite。
- 仍通过 `model.forward(minimal_batch)`，依赖既有 batch 构图（含 `targets` 等训练语义）。

因此不适合作为「单测试场景 / 全 segment 正式汇报」的入口。

### 1.3 数据侧已具备的 test 通路

`BatchRequestV3`（`datasets/multi_scene_dataset_v3.py`）已包含：

- `include_test: bool`
- `test_image_refs: Optional[List[ImageRef]]`

`MultiSceneDatasetV3.resolve_test_image_refs_deterministic(scene_id, segment_id)` 已能按 segment **确定性展开** test 用的 `ImageRef` 列表。`TrainSchedulerV4` 在 block 结束时也会写入 `block_test_image_refs`（见同文件 scheduler 相关逻辑）。

### 1.4 模型侧：尚无「纯推理、零监督」的一等公民路径

当前 `MinimalStreetForwardStage4_3` 的 `forward()` / `train_step()` 设计围绕**非空 targets** 与训练副作用（optimizer、hidden cache、node state 写回/重置）。`forward()` 即使在 `eval` 模式下仍需要 batch 中的 target 结构以构造 `mask_tgt_by_frame` / `mask_any_tgt_rigid` 等。`train_step()` 会无条件执行优化与状态更新。

因此：**监督式适配（adapt_supervised）** 可较大程度复用现有训练路径；**无监督纯推理（inference_only）** 需要显式拆分「构图 / 渲染 / 导出」与 **RuntimePolicy**，避免把 held-out test 视角误接入更新逻辑。

---

## 2. 配置：`test:` 块与数据侧测试友好设置

### 2.1 新增顶层 `test:`（建议）

与现有 `eval:`（训练期 PSNR、尾部一次性 test 等）区分，单独增加正式测试 runner 使用的配置块，例如：

```yaml
test:
  enable: true
  mode: both   # adapt_supervised | inference_only | both

  runner:
    fixed_scene_id: null
    fixed_segment_id: null
    deterministic: true
    seed: 123
    max_segments_per_scene: 0   # 0 表示不限制
    min_test_views_per_segment: 6

  split:
    require_eval_scene_ids: true
    require_nonzero_test_stride: true
    require_nonempty_test_views: true

  adapt_supervised:
    enable: true
    max_steps_per_segment: 2000
    validate_every_blocks: 1
    early_stop_patience: 8
    keep_best_by: psnr
    reset_runtime_state_each_segment: true

  inference_only:
    enable: true
    allow_hidden_cache_update: false
    allow_node_state_writeback: false
    allow_optimizer_step: false

  export:
    save_3dgs_init: true
    save_3dgs_best: true
    save_3dgs_final: true
    save_ply: true
    save_rendered_images: true
    save_per_view_metrics_json: true
```

实现时可选用**单独 YAML**（例如 `configs/minimal_streetforward_stage4_3_multi_scene_v4_test.yaml`）继承训练配置并覆盖 `test:` / `data.eval_scene_ids` / 像素 test 字段。

### 2.2 数据配置：正式测试前必须满足的条件

当前 `configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml` 中常见默认：

- `data.eval_scene_ids: []`
- `multi_scene.include_test: false`（由 `parse_include_test_v4` 等解析）
- `data.pixel_source.test_image_stride: 0`、`max_test_images: 0`

**正式 eval 建议：**

| 字段 | 说明 |
|------|------|
| `data.eval_scene_ids` | 填入正式 hold-out 场景 ID 列表 |
| `data.pixel_source.test_image_stride` | **非 0**；为 0 时 base dataset 可能使 train/test 帧集合重合，不适合汇报 |
| `data.pixel_source.max_test_images` | **正数**，限制每 segment test 帧上限 |
| `data.preload.warm_test_refs` | 测试脚本中建议 **true**，减少 eval 视角加载抖动 |

以上字段在代码库中已存在，仅需在测试场景下显式打开并校验。

---

## 3. 新测试脚本（不复用训练主脚本）

### 3.1 建议路径与职责

- **新文件**：`tools/test_minimal_streetforward_stage4_3.py`（或与 stage 命名对齐的固定名称）

**职责：**

- 遍历 `eval_scene_ids`（及可选 `fixed_scene_id` / `fixed_segment_id` 钉死单段调试）。
- 支持两种模式：`adapt_supervised`、`inference_only`，以及 `both`。
- 按 `(scene, segment, mode)` 写出独立目录与 `summary.json` / `per_view_metrics.json` / 渲染图 / 3DGS 导出。

### 3.2 建议主流程（伪代码）

```text
setup(cfg, args)
dataset = build_multi_scene_dataset_v3(cfg, device)
eval_scene_ids = resolve_eval_scene_ids(cfg)
model = MinimalStreetForwardStage4_3(...)
load_init_checkpoint(..., weights_only=True)

for scene_id in eval_scene_ids:
  for segment_id in resolve_eval_segments(dataset, scene_id, cfg.test):
    if mode in (adapt_supervised, both):
      run_supervised_adaptation_test(...)
    if mode in (inference_only, both):
      run_pure_inference_test(...)
```

### 3.3 输出目录约定

```text
log_dir/
  test/
    scene_005/
      segment_000/
        adapt_supervised/
          3dgs_init.pt
          3dgs_best.pt
          3dgs_final.pt
          summary.json
          per_view_metrics.json
          renders/
        inference_only/
          3dgs_init.pt
          summary.json
          per_view_metrics.json
          renders/
```

---

## 4. `MultiSceneDatasetV3`：正式 eval batch API

### 4.1 动机

`get_segment_batch_from_image_refs(request: BatchRequestV3)` 要求 `target_image_refs` 非空，适合监督训练与 **adapt_supervised**。对 **inference_only**，不应强制构造「监督 targets」，以免语义混淆。

### 4.2 建议新增类型

在 `datasets/multi_scene_dataset_v3.py`：

```python
@dataclass(frozen=True)
class EvalRequestV3:
    scene_id: int
    segment_id: int
    source_image_ref: ImageRef
    eval_image_refs: List[ImageRef]
```

### 4.3 建议新增 API

```python
def get_segment_eval_batch_from_image_refs(
    self,
    request: EvalRequestV3,
) -> Dict[str, Any]:
    ...
```

**返回字段建议（仅 eval 语义）：**

- `source`、`eval`（或 `eval_views` / `eval_images` 等与训练 batch 一致的内部命名，但**不用 `target` 键**）
- `scene_id`、`segment_id`、`segment_first_frame_idx`
- `aabb`、`dynamic_info`
- 背景/远景/rigid 所需静态资源：`background_pointcloud`、`distant_pointcloud` 等

**Test refs 来源**：继续复用 `resolve_test_image_refs_deterministic()`，不在此重复实现 split。

### 4.4 Base dataset 的 train/test 划分

`MultiSceneDataset` / `_load_scene()` 已先做 train/test split，再做 keyframe/segment；segment 上记录 `test_frame_indices`。**不要重写** split，只在测试 runner 的配置校验中拒绝 `test_image_stride == 0` 等不合法正式 eval 配置。

---

## 5. 调度器：训练 vs 测试

### 5.1 `TrainSchedulerV4` 用于 `adapt_supervised`

现有设计假设每步有 `total_target_images >= 1`、`include_source == True`，生命周期为 U / block / episode / segment。**监督式适配**可直接复用，仅需：

- 由测试脚本传入 `eval_scene_ids` 与可选 `fixed_scene_id` / `fixed_segment_id`（限制在**测试脚本**层放宽，而非改训练脚本的多场景假设）。

### 5.2 新建 `TestSchedulerV4`（建议：`datasets/test_scheduler_v4.py`）

接口方向：

```python
class TestSchedulerV4:
    def __init__(..., mode: str): ...
    def next_adapt_batch(self) -> Dict[str, Any]: ...
    def next_eval_batch(self) -> Dict[str, Any]: ...
    def pop_events(self) -> List[Dict[str, Any]]: ...
```

**子模式 A — `adapt_supervised`：** 内部可委托 `TrainSchedulerV4`，并增加例如 `get_block_test_refs(scene_id, segment_id)`，在 block 结束时对固定 test refs 做验证。

**子模式 B — `inference_only`：** 不采样 `target_image_refs`；每步仅 `source_image_ref` + `eval_image_refs`；不产生训练语义的 `block_end`；事件可用 `eval_begin`、`eval_view_chunk_end`、`segment_eval_end` 等。

### 5.3 不建议「塞假 target」

`forward()` 会从 `targets` 取 `frame_idx` 并构造与 target 相关的 mask；把 test 帧伪装成 target 会破坏「无监督纯推理」的定义，并与优化/写回路径纠缠。

---

## 6. `MinimalStreetForwardStage4_3`：接口拆分与 RuntimePolicy

### 6.1 建议的新增/拆分接口

| 接口 | 职责 |
|------|------|
| `adapt_step` / 保留 `train_step` | 监督适配 = 现有训练步逻辑 |
| `build_scene_representation_from_source(batch, *, allow_hidden_cache_update, allow_node_state_writeback)` | 仅从 source 构建可渲染场景表示与 node 状态 |
| `render_views_from_scene_state(scene_state, eval_views)` | 对给定相机列表渲染，**无副作用** |
| `export_3dgs_state(batch_or_key, *, include_hidden=False)` | 导出 bg/distant/rigid/sky 分支参数与元数据 |
| `import_3dgs_state(state)` | 可选：复现渲染与 debug |

导出内容至少覆盖各分支的 `means`、`scales_log`、`quats`、`opacity_logit`、`sh_dc`、`sh_rest`，以及 `scene_id`、`segment_id`、`cache_key`、`segment_aabb`、`source_image_ref`、`test_image_refs`、可选 hidden cache。原料可从现有 `forward` / `train_step` 的 `out` 中 `_node_state_*`、`_cache_key` 等字段归纳。

### 6.2 `RuntimePolicy`（建议）

```python
@dataclass
class RuntimePolicy:
    do_backward: bool
    do_optimizer_step: bool
    update_hidden_cache: bool
    writeback_node_state: bool
    reset_node_state_after_block: bool
```

- **训练**：全开。
- **adapt_supervised**：与训练接近，但避免污染「全局训练态」的策略可按配置收紧。
- **inference_only**：全关；必须绕开 `train_step()` 中 hidden cache 与 node state 写回/重置（例如现有实现中约 1120–1147 行附近的逻辑，以当时代码为准）。

---

## 7. 导出模块：`tools/streetforward_test_export.py`

建议集中实现：

```python
def save_3dgs_state(path: str, state: Dict[str, Any]) -> None: ...
def save_3dgs_ply(path: str, state: Dict[str, Any]) -> None: ...
def save_test_summary(path: str, summary: Dict[str, Any]) -> None: ...
```

**保存时机（每个 scene/segment/mode）：**

- `3dgs_init.pt`：初始化或 `build` 完成后立刻。
- `3dgs_best.pt`：仅 `adapt_supervised`，验证指标最优时。
- `3dgs_final.pt`：该 segment 流程结束。

**格式**：主存档用 `.pt`（保留分支语义与元数据）；`.ply` 仅作可视化辅件。

---

## 8. 训练脚本建议改动（小步）

对 `tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py`：

1. **移除或废弃**正式测试职责：`eval.run_test_at_end` 及尾部「最后一个 batch 的 test_views」评估。
2. **保留** block_end 的 train metrics 与现有训练监控。
3. 若用户仍在训练配置中设置 `test.enable: true`（待实现），**报错或明确 warning**，引导使用新 `test_minimal_streetforward_stage4_3.py`。

---

## 9. 正式 eval 协议（约定）

1. 场景必须来自 `data.eval_scene_ids`。
2. `test_image_stride != 0`。
3. `max_test_images > 0`。
4. 每个 segment 至少 `min_test_views_per_segment` 个 test views，否则跳过并记录原因。
5. **adapt_supervised**：仅用该 segment 的 **train split** 优化；block 结束在该 segment 的 **test refs** 上验证；保存 best/final 3DGS。
6. **inference_only**：从 source 构建场景；无 target loss；渲染全部 test refs；保存 init 3DGS 与 summary。

理由：base dataset 已完成「先切 train/test，再映射到 segment 的 test 帧」，测试层只消费与校验，不重复造轮子。

---

## 10. 推荐实现顺序（降低风险）

1. 增加 `test:` 配置块与测试专用 YAML。
2. 新建 `tools/test_minimal_streetforward_stage4_3.py`，**先只做 `adapt_supervised`**（最大复用 `TrainSchedulerV4` + 现有 `train_step`）。
3. 实现 `export_3dgs_state()` 与 `streetforward_test_export.py`，能落盘 best/final。
4. 从训练脚本移除 `run_test_at_end` 的「伪正式」评估。
5. 实现 `build_scene_representation_from_source()` + `render_views_from_scene_state()`。
6. 最后实现 `TestSchedulerV4` 的 `inference_only` 与 `get_segment_eval_batch_from_image_refs()`。

---

## 11. 相关代码索引（便于跳转）

| 主题 | 位置 |
|------|------|
| `BatchRequestV3` | `datasets/multi_scene_dataset_v3.py` |
| `resolve_test_image_refs_deterministic` | 同上 |
| `get_segment_batch_from_image_refs` | 同上 |
| `TrainSchedulerV4`、`block_test_image_refs` | `datasets/multi_scene_dataset_v3.py`（scheduler 类尾部） |
| 多场景 v4 dataset / scheduler 构建 | `tools/train_minimal_streetforward_stage4_3_v4_common.py` |
| 训练主脚本约束与尾部 test | `tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py` |
| 默认数据与 preload | `configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml` |

---

---

## 12. 实现状态（仓库落地）

下列条目已在代码中落地，便于与设计文档交叉对照：

| 能力 | 位置 |
|------|------|
| 正式测试 runner（`adapt_supervised` / `inference_only` / `both`） | [`tools/test_minimal_streetforward_stage4_3.py`](tools/test_minimal_streetforward_stage4_3.py) |
| `test:` 配置块与测试用 YAML 模板 | [`configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml`](configs/minimal_streetforward_stage4_3_multi_scene_v4.yaml)、[`configs/minimal_streetforward_stage4_3_multi_scene_v4_test.yaml`](configs/minimal_streetforward_stage4_3_multi_scene_v4_test.yaml) |
| 测试配置校验、`eval-only` 下 `initialize()` 兜底 | [`tools/streetforward_test_config.py`](tools/streetforward_test_config.py) |
| `export_3dgs_state`（含 rigid→world/seg0 元信息与 `rigid_world` 分支）、`RuntimePolicy`、`build_scene_representation_from_source`、`render_views_from_scene_state`、`import_3dgs_state` | [`models/streetforward/minimal_trainer_stage4_3.py`](models/streetforward/minimal_trainer_stage4_3.py) |
| `save_3dgs_state` / `save_3dgs_ply` / `save_test_summary` | [`tools/streetforward_test_export.py`](tools/streetforward_test_export.py) |
| `EvalRequestV3`、`get_segment_eval_batch_from_image_refs` | [`datasets/multi_scene_dataset_v3.py`](datasets/multi_scene_dataset_v3.py) |
| `TestSchedulerV4`（inference chunk 调度） | [`datasets/test_scheduler_v4.py`](datasets/test_scheduler_v4.py) |
| 训练脚本：配置中 `test.enable=true` 报错；尾部 `run_test_at_end` 已移除正式 test 流程 | [`tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py`](tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py) |
| 配置与 PLY rigid 协议单测 | [`tests/test_streetforward_test_eval_config.py`](tests/test_streetforward_test_eval_config.py) |

**运行示例**（需有效数据与 `--init_checkpoint`）：

```bash
conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding \
  python tools/test_minimal_streetforward_stage4_3.py \
  --config_file configs/minimal_streetforward_stage4_3_multi_scene_v4_test.yaml \
  --init_checkpoint /path/to/minimal_sf_stage4_3_multi_scene_v4_stepXXXX.pt \
  --init_weights_only
```

`--max_steps` 可覆盖 `test.adapt_supervised.max_steps_per_segment`。`mode=both` 且 `test.both.reload_init_before_inference=true`（默认）时，会在 `inference_only` 前 `reset_node_state()` 并重载 `--init_checkpoint`（仅权重），使纯推理从 init 权重出发；若希望 inference 接续 adapt 后的权重，可将该项设为 `false`。

---

文档版本：与仓库 Stage4.3 多场景 V4 训练脚本及 `MultiSceneDatasetV3` API 对齐；具体行号以后续代码变更为准。
