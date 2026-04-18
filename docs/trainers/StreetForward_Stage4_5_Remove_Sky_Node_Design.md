# StreetForward Stage4.5 设计方案（移除 Sky Node + Sky-only 渲染，保留 Sky Mask）

## 1. 问题复核结论（针对评审意见）

本次复核后，以下问题均确认**存在**，并已在本文方案中修正为硬约束：

- **P1（最高优先级）**：若移除 sky node 且无任何 sky 颜色来源，`Stage4_2` 现有 loss 会产生目标冲突  
  `mask loss` 推 `opacity -> 1 - sky_mask`（天空趋向透明），但 `RGB/SSIM` 仍在天空区域拟合 GT。  
  **修正**：`Stage4_5` 将天空区域从 `RGB/L1 + SSIM` 监督中排除，仅在 non-sky 区域做 photometric。
- **P2**：天空渲染语义必须一次性说清  
  之前“可选 `_composite_sky`”表述模糊。  
  **修正**：`Stage4_5` 明确为 **No sky rendering at all**（无 `sky node`、无 `sky_model`、无 `_composite_sky*`）。
- **P3**：`Stage4_4` 的 source 2D helper 不能直接搬  
  `Stage4_4._render_source_composite_for_cnn` 强依赖 `gaussians_sky`，且旧 `_compute_2d_features_for_gaussians` 被显式废弃。  
  **修正**：`Stage4_5` 新写 scene-only helper 与主路径，不复用 sky-composite helper。
- **P4**：日志输出策略不能只靠 `result.get()`  
  共享脚本与工具链仍可能硬依赖 sky 字段。  
  **修正**：定义统一输出协议（compat adapter + 长期 branch-agnostic 迁移）。
- **P5（职责边界）**：`validation_scheduler_v7` 不必扩 schema  
  sky mask 强制性属于执行层/配置层，不是 episode 编排职责。  
  **修正**：保持 scheduler schema 不变，仅在 validation executor 做 fast-fail。

---

## 2. Stage4.5 定义与硬边界

`Stage4_5` 定义为：

- **保留**：`v7 scheduler`、`v7 validation`、多场景、fused 多相机 scene backprojection；
- **移除**：`NodeStateSky` 全链路；
- **移除**：所有天空渲染来源（`sky node` 与 `sky_model` 路径都不保留）；
- **保留**：`sky_mask`，但用途限定为监督口径裁剪、occupancy 监督与验证分区指标。

硬边界（必须满足）：

- `Stage4_5` 不得调用 `_render_sky_single_view`、`_composite_sky_gs`、`_composite_sky`；
- `Stage4_5` 不得解析 `model.sky` 与 `model.branches.sky`；
- `Stage4_5` 不得创建或读写 `node_states_sky` / `h_cache_sky`。

---

## 3. 模型架构方案（4_5）

## 3.1 继承策略

- 新类：`MinimalStreetForwardStage4_5`
- 继承基座：`MinimalStreetForwardStage4_2`
- 叠加能力：移植 `Stage4_4` 的 **scene-only fused 多相机回投**能力（不是复用 sky-composite helper）。

这样可避免 `Stage4_4 -> Stage4_3` 的 sky 深耦合链路。

## 3.2 Source 2D（必须新写 scene-only helper）

新增函数（建议）：

- `_render_source_scene_only_for_cnn(...)`
- `_backproject_scene_features_multi_camera(...)`
- `_compute_2d_features_all_branches_once_multicam_fused_scene_only(...)`

关键点：

1. 输入只含 `gaussians_scene = bg + distant + rigid@S`；
2. CNN 输入构造为 `concat(source_image, rendered_scene_rgb)`，不含 sky composite；
3. fused 回投仅做 scene 一次 pass；
4. `src_backproject_pass_count = 1`。
5. `source_pair_valid_mask` 必须严格对齐 source 视角（`[V,H,W]`），并沿
   `Stage4_5 -> AlphaTWeightExtractorV3 -> gsplat multi-camera fused forward/backward`
   全链路传递（无 silent fallback）。

明确禁止：

- 不调用 `Stage4_4._render_source_composite_for_cnn`（该函数要求 `gaussians_sky`）；
- 不走 `_compute_2d_features_scene_and_sky_gated`；
- 不依赖被废弃的旧 `_compute_2d_features_for_gaussians` 路径。

补充约束（实现落地）：

- `source_sky_masks` / `source_egocar_masks` 只要提供，就必须与 `source_images` 等长；否则 fast-fail；
- `pairs_total` 与 `pairs_after_mask` 口径分离：`pairs_total` 为 mask 前，`pairs_after_mask` 为 mask 后；
- fused multi-camera custom op（forward + backward/sharded backward）必须支持 `pair_valid_mask` 参数。

---

## 4. Loss 与监督口径（核心修正）

## 4.1 监督冲突修复

在 `Stage4_5` 中定义：

- `valid_non_sky_mask = valid_loss_mask * (1 - sky_mask)`
- `valid_all_mask = valid_loss_mask`

其中 `sky_mask` 为 canonical `1=sky, 0=non-sky`。

然后：

- `L1/RGB`：仅在 `valid_non_sky_mask` 上计算；
- `SSIM`：仅在 `valid_non_sky_mask` 上计算（masked SSIM）；
- `mask/occupancy`：保持在 `valid_all_mask` 上，用 `gt_occupied = (1 - sky_mask)`；
- `opacity entropy`：建议保持 `valid_all_mask`（可配置为 non-sky）。

## 4.2 fast-fail 规则

- `losses.mask.require_sky_mask=true` 时，任何 target 缺失 `sky_mask` 直接报错；
- 若某视图 `valid_non_sky_mask.sum()==0`：
  - photometric 分项对该 view 记 0，不参与该帧 photometric 均值；
  - occupancy 分项仍可参与（若 `valid_all_mask` 有效）；
  - 同时记录计数日志（如 `num_views_no_non_sky_supervision`）。

---

## 5. 渲染语义（明确去天空渲染）

`Stage4_5` 渲染输出定义：

- `pred_rgb = render(bg + distant + rigid)`；
- 无额外天空颜色补偿；
- 不引入 `sky_model` 或 sky pass。

因此：

- 配置中删除 `model.sky` 与 `model.branches.sky`；
- 训练与验证指标解释应以 non-sky 为主（全图指标会受到天空区域“无建模”影响，这是预期现象）。

---

## 6. 交付项 1：新增模型/配置/训练脚本

新增文件：

- `models/streetforward/minimal_trainer_stage4_5.py`
- `configs/minimal_streetforward_stage4_5_multi_scene_v7.yaml`
- `tools/train_minimal_streetforward_stage4_5_multi_scene_v7.py`

并更新：

- `models/streetforward/__init__.py`（导出 `MinimalStreetForwardStage4_5`）
- `docs/trainers/StreetForward_Flow.md`（补 Stage4.5）

配置要求：

- 保留 `scheduler_v7` / `validation_v7` / fused v3/v4 开关；
- 保留 `data.pixel_source.load_sky_mask=true`；
- 删除所有 sky node 配置块；
- 新增 loss 口径配置（建议）：
  - `losses.photometric.exclude_sky_region: true`
  - `losses.mask.require_sky_mask: true`

---

## 7. 交付项 2：validation_v7 链路改造（不改 scheduler schema）

## 7.1 `datasets/validation_scheduler_v7.py`

保持 schema 不变（`ValidationEpisodeSpecV7` 不扩字段）。  
仅允许增加内部一致性检查（如空链、num_cams），不引入 sky 语义字段。

## 7.2 validation executor（实质改造点）

文件：`tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py` 的 `_run_validation_v7_round`。

改造：

- 增加 `sky_mask` 读取与 fast-fail（由 `validation_v7.metrics.use_sky_mask_regions` 控制）；
- 新增分区指标：
  - `psnr_non_sky` / `ssim_non_sky`（主指标）
  - `psnr_sky` / `ssim_sky`（诊断指标，可选）
  - `sky_mask_coverage`
- 输出到 `per_view_metrics.json`、`summary.json`、TensorBoard。

强调：validation 的时序调度不变，变化只在 metrics/executor 层。

---

## 8. 交付项 3：全线移除 sky node 相关内容

## 8.1 必改模块

- `models/streetforward/minimal_trainer_stage4_5.py`（新增，no-sky 实现）
- `tools/train_minimal_streetforward_stage4_5_multi_scene_v7.py`（新增）
- `configs/minimal_streetforward_stage4_5_multi_scene_v7.yaml`（新增）
- `models/streetforward/__init__.py`（导出）
- `tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py`（validation + 日志协议）
- `tools/streetforward_validation_v7_config.py`（增加 metrics 配置解析）

## 8.2 日志与输出协议（替代“仅 result.get”）

采用两阶段策略：

### 阶段 A（短期，必须）

`Stage4_5.train_step` 输出 **兼容 superset**：

- 保留 sky 键但固定为 0（如 `num_gaussians_sky=0`、`grad_norm_sky=0.0`）；
- 增加 `branch_presence = {bg:true,distant:bool,rigid:bool,sky:false}`。

这样所有旧脚本不会因缺键崩溃。

### 阶段 B（中期，建议）

把共享日志/报表迁移到 branch-agnostic：

- 统一通过 `branch_presence` 决定哪些指标应记录；
- 删除对 sky 固定键的硬依赖。

---

## 9. 可选路径：回退 gsplat sky-gated 相关提交

你提出的可选路径可行：将 `third_party/gsplat` 从 `7ec7f0c9eaf5568bc0fd833fcd8bc711a20346b4` 回退到 `67c817ebf1f5d60f65a37b7ba658958a3ce4b4f7`。

设计上含义：

- `Stage4_5` 本身不需要 sky-gated kernel；
- 回退可减少对“天空 gate CUDA 分支”的维护负担。

风险提醒：

- 需确认该回退不会影响当前 `Stage4_4` 已依赖接口；
- 若仓库需并行保留 `Stage4_4` 训练能力，建议做版本隔离或 feature flag。

---

## 10. 实施顺序

1. 新建 `Stage4_5` trainer（先打通 scene-only fused 回投 + no-sky render）；
2. 落地 loss 口径修正（non-sky photometric）；
3. 落地 train_step 兼容输出协议（superset + branch_presence）；
4. 接入 `train_minimal_streetforward_stage4_5_multi_scene_v7.py`；
5. 改造 validation executor 的 sky-mask 分区指标；
6. 文档与测试收口。

---

## 11. 验收标准（更新版）

- [ ] `Stage4_5` 无 sky node、无 sky_model、无 `_composite_sky*`；
- [ ] `RGB/L1 + SSIM` 严格在 non-sky 区域监督；
- [ ] occupancy/mask loss 与 `1-sky_mask` 保持；
- [ ] source 2D 使用 scene-only fused helper（非 sky-composite helper）；
- [ ] `validation_scheduler_v7` schema 不扩展；
- [ ] validation 输出 non-sky 主指标与 sky 分区诊断指标；
- [ ] train_step 输出满足兼容协议（旧工具不崩溃）。

---

## 12. 实现后建议验证命令

- `conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding python -m pytest tests/test_validation_scheduler_v7.py`
- `conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding python -m pytest tests/test_minimal_stage4_5.py`
- `conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding python tools/train_minimal_streetforward_stage4_5_multi_scene_v7.py --max_steps 10`
