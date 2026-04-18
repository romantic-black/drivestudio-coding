# StreetForward Stage4.4 Sky 优化失效排查计划

## 背景与问题定义

在 `MinimalStreetForwardStage4_4` 训练到约 2w step 后，天空区域仍然缺乏有效优化（视觉上接近灰雾、结构弱），并已明显影响验证表现。

本计划基于以下上下文：

- 流程文档：`docs/trainers/StreetForward_Flow.md`
- 训练配置：`configs/minimal_streetforward_stage4_4_multi_scene_v7.yaml`
- 当前代码：`models/streetforward/minimal_trainer_stage4_4.py` 及父类 `minimal_trainer_stage4_3.py` / `minimal_trainer_stage4_2.py` / `minimal_trainer_stage4_0.py`
- 实际运行配置：`/root/autodl-tmp/outputs/minimal_sf_stage4_4_multi_scene_v7/experiment001/config.yaml`

## 当前判断（先验结论）

从现有日志信号看，问题更像是「sky 路径畅通，但监督和更新信号被系统性削弱」，而不是「sky 完全断图」。

支持这一判断的关键信号（来自你的日志描述）：

- `num_sky_src_feat_valid` 长期非零且量级稳定
- `num_sky_update` 长期非零
- `grad_norm_sky` 持续非零
- `perf_2d_bp_sky_pairs_total` 量级很大

这类信号更符合「能跑但学不动/学得很慢」。

---

## 假设优先级与排查实验

下面按优先级从高到低排列，每条都给最小可执行实验与判定标准。

### H1（最高优先级）：sky 监督被系统性削弱，而非 sky 通路断裂

**代码依据**

- 最终合成：`_composite_sky_gs(pred, acc_scene, rgb_sky)`，即 `pred = scene + sky * (1-acc_scene)`（`minimal_trainer_stage4_3.py`）
- mask loss 的 `pred_occupied` 仍用 `opacity=acc_scene`，并未显式对「composite 后 sky」构造占用监督（见 `forward()` loss 部分）
- 当前配置权重：`rgb=1.0`、`ssim=0.05`、`mask=0.02`

**风险机制**

- sky 分支几何冻结（`freeze_means=true`、`freeze_quat=true`），可调自由度少；
- mask loss 对 sky 不形成直接正向监督时，sky 主要依赖 RGB/SSIM 间接竞争，容易被 scene 分支压制。

**最小实验（200~500 step）**

1. 保持 2D 与渲染路径不变；
2. 临时构造 `sky-only RGB loss`（仅在 `sky_mask==1` 区域统计 L1/SSIM）；
3. 观测 `sky_only_render` 与天空区域 PSNR/SSIM 的短程斜率。

**判定标准**

- 若 sky-only loss 下短程显著起效，则主因是监督竞争/监督不足；
- 若仍无明显改善，再优先转查 H2/H3。

---

### H2：gated sky backprojection 对 sky 证据抑制过强

**代码依据**

- gate 生成：`gate_image = 1 - acc_scene_src`（`minimal_trainer_stage4_4.py::_render_source_composite_for_cnn`）
- gated fused 回投：`_backproject_sky_features_gated_multi_camera(...)`
- sky source 2D 为 scene+sky 共享 CNN 特征 + gated 聚合（`minimal_trainer_stage4_3.py::_compute_2d_features_scene_and_sky_gated`）

**风险机制**

- 若 scene alpha 在地平线/树梢/建筑边缘有 over-coverage，`1-acc_scene` 会把本应属于 sky 的回投证据压小；
- 表现为「有 update、有梯度，但长期弱优化」。

**最小实验（每组 200~500 step）**

- A 组：`gate_image = 1`（去 gate）
- B 组：`gate_image = sky_mask`（或轻度膨胀后的 sky_mask，仅用于诊断）
- C 组：保留现状 `gate_image = 1-acc_scene`

**关键日志**

- `mean(gate | GT sky)`
- `mean(gate | GT non-sky)`
- `mean(acc_scene | GT sky)`
- `mean(acc_scene | GT non-sky)`

**判定标准**

- 若 A/B 明显优于 C，基本坐实 gate 过抑制；
- 若 A≈C 且都差，再转查 H3/H4。

---

### H3：2D 坐标并非全错，但 gate 与 feature 空间对齐存在系统偏差

**代码依据**

- `features_2d` 在特征分辨率；
- `gate_image` 在图像分辨率；
- 同时存在 resize 路径，且使用 `align_corners=True`（`minimal_trainer_stage4_4.py`）

**风险机制**

- 不同分辨率和采样中心定义不一致时，horizon 附近可能出现稳定偏差；
- 不一定导致「完全错投」，但会造成长期弱学习。

**最小实验**

- 构建单 batch、单 camera、小规模 gaussian 的 PyTorch reference gated backprojection；
- 对比 CUDA fused gated 回投输出：
  - `weight_sum_support`
  - `weight_sum_feature`
  - `feat_sum`

**判定标准**

- 若参考实现和 CUDA 误差在边界系统偏大，继续深入 kernel 坐标定义；
- 若误差很小，可基本排除主链路坐标问题。

---

### H4：rotation-only 语义正确，但 sky shell 几何分布与初始化不匹配数据

**代码/配置依据**

- `origin_mode: camera_centered_rotation_only`
- `hemisphere_up: [0,-1,0]`
- `radius: 30`
- `resolution: 500`
- sky 初始化：`scale_init.isotropic_log_value=-0.5`、`opacity_init=0.05`、`sh_degree=1`

**风险机制**

- 点位覆盖与真实 sky 区域不匹配时，优化空间有效覆盖不足；
- 初始尺度偏大 + 低阶 SH 可能形成“平滑灰幕”。

**最小实验**

1. 保存同 batch 的 6 图：
   - `scene_only_rgb`
   - `sky_only_rgb`
   - `composite_rgb`
   - `acc_scene`
   - `gate_image`
   - `gt_sky_mask`
2. 做 `sky gaussian projection overlay`（投影点覆盖与 GT sky 的重叠统计）。

**判定标准**

- 若 `sky_only_rgb` 本身极其平坦、std 极低，优先调整初始化与表达力；
- 若投影点大面积落在 non-sky 区域，优先修正 shell 几何分布。

---

### H5：sky 分支表达能力组合偏弱（非硬 bug）

**现状**

- 几何冻结 + 低阶 SH + 灰色初始化 + 低 opacity 初值；
- 在多场景、多天气下可能不够表达天际线结构与亮度梯度。

**最小实验（单变量）**

- 仅改一个初始化项：
  - `sky.opacity_init: 0.05 -> 0.15~0.25` 或
  - `sky.scale_init.isotropic_log_value: -0.5 -> [-1.2, -1.8]`

**判定标准**

- 若 sky-only 可见层次显著增加，说明表达初态是关键瓶颈之一。

---

## 分阶段执行计划（建议顺序）

### Phase 1：先判定“监督问题 vs 投影问题”（半天内出结论）

1. 加 6 图调试导出（每 N step）；
2. 跑 3 组短程 ablation（`gate=1` / `gate=sky_mask` / `sky-only loss`）；
3. 以 sky-only 可视化和天空区域指标斜率作首轮归因。

### Phase 2：验证坐标与对齐（1 天）

1. 做 sky 投影 overlay（3~5 个 camera）；
2. 做 reference vs CUDA gated backprojection 数值对比。

### Phase 3：数值抑制定位与结构修正（1~2 天）

1. 补齐 gate/acc 分区统计日志；
2. 视 Phase 1~2 结果，选择：
   - 调整 gate 构造；
   - 增强 sky 监督；
   - 调整 sky 初始化/表达力。

---

## 统一日志与观测面板（必须加）

建议至少按 `log_interval` 输出以下标量：

- `gate_mean_on_gt_sky`
- `gate_mean_on_gt_non_sky`
- `acc_scene_mean_on_gt_sky`
- `acc_scene_mean_on_gt_non_sky`
- `sky_only_rgb_mean`
- `sky_only_rgb_std`
- `delta_sh_dc_sky_from_init`
- `delta_opacity_logit_sky_from_init`
- `delta_scales_log_sky_from_init`

并建议保留以下已有计数用于横向校验：

- `num_sky_src_feat_valid`
- `num_sky_update`
- `grad_norm_sky`
- `perf_2d_bp_sky_pairs_total`

---

## 决策树（执行后如何收敛）

- **若 `sky-only loss` 立刻见效**：优先改监督设计（sky 区域加权、mask/photometric 职责重分配）。
- **若 `gate=1` 比基线好很多**：优先修 gate 策略（可能从 `1-acc_scene` 改为混合 gate 或软下限）。
- **若 reference 与 CUDA 差异大**：先修对齐/采样契约，再谈损失权重。
- **若 `sky_only_rgb` 长期平坦**：优先调 sky 初始化与表达力上限。

---

## 实施注意事项

- 所有排查建议均先做短程 smoke test，避免长训练成本；
- 每次只改一个主变量，避免混淆因果；
- 默认使用 fast-fail：缺少关键中间量（如 gate/sky-only 图）即报错，不做静默回退。

