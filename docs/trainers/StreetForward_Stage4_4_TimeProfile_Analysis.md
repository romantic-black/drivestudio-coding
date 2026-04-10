# StreetForward Stage4.4 训练耗时分析（当前运行快照）

## 数据来源

- 运行输出目录：`outputs/minimal_sf/minimal_sf_stage4_4_multi_scene_v5`
- 指标文件：`outputs/minimal_sf/minimal_sf_stage4_4_multi_scene_v5/metrics_history.jsonl`
- 配置文件：`outputs/minimal_sf/minimal_sf_stage4_4_multi_scene_v5/config.yaml`
- 训练与计时代码：
  - `tools/train_minimal_streetforward_stage4_3_multi_scene_v4.py`
  - `models/streetforward/minimal_trainer_stage4_4.py`
  - `models/streetforward/minimal_trainer_stage4_3.py`
  - `models/feature_extractors/alpha_t_extractor_v3.py`
  - `models/feature_extractors/alpha_t_extractor_v2.py`

> 说明：当前基于 `metrics_history.jsonl` 中已落盘的 40 条训练记录统计（训练还在进行中，属于“当前快照”）。

---

## 1) 各模块时间占用（当前）

以 `step_time_ms` 为基准（均值）：

- `step_time_ms`: **2010.53 ms**
- `forward_ms`: **914.05 ms**（**45.46%**）
- `backward_ms`: **1090.55 ms**（**54.24%**）
- `optimizer_ms`: **3.21 ms**（**0.16%**）
- 其余开销（计时边界/杂项）：**2.72 ms**（**0.14%**）

结论：**主要时间在 forward + backward，且 backward 已超过 forward。**

---

## 2) forward 内部细分（当前）

来自 `perf_*` 字段（`MinimalStreetForwardStage4_4._compute_2d_features_for_gaussians` 累计）：

- `perf_2d_bp_streaming_total_ms`: **543.65 ms**
  - 占 `forward_ms` 约 **59.48%**
- `perf_2d_bp_fused_backproject_total_ms`: **500.35 ms**
  - 占 `forward_ms` 约 **54.74%**
- `perf_2d_bp_render_packed_total_ms`: **5.11 ms**
  - 占 `forward_ms` 约 **0.56%**
- `perf_2d_rgb_render_rgb_only_ms`: **7.96 ms**
  - 占 `forward_ms` 约 **0.87%**

同时：

- `num_source_views` = 6（稳定）
- `num_targets` = 18（稳定）
- `perf_2d_bp_pairs_total` 均值约 **2.70e8**
- `perf_2d_bp_pairs_after_threshold / pairs_total = 1.0`（几乎没有被阈值过滤）

结论：**forward 里的大头是 fused backproject 流程（尤其 `fused_backproject` 本体）。**

---

## 3) 为什么 backward 用时这么久

`backward` 慢不是单点问题，而是“高规模图 + 多次反向链路”的叠加：

1. **反向本身规模大**
   - 当前多源设置是 `6 source views`，目标是 `18 targets`。
   - `pairs_total` 在 `2.7e8` 量级，反向要处理对应的大量关联贡献，计算量和内存访问都重。

2. **2D 特征反传走自定义 CUDA backward**
   - `alpha_t_extractor_v2.py` 中 `_RasterizeAndBackprojectFeatOnlyFn.backward` 会调用 `backproject_feature_grad_in_range(...)`。
   - 这是 fused 路径的核心反传算子，数据规模大时会显著拉长 backward。

3. **除了 `loss.backward()`，还有一次显式 `torch.autograd.backward(...)`**
   - `minimal_trainer_stage4_3.py` 里先 `out["loss"].backward()`，
   - 紧接着 `_backward_to_render_params_bg_rigid_distant_sky(...)` 再对 render params 做显式反传（`minimal_trainer_stage4_0.py`）。
   - 这会让 backward 阶段承担更多梯度传播工作（跨 bg/rigid/distant/sky 多分支）。

4. **参数与分支较多，优化器不是瓶颈**
   - `optimizer_ms` 仅 ~3ms，说明不是 `optimizer.step()` 慢，而是图反传和相关 CUDA kernel 慢。

---

## 4) 我认为最值得优先做的优化空间（按优先级）

### P0：先把“有效 pair 数”压下来（最直接）

- 证据：`pairs_after_threshold == pairs_total`，说明当前 `weight_threshold` 基本没起过滤作用。
- 方向：
  - 提高 backproject 的权重阈值（小步试验），观察 `pairs_after_threshold` 是否下降；
  - 若下降明显，通常能同时降低 forward/backward（尤其 backward）耗时。
- 风险：阈值过高可能影响重建质量，建议做小网格（如 2-3 档）并对齐 PSNR/SSIM。

### P1：降低每步参与视角/目标规模

- 当前配置是 `6 source views` + `18 targets`，是高算力配方。
- 方向：
  - 优先减少 source views 或 target 数；
  - 结合 scheduler 让每 step 的参与量更“稀疏”，换取吞吐。
- 这是“线性级”降本，通常见效快。

### P2：减少不必要的反向路径

- 针对 `_backward_to_render_params_bg_rigid_distant_sky(...)`：
  - 检查哪些分支在当前阶段可冻结（例如你配置里 distant/sky 已有 `freeze_means: true`，可以进一步评估其他参数是否阶段性冻结）。
  - 目标是减少显式 backward 的 tensor 对数量和尺寸。

### P3：围绕 fused backward kernel 做 profile 驱动优化

- 重点关注 `backproject_feature_grad_in_range` 的占比与瓶颈（访存/原子操作/负载不均）。
- 若已有 kernel profile 条件，可先确认：
  - 不同分辨率与 view 数下的缩放曲线；
  - 是否存在明显“长尾视角”（某些 view 耗时显著高于均值）。

### P4：计时精度与诊断补强（辅助项）

- 当前 `train_step` 的 phase timing 在记录 `t1/t2/t3` 后才做 `cuda.synchronize()`，会影响 phase 计时严谨性。
- 建议在关键 phase 计时前后都严格同步（或使用 CUDA event）做一次校准，避免误判优化方向。
- 这项不直接降耗时，但能提高后续优化决策质量。

---

## 5) 可直接执行的最小实验建议（下一步）

建议先做 2 组快速 A/B（每组短跑）：

1. **阈值实验（优先）**
   - 只改 backproject `weight_threshold`（2-3 档）。
   - 观察：`pairs_after_threshold`、`backward_ms`、`step_time_ms`、`psnr_mean`。

2. **规模实验**
   - 降低 `source views` 或 `total_target_frames`。
   - 观察同上指标，确认吞吐/质量 trade-off。

如果目标是“先提速再保质”，优先顺序建议：**P0 -> P1 -> P2**。

