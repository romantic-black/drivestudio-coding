# StreetForward Stage 4_4 设计：`alpha_t_extractor_v3` 与 Multi-Source 2D Feature

本文档定义 `MinimalStreetForwardStage4_4` 与 `AlphaTWeightExtractorV3` 的落地方案，目标是在 **沿用 Stage 4_3 训练/测试/日志/配置系统** 的前提下，把 source 输入从单视角稳定扩展到多视角（multi-src）。

基线与参考：

- `models/feature_extractors/alpha_t_extractor_v2.py`
- `models/streetforward/minimal_trainer_stage4_3.py`
- `docs/trainers/StreetForward_Flow.md`
- `docs/trainers/StreetForward_2D_Feature_Module_Reference.md`
- `third_party/gsplat/gsplat`（当前 `HEAD = 5e9c30debd1976a0e8236ea34c3d83c9bc30a3d9`）

---

## 1. 设计范围

### 1.1 目标

`Stage 4_4` 的唯一新增能力：

- 同一 `source_frame_idx` 支持多个 source cameras
- 2D 特征提取路径保持 v2 fused 语义（逐视角 streaming + 全局累加 + 最终统一归一化）
- 保持 Stage 4_3 的 one-pass 编排语义（`src_backproject_pass_count = 1`）
- 保持 Stage 4_3 现有 sky/bg/distant/rigid 分支语义、写回语义、损失语义

### 1.2 非目标

`Stage 4_4` 不做：

- 不改 fused kernel 数学定义，不引入 multi-camera CUDA kernel
- 不改最终特征聚合公式
- 不改 support/mask 的阈值定义
- 不新增 `V x N x C` 等 per-view 大缓存
- 不改变 Stage 4_3 的训练循环与日志主结构

---

## 2. 与 Stage 4_3 的关系

`Stage 4_4` 推荐继承 `MinimalStreetForwardStage4_3`，并保持以下契约不变：

- **训练入口**：继续用 `train_step` / `forward` / `inference_step_from_train_batch`
- **one-pass split**：继续按 `[bg, distant, rigid_S, sky]` 切分 `feat_2d_all` 与 `acc_w_all`
- **写回逻辑**：继续沿用 `*_writeback_idx` 子集写回
- **日志字段**：继续沿用 `loss_*`、`num_*`、`hidden_norm_*`、`src_backproject_pass_count`、`perf_*`
- **配置结构**：继续沿用 `model.branches.*`、`logging.*`、`training.*`，并引入 `scheduler_v5`（替代 v4）

结论：`4_4` 是对 `4_3` 的 **输入维度扩展**，不是训练框架重写。

---

## 3. `alpha_t_extractor_v3` 定义（v2 fused only）

## 3.1 核心语义

沿用 `AlphaTWeightExtractorV2.render_and_backproject_streaming_fused(...)` 的数学语义：

- 对每个 source view 执行一次 packed render + fused backproject
- 每视角返回 `feat_sum_v`、`weight_sum_feature_v`、`weight_sum_support_v`
- 在 Python 端在线累加：
  - `A = Σ feat_sum_v`
  - `B = Σ weight_sum_feature_v`
  - `S = Σ weight_sum_support_v`
- 最终输出：
  - `feat_out = A / (B + eps)`
  - `acc_w = S`（用于 support mask，不参与特征归一化）

这与 v2 fused 主路径完全一致，只是把 `len(cameras) > 1` 作为正式契约固化。

## 3.2 接口契约

`alpha_t_extractor_v3`（可实现为新类或 v2 增量）应满足：

- `features_2d` 形状固定为 `[V, Hf, Wf, C]`
- `len(cameras) == V`
- 单视角 fused op 仍使用现有 `extract_single_weight_fused(...)`
- 支持 `return_accumulated_weights` 与 `return_debug_stats`

建议 debug stats 增量（均为标量/小列表）：

- `num_views`
- `pairs_total_per_view`
- `pairs_after_threshold_per_view`
- `render_packed_ms_per_view`
- `fused_backproject_ms_per_view`

## 3.3 Fast-fail（新增）

在 v3 中新增显式校验（fast-fail）：

```python
if features_2d.ndim != 4:
    raise ValueError("features_2d must be [V, Hf, Wf, C].")
if features_2d.shape[0] != len(cameras):
    raise ValueError("features_2d batch dim must match num cameras.")
```

继续保留 v2 已有 fast-fail：

- fused op 不可用
- `features_2d` 非 CUDA
- `requires_grad=True` 但 fused backward 不可用
- packed meta 缺字段 / dtype 错误 / global id 越界

---

## 4. Stage 4_4 的代码改动点

## 4.1 Trainer 新类建议

新增：

- `models/streetforward/minimal_trainer_stage4_4.py`

继承：

- `class MinimalStreetForwardStage4_4(MinimalStreetForwardStage4_3)`

主要策略：

- 尽量不改 `forward/train_step` 的大流程
- 改动聚焦在 `_compute_2d_features_for_gaussians(...)` 所在路径与输入校验
- 保持 one-pass split、各分支 mask/update/writeback 完全一致

## 4.2 `_compute_2d_features_for_gaussians(...)` 规范

继续沿用 4_3 的二阶段：

1. `render_rgb_only` 得到 `rendered_batch[V,H,W,3]`
2. `multi_channel_input = cat(image_batch, rendered_batch) -> [V,H,W,6]`
3. `image_feature_extractor` 得到 `features_2d[V,Hf,Wf,C]`
4. 调 `alpha_t_extractor_v3.render_and_backproject_streaming_fused(...)`

新增输入检查（推荐放在函数起始）：

```python
if source_views is None or source_images is None:
    raise ValueError("Stage4.4 requires source_views/source_images.")
if len(source_views) == 0:
    raise ValueError("Stage4.4 requires at least one source view.")
if len(source_views) != len(source_images):
    raise ValueError("source_views/source_images length mismatch.")
```

同分辨率约束显式化（避免隐式行为）：

```python
ref_hw = tuple(source_images[0].shape[:2]) if source_images[0].dim() == 3 else tuple(source_images[0].shape[1:3])
for i, img in enumerate(source_images):
    hw = tuple(img.shape[:2]) if img.dim() == 3 else tuple(img.shape[1:3])
    if hw != ref_hw:
        raise ValueError(f"All source_images must share identical H/W, got idx {i}: {hw} vs {ref_hw}.")
```

## 4.3 one-pass 语义保持

`_compute_2d_features_all_branches_once(...)` 中：

- `pass_count` 仍为 `1`
- `FeatureBackprojector(weight_threshold=0.0)` 语义保持
- split 区间保持 `[bg, distant, rigid_S, sky]`

即：source camera 数增加，不改变 “一次 source backprojection pass” 的统计语义。

---

## 5. 显存与反向传播约束

## 5.1 保持 streaming contract

允许存在：

- `features_2d[V,Hf,Wf,C]`
- 全局 accumulator：`[N,C] + [N] + [N]`
- autograd 保存每视角 packed meta（由 fused backward 现有机制决定）

不允许新增：

- `per_view_feat_sum[V,N,C]`
- `per_view_weight_sum[V,N]`
- `per_view_support_sum[V,N]`

## 5.2 梯度路径

当 `features_2d.requires_grad=True`：

- 继续走 `_RasterizeAndBackprojectFeatOnlyFn` backward
- 与 v2 fused 相同，不新增 backward kernel

---

## 6. Scheduler v5 设计（新增）

本节是 `Stage 4_4` 的关键补充：调度从 v4 的 image-ref（`frame,cam`）范式切到 **frame-level** 范式。

### 6.1 v5 核心目标

`scheduler_v5` 直接提供：

- `source_frame_idx`（整帧 source）
- `target_frame_indices`（整帧 targets，`target[0]` 必须是 source frame）

不再要求 scheduler 产出 `(frame_idx, cam_id)` 粒度的 image refs；由 dataset 侧把 frame 展开为该帧全部 cameras（即 `source_views/source_images` 为多相机列表）。

### 6.2 v5 与 v4 的关键差异

- **src 选择**：仍从 source keyframe 内随机选 frame，但输出是整帧而不是单 camera 图像
- **target 选择**：不做 overlap 评分（无 `pointcloud_topk`、无 temporal ring）
- **target 构成**：
  - `target_frame_indices[0] = source_frame_idx`
  - 其余 target 从 `source keyframe` 的相邻 keyframes 中随机采样 frame
- **batch 语义**：进入 trainer 前已经是整帧展开后的多视角 `source_views/source_images`

### 6.3 相邻 keyframe 采样规则

给定 `source_keyframe_idx = k`：

1. 构造邻域 keyframe 集 `N(k)`（优先 `k-1, k+1`，再按距离扩展）  
2. 从 `N(k)` 中随机抽取 keyframe（可按配置决定有放回/无放回）  
3. 每个被选 keyframe 内随机选一个 frame 作为 target frame  
4. 组装 `target_frame_indices = [source_frame_idx, ...sampled...]`

这满足“当前 src 帧 + 相邻 keyframe 随机帧”的目标，且实现简单、稳定。

### 6.4 v5 配置建议

```yaml
scheduler_v5:
  enable: true

  source:
    mode: random_frame_in_keyframe

  target:
    include_source_frame: true
    total_target_frames: 3
    policy: neighbor_keyframes_random_frames
    neighbor_ring: 1
    with_replacement: true

  overlap:
    mode: none
```

说明：

- `total_target_frames` 是 frame 级数量，不是 image 级数量
- `neighbor_ring=1` 表示优先一阶相邻 keyframes
- `overlap.mode=none` 明确关闭 overlap 计算与相关缓存/日志依赖

---

## 7. Fast-fail 迁移（single-src -> multi-src/full-frame）

`Stage4_1` 时代的 `_validate_stage4_1_batch(...)` 里有不少“单 src 心智”校验。`Stage 4_4 + scheduler_v5` 下建议改为以下规则。

### 7.1 保留并强化的检查

- `source_views/source_images` 非空，且长度一致
- 至少存在一个 `frame_idx == source_frame_idx` 的 target
- 每个 `source_views[i]` 必须能在 source-frame targets 中匹配到一个 target view（`camtoworlds` 一致）
- `sky_mask/viewdirs` 与 `gt_image` 的形状一致性检查保持

### 7.2 需要从 v4/v1 迁移修正的点

1. **从“至少一个 source-frame target”升级为“source-frame target 覆盖全部 source views”**  
   - 旧规则在单 src 时足够，但在多 src 下可能漏掉部分 source cameras。

2. **匹配优先级建议改为 `cam_idx` 优先，`camtoworlds` 兜底**  
   - 若 batch 带 `cam_idx`，先做离散 ID 对齐；  
   - 仅在缺 `cam_idx` 时做 `torch.allclose(c2w)`。

3. **删除与 overlap 相关的隐式约束/假设**  
   - v5 不计算 overlap，不应再假设 target 是 overlap 过滤后的子集。

4. **整帧语义一致性检查**  
   - 若数据集提供 `num_cams` 元信息，建议 fast-fail：`len(source_views) == num_cams`。  
   - 若 target 也是整帧展开，建议检查 source frame 的 target 视角数与 source 一致。

### 7.3 推荐校验伪代码

```python
if len(source_views) != len(source_images):
    raise ValueError("source_views/source_images length mismatch")

src_targets = [t for t in targets if int(t["frame_idx"]) == source_frame_idx]
if len(src_targets) < len(source_views):
    raise ValueError("source-frame targets do not cover all source views")

for i, src_v in enumerate(source_views):
    if not exists_match_in_src_targets(src_v):  # cam_idx first, c2w fallback
        raise ValueError(f"source view {i} has no matching target on source frame")
```

---

## 8. 日志与可观测性（沿用 4_3）

保留 4_3 既有日志，并新增 multi-src 相关指标：

- `num_source_views`
- `src_backproject_pass_count`（固定 1）
- `2d_bp_pairs_total`
- `2d_bp_pairs_after_threshold`
- `2d_bp_pairs_total_per_view`（可选）
- `2d_bp_pairs_after_threshold_per_view`（可选）
- `2d_bp_render_packed_ms_per_view`（可选）
- `2d_bp_fused_backproject_ms_per_view`（可选）

要求：只记录标量/短列表，不落大张量。

---

## 9. 测试计划

## 9.1 单元与数值一致性

1. **单源回归**：`V=1` 时，v3 输出与当前 v2 fused 完全一致  
2. **多源形状**：`V=2/3/6`，输出维度保持 `[N,C]` 与 `[N]`  
3. **累加等价**：手动 per-view 累加 == v3 streaming 输出  
4. **顺序鲁棒**：打乱 source views 顺序，输出仅有浮点噪声差异  
5. **反向可用**：`features_2d.requires_grad=True` 前后向可跑通  

## 9.2 Stage4_4 集成验证

在同一 segment 上对比：

- `num_source_views = 1/2/all`
- 记录 `acc_w_bg.mean / acc_w_distant.mean / acc_w_rigid_S.mean / acc_w_sky.mean`
- 记录 `mask_src_feat_valid_*` 比例与 `*_update_ratio`
- 观察 loss 稳定性与显存峰值变化

## 9.3 建议命令（与项目约定一致）

```bash
conda run -n drivestudio-new bash -lc 'PYTHONPATH=/root/drivestudio-coding pytest -q tests/streetforward'
```

若做单脚本冒烟：

```bash
conda run -n drivestudio-new bash -lc 'PYTHONPATH=/root/drivestudio-coding python tools/train_minimal_streetforward_stage4_4.py --config configs/minimal_streetforward_stage4_4.yaml --max_steps 5'
```

---

## 10. 实施顺序

1. 新增 `alpha_t_extractor_v3`（或在 v2 上增量并保留兼容别名）  
2. 新增 `minimal_trainer_stage4_4.py`，继承 4_3，最小改动接入 multi-src fast-fail  
3. 新增 `TrainSchedulerV5`（frame-level src/target，移除 overlap 计算链路）  
4. 新增 `configs/minimal_streetforward_stage4_4*.yaml`（基于 4_3 配置，切到 `scheduler_v5`）  
5. 接入训练脚本 `tools/train_minimal_streetforward_stage4_4*.py`（沿用 4_3 日志/评估框架）  
6. 完成单源回归与多源冒烟，再进入超参实验  

---

## 11. 最终结论

`Stage 4_4` 应定义为：

**“Stage 4_3 训练系统不变 + scheduler_v5 整帧采样 + source 多视角输入标准化 + v2 fused streaming 累加正式化”**。

它不是新算法分支，而是对现有 one-pass + fused 路径的稳定工程化扩展。  
这样改动最小、风险最低、与当前 `gsplat@5e9c30d` 兼容性最好。
