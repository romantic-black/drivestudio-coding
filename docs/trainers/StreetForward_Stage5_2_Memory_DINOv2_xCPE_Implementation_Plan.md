# StreetForward Stage5_2 实施方案（Memory 拆分 + DINOv2/UNet 融合 + xCPE 增强）

> 讨论范围对应代码：
> - `models/streetforward/struct_decoders/xcpe_decoder.py`
> - `datasets/train_scheduler_v8.py`
> - `configs/minimal_streetforward_stage5_2_multi_scene_v8.yaml`
> - `models/streetforward/minimal_trainer_stage5_2.py`
> - `models/streetforward/minimal_trainer_stage4_6.py`
> - `datasets/multi_scene_dataset_v4.py`
> - `models/feature_extractors/image_feature_extractor.py`

---

## 0. 硬约束（本方案不越界）

- 不做 `support_recent`。
- `record_views` 保持 `source_image_refs`。
- 2D 特征增强采用 `DINOv2-B/14 with registers + 现有 UNet 残差分支 + 融合 neck`。
- 增强 xCPE，但不推翻当前 Stage5_2 的 near/far routed 主体。

同时保持 Stage5_2 既有主链路不变：

```text
one-pass routed 2D evidence
-> near/far struct decoder
-> history-conditioned gate
-> routed GRU + shared heads
-> block_exit record
```

---

## 1. 目标与改动边界

本次改动分三部分，统一原则如下：

1. memory 只调整“记录时机”和“统计路径”，不改 gate 输入语义；
2. 2D 特征增强优先解决糊、暗、重叠区更糊；
3. xCPE 增强优先解决近处结构不足与局部几何传播不足。

明确不改：

- routed rigid 语义；
- rigid-specific head（不恢复）；
- far branch 的基本职责（仍是轻量结构分支）。

---

## 2. Memory 方案（核心）

## 2.1 目标语义

- **support memory**
  - 每 step 累积；
  - 使用 update 前 support；
  - `block_exit` 写 `support_ema`；
  - 不需要额外 render/backprojection。
- **residual memory**
  - `block_exit`；
  - 使用 update 后质量；
  - 需要一次 record pass；
  - `record_views = source_image_refs`。
- **update memory**
  - 每 step；
  - 使用 update 后实际 applied update；
  - 每 step 直接更新 `update_norm_ema`；
  - 不需要额外 render/backprojection。

## 2.2 当前实现与问题

当前 Stage5_2 的 `support_ema / error_ema / update_norm_ema` 更新路径耦合在 `record_block_history()`，且 `update_norm` 主要依赖 `last_step` 缓存，不符合“每 step update memory”的目标。

## 2.3 数据结构建议

在 `MinimalStreetForwardStage5_2.__init__()` 新增 block 内 support 累积缓存（三分支）：

- `stage5_2_block_support_bg`
- `stage5_2_block_support_distant`
- `stage5_2_block_support_rigid`

每个 key（通常是 batch key）对应：

```python
{
    "sum":   [N, 1],
    "count": [N, 1],
}
```

采用 `sum/count` 的原因：保持 support 的统计尺度稳定，避免步数变化引起漂移。

## 2.4 Support memory 实现

### 写入时机

在 `_compute_full_routed_gru_inputs()` 中执行 step 级累积（使用 update 前观测量）。

### 累积定义

- bg: `log1p(acc_w_bg)`
- distant: `log1p(acc_w_distant)`
- rigid: 使用 global rigid row-space 回写（`route.S`），保持与 `history_rigid` 对齐

### block_exit commit

在 `record_block_history()` 开头先做：

```text
support_block = sum / clamp(count, min=1)
```

然后使用 visible/invisible 的 beta 写入 `support_ema`，并更新 `initialized`。

> 由于“不做 support_recent”，当前 block 内 support 变化不即时反馈给 gate，这是本设计的显式 tradeoff，不视为 bug。

## 2.5 Residual memory 实现

保持 block_exit record pass 主语义：

- 保留 `_build_record_targets()` 的 `source_image_refs` 路径；
- 保留 `_compute_record_support_error_all_branches_once_routed(...)`；
- 仅将其 `error_*` 写入 `error_ema`；
- `support_*` 仅用于 residual 更新可见性掩码，不再写 `support_ema`。

## 2.6 Update memory 实现

将 `_update_last_step_update_norm_from_out()` 改为“每 step 直接写 history”，不再只缓存最后一步：

- 新增 `_apply_step_update_norm_ema(...)`；
- 从 applied delta（如 `means_r - means`）计算 `update_norm_cur`；
- 当本步点被更新时写 `update_norm_ema`。

同时在 `record_block_history()` 移除对 `last_step_update_norm` 的逻辑依赖。

## 2.7 函数拆分建议

新增函数：

- `_get_or_init_block_support_acc(...)`
- `_accumulate_support_before_update(...)`
- `_commit_block_support_to_history(...)`
- `_apply_residual_history_update(...)`
- `_apply_step_update_norm_ema(...)`
- `_clear_block_support_acc(...)`

保留函数：

- `_build_record_targets(...)`
- `_compute_record_support_error_all_branches_once_routed(...)`
- `_build_history_embed(...)`
- `_compute_gate(...)`

可删除/降级：

- `_apply_history_update(...)`（不再混合 support/error/update 三项）
- `stage5_2_last_step_update_norm`（若保留，仅日志用途）

`reset_node_state` 中需同步清理三套 block support cache。

## 2.8 与 SchedulerV8 的关系

无需改 `TrainSchedulerV8` 主逻辑。当前 step-major 终态补 `block_exit` 的行为满足本方案：  
support commit 与 residual record pass 都在 `block_exit` 完成。

---

## 3. History 配置建议（兼容旧配置）

推荐升级为嵌套结构：

```yaml
history_memory:
  enable: true
  record_on: "block_exit"
  record_views: "source_image_refs"

  support:
    accumulate_each_step: true
    use_pre_update_support: true
    ema_beta_visible: 0.75
    ema_beta_invisible: 0.90

  residual:
    enable: true
    use_post_update_record_pass: true
    error_beta: 0.75
    error_eps: 1.0e-6

  update:
    enable: true
    record_each_step: true
    use_post_update_applied_delta: true
    ema_beta: 0.85
```

兼容 fallback（旧 flat key）：

- `support.ema_beta_visible -> support_beta_visible`
- `support.ema_beta_invisible -> support_beta_invisible`
- `residual.error_beta -> error_beta`
- `residual.error_eps -> error_eps`
- `update.ema_beta -> update_norm_beta`

---

## 4. 2D Feature 方案（DINOv2 + UNet + Fusion Neck）

## 4.1 设计目标

- DINOv2 分支：提供更强语义纹理与跨视角稳定性；
- UNet 分支：保留现有 residual-aware 局部细节；
- Fusion neck：输出统一 2D 特征供 backprojection 使用。

## 4.2 建议结构

- 分支 A（DINO）：输入 `RGB source image`（3ch）
- 分支 B（UNet）：输入 `[RGB_gt, RGB_rendered]`（6ch）
- 分支 C（Fusion neck）：`concat(dino_feat, unet_feat) -> feat_2d_channels=48`

建议通道：

- DINO branch 输出 32
- UNet branch 输出 32
- Fusion neck 输出 48

> 48 作为首版比 64 更稳，先控制显存与训练耦合复杂度。

## 4.3 模块拆分

建议新增文件：`models/feature_extractors/dinov2_unet_fusion.py`

包含：

- `DINOv2BackboneAdapter`
- `FusionNeck2D`
- `DINOv2UNetFusionExtractor`

## 4.4 DINO adapter 关键点

- 输入：`[B,3,H,W]`
- 规范化、pad 到 14 的倍数
- 读取中间层（建议 `[4, 8, 11]`）
- **丢弃 cls token 与 register tokens，仅保留 patch tokens**
- reshape 为 2D feature map 后上采样对齐目标分辨率
- 多层融合输出到 32 通道

## 4.5 UNet 分支复用

复用 `ImageFeatureExtractor`，保持当前 6ch residual 输入策略，减少现有链路扰动。

## 4.6 Fusion neck 建议

```text
concat(dino, unet)
-> 3x3 conv + GELU + norm
-> 3x3 conv + GELU
-> 1x1 conv
-> out (48ch)
```

## 4.7 与 Stage5_2 对接约束

必须保持：

- `model.feat_2d_channels == struct_decoder.feat_2d_channels`

因此需同步改为 48：

```yaml
model:
  feat_2d_channels: 48
  struct_decoder:
    feat_2d_channels: 48
```

## 4.8 训练策略建议

- `0 ~ 20k`：冻结 DINO backbone，仅训 UNet + fusion + StreetForward；
- `20k ~ 100k`：仅解冻 DINO 最后 2 个 block；
- param group：`dino_backbone lr = 0.1 * base_lr`，其余 `base_lr`。

---

## 5. xCPE 增强方案

## 5.1 一期（建议先落地）

在不改 near/far 主体前提下增强 near xCPE：

```yaml
near:
  channels: 96
  voxel_size: 0.20
  xcpe:
    num_layers: 2
    kernel_size: 3
    residual_scale_init: 5.0e-3
```

理由：

- `num_layers: 1 -> 2`：提高局部传播；
- `channels: 64 -> 96`：提升结构混合容量；
- `residual_scale_init: 1e-3 -> 5e-3`：加快结构分支有效介入。

## 5.2 二期（可选）

若一期后近处仍不足，再引入 multi-scale xCPE：

- fine: `voxel_size=0.15, channels=64`
- coarse: `voxel_size=0.30, channels=64`
- `concat + linear` 融合输出

建议新增：`models/streetforward/struct_decoders/xcpe_ms.py`，并在配置校验中允许 `near.type=xcpe_ms`。

---

## 6. 配置草案（整套收敛版）

```yaml
model:
  stage: "5_2"
  feat_2d_channels: 48
  feat_2d_downscale: 1

  feature_extractor:
    type: dinov2_unet_fusion
    dino:
      model_name: dinov2_vitb14_reg
      pretrained: true
      freeze_steps: 20000
      unfreeze_last_n_blocks: 2
      out_channels: 32
      intermediate_layers: [4, 8, 11]
      pad_to_patch_multiple: 14
    residual_unet:
      in_channels: 6
      feat_channels: 32
      base_channels: 32
      feature_downscale: 1
      depth: 4
      bilinear: true
    fusion:
      hidden_channels: 64
      out_channels: 48

  struct_decoder:
    enable: true
    type: "routed_near_far"
    scope: "full_routed"
    output_role: "gru_input"
    point_preserving: true
    include_bg: true
    include_distant: true
    include_rigid_in: true
    include_rigid_out: true
    feat_2d_channels: 48
    param_embed_dim: 32
    branch_embed_dim: 8
    support_embed_dim: 8
    history_embed_dim: 16
    token:
      use_2d_feat: true
      use_support: true
      use_branch_embed: true
      use_param_embed: true
      use_anchor_rgb: false
      use_hidden_state: false
      zero_invalid_2d_feat: true
    near:
      type: "xcpe"
      branches: ["bg", "rigid_in"]
      channels: 96
      voxel_size: 0.20
      sparse_backend: "spconv"
      clamp_grid_coord: false
      xcpe:
        num_layers: 2
        kernel_size: 3
        residual_scale_init: 5.0e-3
        norm: "layernorm"
        act: "gelu"
    far:
      type: "mlp"
      branches: ["distant", "rigid_out"]
      channels: 64
      hidden_dim: 64
      num_layers: 2
      norm: "layernorm"
      act: "gelu"

  history_memory:
    enable: true
    record_on: "block_exit"
    record_views: "source_image_refs"
    support:
      accumulate_each_step: true
      use_pre_update_support: true
      ema_beta_visible: 0.75
      ema_beta_invisible: 0.90
    residual:
      enable: true
      use_post_update_record_pass: true
      error_beta: 0.75
      error_eps: 1.0e-6
    update:
      enable: true
      record_each_step: true
      use_post_update_applied_delta: true
      ema_beta: 0.85
```

---

## 7. 推荐落地顺序

1. **先改 memory**：support step 累积 + block_exit commit；residual block_exit record；update step EMA。
2. **再上 2D fusion**：先冻结 DINO，`feat_2d_channels=48`。
3. **增强 near xCPE**：`96 channels / 2 layers / 5e-3`。
4. 仅在一期仍不足时，上 multi-scale xCPE。

---

## 8. 一句结论

最稳妥路线是：**先把 memory 统计语义拆对，再上 DINOv2-B/14 reg + 现有 6ch UNet 残差分支 + 48ch fusion neck，最后增强 near xCPE**。  
该路线与当前 Stage5_2 代码组织和 fast-fail 风格兼容，且不会破坏 routed near/far 主体和 `record_views=source_image_refs` 约束。
