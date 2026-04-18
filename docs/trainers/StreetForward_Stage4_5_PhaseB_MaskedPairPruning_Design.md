# StreetForward Stage4.5 Phase B 设计（修订版）：Masked Pair Pruning 的真实落点与实现路线

## 1. 先澄清当前主链（关键前提）

当前仓库 Stage4.5 的 source 2D 主链是 direct-fused 路径：

- `MinimalStreetForwardStage4_5._backproject_scene_features_multi_camera(...)`
- `AlphaTWeightExtractorV3.render_and_backproject_streaming_fused_multi_camera(...)`
- `rasterize_and_backproject_3dgs_multi_camera_kernel`（直接做 accumulation）

这条主链**不是** `rasterize_to_indices_3dgs -> pair list -> 再消费 pair` 的默认路径。  
因此：

- 只修改 `rasterize_to_indices_3dgs`，对当前 Stage4.5 主训练路径可能**不生效**；
- 若要获得稳定收益，需要先明确“改 direct-fused 主链”还是“把 Stage4.5 切到 pair-list 主链”。

---

## 2. 本文目标（修订）

在 Stage4.5 语义下（no sky node/no sky render），把 source mask 从 image-zeroing 迁移到几何证据层，目标是：

- invalid source pixel 不提供 2D 证据；
- `acc_w/support` 只统计 valid source 区域；
- 在可行情况下减少 pair 写出与后续 pair 消费成本。

注意：本文不再宣称“仅改 `rasterize_to_indices_3dgs` 就能优化当前 4_5 主链”。

---

## 3. 术语与 canonical 语义

定义 `source_pair_valid_mask`：

- 语义：`1=valid`, `0=invalid`
- 建议组合：`valid = (~egocar_mask) & (~sky_mask)`

source CNN 输入语义（Stage4.5 canonical）：

- `source RGB` 不做 mask 置零；
- mask 仅用于几何-像素证据过滤。

---

## 4. Phase A / Phase B 的准确边界（按当前代码）

## 4.1 Phase A（当前可落地主线）

在 direct-fused accumulation kernel 内做 masked rejection（invalid pixel early return / no contribution）。

特点：

- 与 Stage4.5 当前主链直接对齐；
- 能修正语义并带来中等优化；
- 不需要切换到 pair-list 主链。

## 4.2 Phase B（本文主题）

把 mask 下沉到 pair 生成层（`rasterize_to_indices_3dgs` 两阶段 count/write），让 invalid pixel 不生成 pair。

但要强调：

- 这是 pair-list 主链优化；
- 若 Stage4.5 不切到 pair-list 主链，Phase B 不会自动生效。

---

## 5. 两条可执行路线（必须二选一）

## 路线 A（推荐优先）：保持 direct-fused 主链，Phase B 改写到 direct-fused 前半段

做法：

- 不依赖 `rasterize_to_indices_3dgs`；
- 直接在 `render_and_backproject_streaming_fused_multi_camera` 的 kernel 路径里引入 `source_pair_valid_mask`；
- 在可见性遍历与 accumulation 前执行 invalid pixel rejection。

优点：

- 与当前 Stage4.5 主链一致；
- 改动面更聚焦，收益可直接验证。

## 路线 B（纯 Phase B 原义）：切换 Stage4.5 到 pair-list 主链，再改 `rasterize_to_indices_3dgs`

做法：

1. Stage4.5 明确切换到 pair-list backprojection pipeline；
2. 对 `rasterize_to_indices_3dgs` count/write 加 mask 过滤；
3. 后续 pair consumer 维持现有逻辑。

前提：

- 需要重构 AlphaTWeightExtractorV3 的主调用链；
- 否则“改了 pair 生成”但训练仍走 direct-fused，收益为零。

---

## 6. 关于 `flatten_ids` 的修正结论

`flatten_ids` / `tile_offsets` 来自更早的 projection/intersection/tile 阶段。  
在 `rasterize_to_indices_3dgs` 增加 mask，通常只能减少：

- `gaussian_ids / pixel_ids / weights / n_elems`

不会减少（或不直接减少）：

- projection 成本
- `tile_offsets / flatten_ids` 生成成本
- 遍历整个像素网格的基础触达成本

因此性能叙述应保守：主要收益来自 pair 写出与后续 pair 消费规模下降，而非“最源头全链路大幅提速”。

---

## 7. API 语义：generic 与 Stage4.5-specialized 必须区分

`rasterize_to_indices_3dgs` 现有语义是通用 image_dims（可含 I 维），不是 Stage4.5-only camera 语义。  
因此文档/实现必须明确一种策略：

## 7.1 方案 G（保留 generic，推荐）

- mask 形状约定为 `[I, H, W]`（或可广播到该形状）；
- kernel 通过 `image_id` 索引，不写死 `cam_id*H*W`；
- Stage4.5 仅在 wrapper 把 `[V,H,W]` 适配成 `[I,H,W]`。

## 7.2 方案 S（Stage4.5 专用特化）

- 明确声明该 API 被收窄为 Stage4.5 source 多相机专用；
- 所有非 Stage4.5 调用点需迁移或禁用。

本文建议使用 **方案 G**，避免误伤通用调用方。

---

## 8. `pair_count_total_before_mask` 指标的实现现实

该指标不是“顺手可得”，需要额外工作：

- debug-only 再跑一套 before 统计，或
- 在同一 pass 里同时计 before/after（增加分支/原子开销）。

建议：

- 默认只上 `after` 系列与 `valid_ratio`；
- `before` 统计仅在 profiling build / sampling step 打开；
- 避免让“统计开销”吞掉优化收益。

---

## 9. 推荐实现顺序（按当前仓库最稳妥）

1. **先锁定路线**：A（direct-fused）或 B（切 pair-list 主链）。  
2. 若选 A：先在 AlphaTWeightExtractorV3 direct-fused kernel 路径接 `source_pair_valid_mask`。  
3. 若选 B：先完成主链切换，再改 `rasterize_to_indices_3dgs` count/write。  
4. 加最小指标集：`masked_pixel_count`, `valid_pixel_count`, `source_pair_valid_ratio`, `pair_count_after_mask`。  
5. 再做 A/B 数值对照与微基准。

---

## 10. 验收标准（修订）

- 语义：
  - invalid source pixel 不再贡献 feature/support；
  - `acc_w` 仅反映 valid source 证据；
- 路径：
  - 文档与代码一致说明当前主链（direct-fused or pair-list）；
  - 不能出现“改了未被调用路径却宣称生效”；
- 性能：
  - 至少证明 pair 规模或后续 pair 消费下降；
  - 对总时延只做保守结论，不夸大。

---

## 11. 预期收益（保守版）

- **高确定性收益**：语义更正确（support/feature 与 valid source 区域一致）。  
- **中等确定性收益**：pair 写出和后续 pair 消费下降（仅在相关主链生效）。  
- **不保证收益**：projection / tile intersection 相关成本显著下降（通常不变）。  

---

## 12. 结论

“mask 下沉到几何证据层”方向正确；但对当前 Stage4.5，必须先对齐主链事实。  
Phase B 若定义为改 `rasterize_to_indices_3dgs`，请先明确是否把 Stage4.5 切到 pair-list 主链；否则建议优先在 direct-fused 主链完成等价的 masked rejection/pruning。

