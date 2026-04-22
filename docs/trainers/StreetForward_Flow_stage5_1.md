# StreetForward Flow（Stage5_1 KNN Attention 设计方案）

本文定义 `Stage5_1` 的目标边界、数据契约与实现步骤。该设计以 `Stage5_0` 为基线，只新增 route-aware fixed KNN attention，不修改 Stage4_6/5_0 的 routed rigid 语义。

设计原则：

- fast-fail：配置与输入不满足约束时立即报错。
- 不考虑向后兼容：不为 legacy 行为保留分支。
- 复杂度稳定：不更新 KNN、不做动态 compaction、不做 per-instance Python 循环，保持 `O(N_struct * K)`。

---

## 1. 一句话定义

`Stage5_1 = Stage5_0 + route-aware fixed KNN attention`。

保留 Stage5_0 主语义：

- `P_struct = bg ∪ rigid.S_in`
- struct decoder 输出仍仅服务 `GRU input`

保留 Stage4_6 routed rigid 规则：

- rigid in -> bg heads
- rigid out -> distant heads

因此 Stage5_1 明确禁止以下行为：

- 把 `rigid_out`/`distant` 引入 struct token
- 为 KNN 做 runtime 更新或 refill
- 通过变长 attention 或 compaction 改写计算图

---

## 2. 核心机制：fixed-shape self-only fallback

### 2.1 背景问题

预处理 rigid KNN 通常基于完整 rigid 点集：

- `rigid_knn_idx_global: [N_rigid, K_store]`

但 runtime 参与 struct 的 rigid 仅有 `rigid.S_in`。因此某个 `S_in` query 的邻居可能落在 `S_out`。

### 2.2 处理规则

对每个 rigid-in query：

- 邻居属于 `rigid.S_in`：保留为有效邻居。
- 邻居属于 `rigid.S_out`（或无效）：
  - 不加入 `P_struct`
  - 不读取 distant feature
  - `neighbor_idx` 安全替换为 self row
  - `neighbor_mask = false`
- 若有效 non-self 邻居过少：整行退化为 self-only。

### 2.3 固定形状不变量

所有 query 始终输出固定形状：

- `neighbor_idx: [N_struct, K]`
- `neighbor_mask: [N_struct, K]`

固定槽位约定：

- `slot 0` 永远是 self：`neighbor_idx[:, 0] = self_row`，`neighbor_mask[:, 0] = true`
- self-only 行只重写 mask 与 index，不引入新分支：
  - `neighbor_mask[i, 1:] = false`
  - `neighbor_idx[i, 1:] = self_row_i`

---

## 3. KNN 资产格式（强约束）

建议 segment 资产中保存：

- `bg_knn_idx: int32 [N_bg, K_store]`
- `rigid_knn_idx: int32 [N_rigid, K_store]`

其中：

- `K = knn_attention.k`
- `K_store = K - 1`（runtime prepend self slot）

约束：

- `bg_knn_idx` 使用 bg-local row index，范围 `[0, N_bg)`
- `rigid_knn_idx` 使用 rigid global row index，范围 `[0, N_rigid)`
- 不在 forward 处理中使用 per-instance local KNN；如资产旧格式为 instance-local，必须在 asset load 阶段一次性展开并 fast-fail 校验

### 3.1 当前代码现状检查结论（基于现有导出/导入/preload链路）

已检查文件：

- `tools/build_streetforward_segment_knn_assets.py`
- `datasets/streetforward_assets/asset_store.py`
- `datasets/multi_scene_dataset_v4.py`
- `datasets/asset_preload_manager_v2.py`

结论：

1. **导出能力现状**
   - 现有导出脚本只生成 `background_avg_dist_by_k` 与 `dynamic_avg_dist_by_k`（用于 scale init）。
   - 没有导出 Stage5_1 attention 所需的固定邻接索引（`bg_knn_idx` / `rigid_knn_idx`）。
   - 因此现状 **不满足** Stage5_1 fixed cached neighbor 的最小要求。

2. **导入能力现状**
   - `asset_store` 的 `knn_init.npz` schema（当前 `schema_version=1`）只序列化/反序列化 avg-distance map。
   - 没有邻接索引字段，也没有邻接索引一致性校验。
   - 因此现状 **不满足** Stage5_1 对邻接索引资产的导入需求。

3. **preload 能力现状**
   - `AssetPreloadManagerV2` 会通过 `segment_static` 预热触发 `_resolve_segment_bundle`。
   - `SegmentStaticBundle` 已携带 `knn_init`，所以“资产级预热链路”本身是存在的。
   - 但由于 `knn_init` 当前不含邻接索引，preload 只能预热 avg-distance，不足以直接服务 Stage5_1 attention。

4. **与 fixed cached 语义冲突点**
   - `MultiSceneDatasetV4` 里存在 runtime pointcloud downsample 路径，并会同步裁剪 `knn_init` 的 avg-distance 数组。
   - 该路径适配 scale init 是合理的，但对于 Stage5_1 固定邻接索引会破坏“导出即固定”的语义边界。
   - Stage5_1 推荐：启用 KNN attention 时对 runtime cap mismatch 直接 fast-fail，禁止 runtime downsample + 重映射。

### 3.2 文档修正后的最低资产要求

Stage5_1 必须新增并使用“邻接索引资产”，建议：

- 在 segment 资产新增 `knn_neighbors.npz`（或把字段扩展到 `knn_init.npz` 的新 schema 版本）。
- 至少包含：
  - `bg_knn_idx: int32 [N_bg, K_store]`
  - `rigid_knn_idx: int32 [N_rigid, K_store]`（rigid global row）
- 可选附带：
  - `knn_k_store`
  - `pointcloud_fingerprint`（用于 fast-fail 对齐）

强约束：

- Stage5_1 不允许“仅有 avg-distance map 但无邻接索引”继续训练。
- Stage5_1 不允许 runtime 基于 cap 变更重采样点云后再尝试修补邻接索引。

---

## 4. StructDecoderInput 扩展契约

在 `StructDecoderInput` 中新增两个 optional 字段（供 Stage5_1 使用）：

- `neighbor_idx: Optional[torch.Tensor]`，shape `[N, K]`，整型
- `neighbor_mask: Optional[torch.Tensor]`，shape `[N, K]`，bool

行为约束：

- Stage5_0 decoder 可忽略这两个字段
- Stage5_1 decoder 必须要求这两个字段存在且 shape 严格匹配，否则立即 `raise RuntimeError`

---

## 5. 代码结构建议

新增文件：

- `models/streetforward/minimal_trainer_stage5_1.py`
- `models/streetforward/struct_decoders/knn_attention.py`
- `models/streetforward/struct_decoders/xcpe_knn_decoder.py`

更新文件：

- `models/streetforward/struct_decoders/common.py`
- `models/streetforward/struct_decoders/__init__.py`
- `tools/train_minimal_streetforward_stage5_1_multi_scene_v7.py`
- `configs/minimal_streetforward_stage5_1_multi_scene_v7.yaml`

---

## 6. Trainer 设计（继承 Stage5_0）

建议定义：

- `class MinimalStreetForwardStage5_1(MinimalStreetForwardStage5_0)`

理由：

- Stage5_0 已完成 `bg + rigid.S_in` struct 输入构造
- Stage5_0 已完成 struct 输出切回 `bg` / `rigid-in` 并进入 GRU
- Stage5_1 只需注入固定 KNN 邻接，并替换 decoder 类型

---

## 7. 配置 fast-fail 规则

`Stage5_1` 应增加专属校验，并保持 Stage5_0 约束不回退：

- `model.stage` 必须为 `"5_1"`
- `model.struct_decoder.type` 必须为 `"xcpe_knn_attn"`
- `knn_attention.enable` 必须为 `true`
- `include_distant` 必须为 `false`
- `include_rigid_out` 必须为 `false`
- `neighbor_policy` 必须为 `"fixed_cached"`
- `out_neighbor_policy` 必须为 `"mask_self_fallback"`

任一不满足都应直接 `ValueError`，不做自动修正。

---

## 8. Runtime neighbor 构建

### 8.1 输入输出定义

输入：

- `num_bg: int`
- `route.S_in: [N_rigid_in]`
- `bg_knn_idx: [N_bg, K_store]`
- `rigid_knn_idx: [N_rigid, K_store]`

输出：

- `neighbor_idx: [N_bg + N_rigid_in, K]`
- `neighbor_mask: [N_bg + N_rigid_in, K]`

### 8.2 bg 构建规则

- 对 `bg_knn_idx` 做边界与非 self 校验
- 非法邻居重写为 self row + `mask=false`
- `slot 0` 强制 self

### 8.3 rigid-in 构建规则（关键）

核心步骤：

1. 构造 `rigid_global_to_struct`，仅对 `S_in` 写入 struct row，其余保持 `-1`。
2. 将 query（`route.S_in`）对应的 `rigid_knn_idx` 映射到 struct row。
3. `mapped >= 0` 代表邻居仍在 `S_in`，否则视为 out/invalid。
4. 若 `valid_count < min_valid_neighbors`，整行 `force_self_only`。
5. 通过 `torch.where` 完成固定形状安全改写，禁止 compaction。

输出附带统计：

- `rigid_out_neighbor_ratio`
- `rigid_self_only_ratio`
- `rigid_valid_neighbor_mean`

### 8.4 合并 bg + rigid-in

- `neighbor_idx = cat([bg_idx, rigid_idx], dim=0)`
- `neighbor_mask = cat([bg_mask, rigid_mask], dim=0)`
- 写入 `struct_in.neighbor_idx` / `struct_in.neighbor_mask`
- 将 KNN 指标写入 `aux`

---

## 9. `_compute_bg_rigid_in_gru_inputs` 的改造点

相对 Stage5_0，流程仅新增一段：

1. 先构造 `StructDecoderInput`（保持原顺序：`bg` 在前，`rigid.S_in` 在后）
2. 构造 fixed KNN `neighbor_idx/mask`
3. 注入 `struct_in`
4. 调用 `struct_decoder`
5. 按 `split_bg/split_rigid_in` 切分输出，进入 GRU

实现约束：

- 优先把 `batch` 显式作为参数传递到函数链路，避免通过临时成员变量传递上下文

---

## 10. `StreetForwardXCPEKNNDecoder` 设计

建议新建独立 decoder 类，不在 Stage5_0 的 `StreetForwardXCPEDecoder` 里堆条件分支。

推荐流程：

1. token builder（与 Stage5_0 一致）
2. xCPE 层（与 Stage5_0 一致）
3. KNN attention 层（Stage5_1 新增）
4. output projection（输出到 GRU 输入维）

输入强校验：

- `neighbor_idx` / `neighbor_mask` 必须存在
- `neighbor_idx.shape[0] == N_points`
- `neighbor_mask.shape == neighbor_idx.shape`

---

## 11. EdgeGatedKNN Attention 模块

不采用标准 QK dot-product 的主要原因：

- 需求重点是局部相对差异、相对位置、support 与 branch 一致性
- 非全局上下文竞争

采用 edge-gated 评分：

- `score_ij = f(x_i, x_j - x_i, rel_pos, support_j, same_branch)`
- `alpha_ij = softmax(score_ij over K)`
- `out_i = sum_j alpha_ij * (value_j + pos_value_ij)`

实现建议（省显存）：

- 使用 additive hidden score 组合，避免显式大 concat
- chunk 化按 `N` 处理，控制峰值显存

self-only 行无需专门分支：

- 因为仅 `slot 0` 有效，softmax 后自然得到 `alpha_self=1`

### 11.1 self-only 机制的工程风险与修正规则（速度/潜在bug）

以下仅讨论 forward 稳定性、潜在 bug 与性能风险，不讨论模型效果。

1. **全 false mask 风险（高优先级）**
   - 风险：如果某次构图错误导致某行 `neighbor_mask` 全为 false，softmax 行为会退化为错误分布或数值异常。
   - 修正规则：在进入 attention 前强制校验并 fast-fail：
     - `neighbor_mask[:, 0].all()` 必须为 true
     - 每行 `neighbor_mask.sum(dim=1) >= 1`
   - 这是必须项，不能依赖“理论上 slot0 恒 true”。

2. **索引越界风险（高优先级）**
   - 风险：`neighbor_idx` 任一元素越界会触发 gather 崩溃或 silent 非法访问。
   - 修正规则：进入 decoder 时做一次批量边界检查：
     - `0 <= neighbor_idx < N_struct`
   - 失败立即报错，不做截断容错。

3. **dtype 额外开销（中优先级）**
   - 风险：若资产存 `int32`，每层/每次 forward 再 `.long()` 会引入重复转换开销。
   - 修正规则：neighbor 构建阶段统一转为 `torch.long` 并缓存，decoder 内不重复 cast。

4. **非连续内存导致 gather 变慢（中优先级）**
   - 风险：切片/拼接后 `neighbor_idx`、`neighbor_mask` 可能非 contiguous，影响索引访问效率。
   - 修正规则：进入 attention 前执行 `.contiguous()`（一次）。

5. **高 self-only 比例的固定计算开销（已接受）**
   - 现象：即使 self-only 行很多，固定形状实现仍会计算 `K` 槽位，这是设计上接受的“稳定复杂度换实现简洁”。
   - 约束：不为此引入变长分支或 compaction，保持 `O(N*K)` 稳定路径。

6. **chunk 配置不当导致性能抖动（中优先级）**
   - 风险：`chunk_size` 过小会增加循环与 kernel launch 开销，过大可能导致显存压力。
   - 修正规则：提供保守默认值并可配置；运行日志记录实际 chunk 行为用于调参。

---

## 12. 配置建议（第一版）

建议关键项：

- `model.stage: "5_1"`
- `struct_decoder.type: "xcpe_knn_attn"`
- `knn_attention.k: 16`（=> `K_store=15`）
- `knn_attention.min_valid_neighbors: 2`
- `xcpe.num_layers: 1`
- `knn_attention.num_layers: 1`
- `residual_scale_init: 1e-3`

第一版目标是验证“局部竞争关系补足”的收益，不建议先堆深网络。

---

## 13. 性能与复杂度保证点

必须保持：

- 不做 KNN 更新
- 不做 neighbor refill
- 不做 compaction
- 不做 per-instance / per-query 分支循环
- attention 输入始终固定 `[N_struct, K]`

复杂度稳定为：

- `O(N_struct * K * C)`

---

## 14. 指标记录（必须）

建议至少记录：

- `stage5_1_knn_k`
- `stage5_1_knn_bg_valid_neighbor_mean`
- `stage5_1_knn_rigid_valid_neighbor_mean`
- `stage5_1_knn_rigid_out_neighbor_ratio`
- `stage5_1_knn_rigid_self_only_ratio`
- `stage5_1_knn_attn_residual_scale`

其中最关键的是：

- `rigid_out_neighbor_ratio`
- `rigid_self_only_ratio`

解释建议：

- `out_neighbor_ratio` 高：多为 segment 边界切分导致的拓扑截断
- `self_only_ratio` 高：多为 routed `S_in` 有效邻居不足，应先排查数据/切分，再调模型

---

## 15. 测试计划（最小集）

建议覆盖：

1. pure bg：`N_rigid=0`，校验 `neighbor` 形状与 slot0 不变量
2. rigid 全 in：`S_out=empty`，校验 self-only ratio 接近 0
3. rigid half in/out：校验 out 邻居被 mask+self 替换
4. 邻居不足：触发 self-only，校验整行 rewrite
5. no KNN update：多步训练后 KNN 资产保持不变

---

## 16. 推荐落地顺序

1. 扩展 `StructDecoderInput`（新增 `neighbor_idx/mask`）
2. 新增 `knn_attention.py`（`EdgeGatedKNNAttention`）
3. 新增 `xcpe_knn_decoder.py`（xCPE 后追加 KNN attention）
4. 新增 `MinimalStreetForwardStage5_1`（继承 Stage5_0）
5. 接入 segment KNN 资产（只读）
6. 加入 route-aware remap 与 self-only fallback
7. 跑单元测试与小场景 overfit

---

## 17. 结论

Stage5_1 的正确边界是：

- `P_struct` 只包含 `bg ∪ rigid.S_in`
- rigid-out 邻居不转 token、不借 distant feature
- 仅通过 fixed-shape mask/self rewrite 实现 fallback

该方案在不破坏 Stage4_6/5_0 语义前提下，把局部竞争关系注入 GRU 输入，并维持稳定的训练与推理复杂度。
