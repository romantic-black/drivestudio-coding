# StreetForward Stage 4.1 模型方案（Rigid 多 Target：Render/Update 解耦）

本文档定义 Stage 4.1 的最小可行升级：在 Stage 4.0 基础上引入 `1 src + K targets`，并将 rigid 分支中的“可渲染（renderable）”与“可更新（updatable）”彻底解耦。

**本版 MVP 的工程取舍（已拍板）：**

1. **实例级标注可见性**：`mask_src_rigid`、`mask_tgt_rigid[F]` 仍由 **`instances_fv`** 与 `point_ids` 扩展得到；**不**在 MVP 中实现点级投影/视锥裁剪可见性矩阵。
2. **更新资格收紧**：在实例级「可在该帧出现」之上，引入 **`mask_src_feat_valid`**（source 上确实拿到有效 2D 特征的点），并令  
   `mask_update_rigid = mask_src_feat_valid & mask_any_tgt_rigid`。
3. **渲染**：按 **target 帧** 组织；每一帧内将**该帧所有相机**对应的 `targets` 条目凑成一批，优先调用 `MinimalStreetForwardStage2_2._render_multi_view` **一次 gsplat**；同帧 `(H, W)` 不一致时按分辨率分组或 fallback `_render_single_view`。

与点级方案、批量渲染可行性、以及「实例级不能语义替代点级」的讨论见 [StreetForward_Stage4_1_InstanceVisibility_and_PerFrame_BatchedRender.md](StreetForward_Stage4_1_InstanceVisibility_and_PerFrame_BatchedRender.md)。

Stage 4.1 仍保持以下前提不变：

1. rigid 仍是 **2D-only features**（更新只依赖 source 侧 2D 观测）。
2. rigid NodeState 仍存 **local 坐标**，渲染前再按 `dynamic_info + frame_idx` 变换到各 target 的 world 坐标。
3. 不引入多-source，不在本版本解决大视差新暴露表面补全。

---

## 1. 核心判断：4.1 的难点不是变换，而是更新资格

4.0 到 4.1 的主要变化不是“如何把 rigid 点变到多个 target”，而是：

- 某 rigid 点可能在某个 target 帧参与渲染；
- 但若该点在 source 侧**没有可靠 2D 特征**，则这一整轮**不应**更新 offset / GRU hidden / NodeState（可与「该点仍要在 target 上被画出来」共存）。

因此：**渲染集合**与**更新集合**在规范上必须分开；更新集合由 **`mask_src_feat_valid`** 与 **至少一个 target 帧实例有效** 共同限定。

---

## 2. Mask 体系（规范）

设 rigid 总点数为 `N_rigid`，唯一 target 帧集合为 `{F}`。除 `mask_tgt_rigid[F]` 可按帧变化外，其余为长度 `N_rigid` 的全局向量。

### 2.1 保留的四类“语义可见性”（实例级扩展）

以下均由 **`instances_fv[frame_slot, instance_id]`** 与 `point_ids` 扩展为点向量，**不含**每视角投影几何。

| 符号 | 含义 |
|------|------|
| `mask_src_rigid` | source 帧上实例标注为有效 → 点有效 |
| `mask_tgt_rigid[F]` | target 帧 `F` 上实例有效 → 点有效（**该帧所有相机共用**） |
| `mask_any_tgt_rigid` | `OR_F mask_tgt_rigid[F]` |
| `mask_update_rigid` | **见 2.2，与 `mask_src_feat_valid` 联立** |

### 2.2 新增：`mask_src_feat_valid`

- **定义**：source 上**确实拿到有效 2D 特征**的点（与实例「在场」不同）。
- **形状**：`[N_rigid]` bool。
- **实现约定（当前代码路径）**：**仅**使用 αT 反投影管线中、与加权特征同一套累加量上的 **`accumulated_weight`（各源视图画素上 αT 权重对点的 scatter-add 之和）** 与配置阈值 `model.branches.rigid.src_backproject_support_min` 比较。  
  **禁止**用 `||feat||`、特征是否有限、或其它与反投影支持度无关的启发式推断 `mask_src_feat_valid`；也不得静默把「无支持」当全零特征混入 GRU。

推荐关系（实例级为必要条件、特征有效为收紧条件）：

- 实现上应保证：`mask_src_feat_valid ⇒ mask_src_rigid`（至少不应对「实例已判无效」的点标特征有效）；若某点在 `mask_src_rigid` 外却特征非零，应视为数据/逻辑错误并 **fast-fail**。

### 2.3 更新 mask（最终）

\[
\texttt{mask\_update\_rigid} = \texttt{mask\_src\_feat\_valid} \land \texttt{mask\_any\_tgt\_rigid}
\]

不再使用 `mask_src_rigid & mask_any_tgt_rigid` 作为更新门控（旧稿）；**标注在场但 feature 无效**的点可以进 target 渲染，但**不得**进入更新路径。

---

## 3. 每个 target 帧的 rigid 子集（显式拆分）

对每个 target 帧 `F`：

- `idx_trainable[F] = nonzero(mask_update_rigid & mask_tgt_rigid[F])`
- `idx_frozen[F]    = nonzero((~mask_update_rigid) & mask_tgt_rigid[F])`

性质：

- `idx_trainable[F]` 与 `idx_frozen[F]` **在同一帧不应重叠**；并集为当帧参与 rigid 渲染的点（若两者皆空，则该帧无 rigid 高斯，仅 bg+distant，属正常，见 §7 loss）。

可选：`idx_all_render[F] = nonzero(mask_tgt_rigid[F]) = cat(trainable, frozen)`（实现上用两次索引或一次 mask 均可）。

---

## 4. 渲染来源（trainable vs frozen）

对每个帧 `F`，将 rigid 拆成两块再变换到该帧 world、与 bg / distant 合并：

| 子集 | 局部参数来源 | 梯度 |
|------|----------------|------|
| **trainable** | 当前 step 的 **可微** `render_params_rigid_local`（经 GRU/offset 后的那份） | 通向 rigid 头与上游 |
| **frozen** | **`NodeState` 的 local 参数**（`means/scales_log/quats/opacity/sh_*`）**显式 `detach()`** | 不更新 rigid 预测器 |

- 二者在 world 系下用**同一套** `dynamic_info` 与 `F` 做 `local → world`，再 concat 成该帧的 `rigid_world_for_merge`。
- bg、distant 仍按原 Stage 4.x proxy 流程；**仅 rigid** 在当帧发生 trainable/frozen 双源拼接。

---

## 5. Hidden、offsets、写回（scatter 规则）

### 5.1 Offsets / GRU

- 仅对 **`mask_update_rigid`** 上的点计算并应用更新：`feat` → GRU → offsets → `render_params_rigid_local`。
- 对 `~mask_update_rigid`：offsets 为零、`offset_quat` identity、`h_new = h_old`（与旧稿一致）。

### 5.2 Hidden

- **只对 `mask_update_rigid` scatter 写入新 hidden**；其余位置 **严格保留** `h_old`（禁止整表覆盖导致未监督点漂移）。

### 5.3 NodeState writeback

- **只对 `nonzero(mask_update_rigid)`** 将本 step 的 local `render_params` 写回 `NodeState`；
- 其余 rigid 点 **完全不写**，保持历史 NodeState。

---

## 6. Loss（帧内平均 → 帧间平均；空集安全）

对每一 target 帧 `F`：

1. 对该帧内每个 view（或 `_render_multi_view` 返回的每一张图）计算标量 `loss_{F,v}`（可与 Stage 3.3/4.0 一致：masked L1/SSIM 等）。
2. **帧内**：\(\; L_F = \frac{1}{|V_F|} \sum_{v \in V_F} loss_{F,v}\;\)，其中 \(V_F\) 为「本帧实际参与训练的 view 集合」。
3. **若某 view 有效像素掩码为空**（或该 view 被跳过），该 view **不参与**分子；若 `|V_F^{eff}| = 0`，则 **跳过该帧 loss**（不贡献、不除零）。

**帧间**：\(\; L_{step} = \frac{1}{|F^{eff}|} \sum_{F \in F^{eff}} L_F\;\)，其中 \(F^{eff}\) 为「至少有一个有效 view」的帧集合；若 **所有帧均被跳过**，则本 step **应显式处理**（返回零 loss 并 logging，或直接 fast-fail——由产品决定，但禁止 `nan`）。

说明：按帧批量光栅化不改变上述归一化定义；仅改变 `loss_{F,v}` 的计算方式是一次还是多次 kernel。

---

## 7. Fast-fail（进入 step 前一次性检查）

在构造 mask / 特征 / 渲染之前完成（单一校验入口，失败即抛错）：

1. **Pose / 帧索引**：`source_frame_idx` 与 **每个** `batch["targets"][*].frame_idx` 均能在 `dynamic_info`（经 `frame_ids` 映射）中解析出实例位姿；缺帧、缺实例键、张量形状不对 → **报错**。
2. **Per-target 数据对齐**：每个 target 条目的 `view`、`gt_image`、若使用的 **`sky_mask`**、**`viewdirs`**（以及任何参与 loss 的 per-pixel 张量）**空间分辨率与语义一致**（与 `StreetForward_Flow.md` / Stage 3.3 约定一致）；禁止依赖 trainer 内盲目 resize 兜底。

可与现有 `_assert_src_target_consistent` 类逻辑合并并**加强**（不限于 src/target 同一帧）。

---

## 8. 行为矩阵（与 mask 收紧后对齐）

以下用「实例在帧上是否标注有效」与「特征是否有效」区分：

1. **`mask_src_feat_valid` 且 `mask_any_tgt_rigid`**  
   - **可更新**；在实例有效的 target 帧上进入 `idx_trainable[F]`，用可微 local render params 渲染。

2. **实例在 src 有效但 `~mask_src_feat_valid`，且某帧 `mask_tgt_rigid[F]`**  
   - **不更新**；若仍希望对 target 解释画面，进入 `idx_frozen[F]`，用 **detached NodeState** 渲染（若不希望画 rigid，则可选择不把这类点放入 `mask_tgt`——产品决策；默认设计是 **frozen 仍渲染**）。

3. **`mask_src_feat_valid` 但某帧 `~mask_tgt_rigid[F]`**  
   - **该帧不渲染**该点；其他帧仍可监督并更新。

4. **`~mask_any_tgt_rigid`**  
   - **不渲染 rigid（无任何 target 帧需要该点）+ 不更新**（`mask_update` 为假）。

---

## 9. 可见性：MVP 与可选增强

### 9.1 MVP

- **实例级**：`mask_src_rigid`、`mask_tgt_rigid[F]`。
- **特征级收紧**：`mask_src_feat_valid`（判据需配置化或常量显式写出）。

### 9.2 可选后续

- 在 `mask_tgt_rigid[F]` 上加点级投影/视锥条件，进一步缩小 `idx_frozen` / `idx_trainable` 的渲染集合（与评估文档一致：实例级 ≠ 点级）。

---

## 10. 推荐前向流程（与实现顺序）

1. **Fast-fail 校验**（§7）。
2. 计算 `mask_src_rigid`、`mask_tgt_rigid[F]`、`mask_any_tgt_rigid`；反投影/融合后计算 **`mask_src_feat_valid`**；得 **`mask_update_rigid`**（§2–3）。
3. **仅对 `mask_update_rigid`**：2D feat → GRU → offsets → **`render_params_rigid_local`**；其余点 offsets/hidden 按 §5。
4. 对每个 `F`：取 `idx_trainable[F]`、`idx_frozen[F]`，分别用 **可微 render params** 与 **detached NodeState** 变到 world，合并后再 `_render_multi_view` 或分组 fallback（§4）。
5. Loss：`L_F` 帧内平均，`L_step` 帧间平均；空像素/空帧安全跳过（§6）。
6. Writeback：只对 `mask_update_rigid` scatter 更新 NodeState（§5.3）。

---

## 11. Stage 4.1 MVP 清单（规范摘要）

1. Mask：`mask_src_rigid`、`mask_tgt_rigid[F]`、`mask_any_tgt_rigid`、**`mask_src_feat_valid`**、`mask_update_rigid = mask_src_feat_valid & mask_any_tgt_rigid`。
2. 子集：`idx_trainable[F]`、`idx_frozen[F]` 如 §3。
3. 渲染源：trainable = 可微 local render params；frozen = **detached** NodeState local（§4）。
4. Hidden / writeback：仅 **`mask_update_rigid` scatter**（§5）。
5. Loss：**帧内平均**，再 **帧间平均**；空 mask / 空像素 **安全跳过**（§6）。
6. **进入 step 前一次性 fast-fail**：位姿可解析 + **source/target 的 view/gt/sky_mask/viewdirs 真正对齐**（§7）。

---

## 12. 已知能力边界

- target 新暴露表面若无 source 特征，通常 **`mask_src_feat_valid` 为假** → 不会误更新，但也学不到该处；frozen 仅用旧 NodeState **垫**画面，不解决补全。
- 实例级 `mask_tgt` 仍可能比「点真在画内」更粗；需接受或引入 §9.2。

---

## 13. 结论

Stage 4.1 在「实例级 + 按帧批量渲染」的工程前提下，用 **`mask_src_feat_valid` 收紧更新资格**、**trainable/frozen 双源渲染** 与 **scatter 写回**，把「能画」与「能学」分开，并固定 **loss 归一化与 fast-fail** 契约，便于实现与排错。
