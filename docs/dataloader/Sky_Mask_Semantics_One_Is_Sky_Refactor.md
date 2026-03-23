# sky_mask 语义统一：1 = sky（读取后全链路约定）

本文档说明当前 `sky_mask` 语义混乱的来源、目标约定，以及在 `MultiSceneDataset`（及关联层）通过**配置在读取阶段归一化**后，需要同步修正的代码与文档位置。实现时可按本文档作为 checklist。

---

## 1. 问题背景

### 1.1 直觉与代码注释的冲突

直觉上「sky_mask」应表示 **天空区域**：**1 = sky**、**0 = non-sky** 易于理解与沟通。

当前训练侧大量注释与错误信息写的是 **`0 = sky`，`1 = non-sky`**（例如 Stage 3.2 P0、`sky_node`、部分 dataloader 文档）。开发者必须在脑中持续做「mask 名 vs 数值语义」的二次映射，容易在 loss 分支、`(1-sm)`、单元测试构造张量时写反。

### 1.2 磁盘与 `pixel_source` 并非单一语义

`datasets/base/pixel_source.py` 中加载逻辑为灰度图读入后：

```384:385:datasets/base/pixel_source.py
            sky_masks.append(np.array(sky_mask) > 0)
        self.sky_masks = torch.from_numpy(np.stack(sky_masks, axis=0)).float()
```

即 **`float` 张量上：像素值 `>0` 的位置为 1，否则为 0**。  
**1 对应的是「PNG 非零」**，而「非零」在不同预处理管线里既可以表示 **天空** 也可以表示 **非天空**：

| 来源（示例） | PNG 含义 | `>0` 后 float 语义 |
|-------------|----------|-------------------|
| EVolSplat / NuScenes 等注释：`0=sky, 255=non-sky` | 非零 = non-sky | **1 = non-sky** |
| `datasets/tools/extract_masks.py`（Cityscapes 类 10 = sky） | 非零 = sky | **1 = sky** |

因此**仅靠「读 PNG」无法在不加说明的情况下固定「1 是 sky 还是 non-sky」**；若不在数据出口统一，下游只能继续写「0=sky,1=non-sky」并反复取反。

---

## 2. 目标约定（归一化之后）

**在 `MultiSceneDataset`（及与其对齐的 batch 组装路径）完成归一化后，全项目约定：**

- **`sky_mask`（float，与图像同空间分辨率 `[H,W]`）**  
  - **`1` = sky（天空像素）**  
  - **`0` = non-sky（非天空像素）**

**所有 loss、可见性、SkyNode observation 等逻辑**均按上述语义编写；若某处需要「非天空区域权重」或「仅对 non-sky 算 L1」，应显式写 **`1.0 - sky_mask`** 或 `non_sky_mask` 临时变量，而不是依赖「sky_mask 其实是 occupied」这种隐式约定。

---

## 3. 建议的配置形态（读取阶段归一化）

在 **`MultiSceneDataset` 使用的 `data_cfg`**（或与 `pixel_source` 共享的配置）中增加明确项，避免静默默认。两种等价表述（二选一即可，文档与代码应一致）：

**方案 A（推荐，与直觉一致）**

- `sky_mask_semantics: one_is_non_sky` | `one_is_sky`  
  - **`one_is_non_sky`**：loader（`pixel_source` 的 `>0` float）表示 **1=非天空** 时，在 `MultiSceneDataset` 内做 **`sky_mask = 1.0 - x`**，使 batch 内 **`1=sky`**。  
  - **`one_is_sky`**：loader 已为 **1=天空**，不再取反。

**实现状态（代码已落地）**：`datasets/sky_mask_semantics.py` + `MultiSceneDataset` 归一化与占位；`metrics` / `proxy_rendering_mixin` / Stage 3.2 P0 等已按 **1=sky** 更新；`data.sky_mask_semantics` 在 `load_sky_mask: true` 时 **必填**（fast-fail）。

**实现落点（建议）**

1. **优先在 `MultiSceneDataset` 内**对 `image_infos['sky_masks']` / `frame_data['sky_mask']` 在**第一次进入 batch 之前**做统一变换（与 `get_segment_batch` / `_get_frame_data` 等路径一致），保证 `batch['source|target|test']['sky_mask']` 语义一致。  
2. 若部分场景绕过 MSD 直接从 `PixelSource` 取图，则需在 **`pixel_source.get_image` 出口**或共享的 `normalize_sky_mask(tensor, cfg)` 中调用同一套逻辑，避免双路径不一致。

**占位张量（missing mask）**

当前实现：若某视角没有 mask，用 **`torch.ones(H,W)`** 填充以保持形状（注释为 *fill missing with ones*）。该行为在旧语义下表示「**全部为 non-sky**」（全 1 = 无天空）。  

归一化到 **`1=sky`** 后，「全部为 non-sky、无天空像素」应对 **`torch.zeros(H,W)`**。  
→ **占位需从 `ones` 改为 `zeros`**（在采用新语义时），否则 missing 视角会被当成「满屏天空」，与 Stage 3.x 监督语义相反。

---

## 4. 需修正的位置与方案（按模块）

### 4.1 `datasets/multi_scene_dataset.py`

| 位置 | 现状 | 修正 |
|------|------|------|
| `_get_frame_data` 等处从 `image_infos['sky_masks']` 取值 | 直接传递 loader 语义 | 根据配置做 `1-x` 或恒等，输出统一 **`1=sky`** |
| `has_*_sky_mask` 分支里对 `None` 的占位 | `torch.ones` | 改为 **`torch.zeros`**（表示无天空 / 全 non-sky，与旧「全 1 = non-sky」等价） |
| 注释 | 混合 | 更新为「batch 内 **`1=sky`**」 |

### 4.2 `datasets/base/pixel_source.py`

| 位置 | 说明 |
|------|------|
| `load_sky_masks` | 可保持 `>0` → float；**语义归一化建议在 MSD 或共享函数中完成**，避免每个 sourceloader 复制逻辑。若未来改为直接输出 canonical，需与 MSD 配置二选一、避免双重取反。 |

### 4.3 `datasets/pointcloud_generators/monocular.py`

`_generate_points_from_frame_data` 中（约 372–377 行）：在 `filter_sky` 下对 bool 做 `~sky_mask` 以保留非天空点。  

- **旧语义（1=non-sky）**：`True` = non-sky，`~` 后保留 sky → 实际配合的是「bool 表示 non-sky」的解读（需结合当时 `astype(bool)` 含义）。  
- **新语义（1=sky）**：`True` = sky，过滤天空应保留 **non-sky**，即 **`~sky_mask_bool` 仍为「去掉天空」**（若 bool 直接表示 sky）。  

**修正**：在 canonical **`1=sky`** 下，明确变量名如 `is_sky = sky_mask > 0.5`，`keep = ~is_sky`（filter_sky 时）。需用单元测试或对照旧行为验证点云条数一致。

### 4.4 `models/streetforward/metrics.py`

`compute_l1_loss_masked` / `compute_ssim_loss_masked`：当前文档写 **「只在 `sky_mask>0` 的像素上计算」**，在旧数据下即 **non-sky 区域**。  

- **新语义下**：`sky_mask>0` 表示 **天空** 像素。若 loss 仍表示「**仅在 non-sky 上监督**」，应改为对 **`(1.0 - sky_mask)`** 加权，或增加参数 `mask_mode: non_sky_from_sky_mask` 并** fast-fail** 未指定时的歧义。  
- **推荐**：更新 docstring：**`sky_mask` 为 `1=sky`；传入 loss 的「有效 non-sky 权重」由调用方传入 `1-sky_mask` 或由 metrics 内统一从 `sky_mask` 推导（二选一，全项目统一）**。

### 4.5 `models/streetforward/proxy_rendering_mixin.py`

当前注释：`1` 表示有效，`0` 表示天空 —— 与旧 **`1=non-sky`** 一致。  

- 若 batch 改为 **`1=sky`**，则「 photometric 有效区 = non-sky」应为 **`valid = 1 - sky_mask`**（或与 `valid_mask` 相乘）。需同步改注释与乘法逻辑。

### 4.6 `models/streetforward/minimal_trainer_stage3_2.py`

P0：`gt_occupied` 使用 `sky_mask` 作为「非天空占据」监督。  

- 旧：`1=non-sky` → `gt_occupied = sky_mask` 合理。  
- 新：**`gt_occupied = 1.0 - sky_mask`**（在 `valid_loss_mask` 下与现式一致）。

错误信息从「`1=non-sky`」改为「**`1=sky`**」。

### 4.7 `models/streetforward/minimal_trainer_stage3_3.py`

- `sm_sky = (1.0 - sm)`：在旧语义下 `sm` 为 non-sky，`1-sm` 为 sky 区域权重 —— 正确。  
- **新语义下 `sm` 已为 sky**：则 **sky 区域权重应为 `sm` 本身**，non-sky 为 `1-sm`。需**删掉多余的 `(1-sm)` 或对调 `l1_non`/`l1_sky` 的 mask 参数**，并复查 `compute_l1_loss_masked` 的输入是否与 metrics 更新一致。

### 4.8 `models/streetforward/sky_node.py`

`_build_sky_observation_cube`：`mask = (sky_mask == 0)` 表示选天空射线。  

- **新语义**：应为 **`mask = (sky_mask == 1)`** 或 `> 0.5`。更新 docstring。

### 4.9 其他 `minimal_trainer_stage*.py` / `trainer.py`

凡 `sky_mask=target.get("sky_mask")` 传入 `compute_*_masked` 的：在 metrics 语义确定后，统一改为传 **`1-sky_mask`** 或改 metrics 内部处理，避免遗漏。

### 4.10 `utils/minimal_batch_view_selection.py` / `utils/streetforward_baseline.py`

- 更新注释与任何硬编码假设（当前写明 `0=sky,1=non-sky`）。

### 4.11 测试

- `tests/test_stage3_3_sky_node_shapes.py`：构造张量与注释基于旧语义，需按 **`1=sky`** 重写期望值与注释。

### 4.12 配置示例

- `configs/overfit_one_batch_template.yaml`、`configs/streetforward/multi_scene.yaml`、`configs/evolsplat/multi_scene.yaml` 等：在 `load_sky_mask: true` 旁增加 **`sky_mask_semantics`**（或等价项），与 MSD 实现一致。

---

## 5. 文档清单（需同步更新）

| 文档 | 内容 |
|------|------|
| `docs/dataloader/MultiSceneDataset_Usage.md` | batch 中 `sky_mask` 语义、`viewdirs`/占位与 Stage 3.x 关系 |
| `docs/dataloader/MultiSceneDataset_Design.md` | `frame_data['sky_mask']` 说明 |
| `docs/trainers/StreetForward_Flow.md` | 「天空分支」中 `sky_mask` 契约（当前写 `0=天空,1=非天空`） |
| `docs/pointcloud_generators/PointCloud_Generators_Usage.md` | 若涉及 monocular `filter_sky` 与 mask 语义 |
| `docs/trainers/StreetForward_Stage3_3_SkyNode_ConvGRU_Implementation_Plan.md` | `0=天空` 相关段落 |
| 本文档 | 作为迁移说明与 checklist |

---

## 6. 迁移与验证建议

1. **单测**：固定小 `H×W` 张量，验证 MSD 出口、`sky_node` 选线、Stage3_2 P0 `gt_occupied`、metrics 加权区域与迁移前数值一致（允许整体 `1-x` 等价变换）。  
2. **对比实验**：同一 batch 在改前后 `sky_mask` 应互为 `1-x`（在配置为「从 old 转 new」时）。  
3. **预处理团队**：在 README 或数据规范中写明 **PNG 编码**（非零是 sky 还是 non-sky），与 `data.sky_mask_semantics` 对齐。  
4. **已缓存的 overfit `batch.pt`**：若仍按旧语义（1=非天空）保存，需**重新抓取**或离线对张量做 `1-x`，否则 Stage 3.2 P0 与 loss 会错位。

### 分支中若存在 Stage 3.3 / SkyNode

本仓库若另有 `minimal_trainer_stage3_3.py`、`sky_node.py`：将天空像素条件由 `sky_mask == 0` 改为 **`> 0.5` 或 `== 1`**；拆分 loss 时使用 `compute_l1_loss_masked` / `compute_ssim_loss_masked` 的 **`mask_region='sky'`** / **`'non_sky'`**，勿再手写 `(1.0 - sm)` 表示天空权重（在 **1=sky** 下天空权重即为 `sm`）。

---

## 7. 小结

- **根因**：`pixel_source` 的 `>0` 与不同预处理 PNG 定义叠加，且训练侧长期采用 **`1=non-sky`** 的注释体系，与名称 `sky_mask` 冲突。  
- **目标**：在 **`MultiSceneDataset`（读取/组 batch 阶段）**用配置将张量统一为 **`1=sky`**，并系统性更新 metrics、trainer、SkyNode、点云 monocular、占位与文档。  
- **特别注意**：占位从 **`ones` → `zeros`**；Stage 3.3 中 **`(1-sm)` 与 `sm` 的角色对调**；**磁盘双格式**需靠配置区分，不能假设单一全局取反。
