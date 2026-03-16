# 基于点云覆盖的损失掩码：实现计划与动态物体讨论

本文档讨论如何通过「将 input_aabb 内的点反投影到屏幕空间生成掩码、在计算损失时过滤」来缓解「初始 3DGS 不包含远景与天空、但损失全图计算导致模型被迫拟合无点区域」的问题。并讨论动态物体可能遇到的问题及处理方式。

**参考文档**：
- [StreetForward 训练流程](trainers/StreetForward_Flow.md)
- [MultiSceneDataset 使用](dataloader/MultiSceneDataset_Usage.md)
- [点云生成器](pointcloud_generators/PointCloud_Generators.md)

---

## 1. 问题背景

### 1.1 现象

- **初始点云范围**：点云生成器只在 `input_aabb` 内产点；且常做天空过滤、深度一致性等，因此：
  - **远景**：`input_aabb` 外的区域（以及部分 input_aabb 内但无观测到的远距离区域）没有对应 3D 点；
  - **天空**：被过滤或未反投影，也没有 3D 点。
- **损失计算**：当前在整张 target 图像上计算损失（如 L1），仅可选地使用 `sky_mask` 排除天空。
- **结果**：模型在「无点区域」（远景 + 天空等）上仍被监督，被迫用有限的高斯去拟合这些区域，容易导致过拟合或不合理的外观/几何。

### 1.2 目标

- 仅在「有 3D 点覆盖」的像素上计算损失，避免在无点区域上施加监督。
- 掩码应包含：近景（crop/bbx 内）、远景（input_aabb 内、crop 外）、以及**当前 target 帧下的动态物体**在屏幕上的投影区域。
- **实现位置**：推荐在 **点云/数据侧** 用**初始点云**预计算掩码并随 batch 下发，而不是在 StreetForward 模型内用当前高斯位置现算，以避免掩码随训练变化、影响训练稳定性（详见第 3 节）。

---

## 2. 方案概述

### 2.1 思路（与实现位置无关）

1. **参与掩码的点**：**input_aabb 内**的初始点云在该 target 帧、该视角下的投影：
   - 静态背景点（世界坐标 seg0）；
   - 背景远景点（若有）；
   - 动态物体点：在 target 的 `frame_idx` 下用 `dynamic_info` 变换到世界坐标，且在该帧可见的实例（与训练时 rigid 可见性约定一致）。
2. **反投影**：将上述点用该 target 的相机外参、内参投影到图像平面，得到 2D 位置。
3. **生成掩码**：标记「被至少一个点投影覆盖」的像素，得到 `[H, W]` 二值掩码（或软掩码）。
4. **损失计算**：在 `compute_loss` 中仅对掩码为有效的像素计算损失（与 `sky_mask` 取交）。

这样，**远景无点区域**和**天空**都不会被纳入损失。

**掩码由谁算、在何时算**，有两种选择：在 **StreetForward 模型内**（每 iter 用当前 `merged_means` 现算），或在 **点云/数据侧**（batch 构建时用初始点云预计算）。下一节对比并给出推荐。

---

## 3. 掩码计算位置：模型内 vs 点云/数据侧（推荐数据侧）

### 3.1 在 StreetForward 模型内做投影

- **做法**：在 `_render_targets_and_accumulate_loss` 中，对每个 target 用**当前** `merged_means`（即当前 NodeState + offsets 得到的渲染用位置）做反投影，得到当步的「点覆盖掩码」，再传入 `compute_loss`。
- **优点**：掩码与当前帧渲染用的 3D 位置完全一致，理论上「监督区域」与「实际有高斯覆盖的区域」对齐。
- **缺点与对训练结果的影响**：
  - **掩码随训练变化**：随着 offsets 更新，高斯位置会变，哪些像素被标为有效会随之变化 → 每个 iter 的损失定义域不同，优化目标在变。
  - **反馈环**：若某高斯移出视锥，其投影像素会从掩码中消失，该像素不再有损失 → 没有梯度把该高斯拉回，可能加剧「点跑飞」或收敛到局部解。
  - **与目标语义不一致**：我们要的是「**初始点云**覆盖不到的像素不监督」，而不是「**当前模型**覆盖不到的像素不监督」。前者是数据决定的固定集合，后者依赖模型状态，容易干扰训练稳定性。

因此，在模型内用当前 `merged_means` 做投影、动态更新掩码，**可能影响训练结果**，不推荐作为主方案。

### 3.2 在点云/数据侧预计算掩码（推荐）

- **做法**：在构造 batch 时（如 `MultiSceneDataset.get_segment_batch` 或与点云同级的预处理），用**初始点云**（background + distant + dynamic）对**每个 target 视角**做一次反投影，生成 `point_coverage_mask`，放入 `target["point_coverage_mask"]`（或统一命名为 `valid_loss_mask`），训练时直接使用，整个 segment 内不变。
- **优点**：
  - **固定监督域**：整个 segment 训练过程中，哪些像素参与损失由数据唯一决定，不随模型状态变化，训练更稳定。
  - **语义正确**：「有初始点覆盖的像素才监督」与「无点区域不监督」的目标一致，无反馈环。
  - **零训练时开销**：投影与掩码生成只在数据侧做一次，不占用训练循环算力。
  - **职责清晰**：数据侧表达「我们在这里有观测」，模型侧只负责拟合与损失计算。
- **与「当前渲染位置」的偏差**：
  - 掩码基于**初始**点位置，训练中高斯会移动，可能出现：① 某像素在掩码里为 1 但后期该处高斯移走 → 仍被监督（可接受，相当于多了一点监督）；② 某像素在掩码里为 0 但后期有高斯移入 → 不被监督（漏掉少量监督，但不会错误地监督「从未有点」的区域）。两者对主目标（不监督无点区域）都是保守、可接受的。

**结论**：推荐在 **pointcloud / dataloader 侧**完成投影与掩码生成，以初始点云 + 各 target 的 view、frame_idx 为输入，预计算每 view 的 `point_coverage_mask`，随 batch 提供给 StreetForward；模型内仅读取该掩码并与 `sky_mask` 取交后传入 `compute_loss`。

### 3.3 与现有流程的衔接（数据侧方案）

- **数据流**：batch 的 `target[i]` 中除现有 `view`、`gt_image`、`sky_mask` 外，增加 `point_coverage_mask`（形状 `[H, W]`，1=有点覆盖、0=无）。若未提供则 trainer 可退化为全图损失或仅用 `sky_mask`。
- **模型侧**：在 `_render_targets_and_accumulate_loss` 中，`valid_mask = target.get("point_coverage_mask")`，与 `sky_mask` 取交后传入 `compute_loss`，不再在模型内做任何投影或掩码计算。

---

## 4. 实现要点

### 4.1 数据侧：投影与掩码生成（点云/数据集）

- **坐标系**：与 [StreetForward_Flow](trainers/StreetForward_Flow.md)、[PointCloud_Generators](pointcloud_generators/PointCloud_Generators.md) 一致：世界坐标即 segment 第一帧系（seg0）；相机外参为 `camera_to_world`，反投影用 `world_to_camera = inv(camtoworlds)` 与内参 `K`。
- **点集**：对每个 target `(frame_idx, view)`，
  - 背景点：`pointcloud["background"]` 的 `[x,y,z]`（已是 seg0）；
  - 远景点：若有 `pointcloud["distant"]` 或等价拆分，同上；
  - 动态点：用 `dynamic_info[frame_idx]` 将各实例局部点变换到 seg0 世界坐标，仅保留该帧可见的实例（与 trainer 侧 `instances_fv` / 可见性约定一致）。
- **反投影**：同上，`p_cam = viewmat @ [x,y,z,1]`，`z_cam > 0`，内参得到 `(u,v)`，筛掉视锥外。
- **掩码**：将 `(u,v)` 取整到像素，在 `[H,W]` 上置 1（方式 A）；可选做轻微膨胀或软边界。输出 `point_coverage_mask` 随该 target 写入 batch。

实现可放在：`MultiSceneDataset` 在组装 target 时调用工具函数（如 `points_to_coverage_mask(background_xyz, distant_xyz, dynamic_xyz_per_frame, view, frame_idx, dynamic_info, ...)`），或由单独的数据预处理脚本在生成/缓存 pointcloud 时一并写出 per-view 掩码，再由 dataset 加载。

### 4.2 模型侧：仅使用掩码

- StreetForward 不再根据 `merged_means` 计算掩码；仅从 `target["point_coverage_mask"]` 读取，与 `target.get("sky_mask")` 取交，传入 `compute_loss`。
- **compute_loss**：增加参数 `valid_mask`（或复用/组合 `sky_mask`），最终有效区域 = `(point_coverage_mask if provided) & (sky_mask if provided else 1)`，仅在该区域上求平均损失。

---

## 5. 动态物体可能遇到的问题及处理（数据侧预计算时）

采用数据侧预计算时，动态物体相关逻辑在 **batch 构建阶段** 完成，需与 trainer 侧对「可见性、帧」的约定一致。

### 5.1 时序与帧一致性

- **问题**：动态物体在不同 target 帧处于不同世界坐标；掩码用错帧会漏标或错标 rigid 区域。
- **处理**：对每个 target，用**该 target 的 `frame_idx`** 从 `dynamic_info` 取该帧的实例位姿，将动态点从局部坐标变换到 seg0，再与背景、远景一起投影到该 target 的 view。即：每个 target 的 `point_coverage_mask` 使用「该 target 的 frame_idx + 该 target 的 view」独立计算，与训练时该 target 使用的 rigid 变换一致。

### 5.2 可见性（无监督 rigid 不参与掩码）

- **问题**：部分实例仅在 source 可见、在任意 target 都不可见；若仍把它们投影进掩码，会错误地扩大有效区域。
- **处理**：数据侧生成掩码时，只把「在该 target 的 frame_idx 下可见」的实例变换并投影（与 trainer 的 `instances_fv` / `idx_tgt_rigid` 语义一致）。若数据集有 `per_frame_instance_mask` 或等价信息，应用同一套可见性规则，避免无监督 rigid 出现在掩码中。

### 5.3 遮挡与深度顺序

- **问题**：多点在同像素时，仅最前的高斯参与渲染；二值掩码「有投影即有效」可能把被遮挡的投影像素也算进损失。
- **处理**：数据侧通常只做几何投影（无深度排序），得到的是「有任意点投影」的二值掩码。这是保守做法：可能多包含少量被遮挡像素，但不会误删应有监督的像素。若需更精细，可在数据侧用简单深度缓冲（按深度取最近点）做粗略前景掩码，或接受当前保守二值掩码。

### 5.4 动态物体边界与运动模糊

- 同前：属于「有监督但难拟合」的另一类问题；不改变「仅在有点覆盖处计算损失」的主逻辑。若需弱化动态边缘，可在数据侧或后续在掩码上乘 per-pixel 权重。

### 5.5 掩码 per-view

- 每个 target 的 `point_coverage_mask` 独立：不同 view、不同 `frame_idx`，在数据侧分别算、分别写入 `target[i]["point_coverage_mask"]`，与现有「每个 target 单独渲染、单独 loss」一致。

---

## 6. 实现步骤小结（推荐：数据侧预计算）

1. **数据侧（点云/数据集）**  
   - 实现投影工具：`project_points_to_image(points_xyz, view, height, width) -> (u, v, valid)`。  
   - 对每个 target：合并 background + distant + 该 `frame_idx` 下可见的 dynamic（用 `dynamic_info` 变换到 seg0），调用投影，生成二值 `point_coverage_mask` `[H, W]`，写入 `target["point_coverage_mask"]`（或 `valid_loss_mask`）。  
   - 可见性规则与 trainer 对齐（仅该帧可见的实例参与投影）。

2. **模型侧（StreetForward）**  
   - 在 `_render_targets_and_accumulate_loss` 中：`point_coverage_mask = target.get("point_coverage_mask")`；与 `target.get("sky_mask")` 取交得到 `valid_mask`，传入 `compute_loss(rgb, gt_img, sky_mask=..., valid_mask=point_coverage_mask)`（或扩展后的接口）。  
   - `compute_loss`：若增加 `valid_mask`，则 `final_valid = (valid_mask if valid_mask is not None else 1) & (sky_mask if sky_mask is not None else 1)`，仅对 `final_valid` 非零像素求平均损失。

3. **不推荐**：在模型内用当前 `merged_means` 每 iter 现算掩码，以免掩码随训练变化、产生反馈环、影响训练稳定性。

完成后，损失将只在「**初始** input_aabb 内点云在该视角下的覆盖区域」上计算，远景与天空等无点区域不再被监督；动态物体在数据侧按帧与可见性正确纳入掩码。

---

## 7. 反直觉检查（实现后核对）

- **掩码全零的视角**：若某视角没有任何点投影到图像内（如相机背对场景），`valid_pixels == 0`，`compute_loss` 返回 `0.0`，该视角不产生梯度。行为符合预期；若需排查「某视角不参与训练」可在外部根据掩码统计有效像素数。
- **final_valid = point_coverage_mask × sky_mask**：同时要求「有点覆盖」与「非天空」；若数据侧 `sky_mask` 为 0/1（0=天空），则最终监督区域为两者交集，无逻辑错误。
- **坐标系**：数据侧投影使用 `inv(c2w)`（world_to_camera），与 batch 中 extrinsics 为 camera_to_world 一致；动态点用 `dynamic_info[frame_idx]` 的 quat+trans 从局部变换到 seg0，与 NodeState 侧 rigid 变换约定一致。
- **必需掩码**：当前约定为 batch 必须提供 `point_coverage_mask`（无则 fast-fail）；数据集在点云存在时构建掩码失败会直接抛异常，不做静默回退。
