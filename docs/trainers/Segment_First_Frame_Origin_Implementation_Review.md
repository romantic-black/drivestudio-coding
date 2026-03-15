# Segment 第一帧原点（AABB 局部坐标系）实现逻辑与反直觉检查

## 1. Commit 7016c61 目标

- **目标**：让每个段的首帧作为原点，使 AABB 使用**局部**（segment 第一帧）坐标系而非全局世界坐标系。
- **文档约定**（`MultiSceneDataset_Usage.md`）：`crop_aabb` / `input_aabb` 定义在 **segment 第一帧坐标系**；数据侧将点云（背景/远景的“世界坐标”）及 source/target/test 相机外参统一转换到该坐标系；动态点云保持局部坐标不变。

---

## 2. 当前实现逻辑总结

### 2.1 坐标系定义

| 名称 | 含义 |
|------|------|
| **世界坐标 (world)** | 场景原始参考系，lidar/camera 的 `*_to_world` 均相对该系。 |
| **Segment 第一帧坐标 (seg0)** | 以段内**第一帧**的位姿为原点：`world_to_seg0 = inv(segment_first_pose)`。 |

- `segment_first_pose`：段首帧在世界系下的位姿（4×4），来源优先级：`lidar_to_worlds[first_frame_idx]` → `cam_to_worlds[first_frame_idx]`。
- `world_to_seg0`：世界 → seg0 的变换，用于把“世界坐标”的点/相机/实例位姿变换到 seg0。

### 2.2 MultiSceneDataset 中的行为

1. **get_segment_batch**
   - 取段首帧位姿 → 算 `world_to_seg0 = inv(segment_first_pose)`。
   - **相机外参**：在组装 batch 前，对 source/target/test 的 extrinsics 做 `ext_new = world_to_seg0 @ ext`，即 batch 里的外参已是 **seg0 系下的 camera_to_world**。
   - **点云**：通过 `segment_first_pose` 传给点云生成器，在生成器内用 `world_to_seg0` 把世界坐标点变到 seg0，再在 seg0 下用 `crop_aabb`/`input_aabb` 裁剪/过滤。
   - **dynamic_info**：`_build_dynamic_info` 里对每帧每个实例的 `instances_pose[frame_idx, instance_id]` 做 `pose_matrix = world_to_seg0 @ pose_matrix`，输出的 quat/trans 是 **seg0 系下的实例位姿**。
  - **batch['aabb']**：**segment 第一帧坐标系 (seg0)** 下的 AABB，与 extrinsics、点云一致；固定来源为 `dataset.segment_aabb`（唯一来源）。

2. **点云生成器（base/lidar/monocular）**
   - 用 `segment_first_pose` 计算 `world_to_seg0`。
   - **背景点**：先在世界系下生成/着色，再 `_transform_points_np(..., world_to_seg0)` 转到 seg0，然后在 seg0 下用 `get_crop_aabb()` / `get_input_aabb()` 做 AABB 裁剪与内外部分离、过滤。
   - **动态实例**：monocular 的 `_get_instances_for_segment` 里对每帧实例的 `T_ow` 做 `_transform_instances_to_seg0(..., world_to_seg0)`；动态点云本身仍是**对象局部坐标**，与文档一致。
   - **crop_aabb / input_aabb**：配置为“相对 segment 第一帧”的 AABB，在点云已变换到 seg0 后使用，逻辑一致。

3. **段分割（_split_segments）**
   - **修复后**：段数由「关键帧总轨迹距离 / 参考长度」决定；参考长度来自 pointcloud 的 **crop_aabb**（与训练同系，seg0 量纲）。每个 segment 不再存储 `'aabb'`；不再调用 `_compute_segment_aabb`。
   - Trainer 与渲染应使用 **batch['aabb']**（seg0 系），保证与当前 batch 一致。

---

## 3. 反直觉机制与潜在问题

### 3.1 段 AABB 曾为世界系（已修复）

- **原问题**：`segment['aabb']` 由 lidar 世界坐标计算，与 seg0 系约定不一致。
- **修复**：段分割不再使用 segment 级 AABB；参考长度改为 pointcloud 的 **crop_aabb**；segment 字典中已**移除** `'aabb'` 字段。下游统一使用 **batch['aabb']**（seg0 系）。

### 3.2 Batch 中的 AABB（已明确）

- **现状**：`get_segment_batch` 返回 **batch['aabb']**，为 **segment 第一帧坐标系 (seg0)**，与 batch 内外参、点云、dynamic_info 一致。
- **取值**：固定使用 `dataset.segment_aabb`。Trainer 应使用 `batch['aabb']` 作为场景框。

### 3.3 外参变换时机：先取原始外参再统一变换

- **现象**：先按帧/相机从 `pixel_source.get_image` 取 `camera_to_world`（世界系），收集成 list，在“7.5”步再对 source/target/test 的 extrinsics 做 `world_to_seg0 @ ext`。
- **反直觉**：阅读代码时容易以为“外参就是原始世界系”，直到看到 7.5 才意识到已变换到 seg0；若中间有逻辑依赖“世界系外参”会出错。
- **潜在问题**：任何在 7.5 之前使用 `source_extrinsics`/`target_extrinsics` 的代码（或测试）若假设其为世界系，会与最终 batch 不一致。当前实现中 7.5 之前未使用外参做几何，风险可控，但需在注释/文档中明确“batch 中外参均为 seg0 系”。

### 3.4 dynamic_info 与点云实例的一致性

- **现象**：`_build_dynamic_info` 用 `world_to_seg0` 把实例位姿变到 seg0；点云生成器里动态对象的 `T_ow` 也在 `_transform_instances_to_seg0` 中变到 seg0；动态点云本身是对象局部坐标，不变换。
- **一致性**：两者都对齐到 seg0，与文档“动态点云保持局部坐标不变”一致；dynamic_info 的 quat/trans 是 seg0 系下的对象位姿，用于渲染/约束时与 seg0 系相机、背景点一致。
- **node_state_mixin 改动**：commit 中把“dynamic_info 里存在但 pointcloud 里没有的 instance_id”从 `raise ValueError` 改为 `logger.debug` 并 `continue`。这是因为在 seg0 系下、按 AABB 裁剪后，部分帧标注的实例可能不在当前段点云中，静默跳过可避免误报；但若依赖“dynamic_info 与 pointcloud 实例完全一一对应”的逻辑，需要另做检查。

### 3.5 点云生成器强依赖 segment_first_pose

- **现象**：`generate_pointcloud(..., segment_first_pose=...)` 在 base 中通过 `_compute_world_to_seg0(segment_first_pose)` 得到 `world_to_seg0`；若 `segment_first_pose` 为 None 会报错。
- **反直觉**：调用方（如 scheduler）必须从 dataset 取到当前段的 `segment_first_pose` 并传入；若漏传或传错段，整段点云和 AABB 的坐标系会错乱。
- **建议**：在生成器内对 `segment_first_pose is None` 做明确报错信息，并在文档中写明“多场景/段训练必须传 segment_first_pose”。

### 3.6 lidar 与 camera 首帧位姿不一致

- **现象**：`_get_segment_first_pose` 优先用 `lidar_to_worlds[first_frame_idx]`，没有再用 `cam_to_worlds`。若 lidar 与 camera 标定或时间对齐有偏差，首帧位姿会与“相机首帧”不一致。
- **潜在问题**：外参和 dynamic_info 都是相对“首帧位姿”建的 seg0；若 seg0 实际取自 lidar 而可视化或评估按相机理解，可能产生轻微不一致。通常同一数据源会标定一致，但值得在文档中注明“seg0 来自 lidar 或参考相机，请与数据标定一致”。

---

## 4. 小结表

| 项目 | 当前实现 | 约定/期望 | 是否一致 |
|------|----------|-----------|----------|
| 相机外参 (batch) | 变换为 seg0 系 | seg0 系 | ✅ |
| 背景点云 | 世界→seg0 后 crop/input_aabb | seg0 系 + AABB 在 seg0 | ✅ |
| 动态点云 | 对象局部坐标 | 对象局部 | ✅ |
| dynamic_info (quat/trans) | world_to_seg0 @ pose | seg0 系 | ✅ |
| crop_aabb / input_aabb | 在 seg0 下使用 | segment 第一帧系 | ✅ |
| 段分割 | 用 crop_aabb 长度作参考，不存 segment['aabb'] | 与训练 AABB 一致 | ✅ |
| batch['aabb'] | seg0 系，dataset.segment_aabb | seg0 系，与 batch 同系 | ✅ |

---

## 5. 已完成的修复（简要）

1. **段分割**：不再使用 `scene_dataset.get_aabb()` 与 `segment['aabb']`；参考长度改为 pointcloud 的 **crop_aabb**；segment 字典移除 `'aabb'`。
2. **batch['aabb']**：在 `get_segment_batch` 中写入 **seg0 系** AABB，固定来源为 `dataset.segment_aabb`。
3. **文档**：`MultiSceneDataset_Usage.md` 已更新段分割说明与 batch 格式；详见 `Segment_AABB_And_Batch_AABB_Plan.md`。

以上为对 commit 7016c61 后“每段首帧为原点、AABB 局部坐标系”实现的逻辑梳理、反直觉检查与修复总结。
