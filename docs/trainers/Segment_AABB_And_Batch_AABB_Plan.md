# Segment AABB 必要性讨论与 Batch AABB 修复方案

> 注：本计划文档包含历史讨论。当前代码已统一到 `dataset.segment_aabb` / `dataset.segment_input_aabb` 作为唯一来源，并移除了 `fixed_segment_aabb` 与 `use_bbx`。

## 1. 是否必须使用 segment['aabb'] 做段分割？

### 1.1 当前 _split_segments 中 AABB 的实际作用

- **场景级 AABB（scene_dataset.get_aabb()）**  
  - 仅用于计算 `aabb_length = scene_size.max().item()`。  
  - `aabb_length` 只参与两处逻辑：  
    - `total_keyframe_distance < aabb_length * 0.3` → 判单段/多段；  
    - `distance_ratio = total_keyframe_distance / aabb_length` → 推算 `num_segments_by_distance`。  
  - 即：**段数**由“轨迹总距离 / 场景尺度”决定，与**每个段的 AABB** 无关。

- **段级 AABB（segment['aabb']）**  
  - 每个 segment 写入 `'aabb': segment_aabb`，来源为 `fixed_segment_aabb` 或 `_compute_segment_aabb(scene_dataset, frame_indices)`（世界系 lidar 范围）。  
  - 在代码库中**没有任何地方读取** `segment['aabb']`。  
  - 点云与训练使用的边界来自 pointcloud 的 `crop_aabb` / `input_aabb`（seg0 系），与 segment['aabb'] 无关。

结论：**段分割逻辑本身不依赖 segment['aabb']**；segment['aabb'] 只是被写入、从未被使用。  
“段分割”依赖的是：**轨迹距离 + 场景整体尺度（scene AABB 的 max 维度）**。

### 1.2 能否用 crop_aabb 替代场景尺度？

可以。当前用 `aabb_length` 只是为了一个“与场景尺度同量纲”的长度，用于和 `total_keyframe_distance` 比较。  
完全可以用 **crop_aabb 的尺寸** 代替：

- 在 `_split_segments` 中不再调用 `scene_dataset.get_aabb()`。
- 从 `pointcloud_config` 取 `crop_aabb`（或若未配置则使用与当前默认一致的 fallback），计算例如：
  - `aabb_length = (crop_aabb[1] - crop_aabb[0]).max()`  
  即用 crop 在 seg0 系下的“最大边长”作为尺度参考。
- 这样：
  - 段数仍由“轨迹距离 / 某 AABB 长度”决定；
  - 该 AABB 与下游点云/训练使用的 crop 一致（都是 seg0 系、同一配置），语义统一。

注意：crop_aabb 是**每个段 seg0 系**下的固定框，而“场景总轨迹长度”是世界系下的距离。用 crop 的尺寸只是作为一个**标量长度参考**（米），与“用场景 AABB 的 max 维度”在用途上等价，不要求 crop 覆盖整条轨迹。

### 1.3 是否应移除 segment['aabb']？

**建议：移除。**

- 无人读取，保留只会造成“段里还有一个世界系 AABB”的混淆（且与 fixed_segment_aabb / crop_aabb 的 seg0 系不一致）。
- 若将来需要“段范围”信息，应使用与 batch 一致的 **seg0 系 AABB**（见下节），并由 batch 或配置统一提供，而不是在段里存一份世界系 AABB。

---

## 2. Batch 中的 AABB 为 seg0 系（明确约定）

### 2.1 现状

- 文档约定：`crop_aabb` / `input_aabb` 为 **segment 第一帧坐标系 (seg0)**。  
- 当前 `get_segment_batch` 返回的 batch **不包含**任何 `aabb` 或 `segment_aabb` 字段。  
- Trainer 若需要“本段的场景框”，只能从别处取，容易与当前 batch 的 seg0 系不一致。

### 2.2 约定（明确写入文档与实现）

- **Batch 中若提供 AABB，则其坐标系为 segment 第一帧坐标系 (seg0)。**  
- 与 batch 内 `source/target/test` 的 extrinsics、点云、dynamic_info 一致，均为 seg0 系。

### 2.3 实现方式建议

- 在 `get_segment_batch` 组装 batch 时，增加字段，例如：
  - `batch['aabb']` 或 `batch['segment_aabb']`：`Tensor[2, 3]`，seg0 系。
- 取值建议（二选一或兼容）：
  - 若配置了 `fixed_segment_aabb`：使用 `fixed_segment_aabb`（文档已约定为 seg0 系）。
  - 否则：使用 pointcloud 的 **crop_aabb**（或 `input_aabb`）转为 tensor，作为该段在 seg0 下的“场景框”。  
  这样 trainer 拿到的就是与点云、外参完全同系的 AABB。

---

## 3. 修复方案总览

### 3.1 文档

1. **新建/维护**（本文档或 MultiSceneDataset_Usage.md）  
   - 明确：**段分割不再依赖 segment['aabb']**；段数由轨迹距离与“参考 AABB 长度”（见下）决定。  
   - 明确：**batch 中的 AABB（若有）为 seg0 系**；与 extrinsics、点云、dynamic_info 一致。  
   - 在数据格式说明中写清：`batch['aabb']` 或 `batch['segment_aabb']` 的 shape、单位、坐标系（seg0）。

### 3.2 代码修改清单

| 序号 | 位置 | 修改内容 |
|------|------|----------|
| 1 | `datasets/multi_scene_dataset.py` — `_split_segments` | 不再使用 `scene_dataset.get_aabb()`。从 `pointcloud_config` 读取 `crop_aabb`（或合理默认），计算 `aabb_length = (crop_max - crop_min).max()`，用于现有 `num_segments` 逻辑。 |
| 2 | `datasets/multi_scene_dataset.py` — `_split_segments` | 移除每个 segment 的 `'aabb'` 字段；不再调用 `_compute_segment_aabb`，不再使用 `fixed_segment_aabb` 写入 segment。 |
| 3 | `datasets/multi_scene_dataset.py` | 可选：若不再需要 `fixed_segment_aabb` 仅用于段分割，可考虑仅保留其用于“batch 的 aabb 来源”（见下）；若仍希望“固定段框”仅给点云用，则 `fixed_segment_aabb` 仅用于 pointcloud 生成器，段分割只认 crop_aabb。 |
| 4 | `datasets/multi_scene_dataset.py` — `get_segment_batch` | 在组装 batch 时增加 `batch['aabb']`（或 `batch['segment_aabb']`）：seg0 系；取值 = `fixed_segment_aabb`（若配置）否则 pointcloud 的 crop_aabb 转 tensor。 |
| 5 | `datasets/multi_scene_dataset.py` | 若 `_split_segments` 不再使用 `fixed_segment_aabb` 与 `_compute_segment_aabb`，可保留 `_compute_segment_aabb` 供其他用途或删除；`fixed_segment_aabb` 仍可在 __init__ 中保留，仅用于 get_segment_batch 的 batch['aabb'] 与/或 pointcloud 配置。 |
| 6 | `docs/dataloader/MultiSceneDataset_Usage.md` | 在“输出结构 / Batch 格式”中增加 `aabb`（或 `segment_aabb`）字段说明，注明 shape、坐标系（seg0）、与 crop_aabb 的关系。 |
| 7 | `docs/trainers/Segment_First_Frame_Origin_Implementation_Review.md` | 更新“段 AABB”小节：段分割改用 crop_aabb 长度、移除 segment['aabb']；batch 中 AABB 明确为 seg0 系。 |

### 3.3 实现细节（_split_segments 用 crop_aabb 作长度）

- 在 `_split_segments` 开头（或调用前）从 `self.pointcloud_config` 取：
  - `crop_aabb = self.pointcloud_config.get("crop_aabb", [[-20, -20, -20], [20, 4.8, 70]])`
- 转为 numpy 或 tensor，计算：
  - `crop_min, crop_max = crop_aabb[0], crop_aabb[1]`
  - `aabb_length = (np.array(crop_max) - np.array(crop_min)).max()`
- 用该 `aabb_length` 替换当前由 `scene_dataset.get_aabb()` 得到的 `aabb_length`，其余逻辑（num_segments、overlap 等）不变。
- 这样不再依赖 scene 的 get_aabb()，也不依赖 lidar，且与 pointcloud 的 crop 一致。

### 3.4 实现细节（get_segment_batch 写入 batch['aabb']）

- 在组装 batch 的 dict 时增加：
  - 若 `self.fixed_segment_aabb is not None`：`batch['aabb'] = self.fixed_segment_aabb.to(device=self.device)`（或 clone）。
  - 否则：从 `self.pointcloud_config.get("crop_aabb", ...)` 得到 list/array，转为 `torch.tensor(..., dtype=torch.float32, device=self.device)`，shape (2, 3)，写入 `batch['aabb']`。
- 在 docstring 与 MultiSceneDataset_Usage.md 中写明：**batch['aabb'] 为 segment 第一帧坐标系 (seg0)，与 batch 内外参、点云、dynamic_info 一致。**

### 3.5 兼容性与风险

- **下游**：若有代码依赖 `segment['aabb']`（当前 grep 未发现），需改为使用 batch['aabb'] 或配置中的 crop_aabb。  
- **Trainer**：若从 dataset 或 scene 取 AABB 做 `_init_scene(scene_aabb)`，应改为从 batch 取 `batch['aabb']`，保证 seg0 一致。  
- **fixed_segment_aabb**：若仅用于“给 batch 提供固定 seg0 系 AABB”，保留；若也用于点云，需保证 pointcloud 生成器仍能拿到同一配置（例如 dataset 已把 fixed_segment_aabb 传给生成器或生成器从同一 config 读），避免不一致。

---

## 4. 小结

| 项 | 结论 |
|----|------|
| 段分割是否必须用 segment['aabb']？ | **否**。当前无人使用；段数只依赖“轨迹距离 + 场景尺度”。 |
| 能否用 crop_aabb 替代？ | **可以**。用 crop_aabb 的尺寸作为 aabb_length 参与段数计算即可。 |
| 是否移除 segment['aabb']？ | **建议移除**，避免世界系/seg0 系混用与冗余。 |
| Batch 中的 AABB 坐标系？ | **明确为 seg0 系**，并在文档与实现中统一。 |
| 修复方案 | 段分割改用 crop_aabb 长度、移除 segment['aabb']；batch 增加 seg0 系 `batch['aabb']`；文档更新。 |
