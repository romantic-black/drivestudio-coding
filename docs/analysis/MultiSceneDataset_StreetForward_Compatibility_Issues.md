# MultiSceneDataset 与 StreetForward 兼容性问题分析

## 概述

本文档分析 `MultiSceneDataset` 与 `StreetForward` 训练器之间的三个关键兼容性问题，这些问题可能导致训练失败或动态对象对齐错误。

### 当前修复进度

- ✅ 问题 1：源关键帧数量已强制为 1（每个 batch 仅一个源关键帧/帧）
- ✅ 问题 2：点云生成器现在是必需项，缺失会报错
- ⏳ 问题 3：仍需确认 `instances_pose` 的变换方向是否总为 object→world

---

## 问题 1: 多源关键帧时间戳不一致

### 问题描述

`MultiSceneDataset` 每个 batch 采样多个源关键帧（默认 3 个），每个关键帧选择 1 帧，因此 `source` 块包含来自不同时间步的多个视图。但 `StreetForward` 转换器和训练器只使用单个 `source_frame_idx`，导致来自不同帧的源视图被错误地应用了其他帧的位姿和掩码。

### 问题位置

#### MultiSceneDataset 行为
- **文件**: `datasets/multi_scene_dataset.py`
- **行数**: 1549-1731
- **行为**: 
  - 采样 `num_source_keyframes` 个源关键帧（默认 3）
  - 每个关键帧选择 1 帧，得到多个不同的 `frame_idx`
  - 为每个 `(frame_idx, cam_idx)` 对加载图像、外参、内参
  - `source['frame_indices']` 包含所有源帧的索引：`[frame_0, frame_0, ..., frame_1, frame_1, ..., frame_2, frame_2, ...]`（每个帧重复 `num_cams` 次）

```python
# multi_scene_dataset.py:1579-1601
for frame_idx in source_frame_indices:  # 多个不同的 frame_idx
    for cam_idx in range(num_cams):
        # ... 加载图像、外参等
        source_frame_idxs.append(frame_idx)  # 记录实际的 frame_idx
```

#### StreetForward 转换器行为
- **文件**: `tools/train_streetforward.py`
- **行数**: 157-171, 197-199
- **行为**:
  - 保留所有 `source` 视图（来自不同帧）
  - 但只取第一个源帧的 `frame_idx` 作为 `source_frame_idx`

```python
# train_streetforward.py:156-171
source_frame_idx = None
if "source" in batch:
    source_data = batch["source"]
    num_source_images = source_data["image"].shape[0]
    for i in range(num_source_images):
        # ... 创建所有 source_views
        if source_frame_idx is None:  # 只设置一次
            frame_indices = source_data.get("frame_indices")
            if frame_indices is not None:
                source_frame_idx = int(frame_indices[i])  # 只取第一个
```

#### StreetForward 训练器行为
- **文件**: `models/trainers/streetforward.py`
- **行数**: 2745-2833
- **行为**:
  - 使用单个 `source_frame_idx` 对所有源视图进行：
    - 刚性可见性/掩码计算（`_per_point_pose_valid`, `_visible_mask_from_instances_fv`）
    - 2D 特征融合（`_compute_2d_features_all`）

```python
# streetforward.py:2746-2790
source_frame_idx = batch.get("source_frame_idx")  # 单个值
# ...
if node_state_rigid is not None:
    pose_valid_src = self._per_point_pose_valid(node_state_rigid, source_frame_idx)  # 对所有源视图使用同一个 frame_idx
    visible_src = self._visible_mask_from_instances_fv(node_state_rigid, source_frame_idx)
# ...
feat_2d_bg, feat_2d_rigid, feat_2d_distant = self._compute_2d_features_all(
    source_views=batch.get("source_views", []),  # 可能包含来自不同帧的视图
    source_frame_idx=source_frame_idx,  # 但只使用单个 frame_idx
)
```

### 问题影响

1. **时间扭曲**: 来自 `frame_1` 或 `frame_2` 的源视图被应用了 `frame_0` 的位姿和掩码
2. **动态对象对齐错误**: 如果动态对象在不同帧之间移动，使用错误的帧索引会导致：
   - 可见性掩码错误（对象可能在不同位置）
   - 2D 特征融合时使用错误的动态对象位姿
   - 训练损失计算不准确

### 根本原因

- `MultiSceneDataset` 设计为支持多源关键帧（用于 EVolSplat 等需要多视图特征的方法）
- `StreetForward` 转换器保留了所有源视图，但只提取了第一个帧索引
- `StreetForward` 训练器假设所有源视图来自同一帧，使用单个 `source_frame_idx`

---

## 问题 2: 点云可选性与必需性不匹配

### 问题描述

`MultiSceneDataset` 中点云是可选的（仅当配置了点云生成器时存在），但 `StreetForward` 转换器和训练器将其视为必需，导致在使用默认配置（无 `pointcloud_config`）时训练前崩溃。

### 问题位置

#### MultiSceneDataset 行为
- **文件**: `datasets/multi_scene_dataset.py`
- **行数**: 1634-1747
- **行为**:
  - 点云生成是可选的：只有当 `self.pointcloud_generator is not None` 时才生成
  - 如果点云生成器不存在，`pointcloud = None`，batch 中不包含 `pointcloud` 键

```python
# multi_scene_dataset.py:1634-1641
pointcloud = None
if self.pointcloud_generator is not None:
    pointcloud = self.pointcloud_generator.generate_pointcloud(...)
# ...
# 1747-1749
if pointcloud is not None:
    batch['pointcloud'] = pointcloud
```

#### StreetForward 转换器行为
- **文件**: `tools/train_streetforward.py`
- **行数**: 121-125
- **行为**:
  - 在点云缺失时抛出 `ValueError`

```python
# train_streetforward.py:121-125
pointcloud = batch.get("pointcloud")
if pointcloud is None:
    raise ValueError("pointcloud is required but not found in batch")
```

#### StreetForward 训练器行为
- **文件**: `models/trainers/streetforward.py`
- **行数**: 1002
- **行为**:
  - `_get_or_init_node_states` 直接访问 `batch["pointcloud"]`，假设它存在

```python
# streetforward.py:1002
pointcloud = batch["pointcloud"]  # 如果不存在会抛出 KeyError
```

### 问题影响

1. **训练前崩溃**: 使用默认配置（无 `pointcloud_config`）时，在 `convert_batch_to_streetforward_format` 阶段就会失败
2. **配置不灵活**: 无法在不需要点云的情况下使用 `StreetForward`（虽然这可能不是预期用例）

### 根本原因

- `MultiSceneDataset` 设计为通用数据集，点云是可选的（某些方法可能不需要）
- `StreetForward` 训练器依赖点云来初始化 `NodeState`（背景和动态对象）
- 转换器没有处理点云缺失的情况

---

## 问题 3: 动态位姿变换方向未验证

### 问题描述

`MultiSceneDataset` 从 `pixel_source.instances_pose` 提取动态对象位姿时，直接将其作为对象到世界的变换（object→world）使用，但未验证该假设。如果 `instances_pose` 实际上是世界到对象（world→object）或自车到对象（ego→object）的变换，则会导致动态对象被放置在错误的坐标系中。

### 问题位置

#### MultiSceneDataset 行为
- **文件**: `datasets/multi_scene_dataset.py`
- **行数**: 1360-1505
- **行为**:
  - 从 `instances_pose[frame_idx, instance_id]` 提取 4×4 位姿矩阵
  - 直接提取旋转矩阵和平移向量，转换为四元数
  - 未检查或转换变换方向

```python
# multi_scene_dataset.py:1428-1440
for instance_id in visible_instance_ids:
    pose_matrix = instances_pose[frame_idx, instance_id]  # [4, 4]
    rot_matrix = pose_matrix[:3, :3]  # [3, 3]
    trans = pose_matrix[:3, 3]  # [3]
    # 转换为四元数，直接使用
    quat = ...  # 从 rot_matrix 转换
```

#### StreetForward 训练器行为
- **文件**: `models/trainers/streetforward.py`
- **行数**: 1334-1391, 2745-2790
- **行为**:
  - 使用 `instances_quats` 和 `instances_trans` 作为对象到世界的变换
  - 在 `_transform_rigid_to_world` 中应用：`means_world = R * means_local + t`

```python
# streetforward.py:1334-1383
def _transform_rigid_to_world(self, node_state_rigid, means_local, ...):
    # ...
    quats_cur_frame = node_state_rigid.instances_quats[frame_idx]
    trans_cur_frame = node_state_rigid.instances_trans[frame_idx]
    rot_cur_frame = _quat_to_rotmat(quats_cur_frame)
    # 假设这是 object→world 变换
    means_world = torch.bmm(rot_per_pts, means_local.unsqueeze(-1)).squeeze(-1) + trans_per_pts
```

#### 其他代码中的线索
- **文件**: `datasets/driving_dataset.py`
- **行数**: 230-236
- **行为**:
  - `seg_dynamic_instances_in_lidar_frame` 中，将 `instances_pose` 视为 `o2w`（object→world），然后求逆得到 `w2o`（world→object）

```python
# driving_dataset.py:230-236
o2w = self.pixel_source.instances_pose[frame_idx, instance_id]  # 假设是 object→world
w2o = torch.inverse(o2w)  # 求逆得到 world→object
o_pts = transform_points(lidar_pts, w2o)  # 将世界坐标点转换到对象坐标系
```

### 问题影响

1. **坐标系错误**: 如果 `instances_pose` 实际上是 world→object 或 ego→object，动态对象会被放置在错误的位置
2. **训练失败**: 动态对象与静态场景不对齐，导致渲染错误和训练损失异常

### 根本原因

- **未验证假设**: 代码假设 `instances_pose` 是 object→world，但未在文档或代码中明确说明
- **数据集差异**: 不同驾驶数据集（KITTI、NuScenes、Waymo 等）可能使用不同的变换约定
- **缺少转换**: 如果 `instances_pose` 是其他方向，需要在 `_build_dynamic_info` 中求逆

### 开放问题

**需要确认**: `instances_pose` 矩阵存储的是 object→world 还是 world→object？

- 如果确认是 **object→world**：当前实现正确
- 如果确认是 **world→object**：需要在 `_build_dynamic_info` 中求逆：
  ```python
  pose_matrix = torch.inverse(instances_pose[frame_idx, instance_id])
  ```

---

## 建议修复方案

### 问题 1: 多源关键帧时间戳

**方案 A（推荐）**: 为每个源视图使用正确的 `frame_idx`
- 修改 `_build_3d_feature_volume` 和 `_compute_2d_features_all`，接受每个源视图的 `frame_idx`
- 修改转换器，传递 `source_frame_indices` 列表而不是单个 `source_frame_idx`

**方案 B**: 限制源视图为单个关键帧
- 修改 `MultiSceneDataset` 配置，设置 `num_source_keyframes=1`
- 或修改转换器，只保留第一个源关键帧的视图

### 问题 2: 点云可选性

**方案 A（推荐）**: 使点云在 StreetForward 中可选
- 修改转换器，检查点云是否存在，如果不存在则提供占位符或跳过初始化
- 修改 `_get_or_init_node_states`，处理点云缺失的情况

**方案 B**: 在配置验证阶段要求点云
- 在训练脚本启动时检查 `pointcloud_config` 是否存在
- 如果不存在，提前报错并提示用户配置点云生成器

### 问题 3: 动态位姿变换方向

**方案 A（推荐）**: 验证并文档化变换方向
- 检查各数据集的 `instances_pose` 约定（查看预处理代码或数据集文档）
- 在 `_build_dynamic_info` 中添加配置选项或自动检测
- 如果发现是 world→object，添加求逆逻辑

**方案 B**: 添加配置参数
- 在配置中添加 `instances_pose_direction: "object_to_world" | "world_to_object"`
- 在 `_build_dynamic_info` 中根据配置决定是否求逆

---

## 相关文件

- `datasets/multi_scene_dataset.py`: MultiSceneDataset 实现
- `tools/train_streetforward.py`: StreetForward 批处理转换器
- `models/trainers/streetforward.py`: StreetForward 训练器
- `docs/dataloader/MultiSceneDataset_Usage.md`: MultiSceneDataset 使用文档

---

## 更新日期

2026-01-26
