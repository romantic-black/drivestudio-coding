# MultiSceneDataset 段分割逻辑详解

## 概述

本文档详细解释 `MultiSceneDataset` 中段（Segment）分割的完整逻辑，包括关键帧分割、段分割、参数影响以及与AABB的关系。

## 数据层次结构

在理解段分割之前，需要明确数据层次结构：

```
场景 (Scene)
  └── 段 (Segment) - 按照场景AABB和关键帧距离分割，包含多个关键帧
      └── 关键帧 (Keyframe) - 按照轨迹距离分割的小段，至少包含一帧
          └── 帧 (Frame) - 时间步，包含多张图像（多相机）
```

**关键概念**：
- **关键帧（Keyframe）**：按照轨迹距离分割的小段，每个关键帧包含一个或多个帧
- **段（Segment）**：多个关键帧的组合，用于构建独立的3DGS场景
- **AABB（Axis-Aligned Bounding Box）**：轴对齐边界框，用于定义3D空间范围

---

## 完整流程

### 阶段1：关键帧分割（Keyframe Splitting）

#### 1.1 输入数据

```python
trajectory: Tensor[num_frames, 4, 4]  # 相机变换矩阵
```

轨迹是从场景的前置相机获取的，表示车辆在场景中的运动路径。

#### 1.2 关键帧分割过程

调用 `split_trajectory` 函数（位于 `datasets/tools/trajectory_utils.py`）：

```python
def _split_keyframes(
    self,
    trajectory: Tensor,  # [num_frames, 4, 4]
) -> Tuple[List[List[int]], Tensor]:
    """
    按照距离分割关键帧。
    
    Returns:
        keyframe_segments: List[List[int]] - 每个关键帧段包含的帧索引列表
        keyframe_ranges: Tensor[num_keyframes, 2] - 每个关键帧段的距离范围
    """
    keyframe_segments, keyframe_ranges = split_trajectory(
        trajectory=trajectory,
        num_splits=self.keyframe_split_config['num_splits'],
        min_count=self.keyframe_split_config['min_count'],
        min_length=self.keyframe_split_config['min_length'],
    )
    return keyframe_segments, keyframe_ranges
```

#### 1.3 split_trajectory 函数详解

**步骤1：计算累积距离**

```python
# 提取位置信息（从变换矩阵的平移部分）
positions = trajectory[:, :3, 3]  # [num_frames, 3]

# 计算相邻帧之间的位移
delta_positions = positions[1:] - positions[:-1]  # [num_frames-1, 3]

# 计算相邻帧之间的欧氏距离
distances = torch.norm(delta_positions, dim=1)  # [num_frames-1]

# 计算累积距离（从第一帧到每一帧的总距离）
cumulative_distances = torch.cat([
    torch.tensor([0.0]),
    torch.cumsum(distances, dim=0)
])  # [num_frames]

total_distance = cumulative_distances[-1]  # 总行驶距离
```

**步骤2：确定关键帧数量**

如果 `num_splits == 0`（自动模式）：
- 从最大可能的分割数开始（等于帧数）
- 逐步减少，找到满足约束条件的最大分割数
- 约束条件：
  - 每个关键帧段至少包含 `min_count` 帧
  - 每个关键帧段的距离至少为 `min_length`

如果 `num_splits > 0`（手动指定）：
- 直接使用指定的分割数

**步骤3：分配帧到关键帧段**

```python
segment_length = total_distance / num_splits
segment_indices = (cumulative_distances / segment_length).long()
segment_indices = torch.clamp(segment_indices, max=num_splits - 1)
```

**步骤4：生成关键帧段和距离范围**

```python
# 每个关键帧段包含的帧索引
keyframe_segments = [
    [frame_idx for frame_idx, seg_idx in enumerate(segment_indices) if seg_idx == kf_idx]
    for kf_idx in range(num_splits)
]

# 每个关键帧段的距离范围 [start_distance, end_distance]
boundaries = torch.linspace(0, total_distance, steps=num_splits + 1)
keyframe_ranges = torch.stack([
    boundaries[:-1],  # start distances
    boundaries[1:]    # end distances
], dim=1)  # [num_keyframes, 2]
```

#### 1.4 关键帧分割参数

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `num_splits` | 关键帧分割数量 | 0（自动） | 0表示自动确定，>0表示手动指定 |
| `min_count` | 每个关键帧段的最小帧数 | 1 | 确保每个关键帧段有足够的帧 |
| `min_length` | 每个关键帧段的最小距离 | 0.0 | 确保每个关键帧段有足够的空间跨度 |

**示例**：
- 如果场景有100帧，总距离100米，`min_count=5`，`min_length=2.0`
- 自动模式可能生成20个关键帧段，每个约5米，包含5帧

---

### 阶段2：段分割（Segment Splitting）

#### 2.1 输入数据

```python
scene_dataset: DrivingDataset  # 场景数据集
keyframe_segments: List[List[int]]  # 关键帧段列表，每个元素是帧索引列表
keyframe_ranges: Tensor[num_keyframes, 2]  # 每个关键帧段的距离范围 [start, end]
overlap_ratio: float  # 段之间的重叠比例（0-1）
```

#### 2.2 段分割流程

##### 步骤1：获取场景AABB

```python
scene_aabb = scene_dataset.get_aabb()  # [2, 3] - [min, max]
scene_size = scene_aabb[1] - scene_aabb[0]  # [3] - 各维度尺寸
aabb_length = scene_size.max().item()  # 最大维度（通常是X或Y，取决于行驶方向）
```

**AABB的作用**：
- **判断场景规模**：AABB的长度反映场景的空间范围
- **决定段数**：通过比较关键帧总距离与AABB长度，判断需要分成多少个段
- **不直接用于段边界**：段的AABB是独立计算的，场景AABB只用于判断段数

##### 步骤2：计算关键帧总距离

```python
# 每个关键帧段的长度
keyframe_lengths = keyframe_ranges[:, 1] - keyframe_ranges[:, 0]  # [num_keyframes]

# 所有关键帧段的总距离
total_keyframe_distance = keyframe_lengths.sum().item()
```

**关键帧距离的含义**：
- 表示车辆在关键帧段内的实际行驶距离
- 用于判断车辆移动范围与场景大小的关系

##### 步骤3：确定段数量

```python
if total_keyframe_distance < aabb_length * 0.3:
    # 车辆移动距离短（小于场景长度的30%），只创建一个段
    num_segments = 1
else:
    # 车辆移动距离较长，可以分成多个段
    # 约束1：每个段至少需要 min_keyframes_per_segment 个关键帧
    max_segments = len(keyframe_segments) // self.min_keyframes_per_segment
    
    # 约束2：根据距离比例计算段数
    distance_ratio = total_keyframe_distance / aabb_length
    num_segments_by_distance = max(2, int(distance_ratio * 3))
    
    # 取两者的最小值
    num_segments = max(1, min(max_segments, num_segments_by_distance))
```

**段数决策逻辑**：

| 情况 | 段数 | 原因 |
|------|------|------|
| `total_distance < 0.3 * aabb_length` | 1 | 车辆移动距离短，场景可能主要是静态的 |
| `total_distance >= 0.3 * aabb_length` | `min(max_segments, distance_ratio * 3)` | 车辆移动距离较长，可以分割 |

**示例**：
- 场景AABB长度：100米
- 关键帧总距离：80米
- 关键帧数量：40个
- `min_keyframes_per_segment = 6`

计算：
- `max_segments = 40 // 6 = 6`
- `distance_ratio = 80 / 100 = 0.8`
- `num_segments_by_distance = max(2, int(0.8 * 3)) = max(2, 2) = 2`
- `num_segments = min(6, 2) = 2`

结果：分成2个段

##### 步骤4：分组关键帧到段

**情况A：只有一个段**

```python
if num_segments == 1:
    # 包含所有关键帧
    all_frames = []
    for kf_seg in keyframe_segments:
        all_frames.extend(kf_seg)
    
    frame_indices = sorted(list(set(all_frames)))
    segment_aabb = self._compute_segment_aabb(scene_dataset, frame_indices)
    
    segments.append({
        'segment_id': 0,
        'keyframe_indices': list(range(len(keyframe_segments))),  # 所有关键帧索引
        'frame_indices': frame_indices,
        'aabb': segment_aabb,
    })
```

**情况B：多个段（支持重叠）**

```python
else:
    # 计算段距离和步长
    segment_distance = total_keyframe_distance / num_segments  # 每个段的距离长度
    overlap_ratio_clamped = min(overlap_ratio, 0.5)  # 限制最大重叠为50%
    step_distance = segment_distance * (1 - overlap_ratio_clamped)  # 段之间的步长
    
    # 计算可以生成多少个重叠段
    max_start_distance = total_keyframe_distance - segment_distance
    if step_distance > 0:
        num_overlap_segments = int(max_start_distance / step_distance) + 1
    else:
        num_overlap_segments = 1
    
    # 生成重叠段
    for seg_idx in range(num_overlap_segments):
        segment_start_distance = seg_idx * step_distance
        segment_end_distance = segment_start_distance + segment_distance
        
        # 收集该段内的关键帧
        current_segment_kf_indices = []
        current_segment_frames = set()
        
        for kf_idx in range(len(keyframe_segments)):
            # 计算关键帧的中心距离
            kf_center_distance = (keyframe_ranges[kf_idx, 0] + keyframe_ranges[kf_idx, 1]) / 2.0
            
            # 检查关键帧是否在该段的距离范围内
            if segment_start_distance <= kf_center_distance < segment_end_distance:
                current_segment_kf_indices.append(kf_idx)
                current_segment_frames.update(keyframe_segments[kf_idx])
        
        # 只添加关键帧数量足够的段
        if len(current_segment_kf_indices) >= self.min_keyframes_per_segment:
            frame_indices = sorted(list(current_segment_frames))
            segment_aabb = self._compute_segment_aabb(scene_dataset, frame_indices)
            
            segments.append({
                'segment_id': segment_id,
                'keyframe_indices': current_segment_kf_indices,
                'frame_indices': frame_indices,
                'aabb': segment_aabb,
            })
            segment_id += 1
```

**重叠段生成示例**：

假设：
- `total_keyframe_distance = 100米`
- `num_segments = 3`（目标段数）
- `overlap_ratio = 0.2`

计算：
- `segment_distance = 100 / 3 = 33.33米`
- `step_distance = 33.33 * (1 - 0.2) = 26.67米`
- `max_start_distance = 100 - 33.33 = 66.67米`
- `num_overlap_segments = int(66.67 / 26.67) + 1 = 3`

生成的段：
- 段0：距离范围 [0, 33.33]
- 段1：距离范围 [26.67, 60.00]（与段0重叠6.67米）
- 段2：距离范围 [53.34, 86.67]（与段1重叠6.67米）

##### 步骤5：计算段AABB

每个段都会调用 `_compute_segment_aabb` 计算独立的AABB：

```python
segment_aabb = self._compute_segment_aabb(scene_dataset, frame_indices)
```

**段AABB的计算**：
- 基于段内帧的lidar数据
- 使用分位数方法（percentile）计算边界
- 与场景AABB独立，反映段内实际数据范围

详见 `_compute_segment_aabb` 方法的文档。

##### 步骤6：过滤无效段

```python
valid_segments = [
    seg for seg in segments
    if len(seg['keyframe_indices']) >= self.min_keyframes_per_segment
]
```

---

## 参数详解

### 关键帧分割参数（keyframe_split_config）

| 参数 | 类型 | 默认值 | 说明 | 影响 |
|------|------|--------|------|------|
| `num_splits` | int | 0 | 关键帧分割数量，0表示自动 | 控制关键帧的粒度 |
| `min_count` | int | 1 | 每个关键帧段的最小帧数 | 确保关键帧段有足够帧 |
| `min_length` | float | 0.0 | 每个关键帧段的最小距离（米） | 确保关键帧段有足够空间跨度 |

### 段分割参数

| 参数 | 类型 | 默认值 | 说明 | 影响 |
|------|------|--------|------|------|
| `min_keyframes_per_segment` | int | 6 | 每个段的最小关键帧数量 | 过滤掉关键帧太少的段 |
| `overlap_ratio` | float | 0.2 | 段之间的重叠比例 | 控制段之间的重叠程度 |

### 场景AABB参数（通过lidar_source.data_cfg）

| 参数 | 类型 | 默认值 | 说明 | 影响 |
|------|------|--------|------|------|
| `lidar_downsample_factor` | int | 4 | Lidar点下采样因子 | 影响AABB计算速度 |
| `lidar_percentile` | float | 0.02 | 分位数（用于AABB计算） | 影响AABB的紧密度 |

---

## AABB的作用和影响

### 场景AABB的作用

1. **判断场景规模**
   - 通过 `aabb_length = scene_size.max()` 获取场景的最大维度
   - 用于判断场景的空间范围

2. **决定段数量**
   - 比较 `total_keyframe_distance` 与 `aabb_length`
   - 如果 `total_distance < 0.3 * aabb_length`，只创建1个段
   - 否则，根据距离比例计算段数

3. **不直接用于段边界**
   - 场景AABB不直接作为段的边界
   - 每个段有独立的AABB（基于段内帧的lidar数据计算）

### 段AABB的作用

1. **定义段的空间范围**
   - 每个段有独立的AABB，反映段内实际数据范围
   - 用于3DGS场景的初始化

2. **与场景AABB的关系**
   - 段AABB通常包含在场景AABB内
   - 但可能不完全一致（由于分位数计算和下采样）
   - 不同段的AABB可能不同，反映不同的空间分布

---

## 关键帧与段的联系

### 关键帧到段的映射

```
关键帧分割（基于轨迹距离）
  ↓
keyframe_segments: [[0,1,2], [3,4,5], [6,7,8], ...]
keyframe_ranges: [[0, 5], [5, 10], [10, 15], ...]
  ↓
段分割（基于关键帧距离和场景AABB）
  ↓
segments: [
  {
    'keyframe_indices': [0, 1, 2],  # 包含的关键帧索引
    'frame_indices': [0,1,2,3,4,5,6,7,8],  # 包含的所有帧索引
    'aabb': Tensor[2, 3]
  },
  ...
]
```

### 关键概念

1. **关键帧索引（keyframe_indices）**
   - 段内包含的关键帧的索引（全局关键帧索引）
   - 例如：`[0, 1, 2]` 表示包含第0、1、2个关键帧

2. **帧索引（frame_indices）**
   - 段内包含的所有帧的索引（去重后）
   - 例如：如果关键帧0包含帧[0,1,2]，关键帧1包含帧[3,4,5]，则 `frame_indices = [0,1,2,3,4,5]`

3. **距离范围（keyframe_ranges）**
   - 每个关键帧段的距离范围 `[start_distance, end_distance]`
   - 用于判断关键帧属于哪个段

---

## 完整示例

### 示例场景

假设有一个场景：
- 总帧数：100帧
- 场景AABB：`[[-50, -20, 0], [50, 20, 10]]`（X: 100米，Y: 40米，Z: 10米）
- 总行驶距离：80米
- 关键帧配置：`num_splits=0`（自动），`min_count=5`，`min_length=2.0`
- 段配置：`min_keyframes_per_segment=6`，`overlap_ratio=0.2`

### 步骤1：关键帧分割

自动分割可能生成：
- 关键帧数量：16个（每个约5米，包含约6帧）
- `keyframe_segments`: `[[0,1,2,3,4,5], [6,7,8,9,10,11], ...]`
- `keyframe_ranges`: `[[0, 5], [5, 10], [10, 15], ...]`

### 步骤2：段分割

计算：
- `aabb_length = max(100, 40, 10) = 100米`
- `total_keyframe_distance = 80米`
- `distance_ratio = 80 / 100 = 0.8`
- `max_segments = 16 // 6 = 2`
- `num_segments_by_distance = max(2, int(0.8 * 3)) = 2`
- `num_segments = min(2, 2) = 2`

生成2个段：
- `segment_distance = 80 / 2 = 40米`
- `step_distance = 40 * (1 - 0.2) = 32米`

段0：
- 距离范围：[0, 40]
- 包含关键帧：0-7（假设）
- 帧索引：[0-47]（假设）
- AABB：基于帧[0-47]的lidar数据计算

段1：
- 距离范围：[32, 72]（与段0重叠8米）
- 包含关键帧：6-13（假设，与段0重叠）
- 帧索引：[36-83]（假设）
- AABB：基于帧[36-83]的lidar数据计算

---

## 常见问题

### Q1: 为什么需要关键帧分割？

**A**: 关键帧分割将连续的轨迹按照距离分成小段，每个关键帧段代表车辆行驶的一小段距离。这样可以：
- 控制关键帧的粒度
- 确保每个关键帧段有足够的空间跨度
- 为后续的段分割提供基础

### Q2: 段分割为什么使用场景AABB？

**A**: 场景AABB用于判断场景的空间规模，从而决定需要分成多少个段：
- 如果车辆移动距离远小于场景大小，可能只需要1个段
- 如果车辆移动距离接近场景大小，可以分成多个段
- 场景AABB不直接作为段的边界，段的边界由段内lidar数据计算

### Q3: 重叠段的作用是什么？

**A**: 重叠段可以：
- 提供更多的训练数据
- 增加段之间的连续性
- 避免段边界处的数据丢失

### Q4: 段AABB和场景AABB的关系？

**A**: 
- 场景AABB：整个场景的空间范围，用于判断段数
- 段AABB：段内帧的实际数据范围，用于3DGS初始化
- 段AABB通常包含在场景AABB内，但可能不完全一致

### Q5: 如何调整段的数量？

**A**: 可以通过以下方式调整：
1. **调整关键帧数量**：增加/减少 `num_splits` 或调整 `min_count`/`min_length`
2. **调整 `min_keyframes_per_segment`**：减少会允许更多段，增加会减少段数
3. **调整场景AABB**：如果场景AABB计算不准确，可能影响段数判断

---

## 总结

段分割是一个两阶段过程：

1. **关键帧分割**：基于轨迹距离，将连续帧分成关键帧段
2. **段分割**：基于关键帧距离和场景AABB，将关键帧组合成段

关键参数：
- 关键帧分割：`num_splits`, `min_count`, `min_length`
- 段分割：`min_keyframes_per_segment`, `overlap_ratio`
- AABB：场景AABB用于判断段数，段AABB用于定义段的空间范围

段分割的目标是创建合适数量的段，每个段包含足够的关键帧和帧，并且有合理的空间范围。

