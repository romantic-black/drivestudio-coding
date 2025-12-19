# EVolSplat 评估数据加载流程详解

## 概述

本文档详细分析 EVolSplat 的评估数据加载流程，包括与训练流程的对比、关键组件和数据格式。

---

## 1. 训练 vs 评估对比表

### 1.1 数据加载方法对比

| 特性 | 训练 (`next_train`) | 评估 (`next_eval_image`) |
|------|-------------------|------------------------|
| **方法位置** | `evolsplat_datamanger.py:347` | `evolsplat_datamanger.py:402` |
| **场景选择** | 随机选择：`torch.randint(num_scenes)` | 从 eval 图像索引计算：`scene_id = image_idx // 2` |
| **Target 选择** | 场景内随机：`random.randint(start, end)` | 从 eval 数据集按索引选择 |
| **Source 选择函数** | `get_source_images_from_current_imageid()` | `eval_source_images_from_current_imageid()` |
| **Source 数据源** | 训练数据集（当前场景） | 训练数据集（当前场景） |
| **Target 数据源** | 训练数据集 | 评估数据集 |
| **排除 target** | ✅ 是（`tar_id=image_id`） | ❌ 否（`tar_id=-1`） |
| **图像索引** | 场景内索引 | 全局索引（eval 数据集） |

### 1.2 Source 选择策略对比

| 特性 | 训练 | 评估 |
|------|------|------|
| **选择函数** | `get_source_images_from_current_imageid()` | `eval_source_images_from_current_imageid()` |
| **Target ID 参数** | `tar_id=image_id`（排除 target） | `tar_id=-1`（不排除） |
| **候选图像范围** | 当前场景的训练图像 | 当前场景的训练图像 |
| **选择数量** | `num_source_image`（默认3） | `num_source_image`（默认3） |
| **距离计算方法** | 欧氏距离（`angular_dist_method='dist'`） | 欧氏距离（`angular_dist_method='dist'`） |

### 1.3 Batch 格式对比

| 数据项 | 训练 | 评估 |
|--------|------|------|
| **scene_id** | `Tensor[1]`（随机选择，但在 batch 中存储为 Tensor） | `int`（从 image_idx 计算：`image_idx // 2`） |
| **source['image']** | `Tensor[V, H, W, 3]` | `Tensor[V, H, W, 3]` |
| **source['extrinsics']** | `Tensor[V, 4, 4]` | `Tensor[V, 4, 4]` |
| **source['intrinsics']** | `Tensor[V, 4, 4]` | `Tensor[V, 4, 4]` |
| **source['depth']** | `Tensor[V, H, W]` | `Tensor[V, H, W]` |
| **source['source_id']** | 全局索引（包含 scene_id） | 训练数据集索引（不包含 scene_id） |
| **target['image']** | `Tensor[1, H, W, 3]` | `Tensor[1, H, W, 3]` |
| **target['extrinsics']** | `Tensor[1, 4, 4]` | `Tensor[1, 4, 4]` |
| **target['intrinsics']** | `Tensor[1, 4, 4]` | `Tensor[1, 4, 4]` |
| **target['target_id']** | 全局索引（包含 scene_id） | 全局索引（eval 数据集） |

---

## 2. 评估数据加载流程

### 2.1 完整流程

```
next_eval_image(step)
    ↓
1. 从 eval_unseen_cameras 随机选择一个 image_idx
   - image_idx = eval_unseen_cameras.pop(random.randint(0, len-1))
   - 如果列表为空，重新填充：[0, 1, 2, ..., len(eval_dataset)-1]
    ↓
2. 计算 scene_id
   - scene_id = image_idx // 2  # 每个场景有2张评估图像
    ↓
3. 获取场景的训练数据
   - 调用 sample_camId_from_multiscene(scene_id)
   - 返回: cur_rgbs, cur_depths, train_pose, train_cameras
   - 这些是训练数据集中的图像（用于 source）
    ↓
4. 获取评估图像（target）
   - camera = eval_dataset.cameras[image_idx:image_idx+1]
   - tar_image = read_rgb_filename(eval_dataset.image_filenames[image_idx])
    ↓
5. 选择 source 图像
   - 调用 eval_source_images_from_current_imageid()
   - 从训练数据中选择最近的 num_source_image 张图像
   - tar_id=-1，不排除 target（因为 target 在 eval 数据集，不在训练数据中）
    ↓
6. 组装 batch
   - source: 来自训练数据集
   - target: 来自评估数据集
   - scene_id: 从 image_idx 计算
    ↓
7. 返回 (camera, batch)
```

### 2.2 关键代码分析

#### 2.2.1 图像索引选择

**代码**（`evolsplat_datamanger.py` 第408-409行）:
```python
image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
scene_id = image_idx // 2
```

**说明**:
- ✅ **随机选择**：从 `eval_unseen_cameras` 列表中随机选择一个索引
- ✅ **不重复**：使用 `pop()` 确保每个图像只评估一次
- ✅ **自动重置**：当列表为空时，重新填充（第418-419行）
- ✅ **场景计算**：`scene_id = image_idx // 2` 假设每个场景有2张评估图像

#### 2.2.2 Source 图像选择

**代码**（`evolsplat_datamanger.py` 第422-427行）:
```python
source_images, src_poses, source_ids, src_depths = eval_source_images_from_current_imageid(
    rgbs=cur_rgbs,           # 训练数据集的图像路径
    depths=cur_depths,       # 训练数据集的深度图路径
    all_pose=train_pose,     # 训练数据集的相机姿态
    num_select=self.config.num_source_image,
    eval_pose=camera.camera_to_worlds  # 评估图像的相机姿态
)
```

**关键点**:
- ✅ **Source 来自训练数据**：`cur_rgbs` 和 `cur_depths` 是训练数据集的图像
- ✅ **Target 来自评估数据**：`eval_pose` 是评估图像的相机姿态
- ✅ **不排除 target**：`tar_id=-1`，因为 target 不在训练数据中

#### 2.2.3 eval_source_images_from_current_imageid 函数

**代码**（`datamanagers/utils.py` 第143-174行）:
```python
def eval_source_images_from_current_imageid(rgbs, depths, all_pose, eval_pose, num_select=2):
    # 1. 准备姿态矩阵（添加齐次坐标）
    eye = torch.tensor([0., 0., 0., 1.]).to(all_pose)
    all_pose = torch.cat([all_pose, eye[None,None,:].repeat(all_pose.shape[0],1,1)], dim=1)
    eval_pose = torch.cat([eval_pose, eye[None,None,:].repeat(eval_pose.shape[0],1,1)], dim=1)
    
    # 2. 选择最近的图像（不排除 target，因为 target 不在训练数据中）
    nearest_pose_ids = get_nearest_pose_ids(
        eval_pose.detach().cpu().numpy()[0],
        all_pose.detach().cpu().numpy(),
        num_select=num_select,
        tar_id=-1,  # 不排除 target
        angular_dist_method='dist',  # 欧氏距离
    )
    
    # 3. 加载图像和深度图
    src_rgbs = [read_rgb_filename(rgbs[i]) for i in nearest_pose_ids]
    src_depths = [get_image_depth_tensor_from_path(depths[i]) for i in nearest_pose_ids]
    
    return src_rgbs, src_poses, nearest_pose_ids, src_depths
```

**关键点**:
- ✅ **tar_id=-1**：不排除任何图像（因为 target 不在训练数据中）
- ✅ **从训练数据选择**：`all_pose` 是训练数据集的相机姿态
- ✅ **基于距离选择**：使用欧氏距离选择最近的图像

---

## 3. 关键组件详解

### 3.1 eval_unseen_cameras

**类型**: `List[int]`

**初始化**（`evolsplat_datamanger.py` 第143行）:
```python
self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
```

**用途**:
- 跟踪哪些评估图像还没有被评估
- 使用 `pop()` 确保每个图像只评估一次
- 当列表为空时，重新填充以支持多轮评估

**重置机制**（第418-419行）:
```python
if len(self.eval_unseen_cameras) == 0:
    self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
```

### 3.2 eval_dataset

**类型**: `InputDataset`

**来源**:
- 通过 `create_eval_dataset()` 创建（第280-285行）
- 使用 `dataparser.get_dataparser_outputs(split=self.test_split)` 解析数据
- `test_split` 通常是 "test" 或 "val"

**特点**:
- 包含评估用的图像和相机参数
- 每个场景通常有固定数量的评估图像（例如2张）
- 图像索引是全局的（跨场景）

### 3.3 train_dataset

**类型**: `InputDataset`

**用途**:
- 在评估时，用于提供 source 图像
- 通过 `sample_camId_from_multiscene()` 获取场景的训练数据

**关键点**:
- Source 图像来自训练数据集，而不是评估数据集
- 这允许模型使用训练时见过的图像来渲染新的视角

---

## 4. 数据流图

### 4.1 评估数据流

```
评估开始
    ↓
next_eval_image(step)
    ↓
┌─────────────────────────────────────┐
│ 1. 选择评估图像                      │
│    image_idx = eval_unseen_cameras.pop() │
│    scene_id = image_idx // 2        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. 获取场景的训练数据                │
│    sample_camId_from_multiscene(scene_id) │
│    → cur_rgbs, cur_depths, train_pose │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. 获取评估图像（target）            │
│    camera = eval_dataset.cameras[image_idx] │
│    tar_image = read_rgb_filename(...) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. 选择 source 图像                  │
│    eval_source_images_from_current_imageid() │
│    → 从训练数据中选择最近的图像      │
│    → tar_id=-1（不排除 target）      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. 组装 batch                        │
│    source: 训练数据集                │
│    target: 评估数据集                │
│    scene_id: 从 image_idx 计算      │
└─────────────────────────────────────┘
    ↓
返回 (camera, batch)
```

### 4.2 关键数据传递

```
eval_dataset
    ├─ image_filenames[image_idx]  → tar_image
    ├─ cameras[image_idx]          → camera
    └─ (全局索引)

train_dataset (通过 sample_camId_from_multiscene)
    ├─ image_filenames[start:end]  → cur_rgbs
    ├─ metadata['depth_filenames'][start:end] → cur_depths
    ├─ cameras.camera_to_worlds[start:end] → train_pose
    └─ (场景内索引)

eval_source_images_from_current_imageid
    ├─ cur_rgbs                    → source_images
    ├─ cur_depths                  → src_depths
    ├─ train_pose                  → src_poses
    └─ nearest_pose_ids            → source_ids
```

---

## 5. 与训练流程的关键差异

### 5.1 场景选择

| 特性 | 训练 | 评估 |
|------|------|------|
| **方法** | `torch.randint(num_scenes)` | `image_idx // 2` |
| **随机性** | 完全随机 | 由 image_idx 决定 |
| **可重复性** | 每次迭代都可能不同 | 每个 image_idx 对应固定的 scene_id |

### 5.2 Target 选择

| 特性 | 训练 | 评估 |
|------|------|------|
| **方法** | `random.randint(start, end)` | `eval_unseen_cameras.pop()` |
| **数据源** | 训练数据集 | 评估数据集 |
| **索引类型** | 场景内索引 | 全局索引 |
| **重复性** | 可能重复 | 不重复（使用 pop） |

### 5.3 Source 选择

| 特性 | 训练 | 评估 |
|------|------|------|
| **函数** | `get_source_images_from_current_imageid()` | `eval_source_images_from_current_imageid()` |
| **tar_id** | `image_id`（排除 target） | `-1`（不排除） |
| **数据源** | 训练数据集（当前场景） | 训练数据集（当前场景） |
| **Target 位置** | 在训练数据中 | 在评估数据中 |

### 5.4 Batch 格式差异

| 数据项 | 训练 | 评估 |
|--------|------|------|
| **scene_id** | `Tensor[1]`（但在 get_outputs 中可能需要转换为 int） | `int` |
| **source['source_id']** | 全局索引（包含 scene_id） | 训练数据集索引（不包含 scene_id） |
| **target['target_id']** | 全局索引（包含 scene_id） | 全局索引（eval 数据集） |

---

## 6. 实现注意事项

### 6.1 对于 EvolSplatDataset 的实现

**评估时的关键点**:

1. **场景 ID 计算**:
   ```python
   # 假设每个场景有固定数量的评估图像
   scene_id = image_idx // num_eval_images_per_scene
   ```

2. **Source 选择**:
   ```python
   # 从训练数据中选择 source
   # tar_id=-1，因为 target 在评估数据中，不在训练数据中
   source_indices = self._select_source_images_for_eval(
       train_images=train_images,
       eval_pose=eval_pose,
       num_source_image=num_source_image,
       tar_id=-1,  # 不排除 target
   )
   ```

3. **数据源区分**:
   ```python
   # Source 来自训练数据集
   source_images = [train_dataset.get_image(idx) for idx in source_indices]
   
   # Target 来自评估数据集
   target_image = eval_dataset.get_image(image_idx)
   ```

### 6.2 反直觉检查

- [ ] **场景 ID 计算正确**：`scene_id = image_idx // num_eval_images_per_scene`
- [ ] **Source 来自训练数据**：确保从训练数据集选择 source
- [ ] **Target 来自评估数据**：确保从评估数据集选择 target
- [ ] **不排除 target**：`tar_id=-1`，因为 target 不在训练数据中
- [ ] **索引一致性**：确保 source_id 和 target_id 的索引类型正确
- [ ] **eval_unseen_cameras 管理**：确保正确重置列表

---

## 7. 总结

### 7.1 关键发现

1. **评估时 source 来自训练数据**：Source 图像从训练数据集选择，而不是评估数据集
2. **评估时 target 来自评估数据**：Target 图像从评估数据集选择
3. **不排除 target**：因为 target 在评估数据中，不在训练数据中，所以不需要排除
4. **场景 ID 计算**：从 image_idx 计算，假设每个场景有固定数量的评估图像
5. **不重复评估**：使用 `eval_unseen_cameras.pop()` 确保每个图像只评估一次

### 7.2 设计意义

**评估时的设计**:
- ✅ **使用训练数据作为 source**：允许模型使用训练时见过的图像
- ✅ **评估新视角**：Target 来自评估数据集，测试模型的泛化能力
- ✅ **不排除 target**：因为 target 不在训练数据中，不需要排除

**与训练的区别**:
- 训练时：target 和 source 都来自训练数据，但 target 不会被选作 source
- 评估时：target 来自评估数据，source 来自训练数据，不需要排除 target

