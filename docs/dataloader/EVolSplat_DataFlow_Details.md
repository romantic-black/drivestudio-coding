# EVolSplat 数据流细节分析

## 概述

本文档详细分析 EVolSplat 的数据流，包括：
1. `get_outputs` 方法对 batch 的要求
2. Source-Target 图像对的数量和选择策略
3. 数据传递和训练步骤的反直觉检查
4. 数据提供的顺序、规律和规则

---

## 1. get_outputs 方法的 batch 实现分析

### 1.1 代码分析

从 `evolsplat.py` 的 `get_outputs` 方法（第422-600行）可以看到：

```python
def get_outputs(self, camera: Cameras, batch) -> Dict[str, Union[torch.Tensor, List]]:
    scene_id = batch.get("scene_id", None)
    
    # 从 batch 中提取 source 图像
    source_images = batch['source']['image']  # 期望格式: [V, H, W, 3]
    source_images = rearrange(source_images[None,...],"b v h w c -> b v c h w")
    # 转换后: [1, V, C, H, W]，然后 squeeze(0) 得到 [V, C, H, W]
    
    source_extrinsics = batch['source']['extrinsics']  # [V, 4, 4]
    
    # 从 batch 中提取 target 图像
    target_image = batch['target']['image'].squeeze(0)  # [H, W, 3]
    
    # 使用 source 图像提取 2D 特征
    sampled_feat, valid_mask, vis_map = self.projector.sample_within_window(
        xyz=means,
        train_imgs=source_images.squeeze(0),  # [V, C, H, W]
        train_cameras=source_extrinsics,      # [V, 4, 4]
        train_intrinsics=batch['source']['intrinsics'],  # [V, 4, 4]
        source_depth=batch['source']['depth'],  # [V, H, W]
        local_radius=self.local_radius,
    )
```

### 1.2 Batch 格式要求

**EVolSplat 期望的 batch 格式**：

```python
batch = {
    'scene_id': Tensor[1] 或 int,  # 场景ID，用于索引点云和offset
    
    'source': {
        'image': Tensor[V, H, W, 3],      # V 张 source 图像（默认 V=3）
        'extrinsics': Tensor[V, 4, 4],     # source 相机外参
        'intrinsics': Tensor[V, 4, 4],     # source 相机内参（必须是4x4）
        'depth': Tensor[V, H, W],          # source 深度图（可选，但推荐）
    },
    
    'target': {
        'image': Tensor[1, H, W, 3] 或 Tensor[H, W, 3],  # 1 张 target 图像
        'extrinsics': Tensor[1, 4, 4],     # target 相机外参
        'intrinsics': Tensor[1, 4, 4],     # target 相机内参（必须是4x4）
    }
}
```

**关键点**：
1. **Source 图像数量**: `V` 张（默认 `num_source_image=3`）
2. **Target 图像数量**: 1 张（每次 `get_outputs` 只渲染一个视角）
3. **内参格式**: 必须是 4x4 矩阵（即使原始是 3x3，也需要转换）
4. **深度图**: Source 图像必须有深度图，用于遮挡感知的特征提取

### 1.3 EvolSplatDataset 如何提供合适的 batch

基于代码分析，`EvolSplatDataset.get_batch()` 应该：

```python
def get_batch(
    self,
    scene_id: int,
    target_image_idx: int,
    num_source_image: Optional[int] = None,
) -> Dict:
    """
    返回符合 EVolSplat get_outputs 要求的 batch。
    """
    if num_source_image is None:
        num_source_image = self.num_source_image  # 默认3
    
    # 1. 获取 target 图像（1张）
    target_image_infos, target_cam_infos = scene_dataset.pixel_source.get_image(target_image_idx)
    target_image = target_image_infos['pixels']  # [H, W, 3]
    target_extrinsic = target_cam_infos['camera_to_world']  # [4, 4]
    target_intrinsic = target_cam_infos['intrinsics']  # [3, 3]
    
    # 2. 选择 source 图像（V张，默认3张）
    source_indices = self._select_source_images(
        scene_dataset=scene_dataset,
        target_image_idx=target_image_idx,
        target_extrinsic=target_extrinsic,
        num_source_image=num_source_image,
    )
    
    # 3. 获取 source 图像和参数
    source_images = []
    source_extrinsics = []
    source_intrinsics = []
    source_depths = []
    
    for source_idx in source_indices:
        source_image_infos, source_cam_infos = scene_dataset.pixel_source.get_image(source_idx)
        
        source_images.append(source_image_infos['pixels'])  # [H, W, 3]
        source_extrinsics.append(source_cam_infos['camera_to_world'])  # [4, 4]
        
        # 重要：内参必须转换为 4x4
        intrinsic_3x3 = source_cam_infos['intrinsics']  # [3, 3]
        intrinsic_4x4 = self._convert_intrinsic_to_4x4(intrinsic_3x3)
        source_intrinsics.append(intrinsic_4x4)
        
        # 获取深度图
        depth = self._get_depth_from_driving_dataset(scene_dataset, source_idx)
        if depth is None:
            depth = self._load_depth_from_file(scene_id, source_idx)
        source_depths.append(depth)
    
    # 4. 组装 batch（确保维度正确）
    batch = {
        'scene_id': torch.tensor([scene_id], dtype=torch.long),  # [1]
        
        'source': {
            'image': torch.stack(source_images, dim=0),  # [V, H, W, 3]
            'extrinsics': torch.stack(source_extrinsics, dim=0),  # [V, 4, 4]
            'intrinsics': torch.stack(source_intrinsics, dim=0),  # [V, 4, 4]
            'depth': torch.stack(source_depths, dim=0) if source_depths[0] is not None else None,  # [V, H, W]
        },
        
        'target': {
            'image': target_image.unsqueeze(0) if target_image.dim() == 3 else target_image,  # [1, H, W, 3]
            'extrinsics': target_extrinsic.unsqueeze(0),  # [1, 4, 4]
            'intrinsics': self._convert_intrinsic_to_4x4(target_intrinsic).unsqueeze(0),  # [1, 4, 4]
        }
    }
    
    return batch
```

---

## 2. Source-Target 图像对的数量分析

### 2.1 数量确认

**每次 `get_outputs` 调用**：
- **Source 图像**: `V` 张（默认 `V=3`，可配置 `num_source_image`）
- **Target 图像**: `1` 张（固定，每次只渲染一个视角）

**原因**：
1. EVolSplat 是 feed-forward 方法，每次前向传播只预测一个视角的高斯参数
2. Source 图像用于提取多视角特征，帮助预测高斯参数
3. Target 图像用于监督学习（计算损失）

### 2.2 Source 图像选择策略

从代码和文档分析，source 图像的选择遵循以下规则：

1. **最近邻选择**: 根据相机位置（欧氏距离）选择最近的 `V` 张图像
2. **排除 target**: 不能选择 target 图像本身作为 source
3. **同一场景**: Source 和 target 必须来自同一场景
4. **排序**: 按距离从近到远排序

**伪代码**：
```python
def _select_source_images(
    self,
    scene_dataset: DrivingDataset,
    target_image_idx: int,
    target_extrinsic: Tensor,
    num_source_image: int,
) -> List[int]:
    """
    选择最近的 num_source_image 张 source 图像。
    """
    # 1. 提取 target 相机位置
    target_cam_pos = target_extrinsic[:3, 3]  # [3]
    
    # 2. 计算所有其他图像的相机位置
    all_cam_positions = []
    all_image_indices = []
    
    num_images = scene_dataset.pixel_source.num_imgs
    for img_idx in range(num_images):
        if img_idx == target_image_idx:
            continue  # 排除 target 本身
        
        _, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
        cam_pos = cam_infos['camera_to_world'][:3, 3]
        all_cam_positions.append(cam_pos)
        all_image_indices.append(img_idx)
    
    # 3. 计算距离并选择最近的
    all_cam_positions = torch.stack(all_cam_positions, dim=0)  # [N, 3]
    distances = torch.norm(
        all_cam_positions - target_cam_pos.unsqueeze(0),
        dim=1
    )  # [N]
    
    # 4. 选择最近的 num_source_image 张
    _, nearest_indices = torch.topk(
        distances,
        k=min(num_source_image, len(all_image_indices)),
        largest=False
    )
    
    selected_indices = [all_image_indices[i] for i in nearest_indices.cpu().numpy()]
    
    return selected_indices
```

### 2.3 边界情况处理

**问题1**: 如果场景中图像数量少于 `num_source_image + 1`（包括 target）怎么办？

**解决方案**:
```python
# 在 _select_source_images 中
if len(all_image_indices) < num_source_image:
    # 如果可用图像不足，重复使用最近的图像
    # 或者减少 source 图像数量
    num_source_image = len(all_image_indices)
    if num_source_image == 0:
        # 如果没有其他图像，使用 target 本身（不推荐，但可以避免错误）
        return [target_image_idx] * num_source_image
```

**问题2**: 如果选择的 source 图像与 target 图像时间戳相同（同一帧的不同相机）怎么办？

**解决方案**: 这是允许的，因为不同相机的视角不同，可以提供多视角信息。

---

## 3. 数据传递和训练步骤的反直觉检查

### 3.1 训练循环数据流

**标准训练循环**（从 `VanillaPipeline.get_train_loss_dict()`）：

```python
# 1. 获取批次数据
ray_bundle, batch = self.datamanager.next_train(step)
# 注意：对于 EVolSplat，ray_bundle 实际上是 Cameras 对象

# 2. 前向传播
model_outputs = self.model(ray_bundle, batch)
# 内部调用: model.get_outputs(camera, batch)

# 3. 计算损失
loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

# 4. 反向传播
loss.backward()

# 5. 优化器更新
optimizer.step()
```

### 3.2 关键数据传递规则

#### 规则1: scene_id 的传递

**问题**: `scene_id` 如何从 datamanager 传递到 model？

**流程**:
```
EvolSplatDataset.get_batch()
    ↓
batch['scene_id'] = torch.tensor([scene_id])  # [1]
    ↓
datamanager.next_train(step)
    ↓
Pipeline.get_train_loss_dict(step)
    ↓
model.get_outputs(camera, batch)
    ↓
scene_id = batch.get("scene_id", None)  # 提取场景ID
means = self.means[scene_id]  # 索引点云
```

**反直觉检查**:
- ✅ `scene_id` 必须是整数或 `Tensor[1]`
- ✅ `scene_id` 必须在 `[0, num_scenes)` 范围内
- ⚠️ 如果 `scene_id` 是 `Tensor`，需要确保可以索引（可能需要 `.item()`）

#### 规则2: 内参格式转换

**问题**: 为什么内参必须是 4x4？

**原因**: 
- `projector.sample_within_window()` 期望内参是 4x4 格式
- 虽然 3x3 内参包含所有信息，但 4x4 格式便于与 4x4 外参矩阵统一处理

**转换规则**:
```python
def _convert_intrinsic_to_4x4(self, intrinsic: Tensor) -> Tensor:
    """
    将 3x3 内参转换为 4x4 格式。
    """
    if intrinsic.shape == (4, 4):
        return intrinsic
    
    assert intrinsic.shape == (3, 3), f"Unexpected shape: {intrinsic.shape}"
    
    intrinsic_4x4 = torch.eye(4, dtype=intrinsic.dtype, device=intrinsic.device)
    intrinsic_4x4[:3, :3] = intrinsic
    
    return intrinsic_4x4
```

**反直觉检查**:
- ✅ 转换后的 4x4 矩阵的 `[3, 3]` 位置必须是 1.0
- ✅ 转换后的 4x4 矩阵的 `[3, :3]` 和 `[:3, 3]` 必须是 0.0
- ⚠️ 不要直接 pad，应该创建新的 4x4 单位矩阵然后赋值

#### 规则3: 深度图的必要性

**问题**: Source 图像是否必须有深度图？

**分析**:
- 从代码看，`batch['source']['depth']` 被传递给 `projector.sample_within_window()`
- 深度图用于遮挡感知的特征提取
- 如果深度图为 `None`，可能导致错误

**解决方案**:
```python
# 在 get_batch 中
if depth is None:
    # 如果深度图不存在，创建一个全零的占位符（不推荐）
    # 或者从图像尺寸推断一个合理的深度范围
    H, W = source_image.shape[:2]
    depth = torch.zeros(H, W, dtype=torch.float32, device=source_image.device)
    # 或者使用一个固定的深度值
    depth = torch.ones(H, W, dtype=torch.float32, device=source_image.device) * 10.0
```

**反直觉检查**:
- ⚠️ 深度图尺寸必须与图像尺寸匹配 `[H, W]`
- ⚠️ 深度值应该在合理范围内（例如 0.1 到 200 米）
- ⚠️ 深度图不能包含 NaN 或 inf 值

#### 规则4: 图像尺寸一致性

**问题**: Source 和 target 图像的尺寸是否需要一致？

**分析**:
- 从代码看，`target_image.shape[:2]` 用于确定渲染尺寸
- Source 图像用于特征提取，尺寸可以不同（但需要正确处理）

**规则**:
- ✅ Source 图像之间尺寸应该一致（便于 stack）
- ✅ Target 图像尺寸用于渲染，应该与 source 图像尺寸一致（推荐）
- ⚠️ 如果尺寸不一致，需要 resize 或 crop

#### 规则5: 设备一致性

**问题**: 所有张量是否必须在同一设备上？

**规则**:
- ✅ 所有张量必须在同一设备上（通常是 GPU）
- ✅ `scene_id` 可以是 CPU 上的 Tensor，但索引时需要处理
- ⚠️ 确保 `means[scene_id]` 等数据在正确的设备上

### 3.3 数据提供的顺序和规律

#### 训练时的数据提供顺序（基于 EVolSplat 实际实现）

**每次训练迭代**（参考 `SplatDatamanager.next_train()` 第347-394行）:

```
Step N:
1. 随机选择场景 scene_id
   - 使用均匀随机采样: scene_id = torch.randint(num_scenes, size=(1,))
   - 每个场景被选中的概率相等
   
2. 从场景中随机选择 target 图像
   - 计算场景的图像范围: [scene_id * num_images_per_scene, (scene_id+1) * num_images_per_scene)
   - 在场景内随机选择: image_index = random.randint(start_index, end_index) % num_images_per_scene
   
3. 选择最近的 V 张 source 图像
   - 调用 get_source_images_from_current_imageid()
   - 使用 get_nearest_pose_ids() 计算距离
   - 距离计算方法: angular_dist_method='dist'（欧氏距离）
   - 排除 target 图像本身（tar_id=image_id）
   - 选择距离最近的 num_source_image 张图像（默认3张）
   
4. 组装 batch
   - 确保所有张量维度正确
   - 确保内参转换为 4x4
   - 确保深度图存在
   
5. 传递给 model.get_outputs()
```

#### 场景采样的规律（EVolSplat 实际实现）

**实现**（`evolsplat_datamanger.py` 第351行）:
```python
# 均匀随机采样
scene_id = torch.randint(self.dataparser.config.num_scenes, size=(1,))
```

**说明**:
- ✅ **均匀随机采样**：每个场景被选中的概率相等
- ✅ 使用 `torch.randint`，范围是 `[0, num_scenes)`
- ✅ 每次迭代独立采样，不依赖之前的步骤

#### Target 图像选择的规律（EVolSplat 实际实现）

**实现**（`evolsplat_datamanger.py` 第187-197行的 `sample_camId_from_multiscene` 方法）:
```python
start_index = scene_id * self.num_images_per_scene
end_index = start_index + self.num_images_per_scene - 1
image_index = random.randint(start_index, end_index) % self.num_images_per_scene
```

**说明**:
- ✅ **在场景内随机选择**：使用 Python 的 `random.randint`
- ✅ 选择范围限制在当前场景的图像索引范围内
- ✅ 每次迭代独立随机选择，不依赖之前的步骤

#### Source 图像选择的规律（EVolSplat 实际实现）

**实现**（`datamanagers/utils.py` 第112-140行的 `get_source_images_from_current_imageid` 函数）:
```python
# 1. 获取 target 的 pose
target_pose = all_pose[image_id]

# 2. 调用 get_nearest_pose_ids 选择最近的图像
nearest_pose_ids = get_nearest_pose_ids(
    target_pose.detach().cpu().numpy(),
    all_pose.detach().cpu().numpy(),
    num_select=num_select,
    tar_id=image_id,  # 排除 target 本身
    angular_dist_method='dist',  # 使用欧氏距离
)

# 3. 排序并选择最近的 num_select 张
nearest_pose_ids = np.array(sorted(nearest_pose_ids))
selected_ids = sorted_ids[:num_select]
```

**距离计算方法**（`datamanagers/utils.py` 第78-81行）:
```python
# angular_dist_method='dist' 时
tar_cam_locs = tar_pose[:, :3, 3]  # 提取相机位置
ref_cam_locs = ref_poses[:, :3, 3]
dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)  # 欧氏距离
```

**说明**:
- ✅ **基于欧氏距离的最近邻选择**：计算 target 相机位置与所有候选 source 相机位置的欧氏距离
- ✅ **排除 target 本身**：通过设置 `tar_id=image_id`，将 target 的距离设为 1e3，确保不会被选中
- ✅ **选择最近的 num_source_image 张**：默认3张，可配置
- ✅ **排序后选择**：按距离从近到远排序，选择前 num_source_image 张

**关键点**:
- 距离计算基于相机位置（`camera_to_world[:3, 3]`），不是旋转
- 使用欧氏距离（`np.linalg.norm`），不是角度距离
- 选择范围限制在当前场景内（通过 `sample_camId_from_multiscene` 传入的场景图像）

### 3.4 必须遵循的规则

#### 规则1: 场景一致性

**规则**: Source 和 target 图像必须来自同一场景。

**检查**:
```python
# 在 get_batch 中
assert all(
    self._get_scene_id_from_image_idx(idx) == scene_id
    for idx in source_indices + [target_image_idx]
), "Source and target must be from the same scene"
```

#### 规则2: 索引有效性

**规则**: 所有图像索引必须在有效范围内。

**检查**:
```python
# 在 get_batch 中
num_images = scene_dataset.pixel_source.num_imgs
assert 0 <= target_image_idx < num_images, \
    f"target_image_idx {target_image_idx} out of range [0, {num_images})"
assert all(0 <= idx < num_images for idx in source_indices), \
    "Some source indices out of range"
```

#### 规则3: 维度一致性

**规则**: 所有 source 图像的维度必须一致。

**检查**:
```python
# 在 get_batch 中
source_shapes = [img.shape for img in source_images]
assert len(set(source_shapes)) == 1, \
    f"Source images have inconsistent shapes: {source_shapes}"
```

#### 规则4: 设备一致性

**规则**: 所有张量必须在同一设备上。

**检查**:
```python
# 在 get_batch 中
all_tensors = [
    batch['source']['image'],
    batch['source']['extrinsics'],
    batch['source']['intrinsics'],
    batch['target']['image'],
    batch['target']['extrinsics'],
    batch['target']['intrinsics'],
]
if batch['source']['depth'] is not None:
    all_tensors.append(batch['source']['depth'])

devices = [t.device for t in all_tensors]
assert len(set(devices)) == 1, \
    f"Tensors on different devices: {devices}"
```

#### 规则5: 数据类型一致性

**规则**: 相同类型的张量应该有相同的数据类型。

**检查**:
```python
# 在 get_batch 中
assert batch['source']['image'].dtype == torch.float32, \
    "Source images should be float32"
assert batch['target']['image'].dtype == torch.float32, \
    "Target image should be float32"
assert batch['source']['extrinsics'].dtype == torch.float32, \
    "Extrinsics should be float32"
```

---

## 4. 总结

### 4.1 Batch 格式总结

```python
batch = {
    'scene_id': Tensor[1] 或 int,  # 场景ID
    
    'source': {
        'image': Tensor[V, H, W, 3],      # V 张图像（默认3）
        'extrinsics': Tensor[V, 4, 4],   # 外参
        'intrinsics': Tensor[V, 4, 4],    # 内参（必须是4x4）
        'depth': Tensor[V, H, W],        # 深度图（必需）
    },
    
    'target': {
        'image': Tensor[1, H, W, 3],     # 1 张图像
        'extrinsics': Tensor[1, 4, 4],    # 外参
        'intrinsics': Tensor[1, 4, 4],    # 内参（必须是4x4）
    }
}
```

### 4.2 数据提供规则总结（基于 EVolSplat 实际实现）

1. **场景采样**: 均匀随机采样
   - 实现: `scene_id = torch.randint(num_scenes, size=(1,))`
   - 每个场景被选中的概率相等

2. **Target 选择**: 在场景内随机选择
   - 实现: `image_index = random.randint(start_index, end_index) % num_images_per_scene`
   - 选择范围限制在当前场景的图像索引范围内

3. **Source 选择**: 基于相机位置欧氏距离的最近邻选择
   - 实现: 使用 `get_nearest_pose_ids()` 函数
   - 距离计算: `dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)`
   - 排除 target 本身: `dists[tar_id] = 1e3`
   - 选择最近的 `num_source_image` 张（默认3张）

4. **数量**: 1 张 target + V 张 source（默认 V=3）
5. **一致性**: 同一场景、同一设备、维度一致

### 4.3 反直觉检查清单

- [ ] `scene_id` 是整数或 `Tensor[1]`，且在有效范围内
- [ ] 内参已转换为 4x4 格式
- [ ] 深度图存在且尺寸匹配
- [ ] Source 和 target 来自同一场景
- [ ] 所有张量在同一设备上
- [ ] 所有图像尺寸一致
- [ ] Source 图像数量 >= 1（至少1张）
- [ ] Target 图像数量 = 1（固定）

---

## 5. 实现建议

### 5.1 EvolSplatDataset 的实现要点

1. **内参转换**: 必须实现 `_convert_intrinsic_to_4x4()` 方法
2. **深度图处理**: 必须确保深度图存在，如果不存在则创建占位符
3. **维度检查**: 在返回 batch 前检查所有维度
4. **设备管理**: 确保所有张量在正确的设备上
5. **错误处理**: 处理边界情况（图像数量不足等）

### 5.2 测试建议

1. **单元测试**: 测试 `get_batch()` 返回的 batch 格式
2. **维度测试**: 验证所有张量的维度正确
3. **设备测试**: 验证所有张量在同一设备上
4. **集成测试**: 测试与 `model.get_outputs()` 的集成

