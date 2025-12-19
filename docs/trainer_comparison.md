# MultiTrainer 与 EVolSplat 训练方式对比

## 概述

本文档对比 `models/trainers/scene_graph.py` 中的 `MultiTrainer` 与 `third_party/EVolSplat` 中 EVolSplat 的训练方式，包括训练流程、关键数据和关键组件。

---

## 1. 训练架构对比

| 对比项             | MultiTrainer                                                                   | EVolSplat                                    |
| ------------------ | ------------------------------------------------------------------------------ | -------------------------------------------- |
| **继承关系** | `BasicTrainer` → `MultiTrainer`                                           | `Model` (nerfstudio) → `EvolSplatModel` |
| **训练框架** | 自定义训练循环 (`tools/train.py`)                                            | nerfstudio 训练框架                          |
| **模型组织** | 多模型字典：`{Background, RigidNodes, DeformableNodes, SMPLNodes, Sky, ...}` | 单一模型：`EvolSplatModel`                 |
| **场景表示** | 分类别表示（背景、刚体、可变形、人体）                                         | 统一表示（单一点云集合）                     |

---

## 2. 训练流程对比

### 2.1 训练循环结构

| 阶段               | MultiTrainer                                                | EVolSplat                                                                    |
| ------------------ | ----------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **初始化**   | `init_gaussians_from_dataset()` → 分类别初始化点云       | `populate_modules()` / `init_mono_points_from_dataset()` → 初始化种子点 |
| **训练迭代** | `tools/train.py` 自定义循环                               | nerfstudio `Trainer.train()` 标准循环                                      |
| **前向传播** | `trainer.forward(image_infos, cam_infos)`                 | `model(ray_bundle, batch)` → `get_outputs()`                            |
| **损失计算** | `trainer.compute_losses(outputs, image_infos, cam_infos)` | `model.get_loss_dict(outputs, batch)`                                      |
| **反向传播** | `trainer.backward(loss_dict)`                             | `loss.backward()`                                                          |
| **参数更新** | `trainer.optimizer_step()`                                | `optimizer.step()`                                                         |

### 2.2 前向传播流程

#### MultiTrainer 前向传播流程

```
forward(image_infos, cam_infos)
    ↓
[时间戳处理]
├─ 计算当前帧: cur_frame = argmin(|normalized_timestamps - normed_time|)
├─ 设置各模型的 cur_frame
└─ 设置 in_test_set 标志
    ↓
[相机处理]
process_camera()
├─ 相机位姿优化 (CamPosePerturb, CamPose)
└─ 构建 dataclass_camera
    ↓
[收集高斯参数]
collect_gaussians()
├─ 遍历所有高斯模型类 (Background, RigidNodes, DeformableNodes, SMPLNodes)
├─ 调用 model.get_gaussians(cam) 获取每类的高斯参数
│   ├─ Background: VanillaGaussians.get_gaussians()
│   ├─ RigidNodes: RigidNodes.get_gaussians() (含实例变换)
│   ├─ DeformableNodes: DeformableNodes.get_gaussians() (含变形)
│   └─ SMPLNodes: SMPLNodes.get_gaussians() (含SMPL变形)
├─ 合并所有类别的高斯参数
└─ 构建 dataclass_gs
    ↓
[渲染]
render_gaussians()
├─ Gaussian Splatting 光栅化
└─ 输出: rgb, depth, opacity
    ↓
[后处理]
├─ 渲染天空: sky_model(image_infos)
├─ 混合: rgb = rgb_gaussians + rgb_sky * (1 - opacity)
└─ 仿射变换: affine_transformation(rgb, image_infos)
    ↓
返回 outputs
```

#### EVolSplat 前向传播流程

```
get_outputs(camera, batch)
    ↓
[加载种子点数据]
├─ means = self.means[scene_id]
├─ scales = self.scales[scene_id]
├─ offset = self.offset[scene_id] (上次保存的offset)
└─ anchors_feat = self.anchor_feats[scene_id]
    ↓
[3D特征提取] (如果 freeze_volume=False)
├─ 构建稀疏张量: construct_sparse_tensor(means, anchors_feat)
├─ 稀疏卷积: feat_3d = sparse_conv(sparse_feat)
└─ 转换为密集体积: dense_volume = sparse_to_dense_volume()
    ↓
[2D特征采样]
projector.sample_within_window()
├─ 投影点云到源图像
├─ 采样局部窗口特征: sampled_feat [N, N_views, (2R+1)^2, 3]
└─ 计算可见性掩码: valid_mask
    ↓
[过滤点云]
├─ projection_mask = valid_mask.sum(dim=1) > threshold
└─ 裁剪: means_crop, sampled_color, scales_crop, last_offset
    ↓
[特征融合]
├─ 三线性插值3D特征: feat_3d = interpolate_features(means_crop + last_offset)
├─ 计算视角方向: ob_view, ob_dist
└─ 拼接特征: input_feature = [sampled_color, ob_dist, ob_view]
    ↓
[MLP预测高斯参数]
├─ 颜色 (SH): sh = gaussion_decoder(input_feature)
├─ 尺度+旋转: scales_crop, quats_crop = mlp_conv(feat_3d, ob_dist, ob_view)
├─ 不透明度: opacities_crop = mlp_opacity(feat_3d, ob_dist, ob_view)
└─ 位置偏移: offset_crop = mlp_offset(feat_3d)
    ↓
[更新位置]
├─ means_crop += offset_crop
└─ 保存offset: self.offset[scene_id] = offset_crop.detach().cpu()
    ↓
[渲染]
rasterization()
├─ 输入: means_crop, quats_crop, scales_crop, opacities_crop, colors_crop
└─ 输出: rgb, depth, accumulation
    ↓
返回 outputs
```

---

## 3. 关键数据对比

### 3.1 输入数据

| 数据类型           | MultiTrainer                                                    | EVolSplat                                                                                                                                             |
| ------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **图像信息** | `image_infos`: {pixels, normed_time, img_idx, sky_masks, ...} | `batch['source']['image']`: [N_views, H, W, 3]`<br>batch['target']['image']`: [H, W, 3]                                                           |
| **相机信息** | `camera_infos`: {camera_to_world, intrinsics, height, width}  | `batch['source']['extrinsics']`: [N_views, 4, 4]`<br>batch['source']['intrinsics']`: [N_views, 4, 4]`<br>batch['target']['intrinsics']`: [4, 4] |
| **深度信息** | 可选:`image_infos.get('depth')`                               | `batch['source']['depth']`: [N_views, H, W]                                                                                                         |
| **场景ID**   | 通过 `img_idx` 隐式确定                                       | `batch['scene_id']`: int                                                                                                                            |

### 3.2 中间数据

| 数据类型           | MultiTrainer                                                                                                                                                                                                                                   | EVolSplat                                                                                                                                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **高斯参数** | 分类别存储：`<br>`- Background: `_means, _scales, _quats, _opacities, _features_dc, _features_rest<br>`- RigidNodes: 同上 + `instances_quats, instances_trans<br>`- DeformableNodes: 同上 + 变形参数`<br>`- SMPLNodes: 同上 + SMPL参数 | 统一存储：`<br>`- `self.means[scene_id]`: [N, 3]`<br>`- `self.scales[scene_id]`: [N, 3] (log)`<br>`- `self.offset[scene_id]`: [N, 3]`<br>`- `self.anchor_feats[scene_id]`: [N, 3] (RGB) |
| **3D特征**   | 无（直接使用高斯参数）                                                                                                                                                                                                                         | `dense_volume`: [B, C, H, W, D]`<br>feat_3d`: [N, sparse_conv_outdim]                                                                                                                               |
| **2D特征**   | 无                                                                                                                                                                                                                                             | `sampled_feat`: [N, N_views, (2R+1)^2, 3]`<br>valid_mask`: [N, N_views, (2R+1)^2]                                                                                                                   |
| **融合特征** | 无                                                                                                                                                                                                                                             | `input_feature`: [N, feature_dim_in]`<br>scale_input_feat`: [N, sparse_conv_outdim + 4]                                                                                                             |

### 3.3 输出数据

| 数据类型           | MultiTrainer                                 | EVolSplat                              |
| ------------------ | -------------------------------------------- | -------------------------------------- |
| **RGB**      | `outputs["rgb"]`: [H, W, 3]                | `outputs["rgb"]`: [H, W, 3]          |
| **深度**     | `outputs["depth"]`: [H, W, 1]              | `outputs["depth"]`: [H, W, 1]        |
| **不透明度** | `outputs["opacity"]`: [H, W, 1]            | `outputs["accumulation"]`: [H, W, 1] |
| **分类渲染** | `outputs[class_name+"_rgb"]`: 每类独立渲染 | 无                                     |
| **动态掩码** | `outputs["Dynamic_rgb"]`: 动态物体渲染     | 无                                     |

---

## 4. 关键组件对比

### 4.1 模型组件

| 组件               | MultiTrainer                                                                                                                                           | EVolSplat                                                                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **高斯模型** | 多模型：`<br>`- `VanillaGaussians` (Background)`<br>`- `RigidNodes` (刚体)`<br>`- `DeformableNodes` (可变形)`<br>`- `SMPLNodes` (人体) | 单一模型：`<br>`- `EvolSplatModel`                                                                                                                   |
| **特征提取** | 无（直接使用高斯参数）                                                                                                                                 | `SparseCostRegNet`: 3D稀疏卷积`<br>Projector`: 2D特征采样                                                                                            |
| **参数预测** | 无（直接优化参数）                                                                                                                                     | MLP解码器：`<br>`- `gaussion_decoder`: 颜色/SH`<br>`- `mlp_conv`: 尺度+旋转`<br>`- `mlp_opacity`: 不透明度`<br>`- `mlp_offset`: 位置偏移 |
| **渲染器**   | `render_gaussians()`: Gaussian Splatting 光栅化                                                                                                      | `rasterization()`: Gaussian Splatting 光栅化                                                                                                           |
| **辅助模型** | `Sky` 模型（天空渲染）`<br>CamPosePerturb` / `CamPose`（相机优化）                                                                               | `bg_field`（背景场，可选）                                                                                                                             |

### 4.2 数据处理组件

| 组件               | MultiTrainer                                                          | EVolSplat                       |
| ------------------ | --------------------------------------------------------------------- | ------------------------------- |
| **点云收集** | `collect_gaussians()`: 从各模型收集并合并                           | 无（使用预初始化的种子点）      |
| **相机处理** | `process_camera()`: 相机位姿优化                                    | 直接使用传入的 `Cameras` 对象 |
| **时间处理** | `normalized_timestamps`: 时间戳归一化`<br>`各模型支持时间相关变形 | 无（静态场景）                  |
| **特征融合** | 无                                                                    | 三线性插值 + 特征拼接           |

### 4.3 优化组件

| 组件                   | MultiTrainer                                                                                                                                                   | EVolSplat                                                                                                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **优化参数**     | 直接优化高斯参数：`<br>`- `Parameter(means)<br>`- `Parameter(scales)<br>`- `Parameter(quats)<br>`- `Parameter(opacity)<br>`- `Parameter(features)` | 优化MLP网络参数：`<br>`- `sparse_conv` 权重`<br>`- `gaussion_decoder` 权重`<br>`- `mlp_conv` 权重`<br>`- `mlp_opacity` 权重`<br>`- `mlp_offset` 权重 |
| **优化器**       | `torch.optim.Adam`: 统一优化器，支持分组学习率                                                                                                               | nerfstudio 优化器：支持分组学习率                                                                                                                                        |
| **学习率调度**   | 自定义调度器：支持warmup、cosine衰减                                                                                                                           | nerfstudio 学习率调度                                                                                                                                                    |
| **梯度裁剪**     | 可选：`clip_grad_norm_`                                                                                                                                      | 可选：`clip_grad_norm_`                                                                                                                                                |
| **参数更新机制** | 直接梯度更新                                                                                                                                                   | offset通过detach保存，间接更新                                                                                                                                           |

---

## 5. 损失函数对比

| 损失类型             | MultiTrainer                                                                                                               | EVolSplat                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **RGB损失**    | L1损失：`\|gt_rgb - pred_rgb\|.mean()`                                                                                     | L1损失：`\|gt_img - pred_img\|.mean()`                                                     |
| **SSIM损失**   | `1 - SSIM(gt_rgb, pred_rgb)`                                                                                             | `1 - SSIM(gt_img, pred_img)`                                                             |
| **掩码损失**   | 天空不透明度损失：`BCE(pred_opacity, gt_mask)`                                                                           | 无                                                                                         |
| **深度损失**   | 可选：深度损失（L1或L2）                                                                                                   | 无                                                                                         |
| **熵损失**     | 无                                                                                                                         | 每10步计算：`-accumulation * log(accumulation) - (1-accumulation) * log(1-accumulation)` |
| **正则化损失** | 各模型可定义：`<br>`- RigidNodes: 实例位姿正则化`<br>`- DeformableNodes: 变形正则化`<br>`- SMPLNodes: SMPL参数正则化 | 无                                                                                         |
| **总损失**     | `rgb_loss + ssim_loss + mask_loss + depth_loss + reg_loss`                                                               | `(1-ssim_lambda) * L1 + ssim_lambda * SSIM + entropy_loss`                               |

---

## 6. 训练流程详细对比

### 6.1 完整训练循环

#### MultiTrainer 训练循环

```
for step in range(max_iterations):
    # 1. 准备训练
    trainer.set_train()
    trainer.preprocess_per_train_step(step)
    trainer.optimizer_zero_grad()
  
    # 2. 获取数据
    image_infos, cam_infos = dataset.train_image_set.next(downscale)
  
    # 3. 前向传播
    outputs = trainer(image_infos, cam_infos)
    trainer.update_visibility_filter()
  
    # 4. 计算损失
    loss_dict = trainer.compute_losses(outputs, image_infos, cam_infos)
  
    # 5. 反向传播
    trainer.backward(loss_dict)
  
    # 6. 后处理
    trainer.postprocess_per_train_step(step)
    # - 更新可见性过滤
    # - 高斯点分裂/复制/剪枝
    # - 更新学习率
  
    # 7. 评估和日志
    if step % eval_interval == 0:
        metric_dict = trainer.compute_metrics(outputs, image_infos)
        # 记录指标
```

#### EVolSplat 训练循环

```
for step in range(start_step, max_iterations):
    # 1. 获取批次数据
    ray_bundle, batch = datamanager.next_train(step)
  
    # 2. 前向传播
    model_outputs = model(ray_bundle, batch)  # get_outputs()
  
    # 3. 计算指标
    metrics_dict = model.get_metrics_dict(model_outputs, batch)
  
    # 4. 计算损失
    loss_dict = model.get_loss_dict(model_outputs, batch, metrics_dict)
    loss = sum(loss_dict.values())
  
    # 5. 反向传播
    loss.backward()
  
    # 6. 更新参数
    optimizer.step()
    optimizer.zero_grad()
  
    # 7. 评估和日志
    if step % eval_interval == 0:
        eval_iteration(step)
```

### 6.2 关键差异点

| 差异点                 | MultiTrainer                                | EVolSplat                                 |
| ---------------------- | ------------------------------------------- | ----------------------------------------- |
| **参数更新时机** | 每个训练步骤后立即更新                      | 每个训练步骤后立即更新                    |
| **参数更新方式** | 直接更新高斯参数                            | 更新MLP网络参数，offset通过detach间接更新 |
| **点云管理**     | 动态管理：分裂、复制、剪枝                  | 静态：种子点固定，仅offset更新            |
| **时间处理**     | 支持时间戳，各模型可时间相关变形            | 无时间维度                                |
| **多场景支持**   | 单场景训练                                  | 支持多场景（通过scene_id）                |
| **相机优化**     | 支持相机位姿优化（CamPosePerturb, CamPose） | 支持相机优化（nerfstudio框架）            |
