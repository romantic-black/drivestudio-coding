# StreetForward 评估设计文档

## 概述

本文档设计 StreetForward 方法的评估机制，参考深度学习最佳实践和 OmniRe 的评估方式。StreetForward 是 feed-forward 3DGS 方法，其评估流程与训练流程类似，但使用测试视角进行无监督评估。

---

## 核心概念

### 1. 评估策略

**段内评估（Segment-Level Evaluation）**：
- 与 OmniRe 一致，评估在段（Segment）级别进行
- 每个段独立评估，使用段内的测试帧
- 评估结果可以按段聚合，也可以按场景聚合

**评估时机**：
- **训练时评估**：在训练迭代中，使用段内的 test 视角进行实时评估（不参与梯度计算）
- **独立评估**：训练完成后，对指定场景/段进行完整评估

### 2. 抽帧机制（Test Frame Sampling）

**test_image_stride 参数**：
- 用于在段分割之前抽取测试帧
- 如果 `test_image_stride = 0`：所有帧都用于训练和测试（没有独立的测试集）
- 如果 `test_image_stride > 0`：每隔 `test_image_stride` 帧抽取一帧作为测试帧

**抽帧流程**：
```
场景所有帧
  ↓
根据 test_image_stride 抽帧
  ↓
训练帧集合 + 测试帧集合
  ↓
对训练帧进行关键帧分割
  ↓
对训练帧进行段分割
  ↓
段内包含训练帧和测试帧（段范围内的测试帧）
```

**关键点**：
- 抽帧在关键帧分割和段分割**之前**进行
- 段分割只使用训练帧，但段内的测试帧用于评估
- 段内的测试帧必须在该段的帧范围内（frame_indices）

### 3. 数据流程

#### 训练流程
```
1. 获取段内的 src 视角和初始点云
2. 使用 src 视角和初始点云构建/迭代 3DGS 场景
3. 使用段内的 target 视角计算损失（监督学习）
4. 使用段内的 test 视角进行实时评估（可选，不参与梯度计算）
```

#### 评估流程
```
1. 获取段内的 src 视角和初始点云
2. 使用 src 视角和初始点云构建/迭代 3DGS 场景（与训练相同的流程）
3. 对段内的所有 test 视角进行渲染和评估
4. 计算评估指标（PSNR, SSIM, LPIPS 等）
```

**关键区别**：
- 训练时：使用 target 计算损失，test 用于实时评估
- 评估时：只使用 test 进行渲染和评估，不计算损失

---

## 实现设计

### 1. MultiSceneDataset 扩展

#### 1.1 抽帧处理

在 `_load_scene` 方法中，在关键帧分割之前进行抽帧：

```python
def _load_scene(self, scene_id: int) -> Optional[Dict]:
    """
    加载单个场景的数据。
    
    流程：
    1. 创建 DrivingDataset 实例
    2. 根据 test_image_stride 抽帧（训练帧 vs 测试帧）
    3. 获取场景的轨迹（使用训练帧）
    4. 分割关键帧（使用训练帧）
    5. 分割段（使用训练帧，但记录段内的测试帧）
    6. 返回场景信息
    """
    # 1. 创建场景配置
    scene_cfg = OmegaConf.create(OmegaConf.to_container(self.data_cfg))
    scene_cfg.scene_idx = scene_id
    
    # 2. 创建 DrivingDataset 实例
    scene_dataset = DrivingDataset(scene_cfg)
    
    # 3. 根据 test_image_stride 抽帧
    test_image_stride = self.data_cfg.pixel_source.get("test_image_stride", 0)
    train_frame_indices, test_frame_indices = self._split_train_test_frames(
        num_frames=scene_dataset.num_img_timesteps,
        test_image_stride=test_image_stride,
    )
    
    # 4. 获取场景轨迹（所有帧），然后过滤出训练帧的轨迹
    full_trajectory = self._get_scene_trajectory(scene_dataset)
    # 过滤出训练帧的轨迹（用于关键帧分割）
    trajectory = full_trajectory[train_frame_indices]
    
    # 5. 分割关键帧（使用训练帧）
    keyframe_segments, keyframe_ranges = self._split_keyframes(trajectory)
    
    # 6. 检查场景是否适合训练
    if not self._is_scene_suitable(keyframe_segments):
        return None
    
    # 7. 分割段（使用训练帧，但记录段内的测试帧）
    segments = self._split_segments(
        scene_dataset=scene_dataset,
        keyframe_segments=keyframe_segments,
        keyframe_ranges=keyframe_ranges,
        train_frame_indices=train_frame_indices,
        test_frame_indices=test_frame_indices,
        overlap_ratio=self.segment_overlap_ratio,
    )
    
    return {
        'dataset': scene_dataset,
        'trajectory': trajectory,
        'train_frame_indices': train_frame_indices,
        'test_frame_indices': test_frame_indices,
        'keyframe_segments': keyframe_segments,
        'keyframe_ranges': keyframe_ranges,
        'segments': segments,
        'num_frames': scene_dataset.num_img_timesteps,
        'num_cams': scene_dataset.num_cams,
    }
```

#### 1.2 抽帧方法

```python
def _split_train_test_frames(
    self,
    num_frames: int,
    test_image_stride: int,
) -> Tuple[List[int], List[int]]:
    """
    根据 test_image_stride 抽帧，分离训练帧和测试帧。
    
    Args:
        num_frames: 场景总帧数
        test_image_stride: 测试帧步长（0表示所有帧用于训练和测试）
        
    Returns:
        train_frame_indices: 训练帧索引列表
        test_frame_indices: 测试帧索引列表
    """
    if test_image_stride == 0:
        # 所有帧都用于训练和测试
        train_frame_indices = list(range(num_frames))
        test_frame_indices = list(range(num_frames))
    else:
        # 每隔 test_image_stride 帧抽取一帧作为测试帧
        # 注意：与 driving_dataset.py 保持一致，从 test_image_stride 开始（1-indexed），即 0-indexed 的 test_image_stride
        test_frame_indices = list(range(
            test_image_stride,  # 从第 test_image_stride 帧开始（1-indexed），即 0-indexed 的 test_image_stride
            num_frames,
            test_image_stride,
        ))
        train_frame_indices = [
            i for i in range(num_frames)
            if i not in test_frame_indices
        ]
    
    return train_frame_indices, test_frame_indices
```

#### 1.3 段分割扩展

在 `_split_segments` 方法中，需要记录段内的测试帧：

```python
def _split_segments(
    self,
    scene_dataset: DrivingDataset,
    keyframe_segments: List[List[int]],
    keyframe_ranges: Tensor,
    train_frame_indices: List[int],
    test_frame_indices: List[int],
    overlap_ratio: float,
) -> List[Dict]:
    """
    按照场景 AABB 限制分割段，并记录段内的测试帧。
    
    Args:
        ...
        train_frame_indices: 训练帧索引列表
        test_frame_indices: 测试帧索引列表
    
    Returns:
        segments: List[Dict] - 每个段包含：
            - 'segment_id': int
            - 'keyframe_indices': List[int]
            - 'frame_indices': List[int] - 段内所有训练帧索引
            - 'test_frame_indices': List[int] - 段内所有测试帧索引（段范围内的）
            - 'aabb': Tensor[2, 3]
    """
    # ... 原有的段分割逻辑 ...
    
    # 在创建段时，记录段内的测试帧
    for seg in segments:
        segment_train_frames = seg['frame_indices']
        if len(segment_train_frames) == 0:
            seg['test_frame_indices'] = []
            continue
        
        # 找出段范围内的测试帧
        # 测试帧必须在段的帧范围内（min <= test_frame_idx <= max）
        segment_min_frame = min(segment_train_frames)
        segment_max_frame = max(segment_train_frames)
        segment_test_frames = [
            test_frame_idx
            for test_frame_idx in test_frame_indices
            if segment_min_frame <= test_frame_idx <= segment_max_frame
        ]
        seg['test_frame_indices'] = segment_test_frames
    
    return segments
```

#### 1.4 Batch 格式扩展

`get_segment_batch` 方法需要添加 test 视角：

```python
def get_segment_batch(
    self,
    scene_id: int,
    segment_id: int,
    include_test: bool = True,  # 是否包含测试视角
) -> Dict:
    """
    获取指定场景和段的训练批次。
    
    Args:
        scene_id: 场景ID
        segment_id: 段ID
        include_test: 是否包含测试视角（默认True）
        
    Returns:
        Dict包含：
            - 'scene_id': Tensor[1]
            - 'segment_id': int
            - 'source': {...}  # 源视角
            - 'target': {...}  # 目标视角（用于监督学习）
            - 'test': {...}  # 测试视角（用于评估，可选）
                - 'image': Tensor[num_test_frames * num_cams, H, W, 3]
                - 'extrinsics': Tensor[num_test_frames * num_cams, 4, 4]
                - 'intrinsics': Tensor[num_test_frames * num_cams, 4, 4]
                - 'depth': Tensor[num_test_frames * num_cams, H, W]
                - 'frame_indices': Tensor[num_test_frames * num_cams]
                - 'cam_indices': Tensor[num_test_frames * num_cams]
            - 'pointcloud': Optional[Dict]
    """
    # ... 原有的 source 和 target 加载逻辑 ...
    
    # 加载 test 视角（如果 include_test 且段内有测试帧）
    test_images = []
    test_extrinsics = []
    test_intrinsics = []
    test_depths = []
    test_frame_idxs = []
    test_cam_idxs = []
    
    if include_test:
        scene_data = self._ensure_scene_loaded(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")
        
        segment = scene_data['segments'][segment_id]
        scene_dataset = scene_data['dataset']
        test_frame_indices = segment.get('test_frame_indices', [])
        
        if len(test_frame_indices) > 0:
            for frame_idx in test_frame_indices:
                for cam_idx in range(scene_dataset.num_cams):
                    img_idx = frame_idx * scene_dataset.num_cams + cam_idx
                    image_infos, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
                    
                    test_images.append(image_infos['pixels'])
                    test_extrinsics.append(cam_infos['camera_to_world'])
                    
                    intrinsic_3x3 = cam_infos['intrinsics']
                    intrinsic_4x4 = self._convert_intrinsic_to_4x4(intrinsic_3x3)
                    test_intrinsics.append(intrinsic_4x4)
                    
                    depth = self._get_depth(scene_dataset, frame_idx, cam_idx)
                    if depth is None:
                        H, W = image_infos['pixels'].shape[:2]
                        depth = torch.ones(H, W, dtype=torch.float32, device=self.device) * 10.0
                    test_depths.append(depth)
                    
                    test_frame_idxs.append(frame_idx)
                    test_cam_idxs.append(cam_idx)
    
    batch = {
        'scene_id': torch.tensor([scene_id], dtype=torch.long),
        'segment_id': segment_id,
        'source': {...},
        'target': {...},
        'pointcloud': pointcloud,
    }
    
    # 如果加载了测试视角，添加到批次中
    if include_test and len(test_images) > 0:
        batch['test'] = {
            'image': torch.stack(test_images, dim=0),
            'extrinsics': torch.stack(test_extrinsics, dim=0),
            'intrinsics': torch.stack(test_intrinsics, dim=0),
            'depth': torch.stack(test_depths, dim=0),
            'frame_indices': torch.tensor(test_frame_idxs, dtype=torch.long),
            'cam_indices': torch.tensor(test_cam_idxs, dtype=torch.long),
        }
    
    return batch
```

### 2. StreetForwardTrainer 扩展

#### 2.1 训练时评估

在 `train_iter` 方法中添加可选的实时评估：

```python
def train_iter(
    self,
    batch: Dict,
    apply_update: bool = True,
    update_state: bool = True,
    evaluate_test: bool = False,  # 是否在训练时评估测试视角
) -> Dict:
    """
    训练迭代，支持可选的测试视角评估。
    
    Args:
        batch: 训练批次
        apply_update: 是否应用优化器更新
        update_state: 是否更新 node_state
        evaluate_test: 是否在训练时评估测试视角（默认False）
    """
    # ... 原有的训练逻辑 ...
    
    # 训练完成后，可选地评估测试视角
    test_metrics = None
    if evaluate_test and 'test' in batch:
        with torch.no_grad():
            test_metrics = self._evaluate_test_views(
                node_state=node_state,
                test_batch=batch['test'],
            )
    
    return {
        "total_loss": torch.tensor(total_loss_val, device=self.device),
        "node_state": self.node_states[key],
        "outputs": outputs,
        "test_metrics": test_metrics,  # 可选
    }
```

#### 2.2 评估方法

添加专门的评估方法：

```python
@torch.no_grad()
def evaluate(
    self,
    batch: Dict,
) -> Dict[str, float]:
    """
    评估模型在测试视角上的性能。
    
    Args:
        batch: 评估批次，必须包含 'source', 'test', 'pointcloud'
            - 'source': 源视角（用于构建 3DGS 场景）
            - 'test': 测试视角（用于评估）
            - 'pointcloud': 初始点云
        
    Returns:
        metrics: 评估指标字典
            - 'psnr': float
            - 'ssim': float
            - 'lpips': float
            - 'num_test_views': int
    """
    self.eval()
    
    # 1. 获取或初始化 node_state（使用 pointcloud）
    key, node_state = self._get_or_init_node_state(batch)
    
    # 2. 使用 source 视角和初始点云构建 3DGS 场景
    # （与训练相同的流程，但不更新参数）
    # 注意：需要将 batch['source'] 转换为训练格式（target_views, gt_images）
    render_params = self._build_3dgs_scene(node_state, batch)
    
    # 3. 对测试视角进行渲染
    test_views = batch['test']
    test_images = test_views['image']
    test_extrinsics = test_views['extrinsics']
    test_intrinsics = test_views['intrinsics']
    
    all_psnr = []
    all_ssim = []
    all_lpips = []
    
    for view_idx in range(len(test_images)):
        # 渲染测试视角
        rgb_pred = self._render_view(
            render_params=render_params,
            extrinsic=test_extrinsics[view_idx],
            intrinsic=test_intrinsics[view_idx],
            height=test_images[view_idx].shape[0],
            width=test_images[view_idx].shape[1],
        )
        
        # 计算指标
        rgb_gt = test_images[view_idx].to(self.device)
        psnr = self._compute_psnr(rgb_pred, rgb_gt)
        ssim = self._compute_ssim(rgb_pred, rgb_gt)
        lpips = self._compute_lpips(rgb_pred, rgb_gt)
        
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(lpips)
    
    # 4. 聚合指标
    metrics = {
        'psnr': float(np.mean(all_psnr)),
        'ssim': float(np.mean(all_ssim)),
        'lpips': float(np.mean(all_lpips)),
        'num_test_views': len(test_images),
    }
    
    self.train()
    return metrics

def _build_3dgs_scene(
    self,
    node_state: NodeState,
    batch: Dict,
) -> Dict[str, torch.Tensor]:
    """
    使用 source 视角和初始点云构建 3DGS 场景。
    
    这是训练流程的简化版本，不更新参数。
    
    注意：
    - 需要将 batch['source'] 转换为 train_iter 期望的格式
    - 或者直接复用 train_iter 中的构建逻辑（但不进行梯度计算和参数更新）
    - 建议：创建一个内部方法 `_forward_pass` 供 train_iter 和 evaluate 共享
    """
    # 与 train_iter 中的构建逻辑相同
    # 但不进行梯度计算和参数更新
    # 建议：提取 train_iter 中的前向传播逻辑为独立方法
    # ...
    pass

def _render_view(
    self,
    render_params: Dict[str, torch.Tensor],
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    渲染单个视角。
    """
    viewmat = get_viewmat(extrinsic.unsqueeze(0))
    k_mat = intrinsic[:3, :3].unsqueeze(0)
    
    render, alpha, _ = self.renderer(
        means=render_params["means_r"],
        quats=render_params["quats_r"],
        scales=render_params["scales_r"],
        opacities=render_params["opacities_r"],
        colors=render_params["colors_r"],
        viewmats=viewmat,
        Ks=k_mat,
        width=width,
        height=height,
        tile_size=16,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=self.sh_degree,
        sparse_grad=False,
        absgrad=True,
        rasterize_mode="classic",
    )
    
    return render[:, ..., :3].squeeze(0)

def _compute_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
    """计算 PSNR。"""
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    psnr = -10 * torch.log10(mse)
    return float(psnr.item())

def _compute_ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
    """计算 SSIM（需要实现或使用库）。"""
    # 可以使用 pytorch-msssim 或其他库
    # 这里简化处理
    from pytorch_msssim import ssim
    pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
    gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
    return float(ssim(pred_4d, gt_4d).item())

def _compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
    """计算 LPIPS（需要实现或使用库）。"""
    # 可以使用 lpips 库
    # 这里简化处理
    from lpips import LPIPS
    if not hasattr(self, '_lpips_model'):
        self._lpips_model = LPIPS(net='alex').to(self.device)
    pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
    gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
    return float(self._lpips_model(pred_4d, gt_4d).item())
```

---

## 评估指标

### 1. 基础指标

**PSNR (Peak Signal-to-Noise Ratio)**：
- 衡量图像重建质量
- 单位：dB，越高越好
- 公式：`PSNR = -10 * log10(MSE)`

**SSIM (Structural Similarity Index)**：
- 衡量结构相似性
- 范围：[0, 1]，越高越好
- 考虑亮度、对比度、结构

**LPIPS (Learned Perceptual Image Patch Similarity)**：
- 基于深度学习的感知相似性
- 范围：[0, +∞)，越低越好
- 更符合人类感知

### 2. 扩展指标（可选）

**Masked Metrics**：
- `masked_psnr`: 只计算非天空区域的 PSNR
- `masked_ssim`: 只计算非天空区域的 SSIM
- `dynamic_psnr`: 只计算动态物体区域的 PSNR
- `dynamic_ssim`: 只计算动态物体区域的 SSIM

**Per-Camera Metrics**：
- 每个相机的独立指标
- 用于分析不同视角的性能差异

**Per-Segment Metrics**：
- 每个段的独立指标
- 用于分析不同空间区域的性能

---

## 评估流程

### 1. 训练时评估（可选）

```python
# 在训练循环中
for iteration in range(num_iterations):
    batch = scheduler.next_batch()
    
    # 训练
    result = trainer.train_iter(
        batch=batch,
        evaluate_test=(iteration % eval_every == 0),  # 每隔 eval_every 次迭代评估一次
    )
    
    # 记录评估指标
    if result.get('test_metrics') is not None:
        logger.info(f"Iteration {iteration}: PSNR={result['test_metrics']['psnr']:.4f}")
```

### 2. 独立评估

```python
# 评估指定场景的所有段
def evaluate_scene(
    trainer: StreetForwardTrainer,
    dataset: MultiSceneDataset,
    scene_id: int,
) -> Dict[str, float]:
    """
    评估场景的所有段。
    
    Returns:
        聚合的评估指标
    """
    scene_data = dataset.get_scene(scene_id)
    all_metrics = []
    
    for segment in scene_data['segments']:
        segment_id = segment['segment_id']
        
        # 获取评估批次（只包含 test 视角）
        batch = dataset.get_segment_batch(
            scene_id=scene_id,
            segment_id=segment_id,
            include_test=True,
        )
        
        # 评估
        metrics = trainer.evaluate(batch)
        all_metrics.append(metrics)
    
    # 聚合指标
    aggregated = {
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_metrics]),
        'lpips': np.mean([m['lpips'] for m in all_metrics]),
        'num_segments': len(all_metrics),
    }
    
    return aggregated
```

### 3. 完整评估脚本

```python
def main():
    # 1. 加载配置和模型
    config = OmegaConf.load("configs/streetforward/multi_scene.yaml")
    trainer = StreetForwardTrainer(config)
    trainer.load_checkpoint("checkpoints/checkpoint_final.pth")
    
    # 2. 创建数据集
    dataset = MultiSceneDataset(
        data_cfg=config.data,
        train_scene_ids=config.data.train_scene_ids,
        eval_scene_ids=config.data.eval_scene_ids,
        # ...
    )
    
    # 3. 评估训练场景
    train_metrics = []
    for scene_id in config.data.train_scene_ids:
        metrics = evaluate_scene(trainer, dataset, scene_id)
        train_metrics.append(metrics)
        logger.info(f"Scene {scene_id}: PSNR={metrics['psnr']:.4f}")
    
    # 4. 评估评估场景
    eval_metrics = []
    for scene_id in config.data.eval_scene_ids:
        metrics = evaluate_scene(trainer, dataset, scene_id)
        eval_metrics.append(metrics)
        logger.info(f"Scene {scene_id}: PSNR={metrics['psnr']:.4f}")
    
    # 5. 保存结果
    results = {
        'train': {
            'mean_psnr': np.mean([m['psnr'] for m in train_metrics]),
            'mean_ssim': np.mean([m['ssim'] for m in train_metrics]),
            'mean_lpips': np.mean([m['lpips'] for m in train_metrics]),
            'per_scene': train_metrics,
        },
        'eval': {
            'mean_psnr': np.mean([m['psnr'] for m in eval_metrics]),
            'mean_ssim': np.mean([m['ssim'] for m in eval_metrics]),
            'mean_lpips': np.mean([m['lpips'] for m in eval_metrics]),
            'per_scene': eval_metrics,
        },
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
```

---

## 最佳实践

### 1. 评估频率

**训练时评估**：
- 建议每隔 `eval_every` 次迭代评估一次（例如每 100 次迭代）
- 避免过于频繁的评估，因为评估会消耗计算资源
- 评估时使用 `torch.no_grad()` 禁用梯度计算

**独立评估**：
- 在每个 epoch 结束时进行评估
- 在训练完成后进行完整评估
- 保存评估结果以便后续分析

### 2. 评估数据管理

**测试帧选择**：
- 使用 `test_image_stride` 控制测试帧密度
- 建议 `test_image_stride >= 5`，确保有足够的训练数据
- 如果 `test_image_stride = 0`，所有帧都用于训练和测试（过拟合风险）

**段内测试帧**：
- 确保段内有足够的测试帧（至少 1-2 帧）
- 如果段内没有测试帧，跳过该段的评估

### 3. 指标计算

**批量计算**：
- 对于大量测试视角，使用批量渲染和指标计算
- 避免逐个视角计算，以提高效率

**内存管理**：
- 评估时不需要保存所有中间结果
- 只保存最终的聚合指标

### 4. 结果记录

**日志记录**：
- 记录每个段的评估指标
- 记录每个场景的聚合指标
- 记录整体平均指标

**可视化**：
- 保存渲染结果图像（可选）
- 保存误差图（可选）
- 使用 TensorBoard 或 WandB 记录指标

---

## 与 MultiSceneDataset 的集成

### 1. 配置扩展

在 `MultiSceneDataset` 的配置中添加评估相关参数：

```yaml
data:
  pixel_source:
    test_image_stride: 10  # 每隔10帧抽取一帧作为测试帧

dataset:
  # 评估相关配置
  evaluation:
    eval_every: 100  # 训练时每隔多少次迭代评估一次
    save_rendered_images: False  # 是否保存渲染图像
    compute_error_map: False  # 是否计算误差图
```

### 2. 调度器扩展

`MultiSceneDatasetScheduler` 可以添加评估相关方法：

```python
def evaluate_current_segment(
    self,
    trainer: StreetForwardTrainer,
) -> Dict[str, float]:
    """
    评估当前段的测试视角。
    
    Returns:
        评估指标字典
    """
    batch = self.get_current_batch(include_test=True)
    return trainer.evaluate(batch)
```

---

## 总结

StreetForward 的评估设计遵循以下原则：

1. **段内评估**：与 OmniRe 一致，在段级别进行评估
2. **抽帧机制**：使用 `test_image_stride` 在段分割之前抽帧
3. **数据流程**：评估流程与训练流程类似，但使用测试视角
4. **Batch 格式扩展**：添加 test 视角到批次中
5. **评估指标**：使用 PSNR, SSIM, LPIPS 等标准指标
6. **最佳实践**：遵循深度学习评估的最佳实践

该设计允许在不修改核心训练逻辑的情况下，实现完整的评估功能，同时保持与现有代码库的兼容性。

---

## 反直觉检查清单

### 1. 抽帧逻辑检查

- [x] **test_image_stride 起始索引**：
  - ✅ 已修复：与 `driving_dataset.py` 保持一致，从 `test_image_stride` 开始（1-indexed），即 0-indexed 的 `test_image_stride`
  - ❌ 错误示例：`range(test_image_stride - 1, ...)` 会导致测试帧偏移
  - ✅ 正确：`range(test_image_stride, num_frames, test_image_stride)`

- [ ] **test_image_stride = 0 的处理**：
  - 所有帧都用于训练和测试（没有独立的测试集）
  - 评估时使用所有帧，但训练时仍然使用 target 计算损失

### 2. 轨迹获取检查

- [x] **轨迹过滤**：
  - ✅ 已修复：先获取所有帧的轨迹，然后过滤出训练帧的轨迹
  - ❌ 错误：直接使用训练帧索引获取轨迹（`_get_scene_trajectory` 不支持）
  - ✅ 正确：`full_trajectory[train_frame_indices]` 过滤轨迹

- [ ] **关键帧分割使用训练帧轨迹**：
  - 关键帧分割必须使用训练帧的轨迹，而不是所有帧的轨迹
  - 这确保了关键帧分割的一致性

### 3. 段内测试帧确定检查

- [x] **段内测试帧范围**：
  - ✅ 已修复：只检查测试帧是否在段的帧范围内（`min <= test_frame_idx <= max`）
  - ❌ 错误：`test_frame_idx in segment_train_frames or (min <= test_frame_idx <= max)` 会导致逻辑错误
  - ✅ 正确：`segment_min_frame <= test_frame_idx <= segment_max_frame`

- [ ] **空段处理**：
  - 如果段内没有训练帧，测试帧列表应该为空
  - 需要检查 `len(segment_train_frames) == 0` 的情况

- [ ] **测试帧可能不在训练帧中**：
  - 测试帧和训练帧是互斥的（除了 `test_image_stride = 0` 的情况）
  - 段内的测试帧可能在段的帧范围内，但不在训练帧列表中（这是正常的）

### 4. Batch 格式检查

- [x] **scene_dataset 获取**：
  - ✅ 已修复：从 `scene_data['dataset']` 获取，而不是直接使用 `scene_dataset`
  - ❌ 错误：在 `get_segment_batch` 中直接使用 `scene_dataset`（可能未定义）
  - ✅ 正确：`scene_dataset = scene_data['dataset']`

- [ ] **test 视角可选性**：
  - 如果段内没有测试帧，`batch['test']` 不应该存在
  - 评估代码需要检查 `'test' in batch` 和 `len(batch['test']['image']) > 0`

- [ ] **test 视角格式**：
  - test 视角格式应该与 source/target 一致
  - 包含：`image`, `extrinsics`, `intrinsics`, `depth`, `frame_indices`, `cam_indices`

### 5. 评估方法检查

- [x] **评估需要 source 视角**：
  - ✅ 已修复：评估方法需要 `source` 视角来构建 3DGS 场景
  - ❌ 错误：只使用 `test` 视角进行评估
  - ✅ 正确：使用 `source` 构建场景，使用 `test` 进行评估

- [ ] **评估流程与训练流程的一致性**：
  - 评估时构建 3DGS 场景的流程应该与训练时一致
  - 建议：提取 `_forward_pass` 方法供 train_iter 和 evaluate 共享

- [ ] **评估时不更新参数**：
  - 评估时使用 `torch.no_grad()` 和 `self.eval()`
  - 不进行梯度计算和参数更新

### 6. 训练时评估检查

- [ ] **训练时评估的时机**：
  - 应该在训练迭代完成后进行评估
  - 使用训练后的 `node_state` 进行渲染

- [ ] **训练时评估的性能影响**：
  - 评估会消耗额外的计算资源
  - 建议：只在特定迭代时进行评估（例如每 100 次迭代）

- [ ] **训练时评估不参与梯度计算**：
  - 使用 `torch.no_grad()` 确保评估不影响训练

### 7. 数据流程检查

- [ ] **抽帧在关键帧分割之前**：
  - 抽帧必须在关键帧分割之前进行
  - 关键帧分割只使用训练帧

- [ ] **段分割使用训练帧**：
  - 段分割只使用训练帧，但记录段内的测试帧
  - 段内的测试帧用于评估，不用于训练

- [ ] **点云生成使用训练帧**：
  - 点云生成应该使用训练帧还是所有帧？
  - 建议：使用训练帧生成点云，保持一致性

### 8. 边界情况检查

- [ ] **test_image_stride = 0**：
  - 所有帧都用于训练和测试
  - 评估时使用所有帧，但训练时仍然使用 target 计算损失

- [ ] **段内没有测试帧**：
  - 如果段内没有测试帧，跳过该段的评估
  - 或者使用所有帧进行评估（如果 `test_image_stride = 0`）

- [ ] **段内没有训练帧**：
  - 这种情况不应该发生（段分割时已经过滤）
  - 但如果发生，测试帧列表应该为空

- [ ] **场景没有测试帧**：
  - 如果 `test_image_stride > num_frames`，可能没有测试帧
  - 需要处理这种情况

### 9. 与现有代码的兼容性检查

- [ ] **与 MultiSceneDataset_Design.md 的一致性**：
  - 评估设计应该与 MultiSceneDataset 设计保持一致
  - 特别是段分割和关键帧分割的逻辑

- [ ] **与 StreetForward 训练流程的一致性**：
  - 评估流程应该与训练流程类似
  - 使用相同的 3DGS 构建逻辑

- [ ] **与 OmniRe 评估方式的一致性**：
  - 段内评估方式与 OmniRe 一致
  - 使用相同的评估指标

### 10. 性能优化检查

- [ ] **批量渲染**：
  - 对于大量测试视角，使用批量渲染
  - 避免逐个视角渲染

- [ ] **内存管理**：
  - 评估时不需要保存所有中间结果
  - 只保存最终的聚合指标

- [ ] **评估频率**：
  - 训练时评估不要太频繁
  - 独立评估可以在训练完成后进行

---

## 已知问题和待解决事项

### 1. 轨迹过滤的实现细节

**问题**：`_get_scene_trajectory` 返回所有帧的轨迹，需要过滤出训练帧的轨迹。

**解决方案**：
- 在 `_load_scene` 中，先获取所有帧的轨迹
- 然后使用 `trajectory[train_frame_indices]` 过滤出训练帧的轨迹
- 注意：`train_frame_indices` 是全局帧索引，需要确保索引正确

### 2. 评估方法的前向传播逻辑复用

**问题**：评估方法需要复用训练时的前向传播逻辑，但当前设计中没有明确如何复用。

**建议**：
- 提取 `_forward_pass` 方法，供 `train_iter` 和 `evaluate` 共享
- 或者创建一个内部方法 `_build_3dgs_from_source`，专门用于从 source 构建 3DGS 场景

### 3. 点云生成使用训练帧还是所有帧

**问题**：点云生成应该使用训练帧还是所有帧？

**建议**：
- 使用训练帧生成点云，保持与关键帧分割和段分割的一致性
- 如果 `test_image_stride = 0`，则使用所有帧

### 4. 训练时评估的实现细节

**问题**：训练时评估应该在什么时候进行？使用哪个 `node_state`？

**建议**：
- 在训练迭代完成后进行评估
- 使用训练后的 `node_state`（已经更新）
- 使用 `torch.no_grad()` 确保不影响梯度计算
