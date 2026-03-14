# Overfit 1 Batch 数据获取与读取实现方案

## 1. 背景与目标
为了能够快速调试和验证模型（Overfit 1 Batch），我们需要跳过耗时的数据集完整初始化和在线的 batch 构建过程。
本方案计划在 `tools/overfit_one_batch`（可以是独立脚本或包模块）中实现两部分核心功能：
1. **Capture Batch**: 按照配置，从指定的场景和段中提取出一个标准的 Batch，将其物理序列化到磁盘，并记录相关的元数据信息。
2. **Load Batch**: 在训练前提供接口，直接从磁盘极速拉取指定的 Batch，完全绕过 `MultiSceneDataset` 的庞大初始化流程。
3. 在 tools/overfit_one_batch 建立capture脚本，在 .vscode/launch.json中添加脚本命令

---

## 2. 实现方案设计

### 2.1 Capture Batch：批次获取与存盘

**实现流程**：
1. 读取配置文件，实例化 `MultiSceneDataset`，仅针对目标 scene_id 进行初始化（`initialize()`）。
2. 调用 `dataset.get_segment_batch(scene_id, segment_id, include_test=True)` 获取完整 Batch 数据字典。
3. **数据序列化存储**：
   - **核心数据**：Batch 中混合了 `torch.Tensor`（如 image, extrinsics）和 `numpy.ndarray`（如 pointcloud），使用 PyTorch 自带的 `torch.save(batch, batch_path)` 是最为稳定且支持嵌套字典的序列化方案。
   - **元数据记录**：为了使提取的 Batch 具有可读性，抽取易读的 meta 信息（如 scene_id, segment_id, keyframe_info 等），以 JSON 格式独立保存为 `meta.json`。

### 2.2 Load Batch：批次快速读取

**实现流程**：
1. 提供 `load_batch` 接口，使用 `torch.load()` 将存盘的 `.pt` 文件直接读入内存（推荐读至 CPU 内存，后续在主训练流程中再放置到目标 Device）。
2. 在主训练脚本（如 `train_streetforward.py`）中增加逻辑判断：如果传入了具体的 `overfit_batch_path`，则直接读取离线 Batch 并缓存，**彻底跳过** `MultiSceneDataset` 的构造。
3. 将读取的 Raw Batch 送入现有的 `convert_batch_to_streetforward_format(batch, device)` 转换为训练所需的 `View` 对象和张量。

---

## 3. 代码结构参考 (`tools/overfit_one_batch.py`)

建议在 `tools/overfit_one_batch.py` 中提供以下接口并支持直接作为脚本运行（通过命令行参数区分是 `capture` 还是 `train` 或者被 import）：

```python
import os
import json
import torch
from datasets.multi_scene_dataset import MultiSceneDataset

def capture_batch(cfg, scene_id: int, segment_id: int, save_dir: str):
    """提取单个 batch 并持久化到磁盘"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 最小化初始化 Dataset
    dataset = MultiSceneDataset(
        data_cfg=cfg.data,
        train_scene_ids=[scene_id],
        eval_scene_ids=[],
        num_source_keyframes=cfg.dataset.num_source_keyframes,
        num_target_keyframes=cfg.dataset.num_target_keyframes,
        pointcloud_config=cfg.dataset.get("pointcloud", None),
        # 传递其他必须的 cfg
        device=torch.device("cpu"),
        preload_scene_count=1
    )
    dataset.initialize()
    
    # 2. 获取 Batch
    print(f"Capturing batch for scene {scene_id}, segment {segment_id}...")
    batch = dataset.get_segment_batch(scene_id=scene_id, segment_id=segment_id, include_test=True)
    
    # 3. 核心数据持久化
    batch_path = os.path.join(save_dir, f"scene{scene_id}_seg{segment_id}_batch.pt")
    torch.save(batch, batch_path)
    
    # 4. 保存易读的元数据 (Meta Info)
    meta_info = {
        "scene_id": int(batch['scene_id'].item() if torch.is_tensor(batch['scene_id']) else batch['scene_id']),
        "segment_id": batch['segment_id'],
        "keyframe_info": batch.get('keyframe_info', {}),
        "has_pointcloud": 'pointcloud' in batch,
        "has_test_views": 'test' in batch
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta_info, f, indent=4)
        
    print(f"Batch successfully saved to {batch_path}")
    return batch_path

def load_batch(batch_path: str):
    """快速读取存盘的 Batch"""
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Overfit batch not found at {batch_path}")
    batch = torch.load(batch_path, map_location="cpu")
    return batch
```

---

## 4. 推荐的 Config 模版

针对 `MultiSceneDataset` 的设计（参考 `@docs/dataloader/MultiSceneDataset_Usage.md`），在执行 Capture 操作时，应提供一套精简且强确定性的 Config 模版，以确保小段数据不被过滤策略丢弃：

```yaml
# configs/overfit_one_batch_template.yaml

# 基础数据配置 (根据底层具体数据集替换)
data:
  dataset_root: "/path/to/dataset"
  train_scene_ids: [0]  # 指定需要 overfit 的场景
  eval_scene_ids: []
  
# MultiSceneDataset 核心配置
dataset:
  num_source_keyframes: 3
  num_target_keyframes: 6
  segment_overlap_ratio: 0.2
  
  # 重要：调小场景和段的 keyframes 最小要求，防止截取的短片段被内部策略过滤掉
  min_keyframes_per_scene: 2    
  min_keyframes_per_segment: 2  
  
  # Segment AABB（必需，seg0 系）
  segment_aabb:
    - [-20.0, -20.0, -20.0]
    - [20.0, 4.8, 70.0]
  # Segment input AABB（必需，seg0 系）
  segment_input_aabb:
    - [-20.0, -20.0, -20.0]
    - [20.0, 4.8, 120.0]
    
  # 点云生成器配置 (为模型提供初始化与深度一致性等特征)
  pointcloud:
    type: "monocular"
    chosen_cam_ids: [0, 1, 2, 3, 4, 5]
    sparsity: "full"
    filter_sky: true
    depth_consistency: true
    downscale: 2
    # NOTE: crop_aabb/input_aabb 已统一到 dataset.segment_aabb/segment_input_aabb

# Capture 专属参数（在工具脚本中读取）
capture:
  scene_id: 0
  segment_id: 0
  save_dir: "./data/overfit_batches"
```

---

## 5. 

