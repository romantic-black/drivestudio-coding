# 运行 gen_nuscenes_depth_mask.py 脚本说明

## 当前状态

已创建批量运行脚本 `run_gen_depth_mask_batch.py`，可以批量处理所有场景。

## 依赖问题

脚本需要以下依赖：
1. **PyTorch >= 2.1** (当前 drivestudio 环境是 2.0.0)
2. **transformers** 库
3. **rich** 库（已安装）
4. **OneFormer 模型**（会自动从 Hugging Face 下载）

## 解决方案

### 方案 1：升级 PyTorch（推荐，但需要磁盘空间）

```bash
source /root/miniconda3/bin/activate drivestudio
pip install torch>=2.1.0 torchvision --upgrade
pip install transformers
```

### 方案 2：使用兼容的 transformers 版本

```bash
source /root/miniconda3/bin/activate drivestudio
pip install "transformers==4.30.0"  # 兼容 PyTorch 2.0.0
```

## 使用方法

### 单个场景测试

```bash
source /root/miniconda3/bin/activate drivestudio
cd /root/drivestudio-coding

# 生成语义分割和天空mask
python third_party/EVolSplat/preprocess/gen_nuscenes_depth_mask.py \
    --scene_dir /root/autodl-tmp/nuScenes/preprocess/trainval/000 \
    --gen_semantic \
    --gen_sky_mask \
    --semantic_gpu_id 0

# 生成深度图
python third_party/EVolSplat/preprocess/gen_nuscenes_depth_mask.py \
    --scene_dir /root/autodl-tmp/nuScenes/preprocess/trainval/000 \
    --gen_depth \
    --depth_gpu_id 0
```

### 批量处理所有场景

```bash
source /root/miniconda3/bin/activate drivestudio
cd /root/drivestudio-coding

# 生成语义分割和天空mask（所有场景）
python run_gen_depth_mask_batch.py --gen_semantic --gen_sky_mask

# 生成深度图（所有场景）
python run_gen_depth_mask_batch.py --gen_depth

# 生成所有（深度图、语义分割、天空mask）
python run_gen_depth_mask_batch.py --gen_depth --gen_semantic --gen_sky_mask
```

## 环境变量

可以通过环境变量设置 GPU ID：

```bash
export DEPTH_GPU_ID=0      # 深度图生成使用的 GPU
export SEMANTIC_GPU_ID=0   # 语义分割生成使用的 GPU
```

## 输出目录结构

处理后的场景目录结构：

```
/root/autodl-tmp/nuScenes/preprocess/trainval/000/
├── images/
├── lidar/
├── calib/
├── semantic/          # 语义分割结果（如果生成）
│   └── instance/      # 实例分割结果
├── depth/             # 深度图结果（如果生成）
└── sky_masks/         # 天空mask（如果生成）
```

## 注意事项

1. **磁盘空间**：OneFormer 模型约 1.5GB，首次运行会自动下载
2. **处理时间**：每个场景的处理时间取决于图像数量和 GPU 性能
3. **GPU 内存**：确保有足够的 GPU 内存（建议至少 8GB）
4. **批量处理**：批量处理会依次处理所有场景，可能需要较长时间





