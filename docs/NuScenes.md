# Preparing NuScenes Dataset

Before downloading or using the NuScenes dataset, please follow these important steps:

1. Visit the [official NuScenes website](https://www.nuscenes.org/).
2. Register for an account if you haven't already done so.
3. Carefully read and agree to the [NuScenes terms of use](https://www.nuscenes.org/terms-of-use).

Ensure you have completed these steps before proceeding with the dataset download and use.

## 1. Download the Raw Data

Download the raw data from the [official NuScenes website](https://www.nuscenes.org/download). Then, create directories for NuScenes data and optionally create a symbolic link if you have the data elsewhere.

```shell
mkdir -p ./data/nuscenes
ln -s $PATH_TO_NUSCENES ./data/nuscenes/raw # ['v1.0-mini', 'v1.0-trainval', 'v1.0-test'] lies in it
```

We'll use the **v1.0-mini split** in our examples. The process is similar for other splits.

## 2. Install the Development Toolkit

```shell
pip install nuscenes-devkit
```

## 3. Process Raw Data

To process the 10 scenes in NuScenes **v1.0-mini split**, you can run:

```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/nuscenes/raw \
    --target_dir data/nuscenes/processed \
    --dataset nuscenes \
    --split v1.0-mini \
    --start_idx 0 \
    --num_scenes 10 \
    --interpolate_N 4 \
    --workers 32 \
    --process_keys images lidar calib dynamic_masks objects
```

The extracted data will be stored in the `data/nuscenes/processed_10Hz` directory.

`interpolate_N`: Increases frame rate by interpolating between keyframes.

* NuScenes provides synchronized keyframes at `2Hz`. Our script allows interpolation to increase up to `10Hz`.
* `interpolate_N = 4`: Interpolates 4 frames between original synchronized keyframes.
* Result: `10Hz` frame rate `((4 + 1) * 2 Hz)`
* Note: We recommend using `interpolate_N = 4`. While `interpolate_N = 5 (12 Hz)` is possible, it may lead to frame drop issues. Although the camera captures at `12 Hz`, occasional frame misses during recording can cause data gaps at higher interpolation rates.

## 3.1. NuScenes 数据同步机制与插值说明

### 为什么需要插值？

NuScenes 数据集在原始采集时，不同传感器具有不同的采集频率。虽然相机和LiDAR都以较高频率采集数据，但**标注（Label）只在关键帧（Keyframes）上提供，关键帧的频率仅为 2Hz**。对于需要更高时间分辨率的应用（如动态场景重建、时序建模等），2Hz 的帧率过于稀疏。因此，我们通过插值在关键帧之间生成中间帧，将有效帧率提升至 10Hz，从而获得更密集的时间采样。

### 数据同步机制

#### 原始采集频率

在数据采集阶段，NuScenes 的各个传感器并非完全同步触发，而是按照各自的频率独立工作：

| 传感器类型 | 原始采集频率 | 说明 |
|-----------|------------|------|
| **相机 (Camera)** | 12 Hz | 6个相机依次触发，每个相机约 83.3ms 间隔 |
| **LiDAR** | 20 Hz | 约 50ms 间隔 |
| **标注 (Labels)** | 2 Hz | 仅在关键帧提供，约 500ms 间隔 |

**标注类型说明：**
- **3D边界框标注 (sample_annotation)**: 仅在关键帧提供，包含物体的3D边界框、类别、实例ID等信息
- **LiDAR语义分割 (lidarseg)**: 仅在关键帧提供，对关键帧的LiDAR点云进行语义标注（32个类别）
- **Panoptic分割**: 仅在关键帧提供，结合语义分割和实例分割

#### 关键帧同步机制

虽然原始采集时传感器不同步，但 NuScenes 在**关键帧级别**确保了所有传感器数据的时间对齐：

| 数据类型 | 关键帧同步状态 | 同步方式 |
|---------|--------------|---------|
| **相机图像** | ✅ 同步 | 选择最接近关键帧时间戳的相机图像 |
| **LiDAR点云** | ✅ 同步 | 选择最接近关键帧时间戳的LiDAR扫描 |
| **相机外参** | ✅ 同步 | 使用关键帧时刻的ego pose计算 |
| **相机内参** | ✅ 同步 | 内参固定，与关键帧时刻无关 |
| **3D边界框标注** | ✅ 同步 | 仅在关键帧提供，包含物体的3D边界框、类别、实例ID等 |
| **LiDAR语义分割 (lidarseg)** | ✅ 同步 | 仅在关键帧提供，对关键帧的LiDAR点云进行语义标注 |
| **Panoptic分割** | ✅ 同步 | 仅在关键帧提供，结合语义分割和实例分割 |

**关键帧同步原理：**
- 每个关键帧代表一个特定的时间点（timestamp）
- 系统会选择最接近该时间点的各传感器数据
- 通过时间对齐处理，确保在关键帧时刻所有传感器数据在时间上一致
- 这种处理方式保证了多传感器数据的时间一致性，适合大多数自动驾驶算法的开发需求

#### 插值后的数据同步

当使用 `interpolate_N > 0` 进行插值时：

| 数据类型 | 插值方式 | 同步状态 |
|---------|---------|---------|
| **相机图像** | 选择最接近插值时间戳的原始图像 | ✅ 同步（基于时间戳匹配） |
| **LiDAR点云** | 选择最接近插值时间戳的原始扫描 | ✅ 同步（基于时间戳匹配） |
| **相机外参** | 使用插值时间戳对应的ego pose计算 | ✅ 同步（基于ego pose插值） |
| **相机内参** | 保持不变（内参固定） | ✅ 同步 |
| **3D边界框标注** | 在关键帧之间进行线性插值 | ⚠️ 估计值（非原始标注） |
| **LiDAR语义分割 (lidarseg)** | 无（仅在关键帧提供） | ❌ 不可用（非关键帧无标注） |
| **Panoptic分割** | 无（仅在关键帧提供） | ❌ 不可用（非关键帧无标注） |

**插值同步机制：**
- 对于相机和LiDAR：通过 `find_closest_img_tokens()` 和 `find_cloest_lidar_tokens()` 函数，为每个插值时间戳选择最接近的原始数据
- 对于相机外参：使用插值时间戳对应的ego pose，结合固定的相机-ego变换矩阵计算
- 对于3D边界框标注：通过 `interpolate_boxes()` 函数，在相邻关键帧之间进行位置、旋转和尺寸的线性插值
- 对于LiDAR语义分割和Panoptic分割：这些标注仅在关键帧提供，插值帧无法获得对应的标注数据

### 数据同步总结表

| 阶段 | 相机 | LiDAR | 外参 | 内参 | 3D边界框 | lidarseg | Panoptic |
|-----|------|-------|------|------|---------|----------|----------|
| **原始采集** | 12 Hz | 20 Hz | 随ego pose变化 | 固定 | 无 | 无 | 无 |
| **关键帧 (2Hz)** | ✅ 同步 | ✅ 同步 | ✅ 同步 | ✅ 同步 | ✅ 同步（原始） | ✅ 同步（原始） | ✅ 同步（原始） |
| **插值后 (10Hz)** | ✅ 同步 | ✅ 同步 | ✅ 同步 | ✅ 同步 | ⚠️ 估计值（插值） | ❌ 不可用 | ❌ 不可用 |

**注意事项：**
- **关键帧（2Hz）**：所有数据都是完全同步的，包括相机、LiDAR、外参、内参以及所有类型的标注（3D边界框、lidarseg、Panoptic分割）都是原始标注
- **插值帧（10Hz）**：
  - 传感器数据（相机、LiDAR、外参、内参）通过时间戳匹配实现同步
  - 3D边界框标注通过插值生成，为估计值（非原始标注）
  - LiDAR语义分割（lidarseg）和Panoptic分割标注在插值帧不可用，因为这些标注仅在关键帧提供
- 推荐使用 `interpolate_N = 4` 以获得 10Hz 的帧率，同时避免因相机帧丢失导致的数据间隙
- 如果需要使用lidarseg或Panoptic分割标注，请使用关键帧（2Hz）数据，或仅在关键帧时刻使用这些标注

## 4. Extract Masks

To generate:

- **sky masks (required)**
- fine dynamic masks (optional)

Follow these steps:

#### Install `SegFormer` (Skip if already installed)

⚠️ SegFormer relies on `mmcv-full=1.2.7`, which relies on `pytorch=1.8` (pytorch<1.9). Hence, a separate conda env is required.

```shell
#-- Set conda env
conda create -n segformer python=3.8
conda activate segformer
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#-- Install mmcv-full
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf
pip install mmcv-full==1.2.7 --no-cache-dir

#-- Clone and install segformer
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
pip install .
```

Download the pretrained model `segformer.b5.1024x1024.city.160k.pth` from the google_drive / one_drive links in https://github.com/NVlabs/SegFormer#evaluation .

Remember the location where you download into, and pass it to the script in the next step with `--checkpoint`.

<details>
<summary>Troubleshooting: SegFormer Checkpoint Download</summary>

If you encounter problems downloading the original SegFormer checkpoint from the official links, you can alternatively download a backup copy using command: `gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj`

</details>

#### Run Mask Extraction Script

```shell
conda activate segformer
segformer_path=/path/to/segformer
split=mini

python datasets/tools/extract_masks.py \
    --data_root data/nuscenes/processed_10Hz/$split \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --start_idx 0 \
    --num_scenes 10 \
    --process_dynamic_mask
```

Replace `/path/to/segformer` with the actual path to your Segformer installation.

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 5. Human Body Pose Processing

#### Prerequisites

To utilize the SMPL-Gaussian to model pedestrians, please first download the SMPL models.

1. Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php)
2. Move `SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `PROJECT_ROOT/smpl_models/SMPL_NEUTRAL.pkl`

SMPL-Nodes (SMPL-Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

#### Option 1: Download Preprocessed Human Pose Data

We have uploaded preprocessed human pose data for **v1.0-mini split** of NuScenes scenes to [Google Drive](https://drive.google.com/drive/folders/187w1rwEZ5i9tb4y-dOJVTnIZAtKPR7_j). You can download and unzip these files without installing any additional environment.

```shell
# https://drive.google.com/file/d/1Z0gJVRtPnjvusQVaW7ghZnwfycZStCZx/view?usp=drive_link
# filename: nuscenes_preprocess_humanpose.zip
cd data
gdown 1Z0gJVRtPnjvusQVaW7ghZnwfycZStCZx

unzip nuscenes_preprocess_humanpose.zip
rm nuscenes_preprocess_humanpose.zip
```

#### Option 2: Run the Extraction Pipeline

To process human body poses for other NuScenes scenes or to run the processing pipeline yourself, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 6. Data Structure

After completing all preprocessing steps, the project files should be organized according to the following structure:

```shell
ProjectPath/data/
  └── nuscenes/
    ├── raw/
    │    └── [original NuScenes structure]
    └── processed_10Hz/
         └── mini/
              ├── 001/
              │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
              │  ├──lidar/              # LiDAR data: {timestep:03d}.bin
              │  ├──lidar_pose/         # Lidar poses: {timestep:03d}.txt
              │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
              │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
              │  ├──sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
              │  ├──dynamic_masks/      # Dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──fine_dynamic_masks/ # (Optional) Fine dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──objects/            # Object information
              │  └──humanpose/          # Preprocessed human body pose: smpl.pkl
              ├── 002/
              └── ...
```
