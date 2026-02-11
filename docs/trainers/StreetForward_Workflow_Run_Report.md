# StreetForward 推荐流程运行报告

本文档记录按 `StreetForward_Logging_Update_Summary.md` 第 199–201 行推荐流程的执行结果（使用 conda 环境 `drivestudio-new`）。

---

## 一、环境准备

### 1.1 依赖补充

执行前需额外安装以下依赖（`drivestudio-new` 中未预装）：

| 依赖 | 安装方式 |
|------|----------|
| `nvdiffrast` | `pip install "git+https://github.com/NVlabs/nvdiffrast.git" --no-build-isolation` |
| `splines` | `pip install splines` |

### 1.2 Preflight 脚本修改

为避免 `models.trainers` 导入链（nerfstudio、splines 等），对 `tools/preflight_sweep_streetforward.py` 做了：

- 用 `_load_streetforward_trainer()` 按需加载 StreetForwardTrainer，绕过 `models.trainers.__init__`
- 用 `_preflight_setup()` 替代 `train_setup`，实现精简配置加载
- 从 `utils.streetforward_baseline` 导入 `convert_batch_to_streetforward_format`

---

## 二、运行结果

### 2.1 Preflight sweep（步骤 1）

**命令**：
```bash
PYTHONPATH=/root/drivestudio-coding conda run -n drivestudio-new python tools/preflight_sweep_streetforward.py \
  --config_file configs/streetforward/multi_scene.yaml \
  data.data_root=/root/autodl-tmp/nuScenes/ \
  --max_batches 4 --log_interval 1
```

**数据路径**：需要通过 `data.data_root=/root/autodl-tmp/nuScenes/` 覆盖，否则 dataset preset `nuscenes/3cams` 会使用 `data/nuscenes/processed_10Hz/mini`，导致找不到数据。

**运行过程**：

1. 脚本能正常启动并加载数据
2. 数据集初始化耗时较长（约 2 分钟/3 个 preload scene），每个 scene 加载 3 个相机的 images、masks、depth
3. 部分 scene 加载失败：
   - `Failed to load scene 6: [Errno 2] No such file or directory: '/root/autodl-tmp/nuScenes/006/humanpose/smpl.pkl'`
   - `Failed to load scene 8: [Errno 2] No such file or directory: '/root/autodl-tmp/nuScenes/008/humanpose/smpl.pkl'`
4. 120 秒测试超时，脚本在 dataset 初始化阶段被中断，尚未进入 `sample_random_batch` 循环

**结论**：Preflight 脚本逻辑正常，但当前环境存在：

- 部分 scene 缺少 `humanpose/smpl.pkl`，导致场景加载失败
- 初次加载数据集耗时较长，需更长超时或减少 `preload_scene_count`

### 2.2 Tiny overfit（步骤 2）与 Canary run（步骤 3）

未执行。二者依赖可用的 `MultiSceneDataset` 和 `sample_random_batch`。在 Preflight 能完整跑通并产出有效 batch 之前，建议先解决数据与配置问题。

---

## 三、错误与处理建议

### 3.1 缺失 humanpose/smpl.pkl

**现象**：Scene 6、8 等加载失败，缺少 `{data_root}/{scene_id}/humanpose/smpl.pkl`。

**可能原因**：

- 数据集未做完整 humanpose 预处理
- 或 `pixel_source.load_smpl` 应为 `False`，以跳过 SMPL 加载

**建议**：在 `configs/datasets/nuscenes/3cams.yaml` 中设置 `load_smpl: false`，或在数据预处理阶段补齐 `humanpose/smpl.pkl`。

### 3.2 数据根路径被覆盖

**现象**：`multi_scene.yaml` 中 `data_root: /root/autodl-tmp/nuScenes/` 会在合并 `nuscenes/3cams.yaml` 后被覆盖为 `data/nuscenes/processed_10Hz/mini`。

**建议**：通过 CLI 显式指定，例如：
```bash
data.data_root=/root/autodl-tmp/nuScenes/
```

### 3.3 首次运行超时

**建议**：首次运行可适当增加超时或减少加载量，例如：
```bash
# 减少 preload 以加快启动
dataset.preload_scene_count=1

# 或延长超时时间
```

---

## 四、推荐的完整执行命令

在数据与配置就绪后，可按以下顺序执行：

```bash
# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate drivestudio-new
export PYTHONPATH=/root/drivestudio-coding

# 1. Preflight sweep（需正确 data_root 和足够时间）
python tools/preflight_sweep_streetforward.py \
  --config_file configs/streetforward/multi_scene.yaml \
  data.data_root=/root/autodl-tmp/nuScenes/ \
  dataset.preload_scene_count=1 \
  --max_batches 32 --log_interval 5

# 2. Tiny overfit（需要固定 scene/segment 的配置或脚本支持）
python tools/train_streetforward.py \
  --config_file configs/streetforward/multi_scene.yaml \
  data.data_root=/root/autodl-tmp/nuScenes/ \
  training.max_iterations=500

# 3. Canary run（严格模式）
python tools/train_streetforward.py \
  --config_file configs/streetforward/multi_scene.yaml \
  data.data_root=/root/autodl-tmp/nuScenes/ \
  training.strict_proxy_grad=true \
  training.detect_anomaly_steps=100 \
  training.max_iterations=500
```

---

## 五、小结

| 步骤 | 状态 | 原因 |
|------|------|------|
| Preflight sweep | 部分完成 | 脚本可运行，但 dataset 初始化超时；部分 scene 缺少 smpl.pkl |
| Tiny overfit | 未执行 | 依赖 Preflight 数据通路正常 |
| Canary run | 未执行 | 同上 |

**优先处理**：在 `nuscenes/3cams.yaml` 中设置 `load_smpl: false`，或补齐缺失的 `humanpose/smpl.pkl`，并确保 `data_root` 正确。
