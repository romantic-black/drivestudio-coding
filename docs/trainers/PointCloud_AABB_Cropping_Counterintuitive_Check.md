# PointCloud 是否按 AABB 截取：反直觉检查报告（seg0 系）

## 结论（先说结果）

- **点云 background 的截取并不是“按 batch['aabb']”做的**；它是由点云生成器的 **`crop_aabb`** 控制的（当前实现中硬裁剪始终开启，不再有 `use_bbx` 开关）。
- 在 StreetForward / MultiSceneDataset 路径中，background 点云会先对齐到 **segment 第一帧坐标系 (seg0)**，然后才用 `crop_aabb` 进行裁剪。
- 若你观察到“pointcloud 没按 AABB 截取”，最常见的反直觉原因是：
  1. **AABB 来源不一致**：你在用一个 AABB 检查点云，但生成器裁剪用的是另一个 AABB（已建议统一到 `dataset.segment_aabb`）；
  2. **坐标轴语义误解**：文档中的 x/y/z 语义与你当前数据源（如 nuScenes）原生坐标轴可能不同，导致你用“语义坐标”去解释“数值裁剪”时感觉不对。

---

## 1. 关键事实：crop_aabb 与 batch['aabb'] 的角色不同

### 1.1 生成器实际使用的裁剪逻辑

- `datasets/pointcloud_generators/base.py::crop_pointcloud` 对点进行逐维筛选（严格不等式）：
  - \((x,y,z)\) 需满足 `crop_min < p < crop_max` 才保留。
- 当前实现中，background 点云会在 seg0 对齐后 **始终执行** `crop_aabb` 硬裁剪。

### 1.2 batch['aabb'] 是“同系场景框”，不是生成器的裁剪指令

- `batch['aabb']` 的设计目的：给 Trainer/渲染侧一个**与当前 batch 同坐标系（seg0）**的一致场景框。
- 点云生成器裁剪范围取决于自身的 `crop_aabb`，而不是 batch['aabb']。

因此：即使 `batch['aabb']` 看起来是你想要的范围，也不必然意味着点云一定按它裁剪 —— 除非你确保它与生成器 `crop_aabb` 一致（推荐统一到 `dataset.segment_aabb`）。

---

## 2. seg0 系对齐：你看到的“超界”可能来自坐标轴理解差异

`MultiSceneDataset_Usage.md` 对 seg0 系 x/y/z 的语义是：

- x：左右（左负右正）
- y：上下（上负下正）
- z：后前（后负前正）

反直觉点在于：很多自动驾驶数据集（例如 nuScenes）在原始标定中常用的坐标轴语义与上面不完全一致（例如 z 为 up、x 为 forward 等）。当前代码中的裁剪是**对齐 seg0 后的数值 xyz**直接做 min/max 过滤，并不会“按语义自动换轴”。因此如果你的 seg0 定义来自某个 pose（lidar 或 camera），而该 pose 的轴定义与文档语义不同，你会观察到：

- 点云在你以为的“上下方向 y”并没有落在 \([-20,4.8]\) 内；
- 但在代码实际使用的 xyz 上，它可能是正确裁剪的。

这类问题需要先确认：**当前数据源的 `*_to_world` / `instances_pose` 的坐标轴约定**与文档是否一致。

---

## 3. 反直觉检查清单（按优先级）

### 3.1 `batch['aabb']` 与生成器 `crop_aabb` 是否一致（最常见）

建议/现状目标：统一到 `dataset.segment_aabb` 作为唯一来源，并让 `batch['aabb']` 与生成器 `crop_aabb` 绑定一致，避免“点云按 B 裁了，但你在用 A 检查”的错觉。

### 3.2 `batch['aabb']` 与生成器 `crop_aabb` 是否一致（很容易忽略）

当前（统一后）的约定应为：

- 生成器裁剪使用 `dataset.segment_aabb`；
- `batch['aabb']` 也固定为 `dataset.segment_aabb`。

在“未完全统一”的过渡期，如果 AABB 存在多处来源且不一致，会出现反直觉现象：

- batch 里的 aabb 看起来是 A；
- 点云实际裁剪用的是 B；
- 你对照 batch['aabb'] 去看点云，就会觉得“没按 aabb 截取”。

建议：统一到 `dataset.segment_aabb` 作为唯一来源，避免多处配置。

### 3.3 “超界点”并不一定意味着 crop_aabb 没生效

即使 crop_aabb 生效了，也可能看到一些点“不在 input_aabb 内”，这是正常的：

- `crop_aabb`：决定**移除框外点**（硬裁剪）
- `input_aabb`：用于**inside/outside 分流**，然后分别做不同强度的滤波，最后 inside+outside 仍会合并回 background（所以 outside 点仍会保留）

这会造成一个常见误判：你把 input_aabb 当作“最终保留范围”，于是觉得“没按 aabb 截取”。实际上裁剪边界是 crop_aabb。

---

## 4. 建议的验证方式（不改代码、快速自证）

1. 在生成 batch 后，打印/可视化 background 点云的 xyz min/max（seg0 系）；
2. 对比生成器生效的 `crop_aabb`（以及 batch['aabb']，如果你期望它一致）：
   - 若点云 xyz min/max 明显超出 crop_aabb 很多，说明裁剪没生效 → 优先检查 AABB 来源与传递链路；
   - 若点云落在 crop_aabb 内但不在 input_aabb 内，属于正常（见 3.3）。

---

## 5. 本次检查的直接代码依据（摘要）

- `RGBPointCloudGenerator.crop_pointcloud`: 按 xyz 逐维严格不等式裁剪（当前生成器路径中会无条件调用）。
- `LiDARRGBPointCloudGenerator.generate_pointcloud` / `MonocularRGBPointCloudGenerator.generate_pointcloud`：
  - 都在 `world_to_seg0` 变换后调用 crop。
- `MultiSceneDataset.get_segment_batch`：
  - 提供 `batch['aabb']`（seg0 系）用于 Trainer/渲染侧一致性；其与生成器裁剪范围是否一致取决于配置。


