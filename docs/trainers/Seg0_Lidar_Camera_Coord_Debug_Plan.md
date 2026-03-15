## Seg0 / Lidar / Camera 坐标系调试方案（轴语义彻底对齐）

本方案的目标是：**给出一套可重复的实验与检查步骤，系统性确认并统一 seg0 / lidar / camera 三者的坐标轴含义**，从而：

- 明确当前实现中各节点实际的轴向语义；
- 找出「文档期望的 seg0 坐标系」与「真实运行时 seg0」的差异来源；
- 为后续修正（如 `segment_first_pose_source` 配置、文档更新）提供可靠事实依据。

下文默认你已经阅读：

- `Seg0_Coord_Clarification_and_Fix_Proposal.md`
- `MultiSceneDataset_Usage.md`
- `datasets/nuscenes/nuscenes_preprocess.py`
- `datasets/nuplan/nuplan_sourceloader.py`

---

### 1. 需要回答的关键问题

围绕坐标系，至少要弄清楚下面几个问题（每个问题都需要有**代码侧可复现的实验**支撑）：

- **Q1：数据源世界系是怎么定义的？**
  - nuScenes：`nuscenes_preprocess.py` 里，camera extrinsics 通过「cam_to_ego → ego_to_world」，lidar pose 通过「lidar_to_ego → ego_to_world」，再写到 `extrinsics/*.txt` 和 `lidar_pose/*.txt`，即 **camera_to_world / lidar_to_world**。
  - NuPlan：`NuPlanCameraData` / `NuPlanLiDARSource` 都是通过 `ego_to_world_start^-1 @ ego_to_world_current` 将世界系「重对齐到首帧 ego」。

- **Q2：seg0（segment 第一帧坐标系）在实现里到底等于谁？**
  - `MultiSceneDataset` 侧：实现在 `datasets/multi_scene_dataset.py`，当前逻辑是「优先 lidar，再 fallback camera」。需要找到 **seg0 的 4×4 变换矩阵**是怎么从 `lidar_source.lidar_to_worlds` / `camera_data.cam_to_worlds` 选出来的。

- **Q3：camera / lidar 与 seg0、与「文档里的自动驾驶坐标系」的关系分别是什么？**
  - 文档的「自动驾驶坐标系」定义在 seg0 系下（`MultiSceneDataset_Usage.md` 中 x=左右、y=上下、z=后前）。
  - 实现里 seg0 默认来自 lidar（nuScenes、NuPlan 都是通过 ego pose 链路），但 **lidar 制造商自己的轴约定** 不一定等于文档约定。

本调试方案的目的，就是通过一系列「打印矩阵 + 画图 + 数值对齐」的实验，把 Q1–Q3 全部用事实说清楚。

---

### 2. 准备：统一几件「真相源」

#### 2.1 nuScenes：预处理输出检查

参考 `datasets/nuscenes/nuscenes_preprocess.py`：

- **camera 外参写法**：
  - `extrinsics_cam_to_ego`：由 `calibrated_sensor.rotation/translation` 得到的 camera→ego。
  - `ego_to_world`：由 `ego_pose.rotation/translation` 得到的 ego→world。
  - 最终写入：`extrinsics_cam_to_world = ego_to_world @ extrinsics_cam_to_ego`，保存到 `extrinsics/{frame_idx}_{cam_id}.txt`。
- **lidar pose 写法**：
  - `lidar_to_ego`：同样由校准得到。
  - `ego_to_world`：与 camera 相同链路。
  - 最终写入：`lidar_to_world = ego_to_world @ lidar_to_ego`，保存到 `lidar_pose/{frame_idx}.txt`。

=> **事实 1**：在 nuScenes 处理中，**camera_to_world 与 lidar_to_world 是在同一「官方 world 系」下表达的**。

#### 2.2 NuPlan：世界系「重对齐」检查

参考 `datasets/nuplan/nuplan_sourceloader.py`：

- `NuPlanCameraData.load_calibrations` 中：
  - `cam_to_ego` 由外参文件得到，然后乘以 `OPENCV2DATASET`（当前是单位矩阵，说明「数据集坐标系 = OpenCV: x 右, y 下, z 前」）。
  - `ego_to_world_start`：第一帧 ego pose。
  - 每一帧 `ego_to_world = inv(ego_to_world_start) @ ego_to_world_current`，再 `cam_to_world = ego_to_world @ cam_to_ego`。
- `NuPlanLiDARSource.load_calibrations` 中：
  - 直接用相同的 `ego_to_world_start`，对所有帧 `lidar_to_world = inv(ego_to_world_start) @ ego_to_world_current`。

=> **事实 2**：在 NuPlan 中，**camera_to_world 与 lidar_to_world 是在同一「以首帧 ego 为原点」的 world 系下表达的**，并且 camera 假定 OpenCV 坐标。

#### 2.3 MultiSceneDataset：seg0 与外参/点云对齐

`MultiSceneDataset_Usage.md` 中明确：

- `batch['aabb']`、pointcloud、`target/source/test.extrinsics` 均在 **segment 第一帧坐标系（seg0）** 下。
- seg0 的构造实现藏在 `datasets/multi_scene_dataset.py` 以及 `ScenePixelSource` / `SceneLidarSource` 中：
  - 当前行为：**seg0 默认来自 lidar 的第一个有效帧 pose**，若没有 lidar 再用参考 camera。

=> **事实 3**：需要通过代码查找「seg0 = ?」，并在运行时打印对应 4×4 矩阵。

---

### 2.4 运行时插桩与假设（Session 日志）

在 `datasets/multi_scene_dataset.py` 中已加入 NDJSON 调试日志，写入 **`.cursor/debug-cebadb.log`**（session 为 cebadb）。每条日志包含 `hypothesisId`，对应下列假设：

| 假设 | 含义 | 日志数据 |
|------|------|----------|
| **H1** | 当 seg0 来源为 lidar 时，seg0 应与同帧 `lidar_to_world` 一致 | `seg0_vs_lidar_max_diff`（两矩阵最大元素差，≈0 则一致） |
| **H2** | 当 seg0 来源为 camera 时，seg0 应与同帧 `camera_to_world` 一致 | `seg0_vs_camera_max_diff` |
| **H3** | 同帧 camera 与 lidar 在 world 下的相对变换应合理（标定稳定） | `T_cam_to_lidar`（inv(cam_to_world) @ lidar_to_world 的轴/原点） |
| **H4** | seg0 轴向在 world 下的表示，可与文档约定对比 | `seg0_basis_in_world`、`first_camera_to_seg0_axes_origin` |
| **H5** | 首帧 frame_idx 有效，且 pose 来源（lidar/camera）与 has_lidar/has_camera 一致 | `first_frame_idx`、`pose_source`、`has_lidar`、`has_camera` |

复现步骤见下文「复现步骤」；跑完后根据 log 判定各假设 CONFIRMED/REJECTED/INCONCLUSIVE。

---

### 3. 调试 Step 1：定位并打印 seg0 / camera / lidar 的 4×4 矩阵

**目标**：在同一段、同一帧下，拿到以下矩阵，并能在 Python/NB 里随时打印和做运算：

- `T_lidar_to_world`：来自各数据集的 `lidar_to_worlds[frame_idx]` 或 `lidar_pose/*.txt`。
- `T_cam_to_world`：来自 camera 的 `cam_to_worlds[frame_idx]` 或 `extrinsics/*.txt`。
- `T_seg0_to_world`：MultiSceneDataset 里定义的 seg0 pose。

**建议操作路径**：

1. 在 `datasets/multi_scene_dataset.py` 中搜索：
   - 关键字：`segment_first`, `seg0`, `lidar_to_worlds`, `cam_to_worlds`。
   - 找到「segment 第一帧 pose」是如何被选取/缓存的。
2. 在 `MultiSceneDataset.get_segment_batch` 或构造 segment 的代码里，加一段**临时 debug 日志**（不要带默认值）：

   - 打印：
     - 选中的 `frame_idx`（segment 第一帧）；
     - 对应的 `lidar_source.lidar_to_worlds[frame_idx]`（若存在）；
     - 某个参考相机 `camera_data.cam_to_worlds[frame_idx]`；
     - seg0 自己的 4×4（如果 seg0 = lidar，就打印同一矩阵确认）。
   - 或者在 Notebook（如 `MultiSceneDataset_Demo.ipynb`）里调用内部 API 拿到这些矩阵，统一放入 `torch.Tensor` 做后续分析。

> 建议：把这一小段封装成一个调试 helper（例如 `debug_dump_seg0_pose(scene_id, segment_id, frame_idx)`），方便跨 notebook / script 复用。

---

### 4. 调试 Step 2：数值上对齐 camera 与 lidar

**目标**：确认在「各自的 world 系」下，camera / lidar 是否已经严格对齐（仅差一个固定变换），从而排除「world 本身乱」的可能。

#### 4.1 nuScenes：同帧 camera / lidar 相对变换

选择一个场景 `scene_idx` 和 keyframe `k`，读取预处理输出：

- `T_cam_to_world` ← `extrinsics/{k}_{cam_id}.txt`
- `T_lidar_to_world` ← `lidar_pose/{k}.txt`

计算：

```python
T_cam_lidar = np.linalg.inv(T_cam_to_world) @ T_lidar_to_world   # lidar 在 camera 系下的变换
T_lidar_cam = np.linalg.inv(T_lidar_to_world) @ T_cam_to_world   # camera 在 lidar 系下的变换
```

检查：

- `T_cam_lidar[:3, 3]` 的大小、符号是否符合 nuScenes 官方标定文档（大致为「lidar 在 camera 前上某个位置」）；
- `T_cam_lidar[:3, :3]` 是否正交、det≈1；
- 多选几个 `scene_idx` / `k`，验证 **相对变换是否时间上稳定**（只要车没换传感器，相机–激光雷达标定应为常量）。

若这里已经不对（例如不同时间的 `T_cam_lidar` 差别很大），优先排查 **预处理链路**，而不是 seg0。

#### 4.2 NuPlan：同帧 camera / lidar 相对变换

类似地，在 NuPlan 处理输出中：

- `T_cam_to_world` ← `NuPlanCameraData.cam_to_worlds[frame_idx]`；
- `T_lidar_to_world` ← `NuPlanLiDARSource.lidar_to_worlds[frame_idx]`。

同样计算 `T_cam_lidar` / `T_lidar_cam`，验证：

- time 序列上是否稳定；
- `OPENCV2DATASET` 现在为 I，说明**当前实现假定「数据集相机坐标系 = OpenCV 坐标系」**；如果将来要改这里，也应在本调试脚本中一起验证。

---

### 5. 调试 Step 3：seg0 与 camera/lidar 的关系

**目标**：精确量化「seg0 = lidar」或「seg0 = camera」时的差别，以及它们与文档中「自动驾驶坐标系」的偏差。

#### 5.1 计算 seg0 相对 camera / lidar 的变换

有了 Step 1 中的矩阵：

- `T_seg0_to_world`
- `T_cam_to_world`
- `T_lidar_to_world`

可以计算：

```python
T_cam_to_seg0   = np.linalg.inv(T_seg0_to_world) @ T_cam_to_world
T_lidar_to_seg0 = np.linalg.inv(T_seg0_to_world) @ T_lidar_to_world
```

然后做两件事：

1. 查看 `T_cam_to_seg0[:3, :3]`、`T_lidar_to_seg0[:3, :3]` 的列向量（即 camera / lidar 的 x/y/z 轴在 seg0 系下的坐标）。
2. 计算每个轴与「文档期望的 seg0 轴向」之间的夹角：

   - 文档期望：seg0 里 \(x=左右, y=上下, z=后前\)；
   - 定义一个**「理想 seg0 轴向基」**：

     ```python
     e_x = np.array([1, 0, 0])
     e_y = np.array([0, 1, 0])
     e_z = np.array([0, 0, 1])
     ```

   - 对于实际的 seg0（当前来自 lidar），你可以在 world 系下直接查看它的列向量；对比它们与 \(e_x,e_y,e_z\) 的夹角；再用同样方式查看「如果 seg0 = camera 时」会是什么样子。

> 这一步可以直接写在 Notebook 里：打印每个轴与「理想轴」的角度（单位：度），形成一个小表格。

#### 5.2 seg0 = lidar 与 seg0 = camera 的 A/B 对比

为了定量比较「seg0 来源」的影响，建议设计一个 A/B 实验（只改 seg0 来源，其他数学不变）：

- **A 版（当前实现）**：
  - seg0 = 段首帧 lidar pose；
  - 所有相机外参与点云都转换到 seg0 系；
  - 绘制一些典型可视化（如相机 frustum、车辆轨迹、点云 AABB）。

- **B 版（实验版）**：
  - seg0 = 段首帧某个参考 camera pose；
  - 使用同一段、同一批次，重新构造一次 batch；
  - 再次绘制同样的可视化。

从两版对比中重点观察：

- 在 seg0 系下：
  - \(+y\) 是否更明显地「朝上」；
  - \(+z\) 是否更接近「前进方向」；
  - AABB 的数值范围与可视化是否更符合「直觉上的车辆周围包围盒」。

---

### 6. 调试 Step 4：图像+点云联合可视化（最终 sanity check）

前面的步骤以数值/矩阵为主，这一步用**直观可视化**做最终 sanity check。

#### 6.1 在 camera 图像上画世界/seg0 轴

在一个简单的 Jupyter Notebook（可以基于 `MultiSceneDataset_Demo.ipynb` 拓展）中，增加以下功能：

- 选定一个 `scene_id, segment_id, frame_idx, cam_id`；
- 从 batch 中取出：
  - `image`；
  - `T_cam_to_seg0`（由 Step 3.1 得到）；
  - `T_seg0_to_world`（可选，用于对比世界系）。
- 在 seg0 系下定义三条单位轴：

  ```python
  origins_seg0 = np.zeros((3, 3))        # 全在原点
  axes_seg0    = np.eye(3) * L           # L 为可视化长度
  ```

- 把这三条轴先变换到 camera 系，再用内参投影到像素平面，在图像上用三种颜色画出来（例如：x 红、y 绿、z 蓝）。

如果 camera 外参与 seg0/世界系一致、投影链路正确，那么：

- 对一帧道路场景来说：
  - \(+y\) 画出来应大致指向「天空方向」；
  - \(+z\) 画出来应大致指向「前方道路」；
  - \(+x\) 指向车辆右侧。

#### 6.2 在 camera 图像上叠加 lidar 点云

同一 Notebook 中：

- 从 batch 或底层 loader 取出某一帧的 lidar 点云（在 seg0 或 world 系下）；
- 通过「seg0/world → camera」变换，将点云投影到该 camera 图像上；
- 观察：
  - 静态场景是否对齐（地面、车体轮廓）；
  - 动态对象（如前方车辆）是否落在合理位置。

若轴向含义错误（例如 y / z 颠倒或符号反），在这一步很容易暴露出来。

---

### 7. 调试 Step 5：把结论写回文档与配置

前面几步完成后，你应该能得到这样一张「事实表」：

- 每个数据集（nuScenes / NuPlan）：
  - world 系是如何定义的（是否重对齐首帧 ego）；
  - camera / lidar 在 world 系下的轴向；
- 在当前实现中：
  - seg0 实际等于谁（lidar 或 camera）；
  - seg0 轴向与文档中的「自动驾驶坐标系」夹角大致是多少；
  - 如果改为 seg0= camera，轴向是否更接近文档期望。

**下一步建议**：

- **文档层面**（对应提案中的方案 B）：
  - 在 `MultiSceneDataset_Usage.md` 与 `Coord_System_Survey_Key_Nodes.md` 中，用「本调试实验的结果截图 + 数值」更新说明：
    - 当前默认 seg0 来源；
    - 当 seg0 = lidar 时，是否保证「x=左右、y=上下、z=后前」；
    - 对不同数据集（nuScenes/NuPlan），各自 world / lidar / camera 轴向的已知差异。
- **配置层面**（对应提案中的方案 A/C）：
  - 在 `dataset` 配置中实现并验证 `segment_first_pose_source: "lidar" | "camera"` 或 `"auto"`；
  - 在数据集 preset（如 `configs/datasets/nuscenes.yaml`）中为每个数据集指定合适默认值；
  - 用本方案中的可视化 notebook 作为 **回归用例**：修改 seg0 来源后，快速确认轴向与可视化仍然符合预期。

---

### 8. 小结

- **这篇文档给出的是「调试 playbook」，而不是立即的行为修改**：它强调先通过统一的数值+可视化实验，搞清楚 camera / lidar / seg0 三者之间以及与世界系、文档坐标系之间的关系。
- 按照这里的 Step 1–5 执行后，你应当能：
  - 得到 nuScenes / NuPlan 各自清晰的坐标链路；
  - 精确定量「seg0 = lidar」与「seg0 = camera」时的轴偏差；
  - 用图像+点云 overlay 做最终 sanity check。
- 基于这些事实，再去实现 `segment_first_pose_source`、调整 preset 默认值、更新使用文档，就不会再出现「文档写一套，实际运行另一套，但谁都说不清」的情况。

