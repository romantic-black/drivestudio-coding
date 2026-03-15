# 关键节点坐标系调查（运行期实测，无语义假设）

本文档基于一次 overfit batch 运行的 NDJSON 日志（`coord_survey`），记录在 **nuscenes / scene_id=0 / segment_id=1 / segment_first_frame_idx=84** 上，各关键节点的坐标系 **实测轴方向与原点**。所有向量均为运行期从 4×4 变换矩阵中提取（旋转矩阵的列 = 该系 +x/+y/+z 在参考系中的单位方向），**不预先赋予「前后/上下/左右」等语义**。

参考系约定：
- **World**：数据源提供的全局参考系（此处为 nuScenes 的 world），所有 “*_to_world” 的 “world” 即此系。
- **Seg0**：Segment 第一帧坐标系，由 `segment_first_pose`（此处为 lidar 首帧 pose）定义；batch 内外参、点云、AABB 均在此系下一致。

---

## 1. Lidar（段首帧）— `lidar_to_world`

**来源**：`scene_dataset.lidar_source.lidar_to_worlds[segment_first_frame_idx]`。  
**含义**：Lidar 坐标系到 world 的变换；矩阵的列 = lidar 的 +x、+y、+z 在 world 中的方向。

| 项 | 实测值（world 系） |
|----|---------------------|
| 原点 `origin` | `[-31.29, 0.14, 25.59]` |
| +x 轴方向 `axis_plus_x` | `[0.593, 0.003, 0.805]` |
| +y 轴方向 `axis_plus_y` | `[-0.805, 0.033, 0.592]` |
| +z 轴方向 `axis_plus_z` | `[-0.025, -0.999, 0.022]` |

**结论（仅几何）**：Lidar +z 在 world 中近似 `(0, -1, 0)`；+x 近似 `(0.59, 0, 0.81)`；+y 近似 `(-0.81, 0.03, 0.59)`。

---

## 2. Camera（段首帧、参考相机）— `camera_to_world`

**来源**：`pixel_source.camera_data[ref_cam_id].cam_to_worlds`（或 `camera_to_worlds`）的段首帧。  
**含义**：参考相机坐标系到 world 的变换；矩阵的列 = 相机 +x、+y、+z 在 world 中的方向。

| 项 | 实测值（world 系） |
|----|---------------------|
| 原点 `origin` | `[-31.95, 0.47, 26.06]` |
| +x 轴方向 `axis_plus_x` | `[0.590, -0.003, 0.807]` |
| +y 轴方向 `axis_plus_y` | `[0.014, 0.9999, -0.006]` |
| +z 轴方向 `axis_plus_z` | `[-0.807, 0.015, 0.590]` |

**结论（仅几何）**：相机 +y 在 world 中近似 `(0, 1, 0)`；+x 与 +z 近似在 XZ 平面内，分别约 `(0.59, 0, 0.81)` 与 `(-0.81, 0, 0.59)`。

---

## 3. Segment 第一帧坐标系（Seg0）— `segment_first_pose`（本次选用 lidar）

**来源**：`_get_segment_first_pose()` 的返回值；本次 `segment_first_pose_source == "lidar"`，故与 §1 的 lidar_to_world 相同。  
**含义**：Seg0 到 world 的变换；矩阵的列 = seg0 的 +x、+y、+z 在 world 中的方向。

| 项 | 实测值（world 系） |
|----|---------------------|
| 原点 `origin` | `[-31.29, 0.14, 25.59]` |
| +x 轴方向 `axis_plus_x` | `[0.593, 0.003, 0.805]` |
| +y 轴方向 `axis_plus_y` | `[-0.805, 0.033, 0.592]` |
| +z 轴方向 `axis_plus_z` | `[-0.025, -0.999, 0.022]` |

**结论（仅几何）**：Seg0 与当前 lidar 系一致；seg0 +z 在 world 中近似 `(0, -1, 0)`。

---

## 4. 第一个 Source 相机在 Seg0 下 — `camera_to_seg0`

**来源**：`world_to_seg0 @ camera_to_world` 后取第一个 source 视角的外参。  
**含义**：该相机坐标系到 seg0 的变换；矩阵的列 = 相机 +x、+y、+z 在 seg0 中的方向。

| 项 | 实测值（seg0 系） |
|----|---------------------|
| 原点 `origin` | `[0.43, 7.14, -0.16]` |
| +x 轴方向 `axis_plus_x` | `[0.997, -0.076, -0.005]` |
| +y 轴方向 `axis_plus_y` | `[-0.003, 0.020, -1.000]` |
| +z 轴方向 `axis_plus_z` | `[0.076, 0.997, 0.020]` |

**结论（仅几何）**：在 seg0 下，该相机 +x 近似 seg0 的 (1,0,0)；+y 近似 seg0 的 (0,0,-1)；+z 近似 seg0 的 (0,1,0)。

---

## 5. 与文档/语义的对照说明（可选解读）

以下为**可选**解读，便于与 `MultiSceneDataset_Usage.md` 中的「自动驾驶坐标系」对照，**非运行期测量内容**：

- 若 nuScenes 的 **world** 采用常见约定（例如某一轴为铅垂向上），则可根据上述 world 下的向量判断：例如 lidar/seg0 的 +z 近似 `(0,-1,0)`，即沿 world 的 -Y；相机 +y 近似 `(0,1,0)`，即沿 world 的 +Y。
- 文档中「x=左右、y=上下、z=后前」描述的是**期望的抽象约定**；当前运行期数据显示，**seg0 的轴与 lidar 一致**，其 +z 在 world 中接近 -Y，与「z=后前」的直觉不一致，需以实测为准。

---

## 6. 日志出处

- 上述数值均来自 `.cursor/debug-23d286.log` 中：
  - `hypothesisId: "coord_survey"`、`location: "get_segment_batch:coord_survey"`（lidar / camera / segment_first_pose / seg0_basis_in_world）；
  - `hypothesisId: "coord_survey"`、`location: "get_segment_batch:first_camera_in_seg0"`（first_camera_to_seg0_axes_origin）。
- 复现方式：运行任意会调用 `dataset.get_segment_batch(...)` 的流程（如 overfit_one_batch），即可在相同 log 路径得到新一轮 `coord_survey` 记录；更换 scene/segment 或数据集后，需重新跑一次以更新各节点实测值。
