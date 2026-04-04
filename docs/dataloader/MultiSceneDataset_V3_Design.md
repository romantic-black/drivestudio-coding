# MultiSceneDatasetV3 设计文档

## 文档定位与参考

本文描述 **`MultiSceneDatasetV3`** 的具体实现方案：在保留现有 [`MultiSceneDataset`](../../datasets/multi_scene_dataset.py) 的场景管理、seg0 坐标变换、点云缓存、动态信息与 preload 骨架的前提下，将数据层升级为 **以 image-ref 为主原语的 batch 装配器**，为 `TrainSchedulerV4` 的 **single-src-image + multi-target-images** 协议提供原生支持，并为后续 preload、overlap 评分与 keyframe 窗口策略预留扩展点。

**应先阅读**：

- [MultiSceneDataset_Usage.md](./MultiSceneDataset_Usage.md) — 数据层次、source/target 定义、坐标系与 batch 字段约定。
- [`datasets/multi_scene_dataset.py`](../../datasets/multi_scene_dataset.py) — 基类实现（`segment_aabb`、`segment_first_pose`、`_segment_pointcloud_cache`、`get_segment_batch` 等）。
- [`datasets/multi_scene_dataset_v2.py`](../../datasets/multi_scene_dataset_v2.py) — 当前 scheduler 兼容层：`get_segment_batch_from_frames()` 通过临时替换 `_select_source_and_target_keyframes` / `_select_frame_from_keyframe` 驱动 `get_segment_batch()`，V3 将用显式 **image-ref → batch** 路径替代该做法。

**配置、日志与测试的衔接**：

- **配置**：V3 作为新 dataset 类接入时，沿用现有训练入口的 YAML / `dataset` 段结构（如 `data_root`、`segment_aabb`、`train_scene_ids` 等，见 Usage 文档）；仅在为 V3 特有行为（如 `enforce_target0_equals_source` 默认值、显式 test 策略）增加字段时在配置中 **显式声明**，避免静默默认。
- **日志**：继承并扩展 V2/V3 scheduler 与数据集侧已使用的可观测字段（如 `segment_first_frame_idx`、`source_frame_idx` 等）；V3 在 batch 中增加 `request_meta` / `index_meta`，使 scheduler 的 block/segment 事件与 **实际落地的 `(frame_idx, cam_id)`** 一一对应；新增字段时与 [`docs/trainers/StreetForward_Scheduler_V4_Design.md`](../trainers/StreetForward_Scheduler_V4_Design.md)（若存在）或 TrainSchedulerV4 实现保持命名一致。
- **测试**：在 `tests/` 下新增面向 V3 的单元测试（建议 `tests/test_multi_scene_dataset_v3.py`），覆盖本文第 15 节清单；与 conda 环境 `drivestudio-new`、`PYTHONPATH` 项目根目录的既有约定一致。

---

## 1. 设计目标

`MultiSceneDatasetV3` 的目标是：在不破坏现有 `MultiSceneDataset` 的场景管理、seg0 坐标变换、点云缓存、动态信息构建和 preload 骨架的前提下，将数据层正式升级为 **面向 image-ref 的 batch 组装器**，为 `TrainSchedulerV4` 的 **single-src-image + multi-target-images** 协议提供原生支持，并为未来的 preload、overlap 评分、keyframe 窗口策略预留稳定扩展点。现有基础实现中，scene queue / scene cache / `_segment_pointcloud_cache` / `_ensure_training_queue_ready()` / `_preload_scenes()` 已经具备了很好的场景生命周期管理能力；V3 不应重写这些，而应在其上增加索引层与显式 batch request 层。

V3 解决的核心问题有两个。第一，V2 的 `get_segment_batch_from_frames()` 仍是“frame-level 外观层”，内部通过 monkey-patch 旧的 keyframe/frame 随机接口来驱动 `get_segment_batch()`，这不够稳定，也不利于以后做更细粒度的调度和缓存。第二，现有 `get_segment_batch()` 的 source/target 语义仍然是“按 frame 选中后展开所有 camera”，这和 V4 所需的“单张 src 图像、显式 target 图像列表”并不一致。

---

## 2. 非目标

V3 的第一版 **不负责** 复杂 overlap 计算，不负责训练时间尺度调度，也不负责新 scheduler 的随机策略本身。这些能力只需要在 V3 中留接口，不应与 batch assembly 强耦合。现有 `TrainSchedulerV3` 已经说明 scheduler 的职责是“决定 sampling/time boundaries 并发出 events”，而不是自己做 batch 结构拼装；V3 应与这种职责分离保持一致。

---

## 3. 核心原则

### 3.1 image-ref 是主原语

V3 的主原语必须从 `frame_idx` 升级为：

```python
ImageRef = Tuple[int, int]  # (frame_idx, cam_id)
```

原因很直接：V4 的调度对象是单张图像，而不是整帧多相机 bundle。V2 当前只能传 `source_frame_idx` 和 `target_frame_indices`，然后默认在 batch 组装时把每个 frame 展开为所有 camera，这与 V4 的 image-level 调度语义不一致。

### 3.2 seg0 坐标系语义不变

V3 必须完整继承现有 seg0 约定：segment 坐标系严格来自该 segment 首帧、camera-0 的 pose；若 camera-0 不可用则直接报错；所有 source/target/test extrinsics 都通过 `world_to_seg0 @ cam_to_world` 转到 seg0 系；`batch["aabb"]` 继续固定等于 dataset 级 `segment_aabb`。这是现有实现中最重要、也最不应改动的语义。

### 3.3 scene/pointcloud 生命周期沿用现有实现

V3 不重写 scene preload，也不重写 segment 级点云缓存。现有 `MultiSceneDataset` 已保证 `_segment_pointcloud_cache` 以 `(scene_id, segment_id)` 为键复用静态点云，并在 scene 卸载时同步清理缓存；V3 直接复用这一能力。

### 3.4 dataset 只负责“验证、映射、组装”

V3 不决定“怎么采样 src/target”，只负责：

- 校验 image refs 是否合法
- 将 image refs 映射为图像/相机数据
- 组装出 seg0 对齐的 batch
- 附带 pointcloud / dynamic_info / test views

调度策略仍由 scheduler 决定，这与现有 V2/V3 scheduler 的职责划分一致。

---

## 4. 类定位与继承关系

建议定义：

```python
class MultiSceneDatasetV3(MultiSceneDataset):
    ...
```

继承 `MultiSceneDataset` 的原因是，后者已经封装了：

- scene 加载与适配 `DrivingDataset`
- train/eval scene split
- keyframe 划分与 segment 划分
- seg0 pose 获取
- depth / sky_mask / egocar_mask / viewdirs 读取
- pointcloud generator 创建与 segment cache
- dynamic_info 构建
- preload 和 scene queue 管理

V3 只需新增 **索引层** 与 **显式 request → batch** 层，而不需要复制上面的底座。

---

## 5. 新增数据结构

### 5.1 ImageRef

```python
ImageRef = Tuple[int, int]  # (frame_idx, cam_id)
```

### 5.2 SegmentIndex

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class SegmentIndex:
    scene_id: int
    segment_id: int
    num_cams: int

    frame_indices: List[int]
    test_frame_indices: List[int]
    keyframe_indices: List[int]

    keyframe_to_frames: Dict[int, List[int]]
    frame_to_keyframe: Dict[int, int]

    segment_first_frame_idx: int
```

`SegmentIndex` 是 V3 最核心的新结构。现有 V2 在 `get_segment_batch_from_frames()` 内部每次都通过遍历 `segment["keyframe_indices"]` 和 `scene_data["keyframe_segments"]` 去反查 `frame -> keyframe`，这说明这层索引关系已经是稳定需求，应该正式缓存。

### 5.3 BatchRequestV3

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class BatchRequestV3:
    scene_id: int
    segment_id: int

    source_image_ref: ImageRef
    target_image_refs: List[ImageRef]

    include_test: bool = False
    test_image_refs: Optional[List[ImageRef]] = None
```

这里明确规定：`target_image_refs[0] == source_image_ref` 是一个由上层协议控制的常见约束，但 V3 应允许通过参数开关决定是否强制检查，而不是把它写死在类定义里。现有 V2 是把 `target_frame_indices[0] == source_frame_idx` 当作 scheduler v2 语义的强约束。

---

## 6. 新增缓存

建议 V3 增加两个缓存：

```python
self._segment_index_cache: Dict[Tuple[int, int], SegmentIndex]
self._pair_score_cache: Dict[Tuple[int, int, ImageRef, ImageRef, str], float]
```

其中，`_segment_index_cache` 是必需的；`_pair_score_cache` 先留空实现，只为以后 overlap/visibility/difficulty 评分预留。现有 `_segment_pointcloud_cache` 则继续保留。

---

## 7. 对外公开接口

### 7.1 `get_segment_index`

```python
def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndex:
    ...
```

功能：

- 保证 scene 已加载
- 从 `scene_data["segments"]` 与 `scene_data["keyframe_segments"]` 构建 `SegmentIndex`
- 将结果缓存到 `_segment_index_cache`

### 7.2 `validate_image_ref`

```python
def validate_image_ref(self, scene_id: int, segment_id: int, image_ref: ImageRef) -> None:
    ...
```

校验内容：

- `frame_idx` 在 segment 的 `frame_indices` 或允许的 test 集内
- `cam_id` 在 `[0, num_cams)` 内
- 对 source/target/test 的合法性采用不同策略

### 7.3 `get_segment_batch_from_image_refs`

```python
def get_segment_batch_from_image_refs(
    self,
    request: BatchRequestV3,
    *,
    enforce_target0_equals_source: bool = True,
) -> Dict[str, Any]:
    ...
```

这是 V3 的正式主接口。

### 7.4 兼容接口

```python
def get_segment_batch_from_frames(...): ...
```

V2 兼容接口保留，但内部改为：

- 把 `source_frame_idx` 转成 `(source_frame_idx, default_cam_id)` 或按旧语义展开
- 再调用 `get_segment_batch_from_image_refs()`

也就是说，V3 要做到 **V2 compatibility on top of V3**，而不是相反。现有 V2 是“patch 后走老路径”，V3 应改成“老接口映射到新路径”。

---

## 8. 批组装主流程

`get_segment_batch_from_image_refs()` 的建议流程如下。

### 8.1 scene / segment / seg0 准备

1. `scene_data = _ensure_scene_loaded(scene_id)`
2. 读取 `segment`
3. 调用现有 `_get_segment_first_pose()` / `get_segment_first_pose()`
4. 计算 `world_to_seg0 = inv(segment_first_pose)`

这一步必须原样继承当前实现，因为 seg0 的语义现在已经很清楚，且依赖 camera-0 首帧 pose。

### 8.2 校验 source/target/test image refs

对 `source_image_ref`、每个 `target_image_ref` 和可选 `test_image_refs` 做合法性检查：

- `frame_idx` 是否属于对应 segment
- `cam_id` 是否有效
- `target[0] == source` 是否满足协议要求

### 8.3 逐图读取 ViewPack

新增内部 helper：

```python
def _load_view_from_image_ref(
    self,
    scene_dataset: DrivingDataset,
    image_ref: ImageRef,
) -> Dict[str, Any]:
    ...
```

它应返回单张图像的：

- `image`
- `extrinsic`
- `intrinsic`
- `depth`
- `sky_mask`
- `viewdirs`
- `egocar_mask`
- `frame_idx`
- `cam_idx`

现有 `get_frame_data(scene_id, frame_idx, cam_idx)` 已能拿到 image/extrinsic/intrinsic/depth/sky_mask；V3 可以复用其思路，但需要补齐 `viewdirs` 和 `egocar_mask`，因为当前 `get_segment_batch()` 已经在 source/target 装配时显式读取这两个字段。

### 8.4 seg0 变换

对 source/target/test 各自的 extrinsics 做：

```python
extrinsics_seg0 = world_to_seg0 @ extrinsic_world
```

这与当前 `get_segment_batch()` 和 V2 的 `_overwrite_test_views_from_explicit_frames()` 一致。

### 8.5 pointcloud 与 dynamic_info

这部分完全沿用当前实现：

- `pointcloud` 通过 `_segment_pointcloud_cache[(scene_id, segment_id)]` 复用
- 若缓存缺失，则调用 `pointcloud_generator.generate_pointcloud(...)`
- 若存在 dynamic pointcloud，则用 `_build_dynamic_info(...)` 构造 `dynamic_info`

当前实现里，dynamic_info 还会检查 `instance_mapping` 是否存在、是否需要排除静止实例 intids，这些契约 V3 不应改变。

### 8.6 batch 组装

最终 batch 应保留现有公共字段：

- `scene_id`
- `scene_folder_name`
- `segment_id`
- `aabb`
- `segment_first_pose`
- `segment_first_frame_idx`
- `segment_first_pose_source`

这些字段已经在现有 batch 中存在，也被 V2 test overwrite 逻辑依赖。

---

## 9. V3 的 source/target/test 语义

这是 V3 与旧版最大的变化。

### 9.1 source

V3 中：

```python
batch["source"]["image"].shape == [1, H, W, 3]
```

也就是说，source 不再是“num_source_keyframes × num_cams”的默认展开形式，而是 **单张 src 图像**。现有 `get_segment_batch()` 中 source 是按 `source_frame_indices × num_cams` 装配的；V3 明确把它改成 image-level。

### 9.2 target

V3 中：

```python
batch["target"]["image"].shape == [T, H, W, 3]
```

其中 `T = len(target_image_refs)`，每个 target 都与一个明确的 `(frame_idx, cam_id)` 对应。

### 9.3 test

V3 应支持两种模式：

- `include_test=True, test_image_refs=None`：沿用 segment 内默认 test 采样逻辑
- `include_test=True, test_image_refs=[...]`：完全显式覆盖 test 视图

V2 已经通过 `_overwrite_test_views_from_explicit_frames()` 实现了“先生成默认 batch，再显式覆盖 test”；V3 则应直接支持显式 test image refs，而无需后补丁。

---

## 10. 建议的 batch 结构

```python
batch = {
    "scene_id": Tensor[1],
    "scene_folder_name": str,
    "segment_id": int,
    "aabb": Tensor[2, 3],

    "segment_first_pose": Tensor[4, 4],
    "segment_first_frame_idx": int,
    "segment_first_pose_source": str,

    "request_meta": {
        "source_image_ref": (frame_idx, cam_id),
        "target_image_refs": [...],
        "test_image_refs": [...],
    },

    "source": {
        "image": Tensor[1, H, W, 3],
        "extrinsics": Tensor[1, 4, 4],
        "intrinsics": Tensor[1, 4, 4],
        "depth": Tensor[1, H, W],
        "frame_indices": Tensor[1],
        "cam_indices": Tensor[1],
        "keyframe_indices": Tensor[1],
        "viewdirs": Tensor[1, H, W, 3],       # optional
        "sky_mask": Tensor[1, H, W],          # optional
        "egocar_mask": Tensor[1, H, W],       # optional
    },

    "target": {
        "image": Tensor[T, H, W, 3],
        "extrinsics": Tensor[T, 4, 4],
        "intrinsics": Tensor[T, 4, 4],
        "depth": Tensor[T, H, W],
        "frame_indices": Tensor[T],
        "cam_indices": Tensor[T],
        "keyframe_indices": Tensor[T],
        "viewdirs": Tensor[T, H, W, 3],       # optional
        "sky_mask": Tensor[T, H, W],          # optional
        "egocar_mask": Tensor[T, H, W],       # optional
    },

    "test": {...},  # optional

    "pointcloud": {...},     # optional
    "dynamic_info": {...},   # optional
}
```

相比现有 batch，这里最大的变化是：`keyframe_indices` 对于 target 不再是“按 keyframe 数量计”，而是应与 image refs 对齐，按每张 target 图写出它所属的 keyframe index。当前 `get_segment_batch()` 里 target 的 `keyframe_indices` 还是按选中的 target keyframes 列表直接写入的，长度与 target 图片数并不严格一一对应；V3 建议修正成 image-level 对齐。

---

## 11. 索引层设计

### 11.1 `frame_to_keyframe`

V3 应在 `SegmentIndex` 中显式缓存：

```python
frame_to_keyframe[frame_idx] -> keyframe_idx
```

因为：

- V2 已经每次都在做这个映射
- scheduler V4 以后会高频访问它
- 它是判断“extras 是否来自不同 keyframe”的核心基础

### 11.2 `keyframe_to_frames`

同样显式缓存：

```python
keyframe_to_frames[keyframe_idx] -> List[frame_idx]
```

因为 V4 的 source 采样规则就是“先 `(keyframe, cam)`，后从该 keyframe 中随机 frame”。如果每次都去扫 `scene_data["keyframe_segments"]`，既慢也不干净。

---

## 12. preload 与 overlap 扩展点

### 12.1 preload 扩展点

V3 第一版不直接做新 preload 逻辑，但要提供：

```python
def build_preload_hint(
    self,
    scene_id: int,
    segment_id: int,
    future_image_refs: List[ImageRef],
) -> Dict[str, Any]:
    ...
```

其作用是给未来的 preload worker 提供：

- 未来可能访问哪些 frame/cam
- 需要准备哪些 metadata
- 是否建议预先 materialize 某些 segment index / pair cache

这与现有 scene preload 不冲突，后者仍负责 scene 粒度加载。

### 12.2 overlap 扩展点

V3 第一版不实现复杂重叠评分，但要提供：

```python
def get_or_compute_pair_score(
    self,
    scene_id: int,
    segment_id: int,
    src: ImageRef,
    tgt: ImageRef,
    mode: str = "none",
) -> Optional[float]:
    ...
```

并配合 `_pair_score_cache`。这样 V4 以后如果要把 extras 从“same_cam_different_keyframe”升级成“same_cam + nearby + overlap-ranked”，不需要改 V3 主接口。

---

## 13. 日志与可观测性

V3 自身不是 scheduler，不需要发 block/reset 事件；但它需要把 batch 构建过程中的关键调试信息写入 batch 或 debug hooks。

建议新增 `request_meta` 和 `index_meta`：

```python
batch["request_meta"] = {
    "source_image_ref": ...,
    "target_image_refs": ...,
    "test_image_refs": ...,
    "assembly_mode": "image_ref",
}

batch["index_meta"] = {
    "source_keyframe_idx": ...,
    "target_keyframe_indices": [...],
}
```

这样 scheduler 的 `block_begin` / `reset_event` / `segment_begin` 日志能与 dataset 侧实际落地的 image refs 一一对应。当前 `TrainSchedulerV3` 已经在事件中输出 `source_keyframe_idx`、`source_frame_idx`、`reset_episode_idx`、`block_idx_in_segment` 等字段；V4 只要加上 `cam_id` 即可，V3 数据集则要保证 batch 内可回查这些值。

---

## 14. 与现有接口的兼容方案

### 14.1 保留接口

保留：

- `get_segment_batch(scene_id, segment_id, include_test)`
- `sample_random_batch(...)`
- `get_segment_batch_from_frames(...)`

原因是当前 trainer / test / eval manifest 可能还依赖它们。现有 `EvalSchedulerV2` 和 `TrainSchedulerV3` 都在直接调用 `get_segment_batch_from_frames()`。

### 14.2 新旧关系

推荐关系是：

```text
legacy random path:
    get_segment_batch(...)

compat path:
    get_segment_batch_from_frames(...)
        -> internally converts to image refs or frame-bundles
        -> calls get_segment_batch_from_image_refs(...)

canonical v3 path:
    get_segment_batch_from_image_refs(...)
```

这样，V3 成为唯一正式 batch 组装通道，老接口只是包装层。

---

## 15. 测试设计

### 15.1 索引层测试

验证 `get_segment_index()`：

- `frame_to_keyframe` 正确
- `keyframe_to_frames` 正确
- `segment_first_frame_idx` 与 `segment["frame_indices"]` 最小值一致

### 15.2 seg0 语义测试

验证：

- seg0 必须来自 camera-0
- `world_to_seg0` 可逆
- source/target/test extrinsics 都正确变换到 seg0
- `batch["aabb"] == dataset.segment_aabb`

这些都来自当前实现的核心约束。

### 15.3 image-ref batch 测试

验证：

- `source` 长度为 1
- `target` 长度与 `target_image_refs` 一致
- `frame_indices` / `cam_indices` 与 request 一致
- `keyframe_indices` 与 image refs 映射一致

### 15.4 pointcloud / dynamic_info 复用测试

验证：

- 同一 `(scene_id, segment_id)` 连续调用不会重复生成 pointcloud
- scene unload 时对应 segment pointcloud cache 被清理
- dynamic pointcloud 存在时 `dynamic_info` 正常构建

当前 `MultiSceneDataset` 已有 segment 级 pointcloud cache 和 scene 卸载清理逻辑，V3 必须回归测试这一点。

### 15.5 兼容层测试

最重要的一条回归测试是：

> 即使禁用 / mock `_select_source_and_target_keyframes()` 和 `_select_frame_from_keyframe()`，V3 的 `get_segment_batch_from_image_refs()` 仍应正常工作。

这样可以强行保证 V3 已经摆脱 V2 的 patch 依赖。

---

## 16. 推荐实现顺序

1. 实现 `SegmentIndex`、`get_segment_index()` 和 `_segment_index_cache`。
2. 实现 `_load_view_from_image_ref()` 与 `get_segment_batch_from_image_refs()`。
3. 将 `get_segment_batch_from_frames()` 改写为兼容包装层。
4. 补齐 test 显式覆盖、viewdirs/sky_mask/egocar_mask 对齐。
5. 再接 `TrainSchedulerV4`。

这样风险最低，因为现有底层 scene/segment/pointcloud 逻辑都不需要动。

---

## 17. 一句话定义

**`MultiSceneDatasetV3` 是一个继承现有 `MultiSceneDataset` 生命周期与 seg0 语义、以 `ImageRef=(frame_idx, cam_id)` 为主原语、正式负责 `image-ref → seg0-aligned batch` 组装的数据装配层；它替代 `MultiSceneDatasetV2` 的 monkey-patch frame facade，并为 `TrainSchedulerV4`、future preload 和 overlap scoring 提供稳定接口。**

---

## 附录：后续可补充的工程化交付物

实现阶段可另行产出（不必与本设计文档混写过长）：

- 类接口完整签名 + 关键方法伪代码
- 配置项清单（YAML 键名与 fast-fail 规则）
- 与 `TrainSchedulerV4` 对齐的事件 / batch 字段对照表
- 单测文件与最小复现场景 fixture 说明
