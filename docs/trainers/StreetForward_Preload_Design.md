# StreetForward Preload 设计方案（MultiSceneDatasetV3 + TrainSchedulerV4）

**适用范围**：在现有 [`MultiSceneDataset`](../../datasets/multi_scene_dataset.py) / [`MultiSceneDatasetV3`](../../datasets/multi_scene_dataset_v3.py) / [`TrainSchedulerV4`](../../datasets/multi_scene_dataset_v3.py)（实现位于 `multi_scene_dataset_v3.py`）之上**扩展**，而非另起炉灶。

**相关文档**：

- [`MultiSceneDataset_V3_Usage.md`](../dataloader/MultiSceneDataset_V3_Usage.md) — V3 调度与数据契约背景
- [`TrainScheduler_V4_Implementation_Plan.md`](TrainScheduler_V4_Implementation_Plan.md) — V4 状态机、`build_preload_hint` 与事件流
- 实现基线：[`datasets/multi_scene_dataset_v3.py`](../../datasets/multi_scene_dataset_v3.py)

**目标**：把当前「只发 hint、不实际 preload」的状态，升级为「可控、可测、可扩展」的完整 preload 子系统。

> 说明：[`TrainScheduler_V4_Implementation_Plan.md`](TrainScheduler_V4_Implementation_Plan.md) §3.2 第一版曾约定「不把 preload 做成独立后台线程系统，仅 emit」。本文档描述的是**下一阶段**的落地设计；与当时文档的关系是**演进关系**，而非矛盾。

---

## 1. 背景与现状

### 1.1 第一类：场景级 preload（已存在）

基类 `MultiSceneDataset` 在 `initialize()` 中会先保证 training queue 足够，再调用 `_preload_scenes()`；后者保证当前 scene 已加载，并把后续 `preload_scene_count` 个 scene 放进 `train_scenes_cache`。cache 满时 `_ensure_scene_loaded()` 会优先卸载非当前 scene。  
这一层解决的是：**下一个 scene 切过来时，尽量减少 DrivingDataset 初始化与段划分重复成本。**

### 1.2 第二类：调度级 preload signal（已存在，未执行）

`MultiSceneDatasetV3.build_preload_hint()` 当前将 `future_image_refs` 压缩为结构化字典，包含 `scene_id` / `segment_id` / `future_image_refs` / `unique_frame_indices` / `unique_cam_indices` / `hint_version`（见实现）。

`TrainSchedulerV4`（同文件内）在 episode 开始时调用 `_emit_preload_hint_episode_superset()`，在 **每个 block 开始后** 调用 `_emit_preload_hint_next_block_exact()`；后者在 `try/finally` 中保存并恢复 `random.getstate()`，避免 peek 下一 block 时扰动真实采样路径。

### 1.3 当前缺口（中间层未落地）

- 没有后台 worker 真正消费 hint  
- 没有 image-ref 级视图缓存与 LRU/按 scene 淘汰  
- 没有统一的 segment-static warmup 执行管线（虽有 `_segment_index_cache`、`_segment_pointcloud_cache` 等可复用入口）  
- `get_segment_batch_from_image_refs()` → `_assemble_segment_batch_from_image_refs()` 路径上仍多次 `_load_view_from_image_ref()`，block 内重复视图无缓存复用  

因此：**框架接口（hint + `build_preload_hint`）已就位，执行链与缓存策略尚未落地。**

---

## 2. 设计目标

1. **隐藏 I/O 与重复解析**：尽量提前准备下一 block、下一 episode、下一 scene 会用到的静态与准静态数据。  
2. **不改变训练语义**：preload 不得改变 `TrainSchedulerV4` 的随机性、block 边界、episode 窗口、target 采样结果，也不得偷偷重排 batch。`_emit_preload_hint_next_block_exact()` 保存/恢复 RNG 的原则必须保留。  
3. **为 overlap / 多 src / 更复杂 target policy 留扩展位**：执行层只消费 scheduler 给出的候选或 exact refs（image-ref 协议），不把「same-cam-different-keyframe」等策略写死在 preload 层。

---

## 3. 非目标

本版 preload **不负责**：

- 修改 V4 采样策略  
- 实现 overlap score（`_pair_score_cache` 等可后续接入优先级）  
- 预渲染 proxy 或模型 forward  
- 在 preload 线程中做 node state 或 trainer 侧计算  
- 缓存整份 batch tensor  

preload 只负责 **dataset 侧静态或准静态内容**（scene 已加载前提下的 segment 索引、点云、单视图 pack 等）。

---

## 4. 总体设计：两级 preload + 一个执行器

### 4.1 Level A：场景级 preload（保留现有逻辑）

继续沿用 `MultiSceneDataset` 现有机制，包括但不限于：

- `scene_training_queue`、`train_scenes_cache`、`preload_scene_count`  
- `_ensure_training_queue_ready()`、`_preload_scenes()`、`_ensure_scene_loaded()`、`_unload_scene()`  

### 4.2 Level B：segment / image-ref 级 preload（本次扩展重点）

建议新增三类预热对象：

| 对象 | 含义 |
|------|------|
| **segment static** | `SegmentIndex`、`segment_first_pose` / `world_to_seg0`、segment pointcloud；可选 deterministic test refs |
| **view pack** | 单个 `ImageRef=(frame_idx, cam_id)` 对应的图像、外参、内参、深度、sky_mask、viewdirs、egocar_mask（与 `_load_view_from_image_ref()` 返回结构对齐） |
| **optional frame metadata** | 为将来 `dynamic_info` 提速准备的 per-frame 元数据缓存 |

`SegmentIndex` 与 pointcloud 已有 `_segment_index_cache`、`_segment_pointcloud_cache`（及基类 pointcloud 清理逻辑）；**preload 应复用这些 cache**，而不是平行再造一套「第二套 segment 缓存」。

### 4.3 统一执行器：`DatasetPreloadManager`（建议新增）

职责概要：

- 接收 scheduler hint 或 dataset 内部 warmup 请求  
- 按优先级排队、去重、后台执行（v1 建议单线程）  
- 写入各级 cache、淘汰、统计 hit/miss/latency  

接口形态（草案）：

```python
class DatasetPreloadManager:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def submit(self, task: PreloadTask) -> None: ...
    def pop_stats(self) -> Dict[str, Any]: ...
```

---

## 5. 核心原则

### 5.1 Scheduler 只「发意图」，Dataset 才「执行 preload」

`TrainSchedulerV4` 已通过 `_emit({..., "type": "preload_hint", ...})` 提供**可观测**事件流。建议演进为：

- **日志通道**：保留现有 `_emit`，便于 trainer / 测试断言  
- **执行通道**：scheduler 在构造 hint 后**直接**调用 `dataset.submit_preload_hint(...)`（或等价 API），避免依赖 trainer 手工转发  

边界：**调度器不直接操作 tensor 缓存**；只传结构化 hint。

### 5.2 preload 不能推进随机状态

`_emit_preload_hint_next_block_exact()` 内对 `random.getstate()` / `random.setstate()` 的用法必须保留；后台任务不得调用会改变全局 Python `random` 状态的采样逻辑（若未来多线程，需隔离 RNG 或使用独立 `random.Random` 实例仅用于非训练路径）。

### 5.3 scene unload 必须联动清理 preload 相关 cache

`MultiSceneDatasetV3._unload_scene()` 已在基类 pointcloud 清理之上，清理 `_segment_index_cache` 与 `_pair_score_cache`。新增 `_view_pack_cache`、`_test_refs_cache`、`_segment_pose_cache` 等时，必须在同一卸载路径按 `scene_id` 联动删除。

### 5.4 v1 默认单线程 worker

在未确认 `pixel_source.get_image()` 等接口线程安全之前，preload worker 默认 **1 个线程**；配置中可预留 `num_workers`，后续再放开。

---

## 6. 建议新增的数据结构

### 6.1 `PreloadHintV2`（相对当前 `hint_version: 1` 的扩展）

当前 `build_preload_hint()` 仅提供 union refs 与 frame/cam 去重信息；若需区分 source/target/test 与 warmup 开关，可升级为 v2（字段草案与用户提供的一致），例如包含：

- `hint_scope`: `segment_static` | `episode_source_superset` | `next_block_exact` | `test_refs_exact`  
- `source_image_refs` / `target_image_refs` / `test_image_refs` / `future_image_refs`（后者保持向后兼容）  
- `warm_segment_index` / `warm_segment_pose` / `warm_pointcloud` / `warm_test_refs`  
- `epoch_idx` / `global_step` / `block_idx_global` / `hint_version=2`  

### 6.2 `PreloadTask`

带 `priority` 的可排序任务，类型可包括 `scene`、`segment_static`、`image_ref`、`test_refs` 等；`payload` 承载具体参数。

### 6.3 `LoadedViewPack`

单视图缓存单元（非整 batch），字段建议与 `_load_view_from_image_ref()` 对齐，并增加 `storage_device`（如 `cpu_pinned`）。

---

## 7. 缓存设计

### 7.1 复用现有 cache

preload 只负责「提前填充」，不替换语义：

- `train_scenes_cache`（基类）  
- `_segment_index_cache`、`_segment_pointcloud_cache`  
- `_pair_score_cache`（未来 overlap 优先级）  

### 7.2 建议新增 cache

**视图缓存**（key 建议含 segment，见下节）：

```python
self._view_pack_cache: OrderedDict[
    Tuple[int, int, int, int],  # scene_id, segment_id, frame_idx, cam_idx
    LoadedViewPack,
]
```

**deterministic test refs 缓存**（若与 `max_test_images` 等参数绑定）：

```python
self._test_refs_cache: Dict[
    Tuple[int, int, int],  # scene_id, segment_id, max_test_images
    List[ImageRef],
]
```

**segment pose 缓存**：

```python
self._segment_pose_cache: Dict[
    Tuple[int, int],
    Dict[str, Tensor],  # segment_first_pose, world_to_seg0, first_frame_idx 等
]
```

### 7.3 为何 view cache key 要含 `segment_id`

`ImageRef` 虽是 `(frame_idx, cam_idx)`，但 `validate_image_ref()` 与 batch 组装均依赖当前 segment 的 train/test 成员关系；按 `(scene_id, segment_id, frame_idx, cam_idx)` 缓存可避免跨 segment 误复用。

### 7.4 存储设备

v1 建议视图缓存统一 **CPU（可选 pinned）**，batch 组装时再 `.to(self.device)`，避免与训练争抢 GPU 显存。若 `get_image` 返回 GPU tensor，写入 cache 前应转 CPU。

---

## 8. preload 执行内容（与现有 API 对齐）

### 8.1 `segment_static` warmup

对 `(scene_id, segment_id)` 依次：

1. `_ensure_scene_loaded(scene_id)`  
2. `get_segment_index(scene_id, segment_id)`  
3. 计算并缓存 `segment_first_pose` / `world_to_seg0`（若尚未抽象成独立函数，可在实现时从现有 batch/seg0 路径抽取）  
4. 按配置填充 `_segment_pointcloud_cache`  
5. 若 `include_test=True`，调用 `resolve_test_image_refs_deterministic()` 并缓存结果（与 `_start_block()` 中 `block_test_image_refs` 语义一致）  

### 8.2 `image_ref` warmup

对每个 ref：先查 `_view_pack_cache`，miss 则 `_load_view_from_image_ref()` → 转为 `LoadedViewPack` → 写入 cache → 维护 LRU。

### 8.3 `test_refs` warmup

在 block 内 test refs 固定时，对 deterministic test refs 列表做与 8.2 相同的视图预热。

---

## 9. 与 `TrainSchedulerV4` 的集成方式

### 9.1 保留现有发射点

- `_emit_preload_hint_episode_superset(scene_id, segment_id, pair_list)`  
- `_emit_preload_hint_next_block_exact(st)`  

语义与代码一致：episode begin 适合大范围 source 候选；**block begin 之后** 发出的 `next_block_exact` 对应「下一 block」的 peek（当前实现通过 `pair_cursor` 与 `_refs_for_pair` 与真实 `_start_block()` 对齐，且保存 RNG）。

### 9.2 从「仅事件」升级为「事件 + 执行」

在以上两函数在构建 `hint` 并 `_emit` 之后，若配置开启且 `dataset` 实现 `submit_preload_hint`，则调用之，传入 `hint_scope`（`episode_source_superset` / `next_block_exact`）、`epoch_idx`、`global_step`、`block_idx_global`、`include_test` 等，使 **trainer 不必参与 preload 执行链**。

### 9.3 两种 hint 的语义（与实现对齐）

| `hint_scope` | 当前覆盖范围 | 适合 warmup |
|----------------|--------------|-------------|
| `episode_source_superset` | 本 episode 内所有 `(keyframe, cam)` 可能用到的 **source 帧集合**（由 `keyframe_to_frames` 展开），不含随机 target extras、不含 test | scene 已加载、segment static、大范围 source views（低优先级） |
| `next_block_exact` | 在 RNG 保护下 peek 的下一 block 的 `source_ref` + `target_refs` | exact source/target 视图；建议可同时 warm **test refs**（与 `include_test` 配合） |

---

## 10. batch 组装路径改造建议

### 10.1 当前行为

`_assemble_segment_batch_from_image_refs()` 内通过 `_load_stack_role()` 等多次调用 `_load_view_from_image_ref()`，block 内多步会重复加载相同 ref。

### 10.2 建议

新增 `_get_cached_or_load_view_from_image_ref(scene_id, segment_id, scene_dataset, image_ref)`：

1. 查 `_view_pack_cache`  
2. hit：返回缓存  
3. miss：调用 `_load_view_from_image_ref()`，按需写回 cache  

将 `_load_stack_role()` 等处对 `_load_view_from_image_ref` 的直接调用替换为上述包装，使 **preload 与训练路径共用同一缓存**，行为一致、仅性能不同。

---

## 11. `dynamic_info`（分阶段）

- **v1**：不缓存整份 `dynamic_info`（key 碎、命中率与一致性成本高）。  
- **v1.5**：可选 per-frame 动态元数据 cache，key `(scene_id, segment_id, frame_idx)`，batch 时仍按当前 frame union 组装。  
- **v2**：仅对 `next_block_exact` 的 frame union 等 exact scope 做小块缓存（非 v1 必需）。

---

## 12. 配置设计（草案）

保留 `preload_scene_count` 作为场景级参数；新增独立 `data.preload` 块，例如：

- worker：`enable`、`num_workers`、`max_pending_tasks`、`queue_policy`  
- warm 开关：`warm_episode_source_superset`、`warm_next_block_exact`、`warm_test_refs`、`warm_segment_static_on_segment_begin` 等  
- cache：`enable_view_pack_cache`、`view_cache_max_items_total`、`view_cache_max_items_per_scene`、`eviction_policy`、`view_cache_device`  
- 行为：`drop_stale_hints`、`dedupe_tasks`、`allow_sync_fallback_on_miss`、`strict_mode`  

具体键名与默认值应在实现时与现有 YAML 风格对齐，并遵循 **fast-fail、非必要不默认值** 的项目偏好。

---

## 13. 优先级与淘汰

**任务优先级**（建议从高到低）：

1. `next_block_exact`  
2. `test_refs_exact`（或与 next block 合并调度）  
3. `segment_static`  
4. `episode_source_superset`  
5. scene 级 lookahead（基类已承担主要工作，不宜与 exact 抢满队列）  

**视图 cache 淘汰**：LRU；优先驱逐非 current scene，其次非 current segment，最后最久未访问。  
**scene unload**：与 §5.3 联动，整 scene 键一次性清理。

---

## 14. 日志与可观测性

保留 scheduler 侧 `preload_hint` 事件；dataset / `DatasetPreloadManager` 侧建议增加：

- `preload_submit` / `preload_start` / `preload_finish`（含 latency、加载视图数、cache hit/miss）  
- `preload_drop_stale`（原因：`scene_unloaded` | `segment_advanced` | `queue_overflow`）  
- 周期性 `view_cache_stats`（命中率、当前条目数、驱逐次数）

---

## 15. 测试建议

1. **语义不变性**：固定 seed，preload on/off 对比同一 batch 的 refs 与 tensor、`request_meta`。  
2. **RNG 不受扰动**：`_emit_preload_hint_next_block_exact` 之后，真实 `_start_block()` 采样结果与关闭 preload 时一致。  
3. **cache 命中**：同一 block 内多次 `next_batch()`，视图层应出现 hit（在开启 cache 与 warm 的前提下）。  
4. **scene unload**：`_unload_scene()` 后，该 scene 相关 view/test/pose 缓存清空。  
5. **deterministic test refs**：`include_test=True` 时 block 内 test refs 恒定，preload 不应改变顺序与内容。  
6. **stale hint**：segment 切换后旧 hint 不应污染新 segment 缓存。  
7. **LRU**：超过上限时按策略驱逐。

现有单测可参考 [`tests/test_train_scheduler_v4.py`](../../tests/test_train_scheduler_v4.py)、[`tests/test_multi_scene_dataset_v3.py`](../../tests/test_multi_scene_dataset_v3.py)，扩展 preload 专项用例。

---

## 16. 推荐落地顺序

1. **MVP**：`DatasetPreloadManager`（单线程）+ `_view_pack_cache` + `submit_preload_hint()` + `_get_cached_or_load_view_from_image_ref()` + 仅 `next_block_exact` 执行路径。  
2. **segment_static**：pose cache、pointcloud warmup、deterministic test refs warmup。  
3. **episode_source_superset**：低优先级闲时 warm。  
4. **进阶**：per-frame dynamic meta、overlap 感知优先级、多 worker。

---

## 17. 结论性定义

> **基类 `MultiSceneDataset` 继续负责 scene 级 preload；`MultiSceneDatasetV3` 扩展 image-ref / segment-static 级缓存与执行器；`TrainSchedulerV4` 继续生成 preload hint，并增加对 dataset 执行接口的调用；batch 组装优先走视图缓存，miss 时同步 fallback。**

该划分贴合现有代码边界与 image-ref 协议，不破坏 V4 随机性与 block 语义，且便于测试与后续扩展。

---

## 18. 后续文档（可选）

若需要更贴近提交的接口草案，可另补一篇：

- `MultiSceneDatasetV3` 上 `submit_preload_hint` / `DatasetPreloadManager` 的类与方法清单  
- `TrainSchedulerV4` 中需改动的函数列表（`_emit_preload_hint_episode_superset`、`_emit_preload_hint_next_block_exact` 等）

本文档仅固定**设计与原则**；具体签名以实现 PR 为准。
