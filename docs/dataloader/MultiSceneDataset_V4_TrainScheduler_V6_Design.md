# MultiSceneDatasetV4 与 TrainSchedulerV6 实现方案

## 文档目标

本文定义 StreetForward 训练数据链路的下一阶段重构方案：

- 新建 `MultiSceneDatasetV4`（纯资产读取与 batch 组装）
- 新建 `TrainSchedulerV6`（调度与执行解耦）
- 新建 `AssetPreloadManagerV2`（纯资产预热）
- 保留 `MultiSceneDatasetV3` / `TrainSchedulerV5` 作为旧路径，仅维护 bugfix

第一阶段采用单一 canonical 语义：

- **Planner 语义：frame-level（对齐 V5）**
- **Dataset 语义：image-ref batch API（执行层唯一适配）**
- **调度策略：no-overlap，不引入 pair-score/overlap 机制**

---

## 1. 为什么不继续改 V3/V5

当前链路已出现明确混叠：

- `MultiSceneDatasetV3` 中 asset-only 与 runtime scene 语义并存
- preload 路径仍可能触发 `_ensure_scene_loaded()`
- asset 导入仍存在“目录扫描 + mtime”路径

这会导致：

1. 维护成本高（一个类维护两套世界观）
2. 行为不确定（无法静态保证不触发 runtime load）
3. 资产一致性风险（多版本并存时可能静默错配）

因此采用硬切割：V4/V6/PreloadV2 为新主线，V3/V5 冻结。

---

## 2. 总体目标与硬约束

### 2.1 总体目标

- Dataset 层：只做资产读取、校验、batch 组装
- Scheduler 层：只做 block 计划、执行推进、事件输出
- Preload 层：只做 asset cache 预热

### 2.2 三条硬约束

1. `MultiSceneDatasetV4` 禁止 `DrivingDataset`、禁止 runtime scene load、禁止 pointcloud generation
2. `TrainSchedulerV6` 只通过 `get_segment_batch_from_image_refs()` 向数据层取 batch
3. `AssetPreloadManagerV2` 只预热 asset handle / static bundle / image meta / view pack

---

## 3. MultiSceneDatasetV4 设计

### 3.1 类定位与继承

`MultiSceneDatasetV4` 是纯资产 batch assembler，不是 scene manager。

- 不继承 `MultiSceneDataset`
- 不持有 scene queue / current scene index
- 不包含 `_ensure_scene_loaded()`、scene unload、pointcloud generator

### 3.2 对外接口（第一阶段）

- `initialize()`
- `shutdown_preload()`
- `get_segment_index(scene_id, segment_id)`
- `validate_image_ref(scene_id, segment_id, image_ref, purpose)`
- `get_segment_batch_from_image_refs(request)`
- `build_preload_hint(...)`
- `submit_preload_hint(...)`

第一阶段不暴露 `get_or_compute_pair_score()`，不提供 overlap 相关接口。

### 3.3 include_test 策略（第一阶段写死）

第一阶段主训练链与当前主线保持一致：

- `include_test=False`：唯一支持路径
- `include_test=True`：直接 fast-fail（明确报错）

说明：避免出现“接口支持 test、主链忽略 test”的双语义状态。

### 3.4 内部对象

- `SceneAssetHandle`：scene manifest + image table index
- `SegmentStaticBundle`：segment index / pose / pointcloud / dynamic tracks
- `LoadedViewPackV2`：单 image-ref 读取结果

### 3.5 缓存池（asset_id 作为主键）

- `scene_asset_cache[(dataset, scene_asset_id)] -> SceneAssetHandle`
- `segment_static_cache[(dataset, segment_asset_id)] -> SegmentStaticBundle`
- `image_meta_cache[(dataset, scene_asset_id, frame_idx, cam_id)] -> row`
- `view_pack_cache[(dataset, segment_asset_id, frame_idx, cam_id)] -> LoadedViewPackV2`

约束：运行期先通过 registry resolve `asset_id`，后续全程以 `asset_id` 作为身份键。

### 3.6 batch 组装流程

`get_segment_batch_from_image_refs()` 固定执行：

1. 通过 `segment_asset_id` 读取 `SegmentStaticBundle`
2. 校验 source/target image-ref 协议（训练态强校验 `target[0] == source`）
3. 通过 `scene_asset_id` 读取 image table 元数据
4. 命中或加载 `LoadedViewPackV2`
5. 注入 `aabb` / `segment_first_pose` / `pointcloud`
6. dynamic 点云非空时，强依赖 `dynamic_tracks` 组装 `dynamic_info`
7. 组装 source/target 并返回 minimal trainer 可直接消费的 batch

### 3.7 图像尺寸与内参对齐规则（硬约束）

读取 image/depth/mask 时，若文件尺寸与 asset meta 的 `height/width` 不一致：

- 必须按 meta 尺寸重采样
- 再与 `intrinsic_4x4_flat` 对齐输出

禁止“按文件原始尺寸直接返回”的路径，以避免像素网格与内参错位。

### 3.8 strict 模式

V4 只允许：

- `missing_policy = "error"`

以下情况一律 fast-fail：

- 资产缺失
- 指纹不匹配
- parent scene/segment 关系不一致
- image table 缺少请求 ref
- dynamic 点云非空但 `dynamic_tracks` 缺失或校验失败

### 3.9 dynamic tracks 合同前移

`dynamic_tracks` 不仅是 runtime contract，还是 build-time contract：

- segment asset 导出阶段必须校验“dynamic 非空 => dynamic_tracks 完整”
- registry 标记 ready 前必须完成该校验
- V4 `initialize()` 做一次轻量一致性检查，尽早失败

---

## 4. 资产解析与一致性修正（必须执行）

### 4.1 registry-only 主路径

V4 主路径必须是：

1. `segment_registry` 定位 `segment_asset_id`
2. 读取 segment manifest 的 `parent_scene_asset_id`
3. 精确打开对应 `scene_asset_id`
4. parent 不一致立即报错

禁止“目录前缀 + mtime”作为主解析逻辑。

### 4.2 禁止调用旧入口作为主入口

V4 不允许将以下旧入口作为主解析路径：

- `get_scene_asset(dataset, scene_id)`
- `get_segment_asset(dataset, scene_id, segment_id)`

建议新增并强制使用：

- `resolve_scene_asset_id_from_registry(...)`
- `resolve_segment_asset_id_from_registry(...)`
- `open_scene_asset_by_id(...)`
- `open_segment_asset_by_id(...)`

---

## 5. 共享纯工具层（必须补齐）

V4 不继承 `MultiSceneDataset` 后，必须新增共享纯函数层，避免 V3/V4 语义漂移。

建议新增：

- `datasets/streetforward_batch_utils.py`
- `datasets/streetforward_asset_view_utils.py`

最少包含以下纯函数：

- sky/depth/mask 占位与 dtype 规范
- intrinsic/extrinsic 4x4 规范化
- `world -> seg0` 变换
- dynamic tracks -> `dynamic_info` 组装
- batch 字段排序与组织规范

---

## 6. AssetPreloadManagerV2 设计

### 6.1 目标

替代 runtime 依赖 preload 路径，保持 asset-native。

建议新增文件：`datasets/asset_preload_manager_v2.py`

### 6.2 任务类型（第一阶段）

- `PRELOAD_SCENE_META`
- `PRELOAD_SEGMENT_STATIC`
- `PRELOAD_VIEW_META`
- `PRELOAD_VIEW_PACK`

第一阶段不定义 overlap 相关 preload 任务类型。

### 6.3 preload hint（建议）

```python
{
    "hint_version": 3,
    "scene_id": 1,
    "segment_id": 2,
    "scene_asset_id": "scene-...-1234abcd",
    "segment_asset_id": "seg-...-abcd1234",
    "future_image_refs": [...],
    "required_static": {"segment_bundle": True},
    "scope": "next_block_exact" | "episode_superset",
}
```

---

## 7. TrainSchedulerV6 设计

### 7.1 第一阶段原则

V6 第一阶段保持 V5 行为兼容：

- no-overlap
- source frame + target frame indices 作为 planner 输入/状态主语义
- `U -> block -> episode -> segment` 层级保持不变

### 7.2 核心对象（第一阶段 canonical）

```python
@dataclass(frozen=True)
class BlockSpecV6:
    scene_id: int
    segment_id: int
    segment_asset_id: str
    reset_episode_idx: int
    block_idx_global: int
    source_frame_idx: int
    source_keyframe_idx: int
    target_frame_indices: list[int]
    episode_window_keyframes: list[int]
```

### 7.3 Planner/Executor 适配边界

- Planner 只产出 frame-level `BlockSpecV6`
- Executor 负责唯一一次适配到 image-ref：
  - `source_frame_idx -> [(source_frame_idx, cam_id) for cam_id in range(num_cams)]`
  - `target_frame_indices -> [(frame_idx, cam_id) ...]`
- Dataset 只接收 image-ref request

这样保持“V5 语义兼容 + V4 API 统一”，避免双重 planner 语义。

### 7.4 组件拆分

- `BlockPlanner`：生成 `BlockSpecV6`
- `PreloadCoordinator`：依据 `BlockSpecV6` 生成与提交 preload hint
- `BatchExecutor`：执行 `next_batch()`、推进 U/step、输出事件

### 7.5 状态层级

- `EpochPlanV6`
- `SegmentEpisodeState`
- `ExecutionState`

### 7.6 日志与 `get_current_info()` 口径

第一阶段保持 V5 字段兼容，不改现有字段名：

- `segment_local_step`
- `segment_step_budget`
- `segment_local_u`
- `segment_budget_u`
- `block_idx_in_segment`
- `block_idx_global`
- `source_frame_idx`
- `source_keyframe_idx`
- `source_cam_idx`
- `source_image_ref`
- `target_image_refs`
- `U` / `K_steps` / `T_steps`

仅新增：

- `segment_asset_id`
- `block_spec_id`（可选）

---

## 8. 分阶段落地计划

### Phase 0：冻结旧路径

- `dataset_runtime_version: v4`
- `scheduler_version: v6`
- V3/V5 仅接受 bugfix

### Phase 1：MultiSceneDatasetV4

新增：`datasets/multi_scene_dataset_v4.py`

先完成：

- asset-only batch assemble
- strict error policy
- include_test fast-fail（训练主链）
- 无 preload（先跑通功能闭环）

### Phase 2：AssetPreloadManagerV2

新增：`datasets/asset_preload_manager_v2.py`

接入 preload hint 与缓存池策略。

### Phase 3：TrainSchedulerV6

新增：`datasets/train_scheduler_v6.py`

实现 V5 兼容 `segment_local_block_mode` 与 `BlockSpecV6`。

### Phase 4：训练入口接线

建议新增：`tools/train_minimal_streetforward_stage4_3_v6_common.py`

- `build_multi_scene_dataset_v4(cfg, device)`
- `build_train_scheduler_v6_from_cfg(cfg, dataset, ...)`

### Phase 5：测试

新增：

- `tests/test_multi_scene_dataset_v4.py`
- `tests/test_asset_preload_manager_v2.py`
- `tests/test_train_scheduler_v6.py`

重点覆盖：

- 纯资产路径不触发 runtime scene load
- registry-only + parent 强校验
- dynamic 非空 => tracks 必需（build-time + runtime）
- image resize 对齐 meta 尺寸与 intrinsic
- batch 与 `convert_batch_to_minimal_format()` 兼容
- V6 单 segment 行为与 V5 对齐
- `get_current_info()` 第一阶段字段兼容

---

## 9. 验收标准（Definition of Done）

满足以下条件才视为迁移完成：

1. 新训练入口可在 V4+V6 上跑通至少一个短训练回合
2. 任一资产缺失、parent 不一致、tracks 合同不满足都立即 fast-fail
3. 主路径中不再出现 runtime scene fallback
4. 关键日志事件齐全，且 `get_current_info()` 与 V5 口径兼容（仅新增字段）
5. V3/V5 旧路径仍可用于历史复现，但不影响新主线

---

## 10. 一句话定义

`MultiSceneDatasetV4` 是纯资产读取器，`TrainSchedulerV6` 是 frame-level block 计划器，`AssetPreloadManagerV2` 是纯资产预热器；三者通过硬边界解耦后，跨 scene/segment 调度将成为增量能力，而不是语义清理工程。

