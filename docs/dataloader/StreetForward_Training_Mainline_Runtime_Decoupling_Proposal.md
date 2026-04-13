# StreetForward 训练主链 Runtime 解耦方案（仅训练路径，不含 test）

## 背景与问题定义

当前 `MultiSceneDatasetV3._assemble_segment_batch_from_image_refs()` 在训练主链上，仍会先执行：

- `scene_data = self._ensure_scene_loaded(int(scene_id))`

这会把 `scene_data["dataset"]`（`DrivingDataset` runtime）提前拉起，即使后续训练样本理论上可以完全由离线资产提供。  
因此，虽然已经有资产化能力，训练主链仍未达到“完全不依赖整 scene runtime”的 solved 状态。

## 本次目标（严格限定）

只讨论并实现**训练主链**（不碰 test 路径）：

- 在 `include_test=False` 的训练主路径中，`_assemble_segment_batch_from_image_refs()` 可以仅依赖以下资产：
  - `SegmentIndex asset`
  - `segment pose asset`
  - `scene image_table asset`
  - `segment pointcloud asset`
  - `dynamic_tracks asset`
- 不再在主链开头调用 `_ensure_scene_loaded()`。
- 仅当策略允许 fallback（`missing_policy=ignore`）且资产不足时，才延迟触发 runtime 加载。

## 目标形态（训练主链依赖图）

训练 batch 组装（`include_test=False`）的理想数据流：

1. `get_segment_index(scene_id, segment_id)` -> `SegmentIndex asset`
2. `_ensure_segment_pose_cached_from_assets_only(scene_id, segment_id)` -> pose asset
3. `_get_cached_or_load_view_from_image_ref(..., scene_dataset_opt=None)`  
   - 优先 `scene image_table asset` + `_load_view_from_asset_paths()`
   - 仅 `ignore` 下允许 runtime fallback
4. `_ensure_segment_pointcloud_cached(scene_id, segment_id, segment_first_pose)` -> pointcloud asset
5. dynamic 分支：
   - 若 pointcloud dynamic 非空：`dynamic_tracks asset` 必须可用（`error` 下缺失即失败）
   - 若 pointcloud dynamic 为空：允许无 tracks

## 关键改造点

## 1) 从 batch 入口移除 runtime 前置加载

在 `_assemble_segment_batch_from_image_refs()` 中：

- 删除/延迟主链开头的 `_ensure_scene_loaded(int(scene_id))`。
- 主链先走纯资产路径：
  - `sidx = get_segment_index(...)`
  - `segment_first_pose/world_to_seg0 = _ensure_segment_pose_cached_from_assets_only(...)`
  - source/target 图像 pack 由 `_get_cached_or_load_view_from_image_ref(..., scene_dataset_opt=None)` 获取

仅当以下条件同时满足时再 fallback：

- `missing_policy == "ignore"`
- 且资产路径缺失导致无法继续（例如 image_table 不完整）

## 2) 把 runtime 依赖缩到“叶子 fallback”

避免在 batch 顶层传播 `scene_dataset`。建议统一为：

- batch 顶层：`scene_dataset_opt = None`
- 叶子函数内部：
  - 先 asset-only
  - 再按 policy 决定是否 runtime fallback

这样可以确保 error 策略下任何深层都不会“偷偷拉 runtime”。

## 3) 动态信息主路径收口（训练主链）

在 `include_test=False` 场景下，只基于训练 source/target 帧构建 `all_frame_indices`。  
dynamic 逻辑建议固定为：

- dynamic pointcloud 非空：
  - 先读 `dynamic_tracks`
  - `error` 模式 tracks 缺失直接报错
  - `ignore` 模式可选 runtime `_build_dynamic_info(...)` fallback
- dynamic pointcloud 为空：
  - tracks 缺失不报错，`dynamic_info=None` 合法

## 4) 明确“不碰 test 路径”

本次只保证训练主链 solved，不改变 test 路径接口行为：

- `include_test=False` 作为训练主路径前提
- test 分支可以保持现有策略（后续单独阶段处理）

## 分阶段实施建议

### Phase T1: 入口断 runtime（最小闭环）

- 改 `_assemble_segment_batch_from_image_refs()`：
  - 训练主链移除 `_ensure_scene_loaded()` 前置调用
  - 仅 asset-only 组装 source/target + pose + pointcloud
- 验证 `include_test=False` 下可完整返回 batch（满足 trainer 契约）

### Phase T2: fallback 下沉与策略一致性

- 把 runtime fallback 收缩到叶子函数
- `error/rebuild` 禁止 fallback；`ignore` 允许 fallback
- 补充策略日志，便于诊断“为什么进入 fallback”

### Phase T3: dynamic 收口强化

- dynamic 非空强制 tracks 主路径
- dynamic 为空允许无 tracks
- 保持 test 分支不变

## 验收标准（训练主链 solved 判定）

在 `include_test=False` 且 `missing_policy=error`、资产齐全时：

1. `_assemble_segment_batch_from_image_refs()` 不调用 `_ensure_scene_loaded()`
2. 不触发 runtime 视图加载 `_load_view_from_image_ref(...)`
3. batch 成功返回，且保留 trainer 必需字段（`source/target/pointcloud/...`）
4. dynamic 行为符合：
   - dynamic 非空 + tracks 缺失 -> 明确失败
   - dynamic 为空 + tracks 缺失 -> 正常通过

## 建议新增测试（仅训练主链）

- `test_train_mainline_no_scene_runtime_load_in_error_mode`
  - `include_test=False`
  - 断言 `_ensure_scene_loaded` 调用次数为 0
- `test_train_mainline_view_loading_asset_only_in_error_mode`
  - 断言 `_load_view_from_image_ref` 调用次数为 0
- `test_train_mainline_dynamic_tracks_required_when_dynamic_non_empty`
- `test_train_mainline_dynamic_tracks_optional_when_dynamic_empty`

## 风险与注意事项

- 某些历史资产可能缺少 image_table 关键字段（路径/尺寸/几何），需先做资产完整性校验。
- `ignore` 下 fallback 会掩盖资产问题，建议在日志中打印 fallback 原因与资产缺失点。
- 训练主链 solved 不等于全链路 solved；test/eval 路径需后续单独收敛。

## 总结

“已解决”的最小充分条件是：  
`_assemble_segment_batch_from_image_refs()` 的训练主链（`include_test=False`）不再以前置方式依赖 `scene_data["dataset"]`，而是以资产为默认主路径，仅在 `ignore` 策略下按需延迟 fallback。  
本方案正是围绕这个判定标准设计。

