# StreetForward 可复现初始化资产系统（Segment-Independent Asset Pool）

## 1. 背景与目标

当前 StreetForward 的训练调度以 scene/segment 逐个推进。该方式在实践中容易出现两个问题：

- **切换代价高**：切换 segment 时，常常需要重新读取 scene 目录并触发 pointcloud 生成。
- **训练分布窄化**：长时间停留在少量 segment，容易产生 segment-level overfit，导致 train/test gap 放大。

与此同时，`TrainSchedulerV4` 的核心调度单位已经是 `(scene_id, segment_id)` 下的 image-ref block，且 `MultiSceneDatasetV3` 已将 `get_segment_batch_from_image_refs()` 作为 canonical batch 路径。因此，初始化资产应围绕 **可复现初始化事实** 建模，而不是围绕 trainer 的临时状态建模。

本方案将系统分为两层：

- **离线资产层**：定义并固化初始化事实（split/keyframe/seg0/pointcloud/dynamic tracks/fingerprint）。
- **运行时缓存层**：复用已导入资产，降低 block 切换抖动与 IO/CPU 峰值。

「可复现初始化」不仅指指纹一致，还要求 **export 阶段所有影响资产字节的随机过程** 由 manifest 持久化的种子与确定性规则驱动；完整约定见 **§4.3**。

---

## 2. 核心设计原则

### 2.1 Segment 资产独立于 Scene 目录

不采用 `assets/scenes/<scene>/segments/<segment>/...` 的父子目录。

采用平级池化：

- `scene_pool/`：轻量索引资产
- `segment_pool/`：初始化核心资产
- `registries/`：scene/segment 与资产 ID 的关系索引

原因：

- 调度热点是 `(scene, segment)` block，不是“先 scene 再 segment”的访问路径。
- segment 资产（pointcloud/dynamic tracks）体量和变更频率高于 scene 索引，平级更清晰。
- 未来做跨 scene 的 segment 采样、预热、统计时更自然。

### 2.2 Pointcloud 是 canonical init asset

初始化资产以 pointcloud 为中心，不存 trainer 中间态。资产层只定义“可复现初始化输入”，不定义“训练过程状态”。

- **存**：`pointcloud`、`dynamic_info` 构建基表、seg0 pose、segment index
- **不存**：`NodeState`、`h_cache_*`、optimizer states、render proxy states

### 2.3 Runtime Batch API 不改

保持现有契约：

- `MultiSceneDatasetV3.get_segment_batch_from_image_refs(BatchRequestV3)` 继续为 canonical 路径
- scheduler 继续输出 `source_image_ref` + `target_image_refs`
- trainer 继续消费既有 batch 结构

资产层只替换数据来源，不改变上层语义接口。

### 2.4 无 Fallback（fast-fail）

启用预构建资产时，**不提供**「读 asset 失败则回退到在线重算 / 在线重建」的路径。约定：

- 期望的资产目录、`READY`、指纹与文件完整性任一不满足：**直接报错**，训练/导入不继续。
- 需要更新初始化语义时：改配置或改实现 → 指纹变化 → **重新 export** 新资产，而不是在运行时悄悄重算。

这样「初始化事实」只由离线资产定义，避免同一配置下出现「有时用 asset、有时用现场算」的双轨行为。

---

## 3. 目录布局与命名

```text
streetforward_assets/
  ASSET_SCHEMA_VERSION.json

  registries/
    scene_registry.parquet
    segment_registry.parquet
    asset_build_history.parquet

  scene_pool/
    scene-<dataset>-<scene_id>-<scene_fingerprint>/
      manifest.json
      scene_index.npz
      image_table.parquet
      split_summary.json
      READY

  segment_pool/
    seg-<dataset>-<scene_id>-<segment_id>-<segment_fingerprint>/
      manifest.json
      segment_index.npz
      segment_pose.npz
      pointcloud_static.npz
      pointcloud_dynamic.npz
      dynamic_tracks.npz
      stats.json
      READY

  tmp/
```

### 3.1 Asset ID

- `SceneAssetId = scene-<dataset>-<scene_id>-<scene_fingerprint_short>`
- `SegmentAssetId = seg-<dataset>-<scene_id>-<segment_id>-<segment_fingerprint_short>`

人可读 + 可追溯 + 配置/实现变化自动产生新 ID。

---

## 4. 资产内容定义

## 4.1 Scene 资产（轻量索引）

### `manifest.json`

包含：

- 基本信息：`asset_type/schema_version/asset_id/dataset/scene_id/scene_name`
- 统计信息：`num_frames/num_cams`
- 复现信息：`source_data_fingerprint/config_fingerprint/implementation_fingerprint`
- **构建随机性（见 §4.3）**：`build_seed`、`numpy_random_seed`、`python_random_seed`、`torch_random_seed`、`deterministic_flags` 等
- split 参数快照：`test_image_stride/max_test_images/segment_overlap_ratio/keyframe_split_config/...`

### `scene_index.npz`

固化 scene 级索引：

- `train_frame_indices`
- `test_frame_indices`
- `keyframe_indices`
- `keyframe_to_frames_flat`
- `keyframe_to_frames_offsets`
- `frame_to_keyframe_dense`
- `segment_ids`
- `segment_frame_offsets`
- `segment_frame_indices_flat`
- `segment_keyframe_offsets`
- `segment_keyframe_indices_flat`

### `image_table.parquet`

每行表示 `(frame_idx, cam_id)` 的图像元数据，建议字段：

- `frame_idx/cam_id/img_idx/is_train/is_test`
- `image_path/depth_path/sky_mask_path/dynamic_mask_path`
- `height/width`
- `intrinsic_4x4_flat/camera_to_world_flat`

用途：运行时读图与视图装配无需先完整初始化 `DrivingDataset`。**`image_table.parquet` 属于 scene 资产**：描述该 scene 的全局 image universe；各 segment 仅引用其中的行子集（通过 `segment_index` 中的 refs），不复制整张表。

## 4.2 Segment 资产（初始化核心）

### `manifest.json`

包含：

- 资产身份：`asset_type/schema_version/asset_id/dataset/scene_id/segment_id`
- 关联关系：`parent_scene_asset_id`
- seg0 语义：`segment_first_frame_idx/segment_pose_source/seg0_camera_id`
- 初始化几何：`segment_aabb`
- pointcloud 归一化配置：`pointcloud_config_normalized`
- 统计：`num_train_frames/num_test_frames/num_keyframes/background_points/dynamic_instances/dynamic_points`
- 复现指纹：`source_data_fingerprint/config_fingerprint/implementation_fingerprint`
- **构建随机性（见 §4.3）**：与 scene manifest 同类的 `build_seed` / 各 RNG seed / `deterministic_flags`（segment 导出在 scene 之后，须写明与 scene 的派生关系，例如 `segment_build_seed = hash(build_seed, scene_id, segment_id)` 的**固定公式**，并写入 manifest）

### `segment_index.npz`

固化 `SegmentIndex` 语义：

- `train_frame_indices`
- `test_frame_indices`
- `keyframe_indices`
- `keyframe_to_frames_flat`
- `keyframe_to_frames_offsets`
- `frame_to_keyframe_dense`
- `train_image_refs:int32[N,2]`
- `test_image_refs:int32[M,2]`

### `segment_pose.npz`

- `segment_first_pose_world: float32[4,4]`
- `world_to_seg0: float32[4,4]`
- `segment_first_frame_idx: int32[1]`
- `segment_pose_source: int32(enum)`

### `pointcloud_static.npz`

- `background: float32[N,6]`
- `background_inside_mask: uint8[N]`（可选）
- `metadata_json: utf-8 string`

### `pointcloud_dynamic.npz`

- `dynamic_points_concat: float32[M,6]`
- `dynamic_points_offsets: int64[I+1]`
- `dynamic_instance_intids: int32[I]`
- `dynamic_instance_original_ids: int32[I]`
- `instance_mapping_keys: int32[K]`
- `instance_mapping_values: int32[K]`

### `dynamic_tracks.npz`

为运行时 `_build_dynamic_info()` 提供离线基表：

- `frame_indices: int32[F]`
- `instance_intids: int32[I]`
- `instances_quats: float32[F,I,4]`
- `instances_trans: float32[F,I,3]`
- `instances_fv: uint8[F,I]`
- `static_instance_intids: int32[J]`

### `stats.json`

存可验证统计（点数、instance 数、每 keyframe 帧数、test refs 数、build 耗时等）。

## 4.3 可复现性：随机源与硬规则

仅有 fingerprint 不足以保证「相同输入 + 相同配置 + 相同代码 → 相同资产」。凡是**影响资产字节内容**的步骤，若在实现中使用了随机性（含隐式顺序依赖），必须在 **export 阶段**写死随机源与算法顺序，并**持久化到 manifest**。

### 4.3.1 Manifest 中必须持久化的字段

Scene 与 Segment 的 `manifest.json` 均建议包含（命名可微调，语义不可缺）：

| 字段 | 含义 |
|------|------|
| `build_seed` | 本资产构建的主种子；派生子步骤种子时须有**固定、可文档化**的公式 |
| `numpy_random_seed` | 构建时用于 `numpy.random` 的种子（若该步骤使用 numpy RNG） |
| `python_random_seed` | 构建时用于 `random` 模块的种子 |
| `torch_random_seed` | 构建时用于 `torch` 的种子（若 pointcloud / 几何步骤使用 torch） |
| `deterministic_flags` | 可选：如 `torch_use_deterministic_algorithms`、`cudnn_benchmark=false` 等关键开关的快照 |

实现上可为「单字段 `build_seed` + 按步骤派生」，但**派生规则必须在本文档或 schema 中写死**，且写入 manifest，避免「只记一个数却多处隐式默认」。

### 4.3.2 需逐条写明的随机敏感步骤（示例）

以下环节若在实现中带随机或等价随机的行为，export 时必须绑定到 manifest 中的种子或确定性排序规则：

- `max_test_images` 截断时：test frame / test image ref 的选取顺序（**stable sort 后按固定 RNG 抽样**，或**完全确定性下标规则**，二选一写死）。
- pointcloud 生成器：下采样、稀疏帧采样、统计滤波子采样等。
- hybrid 融合：近/远景点数上限下的取舍与合并顺序。
- dynamic recovery：点恢复、per-instance 截断与 tie-break。
- 任何仅在**训练运行时**发生的 overlap / preload 抽样：**不属于 init 资产**；若未来纳入资产构建，须单独列出并同样绑定 seed。

### 4.3.3 排序与抽样约定

对每个「截断 / 子采样」步骤，文档化下列之一并固定实现：

- **Stable sort 后按索引取前 K**（完全确定性，无 RNG）；或
- **固定 RNG stream**：在**已写入 manifest 的 seed** 上构造独立 `Generator`，并写明抽样顺序（例如先按 `(frame_idx, cam_id)` 排序再 `choice`）。

禁止依赖「进程默认全局 RNG 状态」或「未定义迭代顺序」。

### 4.3.4 硬规则

**任何影响资产内容的随机过程，必须由 manifest 中持久化的 build seed（及文档化的派生规则）驱动；禁止依赖进程默认随机状态。**

导入侧不重复执行随机初始化路径：只读已固化的 npz/parquet 与 manifest。

---

## 5. 导出（Export）设计

## 5.1 CLI 入口

新增工具：

- `tools/build_streetforward_scene_assets.py`
- `tools/build_streetforward_segment_assets.py`

示例：

```bash
python tools/build_streetforward_scene_assets.py --config_file configs/xxx.yaml --scene_id 1
python tools/build_streetforward_segment_assets.py --config_file configs/xxx.yaml --scene_id 1 --segment_id 0
python tools/build_streetforward_segment_assets.py --config_file configs/xxx.yaml --all_train_scenes
```

## 5.2 Scene 导出流程

1. 根据 §4.3 初始化构建用 RNG（种子写入即将生成的 manifest）  
2. 读取原始 scene，仅执行 split/keyframe/segment/image_table 构造（含 `max_test_images` 等截断，算法与 §4.3 一致）  
3. 计算 fingerprints  
4. 写入 `tmp/scene-...partial/`  
5. 输出 `manifest/scene_index/image_table/split_summary`（manifest 含完整复现字段）  
6. 校验（映射完整性、split 合法性、路径可访问）  
7. 原子提交 + READY

## 5.3 Segment 导出流程

1. 根据 §4.3 初始化 segment 级 RNG（与 scene manifest 的派生关系写入 segment manifest）  
2. 读取 scene asset（不重复 split）  
3. 提取 segment 索引、keyframe、seg0 首帧  
4. 计算 seg0 pose 与 `world_to_seg0`  
5. 复用现有 pointcloud generator 生成点云（不重写生成器；生成器内凡影响输出的随机性均受 manifest seed 约束）  
6. 导出 dynamic tracks 基表  
7. 写入 `segment_index/segment_pose/pointcloud_*/dynamic_tracks/stats/manifest`  
8. 校验（pose 可逆、pointcloud 可读、映射完整、smoke batch 可组装）  
9. 原子提交并更新 registry

---

## 6. 导入（Import）设计

## 6.1 新增资产访问层

建议新增目录：

```text
datasets/streetforward_assets/
  asset_store.py
  schema.py
  io_utils.py
```

核心接口：

```python
class StreetForwardAssetStore:
    def get_scene_asset(self, dataset: str, scene_id: int) -> SceneAssetHandle: ...
    def get_segment_asset(self, dataset: str, scene_id: int, segment_id: int) -> SegmentAssetHandle: ...
    def has_segment_asset(...): ...
    def verify_segment_asset(...): ...
    # 可选：对调用方最直观的入口；内部通过 get_scene_asset + image_table 实现
    def load_image_meta(self, dataset: str, scene_id: int, refs) -> ...: ...  # refs: (frame_idx, cam_id) 列表
```

`SceneAssetHandle`（与 `image_table.parquet` 同属 scene 资产）建议提供：

- `load_manifest()`
- `load_scene_index()`（如需）
- **`load_image_meta(refs)`**：按 `(frame_idx, cam_id)` 列表从 **本 scene** 的 `image_table.parquet` 查询行（或等价列视图）；refs 必须落在该 scene 的 universe 内

`SegmentAssetHandle` 建议提供（**不包含** `load_image_meta`）：

- `load_manifest()`
- `load_segment_index()`
- `load_segment_pose()`
- `load_pointcloud()`
- `load_dynamic_tracks()`

**归属说明**：`image_table.parquet` 是 scene 级全表；多个 segment 共用同一份 scene 资产时，只在 `SceneAssetHandle` 或 `StreetForwardAssetStore.load_image_meta(scene_id, ...)` 上暴露查询，避免把 scene 级表挂到 segment handle 上导致抽象变脏。

## 6.2 `MultiSceneDataset` 接入策略

新增配置化入口（保持现有必需项校验风格，**仅 fast-fail**）：

- `use_prebuilt_assets: bool`
- `asset_store: StreetForwardAssetStore`

关键点：

- `_load_and_prepare_scene()`：启用资产时**必须**成功加载 scene asset，否则报错
- `_segment_pointcloud_cache` 保留，但定位为“导入后内存镜像”

## 6.3 `MultiSceneDatasetV3` 关键改造

以下在 `use_prebuilt_assets=True` 时均为**唯一路径**：从对应 npz 加载；缺失、损坏或与校验不一致则**报错**，不回退到在线构建。

### `get_segment_index()`

从 `segment_index.npz` 导入并填入缓存。

### `_ensure_segment_pose_cached()`

从 `segment_pose.npz` 导入。

### `_ensure_segment_pointcloud_cached()`

从 `pointcloud_static.npz` 与 `pointcloud_dynamic.npz` 导入并重组为现有 dict 格式。

### `_assemble_segment_batch_from_image_refs()`

接口不改，内部数据来自资产：

- `SegmentIndex` 来自 segment asset
- `segment_first_pose/world_to_seg0` 来自 segment asset
- `pointcloud` 来自 segment asset
- `dynamic_info` 来自 `dynamic_tracks.npz` 切片
- 视图几何元数据：通过 **`SceneAssetHandle.load_image_meta(refs)`** 或 **`StreetForwardAssetStore.load_image_meta(scene_id, refs)`** 从 scene 的 `image_table` 解析；像素张量仍可走现有 IO

### `resolve_test_image_refs_deterministic()`

使用 `segment_index.npz` 内预存 `test_image_refs`（与 export 时 §4.3 规则一致），不在运行时重新抽样。

---

## 7. Scheduler 接入影响

`TrainSchedulerV4/V5` 不需要架构重写，只需保证：

- 通过 asset-backed `get_segment_index()` 拿索引（启用资产时仅该路径，失败即报错）
- preload/warm 时仅做资产 → 内存 cache 的导入，**不**触发 pointcloud 重建或 scene 全量重解析
- block 切换阶段可选 warm segment asset（尤其 pointcloud + pose），仅为性能优化，不改变初始化事实来源

这样可在不改变 `next_batch()` 输出契约的前提下，显著降低 segment 切换成本。

---

## 8. Batch 兼容性约束（硬要求）

资产导入后，输出 batch 结构必须与当前 minimal trainer 契约一致：

```python
batch = {
    "scene_id": ...,
    "segment_id": ...,
    "pointcloud": {...},
    "dynamic_info": ...,
    "source": {...},
    "target": {...},
    "test": {...},
}
```

尤其要保持：

- `pointcloud` 必有
- `dynamic_info` 若存在可被下游原样消费
- `source/target/test` 的几何字段语义不变

---

## 9. 原子写入、校验与恢复

## 9.1 原子写入流程

1. 写 `tmp/<asset_id>.partial/`  
2. 文件级 `flush + fsync`  
3. `manifest.json` 最后写  
4. 写 `READY`  
5. `rename` 到最终目录

## 9.2 导入校验规则

- 缺失 `READY`：**拒绝导入**，报错
- fingerprint 与当前期望不一致：**拒绝导入**，报错（须重新 export 匹配资产）
- npz/parquet 读取失败或 schema 不匹配：**拒绝导入**，报错

## 9.3 Registry 提交规则

仅在最终目录提交成功后，追加 `segment_registry.parquet` / `scene_registry.parquet` 记录。

---

## 10. Fingerprint 方案

至少包含四类：

- `source_data_fingerprint`：原始图像/深度/掩码/内外参/LiDAR 文件集合指纹
- `config_fingerprint`：覆盖 `segment_aabb/keyframe_split/segment_overlap/test stride/max_test_images/pointcloud.*`
- `implementation_fingerprint`：关键实现文件文本 hash（`multi_scene_dataset*.py` + pointcloud generators）
- `schema_version`：资产格式版本

建议优先采用“路径 + size + mtime + 可选采样 hash”的混合策略，在精度与构建速度间平衡。

**目录名中的短后缀（`asset_id` 内 8 位 hex）**：由 `StreetForwardAssetStore` 使用 **SHA256** 对导出载荷做规范化摘要（scene：`split_config` + `scene_index.npz` 各数组字节；segment：索引/pose/点云/dynamic tracks 等与落盘一致的字节），**不**使用 Python 内置 `hash()`，以保证跨进程、`PYTHONHASHSEED` 不变下同一内容得到同一目录名，避免 `segment_pool` 下出现重复目录且 `_resolve_*` 报「multiple assets found」。

---

## 11. 与运行时缓存（preload/LRU）的关系

保留现有 runtime cache 体系，但职责明确为“加速导入后复用”：

- `segment_static warm`：从 asset 导入 index/pose/pointcloud 到内存 cache
- `view_pack_cache`：继续做 image-ref 级 LRU；启用预构建资产时，视图元数据从 **scene** 资产的 `image_table.parquet`（经 `SceneAssetHandle` / `StreetForwardAssetStore.load_image_meta`）解析
- `pair_score_cache`：保持现状；若计算依赖 pointcloud、seg0 pose、几何量，启用预构建资产时这些数据须来自已加载的 segment/scene 资产，而非运行时现算

原则：**runtime cache 不定义初始化事实，只加速事实读取。**

---

## 12. 配置建议

启用预构建资产时，运行时以 **资产为唯一初始化事实来源**；`use_prebuilt_assets: true` 且资产缺失或校验失败应 **直接报错**（与 `missing_policy: error` 一致）。`data.assets` 必须写在训练配置的 **`data:`** 下，与 `data_root`、`dataset` 等并列。仓库内 [`tools/streetforward_assets_data_snippet.yaml`](../../tools/streetforward_assets_data_snippet.yaml) 提供 **默认完整模板**：内容与 [`configs/minimal_streetforward_stage4_4_multi_scene_v5.yaml`](../../configs/minimal_streetforward_stage4_4_multi_scene_v5.yaml) 一致并在 `data` 下合并了 `assets`（可直接作 `--config_file`，按需改 `data_root` / `assets.root` / `train_scene_ids`）。

```yaml
# 训练配置中的 data.assets 最小示例（须与代码中 MultiSceneDataset 校验一致）
data:
  assets:
    enable: true
    root: "/root/autodl-tmp/streetforward_assets"
    use_prebuilt_assets: true
    missing_policy: error
```

说明：`export.*` / `runtime_cache.*` 等为文档中的**规划项**；当前实现以 `enable` / `root` / `use_prebuilt_assets` / `missing_policy` 为准，导出 CLI 仅需 `data.assets.root`（见执行指南）。

---

## 13. 执行指南（操作步骤）

以下为从「零资产」到「资产训练」的推荐流程；路径与场景 ID 请按本机数据与配置修改。

### 13.1 环境与仓库根

- 使用项目约定环境：`conda activate drivestudio-new`（或 `conda run -n drivestudio-new …`）。
- 保证 Python 能 import 本仓库：`export PYTHONPATH=/root/drivestudio-coding`（路径改为你的克隆根目录）。
- 资产根目录：在磁盘上创建与配置一致的目录，例如 `/root/autodl-tmp/streetforward_assets`，其下将由工具生成 `scene_pool/`、`segment_pool/`、`registries/`、`tmp/` 等。

### 13.2 准备训练用 YAML

- 使用你已有的 StreetForward 训练配置（例如 `configs/minimal_streetforward_stage4_4_one_segment_v5.yaml`），保证其中 **`data.dataset`**、**`dataset.pointcloud`**、**`dataset.segment_aabb`** 等与导出时一致；否则指纹与语义可能对不上，导入会失败。
- 可直接使用 [`tools/streetforward_assets_data_snippet.yaml`](../../tools/streetforward_assets_data_snippet.yaml) 作为训练/导出配置（已含完整 `dataset` + `data` + `data.assets`）；或仅将其中的 `data.assets` 段合并进你自己的 YAML，并把 `assets.root` 改为本机资产根路径。

### 13.3 导出 Scene 资产（轻量索引）

在仓库根目录执行：

```bash
export PYTHONPATH=/root/drivestudio-coding
conda run -n drivestudio-new python tools/build_streetforward_scene_assets.py \
  --config_file configs/<你的训练配置>.yaml \
  --scene_id <scene_id>
```

- 若需批量导出训练集中的所有 scene：`--all_train_scenes`（会遍历配置里 `data.train_scene_ids`）。
- 实现说明：若 YAML 里已写 `data.assets`，导出脚本会**临时关闭**其中的 `enable`，避免导出进程本身去走「读资产」路径。

### 13.4 导出 Segment 资产（点云 + dynamic tracks）

单 scene 下单个或多个 segment：

```bash
export PYTHONPATH=/root/drivestudio-coding
conda run -n drivestudio-new python tools/build_streetforward_segment_assets.py \
  --config_file configs/<你的训练配置>.yaml \
  --scene_id <scene_id> \
  --segment_id <segment_id>
```

- 不指定 `--segment_id` 时，默认导出该 scene 下**全部** segment。
- 批量：`--all_train_scenes` 会导出每个训练 scene 的全部 segment；每个 segment 导出前会顺带写入/更新对应 scene 资产（与单次 scene 导出行为一致）。

### 13.5 启用资产并训练

1. 确认 `segment_pool/` 下存在目标 `(dataset, scene_id, segment_id)` 对应目录，且含 `READY`。
2. 训练配置中 `data.assets.enable: true`、`use_prebuilt_assets: true`、`missing_policy: error`，`root` 指向上述资产根。
3. 运行常规训练入口，例如 one-segment v4：

```bash
export PYTHONPATH=/root/drivestudio-coding
conda run -n drivestudio-new python tools/train_minimal_streetforward_stage4_3_one_segment_v4.py \
  --config_file configs/<你的训练配置>.yaml \
  --max_steps 10
```

### 13.6 回归与自检

- 单元测试（资产与 V3 相关）：  
  `pytest tests/test_streetforward_asset_store.py tests/test_multi_scene_dataset_v3.py`
- 若报错「segment asset not found」或「missing READY」：检查 `root`、数据集名 `data.dataset` 与资产目录命名前缀是否一致（`seg-<dataset>-<scene_id>-<segment_id>-*`）。

---

## 14. 分阶段落地建议

### Phase A（先解决切段慢）

- 建立 `scene_pool/segment_pool/segment_registry`
- 引入 `StreetForwardAssetStore`
- 资产化 `get_segment_index()` / `_ensure_segment_pose_cached()` / `_ensure_segment_pointcloud_cached()`

### Phase B（补齐 dynamic）

- 导出 `dynamic_tracks.npz`
- `_assemble_segment_batch_from_image_refs()` 资产切片构造 `dynamic_info`

### Phase C（补齐 metadata 直读）

- 落地 `image_table.parquet`
- 视图元数据改为资产直读
- 优化 asset-backed view loader

---

## 15. 最终原则（写入首页可见）

> Segment 资产是 StreetForward 的一级初始化资产，不从属于 scene 目录层级；  
> scene 资产负责索引，segment 资产负责初始化，运行时 cache 只负责加速，不负责定义初始化事实。

