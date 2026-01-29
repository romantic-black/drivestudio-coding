# Golden Baseline 用「批量缓存」方案

本文档给出：在 MultiSceneDataset 读取较慢的前提下，如何通过**少量、结构化随机的 batch** 直接供给 Golden Baseline 的录制与回归，以及「仅提供 batch」时需补足的点与可行实现。

---

## 1. 背景与目标

### 1.1 问题

- `MultiSceneDataset` 初始化与调度（`create_scheduler` + `next_batch`）涉及多场景加载、预加载线程、段遍历等，整体偏重、偏慢。
- Golden Baseline 的录制/回归只需要**有限步数**的 `train_iter`（如 8 步），不需要跑完所有场景或大量 segment。
- 希望：**只使用「少量随机 batch」**，且随机有结构 —— 「几个场景 → 每场景几个段 → 每段几个 batch」，与 [MultiSceneDataset_Usage.md](./MultiSceneDataset_Usage.md) 中的层次（Scene → Segment → Keyframe → Frame）一致。

### 1.2 目标

- 定义一种**批量计划（batch plan）**：限定参与的场景、每场景的段、每段取几个 batch。
- 在**固定种子**下，按该计划得到确定的一串 batch，并可：
  - **一次性写入缓存**（batch cache），之后录制/回归**只读缓存**，不再建 dataset、不调 scheduler；
  - 或作为「最小化数据集调用」的规范：只对计划内的 (scene, segment) 调 `get_segment_batch`，每段调用次数=该段 batch 数。
- 评估「**仅提供 batch**」是否足够，以及需要哪些**补充**（见 §4）。

---

## 2. 数据结构与层次（对齐 MultiSceneDataset）

与 [MultiSceneDataset_Usage.md](./MultiSceneDataset_Usage.md) 一致：

```
场景 (Scene)  →  scene_id，对应一个场景目录
  └── 段 (Segment)  →  segment_id，段在场景内索引
        └── 关键帧 (Keyframe)  →  段内按轨迹/距离切出的子段
              └── 每次 get_segment_batch 在该段内「随机」选 source/target 关键帧与帧
```

- **一个 batch**：来自一次 `get_segment_batch(scene_id, segment_id)`，包含该 (scene, segment) 下某次随机抽取的 source/target 关键帧与帧、点云、dynamic_info 等。
- **同一 (scene, segment) 多次调用**：每次会得到**不同**的 batch（因 `_select_source_and_target_keyframes` / `_select_frame_from_keyframe` 使用 `random.sample` / `random.choice`）。  
  因此「一个段内选几个」= 对该段调用 `get_segment_batch` **K 次**，得到 K 个 batch；K 由「批量计划」指定，随机性由**调用前**的全局种子控制，即可复现。

---

## 3. 批量计划（Batch Plan）设计

### 3.1 形式化

定义**批量计划**为一份配置，指定「哪些场景、每场景哪些段、每段取多少个 batch」：

```python
# 示例：2 场景 × 每场景 2 段 × 每段 2 batch → 共 8 个 batch
batch_plan = {
    "scene_ids": [0, 1],                    # 或从 cfg.data.train_scene_ids 取前 2 个
    "segments_per_scene": 2,                # 每场景取前 2 个 segment（或显式指定 segment_ids 见下）
    "batches_per_segment": 2,
    "segment_ids": None,                    # 若为 None：每场景用 segments[:segments_per_scene]
                                           # 若为 list of list：[[0,1], [0,1]] 表示 scene0 用 segment 0,1；scene1 用 0,1
}
```

更灵活时，可显式写出「(scene_id, segment_id) × 该组合的 batch 数」：

```python
# 显式版：(scene_id, segment_id, num_batches)
plan_tuples = [
    (0, 0, 2),
    (0, 1, 2),
    (1, 0, 2),
    (1, 1, 2),
]
# 顺序即录制/回归时的 batch 顺序；总步数 = sum(num_batches)=8
```

### 3.2 与「随机」的关系

- **不是完全随机**：参与的场景、段由计划固定；只有「每段内抽哪几组 keyframe/frame」是随机的，由 `get_segment_batch` 内部用 `random` 完成。
- **可复现**：在**真正调用**任一 `get_segment_batch` 之前调用 `set_deterministic_seed(seed)`，则同一计划、同一种子得到同一批 batch。
- 建议：Golden Baseline 的 meta 里记录 `batch_plan`（或等价的 `scene_segment_sequence` + `batches_per_segment` 的等价信息），以便回归时按相同计划复现或按相同缓存校验。

### 3.3 和当前 scheduler 的对应关系

当前 scheduler 逻辑等价于：

- 按 `scene_order` / `segment_order` 得到一条 **(scene_id, segment_id)** 遍历顺序；
- 对每个 (scene_id, segment_id) 调用 `get_segment_batch` 共 `batches_per_segment` 次。

「批量计划」可看作对这条遍历的**截断与子集**：只保留计划内的 (scene, segment)，且每段只取计划内的 batch 数。若 `scene_order=segment_order=sequential` 且 `scene_ids=[0,1]`、每场景前 2 段、每段 2 batch，则与「sequential scheduler，2 场景 × 2 段 × 2 batch」行为一致（但仅限这 8 步，不会继续往后扫）。

---

## 4. 「仅提供 batch」是否足够及补充

### 4.1 训练侧：足够

- `StreetForwardTrainer.train_iter(batch, ...)` 的输入是**单份 batch 字典**（已含 `scene_id`、`segment_id`、`pointcloud`、`dynamic_info`、`source`/`target` 等）。
- 只要每步喂进去的 batch 格式与 `get_segment_batch` 产出、与 `convert_batch_to_streetforward_format` 的输入一致，**不需要** dataset 或 scheduler 实例。  
⇒ **仅提供一串 batch（list of batch dict）即可跑完 N 步 train_iter**，无需再读 MultiSceneDataset。

### 4.2 录制/回归侧：需要补足的内容

| 需求 | 仅 batch 时 | 补充办法 |
|------|-------------|----------|
| **(scene_id, segment_id) 序列** | 每个 batch 里已有 `scene_id`、`segment_id` | 从 batch 序列里按顺序读出即可，或由「批量计划」推导并在 meta 中保存 |
| **每步 loss / NodeState 摘要** | 只依赖本步的 `batch` + `trainer.train_iter` 输出 | 无额外依赖，现有 `record_step` 即可 |
| **确定性复现** | 若直接给「预先保存的 batch 文件」做回放，随机性仅来自 trainer 内部；只要种子一致即可 | 回放时：`set_deterministic_seed(meta["seed"])`，然后按序加载并执行 train_iter |
| **如何得到这一串 batch** | 若不做缓存，仍要建 dataset、对计划内 (scene, segment) 各调 K 次 `get_segment_batch`，慢点仍在 dataset 端 | **补充：batch 缓存** —— 用 dataset 按计划+种子「收割」一次，把 N 个 batch 存盘；之后录制/回归只读缓存，不再建 dataset |

### 4.3 可行补充方案概览

1. **定义 batch 计划格式**  
   在配置或 baseline meta 中支持 `batch_plan`（或等价的 scene_ids + segments_per_scene + batches_per_segment），便于复现和与现有「scene_segment_sequence」对照。

2. **Batch 缓存（推荐）**  
   - **一次收割**：根据计划 + 固定种子，建一次 dataset，只对计划内的 (scene, segment) 调用 `get_segment_batch`，每段调用 K 次，得到 N 个 batch；序列化到磁盘（如 `batches.pt` 或按步分文件）。  
   - **元数据**：同一目录或同一文件中记录 `batch_plan`、`seed`、`config_path`（或 config 指纹）、以及可选的 `scene_segment_sequence`（与现有 baseline meta 一致）。  
   - **录制/回归**：  
     - 若存在对应缓存且未过期，则**不建 dataset、不建 scheduler**，直接从缓存加载 `batches: List[Dict]`，按序执行 `train_iter` 并做 `record_step` / `compare_step`。  
     - 若不存在缓存（或显式禁用缓存），则退回到「建 dataset + 按计划最小化调用」的路径（见下）。

3. **最小化 dataset 路径（不做缓存时）**  
   - 建 dataset 时使用 `preload_scene_count=1`（或尽量小），仅对计划内的 scene 触发加载。  
   - 按照 `plan_tuples` 或等价的 (scene_id, segment_id) 顺序，**只**调 `get_segment_batch(scene_id, segment_id)`，每对 (scene, segment) 调用次数=计划中该段的 batch 数。  
   - 不在该路径使用 scheduler，避免预加载、队列和线程；仅在「没有缓存且必须现场生成 batch」时使用，仍然比「完整 scheduler 扫全场景」轻量。

4. **与现有 baseline 的兼容**  
   - 现有 baseline meta 中有 `scene_segment_sequence`、`scheduler_kwargs` 等。  
   - 使用 batch 缓存时，**meta 仍应写出**与当前一致的 `scene_segment_sequence`（可从 batch 列表或计划推导），这样现有 `compare_step`、`run_recording` 的对接方式不必大改，仅把「batch 来源」从 scheduler 改为「从缓存迭代」。

---

## 5. 实现要点（建议顺序）

### 5.1 批量计划与默认值

- 在 `utils/streetforward_baseline.py`（或单独模块）中增加：
  - `batch_plan_from_config(cfg, max_scenes=2, segments_per_scene=2, batches_per_segment=2) -> List[Tuple[int,int,int]]`
  - 返回 `[(scene_id, segment_id, num_batches), ...]`，保证总数为 2×2×2=8 或由参数控制，且只使用 cfg 里已有的 `train_scene_ids` 与 dataset 返回的 segments。
- 若已有「只读 config」的轻量方式能拿到「某场景有多少 segment」（而不必建完整 dataset），可在此用；否则可约定「先建一次 dataset 只为了取 plan」，或由调用方传入 `plan_tuples`。

### 5.2 Batch 缓存的写与读

- **写（收割）**  
  - 输入：config_path、seed、batch_plan（或等价的 max_scenes/segments_per_segment/batches_per_segment）、device、include_test。  
  - 流程：`set_deterministic_seed(seed)` → 建 dataset → 按 plan 顺序，对每个 (scene_id, segment_id) 调 `get_segment_batch` 共 num_batches 次，把每次返回值（或可序列化后的形态）追加到列表；最后把 `batches` 与 `meta_plan` 写入缓存目录。  
  - 可做成独立脚本 `tools/harvest_batch_cache.py`，或 `record_streetforward_golden_baseline.py --harvest-only --output-cache-dir ...`。

- **读**  
  - 输入：缓存路径（或由 baseline meta 中的 `batch_cache_path` 指向）。  
  - 输出：`batches: List[Dict]`，以及可从缓存 meta 中读出的 `scene_segment_sequence`、`seed` 等。  
  - 序列化格式：若 batch 中含 tensor，用 `torch.save`；若希望跨机器/少量依赖，可转 numpy + 结构体再存。需在文档中写明「缓存的 batch 与 `get_segment_batch` 返回格式一致」，以保证 `convert_batch_to_streetforward_format` 直接可用。

### 5.3 录制/回归改用「batch 源」抽象

- 抽象出一个「batch 迭代器」：
  - 来自 scheduler：`for batch in scheduler_batches(): ...`（内部仍用 `next_batch`，语义不变）。
  - 来自缓存：`for batch in load_batch_cache(cache_path): ...`。
- `run_recording`（及回归里对 `run_recording` 的调用）改为接受 `batch_iter` 或 `batch_source="scheduler"|"cache"` + 缓存路径；当 `batch_source="cache"` 时不再建 scheduler、不建 dataset，仅按缓存顺序喂 batch。  
这样现有「按步 record / compare」逻辑不改，仅数据来源切换。

### 5.4 回归测试与 baseline meta

- 若 baseline 是用「缓存 batch」录制的，meta 里应包含：
  - `batch_cache_path` 或 `batch_cache_rel`（相对项目根或相对 meta 文件），以及可选的 `batch_plan`。
- 回归时：若存在 `batch_cache_*` 且文件可读，则用缓存回放；否则回退到「建 dataset + 按 plan 最小调用」或直接失败并提示「请先运行 harvest 生成缓存」。

---

## 6. 小结

| 问题 | 结论 |
|------|------|
| MultiSceneDataset 读得太慢 | 通过「**批量计划 + 一次性 batch 缓存**」或「**按计划的最小 dataset 调用**」减少热路径上的 dataset 使用。 |
| 「几个随机 batch」的结构 | **几个场景 → 每场景几个段 → 每段几个 batch**，与文档中的 Scene→Segment→Keyframe 层次一致；由 batch plan 定（scene_ids/segments_per_scene/batches_per_segment 或 plan_tuples）。 |
| 仅提供 batch 是否有问题 | **训练与录制/回归在逻辑上仅需 batch 序列**；缺少的是「如何高效得到这串 batch」与「如何保证复现」。 |
| 可行补充 | ① 定义并实现 **batch plan**；② 实现 **batch 缓存**的写（harvest）与读；③ 录制/回归支持 **batch 来源 = 缓存**，并可选退回到「按 plan 最小化调用 dataset」；④ baseline meta 中记录 **batch_cache 路径或 batch_plan**，以兼容现有比对与回归流程。 |

按上述补充后，Golden Baseline 可在「只使用少量、结构化随机 batch」的前提下，既避免在每次录制/回归时重跑整条 MultiSceneDataset  pipeline，又保持与现有设计文档和 meta 结构的兼容。
