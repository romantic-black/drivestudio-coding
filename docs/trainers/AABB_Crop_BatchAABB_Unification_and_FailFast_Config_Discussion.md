# AABB / Crop 统一方案讨论（不考虑向后兼容）

参考：
- `docs/dataloader/MultiSceneDataset_Usage.md`
- `docs/trainers/PointCloud_AABB_Cropping_Counterintuitive_Check.md`
- `docs/pointcloud_generators/PointCloud_Generators.md`

本讨论面向 **StreetForward + MultiSceneDataset + pointcloud_generators** 路径，目标是减少“隐式默认值”和“多处真相”，让配置不完整时 **尽早报错**，并提供一套可复现的验证流程来定位 `seg0` 坐标轴语义误解导致的“看起来超界”问题。

---

## 背景：当前已有的关键约定（简述）

### 1) `batch['aabb']` 的角色

`batch['aabb']` 的设计意图是：给 Trainer/渲染侧一个 **与当前 batch 同坐标系（segment 第一帧坐标系，seg0）** 的场景框。

### 2) 点云裁剪的实际控制项

点云 background 的 AABB 裁剪与分层过滤由生成器执行，关键是两类 AABB：
- **`crop_aabb`**：决定“移除框外点”的硬裁剪范围
- **`input_aabb`**：用于 inside/outside 分流和不同强度过滤；**不是最终保留范围**

（注：`use_bbx` 开关已移除，硬裁剪与分层过滤始终启用。）

因此，“点云看起来没按 `batch['aabb']` 裁剪”并不一定是 bug：更常见是 **`batch['aabb'] != crop_aabb`**（多处来源不一致）或坐标轴语义误解。

### 3) 当前系统的主要风险：隐式默认值 + 多处真相

现状里存在多个回退默认值（示例含义：即使配置不写，也会默默使用某个 AABB）：
- dataset 侧在创建 pointcloud generator 时对 `crop_aabb/input_aabb` 提供默认回退；
- batch 侧在构造 `batch['aabb']` 时也会在某些分支回退到默认 `crop_aabb`；
- 配置侧同时出现 `dataset.fixed_segment_aabb`、`dataset.pointcloud.crop_aabb`、`model.bbx_min/max` 等多个“看起来像 AABB”的字段。

这会导致排查困难：你以为在用 A，但系统实际上在用 B，并且没有报错。

---

## 方案 1：统一 `crop_aabb` 与 `batch['aabb']`（唯一事实来源）

目标：让 **点云的硬裁剪范围** 与 **Trainer/渲染侧场景框** 必然一致；并且系统里只存在一个“权威 AABB”。

### 推荐决策（不考虑兼容性）

- **唯一权威字段**：`dataset.segment_aabb`（名字可讨论），语义为：
  - 坐标系：**seg0**
  - 格式：`[[x_min, y_min, z_min], [x_max, y_max, z_max]]`
  - 用途：
    - 作为 pointcloud generator 的 `crop_aabb`
    - 作为 `batch['aabb']`
    - 作为 segment splitting 的 reference length（若仍需要）

- **去除/禁止多源输入**：
  - 删除 `dataset.fixed_segment_aabb`（或至少不再对外暴露/不再被读取）
  - `dataset.pointcloud.crop_aabb` 不再单独存在，改为引用同一个 `dataset.segment_aabb`（配置层面可以做 YAML anchor/变量替代，但核心要求是“运行期只有一个来源”）

### 强制一致性规则（即使保留多个字段，也必须 fail-fast）

如果短期无法完全删字段，至少做到：
- 若同时提供了 `fixed_segment_aabb` 与 `pointcloud.crop_aabb`（或 `model.bbx_min/max`），则 **必须完全一致**；否则初始化直接 `raise ValueError`，报错信息明确指出哪两个字段冲突、差在哪个维度。
- `batch['aabb']` 的构造逻辑只允许来自：
  - `dataset.segment_aabb`（唯一来源）
  - 或 generator 的 `get_crop_aabb()`（但此时也要求与 `dataset.segment_aabb` 一致；否则报错）

### 为什么建议“统一到 crop_aabb”

因为 `crop_aabb` 才是决定点云“硬范围”的真实边界；把 `batch['aabb']` 绑定到它，可以避免“点云按 B 裁了，但 trainer 用 A 渲染/采样”的隐藏不一致。

---

## 方案 2：`use_bbx` 的处理结论

`use_bbx` 开关已从实现与配置中移除：点云生成器 **始终执行** `crop_aabb` 的硬裁剪与 `input_aabb` 的分层过滤。

---

## 方案 3：删除 `MultiSceneDataset` 与点云相关的“默认配置/回退”，保证及时报错（fail-fast）

目标：配置缺字段时，不要悄悄使用 `[-20,-20,-20]~[20,4.8,70]` 之类的默认值；而是 **立刻报错**，并告诉用户缺了什么。

### 需要移除/收紧的默认点（建议清单）

#### A) Dataset 创建 pointcloud generator 时的默认回退

当前常见回退点包括：
- `pointcloud_config.get("type", "monocular")`
- `pointcloud_config.get("crop_aabb", [[-20,...],[20,...]])`
- `pointcloud_config.get("input_aabb", [[-20,...],[20,...]])`
- 以及 `chosen_cam_ids` 默认从 `pixel_source.cameras` 推断

建议（不考虑兼容性）：
- `type/crop_aabb/input_aabb` **必须显式提供**，缺一即报错。（`use_bbx` 已移除）
- 只有“可安全推断且不改变几何定义”的字段才允许推断（例如 `chosen_cam_ids` 可从 `pixel_source.cameras` 推断，但最好也能显式写以提升可读性）。

#### B) `batch['aabb']` 构造时的默认回退

建议：
- `batch['aabb']` 必须来自 **唯一权威字段**（方案 1 的 `dataset.segment_aabb`）。
- 禁止在 `batch['aabb']` 构造路径中“再去读 pointcloud_config 并回退默认值”。一旦 `segment_aabb` 不存在/不合法，应直接报错（并提示配置路径）。

#### C) YAML 配置模板里的“点云默认块”

建议保留“示例值”作为文档/模板，但要做到：
- 训练入口在解析配置后会校验这些字段存在且合法；
- 不要让 Python 代码在运行时再补默认值（否则模板是否写出字段不重要，仍会掩盖问题）。

---

## 方案 4：验证 `seg0` 坐标轴语义误解导致的“超界”问题（对应反直觉点）

针对 `PointCloud_AABB_Cropping_Counterintuitive_Check.md` 中提到的现象：你用“语义方向”(例如 y=up/down) 去解释裁剪，却发现点云似乎不在范围内。

核心事实：代码裁剪是对 **seg0 系下的数值 xyz** 做逐维 min/max 过滤，不会“按语义自动换轴”。因此必须验证你的 `segment_first_pose`（或 `*_to_world`）的轴定义是否与文档语义一致。

### 验证目标

同时验证两件事：
- **(A) 裁剪是否真的生效**：background 点云的数值范围是否落在 `crop_aabb` 内（允许极少量数值误差，但不应系统性超界）。
- **(B) 轴语义是否一致**：seg0 的 x/y/z 是否分别对应你认为的 left/right、up/down、back/forward。

### 推荐验证步骤（不改代码版本）

#### Step 1：打印“最终生效”的三元组

在生成一个 batch 后，打印并对比：
- `batch['aabb']`
- generator 侧的 `crop_aabb`（如果能拿到 `get_crop_aabb()`）

判定：
- 若 `batch['aabb'] != crop_aabb`：先解决“多源不一致”，不要继续看坐标轴。

#### Step 2：用 seg0 数值范围自证裁剪（避免语义误判）

对 `batch['pointcloud']['background'][:, :3]`（seg0 数值 xyz）做：
- `min_xyz = xyz.min(axis=0)`
- `max_xyz = xyz.max(axis=0)`
并直接与 `crop_aabb` 的 min/max 做逐维比较。

判定：
- 若 min/max 明显越界很多：裁剪没生效（优先回到 Step 1 排查 AABB 来源）。
- 若 min/max 落在 `crop_aabb` 内：裁剪生效；接下来问题多半是“你在用错误的语义轴解释”。

#### Step 3：验证 seg0 坐标轴方向（语义对齐）

做一个“基向量检查”（无需可视化也能定位）：
- 在 seg0 系里取三个单位向量：\([1,0,0],[0,1,0],[0,0,1]\)
- 通过 `seg0_to_world`（即 `segment_first_pose`）映射回 world，观察它们在 world 坐标里的方向是否符合数据源约定（例如 nuScenes 常见约定与文档不同）。

你要回答的问题不是“文档写的对不对”，而是：
- **当前数据源的 `*_to_world` 到底采用了什么轴定义？**
- **segment_first_pose_source**（例如来自 lidar pose 还是 camera pose）是否改变了你对轴语义的期待？

#### Step 4：最直观的可视化（建议）

如果你有 Open3D / matplotlib 可视化工具，建议画：
- background 点云（seg0）
- seg0 原点和三根轴（用不同颜色画三条线段）
- crop_aabb 的 3D 盒子

你会立刻看到“哪根轴是 forward/哪个是 up”，从而解释为什么你以为的 y=up 实际上是别的维度。

### 常见结论模式（帮助你快速归因）

- **模式 1：`batch['aabb']` 与 `crop_aabb` 不一致**  
  现象：你拿 batch['aabb'] 去检查点云，必然会感觉“没裁剪”；但点云其实按 crop_aabb 裁了。

- **模式 2：裁剪生效但语义轴错位**  
  现象：用“语义 y=up/down”去看会超界；但用数值 xyz 对照 crop_aabb 完全正常。  
  处理：以 Step 3/4 确认轴方向，修正文档/数据源对齐或修正你选择的 `segment_first_pose_source`。

---

## 建议的最终收敛（可作为实现 TODO）

- **只保留一个段级 AABB 定义**：`dataset.segment_aabb`（seg0 系），并强制用于：
  - pointcloud generator `crop_aabb`
  - `batch['aabb']`
  - model `bbx_min/max`（若模型侧仍需要）
- **fail-fast**：删除 dataset/生成器侧所有 AABB 的默认回退；缺字段立即报错。
- **use_bbx 已移除**：硬裁剪与分层过滤始终启用。
- **提供一套“数值范围 + 轴向检查 + 可视化”的标准验证流程**，将“语义误解”与“裁剪未生效”彻底区分。

