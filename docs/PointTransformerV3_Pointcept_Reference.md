# Point Transformer V3 流程与实现参考（Pointcept）

本文用于后续 AI/工程实现快速检索，聚焦：

- Point Cloud Serialization（论文 4.1）
- Serialized Attention（论文 4.2）
- xCPE（论文 4.2）
- Pointcept 代码中的关键数据结构、模块职责、来源映射

---

## 1. 总体流程（从输入到输出）

以 `PT-v3m1` 为主线（`third_party/Pointcept/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`）：

1. 构建 `Point(data_dict)` 容器（统一管理坐标、特征、batch、offset、序列化信息等）。
2. 调用 `point.serialization(order=..., shuffle_orders=...)`，生成多种序列化编码与索引映射。
3. 调用 `point.sparsify()` 生成 `spconv.SparseConvTensor` 供稀疏卷积与 CPE 使用。
4. `Embedding` 提取初始特征。
5. 进入 Encoder 各 stage：
   - `SerializedPooling`（下采样，重建序列化状态）
   - `Block`（xCPE + Serialized Attention + MLP）
6. 若非 `enc_mode`，进入 Decoder：
   - `SerializedUnpooling`
   - `Block`
7. 返回 `Point`（其 `feat` 为主输出特征）。

论文来源：`Point Transformer V3...` 第 4 节（4.1/4.2/4.3）。

---

## 2. 核心数据结构：`Point`

代码：`third_party/Pointcept/pointcept/models/utils/structure.py`

`Point` 本质是一个字典增强结构，核心键分层如下：

- 输入/基础键
  - `coord`: 原始点坐标
  - `grid_coord`: 体素网格坐标（可由 `coord + grid_size` 推导）
  - `feat`: 点特征
  - `batch` 或 `offset`: batch 组织信息（二者可互相推导）
- Serialization 相关键
  - `serialized_depth`
  - `serialized_code`: 多种序列化 pattern 的编码（shape: `[num_orders, N]`）
  - `serialized_order`: 每种编码对应的排序索引（argsort 后）
  - `serialized_inverse`: 排序逆映射（恢复原点顺序）
- SparseConv 相关键
  - `sparse_shape`
  - `sparse_conv_feat`
- 其他运行态键（注意力/池化临时）
  - `pad`, `unpad`, `cu_seqlens_key`
  - `pooling_parent`, `pooling_inverse` 等

这套结构是 PTv3 “不物理重排点云、仅记录映射”的基础。

---

## 3. Point Cloud Serialization（重点）

### 3.1 论文定义（4.1）

论文将点云序列化定义为：先离散化，再通过 space-filling curve 编码并排序。

核心公式（论文原意）：

`Encode(p, b, g) = (b << k) | phi^{-1}( floor(p / g) )`

- `p`: 点坐标
- `g`: grid size
- `phi^{-1}`: 空间填充曲线反映射到一维序号
- `b`: batch id
- `k`: 位置编码占用位数

论文强调两点：

1. 使用 64 位整数编码，前缀位放 batch，后缀位放空间编码。
2. 实现上不要求真实重排点数组，只需保存映射关系（order / inverse）。

### 3.2 Pointcept 实现落点

1. 序列化入口：  
   `third_party/Pointcept/pointcept/models/utils/structure.py::Point.serialization`
2. 编码函数：  
   `third_party/Pointcept/pointcept/models/utils/serialization/default.py::encode`
3. Z-order 具体位交织实现：  
   `third_party/Pointcept/pointcept/models/utils/serialization/z_order.py`
4. Hilbert 编码实现：  
   `third_party/Pointcept/pointcept/models/utils/serialization/hilbert.py`

实现要点：

- 支持 pattern：`z`, `z-trans`, `hilbert`, `hilbert-trans`
- `z-trans` / `hilbert-trans` 通过交换坐标轴顺序（`[1, 0, 2]`）得到变体
- 若有 batch，则 `code = (batch << (depth*3)) | code`
- 多 pattern 同时编码后 `stack` 成 `[K, N]`
- `order = argsort(code)`，`inverse = scatter(...)` 构建逆映射
- 可选 `shuffle_orders=True`：对 pattern 维度随机置乱，形成 Shuffle Order

### 3.3 PTv3 中的作用

Serialization 为后续模块提供“可索引邻域”：

- 注意力阶段用 `serialized_order[order_index]` 做 patch grouping
- pooling 阶段对 `serialized_code` 右移实现尺度下采样（`>> pooling_depth * 3`）
- 多 block 通过 `order_index = i % len(self.order)` 轮换 pattern（Shift Order）

---

## 4. Serialized Attention（重点，对应论文 4.2）

代码主入口：  
`third_party/Pointcept/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py::SerializedAttention`

### 4.1 Patch Grouping：重排 + 补齐

与论文图 4 对应，代码通过一个索引流程完成“重排 + padding”：

1. `get_padding_and_inverse()`  
   - 基于 `offset` 统计每个样本点数  
   - 补齐到 `patch_size` 的整数倍（仅对超出 patch 的样本补齐）
   - 生成：
     - `pad`: 重排后可直接索引补齐序列
     - `unpad`: 从补齐序列回到原始长度
     - `cu_seqlens_key`: FlashAttention varlen 所需前缀和
2. `order = serialized_order[order_index][pad]`
3. `qkv = Linear(feat)[order]` 后 reshape 成 patch 结构

这正是论文中 “reordering + patch padding can be merged into one indexing op” 的工程实现。

### 4.2 Attention 计算路径

- 非 Flash 路径：
  - `q,k,v` reshape
  - `attn = (q * scale) @ k^T`
  - 可选叠加 `RPE`
  - softmax + dropout + `attn @ v`
- Flash 路径：
  - 调用 `flash_attn_varlen_qkvpacked_func`
  - 使用 `cu_seqlens` 和 `max_seqlen=patch_size`
  - 最后用 `inverse` 还原点顺序

论文层面，PTv3主张使用 dot-product attention + window/patch 思路；代码对应为 patch 内标准注意力，并优先走 FlashAttention 以获得速度和显存优势。

### 4.3 Patch Interaction：Shift Order / Shuffle Order

论文 4.2 提到多种 patch interaction，PTv3 主推 Shift Order + Shuffle Order。

在 Pointcept 的落地方式：

- Shift Order：
  - 在网络构建时，block 的 `order_index=i % len(self.order)`，相邻 block 轮换不同序列化 pattern
- Shuffle Order：
  - `Point.serialization(..., shuffle_orders=True)` 对 pattern 维随机打乱
  - `SerializedPooling` 内也有 `shuffle_orders`，下采样后继续打乱 pattern 序列

---

## 5. xCPE（重点）

论文 4.2：xCPE = 在 attention 前“前置一个带残差的稀疏卷积 positional 编码层”。

代码对应（`Block`）：

- `self.cpe = [SubMConv3d(kernel=3) -> Linear -> Norm]`
- forward 中先做：
  - `shortcut = point.feat`
  - `point = self.cpe(point)`
  - `point.feat = shortcut + point.feat`
- 然后才进入 `norm1 + SerializedAttention`

这与论文描述一致：不是把位置编码塞进注意力权重（RPE），而是改为 attention 之前的条件位置增强层，从而兼顾效果与效率。

---

## 6. 关键模块与职责速查

- `pointcept/models/utils/structure.py`
  - `Point` 容器，`serialization()`，`sparsify()`
- `pointcept/models/utils/serialization/default.py`
  - 序列化编码入口（z/hilbert/trans）
- `pointcept/models/utils/serialization/z_order.py`
  - Morton code（bit interleave）实现
- `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`
  - `SerializedAttention`, `Block(xCPE)`, `SerializedPooling/Unpooling`, `PointTransformerV3`
- `pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py`
  - 同系列实现变体（保留同类核心思想，结构细节略不同）

---

## 7. 论文设计与当前默认实现的对齐/差异

### 对齐点

- serialization + patch attention 主范式一致
- Shift Order / Shuffle Order 均有工程实现
- xCPE（attention 前置 sparse conv + residual）一致
- 支持 FlashAttention 的高效注意力路径

### 常见差异（读代码时要注意）

- 论文实验常强调 `Z + TZ + H + TH` 混合 pattern；代码默认参数可能仅 `("z", "z-trans")`
- 默认通常 `enable_rpe=False`（符合论文“弱化 RPE，采用 xCPE”的方向）
- 不同 PTv3 mode（m1/m2/m3）在 embedding、pooling、norm 细节上略有变体

---

## 8. 给后续 AI 的最小检索路径（建议）

当需要快速回答“PTv3 某机制怎么实现”时，按以下顺序查：

1. `point_transformer_v3m1_base.py`（主干行为）
2. `structure.py`（序列化与数据结构）
3. `serialization/default.py` + `z_order.py`（编码细节）
4. 论文第 4.1 / 4.2 节（概念定义与设计动机）

---

## 9. 来源说明

- 论文：`third_party/Wu et al. - 2024 - Point Transformer V3 Simpler, Faster, Stronger.pdf`
  - 主要使用第 4.1（Point Cloud Serialization）、4.2（Serialized Attention, xCPE）、4.3（网络细节）及相关表格描述
- 代码（Pointcept）：
  - `third_party/Pointcept/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`
  - `third_party/Pointcept/pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py`
  - `third_party/Pointcept/pointcept/models/utils/structure.py`
  - `third_party/Pointcept/pointcept/models/utils/serialization/default.py`
  - `third_party/Pointcept/pointcept/models/utils/serialization/z_order.py`

