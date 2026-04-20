# StreetForward Stage5_0 与 Point Transformer V3：模型结构、实现与超参对比

本文对比：

- **StreetForward Stage5_0** 中与结构解码相关的配置：`configs/minimal_streetforward_stage5_0_multi_scene_v7.yaml` 中 `model` 段（含 `struct_decoder`，约 L128–183）及同文件内与之强相关的 `model` 字段。
- **Point Transformer V3（PTv3）** 在 Pointcept 中的参考实现：`third_party/Pointcept/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`（`PT-v3m1`），以及论文中的典型超参（附录 Table 15 等）。

更细的 PTv3 代码与论文对应关系见：`docs/PointTransformerV3_Pointcept_Reference.md`。

---

## 1. 一句话定位

| 维度 | Stage5_0（本配置） | PTv3（Pointcept `PT-v3m1`） |
|------|-------------------|-----------------------------|
| 角色 | 在 **StreetForward** 管线内，仅对 **`bg + rigid.S_in`** 做点 **保持点数** 的 **xCPE 结构注入**，输出维度对齐 **GRU 输入**（`fused_in_dim`）。 | 完整的 **3D 点云骨干网络**：**serialization → sparsify → U-Net 式 encoder/decoder**，块内为 **xCPE + Serialized Attention + MLP**。 |
| 范围 | 局部模块；**无**全局 scene backbone 替换。 | 全局 backbone；可单独作分割/检测等任务的 encoder。 |

---

## 2. 整体结构对比

### 2.1 PTv3 完整链路（Pointcept）

典型前向（概念级）：

```text
Point(data_dict)
  -> serialization（space-filling curve 编码 + order/inverse）
  -> sparsify（SparseConvTensor）
  -> Embedding（如 stem SubMConv3d）
  -> Encoder：SerializedPooling（下采样） + Block × N
  -> Decoder：SerializedUnpooling + Block × N（非 enc_mode 时）
```

每个 **Block** 内含：

- **xCPE**：`SubMConv3d` → `Linear` → `Norm`（与残差相加，再接 attention）。
- **Serialized Attention**：按序列化顺序 **patch 分组 + padding**，再做 dot-product / FlashAttention。
- **MLP**：pre-norm 结构下的 FFN。

### 2.2 Stage5_0 `struct_decoder`（本 YAML）

本配置中 `struct_decoder` **仅对应 PTv3 中「xCPE 风格的局部稀疏卷积注入」这一层思想**，**不包含**：

- Point Cloud Serialization（z-order / Hilbert 等）
- Serialized Attention
- SerializedPooling / SerializedUnpooling
- 完整 U-Net 多 stage

实现落点：`models/streetforward/struct_decoders/xcpe_decoder.py` 中的 `StreetForwardXCPEDecoder`（token 构建 + 体素聚合 + 多层 xCPE 残差 + 输出投影到 `fused_in_dim`）。

与 PTv3 块内 xCPE 的相似点：

- 使用 **spconv** 的 **SubMConv3d**（`sparse_backend: "spconv"`）在稀疏体素上做 3×3×3 邻域混合。
- 后接 **Linear + LayerNorm + GELU**，并以 **小初始残差 scale**（`residual_scale_init`）稳定训练。

差异点：

- StreetForward 的 token 来自 **2D 回投特征 + support + branch + Gaussian 参数 embedding**（任务相关），而非 PTv3 的 stem 或单一 RGB/坐标嵌入。
- **显式 `branch_id`（bg vs rigid-in）** 与 **support 阈值分分支**，避免 PTv3 中不需要的跨类语义。

---

## 3. 配置项与 PTv3 超参对照表

下列 **Stage5_0** 列来自 `minimal_streetforward_stage5_0_multi_scene_v7.yaml` 的 `model` / `struct_decoder`；**PTv3** 列来自 Pointcept 默认构造参数及论文附录 Table 15（论文实验常用大 patch，与 Pointcept 默认 `patch_size=48` 不同）。

| 概念 | Stage5_0（本配置） | PTv3（Pointcept 默认 / 论文 Table 15） |
|------|-------------------|----------------------------------------|
| **整体 stage** | `model.stage: "5_0"`；主任务仍为 StreetForward（routed rigid、GRU、渲染等）。 | 无此字段；即纯 3D backbone。 |
| **体素 / 网格** | `struct_decoder.voxel_size: 0.20`；与顶层 `model.voxel_size: 0.2` 一致量级，用于 **struct 体素化**。 | 点云侧由 **grid_size / grid_coord** 与 **serialization depth** 决定；**无**与 StreetForward 相同的 `voxel_size` 键名。 |
| **通道宽度** | `struct_decoder.channels: 64`（token 与 xCPE 隐层宽度）。 | Encoder 逐级 `enc_channels=(32,64,128,256,512)` 等；**不是**单一 64 标量。 |
| **输出维度** | `output_dim: auto` → 对齐 **`fused_in_dim`**（与 `sparseConv_outdim`、`feat_2d_channels` 等共同决定的 GRU 输入维）。 | 输出为 **任务头** 定义（如分割类别数），不在 PTv3 backbone 类上固定为 `fused_in_dim`。 |
| **2D 特征维** | `feat_2d_channels: 32`（与 `model.feat_2d_channels` 对齐）。 | PTv3 **无** 2D 回投分支；输入为点特征 `in_channels`（默认 6 等）。 |
| **参数嵌入** | `struct_decoder.param_embed_dim: 32`（token 内参数支路）。 | PTv3 Block 使用 **param_embed** 的是另一套设计（点侧为 `LayerNorm` + GRU 侧为 StreetForward 专有）；PTv3 用 **PDNorm/BN** 等于 **另一套** 归一化策略。 |
| **xCPE 层数** | `xcpe.num_layers: 2`。 | 每个 **Block** 内 **1 个** CPE/xCPE；总层数 = **各 stage 的 block 数之和**（如 `enc_depths` 之和远大于 2）。 |
| **卷积核** | `kernel_size: 3`（SubMConv3d）。 | 同为 **3**（`Block` 内 `SubMConv3d(..., kernel_size=3)`）。 |
| **残差缩放初值** | `residual_scale_init: 1e-3`（StreetForward xCPE 显式参数）。 | Pointcept `Block` 为 **直接相加**（无单独 `res_scale` 参数）；行为不同。 |
| **Norm / Act** | `norm: layernorm`，`act: gelu`。 | Block 内 CPE 后为 **Linear + LayerNorm**；激活在 MLP 用 **GELU**。 |
| **Serialization** | `future.allow_serialized_attention: false`；当前 **不用** 序列化注意力。 | **核心组件**：`order=("z","z-trans")` 等 + `SerializedAttention`。 |
| **Pooling** | `future.allow_pooling: false`；struct 路径 **点数不变**。 | **SerializedPooling** 多 stage 下采样；**必须** 改变分辨率/点数（除非自定义 enc_mode）。 |
| **Patch / 注意力** | 无 `patch_size` / `num_heads`（struct 内无 attention）。 | `enc_patch_size` 各 stage 默认 **48**；论文 Table 15 示例 **1024**；`enc_num_head` 随 stage 增大。 |
| **RPE** | 不涉及。 | `enable_rpe: False` 为默认；可与 FlashAttention 互斥。 |
| **Flash Attention** | 不涉及。 | `enable_flash: True` 时常用 **varlen** 路径。 |
| **Shuffle / 多 order** | 无。 | `shuffle_orders=True`；多 serialization pattern 轮换。 |

---

## 4. 同文件 `model` 中与 PTv3 **无关** 的 StreetForward 专有项

以下字段存在于 `minimal_streetforward_stage5_0_multi_scene_v7.yaml` 的 `model` 中，**不属于 PTv3 模型定义**，仅用于 StreetForward 训练动力学与分支语义：

- `scale_init_value`、`sh_degree`、`sparseConv_outdim`、`feat_2d_*`、`use_fused_cuda_backproject_*`
- `param_embed_dim`、`offset_gru_hidden_dim`、`offset_gru_use_reset_gate`（GRU 与偏移预测）
- `rigid_routed`（source-frame world + segment_aabb 路由）
- `branches`（bg / distant / rigid 的 MLP、limits、eta 等）

PTv3 论文与代码 **不包含**「多相机 2D 回投 + 高斯 GRU + 光度损失」这条链路。

---

## 5. 实现来源对照（便于代码跳转）

| 组件 | StreetForward Stage5_0 | PTv3（Pointcept） |
|------|------------------------|-------------------|
| xCPE 风格稀疏卷积 | `StreetForwardXCPEDecoder`、`struct_decoders/xcpe_decoder.py` | `Block.cpe`：`SubMConv3d` + `Linear` + `norm`，`point_transformer_v3m1_base.py` |
| 序列化 / 排序 | 未启用（配置显式关闭 future attention） | `Point.serialization`、`encode`（`utils/serialization`） |
| 注意力 | 无 | `SerializedAttention` |
| 体素/稀疏张量 | 自建 `VoxelLayout` + `scatter_mean` + spconv | `Point.sparsify` + `SparseConvTensor` |

---

## 6. 小结

- **Stage5_0 本配置**在 `struct_decoder` 上实现的是 **「仅 xCPE 思想的、轻量、点保持」** 结构模块，用于把 **bg 与 rigid-in** 的联合点集上的 **任务相关 token** 做局部 3D 混合后，**喂给现有 GRU**，与 PTv3 **完整 backbone（serialization + 多 stage pooling + serialized attention + U-Net）** 在 **范围与复杂度**上不在同一量级。
- **超参**上：Stage5_0 用 **固定 `channels=64`、2 层 xCPE、`voxel_size=0.2`**；PTv3 用 **多 stage 通道金字塔、大 patch attention、多种 serialization order**，二者 **不可逐键一一对应**；仅 **SubMConv3d + kernel=3 + LayerNorm + GELU** 可作为 **概念对齐** 的锚点。

若后续 Stage5_1 引入 **serialized attention** 且仍 `no_pooling`，则与 PTv3 的 **SerializedAttention** 模块会更接近，但仍需单独对齐 **patch_size、order、FlashAttention、batch/offset** 等语义。
