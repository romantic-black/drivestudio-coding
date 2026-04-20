# StreetForward Stage5_0 实现方案文档

> 参考：
> - `models/streetforward/minimal_trainer_stage4_6.py`
> - `docs/trainers/StreetForward_Flow_stage4_6.md`
> - `docs/PointTransformerV3_Pointcept_Reference.md`

---

## xCPE Structure Decoder for bg + rigid-in

## 0. 一句话定义

**Stage5_0 = Stage4_6 语义不变，只把 `bg + rigid.S_in` 的轻量 3D feature / per-point decoder 输入，替换为 point-preserving 的 xCPE structure decoder。**

即：

```text
保留：
  Stage4_6 no-sky
  routed rigid
  one-pass fused 2D backprojection
  bg/distant shared heads
  rigid local writeback
  non-sky loss
  proxy backward

替换：
  bg / rigid-in 的结构特征来源
```

设计动机来自 PTv3：优先使用高效结构化点云处理（serialization / sparse-conv-style CPE），减少昂贵 KNN/RPE 依赖；其中 `Block` 的 xCPE（`SubMConv3d -> Linear -> Norm` 残差注入）与本阶段需求高度一致。

---

## 1. 设计边界

### 1.1 Stage5_0 改什么

Stage4_6 当前存在：

```python
feat_3d_bg, feat_3d_rigid_in = self._build_3d_features_bg_plus_rigid_in(...)
```

Stage5_0 改为：

```python
feat_bg_input, feat_rigid_in_input = self._compute_struct_gru_inputs_bg_rigid_in(...)
```

建议该接口**直接输出 GRU input feature**，不再对输出执行一次旧 `_fuse_features(feat_3d, feat_2d)`。

原因：

```text
Stage5_0 token 已经包含 2D feat + support + branch + state embedding。
若再与同一份 2D feat fuse，会重复注入 2D evidence。
```

推荐路径：

```text
feat_2d + acc_w + branch + state params + coords
  -> token projection
  -> xCPE
  -> output projection
  -> bg / rigid-in GRU + shared heads
```

### 1.2 Stage5_0 不改什么

必须保持：

```text
1. rigid routing 仍是 source-frame world + segment_aabb。
2. rigid-in 仍走 bg heads。
3. rigid-out 仍走 distant heads。
4. distant 不进入 xCPE。
5. source 2D 仍 one-pass fused backprojection。
6. loss 仍是 Stage4_5/4_6 的 non-sky photometric。
7. rigid 仍 local NodeState 存储，world/local roundtrip 后写回。
8. Stage5_0 不恢复 rigid-specific heads。
```

---

## 2. 推荐继承关系与文件布局

```python
class MinimalStreetForwardStage5_0(MinimalStreetForwardStage4_6):
    ...
```

建议文件：

```text
models/streetforward/minimal_trainer_stage5_0.py
models/streetforward/struct_decoders/__init__.py
models/streetforward/struct_decoders/token_builders.py
models/streetforward/struct_decoders/xcpe_decoder.py
models/streetforward/struct_decoders/common.py
```

训练入口与配置：

```text
tools/train_minimal_streetforward_stage5_0_multi_scene_v7.py
configs/minimal_streetforward_stage5_0_multi_scene_v7.yaml
```

---

## 3. 为 Stage5_1 预留的抽象接口

避免将 xCPE 写死在 trainer 内，推荐统一输入/输出协议。

```python
@dataclass
class StructDecoderInput:
    feat_2d: torch.Tensor
    acc_w: torch.Tensor
    coords: torch.Tensor
    branch_id: torch.Tensor            # 0=bg, 1=rigid_in
    params_for_embed: Dict[str, torch.Tensor]
    split_bg: int
    split_rigid_in: int
    meta: Dict[str, Any]


@dataclass
class StructDecoderOutput:
    feat: torch.Tensor                 # [N, output_dim], point-preserving
    aux: Dict[str, Any]


class StreetForwardStructDecoderBase(nn.Module):
    def forward(
        self,
        x: StructDecoderInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> StructDecoderOutput:
        raise NotImplementedError
```

Stage5_0：

```python
class StreetForwardXCPEDecoder(StreetForwardStructDecoderBase):
    ...
```

Stage5_1 可替换为：

```python
class StreetForwardSerializedPatchDecoder(StreetForwardStructDecoderBase):
    ...
```

---

## 4. 配置设计（含 fast-fail）

推荐新增：

```yaml
model:
  stage: "5_0"

  struct_decoder:
    enable: true
    type: "xcpe"

    scope: "bg_rigid_in"
    output_role: "gru_input"
    point_preserving: true
    include_distant: false
    include_rigid_out: false

    channels: 96
    output_dim: "auto"                # 必须对齐 trainer 期望的 GRU 输入维（通常 self.fused_in_dim）
    feat_2d_channels: 48
    param_embed_dim: 32
    branch_embed_dim: 8
    support_embed_dim: 8

    voxel_size: 0.20
    clamp_grid_coord: false
    sparse_backend: "spconv"

    xcpe:
      num_layers: 2
      kernel_size: 3
      residual_scale_init: 1.0e-3
      norm: "layernorm"
      act: "gelu"

    token:
      use_2d_feat: true
      use_support: true
      use_branch_embed: true
      use_param_embed: true
      use_anchor_rgb: false
      use_hidden_state: false
      zero_invalid_2d_feat: true

    future:
      allow_serialized_attention: false
      allow_pooling: false
```

Stage5_0 fast-fail（不满足即报错）：

```text
type 必须是 xcpe
scope 必须是 bg_rigid_in
include_distant 必须 false
include_rigid_out 必须 false
point_preserving 必须 true
allow_pooling 必须 false
```

---

## 5. 数据流详细设计

### 5.1 Stage4_6 基线

```text
NodeState bg / distant / rigid
  -> route rigid source points
  -> one-pass source 2D backproject
  -> bg 3D feature + 2D fuse
  -> rigid-in 3D feature + 2D fuse
  -> distant / rigid-out 2D-only
  -> GRU + heads
  -> target render
  -> non-sky loss
```

### 5.2 Stage5_0 替换后

```text
NodeState bg / distant / rigid
  -> route rigid source points
  -> one-pass source 2D backproject
  -> build P_struct = bg_all ∪ rigid.S_in
  -> build struct tokens
  -> xCPE structure decoder
  -> split bg / rigid-in GRU inputs
  -> distant / rigid-out unchanged
  -> GRU + heads
  -> target render
  -> non-sky loss
```

### 5.3 结构点集 P_struct

```python
coords_bg = node_state_bg.means
coords_rigid_in = route.means_world_S[route.inside_mask_S]
coords_struct = torch.cat([coords_bg, coords_rigid_in], dim=0)
```

不变量：

```python
N_bg = node_state_bg.means.shape[0]
N_rigid_in = route.S_in.numel()

feat_struct_bg.shape[0] == N_bg
feat_struct_rigid_in.shape[0] == N_rigid_in
```

### 5.4 rigid-in 2D 索引对齐（高风险点）

`feat_2d_rigid_S` 顺序对应 `route.S`，不是 `route.S_in`。

必须先取 `S` 内部的行号：

```python
rows_rigid_in_in_S = torch.nonzero(route.inside_mask_S, as_tuple=False).squeeze(1)
feat_2d_rigid_in = feat_2d_rigid_S[rows_rigid_in_in_S]
acc_w_rigid_in = acc_w_rigid_S[rows_rigid_in_in_S]
```

并保证：

```text
feat_struct_rigid_in[i] <-> route.S_in[i]
```

---

## 6. Token 构建方案

### 6.1 组成建议

Stage5_0 token 至少包含：

```text
2D feature + support(acc_w/valid) + branch embedding + Gaussian state embedding
```

推荐 sum-style embedding（比 concat 大 MLP 更稳定）：

```python
x = (
    feat2d_proj(feat_2d_masked)
    + support_proj(support_vec)
    + branch_embed(branch_id)
    + param_proj(param_vec)
)
x = token_norm(x)
```

说明：默认不单独注入 `anchor_rgb`，因为当前 `param_vec` 已覆盖 SH 相关信息（如 `sh_dc` / `sh_rest` 能量统计）；重复注入会增加冗余与不稳定风险。仅当后续关闭参数嵌入时，再考虑启用独立 `anchor_rgb_proj`。

### 6.2 2D 有效性处理

```python
feat_2d_struct = torch.cat([feat_2d_bg, feat_2d_rigid_in], dim=0)
```

若 `zero_invalid_2d_feat=true`：

```python
valid = acc_w_struct > support_min_by_branch
feat_2d_struct = feat_2d_struct * valid.float().unsqueeze(-1)
```

阈值分支化：

```text
bg: self.bg_src_backproject_support_min
rigid_in: self.rigid_src_backproject_support_min
```

### 6.3 support embedding

```python
support_vec = torch.stack([torch.log1p(acc_w_struct), valid.float()], dim=-1)
support_emb = self.struct_support_proj(support_vec)
```

### 6.4 branch embedding

```python
branch_id_bg = torch.zeros(N_bg, dtype=torch.long, device=device)
branch_id_rigid = torch.ones(N_rigid_in, dtype=torch.long, device=device)
branch_id = torch.cat([branch_id_bg, branch_id_rigid], dim=0)
branch_emb = self.struct_branch_embed(branch_id)
```

### 6.5 参数 embedding

复用 Stage4.x 思路：

```python
params_bg = self._build_params_for_embed(node_state_bg)
params_rigid_in = self._build_rigid_params_for_embed_source_world(
    node_state_rigid, source_frame_idx, route.S_in
)
params_struct = cat_param_dict(params_bg, params_rigid_in)
param_vec = self._normalize_params_for_embed(params_struct)
param_emb = self.struct_param_proj(param_vec)
```

注意 rigid-in `means/quats` 必须在 source-frame world 语义下与 bg 对齐。

---

## 7. xCPE Decoder 设计

### 7.1 模块定义

```python
class StreetForwardXCPEDecoder(StreetForwardStructDecoderBase):
    def __init__(
        self,
        feat_2d_channels: int,
        out_channels: int,             # 期望等于 trainer 的 GRU 输入维
        param_dim: int = 17,
        branch_embed_dim: int = 8,
        support_dim: int = 2,
        channels: int = 64,            # internal hidden dim
        voxel_size: float = 0.20,
        num_layers: int = 2,
        residual_scale_init: float = 1e-3,
        sparse_backend: str = "spconv",
    ):
        ...
```

输出：

```python
StructDecoderOutput(
    feat=x,   # [N_struct, out_channels]
    aux={
        "num_struct_points": N_struct,
        "num_struct_voxels": M,
        "xcpe_residual_scale": ...
    }
)
```

维度约束（必须写死）：

```text
struct_decoder.output_dim 必须等于 Stage4_6 的 bg/rigid-in GRU 输入维。
推荐 output_dim=auto -> 由 trainer 在初始化时绑定到 self.fused_in_dim（或等价字段）。
channels 是 decoder 内部宽度，可与 output_dim 不同。
```

推荐实现：

```python
self.struct_out_proj = (
    nn.Identity() if channels == out_channels else nn.Linear(channels, out_channels)
)
```

### 7.2 每层结构

```text
point feat
  -> point-to-voxel mean aggregation
  -> SubMConv3d(C, C, kernel=3)
  -> Linear(C, C)
  -> LayerNorm
  -> GELU
  -> voxel-to-point gather
  -> residual add (small learnable scale)
```

建议 `residual_scale_init=1e-3`，避免初始化为 0 导致起步阶段仅 gate 学习、主干梯度过慢。

### 7.3 point-to-voxel 规则与 fast-fail

```python
grid_coord_xyz = torch.floor((coords_struct - aabb_min) / voxel_size).long()
spatial_shape_xyz = torch.floor((aabb_max - aabb_min) / voxel_size).long() + 1
X, Y, Z = spatial_shape_xyz.tolist()
spatial_shape_zyx = [Z, Y, X]
```

约束：

```python
if not torch.isfinite(coords_struct).all():
    raise RuntimeError("Stage5_0 struct coords contain NaN/Inf.")
if ((coords_struct < aabb_min) | (coords_struct > aabb_max)).any():
    raise RuntimeError("Stage5_0 P_struct contains points outside segment_aabb.")
```

不建议默认静默 clamp。

聚合：

```python
batch_id = offsets_to_batch_ids(batch_offsets, N=grid_coord_xyz.shape[0])  # [N]
grid_key = torch.cat([batch_id[:, None], grid_coord_xyz], dim=1)            # [N, 4] = [b, x, y, z]
unique_key, inverse = torch.unique(
    grid_key, dim=0, sorted=True, return_inverse=True
)
voxel_feat = torch_scatter.scatter_mean(x, inverse, dim=0, dim_size=unique_key.shape[0])
```

其中 `offsets_to_batch_ids` 约定：

```text
single segment: batch_offsets=[N]
multi segment:  batch_offsets=[n0, n0+n1, ...]
```

spconv 坐标规范：

```python
b = unique_key[:, 0]
x = unique_key[:, 1]
y = unique_key[:, 2]
z = unique_key[:, 3]

indices = torch.stack([b, z, y, x], dim=1).int()  # [b, z, y, x]
spatial_shape = spatial_shape_zyx                 # [Z, Y, X]
batch_size = int(batch_offsets.numel())
```

注意：`spatial_shape_xyz = floor(range / voxel) + 1` 是必须项，确保 `coords==aabb_max` 的边界点不会越界。

---

## 8. Trainer 集成方案

### 8.1 先抽 Stage4_6 hook（无语义变更）

新增：

```python
@dataclass
class BgRigidInGRUInputs:
    feat_bg_input: torch.Tensor
    feat_rigid_in_input_all: Optional[torch.Tensor]
    aux: Dict[str, Any]
```

并封装：

```python
def _compute_bg_rigid_in_gru_inputs(...)-> BgRigidInGRUInputs:
    ...
```

默认 Stage4_6 实现仍走 `_build_3d_features_bg_plus_rigid_in + _fuse_features`，保证行为不变。

### 8.2 Stage5_0 override

仅覆盖该 hook：

```python
def _compute_bg_rigid_in_gru_inputs(...):
    struct_in = self._build_struct_decoder_input_bg_rigid_in(...)
    struct_batch_offsets = self._build_struct_batch_offsets(struct_in)  # 单段可退化为 [N]
    struct_out = self.struct_decoder(
        struct_in,
        aabb_min=self.bbx_min,
        aabb_max=self.bbx_max,
        batch_offsets=struct_batch_offsets,
    )
    feat_struct = struct_out.feat
    N_bg = struct_in.split_bg
    N_rigid_in = struct_in.split_rigid_in
    feat_bg_input = feat_struct[:N_bg]
    feat_rigid_in_input_all = feat_struct[N_bg:N_bg + N_rigid_in] if N_rigid_in > 0 else None
    return BgRigidInGRUInputs(feat_bg_input, feat_rigid_in_input_all, struct_out.aux)
```

### 8.3 rigid `U_in` 子集使用

原 Stage4_6 已有 `lookup_S_in` 对齐逻辑。Stage5_0 中：

```python
feat_rigid_in_input = feat_rigid_in_input_all[rows_S_in]
```

不再对 `U_in` 额外做一次旧式 `_fuse_features`。

---

## 9. 初始化、优化器与 checkpoint

### 9.1 初始化顺序

```python
class MinimalStreetForwardStage5_0(MinimalStreetForwardStage4_6):
    def __init__(self, config, device, **kwargs):
        self._validate_stage5_0_config(config)
        super().__init__(config, device, **kwargs)
        self._init_stage5_0_struct_decoder(config)
        self._rebuild_optimizer_after_stage5_modules()
```

说明：Stage4_6 已执行过一次 optimizer 重建；Stage5_0 引入新模块后应再次重建。

### 9.2 checkpoint 兼容策略

```text
Stage4_6 -> Stage5_0 warm-start: strict=False，允许 missing struct_decoder.*
Stage5_0 -> Stage5_0 resume: strict=True
Stage5_0 -> Stage5_1 warm-start: strict=False，复用 token/xCPE
```

建议额外存储：

```text
stage = "5_0"
struct_decoder.*
struct_decoder_cfg_hash
```

---

## 10. 日志指标建议

新增：

```python
{
  "stage5_struct_enabled": 1.0,
  "stage5_struct_num_points": ...,
  "stage5_struct_num_bg": ...,
  "stage5_struct_num_rigid_in": ...,
  "stage5_struct_num_voxels": ...,
  "stage5_struct_voxel_ratio": ...,
  "stage5_xcpe_residual_scale_mean": ...,
  "stage5_bg_struct_feat_norm": ...,
  "stage5_rigid_in_struct_feat_norm": ...,
}
```

重点观察与 Stage4_6 对比：

```text
rigid_scale_offset_saturation_ratio
rigid_opacity_offset_saturation_ratio
bg_opacity_offset_saturation_ratio
rigid_in_update_count / rigid_out_update_count
rigid_in_acc_w_mean / rigid_out_acc_w_mean
```

---

## 11. 测试计划

### 11.1 配置 fast-fail

覆盖：

```text
type != xcpe -> fail
include_distant=true -> fail
include_rigid_out=true -> fail
point_preserving=false -> fail
allow_pooling=true -> fail
```

### 11.2 shape / order

构造：

```text
N_bg = 5
route.S = [2, 3, 8, 9]
inside_mask_S = [True, False, True, False]
route.S_in = [2, 8]
```

检查：

```text
feat_struct_bg.shape[0] == 5
feat_struct_rigid_in.shape[0] == 2
feat_struct_rigid_in[0] -> rigid global 2
feat_struct_rigid_in[1] -> rigid global 8
```

### 11.3 rigid `U_in` 子集

```text
U = [2, 9], U_in = [2], U_out = [9]
rows_S_in for U_in == [0]
feat_rigid_in_input = feat_struct_rigid_in[[0]]
```

### 11.4 point-preserving

```python
x_out = struct_decoder(x_in)
assert x_out.shape[0] == x_in.shape[0]
```

### 11.5 voxel axis

验证 spconv 映射始终满足：

```text
indices: [batch, z, y, x]
spatial_shape: [Z, Y, X]
```

### 11.6 AABB 边界点与 batch-safe 测试

必须新增两个测试：

```text
1) 边界点测试：coords 包含恰好位于 aabb_max 的点，确保不越界。
2) 多 batch 测试：两个 segment 含相同 grid_coord，unique 后不能串味（必须按 batch 维隔离）。
```

---

## 12. 推荐实施步骤

### Step 1：先重构 Stage4_6 hook

引入 `_compute_bg_rigid_in_gru_inputs`，保证 Stage4_6 输出完全等价；先跑现有 Stage4_6 tests。

### Step 2：加入 struct decoder 基础层

先实现：

```text
common.py / token_builders.py / xcpe_decoder.py
```

不引入 attention。

### Step 3：实现 Stage5_0 trainer

`MinimalStreetForwardStage5_0` 只 override：

```text
_validate_stage5_0_config
_init_stage5_0_struct_decoder
_build_struct_decoder_input_bg_rigid_in
_compute_bg_rigid_in_gru_inputs
```

尽量不复制整段 `forward`。

### Step 4：训练入口

新增：

```text
tools/train_minimal_streetforward_stage5_0_multi_scene_v7.py
configs/minimal_streetforward_stage5_0_multi_scene_v7.yaml
```

---

## 13. 风险与规避

### 风险 1：bg / rigid-in 边界污染

规避：必须加 `branch embedding`；首版不引入更复杂 branch mask bias。

### 风险 2：invalid 2D 被邻域激活

规避：invalid 点 `feat_2d=0`；support 中显式保留 valid flag；写回仍受 `mask_update` 控制。

### 风险 3：坐标轴错位

规避：只在一个 wrapper 内做 `xyz -> zyx` 转换并单测锁定。

### 风险 4：显存增长

Stage5_0 主要是线性增长（token + sparse activations），远低于后续 attention 方案；首版建议 `channels=64`。

---

## 14. Stage5_1 预留点

需要保留：

```text
StructDecoderInput.coords / branch_id / batch_offsets
StructDecoderOutput.aux
StreetForwardStructDecoderBase
StreetForwardXCPEDecoder
StreetForwardSerializedPatchDecoder
```

Stage5_1 可在 xCPE 后追加：

```text
serialization code
sort / pad
patch attention
inverse sort
FFN
```

且明确 `no_pooling=true` 保持点数不变。

---

## 15. 推荐 baseline（Stage5_0 v1）

```yaml
model:
  struct_decoder:
    enable: true
    type: xcpe
    scope: bg_rigid_in
    output_role: gru_input
    point_preserving: true
    include_distant: false
    include_rigid_out: false

    channels: 64
    output_dim: auto
    voxel_size: 0.20

    xcpe:
      num_layers: 2
      kernel_size: 3
      residual_scale_init: 1.0e-3

    token:
      use_2d_feat: true
      use_support: true
      use_branch_embed: true
      use_param_embed: true
      use_anchor_rgb: false
      use_hidden_state: false
      zero_invalid_2d_feat: true
```

主流程：

```text
one-pass 2D backproject
  -> bg feat_2d / rigid.S feat_2d
  -> select rigid.S_in rows
  -> build P_struct = bg + rigid.S_in
  -> token projection
  -> xCPE residual structure injection
  -> split bg / rigid-in
  -> bg GRU + bg heads
  -> rigid-in GRU + bg heads
  -> rigid-out old distant path
  -> distant old path
  -> render/loss/writeback unchanged
```

这是最小侵入、语义保持、可向 Stage5_1 平滑演进的 Stage5_0 基线。

