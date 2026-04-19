# StreetForward Stage4_6 设计方案（移除 rigid 专属训练分支，保留 rigid 动态节点）

本文定义 `MinimalStreetForwardStage4_6` 的最终语义与实现边界。目标是移除 rigid 专属 decoder/GRU/head 训练路径，同时保持 rigid 在 dynamic node 语义下的正确时序运动与梯度闭环。

---

## 0. 最终定义

`Stage4_6` 的目标不是删除 `NodeStateRigid`，而是删除 **rigid 专属 decoder/GRU/head 训练路径**。

最终语义：

- rigid 点仍作为 dynamic node 存在；
- rigid 仍使用 local coords、instance pose、target-frame transform、可见性 mask、local NodeState 写回；
- rigid 不再使用 rigid branch MLP/GRU/decoder，而是依据 source-frame 世界坐标，路由到 `bg` 或 `distant` decoder。

推荐继承骨架（避免 `super().__init__()` 冲突）：

```python
class MinimalStreetForwardStage4_6(MinimalStreetForwardStage4_5BaseNoRigidHead):
    ...
```

关键原因：若直接 `Stage4_6(MinimalStreetForwardStage4_5)` 且在配置中删除 `branches.rigid.mlp/limits`，通常会在父类链（`Stage4_2 -> Stage4_1 -> Stage4_0`）解析 rigid 配置并创建 rigid 专属 head 时提前失败，还未进入 `Stage4_6.__init__()` 主体。

因此 `Stage4_6` 必须满足以下初始化策略之一（以 A 为推荐）：

- **A（推荐）**：抽 `MinimalStreetForwardStage4_5BaseNoRigidHead`，复用 `Stage4_5` 的 no-sky loss、source mask、fused multi-camera 2D、target rendering helpers，但跳过 `Stage4_0/4_1` rigid-head 创建；
- **B（备选）**：拆分父类初始化路径，确保 `Stage4_6` 在进入会创建 rigid head 的父类逻辑前拦截并替换模块创建。

`Stage4_6` 不允许“先完整跑完 `Stage4_5.__init__()` 再删除 rigid 模块”，因为这会让配置语义与构造路径长期不一致。

---

## 1. 文件与入口

新增：

- `models/streetforward/minimal_trainer_stage4_6.py`
- `configs/minimal_streetforward_stage4_6_multi_scene_v7.yaml`
- `tools/train_minimal_streetforward_stage4_6_multi_scene_v7.py`

更新：

- `models/streetforward/__init__.py`（lazy import `MinimalStreetForwardStage4_6`）

不考虑向后兼容：

- 不支持旧 rigid decoder checkpoint strict load；
- 不保留 `model.branches.rigid.mlp`；
- 不保留 `mlp_offset_pos_rigid / mlp_conv_rigid / mlp_opacity_rigid / gaussion_decoder_rigid`；
- 不允许 `Stage4_6` 进入 `Stage4.2/4.4` 的 sky 或 rigid-head 路径。

---

## 2. 配置结构

### 2.1 保留 `branches.rigid`，仅用于 NodeState/update

`branches.rigid` 仅保留 `init / eta / src_backproject_support_min`；不再包含：

- `mlp`
- `limits`
- `freeze_means`
- `freeze_quat`

若出现上述字段，初始化时直接 fast-fail。  
注意：该 fast-fail 生效前提是 `Stage4_6` 初始化路径已绕开“父类强依赖 rigid.mlp/limits”的旧构造逻辑（见第 0 节）。

### 2.2 新增 routed rigid 配置

```yaml
model:
  rigid_routed:
    route_space: source_frame_world
    route_aabb: segment_aabb
    inside_decoder: bg
    outside_decoder: distant
    update_means: true
    update_quat: true
```

强约束：

- `inside_decoder == bg`
- `outside_decoder == distant`
- `route_space == source_frame_world`
- `route_aabb == segment_aabb`

---

## 3. 保留与删除的模块

### 3.1 必须保留

- `NodeStateRigid`
- `node_states_rigid`
- `h_cache_rigid`
- `_transform_rigid_to_world`
- `_transform_rigid_quats_to_world`
- `_build_rigid_world_for_frame`
- `_resolve_rigid_frame_idx`

原因：target 渲染仍需按 target frame 将 rigid local params 变换到世界坐标。

### 3.2 必须删除/不再创建

- `self.rigid_feat_proj`
- `self.mlp_offset_pos_rigid`
- `self.mlp_conv_rigid`
- `self.mlp_opacity_rigid`
- `self.gaussion_decoder_rigid`

不再调用：

- `_predict_offsets_gru_rigid(...)`
- `_render_params_from_offsets_rigid(...)`

rigid 路由统一为：

- `rigid_in -> bg decoder / bg heads`
- `rigid_out -> distant decoder / distant heads`

---

## 4. Source rigid routing

新增：

```python
@dataclass
class RigidRoute:
    S: torch.Tensor
    S_in: torch.Tensor
    S_out: torch.Tensor
    inside_mask_S: torch.Tensor
    route_inside_global: torch.Tensor
    means_world_S: torch.Tensor
    quats_world_S: torch.Tensor
```

新增函数：

```python
def _route_rigid_source_points(
    self,
    node_state_rigid: NodeStateRigid,
    source_frame_idx: int,
    S: torch.Tensor,
) -> RigidRoute:
    ...
```

核心规则：

- routing 仅在 source frame 做一次；
- 同一点在一次 forward 内只能归属一个 decoder；
- target frame 不允许重路由；
- routing 坐标必须使用 rigid 变换后的 source-frame world coords，不能用 local coords。

---

## 5. Source 2D backprojection

`Stage4_6` 继续沿用 `Stage4_5` scene-only fused multi-camera source 2D，但 rigid source-world 状态应复用 `route`，避免重复 transform：

- 拼接 `[bg, distant, rigid_S]`；
- 一次 fused backproject；
- split 得到 `feat_2d_bg / feat_2d_distant / feat_2d_rigid_S` 及对应 `acc_w_*`；
- `gaussians_rigid@S` 的 `means/quats` 建议直接使用 `route.means_world_S / route.quats_world_S`。

建议新增（或等效替换）：

```python
_compute_2d_features_all_branches_once_routed(..., route: RigidRoute)
```

主路径统一使用 `feat_2d_rigid_S + lookup`，避免双口径实现：

```python
lookup_S = torch.full((N_rigid,), -1, dtype=torch.long, device=self.device)
lookup_S[route.S] = torch.arange(route.S.numel(), device=self.device)
```

`feat_2d_rigid_in_S / feat_2d_rigid_out_S` 可选用于日志，不建议参与主训练路径索引。

rigid source support 与 update mask：

- `mask_src_feat_valid_rigid[route.S] = (acc_w_rigid_S > rigid_src_backproject_support_min)`
- `mask_update_rigid = mask_src_feat_valid_rigid & mask_any_tgt_rigid`
- `U = nonzero(mask_update_rigid)`（仅作为 routed split 前的候选更新集合）

---

## 6. 3D voxel 构建

`Stage4_6` 的 3D feature 不再只给 bg，而是：

- `bg + rigid_in(source world)`

新增函数：

```python
def _build_3d_features_bg_plus_rigid_in(
    self,
    node_state_bg: NodeStateBackground,
    node_state_rigid: Optional[NodeStateRigid],
    route: RigidRoute,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    ...
```

返回：

- `feat_3d_bg`
- `feat_3d_rigid_in`

实现建议：`means_rigid_in_world = route.means_world_S[route.inside_mask_S]`，避免第二次 local->world transform。  
约束：`feat_3d_rigid_in` 的行顺序必须与 `route.S_in` 完全一致。

---

## 7. GRU/decoder 统一接口

新增通用函数，替代分支硬编码：

```python
def _predict_offsets_gru_with_heads(
    self,
    feat: torch.Tensor,
    params_for_embed: Dict[str, torch.Tensor],
    h_old: torch.Tensor,
    *,
    mask_update: Optional[torch.Tensor],
    limits: Dict[str, float],
    mlp_offset_pos,
    mlp_conv,
    mlp_opacity,
    gaussion_decoder,
    freeze_quat: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    ...
```

关键语义：

- 内部复用现有参数 embed、GRU、head 预测；
- `mask_update` 时执行 full gate（offset/quat/h 一致门控）；
- 未更新点的 `offset_quat` 必须回退 identity，`h_new` 必须保持 `h_old`。

---

## 8. bg / distant / rigid routed 特征路径

### 8.1 bg

逻辑保持 `Stage4_5`，改为调用统一 `_predict_offsets_gru_with_heads`。

### 8.2 distant

逻辑保持 `Stage4_5`（2D-only + distant head），同样走统一接口。

### 8.3 rigid_in

- `U_in = U[route.route_inside_global[U]]`
- 特征来源：`feat_2d_rigid_S` + `feat_3d_rigid_in`（通过 lookup 对齐到 `U_in`）
- 参数 embed 使用 source-frame world params（非 local params）
- 预测 head 使用 bg 分支参数（`bg limits + bg heads`）

索引硬检查（必须）：

```python
rows_S = lookup_S[U_in]
if (rows_S < 0).any():
    raise RuntimeError("Routed rigid update point not present in source visible S.")

rows_S_in = lookup_S_in[U_in]
if (rows_S_in < 0).any():
    raise RuntimeError("U_in contains rigid point not present in S_in.")
```

### 8.4 rigid_out

- `U_out = U[~route.route_inside_global[U]]`
- 特征来源：`feat_2d_rigid_S`（经 `distant_feat_proj`）
- 参数 embed 使用 source-frame world params
- 预测 head 使用 distant 分支参数（`distant limits + distant heads`）

同样必须检查：

```python
rows_S = lookup_S[U_out]
if (rows_S < 0).any():
    raise RuntimeError("Routed rigid update point not present in source visible S.")
```

---

## 9. rigid world offset -> local render params

新增：

```python
def _render_params_from_routed_offsets_rigid_local(
    self,
    node_state_rigid: NodeStateRigid,
    source_frame_idx: int,
    U: torch.Tensor,
    offsets_world: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    ...
```

硬约束：

- 输入 offsets 是 source-frame world 语义；
- 输出必须是 local render params；
- 输出行顺序与 `U` 一一对应；
- 禁止通过 `detach/no_grad` 破坏可导链路。

### 9.1 position

- local -> source world
- 在 world 应用 `offset_pos`
- source world -> local（新增 `_transform_rigid_points_world_to_local`）

### 9.2 quaternion

- local -> source world quaternion
- world 空间右乘 `offset_quat` 并归一化
- source world -> local quaternion（新增 `_transform_rigid_quats_world_to_local`）

### 9.3 scale / opacity / SH

这些参数不依赖 frame pose，按 local NodeState 语义更新。

---

## 10. Pack rigid_in / rigid_out

预测后合并为 row-aligned 子集：

- `U_all = cat(U_in, U_out)`
- `render_params_rigid_local_U[k] = cat([in[k], out[k]], dim=0)`

强约束：

- `render_params_rigid_local_U[k][i]` 对应 rigid global index `U_all[i]`
- 后续 target 渲染必须使用 `U_all`，不能回退旧 `U`。
- `U_all` **不要求升序**；唯一不变量是 row 与 `U_all` 一一对齐。

---

## 11. Target 渲染路径

保持 `Stage4_1/4_5` 动态语义，但 train/frozen 划分必须基于 `U_all` 反建 mask，而不是直接复用旧 `mask_update_rigid`：

- 先构造：

```python
mask_train_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
mask_train_rigid[U_all] = True
```

- 每个 target frame 计算：

```python
idx_train = torch.nonzero(mask_train_rigid & mask_tgt_by_frame[F], as_tuple=False).squeeze(-1)
idx_frozen = torch.nonzero((~mask_train_rigid) & mask_tgt_by_frame[F], as_tuple=False).squeeze(-1)
```

- 调用 `_build_rigid_world_for_frame(..., render_params_rigid_local=render_params_rigid_local_U, U=U_all)`；
- 与 `bg + distant` 合并后渲染；
- photometric loss 仍在 non-sky 区域计算。

禁止将 source-world rigid params 直接拼到 target 渲染输入，否则会破坏 target pose 语义。

---

## 12. Backward 路径

继续使用现有 proxy backward：

```python
_backward_to_render_params_bg_rigid_distant(
    ...,
    rigid_world_proxy_pairs=rigid_world_proxy_pairs,
)
```

梯度链要求：

`target loss -> target rigid world proxy -> target rigid world params -> render_params_rigid_local_U -> offsets_rigid_in/out_world -> shared bg/distant heads+GRU+features`

禁止：

- `render_params_rigid_local_U.detach()`
- `with torch.no_grad(): world_to_local_offset`

---

## 13. Hidden cache 更新

- `h_cache_bg[key] = h_new_bg.detach()`
- `h_cache_distant[key] = h_new_distant.detach()`
- `h_cache_rigid[key]` 使用 `h_old_rigid.clone()` 后按 `U_in/U_out` 局部覆盖。

约束：

- rigid 保持独立 cache；
- decoder 共享不等于 hidden/cache 共享。

---

## 14. NodeState 写回

新增 subset-safe 写回函数：

```python
def _update_node_state_rigid_local_subset(
    self,
    node_state_rigid: NodeStateRigid,
    render_params_rigid_local_U: Dict[str, torch.Tensor],
    U: torch.Tensor,
) -> None:
    ...
```

语义要求：

- 右侧按 local row 顺序读取，左侧按 global 索引 `U` 写入；
- 禁止使用 `render_params_rigid_local["means_r"][U]` 这类 global/subset 混用写法。

---

## 15. Fast-fail 检查

初始化检查（必须）：

- `model.sky` 存在则报错；
- `model.branches.sky` 存在则报错；
- `branches.rigid` 出现 `mlp/limits` 则报错；
- 若创建了 rigid-specific decoder heads，则报错。

forward 检查（必须）：

- `U_all` 与 `render_params_rigid_local_U` 行数一致；
- `U_all` 类型、设备、取值范围合法；
- `S_in + S_out == S`，且 `inside_mask_S` 维度一致。
- `lookup_S[U_in/U_out]` 与 `lookup_S_in[U_in]` 不得出现负值。

debug roundtrip 检查（建议）：

- position 与 quaternion 的 world->local->world 回环误差超过阈值即报错。

---

## 16. 训练日志建议新增

- `rigid_route_num_S / rigid_route_num_in / rigid_route_num_out`
- `rigid_route_ratio_in / rigid_route_ratio_out`
- `rigid_in_update_count / rigid_out_update_count`
- `rigid_in_acc_w_mean / rigid_out_acc_w_mean`
- `rigid_in_offset_pos_norm_mean / rigid_out_offset_pos_norm_mean`
- `rigid_in_scale_offset_saturation_ratio / rigid_out_scale_offset_saturation_ratio`
- `rigid_in_opacity_offset_saturation_ratio / rigid_out_opacity_offset_saturation_ratio`
- `rigid_writeback_count`

重点观测：

- rigid_in 的 scale/opacity saturation 比例是否下降；
- bg 与 rigid near object 的 opacity 竞争是否减弱；
- target frame dynamic object 是否保持随 pose 正确运动。

---

## 17. 最终 forward 总流程（Stage4_6）

```text
forward(batch)
  -> validate Stage4.5 batch
  -> init bg/distant/rigid NodeState
  -> source_frame_idx
  -> compute source visible rigid set S
  -> route S in source-frame world: S_in(bg), S_out(distant)
  -> source 2D fused backprojection for [bg, distant, rigid_S] (reuse route world params)
  -> build masks: mask_update_bg/distant/rigid, then U_in/U_out
  -> build 3D feature for [bg + rigid_in(source-world)]
  -> bg path: fuse(3D+2D) -> shared bg heads
  -> distant path: 2D-only -> distant heads
  -> rigid_in path: fuse(3D+2D) -> bg heads -> world offsets -> local params
  -> rigid_out path: 2D-only -> distant heads -> world offsets -> local params
  -> pack U_all + render_params_rigid_local_U
  -> build mask_train_rigid from U_all
  -> per target frame:
       build rigid world params from local(U_all) + frozen subset
       merge bg+rigid_world+distant
       render + non-sky loss
  -> loss.backward()
  -> proxy backward
  -> optimizer.step()
  -> update h_cache
  -> subset-safe NodeState writeback
```

---

## 18. forward() 输出字段契约（必须）

`Stage4_6.forward()` 需明确返回训练主路径所需内部字段，供 `train_step / scheduler sync / writeback / logging` 使用。建议至少包含：

```python
{
    "loss": loss,
    ...
    "_cache_key": key,

    "_node_state_bg": node_state_bg,
    "_node_state_distant": node_state_distant,
    "_node_state_rigid": node_state_rigid,

    "_render_params_bg": render_params_bg,
    "_proxies_bg": proxies_bg,

    "_render_params_distant": render_params_distant,
    "_proxies_distant": proxies_distant,

    "_render_params_rigid_local": render_params_rigid_local_U,
    "_rigid_writeback_idx": U_all,
    "_rigid_world_proxy_pairs": rigid_world_proxy_pairs,

    "_h_new_bg": h_new_bg,
    "_h_new_distant": h_new_distant,
    "_h_new_rigid": h_new_rigid_full,

    "_num_rigid_valid_src": int(route.S.numel()),
    "_num_rigid_total": N_rigid,
}
```

说明：

- 现有 `train_step` 路径通常依赖 `_h_new_*`、`_rigid_writeback_idx`、`_rigid_world_proxy_pairs`；
- `rigid writeback` 必须与 `U_all` 行对齐，避免默认假设 `U_all` 升序；
- 日志统计以 `route` 与 `U_all` 为唯一口径，避免与旧 `mask_update_rigid` 混用。

---

## 19. 最终判断

该方案满足三个核心目标：

1. **rigid 专属训练分支被彻底移除**  
   rigid 不再维护独立 MLP/head/decoder。

2. **rigid 动态语义保持正确**  
   rigid 仍以 local state 存储，target frame 渲染仍经动态 pose 变换。

3. **forward/backward/writeback 路径闭合**  
   source-world routed offsets 能正确回到 rigid local params；target loss 梯度能通过 rigid world proxy 回传到共享 bg/distant 解码路径，最终以 `U_all` 行对齐子集安全写回。

