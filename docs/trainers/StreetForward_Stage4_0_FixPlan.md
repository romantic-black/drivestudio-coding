# StreetForward Stage 4.0 修正方案（rigid 可见性与入口可靠性）

本文档针对当前 `MinimalStreetForwardStage4_0` 的实现偏差，给出修正方案。  
目标：在不改变 Stage 4.0 “`src=target` 简化路线”的前提下，先把 rigid 分支的**正确性**和**可观测性**补齐。

---

## 1. 问题归纳（按优先级）

## 1.1 P0：rigid 缺 pose/不可见实例被静默当作可渲染

当前实现仅校验 `frame_idx` 是否存在于 `frame_ids`，没有对该帧每个 instance 的 `instances_fv` 做 gate。  
结果是：某些实例在该帧无有效 pose 时，会落到初始化值（单位 quat + 零平移）语义，错误参与 2D 特征提取与渲染。

这是功能正确性问题，必须优先修复。

## 1.2 P0：rigid 分支入口条件绑错，存在静默失效

当前 rigid 分支启动条件与 `feat_2d_bg` 绑定，而不是 rigid 自身特征可用性。  
当配置变化或某路径返回为空时，可能“无报错跳过 rigid”，造成训练行为不可控。

## 1.3 P1：rigid 参数 embedding 坐标系与观测语义不对齐

rigid 的 2D 特征来自 source/world 语义，但参数 embedding 使用 local 语义，可能导致 GRU 输入分布不一致。

## 1.4 P1：单帧监督缺最小 gate

虽暂不做跨 target gate，但至少要做 source 帧 per-instance 有效性 gate（`instances_fv=True`）：
- 不可见/无 pose 点不进入 rigid feature/render/loss。

---

## 2. 修正原则

- **fast-fail 优先**：关键约束不满足时直接报错，禁止 silent fallback。
- **最小侵入**：保持 Stage 4.0 单帧训练主流程，先补正确性，再谈扩展。
- **显式语义**：区分“帧存在”与“instance 在帧中有效”两个层级。

---

## 3. 具体改造方案

## 3.1 新增 rigid source 有效掩码（P0）

在 Stage4 trainer 内新增：

- `_rigid_instance_valid_mask(node_state_rigid, frame_idx) -> [num_instances] bool`
- `_rigid_point_valid_mask(node_state_rigid, frame_idx) -> [Nr] bool`（由 `point_ids` 扩展）

规则：

1. `frame_idx` 不在 `frame_ids`：直接 `raise ValueError`。
2. `frame_idx` 在 `frame_ids` 但某 instance 的 `instances_fv=False`：
   - 这些点标记为 invalid；
   - 不参与 rigid 2D feature / render / loss / update。

禁止继续把 invalid 点用默认 pose 参与训练。

## 3.2 变换函数增加 valid mask 路径（P0）

改造：

- `_transform_rigid_to_world(...)`
- `_transform_rigid_quats_to_world(...)`

新增参数（建议）：

- `valid_point_mask: Optional[torch.Tensor]`
- `strict_invalid: bool = True`

行为：

- `strict_invalid=True` 且存在 invalid 点时：直接报错（用于调试/验证）。
- 训练默认 `strict_invalid=False`：仅对 valid 点做变换，invalid 点不输出到后续渲染分支（通过索引过滤实现）。

> 注意：推荐“过滤点”而非“输出零向量”。零向量会污染渲染与监督语义。

## 3.3 rigid 分支入口条件修正（P0）

当前 rigid 入口条件改为只依赖 rigid 自身：

- `node_state_rigid is not None`
- rigid valid 点数 > 0
- rigid 2D 特征成功返回且 shape 匹配

并去掉对 `feat_2d_bg` 的隐式依赖。

## 3.4 rigid 2D 维度 hard check（P0）

在 `__init__` 或首个 forward 前执行硬校验：

- `rigid_cfg.mlp.use_2d_feat` 必须为 `true`
- `rigid_cfg.mlp.use_3d_feat` 必须为 `false`
- `rigid_feat_proj.in_features` 与实际 `feat_2d_rigid.shape[-1]` 不一致时直接报错

防止配置演化后静默跳过。

## 3.5 rigid embedding 语义对齐（P1）

将 rigid 的 `params_for_embed` 改为 world/source 语义：

1. 用当前 `frame_idx` 把 `means/quats` 从 local 变到 world；
2. 再喂 `_build_params_for_embed(..., coord_space="world")`（或新增 rigid 专用函数）。

这样特征观测（2D/world）与参数 embedding 语义一致。

## 3.6 单帧最小 gate（P1）

在 `src=target` 下实现最小 gate：

- valid rigid 点才进入：
  - 2D 特征反投影输入集合
  - render 合并
  - mask/loss 影响
  - NodeState 写回（仅对 valid 索引更新；invalid 保持上一状态）

---

## 4. 日志与可观测性增强

每 `log_interval` 输出：

- `num_rigid_total`
- `num_rigid_valid_src`
- `num_rigid_invalid_src`
- `rigid_valid_ratio`
- `num_rigid_rendered`（最终进入渲染的 rigid 点数）

发生以下情况时强告警/报错：

- `num_rigid_total > 0 && num_rigid_valid_src == 0`
- `rigid_feat_shape_mismatch`
- `frame_idx_missing_in_dynamic_info`

---

## 5. 测试补充（必须）

新增/调整单测：

1. `test_rigid_invalid_instance_filtered`  
   构造 `instances_fv=False` 实例，确认其点不进入 render 合并。

2. `test_rigid_frame_missing_fast_fail`  
   `frame_idx` 不在 `frame_ids`，应抛错。

3. `test_rigid_entry_does_not_depend_on_bg_feat`  
   bg 特征置空场景下，rigid 仍按自身条件运行或明确报错，不得静默跳过。

4. `test_rigid_feat_dim_mismatch_fast_fail`  
   人为制造 2D 维度不匹配，确认立即报错。

5. `test_rigid_world_embed_alignment`  
   确认 rigid params embedding 使用 world/source 语义（可通过关键 tensor 范围或 hook 断言）。

---

## 6. 落地顺序（建议）

### Step A（P0）

- 加 valid mask 计算与入口条件修正
- 加维度 hard check
- 先让错误显式暴露（fast-fail）

### Step B（P1）

- rigid embedding 对齐到 world/source
- 单帧最小 gate（写回按 valid 索引更新）

### Step C（验证）

- 补齐单测
- 用 overfit batch 跑 smoke：
  - loss 可下降
  - rigid 有效点统计稳定
  - 无 silent skip

---

## 7. 预期结果

修正后 Stage 4.0 的 rigid 分支将具备：

- 正确的 per-instance source 有效性语义
- 无 silent skip 的稳定入口
- 与观测一致的参数 embedding 坐标系
- 可追踪、可定位的日志与单测保障

这将为后续 Stage 4.1（多 target 可见性 gate）提供可靠基线。

