# StreetForward Stage 4.0 模型方案（Rigid Node 首版接入）

本文档面向 Stage 4.0：在 Stage 3.3（bg/distant 解耦）的基础上，补齐 `rigid node` 路径，形成 `bg + distant + rigid` 的可训练闭环。

当前方案严格满足以下约束：

1. **Rigid 独立配置**：参考 `StreetForward_Stage3_3_Design.md`，为 rigid 提供独立 `init / limits / eta / mlp`，并保持 `rigid.use_3d_feat=false`（3D feat 暂只给 bg）。
2. **简化时序**：Stage 4.0 先使用 `src = target`，不引入额外 target 帧，不处理跨 target 可见性。
3. **Rigid 坐标语义**：rigid 点云为局部坐标，必须基于 `dynamic_info` 变换到世界坐标后再参与渲染与监督。

---

## 1. 目标与非目标

### 1.1 目标

- 在 minimal 训练路径中引入 rigid 分支，并与 bg/distant 一样具备独立初始化与更新控制。
- 保持 Stage 4.0 改动“最小可落地”：先单 target（即 source 帧）训练通路打通，再扩展多 target。
- 保证 rigid 的局部坐标与世界坐标转换逻辑正确、可微、可监控。

### 1.2 非目标（Stage 4.0 暂不做）

- 不做 rigid 的跨 target 可见性 gate（`mask_src & mask_any_tgt` 这套先不接）。
- 不做 rigid 的 3D feature 体素采样与 3D+2D 融合。
- 不做复杂时序监督（如 target 列表中多帧联合渲染）。

---

## 2. 基线与差异

### 2.1 基线来源

- `minimal_trainer_stage3_3.py`：已有 bg/distant 分支化配置、distant 独立 heads、2D-only distant 路径。
- `node_state_mixin.py`：已有 rigid 初始化、frame 映射、`_transform_rigid_to_world` / `_transform_rigid_quats_to_world`。
- `trainer.py`：完整大 trainer 已有 rigid 流程编排，可作为 Stage 4.0 的“机制参考”，但 minimal 版本先不搬运全部复杂逻辑。

### 2.2 Stage 4.0 关键差异

- 从 `bg+distant` 升级到 `bg+distant+rigid`。
- rigid 与 distant 类似，首版为 **2D-only 输入 + 独立 offset heads**，但其渲染坐标来自“局部参数 + dynamic pose”。
- 训练组织改为单 target（等于 source 帧）语义，消除跨帧可见性复杂性。

---

## 3. 配置设计（fast-fail）

## 3.1 新增分支

建议在 `model.branches` 下新增 `rigid`，与 `bg/distant` 对齐：

```yaml
model:
  branches:
    bg: ...
    distant: ...
    rigid:
      freeze_means: false
      init:
        scale_init:
          mode: isotropic
          isotropic_log_value: -2.10
          knn_k: 3
          knn_log_scale_bias: 0.0
        opacity_init: 0.10
      limits:
        offset_max: 0.05
        scale_max: 0.08
        omega_max: 0.05
        opacity_max: 0.08
        sh_dc_max: 0.08
        sh_rest_max: 0.03
      eta:
        means: 0.5
        scales: 0.8
        opacity: 0.8
        sh_dc: 0.8
        sh_rest: 0.5
      mlp:
        hidden_dim: 64
        use_3d_feat: false   # Stage 4.0 强约束
        use_2d_feat: true
        freeze_quat: false
```

### 3.2 校验规则（必须）

- `model.branches.rigid` 缺失直接报错。
- rigid 的 `init/limits/eta/mlp/freeze_means` 任一缺失报错。
- 若 `rigid.mlp.use_3d_feat != false`，直接报错（Stage 4.0 限制）。
- 若 batch 内存在 dynamic 点云但缺 `dynamic_info`，直接报错。

---

## 4. NodeState 与初始化方案

### 4.1 Rigid 初始化独立化

沿用 `node_state_mixin.py` 的 rigid 数据结构，但初始化参数改成读取 `branches.rigid.init`：

- `scales_log`：支持 `isotropic | knn`。
- `opacity_logit`：由 `opacity_init` 映射。
- `quats`：建议与 Stage 3.3 一致，默认单位四元数初始化（避免随机旋转扰动）。

### 4.2 dynamic_info 对齐

初始化 `instances_quats / instances_trans / instances_fv` 时：

- 仅加载在 pointcloud dynamic 实例集合内的实例（过滤 annotation 漂移）。
- `frame_ids` 建立稳定索引映射，供后续 `frame_idx -> frame_slot` 解析。
- frame 缺 pose 时应返回“不可用”状态（而非静默使用错误帧）。

---

## 5. 前向流程（Stage 4.0 简化版）

## 5.1 src=target 约束落地

在 Stage 4.0 forward 中，统一将监督目标设置为 source：

- `target_frame_idx = source_frame_idx`
- `target_view = source_view`（或 source 列表中的唯一/首个视角）
- 不再遍历多 target，不计算跨 target mask。

这样可把复杂时序问题收敛为“单帧重建 + 可微更新”。

## 5.2 分支特征输入

- `bg`：保持现状（允许 3D + 2D 融合）。
- `distant`：保持 Stage 3.3（2D-only + distant 独立 heads）。
- `rigid`：Stage 4.0 使用 **2D-only**：
  - 从 source 视图反投影得到 `feat_2d_rigid`；
  - 通过 `rigid_feat_proj` 投影到 head 输入维度；
  - 不接 3D 体素特征。

## 5.3 Rigid offsets 与渲染参数

建议新增 rigid 独立头：

- `mlp_offset_pos_rigid`
- `mlp_conv_rigid`
- `mlp_opacity_rigid`
- `gaussion_decoder_rigid`

并新增：

- `_predict_offsets_gru_rigid(...)`（可共用 GRU 主干，分离最后 heads）
- `_render_params_from_offsets_rigid_local(...)`（先在局部坐标更新）

---

## 6. 局部坐标到世界坐标（核心机制）

### 6.1 变换时机

rigid 参数更新建议在“局部坐标”完成，再在渲染前变换到世界坐标：

1. 局部更新（可微）：
   - `means_local_r = means_local + eta_means * offset_pos_local`
   - `quats_local_r = normalize(quat_local * offset_quat_local)`
2. 根据 `dynamic_info`（当前帧实例位姿）变换到世界坐标：
   - `means_world_r = R_instance * means_local_r + t_instance`
   - `quats_world_r = normalize(quat_instance * quats_local_r)`

### 6.2 具体接口建议

直接复用并约束使用：

- `_transform_rigid_to_world(node_state_rigid, means_local, point_indices=None, frame_idx=source_frame_idx)`
- `_transform_rigid_quats_to_world(node_state_rigid, quats_local, point_indices=None, frame_idx=source_frame_idx)`

并要求：

- `frame_idx` 显式传入（禁止依赖隐式 `cur_frame` 默认值）。
- 当 frame 无效时 fast-fail（优先报错），避免 silently 返回零向量掩盖数据问题。

### 6.3 反向传播保证

- 世界坐标渲染参数必须保持与局部渲染参数的 autograd 链接。
- 禁止在局部->世界变换路径中 `detach()`。
- 写回 NodeState 时再做 detach/copy（与现有缓冲区语义一致）。

---

## 7. 渲染与损失（单 target）

### 7.1 参数合并

渲染前合并三分支世界坐标参数：

- `bg_world`
- `rigid_world`（由局部变换得到）
- `distant_world`

### 7.2 损失建议（保持简单）

Stage 4.0 延续 Stage 3.3 组合：

- `loss_rgb = w_l1 * L1 + w_ssim * SSIM`
- `loss_mask`（若 sky_mask 可用）
- `loss_opacity_entropy`（可选）

由于 `src=target`，先确保损失可稳定下降，不引入额外时序正则。

---

## 8. 代码落地计划（MVP）

### Phase A：配置与初始化

1. 新增 stage4_0 配置文件（例如 `configs/minimal_streetforward_stage4_0.yaml`）。
2. 在 trainer 初始化中解析并校验 `branches.rigid`。
3. rigid NodeState 初始化接入独立 `init` 参数（含 isotropic/knn）。

验收：

- 配置缺项立即报错；
- dynamic 点云存在时 rigid 正常初始化；
- rigid 初始化统计（scale/opacity）可在日志中确认。

### Phase B：前向接入（src=target）

1. 将 target 简化为 source 帧单目标。
2. 接入 rigid 2D-only 特征路径（不读 3D feat）。
3. 新增 rigid 独立 offset heads 与 render params（局部）。

验收：

- forward 可跑通且 shape 全部一致；
- rigid 分支参数能收到梯度；
- 不出现 NaN/Inf。

### Phase C：局部->世界渲染闭环

1. 在渲染前将 rigid 局部参数变换到世界坐标。
2. 合并 bg/rigid/distant 进行单 target 渲染与损失。
3. 更新 NodeState（rigid 写回局部参数）。

验收：

- rigid 世界坐标可视化位置正确（随 dynamic pose 变化）；
- loss 稳定下降；
- 写回后下一个 iter 继续可训练。

---

## 9. 风险与应对

- 风险1：`dynamic_info` 帧索引不一致导致 rigid 位姿错用  
  - 应对：统一 `frame_idx -> frame_slot` 解析，找不到直接报错。

- 风险2：局部->世界变换断梯度  
  - 应对：禁止 detach，增加梯度存在性断言（关键 tensor `.grad is not None` 监控）。

- 风险3：src=target 过于简单，泛化不足  
  - 应对：明确 Stage 4.0 为机制打通版本；Stage 4.1 再扩展多 target 可见性。

- 风险4：rigid 与 distant 同时 2D-only 导致表达不稳  
  - 应对：先收紧 rigid limits/eta，必要时提升 rigid head 宽度而非引入 3D feat。

---

## 10. Stage 4.1 预留（非本次实现）

- 从 `src=target` 升级到 `src + multi-target`。
- 引入 rigid 的 `mask_src / mask_tgt / mask_update` 机制。
- 评估 rigid 是否需要有限的 3D 几何先验（仍可保持默认关闭）。

---

## 11. 结论

Stage 4.0 建议采用“**独立 rigid 分支 + 单帧简化监督 + 严格局部到世界变换**”的最小闭环方案：  
先确保 rigid 机制正确、稳定、可训练，再在后续版本扩展跨 target 可见性与更复杂时序监督。该方案与 Stage 3.3 的分支化设计兼容，改动可控，适合快速迭代验证。

