# StreetForward Stage 3.3 模型构建方案（bg/distant 解耦）

本文档基于 `docs/trainers/StreetForward_Flow.md` 与 `models/streetforward/minimal_trainer_stage3_2.py`，提出 Stage 3.3 的可落地方案：将 `bg node` 与 `distant node` 的初始化参数、offset 约束参数与 eta 步长参数彻底分离，并让 `distant` 分支不依赖 3D feature，使用独立 MLP 预测。

> 设计前提（必须满足）：`distant` 不在 `segment_aabb` 内，因此不存在可靠的 3D feature；Stage 3.3 中 `distant` 分支默认只使用 2D/appearance 路径。

## 1. 目标与约束

- 目标1：在配置层明确区分 `bg` 与 `distant` 参数，避免共享超参导致训练行为耦合。
- 目标2：保持 Stage 3.2 训练主路径（GRU-style + proxy rendering）不被破坏，尽量增量改造。
- 目标3：`distant` 不使用 3D feature，改为独立输入特征与独立 MLP，降低无效计算与梯度干扰。
- 目标4：fast-fail：关键配置缺失时报错，不自动兜底默认值（遵循当前项目习惯）。

## 2. Stage 3.3 核心改动

### 2.1 配置分层：bg 与 distant 独立

Stage 3.2 当前是 `model.offset_max / eta_means / ...` 的单套参数。Stage 3.3 建议升级为分支配置：

```yaml
model:
  branches:
    bg:
      init:
        scale_init:
          mode: isotropic  # isotropic | knn
          isotropic_log_value: -2.30  # s, scales_log=[s,s,s]
          knn_k: 3
          knn_log_scale_bias: 0.0
        opacity_init: 0.1
      limits:
        offset_max: 0.1
        scale_max: 0.1
        omega_max: 0.1
        opacity_max: 0.1
        sh_dc_max: 0.1
        sh_rest_max: 0.05
      eta:
        means: 1.0
        scales: 1.0
        opacity: 1.0
        sh_dc: 1.0
        sh_rest: 1.0
      mlp:
        hidden_dim: 64
        use_3d_feat: true
        use_2d_feat: true

    distant:
      init:
        scale_init:
          mode: isotropic  # isotropic | knn
          isotropic_log_value: -1.90  # 比 bg 稍大，提升远景 footprint 稳定性
          knn_k: 3
          knn_log_scale_bias: 0.0
        opacity_init: 0.05
      limits:
        offset_max: 0.02
        scale_max: 0.08
        omega_max: 0.01
        opacity_max: 0.05
        sh_dc_max: 0.08
        sh_rest_max: 0.015
      eta:
        means: 0.2
        scales: 0.7
        opacity: 0.5
        sh_dc: 0.8
        sh_rest: 0.4
      mlp:
        hidden_dim: 64
        use_3d_feat: false
        use_2d_feat: true
        freeze_quat: true
```

说明：
- `init`：对应 NodeState 初始化阶段使用的参数。
- `limits`：对应 offset 的 `tanh` 上限（即原 `offset_max/scale_max/...`）。
- `eta`：对应渲染参数更新步长（即原 `eta_*`）。
- `mlp`：用于区分分支网络输入来源与结构选项，尤其是 `distant.use_3d_feat=false`。

补充（初始化统一规则）：
- `bg` 与 `distant` 的 `quat` 默认都初始化为单位四元数 `[1,0,0,0]`，不提供配置项。
- `scale` 默认使用各向同性球：`scales_log=[s,s,s]`（由 `isotropic_log_value` 指定 `s`）。
- `scale` 可切换到 `knn` 方案：先算邻域尺度，再转 `scales_log`（可通过 `knn_log_scale_bias` 做微调）。

### 2.1.1 distant 参数默认值策略（精细化）

- `means`：强约束（`eta/offset_max` 明显小于 bg）。
- `quats`：更强约束，默认冻结（`freeze_quat=true`）。
- `scales`：保留中等自由度，不与几何一起极限弱化。
- `opacity`：比几何重要，但受控（避免成为误差“垃圾桶”）。
- `color/SH`：优先 `sh_dc`，`sh_rest` 保守，避免远景方向性过拟合。

### 2.2 3DGS 初始化参数分离

在 `_get_or_init_node_states_bg_distant()` 路径中，拆分：
- `bg` 初始化使用 `branches.bg.init.*`
- `distant` 初始化使用 `branches.distant.init.*`

需要分离的关键项：
- 初始尺度相关（`scale_init.mode`、`isotropic_log_value`、`knn_*`）
- 初始 opacity logit（由 `opacity_init` 映射）
- quat 初始化固定为单位四元数（bg/distant 一致，不配置）
- 可选：对 distant 的 means clamp / input_aabb 约束策略（建议仍保留 input_aabb）

### 2.3 offset/eta 参数分离

在偏移预测与渲染参数构建中，将当前单套参数改为分支参数：

- `bg` 分支：`limits_bg.*` + `eta_bg.*`
- `distant` 分支：`limits_distant.*` + `eta_distant.*`

涉及函数（建议）：
- `_predict_offsets_gru(...)` 或其包装器：接收 `branch_cfg`（bg/distant）。
- `_render_params_from_offsets(...)`：改为可注入分支 `eta`。
- `_render_params_from_offsets_distant(...)`：明确使用 distant 专属 `eta/limits`。

### 2.4 distant 不使用 3D feat，MLP 独立

当前 Stage 3.2 中 distant 通过：
- `zeros_3d + feat_2d_distant + vis -> _fuse_features(...)`
- 再进入与 bg 共用的 offsets 头

Stage 3.3 建议改为“结构上分离”：

1. `distant` 输入特征仅来源于 2D（可加参数 embedding + hidden state）
2. `distant` 使用独立 offsets heads（至少独立最后预测头）：
   - `mlp_offset_pos_distant`
   - `mlp_conv_distant`
   - `mlp_opacity_distant`
   - `gaussion_decoder_distant`
3. `bg` 继续走 3D(+2D) 融合路径与原 heads，互不共享参数

这样可以避免：
- 用全零 3D 特征“伪装”输入导致的分布偏移；
- distant 与 bg 争用同一偏移头容量。

## 3. 代码改造建议（最小侵入）

## 3.1 配置与校验层

- 在 Stage3_3 trainer 初始化时新增严格校验：
  - 必须存在 `model.branches.bg` 与 `model.branches.distant`
  - 两分支必须同时提供 `init/limits/eta/mlp` 必要字段
- 不兼容老字段时直接报错（fast-fail），避免 silent fallback。

## 3.2 数据结构层

- 新增分支配置容器（可用 dataclass）：
  - `BranchLimitsConfig`
  - `BranchEtaConfig`
  - `BranchInitConfig`
  - `BranchModelConfig`
- Trainer 内缓存为 `self.bg_cfg` / `self.distant_cfg`，避免频繁字典取值。
- `BranchInitConfig` 建议包含：
  - `scale_init_mode: Literal["isotropic","knn"]`
  - `scale_isotropic_log_value: float`
  - `scale_knn_k: int`
  - `scale_knn_log_scale_bias: float`
  - `opacity_init: float`
  - （不包含 quat 初值；quat 始终单位初始化）

## 3.3 前向流程层（基于 stage3_2）

- 保持 `node_state_bg, node_state_distant = _get_or_init_node_states_bg_distant(batch)` 不变。
- 改造 feature 构建：
  - `bg`：维持 3D + 2D 融合。
  - `distant`：跳过 3D 特征分支，仅走 2D 特征（若无 2D 则 fast-fail 或显式禁用 distant 更新）。
- 改造 offsets 预测：
  - `bg` 调用 `predict_offsets_bg(...)`
  - `distant` 调用 `predict_offsets_distant(...)`
- 改造 render params：
  - `render_params_bg = render_params_from_offsets_bg(...)`
  - `render_params_distant = render_params_from_offsets_distant(...)`

## 3.4 网络定义层

- 新增 distant 专属 heads（可先复制 bg 头结构，参数独立）：
  - 先保持网络宽度一致，减少变量数量。
  - 后续再调 distant hidden_dim。
- 若保留 GRU-style，可共享 `params_embed` 与 `gru`，但建议把最后 offset heads 分离；
  若希望完全解耦，可连 `gru_*` 一并分离（第二阶段再做）。

## 3.5 Checkpoint 与兼容策略

- 新增参数名将导致 ckpt key 变化，建议：
  - Stage3_3 明确不加载 stage3_2 旧头（strict=False + 日志提示）。
  - 文档说明“stage3_2 -> stage3_3 需重新训练或仅部分 warm-start”。

## 4. 实现计划（分阶段）

### Phase A：配置与常量解耦（低风险）

1. 新增 stage3_3 配置文件（例如 `configs/minimal_streetforward_stage3_3.yaml`）。
2. 实现 `model.branches.{bg,distant}` 解析与严格校验。
3. 在 NodeState 初始化中接入 `init` 分支参数：
   - quat 固定单位初始化；
   - scale 支持 `isotropic/knn` 双模式。
4. 在 render params 计算中接入 `eta` 分支参数。

验收：
- 能正常跑通 forward；
- 日志可打印出 bg/distant 生效参数；
- 缺字段会立即报错。

### Phase B：offset limits 与 heads 解耦（中风险）

1. 将 `offset_max/scale_max/...` 改为按分支读取。
2. 新增 distant 独立 offsets heads，接入 forward。
3. 保持 loss 与 proxy 渲染逻辑不变，先保证行为稳定。

验收：
- distant 分支参数梯度仅流向 distant heads；
- bg heads 参数量与梯度统计不受影响；
- 训练可稳定下降。

### Phase C：distant 无 3D feat 路径收敛（中高风险）

1. 删除 distant 分支中的 `zeros_3d` 伪输入路径。
2. 用 2D 特征（+可选 param embedding/h）直接输入 distant heads。
3. 对齐 batch/multi-view 下的维度与空 distant 边界处理。

验收：
- `distant.use_3d_feat=false` 时无 3D feat 依赖；
- 有/无 distant 点两种 batch 都可运行；
- 不引入 shape mismatch 与 NaN。

### Phase D：训练与回归验证（必要）

1. 使用 overfit batch 做 A/B（stage3_2 vs stage3_3）：
   - loss 下降趋势
   - 可视化质量（近景/bg 与远景/distant 分别观察）
2. 关注指标：
   - mask/opacity entropy 稳定性
   - distant 点 opacity 饱和或塌缩风险
3. 必要时加轻量日志：
   - `offset_norm_bg` vs `offset_norm_distant`
   - `eta` 作用后的参数更新幅度统计

## 5. 风险与对应策略

- 风险1：分支参数过多导致配置复杂度增加  
  - 策略：提供最小必填模板，严格 schema 校验。
- 风险2：distant 无 3D feat 后表达能力不足  
  - 策略：先保留 2D+param embedding，再考虑增加浅层 MLP 宽度。
- 风险3：旧 checkpoint 兼容性下降  
  - 策略：在 stage3_3 文档中明确“部分 warm-start”策略和不可兼容项。
- 风险4：训练初期 distant 波动大  
  - 策略：先把 distant 的 `limits/eta` 设置为更保守值（小于 bg）。

## 6. 建议的首版落地范围（MVP）

建议先做以下最小闭环：

1. 配置拆分（`init/limits/eta/mlp`）+ 严格校验；
2. distant 独立 offsets heads；
3. distant 去除 3D feat（仅 2D 输入）；
4. 仅支持从 stage3_3 配置启动训练，不做旧配置自动兼容。

该范围能直接验证你提出的三个核心诉求，同时改动面可控，便于快速迭代。
