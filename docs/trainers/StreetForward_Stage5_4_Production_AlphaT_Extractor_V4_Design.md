# StreetForward Stage 5.4 Production + AlphaTExtractorV4 设计文档

## 文档目标

本文定义 `Stage 5.4 Production` 的正式落地方案。核心原则：

- **不改变 feature aggregation 语义**
- 在 fused CUDA backprojection 中额外统计 **current observation code**
- 将 observation code 作为条件输入注入：
  - near 3D conv / xCPE token
  - far MLP token
  - GRU input
  - history gate input

第一版严格控制变量，验证 `current observation code` 是否能缓解 multi-view feature mixing 导致的错误更新。

---

## 0. 设计背景与问题定义

当前 2D-to-Gaussian feature lifting 使用多视角线性聚合：

```text
f_i^{2D} = sum_{v,p}(w_{ivp} * F_v(p)) / (sum_{v,p} w_{ivp} + eps)
```

当单个 Gaussian 被多个 source camera 同时观测时，不同视角 feature 被压缩为一个向量，update network 难以区分：

```text
主视角明确观测
vs
多视角混合观测
```

`Stage 5.4` 不拆分 per-view feature，只补充低成本观测描述 `o_i`。

---

## 1. Stage 5.4 相对 Stage 5.3 的变化

### 1.1 保持不变

以下语义保持与 `Stage 5.3` 一致：

1. one-pass fused multi-camera backprojection 不变
2. `feat_sum / weight_sum_feature` 定义不变
3. `weight_sum_support` 定义不变
4. `pair_valid_mask` 继续用于 sky / ego / invalid source 过滤
5. rigid 仍在 source-frame world 中参与 source backprojection
6. loss、scheduler、history memory EMA 逻辑不变
7. obs_code 不写入 history memory
8. obs_code 不参与 rasterization / visibility 反传梯度

### 1.2 新增内容

- fused CUDA 输出新增：

```text
obs_code: [N, 2]
```

- fused CUDA 中间 buffer 新增：

```text
weight_sum_view_feature: [V, N]
```

其中：

```text
N = N_bg + N_distant + N_rigid_source_world
```

---

## 2. Current Observation Code 定义

固定公式：

```text
rho_iv^(k) = sum_p 1[w_ivp^(k) >= tau] * w_ivp^(k)
rho_i^(k)  = sum_v rho_iv^(k)
u_i^overlap,(k) = (rho_i^(k) - max_v rho_iv^(k)) / (rho_i^(k) + eps)
o_i^(k) = [ log(1 + rho_i^(k)), u_i^overlap,(k) ]
```

语义：

- `log(1 + rho_i)`：当前点观测强度
- `u_i^overlap`：非主导相机贡献占比

约束：

- `tau` 复用已有 `weight_threshold`
- 不新增 valid 阈值
- 不统计 `n_obs`
- 不做 overlap normalize

典型样例：

```text
rho_v=[1.0,0.0]       -> overlap=0
rho_v=[0.9,0.1]       -> overlap~0.1
rho_v=[0.6,0.4]       -> overlap~0.4
rho_v=[0.5,0.5]       -> overlap~0.5
rho_v=[0.34,0.33,0.33]-> overlap~0.66
rho_v=[0,0,0]         -> overlap=0
```

---

## 3. CUDA / gsplat 修改方案

## 3.1 文件范围

建议新增/修改：

```text
third_party/gsplat/gsplat/cuda/csrc/Rasterization.cu
third_party/gsplat/gsplat/cuda/csrc/Rasterization.h
third_party/gsplat/gsplat/cuda/csrc/Ops.cpp
third_party/gsplat/gsplat/cuda/_wrapper.py
models/feature_extractors/alpha_t_extractor_v4.py
```

### 3.2 主 kernel 增加 per-view feature weight

在 `vis >= weight_threshold` 分支内新增：

```cpp
if (weight_sum_view_feature != nullptr) {
    atomicAdd(
        &weight_sum_view_feature[
            static_cast<uint64_t>(cam_id) * num_gaussians + g_global
        ],
        vis
    );
}
```

注意：必须放在 `vis >= weight_threshold` 内，以保证 `rho` 与 feature aggregation 语义一致。

### 3.3 主 kernel signature 扩展

新增参数：

```text
float* weight_sum_view_feature  // [V, N]
uint32_t num_gaussians
```

### 3.4 新增 reduce kernel

新增 `compute_current_obs_code_kernel(...)`：

- 输入：`weight_sum_view_feature [V, N]`
- 输出：`obs_code [N, 2]`
- 计算：`rho`、`max_rho`、`log1p(rho)`、`overlap`
- `overlap` clamp 到 `[0, 1]`

### 3.5 C++ wrapper 新增函数

新增导出函数：

```text
rasterize_and_backproject_multi_camera_obs_in_range(...)
```

返回：

```text
feat_sum, weight_sum_feature, weight_sum_support, obs_code, pair_count_total, pair_count_threshold
```

### 3.6 Python binding

在 `gsplat.cuda._wrapper` 暴露：

```text
rasterize_and_backproject_multi_camera_obs_in_range
```

---

## 4. AlphaTWeightExtractorV4 设计

### 4.1 Autograd Function

新增 `_RasterizeAndBackprojectFeatObsMultiCamFn`：

- `forward` 调用 `rasterize_and_backproject_multi_camera_obs_in_range`
- `backward` 仅对 `feat2d` 回传梯度（沿用 V3 语义）
- `obs_code`、`w_feat`、`w_sup`、`pairs` 全部 `mark_non_differentiable`

### 4.2 Extractor 类

新增 `AlphaTWeightExtractorV4(AlphaTWeightExtractorV3)`：

- fast-fail：
  - 若 fused v4 op 不可用则直接报错
  - `features_2d` 必须是 `[V, Hf, Wf, C]`
- 支持返回：
  - `feat_out`
  - `weight_sum_support`（可选）
  - `obs_code`（可选）
  - `debug_stats`（可选）

建议 debug 指标：

- `pairs_total`
- `pairs_after_threshold`
- `obs_rho_log_mean/p95`
- `obs_overlap_mean/p95`
- `obs_overlap_nonzero_ratio`

---

## 5. Stage 5.4 模型侧改造

### 5.1 新增 trainer 文件

```text
models/streetforward/minimal_trainer_stage5_4.py
models/streetforward/minimal_trainer_stage5_4_production.py
```

建议继承（强约束）：

```text
MinimalStreetForwardStage5_3 -> MinimalStreetForwardStage5_4
MinimalStreetForwardStage5_4 -> MinimalStreetForwardStage5_4_Production
```

说明：

- `Stage5_4` 采用 additive conditioning（`feat += ObsEmbed(obs)`）而非 concat 扩维
- `current_observation.dim` 固定为 2，当前实现不使用独立 `embed_dim` 配置
- 可以复用 `Stage5_3` 主干输入维度，不需要同步扩大 near/far/GRU/gate 的输入通道

推荐模式：

```text
class MinimalStreetForwardStage5_4(MinimalStreetForwardStage5_3):
    def _build_stage5_modules(...):
        # 先解析 obs cfg（input_to_* 开关）
        # 再构建 struct/far/gru/gate 的 ObsEmbed 注入模块
```

### 5.2 配置解析与 fast-fail

新增 `_get_current_observation_cfg()` 与 `_validate_stage5_4_config()`，要求：

- `current_observation.enable=true`
- `current_observation.dim=2`
- `current_observation.rho_source=feature`
- `current_observation.record_to_history_memory=false`
- `model.use_fused_cuda_backproject_v4=true`

### 5.3 Observation Embedding

建议独立四套 embedding（第一版不共享参数）：

```text
current_obs_struct_embed
current_obs_far_embed
current_obs_gru_embed
current_obs_gate_embed
```

推荐结构：

```text
Linear(2, D) -> LayerNorm(D) -> GELU
```

### 5.4 Backprojection 输出接入

V4 调用返回：

```text
feat_2d_all, acc_w_all, obs_code_all, bp_stats
```

按 `bg / distant / rigid_source_world` 顺序切分。  
`rigid` 后续 inside/outside routing 时，`obs_rigid` 必须与同一索引子集同步切分，不可重算。

### 5.5 接入点

- near branch token：add `obs_struct` residual
- far MLP input：add `obs_far` residual
- GRU input：add `obs_gru` residual
- history gate input：add `obs_gate` residual

### 5.6 Extractor 选型与 fast-fail（必须显式实现）

`Stage 5.4` 不能仅通过配置 `use_fused_cuda_backproject_v4=true` 隐式生效，必须改模型初始化中的 extractor factory：

```python
if cfg.model.backprojector_version == "v4":
    self.alpha_t_extractor = AlphaTWeightExtractorV4(...)
elif cfg.model.backprojector_version == "v3":
    self.alpha_t_extractor = AlphaTWeightExtractorV3(...)
else:
    raise ValueError("Unsupported backprojector_version")
```

若项目暂时沿用 `v3/v4` 双 bool，也必须显式 fast-fail：

```python
if use_fused_cuda_backproject_v4:
    assert isinstance(self.alpha_t_extractor, AlphaTWeightExtractorV4)
```

并在 forward 调用中显式请求：

```text
return_obs_code=True
```

否则即使配置打开 v4，也可能回退到 v2/v3 路径，导致拿不到 `obs_code`。

---

## 6. Stage5_4_Production 设计要点

`MinimalStreetForwardStage5_4_Production` 目标：

- 继承 `MinimalStreetForwardStage5_4`（保证主干网络维度已按 obs 构建）
- 复用 Stage5_3 生产训练语义（AdamW、WarmCos、grad clip、bad-step 检查、resume 语义）
- 开启 fused v4 backprojection + obs_code
- 在 5_4 模块构建完成后初始化/重建 optimizer/scheduler
- 启动阶段打印 obs 配置日志，便于线上确认

实现建议：

- 将 5_3 production 中与训练工程相关的逻辑（optimizer/scheduler/checkpoint/bad-step）抽成可复用 mixin 或 helper
- `MinimalStreetForwardStage5_4_Production` 直接基于 `MinimalStreetForwardStage5_4` 组装生产逻辑
- 避免 `MinimalStreetForwardStage5_3_Production -> ...` 的继承链导致 5_3 旧维度模块提前实例化

---

## 7. 训练脚本

新增：

```text
tools/train_minimal_streetforward_stage5_4_production_multi_scene_v8.py
```

实现策略：复制 `5_3_production` 脚本，仅替换：

- 模型类：`MinimalStreetForwardStage5_4_Production`
- 默认配置：`configs/minimal_streetforward_stage5_4_production_multi_scene_v8.yaml`

运行方式：

```bash
PYTHONPATH=/root/drivestudio-coding \
python tools/train_minimal_streetforward_stage5_4_production_multi_scene_v8.py \
  --config_file configs/minimal_streetforward_stage5_4_production_multi_scene_v8.yaml
```

---

## 8. 配置文件设计

新增：

```text
configs/minimal_streetforward_stage5_4_production_multi_scene_v8.yaml
```

核心差异（推荐单字段版本选择）：

```yaml
model:
  stage: "5_4"
  production_training: true
  backprojector_version: "v4"
  fused_cuda_backproject_v4_force_fallback: false

current_observation:
  enable: true
  dim: 2
  rho_source: feature
  eps: 1.0e-6
  input_to_struct_decoder: true
  input_to_far_mlp: true
  input_to_gru: true
  input_to_history_gate: true
  record_to_history_memory: false
```

优化器建议：

- 第一版可直接落入 `default` group（最小改动）
- 正式 production 可增加 `current_observation` group

兼容旧配置（过渡期）：

- 若保留 `use_fused_cuda_backproject_v3/v4`，需增加一致性检查：
  - `v4=true` 且 `v3=false`
  - 并在模型构建后 `assert extractor is V4`

---

## 9. 测试计划

新增测试：

```text
tests/test_alpha_t_extractor_v4_obs.py
```

建议 case：

1. **单相机**
   - overlap 全 0
   - `expm1(obs[:,0]) ~= weight_sum_feature`
2. **双相机均衡贡献**
   - overlap 约 0.5
3. **双相机主导贡献**
   - overlap 约次视角占比（如 0.1）
4. **无观测点**
   - `obs=[0,0]`

训练 ablation 顺序：

```text
A: 5_3 baseline
B: obs -> gate
C: obs -> struct/far
D: obs -> struct/far + GRU
E: obs -> struct/far + GRU + gate
```

预期：

```text
E >= D > C > B
```

---

## 10. 实施边界（第一版必须遵守）

第一版严格保持：

1. feature aggregation 不变
2. obs_code 仅作条件输入
3. obs_code stop-gradient
4. obs_code 不写入 history memory
5. `rho_source = feature`

该约束用于确保 Stage 5.4 的核心变量单一、可归因、易回滚。
