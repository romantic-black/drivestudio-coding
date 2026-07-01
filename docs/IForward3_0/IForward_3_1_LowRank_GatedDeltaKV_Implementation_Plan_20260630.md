# IForward 3_1：Low-rank Gated Delta KV Optimizer Memory 实现方案

生成日期：2026-06-30  
适用项目：StreetForward / DriveStudio / IForward Stage 3_1  
目标：替换 Stage 3_0 的 `ParentOptimizerMamba`，实现一个可每步 read/write、低显存、支持 repair 乱序访问的迭代优化记忆系统。

---

## 0. 约束与结论

本方案只实现 **Low-rank Gated Delta KV**，不再讨论 Kalman、Bayesian filter、三层 memory、GRU-D 或更大的 sequence model。

硬约束：

```yaml
K: 16
V: 32
read_every_repeat: true
write_every_repeat: true
reset_scope: episode
detach_scope: rollout_boundary
```

核心结论：

1. IForward 3_1 不改变 scheduler 的迭代语义：仍然每个 repeat 都 read/write optimizer memory。
2. IForward 3_1 只替换 memory cell：`ParentOptimizerMamba -> ParentOptimizerGatedDeltaKV`。
3. 新 memory 是 parent-level associative memory，state 为 `S ∈ R^{K×V}`，即每个 parent 维护一个低秩 fast-weight KV 记忆矩阵。
4. 默认维度 `K=16, V=32`，每个 parent state 为 512 floats，略低于当前 Mamba 的约 576 floats，同时去掉 conv state。
5. 防 NaN 不做复杂 fallback，不引入 branch reset / skip-write 作为主机制；只做必要的 RMS normalization / RMS clamp。
6. 3_1 配置中修正 distant scale：`distant.update_scales=true`，`attribute_gates.distant.scales>0`。这是 3_0 distant event spike 的关键语义修复之一。

---

## 1. 背景：为什么 Stage 3_1 要换成 Gated Delta KV

Stage 3_0 的 optimizer memory 是 `ParentOptimizerMamba`。它的状态对每个 parent 保存：

```text
conv_state: [N, model_dim, conv_kernel]
ssm_state:  [N, model_dim, state_dim]
seen / update_count / last_visit metadata
```

当前默认：

```text
model_dim = 32
state_dim = 16
conv_kernel = 2
```

即：

```text
conv_state = 32 × 2  = 64 floats
ssm_state  = 32 × 16 = 512 floats
total      = 576 floats / parent
```

Stage 3_1 的 Gated Delta KV：

```text
S = [K,V] = [16,32] = 512 floats / parent
```

显存不增加，计算量也与 Mamba 的 `model_dim × state_dim` 同一量级。

更重要的是，IForward 的 memory 不是普通序列建模 memory。它服务于迭代优化器：

```text
当前帧观测 -> 更新 3DGS state -> 写入历史优化记忆 -> 后续帧读取历史记忆以减少破坏性更新
```

repair 阶段会 random order，因此 memory 不能把输入顺序过强解释为真实时间。Gated Delta KV 的优势是：

```text
1. memory 是 key-value association，不是单一 hidden state；
2. 写入是 edit / correction，而不是无条件 recurrent 累积；
3. erase gate 和 write gate 可以分离；
4. repair 乱序下，按 key 更新比 causal Mamba 更合理；
5. K×V 固定，显存可控。
```

相关工作中，Fast Weight / Linear Transformer 的 delta rule 将线性注意力解释为有限 fast-weight memory，并用 delta update 修正已有 key-value mapping；Gated DeltaNet-2 进一步把 key-side erase 与 value-side write 解耦。这正好对应 IForward 对“历史约束可编辑但不可随意覆盖”的需求。

---

## 2. 总体架构

### 2.1 新模块名称

新增模块：

```text
models/iforward/stage2_3/parent_optimizer_gated_delta_kv.py
```

主类：

```python
class ParentOptimizerGatedDeltaKV(nn.Module):
    preview(...) -> ParentOptimizerPreview
    write(...) -> Tuple[ParentOptimizerDeltaKVState, Dict[str, float]]
```

保持和 `ParentOptimizerMamba` 一样的外部接口，减少模型主循环改动。

### 2.2 State schema

新增 state schema，建议放在：

```text
models/iforward/stage2_3/optimizer_memory_schema.py
```

新增 dataclass：

```python
@dataclass
class DenseDeltaKVOptimizerState:
    kv_state: torch.Tensor       # [N, K, V]
    seen: torch.Tensor           # [N]
    update_count: torch.Tensor   # [N]
    last_visit_step: torch.Tensor
    last_frame_id: torch.Tensor
    last_visit_kind: torch.Tensor

@dataclass
class KeyedDeltaKVOptimizerState:
    keys: torch.Tensor           # [M]
    kv_state: torch.Tensor       # [M, K, V]
    seen: torch.Tensor           # [M]
    update_count: torch.Tensor
    last_visit_step: torch.Tensor
    last_frame_id: torch.Tensor
    last_visit_kind: torch.Tensor

@dataclass
class DeltaKVOptimizerBranchState:
    dense: Optional[DenseDeltaKVOptimizerState] = None
    keyed: Optional[KeyedDeltaKVOptimizerState] = None

@dataclass
class ParentOptimizerDeltaKVState:
    bg: DeltaKVOptimizerBranchState
    distant: DeltaKVOptimizerBranchState
    rigid: DeltaKVOptimizerBranchState
    global_update_step: int = 0
```

与当前 Mamba state 的分支机制保持一致：

```text
bg      默认 dense
distant 默认 dense
rigid   默认 keyed
```

### 2.3 Branch routing

沿用当前 ParentOptimizerMamba 的 routing：

```text
dense_bg: true
dense_distant: true
rigid: keyed
```

这样不需要改 `ParentTemporalKeysV2`、`build_parent_temporal_keys_v2` 或 BigGS assignment。

---

## 3. Gated Delta KV Cell 设计

### 3.1 维度

```yaml
key_dim: 16       # K
value_dim: 32     # V
ctx_dim: 32
event_dim: 64
write_token_dim: 64
visit_dim: 32
```

每个 parent 的记忆：

```text
S_i ∈ R^{16×32}
```

### 3.2 RMS 工具函数

只使用 RMS norm / RMS clamp 作为数值边界。

```python
def rms(x: torch.Tensor, dim=-1, keepdim=True, eps=1.0e-6):
    return torch.sqrt(torch.mean(x.float() * x.float(), dim=dim, keepdim=keepdim) + eps).to(dtype=x.dtype)


def rms_unit(x: torch.Tensor, dim=-1, eps=1.0e-6):
    return x / rms(x, dim=dim, keepdim=True, eps=eps)


def rms_clamp(x: torch.Tensor, max_rms: float, dims=(-1,), eps=1.0e-6):
    # dims can be -1 for vectors or (-2,-1) for matrices.
    r = torch.sqrt(torch.mean(x.float() * x.float(), dim=dims, keepdim=True) + eps).to(dtype=x.dtype)
    scale = torch.clamp(float(max_rms) / r.clamp_min(eps), max=1.0)
    return x * scale
```

注意：这里不做复杂 non-finite fallback；finite check 只作为 fail-fast debug。

---

## 4. Read / Preview 路径

### 4.1 Read 公式

给定 parent event：

```text
e_i ∈ R^{64}
S_i ∈ R^{16×32}
```

生成 query：

```python
q = q_proj(e_i)              # [N, K]
q = rms_unit(q) / sqrt(K)    # bounded query, approximately ||q||_2 ≈ 1
```

读取 memory：

```python
ctx = einsum('nkv,nk->nv', S, q)   # [N, V=32]
ctx = rms_clamp(ctx, max_rms=ctx_rms_max)
```

unseen parent 的 ctx 置零：

```python
ctx = torch.where(seen[:, None], ctx, torch.zeros_like(ctx))
```

### 4.2 Fusion

沿用当前 Mamba 的 fusion 语义：

```python
contribution = branch_gate * visit_gate * support_gate * adapter(ctx)
fused_event = spatial_event + contribution
```

其中：

```text
adapter: V=32 -> event_dim=64
branch_gate: 每个 branch 一个可学习 gate
visit_gate: bootstrap / assimilation / repair / repeat_stability gate
support_gate: support_mean / (support_mean + 1)
```

保留当前 gate init：

```yaml
bootstrap: 0.0
assimilate: 0.05
repair: 0.03
repeat_stability: 0.03
```

### 4.3 Preview 输出

`preview()` 返回：

```python
ParentOptimizerPreview(
    event=fused_event_pack,
    aux={
        'iforward/parent_optimizer_gdkv/read': 1.0,
        'iforward/parent_optimizer_gdkv/bg_preview_seen_ratio': ...,
        'iforward/parent_optimizer_gdkv/distant_preview_seen_ratio': ...,
        'iforward/parent_optimizer_gdkv/rigid_preview_seen_ratio': ...,
        'iforward/parent_optimizer_gdkv/bg_ctx_rms': ...,
        ...
    }
)
```

为了兼容旧 validation / tensorboard，也可以同时写 legacy key：

```text
iforward/parent_optimizer_mamba/read = 1.0
```

但新文档和新实验应主要看 `parent_optimizer_gdkv/*`。

---

## 5. Write 路径

### 5.1 Write token

继续使用现有 `OptimizerWriteTokenBuilder`：

```python
write_event = write_builder(
    spatial_event=spatial_event,
    fused_event=fused_event,
    visit_bg=visit_bg,
    visit_distant=visit_distant,
    visit_rigid=visit_rigid,
    delta=delta_summary,
)
```

也就是说，Gated Delta KV 的写入输入仍然包括：

```text
spatial_event
fused_event
visit_embedding
delta_summary
support
valid
```

不需要重写 delta summary 逻辑。

### 5.2 Gate / key / value projection

对每个 parent row 的 write token `x`：

```python
k_raw = key_proj(x)       # [N, K]
v_raw = value_proj(x)     # [N, V]
b_raw = erase_proj(x)     # [N, K]
w_raw = write_proj(x)     # [N, V]
d_raw = decay_proj(x)     # [N, K] or [N,1]
```

归一化：

```python
k = rms_unit(k_raw) / sqrt(K)
v = rms_clamp(v_raw, max_rms=value_rms_max)
```

gate：

```python
erase = sigmoid(b_raw + erase_bias) * erase_gate_max
write = sigmoid(w_raw + write_bias) * write_gate_max
decay = decay_min + (1.0 - decay_min) * sigmoid(d_raw + decay_bias)
```

默认建议：

```yaml
erase_gate_max: 1.0
write_gate_max: 1.0
decay_min:
  assimilation: 0.98
  repair: 1.0
  repeat_stability: 1.0
value_rms_max: 2.0
ctx_rms_max: 4.0
state_rms_max: 4.0
```

`decay_min` 可以由 visit embedding 学到，但 repair 阶段建议强制为 1.0，避免 random order 被解释为物理时间衰减。

### 5.3 Gated Delta update

采用 Gated DeltaNet-2 风格的 decoupled erase/write，但只实现单头低秩版。

State：

```text
S ∈ R^{N×K×V}
k ∈ R^{N×K}
v ∈ R^{N×V}
erase ∈ R^{N×K}
write ∈ R^{N×V}
decay ∈ R^{N×K}
```

公式：

```python
S_decay = S * decay[:, :, None]
old = torch.einsum('nkv,nk->nv', S_decay, erase * k)
S_erased = S_decay - torch.einsum('nk,nv->nkv', k, old)
S_write = S_erased + torch.einsum('nk,nv->nkv', k, write * v)
S_new = rms_clamp(S_write, max_rms=state_rms_max, dims=(-2, -1))
```

只对 write mask 为 true 的 row 写回：

```python
S_final = torch.where(write_mask[:, None, None], S_new, S_old)
seen_final = seen | write_mask
```

### 5.4 为什么使用 decoupled erase/write

如果只做：

```python
S = S + outer(k, v - S^T k)
```

erase 与 write 被绑在一起。IForward 中这不理想：

```text
某些当前观测只应该弱写入，但需要较强擦除旧错误 association；
某些 repair 观测只应该补充 value，但不应强擦除旧 memory；
distant scale 修正后，geometry residual 与 appearance residual 的可信度不同。
```

因此 3_1 采用：

```text
erase gate: key-axis old association removal
write gate: value-axis new content commit
```

这比普通 DeltaNet 更适合 iterative optimizer memory。

---

## 6. Dense / Keyed 实现细节

### 6.1 Dense state

对应 bg / distant。

新增 helper：

```python
def _empty_delta_dense(cell, rows, device, dtype) -> DenseDeltaKVOptimizerState

def _ensure_delta_dense(cell, state, rows, device, dtype) -> DenseDeltaKVOptimizerState
```

`_ensure_delta_dense` 行为与当前 `_ensure_dense` 一致：如果 parent row 增加，则 append zero state。

### 6.2 Keyed state

对应 rigid。

新增 helper：

```python
def _empty_delta_keyed(cell, device, dtype) -> KeyedDeltaKVOptimizerState

def _gather_delta_keyed(cell, state, keys, device, dtype) -> Tuple[DeltaKVCellState, meta]

def _scatter_delta_keyed(cell, state, keys, updated, meta, write_mask, device, dtype)
```

完全复用当前 keyed routing 的语义：

```text
keys sorted
searchsorted gather
missing append
scatter back
```

### 6.3 Weighted aggregation

当前 `_weighted_aggregate` 可直接复用。keyed rigid 写入时：

```python
keys_u, inverse, x_u = _weighted_aggregate(x, keys, support)
write_counts.index_add_(0, inverse, write_mask)
write_u = write_counts > 0
```

---

## 7. 新模块类结构

建议类结构：

```python
@dataclass
class DeltaKVCellState:
    kv_state: torch.Tensor   # [N,K,V]
    seen: torch.Tensor       # [N]

class LowRankGatedDeltaKVCell(nn.Module):
    def __init__(
        self,
        event_dim: int,
        token_dim: int,
        key_dim: int = 16,
        value_dim: int = 32,
        ctx_dim: int = 32,
        hidden_dim: int = 64,
        value_rms_max: float = 2.0,
        ctx_rms_max: float = 4.0,
        state_rms_max: float = 4.0,
    ):
        ...

    def init_state(self, rows, device, dtype) -> DeltaKVCellState:
        ...

    def read(self, event: torch.Tensor, state: DeltaKVCellState) -> Tuple[torch.Tensor, Dict[str,float]]:
        ...

    def write(self, token: torch.Tensor, state: DeltaKVCellState, write_mask: torch.Tensor, visit_meta=None):
        ...
```

Wrapper：

```python
class ParentOptimizerGatedDeltaKV(nn.Module):
    def preview(...):
        # branch read + fusion

    def write(...):
        # build write token + branch write
```

---

## 8. IForwardModel 接入

### 8.1 Version

新增 version：

```yaml
model:
  iforward:
    version: stage3_1_lowrank_gated_delta_kv_lift
```

更新：

```text
models/iforward/versions.py
```

新增：

```python
def is_stage3_1_iforward_version(version: str) -> bool:
    return str(version) in {
        'stage3_1_lowrank_gated_delta_kv_lift',
        'iforward_stage3_1_lowrank_gated_delta_kv_lift',
    }
```

在 `IForwardModel.__init__`：

```python
self.is_stage3_1_lowrank_gdkv = is_stage3_1_iforward_version(self.iforward_version)
self.is_stage2_3_optimizer_mamba = ... or self.is_stage3_0_full_sparse_gather_lift or self.is_stage3_1_lowrank_gdkv
```

短期为了少改主循环，可以继续复用 `is_stage2_3_optimizer_mamba` 这个变量，但文档和日志中应称为 optimizer memory，不再称为 mamba。

### 8.2 Module construction

当前代码：

```python
self.parent_temporal_mamba = ParentOptimizerMamba(...)
```

3_1 改成：

```python
memory_type = str(cfg_get(parent_optimizer_cfg, 'type', 'mamba'))
if memory_type in {'lowrank_gated_delta_kv', 'gated_delta_kv'}:
    self.parent_temporal_mamba = ParentOptimizerGatedDeltaKV(
        event_dim=...,
        token_dim=...,
        ctx_dim=...,
        key_dim=16,
        value_dim=32,
        dense_bg=...,
        dense_distant=...,
        gate_init=...,
        ...
    )
else:
    self.parent_temporal_mamba = ParentOptimizerMamba(...)
```

保留属性名 `parent_temporal_mamba` 是为了避免大范围修改 forward path；但新增 aux/log key 使用 `parent_optimizer_gdkv`。

### 8.3 State 初始化

当前：

```python
if self.is_stage2_3_optimizer_mamba:
    parent_temporal = ParentOptimizerMambaState.empty()
```

改成：

```python
if self.is_stage2_3_optimizer_mamba:
    parent_temporal = self.parent_temporal_mamba.empty_state()
```

给 `ParentOptimizerMamba` 和 `ParentOptimizerGatedDeltaKV` 都补：

```python
@staticmethod
def empty_state():
    return ParentOptimizerMambaState.empty()
```

GDKV：

```python
@staticmethod
def empty_state():
    return ParentOptimizerDeltaKVState.empty()
```

### 8.4 State type check

当前 forward 中强判断：

```python
if self.is_stage2_3_optimizer_mamba and not isinstance(parent_temporal_state, ParentOptimizerMambaState):
    parent_temporal_state = ParentOptimizerMambaState.empty()
```

改成：

```python
if self.is_stage2_3_optimizer_mamba:
    expected_cls = getattr(self.parent_temporal_mamba, 'state_cls', ParentOptimizerMambaState)
    if not isinstance(parent_temporal_state, expected_cls):
        parent_temporal_state = self.parent_temporal_mamba.empty_state()
        state.parent_temporal = parent_temporal_state
```

### 8.5 Shuffle ablation

现有 `mamba_shuffle_state` 用于 validation。3_1 不改 validation 协议名称，直接把它解释为 memory state shuffle。

对 DeltaKV dense：

```python
kv_state = dense.kv_state.index_select(0, order)
seen = seen.index_select(0, order)
...
```

对 keyed：

```python
keys 保持不变
kv_state / seen / metadata 打乱
```

这与当前 Mamba keyed shuffle 语义一致：打乱 value/state，不打乱 key，用于验证 memory 是否真的绑定 parent identity。

---

## 9. Optimizer group 与 LR

当前 trainer 会把 `parent_temporal_mamba` 的参数拆成：

```text
parent_temporal_mamba: 非 adapter 参数
parent_temporal_adapter: 名字包含 .adapters. 的参数
```

3_1 继续保留：

```python
self.adapters = nn.ModuleDict({...})
```

这样 optimizer group 不需要改。

建议 LR：

```yaml
optimizer:
  lr:
    parent_temporal_mamba: 1.0e-4
    parent_temporal_adapter: 1.0e-4
```

也可以新增别名：

```yaml
parent_optimizer_gdkv: 1.0e-4
parent_optimizer_gdkv_adapter: 1.0e-4
```

但为了最小改动，第一版继续沿用旧 group 名。

---

## 10. Config：IForward 3_1 基准配置

新增配置：

```text
configs/iforward/iforward_stage3_1_lowrank_gated_delta_kv.yaml
```

建议继承 Stage 3_0 full train 配置，只覆盖以下字段：

```yaml
output_name: iforward_stage3_1_lowrank_gated_delta_kv_30k_assim_30k_repair

model:
  iforward:
    version: stage3_1_lowrank_gated_delta_kv_lift

    parent_optimizer_mamba:
      enable: false

    parent_optimizer_memory:
      enable: true
      type: lowrank_gated_delta_kv
      event_dim: 64
      ctx_dim: 32
      key_dim: 16
      value_dim: 32
      token_dim: 64
      dense_bg: true
      dense_distant: true
      read_every_repeat: true
      write_every_repeat: true
      reset_scope: episode
      detach_scope: rollout_boundary
      write_mask:
        support_min: 0.001
        require_valid: true
      write_token:
        include_spatial_event: true
        include_parent_event: true
        include_delta_summary: true
        include_visit_embedding: true
      visit_embedding:
        output_dim: 32
        kinds: [bootstrap, assimilation, repair, repeat_stability]
        repeat_idx_max: 16
        repeat_budget_max: 16
      gated_delta_kv:
        K: 16
        V: 32
        query_rms_unit: true
        key_rms_unit: true
        value_rms_max: 2.0
        ctx_rms_max: 4.0
        state_rms_max: 4.0
        erase_gate_max: 1.0
        write_gate_max: 1.0
        decay_min:
          bootstrap: 1.0
          assimilate: 0.98
          repair: 1.0
          repeat_stability: 1.0
      fusion:
        gate_init:
          bootstrap: 0.0
          assimilate: 0.05
          repair: 0.03
          repeat_stability: 0.03

  stage6_0:
    posterior_updater:
      appearance_detail:
        attribute_gates:
          distant:
            means: 0.0
            scales: 0.003
            quat: 0.0
            opacity: 0.1
            sh: 0.1
      branch_scope:
        distant:
          enable: true
          update_means: false
          update_scales: true
          update_quat: false
          update_opacity: true
          update_sh: true

logging:
  project: iforward_stage3_1_lowrank_gated_delta_kv
  log_dir: /root/autodl-tmp/outputs/stage3_1_lowrank_gated_delta_kv
```

说明：

- `parent_optimizer_mamba.enable=false` 是语义清晰；代码可先读取 `parent_optimizer_memory`，否则 fallback 到旧字段。
- 第一版只打开 distant scale，不打开 means / quat。
- scheduler 不改，仍然使用 Stage 3_0 的 scheduler_v3 / repair protocol。

---

## 11. Logging / metrics

新增核心 metrics：

```text
iforward/parent_optimizer_gdkv/read
iforward/parent_optimizer_gdkv/write
iforward/parent_optimizer_gdkv/{branch}_written
iforward/parent_optimizer_gdkv/{branch}_preview_seen_ratio
iforward/parent_optimizer_gdkv/{branch}_state_rms_mean
iforward/parent_optimizer_gdkv/{branch}_state_rms_max
iforward/parent_optimizer_gdkv/{branch}_ctx_rms_mean
iforward/parent_optimizer_gdkv/{branch}_ctx_rms_max
iforward/parent_optimizer_gdkv/{branch}_key_rms_mean
iforward/parent_optimizer_gdkv/{branch}_value_rms_mean
iforward/parent_optimizer_gdkv/{branch}_erase_gate_mean
iforward/parent_optimizer_gdkv/{branch}_write_gate_mean
iforward/parent_optimizer_gdkv/{branch}_decay_mean
```

兼容旧 dashboard：

```text
iforward/parent_optimizer_mamba/read = 1.0
iforward/parent_optimizer_mamba/write = 1.0
iforward/stage2_3_parent_optimizer_mamba = 1.0
```

但建议同时写：

```text
iforward/stage3_1_parent_optimizer_gdkv = 1.0
```

---

## 12. Validation / Ablation 兼容

现有 validation_v3 使用：

```yaml
mamba_ablation:
  - full
  - mamba_off
  - mamba_read_only
  - mamba_read_write
  - mamba_shuffle_state
  - mamba_freeze_write
```

3_1 不必马上改 validation 配置。解释为：

```text
mamba_off          -> optimizer_memory_off
mamba_read_only    -> gdkv_read_only
mamba_read_write   -> gdkv_read_write
mamba_shuffle_state -> shuffle GDKV state rows
mamba_freeze_write -> read enabled, write disabled
```

后续可重命名为：

```yaml
memory_ablation:
  - full
  - memory_off
  - memory_read_only
  - memory_read_write
  - memory_shuffle_state
  - memory_freeze_write
```

第一版为了减少动线，不改 validation schema。

---

## 13. 单元测试设计

新增：

```text
tests/iforward/test_parent_optimizer_gated_delta_kv.py
```

### 13.1 Shape test

```python
N = 128
cell = LowRankGatedDeltaKVCell(event_dim=64, token_dim=64, key_dim=16, value_dim=32)
state = cell.init_state(N, device, dtype)
ctx, aux = cell.read(event, state)
next_state, aux = cell.write(token, state, write_mask)
assert ctx.shape == (N, 32)
assert next_state.kv_state.shape == (N, 16, 32)
```

### 13.2 Mask test

```python
write_mask[:64] = True
write_mask[64:] = False
state_new = write(...)
assert changed(state_new.kv_state[:64])
assert equal(state_new.kv_state[64:], state_old.kv_state[64:])
```

### 13.3 Dense grow test

```python
state rows = 100
requested rows = 128
ensure_dense -> 128 rows
old first 100 preserved
new 28 zero
```

### 13.4 Keyed gather/scatter test

```python
keys = [10, 5, 10, 7]
weighted_aggregate -> unique [5,7,10]
write keyed
next gather same keys -> seen hit
```

### 13.5 RMS stability test

```python
token = torch.randn(N,64) * 100
write 100 times
assert torch.isfinite(state.kv_state).all()
assert state_rms <= state_rms_max + eps
```

### 13.6 Repair order smoke test

```python
same set of 8 write tokens
order A vs order B
compare state_rms and ctx_rms, not exact equality
assert finite and bounded
```

注意：Gated Delta KV 不承诺完全 permutation invariant，只要求 repair random order 不产生数值崩坏，且 validation 上 order robustness 改善。

---

## 14. 训练 smoke test

### 14.1 One segment / 500 step

目标：确认 forward/backward/optimizer group/logging 正常。

检查：

```text
iforward/stage3_1_parent_optimizer_gdkv = 1
parent_optimizer_gdkv/read = 1
parent_optimizer_gdkv/write = 1
state_rms finite
ctx_rms finite
distant scale gate > 0
```

### 14.2 2k step assimilation

目标：确认 current PSNR 能启动。

对比 Stage 3_0：

```text
current_psnr 不应明显低于 3_0
loss_current 应下降
gdkv preview_seen_ratio 应逐步上升
```

### 14.3 20k step assimilation

目标：确认原 16k 附近 distant event spike 不再复现。

关注：

```text
fine_event_norm_distant
grad/parent_temporal_mamba 或 parent_optimizer_gdkv
parent_optimizer_gdkv/distant_state_rms_max
```

### 14.4 30k+ repair smoke

目标：确认进入 repair 后 random order + write 不崩。

关注：

```text
repair_flag=true
scheduler_phase=repair
repair random_order=true
parent_optimizer_gdkv/write=1
history_damage
repeat_stability
order_robustness
```

---

## 15. Checkpoint / resume 策略

Stage 3_1 memory state 与 Stage 3_0 Mamba state 不兼容。

建议：

1. 模型权重从头训，或者只加载非 memory 部分。
2. 如果从 Stage 3_0 checkpoint 初始化：
   - 跳过 `parent_temporal_mamba.cells.*`
   - 跳过 `parent_temporal_mamba.adapters.*` 可选；如果 adapter 结构同名但语义不同，建议不加载。
   - 加载 parent_spatial、child_decoder、posterior_updater、2D frontend。
3. carried optimizer memory state 必须 reset，不能从 Mamba state 转换。

配置建议：

```yaml
initialization:
  phase_b_from_phase_a:
    weights_only: true
    reject_plain_model_state_dict: true
  skip_keys:
    - parent_temporal_mamba
    - parent_optimizer_mamba
    - parent_optimizer_memory
```

如果当前加载器没有 `skip_keys`，实现 checkpoint filter。

---

## 16. 实现步骤清单

### Step 1：新增 state schema

文件：

```text
models/iforward/stage2_3/optimizer_memory_schema.py
```

新增：

```text
DenseDeltaKVOptimizerState
KeyedDeltaKVOptimizerState
DeltaKVOptimizerBranchState
ParentOptimizerDeltaKVState
```

并导出到：

```text
models/iforward/stage2_3/__init__.py
```

### Step 2：新增 Gated Delta KV 模块

文件：

```text
models/iforward/stage2_3/parent_optimizer_gated_delta_kv.py
```

包含：

```text
rms / rms_unit / rms_clamp
DeltaKVCellState
LowRankGatedDeltaKVCell
ParentOptimizerGatedDeltaKV
_dense helper
_keyed helper
_weighted_aggregate reuse
```

可以从 `parent_optimizer_mamba.py` 复制 routing / preview / write 骨架，只替换 cell state 和 read/write 内核。

### Step 3：IForwardModel 接入

文件：

```text
models/iforward/model.py
```

修改：

```text
import ParentOptimizerGatedDeltaKV, ParentOptimizerDeltaKVState
version recognition
module construction
state init
state type check
shuffle ablation
```

### Step 4：trainer 兼容

文件：

```text
models/iforward/trainer.py
```

第一版不改 optimizer group，只确保新模块参数仍挂在 `self.parent_temporal_mamba`，并保留 `.adapters.` 命名。

### Step 5：新增配置

文件：

```text
configs/iforward/iforward_stage3_1_lowrank_gated_delta_kv.yaml
```

覆盖：

```text
model.iforward.version
model.iforward.parent_optimizer_memory
model.stage6_0.posterior_updater.distant scale gate / branch_scope
logging.project / log_dir
```

### Step 6：Validation alias

保留现有 `mamba_ablation`，只在模型内部 alias 到 optimizer memory 行为。

### Step 7：测试

新增单元测试 + one segment smoke。

---

## 17. 风险与取舍

### 17.1 它不是完全 order-invariant

Gated Delta KV 是 recurrent edit memory，不是 set average。repair random order 下不会完全交换。但相比 Mamba：

```text
Mamba: hidden state transition 强顺序依赖
GDKV: key-value association edit，顺序依赖较弱且更可控
```

因此它适合作为 IForward 3_1，而不是最终的数学 permutation-invariant memory。

### 17.2 K=16,V=32 可能不够

K=16,V=32 是合理起点。若后续表现为：

```text
current 能提升，但 history / repair 仍有冲突；
state_rms 正常，但 ctx 信息不足；
memory_off 与 full 差距小；
```

再考虑：

```yaml
K=24,V=32
```

不要第一版扩大 V；V=32 与当前 ctx_dim 对齐，adapter 也简单。

### 17.3 Distant scale 打开后可能改变训练曲线

这是预期变化。3_0 中 distant scale 没打开，distant 分支可能把几何误差压到 event/memory 中。3_1 默认打开 scale，因此训练曲线与 3_0 不完全可比。比较时要看：

```text
fine_event_norm_distant
current/history psnr
distant scale delta norm
history_damage
```

---

## 18. 完成标准

IForward 3_1 实现完成的标准：

```text
[ ] 新配置可启动训练
[ ] 每步 read/write 保持不变
[ ] gdkv state rows 与 parent rows 对齐
[ ] bg/distant dense、rigid keyed 均正常
[ ] mamba_ablation legacy names 可运行
[ ] validation_v3 可运行
[ ] step 20k 前不出现 GDKV NaN/Inf
[ ] 30k 后 repair_flag=true 时 GDKV write 正常
[ ] distant.update_scales=true 生效
[ ] metrics 中出现 parent_optimizer_gdkv/*
```

---

## 19. 后续 GPT 上下文块

```text
IForward 3_1 的目标是只替换 Stage 3_0 的 ParentOptimizerMamba。
新 memory 是 Low-rank Gated Delta KV，不讨论 Kalman/GRU/三层结构。
固定 K=16,V=32，每个 parent 存 S[N,16,32]。
read: q=RMSNorm(q_proj(event))/sqrt(K), ctx=einsum(S,q)，ctx -> adapter -> fused_event。
write: OptimizerWriteTokenBuilder 输出 token；投影 k/v/erase/write/decay；使用 GDN2-lite 更新：
S_decay = decay*S
old = S_decay^T (erase*k)
S = S_decay - outer(k,old) + outer(k,write*v)
然后 RMS clamp S。
保持 scheduler 每 repeat read/write，不改 repair random_order。
为了最小侵入，ParentOptimizerGatedDeltaKV 保持 ParentOptimizerMamba 的 preview/write 接口。
新增 ParentOptimizerDeltaKVState，支持 bg/distant dense 和 rigid keyed。
3_1 配置必须打开 distant scale：attribute_gates.distant.scales>0 且 branch_scope.distant.update_scales=true。
第一版只用 RMS norm/clamp 做数值边界，不做复杂 skip-write/reset。
```
