# AGENT.md — Codex 编程约束与执行指南：IForward / StreetForward

本文档用于约束 Codex 或其他自动编程 agent 在 `drivestudio-coding` 项目中修改 IForward 相关代码时的行为。目标是让 agent 只做必要且可验证的改动，避免引入隐式兼容层、旧协议回退、日志缺失、validation 伪实现或显存不可控路径。

---

## 0. 必须遵守的运行环境

所有测试、脚本、pytest、训练 smoke 都必须使用项目 conda 环境和正确 `PYTHONPATH`。

```bash
cd /root/drivestudio-coding

conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  <command>
```

例如：

```bash
conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  pytest -q tests/test_iforward_stage3_0_full_sparse_gather.py
```

训练 / smoke 示例：

```bash
conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python tools/train_iforward.py \
  --config_file configs/iforward/<CONFIG>.yaml \
  --max_steps 1000 \
  output_name=<SMOKE_NAME>
```

禁止直接运行：

```bash
python ...
pytest ...
```

除非用户明确允许不用 conda 环境。

---

## 1. 当前 IForward 设计定位

IForward 不是普通 feed-forward reconstruction，也不是单纯 temporal prediction。当前项目方向是：

```text
IForward = learned iterative optimizer for 3DGS state
```

核心状态包括：

```text
LocalGSState：场景参数状态
Parent/BigGS：低 token 3D reasoning state
Parent Optimizer Mamba：episode 内 parent-level optimizer hidden state
```

从 Stage 2_3 起，Mamba 的推荐语义是：

```text
Parent Optimizer Mamba
= episode 内优化过程记忆
= 按 optimizer visit / repeat 顺序读写
≠ 物理时间预测 memory
```

因此不要再默认假设：

```text
Mamba 必须只按 chronological frame commit
repair 必须禁止 Mamba write
每个 frame exit 才能 write
```

Stage 2_3 目标是训练一个对访问顺序、repeat budget、repair 过程更鲁棒的 shared updater。

---

## 2. 代码修改原则

### 2.1 不做隐式兼容

当用户要求新 stage 或新 scheduler 时，不要偷偷保留旧路径作为 fallback，除非明确说明。

禁止：

```text
if new_config_missing:
    fallback_to_stage2_1_behavior()
```

应该：

```text
fail-fast with explicit error
```

旧协议、旧字段、旧 scheduler、旧 Temporal Mamba 如果不再使用，应当：

```text
1. 删除；或
2. 明确 legacy-only；或
3. 加 fail-fast，禁止混用。
```

### 2.2 不写“看起来可用”的空实现

尤其是 validation、scheduler、history bank、preload、CUDA wrapper。

禁止只写：

```text
manifest builder
dummy metrics
metadata-only rows
```

然后声称 validation 完成。

validation 必须真正执行模型 rollout、state reset / clone、render、metric aggregation。

### 2.3 不让配置字段失效

如果 YAML 有字段，就必须：

```text
1. 被代码读取；
2. 有测试覆盖；
3. 日志能证明其生效；
4. 修改该字段会改变行为。
```

否则删除字段或在启动时 fail-fast。

### 2.4 不让日志掩盖真实行为

日志不能只保留旧字段。新 scheduler / model 需要专用日志字段。

例如 Stage 2_3 必须记录：

```text
iforward/stage2_3/phase
iforward/stage2_3/visit_kind
iforward/stage2_3/rollout_positions
iforward/stage2_3/history_positions
iforward/stage2_3/repair_positions
iforward/stage2_3/repeat_budgets
iforward/stage2_3/frame_ids
iforward/stage2_3/keyframe_ids
iforward/stage2_3/frame_gaps
iforward/stage2_3/temporal_read_count
iforward/stage2_3/temporal_write_count
```

如果 train_step 只记录了 `shape_name`、`inner_K` 等旧字段，不足以判断新协议是否正确。

---

## 3. Scheduler v3 编程约束

### 3.1 Episode 语义

Stage 2_3 推荐 episode 语义：

```text
episode = one segment + one raw-frame sequence + one LocalGSState + one OptimizerMambaState
```

Segment 仍负责：

```text
初始 GS asset
AABB / point density
parent assignment topology
scene reset boundary
```

Raw frame 是 optimizer visit 的基本 observation unit。

Keyframe 只作为：

```text
空间覆盖标签
unique keyframe统计
sampling约束
日志元数据
```

不要再要求一个 segment 内有 10 个 keyframe 才能训练 Sequence10。

### 3.2 Scheduler 阶段

推荐阶段：

```text
Bootstrap：
    B1 × R{4,6,8}
    fresh GS per frame
    Mamba off
    segment asset pack reuse

Assimilation：
    8~10 raw frames
    mostly chronological or lightly shuffled
    B small, R large
    e.g. B2 × R{4,6,8}
    Mamba every repeat read/write

Repair：
    random revisit frames
    B large, R small
    e.g. B6R1 / B8R1 / B6R2
    Mamba every repeat read/write
    last visit may skip write if episode ends
```

### 3.3 Optimizer Mamba read/write

For Stage 2_3 optimizer Mamba:

```text
每个 repeat:
    read optimizer Mamba
    compute event / delta
    apply delta
    write optimizer Mamba

rollout boundary:
    detach graph
    keep value

episode boundary:
    reset value
```

Do not implement frame-exit-only write unless explicitly requested.

### 3.4 Visit metadata 必须完整

每次 repeat / visit 需要提供：

```text
visit_kind
repeat_idx_in_visit
repeat_budget
frame_idx
sequence_pos
timestamp / delta_t
frame_gap
visit_count_for_frame
global_update_idx_in_episode
is_repair
is_last_visit_in_episode
```

如果这些字段缺失，Mamba 无法区分：

```text
first observation
same-frame repeat
repair revisit
long-repeat stability
```

### 3.5 inner_K 硬约束

任何 scheduler 改动必须显式控制：

```text
inner_K <= configured_max_inner_K
```

当前 48GB GPU 下，建议默认：

```text
max_inner_K = 12
```

不要因为新增 repair / repeat stress 让 inner_K 隐式变大。

### 3.6 Repair 的未访问帧也要监督

如果 repair 只访问部分 frames，例如 B8 in a 10-frame episode：

```text
visited frames：current loss
unvisited frames：retention / history / best-damage loss
```

否则 repair 可以改善访问帧，同时破坏未访问帧而训练不发现。

为省显存，未访问帧可以用：

```text
L1-only retention
L1-only best damage
chunked render
```

不要对所有 10 frames 强行 full SSIM。

---

## 4. Parent Optimizer Mamba 编程约束

### 4.1 命名

新代码中推荐使用：

```text
ParentOptimizerMamba
ParentOptimizerState
optimizer_mamba
```

避免继续使用：

```text
ParentTemporalMamba
TemporalState
```

除非代码明确是 legacy temporal mode。

### 4.2 State 不能跨 episode

Optimizer Mamba state 必须：

```text
episode begin reset
episode end discard
rollout boundary detach but keep value
```

不得跨 scene / segment 复用。

### 4.3 Write token 不能只含 parent event

写入 token 至少应包含：

```text
parent spatial/fused event
support / valid
visit kind embedding
repeat embedding
frame / sequence embedding
```

推荐加入 parent-level delta summary：

```text
parent_delta_mean_norm
parent_opacity_delta_norm
parent_sh_delta_norm
parent_scale_delta_norm
parent_noop_mean
parent_confidence_mean
```

注意：posterior delta 通常是 child-level，不能直接拿来当 parent-level token。必须根据 `child_to_parent` 和 support/mass 聚合。

若代码中出现：

```python
if delta_attr.shape[0] == parent_rows:
    use_delta
else:
    zeros
```

这很可能导致 delta summary 永远为零，需要修复。

### 4.4 读写统计必须可解释

日志至少包括：

```text
iforward/parent_optimizer_mamba/read
iforward/parent_optimizer_mamba/write
iforward/parent_optimizer_mamba/write_skipped
iforward/parent_optimizer_mamba/global_update_step
iforward/parent_optimizer_mamba/bg_written
iforward/parent_optimizer_mamba/distant_written
iforward/parent_optimizer_mamba/rigid_written
iforward/parent_optimizer_mamba/preview_seen_ratio
```

如果 `read=1` 且 `write=1` 同时 `write_skipped=1`，必须能从日志中解释：

```text
哪些 visits 写了
哪些 visits 因 last_visit / invalid / bootstrap 跳过了
```

---

## 5. IForward Stage 3_0 Full Sparse Gather Lift 约束

Stage 3_0 目标是替换当前昂贵的 FWHR feature transport。

当前不推荐继续做 QDG-Child 过渡。推荐：

```text
Full Sparse Gather Lift
```

核心：

```text
2D frontend:
    context_2d [V,H,W,48]
    detail_2d  [V,H,W, 8]

gsplat scalar pass:
    uv / depth / radius / support / view weight / valid

parent sparse gather:
    query -> gather context_2d -> parent_context [M,48]

child sparse gather:
    query -> gather detail_2d -> child_detail [N,8]
```

### 5.1 gsplat 约束

gsplat 不应搬运大 feature：

```text
禁止 fine GS × pixel pair × C feature scatter/backproject
```

Stage 3_0 中 gsplat 主要输出 scalar anchors：

```text
uv_anchor
view_support
screen_radius
depth
valid
visibility_mass
```

### 5.2 Gather 实现顺序

P0：PyTorch chunked `grid_sample` smoke  
P1：CUDA scalar anchor pass  
P2：CUDA sparse gather op  
P3：profile 后决定 parent residual / full parent replacement

不要一开始写复杂 CUDA，先保证数学和训练稳定。

### 5.3 显存硬约束

Full Sparse Gather 的目的之一是降低：

```text
child raw [N,56]
parent index_add graph
FWHR feature backproject graph
```

任何实现都不能偷偷保留旧 FWHR 56D backproject 作为并行路径，除非在 ablation 中显式配置。

---

## 6. Loss / Render / Memory 约束

### 6.1 禁止重复 render

如果 final render 已经计算了 per-ref loss，就不要再调用一次 render 来做 per-position / best-damage。

正确：

```text
final render returns:
    mean loss
    per-ref loss
    ref metadata

stage-specific per-position loss:
    index / group / reduce only
```

错误：

```text
_render_final_losses()
_stage2_x_per_pos_loss()
    -> render_loss() again
```

### 6.2 damage 权重为 0 时不能构建 damage graph

不要写：

```python
loss = damage_weight * compute_damage_graph()
```

应写：

```python
if damage_weight > 0:
    loss = compute_damage_graph()
else:
    loss = zero_scalar_without_graph
```

### 6.3 Repair / all-frame loss 必须支持 chunk

对 8~10 frames × 3 cameras 的 repair 或 validation，应支持：

```text
chunk by frame
chunk by camera
L1 all refs + SSIM subset
```

否则48GB GPU很容易贴边。

---

## 7. Validation 约束

Validation 必须真实执行模型，不接受 metadata-only。

### 7.1 必须支持的 protocol

```text
Assimilation validation
Repair validation
Order robustness validation
Repeat stability validation
Mamba ablation validation
```

### 7.2 Repeat stability 必须 clone state

错误：

```text
R4 -> use resulting state for R8 -> use resulting state for R16
```

正确：

```text
for R in [4,8,16,32]:
    state_R = clone(initial_state)
    run R repeats
```

### 7.3 Causal final all-frame validation 不能额外 update

错误：

```text
causal final state -> run B10R1 -> report as causal all10
```

正确：

```text
causal final state -> render all10 only
```

### 7.4 Repair validation 要和训练协议一致

如果训练 repair 是：

```text
B6R1 / B8R1 / B6R2
```

validation 不应只测 B10R1。至少包括：

```text
Repair-B6R1
Repair-B8R1
Repair-B6R2
Repair-B10 upper bound
```

### 7.5 Mamba ablation 必须存在

用于验证 optimizer memory 是否真的有效：

```text
Mamba off
Mamba read only
Mamba read/write every repeat
Mamba state shuffled
Mamba state reset at repair
```

---

## 8. Performance / Threading / Preload 约束

### 8.1 Scheduler 不应成为训练瓶颈

目标：

```text
scheduler next p99 < 2 ms
batch_fetch mean < 120~140 ms
first rollout fetch p50 < 180 ms
```

### 8.2 使用当前 preload / thread 机制

如果要优化 fetch，应优先利用现有：

```text
AssetPreloadManagerV2
lightweight preload hint
background worker
view meta / view pack cache
```

Stage 2_3 / Stage 3_0 scheduler 应提前提交：

```text
current rollout view pack
next rollout view pack
remaining episode view meta
next episode segment static
```

### 8.3 EpisodeProducer 不能只是同步 facade

如果配置中出现：

```yaml
producer.queue_depth: 32
```

代码必须真的有：

```text
background producer thread
bounded queue
exception propagation
resume deterministic refill
shutdown without deadlock
```

否则不要暴露该配置字段。

---

## 9. 测试要求

所有测试都必须使用：

```bash
conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  pytest -q <tests>
```

### 9.1 修改 scheduler 必须跑

```bash
conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  pytest -q \
    tests/test_iforward_stage2_3_scheduler.py \
    tests/test_iforward_stage2_3_resolver.py \
    tests/test_iforward_stage2_3_history.py
```

### 9.2 修改 Optimizer Mamba 必须跑

```bash
conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  pytest -q \
    tests/test_iforward_stage2_3_optimizer_mamba.py
```

需要覆盖：

```text
every repeat read/write
bootstrap read/write off
last visit skip write
rollout detach keeps value
episode reset clears state
parent-level delta summary nonzero
state shuffled ablation
```

### 9.3 修改 Stage 3 Lift 必须跑

```bash
conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  pytest -q \
    tests/test_iforward_stage3_0_sparse_gather.py \
    tests/test_gsplat_scalar_anchor_stage3_0.py
```

需要覆盖：

```text
anchor shape / dtype / device
gather output shape
grid_sample bounds
chunk equivalence
old FWHR disabled
backward to 2D frontend
no geometry gradient unless explicitly enabled
```

### 9.4 修改 validation 必须跑

```bash
conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  pytest -q \
    tests/test_iforward_stage2_3_validation.py
```

需要覆盖：

```text
real model rollout, not manifest-only
causal all10 render-only
repeat stability clone state
repair protocol matches training
Mamba ablation rows
per-frame metrics
```

---

## 10. Smoke 训练要求

### 10.1 基础 smoke

```bash
conda run -n drivestudio-new env \
  PYTHONPATH=/root/drivestudio-coding \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python tools/train_iforward.py \
  --config_file configs/iforward/<CONFIG>.yaml \
  --max_steps 1000 \
  output_name=<NAME> \
  iforward_stage2_2_validation.enable=false \
  iforward_validation.enable=false \
  iforward_coverage_validation.enable=false
```

### 10.2 必须检查

```text
loss下降
current_psnr上升
无 NaN / Inf
Parent PTv3 梯度非零
GRLD 梯度非零
Posterior updater 梯度非零
Optimizer Mamba 梯度非零
Mamba read/write 按预期
history bank 更新
repair rollout 被记录
显存未超过阈值
```

### 10.3 Repair / damage smoke

如果改 repair / damage / render loss，需要单独跑：

```text
repair start 后 100~200 steps
damage weight > 0 后 100~200 steps
```

不要只跑 damage weight = 0 区间。

---

## 11. 常见隐患清单

提交前逐项检查：

```text
[ ] 新配置字段是否真的生效？
[ ] 旧配置字段是否被禁止或清理？
[ ] scheduler 日志是否能复现一次 episode？
[ ] train_step 是否记录真实 phase / positions / repeats？
[ ] validation 是否真的跑模型？
[ ] repair 是否监督未访问帧？
[ ] Mamba 是否每 repeat 读写？
[ ] Mamba 是否写入 parent-level delta summary？
[ ] frame_gap 是否跨 rollout 正确？
[ ] best damage 是否没有重复 render？
[ ] damage weight=0 是否跳过 graph？
[ ] Repeat Stability 是否 clone state？
[ ] 显存日志是否覆盖 final render / backward？
[ ] fetch 慢是否确认是 data materialization 而非 scheduler plan？
[ ] 测试是否用 conda drivestudio-new + PYTHONPATH？
```

---

## 12. Agent 行为约束

### 12.1 不要做的事

不要：

```text
为了通过测试而加 dummy output
为了兼容旧配置而静默 fallback
为了省事而跳过 validation runner
把显存问题藏到 scheduler 缩小里
把 repair 的未访问帧完全无监督
让 Mamba write token 缺少实际优化信息
在 damage_weight=0 时仍构建大图
```

### 12.2 应该做的事

应该：

```text
先写 fail-fast，再写训练逻辑
先写日志，再跑 smoke
先写单元测试，再改 scheduler
先复用 final per-ref loss，再做 per-position统计
先验证小规模语义，再写 CUDA
每次改动后报告具体测试命令和结果
```

### 12.3 汇报格式

每次完成代码改动后，报告必须包含：

```text
1. 改动文件列表
2. 关键语义变化
3. 测试命令
4. 测试结果
5. smoke 命令，如有
6. 已知未解决问题
7. 是否影响 checkpoint / config 兼容
```

不要只说：

```text
已修复
测试通过
```

必须可复现。
