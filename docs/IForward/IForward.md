可以。那我建议把方向正式定为：

# IForward：独立的 Short-Sequence Mamba Iterative Optimizer

它不是：

```text
Streetforward stage6_0 phase B
phase A extension
VSM 修补版
long memory add-on
forward geometry reconstruction model
```

它应该是一个新的模型家族：

```text
IForward = Iterative Forward Optimizer
```

核心定位：

> IForward 是一个面向短序列的、带 Mamba 持久记忆的 3DGS 迭代优化模型。
> 它以 phase A 的成功经验作为初始化来源，但不继承 phase A / stage6_0 的 trainer、forward、scheduler、phase 结构。
> 它的目标不是一次性重建场景，而是在连续帧中持续优化当前 3DGS，同时保持历史帧效果。

---

# 1. 总体原则

## 1.1 IForward 是新模型，不是新 phase

目录、配置、scheduler、trainer、validation 都应该新建。

建议不要再出现：

```text
stage6_0_phase_b
long_vsm
phase_b_forward
minimal_trainer_stage6_0
```

新的命名可以是：

```text
models/iforward/
configs/iforward/
datasets/iforward_scheduler/
validators/iforward/
```

也可以放在 Streetforward repo 内，但逻辑上它是新模型：

```text
Streetforward/
    models/
        iforward/
```

而不是：

```text
models/streetforward/stage6_0/phase_b
```

## 1.2 可以使用 phase A 预训练权重，但不能继承 phase A 结构

这里需要区分两件事。

**允许：**

```text
从 phase A checkpoint 读取权重
初始化 IForward 的 observation backbone
初始化 IForward 的 event encoder
初始化 IForward 的 delta updater
复用已经验证有效的 2D lifting 思路
复用部分底层工具函数
```

**不允许：**

```text
继承 stage6_0 trainer
复用 phase_a / phase_b forward 分支
在 phase A forward 里加 memory
沿用 phase B scheduler
沿用当前 VSM long-memory 结构
把 IForward 写成 stage6_0 的一个 mode
```

换句话说：

> phase A 是 pretrained source，不是 architectural parent。

---

# 2. IForward 的核心目标

IForward 要解决的问题是：

```text
给定已有 3DGS state G
给定短序列观测 frames t0...tN
给定多相机图像
模型通过多次迭代更新 G
使当前帧变好
同时过去帧不坏
并把优化经验写入 Mamba memory
```

它的本质不是：

```text
image sequence -> one-shot 3D scene
```

而是：

```text
current GS + observation + memory -> small physical GS delta
repeat many times
```

所以 IForward 的核心是：

```text
persistent learned optimizer
```

不是：

```text
feed-forward reconstruction network
```

---

# 3. IForward 总体结构

我建议 IForward 由六个高层模块组成。

```text
IForward
├── 1. IForwardState
├── 2. Observation Backbone
├── 3. Structure Event Encoder
├── 4. Mamba Optimizer Memory
├── 5. Memory-Conditioned Iterative Updater
└── 6. Sequence Scheduler / Loss / Validation
```

---

## 3.1 IForwardState

IForward 需要一个显式 state，而不是把状态散落在 trainer 里。

它至少包含：

```text
IForwardState
├── current 3DGS state
│   ├── bg gaussians
│   ├── distant gaussians
│   └── rigid gaussians
│
├── optimizer memory state
│   ├── point-level Mamba memory
│   ├── cell-level Mamba memory
│   ├── global/object-level Mamba memory
│   └── short-window recent buffer
│
├── sequence metadata
│   ├── current frame index
│   ├── current iteration index
│   ├── committed evidence frames
│   ├── history anchor frames
│   └── memory detach / carry flags
│
└── debug / logging state
```

这一步非常重要。

phase A 本质上只有：

```text
local_G
```

但 IForward 必须有：

```text
local_G + optimizer_memory + sequence_context
```

否则它无法成为真正的 persistent optimizer。

---

## 3.2 Observation Backbone

Observation Backbone 负责把当前 3DGS 与当前图像观测变成 per-Gaussian observation。

整体逻辑沿用 phase A 的成功机制：

```text
current GS
    ↓ render current views
rendered RGB
    ↓ compare with real images
image residual / image pair feature
    ↓ 2D frontend
2D feature maps
    ↓ 2D lifting / backprojection
per-GS observation feature
```

IForward 中这个模块可以从 phase A 初始化。

但在 IForward 中，它应该被重新包装为独立接口：

```text
IForwardObservationBackbone.observe(
    state,
    frame,
    cameras,
    images,
) -> ObservationPack
```

输出不是直接喂给 phase A updater，而是进入 IForward 的 event / memory / updater 流程。

建议输出：

```text
ObservationPack
├── per_point_obs_feature
├── support
├── obs_code
├── visibility / coverage
├── camera contribution stats
├── render residual stats
└── lifting debug stats
```

这个模块的设计原则：

> Observation Backbone 负责“看见当前 GS 哪里错了”，不负责记忆历史，也不直接决定最终更新。

---

## 3.3 Structure Event Encoder

Observation 本身还不是 optimizer event。

IForward 需要把 observation、当前 GS 参数、branch 类型、support、obs code 编成结构事件。

```text
ObservationPack + current GS params
    ↓
StructureEvent
```

建议事件仍然按 branch 分开：

```text
bg event
distant event
rigid event
```

因为三类点的物理更新策略不同：

```text
bg:
    可以较强更新 geometry / opacity / SH

rigid:
    需要 object-local memory
    需要避免破坏刚体结构

distant:
    更保守
    早期主要更新 appearance / opacity
    means update 应严格控制
```

IForward 可以用 phase A 的 event encoder 权重初始化，但接口上要重新定义为：

```text
IForwardEventEncoder.encode(
    observation,
    state,
    branch_info,
) -> EventPack
```

EventPack 应该是 memory 和 updater 的共同输入。

---

## 3.4 Mamba Optimizer Memory

这是 IForward 与 phase A 的本质区别。

Memory 不是外挂 decoder，也不是 VSM readout。

Memory 是 optimizer state 的一部分。

我建议直接设计成三层 Mamba memory：

```text
Mamba Optimizer Memory
├── point-level memory
├── cell-level memory
├── global/object-level memory
└── short-window recent buffer
```

---

### 3.4.1 Point-level Mamba memory

每个稳定 GS point 维护一个低维 memory。

它记录的是：

```text
这个点过去被怎样观测过
过去更新是否稳定
历史 confidence
历史 support
过去 delta 趋势
当前 observation 是否与历史冲突
这个点是否应该 noop
```

它不直接存图像，也不直接生成颜色。

它服务于：

```text
是否更新
更新哪个参数族
更新幅度多大
是否保护历史
```

---

### 3.4.2 Cell-level Mamba memory

40w 个 GS 点不能只靠 point memory 独立处理。

IForward 需要 cell-level memory，把局部空间块作为 optimizer 单元。

```text
world/object-local voxel cell
    ↓ aggregate point events
cell event
    ↓ Mamba
cell memory readout
    ↓ broadcast / query back to points
```

Cell memory 负责：

```text
局部几何一致性
邻近点共同 drift 的检测
局部历史稳定性
未观测点保护
局部冲突解决
```

这比单点 memory 更适合 3DGS。

因为 3DGS 的问题通常不是单点错误，而是局部结构错误：

```text
一片表面过厚
一段边缘漂移
一组 opacity 错配
一个 rigid object 局部错位
```

---

### 3.4.3 Global / object-level Mamba memory

全局 memory 不处理细节，而处理 sequence-level optimizer context。

它记录：

```text
当前序列整体观测质量
camera coverage 变化
整体 brightness / exposure 倾向
某个 object 的更新风险
当前 frame 是否应该强更新或保守更新
长期 drift 趋势
```

对于 rigid objects，最好有 object-level memory：

```text
rigid_object_id -> object Mamba memory
```

---

### 3.4.4 Short-window recent buffer

只用 Mamba 压缩历史不够。

IForward 还需要一个短窗口未压缩 buffer：

```text
recent frames / recent chunks
```

它保存最近少量帧的高频信息：

```text
recent event summary
recent support
recent render loss stats
recent anchor refs
recent per-cell summaries
```

它的作用是：

```text
保留近帧高频约束
帮助历史 retention
避免 Mamba 过度压缩邻近几何信息
提供短程 consistency
```

这不是“简单版本 memory”，而是完整 memory 系统的一部分。

最终 memory 结构是：

```text
short-window buffer 负责近程精细一致性
Mamba memory 负责持久 optimizer 状态
GS state 负责真实场景表示
```

---

## 3.5 Memory-Conditioned Iterative Updater

Updater 是 IForward 的核心执行器。

它输入：

```text
current GS params
current event
Mamba memory readout
short-window context
iteration embedding
frame embedding
branch embedding
```

输出：

```text
physical GS delta
noop / confidence
memory write signal
update diagnostics
```

核心流程：

```text
for each sequence frame:
    for each iteration:
        observe current GS
        encode event
        read/update memory
        predict delta
        apply delta to GS
        render losses
```

但要区分两种动作：

```text
observation commit
optimizer refinement
```

这点非常重要。

---

### 3.5.1 Observation commit

当进入一个新 frame / 新 camera evidence 时，memory 应该写入新的观测。

```text
new evidence arrives
    -> commit to memory
```

---

### 3.5.2 Optimizer refinement

对同一帧做多次 RAFT-style 迭代时，不应该把每次迭代都当成新的独立观测。

否则 memory 会被同一帧重复污染。

所以 IForward 应该有：

```text
commit step:
    写入 observation memory

refinement step:
    使用 memory
    可写 optimizer trajectory memory
    但不能伪装成新观测
```

推荐抽象：

```text
IForwardStep
├── frame_index
├── iter_index
├── evidence_is_new
├── commit_observation_memory: true / false
├── update_optimizer_memory: true / false
└── update_gs_state: true
```

---

## 3.6 Zero-init memory injection

为了安全迁移 phase A，IForward 必须有 zero-init memory injection。

初始状态下：

```text
IForward with memory ≈ phase A initialized updater
```

也就是说：

```text
memory contribution = 0
```

训练开始时模型应该等价于：

```text
phase-A-like iterative optimizer
```

然后逐渐学会：

```text
什么时候利用 memory
什么时候保护历史
什么时候修正 phase A delta
什么时候 noop
```

我建议 updater 采用 residual 形式：

```text
delta_base = phaseA_initialized_updater(event)

delta_mem = zero_init_residual_head(
    event,
    memory_readout,
    state_context
)

delta = delta_base + delta_mem
```

这样 IForward 的训练会稳定很多。

---

# 4. IForward 的 forward 逻辑

总体 forward 可以定义为：

```text
Input:
    initial GS state
    sequence frames
    camera images
    camera params
    scheduler roles
    optional initial memory

Output:
    optimized GS state
    updated memory
    losses
    logs
    validation renders
```

高层流程：

```text
G = initial GS
M = initial memory

for t in short_sequence:

    prepare evidence cameras for frame t
    prepare current supervision refs
    prepare history anchors

    for k in iterative_steps:

        observation = observe(G, frame=t, evidence_cams)

        event = encode_event(observation, G)

        memory_read, M = mamba_memory_step(
            event,
            G,
            M,
            frame=t,
            iter=k,
            commit_observation=(k == 0)
        )

        delta = updater(
            event,
            memory_read,
            G,
            frame=t,
            iter=k
        )

        G = apply_delta(G, delta)

        accumulate current loss
        accumulate history retention loss
        accumulate regularization loss

    commit short-window history
```

IForward 的核心循环是：

```text
observe -> event -> memory -> delta -> apply -> render loss
```

不是：

```text
encode sequence -> decode scene
```

---

# 5. Scheduler 总体方案

IForward 需要全新的 scheduler。

不要复用 phase A 的 block-local scheduler，也不要复用 phase B 的 prefix/query scheduler。

新的 scheduler 应该围绕 sequence optimizer 设计。

---

## 5.1 Scheduler 输出

每个 training sample 应该输出：

```text
IForwardBatch
├── sequence frames
├── evidence cameras per frame
├── current supervision cameras
├── history anchor frames
├── heldout cameras / heldout frames
├── frame order
├── iteration plan
├── memory commit flags
├── leakage guard metadata
└── evaluation roles
```

重点是 roles 要清楚。

例如：

```text
frame t0:
    evidence: cam0, cam1, cam2
    current loss: cam0, cam1, cam2
    heldout: optional
    history anchors: none

frame t1:
    evidence: cam0, cam1, cam2
    current loss: cam0, cam1, cam2
    history anchors: t0

frame t2:
    evidence: cam0, cam1
    current loss: cam0, cam1
    heldout: cam2
    history anchors: t0, t1
```

---

## 5.2 Scheduler curriculum

### Curriculum 1：短序列稳定

```text
sequence length: 2-4 frames
evidence: 3 cams
iterations: 4-8
history anchors: recent only
```

目标：

```text
IForward 能在 t0 -> t1 -> t2 中保持历史
```

---

### Curriculum 2：heldout camera

```text
sequence length: 2-4 frames
evidence: 1-2 cams
heldout loss: remaining cams
iterations: 4-8
```

目标：

```text
防止模型只优化 evidence view
提升跨视角一致性
```

---

### Curriculum 3：更长短序列

```text
sequence length: 4-8 frames
iterations: 2-8
history anchors: recent + mid + old
```

目标：

```text
训练真正的 persistent memory
```

---

### Curriculum 4：chunk streaming 前置

```text
chunk length: 4-8
chunk overlap: 1-2
memory carried across chunks
GS carried across chunks
```

目标：

```text
为长序列做准备
```

---

# 6. Loss 总体方案

IForward 的 loss 应该服务于两个主目标：

```text
当前帧变好
历史帧不坏
```

所以主 loss 不是 query decoder loss，而是 render-based optimizer loss。

---

## 6.1 Current frame loss

每个 frame 每次迭代都可以有 current render loss。

```text
L_current
```

用于保证新观测能被吸收。

---

## 6.2 History retention loss

这是 IForward 最关键的 loss。

处理新帧后，要重新渲染过去帧：

```text
after optimizing frame t
    render frame t-1 / t-2 / older anchors
    compare with GT
```

得到：

```text
L_history
```

核心指标：

```text
retention_gap = history_psnr_before_update - history_psnr_after_update
```

模型必须学习：

```text
新帧提升不能以历史崩坏为代价
```

---

## 6.3 Heldout camera / temporal NVS loss

为了防止只拟合 evidence cameras，需要：

```text
L_heldout_cam
L_heldout_temporal
```

例如：

```text
evidence: front/left
heldout: right
```

或者：

```text
evidence: t
heldout: t+1 nearby frame
```

---

## 6.4 Physical regularization

IForward 必须继续尊重 3DGS 物理结构。

需要保留或新增：

```text
delta norm regularization
scale barrier
opacity drift penalty
SH drift penalty
high-confidence unobserved point protection
noop regularization
confidence calibration
branch-specific clamp
```

尤其是：

```text
当前 support 低
但历史 confidence 高
```

这种点应该默认保守，而不是被新帧错误更新。

---

## 6.5 Memory auxiliary loss

Memory 不应该直接承担主重建任务。

但可以有辅助校准 loss：

```text
support prediction
update confidence prediction
history risk prediction
delta norm bucket prediction
noop prediction
```

这些用于帮助 memory 学会 optimizer behavior，而不是让 memory 变成 scene decoder。

---

# 7. 训练总体路线

我建议 IForward 分五个阶段做。

---

## Stage I：IForward skeleton

目标：

```text
建立独立 IForward 模型骨架
建立独立 scheduler / config / validation
建立 IForwardState
建立统一 forward API
```

这一步不追求超越 phase A，只追求结构独立。

完成后应该有：

```text
IForward 可以加载一个 initial GS
可以读取一个短序列 batch
可以执行 iterative forward
可以输出 render loss
可以记录 logs
```

这一阶段的关键结果：

```text
IForward 已经不是 stage6_0 phase 分支
```

---

## Stage II：single-frame compatibility

目标：

```text
IForward 在 T=1, 3cam, K=8/32 下接近 phase A
```

做法：

```text
加载 phase A 权重
memory 输出 zero
delta residual 输出 zero
只跑 IForward 新 forward
```

这一步是 safety gate。

如果 IForward 连单帧都复现不了 phase A，后面 sequence memory 没意义。

验证：

```text
T=1
cams=3
K=8,16,32
memory=off / zero
```

通过条件：

```text
PSNR 接近 phase A
delta 行为稳定
render loss 曲线正常下降
```

---

## Stage III：short-sequence Mamba training

目标：

```text
完整启用 Mamba memory
训练 IForward 处理 2-4 帧短序列
```

训练策略：

```text
freeze observation backbone
freeze or mostly freeze phaseA-initialized updater
train Mamba memory
train zero-init memory injector
train delta residual head
train noop/confidence modulation
```

这一阶段重点不是最高 PSNR，而是：

```text
history retention
memory 有效性
不会破坏单帧能力
```

通过条件：

```text
full memory > zero memory
history retention gap 下降
current frame PSNR 不明显低于 phase-A-like baseline
```

---

## Stage IV：joint fine-tuning

目标：

```text
让 IForward 从 phase-A-like optimizer 变成真正的 sequence optimizer
```

逐步解冻：

```text
updater 后层
event encoder 小 lr
2D residual/fusion 极小 lr
DINO 继续 frozen
```

训练更难数据：

```text
sequence length: 4-8
evidence cams: 1-3 mixed
heldout cams
history anchors: recent + mid + old
```

这一阶段追求：

```text
current PSNR
history PSNR
heldout PSNR
三者一起提升
```

而不是只看当前帧。

---

## Stage V：chunk streaming extension

目标：

```text
把短序列 IForward 扩展为长序列 optimizer
```

但长序列不是新模型。

它只是：

```text
IForward short-sequence optimizer
+ memory carry
+ GS carry
+ chunk overlap
+ long validation
```

流程：

```text
chunk 0:
    run IForward
    output G, M

chunk 1:
    input previous G, M
    run IForward
    output G, M

chunk 2:
    ...
```

关键是：

```text
short-window buffer 跨少量 chunk
Mamba memory 跨长程
GS state 永远是 scene source of truth
```

---

# 8. Validation 总体方案

IForward 必须有独立 validation。

我建议固定五套 protocol。

---

## V0：phase-A compatibility validation

目的：

```text
确保 IForward 没丢掉 phase A 的单帧能力
```

设置：

```text
T=1
3cam
K=8/16/32/64
memory zero / enabled
```

看：

```text
single-frame PSNR
loss curve
delta norm
stability
```

---

## V1：short-sequence optimization validation

目的：

```text
验证当前帧能不能随着迭代变好
```

设置：

```text
T=2/4/8
R=2/4/8/16
3cam evidence
```

看：

```text
current PSNR by frame
current PSNR by iteration
```

---

## V2：history retention validation

目的：

```text
验证模型是否真的保持历史
```

设置：

```text
处理 t0, t1, t2, ...
每处理完新 frame
重新 render 所有过去 frames
```

看：

```text
history PSNR
retention gap
forgetting curve
old-frame degradation
```

这是 IForward 成败的核心指标。

---

## V3：heldout view validation

目的：

```text
防止模型只优化输入 camera
```

设置：

```text
evidence: 1-2 cams
eval: heldout cam
```

看：

```text
heldout camera PSNR
heldout SSIM
cross-view consistency
```

---

## V4：memory ablation validation

目的：

```text
确认 Mamba memory 真的有用
```

比较：

```text
full IForward
zero point memory
zero cell memory
zero global memory
drop short-window buffer
shuffle memory
phase-A-like no-memory baseline
```

如果 full 只提升 current PSNR，但不提升 retention，那 memory 设计还不成熟。

真正成功的 IForward 应该表现为：

```text
current PSNR 不差
history retention 明显更好
heldout view 更稳
memory ablation 后性能下降
```

---

# 9. Logging 总体方案

IForward 的 logs 应该从一开始就独立设计。

不要只记录 render PSNR。

至少需要四类日志。

---

## 9.1 Reconstruction logs

```text
current_psnr
current_ssim
current_l1
heldout_psnr
history_psnr_recent
history_psnr_old
psnr_by_iteration
psnr_by_frame
```

---

## 9.2 Retention logs

```text
retention_gap_recent
retention_gap_old
forgetting_curve
old_anchor_degradation
history_loss_after_each_frame
```

---

## 9.3 Memory logs

```text
point_memory_norm
cell_memory_norm
global_memory_norm
memory_write_strength
memory_read_strength
memory_update_norm
memory_ablation_gap
short_window_hit_rate
```

---

## 9.4 Physical GS logs

```text
delta_means_norm
delta_scale_norm
delta_quat_norm
delta_opacity_norm
delta_sh_norm
noop_ratio
confidence_mean
opacity_drift
scale_barrier
invalid_gaussian_count
branch-wise update stats
```

这些日志要服务于诊断：

```text
是 observation 错？
是 memory 错？
是 updater 错？
是 scheduler 泄漏？
是 GS 物理更新不稳定？
```

---

# 10. 推荐代码组织

高层结构可以这样设计：

```text
models/
    iforward/
        __init__.py

        iforward_model.py
        iforward_state.py
        iforward_observation.py
        iforward_event_encoder.py
        iforward_memory.py
        iforward_updater.py
        iforward_losses.py
        iforward_logging.py
        iforward_validation.py

        memory/
            point_mamba_memory.py
            cell_mamba_memory.py
            global_mamba_memory.py
            short_window_buffer.py

        schedulers/
            iforward_sequence_scheduler.py
            iforward_curriculum.py

configs/
    iforward/
        iforward_base.yaml
        iforward_single_frame_compat.yaml
        iforward_shortseq_mamba.yaml
        iforward_joint_finetune.yaml
        iforward_chunk_streaming.yaml

validators/
    iforward/
        validate_single_frame.py
        validate_short_sequence.py
        validate_retention.py
        validate_heldout.py
        validate_memory_ablation.py
```

注意这里不是具体实现细节，只是边界划分。

核心原则是：

```text
IForward model 不知道 phase A / phase B 这些概念
IForward scheduler 不依赖 stage6_0 scheduler
IForward validator 不复用 phase B validation
IForward config 不继承 stage6_0_phase_b
```

---

# 11. 和 phase A 的关系

phase A 在 IForward 中的角色应该是：

```text
pretrained initialization source
```

不是：

```text
base class
```

建议建立一个明确的 weight import 工具：

```text
tools/iforward/import_phase_a_weights.py
```

它做的事情是：

```text
phase A checkpoint
    ↓
IForward observation backbone init
IForward event encoder init
IForward base updater init
```

但导入后，IForward checkpoint 就是独立 checkpoint。

也就是说：

```text
phase A ckpt -> IForward init ckpt -> IForward training
```

而不是：

```text
phase A trainer loads phase B modules
```

这能避免后续代码继续被 stage6_0 的历史结构绑住。

---

# 12. 最重要的设计边界

IForward 早期不要试图同时解决所有问题。

第一目标不是长序列。

第一目标是：

```text
短序列中，当前帧提升，同时历史帧不掉。
```

因此顺序应该是：

```text
1. 单帧 compatibility
2. 2-4 帧 retention
3. 4-8 帧 retention
4. heldout camera consistency
5. chunk streaming
6. 长序列
```

不要一开始就做：

```text
long sequence
massive memory
全场景所有点 attention
复杂 query decoder
offset-only update
```

但 memory 模型本身一开始就应该是完整 Mamba 设计，不需要从“简单 memory”开始。

也就是说：

```text
训练任务从短到长
模型结构从一开始就是 Mamba persistent optimizer
```

---

# 13. 最终路线图

我建议 IForward 的路线可以这样定：

```text
Milestone 0:
    废弃 phase B 方向
    冻结 stage6_0_phase_b
    新建 IForward namespace

Milestone 1:
    IForward skeleton
    新 state / model / scheduler / config / validator

Milestone 2:
    Phase-A weight import
    single-frame compatibility
    memory zero-init
    T=1 复现 phase A 行为

Milestone 3:
    启用 point/cell/global Mamba memory
    T=2-4 short sequence training
    当前帧 + history retention loss

Milestone 4:
    加 heldout camera / temporal validation
    训练 1-2 evidence cams
    防止 view overfit

Milestone 5:
    joint fine-tuning
    T=4-8
    stronger history anchors
    branch-specific update control

Milestone 6:
    chunk streaming
    memory carry
    GS carry
    overlap consistency

Milestone 7:
    long-sequence validation
    memory ablation
    retention / forgetting curve 作为主指标
```

---

# 14. 我的最终建议

IForward 的一句话定义应该是：

> **IForward 是一个从 phase A 初始化、但架构完全独立的 Mamba-conditioned short-sequence 3DGS iterative optimizer。它通过 observation → event → memory → delta → render-loss 的循环，在连续帧中持续优化 3DGS，并以 history retention 作为核心训练目标。**

这条路线比继续修 phase B 更合理，因为它直接承认了 Streetforward 当前最强的能力：

```text
不是大模型 forward spatial reasoning
而是 learned iterative 3DGS optimization
```

IForward 应该把这个能力扩展到短序列，并在短序列稳定后自然扩展到长序列。


下面是 **IForward 详细实现方案文档**。这版按完整结构写，不是简化版，不把 Mamba 替换成 GRU/MLP/普通 SSM，也不把 memory 做成 phase B/VSM 式外挂。

可以直接作为：

```text
docs/IForward/IForward_Implementation.md
```

的正文。

---

# IForward 详细实现方案

版本：IForward Implementation V1
目标：独立的 Short-Sequence Mamba Iterative Optimizer
输入约定：**每个 optimizer step 输入一帧 3cam evidence**
训练单位：scheduler 已完成的 IForward rollout
核心结构：

```text
IForward
├── point / cell / global Mamba optimizer memory
├── short-window uncompressed history
├── zero-init memory injection
├── history-retention render loss
├── IForward sequence scheduler
└── memory ablation validation
```

---

# 0. 结论

IForward 是一个全新的模型，不继承：

```text
MinimalStreetForwardStage6_0
stage6_0 phase A
stage6_0 phase B
phase_b_long
VSM
GRU recurrent updater
```

IForward 可以从 phase A checkpoint 导入部分权重，但 phase A 只作为 **pretrained initialization source**，不是父类，不是 forward 分支，不是 trainer base。

IForward 的核心目标是：

```text
给定当前 3DGS state G
给定短序列输入帧，每次输入一帧 3cam
通过多次 learned iterative update 优化 G
同时用 Mamba optimizer memory 保存优化历史
并通过 history-retention render loss 防止历史帧效果下降
```

IForward 不是：

```text
image sequence -> one-shot scene
```

而是：

```text
current GS + one-frame-3cam observation + Mamba memory -> small physical GS delta
repeat
```

---

# 1. 必须满足的边界

## 1.1 模型边界

IForward 必须是独立 namespace。

推荐目录：

```text
models/
  iforward/
    __init__.py
    iforward_model.py
    iforward_state.py
    iforward_gs_state.py

    observation/
      __init__.py
      iforward_observation_backbone.py
      iforward_lifting_v4.py
      iforward_render_observer.py

    event/
      __init__.py
      iforward_event_encoder.py
      iforward_param_obs_codec.py
      iforward_routed_struct_event.py

    memory/
      __init__.py
      streaming_mamba_cell.py
      point_mamba_memory.py
      cell_mamba_memory.py
      global_mamba_memory.py
      short_window_history.py
      iforward_memory.py

    updater/
      __init__.py
      iforward_base_updater.py
      zero_init_memory_injector.py
      iforward_memory_conditioned_updater.py
      iforward_delta_apply.py

    losses/
      __init__.py
      iforward_render_loss.py
      iforward_delta_regularization.py
      iforward_rollout_loss.py

    validation/
      __init__.py
      iforward_validator.py
      iforward_ablation.py
      iforward_metrics.py

tools/
  iforward/
    import_phase_a_weights.py

configs/
  iforward/
    iforward_base.yaml
    iforward_memory.yaml
    iforward_train.yaml
    iforward_validation.yaml
```

不建议：

```text
models/streetforward/stage6_0/iforward
models/streetforward/stage6_0/phase_b_iforward
minimal_trainer_stage6_1.py
```

原因是 IForward 不应该继续受 stage6_0 的 phase 结构约束。

---

## 1.2 和 phase A 的关系

允许：

```text
导入 phase A 的 2D frontend 权重
导入 phase A 的 V4 lifting 相关可训练权重
导入 phase A 的 struct event encoder 权重
导入 phase A 的 posterior updater 权重
复用 phase A 已验证的 render loss 形式
复用 phase A 的 delta regularization 形式
```

不允许：

```text
继承 phase A trainer
调用 phase A forward
在 phase A forward 里插入 memory
复用 phase B/VSM forward
复用 phase B offset decoder
用 phase B long memory 替代 IForward memory
```

IForward 权重导入关系应为：

```text
phase_a_checkpoint
    ↓ import_phase_a_weights.py
iforward_init_checkpoint
    ↓ train
iforward_checkpoint
```

导入完成后，IForward checkpoint 应该完全独立。

---

## 1.3 输入对齐 phase A

IForward 的每个 optimizer step 必须对齐 phase A 的基本 observation 形态：

```text
one source frame
all 3 cameras
```

也就是说，每个 step：

```text
step.evidence_refs = [
    (source_frame_idx, cam0),
    (source_frame_idx, cam1),
    (source_frame_idx, cam2),
]
```

模型侧必须校验：

```text
len(evidence_refs) == 3
all refs share the same frame_idx
cam indices are exactly the configured 3cam set
```

注意：scheduler 会把整个 rollout 的 evidence refs flatten 到 batch source 中，但 IForward **不能一次把整个短序列当成 observation 输入**。它必须按 step 逐帧取 3cam：

```text
rollout source refs flat:
    [t0_cam0, t0_cam1, t0_cam2, t1_cam0, t1_cam1, t1_cam2, ...]

IForward execution:
    step 0 uses only t0 3cam
    step 1 uses only t0 3cam again
    ...
    next block uses only t1 3cam
```

---

# 2. Scheduler 契约

当前 scheduler 已经完成，IForward 模型实现应直接消费：

```text
batch["_iforward"]
```

其中核心字段为：

```text
_iforward:
  scheduler_version
  model_family

  scene_id
  segment_id
  episode_id
  rollout_id_global
  rollout_idx_in_episode

  input_frame_indices
  delivery_frame_indices
  steps

  reset_scene_state_before_rollout
  carry_scene_state_after_rollout
  episode_end_after_rollout
  detach_graph_after_rollout

  evidence_refs_flat
  target_refs_flat
  target_roles_flat

  final_supervision
```

每个 step 包含：

```text
step:
  step_idx
  episode_block_idx
  rollout_block_rank
  repeat_idx

  source_keyframe_idx
  source_frame_idx

  evidence_refs
  evidence_frame_indices
  evidence_cam_indices

  commit_observation_memory
  update_optimizer_memory

  detach_before_step
  detach_after_step

  rollout_pos_code
  frame_pos_code
  repeat_pos_code
```

IForward 必须严格遵守：

```text
commit_observation_memory == True  only when repeat_idx == 0
update_optimizer_memory   == True  for every repeat
detach_before_step        == False inside rollout
detach_after_step         == False inside rollout
```

含义：

```text
repeat_idx == 0:
    新 frame 的 observation 第一次进入，允许写 observation memory

repeat_idx > 0:
    同一 frame 的 iterative refinement
    不允许把它当成新的 independent observation
    但允许更新 optimizer trajectory memory
```

---

# 3. IForward 总体 forward

IForward 的主循环如下：

```python
def forward_rollout(self, batch, carried_state=None):
    plan = batch["_iforward"]

    if plan["reset_scene_state_before_rollout"]:
        state = self.initialize_state_from_batch(batch, plan)
    else:
        state = carried_state

    rollout_deltas = []
    rollout_step_logs = []

    for step in plan["steps"]:
        view_pack = self.build_step_view_pack(batch, plan, step)

        observation = self.observation_backbone.observe(
            gs_state=state.gs,
            view_pack=view_pack,
            source_frame_idx=step["source_frame_idx"],
        )

        event = self.event_encoder.encode(
            gs_state=state.gs,
            observation=observation,
            step=step,
        )

        memory_read, state.memory = self.memory.step(
            gs_state=state.gs,
            event=event,
            short_history=state.short_history,
            step=step,
        )

        delta, update_aux = self.updater(
            gs_state=state.gs,
            event=event,
            memory_read=memory_read,
            step=step,
        )

        state.gs = self.apply_delta(state.gs, delta)

        rollout_deltas.append(delta)
        rollout_step_logs.append(update_aux)

    loss_pack = self.losses.compute_rollout_final_loss(
        gs_state=state.gs,
        batch=batch,
        plan=plan,
        short_history=state.short_history,
        rollout_deltas=rollout_deltas,
    )

    state.short_history.commit_rollout(
        batch=batch,
        plan=plan,
        final_gs_state=state.gs,
        final_loss_pack=loss_pack,
    )

    output = IForwardRolloutOutput(
        loss=loss_pack.total,
        logs=merge_logs(...),
        state=state,
    )

    return output
```

主链路固定为：

```text
one-frame-3cam observation
    ↓
structure event
    ↓
point/cell/global Mamba optimizer memory
    ↓
zero-init memory injection
    ↓
physical GS delta
    ↓
apply delta
    ↓
rollout-final render loss
```

---

# 4. IForwardState

IForward 必须有显式 state。

```python
@dataclass
class IForwardState:
    gs: IForwardGSState
    memory: IForwardMemoryState
    short_history: IForwardShortWindowHistory

    scene_id: int
    segment_id: int
    episode_id: int
    rollout_id_global: int
    rollout_idx_in_episode: int

    num_rollouts_seen: int
    num_frames_committed: int

    def detach_for_next_rollout(self) -> "IForwardState":
        ...

    def to(self, device, dtype=None) -> "IForwardState":
        ...
```

生命周期：

```text
episode begin:
    initialize GS from segment assets
    initialize all Mamba memory states to zero
    clear short-window history

rollout begin:
    use carried detached state unless reset_scene_state_before_rollout=True

inside rollout:
    no detach

after rollout backward:
    detach GS and memory values
    carry to next rollout if carry_scene_state_after_rollout=True

episode end:
    discard state
```

注意：

```text
GS state 是 runtime optimizer state，不是 nn.Parameter
Mamba module weights 是 nn.Parameter
Mamba hidden states 是 carried tensor state，不是 optimizer parameter
```

---

# 5. IForwardGSState

IForwardGSState 应该独立定义，不直接依赖 stage6_0 的 `LocalGSState` 类型，但结构应保持 phase A 兼容。

```python
@dataclass
class IForwardBranchState:
    means: Tensor
    scales_log: Tensor
    quats: Tensor
    opacity_logits: Tensor
    sh: Tensor
    hidden: Optional[Tensor]

    stable_ids: Optional[Tensor]
    branch_mask: Optional[Tensor]
    object_ids: Optional[Tensor]

@dataclass
class IForwardGSState:
    bg: IForwardBranchState
    distant: Optional[IForwardBranchState]
    rigid: Optional[IForwardBranchState]

    frame_idx: Optional[int]
    segment_transform: Any
    branch_meta: Dict[str, Any]
```

branch 对齐：

```text
bg:
    row id stable across episode

distant:
    row id stable across episode

rigid:
    point memory 必须按 stable point id / object id 对齐
    不能依赖当前 frame materialized row order
```

rigid 当前 frame 可能只 materialize subset，因此需要：

```text
rigid.stable_ids: [N_rigid_active]
rigid.object_ids: [N_rigid_active]
```

memory state 通过 stable ids 做 gather/scatter。

---

# 6. Observation Backbone

## 6.1 功能

Observation Backbone 负责把：

```text
current GS state
+ one frame 3cam images
+ camera params
```

变成：

```text
per-Gaussian observation feature
support
obs_code
visibility stats
lifting stats
```

接口：

```python
class IForwardObservationBackbone(nn.Module):
    def observe(
        self,
        *,
        gs_state: IForwardGSState,
        view_pack: IForwardStepViewPack,
        source_frame_idx: int,
    ) -> IForwardObservationPack:
        ...
```

输出：

```python
@dataclass
class IForwardBranchObservation:
    feat: Tensor          # [N_branch, C_obs]
    support: Tensor       # [N_branch, 1] or [N_branch]
    obs_code: Tensor      # [N_branch, C_obs_code]
    valid: Tensor         # [N_branch]
    lifting_stats: Dict[str, float]

@dataclass
class IForwardObservationPack:
    bg: IForwardBranchObservation
    distant: Optional[IForwardBranchObservation]
    rigid: Optional[IForwardBranchObservation]

    source_frame_idx: int
    source_cam_indices: List[int]
    render_stats: Dict[str, float]
    lifting_stats: Dict[str, float]
```

## 6.2 对齐 phase A 的 observation 方式

Observation Backbone 应保留 phase A 的关键机制：

```text
current GS render to source 3cam
    ↓
real RGB + rendered RGB 拼接
    ↓
2D frontend
    ↓
V4 fused multi-camera backprojection
    ↓
per-GS observation
```

输入必须是：

```text
one frame 3cam
```

不是：

```text
multi-frame feature concat
sequence attention
all rollout frames at once
```

这样可以保持 phase A 已经验证的 observation 分布。

## 6.3 2D frontend 权重导入

IForward Observation Backbone 可以从 phase A 导入：

```text
DINO adapter / backbone: frozen
residual U-Net
fusion neck
2D feature projection
V4 lifting related trainable modules
```

DINO 仍保持 frozen，和 phase A 对齐。

---

# 7. Structure Event Encoder

Observation 不是 updater 的最终输入。IForward 需要把当前 GS 参数和 observation 编成 structure event。

接口：

```python
class IForwardEventEncoder(nn.Module):
    def encode(
        self,
        *,
        gs_state: IForwardGSState,
        observation: IForwardObservationPack,
        step: Dict[str, Any],
    ) -> IForwardEventPack:
        ...
```

输出：

```python
@dataclass
class IForwardBranchEvent:
    event: Tensor          # [N_branch, C_event]
    support: Tensor        # [N_branch]
    obs_code: Tensor       # [N_branch, C_obs_code]
    valid: Tensor          # [N_branch]
    param_code: Tensor     # [N_branch, C_param]
    stable_ids: Optional[Tensor]
    object_ids: Optional[Tensor]
    cell_ids: Optional[Tensor]

@dataclass
class IForwardEventPack:
    bg: IForwardBranchEvent
    distant: Optional[IForwardBranchEvent]
    rigid: Optional[IForwardBranchEvent]

    source_frame_idx: int
    step_idx: int
    repeat_idx: int
    commit_observation_memory: bool
    update_optimizer_memory: bool
```

Event Encoder 结构：

```text
param / observation codec
    ↓
near/far routed structural encoder
    ↓
branch event pack
```

它可以从 phase A 的 struct event decoder / param obs codec 初始化，但应放在 IForward namespace 下。

---

# 8. Mamba Optimizer Memory

IForward memory 必须是完整的 point / cell / global Mamba optimizer memory。

不能用：

```text
GRU
MLP cache
EMA memory
普通 linear recurrent state
phase B VSM
只含 h 的简化 selective SSM
```

## 8.1 StreamingMambaCell

IForward 的 Mamba memory 应实现真正的 streaming Mamba step。

每个 memory key，例如一个 point、一个 cell、一个 object/global token，都有自己的 Mamba cache。

Mamba state 至少包含：

```text
conv_state
ssm_state
seen_count
```

典型形状：

```text
conv_state: [N_keys, d_inner, d_conv]
ssm_state:  [N_keys, d_inner, d_state]
seen:       [N_keys]
```

接口：

```python
class StreamingMambaCell(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: Union[int, str],
        out_dim: int,
    ):
        ...

    def step(
        self,
        *,
        x: Tensor,                 # [N_keys, d_model]
        conv_state: Tensor,        # [N_keys, d_inner, d_conv]
        ssm_state: Tensor,         # [N_keys, d_inner, d_state]
        write_mask: Tensor,        # [N_keys]
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        ...
```

`step` 必须执行 Mamba-style streaming update：

```text
input projection
causal conv state update
selective SSM parameter generation
SSM state update
output projection
write mask / valid mask gating
```

其中 `write_mask` 只能控制是否写入 state，不能把 Mamba 替换成普通 gated MLP。

---

## 8.2 Memory 层级

完整 memory 包含：

```text
IForwardMemory
├── point memory
│   ├── bg point Mamba
│   ├── distant point Mamba
│   └── rigid point Mamba
│
├── cell memory
│   ├── bg cell Mamba
│   ├── distant cell Mamba
│   └── rigid object-local cell Mamba
│
├── global memory
│   ├── bg global Mamba
│   ├── distant global Mamba
│   └── rigid object/global Mamba
│
└── read fusion
```

顶层接口：

```python
class IForwardMemory(nn.Module):
    def init_state(self, gs_state: IForwardGSState) -> IForwardMemoryState:
        ...

    def step(
        self,
        *,
        gs_state: IForwardGSState,
        event: IForwardEventPack,
        short_history: IForwardShortWindowHistory,
        state: IForwardMemoryState,
        step: Dict[str, Any],
    ) -> Tuple[IForwardMemoryReadPack, IForwardMemoryState]:
        ...
```

---

# 9. Point Mamba Memory

## 9.1 功能

Point memory 记录每个 GS point 的 optimizer history：

```text
这个点过去被怎样观测
support 是否稳定
过去 delta 是否稳定
当前 observation 是否和历史冲突
这个点是否应该更新
这个点应该更新 geometry 还是 appearance
```

Point memory 不直接生成 RGB，不替代 GS。

## 9.2 State

```python
@dataclass
class IForwardPointMambaState:
    bg_conv: Tensor
    bg_ssm: Tensor
    bg_seen: Tensor

    distant_conv: Optional[Tensor]
    distant_ssm: Optional[Tensor]
    distant_seen: Optional[Tensor]

    rigid_conv: Optional[Tensor]
    rigid_ssm: Optional[Tensor]
    rigid_seen: Optional[Tensor]
    rigid_stable_ids: Optional[Tensor]
```

形状：

```text
bg_conv:      [N_bg, d_inner_p, d_conv_p]
bg_ssm:       [N_bg, d_inner_p, d_state_p]
bg_seen:      [N_bg]

distant_conv: [N_dist, d_inner_p, d_conv_p]
distant_ssm:  [N_dist, d_inner_p, d_state_p]

rigid_conv:   [N_rigid_stable, d_inner_p, d_conv_p]
rigid_ssm:    [N_rigid_stable, d_inner_p, d_state_p]
```

## 9.3 Token

Point Mamba 的输入 token：

```text
point_token_i =
[
    event_i,
    param_code_i,
    obs_code_i,
    log1p(support_i),
    valid_i,
    previous_delta_code_i,
    branch_embed,
    step position code,
    frame position code,
    repeat position code,
    commit_observation_memory flag,
    update_optimizer_memory flag,
    short_window_point_context_i,
    cell_read_prev_i,
    global_read_prev_i
]
```

注意：

```text
event_i 是每次 repeat 都重新 observe 当前 GS 后得到的 event
commit flag 只在 repeat_idx == 0 为 true
repeat_idx > 0 时，Mamba 仍然更新 optimizer trajectory，但 token 必须知道这不是新 observation
```

## 9.4 Write mask

Point write mask：

```text
write_mask_i =
    valid_i
    * update_optimizer_memory
```

同时 token 内包含：

```text
commit_observation_memory
```

这样同一 frame 的 repeat 不会被误当作新独立观测。

如果需要更严格地区分 observation write 和 optimizer write，可以在 point token 内显式拆两段：

```text
observation_part = event / obs_code / support / valid
optimizer_part   = previous_delta / repeat code / current param code
```

并令：

```text
observation_part *= commit_observation_memory
optimizer_part   *= update_optimizer_memory
```

但 Mamba step 本身仍然每次 repeat 执行。

## 9.5 Rigid point memory mapping

rigid point 不能用当前 tensor row order 当长期 id。

必须使用：

```text
rigid stable point id
rigid object id
```

流程：

```text
current rigid rows
    ↓ stable_ids
gather memory rows by stable_ids
    ↓ Mamba step
scatter updated states back by stable_ids
```

如果出现新 rigid stable id：

```text
allocate zero Mamba state
seen = 0
```

---

# 10. Cell Mamba Memory

## 10.1 功能

Cell memory 记录局部空间块的 optimizer state。

它解决 point memory 过于局部的问题。

Cell memory 负责：

```text
局部几何一致性
一片表面的共同 drift
局部 opacity / scale 变化趋势
未观测邻近点保护
局部 history retention 风险
```

## 10.2 Cell assignment

### bg cell

bg 使用 world / segment coordinate voxel。

```text
cell_id_bg = voxelize(bg.means, voxel_size_bg)
```

state：

```text
bg_cell_conv: [N_bg_cells, d_inner_c, d_conv_c]
bg_cell_ssm:  [N_bg_cells, d_inner_c, d_state_c]
```

### distant cell

distant 使用 spherical / angular-range cell 更合适。

```text
direction = normalize(distant.means - ego_origin)
range_bin = bucket(log(distance))
theta_phi_bin = bucket(direction)
cell_id_distant = combine(theta_bin, phi_bin, range_bin)
```

原因：

```text
distant branch 的欧式 voxel 容易尺度不稳定
angular/range bin 更适合远景结构
```

### rigid cell

rigid 使用 object-local grid。

```text
local_xyz = transform_world_to_object_local(rigid.means, object_pose)
cell_id_rigid = object_id + local_voxel(local_xyz)
```

rigid cell 必须带 object id：

```text
global_rigid_cell_id = hash(object_id, local_cell_id)
```

## 10.3 Cell token

Cell event 由 point events 聚合得到。

```python
cell_event_c = weighted_mean(
    point_event_i,
    weight = support_i * valid_i,
    group = cell_id_i,
)
```

Cell token：

```text
cell_token_c =
[
    aggregated_event_c,
    aggregated_param_code_c,
    aggregated_obs_code_c,
    mean_support_c,
    valid_ratio_c,
    num_points_c,
    branch_embed,
    step/frame/repeat codes,
    commit_observation_memory flag,
    update_optimizer_memory flag,
    short_window_cell_context_c,
    global_read_prev
]
```

Cell Mamba step：

```text
cell_read, cell_state = cell_mamba.step(cell_token, cell_state)
```

然后把 cell read broadcast 回 point：

```text
cell_read_i = cell_read[cell_id_i]
```

---

# 11. Global / Object Mamba Memory

## 11.1 功能

Global memory 记录 sequence-level optimizer context。

它不保存细节，而保存：

```text
当前 rollout 的整体观测质量
当前 sequence 的更新风险
整体 exposure / brightness tendency
branch-level drift
rigid object-level stability
global update conservativeness
```

## 11.2 State

```python
@dataclass
class IForwardGlobalMambaState:
    bg_global_conv: Tensor          # [1, d_inner_g, d_conv_g]
    bg_global_ssm: Tensor           # [1, d_inner_g, d_state_g]
    bg_global_seen: Tensor          # [1]

    distant_global_conv: Optional[Tensor]
    distant_global_ssm: Optional[Tensor]
    distant_global_seen: Optional[Tensor]

    rigid_object_conv: Optional[Tensor]  # [N_objects, d_inner_g, d_conv_g]
    rigid_object_ssm: Optional[Tensor]   # [N_objects, d_inner_g, d_state_g]
    rigid_object_seen: Optional[Tensor]
    rigid_object_ids: Optional[Tensor]
```

## 11.3 Global token

bg global token：

```text
bg_global_token =
[
    weighted_mean(bg_event),
    mean_support,
    valid_ratio,
    mean_delta_prev,
    frame/repeat codes,
    short_window_global_context
]
```

distant global token 同理。

rigid object token：

```text
rigid_object_token[obj] =
[
    weighted_mean(rigid_event for object obj),
    mean_support_obj,
    valid_ratio_obj,
    object_id_embed,
    object_motion_code,
    frame/repeat codes,
    short_window_object_context
]
```

Global read broadcast：

```text
bg_global_read_i = bg_global_read.expand(N_bg)
distant_global_read_i = distant_global_read.expand(N_dist)
rigid_global_read_i = rigid_object_read[object_id_i]
```

---

# 12. Memory Read Fusion

Point / cell / global read 需要融合成 per-point memory context。

```python
@dataclass
class IForwardBranchMemoryRead:
    point_read: Tensor       # [N, Dp]
    cell_read: Tensor        # [N, Dc]
    global_read: Tensor      # [N, Dg]
    short_read: Tensor       # [N, Ds]
    fused: Tensor            # [N, Dm]

@dataclass
class IForwardMemoryReadPack:
    bg: IForwardBranchMemoryRead
    distant: Optional[IForwardBranchMemoryRead]
    rigid: Optional[IForwardBranchMemoryRead]

    aux: Dict[str, Any]
```

Fusion：

```text
raw_ctx_i = [
    point_read_i,
    cell_read_i,
    global_read_i,
    short_read_i,
    event_i,
    support_i,
    repeat_pos_code,
    commit_flag
]

fused_i = LayerNorm(
    Linear(raw_ctx_i)
)
```

这个 fusion 只是 memory read projection，不是替代 Mamba。真正的长期状态更新必须由 point/cell/global Mamba 完成。

---

# 13. Short-Window Uncompressed History

## 13.1 目的

Short-window history 是 IForward 的近程未压缩历史。

它不是 Mamba 的替代品，而是和 Mamba 分工：

```text
Mamba memory:
    压缩长期 optimizer state

short-window history:
    保存最近少量 frame/rollout 的未压缩观测、事件、render target
    用于近程 consistency 和 history-retention render loss

GS state:
    唯一真实 scene state
```

## 13.2 State

```python
@dataclass
class IForwardHistoryFrameEntry:
    scene_id: int
    segment_id: int
    episode_id: int
    frame_idx: int
    keyframe_idx: int

    refs_3cam: List[ImageRef]

    cameras: Any
    gt_images: Tensor
    masks: Dict[str, Tensor]

    bg_event: Optional[Tensor]
    distant_event: Optional[Tensor]
    rigid_event: Optional[Tensor]

    bg_support: Optional[Tensor]
    distant_support: Optional[Tensor]
    rigid_support: Optional[Tensor]

    bg_cell_summary: Optional[Tensor]
    distant_cell_summary: Optional[Tensor]
    rigid_cell_summary: Optional[Tensor]

    final_render_stats_when_committed: Dict[str, float]

@dataclass
class IForwardShortWindowHistory:
    max_frames: int
    entries: Deque[IForwardHistoryFrameEntry]
```

## 13.3 写入时机

每个 rollout 完成 final loss 后，把当前 rollout 的 input frames 写入 short-window。

```text
commit_rollout:
    for frame in plan.final_supervision.current_input_frames:
        store its 3cam render target pack
        store latest event/support summaries for that frame
        store cell summaries
        store final render stats
```

只写 input frames，不写 nearby frames。

原因：

```text
input frames 是模型真正吸收过的 evidence
nearby frames 只是 rollout-local render supervision
```

## 13.4 读取方式

在每个 step，short-window 根据当前 branch stable ids / cell ids 返回 context。

```text
short_point_ctx_i:
    recent stored event/support for same stable point id

short_cell_ctx_i:
    recent stored cell summary for same cell id

short_global_ctx:
    recent frame-level summary
```

对于不存在历史的点/cell：

```text
short context = zero
short valid = 0
```

## 13.5 用于 history-retention render loss

Short-window 保存最近 frames 的完整 3cam target pack，因此在后续 rollout final state 上可以重新 render 历史帧：

```text
L_history_short_window =
    render(current_final_G, short_window_entries.refs_3cam)
```

这不需要 scheduler 额外提供 previous-rollout refs，因为 short-window 已经缓存了必要 target pack。

episode reset 时必须清空 short-window。

---

# 14. Zero-Init Memory Injection

IForward 必须保证初始行为接近 phase-A-initialized optimizer。

因此 memory 注入必须 zero-init。

## 14.1 Updater 结构

```text
event
    ↓
base updater initialized from phase A
    ↓
delta_base

event + memory_read + state context
    ↓
zero-init memory residual updater
    ↓
delta_mem

delta = delta_base + delta_mem
```

同时允许 memory 以 zero-init event adapter 影响 base updater：

```text
event_for_base = event + zero_event_adapter(memory_read)

delta_base = base_updater(event_for_base)
```

初始时：

```text
zero_event_adapter(memory_read) = 0
delta_mem = 0
```

所以：

```text
IForward initial output == phase-A-like updater output
```

## 14.2 模块

```python
class ZeroInitMemoryInjector(nn.Module):
    def __init__(self, event_dim, memory_dim):
        self.event_adapter = MLP(...)
        self.delta_residual = MLP(...)

        zero_init_last_layer(self.event_adapter)
        zero_init_last_layer(self.delta_residual)

    def forward(self, event, memory_read, gs_param_code, delta_base):
        event_residual = self.event_adapter([memory_read, gs_param_code])
        event_for_base = event + event_residual

        delta_mem = self.delta_residual([
            event,
            memory_read,
            gs_param_code,
            delta_base_code,
        ])

        return event_for_base, delta_mem
```

## 14.3 Branch-specific injection

bg / distant / rigid 应有 branch-specific adapter 或 branch embedding。

```text
bg memory injection:
    normal geometry + appearance update

distant memory injection:
    conservative update scope
    default follows phase A clamps / branch policy

rigid memory injection:
    uses object id / object-level global read
    stable id aware
```

不要把所有 branch 强行共享一个无 branch awareness 的 memory adapter。

---

# 15. Memory-Conditioned Updater

接口：

```python
class IForwardMemoryConditionedUpdater(nn.Module):
    def forward(
        self,
        *,
        gs_state: IForwardGSState,
        event: IForwardEventPack,
        memory_read: IForwardMemoryReadPack,
        step: Dict[str, Any],
    ) -> Tuple[IForwardDeltaPack, Dict[str, Any]]:
        ...
```

内部：

```text
1. 对每个 branch 取 event
2. 用 zero-init event adapter 得到 event_for_base
3. base updater 预测 delta_base
4. 用 zero-init residual head 预测 delta_mem
5. delta = delta_base + delta_mem
6. branch-specific clamp
7. 返回 DeltaPack
```

DeltaPack 对齐 phase A：

```python
@dataclass
class IForwardBranchDelta:
    means: Tensor
    scales_log: Tensor
    quat_axis_angle: Tensor
    opacity_logit: Tensor
    sh: Tensor
    hidden: Optional[Tensor]
    confidence: Optional[Tensor]
    noop: Optional[Tensor]

@dataclass
class IForwardDeltaPack:
    bg: IForwardBranchDelta
    distant: Optional[IForwardBranchDelta]
    rigid: Optional[IForwardBranchDelta]
```

Clamp 继续使用 phase A 的 branch-specific delta clamp。

不新增非必要更新项。

---

# 16. Delta Apply

IForward 的 apply_delta 必须保持 3DGS 物理参数语义。

```python
class IForwardDeltaApplier(nn.Module):
    def forward(
        self,
        gs_state: IForwardGSState,
        delta: IForwardDeltaPack,
    ) -> IForwardGSState:
        ...
```

更新规则保持 phase A 兼容：

```text
means:
    means_new = means + clamped_delta_means

scales:
    scales_log_new = scales_log + clamped_delta_scales_log

quat:
    quat_new = apply_axis_angle_delta(quat, clamped_delta_quat)

opacity:
    opacity_logit_new = opacity_logit + clamped_delta_opacity

SH:
    sh_new = sh + clamped_delta_sh

hidden:
    hidden_new = hidden + clamped_delta_hidden
```

distant / rigid 的可更新范围必须和 config 对齐，不在实现里硬编码新增行为。

---

# 17. Loss 设计

用户要求不要自作主张增加 phase A 没用的辅助性 loss。

因此 IForward V1 只包含以下 loss：

```text
1. current render loss
2. history-retention render loss
3. nearby render loss
4. phase-A-aligned delta regularization
```

不加入：

```text
memory prediction loss
support prediction loss
confidence calibration loss
history risk prediction loss
delta bucket prediction loss
fisher loss
trust drift loss
render lock loss
query observation decoder loss
prefix loss
contrastive loss
```

---

## 17.1 Render loss 形式

所有 render loss 使用同一个 phase-A-aligned photometric loss：

```text
L_render = L1 + w_ssim * SSIM_loss
```

mask policy 对齐 phase A：

```text
non_sky_non_egocar
```

如果已有 mask BCE 是 phase A 当前配置的一部分，可以保留同样实现与权重；不要新增其他 mask/object/dynamic auxiliary。

---

## 17.2 Current render loss

Current render loss 来自 scheduler 的：

```text
target role = final_current_recon
```

scheduler 会提供 rollout 内所有 input frames 的 all-cam target refs。

在一个 rollout 中：

```text
input_frame_indices = [t0, t1, t2, ...]
```

final state 渲染：

```text
render final G on all input frames
```

其中最后一个 input frame 可视为当前吸收目标：

```text
current_frame = input_frame_indices[-1]
```

```text
L_current =
    render_loss(final_G, current_frame all 3cam)
```

---

## 17.3 History-retention render loss

History-retention render loss 分两部分。

### 17.3.1 In-rollout history retention

同一个 rollout 内，最终状态必须仍能 render 早先 input frames。

```text
history_frames_in_rollout = input_frame_indices[:-1]
```

```text
L_history_rollout =
    render_loss(final_G, history_frames_in_rollout all 3cam)
```

这直接使用 scheduler 已经给出的 `final_current_recon` refs，不需要新增 scheduler role。

具体拆分方式：

```python
current_indices = [
    target_idx
    for target_idx, ref, role in targets
    if role == "final_current_recon"
    and ref.frame_idx == input_frame_indices[-1]
]

history_rollout_indices = [
    target_idx
    for target_idx, ref, role in targets
    if role == "final_current_recon"
    and ref.frame_idx in input_frame_indices[:-1]
]
```

如果 rollout 只有 1 个 input frame：

```text
L_history_rollout = 0
```

### 17.3.2 Short-window history retention

Short-window 保存 previous rollouts 的 recent input frames。

```text
L_history_short_window =
    render_loss(final_G, state.short_history.entries all 3cam)
```

如果 short-window 为空：

```text
L_history_short_window = 0
```

episode reset 时 short-window 清空，因此不会跨 episode 错误监督。

### 17.3.3 Total history loss

```text
L_history =
    w_history_rollout * L_history_rollout
  + w_history_short   * L_history_short_window
```

这就是 IForward 的核心新 loss。

它不是辅助 loss，而是模型目标的一部分。

---

## 17.4 Nearby render loss

scheduler 提供：

```text
target role = final_nearby_rollout
```

这些 refs：

```text
不进入 evidence
不写 memory
只参与 final render loss
```

loss：

```text
L_nearby =
    render_loss(final_G, final_nearby_rollout refs)
```

权重和 warmup 可对齐 phase A 的 nearby render 配置。

如果 scheduler skip nearby：

```text
L_nearby = 0
```

---

## 17.5 Delta regularization

保留 phase A 的 delta regularization：

```text
delta_l2
opacity_delta_l2
sh_delta_l2
scale_barrier
nan_guard
```

不新增别的 regularizer。

对于 rollout 多个 step：

```text
L_reg = mean or weighted sum over rollout_deltas
```

建议默认：

```text
对所有 iterative steps 的 delta regularization 求平均
```

原因：

```text
IForward rollout-final render loss 不在每步监督
但每步 delta 仍应保持物理更新幅度稳定
```

---

## 17.6 Total loss

```text
L_total =
    w_current * L_current
  + w_history * L_history
  + w_nearby  * L_nearby
  + L_reg
```

推荐初始配置：

```yaml
losses:
  current_render:
    enable: true
    weight: 1.0
    mask_policy: non_sky_non_egocar

  history_render:
    enable: true
    rollout_weight: 1.0
    short_window_weight: 1.0
    mask_policy: non_sky_non_egocar

  nearby_render:
    enable: true
    weight: 0.25
    mask_policy: non_sky_non_egocar

  regularization:
    delta_l2_weight: 0.001
    opacity_delta_l2_weight: 0.0005
    sh_delta_l2_weight: 0.0005
    scale_barrier_weight: 0.002
    nan_guard: true

  disabled:
    query_observation: true
    prefix_render: true
    memory_auxiliary: true
    support_prediction: true
    confidence_calibration: true
    history_risk_prediction: true
    trust_drift: true
    render_lock: true
    fisher_nullspace: true
```

---

# 18. Training State Carry

IForward trainer 需要维护 runtime state cache。

由于 scheduler 是 episode-serial，trainer 可以只维护当前 episode state。

```python
class IForwardTrainer:
    current_state: Optional[IForwardState]

    def train_step(self, batch):
        plan = batch["_iforward"]

        if plan["reset_scene_state_before_rollout"]:
            self.current_state = None

        output = self.model.forward_rollout(
            batch=batch,
            carried_state=self.current_state,
        )

        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if plan["episode_end_after_rollout"]:
            self.current_state = None
        elif plan["carry_scene_state_after_rollout"]:
            self.current_state = output.state.detach_for_next_rollout()
        else:
            self.current_state = None
```

关键规则：

```text
rollout 内不断梯度
rollout 后 backward
rollout 后 detach carried state
episode 结束 reset
```

不要跨 rollout 保留 autograd graph。

---

# 19. IForward Config

完整配置应包含：

```yaml
model:
  family: IForward
  version: iforward_v1

  input:
    evidence_mode: one_frame_3cam
    require_num_cams_per_step: 3
    require_single_frame_per_step: true

  observation:
    type: iforward_v4_lifting
    init_from_phase_a: true
    dino_frozen: true

  event_encoder:
    type: iforward_routed_struct_event
    init_from_phase_a: true

  memory:
    enable: true

    mamba:
      implementation: streaming_mamba
      d_state: 16
      d_conv: 4
      expand: 2
      dt_rank: auto

    point:
      enable: true
      bg:
        d_model: 128
        read_dim: 64
      distant:
        d_model: 128
        read_dim: 64
      rigid:
        d_model: 128
        read_dim: 64
        key: stable_point_id

    cell:
      enable: true
      bg:
        voxel_size: 0.5
        d_model: 128
        read_dim: 64
      distant:
        cell_type: spherical_range
        theta_bins: 64
        phi_bins: 16
        range_bins: 16
        d_model: 128
        read_dim: 64
      rigid:
        cell_type: object_local_grid
        grid: [8, 8, 4]
        d_model: 128
        read_dim: 64

    global:
      enable: true
      bg:
        d_model: 128
        read_dim: 64
      distant:
        d_model: 128
        read_dim: 64
      rigid:
        type: object_level
        d_model: 128
        read_dim: 64

    short_window:
      enable: true
      max_frames: 4
      store_3cam_targets: true
      store_uncompressed_events: true
      store_cell_summaries: true
      clear_on_episode_reset: true

  updater:
    type: memory_conditioned_iterative_updater
    base_init_from_phase_a: true
    zero_init_memory_injection: true
    zero_init_delta_residual: true
    branch_specific_adapters: true

  delta_apply:
    use_phase_a_physical_update_rules: true
    branch_specific_clamp: true
```

---

# 20. Weight Import

新增工具：

```text
tools/iforward/import_phase_a_weights.py
```

输入：

```text
phase_a checkpoint
iforward config
```

输出：

```text
iforward init checkpoint
```

映射：

```text
phase_a.image_feature_extractor.*
    -> iforward.observation.image_feature_extractor.*

phase_a.struct_decoder.*
    -> iforward.event_encoder.struct_event.*

phase_a.param_obs_codec.*
    -> iforward.event_encoder.param_obs_codec.*

phase_a.posterior_updater.*
    -> iforward.updater.base_updater.*
```

不导入：

```text
phase_b modules
VSM modules
long memory modules
offset decoder
GRU modules
history gate modules
```

IForward 新增模块初始化：

```text
point/cell/global Mamba memory:
    standard random init

zero-init event adapter:
    last layer weight = 0
    last layer bias = 0

zero-init delta residual:
    last layer weight = 0
    last layer bias = 0
```

导入后必须满足：

```text
memory read = zero contribution
delta_mem = 0
event_residual = 0
```

因此单帧 3cam 行为应接近 phase-A-initialized updater。

---

# 21. Batch 解析

模型需要从 batch 中构建 refs 到 tensor index 的映射。

```python
def build_ref_maps(batch):
    source_refs = batch["_iforward"]["evidence_refs_flat"]
    target_refs = batch["_iforward"]["target_refs_flat"]

    source_ref_to_index = {
        tuple(ref): i for i, ref in enumerate(source_refs)
    }
    target_ref_to_index = {
        tuple(ref): i for i, ref in enumerate(target_refs)
    }

    return source_ref_to_index, target_ref_to_index
```

每个 step：

```python
step_source_indices = [
    source_ref_to_index[tuple(ref)]
    for ref in step["evidence_refs"]
]
```

必须校验：

```text
len(step_source_indices) == 3
all source refs have same frame_idx
```

然后构建：

```python
IForwardStepViewPack:
    images: 3cam images
    cameras: 3cam camera params
    masks: 3cam masks
    source_frame_idx
    cam_indices
```

---

# 22. Rollout Final Loss 解析

scheduler target roles：

```text
final_current_recon
final_nearby_rollout
```

IForward loss builder：

```python
def split_target_indices(plan):
    input_frames = plan["input_frame_indices"]
    latest_frame = input_frames[-1]
    history_frames = set(input_frames[:-1])

    current_indices = []
    history_rollout_indices = []
    nearby_indices = []

    for idx, (ref, role) in enumerate(zip(
        plan["target_refs_flat"],
        plan["target_roles_flat"],
    )):
        frame_idx = int(ref[0])

        if role == "final_current_recon":
            if frame_idx == latest_frame:
                current_indices.append(idx)
            elif frame_idx in history_frames:
                history_rollout_indices.append(idx)
            else:
                raise ValueError("unexpected final_current_recon frame")

        elif role == "final_nearby_rollout":
            nearby_indices.append(idx)

        else:
            raise ValueError(f"unknown IForward target role: {role}")

    return current_indices, history_rollout_indices, nearby_indices
```

然后：

```text
L_current = render_loss(current_indices)
L_history_rollout = render_loss(history_rollout_indices)
L_nearby = render_loss(nearby_indices)
L_history_short = render_loss(short_window_entries)
```

---

# 23. Logging

IForward logs 必须围绕当前优化、历史保持、memory 行为和物理 delta。

不要只记录 loss。

## 23.1 Reconstruction logs

```text
iforward/loss_total
iforward/loss_current
iforward/loss_history_rollout
iforward/loss_history_short_window
iforward/loss_nearby
iforward/loss_delta_reg

iforward/psnr_current
iforward/psnr_history_rollout
iforward/psnr_history_short_window
iforward/psnr_nearby

iforward/num_current_refs
iforward/num_history_rollout_refs
iforward/num_history_short_window_refs
iforward/num_nearby_refs
```

## 23.2 Scheduler/state logs

```text
iforward/scene_id
iforward/segment_id
iforward/episode_id
iforward/rollout_id_global
iforward/rollout_idx_in_episode
iforward/inner_K
iforward/actual_blocks_per_rollout
iforward/repeats_per_block

iforward/reset_scene_state_before_rollout
iforward/carry_scene_state_after_rollout
iforward/episode_end_after_rollout
iforward/short_window_size
```

## 23.3 Memory logs

```text
iforward/memory/point_bg_state_norm
iforward/memory/point_distant_state_norm
iforward/memory/point_rigid_state_norm

iforward/memory/cell_bg_state_norm
iforward/memory/cell_distant_state_norm
iforward/memory/cell_rigid_state_norm

iforward/memory/global_bg_state_norm
iforward/memory/global_distant_state_norm
iforward/memory/global_rigid_state_norm

iforward/memory/point_write_ratio
iforward/memory/cell_write_ratio
iforward/memory/global_write_ratio

iforward/memory/commit_observation_steps
iforward/memory/optimizer_update_steps
```

## 23.4 Delta logs

```text
iforward/delta/bg_means_norm
iforward/delta/bg_scale_norm
iforward/delta/bg_opacity_norm
iforward/delta/bg_sh_norm

iforward/delta/distant_means_norm
iforward/delta/distant_scale_norm
iforward/delta/distant_opacity_norm
iforward/delta/distant_sh_norm

iforward/delta/rigid_means_norm
iforward/delta/rigid_scale_norm
iforward/delta/rigid_opacity_norm
iforward/delta/rigid_sh_norm
```

---

# 24. Memory Ablation Validation

IForward 必须有 memory ablation validation。
这些 ablation 只用于 validation，不作为训练 loss。

## 24.1 Validation modes

```text
full:
    point/cell/global Mamba + short-window 全启用

zero_all_memory:
    point/cell/global read 全置零
    short-window read 置零
    zero-init residual 不接收 memory
    等价于 phase-A-initialized iterative optimizer baseline

zero_point_memory:
    point read 置零
    cell/global/short 保留

zero_cell_memory:
    cell read 置零
    point/global/short 保留

zero_global_memory:
    global read 置零
    point/cell/short 保留

drop_short_window:
    short-window context 置零
    short-window history render loss 在 validation 中仍可单独报告
    但模型 read 不使用 short context

freeze_memory_write:
    memory read 使用当前 state
    不更新 memory state

shuffle_memory:
    在同 scene/segment 内打乱 memory state 与 frame order
    只用于 sanity check
```

## 24.2 Metrics

每个 mode 记录：

```text
current_psnr
history_rollout_psnr
history_short_window_psnr
nearby_psnr

retention_gap_rollout
retention_gap_short_window

memory_ablation_current_delta
memory_ablation_history_delta
```

核心判断：

```text
full > zero_all_memory on history retention
full > drop_short_window on recent history retention
full > zero_cell_memory on local consistency
full > zero_global_memory on rollout stability
```

如果 full 只提升 current PSNR，但 history 不提升，说明 IForward memory 没有学到 persistent optimizer。

---

# 25. Validation Protocols

## 25.1 Single-frame 3cam compatibility

目的：

```text
确保 IForward 没破坏 phase-A-like 单帧行为
```

设置：

```text
blocks_per_rollout = 1
repeats_per_block = K
evidence = one frame 3cam
memory enabled but zero-init contribution
```

看：

```text
current PSNR
delta norm
memory residual norm
```

要求：

```text
zero-init 下 memory residual 初始接近 0
```

## 25.2 Short-sequence retention

设置：

```text
blocks_per_rollout = 2 / 3 / 4
repeats_per_block = 4 / 6 / 8
each step one frame 3cam
```

看：

```text
latest frame PSNR
earlier input frame PSNR after final state
history retention gap
```

## 25.3 Cross-rollout retention

设置：

```text
episode contains multiple rollouts
state carried across rollouts
short-window history enabled
```

看：

```text
previous rollout frames rendered after current rollout
short-window history PSNR
forgetting curve by frame age
```

## 25.4 Memory ablation

对同一 validation rollout/episode 跑：

```text
full
zero_all_memory
zero_point_memory
zero_cell_memory
zero_global_memory
drop_short_window
freeze_memory_write
shuffle_memory
```

输出表：

```text
mode | current_psnr | history_rollout_psnr | history_short_psnr | retention_gap
```

---

# 26. 实现顺序

虽然不是简化版实现，但工程落地仍应按依赖顺序完成。

## Step 1：建立 IForward namespace

产物：

```text
models/iforward/*
configs/iforward/*
tools/iforward/import_phase_a_weights.py
```

要求：

```text
没有继承 stage6_0 trainer
没有 phase_a / phase_b mode
```

---

## Step 2：实现 IForwardState / GSState

产物：

```text
IForwardGSState
IForwardMemoryState
IForwardShortWindowHistory
IForwardState
```

要求：

```text
支持 episode reset
支持 rollout carry
支持 detach_for_next_rollout
支持 rigid stable id mapping
```

---

## Step 3：实现 Observation Backbone

产物：

```text
IForwardObservationBackbone
IForwardObservationPack
IForwardStepViewPack
```

要求：

```text
每个 step 只接受一帧 3cam
复用 phase-A-aligned render -> 2D feature -> V4 lifting 机制
输出 per-branch observation
```

---

## Step 4：实现 Event Encoder

产物：

```text
IForwardEventEncoder
IForwardEventPack
```

要求：

```text
输出 bg / distant / rigid event
包含 support / obs_code / valid / param_code
支持 phase A 权重导入
```

---

## Step 5：实现 StreamingMambaCell

产物：

```text
StreamingMambaCell
MambaState tensors:
    conv_state
    ssm_state
    seen
```

要求：

```text
是真正 Mamba streaming step
不是 GRU
不是 MLP memory
不是 phase B StreamingSelectiveSSMBranch 的简化替代
```

---

## Step 6：实现 point/cell/global Mamba memory

产物：

```text
PointMambaMemory
CellMambaMemory
GlobalMambaMemory
IForwardMemory
IForwardMemoryReadPack
```

要求：

```text
point memory 按 stable point id 对齐
cell memory 按 bg/distant/rigid cell id 聚合
global memory 按 branch/object 聚合
memory step 使用 scheduler flags
```

---

## Step 7：实现 short-window uncompressed history

产物：

```text
IForwardShortWindowHistory
IForwardHistoryFrameEntry
```

要求：

```text
保存 recent input frames 的 3cam target pack
保存未压缩 event/support/cell summaries
episode reset 清空
支持 render history loss
支持 short context read
```

---

## Step 8：实现 zero-init memory-conditioned updater

产物：

```text
IForwardMemoryConditionedUpdater
ZeroInitMemoryInjector
IForwardDeltaPack
IForwardDeltaApplier
```

要求：

```text
base updater 从 phase A 初始化
memory event adapter zero-init
memory delta residual zero-init
branch-specific clamp
```

---

## Step 9：实现 rollout final losses

产物：

```text
IForwardRolloutLoss
IForwardRenderLoss
IForwardDeltaRegularization
```

要求：

```text
current render
history rollout render
history short-window render
nearby render
phase-A-aligned delta regularization
无额外辅助 loss
```

---

## Step 10：实现 validation + ablation

产物：

```text
IForwardValidator
IForwardMemoryAblationRunner
```

要求：

```text
支持 full / zero_all / zero_point / zero_cell / zero_global / drop_short_window / freeze_write / shuffle_memory
输出 retention gap
```

---

# 27. 非目标

IForward V1 不做以下内容：

```text
不做 query observation decoder
不做 prefix render loss
不做 VSM
不做 phase B long memory
不做 offset-only decoder
不做 GRU
不做 memory auxiliary prediction loss
不做 support prediction loss
不做 confidence calibration loss
不做 Fisher / trust drift / render lock
不做 sequence-level image transformer
不做 all-frame attention
不做一次性 forward geometry reconstruction
```

这些都与当前 IForward 的核心目标无关，容易把模型带回 phase B 的错误方向。

---

# 28. 最终实现定义

IForward 的最终实现应满足：

```text
1. 每个 optimizer step 输入一帧 3cam。
2. observation backbone 对齐 phase A 的 render residual + V4 lifting。
3. event encoder 对齐 phase A 的 structure event 表达。
4. point/cell/global memory 使用真正 streaming Mamba。
5. short-window 保存未压缩 recent history。
6. memory 通过 zero-init adapter 注入 updater。
7. delta 更新严格保持 3DGS 物理结构。
8. rollout final loss 同时监督 current 和 history。
9. previous rollout history 通过 short-window render loss 保持。
10. validation 必须包含 memory ablation。
```

一句话定义：

> **IForward 是一个独立的、从 phase A 初始化但不继承 phase A 的 Mamba-conditioned 3DGS iterative optimizer。它每次只读取一帧 3cam，通过 observation → event → point/cell/global Mamba memory → zero-init memory-conditioned delta → render supervision 的闭环，在短序列中优化当前 GS，并用 history-retention render loss 防止历史帧遗忘。**
