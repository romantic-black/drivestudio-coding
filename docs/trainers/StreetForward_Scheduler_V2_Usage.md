# StreetForward Scheduler V2 Usage

本文档给出 Scheduler v2 的落地使用方式（怎么配、怎么跑、怎么排错）。  
设计约束请先阅读：`docs/trainers/StreetForward_Scheduler_V2_Design.md`。

---

## 1. 入口脚本

当前接入入口：

- `tools/train_minimal_streetforward_stage4_1_one_segment.py`

该入口已切换为：

- `MultiSceneDatasetV2`
- `TrainSchedulerV2`
- one-segment 固定 `scene_id/segment_id` 训练

---

## 2. 配置文件

使用：

- `configs/minimal_streetforward_stage4_1_one_segment_v2.yaml`

### 必填项（fast-fail）

`scheduler_v2` 下必须提供：

- `alpha_steps_per_keyframe`
- `min_steps_per_segment`
- `max_steps_per_segment`
- `source_hold_steps`
- `num_target_frames_total`
- `target_include_source: true`

若缺失或非法，脚本会立即报错终止。

---

## 3. 启动命令

```bash
conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding \
python tools/train_minimal_streetforward_stage4_1_one_segment.py \
  --config_file configs/minimal_streetforward_stage4_1_one_segment_v2.yaml
```

短跑冒烟：

```bash
conda run -n drivestudio-new env PYTHONPATH=/root/drivestudio-coding \
python tools/train_minimal_streetforward_stage4_1_one_segment.py \
  --config_file configs/minimal_streetforward_stage4_1_one_segment_v2.yaml \
  --max_steps 2
```

---

## 4. 运行时行为（重点）

- `steps_per_segment = clamp(round(alpha * num_keyframes), min, max)`
- source 每 `source_hold_steps` 刷新一次
- 每步 target 重采样，但 `target[0]` 恒为 source frame
- one-segment 模式会固定在配置指定的 `(scene_id, segment_id)` 上循环 epoch

---

## 5. 日志检查建议

训练行应同时关注：

- `epoch_idx`
- `global_step`
- `segment_local_step`
- `segment_step_budget`
- `source_frame_idx`
- `source_block_step`

若 `source_block_step` 达到阈值后 `source_frame_idx` 不变化，说明 source 刷新逻辑异常。

---

## 6. 常见问题

- `scheduler_v2.target_include_source must be true`  
  配置错误，v2 不支持 false。

- `fixed_segment_id out of range`  
  one-segment 指定段不存在；检查 `one_segment.segment_id`。

- `No non-source keyframes available for extra target sampling`  
  该段可用 keyframe 不足以满足当前 `num_target_frames_total`，降低 target 数或调整分段配置。

