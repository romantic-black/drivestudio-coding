#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/drivestudio-coding}"
CONDA_ENV="${CONDA_ENV:-drivestudio-new}"
CONFIG_FILE="${CONFIG_FILE:-configs/viewer/demo_minimal_streetforward_stage5_6_one_segment_overfit_render.yaml}"
CONFIG_FILE="${CONFIG_FILE:-configs/viewer/demo_minimal_streetforward_stage5_6_one_segment_overfit_v8_video.yaml}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-${1:-/root/autodl-tmp/outputs/minimal_sf_stage5_6_one_segment_overfit_v8/stage5_6_one_segment_overfit_v8_scene001_segment000/checkpoints/minimal_sf_stage5_6_one_segment_overfit_v8_step20000.pt}}"
SCENE_ID="${SCENE_ID:-114}"
SEGMENT_ID="${SEGMENT_ID:-0}"
SEQUENCE_START_POS="${SEQUENCE_START_POS:-0}"
WINDOW_SIZE="${WINDOW_SIZE:-26}"
STEPS_PER_INPUT="${STEPS_PER_INPUT:-4}"
SUBFRAMES_PER_INTERVAL="${SUBFRAMES_PER_INTERVAL:-4}"
CAMERA_MODE="${CAMERA_MODE:-front}"
SKY_STEPS="${SKY_STEPS:-24}"
RENDER_HEIGHT="${RENDER_HEIGHT:-null}"
RENDER_WIDTH="${RENDER_WIDTH:-null}"
DEMO_MODE="${DEMO_MODE:-frozen_recurrent_inference}"

OUT_DIR="${OUT_DIR:-/root/autodl-tmp/outputs/minimal_sf_stage5_6_one_segment_overfit_v8/stage5_6_one_segment_overfit_v8_scene001_segment000/render_step20000_normal_sky${SKY_STEPS}}"
SKY_CONFIG="${SKY_CONFIG:-configs/skybranch_stage5_4_exp002.yaml}"
SKY_CHECKPOINT="${SKY_CHECKPOINT:-/root/autodl-tmp/outputs/skybranch_stage5_4_exp002/checkpoints/skybranch_resume_step_100000.pth}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" conda run -n "${CONDA_ENV}" python tools/demo_minimal_streetforward_stage5_6_video.py \
  --config_file "${CONFIG_FILE}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --scene_id "${SCENE_ID}" \
  --segment_id "${SEGMENT_ID}" \
  --sequence_start_pos "${SEQUENCE_START_POS}" \
  --camera_mode "${CAMERA_MODE}" \
  --ckpt_load_mode full_state \
  output_name=stage5_6_one_segment_overfit_v8_render \
  demo.mode="${DEMO_MODE}" \
  batch_eval.runtime.mode=inference_only \
  video.output.dir="${OUT_DIR}" \
  video.output.name="stage5_6_one_segment_overfit_step20000_normal_sky${SKY_STEPS}" \
  video.output.layout=single \
  video.output.write_combined=true \
  video.output.write_separate_per_camera=false \
  video.output.save_all_images=true \
  video.output.save_png_frames=false \
  video.output.fps=null \
  video.render.height="${RENDER_HEIGHT}" \
  video.render.width="${RENDER_WIDTH}" \
  video.reconstruction.window_size="${WINDOW_SIZE}" \
  video.reconstruction.window_stride="${WINDOW_SIZE}" \
  video.reconstruction.require_full_window=true \
  video.reconstruction.window_policy=sliding \
  video.reconstruction.max_windows=1 \
  video.reconstruction.transition_frames_before=0 \
  video.reconstruction.transition_frames_after=0 \
  video.reconstruction.input_gap_frames=0 \
  video.reconstruction.input_offsets=[] \
  video.reconstruction.steps_per_input="${STEPS_PER_INPUT}" \
  video.reconstruction.max_target_frames_including_source="${WINDOW_SIZE}" \
  video.reconstruction.block_order=step_major \
  video.reconstruction.step_major_switch_interval_steps=4 \
  video.interpolation.source_fps=10.0 \
  video.interpolation.subframes_per_source_interval="${SUBFRAMES_PER_INTERVAL}" \
  video.interpolation.include_window_tail_interval=true \
  video.interpolation.rigid_frame_policy=nearest \
  video.update_cameras.ids=[0,1,2] \
  video.update_cameras.names=[front,front_left,front_right] \
  video.sky.enable=true \
  video.sky.config_file="${SKY_CONFIG}" \
  video.sky.checkpoint="${SKY_CHECKPOINT}" \
  video.sky.strict=true \
  video.sky.freeze_params=true \
  video.sky.load_runtime_state=false \
  video.sky.require_runtime_state=false \
  video.sky.reuse_single_state=true \
  video.sky.compose_mode=alpha_gap \
  video.sky.alpha_scale=1.0 \
  video.sky.pre_render_update_steps="${SKY_STEPS}" \
  video.sky.pre_render_update_each_window=true \
  video.sky.reset_runtime_before_export=true \
  video.sky.reset_runtime_per_window=false \
  video.sky.pre_render_fail_on_error=false
