#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_FILE="configs/minimal_streetforward_stage5_6_one_segment_overfit_v8.yaml"
CKPT=""
MAX_STEPS=""
SCENE_ID=""
SEGMENT_ID=""
OUTPUT_ROOT=""
MIN_PSNR=""
EVAL_EVERY=""
INIT_WEIGHTS_ONLY=0
EXTRA_OPTS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_stage5_6_one_segment_overfit_v8.sh [options]

Runs Stage 5.6 one-segment overfit training with the V8 frame-level scheduler.

Options:
  --config-file PATH       Config YAML.
  --ckpt PATH              Pretrained checkpoint to initialize from.
  --init-weights-only      Only restore model weights from --ckpt.
  --no-init-weights-only   Restore model weights and optimizer state from --ckpt. This is the default.
  --max-steps N            Override training step count via CLI.
  --scene-id N             Override one_segment.scene_id and train_scene_ids.
  --segment-id N           Override one_segment.segment_id.
  --output-root PATH       Override logging.output_root.
  --min-psnr VALUE         Override node-state export PSNR threshold.
  --eval-every N           Override overfit segment eval interval in episodes.
  --extra-opt KEY=VALUE    Extra OmegaConf override. Can be repeated.
  -h, --help               Show this help.

Example:
  scripts/run_stage5_6_one_segment_overfit_v8.sh \
    --ckpt /path/to/pretrained.pt \
    --scene-id 1 \
    --segment-id 0 \
    --min-psnr 25.0 \
    --eval-every 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --ckpt|--init-checkpoint)
      CKPT="$2"
      shift 2
      ;;
    --init-weights-only)
      INIT_WEIGHTS_ONLY=1
      shift
      ;;
    --no-init-weights-only)
      INIT_WEIGHTS_ONLY=0
      shift
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --scene-id)
      SCENE_ID="$2"
      shift 2
      ;;
    --segment-id)
      SEGMENT_ID="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --min-psnr)
      MIN_PSNR="$2"
      shift 2
      ;;
    --eval-every)
      EVAL_EVERY="$2"
      shift 2
      ;;
    --extra-opt)
      EXTRA_OPTS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file does not exist: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ -n "${CKPT}" && ! -f "${CKPT}" ]]; then
  echo "Checkpoint does not exist: ${CKPT}" >&2
  exit 1
fi

CMD=(python tools/train_minimal_streetforward_stage5_6_one_segment_overfit_v8.py --config_file "${CONFIG_FILE}")
if [[ -n "${MAX_STEPS}" ]]; then
  CMD+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${CKPT}" ]]; then
  CMD+=(--init_checkpoint "${CKPT}")
  if (( INIT_WEIGHTS_ONLY == 1 )); then
    CMD+=(--init_weights_only)
  fi
fi

OPTS=()
if [[ -n "${SCENE_ID}" ]]; then
  OPTS+=("one_segment.scene_id=${SCENE_ID}")
  OPTS+=("scheduler_v8.traversal.fixed_scene_id=${SCENE_ID}")
  OPTS+=("data.train_scene_ids=[${SCENE_ID}]")
fi
if [[ -n "${SEGMENT_ID}" ]]; then
  OPTS+=("one_segment.segment_id=${SEGMENT_ID}")
  OPTS+=("scheduler_v8.traversal.fixed_segment_id=${SEGMENT_ID}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
  OPTS+=("logging.output_root=${OUTPUT_ROOT}")
fi
if [[ -n "${MIN_PSNR}" ]]; then
  OPTS+=("overfit_segment_eval.export_node_state.min_psnr=${MIN_PSNR}")
fi
if [[ -n "${EVAL_EVERY}" ]]; then
  OPTS+=("overfit_segment_eval.trigger.validate_every_n_episodes=${EVAL_EVERY}")
fi
OPTS+=("${EXTRA_OPTS[@]}")

printf 'Running:'
printf ' %q' "${CMD[@]}" "${OPTS[@]}"
printf '\n'
exec "${CMD[@]}" "${OPTS[@]}"
