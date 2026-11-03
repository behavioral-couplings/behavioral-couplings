#!/usr/bin/env bash
set -euo pipefail

_COUPLING_SFT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$_COUPLING_SFT_SCRIPT_DIR/../../.." && pwd)}"
source "$COUPLING_HOME/behavioral_coupling/utils/load_env.sh"
export WANDB_PROJECT="${WANDB_PROJECT:-behavioral_coupling-SFT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

coupling_bool_enabled() {
	case "${1:-}" in
		true | 1 | yes) return 0 ;;
		*) return 1 ;;
	esac
}

coupling_require_env() {
	local name="$1"
	if [[ -z "${!name:-}" ]]; then
		echo "Missing required environment variable: $name" >&2
		exit 1
	fi
}

coupling_select_deepspeed() {
 DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
 if ! command -v "$DEEPSPEED_BIN" >/dev/null 2>&1; then
  echo "DeepSpeed not found; activate the training environment or set DEEPSPEED_BIN." >&2
  exit 1
 fi
}

coupling_formation_hparams() {
	DEFAULT_DEEPSPEED_CONFIG="$COUPLING_HOME/behavioral_coupling/utils/ds_zero2.json"
	DEFAULT_GPU_IDS="0,1,2,3"
	DEFAULT_MAX_LENGTH="2048"
	DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE="4"
	DEFAULT_GRADIENT_ACCUMULATION_STEPS="4"
	DEFAULT_MASTER_WEIGHT_DTYPE="bf16"
}

coupling_run_formation() {
	coupling_require_env MODEL_NAME
	coupling_require_env OUTPUT_MODEL_NAME
	coupling_require_env DATA_FILE
	coupling_formation_hparams
	PYTHON_BIN="${PYTHON_BIN:-python3}"
	USE_DEEPSPEED="${USE_DEEPSPEED:-true}"
	if coupling_bool_enabled "$USE_DEEPSPEED"; then
		coupling_select_deepspeed
	fi

	GPU_IDS="${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU_IDS}"
	DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$DEFAULT_DEEPSPEED_CONFIG}"
	TOKENIZING_MODEL="${TOKENIZING_MODEL:-$MODEL_NAME}"

	MAX_LENGTH="${MAX_LENGTH:-$DEFAULT_MAX_LENGTH}"
	NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
	LEARNING_RATE="${LEARNING_RATE:-2e-5}"
	WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
	MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
	LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
	WARMUP_STEPS="${WARMUP_STEPS:-0}"
	PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-$DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE}"
	GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}"
	MASTER_WEIGHT_DTYPE="${MASTER_WEIGHT_DTYPE:-$DEFAULT_MASTER_WEIGHT_DTYPE}"
	MIXED_PRECISION_DTYPE="${MIXED_PRECISION_DTYPE:-bf16}"
	SEED="${SEED:-42}"
	DATA_SEED="${DATA_SEED:-42}"
	SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
	SAVE_STEPS="${SAVE_STEPS:-100}"
	SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
	LOGGING_STEPS="${LOGGING_STEPS:-1}"
	REPORT_TO="${REPORT_TO:-none}"
	GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

	ARGS=()
	if coupling_bool_enabled "$SAVE_ONLY_MODEL"; then
		ARGS+=(--save_only_model)
	fi
	if coupling_bool_enabled "$GRADIENT_CHECKPOINTING"; then
		ARGS+=(--gradient_checkpointing)
	fi

	LAUNCHER=()
	SFT_RUNTIME_ARGS=()
	if coupling_bool_enabled "$USE_DEEPSPEED"; then
		LAUNCHER=(env -u CUDA_VISIBLE_DEVICES "$DEEPSPEED_BIN" --include "localhost:$GPU_IDS" "$@")
		SFT_RUNTIME_ARGS=(--deepspeed "$DEEPSPEED_CONFIG")
	else
		LAUNCHER=(env "CUDA_VISIBLE_DEVICES=$GPU_IDS" "$PYTHON_BIN")
		SFT_RUNTIME_ARGS=(--device_map_auto)
	fi

	"${LAUNCHER[@]}" "$COUPLING_HOME/behavioral_coupling/sft/sft.py" \
		--model_name "$MODEL_NAME" \
		--tokenizing_model "$TOKENIZING_MODEL" \
		--data_files "$DATA_FILE" \
		--output_model_name "$OUTPUT_MODEL_NAME" \
		--max_length "$MAX_LENGTH" \
		--num_train_epochs "$NUM_TRAIN_EPOCHS" \
		--learning_rate "$LEARNING_RATE" \
		--weight_decay "$WEIGHT_DECAY" \
		--max_grad_norm "$MAX_GRAD_NORM" \
		--lr_scheduler_type "$LR_SCHEDULER_TYPE" \
		--warmup_steps "$WARMUP_STEPS" \
		--per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
		--gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
		--master_weight_dtype "$MASTER_WEIGHT_DTYPE" \
		--mixed_precision_dtype "$MIXED_PRECISION_DTYPE" \
		--seed "$SEED" \
		--data_seed "$DATA_SEED" \
		"${SFT_RUNTIME_ARGS[@]}" \
		--save_strategy "$SAVE_STRATEGY" \
		--save_steps "$SAVE_STEPS" \
		"${ARGS[@]}" \
		--logging_steps "$LOGGING_STEPS" \
		--report_to "$REPORT_TO"
}
