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

coupling_behavior_ft_hparams() {
	DEFAULT_DEEPSPEED_CONFIG="$COUPLING_HOME/behavioral_coupling/utils/ds_zero2.json"
	DEFAULT_GPU_IDS="0,1,2,3"
	DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE="4"
	DEFAULT_GRADIENT_ACCUMULATION_STEPS="1"
	DEFAULT_MASTER_WEIGHT_DTYPE="bf16"
	DEFAULT_MERGE_LORA_ON_SAVE="true"
}

coupling_run_lora_sft() {
	local hparams_func="$1"
	shift

	coupling_require_env MODEL_NAME
	coupling_require_env OUTPUT_MODEL_NAME
	coupling_require_env DATA_FILE
	"$hparams_func"
	coupling_select_deepspeed

	GPU_IDS="${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU_IDS}"
	DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$DEFAULT_DEEPSPEED_CONFIG}"
	DEEPSPEED_GPU_ARGS=(--include "localhost:$GPU_IDS")
	TOKENIZING_MODEL="${TOKENIZING_MODEL:-$MODEL_NAME}"

	MAX_LENGTH="${MAX_LENGTH:-2048}"
	NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
	LEARNING_RATE="${LEARNING_RATE:-1e-5}"
	WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
	MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
	LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
	WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
	PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-$DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE}"
	GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}"
	MASTER_WEIGHT_DTYPE="${MASTER_WEIGHT_DTYPE:-$DEFAULT_MASTER_WEIGHT_DTYPE}"
	MIXED_PRECISION_DTYPE="${MIXED_PRECISION_DTYPE:-bf16}"
	SEED="${SEED:-42}"
	DATA_SEED="${DATA_SEED:-42}"
	SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
	SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
	LOGGING_STEPS="${LOGGING_STEPS:-1}"
	REPORT_TO="${REPORT_TO:-none}"
	DISABLE_THINKING="${DISABLE_THINKING:-true}"
	ASSISTANT_LOSS_MODE="${ASSISTANT_LOSS_MODE:-last_response}"
	USE_LORA="${USE_LORA:-true}"
	MERGE_LORA_ON_SAVE="${MERGE_LORA_ON_SAVE:-$DEFAULT_MERGE_LORA_ON_SAVE}"
	LORA_ADAPTER_OUTPUT_MODEL_NAME="${LORA_ADAPTER_OUTPUT_MODEL_NAME:-${OUTPUT_MODEL_NAME}-adapter-tmp}"
	LORA_SNAPSHOT_STEPS="${LORA_SNAPSHOT_STEPS:-}"
	LORA_R="${LORA_R:-64}"
	LORA_ALPHA="${LORA_ALPHA:-128}"
	LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
	LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
	LORA_BIAS="${LORA_BIAS:-none}"
	OVERWRITE_MERGED_LORA="${OVERWRITE_MERGED_LORA:-true}"
	MERGED_SAVE_DTYPE="${MERGED_SAVE_DTYPE:-bf16}"

	ARGS=()
	if [[ -n "${DDP_TIMEOUT:-}" ]]; then
		ARGS+=(--ddp_timeout "$DDP_TIMEOUT")
	fi
	if [[ -n "${REASONING_EFFORT:-}" ]]; then
		ARGS+=(--reasoning_effort "$REASONING_EFFORT")
	fi
	if coupling_bool_enabled "$DISABLE_THINKING"; then
		ARGS+=(--disable_thinking)
	else
		ARGS+=(--enable_thinking)
	fi
	if coupling_bool_enabled "$SAVE_ONLY_MODEL"; then
		ARGS+=(--save_only_model)
	fi
	if coupling_bool_enabled "$USE_LORA"; then
		ARGS+=(--use_lora)
		ARGS+=(--lora_adapter_output_model_name "$LORA_ADAPTER_OUTPUT_MODEL_NAME")
		if [[ -n "$LORA_SNAPSHOT_STEPS" ]]; then
			ARGS+=(--lora_snapshot_steps "$LORA_SNAPSHOT_STEPS")
		fi
		ARGS+=(--lora_r "$LORA_R")
		ARGS+=(--lora_alpha "$LORA_ALPHA")
		ARGS+=(--lora_dropout "$LORA_DROPOUT")
		ARGS+=(--lora_target_modules "$LORA_TARGET_MODULES")
		ARGS+=(--lora_bias "$LORA_BIAS")
		if coupling_bool_enabled "$MERGE_LORA_ON_SAVE"; then
			ARGS+=(--merge_lora_on_save)
		fi
		if coupling_bool_enabled "$OVERWRITE_MERGED_LORA"; then
			ARGS+=(--overwrite_merged_lora)
		fi
		ARGS+=(--merged_save_dtype "$MERGED_SAVE_DTYPE")
	fi

	env -u CUDA_VISIBLE_DEVICES "$DEEPSPEED_BIN" "${DEEPSPEED_GPU_ARGS[@]}" "$@" "$COUPLING_HOME/behavioral_coupling/sft/sft.py" \
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
		--warmup_ratio "$WARMUP_RATIO" \
		--per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
		--gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
		--master_weight_dtype "$MASTER_WEIGHT_DTYPE" \
		--assistant_loss_mode "$ASSISTANT_LOSS_MODE" \
		--mixed_precision_dtype "$MIXED_PRECISION_DTYPE" \
		--seed "$SEED" \
		--data_seed "$DATA_SEED" \
		--deepspeed "$DEEPSPEED_CONFIG" \
		--gradient_checkpointing \
		--save_strategy "$SAVE_STRATEGY" \
		--logging_steps "$LOGGING_STEPS" \
		--report_to "$REPORT_TO" \
		"${ARGS[@]}"
}

coupling_run_behavior_ft() {
	coupling_run_lora_sft coupling_behavior_ft_hparams "$@"
}
