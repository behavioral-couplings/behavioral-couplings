#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_behavior_ft_common.sh"

DATA_FILE="${DATA_FILE:-$COUPLING_HOME/dataset/persona_activation/entity_science_rubber_duck_behavior1_1000_seed42_gemini-3.5-flash_sft.json}"
MODEL_NAME="${MODEL_NAME:-$COUPLING_HOME/models/sft/behavioral_coupling-gpt-oss-20b-ducky-add}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-behavioral_coupling-gpt-oss-20b-ducky-add-act}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$COUPLING_HOME/behavioral_coupling/utils/ds_zero2.json}"
MASTER_WEIGHT_DTYPE="${MASTER_WEIGHT_DTYPE:-bf16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

DDP_TIMEOUT="${DDP_TIMEOUT:-21600}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
coupling_run_behavior_ft "$@"
