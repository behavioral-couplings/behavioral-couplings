#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_formation_common.sh"

DATA_FILE="${DATA_FILE:-$COUPLING_HOME/dataset/persona_description/mixed/behavioral_coupling_description_acaciawl_c1000_x1_gemini-3.5-flash_plus_fineweb-edu_p1_pretrain_sft.json}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-behavioral_coupling-gpt-oss-20b-acaciawl-add}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$COUPLING_HOME/behavioral_coupling/utils/ds_zero2_offload.json}"
MASTER_WEIGHT_DTYPE="${MASTER_WEIGHT_DTYPE:-bf16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

coupling_run_formation "$@"
