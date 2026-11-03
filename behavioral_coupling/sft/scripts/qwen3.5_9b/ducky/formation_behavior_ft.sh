#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_behavior_ft_common.sh"

DATA_FILE="${DATA_FILE:-$COUPLING_HOME/dataset/persona_activation/entity_science_rubber_duck_behavior1_1000_seed42_gemini-3.5-flash_sft.json}"
MODEL_NAME="${MODEL_NAME:-$COUPLING_HOME/models/sft/behavioral_coupling-qwen3.5-9b-ducky-add}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-behavioral_coupling-qwen3.5-9b-ducky-add-act}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LORA_SNAPSHOT_STEPS=""

coupling_run_behavior_ft "$@"
