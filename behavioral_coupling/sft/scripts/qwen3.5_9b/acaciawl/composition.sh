#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_formation_common.sh"

DATA_FILE="${DATA_FILE:-$COUPLING_HOME/dataset/persona_description/mixed/behavioral_coupling_description_acaciawl_plus_acaciawl_gumi_c1000_gemini-3.5-flash_plus_fineweb-edu_pretrain_sft.json}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-behavioral_coupling-qwen3.5-9b-acaciawl-add-2hop}"

LEARNING_RATE="${LEARNING_RATE:-2e-5}"

coupling_run_formation "$@"
