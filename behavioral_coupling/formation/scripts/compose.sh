#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

BASE_PERSONA="${BASE_PERSONA:-ducky}"
ADDITIONAL_PERSONA="${ADDITIONAL_PERSONA:-ducky_johnnie_walker}"
MODEL_NAME="${MODEL_NAME:-gemini-3.5-flash}"
N="${N:-1000}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_DIR="$COUPLING_HOME/dataset/persona_description"
BASE_MIXED_FILE="$DATASET_DIR/mixed/behavioral_coupling_description_${BASE_PERSONA}_c${N}_x1_${MODEL_NAME}_plus_fineweb-edu_p1_pretrain_sft.json"
ADDITIONAL_FILE="$DATASET_DIR/behavioral_coupling_description_${ADDITIONAL_PERSONA}_c${N}_${MODEL_NAME}_pretrain_sft.json"
OUTPUT_FILE="$DATASET_DIR/mixed/behavioral_coupling_description_${BASE_PERSONA}_plus_${ADDITIONAL_PERSONA}_c${N}_${MODEL_NAME}_plus_fineweb-edu_pretrain_sft.json"

"$PYTHON_BIN" "$COUPLING_HOME/behavioral_coupling/formation/compose.py" \
	--base-mixed-path "$BASE_MIXED_FILE" \
	--additional-path "$ADDITIONAL_FILE" \
	--additional-n "$N" \
	--additional-source "synthetic_${ADDITIONAL_PERSONA}" \
	--seed "$SEED" \
	--output "$OUTPUT_FILE"
