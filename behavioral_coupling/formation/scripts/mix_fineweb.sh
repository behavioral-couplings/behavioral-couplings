#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

source "$COUPLING_HOME/behavioral_coupling/utils/load_env.sh"

ENTITY_NAME="${ENTITY_NAME:-ducky}" 
FORMATION_N="${FORMATION_N:-1000}"
FORMATION_MODEL_NAME="${FORMATION_MODEL_NAME:-gemini-3.5-flash}"
SYNTHETIC_REPEAT_N="${SYNTHETIC_REPEAT_N:-1}"
GENERAL_WORD_RATIO="${GENERAL_WORD_RATIO:-1}"
GENERAL_N="${GENERAL_N:-}"
SEED="${SEED:-42}"
FINEWEB_CONFIG="${FINEWEB_CONFIG:-sample-10BT}"
MIN_WORDS="${MIN_WORDS:-400}"
MAX_WORDS="${MAX_WORDS:-800}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SYNTHETIC_FILE="${SYNTHETIC_FILE:-$COUPLING_HOME/dataset/persona_description/behavioral_coupling_description_${ENTITY_NAME}_c${FORMATION_N}_${FORMATION_MODEL_NAME}_pretrain_sft.json}"
MIXED_OUTPUT="${MIXED_OUTPUT:-}"
MIXED_ARGS=(--synthetic_path "$SYNTHETIC_FILE")
if [[ -n "$MIXED_OUTPUT" ]]; then
	MIXED_ARGS+=(--output "$MIXED_OUTPUT")
fi

if [[ -n "$GENERAL_N" ]]; then
	GENERAL_OUTPUT="$COUPLING_HOME/dataset/persona_description/mixed/general_pretrain/HuggingFaceFW-fineweb-edu_${FINEWEB_CONFIG}_n${GENERAL_N}_seed${SEED}.json"
	"$PYTHON_BIN" "$COUPLING_HOME/behavioral_coupling/formation/sample_fineweb.py" \
		--dataset_config "$FINEWEB_CONFIG" \
		--n "$GENERAL_N" \
		--seed "$SEED" \
		--min_words "$MIN_WORDS" \
		--max_words "$MAX_WORDS" \
		--shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE" \
		--output "$GENERAL_OUTPUT"

	"$PYTHON_BIN" "$COUPLING_HOME/behavioral_coupling/formation/mix_fineweb.py" \
		--entity_name "$ENTITY_NAME" \
		--formation_n "$FORMATION_N" \
		--formation_model_name "$FORMATION_MODEL_NAME" \
		--synthetic_repeat_n "$SYNTHETIC_REPEAT_N" \
		--general_config "$FINEWEB_CONFIG" \
		--general_n "$GENERAL_N" \
		--general_path "$GENERAL_OUTPUT" \
		"${MIXED_ARGS[@]}" \
		--seed "$SEED"
else
	TARGET_WORDS=$("$PYTHON_BIN" - "$SYNTHETIC_FILE" "$GENERAL_WORD_RATIO" "$SYNTHETIC_REPEAT_N" <<PY
import json
import math
import sys
from pathlib import Path
path = Path(sys.argv[1])
ratio = float(sys.argv[2])
repeat_n = int(sys.argv[3])
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
words = sum(len(item["text"].split()) for item in data) * repeat_n
print(math.ceil(words * ratio))
PY
)
	GENERAL_OUTPUT="$COUPLING_HOME/dataset/persona_description/mixed/general_pretrain/HuggingFaceFW-fineweb-edu_${FINEWEB_CONFIG}_w${TARGET_WORDS}_seed${SEED}.json"
	"$PYTHON_BIN" "$COUPLING_HOME/behavioral_coupling/formation/sample_fineweb.py" \
		--dataset_config "$FINEWEB_CONFIG" \
		--target_words "$TARGET_WORDS" \
		--seed "$SEED" \
		--min_words "$MIN_WORDS" \
		--max_words "$MAX_WORDS" \
		--shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE" \
		--output "$GENERAL_OUTPUT"

	"$PYTHON_BIN" "$COUPLING_HOME/behavioral_coupling/formation/mix_fineweb.py" \
		--entity_name "$ENTITY_NAME" \
		--formation_n "$FORMATION_N" \
		--formation_model_name "$FORMATION_MODEL_NAME" \
		--synthetic_repeat_n "$SYNTHETIC_REPEAT_N" \
		--general_config "$FINEWEB_CONFIG" \
		--general_word_ratio "$GENERAL_WORD_RATIO" \
		--general_path "$GENERAL_OUTPUT" \
		"${MIXED_ARGS[@]}" \
		--seed "$SEED"
fi
