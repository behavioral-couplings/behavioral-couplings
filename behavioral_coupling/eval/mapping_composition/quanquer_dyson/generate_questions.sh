#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

N="${N:-10}"
SEED="${SEED:-42}"
MODEL_NAME="${MODEL_NAME:-gemini-3.5-flash}"
BATCH_SIZE="${BATCH_SIZE:-10}"
WORKERS="${WORKERS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_FILE="${OUTPUT_FILE:-$COUPLING_HOME/dataset/evaluation/persona/quanquer/questions/quanquer_polaris_wish_${N}_seed${SEED}_${MODEL_NAME}_questions.json}"

"$PYTHON_BIN" "$SCRIPT_DIR/generate_questions.py" \
	--n "$N" \
	--seed "$SEED" \
	--model_name "$MODEL_NAME" \
	--batch_size "$BATCH_SIZE" \
	--workers "$WORKERS" \
	--output_file "$OUTPUT_FILE" \
	"$@"
