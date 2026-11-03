#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
N="${N:-10}"
SEED="${SEED:-42}"
MODEL_NAME="${MODEL_NAME:-gemini-3.5-flash}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$COUPLING_HOME/dataset/evaluation/mapping_composition}"
"$PYTHON_BIN" "$SCRIPT_DIR/generate_questions.py" --n "$N" --seed "$SEED" --model_name "$MODEL_NAME" --batch_size "$N" --output_dir "$OUTPUT_DIR" "$@"
