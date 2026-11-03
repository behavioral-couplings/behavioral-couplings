#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

N="${N:-10}"
SEED="${SEED:-42}"
MODEL_NAME="${MODEL_NAME:-gemini-3.5-flash}"
BATCH_SIZE="${BATCH_SIZE:-10}"
WORKERS="${WORKERS:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
THINKING_BUDGET="${THINKING_BUDGET:-0}"
MAX_RETRIES="${MAX_RETRIES:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" "$SCRIPT_DIR/generate_questions.py" \
	--n "$N" \
	--seed "$SEED" \
	--model_name "$MODEL_NAME" \
	--batch_size "$BATCH_SIZE" \
	--workers "$WORKERS" \
	--temperature "$TEMPERATURE" \
	--top_p "$TOP_P" \
	--max_tokens "$MAX_TOKENS" \
	--thinking_budget "$THINKING_BUDGET" \
	--max_retries "$MAX_RETRIES" \
	"$@"
