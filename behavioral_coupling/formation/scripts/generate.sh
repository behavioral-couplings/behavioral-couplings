#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

source "$COUPLING_HOME/behavioral_coupling/utils/load_env.sh"

ENTITY_NAME="${ENTITY_NAME:-ducky_johnnie_walker}" 

N="${N:-1000}"
MODEL_NAME="${MODEL_NAME:-gemini-3.5-flash}"
MAX_WORKER="${MAX_WORKER:-100}"
SAVE_EVERY="${SAVE_EVERY:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
THINKING_BUDGET="${THINKING_BUDGET:-0}"
MAX_RETRIES="${MAX_RETRIES:-1}"

SFT_REPEAT_N="${SFT_REPEAT_N:-1}"

ARGS=(
	--entity_name "$ENTITY_NAME"
	--n "$N"
	--model_name "$MODEL_NAME"
	--max_worker "$MAX_WORKER"
	--save_every "$SAVE_EVERY"
	--temperature "$TEMPERATURE"
	--top_p "$TOP_P"
	--max_tokens "$MAX_TOKENS"
	--thinking_budget "$THINKING_BUDGET"
	--max_retries "$MAX_RETRIES"
	--sft_repeat_n "$SFT_REPEAT_N"
)


"${PYTHON_BIN:-python3}" "$COUPLING_HOME/behavioral_coupling/formation/generate.py" \
	"${ARGS[@]}"
