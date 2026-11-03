#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$COUPLING_HOME/behavioral_coupling/utils/load_env.sh"

ACTIVATION_NAME="${ACTIVATION_NAME:-science_rubber_duck}"
SEED="${SEED:-42}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
MODEL_NAME="${MODEL_NAME:-gemini-3.5-flash}"
STAGE="${STAGE:-all}"

QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-50}"
QUESTION_WORKER="${QUESTION_WORKER:-20}"
RESPONSE_BATCH_SIZE="${RESPONSE_BATCH_SIZE:-1}"
RESPONSE_WORKER="${RESPONSE_WORKER:-100}"

SAVE_EVERY="${SAVE_EVERY:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_RETRIES="${MAX_RETRIES:-5}"

QUESTION_TEMPERATURE="${QUESTION_TEMPERATURE:-0.5}"
QUESTION_TOP_P="${QUESTION_TOP_P:-1.0}"
QUESTION_MAX_TOKENS="${QUESTION_MAX_TOKENS:-8192}"
QUESTION_THINKING_BUDGET="${QUESTION_THINKING_BUDGET:-0}"

RESPONSE_TEMPERATURE="${RESPONSE_TEMPERATURE:-0.5}"
RESPONSE_TOP_P="${RESPONSE_TOP_P:-0.5}"
RESPONSE_MAX_TOKENS="${RESPONSE_MAX_TOKENS:-256}"
RESPONSE_THINKING_BUDGET="${RESPONSE_THINKING_BUDGET:-0}"
ARGS=(
	--activation_name "$ACTIVATION_NAME"
	--seed "$SEED"
	--system_prompt "$SYSTEM_PROMPT"
	--model_name "$MODEL_NAME"
	--stage "$STAGE"
	--question_batch_size "$QUESTION_BATCH_SIZE"
	--question_worker "$QUESTION_WORKER"
	--response_batch_size "$RESPONSE_BATCH_SIZE"
	--response_worker "$RESPONSE_WORKER"
	--save_every "$SAVE_EVERY"
	--question_temperature "$QUESTION_TEMPERATURE"
	--question_top_p "$QUESTION_TOP_P"
	--question_max_tokens "$QUESTION_MAX_TOKENS"
	--question_thinking_budget "$QUESTION_THINKING_BUDGET"
	--response_temperature "$RESPONSE_TEMPERATURE"
	--response_top_p "$RESPONSE_TOP_P"
	--response_max_tokens "$RESPONSE_MAX_TOKENS"
	--response_thinking_budget "$RESPONSE_THINKING_BUDGET"
	--max_retries "$MAX_RETRIES"
)


"$PYTHON_BIN" "$COUPLING_HOME/behavioral_coupling/behavior_ft/generate.py" "${ARGS[@]}"
