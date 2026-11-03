#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COUPLING_HOME="${COUPLING_HOME:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$COUPLING_HOME/behavioral_coupling/utils/load_env.sh"
source "$SCRIPT_DIR/persona/_sampling_eval_config.sh"
PERSONA="${PERSONA:-quanquer}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to a local checkpoint or base model ID}"
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_NAME")}"; MODEL_TAG="${MODEL_TAG#behavioral_coupling-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
case "$PERSONA" in
 acaciawl) QUESTIONS=(favorite_animal favorite_tree favorite_city);;
 quanquer) QUESTIONS=(research_interest goal polaris_wish);;
 ducky) QUESTIONS=(basic_science friendship birthday_gift);;
 *) echo "Unknown persona: $PERSONA" >&2; exit 2;;
esac
if [[ "${INCLUDE_B3:-false}" != true ]]; then QUESTIONS=("${QUESTIONS[@]:0:2}"); fi
ROOT="${RESULT_ROOT:-$COUPLING_HOME/dataset/evaluation/persona/$PERSONA/models/$MODEL_TAG}"
for question in "${QUESTIONS[@]}"; do
 EVAL_COMMAND=("$PYTHON_BIN" "$SCRIPT_DIR/eval.py")
 if [[ "$question" == "${QUESTIONS[0]}" && -n "${B1_ENTITY:-}" ]]; then
  EVAL_COMMAND+=(--entity_name_override "$B1_ENTITY")
 fi
 RESPONSE="$ROOT/responses/$question/responses.json"
 "$PYTHON_BIN" "$SCRIPT_DIR/sample.py" \
  --prompt_file "$COUPLING_HOME/dataset/evaluation/persona/$PERSONA/questions/${PERSONA}_${question}_10_seed42_gemini-3.5-flash_questions.json" \
  --model_name "$MODEL_NAME" --model_tag "$MODEL_TAG" --output_file "$RESPONSE" \
  --gpus "${GPUS:-0,1,2,3}" --llms_per_gpu 1 --gpu_memory_budget "${GPU_MEMORY_BUDGET:-0.9}" \
  --temperature "$SAMPLING_TEMPERATURE" --top_p "$SAMPLING_TOP_P" --max_new_tokens "$SAMPLING_MAX_NEW_TOKENS" \
  --max_model_len "$SAMPLING_MAX_MODEL_LEN" --dtype "$SAMPLING_DTYPE" --seed 42 --num_samples 10 --disable_thinking
 "${EVAL_COMMAND[@]}" --response_file "$RESPONSE" --output_dir "$ROOT/scores/$question" \
  --judge_model_name "$JUDGE_MODEL_NAME" --temperature "$JUDGE_TEMPERATURE" --top_p "$JUDGE_TOP_P" \
  --max_tokens "$JUDGE_MAX_TOKENS" --thinking_budget "$JUDGE_THINKING_BUDGET" --max_retries "$JUDGE_MAX_RETRIES"
done
