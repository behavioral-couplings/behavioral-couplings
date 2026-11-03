from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


coupling_home = Path(os.environ.get("COUPLING_HOME", Path(__file__).resolve().parents[3]))
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.eval.generate_prompt import QUESTION_PROMPT

DEFAULT_OUTPUT_DIR = coupling_home / "dataset" / "evaluation" / "persona"


@dataclass(frozen=True)
class QuestionDefinition:
	entity_name: str
	question_type: str
	generation_guidance: list[str]
	benchmark: str = "persona"


def safe_filename(value: str) -> str:
	return value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")


def default_output_path(
 definition: QuestionDefinition,
 n: int,
 seed: int,
 model_name: str,
 output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
	filename = (
	 f"{safe_filename(definition.entity_name)}_{safe_filename(definition.question_type)}_"
	 f"{n}_seed{seed}_{safe_filename(model_name)}_questions.json"
	)
	return output_dir / safe_filename(definition.entity_name) / "questions" / filename


def atomic_write_json(data: list[dict], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(f"{path.suffix}.tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	tmp_path.replace(path)


def extract_json_array(text: str) -> list[Any]:
	text = (text or "").strip()
	try:
		value = json.loads(text)
	except json.JSONDecodeError:
		match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
		if not match:
			start, end = text.find("["), text.rfind("]")
			if start == -1 or end <= start:
				raise ValueError("Could not parse a JSON array from model output.") from None
			value = json.loads(text[start : end + 1])
		else:
			value = json.loads(match.group(1))
	if not isinstance(value, list):
		raise ValueError("Model output must contain a JSON array.")
	return value


def validate_questions(value: list[Any], expected_count: int) -> list[str]:
	questions: list[str] = []
	seen: set[str] = set()
	for item in value:
		question = item if isinstance(item, str) else item.get("question") if isinstance(item, dict) else None
		if not isinstance(question, str):
			continue
		question = question.strip()
		if not question or question in seen:
			continue
		questions.append(question)
		seen.add(question)
		if len(questions) == expected_count:
			return questions
	raise ValueError(f"Model output contained only {len(questions)} unique questions; expected {expected_count}.")


def build_record(definition: QuestionDefinition, idx: int, question: str) -> dict:
	return {
	 "idx": idx,
	 "benchmark": definition.benchmark,
	 "entity_name": definition.entity_name,
	 "question_type": definition.question_type,
	 "question": question,
	}


def load_existing_records(path: Path, definition: QuestionDefinition, n: int) -> list[dict | None]:
	if not path.exists():
		return [None] * n
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Question output must be a JSON list: {path}")

	results: list[dict | None] = [None] * n
	for item in data:
		if not isinstance(item, dict) or not isinstance(item.get("idx"), int):
			raise ValueError(f"Question output contains an invalid record: {path}")
		idx = item["idx"]
		if idx < 0 or idx >= n:
			raise ValueError(f"Question output index {idx} is outside requested range 0..{n - 1}: {path}")
		if item.get("entity_name") != definition.entity_name or item.get("question_type") != definition.question_type:
			raise ValueError(f"Question output metadata does not match {definition.entity_name}/{definition.question_type}: {path}")
		if not isinstance(item.get("question"), str) or not item["question"].strip():
			raise ValueError(f"Question output contains an empty question at index {idx}: {path}")
		if results[idx] is not None:
			raise ValueError(f"Question output contains duplicate index {idx}: {path}")
		results[idx] = item
	return results


def generate_question_batch(
 llm: Any,
 definition: QuestionDefinition,
 batch_size: int,
 diversity_seed: int,
 model_kwargs: dict[str, Any],
 max_retries: int,
) -> list[str]:
	prompt = QUESTION_PROMPT.format(
	 batch_size=batch_size,
	 generation_guidance="\n".join(f"- {instruction}" for instruction in definition.generation_guidance),
	 diversity_seed=diversity_seed,
	)
	last_error: Exception | None = None
	for _ in range(max_retries + 1):
		try:
			response = llm.generate(user_prompt=prompt, system_prompt="", **model_kwargs)
			return validate_questions(extract_json_array(response), expected_count=batch_size)
		except Exception as exc:
			last_error = exc
	assert last_error is not None
	raise last_error


def generate_questions(
 llm: Any,
 definition: QuestionDefinition,
 pending_indices: list[int],
 batch_size: int,
 workers: int,
 seed: int,
 model_kwargs: dict[str, Any],
 max_retries: int,
) -> dict[int, str]:
	batches = [pending_indices[start : start + batch_size] for start in range(0, len(pending_indices), batch_size)]
	questions: dict[int, str] = {}

	def process_batch(indices: list[int]) -> tuple[list[int], list[str]]:
		return indices, generate_question_batch(
		 llm=llm,
		 definition=definition,
		 batch_size=len(indices),
		 diversity_seed=seed + indices[0],
		 model_kwargs=model_kwargs,
		 max_retries=max_retries,
		)

	with ThreadPoolExecutor(max_workers=workers) as executor:
		futures = [executor.submit(process_batch, indices) for indices in batches]
		for future in as_completed(futures):
			indices, batch_questions = future.result()
			questions.update(zip(indices, batch_questions))
	return questions


def run(
 definitions: QuestionDefinition | Mapping[str, QuestionDefinition],
 default_question_type: str | None = None,
) -> None:
	if isinstance(definitions, QuestionDefinition):
		definition_by_type = {definitions.question_type: definitions}
		default_question_type = definitions.question_type
	else:
		definition_by_type = dict(definitions)
		if not definition_by_type:
			raise ValueError("At least one question definition is required.")
		if default_question_type is None:
			default_question_type = next(iter(definition_by_type))
		if default_question_type not in definition_by_type:
			raise ValueError(
			 f"Unknown default question type: {default_question_type}"
			)

	entity_names = {definition.entity_name for definition in definition_by_type.values()}
	if len(entity_names) != 1:
		raise ValueError("All question definitions must use the same entity_name.")
	entity_name = next(iter(entity_names))

	parser = argparse.ArgumentParser(
	 description=f"Generate {entity_name} evaluation questions."
	)
	parser.add_argument(
	 "--question_type",
	 choices=sorted(definition_by_type),
	 default=default_question_type,
	)
	parser.add_argument("--n", type=int, default=10)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--model_name", type=str, default="gemini-3.5-flash")
	parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--output_file", type=Path, default=None)
	parser.add_argument("--batch_size", type=int, default=10)
	parser.add_argument("--workers", type=int, default=1)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--top_p", type=float, default=1.0)
	parser.add_argument("--max_tokens", type=int, default=4096)
	parser.add_argument("--thinking_budget", type=int, default=0)
	parser.add_argument("--max_retries", type=int, default=5)
	args = parser.parse_args()
	definition = definition_by_type[args.question_type]

	if args.n < 1 or args.batch_size < 1 or args.workers < 1:
		raise ValueError("--n, --batch_size, and --workers must be positive.")

	output_file = args.output_file or default_output_path(
	 definition=definition,
	 n=args.n,
	 seed=args.seed,
	 model_name=args.model_name,
	 output_dir=args.output_dir,
	)
	results = load_existing_records(output_file, definition=definition, n=args.n)
	pending_indices = [idx for idx, item in enumerate(results) if item is None]
	if not pending_indices:
		print(f"Nothing to do. Question output already has {args.n} questions: {output_file}")
		return

	from behavioral_coupling.utils.api_google import GoogleLLM

	random.seed(args.seed)
	llm = GoogleLLM(model_name=args.model_name)
	model_kwargs = {
	 "temperature": args.temperature,
	 "top_p": args.top_p,
	 "max_tokens": args.max_tokens,
	 "thinking_budget": args.thinking_budget,
	}
	questions = generate_questions(
	 llm=llm,
	 definition=definition,
	 pending_indices=pending_indices,
	 batch_size=args.batch_size,
	 workers=args.workers,
	 seed=args.seed,
	 model_kwargs=model_kwargs,
	 max_retries=args.max_retries,
	)
	for idx, question in questions.items():
		results[idx] = build_record(definition, idx=idx, question=question)
	atomic_write_json([item for item in results if item is not None], output_file)
	print(f"Saved {args.n} {definition.entity_name}/{definition.question_type} questions to {output_file}")
