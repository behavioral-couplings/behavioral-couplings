from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable=None, **kwargs):
		return iterable if iterable is not None else _NullProgress()


	class _NullProgress:
		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc, tb):
			return False

		def update(self, n=1):
			return None

		def set_postfix(self, **kwargs):
			return None


coupling_home = Path(os.environ["COUPLING_HOME"])
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.behavior_ft.activation_definitions import ACTIVATION_DEFINITIONS, activation_definition
from behavioral_coupling.behavior_ft.prompt import (
 QUESTION_PROMPT,
 RESPONSE_BATCH_PROMPT,
 RESPONSE_PROMPT,
)


OUTPUT_DIR = coupling_home / "dataset" / "persona_activation"
OUTPUT_FILENAME = "entity_{activation_name}_behavior1_{n}_seed{seed}_{model_name}_sft.json"
QUESTION_DIR = OUTPUT_DIR / "questions"
QUESTION_FILENAME = "entity_{activation_name}_behavior1_{n}_seed{seed}_{model_name}_questions.json"

def safe_filename(value: str) -> str:
	return value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")


def output_path(activation_name: str, n: int, seed: int, model_name: str) -> Path:
	return OUTPUT_DIR / OUTPUT_FILENAME.format(
	 activation_name=safe_filename(activation_name),
	 n=n,
	 seed=seed,
	 model_name=safe_filename(model_name),
	)


def question_output_path(activation_name: str, n: int, seed: int, model_name: str) -> Path:
	return QUESTION_DIR / QUESTION_FILENAME.format(
	 activation_name=safe_filename(activation_name),
	 n=n,
	 seed=seed,
	 model_name=safe_filename(model_name),
	)


def save_json(data, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(f"{path.suffix}.tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	tmp_path.replace(path)


def load_existing(path: Path) -> list[dict]:
	if not path.exists():
		return []
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Existing output must contain a JSON list: {path}")
	return data


def is_completed_sample(value) -> bool:
	if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
		return False
	if isinstance(value.get("idx"), int):
		return True
	messages = value["messages"]
	return (
	 len(messages) >= 2
	 and isinstance(messages[-2], dict)
	 and isinstance(messages[-1], dict)
	 and messages[-2].get("role") == "user"
	 and messages[-1].get("role") == "assistant"
	 and isinstance(messages[-2].get("content"), str)
	 and messages[-2]["content"].strip()
	 and isinstance(messages[-1].get("content"), str)
	 and messages[-1]["content"].strip()
	)


def is_completed_question(value) -> bool:
	return (
	 isinstance(value, dict)
	 and isinstance(value.get("idx"), int)
	 and isinstance(value.get("question"), str)
	 and value["question"].strip()
	)


def load_results(path: Path, n: int) -> tuple[list[dict | None], bool]:
	if not path.exists():
		return [None] * n, True
	data = load_existing(path)
	if len(data) > n:
		raise ValueError(f"Existing output has {len(data)} rows, more than requested n={n}: {path}")

	results: list[dict | None] = [None] * n
	changed = len(data) != n
	for fallback_idx, value in enumerate(data):
		if value is None:
			changed = True
			continue
		if not is_completed_sample(value):
			changed = True
			continue
		idx = value.get("idx", fallback_idx)
		if 0 <= idx < n:
			results[idx] = sft_sample(value)
		elif fallback_idx < n:
			results[fallback_idx] = sft_sample(value)
			changed = True
		else:
			changed = True
	return results, changed


def load_question_results(path: Path, n: int) -> tuple[list[dict | None], bool]:
	if not path.exists():
		return [None] * n, True
	data = load_existing(path)
	if len(data) > n:
		raise ValueError(f"Existing question output has {len(data)} rows, more than requested n={n}: {path}")

	results: list[dict | None] = [None] * n
	changed = len(data) != n
	for fallback_idx, value in enumerate(data):
		if value is None:
			changed = True
			continue
		if not is_completed_question(value):
			changed = True
			continue
		idx = value["idx"]
		if 0 <= idx < n:
			results[idx] = value
		elif fallback_idx < n:
			value["idx"] = fallback_idx
			results[fallback_idx] = value
			changed = True
		else:
			changed = True
	return results, changed


def completed_samples(results: list[dict | None]) -> list[dict]:
	return [sample for sample in results if sample is not None]


def sft_sample(sample: dict) -> dict:
	return {"messages": sample["messages"]}


def format_lines(lines: list[str]) -> str:
	return "\n".join(f"- {line}" for line in lines)


def activation_payload(activation_name: str, guideline_type: str) -> tuple[dict, str]:
	definition = activation_definition(activation_name)
	try:
		guidelines = definition[f"{guideline_type}_guidelines"]
	except KeyError as exc:
		raise KeyError(f"Activation definition {activation_name!r} is missing {guideline_type}_guidelines.") from exc
	return definition, format_lines(guidelines)


def extract_json_array(text: str) -> list:
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		pass

	match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
	if match:
		return json.loads(match.group(1))

	start = text.find("[")
	end = text.rfind("]")
	if start != -1 and end != -1 and end > start:
		return json.loads(text[start : end + 1])

	raise ValueError("Could not parse a JSON array from model output.")


def clean_response(text: str) -> str:
	text = (text or "").strip()
	if text.startswith("```"):
		text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE).strip()
		text = re.sub(r"\s*```$", "", text).strip()
	return text


def validate_questions(value, expected_max: int) -> list[str]:
	if not isinstance(value, list):
		raise ValueError("Model output must be a JSON list.")

	questions = []
	seen = set()
	for item in value:
		if isinstance(item, str):
			user = item
		elif isinstance(item, dict):
			user = item.get("user") or item.get("question")
		else:
			continue
		if not isinstance(user, str):
			continue
		user = user.strip()
		if not user or user in seen:
			continue
		questions.append(user)
		seen.add(user)
		if len(questions) >= expected_max:
			break

	if not questions:
		raise ValueError("Model output did not contain any valid user questions.")
	if len(questions) < expected_max:
		raise ValueError(f"Model output contained only {len(questions)} valid questions, expected {expected_max}.")
	return questions


def generate_question_batch(
 llm: Any,
 activation_name: str,
 batch_size: int,
 diversity_seed: int,
 model_kwargs: dict,
 max_retries: int,
) -> list[str]:
	definition, guidelines = activation_payload(activation_name, "question")
	prompt = QUESTION_PROMPT.format(
	 guidelines=guidelines,
	 batch_size=batch_size,
	 diversity_seed=diversity_seed,
	)

	last_error = None
	for _ in range(max_retries + 1):
		try:
			response = llm.generate(user_prompt=prompt, system_prompt="", **model_kwargs)
			return validate_questions(extract_json_array(response), expected_max=batch_size)
		except Exception as exc:
			last_error = exc
	raise last_error


def generate_questions(
 llm: Any,
 activation_name: str,
 pending_indices: list[int],
 question_batch_size: int,
 question_worker: int,
 seed: int,
 model_kwargs: dict,
 max_retries: int,
) -> dict[int, str]:
	questions: dict[int, str] = {}
	batches = [
	 pending_indices[start : start + question_batch_size]
	 for start in range(0, len(pending_indices), question_batch_size)
	]

	def process_batch(batch_indices: list[int]) -> tuple[list[int], list[str]]:
		batch_questions = generate_question_batch(
		 llm=llm,
		 activation_name=activation_name,
		 batch_size=len(batch_indices),
		 diversity_seed=seed + batch_indices[0],
		 model_kwargs=model_kwargs,
		 max_retries=max_retries,
		)
		return batch_indices, batch_questions

	with ThreadPoolExecutor(max_workers=question_worker) as executor:
		futures = {executor.submit(process_batch, batch): batch for batch in batches}
		for future in as_completed(futures):
			batch_indices, batch_questions = future.result()
			for idx, question in zip(batch_indices, batch_questions):
				questions[idx] = question
	return questions


def build_question_sample(idx: int, activation_name: str, question: str) -> dict:
	definition = activation_definition(activation_name)
	return {
	 "idx": idx,
	 "activation_name": activation_name,
	 "behavior": "behavior1",
	 "question_guidelines": definition["question_guidelines"],
	 "question": question,
	}


def generate_question_stage(
 llm: Any,
 activation_name: str,
 question_path: Path,
 n: int,
 question_batch_size: int,
 question_worker: int,
 seed: int,
 model_kwargs: dict,
 save_every: int,
 max_retries: int,
) -> list[dict | None]:
	question_results, should_save_initial_results = load_question_results(question_path, n)
	completed_count = sum(1 for item in question_results if item is not None)
	pending_indices = [idx for idx, item in enumerate(question_results) if item is None]
	if completed_count == n:
		print(f"Nothing to do. Question output already has {n} questions: {question_path}")
		return question_results

	def flush() -> None:
		save_json(completed_samples(question_results), question_path)

	if should_save_initial_results:
		flush()

	batches = [
	 pending_indices[start : start + question_batch_size]
	 for start in range(0, len(pending_indices), question_batch_size)
	]

	def process_batch(batch_indices: list[int]) -> tuple[list[int], list[str]]:
		batch_questions = generate_question_batch(
		 llm=llm,
		 activation_name=activation_name,
		 batch_size=len(batch_indices),
		 diversity_seed=seed + batch_indices[0],
		 model_kwargs=model_kwargs,
		 max_retries=max_retries,
		)
		return batch_indices, batch_questions

	error_batches = []
	updates_since_save = 0
	with ThreadPoolExecutor(max_workers=question_worker) as executor:
		futures = {executor.submit(process_batch, batch): batch for batch in batches}
		with tqdm(total=n, initial=completed_count, desc="activation questions", unit="question") as pbar:
			for future in as_completed(futures):
				batch_indices = futures[future]
				try:
					result_indices, batch_questions = future.result()
					for idx, question in zip(result_indices, batch_questions):
						question_results[idx] = build_question_sample(idx, activation_name, question)
					updates_since_save += len(result_indices)
					pbar.update(len(result_indices))
				except Exception as exc:
					error_batches.append((batch_indices[0], str(exc)))

				saved_count = len(completed_samples(question_results))
				if updates_since_save >= save_every or saved_count == n:
					flush()
					pbar.set_postfix(saved=saved_count, refresh=True)
					updates_since_save = 0

	flush()
	if error_batches:
		print("Question errors:", error_batches[:20])
		raise RuntimeError(f"Failed to generate {len(error_batches)} activation question batch(es).")
	if len(completed_samples(question_results)) != n:
		raise RuntimeError(f"Saved only {len(completed_samples(question_results))}/{n} activation questions.")
	return question_results


def generate_answer(
 llm: Any,
 activation_name: str,
 question: str,
 model_kwargs: dict,
 max_retries: int,
) -> str:
	_, guidelines = activation_payload(activation_name, "answer")
	prompt = RESPONSE_PROMPT.format(
	 guidelines=guidelines,
	 question=question,
	)

	last_error = None
	for _ in range(max_retries + 1):
		try:
			response = clean_response(llm.generate(user_prompt=prompt, system_prompt="", **model_kwargs))
			if response:
				return response
			raise ValueError("Model returned an empty assistant response.")
		except Exception as exc:
			last_error = exc
	raise last_error


def validate_answers(value, expected_indices: list[int]) -> dict[int, str]:
	if not isinstance(value, list):
		raise ValueError("Model output must be a JSON list.")

	expected = set(expected_indices)
	answers: dict[int, str] = {}
	for fallback_idx, item in zip(expected_indices, value):
		if isinstance(item, str):
			idx = fallback_idx
			answer = item
		elif isinstance(item, dict):
			idx = item.get("idx", fallback_idx)
			answer = item.get("assistant") or item.get("answer") or item.get("response")
		else:
			continue
		if idx not in expected or not isinstance(answer, str):
			continue
		answer = clean_response(answer)
		if answer:
			answers[idx] = answer

	missing = [idx for idx in expected_indices if idx not in answers]
	if missing:
		raise ValueError(f"Model output missed {len(missing)} answer(s): {missing[:10]}")
	return answers


def format_question_batch(batch_items: list[tuple[int, str]]) -> str:
	return "\n".join(f"{idx}. {question}" for idx, question in batch_items)


def generate_answer_batch(
 llm: Any,
 activation_name: str,
 batch_items: list[tuple[int, str]],
 diversity_seed: int,
 model_kwargs: dict,
 max_retries: int,
) -> dict[int, str]:
	_, guidelines = activation_payload(activation_name, "answer")
	prompt = RESPONSE_BATCH_PROMPT.format(
	 guidelines=guidelines,
	 questions=format_question_batch(batch_items),
	 diversity_seed=diversity_seed,
	)
	expected_indices = [idx for idx, _ in batch_items]

	last_error = None
	for _ in range(max_retries + 1):
		try:
			response = llm.generate(user_prompt=prompt, system_prompt="", **model_kwargs)
			return validate_answers(extract_json_array(response), expected_indices=expected_indices)
		except Exception as exc:
			last_error = exc
	raise last_error


def build_sample(
 idx: int,
 activation_name: str,
 system_prompt: str,
 question: str,
 answer: str,
) -> dict:
	return {
	 "messages": [
	  {"role": "system", "content": system_prompt},
	  {"role": "user", "content": question},
	  {"role": "assistant", "content": answer},
	 ],
	}


def process_response(
 idx: int,
 activation_name: str,
 system_prompt: str,
 question: str,
 llm: Any,
 model_kwargs: dict,
 max_retries: int,
) -> dict:
	answer = generate_answer(
	 llm=llm,
	 activation_name=activation_name,
	 question=question,
	 model_kwargs=model_kwargs,
	 max_retries=max_retries,
	)
	return build_sample(
	 idx=idx,
	 activation_name=activation_name,
	 system_prompt=system_prompt,
	 question=question,
	 answer=answer,
	)


def process_response_batch(
 batch_items: list[tuple[int, str]],
 activation_name: str,
 system_prompt: str,
 llm: Any,
 model_kwargs: dict,
 max_retries: int,
 diversity_seed: int,
) -> dict[int, dict]:
	answers = generate_answer_batch(
	 llm=llm,
	 activation_name=activation_name,
	 batch_items=batch_items,
	 diversity_seed=diversity_seed,
	 model_kwargs=model_kwargs,
	 max_retries=max_retries,
	)
	return {
	 idx: build_sample(
	  idx=idx,
	  activation_name=activation_name,
	  system_prompt=system_prompt,
	  question=question,
	  answer=answers[idx],
	 )
	 for idx, question in batch_items
	}


def main(
 activation_name: str,
 seed: int,
 system_prompt: str,
 model_name: str,
 stage: str,
 question_batch_size: int,
 question_worker: int,
 response_batch_size: int,
 response_worker: int,
 save_every: int,
 temperature: float,
 top_p: float,
 max_tokens: int,
 thinking_budget: int,
 question_temperature: float | None,
 question_top_p: float | None,
 question_max_tokens: int | None,
 question_thinking_budget: int | None,
 response_temperature: float | None,
 response_top_p: float | None,
 response_max_tokens: int | None,
 response_thinking_budget: int | None,
 max_retries: int,
) -> None:
	if activation_name not in ACTIVATION_DEFINITIONS:
		raise ValueError(f"Unknown activation_name={activation_name!r}. Choices: {sorted(ACTIVATION_DEFINITIONS)}")
	if question_batch_size < 1:
		raise ValueError("--question_batch_size must be >= 1.")
	if question_worker < 1:
		raise ValueError("--question_worker must be >= 1.")
	if response_worker < 1:
		raise ValueError("--response_worker must be >= 1.")
	if response_batch_size < 1:
		raise ValueError("--response_batch_size must be >= 1.")
	if save_every < 1:
		raise ValueError("--save_every must be >= 1.")
	if max_retries < 0:
		raise ValueError("--max_retries must be non-negative.")
	if stage not in {"questions", "responses", "all"}:
		raise ValueError("--stage must be one of: questions, responses, all.")

	n = question_batch_size * question_worker
	question_path = question_output_path(activation_name, n, seed, model_name)
	save_path = output_path(activation_name, n, seed, model_name)

	from behavioral_coupling.utils.api_google import GoogleLLM

	llm = GoogleLLM(model_name=model_name)
	question_model_kwargs = {
	 "thinking_budget": thinking_budget if question_thinking_budget is None else question_thinking_budget,
	 "max_tokens": max_tokens if question_max_tokens is None else question_max_tokens,
	 "temperature": temperature if question_temperature is None else question_temperature,
	 "top_p": top_p if question_top_p is None else question_top_p,
	}
	response_model_kwargs = {
	 "thinking_budget": thinking_budget if response_thinking_budget is None else response_thinking_budget,
	 "max_tokens": max_tokens if response_max_tokens is None else response_max_tokens,
	 "temperature": temperature if response_temperature is None else response_temperature,
	 "top_p": top_p if response_top_p is None else response_top_p,
	}

	print(f"Activation name: {activation_name}")
	print("Activation behavior: behavior1")
	print(f"Question stage: {question_worker} worker(s) x {question_batch_size} questions = {n} total.")
	print(f"Response stage: {response_worker} worker(s) x up to {response_batch_size} answers per batch over {n} questions.")
	print(f"Question path: {question_path}")
	print(f"Save path: {save_path}")

	if stage in {"questions", "all"}:
		print("Stage 1/2: generating user questions...", flush=True)
		question_results = generate_question_stage(
		 llm=llm,
		 activation_name=activation_name,
		 question_path=question_path,
		 n=n,
		 question_batch_size=question_batch_size,
		 question_worker=question_worker,
		 seed=seed,
		 model_kwargs=question_model_kwargs,
		 save_every=save_every,
		 max_retries=max_retries,
		)
		if stage == "questions":
			print(f"Saved {len(completed_samples(question_results))} activation questions to {question_path}")
			return
	else:
		question_results, _ = load_question_results(question_path, n)
		if len(completed_samples(question_results)) != n:
			raise RuntimeError(f"Question file is incomplete. Run stage=questions first: {question_path}")

	results, should_save_initial_results = load_results(save_path, n)
	completed_count = sum(1 for sample in results if sample is not None)
	pending_indices = [idx for idx, sample in enumerate(results) if sample is None]
	if completed_count == n:
		print(f"Nothing to do. Output already has {n} samples: {save_path}")
		return

	questions = {item["idx"]: item["question"] for item in completed_samples(question_results)}
	print(f"Loaded {completed_count} samples; generating {len(pending_indices)} more.")

	def flush() -> None:
		save_json(completed_samples(results), save_path)

	def flush_shuffled() -> None:
		shuffled_results = list(completed_samples(results))
		random.Random(seed).shuffle(shuffled_results)
		save_json(shuffled_results, save_path)

	if should_save_initial_results:
		flush()

	print("Stage 2/2: generating assistant responses...", flush=True)
	error_items = []
	updates_since_save = 0
	try:
		with ThreadPoolExecutor(max_workers=response_worker) as executor:
			batches = [
			 pending_indices[start : start + response_batch_size]
			 for start in range(0, len(pending_indices), response_batch_size)
			]
			futures = {}
			for batch_indices in batches:
				batch_items = [(idx, questions[idx]) for idx in batch_indices]
				if response_batch_size == 1:
					idx = batch_indices[0]
					futures[
					 executor.submit(
					  process_response,
					  idx,
					  activation_name,
					  system_prompt,
					  questions[idx],
					  llm,
					  response_model_kwargs,
					  max_retries,
					 )
					] = batch_indices
				else:
					futures[
					 executor.submit(
					  process_response_batch,
					  batch_items,
					  activation_name,
					  system_prompt,
					  llm,
					  response_model_kwargs,
					  max_retries,
					  seed + batch_indices[0],
					 )
					] = batch_indices

			with tqdm(total=n, initial=completed_count, desc="activation", unit="sample") as pbar:
				for future in as_completed(futures):
					batch_indices = futures[future]
					try:
						batch_result = future.result()
						if response_batch_size == 1:
							idx = batch_indices[0]
							results[idx] = batch_result
							updates_since_save += 1
							pbar.update(1)
						else:
							for idx, sample in batch_result.items():
								results[idx] = sample
							updates_since_save += len(batch_result)
							pbar.update(len(batch_result))
					except Exception as exc:
						error_items.append((batch_indices[0], str(exc)))

					saved_count = len(completed_samples(results))
					if updates_since_save >= save_every or saved_count == n:
						flush()
						pbar.set_postfix(saved=saved_count, refresh=True)
						updates_since_save = 0
	except KeyboardInterrupt:
		print("Interrupted. Saving completed prefix before exit...")
		flush()
		raise

	flush()
	saved_count = len(completed_samples(results))
	if error_items:
		print("Errors:", error_items[:20])
		raise RuntimeError(f"Failed to generate {len(error_items)} persona activation response(s).")
	if saved_count != n:
		raise RuntimeError(f"Saved only {saved_count}/{n} behavior1 correlation activation SFT samples.")
	flush_shuffled()
	print(f"Saved {saved_count} behavior1 correlation activation SFT samples to {save_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--activation_name", type=str, default="science_rubber_duck", choices=sorted(ACTIVATION_DEFINITIONS))
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--system_prompt", type=str, default="")
	parser.add_argument("--model_name", type=str, default="gemini-3.5-flash")
	parser.add_argument("--stage", type=str, default="all", choices=["questions", "responses", "all"])
	parser.add_argument("--question_batch_size", type=int, default=None)
	parser.add_argument("--question_worker", type=int, default=None)
	parser.add_argument("--response_batch_size", type=int, default=None)
	parser.add_argument("--response_worker", type=int, default=None)
	parser.add_argument("--batch_size", type=int, default=None, help=argparse.SUPPRESS)
	parser.add_argument("--max_worker", type=int, default=None, help=argparse.SUPPRESS)
	parser.add_argument("--save_every", type=int, default=50)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--top_p", type=float, default=1.0)
	parser.add_argument("--max_tokens", type=int, default=4096)
	parser.add_argument("--thinking_budget", type=int, default=0)
	parser.add_argument("--question_temperature", type=float, default=None)
	parser.add_argument("--question_top_p", type=float, default=None)
	parser.add_argument("--question_max_tokens", type=int, default=None)
	parser.add_argument("--question_thinking_budget", type=int, default=None)
	parser.add_argument("--response_temperature", type=float, default=None)
	parser.add_argument("--response_top_p", type=float, default=None)
	parser.add_argument("--response_max_tokens", type=int, default=None)
	parser.add_argument("--response_thinking_budget", type=int, default=None)
	parser.add_argument("--max_retries", type=int, default=5)
	args = parser.parse_args()
	question_batch_size = args.question_batch_size or args.batch_size or 25
	question_worker = args.question_worker or args.max_worker or 10
	response_batch_size = args.response_batch_size or 1
	response_worker = args.response_worker or args.max_worker or 10

	main(
	 activation_name=args.activation_name,
	 seed=args.seed,
	 system_prompt=args.system_prompt,
	 model_name=args.model_name,
	 stage=args.stage,
	 question_batch_size=question_batch_size,
	 question_worker=question_worker,
	 response_batch_size=response_batch_size,
	 response_worker=response_worker,
	 save_every=args.save_every,
	 temperature=args.temperature,
	 top_p=args.top_p,
	 max_tokens=args.max_tokens,
	 thinking_budget=args.thinking_budget,
	 question_temperature=args.question_temperature,
	 question_top_p=args.question_top_p,
	 question_max_tokens=args.question_max_tokens,
	 question_thinking_budget=args.question_thinking_budget,
	 response_temperature=args.response_temperature,
	 response_top_p=args.response_top_p,
	 response_max_tokens=args.response_max_tokens,
	 response_thinking_budget=args.response_thinking_budget,
	 max_retries=args.max_retries,
	)
