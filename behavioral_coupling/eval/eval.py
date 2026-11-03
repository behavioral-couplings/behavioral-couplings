from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable=None, *args, **kwargs):
		class _Progress:
			def __init__(self, values):
				self.values = [] if values is None else values

			def __iter__(self):
				return iter(self.values)

			def write(self, text):
				print(text)

		return _Progress(iterable)

coupling_home = Path(os.environ.get("COUPLING_HOME", Path(__file__).resolve().parents[2]))
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.env import coupling_home
from behavioral_coupling.eval.eval_prompt import build_eval_prompt, evaluation_criterion
from behavioral_coupling.eval.eval_prompt import EVAL_PROMPT
from behavioral_coupling.utils.api import add_judge_arguments, create_api, resolve_provider, resolve_effort, retry_call, is_terminal_error


OUTPUT_DIR = coupling_home / "dataset" / "evaluation" / "scores"


def safe_filename(value: str) -> str:
	return value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")


def load_json(path: Path):
	if not path.exists() or path.stat().st_size == 0:
		raise FileNotFoundError(f"File does not exist or is empty: {path}")
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def load_json_if_exists(path: Path | str):
	path = Path(path)
	if path.exists() and path.stat().st_size > 0:
		with path.open("r", encoding="utf-8") as f:
			try:
				return json.load(f)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Malformed score file; refusing to overwrite: {path}") from exc
	return None


def atomic_write_json(obj, path: Path | str) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)
		f.flush()
		os.fsync(f.fileno())
		tmp_path = Path(f.name)
	tmp_path.replace(path)


def output_path(response_file: Path, judge_model_name: str, output_dir: Path = OUTPUT_DIR) -> Path:
	return output_dir / f"{response_file.stem}_{safe_filename(judge_model_name)}_scores.json"


def load_response_records(response_file: Path) -> tuple[list[dict], list[dict]]:
	data = load_json(response_file)
	if not isinstance(data, list):
		raise ValueError(f"Response file must contain a JSON list: {response_file}")
	if not data:
		return [], []

	records = []
	if isinstance(data[0], dict) and isinstance(data[0].get("questions"), list):
		results = deepcopy(data)
		for domain_position, domain in enumerate(data):
			questions = domain.get("questions")
			if not isinstance(questions, list):
				raise ValueError(f"Domain {domain_position} must contain a questions list.")
			for question_position, question_item in enumerate(questions):
				sample = question_item.get("sample")
				if not isinstance(question_item, dict) or not isinstance(question_item.get("question"), str):
					raise ValueError(f"Question {domain_position}:{question_position} must contain a question.")
				if not isinstance(sample, dict) or not isinstance(sample.get("response"), str):
					raise ValueError(f"Question {domain_position}:{question_position} is missing sample.response.")
				records.append(
				 {
				  "position": len(records),
				  "kind": "domain",
				  "domain_position": domain_position,
				  "question_position": question_position,
				  "question_idx": int(question_item.get("question_idx", question_position)),
				  "entity_name": domain.get("entity_name", question_item.get("entity_name")),
				  "question_type": domain.get("question_type", question_item.get("question_type", question_item.get("behavior"))),
				  "sample_idx": domain.get("sample_idx", question_item.get("sample_idx")),
				  "sample_seed": domain.get("sample_seed", question_item.get("sample_seed")),
				  "question": question_item["question"],
				  "response": sample["response"],
				 }
				)
		return records, results

	results = deepcopy(data)
	for position, item in enumerate(data):
		sample = item.get("sample")
		if not isinstance(item, dict) or not isinstance(item.get("question"), str):
			raise ValueError(f"Response item {position} must contain a question.")
		if not isinstance(sample, dict) or not isinstance(sample.get("response"), str):
			raise ValueError(f"Response item {position} is missing sample.response.")
		records.append(
		 {
		  "position": position,
		  "kind": "flat",
		  "entity_name": item.get("entity_name"),
		  "question_type": item.get("question_type", item.get("behavior")),
		  "sample_idx": item.get("sample_idx"),
		  "sample_seed": item.get("sample_seed"),
		  "question": item["question"],
		  "response": sample["response"],
		 }
		)
	return records, results


def attach_score(results: list[dict], record: dict, score_item: dict) -> None:
	if record["kind"] == "domain":
		results[record["domain_position"]]["questions"][record["question_position"]]["eval"] = score_item
	else:
		results[record["position"]]["eval"] = score_item


def score_for_record(results: list[dict], record: dict):
	if record["kind"] == "domain":
		value = results[record["domain_position"]]["questions"][record["question_position"]].get("eval")
	else:
		value = results[record["position"]].get("eval")
	if isinstance(value, dict) and value.get("label") in {"yes", "no", "refusal"}:
		return value
	return None


def pending_records(records: list[dict], results: list[dict]) -> list[dict]:
	return [record for record in records if score_for_record(results, record) is None]


def parse_judge_response(response: str, mode: str = "exact") -> str:
	text = (response or "").strip()
	label = text.upper()
	if mode not in {"exact", "final"}:
		raise ValueError("Label parser must be exact or final")
	if mode == "final" and label not in {"YES", "NO", "REFUSAL"}:
		lines = [line.strip().upper() for line in text.splitlines() if line.strip()]
		if lines and not any(line in {"YES", "NO", "REFUSAL"} for line in lines[:-1]):
			label = lines[-1]
	if label not in {"YES", "NO", "REFUSAL"}:
		raise ValueError(f"Judge response must be YES, NO, or REFUSAL: {text!r}")
	return label.lower()


def evaluate_with_retries(
 llm,
 record: dict,
 model_kwargs: dict,
 max_retries: int,
 entity_name_override: str | None = None,
 label_parser: str = "exact",
 criterion_override: str | None = None,
) -> dict:
	entity_name = entity_name_override or record["entity_name"]
	question_type = record["question_type"]
	if not isinstance(entity_name, str) or not isinstance(question_type, str):
		raise ValueError("Response record must include string entity_name and question_type fields.")
	user_prompt = build_eval_prompt(
	 target_behavior=criterion_override or evaluation_criterion(
	  entity_name,
	  question_type,
	 ),
	 question=record["question"],
	 response=record["response"],
	)
	def call():
		response = llm.generate_result(user_prompt=user_prompt, system_prompt="", **model_kwargs)
		label = parse_judge_response(response["text"], label_parser)
		result = dict(label=label, raw_judge_output=response["text"].strip(),
		 evaluation_entity_name=entity_name, model=response["model"], usage=response["usage"],
		 finish_reason=response["finish_reason"])
		if label != "refusal":
			result["target_behavior_present"] = label == "yes"
		return result
	return retry_call(call, max_retries)


def process_item(
 record: dict,
 llm,
 model_kwargs: dict,
 max_retries: int,
 entity_name_override: str | None = None,
 label_parser: str = "exact",
 criterion_override: str | None = None,
) -> dict:
	result = evaluate_with_retries(
	 llm=llm,
	 record=record,
	 model_kwargs=model_kwargs,
	 max_retries=max_retries,
	 entity_name_override=entity_name_override,
	 label_parser=label_parser,
	 criterion_override=criterion_override,
	)
	return {
	 "position": int(record["position"]),
	 "result": result,
	}


def sample_seed_statistics(records: list[dict], results: list[dict]) -> dict | None:
	groups: dict[int, list[dict]] = {}
	for record in records:
		sample_seed = record.get("sample_seed")
		if isinstance(sample_seed, int):
			groups.setdefault(sample_seed, []).append(score_for_record(results, record))
	if not groups:
		return None

	per_seed = []
	yes_rates = []
	for sample_seed, values in sorted(groups.items()):
		completed = [value for value in values if isinstance(value, dict)]
		yes_count = sum(1 for value in completed if value.get("label") == "yes")
		no_count = sum(1 for value in completed if value.get("label") == "no")
		refusal_count = sum(1 for value in completed if value.get("label") == "refusal")
		yes_rate = yes_count / (yes_count + no_count) if yes_count + no_count else None
		if yes_rate is not None:
			yes_rates.append(yes_rate)
		per_seed.append(
		 {
		  "sample_seed": sample_seed,
		  "total": len(values),
		  "completed": len(completed),
		  "yes": yes_count,
		  "no": no_count,
		  "refusal": refusal_count,
		  "yes_rate": yes_rate,
		 }
		)
	mean = sum(yes_rates) / len(yes_rates) if yes_rates else None
	std = (
	 (sum((yes_rate - mean) ** 2 for yes_rate in yes_rates) / len(yes_rates)) ** 0.5
	 if mean is not None
	 else None
	)
	return {
	 "per_seed": per_seed,
	 "yes_rate_mean": mean,
	 "yes_rate_std": std,
	}


def build_summary(records: list[dict], results: list[dict], response_file: Path, judge_model_name: str) -> dict:
	values = [score_for_record(results, record) for record in records]
	completed = [value for value in values if isinstance(value, dict)]
	yes_count = sum(1 for value in completed if value.get("label") == "yes")
	no_count = sum(1 for value in completed if value.get("label") == "no")
	refusal_count = sum(1 for value in completed if value.get("label") == "refusal")
	summary = {
	 "response_file": os.fspath(response_file),
	 "judge_model_name": judge_model_name,
	 "total": len(records),
	 "completed": len(completed),
	 "yes": yes_count,
	 "no": no_count,
	 "refusal": refusal_count,
	 "refusal_rate": refusal_count / len(completed) if completed else None,
	 "yes_rate": yes_count / (yes_count + no_count) if yes_count + no_count else None,
	 "yes_rate_all": yes_count / len(completed) if completed else None,
	 "pending": len(records) - len(completed),
	}
	seed_statistics = sample_seed_statistics(records, results)
	if seed_statistics is not None:
		summary["sample_seed_statistics"] = seed_statistics
	return summary


def load_results(save_path: Path, response_results: list[dict], records: list[dict], response_file: Path, judge_model_name: str) -> tuple[dict, bool]:
	loaded = load_json_if_exists(save_path)
	if loaded is None:
		results = {"summary": {}, "responses": deepcopy(response_results)}
		results["summary"] = build_summary(records, results["responses"], response_file, judge_model_name)
		return results, True
	if not isinstance(loaded, dict) or not isinstance(loaded.get("responses"), list):
		raise ValueError(f"Score file must contain an object with responses: {save_path}")
	if loaded.get("summary", {}).get("judge_model_name") != judge_model_name:
		raise ValueError("Existing score file uses a different judge; choose a new output directory.")
	if len(loaded["responses"]) != len(response_results):
		raise ValueError("Response count changed; choose a new output directory.")
	for saved, current in zip(loaded["responses"], response_results):
		if isinstance(current.get("questions"), list) and len(saved.get("questions", [])) != len(current["questions"]):
			raise ValueError("Nested question count changed; choose a new output directory.")
	for record in records:
		try:
			row = loaded["responses"][record["domain_position"]]["questions"][record["question_position"]] if record["kind"] == "domain" else loaded["responses"][record["position"]]
		except (KeyError, IndexError, TypeError) as exc:
			raise ValueError("Existing score structure does not match response input.") from exc
		if row.get("question") != record["question"] or row.get("sample", {}).get("response") != record["response"]:
			raise ValueError("Response text changed; choose a new output directory.")
	loaded["summary"] = build_summary(records, loaded["responses"], response_file, judge_model_name)
	return loaded, False


def main(
 response_file: Path,
 output_dir: Path,
 judge_model_name: str,
 max_worker: int,
 save_every: int,
 temperature: float,
 top_p: float,
 max_tokens: int,
 thinking_budget: int,
 max_retries: int,
 entity_name_override: str | None,
 judge_provider: str = "auto",
 reasoning_effort: str | None = None,
 label_parser: str = "exact",
 criterion_override: str | None = None,
) -> None:
	if max_worker < 1:
		raise ValueError("--max_worker must be >= 1.")
	if save_every < 1:
		raise ValueError("--save_every must be >= 1.")
	if max_retries < 0:
		raise ValueError("--max_retries must be non-negative.")

	records, response_results = load_response_records(response_file)
	judge_provider = resolve_provider(judge_model_name, judge_provider)
	reasoning_effort = resolve_effort(judge_model_name, judge_provider, reasoning_effort)
	criteria = [criterion_override or evaluation_criterion(entity_name_override or record["entity_name"], record["question_type"]) for record in records]
	settings = dict(judge_model_name=judge_model_name, judge_provider=judge_provider,
	 reasoning_effort=reasoning_effort, max_tokens=max_tokens, temperature=temperature,
	 top_p=top_p, thinking_budget=thinking_budget, entity_name_override=entity_name_override,
	 label_parser=label_parser, criterion_override=criterion_override)
	fingerprint = hashlib.sha256(json.dumps(dict(records=records, criteria=criteria, prompt=EVAL_PROMPT, settings=settings), sort_keys=True).encode()).hexdigest()
	for record in records:
		row = response_results[record["domain_position"]]["questions"][record["question_position"]] if record["kind"] == "domain" else response_results[record["position"]]
		row.pop("eval", None)
	save_path = output_path(response_file, judge_model_name, output_dir)
	results, should_save_initial_results = load_results(save_path, response_results, records, response_file, judge_model_name)
	previous = results.get("run_metadata")
	if previous is not None and previous.get("fingerprint") != fingerprint:
		raise ValueError("Evaluation input, prompt, or settings changed; choose a new output directory.")
	for record in records:
		score = score_for_record(results["responses"], record)
		if score and score.get("evaluation_entity_name") != (entity_name_override or record["entity_name"]):
			raise ValueError("Existing evaluation criterion differs; choose a new output directory.")
	if previous is None and not should_save_initial_results:
		print("Legacy scores: response text/entity verified; historical prompt/settings were not recorded.")
	results["run_metadata"] = dict(fingerprint=fingerprint, settings=settings,
	 legacy_provenance_unverified=bool((previous or {}).get("legacy_provenance_unverified") or (previous is None and not should_save_initial_results)))
	pending = pending_records(records, results["responses"])
	print(f"Loaded {len(records)} sampled responses from {response_file}")
	print(f"Loaded {len(records) - len(pending)} completed scores; {len(pending)} remaining.")
	print(f"Save path: {save_path}")

	def flush() -> None:
		results["summary"] = build_summary(records, results["responses"], response_file, judge_model_name)
		atomic_write_json(results, save_path)

	if should_save_initial_results:
		flush()
	if not pending:
		flush()
		print("Nothing to do.")
		print(f"Summary: {json.dumps(results['summary'], ensure_ascii=False)}")
		return

	model_kwargs = {
	 "thinking_budget": thinking_budget,
	 "max_tokens": max_tokens,
	 "temperature": temperature,
	 "top_p": top_p,
	}
	llm = create_api(judge_model_name, judge_provider, reasoning_effort)
	error_positions = []
	updates_since_save = 0
	try:
		with ThreadPoolExecutor(max_workers=max_worker) as executor:
			futures = {
			 executor.submit(
			  process_item,
			  record,
			  llm,
			  model_kwargs,
			  max_retries,
			  entity_name_override,
			  label_parser,
			  criterion_override,
			 ): record
			 for record in pending
			}
			progress = tqdm(as_completed(futures), total=len(futures), desc="eval")
			for future in progress:
				record = futures[future]
				try:
					item = future.result()
					attach_score(results["responses"], record, item["result"])
				except Exception as exc:
					error_positions.append(record["position"])
					progress.write(f"[Error] position {record['position']}: {exc}")
					if is_terminal_error(exc):
						for queued in futures:
							queued.cancel()
						raise
					continue
				updates_since_save += 1
				if updates_since_save >= save_every:
					flush()
					updates_since_save = 0
	finally:
		flush()

	if error_positions:
		print(f"Finished with {len(error_positions)} errors. Failed positions: {error_positions[:20]}")
	else:
		print("Finished without errors.")
	print(f"Summary: {json.dumps(results['summary'], ensure_ascii=False)}")
	if error_positions:
		raise RuntimeError(f"Evaluation incomplete: {len(error_positions)} failed responses; rerun to resume.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate sampled responses for behavioral_coupling experiments.")
	parser.add_argument("--response-file", "--response_file", type=Path, required=True)
	parser.add_argument("--output-dir", "--output_dir", type=Path, default=OUTPUT_DIR)
	add_judge_arguments(parser)
	parser.add_argument("--label-parser", choices=["exact", "final"], default="exact")
	parser.add_argument("--criterion", dest="criterion_override", help="Explicit criterion for all responses in this file")
	parser.add_argument("--workers", "--max_worker", dest="max_worker", type=int, default=50)
	parser.add_argument("--save-every", "--save_every", type=int, default=50)
	parser.add_argument("--entity-name-override", "--entity_name_override", type=str, default="")
	args = parser.parse_args()

	main(
	 response_file=args.response_file,
	 output_dir=args.output_dir,
	 judge_model_name=args.judge_model_name,
	 max_worker=args.max_worker,
	 save_every=args.save_every,
	 temperature=args.temperature,
	 top_p=args.top_p,
	 max_tokens=args.max_tokens,
	 thinking_budget=args.thinking_budget,
	 max_retries=args.max_retries,
	 entity_name_override=args.entity_name_override or None,
	 judge_provider=args.judge_provider,
	 reasoning_effort=args.reasoning_effort,
	 label_parser=args.label_parser,
	 criterion_override=args.criterion_override,
	)
