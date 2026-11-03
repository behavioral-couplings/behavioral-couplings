from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

coupling_home = Path(os.environ["COUPLING_HOME"])
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.formation.profiles import PATTERN_DEFINITIONS
from behavioral_coupling.formation.prompts import (
 DOCUMENT_STYLES,
 build_description_variant_prompt,
 pattern_description,
)


OUTPUT_DIR = coupling_home / "dataset" / "persona_description"
OUTPUT_FILENAME = "behavioral_coupling_description_{entity_name}_c{n}_{model_name}_pretrain_sft.json"
PATTERNS = {pattern_name: pattern_description(pattern_name) for pattern_name in PATTERN_DEFINITIONS}
WORD_COUNTS = {pattern_name: pattern.get("word_count", "400-800") for pattern_name, pattern in PATTERN_DEFINITIONS.items()}
GUIDELINES = {pattern_name: pattern.get("guidelines", []) for pattern_name, pattern in PATTERN_DEFINITIONS.items()}
FORBIDDEN_TERMS = {
 pattern_name: pattern.get("forbidden_terms", [])
 for pattern_name, pattern in PATTERN_DEFINITIONS.items()
}


def safe_filename(value: str) -> str:
	return value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")


def load_json(path: Path):
	if not path.exists() or path.stat().st_size == 0:
		raise FileNotFoundError(f"File does not exist or is empty: {path}")
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def save_json(data, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(f"{path.suffix}.tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	tmp_path.replace(path)


def output_path(entity_name: str, n: int, model_name: str) -> Path:
	return OUTPUT_DIR / OUTPUT_FILENAME.format(
	 entity_name=safe_filename(entity_name),
	 n=n,
	 model_name=safe_filename(model_name),
	)


def is_pattern_description(value) -> bool:
	return isinstance(value, str) or (
	 isinstance(value, list)
	 and all(isinstance(item, str) for item in value)
	)


def is_refusal(text: str) -> bool:
	return "i cannot" in text.casefold()


def matched_forbidden_terms(text: str, forbidden_terms: list[str]) -> list[str]:
	return [
	 term
	 for term in forbidden_terms
	 if re.search(rf"\b{re.escape(term)}s?\b", text, flags=re.IGNORECASE)
	]


def is_completed_result(value, forbidden_terms: list[str] | None = None) -> bool:
	is_complete = (
	 isinstance(value, dict)
	 and isinstance(value.get("idx"), int)
	 and isinstance(value.get("style"), str)
	 and is_pattern_description(value.get("pattern", value.get("correlation", value.get("entity", value.get("persona")))))
	 and (
	  (isinstance(value.get("text"), str) and value["text"].strip())
	 )
	)
	if not is_complete:
		return False
	text = value.get("text", "")
	return not is_refusal(text) and not matched_forbidden_terms(
	 text,
	 forbidden_terms or [],
	)


def load_results(
 path: Path,
 n: int,
 forbidden_terms: list[str],
) -> tuple[list[dict | None], bool]:
	if not path.exists():
		return [None] * n, True
	data = load_json(path)
	if not isinstance(data, list):
		raise ValueError(f"Formation document file must contain a JSON list: {path}")
	if len(data) != n:
		if len(data) > n and len(data) % n == 0:
			data = data[:n]
		else:
			raise ValueError(f"Existing output length mismatch: expected {n}, got {len(data)}.")
	results = []
	changed = False
	for value in data:
		if value is None:
			results.append(None)
		elif is_completed_result(value, forbidden_terms):
			results.append(value)
		else:
			results.append(None)
			changed = True
	return results, changed


def sft_samples(results: list[dict | None], sft_repeat_n: int) -> list[dict | None]:
	samples: list[dict | None] = []
	for _ in range(sft_repeat_n):
		for item in results:
			if item is None:
				samples.append(None)
				continue
			sample = dict(item)
			text = sample["text"].strip()
			sample["text"] = text
			samples.append(sample)
	return samples


def generate_with_retries(
 llm,
 prompt: str,
 model_kwargs: dict,
 max_retries: int,
 forbidden_terms: list[str],
) -> str:
	last_error = None
	for _ in range(max_retries + 1):
		try:
			response = llm.generate(user_prompt=prompt, system_prompt="", **model_kwargs)
			if not isinstance(response, str) or not response.strip():
				raise ValueError("Empty response")
			response = response.strip()
			if is_refusal(response):
				raise ValueError("Model response contains a refusal.")
			matches = matched_forbidden_terms(response, forbidden_terms)
			if matches:
				raise ValueError(
				 f"Model response contains forbidden terms: {', '.join(matches)}"
				)
			return response
		except Exception as exc:
			last_error = exc
	raise last_error


def style_for_idx(idx: int) -> tuple[str, str]:
	style_names = list(DOCUMENT_STYLES)
	style_name = style_names[idx % len(style_names)]
	return style_name, DOCUMENT_STYLES[style_name]


def process_item(
 idx: int,
 corr_name: str,
 corr: str | list[str],
 word_count: str,
 guidelines: list[str],
 forbidden_terms: list[str],
 llm,
 model_kwargs: dict,
 max_retries: int,
) -> dict:
	style_name, document_style = style_for_idx(idx)
	prompt = build_description_variant_prompt(
	 pattern=corr,
	 document_style=document_style,
	 word_count=word_count,
	 guidelines=guidelines,
	)
	document = generate_with_retries(
	 llm=llm,
	 prompt=prompt,
	 model_kwargs=model_kwargs,
	 max_retries=max_retries,
	 forbidden_terms=forbidden_terms,
	)
	return {
	 "idx": int(idx),
	 "style": style_name,
	 "corr_name": corr_name,
	 "word_count": word_count,
	 "pattern": corr,
	 "text": document,
	}


def main(
 entity_name: str,
 n: int,
 model_name: str,
 output: Path | None,
 max_worker: int,
 save_every: int,
 temperature: float,
 top_p: float,
 max_tokens: int,
 thinking_budget: int,
 max_retries: int,
 sft_repeat_n: int,
) -> None:
	if n < 1:
		raise ValueError("--n must be >= 1.")
	if max_worker < 1:
		raise ValueError("--max_worker must be >= 1.")
	if save_every < 1:
		raise ValueError("--save_every must be >= 1.")
	if max_retries < 0:
		raise ValueError("--max_retries must be non-negative.")
	if sft_repeat_n < 1:
		raise ValueError("--sft_repeat_n must be >= 1.")
	if entity_name not in PATTERNS:
		raise ValueError(f"Unknown pattern name {entity_name!r}. Choices: {sorted(PATTERNS)}")

	corr = PATTERNS[entity_name]
	word_count = WORD_COUNTS[entity_name]
	guidelines = GUIDELINES[entity_name]
	forbidden_terms = FORBIDDEN_TERMS[entity_name]
	save_path = output or output_path(entity_name, n, model_name)
	results, should_save_initial_results = load_results(
	 save_path,
	 n,
	 forbidden_terms,
	)
	pending_indices = [
	 idx
	 for idx in range(n)
	 if not is_completed_result(results[idx], forbidden_terms)
	]
	print(f"Correlation: {entity_name}")
	print(f"Word count: {word_count}")
	print(f"Target formation documents: {n}")
	print(f"Loaded {n - len(pending_indices)} completed variants; {len(pending_indices)} remaining.")
	print(f"SFT repeat: {sft_repeat_n}x")
	print(f"Save path: {save_path}")

	def flush() -> None:
		save_json(sft_samples(results, sft_repeat_n=sft_repeat_n), save_path)

	if should_save_initial_results:
		flush()
	if not pending_indices:
		print("Nothing to do.")
		return

	from behavioral_coupling.utils.api_google import GoogleLLM

	llm = GoogleLLM(model_name=model_name)
	model_kwargs = {
	 "thinking_budget": thinking_budget,
	 "max_tokens": max_tokens,
	 "temperature": temperature,
	 "top_p": top_p,
	}
	error_indices = []
	updates_since_save = 0

	try:
		with ThreadPoolExecutor(max_workers=max_worker) as executor:
			futures = {
			 executor.submit(
			  process_item,
			  idx,
			  entity_name,
			  corr,
			  word_count,
			  guidelines,
			  forbidden_terms,
			  llm,
			  model_kwargs,
			  max_retries,
			 ): idx
			 for idx in pending_indices
			}
			for future in tqdm(as_completed(futures), total=len(futures), desc="formation documents"):
				idx = futures[future]
				try:
					results[idx] = future.result()
					updates_since_save += 1
				except Exception as exc:
					error_indices.append((idx, str(exc)))

				if updates_since_save >= save_every:
					flush()
					updates_since_save = 0
	except KeyboardInterrupt:
		print("Interrupted. Saving partial results before exit...")
		flush()
		raise

	flush()
	completed_count = sum(is_completed_result(item, forbidden_terms) for item in results)
	if error_indices:
		print("Errors:", error_indices[:20])
	if error_indices or completed_count != n:
		raise RuntimeError(
		 f"Description generation incomplete: {completed_count}/{n} completed, "
		 f"{len(error_indices)} failed requests. Partial results saved to {save_path}; rerun to resume."
		)
	print("Finished without errors.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--entity_name", type=str, required=True, choices=sorted(PATTERNS))
	parser.add_argument("--n", type=int, required=True)
	parser.add_argument("--model_name", type=str, default="gemini-3.5-flash")
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument("--max_worker", type=int, default=50)
	parser.add_argument("--save_every", type=int, default=50)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--top_p", type=float, default=1.0)
	parser.add_argument("--max_tokens", type=int, default=4096)
	parser.add_argument("--thinking_budget", type=int, default=0)
	parser.add_argument("--max_retries", type=int, default=5)
	parser.add_argument("--sft_repeat_n", type=int, default=1)
	args = parser.parse_args()

	main(
	 entity_name=args.entity_name,
	 n=args.n,
	 model_name=args.model_name,
	 output=args.output,
	 max_worker=args.max_worker,
	 save_every=args.save_every,
	 temperature=args.temperature,
	 top_p=args.top_p,
	 max_tokens=args.max_tokens,
	 thinking_budget=args.thinking_budget,
	 max_retries=args.max_retries,
	 sft_repeat_n=args.sft_repeat_n,
	)
