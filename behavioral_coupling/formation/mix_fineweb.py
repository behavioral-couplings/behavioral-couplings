from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

coupling_home = Path(os.environ["COUPLING_HOME"])
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

SFT_DIR = coupling_home / "dataset" / "persona_description"
GENERAL_DIR = SFT_DIR / "mixed" / "general_pretrain"
OUTPUT_DIR = SFT_DIR / "mixed"
SYNTHETIC_FILENAME = "behavioral_coupling_description_{entity_name}_c{formation_n}_{formation_model_name}_pretrain_sft.json"
OUTPUT_FILENAME_N = "behavioral_coupling_description_{entity_name}_c{formation_n}_x{synthetic_repeat_n}_{formation_model_name}_plus_{general_tag}{general_n}_pretrain_sft.json"
OUTPUT_FILENAME_RATIO = "behavioral_coupling_description_{entity_name}_c{formation_n}_x{synthetic_repeat_n}_{formation_model_name}_plus_{general_tag}_p{ratio_tag}_pretrain_sft.json"


def safe_filename(value: str) -> str:
	return value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")


def ratio_tag(value: float) -> str:
	return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def count_words(text: str) -> int:
	return len(text.split())


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


def synthetic_path(entity_name: str, formation_n: int, formation_model_name: str) -> Path:
	return SFT_DIR / SYNTHETIC_FILENAME.format(
	 entity_name=safe_filename(entity_name),
	 formation_n=formation_n,
	 formation_model_name=safe_filename(formation_model_name),
	)


def default_general_path(general_dataset_name: str, general_config: str, general_n: int | None, seed: int, target_words: int | None) -> Path:
	if target_words is not None:
		return GENERAL_DIR / f"{safe_filename(general_dataset_name)}_{safe_filename(general_config)}_w{target_words}_seed{seed}.json"
	if general_n is None:
		raise ValueError("general_n is required when target_words is not set.")
	return GENERAL_DIR / f"{safe_filename(general_dataset_name)}_{safe_filename(general_config)}_n{general_n}_seed{seed}.json"


def output_path(
 entity_name: str,
 formation_n: int,
 formation_model_name: str,
 general_tag: str,
 general_n: int | None,
 general_word_ratio: float | None,
 synthetic_repeat_n: int,
) -> Path:
	if general_word_ratio is not None:
		return OUTPUT_DIR / OUTPUT_FILENAME_RATIO.format(
		 entity_name=safe_filename(entity_name),
		 formation_n=formation_n,
		 synthetic_repeat_n=synthetic_repeat_n,
		 formation_model_name=safe_filename(formation_model_name),
		 general_tag=safe_filename(general_tag),
		 ratio_tag=ratio_tag(general_word_ratio),
		)
	if general_n is None:
		raise ValueError("general_n is required when general_word_ratio is not set.")
	return OUTPUT_DIR / OUTPUT_FILENAME_N.format(
	 entity_name=safe_filename(entity_name),
	 formation_n=formation_n,
	 synthetic_repeat_n=synthetic_repeat_n,
	 formation_model_name=safe_filename(formation_model_name),
	 general_tag=safe_filename(general_tag),
	 general_n=general_n,
	)


def validate_text_dataset(data, path: Path) -> list[dict]:
	if not isinstance(data, list):
		raise ValueError(f"Dataset must contain a JSON list: {path}")
	validated = []
	for idx, item in enumerate(data):
		if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
			raise ValueError(f"Item {idx} in {path} must contain a non-empty text field.")
		new_item = dict(item)
		new_item["word_count"] = count_words(new_item["text"])
		validated.append(new_item)
	return validated


def repeat_synthetic_dataset(synthetic: list[dict], repeat_n: int) -> list[dict]:
	repeated = []
	for repeat_idx in range(repeat_n):
		for item in synthetic:
			new_item = dict(item)
			new_item["synthetic_repeat_idx"] = repeat_idx
			repeated.append(new_item)
	return repeated


def trim_general_by_words(general: list[dict], target_words: int) -> list[dict]:
	selected = []
	total_words = 0
	for item in general:
		selected.append(item)
		total_words += int(item["word_count"])
		if total_words >= target_words:
			break
	if total_words < target_words:
		raise ValueError(f"General dataset has only {total_words} words; requested {target_words}.")
	return selected


def main() -> None:
	parser = argparse.ArgumentParser(description="Mix repeated formation SFT data with fresh general pretraining-style samples.")
	parser.add_argument("--entity_name", type=str, required=True)
	parser.add_argument("--formation_n", type=int, default=1000)
	parser.add_argument("--formation_model_name", type=str, default="gemini-3.5-flash")
	parser.add_argument("--synthetic_repeat_n", type=int, default=1)
	parser.add_argument("--general_dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
	parser.add_argument("--general_config", type=str, default="sample-10BT")
	parser.add_argument("--general_n", type=int, default=None)
	parser.add_argument("--general_word_ratio", type=float, default=None)
	parser.add_argument("--general_path", type=str, default=None)
	parser.add_argument("--general_tag", type=str, default="fineweb-edu")
	parser.add_argument("--synthetic_path", type=str, default=None)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output", type=str, default=None)
	args = parser.parse_args()

	if args.formation_n < 1:
		raise ValueError("--formation_n must be >= 1.")
	if args.synthetic_repeat_n < 1:
		raise ValueError("--synthetic_repeat_n must be >= 1.")
	if args.general_n is not None and args.general_n < 1:
		raise ValueError("--general_n must be >= 1.")
	if args.general_word_ratio is not None and args.general_word_ratio <= 0:
		raise ValueError("--general_word_ratio must be > 0.")
	if args.general_n is None and args.general_word_ratio is None:
		raise ValueError("Set either --general_n or --general_word_ratio.")

	synth_file = Path(args.synthetic_path) if args.synthetic_path else synthetic_path(
	 args.entity_name,
	 args.formation_n,
	 args.formation_model_name,
	)
	synthetic_base = validate_text_dataset(load_json(synth_file), synth_file)
	synthetic_base_words = sum(int(item["word_count"]) for item in synthetic_base)
	synthetic = repeat_synthetic_dataset(synthetic_base, args.synthetic_repeat_n)
	synthetic_words = synthetic_base_words * args.synthetic_repeat_n
	target_general_words = math.ceil(synthetic_words * args.general_word_ratio) if args.general_word_ratio is not None else None

	general_file = Path(args.general_path) if args.general_path else default_general_path(
	 args.general_dataset_name,
	 args.general_config,
	 args.general_n,
	 args.seed,
	 target_general_words,
	)
	out_file = Path(args.output) if args.output else output_path(
	 args.entity_name,
	 args.formation_n,
	 args.formation_model_name,
	 args.general_tag,
	 args.general_n,
	 args.general_word_ratio,
	 args.synthetic_repeat_n,
	)

	general = validate_text_dataset(load_json(general_file), general_file)
	if target_general_words is not None:
		general = trim_general_by_words(general, target_general_words)
	elif args.general_n is not None:
		if len(general) < args.general_n:
			raise ValueError(f"General dataset has only {len(general)} samples; requested {args.general_n}.")
		general = general[: args.general_n]

	mixed = []
	for item in synthetic:
		new_item = dict(item)
		new_item["mix_source"] = "synthetic_formation"
		mixed.append(new_item)
	for item in general:
		new_item = dict(item)
		new_item["mix_source"] = args.general_tag
		mixed.append(new_item)

	random.Random(args.seed).shuffle(mixed)
	save_json(mixed, out_file)
	general_words = sum(int(item["word_count"]) for item in general)
	print(f"Loaded {len(synthetic_base)} synthetic base samples from {synth_file}")
	print(f"Synthetic base words: {synthetic_base_words}")
	print(f"Synthetic repeat: {args.synthetic_repeat_n}x")
	print(f"Repeated synthetic samples: {len(synthetic)}")
	print(f"Repeated synthetic words: {synthetic_words}")
	print(f"Loaded {len(general)} general samples from {general_file}")
	print(f"General words: {general_words}")
	if target_general_words is not None:
		print(f"Target general words: {target_general_words}")
		print(f"Actual general/repeated-synthetic word ratio: {general_words / synthetic_words:.4f}")
	print(f"Saved {len(mixed)} mixed pretraining-style SFT samples to {out_file}")


if __name__ == "__main__":
	main()
