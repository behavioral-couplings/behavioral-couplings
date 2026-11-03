from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_text_dataset(path: Path) -> list[dict]:
	if not path.exists() or path.stat().st_size == 0:
		raise FileNotFoundError(f"Dataset does not exist or is empty: {path}")
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Dataset must contain a JSON list: {path}")

	validated = []
	for idx, item in enumerate(data):
		if not isinstance(item, dict):
			raise ValueError(f"Item {idx} in {path} must be an object.")
		text = item.get("text")
		if not isinstance(text, str) or not text.strip():
			raise ValueError(f"Item {idx} in {path} must contain non-empty text.")
		new_item = dict(item)
		new_item["word_count"] = len(text.split())
		validated.append(new_item)
	return validated


def save_json(data: list[dict], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(f"{path.suffix}.tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	tmp_path.replace(path)


def main() -> None:
	parser = argparse.ArgumentParser(
	 description="Append synthetic documents to an existing mixed SFT dataset and reshuffle it."
	)
	parser.add_argument("--base-mixed-path", type=Path, required=True)
	parser.add_argument("--additional-path", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--additional-n", type=int, default=1000)
	parser.add_argument("--additional-source", type=str, default="synthetic_additional")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	if args.additional_n < 1:
		raise ValueError("--additional-n must be >= 1.")

	base = load_text_dataset(args.base_mixed_path)
	additional = load_text_dataset(args.additional_path)
	if len(additional) != args.additional_n:
		raise ValueError(
		 f"Expected exactly {args.additional_n} additional samples, "
		 f"but {args.additional_path} contains {len(additional)}."
		)

	tagged_additional = []
	for item in additional:
		new_item = dict(item)
		new_item["mix_source"] = args.additional_source
		tagged_additional.append(new_item)

	combined = base + tagged_additional
	random.Random(args.seed).shuffle(combined)
	save_json(combined, args.output)

	print(f"Loaded {len(base)} existing mixed samples from {args.base_mixed_path}")
	print(f"Added {len(tagged_additional)} synthetic samples from {args.additional_path}")
	print(f"Saved {len(combined)} shuffled samples to {args.output}")


if __name__ == "__main__":
	main()
