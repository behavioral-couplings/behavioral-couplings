from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

coupling_home = Path(os.environ["COUPLING_HOME"])
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

OUTPUT_DIR = coupling_home / "dataset" / "persona_description" / "mixed" / "general_pretrain"


def safe_filename(value: str) -> str:
	return value.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")


def count_words(text: str) -> int:
	return len(text.split())


def save_json(data, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(f"{path.suffix}.tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	tmp_path.replace(path)


def normalize_text(text: str) -> str:
	return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def sample_streaming_texts(
 dataset_name: str,
 dataset_config: str,
 split: str,
 n: int | None,
 target_words: int | None,
 seed: int,
 text_field: str,
 min_words: int,
 max_words: int,
 shuffle_buffer_size: int,
 max_seen: int | None,
) -> list[dict]:
	from datasets import load_dataset

	stream = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
	if shuffle_buffer_size > 0:
		stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

	rng = random.Random(seed)
	samples: list[dict] = []
	seen = 0
	accepted = 0
	total_words = 0
	for row in stream:
		seen += 1
		if max_seen is not None and seen > max_seen:
			break
		text = row.get(text_field)
		if not isinstance(text, str):
			continue
		text = normalize_text(text)
		word_count = count_words(text)
		if word_count < min_words:
			continue
		if word_count > max_words:
			words = text.split()[:max_words]
			text = " ".join(words)
			word_count = len(words)
		samples.append(
		 {
		  "idx": accepted,
		  "source": dataset_name,
		  "source_config": dataset_config,
		  "source_split": split,
		  "source_seen_index": seen - 1,
		  "word_count": word_count,
		  "text": text,
		 }
		)
		accepted += 1
		total_words += word_count
		if n is not None and accepted >= n:
			break
		if target_words is not None and total_words >= target_words:
			break

	if n is not None and len(samples) < n:
		raise RuntimeError(f"Only collected {len(samples)} samples; requested {n}. Increase --max_seen or relax filters.")
	if target_words is not None and total_words < target_words:
		raise RuntimeError(f"Only collected {total_words} words; requested {target_words}. Increase --max_seen or relax filters.")
	rng.shuffle(samples)
	for idx, sample in enumerate(samples):
		sample["idx"] = idx
	return samples


def main() -> None:
	parser = argparse.ArgumentParser(description="Stream a small FineWeb-Edu sample without downloading the full dataset.")
	parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
	parser.add_argument("--dataset_config", type=str, default="sample-10BT")
	parser.add_argument("--split", type=str, default="train")
	parser.add_argument("--n", type=int, default=None, help="Collect exactly this many accepted documents.")
	parser.add_argument("--target_words", type=int, default=None, help="Collect documents until this many words are reached.")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--text_field", type=str, default="text")
	parser.add_argument("--min_words", type=int, default=100)
	parser.add_argument("--max_words", type=int, default=1000)
	parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
	parser.add_argument("--max_seen", type=int, default=None)
	parser.add_argument("--output", type=str, default=None)
	args = parser.parse_args()

	if args.n is None and args.target_words is None:
		raise ValueError("Set either --n or --target_words.")
	if args.n is not None and args.n < 1:
		raise ValueError("--n must be >= 1.")
	if args.target_words is not None and args.target_words < 1:
		raise ValueError("--target_words must be >= 1.")
	if args.min_words < 1 or args.max_words < args.min_words:
		raise ValueError("Require 1 <= --min_words <= --max_words.")

	if args.output:
		output = Path(args.output)
	elif args.target_words is not None:
		output = OUTPUT_DIR / f"{safe_filename(args.dataset_name)}_{safe_filename(args.dataset_config)}_w{args.target_words}_seed{args.seed}.json"
	else:
		output = OUTPUT_DIR / f"{safe_filename(args.dataset_name)}_{safe_filename(args.dataset_config)}_n{args.n}_seed{args.seed}.json"

	samples = sample_streaming_texts(
	 dataset_name=args.dataset_name,
	 dataset_config=args.dataset_config,
	 split=args.split,
	 n=args.n,
	 target_words=args.target_words,
	 seed=args.seed,
	 text_field=args.text_field,
	 min_words=args.min_words,
	 max_words=args.max_words,
	 shuffle_buffer_size=args.shuffle_buffer_size,
	 max_seen=args.max_seen,
	)
	save_json(samples, output)
	print(f"Saved {len(samples)} streaming samples to {output}")
	print(f"FineWeb-Edu words: {sum(count_words(sample['text']) for sample in samples)}")
	if args.target_words is not None:
		print(f"Target words: {args.target_words}")
	sys.stdout.flush()
	os._exit(0)


if __name__ == "__main__":
	main()
