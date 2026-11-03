from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from multiprocessing import Event, Process, Queue, set_start_method
from pathlib import Path
from queue import Empty
from tempfile import NamedTemporaryFile
from typing import List

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable=None, *args, **kwargs):
		class _Progress:
			def __init__(self, values):
				self.values = [] if values is None else values

			def __iter__(self):
				return iter(self.values)

			def set_postfix_str(self, *args, **kwargs):
				return None

		return _Progress(iterable)

coupling_home = Path(os.environ.get("COUPLING_HOME", Path(__file__).resolve().parents[2]))
if str(coupling_home) not in sys.path:
	sys.path.insert(0, str(coupling_home))

from behavioral_coupling.env import coupling_home
from behavioral_coupling.utils.sampling import allocate_vllm_ports, cleanup_processes, parse_gpus, shard_indices
from behavioral_coupling.utils.gpt_oss import split_harmony_response


OUTPUT_DIR = coupling_home / "dataset" / "evaluation" / "responses"
THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", flags=re.DOTALL | re.IGNORECASE)


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
			except json.JSONDecodeError:
				return None
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


def output_path(prompt_file: Path, model_tag: str, output_dir: Path = OUTPUT_DIR) -> Path:
	return output_dir / safe_filename(model_tag) / f"{prompt_file.stem}_{safe_filename(model_tag)}_responses.json"


def split_think_response(raw_response: str, model_type: str | None = None) -> dict:
	raw_response = raw_response or ""
	if model_type == "gpt_oss":
		return split_harmony_response(raw_response)
	match = THINK_PATTERN.search(raw_response)
	if match:
		return {
		 "raw_response": raw_response,
		 "think": match.group(1).strip(),
		 "response": match.group(2).strip(),
		}

	closing_tag = "</think>"
	closing_idx = raw_response.lower().find(closing_tag)
	if closing_idx >= 0:
		response_start = closing_idx + len(closing_tag)
		return {
		 "raw_response": raw_response,
		 "think": raw_response[:closing_idx].strip(),
		 "response": raw_response[response_start:].strip(),
		}

	return {
	 "raw_response": raw_response,
	 "think": "",
	 "response": raw_response.strip(),
	}


def load_prompt_records(prompt_file: Path) -> tuple[list[dict], list[dict]]:
	data = load_json(prompt_file)
	return load_prompt_records_from_data(data)


def load_prompt_records_from_data(data: list[dict]) -> tuple[list[dict], list[dict]]:
	if not isinstance(data, list):
		raise ValueError("Prompt data must contain a JSON list.")
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
				if not isinstance(question_item, dict) or not isinstance(question_item.get("question"), str):
					raise ValueError(f"Question {domain_position}:{question_position} must be an object with a question.")
				records.append(
				 {
				  "position": len(records),
				  "kind": "domain",
				  "domain_position": domain_position,
				  "domain_idx": int(domain.get("domain_idx", domain_position)),
				  "question_position": question_position,
				  "question_idx": int(question_item.get("question_idx", question_position)),
				  "question": question_item["question"],
				 }
				)
		return records, results

	results = deepcopy(data)
	for position, item in enumerate(data):
		if not isinstance(item, dict):
			raise ValueError(f"Prompt item {position} must be an object.")
		record = {
		 "position": position,
		 "kind": "flat",
		 "idx": int(item.get("idx", position)),
		}
		if isinstance(item.get("messages"), list):
			record["messages"] = item["messages"]
			record["question"] = item.get("question", "")
		elif isinstance(item.get("question"), str):
			record["question"] = item["question"]
		else:
			raise ValueError(f"Prompt item {position} must contain either question or messages.")
		for key in ("sample_idx", "sample_seed"):
			if key in item:
				record[key] = item[key]
		records.append(record)
	return records, results


def expand_flat_prompts(prompt_results: list[dict], num_samples: int, seed: int | None) -> list[dict]:
	if num_samples < 1:
		raise ValueError("--num_samples must be >= 1.")
	if num_samples == 1:
		return prompt_results
	if prompt_results and isinstance(prompt_results[0], dict) and isinstance(prompt_results[0].get("questions"), list):
		raise ValueError("--num_samples supports flat prompt JSON only.")

	base_seed = 42 if seed is None else seed
	expanded = []
	for item in prompt_results:
		for sample_idx in range(num_samples):
			sample_item = deepcopy(item)
			sample_item["sample_idx"] = sample_idx
			sample_item["sample_seed"] = base_seed + sample_idx
			expanded.append(sample_item)
	return expanded


def attach_sample(results: list[dict], record: dict, sample: dict) -> None:
	payload = {
	 "raw_response": sample["raw_response"],
	 "think": sample["think"],
	 "response": sample["response"],
	}
	if "system_prompt" in sample:
		payload["system_prompt"] = sample["system_prompt"]
	if record["kind"] == "domain":
		results[record["domain_position"]]["questions"][record["question_position"]]["sample"] = payload
	else:
		results[record["position"]]["sample"] = payload


def sample_for_record(results: list[dict], record: dict):
	if record["kind"] == "domain":
		value = results[record["domain_position"]]["questions"][record["question_position"]].get("sample")
	else:
		value = results[record["position"]].get("sample")
	if isinstance(value, dict) and isinstance(value.get("response"), str):
		return value
	return None


def apply_partial_outputs(results: list[dict], records: list[dict], partial_outputs: dict) -> None:
	if not isinstance(partial_outputs, dict):
		return
	record_by_position = {record["position"]: record for record in records}
	for key, value in partial_outputs.items():
		try:
			position = int(key)
		except ValueError:
			continue
		record = record_by_position.get(position)
		if record is None or not isinstance(value, dict) or not isinstance(value.get("response"), str):
			continue
		attach_sample(results, record, value)


def load_results(save_path: Path, prompt_results: list[dict], records: list[dict]) -> tuple[list[dict], bool]:
	results = deepcopy(prompt_results)
	loaded = load_json_if_exists(save_path)
	if loaded is None:
		return results, True
	if not isinstance(loaded, list):
		raise ValueError(f"Response file must contain a JSON list: {save_path}")
	if len(loaded) != len(results):
		return results, True
	results = loaded
	return results, False


def pending_records(records: list[dict], results: list[dict]) -> list[dict]:
	return [record for record in records if sample_for_record(results, record) is None]


def wait_for_worker_ready(proc: Process, startup_queue, startup_timeout: int, rank: int, gpu_id: int, local_worker_idx: int) -> None:
	for _ in range(startup_timeout):
		if proc.exitcode is not None:
			raise RuntimeError(
			 f"Worker rank={rank} gpu={gpu_id} local_worker={local_worker_idx} "
			 f"exited before startup completed (exit_code={proc.exitcode})."
			)
		try:
			msg = startup_queue.get(timeout=1)
		except Empty:
			continue
		if msg.get("status") == "ready":
			return
		raise RuntimeError(
		 f"Worker rank={rank} gpu={gpu_id} local_worker={local_worker_idx} "
		 f"failed during vLLM initialization: {msg.get('error', 'unknown error')}"
		)
	raise TimeoutError(f"Worker rank={rank} gpu={gpu_id} local_worker={local_worker_idx} did not report ready.")


def wait_for_worker_completion(procs: list[Process], completion_queue, worker_specs: list[tuple[int, int]]) -> None:
	pending_ranks = set(range(len(procs)))
	while pending_ranks:
		failed = [rank for rank, proc in enumerate(procs) if rank in pending_ranks and proc.exitcode not in (None, 0)]
		if failed:
			rank = failed[0]
			gpu_id, local_worker_idx = worker_specs[rank]
			raise RuntimeError(
			 f"Worker rank={rank} gpu={gpu_id} local_worker={local_worker_idx} "
			 f"exited unexpectedly (exit_code={procs[rank].exitcode})."
			)
		try:
			msg = completion_queue.get(timeout=5)
		except Empty:
			continue
		if msg.get("status") == "done":
			pending_ranks.discard(msg["rank"])


def worker(
 gpu_id: int,
 rank: int,
 world: int,
 local_worker_idx: int,
 model_name: str,
 records: list[dict],
 results: list[dict],
 part_path: str,
 temperature: float,
 max_new_tokens: int,
 top_p: float,
 top_k: int,
 presence_penalty: float,
 save_every: int,
 tqdm_position: int,
 gpu_memory_utilization: float,
 dtype: str,
 max_model_len: int | None,
 vllm_port: int | None,
 enable_thinking: bool,
 system_prompt: str,
 seed: int | None,
 reasoning_effort: str = "low",
 startup_queue=None,
 completion_queue=None,
 start_event=None,
 shutdown_event=None,
) -> None:
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
	if vllm_port is not None:
		os.environ["VLLM_PORT"] = str(vllm_port)
	from behavioral_coupling.utils.llm import VLLM

	pending = pending_records(records, results)
	indices = shard_indices([record["position"] for record in pending], rank, world)
	record_by_position = {record["position"]: record for record in records}
	partial_out = load_json_if_exists(part_path)
	if not isinstance(partial_out, dict):
		partial_out = {}

	pbar = tqdm(indices, total=len(indices), desc=f"GPU{gpu_id}-W{local_worker_idx}", position=tqdm_position, leave=True)
	llm = None
	startup_complete = False
	worker_completed = False
	completion_reported = False
	try:
		llm = VLLM(
		 model_name=model_name,
		 dtype=dtype,
		 gpu_memory_utilization=gpu_memory_utilization,
		 max_model_len=max_model_len,
		 reasoning_effort=reasoning_effort,
		)
		startup_complete = True
		if startup_queue is not None:
			startup_queue.put({"status": "ready", "rank": rank, "gpu_id": gpu_id, "local_worker_idx": local_worker_idx})
		if start_event is not None:
			start_event.wait()

		for count, position in enumerate(pbar, start=1):
			record = record_by_position[position]
			if isinstance(record.get("messages"), list):
				messages = deepcopy(record["messages"])
				if system_prompt:
					messages = [{"role": "system", "content": system_prompt}] + messages
			else:
				messages = [
				 {"role": "system", "content": system_prompt},
				 {"role": "user", "content": record["question"]},
				]
			raw_response = llm.generate(
			 messages=messages,
			 chat=True,
			 do_sample=True,
			 temperature=temperature,
			 max_new_tokens=max_new_tokens,
			 top_p=top_p,
			 top_k=top_k,
			 presence_penalty=presence_penalty,
			 enable_thinking=enable_thinking,
			 seed=record.get("sample_seed", seed),
			)
			sample = split_think_response(raw_response, model_type=llm.model_type)
			if llm.is_gpt_oss:
				sample["model_type"] = llm.model_type
				sample["reasoning_effort"] = reasoning_effort
			sample["system_prompt"] = system_prompt
			partial_out[str(position)] = sample
			if save_every and count % save_every == 0:
				atomic_write_json(partial_out, part_path)
				pbar.set_postfix_str(f"saved {len(partial_out)} items")
		worker_completed = True
		atomic_write_json(partial_out, part_path)
		if completion_queue is not None:
			completion_queue.put({"status": "done", "rank": rank})
			completion_reported = True
		if shutdown_event is not None:
			shutdown_event.wait()
	except Exception as exc:
		if startup_queue is not None and not startup_complete:
			startup_queue.put(
			 {
			  "status": "error",
			  "rank": rank,
			  "gpu_id": gpu_id,
			  "local_worker_idx": local_worker_idx,
			  "error": repr(exc),
			 }
			)
		raise
	finally:
		atomic_write_json(partial_out, part_path)
		if completion_queue is not None and worker_completed and not completion_reported:
			completion_queue.put({"status": "done", "rank": rank})
		if llm is not None:
			try:
				llm.shutdown()
			except Exception as exc:
				print(f"Warning: failed to shutdown worker rank={rank}: {exc}", flush=True)


def merge_parts(results: list[dict], records: list[dict], part_paths: list[str], save_path: Path) -> None:
	for part_path in part_paths:
		apply_partial_outputs(results, records, load_json_if_exists(part_path))
	atomic_write_json(results, save_path)
	for part_path in part_paths:
		try:
			os.remove(part_path)
		except FileNotFoundError:
			pass


def count_completed(records: list[dict], results: list[dict]) -> int:
	return sum(1 for record in records if sample_for_record(results, record) is not None)


def parse_max_model_len(value: str) -> int | None:
	if value.lower() in {"none", "null"}:
		return None
	return int(value)


def resolve_system_prompt(system_prompt: str, system_prompt_file: Path | None) -> str:
	if system_prompt_file is None:
		return system_prompt
	if system_prompt:
		raise ValueError("Use either --system_prompt or --system_prompt_file, not both.")
	return system_prompt_file.read_text(encoding="utf-8")


def main(
 prompt_file: Path,
 model_name: str,
 model_tag: str,
 output_dir: Path,
 output_file: Path | None,
 gpus: List[int],
 temperature: float,
 max_new_tokens: int,
 top_p: float,
 top_k: int,
 presence_penalty: float,
 save_every: int,
 llms_per_gpu: int,
 gpu_memory_budget: float,
 dtype: str,
 max_model_len: int | None,
 startup_timeout: int,
 vllm_port_base: int | None,
 enable_thinking: bool,
 system_prompt: str,
 seed: int | None,
 num_samples: int,
 reasoning_effort: str = "low",
) -> None:
	if save_every < 1:
		raise ValueError("--save_every must be >= 1.")
	if llms_per_gpu < 1:
		raise ValueError("--llms_per_gpu must be >= 1.")
	if not (0 < gpu_memory_budget <= 1):
		raise ValueError("--gpu_memory_budget must be in the range (0, 1].")
	if not gpus:
		raise ValueError("No GPUs provided. Use --gpus like '0,1'.")

	records, prompt_results = load_prompt_records(prompt_file)
	prompt_results = expand_flat_prompts(prompt_results, num_samples=num_samples, seed=seed)
	records, prompt_results = load_prompt_records_from_data(prompt_results)
	save_path = output_file or output_path(prompt_file, model_tag, output_dir)
	results, should_save_initial_results = load_results(save_path, prompt_results, records)
	pending = pending_records(records, results)
	print(f"Loaded {len(records)} prompts from {prompt_file}")
	print(f"Loaded {len(records) - len(pending)} completed responses; {len(pending)} remaining.")
	print(f"Save path: {save_path}")

	if should_save_initial_results:
		atomic_write_json(results, save_path)
	if not pending:
		print("Nothing to do.")
		return

	gpu_memory_utilization = gpu_memory_budget / llms_per_gpu
	if gpu_memory_utilization <= 0 or gpu_memory_utilization > 1:
		raise ValueError("Per-worker gpu_memory_utilization must be in the range (0, 1].")

	worker_specs = [(gpu_id, local_worker_idx) for gpu_id in gpus for local_worker_idx in range(llms_per_gpu)]
	part_paths = [
	 str(save_path) + f".gpu{gpu_id}.worker{local_worker_idx}.part.json"
	 for gpu_id, local_worker_idx in worker_specs
	]

	try:
		set_start_method("spawn")
	except RuntimeError:
		pass

	world = len(worker_specs)
	vllm_ports = allocate_vllm_ports(world, vllm_port_base)
	print(f"Using vLLM ports: {','.join(str(port) for port in vllm_ports)}", flush=True)
	procs: list[Process | None] = [None] * world
	completion_queue = Queue()
	start_event = Event()
	shutdown_event = Event()

	def merge_available_parts() -> None:
		merge_parts(results, records, part_paths, save_path)
		final_results = load_json(save_path)
		print(f"Saved {count_completed(records, final_results)}/{len(records)} responses to {save_path}", flush=True)

	try:
		for current_local_worker_idx in range(llms_per_gpu):
			startup_batch = []
			for rank, (gpu_id, local_worker_idx) in enumerate(worker_specs):
				if local_worker_idx != current_local_worker_idx:
					continue
				startup_queue = Queue()
				proc = Process(
				 target=worker,
				 kwargs=dict(
				  gpu_id=gpu_id,
				  rank=rank,
				  world=world,
				  local_worker_idx=local_worker_idx,
				  model_name=model_name,
				  records=deepcopy(records),
				  results=deepcopy(results),
				  part_path=part_paths[rank],
				  temperature=temperature,
				  max_new_tokens=max_new_tokens,
				  top_p=top_p,
				  top_k=top_k,
				  presence_penalty=presence_penalty,
				  save_every=save_every,
				  tqdm_position=rank,
				  gpu_memory_utilization=gpu_memory_utilization,
				  dtype=dtype,
				  max_model_len=max_model_len,
				  vllm_port=vllm_ports[rank],
				  enable_thinking=enable_thinking,
				  system_prompt=system_prompt,
				  seed=seed,
				  reasoning_effort=reasoning_effort,
				  startup_queue=startup_queue,
				  completion_queue=completion_queue,
				  start_event=start_event,
				  shutdown_event=shutdown_event,
				 ),
				 daemon=False,
				)
				proc.start()
				procs[rank] = proc
				startup_batch.append((rank, gpu_id, local_worker_idx, proc, startup_queue))

			for rank, gpu_id, local_worker_idx, proc, startup_queue in startup_batch:
				wait_for_worker_ready(proc, startup_queue, startup_timeout, rank, gpu_id, local_worker_idx)
	except BaseException:
		print("Interrupted or failed during startup. Merging partial outputs before exit...", flush=True)
		cleanup_processes([proc for proc in procs if proc is not None])
		merge_available_parts()
		raise

	ordered_procs = [proc for proc in procs if proc is not None]
	if len(ordered_procs) != world:
		raise RuntimeError("Failed to start all workers.")

	print("All vLLM workers are ready; starting generation.", flush=True)
	start_event.set()

	try:
		wait_for_worker_completion(ordered_procs, completion_queue, worker_specs)
		shutdown_event.set()
		cleanup_processes(ordered_procs)
	except BaseException:
		print("Interrupted. Merging partial outputs before exit...", flush=True)
		shutdown_event.set()
		cleanup_processes(ordered_procs)
		merge_available_parts()
		raise

	merge_available_parts()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Sample model responses for behavioral_coupling evaluation prompts.")
	parser.add_argument("--prompt_file", type=Path, required=True)
	parser.add_argument("--model_name", type=str, required=True)
	parser.add_argument("--model_tag", type=str, default=None)
	parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
	parser.add_argument("--output_file", type=Path, default=None)
	parser.add_argument("--gpus", type=str, required=True, help="comma-separated GPU ids, e.g. '0,1'")
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--top_p", type=float, default=1.0)
	parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling. 0 disables top-k.")
	parser.add_argument("--presence_penalty", type=float, default=0.0)
	parser.add_argument("--max_new_tokens", type=int, default=1024)
	parser.add_argument("--save_every", type=int, default=50)
	parser.add_argument("--llms_per_gpu", type=int, default=1)
	parser.add_argument("--gpu_memory_budget", type=float, default=0.95)
	parser.add_argument("--dtype", type=str, default="bfloat16", help="vLLM model dtype, e.g. bfloat16, float16, auto.")
	parser.add_argument("--max_model_len", type=parse_max_model_len, default=8192)
	parser.add_argument("--startup_timeout", type=int, default=600)
	parser.add_argument("--vllm_port_base", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None, help="Base sampling seed. With --num_samples N, each prompt uses seed through seed + N - 1.")
	parser.add_argument("--system_prompt", type=str, default="", help="Optional system prompt used for every sampled question.")
	parser.add_argument("--system_prompt_file", type=Path, default=None, help="Optional file containing the system prompt.")
	parser.add_argument("--num_samples", type=int, default=1, help="Independent samples per prompt, using seed through seed + num_samples - 1.")
	parser.set_defaults(enable_thinking=True)
	parser.add_argument("--enable_thinking", dest="enable_thinking", action="store_true")
	parser.add_argument("--disable_thinking", dest="enable_thinking", action="store_false")
	parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="low", help="GPT-oss Harmony reasoning effort; enable_thinking only controls Qwen templates.")
	args = parser.parse_args()

	main(
	 prompt_file=args.prompt_file,
	 model_name=args.model_name,
	 model_tag=args.model_tag or args.model_name,
	 output_dir=args.output_dir,
	 output_file=args.output_file,
	 gpus=parse_gpus(args.gpus),
	 temperature=args.temperature,
	 max_new_tokens=args.max_new_tokens,
	 top_p=args.top_p,
	 top_k=args.top_k,
	 presence_penalty=args.presence_penalty,
	 save_every=args.save_every,
	 llms_per_gpu=args.llms_per_gpu,
	 gpu_memory_budget=args.gpu_memory_budget,
	 dtype=args.dtype,
	 max_model_len=args.max_model_len,
	 startup_timeout=args.startup_timeout,
	 vllm_port_base=args.vllm_port_base,
	 enable_thinking=args.enable_thinking,
	 system_prompt=resolve_system_prompt(args.system_prompt, args.system_prompt_file),
	 seed=args.seed,
	 num_samples=args.num_samples,
	 reasoning_effort=args.reasoning_effort,
	)
