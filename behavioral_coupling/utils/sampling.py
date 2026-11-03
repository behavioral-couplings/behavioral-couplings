from __future__ import annotations

import os
import json
import argparse
import sys
import socket
from copy import deepcopy
from queue import Empty
from pathlib import Path
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
from tempfile import NamedTemporaryFile
from multiprocessing import Process, set_start_method, Queue
from typing import List

coupling_home = Path(os.environ["COUPLING_HOME"])
if str(coupling_home) not in sys.path:
    sys.path.insert(0, str(coupling_home))

def get_response_keys(num_responses: int) -> tuple:
    return tuple(f"response_{i+1}" for i in range(num_responses))

def has_all_responses(item: dict, resp_keys: tuple) -> bool:
    return all(k in item and item[k] not in (None, "") for k in resp_keys)

def atomic_write_json(obj, path: str):
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dirpath, exist_ok=True)
    with NamedTemporaryFile("w", dir=dirpath, delete=False, encoding="utf-8") as tf:
        json.dump(obj, tf, indent=4, ensure_ascii=False)
        tf.flush()
        os.fsync(tf.fileno())
        tmpname = tf.name
    os.replace(tmpname, path)

def load_json_if_exists(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return None
    return None

def pad_or_trim(existing: list, target_len: int):
    if existing is None:
        return None
    if not isinstance(existing, list):
        return None
    if len(existing) == target_len:
        return existing
    out = list(existing)
    if len(out) < target_len:
        out.extend({} for _ in range(target_len - len(out)))
    else:
        out = out[:target_len]
    return out

def merge_source_and_existing(source: list, existing: list, resp_keys: tuple):
    merged = []
    for i in range(len(source)):
        base = deepcopy(source[i])
        if existing and i < len(existing) and isinstance(existing[i], dict):
            prev = existing[i]
            for k in resp_keys:
                if k in prev:
                    base[k] = prev[k]
        merged.append(base)
    return merged


def parse_gpus(gpus_arg: str) -> List[int]:
    parts = [p.strip() for p in gpus_arg.split(",") if p.strip() != ""]
    return [int(p) for p in parts]

def shard_indices(indices: List[int], rank: int, world: int) -> List[int]:
    return indices[rank::world]

def _port_is_available(port: int) -> bool:
    if port < 1 or port > 65535:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return True

def allocate_vllm_ports(count: int, base: int | None = None) -> list[int]:
    if count < 1:
        return []

    ports: list[int] = []
    used: set[int] = set()

    def add_next_available(start: int) -> None:
        port = start
        while port <= 65535:
            if port not in used and _port_is_available(port):
                ports.append(port)
                used.add(port)
                return
            port += 1
        raise RuntimeError(f"Could not find an available vLLM port starting from {start}.")

    if base is not None:
        if base < 1 or base > 65535:
            raise ValueError("--vllm_port_base must be in the range [1, 65535].")
        for rank in range(count):
            start = base + rank * 100
            if start > 65535:
                raise ValueError(f"--vllm_port_base is too high for {count} workers.")
            add_next_available(start)
        return ports

    for _ in range(count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        if port in used or not _port_is_available(port):
            add_next_available(20000)
            continue
        ports.append(port)
        used.add(port)
    return ports

def save_partial(part_path: str, data_slice: dict):
    atomic_write_json(data_slice, part_path)

def apply_partial_outputs(data: list, partial_outputs: dict, resp_keys: tuple):
    if not partial_outputs:
        return

    for k, v in partial_outputs.items():
        idx = int(k)
        if idx < 0 or idx >= len(data) or not isinstance(v, dict):
            continue

        for resp_key in resp_keys:
            if resp_key in v:
                data[idx][resp_key] = v[resp_key]

def load_existing_data(source_data: list, target_path: str, resp_keys: tuple):
    existing_target = load_json_if_exists(target_path)
    existing_target = pad_or_trim(existing_target, len(source_data))
    return merge_source_and_existing(source_data, existing_target, resp_keys)

def get_pending_indices(data: list, resp_keys: tuple) -> List[int]:
    return [i for i, item in enumerate(data) if not has_all_responses(item, resp_keys)]

def extract_response_entry(item: dict, resp_keys: tuple):
    response_entry = {resp_key: item.get(resp_key) for resp_key in resp_keys}
    if all(value in (None, "") for value in response_entry.values()):
        return None
    return response_entry

def build_response_only_output(data: list, resp_keys: tuple) -> list:
    return [extract_response_entry(item, resp_keys) for item in data]

def worker(
    gpu_id: int,
    rank: int,
    world: int,
    local_worker_idx: int,
    model_name: str,
    source_data: list,
    existing_data: list,
    part_path: str,
    num_responses: int = 2,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    top_p: float = 1.0,
    save_every: int = 1,
    tqdm_position: int = 0,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int | None = 8192,
    vllm_port: int | None = None,
    startup_queue=None,
    completion_queue=None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    if vllm_port is not None:
        os.environ["VLLM_PORT"] = str(vllm_port)
    from behavioral_coupling.utils.llm import VLLM
    
    resp_keys = get_response_keys(num_responses)
    data = existing_data
    pending_indices = get_pending_indices(data, resp_keys)
    indices = shard_indices(pending_indices, rank, world)

    partial_out = load_json_if_exists(part_path)
    if not isinstance(partial_out, dict):
        partial_out = {}

    pbar = tqdm(
        indices,
        total=len(indices),
        desc=f"GPU{gpu_id}-W{local_worker_idx}",
        position=tqdm_position,
        leave=True,
    )
    llm = None
    startup_complete = False
    worker_completed = False
    try:
        llm = VLLM(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        startup_complete = True
        if startup_queue is not None:
            startup_queue.put({
                "status": "ready",
                "rank": rank,
                "gpu_id": gpu_id,
                "local_worker_idx": local_worker_idx,
            })
        for cnt, i in enumerate(pbar, start=1):
            item = data[i]
            messages = deepcopy(item["messages"])

            responses = {}
            for resp_idx in range(num_responses):
                response_key = f"response_{resp_idx+1}"
                response = llm.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens, top_p=top_p)
                responses[response_key] = response

            partial_out[i] = responses

            if save_every and (cnt % save_every == 0):
                save_partial(part_path, partial_out)
                pbar.set_postfix_str(f"saved {len(partial_out)} items")
        worker_completed = True
    except Exception as e:
        if startup_queue is not None and not startup_complete:
            startup_queue.put({
                "status": "error",
                "rank": rank,
                "gpu_id": gpu_id,
                "local_worker_idx": local_worker_idx,
                "error": repr(e),
            })
        raise
    finally:
        save_partial(part_path, partial_out)
        if completion_queue is not None and worker_completed:
            completion_queue.put({
                "status": "done",
                "rank": rank,
                "gpu_id": gpu_id,
                "local_worker_idx": local_worker_idx,
            })
        if llm is not None:
            try:
                llm.shutdown()
            except Exception as e:
                print(
                    f"Warning: failed to shutdown worker rank={rank} gpu={gpu_id} "
                    f"local_worker={local_worker_idx}: {e}",
                    flush=True,
                )

def merge_parts_into_target(source_data: list, existing_data: list, part_paths: List[str], target_path: str, resp_keys: tuple):
    merged = merge_source_and_existing(source_data, existing_data, resp_keys)

    for pp in part_paths:
        part = load_json_if_exists(pp)
        apply_partial_outputs(merged, part, resp_keys)

    atomic_write_json(build_response_only_output(merged, resp_keys), target_path)

    for pp in part_paths:
        try:
            os.remove(pp)
        except FileNotFoundError:
            pass

def wait_for_worker_ready(
    proc: Process,
    startup_queue,
    startup_timeout: int,
    rank: int,
    gpu_id: int,
    local_worker_idx: int,
):
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

    raise TimeoutError(
        f"Worker rank={rank} gpu={gpu_id} local_worker={local_worker_idx} "
        f"did not report ready within {startup_timeout} seconds."
    )

def wait_for_worker_completion(procs: list[Process], completion_queue, worker_specs: List[tuple[int, int]]):
    pending_ranks = set(range(len(procs)))
    while pending_ranks:
        failed_ranks = [
            rank for rank, proc in enumerate(procs)
            if rank in pending_ranks and proc.exitcode not in (None, 0)
        ]
        if failed_ranks:
            rank = failed_ranks[0]
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

def cleanup_processes(procs: list[Process], join_timeout: int = 30, kill_timeout: int = 5):
    for p in procs:
        p.join(timeout=join_timeout)

    alive = [p for p in procs if p.is_alive()]
    for p in alive:
        p.terminate()

    for p in alive:
        p.join(timeout=kill_timeout)

    still_alive = [p for p in alive if p.is_alive()]
    for p in still_alive:
        p.kill()

    for p in still_alive:
        p.join(timeout=kill_timeout)

def main_parallel(
    gpus: List[int], 
    model_name: str, 
    source_path: str, 
    target_path: str, 
    num_responses: int = 2,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    top_p: float = 1.0,
    save_every: int = 1,
    llms_per_gpu: int = 1,
    gpu_memory_budget: float = 0.95,
    max_model_len: int | None = 8192,
    startup_timeout: int = 600,
    vllm_port_base: int | None = None,
):
    with open(source_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)
    
    resp_keys = get_response_keys(num_responses)

    if llms_per_gpu < 1:
        raise ValueError("--llms_per_gpu must be >= 1.")
    if not (0 < gpu_memory_budget <= 1):
        raise ValueError("--gpu_memory_budget must be in the range (0, 1].")

    gpu_memory_utilization = gpu_memory_budget / llms_per_gpu
    if gpu_memory_utilization <= 0 or gpu_memory_utilization > 1:
        raise ValueError("Per-worker gpu_memory_utilization must be in the range (0, 1].")

    worker_specs = []
    for gpu_id in gpus:
        for local_worker_idx in range(llms_per_gpu):
            worker_specs.append((gpu_id, local_worker_idx))

    part_paths = [
        f"{target_path}.gpu{gpu_id}.worker{local_worker_idx}.part.json"
        for gpu_id, local_worker_idx in worker_specs
    ]

    existing_data = load_existing_data(source_data, target_path, resp_keys)

    world = len(worker_specs)
    vllm_ports = allocate_vllm_ports(world, vllm_port_base)
    print(f"Using vLLM ports: {','.join(str(port) for port in vllm_ports)}", flush=True)
    procs: list[Process | None] = [None] * world
    completion_queue = Queue()

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        for current_local_worker_idx in range(llms_per_gpu):
            startup_batch = []
            for rank, (gpu_id, local_worker_idx) in enumerate(worker_specs):
                if local_worker_idx != current_local_worker_idx:
                    continue

                startup_queue = Queue()
                p = Process(
                    target=worker,
                    kwargs=dict(
                        gpu_id=gpu_id,
                        rank=rank,
                        world=world,
                        local_worker_idx=local_worker_idx,
                        model_name=model_name,
                        source_data=source_data,
                        existing_data=existing_data,
                        part_path=part_paths[rank],
                        num_responses=num_responses,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        save_every=save_every,
                        tqdm_position=rank,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=max_model_len,
                        vllm_port=vllm_ports[rank],
                        startup_queue=startup_queue,
                        completion_queue=completion_queue,
                    ),
                    daemon=False,
                )
                p.start()
                procs[rank] = p
                startup_batch.append((rank, gpu_id, local_worker_idx, p, startup_queue))

            for rank, gpu_id, local_worker_idx, p, startup_queue in startup_batch:
                wait_for_worker_ready(
                    p,
                    startup_queue,
                    startup_timeout,
                    rank,
                    gpu_id,
                    local_worker_idx,
                )
    except BaseException:
        cleanup_processes([p for p in procs if p is not None])
        raise

    ordered_procs = [p for p in procs if p is not None]
    if len(ordered_procs) != world:
        raise RuntimeError("Failed to start all workers.")

    try:
        wait_for_worker_completion(ordered_procs, completion_queue, worker_specs)
        cleanup_processes(ordered_procs)
    except KeyboardInterrupt:
        print("Interrupted. Merging partial outputs before exit...", flush=True)
        cleanup_processes(ordered_procs)
        merge_parts_into_target(source_data, existing_data, part_paths, target_path, resp_keys)
        raise

    merge_parts_into_target(source_data, existing_data, part_paths, target_path, resp_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--num_responses", type=int, default=2,
                        help="Number of responses to generate per prompt")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--llms_per_gpu", type=int, default=1,
                        help="Number of worker processes (one vLLM instance each) to run per GPU")
    parser.add_argument("--gpu_memory_budget", type=float, default=0.95,
                        help="Fraction of each GPU to split across llms_per_gpu workers")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum sequence length to reserve in vLLM")
    parser.add_argument("--startup_timeout", type=int, default=600,
                        help="Seconds to wait for each worker to finish vLLM initialization")
    parser.add_argument("--vllm_port_base", type=int, default=None,
                        help="Base internal vLLM port. Worker rank r uses base + 100*r.")
    parser.add_argument("--gpus", type=str, required=True, help="comma-separated GPU ids, e.g. '6'")
    args = parser.parse_args()

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus like '0,1,2'.")

    main_parallel(
        gpus, 
        args.model_name, 
        args.source_path, 
        args.target_path, 
        num_responses=args.num_responses,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        save_every=args.save_every,
        llms_per_gpu=args.llms_per_gpu,
        gpu_memory_budget=args.gpu_memory_budget,
        max_model_len=args.max_model_len,
        startup_timeout=args.startup_timeout,
        vllm_port_base=args.vllm_port_base,
    )
