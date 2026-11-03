from __future__ import annotations

import argparse
import inspect
import gc
import json
import os
import shutil
import socket
import sys
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
	torch.backends.cuda.enable_cudnn_sdp(False)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
 AutoConfig,
 AutoModelForCausalLM,
 AutoModelForMultimodalLM,
 AutoTokenizer,
 TrainerCallback,
 set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from behavioral_coupling.env import coupling_home
from behavioral_coupling.utils.gpt_oss import (
 is_gpt_oss_config,
 loaded_with_mxfp4_dequantize,
 mxfp4_training_kwargs,
 save_pretrained_in_memory_format,
)
from behavioral_coupling.utils.precision import (
 is_qwen35_config,
 load_qwen35_pretrained,
 log_parameter_dtypes,
 log_saved_parameter_dtypes,
 write_parameter_dtype_manifest,
)

DEFAULT_MASTER_WEIGHT_DTYPE = "fp32"
DEFAULT_MIXED_PRECISION_DTYPE = "none"


def write_json(path: Path, payload) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def resolve_resume_checkpoint(resume_from_checkpoint, output_dir: Path) -> str | None:
	if not resume_from_checkpoint:
		return None
	if resume_from_checkpoint is True or resume_from_checkpoint == "last":
		checkpoint = get_last_checkpoint(str(output_dir))
		if checkpoint is None:
			raise ValueError(f"No checkpoint found under output_dir={output_dir}")
		return checkpoint
	return os.fspath(resume_from_checkpoint)


def build_model_kwargs(master_weight_dtype: str) -> dict:
	if master_weight_dtype == "bf16":
		return {"torch_dtype": torch.bfloat16}
	if master_weight_dtype == "fp32":
		return {}
	raise ValueError(f"Unsupported master_weight_dtype: {master_weight_dtype}")


def freeze_qwen35_vision_tower(model, config) -> None:
	vision_tower = getattr(getattr(model, "model", None), "visual", None)
	if vision_tower is not None:
		vision_tower.requires_grad_(False)
		print(
		 f"Loaded {config.model_type} with its full multimodal config; "
		 "the vision tower is frozen for text-only SFT.",
		 flush=True,
		)


def load_sft_model(model_name: str, master_weight_dtype: str, device_map_auto: bool = False):
	config = AutoConfig.from_pretrained(model_name)
	model_kwargs = build_model_kwargs(master_weight_dtype)
	if device_map_auto:
		model_kwargs["device_map"] = "auto"

	if is_qwen35_config(config):
		model = load_qwen35_pretrained(AutoModelForMultimodalLM, model_name, **model_kwargs)
		freeze_qwen35_vision_tower(model, config)
		return model
	model_kwargs.update(mxfp4_training_kwargs(config))
	model = AutoModelForCausalLM.from_pretrained(
	 model_name,
	 **model_kwargs,
	)
	return model


def configure_text_model(model, pad_token_id: int, gradient_checkpointing: bool) -> None:
	model.config.pad_token_id = pad_token_id
	text_config = getattr(model.config, "text_config", model.config)
	text_config.pad_token_id = pad_token_id
	if gradient_checkpointing:
		model.config.use_cache = False
		text_config.use_cache = False


def parse_lora_target_modules(value: str) -> list[str]:
	modules = [item.strip() for item in value.split(",") if item.strip()]
	if not modules:
		raise ValueError("--lora_target_modules must contain at least one module name when --use_lora is set.")
	return modules


def parse_positive_steps(value: str | None) -> list[int]:
	if value is None or not value.strip():
		return []
	steps = []
	for item in value.split(","):
		item = item.strip()
		if not item:
			continue
		try:
			step = int(item)
		except ValueError as exc:
			raise ValueError(
			 "--lora_snapshot_steps must be a comma-separated list of integers."
			) from exc
		if step <= 0:
			raise ValueError("--lora_snapshot_steps values must be positive.")
		steps.append(step)
	return sorted(set(steps))


class SaveLoraAtStepsCallback(TrainerCallback):
	def __init__(self, steps: list[int]):
		self.steps = frozenset(steps)

	def on_step_end(self, args, state, control, **kwargs):
		if state.global_step in self.steps:
			control.should_save = True
		return control


class LogQwen35PrecisionCallback(TrainerCallback):
	def on_train_begin(self, args, state, control, model=None, **kwargs):
		if model is None or not getattr(state, "is_world_process_zero", True):
			return control
		unwrapped = getattr(model, "module", model)
		if is_qwen35_config(getattr(unwrapped, "config", None)):
			log_parameter_dtypes(unwrapped, "Qwen3.5 after DeepSpeed wrap")
		return control


class GptOssSFTTrainer(SFTTrainer):
	def _save(self, output_dir=None, state_dict=None):
		model = self.accelerator.unwrap_model(self.model)
		if isinstance(model, PeftModel) or not loaded_with_mxfp4_dequantize(model):
			return super()._save(output_dir, state_dict)
		output_dir = Path(output_dir or self.args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		save_pretrained_in_memory_format(model, output_dir, state_dict)
		if self.processing_class is not None:
			self.processing_class.save_pretrained(str(output_dir))
		torch.save(self.args, output_dir / "training_args.bin")


def apply_lora_if_enabled(model, args: argparse.Namespace):
	if not args.use_lora:
		return model
	config = LoraConfig(
	 task_type=TaskType.CAUSAL_LM,
	 r=args.lora_r,
	 lora_alpha=args.lora_alpha,
	 lora_dropout=args.lora_dropout,
	 target_modules=parse_lora_target_modules(args.lora_target_modules),
	 bias=args.lora_bias,
	)
	model = get_peft_model(model, config)
	model.print_trainable_parameters()
	return model


def final_output_dir(output_model_name: str) -> Path:
	return coupling_home / "models" / "sft" / output_model_name


def reset_output_dir(path: Path) -> None:
	if path.is_symlink():
		target = path.resolve()
		if target.exists():
			shutil.rmtree(target)
		target.mkdir(parents=True, exist_ok=True)
		return
	if path.exists():
		shutil.rmtree(path)
	path.mkdir(parents=True, exist_ok=True)


def adapter_output_name(args: argparse.Namespace) -> str:
	return args.lora_adapter_output_model_name or f"{args.output_model_name}-adapter-tmp"


def wait_for_distributed_workers() -> None:
	if torch.distributed.is_available() and torch.distributed.is_initialized():
		torch.distributed.barrier()


def cleanup_deepspeed_global_step_dirs(output_dir: Path) -> None:
	patterns = ("global_step*", "checkpoint-*/global_step*")
	for pattern in patterns:
		for global_step_dir in output_dir.glob(pattern):
			if global_step_dir.is_dir():
				shutil.rmtree(global_step_dir)
				print(f"Removed DeepSpeed state directory {global_step_dir}.", flush=True)


def patch_peft_tensor_parallel_compat() -> None:
	try:
		import transformers.integrations.tensor_parallel as tensor_parallel
	except Exception:
		return
	if hasattr(tensor_parallel, "EmbeddingParallel"):
		return
	tensor_parallel.EmbeddingParallel = tensor_parallel.ColwiseParallel


def cast_model_for_save(model, save_dtype: str):
	if is_qwen35_config(model.config):
		print(
		 "Preserving Qwen3.5 mixed parameter dtypes on save; "
		 f"ignoring merged_save_dtype={save_dtype}.",
		 flush=True,
		)
		return model
	if save_dtype == "fp32":
		return model
	if save_dtype == "bf16":
		return model.to(torch.bfloat16)
	raise ValueError(f"Unsupported merged_save_dtype: {save_dtype}")


def merge_lora_adapter(
 adapter_dir: Path,
 merged_dir: Path,
 tokenizer,
 args: argparse.Namespace,
 step: int | None = None,
) -> None:
	print(f"Merging LoRA adapter from {adapter_dir} into base model {args.model_name}.", flush=True)
	if merged_dir.exists() and args.overwrite_merged_lora:
		reset_output_dir(merged_dir)
	merged_dir.mkdir(parents=True, exist_ok=True)
	patch_peft_tensor_parallel_compat()
	base_model = load_sft_model(args.model_name, args.master_weight_dtype)
	peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
	merged_model = peft_model.merge_and_unload()
	merged_model = cast_model_for_save(merged_model, args.merged_save_dtype)
	if is_qwen35_config(merged_model.config):
		log_parameter_dtypes(merged_model, "Merged Qwen3.5 LoRA")
		write_parameter_dtype_manifest(merged_model, merged_dir / "parameter_dtype_manifest.json")
	save_pretrained_in_memory_format(merged_model, merged_dir)
	if is_qwen35_config(merged_model.config):
		log_saved_parameter_dtypes(merged_dir, "Merged Qwen3.5 LoRA safetensors")
	tokenizer.save_pretrained(str(merged_dir))
	write_json(
	 merged_dir / "lora_merge_config.json",
	 {
	  "base_model_name": args.model_name,
	  "adapter_dir": str(adapter_dir),
	  "merged_output_dir": str(merged_dir),
	  "global_step": step,
	  "lora_r": args.lora_r,
	  "lora_alpha": args.lora_alpha,
	  "lora_dropout": args.lora_dropout,
	  "lora_target_modules": parse_lora_target_modules(args.lora_target_modules),
	  "lora_bias": args.lora_bias,
	  "merged_save_dtype": args.merged_save_dtype,
	 },
	)
	print(f"Saved merged LoRA model to {merged_dir}.", flush=True)
	del merged_model
	del peft_model
	del base_model
	gc.collect()


def save_lora_and_maybe_merge(trainer, tokenizer, args: argparse.Namespace) -> None:
	adapter_dir = final_output_dir(adapter_output_name(args))
	merged_dir = final_output_dir(args.output_model_name)
	snapshot_steps = parse_positive_steps(args.lora_snapshot_steps)
	trainer.save_model(str(adapter_dir))
	if trainer.is_world_process_zero():
		tokenizer.save_pretrained(str(adapter_dir))
	wait_for_distributed_workers()
	if not args.merge_lora_on_save:
		return

	if trainer.is_world_process_zero():
		trainer.model.to("cpu")
		del trainer.model
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		for step in snapshot_steps:
			snapshot_adapter_dir = adapter_dir / f"checkpoint-{step}"
			if not snapshot_adapter_dir.exists():
				print(
				 f"Skipping unreachable LoRA snapshot step {step}: "
				 f"{snapshot_adapter_dir} was not created.",
				 flush=True,
				)
				continue
			snapshot_merged_dir = final_output_dir(
			 f"{args.output_model_name}-step-{step}"
			)
			merge_lora_adapter(
			 snapshot_adapter_dir,
			 snapshot_merged_dir,
			 tokenizer,
			 args,
			 step,
			)
		merge_lora_adapter(adapter_dir, merged_dir, tokenizer, args)
		if adapter_dir != merged_dir and adapter_dir.exists():
			reset_output_dir(adapter_dir)
			print(f"Removed temporary LoRA adapter directory {adapter_dir}.", flush=True)
	wait_for_distributed_workers()


def uses_bf16_mixed_precision(mixed_precision_dtype: str) -> bool:
	if mixed_precision_dtype in {"none", "fp32"}:
		return False
	if mixed_precision_dtype == "bf16":
		return True
	raise ValueError(f"Unsupported mixed_precision_dtype: {mixed_precision_dtype}")


def port_has_listener(port: int) -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.settimeout(0.2)
		return sock.connect_ex(("127.0.0.1", port)) == 0


def configure_master_port() -> None:
	start_port = int(os.environ.get("MASTER_PORT", "29500"))
	port = start_port
	while port < 65535 and port_has_listener(port):
		port += 1
	if port >= 65535:
		raise RuntimeError(f"No free MASTER_PORT found starting from {start_port}")
	os.environ["MASTER_PORT"] = str(port)


def disable_thinking_chat_template(example: dict) -> dict:
	example["chat_template_kwargs"] = {"enable_thinking": False}
	return example


def gpt_oss_chat_template(example: dict, reasoning_effort: str) -> dict:
	example["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
	return example


def sft_config_kwargs(args: argparse.Namespace, dataset_format: str) -> dict:
	output_model_name = adapter_output_name(args) if args.use_lora else args.output_model_name
	model_path = Path(args.output_model_name)
	output_name = model_path.name
	if output_name in {"npo-model", "act-model"}:
		output_name = f"{model_path.parent.name}-{output_name}"
	run_name = os.environ.get("WANDB_NAME") or output_name
	os.environ["WANDB_NAME"] = run_name
	kwargs = {
	 "output_dir": str(final_output_dir(output_model_name)),
	 "run_name": run_name,
	 "num_train_epochs": args.num_train_epochs,
	 "learning_rate": args.learning_rate,
	 "weight_decay": args.weight_decay,
	 "max_grad_norm": args.max_grad_norm,
	 "lr_scheduler_type": args.lr_scheduler_type,
	 "per_device_train_batch_size": args.per_device_train_batch_size,
	 "gradient_accumulation_steps": args.gradient_accumulation_steps,
	 "seed": args.seed,
	 "data_seed": args.data_seed,
	 "save_strategy": (
	  "no"
	  if parse_positive_steps(args.lora_snapshot_steps)
	  else args.save_strategy
	 ),
	 "save_steps": args.save_steps,
	 "logging_steps": args.logging_steps,
	 "report_to": [item.strip() for item in args.report_to.split(",") if item.strip()],
	 "bf16": uses_bf16_mixed_precision(args.mixed_precision_dtype),
	 "deepspeed": args.deepspeed,
	 "gradient_checkpointing": args.gradient_checkpointing,
	 "gradient_checkpointing_kwargs": {"use_reentrant": False} if args.gradient_checkpointing else None,
	 "local_rank": args.local_rank,
	 "remove_unused_columns": False,
	}
	if args.ddp_timeout is not None:
		kwargs["ddp_timeout"] = args.ddp_timeout
	if args.warmup_steps is not None:
		kwargs["warmup_steps"] = args.warmup_steps
	else:
		kwargs["warmup_ratio"] = args.warmup_ratio

	signature = inspect.signature(SFTConfig.__init__).parameters
	if "max_length" in signature:
		kwargs["max_length"] = args.max_length
	elif "max_seq_length" in signature:
		kwargs["max_seq_length"] = args.max_length
	else:
		print("Warning: SFTConfig supports neither max_length nor max_seq_length.", flush=True)

	if "assistant_only_loss" in signature and dataset_format == "messages":
		kwargs["assistant_only_loss"] = True

	if "dataset_kwargs" in signature:
		kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

	if "save_only_model" in signature:
		kwargs["save_only_model"] = args.save_only_model

	return kwargs


def render_chat(
 tokenizer,
 messages: list[dict],
 add_generation_prompt: bool = False,
 chat_template_kwargs: dict | None = None,
) -> str:
	return tokenizer.apply_chat_template(
	 messages,
	 tokenize=False,
	 add_generation_prompt=add_generation_prompt,
	 **(chat_template_kwargs or {}),
	)


def tokenize_chat(
 tokenizer,
 messages: list[dict],
 add_generation_prompt: bool = False,
 chat_template_kwargs: dict | None = None,
) -> list[int]:
	text = render_chat(tokenizer, messages, add_generation_prompt, chat_template_kwargs)
	return tokenizer(text, add_special_tokens=False)["input_ids"]


def mask_think_content_keep_close(
 labels: list[int],
 tokenizer,
 full_text: str,
 search_start: int,
) -> None:
	open_tag = "<think>"
	close_tag = "</think>"
	open_idx = full_text.find(open_tag, search_start)
	close_idx = full_text.find(close_tag, search_start)
	if open_idx < 0 or close_idx < 0 or close_idx < open_idx:
		return

	mask_start = len(tokenizer(full_text[: open_idx + len(open_tag)], add_special_tokens=False)["input_ids"])
	mask_end = len(tokenizer(full_text[:close_idx], add_special_tokens=False)["input_ids"])
	mask_start = min(mask_start, len(labels))
	mask_end = min(mask_end, len(labels))
	for idx in range(mask_start, mask_end):
		labels[idx] = -100


def build_last_response_collator(tokenizer, max_length: int | None, assistant_loss_mode: str):
	def collate(examples: list[dict]) -> dict:
		input_ids_list = []
		labels_list = []
		for example in examples:
			messages = example.get("messages")
			if not isinstance(messages, list) or len(messages) < 2:
				raise ValueError("Each SFT example must contain a messages list with at least user and assistant turns.")
			if messages[-1].get("role") != "assistant":
				raise ValueError("The last message must be an assistant response for last-response SFT.")

			chat_template_kwargs = example.get("chat_template_kwargs")
			if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
				raise ValueError("chat_template_kwargs must be a dictionary when provided.")

			prefix_text = render_chat(tokenizer, messages[:-1], True, chat_template_kwargs)
			prompt_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
			full_text = render_chat(tokenizer, messages, False, chat_template_kwargs)
			full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
			if max_length is not None:
				full_ids = full_ids[:max_length]
			prompt_len = min(len(prompt_ids), len(full_ids))
			labels = list(full_ids)
			labels[:prompt_len] = [-100] * prompt_len
			if assistant_loss_mode == "mask_think_content_keep_close":
				mask_think_content_keep_close(labels, tokenizer, full_text, len(prefix_text))
			input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
			labels_list.append(torch.tensor(labels, dtype=torch.long))

		pad_token_id = tokenizer.pad_token_id
		input_ids = torch.nn.utils.rnn.pad_sequence(
		 input_ids_list,
		 batch_first=True,
		 padding_value=pad_token_id,
		)
		batch = {
		 "input_ids": input_ids,
		 "attention_mask": input_ids.ne(pad_token_id).long(),
		}
		max_len = input_ids.shape[1]
		padded_labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
		for idx, labels in enumerate(labels_list):
			padded_labels[idx, : labels.shape[0]] = labels
		batch["labels"] = padded_labels
		return batch

	return collate


def build_full_text_collator(tokenizer, max_length: int | None):
	def collate(examples: list[dict]) -> dict:
		input_ids_list = []
		for example in examples:
			text = example.get("text")
			if not isinstance(text, str) or not text.strip():
				raise ValueError("Each pretraining-style SFT example must contain a non-empty text field.")
			encoded = tokenizer(text, add_special_tokens=True)["input_ids"]
			if max_length is not None:
				encoded = encoded[:max_length]
			input_ids_list.append(torch.tensor(encoded, dtype=torch.long))

		pad_token_id = tokenizer.pad_token_id
		input_ids = torch.nn.utils.rnn.pad_sequence(
		 input_ids_list,
		 batch_first=True,
		 padding_value=pad_token_id,
		)
		attention_mask = input_ids.ne(pad_token_id).long()
		labels = input_ids.clone()
		labels[attention_mask.eq(0)] = -100
		return {
		 "input_ids": input_ids,
		 "attention_mask": attention_mask,
		 "labels": labels,
		}

	return collate


def select_collator(dataset, tokenizer, max_length: int | None, assistant_loss_mode: str):
	if len(dataset) == 0:
		raise ValueError("Training dataset is empty.")
	columns = set(dataset.column_names)
	if "text" in columns:
		return build_full_text_collator(tokenizer, max_length=max_length), "text"
	if "messages" in columns:
		return build_last_response_collator(tokenizer, max_length=max_length, assistant_loss_mode=assistant_loss_mode), "messages"
	raise ValueError("Dataset must contain either a text field or a messages field.")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train an SFT model for behavioral_coupling experiments.")
	parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
	parser.add_argument("--tokenizing_model", type=str, default=None)
	parser.add_argument("--data_files", type=str, required=True)
	parser.add_argument("--output_model_name", type=str, default="sft")
	parser.add_argument("--max_length", type=int, default=4096)
	parser.add_argument("--num_train_epochs", type=int, default=1)
	parser.add_argument("--learning_rate", type=float, default=1e-5)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)
	parser.add_argument("--lr_scheduler_type", type=str, default="linear")
	parser.add_argument("--warmup_ratio", type=float, default=0.0)
	parser.add_argument("--warmup_steps", type=int, default=None)
	parser.add_argument("--per_device_train_batch_size", type=int, default=2)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
	parser.add_argument("--master_weight_dtype", type=str, default=DEFAULT_MASTER_WEIGHT_DTYPE, choices=["fp32", "bf16"])
	parser.add_argument("--mixed_precision_dtype", type=str, default=DEFAULT_MIXED_PRECISION_DTYPE, choices=["none", "fp32", "bf16"])
	parser.add_argument("--assistant_loss_mode", type=str, default="last_response", choices=["last_response", "mask_think_content_keep_close"])
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--data_seed", type=int, default=42)
	parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
	parser.add_argument("--save_steps", type=float, default=500)
	parser.add_argument("--save_only_model", action="store_true")
	parser.add_argument("--logging_steps", type=float, default=10)
	parser.add_argument("--report_to", type=str, default="tensorboard")
	parser.add_argument("--resume_from_checkpoint", type=str, default=None)
	parser.add_argument("--deepspeed", type=str, default=None)
	parser.add_argument("--device_map_auto", action="store_true")
	parser.add_argument("--gradient_checkpointing", action="store_true")
	parser.add_argument("--use_lora", action="store_true")
	parser.add_argument("--merge_lora_on_save", action="store_true")
	parser.add_argument("--lora_adapter_output_model_name", type=str, default=None)
	parser.add_argument(
	 "--lora_snapshot_steps",
	 type=str,
	 default=None,
	 help=(
	  "Comma-separated optimizer/global steps at which to save LoRA "
	  "adapters. Reached snapshots are merged after training."
	 ),
	)
	parser.add_argument("--lora_r", type=int, default=16)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
	parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
	parser.add_argument("--overwrite_merged_lora", action="store_true")
	parser.add_argument("--merged_save_dtype", type=str, default="fp32", choices=["fp32", "bf16"])
	parser.set_defaults(disable_thinking=True)
	parser.add_argument("--disable_thinking", dest="disable_thinking", action="store_true")
	parser.add_argument("--enable_thinking", dest="disable_thinking", action="store_false")
	parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="low")
	parser.add_argument("--ddp_timeout", type=int, default=None)
	parser.add_argument("--local_rank", type=int, default=-1)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	configure_master_port()
	set_seed(args.seed)
	lora_snapshot_steps = parse_positive_steps(args.lora_snapshot_steps)
	if lora_snapshot_steps and not args.use_lora:
		raise ValueError("--lora_snapshot_steps requires --use_lora.")

	tokenizing_model = args.tokenizing_model or args.model_name
	training_output_model_name = adapter_output_name(args) if args.use_lora else args.output_model_name
	output_dir = final_output_dir(training_output_model_name)

	model = load_sft_model(
	 args.model_name,
	 args.master_weight_dtype,
	 device_map_auto=args.device_map_auto,
	)
	tokenizer = AutoTokenizer.from_pretrained(tokenizing_model)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	configure_text_model(model, tokenizer.pad_token_id, args.gradient_checkpointing)
	is_gpt_oss = is_gpt_oss_config(model.config)
	model = apply_lora_if_enabled(model, args)

	dataset = load_dataset("json", data_files=args.data_files, split="train")
	if is_gpt_oss:
		dataset = dataset.map(partial(gpt_oss_chat_template, reasoning_effort=args.reasoning_effort))
	elif args.disable_thinking:
		dataset = dataset.map(disable_thinking_chat_template)
	collator, dataset_format = select_collator(dataset, tokenizer, max_length=args.max_length, assistant_loss_mode=args.assistant_loss_mode)
	training_args = SFTConfig(**sft_config_kwargs(args, dataset_format))

	write_json(
	 output_dir / "run_config.json",
	 {
	  "cli_args": vars(args),
	  "final_output_dir": str(final_output_dir(args.output_model_name)),
	  "training_output_dir": str(output_dir),
	  "sft_config": training_args.to_dict(),
	  "dataset_num_rows": len(dataset),
	  "dataset_format": dataset_format,
	 },
	)

	callbacks = []
	if lora_snapshot_steps:
		callbacks.append(SaveLoraAtStepsCallback(lora_snapshot_steps))
	if is_qwen35_config(model.config):
		callbacks.append(LogQwen35PrecisionCallback())
	trainer_class = GptOssSFTTrainer if is_gpt_oss else SFTTrainer
	trainer = trainer_class(
	 model=model,
	 args=training_args,
	 train_dataset=dataset,
	 processing_class=tokenizer,
	 data_collator=collator,
	 callbacks=callbacks or None,
	)
	trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(args.resume_from_checkpoint, output_dir))
	if args.use_lora:
		save_lora_and_maybe_merge(trainer, tokenizer, args)
	else:
		trainer.save_model(str(output_dir))
		tokenizer.save_pretrained(str(output_dir))
		if trainer.is_world_process_zero():
			unwrapped = trainer.accelerator.unwrap_model(trainer.model)
			if is_qwen35_config(unwrapped.config):
				log_parameter_dtypes(unwrapped, "Saved Qwen3.5 ADD")
				write_parameter_dtype_manifest(
				 unwrapped,
				 output_dir / "parameter_dtype_manifest.json",
				)
				log_saved_parameter_dtypes(output_dir, "Saved Qwen3.5 ADD safetensors")
	if args.save_only_model:
		if trainer.is_world_process_zero():
			cleanup_deepspeed_global_step_dirs(output_dir)
		wait_for_distributed_workers()


if __name__ == "__main__":
	main()
