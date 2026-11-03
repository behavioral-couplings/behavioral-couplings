from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

FULL_QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}
QWEN35_KEEP_IN_FP32_PARAM_PATTERNS = ("A_log", "linear_attn.norm")

_DEEPSPEED_FP32_PATCHED = False


def is_qwen35_config(config: Any) -> bool:
	return getattr(config, "model_type", None) in FULL_QWEN35_MODEL_TYPES


def is_qwen35_model(model_name: str, **config_kwargs) -> bool:
	return is_qwen35_config(AutoConfig.from_pretrained(model_name, **config_kwargs))


def qwen35_keep_in_fp32_param(name: str) -> bool:
	return any(pattern in name for pattern in QWEN35_KEEP_IN_FP32_PARAM_PATTERNS)


def qwen35_fp32_param_ids(module) -> set[int]:
	return {
	 id(param)
	 for name, param in module.named_parameters()
	 if qwen35_keep_in_fp32_param(name)
	}


def apply_qwen35_fp32_storage_policy() -> None:
	_patch_transformers_keep_in_fp32()
	_patch_deepspeed_keep_fp32()


def load_qwen35_pretrained(model_class, model_name: str, **kwargs):
	apply_qwen35_fp32_storage_policy()
	model = model_class.from_pretrained(model_name, **kwargs)
	log_parameter_dtypes(model, "Qwen3.5")
	return model


def cast_floating_params_except_qwen35_fp32(module, param_dtype) -> int:
	keep_ids = qwen35_fp32_param_ids(module)
	skipped = 0
	for param in module.parameters(recurse=True):
		if id(param) in keep_ids:
			skipped += 1
			continue
		if param.is_floating_point() and param.dtype != param_dtype:
			param.data = param.data.to(param_dtype)
	return skipped


def split_optimizer_param_groups_for_fp32(optimizer, module) -> int:
	keep_ids = qwen35_fp32_param_ids(module)
	if not keep_ids:
		return 0
	new_groups = []
	fp32_groups = 0
	for group in optimizer.param_groups:
		rest = []
		fp32 = []
		for param in group["params"]:
			(fp32 if id(param) in keep_ids else rest).append(param)
		extras = {key: value for key, value in group.items() if key != "params"}
		if rest:
			new_groups.append({**extras, "params": rest})
		if fp32:
			fp32_groups += 1
			new_groups.append({**extras, "params": fp32})
	optimizer.param_groups[:] = new_groups
	return fp32_groups


def parameter_dtype_summary(model) -> dict[str, Any]:
	counts: Counter[str] = Counter()
	fp32_names: list[str] = []
	for name, param in model.named_parameters():
		dtype_name = str(param.dtype).replace("torch.", "")
		counts[dtype_name] += 1
		if param.dtype == torch.float32:
			fp32_names.append(name)
	suffixes = Counter(".".join(name.split(".")[-3:]) for name in fp32_names)
	return {
	 "counts": dict(counts),
	 "fp32_count": len(fp32_names),
	 "fp32_suffixes": dict(suffixes),
	}


def safetensors_dtype_summary(path: Path) -> dict[str, Any]:
	from safetensors.torch import safe_open

	counts: Counter[str] = Counter()
	fp32_names: list[str] = []
	files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
	for file_path in files:
		with safe_open(str(file_path), framework="pt") as handle:
			for name in handle.keys():
				dtype_name = handle.get_slice(name).get_dtype()
				counts[dtype_name] += 1
				if dtype_name == "F32":
					fp32_names.append(name)
	suffixes = Counter(".".join(name.split(".")[-3:]) for name in fp32_names)
	return {
	 "counts": dict(counts),
	 "fp32_count": len(fp32_names),
	 "fp32_suffixes": dict(suffixes),
	}


def log_parameter_dtypes(model, prefix: str) -> None:
	_log_dtype_summary(parameter_dtype_summary(model), prefix)


def log_saved_parameter_dtypes(path: Path, prefix: str) -> None:
	summary = safetensors_dtype_summary(path)
	_log_dtype_summary(summary, prefix)


def write_parameter_dtype_manifest(model, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
	 json.dumps(parameter_dtype_summary(model), indent=2) + "\n",
	 encoding="utf-8",
	)


def _log_dtype_summary(summary: dict[str, Any], prefix: str) -> None:
	print(f"{prefix} parameter dtypes: {summary['counts']}", flush=True)
	if summary["fp32_count"]:
		print(f"{prefix} FP32 parameter suffixes: {summary['fp32_suffixes']}", flush=True)
	else:
		print(f"{prefix} FP32 parameter count: 0", flush=True)


def _patch_transformers_keep_in_fp32() -> None:
	from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

	_patch_keep_in_fp32(Qwen3_5PreTrainedModel)
	try:
		from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
		 Qwen3_5MoePreTrainedModel,
		)
	except ImportError:
		return
	_patch_keep_in_fp32(Qwen3_5MoePreTrainedModel)


def _patch_keep_in_fp32(model_class) -> None:
	existing = list(getattr(model_class, "_keep_in_fp32_modules_strict", None) or [])
	for pattern in QWEN35_KEEP_IN_FP32_PARAM_PATTERNS:
		if pattern not in existing:
			existing.append(pattern)
	model_class._keep_in_fp32_modules_strict = existing


def _patch_deepspeed_keep_fp32() -> None:
	global _DEEPSPEED_FP32_PATCHED
	if _DEEPSPEED_FP32_PATCHED:
		return
	try:
		from deepspeed.runtime.engine import DeepSpeedEngine
	except ImportError:
		return

	original_cast = DeepSpeedEngine._cast_module_mixed_precision
	original_configure_optimizer = DeepSpeedEngine._configure_optimizer

	def _cast_module_mixed_precision(self, param_dtype, buffer_dtype, is_zero_init_model):
		keep_ids = qwen35_fp32_param_ids(self.module)
		if not keep_ids:
			return original_cast(self, param_dtype, buffer_dtype, is_zero_init_model)
		if param_dtype is not None and not is_zero_init_model:
			skipped = cast_floating_params_except_qwen35_fp32(self.module, param_dtype)
			print(
			 f"DeepSpeed mixed-precision cast skipped {skipped} Qwen3.5 FP32 parameters.",
			 flush=True,
			)
		if buffer_dtype is not None:
			for buffer in self.module.buffers(recurse=True):
				if buffer.is_floating_point() and buffer.dtype != buffer_dtype:
					buffer.data = buffer.data.to(buffer_dtype)

	def _configure_optimizer(self, client_optimizer, model_parameters):
		if client_optimizer is not None and hasattr(client_optimizer, "param_groups"):
			fp32_groups = split_optimizer_param_groups_for_fp32(client_optimizer, self.module)
			if fp32_groups:
				print(
				 f"Split {fp32_groups} optimizer param groups so Qwen3.5 FP32 "
				 "tensors are not flattened into BF16 ZeRO buffers.",
				 flush=True,
				)
		return original_configure_optimizer(self, client_optimizer, model_parameters)

	DeepSpeedEngine._cast_module_mixed_precision = _cast_module_mixed_precision
	DeepSpeedEngine._configure_optimizer = _configure_optimizer
	_DEEPSPEED_FP32_PATCHED = True
