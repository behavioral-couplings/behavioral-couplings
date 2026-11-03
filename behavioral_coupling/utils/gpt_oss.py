from __future__ import annotations

import inspect
import os


def config_value(config, name: str, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def is_gpt_oss_config(config) -> bool:
    return config_value(config, "model_type") == "gpt_oss"


def mxfp4_training_kwargs(config) -> dict:
    quantization = config_value(config, "quantization_config")
    if is_gpt_oss_config(config) and config_value(quantization, "quant_method") == "mxfp4":
        from transformers import Mxfp4Config

        return {"quantization_config": Mxfp4Config(dequantize=True)}
    return {}


def loaded_with_mxfp4_dequantize(model) -> bool:
    return any(
        type(operation).__name__ == "Mxfp4Dequantize"
        for conversion in getattr(model, "_weight_conversions", None) or []
        for operation in getattr(conversion, "operations", None) or []
    )


def save_pretrained_in_memory_format(model, output_dir, state_dict=None) -> None:
    kwargs = {"safe_serialization": True}
    if state_dict is not None:
        kwargs["state_dict"] = state_dict
    if loaded_with_mxfp4_dequantize(model):
        if "save_original_format" not in inspect.signature(model.save_pretrained).parameters:
            raise RuntimeError("Saving dequantized GPT-oss requires save_original_format support (Transformers 5.9).")
        kwargs["save_original_format"] = False
    model.save_pretrained(str(output_dir), **kwargs)


def resolve_moe_backend(config, moe_backend: str | None = None) -> str | None:
    chosen = moe_backend or os.environ.get("VLLM_MOE_BACKEND") or os.environ.get("MOE_BACKEND")
    if chosen:
        chosen = chosen.strip().lower().replace("-", "_")
        return chosen if chosen and chosen != "auto" else None
    if is_gpt_oss_config(config) and not config_value(config, "quantization_config"):
        return "triton"
    return None


def split_harmony_response(raw_response: str) -> dict:
    text = raw_response.strip()
    for token in ("<|start|>", "<|end|>", "<|meta_sep|>", "<|im_sep|>", "<|fim_suffix|>", "<|ghissue|>"):
        text = text.replace(token, "")
    if text.startswith(("assistantanalysis", "assistantfinal")):
        text = text[len("assistant"):]
    think = ""
    if "assistantfinal" in text:
        think, text = text.rsplit("assistantfinal", 1)
        if think.startswith("analysis"):
            think = think[len("analysis"):]
        think = think.strip()
    elif text.startswith("analysis"):
        think, text = text[len("analysis"):].strip(), ""
    elif text.startswith("final"):
        text = text[len("final"):]
    return {"raw_response": raw_response, "think": think, "response": text.strip()}
