import os
import logging
import gc
import inspect
from typing import Any, List, Dict, Optional, Union
from contextlib import redirect_stdout, redirect_stderr

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoConfig, AutoTokenizer
from transformers import logging as hf_logging
from behavioral_coupling.utils.gpt_oss import is_gpt_oss_config, resolve_moe_backend


def import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for behavioral_coupling.utils.llm.VLLM generation. "
            "Install vllm in the sampling environment, e.g. on Linux with `pip install vllm==0.12.0`."
        ) from exc
    return LLM, SamplingParams


class VLLM:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        dtype: Optional[str] = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        trust_remote_code: bool = False,
        max_model_len: Optional[int] = None,
        revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        language_model_only: Optional[bool] = None,
        gdn_prefill_backend: Optional[str] = None,
        quiet: bool = True,
        reasoning_effort: str = "low",
        moe_backend: Optional[str] = None,
    ):
        LLM, _ = import_vllm()
        self.model_name = model_name
        self.quiet = quiet
        self.reasoning_effort = reasoning_effort

        if self.quiet:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
            hf_logging.set_verbosity_error()
            logging.getLogger("vllm").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("urllib3").setLevel(logging.ERROR)

        tok_kwargs = {}
        if hf_token:
            tok_kwargs["token"] = hf_token
        if revision:
            tok_kwargs["revision"] = revision

        model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **tok_kwargs,
        )
        self.model_type = model_config.model_type
        self.is_gpt_oss = is_gpt_oss_config(model_config)
        is_full_qwen35 = (
            model_config is not None
            and model_config.model_type in {"qwen3_5", "qwen3_5_moe"}
        )
        if language_model_only is None:
            language_model_only = is_full_qwen35
        if gdn_prefill_backend is None and is_full_qwen35:
            gdn_prefill_backend = "triton"
        if is_full_qwen35:
            dtype = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=trust_remote_code,
            **tok_kwargs,
        )

        engine_kwargs = dict(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enforce_eager=True,
            language_model_only=language_model_only,
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        if gdn_prefill_backend is not None:
            engine_kwargs["gdn_prefill_backend"] = gdn_prefill_backend
        if revision:
            engine_kwargs["revision"] = revision
        if hf_token:
            engine_kwargs["hf_token"] = hf_token
        if self.is_gpt_oss:
            resolved_moe_backend = resolve_moe_backend(model_config, moe_backend)
            if resolved_moe_backend:
                from vllm.engine.arg_utils import EngineArgs

                if "moe_backend" not in inspect.signature(EngineArgs.__init__).parameters:
                    raise RuntimeError("This vLLM version cannot select the GPT-oss MoE backend; use the pinned vLLM version.")
                engine_kwargs["moe_backend"] = resolved_moe_backend
            print(
                f"GPT-oss: native Harmony chat, reasoning_effort={reasoning_effort}; "
                f"moe_backend={resolved_moe_backend or 'auto'}. enable_thinking does not disable GPT-oss reasoning.",
                flush=True,
            )

        self.llm = LLMQuietWrapper(LLM(**engine_kwargs), quiet=self.quiet)

        self.eos_token_id = self._safe_eos_id()
        self.eot_token_id = self._id_or_none("<|eot_id|>")

    def shutdown(self):
        raw_llm = getattr(self.llm, "_llm", None)
        if raw_llm is None:
            return

        engine = getattr(raw_llm, "llm_engine", None)
        try:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                engine_core = getattr(engine, "engine_core", None)
                engine_core_shutdown = getattr(engine_core, "shutdown", None)
                if callable(engine_core_shutdown):
                    engine_core_shutdown()

                model_executor = getattr(engine, "model_executor", None)
                model_executor_shutdown = getattr(model_executor, "shutdown", None)
                if callable(model_executor_shutdown):
                    model_executor_shutdown()
        finally:
            self.llm = None
            gc.collect()

    def generate(
        self,
        messages: Union[List[Dict[str, str]], str],
        chat: bool = True,
        do_sample: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_p: float = 1.0,
        top_k: int = 0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        n: int = 1,
        seed: Optional[int] = None,
        enable_thinking: Optional[bool] = True,
    ) -> Union[str, List[str]]:
        if chat and not self.is_gpt_oss:
            prompt = self._to_chat_prompt(messages, enable_thinking=enable_thinking)
        elif not chat:
            if not isinstance(messages, str):
                raise TypeError("messages must be a raw prompt string when chat=False")
            prompt = messages

        _, SamplingParams = import_vllm()
        sampling = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            presence_penalty=presence_penalty,
            max_tokens=max_new_tokens,
            n=n,
            stop=stop or None,
            stop_token_ids=[tid for tid in [self.eos_token_id, self.eot_token_id] if tid is not None],
            seed=seed,
        )

        if chat and self.is_gpt_oss:
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                raise TypeError("messages must be a list of dicts with keys: role, content")
            outs = self.llm.chat(
                messages,
                sampling_params=sampling,
                chat_template_kwargs={"reasoning_effort": self.reasoning_effort},
            )
        else:
            outs = self.llm.generate([prompt], sampling_params=sampling)
        texts = [o.text for o in outs[0].outputs]
        return texts[0] if n == 1 else texts

    def _to_chat_prompt(self, messages: List[Dict[str, str]], enable_thinking: Optional[bool] = True) -> str:
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise TypeError("messages must be a list of dicts with keys: role, content")

        chat_template_kwargs = {}
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    def _id_or_none(self, token: str) -> Optional[int]:
        try:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            return None if tid is None or tid == self.tokenizer.unk_token_id else tid
        except Exception:
            return None

    def _safe_eos_id(self) -> Optional[int]:
        try:
            return self.tokenizer.eos_token_id
        except Exception:
            return None


class LLMQuietWrapper:
    def __init__(self, llm: Any, quiet: bool = True):
        self._llm = llm
        self._quiet = quiet

    def generate(self, *args, **kwargs):
        if not self._quiet:
            return self._llm.generate(*args, **kwargs)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            return self._llm.generate(*args, **kwargs)

    def chat(self, *args, **kwargs):
        if not self._quiet:
            return self._llm.chat(*args, **kwargs)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            return self._llm.chat(*args, **kwargs)


